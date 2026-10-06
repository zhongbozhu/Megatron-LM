# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CPU references for captured Qwen MoE activations; never used by training.

The reference keeps the stored BF16 input/weight values, then computes in FP64.
It preserves Megatron's probability placement (SwiGLU -> probability -> FC2).
It does not emulate TE/HybridEP rounding or certify their kernels. In particular,
an isolated module VJP excludes the step-global auxiliary objective and is not a
native DDP FP32-main-grad comparison.
"""

import argparse
import bisect
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass
class ExpertWeights:
    """FC1 is [gate; up], FC2 is down; biases are disabled in this recipe."""

    fc1: torch.Tensor
    fc2: torch.Tensor


def frozen_bf16_values(value, dtype=torch.float64):
    """Round once as native BF16 parameter destinations do, then upcast."""
    return value.detach().cpu().to(torch.bfloat16).to(dtype)


def router_reference(inputs, weight, topk, fixed_ids=None):
    """Qwen post-top-k softmax; ties use increasing expert ID in the reference.

    Fixed IDs retain the differentiable logits/probability path. Supplying
    fixed probabilities to ``moe_reference`` instead deliberately removes it.
    """
    logits = F.linear(inputs, weight)
    if not 0 < topk < logits.shape[-1]:
        raise ValueError('topk must leave at least one expert outside the selection')
    ordered = torch.argsort(logits.detach(), dim=-1, descending=True, stable=True)
    ids = ordered[:, :topk] if fixed_ids is None else fixed_ids
    if ids.shape != (len(inputs), topk) or ids.dtype != torch.long:
        raise ValueError('Expert IDs must be int64 [tokens, topk]')
    if bool((ids < 0).any() or (ids >= weight.shape[0]).any()):
        raise ValueError('Expert ID out of range')
    if bool((ids.sort(dim=-1).values.diff(dim=-1) == 0).any()):
        raise ValueError('Repeated expert ID for a token')
    probs = logits.gather(1, ids).softmax(dim=-1)
    sorted_logits = logits.detach().gather(1, ordered)
    margin = sorted_logits[:, topk - 1] - sorted_logits[:, topk]
    return logits, ids, probs, margin


def expert_reference(inputs, weights, probabilities=None):
    """Probability multiplication precedes FC2, including in the FP64 oracle."""
    gate, up = F.linear(inputs, weights.fc1).chunk(2, dim=-1)
    activated = F.silu(gate) * up
    if probabilities is not None:
        activated = activated * probabilities.reshape(-1, 1)
    return F.linear(activated, weights.fc2)


def shared_reference(inputs, weights, gate_weight):
    return expert_reference(inputs, weights) * F.linear(inputs, gate_weight).sigmoid()


def canonical_dispatch(inputs, routing_map):
    """CPU gather ordered by (expert ID, canonical token row); no communication."""
    if inputs.device.type != 'cpu' or routing_map.device.type != 'cpu':
        raise ValueError('Canonical reference dispatch is CPU-only')
    if routing_map.dtype != torch.bool or routing_map.ndim != 2:
        raise ValueError('routing_map must be boolean [tokens, experts]')
    if routing_map.shape[0] != inputs.shape[0]:
        raise ValueError('Token counts differ')
    experts, tokens = routing_map.T.nonzero(as_tuple=True)
    return inputs.index_select(0, tokens), tokens, experts


def canonical_combine(contributions, tokens, num_tokens):
    """Ordered CPU SUM of already weighted FC2 outputs; its adjoint is gather."""
    if contributions.device.type != 'cpu' or tokens.device.type != 'cpu':
        raise ValueError('Canonical reference combine is CPU-only')
    if tokens.dtype != torch.long or tokens.ndim != 1 or len(tokens) != len(contributions):
        raise ValueError('One int64 token index is required per contribution')
    if num_tokens < 0 or bool((tokens < 0).any() or (tokens >= num_tokens).any()):
        raise ValueError('Invalid destination token index')
    # Explicit ordered additions avoid relying on an atomic scatter's order.
    zero = contributions.sum(dim=0) * 0
    rows = [zero for _ in range(num_tokens)]
    for index, token in enumerate(tokens.tolist()):
        rows[token] = rows[token] + contributions[index]
    return torch.stack(rows) if rows else contributions.new_empty((0, *contributions.shape[1:]))


def moe_reference(
    inputs,
    router_weight,
    shared_weights,
    shared_gate,
    expert_loader,
    topk,
    fixed_ids=None,
    fixed_probs=None,
):
    """Differentiable MoE formula with lazy loading of selected expert weights.

    ``expert_loader(id)`` returns ExpertWeights. Input and parameter gradients
    can be obtained with normal autograd, without importing Megatron or TE.
    """
    logits, ids, probs, margin = router_reference(inputs, router_weight, topk, fixed_ids)
    if fixed_probs is not None:
        if fixed_ids is None or fixed_probs.shape != probs.shape:
            raise ValueError('Fixed probabilities require matching fixed expert IDs')
        if not torch.isfinite(fixed_probs).all() or bool((fixed_probs < 0).any()):
            raise ValueError('Invalid fixed probabilities')
        probs = fixed_probs.detach()
    contributions, destinations = [], []
    for expert_id in sorted(ids.unique().tolist()):
        token, slot = (ids == expert_id).nonzero(as_tuple=True)
        contributions.append(
            expert_reference(
                inputs.index_select(0, token), expert_loader(expert_id), probs[token, slot]
            )
        )
        destinations.append(token)
    routed = canonical_combine(torch.cat(contributions), torch.cat(destinations), len(inputs))
    shared = shared_reference(inputs, shared_weights, shared_gate)
    return dict(
        logits=logits,
        ids=ids,
        probabilities=probs,
        margin=margin,
        routed=routed,
        shared=shared,
        output=routed + shared,
    )


class TorchDistSliceReader:
    """Read selected CPU tensor slices from torch_dist, without distributed init.

    Only intersecting tensor chunks are deserialized. Expert tensors may have a
    leading global expert axis and FC1 gate/up chunks; both are handled by the
    checkpoint metadata, not by guessed byte offsets or weight-name rewrites.
    """

    def __init__(self, checkpoint):
        from torch.distributed.checkpoint import FileSystemReader

        self.path = Path(checkpoint)
        self.metadata = FileSystemReader(self.path).read_metadata()
        self.storage = {}
        for index, value in self.metadata.storage_data.items():
            if index.offset is not None:
                identity = (index.fqn, tuple(index.offset))
                if identity in self.storage:
                    raise ValueError(f'Duplicate stored checkpoint chunk: {identity}')
                self.storage[identity] = value

    def shape(self, key):
        metadata = self.metadata.state_dict_metadata.get(key)
        if metadata is None or not hasattr(metadata, 'size'):
            raise KeyError(f'Missing tensor checkpoint key: {key}')
        return tuple(metadata.size)

    def tensor(self, key, starts=None, sizes=None):
        shape = self.shape(key)
        starts = (0,) * len(shape) if starts is None else tuple(starts)
        sizes = shape if sizes is None else tuple(sizes)
        if len(starts) != len(shape) or len(sizes) != len(shape):
            raise ValueError('Slice rank differs from checkpoint tensor')
        if any(s < 0 or n <= 0 or s + n > dim for s, n, dim in zip(starts, sizes, shape)):
            raise ValueError('Slice outside checkpoint tensor')
        metadata = self.metadata.state_dict_metadata[key]
        output = torch.empty(sizes, dtype=metadata.properties.dtype)
        covered = torch.zeros(sizes, dtype=torch.bool)
        for chunk in metadata.chunks:
            lo = tuple(max(s, c) for s, c in zip(starts, chunk.offsets))
            hi = tuple(
                min(s + n, c + m) for s, n, c, m in zip(starts, sizes, chunk.offsets, chunk.sizes)
            )
            if any(a >= b for a, b in zip(lo, hi)):
                continue
            storage = self.storage[(key, tuple(chunk.offsets))]
            with (self.path / storage.relative_path).open('rb') as stream:
                stream.seek(storage.offset)
                payload = stream.read(storage.length)
            value = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
            if value.numel() != chunk.sizes.numel():
                raise ValueError('Stored tensor size differs from chunk metadata')
            value = value.reshape(tuple(chunk.sizes))
            destination = tuple(slice(a - s, b - s) for a, b, s in zip(lo, hi, starts))
            source = tuple(slice(a - c, b - c) for a, b, c in zip(lo, hi, chunk.offsets))
            if covered[destination].any():
                raise ValueError('Overlapping checkpoint chunks')
            output[destination] = value[source]
            covered[destination] = True
        if not covered.all():
            raise ValueError(f'Incomplete checkpoint slice for {key}')
        return output


class QwenMoECheckpoint:
    """Pretrained Qwen router/shared/selected-expert weights with native rounding."""

    def __init__(self, checkpoint, layer=0, dtype=torch.float64):
        self.reader = TorchDistSliceReader(checkpoint)
        self.prefix = f'language_model.decoder.layers.{layer}.mlp.'
        self.dtype = dtype

    def _weight(self, suffix, **slice_kwargs):
        return frozen_bf16_values(
            self.reader.tensor(self.prefix + suffix, **slice_kwargs), self.dtype
        )

    def router(self):
        return self._weight('router.weight')

    def shared(self):
        return ExpertWeights(
            self._weight('shared_experts.linear_fc1.weight'),
            self._weight('shared_experts.linear_fc2.weight'),
        )

    def shared_gate(self):
        return self._weight('shared_experts.gate_weight')

    def expert(self, expert_id):
        values = []
        for name in ('linear_fc1', 'linear_fc2'):
            suffix = f'experts.experts.{name}.weight'
            shape = self.reader.shape(self.prefix + suffix)
            if len(shape) != 3:
                raise ValueError(f'Expected [experts, out, in] for {suffix}: {shape}')
            values.append(
                self._weight(suffix, starts=(expert_id, 0, 0), sizes=(1, *shape[1:])).squeeze(0)
            )
        return ExpertWeights(*values)


def capture_rows(snapshot, token_keys, layer=0):
    """Select real token rows from schema-v1 MoE boundary snapshots.

    This accepts any CP ownership. It refuses mixed original/recompute values
    when returning the saved output cotangent, instead of silently combining
    different executions into one VJP experiment.
    """
    if snapshot['step'] != 0 or snapshot['phase'] != 'training':
        raise ValueError('Pretrained reference requires source-step-zero training capture')
    entry = snapshot.get('moe_layers', {}).get(str(layer), {})
    if entry.get('schema_version') != 1:
        raise ValueError('Missing schema-v1 MoE capture')
    keys = []
    indices = []
    requested = set(map(tuple, token_keys))
    for row, physical in enumerate(snapshot['physical_indices'].tolist()):
        slot = bisect.bisect_right(snapshot['padded_boundaries'], physical) - 1
        if not 0 <= slot < len(snapshot['sample_ids']):
            continue
        position = physical - snapshot['padded_boundaries'][slot]
        if not 0 <= position < snapshot['sample_lengths'][slot]:
            continue
        key = (snapshot['sample_ids'][slot], position)
        if key in requested:
            keys.append(key)
            indices.append(row)
    if not indices:
        return keys, {}
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate canonical token rows in capture')
    selected = torch.tensor(indices, dtype=torch.long)
    n = len(snapshot['physical_indices'])
    boundaries = entry['boundaries']
    output = {}
    for name in (
        'moe_input',
        'router_logits',
        'router_probs',
        'routing_map',
        'shared_output',
        'routed_output',
        'moe_output',
    ):
        records = boundaries.get(name, [])
        if not records or records[0].get('kind') != 'original':
            raise ValueError(f'Missing original MoE boundary: {name}')
        value = records[0]['value']
        if value.shape[0] != n:
            raise ValueError(f'Incorrect token dimension for {name}')
        output[name] = value.reshape(n, -1).index_select(0, selected)
    records = boundaries['moe_output']
    gradients = [record for record in records if 'gradient' in record]
    if len(gradients) != 1:
        raise ValueError('Expected exactly one observed MoE output cotangent')
    gradient_record = gradients[0]
    original = records[0]['value'].reshape(n, -1).index_select(0, selected)
    repeated = gradient_record['value'].reshape(n, -1).index_select(0, selected)
    if not torch.equal(original, repeated):
        raise ValueError('Original and cotangent-bearing recompute values differ')
    matched_inputs = [
        record
        for record in boundaries['moe_input']
        if record['invocation'] == gradient_record['invocation']
    ]
    if len(matched_inputs) != 1 or not torch.equal(
        output['moe_input'], matched_inputs[0]['value'].reshape(n, -1).index_select(0, selected)
    ):
        raise ValueError('Original and cotangent-bearing recompute inputs differ or are missing')
    output['output_gradient'] = gradient_record['gradient'].reshape(n, -1).index_select(0, selected)
    if output['moe_input'].dtype != torch.bfloat16:
        raise ValueError('Reference expects a native BF16 activation capture')
    if output['routing_map'].dtype != torch.bool:
        raise ValueError('Expected boolean routing map')
    if not all(torch.isfinite(value).all() for value in output.values()):
        raise ValueError('Nonfinite capture')
    return keys, output


def load_capture_selection(directory, token_keys, layer=0):
    """Read all source-zero training shards; reject missing or repeated tokens."""
    requested = list(map(tuple, token_keys))
    if not requested or len(set(requested)) != len(requested):
        raise ValueError('Require a nonempty list of unique [sample_id, position] keys')
    rows = {}
    for path in sorted(Path(directory).glob('rank*/training-round*.pt')):
        snapshot = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        keys, values = capture_rows(snapshot, requested, layer)
        for index, key in enumerate(keys):
            if key in rows:
                raise ValueError(f'Duplicate selected token across shards: {key}')
            rows[key] = {name: value[index] for name, value in values.items()}
    missing = set(requested) - rows.keys()
    if missing:
        raise ValueError(f'Missing {len(missing)} requested real token rows')
    return {
        name: torch.stack([rows[key][name] for key in requested]) for name in rows[requested[0]]
    }


def difference(actual, reference):
    actual, reference = actual.detach().double(), reference.detach().double()
    if actual.shape != reference.shape:
        raise ValueError('Comparison shapes differ')
    error = (actual - reference).norm().item()
    norm = reference.norm().item()
    return dict(
        exact=torch.equal(actual, reference),
        error_l2=error,
        reference_l2=norm,
        relative_l2=error / norm if norm else (None if error else 0.0),
        max_abs=(actual - reference).abs().max().item(),
        all_finite=bool(torch.isfinite(actual).all() and torch.isfinite(reference).all()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--capture-dir', type=Path, required=True, help='decoder/step00000000 directory'
    )
    parser.add_argument(
        '--token-keys', type=Path, required=True, help='JSON list of [sample_id, position]'
    )
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument(
        '--tensor-output', type=Path, help='Optional selected-row reference outputs/VJPs'
    )
    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.tensor_output is not None and args.tensor_output.exists():
        raise FileExistsError(args.tensor_output)
    if torch.cuda.is_initialized():
        raise RuntimeError('This reference must not initialize CUDA')
    if args.checkpoint.name != 'iter_0000000':
        raise ValueError('Only pretrained source-step-zero captures are supported')
    torch.set_num_threads(args.threads)
    keys = json.loads(args.token_keys.read_text())
    data = load_capture_selection(args.capture_dir, keys, args.layer)
    weights = QwenMoECheckpoint(args.checkpoint, args.layer)
    router, shared, gate = weights.router(), weights.shared(), weights.shared_gate()
    counts = data['routing_map'].sum(-1)
    if not torch.equal(counts, counts[:1].expand_as(counts)):
        raise ValueError('Captured rows have unequal top-k counts')
    topk = int(counts[0])
    ids = data['routing_map'].nonzero()[:, 1].reshape(len(keys), topk)
    fixed_probs = data['router_probs'].gather(1, ids).double()
    report = dict(
        scope='Selected-row CPU FP64 formula and isolated module input VJP; not native main_grad or auxiliary-loss parity',
        checkpoint=str(args.checkpoint),
        capture_dir=str(args.capture_dir),
        token_keys=keys,
        weight_values='Rounded to native BF16 before FP64 upcast',
        modes={},
        threshold=None,
    )
    report['provenance'] = dict(
        torch_version=torch.__version__,
        reference_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        checkpoint_metadata_sha256=hashlib.sha256(
            (args.checkpoint / '.metadata').read_bytes()
        ).hexdigest(),
        token_keys_sha256=hashlib.sha256(args.token_keys.read_bytes()).hexdigest(),
        capture_files=[
            dict(path=str(path), bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
            for path in sorted(args.capture_dir.glob('rank*/training-round*.pt'))
        ],
        capture_file_hash_note='Sizes/mtimes only; selected tensor values are separately hashed.',
        selected_tensors={
            name: dict(
                shape=list(value.shape),
                dtype=str(value.dtype),
                sha256=hashlib.sha256(
                    value.contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
            )
            for name, value in data.items()
        },
    )
    tensors = {}
    for mode in ('reference_selection', 'fixed_ids', 'fixed_ids_and_probabilities'):
        inputs = data['moe_input'].double().requires_grad_()
        result = moe_reference(
            inputs,
            router,
            shared,
            gate,
            weights.expert,
            topk,
            fixed_ids=None if mode == 'reference_selection' else ids,
            fixed_probs=fixed_probs if mode == 'fixed_ids_and_probabilities' else None,
        )
        (gradient,) = torch.autograd.grad(
            result['output'], inputs, data['output_gradient'].double()
        )
        item = {
            name: difference(data[native], result[name])
            for name, native in (
                ('logits', 'router_logits'),
                ('shared', 'shared_output'),
                ('routed', 'routed_output'),
                ('output', 'moe_output'),
            )
        }
        item['changed_expert_set_rows'] = int(
            (result['ids'].sort(-1).values != ids.sort(-1).values).any(-1).sum()
        )
        item['topk_margin_per_token'] = result['margin'].tolist()
        logit_error = (result['logits'].detach() - data['router_logits'].double()).abs().amax(-1)
        item['logit_max_abs_error_per_token'] = logit_error.tolist()
        item['margin_exceeds_twice_observed_logit_error'] = (
            result['margin'] > 2 * logit_error
        ).tolist()
        item['selected_probabilities'] = difference(
            data['router_probs'].gather(1, result['ids']), result['probabilities']
        )
        item['input_vjp_l2'] = gradient.norm().item()
        item['input_vjp_finite'] = bool(torch.isfinite(gradient).all())
        item['input_vjp_compared_to_native'] = False
        item['input_vjp_note'] = (
            'Native input cotangents may include global auxiliary VJP; frozen probabilities exclude router VJP as well.'
        )
        report['modes'][mode] = item
        if args.tensor_output is not None:
            tensors[mode] = {name: value.detach().cpu() for name, value in result.items()}
            tensors[mode]['input_vjp'] = gradient.detach().cpu()
        del result, gradient, inputs
    report['cuda_initialized'] = torch.cuda.is_initialized()
    if report['cuda_initialized']:
        raise RuntimeError('Unexpected CUDA context')
    if args.tensor_output is not None:
        with args.tensor_output.open('xb') as stream:
            torch.save(dict(token_keys=keys, modes=tensors), stream)
        report['tensor_output'] = str(args.tensor_output)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(args.output)


if __name__ == '__main__':
    main()
