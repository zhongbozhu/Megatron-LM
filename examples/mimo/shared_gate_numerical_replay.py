# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Isolated replay of the native SharedExpertMLP scalar gate, never training code.

The production gate is F.linear(input, gate_weight).sigmoid(). Actual captured
full batch shapes and row locations are retained; inputs/weights are the same
stored BF16 values in every precision. FP32 uses TF32-disabled matmul. FP64 is
a CPU formula reference, not a reproduction of the native GPU kernel.

No shared FC1/FC2, distributed gradient accumulation, or auxiliary VJP is
covered. A gate difference alone does not prove the entire shared MLP correct.
"""

import argparse
import bisect
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F

from examples.mimo.moe_numerical_reference import QwenMoECheckpoint, capture_rows, difference


@contextmanager
def bf16_reduction_mode(mode):
    """Temporarily select the cuBLAS BF16 reduction policy, including on failure."""
    if mode not in ('default', 'on', 'off'):
        raise ValueError('BF16 reduction mode must be default, on, or off')
    backend = torch.backends.cuda.matmul
    original = backend.allow_bf16_reduced_precision_reduction
    state = dict(requested=mode, original=original)
    try:
        if mode != 'default':
            backend.allow_bf16_reduced_precision_reduction = mode == 'on'
        state['active'] = backend.allow_bf16_reduced_precision_reduction
        yield state
    finally:
        if mode != 'default':
            backend.allow_bf16_reduced_precision_reduction = original
        state['restored'] = backend.allow_bf16_reduced_precision_reduction


def tensor_sha256(value):
    """Hash the exact CPU values together with their dtype and shape."""
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(json.dumps([str(value.dtype), list(value.shape)]).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def selected_local_rows(snapshot, keys):
    """Return canonical keys and local physical rows, excluding THD padding."""
    requested = set(map(tuple, keys))
    found, rows = [], []
    for row, physical in enumerate(snapshot['physical_indices'].tolist()):
        slot = bisect.bisect_right(snapshot['padded_boundaries'], physical) - 1
        if 0 <= slot < len(snapshot['sample_ids']):
            position = physical - snapshot['padded_boundaries'][slot]
            key = (snapshot['sample_ids'][slot], position)
            if 0 <= position < snapshot['sample_lengths'][slot] and key in requested:
                found.append(key)
                rows.append(row)
    if len(set(found)) != len(found):
        raise ValueError('Duplicate selected token in one capture')
    return found, torch.tensor(rows, dtype=torch.long)


def gate_forward(inputs, weight):
    """Exact production gate operations, with both intermediate values exposed."""
    logits = F.linear(inputs, weight)
    return {'logits': logits, 'sigmoid': F.sigmoid(logits)}


def embedded_geometry(inputs, total, placement='front'):
    """Keep selected input values fixed, varying only geometry/zero-filled rows."""
    if inputs.ndim != 2 or total < len(inputs) or not len(inputs):
        raise ValueError('Need nonempty [selected, hidden] inputs and enough rows')
    if placement == 'front':
        rows = torch.arange(len(inputs), dtype=torch.long)
    elif placement == 'spread':
        rows = torch.linspace(0, total - 1, len(inputs), dtype=torch.float64).long()
    else:
        raise ValueError('Unknown placement')
    result = inputs.new_zeros(total, 1, inputs.shape[-1])
    result[rows, 0] = inputs
    return result, rows


def metric(actual, reference, keys):
    result = difference(actual, reference)
    changed = (actual != reference).reshape(len(keys), -1).any(dim=1)
    result['changed_rows'] = int(changed.sum())
    result['changed_keys'] = [list(k) for k, flag in zip(keys, changed.tolist()) if flag]
    return result


def replay_captures(directory, keys, weight, dtype, device):
    """Each original full input goes through the production gate unchanged."""
    collected, provenance = {}, []
    for path in sorted(Path(directory).glob('rank*/training-round*.pt')):
        snapshot = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
        found, rows = selected_local_rows(snapshot, keys)
        if not found:
            continue
        checked, selected = capture_rows(snapshot, keys)
        if found != checked:
            raise ValueError('Canonical ownership disagrees with strict capture adapter')
        original = snapshot['moe_layers']['0']['boundaries']['moe_input'][0]['value']
        inputs = original.to(device=device, dtype=dtype)
        with torch.no_grad():
            values = gate_forward(inputs, weight.to(device=device, dtype=dtype))
        values = {
            name: value.cpu().reshape(len(original), -1)[rows] for name, value in values.items()
        }
        for index, key in enumerate(found):
            if key in collected:
                raise ValueError(f'Duplicate captured key: {key}')
            collected[key] = {
                **{name: value[index] for name, value in values.items()},
                'input': selected['moe_input'][index],
                'native_shared': selected['shared_output'][index],
            }
        provenance.append(
            {
                'path': str(path.resolve()),
                'shape': list(original.shape),
                'rows': rows.tolist(),
                'keys': found,
            }
        )
        del inputs, values, snapshot
    missing = set(map(tuple, keys)) - collected.keys()
    if missing:
        raise ValueError(f'Missing selected keys: {sorted(missing)}')
    ordered = [collected[tuple(key)] for key in keys]
    return {name: torch.stack([row[name] for row in ordered]) for name in ordered[0]}, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-cp1', type=Path, required=True)
    parser.add_argument('--capture-cp8', type=Path, required=True)
    parser.add_argument('--token-keys', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tensor-output', type=Path, required=True)
    parser.add_argument('--geometries', type=int, nargs='+', default=[64, 8192, 45056])
    parser.add_argument(
        '--bf16-reduced-precision-reduction',
        choices=('default', 'on', 'off'),
        default='default',
        help='Diagnostic cuBLAS BF16 reduction policy; default preserves the existing setting',
    )
    args = parser.parse_args()
    with bf16_reduction_mode(args.bf16_reduced_precision_reduction) as reduction:
        reports = run_replay(args, reduction)
    args.output.write_text(json.dumps(reports, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'completed': True}))


def run_replay(args, reduction):
    """Run the unchanged gate/geometry controls under the caller's backend policy."""
    if args.output.exists() or args.tensor_output.exists():
        raise FileExistsError('Refusing to overwrite diagnostics')
    keys = json.loads(args.token_keys.read_text())
    if not keys or len(set(map(tuple, keys))) != len(keys):
        raise ValueError('Need unique nonempty token keys')
    if args.checkpoint.name != 'iter_0000000':
        raise ValueError('Source-zero captures require initial checkpoint')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    weight = QwenMoECheckpoint(args.checkpoint).shared_gate()
    tensors, reports = {}, {}
    common = None
    for name, dtype in [('bf16', torch.bfloat16), ('fp32', torch.float32)]:
        captures = {}
        for layout, directory in [('cp1', args.capture_cp1), ('cp8', args.capture_cp8)]:
            values, sources = replay_captures(directory, keys, weight, dtype, 'cuda')
            captures[layout] = values
            tensors[f'{name}_{layout}'] = values
            reports[f'{name}_{layout}_sources'] = sources
        if not torch.equal(captures['cp1']['input'], captures['cp8']['input']):
            raise ValueError('CP1/CP8 selected input values differ')
        if common is None:
            common = captures['cp1']['input']
            with torch.no_grad():
                tensors['fp64_reference'] = gate_forward(common.double(), weight)
        for layout in ('cp1', 'cp8'):
            reports[f'{name}_{layout}_vs_fp64'] = {
                field: metric(captures[layout][field], tensors['fp64_reference'][field], keys)
                for field in ('logits', 'sigmoid')
            }
        reports[f'{name}_cross_layout'] = {
            field: metric(captures['cp8'][field], captures['cp1'][field], keys)
            for field in ('logits', 'sigmoid', 'native_shared')
        }
        for total in args.geometries:
            for placement in ('front', 'spread'):
                inputs, rows = embedded_geometry(common, total, placement)
                with torch.no_grad():
                    value = gate_forward(inputs.cuda().to(dtype), weight.cuda().to(dtype))
                result = {
                    field: tensor.cpu().reshape(total, -1)[rows] for field, tensor in value.items()
                }
                label = f'{name}_zeros{total}_{placement}'
                tensors[label] = result
                reports[label] = {
                    field: metric(result[field], captures['cp1'][field], keys) for field in result
                }
    reports.update(
        scope='Native scalar gate forward only; no FC1/FC2, backward, or full-model acceptance',
        token_keys=keys,
        selected_inputs_equal=True,
        bf16_values_preserved_before_upcast=True,
        bf16_reduced_precision_reduction=reduction,
        selected_inputs_sha256=tensor_sha256(common),
        gate_weight_sha256=tensor_sha256(weight),
        fp64_reference_sha256={
            field: tensor_sha256(value) for field, value in tensors['fp64_reference'].items()
        },
        geometries=args.geometries,
        tf32_enabled=torch.backends.cuda.matmul.allow_tf32,
        torch_version=torch.__version__,
        device=torch.cuda.get_device_name(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        checkpoint=str(args.checkpoint.resolve()),
        token_keys_sha256=hashlib.sha256(args.token_keys.read_bytes()).hexdigest(),
        threshold=None,
        overall_acceptance_claimed=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensors, args.tensor_output)
    return reports


if __name__ == '__main__':
    main()
