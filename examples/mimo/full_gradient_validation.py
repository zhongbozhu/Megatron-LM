# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Opt-in, bounded-memory comparison of every native optimizer-owned gradient.

Call after ``finalize_model_grads`` and before optimizer clipping/update. This
reads the already reduced, token-normalized FP32 shards; it never reduces or
changes parameter gradients. References require identical parameter names and
distributed-optimizer ownership (including EP layout) across compared runs.
"""

import json
import math
from pathlib import Path

import torch
import torch.distributed as dist

from examples.mimo.gradient_diagnostics import _COMPONENTS, _component_names


def _owned_gradients(models):
    """Use the same bucket/range intersections as DistributedOptimizer."""
    entries = []
    for model_index, model in enumerate(models):
        config, ddp_config = model.config, model.ddp_config
        if (
            not ddp_config.use_distributed_optimizer
            or ddp_config.num_distributed_optimizer_instances != 1
            or config.tensor_model_parallel_size != 1
            or config.pipeline_model_parallel_size != 1
            or config.fp16
            or not config.calculate_per_token_loss
        ):
            raise ValueError(
                'Full gradient validation requires TP1/PP1, per-token BF16/FP32 '
                'native DDP with one distributed optimizer instance'
            )
        names = {parameter: name for name, parameter in model.named_parameters()}
        seen = set()
        for buffer in model.buffers + model.expert_parallel_buffers:
            size, rank = buffer.data_parallel_group.size(), buffer.data_parallel_group.rank()
            if buffer.num_optimizer_shards not in (None, size):
                raise ValueError('Gradient buffer ownership does not match its DP group')
            for parameter, (start, end, bucket_index) in buffer.param_index_map.items():
                if parameter in seen:
                    raise ValueError('Parameter appears in multiple native gradient buffers')
                seen.add(parameter)
                bucket = buffer.buckets[bucket_index]
                if bucket.grad_data.dtype != torch.float32 or bucket.grad_data.numel() % size:
                    raise ValueError('Expected evenly sharded FP32 native gradient buckets')
                shard_size = bucket.grad_data.numel() // size
                shard_start = bucket.offset + rank * shard_size
                left, right = max(start, shard_start), min(end, shard_start + shard_size)
                if right <= left:
                    continue
                name, components = _component_names(names[parameter])
                metadata = {
                    'model': model_index,
                    'name': name,
                    'shape': list(parameter.shape),
                    'parameter_offset': left - start,
                    'elements': right - left,
                    'components': list(components),
                    'dp_ranks': dist.get_process_group_ranks(buffer.data_parallel_group),
                }
                gradient = bucket.grad_data.view(-1)[left - bucket.offset : right - bucket.offset]
                entries.append((metadata, gradient))
        if seen != {parameter for parameter in names if parameter.requires_grad}:
            raise ValueError('Native gradient buffers do not cover all trainable parameters')
    entries.sort(key=lambda item: (item[0]['model'], item[0]['name'], item[0]['parameter_offset']))
    return entries


def _accumulate_statistics(statistics, indices, current, reference):
    # FP64 reductions with <=8MiB temporaries, independent of parameter size.
    for start in range(0, current.numel(), 1 << 20):
        value = current[start : start + (1 << 20)].double()
        original = reference[start : start + (1 << 20)].double()
        delta = value - original
        values = torch.tensor(
            [
                value.numel(),
                value.square().sum(),
                original.square().sum(),
                delta.square().sum(),
                (~torch.isfinite(value)).sum(),
                (~torch.isfinite(original)).sum(),
            ],
            dtype=torch.float64,
        )
        statistics[indices, :6] += values
        maxima = torch.tensor([delta.abs().max(), value.abs().max(), original.abs().max()])
        statistics[indices, 6:] = torch.maximum(statistics[indices, 6:], maxima)


@torch.no_grad()
def validate_encoder_boundary(
    features, gradient, media, reference_dir, *, step, num_tokens, write_reference=False, group=None
):
    """Compare producer rows and their token-normalized gradient before encoder backward.

    Only detached CPU copies and compact distributed statistics are produced.
    References include image order and ownership; empty producers participate.
    ``media`` contains immutable identity/row metadata, never image tensors.
    """
    group = dist.group.WORLD if group is None else group
    path = Path(reference_dir) / f'step{step:08d}-rank{dist.get_rank():05d}.pt'
    statistics = torch.zeros((2, 9), dtype=torch.float64)
    failure = None
    try:
        denominator = float(num_tokens)
        if denominator <= 0 or not math.isfinite(denominator):
            raise ValueError('Boundary normalization needs positive finite global token count')
        if (
            features.ndim != 2
            or features.shape != gradient.shape
            or gradient.dtype != torch.float32
        ):
            raise ValueError('Boundary requires matching features and FP32 gradient matrices')
        if sum(item['length'] for item in media) != features.shape[0]:
            raise ValueError('Boundary image metadata does not cover producer rows')
        metadata = dict(
            format='mimo-encoder-boundary-v1',
            step=step,
            rank=dist.get_rank(),
            producer_ranks=dist.get_process_group_ranks(group),
            media=media,
            shape=list(features.shape),
            dtype=str(features.dtype),
            num_tokens=denominator,
        )
        tensors = dict(
            features=features.detach().cpu().contiguous(),
            gradient=gradient.detach().cpu().contiguous() / denominator,
        )
        if write_reference:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise FileExistsError(f'Encoder boundary reference already exists: {path}')
            temporary = path.with_suffix('.pt.incomplete')
            torch.save(dict(metadata=metadata, **tensors), temporary)
            temporary.replace(path)
            reference = tensors
        else:
            reference = torch.load(path, map_location='cpu', weights_only=True)
            if reference['metadata'] != metadata:
                raise ValueError(f'Encoder boundary ownership/image manifest mismatch: {path}')
        for index, (name, current) in enumerate(tensors.items()):
            original = reference[name]
            if original.shape != current.shape or original.dtype != current.dtype:
                raise ValueError(f'Encoder boundary tensor schema mismatch: {name}')
            _accumulate_statistics(statistics, [index], current.view(-1), original.view(-1))
    except Exception as error:
        failure = f'{type(error).__name__}: {error}'
    failed = torch.tensor([failure is not None], device=features.device, dtype=torch.int32)
    dist.all_reduce(failed, group=group)
    if failed.item():
        raise RuntimeError(
            f'Encoder boundary validation failed on {failed.item()} ranks: '
            f'{failure or "see failing peer log"}'
        )
    sums = statistics[:, :6].to(features.device).contiguous()
    maxima = statistics[:, 6:].to(features.device).contiguous()
    dist.all_reduce(sums, group=group)
    dist.all_reduce(maxima, op=dist.ReduceOp.MAX, group=group)
    components = {}
    for name, row in zip(tensors, torch.cat((sums, maxima), dim=1).cpu().tolist()):
        count, square, ref_square, delta, bad, ref_bad, maximum, _, _ = row
        components[name] = dict(
            elements=int(count),
            l2=math.sqrt(square),
            reference_l2=math.sqrt(ref_square),
            relative_l2=(
                math.sqrt(delta / ref_square) if ref_square else (math.inf if delta else 0.0)
            ),
            max_abs_error=maximum,
            all_finite=bad == 0 and ref_bad == 0,
        )
    return dict(
        kind='full_producer_encoder_boundary',
        mode='write' if write_reference else 'compare',
        reference_dir=str(path.parent),
        components=components,
        all_finite=all(component['all_finite'] for component in components.values()),
    )


@torch.no_grad()
def validate_full_gradients(
    model, reference_dir, *, write_reference=False, group=None, chunk_elements=16 * 1024 * 1024
):
    """Save/compare every owned normalized gradient, without full all-gathers.

    Each rank writes one contiguous little-endian FP32 file plus a JSON manifest.
    Padding and non-owned buffer entries are excluded. I/O and comparison use
    bounded CPU chunks, with only compact statistics reduced across ``group``.
    The reference is an optional diagnostic artifact, not an optimizer checkpoint.
    ``global_l2`` is the full unclipped gradient norm, not a sampled estimate.
    Components overlap: decoder includes its MoE and MTP submodules.
    """
    if chunk_elements <= 0:
        raise ValueError('chunk_elements must be positive')
    models = model if isinstance(model, (list, tuple)) else [model]
    device = next(models[0].parameters()).device
    group = dist.group.WORLD if group is None else group
    rank = dist.get_rank()
    directory = Path(reference_dir)
    manifest_path = directory / f'rank{rank:05d}.json'
    data_path = directory / f'rank{rank:05d}.f32'
    temporary_data = data_path.with_suffix('.f32.incomplete')
    temporary_manifest = manifest_path.with_suffix('.json.incomplete')
    statistics = torch.zeros((1 + len(_COMPONENTS), 9), dtype=torch.float64)
    failure = None
    try:
        entries = _owned_gradients(models)
        manifest = {
            'format': 'mimo-native-owned-gradient-v1',
            'rank': rank,
            'world_size': dist.get_world_size(group),
            'dtype': 'float32',
            'entries': [metadata for metadata, _ in entries],
        }
        if write_reference:
            directory.mkdir(parents=True, exist_ok=True)
            if manifest_path.exists() or data_path.exists():
                raise FileExistsError(f'Gradient reference already exists: {manifest_path}')
            stream = temporary_data.open('wb')
        else:
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError(
                    f'Gradient reference ownership/parameter manifest mismatch: {manifest_path}'
                )
            expected_bytes = sum(entry['elements'] for entry in manifest['entries']) * 4
            if data_path.stat().st_size != expected_bytes:
                raise ValueError(f'Incomplete gradient reference data: {data_path}')
            stream = data_path.open('rb')
        with stream:
            for metadata, gradient in entries:
                indices = [0] + [_COMPONENTS.index(name) + 1 for name in metadata['components']]
                for part in gradient.split(chunk_elements):
                    current = part.detach().to(device='cpu', copy=True).contiguous()
                    if write_reference:
                        stream.write(memoryview(current.numpy()))
                        reference = current
                    else:
                        raw = bytearray(stream.read(current.numel() * 4))
                        if len(raw) != current.numel() * 4:
                            raise ValueError(f'Truncated gradient reference: {data_path}')
                        reference = torch.frombuffer(raw, dtype=torch.float32)
                    _accumulate_statistics(statistics, indices, current, reference)
        if write_reference:
            temporary_manifest.write_text(json.dumps(manifest, indent=2) + '\n')
            temporary_data.replace(data_path)
            temporary_manifest.replace(manifest_path)
    except Exception as error:
        # Peers may still be doing I/O. All ranks rendezvous at the same small
        # collective, so a local filesystem/schema failure cannot strand them.
        failure = f'{type(error).__name__}: {error}'
    failed = torch.tensor([failure is not None], device=device, dtype=torch.int32)
    dist.all_reduce(failed, group=group)
    if failed.item():
        raise RuntimeError(
            f'Full gradient validation failed on {failed.item()} ranks: '
            f'{failure or "see failing peer log"}'
        )
    reduced = statistics.to(device=device)
    sums = reduced[:, :6].contiguous()
    maxima = reduced[:, 6:].contiguous()
    dist.all_reduce(sums, group=group)
    dist.all_reduce(maxima, op=dist.ReduceOp.MAX, group=group)
    reduced = torch.cat((sums, maxima), dim=1).cpu()

    def report(row):
        (
            count,
            square,
            reference_square,
            delta_square,
            bad,
            reference_bad,
            error,
            maximum,
            ref_max,
        ) = row.tolist()
        denominator = math.sqrt(reference_square)
        relative = (
            math.sqrt(delta_square) / denominator
            if denominator
            else (0.0 if delta_square == 0 else math.inf)
        )
        return {
            'elements': int(count),
            'l2': math.sqrt(square),
            'reference_l2': denominator,
            'relative_l2': relative,
            'max_abs_error': error,
            'max_abs': maximum,
            'reference_max_abs': ref_max,
            'all_finite': bad == 0 and reference_bad == 0,
        }

    totals = report(reduced[0])
    return {
        'kind': 'full_owned_normalized_gradients',
        'mode': 'write' if write_reference else 'compare',
        'reference_dir': str(directory),
        'global_elements': totals['elements'],
        'all_finite': totals['all_finite'],
        'global_l2': totals['l2'],
        'totals': totals,
        'components': {name: report(reduced[index + 1]) for index, name in enumerate(_COMPONENTS)},
    }
