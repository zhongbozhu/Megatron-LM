# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CPU comparison of complete real-token fields in native MIMO snapshots.

Canonical sample/token identity comes from captured physical CP indices. No
model execution or all-gather is needed. This measures differences, without
assigning a numerical acceptance threshold.
"""

import argparse
import gc
import json
import math
from pathlib import Path

import torch


def _load(path):
    return torch.load(path, map_location='cpu', weights_only=True, mmap=True)


def _field(data, path):
    value = data
    for key in path.split('/'):
        value = value[int(key)] if isinstance(value, (list, tuple)) else value[key]
    if not isinstance(value, torch.Tensor) or value.ndim < 2:
        raise ValueError(f'Expected a token-aligned tensor at {path}')
    return value.reshape(-1, value.shape[-1])


def index_snapshots(directory, fields, *, step, phase, world_size, rounds, samples, tokens):
    """Require complete files and exactly one owner of every physical/real row."""
    directory = Path(directory) / f'step{step:08d}'
    paths = {
        directory / f'rank{rank:05d}' / f'{phase}-round{round_id:04d}.pt'
        for rank in range(world_size)
        for round_id in range(rounds)
    }
    actual = set(directory.glob(f'rank*/{phase}-round*.pt'))
    if paths != actual:
        raise ValueError(
            f'Incomplete snapshot coverage: missing={paths - actual}, extra={actual - paths}'
        )
    canonical, packs, schemas = {}, {}, {}
    padding_tokens = dummy_tokens = 0
    for path in sorted(paths):
        data = _load(path)
        rank = int(path.parent.name[4:])
        round_id = int(path.stem.rsplit('round', 1)[1])
        if (data['step'], data['phase'], data['round']) != (step, phase, round_id):
            raise ValueError(f'Snapshot identity mismatch: {path}')
        group = tuple(data['cp_ranks'])
        if (
            rank not in group
            or len(group) != len(set(group))
            or not set(group) <= set(range(world_size))
        ):
            raise ValueError(f'Invalid CP group: {path}')
        ids, lengths = data['sample_ids'], data['sample_lengths']
        padded, logical = data['padded_boundaries'], data['logical_boundaries']
        if len(ids) != len(lengths) or len(padded) not in (len(ids) + 1, len(ids) + 2):
            raise ValueError(f'Source/boundary lengths disagree: {path}')
        if padded[0] != 0 or any(a >= b for a, b in zip(padded, padded[1:])):
            raise ValueError(f'Invalid physical boundaries: {path}')
        expected_logical = [0]
        for length in lengths:
            expected_logical.append(expected_logical[-1] + length)
        dummy = padded[-1] - padded[len(ids)]
        if dummy:
            expected_logical.append(expected_logical[-1] + dummy)
        if logical != expected_logical:
            raise ValueError(f'Logical source/dummy boundaries disagree: {path}')
        signature = (tuple(ids), tuple(lengths), tuple(padded), tuple(logical))
        pack = packs.setdefault(
            (round_id, group),
            dict(
                signature=signature, ranks=set(), owners=torch.zeros(padded[-1], dtype=torch.int16)
            ),
        )
        if pack['signature'] != signature:
            raise ValueError(f'CP peers disagree on their pack: {path}')
        pack['ranks'].add(rank)
        index = data['physical_indices']
        local_tokens = int(data['local_tokens'])
        if (
            index.dtype != torch.int64
            or index.ndim != 1
            or len(index) != local_tokens
            or index.numel() != torch.unique(index).numel()
            or (index < 0).any()
            or (index >= padded[-1]).any()
        ):
            raise ValueError(f'Invalid local physical ownership: {path}')
        pack['owners'].index_add_(0, index, torch.ones_like(index, dtype=torch.int16))
        for name in fields:
            value = _field(data, name)
            schema = (value.shape[1], value.dtype)
            if value.shape[0] != local_tokens or schemas.setdefault(name, schema) != schema:
                raise ValueError(f'Invalid tensor schema at {path}/{name}')
        real_local = 0
        for slot, (sid, length) in enumerate(zip(ids, lengths)):
            begin, end = padded[slot : slot + 2]
            if not 0 < length <= end - begin:
                raise ValueError(f'Invalid real sample length: {path}')
            info = canonical.setdefault(
                sid, dict(length=length, pieces=[], owners=torch.zeros(length, dtype=torch.int16))
            )
            if info['length'] != length:
                raise ValueError(f'Sample {sid} changes length')
            rows = ((index >= begin) & (index < begin + length)).nonzero().flatten()
            positions = index.index_select(0, rows) - begin
            info['owners'].index_add_(0, positions, torch.ones_like(positions, dtype=torch.int16))
            if rows.numel():
                info['pieces'].append((path, rows, positions))
            real_local += rows.numel()
        padding_tokens += local_tokens - real_local
        dummy_tokens += int((index >= padded[len(ids)]).sum())
    for (_, group), pack in packs.items():
        if pack['ranks'] != set(group) or not torch.all(pack['owners'] == 1):
            raise ValueError('Missing or multiply-owned physical CP rows')
    if (
        set(canonical) != set(range(samples))
        or sum(item['length'] for item in canonical.values()) != tokens
    ):
        raise ValueError('Canonical source sample/token coverage differs')
    for sid, item in canonical.items():
        if not torch.all(item.pop('owners') == 1):
            raise ValueError(f'Sample {sid} has missing or multiply-owned real rows')
    return dict(
        samples=canonical,
        schemas=schemas,
        coverage=dict(
            files=len(paths),
            samples=samples,
            real_tokens=tokens,
            padding_tokens=padding_tokens,
            trailing_dummy_tokens=dummy_tokens,
        ),
    )


def _sample(layout, sid, field):
    info = layout['samples'][sid]
    width, dtype = layout['schemas'][field]
    result = torch.empty((info['length'], width), dtype=dtype)
    for path, rows, positions in info['pieces']:
        data = _load(path)
        value = _field(data, field)
        for start in range(0, len(rows), 1024):
            chunk = slice(start, start + 1024)
            result.index_copy_(0, positions[chunk], value.index_select(0, rows[chunk]))
    return result


def compare_field(reference, candidate, field):
    if reference['schemas'][field] != candidate['schemas'][field]:
        raise ValueError(f'Field schemas disagree: {field}')
    result = dict(
        elements=0,
        changed_elements=0,
        changed_rows=0,
        all_finite=True,
        nonfinite_reference_elements=0,
        nonfinite_candidate_elements=0,
        max_abs=0.0,
        changed_samples={},
    )
    ref_square = cur_square = delta_square = dot = 0.0
    for sid, info in reference['samples'].items():
        if candidate['samples'][sid]['length'] != info['length']:
            raise ValueError(f'Sample {sid} length differs')
        ref, cur = _sample(reference, sid, field), _sample(candidate, sid, field)
        changed_positions = []
        for start in range(0, len(ref), 256):
            a, b = ref[start : start + 256].double(), cur[start : start + 256].double()
            finite_a, finite_b = torch.isfinite(a), torch.isfinite(b)
            result['nonfinite_reference_elements'] += int((~finite_a).sum())
            result['nonfinite_candidate_elements'] += int((~finite_b).sum())
            result['all_finite'] &= bool(finite_a.all() and finite_b.all())
            result['elements'] += a.numel()
            result['changed_elements'] += int((a != b).sum())
            changed_positions.extend(((a != b).any(dim=1).nonzero().flatten() + start).tolist())
            # Nonfinite inputs invalidate numerical metrics for the whole field;
            # still count coverage/mismatches, without emitting NaN JSON values.
            a, b = a[finite_a & finite_b], b[finite_a & finite_b]
            delta = b - a
            if delta.numel():
                result['max_abs'] = max(result['max_abs'], float(delta.abs().max()))
            ref_square += float(a.square().sum())
            cur_square += float(b.square().sum())
            delta_square += float(delta.square().sum())
            dot += float((a * b).sum())
        if changed_positions:
            result['changed_samples'][str(sid)] = dict(
                rows=len(changed_positions),
                first=changed_positions[0],
                last=changed_positions[-1],
                first_positions=changed_positions[:32],
            )
        result['changed_rows'] += len(changed_positions)
        del ref, cur
        gc.collect()
    result.update(
        exact=result['all_finite'] and result['changed_elements'] == 0,
        absolute_l2=math.sqrt(delta_square),
        reference_l2=math.sqrt(ref_square),
        candidate_l2=math.sqrt(cur_square),
        relative_l2=math.sqrt(delta_square / ref_square) if ref_square else None,
        cosine=dot / math.sqrt(ref_square * cur_square) if ref_square and cur_square else None,
        scale=dot / ref_square if ref_square else None,
    )
    metrics = (
        'max_abs',
        'absolute_l2',
        'reference_l2',
        'candidate_l2',
        'relative_l2',
        'cosine',
        'scale',
    )
    result['numerical_metrics_valid'] = result['all_finite'] and all(
        result[name] is None or math.isfinite(result[name]) for name in metrics
    )
    if not result['numerical_metrics_valid']:
        for name in metrics:
            result[name] = None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fields', nargs='+', required=True)
    parser.add_argument('--step', type=int, default=0)
    parser.add_argument('--phase', choices=('training', 'statistics'), default='training')
    parser.add_argument('--reference-phase', choices=('training', 'statistics'))
    parser.add_argument('--world-size', type=int, required=True)
    parser.add_argument('--reference-rounds', type=int, required=True)
    parser.add_argument('--candidate-rounds', type=int, required=True)
    parser.add_argument('--samples', type=int, required=True)
    parser.add_argument('--tokens', type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    kwargs = dict(
        step=args.step, world_size=args.world_size, samples=args.samples, tokens=args.tokens
    )
    reference = index_snapshots(
        args.reference,
        args.fields,
        phase=args.reference_phase or args.phase,
        rounds=args.reference_rounds,
        **kwargs,
    )
    candidate = index_snapshots(
        args.candidate, args.fields, phase=args.phase, rounds=args.candidate_rounds, **kwargs
    )
    report = dict(
        reference=str(args.reference),
        candidate=str(args.candidate),
        step=args.step,
        phase=args.phase,
        reference_phase=args.reference_phase or args.phase,
        reference_coverage=reference['coverage'],
        candidate_coverage=candidate['coverage'],
        complete=False,
        numerical_threshold=None,
        fields={},
    )
    for field in args.fields:
        print(f'Comparing full real-token field: {field}', flush=True)
        report['fields'][field] = compare_field(reference, candidate, field)
        print(
            json.dumps(
                {k: v for k, v in report['fields'][field].items() if k != 'changed_samples'}
            ),
            flush=True,
        )
    report['complete'] = True
    report['cuda_initialized'] = torch.cuda.is_initialized()
    if report['cuda_initialized']:
        raise RuntimeError('Offline tensor comparison unexpectedly initialized CUDA')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
