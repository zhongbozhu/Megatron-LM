# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CPU assembly of partial canonical expert-ID tables for diagnostic recording.

Inputs already contain sample-relative positions from the real partition adapter.
This module does not implement CP slicing or router math. Ranks exchange compact
coverage metadata, write CPU ID tables to shared storage, and let one writer per
sample assemble the existing full-table format. No feature or ID tensor is sent
through a distributed collective. Existing CP1 references need no format change.
"""

import hashlib
import json
from pathlib import Path

import torch
import torch.distributed as dist


def _phase(operation, group, *, exchange=False):
    value, error = None, None
    try:
        value = operation()
    except Exception as exception:
        error = f'{type(exception).__name__}: {exception}'
    errors = [error]
    if group is not None:
        errors = [None] * dist.get_world_size(group)
        dist.all_gather_object(errors, error, group=group)
    if any(errors):
        raise RuntimeError(
            'Fixed-routing shard assembly failed: '
            + '; '.join(f'rank {rank}: {message}' for rank, message in enumerate(errors) if message)
        )
    if not exchange:
        return value
    values = [value]
    if group is not None:
        values = [None] * dist.get_world_size(group)
        dist.all_gather_object(values, value, group=group)
    return values


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _segments(positions, length):
    """Compress existing canonical positions; retain their local table offsets."""
    if (
        positions.device.type != 'cpu'
        or positions.ndim != 1
        or positions.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError('Canonical positions must be a one-dimensional CPU integer tensor')
    if not positions.numel():
        return []
    if positions.min() < 0 or positions.max() >= length:
        raise ValueError('Canonical positions fall outside their source sample')
    if positions.unique().numel() != positions.numel():
        raise ValueError('A rank contains duplicate canonical positions')
    breaks = (positions[1:] != positions[:-1] + 1).nonzero().flatten() + 1
    offsets = [0, *breaks.tolist(), positions.numel()]
    return [
        (int(positions[start]), int(positions[end - 1]) + 1, start)
        for start, end in zip(offsets, offsets[1:])
    ]


def _describe(records, positions, lengths, rank):
    if (
        not records
        or not lengths
        or any(not isinstance(sid, int) or length <= 0 for sid, length in lengths.items())
    ):
        raise ValueError('Routing records require named routers and positive source lengths')
    if set(positions) - set(lengths):
        raise ValueError('Unknown source sample in canonical positions')
    segments = {sid: _segments(value, lengths[sid]) for sid, value in positions.items()}
    shapes = {}
    for name, samples in records.items():
        if not isinstance(name, str) or not name or set(samples) != set(positions):
            raise ValueError('Every router must cover the same rank-local sample set')
        descriptor = None
        for sid, table in samples.items():
            if (
                table.device.type != 'cpu'
                or table.ndim != 2
                or table.dtype not in (torch.int16, torch.int32, torch.int64)
            ):
                raise ValueError('Expert-ID parts must be two-dimensional CPU integer tensors')
            if table.shape[0] != positions[sid].numel() or table.shape[1] <= 0:
                raise ValueError('Expert-ID rows must match the canonical positions')
            if table.numel() and (
                table.min() < 0
                or (table.shape[1] > 1 and (table.sort(-1).values.diff(dim=-1) == 0).any())
            ):
                raise ValueError('Expert IDs must be nonnegative and unique within each token')
            current = (str(table.dtype), table.shape[1])
            if descriptor is not None and descriptor != current:
                raise ValueError('A router must use the same ID dtype and top-k for every sample')
            descriptor = current
        shapes[name] = descriptor
    return dict(rank=rank, lengths=lengths, segments=segments, shapes=shapes)


def _coverage(metadata):
    """Check a partition of every real sample before writing any reference file."""
    baseline = metadata[0]
    lengths, names = baseline['lengths'], set(baseline['shapes'])
    specs = {}
    parts = {sid: [] for sid in lengths}
    ranks = set()
    for item in metadata:
        if item['rank'] in ranks or item['lengths'] != lengths or set(item['shapes']) != names:
            raise ValueError('Ranks disagree on source lengths, routers, or rank identity')
        ranks.add(item['rank'])
        for name, descriptor in item['shapes'].items():
            if descriptor is not None:
                if name in specs and specs[name] != descriptor:
                    raise ValueError('Ranks disagree on expert-ID dtype or top-k')
                specs[name] = descriptor
        for sid, segments in item['segments'].items():
            parts[sid].extend((start, end, item['rank'], offset) for start, end, offset in segments)
    if set(specs) != names:
        raise ValueError('No recorded expert IDs for one or more routers')
    owners = {}
    for sid, length in lengths.items():
        cursor = 0
        parts[sid].sort()
        for start, end, _, _ in parts[sid]:
            if start != cursor:
                raise ValueError(
                    f'Canonical coverage gap or overlap for sample {sid} at position {cursor}'
                )
            cursor = end
        if cursor != length:
            raise ValueError(f'Incomplete canonical coverage for sample {sid}: {cursor}/{length}')
        owners[sid] = min(part[2] for part in parts[sid])
    return owners, parts, specs


def assemble_canonical_records(records, positions, sample_lengths, group, directory):
    """Return schema-1-compatible full sample tables on designated CPU writers.

    ``records[router_name][sample_id]`` is an integer ``[owned_rows, topk]`` CPU
    table. ``positions[sample_id]`` maps those rows to original sample positions,
    shared by every router. Empty local ownership is supported: keep every router
    key and use empty dictionaries. The caller already validates expert-ID upper
    bounds against router configuration; this helper checks storage and coverage.

    ``directory`` must not exist and is kept as diagnostic evidence. The selected
    writer is the lowest global rank owning any real rows of each sample. Returned
    dictionaries retain every router key, including on ranks that own no full
    samples, so the existing full-table sealing and replay code can be reused.
    """
    directory = Path(directory)
    rank = dist.get_rank() if group is not None else 0
    group_rank = dist.get_rank(group) if group is not None else 0
    metadata = _phase(
        lambda: _describe(records, positions, sample_lengths, rank), group, exchange=True
    )
    owners, parts, specs = _phase(lambda: _coverage(metadata), group)
    _phase(lambda: directory.mkdir(parents=True) if group_rank == 0 else None, group)

    def write_parts():
        temporary = directory / f'rank{rank:05d}.tmp'
        filename = directory / f'rank{rank:05d}.pt'
        torch.save(records, temporary)
        temporary.replace(filename)
        payload = dict(**metadata[group_rank], sha256=_file_sha256(filename))
        (directory / f'rank{rank:05d}.json').write_text(
            json.dumps(payload, indent=2, sort_keys=True) + '\n'
        )
        return dict(rank=rank, sha256=payload['sha256'])

    receipts = _phase(write_parts, group, exchange=True)
    hashes = {item['rank']: item['sha256'] for item in receipts}

    def assemble():
        result = {name: {} for name in records}
        selected = [sid for sid, writer in owners.items() if writer == rank]
        shards = {}
        for source in sorted({source for sid in selected for _, _, source, _ in parts[sid]}):
            filename = directory / f'rank{source:05d}.pt'
            if _file_sha256(filename) != hashes[source]:
                raise ValueError(f'ID part file changed after publication: rank {source}')
            shards[source] = torch.load(filename, map_location='cpu', weights_only=True, mmap=True)
        for sid in selected:
            for name, (dtype, topk) in specs.items():
                output = torch.empty(
                    (sample_lengths[sid], topk), dtype=getattr(torch, dtype.removeprefix('torch.'))
                )
                for start, end, source, offset in parts[sid]:
                    output[start:end].copy_(
                        shards[source][name][sid][offset : offset + end - start]
                    )
                result[name][sid] = output
        return result

    # This phase gathers only errors, never the returned full expert-ID tables.
    return _phase(assemble, group)
