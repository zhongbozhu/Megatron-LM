# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Opt-in replay of intact packs or explicit diagnostic DCP schedules.

The core scheduler still chooses the reference packs. This module only records
their order and places the same packs on a different number of CP groups. The
core THD builder and padding helper validate that replay preserves *physical*
boundaries as well as sample membership; no tokens or dummy samples are added
here. Incomplete rounds fail instead of repeating or dropping source samples.
Dynamic schedule replay is a separate diagnostic, guarded by both the source
identity and the original scheduler plan. It never changes scheduler policy.
"""

import hashlib
import json
from pathlib import Path

import torch


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    ).hexdigest()


def _content_identity(value):
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return dict(
            shape=list(tensor.shape),
            dtype=str(tensor.dtype),
            sha256=hashlib.sha256(
                tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
        )
    if isinstance(value, dict):
        return {str(key): _content_identity(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_content_identity(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f'Unsupported source identity value: {type(value).__name__}')


def _source_identity(samples, media):
    descriptors = []
    for sid, sample in sorted(samples.items()):
        original, padded = int(sample['original_seq_len']), int(sample['padded_seq_len'])
        if original <= 0 or padded < original:
            raise ValueError(f'Invalid sequence lengths for sample {sid}: {original}, {padded}')
        for key in ('tokens', 'labels', 'loss_mask'):
            if key in sample and sample[key].numel() != padded:
                raise ValueError(f'Sample {sid} {key} does not match padded_seq_len')
        if 'position_ids' in sample and sample['position_ids'].shape[-1] != padded:
            raise ValueError(f'Sample {sid} position_ids do not match padded_seq_len')
        descriptors.append(
            dict(
                sample_id=sid,
                original_seq_len=original,
                padded_seq_len=padded,
                content_sha256=_digest(_content_identity(sample)),
            )
        )
    return dict(samples=descriptors, media_sha256=_digest(_content_identity(media)))


def _check_packs(packs, samples):
    if not packs or any(not pack for pack in packs):
        raise ValueError('Fixed packing requires nonempty packs')
    scheduled = [sid for pack in packs for sid in pack]
    if sorted(scheduled) != sorted(samples):
        raise ValueError('Fixed packing must contain every source sample exactly once')


def _pack_layout(sample_ids, samples, config, cp_size):
    # Import lazily: the disabled path need not import any packing machinery.
    from megatron.core.datasets.data_schedule_utils import (
        build_packed_microbatches,
        pad_packed_batch_before_cp_slice,
    )

    lengths = {
        sid: {
            key: torch.tensor(int(samples[sid][key]), dtype=torch.int32)
            for key in ('original_seq_len', 'padded_seq_len')
        }
        for sid in sample_ids
    }
    batch = build_packed_microbatches(lengths, [[sample_ids]], 0, torch.device('cpu'))[0]
    pad_packed_batch_before_cp_slice(batch, config, cp_size, 1)
    physical = batch['cu_seqlens_padded']
    alignment = 2 * cp_size if cp_size > 1 else 1
    if torch.any(physical.diff() % alignment):
        raise ValueError(f'Fixed pack is not aligned for zigzag CP{cp_size}')
    total = int(physical[-1])
    limit = config.max_seqlen_per_dp_cp_rank
    if limit is None or total // cp_size > limit:
        raise ValueError(f'Fixed pack has {total // cp_size} local tokens, exceeding limit {limit}')
    return dict(
        sample_ids=list(sample_ids),
        original_lengths=[int(samples[sid]['original_seq_len']) for sid in sample_ids],
        padded_lengths=[int(samples[sid]['padded_seq_len']) for sid in sample_ids],
        logical_boundaries=batch['cu_seqlens'].tolist(),
        physical_boundaries=physical.tolist(),
        max_seqlen=int(batch['max_seqlen']),
    )


def apply_dynamic_schedule_replay(
    samples,
    media,
    assignments,
    *,
    step,
    domain_ranks,
    cp_group_sizes,
    config,
    replay_path,
    audit_path=None,
    write_audit=False,
):
    """Replay an explicit DCP plan for a controlled scheduling intervention.

    Every rank validates the source, baseline scheduler decision, group geometry
    and real core THD padding. The normal adapter still constructs all tensors,
    runtime groups and visual routes. Only a designated caller writes the actual
    post-replay plan; the usual source audit describes the baseline scheduler.
    """
    filename = f'step{step:08d}.json'
    with (Path(replay_path) / filename).open() as stream:
        record = json.load(stream)
    payload = record['payload']
    if record['sha256'] != _digest(payload):
        raise ValueError('Diagnostic DCP schedule checksum mismatch')
    if payload['schema_version'] != 2 or payload['step'] != step:
        raise ValueError('Diagnostic DCP schedule schema or source step mismatch')
    world = len(domain_ranks)
    if (
        not world
        or len(set(domain_ranks)) != world
        or payload['domain_ranks'] != list(domain_ranks)
    ):
        raise ValueError('Diagnostic DCP schedule rank domain mismatch')
    if payload['source'] != _source_identity(samples, media):
        raise ValueError('Diagnostic DCP schedule source fingerprint mismatch')
    baseline_digest = _digest(assignments)
    if payload['baseline_assignments_sha256'] != baseline_digest:
        raise ValueError('Diagnostic DCP schedule baseline assignments mismatch')

    replayed = payload['assignments']
    if not isinstance(replayed, list) or not replayed:
        raise ValueError('Diagnostic DCP schedule requires nonempty rounds')
    packs, layouts = [], []
    for round_id, assignment in enumerate(replayed):
        if not isinstance(assignment, list) or len(assignment) != world:
            raise ValueError('Diagnostic DCP round must cover every domain rank')
        groups = {}
        for rank, ids in enumerate(assignment):
            if (
                not isinstance(ids, list)
                or not ids
                or any(type(sid) is not int or sid not in samples for sid in ids)
            ):
                raise ValueError('Diagnostic DCP packs require nonempty known sample IDs')
            groups.setdefault(tuple(ids), []).append(rank)
        for ids, ranks in groups.items():
            cp_size = len(ranks)
            if (
                cp_size not in cp_group_sizes
                or cp_size & (cp_size - 1)
                or ranks[0] % cp_size
                or ranks != list(range(ranks[0], ranks[0] + cp_size))
            ):
                raise ValueError('Diagnostic DCP pack needs an allowed aligned contiguous CP group')
            packs.append(list(ids))
            layouts.append(
                dict(
                    round=round_id,
                    cp_ranks=[domain_ranks[rank] for rank in ranks],
                    **_pack_layout(ids, samples, config, cp_size),
                )
            )
    _check_packs(packs, samples)
    if audit_path is not None and write_audit:
        directory = Path(audit_path)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / filename).open('x') as stream:
            json.dump(
                dict(
                    step=step,
                    replay_manifest_sha256=record['sha256'],
                    source_identity_sha256=_digest(payload['source']),
                    baseline_assignments_sha256=baseline_digest,
                    assignments_sha256=_digest(replayed),
                    assignments=replayed,
                    pack_layouts=layouts,
                ),
                stream,
                indent=2,
                allow_nan=False,
            )
            stream.write('\n')
    return replayed


def apply_fixed_packing(
    samples,
    media,
    assignments,
    *,
    step,
    domain_ranks,
    cp_size,
    config,
    record_path=None,
    replay_path=None,
    write_record=False,
):
    """Record CP1 packs or replay them on contiguous fixed-CP groups.

    ``record_path`` / ``replay_path`` are directories containing one
    ``stepXXXXXXXX.json`` per source step. Every rank validates its local source
    copy; only the caller with ``write_record=True`` writes. No collectives are
    performed. A replay should start after the reference process has finished.

    TP=PP=1 and fixed contiguous CP groups are required by the native diagnostic
    caller. Replaying under DCP is intentionally unsupported: the actual DCP
    experiment must still use the scheduler's own decisions. The disabled path
    returns the original assignment object without validation or source hashing.
    """
    if record_path is None and replay_path is None:
        return assignments
    if record_path is not None and replay_path is not None:
        raise ValueError('Fixed packing record and replay are mutually exclusive')
    if step < 0 or cp_size < 1:
        raise ValueError('Fixed packing requires a nonnegative step and positive CP size')
    world = len(domain_ranks)
    if not world or len(set(domain_ranks)) != world or world % cp_size:
        raise ValueError('Fixed CP size must divide a domain of unique ranks')
    source = _source_identity(samples, media)
    filename = f'step{step:08d}.json'
    if record_path is not None:
        if cp_size != 1:
            raise ValueError('Fixed packing reference recording requires CP1')
        if not assignments or any(len(assignment) != world for assignment in assignments):
            raise ValueError('Reference assignments must cover every domain rank in each round')
        packs = [list(pack) for assignment in assignments for pack in assignment]
        _check_packs(packs, samples)
        payload = dict(
            schema_version=1,
            step=step,
            source=source,
            reference_world_size=world,
            packs=[_pack_layout(pack, samples, config, cp_size) for pack in packs],
        )
        if write_record:
            directory = Path(record_path)
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / filename).open('x') as stream:
                json.dump(
                    dict(payload=payload, sha256=_digest(payload)),
                    stream,
                    indent=2,
                    allow_nan=False,
                )
                stream.write('\n')
        return assignments

    with (Path(replay_path) / filename).open() as stream:
        record = json.load(stream)
    payload = record['payload']
    if record['sha256'] != _digest(payload):
        raise ValueError('Fixed packing reference checksum mismatch')
    if payload['schema_version'] != 1 or payload['step'] != step:
        raise ValueError('Fixed packing reference schema or source step mismatch')
    if payload['source'] != source:
        raise ValueError('Fixed packing source fingerprint mismatch')
    layouts = payload['packs']
    packs = [layout['sample_ids'] for layout in layouts]
    _check_packs(packs, samples)
    groups_per_round = world // cp_size
    if len(packs) % groups_per_round:
        raise ValueError(
            f'{len(packs)} intact packs cannot fill {groups_per_round} CP groups per round; '
            'choose a controlled batch with complete rounds'
        )
    for index, (pack, expected) in enumerate(zip(packs, layouts)):
        actual = _pack_layout(pack, samples, config, cp_size)
        if actual != expected:
            raise ValueError(
                f'Fixed pack {index} physical/logical layout changes at CP{cp_size}; '
                'use compatible source padding and numeric packed alignment in both runs'
            )
    return [
        [list(pack) for pack in packs[start : start + groups_per_round] for _ in range(cp_size)]
        for start in range(0, len(packs), groups_per_round)
    ]
