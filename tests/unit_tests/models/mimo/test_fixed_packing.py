# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CPU checks for intact-pack replay through the real core THD utilities."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from examples.mimo.fixed_packing import (
    _digest,
    _source_identity,
    apply_dynamic_schedule_replay,
    apply_fixed_packing,
)


def _samples(count=6):
    return {
        sid: dict(
            tokens=torch.arange(16) + sid * 16,
            labels=torch.arange(16) + sid * 16 + 1,
            loss_mask=torch.tensor([1.0] * 9 + [0.0] * 7),
            position_ids=torch.arange(16).expand(3, -1),
            original_seq_len=9,
            padded_seq_len=16,
            media_ids=[sid],
        )
        for sid in range(count)
    }


def _config(limit=128, alignment=2):
    return SimpleNamespace(
        max_seqlen_per_dp_cp_rank=limit,
        pad_packed_seq_alignment=alignment,
        sequence_parallel=False,
        fp8=None,
    )


def _record(directory, *, samples=None, assignments=None, world=4, config=None):
    samples = _samples() if samples is None else samples
    assignments = [[[0, 1], [2], [3, 4], [5]]] if assignments is None else assignments
    media = [dict(image_id=0, pixel_values=torch.tensor([[1.5, 2.5]], dtype=torch.bfloat16))]
    actual = apply_fixed_packing(
        samples,
        media,
        assignments,
        step=7,
        domain_ranks=tuple(range(world)),
        cp_size=1,
        config=config or _config(),
        record_path=directory,
        write_record=True,
    )
    assert actual is assignments
    return samples, media, assignments


def _replay(directory, samples, media, *, cp=2, world=4, config=None, step=7):
    return apply_fixed_packing(
        samples,
        media,
        object(),  # Replay does not silently use the candidate scheduler's packs.
        step=step,
        domain_ranks=tuple(range(world)),
        cp_size=cp,
        config=config or _config(),
        replay_path=directory,
    )


def test_disabled_path_returns_original_without_accessing_sources():
    assignments = object()
    assert (
        apply_fixed_packing(
            None, None, assignments, step=None, domain_ranks=None, cp_size=None, config=None
        )
        is assignments
    )


def test_replay_changes_cp_placement_without_changing_packs(tmp_path):
    samples, media, reference = _record(tmp_path)
    result = _replay(tmp_path, samples, media)
    assert result == [[[0, 1], [0, 1], [2], [2]], [[3, 4], [3, 4], [5], [5]]]
    assert _replay(tmp_path, samples, media, cp=1) == reference
    assert _replay(tmp_path, samples, media, cp=4) == [
        [[0, 1]] * 4,
        [[2]] * 4,
        [[3, 4]] * 4,
        [[5]] * 4,
    ]
    record = json.loads((tmp_path / 'step00000007.json').read_text())
    assert record['payload']['packs'][0]['logical_boundaries'] == [0, 9, 18]
    assert record['payload']['packs'][0]['physical_boundaries'] == [0, 16, 32]
    # Replicated assignments are independent lists, not aliased mutable state.
    result[0][0].append(99)
    assert result[0][1] == [0, 1]


@pytest.mark.parametrize('field', ['tokens', 'labels', 'loss_mask', 'position_ids', 'media'])
def test_replay_rejects_changed_source_content(tmp_path, field):
    samples, media, _ = _record(tmp_path)
    samples, media = copy.deepcopy(samples), copy.deepcopy(media)
    if field == 'media':
        media[0]['pixel_values'][0, 0] += 1
    else:
        samples[0][field] = samples[0][field].clone()
        samples[0][field].reshape(-1)[0] += 1
    with pytest.raises(ValueError, match='source fingerprint'):
        _replay(tmp_path, samples, media)


def test_replay_rejects_padding_changes_between_cp_sizes(tmp_path):
    samples, media, _ = _record(tmp_path, config=_config(alignment=32))
    with pytest.raises(ValueError, match='physical/logical layout changes'):
        _replay(tmp_path, samples, media, config=_config(alignment=32))


def test_replay_checks_real_padded_capacity(tmp_path):
    samples, media, _ = _record(tmp_path)
    with pytest.raises(ValueError, match='exceeds|exceeding'):
        _replay(tmp_path, samples, media, config=_config(limit=8))


def test_disabled_padding_cannot_bypass_cp_alignment(tmp_path):
    samples = _samples(1)
    samples[0] = {
        key: value[..., :10].clone() if isinstance(value, torch.Tensor) else value
        for key, value in samples[0].items()
    }
    samples[0]['padded_seq_len'] = 10
    samples, media, _ = _record(
        tmp_path, samples=samples, assignments=[[[0]]], world=1, config=_config(alignment=None)
    )
    with pytest.raises(ValueError, match='not aligned'):
        _replay(tmp_path, samples, media, cp=2, world=2, config=_config(alignment=None))


def test_replay_rejects_incomplete_round_without_duplication(tmp_path):
    samples, media, _ = _record(
        tmp_path, samples=_samples(3), assignments=[[[0]], [[1]], [[2]]], world=1
    )
    with pytest.raises(ValueError, match='cannot fill'):
        _replay(tmp_path, samples, media)


@pytest.mark.parametrize(
    'assignments',
    [[[[0, 1], [2], [3, 4], [4]]], [[[0, 1], [2], [3, 4], []]], [[[0, 1], [2], [3, 4]]]],
)
def test_record_rejects_invalid_source_coverage(tmp_path, assignments):
    with pytest.raises(ValueError, match='exactly once|nonempty|every domain rank'):
        _record(tmp_path, assignments=assignments)
    assert not list(tmp_path.glob('*.json'))


def test_only_designated_writer_publishes_and_existing_records_are_preserved(tmp_path):
    samples, media, assignments = _record(tmp_path / 'first')
    apply_fixed_packing(
        samples,
        media,
        assignments,
        step=7,
        domain_ranks=tuple(range(4)),
        cp_size=1,
        config=_config(),
        record_path=tmp_path / 'nonwriter',
    )
    assert not (tmp_path / 'nonwriter').exists()
    with pytest.raises(FileExistsError):
        _record(tmp_path / 'first')


def test_corrupt_or_stale_reference_cannot_pass(tmp_path):
    samples, media, _ = _record(tmp_path)
    path = tmp_path / 'step00000007.json'
    record = json.loads(path.read_text())
    record['payload']['packs'][0]['sample_ids'].reverse()
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='checksum'):
        _replay(tmp_path, samples, media)
    record['payload']['step'] = 8
    record['sha256'] = _digest(record['payload'])
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='source step'):
        _replay(tmp_path, samples, media)


def test_reference_recording_requires_cp1_and_exclusive_mode(tmp_path):
    kwargs = dict(step=7, domain_ranks=tuple(range(4)), config=_config(), record_path=tmp_path)
    with pytest.raises(ValueError, match='requires CP1'):
        apply_fixed_packing(_samples(), [], [], cp_size=2, **kwargs)
    with pytest.raises(ValueError, match='mutually exclusive'):
        apply_fixed_packing(None, None, None, cp_size=1, replay_path=tmp_path, **kwargs)


def test_diagnostic_dcp_tail_merge_validates_source_groups_coverage_and_capacity(tmp_path):
    samples, media = _samples(5), []
    baseline = [[[0], [0], [1], [1], [2], [2], [3], [3]], [[4]] * 8]
    merged = [[[0, 1, 4]] * 4 + [[2], [2], [3], [3]]]
    payload = dict(
        schema_version=2,
        step=7,
        domain_ranks=list(range(8)),
        source=_source_identity(samples, media),
        baseline_assignments_sha256=_digest(baseline),
        assignments=merged,
    )

    def write(value):
        (tmp_path / 'step00000007.json').write_text(
            json.dumps(dict(payload=value, sha256=_digest(value)))
        )

    kwargs = dict(
        step=7,
        domain_ranks=tuple(range(8)),
        cp_group_sizes=(1, 2, 4, 8),
        config=_config(),
        replay_path=tmp_path,
    )
    write(payload)
    result = apply_dynamic_schedule_replay(
        samples, media, baseline, **kwargs, audit_path=tmp_path / 'actual', write_audit=True
    )
    assert result == merged
    audit = json.loads((tmp_path / 'actual/step00000007.json').read_text())
    assert audit['assignments'] == merged
    assert [len(layout['cp_ranks']) for layout in audit['pack_layouts']] == [4, 2, 2]
    assert audit['pack_layouts'][0]['physical_boundaries'] == [0, 16, 32, 48]
    result[0][0].append(99)
    assert result[0][1] == [0, 1, 4]

    noncontiguous = copy.deepcopy(payload)
    noncontiguous['assignments'][0][0], noncontiguous['assignments'][0][4] = (
        noncontiguous['assignments'][0][4],
        noncontiguous['assignments'][0][0],
    )
    missing = copy.deepcopy(payload)
    missing['assignments'][0][:4] = [[0, 1]] * 4
    stale = copy.deepcopy(payload)
    stale['baseline_assignments_sha256'] = 'different scheduler plan'
    for invalid, message in [
        (noncontiguous, 'aligned contiguous'),
        (missing, 'exactly once'),
        (stale, 'baseline assignments'),
    ]:
        write(invalid)
        with pytest.raises(ValueError, match=message):
            apply_dynamic_schedule_replay(samples, media, baseline, **kwargs)
    write(payload)
    with pytest.raises(ValueError, match='allowed aligned contiguous'):
        apply_dynamic_schedule_replay(
            samples, media, baseline, **dict(kwargs, cp_group_sizes=(2, 8))
        )
    with pytest.raises(ValueError, match='exceed'):
        apply_dynamic_schedule_replay(
            samples, media, baseline, **dict(kwargs, config=_config(limit=8))
        )
    samples[0]['tokens'][0] += 1
    with pytest.raises(ValueError, match='source fingerprint'):
        apply_dynamic_schedule_replay(samples, media, baseline, **kwargs)
