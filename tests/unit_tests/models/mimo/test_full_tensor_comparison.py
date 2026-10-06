# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CPU coverage checks for complete real-token comparisons across CP layouts."""

import json

import pytest
import torch

from examples.mimo.full_tensor_comparison import compare_field, index_snapshots

FIELD = 'moe_layers/0/boundaries/moe_output/0/value'
LENGTHS = {0: 3, 1: 5}


def _save(directory, rank, group, ids, padded, indices):
    logical = [0]
    for sid in ids:
        logical.append(logical[-1] + LENGTHS[sid])
    if len(padded) == len(ids) + 2:
        logical.append(logical[-1] + padded[-1] - padded[-2])
    full = torch.full((padded[-1], 1, 2), 999.0, dtype=torch.float64)
    for slot, sid in enumerate(ids):
        for position in range(LENGTHS[sid]):
            full[padded[slot] + position, 0] = torch.tensor([sid * 100 + position, position + 1])
    data = dict(
        step=0,
        phase='training',
        round=0,
        cp_ranks=group,
        sample_ids=ids,
        sample_lengths=[LENGTHS[sid] for sid in ids],
        padded_boundaries=padded,
        logical_boundaries=logical,
        physical_indices=torch.tensor(indices),
        local_tokens=len(indices),
        moe_layers={'0': {'boundaries': {'moe_output': [{'value': full[indices]}]}}},
    )
    path = directory / 'step00000000' / f'rank{rank:05d}' / 'training-round0000.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    return path


def _layouts(tmp_path):
    reference, candidate = tmp_path / 'reference', tmp_path / 'candidate'
    for rank in range(2):
        _save(reference, rank, [rank], [rank], [0, 8], list(range(8)))
    # Reverse sample order and split every physical sequence, including dummy,
    # into complementary zigzag CP shards.
    _save(candidate, 0, [0, 1], [1, 0], [0, 8, 16, 20], [0, 1, 6, 7, 8, 9, 14, 15, 16, 19])
    _save(candidate, 1, [0, 1], [1, 0], [0, 8, 16, 20], [2, 3, 4, 5, 10, 11, 12, 13, 17, 18])
    return reference, candidate


def _index(directory):
    return index_snapshots(
        directory, [FIELD], step=0, phase='training', world_size=2, rounds=1, samples=2, tokens=8
    )


def test_repacking_zigzag_and_dummy_preserve_every_real_row(tmp_path):
    reference, candidate = _layouts(tmp_path)
    a, b = _index(reference), _index(candidate)
    result = compare_field(a, b, FIELD)
    assert result['elements'] == 16 and result['exact']
    assert result['relative_l2'] == 0 and result['scale'] == 1
    assert b['coverage']['padding_tokens'] == 12
    assert b['coverage']['trailing_dummy_tokens'] == 4


def test_full_reader_detects_unsampled_single_token(tmp_path):
    reference, candidate = _layouts(tmp_path)
    path = candidate / 'step00000000/rank00001/training-round0000.pt'
    data = torch.load(path, weights_only=True)
    # Physical row 2 is sample 1, logical position 2.
    data['moe_layers']['0']['boundaries']['moe_output'][0]['value'][0, 0, 0] += 0.25
    torch.save(data, path)
    result = compare_field(_index(reference), _index(candidate), FIELD)
    assert not result['exact'] and result['changed_rows'] == result['changed_elements'] == 1
    assert result['max_abs'] == 0.25
    assert result['changed_samples']['1']['first_positions'] == [2]


@pytest.mark.parametrize(
    'failure', ['missing_file', 'duplicate_owner', 'logical_boundary', 'missing_field']
)
def test_invalid_coverage_or_schema_is_rejected(tmp_path, failure):
    _, candidate = _layouts(tmp_path)
    path = candidate / 'step00000000/rank00001/training-round0000.pt'
    if failure == 'missing_file':
        path.unlink()
    else:
        data = torch.load(path, weights_only=True)
        if failure == 'duplicate_owner':
            data['physical_indices'][0] = 0
        elif failure == 'logical_boundary':
            data['logical_boundaries'][-1] += 1
        else:
            del data['moe_layers']
        torch.save(data, path)
    with pytest.raises((ValueError, KeyError)):
        _index(candidate)


@pytest.mark.parametrize('nonfinite', [float('nan'), float('inf'), float('-inf')])
def test_nonfinite_values_cannot_report_numeric_success(tmp_path, nonfinite):
    reference, candidate = _layouts(tmp_path)
    path = candidate / 'step00000000/rank00000/training-round0000.pt'
    data = torch.load(path, weights_only=True)
    data['moe_layers']['0']['boundaries']['moe_output'][0]['value'][0, 0, 0] = nonfinite
    torch.save(data, path)
    result = compare_field(_index(reference), _index(candidate), FIELD)
    assert not result['exact'] and not result['all_finite']
    assert not result['numerical_metrics_valid']
    assert result['nonfinite_candidate_elements'] == 1
    assert result['absolute_l2'] is result['max_abs'] is result['relative_l2'] is None
    json.dumps(result, allow_nan=False)


def test_zero_norm_reports_absolute_error_without_inventing_relative_floor(tmp_path):
    reference, candidate = _layouts(tmp_path)
    for path in reference.glob('step*/rank*/*.pt'):
        data = torch.load(path, weights_only=True)
        data['moe_layers']['0']['boundaries']['moe_output'][0]['value'].zero_()
        torch.save(data, path)
    result = compare_field(_index(reference), _index(candidate), FIELD)
    assert result['numerical_metrics_valid'] and result['absolute_l2'] > 0
    assert result['relative_l2'] is result['cosine'] is result['scale'] is None


def test_dtype_mismatch_is_not_silently_upcast(tmp_path):
    reference, candidate = _layouts(tmp_path)
    for path in candidate.glob('step*/rank*/*.pt'):
        data = torch.load(path, weights_only=True)
        entry = data['moe_layers']['0']['boundaries']['moe_output'][0]
        entry['value'] = entry['value'].float()
        torch.save(data, path)
    with pytest.raises(ValueError, match='schemas disagree'):
        compare_field(_index(reference), _index(candidate), FIELD)
