# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Full encoder boundary comparisons must follow images across producer layouts."""

import json
import math
from types import SimpleNamespace

import pytest
import torch

from examples.mimo.boundary_diagnostics import (
    BoundaryDiagnostics,
    analyze_cp_image_coverage,
    compare_boundary_directories,
)


def _record(directory, *, reordered=False, gradient_delta=0.0, zero_gradient=False):
    features = torch.arange(1, 7, dtype=torch.bfloat16).reshape(3, 2)
    gradient = torch.zeros((3, 2)) if zero_gradient else torch.arange(2, 8).reshape(3, 2).float()
    gradient[0, 0] += gradient_delta
    # Two images in the same sample ensure image identity is not replaced by source identity.
    image_a = dict(source_id='sample', image_id='a', offset=0, length=2)
    image_b = dict(source_id='sample', image_id='b', offset=0, length=1)
    if reordered:
        rows = torch.tensor([2, 0, 1])
        image_a['offset'] = 1
        layout = [(2, rows, [image_b, image_a]), (7, rows[:0], [])]
    else:
        layout = [(0, torch.tensor([0, 1]), [image_a]), (1, torch.tensor([2]), [image_b])]
    for rank, rows, media in layout:
        recorder = BoundaryDiagnostics(directory, rank)
        recorder.save_producer(7, features[rows], media, num_rounds=2)
        recorder.save_producer_gradient(7, gradient[rows], normalizer=8)


def _mutate(directory, kind, rank, change):
    path = directory / 'step00000007' / f'{kind}-rank{rank:05d}.pt'
    record = torch.load(path, weights_only=True)
    change(record)
    torch.save(record, path)


def test_every_image_row_compared_after_reassignment_and_reordering(tmp_path):
    reference, candidate = tmp_path / 'reference', tmp_path / 'candidate'
    _record(reference)
    _record(candidate, reordered=True)
    report = compare_boundary_directories(reference, candidate, 7, expected_images=2)
    assert report['all_exact'] and report['all_finite']
    assert report['images'] == 2
    assert report['reference_producers'] == report['candidate_producers'] == 2
    assert report['normalizer'] == 8
    for value in report['totals'].values():
        assert value['elements'] == 6
        assert value['relative_l2'] == 0
        assert value['cosine'] == pytest.approx(1)
    assert {row['image_id']: row['reference_rank'] for row in report['image_comparisons']} == {
        'a': 0,
        'b': 1,
    }
    assert {row['candidate_rank'] for row in report['image_comparisons']} == {2}
    json.dumps(report, allow_nan=False)


def test_reports_normalized_full_gradient_difference_without_changing_features(tmp_path):
    reference, candidate = tmp_path / 'reference', tmp_path / 'candidate'
    _record(reference)
    _record(candidate, reordered=True, gradient_delta=2)
    report = compare_boundary_directories(reference, candidate, 7)
    assert not report['all_exact']
    assert report['totals']['features']['exact']
    value = report['totals']['gradient']
    assert value['squared_error'] == pytest.approx((2 / 8) ** 2)
    assert value['squared_reference'] == pytest.approx(sum(x * x for x in range(2, 8)) / 64)
    assert value['relative_l2'] == pytest.approx(math.sqrt(4 / sum(x * x for x in range(2, 8))))
    images = {row['image_id']: row for row in report['image_comparisons']}
    assert images['b']['gradient']['exact']
    assert not images['a']['gradient']['exact']


@pytest.mark.parametrize('missing', ['gradient_file', 'image_metadata', 'image_pair', 'source_id'])
def test_missing_or_mismatched_identity_cannot_pass(tmp_path, missing):
    reference, candidate = tmp_path / 'reference', tmp_path / 'candidate'
    _record(reference)
    _record(candidate)
    if missing == 'gradient_file':
        (candidate / 'step00000007' / 'producer_gradient-rank00001.pt').unlink()
    elif missing == 'image_metadata':
        _mutate(candidate, 'producer', 1, lambda row: row.update(media=[]))
    elif missing == 'image_pair':
        for path in (candidate / 'step00000007').glob('*-rank00001.pt'):
            path.unlink()
    else:
        _mutate(candidate, 'producer', 1, lambda row: row['media'][0].update(source_id='other'))
    with pytest.raises(ValueError):
        compare_boundary_directories(reference, candidate, 7)


@pytest.mark.parametrize('invalid', ['duplicate', 'offset', 'normalizer', 'nan'])
def test_ambiguous_mapping_or_invalid_gradients_cannot_pass(tmp_path, invalid):
    reference, candidate = tmp_path / 'reference', tmp_path / 'candidate'
    _record(reference)
    _record(candidate, reordered=True)
    if invalid == 'duplicate':
        _mutate(candidate, 'producer', 2, lambda row: row['media'][1].update(image_id='b'))
    elif invalid == 'offset':
        _mutate(candidate, 'producer', 2, lambda row: row['media'][1].update(offset=0))
    elif invalid == 'normalizer':
        _mutate(candidate, 'producer_gradient', 2, lambda row: row.update(normalizer=4))
    else:
        _mutate(candidate, 'producer_gradient', 2, lambda row: row['values'].fill_(float('nan')))
    with pytest.raises(ValueError):
        compare_boundary_directories(reference, candidate, 7)


def test_expected_source_coverage_rejects_identically_incomplete_directories(tmp_path):
    reference, candidate = tmp_path / 'reference', tmp_path / 'candidate'
    _record(reference)
    _record(candidate)
    for directory in (reference, candidate):
        for path in (directory / 'step00000007').glob('*-rank00001.pt'):
            path.unlink()
    with pytest.raises(ValueError, match='expected source coverage'):
        compare_boundary_directories(reference, candidate, 7, expected_images=2)


def test_zero_reference_gradient_still_reports_nonzero_error_as_valid_json(tmp_path):
    reference, candidate = tmp_path / 'reference', tmp_path / 'candidate'
    _record(reference, zero_gradient=True)
    _record(candidate, zero_gradient=True, gradient_delta=2)
    report = compare_boundary_directories(reference, candidate, 7)
    assert report['totals']['gradient']['relative_l2'] is None
    assert report['totals']['gradient']['squared_error'] == pytest.approx(1 / 16)
    assert not report['all_exact']
    json.dumps(report, allow_nan=False)


def _cp_capture(boundary, execution):
    features = torch.arange(6, dtype=torch.bfloat16).reshape(3, 2)
    BoundaryDiagnostics(boundary, 7).save_producer(
        7,
        features,
        [
            dict(source_id='sample', image_id='a', offset=0, length=2),
            dict(source_id='sample', image_id='b', offset=2, length=1),
        ],
    )
    plan = SimpleNamespace(
        cp_ranks=[0, 1, 2, 3],
        features=[
            SimpleNamespace(producer_rank=7, offset=2, length=1),
            SimpleNamespace(producer_rank=7, offset=0, length=2),
        ],
    )
    transfer = SimpleNamespace(features=features[[2, 0, 1]], plans=[plan], local_plan=0)
    for rank, rows in enumerate(([1], [2], [0], [])):
        BoundaryDiagnostics(boundary, rank).save_receiver(7, 0, 'training', transfer)
        directory = execution / 'step00000007' / f'rank{rank:05d}'
        directory.mkdir(parents=True)
        # Original forward and recompute must not count the same owner twice.
        record = dict(
            step=7,
            round=0,
            phase='training',
            cp_ranks=plan.cp_ranks,
            cp_input=[dict(feature_rows=torch.tensor(rows, dtype=torch.long))] * 2,
        )
        torch.save(record, directory / 'training-round0000.pt')


def test_actual_cp_coverage_counts_split_image_and_empty_local_vision(tmp_path):
    boundary, execution = tmp_path / 'boundary', tmp_path / 'execution'
    _cp_capture(boundary, execution)
    report = analyze_cp_image_coverage(boundary, execution, 7, expected_images=2)
    assert report['all_image_rows_owned_once']
    assert report['images'] == 2 and report['visual_rows'] == 3
    assert report['split_images'] == 1 and report['max_image_owners'] == 2
    assert report['rank_rounds'] == 4
    assert (
        report['empty_local_vision_rank_rounds'] == report['empty_local_vision_in_image_packs'] == 1
    )
    assert {row['image_id']: row['owners'] for row in report['image_owners']} == {
        'a': [0, 1],
        'b': [2],
    }


@pytest.mark.parametrize(
    'invalid', ['duplicate', 'missing', 'missing_empty', 'out_of_bounds', 'group']
)
def test_invalid_cp_ownership_cannot_be_counted_as_coverage(tmp_path, invalid):
    boundary, execution = tmp_path / 'boundary', tmp_path / 'execution'
    _cp_capture(boundary, execution)
    path = execution / 'step00000007' / 'rank00001' / 'training-round0000.pt'
    if invalid == 'missing_empty':
        path = execution / 'step00000007' / 'rank00003' / 'training-round0000.pt'
    if invalid in ('missing', 'missing_empty'):
        path.unlink()
    else:
        record = torch.load(path, weights_only=True)
        if invalid == 'group':
            record['cp_ranks'] = [0, 1]
        else:
            record['cp_input'][0]['feature_rows'][0] = 1 if invalid == 'duplicate' else 3
        torch.save(record, path)
    with pytest.raises(ValueError):
        analyze_cp_image_coverage(boundary, execution, 7)
