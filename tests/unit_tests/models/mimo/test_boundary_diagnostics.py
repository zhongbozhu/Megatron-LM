# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CPU snapshots exercise the transport reference without requiring a process group."""

from types import SimpleNamespace

import pytest
import torch

from examples.mimo.boundary_diagnostics import BoundaryDiagnostics, analyze_boundary_directory


def _transfer(features, ranks, slices):
    plan = SimpleNamespace(
        cp_ranks=ranks,
        features=[
            SimpleNamespace(producer_rank=rank, offset=start, length=count)
            for rank, start, count in slices
        ],
    )
    return SimpleNamespace(features=features, plans=[plan], local_plan=0)


def _record_step(directory, *, corrupt_receiver=False, corrupt_return=False):
    diagnostics = [BoundaryDiagnostics(directory, rank) for rank in range(2)]
    source = torch.arange(1, 7, dtype=torch.bfloat16).reshape(3, 2).requires_grad_()
    media = [dict(image_id='image-a', source_id='sample-a', offset=0, length=3)]
    diagnostics[0].save_producer(7, source, media, num_rounds=2)
    diagnostics[1].save_producer(7, source[:0], [], num_rounds=2)
    # Reordered and repeated source slices exercise an independent adjoint reference.
    slices = [(0, 2, 1), (0, 0, 2), (0, 2, 1)]
    expected = torch.cat((source[2:], source[:2], source[2:])).detach()
    returned = torch.zeros_like(source, dtype=torch.float32)
    for round_id in range(2):
        for rank, recorder in enumerate(diagnostics):
            leaf = expected.clone().requires_grad_()
            if corrupt_receiver and rank == 1 and round_id == 0:
                with torch.no_grad():
                    leaf[0, 0] += 1
            transfer = _transfer(leaf, [0, 1], slices)
            recorder.save_receiver(7, round_id, 'stats', transfer)
            recorder.save_receiver(7, round_id, 'training', transfer)
            # CP ranks consume disjoint positions; one rank can have no leaf gradient.
            if round_id == 0 or rank == 0:
                leaf.grad = torch.zeros_like(leaf)
                leaf.grad[rank::2] = (rank + 1) * (round_id + 1)
                returned[2] += leaf.grad[0].float() + leaf.grad[3].float()
                returned[:2] += leaf.grad[1:3].float()
            recorder.save_receiver_gradient(7, round_id, transfer, normalizer=8)
    if corrupt_return:
        returned[0, 0] += 1
    diagnostics[0].save_producer_gradient(7, returned, normalizer=8)
    diagnostics[1].save_producer_gradient(7, returned[:0], normalizer=8)
    assert source.grad is None
    torch.testing.assert_close(
        source.detach(), torch.arange(1, 7, dtype=torch.bfloat16).reshape(3, 2)
    )
    return diagnostics


def test_boundary_roundtrip_with_empty_producer_repeated_rows_and_missing_leaf(tmp_path):
    _record_step(tmp_path)
    report = analyze_boundary_directory(tmp_path, 7)
    assert report['all_exact'] and report['all_finite']
    assert len(report['receiver_comparisons']) == 8
    assert len(report['producer_gradient_comparisons']) == 2
    assert report['image_gradient_comparisons'][0]['source_id'] == 'sample-a'
    empty = report['producer_gradient_comparisons'][1]
    assert empty['elements'] == 0 and empty['relative_l2'] == 0


@pytest.mark.parametrize('kind', ['receiver', 'return'])
def test_boundary_finds_changed_receiver_or_returned_gradient(tmp_path, kind):
    _record_step(tmp_path, corrupt_receiver=kind == 'receiver', corrupt_return=kind == 'return')
    report = analyze_boundary_directory(tmp_path, 7)
    assert not report['all_exact']
    assert report['all_finite']
    key = 'receiver_comparisons' if kind == 'receiver' else 'producer_gradient_comparisons'
    assert max(item['max_abs_error'] for item in report[key]) > 0
    other = 'producer_gradient_comparisons' if kind == 'receiver' else 'receiver_comparisons'
    assert all(item['exact'] for item in report[other])


@pytest.mark.parametrize('kind', ['producer', 'producer_gradient', 'receiver', 'receiver_gradient'])
def test_partial_recording_cannot_pass(tmp_path, kind):
    _record_step(tmp_path)
    directory = tmp_path / 'step00000007'
    next(directory.glob(f'{kind}-rank00000*.pt')).unlink()
    with pytest.raises(ValueError):
        analyze_boundary_directory(tmp_path, 7)


def test_opt_in_steps_and_no_overwrite(tmp_path):
    recorder = BoundaryDiagnostics(tmp_path, 0, selected_steps=[3])
    features = torch.zeros((0, 2))
    recorder.save_producer(2, features, [])
    assert not list(tmp_path.iterdir())
    recorder.save_producer(3, features, [])
    with pytest.raises(FileExistsError):
        recorder.save_producer(3, features, [])
    with pytest.raises(ValueError, match='normalizer'):
        recorder.save_producer_gradient(3, features, normalizer=0)


def test_missing_whole_round_cannot_pass(tmp_path):
    _record_step(tmp_path)
    for path in (tmp_path / 'step00000007').glob('*-round00001-*.pt'):
        path.unlink()
    with pytest.raises(ValueError, match='Missing complete decoder rounds'):
        analyze_boundary_directory(tmp_path, 7)


def test_normalization_and_nonfinite_values_are_checked(tmp_path):
    _record_step(tmp_path)
    path = tmp_path / 'step00000007' / 'producer_gradient-rank00000.pt'
    record = torch.load(path, weights_only=True)
    record['values'][0, 0] = float('nan')
    torch.save(record, path)
    assert not analyze_boundary_directory(tmp_path, 7)['all_finite']
    record['normalizer'] = 2
    torch.save(record, path)
    with pytest.raises(ValueError, match='normalization'):
        analyze_boundary_directory(tmp_path, 7)
