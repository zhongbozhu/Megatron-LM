# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CPU tests use real torch AdamW, with a minimal native ownership/step adapter.

Run with --confcutdir=tests/unit_tests/models/mimo to avoid the parent CUDA fixture.
The adapter only supplies native shard metadata and clipping; it never implements
Adam's update equations. All weights and moments are updated by torch AdamW.
"""

import copy
import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from examples.mimo import optimizer_diagnostics as diagnostics


class _NativeOptimizer:
    def __init__(self, model):
        self.config = SimpleNamespace(optimizer='adam', decoupled_weight_decay=True, clip_grad=0.7)
        parameters = list(model.parameters())
        self.masters = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
        self.optimizer = torch.optim.AdamW(
            [
                {'params': self.masters[:1], 'weight_decay': 0.15},
                {'params': self.masters[1:], 'weight_decay': 0.0},
            ],
            lr=0.02,
            betas=(0.8, 0.9),
            eps=1e-6,
            foreach=False,
        )
        self.model_float16_groups, self.shard_fp32_from_float16_groups = [], []
        self.model_fp32_groups, self.shard_fp32_groups = [parameters], [self.masters]
        self.grad_norms_by_group = {}
        self.calls = 0

    def _get_model_param_range_map(self, parameter):
        return {'param': SimpleNamespace(start=0, end=parameter.numel())}

    def step(self):
        self.calls += 1
        for parameter, master in zip(self.model_fp32_groups[0], self.masters):
            master.grad = parameter.main_grad
        norm = torch.nn.utils.clip_grad_norm_(self.masters, self.config.clip_grad)
        self.optimizer.step()
        return True, float(norm), 0


def _fixture(monkeypatch, *, warm=True):
    model = torch.nn.Module()
    model.language_model = torch.nn.Module()
    model.language_model.decoder = torch.nn.Module()
    model.language_model.decoder.weight = torch.nn.Parameter(torch.arange(1, 10).float() / 10)
    model.modality_submodules = torch.nn.Module()
    model.modality_submodules.vision = torch.nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        model.modality_submodules.vision.weight.copy_(torch.arange(6).view(2, 3).float() / 8)
    model.config = SimpleNamespace(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        fp16=False,
        calculate_per_token_loss=True,
    )
    model.ddp_config = SimpleNamespace(
        use_distributed_optimizer=True, num_distributed_optimizer_instances=1
    )
    parameters = list(model.parameters())
    gradient = torch.linspace(-1.0, 1.5, sum(p.numel() for p in parameters))
    group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    monkeypatch.setattr(diagnostics.dist, 'get_process_group_ranks', lambda _: [0])
    ranges = {}
    start = 0
    for parameter in parameters:
        end = start + parameter.numel()
        ranges[parameter] = (start, end, 0)
        parameter.main_grad = gradient[start:end].view(parameter.shape)
        start = end
    model.buffers = [
        SimpleNamespace(
            data_parallel_group=group,
            num_optimizer_shards=1,
            param_index_map=ranges,
            buckets=[SimpleNamespace(offset=0, grad_data=gradient)],
        )
    ]
    model.expert_parallel_buffers = []
    optimizer = _NativeOptimizer(model)
    if warm:
        optimizer.step()
        optimizer.calls = 0
        gradient.copy_(torch.linspace(0.2, 1.7, gradient.numel()))
    return model, optimizer


def _options(tmp_path, *, write_reference, expected_step=1, suffix='candidate', **kwargs):
    return dict(
        reference_dir=tmp_path / 'reference',
        write_reference=write_reference,
        scratch_dir=tmp_path / 'scratch',
        report_path=tmp_path / f'{"reference" if write_reference else suffix}.json',
        expected_step=expected_step,
        chunk_elements=4,
        context={'scheduler': {'num_steps': expected_step}, 'source_batch': [37, 38]},
        **kwargs,
    )


def _run(model, optimizer, options):
    original = optimizer.step
    inner_original = optimizer.optimizer.step
    with diagnostics.optimizer_step_diagnostics(model, optimizer, **options) as observer:
        result = optimizer.step()
    assert optimizer.step == original
    assert optimizer.optimizer.step == inner_original
    assert result[0]
    assert observer.report is not None
    return observer.report


def _reference(monkeypatch, tmp_path, *, warm=True):
    model, optimizer = _fixture(monkeypatch, warm=warm)
    report = _run(
        model, optimizer, _options(tmp_path, write_reference=True, expected_step=int(warm))
    )
    return report


def test_complete_warm_adamw_update_and_clipping_are_observed(monkeypatch, tmp_path):
    reference_report = _reference(monkeypatch, tmp_path)
    model, optimizer = _fixture(monkeypatch)
    report = _run(model, optimizer, _options(tmp_path, write_reference=False))
    assert report['initial_state_exact']
    assert report['full_update_elements'] == 15
    assert report['positive_lr_owned_elements'] == 15
    assert report['zero_lr_owned_elements'] == 0
    assert report['nonzero_master_update']
    assert report['tolerance_pass'] is None
    for field in diagnostics._FIELDS:
        assert report['metrics'][field]['totals']['max_abs_error'] == 0
        assert report['metrics'][field]['totals']['elements'] == 15
    norms = report['ranks'][0]['gradients']['squared_l2_by_grad_norm_group']
    assert norms['at_native_adam']['main'] < norms['before_clipping']['main']
    assert norms['at_native_adam']['main'] ** 0.5 == pytest.approx(0.7, abs=1e-6)
    assert report['ranks'][0]['native_result'][1] == pytest.approx(
        norms['before_clipping']['main'] ** 0.5
    )
    assert report['ranks'][0]['native_result'] == reference_report['ranks'][0]['native_result']
    assert report['observed_gradient_l2']['at_native_adam']['main'] == pytest.approx(0.7, abs=1e-6)
    candidate_manifest = json.loads((tmp_path / 'candidate.rank00000.json').read_text())
    assert not candidate_manifest['vectors_stored']
    assert len(candidate_manifest['gradients']['at_native_adam']) == 2
    assert list((tmp_path / 'scratch').iterdir()) == []
    assert (tmp_path / 'reference/rank00000.f32').stat().st_size == 15 * 12


def test_last_unsampled_element_changes_real_update_and_moments(monkeypatch, tmp_path):
    _reference(monkeypatch, tmp_path)
    model, optimizer = _fixture(monkeypatch)
    model.buffers[0].buckets[0].grad_data[-1] += 0.25
    report = _run(model, optimizer, _options(tmp_path, write_reference=False))
    for field in diagnostics._FIELDS:
        assert report['metrics'][field]['totals']['max_abs_error'] > 0
    assert report['metrics']['master_update']['totals']['cosine'] < 1
    assert report['tolerance_pass'] is None


@pytest.mark.parametrize('changed', ['master', 'exp_avg', 'exp_avg_sq', 'step', 'lr', 'context'])
def test_initial_state_difference_fails_before_native_step(monkeypatch, tmp_path, changed):
    _reference(monkeypatch, tmp_path)
    model, optimizer = _fixture(monkeypatch)
    options = _options(tmp_path, write_reference=False)
    master = optimizer.masters[0]
    with torch.no_grad():
        if changed == 'master':
            master[-1] += 0.01
        elif changed in ('exp_avg', 'exp_avg_sq', 'step'):
            optimizer.optimizer.state[master][changed].view(-1)[-1] += 1
        elif changed == 'lr':
            optimizer.optimizer.param_groups[0]['lr'] *= 2
        else:
            options['context']['scheduler']['num_steps'] += 1
    before = [p.detach().clone() for p in optimizer.masters]
    original = optimizer.step
    with pytest.raises(RuntimeError, match='state|step|hyperparameters'):
        with diagnostics.optimizer_step_diagnostics(model, optimizer, **options) as observer:
            optimizer.step()
    assert optimizer.calls == 0
    assert optimizer.step == original
    assert observer.report is None
    assert all(torch.equal(p, q) for p, q in zip(before, optimizer.masters))
    assert list((tmp_path / 'scratch').iterdir()) == []


@pytest.mark.parametrize('missing', ['exp_avg', 'exp_avg_sq', 'step'])
def test_missing_warm_state_never_passes(monkeypatch, tmp_path, missing):
    _reference(monkeypatch, tmp_path)
    model, optimizer = _fixture(monkeypatch)
    del optimizer.optimizer.state[optimizer.masters[0]][missing]
    with pytest.raises(RuntimeError, match='Missing'):
        _run(model, optimizer, _options(tmp_path, write_reference=False))
    assert optimizer.calls == 0


def test_explicit_cold_lazy_initialization_and_exact_shape_of_state(monkeypatch, tmp_path):
    _reference(monkeypatch, tmp_path, warm=False)
    model, optimizer = _fixture(monkeypatch, warm=False)
    report = _run(model, optimizer, _options(tmp_path, write_reference=False, expected_step=0))
    assert report['cold_lazy_entries'] == 2
    assert report['metrics']['master_update']['totals']['max_abs_error'] == 0
    model, optimizer = _fixture(monkeypatch, warm=False)
    master = optimizer.masters[0]
    optimizer.optimizer.state[master] = {
        'step': torch.tensor(0.0),
        'exp_avg': torch.zeros_like(master),
        'exp_avg_sq': torch.zeros_like(master),
    }
    with pytest.raises(RuntimeError, match='Starting optimizer state'):
        _run(
            model,
            optimizer,
            _options(tmp_path, write_reference=False, expected_step=0, suffix='different-cold'),
        )
    assert optimizer.calls == 0


def test_reference_corruption_and_truncation_cannot_pass(monkeypatch, tmp_path):
    _reference(monkeypatch, tmp_path)
    path = tmp_path / 'reference/rank00000.f32'
    original = path.read_bytes()
    path.write_bytes(original[:-4])
    model, optimizer = _fixture(monkeypatch)
    with pytest.raises(RuntimeError, match='byte count'):
        _run(model, optimizer, _options(tmp_path, write_reference=False))
    assert optimizer.calls == 0
    changed = bytearray(original)
    changed[-1] ^= 1
    path.write_bytes(changed)
    model, optimizer = _fixture(monkeypatch)
    with pytest.raises(RuntimeError, match='checksum'):
        _run(model, optimizer, _options(tmp_path, write_reference=False))
    assert not (tmp_path / 'candidate.json').exists()


def test_wrapper_observes_once_and_never_overwrites_reference(monkeypatch, tmp_path):
    model, optimizer = _fixture(monkeypatch)
    with diagnostics.optimizer_step_diagnostics(
        model, optimizer, **_options(tmp_path, write_reference=True)
    ) as observer:
        optimizer.step()
        first_report = copy.deepcopy(observer.report)
        optimizer.step()
    assert optimizer.calls == 2
    assert observer.report == first_report
    model, optimizer = _fixture(monkeypatch)
    options = _options(tmp_path, write_reference=True)
    options['report_path'] = tmp_path / 'another-reference.json'
    with pytest.raises(RuntimeError, match='existing optimizer reference'):
        _run(model, optimizer, options)
    assert optimizer.calls == 0


def test_decay_removed_update_uses_actual_master_delta(monkeypatch, tmp_path):
    model, optimizer = _fixture(monkeypatch, warm=False)
    model.buffers[0].buckets[0].grad_data.zero_()
    before = optimizer.masters[0].detach().double().clone()
    report = _run(model, optimizer, _options(tmp_path, write_reference=True, expected_step=0))
    actual = optimizer.masters[0].detach().double() - before
    removed = actual + 0.02 * 0.15 * before
    assert report['metrics']['master_update']['totals']['l2'] == pytest.approx(actual.norm().item())
    assert report['metrics']['decay_removed_update']['totals']['l2'] == pytest.approx(
        removed.norm().item(), abs=1e-15
    )
    # This includes actual native FP32 rounding; it is deliberately not forced to zero.
    assert 0 < removed.norm().item() < actual.norm().item() * 1e-3


def test_mismatched_native_ownership_and_duplicate_master_fail(monkeypatch, tmp_path):
    model, optimizer = _fixture(monkeypatch)
    optimizer._get_model_param_range_map = lambda _: {'param': SimpleNamespace(start=1, end=10)}
    with pytest.raises(RuntimeError, match='ownership mismatch'):
        _run(model, optimizer, _options(tmp_path, write_reference=True))
    assert optimizer.calls == 0
    model, optimizer = _fixture(monkeypatch)
    optimizer.optimizer.param_groups[1]['params'].append(optimizer.masters[0])
    with pytest.raises(RuntimeError, match='multiple optimizer groups'):
        _run(model, optimizer, _options(tmp_path, write_reference=True))
    assert optimizer.calls == 0


def test_nonfinite_state_and_failed_native_step_never_produce_success(monkeypatch, tmp_path):
    model, optimizer = _fixture(monkeypatch)
    optimizer.optimizer.state[optimizer.masters[0]]['exp_avg'][0] = float('nan')
    with pytest.raises(RuntimeError, match='Nonfinite'):
        _run(model, optimizer, _options(tmp_path, write_reference=True))
    assert optimizer.calls == 0
    model, optimizer = _fixture(monkeypatch)
    optimizer.step = lambda: (False, None, None)
    with pytest.raises(RuntimeError, match='successful update'):
        _run(model, optimizer, _options(tmp_path, write_reference=True))
    assert not (tmp_path / 'reference.json').exists()


def test_zero_reference_norm_is_explicit_and_json_does_not_claim_tolerance_pass():
    comparison = diagnostics._Comparisons()
    metadata = {'components': ['vision']}
    comparison.add('master_update', metadata, torch.ones(3), torch.zeros(3))
    result = comparison.finish(None, torch.device('cpu'))
    totals = result['master_update']['totals']
    assert totals['relative_l2'] is None and totals['cosine'] is None
    assert totals['delta_l2'] == pytest.approx(3**0.5)
    json.dumps(result, allow_nan=False)


def test_chained_native_optimizers_preserve_actual_step_and_unique_ownership(monkeypatch, tmp_path):
    model, original = _fixture(monkeypatch)
    children = []
    for index, parameter in enumerate(original.model_fp32_groups[0]):
        child = copy.copy(original)
        master = original.masters[index]
        child.masters = [master]
        child.model_fp32_groups, child.shard_fp32_groups = [[parameter]], [[master]]
        group = original.optimizer.param_groups[index]
        child.optimizer = torch.optim.AdamW(
            [dict(group, params=[master])], lr=group['lr'], foreach=False
        )
        child.optimizer.state[master] = copy.deepcopy(original.optimizer.state[master])
        children.append(child)
    chain = SimpleNamespace(chained_optimizers=children, grad_norms_by_group={})

    def step():
        results = [child.step() for child in children]
        return (
            all(result[0] for result in results),
            sum(result[1] ** 2 for result in results) ** 0.5,
            0,
        )

    chain.step = step
    with diagnostics.optimizer_step_diagnostics(
        model, chain, **_options(tmp_path, write_reference=True)
    ) as observer:
        chain.step()
    assert chain.step is step
    assert observer.report['full_update_elements'] == 15
    manifest = json.loads((tmp_path / 'reference/rank00000.json').read_text())
    assert {entry['metadata']['optimizer'] for entry in manifest['entries']} == {0, 1}
    assert all(child.calls == 1 for child in children)


@pytest.mark.parametrize('cold_state', ['only_step', 'moments_without_torch_step'])
def test_invalid_partial_cold_state_is_not_treated_as_native_lazy(
    monkeypatch, tmp_path, cold_state
):
    model, optimizer = _fixture(monkeypatch, warm=False)
    master = optimizer.masters[0]
    optimizer.optimizer.state[master] = (
        {'step': torch.tensor(0.0)}
        if cold_state == 'only_step'
        else {'exp_avg': torch.zeros_like(master), 'exp_avg_sq': torch.zeros_like(master)}
    )
    with pytest.raises(RuntimeError, match='Missing'):
        _run(model, optimizer, _options(tmp_path, write_reference=True, expected_step=0))
    assert optimizer.calls == 0


def test_zero_learning_rate_is_explicitly_uninformative(monkeypatch, tmp_path):
    model, optimizer = _fixture(monkeypatch)
    for group in optimizer.optimizer.param_groups:
        group['lr'] = 0.0
    report = _run(model, optimizer, _options(tmp_path, write_reference=True))
    assert report['positive_lr_owned_elements'] == 0
    assert report['zero_lr_owned_elements'] == 15
    assert not report['nonzero_master_update']
    assert report['tolerance_pass'] is None


def test_nonfinite_scalar_metadata_and_native_exception_do_not_pass(monkeypatch, tmp_path):
    model, optimizer = _fixture(monkeypatch)
    optimizer.optimizer.param_groups[0]['lr'] = float('nan')
    with pytest.raises(RuntimeError, match='Nonfinite optimizer/context scalar'):
        _run(model, optimizer, _options(tmp_path, write_reference=True))
    assert optimizer.calls == 0
    model, optimizer = _fixture(monkeypatch)

    def broken_step():
        raise ValueError('expected native exception')

    optimizer.step = broken_step
    with pytest.raises(RuntimeError, match='Native step raised ValueError'):
        _run(model, optimizer, _options(tmp_path, write_reference=True))
    assert optimizer.step is broken_step


def test_exact_starting_state_does_not_depend_on_cpu_chunk_size(monkeypatch, tmp_path):
    _reference(monkeypatch, tmp_path)
    model, optimizer = _fixture(monkeypatch)
    options = _options(tmp_path, write_reference=False)
    options['chunk_elements'] = 5
    report = _run(model, optimizer, options)
    assert report['initial_state_exact']
    assert report['metrics']['master_update']['totals']['max_abs_error'] == 0


def test_fused_adam_empty_group_counter_must_also_advance():
    group = {'params': []}
    optimizer = SimpleNamespace(optimizer=SimpleNamespace(param_groups=[group]))
    descriptions = [{'optimizer': 0, 'group': 0, 'implementation': 'example.FusedAdam'}]
    assert diagnostics._snapshot_groups(optimizer, descriptions, 0)[0]['settings'] == {}
    with pytest.raises(ValueError, match='group step mismatch'):
        diagnostics._snapshot_groups(optimizer, descriptions, 1)
    group['step'] = 1
    assert diagnostics._snapshot_groups(optimizer, descriptions, 1)[0]['settings']['step'] == 1


def test_cold_moment_bands_partition_signs_and_actual_update_error():
    observer = diagnostics._ColdMomentBands()
    reference = torch.tensor([0, 0, 0.05, 0.5, 2, 20, 200, -2])
    current = torch.tensor([0, 1, -0.05, 0, 3, -20, 200, 1])
    reference_update = torch.tensor([0, 0, 2, 1, 2, 5, 7, -3])
    update = torch.tensor([0, 1, -2, 0, 2, -5, 7, 3])
    observer.add({'components': ['vision']}, current, reference, update, reference_update, 0.5, 2.0)
    report = observer.finish(None, torch.device('cpu'))
    total = report['totals']
    assert total['elements'] == 8 and total['elements_excluding_joint_zero'] == 7
    assert total['opposite_sign_fraction'] == pytest.approx(3 / 7)
    assert total['delta_update_square'] == 154
    assert total['opposite_sign_update_error_fraction'] == pytest.approx(152 / 154)
    assert {cell['band'] for cell in total['cells']} == set(observer.bands)
    assert report['components']['vision'] == total
    assert sum(cell['delta_update_square'] for cell in total['cells']) == 154


def test_cold_bands_use_actual_adam_updates_and_existing_reference_format(monkeypatch, tmp_path):
    _reference(monkeypatch, tmp_path, warm=False)
    vector = tmp_path / 'reference/rank00000.f32'
    original_bytes = vector.read_bytes()
    model, optimizer = _fixture(monkeypatch, warm=False)
    model.buffers[0].buckets[0].grad_data[0] *= -1
    options = _options(tmp_path, write_reference=False, expected_step=0)
    options['cold_moment_bands'] = True
    report = _run(model, optimizer, options)
    bands = report['cold_moment_bands']['totals']
    assert bands['elements'] == report['full_update_elements'] == 15
    assert bands['opposite_sign_fraction'] == pytest.approx(1 / 15)
    assert bands['delta_update_square'] == pytest.approx(
        report['metrics']['master_update']['totals']['delta_l2'] ** 2, rel=1e-12
    )
    assert bands['opposite_sign_update_error_fraction'] == 1
    assert report['cold_moment_bands']['accounting']['element_count_matches']
    assert report['cold_moment_bands']['accounting']['relative_disagreement'] < 1e-12
    assert vector.read_bytes() == original_bytes


def test_cold_moment_bands_are_opt_in_and_never_interpret_warm_moments(monkeypatch, tmp_path):
    reference = _reference(monkeypatch, tmp_path)
    assert reference['cold_moment_bands'] is None
    model, optimizer = _fixture(monkeypatch)
    options = _options(tmp_path, write_reference=False)
    options['cold_moment_bands'] = True
    report = _run(model, optimizer, options)
    assert report['cold_moment_bands'] is None
    assert report['metrics']['master_update']['totals']['max_abs_error'] == 0


def test_cold_sign_counts_do_not_underflow_and_streaming_preserves_totals():
    reference = torch.tensor([1e-38, -1e-38, 0, 3e-7, -2e-4])
    current = -reference
    update = torch.tensor([1.0, -1.0, 0, 0.1, -0.2])
    full, chunked = diagnostics._ColdMomentBands(), diagnostics._ColdMomentBands()
    full.add({'components': []}, current, reference, update, -update, 0.9, 1e-8)
    for start in range(0, reference.numel(), 2):
        sl = slice(start, start + 2)
        chunked.add(
            {'components': []}, current[sl], reference[sl], update[sl], -update[sl], 0.9, 1e-8
        )
    torch.testing.assert_close(full.values, chunked.values, rtol=1e-12, atol=1e-90)
    total = full.finish(None, torch.device('cpu'))['totals']
    assert total['opposite_sign_fraction'] == 1
    assert total['elements_excluding_joint_zero'] == 4
    for eps in (0.0, -1.0, float('nan'), float('inf')):
        with pytest.raises(ValueError, match='positive epsilon'):
            full.add({'components': []}, current, reference, update, -update, 0.9, eps)


def _gloo_worker(rank, directory):
    """Two independent owner shards exercise compact reduction/error rendezvous."""
    directory = Path(directory)
    diagnostics.dist.init_process_group(
        'gloo',
        init_method=f'file://{directory / "gloo-init"}',
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=40),
    )
    try:
        with pytest.MonkeyPatch.context() as monkeypatch:
            for reference in (True, False):
                model, optimizer = _fixture(monkeypatch)
                # Each rank represents a disjoint optimizer-owned parameter shard.
                monkeypatch.setattr(diagnostics.dist, 'get_process_group_ranks', lambda _: [rank])
                _run(model, optimizer, _options(directory, write_reference=reference))
            model, optimizer = _fixture(monkeypatch)
            monkeypatch.setattr(diagnostics.dist, 'get_process_group_ranks', lambda _: [rank])
            if rank == 1:
                optimizer.optimizer.state[optimizer.masters[0]]['exp_avg'][-1] += 0.5
            with pytest.raises(RuntimeError, match='failed on 1 ranks'):
                _run(
                    model, optimizer, _options(directory, write_reference=False, suffix='must-fail')
                )
            assert optimizer.calls == 0
        assert not torch.cuda.is_initialized()
    finally:
        diagnostics.dist.destroy_process_group()


def test_cpu_gloo_all_ranks_report_and_mismatch_stops_every_rank_before_update(tmp_path):
    # The CPU-only launcher runs pytest from stdin, which multiprocessing spawn
    # cannot re-import. Fork is safe here only because CUDA was never initialized.
    assert not torch.cuda.is_initialized()
    torch.multiprocessing.start_processes(
        _gloo_worker, args=(str(tmp_path),), nprocs=2, join=True, start_method='fork'
    )
    report = json.loads((tmp_path / 'candidate.json').read_text())
    assert {rank['rank'] for rank in report['ranks']} == {0, 1}
    assert report['full_update_elements'] == 30
    assert report['metrics']['master_update']['totals']['max_abs_error'] == 0
    assert not (tmp_path / 'must-fail.json').exists()
