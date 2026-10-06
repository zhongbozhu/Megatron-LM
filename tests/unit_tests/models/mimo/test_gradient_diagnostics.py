# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Linear gradient fingerprints under changing data ownership and EP shards."""

import os

import pytest
import torch
import torch.distributed as dist

from examples.mimo.gradient_diagnostics import _sampling_plan, collect_gradient_diagnostics


@pytest.fixture(scope='module')
def diagnostics_group():
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    return dist.group.WORLD


def _model():
    model = torch.nn.Module()
    vision = torch.nn.Module()
    vision.encoder = torch.nn.Linear(4, 2, bias=False)
    vision.merger = torch.nn.Linear(2, 2, bias=False)
    model.modality_submodules = torch.nn.ModuleDict(
        {'images': torch.nn.ModuleDict({'encoders': torch.nn.ModuleDict({'qwen35': vision})})}
    )
    model.language_model = torch.nn.Module()
    model.language_model.router = torch.nn.Linear(4, 2, bias=False)
    model.language_model.experts = torch.nn.Linear(4, 3, bias=False)
    model.language_model.experts.weight.allreduce = False
    model.language_model.shared_experts = torch.nn.Linear(4, 2, bias=False)
    model.language_model.mtp = torch.nn.Linear(4, 2, bias=False)
    return model.cuda()


def _assign_grads(model, factor, owner):
    expected = {}
    for name, parameter in model.named_parameters():
        pattern = torch.arange(1, parameter.numel() + 1, device='cuda', dtype=torch.float64)
        pattern = pattern.reshape(parameter.shape)
        if not getattr(parameter, 'allreduce', True):
            gradient = pattern * (owner + 1)
            expected['experts'] = pattern.sum().item() * (owner + 1)
        else:
            gradient = pattern * factor
            component = (
                ('merger' if '.merger.' in name else 'vision')
                if name.startswith('modality_submodules.')
                else name.split('.')[1]
            )
            expected[component] = pattern.sum().item()
        # main_grad takes priority over an intentionally incorrect .grad tensor.
        parameter.main_grad = gradient.clone()
        parameter.grad = torch.full_like(parameter, -999)
    return expected


def test_fingerprints_equal_dense_global_sums_and_ignore_partition(diagnostics_group):
    group = diagnostics_group
    rank, world = group.rank(), group.size()
    model = _model()
    expected = _assign_grads(model, 1 / world, rank)
    snapshots = {name: parameter.main_grad.clone() for name, parameter in model.named_parameters()}
    required = ('vision', 'merger', 'decoder', 'routers', 'experts', 'mtp')
    balanced = collect_gradient_diagnostics(
        model, group, 37, ep_group=group, required_components=required
    )
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.main_grad, snapshots[name], rtol=0, atol=0)
        assert (parameter.grad == -999).all()

    # Dense contributions move to unequal rank shares; expert ownership remains
    # fixed because EP dispatch already aggregates expert token work to owners.
    factor = 2 * (rank + 1) / (world * (world + 1))
    _assign_grads(model, factor, rank)
    repacked = collect_gradient_diagnostics(
        model, group, 37, ep_group=group, required_components=required
    )
    expected['routers'] = expected.pop('router')
    expected['experts'] = 78 * world * (world + 1) / 2
    expected['decoder'] = sum(
        expected[name] for name in ('routers', 'experts', 'shared_experts', 'mtp')
    )
    for component, dense_sum in expected.items():
        original = balanced['components'][component]
        moved = repacked['components'][component]
        # Every tiny test parameter fits the sample budget, so projection zero
        # is an exact independently known global gradient checksum.
        assert original['projections'][0] == pytest.approx(dense_sum / 37, rel=1e-12)
        assert moved['projections'] == pytest.approx(original['projections'], rel=1e-12, abs=1e-12)
        assert original['all_finite']
        assert original['active_ranks'] == world
    assert (
        balanced['components']['vision']['local_raw_l2']
        != repacked['components']['vision']['local_raw_l2']
        or world == 1
    )


def test_nonfinite_unsampled_gradient_is_detected_on_every_rank(diagnostics_group):
    model = torch.nn.Module()
    model.language_model = torch.nn.Linear(16, 16, bias=False).cuda()
    parameter = model.language_model.weight
    parameter.main_grad = torch.ones_like(parameter).t()  # Also cover noncontiguous storage.
    selected, _ = _sampling_plan('language_model.weight', 256, None, 4, 3)
    if diagnostics_group.rank() == 0:
        unsampled = next(index for index in range(256) if index not in selected)
        parameter.main_grad[unsampled // 16, unsampled % 16] = float('nan')
    with pytest.raises(RuntimeError, match='decoder: nonfinite gradient'):
        collect_gradient_diagnostics(
            model,
            diagnostics_group,
            7,
            ep_group=diagnostics_group,
            samples_per_parameter=4,
            projections=3,
        )


def test_required_component_rejects_silent_missing_gradient(diagnostics_group):
    model = _model()
    _assign_grads(model, 1, diagnostics_group.rank())
    model.language_model.mtp.weight.main_grad.zero_()
    with pytest.raises(RuntimeError, match='mtp: missing or zero gradient contributions'):
        collect_gradient_diagnostics(
            model, diagnostics_group, 7, ep_group=diagnostics_group, required_components=('mtp',)
        )
