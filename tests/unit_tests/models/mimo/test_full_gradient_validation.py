# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Full-gradient parity uses actual native DDP reduce-scatter ownership."""

import json
import math

import pytest
import torch
import torch.distributed as dist

from examples.mimo.full_gradient_validation import _owned_gradients, validate_full_gradients
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(scope='module')
def groups():
    Utils.initialize_model_parallel(expert_model_parallel_size=Utils.world_size)
    yield dist.group.WORLD
    Utils.destroy_model_parallel()


def _finalized_model(group):
    model = torch.nn.Module()
    vision = torch.nn.Module()
    vision.encoder = torch.nn.Linear(13, 37, bias=False)
    vision.merger = torch.nn.Linear(13, 37, bias=False)
    model.modality_submodules = torch.nn.ModuleDict({'images': vision})
    model.language_model = torch.nn.Module()
    for name in ('router', 'experts', 'shared_experts', 'mtp'):
        setattr(model.language_model, name, torch.nn.Linear(13, 37, bias=False))
    model.language_model.experts.weight.allreduce = False
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=1,
        num_moe_experts=group.size(),
        expert_model_parallel_size=group.size(),
        calculate_per_token_loss=True,
    )
    ddp = DistributedDataParallel(
        config,
        DistributedDataParallelConfig(
            use_distributed_optimizer=True, overlap_grad_reduce=False, grad_reduce_in_fp32=True
        ),
        model.cuda(),
    )
    ddp.zero_grad_buffer()
    expected_square = 0.0
    expected_elements = 0
    denominator = group.size() * 3
    for name, parameter in model.named_parameters():
        pattern = torch.arange(1, parameter.numel() + 1, dtype=torch.float64).reshape(
            parameter.shape
        )
        expert = not getattr(parameter, 'allreduce', True)
        parameter.main_grad.copy_(pattern.cuda() * (group.rank() + 1))
        expected = pattern * (
            (group.rank() + 1) if expert else group.size() * (group.size() + 1) / 2
        )
        expected /= denominator
        if expert or group.rank() == 0:
            expected_square += expected.square().sum().item()
            expected_elements += parameter.numel()
    finalize_model_grads([ddp], num_tokens=torch.tensor(3, device='cuda', dtype=torch.int64))
    expected = torch.tensor(
        [expected_square, expected_elements], device='cuda', dtype=torch.float64
    )
    dist.all_reduce(expected, group=group)
    return ddp, expected.tolist()


def test_full_owned_shards_match_global_reference_and_detect_single_entry(groups, tmp_path):
    model, (expected_square, expected_elements) = _finalized_model(groups)
    buffers = model.buffers + model.expert_parallel_buffers
    snapshots = [buffer.grad_data.clone() for buffer in buffers]
    baseline = validate_full_gradients([model], tmp_path, write_reference=True, chunk_elements=71)
    assert baseline['global_elements'] == expected_elements
    assert baseline['global_l2'] == pytest.approx(math.sqrt(expected_square), rel=1e-7)
    assert baseline['totals']['relative_l2'] == 0
    manifest = json.loads((tmp_path / f'rank{dist.get_rank():05d}.json').read_text())
    count = sum(entry['elements'] for entry in manifest['entries'])
    assert (tmp_path / f'rank{dist.get_rank():05d}.f32').stat().st_size == count * 4
    exact = validate_full_gradients(model, tmp_path, chunk_elements=127)
    assert exact['totals']['relative_l2'] == 0
    assert exact['totals']['max_abs_error'] == 0
    for buffer, before in zip(buffers, snapshots):
        torch.testing.assert_close(buffer.grad_data, before, rtol=0, atol=0)
    # Change one element on one owner, including values outside any sparse plan.
    if groups.rank() == 0:
        _, gradient = _owned_gradients([model])[0]
        gradient[0].add_(0.125)
    changed = validate_full_gradients(model, tmp_path, chunk_elements=53)
    assert changed['totals']['max_abs_error'] == pytest.approx(0.125, rel=1e-7)
    assert changed['totals']['relative_l2'] == pytest.approx(
        0.125 / baseline['global_l2'], rel=1e-7
    )
    assert changed['all_finite']


def test_manifest_mismatch_fails_on_all_ranks(groups, tmp_path):
    model, _ = _finalized_model(groups)
    validate_full_gradients(model, tmp_path, write_reference=True)
    if groups.rank() == 0:
        path = tmp_path / f'rank{dist.get_rank():05d}.json'
        manifest = json.loads(path.read_text())
        manifest['entries'][0]['parameter_offset'] += 1
        path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match='Full gradient validation failed on 1 ranks'):
        validate_full_gradients(model, tmp_path)


def test_nonfinite_owned_entry_is_reported_globally(groups, tmp_path):
    model, _ = _finalized_model(groups)
    validate_full_gradients(model, tmp_path, write_reference=True)
    if groups.rank() == 0:
        _, gradient = _owned_gradients([model])[0]
        gradient[-1] = float('nan')
    result = validate_full_gradients(model, tmp_path)
    assert not result['all_finite']
