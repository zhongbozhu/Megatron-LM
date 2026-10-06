# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""GPU checks for pack-directed feature transport and its explicit adjoint.

Run under torch.distributed.run with four or eight GPUs. These tests deliberately
use real P2P and changing CP groups, without a global feature all-gather.
"""

import os

import pytest
import torch
import torch.distributed as dist

from megatron.core.models.mimo.comm.pack_bridge import (
    FeatureSlice,
    PackFeatureBridge,
    PackFeaturePlan,
)


@pytest.fixture(scope="module")
def bridge_groups():
    """Four-rank transport domains, compatible with the standard eight-GPU runner."""
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    world = dist.get_world_size()
    if world % 4:
        pytest.skip("Pack bridge tests require a multiple of four ranks")
    rank = dist.get_rank()
    owned, created = {}, []
    for base in range(0, world, 4):
        for ranks in (
            tuple(range(base, base + 4)),
            (base, base + 1),
            (base + 2, base + 3),
            *((i,) for i in range(base, base + 4)),
        ):
            group = dist.new_group(ranks=list(ranks), backend="nccl")
            if rank in ranks:
                owned[ranks] = group
                created.append(group)
    yield owned
    for group in reversed(created):
        dist.destroy_process_group(group)


def _source(rank, base, counts, hidden=4):
    local_rank = rank - base
    return (
        torch.arange(counts[local_rank] * hidden, device="cuda", dtype=torch.float64)
        .reshape(counts[local_rank], hidden)
        .add_(100 * local_rank + 1)
    )


def _round_plans(base):
    """Uneven producers, reordered/repeated rows, empty packs and CP1/2/4."""
    a = FeatureSlice(base, 1, 3)
    b = FeatureSlice(base + 2, 0, 2)
    c = FeatureSlice(base + 3, 2, 5)
    d = FeatureSlice(base + 3, 0, 2)
    return (
        (
            PackFeaturePlan((base, base + 1), (c, a, b)),
            PackFeaturePlan((base + 2, base + 3), (b, a, a, d)),
        ),
        (
            PackFeaturePlan((base, base + 1), (b,)),
            PackFeaturePlan((base + 2,), (c, a)),
            PackFeaturePlan((base + 3,), ()),
        ),
        (PackFeaturePlan(tuple(range(base, base + 4)), (d, a, c, b, a)),),
        tuple(PackFeaturePlan((rank,), ()) for rank in range(base, base + 4)),
    )


def _cotangent(plan, consumer, base, round_index):
    """Rank 1 has no local visual-input gradient but must still participate."""
    gradient = torch.arange(plan.num_rows * 4, device="cuda", dtype=torch.float64).reshape(
        plan.num_rows, 4
    )
    if consumer == base + 1:
        return torch.zeros_like(gradient)
    return gradient.mul_(0.125).add_(1 + consumer - base + round_index)


def test_directed_pack_forward_and_explicit_backward(bridge_groups):
    rank = dist.get_rank()
    base = rank // 4 * 4
    ranks = tuple(range(base, base + 4))
    transport = bridge_groups[ranks]
    counts = (5, 0, 3, 7)
    source_input = _source(rank, base, counts)
    encoder_weight = torch.tensor(2.0, device="cuda", dtype=torch.float64, requires_grad=True)
    source_output = source_input * encoder_weight
    source_output.retain_grad()
    bridge = PackFeatureBridge(transport)

    states, expected_gradients = [], []
    for round_index, plans in enumerate(_round_plans(base)):
        local_plan = next(plan for plan in plans if rank in plan.cp_ranks)
        state = bridge.forward(source_output, plans, bridge_groups[local_plan.cp_ranks])
        assert state.features.is_leaf and state.features.requires_grad
        assert state.local_features is source_output
        expected = (
            torch.cat(
                [
                    _source(feature.producer_rank, base, counts).narrow(
                        0, feature.offset, feature.length
                    )
                    * 2
                    for feature in local_plan.features
                ]
            )
            if local_plan.features
            else source_output.new_empty((0, 4))
        )
        torch.testing.assert_close(state.features, expected, rtol=0, atol=0)
        local_cotangent = _cotangent(local_plan, rank, base, round_index)
        if rank != base + 1 and local_plan.num_rows:
            (state.features * local_cotangent).sum().backward()
        assert source_output.grad is None  # Even local routing is an explicit boundary.
        assert encoder_weight.grad is None

        expected_gradient = torch.zeros_like(source_output)
        for plan in plans:
            offset = 0
            for feature in plan.features:
                if feature.producer_rank == rank:
                    for consumer in plan.cp_ranks:
                        expected_gradient.narrow(0, feature.offset, feature.length).add_(
                            _cotangent(plan, consumer, base, round_index).narrow(
                                0, offset, feature.length
                            )
                        )
                offset += feature.length
        states.append(state)
        expected_gradients.append(expected_gradient)

    accumulated = torch.zeros_like(source_output)
    for state, expected_gradient in zip(states, expected_gradients):
        gradient = bridge.backward(state, gradient_dtype=torch.float64)
        torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=0)
        accumulated.add_(gradient)
        with pytest.raises(RuntimeError, match="only be returned once"):
            bridge.backward(state)
    source_output.backward(accumulated)
    torch.testing.assert_close(source_output.grad, accumulated, rtol=0, atol=0)
    torch.testing.assert_close(
        encoder_weight.grad, (source_input * accumulated).sum(), rtol=0, atol=0
    )


def test_feature_transport_adjoint_identity(bridge_groups):
    rank = dist.get_rank()
    base = rank // 4 * 4
    transport = bridge_groups[tuple(range(base, base + 4))]
    source = _source(rank, base, (5, 0, 3, 7))
    plans = _round_plans(base)[0]
    local_plan = next(plan for plan in plans if rank in plan.cp_ranks)
    bridge = PackFeatureBridge(transport)
    state = bridge.forward(source, plans, bridge_groups[local_plan.cp_ranks])
    cotangent = _cotangent(local_plan, rank, base, 0)
    source_gradient = bridge.backward(state, cotangent, gradient_dtype=torch.float64)
    products = torch.stack(
        [(state.features.detach() * cotangent).sum(), (source * source_gradient).sum()]
    )
    dist.all_reduce(products, group=transport)
    torch.testing.assert_close(products[0], products[1], rtol=0, atol=0)


def test_plan_validation(bridge_groups):
    rank = dist.get_rank()
    base = rank // 4 * 4
    ranks = tuple(range(base, base + 4))
    bridge = PackFeatureBridge(bridge_groups[ranks])
    source = _source(rank, base, (5, 0, 3, 7))
    with pytest.raises(ValueError, match="partition"):
        bridge.forward(source, [PackFeaturePlan((rank,), ())], None)
    with pytest.raises(ValueError, match="explicit CP"):
        bridge.forward(source, [PackFeaturePlan(ranks, ())], None)
    with pytest.raises(ValueError, match="group ranks"):
        bridge.forward(source, [PackFeaturePlan(ranks, ())], bridge_groups[(rank,)])
