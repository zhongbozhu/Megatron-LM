# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Exact global router objective across different token ownership and packing."""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from megatron.core.tensor_parallel.random import checkpoint
from megatron.core.transformer.moe.global_aux_loss import GlobalAuxLossStep
from megatron.core.transformer.moe.moe_utils import (
    MoEAuxLossAutoScaler,
    compute_routing_scores_for_aux_loss,
    switch_load_balancing_loss_func,
)
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig


@pytest.fixture(scope="module")
def router_groups():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    singleton = None
    for peer in range(dist.get_world_size()):
        group = dist.new_group([peer], backend="nccl")
        if rank == peer:
            singleton = group
    yield SimpleNamespace(tp=singleton, cp=singleton, tp_cp=singleton, tp_dp_cp=dist.group.WORLD)
    dist.destroy_process_group(singleton)


@pytest.mark.parametrize("partition", ("contiguous", "repacked", "idle_rank"))
def test_global_aux_matches_dense_objective_and_gradient(router_groups, partition):
    rank, world = dist.get_rank(), dist.get_world_size()
    if world < 2:
        pytest.skip("Distributed global auxiliary loss needs at least two ranks")
    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_router_load_balancing_type="global_aux_loss",
        moe_aux_loss_coeff=0.001,
        moe_router_dtype="fp32",
        calculate_per_token_loss=True,
    )
    torch.manual_seed(91)
    router = TopKRouter(config, pg_collection=router_groups).cuda()
    model = torch.nn.ModuleDict({"router": router})
    values = torch.randn(17, 1, 8, device="cuda")
    padding = torch.zeros(17, 1, device="cuda", dtype=torch.bool)
    padding[4] = True
    padding[15] = True

    reference_input = values.detach().clone().requires_grad_()
    logits = router.gating(reference_input).reshape(-1, config.num_moe_experts)
    routing, scores = compute_routing_scores_for_aux_loss(
        logits, router.topk, router.score_function, padding_mask=padding.reshape(-1)
    )
    reference_loss = switch_load_balancing_loss_func(
        probs=scores,
        tokens_per_expert=routing.sum(0),
        total_num_tokens=15,
        topk=router.topk,
        num_experts=router.num_experts,
        moe_aux_loss_coeff=0.001,
    )
    expected_weight, expected_input = torch.autograd.grad(
        reference_loss, (router.weight, reference_input)
    )

    inputs = values.detach().clone().requires_grad_()
    token_ids = torch.arange(17, device="cuda")
    if partition == "contiguous":
        owners = token_ids * world // len(token_ids)
        round_ids = torch.zeros_like(token_ids)
        num_rounds = 1
    elif partition == "repacked":
        owners = (token_ids * 3 + 1) % world
        round_ids = token_ids % 3
        num_rounds = 3
    else:
        owners = token_ids % (world - 1)
        round_ids = token_ids % 4
        num_rounds = 4

    with GlobalAuxLossStep(model, dist.group.WORLD) as auxiliary:
        for round_id in range(num_rounds):
            auxiliary.set_round(round_id)
            selected = token_ids[(owners == rank) & (round_ids == round_id)]
            if selected.numel():
                router(
                    inputs.index_select(0, selected), padding_mask=padding.index_select(0, selected)
                )
        metrics = auxiliary.finalize()
        assert metrics["layers"]["router"]["routed_tokens"] == 15
        torch.testing.assert_close(
            torch.tensor(metrics["loss"], device="cuda"), reference_loss, rtol=1e-6, atol=1e-10
        )
        for round_id in range(num_rounds):
            loss = auxiliary.loss_for_round(round_id)
            if loss.requires_grad:
                loss.backward()

    weight_gradient = (
        router.weight.grad if router.weight.grad is not None else torch.zeros_like(router.weight)
    )
    input_gradient = inputs.grad if inputs.grad is not None else torch.zeros_like(inputs)
    dist.all_reduce(weight_gradient)
    dist.all_reduce(input_gradient)
    torch.testing.assert_close(weight_gradient, expected_weight, rtol=1e-5, atol=1e-9)
    torch.testing.assert_close(input_gradient, expected_input, rtol=1e-5, atol=1e-9)
    assert not hasattr(router, "_global_aux_loss_step")
    # The opt-in context does not update the legacy per-microbatch running average.
    assert router.ga_steps.item() == 0
    assert torch.count_nonzero(router.global_tokens_per_expert).item() == 0


def test_global_aux_rejects_shared_router_across_mtp_depths(router_groups):
    """A shared router needs per-depth statistics, not one pooled router loss."""
    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        num_moe_experts=4,
        moe_router_load_balancing_type="global_aux_loss",
        moe_aux_loss_coeff=0.001,
        mtp_num_layers=2,
        mtp_use_repeated_layer=True,
    )
    router = TopKRouter(config, pg_collection=router_groups, is_mtp_layer=True).cuda()
    model = torch.nn.ModuleDict({"router": router})
    with pytest.raises(ValueError, match="separate per-depth"):
        with GlobalAuxLossStep(model, dist.group.WORLD):
            pytest.fail("Shared multi-depth MTP must be rejected before collecting statistics")
    assert not hasattr(router, "_global_aux_loss_step")


@pytest.mark.parametrize("recompute", (None, "selective", "full"))
@pytest.mark.parametrize("partition", ("repacked", "idle_rank"))
def test_two_pass_native_aux_matches_global_objective(router_groups, recompute, partition):
    """Sequential F/B and native checkpoint replay preserve the full-step objective."""
    rank, world = dist.get_rank(), dist.get_world_size()
    if world < 2:
        pytest.skip("Distributed global auxiliary loss needs at least two ranks")
    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        num_moe_experts=world * 2,
        expert_model_parallel_size=world,
        moe_router_topk=2,
        moe_router_load_balancing_type="global_aux_loss",
        moe_aux_loss_coeff=0.001,
        # Different pack GEMM shapes can select different TF32 algorithms. Test
        # normalization/replay independently of that kernel precision variation.
        moe_router_dtype="fp64",
        calculate_per_token_loss=True,
        recompute_granularity=recompute,
        recompute_method="uniform" if recompute == "full" else None,
        recompute_num_layers=1 if recompute == "full" else None,
    )
    torch.manual_seed(137)
    router = TopKRouter(config, pg_collection=router_groups).cuda().double()
    model = torch.nn.ModuleDict({"router": router})
    values = torch.randn(31, 1, 8, device="cuda", dtype=torch.float64)
    padding = torch.zeros(31, 1, device="cuda", dtype=torch.bool)
    padding[[4, 16, 28]] = True

    reference_input = values.detach().clone().requires_grad_()
    logits = router.gating(reference_input).reshape(-1, router.num_experts)
    routing, scores = compute_routing_scores_for_aux_loss(
        logits, router.topk, router.score_function, padding_mask=padding.reshape(-1)
    )
    reference_loss = switch_load_balancing_loss_func(
        probs=scores,
        tokens_per_expert=routing.sum(0),
        total_num_tokens=28,
        topk=router.topk,
        num_experts=router.num_experts,
        moe_aux_loss_coeff=0.001,
    )
    expected_weight, expected_input = torch.autograd.grad(
        reference_loss, (router.weight, reference_input)
    )

    token_ids = torch.arange(len(values), device="cuda")
    active_ranks = world - 1 if partition == "idle_rank" else world
    owners = (token_ids * 3 + 1) % active_ranks
    round_ids = token_ids % 4
    inputs = values.detach().clone().requires_grad_()
    supervised_tokens = torch.tensor(73, device="cuda")
    loss_scale = torch.tensor(2.5, device="cuda")
    previous_scale = MoEAuxLossAutoScaler.main_loss_backward_scale
    previous_scale = previous_scale.clone() if previous_scale is not None else None
    MoEAuxLossAutoScaler.set_loss_scale(loss_scale)
    try:
        with GlobalAuxLossStep(model, dist.group.WORLD, two_pass=True) as auxiliary:
            with pytest.raises(RuntimeError, match="after finalizing"):
                auxiliary.begin_training(supervised_tokens)
            with torch.no_grad():
                for round_id in range(4):
                    auxiliary.set_round(round_id)
                    selected = token_ids[(owners == rank) & (round_ids == round_id)]
                    if selected.numel():
                        router(
                            values.index_select(0, selected),
                            padding_mask=padding.index_select(0, selected),
                        )
            for records in auxiliary._records.values():
                assert all(not scores.requires_grad for scores, _ in records.values())
            metrics = auxiliary.finalize()
            assert metrics["layers"]["router"]["routed_tokens"] == 28
            torch.testing.assert_close(
                torch.tensor(metrics["loss"], device="cuda"), reference_loss, rtol=1e-6, atol=1e-10
            )
            auxiliary.begin_training(supervised_tokens)
            assert not auxiliary._records
            fixed_counts = auxiliary._global_counts[router][0].clone()
            # All communication is finished before the native sequential F/B pass.
            # The router and its checkpoint replay must neither reduce nor recount.
            with patch.object(dist, "all_reduce", side_effect=AssertionError("unexpected reduce")):
                for round_id in range(4):
                    auxiliary.set_round(round_id)
                    selected = token_ids[(owners == rank) & (round_ids == round_id)]
                    if not selected.numel():
                        continue
                    selected_inputs = inputs.index_select(0, selected)
                    selected_padding = padding.index_select(0, selected)

                    def forward(values, mask):
                        return router(values, padding_mask=mask)[0]

                    if recompute is None:
                        output = forward(selected_inputs, selected_padding)
                    else:
                        output = checkpoint(forward, False, selected_inputs, selected_padding)
                    # A zero main objective isolates the attached router gradient.
                    (output.sum() * 0).backward()
            torch.testing.assert_close(auxiliary._global_counts[router][0], fixed_counts)
            assert not auxiliary._records
    finally:
        MoEAuxLossAutoScaler.main_loss_backward_scale = previous_scale

    weight_gradient = (
        router.weight.grad if router.weight.grad is not None else torch.zeros_like(router.weight)
    )
    input_gradient = inputs.grad if inputs.grad is not None else torch.zeros_like(inputs)
    dist.all_reduce(weight_gradient)
    dist.all_reduce(input_gradient)
    # Native BF16 training uses scale 1, but a non-unit test scale verifies that
    # the native autograd scaler and final token normalization each apply once.
    weight_gradient /= supervised_tokens * loss_scale
    input_gradient /= supervised_tokens * loss_scale
    torch.testing.assert_close(weight_gradient, expected_weight, rtol=1e-5, atol=1e-9)
    torch.testing.assert_close(input_gradient, expected_input, rtol=1e-5, atol=1e-9)
    assert not hasattr(router, "_global_aux_loss_step")
    assert router.ga_steps.item() == 0
