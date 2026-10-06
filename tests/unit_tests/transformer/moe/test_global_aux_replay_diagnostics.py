# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CPU/Gloo audits of detached router replay diagnostics, including recompute."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from megatron.core.transformer.moe.global_aux_loss import GlobalAuxLossStep
from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig


@pytest.fixture(scope="module", autouse=True)
def initialize_gloo():
    """Also allow a CPU torchrun with --confcutdir pointing at this directory."""
    initialized_here = not dist.is_initialized()
    if initialized_here:
        dist.init_process_group(backend="gloo")
    yield
    if initialized_here:
        dist.destroy_process_group()


def _run_replay(
    group, *, audit=True, recompute=None, perturb=None, omit=False, token_assignments=False
):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=4,
        num_attention_heads=1,
        num_moe_experts=3,
        moe_router_topk=1,
        moe_router_pre_softmax=True,
        moe_router_load_balancing_type="global_aux_loss",
        moe_aux_loss_coeff=0.01,
        calculate_per_token_loss=True,
    )
    singleton = SimpleNamespace(size=lambda: 1)
    groups = SimpleNamespace(tp=singleton, cp=singleton, tp_cp=singleton, tp_dp_cp=group)
    # The diagnostic consumes router scores; it does not execute the GPU-only gate.
    with patch.object(torch.cuda, "current_device", return_value="cpu"):
        router = TopKRouter(config, pg_collection=groups)
    model = torch.nn.ModuleDict({"router": router})
    reference = torch.tensor([[3.0, 1.0, 0.0], [1.0, 3.0, 0.0]], dtype=torch.float32)
    inputs = [reference.clone().requires_grad_() for _ in range(2)]
    saved_scale = MoEAuxLossAutoScaler.main_loss_backward_scale
    MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(1.0)
    try:
        with GlobalAuxLossStep(
            model,
            group,
            two_pass=True,
            audit_replay=audit,
            audit_token_assignments=token_assignments,
        ) as step:
            training_calls = {}

            def forward(logits):
                if step._training_pass:
                    call = training_calls.get(step._round, 0)
                    training_calls[step._round] = call + 1
                    if perturb == "recompute_only" and call:
                        logits = logits.flip(0)
                # Exercise the production router's no-grad checkpoint audit gate,
                # actual dispatch observation, and independent auxiliary top-k map.
                return router.routing(logits[:, None, :])[0]

            with torch.no_grad():
                for round_id in range(2):
                    step.set_round(round_id)
                    forward(reference)
            step.finalize()
            step.begin_training(torch.tensor(11))
            assert bool(step._records) is audit
            with patch.object(
                dist, "all_reduce", side_effect=AssertionError("training collective")
            ):
                for round_id, logits in enumerate(inputs):
                    step.set_round(round_id)
                    if omit and round_id == 1 and dist.get_rank(group) == 0:
                        continue
                    rank = dist.get_rank(group)
                    if perturb == "round_cancel" and rank == 0:
                        # Opposite histogram changes cancel across rounds; the audit must
                        # still detect both local round mismatches before reducing metrics.
                        with torch.no_grad():
                            logits[round_id, round_id] = -4.0
                    if perturb == "rank_cancel" and rank < 2 and round_id == 0:
                        with torch.no_grad():
                            logits[rank, rank] = -4.0
                    if perturb == "global_change" and rank == 0 and round_id == 0:
                        with torch.no_grad():
                            logits[0, 0] = -4.0
                    if perturb == "token_swap" and rank == 0:
                        logits = logits.flip(0)
                    if recompute:
                        with set_checkpoint_early_stop(False):
                            output = checkpoint(
                                forward, logits, use_reentrant=recompute == "reentrant"
                            )
                    else:
                        output = forward(logits)
                    (output.square().sum()).backward()
            if audit:
                for records in step._replay_records.values():
                    assert all(not scores.requires_grad for scores, _ in records.values())
                metrics = step.replay_metrics()
            else:
                assert not step._records and not step._replay_records
                with pytest.raises(RuntimeError, match="diagnostic training pass"):
                    step.replay_metrics()
                metrics = None
            return [value.grad for value in inputs], metrics
    finally:
        MoEAuxLossAutoScaler.main_loss_backward_scale = saved_scale


@pytest.mark.parametrize("recompute", [None, "reentrant", "nonreentrant"])
def test_replay_audit_preserves_gradients_and_handles_checkpoint(recompute):
    group = dist.new_group(backend="gloo")
    try:
        plain_gradients, _ = _run_replay(group, audit=False, recompute=recompute)
        audited_gradients, metrics = _run_replay(group, recompute=recompute, token_assignments=True)
        for plain, audited in zip(plain_gradients, audited_gradients):
            torch.testing.assert_close(plain, audited, rtol=0, atol=0)
        assert metrics["pairs"] == 2 * dist.get_world_size(group)
        assert metrics["missing"] == metrics["unexpected"] == 0
        assert metrics["histogram_mismatches"] == 0
        assert metrics["global_histogram_mismatches"] == 0
        assert metrics["score_sum_max_abs"] == 0
        assert metrics["recompute"]["pairs"] == (2 * dist.get_world_size(group) if recompute else 0)
        assert metrics["recompute"]["histogram_mismatches"] == 0
        assert metrics["first_mismatches"] == []
        for kind in ("auxiliary", "dispatch"):
            assert metrics["token_assignments"][f"{kind}_training"]["different_rows"] == 0
            assert metrics["token_assignments"][f"{kind}_recompute"]["different_rows"] == 0
    finally:
        dist.destroy_process_group(group)


def test_replay_audit_detects_per_round_differences_and_missing_replay():
    group = dist.new_group(backend="gloo")
    try:
        _, metrics = _run_replay(group, perturb="round_cancel", recompute="reentrant")
        assert metrics["histogram_mismatches"] == 2
        assert metrics["histogram_l1"] == 4
        assert metrics["score_sum_max_abs"] > 0
        assert metrics["layers"]["router"]["score_sum_relative_l2"] > 0
        assert metrics["global_histogram_mismatches"] == 0
        assert metrics["global_histogram_l1"] == 0
        assert metrics["recompute"]["histogram_mismatches"] == 0
        assert {item["round"] for item in metrics["first_mismatches"]} == {0, 1}
        _, missing = _run_replay(group, omit=True)
        assert missing["missing"] == 1 and missing["unexpected"] == 0
        assert missing["histogram_mismatches"] == 1
        assert missing["global_histogram_mismatches"] == 1
    finally:
        dist.destroy_process_group(group)


def test_global_counts_distinguish_cross_rank_cancellation_from_changed_objective():
    group = dist.new_group(backend="gloo")
    try:
        if dist.get_world_size(group) < 2:
            pytest.skip("Cross-rank histogram cancellation requires two Gloo ranks")
        _, cancelled = _run_replay(group, perturb="rank_cancel")
        assert cancelled["histogram_mismatches"] == 2
        assert cancelled["histogram_l1"] == 4
        assert cancelled["global_histogram_mismatches"] == 0
        assert cancelled["global_histogram_l1"] == 0
        _, changed = _run_replay(group, perturb="global_change")
        assert changed["global_histogram_mismatches"] == 1
        assert changed["global_histogram_l1"] == 2
        layer = changed["layers"]["router"]
        assert layer["C_training"][0] == layer["C_stats"][0] - 1
        assert layer["C_training"][1] == layer["C_stats"][1] + 1
        assert changed["first_mismatches"][0]["rank"] == 0
        assert changed["first_mismatches"][0]["round"] == 0
    finally:
        dist.destroy_process_group(group)


def test_exact_assignments_detect_swapped_tokens_with_equal_histograms():
    group = dist.new_group(backend="gloo")
    try:
        _, metrics = _run_replay(group, perturb="token_swap", token_assignments=True)
        assert metrics["histogram_mismatches"] == 0
        assert metrics["global_histogram_mismatches"] == 0
        assert metrics["score_sum_max_abs"] == 0
        for kind in ("auxiliary", "dispatch"):
            assert metrics["token_assignments"][f"{kind}_training"]["different_rows"] == 4
        example = metrics["first_mismatches"][0]
        assert example["rank"] == 0 and example["router"] == "router"
        assert example["examples"] == [
            {"row": 0, "expected": [0], "actual": [1]},
            {"row": 1, "expected": [1], "actual": [0]},
        ]
    finally:
        dist.destroy_process_group(group)


@pytest.mark.parametrize("recompute", ["reentrant", "nonreentrant"])
def test_recompute_is_compared_separately_without_double_counting(recompute):
    group = dist.new_group(backend="gloo")
    try:
        _, metrics = _run_replay(
            group, recompute=recompute, perturb="recompute_only", token_assignments=True
        )
        assert metrics["histogram_mismatches"] == 0
        assert metrics["global_histogram_mismatches"] == 0
        expected_pairs = 2 * dist.get_world_size(group)
        assert metrics["pairs"] == metrics["recompute"]["pairs"] == expected_pairs
        for kind in ("auxiliary", "dispatch"):
            assert metrics["token_assignments"][f"{kind}_training"]["different_rows"] == 0
            assert metrics["token_assignments"][f"{kind}_recompute"]["different_rows"] == (
                2 * expected_pairs
            )
        assert {item["phase"] for item in metrics["first_mismatches"]} == {"recompute"}
    finally:
        dist.destroy_process_group(group)


def test_assignment_bitsets_preserve_padding_and_partial_last_byte():
    routing = torch.tensor([[True] * 9, [False] * 9, [False] * 8 + [True]], dtype=torch.bool)
    packed = GlobalAuxLossStep._pack_assignments(routing)
    assert packed.dtype == torch.uint8 and packed.device.type == "cpu"
    assert packed.tolist() == [[255, 1], [0, 0], [0, 1]]
    assert GlobalAuxLossStep._pack_assignments(routing[:0]).shape == (0, 2)
