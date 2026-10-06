# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Independent objective oracle for the native two-pass global MoE auxiliary loss.

Run with four GPU processes. CP layouts here define token ownership; this test
does not execute attention, expert dispatch, the DCP scheduler, or native DDP.
The production router, padding handling, statistics reduction, checkpoint replay,
auxiliary attachment and scaler are exercised without mocking their arithmetic.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core.tensor_parallel.random import checkpoint
from megatron.core.transformer.moe.global_aux_loss import GlobalAuxLossStep
from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig


@pytest.fixture(scope="module")
def objective_router_groups():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    assert dist.get_world_size() == 4, "This ownership oracle requires exactly four ranks"
    singleton = None
    for rank in range(4):
        group = dist.new_group([rank], backend="nccl")
        if rank == dist.get_rank():
            singleton = group
    yield SimpleNamespace(tp=singleton, cp=singleton, tp_cp=singleton, tp_dp_cp=dist.group.WORLD)
    dist.destroy_process_group(singleton)


class _FixedIDs:
    """A minimal controller supplies canonical choices, never fixed probabilities."""

    indices = None

    def select(self, router, logits, default_selector, padding_mask, packed_seq_params):
        assert self.indices.shape == (logits.shape[0], router.topk)
        return self.indices


def _source():
    """Five variable-length samples, with independently identified semantic rows."""
    lengths = (3, 5, 7, 10, 2)
    padded_lengths = (8, 8, 16, 16, 8)
    sample_rows, valid, mtp_valid, supervised, kinds = [], [], [], [], []
    offset = 0
    for length, padded in zip(lengths, padded_lengths):
        sample_rows.append(list(range(offset, offset + padded)))
        for position in range(padded):
            real = position < length
            valid.append(real)
            # One-depth MTP excludes the final real row, regardless of supervision.
            mtp_valid.append(position + 1 < length)
            supervised.append(real and position >= 3)
            kinds.append(
                "padding"
                if not real
                else "prompt" if position == 0 else "vision" if position < 3 else "answer"
            )
        offset += padded
    ids = torch.arange(offset)
    logits = torch.stack(
        [torch.sin(ids.double() * 0.17 + expert * 0.61) for expert in range(4)], dim=-1
    )
    return SimpleNamespace(
        rows=sample_rows,
        logits=logits,
        valid=torch.tensor(valid),
        mtp_valid=torch.tensor(mtp_valid),
        supervised=torch.tensor(supervised),
        kinds=kinds,
    )


def _ownership(source, layout):
    """Independent zigzag owner table, including empty rank/round combinations."""
    if layout == "cp1":
        placements = [(sample // 4, [sample % 4]) for sample in range(5)]
    elif layout == "static_cp2":
        placements = [
            (sample // 2, [2 * (sample % 2), 2 * (sample % 2) + 1]) for sample in range(5)
        ]
    else:
        assert layout == "mixed_cp"
        # Sample 3 uses CP4; round 1 concurrently uses CP2, CP1, CP1.
        placements = [(1, [3]), (1, [2]), (1, [0, 1]), (0, [0, 1, 2, 3]), (2, [0])]
    owners = torch.full((len(source.logits),), -1, dtype=torch.long)
    rounds = torch.full_like(owners, -1)
    cp_sizes = []
    for rows, (round_id, ranks) in zip(source.rows, placements):
        cp_size = len(ranks)
        cp_sizes.append(cp_size)
        assert len(rows) % (2 * cp_size) == 0
        chunk = len(rows) // (2 * cp_size)
        for cp_rank, rank in enumerate(ranks):
            indices = rows[cp_rank * chunk : (cp_rank + 1) * chunk]
            mirror = 2 * cp_size - cp_rank - 1
            indices += rows[mirror * chunk : (mirror + 1) * chunk]
            assert (owners[indices] == -1).all()
            owners[indices], rounds[indices] = rank, round_id
    assert (owners >= 0).all() and (rounds >= 0).all()
    return owners, rounds, cp_sizes


def _oracle(logits, ids, valid, coefficient):
    """CPU FP64 formula and analytic derivative, independent of production helpers.

    L = alpha E / (k N**2) sum_e C_e sum_(t valid) softmax(z_t)_e.
    dL/dz_te = alpha E / (k N**2) p_te (C_e - sum_j p_tj C_j).
    Expert counts are fixed; prompt and vision rows participate like answer rows.
    """
    assert logits.device.type == "cpu" and logits.dtype == torch.float64
    num_tokens, experts, topk = int(valid.sum()), logits.shape[1], ids.shape[1]
    counts = torch.bincount(ids[valid].flatten(), minlength=experts).double()
    exponentials = (logits - logits.max(dim=-1, keepdim=True).values).exp()
    probabilities = exponentials / exponentials.sum(dim=-1, keepdim=True)
    factor = coefficient * experts / (topk * num_tokens**2)
    weighted_mean = (probabilities * counts).sum(-1, keepdim=True)
    gradient = factor * probabilities * (counts - weighted_mean) * valid[:, None]
    loss = factor * (probabilities[valid].sum(0) * counts).sum()
    return loss, gradient, counts


@pytest.mark.parametrize("layout", ("cp1", "static_cp2", "mixed_cp"))
@pytest.mark.parametrize("recompute", (False, True))
def test_global_aux_objective_oracle(objective_router_groups, layout, recompute):
    source = _source()
    owners, rounds, cp_sizes = _ownership(source, layout)
    supervised_tokens = int(source.supervised.sum())
    assert supervised_tokens == 13
    assert int(source.valid.sum()) == 27 and int(source.mtp_valid.sum()) == 22
    rank = dist.get_rank()
    routers, values, expert_ids, expected = {}, {}, {}, {}
    for index, name in enumerate(("decoder", "mtp")):
        coefficient = (0.037, 0.019)[index]
        config = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=2,
            num_moe_experts=4,
            moe_router_topk=2,
            moe_router_score_function="softmax",
            moe_router_load_balancing_type="global_aux_loss",
            moe_aux_loss_coeff=coefficient,
            moe_router_dtype="fp64",
            calculate_per_token_loss=True,
            mtp_num_layers=1,
            recompute_granularity="full" if recompute else None,
            recompute_method="uniform" if recompute else None,
            recompute_num_layers=1 if recompute else None,
        )
        router = (
            TopKRouter(config, pg_collection=objective_router_groups, is_mtp_layer=name == "mtp")
            .cuda()
            .double()
        )
        router._fixed_routing = _FixedIDs()
        routers[name] = router
        row_ids = torch.arange(len(source.logits))
        second = 1 + ((row_ids + index) % 5 > 1) + ((row_ids + index) % 5 > 3)
        ids = torch.stack((torch.zeros_like(row_ids), second), -1)
        expert_ids[name] = ids.cuda()
        logits = source.logits + index * torch.tensor([0.1, -0.3, 0.2, 0.4])
        values[name] = logits.cuda().requires_grad_()
        valid = source.valid if name == "decoder" else source.mtp_valid
        expected[name] = _oracle(logits, ids, valid, coefficient)
        # This construction detects replacing router validity by the LM loss mask.
        wrong_loss, wrong_gradient, _ = _oracle(logits, ids, source.supervised, coefficient)
        assert abs(wrong_loss - expected[name][0]) > 1e-5
        assert torch.linalg.vector_norm(wrong_gradient - expected[name][1]) > 1e-4
        # Replacing routed N by the unrelated supervised N changes the N**2 factor.
        denominator_mutant = expected[name][1] * (int(valid.sum()) / supervised_tokens) ** 2
        assert not torch.allclose(denominator_mutant, expected[name][1], rtol=1e-3, atol=1e-8)
    model = torch.nn.ModuleDict(routers)
    old_scale = MoEAuxLossAutoScaler.main_loss_backward_scale
    scale = torch.tensor(2.5, device="cuda")
    MoEAuxLossAutoScaler.set_loss_scale(scale)
    try:
        with GlobalAuxLossStep(model, dist.group.WORLD, two_pass=True) as auxiliary:
            with torch.no_grad():
                for round_id in range(int(rounds.max()) + 1):
                    auxiliary.set_round(round_id)
                    selected = ((owners == rank) & (rounds == round_id)).nonzero().flatten().cuda()
                    if not selected.numel():
                        continue
                    for name, router in routers.items():
                        valid = source.valid if name == "decoder" else source.mtp_valid
                        router._fixed_routing.indices = expert_ids[name].index_select(0, selected)
                        router.routing(
                            values[name].index_select(0, selected)[:, None, :],
                            padding_mask=(~valid).cuda().index_select(0, selected),
                        )
            metrics = auxiliary.finalize()
            for name, (loss, _, counts) in expected.items():
                result = metrics["layers"][name]
                assert result["routed_tokens"] == (27 if name == "decoder" else 22)
                assert result["tokens_per_expert"] == counts.tolist()
                torch.testing.assert_close(
                    torch.tensor(result["loss"], dtype=torch.float64), loss, rtol=2e-6, atol=1e-9
                )
            assert abs(metrics["loss"] - sum(item[0].item() for item in expected.values())) < 1e-7
            assert all(value.grad is None for value in values.values())
            auxiliary.begin_training(torch.tensor(supervised_tokens, device="cuda"))
            for round_id in range(int(rounds.max()) + 1):
                auxiliary.set_round(round_id)
                selected = ((owners == rank) & (rounds == round_id)).nonzero().flatten().cuda()
                if not selected.numel():
                    continue
                for name, router in routers.items():
                    valid = source.valid if name == "decoder" else source.mtp_valid
                    router._fixed_routing.indices = expert_ids[name].index_select(0, selected)
                    local_logits = values[name].index_select(0, selected)[:, None, :]
                    local_padding = (~valid).cuda().index_select(0, selected)

                    def forward(logits, padding, router=router):
                        return router.routing(logits, padding_mask=padding)[0]

                    output = (
                        checkpoint(forward, False, local_logits, local_padding)
                        if recompute
                        else forward(local_logits, local_padding)
                    )
                    # Zero main objective isolates the production auxiliary attachment.
                    (output.sum() * 0).backward()
    finally:
        MoEAuxLossAutoScaler.main_loss_backward_scale = old_scale

    report = {
        "layout": layout,
        "recompute": recompute,
        "cp_sizes": cp_sizes,
        "supervised_tokens": supervised_tokens,
        "routers": {},
    }
    for name, value in values.items():
        gradient = value.grad if value.grad is not None else torch.zeros_like(value)
        dist.all_reduce(gradient)
        # Token ownership SUM plus known finalization factors. Native DDP is tested separately.
        actual = (gradient / (supervised_tokens * scale)).cpu()
        loss, reference, counts = expected[name]
        torch.testing.assert_close(actual, reference, rtol=3e-6, atol=2e-9)
        valid = source.valid if name == "decoder" else source.mtp_valid
        assert torch.count_nonzero(actual[~valid]) == 0
        for kind in ("prompt", "vision", "answer"):
            selected = torch.tensor([value == kind for value in source.kinds]) & valid
            assert selected.any() and torch.linalg.vector_norm(actual[selected]) > 0
        # Fixed counts must not freeze probabilities, including unselected expert logits.
        assert torch.count_nonzero(actual[valid, 3]) > 0
        assert not hasattr(routers[name], "_global_aux_loss_step")
        report["routers"][name] = {
            "routed_tokens": int(valid.sum()),
            "tokens_per_expert": counts.tolist(),
            "actual_loss": metrics["layers"][name]["loss"],
            "oracle_loss": loss.item(),
            "gradient_max_abs_error": (actual - reference).abs().max().item(),
            "gradient_relative_l2": (
                torch.linalg.vector_norm(actual - reference) / torch.linalg.vector_norm(reference)
            ).item(),
            "padding_gradient_nonzero": int(torch.count_nonzero(actual[~valid])),
        }
    if rank == 0:
        print("GLOBAL_AUX_OBJECTIVE_ORACLE " + json.dumps(report, sort_keys=True))
        output_directory = os.environ.get("MIMO_OBJECTIVE_ORACLE_REPORT_DIR")
        if output_directory:
            path = Path(output_directory) / f"global_aux_{layout}_recompute{int(recompute)}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
