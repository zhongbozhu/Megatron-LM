# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MTP backward must be invariant to the CP partition of a logical microbatch."""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import multi_token_prediction as mtp


@pytest.fixture(autouse=True)
def loss_scale(monkeypatch):
    """Do not leak the pipeline's class-level backward scale between tests."""
    monkeypatch.setattr(mtp.MTPLossAutoScaler, "main_loss_backward_scale", torch.tensor(1.0))


def _config(num_layers=1, per_token=True):
    return SimpleNamespace(
        mtp_num_layers=num_layers,
        mtp_loss_scaling_factor=0.3,
        calculate_per_token_loss=per_token,
        mtp_detach_heads=False,
        context_parallel_size=1,
    )


@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("cp_size", [None, 1, 2])
@pytest.mark.parametrize("runtime_metadata", [False, True])
def test_normalization_uses_runtime_counts_without_changing_logging(
    monkeypatch, per_token, cp_size, runtime_metadata
):
    """Reduce both counts, use the runtime group, and retain local logging counts."""
    static_group = SimpleNamespace(size=lambda: 1)
    runtime_group = SimpleNamespace(size=lambda: cp_size) if cp_size else None
    params = (
        PackedSeqParams(local_cp_size=cp_size, cp_group=runtime_group)
        if cp_size and runtime_metadata
        else None
    )
    reductions = []
    logged = []

    def roll(tensor, return_sum=True, cp_group=None, **kwargs):
        assert cp_group is runtime_group
        if not return_sum:
            return tensor, None
        mask = tensor.new_tensor([[1, 1, 1, 0]])
        return mask, mask.sum()

    def reduce(counts, op=None, group=None):
        assert cp_size == 2 and group is runtime_group
        assert op == torch.distributed.ReduceOp.SUM
        torch.testing.assert_close(counts, torch.tensor([4.0, 3.0]))
        counts.add_(counts.new_tensor([4, 4]))
        reductions.append(group)

    monkeypatch.setattr(mtp, "roll_tensor", roll)
    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    monkeypatch.setattr(
        mtp.MTPLossLoggingHelper,
        "save_metrics_to_tracker",
        lambda *args, **kwargs: logged.append((args[0].detach(), kwargs["num_tokens"])),
    )
    hidden = torch.ones(8, 1, 1, requires_grad=True)
    output = mtp.process_mtp_loss(
        hidden_states=hidden,
        labels=torch.zeros(1, 4, dtype=torch.long),
        loss_mask=torch.ones(1, 4),
        output_layer=lambda value, **kwargs: (value, None),
        output_weight=None,
        runtime_gather_output=True,
        is_training=True,
        compute_language_model_loss=lambda labels, logits: logits.squeeze(-1).transpose(0, 1),
        config=_config(per_token=per_token),
        cp_group=static_group if params else runtime_group,
        packed_seq_params=params,
        metric_avg_group=object(),
    )
    (output.sum() * 0).backward()
    assert len(reductions) == int(per_token and cp_size == 2)
    if per_token:
        main_tokens, mtp_tokens = (8, 7) if cp_size == 2 else (4, 3)
        torch.testing.assert_close(
            hidden.grad[4:, 0, 0] / main_tokens, torch.tensor([0.3 / mtp_tokens] * 3 + [0.0])
        )
        assert logged[0][0].item() == 3
        assert logged[0][1].item() == 3
    else:
        torch.testing.assert_close(hidden.grad[4:, 0, 0], torch.tensor([0.1] * 3 + [0.0]))
        assert logged[0][0].item() == 1
        assert logged[0][1] is None


@pytest.fixture(scope="module")
def cp_groups():
    """Create CP1/2/4 groups; the normal unit-test runner uses NCCL on GPUs."""
    if int(os.environ.get("WORLD_SIZE", "1")) < 4:
        pytest.skip("CP1/2/4 gradient parity requires at least four distributed ranks")
    if not torch.distributed.is_initialized():
        from tests.unit_tests.test_utilities import Utils

        Utils.initialize_distributed()
    world_size = torch.distributed.get_world_size()
    assert world_size % 4 == 0
    rank = torch.distributed.get_rank()
    local_groups = {}
    for cp_size in (1, 2, 4):
        for first_rank in range(0, world_size, cp_size):
            ranks = list(range(first_rank, first_rank + cp_size))
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                local_groups[cp_size] = group
    yield local_groups
    torch.distributed.barrier()
    for group in reversed(list(local_groups.values())):
        torch.distributed.destroy_process_group(group)


def _masks(case, device):
    mask = torch.ones(1, 16, device=device)
    input_mask = None
    if case == "uneven":
        mask[0] = mask.new_tensor([0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1])
    elif case == "empty_rank":
        # CP2 rank 0 has no main tokens, but rolling brings in valid MTP targets.
        mask[0] = mask.new_tensor([0, 0, 1, 1, 1, 1, 0, 0] * 2)
    elif case == "zero_mtp":
        mask.zero_()
        mask[0, [0, 8]] = 1
    elif case == "all_zero":
        mask.zero_()
    elif case == "input_holes":
        input_mask = torch.tensor([[1, 1, 0, 1, 1, 0, 1, 1] * 2], device=device).bool()
    elif case == "zero_input":
        input_mask = torch.zeros_like(mask, dtype=torch.bool)
    return mask, input_mask


def _backward(cp_group, static_group, mask, input_mask, num_layers, derive_labels):
    """Use real packed rolling, cross entropy, and trainable hidden/output parameters."""
    device = mask.device
    cp_size = cp_group.size()
    cp_rank = cp_group.rank()
    chunks = torch.arange(16, device=device).reshape(2, 2 * cp_size, -1)
    indices = chunks[:, [cp_rank, 2 * cp_size - cp_rank - 1]].reshape(-1)
    cu_seqlens = torch.tensor([0, 8, 16], dtype=torch.int32, device=device)
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        local_cp_size=cp_size,
        cp_group=cp_group,
    )
    head = torch.nn.Parameter(torch.arange(20, device=device).reshape(5, 4).float() / 20)
    hidden = torch.nn.Parameter(
        torch.arange((num_layers + 1) * 16 * 4, device=device)
        .reshape(num_layers + 1, 16, 1, 4)
        .float()
        .remainder(19)
        / 19
    )
    token_ids = torch.arange(16, device=device).remainder(5).unsqueeze(0)
    local_hidden = hidden[:, indices].reshape(-1, 1, 4)
    result = mtp.process_mtp_loss(
        hidden_states=local_hidden,
        labels=None if derive_labels else token_ids[:, indices],
        input_ids=token_ids[:, indices] if derive_labels else None,
        loss_mask=mask[:, indices],
        mtp_input_mask=input_mask[:, indices] if input_mask is not None else None,
        output_layer=lambda value, **kwargs: (F.linear(value, head), None),
        output_weight=None,
        runtime_gather_output=True,
        is_training=False,
        compute_language_model_loss=lambda labels, logits: F.cross_entropy(
            logits.transpose(0, 1).reshape(-1, 5), labels.reshape(-1), reduction="none"
        ).reshape_as(labels),
        config=_config(num_layers=num_layers),
        # Deliberately pass the build-time CP1 group: packed runtime metadata wins.
        cp_group=static_group,
        packed_seq_params=params,
    )
    (result.sum() * 0).backward()

    main_mask = mask
    if derive_labels:
        main_mask, _ = mtp.roll_tensor(
            mask, packed_seq_params=PackedSeqParams(cu_seqlens_q=cu_seqlens)
        )
    num_tokens = main_mask[:, indices].sum()
    grads = [head.grad, hidden.grad]
    # Match per-token DDP SUM followed by finalize_model_grads' main-token divisor.
    for value in [*grads, num_tokens]:
        torch.distributed.all_reduce(value, group=cp_group)
    return [grad / num_tokens.clamp(min=1) for grad in grads]


@pytest.mark.parametrize(
    "case",
    ["all_valid", "uneven", "empty_rank", "zero_mtp", "all_zero", "input_holes", "zero_input"],
)
@pytest.mark.parametrize("num_layers", [1, 3])
@pytest.mark.parametrize("derive_labels", [False, True])
def test_cp2_cp4_parameter_gradients_match_cp1(cp_groups, case, num_layers, derive_labels):
    """Changing runtime CP size preserves parameter gradients, including empty ranks."""
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    mask, input_mask = _masks(case, device)
    reference = _backward(cp_groups[1], cp_groups[1], mask, input_mask, num_layers, derive_labels)
    if case in ("zero_mtp", "all_zero", "zero_input"):
        for grad in reference:
            assert torch.count_nonzero(grad) == 0
    for cp_size in (2, 4):
        actual = _backward(
            cp_groups[cp_size], cp_groups[1], mask, input_mask, num_layers, derive_labels
        )
        for expected_grad, actual_grad in zip(reference, actual):
            assert torch.isfinite(actual_grad).all()
            torch.testing.assert_close(actual_grad, expected_grad, atol=2e-7, rtol=2e-5)


def _repacking_fixture(device, input_mask_case):
    logical_lengths = (5, 9, 7)
    physical_lengths = (8, 16, 8)
    masks = []
    conditioning = []
    for sample, (logical, physical) in enumerate(zip(logical_lengths, physical_lengths)):
        mask = torch.zeros(1, physical, device=device)
        mask[0, (0, 6, 1)[sample] : logical - 1] = 1
        input_mask = torch.ones_like(mask, dtype=torch.bool)
        input_mask[0, logical:] = False
        if input_mask_case == "holes":
            input_mask[0, (1, 7, 4)[sample]] = False
        elif input_mask_case == "zero":
            input_mask.zero_()
        masks.append(mask)
        conditioning.append(input_mask)
    return logical_lengths, physical_lengths, masks, conditioning


@pytest.mark.parametrize("input_mask_case", ["none", "holes", "zero"])
@pytest.mark.parametrize("num_layers", [1, 3])
def test_step_global_mtp_counts_match_independent_boundary_reference(input_mask_case, num_layers):
    """Count original samples without copying the production rolling algorithm."""
    logical, physical, masks, conditioning = _repacking_fixture("cpu", input_mask_case)
    expected = torch.zeros(num_layers + 1)
    actual = torch.zeros_like(expected)
    for length, padded, mask, valid in zip(logical, physical, masks, conditioning):
        params = PackedSeqParams(
            cu_seqlens_q=torch.tensor([0, length], dtype=torch.int32),
            cu_seqlens_q_padded=torch.tensor([0, padded], dtype=torch.int32),
        )
        actual += mtp.get_mtp_loss_token_counts(
            mask,
            num_layers,
            mtp_input_mask=valid if input_mask_case != "none" else None,
            packed_seq_params=params,
        )
        expected[0] += mask.sum()
        for depth in range(1, num_layers + 1):
            for token in range(max(length - depth, 0)):
                if input_mask_case == "none" or bool(valid[0, token + 1 : token + depth + 1].all()):
                    expected[depth] += mask[0, token + depth]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("input_mask_case", ["none", "holes", "zero"])
@pytest.mark.parametrize("num_layers", [1, 3])
def test_step_global_mtp_gradients_are_invariant_to_repacking_and_cp(
    cp_groups, input_mask_case, num_layers
):
    """Real rolling, cross entropy and parameter gradients across reordered CP1/2/4 packs."""
    device = torch.device("cuda", torch.cuda.current_device())
    logical, physical, masks, conditioning = _repacking_fixture(device, input_mask_case)
    boundaries = torch.tensor([0, *physical], dtype=torch.int32, device=device).cumsum(
        0, dtype=torch.int32
    )
    logical_boundaries = torch.tensor([0, *logical], dtype=torch.int32, device=device).cumsum(
        0, dtype=torch.int32
    )
    full_mask = torch.cat(masks, dim=-1)
    full_input_mask = torch.cat(conditioning, dim=-1) if input_mask_case != "none" else None
    counts = mtp.get_mtp_loss_token_counts(
        full_mask,
        num_layers,
        mtp_input_mask=full_input_mask,
        packed_seq_params=PackedSeqParams(
            cu_seqlens_q=logical_boundaries, cu_seqlens_q_padded=boundaries
        ),
    )
    token_count = sum(physical)

    def execute(schedule, use_global_counts=True):
        # Isolate normalization from TF32 GEMM precision changes at different pack sizes.
        head_total = torch.zeros(5, 4, device=device, dtype=torch.float64)
        hidden_total = torch.zeros(
            num_layers + 1, token_count, 1, 4, device=device, dtype=torch.float64
        )
        for sample_ids, cp_size in schedule:
            group = cp_groups[cp_size]
            head = torch.nn.Parameter(
                torch.arange(20, device=device, dtype=torch.float64).reshape(5, 4) / 20
            )
            hidden = torch.nn.Parameter(
                torch.arange((num_layers + 1) * token_count * 4, device=device, dtype=torch.float64)
                .reshape_as(hidden_total)
                .remainder(19)
                / 19
            )
            indices = torch.cat(
                [
                    torch.arange(boundaries[sample], boundaries[sample + 1], device=device)
                    .reshape(2 * cp_size, -1)[[group.rank(), 2 * cp_size - group.rank() - 1]]
                    .reshape(-1)
                    for sample in sample_ids
                ]
            )
            packed = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=torch.tensor(
                    [0, *[logical[sample] for sample in sample_ids]],
                    dtype=torch.int32,
                    device=device,
                ).cumsum(0, dtype=torch.int32),
                cu_seqlens_q_padded=torch.tensor(
                    [0, *[physical[sample] for sample in sample_ids]],
                    dtype=torch.int32,
                    device=device,
                ).cumsum(0, dtype=torch.int32),
                local_cp_size=cp_size,
                cp_group=group,
                mtp_loss_token_counts=counts if use_global_counts else None,
            )
            labels = torch.arange(token_count, device=device).remainder(5).unsqueeze(0)
            result = mtp.process_mtp_loss(
                hidden_states=hidden[:, indices].reshape(-1, 1, 4),
                labels=labels[:, indices],
                loss_mask=full_mask[:, indices],
                mtp_input_mask=full_input_mask[:, indices] if full_input_mask is not None else None,
                output_layer=lambda value, **kwargs: (F.linear(value, head), None),
                output_weight=None,
                runtime_gather_output=True,
                is_training=False,
                compute_language_model_loss=lambda labels, logits: F.cross_entropy(
                    logits.transpose(0, 1).reshape(-1, 5), labels.reshape(-1), reduction="none"
                ).reshape_as(labels),
                config=_config(num_layers=num_layers),
                cp_group=cp_groups[1],
                packed_seq_params=packed,
            )
            (result.sum() * 0).backward()
            for gradient in (head.grad, hidden.grad):
                torch.distributed.all_reduce(gradient, group=group)
            head_total += head.grad / counts[0]
            hidden_total += hidden.grad / counts[0]
        return head_total, hidden_total

    reference = execute([((0, 1, 2), 1)])
    candidate = execute([((2,), 4), ((1, 0), 2)])
    for expected, actual in zip(reference, candidate):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)
        if input_mask_case == "zero":
            assert torch.count_nonzero(actual) == 0
    if input_mask_case == "none":
        # The same counterexample must detect the original per-pack normalization.
        legacy = execute([((2,), 4), ((1, 0), 2)], use_global_counts=False)
        assert not torch.allclose(legacy[0], reference[0], atol=2e-7, rtol=2e-5)


@pytest.mark.parametrize(
    "counts,per_token,message",
    [(torch.ones(1), True, "every MTP depth"), (torch.ones(2), False, "calculate_per_token_loss")],
)
def test_step_global_mtp_counts_validate_contract(counts, per_token, message):
    with pytest.raises(ValueError, match=message):
        mtp.process_mtp_loss(
            hidden_states=torch.ones(8, 1, 1),
            labels=torch.zeros(1, 4, dtype=torch.long),
            loss_mask=torch.ones(1, 4),
            output_layer=lambda value, **kwargs: (value, None),
            output_weight=None,
            runtime_gather_output=True,
            is_training=False,
            compute_language_model_loss=lambda labels, logits: logits.squeeze(-1).transpose(0, 1),
            config=_config(per_token=per_token),
            packed_seq_params=PackedSeqParams(mtp_loss_token_counts=counts),
        )


def test_mtp_padding_stays_masked_across_packed_cp_boundaries(cp_groups):
    """Shifted conditioning beyond a document or inside its gap never routes to MoE."""
    device = torch.device("cuda", torch.cuda.current_device())
    positions = torch.arange(16, device=device).unsqueeze(0)
    full_padding = ((positions >= 5) & (positions < 8)) | (positions >= 14)
    for cp_size in (1, 2, 4):
        group = cp_groups[cp_size]
        indices = positions.reshape(2, 2 * cp_size, -1)[
            :, [group.rank(), 2 * cp_size - group.rank() - 1]
        ].reshape(-1)
        packed = PackedSeqParams(
            cu_seqlens_q=torch.tensor([0, 5, 11], dtype=torch.int32, device=device),
            cu_seqlens_q_padded=torch.tensor([0, 8, 16], dtype=torch.int32, device=device),
            local_cp_size=cp_size,
            cp_group=group,
        )
        layer = SimpleNamespace(
            cp_group=cp_groups[1],
            config=SimpleNamespace(sequence_parallel=False, mtp_detach_heads=False),
        )
        local_positions = positions[:, indices]
        actual = mtp.MultiTokenPredictionLayer._get_embeddings(
            layer,
            input_ids=local_positions,
            position_ids=local_positions,
            hidden_states=torch.zeros(len(indices), 1, 2, device=device),
            embedding=lambda input_ids, position_ids: torch.zeros(
                len(indices), 1, 2, device=device
            ),
            packed_seq_params=packed,
            padding_mask=full_padding[:, indices],
        )[2]
        # Logical documents occupy [0,5) and [8,14). Each loses its last
        # conditioning slot after the shift; all physical gaps remain padding.
        expected = ((positions >= 4) & (positions < 8)) | (positions >= 13)
        torch.testing.assert_close(actual, expected[:, indices], atol=0, rtol=0)
