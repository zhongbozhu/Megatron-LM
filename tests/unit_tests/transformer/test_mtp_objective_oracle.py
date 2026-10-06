# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Independent LM/MTP objective oracle; no model, GEMM, or visual transport.

Run with four distributed GPU ranks. The reference is a CPU FP64 calculation
over original samples, using explicit targets and the analytic softmax gradient.
The candidate uses FP32 logits, native MIMO LM reduction, real MTP packed P2P
rolling and loss attachment. Its final SUM/main-token divisor is explicit here;
native DDP/optimizer finalization needs a separate integration test.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from examples.mimo.native_step import NativeMimoStep
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import multi_token_prediction as mtp

_VOCAB = 7
_PHYSICAL = 16
_MTP_WEIGHT = 0.3
_SCHEDULES = {
    "cp1": [(((0,), (0,)), ((1,), (1,)), ((2,), (2,)), ((3,), (3,)))],
    "static_cp2": [(((0, 1), (0, 1)), ((2, 3), (2, 3)))],
    # All ranks first execute CP4, then switch to CP2 or CP1. Sample order and
    # pack membership differ from both reference layouts.
    "mixed_cp": [(((3,), (0, 1, 2, 3)),), (((2,), (0, 1)), ((0,), (2,)), ((1,), (3,)))],
}


@pytest.fixture(scope="module")
def oracle_groups():
    """Precreate CP communicators; runtime metadata selects them for each pack."""
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("The LM/MTP objective oracle requires exactly four distributed ranks")
    if not torch.cuda.is_available():
        pytest.skip("This oracle exercises the GPU/NCCL packed MTP path")
    if not torch.distributed.is_initialized():
        from tests.unit_tests.test_utilities import Utils

        Utils.initialize_distributed()
    assert torch.distributed.get_world_size() == 4
    groups = {}
    for size in (1, 2, 4):
        for first in range(0, 4, size):
            ranks = tuple(range(first, first + size))
            group = torch.distributed.new_group(list(ranks))
            if torch.distributed.get_rank() in ranks:
                groups[ranks] = group
    yield groups
    torch.distributed.barrier()
    for group in reversed(list(groups.values())):
        torch.distributed.destroy_process_group(group)


def _source(case):
    """Original samples only: no CP indexing, packed rolls, or production helpers."""
    samples = []
    # Sample 2 supervises position zero: an MTP shift loses that target, making
    # its main/MTP token ratio differ from the samples with a masked prompt.
    for sid, (length, prompt) in enumerate(zip((11, 9, 7, 13), (4, 2, 0, 9))):
        tokens = [(3 * sid + 2 * position + 1) % _VOCAB for position in range(length)]
        supervision = [prompt <= position < length - 1 for position in range(length)]
        if sid == 1:
            supervision[4] = False  # An internal masked label, distinct from padding.
        if case == "no_supervision":
            supervision = [False] * length
        conditioning = [True] * length
        if case == "modality_holes":
            # In sample 3, the CP4 seam from position 9 to position 10 must
            # remain supervised. Position 11 still removes real MTP targets
            # and exercises cumulative conditioning masks at greater depths.
            conditioning[(6, 3, 4, 11)[sid]] = False
        elif case == "no_mtp":
            conditioning = [False] * length
        samples.append((tokens, supervision, conditioning))
    return samples


def _canonical_oracle(samples, logits):
    """Compute targets, masks, counts, losses and analytic gradients in FP64."""
    assert logits.device.type == "cpu" and logits.dtype == torch.float64
    layers, rows, vocab = logits.shape
    targets = torch.zeros(layers, rows, dtype=torch.long)
    valid = torch.zeros(layers, rows, dtype=torch.bool)
    for sid, (tokens, supervision, conditioning) in enumerate(samples):
        for position in range(len(tokens)):
            for depth in range(layers):
                target_position = position + depth + 1
                if target_position >= len(tokens):
                    continue
                row = sid * _PHYSICAL + position
                targets[depth, row] = tokens[target_position]
                valid[depth, row] = supervision[position + depth] and all(
                    conditioning[position + offset] for offset in range(1, depth + 1)
                )
    counts = valid.sum(dim=1).double()
    log_normalizer = torch.logsumexp(logits, dim=-1)
    selected = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    numerators = ((log_normalizer - selected) * valid).sum(dim=1)
    coefficient = torch.full((layers,), _MTP_WEIGHT / (layers - 1), dtype=torch.float64)
    coefficient[0] = 1
    gradient = torch.exp(logits - log_normalizer.unsqueeze(-1))
    gradient.scatter_add_(-1, targets.unsqueeze(-1), -torch.ones(layers, rows, 1).double())
    gradient *= (valid * (coefficient / counts.clamp(min=1)).unsqueeze(-1)).unsqueeze(-1)
    return dict(
        targets=targets,
        valid=valid,
        counts=counts,
        numerators=numerators,
        objective=(coefficient * numerators / counts.clamp(min=1)).sum(),
        gradient=gradient,
    )


def _production_fields(samples, device):
    labels = torch.zeros(1, len(samples) * _PHYSICAL, device=device, dtype=torch.long)
    mask = torch.zeros_like(labels, dtype=torch.float32)
    conditioning = torch.zeros_like(labels, dtype=torch.bool)
    for sid, (tokens, supervision, valid_inputs) in enumerate(samples):
        first = sid * _PHYSICAL
        length = len(tokens)
        labels[0, first : first + length - 1] = torch.tensor(tokens[1:], device=device)
        mask[0, first : first + length] = torch.tensor(supervision, device=device)
        conditioning[0, first : first + length] = torch.tensor(valid_inputs, device=device)
    return labels, mask, conditioning


def _metadata(sample_ids, samples, group, device, counts=None):
    lengths = [len(samples[sid][0]) for sid in sample_ids]
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, *lengths], device=device, dtype=torch.int32).cumsum(
            0, dtype=torch.int32
        ),
        cu_seqlens_q_padded=torch.arange(len(sample_ids) + 1, device=device, dtype=torch.int32)
        * _PHYSICAL,
        local_cp_size=group.size(),
        cp_group=group,
        mtp_loss_token_counts=counts,
    )


def _execute(schedule, groups, samples, values, expected, counts, monkeypatch):
    """Each original sample contributes once across the four-rank DP x CP domain."""
    device = torch.device("cuda", torch.cuda.current_device())
    rank = torch.distributed.get_rank()
    labels, mask, conditioning = _production_fields(samples, device)
    logits = torch.nn.Parameter(values.to(device=device, dtype=torch.float32))
    num_mtp_layers = logits.shape[0] - 1
    observed = torch.zeros(num_mtp_layers + 1, 2, dtype=torch.float64, device=device)
    state = SimpleNamespace(
        loss_sum=torch.zeros((), dtype=torch.float64, device=device),
        token_count=torch.zeros((), dtype=torch.float64, device=device),
        aux=None,
    )
    zero_supervision_packs = torch.zeros((), device=device, dtype=torch.int64)
    seam_targets = torch.zeros((), device=device, dtype=torch.int64)

    def record_metric(loss, correct, total, layer_number, num_layers, *, num_tokens, **kwargs):
        observed[layer_number + 1, 0] += loss.detach().double()
        observed[layer_number + 1, 1] += num_tokens.detach().double()

    monkeypatch.setattr(mtp.MTPLossLoggingHelper, "save_metrics_to_tracker", record_metric)
    monkeypatch.setattr(mtp.MTPLossAutoScaler, "main_loss_backward_scale", torch.tensor(1.0))
    for round_packs in schedule:
        for sample_ids, ranks in round_packs:
            if rank not in ranks:
                continue
            group = groups[ranks]
            chunks = (group.rank(), 2 * group.size() - group.rank() - 1)
            indices = torch.cat(
                [
                    torch.arange(sid * _PHYSICAL, (sid + 1) * _PHYSICAL, device=device)
                    .reshape(2 * group.size(), -1)[list(chunks)]
                    .reshape(-1)
                    for sid in sample_ids
                ]
            )
            params = _metadata(sample_ids, samples, group, device, counts)
            local_mask = mask[:, indices]
            local_conditioning = conditioning[:, indices]
            zero_supervision_packs += (local_mask.sum() == 0).long()
            masks = [local_mask]
            masks.extend(
                local_mask
                for local_mask, _ in mtp._iter_mtp_loss_masks(
                    local_mask, num_mtp_layers, local_conditioning, group, params
                )
            )
            canonical_rows = indices.cpu()
            for depth, actual_mask in enumerate(masks):
                torch.testing.assert_close(
                    actual_mask.cpu().bool().flatten(),
                    expected["valid"][depth, canonical_rows],
                    rtol=0,
                    atol=0,
                )
                for row in canonical_rows.tolist():
                    if expected["valid"][depth, row] and row + depth not in canonical_rows:
                        seam_targets += 1

            captured_targets = []

            def cross_entropy(target, scores):
                captured_targets.append(target.detach().cpu().flatten())
                return F.cross_entropy(
                    scores.transpose(0, 1).reshape(-1, _VOCAB), target.reshape(-1), reduction="none"
                ).reshape_as(target)

            main_logits = mtp.process_mtp_loss(
                hidden_states=logits[:, indices].reshape(-1, 1, _VOCAB),
                labels=labels[:, indices],
                loss_mask=local_mask,
                mtp_input_mask=local_conditioning,
                output_layer=lambda value, **kwargs: (value, None),
                output_weight=None,
                runtime_gather_output=True,
                is_training=True,
                compute_language_model_loss=cross_entropy,
                config=SimpleNamespace(
                    mtp_num_layers=num_mtp_layers,
                    mtp_loss_scaling_factor=_MTP_WEIGHT,
                    calculate_per_token_loss=True,
                    mtp_detach_heads=False,
                    context_parallel_size=1,
                ),
                # Build-time CP1 must not override each pack's runtime group.
                cp_group=groups[(rank,)],
                packed_seq_params=params,
                metric_avg_group=torch.distributed.group.WORLD,
            )
            for depth, target in enumerate(captured_targets, start=1):
                valid = expected["valid"][depth, canonical_rows]
                torch.testing.assert_close(
                    target[valid], expected["targets"][depth, canonical_rows][valid], rtol=0, atol=0
                )
            token_losses = F.cross_entropy(main_logits[:, 0], labels[0, indices], reduction="none")
            raw_loss, _, _ = NativeMimoStep.loss(state, local_mask, token_losses)
            raw_loss.backward()

    observed[0] = torch.stack((state.loss_sum, state.token_count))
    # Deliberately exposed: this test verifies the objective, not native DDP.
    # Every physical token is owned once; there is no additional CP average.
    gradient = logits.grad
    for tensor in (observed, gradient, zero_supervision_packs, seam_targets):
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    gradient /= expected["counts"][0].clamp(min=1).item()
    return gradient.cpu().double(), observed.cpu(), int(zero_supervision_packs), int(seam_targets)


@pytest.mark.parametrize("case", ["text", "modality_holes", "no_mtp", "no_supervision"])
@pytest.mark.parametrize("num_mtp_layers", [1, 3])
def test_lm_mtp_match_original_sample_oracle(oracle_groups, monkeypatch, case, num_mtp_layers):
    """Check both the full logits VJP and independently defined objective/targets."""
    samples = _source(case)
    rows = len(samples) * _PHYSICAL
    # Dyadic values are identical when converted between FP64 and FP32.
    values = (
        torch.arange((num_mtp_layers + 1) * rows * _VOCAB, dtype=torch.float64)
        .reshape(num_mtp_layers + 1, rows, _VOCAB)
        .remainder(31)
        - 15
    ) / 16
    expected = _canonical_oracle(samples, values)
    device = torch.device("cuda", torch.cuda.current_device())
    _, mask, conditioning = _production_fields(samples, device)
    rank = torch.distributed.get_rank()
    counts = mtp.get_mtp_loss_token_counts(
        mask,
        num_mtp_layers,
        mtp_input_mask=conditioning,
        packed_seq_params=_metadata(
            tuple(range(len(samples))), samples, oracle_groups[(rank,)], device
        ),
        cp_group=oracle_groups[(rank,)],
    )
    torch.testing.assert_close(counts.cpu().double(), expected["counts"], rtol=0, atol=0)
    records = []
    for name, schedule in _SCHEDULES.items():
        gradient, observed, empty_packs, seams = _execute(
            schedule, oracle_groups, samples, values, expected, counts, monkeypatch
        )
        torch.testing.assert_close(observed[:, 1], expected["counts"], rtol=0, atol=0)
        torch.testing.assert_close(observed[:, 0], expected["numerators"], rtol=2e-6, atol=2e-6)
        # Tiny FP32 CE/reduction operators against an analytic FP64 oracle; this
        # is not a proposed tolerance for BF16 model gradients.
        torch.testing.assert_close(gradient, expected["gradient"], rtol=2e-6, atol=2e-8)
        assert torch.count_nonzero(gradient[~expected["valid"]]) == 0
        if name == "mixed_cp":
            assert empty_packs > 0
            if case in ("text", "modality_holes"):
                assert seams > 0
        coefficients = torch.tensor(
            [1] + [_MTP_WEIGHT / num_mtp_layers] * num_mtp_layers, dtype=torch.float64
        )
        objective = (coefficients * observed[:, 0] / observed[:, 1].clamp(min=1)).sum()
        torch.testing.assert_close(objective, expected["objective"], rtol=2e-6, atol=2e-7)
        records.append(
            dict(
                layout=name,
                counts=observed[:, 1].tolist(),
                objective=float(objective),
                objective_abs_error=float((objective - expected["objective"]).abs()),
                loss_numerator_max_abs=float((observed[:, 0] - expected["numerators"]).abs().max()),
                gradient_max_abs=float((gradient - expected["gradient"]).abs().max()),
                zero_local_supervision_packs=empty_packs,
                supervised_mtp_cp_seams=seams,
            )
        )
    negative_control_error = None
    if case == "text" and num_mtp_layers == 1:
        # A negative control: per-pack MTP normalization preserves neither the
        # original-sample objective nor its gradient when samples are repacked.
        legacy, _, _, _ = _execute(
            _SCHEDULES["mixed_cp"], oracle_groups, samples, values, expected, None, monkeypatch
        )
        negative_control_error = float((legacy - expected["gradient"]).abs().max())
        assert negative_control_error > 1e-4
    report = dict(
        case=case,
        mtp_layers=num_mtp_layers,
        rank=rank,
        reference="original-sample CPU FP64 logsumexp and analytic softmax gradient",
        candidate="FP32 NativeMimoStep.loss and process_mtp_loss, real packed NCCL P2P",
        normalization="explicit world SUM divided by original main supervised token count",
        gradient_tolerance=dict(rtol=2e-6, atol=2e-8),
        loss_numerator_tolerance=dict(rtol=2e-6, atol=2e-6),
        objective_tolerance=dict(rtol=2e-6, atol=2e-7),
        legacy_per_pack_negative_control_max_abs=negative_control_error,
        results=records,
    )
    report_dir = os.environ.get("MIMO_OBJECTIVE_ORACLE_REPORT_DIR")
    if report_dir:
        destination = Path(report_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / f"mtp_oracle_{case}_depth{num_mtp_layers}_rank{rank}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    if rank == 0:
        print("MTP_OBJECTIVE_ORACLE " + json.dumps(report))
