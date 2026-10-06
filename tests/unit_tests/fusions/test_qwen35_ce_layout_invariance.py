# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Full-vocabulary BF16 CE diagnostic, independent of a decoder/model graph."""

import json
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy
from megatron.core.tensor_parallel.cross_entropy import unfused_cross_entropy


# Declare acceptance before observing GPU results. BF16 rounding is included in
# comparisons against FP64; very confident target gradients also have a separate
# absolute bound, since relative errors near a zero derivative are ill-conditioned.
TOLERANCES = dict(
    loss_max_abs=2e-5,
    gradient_relative_l2=0.005,
    gradient_max_abs=0.0025,
    confident_target_gradient_max_abs=5e-7,
    layout_gradient_relative_l2=0.0005,
    peak_allocated_gib=60.0,
)


def _empty_stats():
    return torch.zeros(3, dtype=torch.float64, device="cuda")


def _add_stats(stats, candidate, reference):
    a, b = candidate.double(), reference.double()
    difference = a - b
    stats[0].add_(difference.square().sum())
    stats[1].add_(b.square().sum())
    stats[2] = torch.maximum(stats[2], difference.abs().max())


def _report_stats(stats):
    difference, reference, maximum = stats.tolist()
    return dict(
        relative_l2=(difference / reference) ** 0.5 if reference else None,
        reference_l2=reference**0.5,
        zero_reference=reference == 0,
        max_abs=maximum,
    )


def _execute(function, logits, targets, upstream):
    leaf = logits.unsqueeze(1).clone().requires_grad_()
    losses = function(leaf, targets.unsqueeze(1), dist.group.WORLD)
    losses.backward(upstream.unsqueeze(1))
    return losses.detach().flatten(), leaf.grad.detach().squeeze(1)


def _compare_reference(logits, targets, upstream, outputs):
    stats = {name: _empty_stats() for name in outputs}
    class_names = ("diffuse", "moderate", "confident", "saturated")
    class_stats = {name: {kind: _empty_stats() for kind in class_names} for name in outputs}
    target_stats = {name: {kind: _empty_stats() for kind in class_names} for name in outputs}
    loss_error = {name: torch.zeros((), device="cuda") for name in outputs}
    confident_error = {name: torch.zeros((), device="cuda") for name in outputs}
    masked_error = {name: torch.zeros((), device="cuda") for name in outputs}
    cross = _empty_stats()
    for start in range(0, len(logits), 32):
        stop = min(start + 32, len(logits))
        reference_input = logits[start:stop].double().requires_grad_()
        reference_loss = F.cross_entropy(reference_input, targets[start:stop], reduction="none")
        reference_loss.backward(upstream[start:stop].double())
        reference_gradient = reference_input.grad
        rows = torch.arange(start, stop, device="cuda")
        confident = (rows % 4 >= 2) & (upstream[start:stop] != 0)
        masked = upstream[start:stop] == 0
        for name, (losses, gradient) in outputs.items():
            _add_stats(stats[name], gradient[start:stop], reference_gradient)
            loss_error[name] = torch.maximum(
                loss_error[name],
                (losses[start:stop].double() - reference_loss.detach()).abs().max(),
            )
            actual_target = gradient[start:stop].gather(1, targets[start:stop, None]).flatten()
            expected_target = reference_gradient.gather(1, targets[start:stop, None]).flatten()
            for kind, class_name in enumerate(class_names):
                selected = rows % 4 == kind
                _add_stats(
                    class_stats[name][class_name],
                    gradient[start:stop][selected],
                    reference_gradient[selected],
                )
                _add_stats(
                    target_stats[name][class_name],
                    actual_target[selected],
                    expected_target[selected],
                )
            if confident.any():
                confident_error[name] = torch.maximum(
                    confident_error[name],
                    (actual_target[confident] - expected_target[confident]).abs().max(),
                )
            if masked.any():
                masked_error[name] = torch.maximum(
                    masked_error[name], gradient[start:stop][masked].abs().max()
                )
        _add_stats(cross, outputs["fused"][1][start:stop], outputs["unfused"][1][start:stop])
    return dict(
        implementations={
            name: dict(
                gradient=_report_stats(stats[name]),
                loss_max_abs=float(loss_error[name]),
                confident_target_gradient_max_abs=float(confident_error[name]),
                masked_gradient_max_abs=float(masked_error[name]),
                confidence_classes={
                    kind: dict(
                        gradient=_report_stats(class_stats[name][kind]),
                        target_gradient=_report_stats(target_stats[name][kind]),
                    )
                    for kind in class_names
                },
            )
            for name in outputs
        },
        fused_vs_unfused_gradient=_report_stats(cross),
    )


def test_full_vocab_ce_layout_invariance():
    if not os.environ.get("MIMO_CE_REPORT"):
        pytest.skip("Set MIMO_CE_REPORT to opt into the full-vocabulary GPU diagnostic")
    assert int(os.environ.get("WORLD_SIZE", "1")) == 1, "This bounded diagnostic uses one GPU"
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    try:
        torch.manual_seed(314159)
        torch.cuda.reset_peak_memory_stats()
        vocabulary, largest = 248320, 8192
        logits = torch.randn(largest, vocabulary, device="cuda", dtype=torch.bfloat16)
        rows = torch.arange(largest, device="cuda")
        targets = (rows * 997 + 101) % vocabulary
        # All row-count cases use views of exactly the same logits, avoiding RNG
        # launch-geometry differences masquerading as CE layout differences.
        for kind, boost in ((1, 14.0), (2, 22.0), (3, 30.0)):
            selected = rows[rows % 4 == kind]
            logits[selected, targets[selected]] = boost
        upstream = torch.tensor([1.0, 0.1, 0.0], device="cuda")[rows % 3]
        functions = dict(fused=fused_vocab_parallel_cross_entropy, unfused=unfused_cross_entropy)
        cases, prefix = [], {}
        for count in (32, 512, largest):
            print(f"QWEN35_CE_START rows={count}", flush=True)
            outputs = {
                name: _execute(function, logits[:count], targets[:count], upstream[:count])
                for name, function in functions.items()
            }
            result = dict(
                rows=count,
                **_compare_reference(logits[:count], targets[:count], upstream[:count], outputs),
            )
            if count == 32:
                prefix = {
                    name: (loss.clone(), grad.clone()) for name, (loss, grad) in outputs.items()
                }
            else:
                result["prefix32"] = {}
                for name, (loss, gradient) in outputs.items():
                    stats = _empty_stats()
                    _add_stats(stats, gradient[:32], prefix[name][1])
                    result["prefix32"][name] = dict(
                        gradient=_report_stats(stats),
                        loss_max_abs=float((loss[:32] - prefix[name][0]).abs().max()),
                    )
            if count == largest:
                stats, loss_error = _empty_stats(), torch.zeros((), device="cuda")
                for start in range(0, largest, 512):
                    stop = start + 512
                    loss, gradient = _execute(
                        functions["fused"],
                        logits[start:stop],
                        targets[start:stop],
                        upstream[start:stop],
                    )
                    for offset in range(0, 512, 32):
                        _add_stats(
                            stats,
                            gradient[offset : offset + 32],
                            outputs["fused"][1][start + offset : start + offset + 32],
                        )
                    loss_error = torch.maximum(
                        loss_error, (loss - outputs["fused"][0][start:stop]).abs().max()
                    )
                result["split512"] = dict(
                    gradient=_report_stats(stats), loss_max_abs=float(loss_error)
                )
            cases.append(result)
            print("QWEN35_CE_CASE " + json.dumps(result), flush=True)
            del outputs
        report = dict(
            vocabulary=vocabulary,
            cases=cases,
            tolerances=TOLERANCES,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            scope="Identical synthetic BF16 logits; main/MTP-style upstream weights1/.1 and zero padding. No model/transport claim.",
        )
        checks = [report["peak_allocated_gib"] < TOLERANCES["peak_allocated_gib"]]
        for case in cases:
            checks.append(
                case["fused_vs_unfused_gradient"]["relative_l2"]
                < TOLERANCES["layout_gradient_relative_l2"]
            )
            for result in case["implementations"].values():
                checks.extend(
                    (
                        result["loss_max_abs"] < TOLERANCES["loss_max_abs"],
                        result["gradient"]["relative_l2"] < TOLERANCES["gradient_relative_l2"],
                        result["gradient"]["max_abs"] < TOLERANCES["gradient_max_abs"],
                        result["confident_target_gradient_max_abs"]
                        < TOLERANCES["confident_target_gradient_max_abs"],
                        result["masked_gradient_max_abs"] == 0,
                    )
                )
            for result in [
                *case.get("prefix32", {}).values(),
                *([case["split512"]] if "split512" in case else []),
            ]:
                checks.extend(
                    (
                        result["gradient"]["relative_l2"]
                        < TOLERANCES["layout_gradient_relative_l2"],
                        result["loss_max_abs"] < TOLERANCES["loss_max_abs"],
                    )
                )
        report["passed"] = all(checks)
        if os.environ.get("MIMO_CE_REPORT"):
            Path(os.environ["MIMO_CE_REPORT"]).write_text(json.dumps(report, indent=2) + "\n")
        print("QWEN35_CE_RESULT " + json.dumps(report), flush=True)
        assert report["passed"], report
    finally:
        dist.destroy_process_group()
