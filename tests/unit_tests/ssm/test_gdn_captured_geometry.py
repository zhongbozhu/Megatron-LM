# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Opt-in replay of one captured Qwen3.5 GDN pack, without the full VL model.

Launch on 16 ranks with MIMO_GDN_CAPTURE pointing at a training-round .pt,
MIMO_GDN_PRETRAINED_CHECKPOINT pointing at its initial iteration directory, and
MIMO_GDN_CAPTURE_REPORT naming a new JSON output. MIMO_GDN_CAPTURE_DTYPES defaults
to bf16,fp32. MIMO_GDN_CAPTURE_LAYER_INDEX defaults to 0; only that pretrained
GDN module is instantiated. No training source is modified.

Numerical differences are measurements, not a relaxed full-model parity gate.
The fixed-projection cases use the baseline pre-output-projection cotangent,
so neither input nor output GEMM geometry can contaminate core VJP isolation.
MIMO_GDN_CAPTURE_CASES optionally selects comma-separated case names without the
dtype prefix; reference cases are always run when needed. Additional sequence
reordering/intersequence-padding cases run by default only for BF16, but explicit
case selection also enables them for FP32.

CP4 cases are opt-in through MIMO_GDN_CAPTURE_CASES=cp4,fixed_core_cp4.
MIMO_GDN_CAPTURE_FOCUS_SAMPLE_ID adds real-token metrics for one sample without
changing the full pack, padding, or cotangent. MIMO_GDN_CAPTURE_COMMON_BF16_CORE=1
reuses identical BF16 CP1 projection/cotangent tensors in both precisions; BF16
must run first, and all ranks must produce identical source tensors.
For a CP-sharded capture, peer files from the same round are loaded and restored
to the complete pack. A replay at the captured CP size checks native geometry;
CP1 remains the fixed-input numerical reference, not a native-layout parity claim.
MIMO_GDN_CAPTURE_ISOLATION_SAMPLES=47,62 selects a separate BF16 diagnostic:
each sample is the sole cotangent source in turn, while the other sample's real
input rows are negated. CP1 and captured-CP baseline/repeat/perturb cases retain
identical shapes, boundaries, padding and weights within each comparison.
"""

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_gated_delta_net_module_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import convert_module_to_dtype_except_fp32_marked
from tests.unit_tests.ssm.test_gdn_qwen35_cp_geometry import (
    _indices,
    _load_pretrained_layers,
    _packing,
    _statistics,
)
from tests.unit_tests.test_utilities import Utils


def _canonicalize_capture(data, path, layer_index):
    """Restore one captured CP pack without changing rows, boundaries or values."""
    ranks = data["cp_ranks"]
    if len(ranks) == 1:
        return data
    assert len(set(ranks)) == len(ranks), "Duplicate captured CP ranks"
    metadata = (
        "step",
        "phase",
        "round",
        "cp_ranks",
        "sample_ids",
        "sample_lengths",
        "padded_boundaries",
        "logical_boundaries",
    )
    peers = []
    for rank in ranks:
        peer_path = Path(path).parent.parent / f"rank{rank:05d}" / Path(path).name
        peer = torch.load(peer_path, map_location="cpu", weights_only=True, mmap=True)
        assert all(peer[key] == data[key] for key in metadata), "CP peer metadata mismatch"
        assert str(layer_index) in peer.get("gdn_layers", {}), "CP peer lacks full GDN layer"
        assert peer["records"].keys() == data["records"].keys(), "CP peer probe schema differs"
        peers.append(peer)
    physical = torch.cat([peer["physical_indices"] for peer in peers])
    total = data["padded_boundaries"][-1]
    assert physical.dtype == torch.long and physical.ndim == 1
    order = physical.argsort()
    assert torch.equal(
        physical[order], torch.arange(total)
    ), "CP rows lack unique complete coverage"
    selected = {}
    for name in ("input", "output", "output_gradient"):
        values = []
        for peer in peers:
            value = peer["gdn_layers"][str(layer_index)][name]
            assert value.shape[0] == peer["physical_indices"].numel()
            assert value.dtype == torch.bfloat16 and value.shape[1:] == (1, 2048)
            assert torch.isfinite(value).all(), f"Nonfinite captured GDN {name}"
            values.append(value)
        selected[name] = torch.cat(values).index_select(0, order)
    records = {}
    for name, invocations in data["records"].items():
        assert all(
            len(peer["records"][name]) == len(invocations) for peer in peers
        ), "CP peers have different forward/recompute invocation counts"
        records[name] = []
        for index, record in enumerate(invocations):
            parts = [peer["records"][name][index] for peer in peers]
            assert all(part.keys() == record.keys() for part in parts), "CP probe fields differ"
            merged = {}
            for field, value in record.items():
                if isinstance(value, torch.Tensor):
                    assert all(
                        part[field].shape[0] == len(peer["keys"])
                        and part[field].shape[1:] == value.shape[1:]
                        and part[field].dtype == value.dtype
                        for peer, part in zip(peers, parts)
                    ), "CP probe shape or dtype mismatch"
                    merged[field] = torch.cat([part[field] for part in parts])
                else:
                    assert all(part[field] == value for part in parts), "CP probe metadata differs"
                    merged[field] = value
            records[name].append(merged)
    keys = [tuple(key) for peer in peers for key in peer["keys"]]
    assert len(keys) == len(set(keys)), "Duplicate sample/token probe owner"
    return dict(
        data,
        gdn_layers={str(layer_index): selected},
        records=records,
        keys=keys,
        physical_indices=torch.arange(total),
        local_tokens=total,
        capture_peer_files=[
            str(Path(path).parent.parent / f"rank{rank:05d}" / Path(path).name) for rank in ranks
        ],
    )


def _load_capture(path, checkpoint, layer_index=0):
    data = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    data = _canonicalize_capture(dict(data), path, layer_index)
    selected = data.get("gdn_layers", {}).get(str(layer_index))
    if selected is not None:
        assert all(key in selected for key in ("input", "output", "output_gradient"))
        data["gdn_input"] = selected["input"]
        data["gdn_output"] = selected["output"]
        data["gdn_output_gradient"] = selected["output_gradient"]
    else:
        assert layer_index == 0, f"Capture lacks GDN layer {layer_index}"
    data["gdn_layer_index"] = layer_index
    required = (
        "step",
        "phase",
        "cp_ranks",
        "sample_ids",
        "sample_lengths",
        "physical_indices",
        "padded_boundaries",
        "logical_boundaries",
        "gdn_input",
        "gdn_output_gradient",
        "records",
        "keys",
    )
    assert all(key in data for key in required), "Capture lacks full GDN input/cotangent metadata"
    assert (
        data["step"] == 0 and Path(checkpoint).name == "iter_0000000"
    ), "This diagnostic loads pretrained weights; capture must precede the first optimizer update"
    assert data["phase"] == "training"
    assert len(data["cp_ranks"]) in (1, 2, 4, 8, 16), "Unsupported captured CP size"
    value, gradient = data["gdn_input"], data["gdn_output_gradient"]
    assert value.dtype == gradient.dtype == torch.bfloat16
    assert value.ndim == 3 and value.shape[1:] == (1, 2048)
    assert value.shape == gradient.shape and value.numel() > 0
    if "gdn_output" in data:
        assert data["gdn_output"].shape == value.shape
        assert data["gdn_output"].dtype == value.dtype
        assert torch.isfinite(data["gdn_output"]).all()
    assert torch.isfinite(value).all() and torch.isfinite(gradient).all()
    total = value.shape[0]
    assert torch.equal(data["physical_indices"], torch.arange(total))
    padded, logical = data["padded_boundaries"], data["logical_boundaries"]
    assert padded[0] == logical[0] == 0 and padded[-1] == total
    assert len(padded) == len(logical)
    assert len(data["sample_ids"]) == len(data["sample_lengths"])
    assert len(set(data["sample_ids"])) == len(data["sample_ids"])
    assert len(data["sample_ids"]) < len(padded)
    for index, (begin, end) in enumerate(zip(padded, padded[1:])):
        assert end > begin and (end - begin) % 16 == 0, "Same boundaries must support CP8"
        assert 0 <= logical[index + 1] - logical[index] <= end - begin
        if index < len(data["sample_ids"]):
            assert logical[index + 1] - logical[index] == data["sample_lengths"][index]
    return data


def _write_sharded_fixture(directory):
    full = torch.arange(32 * 2048).remainder(127).reshape(32, 1, 2048).bfloat16()
    for local_rank, rank in enumerate([4, 5, 6, 7]):
        rows = torch.arange(local_rank, 32, 4)
        real = rows[(rows < 13) | ((rows >= 16) & (rows < 27))]
        keys = [(47, int(row)) if row < 16 else (62, int(row) - 16) for row in real]
        records = {}
        for name, value in (
            ("layer00.gdn_input", full),
            ("layer00.input_projection", full),
            ("layer00.attention", full + 1),
        ):
            sample = value[real].flatten(1)
            records[name] = [
                dict(grad_enabled=False, value=sample),
                dict(grad_enabled=True, value=sample, gradient=sample + 4),
            ]
        data = dict(
            step=0,
            phase="training",
            round=3,
            cp_ranks=[4, 5, 6, 7],
            sample_ids=[47, 62],
            sample_lengths=[13, 11],
            padded_boundaries=[0, 16, 32],
            logical_boundaries=[0, 13, 24],
            physical_indices=rows,
            keys=keys,
            records=records,
            gdn_layers={
                "0": dict(input=full[rows], output=full[rows] + 1, output_gradient=full[rows] + 4)
            },
        )
        path = directory / f"rank{rank:05d}" / "training-round0003.pt"
        path.parent.mkdir(parents=True)
        torch.save(data, path)
    return directory / "rank00004" / "training-round0003.pt", full


def test_cp_capture_reconstructs_full_pack_and_aligned_probes(tmp_path):
    path, full = _write_sharded_fixture(tmp_path)
    data = _load_capture(path, tmp_path / "iter_0000000")
    assert data["cp_ranks"] == [4, 5, 6, 7]
    assert len(data["capture_peer_files"]) == 4
    torch.testing.assert_close(data["gdn_input"], full, rtol=0, atol=0)
    torch.testing.assert_close(data["gdn_output"], full + 1, rtol=0, atol=0)
    torch.testing.assert_close(data["gdn_output_gradient"], full + 4, rtol=0, atol=0)
    torch.testing.assert_close(data["physical_indices"], torch.arange(32), rtol=0, atol=0)
    rows = torch.tensor([position + (16 if sid == 62 else 0) for sid, position in data["keys"]])
    assert rows.numel() == 24 and len(set(data["keys"])) == 24
    torch.testing.assert_close(
        data["records"]["layer00.gdn_input"][0]["value"], full[rows].flatten(1), rtol=0, atol=0
    )
    assert data["padded_boundaries"] == [0, 16, 32]
    assert data["sample_ids"] == [47, 62]


@pytest.mark.parametrize("invalid", ["missing_peer", "duplicate_row", "boundaries", "probe_owner"])
def test_cp_capture_rejects_partial_or_ambiguous_reconstruction(tmp_path, invalid):
    path, _ = _write_sharded_fixture(tmp_path)
    peer_path = tmp_path / "rank00005" / "training-round0003.pt"
    if invalid == "missing_peer":
        peer_path.unlink()
        expected = FileNotFoundError
    else:
        peer = torch.load(peer_path, weights_only=True)
        if invalid == "duplicate_row":
            peer["physical_indices"][0] = 0
        elif invalid == "boundaries":
            peer["padded_boundaries"][1] = 15
        else:
            peer["keys"][0] = (47, 0)
        torch.save(peer, peer_path)
        expected = AssertionError
    with pytest.raises(expected):
        _load_capture(path, tmp_path / "iter_0000000")


@pytest.mark.parametrize("empty_probes", [False, True])
def test_native_cp_probes_compare_only_the_local_original_rows(tmp_path, monkeypatch, empty_probes):
    path, full = _write_sharded_fixture(tmp_path)
    data = _load_capture(path, tmp_path / "iter_0000000")
    rows = (
        torch.tensor([13, 14, 15, 27, 28, 29, 30, 31]) if empty_probes else torch.arange(3, 32, 4)
    )
    observations = dict(original_rows=rows, projection=full[rows], output=full[rows] + 1)
    calls = []

    def local_compare(actual, expected, group):
        calls.append(actual.shape[0])
        return dict(exact=torch.equal(actual, expected))

    monkeypatch.setitem(_native_probes.__globals__, "_row_comparison", local_compare)
    group = type("CapturedCP", (), {"size": lambda self: 4})()
    result = _native_probes(observations, data, group)
    assert result["same_cp_geometry"] and result["captured_cp_size"] == 4
    assert result["full_original_output"]["exact"]
    assert all(
        item["exact"]
        for name in ("layer00.input_projection", "layer00.attention")
        for item in result[name]
    )
    assert calls[0] == rows.numel()
    assert all(count == 0 for count in calls[1:]) == empty_probes


def _make_layer(pg, dtype, checkpoint, layer_index=0):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=2048,
        num_attention_heads=16,
        num_query_groups=2,
        kv_channels=256,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        layernorm_zero_centered_gamma=True,
        params_dtype=dtype,
        bf16=dtype == torch.bfloat16,
        gradient_accumulation_fusion=False,
        context_parallel_size=8,
        activation_func=F.silu,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        experimental_attention_variant="gdn",
        linear_attention_freq=[1],
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        transformer_impl="transformer_engine",
    )
    spec = get_gated_delta_net_module_spec(config)
    layer = spec.module(
        config,
        submodules=spec.submodules,
        layer_number=layer_index + 1,
        bias=False,
        conv_bias=False,
        conv_init=0.1,
        use_qk_l2norm=True,
        A_init_range=(1, 16),
        pg_collection=pg,
    ).cuda()
    # Load into the same BF16 destinations as native Float16Module, including
    # GDN1 A_log/dt_bias. FP32 replay then upcasts these exact quantized values;
    # it must not recover any extra checkpoint weight precision.
    convert_module_to_dtype_except_fp32_marked(layer, torch.bfloat16)
    _load_pretrained_layers(
        torch.nn.ModuleList([layer]), pg, checkpoint, layer_indices=[layer_index]
    )
    convert_module_to_dtype_except_fp32_marked(layer, dtype)
    for parameter in layer.parameters():
        if not getattr(parameter, "keep_in_fp32", False):
            assert torch.equal(parameter, parameter.bfloat16().to(dtype))
    layer.train()
    return layer


def _geometry(data, *, extra=0, reverse=False, sequence_padding=0):
    """Map each new physical row to its original row; new zero rows use -1."""
    original_padded = data["padded_boundaries"]
    original_logical = data["logical_boundaries"]
    order = list(range(len(original_padded) - 1))
    if reverse:
        order.reverse()
    padded, logical, pieces = [0], [0], []
    for index in order:
        begin, end = original_padded[index : index + 2]
        pieces.append(torch.arange(begin, end))
        if sequence_padding:
            pieces.append(torch.full((sequence_padding,), -1, dtype=torch.long))
        padded.append(padded[-1] + end - begin + sequence_padding)
        logical.append(logical[-1] + original_logical[index + 1] - original_logical[index])
    if extra:
        pieces.append(torch.full((extra,), -1, dtype=torch.long))
        padded.append(padded[-1] + extra)
        logical.append(logical[-1] + extra)
    canonical = torch.cat(pieces)
    assert torch.equal(
        canonical[canonical >= 0].sort().values, torch.arange(original_padded[-1])
    ), "Every original physical row, including original padding, must occur exactly once"
    return padded, logical, canonical


def _remap(value, canonical):
    """Move original rows and leave newly inserted rows exactly zero."""
    valid = canonical >= 0
    result = value.new_zeros((canonical.numel(), *value.shape[1:]))
    result[valid] = value.index_select(0, canonical[valid])
    return result


def test_canonical_geometry_preserves_original_rows_and_adjoint():
    data = dict(padded_boundaries=[0, 16, 48], logical_boundaries=[0, 11, 40])
    padded, logical, canonical = _geometry(data, reverse=True, sequence_padding=16, extra=32)
    assert padded == [0, 48, 80, 112]
    assert logical == [0, 29, 40, 72]
    assert canonical[:32].tolist() == list(range(16, 48))
    assert canonical[48:64].tolist() == list(range(16))
    assert int((canonical < 0).sum()) == 64
    value = torch.arange(48, dtype=torch.float64).unsqueeze(1).requires_grad_()
    cotangent = torch.linspace(-1, 1, 48, dtype=torch.float64).unsqueeze(1)
    remapped = _remap(value, canonical)
    assert torch.count_nonzero(remapped[canonical < 0]) == 0
    (remapped * _remap(cotangent, canonical)).sum().backward()
    torch.testing.assert_close(value.grad, cotangent, atol=0, rtol=0)


def _run_case(layer, group, data, *, extra=0, reverse=False, sequence_padding=0, fixed=None):
    """Return this CP rank's outputs/VJPs, with exact original token indices."""
    dtype = next(layer.parameters()).dtype
    padded, logical, canonical = _geometry(
        data, extra=extra, reverse=reverse, sequence_padding=sequence_padding
    )
    packing = _packing(group, padded, logical)
    rows = _indices(packing)
    canonical = canonical.index_select(0, rows.cpu())
    inputs = _remap(data["gdn_input"], canonical).cuda().to(dtype)
    inputs.requires_grad_(True)
    output_gradient = _remap(data["gdn_output_gradient"], canonical).cuda().to(dtype)
    original_rows = canonical >= 0
    observations, live = {}, {}
    layer.zero_grad(set_to_none=True)

    def input_projection(module, args, output):
        value, bias = output
        assert bias is None
        if fixed is not None:
            value = (
                _remap(fixed["projection"], canonical)
                .cuda()
                .to(dtype)
                .detach()
                .requires_grad_(True)
            )
        observations["projection"] = value.detach().cpu()
        value.register_hook(
            lambda gradient: observations.update(projection_gradient=gradient.detach().cpu())
        )
        return value, bias

    def before_output_projection(module, args):
        value = args[0]
        live["core_output"] = value
        observations["core_output"] = value.detach().cpu()
        value.register_hook(
            lambda gradient: observations.update(core_gradient=gradient.detach().cpu())
        )

    handles = [
        layer.in_proj.register_forward_hook(input_projection),
        layer.out_proj.register_forward_pre_hook(before_output_projection),
    ]
    try:
        with torch.enable_grad():
            output, bias = layer(inputs, None, packed_seq_params=packing)
            assert bias is None
            observations["output"] = output.detach().cpu()
            if fixed is None:
                output.backward(output_gradient)
            else:
                core_gradient = _remap(fixed["core_gradient"], canonical).cuda().to(dtype)
                live["core_output"].backward(core_gradient)
    finally:
        for handle in handles:
            handle.remove()
    observations["input_gradient"] = None if inputs.grad is None else inputs.grad.cpu()
    parameters = {}
    for name, parameter in layer.named_parameters():
        if fixed is not None and name.startswith(("in_proj.", "out_proj.")):
            assert parameter.grad is None, name
            continue
        assert parameter.grad is not None, name
        gradient = parameter.grad.float()
        if group.size() > 1:
            dist.all_reduce(gradient, group=group)
        parameters[name] = gradient.cpu()
    observations["parameters"] = parameters
    observations["original_rows"] = canonical[original_rows]
    # Exclude only newly inserted rows from output/VJP comparisons. Keep
    # all original padding and its cotangent exactly as captured.
    for name, value in list(observations.items()):
        if isinstance(value, torch.Tensor) and name != "original_rows":
            observations[name] = value[original_rows]
    observations["geometry"] = dict(
        added_dummy_tokens=extra,
        reverse_sequence_order=reverse,
        added_padding_per_sequence=sequence_padding,
        physical_boundaries=padded,
        logical_boundaries=logical,
    )
    return observations


def _focus_sample(data, sample_id):
    if sample_id is None:
        return None
    assert sample_id in data["sample_ids"], f"Sample {sample_id} is absent from this capture"
    slot = data["sample_ids"].index(sample_id)
    begin = data["padded_boundaries"][slot]
    return dict(
        sample_id=sample_id,
        begin=begin,
        end=begin + data["sample_lengths"][slot],
        padded_end=data["padded_boundaries"][slot + 1],
    )


def _focus_rows(rows, focus):
    return (rows >= focus["begin"]) & (rows < focus["end"])


def _fixed_core_source(baseline):
    """Keep the original quantized CPU values, not a new per-dtype projection."""
    source, hashes = {}, {}
    for name in ("projection", "core_gradient"):
        value = baseline[name].detach().cpu().contiguous()
        assert value.dtype == torch.bfloat16 and torch.isfinite(value).all(), name
        source[name] = value
        digest = hashlib.sha256(str(tuple(value.shape)).encode())
        digest.update(memoryview(value.view(torch.uint8).numpy()))
        hashes[name] = digest.hexdigest()
    return source, hashes


def _row_comparison(value, expected, group, *, device="cuda"):
    # A CP rank can own none of the focus sample's real rows. Its empty local
    # contribution must still enter the same collectives, with zero max error.
    actual_gpu = value.to(device) if value.numel() else torch.zeros(1, device=device)
    expected_gpu = expected.to(device) if expected.numel() else torch.zeros(1, device=device)
    difference = _statistics(actual_gpu, expected_gpu, group)
    exact = torch.tensor(int(torch.equal(value, expected)), device=device)
    finite = torch.tensor(
        int(torch.isfinite(value).all() and torch.isfinite(expected).all()), device=device
    )
    dist.all_reduce(exact, op=dist.ReduceOp.MIN, group=group)
    dist.all_reduce(finite, op=dist.ReduceOp.MIN, group=group)
    return dict(defined=True, exact=bool(exact), all_finite=bool(finite), **difference)


def test_focus_metrics_follow_original_rows_without_repacking():
    data = dict(
        sample_ids=[4, 47],
        sample_lengths=[11, 29],
        padded_boundaries=[0, 16, 48],
        logical_boundaries=[0, 11, 40],
    )
    focus = _focus_sample(data, 47)
    assert focus == dict(sample_id=47, begin=16, end=45, padded_end=48)
    _, _, canonical = _geometry(data, reverse=True, sequence_padding=16, extra=32)
    value = torch.arange(48, dtype=torch.float32)
    moved = _remap(value, canonical)
    mask = _focus_rows(canonical, focus)
    assert int(mask.sum()) == 29
    torch.testing.assert_close(moved[mask], value[16:45], rtol=0, atol=0)
    assert not mask[canonical < 0].any()
    assert not mask[(canonical >= 45) & (canonical < 48)].any()
    assert _focus_sample(data, None) is None
    with pytest.raises(AssertionError, match="absent"):
        _focus_sample(data, 48)


def test_common_bf16_core_keeps_same_values_and_hashes_across_precision():
    baseline = dict(
        projection=torch.linspace(-3, 3, 18).reshape(3, 1, 6).bfloat16().requires_grad_(),
        core_gradient=torch.linspace(1, 2, 12).reshape(3, 1, 4).bfloat16(),
    )
    source, hashes = _fixed_core_source(baseline)
    for name, value in source.items():
        assert not value.requires_grad and value.dtype == torch.bfloat16
        torch.testing.assert_close(value.float(), baseline[name].float(), rtol=0, atol=0)
    assert (
        _fixed_core_source({name: value.clone() for name, value in baseline.items()})[1] == hashes
    )
    changed = dict(baseline, core_gradient=baseline["core_gradient"] + 1)
    changed_hashes = _fixed_core_source(changed)[1]
    assert changed_hashes["projection"] == hashes["projection"]
    assert changed_hashes["core_gradient"] != hashes["core_gradient"]
    with pytest.raises(AssertionError, match="projection"):
        _fixed_core_source(dict(baseline, projection=baseline["projection"].float()))


def test_empty_focus_rank_still_enters_all_metric_collectives(monkeypatch):
    calls = []
    group = object()

    def reduce(value, op=dist.ReduceOp.SUM, group=None):
        calls.append((value.clone(), op, group))

    monkeypatch.setattr(dist, "all_reduce", reduce)
    report = _row_comparison(torch.empty((0, 3)), torch.empty((0, 3)), group, device="cpu")
    assert report == dict(defined=True, exact=True, all_finite=True, relative_l2=0.0, max_abs=0.0)
    assert len(calls) == 4 and all(call[2] is group for call in calls)
    assert torch.count_nonzero(calls[0][0]) == torch.count_nonzero(calls[1][0]) == 0
    # A nonempty local contribution uses the same reduction order and formula.
    calls.clear()
    report = _row_comparison(torch.tensor([3.0]), torch.tensor([2.0]), group, device="cpu")
    assert not report["exact"] and report["relative_l2"] == pytest.approx(0.5)
    assert report["max_abs"] == 1 and len(calls) == 4


def _compare(candidate, reference, group, focus=None):
    rows = candidate["original_rows"]
    report = {
        "reported_cp_ranks": dist.get_process_group_ranks(group),
        "geometry": candidate["geometry"],
    }
    if focus is not None:
        mask = _focus_rows(rows, focus)
        count = mask.sum().cuda()
        dist.all_reduce(count, group=group)
        assert int(count) == focus["end"] - focus["begin"], "Incomplete focus sample ownership"
        report["focus_sample"] = dict(**focus, real_rows=int(count), statistics={})
    for name in ("projection", "core_output", "output", "input_gradient", "projection_gradient"):
        value, expected = candidate[name], reference[name]
        if value is None or expected is None:
            report[name] = {"defined": False}
            if focus is not None:
                report["focus_sample"]["statistics"][name] = {"defined": False}
            continue
        expected = expected.index_select(0, rows)
        report[name] = _row_comparison(value, expected, group)
        if focus is not None:
            report["focus_sample"]["statistics"][name] = _row_comparison(
                value[mask], expected[mask], group
            )
    report["parameters"] = {
        name: _statistics(value, reference["parameters"][name])
        for name, value in candidate["parameters"].items()
    }
    return report


def _native_probes(observations, data, group):
    """Check reconstructed module computation against captured real-model probes."""
    starts = dict(zip(data["sample_ids"], data["padded_boundaries"]))
    rows = torch.tensor([starts[sid] + position for sid, position in data["keys"]])
    layer_index = data["gdn_layer_index"]
    name = f"layer{layer_index:02d}"
    input_records = data["records"].get(name + ".gdn_input", [])
    if not input_records and layer_index == 0:
        input_records = data["records"].get("decoder_input", [])
    assert input_records, "Capture lacks selected GDN input probes"
    sampled_input = data["gdn_input"].index_select(0, rows).reshape(rows.numel(), -1)
    assert torch.equal(
        sampled_input, input_records[0]["value"]
    ), "Full original GDN input differs from its original input probes"
    original_rows = observations["original_rows"]
    inverse = torch.full((data["gdn_input"].shape[0],), -1, dtype=torch.long)
    inverse[original_rows] = torch.arange(original_rows.numel())
    owned = inverse[rows] >= 0
    local_rows = inverse[rows[owned]]
    report = {
        "captured_cp_size": len(data["cp_ranks"]),
        "replay_cp_size": group.size(),
        "same_cp_geometry": len(data["cp_ranks"]) == group.size(),
        "full_original_output": (
            _row_comparison(
                observations["output"], data["gdn_output"].index_select(0, original_rows), group
            )
            if "gdn_output" in data
            else {"available": False}
        ),
    }
    for key, name in (("projection", name + ".input_projection"), ("output", name + ".attention")):
        measured = observations[key].index_select(0, local_rows).flatten(1)
        records = data["records"].get(name, [])
        assert records, f"Missing real-model probe {name}"
        report[name] = [
            {
                "grad_enabled": item["grad_enabled"],
                **_row_comparison(measured, item["value"][owned], group),
            }
            for item in records
        ]
    return report


def test_gdn_captured_pack_geometry():
    if os.environ.get("MIMO_GDN_CAPTURE_ISOLATION_SAMPLES"):
        pytest.skip("Companion isolation uses its separate same-geometry diagnostic")
    capture = os.environ.get("MIMO_GDN_CAPTURE")
    if not capture:
        pytest.skip("Set MIMO_GDN_CAPTURE to opt into actual-activation replay")
    if Utils.world_size != 16:
        pytest.skip("Captured Qwen CP1/CP8 replay uses the existing 16-rank allocation")
    checkpoint = os.environ["MIMO_GDN_PRETRAINED_CHECKPOINT"]
    output_path = Path(os.environ["MIMO_GDN_CAPTURE_REPORT"])
    assert not output_path.exists(), f"Refusing to overwrite {output_path}"
    extra = int(os.environ.get("MIMO_GDN_CAPTURE_EXTRA_PADDING", "4096"))
    assert extra > 0 and extra % 16 == 0
    # (name, CP size, geometry, core-only VJP, additional BF16-default control)
    cases = [
        ("cp1_repeat", 1, {}, False, False),
        ("cp8", 8, {}, False, False),
        ("cp1_extra_dummy", 1, dict(extra=extra), False, False),
        ("cp1_reverse_sequences", 1, dict(reverse=True), False, True),
        ("cp1_intersequence_padding", 1, dict(sequence_padding=16), False, True),
        ("fixed_core_cp1", 1, {}, True, False),
        ("fixed_core_cp8", 8, {}, True, False),
        ("fixed_core_cp1_extra_dummy", 1, dict(extra=extra), True, False),
        ("fixed_core_cp1_reverse_sequences", 1, dict(reverse=True), True, True),
        ("fixed_core_cp1_intersequence_padding", 1, dict(sequence_padding=16), True, True),
        ("cp4", 4, {}, False, False),
        ("fixed_core_cp4", 4, {}, True, False),
    ]
    requested = os.environ.get("MIMO_GDN_CAPTURE_CASES")
    requested = {name.strip() for name in requested.split(",")} if requested else None
    assert requested is None or requested <= {case[0] for case in cases}, requested
    layer_index = int(os.environ.get("MIMO_GDN_CAPTURE_LAYER_INDEX", "0"))
    assert layer_index >= 0
    data = _load_capture(capture, checkpoint, layer_index)
    focus_id = os.environ.get("MIMO_GDN_CAPTURE_FOCUS_SAMPLE_ID")
    focus = _focus_sample(data, json.loads(focus_id) if focus_id is not None else None)
    dtype_names = os.environ.get("MIMO_GDN_CAPTURE_DTYPES", "bf16,fp32").split(",")
    assert dtype_names and all(name in ("bf16", "fp32") for name in dtype_names)
    common_core = os.environ.get("MIMO_GDN_CAPTURE_COMMON_BF16_CORE", "0")
    assert common_core in ("0", "1")
    common_core = common_core == "1"
    assert (
        not common_core or dtype_names[0] == "bf16"
    ), "Common BF16 core source requires BF16 first"
    captured_cp = len(data["cp_ranks"])
    extra_cp_sizes = {captured_cp} - {1, 8}
    if requested is not None and requested & {"cp4", "fixed_core_cp4"}:
        extra_cp_sizes.add(4)
    digest = None
    if Utils.rank == 0:
        checksum = hashlib.sha256()
        with Path(capture).open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                checksum.update(chunk)
        digest = checksum.hexdigest()
    report = {
        "capture": str(Path(capture).resolve()),
        "capture_sha256": digest,
        "capture_sha256_scope": "selected input file; capture_peer_files lists all reconstructed peers",
        "checkpoint": checkpoint,
        "sample_ids": data["sample_ids"],
        "padded_boundaries": data["padded_boundaries"],
        "logical_boundaries": data["logical_boundaries"],
        "source_step": data["step"],
        "gdn_layer_index": layer_index,
        "captured_cp_ranks": data["cp_ranks"],
        "capture_peer_files": data.get("capture_peer_files", [str(Path(capture))]),
        "scope": "One pretrained GDN and one captured real pack; no full-model equivalence claim",
        "precision_note": "FP32 inputs and parameters are exact upcasts of native BF16 values (preserving explicitly marked FP32 parameters). Installed FLA Triton dot precision is not certified IEEE by the PyTorch TF32 switches.",
        "gradient_note": "Regular cases use the saved raw module cotangent. Fixed-projection cases use baseline core-output cotangent, excluding both projections from VJP. Parameter .grad uses parameter dtype before CP SUM; this is not native DDP FP32 main_grad validation.",
        "threshold": None,
        "numerical_acceptance": "not_evaluated; measurement completion is not a parity gate",
        "focus_sample": focus,
        "fixed_core_source": "common_bf16_cp1" if common_core else "per_dtype_cp1",
        "requested_cases": sorted(requested) if requested is not None else None,
        "completed": False,
        "cases": {},
        "per_rank": {},
    }
    Utils.initialize_model_parallel(
        1, 1, context_parallel_size=8, dynamic_context_parallel=bool(extra_cp_sizes)
    )
    try:
        model_parallel_cuda_manual_seed(2026)
        pg = ProcessGroupCollection.use_mpu_process_groups()
        groups = {1: pg.tp, 8: pg.cp}
        for cp_size in sorted(extra_cp_sizes):
            groups[cp_size] = parallel_state.get_dynamic_data_context_parallel_groups(
                group_size=cp_size
            )
        # These switches control PyTorch/cuBLAS, not arbitrary Triton tl.dot.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

        def publish():
            # The same CP1 geometry may still differ on an individual GPU.
            # Collect only the just-completed case's scalar summaries, never
            # activation tensors; all ranks enter at the same fixed case points.
            case_name = next(reversed(report["cases"]), None)
            local = {
                "rank": dist.get_rank(),
                "case_name": case_name,
                "summary": report["cases"][case_name] if case_name else None,
            }
            gathered = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
            dist.gather_object(local, gathered, dst=0)
            if gathered is not None:
                ranks = sorted(item["rank"] for item in gathered)
                assert ranks == list(range(16))
                report["rank_coverage"] = dict(expected=16, observed=ranks, complete=True)
                for item in gathered:
                    rank_report = report["per_rank"].setdefault(str(item["rank"]), {"cases": {}})
                    if item["case_name"] is not None:
                        rank_report["cases"][item["case_name"]] = item["summary"]
                output_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = output_path.with_suffix(output_path.suffix + ".tmp")
                temporary.write_text(json.dumps(report, indent=2) + "\n")
                temporary.replace(output_path)

        publish()
        common_fixed = None
        for dtype_name in dtype_names:
            dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float32
            layer = _make_layer(pg, dtype, checkpoint, layer_index)
            print(
                f"GDN_CAPTURE_START dtype={dtype_name} tokens={data['gdn_input'].shape[0]}",
                flush=True,
            )
            baseline = _run_case(layer, pg.tp, data)
            if common_core and common_fixed is None:
                common_fixed, hashes = _fixed_core_source(baseline)
                all_hashes = [None] * dist.get_world_size()
                dist.all_gather_object(all_hashes, hashes)
                assert all(
                    item == hashes for item in all_hashes
                ), "BF16 CP1 core sources differ across ranks; cannot use a common CP input"
                report["common_bf16_core_sha256"] = hashes
            report["cases"][f"{dtype_name}_cp1_reference"] = {
                "native_probes" if captured_cp == 1 else "cross_cp_capture_probes": _native_probes(
                    baseline, data, pg.tp
                ),
                "self_check": _compare(baseline, baseline, pg.tp, focus),
            }
            publish()
            selected = [
                case
                for case in cases
                if (
                    case[0] in requested
                    if requested is not None
                    else case[1] != 4 and (dtype_name == "bf16" or not case[4])
                )
            ]
            if captured_cp != 1 and not any(
                case[1] == captured_cp and not case[2] and not case[3] for case in selected
            ):
                selected.append((f"captured_cp{captured_cp}", captured_cp, {}, False, False))
            for name, cp_size, geometry, core_only, _ in selected:
                if core_only:
                    continue
                group = groups[cp_size]
                candidate = _run_case(layer, group, data, **geometry)
                report["cases"][f"{dtype_name}_{name}"] = _compare(
                    candidate, baseline, group, focus
                )
                if captured_cp != 1 and cp_size == captured_cp and not geometry:
                    report["cases"][f"{dtype_name}_{name}"]["native_probes"] = _native_probes(
                        candidate, data, group
                    )
                del candidate
                publish()
            if any(case[3] for case in selected):
                fixed = common_fixed if common_core else baseline
                fixed_reference = _run_case(layer, pg.tp, data, fixed=fixed)
                report["cases"][f"{dtype_name}_fixed_core_cp1"] = _compare(
                    fixed_reference, baseline, pg.tp, focus
                )
                publish()
                for name, cp_size, geometry, core_only, _ in selected:
                    if not core_only or name == "fixed_core_cp1":
                        continue
                    group = groups[cp_size]
                    candidate = _run_case(layer, group, data, fixed=fixed, **geometry)
                    report["cases"][f"{dtype_name}_{name}"] = _compare(
                        candidate, fixed_reference, group, focus
                    )
                    del candidate
                    publish()
                del fixed_reference
            del baseline, layer
            torch.cuda.empty_cache()
        report["completed"] = True
        publish()
        if dist.get_rank() == 0:
            print(f"GDN_CAPTURE_REPORT {output_path}", flush=True)
    finally:
        Utils.destroy_model_parallel()


def _companion_isolation_data(
    data, focus_id, companion_id, *, perturb=False, perturb_padding=False
):
    """Change only companion real inputs; keep only focus real output cotangent."""
    assert focus_id != companion_id
    focus, companion = (_focus_sample(data, sid) for sid in (focus_id, companion_id))
    value = data["gdn_input"].clone()
    cotangent = torch.zeros_like(data["gdn_output_gradient"])
    cotangent[focus["begin"] : focus["end"]] = data["gdn_output_gradient"][
        focus["begin"] : focus["end"]
    ]
    assert torch.count_nonzero(cotangent), "Focus sample has no captured output cotangent"
    if perturb:
        begin, end = companion["begin"], companion["end"]
        value[begin:end].neg_()
        assert not torch.equal(value[begin:end], data["gdn_input"][begin:end])
    if perturb_padding:
        padding = torch.ones(value.shape[0], dtype=torch.bool)
        for sample_id in data["sample_ids"]:
            sample = _focus_sample(data, sample_id)
            padding[sample["begin"] : sample["end"]] = False
        assert padding.any(), "Padding perturbation requires existing padded rows"
        value[padding] += 0.5
        assert not torch.equal(value[padding], data["gdn_input"][padding])
    return dict(data, gdn_input=value, gdn_output_gradient=cotangent)


@pytest.mark.parametrize("focus_id,companion_id", [(47, 62), (62, 47)])
def test_companion_isolation_preserves_geometry_padding_and_focus(tmp_path, focus_id, companion_id):
    path, full = _write_sharded_fixture(tmp_path)
    data = _load_capture(path, tmp_path / "iter_0000000")
    focus = _focus_sample(data, focus_id)
    companion = _focus_sample(data, companion_id)
    baseline = _companion_isolation_data(data, focus_id, companion_id)
    changed = _companion_isolation_data(data, focus_id, companion_id, perturb=True)
    rows = torch.arange(full.shape[0])
    changed_rows = _focus_rows(rows, companion)
    supervised = _focus_rows(rows, focus)
    for value in (baseline, changed):
        for key in ("padded_boundaries", "logical_boundaries", "sample_ids", "sample_lengths"):
            assert value[key] is data[key]
        assert value["gdn_input"].shape == full.shape
        assert value["gdn_input"].dtype == full.dtype
        assert torch.equal(value["gdn_output_gradient"][supervised], (full + 4)[supervised])
        assert not torch.count_nonzero(value["gdn_output_gradient"][~supervised])
    assert torch.equal(baseline["gdn_input"], full)
    assert torch.equal(changed["gdn_input"][changed_rows], -full[changed_rows])
    assert torch.equal(changed["gdn_input"][~changed_rows], full[~changed_rows])
    assert torch.equal(data["gdn_input"], full), "The capture itself must remain immutable"
    assert torch.equal(data["gdn_output_gradient"], full + 4)
    with pytest.raises(AssertionError):
        _companion_isolation_data(data, focus_id, focus_id)


def _compare_companion_isolation(candidate, baseline, group, focus, companion):
    """Compare identical rank-local ownership; do not remap to a CP1 reference."""
    rows = candidate["original_rows"]
    assert torch.equal(rows, baseline["original_rows"])
    assert candidate["geometry"] == baseline["geometry"]
    mask = _focus_rows(rows, focus)
    companion_mask = _focus_rows(rows, companion)
    counts = torch.tensor([int(mask.sum()), int(companion_mask.sum())], device="cuda")
    dist.all_reduce(counts, group=group)
    assert counts.tolist() == [focus["end"] - focus["begin"], companion["end"] - companion["begin"]]
    focus_metrics = {
        name: _row_comparison(candidate[name][mask], baseline[name][mask], group)
        for name in ("projection", "core_output", "output", "input_gradient", "projection_gradient")
    }
    parameters = {
        name: _row_comparison(value, baseline["parameters"][name], group)
        for name, value in candidate["parameters"].items()
    }
    # All nonfocus original rows include companion tokens and original padding.
    # They must have zero input VJP when independent sequences are truly isolated.
    nonfocus = candidate["input_gradient"][~mask]
    companion_vjp = candidate["input_gradient"][companion_mask]
    nonfocus_zero = _row_comparison(nonfocus, torch.zeros_like(nonfocus), group)
    companion_zero = _row_comparison(companion_vjp, torch.zeros_like(companion_vjp), group)
    checks = [*focus_metrics.values(), *parameters.values(), nonfocus_zero, companion_zero]
    finite = torch.tensor(
        int(
            all(
                torch.isfinite(value).all()
                for value in candidate.values()
                if isinstance(value, torch.Tensor)
            )
        ),
        device="cuda",
    )
    dist.all_reduce(finite, op=dist.ReduceOp.MIN, group=group)
    return dict(
        reported_cp_ranks=dist.get_process_group_ranks(group),
        geometry=candidate["geometry"],
        focus_sample=dict(**focus, real_rows=int(counts[0]), statistics=focus_metrics),
        companion_sample=dict(**companion, real_rows=int(counts[1])),
        parameters=parameters,
        all_nonfocus_input_vjp_zero=nonfocus_zero,
        companion_input_vjp_zero=companion_zero,
        all_finite=bool(finite) and all(item["all_finite"] for item in checks),
        real_effect_exact=bool(finite)
        and all(
            item["exact"] and item["all_finite"]
            for item in [*focus_metrics.values(), *parameters.values()]
        ),
        exact_isolation_pass=bool(finite)
        and all(item["exact"] and item["all_finite"] for item in checks),
    )


def test_gdn_captured_companion_isolation():
    selection = os.environ.get("MIMO_GDN_CAPTURE_ISOLATION_SAMPLES")
    if not selection:
        pytest.skip("Set MIMO_GDN_CAPTURE_ISOLATION_SAMPLES for same-shape boundary isolation")
    assert Utils.world_size == 16, "Captured isolation uses the existing 16-rank allocation"
    sample_ids = [int(value) for value in selection.split(",")]
    assert len(sample_ids) == len(set(sample_ids)) == 2
    assert os.environ.get("MIMO_GDN_CAPTURE_DTYPES", "bf16") == "bf16"
    capture, checkpoint = (
        os.environ["MIMO_GDN_CAPTURE"],
        os.environ["MIMO_GDN_PRETRAINED_CHECKPOINT"],
    )
    layer_index = int(os.environ.get("MIMO_GDN_CAPTURE_LAYER_INDEX", "0"))
    output_path = Path(os.environ["MIMO_GDN_CAPTURE_REPORT"])
    assert not output_path.exists(), f"Refusing to overwrite {output_path}"
    data = _load_capture(capture, checkpoint, layer_index)
    assert len(data["cp_ranks"]) == 4, "This control replays the captured CP4 pack and CP1"
    report = dict(
        capture=str(Path(capture).resolve()),
        capture_peer_files=data["capture_peer_files"],
        checkpoint=checkpoint,
        gdn_layer_index=layer_index,
        sample_ids=data["sample_ids"],
        padded_boundaries=data["padded_boundaries"],
        logical_boundaries=data["logical_boundaries"],
        dtype="bf16",
        perturbation=(
            "Negate companion real inputs; optional separate +0.5 perturbation of all existing padding"
            if os.environ.get("MIMO_GDN_CAPTURE_PERTURB_PADDING") == "1"
            else "Negate only companion real input rows; preserve every padding row"
        ),
        cotangent="Captured focus real-token output cotangent; zero on all other rows",
        scope="Same-CP baseline/repeat/perturb; no changes to shape, boundaries, order or weights",
        gradient_note="Standalone parameter-dtype .grad then FP32 CP SUM; not native DDP main_grad",
        causality_note="Both directions are tested; perturbing only the later sample cannot reveal causal state leakage into it",
        numerical_acceptance="Exact and finite outcomes are reported without a tolerance",
        completed=False,
        per_rank={},
    )
    Utils.initialize_model_parallel(1, 1, context_parallel_size=8, dynamic_context_parallel=True)
    try:
        model_parallel_cuda_manual_seed(2026)
        pg = ProcessGroupCollection.use_mpu_process_groups()
        groups = {
            4: parallel_state.get_dynamic_data_context_parallel_groups(group_size=4),
            1: pg.tp,
        }
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        layer = _make_layer(pg, torch.bfloat16, checkpoint, layer_index)

        def publish(case=None, summary=None):
            payload = dict(rank=dist.get_rank(), case=case, summary=summary)
            gathered = [None] * 16 if dist.get_rank() == 0 else None
            dist.gather_object(payload, gathered, dst=0)
            if gathered is not None:
                assert sorted(item["rank"] for item in gathered) == list(range(16))
                for item in gathered:
                    if item["case"] is not None:
                        report["per_rank"].setdefault(str(item["rank"]), {})[case] = item["summary"]
                report["exact_isolation_all_pass"] = bool(report["completed"]) and all(
                    item["exact_isolation_pass"]
                    for rank_cases in report["per_rank"].values()
                    for item in rank_cases.values()
                )
                report["real_effect_all_exact"] = bool(report["completed"]) and all(
                    item["real_effect_exact"]
                    for rank_cases in report["per_rank"].values()
                    for item in rank_cases.values()
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = output_path.with_suffix(output_path.suffix + ".tmp")
                temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                temporary.replace(output_path)

        publish()
        for cp_size, group in groups.items():
            for focus_id, companion_id in (sample_ids, sample_ids[::-1]):
                focus, companion = (_focus_sample(data, sid) for sid in (focus_id, companion_id))
                isolated = _companion_isolation_data(data, focus_id, companion_id)
                baseline = _run_case(layer, group, isolated)
                label = f"bf16_cp{cp_size}_focus{focus_id}_companion{companion_id}"
                publish(
                    label + "_baseline",
                    _compare_companion_isolation(baseline, baseline, group, focus, companion),
                )
                controls = [("repeat", {}), ("perturb", {"perturb": True})]
                if os.environ.get("MIMO_GDN_CAPTURE_PERTURB_PADDING") == "1":
                    controls.append(("padding", {"perturb_padding": True}))
                for name, perturbation in controls:
                    candidate_data = _companion_isolation_data(
                        data, focus_id, companion_id, **perturbation
                    )
                    candidate = _run_case(layer, group, candidate_data)
                    publish(
                        label + "_" + name,
                        _compare_companion_isolation(candidate, baseline, group, focus, companion),
                    )
                    del candidate, candidate_data
                del baseline, isolated
                torch.cuda.empty_cache()
        report["completed"] = True
        publish()
        if dist.get_rank() == 0:
            print(f"GDN_ISOLATION_REPORT {output_path}", flush=True)
    finally:
        Utils.destroy_model_parallel()
