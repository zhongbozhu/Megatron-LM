# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Full Qwen3.5 GDN head geometry at runtime CP1/8/16 (requires 16 GPUs).

The exact permutation/adjoint check is separate from BF16 kernel tolerance.
The isolated two-layer comparison excludes MoE, vision, and rotary attention.
"""

import json
import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformer_engine.pytorch  # noqa: F401; loads the binary extension search path
import transformer_engine_torch as tex

from megatron.core import dist_checkpointing
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_gated_delta_net_module_spec,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net.common import (
    a2a_cp_to_hp,
    a2a_hp_to_cp,
    get_parameter_local_cp,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _packing(group, padded, actual=None):
    cu = torch.tensor(padded, dtype=torch.int32, device="cuda")
    real = cu if actual is None else torch.tensor(actual, dtype=torch.int32, device="cuda")
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=real,
        cu_seqlens_kv=real,
        cu_seqlens_q_padded=cu,
        cu_seqlens_kv_padded=cu,
        max_seqlen_q=max(b - a for a, b in zip(padded, padded[1:])),
        max_seqlen_kv=max(b - a for a, b in zip(padded, padded[1:])),
        cp_group=group,
        local_cp_size=group.size(),
    )


def _indices(packing):
    total = int(packing.cu_seqlens_q_padded[-1])
    group = packing.cp_group
    if group.size() == 1:
        return torch.arange(total, device="cuda")
    return tex.thd_get_partitioned_indices(
        packing.cu_seqlens_q_padded, total, group.size(), group.rank()
    ).long()


def _columns(sections, group):
    # Independent expected layout: contiguous heads in each original section.
    offset, result = 0, []
    for section in sections:
        width = section // group.size()
        result.extend(range(offset + group.rank() * width, offset + (group.rank() + 1) * width))
        offset += section
    return torch.tensor(result, device="cuda")


def _exact_layout_and_adjoint(group):
    packing = _packing(group, [0, 96, 256])
    rows = _indices(packing)
    sections = (2048, 2048, 4096, 4096, 32, 32)
    full = torch.arange(256 * sum(sections), dtype=torch.float64, device="cuda")
    full = full.reshape(256, 1, -1)
    probe = full.remainder(31) - 15
    local = full[rows].detach().requires_grad_()
    hp, inverse = a2a_cp_to_hp(
        local, sections, group.size(), group, packing.cu_seqlens_q_padded, 256, packing
    )
    columns = _columns(sections, group)
    torch.testing.assert_close(hp, full[:, :, columns], atol=0, rtol=0)
    hp.backward(probe[:, :, columns])
    torch.testing.assert_close(local.grad, probe[rows], atol=0, rtol=0)

    # Output path has only V heads, in natural head order.
    values, value_probe = full[:, :, :4096], probe[:, :, :4096]
    columns = _columns((4096,), group)
    head_local = values[:, :, columns].detach().requires_grad_()
    cp_output = a2a_hp_to_cp(head_local, group.size(), group, packing, inverse)
    torch.testing.assert_close(cp_output, values[rows], atol=0, rtol=0)
    cp_output.backward(value_probe[rows])
    torch.testing.assert_close(head_local.grad, value_probe[:, :, columns], atol=0, rtol=0)

    # Conv parameter selection must use the same Q/K/V ownership, including its adjoint.
    weight = torch.arange(8192 * 4, dtype=torch.float64, device="cuda").reshape(8192, 1, 4)
    weight.requires_grad_()
    sliced = get_parameter_local_cp(weight, 0, group, [2048, 2048, 4096])
    columns = _columns((2048, 2048, 4096), group)
    torch.testing.assert_close(sliced, weight[columns], atol=0, rtol=0)
    sliced.square().sum().backward()
    dist.all_reduce(weight.grad, group=group)
    torch.testing.assert_close(weight.grad, weight.detach() * 2, atol=0, rtol=0)


def _statistics(candidate, reference, group=None):
    a, b = candidate.double(), reference.double()
    stats = torch.stack(((a - b).square().sum(), b.square().sum(), (a - b).abs().max()))
    if group is not None:
        dist.all_reduce(stats[:2], group=group)
        dist.all_reduce(stats[2:], op=dist.ReduceOp.MAX, group=group)
    return dict(
        relative_l2=float((stats[0] / stats[1].clamp_min(1e-30)).sqrt()), max_abs=float(stats[2])
    )


def _load_pretrained_layers(layers, pg, checkpoint, layer_indices=None):
    """Use native GDN checkpoint factories; request only these two modules."""
    state = {}
    layer_indices = list(range(len(layers))) if layer_indices is None else list(layer_indices)
    assert len(layer_indices) == len(layers)
    for i, layer in zip(layer_indices, layers):
        state.update(
            layer.sharded_state_dict(
                prefix=f"language_model.decoder.layers.{i}.self_attention.",
                metadata={"dp_cp_group": pg.dp_cp},
            )
        )
    # Reject requested keys absent from the checkpoint. Unrequested model keys
    # are intentionally ignored because this is a two-module subset diagnostic.
    loaded = dist_checkpointing.load(
        {"model": state}, checkpoint, strict="raise_unexpected", validate_access_integrity=True
    )["model"]
    for i, layer in zip(layer_indices, layers):
        prefix = f"language_model.decoder.layers.{i}.self_attention."
        layer.load_state_dict(
            {key[len(prefix) :]: value for key, value in loaded.items() if key.startswith(prefix)},
            strict=True,
        )


@pytest.mark.parametrize(
    "long_tokens",
    [0]
    + (
        [int(os.environ["MIMO_GDN_LONG_TOKENS"])]
        if int(os.environ.get("MIMO_GDN_LONG_TOKENS", "0")) > 0
        else []
    ),
    ids=lambda length: f"long_{length}" if length else "uneven_4352",
)
def test_qwen35_asymmetric_gdn_runtime_cp_geometry(long_tokens):
    if Utils.world_size != 16:
        pytest.skip("Qwen3.5 CP8/16 geometry requires exactly 16 ranks")
    Utils.initialize_model_parallel(1, 1, context_parallel_size=8)
    try:
        model_parallel_cuda_manual_seed(2026)
        torch.manual_seed(2026)
        pg = ProcessGroupCollection.use_mpu_process_groups()
        groups = (pg.tp, pg.cp, dist.group.WORLD)
        for group in groups[1:]:
            _exact_layout_and_adjoint(group)
        if dist.get_rank() == 0:
            print("QWEN35_GDN_EXACT_LAYOUT_AND_ADJOINT_PASS CP8 CP16", flush=True)

        config = TransformerConfig(
            num_layers=2,
            hidden_size=2048,
            num_attention_heads=16,
            num_query_groups=2,
            normalization="RMSNorm",
            layernorm_epsilon=1e-6,
            layernorm_zero_centered_gamma=True,
            params_dtype=torch.bfloat16,
            bf16=True,
            gradient_accumulation_fusion=False,
            context_parallel_size=8,
            activation_func=F.silu,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            experimental_attention_variant="gdn",
            linear_attention_freq=[1, 1],
            linear_num_key_heads=16,
            linear_num_value_heads=32,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
            transformer_impl="transformer_engine",
        )
        spec = get_gated_delta_net_module_spec(config)
        layers = (
            torch.nn.ModuleList(
                [
                    spec.module(
                        config,
                        submodules=spec.submodules,
                        layer_number=i + 1,
                        bias=False,
                        conv_bias=False,
                        conv_init=0.1,
                        use_qk_l2norm=True,
                        A_init_range=(1, 16),
                        pg_collection=pg,
                    )
                    for i in range(2)
                ]
            )
            .cuda()
            .bfloat16()
        )
        checkpoint = os.environ.get("MIMO_GDN_PRETRAINED_CHECKPOINT")
        if checkpoint:
            _load_pretrained_layers(layers, pg, checkpoint)
        for parameter in layers.parameters():
            dist.broadcast(parameter.data, src=0)
        torch.manual_seed(2027)
        if long_tokens:
            assert long_tokens >= 128 and long_tokens % 32 == 0
            padded, actual = [0, long_tokens], [0, long_tokens - 4]
        else:
            padded, actual = [0, 96, 4352], [0, 77, 4200]
        inputs = torch.randn(padded[-1], 1, 2048, device="cuda", dtype=torch.bfloat16)
        probe = torch.randn_like(inputs)
        for i in range(len(actual) - 1):
            probe[padded[i] + actual[i + 1] - actual[i] : padded[i + 1]] = 0
        reference = None
        reports = []
        for group in groups:
            if dist.get_rank() == 0:
                print(f"QWEN35_GDN_START CP{group.size()} tokens={padded[-1]}", flush=True)
            layers.zero_grad(set_to_none=True)
            packing = _packing(group, padded, actual)
            rows = _indices(packing)
            local_input = inputs[rows].detach().requires_grad_()
            output = local_input
            for layer in layers:
                update, bias = layer(output, None, packed_seq_params=packing)
                assert bias is None
                output = output + update
            (output.float() * probe[rows].float()).sum().div(inputs.numel()).backward()
            gradients = {}
            for name, parameter in layers.named_parameters():
                assert parameter.grad is not None, name
                grad = parameter.grad.float()
                dist.all_reduce(grad, group=group)
                gradients[name] = grad.cpu()
            if reference is None:
                reference = dict(
                    output=output.detach().cpu(),
                    input_grad=local_input.grad.cpu(),
                    gradients=gradients,
                )
                continue
            report = dict(
                cp=group.size(),
                pretrained_checkpoint=checkpoint,
                padded_boundaries=padded,
                actual_boundaries=actual,
                exact_layout_and_adjoint=True,
                output=_statistics(output.detach(), reference["output"][rows.cpu()].cuda(), group),
                input_gradient=_statistics(
                    local_input.grad, reference["input_grad"][rows.cpu()].cuda(), group
                ),
                parameters={
                    name: _statistics(grad, reference["gradients"][name])
                    for name, grad in gradients.items()
                },
            )
            reports.append(report)
        if dist.get_rank() == 0:
            print("QWEN35_GDN_CP_GEOMETRY " + json.dumps(reports), flush=True)
        # Preserve a declared threshold; failures provide diagnostics, not a relaxed pass.
        for report in reports:
            for name, stats in {
                "output": report["output"],
                "input_gradient": report["input_gradient"],
                **report["parameters"],
            }.items():
                assert stats["relative_l2"] < 0.02, (report["cp"], name, stats)
    finally:
        Utils.destroy_model_parallel()
