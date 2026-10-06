# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compare the complete pretrained native Qwen3.5 vision tower with Hugging Face.

Run through torch.distributed.run. Only visual weights are read from the existing
VL checkpoint; no language model or optimizer is instantiated by this check.
"""

import argparse
import inspect
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeVisionModel

from megatron.core import dist_checkpointing, parallel_state
from megatron.core.models.vision.qwen35_vit import Qwen35VisionModel, qwen35_vision_config
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig


class _ReferencePackedAttention(torch.nn.Module):
    """Diagnostic SDPA only; production uses the unchanged TE attention."""

    def forward(self, query, key, value, attention_mask, *, packed_seq_params, **kwargs):
        boundaries = packed_seq_params.cu_seqlens_q.tolist()
        outputs = []
        for start, end in zip(boundaries, boundaries[1:]):
            q, k, v = [tensor[start:end].transpose(0, 1) for tensor in (query, key, value)]
            outputs.append(
                torch.nn.functional.scaled_dot_product_attention(q, k, v).transpose(0, 1)
            )
        return torch.cat(outputs)


class _UnfusedBiasLinear(torch.nn.Module):
    """Diagnostic: match Megatron's separate BF16 bias addition in the HF tower."""

    def __init__(self, linear):
        super().__init__()
        self.weight, self.bias = linear.weight, linear.bias

    def forward(self, hidden_states):
        return torch.nn.functional.linear(hidden_states, self.weight) + self.bias


def _hf_state(native, *, gradients=False):
    """Reverse the checkpoint's visual QKV interleaving and module name mapping."""
    result = {}
    replacements = {
        "self_attention.linear_qkv.layer_norm_weight": "norm1.weight",
        "self_attention.linear_qkv.layer_norm_bias": "norm1.bias",
        "mlp.linear_fc1.layer_norm_weight": "norm2.weight",
        "mlp.linear_fc1.layer_norm_bias": "norm2.bias",
        "self_attention.linear_proj": "attn.proj",
        "self_attention.linear_qkv": "attn.qkv",
        "merger.patch_norm": "merger.norm",
    }
    values = (
        {name: parameter.grad for name, parameter in native.named_parameters()}
        if gradients
        else native.state_dict()
    )
    for name, value in values.items():
        if value is None:
            raise AssertionError(f"Missing vision gradient: {name}")
        if name.endswith("_extra_state"):
            continue
        name = name.replace("decoder.layers.", "blocks.")
        for source, target in replacements.items():
            name = name.replace(source, target)
        if ".attn.qkv." in name:
            heads = native.config.num_attention_heads
            dim = native.config.kv_channels
            value = (
                value.reshape(heads, 3, dim, *value.shape[1:]).transpose(0, 1).reshape(value.shape)
            )
        result[name] = value
    return result


def _error(actual, expected):
    actual, expected = actual.float(), expected.float()
    difference = actual - expected
    return {
        "relative_l2": (difference.norm() / expected.norm().clamp_min(1e-30)).item(),
        "max_abs": difference.abs().max().item(),
        "difference_squared_l2": difference.square().sum(dtype=torch.float64).item(),
        "reference_squared_l2": expected.square().sum(dtype=torch.float64).item(),
        "exact": torch.equal(actual, expected),
        "all_finite": bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
    }


def _snapshot(module, output, inputs):
    parameters = (
        _hf_state(module, gradients=True)
        if isinstance(module, Qwen35VisionModel)
        else {name: parameter.grad for name, parameter in module.named_parameters()}
    )
    assert all(value is not None for value in parameters.values()), "Missing parameter gradients"
    return {
        "output": output.detach().float().cpu(),
        "input_gradient": inputs.grad.detach().float().cpu(),
        "parameters": {name: value.detach().float().cpu() for name, value in parameters.items()},
    }


def _compare_snapshots(actual, expected):
    assert actual["parameters"].keys() == expected["parameters"].keys()
    parameters = {
        name: _error(value, expected["parameters"][name])
        for name, value in actual["parameters"].items()
    }
    difference = sum(value["difference_squared_l2"] for value in parameters.values())
    reference = sum(value["reference_squared_l2"] for value in parameters.values())
    return {
        "output": _error(actual["output"], expected["output"]),
        "input_gradient": _error(actual["input_gradient"], expected["input_gradient"]),
        "parameter_gradient": {
            "relative_l2": (difference / max(reference, 1e-60)) ** 0.5,
            "max_abs": max(value["max_abs"] for value in parameters.values()),
            "exact": all(value["exact"] for value in parameters.values()),
            "all_finite": all(value["all_finite"] for value in parameters.values()),
            "tensor_count": len(parameters),
        },
        "per_parameter": parameters,
    }


def _run_snapshot(module, patches, grids, target, *, single=False):
    module.zero_grad(set_to_none=True)
    inputs = patches.detach().clone().requires_grad_(True)
    if single:
        lengths = grids.prod(dim=1).tolist()
        outputs = [module(piece, grid[None]) for piece, grid in zip(inputs.split(lengths), grids)]
        output = torch.cat(
            [item if isinstance(item, torch.Tensor) else item.pooler_output for item in outputs]
        )
    else:
        output = module(inputs, grids)
        if not isinstance(output, torch.Tensor):
            output = output.pooler_output
    (output.float() * target.float()).sum().backward()
    return _snapshot(module, output, inputs)


def _full_suite(native, reference, patches, grids, target, actual, expected, reference_patches):
    """Same-weight controls; preserve the production attention/backend in native runs."""
    baseline = _snapshot(native, actual, patches)
    hf_baseline = _snapshot(reference, expected, reference_patches)
    comparisons = {"native_vs_hf_packed": _compare_snapshots(baseline, hf_baseline)}
    for name, model, single, expected_snapshot in (
        ("native_repeat", native, False, baseline),
        ("hf_repeat", reference, False, hf_baseline),
        ("hf_single_vs_packed", reference, True, hf_baseline),
    ):
        candidate = _run_snapshot(model, patches, grids, target, single=single)
        comparisons[name] = _compare_snapshots(candidate, expected_snapshot)
        del candidate
    recompute = native.config.recompute_granularity
    native.config.recompute_granularity = None
    try:
        for name, single in (
            ("native_recompute_off", False),
            ("native_single_recompute_off", True),
        ):
            candidate = _run_snapshot(native, patches, grids, target, single=single)
            comparisons[name] = _compare_snapshots(candidate, baseline)
            del candidate
    finally:
        native.config.recompute_granularity = recompute
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--fp32-reference-attention", action="store_true")
    parser.add_argument("--reference-attention", action="store_true")
    parser.add_argument("--reference-unfused-bias", action="store_true")
    parser.add_argument("--fp32-baseline", action="store_true")
    parser.add_argument("--full-suite", action="store_true")
    parser.add_argument("--image-sizes", help="Optional WIDTHxHEIGHT per image, comma separated")
    args = parser.parse_args()

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    parallel_state.initialize_model_parallel()
    model_parallel_cuda_manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dtype = torch.float32 if args.fp32_reference_attention else torch.bfloat16
    pg = ProcessGroupCollection.use_mpu_process_groups()
    language = TransformerConfig(
        num_layers=40,
        hidden_size=2048,
        num_attention_heads=16,
        params_dtype=dtype,
        bf16=dtype == torch.bfloat16,
        gradient_accumulation_fusion=False,
        attention_backend=AttnBackend.fused,
    )
    native = Qwen35VisionModel(qwen35_vision_config(language, recompute=True), 2048, pg).cuda()
    if args.fp32_reference_attention or args.reference_attention:
        for layer in native.decoder.layers:
            layer.self_attention.core_attention = _ReferencePackedAttention()
    state = native.sharded_state_dict(prefix="vision_model.", metadata={"dp_cp_group": pg.dp_cp})
    loaded = dist_checkpointing.load({"model": state}, args.checkpoint, strict="raise_unexpected")
    native.load_state_dict({k.removeprefix("vision_model."): v for k, v in loaded["model"].items()})
    config = AutoConfig.from_pretrained(args.hf_model, local_files_only=True).vision_config
    config._attn_implementation = "sdpa"
    reference = Qwen3_5MoeVisionModel(config).cuda()
    # Casting a completed HF module also rounds nonpersistent RoPE buffers.
    # Preserve the FP32 frequency table used by pretrained initialization;
    # otherwise this harness compares different rotary frequencies in BF16.
    rotary_inv_freq = reference.rotary_pos_emb.inv_freq.clone()
    reference = reference.to(dtype=dtype)
    reference.rotary_pos_emb.inv_freq = rotary_inv_freq
    reference.load_state_dict(_hf_state(native), strict=True)
    if args.reference_unfused_bias:
        for block in reference.blocks:
            block.attn.proj = _UnfusedBiasLinear(block.attn.proj)
            block.mlp.linear_fc1 = _UnfusedBiasLinear(block.mlp.linear_fc1)
            block.mlp.linear_fc2 = _UnfusedBiasLinear(block.mlp.linear_fc2)
    processor = AutoProcessor.from_pretrained(args.hf_model, local_files_only=True)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    sizes = (
        [tuple(map(int, item.split("x"))) for item in args.image_sizes.split(",")]
        if args.image_sizes
        else None
    )
    images = [
        Image.open(manifest_path.parent / sample["image_paths"][0]).convert("RGB")
        for sample in manifest["samples"][: len(sizes) if sizes else 2]
    ]
    if sizes:
        images = [image.resize(size) for image, size in zip(images, sizes)]
    processed = processor.image_processor(images=images, return_tensors="pt")
    patches = processed["pixel_values"].cuda().to(dtype=dtype).requires_grad_(True)
    reference_patches = patches.detach().clone().requires_grad_(True)
    grids = processed["image_grid_thw"].cuda()
    stage_outputs = {"native": {}, "reference": {}}
    if args.diagnose:
        if dist.get_rank() == 0:
            print(inspect.getsource(type(reference.merger)), flush=True)
            print(inspect.getsource(type(reference.blocks[0])), flush=True)
            print(inspect.getsource(reference.fast_pos_embed_interpolate), flush=True)
            print(inspect.getsource(reference.forward), flush=True)

        def capture(which, name, *, qkv=False):
            def hook(module, inputs, output):
                if isinstance(output, tuple):
                    output, bias = output
                    if isinstance(bias, torch.Tensor):
                        output = output + bias
                if qkv:
                    output = output.reshape(-1, 16, 3, 72).transpose(1, 2).flatten(1)
                stage_outputs[which][name] = (
                    output.detach().reshape(-1, output.shape[-1]).float().cpu()
                )

            return hook

        for index, (actual_layer, expected_layer) in enumerate(
            zip(native.decoder.layers, reference.blocks)
        ):
            actual_layer.register_forward_hook(capture("native", str(index)))
            expected_layer.register_forward_hook(capture("reference", str(index)))
        pairs = [
            (
                native.decoder.layers[0].self_attention.linear_qkv,
                reference.blocks[0].attn.qkv,
                "first_qkv",
            ),
            (
                native.decoder.layers[0].self_attention.linear_proj,
                reference.blocks[0].attn.proj,
                "first_proj",
            ),
            (
                native.decoder.layers[0].mlp.linear_fc1,
                reference.blocks[0].mlp.linear_fc1,
                "first_fc1",
            ),
            (
                native.decoder.layers[0].mlp.linear_fc2,
                reference.blocks[0].mlp.linear_fc2,
                "first_fc2",
            ),
        ]
        for actual_module, expected_module, name in pairs:
            actual_module.register_forward_hook(capture("native", name, qkv=name == "first_qkv"))
            expected_module.register_forward_hook(capture("reference", name))
    actual = native(patches, grids)
    expected = reference(reference_patches, grids)
    if not isinstance(expected, torch.Tensor):
        expected = expected.pooler_output
    target_generator = torch.Generator(device=actual.device).manual_seed(20261002)
    target = torch.randn(actual.shape, device=actual.device, generator=target_generator)
    # Both precision runs use the same exactly representable output cotangent.
    target = target.bfloat16().to(dtype)
    (actual.float() * target.float()).sum().backward()
    (expected.float() * target.float()).sum().backward()

    def error(actual, expected):
        difference = actual.float() - expected.float()
        return {
            "relative_l2": (difference.norm() / expected.float().norm().clamp_min(1e-8)).item(),
            "max_abs": difference.abs().max().item(),
        }

    metrics = {
        "layers": len(native.decoder.layers),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "manifest": str(manifest_path.resolve()),
        "hf_model": str(Path(args.hf_model).resolve()),
        "input_shape": list(patches.shape),
        "parameter_dtype": str(dtype),
        "cotangent_seed": 20261002,
        "requested_image_sizes": sizes,
        "parameters": sum(p.numel() for p in native.parameters()),
        "vision_rows": actual.shape[0],
        "output": error(actual, expected),
        "input_gradient": error(patches.grad, reference_patches.grad),
        "native_recompute": True,
        "fp32_reference_attention": args.fp32_reference_attention,
        "reference_attention": args.reference_attention,
        "reference_unfused_bias": args.reference_unfused_bias,
        "tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        "reference": "transformers Qwen3_5MoeVisionModel, SDPA, same pretrained weights",
    }
    if args.diagnose:
        metrics["layer_outputs"] = {
            name: error(value, stage_outputs["reference"][name])
            for name, value in stage_outputs["native"].items()
        }
        native_positions, _ = native._position_embeddings(grids, grids.device)
        metrics["learned_positions"] = error(
            native_positions, reference.fast_pos_embed_interpolate(grids).to(dtype)
        )
    if args.full_suite:
        metrics["suite"] = _full_suite(
            native, reference, patches, grids, target, actual, expected, reference_patches
        )
        metrics["image_grid_thw"] = grids.tolist()
    if args.fp32_baseline:
        reference.zero_grad(set_to_none=True)
        reference.float()
        fp32_patches = patches.detach().float().requires_grad_(True)
        fp32_output = reference(fp32_patches, grids)
        if not isinstance(fp32_output, torch.Tensor):
            fp32_output = fp32_output.pooler_output
        (fp32_output * target.float()).sum().backward()
        metrics["native_vs_fp32_output"] = error(actual, fp32_output)
        metrics["hf_vs_fp32_output"] = error(expected, fp32_output)
        metrics["native_vs_fp32_input_gradient"] = error(patches.grad, fp32_patches.grad)
        metrics["hf_vs_fp32_input_gradient"] = error(reference_patches.grad, fp32_patches.grad)
    print(json.dumps(metrics), flush=True)
    if dist.get_rank() == 0:
        Path(args.output).write_text(json.dumps(metrics, indent=2) + "\n")
    assert metrics["output"]["relative_l2"] < 0.03, metrics
    assert metrics["input_gradient"]["relative_l2"] < 0.05, metrics
    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
