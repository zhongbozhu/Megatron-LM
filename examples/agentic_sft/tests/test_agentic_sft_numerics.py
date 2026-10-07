# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Bias-free dense BF16 GPT reference for the agentic SFT packing boundary.

Run with one GPU through torch.distributed.run. This uses TE attention, actual
model backward passes and a real optimizer step; no model or kernel is mocked.
Independent trajectory gradients are accumulated in FP32 after BF16 backward;
this does not exercise DDP fused main_grad accumulation. The small dense model
does not establish full Qwen/GDN/MoE/MTP/CP gradient equivalence.
"""

import json
import os

import pytest
import torch

from examples.agentic_sft.packing import make_packed_row
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.datasets.packed_sft_dataset import packed_row_to_tensors
from tests.unit_tests.test_utilities import Utils

# BF16 backward reductions differ between one packed GEMM and three independent
# GEMMs. Check both pure maximum-absolute and global relative-L2 error bounds.
LOSS_RTOL = 2e-3
LOSS_ATOL = 2e-2
GRAD_RELATIVE_L2_TOL = 4e-2
GRAD_ABSOLUTE_TOL = 5e-4
STEP_RELATIVE_L2_TOL = 4e-2
STEP_ABSOLUTE_TOL = 5e-5
LEARNING_RATE = 0.1


def _samples():
    """Three unequal trajectories, two assistant spans each, real EOS == pad."""
    generator = torch.Generator().manual_seed(917)
    samples = []
    for runtime_length in (31, 47, 63):
        tokens = torch.randint(2, 256, (runtime_length + 1,), generator=generator).tolist()
        tokens[-1] = 1
        targets = [-100] * len(tokens)
        for start, end in ((4, len(tokens) // 2), (3 * len(tokens) // 4, len(tokens))):
            targets[start:end] = tokens[start:end]
        samples.append({"input_ids": tokens, "targets": targets})
    return samples


def _model():
    config = TransformerConfig(
        num_layers=2,
        hidden_size=128,
        ffn_hidden_size=256,
        num_attention_heads=2,
        kv_channels=64,
        params_dtype=torch.bfloat16,
        bf16=True,
        use_cpu_initialization=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        add_bias_linear=False,
        add_qkv_bias=False,
        attention_backend=AttnBackend.fused,
        gradient_accumulation_fusion=False,
        masked_softmax_fusion=False,
        bias_activation_fusion=False,
        bias_dropout_fusion=False,
        apply_rope_fusion=False,
        calculate_per_token_loss=True,
    )
    return (
        GPTModel(
            config=config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=256,
            max_sequence_length=256,
            position_embedding_type="rope",
            share_embeddings_and_output_weights=False,
            parallel_output=False,
        )
        .cuda()
        .train()
    )


def _token_losses(model, tokens, labels, positions, packed_seq_params=None):
    # Padding/ignored positions are masked after CE. Give CE only valid IDs,
    # so the test does not rely on a particular CE kernel's ignore-index support.
    return (
        model(
            input_ids=tokens.unsqueeze(0),
            position_ids=positions.unsqueeze(0),
            attention_mask=None,
            labels=labels.clamp_min(0).unsqueeze(0),
            packed_seq_params=packed_seq_params,
        )
        .float()
        .reshape(-1)
    )


def _errors(actual, expected):
    difference = actual.float() - expected.float()
    return {
        "max_abs": difference.abs().max().item(),
        "relative_l2": (difference.norm() / expected.float().norm().clamp_min(1e-12)).item(),
    }


def _master_step(model, gradients):
    """Use FP32 optimizer masters for real BF16 model gradients, as in training."""
    masters = [torch.nn.Parameter(param.detach().float().clone()) for param in model.parameters()]
    optimizer = torch.optim.SGD(masters, lr=LEARNING_RATE)
    before = [param.detach().clone() for param in masters]
    for master, gradient in zip(masters, gradients, strict=True):
        assert gradient.dtype == torch.float32
        master.grad = gradient.detach().clone()
    optimizer.step()
    updates = torch.cat(
        [
            (param.detach() - initial).reshape(-1)
            for param, initial in zip(masters, before, strict=True)
        ]
    )
    assert updates.norm().item() > 0, "The optimizer must perform a nonzero update"
    return torch.cat([param.detach().reshape(-1) for param in masters]), updates


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a GPU and TE fused attention")
def test_bf16_packed_matches_trajectories_loss_gradients_and_optimizer_step(record_property):
    """Compare production pack/read adapters to independently shifted trajectories."""
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("Run this numerical reference with torchrun --nproc-per-node=1")
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    try:
        # Suite fixtures disable fused attention; THD requires a packed-capable backend.
        os.environ["NVTE_FUSED_ATTN"] = "1"
        os.environ["NVTE_FLASH_ATTN"] = "0"
        os.environ["NVTE_UNFUSED_ATTN"] = "0"
        torch.manual_seed(1234)
        model_parallel_cuda_manual_seed(1234)
        reference = _model()
        packed_model = _model()
        packed_model.load_state_dict(reference.state_dict())
        samples = _samples()
        supervised_count = sum(
            target != -100 for sample in samples for target in sample["targets"][1:]
        )

        reference_sum = torch.zeros((), device="cuda", dtype=torch.float32)
        # Avoid rounding after every trajectory's BF16 .grad accumulation.
        # This sums BF16 backward results in FP32; it does not exercise the
        # production DDP fused-main-grad GEMM accumulation implementation.
        reference_gradients = [
            torch.zeros_like(param, dtype=torch.float32) for param in reference.parameters()
        ]
        for sample in samples:
            reference.zero_grad(set_to_none=True)
            tokens = torch.tensor(sample["input_ids"][:-1], device="cuda")
            labels = torch.tensor(sample["input_ids"][1:], device="cuda")
            mask = torch.tensor([target != -100 for target in sample["targets"][1:]], device="cuda")
            positions = torch.arange(tokens.numel(), device="cuda")
            losses = _token_losses(reference, tokens, labels, positions)
            loss_sum = (losses * mask).sum()
            reference_sum += loss_sum.detach()
            (loss_sum / supervised_count).backward()
            for gradient, param in zip(reference_gradients, reference.parameters(), strict=True):
                assert param.grad is not None
                gradient.add_(param.grad.detach().float())

        batch = packed_row_to_tensors(
            make_packed_row(samples), pad_token_id=1, padding_divisor=16, sequence_length=256
        )
        batch = {key: value.cuda() for key, value in batch.items()}
        assert int(batch["loss_mask"].sum()) == supervised_count
        assert batch["cu_seqlens"].tolist() == [0, 31, 78, 141]
        assert batch["cu_seqlens_padded"].tolist() == [0, 32, 80, 144]
        params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=batch["cu_seqlens"],
            cu_seqlens_kv=batch["cu_seqlens"],
            cu_seqlens_q_padded=batch["cu_seqlens_padded"],
            cu_seqlens_kv_padded=batch["cu_seqlens_padded"],
            max_seqlen_q=int(batch["max_seqlen"]),
            max_seqlen_kv=int(batch["max_seqlen"]),
        )
        packed_losses = _token_losses(
            packed_model, batch["tokens"], batch["labels"], batch["position_ids"], params
        )
        packed_sum = (packed_losses * batch["loss_mask"]).sum()
        (packed_sum / supervised_count).backward()

        packed_gradients = []
        for (name, expected), (packed_name, actual) in zip(
            reference.named_parameters(), packed_model.named_parameters(), strict=True
        ):
            assert name == packed_name
            assert expected.grad is not None and actual.grad is not None, name
            packed_gradients.append(actual.grad.detach().float())
        reference_flat = torch.cat([gradient.reshape(-1) for gradient in reference_gradients])
        packed_flat = torch.cat([gradient.reshape(-1) for gradient in packed_gradients])
        gradient_errors = _errors(packed_flat, reference_flat)
        reference_weights, reference_update = _master_step(reference, reference_gradients)
        packed_weights, packed_update = _master_step(packed_model, packed_gradients)
        update_errors = _errors(packed_update, reference_update)
        weight_errors = _errors(packed_weights, reference_weights)
        report = {
            "reference_loss_sum": reference_sum.item(),
            "packed_loss_sum": packed_sum.item(),
            "supervised_tokens": supervised_count,
            "model_scope": "bias-free dense BF16 GPT; no GDN/MoE/MTP/CP",
            "reference_accumulation": "FP32 sum of per-trajectory BF16 parameter gradients",
            "gradient_errors": gradient_errors,
            "optimizer_update_errors": update_errors,
            "optimizer_weight_errors": weight_errors,
            "tolerances": {
                "loss_rtol": LOSS_RTOL,
                "loss_atol": LOSS_ATOL,
                "gradient_relative_l2": GRAD_RELATIVE_L2_TOL,
                "gradient_max_abs": GRAD_ABSOLUTE_TOL,
                "step_relative_l2": STEP_RELATIVE_L2_TOL,
                "step_max_abs": STEP_ABSOLUTE_TOL,
                "final_weight_rtol": 0,
                "final_weight_atol": STEP_ABSOLUTE_TOL,
            },
        }
        record_property("agentic_sft_numerical_parity", json.dumps(report, sort_keys=True))
        print("AGENTIC_SFT_NUMERICAL_PARITY " + json.dumps(report, sort_keys=True), flush=True)
        torch.testing.assert_close(
            packed_sum.detach(), reference_sum, rtol=LOSS_RTOL, atol=LOSS_ATOL
        )
        assert gradient_errors["relative_l2"] < GRAD_RELATIVE_L2_TOL, report
        assert gradient_errors["max_abs"] < GRAD_ABSOLUTE_TOL, report
        assert update_errors["relative_l2"] < STEP_RELATIVE_L2_TOL, report
        assert update_errors["max_abs"] < STEP_ABSOLUTE_TOL, report
        torch.testing.assert_close(
            packed_weights, reference_weights, rtol=0, atol=STEP_ABSOLUTE_TOL
        )
    finally:
        Utils.destroy_model_parallel()
