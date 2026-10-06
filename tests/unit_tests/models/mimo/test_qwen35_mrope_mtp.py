# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Real Qwen/GDN/MTP execution with multimodal positions across CP boundaries."""

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from examples.mimo.model_providers.qwen35_native import Qwen35NativeGPT
from megatron.core.models.mimo import MimoModel, MimoModelConfig
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.multi_token_prediction import (
    MTPLossLoggingHelper,
    get_mtp_loss_token_counts,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _execute(cp, checkpoint=None):
    Utils.initialize_model_parallel(1, 1, context_parallel_size=cp)
    model_parallel_cuda_manual_seed(123)
    torch.manual_seed(321)
    pg = ProcessGroupCollection.use_mpu_process_groups()
    config = TransformerConfig(
        num_layers=4,
        hidden_size=128,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=256,
        ffn_hidden_size=256,
        context_parallel_size=cp,
        params_dtype=torch.bfloat16,
        bf16=True,
        gradient_accumulation_fusion=False,
        attention_backend=AttnBackend.fused,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        experimental_attention_variant="gdn",
        linear_attention_freq=4,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        normalization="RMSNorm",
        layernorm_zero_centered_gamma=True,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        qk_layernorm=True,
        attention_output_gate=True,
        mrope_section=[11, 11, 10],
        calculate_per_token_loss=True,
        apply_rope_fusion=False,
        mtp_num_layers=1,
        mtp_loss_scaling_factor=0.1,
    )
    language = ModuleSpec(
        module=Qwen35NativeGPT,
        params=dict(
            config=config,
            vocab_size=512,
            max_sequence_length=1024,
            position_embedding_type="mrope",
            rotary_percent=0.25,
            rotary_base=10000000,
            share_embeddings_and_output_weights=False,
            scatter_embedding_sequence_parallel=False,
            pg_collection=pg,
        ),
    )
    model = MimoModel(
        MimoModelConfig(
            language_model_spec=language,
            modality_submodules_spec={"images": ModuleSpec(module=VisionModalitySubmodules)},
            special_token_ids={"images": 511},
            kv_format="thd",
        ),
        cp_group=pg.cp,
        tp_group=pg.tp,
        external_modality_transport=True,
    ).cuda()
    if checkpoint is not None:
        model.load_state_dict(checkpoint)
    checkpoint = {k: v.cpu() if torch.is_tensor(v) else v for k, v in model.state_dict().items()}
    tokens = torch.arange(160, device="cuda").unsqueeze(0) % 127
    tokens[:, 5:25] = tokens[:, 80:100] = 511
    labels = torch.roll(tokens, -1, -1)
    mask = torch.ones_like(tokens, dtype=torch.float32)
    mask[:, 59:64] = mask[:, 150:] = 0
    mask[tokens == 511] = 0
    positions = torch.arange(160, device="cuda").reshape(1, 1, -1).repeat(3, 1, 1)
    positions[:, :, 64:] -= 64
    for start in (5, 80):
        positions[0, 0, start : start + 20] = start
        positions[1, 0, start : start + 20] = start + torch.arange(20, device="cuda") // 5
        positions[2, 0, start : start + 20] = start + torch.arange(20, device="cuda") % 5
    vision = torch.sin(torch.arange(40 * 128, device="cuda").float()).reshape(40, 128)
    vision = vision.bfloat16().requires_grad_(True)
    packing = dict(
        cu_seqlens_q=torch.tensor([0, 59, 145], device="cuda", dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 59, 145], device="cuda", dtype=torch.int32),
        cu_seqlens_q_padded=torch.tensor([0, 64, 160], device="cuda", dtype=torch.int32),
        cu_seqlens_kv_padded=torch.tensor([0, 64, 160], device="cuda", dtype=torch.int32),
        max_seqlen_q=96,
        max_seqlen_kv=96,
        local_cp_size=1,
        cp_group=pg.tp,
    )
    padding_mask = torch.zeros_like(tokens, dtype=torch.bool)
    padding_mask[:, 59:64] = padding_mask[:, 150:] = True
    counts = get_mtp_loss_token_counts(
        mask,
        1,
        mtp_input_mask=tokens != 511,
        packed_seq_params=PackedSeqParams(qkv_format="thd", **packing),
        cp_group=pg.tp,
    )
    packing.update(cp_group=pg.cp, local_cp_size=cp, mtp_loss_token_counts=counts)
    MTPLossLoggingHelper.clean_metrics_in_tracker()
    output, local_mask = model(
        input_ids=tokens,
        position_ids=positions,
        labels=labels,
        loss_mask=mask,
        padding_mask=padding_mask,
        modality_embeddings={"images": vision},
        packing_kwargs=packing,
    )
    loss = (output.float() * local_mask).sum() / mask.sum()
    loss.backward()
    dist.all_reduce(loss.detach(), group=pg.cp)
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            grad = parameter.grad.float()
            dist.all_reduce(grad, group=pg.cp)
            gradients[name] = grad.cpu()
    vision_grad = vision.grad.float()
    dist.all_reduce(vision_grad, group=pg.cp)
    tracker = MTPLossLoggingHelper.tracker
    mtp = torch.stack((tracker["loss_sums"].clone(), tracker["loss_token_counts"].clone()))
    dist.all_reduce(mtp, group=pg.cp)
    result = dict(
        loss=loss.item(),
        mtp=(mtp[0] / mtp[1]).item(),
        gradients=gradients,
        vision=vision_grad.cpu(),
    )
    del model
    Utils.destroy_model_parallel()
    return checkpoint, result


@pytest.mark.parametrize("cp", [2, 4])
def test_qwen35_mrope_mtp_cp_parity(cp):
    checkpoint, reference = _execute(1)
    _, candidate = _execute(cp, checkpoint)
    assert abs(reference["loss"] - candidate["loss"]) < 0.002
    assert abs(reference["mtp"] - candidate["mtp"]) < 0.002
    for group in ("mtp", "decoder", "embedding", "output_layer"):
        names = [name for name in reference["gradients"] if f".{group}." in name]
        a = torch.cat([reference["gradients"][name].flatten() for name in names])
        b = torch.cat([candidate["gradients"][name].flatten() for name in names])
        assert (a - b).norm() / a.norm().clamp_min(1e-8) < 0.025, group
    a, b = reference["vision"], candidate["vision"]
    assert (a - b).norm() / a.norm().clamp_min(1e-8) < 0.025
