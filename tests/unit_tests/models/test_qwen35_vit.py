# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from copy import deepcopy

import pytest
import torch

from megatron.core.models.vision.qwen35_rope import Qwen35MultimodalRotaryEmbedding
from megatron.core.models.vision.qwen35_vit import Qwen35VisionModel, qwen35_vision_config
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils, clear_nvte_env_vars


def test_qwen35_mrope_local_positions_and_bf16_buffer():
    """CP token selection must commute with mRoPE, including positions above 64K."""
    positions = torch.tensor(
        [[[0, 1, 65537, 65538]], [[0, 1, 65537, 65539]], [[0, 1, 65538, 65537]]], device="cuda"
    )
    rope = Qwen35MultimodalRotaryEmbedding(256).cuda().bfloat16()
    full = rope(positions, [11, 11, 10])
    local = rope(positions[..., [0, 3]], [11, 11, 10])
    torch.testing.assert_close(local, full[[0, 3]], rtol=0, atol=0)
    frequency = 10000000.0 ** (-torch.arange(32, device="cuda").float() / 32)
    expected = positions[0, 0, :, None].float() * frequency
    for axis, count in ((1, 11), (2, 10)):
        indices = torch.arange(axis, count * 3, 3, device="cuda")
        expected[:, indices] = positions[axis, 0, :, None].float() * frequency[indices]
    torch.testing.assert_close(full[:, 0, 0, :32], expected, rtol=2e-7, atol=0)


@pytest.mark.parametrize("recompute", [False, True])
def test_qwen35_vision_packing_and_recompute(recompute):
    """Packed independent images must agree with separately encoded images and gradients."""
    Utils.initialize_model_parallel(1, 1)
    clear_nvte_env_vars()
    model_parallel_cuda_manual_seed(123)
    try:
        language = TransformerConfig(
            num_layers=2,
            hidden_size=144,
            num_attention_heads=2,
            params_dtype=torch.bfloat16,
            bf16=True,
            gradient_accumulation_fusion=False,
            attention_backend=AttnBackend.fused,
        )
        config = qwen35_vision_config(language, recompute=recompute)
        config.num_layers = 2
        config.hidden_size = 144
        config.num_attention_heads = config.num_query_groups = 2
        config.ffn_hidden_size = 256
        pg = ProcessGroupCollection.use_mpu_process_groups()
        packed_model = Qwen35VisionModel(config, 128, pg).cuda()
        separate_model = Qwen35VisionModel(deepcopy(config), 128, pg).cuda()
        separate_model.load_state_dict(packed_model.state_dict())
        patches = torch.randn(40, 1536, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        reference_patches = patches.detach().clone().requires_grad_(True)
        grids = torch.tensor([[1, 4, 4], [1, 4, 6]], device="cuda")
        actual = packed_model(patches, grids)
        expected = torch.cat(
            (
                separate_model(reference_patches[:16], grids[:1]),
                separate_model(reference_patches[16:], grids[1:]),
            )
        )
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.005)
        actual.float().square().sum().backward()
        expected.float().square().sum().backward()
        torch.testing.assert_close(patches.grad, reference_patches.grad, rtol=0.03, atol=0.005)
        for (_, parameter), (_, reference) in zip(
            packed_model.named_parameters(), separate_model.named_parameters()
        ):
            relative = (
                parameter.grad.float() - reference.grad.float()
            ).norm() / reference.grad.float().norm().clamp_min(1e-8)
            assert relative < 0.03
    finally:
        Utils.destroy_model_parallel()
