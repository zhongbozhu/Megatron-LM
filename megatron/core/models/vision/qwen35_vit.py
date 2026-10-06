# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Qwen3.5's packed vision tower, using the native Megatron transformer block.

The parameter names match the Qwen3.5 Megatron checkpoint. Qwen3.5 has no
deepstack branches, so its full-recompute path is the ordinary TransformerBlock
implementation. Images are independent THD sequences before the spatial merger.
"""

from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.models.vision.qwen35_rope import Qwen35SelfAttention
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig


def qwen35_vision_config(language_config, *, recompute=True):
    """The pretrained 35B-A3B VL tower; decoder parallel/recompute knobs are independent."""
    config = TransformerConfig(
        num_layers=27,
        hidden_size=1152,
        num_attention_heads=16,
        ffn_hidden_size=4304,
        add_bias_linear=True,
        add_qkv_bias=True,
        normalization="LayerNorm",
        layernorm_epsilon=1e-6,
        activation_func=partial(F.gelu, approximate="tanh"),
        hidden_dropout=0.0,
        attention_dropout=0.0,
        attention_softmax_in_fp32=True,
        apply_rope_fusion=False,
        bias_activation_fusion=False,
        bias_dropout_fusion=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        params_dtype=language_config.params_dtype,
        bf16=language_config.bf16,
        fp16=language_config.fp16,
        attention_backend=language_config.attention_backend,
        gradient_accumulation_fusion=language_config.gradient_accumulation_fusion,
        use_cpu_initialization=language_config.use_cpu_initialization,
        recompute_granularity="full" if recompute else None,
        recompute_method="uniform" if recompute else None,
        recompute_num_layers=1 if recompute else None,
    )
    # The released Qwen vision tower rotates Q/K in FP32 before casting back.
    config.apply_rotary_pos_emb_in_fp32 = True
    return config


class Qwen35PatchMerger(MegatronModule):
    """Normalize patches and project each spatial 2x2 group to the decoder width."""

    def __init__(self, config, output_size, tp_group):
        super().__init__(config)
        self.tp_group = tp_group
        self.merged_size = config.hidden_size * 4
        self.patch_norm = TENorm(config, config.hidden_size, eps=config.layernorm_epsilon)
        self.linear_fc1 = TEColumnParallelLinear(
            self.merged_size,
            self.merged_size,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=True,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="patch_fc1",
            tp_group=tp_group,
        )
        self.linear_fc2 = TERowParallelLinear(
            self.merged_size,
            output_size,
            config=config,
            init_method=config.output_layer_init_method,
            input_is_parallel=True,
            bias=True,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="patch_fc2",
            tp_group=tp_group,
        )

    def forward(self, hidden_states):
        hidden_states = self.patch_norm(hidden_states).reshape(-1, self.merged_size)
        hidden_states, _ = self.linear_fc1(hidden_states)
        # Qwen uses exact GELU in the merger; the vision-block MLP uses tanh GELU.
        output, _ = self.linear_fc2(F.gelu(hidden_states))
        return output


class Qwen35VisionModel(VisionModule):
    """Full Qwen3.5 vision encoder and merger, independently data parallel.

    ``hidden_states`` contains flattened processor patches in spatial-merge
    order, [sum(T*H*W), 3*2*16*16]; ``grid_thw`` is [number_of_images, 3].
    """

    def __init__(self, transformer_config, output_size, pg_collection):
        super().__init__(transformer_config)
        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp
        if pg_collection.tp.size() != 1 or pg_collection.cp.size() != 1:
            raise ValueError("The colocated Qwen vision tower requires encoder TP=CP=1")
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv3d(
            3,
            transformer_config.hidden_size,
            kernel_size=(2, 16, 16),
            stride=(2, 16, 16),
            dtype=transformer_config.params_dtype,
        )
        self.pos_embed = nn.Embedding(
            2304, transformer_config.hidden_size, dtype=transformer_config.params_dtype
        )
        layer_spec = get_vit_layer_with_transformer_engine_spec()
        layer_spec.submodules.self_attention.module = Qwen35SelfAttention
        self.decoder = TransformerBlock(
            config=transformer_config,
            spec=layer_spec,
            pre_process=True,
            post_process=True,
            post_layer_norm=False,
            pg_collection=pg_collection,
        )
        self.merger = Qwen35PatchMerger(transformer_config, output_size, pg_collection.tp)

    def _position_embeddings(self, grid_thw, device):
        """Bilinear learned positions and 2-D rotary phases in processor patch order."""
        learned, phases = [], []
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, head_dim // 2, 2, device=device).float() / (head_dim // 2))
        )
        for temporal, height, width in grid_thw.tolist():
            if height % 2 or width % 2:
                raise ValueError("Qwen vision grids must be divisible by spatial_merge_size=2")
            rows = torch.linspace(0, 47, height, device=device)
            cols = torch.linspace(0, 47, width, device=device)
            y, x = torch.meshgrid(rows, cols, indexing="ij")
            y0, x0 = y.long(), x.long()
            y1, x1 = (y0 + 1).clamp(max=47), (x0 + 1).clamp(max=47)
            dy, dx = y - y0, x - x0
            indices = torch.stack((y0 * 48 + x0, y0 * 48 + x1, y1 * 48 + x0, y1 * 48 + x1))
            weights = torch.stack(((1 - dy) * (1 - dx), (1 - dy) * dx, dy * (1 - dx), dy * dx))
            # HF accumulates bilinear interpolation in FP32, then casts once
            # before adding to the patch embeddings. Rounding each corner in
            # BF16 noticeably perturbs the pretrained vision tower.
            values = (self.pos_embed(indices).float() * weights[..., None]).sum(0)
            values = values.to(self.pos_embed.weight.dtype)
            values = values.reshape(height // 2, 2, width // 2, 2, -1)
            learned.append(
                values.permute(0, 2, 1, 3, 4)
                .reshape(-1, self.config.hidden_size)
                .repeat(temporal, 1)
            )
            y, x = torch.meshgrid(
                torch.arange(height, device=device),
                torch.arange(width, device=device),
                indexing="ij",
            )
            coords = torch.stack((y, x), dim=-1).reshape(height // 2, 2, width // 2, 2, 2)
            coords = coords.permute(0, 2, 1, 3, 4).reshape(-1, 2).repeat(temporal, 1)
            phase = (coords.float()[..., None] * inv_freq).flatten(1)
            phases.append(torch.cat((phase, phase), dim=-1))
        return torch.cat(learned), torch.cat(phases)[:, None, None, :]

    def forward(self, hidden_states, grid_thw):
        patches = hidden_states.reshape(-1, 3, 2, 16, 16)
        hidden_states = self.patch_embed.proj(patches.to(self.patch_embed.proj.weight.dtype))
        hidden_states = hidden_states.reshape(-1, self.config.hidden_size)
        positions, rotary = self._position_embeddings(grid_thw, hidden_states.device)
        hidden_states = (hidden_states + positions)[:, None, :]
        lengths = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
        cu_seqlens = F.pad(lengths.cumsum(0), (1, 0)).to(
            device=hidden_states.device, dtype=torch.int32
        )
        max_seqlen = int(lengths.max())
        packed = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            cp_group=self.pg_collection.cp,
        )
        hidden_states = self.decoder(
            hidden_states=hidden_states,
            attention_mask=None,
            rotary_pos_emb=rotary,
            packed_seq_params=packed,
        )
        return self.merger(hidden_states)
