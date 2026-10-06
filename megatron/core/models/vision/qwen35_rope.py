# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Qwen3.5 interleaved multimodal positions on already partitioned token rows."""

import torch
from torch import nn

from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd
from megatron.core.transformer.attention import SelfAttention


class Qwen35MultimodalRotaryEmbedding(nn.Module):
    """Generate absolute phases from MIMO's CP-local T/H/W position IDs.

    H/W frequencies replace every third temporal frequency, matching Qwen3.5
    interleaved mRoPE. The token layout is already selected by PartitionAdapter;
    neither this module nor its attention consumer repartitions or resets it.
    """

    def __init__(self, kv_channels, rotary_percent=0.25, rotary_base=10000000):
        super().__init__()
        dim = int(kv_channels * rotary_percent)
        self.rotary_base = rotary_base
        self.register_buffer(
            "inv_freq",
            1.0 / rotary_base ** (torch.arange(0, dim, 2).float() / dim),
            persistent=False,
        )

    def forward(self, position_ids, mrope_section, cp_group=None):
        if position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        # Float16Module also converts buffers. Recompute frequencies in FP32 to
        # retain exact long-context positions rather than widening BF16 values.
        dim = self.inv_freq.numel() * 2
        inv_freq = 1.0 / self.rotary_base ** (
            torch.arange(0, dim, 2, device=position_ids.device).float() / dim
        )
        with torch.autocast(device_type=position_ids.device.type, enabled=False):
            freqs = position_ids.float()[..., None] * inv_freq
            temporal = freqs[0].clone()
            for axis in (1, 2):
                indices = slice(axis, mrope_section[axis] * 3, 3)
                temporal[..., indices] = freqs[axis, ..., indices]
            result = torch.cat((temporal, temporal), dim=-1)
        return result.transpose(0, 1).unsqueeze(2).contiguous()


class Qwen35SelfAttention(SelfAttention):
    """Apply local absolute mRoPE; retain native attention, CP and recompute paths."""

    def _adjust_key_value_for_inference(self, inference_context, *args, **kwargs):
        if inference_context is not None:
            raise NotImplementedError("The native colocated Qwen3.5 provider is a training path")
        query, key, value, rotary, mask_type, block_table = super()._adjust_key_value_for_inference(
            inference_context, *args, **kwargs
        )
        if rotary is not None:
            query_rotary, key_rotary = rotary
            query_dtype, key_dtype = query.dtype, key.dtype
            if getattr(self.config, "apply_rotary_pos_emb_in_fp32", False):
                query, key = query.float(), key.float()
            if query_rotary is not None:
                if query.shape[0] != query_rotary.shape[0]:
                    raise ValueError("Qwen mRoPE positions must match the local CP token rows")
                query = _apply_rotary_pos_emb_bshd(query, query_rotary)
            if key_rotary is not None:
                key = _apply_rotary_pos_emb_bshd(key, key_rotary)
            query, key = query.to(query_dtype), key.to(key_dtype)
        return query, key, value, None, mask_type, block_table
