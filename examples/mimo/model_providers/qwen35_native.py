# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pretrained Qwen3.5-35B-A3B VL for the native colocated MIMO training entrypoint."""

from copy import copy
from pathlib import Path

from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.mapping import ShardedBase
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
    get_transformer_layer_with_experimental_attention_variant_spec,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo import MimoModel, MimoModelConfig
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.qwen35_rope import (
    Qwen35MultimodalRotaryEmbedding,
    Qwen35SelfAttention,
)
from megatron.core.models.vision.qwen35_vit import Qwen35VisionModel, qwen35_vision_config
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args


def add_qwen35_native_args(parser):
    """Only model-specific options; native arguments own all decoder settings."""
    group = parser.add_argument_group("native Qwen3.5 MIMO")
    group.add_argument("--mimo-hf-model", type=str, help="Local HF tokenizer/processor snapshot")
    group.add_argument(
        "--mimo-pretrained-checkpoint",
        type=str,
        help="Initial Bridge torch_dist iteration directory; use --load for native resume",
    )
    group.add_argument(
        "--mimo-vision-recompute",
        action="store_true",
        help="Full uniform layer recompute for the independent Qwen vision encoder",
    )
    return parser


def _patch_attention_spec(spec):
    """Change only standard attention; GDN and all native layer/MTP logic are reused."""
    if hasattr(spec, "layer_specs"):
        for layer in spec.layer_specs:
            _patch_attention_spec(layer)
        return
    submodules = getattr(spec, "submodules", None)
    if hasattr(submodules, "mtp_model_layer"):
        _patch_attention_spec(submodules.mtp_model_layer)
    attention = getattr(submodules, "self_attention", None)
    if attention is not None and issubclass(attention.module, SelfAttention):
        attention.module = Qwen35SelfAttention


class Qwen35NativeGPT(GPTModel):
    """Native GPT/GDN/MoE/MTP with Qwen3.5's absolute, interleaved mRoPE."""

    def __init__(self, config, **kwargs):
        block = get_transformer_block_with_experimental_attention_variant_spec(config)
        _patch_attention_spec(block)
        mtp = None
        if config.mtp_num_layers:
            last_layer = get_transformer_layer_with_experimental_attention_variant_spec(config)[-1]
            _patch_attention_spec(last_layer)
            mtp = get_gpt_mtp_block_spec(config, last_layer, use_transformer_engine=True, pp_rank=0)
        super().__init__(config=config, transformer_layer_spec=block, mtp_block_spec=mtp, **kwargs)
        self.rotary_pos_emb = Qwen35MultimodalRotaryEmbedding(
            config.kv_channels, kwargs["rotary_percent"], kwargs["rotary_base"]
        )


def _load_pretrained(model, checkpoint):
    """Import VL weights once without altering native MIMO save/resume namespaces."""
    checkpoint = Path(checkpoint)
    if not (checkpoint / "metadata.json").is_file():
        raise ValueError("--mimo-pretrained-checkpoint must point to a torch_dist iteration folder")
    state = model.sharded_state_dict()
    vision_prefix = "modality_submodules.images.encoders.qwen35."

    def remap(value):
        if isinstance(value, ShardedBase) and value.key.startswith(vision_prefix):
            value.key = "vision_model." + value.key[len(vision_prefix) :]
        elif isinstance(value, dict):
            for item in value.values():
                remap(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                remap(item)

    remap(state)
    loaded = dist_checkpointing.load(
        {"model": state}, str(checkpoint), strict="raise_all", validate_access_integrity=True
    )
    model.load_state_dict(loaded["model"], strict=True)
    print_rank_0(
        f"Imported pretrained Qwen3.5 VL encoder, merger, decoder and MTP from {checkpoint}"
    )


def model_provider_qwen35_native(
    pre_process=True, post_process=True, vp_stage=None, pg_collection=None
):
    """Build the full VL model using native CLI configuration and checkpointing."""
    args = get_args()
    if not pre_process or not post_process or args.pipeline_model_parallel_size != 1:
        raise ValueError("The native colocated Qwen3.5 provider currently requires PP=1")
    if args.tensor_model_parallel_size != 1:
        raise ValueError("The native colocated Qwen3.5 provider currently requires TP=1")
    if args.position_embedding_type != "mrope" or args.mrope_section != [11, 11, 10]:
        raise ValueError(
            "Qwen3.5 VL requires --position-embedding-type mrope --mrope-section 11 11 10"
        )
    if args.apply_rope_fusion:
        raise ValueError("Qwen3.5 VL's absolute mRoPE requires --no-rope-fusion")
    pg = pg_collection or ProcessGroupCollection.use_mpu_process_groups()
    config = core_transformer_config_from_args(args)
    language_spec = ModuleSpec(
        module=Qwen35NativeGPT,
        params=dict(
            config=config,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=True,
            post_process=True,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type="mrope",
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            scatter_embedding_sequence_parallel=False,
            pg_collection=pg,
        ),
    )
    encoder_pg = copy(pg)
    for name in ("tp", "cp", "tp_cp", "ep", "expt_tp", "tp_ep", "tp_ep_pp"):
        setattr(encoder_pg, name, pg.tp)
    for name in ("dp", "dp_cp", "dp_cp_gtp_remat", "expt_dp", "expt_dp_gtp_remat"):
        setattr(encoder_pg, name, pg.dp_cp)
    encoder = ModuleSpec(
        module=Qwen35VisionModel,
        params=dict(
            transformer_config=qwen35_vision_config(config, recompute=args.mimo_vision_recompute),
            output_size=config.hidden_size,
            pg_collection=encoder_pg,
        ),
    )
    images = ModuleSpec(
        module=VisionModalitySubmodules,
        params={"pg_collection": encoder_pg},
        submodules={"encoders": {"qwen35": encoder}},
    )
    model = MimoModel(
        MimoModelConfig(
            language_model_spec=language_spec,
            modality_submodules_spec={"images": images},
            special_token_ids={"images": 248056},
            kv_format="thd",
        ),
        cp_group=pg.cp,
        tp_group=pg.tp,
        external_modality_transport=True,
    )
    model.pg_collection = pg
    # The outer native DDP wrapper synchronizes every dense parameter over dp_cp,
    # including this encoder; decoder expert parameters retain their EP grouping.
    if args.mimo_pretrained_checkpoint and not args.load:
        _load_pretrained(model, args.mimo_pretrained_checkpoint)
    return model
