# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
This script provides a basic training loop for MIMO models.
"""

import os
import sys
from functools import partial
from typing import Any, Dict, Iterator

import torch

from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_data_parallel_group,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_src_rank,
)
from megatron.training import get_args, pretrain, print_rank_0
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.arguments import parse_and_validate_args

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)
from data.energon_avlm_task_encoder import llava_avlm_dataloader_provider
from data.energon_vlm_task_encoder import llava_vlm_dataloader_provider
from data.mock import train_valid_test_datasets_provider as mock_train_valid_test_datasets_provider
from model_providers.llava_avlm import model_provider_llava_avlm
from model_providers.llava_vlm import model_provider_llava_vlm
from model_providers.mock import model_provider_mock_vlm_single_encoder
from utils.data_helpers import broadcast_nested_data_batch

from examples.mimo.model_providers.qwen35_native import (
    add_qwen35_native_args,
    model_provider_qwen35_native,
)
from megatron.core.enums import ModelType

_MODEL_PROVIDERS = {
    "mock": model_provider_mock_vlm_single_encoder,
    "llava_vlm": model_provider_llava_vlm,
    "video_llava_vlm": partial(model_provider_llava_vlm, is_video_input=True),
    "llava_avlm": model_provider_llava_avlm,
    "qwen35_native": model_provider_qwen35_native,
}

_DATASET_PROVIDERS = {
    "mock": mock_train_valid_test_datasets_provider,
    "llava_vlm": llava_vlm_dataloader_provider,
    "video_llava_vlm": partial(llava_vlm_dataloader_provider, is_video_input=True),
    "llava_avlm": llava_avlm_dataloader_provider,
}


def add_mimo_args(parser):
    """Add MIMO-specific arguments to the parser."""
    group = parser.add_argument_group('MIMO', 'MIMO specific arguments')

    # MIMO-specific parameters
    group.add_argument(
        '--dataset-provider',
        type=str,
        default='mock',
        help='Dataset provider to choose from [mock, llava_vlm, video_llava_vlm, llava_avlm]',
    )
    group.add_argument(
        '--model-provider',
        type=str,
        default='mock',
        help='Model provider to choose from [mock, llava_vlm, video_llava_vlm, llava_avlm]',
    )

    # mock dataloader related args
    # can control mock samples with total seq length and image seq length
    group.add_argument('--image-size', type=int, default=224, help='Image size for vision encoder')
    group.add_argument('--total-seq-length', type=int, default=512, help='Total sequence length')
    group.add_argument('--pad-token-id', type=int, default=0, help='Padding token ID')
    group.add_argument('--image-token-id', type=int, default=32000, help='Image token ID')
    group.add_argument(
        '--image-seq-length', type=int, default=197, help='Number of image tokens to pad'
    )
    group.add_argument(
        '--audio-encoder-model', type=str, default=None, help='Audio encoder model name'
    )
    group.add_argument(
        '--hf-assign-unused-tokens',
        type=str,
        nargs='+',
        default=None,
        help='Assigning unused tokens to special tokens. Example: '
        '--hf-assign-unused-tokens "<audio>,32002" "<video>,32003"',
    )
    # checkpoint related args
    group.add_argument(
        '--language-model-checkpoint',
        type=str,
        default=None,
        help='Path to language model checkpoint to load',
    )
    # energon dataloader related args
    group.add_argument(
        '--packing-buffer-size',
        type=int,
        default=None,
        help='Packing buffer size when using sequence packing',
    )

    group.add_argument('--mimo-manifest', type=str, help='Real image conversation manifest')
    group.add_argument('--mimo-heldout-images', type=int, default=64)
    group.add_argument('--mimo-data-split-seed', type=int, default=None)
    group.add_argument(
        '--mimo-diagnostic-source-batch',
        type=str,
        help='Diagnostic only: repeat a frozen source batch for controlled boundary comparisons',
    )
    group.add_argument(
        '--mimo-feature-transport',
        choices=('pack', 'direct_reference'),
        default='pack',
        help='Diagnostic direct reference bypasses the production pack bridge using image IDs',
    )
    group.add_argument(
        '--mimo-long-sample-min-tokens',
        type=int,
        default=0,
        help='Join distinct complete conversations into long multi-image samples',
    )
    group.add_argument(
        '--mimo-gradient-diagnostics',
        action='store_true',
        help='Record sparse gradient fingerprints for layout parity checks',
    )
    group.add_argument(
        '--mimo-gradient-reference',
        type=str,
        help='Directory for complete normalized gradient shard comparison',
    )
    group.add_argument(
        '--mimo-audit-token-assignments',
        action='store_true',
        help='Audit exact per-token MoE choices across statistics, training and recomputation',
    )
    routing = group.add_mutually_exclusive_group()
    routing.add_argument(
        '--mimo-fixed-routing-record',
        type=str,
        help='Diagnostic: record canonical expert IDs in CP1 statistics, then replay them',
    )
    routing.add_argument(
        '--mimo-fixed-routing-replay',
        type=str,
        help='Diagnostic: replay recorded expert IDs with live router probabilities and gradients',
    )
    packing = group.add_mutually_exclusive_group()
    packing.add_argument(
        '--mimo-fixed-packing-record',
        type=str,
        help='Diagnostic: record intact CP1 packs for a controlled static-CP comparison',
    )
    packing.add_argument(
        '--mimo-fixed-packing-replay',
        type=str,
        help='Diagnostic: place recorded intact packs on static CP groups without repacking',
    )
    packing.add_argument(
        '--mimo-diagnostic-schedule-replay',
        type=str,
        help='Diagnostic only: replay an explicit DCP schedule bound to source and baseline plan',
    )
    group.add_argument(
        '--mimo-diagnostic-schedule-audit-dir',
        type=str,
        help='Diagnostic: record the actual validated post-replay DCP assignments and THD layouts',
    )
    group.add_argument(
        '--mimo-execution-diagnostics-dir',
        type=str,
        help='Opt-in receiver boundary snapshots and sample-aligned decoder probes',
    )
    group.add_argument(
        '--mimo-diagnostic-full-decoder-input',
        action='store_true',
        help='Diagnostic only: save every first-layer input row and its gradient',
    )
    group.add_argument(
        '--mimo-diagnostic-steps',
        type=int,
        nargs='+',
        help='Source step indices to snapshot; defaults to every training step when enabled',
    )
    group.add_argument(
        '--mimo-diagnostic-gdn-layers',
        type=int,
        nargs='*',
        default=[0],
        help='GDN layers for full tensor capture; pass no indices for boundary/sampled probes only',
    )
    group.add_argument(
        '--mimo-diagnostic-moe-layers',
        type=int,
        nargs='+',
        default=[],
        help='MoE layer indices whose complete boundaries are captured when diagnostics run',
    )
    group.add_argument(
        '--mimo-boundary-reference',
        type=str,
        help='Compare encoder features and normalized gradients before ViT backward',
    )
    group.add_argument(
        '--mimo-write-gradient-reference',
        action='store_true',
        help='Write the baseline gradient shards instead of comparing them',
    )
    group.add_argument(
        '--mimo-metrics-dir', type=str, help='Directory for native MIMO step metrics'
    )
    group.add_argument(
        '--mimo-optimizer-reference',
        type=str,
        help='Opt-in reference directory for one complete native AdamW update',
    )
    group.add_argument('--mimo-write-optimizer-reference', action='store_true')
    group.add_argument(
        '--mimo-optimizer-expected-step',
        type=int,
        help='Required starting Adam step; zero explicitly permits native lazy initialization',
    )
    group.add_argument('--mimo-optimizer-scratch-dir', type=str)
    group.add_argument('--mimo-optimizer-report', type=str)
    group.add_argument(
        '--mimo-optimizer-cold-moment-bands',
        action='store_true',
        help='Diagnose cold Adam update errors by moment sign and scale; requires step zero',
    )
    add_qwen35_native_args(parser)

    return parser


def get_batch(data_iterator: Iterator[Dict[str, Any]]):
    """Generate a batch for MIMO model training.

    Args:
        data_iterator: Iterator over the dataset

    Returns:
        tuple: Batch data for model training
    """
    args = get_args()

    # Assert that pipeline parallelism are not supported yet
    assert (
        getattr(args, 'pipeline_model_parallel_size', 1) == 1
    ), "Pipeline parallelism is not supported yet in MIMO implementation"

    # Broadcast data - only get data on tensor parallel rank 0
    # data iterator is None on other tp ranks
    # TP Rank-0 reads next batch.
    if get_tensor_model_parallel_rank() == 0:
        try:
            data = next(data_iterator)
            has_data = torch.tensor([1], dtype=torch.uint8, device='cuda')
        except StopIteration:
            has_data = torch.tensor([0], dtype=torch.uint8, device='cuda')
            data = None
    else:
        has_data = torch.empty(1, dtype=torch.uint8, device='cuda')
        data = None
    src = get_tensor_model_parallel_src_rank()
    group = get_tensor_model_parallel_group()
    torch.distributed.broadcast(has_data, src, group=group)

    if has_data.item() == 0:
        # iterator exhausted on all ranks
        # we need this to avoid race condition when first tp rank hits StopIteration
        return None

    # MiMo forward pass expects
    # input_ids: torch.Tensor,
    # position_ids: Optional[torch.Tensor] = None,
    # attention_mask: Optional[torch.Tensor] = None,
    # loss_mask: Optional[torch.Tensor] = None,
    # labels: Optional[torch.Tensor] = None,
    # modality_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
    # packing_kwargs: Optional[dict] = None,

    # For the modality inputs, the keys can be arbitrary
    # so we do a broadcast of the schema followed by a broadcast of the actual data
    # check broadcast_nested_data_batch for more details
    batch = broadcast_nested_data_batch(data)

    return batch


def loss_func(loss_mask, output_tensor):
    """Simple loss function for MIMO model training.

    Args:
        loss_mask: mask indicating which tokens contribute to the loss
        output_tensor: model output tensor
    Returns:
        tuple: (loss, num_tokens, metrics_dict)
    """
    args = get_args()
    losses = output_tensor.float()

    loss_mask = loss_mask.contiguous().view(-1).float()

    total_tokens = loss_mask.sum().clone().detach().to(torch.int)
    total_loss = torch.sum(losses.view(-1) * loss_mask)

    loss = torch.cat([total_loss.view(1), total_tokens.view(1)])

    loss_for_backward = loss[0].clone()
    # If CP is active, reduce the loss across all CP ranks
    # as they have loss calculated for their own sequence shards.
    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=get_context_parallel_group())
        loss_for_backward = loss[0].clone()
    # For reporting, clone and detach the loss. This creates a new tensor
    # that doesn't require gradients and is independent of the computation graph.
    reporting_loss = loss.clone().detach()
    torch.distributed.all_reduce(reporting_loss, group=get_data_parallel_group())

    local_num_tokens = loss[1].clone().detach().to(torch.int)

    return (loss_for_backward, local_num_tokens, {'lm loss': (reporting_loss)})


def forward_step(data_iterator, model):
    """Forward step for MIMO model training.

    Args:
        data_iterator: iterator over the dataset
        model: MIMO model instance

    Returns:
        tuple: (output_tensor, loss_function)
    """
    if get_args().model_provider == 'qwen35_native':
        from examples.mimo.native_step import forward_step as native_forward_step

        return native_forward_step(data_iterator, model)
    data_batch = get_batch(data_iterator)
    output_tensor, loss_mask = model(**data_batch)

    # Return output and loss function
    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(*provider_args, **provider_kwargs):
    """Dataset provider for MIMO model training.

    Args:
        *provider_args: Additional arguments for the dataset provider
        **provider_kwargs: Additional keyword arguments for the dataset provider
    """
    runtime_args = get_args()
    if runtime_args.dataset_provider == 'qwen35_native':
        from examples.mimo.data.qwen35_native import Qwen35Dataset
        from examples.mimo.native_step import NativeSourceIterator

        if runtime_args.dataloader_type != 'external':
            raise ValueError('Native MIMO source batches require --dataloader-type external')

        def iterator(split, consumed):
            if runtime_args.mimo_diagnostic_source_batch:
                from examples.mimo.data.joint_cp_fixture import FrozenSourceDataset

                dataset = FrozenSourceDataset(runtime_args.mimo_diagnostic_source_batch)
                return NativeSourceIterator(dataset, runtime_args.global_batch_size, consumed)
            dataset = Qwen35Dataset(
                runtime_args.mimo_manifest,
                runtime_args.mimo_hf_model,
                runtime_args.seq_length,
                seed=runtime_args.seed,
                split=split,
                heldout_images=runtime_args.mimo_heldout_images,
                split_seed=runtime_args.mimo_data_split_seed,
                long_sample_min_tokens=runtime_args.mimo_long_sample_min_tokens,
            )
            return NativeSourceIterator(dataset, runtime_args.global_batch_size, consumed)

        return (
            iterator('train', runtime_args.consumed_train_samples),
            (
                iterator('valid', runtime_args.consumed_valid_samples)
                if runtime_args.eval_iters
                else None
            ),
            None,
        )
    try:
        dataset_provider = _DATASET_PROVIDERS[runtime_args.dataset_provider]
        if runtime_args.dataset_provider != "mock":
            # Calculate max_seq_length from total_seq_length
            max_seq_length = runtime_args.total_seq_length
            print_rank_0(
                f"MIMO Training: Using max_seq_length = {max_seq_length} "
                f"(total_seq_length: {runtime_args.total_seq_length})"
            )

            # Add configs to provider_kwargs
            provider_kwargs['max_seq_length'] = max_seq_length
    except KeyError as e:
        raise ValueError(
            f"Unsupported dataset provider '{runtime_args.dataset_provider}'. "
            f"Available providers: {list(_DATASET_PROVIDERS.keys())}"
        ) from e

    return dataset_provider(*provider_args, **provider_kwargs)


def model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    add_encoder: bool = True,
    add_decoder: bool = True,
    image_special_token_id: int = 32000,
    audio_special_token_id: int = 32002,
    **framework_kwargs,
):
    """Model provider for MIMO model training.

    Args:
        pre_process: Whether to pre-process the model
        post_process: Whether to post-process the model
        add_encoder: Whether to add an encoder to the model (not supported yet)(default: True)
        add_decoder: Whether to add a decoder to the model (not supported yet)(default: True)
        image_special_token_id: Special token ID for the image modality (default: 32000)
        audio_special_token_id: Special token ID for the audio modality (default: 32002)
        **framework_kwargs: Framework-injected kwargs from Megatron's training loop,
            including `config` (TransformerConfig) and `pg_collection` (ProcessGroupCollection).
            `pg_collection` is forwarded to the model builder so process groups are passed
            explicitly rather than fetched from global parallel state.
    """
    runtime_args = get_args()
    pg_collection = framework_kwargs.get('pg_collection')

    if runtime_args.model_provider == 'qwen35_native':
        from examples.mimo.native_step import NativeMimoStep

        model = model_provider_qwen35_native(pre_process, post_process, pg_collection=pg_collection)
        model.native_mimo_step = NativeMimoStep(model, model.pg_collection, runtime_args)
        return model

    try:
        builder_fn = _MODEL_PROVIDERS[runtime_args.model_provider]
    except KeyError as e:
        raise ValueError(
            f"Unsupported model provider '{runtime_args.model_provider}'. "
            f"Available providers: {list(_MODEL_PROVIDERS.keys())}"
        ) from e

    if runtime_args.model_provider == "llava_vlm":
        builder_kwargs = {
            "image_special_token_id": image_special_token_id,
            "pg_collection": pg_collection,
        }
    elif runtime_args.model_provider == "llava_avlm":
        builder_kwargs = {
            "image_special_token_id": image_special_token_id,
            "audio_special_token_id": audio_special_token_id,
            "pg_collection": pg_collection,
        }
    else:
        raise ValueError(
            f"Unknown model provider: {runtime_args.model_provider}. Must be one of ['llava_vlm', 'llava_avlm', 'mock]"
        )

    return builder_fn(pre_process, post_process, add_encoder, add_decoder, **builder_kwargs)


if __name__ == "__main__":

    train_valid_test_datasets_provider.is_distributed = True
    args = parse_and_validate_args(args_defaults={}, extra_args_provider=add_mimo_args)
    full_config = pretrain_cfg_container_from_args(args)
    from examples.mimo.native_optimizer_diagnostics import observe_native_optimizer

    with observe_native_optimizer(args):
        pretrain(
            full_config,
            train_valid_test_datasets_provider,
            ModelType.encoder_or_decoder,
            forward_step,
            model_provider,
        )
