# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Colocated vision routing inside Megatron's native PP1 training schedule.

Only the modality boundary has an explicit backward. Decoder backward, gradient
buffers, DP/EP reductions, optimizer updates and checkpoints remain native.
"""

from functools import partial

import torch
import torch.distributed as dist

from examples.mimo.data.packed_multimodal import assign_encoder_media, build_round_plans
from megatron.core import parallel_state
from megatron.core.datasets.data_schedule import _build_thd_padding_mask
from megatron.core.datasets.data_schedule_utils import (
    build_packed_microbatches,
    pad_packed_batch_before_cp_slice,
)
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.models.mimo.comm.pack_bridge import PackFeatureBridge
from megatron.core.rerun_state_machine import RerunDataIterator
from megatron.core.utils import get_attr_wrapped_model


def _packing(batch, cp_group):
    return dict(
        cu_seqlens_q=batch['cu_seqlens'],
        cu_seqlens_kv=batch['cu_seqlens'],
        cu_seqlens_q_padded=batch['cu_seqlens_padded'],
        cu_seqlens_kv_padded=batch['cu_seqlens_padded'],
        max_seqlen_q=int(batch['max_seqlen']),
        max_seqlen_kv=int(batch['max_seqlen']),
        local_cp_size=cp_group.size(),
        cp_group=cp_group,
    )


class NativeSourceIterator:
    """Deterministic source global batches; resume uses native consumed samples."""

    def __init__(self, dataset, global_batch_size, consumed_samples=0):
        if consumed_samples % global_batch_size:
            raise ValueError('MIMO resume requires a complete source global batch')
        self.dataset = dataset
        self.global_batch_size = global_batch_size
        self.step = consumed_samples // global_batch_size

    def __iter__(self):
        return self

    def __next__(self):
        samples, media = self.dataset.build_global_batch(self.step, self.global_batch_size)
        result = dict(step=self.step, samples=samples, media=media)
        self.step += 1
        return result


class NativeMimoStep:
    """Own one source batch from scheduling through encoder-gradient completion."""

    def __init__(self, model, pg_collection, args):
        config = model.config
        if pg_collection.tp.size() != 1 or pg_collection.pp.size() != 1:
            raise ValueError('Colocated MIMO packing currently requires TP=PP=1')
        if not config.calculate_per_token_loss or not config.sequence_packing_scheduler:
            raise ValueError('Colocated MIMO requires packed scheduling and per-token loss')
        if args.overlap_grad_reduce or args.overlap_param_gather:
            raise ValueError('Delayed encoder backward currently requires non-overlapped DDP')
        if args.rerun_mode != 'disabled':
            raise ValueError('Colocated MIMO currently requires --rerun-mode disabled')
        self.model, self.pg = model, pg_collection
        self.config = config
        self.domain = tuple(dist.get_process_group_ranks(pg_collection.dp_cp))
        self.rank = dist.get_rank()
        self.domain_rank = self.domain.index(self.rank)
        self.bridge = PackFeatureBridge(pg_collection.dp_cp)
        self.active = False
        config.sequence_packing_data_adapter = self.prepare
        config.finalize_model_grads_func = self.finalize

    def prepare(self, data_iterator, num_microbatches, scheduler, pg_collection):
        """Reuse the selected core scheduler and THD builder; attach visual routes."""
        from examples.mimo.data.qwen35_native import build_vision_inputs

        if self.active:
            raise RuntimeError('Previous MIMO step has not completed its backward boundary')
        source_batch = next(data_iterator)
        samples, media = source_batch['samples'], source_batch['media']
        if sorted(samples) != list(range(len(samples))):
            raise ValueError('Source sample IDs must be contiguous')
        lengths = [(sid, int(sample['padded_seq_len'])) for sid, sample in samples.items()]
        assignments = scheduler.get_groups_and_subsamples(lengths)
        scheduled = [
            sid
            for assignment in assignments
            for pack in dict.fromkeys(tuple(ids) for ids in assignment)
            for sid in pack
        ]
        if sorted(scheduled) != sorted(samples):
            raise ValueError('Every source sample must appear exactly once in the schedule')
        encoder_tasks, slices = assign_encoder_media(media, self.domain)
        device = torch.device('cuda', torch.cuda.current_device())
        encoded = self.model.encode_modalities(
            build_vision_inputs(encoder_tasks[self.rank], device)
        )
        self.source = encoded.get('images')
        if self.source is None:
            self.source = torch.empty(
                (0, self.config.hidden_size), device=device, dtype=self.config.params_dtype
            )
        expected_rows = sum(item['length'] for item in encoder_tasks[self.rank])
        if self.source.shape[0] != expected_rows:
            raise ValueError(
                f'Encoder emitted {self.source.shape[0]} rows, expected {expected_rows}'
            )
        self.source_gradient = torch.zeros_like(self.source, dtype=torch.float32)
        self.pending = None
        self.training = torch.is_grad_enabled()
        self.active = True
        # Core packing consumes 1-D text fields. Absolute multimodal positions
        # use the same physical boundaries below, without changing pack decisions.
        text_samples = {
            sid: {
                key: value.to(device)
                for key, value in sample.items()
                if isinstance(value, torch.Tensor) and key != 'position_ids'
            }
            for sid, sample in samples.items()
        }
        for sample in text_samples.values():
            sample['position_ids'] = torch.arange(sample['tokens'].numel(), device=device)
        for sid, sample in text_samples.items():
            for key in ('original_seq_len', 'padded_seq_len'):
                sample[key] = torch.as_tensor(samples[sid][key], device=device, dtype=torch.int32)
        batches = build_packed_microbatches(
            text_samples, assignments, self.domain_rank, device, scheduler.is_dynamic_cp
        )
        self.rounds = []
        for round_id, (assignment, packed) in enumerate(zip(assignments, batches)):
            plans = build_round_plans(assignment, samples, slices, self.domain)
            plan = next(plan for plan in plans if self.rank in plan.cp_ranks)
            cp_group = (
                parallel_state.get_dynamic_data_context_parallel_groups(
                    group_size=len(plan.cp_ranks)
                )
                if scheduler.is_dynamic_cp
                else self.pg.cp
            )
            packed['padding_mask'] = _build_thd_padding_mask(
                packed['cu_seqlens'], packed['cu_seqlens_padded']
            )
            pad_packed_batch_before_cp_slice(packed, self.config, cp_group.size(), 1)
            positions = torch.zeros((3, packed['tokens'].numel()), device=device, dtype=torch.long)
            for slot, sid in enumerate(assignment[self.domain_rank]):
                pos = samples[sid]['position_ids'].to(device)
                start = int(packed['cu_seqlens_padded'][slot])
                positions[:, start : start + pos.shape[-1]] = pos
            packing = _packing(packed, cp_group)
            kwargs = dict(
                input_ids=packed['tokens'].unsqueeze(0),
                position_ids=positions.unsqueeze(1),
                labels=packed['labels'].unsqueeze(0),
                loss_mask=packed['loss_mask'].unsqueeze(0),
                packing_kwargs=packing,
                padding_mask=packed['padding_mask'].unsqueeze(0),
            )
            self.rounds.append(
                dict(round_id=round_id, plans=plans, cp_group=cp_group, kwargs=kwargs)
            )
        return (
            RerunDataIterator(iter(self.rounds)),
            len(self.rounds),
            float(sum(length for _, length in lengths)),
            float(sum(length * length for _, length in lengths)),
        )

    def _return_feature_gradients(self):
        if self.pending is not None:
            self.source_gradient.add_(self.bridge.backward(self.pending))
            self.pending = None

    def forward(self, item, wrapped_model):
        if self.training:
            self._return_feature_gradients()
        transfer = self.bridge.forward(self.source, item['plans'], item['cp_group'])
        output, mask = wrapped_model(
            **item['kwargs'],
            modality_embeddings={'images': transfer.features} if transfer.features.shape[0] else {},
        )
        self.pending = transfer if self.training else None
        return output, partial(self.loss, mask, last=item['round_id'] == len(self.rounds) - 1)

    def loss(self, mask, output, *, last=False):
        raw = (output.float().view(-1) * mask.float().view(-1)).sum()
        count = mask.sum().detach().to(torch.int)
        metrics = {'lm loss': torch.stack((raw.detach(), count))}
        if last and not self.training:
            self._clear()
        return raw, count, metrics

    def finalize(self, model, num_tokens=None, **kwargs):
        """Finish only the modality boundary, then use native gradient finalization."""
        if not self.active or not self.training:
            raise RuntimeError('MIMO gradient finalization has no active training step')
        self._return_feature_gradients()
        if self.source.requires_grad:
            self.source.backward(self.source_gradient.to(self.source.dtype))
        finalize_model_grads(model, num_tokens, **kwargs)
        self._clear()

    def _clear(self):
        self.source = self.source_gradient = self.pending = None
        self.rounds = []
        self.active = False


def forward_step(data_iterator, model):
    """Native schedule callback, including the standard three-value loss contract."""
    coordinator = get_attr_wrapped_model(model, 'native_mimo_step')
    return coordinator.forward(next(data_iterator), model)
