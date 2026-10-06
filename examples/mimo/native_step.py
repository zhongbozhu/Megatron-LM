# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Colocated vision routing inside Megatron's native PP1 training schedule.

Only the modality boundary has an explicit backward. Decoder backward, gradient
buffers, DP/EP reductions, optimizer updates and checkpoints remain native.
"""

import json
import sys
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist

from examples.mimo.data.packed_multimodal import assign_encoder_media, build_round_plans
from megatron.core import parallel_state, tensor_parallel
from megatron.core.datasets.data_schedule import _build_thd_padding_mask
from megatron.core.datasets.data_schedule_utils import (
    build_packed_microbatches,
    pad_packed_batch_before_cp_slice,
)
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.models.mimo.comm.pack_bridge import PackFeatureBridge
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.rerun_state_machine import RerunDataIterator
from megatron.core.transformer.moe.global_aux_loss import GlobalAuxLossStep
from megatron.core.transformer.multi_token_prediction import (
    MTPLossLoggingHelper,
    get_mtp_loss_token_counts,
)
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
        if config.hidden_dropout or config.attention_dropout:
            raise ValueError('Two-pass global auxiliary statistics currently require dropout=0')
        if getattr(args, 'mimo_audit_token_assignments', False) and not getattr(
            args, 'mimo_gradient_diagnostics', False
        ):
            raise ValueError('Token assignment auditing requires --mimo-gradient-diagnostics')
        self.model, self.pg, self.args = model, pg_collection, args
        self.config = config
        self.domain = tuple(dist.get_process_group_ranks(pg_collection.dp_cp))
        self.rank = dist.get_rank()
        self.domain_rank = self.domain.index(self.rank)
        self.bridge = PackFeatureBridge(pg_collection.dp_cp)
        self.reference_bridge = None
        self.active = False
        self.aux = None
        self.execution_diagnostics = None
        self.boundary_diagnostics = None
        self.fixed_routing = None
        config.sequence_packing_data_adapter = self.prepare
        config.finalize_model_grads_func = self.finalize

    def prepare(self, data_iterator, num_microbatches, scheduler, pg_collection):
        """Reuse the selected core scheduler and THD builder; attach visual routes."""
        from examples.mimo.data.qwen35_native import build_vision_inputs

        if self.active:
            raise RuntimeError('Previous MIMO step has not completed its backward boundary')
        source_batch = next(data_iterator)
        self.step = source_batch['step']
        samples, media = source_batch['samples'], source_batch['media']
        if sorted(samples) != list(range(len(samples))):
            raise ValueError('Source sample IDs must be contiguous')
        lengths = [(sid, int(sample['padded_seq_len'])) for sid, sample in samples.items()]
        assignments = scheduler.get_groups_and_subsamples(lengths)
        pack_record = self._diagnostic_path('mimo_fixed_packing_record')
        pack_replay = self._diagnostic_path('mimo_fixed_packing_replay')
        schedule_replay = self._diagnostic_path('mimo_diagnostic_schedule_replay')
        schedule_audit = self._diagnostic_path('mimo_diagnostic_schedule_audit_dir')
        if schedule_audit and not schedule_replay:
            raise ValueError('Diagnostic schedule auditing requires schedule replay')
        if schedule_replay:
            from examples.mimo.fixed_packing import apply_dynamic_schedule_replay

            if not scheduler.is_dynamic_cp or pack_record or pack_replay:
                raise ValueError('Diagnostic schedule replay requires DCP without fixed packing')
            assignments = apply_dynamic_schedule_replay(
                samples,
                media,
                assignments,
                step=self.step,
                domain_ranks=self.domain,
                cp_group_sizes=[
                    size for size in scheduler.cp_group_sizes if size >= scheduler.min_cp_size
                ],
                config=self.config,
                replay_path=schedule_replay,
                audit_path=schedule_audit,
                write_audit=self.domain_rank == 0,
            )
        if pack_record or pack_replay:
            from examples.mimo.fixed_packing import apply_fixed_packing

            if scheduler.is_dynamic_cp:
                raise ValueError('Fixed packing is a static-CP control; DCP uses its own scheduler')
            assignments = apply_fixed_packing(
                samples,
                media,
                assignments,
                step=self.step,
                domain_ranks=self.domain,
                cp_size=self.pg.cp.size(),
                config=self.config,
                record_path=pack_record,
                replay_path=pack_replay,
                write_record=self.domain_rank == 0,
            )
        scheduled = [
            sid
            for assignment in assignments
            for pack in dict.fromkeys(tuple(ids) for ids in assignment)
            for sid in pack
        ]
        if sorted(scheduled) != sorted(samples):
            raise ValueError('Every source sample must appear exactly once in the schedule')
        encoder_tasks, slices = assign_encoder_media(media, self.domain)
        if getattr(self.args, 'mimo_feature_transport', 'pack') == 'direct_reference':
            from examples.mimo.direct_feature_reference import DirectFeatureReference

            self.reference_bridge = DirectFeatureReference(
                self.pg.dp_cp, samples, encoder_tasks, assignments
            )
        diagnostic_dir = getattr(self.args, 'mimo_execution_diagnostics_dir', None)
        diagnostic_steps = getattr(self.args, 'mimo_diagnostic_steps', None)
        if diagnostic_steps is not None and self.step not in diagnostic_steps:
            diagnostic_dir = None
        if not torch.is_grad_enabled():
            diagnostic_dir = None
        if getattr(self.args, 'mimo_boundary_reference', None) or diagnostic_dir:
            self.boundary_media = [
                dict(
                    image_id=item['image_id'],
                    source_id=item['source_id'],
                    offset=slices[item['image_id']].offset,
                    length=item['length'],
                    grid=item['grid'].tolist(),
                )
                for item in encoder_tasks[self.rank]
            ]
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
        if diagnostic_dir:
            from examples.mimo.boundary_diagnostics import BoundaryDiagnostics
            from examples.mimo.execution_diagnostics import ExecutionDiagnostics

            self.boundary_diagnostics = BoundaryDiagnostics(
                Path(diagnostic_dir) / 'boundary', self.rank
            )
            self.boundary_diagnostics.save_producer(
                self.step, self.source, self.boundary_media, num_rounds=len(assignments)
            )
            self.execution_diagnostics = ExecutionDiagnostics(
                self.model,
                Path(diagnostic_dir) / 'decoder',
                self.rank,
                self.step,
                samples,
                gdn_layers=getattr(self.args, 'mimo_diagnostic_gdn_layers', (0,)),
                moe_layers=getattr(self.args, 'mimo_diagnostic_moe_layers', ()),
                capture_full_input=getattr(self.args, 'mimo_diagnostic_full_decoder_input', False),
            )
        self.pending = None
        self.training = torch.is_grad_enabled()
        self.active = True
        self.loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        self.token_count = torch.zeros_like(self.loss_sum)
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
        full = build_packed_microbatches(text_samples, [[list(samples)]], 0, device)[0]
        self.global_tokens = full['loss_mask'].sum()
        if self.global_tokens <= 0:
            raise ValueError('A source global batch needs supervised tokens')
        mtp_counts = None
        if self.config.mtp_num_layers:
            mtp_counts = get_mtp_loss_token_counts(
                full['loss_mask'].unsqueeze(0),
                self.config.mtp_num_layers,
                mtp_input_mask=self.model._materialize_mtp_input_mask(
                    full['tokens'].unsqueeze(0), self.model.special_token_ids
                ),
                packed_seq_params=PackedSeqParams(qkv_format='thd', **_packing(full, self.pg.tp)),
                cp_group=self.pg.tp,
            )
        self.rounds = []
        cp_sizes = []
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
            packing['mtp_loss_token_counts'] = mtp_counts
            kwargs = dict(
                input_ids=packed['tokens'].unsqueeze(0),
                position_ids=positions.unsqueeze(1),
                labels=packed['labels'].unsqueeze(0),
                loss_mask=packed['loss_mask'].unsqueeze(0),
                packing_kwargs=packing,
                padding_mask=packed['padding_mask'].unsqueeze(0),
            )
            self.rounds.append(
                dict(
                    round_id=round_id,
                    plans=plans,
                    cp_group=cp_group,
                    kwargs=kwargs,
                    diagnostic_sample_ids=tuple(assignment[self.domain_rank]),
                )
            )
            cp_sizes.extend(len(p.cp_ranks) for p in plans)
        self.metrics = dict(
            step=self.step,
            training=self.training,
            source_samples=len(samples),
            source_images=len(media),
            source_tokens=sum(int(s['original_seq_len']) for s in samples.values()),
            longest_sample=max(int(s['original_seq_len']) for s in samples.values()),
            rounds=len(self.rounds),
            cp_sizes=cp_sizes,
        )
        if self.domain_rank == 0:
            print('MIMO_NATIVE_SCHEDULE ' + json.dumps(self.metrics), flush=True)
        route_record = self._diagnostic_path('mimo_fixed_routing_record')
        route_replay = self._diagnostic_path('mimo_fixed_routing_replay')
        if route_record or route_replay:
            from examples.mimo.fixed_routing import FixedRouting

            if route_record and self.training and not self.config.moe_aux_loss_coeff:
                raise ValueError('Fixed routing recording requires the auxiliary statistics pass')
            self.fixed_routing = FixedRouting(
                self.model,
                route_record or route_replay,
                'record' if route_record else 'replay',
                self.step,
                samples,
                group=self.pg.dp_cp,
                media=media,
                context={
                    name: getattr(self.args, name, None)
                    for name in ('load', 'ckpt_step', 'mimo_pretrained_checkpoint', 'seed')
                },
            )
        if self.training and self.config.moe_aux_loss_coeff:
            try:
                self._collect_auxiliary_statistics()
                if self.fixed_routing is not None:
                    self.fixed_routing.seal_recording()
                self._empty_unused_memory()
            except BaseException:
                self._clear()
                raise
        else:
            self.aux = None
        # The stats pass is not an additional training/evaluation observation.
        if self.config.mtp_num_layers and self.training:
            MTPLossLoggingHelper.clean_metrics_in_tracker()
        return (
            RerunDataIterator(iter(self.rounds)),
            len(self.rounds),
            float(sum(length for _, length in lengths)),
            float(sum(length * length for _, length in lengths)),
        )

    def _diagnostic_path(self, name):
        directory = getattr(self.args, name, None)
        if directory is None:
            return None
        # Evaluation has a separate source iterator and may also start at step 0.
        return Path(directory) / ('training' if torch.is_grad_enabled() else 'evaluation')

    def _collect_auxiliary_statistics(self):
        self.aux = GlobalAuxLossStep(
            self.model,
            self.pg.dp_cp,
            two_pass=True,
            audit_replay=getattr(self.args, 'mimo_gradient_diagnostics', False),
            audit_token_assignments=getattr(self.args, 'mimo_audit_token_assignments', False),
        )
        self.aux.__enter__()
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state()
        tracker_rng = tensor_parallel.get_cuda_rng_tracker().get_states()
        language_model = self.model.language_model
        post_process = language_model.post_process
        try:
            # GPT runs its MTP blocks before the post_process guard. Keep those
            # routers in the statistics pass, but omit the unused vocabulary
            # projections and cross-entropies for both main and MTP outputs.
            language_model.post_process = False
            # Match training's grad-mode/compiler specialization. Only detached
            # router statistics survive; each decoder graph is released below.
            with torch.enable_grad():
                for item in self.rounds:
                    self.aux.set_round(item['round_id'])
                    transfer = self._feature_transfer(item)
                    self._begin_diagnostics(item, 'statistics', transfer)
                    try:
                        self.model(
                            **item['kwargs'],
                            modality_embeddings=(
                                {'images': transfer.features} if transfer.features.shape[0] else {}
                            ),
                        )
                    finally:
                        if getattr(self, 'execution_diagnostics', None) is not None:
                            self.execution_diagnostics.flush()
                        # Selective checkpoints are retained on decoder modules.
                        # Dropping only the final output leaves their saved inputs
                        # alive. Do not clear the independent encoder's graph.
                        for module in language_model.modules():
                            for value in vars(module).values():
                                if isinstance(value, tensor_parallel.CheckpointWithoutOutput):
                                    value.ctx = value.outputs = value.run_function = None
                                    value.rng_states = None
            self.metrics['aux'] = self.aux.finalize()
            self.aux.begin_training(self.global_tokens)
        except BaseException:
            self.aux.__exit__(*sys.exc_info())
            self.aux = None
            raise
        finally:
            language_model.post_process = post_process
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng)
            tensor_parallel.get_cuda_rng_tracker().set_states(tracker_rng)

    def _empty_unused_memory(self):
        # Decoder temporaries can fill the caching allocator while NCCL/Triton
        # still need allocations outside it. Honor the native opt-in at the
        # extra statistics/communication boundaries introduced by MIMO.
        if getattr(self.args, 'empty_unused_memory_level', 0) >= 1:
            torch.cuda.empty_cache()

    def _return_feature_gradients(self):
        if self.pending is not None:
            if self.execution_diagnostics is not None:
                self.execution_diagnostics.flush()
            if self.boundary_diagnostics is not None:
                self.boundary_diagnostics.save_receiver_gradient(
                    self.step, self.pending_round, self.pending, normalizer=self.global_tokens
                )
            self._empty_unused_memory()
            bridge = getattr(self, 'reference_bridge', None) or self.bridge
            self.source_gradient.add_(bridge.backward(self.pending))
            self.pending = None

    def _feature_transfer(self, item):
        if getattr(self, 'reference_bridge', None) is not None:
            return self.reference_bridge.forward(self.source, item['round_id'], item['cp_group'])
        return self.bridge.forward(self.source, item['plans'], item['cp_group'])

    def _begin_diagnostics(self, item, phase, transfer):
        if getattr(self, 'fixed_routing', None) is not None:
            self.fixed_routing.begin_round(item)
        if getattr(self, 'boundary_diagnostics', None) is not None:
            self.boundary_diagnostics.save_receiver(self.step, item['round_id'], phase, transfer)
            self.execution_diagnostics.begin(item, phase, transfer)

    def forward(self, item, wrapped_model):
        if self.training:
            self._return_feature_gradients()
        if self.aux is not None:
            self.aux.set_round(item['round_id'])
        transfer = self._feature_transfer(item)
        self._begin_diagnostics(item, 'training' if self.training else 'evaluation', transfer)
        output, mask = wrapped_model(
            **item['kwargs'],
            modality_embeddings={'images': transfer.features} if transfer.features.shape[0] else {},
        )
        self.pending = transfer if self.training else None
        self.pending_round = item['round_id']
        return output, partial(self.loss, mask, last=item['round_id'] == len(self.rounds) - 1)

    def loss(self, mask, output, *, last=False):
        raw = (output.float().view(-1) * mask.float().view(-1)).sum()
        if not torch.isfinite(raw):
            raise FloatingPointError('Non-finite native MIMO language-model loss')
        count = mask.sum().detach().to(torch.int)
        self.loss_sum.add_(raw.detach().double())
        self.token_count.add_(count.double())
        metrics = {'lm loss': torch.stack((raw.detach(), count))}
        if self.aux is not None:
            aux = self.metrics['aux']['loss']
            metrics['mimo global aux loss'] = torch.stack((count * aux, count))
        if last and not self.training:
            if self.fixed_routing is not None:
                self.fixed_routing.seal_recording()
            self._record_metrics()
            self._clear()
        return raw, count, metrics

    def finalize(self, model, num_tokens=None, **kwargs):
        """Finish only the modality boundary, then use native gradient finalization."""
        if not self.active or not self.training:
            raise RuntimeError('MIMO gradient finalization has no active training step')
        self._return_feature_gradients()
        if self.boundary_diagnostics is not None:
            self.boundary_diagnostics.save_producer_gradient(
                self.step, self.source_gradient, normalizer=self.global_tokens
            )
        if getattr(self.args, 'mimo_boundary_reference', None):
            from examples.mimo.full_gradient_validation import validate_encoder_boundary

            self.metrics['encoder_boundary_validation'] = validate_encoder_boundary(
                self.source,
                self.source_gradient,
                self.boundary_media,
                self.args.mimo_boundary_reference,
                step=self.step,
                num_tokens=self.global_tokens,
                write_reference=self.args.mimo_write_gradient_reference,
                group=self.pg.dp_cp,
            )
        if self.source.requires_grad:
            self.source.backward(self.source_gradient.to(self.source.dtype))
        if getattr(self.args, 'mimo_gradient_diagnostics', False):
            from examples.mimo.gradient_diagnostics import collect_gradient_diagnostics

            self.metrics['gradient_diagnostics'] = collect_gradient_diagnostics(
                self.model,
                self.pg.dp_cp,
                self.global_tokens,
                ep_group=self.pg.ep,
                required_components=('vision', 'merger', 'decoder', 'routers', 'experts', 'mtp'),
            )
        if self.aux is not None and self.aux.audit_replay:
            self.metrics['aux_replay'] = self.aux.replay_metrics()
        finalize_model_grads(model, num_tokens, **kwargs)
        if getattr(self.args, 'mimo_gradient_reference', None):
            from examples.mimo.full_gradient_validation import validate_full_gradients

            self.metrics['full_gradient_validation'] = validate_full_gradients(
                model,
                self.args.mimo_gradient_reference,
                write_reference=self.args.mimo_write_gradient_reference,
                group=self.pg.dp_cp,
            )
        self._record_metrics()
        self._clear()

    def _record_metrics(self):
        if getattr(self, 'fixed_routing', None) is not None:
            self.metrics['fixed_routing'] = dict(self.fixed_routing.metrics)
        totals = torch.stack((self.loss_sum, self.token_count))
        dist.all_reduce(totals, group=self.pg.dp_cp)
        if totals[1] != self.global_tokens:
            raise RuntimeError(f'Supervised token mismatch: {totals[1]} vs {self.global_tokens}')
        self.metrics.update(loss=(totals[0] / totals[1]).item(), supervised_tokens=int(totals[1]))
        tracker = MTPLossLoggingHelper.tracker
        if self.training and self.config.mtp_num_layers and 'loss_sums' in tracker:
            values = torch.stack([tracker[k].clone() for k in ('loss_sums', 'loss_token_counts')])
            dist.all_reduce(values, group=self.pg.dp_cp)
            self.metrics['mtp_loss'] = (values[0] / values[1].clamp_min(1)).tolist()
            self.metrics['mtp_tokens'] = values[1].tolist()
        self.metrics['max_memory_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
        if self.domain_rank == 0:
            summary = dict(self.metrics)
            if 'aux' in summary:
                summary['aux'] = {'loss': summary['aux']['loss']}
            print('MIMO_NATIVE_METRICS ' + json.dumps(summary), flush=True)
            if self.args.mimo_metrics_dir:
                path = Path(self.args.mimo_metrics_dir)
                path.mkdir(parents=True, exist_ok=True)
                with (path / 'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(self.metrics) + '\n')

    def _clear(self):
        if getattr(self, 'fixed_routing', None) is not None:
            self.fixed_routing.close()
            self.fixed_routing = None
        if self.execution_diagnostics is not None:
            self.execution_diagnostics.close()
        self.execution_diagnostics = self.boundary_diagnostics = None
        if self.aux is not None:
            self.aux.__exit__(None, None, None)
        self.aux = None
        self.source = self.source_gradient = self.pending = None
        self.reference_bridge = None
        self.rounds = []
        self.active = False


def forward_step(data_iterator, model):
    """Native schedule callback, including the standard three-value loss contract."""
    coordinator = get_attr_wrapped_model(model, 'native_mimo_step')
    return coordinator.forward(next(data_iterator), model)
