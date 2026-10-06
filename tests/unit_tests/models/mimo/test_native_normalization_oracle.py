# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Independent supervised-token oracle through real native MIMO/DDP/optimizers.

This deliberately has no transformer kernels: the target is feature-boundary
backward, packed CP ownership, dense DP SUM and one global-token normalization.
MTP and MoE objectives have separate operator oracles. Decoder rounds are driven
explicitly, so pipeline-schedule timing is outside this test's scope.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from examples.mimo.native_step import NativeMimoStep
from megatron.core.datasets.data_schedule import wrap_data_iterator
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.mimo.partition.utils import PartitionAdapter, PartitionConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.models.mimo.test_native_mimo_step import _source_batch
from tests.unit_tests.test_utilities import Utils

_WIDTH = 512
_LR = 0.125


def _vision_inputs(media, device):
    if not media:
        return {}
    rows = torch.cat([torch.arange(item['length']) + 10 * item['image_id'] + 1 for item in media])
    return {'rows': rows.to(device=device, dtype=torch.float32)[:, None] / 32}


class _NormalizationModel(torch.nn.Module):
    def __init__(self, config, pg):
        super().__init__()
        self.config = config
        self.encoder_weight = torch.nn.Parameter(torch.full((_WIDTH,), 0.5, device='cuda'))
        self.decoder_weight = torch.nn.Parameter(torch.full((_WIDTH,), 0.25, device='cuda'))
        self.special_token_ids = {'images': 511}
        self.partition = PartitionAdapter(
            PartitionConfig.from_mp_config(
                config, max_seq_len=128, kv_format='thd', cp_group=pg.cp, tp_group=pg.tp
            )
        )

    def encode_modalities(self, inputs):
        return {'images': inputs['rows'] * self.encoder_weight} if inputs else {}

    def forward(self, input_ids, labels, loss_mask, packing_kwargs, modality_embeddings, **kwargs):
        embeddings = (input_ids.float()[..., None] / 8).expand(-1, -1, _WIDTH).clone()
        if modality_embeddings:
            embeddings[input_ids == 511] = modality_embeddings['images']
        embeddings, _, mask, _ = self.partition.shard(
            embeddings.transpose(0, 1),
            labels,
            loss_mask,
            PackedSeqParams(qkv_format='thd', **packing_kwargs),
        )
        output = (embeddings.transpose(0, 1) * self.decoder_weight).square().mean(dim=-1)
        return output, mask


def _independent_reference(source):
    """Original samples only; no scheduler, bridge, partition or loss helpers."""
    encoder = torch.full((_WIDTH,), 0.5, dtype=torch.float64, requires_grad=True)
    decoder = torch.full((_WIDTH,), 0.25, dtype=torch.float64, requires_grad=True)
    total = torch.zeros((), dtype=torch.float64)
    count = 0
    for sid, sample in source['samples'].items():
        length = int(sample['original_seq_len'])
        tokens = sample['tokens'][:length]
        values = (tokens.double()[:, None] / 8).expand(-1, _WIDTH).clone()
        if sample['media_ids']:
            assert sample['media_ids'] == [sid]
            image_rows = (torch.arange(4, dtype=torch.float64) + 10 * sid + 1) / 32
            values[tokens == 511] = image_rows[:, None] * encoder
        mask = sample['loss_mask'][:length].double()
        total = total + ((values * decoder).square().mean(dim=-1) * mask).sum()
        count += int(mask.sum())
    loss = total / count
    loss.backward()
    return {
        'loss': loss.item(),
        'tokens': count,
        'initial': {'encoder_weight': encoder.detach(), 'decoder_weight': decoder.detach()},
        'grads': {'encoder_weight': encoder.grad, 'decoder_weight': decoder.grad},
    }


def _comparison(actual, expected, *, rtol=2e-6, atol=2e-8):
    actual = actual.detach().cpu().double()
    expected = expected.detach().cpu().double()
    delta = actual - expected
    return {
        'elements': actual.numel(),
        'max_abs': delta.abs().max().item(),
        'relative_l2': (delta.norm() / expected.norm().clamp_min(1e-30)).item(),
        'rtol': rtol,
        'atol': atol,
        'passed': torch.allclose(actual, expected, rtol=rtol, atol=atol),
    }


@pytest.mark.parametrize('layout', ('cp1', 'static_cp2', 'dynamic_cp'))
@pytest.mark.parametrize('mask_kind', ('prompt_and_padding', 'sparse'))
@pytest.mark.parametrize('optimizer_kind', ('sgd', 'distributed_adam'))
def test_native_supervised_token_normalization(monkeypatch, layout, mask_kind, optimizer_kind):
    if int(os.environ.get('WORLD_SIZE', '1')) != 4:
        pytest.skip('This oracle uses exactly four CUDA ranks')
    dynamic = layout == 'dynamic_cp'
    cp_size = 1 if layout == 'cp1' else 2
    distributed = optimizer_kind == 'distributed_adam'
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=cp_size,
        expert_model_parallel_size=1,
        dynamic_context_parallel=dynamic,
        create_gloo_process_groups=False,
    )
    try:
        pg = ProcessGroupCollection.use_mpu_process_groups()
        config = TransformerConfig(
            num_layers=1,
            hidden_size=_WIDTH,
            num_attention_heads=1,
            params_dtype=torch.float32,
            use_cpu_initialization=True,
            context_parallel_size=cp_size,
            calculate_per_token_loss=True,
            sequence_packing_scheduler='default_dynamic_cp' if dynamic else 'dp_balanced',
            dynamic_context_parallel=dynamic,
            max_seqlen_per_dp_cp_rank={'cp1': 128, 'static_cp2': 64, 'dynamic_cp': 32}[layout],
            pad_packed_seq_alignment='max',
            hidden_dropout=0,
            attention_dropout=0,
            mtp_num_layers=None,
            moe_aux_loss_coeff=0,
        )
        module = _NormalizationModel(config, pg)
        ddp = DistributedDataParallel(
            config,
            DistributedDataParallelConfig(
                grad_reduce_in_fp32=True,
                use_distributed_optimizer=distributed,
                overlap_grad_reduce=False,
                overlap_param_gather=False,
                average_in_collective=False,
            ),
            module,
            pg_collection=pg,
        )
        opt_config = OptimizerConfig(
            optimizer='adam' if distributed else 'sgd',
            lr=_LR,
            weight_decay=0,
            sgd_momentum=0,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_eps=1e-8,
            clip_grad=0,
            params_dtype=torch.float32,
            use_distributed_optimizer=distributed,
        )
        optimizer = get_megatron_optimizer(
            opt_config, [ddp], config_overrides={}, pg_collection=pg, use_gloo_process_groups=False
        )
        coordinator = NativeMimoStep(
            module,
            pg,
            SimpleNamespace(
                overlap_grad_reduce=False,
                overlap_param_gather=False,
                mimo_metrics_dir=None,
                rerun_mode='disabled',
            ),
        )
        monkeypatch.setattr('examples.mimo.data.qwen35_native.build_vision_inputs', _vision_inputs)
        source = _source_batch()
        if mask_kind == 'sparse':
            for sample in source['samples'].values():
                sample['loss_mask'].zero_()
            source['samples'][0]['loss_mask'][4:8] = 1
        reference = _independent_reference(source)
        ddp.zero_grad_buffer()
        optimizer.zero_grad()
        iterator, rounds, _, _ = wrap_data_iterator(iter([source]), config, 1, pg)
        local_tokens = torch.zeros((), device='cuda', dtype=torch.int)
        runtime_sizes = []
        zero_supervision_rounds = 0
        for _ in range(rounds):
            item = next(iterator)
            runtime_sizes.append(item['cp_group'].size())
            output, loss_fn = coordinator.forward(item, ddp)
            raw_loss, tokens, _ = loss_fn(output)
            raw_loss.backward()
            local_tokens += tokens
            zero_supervision_rounds += int(tokens.item() == 0)
        original_local_count = int(local_tokens.item())
        coordinator.finalize([ddp], local_tokens, pg_collection=pg)
        checks = {
            'loss': _comparison(
                torch.tensor(coordinator.metrics['loss'], dtype=torch.float64),
                torch.tensor(reference['loss'], dtype=torch.float64),
            )
        }
        inner = optimizer.chained_optimizers[0]
        owned_ranges = {}
        for name, parameter in module.named_parameters():
            if distributed:
                if parameter not in inner.model_param_group_index_map:
                    continue
                shard = inner._get_model_param_range_map(parameter)['param']
                start, end = shard.start, shard.end
            else:
                start, end = 0, parameter.numel()
            owned_ranges[name] = [start, end]
            checks[f'{name}.normalized_gradient'] = _comparison(
                parameter.main_grad.view(-1)[start:end], reference['grads'][name][start:end], atol=0
            )
        update_successful, _, _ = optimizer.step()
        for name, parameter in module.named_parameters():
            gradient = reference['grads'][name]
            if distributed:
                expected_delta = -_LR * gradient / (gradient.abs() + opt_config.adam_eps)
                if name in owned_ranges:
                    start, end = owned_ranges[name]
                    states = inner._get_main_param_and_optimizer_states(parameter)
                    checks[f'{name}.exp_avg'] = _comparison(
                        states['exp_avg'], (1 - opt_config.adam_beta1) * gradient[start:end], atol=0
                    )
                    checks[f'{name}.exp_avg_sq'] = _comparison(
                        states['exp_avg_sq'],
                        (1 - opt_config.adam_beta2) * gradient[start:end].square(),
                        atol=0,
                    )
            else:
                expected_delta = -_LR * gradient
            checks[f'{name}.parameter_delta'] = _comparison(
                parameter.detach().cpu().double() - reference['initial'][name],
                expected_delta,
                atol=4e-8,
            )
        report = {
            'layout': layout,
            'mask': mask_kind,
            'optimizer': optimizer_kind,
            'rank': dist.get_rank(),
            'rounds': rounds,
            'runtime_cp_sizes': runtime_sizes,
            'zero_supervision_rounds': zero_supervision_rounds,
            'local_supervised_tokens': original_local_count,
            'finalized_global_tokens': int(local_tokens.item()),
            'reference_global_tokens': reference['tokens'],
            'native_loss': coordinator.metrics['loss'],
            'reference_loss': reference['loss'],
            'group_sizes': {
                name: getattr(pg, name).size() for name in ('tp', 'pp', 'cp', 'dp', 'dp_cp')
            },
            'dense_gradient_scaling_factors': [
                buffer.gradient_scaling_factor for buffer in ddp.buffers
            ],
            'owned_ranges': owned_ranges,
            'checks': checks,
            'update_successful': bool(update_successful),
            'scope': 'LM supervised mean; native encoder backward, DDP/finalize/optimizer; no transformer, MTP or MoE kernels',
        }
        reports = [None] * dist.get_world_size()
        dist.all_gather_object(reports, report, group=pg.dp_cp)
        observed_cp_sizes = {size for item in reports for size in item['runtime_cp_sizes']}
        expected_cp_sizes = {'cp1': {1}, 'static_cp2': {2}, 'dynamic_cp': {1, 2, 4}}[layout]
        report['observed_global_cp_sizes'] = sorted(observed_cp_sizes)
        report['expected_global_cp_sizes'] = sorted(expected_cp_sizes)
        metadata_ok = (
            sum(item['local_supervised_tokens'] for item in reports) == reference['tokens']
            and all(item['finalized_global_tokens'] == reference['tokens'] for item in reports)
            and all(item['dense_gradient_scaling_factors'] == [1.0] for item in reports)
            and all(item['update_successful'] for item in reports)
            and observed_cp_sizes == expected_cp_sizes
            and not coordinator.active
        )
        if mask_kind == 'sparse':
            metadata_ok = metadata_ok and any(
                item['local_supervised_tokens'] == 0 for item in reports
            )
        if distributed:
            global_owned_ranges = {
                name: sorted(
                    item['owned_ranges'][name] for item in reports if name in item['owned_ranges']
                )
                for name in reference['grads']
            }
            report['global_owned_ranges'] = global_owned_ranges
            metadata_ok = (
                metadata_ok
                and all(
                    intervals
                    and intervals[0][0] == 0
                    and intervals[-1][1] == _WIDTH
                    and all(0 <= start < end <= _WIDTH for start, end in intervals)
                    and all(left[1] == right[0] for left, right in zip(intervals, intervals[1:]))
                    for intervals in global_owned_ranges.values()
                )
                and all(item['owned_ranges'] for item in reports)
            )
        report['metadata_passed'] = bool(metadata_ok)
        report['all_ranks_passed'] = metadata_ok and all(
            check['passed'] for item in reports for check in item['checks'].values()
        )
        directory = os.environ.get('MIMO_OBJECTIVE_ORACLE_REPORT_DIR')
        if directory:
            destination = Path(directory)
            destination.mkdir(parents=True, exist_ok=True)
            filename = f'native_{layout}_{mask_kind}_{optimizer_kind}_rank{dist.get_rank()}.json'
            (destination / filename).write_text(json.dumps(report, indent=2) + '\n')
        assert report['all_ranks_passed'], json.dumps(report, indent=2)
    finally:
        Utils.destroy_model_parallel()
