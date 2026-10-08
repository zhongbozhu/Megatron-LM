# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Native adapter contracts with real packing and distributed feature transport.

The tiny differentiable model isolates scheduling, metadata and encoder-backward
lifetime from transformer kernels. Native DDP/optimizer normalization is tested
against a separate original-sample objective below.
"""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from examples.mimo import native_step
from examples.mimo.native_step import NativeMimoStep, NativeSourceIterator
from megatron.core.datasets.data_schedule import wrap_data_iterator
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.mimo.partition.utils import PartitionAdapter, PartitionConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(scope="module")
def native_groups():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    world, rank = dist.get_world_size(), dist.get_rank()
    if world % 4:
        pytest.skip("Native MIMO adapter tests require a multiple of four ranks")
    owned, created = {}, []
    for base in range(0, world, 4):
        for ranks in (
            tuple(range(base, base + 4)),
            (base, base + 1),
            (base + 2, base + 3),
            (base, base + 2),
            (base + 1, base + 3),
            *((peer,) for peer in range(base, base + 4)),
        ):
            group = dist.new_group(list(ranks), backend="nccl")
            if rank in ranks:
                owned[ranks] = group
                created.append(group)
    base, local_rank = rank // 4 * 4, rank % 4
    singleton = owned[(rank,)]
    cp = owned[tuple(range(base + local_rank // 2 * 2, base + local_rank // 2 * 2 + 2))]
    domain = owned[tuple(range(base, base + 4))]
    dp = owned[(base + local_rank % 2, base + local_rank % 2 + 2)]
    yield SimpleNamespace(tp=singleton, pp=singleton, cp=cp, dp=dp, dp_cp=domain), {
        1: singleton,
        2: cp,
        4: domain,
    }
    for group in reversed(created):
        dist.destroy_process_group(group)


def _source_batch(pure_text=False):
    samples, media = {}, []
    for sid, physical in enumerate((96, 64, 48, 32, 32, 16, 16, 16, 16)):
        logical = physical - 3
        tokens = torch.full((physical,), sid + 1, dtype=torch.long)
        tokens[logical:] = 0
        image_ids = []
        if not pure_text and sid != 8:
            tokens[1:5] = 511
            image_ids = [sid]
            media.append(dict(image_id=sid, size=(16, 16), length=4))
        loss_mask = torch.zeros(physical)
        loss_mask[:logical] = 1
        loss_mask[1:4] = 0
        positions = torch.zeros(3, physical, dtype=torch.long)
        positions[:, :logical] = (
            torch.arange(logical)[None, :] + 100 * sid + 1000 * torch.arange(3)[:, None]
        )
        samples[sid] = dict(
            tokens=tokens,
            labels=tokens.clone(),
            loss_mask=loss_mask,
            position_ids=positions,
            original_seq_len=logical,
            padded_seq_len=physical,
            media_ids=image_ids,
        )
    return dict(step=7, samples=samples, media=media)


def _vision_inputs(media, device):
    if not media:
        return {}
    rows = torch.cat([torch.arange(4) + 10 * item['image_id'] + 1 for item in media])
    return dict(rows=rows.to(device=device, dtype=torch.float32)[:, None].expand(-1, 4))


class _TinyMimo(torch.nn.Module):
    def __init__(self, *, dynamic):
        super().__init__()
        self.encoder_weight = torch.nn.Parameter(torch.tensor(2.0, device='cuda'))
        self.decoder_weight = torch.nn.Parameter(torch.tensor(0.25, device='cuda'))
        self.special_token_ids = {'images': 511}
        self.config = SimpleNamespace(
            hidden_size=4,
            params_dtype=torch.float32,
            calculate_per_token_loss=True,
            sequence_packing_scheduler='default_dynamic_cp' if dynamic else 'dp_balanced',
            dynamic_context_parallel=dynamic,
            min_dynamic_context_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            microbatch_group_size_per_vp_stage=None,
            max_seqlen_per_dp_cp_rank=32 if dynamic else 64,
            pad_packed_seq_alignment='max',
        )

    def encode_modalities(self, inputs):
        return {'images': inputs['rows'] * self.encoder_weight} if inputs else {}

    def forward(self, input_ids, loss_mask, packing_kwargs, modality_embeddings, **kwargs):
        values = input_ids.float().clone()
        if modality_embeddings:
            values[input_ids == 511] = modality_embeddings['images'][:, 0]
        group = packing_kwargs['cp_group']
        # A deliberately simple disjoint partition. Actual zigzag model slicing
        # has its own tests; this isolates boundary communication and lifetime.
        values = values[:, group.rank() :: group.size()]
        mask = loss_mask[:, group.rank() :: group.size()]
        return (values * self.decoder_weight).square(), mask


def _coordinator(groups, monkeypatch, *, dynamic):
    pg, runtime_groups = groups
    monkeypatch.setattr('examples.mimo.data.qwen35_native.build_vision_inputs', _vision_inputs)
    monkeypatch.setattr(
        native_step.parallel_state,
        'get_dynamic_data_context_parallel_groups',
        lambda group_size: runtime_groups[group_size],
    )
    model = _TinyMimo(dynamic=dynamic)
    args = SimpleNamespace(
        overlap_grad_reduce=False, overlap_param_gather=False, rerun_mode='disabled'
    )
    return model, NativeMimoStep(model, pg, args), pg


def test_source_resume_reconstructs_the_same_global_batch():
    class Dataset:
        def build_global_batch(self, step, size):
            return tuple(range(step * size, (step + 1) * size)), ()

    source = NativeSourceIterator(Dataset(), 64)
    for _ in range(5):
        next(source)
    resumed = NativeSourceIterator(Dataset(), 64, consumed_samples=320)
    assert next(source) == next(resumed)
    assert next(source) == next(resumed)
    with pytest.raises(ValueError, match='complete source global batch'):
        NativeSourceIterator(Dataset(), 64, consumed_samples=321)


@pytest.mark.parametrize('dynamic', (False, True))
def test_native_packing_preserves_mrope_and_masks(native_groups, monkeypatch, dynamic):
    model, coordinator, pg = _coordinator(native_groups, monkeypatch, dynamic=dynamic)
    source = _source_batch()
    snapshots = {
        sid: {key: value.clone() for key, value in sample.items() if torch.is_tensor(value)}
        for sid, sample in source['samples'].items()
    }
    iterator, count, _, _ = wrap_data_iterator(
        iter([source]), model.config, num_microbatches=17, pg_collection=pg
    )
    assert count == len(coordinator.rounds)
    assert count != 17  # The native scheduler controls physical microbatch count.

    observed_tail = False
    for _ in range(count):
        item = next(iterator)
        kwargs = item['kwargs']
        packing = kwargs['packing_kwargs']
        assert kwargs['position_ids'].shape == (3, 1, kwargs['input_ids'].numel())
        assert (
            kwargs['input_ids'].numel()
            == model.config.max_seqlen_per_dp_cp_rank * item['cp_group'].size()
        )
        boundaries = packing['cu_seqlens_q_padded'].tolist()
        logical = packing['cu_seqlens_q'].diff().tolist()
        for slot, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            if kwargs['padding_mask'][0, start]:
                observed_tail = True
                assert kwargs['padding_mask'][0, start:end].all()
                assert not kwargs['loss_mask'][0, start:end].any()
                assert not kwargs['position_ids'][:, 0, start:end].any()
                continue
            sid = int(kwargs['input_ids'][0, start]) - 1
            sample = source['samples'][sid]
            assert logical[slot] == sample['original_seq_len']
            valid_end = start + sample['original_seq_len']
            torch.testing.assert_close(
                kwargs['position_ids'][:, 0, start:valid_end].cpu(),
                sample['position_ids'][:, : sample['original_seq_len']],
            )
            assert not kwargs['padding_mask'][0, start:valid_end].any()
            assert kwargs['padding_mask'][0, valid_end:end].all()
            assert not kwargs['loss_mask'][0, valid_end:end].any()
        plan = next(plan for plan in item['plans'] if dist.get_rank() in plan.cp_ranks)
        assert plan.num_rows == int((kwargs['input_ids'] == 511).sum())
    tail = torch.tensor(int(observed_tail), device='cuda')
    dist.all_reduce(tail, group=pg.dp_cp)
    assert tail.item() > 0
    for sid, sample in snapshots.items():
        for key, value in sample.items():
            torch.testing.assert_close(source['samples'][sid][key], value)
    with pytest.raises(RuntimeError, match='Previous MIMO step'):
        coordinator.prepare(iter([source]), 1, None, pg)
    coordinator._clear()


@pytest.mark.parametrize('dynamic', (False, True))
def test_native_boundary_lifecycle_across_source_steps(native_groups, monkeypatch, dynamic):
    """Mixed → text-only → changed images reuse one coordinator without stale graphs."""
    model, coordinator, pg = _coordinator(native_groups, monkeypatch, dynamic=dynamic)
    backward_steps, finalized = [], []
    model.encoder_weight.register_hook(lambda gradient: backward_steps.append(step))
    for step, pure_text in enumerate((False, True, False)):
        model.zero_grad(set_to_none=True)
        source = _source_batch(pure_text=pure_text)
        source['step'] = step
        if step == 2:
            # Reverse sample order and replace image identities. This changes
            # encoder ownership and pack order without replacing the coordinator.
            # Static packing uses source positions as IDs, so reindex the new order.
            source['samples'] = dict(enumerate(reversed(source['samples'].values())))
            for sample in source['samples'].values():
                sample['media_ids'] = [100 + image for image in sample['media_ids']]
            for image in source['media']:
                image['image_id'] += 100
        iterator, count, _, _ = wrap_data_iterator(iter([source]), model.config, 1, pg)
        local_tokens = torch.zeros((), device='cuda', dtype=torch.int)
        before = list(backward_steps)
        for _ in range(count):
            output, loss = coordinator.forward(next(iterator), model)
            raw, tokens, _ = loss(output)
            raw.backward()
            local_tokens += tokens
            assert model.encoder_weight.grad is None
            assert backward_steps == before

        def native_finalize(model_chunks, num_tokens, **kwargs):
            assert model_chunks == [model] and num_tokens is local_tokens
            assert model.decoder_weight.grad is not None
            assert (model.encoder_weight.grad is None) == pure_text
            assert backward_steps == before + ([] if pure_text else [step])
            finalized.append(step)

        monkeypatch.setattr(native_step, 'finalize_model_grads', native_finalize)
        coordinator.finalize([model], local_tokens, pg_collection=pg)
        assert finalized == list(range(step + 1))
        assert not coordinator.active and coordinator.rounds == []
        assert coordinator.source is coordinator.pending is coordinator.source_gradient is None
        with pytest.raises(RuntimeError, match='no active training step'):
            coordinator.finalize([model], local_tokens, pg_collection=pg)

        encoder = torch.tensor(2.0, device='cuda', requires_grad=True)
        decoder = torch.tensor(0.25, device='cuda', requires_grad=True)
        reference = torch.zeros((), device='cuda')
        for sample in source['samples'].values():
            values = sample['tokens'].to(device='cuda', dtype=torch.float32)
            if sample['media_ids']:
                image_id = sample['media_ids'][0]
                values[values == 511] = (
                    torch.arange(4, device='cuda') + 10 * image_id + 1
                ) * encoder
            reference += ((values * decoder).square() * sample['loss_mask'].cuda()).sum()
        reference.backward()
        dist.all_reduce(model.decoder_weight.grad, group=pg.dp_cp)
        torch.testing.assert_close(model.decoder_weight.grad, decoder.grad, rtol=1e-6, atol=1e-6)
        if not pure_text:
            dist.all_reduce(model.encoder_weight.grad, group=pg.dp_cp)
            torch.testing.assert_close(
                model.encoder_weight.grad, encoder.grad, rtol=1e-6, atol=1e-6
            )


_WIDTH = 512
_LR = 0.125


def _normalization_vision_inputs(media, device):
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
                overlap_grad_reduce=False, overlap_param_gather=False, rerun_mode='disabled'
            ),
        )
        monkeypatch.setattr(
            'examples.mimo.data.qwen35_native.build_vision_inputs', _normalization_vision_inputs
        )
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
        loss_sum = torch.zeros((), device="cuda", dtype=torch.float64)
        for _ in range(rounds):
            item = next(iterator)
            runtime_sizes.append(item['cp_group'].size())
            output, loss_fn = coordinator.forward(item, ddp)
            raw_loss, tokens, _ = loss_fn(output)
            raw_loss.backward()
            local_tokens += tokens
            loss_sum += raw_loss.detach().double()
        original_local_count = int(local_tokens.item())
        coordinator.finalize([ddp], local_tokens, pg_collection=pg)
        dist.all_reduce(loss_sum, group=pg.dp_cp)
        torch.testing.assert_close(
            loss_sum.cpu() / reference['tokens'],
            torch.tensor(reference['loss'], dtype=torch.float64),
            rtol=2e-6,
            atol=2e-8,
        )
        assert int(local_tokens) == reference['tokens']
        assert [buffer.gradient_scaling_factor for buffer in ddp.buffers] == [1.0]
        assert not coordinator.active
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
            torch.testing.assert_close(
                parameter.main_grad.view(-1)[start:end].cpu().double(),
                reference['grads'][name][start:end],
                rtol=2e-6,
                atol=0,
            )
        update_successful, _, _ = optimizer.step()
        assert update_successful
        for name, parameter in module.named_parameters():
            gradient = reference['grads'][name]
            if distributed:
                expected_delta = -_LR * gradient / (gradient.abs() + opt_config.adam_eps)
                if name in owned_ranges:
                    start, end = owned_ranges[name]
                    states = inner._get_main_param_and_optimizer_states(parameter)
                    torch.testing.assert_close(
                        states['exp_avg'].cpu().double(),
                        (1 - opt_config.adam_beta1) * gradient[start:end],
                        rtol=2e-6,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        states['exp_avg_sq'].cpu().double(),
                        (1 - opt_config.adam_beta2) * gradient[start:end].square(),
                        rtol=2e-6,
                        atol=0,
                    )
            else:
                expected_delta = -_LR * gradient
            torch.testing.assert_close(
                parameter.detach().cpu().double() - reference['initial'][name],
                expected_delta,
                rtol=2e-6,
                atol=4e-8,
            )
        peers = [None] * pg.dp_cp.size()
        dist.all_gather_object(
            peers, (runtime_sizes, original_local_count, owned_ranges), group=pg.dp_cp
        )
        assert sum(count for _, count, _ in peers) == reference['tokens']
        assert {size for sizes, _, _ in peers for size in sizes} == {
            'cp1': {1},
            'static_cp2': {2},
            'dynamic_cp': {1, 2, 4},
        }[layout]
        if mask_kind == 'sparse':
            assert any(count == 0 for _, count, _ in peers)
        if distributed:
            for name in reference['grads']:
                intervals = sorted(ranges[name] for _, _, ranges in peers if name in ranges)
                assert intervals[0][0] == 0 and intervals[-1][1] == _WIDTH
                assert all(0 <= start < end <= _WIDTH for start, end in intervals)
                assert all(left[1] == right[0] for left, right in zip(intervals, intervals[1:]))
    finally:
        Utils.destroy_model_parallel()
