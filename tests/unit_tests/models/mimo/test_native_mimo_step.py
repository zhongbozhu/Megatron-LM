# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Native adapter contracts with real packing and distributed feature transport.

The tiny differentiable model isolates scheduling, metadata and encoder-backward
lifetime from transformer kernels. Full model/DDP/optimizer tests remain e2e.
"""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from examples.mimo import native_step
from examples.mimo.native_step import NativeMimoStep, NativeSourceIterator
from megatron.core.datasets.data_schedule import wrap_data_iterator
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig


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
    _materialize_mtp_input_mask = staticmethod(MimoModel._materialize_mtp_input_mask)

    def __init__(self, *, dynamic, mtp_layers=0):
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
            hidden_dropout=0,
            attention_dropout=0,
            mtp_num_layers=mtp_layers,
            moe_aux_loss_coeff=0,
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


def _coordinator(groups, monkeypatch, *, dynamic, mtp_layers=0):
    pg, runtime_groups = groups
    monkeypatch.setattr('examples.mimo.data.qwen35_native.build_vision_inputs', _vision_inputs)
    monkeypatch.setattr(
        native_step.parallel_state,
        'get_dynamic_data_context_parallel_groups',
        lambda group_size: runtime_groups[group_size],
    )
    model = _TinyMimo(dynamic=dynamic, mtp_layers=mtp_layers)
    args = SimpleNamespace(
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        mimo_metrics_dir=None,
        rerun_mode='disabled',
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
def test_native_packing_preserves_mrope_masks_and_global_mtp_counts(
    native_groups, monkeypatch, dynamic
):
    model, coordinator, pg = _coordinator(native_groups, monkeypatch, dynamic=dynamic, mtp_layers=2)
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
    assert coordinator.step == 7

    expected_counts = [0, 0, 0]
    for sample in source['samples'].values():
        length = sample['original_seq_len']
        expected_counts[0] += int(sample['loss_mask'].sum())
        for depth in (1, 2):
            for start in range(length - depth):
                if all(sample['tokens'][start + offset] != 511 for offset in range(1, depth + 1)):
                    expected_counts[depth] += int(sample['loss_mask'][start + depth])
    assert expected_counts[0] > expected_counts[1] > expected_counts[2]
    observed_tail = False
    for _ in range(count):
        item = next(iterator)
        kwargs = item['kwargs']
        packing = kwargs['packing_kwargs']
        actual_counts = packing['mtp_loss_token_counts'].cpu()
        torch.testing.assert_close(
            actual_counts, torch.tensor(expected_counts, dtype=actual_counts.dtype)
        )
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
@pytest.mark.parametrize('pure_text', (False, True))
def test_native_boundary_finishes_encoder_backward_before_ddp_finalize(
    native_groups, monkeypatch, dynamic, pure_text
):
    model, coordinator, pg = _coordinator(native_groups, monkeypatch, dynamic=dynamic)
    source = _source_batch(pure_text=pure_text)
    iterator, count, _, _ = wrap_data_iterator(iter([source]), model.config, 1, pg)
    local_tokens = torch.zeros((), device='cuda', dtype=torch.int)
    for _ in range(count):
        output, loss = coordinator.forward(next(iterator), model)
        raw, tokens, _ = loss(output)
        raw.backward()
        local_tokens += tokens
        # Even local image producers retain one graph until every consumer has
        # returned its gradient; decoder autograd must not bypass the boundary.
        assert model.encoder_weight.grad is None
    finalized = []

    def native_finalize(model_chunks, num_tokens, **kwargs):
        assert model_chunks == [model]
        assert num_tokens is local_tokens
        assert model.decoder_weight.grad is not None
        assert (model.encoder_weight.grad is None) == pure_text
        finalized.append(True)

    monkeypatch.setattr(native_step, 'finalize_model_grads', native_finalize)
    coordinator.finalize([model], local_tokens, pg_collection=pg)
    assert finalized == [True]
    assert not coordinator.active
    assert coordinator.source is None
    assert coordinator.pending is None
    with pytest.raises(RuntimeError, match='no active training step'):
        coordinator.finalize([model], local_tokens, pg_collection=pg)

    encoder = torch.tensor(2.0, device='cuda', requires_grad=True)
    decoder = torch.tensor(0.25, device='cuda', requires_grad=True)
    reference = torch.zeros((), device='cuda')
    for sid, sample in source['samples'].items():
        values = sample['tokens'].to(device='cuda', dtype=torch.float32)
        if sample['media_ids']:
            values[values == 511] = (torch.arange(4, device='cuda') + 10 * sid + 1) * encoder
        reference += ((values * decoder).square() * sample['loss_mask'].cuda()).sum()
    reference.backward()
    dist.all_reduce(model.decoder_weight.grad, group=pg.dp_cp)
    torch.testing.assert_close(model.decoder_weight.grad, decoder.grad, rtol=1e-6, atol=1e-6)
    if not pure_text:
        dist.all_reduce(model.encoder_weight.grad, group=pg.dp_cp)
        torch.testing.assert_close(model.encoder_weight.grad, encoder.grad, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('fail_mtp', (False, True))
def test_auxiliary_statistics_skip_heads_keep_mtp_routers_and_restore_state(
    native_groups, monkeypatch, fail_mtp
):
    model, coordinator, pg = _coordinator(native_groups, monkeypatch, dynamic=False, mtp_layers=1)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_router_load_balancing_type='global_aux_loss',
        moe_aux_loss_coeff=0.001,
        moe_router_dtype='fp32',
        calculate_per_token_loss=True,
        mtp_num_layers=1,
    )
    router_groups = SimpleNamespace(tp=pg.tp, cp=pg.tp, tp_cp=pg.tp, tp_dp_cp=pg.dp_cp)

    class MTP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.router = TopKRouter(config, pg_collection=router_groups, is_mtp_layer=True).cuda()

        def forward(self, hidden_states, **kwargs):
            self.router(hidden_states)
            # Exercise restoration after a failure inside the actual GPT MTP
            # branch, before its post_process guard is reached.
            if fail_mtp:
                raise RuntimeError('test MTP failure')
            return torch.cat((hidden_states, hidden_states), dim=0)

    class LanguageModel(torch.nn.Module):
        _postprocess = GPTModel._postprocess

        def __init__(self):
            super().__init__()
            self.router = TopKRouter(config, pg_collection=router_groups).cuda()
            self.mtp = MTP()
            self.post_process = True
            self.share_embeddings_and_output_weights = False
            self.embedding = None
            self.config = config

        def forward(self, input_ids):
            hidden = torch.ones((input_ids.numel(), 1, 8), device='cuda')
            self.router(hidden)
            # Production GPT control flow determines whether the MTP block and
            # output heads run. No fake output-head branch is used in this test.
            return self._postprocess(
                hidden_states=hidden,
                input_ids=input_ids,
                position_ids=None,
                labels=input_ids,
                rotary_pos_emb=None,
                rotary_pos_cos=None,
                rotary_pos_sin=None,
                mtp_in_postprocess=True,
            )

    model.language_model = LanguageModel()
    model.config.moe_aux_loss_coeff = 0.001

    def forward(input_ids, packing_kwargs, **kwargs):
        group = packing_kwargs['cp_group']
        return model.language_model(input_ids[:, group.rank() :: group.size()])

    monkeypatch.setattr(model, 'forward', forward)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    if fail_mtp:
        with pytest.raises(RuntimeError, match='test MTP failure'):
            wrap_data_iterator(iter([_source_batch()]), model.config, 1, pg)
        assert coordinator.aux is None
        assert not hasattr(model.language_model.router, '_global_aux_loss_step')
        assert not hasattr(model.language_model.mtp.router, '_global_aux_loss_step')
    else:
        wrap_data_iterator(iter([_source_batch()]), model.config, 1, pg)
        layers = coordinator.metrics['aux']['layers']
        assert set(layers) == {'language_model.router', 'language_model.mtp.router'}
        counts = [item['routed_tokens'] for item in layers.values()]
        expected = sum(
            item['kwargs']['input_ids'].numel() // item['cp_group'].size()
            for item in coordinator.rounds
        )
        expected = torch.tensor(expected, device='cuda')
        dist.all_reduce(expected, group=pg.dp_cp)
        assert counts == [int(expected), int(expected)]
    assert model.language_model.post_process is True
    torch.testing.assert_close(torch.get_rng_state(), cpu_rng, rtol=0, atol=0)
    torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng, rtol=0, atol=0)
    coordinator._clear()


def test_statistics_grad_mode_releases_selective_graphs_without_backward(monkeypatch):
    """Dropping a stats result must also release module-owned selective checkpoints."""
    import weakref

    from megatron.core.tensor_parallel import random as tp_random

    class LanguageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))
            self.post_process = True
            self.norm_out_checkpoint = None
            self.saved = []

        def forward(self, features):
            assert torch.is_grad_enabled() and not self.post_process
            assert all(reference() is None for reference in self.saved)
            hidden = features * self.weight
            self.saved.append(weakref.ref(hidden))
            self.norm_out_checkpoint = tp_random.CheckpointWithoutOutput(fp8=None)
            normalized = self.norm_out_checkpoint.checkpoint(torch.sin, hidden)
            output = normalized.square()
            self.norm_out_checkpoint.discard_output_and_register_recompute(output)
            # A real training-mode graph exists, but stats must never backward it.
            assert output.grad_fn is not None
            return output

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = LanguageModel()
            self.vision = torch.nn.Parameter(torch.tensor(3.0))
            self.vision_checkpoint = tp_random.CheckpointWithoutOutput(fp8=None)
            self.vision_checkpoint.ctx = object()

        def forward(self, modality_embeddings):
            return self.language_model(modality_embeddings['images'])

    class Statistics:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def set_round(self, round_id):
            pass

        def finalize(self):
            return {'loss': 0.0}

        def begin_training(self, tokens):
            pass

    monkeypatch.setattr(native_step, 'GlobalAuxLossStep', Statistics)
    monkeypatch.setattr(torch.cuda, 'get_rng_state', lambda: None)
    monkeypatch.setattr(torch.cuda, 'set_rng_state', lambda state: None)
    tracker = SimpleNamespace(get_states=lambda: {}, set_states=lambda state: None)
    monkeypatch.setattr(native_step.tensor_parallel, 'get_cuda_rng_tracker', lambda: tracker)
    monkeypatch.setattr(tp_random, '_get_all_rng_states', lambda: ())
    model = Model()
    coordinator = object.__new__(NativeMimoStep)
    coordinator.model = model
    coordinator.pg = SimpleNamespace(dp_cp=None)
    coordinator.args = SimpleNamespace(mimo_gradient_diagnostics=False)
    coordinator.source = model.vision.square().reshape(1, 1)
    coordinator.global_tokens = torch.tensor(3)
    coordinator.metrics = {}
    coordinator.rounds = [dict(round_id=i, plans=(), cp_group=None, kwargs={}) for i in range(3)]
    coordinator.bridge = SimpleNamespace(
        forward=lambda source, *args: SimpleNamespace(features=source.detach().requires_grad_())
    )
    rng = torch.get_rng_state().clone()
    with torch.no_grad():
        coordinator._collect_auxiliary_statistics()
    assert model.language_model.post_process
    assert all(reference() is None for reference in model.language_model.saved)
    assert model.language_model.norm_out_checkpoint.ctx is None
    assert model.vision_checkpoint.ctx is not None
    assert all(parameter.grad is None for parameter in model.parameters())
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
