# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Fixed routing through real native rounds, feature transport and recomputation.

The tiny decoder keeps real TopKRouters (including MTP), the native auxiliary
statistics lifecycle and actual THD CP slicing. Native DDP/optimizer behavior is
covered by the full-model experiments rather than this adapter regression.
"""

from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from examples.mimo import native_step
from examples.mimo.native_step import NativeMimoStep
from megatron.core.datasets.data_schedule import wrap_data_iterator
from megatron.core.models.mimo.partition.utils import PartitionAdapter, PartitionConfig
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.models.mimo.test_native_mimo_step import (
    _TinyMimo,
    _vision_inputs,
    native_groups,
)


def _source(step):
    samples, media = {}, []
    for sid in range(8):
        tokens = torch.full((64,), sid + 1, dtype=torch.long)
        tokens[61:] = 0
        image_ids = []
        if sid != 7:
            tokens[1:5] = 511
            image_ids = [sid]
            media.append(
                dict(
                    image_id=sid,
                    source_id=sid,
                    size=(16, 16),
                    length=4,
                    grid=torch.tensor([1, 4, 4]),
                    pixel_values=torch.full((16, 3), float(sid + step)),
                )
            )
        loss_mask = torch.zeros(64)
        loss_mask[:61] = 1
        loss_mask[1:5] = 0
        samples[sid] = dict(
            tokens=tokens,
            labels=tokens.clone(),
            loss_mask=loss_mask,
            position_ids=torch.arange(64).expand(3, -1).clone(),
            original_seq_len=61,
            padded_seq_len=64,
            media_ids=image_ids,
        )
    return dict(step=step, samples=samples, media=media)


class _RoutedDecoder(torch.nn.Module):
    def __init__(self, groups):
        super().__init__()
        config = TransformerConfig(
            num_layers=1,
            hidden_size=4,
            num_attention_heads=1,
            num_moe_experts=4,
            moe_router_topk=2,
            moe_router_load_balancing_type='global_aux_loss',
            moe_aux_loss_coeff=0.001,
            moe_router_dtype='fp32',
            calculate_per_token_loss=True,
            mtp_num_layers=1,
        )
        router_groups = SimpleNamespace(
            tp=groups.tp, cp=groups.cp, tp_cp=groups.cp, tp_dp_cp=groups.dp_cp
        )
        self.router = TopKRouter(config, pg_collection=router_groups).cuda()
        self.mtp_router = TopKRouter(config, pg_collection=router_groups, is_mtp_layer=True).cuda()
        self.post_process = True
        self.calls = []

    def forward(self, hidden, padding, packed):
        scope = self.router._fixed_routing
        expected_binding = (scope.step, scope.current['round'])
        phase = 'training' if self.post_process else 'statistics'

        def run(value):
            assert scope is self.router._fixed_routing
            assert (scope.step, scope.current['round']) == expected_binding
            assert self.mtp_router._fixed_routing is scope
            outputs = []
            for index, router in enumerate((self.router, self.mtp_router)):
                probs, routing = router(value + index * 0.25, padding, packed)
                expected_map = torch.zeros_like(routing).scatter(1, scope.cache[router], True)
                assert torch.equal(routing, expected_map)
                coefficients = torch.arange(1, 5, device=value.device, dtype=value.dtype)
                outputs.append((probs * coefficients).sum(-1))
            self.calls.append((scope.step, scope.current['round'], phase, torch.is_grad_enabled()))
            return (outputs[0] + 0.25 * outputs[1]).square()

        return checkpoint(run, hidden, use_reentrant=True)


class _RoutedMimo(_TinyMimo):
    def __init__(self, groups):
        super().__init__(dynamic=False, mtp_layers=1)
        self.config.max_seqlen_per_dp_cp_rank = 96 if groups.cp.size() == 1 else 64
        self.config.moe_aux_loss_coeff = 0.001
        self.language_model = _RoutedDecoder(groups)
        self.partition_adapter = PartitionAdapter(
            PartitionConfig(
                seq_parallel=False,
                use_cp=groups.cp.size() > 1,
                tp_comm_overlap=False,
                max_seq_len=128,
                kv_format='thd',
                cp_group=groups.cp,
                tp_group=groups.tp,
            )
        )

    def forward(
        self, input_ids, loss_mask, packing_kwargs, modality_embeddings, padding_mask, **kwargs
    ):
        values = input_ids.float().clone()
        if modality_embeddings:
            values[input_ids == 511] = modality_embeddings['images'][:, 0]
        hidden = values.T.unsqueeze(-1).expand(-1, 1, 4) * self.decoder_weight / 100
        packed = PackedSeqParams(qkv_format='thd', **packing_kwargs)
        hidden, _, mask, packed = self.partition_adapter.shard(hidden, None, loss_mask, packed)
        _, local_padding, _, _ = self.partition_adapter.shard(None, padding_mask, None, packed)
        output = self.language_model(hidden, local_padding.T, packed)
        return output, mask


def test_native_fixed_ids_survive_rounds_cp_repacking_recompute_and_next_step(
    native_groups, monkeypatch, tmp_path
):
    pg, _ = native_groups
    directory = [str(tmp_path / 'routing') if pg.dp_cp.rank() == 0 else None]
    dist.broadcast_object_list(
        directory, src=dist.get_process_group_ranks(pg.dp_cp)[0], group=pg.dp_cp
    )
    monkeypatch.setattr('examples.mimo.data.qwen35_native.build_vision_inputs', _vision_inputs)
    monkeypatch.setattr(
        MoEAuxLossAutoScaler, 'main_loss_backward_scale', torch.tensor(1.0, device='cuda')
    )
    baseline = {}

    for mode in ('record', 'replay'):
        groups = SimpleNamespace(**vars(pg))
        # The tiny decoder has no sharded experts; native diagnostic plumbing
        # still receives an explicit singleton EP group before the stub runs.
        groups.ep = pg.tp
        if mode == 'record':
            groups.cp, groups.dp = pg.tp, pg.dp_cp
        torch.manual_seed(619)
        model = _RoutedMimo(groups)
        args = SimpleNamespace(
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            mimo_metrics_dir=None,
            rerun_mode='disabled',
            mimo_gradient_diagnostics=True,
            mimo_audit_token_assignments=True,
            **{f'mimo_fixed_routing_{mode}': directory[0]},
        )
        coordinator = NativeMimoStep(model, groups, args)
        # Full gradient diagnostics expect a complete production model. This test
        # enables the same flag for the independent auxiliary replay audit only.
        monkeypatch.setattr(
            'examples.mimo.gradient_diagnostics.collect_gradient_diagnostics',
            lambda *args, **kwargs: {},
        )
        finalized = []

        def finalize_native(chunks, num_tokens, **kwargs):
            assert chunks == [model]
            assert model.encoder_weight.grad is not None
            assert model.language_model.router.weight.grad is not None
            assert model.language_model.mtp_router.weight.grad is not None
            assert hasattr(model.language_model.router, '_fixed_routing')
            finalized.append(coordinator.step)

        monkeypatch.setattr(native_step, 'finalize_model_grads', finalize_native)
        for step in (7, 8):
            model.zero_grad(set_to_none=True)
            iterator, count, _, _ = wrap_data_iterator(
                iter([_source(step)]), model.config, 1, groups
            )
            assert count == 2
            scope = coordinator.fixed_routing
            assert scope.mode == 'replay' and scope.metrics['mtp_routers'] == 1
            assert scope.metrics['source_tokens'] == 8 * 61
            local_tokens = torch.zeros((), device='cuda', dtype=torch.int)
            for _ in range(count):
                output, loss = coordinator.forward(next(iterator), model)
                raw, tokens, _ = loss(output)
                raw.backward()
                local_tokens += tokens
                assert model.encoder_weight.grad is None
            coordinator.finalize([model], local_tokens, pg_collection=groups)
            assert finalized[-1] == step
            assert scope.closed and coordinator.fixed_routing is None and not coordinator.active
            audit = coordinator.metrics['aux_replay']
            assert audit['missing'] == audit['unexpected'] == audit['global_histogram_l1'] == 0
            for kind in ('auxiliary', 'dispatch'):
                for phase in ('training', 'recompute'):
                    assignments = audit['token_assignments'][f'{kind}_{phase}']
                    assert assignments['pairs'] > 0
                    assert assignments['different_rows'] == assignments['shape_mismatches'] == 0
            for router in (model.language_model.router, model.language_model.mtp_router):
                assert not hasattr(router, '_fixed_routing')
                assert not hasattr(router, '_global_aux_loss_step')
            calls = [item for item in model.language_model.calls if item[0] == step]
            for round_id in range(count):
                assert [(phase, grad) for _, rnd, phase, grad in calls if rnd == round_id] == [
                    ('statistics', False),
                    ('training', False),
                    ('training', True),
                ]
            actual = {}
            for name, parameter in model.named_parameters():
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                gradient = parameter.grad.detach().clone()
                dist.all_reduce(gradient, group=groups.dp_cp)
                actual[name] = gradient / coordinator.global_tokens
            if mode == 'record':
                baseline[step] = actual
            else:
                for name in actual:
                    torch.testing.assert_close(
                        actual[name], baseline[step][name], atol=2e-7, rtol=2e-5
                    )
