# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Native MIMO state reuse across mixed, text-only, and changed-media steps.

Real packing, bridge communication, decoder backward, and encoder backward run
through one coordinator. The tiny decoder uses a strided disjoint partition;
row-level THD/MIMO splicing and native DDP normalization have separate oracles.
The finalizer is an observation point here, not a replacement training stack.
"""

import json
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from examples.mimo import native_step
from megatron.core.datasets.data_schedule import wrap_data_iterator
from tests.unit_tests.models.mimo.test_native_mimo_step import (
    _coordinator,
    _source_batch,
    native_groups,
)


def _sources():
    mixed = _source_batch()
    text = _source_batch(pure_text=True)
    text['step'] = mixed['step'] + 1
    changed = _source_batch(pure_text=True)
    changed['step'] = mixed['step'] + 2
    old_samples = changed['samples']
    changed['samples'], changed['media'] = {}, []
    lengths = (1, 7, 3, 5, 2, 6, 4, 8)
    for sid, previous_sid in enumerate(reversed(range(len(old_samples)))):
        sample = old_samples[previous_sid]
        if previous_sid < len(lengths):
            image_id = 100 + 3 * previous_sid
            length = lengths[previous_sid]
            sample['tokens'][1 : 1 + length] = 511
            sample['labels'] = sample['tokens'].clone()
            sample['media_ids'] = [image_id]
            changed['media'].append(
                dict(image_id=image_id, size=(16 + 8 * previous_sid, 16), length=length)
            )
        changed['samples'][sid] = sample
    # Source media order is intentionally different from both source-sample
    # order and producer-local image-ID order.
    changed['media'].reverse()
    return mixed, text, changed


def _reference(source):
    """CPU FP64 original-sample objective, with an independent weight per image."""
    media = {item['image_id']: item for item in source['media']}
    image_weights = {
        image_id: torch.tensor(2.0, dtype=torch.float64, requires_grad=True) for image_id in media
    }
    decoder = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
    loss = torch.zeros((), dtype=torch.float64)
    count = 0
    for sample in source['samples'].values():
        length = int(sample['original_seq_len'])
        values = sample['tokens'][:length].double().clone()
        if sample['media_ids']:
            rows = []
            for image_id in sample['media_ids']:
                rows.append(
                    (
                        torch.arange(media[image_id]['length'], dtype=torch.float64)
                        + 8 * image_id
                        + 1
                    )
                    / 32
                    * image_weights[image_id]
                )
            values[sample['tokens'][:length] == 511] = torch.cat(rows)
        mask = sample['loss_mask'][:length].double()
        loss = loss + ((values * decoder).square() * mask).sum()
        count += int(mask.sum())
    loss.backward()
    return {
        'raw_loss': loss.item(),
        'tokens': count,
        'decoder_gradient': decoder.grad.item(),
        'image_gradients': {
            image_id: weight.grad.item() for image_id, weight in image_weights.items()
        },
    }


def _close(actual, expected):
    return abs(actual - expected) <= 2e-6 * abs(expected)


@pytest.mark.parametrize('dynamic', (False, True), ids=('static_cp2', 'dynamic_cp'))
def test_native_routing_state_reused_across_source_steps(native_groups, monkeypatch, dynamic):
    model, coordinator, pg = _coordinator(native_groups, monkeypatch, dynamic=dynamic)
    original_coordinator = id(coordinator)
    active = {}
    encoder_hooks, source_hooks, finalize_calls = [], [], []
    reports = []

    def vision_inputs(media, device):
        active['producer_images'] = [item['image_id'] for item in media]
        active['producer_rows'] = sum(item['length'] for item in media)
        if not media:
            return {}
        rows = torch.cat(
            [torch.arange(item['length']) + 8 * item['image_id'] + 1 for item in media]
        )
        return {'rows': rows.to(device=device, dtype=torch.float32)[:, None].expand(-1, 4) / 32}

    def encoder_hook(gradient):
        encoder_hooks.append(active['step'])

    def observe_finalizer(model_chunks, num_tokens, **kwargs):
        # No gradient reductions here: the producer-local encoder gradient is
        # checked against independent per-image derivatives after finalization.
        assert model_chunks == [model]
        assert num_tokens is active['local_tokens']
        assert kwargs['pg_collection'] is pg
        expected_calls = int(bool(active['producer_images']))
        assert encoder_hooks.count(active['step']) == expected_calls
        assert source_hooks.count(active['step']) == expected_calls
        assert (model.encoder_weight.grad is not None) == bool(expected_calls)
        assert coordinator.pending is None
        finalize_calls.append(active['step'])

    monkeypatch.setattr('examples.mimo.data.qwen35_native.build_vision_inputs', vision_inputs)
    monkeypatch.setattr(native_step, 'finalize_model_grads', observe_finalizer)
    handle = model.encoder_weight.register_hook(encoder_hook)
    try:
        for source_index, source in enumerate(_sources()):
            step = source['step']
            model.zero_grad(set_to_none=True)
            active.clear()
            active['step'] = step
            expected = _reference(source)
            hooks_before = list(encoder_hooks)
            source_hooks_before = list(source_hooks)
            iterator, rounds, _, _ = wrap_data_iterator(iter([source]), model.config, 1, pg)
            if coordinator.source.requires_grad:
                coordinator.source.register_hook(
                    lambda gradient, saved_step=step: source_hooks.append(saved_step)
                )
            local_tokens = torch.zeros((), device='cuda', dtype=torch.int)
            active['local_tokens'] = local_tokens
            local_loss = 0.0
            runtime_cp_sizes = []
            for _ in range(rounds):
                item = next(iterator)
                runtime_cp_sizes.append(item['cp_group'].size())
                output, loss_fn = coordinator.forward(item, model)
                raw_loss, tokens, _ = loss_fn(output)
                raw_loss.backward()
                local_tokens += tokens
                local_loss += raw_loss.detach().double().item()
                # The bridge boundary must keep encoder backward deferred even
                # when this rank is both producer and decoder consumer.
                assert encoder_hooks == hooks_before
                assert source_hooks == source_hooks_before
                assert model.encoder_weight.grad is None
            coordinator.finalize([model], local_tokens, pg_collection=pg)
            encoder_gradient = (
                model.encoder_weight.grad.item() if model.encoder_weight.grad is not None else 0.0
            )
            expected_encoder = sum(
                expected['image_gradients'][image_id] for image_id in active['producer_images']
            )
            cleared = (
                not coordinator.active
                and coordinator.pending is None
                and coordinator.source is None
                and coordinator.source_gradient is None
                and coordinator.rounds == []
            )
            expected_new_hooks = [step] if active['producer_images'] else []
            report = {
                'step': step,
                'source_kind': ('mixed', 'pure_text', 'changed_media')[source_index],
                'producer_images': active['producer_images'],
                'producer_rows': active['producer_rows'],
                'source_image_ids': [item['image_id'] for item in source['media']],
                'source_visual_lengths': [item['length'] for item in source['media']],
                'sample_logical_lengths': [
                    item['original_seq_len'] for item in source['samples'].values()
                ],
                'runtime_cp_sizes': runtime_cp_sizes,
                'local_raw_loss': local_loss,
                'local_supervised_tokens': int(local_tokens),
                'local_decoder_gradient': model.decoder_weight.grad.item(),
                'local_encoder_gradient': encoder_gradient,
                'expected_local_encoder_gradient': expected_encoder,
                'encoder_gradient_passed': _close(encoder_gradient, expected_encoder),
                'encoder_backward_calls': encoder_hooks.count(step),
                'source_backward_calls': source_hooks.count(step),
                'native_finalize_calls': finalize_calls.count(step),
                'old_hooks_unchanged': encoder_hooks == hooks_before + expected_new_hooks
                and source_hooks == source_hooks_before + expected_new_hooks,
                'state_cleared': cleared,
                'same_coordinator': id(coordinator) == original_coordinator,
                'native_mean_loss': coordinator.metrics['loss'],
                'reference': expected,
            }
            peer_reports = [None] * pg.dp_cp.size()
            dist.all_gather_object(peer_reports, report, group=pg.dp_cp)
            image_ids = [image_id for item in peer_reports for image_id in item['producer_images']]
            observed_sizes = {size for item in peer_reports for size in item['runtime_cp_sizes']}
            global_loss = sum(item['local_raw_loss'] for item in peer_reports)
            global_decoder_gradient = sum(item['local_decoder_gradient'] for item in peer_reports)
            passed = (
                sorted(image_ids) == sorted(expected['image_gradients'])
                and _close(global_loss, expected['raw_loss'])
                and _close(global_decoder_gradient, expected['decoder_gradient'])
                and sum(item['local_supervised_tokens'] for item in peer_reports)
                == expected['tokens']
                and observed_sizes == ({1, 2, 4} if dynamic else {2})
                and all(
                    item['encoder_gradient_passed']
                    and item['state_cleared']
                    and item['same_coordinator']
                    and item['old_hooks_unchanged']
                    and item['native_finalize_calls'] == 1
                    and item['encoder_backward_calls'] == int(bool(item['producer_images']))
                    and item['source_backward_calls'] == int(bool(item['producer_images']))
                    and _close(item['native_mean_loss'], expected['raw_loss'] / expected['tokens'])
                    for item in peer_reports
                )
            )
            report.update(
                global_raw_loss=global_loss,
                global_decoder_gradient=global_decoder_gradient,
                observed_global_cp_sizes=sorted(observed_sizes),
                all_ranks_passed=passed,
            )
            reports.append(report)
            directory = os.environ.get('MIMO_ROUTING_ORACLE_REPORT_DIR')
            if directory:
                output = Path(directory)
                output.mkdir(parents=True, exist_ok=True)
                filename = f'native_lifecycle_{"dynamic" if dynamic else "static"}_rank{dist.get_rank()}.json'
                (output / filename).write_text(
                    json.dumps(
                        {
                            'rank': dist.get_rank(),
                            'dynamic': dynamic,
                            'steps': reports,
                            'scope': 'NativeMimoStep lifecycle with real bridge; strided tiny decoder; observed finalizer, no DDP/optimizer',
                        },
                        indent=2,
                    )
                    + '\n'
                )
            assert passed, json.dumps(report, indent=2)
        assert encoder_hooks == [reports[0]['step'], reports[2]['step']]
        assert source_hooks == encoder_hooks
        assert finalize_calls == [item['step'] for item in reports]
    finally:
        handle.remove()
