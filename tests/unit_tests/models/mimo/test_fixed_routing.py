# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CPU tests of canonical expert-ID storage, replay and failure boundaries."""

import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import examples.mimo.fixed_routing as fixed_routing
from examples.mimo.fixed_routing import FixedRouting, source_fingerprint


class _Router(torch.nn.Module):
    def __init__(self, mtp=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(4, 2))
        self.num_experts = 4
        self.topk = 2
        self.is_mtp_layer = mtp
        self.tp_group = SimpleNamespace(size=lambda: 1)
        self.config = SimpleNamespace(mtp_num_layers=1, mtp_use_repeated_layer=False)


class _Partition:
    def __init__(self):
        self.indices = None
        self.calls = 0

    def shard(self, embeddings, index, mask, packed):
        self.calls += 1
        if self.indices is not None:
            index = index[:, self.indices]
        return embeddings, index, mask, packed


@pytest.fixture(autouse=True)
def fake_router_type(monkeypatch):
    monkeypatch.setattr(fixed_routing, 'TopKRouter', _Router)


def _model():
    model = torch.nn.Module()
    model.decoder = _Router()
    model.mtp = _Router(mtp=True)
    model.partition_adapter = _Partition()
    return model


def _samples():
    return {
        sid: dict(
            original_seq_len=length,
            padded_seq_len=4,
            tokens=torch.tensor([sid * 10 + i for i in range(length)] + [0] * (4 - length)),
            labels=torch.arange(4),
            loss_mask=torch.ones(4),
            position_ids=torch.arange(4).repeat(3, 1),
            media_ids=[sid],
        )
        for sid, length in ((0, 3), (1, 2))
    }


def _item(samples, order=(0, 1), round_id=0, cp=1, widths=None):
    widths = widths or [4] * len(order)
    tokens, logical, padded = [], [0], [0]
    for sid, width in zip(order, widths):
        length = samples[sid]['original_seq_len']
        tokens.append(torch.nn.functional.pad(samples[sid]['tokens'][:length], (0, width - length)))
        logical.append(logical[-1] + length)
        padded.append(padded[-1] + width)
    return dict(
        round_id=round_id,
        diagnostic_sample_ids=order,
        kwargs=dict(
            input_ids=torch.cat(tokens).unsqueeze(0),
            packing_kwargs=dict(
                cu_seqlens_q=torch.tensor(logical, dtype=torch.int32),
                cu_seqlens_q_padded=torch.tensor(padded, dtype=torch.int32),
                local_cp_size=cp,
            ),
        ),
    )


def _ids(tokens, offset=0):
    first = (tokens.flatten() + offset) % 4
    return torch.stack((first, (first + 2) % 4), dim=-1)


def _record(directory, samples=None, **kwargs):
    samples = _samples() if samples is None else samples
    model = _model()
    scope = FixedRouting(model, directory, 'record', 7, samples, **kwargs)
    item = _item(samples)
    scope.begin_round(item)
    results = {}
    for offset, (name, router) in enumerate(scope.routers):
        ids = _ids(item['kwargs']['input_ids'], offset)
        results[name] = scope.select(router, torch.zeros(8, 4), lambda: ids)
    scope.seal_recording()
    return model, scope, results


def _unexpected_selector():
    raise AssertionError('Replay must never compute new top-k IDs')


def test_cp1_record_replays_repacking_and_partitioned_tokens(tmp_path):
    samples = _samples()
    model, scope, baseline = _record(tmp_path, samples)
    assert scope.metrics['source_tokens'] == 5
    assert scope.metrics['mtp_routers'] == 1
    scope.close()
    replay = FixedRouting(model, tmp_path, 'replay', 7, samples)
    model.partition_adapter.indices = torch.tensor([0, 2, 4, 6])
    replay.begin_round(_item(samples, order=(1, 0), cp=2))
    for name, router in replay.routers:
        selected = replay.select(router, torch.randn(4, 4), _unexpected_selector)
        expected = torch.stack(
            (baseline[name][4], torch.tensor([0, 1]), baseline[name][0], baseline[name][2])
        )
        assert torch.equal(selected, expected)
        # Activation recomputation gets the same immutable choices even with different logits.
        again = replay.select(router, torch.full((4, 4), 100.0), _unexpected_selector)
        assert selected is again
    assert model.partition_adapter.calls == 2
    replay.close()
    assert not hasattr(model.decoder, '_fixed_routing')


def test_record_then_training_and_mtp_masked_tail(tmp_path):
    model, scope, baseline = _record(tmp_path)
    scope.begin_round(_item(_samples()))
    # MTP masks the last real token, but the canonical row identity is not shifted.
    padding = torch.tensor([False, False, True, True, False, True, True, True])
    selected = scope.select(model.mtp, torch.zeros(8, 4), _unexpected_selector, padding)
    assert torch.equal(selected, baseline['mtp'])
    assert torch.equal(selected[[3, 6, 7]], torch.tensor([[0, 1]]).expand(3, -1))
    scope.close()
    scope.close()


def test_replay_accepts_changed_padding_but_rejects_changed_source(tmp_path):
    model, scope, _ = _record(tmp_path)
    scope.close()
    samples = _samples()
    for sample in samples.values():
        sample['tokens'] = torch.nn.functional.pad(sample['tokens'], (0, 4), value=99)
        sample['padded_seq_len'] = 8
    replay = FixedRouting(model, tmp_path, 'replay', 7, samples)
    replay.begin_round(_item(samples, widths=(8, 8)))
    ids = replay.select(model.decoder, torch.zeros(16, 4), _unexpected_selector)
    assert ids.shape == (16, 2)
    replay.close()
    samples[0]['labels'][1] += 1
    with pytest.raises(RuntimeError, match='source, checkpoint context or router configuration'):
        FixedRouting(model, tmp_path, 'replay', 7, samples)


def test_media_fingerprint_checks_actual_pixels_and_grid():
    samples = _samples()
    image = dict(
        image_id=0,
        source_id='sample',
        size=(4, 4),
        length=2,
        grid=torch.tensor([1, 2, 2]),
        pixel_values=torch.zeros(4, 8),
    )
    original = source_fingerprint(samples, [image])
    image['pixel_values'][0, 0] = 1
    assert source_fingerprint(samples, [image]) != original
    image['pixel_values'][0, 0] = 0
    image['grid'][1] = 4
    assert source_fingerprint(samples, [image]) != original


def test_replay_rejects_changed_context_and_router_config(tmp_path):
    model, scope, _ = _record(tmp_path, context={'checkpoint': 'warm52'})
    scope.close()
    with pytest.raises(RuntimeError, match='configuration changed'):
        FixedRouting(model, tmp_path, 'replay', 7, _samples(), context={'checkpoint': 'cold'})
    model.decoder.config.moe_router_pre_softmax = True
    with pytest.raises(RuntimeError, match='configuration changed'):
        FixedRouting(model, tmp_path, 'replay', 7, _samples(), context={'checkpoint': 'warm52'})


def test_record_refuses_to_overwrite_evidence(tmp_path):
    model, scope, _ = _record(tmp_path)
    scope.close()
    with pytest.raises(RuntimeError, match='Refusing to overwrite'):
        FixedRouting(model, tmp_path, 'record', 7, _samples())


def test_incomplete_router_record_has_no_success_manifest(tmp_path):
    model = _model()
    scope = FixedRouting(model, tmp_path, 'record', 7, _samples())
    item = _item(_samples())
    scope.begin_round(item)
    scope.select(model.decoder, torch.zeros(8, 4), lambda: _ids(item['kwargs']['input_ids']))
    with pytest.raises(RuntimeError, match='Not every decoder/MTP router'):
        scope.seal_recording()
    assert not (tmp_path / 'step00000007' / 'manifest.json').exists()
    scope.close()


def test_missing_source_sample_has_no_success_manifest(tmp_path):
    model = _model()
    scope = FixedRouting(model, tmp_path, 'record', 7, _samples())
    item = _item(_samples(), order=(0,))
    scope.begin_round(item)
    for _, router in scope.routers:
        scope.select(router, torch.zeros(4, 4), lambda: _ids(item['kwargs']['input_ids']))
    with pytest.raises(RuntimeError, match='omitted source samples'):
        scope.seal_recording()
    assert not (tmp_path / 'step00000007' / 'manifest.json').exists()
    scope.close()


def test_record_requires_single_call_per_router_round(tmp_path):
    model = _model()
    scope = FixedRouting(model, tmp_path, 'record', 7, _samples())
    item = _item(_samples())
    scope.begin_round(item)
    ids = _ids(item['kwargs']['input_ids'])
    scope.select(model.decoder, torch.zeros(8, 4), lambda: ids)
    with pytest.raises(RuntimeError, match='one invocation'):
        scope.select(model.decoder, torch.zeros(8, 4), lambda: ids)
    scope.close()


@pytest.mark.parametrize('failure', ['shared', 'mtp_depth', 'tp'])
def test_unsupported_model_topologies_rejected_before_attachment(tmp_path, failure):
    model = _model()
    if failure == 'shared':
        model.alias = model.decoder
    elif failure == 'mtp_depth':
        model.mtp.config.mtp_use_repeated_layer = True
        model.mtp.config.mtp_num_layers = 2
    else:
        model.decoder.tp_group = SimpleNamespace(size=lambda: 2)
    with pytest.raises(RuntimeError):
        FixedRouting(model, tmp_path, 'record', 7, _samples())
    assert not hasattr(model.decoder, '_fixed_routing')


@pytest.mark.parametrize(
    'failure', ['cp1_partial', 'duplicate_rows', 'range', 'token_identity', 'length']
)
def test_invalid_round_mapping_rejected(tmp_path, failure):
    model = _model()
    scope = FixedRouting(model, tmp_path, 'record', 7, _samples())
    item = _item(_samples())
    if failure == 'cp1_partial':
        model.partition_adapter.indices = torch.tensor([0, 1, 4, 5])
    elif failure == 'duplicate_rows':
        model.partition_adapter.indices = torch.tensor([0, 0])
    elif failure == 'range':

        class _BadPartition:
            def shard(self, a, b, c, d):
                return a, torch.tensor([[-1, 0]]), c, d

        model.partition_adapter = _BadPartition()
    elif failure == 'token_identity':
        item['kwargs']['input_ids'][0, 1] = 999
    else:
        item['kwargs']['packing_kwargs']['cu_seqlens_q'][1] = 2
    with pytest.raises(ValueError):
        scope.begin_round(item)
    scope.close()


@pytest.mark.parametrize('failure', ['duplicate', 'range', 'dtype', 'shape'])
def test_invalid_expert_ids_rejected(tmp_path, failure):
    model = _model()
    scope = FixedRouting(model, tmp_path, 'record', 7, _samples())
    item = _item(_samples())
    scope.begin_round(item)
    ids = _ids(item['kwargs']['input_ids'])
    if failure == 'duplicate':
        ids[:, 1] = ids[:, 0]
    elif failure == 'range':
        ids[0, 0] = 4
    elif failure == 'dtype':
        ids = ids.float()
    else:
        ids = ids[:2]
    with pytest.raises(ValueError):
        scope.select(model.decoder, torch.zeros(8, 4), lambda: ids)
    scope.close()


def test_corrupt_table_cannot_silently_replay(tmp_path):
    model, scope, _ = _record(tmp_path)
    scope.close()
    path = tmp_path / 'step00000007' / 'rank00000.pt'
    table = torch.load(path, weights_only=True)
    table['decoder'][0][0] = torch.tensor([1, 3])
    torch.save(table, path)
    replay = FixedRouting(model, tmp_path, 'replay', 7, _samples())
    replay.begin_round(_item(_samples()))
    with pytest.raises(ValueError, match='checksum mismatch'):
        replay.select(model.decoder, torch.zeros(8, 4), _unexpected_selector)
    replay.close()


def test_manifest_route_hash_ignores_source_pack_order(tmp_path):
    hashes = []
    for order in ((0, 1), (1, 0)):
        model = _model()
        scope = FixedRouting(model, tmp_path / str(order), 'record', 7, _samples())
        item = _item(_samples(), order=order)
        scope.begin_round(item)
        for offset, (_, router) in enumerate(scope.routers):
            scope.select(
                router,
                torch.zeros(8, 4),
                lambda offset=offset: _ids(item['kwargs']['input_ids'], offset),
            )
        scope.seal_recording()
        hashes.append(scope.metrics['route_sha256'])
        scope.close()
    assert hashes[0] == hashes[1]


def _distributed_worker(rank, rendezvous, directory, failure):
    fixed_routing.TopKRouter = _Router
    torch.distributed.init_process_group(
        'gloo', init_method=f'file://{rendezvous}', rank=rank, world_size=2
    )
    try:
        model = _model()
        samples = _samples()
        scope = FixedRouting(
            model, directory, 'record', 7, samples, group=torch.distributed.group.WORLD
        )
        sid = 0 if failure == 'duplicate_owner' else rank
        item = _item(samples, order=(sid,))
        scope.begin_round(item)
        for offset, (_, router) in enumerate(scope.routers):
            if failure == 'missing_router' and rank == 1 and router is model.mtp:
                continue
            scope.select(
                router,
                torch.zeros(4, 4),
                lambda offset=offset: _ids(item['kwargs']['input_ids'], offset),
            )
        if failure:
            try:
                scope.seal_recording()
            except RuntimeError as error:
                assert 'Fixed routing phase failed' in str(error)
            else:
                raise AssertionError('Every rank must observe the failed recording phase')
            assert not (Path(directory) / 'step00000007' / 'manifest.json').exists()
            scope.close()
            return
        scope.seal_recording()
        scope.close()
        # Each consumer loads the other producer's sample; this is metadata/ID
        # file replay, without assuming encoder ownership or CP rank identity.
        replay = FixedRouting(
            model, directory, 'replay', 7, samples, group=torch.distributed.group.WORLD
        )
        other = 1 - rank
        item = _item(samples, order=(other,))
        replay.begin_round(item)
        for offset, (_, router) in enumerate(replay.routers):
            ids = replay.select(router, torch.zeros(4, 4), _unexpected_selector)
            length = samples[other]['original_seq_len']
            expected = _ids(item['kwargs']['input_ids'], offset)
            assert torch.equal(ids[:length], expected[:length])
        assert replay.metrics['source_samples'] == 2
        replay.close()
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize('failure', ['', 'missing_router', 'duplicate_owner'])
def test_distributed_publication_and_failure_are_synchronized(tmp_path, failure):
    # The CPU-only launcher invokes pytest from stdin, which spawn cannot reload.
    # These workers use only CPU tensors/Gloo; CUDA must remain uninitialized.
    assert not torch.cuda.is_initialized()
    torch.multiprocessing.start_processes(
        _distributed_worker,
        args=(str(tmp_path / 'rendezvous'), str(tmp_path / 'routes'), failure),
        nprocs=2,
        join=True,
        start_method='fork',
    )


@pytest.mark.parametrize(
    'field,value',
    [
        ('moe_router_force_load_balancing', True),
        ('moe_router_force_biased', 0.1),
        ('moe_input_jitter_eps', 0.1),
        ('moe_expert_capacity_factor', 1.0),
        ('moe_expert_rank_capacity_factor', 1.0),
    ],
)
def test_incompatible_router_flags_fail_before_attachment(tmp_path, field, value):
    model = _model()
    setattr(model.decoder.config, field, value)
    with pytest.raises(RuntimeError, match='ordinary dropless routing'):
        FixedRouting(model, tmp_path, 'record', 7, _samples())
    assert not hasattr(model.decoder, '_fixed_routing')


@pytest.mark.parametrize('field', ['linear_cp_layout', 'attention_cp_layout'])
def test_mixed_or_contiguous_router_layouts_are_rejected(tmp_path, field):
    model = _model()
    setattr(model.decoder.config, field, 'contiguous')
    with pytest.raises(RuntimeError, match='zigzag layouts at every router'):
        FixedRouting(model, tmp_path, 'record', 7, _samples())
    assert not hasattr(model.decoder, '_fixed_routing')


def _large_samples():
    samples = {}
    for sid, (length, padded) in enumerate(((131071, 131072), (3, 32))):
        samples[sid] = dict(
            original_seq_len=length,
            padded_seq_len=padded,
            tokens=torch.arange(padded) + sid * 10,
            labels=torch.arange(padded),
            loss_mask=torch.ones(padded),
            position_ids=torch.arange(padded).repeat(3, 1),
            media_ids=[sid],
        )
    return samples


def _record_large_cp1(directory):
    model = _model()
    samples = _large_samples()
    scope = FixedRouting(model, directory, 'record', 7, samples)
    for sid, sample in samples.items():
        item = _item(samples, order=(sid,), round_id=sid, widths=[sample['padded_seq_len']])
        scope.begin_round(item)
        count = item['kwargs']['input_ids'].numel()
        for offset, (_, router) in enumerate(scope.routers):
            scope.select(
                router,
                torch.zeros(count, 4),
                lambda offset=offset: _ids(item['kwargs']['input_ids'], offset),
            )
    scope.seal_recording()
    result = dict(scope.metrics)
    scope.close()
    return result


def _cp16_worker(rank, rendezvous, directory):
    fixed_routing.TopKRouter = _Router
    torch.distributed.init_process_group(
        'gloo',
        init_method=f'file://{rendezvous}',
        rank=rank,
        world_size=16,
        timeout=timedelta(seconds=90),
    )
    try:
        model = _model()
        samples = _large_samples()
        scope = FixedRouting(
            model, directory, 'record', 7, samples, group=torch.distributed.group.WORLD
        )
        for sid, sample in samples.items():
            width = sample['padded_seq_len']
            chunk = width // 32
            # Known fixture ownership, independent of the production partition
            # adapter. Its real zigzag mapping is covered by GPU partition tests.
            indices = torch.cat(
                (
                    torch.arange(rank * chunk, (rank + 1) * chunk),
                    torch.arange((31 - rank) * chunk, (32 - rank) * chunk),
                )
            )
            model.partition_adapter.indices = indices
            item = _item(samples, order=(sid,), round_id=sid, cp=16, widths=[width])
            scope.begin_round(item)
            local_tokens = item['kwargs']['input_ids'][:, indices]
            for offset, (_, router) in enumerate(scope.routers):
                scope.select(
                    router,
                    torch.zeros(indices.numel(), 4),
                    lambda offset=offset: _ids(local_tokens, offset),
                )
        scope.seal_recording()
        assert scope.metrics['recording_cp'] == 16
        assert scope.metrics['recording_cp_sizes'] == [16]
        scope.close()
    finally:
        torch.distributed.destroy_process_group()


def test_cp16_record_128k_and_empty_real_shards_match_cp1_schema(tmp_path):
    assert not torch.cuda.is_initialized()
    reference = _record_large_cp1(tmp_path / 'cp1')
    assert reference['recording_cp'] == 1
    assert not (tmp_path / 'cp1/step00000007/cp_parts').exists()
    torch.multiprocessing.start_processes(
        _cp16_worker,
        args=(str(tmp_path / 'rendezvous16'), str(tmp_path / 'cp16')),
        nprocs=16,
        join=True,
        start_method='fork',
    )
    model = _model()
    samples = _large_samples()
    scope = FixedRouting(model, tmp_path / 'cp16', 'replay', 7, samples)
    assert scope.metrics['route_sha256'] == reference['route_sha256']
    assert scope.manifest['identity']['schema'] == 1
    for sid, sample in samples.items():
        item = _item(samples, order=(sid,), round_id=sid, widths=[sample['padded_seq_len']])
        scope.begin_round(item)
        count = item['kwargs']['input_ids'].numel()
        for offset, (_, router) in enumerate(scope.routers):
            selected = scope.select(router, torch.zeros(count, 4), _unexpected_selector)
            length = sample['original_seq_len']
            assert torch.equal(
                selected[:length], _ids(item['kwargs']['input_ids'], offset)[:length]
            )
    scope.close()


def _mixed_cp_worker(rank, rendezvous, directory):
    fixed_routing.TopKRouter = _Router
    torch.distributed.init_process_group(
        'gloo',
        init_method=f'file://{rendezvous}',
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=60),
    )
    try:
        model = _model()
        samples = _samples()
        samples[2] = {**samples[0], 'tokens': samples[0]['tokens'] + 20, 'media_ids': [2]}
        cp, sid = (1, rank) if rank < 2 else (2, 2)
        if cp == 2:
            model.partition_adapter.indices = torch.tensor([0, 3] if rank == 2 else [1, 2])
        scope = FixedRouting(
            model, directory, 'record', 7, samples, group=torch.distributed.group.WORLD
        )
        item = _item(samples, order=(sid,), cp=cp)
        scope.begin_round(item)
        tokens = item['kwargs']['input_ids']
        if cp == 2:
            tokens = tokens[:, model.partition_adapter.indices]
        for offset, (_, router) in enumerate(scope.routers):
            scope.select(
                router, torch.zeros(tokens.numel(), 4), lambda offset=offset: _ids(tokens, offset)
            )
        scope.seal_recording()
        assert scope.metrics['recording_cp'] is None
        assert scope.metrics['recording_cp_sizes'] == [1, 2]
        scope.close()
    finally:
        torch.distributed.destroy_process_group()


def test_mixed_cp_recording_uses_one_collective_assembly_branch(tmp_path):
    assert not torch.cuda.is_initialized()
    torch.multiprocessing.start_processes(
        _mixed_cp_worker,
        args=(str(tmp_path / 'rendezvous4'), str(tmp_path / 'routes')),
        nprocs=4,
        join=True,
        start_method='fork',
    )
    manifest = json.loads((tmp_path / 'routes/step00000007/manifest.json').read_text())
    assert manifest['sample_owners'] == {'0': 0, '1': 1, '2': 2}


def test_old_cp1_manifest_supplies_source_hash_without_rewriting(tmp_path):
    model, scope, _ = _record(tmp_path)
    scope.close()
    path = tmp_path / 'step00000007/manifest.json'
    manifest = json.loads(path.read_text())
    manifest['metrics'].pop('source_sha256', None)
    manifest['metrics'].pop('recording_cp_sizes', None)
    path.write_text(json.dumps(manifest))
    before = path.read_bytes()
    replay = FixedRouting(model, tmp_path, 'replay', 7, _samples())
    assert replay.metrics['source_sha256'] == replay.identity['source_sha256']
    assert path.read_bytes() == before
    replay.close()
