# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Exact image-row oracle across real packed MIMO routing and CP partitioning.

The schedules are deliberate stress fixtures, not scheduler acceptance tests.
They reference some source images in more than one sample/round to test the
transpose's accumulation. The independent reference uses original image IDs,
sample positions and elementary zigzag arithmetic; it never consumes bridge
plans or TE partition indices. No transformer or optimizer numerics are involved.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from examples.mimo.data.packed_multimodal import assign_encoder_media, build_round_plans
from megatron.core.datasets.data_schedule_utils import build_packed_microbatches
from megatron.core.models.mimo.comm.pack_bridge import PackFeatureBridge
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.models.mimo.partition.utils import PartitionAdapter, PartitionConfig
from megatron.core.packed_seq_params import PackedSeqParams

_WIDTH = 3
_IMAGE_TOKEN = 100000
_IMAGE_LENGTHS = (7, 3, 5, 2, 6)
# (padded length, real length, (image ID, first placeholder position) ...)
_SAMPLES = (
    (32, 29, ((0, 6),)),
    (32, 30, ((2, 2), (1, 10))),
    (16, 14, ((0, 3),)),
    (16, 11, ()),
    (32, 28, ()),
    (32, 29, ((1, 7),)),
    (16, 13, ()),
    (32, 27, ((2, 6),)),
    (16, 12, ()),
)
_ADDITIONAL_IMAGES = {5: ((4, 14),), 7: ((3, 20),)}
_ASSIGNMENTS = (
    ((0, 3), (0, 3), (0, 3), (0, 3)),
    ((1,), (2,), (4,), (6,)),
    ((5, 8), (5, 8), (7,), (7,)),
)


@pytest.fixture(scope='module')
def routing_groups():
    """Explicit four-rank domains also work in the standard eight-GPU runner."""
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    if dist.get_world_size() % 4:
        pytest.skip('This oracle needs a multiple of four GPU ranks')
    rank = dist.get_rank()
    owned, created = {}, []
    for base in range(0, dist.get_world_size(), 4):
        for ranks in (
            tuple(range(base, base + 4)),
            (base, base + 1),
            (base + 2, base + 3),
            *((base + i,) for i in range(4)),
        ):
            group = dist.new_group(ranks=list(ranks), backend='nccl')
            if rank in ranks:
                owned[ranks] = group
                created.append(group)
    yield owned
    for group in reversed(created):
        dist.destroy_process_group(group)


def _identities(sample_ids, pure_text, extra_images=False):
    """Canonical identities from source positions, without candidate metadata."""
    rows = []
    for sid in sample_ids:
        padded, real, images = _SAMPLES[sid]
        if extra_images:
            images += _ADDITIONAL_IMAGES.get(sid, ())
        visual_positions = {}
        if not pure_text:
            for image_id, start in images:
                for row in range(_IMAGE_LENGTHS[image_id]):
                    assert start + row < real
                    assert start + row not in visual_positions
                    visual_positions[start + row] = (image_id, row)
        for pos in range(padded):
            rows.append((sid, pos, visual_positions.get(pos)))
    return rows


def _token(sid, pos, image_row):
    if image_row is not None:
        return _IMAGE_TOKEN
    return 1000 + sid * 64 + pos if pos < _SAMPLES[sid][1] else 0


def _source_row(image_id, row, step):
    return [1 + 32 * image_id + 4 * row + channel + 8 * step for channel in range(_WIDTH)]


def _cotangent(sid, pos, consumer, round_id, step):
    return [
        1 + sid + pos % 11 + consumer + 3 * round_id + 7 * step + channel
        for channel in range(_WIDTH)
    ]


def _owned_positions(sample_ids, cp_size, cp_rank):
    """Independent per-sequence zigzag, including each sequence's own padding."""
    positions, offset = [], 0
    for sid in sample_ids:
        length = _SAMPLES[sid][0]
        if cp_size == 1:
            positions.extend(range(offset, offset + length))
        else:
            assert length % (2 * cp_size) == 0
            chunk = length // (2 * cp_size)
            for part in (cp_rank, 2 * cp_size - 1 - cp_rank):
                positions.extend(range(offset + part * chunk, offset + (part + 1) * chunk))
        offset += length
    return positions


def _source_samples(pure_text, extra_images):
    """Candidate input schema, passed through the existing pack builder."""
    samples = {}
    for sid, (padded, real, images) in enumerate(_SAMPLES):
        if extra_images:
            images += _ADDITIONAL_IMAGES.get(sid, ())
        identities = _identities((sid,), pure_text, extra_images)
        tokens = torch.tensor([_token(*row) for row in identities], device='cuda')
        mask = torch.tensor([int(pos < real and pos >= 2) for pos in range(padded)], device='cuda')
        samples[sid] = {
            'tokens': tokens,
            'labels': torch.arange(padded, device='cuda') + sid * 64,
            'loss_mask': mask.float(),
            'position_ids': torch.arange(padded, device='cuda'),
            'original_seq_len': torch.tensor(real, dtype=torch.int32, device='cuda'),
            'padded_seq_len': torch.tensor(padded, dtype=torch.int32, device='cuda'),
            'media_ids': [] if pure_text else [image_id for image_id, _ in images],
        }
    return samples


def _exact(actual, expected):
    expected = torch.as_tensor(expected, dtype=actual.dtype, device=actual.device).reshape(
        actual.shape
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    return float((actual.detach() - expected).abs().max()) if actual.numel() else 0.0


def _source_oracle(local_image_ids, pure_text, step):
    """Sum every original occurrence once, regardless of the chosen producer."""
    per_round = []
    for round_id, assignment in enumerate(_ASSIGNMENTS):
        image_grads = {
            image: [[0] * _WIDTH for _ in range(length)]
            for image, length in enumerate(_IMAGE_LENGTHS)
        }
        seen_packs = set()
        for consumer, sample_ids in enumerate(assignment):
            if sample_ids in seen_packs:
                continue
            seen_packs.add(sample_ids)
            consumers = [i for i, ids in enumerate(assignment) if ids == sample_ids]
            identities = _identities(sample_ids, pure_text, step == 2)
            for cp_rank, owner in enumerate(consumers):
                for index in _owned_positions(sample_ids, len(consumers), cp_rank):
                    sid, pos, image_row = identities[index]
                    if image_row is None:
                        continue
                    image_id, row = image_row
                    values = _cotangent(sid, pos, owner, round_id, step)
                    image_grads[image_id][row] = [
                        a + b for a, b in zip(image_grads[image_id][row], values)
                    ]
        per_round.append([row for image_id in local_image_ids for row in image_grads[image_id]])
    return per_round


@pytest.mark.parametrize(
    'start_pure_text', (False, True), ids=('mixed_text_mixed', 'pure_text_only')
)
def test_exact_routing_and_saved_round_lifecycle(routing_groups, start_pure_text):
    rank = dist.get_rank()
    base = rank // 4 * 4
    domain = tuple(range(base, base + 4))
    local_rank = rank - base
    bridge = PackFeatureBridge(routing_groups[domain])
    # The static CP2 group must be overridden by runtime CP4 and CP1 packs.
    static_pair = domain[:2] if local_rank < 2 else domain[2:]
    adapter = PartitionAdapter(
        PartitionConfig(
            seq_parallel=False,
            use_cp=True,
            tp_comm_overlap=False,
            max_seq_len=64,
            kv_format='thd',
            cp_group=routing_groups[static_pair],
        )
    )
    splice = SimpleNamespace(
        _validate_precomputed_token_indices=MimoModel._validate_precomputed_token_indices
    )
    report = {
        'case': 'pure_text_only' if start_pure_text else 'mixed_text_mixed',
        'rank': rank,
        'dtype': 'float32',
        'rtol': 0,
        'atol': 0,
        'steps': [],
    }
    previous_states = []
    for step, pure_text in enumerate((True,) if start_pure_text else (False, True, False)):
        # Rotate assignment after the intervening text-only source batch. The
        # canonical initial cost order is image 0, image 2, image 1. The final
        # mixed step adds images 4 then 3 to the least-loaded last producer;
        # encoding sorts their IDs, so image 4 needs a nonzero source offset.
        producers = domain if step == 0 else domain[1:] + domain[:1]
        extra_images = step == 2
        media = (
            []
            if pure_text
            else [
                {'image_id': image, 'size': size, 'length': _IMAGE_LENGTHS[image]}
                for image, size in enumerate((8, 4, 6, 2, 3) if extra_images else (8, 4, 6))
            ]
        )
        tasks, slices = assign_encoder_media(media, producers)
        expected_owner = {0: producers[0], 2: producers[1], 1: producers[2]}
        if extra_images:
            expected_owner.update({3: producers[3], 4: producers[3]})
        local_images = (
            []
            if pure_text
            else sorted(image for image, owner in expected_owner.items() if owner == rank)
        )
        assert [item['image_id'] for item in tasks[rank]] == local_images
        for image, feature in slices.items():
            assert (feature.producer_rank, feature.offset, feature.length) == (
                expected_owner[image],
                2 if image == 4 else 0,
                _IMAGE_LENGTHS[image],
            )
        source_values = [
            _source_row(image, row, step)
            for image in local_images
            for row in range(_IMAGE_LENGTHS[image])
        ]
        source_input = torch.tensor(source_values, dtype=torch.float32, device='cuda').reshape(
            -1, _WIDTH
        )
        gain = torch.tensor(2 + step, dtype=torch.float32, device='cuda', requires_grad=True)
        source = source_input * gain
        source.retain_grad()
        samples = _source_samples(pure_text, extra_images)
        packed_samples = {
            sid: {key: value for key, value in sample.items() if isinstance(value, torch.Tensor)}
            for sid, sample in samples.items()
        }
        batches = build_packed_microbatches(
            packed_samples, _ASSIGNMENTS, local_rank, torch.device('cuda'), is_dynamic_cp=True
        )
        oracle_gradients = _source_oracle(local_images, pure_text, step)
        states, rows_report = [], []
        for round_id, (assignment, batch) in enumerate(zip(_ASSIGNMENTS, batches)):
            sample_ids = assignment[local_rank]
            cp_ranks = tuple(domain[i] for i, ids in enumerate(assignment) if ids == sample_ids)
            cp_size, cp_rank = len(cp_ranks), cp_ranks.index(rank)
            plans = build_round_plans(assignment, samples, slices, domain)
            state = bridge.forward(source, plans, routing_groups[cp_ranks])
            assert state.features.is_leaf and state.features.requires_grad
            assert state.local_features is source
            identities = _identities(sample_ids, pure_text, extra_images)
            expected_full = []
            for sid, pos, image_row in identities:
                if image_row is None:
                    token = _token(sid, pos, image_row)
                    expected_full.append([-token - channel for channel in range(_WIDTH)])
                else:
                    expected_full.append(
                        [value * (2 + step) for value in _source_row(*image_row, step)]
                    )
            image_indices = [i for i, row in enumerate(identities) if row[2] is not None]
            expected_features = [expected_full[i] for i in image_indices]
            forward_error = _exact(state.features, expected_features)
            _exact(batch['tokens'], [_token(*row) for row in identities])
            input_ids = batch['tokens'].unsqueeze(0)
            text_indices = (input_ids.reshape(-1) != _IMAGE_TOKEN).nonzero().flatten()
            text_tokens = input_ids.reshape(-1).index_select(0, text_indices)
            text = (
                (-text_tokens[:, None] - torch.arange(_WIDTH, device='cuda'))
                .float()
                .requires_grad_()
            )
            modality_indices = None
            if step == 2:
                modality_indices = {
                    'text': text_indices,
                    'images': (input_ids.reshape(-1) == _IMAGE_TOKEN).nonzero().flatten(),
                }
            merged = MimoModel.align_embeddings_by_token_positions(
                splice,
                {'text': text, 'images': state.features},
                input_ids,
                {'images': _IMAGE_TOKEN},
                modality_token_indices=modality_indices,
            )
            merge_error = _exact(merged, expected_full)
            packed = PackedSeqParams(
                qkv_format='thd',
                cu_seqlens_q=batch['cu_seqlens'],
                cu_seqlens_kv=batch['cu_seqlens'],
                cu_seqlens_q_padded=batch['cu_seqlens_padded'],
                cu_seqlens_kv_padded=batch['cu_seqlens_padded'],
                max_seqlen_q=max(_SAMPLES[sid][0] for sid in sample_ids),
                max_seqlen_kv=max(_SAMPLES[sid][0] for sid in sample_ids),
                local_cp_size=cp_size,
                cp_group=routing_groups[cp_ranks],
            )
            local, labels, masks, _ = adapter.shard(
                merged, batch['labels'][None], batch['loss_mask'][None], packed
            )
            positions = _owned_positions(sample_ids, cp_size, cp_rank)
            partition_error = _exact(local, [expected_full[i] for i in positions])
            _exact(labels, [identities[i][0] * 64 + identities[i][1] for i in positions])
            _exact(
                masks,
                [int(2 <= identities[i][1] < _SAMPLES[identities[i][0]][1]) for i in positions],
            )
            cotangent = torch.tensor(
                [_cotangent(*identities[i][:2], local_rank, round_id, step) for i in positions],
                dtype=torch.float32,
                device='cuda',
            )[:, None, :]
            (local * cotangent).sum().backward()
            assert source.grad is None and gain.grad is None
            feature_gradient = [[0] * _WIDTH for _ in image_indices]
            feature_slot = {position: slot for slot, position in enumerate(image_indices)}
            for position in positions:
                if position in feature_slot:
                    feature_gradient[feature_slot[position]] = _cotangent(
                        *identities[position][:2], local_rank, round_id, step
                    )
            assert state.features.grad is not None
            leaf_error = _exact(state.features.grad, feature_gradient)
            owners = {}
            for consumer_cp_rank, consumer in enumerate(cp_ranks):
                for position in _owned_positions(sample_ids, cp_size, consumer_cp_rank):
                    image_row = identities[position][2]
                    if image_row is not None:
                        owners.setdefault(str(image_row[0]), set()).add(consumer)
            # These wrong answers are evaluated locally, never sent to peers.
            negative_controls = {}
            if state.features.numel():
                negative_controls['shift_feature_offset_rejected'] = not torch.equal(
                    state.features.detach(), state.features.detach().roll(1, dims=0)
                )
                assert negative_controls['shift_feature_offset_rejected']
            if sample_ids == (1,) and not pure_text:
                swapped = expected_features[5:] + expected_features[:5]
                negative_controls['swap_image_order_rejected'] = not torch.equal(
                    state.features.detach(),
                    torch.tensor(swapped, dtype=torch.float32, device='cuda'),
                )
                assert negative_controls['swap_image_order_rejected']
            rows_report.append(
                {
                    'round': round_id,
                    'cp_size': cp_size,
                    'cp_ranks': cp_ranks,
                    'splice_mode': (
                        'precomputed_indices' if modality_indices is not None else 'token_mask'
                    ),
                    'sample_ids': sample_ids,
                    'pack_visual_rows': len(image_indices),
                    'local_visual_rows': sum(position in feature_slot for position in positions),
                    'image_owners': {image: sorted(ranks) for image, ranks in owners.items()},
                    'forward_max_abs': forward_error,
                    'splice_max_abs': merge_error,
                    'partition_max_abs': partition_error,
                    'leaf_gradient_max_abs': leaf_error,
                    'negative_controls': negative_controls,
                }
            )
            states.append(state)
        assert [len(state.plans[state.local_plan].cp_ranks) for state in states] == [4, 1, 2]
        accumulated = torch.zeros_like(source)
        for round_id in (2, 0, 1):
            state = states[round_id]
            returned = bridge.backward(state)
            rows_report[round_id]['returned_gradient_max_abs'] = _exact(
                returned, oracle_gradients[round_id]
            )
            accumulated.add_(returned)
            if returned.numel() and rows_report[round_id]['cp_size'] > 1 and returned.abs().sum():
                assert not torch.equal(returned, returned / rows_report[round_id]['cp_size'])
                rows_report[round_id]['negative_controls']['cp_average_rejected'] = True
            with pytest.raises(RuntimeError, match='only be returned once'):
                bridge.backward(state)
        # One encoder backward after all decoder rounds. An empty producer still
        # has a valid empty graph, with exactly zero scalar-parameter gradient.
        expected_source_gradient = torch.zeros_like(source)
        for values in oracle_gradients:
            expected_source_gradient.add_(
                torch.tensor(values, dtype=torch.float32, device='cuda').reshape_as(source)
            )
        source.backward(accumulated)
        source_error = _exact(source.grad, expected_source_gradient)
        parameter_error = _exact(gain.grad, (source_input * expected_source_gradient).sum())
        for stale in previous_states:
            with pytest.raises(RuntimeError, match='only be returned once'):
                bridge.backward(stale)
        previous_states = states
        step_report = {
            'step': step,
            'pure_text': pure_text,
            'producer_order': producers,
            'local_images': local_images,
            'local_encoder_rows': source.shape[0],
            'empty_producer': source.shape[0] == 0,
            'feature_offsets': {str(image): slices[image].offset for image in local_images},
            'rounds': rows_report,
            'source_gradient_max_abs': source_error,
            'encoder_parameter_gradient_max_abs': parameter_error,
            'saved_cp_sizes': [4, 1, 2],
            'backward_round_order': [2, 0, 1],
            'double_backward_guard_passed': True,
            'stale_step_guard_passed': True,
        }
        report['steps'].append(step_report)
    if not start_pure_text:
        first = report['steps'][0]['rounds'][0]
        assert first['image_owners']['0'] == [base + 1, base + 2, base + 3]
        if local_rank == 0:
            assert first['pack_visual_rows'] == 7 and first['local_visual_rows'] == 0
        assert report['steps'][0]['local_images'] != report['steps'][2]['local_images']
    destination = os.environ.get('MIMO_ROUTING_ORACLE_REPORT_DIR')
    if destination:
        path = Path(destination)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"routing_lifecycle_{report['case']}_rank{rank}.json").write_text(
            json.dumps(report, indent=2) + '\n'
        )
