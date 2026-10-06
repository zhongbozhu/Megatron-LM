# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Offline evidence for the same-layout native bridge/direct-reference experiment."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from examples.mimo.boundary_diagnostics import (
    _difference,
    analyze_boundary_directory,
    compare_boundary_directories,
)
from examples.mimo.direct_feature_reference import compare_full_decoder_inputs
from examples.mimo.execution_diagnostics import compare_execution_directories

CASES = ('pack_record', 'pack_repeat', 'direct_reference', 'direct_repeat')


def identity_adjoint(frozen_batch, boundary_dir, execution_dir, step):
    """Use original image IDs/token positions, never receiver slices or bridge plans."""
    samples, media = frozen_batch['samples'], frozen_batch['media']
    original_images = {item['image_id']: item for item in media}
    directory = Path(boundary_dir) / f'step{step:08d}'
    source_images, locations, returned = {}, {}, {}
    for path in sorted(directory.glob('producer-rank*.pt')):
        record = torch.load(path, map_location='cpu', weights_only=True)
        rank, offset = record['rank'], 0
        grad_record = torch.load(
            directory / f'producer_gradient-rank{rank:05d}.pt',
            map_location='cpu',
            weights_only=True,
        )
        for item in record['media']:
            image_id, length = item['image_id'], item['length']
            assert image_id not in source_images and item['offset'] == offset
            assert item['source_id'] == original_images[image_id]['source_id']
            assert length == original_images[image_id]['length']
            source_images[image_id] = record['values'][offset : offset + length]
            returned[image_id] = grad_record['values'][offset : offset + length]
            locations[image_id] = dict(rank=rank, offset=offset, rows=length)
            offset += length
        assert offset == record['values'].shape[0]
    assert source_images.keys() == original_images.keys()
    expected = {
        image: torch.zeros_like(value, dtype=torch.float64)
        for image, value in source_images.items()
    }
    counts = {
        image: torch.zeros(len(value), dtype=torch.int32) for image, value in source_images.items()
    }
    owners = {image: set() for image in source_images}
    sample_rows = {}
    for sid, sample in samples.items():
        positions = (
            (sample['tokens'][: sample['original_seq_len']] == 248056).nonzero().flatten().tolist()
        )
        rows = [
            (image, row)
            for image in sample['media_ids']
            for row in range(len(source_images[image]))
        ]
        assert len(positions) == len(rows)
        sample_rows[sid] = list(zip(positions, rows))
    max_forward_error, nonowner_nonzero = 0.0, 0
    rank_coverage = []
    for path in sorted((Path(execution_dir) / f'step{step:08d}').glob('rank*/training-round*.pt')):
        execution = torch.load(path, map_location='cpu', weights_only=True)
        rank, round_id = int(path.parent.name[4:]), execution['round']
        suffix = f'rank{rank:05d}-round{round_id:05d}-training.pt'
        receiver = torch.load(
            directory / f'receiver-{suffix}', map_location='cpu', weights_only=True
        )
        gradient = torch.load(
            directory / f'receiver_gradient-{suffix}', map_location='cpu', weights_only=True
        )
        ids = [image for sid in execution['sample_ids'] for image in samples[sid]['media_ids']]
        canonical_features = torch.cat([source_images[image] for image in ids])
        error = _difference(receiver['values'], canonical_features)
        max_forward_error = max(max_forward_error, error['max_abs_error'])
        assert error['all_finite']
        physical = set(execution['physical_indices'].tolist())
        owner_rows, feature_row = [], 0
        for slot, sid in enumerate(execution['sample_ids']):
            start = execution['padded_boundaries'][slot]
            for position, (image, row) in sample_rows[sid]:
                owned = start + position in physical
                counts[image][row] += owned
                if owned:
                    owners[image].add(rank)
                    owner_rows.append((image, row))
                else:
                    nonowner_nonzero += int(torch.count_nonzero(gradient['values'][feature_row]))
                expected[image][row] += gradient['values'][feature_row].double()
                feature_row += 1
        assert feature_row == gradient['values'].shape[0]
        rank_coverage.append(
            dict(
                rank=rank,
                cp_ranks=execution['cp_ranks'],
                sample_ids=execution['sample_ids'],
                owned_visual_rows=len(owner_rows),
            )
        )
    assert len(rank_coverage) == 16
    assert all(torch.equal(value, torch.ones_like(value)) for value in counts.values())
    comparisons = {
        str(image): _difference(returned[image], value) for image, value in expected.items()
    }
    assert all(value['all_finite'] for value in comparisons.values())
    return dict(
        oracle='Original frozen sample/image/token identities; receiver slices and production plans are never read',
        images=len(source_images),
        visual_rows=sum(len(value) for value in source_images.values()),
        every_visual_row_owned_once=True,
        rank_coverage=rank_coverage,
        image_producers=locations,
        image_owners={image: sorted(ranks) for image, ranks in owners.items()},
        forward_max_abs_error=max_forward_error,
        nonowner_nonzero_gradient_elements=nonowner_nonzero,
        image_returned_gradients=comparisons,
        passed=max_forward_error == 0
        and nonowner_nonzero == 0
        and all(value['exact'] for value in comparisons.values()),
    )


def optimizer_hash_comparison(directory, left, right):
    """All elements, not sampled fingerprints; ownership and starting state also match."""
    fields = ('entries', 'gradients', 'post_state', 'post_groups', 'post_vector_sha256')
    matches = {field: True for field in fields}
    elements = 0
    for rank in range(16):
        before = json.loads((directory / f'{left}-optimizer.rank{rank:05d}.json').read_text())
        after = json.loads((directory / f'{right}-optimizer.rank{rank:05d}.json').read_text())
        for field in fields:
            matches[field] &= before[field] == after[field]
        elements += sum(entry['metadata']['elements'] for entry in before['entries'])
    return dict(
        ranks=16,
        complete_owned_parameter_elements=elements,
        exact=all(matches.values()),
        fields=matches,
        evidence='SHA256 of every byte in every owned pre-clipping/Adam gradient and post-step FP32 master/Adam moment; matching initial state and ownership',
    )


def report(group_dir):
    group_dir = Path(group_dir)
    output = group_dir / 'analysis'
    metrics = {
        case: json.loads(
            (group_dir / 'logs' / f'{case}-metrics/metrics.jsonl').read_text().splitlines()[-1]
        )
        for case in CASES
    }
    steps = {value['step'] for value in metrics.values()}
    assert len(steps) == 1
    step = steps.pop()
    batch_path = group_dir / 'data/joint_source.pt'
    batch = torch.load(batch_path, map_location='cpu', weights_only=True)
    results = {}
    for case in CASES:
        directory = group_dir / 'logs' / f'{case}-diagnostics'
        independent = identity_adjoint(batch, directory / 'boundary', directory / 'decoder', step)
        standard = analyze_boundary_directory(directory / 'boundary', step)
        (output / f'{case}-identity-adjoint.json').write_text(
            json.dumps(independent, indent=2) + '\n'
        )
        (output / f'{case}-boundary-adjoint.json').write_text(json.dumps(standard, indent=2) + '\n')
        results[case] = dict(
            loss=metrics[case]['loss'],
            mtp_loss=metrics[case]['mtp_loss'],
            aux_loss=metrics[case]['aux']['loss'],
            cp_sizes=metrics[case]['cp_sizes'],
            gradient_reference=metrics[case]['full_gradient_validation'],
            independent_identity_adjoint_passed=independent['passed'],
            recorded_boundary_adjoint_exact=standard['all_exact'],
            route_sha256=metrics[case]['fixed_routing']['route_sha256'],
        )
    pairs = {}
    for left, right, purpose in (
        ('pack_record', 'pack_repeat', 'record_to_replay_controller_confound'),
        ('pack_repeat', 'direct_reference', 'same_replay_mode_bridge_vs_direct'),
        ('direct_reference', 'direct_repeat', 'same_mode_direct_repeat'),
    ):
        before, after = (group_dir / 'logs' / f'{case}-diagnostics' for case in (left, right))
        pair = dict(
            purpose=purpose,
            decoder=compare_full_decoder_inputs(before / 'decoder', after / 'decoder', step),
            encoder=compare_boundary_directories(
                before / 'boundary', after / 'boundary', step, expected_images=17
            ),
        )
        if left != 'pack_record':
            pair['complete_optimizer_hash_comparison'] = optimizer_hash_comparison(
                output, left, right
            )
        else:
            probes = compare_execution_directories(before / 'decoder', after / 'decoder', step)
            (output / 'record-vs-replay-decoder-probes.json').write_text(
                json.dumps(probes, indent=2) + '\n'
            )
        key = left + '-vs-' + right
        (output / f'{key}.json').write_text(json.dumps(pair, indent=2) + '\n')
        pairs[key] = pair
    runtime_hashes = [
        json.loads(
            (group_dir.parent / 'analysis' / f'group1_r3_{case}' / 'source-sha256.json').read_text()
        )
        for case in CASES
    ]
    assert all(value == runtime_hashes[0] for value in runtime_hashes)
    same_mode = [pair for pair in pairs.values() if 'complete_optimizer_hash_comparison' in pair]
    passed = all(
        result['independent_identity_adjoint_passed'] for result in results.values()
    ) and all(
        pair['encoder']['all_exact']
        and pair['complete_optimizer_hash_comparison']['exact']
        and all(value['exact'] for value in pair['decoder']['totals'].values())
        for pair in same_mode
    )
    summary = dict(
        group='full_native_same_layout_bridge_reference',
        source_step=step,
        optimizer_update='52 -> 53',
        passed=passed,
        scope='One numerical step from the identical warm checkpoint, real 17 resized CLEVR images and artificial text; full Qwen3.5-35B, MIMO ViT, MTP, MoE EP16, native DDP/Adam',
        limitations=[
            'Equal 16 vision rows/image; no variable-resolution claim',
            'Not a convergence/quality experiment',
            'Record-to-replay discrepancy is a separate diagnostic-controller confound, not same-mode repeat noise or proof of harmless roundoff',
        ],
        source_fixture_sha256=hashlib.sha256(batch_path.read_bytes()).hexdigest(),
        runtime_python_hashes_identical=True,
        runtime_python_files=len(runtime_hashes[0]),
        decoder_capture_coverage=dict(
            full_decoder_input_values_captured=True,
            full_decoder_input_gradient_captured=all(
                'training_gradient' in pair['decoder']['totals'] for pair in same_mode
            ),
            visual_leaf_gradients_exhaustively_captured=True,
        ),
        cases=results,
        comparisons=pairs,
        failed_launcher_attempts=json.loads((output / 'launcher-attempts.json').read_text()),
    )
    if not summary['decoder_capture_coverage']['full_decoder_input_gradient_captured']:
        summary['limitations'].append(
            'Full text-input gradients were not captured; actual visual leaf VJPs, '
            'returned encoder feature gradients, and all parameter gradients were '
            'captured/hashed completely.'
        )
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(
        json.dumps(
            dict(
                passed=passed,
                source_step=step,
                images=17,
                same_mode_complete_hashes=[
                    pair['complete_optimizer_hash_comparison']['exact'] for pair in same_mode
                ],
            )
        ),
        flush=True,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group-dir', required=True)
    report(parser.parse_args().group_dir)
