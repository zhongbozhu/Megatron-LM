# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CPU-only diagnostic of real MIMO source batches and the existing schedulers.

No model, process group, GPU kernel, optimizer or new packing algorithm is used.
Partial coverage remains explicit when the caller's wall-time budget expires.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch

from examples.mimo.data.packed_multimodal import assign_encoder_media, build_round_plans
from examples.mimo.data.qwen35_native import Qwen35Dataset
from examples.mimo.fixed_packing import _pack_layout
from examples.mimo.fixed_routing import source_fingerprint
from megatron.core.datasets.data_schedule import DefaultDynamicCPScheduler, DpBalancedScheduler


def _schedule(samples, media, *, dynamic, cp, capacity, return_assignments=False):
    scheduler_type = DefaultDynamicCPScheduler if dynamic else DpBalancedScheduler
    scheduler = scheduler_type(capacity, cp, 16 // cp, None)
    assignments = scheduler.get_groups_and_subsamples(
        [(sid, int(sample['padded_seq_len'])) for sid, sample in samples.items()]
    )
    config = SimpleNamespace(
        max_seqlen_per_dp_cp_rank=capacity,
        pad_packed_seq_alignment=2,
        sequence_parallel=False,
        fp8=None,
    )
    _, slices = assign_encoder_media(media, tuple(range(16)))
    coverage, sizes, local_lengths = [], Counter(), []
    for assignment in assignments:
        plans = build_round_plans(assignment, samples, slices, tuple(range(16)))
        for plan in plans:
            ids = assignment[plan.cp_ranks[0]]
            size = len(plan.cp_ranks)
            assert dynamic or size == cp
            layout = _pack_layout(ids, samples, config, size)
            assert layout['physical_boundaries'][-1] % size == 0
            local_lengths.append(layout['physical_boundaries'][-1] // size)
            coverage.extend(ids)
            sizes[size] += 1
    assert sorted(coverage) == sorted(samples), 'Lost or duplicated source samples'
    summary = dict(
        rounds=len(assignments),
        packs=sum(sizes.values()),
        cp_pack_counts=dict(sorted(sizes.items())),
        min_local_tokens=min(local_lengths),
        max_local_tokens=max(local_lengths),
        source_coverage_exact=True,
        nonempty_rank_packs=True,
        core_thd_capacity_and_alignment_valid=True,
    )
    return (summary, assignments) if return_assignments else summary


def _inspect(dataset, step, *, long_context=False):
    samples, media = dataset.build_global_batch(step, 64)
    assert sorted(samples) == list(range(64))
    by_id = {image['image_id']: image for image in media}
    assert len(by_id) == len(media)
    lengths, padded, supervision, image_counts = [], [], [], []
    for sample in samples.values():
        length, physical = int(sample['original_seq_len']), int(sample['padded_seq_len'])
        assert 0 < length <= physical <= dataset.max_length
        assert physical % 32 == 0
        assert sample['tokens'].numel() == sample['labels'].numel() == physical
        assert sample['position_ids'].shape == (3, physical)
        assert not sample['loss_mask'][length:].any()
        count = int(sample['loss_mask'].sum())
        assert count > 0
        placeholders = int((sample['tokens'][:length] == dataset.image_token_id).sum())
        assert placeholders == sum(by_id[mid]['length'] for mid in sample['media_ids'])
        lengths.append(length)
        padded.append(physical)
        supervision.append(count)
        image_counts.append(len(sample['media_ids']))
    expected_media = [mid for sample in samples.values() for mid in sample['media_ids']]
    assert sorted(expected_media) == sorted(by_id)
    digest = source_fingerprint(samples, media)
    repeated_samples, repeated_media = dataset.build_global_batch(step, 64)
    assert digest == source_fingerprint(repeated_samples, repeated_media)
    producers, _ = assign_encoder_media(media, tuple(range(16)))
    result = dict(
        split=dataset.split,
        source_step=step,
        source_sha256=digest,
        repeated_address_source_sha256_exact=True,
        source_samples=len(samples),
        source_tokens=sum(lengths),
        padded_tokens=sum(padded),
        min_tokens=min(lengths),
        max_tokens=max(lengths),
        min_supervised_tokens=min(supervision),
        supervised_tokens=sum(supervision),
        source_images=len(media),
        unique_source_images=len({image['source_id'] for image in media}),
        min_images_per_sample=min(image_counts),
        max_images_per_sample=max(image_counts),
        encoder_image_counts=[len(producers[rank]) for rank in range(16)],
        encoder_feature_rows=[
            sum(image['length'] for image in producers[rank]) for rank in range(16)
        ],
        sample_lengths=lengths,
        supervision_counts=supervision,
        images_per_sample=image_counts,
        static=_schedule(
            samples,
            media,
            dynamic=False,
            cp=16 if long_context else 1,
            capacity=8192 if long_context else 40960,
        ),
        dynamic=_schedule(samples, media, dynamic=True, cp=8, capacity=8192),
    )
    if long_context:
        assert min(lengths) >= 120000 and min(image_counts) >= 2
        assert 16 in result['dynamic']['cp_pack_counts']
    assert not torch.cuda.is_initialized()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--hf-model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget-seconds', type=int, default=480)
    args = parser.parse_args()
    assert not torch.cuda.is_available() and not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    started = time.monotonic()
    report = dict(
        scope='CPU source/scheduler/THD metadata only; no model or GPU validation',
        requested_train_steps=list(range(50)),
        requested_eval_steps=list(range(3)),
        records=[],
        complete=False,
        cuda_initialized=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream)

    def save():
        report['elapsed_seconds'] = time.monotonic() - started
        report['cuda_initialized'] = torch.cuda.is_initialized()
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    try:
        train = Qwen35Dataset(args.manifest, args.hf_model, seq_length=40960)
        datasets = {'train': train}
        # Inspect both ends and the long case before filling the middle, so a
        # bounded partial run still covers the highest-value source addresses.
        addresses = (
            [('train', step) for step in (0, 1, 2, 3, 4, 49)]
            + [('heldout', step) for step in range(3)]
            + [('long', 0)]
            + [('train', step) for step in range(5, 49)]
        )
        for split, step in addresses:
            if time.monotonic() - started > args.budget_seconds:
                report['stopped_reason'] = 'CPU wall-time budget reached; coverage is partial'
                break
            if split not in datasets:
                datasets[split] = Qwen35Dataset(
                    args.manifest,
                    args.hf_model,
                    seq_length=131072 if split == 'long' else 40960,
                    split='train' if split == 'long' else split,
                    long_sample_min_tokens=120000 if split == 'long' else 0,
                )
            record = _inspect(datasets[split], step, long_context=split == 'long')
            record['case'] = split
            report['records'].append(record)
            save()
            print(
                json.dumps(
                    {
                        key: record[key]
                        for key in (
                            'case',
                            'source_step',
                            'source_tokens',
                            'source_images',
                            'static',
                            'dynamic',
                        )
                    }
                ),
                flush=True,
            )
        report['complete'] = len(report['records']) == len(addresses)
        report['dataset_summary'] = {name: dataset.summary for name, dataset in datasets.items()}
        save()
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        save()
        raise
    print(
        json.dumps(
            dict(
                output=str(args.output),
                complete=report['complete'],
                cuda_initialized=torch.cuda.is_initialized(),
            )
        ),
        flush=True,
    )
    if not report['complete']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
