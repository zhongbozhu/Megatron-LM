# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CPU/Gloo check of actual MIMO canonical targets across recorded CP1 and DCP.

Uses existing source, packing, zigzag metadata and MTP rolling implementations.
The TE CUDA index kernel itself is outside this diagnostic's scope. Raw values
at masked MTP positions are reported separately from supervised target equality.
"""

import argparse
import json
import time
from collections import Counter
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from examples.mimo.data.packed_multimodal import assign_encoder_media, build_round_plans
from examples.mimo.fixed_packing import apply_fixed_packing
from examples.mimo.fixed_routing import source_fingerprint
from examples.mimo.source_schedule_preflight import Qwen35Dataset, _schedule
from megatron.core.context_parallel.layout import _build_thd_zigzag_metadata
from megatron.core.datasets.data_schedule import _build_thd_padding_mask
from megatron.core.datasets.data_schedule_utils import (
    build_packed_microbatches,
    pad_packed_batch_before_cp_slice,
)
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.multi_token_prediction import _iter_mtp_loss_masks, roll_tensor


def _config(capacity):
    return SimpleNamespace(
        max_seqlen_per_dp_cp_rank=capacity,
        pad_packed_seq_alignment=2,
        sequence_parallel=False,
        fp8=None,
    )


def _params(batch, group=None):
    return PackedSeqParams(
        qkv_format='thd',
        cu_seqlens_q=batch['cu_seqlens'],
        cu_seqlens_kv=batch['cu_seqlens'],
        cu_seqlens_q_padded=batch['cu_seqlens_padded'],
        cu_seqlens_kv_padded=batch['cu_seqlens_padded'],
        local_cp_size=group.size() if group is not None else None,
        cp_group=group,
    )


def _targets(tokens, labels, mask, padding, params, group=None):
    conditioning = MimoModel._materialize_mtp_input_mask(tokens, {'images': 248056})
    shifted_labels, _ = roll_tensor(
        labels, cp_group=group, packed_seq_params=params, return_sum=False
    )
    mtp_mask, _ = next(_iter_mtp_loss_masks(mask, 1, conditioning, group, params))
    metadata, _ = roll_tensor(
        torch.cat((tokens, conditioning.long())),
        cp_group=group,
        packed_seq_params=params,
        return_sum=False,
    )
    shifted_tokens, shifted_conditioning = metadata.chunk(2)
    valid, _ = roll_tensor(~padding, cp_group=group, packed_seq_params=params, return_sum=False)
    return dict(
        tokens=tokens.flatten(),
        labels=labels.flatten(),
        loss_mask=mask.flatten(),
        mtp_labels=shifted_labels.flatten(),
        mtp_loss_mask=mtp_mask.flatten(),
        mtp_input_ids=shifted_tokens.flatten(),
        mtp_input_mask=shifted_conditioning.flatten().bool(),
        mtp_padding_valid=valid.flatten(),
    )


def _oracle(samples):
    result = {}
    for sid, sample in samples.items():
        original, padded = sample['original_seq_len'], sample['padded_seq_len']
        batch = dict(
            cu_seqlens=torch.tensor([0, original], dtype=torch.int32),
            cu_seqlens_padded=torch.tensor([0, padded], dtype=torch.int32),
        )
        result[sid] = _targets(
            sample['tokens'][None],
            sample['labels'][None],
            sample['loss_mask'][None],
            (torch.arange(padded) >= original)[None],
            _params(batch),
        )
    return result


def _check_layout(rank, case, layout, groups):
    samples, oracle = case['samples'], case['oracle']
    assignments = case['assignments'][layout]
    packed_samples = {
        sid: {
            **{key: sample[key] for key in ('tokens', 'labels', 'loss_mask')},
            **{
                key: torch.tensor(sample[key], dtype=torch.int32)
                for key in ('original_seq_len', 'padded_seq_len')
            },
        }
        for sid, sample in samples.items()
    }
    batches = build_packed_microbatches(
        packed_samples, assignments, rank, torch.device('cpu'), layout == 'dcp'
    )
    lengths = [sample['original_seq_len'] for sample in samples.values()]
    offsets = torch.tensor([0, *lengths]).cumsum(0)
    coverage = torch.zeros(sum(lengths), dtype=torch.int32)
    counts, probes = Counter(), []
    exact_fields = ('tokens', 'labels', 'loss_mask', 'mtp_loss_mask', 'mtp_padding_valid')
    for assignment, batch in zip(assignments, batches):
        plans = build_round_plans(assignment, samples, case['slices'], tuple(range(16)))
        plan = next(plan for plan in plans if rank in plan.cp_ranks)
        group = groups[len(plan.cp_ranks)]
        assert tuple(dist.get_process_group_ranks(group)) == plan.cp_ranks
        size, local_rank = group.size(), group.rank()
        batch['padding_mask'] = _build_thd_padding_mask(
            batch['cu_seqlens'], batch['cu_seqlens_padded']
        )
        pad_packed_batch_before_cp_slice(
            batch, _config(45056 if layout == 'cp1' else 8192), size, 1
        )
        metadata = _build_thd_zigzag_metadata(
            batch['cu_seqlens'], batch['cu_seqlens_padded'], size, 1
        )
        assert torch.equal(metadata.cu_seqlens_padded, batch['cu_seqlens_padded'])
        ordered = metadata.rank_order_indices.view(size, -1)
        assert torch.equal(ordered.flatten().sort().values, torch.arange(batch['tokens'].numel()))
        index = ordered[local_rank]
        local = {
            key: batch[key].index_select(0, index)[None]
            for key in ('tokens', 'labels', 'loss_mask', 'padding_mask')
        }
        actual = _targets(
            local['tokens'],
            local['labels'],
            local['loss_mask'],
            local['padding_mask'],
            _params(batch, group),
            group,
        )
        padding = local['padding_mask'].flatten()
        counts['physical_padding_rows'] += int(padding.sum())
        counts['main_supervised_padding_rows'] += int((actual['loss_mask'][padding] != 0).sum())
        counts['mtp_supervised_padding_rows'] += int((actual['mtp_loss_mask'][padding] != 0).sum())
        counts['padding_raw_mtp_label_nonzero'] += int((actual['mtp_labels'][padding] != 0).sum())
        owner = torch.empty(batch['tokens'].numel(), dtype=torch.int64)
        owner[ordered.flatten()] = torch.arange(size).repeat_interleave(ordered.shape[1])
        for slot, sid in enumerate(assignment[rank]):
            length = samples[sid]['original_seq_len']
            start = int(batch['cu_seqlens_padded'][slot])
            rows = ((index >= start) & (index < start + length)).nonzero().flatten()
            positions = index[rows] - start
            coverage.index_add_(
                0, offsets[sid] + positions, torch.ones_like(positions, dtype=torch.int32)
            )
            expected = {key: value[positions] for key, value in oracle[sid].items()}
            observed = {key: value[rows] for key, value in actual.items()}
            for field in exact_fields:
                counts[field + '_mismatches'] += int((observed[field] != expected[field]).sum())
            raw_target_diff = observed['mtp_labels'] != expected['mtp_labels']
            supervised = (observed['mtp_loss_mask'] != 0) | (expected['mtp_loss_mask'] != 0)
            counts['mtp_raw_target_mismatches'] += int(raw_target_diff.sum())
            counts['mtp_supervised_target_mismatches'] += int((raw_target_diff & supervised).sum())
            valid = observed['mtp_padding_valid'] | expected['mtp_padding_valid']
            for field in ('mtp_input_ids', 'mtp_input_mask'):
                different = observed[field] != expected[field]
                counts[field + '_raw_mismatches'] += int(different.sum())
                counts[field + '_valid_mismatches'] += int((different & valid).sum())
            next_index = (index[rows] + 1).clamp_max(owner.numel() - 1)
            seams = (positions + 1 < length) & (owner[next_index] != local_rank)
            counts['real_cp_seam_rows'] += int(seams.sum())
            counts['real_sequence_end_rows'] += int((positions == length - 1).sum())
            if sid == case['focus_sample']:
                vision = (samples[sid]['tokens'][:length] == 248056).nonzero().flatten()
                selected = (positions < 2) | (positions >= length - 3) | seams
                if vision.numel():
                    selected |= (positions - int(vision[0])).abs() <= 1
                    selected |= (positions - int(vision[-1])).abs() <= 1
                for offset in selected.nonzero().flatten().tolist():
                    probes.append(
                        dict(
                            sample_id=sid,
                            position=int(positions[offset]),
                            rank=rank,
                            cp_ranks=list(plan.cp_ranks),
                            cp_seam=bool(seams[offset]),
                            actual={k: v[offset].item() for k, v in observed.items()},
                            expected={k: v[offset].item() for k, v in expected.items()},
                        )
                    )
    dist.all_reduce(coverage)
    results = [None] * 16
    dist.all_gather_object(results, dict(counts=dict(counts), probes=probes))
    if rank != 0:
        return None
    totals = Counter()
    for result in results:
        totals.update(result['counts'])
    required_zero = [field + '_mismatches' for field in exact_fields] + [
        'main_supervised_padding_rows',
        'mtp_supervised_padding_rows',
        'mtp_supervised_target_mismatches',
        'mtp_input_ids_valid_mismatches',
        'mtp_input_mask_valid_mismatches',
    ]
    return dict(
        counts=dict(totals),
        canonical_real_rows=coverage.numel(),
        coverage_mismatches=int((coverage != 1).sum()),
        every_canonical_position_exactly_once=bool((coverage == 1).all()),
        effective_targets_and_masks_exact=all(totals[key] == 0 for key in required_zero),
        masked_raw_targets_may_differ=True,
        focus_probes=sorted([p for r in results for p in r['probes']], key=lambda p: p['position']),
    )


def _worker(rank, cases, rendezvous, output):
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    dist.init_process_group(
        'gloo',
        init_method=f'file://{rendezvous}',
        rank=rank,
        world_size=16,
        timeout=timedelta(seconds=120),
    )
    try:
        groups = {}
        for size in (1, 2, 4, 8, 16):
            for begin in range(0, 16, size):
                ranks = list(range(begin, begin + size))
                group = dist.new_group(ranks, backend='gloo', timeout=timedelta(seconds=120))
                if rank in ranks:
                    groups[size] = group
        reports = []
        for case in cases:
            results = {
                layout: _check_layout(rank, case, layout, groups) for layout in ('cp1', 'dcp')
            }
            if rank == 0:
                reports.append(
                    dict(
                        source_step=case['step'],
                        source_sha256=case['source_sha256'],
                        focus_sample=case['focus_sample'],
                        schedules=case['summaries'],
                        identity_matches_gpu_reference=True,
                        layouts=results,
                    )
                )
                report = json.loads(Path(output).read_text())
                report['cases'] = reports
                report['complete'] = len(reports) == len(cases)
                report['passed'] = report['complete'] and all(
                    layout['every_canonical_position_exactly_once']
                    and layout['effective_targets_and_masks_exact']
                    for item in reports
                    for layout in item['layouts'].values()
                )
                Path(output).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
                print(json.dumps({key: report[key] for key in ('complete', 'passed')}), flush=True)
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--hf-model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cold-reference', default='fixed_route_7604828_cp1')
    parser.add_argument('--warm-reference', default='fixed_route_7604828_step52_boundarygc_cp1')
    args = parser.parse_args()
    assert not torch.cuda.is_available() and not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    started = time.monotonic()
    output = args.output.resolve()
    if not output.is_relative_to(args.study.resolve()):
        parser.error('--output must stay inside --study')
    output.parent.mkdir(parents=True, exist_ok=True)
    initial = dict(
        complete=False,
        passed=False,
        cuda_initialized=False,
        cases=[],
        scope='Actual source/THD/CPU zigzag metadata and real Gloo MTP P2P; not TE CUDA kernel validation',
        raw_target_note='CP1 clears logical ends; CP>1 may retain padded raw IDs at zero-loss positions. Effective masks and supervised targets must match.',
    )
    with output.open('x') as stream:
        json.dump(initial, stream)
    try:
        dataset = Qwen35Dataset(args.manifest, args.hf_model, seq_length=40960)
        assert dataset.image_token_id == 248056, 'This diagnostic uses the native Qwen35 image ID'
        cases = []
        for step, focus, reference in ((52, 60, args.warm_reference), (0, 47, args.cold_reference)):
            samples, media = dataset.build_global_batch(step, 64)
            digest = source_fingerprint(samples, media)
            manifest = (
                args.study
                / 'routing-reference'
                / reference
                / 'training'
                / f'step{step:08d}'
                / 'manifest.json'
            )
            assert json.loads(manifest.read_text())['identity']['source_sha256'] == digest
            summaries, assignments = {}, {}
            for name, dynamic, cp, capacity in (('cp1', False, 1, 45056), ('dcp', True, 8, 8192)):
                summaries[name], assignments[name] = _schedule(
                    samples,
                    media,
                    dynamic=dynamic,
                    cp=cp,
                    capacity=capacity,
                    return_assignments=True,
                )
            assignments['cp1'] = apply_fixed_packing(
                samples,
                media,
                assignments['cp1'],
                step=step,
                domain_ranks=tuple(range(16)),
                cp_size=1,
                config=_config(45056),
                replay_path=args.study / 'packing-reference' / reference / 'training',
            )
            _, slices = assign_encoder_media(media, tuple(range(16)))
            cases.append(
                dict(
                    step=step,
                    focus_sample=focus,
                    source_sha256=digest,
                    samples=samples,
                    slices=slices,
                    assignments=assignments,
                    summaries=summaries,
                    oracle=_oracle(samples),
                )
            )
            print(
                f'Prepared real source {step}, {sum(s["original_seq_len"] for s in samples.values())} tokens',
                flush=True,
            )
        rendezvous = output.with_suffix('.gloo-rendezvous')
        if rendezvous.exists():
            raise FileExistsError(rendezvous)
        mp.start_processes(
            _worker,
            args=(cases, str(rendezvous), str(output)),
            nprocs=16,
            start_method='fork',
            join=True,
        )
    except Exception as exc:
        report = json.loads(output.read_text())
        report['error'] = f'{type(exc).__name__}: {exc}'
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        raise
    report = json.loads(output.read_text())
    report['elapsed_seconds'] = time.monotonic() - started
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if not report['passed']:
        raise SystemExit('Canonical target/mask validation failed; report preserved')


if __name__ == '__main__':
    main()
