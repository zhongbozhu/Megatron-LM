# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Audit paired natural-routing source streams and summarize native training.

No expert choices are recorded or replayed. Numerical quality differences are
reported separately from finite-value, progress, and source-identity checks.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


def preflight(args):
    import torch

    from examples.mimo.data.qwen35_native import Qwen35Dataset
    from megatron.core.datasets.data_schedule import DefaultDynamicCPScheduler

    torch.set_num_threads(4)
    manifest = json.loads(args.manifest.read_text())
    identity = {
        record['sample_id']: record['images'][0]['sha256'] for record in manifest['samples']
    }
    result = dict(manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(), seeds={})
    for seed in (1234, 4321):
        dataset = Qwen35Dataset(
            args.manifest, args.hf_model, 40960, seed=seed, heldout_images=512, split_seed=20261002
        )
        scheduler = DefaultDynamicCPScheduler(8192, 8, 2, None)
        train, heldout, rows = [], [], []
        for step in range(50):
            samples, media = dataset.build_global_batch(step, 64)
            lengths = [(sid, item['padded_seq_len']) for sid, item in samples.items()]
            assignments = scheduler.get_groups_and_subsamples(lengths)
            cp_sizes = [
                sum(tuple(ids) == pack for ids in assignment)
                for assignment in assignments
                for pack in dict.fromkeys(tuple(ids) for ids in assignment)
            ]
            ids = [item['source_id'] for item in media]
            train.extend(ids)
            rows.append(dict(step=step, source_ids=ids, lengths=lengths, cp_sizes=cp_sizes))
            if step % 10 == 0:
                print(
                    json.dumps(dict(seed=seed, step=step, cp_sizes=sorted(set(cp_sizes)))),
                    flush=True,
                )
        for step in range(4):
            _, media = dataset.build_global_batch(step, 64, split='heldout')
            heldout.extend(item['source_id'] for item in media)
        train_images, heldout_images = {identity[i] for i in train}, {identity[i] for i in heldout}
        assert len(train_images) == len(train) == 3200, 'Training repeats images within 50 steps'
        assert len(heldout_images) == len(heldout) == 256, 'Heldout evaluation repeats images'
        assert not train_images & heldout_images
        result['seeds'][str(seed)] = dict(
            dataset=dataset.summary,
            train_steps=rows,
            heldout_source_ids=heldout,
            distinct_train_images=len(train_images),
            distinct_heldout_images=len(heldout_images),
            train_heldout_disjoint=True,
        )
    assert (
        result['seeds']['1234']['heldout_source_ids']
        == result['seeds']['4321']['heldout_source_ids']
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(args.output, flush=True)


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def report(args):
    from examples.mimo.fixed_routing_report import _iteration_lines

    runs, pairs = {}, []
    for seed in (1234, 4321):
        for mode in ('cp1', 'dynamic'):
            name = f'natural_seed{seed}_{mode}'
            path = args.suite / 'logs' / f'{name}-metrics' / 'metrics.jsonl'
            records = _records(path)
            assert len(records) == 50 and [r['step'] for r in records] == list(range(50))
            native = _iteration_lines(args.suite / 'logs' / f'{name}.out')
            assert set(native) == set(range(1, 51))
            for metric in records:
                assert 'fixed_routing' not in metric
                assert math.isfinite(metric['loss']) and all(map(math.isfinite, metric['mtp_loss']))
                assert math.isfinite(metric['aux']['loss'])
                assert all(
                    c['all_finite'] for c in metric['gradient_diagnostics']['components'].values()
                )
                iteration = native[metric['step'] + 1]
                assert iteration['skipped_iterations'] == iteration['nan_iterations'] == 0
                assert iteration['grad_norm'] is not None and iteration['grad_norm'] >= 0
            evaluations = {}
            for iteration in (0, 25, 50):
                evaluation = f'{name}_eval{iteration}'
                values = _records(args.suite / 'logs' / f'{evaluation}-metrics' / 'metrics.jsonl')
                assert len(values) == 4 and all(not r['training'] for r in values)
                assert all(math.isfinite(r['loss']) for r in values)
                count = sum(r['supervised_tokens'] for r in values)
                evaluations[str(iteration)] = (
                    sum(r['loss'] * r['supervised_tokens'] for r in values) / count
                )
            sources = [
                json.loads(
                    (args.suite / 'source-audit' / name / f'train-{i:08d}.json').read_text()
                )['source_sha256']
                for i in range(50)
            ]
            eval_hashes = {
                str(step): [
                    json.loads(
                        (
                            args.suite
                            / 'source-audit'
                            / f'{name}_eval{step}'
                            / f'heldout-{i:08d}.json'
                        ).read_text()
                    )['source_sha256']
                    for i in range(4)
                ]
                for step in (0, 25, 50)
            }
            assert eval_hashes['0'] == eval_hashes['25'] == eval_hashes['50']
            runs[name] = dict(
                source_sha256=sources,
                heldout_source_sha256=eval_hashes['0'],
                train_loss=[r['loss'] for r in records],
                mtp_loss=[r['mtp_loss'] for r in records],
                aux_loss=[r['aux']['loss'] for r in records],
                grad_norm=[native[i]['grad_norm'] for i in range(1, 51)],
                heldout_loss=evaluations,
                mechanical_checks_pass=True,
            )
        first, second = runs[f'natural_seed{seed}_cp1'], runs[f'natural_seed{seed}_dynamic']
        assert first['source_sha256'] == second['source_sha256']
        assert first['heldout_source_sha256'] == second['heldout_source_sha256']
        pairs.append(
            dict(
                seed=seed,
                source_identity_exact=True,
                heldout_delta_dynamic_minus_cp1={
                    str(step): second['heldout_loss'][str(step)] - first['heldout_loss'][str(step)]
                    for step in (0, 25, 50)
                },
            )
        )
    args.output.write_text(
        json.dumps(
            dict(
                runs=runs,
                pairs=pairs,
                scope='Two-seed screening; no statistical equivalence threshold imposed',
            ),
            indent=2,
        )
        + '\n'
    )
    print(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    audit = sub.add_parser('preflight')
    audit.add_argument('--manifest', type=Path, required=True)
    audit.add_argument('--hf-model', type=Path, required=True)
    audit.add_argument('--output', type=Path, required=True)
    compare = sub.add_parser('report')
    compare.add_argument('--suite', type=Path, required=True)
    compare.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    (preflight if args.command == 'preflight' else report)(args)


if __name__ == '__main__':
    main()
