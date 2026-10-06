# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Summarize paired fixed-routing trajectories without a numerical tolerance.

Uses the single-step report's exact assignment checks and native iteration
parser. Checkpoint checks establish file presence only, not resume correctness.
Finite losses and exact IDs do not establish convergence or gradient equivalence.
"""

import argparse
import json
import math
import re
from pathlib import Path

from examples.mimo.fixed_routing_report import _artifact, _loads, _metrics, build_report


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _source_hash(record):
    value = record.get('fixed_routing', {}).get('source_sha256')
    return value if isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) else None


def _checkpoint_summary(study, name, steps, interval):
    directory = study / 'checkpoints' / name
    expected = list(range(interval, steps + 1, interval))
    if not expected or expected[-1] != steps:
        expected.append(steps)
    tracker = directory / 'latest_checkpointed_iteration.txt'
    text = tracker.read_text().strip() if tracker.is_file() else ''
    latest = int(text) if text.isdigit() else None
    checkpoints = []
    for iteration in expected:
        path = directory / f'iter_{iteration:07d}'
        metadata = _artifact(path / '.metadata')
        sizes = [shard.stat().st_size for shard in path.glob('*.distcp') if shard.is_file()]
        checkpoints.append(
            dict(
                iteration=iteration,
                metadata=metadata,
                shard_files=len(sizes),
                shard_bytes=sum(sizes),
                files_present=bool(metadata.get('bytes')) and bool(sizes) and min(sizes) > 0,
            )
        )
    return dict(
        expected_iterations=expected,
        latest_tracker_iteration=latest,
        checkpoints=checkpoints,
        expected_files_present=latest == steps
        and all(entry['files_present'] for entry in checkpoints),
        checkpoint_load_or_resume_tested=False,
    )


def _evaluation_records(path):
    records = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = _loads(line)
            if record.get('training') is not False:
                continue
            step = int(record['step'])
            if step in records:
                raise ValueError(f'Duplicate evaluation source step {step}: {path}')
            records[step] = record
    return records


def _evaluations(reference, candidate, expected_count):
    rows = []
    for step in range(expected_count):
        ref, current = reference.get(step, {}), candidate.get(step, {})
        ref_hash, current_hash = _source_hash(ref), _source_hash(current)
        ref_route = ref.get('fixed_routing', {}).get('route_sha256')
        current_route = current.get('fixed_routing', {}).get('route_sha256')
        valid_route = isinstance(ref_route, str) and re.fullmatch('[0-9a-f]{64}', ref_route)
        rows.append(
            dict(
                source_step=step,
                loss=current.get('loss'),
                reference_loss=ref.get('loss'),
                finite_losses=_finite(ref.get('loss')) and _finite(current.get('loss')),
                source_hash_exact=ref_hash is not None and ref_hash == current_hash,
                route_hash_exact=bool(valid_route) and ref_route == current_route,
            )
        )
    return dict(
        expected_count=expected_count,
        observed_count=len(candidate),
        records=rows,
        complete=set(candidate) == set(range(expected_count)),
        all_finite=all(row['finite_losses'] for row in rows),
        all_source_and_route_hashes_exact=all(
            row['source_hash_exact'] and row['route_hash_exact'] for row in rows
        ),
        scope=(
            'Compare layouts at the same evaluation event. Sample composition can change '
            'between events; first/last loss is not a fixed-batch convergence measure. '
            'Evaluation records LM loss, source and route identity, without MTP loss or '
            'backward/recompute audits.'
        ),
    )


def build_training_report(study, reference_name, names, *, steps=50, checkpoint_interval=25):
    """Check every expected source step; missing observations never pass."""
    if steps < 1 or checkpoint_interval < 1:
        raise ValueError('steps and checkpoint_interval must be positive')
    study = Path(study).resolve()
    report = build_report(study, reference_name, names)
    reference_path = study / 'logs' / f'{reference_name}-metrics' / 'metrics.jsonl'
    reference, _ = _metrics(reference_path)
    reference_evaluations = _evaluation_records(reference_path)
    expected = set(range(steps))
    for name, run in report['runs'].items():
        path = study / 'logs' / f'{name}-metrics' / 'metrics.jsonl'
        raw, _ = _metrics(path)
        rows = []
        for step in sorted(expected):
            metric = raw.get(step, {})
            single = run['steps'].get(str(step), {})
            native = single.get('native_iteration') or {}
            mtp = metric.get('mtp_loss')
            finite = dict(
                lm_loss=_finite(metric.get('loss')),
                mtp_loss=isinstance(mtp, list) and bool(mtp) and all(_finite(v) for v in mtp),
                aux_loss=_finite(metric.get('aux', {}).get('loss')),
                grad_norm=_finite(native.get('grad_norm')) and native['grad_norm'] >= 0,
            )
            ref_hash, current_hash = _source_hash(reference.get(step, {})), _source_hash(metric)
            native_success = (
                native.get('iteration') == step + 1
                and native.get('planned_iterations') == steps
                and native.get('skipped_iterations') == 0
                and native.get('nan_iterations') == 0
                and not native.get('nonfinite_fields')
                and finite['grad_norm']
            )
            rows.append(
                dict(
                    source_step=step,
                    lm_loss=metric.get('loss'),
                    mtp_loss=mtp,
                    aux_loss=metric.get('aux', {}).get('loss'),
                    grad_norm=native.get('grad_norm'),
                    loss_delta=single.get('loss_delta'),
                    finite=finite,
                    native_success=native_success,
                    route_audits_exact=single.get('exact_route_checks_pass') is True,
                    reference_source_sha256=ref_hash,
                    candidate_source_sha256=current_hash,
                    source_hash_exact=ref_hash is not None and ref_hash == current_hash,
                )
            )
        checkpoints = _checkpoint_summary(study, name, steps, checkpoint_interval)
        evaluations = _evaluations(
            reference_evaluations, _evaluation_records(path), steps // checkpoint_interval + 1
        )
        checks = dict(
            exact_step_coverage=set(raw) == expected,
            at_least_50_steps=len(raw) >= 50,
            all_losses_and_norms_finite=all(all(row['finite'].values()) for row in rows),
            all_native_iterations_successful=all(row['native_success'] for row in rows),
            all_route_audits_exact=all(row['route_audits_exact'] for row in rows),
            all_source_hashes_exact=all(row['source_hash_exact'] for row in rows),
            checkpoint_files_present=checkpoints['expected_files_present'],
            evaluations_complete=evaluations['complete'],
            evaluation_losses_finite=evaluations['all_finite'],
            evaluation_source_and_routes_exact=evaluations['all_source_and_route_hashes_exact'],
        )
        run['trajectory'] = dict(
            expected_steps=steps,
            observed_steps=len(raw),
            successful_native_iterations=sum(row['native_success'] for row in rows),
            missing_steps=sorted(expected - set(raw)),
            unexpected_steps=sorted(set(raw) - expected),
            checks=checks,
            mechanical_checks_pass=all(checks.values()),
            records=rows,
            checkpoints=checkpoints,
            evaluation=evaluations,
        )
    report['scope'] = (
        'Paired fixed expert-ID trajectories. Finite losses, exact route/source identity, native '
        'iteration completion and checkpoint presence are mechanical checks only. Numerical '
        'equivalence, convergence tolerance and checkpoint resume are not inferred.'
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--checkpoint-interval', type=int, default=25)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    study = args.study.resolve()
    output = (args.output if args.output.is_absolute() else study / args.output).resolve()
    if not output.is_relative_to(study):
        parser.error('--output must stay inside --study')
    report = build_training_report(
        study,
        args.reference,
        args.runs,
        steps=args.steps,
        checkpoint_interval=args.checkpoint_interval,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(output)
    if not all(run['trajectory']['mechanical_checks_pass'] for run in report['runs'].values()):
        raise SystemExit('Training mechanical checks failed; the detailed report was preserved')


if __name__ == '__main__':
    main()
