# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Report native fixed-routing evidence without defining numerical tolerances.

This stdlib-only reader does not import model code, start CUDA, or modify source
artifacts. Exact ID replay checks and floating-point observations are deliberately
separate. Missing evidence never counts as equality or a successful comparison.
"""

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

_SOURCE_COUNTS = (
    'source_samples',
    'source_images',
    'source_tokens',
    'supervised_tokens',
    'mtp_tokens',
)
_ROUTE_COUNTS = ('source_step', 'source_samples', 'source_tokens', 'routers', 'mtp_routers')
_AUDIT_ERRORS = ('missing', 'unexpected', 'histogram_mismatches', 'histogram_l1')
_ASSIGNMENT_KINDS = (
    'auxiliary_training',
    'dispatch_training',
    'auxiliary_recompute',
    'dispatch_recompute',
)


def _reject_constant(value):
    raise ValueError(f'Non-finite JSON value: {value}')


def _loads(text):
    return json.loads(text, parse_constant=_reject_constant)


def _artifact(path):
    if not path.is_file():
        return dict(path=str(path), exists=False)
    data = path.read_bytes()
    return dict(
        path=str(path), exists=True, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()
    )


def _metrics(path):
    if not path.is_file():
        return {}, ['No completed metrics JSONL is available']
    records, issues = {}, []
    lines = path.read_text().splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = _loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not line.endswith('\n'):
                issues.append('Ignored incomplete final metrics line')
                continue
            raise
        if not record.get('training', False):
            continue
        step = int(record['step'])
        if step in records:
            raise ValueError(f'Duplicate training metrics for source step {step}: {path}')
        records[step] = record
    return records, issues


def _iteration_lines(path):
    observations = {}
    if not path.is_file():
        return observations
    with path.open() as handle:
        for line in handle:
            iteration = re.search(r'\biteration\s+(\d+)\s*/\s*(\d+)\s*\|', line)
            if not iteration:
                continue
            values = dict(iteration=int(iteration[1]), planned_iterations=int(iteration[2]))
            for key, label in (
                ('grad_norm', 'grad norm'),
                ('learning_rate', 'learning rate'),
                ('skipped_iterations', 'number of skipped iterations'),
                ('nan_iterations', 'number of nan iterations'),
                ('lm_loss', 'lm loss'),
                ('mtp_1_loss', 'mtp_1 loss'),
            ):
                match = re.search(r'(?:\||\s)' + re.escape(label) + r':\s*([^|\s]+)', line)
                if match:
                    number = float(match[1])
                    values[key] = number if math.isfinite(number) else None
                    if not math.isfinite(number):
                        values.setdefault('nonfinite_fields', []).append(key)
            observations[values['iteration']] = values
    return observations


def _equal(reference, candidate, keys):
    return {
        key: dict(
            reference=reference.get(key),
            candidate=candidate.get(key),
            equal=(
                key in reference
                and key in candidate
                and reference[key] is not None
                and reference[key] == candidate[key]
            ),
        )
        for key in keys
    }


def summarize_auxiliary(audit):
    """Report observed exact assignments; score sums remain numerical evidence."""
    if not audit:
        return dict(exact_observed_assignments=False, issues=['Auxiliary replay audit is missing'])
    issues = []
    phases = {}
    for name, values in (
        ('statistics_vs_training', audit),
        ('training_vs_recompute', audit.get('recompute', {})),
    ):
        phase = {key: values.get(key) for key in ('pairs', *_AUDIT_ERRORS, 'score_sum_max_abs')}
        phase['error_layers'] = [
            layer
            for layer, record in values.get('layers', {}).items()
            if any(record.get(key, 0) != 0 for key in _AUDIT_ERRORS)
        ]
        phase['exact_histograms'] = values.get('pairs', 0) > 0 and all(
            values.get(key) == 0 for key in _AUDIT_ERRORS
        )
        if not phase['exact_histograms']:
            issues.append(f'{name}: histogram audit failed or was not observed')
        phases[name] = phase
    assignments = audit.get('token_assignments', {})
    if assignments.get('enabled') is not True:
        issues.append('Exact token assignment auditing was not enabled')
    assignment_summaries = {}
    for name in _ASSIGNMENT_KINDS:
        values = assignments.get(name, {})
        exact = values.get('pairs', 0) > 0 and all(
            values.get(key) == 0
            for key in ('mismatched_pairs', 'different_rows', 'shape_mismatches')
        )
        assignment_summaries[name] = {**values, 'exact': exact}
        if not exact:
            issues.append(f'{name}: assignment audit failed or was not observed')
    return dict(
        exact_observed_assignments=not issues,
        issues=issues,
        phases=phases,
        assignments=assignment_summaries,
        global_histogram_mismatches=audit.get('global_histogram_mismatches'),
        global_histogram_l1=audit.get('global_histogram_l1'),
        first_mismatches=audit.get('first_mismatches', []),
        scope='Observed router calls only; a histogram match alone does not establish per-token equality.',
    )


def _vector_metrics(values):
    result = dict(values)
    norm, ref_norm, relative = (result.get(key) for key in ('l2', 'reference_l2', 'relative_l2'))
    if (
        'cosine' not in result
        and all(isinstance(value, (int, float)) for value in (norm, ref_norm, relative))
        and norm > 0
        and ref_norm > 0
    ):
        ratio = norm / ref_norm
        cosine = (ratio * ratio + 1 - relative * relative) / (2 * ratio)
        result['cosine_from_norms'] = max(-1.0, min(1.0, cosine))
        result['norm_ratio'] = ratio
    return result


def _gradient_summary(metrics, reference_name):
    value = metrics.get('full_gradient_validation')
    if not value:
        return dict(available=False)
    return dict(
        available=True,
        kind=value.get('kind'),
        mode=value.get('mode'),
        reference_dir=value.get('reference_dir'),
        reference_matches_requested=Path(value.get('reference_dir', '')).name == reference_name,
        all_finite=value.get('all_finite'),
        global_elements=value.get('global_elements'),
        totals=_vector_metrics(value.get('totals', {})),
        components={
            name: _vector_metrics(record) for name, record in value.get('components', {}).items()
        },
        reported_tolerance_pass=value.get('tolerance_pass'),
        scope='Numerical observations only. Components can overlap; their element counts must not be summed.',
    )


def _optimizer_summary(report, reference_name):
    if report is None:
        return dict(available=False)
    ranks = report.get('ranks', [])
    return dict(
        available=True,
        mode=report.get('mode'),
        expected_step=report.get('expected_step'),
        reference_dir=report.get('reference_dir'),
        reference_matches_requested=Path(report.get('reference_dir', '')).name == reference_name,
        initial_state_exact=report.get('initial_state_exact'),
        full_update_elements=report.get('full_update_elements'),
        positive_lr_owned_elements=report.get('positive_lr_owned_elements'),
        zero_lr_owned_elements=report.get('zero_lr_owned_elements'),
        nonzero_master_update=report.get('nonzero_master_update'),
        observed_gradient_l2=report.get('observed_gradient_l2'),
        metrics=report.get('metrics', {}),
        rank_count=len(ranks),
        native_steps_succeeded=bool(ranks)
        and all(rank.get('native_result', [False])[0] is True for rank in ranks),
        native_grad_norms=sorted(
            {
                rank['native_result'][1]
                for rank in ranks
                if rank.get('native_result')
                and len(rank['native_result']) > 1
                and rank['native_result'][1] is not None
            }
        ),
        reported_tolerance_pass=report.get('tolerance_pass'),
        scope=report.get('scope'),
    )


def _read_run(study, name):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or name in ('.', '..'):
        raise ValueError(f'Run name must be a filename component: {name}')
    paths = dict(
        metrics=study / 'logs' / f'{name}-metrics' / 'metrics.jsonl',
        optimizer=study / 'analysis' / f'{name}-optimizer.json',
        native_log=study / 'logs' / f'{name}.out',
    )
    records, issues = _metrics(paths['metrics'])
    optimizer = _loads(paths['optimizer'].read_text()) if paths['optimizer'].is_file() else None
    return dict(
        name=name,
        metrics=records,
        issues=issues,
        optimizer=optimizer,
        iterations=_iteration_lines(paths['native_log']),
        artifacts={key: _artifact(path) for key, path in paths.items()},
    )


def _loss_differences(reference, candidate):
    result = {}
    for key in ('loss', 'mtp_loss'):
        left, right = reference.get(key), candidate.get(key)
        if left is None or right is None:
            result[key] = None
        elif isinstance(left, list) and isinstance(right, list):
            result[key] = [b - a for a, b in zip(left, right)] if len(left) == len(right) else None
        else:
            result[key] = right - left
    left, right = reference.get('aux', {}).get('loss'), candidate.get('aux', {}).get('loss')
    result['aux_loss'] = right - left if left is not None and right is not None else None
    return result


def build_report(study, reference_name, names):
    study = Path(study).resolve()
    reference = _read_run(study, reference_name)
    runs = {}
    for name in dict.fromkeys((reference_name, *names)):
        run = reference if name == reference_name else _read_run(study, name)
        steps = {}
        for step in sorted(set(reference['metrics']) | set(run['metrics'])):
            ref = reference['metrics'].get(step)
            current = run['metrics'].get(step)
            if ref is None or current is None:
                steps[str(step)] = dict(
                    status='incomplete',
                    reference_metrics_available=ref is not None,
                    candidate_metrics_available=current is not None,
                    exact_route_checks_pass=False,
                )
                continue
            ref_routes, routes = ref.get('fixed_routing', {}), current.get('fixed_routing', {})
            hashes = _equal(ref_routes, routes, ('route_sha256',))
            route_hash_valid = all(
                re.fullmatch(r'[0-9a-f]{64}', value.get('route_sha256', '')) is not None
                for value in (ref_routes, routes)
            )
            source_counts = _equal(ref, current, _SOURCE_COUNTS)
            route_counts = _equal(ref_routes, routes, _ROUTE_COUNTS)
            auxiliary = summarize_auxiliary(current.get('aux_replay'))
            reference_auxiliary = summarize_auxiliary(ref.get('aux_replay'))
            checks = [
                route_hash_valid,
                hashes['route_sha256']['equal'],
                auxiliary['exact_observed_assignments'],
                reference_auxiliary['exact_observed_assignments'],
            ]
            checks.extend(
                value['equal'] for value in (*source_counts.values(), *route_counts.values())
            )
            native = run['iterations'].get(step + 1)
            steps[str(step)] = dict(
                status='metrics_available',
                source_counts=source_counts,
                route_counts=route_counts,
                route_hash=hashes['route_sha256'],
                exact_route_checks_pass=all(checks),
                auxiliary_replay=auxiliary,
                reference_auxiliary_assignments_exact=reference_auxiliary[
                    'exact_observed_assignments'
                ],
                losses={key: current.get(key) for key in ('loss', 'mtp_loss')},
                aux_loss=current.get('aux', {}).get('loss'),
                loss_delta=_loss_differences(ref, current),
                full_gradient=_gradient_summary(current, reference_name),
                encoder_boundary=current.get('encoder_boundary_validation'),
                native_iteration=native,
                native_iteration_observed=native is not None,
                rounds=current.get('rounds'),
                cp_sizes=current.get('cp_sizes'),
                max_memory_allocated_gib=current.get('max_memory_allocated_gib'),
            )
        runs[name] = dict(
            issues=run['issues'],
            artifacts=run['artifacts'],
            steps=steps,
            optimizer=_optimizer_summary(run['optimizer'], reference_name),
        )
    return dict(
        schema=1,
        created_utc=datetime.now(timezone.utc).isoformat(),
        study=str(study),
        reference=reference_name,
        overall_acceptance_claimed=False,
        numerical_tolerance_inferred=False,
        scope='Fixed expert-ID conditioned objective. Exact routing evidence is separate from gradient, loss and optimizer numerical observations; no threshold is introduced or relaxed.',
        runs=runs,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--runs', nargs='*', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    study = args.study.resolve()
    output = (args.output if args.output.is_absolute() else study / args.output).resolve()
    if not output.is_relative_to(study):
        parser.error('--output must stay inside --study')
    report = build_report(study, args.reference, args.runs)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(output)


if __name__ == '__main__':
    main()
