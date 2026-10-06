# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Read-only localization of phase/recompute differences in decoder captures.

Unlike ``compare_execution_directories``, this does not aggregate away rank,
round, runtime CP membership, or invocation identity. It compares observations
exactly, without assigning a numerical tolerance or diagnosing their cause.
"""

import argparse
import bisect
import json
from collections import Counter
from pathlib import Path

import torch


def locate_sample_packs(directory, sample_id, step=0):
    """Read pack metadata for one logical sample without loading activation pages."""
    matches = []
    paths = sorted((Path(directory) / f'step{step:08d}').glob('rank*/statistics-round*.pt'))
    for path in paths:
        data = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        if sample_id in data['sample_ids']:
            matches.append(
                dict(
                    path=str(path),
                    rank=int(path.parent.name.removeprefix('rank')),
                    round=data['round'],
                    cp_ranks=data['cp_ranks'],
                    sample_ids=data['sample_ids'],
                    sample_lengths=data['sample_lengths'],
                    padded_boundaries=data['padded_boundaries'],
                    local_tokens=data['local_tokens'],
                )
            )
    if not matches:
        raise ValueError(f'Sample {sample_id} not captured in {directory}, source step {step}')
    return matches


def _changed_rows(expected, actual):
    """Compare CPU tensor rows with bounded temporary memory, including bool maps."""
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        raise ValueError(f'Capture tensor schema differs: {expected.shape}/{actual.shape}')
    expected = expected.reshape(-1, expected.shape[-1])
    actual = actual.reshape(-1, actual.shape[-1])
    changed, maximum, nonfinite = [], 0.0, 0
    for start in range(0, expected.shape[0], 256):
        a, b = expected[start : start + 256], actual[start : start + 256]
        rows = (a != b).any(dim=1).nonzero().flatten()
        if not rows.numel():
            continue
        changed.extend((rows + start).tolist())
        delta = (a.index_select(0, rows).double() - b.index_select(0, rows).double()).abs()
        finite = torch.isfinite(delta)
        nonfinite += int((~finite).sum())
        if finite.any():
            maximum = max(maximum, float(delta[finite].max()))
    return dict(
        rows=changed, exact=not changed, max_abs_error=maximum, nonfinite_differences=nonfinite
    )


def _compare_probes(expected, actual, keys):
    result = _changed_rows(expected, actual)
    if expected.shape[0] != len(keys):
        raise ValueError('Probe row count differs from logical keys')
    result['changed_keys'] = [list(keys[row]) for row in result.pop('rows')]
    result['changed_tokens'] = len(result['changed_keys'])
    return result


def _full_locations(expected, actual, data):
    """Map complete local GDN tensors through recorded physical CP ownership."""
    result = _changed_rows(expected, actual)
    indices = data['physical_indices'].flatten().tolist()
    if expected.numel() // expected.shape[-1] != len(indices):
        raise ValueError('Full GDN row count differs from physical CP indices')
    boundaries = data['padded_boundaries']
    grouped, padding = {}, 0
    for row in result.pop('rows'):
        physical = indices[row]
        slot = bisect.bisect_right(boundaries, physical) - 1
        if not 0 <= slot < len(data['sample_ids']):
            raise ValueError(f'Physical token outside packed samples: {physical}')
        position = physical - boundaries[slot]
        if position >= data['sample_lengths'][slot]:
            padding += 1
            continue
        grouped.setdefault(str(data['sample_ids'][slot]), []).append(position)
    result['real_changed_tokens'] = sum(len(positions) for positions in grouped.values())
    result['padding_changed_tokens'] = padding
    result['changed_sample_positions'] = {
        sid: sorted(positions) for sid, positions in grouped.items()
    }
    return result


def _validate_pair(statistics, training, rank, step):
    for field in (
        'step',
        'round',
        'cp_ranks',
        'keys',
        'local_tokens',
        'sample_ids',
        'sample_lengths',
        'padded_boundaries',
        'logical_boundaries',
    ):
        if statistics[field] != training[field]:
            raise ValueError(f'Statistics/training metadata differs: rank {rank}, {field}')
    if statistics['step'] != step or rank not in statistics['cp_ranks']:
        raise ValueError('Capture step/rank disagrees with file ownership')
    if statistics['phase'] != 'statistics' or training['phase'] != 'training':
        raise ValueError('Capture phase disagrees with filename')
    if not torch.equal(statistics['physical_indices'], training['physical_indices']):
        raise ValueError('Statistics/training physical CP ownership differs')
    if statistics['records'].keys() != training['records'].keys():
        raise ValueError('Statistics/training hook coverage differs')


def localize_execution_phases(directory, step=0):
    """Pair captures from the same run; distinguish original and later invocations.

    Later invocations are reported as recompute/repeated calls, not assumed to be
    backward by index alone. ``grad_enabled`` and gradient presence are retained.
    Only GDN layers explicitly captured in full have exhaustive token coverage.
    """
    directory = Path(directory) / f'step{step:08d}'
    paths = sorted(directory.glob('rank*/statistics-round*.pt'))
    training_paths = set(directory.glob('rank*/training-round*.pt'))
    if not paths:
        raise FileNotFoundError(f'No statistics captures in {directory}')
    rounds, owners, fields = [], {}, {}
    for path in paths:
        training_path = path.with_name(path.name.replace('statistics-', 'training-', 1))
        if training_path not in training_paths:
            raise FileNotFoundError(training_path)
        training_paths.remove(training_path)
        statistics = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        training = torch.load(training_path, map_location='cpu', weights_only=True, mmap=True)
        rank = int(path.parent.name.removeprefix('rank'))
        _validate_pair(statistics, training, rank, step)
        keys = [tuple(key) for key in training['keys']]
        owner = dict(rank=rank, round=training['round'], cp_ranks=training['cp_ranks'])
        for key in keys:
            if key in owners:
                raise ValueError(f'Duplicate logical probe owner: {key}')
            owners[key] = dict(owner, statistics_training_fields=[], recompute_fields=[])
        result = dict(
            **owner,
            cp_size=len(training['cp_ranks']),
            sample_ids=training['sample_ids'],
            sample_lengths=training['sample_lengths'],
            local_tokens=training['local_tokens'],
            probes=len(keys),
            changed_fields={},
            invocation_patterns={},
            gdn_layers={},
        )
        for name, actual in training['records'].items():
            expected = statistics['records'][name]
            if len(expected) != 1 or not actual:
                raise ValueError(f'Unexpected original forward coverage at {path}/{name}')
            pattern = json.dumps(
                dict(
                    statistics_grad_enabled=expected[0]['grad_enabled'],
                    training=[
                        dict(grad_enabled=entry['grad_enabled'], gradient='gradient' in entry)
                        for entry in actual
                    ],
                ),
                sort_keys=True,
            )
            result['invocation_patterns'].setdefault(pattern, []).append(name)
            first = _compare_probes(expected[0]['value'], actual[0]['value'], keys)
            repeats = []
            for invocation, entry in enumerate(actual[1:], 1):
                comparison = _compare_probes(actual[0]['value'], entry['value'], keys)
                comparison['invocation'] = invocation
                comparison['statistics_exact'] = torch.equal(expected[0]['value'], entry['value'])
                repeats.append(comparison)
            aggregate = fields.setdefault(
                name,
                dict(
                    probes=0,
                    statistics_training_changed_keys=set(),
                    recompute_changed_keys=set(),
                    repeated_invocations=0,
                    statistics_training_max_abs=0.0,
                    recompute_max_abs=0.0,
                ),
            )
            aggregate['probes'] += len(keys)
            aggregate['repeated_invocations'] += len(repeats)
            aggregate['statistics_training_max_abs'] = max(
                aggregate['statistics_training_max_abs'], first['max_abs_error']
            )
            for key in first['changed_keys']:
                key = tuple(key)
                aggregate['statistics_training_changed_keys'].add(key)
                owners[key]['statistics_training_fields'].append(name)
            for repeat in repeats:
                aggregate['recompute_max_abs'] = max(
                    aggregate['recompute_max_abs'], repeat['max_abs_error']
                )
                for key in repeat['changed_keys']:
                    key = tuple(key)
                    aggregate['recompute_changed_keys'].add(key)
                    if name not in owners[key]['recompute_fields']:
                        owners[key]['recompute_fields'].append(name)
            if not first['exact'] or any(not entry['exact'] for entry in repeats):
                result['changed_fields'][name] = dict(
                    statistics_vs_forward=first, forward_vs_repeated=repeats
                )
        for layer, actual in training.get('gdn_layers', {}).items():
            expected = statistics['gdn_layers'][layer]
            layer_result = dict(statistics_vs_forward={}, forward_vs_repeated=[])
            for field in ('input', 'output'):
                layer_result['statistics_vs_forward'][field] = _full_locations(
                    expected[field], actual[field], training
                )
            for invocation, check in enumerate(actual.get('recompute_checks', []), 1):
                repeat = dict(invocation=invocation, output_observed=check['output_observed'])
                for field in ('input', 'output'):
                    if field == 'output' and not check['output_observed']:
                        continue
                    repeated_value = check.get(field, actual[field])
                    repeat[field] = _full_locations(actual[field], repeated_value, training)
                    repeat[field]['statistics_exact'] = torch.equal(expected[field], repeated_value)
                    if repeat[field]['exact'] != check[field + '_exact']:
                        raise ValueError(
                            f'GDN recompute stored tensor/check mismatch: {path}/{layer}'
                        )
                layer_result['forward_vs_repeated'].append(repeat)
            result['gdn_layers'][layer] = layer_result
        rounds.append(result)
    if training_paths:
        raise ValueError(f'Unpaired training captures: {sorted(training_paths)}')
    for value in fields.values():
        for phase in ('statistics_training', 'recompute'):
            key = phase + '_changed_keys'
            value[key] = [list(item) for item in sorted(value[key])]
            value[phase + '_changed_tokens'] = len(value[key])
    changed_owners = [
        dict(sample_id=key[0], position=key[1], **owner)
        for key, owner in sorted(owners.items())
        if owner['statistics_training_fields'] or owner['recompute_fields']
    ]
    return dict(
        kind='decoder_phase_localization',
        directory=str(directory),
        source_step=step,
        overall_acceptance_claimed=False,
        coverage=dict(
            rank_round_pairs=len(rounds),
            unique_probes=len(owners),
            runtime_cp_round_counts=dict(Counter(str(item['cp_size']) for item in rounds)),
            statistics_training_changed_probes=sum(
                bool(owner['statistics_training_fields']) for owner in owners.values()
            ),
            recompute_changed_probes=sum(
                bool(owner['recompute_fields']) for owner in owners.values()
            ),
        ),
        fields=fields,
        changed_probe_owners=changed_owners,
        rounds=rounds,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--directory', type=Path, required=True, help='Run decoder capture directory'
    )
    parser.add_argument('--step', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = localize_execution_phases(args.directory, args.step)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(report['coverage'], indent=2))


if __name__ == '__main__':
    main()
