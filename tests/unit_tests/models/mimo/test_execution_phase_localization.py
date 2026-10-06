# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CPU fixtures distinguish forward phase drift from recompute and padding."""

import copy

import pytest
import torch

from examples.mimo.execution_phase_localization import localize_execution_phases


def _captures(directory):
    for rank, indices, keys in (
        (0, [0, 1, 6, 7], [(47, 0), (47, 1), (48, 2)]),
        (1, [2, 3, 4, 5], [(47, 2), (48, 0), (48, 1)]),
    ):
        value = torch.zeros(3, 2)
        full = torch.zeros(4, 1, 2)
        statistics = dict(
            step=0,
            round=2,
            phase='statistics',
            cp_ranks=[0, 1],
            keys=keys,
            local_tokens=4,
            sample_ids=[47, 48],
            sample_lengths=[3, 3],
            padded_boundaries=[0, 4, 8],
            logical_boundaries=[0, 3, 6],
            physical_indices=torch.tensor(indices),
            records={
                'layer00.attention': [dict(value=value, grad_enabled=False)],
                'layer00.input_projection': [dict(value=value.clone(), grad_enabled=False)],
            },
            gdn_layers={'0': dict(input=full, output=full.clone(), recompute_checks=[])},
        )
        training = copy.deepcopy(statistics)
        training['phase'] = 'training'
        first = training['records']['layer00.attention'][0]
        repeated = dict(value=value.clone(), grad_enabled=True, gradient=value.clone())
        training['records']['layer00.attention'].append(repeated)
        check = dict(input_exact=True, input_max_abs=0.0, output_observed=True)
        if rank == 0:
            first['value'][1, 0] = 0.5
            training['gdn_layers']['0']['output'][1, 0, 0] = 0.5
            training['gdn_layers']['0']['output'][3, 0, 0] = 0.25
            check.update(output_exact=False, output_max_abs=0.5, output=full.clone())
        else:
            check.update(output_exact=True, output_max_abs=0.0)
        training['gdn_layers']['0']['recompute_checks'].append(check)
        target = directory / 'step00000000' / f'rank{rank:05d}'
        target.mkdir(parents=True)
        torch.save(statistics, target / 'statistics-round0002.pt')
        torch.save(training, target / 'training-round0002.pt')


def test_phase_owner_and_recompute_are_separate(tmp_path):
    _captures(tmp_path)
    report = localize_execution_phases(tmp_path)
    assert report['overall_acceptance_claimed'] is False
    assert report['coverage'] == dict(
        rank_round_pairs=2,
        unique_probes=6,
        runtime_cp_round_counts={'2': 2},
        statistics_training_changed_probes=1,
        recompute_changed_probes=1,
    )
    owner = report['changed_probe_owners'][0]
    assert (owner['sample_id'], owner['position'], owner['rank'], owner['round']) == (47, 1, 0, 2)
    assert owner['cp_ranks'] == [0, 1]
    attention = report['rounds'][0]['changed_fields']['layer00.attention']
    assert attention['statistics_vs_forward']['changed_keys'] == [[47, 1]]
    assert attention['forward_vs_repeated'][0]['statistics_exact'] is True
    full = report['rounds'][0]['gdn_layers']['0']
    assert full['statistics_vs_forward']['input']['exact'] is True
    output = full['statistics_vs_forward']['output']
    assert output['real_changed_tokens'] == 1
    assert output['padding_changed_tokens'] == 1
    assert output['changed_sample_positions'] == {'47': [1]}
    assert full['forward_vs_repeated'][0]['output']['statistics_exact'] is True


def test_missing_training_pair_fails(tmp_path):
    _captures(tmp_path)
    (tmp_path / 'step00000000/rank00000/training-round0002.pt').unlink()
    with pytest.raises(FileNotFoundError):
        localize_execution_phases(tmp_path)


def test_physical_ownership_mismatch_fails(tmp_path):
    _captures(tmp_path)
    path = tmp_path / 'step00000000/rank00000/training-round0002.pt'
    data = torch.load(path, weights_only=True)
    data['physical_indices'][0] = 2
    torch.save(data, path)
    with pytest.raises(ValueError, match='physical CP ownership differs'):
        localize_execution_phases(tmp_path)
