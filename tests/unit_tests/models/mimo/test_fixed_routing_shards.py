# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CPU-only checks of canonical ID assembly and synchronized failure handling."""

from datetime import timedelta
from pathlib import Path

import pytest
import torch

from examples.mimo.fixed_routing_shards import _coverage, _describe, assemble_canonical_records


def _tables(positions):
    result = {}
    for offset, name in enumerate(('decoder.router', 'mtp.router')):
        result[name] = {}
        for sid, values in positions.items():
            first = (values + sid + offset) % 8
            result[name][sid] = torch.stack((first, (first + 3) % 8), dim=-1).to(torch.int16)
    return result


def test_singleton_restores_canonical_order_and_preserves_schema(tmp_path):
    positions = {0: torch.tensor([4, 5, 0, 1, 2, 3]), 1: torch.tensor([2, 0, 1])}
    result = assemble_canonical_records(
        _tables(positions), positions, {0: 6, 1: 3}, None, tmp_path / 'parts'
    )
    expected = _tables({0: torch.arange(6), 1: torch.arange(3)})
    for name, samples in expected.items():
        for sid, value in samples.items():
            assert torch.equal(result[name][sid], value)
    assert (tmp_path / 'parts/rank00000.pt').is_file()
    assert (tmp_path / 'parts/rank00000.json').is_file()


def test_compact_metadata_and_different_sample_writers():
    lengths = {0: 8, 1: 2}
    positions = [
        {0: torch.tensor([0, 1, 6, 7])},
        {0: torch.tensor([2, 3, 4, 5]), 1: torch.arange(2)},
    ]
    metadata = [
        _describe(_tables(part), part, lengths, rank) for rank, part in enumerate(positions)
    ]
    owners, parts, specs = _coverage(metadata)
    assert owners == {0: 0, 1: 1}
    assert parts[0] == [(0, 2, 0, 0), (2, 6, 1, 0), (6, 8, 0, 2)]
    assert specs == {'decoder.router': ('torch.int16', 2), 'mtp.router': ('torch.int16', 2)}
    assert 'records' not in metadata[0]


def test_existing_artifacts_cannot_be_overwritten(tmp_path):
    positions = {0: torch.arange(2)}
    directory = tmp_path / 'parts'
    assemble_canonical_records(_tables(positions), positions, {0: 2}, None, directory)
    with pytest.raises(RuntimeError, match='FileExistsError'):
        assemble_canonical_records(_tables(positions), positions, {0: 2}, None, directory)


@pytest.mark.parametrize(
    'failure',
    [
        'duplicate_position',
        'position_range',
        'missing_router_sample',
        'row_count',
        'duplicate_expert',
        'negative_expert',
        'float_ids',
    ],
)
def test_invalid_rank_local_parts_fail_before_publication(tmp_path, failure):
    positions = {0: torch.arange(4)}
    records = _tables(positions)
    if failure == 'duplicate_position':
        positions[0][1] = 0
    elif failure == 'position_range':
        positions[0][0] = 4
    elif failure == 'missing_router_sample':
        records['mtp.router'].clear()
    elif failure == 'row_count':
        records['decoder.router'][0] = records['decoder.router'][0][:2]
    elif failure == 'duplicate_expert':
        records['decoder.router'][0][:, 1] = records['decoder.router'][0][:, 0]
    elif failure == 'negative_expert':
        records['decoder.router'][0][0, 0] = -1
    else:
        records['decoder.router'][0] = records['decoder.router'][0].float()
    directory = tmp_path / 'parts'
    with pytest.raises(RuntimeError, match='assembly failed'):
        assemble_canonical_records(records, positions, {0: 4}, None, directory)
    assert not directory.exists()


@pytest.mark.parametrize('failure', ['gap', 'overlap', 'router_set', 'topk', 'lengths'])
def test_global_coverage_rejects_malformed_partitions(failure):
    lengths = {0: 4}
    positions = [{0: torch.tensor([0, 1])}, {0: torch.tensor([2, 3])}]
    if failure == 'gap':
        positions[1][0] = torch.tensor([3])
    elif failure == 'overlap':
        positions[1][0] = torch.tensor([1, 2, 3])
    metadata = [
        _describe(_tables(part), part, lengths, rank) for rank, part in enumerate(positions)
    ]
    if failure == 'router_set':
        del metadata[1]['shapes']['mtp.router']
    elif failure == 'topk':
        metadata[1]['shapes']['mtp.router'] = ('torch.int16', 3)
    elif failure == 'lengths':
        metadata[1]['lengths'] = {0: 5}
    with pytest.raises(ValueError):
        _coverage(metadata)


def _worker(rank, rendezvous, directory, failure):
    torch.distributed.init_process_group(
        'gloo',
        init_method=f'file://{rendezvous}',
        rank=rank,
        world_size=3,
        timeout=timedelta(seconds=30),
    )
    try:
        lengths = {0: 8, 1: 3}
        positions = [
            {0: torch.tensor([0, 1, 6, 7])},
            {0: torch.tensor([2, 3, 4, 5]), 1: torch.arange(3)},
            {},
        ][rank]
        records = _tables(positions)
        if failure == 'local' and rank == 1:
            records['mtp.router'].clear()
        if failure == 'overlap' and rank == 1:
            positions[0] = torch.tensor([1, 2, 3, 4, 5])
            records = _tables(positions)
        if failure:
            with pytest.raises(RuntimeError, match='assembly failed'):
                assemble_canonical_records(
                    records, positions, lengths, torch.distributed.group.WORLD, directory
                )
            assert not Path(directory).exists()
            return
        result = assemble_canonical_records(
            records, positions, lengths, torch.distributed.group.WORLD, directory
        )
        expected_samples = [0] if rank == 0 else [1] if rank == 1 else []
        expected = _tables({sid: torch.arange(lengths[sid]) for sid in expected_samples})
        assert set(result) == set(expected)
        for name in result:
            assert set(result[name]) == set(expected[name])
            for sid, ids in result[name].items():
                assert torch.equal(ids, expected[name][sid])
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize('failure', ['', 'local', 'overlap'])
def test_three_rank_shared_files_with_empty_owner_and_failures(tmp_path, failure):
    assert not torch.cuda.is_initialized()
    torch.multiprocessing.start_processes(
        _worker,
        args=(str(tmp_path / 'rendezvous'), str(tmp_path / 'parts'), failure),
        nprocs=3,
        join=True,
        start_method='fork',
    )
