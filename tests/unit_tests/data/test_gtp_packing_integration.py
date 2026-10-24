# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Real-group transport and failure tests; runnable directly under torchrun."""

import faulthandler
import os
import sys
import traceback
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.datasets.data_schedule import (
    DpBalancedScheduler,
    _joint_cp_group_members,
    wrap_data_iterator,
)
from megatron.core.process_groups_config import ProcessGroupCollection


def test_joint_cp_group_members_preserve_noncontiguous_lanes():
    groups = _joint_cp_group_members([0, 1, 2, 3], [[0, 2], [1, 3], [0, 2], [1, 3]], 1, 1, True)
    assert set(groups) == {(0,), (2,), (0, 2), (1,), (3,), (1, 3)}
    assert (0, 1) not in groups
    assert set(_joint_cp_group_members([0, 1, 2, 3], [[0, 2], [1, 3]], 1, 1, False)) == {
        (0,),
        (1,),
        (2,),
        (3,),
    }


def test_static_alignment_uses_first_donor_pack():
    scheduler = DpBalancedScheduler(8, 1, 3, None)
    result = scheduler.get_groups_and_subsamples([(0, 4), (1, 4), (2, 8)])
    assert len(result) == 1
    assert sorted(sample_id for pack in result[0] for sample_id in pack) == [0, 1, 2]
    assert all(result[0])


def _initialize(tp, pp, cp, dynamic):
    if not dist.is_initialized():
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        dist.init_process_group(
            'nccl',
            timeout=timedelta(seconds=120),
            device_id=torch.device('cuda', torch.cuda.current_device()),
        )
    if dist.get_world_size() % (tp * pp * cp * 2):
        pytest.skip("Topology does not fit this torchrun world")
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        context_parallel_size=cp,
        gtp_remat_size=2,
        dynamic_context_parallel=dynamic,
        create_gloo_process_groups=False,
        distributed_timeout_minutes=2,
    )
    groups = ProcessGroupCollection.use_mpu_process_groups()
    config = SimpleNamespace(
        max_seqlen_per_dp_cp_rank=64,
        microbatch_group_size_per_vp_stage=pp,
        virtual_pipeline_model_parallel_size=None,
        sequence_packing_scheduler='default_dynamic_cp' if dynamic else 'dp_balanced',
        min_dynamic_context_parallel_size=1,
        pipeline_model_parallel_layout=None,
        mtp_num_layers=None,
        gtp_weight_remat_size=2,
        gtp_remat_num_sequence_shards=1,
        sequence_parallel=False,
    )
    return groups, config


def _sample(sample_id, length):
    dev = torch.cuda.current_device()
    return {
        'tokens': torch.full((length,), sample_id, dtype=torch.int64, device=dev),
        'labels': torch.full((length,), sample_id + 1, dtype=torch.int64, device=dev),
        'loss_mask': torch.ones(length, dtype=torch.float32, device=dev),
        'position_ids': torch.arange(length, dtype=torch.int64, device=dev),
        'original_seq_len': torch.tensor([length], dtype=torch.int64, device=dev),
        'padded_seq_len': torch.tensor([length], dtype=torch.int64, device=dev),
    }


def _finish():
    if sys.exc_info()[0] is not None:
        # Expose the original failed assertion immediately. A finally barrier
        # would hide it while peers wait in a later test collective.
        traceback.print_exc()
        return
    torch.cuda.synchronize()
    dist.barrier()
    parallel_state.destroy_model_parallel()


@pytest.mark.parametrize('tp,pp,cp', [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2)])
@pytest.mark.parametrize('dynamic', [False, True])
def test_global_gtp_transport(tp, pp, cp, dynamic):
    groups, config = _initialize(tp, pp, cp, dynamic)
    try:
        owner = groups.dp_gtp_remat.rank()
        # Opposite GTP coordinates have strongly asymmetric workloads.
        long_length = min(128, 64 * groups.dp_cp.size()) if dynamic else 64
        length = 16 if groups.gtp_remat.rank() == 0 else long_length
        samples = [_sample(1000 + owner * 8 + i, length) for i in range(8)]
        iterator = iter(samples) if groups.tp.rank() == 0 else None
        new_iterator, rounds, total, total_squared = wrap_data_iterator(
            iterator, config, 8, pg_collection=groups
        )
        records = []
        if groups.tp.rank() == 0:
            for _ in range(rounds):
                microbatch = next(new_iterator)
                values = microbatch.get('tokens')
                if values is None:
                    values = microbatch['labels'] - 1
                bounds = microbatch['cu_seqlens_padded'].tolist()
                ids = []
                for start, stop in zip(bounds, bounds[1:]):
                    sample_id = int(values[start].item())
                    assert torch.all(values[start:stop] == sample_id)
                    ids.append(sample_id)
                runtime_cp = int(microbatch['local_cp_size']) if dynamic else cp
                cp_group = (
                    parallel_state.get_dynamic_data_context_parallel_groups(group_size=runtime_cp)
                    if dynamic
                    else groups.cp
                )
                records.append((ids, dist.get_process_group_ranks(cp_group)))
            assert len(records) == rounds
            all_records = [None] * groups.dp_cp_gtp_remat.size()
            dist.all_gather_object(all_records, records, group=groups.dp_cp_gtp_remat)
            full_ranks = dist.get_process_group_ranks(groups.dp_cp_gtp_remat)
            occurrences = {}
            for slot, rank_records in enumerate(all_records):
                assert len(rank_records) == rounds
                for step, (ids, members) in enumerate(rank_records):
                    for sample_id in ids:
                        occurrences.setdefault(sample_id, []).append(
                            (step, full_ranks[slot], members)
                        )
            assert set(occurrences) == set(range(1000, 1000 + groups.dp_gtp_remat.size() * 8))
            for placements in occurrences.values():
                assert len({step for step, _, _ in placements}) == 1
                actual_ranks = sorted(rank for _, rank, _ in placements)
                assert all(sorted(members) == actual_ranks for _, _, members in placements)
        expected = groups.dp_gtp_remat.size() // 2 * 8
        assert total == expected * (16 + long_length)
        assert total_squared == expected * (16**2 + long_length**2)
    finally:
        _finish()


@pytest.mark.parametrize('tp,pp', [(1, 1), (2, 1), (1, 2)])
@pytest.mark.parametrize('failure', ['fetch', 'schema', 'wrong_dtype', 'oversized', 'exhaustion'])
def test_gtp_packing_failure_reaches_every_rank(tp, pp, failure):
    groups, config = _initialize(tp, pp, 1, True)
    try:
        samples = [_sample(1000 + groups.dp_gtp_remat.rank() * 8 + i, 16) for i in range(8)]
        if failure == 'wrong_dtype':
            # Consistently wrong dtypes must fail before TP batch broadcasting,
            # even when every sample owner agrees with every other owner.
            for sample in samples:
                sample['tokens'] = sample['tokens'].to(torch.int32)
        if dist.get_rank() == 0:
            if failure == 'schema':
                samples[0]['tokens'] = samples[0]['tokens'][:1]
            elif failure == 'oversized':
                samples[0] = _sample(1000, 64 * groups.dp_cp.size() + 16)
            elif failure == 'exhaustion':
                samples = []

        def source():
            if dist.get_rank() == 0 and failure == 'fetch':
                raise ValueError("injected local fetch failure")
            yield from samples

        with pytest.raises(RuntimeError):
            wrap_data_iterator(
                source() if groups.tp.rank() == 0 else None, config, 8, pg_collection=groups
            )
        # A successful world collective verifies that peers returned together.
        dist.barrier()
    finally:
        _finish()


def test_gtp_evaluation_retains_partial_global_tail():
    groups, config = _initialize(1, 1, 1, True)
    try:
        owner = groups.dp_gtp_remat.rank()
        count = groups.dp_gtp_remat.size()
        iterator = iter([] if owner == count - 1 else [_sample(1000 + owner, 16)])
        packed, rounds, total, squared = wrap_data_iterator(
            iterator, config, 8, pg_collection=groups, allow_partial_batch=True
        )
        assert total == 16 * (count - 1)
        assert squared == 16**2 * (count - 1)
        for _ in range(rounds):
            next(packed)
        with pytest.raises(StopIteration):
            wrap_data_iterator(iterator, config, 8, pg_collection=groups, allow_partial_batch=True)
        dist.barrier()
    finally:
        _finish()


def _run_case(function, *args):
    label = f"{function.__name__}{args}"
    print(f"rank={os.environ.get('RANK', '?')} START {label}", flush=True)
    function(*args)
    print(f"rank={os.environ.get('RANK', '?')} DONE {label}", flush=True)


if __name__ == '__main__':
    faulthandler.dump_traceback_later(60, repeat=True)
    test_joint_cp_group_members_preserve_noncontiguous_lanes()
    test_static_alignment_uses_first_donor_pack()
    for dynamic in (False, True):
        for tp, pp, cp in ((1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2)):
            _run_case(test_global_gtp_transport, tp, pp, cp, dynamic)
    for failure in ('fetch', 'schema', 'wrong_dtype', 'oversized', 'exhaustion'):
        for tp, pp in ((1, 1), (2, 1), (1, 2)):
            _run_case(test_gtp_packing_failure_reaches_every_rank, tp, pp, failure)
    _run_case(test_gtp_evaluation_retains_partial_global_tail)
    if dist.get_rank() == 0:
        print('GTP_GLOBAL_PACKING_INTEGRATION_PASS', flush=True)
    dist.destroy_process_group()
    faulthandler.cancel_dump_traceback_later()
