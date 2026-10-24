# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Public preparation/runtime contracts with real scheduling and feature transport.

Run with four or eight CUDA ranks. A small differentiable objective checks graph
lifetime and feature-gradient sums without model-specific processors or trainers.
"""

import os

import pytest
import torch
import torch.distributed as dist

from megatron.core.datasets.data_schedule import DefaultDynamicCPScheduler, DpBalancedScheduler
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.models.mimo.packing import prepare_mimo_step
from megatron.core.models.mimo.partition.utils import PartitionAdapter, PartitionConfig
from megatron.core.models.mimo.planning import MediaTask, assign_media_to_producers
from megatron.core.models.mimo.runtime import MimoStepRuntime

_MEDIA_TOKEN = 511


@pytest.fixture(scope="module", params=(False, True), ids=("colocated", "disaggregated"))
def runtime_groups(request):
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    width = 8 if request.param else 4
    if dist.get_world_size() % width:
        pytest.skip(f"This MIMO runtime case requires a multiple of {width} ranks")
    owned, created = {}, []
    rank = dist.get_rank()
    for base in range(0, dist.get_world_size(), width):
        all_ranks = tuple(range(base, base + width))
        decoder_ranks = all_ranks[-4:]
        producer_ranks = all_ranks[:4]
        if request.param:
            group = dist.new_group(ranks=list(all_ranks), backend="nccl")
            if rank in all_ranks:
                transport = group
                created.append(group)
        if rank in all_ranks:
            producers, decoders = producer_ranks, decoder_ranks
        for size in (4, 2, 1):
            for start in range(decoder_ranks[0], decoder_ranks[-1] + 1, size):
                ranks = tuple(range(start, start + size))
                group = dist.new_group(ranks=list(ranks), backend="nccl")
                if rank in ranks:
                    owned[size] = group
                    created.append(group)
                    if not request.param and size == 4:
                        transport = group
    yield transport, producers, decoders, owned
    for group in reversed(created):
        dist.destroy_process_group(group)


def _source(generation, *, pure_text, axes, short_media=False):
    tasks = (
        [] if pure_text else [MediaTask(("asset", generation, i), i + 2, 4 - i) for i in range(3)]
    )
    samples = {}
    for ordinal, physical in enumerate((96, 64, 48, 32, 32, 16, 16, 16, 16)):
        logical = physical - 3
        media = []
        media_samples = (0, 3, 4) if short_media else (0, 1)
        if tasks and ordinal in media_samples:
            media = [tasks[0], tasks[1]] if ordinal == 0 else [tasks[2], tasks[0]]
        tokens = torch.full((physical,), ordinal + 1, dtype=torch.long)
        tokens[1 : 1 + sum(task.length for task in media)] = _MEDIA_TOKEN
        tokens[logical:] = 0
        mask = torch.zeros(physical)
        mask[2:logical] = 1  # Prompt positions remain real tokens despite zero loss.
        positions = torch.arange(physical) + ordinal * 100
        if axes > 1:
            positions = positions[None] + 1000 * torch.arange(axes)[:, None]
        positions[..., logical:] = 777
        samples[f"sample-{generation}-{ordinal * 17 + 3}"] = dict(
            tokens=tokens,
            labels=tokens.clone(),
            loss_mask=mask,
            position_ids=positions,
            original_seq_len=logical,
            padded_seq_len=physical,
            media_ids=[task.media_id for task in media],
        )
    if generation:
        samples = dict(reversed(tuple(samples.items())))
        tasks.reverse()
    return samples, tasks


def _rows(task):
    return torch.arange(task.length, device="cuda", dtype=torch.float64) + 1 + task.media_id[-1] * 5


def _objective(samples, tasks, encoder, decoder):
    """Independent reference in original sample order, without plans or CP indices."""
    rows = {task.media_id: _rows(task) for task in tasks}
    total = decoder * 0
    for sample in samples.values():
        values = sample["tokens"].to(device="cuda", dtype=torch.float64).clone()
        if sample["media_ids"]:
            values[values == _MEDIA_TOKEN] = (
                torch.cat([rows[mid] for mid in sample["media_ids"]]) * encoder
            )
        total = total + ((values * decoder).square() * sample["loss_mask"].cuda()).sum()
    return total


@pytest.mark.parametrize("dynamic", (False, True))
def test_public_preparation_and_runtime_lifecycle(runtime_groups, dynamic):
    domain, producers, ranks, cp_groups = runtime_groups
    rank = dist.get_rank()
    is_decoder = rank in ranks
    disaggregated = producers != ranks
    limit = 32 if dynamic else 64
    scheduler = (DefaultDynamicCPScheduler if dynamic else DpBalancedScheduler)(
        limit, cp_size=2, dp_size=2, microbatch_group_size_per_vp_stage=None
    )
    config = ModelParallelConfig(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=2,
        max_seqlen_per_dp_cp_rank=limit,
        pad_packed_seq_alignment="max",
    )
    partition = (
        PartitionAdapter(
            PartitionConfig(False, True, False, limit * 4, kv_format="thd", cp_group=cp_groups[2])
        )
        if is_decoder
        else None
    )
    runtime = MimoStepRuntime(domain, producer_ranks=producers, decoder_ranks=ranks)
    old_transfer = None
    observed_cp = torch.zeros(5, device="cuda", dtype=torch.int)
    for step, (pure_text, forward_only, axes) in enumerate(
        ((False, False, 3), (True, False, 1), (False, True, 2), (False, False, 3))
    ):
        samples, tasks = _source(
            step // 2, pure_text=pure_text, axes=axes, short_media=disaggregated
        )
        snapshots = {
            sid: {key: value.clone() for key, value in sample.items() if torch.is_tensor(value)}
            for sid, sample in samples.items()
        }
        assigned, slices = assign_media_to_producers(tasks, producers)
        local_tasks = assigned.get(rank, [])
        if is_decoder:
            plan = prepare_mimo_step(
                samples,
                slices,
                scheduler=scheduler,
                domain_ranks=ranks,
                cp_groups=cp_groups,
                config=config,
                device=torch.device("cuda"),
            )
            assert plan.num_source_samples == len(samples)
            assert plan.num_supervised_tokens == sum(
                int(s["loss_mask"].sum()) for s in samples.values()
            )
            assert plan.source_tokens == sum(s["padded_seq_len"] for s in samples.values())
            assert plan.source_tokens_squared == sum(
                s["padded_seq_len"] ** 2 for s in samples.values()
            )
        # Only route descriptors cross the pools; producers never construct decoder batches.
        shared = [tuple(item.feature_plans for item in plan.rounds) if is_decoder else None]
        dist.broadcast_object_list(shared, src=ranks[0], group=domain)
        rounds = shared[0]
        assert len(rounds) > 1
        if disaggregated and tasks:
            assert any(len(pack.features) == 4 for plans in rounds for pack in plans)
        encoder = torch.tensor(2.0, device="cuda", dtype=torch.float64, requires_grad=True)
        decoder = torch.tensor(0.25, device="cuda", dtype=torch.float64, requires_grad=True)
        backward_calls = []
        encoder.register_hook(lambda gradient: backward_calls.append(1))
        with torch.set_grad_enabled(not forward_only):
            features = (
                torch.cat([_rows(task) for task in local_tasks])[:, None] * encoder
                if local_tasks
                else torch.empty((0, 1), device="cuda", dtype=torch.float64)
            )
        # Preparation context must not override the explicit execution mode.
        with torch.no_grad():
            runtime.begin_step(features, len(rounds), forward_only=forward_only)
        with pytest.raises(RuntimeError, match="active MIMO"):
            runtime.begin_step(features, len(rounds), forward_only=forward_only)
        finish = runtime.finish_forward if forward_only else runtime.finish_backward
        with pytest.raises(RuntimeError, match="complete every"):
            finish()
        for round_index, plans in enumerate(rounds):
            item = plan.rounds[round_index] if is_decoder else None
            group = item.cp_group if is_decoder else None
            transfer = runtime.route(plans, group)
            if not is_decoder:
                assert transfer.features is transfer.local_plan is transfer.cp_group is None
            else:
                assert transfer.features.requires_grad == (not forward_only)
            if old_transfer is not None:
                with pytest.raises(RuntimeError, match="stale or completed"):
                    runtime.complete_round(old_transfer)
            with pytest.raises(RuntimeError, match="pending"):
                runtime.route(plans, group)
            with pytest.raises(RuntimeError, match="complete every"):
                finish()
            if is_decoder:
                _decode_round(
                    item, transfer, samples, tasks, partition, decoder, forward_only, axes, limit
                )
                observed_cp[group.size()] = 1
            assert encoder.grad is None and backward_calls == []
            runtime.complete_round(transfer)
            with pytest.raises(RuntimeError, match="stale or completed"):
                runtime.complete_round(transfer)
            old_transfer = transfer
        with pytest.raises(RuntimeError, match="already completed"):
            runtime.route(plans, group)
        wrong_finish = runtime.finish_backward if forward_only else runtime.finish_forward
        with pytest.raises(RuntimeError, match="requested execution mode"):
            wrong_finish()
        finish()
        assert not runtime.active
        assert runtime.local_features is runtime.feature_gradient is runtime.pending is None
        assert backward_calls == ([1] if local_tasks and not forward_only else [])
        with pytest.raises(RuntimeError, match="requested execution mode"):
            finish()
        if forward_only:
            assert encoder.grad is decoder.grad is None
        else:
            encoder_ref = encoder.detach().requires_grad_()
            decoder_ref = decoder.detach().requires_grad_()
            _objective(samples, tasks, encoder_ref, decoder_ref).backward()
            for parameter, reference in ((encoder, encoder_ref), (decoder, decoder_ref)):
                gradient = (
                    parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)
                )
                dist.all_reduce(gradient, group=domain)
                torch.testing.assert_close(
                    gradient,
                    reference.grad if reference.grad is not None else torch.zeros_like(reference),
                    rtol=0,
                    atol=0,
                )
        for sid, snapshot in snapshots.items():
            for key, value in snapshot.items():
                torch.testing.assert_close(samples[sid][key], value, rtol=0, atol=0)
    dist.all_reduce(observed_cp, op=dist.ReduceOp.MAX, group=domain)
    assert observed_cp.nonzero().flatten().tolist() == ([1, 2, 4] if dynamic else [2])


def _decode_round(item, transfer, samples, tasks, partition, decoder, forward_only, axes, limit):
    """Check an independently assembled pack before the existing CP partition and objective."""
    inputs = item.model_inputs
    assert inputs["input_ids"].numel() == limit * item.cp_group.size()
    assert inputs["position_ids"].shape[:-2] == (() if axes == 1 else (axes,))
    for slot, sid in enumerate(item.sample_ids):
        sample = samples[sid]
        start = int(inputs["cu_seqlens_padded"][slot])
        end = start + sample["original_seq_len"]
        assert int(inputs["cu_seqlens"].diff()[slot]) == sample["original_seq_len"]
        torch.testing.assert_close(
            inputs["position_ids"][..., 0, start:end].cpu(),
            sample["position_ids"][..., : sample["original_seq_len"]],
        )
        assert not inputs["padding_mask"][0, start:end].any()
        assert not inputs["loss_mask"][0, start : start + 2].any()
    padding = inputs["padding_mask"][0]
    assert not inputs["position_ids"][..., 0, padding].any()
    assert not inputs["loss_mask"][0, padding].any()
    rows = {task.media_id: _rows(task) for task in tasks}
    pieces = [rows[mid] for sid in item.sample_ids for mid in samples[sid]["media_ids"]]
    expected = torch.cat(pieces)[:, None] * 2 if pieces else transfer.features.new_empty((0, 1))
    torch.testing.assert_close(transfer.features, expected, rtol=0, atol=0)
    with torch.set_grad_enabled(not forward_only):
        values = inputs["input_ids"].to(torch.float64).clone()
        values[values == _MEDIA_TOKEN] = transfer.features[:, 0]
        batch = partition.partition(
            embeddings=values.transpose(0, 1).unsqueeze(-1),
            input_ids=inputs["input_ids"],
            position_ids=inputs["position_ids"],
            labels=inputs["labels"],
            loss_mask=inputs["loss_mask"],
            mtp_input_mask=None,
            cu_seqlens=inputs["cu_seqlens"],
            cu_seqlens_padded=inputs["cu_seqlens_padded"],
            max_seqlen=inputs["max_seqlen"],
            padding_mask=inputs["padding_mask"],
            cp_group=item.cp_group,
        ).get_batch()
        output = (batch["decoder_input"].squeeze(-1).transpose(0, 1) * decoder).square()
        loss = (output * batch["loss_mask"]).sum()
        if not forward_only:
            loss.backward()
