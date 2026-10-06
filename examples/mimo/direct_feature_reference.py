# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Opt-in, deliberately simple image-identity oracle for the native MIMO boundary.

This diagnostic sends an image directly to every consumer, and returns each
consumer's gradient directly to its producer. It does not use the production
bridge's plans, source slices, pack leaders, or CP collectives. Encoder outputs
and the decoder boundary remain separated by an explicit backward, exactly as
in native training. This is a correctness reference, not a production transport.
"""

import torch
import torch.distributed as dist

from megatron.core.models.mimo.comm.pack_bridge import (
    FeatureSlice,
    PackFeaturePlan,
    PackFeatureTransfer,
)


class DirectFeatureReference:
    """Build source identity from encoder inputs and consumption from source samples."""

    def __init__(self, group, samples, encoder_tasks, assignments):
        self.group = group
        self.rank = dist.get_rank()
        self.ranks = tuple(dist.get_process_group_ranks(group))
        self.images = {}
        for producer in self.ranks:
            offset = 0
            for image in encoder_tasks[producer]:
                image_id, length = image['image_id'], int(image['length'])
                if image_id in self.images or length <= 0:
                    raise ValueError('Reference requires unique nonempty source images')
                self.images[image_id] = (producer, offset, length)
                offset += length
        self.rounds = []
        consumed = []
        for assignment in assignments:
            consumers = {}
            groups = {}
            for rank, sample_ids in zip(self.ranks, assignment):
                groups.setdefault(tuple(sample_ids), []).append(rank)
                consumers[rank] = tuple(
                    image_id for sid in sample_ids for image_id in samples[sid]['media_ids']
                )
            # These records are only for the existing detached diagnostic writer;
            # forward and backward below operate on image identities directly.
            metadata = tuple(
                PackFeaturePlan(
                    tuple(ranks),
                    tuple(FeatureSlice(*self.images[image_id]) for image_id in consumers[ranks[0]]),
                )
                for ranks in groups.values()
            )
            for ranks in groups.values():
                consumed.extend(consumers[ranks[0]])
            self.rounds.append((consumers, metadata))
        if sorted(consumed) != sorted(self.images):
            raise ValueError('Reference source images must be consumed exactly once per step')
        self.initialized = False

    @staticmethod
    def _wait(operations):
        if operations:
            for request in dist.batch_isend_irecv(operations):
                request.wait()

    @torch.no_grad()
    def forward(self, source, round_id, cp_group):
        if not self.initialized:
            dist.all_reduce(source.new_zeros(1), group=self.group)
            self.initialized = True
        consumers, metadata = self.rounds[round_id]
        local_ids = consumers[self.rank]
        received, operations, keep_alive = {}, [], []
        for image_id, (producer, offset, length) in sorted(self.images.items()):
            for target in self.ranks:
                if image_id not in consumers[target]:
                    continue
                if self.rank == producer:
                    payload = source[offset : offset + length].contiguous()
                    keep_alive.append(payload)
                    if target == producer:
                        received[image_id] = payload
                    else:
                        operations.append(dist.P2POp(dist.isend, payload, target, group=self.group))
                elif self.rank == target:
                    payload = source.new_empty((length, source.shape[1]))
                    received[image_id] = payload
                    operations.append(dist.P2POp(dist.irecv, payload, producer, group=self.group))
        self._wait(operations)
        features = (
            torch.cat([received[image_id] for image_id in local_ids])
            if local_ids
            else source.new_empty((0, source.shape[1]))
        )
        local_plan = next(i for i, plan in enumerate(metadata) if self.rank in plan.cp_ranks)
        if tuple(dist.get_process_group_ranks(cp_group)) != metadata[local_plan].cp_ranks:
            raise ValueError('Reference source assignment disagrees with runtime CP group')
        transfer = PackFeatureTransfer(
            features.requires_grad_(True), source, metadata, local_plan, cp_group
        )
        # Save identity and round with the boundary; never read mutable runtime CP
        # state during backward or infer correspondence from a production plan.
        transfer.reference_round = round_id
        return transfer

    @torch.no_grad()
    def backward(self, transfer):
        if transfer.backward_complete:
            raise RuntimeError('Reference boundary backward may only run once')
        consumers, _ = self.rounds[transfer.reference_round]
        local_ids = consumers[self.rank]
        gradient = transfer.features.grad
        if gradient is None:
            gradient = torch.zeros_like(transfer.features, dtype=torch.float32)
        else:
            gradient = gradient.detach().float()
        local = {}
        offset = 0
        for image_id in local_ids:
            length = self.images[image_id][2]
            local[image_id] = gradient[offset : offset + length].contiguous()
            offset += length
        returned = torch.zeros_like(transfer.local_features, dtype=torch.float32)
        operations, contributions, keep_alive = [], [], []
        for image_id, (producer, offset, length) in sorted(self.images.items()):
            for consumer in self.ranks:
                if image_id not in consumers[consumer]:
                    continue
                if self.rank == consumer:
                    payload = local[image_id]
                    keep_alive.append(payload)
                    if producer == consumer:
                        contributions.append((offset, payload))
                    else:
                        operations.append(
                            dist.P2POp(dist.isend, payload, producer, group=self.group)
                        )
                elif self.rank == producer:
                    payload = returned.new_empty((length, returned.shape[1]))
                    contributions.append((offset, payload))
                    operations.append(dist.P2POp(dist.irecv, payload, consumer, group=self.group))
        self._wait(operations)
        for offset, payload in contributions:
            returned[offset : offset + len(payload)].add_(payload)
        transfer.backward_complete = True
        return returned


def compare_full_decoder_inputs(reference, candidate, step):
    """Compare every captured decoder input/gradient under identical execution layout."""
    import math
    from pathlib import Path

    from examples.mimo.boundary_diagnostics import _difference

    reference = Path(reference) / f'step{step:08d}'
    candidate = Path(candidate) / f'step{step:08d}'
    paths = sorted(path.relative_to(reference) for path in reference.glob('rank*/*.pt'))
    if not paths or paths != sorted(
        path.relative_to(candidate) for path in candidate.glob('rank*/*.pt')
    ):
        raise ValueError('Complete decoder input capture coverage differs')
    totals = {}
    for path in paths:
        left = torch.load(reference / path, map_location='cpu', weights_only=True)
        right = torch.load(candidate / path, map_location='cpu', weights_only=True)
        for key in (
            'sample_ids',
            'sample_lengths',
            'cp_ranks',
            'padded_boundaries',
            'logical_boundaries',
        ):
            if left[key] != right[key]:
                raise ValueError(f'Decoder execution layout differs: {path}/{key}')
        if not torch.equal(left['physical_indices'], right['physical_indices']):
            raise ValueError(f'Decoder physical ownership differs: {path}')
        before, after = left.get('full_decoder_input'), right.get('full_decoder_input')
        if not before or not after or len(before) != len(after):
            raise ValueError(f'Missing full decoder input or invocation count differs: {path}')
        for original, observed in zip(before, after):
            if (
                original.keys() != observed.keys()
                or original['grad_enabled'] != observed['grad_enabled']
            ):
                raise ValueError(f'Decoder gradient capture or grad mode differs: {path}')
            for field in ('value', 'gradient'):
                if field not in original:
                    continue
                difference = _difference(observed[field], original[field])
                key = left['phase'] + '_' + field
                total = totals.setdefault(
                    key,
                    dict(
                        captures=0,
                        elements=0,
                        squared_error=0.0,
                        squared_reference=0.0,
                        max_abs_error=0.0,
                        exact=True,
                        all_finite=True,
                    ),
                )
                total['captures'] += 1
                for quantity in ('elements', 'squared_error', 'squared_reference'):
                    total[quantity] += difference[quantity]
                total['max_abs_error'] = max(total['max_abs_error'], difference['max_abs_error'])
                total['exact'] &= difference['exact']
                total['all_finite'] &= difference['all_finite']
    for total in totals.values():
        error, baseline = total['squared_error'], total['squared_reference']
        total['relative_l2'] = (
            math.sqrt(error / baseline) if baseline else (0.0 if not error else None)
        )
    return dict(
        step=step,
        files=len(paths),
        reference=str(reference),
        candidate=str(candidate),
        totals=totals,
    )
