# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Joint data scheduling over existing context-parallel communicator families.

The planning/transport domain may include GTP weight peers. Runtime CP groups
remain explicit constraints: gathering data globally does not create new CP
communicators. This module deliberately has no distributed or tensor imports.
"""

from collections import Counter
from typing import Iterable, List, Optional, Sequence, Tuple


def _local_length(physical_length: int, cp_size: int, alignment: int) -> int:
    local = (physical_length + cp_size - 1) // cp_size
    return ((local + alignment - 1) // alignment) * alignment


def _sequence_length(length: int, cp_size: int, pad_enabled: bool) -> int:
    alignment = 2 * cp_size if pad_enabled and cp_size > 1 else 1
    return ((length + alignment - 1) // alignment) * alignment


def _group_family(total_ranks: int, allowed_cp_groups: Iterable[Sequence[int]]):
    groups = sorted(
        {tuple(sorted(group)) for group in allowed_cp_groups}, key=lambda g: (len(g), g)
    )
    if total_ranks < 1 or not groups:
        raise ValueError("Joint CP scheduling needs ranks and at least one allowed CP group")
    for group in groups:
        if not group or len(set(group)) != len(group):
            raise ValueError(f"Invalid CP group members: {group}")
        if any(rank < 0 or rank >= total_ranks for rank in group):
            raise ValueError(f"CP group outside scheduling rank range: {group}")
    sets = {group: frozenset(group) for group in groups}
    for index, first in enumerate(groups):
        for second in groups[index + 1 :]:
            a, b = sets[first], sets[second]
            if a & b and not (a <= b or b <= a):
                raise ValueError("Allowed CP groups must form disjoint or nested runtime families")
    lanes = [group for group in groups if not any(sets[group] < other for other in sets.values())]
    if Counter(rank for lane in lanes for rank in lane) != Counter(range(total_ranks)):
        raise ValueError("Maximal allowed CP groups must partition the scheduling ranks exactly")
    if len({len(lane) for lane in lanes}) != 1:
        raise ValueError("Joint CP scheduling currently requires equally sized maximal CP groups")
    return groups, lanes


def _refine_lane(ids, lane, groups, lengths, capacity, pad_alignment, pad_enabled):
    """Prefer smaller legal CP groups; retain a capacity-safe full-lane fallback."""
    lane_set = set(lane)
    candidates = [group for group in groups if set(group) <= lane_set]
    packs = {}
    occupied = set()

    def physical(sample_ids, group):
        return sum(_sequence_length(lengths[sid], len(group), pad_enabled) for sid in sample_ids)

    for sid in sorted(ids, key=lambda item: -lengths[item]):
        best = None
        current_costs = {
            group: sum(lengths[item] ** 2 for item in pack) / len(group)
            for group, pack in packs.items()
        }
        for group in candidates:
            if group not in packs and occupied.intersection(group):
                continue
            combined = packs.get(group, []) + [sid]
            if _local_length(physical(combined, group), len(group), pad_alignment) > capacity:
                continue
            cost = sum(lengths[item] ** 2 for item in combined) / len(group)
            projected = max(
                [cost] + [value for other, value in current_costs.items() if other != group]
            )
            # At equal work and CP size, fill an unused group before enlarging a
            # pack. This avoids needless CP expansion just to occupy idle ranks.
            key = (
                (len(group), projected, False, group)
                if not packs
                else (projected, len(group), group in packs, group)
            )
            if best is None or key < best[0]:
                best = (key, group)
        if best is None:
            return {lane: list(ids)}
        group = best[1]
        packs.setdefault(group, []).append(sid)
        occupied.update(group)

    # Enlarge only into an existing ancestor whose additional ranks are free.
    # Each success adds at least one rank; there is no unbounded fill loop.
    for _ in range(len(lane)):
        if occupied == lane_set:
            return packs
        expansion = None
        for group, sample_ids in packs.items():
            for ancestor in candidates:
                if not set(group) < set(ancestor):
                    continue
                if (set(ancestor) - set(group)) & occupied:
                    continue
                if (
                    _local_length(physical(sample_ids, ancestor), len(ancestor), pad_alignment)
                    > capacity
                ):
                    continue
                key = (len(ancestor), ancestor, group)
                if expansion is None or key < expansion[0]:
                    expansion = (key, group, ancestor)
        if expansion is None:
            break
        _, group, ancestor = expansion
        packs[ancestor] = packs.pop(group)
        occupied.update(ancestor)
    return {lane: list(ids)}


def _validate_plan(schedule, lengths, groups, total_ranks, capacity, alignment, pad_enabled):
    allowed = set(groups)
    occurrence = Counter()
    for row in schedule:
        if len(row) != total_ranks or any(not ids for ids in row):
            raise ValueError("Joint CP schedule has an empty rank or inconsistent round width")
        placements = {}
        for rank, ids in enumerate(row):
            if len(ids) != len(set(ids)):
                raise ValueError("A joint CP pack repeats a logical sample")
            placements.setdefault(tuple(ids), []).append(rank)
        for ids, members in placements.items():
            group = tuple(members)
            if group not in allowed:
                raise ValueError(f"Schedule selected a nonexistent runtime CP group: {group}")
            physical = sum(_sequence_length(lengths[sid], len(group), pad_enabled) for sid in ids)
            if _local_length(physical, len(group), alignment) > capacity:
                raise ValueError("Joint CP schedule exceeds physical per-rank token capacity")
            occurrence.update(ids)
    if occurrence != Counter({sid: 1 for sid in lengths}):
        raise ValueError("Joint CP schedule does not consume each logical sample exactly once")


def plan_joint_dynamic_cp(
    sample_id_seqlens: Sequence[Tuple[int, int]],
    *,
    total_ranks: int,
    allowed_cp_groups: Iterable[Sequence[int]],
    max_seqlen_per_rank: int,
    microbatch_group_size_per_vp_stage: Optional[int] = None,
    pad_alignment: int = 1,
    pad_enabled: bool = False,
) -> List[List[List[int]]]:
    """Build one global schedule while preserving existing CP group membership.

    ``allowed_cp_groups`` uses the same slot indices as the output's rank axis;
    indices may be noncontiguous in a process group's global-rank ordering.
    With GTP these are normally the runtime CP families at fixed G coordinates.
    Supplying only static CP groups produces a coordinated static-CP baseline.

    Each round first reserves a nonempty pack per maximal allowed group, then
    fills these packs greedily from the global pool. A second pass selects
    smaller legal CP groups where possible. Full-group packing is a verified
    physical-capacity fallback, never a new communicator. Tail and VPP repair
    redistribute complete independent sequences and never duplicate a sample.

    This is a bounded heuristic, not an optimal bin-packing solver. Failure to
    construct a schedule does not establish mathematical infeasibility. Callers
    must propagate failures collectively before entering data rerouting/model
    collectives. No communication is performed by this function.
    """
    groups, lanes = _group_family(total_ranks, allowed_cp_groups)
    if max_seqlen_per_rank < 1 or pad_alignment < 1:
        raise ValueError("Joint CP token capacity and padding alignment must be positive")
    alignment = pad_alignment if pad_enabled else 1
    multiple = microbatch_group_size_per_vp_stage or 1
    if multiple < 1:
        raise ValueError("VPP microbatch group size must be positive")
    lengths = dict(sample_id_seqlens)
    if len(lengths) != len(sample_id_seqlens):
        raise ValueError("Joint CP input contains duplicate sample IDs")
    if not lengths or any(length <= 0 for length in lengths.values()):
        raise ValueError("Joint CP input must contain nonempty independent sequences")

    lane_size = len(lanes[0])
    num_lanes = len(lanes)
    physical = {
        sid: _sequence_length(length, lane_size, pad_enabled) for sid, length in lengths.items()
    }
    if any(
        _local_length(value, lane_size, alignment) > max_seqlen_per_rank
        for value in physical.values()
    ):
        raise ValueError("A sequence exceeds capacity of every allowed runtime CP group")
    remaining = sorted(lengths, key=lambda sid: -lengths[sid])
    rows = []

    def take_donors(count):
        """Keep one whole sequence in every existing maximal-group pack."""
        donors = []
        for row in reversed(rows):
            for pack in row:
                while len(pack) > 1 and len(donors) < count:
                    donors.append(pack.pop())
                if len(donors) == count:
                    return donors
        raise ValueError(
            "Joint CP greedy packing could not fill all ranks without empty work; "
            "increase the number of independent sequences or use a different packing layout"
        )

    # Every iteration consumes at least one sequence. A tail repair yields
    # exactly num_lanes remaining sequences, which finish on the next round.
    while remaining:
        if len(remaining) < num_lanes:
            raise ValueError(
                "Joint CP scheduling needs at least one independent sequence per "
                "maximal allowed CP group in every round"
            )
        row = [[sid] for sid in remaining[:num_lanes]]
        totals = [physical[pack[0]] for pack in row]
        costs = [lengths[pack[0]] ** 2 / lane_size for pack in row]
        leftovers = []
        for sid in remaining[num_lanes:]:
            candidates = [
                index
                for index in range(num_lanes)
                if _local_length(totals[index] + physical[sid], lane_size, alignment)
                <= max_seqlen_per_rank
            ]
            if not candidates:
                leftovers.append(sid)
                continue
            index = min(
                candidates, key=lambda candidate: (costs[candidate], totals[candidate], candidate)
            )
            row[index].append(sid)
            totals[index] += physical[sid]
            costs[index] += lengths[sid] ** 2 / lane_size
        rows.append(row)
        if 0 < len(leftovers) < num_lanes:
            leftovers.extend(take_donors(num_lanes - len(leftovers)))
        remaining = sorted(leftovers, key=lambda sid: -lengths[sid])

    target_rounds = ((len(rows) + multiple - 1) // multiple) * multiple
    if len(lengths) < num_lanes * target_rounds:
        raise ValueError(
            "VPP alignment requires more nonempty maximal CP packs than there "
            "are independent sequences; no dummy work or context splitting is allowed"
        )
    for _ in range(target_rounds - len(rows)):
        rows.append([[sid] for sid in take_donors(num_lanes)])

    schedule = []
    for row in rows:
        output = [[] for _ in range(total_ranks)]
        for lane, ids in zip(lanes, row):
            packs = _refine_lane(
                ids, lane, groups, lengths, max_seqlen_per_rank, alignment, pad_enabled
            )
            for group, sample_ids in packs.items():
                for rank in group:
                    output[rank] = list(sample_ids)
        schedule.append(output)
    _validate_plan(
        schedule, lengths, groups, total_ranks, max_seqlen_per_rank, alignment, pad_enabled
    )
    return schedule
