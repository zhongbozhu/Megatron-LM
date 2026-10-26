# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU-only contract tests for the topology-aware joint packing planner.

Run directly with Python for a dependency-free check. Loading the standalone
module by path avoids Megatron's CUDA/distributed package initialization; normal
distributed pytest discovery can run the same unittest cases.
"""

import importlib.util
import random
import unittest
from collections import Counter
from pathlib import Path

_MODULE = Path(__file__).resolve().parents[3] / "megatron/core/datasets/gtp_dynamic_cp_scheduler.py"
_SPEC = importlib.util.spec_from_file_location("gtp_dynamic_cp_scheduler_cpu", _MODULE)
_PLANNER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PLANNER)
plan_joint_dynamic_cp = _PLANNER.plan_joint_dynamic_cp


class TestJointGTPDynamicCPScheduler(unittest.TestCase):
    def check_schedule(self, schedule, lengths, groups, ranks, capacity, alignment=1, padded=False):
        """Independent output audit, including exact CP membership and padding."""
        allowed = {frozenset(group) for group in groups}
        seen = Counter()
        rank_load = [0] * ranks
        for row in schedule:
            self.assertEqual(len(row), ranks)
            self.assertTrue(all(row))
            owners = {}
            for rank, ids in enumerate(row):
                self.assertEqual(len(ids), len(set(ids)))
                owners.setdefault(tuple(ids), []).append(rank)
            for ids, members in owners.items():
                self.assertIn(frozenset(members), allowed)
                cp = len(members)
                physical = 0
                for sid in ids:
                    seen[sid] += 1
                    seq_alignment = 2 * cp if padded and cp > 1 else 1
                    physical += -(-lengths[sid] // seq_alignment) * seq_alignment
                    self.assertEqual(
                        [rank for rank, pack in enumerate(row) if sid in pack], members
                    )
                local = -(-physical // cp)
                if padded:
                    local = -(-local // alignment) * alignment
                self.assertLessEqual(local, capacity)
                for rank in members:
                    rank_load[rank] += local
        self.assertEqual(seen, Counter({sid: 1 for sid in lengths}))
        return rank_load

    @staticmethod
    def family(lanes):
        groups = []
        for lane in lanes:
            for size in (1, 2, 4, 8):
                if size <= len(lane) and len(lane) % size == 0:
                    groups.extend(
                        tuple(lane[start : start + size]) for start in range(0, len(lane), size)
                    )
            groups.append(tuple(lane))
        return groups

    def plan(self, lengths, groups, capacity=16384, multiple=None, alignment=32, padded=True):
        return plan_joint_dynamic_cp(
            list(lengths.items()),
            total_ranks=len({rank for group in groups for rank in group}),
            allowed_cp_groups=groups,
            max_seqlen_per_rank=capacity,
            microbatch_group_size_per_vp_stage=multiple,
            pad_alignment=alignment,
            pad_enabled=padded,
        )

    def test_four_gpu_asymmetric_lengths_global_three_rounds(self):
        lengths = {sid: (8192 if sid < 4 else 32768) for sid in range(8)}
        groups = self.family([(0, 1), (2, 3)])
        schedule = self.plan(lengths, groups)
        self.assertEqual(len(schedule), 3)
        self.assertEqual(
            self.check_schedule(schedule, lengths, groups, 4, 16384, 32, True), [40960] * 4
        )
        # The full four-rank CP group must never be invented by tail fill.
        self.assertNotEqual(schedule[-1][0], schedule[-1][1])

    def test_noncontiguous_two_rank_communicators(self):
        lengths = {sid: (8192 if sid < 4 else 32768) for sid in range(8)}
        groups = self.family([(0, 2), (1, 3)])
        schedule = self.plan(lengths, groups)
        self.assertEqual(len(schedule), 3)
        self.assertEqual(schedule[0][0], schedule[0][2])
        self.assertEqual(schedule[0][1], schedule[0][3])
        self.check_schedule(schedule, lengths, groups, 4, 16384, 32, True)

    def test_eight_gpu_replica_layout_preserves_actual_members(self):
        groups = self.family([(0, 1, 4, 5), (2, 3, 6, 7)])
        lengths = {0: 65536, 1: 65536, 2: 8192, 3: 8192}
        schedule = self.plan(lengths, groups)
        self.assertEqual(len(schedule), 2)
        self.assertEqual(schedule[0][0], schedule[0][4])
        self.assertNotEqual(schedule[0][0], schedule[0][2])
        self.check_schedule(schedule, lengths, groups, 8, 16384, 32, True)

    def test_static_cp_baseline_uses_only_supplied_groups(self):
        groups = [(0, 1), (2, 3), (4, 5), (6, 7)]
        lengths = {sid: 8192 + 8192 * (sid % 3) for sid in range(32)}
        schedule = self.plan(lengths, groups)
        self.check_schedule(schedule, lengths, groups, 8, 16384, 32, True)
        for row in schedule:
            for left, right in groups:
                self.assertEqual(row[left], row[right])

    def test_tail_repair_moves_complete_sequences_without_duplicates(self):
        # A naive greedy fill leaves a single sample after the first round.
        lengths = {sid: 8 for sid in range(9)}
        groups = self.family([(0, 1), (2, 3)])
        schedule = self.plan(lengths, groups, capacity=16, alignment=1, padded=False)
        self.assertEqual(len(schedule), 2)
        self.check_schedule(schedule, lengths, groups, 4, 16)

    def test_vpp_round_alignment_preserves_all_samples(self):
        lengths = {sid: (8192 if sid < 4 else 32768) for sid in range(8)}
        groups = self.family([(0, 1), (2, 3)])
        schedule = self.plan(lengths, groups, multiple=4)
        self.assertEqual(len(schedule), 4)
        self.check_schedule(schedule, lengths, groups, 4, 16384, 32, True)

    def test_vpp_infeasible_nonempty_target_fails_before_execution(self):
        groups = self.family([(0, 1), (2, 3)])
        with self.assertRaisesRegex(ValueError, "VPP alignment"):
            self.plan({0: 8192, 1: 8192}, groups, multiple=2)

    def test_three_long_samples_does_not_fall_back_to_global_cp4(self):
        groups = self.family([(0, 1), (2, 3)])
        with self.assertRaisesRegex(ValueError, "greedy packing could not"):
            self.plan({0: 32768, 1: 32768, 2: 32768}, groups)

    def test_physical_padding_limits_and_irregular_sizes(self):
        groups = self.family([(0, 1), (2, 3)])
        lengths = {sid: 7 + sid % 4 for sid in range(24)}
        schedule = self.plan(lengths, groups, capacity=16, alignment=8)
        self.check_schedule(schedule, lengths, groups, 4, 16, 8, True)
        with self.assertRaisesRegex(ValueError, "exceeds capacity"):
            self.plan({0: 17, 1: 17}, groups, capacity=8, alignment=8)

    def test_non_power_of_two_lane_fallback_is_existing_cp6(self):
        groups = self.family([(0, 1, 2, 3, 4, 5), (6, 7, 8, 9, 10, 11)])
        lengths = {sid: 128 for sid in range(8)}
        schedule = self.plan(lengths, groups, capacity=256, alignment=2)
        self.check_schedule(schedule, lengths, groups, 12, 256, 2, True)

    def test_rejects_duplicate_ids_and_crossing_group_families(self):
        with self.assertRaisesRegex(ValueError, "duplicate sample IDs"):
            plan_joint_dynamic_cp(
                [(0, 8), (0, 8)],
                total_ranks=2,
                allowed_cp_groups=[(0,), (1,)],
                max_seqlen_per_rank=16,
            )
        with self.assertRaisesRegex(ValueError, "disjoint or nested"):
            self.plan({0: 8, 1: 8}, [(0, 1), (1, 2), (2, 3)])

    def test_seeded_many_sample_cases_and_deterministic_replay(self):
        rng = random.Random(20261024)
        groups = self.family([(0, 1, 4, 5), (2, 3, 6, 7)])
        for _ in range(30):
            lengths = {sid: rng.randrange(1, 32769) for sid in range(128)}
            schedule = self.plan(lengths, groups)
            self.assertEqual(schedule, self.plan(lengths, groups))
            self.check_schedule(schedule, lengths, groups, 8, 16384, 32, True)


if __name__ == "__main__":
    unittest.main()
