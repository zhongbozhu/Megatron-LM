# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Stdlib-only report tests, runnable without Torch or a distributed runtime."""

import json
import tempfile
import unittest
from pathlib import Path

from examples.mimo.fixed_routing_report import build_report, summarize_auxiliary


def _audit():
    histogram = dict(
        pairs=2,
        missing=0,
        unexpected=0,
        histogram_mismatches=0,
        histogram_l1=0,
        score_sum_max_abs=0.0,
        layers={},
    )
    assignments = dict(enabled=True)
    for kind in (
        'auxiliary_training',
        'dispatch_training',
        'auxiliary_recompute',
        'dispatch_recompute',
    ):
        assignments[kind] = dict(
            pairs=2, rows=20, mismatched_pairs=0, different_rows=0, shape_mismatches=0
        )
    return {**histogram, 'recompute': histogram.copy(), 'token_assignments': assignments}


def _metric():
    return dict(
        step=0,
        training=True,
        source_samples=2,
        source_images=3,
        source_tokens=10,
        supervised_tokens=8,
        mtp_tokens=[6],
        loss=0.4,
        mtp_loss=[0.5],
        aux={'loss': 0.1},
        fixed_routing=dict(
            source_step=0,
            source_samples=2,
            source_tokens=10,
            routers=2,
            mtp_routers=1,
            route_sha256='a' * 64,
        ),
        aux_replay=_audit(),
        full_gradient_validation=dict(
            mode='compare',
            reference_dir='/example/reference',
            global_elements=100,
            all_finite=True,
            totals=dict(l2=1.0, reference_l2=1.0, relative_l2=0.2),
            components={'vision': {'relative_l2': 0.3}},
        ),
    )


class TestFixedRoutingReport(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.study = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, name, metric):
        directory = self.study / 'logs' / f'{name}-metrics'
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'metrics.jsonl').write_text(json.dumps(metric) + '\n')
        (self.study / 'logs' / f'{name}.out').write_text(
            'iteration 1/ 1 | learning rate: 2e-6 | grad norm: 1.234 | number of skipped iterations: 0 | number of nan iterations: 0 |\n'
        )

    def test_exact_routes_do_not_imply_numerical_acceptance(self):
        self.write('reference', _metric())
        candidate = _metric()
        candidate['full_gradient_validation']['totals']['relative_l2'] = 0.4
        self.write('candidate', candidate)
        report = build_report(self.study, 'reference', ['candidate'])
        step = report['runs']['candidate']['steps']['0']
        self.assertTrue(step['exact_route_checks_pass'])
        self.assertFalse(report['overall_acceptance_claimed'])
        self.assertFalse(report['numerical_tolerance_inferred'])
        self.assertEqual(step['full_gradient']['totals']['relative_l2'], 0.4)
        self.assertAlmostEqual(step['full_gradient']['totals']['cosine_from_norms'], 0.92)
        self.assertEqual(step['native_iteration']['grad_norm'], 1.234)
        self.assertFalse(report['runs']['candidate']['optimizer']['available'])

    def test_missing_hashes_are_not_equal_evidence(self):
        metric = _metric()
        del metric['fixed_routing']
        self.write('reference', metric)
        self.write('candidate', metric)
        step = build_report(self.study, 'reference', ['candidate'])['runs']['candidate']['steps'][
            '0'
        ]
        self.assertFalse(step['exact_route_checks_pass'])
        self.assertFalse(step['route_hash']['equal'])

    def test_source_mtp_counts_or_route_changes_fail_exact_checks(self):
        self.write('reference', _metric())
        for field, value in [('mtp_tokens', [5]), ('source_images', 2)]:
            metric = _metric()
            metric[field] = value
            self.write('candidate', metric)
            step = build_report(self.study, 'reference', ['candidate'])['runs']['candidate'][
                'steps'
            ]['0']
            self.assertFalse(step['exact_route_checks_pass'])
            self.assertFalse(step['source_counts'][field]['equal'])
        metric = _metric()
        metric['fixed_routing']['route_sha256'] = 'b' * 64
        self.write('candidate', metric)
        self.assertFalse(
            build_report(self.study, 'reference', ['candidate'])['runs']['candidate']['steps']['0'][
                'exact_route_checks_pass'
            ]
        )

    def test_assignment_changes_cannot_hide_behind_equal_histograms(self):
        audit = _audit()
        audit['token_assignments']['dispatch_recompute']['different_rows'] = 2
        result = summarize_auxiliary(audit)
        self.assertFalse(result['exact_observed_assignments'])
        self.assertTrue(result['phases']['training_vs_recompute']['exact_histograms'])
        self.assertEqual(result['assignments']['dispatch_recompute']['different_rows'], 2)

    def test_missing_candidate_remains_incomplete(self):
        self.write('reference', _metric())
        report = build_report(self.study, 'reference', ['pending'])
        step = report['runs']['pending']['steps']['0']
        self.assertEqual(step['status'], 'incomplete')
        self.assertFalse(step['exact_route_checks_pass'])

    def test_incomplete_trailing_line_and_duplicate_steps(self):
        self.write('reference', _metric())
        self.write('candidate', _metric())
        path = self.study / 'logs/candidate-metrics/metrics.jsonl'
        complete = path.read_text()
        path.write_text(complete + '{"step":')
        report = build_report(self.study, 'reference', ['candidate'])
        self.assertIn(
            'Ignored incomplete final metrics line', report['runs']['candidate']['issues']
        )
        path.write_text(complete + complete)
        with self.assertRaisesRegex(ValueError, 'Duplicate training metrics'):
            build_report(self.study, 'reference', ['candidate'])

    def test_optimizer_preserves_actual_update_and_state_evidence(self):
        self.write('reference', _metric())
        self.write('candidate', _metric())
        report = dict(
            mode='compare',
            expected_step=0,
            reference_dir='/example/reference',
            initial_state_exact=True,
            full_update_elements=100,
            positive_lr_owned_elements=100,
            zero_lr_owned_elements=0,
            nonzero_master_update=True,
            metrics={'master_update': {'totals': {'relative_l2': 0.46, 'cosine': 0.89}}},
            ranks=[{'native_result': [True, 1.234, 0]}],
            tolerance_pass=None,
        )
        directory = self.study / 'analysis'
        directory.mkdir()
        (directory / 'candidate-optimizer.json').write_text(json.dumps(report))
        result = build_report(self.study, 'reference', ['candidate'])['runs']['candidate'][
            'optimizer'
        ]
        self.assertTrue(result['native_steps_succeeded'])
        self.assertTrue(result['initial_state_exact'])
        self.assertEqual(result['metrics']['master_update']['totals']['cosine'], 0.89)
        self.assertIsNone(result['reported_tolerance_pass'])

    def test_nonfinite_json_and_path_traversal_are_rejected(self):
        self.write('reference', _metric())
        path = self.study / 'logs/reference-metrics/metrics.jsonl'
        path.write_text(path.read_text().replace('0.4', 'NaN'))
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            build_report(self.study, 'reference', [])
        with self.assertRaisesRegex(ValueError, 'filename component'):
            build_report(self.study, '../reference', [])


if __name__ == '__main__':
    unittest.main()
