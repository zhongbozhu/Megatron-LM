# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Small synthetic trajectory checks, requiring only the Python standard library."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from examples.mimo.fixed_routing_training_report import build_training_report, main


def _metric():
    histogram = dict(pairs=1, missing=0, unexpected=0, histogram_mismatches=0, histogram_l1=0)
    assignments = {
        name: dict(pairs=1, mismatched_pairs=0, different_rows=0, shape_mismatches=0)
        for name in (
            'auxiliary_training',
            'dispatch_training',
            'auxiliary_recompute',
            'dispatch_recompute',
        )
    }
    return dict(
        training=True,
        source_samples=2,
        source_images=1,
        source_tokens=10,
        supervised_tokens=8,
        mtp_tokens=[7],
        mtp_loss=[0.5],
        aux=dict(loss=0.01),
        fixed_routing=dict(
            source_samples=2, source_tokens=10, routers=2, mtp_routers=1, route_sha256='a' * 64
        ),
        aux_replay=dict(
            **histogram,
            recompute=histogram.copy(),
            token_assignments=dict(enabled=True, **assignments),
        ),
    )


class TestFixedRoutingTrainingReport(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.study = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, name, *, missing_step=None, mutation=None):
        directory = self.study / 'logs' / f'{name}-metrics'
        directory.mkdir(parents=True, exist_ok=True)
        records, lines = [], []
        for step in range(50):
            if step == missing_step:
                continue
            record = _metric()
            record.update(step=step, loss=0.4 - step / 1000)
            record['fixed_routing'].update(source_step=step, source_sha256=f'{step:064x}')
            if mutation:
                mutation(step, record)
            records.append(record)
            lines.append(
                f'iteration {step + 1}/ 50 | grad norm: 1.234 | '
                'number of skipped iterations: 0 | number of nan iterations: 0 |\n'
            )
        for step in range(3):
            record = _metric()
            record.update(step=step, training=False, loss=0.3 - step / 100)
            record['fixed_routing'].update(source_step=step, source_sha256=f'{step + 100:064x}')
            records.append(record)
        (directory / 'metrics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
        (self.study / 'logs' / f'{name}.out').write_text(''.join(lines))
        checkpoints = self.study / 'checkpoints' / name
        for iteration in (25, 50):
            path = checkpoints / f'iter_{iteration:07d}'
            path.mkdir(parents=True, exist_ok=True)
            (path / '.metadata').write_bytes(b'test metadata, not a real checkpoint')
            (path / '__0_0.distcp').write_bytes(b'test shard')
        (checkpoints / 'latest_checkpointed_iteration.txt').write_text('50\n')

    def trajectory(self, name='candidate'):
        return build_training_report(self.study, 'reference', [name])['runs'][name]['trajectory']

    def test_complete_trajectory_does_not_claim_numerical_or_resume_acceptance(self):
        self.write('reference')
        self.write('candidate')
        report = build_training_report(self.study, 'reference', ['candidate'])
        trajectory = report['runs']['candidate']['trajectory']
        self.assertTrue(trajectory['mechanical_checks_pass'])
        self.assertEqual(trajectory['successful_native_iterations'], 50)
        self.assertEqual(trajectory['evaluation']['observed_count'], 3)
        self.assertFalse(trajectory['checkpoints']['checkpoint_load_or_resume_tested'])
        self.assertFalse(report['overall_acceptance_claimed'])
        self.assertFalse(report['numerical_tolerance_inferred'])

    def test_missing_middle_step_cannot_pass_with_matching_final_iteration(self):
        self.write('reference')
        self.write('candidate', missing_step=20)
        result = self.trajectory()
        self.assertFalse(result['mechanical_checks_pass'])
        self.assertEqual(result['missing_steps'], [20])
        self.assertEqual(result['successful_native_iterations'], 49)

    def test_source_hash_must_be_observed_and_match(self):
        self.write('reference')
        for value in (None, 'f' * 64):

            def mutate(step, record):
                if step == 10:
                    record['fixed_routing']['source_sha256'] = value

            self.write('candidate', mutation=mutate)
            result = self.trajectory()
            self.assertFalse(result['checks']['all_source_hashes_exact'])
            self.assertTrue(result['checks']['all_route_audits_exact'])
            self.assertFalse(result['mechanical_checks_pass'])

    def test_assignment_failure_is_independent_of_loss_finiteness(self):
        self.write('reference')

        def mutate(step, record):
            if step == 12:
                record['aux_replay']['token_assignments']['dispatch_recompute'][
                    'different_rows'
                ] = 1

        self.write('candidate', mutation=mutate)
        result = self.trajectory()
        self.assertTrue(result['checks']['all_losses_and_norms_finite'])
        self.assertFalse(result['checks']['all_route_audits_exact'])

    def test_nan_logged_norm_and_skipped_native_step_are_not_success(self):
        self.write('reference')
        self.write('candidate')
        path = self.study / 'logs/candidate.out'
        text = path.read_text().replace('grad norm: 1.234', 'grad norm: nan', 1)
        text = text.replace('number of skipped iterations: 0', 'number of skipped iterations: 1', 1)
        path.write_text(text)
        result = self.trajectory()
        self.assertFalse(result['checks']['all_losses_and_norms_finite'])
        self.assertEqual(result['successful_native_iterations'], 49)

    def test_missing_mtp_loss_is_not_silently_ignored(self):
        self.write('reference')

        def mutate(step, record):
            if step == 2:
                record['mtp_loss'] = []

        self.write('candidate', mutation=mutate)
        self.assertFalse(self.trajectory()['checks']['all_losses_and_norms_finite'])

    def test_checkpoint_metadata_alone_does_not_establish_presence(self):
        self.write('reference')
        self.write('candidate')
        (self.study / 'checkpoints/candidate/iter_0000050/__0_0.distcp').unlink()
        result = self.trajectory()
        self.assertFalse(result['checks']['checkpoint_files_present'])
        self.assertTrue(result['checks']['all_native_iterations_successful'])

    def test_evaluation_has_separate_identity_and_expected_coverage(self):
        self.write('reference')
        self.write('candidate')
        path = self.study / 'logs/candidate-metrics/metrics.jsonl'
        records = [json.loads(line) for line in path.read_text().splitlines()]
        records[-1]['fixed_routing']['source_sha256'] = 'f' * 64
        path.write_text(''.join(json.dumps(r) + '\n' for r in records))
        result = self.trajectory()
        self.assertFalse(result['checks']['evaluation_source_and_routes_exact'])
        self.assertTrue(result['checks']['all_source_hashes_exact'])
        path.write_text(''.join(json.dumps(r) + '\n' for r in records[:-1]))
        self.assertFalse(self.trajectory()['checks']['evaluations_complete'])

    def test_cli_fails_after_preserving_report_and_accepts_complete_reference_gate(self):
        self.write('reference')
        self.write('candidate', missing_step=20)
        output = self.study / 'analysis/failed.json'
        argv = [
            'fixed_routing_training_report',
            '--study',
            str(self.study),
            '--reference',
            'reference',
            '--runs',
            'candidate',
            '--output',
            str(output),
        ]
        with patch('sys.argv', argv), self.assertRaisesRegex(SystemExit, 'checks failed'):
            main()
        self.assertTrue(output.is_file())
        report = json.loads(output.read_text())
        self.assertFalse(report['runs']['candidate']['trajectory']['mechanical_checks_pass'])
        argv[argv.index('--runs') + 1] = 'reference'
        argv[-1] = str(self.study / 'analysis/reference-gate.json')
        with patch('sys.argv', argv):
            main()
        report = json.loads(Path(argv[-1]).read_text())
        self.assertTrue(report['runs']['reference']['trajectory']['mechanical_checks_pass'])


if __name__ == '__main__':
    unittest.main()
