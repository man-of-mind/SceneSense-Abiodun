import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import verify_retained_selection as V


class SelectionTests(unittest.TestCase):
    def altered(self, key, value):
        config = json.loads(V.DEFAULT.read_text())
        config[key] = value
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'selection.json'
            filename.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                V.verify(filename)

    def test_real_accepted_configuration_without_launch_or_affinity(self):
        with patch('subprocess.Popen', side_effect=AssertionError('No launch')), \
             patch('os.sched_setaffinity', side_effect=AssertionError('No mutation')):
            result = V.verify()
        self.assertEqual(result['status'], 'RETAINED_CONFIGURATION_VERIFIED')
        self.assertFalse(result['launch_performed'])

    def test_refuse_control(self):
        self.altered('selected_arm', 'control')

    def test_refuse_foreign_manifest(self):
        self.altered('evidence_manifest_sha256', '0' * 64)

    def test_refuse_changed_cpu_pool(self):
        self.altered('edge_cpus', [4, 5, 6, 7])

    def test_refuse_changed_scientific_or_ownership_setting(self):
        for key, value in [('reward_deadline_ms', 200), ('quality_feedback_inside_deadline', False),
                           ('early_evaluator_submit', True), ('radio_priority_change_allowed', True),
                           ('restore_before_teardown', False), ('ue_omp_num_threads', 24)]:
            with self.subTest(key=key):
                self.altered(key, value)

    def test_refuse_bound_evidence_drift(self):
        original = V.sha
        def changed(path):
            return '0' * 64 if path.name == 'RESULT.json' else original(path)
        with patch.object(V, 'sha', side_effect=changed), self.assertRaises(ValueError):
            V.verify()

    def test_refuse_executed_source_drift(self):
        original = V.sha
        def changed(path):
            return '0' * 64 if path.name == 'cpu_partition.py' else original(path)
        with patch.object(V, 'sha', side_effect=changed), self.assertRaises(ValueError):
            V.verify()


if __name__ == '__main__':
    unittest.main()
