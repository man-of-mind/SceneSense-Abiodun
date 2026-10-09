import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import run_diagnostic as D


class ConfigTests(unittest.TestCase):
    def test_create_only_budget_and_unchanged_semantics(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(D.E, 'execute'):
            root=Path(directory)/'diagnostic'
            D.prepare(root)
            registration=json.loads((root/'treatment/registration.json').read_text())
            evidence=json.loads((root/'PROSPECTIVE.json').read_text())
            source=Path(evidence['selection']['evidence_root'])/'treatment/registration.json'
            previous=json.loads(source.read_text())
            for key in ('policy','semantics','profiler','lifecycle','instrumentation_files','binds'):
                self.assertEqual(registration[key],previous[key],key)
            for key in ('env','bundle','ordering'):
                self.assertEqual(registration['arms']['C2'][key],previous['arms']['C2'][key])
            self.assertEqual(registration['transmitted_budget'],300)
            self.assertEqual(registration['g1_boundary_budget'],300)
            self.assertTrue(all(D.E.sha(p)==h for p,h in evidence['sources'].items()))
            digest=registration.pop('registration_sha256')
            self.assertEqual(digest,D.E.canonical(registration))
            with self.assertRaises(FileExistsError):D.prepare(root)

    def test_gpu_hidden_environment_refused(self):
        with patch.dict(D.os.environ, {'CUDA_VISIBLE_DEVICES':''}):
            with self.assertRaises(AssertionError):D.environment(Path('/tmp/test'))


if __name__=='__main__':unittest.main()
