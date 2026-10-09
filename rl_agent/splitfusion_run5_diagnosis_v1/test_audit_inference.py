"""CPU-only regression tests for the read-only diagnostic, not a pipeline fix."""
import json
import unittest

from . import audit_inference as audit


class InferenceAuditTests(unittest.TestCase):
    def test_literal_preprocessing_paths_have_equal_shape_not_equal_content(self):
        result = audit.preprocessing_probe()
        self.assertEqual(result["reference_shape"], result["live_shape"])
        self.assertEqual(result["reference_bottom16_nonzero"], 0)
        self.assertEqual(result["live_bottom16_nonzero"], 7 * 16 * 768)
        self.assertFalse(result["tensor_equal"])
        self.assertFalse(result["cuda_initialized"])
        self.assertAlmostEqual(result["vertical_scale_difference_percent"], 100 * 16 / 432)

    def test_registered_arm_uses_production_packing(self):
        registration = json.loads((audit.CAP / "registration.json").read_text())
        self.assertEqual(registration["arms"]["C2"]["bundle"], "production")
        text = (audit.WT / "rl_agent/splitfusion_single_gpu_profiler_v1/child_v1_threadcap.py").read_text()
        self.assertIn("if bundle == RB.OPTIMIZED:", text)
        self.assertIn("RB.install_packing_short_circuit(live_base)", text)

    def test_saved_audit_reproduces_exactly(self):
        output = audit.ROOT / "experiments/splitfusion_run5_diagnosis_v1/20261003_v1/inference_audit.json"
        self.assertEqual(json.loads(output.read_text()), audit.run())


if __name__ == "__main__":
    unittest.main()
