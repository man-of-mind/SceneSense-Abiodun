from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from . import mcs_transition_acceptance as subject


class McsTransitionAcceptanceTests(unittest.TestCase):
    def test_report_is_held_out_finite_and_beats_persistence(self) -> None:
        report = subject.build_report()
        metrics = report["validation_metrics"]
        self.assertTrue(report["accepted_for_offline_run4_mcs_dynamics"])
        self.assertFalse(report["production_or_deployment_authorized"])
        self.assertEqual(report["fit_transition_count"], 416)
        self.assertEqual(report["validation_transition_count"], 176)
        self.assertGreater(
            metrics["top1_accuracy"], metrics["persistence_accuracy"]
        )
        self.assertTrue(report["acceptance_checks"]["finite_brier_and_nll"])

    def test_create_only_and_registered_artifact_recompute(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "mcs"
            result_path, manifest_path = subject.write_create_only(output)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["result"]["sha256"], subject._sha256(result_path)
            )
            with self.assertRaises(FileExistsError):
                subject.write_create_only(output)
        registered = subject.load_registered_acceptance()
        self.assertEqual(
            registered["model_binding_sha256"],
            registered["validation_metrics"]["model_binding_sha256"],
        )

    def test_hash_pin_refuses_a_different_registered_result(self) -> None:
        with mock.patch.object(
            subject, "REGISTERED_MCS_ACCEPTANCE_RESULT_SHA256", "0" * 64
        ):
            with self.assertRaisesRegex(subject.McsAcceptanceError, "not the registered"):
                subject.load_registered_acceptance()


if __name__ == "__main__":
    unittest.main()
