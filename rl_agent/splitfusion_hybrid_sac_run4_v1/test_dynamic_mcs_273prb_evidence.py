from __future__ import annotations

import copy
import dataclasses
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rl_agent.splitfusion_hybrid_sac_run4_v1 import (
    dynamic_mcs_273prb_evidence as subject,
)


class DynamicMcs273PrbEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.run_dir = cls.root / subject.SOURCE_RUN_RELATIVE_PATH
        cls.evidence = subject.load_dynamic_mcs_273prb_evidence()
        cls.observation_rows = subject._read_csv(
            cls.run_dir / subject.OBSERVATIONS_NAME, subject.OBSERVATION_FIELDS
        )
        cls.verified_observations = subject._parse_observations(
            cls.observation_rows
        )
        cls.transition_rows = subject._read_csv(
            cls.run_dir / subject.TRANSITIONS_NAME, subject.TRANSITION_FIELDS
        )

    def test_real_binding_hashes_inventory_and_canonical_digest(self) -> None:
        evidence = self.evidence
        self.assertEqual(
            evidence.source_manifest_sha256, subject.EXPECTED_MANIFEST_SHA256
        )
        self.assertEqual(
            evidence.source_terminal_sha256,
            subject.EXPECTED_SUCCESS_TERMINAL_SHA256,
        )
        self.assertEqual(
            evidence.source_analysis_sha256, subject.EXPECTED_ANALYSIS_SHA256
        )
        self.assertEqual(
            evidence.source_observations_sha256,
            subject.EXPECTED_OBSERVATIONS_SHA256,
        )
        self.assertEqual(
            evidence.source_transitions_sha256,
            subject.EXPECTED_TRANSITIONS_SHA256,
        )
        self.assertEqual(
            evidence.source_inventory_sha256,
            "5f504f2c9a99d5decd7f6d3d9e13d90251668f944c6a158fd175cb22b246e746",
        )
        self.assertEqual(
            evidence.canonical_evidence_sha256,
            subject.EXPECTED_CANONICAL_EVIDENCE_SHA256,
        )

    def test_exact_population_and_disjoint_frozen_splits(self) -> None:
        self.assertEqual(
            [len(item.observations) for item in self.evidence.fit_sequences],
            [210, 210],
        )
        self.assertEqual(
            [len(item.observations)
             for item in self.evidence.internal_validation_sequences],
            [90, 90],
        )
        self.assertEqual(
            [len(item.transitions) for item in self.evidence.fit_sequences],
            [208, 208],
        )
        self.assertEqual(
            [len(item.transitions)
             for item in self.evidence.internal_validation_sequences],
            [88, 88],
        )
        self.assertEqual(len(self.evidence.fit_transitions), 416)
        self.assertEqual(
            len(self.evidence.internal_validation_transitions), 176
        )

    def test_model_surface_has_no_hidden_metadata(self) -> None:
        observation_fields = {
            item.name for item in dataclasses.fields(subject.PolicyMcsObservationV1)
        }
        transition_fields = {
            item.name for item in dataclasses.fields(subject.PolicyMcsTransitionV1)
        }
        self.assertEqual(
            observation_fields, {"status", "prior_ul_mcs_index"}
        )
        self.assertEqual(
            transition_fields,
            {"current", "successor", "duration_tensors", "learning_eligible"},
        )
        forbidden = (
            "profile", "snr", "gnb", "timestamp", "frame", "decision",
            "trace", "provenance", "rnti", "partition",
        )
        text = json.dumps(
            self.evidence.fit_transitions[0].to_canonical_dict(), sort_keys=True
        ).lower()
        for name in forbidden:
            self.assertNotIn(name, text)
        self.assertEqual(
            self.evidence.fit_transitions[0].current.policy_features(),
            {"prior_ul_mcs_index": 28},
        )

    def test_every_bound_transition_is_duration_two_and_eligible(self) -> None:
        transitions = (
            self.evidence.fit_transitions
            + self.evidence.internal_validation_transitions
        )
        self.assertEqual(len(transitions), 592)
        self.assertTrue(all(item.duration_tensors == 2 for item in transitions))
        self.assertTrue(all(item.learning_eligible for item in transitions))
        self.assertTrue(all(
            0 <= value <= 28
            for item in transitions
            for value in item.policy_values()
        ))

    def test_missing_and_stale_are_explicit_and_never_imputed(self) -> None:
        missing = subject._parse_status(
            "MISSING_NO_PRIOR_GRANT", "", "synthetic"
        )
        stale = subject._parse_status("STALE", "", "synthetic")
        for item, expected in (
            (missing, subject.McsStatus.MISSING_NO_PRIOR_GRANT),
            (stale, subject.McsStatus.STALE),
        ):
            self.assertIs(item.status, expected)
            self.assertIsNone(item.prior_ul_mcs_index)
            with self.assertRaises(subject.PolicyFeatureUnavailable):
                item.policy_features()
        with self.assertRaises(subject.EvidenceSchemaError):
            subject._parse_status("STALE", "0", "synthetic")

    def test_wrong_duration_is_rejected_by_record(self) -> None:
        valid = subject.PolicyMcsObservationV1(subject.McsStatus.VALID, 9)
        with self.assertRaisesRegex(subject.EvidenceSchemaError, "duration=2"):
            subject.PolicyMcsTransitionV1(valid, valid, 1, True)

    def test_wrong_duration_is_rejected_in_source_table(self) -> None:
        rows = [dict(item) for item in self.transition_rows]
        rows[0]["duration_tensors"] = "3"
        with self.assertRaisesRegex(subject.EvidenceIdentityError, "duration"):
            subject._parse_transitions(rows, self.verified_observations)

    def test_fit_validation_boundary_crossing_is_rejected(self) -> None:
        rows = [dict(item) for item in self.transition_rows]
        candidate = next(
            row for row in rows
            if row["profile_id"] == "MID_VARIABLE"
            and row["current_decision_index"] == "207"
        )
        candidate["current_decision_index"] = "208"
        candidate["successor_decision_index"] = "210"
        with self.assertRaisesRegex(
            subject.EvidenceIdentityError, "foreign/duplicate|boundary"
        ):
            subject._parse_transitions(rows, self.verified_observations)

    def test_hidden_metadata_in_policy_json_is_rejected(self) -> None:
        rows = [dict(item) for item in self.observation_rows]
        rows[0]["policy_feature_json"] = json.dumps({
            "prior_ul_mcs_index": 28,
            "profile_id": "MID_VARIABLE",
        })
        with self.assertRaisesRegex(
            subject.EvidenceIdentityError, "hidden/imputed"
        ):
            subject._parse_observations(rows)

    def test_source_partition_relabel_is_rejected(self) -> None:
        rows = [dict(item) for item in self.observation_rows]
        rows[210]["partition"] = "FIT"
        with self.assertRaisesRegex(subject.EvidenceIdentityError, "partition"):
            subject._parse_observations(rows)

    def test_manifest_radio_drift_is_rejected_semantically(self) -> None:
        manifest = subject._load_json(self.run_dir / subject.MANIFEST_NAME)
        changed = copy.deepcopy(manifest)
        changed["radio_profile_id"] = "LEGACY_106PRB"
        with self.assertRaisesRegex(subject.EvidenceIdentityError, "radio"):
            subject._validate_manifest(changed)

    def test_failed_analysis_gate_is_rejected(self) -> None:
        manifest = subject._load_json(self.run_dir / subject.MANIFEST_NAME)
        analysis = copy.deepcopy(manifest["analysis"])
        analysis["aggregate_gates"]["all_profile_gates"] = False
        changed_manifest = copy.deepcopy(manifest)
        changed_manifest["analysis"] = analysis
        with self.assertRaisesRegex(subject.EvidenceIdentityError, "gates"):
            subject._validate_analysis(analysis, changed_manifest)

    def test_core_artifact_tampering_is_rejected(self) -> None:
        source = self.run_dir / subject.OBSERVATIONS_NAME
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "mcs_observations.csv"
            target.write_bytes(source.read_bytes() + b"tamper")
            with self.assertRaises(subject.EvidenceHashMismatch):
                subject._verify_pinned(
                    target, subject.EXPECTED_OBSERVATIONS_SHA256
                )

    def test_hash_failure_precedes_csv_parsing(self) -> None:
        real = subject._sha256_file

        def poisoned(path: Path) -> str:
            if path.name == "mcs_observations.csv":
                return "0" * 64
            return real(path)

        with mock.patch.object(subject, "_sha256_file", side_effect=poisoned):
            with mock.patch.object(
                subject, "_read_csv", side_effect=AssertionError("CSV parsed")
            ):
                with self.assertRaises(subject.EvidenceHashMismatch):
                    subject.load_dynamic_mcs_273prb_evidence()

    def test_import_has_no_evidence_io_cuda_or_runtime_side_effect(self) -> None:
        module = (
            "rl_agent.splitfusion_hybrid_sac_run4_v1."
            "dynamic_mcs_273prb_evidence"
        )
        probe = f'''\
import json, sys
violations=[]
def audit(event,args):
    try:
        if event == "open" and "20260924_target_radio_capture_v1_retry3" in str(args[0]):
            violations.append([event,str(args[0])])
        elif event in ("subprocess.Popen","os.system","os.exec","os.posix_spawn","socket.socket","socket.connect"):
            violations.append([event,str(args)[:100]])
    except Exception:
        pass
sys.addaudithook(audit)
import {module} as subject
assert "torch" not in sys.modules
print("VIOLATIONS:"+json.dumps(violations))
'''
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        marker = next(
            line for line in completed.stdout.splitlines()
            if line.startswith("VIOLATIONS:")
        )
        self.assertEqual(json.loads(marker.removeprefix("VIOLATIONS:")), [])


if __name__ == "__main__":
    unittest.main()
