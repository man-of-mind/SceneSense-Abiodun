"""Offline contract and adversarial tests for the robust bracket amendment."""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from rl_agent.ue_mcs_backlog_robust_bracket_v1 import amendment as A
from rl_agent.ue_mcs_backlog_robust_bracket_v1 import selector as S


class RobustBracketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = A.load_config(A.ROOT)
        cls.catalogue, _ = A.load_catalogue(A.ROOT, cls.config)
        cls.selection = S.select_robust_bracket(cls.catalogue)
        cls.sealed = A.DEFAULT_SEALED_DIR

    def test_exact_ae64_uint8_selection(self):
        self.assertEqual(
            self.selection["selected_action_ids"],
            {"low": 40, "medium": 39, "high": 38})
        self.assertEqual(
            self.selection["selected_payload_bytes"],
            {"low": 126237, "medium": 374264, "high": 619563})
        self.assertEqual(
            self.selection["selected_offered_mbps"],
            {"low": 10.09896, "medium": 29.94112, "high": 49.56504})
        identity = self.selection["winning_group_identity"]
        self.assertEqual((identity["family"], identity["quantizer"]),
                         ("AE64", "UINT8"))
        self.assertIsNotNone(identity["wire_identity"])
        self.assertEqual(identity["wire_identity"]["codec_id"], 2)
        tiers = self.selection["selected_tiers"]
        self.assertTrue(all(row["source_contract_tier"] == "EMERGENCY_ONLY"
                            for row in tiers))
        self.assertFalse(self.selection["perception_endorsement"])

    def test_robust_thresholds_and_complete_candidate_digest(self):
        rule = self.selection["rule"]
        self.assertEqual(rule["tier_admissibility"]["low"]["maximum_mbps"],
                         25.6608)
        self.assertEqual(rule["tier_admissibility"]["high"]["minimum_mbps"],
                         34.7424)
        self.assertEqual(
            self.selection["candidate_universe_sha256"],
            S.canonical_sha256(self.selection["candidate_universe"]))
        self.assertFalse(self.selection["cross_group_mixing_used"])
        self.assertEqual(
            self.selection["uncertainty_set"]["interpretation"],
            S.UNCERTAINTY_INTERPRETATION)
        self.assertFalse(
            self.selection["uncertainty_set"]["population_confidence_interval"])

    def test_candidate_identity_drift_is_refused(self):
        catalogue = copy.deepcopy(self.catalogue)
        row = next(item for item in catalogue["profiles"]
                   if item["action_id"] == 39)
        row["decoder_identity"] = "TAMPERED_DECODER"
        with self.assertRaisesRegex(S.RobustSelectionError,
                                    "no longer selects|registered result"):
            S.select_robust_bracket(catalogue)

    def test_original_refusal_semantics_guard(self):
        result_path = (A.ROOT / A.EXPECTED_ATTEMPT_RELPATH
                       / A.EXPECTED_BINDINGS["result"][0])
        result = json.loads(result_path.read_text())
        accepted = A.validate_original_refusal_document(result)
        self.assertEqual(accepted["status"], A.ORIGINAL_STATUS)
        mutations = (
            ("status", "CAPACITY_QUALIFICATION_CAPTURED"),
            ("qualified", True),
            ("selected_tiers", [{"action_id": 39}]),
        )
        for key, value in mutations:
            with self.subTest(key=key):
                changed = copy.deepcopy(result)
                changed[key] = value
                with self.assertRaises(A.AmendmentError):
                    A.validate_original_refusal_document(changed)
        changed = copy.deepcopy(result)
        changed["audit"]["problems"].append("another problem")
        with self.assertRaisesRegex(A.AmendmentError, "sole exact"):
            A.validate_original_refusal_document(changed)

    def test_old_captured_only_verifier_still_rejects(self):
        proof = A.prove_old_verifier_rejects(A.ROOT)
        self.assertTrue(proof["rejected"])
        self.assertFalse(proof["original_verifier_modified"])

    def test_create_only_builder_and_offline_verifier(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "sealed"
            paths = A.build_amendment(output, repo_root=A.ROOT)
            verified = A.verify_amendment(paths["amendment"], repo_root=A.ROOT)
            self.assertTrue(verified["binding_verified"])
            self.assertTrue(
                verified["preserved_attempt"]["physical_gates"]
                ["final_cold_state_verified"])
            with self.assertRaisesRegex(A.AmendmentError, "create-only"):
                A.build_amendment(output, repo_root=A.ROOT)

    def test_checked_sealed_artifact(self):
        verified = A.verify_amendment(
            self.sealed / A.AMENDMENT_FILENAME, repo_root=A.ROOT)
        self.assertEqual(verified["status"], A.STATUS)
        self.assertEqual(
            verified["adoption"]["selected_action_ids"], S.EXPECTED_ACTION_IDS)

    @staticmethod
    def _resign(root: Path) -> None:
        amendment_path = root / A.AMENDMENT_FILENAME
        manifest_path = root / A.MANIFEST_FILENAME
        terminal_path = root / A.TERMINAL_FILENAME
        amendment = json.loads(amendment_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        terminal = json.loads(terminal_path.read_text())
        manifest["amendment_sha256"] = A.sha256_file(amendment_path)
        manifest["source_inventory_sha256"] = amendment[
            "source_inventory"]["inventory_sha256"]
        manifest["candidate_universe_sha256"] = amendment[
            "selector"]["candidate_universe_sha256"]
        manifest["files"] = A._file_inventory(
            root, excluded=(A.MANIFEST_FILENAME, A.TERMINAL_FILENAME))
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        terminal["amendment_sha256"] = A.sha256_file(amendment_path)
        terminal["manifest_sha256"] = A.sha256_file(manifest_path)
        terminal["source_inventory_sha256"] = amendment[
            "source_inventory"]["inventory_sha256"]
        terminal["candidate_universe_sha256"] = amendment[
            "selector"]["candidate_universe_sha256"]
        terminal_path.write_text(
            json.dumps(terminal, indent=2, sort_keys=True) + "\n")

    def _mutated_copy(self, mutation) -> Path:
        temporary = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, temporary)
        root = temporary / "sealed"
        shutil.copytree(self.sealed, root)
        path = root / A.AMENDMENT_FILENAME
        value = json.loads(path.read_text())
        mutation(value)
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        return root

    def test_unsigned_artifact_tamper_is_refused(self):
        root = self._mutated_copy(
            lambda value: value["adoption"].update({"selected_family": "AE32"}))
        with self.assertRaisesRegex(A.AmendmentError, "digest"):
            A.verify_amendment(root / A.AMENDMENT_FILENAME, repo_root=A.ROOT)

    def test_resigned_source_tamper_is_refused(self):
        root = self._mutated_copy(
            lambda value: value["source_inventory"].update(
                {"inventory_sha256": "0" * 64}))
        self._resign(root)
        with self.assertRaisesRegex(A.AmendmentError, "source inventory"):
            A.verify_amendment(root / A.AMENDMENT_FILENAME, repo_root=A.ROOT)

    def test_resigned_candidate_tamper_is_refused(self):
        root = self._mutated_copy(
            lambda value: value["selector"]["candidate_universe"].pop())
        self._resign(root)
        with self.assertRaisesRegex(A.AmendmentError,
                                    "candidate universe|selection rule"):
            A.verify_amendment(root / A.AMENDMENT_FILENAME, repo_root=A.ROOT)

    def test_resigned_rule_tamper_is_refused(self):
        root = self._mutated_copy(
            lambda value: value["selector"]["rule"].update(
                {"exterior_margin_fraction": 0.01}))
        self._resign(root)
        with self.assertRaisesRegex(A.AmendmentError,
                                    "candidate universe|selection rule"):
            A.verify_amendment(root / A.AMENDMENT_FILENAME, repo_root=A.ROOT)

    def test_resigned_original_refusal_tamper_is_refused(self):
        root = self._mutated_copy(
            lambda value: value["preserved_attempt"]["original_disposition"].update(
                {"qualified": True}))
        self._resign(root)
        with self.assertRaisesRegex(A.AmendmentError, "original refusal"):
            A.verify_amendment(root / A.AMENDMENT_FILENAME, repo_root=A.ROOT)


if __name__ == "__main__":
    unittest.main()

