import copy
import json
import tempfile
import unittest
from pathlib import Path

from rl_agent import ue_n3b_cold_attach_outcome_review_v1 as review


CONFIG_PATH = review.DEFAULT_CONFIG


class N3BColdAttachOutcomeReviewTests(unittest.TestCase):
    def setUp(self):
        self.config = review.load_json(CONFIG_PATH)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = review.resolve_repo_path(self.config["source"]["directory"])

    def source_rows(self):
        rows = []
        for spec in self.config["expected_repetitions"]:
            rep = self.source / "repetitions" / spec["directory"]
            rows.append(review.adjudicate_repetition(
                review.load_json(rep / "repetition_summary.json"),
                review.load_json(rep / "cleanup_report.json"),
                review.load_json(rep / "startup_channel_runtime_gate.json"),
                review.load_json(rep / "source_oai_config_integrity.json"),
                self.config["contract"],
            ))
        return rows

    def test_config_is_offline_and_runtime_sealed(self):
        review.validate_config(self.config, verify_hashes=True)
        authority = self.config["authority"]
        self.assertTrue(authority["offline_review_authorized"])
        for key in (
            "oai_run_authorized", "socket_execution_authorized",
            "carla_run_authorized", "n3c_execution_authorized",
            "target_mapping_promotion_authorized", "numeric_bound_promotion_authorized",
            "operational_bound_promotion_authorized",
            "connectivity_bound_promotion_authorized",
            "usable_service_bound_promotion_authorized",
        ):
            self.assertFalse(authority[key])

    def test_sealed_campaign_nested_repetitions_and_ladder_verify_read_only(self):
        watched = (
            self.source / "manifest.json",
            self.source / "campaign_summary.json",
            self.source / "resolved_config.json",
            review.resolve_repo_path(self.config["command_ladder"]["directory"])
            / "manifest.json",
        )
        before = {path: (path.stat().st_mtime_ns, review.sha256(path)) for path in watched}
        result = review.verify_source(self.config)
        self.assertEqual(result["campaign_manifest_output_count"], 71)
        self.assertEqual(result["verified_repetition_count"], 3)
        self.assertEqual(result["unique_ran_epoch_count"], 3)
        self.assertEqual(result["unique_control_session_count"], 3)
        self.assertTrue(all(
            row["valid_cold_attach_failure_evidence"]
            and row["manifest_output_count"] == 20
            for row in result["repetitions"]
        ))
        ladder = result["command_ladder_expectation_provenance"]
        self.assertEqual(ladder["campaign_manifest_output_count"], 529)
        self.assertEqual(ladder["rung_manifest_output_count"], 103)
        self.assertEqual(ladder["commanded_noise_power_db"], -3.0)
        self.assertEqual(ladder["hot_observed_pusch_snr_db_p50"], 6.5)
        self.assertFalse(ladder["cold_attach_evidence"])
        after = {path: (path.stat().st_mtime_ns, review.sha256(path)) for path in watched}
        self.assertEqual(before, after)

    def test_actual_n3b_outcome_is_zero_of_three_with_null_cold_snr(self):
        rows = self.source_rows()
        self.assertEqual(
            [row["adjudicated_outcome"] for row in rows],
            ["VALID_COLD_ATTACH_FAILURE"] * 3,
        )
        self.assertTrue(all(row["cold_attach_pass"] is False for row in rows))
        self.assertTrue(all(row["cold_achieved_snr_is_null"] for row in rows))
        ladder = review.verify_source(self.config)[
            "command_ladder_expectation_provenance"
        ]
        aggregate = review.aggregate_adjudications(
            rows, self.config["contract"], ladder
        )
        self.assertEqual(aggregate["status"], review.SUCCESS)
        self.assertTrue(aggregate["n3b_outcome_accepted"])
        self.assertEqual(aggregate["cold_attach_passes"], 0)
        self.assertEqual(aggregate["cold_attach_failures"], 3)
        self.assertIsNone(aggregate["cold_achieved_pusch_snr_db_p50"])
        self.assertEqual(
            aggregate["cold_achieved_snr_status"],
            "UNOBSERVED_NO_SERVING_RNTI_PUSCH_WINDOW",
        )
        self.assertEqual(aggregate["n3c_selected_command_db"], -3.0)

    def test_commanded_and_achieved_values_cannot_be_conflated(self):
        rows = self.source_rows()
        for row in rows:
            self.assertEqual(row["commanded_noise_power_db"], -2.5)
            self.assertIsNone(row["cold_achieved_pusch_snr_db_p05"])
            self.assertIsNone(row["cold_achieved_pusch_snr_db_p50"])
            self.assertIsNone(row["cold_achieved_pusch_snr_db_p95"])
            self.assertFalse(row["physical_rf_cutoff_established"])

    def test_each_valid_failure_requires_all_integrity_gates(self):
        spec = self.config["expected_repetitions"][0]
        rep = self.source / "repetitions" / spec["directory"]
        base = review.load_json(rep / "repetition_summary.json")
        cleanup = review.load_json(rep / "cleanup_report.json")
        startup = review.load_json(rep / "startup_channel_runtime_gate.json")
        integrity = review.load_json(rep / "source_oai_config_integrity.json")
        mutations = (
            ("core", "summary", lambda value: value["attach_gate"].update({
                "core_ready_at_terminal": False,
            })),
            ("candidate_count", "summary", lambda value: value.update({
                "candidate_application_count": 1,
            })),
            ("cold_snr", "summary", lambda value: value.update({
                "achieved_pusch_snr_db_p50": 6.0,
            })),
            ("cleanup", "cleanup", lambda value: value.update({"clean": False})),
            ("startup", "startup", lambda value: value["models"]
             ["rfsimu_channel_ue0"].update({"noise_power_db": -3.0})),
            ("source_integrity", "integrity", lambda value: value.update({
                "unchanged": False,
            })),
        )
        originals = {
            "summary": base, "cleanup": cleanup,
            "startup": startup, "integrity": integrity,
        }
        for label, target, mutate in mutations:
            with self.subTest(label=label):
                changed = {key: copy.deepcopy(value) for key, value in originals.items()}
                mutate(changed[target])
                row = review.adjudicate_repetition(
                    changed["summary"], changed["cleanup"],
                    changed["startup"], changed["integrity"],
                    self.config["contract"],
                )
                self.assertFalse(row["evidence_valid"])
                self.assertEqual(row["adjudicated_outcome"], "INVALID_EVIDENCE")

    def test_invalid_or_mixed_evidence_cannot_select_n3c(self):
        rows = self.source_rows()
        changed = copy.deepcopy(rows)
        changed[0]["evidence_valid"] = False
        changed[0]["adjudicated_outcome"] = "INVALID_EVIDENCE"
        ladder = review.verify_source(self.config)[
            "command_ladder_expectation_provenance"
        ]
        result = review.aggregate_adjudications(
            changed, self.config["contract"], ladder
        )
        self.assertEqual(result["status"], review.UNRESOLVED)
        self.assertFalse(result["n3b_outcome_accepted"])
        self.assertIsNone(result["n3c_selected_command_db"])
        self.assertFalse(result["n3c_execution_authorized"])

    def test_minus3_provenance_is_expectation_only_not_mapping_or_bound(self):
        ladder = review.verify_source(self.config)[
            "command_ladder_expectation_provenance"
        ]
        self.assertEqual(
            ladder["provenance_role"],
            "EXPECTATION_ONLY_ALREADY_ATTACHED_HOT_RUNG_NOT_COLD_ATTAINED",
        )
        self.assertFalse(ladder["cold_attach_evidence"])
        self.assertFalse(ladder["target_mapping_promoted"])
        self.assertFalse(ladder["numeric_bound_promoted"])

    def test_create_only_review_emits_terminal_without_execution_authority(self):
        output = self.root / "review"
        source_before = review.sha256(self.source / "manifest.json")
        runner = review.ReviewRunner(CONFIG_PATH, output)
        self.assertEqual(runner.run(), 0)
        terminal_path = output / f"{review.SUCCESS}.json"
        terminal = review.load_json(terminal_path)
        manifest_path = output / "manifest.json"
        manifest = review.load_json(manifest_path)
        self.assertEqual(terminal["manifest_sha256"], review.sha256(manifest_path))
        self.assertEqual(terminal["cold_attach_result"], "0_OF_3_PASS")
        self.assertIsNone(terminal["cold_achieved_pusch_snr_db_p50"])
        self.assertEqual(terminal["n3c_selected_command_db"], -3.0)
        self.assertFalse(terminal["n3c_execution_authorized"])
        self.assertFalse(terminal["n3c_executed"])
        self.assertIn("UNRESOLVED", terminal["l_attach_status"])
        self.assertIn("UNRESOLVED", terminal["l_operational_status"])
        for key in (
            "target_mapping_promoted", "numeric_bound_promoted",
            "operational_bound_promoted", "connectivity_bound_promoted",
            "usable_service_bound_promoted",
        ):
            self.assertFalse(terminal[key])
            self.assertFalse(manifest[key])
        review.verify_inventory(
            output, manifest,
            allowed_uninventoried=("manifest.json", f"{review.SUCCESS}.json"),
        )
        self.assertEqual(source_before, review.sha256(self.source / "manifest.json"))
        with self.assertRaises(review.ReviewFailure):
            review.ReviewRunner(CONFIG_PATH, output)

    def test_config_source_and_authority_drift_are_rejected(self):
        changed = copy.deepcopy(self.config)
        changed["source"]["manifest_sha256"] = "0" * 64
        with self.assertRaises(review.ReviewFailure):
            review.validate_config(changed, verify_hashes=False)
        changed = copy.deepcopy(self.config)
        changed["command_ladder"]["rung_manifest_sha256"] = "0" * 64
        with self.assertRaises(review.ReviewFailure):
            review.validate_config(changed, verify_hashes=False)
        changed = copy.deepcopy(self.config)
        changed["authority"]["n3c_execution_authorized"] = True
        with self.assertRaises(review.ReviewFailure):
            review.validate_config(changed, verify_hashes=False)

    def test_output_cannot_enter_either_sealed_source(self):
        forbidden = self.source / "must_not_be_created_by_review_test"
        self.assertFalse(forbidden.exists())
        with self.assertRaises(review.ReviewFailure):
            review.ReviewRunner(CONFIG_PATH, forbidden)
        self.assertFalse(forbidden.exists())
        ladder = review.resolve_repo_path(self.config["command_ladder"]["directory"])
        forbidden_ladder = ladder / "must_not_be_created_by_review_test"
        self.assertFalse(forbidden_ladder.exists())
        with self.assertRaises(review.ReviewFailure):
            review.ReviewRunner(CONFIG_PATH, forbidden_ladder)
        self.assertFalse(forbidden_ladder.exists())

    def test_validation_failure_is_sealed_offline(self):
        changed = copy.deepcopy(self.config)
        changed["runtime_seals"][0]["sha256"] = "0" * 64
        config_path = self.root / "bad.json"
        config_path.write_text(json.dumps(changed), encoding="utf-8")
        output = self.root / "failed"
        self.assertEqual(review.ReviewRunner(config_path, output).run(), 1)
        terminal = review.load_json(output / "FAILED.json")
        self.assertTrue(terminal["offline_only"])
        self.assertFalse(terminal["n3c_execution_authorized"])
        self.assertEqual(
            terminal["manifest_sha256"], review.sha256(output / "manifest.json")
        )


if __name__ == "__main__":
    unittest.main()
