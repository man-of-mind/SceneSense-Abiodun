import copy
import json
import tempfile
import unittest
from pathlib import Path

from rl_agent import ue_n3c_cold_attach_outcome_review_v1 as review


CONFIG_PATH = review.DEFAULT_CONFIG


class N3CColdAttachOutcomeReviewTests(unittest.TestCase):
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

    def test_config_is_strictly_offline_and_runtime_sealed(self):
        review.validate_config(self.config, verify_hashes=True)
        authority = self.config["authority"]
        self.assertTrue(authority["offline_review_authorized"])
        for key, value in authority.items():
            if key != "offline_review_authorized":
                self.assertFalse(value, key)

    def test_campaign_and_every_nested_inventory_verify_read_only(self):
        watched = tuple(path for path in self.source.rglob("*") if path.is_file())
        before = {path: (path.stat().st_mtime_ns, review.sha256(path)) for path in watched}
        result = review.verify_source(self.config)
        self.assertEqual(result["campaign_manifest_output_count"], 251)
        self.assertEqual(result["verified_repetition_count"], 3)
        self.assertEqual(
            [row["manifest_output_count"] for row in result["repetitions"]],
            [109, 21, 109],
        )
        self.assertEqual(result["unique_ran_epoch_count"], 3)
        self.assertEqual(result["unique_control_session_count"], 3)
        self.assertTrue(all(row["evidence_semantics_verified"]
                            for row in result["repetitions"]))
        after = {path: (path.stat().st_mtime_ns, review.sha256(path)) for path in watched}
        self.assertEqual(before, after)

    def test_actual_outcome_is_two_passes_and_one_valid_nonconfirmation(self):
        rows = self.source_rows()
        self.assertEqual(
            [row["adjudicated_outcome"] for row in rows],
            [
                "JOINT_COLD_ATTACH_AND_SERVICE_CONFIRMATION",
                "VALID_OPERATIONAL_SCREEN_NONCONFIRMATION",
                "JOINT_COLD_ATTACH_AND_SERVICE_CONFIRMATION",
            ],
        )
        self.assertEqual(
            [row["joint_candidate_confirmation_pass"] for row in rows],
            [True, False, True],
        )
        failure = rows[1]
        self.assertTrue(failure["failed_cold_achieved_snr_is_null"])
        self.assertIsNone(failure["failed_cold_achieved_pusch_snr_db_p05"])
        self.assertIsNone(failure["failed_cold_achieved_pusch_snr_db_p50"])
        self.assertIsNone(failure["failed_cold_achieved_pusch_snr_db_p95"])
        self.assertEqual(failure["attach_failure_evidence_role"], review.NONCONFIRM_ROLE)
        self.assertFalse(failure["candidate_causal_attach_failure_confirmed"])
        self.assertFalse(failure["physical_rf_cutoff_established"])

    def test_mixed_result_selects_only_adjacent_minus3p5_as_eligibility(self):
        rows = self.source_rows()
        ladder = review.verify_source(self.config)[
            "command_ladder_expectation_provenance"
        ]
        result = review.aggregate_adjudications(rows, self.config["contract"], ladder)
        self.assertEqual(result["status"], review.SUCCESS)
        self.assertEqual(result["joint_confirmation_result"], "2_OF_3_PASS")
        self.assertEqual(result["required_floor_confirmation_result"],
                         "3_OF_3_REQUIRED_NOT_MET")
        self.assertFalse(result["n3c_floor_confirmed"])
        self.assertEqual(result["n3d_selected_command_db"], -3.5)
        self.assertEqual(result["n3d_selection_scope"], "ELIGIBILITY_ONLY")
        self.assertTrue(result["separate_live_authority_required"])
        self.assertFalse(result["n3d_execution_authorized"])
        self.assertFalse(result["n3d_executed"])
        for key in review.PROMOTION_KEYS:
            self.assertFalse(result[key])
        self.assertIn("UNRESOLVED", result["l_attach_status"])
        self.assertIn("UNRESOLVED", result["l_operational_status"])

    def test_invalid_or_different_mix_cannot_select_n3d(self):
        rows = self.source_rows()
        ladder = review.verify_source(self.config)[
            "command_ladder_expectation_provenance"
        ]
        changed = copy.deepcopy(rows)
        changed[1]["evidence_valid"] = False
        changed[1]["adjudicated_outcome"] = "INVALID_EVIDENCE"
        result = review.aggregate_adjudications(changed, self.config["contract"], ladder)
        self.assertEqual(result["status"], review.UNRESOLVED)
        self.assertIsNone(result["n3d_selected_command_db"])
        self.assertFalse(result["n3d_execution_authorized"])

    def test_semantic_mutations_are_fail_closed(self):
        pass_spec = self.config["expected_repetitions"][0]
        pass_dir = self.source / "repetitions" / pass_spec["directory"]
        pass_parts = {
            "summary": review.load_json(pass_dir / "repetition_summary.json"),
            "cleanup": review.load_json(pass_dir / "cleanup_report.json"),
            "startup": review.load_json(pass_dir / "startup_channel_runtime_gate.json"),
            "integrity": review.load_json(pass_dir / "source_oai_config_integrity.json"),
        }
        changed = copy.deepcopy(pass_parts)
        changed["summary"]["service_window"]["received_frames"] = 593
        row = review.adjudicate_repetition(
            changed["summary"], changed["cleanup"], changed["startup"],
            changed["integrity"], self.config["contract"],
        )
        self.assertFalse(row["evidence_valid"])

        fail_spec = self.config["expected_repetitions"][1]
        fail_dir = self.source / "repetitions" / fail_spec["directory"]
        fail_parts = {
            "summary": review.load_json(fail_dir / "repetition_summary.json"),
            "cleanup": review.load_json(fail_dir / "cleanup_report.json"),
            "startup": review.load_json(fail_dir / "startup_channel_runtime_gate.json"),
            "integrity": review.load_json(fail_dir / "source_oai_config_integrity.json"),
        }
        for label, mutate in (
            ("cold_snr", lambda value: value["summary"].update({
                "observed_cold_achieved_pusch_snr_db_p50": 6.5,
            })),
            ("role", lambda value: value["summary"].update({
                "attach_failure_evidence_role": "PHYSICAL_BOUNDARY_PROOF",
            })),
            ("candidate_causal", lambda value: value["summary"].update({
                "candidate_causal_attach_failure_confirmed": True,
            })),
            ("cleanup", lambda value: value["cleanup"].update({"clean": False})),
            ("startup", lambda value: value["startup"]["models"]
             ["rfsimu_channel_ue0"].update({"noise_power_db": -3.5})),
        ):
            with self.subTest(label=label):
                changed = copy.deepcopy(fail_parts)
                mutate(changed)
                row = review.adjudicate_repetition(
                    changed["summary"], changed["cleanup"], changed["startup"],
                    changed["integrity"], self.config["contract"],
                )
                self.assertFalse(row["evidence_valid"])

    def test_minus3p5_source_is_hot_expectation_only(self):
        ladder = review.verify_source(self.config)[
            "command_ladder_expectation_provenance"
        ]
        self.assertEqual(ladder["commanded_noise_power_db"], -3.5)
        self.assertEqual(ladder["hot_observed_pusch_snr_db_p50"], 7.5)
        self.assertEqual(
            ladder["provenance_role"],
            "EXPECTATION_ONLY_ALREADY_ATTACHED_HOT_RUNG_NOT_COLD_ATTAINED",
        )
        self.assertFalse(ladder["cold_attach_evidence"])
        self.assertFalse(ladder["target_mapping_promoted"])
        self.assertFalse(ladder["numeric_bound_promoted"])

    def test_create_only_review_emits_immutable_contained_terminal(self):
        output_root = self.root / "allowed"
        output = output_root / "review_01"
        source_before = review.sha256(self.source / "manifest.json")
        runner = review.ReviewRunner(
            CONFIG_PATH, output, output_root=output_root
        )
        self.assertEqual(runner.run(), 0)
        terminal_path = output / f"{review.SUCCESS}.json"
        terminal = review.load_json(terminal_path)
        manifest = review.load_json(output / "manifest.json")
        self.assertEqual(terminal["manifest_sha256"],
                         review.sha256(output / "manifest.json"))
        self.assertEqual(terminal["joint_confirmation_result"], "2_OF_3_PASS")
        self.assertEqual(terminal["n3d_selected_command_db"], -3.5)
        self.assertFalse(terminal["n3d_execution_authorized"])
        self.assertFalse(terminal["oai_run_authorized"])
        self.assertFalse(terminal["socket_execution_authorized"])
        self.assertFalse(terminal["carla_run_authorized"])
        review.verify_inventory(
            output, manifest,
            allowed_uninventoried=("manifest.json", f"{review.SUCCESS}.json"),
        )
        self.assertEqual(source_before, review.sha256(self.source / "manifest.json"))
        with self.assertRaises(review.ReviewFailure):
            review.ReviewRunner(CONFIG_PATH, output, output_root=output_root)
        with self.assertRaises(review.ReviewFailure):
            review.ReviewRunner(
                CONFIG_PATH, self.root / "outside" / "review",
                output_root=output_root,
            )

    def test_source_authority_and_candidate_drift_are_rejected(self):
        for label, mutate in (
            ("source", lambda value: value["source"].update({
                "manifest_sha256": "0" * 64,
            })),
            ("authority", lambda value: value["authority"].update({
                "oai_run_authorized": True,
            })),
            ("candidate", lambda value: value["contract"].update({
                "n3d_selected_command_db": -4.0,
            })),
        ):
            with self.subTest(label=label):
                changed = copy.deepcopy(self.config)
                mutate(changed)
                with self.assertRaises(review.ReviewFailure):
                    review.validate_config(changed, verify_hashes=False)

    def test_validation_failure_seals_no_runtime_authority(self):
        changed = copy.deepcopy(self.config)
        changed["runtime_seals"][0]["sha256"] = "0" * 64
        config_path = self.root / "bad.json"
        config_path.write_text(json.dumps(changed), encoding="utf-8")
        output_root = self.root / "failed-root"
        output = output_root / "failed_01"
        runner = review.ReviewRunner(
            config_path, output, output_root=output_root
        )
        self.assertEqual(runner.run(), 1)
        terminal = review.load_json(output / "FAILED.json")
        self.assertTrue(terminal["offline_only"])
        self.assertFalse(terminal["n3d_execution_authorized"])
        self.assertFalse(terminal["oai_run_authorized"])
        self.assertFalse(terminal["carla_run_authorized"])
        self.assertEqual(terminal["manifest_sha256"],
                         review.sha256(output / "manifest.json"))


if __name__ == "__main__":
    unittest.main()
