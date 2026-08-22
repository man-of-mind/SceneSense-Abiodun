import copy
import json
import tempfile
import unittest
from pathlib import Path

from rl_agent import ue_n3a_service_bracket_review_v1 as review


CONFIG_PATH = review.DEFAULT_CONFIG


class ServiceBracketReviewTests(unittest.TestCase):
    def setUp(self):
        self.config = review.load_json(CONFIG_PATH)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = review.resolve_repo_path(self.config["source"]["directory"])

    def source_rows(self):
        rows = []
        for spec in self.config["expected_repetitions"]:
            directory = self.source / "repetitions" / spec["directory"]
            summary = review.load_json(directory / "repetition_summary.json")
            cleanup = review.load_json(directory / "cleanup_report.json")
            rows.append(review.adjudicate_repetition(
                summary, cleanup, self.config["contract"]
            ))
        return rows

    def test_config_is_strictly_offline_and_runtime_sealed(self):
        review.validate_config(self.config, verify_hashes=True)
        authority = self.config["authority"]
        self.assertTrue(authority["offline_review_authorized"])
        self.assertFalse(authority["oai_run_authorized"])
        self.assertFalse(authority["socket_execution_authorized"])
        self.assertFalse(authority["carla_run_authorized"])
        self.assertFalse(authority["n3b_execution_authorized"])

    def test_plan_and_live_campaign_and_six_nested_inventories_verify(self):
        before = {
            path: (path.stat().st_mtime_ns, review.sha256(path))
            for path in (
                self.source / "manifest.json",
                self.source / "campaign_summary.json",
                self.source / "resolved_config.json",
            )
        }
        result = review.verify_source(self.config)
        self.assertEqual(result["frozen_plan_manifest_output_count"], 3)
        self.assertEqual(result["campaign_manifest_output_count"], 641)
        self.assertEqual(result["verified_repetition_count"], 6)
        self.assertEqual(result["unique_ran_epoch_count"], 6)
        self.assertEqual(result["unique_control_session_count"], 6)
        self.assertTrue(result["plan_live_resolved_configs_byte_identical"])
        self.assertFalse(result["frozen_fail_condition_had_preregistered_snr_target"])
        self.assertTrue(all(row["manifest_output_count"] > 0
                            for row in result["repetitions"]))
        after = {
            path: (path.stat().st_mtime_ns, review.sha256(path)) for path in before
        }
        self.assertEqual(before, after)

    def test_actual_endpoint_adjudication_is_three_pass_three_fail(self):
        rows = self.source_rows()
        self.assertEqual(
            [row["usable_service_endpoint"] for row in rows],
            ["PASS", "FAIL", "PASS", "FAIL", "PASS", "FAIL"],
        )
        aggregate = review.aggregate_adjudications(rows, self.config["contract"])
        self.assertEqual(aggregate["status"], review.SUCCESS)
        self.assertTrue(aggregate["contract_bracketed"])
        self.assertEqual(aggregate["pass_repetitions"], 3)
        self.assertEqual(aggregate["fail_repetitions"], 3)
        self.assertEqual(aggregate["observed_achieved_fail_endpoint_snr_db"], 5.0)
        self.assertEqual(aggregate["observed_achieved_pass_endpoint_snr_db"], 6.0)
        self.assertEqual(aggregate["observed_achieved_bracket_width_db"], 1.0)
        self.assertEqual(aggregate["pass_numeric_bracket_eligible_repetitions"], 3)
        self.assertEqual(aggregate["fail_numeric_bracket_eligible_repetitions"], 3)

    def test_mechanism_mismatch_is_separate_from_service_failure(self):
        rows = self.source_rows()
        failing = [row for row in rows if row["commanded_noise_power_db"] == -2.0]
        self.assertTrue(all(row["usable_service_endpoint"] == "FAIL" for row in failing))
        self.assertTrue(all(row["evidence_valid"] for row in failing))
        self.assertTrue(all(not row["mechanism_expectation_match"] for row in failing))
        self.assertTrue(all(
            row["mechanism_observed"] == "EXACT_TAIL_SERVICE_GATE_FAILED"
            for row in failing
        ))

    def test_minus2p5_requires_every_frozen_pass_gate(self):
        spec = self.config["expected_repetitions"][0]
        directory = self.source / "repetitions" / spec["directory"]
        base = review.load_json(directory / "repetition_summary.json")
        cleanup = review.load_json(directory / "cleanup_report.json")
        self.assertEqual(
            review.adjudicate_repetition(base, cleanup, self.config["contract"])
            ["usable_service_endpoint"],
            "PASS",
        )
        mutations = (
            ("primary", lambda row: row["tail_service"].update({
                "primary_99_pass": False, "complete_frame_ratio": 0.98,
            })),
            ("gap", lambda row: row["tail_service"].update({
                "no_one_second_outage_pass": False,
                "maximum_interarrival_or_boundary_gap_s": 1.1,
            })),
            ("snr", lambda row: row["tail"].update({
                "achieved_pusch_snr_db_median": 6.6,
            })),
            ("restore", lambda row: row.update({"clean_restore_verified": False})),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                changed = copy.deepcopy(base)
                mutate(changed)
                result = review.adjudicate_repetition(
                    changed, cleanup, self.config["contract"]
                )
                self.assertFalse(result["numeric_bracket_eligible"])
                if label == "snr":
                    self.assertEqual(result["usable_service_endpoint"], "PASS")
        dirty = copy.deepcopy(cleanup)
        dirty["clean"] = False
        result = review.adjudicate_repetition(base, dirty, self.config["contract"])
        self.assertNotEqual(result["usable_service_endpoint"], "PASS")

        valid_service_surprise = copy.deepcopy(base)
        valid_service_surprise["tail_service"].update({
            "primary_99_pass": False,
            "complete_frame_ratio": 0.98,
        })
        result = review.adjudicate_repetition(
            valid_service_surprise, cleanup, self.config["contract"]
        )
        self.assertTrue(result["evidence_valid"])
        self.assertEqual(result["usable_service_endpoint"], "FAIL")

    def test_exact_tail_is_directly_bound_to_60_seconds_and_600_frames(self):
        spec = self.config["expected_repetitions"][0]
        directory = self.source / "repetitions" / spec["directory"]
        base = review.load_json(directory / "repetition_summary.json")
        cleanup = review.load_json(directory / "cleanup_report.json")
        mutations = (
            lambda row: row["tail"].update({
                "end_wall_time_ns": row["tail"]["end_wall_time_ns"] - 1,
            }),
            lambda row: row["tail_service"]["expected_frame_indices"].pop(),
            lambda row: row["tail_service"]["expected_frame_indices"].__setitem__(
                -1, row["tail_service"]["expected_frame_indices"][-2]
            ),
        )
        for mutate in mutations:
            changed = copy.deepcopy(base)
            mutate(changed)
            result = review.adjudicate_repetition(
                changed, cleanup, self.config["contract"]
            )
            self.assertFalse(result["exact_60s_600_frame_tail_valid"])
            self.assertEqual(result["usable_service_endpoint"], "INVALID")

    def test_minus2_fails_only_with_valid_exact_service_failure(self):
        spec = self.config["expected_repetitions"][1]
        directory = self.source / "repetitions" / spec["directory"]
        base = review.load_json(directory / "repetition_summary.json")
        cleanup = review.load_json(directory / "cleanup_report.json")
        result = review.adjudicate_repetition(base, cleanup, self.config["contract"])
        self.assertEqual(result["usable_service_endpoint"], "FAIL")
        self.assertTrue(result["exact_60s_600_frame_tail_valid"])
        self.assertFalse(result["tail_primary_99_pass"])
        changed = copy.deepcopy(base)
        changed["tail_service"]["integrity_gate"] = False
        result = review.adjudicate_repetition(changed, cleanup, self.config["contract"])
        self.assertEqual(result["usable_service_endpoint"], "INVALID")

        off_endpoint = copy.deepcopy(base)
        off_endpoint["tail"]["achieved_pusch_snr_db_median"] = 5.6
        result = review.adjudicate_repetition(
            off_endpoint, cleanup, self.config["contract"]
        )
        self.assertEqual(result["usable_service_endpoint"], "FAIL")
        self.assertFalse(result["source_sealed_observed_fail_snr_match"])
        self.assertFalse(result["numeric_bracket_eligible"])
        rows = self.source_rows()
        rows[1] = result
        aggregate = review.aggregate_adjudications(rows, self.config["contract"])
        self.assertFalse(aggregate["contract_bracketed"])
        self.assertEqual(aggregate["status"], review.UNRESOLVED)

        valid_service_pass = copy.deepcopy(base)
        valid_service_pass["tail_service"].update({
            "primary_99_pass": True,
            "no_one_second_outage_pass": True,
            "complete_frame_ratio": 1.0,
            "maximum_interarrival_or_boundary_gap_s": 0.2,
        })
        result = review.adjudicate_repetition(
            valid_service_pass, cleanup, self.config["contract"]
        )
        self.assertTrue(result["evidence_valid"])
        self.assertEqual(result["usable_service_endpoint"], "PASS")
        self.assertFalse(result["expected_service_role_match"])
        self.assertFalse(result["numeric_bracket_eligible"])
        self.assertEqual(result["mechanism_observed"], "SUSTAINED_SERVICE")

    def test_recognized_hard_loss_is_also_valid_fail_evidence(self):
        spec = self.config["expected_repetitions"][1]
        directory = self.source / "repetitions" / spec["directory"]
        base = review.load_json(directory / "repetition_summary.json")
        cleanup = review.load_json(directory / "cleanup_report.json")
        cases = (
            ("HARD_SERVICE_LOSS_BEFORE_TARGET_CONFIRMATION",
             "CURRENT_RNTI_PUSCH_SILENCE", True),
            ("DETACHED_BEFORE_TARGET_CONFIRMATION", "UE_TUNNEL_IDENTITY_LOST", False),
            ("RNTI_IDENTITY_DISCONTINUITY_BEFORE_TARGET_CONFIRMATION",
             "RNTI_CHANGED", False),
        )
        for status, reason, outage in cases:
            with self.subTest(reason=reason):
                changed = copy.deepcopy(base)
                changed.update({
                    "engine_status": status,
                    "hard_loss_reason": reason,
                    "receiver_service_outage_detected": outage,
                    "tail": None,
                    "tail_service": None,
                })
                result = review.adjudicate_repetition(
                    changed, cleanup, self.config["contract"]
                )
                self.assertEqual(result["usable_service_endpoint"], "FAIL")
                self.assertTrue(result["recognized_hard_loss_evidence"])
                self.assertTrue(result["mechanism_expectation_match"])

    def test_pusch_silence_requires_receiver_outage_corroboration(self):
        spec = self.config["expected_repetitions"][1]
        directory = self.source / "repetitions" / spec["directory"]
        changed = review.load_json(directory / "repetition_summary.json")
        cleanup = review.load_json(directory / "cleanup_report.json")
        changed.update({
            "engine_status": "HARD_SERVICE_LOSS_BEFORE_TARGET_CONFIRMATION",
            "hard_loss_reason": "CURRENT_RNTI_PUSCH_SILENCE",
            "receiver_service_outage_detected": False,
            "tail": None,
            "tail_service": None,
        })
        result = review.adjudicate_repetition(
            changed, cleanup, self.config["contract"]
        )
        self.assertEqual(result["usable_service_endpoint"], "INVALID")

    def test_create_only_review_emits_requested_terminal_and_no_authority(self):
        output = self.root / "review"
        before_manifest = review.sha256(self.source / "manifest.json")
        runner = review.ReviewRunner(CONFIG_PATH, output)
        self.assertEqual(runner.run(), 0)
        terminal_path = output / f"{review.SUCCESS}.json"
        terminal = review.load_json(terminal_path)
        manifest_path = output / "manifest.json"
        manifest = review.load_json(manifest_path)
        self.assertEqual(terminal["manifest_sha256"], review.sha256(manifest_path))
        self.assertEqual(terminal["n3b_selected_command_db"], -2.5)
        self.assertEqual(
            terminal["n3b_eligibility_status"],
            "UE_N3A_USABLE_SERVICE_BRACKET_ACCEPTED_FOR_N3B",
        )
        self.assertEqual(terminal["selection_scope"], "N3B_ELIGIBILITY_ONLY")
        self.assertFalse(terminal["n3b_execution_authorized"])
        self.assertFalse(terminal["n3b_executed"])
        self.assertEqual(terminal["l_attach_status"], "PENDING_N3B_COLD_ATTACH")
        self.assertEqual(terminal["l_operational_status"], "PENDING_N3B_COLD_ATTACH")
        for key in (
            "numeric_bound_promoted", "operational_bound_promoted",
            "connectivity_bound_promoted", "usable_service_bound_promoted",
        ):
            self.assertFalse(terminal[key])
            self.assertFalse(manifest[key])
        review.verify_inventory(
            output, manifest,
            allowed_uninventoried=("manifest.json", f"{review.SUCCESS}.json"),
        )
        self.assertEqual(before_manifest, review.sha256(self.source / "manifest.json"))
        with self.assertRaises(review.ReviewFailure):
            review.ReviewRunner(CONFIG_PATH, output)

    def test_config_source_retarget_is_rejected(self):
        changed = copy.deepcopy(self.config)
        changed["source"]["manifest_sha256"] = "0" * 64
        with self.assertRaises(review.ReviewFailure):
            review.validate_config(changed, verify_hashes=False)

        promoted = review.load_json(self.source / "manifest.json")
        promoted["operational_bound_promoted"] = True
        with self.assertRaises(review.ReviewFailure):
            review._all_promotions_false(promoted, "synthetic promoted source")

    def test_output_contract_and_sealed_source_location_are_fail_closed(self):
        changed = copy.deepcopy(self.config)
        changed["output"]["summary"] = "../../escape.json"
        with self.assertRaises(review.ReviewFailure):
            review.validate_config(changed, verify_hashes=False)

        forbidden = self.source / "must_not_be_created_by_review_test"
        self.assertFalse(forbidden.exists())
        with self.assertRaises(review.ReviewFailure):
            review.ReviewRunner(CONFIG_PATH, forbidden)
        self.assertFalse(forbidden.exists())

    def test_validation_failure_is_sealed_without_external_execution(self):
        changed = copy.deepcopy(self.config)
        changed["runtime_seals"][0]["sha256"] = "0" * 64
        config_path = self.root / "bad.json"
        config_path.write_text(json.dumps(changed), encoding="utf-8")
        output = self.root / "failed"
        self.assertEqual(review.ReviewRunner(config_path, output).run(), 1)
        terminal = review.load_json(output / "FAILED.json")
        self.assertTrue(terminal["offline_only"])
        self.assertFalse(terminal["n3b_execution_authorized"])
        self.assertEqual(
            terminal["manifest_sha256"], review.sha256(output / "manifest.json")
        )


if __name__ == "__main__":
    unittest.main()
