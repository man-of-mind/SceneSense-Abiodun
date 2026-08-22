import copy
import json
import tempfile
import unittest
from pathlib import Path

from rl_agent import ue_n3c_oai_ul_cold_attach_refinement_v1 as n3c


CONFIG_PATH = (
    n3c.ROOT / "rl_agent/configs/ue_n3c_oai_ul_cold_attach_refinement_v1.json"
)
N3B_CONFIG_PATH = (
    n3c.ROOT / "rl_agent/configs/ue_n3b_oai_ul_cold_attach_confirmation_v1.json"
)


def valid_service_summary(*, median=6.5, recovery_pass=True):
    return {
        "attach_gate": {
            "status": "COLD_ATTACH_PDU_EXT_DN_GATE_PASSED",
            "passed": True,
        },
        "transport": {"integrity_gate": True},
        "service_tail": {
            "status": "TAIL_ACCEPTED",
            "achieved_pusch_snr_db_p05": 6.0,
            "achieved_pusch_snr_db_median": median,
            "achieved_pusch_snr_db_p95": 7.0,
        },
        "service_window": {
            "full_nominal_window_observed": True,
            "exact_frozen_frame_set_pass": True,
            "required_expected_frames": 600,
            "expected_frames": 600,
            "integrity_gate": True,
            "primary_99_pass": True,
            "no_one_second_outage_pass": True,
        },
        "clean_recovery": {"passed": recovery_pass},
        "hard_loss_reason": None,
        "receiver_service_outage_detected": False,
        "candidate_baked_config_verified": True,
        "startup_channel_runtime_verified": True,
        "candidate_application_count": 0,
        "restore_application_count": 1,
        "clean_restore_verified": True,
        "source_oai_configs_unchanged": True,
        "cleanup_clean": True,
    }


class FakeRecoveryRunner:
    def __init__(self, *, passed=True):
        self.passed = passed
        self.calls = []

    def verify_recovery(self, baseline_count, receiver_baseline_count, *, required=True):
        self.calls.append((baseline_count, receiver_baseline_count, required))
        return {"passed": self.passed}


class ColdAttachRefinementPlanTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.n3b = json.loads(N3B_CONFIG_PATH.read_text(encoding="utf-8"))
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_pending_config_validates_offline_and_live_fails_closed(self):
        n3c.validate_config(self.config, verify_hashes=True, require_live=False)
        with self.assertRaisesRegex(n3c.ColdAttachRefinementFailure, "authority is absent"):
            n3c.validate_config(self.config, verify_hashes=False, require_live=True)

        changed = copy.deepcopy(self.config)
        changed["authority"].update({
            "live_oai_run_authorized": True,
            "live_socket_execution_authorized": True,
            "live_authority_basis": n3c.LIVE_AUTHORITY_BASIS,
        })
        with self.assertRaisesRegex(n3c.ColdAttachRefinementFailure, "still pending"):
            n3c.validate_config(changed, verify_hashes=False, require_live=True)

    def test_pending_adjudication_contract_matches_review_tool_constants(self):
        adjudication = self.config["predecessors"]["n3b_adjudication"]
        self.assertEqual(
            adjudication["required_status"],
            "UE_N3B_OUTCOME_ADJUDICATED_N3C_ELIGIBLE_REVIEW_REQUIRED",
        )
        self.assertEqual(
            adjudication["terminal"],
            "UE_N3B_OUTCOME_ADJUDICATED_N3C_ELIGIBLE_REVIEW_REQUIRED.json",
        )
        self.assertEqual(
            adjudication["n3c_eligibility_status"],
            "UE_N3B_VALID_COLD_ATTACH_FAILURE_ACCEPTED_FOR_N3C",
        )
        self.assertEqual(adjudication["n3c_selected_command_db"], -3.0)

    def test_scientific_leaves_match_n3b_except_authorized_changes(self):
        for key in ("rung", "traffic", "transport_gates", "preflight", "radio", "actuator", "telemetry", "output"):
            self.assertEqual(self.config[key], self.n3b[key], key)

        campaign_keys = (
            "repetitions",
            "one_fresh_ran_per_repetition",
            "run_local_configs_only",
            "candidate_baked_before_ue_launch",
            "candidate_application_count",
            "continue_after_valid_attach_or_service_failure",
            "stop_on_invalid_or_unclean_evidence",
        )
        for key in campaign_keys:
            self.assertEqual(self.config["campaign"][key], self.n3b["campaign"][key], key)
        self.assertTrue(self.config["campaign"]["post_restore_recovery_fail_closed"])

        self.assertEqual(
            {key: self.config["startup_channel"][key] for key in ("rfsimu_channel_enB0", "rfsimu_channel_enB1")},
            {key: self.n3b["startup_channel"][key] for key in ("rfsimu_channel_enB0", "rfsimu_channel_enB1")},
        )
        self.assertEqual(self.config["startup_channel"]["rfsimu_channel_ue0"], -3.0)
        self.assertEqual(self.n3b["startup_channel"]["rfsimu_channel_ue0"], -2.5)

        inherited_analysis = (
            "achieved_snr_tolerance_db",
            "scheduler_required_mcs_table",
            "scheduler_required_force_ul_mcs",
            "direct_ul_bler_zero_fill_authorized",
            "timestamp_claim",
            "whole_capture_delivery_role",
            "service_window_delivery_role",
        )
        for key in inherited_analysis:
            self.assertEqual(self.config["analysis"][key], self.n3b["analysis"][key], key)
        self.assertEqual(self.config["analysis"]["expected_achieved_pusch_snr_db"], 6.5)
        self.assertEqual(self.n3b["analysis"]["expected_achieved_pusch_snr_db"], 6)
        self.assertTrue(self.config["analysis"]["post_restore_recovery_required"])

    def test_plan_is_three_fresh_minus3_cold_starts(self):
        rows = n3c.campaign_plan_rows(self.config)
        self.assertEqual(len(rows), 3)
        self.assertEqual([row["repetition_index"] for row in rows], [1, 2, 3])
        self.assertTrue(all(row["commanded_noise_power_db"] == -3.0 for row in rows))
        self.assertTrue(all(row["expected_achieved_pusch_snr_db"] == 6.5 for row in rows))
        self.assertTrue(all(row["candidate_application_count"] == 0 for row in rows))
        self.assertTrue(all(row["fresh_ran_epoch_required"] for row in rows))
        self.assertTrue(all(row["attach_pdu_ext_dn_timeout_s"] == 180 for row in rows))
        self.assertTrue(all(row["expected_service_frames"] == 600 for row in rows))
        self.assertTrue(all(row["measured_service_s"] == 60 for row in rows))
        self.assertTrue(all(row["post_restore_recovery_required"] for row in rows))
        self.assertTrue(all(row["status"] == "BLOCKED_PENDING_PREREQUISITES" for row in rows))

    def test_sealed_n3b_and_command_expectation_verify_read_only(self):
        n3b = n3c.verify_n3b_live(self.config)
        expectation = n3c.verify_command_ladder_expectation(self.config)
        self.assertEqual(n3b["cold_attach_passes"], 0)
        self.assertEqual(n3b["cold_attach_trials"], 3)
        self.assertEqual(expectation["commanded_noise_power_db"], -3.0)
        self.assertEqual(expectation["achieved_pusch_snr_db_p50"], 6.5)
        self.assertFalse(expectation["mapping_promoted"])

    def test_prepare_is_create_only_blocked_and_never_executes_runtime(self):
        output = self.root / "plan"
        runner = n3c.CampaignRunner(CONFIG_PATH, output)
        self.assertEqual(runner.prepare(), 0)
        terminal_path = output / f"{n3c.PLAN_BLOCKED}.json"
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(terminal["n3b_adjudication_predecessor_pending"])
        self.assertTrue(terminal["live_execution_blocked"])
        self.assertFalse(terminal["runtime_executed"])
        self.assertFalse(terminal["socket_executed"])
        self.assertEqual(terminal["live_engine_status"], "NOT_IMPLEMENTED_IN_PREPARE_ONLY_VERSION")
        self.assertFalse((output / "repetitions").exists())
        self.assertEqual(terminal["manifest_sha256"], n3c.n2.sha256(output / "manifest.json"))
        n3c._verify_manifest_inventory(output, manifest)
        with self.assertRaises(n3c.ColdAttachRefinementFailure):
            n3c.CampaignRunner(CONFIG_PATH, output)

    def test_execute_live_fails_before_any_repetition_or_socket(self):
        output = self.root / "blocked_live"
        runner = n3c.CampaignRunner(CONFIG_PATH, output)
        self.assertEqual(runner.execute(), 1)
        terminal = json.loads((output / "FAILED.json").read_text(encoding="utf-8"))
        self.assertFalse(terminal["runtime_executed"])
        self.assertFalse(terminal["socket_executed"])
        self.assertFalse((output / "repetitions").exists())
        self.assertIn("authority is absent", terminal["error"])

    def test_post_restore_recovery_wrapper_always_requests_required(self):
        runner = FakeRecoveryRunner(passed=True)
        result = n3c.verify_post_restore_recovery_fail_closed(runner, 11, 22)
        self.assertTrue(result["passed"])
        self.assertEqual(runner.calls, [(11, 22, True)])

        failed = FakeRecoveryRunner(passed=False)
        with self.assertRaisesRegex(
            n3c.ColdAttachRefinementFailure, "infrastructure-invalid"
        ):
            n3c.verify_post_restore_recovery_fail_closed(failed, 1, 2)
        self.assertEqual(failed.calls, [(1, 2, True)])

    def test_failed_recovery_is_infrastructure_invalid_not_service_failure(self):
        outcome = n3c.classify_repetition(
            valid_service_summary(recovery_pass=False)
        )
        self.assertEqual(
            outcome["classified_outcome"],
            "CLEAN_RECOVERY_INFRASTRUCTURE_INVALID",
        )
        self.assertTrue(outcome["infrastructure_invalid"])
        self.assertFalse(outcome["evidence_valid_for_aggregation"])
        self.assertFalse(outcome["joint_candidate_confirmation_pass"])

    def test_valid_service_requires_exact_window_recovery_and_consistency_band(self):
        outcome = n3c.classify_repetition(valid_service_summary())
        self.assertEqual(
            outcome["classified_outcome"],
            "COLD_ATTACH_AND_CANDIDATE_SERVICE_CONFIRMED",
        )
        self.assertTrue(outcome["evidence_valid_for_aggregation"])
        self.assertTrue(outcome["joint_candidate_confirmation_pass"])
        self.assertTrue(outcome["post_restore_recovery_passed"])
        self.assertFalse(outcome["target_mapping_promoted"])
        self.assertFalse(outcome["numeric_bound_promoted"])
        self.assertFalse(outcome["operational_bound_promoted"])

    def test_consistency_mismatch_is_retained_without_mapping_promotion(self):
        outcome = n3c.classify_repetition(valid_service_summary(median=7.5))
        self.assertEqual(
            outcome["classified_outcome"],
            "ACHIEVED_SNR_OUTSIDE_CONSISTENCY_BAND",
        )
        self.assertTrue(outcome["evidence_valid_for_aggregation"])
        self.assertFalse(outcome["joint_candidate_confirmation_pass"])
        self.assertFalse(outcome["achieved_snr_gate_pass"])

    def test_clean_attach_failure_remains_valid_command_space_evidence(self):
        summary = valid_service_summary()
        summary.update({
            "attach_gate": {
                "status": "COLD_ATTACH_OR_PDU_EXT_DN_GATE_FAILED",
                "passed": False,
                "ran_processes_alive_at_terminal": True,
                "core_ready_at_terminal": True,
            },
            "transport": None,
            "service_tail": None,
            "service_window": None,
            "clean_recovery": {
                "status": "NOT_APPLICABLE_NO_PDU_SESSION",
                "passed": None,
            },
        })
        outcome = n3c.classify_repetition(summary)
        self.assertEqual(outcome["classified_outcome"], "COLD_ATTACH_FAILED")
        self.assertTrue(outcome["evidence_valid_for_aggregation"])
        self.assertFalse(outcome["post_restore_recovery_required"])
        self.assertFalse(outcome["infrastructure_invalid"])

    def test_default_cli_is_prepare_only(self):
        args = n3c.parse_args(["--output-dir", str(self.root / "unused")])
        self.assertEqual(args.mode, n3c.PREPARE_ONLY)
        self.assertEqual(Path(args.config), CONFIG_PATH)


if __name__ == "__main__":
    unittest.main()
