import copy
import json
import tempfile
import unittest
from pathlib import Path

from rl_agent import ue_n3d_oai_ul_cold_attach_refinement_v1 as n3d


CONFIG_PATH = (
    n3d.ROOT / "rl_agent/configs/ue_n3d_oai_ul_cold_attach_refinement_v1.json"
)


def valid_service_summary(*, median=7.5, recovery_pass=True):
    return {
        "attach_gate": {
            "status": "COLD_ATTACH_PDU_EXT_DN_GATE_PASSED",
            "passed": True,
        },
        "transport": {"integrity_gate": True},
        "service_tail": {
            "status": "TAIL_ACCEPTED",
            "achieved_pusch_snr_db_p05": 7.0,
            "achieved_pusch_snr_db_median": median,
            "achieved_pusch_snr_db_p95": 7.5,
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
    def __init__(self, passed=True):
        self.passed = passed
        self.calls = []

    def verify_recovery(self, baseline, receiver_baseline, *, required=True):
        self.calls.append((baseline, receiver_baseline, required))
        return {"passed": self.passed}


class N3DPreparePlanTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_config_is_strictly_prepare_only_and_has_no_live_runner(self):
        n3d.validate_config(self.config, verify_hashes=True, require_live=False)
        authority = self.config["authority"]
        self.assertTrue(authority["offline_plan_authorized"])
        for key in (
            "live_oai_run_authorized",
            "live_socket_execution_authorized",
            "carla_run_authorized",
            "target_mapping_promotion_authorized",
            "numeric_bound_promotion_authorized",
            "connectivity_bound_promotion_authorized",
            "usable_service_bound_promotion_authorized",
            "operational_bound_promotion_authorized",
            "policy_training_authorized",
        ):
            self.assertFalse(authority[key], key)
        self.assertNotIn("n3d_live_runner", self.config["paths"])
        self.assertFalse(hasattr(n3d.CampaignRunner, "execute"))
        with self.assertRaisesRegex(
            n3d.ColdAttachRefinementFailure, "live execution is absent"
        ):
            n3d.validate_config(self.config, verify_hashes=False, require_live=True)

    def test_n3c_adjudication_is_explicitly_pending_not_verified(self):
        block = self.config["predecessors"]["n3c_adjudication"]
        self.assertEqual(block, n3d.EXPECTED_N3C_ADJUDICATION_PENDING)
        for key in (
            "directory",
            "manifest_sha256",
            "terminal_sha256",
            "resolved_config_sha256",
            "source_config_sha256",
            "source_runner_sha256",
        ):
            self.assertEqual(block[key], n3d.PENDING)
        self.assertEqual(
            block["n3d_eligibility_status"],
            "UE_N3C_VALID_MIXED_OUTCOME_ACCEPTED_FOR_ADJACENT_N3D_CANDIDATE",
        )
        self.assertEqual(block["n3d_selected_command_db"], -3.5)
        serialized = json.dumps(self.config)
        for future_live_pin in (
            "dcea1b5d",
            "a9c6547d",
            "9f7d3d",
            "4585e7e",
        ):
            self.assertNotIn(future_live_pin, serialized)

    def test_plan_is_three_blocked_fresh_minus3p5_cold_starts(self):
        rows = n3d.campaign_plan_rows(self.config)
        self.assertEqual(len(rows), 3)
        self.assertEqual([row["repetition_index"] for row in rows], [1, 2, 3])
        self.assertTrue(all(row["commanded_noise_power_db"] == -3.5 for row in rows))
        self.assertTrue(all(row["expected_achieved_pusch_snr_db"] == 7.5 for row in rows))
        self.assertTrue(all(row["achieved_snr_tolerance_db"] == 0.5 for row in rows))
        self.assertTrue(all(row["candidate_application_count"] == 0 for row in rows))
        self.assertTrue(all(row["fresh_ran_epoch_required"] for row in rows))
        self.assertTrue(all(row["attach_pdu_ext_dn_timeout_s"] == 180 for row in rows))
        self.assertTrue(all(row["expected_service_frames"] == 600 for row in rows))
        self.assertTrue(all(row["measured_service_s"] == 60 for row in rows))
        self.assertTrue(all(not row["automatic_next_candidate_authorized"] for row in rows))
        self.assertTrue(all(row["status"].startswith("BLOCKED_PENDING") for row in rows))

    def test_sealed_n3c_and_warm_minus3p5_evidence_verify_read_only(self):
        n3c = n3d.verify_n3c_live(self.config)
        warm = n3d.verify_command_ladder_expectation(self.config)
        self.assertEqual(n3c["cold_attach_passes"], 2)
        self.assertEqual(n3c["joint_candidate_confirmation_passes"], 2)
        self.assertEqual(n3c["trials"], 3)
        self.assertTrue(n3c["review_before_next_action_required"])
        self.assertEqual(warm["commanded_noise_power_db"], -3.5)
        self.assertEqual(warm["achieved_pusch_snr_db_p05"], 7.0)
        self.assertEqual(warm["achieved_pusch_snr_db_p50"], 7.5)
        self.assertEqual(warm["achieved_pusch_snr_db_p95"], 7.5)
        self.assertEqual(warm["pusch_samples"], 699)
        self.assertEqual(warm["service_frames_received"], 50)
        self.assertTrue(warm["warm_already_attached_session"])
        self.assertFalse(warm["cold_attach_evidence"])
        self.assertFalse(warm["mapping_promoted"])

    def test_prepare_creates_only_blocked_sealed_plan(self):
        output = self.root / "plan"
        runner = n3d.CampaignRunner(CONFIG_PATH, output)
        self.assertEqual(runner.prepare(), 0)
        terminal_path = output / f"{n3d.PLAN_BLOCKED}.json"
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(terminal["n3c_adjudication_predecessor_pending"])
        self.assertIsNone(terminal["n3c_adjudication_predecessor"])
        self.assertTrue(terminal["live_execution_blocked"])
        self.assertFalse(terminal["runtime_executed"])
        self.assertFalse(terminal["socket_executed"])
        self.assertFalse(terminal["carla_executed"])
        self.assertFalse(terminal["n3d_executed"])
        self.assertEqual(
            terminal["live_engine_status"],
            "NOT_IMPLEMENTED_IN_PREPARE_ONLY_VERSION",
        )
        self.assertTrue(terminal["authoritative_prepare_source_for_live_pinning"])
        self.assertEqual(
            terminal["supersedes_pre_final_plan"],
            n3d.SUPERSEDED_PRE_FINAL_PLAN,
        )
        self.assertEqual(
            terminal["supersedes_pre_final_plan"]["manifest_sha256"],
            "b80c62d97f46674cae2892a6bbbc5c8c352500bdcafb103b6ca13c02278a6594",
        )
        self.assertFalse((output / "repetitions").exists())
        self.assertFalse((output / "n3c_adjudication_predecessor.json").exists())
        self.assertEqual(terminal["manifest_sha256"], n3d.n2.sha256(output / "manifest.json"))
        self.assertEqual(len(manifest["outputs"]), 5)
        n3d._verify_manifest_inventory(
            output, manifest, expected_output_count=5, strict_complete=True,
        )
        with self.assertRaisesRegex(n3d.ColdAttachRefinementFailure, "create-only"):
            n3d.CampaignRunner(CONFIG_PATH, output)

    def test_output_leaf_mutation_fails_before_any_directory_is_created(self):
        changed = copy.deepcopy(self.config)
        changed["output"]["campaign_summary"] = "../escaped.json"
        config_path = self.root / "mutated.json"
        config_path.write_text(json.dumps(changed), encoding="utf-8")
        output = self.root / "must_not_exist"
        with self.assertRaisesRegex(n3d.ColdAttachRefinementFailure, "leaf contract"):
            n3d.CampaignRunner(config_path, output)
        self.assertFalse(output.exists())
        self.assertFalse((self.root / "escaped.json").exists())

    def test_any_scientific_leaf_mutation_is_digest_fail_closed(self):
        mutations = (
            ("rung", "minimum_service_pusch_samples", 1),
            ("traffic", "remote_port", 56131),
            ("telemetry", "collector_tail_s", 2),
            ("analysis", "scheduler_required_mcs_table", 1),
        )
        for section, key, value in mutations:
            changed = copy.deepcopy(self.config)
            changed[section][key] = value
            with self.subTest(section=section, key=key):
                with self.assertRaisesRegex(
                    n3d.ColdAttachRefinementFailure,
                    "scientific contract digest drift",
                ):
                    n3d.validate_config(
                        changed, verify_hashes=False, require_live=False,
                    )

    def test_all_sealed_evidence_roots_are_immutable_output_containment_boundaries(self):
        roots = tuple(n3d.PROTECTED_SOURCE_DIRECTORIES)
        self.assertIn(
            "rl_agent/experiments/ue_n3c_cold_attach_outcome_review_v1", roots,
        )
        for index, relative in enumerate(roots):
            target = n3d.resolve_repo_path(relative) / f"forbidden_n3d_output_{index}"
            self.assertFalse(target.exists())
            with self.assertRaisesRegex(
                n3d.ColdAttachRefinementFailure, "inside sealed source evidence"
            ):
                n3d.CampaignRunner(CONFIG_PATH, target)
            self.assertFalse(target.exists())

    def test_cli_rejects_execute_live_mode(self):
        with self.assertRaises(SystemExit):
            n3d.parse_args([
                "--output-dir", str(self.root / "never"),
                "--mode", "EXECUTE_LIVE",
            ])
        self.assertFalse((self.root / "never").exists())

    def test_recovery_is_required_and_failure_is_infrastructure_invalid(self):
        runner = FakeRecoveryRunner(passed=True)
        self.assertTrue(
            n3d.verify_post_restore_recovery_fail_closed(runner, 10, 20)["passed"]
        )
        self.assertEqual(runner.calls, [(10, 20, True)])
        failed = FakeRecoveryRunner(passed=False)
        with self.assertRaisesRegex(
            n3d.ColdAttachRefinementFailure, "infrastructure-invalid"
        ):
            n3d.verify_post_restore_recovery_fail_closed(failed, 1, 2)

        outcome = n3d.classify_repetition(
            valid_service_summary(recovery_pass=False)
        )
        self.assertEqual(
            outcome["classified_outcome"],
            "CLEAN_RECOVERY_INFRASTRUCTURE_INVALID",
        )
        self.assertTrue(outcome["infrastructure_invalid"])
        self.assertFalse(outcome["evidence_valid_for_aggregation"])

    def test_service_pass_and_consistency_mismatch_never_promote(self):
        passed = n3d.classify_repetition(valid_service_summary())
        self.assertEqual(
            passed["classified_outcome"],
            "COLD_ATTACH_AND_CANDIDATE_SERVICE_CONFIRMED",
        )
        self.assertTrue(passed["joint_candidate_confirmation_pass"])
        self.assertTrue(passed["review_before_promotion_required"])
        mismatch = n3d.classify_repetition(valid_service_summary(median=6.5))
        self.assertEqual(
            mismatch["classified_outcome"],
            "ACHIEVED_SNR_OUTSIDE_CONSISTENCY_BAND",
        )
        self.assertTrue(mismatch["evidence_valid_for_aggregation"])
        self.assertFalse(mismatch["joint_candidate_confirmation_pass"])
        for outcome in (passed, mismatch):
            self.assertFalse(outcome["automatic_next_candidate_authorized"])
            for key in (
                "target_mapping_promoted",
                "numeric_bound_promoted",
                "connectivity_bound_promoted",
                "usable_service_bound_promoted",
                "operational_bound_promoted",
            ):
                self.assertFalse(outcome[key], key)

    def test_attach_failure_is_noncausal_operational_nonconfirmation(self):
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
        outcome = n3d.classify_repetition(summary)
        self.assertEqual(outcome["classified_outcome"], "COLD_ATTACH_FAILED")
        self.assertTrue(outcome["evidence_valid_for_aggregation"])
        self.assertFalse(outcome["joint_candidate_confirmation_pass"])
        self.assertFalse(outcome["post_restore_clean_reattach_evaluated"])
        self.assertFalse(outcome["candidate_causal_attach_failure_confirmed"])
        self.assertIn(
            "NOT_CANDIDATE_CAUSAL_OR_PHYSICAL_BOUNDARY_PROOF",
            outcome["attach_failure_evidence_role"],
        )
        self.assertTrue(outcome["review_before_next_action_required"])


if __name__ == "__main__":
    unittest.main()
