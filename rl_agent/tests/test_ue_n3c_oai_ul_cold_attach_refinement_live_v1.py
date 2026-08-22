import copy
import inspect
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from rl_agent import ue_n3c_oai_ul_cold_attach_refinement_live_v1 as live


CONFIG_PATH = (
    live.ROOT
    / "rl_agent/configs/ue_n3c_oai_ul_cold_attach_refinement_live_v1.json"
)
PLAN_CONFIG_PATH = (
    live.ROOT / "rl_agent/configs/ue_n3c_oai_ul_cold_attach_refinement_v1.json"
)


def valid_summary(*, recovery_pass=True, median=6.5):
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


def attach_nonconfirmation_summary():
    summary = valid_summary()
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
        "observed_cold_achieved_pusch_snr_db_p05": None,
        "observed_cold_achieved_pusch_snr_db_p50": None,
        "observed_cold_achieved_pusch_snr_db_p95": None,
        "clean_recovery": {
            "status": "NOT_APPLICABLE_NO_PDU_SESSION",
            "passed": None,
        },
        "post_restore_clean_reattach_evaluated": False,
        "candidate_causal_attach_failure_confirmed": False,
    })
    return summary


class N3CLiveRunnerTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.plan_config = json.loads(PLAN_CONFIG_PATH.read_text(encoding="utf-8"))
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def make_rep(self, name="rep"):
        return live.ColdAttachLiveRepetition(
            CONFIG_PATH,
            self.root / name,
            repetition_index=1,
            n3b_live_proof={},
            n3b_adjudication_proof={},
            expectation_proof={},
        )

    def test_live_config_and_all_three_predecessors_verify(self):
        live.validate_config(self.config, verify_hashes=True, require_live=True)
        n3b = live.plan.verify_n3b_live(self.config)
        review = live.plan.verify_n3b_adjudication(self.config)
        ladder = live.plan.verify_command_ladder_expectation(self.config)
        self.assertEqual(
            n3b["manifest_sha256"],
            "ac76763ea9651212f0003c35eb19092d85dbf28b104572e9a9ffc107cb298f3a",
        )
        self.assertEqual(
            review["manifest_sha256"],
            "bda88d7f89e41822bf07209a69f32a9a2f72cacabccc6112364cfa81b135398f",
        )
        self.assertEqual(ladder["achieved_pusch_snr_db_p50"], 6.5)
        self.assertFalse(ladder["mapping_promoted"])

    def test_live_scientific_contract_equals_sealed_prepare_contract(self):
        self.assertEqual(
            live._scientific_projection(self.config),
            live._scientific_projection(self.plan_config),
        )
        self.assertEqual(self.config["startup_channel"]["rfsimu_channel_ue0"], -3.0)
        self.assertEqual(
            self.config["analysis"]["expected_achieved_pusch_snr_db"], 6.5
        )
        self.assertEqual(
            self.config["analysis"]["achieved_snr_expectation_role"],
            "CONSISTENCY_ONLY_FROM_SEALED_WARM_ATTACHED_COMMAND_LADDER_NOT_MAPPING",
        )
        self.assertEqual(self.config["campaign"]["repetitions"], 3)
        self.assertEqual(self.config["campaign"]["candidate_application_count"], 0)
        self.assertEqual(self.config["radio"]["attach_timeout_s"], 180)
        self.assertEqual(self.config["rung"]["measured_service_s"], 60)
        self.assertEqual(self.config["rung"]["expected_service_frames"], 600)

    def test_exact_n3b_adjudication_pin_is_code_frozen(self):
        self.assertEqual(
            self.config["predecessors"]["n3b_adjudication"],
            live.EXPECTED_N3B_ADJUDICATION,
        )
        changed = copy.deepcopy(self.config)
        changed["predecessors"]["n3b_adjudication"]["manifest_sha256"] = (
            "0" * 64
        )
        with self.assertRaisesRegex(
            live.LiveRefinementFailure, "sealed N3B adjudication evidence drift",
        ):
            live.validate_config(changed, verify_hashes=False, require_live=True)

    def test_live_plan_is_ready_but_prepare_never_executes(self):
        rows = live.campaign_plan_rows(self.config)
        self.assertEqual(len(rows), 3)
        self.assertTrue(
            all(row["status"] == "READY_FOR_EXPLICIT_EXECUTE_LIVE" for row in rows)
        )
        output = self.root / "ready_plan"
        runner = live.CampaignRunner(CONFIG_PATH, output)
        self.assertEqual(runner.prepare(), 0)
        terminal = json.loads(
            (output / f"{live.PLAN_READY}.json").read_text(encoding="utf-8")
        )
        self.assertFalse(terminal["runtime_executed"])
        self.assertFalse(terminal["socket_executed"])
        self.assertFalse(terminal["live_execution_blocked"])
        self.assertFalse(terminal["n3b_adjudication_predecessor_pending"])
        self.assertFalse((output / "repetitions").exists())

    def test_authority_disabled_execute_fails_before_runtime(self):
        changed = copy.deepcopy(self.config)
        changed["authority"].update({
            "live_oai_run_authorized": False,
            "live_socket_execution_authorized": False,
            "live_authority_basis": "NOT_AUTHORIZED_PREPARE_ONLY",
        })
        config_path = self.root / "offline.json"
        config_path.write_text(json.dumps(changed), encoding="utf-8")
        output = self.root / "blocked_execute"
        runner = live.CampaignRunner(config_path, output)
        self.assertEqual(runner.execute(), 1)
        terminal = json.loads((output / "FAILED.json").read_text(encoding="utf-8"))
        self.assertFalse(terminal["runtime_executed"])
        self.assertFalse(terminal["socket_executed"])
        self.assertFalse(terminal["live_execution_attempted"])
        self.assertFalse((output / "repetitions").exists())

    def assert_invalid_stage_truth(
        self, *, name, runtime_attempted, socket_attempted,
    ):
        class InvalidRunner:
            def __init__(self, _config, _output, **_kwargs):
                self.runtime_execution_attempted = False
                self.socket_execution_attempted = False

            def run(inner_self):
                inner_self.runtime_execution_attempted = runtime_attempted
                inner_self.socket_execution_attempted = socket_attempted
                return 1

        output = self.root / name
        with mock.patch.object(live, "ColdAttachLiveRepetition", InvalidRunner):
            runner = live.CampaignRunner(CONFIG_PATH, output)
            self.assertEqual(runner.execute(), 1)
        terminal = json.loads((output / "FAILED.json").read_text(encoding="utf-8"))
        self.assertEqual(terminal["runtime_executed"], runtime_attempted)
        self.assertEqual(terminal["socket_executed"], socket_attempted)
        self.assertEqual(
            terminal["live_execution_attempted"],
            runtime_attempted or socket_attempted,
        )
        self.assertEqual(terminal["repetitions_attempted"], 1)
        self.assertEqual(terminal["repetitions_executed"], 0)

    def test_preflight_invalid_repetition_reports_no_runtime_or_socket(self):
        self.assert_invalid_stage_truth(
            name="preflight_invalid",
            runtime_attempted=False,
            socket_attempted=False,
        )

    def test_ran_started_before_telnet_failure_reports_runtime_only(self):
        self.assert_invalid_stage_truth(
            name="ran_before_socket_invalid",
            runtime_attempted=True,
            socket_attempted=False,
        )

    def test_telnet_attempt_failure_reports_runtime_and_socket(self):
        self.assert_invalid_stage_truth(
            name="socket_attempt_invalid",
            runtime_attempted=True,
            socket_attempted=True,
        )

    def test_output_is_rejected_inside_every_sealed_source_tree(self):
        protected = [
            live.plan.resolve_repo_path(relative)
            for relative in live.PROTECTED_SOURCE_DIRECTORIES
        ]
        for index, source in enumerate(protected):
            target = source / f"forbidden_live_output_{index}"
            self.assertFalse(target.exists())
            with self.assertRaisesRegex(live.LiveRefinementFailure, "sealed source"):
                live.CampaignRunner(CONFIG_PATH, target)
            self.assertFalse(target.exists())

    def test_mutated_predecessor_paths_cannot_bypass_source_containment(self):
        changed = copy.deepcopy(self.config)
        for key in (
            "n3b_live_evidence",
            "n3b_adjudication",
            "command_ladder_expectation",
        ):
            changed["predecessors"][key]["directory"] = (
                "rl_agent/experiments/nonexistent_mutated_source"
            )
        config_path = self.root / "mutated_predecessors.json"
        config_path.write_text(json.dumps(changed), encoding="utf-8")
        real_source = live.plan.resolve_repo_path(
            live.EXPECTED_N3B_ADJUDICATION["directory"]
        )
        target = real_source / f"forbidden_bypass_{self.root.name}"
        self.assertFalse(target.exists())
        with self.assertRaisesRegex(live.LiveRefinementFailure, "sealed source"):
            live.CampaignRunner(config_path, target)
        self.assertFalse(target.exists())

    def test_campaign_output_leaf_escape_is_rejected_before_any_write(self):
        cases = (
            ("campaign_summary", "../escaped_campaign_summary.json"),
            ("failure", str((self.root / "absolute_escape.json").resolve())),
        )
        for index, (key, malicious) in enumerate(cases):
            with self.subTest(key=key):
                changed = copy.deepcopy(self.config)
                changed["output"][key] = malicious
                config_path = self.root / f"escaped_campaign_{index}.json"
                config_path.write_text(json.dumps(changed), encoding="utf-8")
                output = self.root / f"escaped_campaign_output_{index}"
                escaped = (
                    self.root / "escaped_campaign_summary.json"
                    if key == "campaign_summary"
                    else self.root / "absolute_escape.json"
                )
                self.assertFalse(output.exists())
                self.assertFalse(escaped.exists())
                with self.assertRaisesRegex(
                    live.LiveRefinementFailure, "output leaf contract drift",
                ):
                    live.CampaignRunner(config_path, output)
                self.assertFalse(output.exists())
                self.assertFalse(escaped.exists())

    def test_repetition_output_leaf_escape_is_rejected_before_any_write(self):
        changed = copy.deepcopy(self.config)
        changed["output"]["repetition_summary"] = "../escaped_repetition.json"
        config_path = self.root / "escaped_repetition_config.json"
        config_path.write_text(json.dumps(changed), encoding="utf-8")
        output = self.root / "escaped_repetition_output"
        escaped = self.root / "escaped_repetition.json"
        self.assertFalse(output.exists())
        self.assertFalse(escaped.exists())
        with self.assertRaisesRegex(
            live.LiveRefinementFailure, "output leaf contract drift",
        ):
            live.ColdAttachLiveRepetition(
                config_path,
                output,
                repetition_index=1,
                n3b_live_proof={},
                n3b_adjudication_proof={},
                expectation_proof={},
            )
        self.assertFalse(output.exists())
        self.assertFalse(escaped.exists())

    def test_strict_inventory_rejects_unlisted_addition(self):
        directory = self.root / "inventory"
        directory.mkdir()
        payload = directory / "payload.json"
        payload.write_text("{}", encoding="utf-8")
        manifest = {
            "status": "TEST_STATUS",
            "outputs": [{
                "path": "payload.json",
                "bytes": payload.stat().st_size,
                "sha256": live.n2.sha256(payload),
            }],
        }
        (directory / "manifest.json").write_text("{}", encoding="utf-8")
        (directory / "TEST_STATUS.json").write_text("{}", encoding="utf-8")
        live._strict_manifest_inventory(
            directory, manifest, expected_output_count=1,
        )
        (directory / "unlisted.txt").write_text("drift", encoding="utf-8")
        with self.assertRaisesRegex(live.LiveRefinementFailure, "completeness"):
            live._strict_manifest_inventory(
                directory, manifest, expected_output_count=1,
            )

    def test_materialization_and_inherited_helpers_are_minus3_config_driven(self):
        source_text = Path(inspect.getsourcefile(live)).read_text(encoding="utf-8")
        for forbidden in ("-2.5", "6.0", "minus2p5"):
            self.assertNotIn(forbidden, source_text)
        runner = self.make_rep("materialized")
        self.assertEqual(runner.command_db, -3.0)
        gnb, ue = runner.materialize_configs()
        self.assertEqual(gnb.name, "effective_gnb_cold_attach_minus3p0.conf")
        self.assertEqual(ue.name, "effective_ue_cold_attach_minus3p0.conf")
        channel = (
            runner.output_dir / "runtime/effective_channel_cold_attach_minus3p0.conf"
        ).read_text(encoding="utf-8")
        self.assertEqual(
            live.n3b.configured_channel_values(channel),
            {
                "rfsimu_channel_enB0": -50.0,
                "rfsimu_channel_enB1": -50.0,
                "rfsimu_channel_ue0": -3.0,
            },
        )
        self.assertTrue(runner.source_integrity()["unchanged"])

    def test_required_recovery_forwards_required_true(self):
        runner = self.make_rep("recovery")
        calls = []

        def fake_verify(_self, baseline, receiver_baseline, *, required=True):
            calls.append((baseline, receiver_baseline, required))
            return {"passed": True}

        runner.verify_recovery = types.MethodType(fake_verify, runner)
        self.assertTrue(runner.verify_required_recovery(10, 20)["passed"])
        self.assertEqual(calls, [(10, 20, True)])

    def test_recovery_failure_is_infrastructure_invalid_never_valid_service_fail(self):
        outcome = live.classify_repetition(valid_summary(recovery_pass=False))
        self.assertEqual(
            outcome["classified_outcome"],
            "CLEAN_RECOVERY_INFRASTRUCTURE_INVALID",
        )
        self.assertTrue(outcome["infrastructure_invalid"])
        self.assertFalse(outcome["evidence_valid_for_aggregation"])
        self.assertFalse(outcome["authoritative_service_gate_pass"])
        self.assertFalse(outcome["joint_candidate_confirmation_pass"])
        with self.assertRaisesRegex(live.LiveRefinementFailure, "invalid repetition"):
            live.aggregate_results([outcome, outcome, outcome])

    def test_attach_failure_is_only_operational_screen_nonconfirmation(self):
        outcome = live.classify_repetition(attach_nonconfirmation_summary())
        self.assertEqual(outcome["classified_outcome"], "COLD_ATTACH_FAILED")
        self.assertTrue(outcome["evidence_valid_for_aggregation"])
        self.assertFalse(outcome["joint_candidate_confirmation_pass"])
        self.assertFalse(outcome["achieved_snr_gate_pass"])
        self.assertIsNone(attach_nonconfirmation_summary()["service_tail"])
        for key in (
            "observed_cold_achieved_pusch_snr_db_p05",
            "observed_cold_achieved_pusch_snr_db_p50",
            "observed_cold_achieved_pusch_snr_db_p95",
        ):
            self.assertIsNone(attach_nonconfirmation_summary()[key])
        self.assertFalse(outcome["post_restore_clean_reattach_evaluated"])
        self.assertFalse(outcome["candidate_causal_attach_failure_confirmed"])
        self.assertTrue(outcome["review_before_next_action_required"])
        self.assertIn("OPERATIONAL_SCREEN_NONCONFIRMATION", outcome["attach_failure_evidence_role"])
        aggregate = live.aggregate_results([outcome, outcome, outcome])
        self.assertEqual(aggregate["status"], live.CAMPAIGN_NOT_3_OF_3)
        self.assertEqual(aggregate["operational_screen_attach_nonconfirmations"], 3)
        self.assertTrue(aggregate["review_before_next_action_required"])
        for key in (
            "target_mapping_promoted",
            "numeric_bound_promoted",
            "connectivity_bound_promoted",
            "usable_service_bound_promoted",
            "operational_bound_promoted",
        ):
            self.assertFalse(aggregate[key])

    def test_only_three_joint_passes_reach_campaign_pass_terminal(self):
        passed = live.classify_repetition(valid_summary())
        self.assertTrue(passed["joint_candidate_confirmation_pass"])
        aggregate = live.aggregate_results([passed, passed, passed])
        self.assertEqual(aggregate["status"], live.CAMPAIGN_PASSED)
        self.assertTrue(aggregate["joint_candidate_confirmation_3_of_3_pass"])
        self.assertFalse(aggregate["operational_bound_promoted"])

    def test_any_joint_nonpass_requires_review_even_when_all_attach(self):
        passed = live.classify_repetition(valid_summary())
        snr_mismatch = live.classify_repetition(valid_summary(median=8.0))
        service_summary = valid_summary()
        service_summary["service_window"]["primary_99_pass"] = False
        service_nonpass = live.classify_repetition(service_summary)
        for nonpass in (snr_mismatch, service_nonpass):
            with self.subTest(outcome=nonpass["classified_outcome"]):
                self.assertTrue(nonpass["cold_attach_gate_pass"])
                self.assertFalse(nonpass["joint_candidate_confirmation_pass"])
                self.assertTrue(nonpass["review_before_next_action_required"])
                aggregate = live.aggregate_results([passed, passed, nonpass])
                self.assertEqual(aggregate["cold_attach_passes"], 3)
                self.assertEqual(aggregate["joint_candidate_confirmation_passes"], 2)
                self.assertEqual(aggregate["status"], live.CAMPAIGN_NOT_3_OF_3)
                self.assertTrue(aggregate["review_before_next_action_required"])


if __name__ == "__main__":
    unittest.main()
