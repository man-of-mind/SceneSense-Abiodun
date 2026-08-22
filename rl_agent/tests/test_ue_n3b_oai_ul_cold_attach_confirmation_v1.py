import copy
import json
import tempfile
import unittest
from pathlib import Path

from rl_agent import ue_n3b_oai_ul_cold_attach_confirmation_v1 as n3b


CONFIG_PATH = (
    n3b.ROOT / "rl_agent/configs/ue_n3b_oai_ul_cold_attach_confirmation_v1.json"
)
LIVE_CONFIG_PATH = (
    n3b.ROOT
    / "rl_agent/configs/ue_n3b_oai_ul_cold_attach_confirmation_live_v1.json"
)


def leaf_differences(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        differences = set()
        for key in set(left) | set(right):
            child = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                differences.add(child)
            else:
                differences.update(leaf_differences(left[key], right[key], child))
        return differences
    return set() if left == right else {prefix}


def runtime_state(*, enb0=-50.0, enb1=-50.0, ue0=-2.5, duplicate=False):
    rows = []
    values = (
        (0, "rfsimu_channel_enB0", enb0, "not"),
        (1, "rfsimu_channel_enB1", enb1, "not"),
        (2, "rfsimu_channel_ue0", ue0, "rfsimulator"),
    )
    for index, name, noise, owner in values:
        rows.extend([
            f"model {index} {name} type AWGN:",
            f"model owner: {owner}",
            f"path loss: 0.000000  noise: {noise:.6f}",
            "----------------",
        ])
    if duplicate:
        rows.extend([
            "model 3 rfsimu_channel_ue0 type AWGN:",
            "model owner: rfsimulator",
            "path loss: 0.000000  noise: -2.500000",
        ])
    return "\n".join(rows)


def valid_summary(*, median=6.0, service_pass=True):
    return {
        "attach_gate": {"status": "COLD_ATTACH_PDU_EXT_DN_GATE_PASSED", "passed": True},
        "transport": {"integrity_gate": True},
        "service_tail": {
            "status": "TAIL_ACCEPTED",
            "achieved_pusch_snr_db_p05": 5.5,
            "achieved_pusch_snr_db_median": median,
            "achieved_pusch_snr_db_p95": 6.0,
        },
        "service_window": {
            "full_nominal_window_observed": True,
            "exact_frozen_frame_set_pass": True,
            "required_expected_frames": 600,
            "expected_frames": 600,
            "integrity_gate": True,
            "primary_99_pass": service_pass,
            "no_one_second_outage_pass": service_pass,
        },
        "clean_recovery": {"passed": True},
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


class FakeTelnet:
    def __init__(self):
        self.commands = []

    def command(self, value):
        self.commands.append(value)
        if value.startswith("channelmod modify"):
            response = "model owner: rfsimulator\npath loss: 0 noise: -50"
        else:
            response = runtime_state(ue0=-50.0)
        return 1, 2, 3, 4, response


class ColdAttachConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_pending_config_validates_offline_and_live_fails_closed(self):
        n3b.validate_config(self.config, verify_hashes=True, require_live=False)
        changed = copy.deepcopy(self.config)
        changed["authority"]["live_oai_run_authorized"] = True
        changed["authority"]["live_socket_execution_authorized"] = True
        changed["authority"]["live_authority_basis"] = n3b.LIVE_AUTHORITY_BASIS
        with self.assertRaisesRegex(n3b.ColdAttachFailure, "still pending"):
            n3b.validate_config(changed, verify_hashes=False, require_live=True)

    def test_live_config_changes_only_authority_and_adjudication_pins(self):
        live = json.loads(LIVE_CONFIG_PATH.read_text(encoding="utf-8"))
        expected = {
            "authority.live_oai_run_authorized",
            "authority.live_socket_execution_authorized",
            "authority.live_authority_basis",
            "predecessors.n3a_adjudication.directory",
            "predecessors.n3a_adjudication.manifest_sha256",
            "predecessors.n3a_adjudication.terminal_sha256",
            "predecessors.n3a_adjudication.resolved_config_sha256",
            "predecessors.n3a_adjudication.source_config_sha256",
            "predecessors.n3a_adjudication.source_runner_sha256",
        }
        self.assertEqual(leaf_differences(self.config, live), expected)
        for key in (
            "directory", "manifest_sha256", "terminal_sha256",
            "resolved_config_sha256", "source_config_sha256", "source_runner_sha256",
        ):
            self.assertEqual(
                self.config["predecessors"]["n3a_adjudication"][key], n3b.PENDING
            )
        self.assertFalse(self.config["authority"]["live_oai_run_authorized"])
        self.assertFalse(self.config["authority"]["live_socket_execution_authorized"])

    def test_live_config_and_both_predecessors_verify(self):
        live = json.loads(LIVE_CONFIG_PATH.read_text(encoding="utf-8"))
        n3b.validate_config(live, verify_hashes=True, require_live=True)
        n3a_proof = n3b.verify_predecessor(live, "n3a_live_evidence")
        adjudication = n3b.verify_predecessor(live, "n3a_adjudication")
        self.assertEqual(
            n3a_proof["manifest_sha256"], n3b.EXPECTED_N3A_LIVE["manifest_sha256"]
        )
        self.assertEqual(
            adjudication["manifest_sha256"],
            "9eb86ac88a56c0e81848a95103a9692dbaabb28db0e92145eaa8c3bccfcbb8fc",
        )

    def test_live_prepare_emits_ready_without_runtime_execution(self):
        output = self.root / "live_ready_plan"
        runner = n3b.CampaignRunner(LIVE_CONFIG_PATH, output)
        self.assertEqual(runner.prepare(), 0)
        terminal = json.loads((output / f"{n3b.PLAN_READY}.json").read_text())
        self.assertFalse(terminal["adjudication_predecessor_pending"])
        self.assertTrue(terminal["live_authority_ready"])
        self.assertFalse(terminal["live_execution_blocked"])
        self.assertFalse(terminal["runtime_executed"])
        self.assertFalse(terminal["socket_executed"])
        self.assertFalse((output / "repetitions").exists())
        self.assertTrue(all(
            row["status"] == "READY_FOR_EXPLICIT_EXECUTE_LIVE"
            for row in terminal["plan"]
        ))

    def test_plan_is_three_fresh_cold_starts_with_zero_candidate_applications(self):
        rows = n3b.campaign_plan_rows(self.config)
        self.assertEqual(len(rows), 3)
        self.assertEqual([row["repetition_index"] for row in rows], [1, 2, 3])
        self.assertTrue(all(row["candidate_baked_before_ue_launch"] for row in rows))
        self.assertTrue(all(row["candidate_application_count"] == 0 for row in rows))
        self.assertTrue(all(row["expected_service_frames"] == 600 for row in rows))
        self.assertTrue(all(row["status"] == "BLOCKED_PENDING_PREREQUISITES" for row in rows))

    def test_prepare_only_is_create_only_and_truthfully_blocked(self):
        output = self.root / "plan"
        runner = n3b.CampaignRunner(CONFIG_PATH, output)
        self.assertEqual(runner.prepare(), 0)
        terminal_path = output / f"{n3b.PLAN_BLOCKED}.json"
        terminal = json.loads(terminal_path.read_text())
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertTrue(terminal["adjudication_predecessor_pending"])
        self.assertTrue(terminal["live_execution_blocked"])
        self.assertFalse(terminal["runtime_executed"])
        self.assertFalse((output / "repetitions").exists())
        self.assertEqual(terminal["manifest_sha256"], n3b.n2.sha256(output / "manifest.json"))
        n3b._verify_manifest_inventory(output, manifest)
        with self.assertRaises(n3b.ColdAttachFailure):
            n3b.CampaignRunner(CONFIG_PATH, output)
        args = n3b.parse_args(["--output-dir", str(self.root / "unused")])
        self.assertEqual(args.mode, n3b.PREPARE_ONLY)

    def test_execute_live_with_pending_config_fails_before_repetition_creation(self):
        output = self.root / "blocked_live"
        runner = n3b.CampaignRunner(CONFIG_PATH, output)
        self.assertEqual(runner.execute(), 1)
        self.assertFalse((output / "repetitions").exists())
        terminal = json.loads((output / "FAILED.json").read_text())
        self.assertEqual(terminal["repetitions_executed"], 0)
        self.assertFalse(terminal["cold_attach_bound_evaluated"])

    def test_plan_rows_become_ready_only_when_both_config_gates_are_frozen(self):
        changed = copy.deepcopy(self.config)
        changed["authority"].update({
            "live_oai_run_authorized": True,
            "live_socket_execution_authorized": True,
            "live_authority_basis": n3b.LIVE_AUTHORITY_BASIS,
        })
        adjudication = changed["predecessors"]["n3a_adjudication"]
        adjudication.update({
            "directory": "rl_agent/experiments/frozen",
            "manifest_sha256": "1" * 64,
            "terminal_sha256": "2" * 64,
            "resolved_config_sha256": "3" * 64,
            "source_config_sha256": "4" * 64,
            "source_runner_sha256": "5" * 64,
        })
        self.assertTrue(all(
            row["status"] == "READY_FOR_EXPLICIT_EXECUTE_LIVE"
            for row in n3b.campaign_plan_rows(changed)
        ))

    def test_run_local_channel_rewrite_changes_only_named_values(self):
        source = (
            n3b.ROOT
            / "OAI/openairinterface5g/targets/PROJECTS/GENERIC-NR-5GC/CONF/channelmod_rfsimu.conf"
        )
        original = source.read_text(encoding="utf-8")
        before = n3b.n2.sha256(source)
        rewritten = n3b.rewrite_channel_noise(original, self.config["startup_channel"])
        self.assertEqual(
            n3b.configured_channel_values(rewritten), self.config["startup_channel"]
        )
        self.assertEqual(n3b.n2.sha256(source), before)

    def test_materialized_configs_are_run_local_and_source_sealed(self):
        output = self.root / "rep"
        runner = n3b.ColdAttachRepetitionRunner(
            CONFIG_PATH, output, repetition_index=1,
            n3a_proof={}, adjudication_proof={},
        )
        gnb, ue = runner.materialize_configs()
        self.assertTrue(gnb.is_relative_to(output))
        self.assertTrue(ue.is_relative_to(output))
        self.assertEqual(
            n3b.configured_channel_values(
                (output / "runtime/effective_channel_cold_attach_minus2p5.conf").read_text()
            ),
            self.config["startup_channel"],
        )
        self.assertTrue(runner.source_integrity()["unchanged"])

    def test_runtime_snapshot_requires_all_three_exact_unambiguous_models(self):
        models = n3b.validate_runtime_channel_state(
            runtime_state(), self.config["startup_channel"]
        )
        self.assertEqual(set(models), set(self.config["startup_channel"]))
        with self.assertRaisesRegex(n3b.ColdAttachFailure, "runtime noise mismatch"):
            n3b.validate_runtime_channel_state(
                runtime_state(enb1=-40.0), self.config["startup_channel"]
            )
        with self.assertRaisesRegex(n3b.ColdAttachFailure, "ambiguous"):
            n3b.validate_runtime_channel_state(
                runtime_state(duplicate=True), self.config["startup_channel"]
            )

    def test_single_restore_guard_does_not_count_as_candidate_application(self):
        runner = n3b.ColdAttachRepetitionRunner(
            CONFIG_PATH, self.root / "restore", repetition_index=1,
            n3a_proof={}, adjudication_proof={},
        )
        runner.telnet = FakeTelnet()
        runner.restore_clean_once(2)
        self.assertEqual(runner.application_count, 0)
        self.assertEqual(runner.restore_application_count, 1)
        self.assertTrue(runner.restored)
        with self.assertRaisesRegex(n3b.ColdAttachFailure, "only once"):
            runner.restore_clean_once(2)

    def test_exact_service_and_achieved_snr_jointly_define_pass(self):
        result = n3b.classify_repetition(valid_summary())
        self.assertTrue(result["evidence_valid_for_aggregation"])
        self.assertTrue(result["authoritative_service_gate_pass"])
        self.assertTrue(result["achieved_snr_gate_pass"])
        self.assertTrue(result["joint_candidate_confirmation_pass"])
        self.assertEqual(result["achieved_pusch_snr_db_p50"], 6.0)

    def test_out_of_band_achieved_snr_is_valid_but_not_candidate_pass(self):
        result = n3b.classify_repetition(valid_summary(median=6.6))
        self.assertTrue(result["evidence_valid_for_aggregation"])
        self.assertTrue(result["authoritative_service_gate_pass"])
        self.assertFalse(result["achieved_snr_gate_pass"])
        self.assertFalse(result["joint_candidate_confirmation_pass"])
        self.assertEqual(
            result["classified_outcome"],
            "ACHIEVED_SNR_OUTSIDE_FROZEN_CANDIDATE_BAND",
        )

    def test_599_frame_window_or_nonzero_candidate_application_never_passes(self):
        summary = valid_summary()
        summary["service_window"]["expected_frames"] = 599
        result = n3b.classify_repetition(summary)
        self.assertFalse(result["evidence_valid_for_aggregation"])
        summary = valid_summary()
        summary["candidate_application_count"] = 1
        result = n3b.classify_repetition(summary)
        self.assertFalse(result["evidence_valid_for_aggregation"])

    def test_clean_attach_and_service_failures_are_valid_nonpasses(self):
        attach = valid_summary()
        attach.update({
            "attach_gate": {
                "status": "COLD_ATTACH_OR_PDU_EXT_DN_GATE_FAILED",
                "passed": False,
                "ran_processes_alive_at_terminal": True,
                "core_ready_at_terminal": True,
            },
            "transport": None, "service_tail": None, "service_window": None,
            "clean_recovery": {"status": "NOT_APPLICABLE_NO_PDU_SESSION", "passed": None},
        })
        result = n3b.classify_repetition(attach)
        self.assertTrue(result["evidence_valid_for_aggregation"])
        self.assertEqual(result["classified_outcome"], "COLD_ATTACH_FAILED")
        service = n3b.classify_repetition(valid_summary(service_pass=False))
        self.assertTrue(service["evidence_valid_for_aggregation"])
        self.assertEqual(service["classified_outcome"], "SERVICE_GATE_FAILED")

    def test_aggregation_requires_three_valid_units_and_three_joint_passes(self):
        passed = [{**n3b.classify_repetition(valid_summary())} for _ in range(3)]
        result = n3b.aggregate_results(passed)
        self.assertEqual(result["status"], n3b.CAMPAIGN_PASSED)
        self.assertTrue(result["cold_attach_3_of_3_pass"])
        self.assertTrue(result["authoritative_service_gate_3_of_3_pass"])
        self.assertTrue(result["achieved_snr_band_3_of_3_pass"])
        self.assertTrue(result["joint_candidate_confirmation_3_of_3_pass"])
        passed[2] = n3b.classify_repetition(valid_summary(median=6.6))
        result = n3b.aggregate_results(passed)
        self.assertEqual(result["status"], n3b.CAMPAIGN_NOT_3_OF_3)
        self.assertEqual(result["cold_attach_passes"], 3)
        self.assertEqual(result["authoritative_service_gate_passes"], 3)
        self.assertEqual(result["achieved_snr_band_passes"], 2)
        self.assertEqual(result["joint_candidate_confirmation_passes"], 2)


if __name__ == "__main__":
    unittest.main()
