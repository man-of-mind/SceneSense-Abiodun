import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

from rl_agent import ue_n3_oai_ul_command_calibration_v1 as calibration
from rl_agent import ue_n3a_oai_ul_sustain_replication_v1 as n3a


CONFIG_PATH = (
    n3a.ROOT / "rl_agent/configs/ue_n3a_oai_ul_sustain_replication_v1.json"
)


def sustained_summary(*, median=6.0, ratio_pass=True, gap_pass=True):
    return {
        "status": calibration.RUNG_CAPTURED,
        "commanded_noise_power_db": -2.5,
        "tail": {
            "status": "TAIL_ACCEPTED",
            "achieved_pusch_snr_db_median": median,
        },
        "tail_service": {
            "full_nominal_window_observed": True,
            "exact_frozen_frame_set_pass": True,
            "required_expected_frames": 600,
            "expected_frames": 600,
            "integrity_gate": True,
            "primary_99_pass": ratio_pass,
            "no_one_second_outage_pass": gap_pass,
        },
        "transport": {
            "integrity_gate": True,
            # The complete clean+transition+tail+recovery capture is diagnostic.
            "primary_99_pass": False,
            "no_one_second_outage_pass": False,
        },
        "clean_recovery": {"passed": True},
        "clean_restore_verified": True,
        "candidate_application_count": 1,
        "hard_loss_reason": None,
        "receiver_service_outage_detected": False,
    }


def hard_loss_summary(status, reason, *, receiver_outage=False):
    return {
        "status": status,
        "commanded_noise_power_db": -2.0,
        "tail": None,
        "tail_service": None,
        "transport": {"integrity_gate": True},
        "clean_recovery": {"passed": True},
        "clean_restore_verified": True,
        "candidate_application_count": 1,
        "hard_loss_reason": reason,
        "receiver_service_outage_detected": receiver_outage,
    }


class SustainReplicationTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_frozen_config_and_runtime_hashes_validate(self):
        n3a.validate_config(self.config, verify_hashes=True)
        self.assertTrue(
            issubclass(n3a.ReplicationRunner, calibration.RungRunner)
        )
        self.assertEqual(
            self.config["authority"]["live_authority_basis"],
            n3a.AUTHORITY_BASIS,
        )
        self.assertEqual(
            self.config["rung"]["expected_tail_frames"], 600
        )

    def test_plan_alternates_paired_conditions_three_times(self):
        rows = n3a.campaign_plan_rows(self.config)
        self.assertEqual(len(rows), 6)
        self.assertEqual(
            [(row["repetition_index"], row["commanded_noise_power_db"])
             for row in rows],
            [(1, -2.5), (1, -2.0), (2, -2.5), (2, -2.0),
             (3, -2.5), (3, -2.0)],
        )
        self.assertTrue(all(row["fresh_ran_epoch_required"] for row in rows))
        self.assertTrue(all(row["expected_tail_frames"] == 600 for row in rows))

    def test_prepare_only_is_default_create_only_and_fully_sealed(self):
        output = self.root / "plan"
        runner = n3a.CampaignRunner(CONFIG_PATH, output)
        self.assertEqual(runner.prepare(), 0)
        terminal_path = output / f"{n3a.PLAN_FROZEN}.json"
        manifest_path = output / "manifest.json"
        terminal = json.loads(terminal_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        self.assertFalse(terminal["runtime_executed"])
        self.assertFalse(terminal["socket_executed"])
        self.assertEqual(terminal["manifest_sha256"], n3a.n2.sha256(manifest_path))
        self.assertEqual(manifest["runner_sha256"], n3a.n2.sha256(n3a.Path(n3a.__file__)))
        n3a._verify_manifest_inventory(output, manifest)
        self.assertFalse((output / "repetitions").exists())
        with self.assertRaises(n3a.SustainReplicationFailure):
            n3a.CampaignRunner(CONFIG_PATH, output)
        args = n3a.parse_args(["--output-dir", str(self.root / "unused")])
        self.assertEqual(args.mode, n3a.PREPARE_ONLY)

    def test_both_predecessors_verify_complete_frozen_inventories(self):
        clean = n3a.verify_predecessor(self.config, "clean_control")
        search = n3a.verify_predecessor(self.config, "command_search")
        self.assertEqual(clean["verified_output_count"], 99)
        self.assertEqual(search["verified_output_count"], 529)
        self.assertEqual(
            search["manifest_sha256"],
            n3a.EXPECTED_PREDECESSORS["command_search"]["manifest_sha256"],
        )

    def test_sustained_gate_uses_exact_tail_not_whole_capture(self):
        condition = self.config["campaign"]["conditions"][0]
        result = n3a.classify_repetition_summary(
            sustained_summary(), condition
        )
        self.assertTrue(result["evidence_valid"])
        self.assertTrue(result["matches_expected_outcome"])
        self.assertTrue(result["tail_primary_99_and_no_gap_pass"])
        self.assertFalse(result["whole_capture_primary_delivery_gate_applied"])

    def test_valid_sustained_target_surprise_is_not_infrastructure_failure(self):
        condition = self.config["campaign"]["conditions"][0]
        result = n3a.classify_repetition_summary(
            sustained_summary(median=6.6), condition
        )
        self.assertTrue(result["evidence_valid"])
        self.assertFalse(result["matches_expected_outcome"])
        self.assertEqual(result["classified_outcome"], "SUSTAINED_SERVICE")

    def test_receiver_only_service_failure_never_passes_as_sustained(self):
        condition = self.config["campaign"]["conditions"][0]
        result = n3a.classify_repetition_summary(
            sustained_summary(ratio_pass=False, gap_pass=False), condition
        )
        self.assertTrue(result["evidence_valid"])
        self.assertFalse(result["matches_expected_outcome"])
        self.assertEqual(result["classified_outcome"], "SERVICE_GATE_FAILED")

    def test_all_three_recognized_hard_loss_classes_are_valid(self):
        condition = self.config["campaign"]["conditions"][1]
        cases = (
            (calibration.RUNG_HARD_LOSS, "CURRENT_RNTI_PUSCH_SILENCE", True),
            (calibration.RUNG_DETACHED, "UE_TUNNEL_IDENTITY_LOST", False),
            (calibration.RUNG_IDENTITY_DISCONTINUITY, "RNTI_CHANGED", False),
        )
        for status, reason, outage in cases:
            with self.subTest(reason=reason):
                result = n3a.classify_repetition_summary(
                    hard_loss_summary(status, reason, receiver_outage=outage),
                    condition,
                )
                self.assertTrue(result["evidence_valid"])
                self.assertTrue(result["matches_expected_outcome"])
                self.assertEqual(result["classified_outcome"], "HARD_SERVICE_LOSS")

    def test_pusch_silence_without_receiver_outage_is_unconfirmed(self):
        condition = self.config["campaign"]["conditions"][1]
        result = n3a.classify_repetition_summary(
            hard_loss_summary(
                calibration.RUNG_HARD_LOSS,
                "CURRENT_RNTI_PUSCH_SILENCE",
                receiver_outage=False,
            ),
            condition,
        )
        self.assertFalse(result["evidence_valid"])
        self.assertFalse(result["matches_expected_outcome"])

    def test_bad_restore_or_recovery_invalidates_scientific_evidence(self):
        condition = self.config["campaign"]["conditions"][0]
        summary = sustained_summary()
        summary["clean_recovery"] = {"passed": False}
        result = n3a.classify_repetition_summary(summary, condition)
        self.assertFalse(result["evidence_valid"])
        self.assertIn("CLEAN_RECOVERY_NOT_VERIFIED", result["validation_errors"])

    def test_frozen_window_evaluator_selects_exact_600_dynamic_frames(self):
        sender = self.root / "sender.csv"
        events = self.root / "events.jsonl"
        origin_s = 1_800_000_000.0
        with sender.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "frame_index", "chunk_index", "wall_time_s", "elapsed_s",
                "scheduled_frame_time_s",
            ])
            writer.writeheader()
            for frame in range(800):
                scheduled = frame / 10.0
                writer.writerow({
                    "frame_index": frame,
                    "chunk_index": 0,
                    "wall_time_s": origin_s + scheduled,
                    "elapsed_s": scheduled,
                    "scheduled_frame_time_s": scheduled,
                })
        with events.open("w", encoding="utf-8") as handle:
            for frame in range(800):
                handle.write(json.dumps({
                    "event_type": "datagram",
                    "status": "ACCEPTED_UNIQUE",
                    "source_ip": "192.168.70.134",
                    "frame_index": frame,
                    "receiver_wall_time_ns": int(
                        (origin_s + frame / 10.0 + 0.01) * 1e9
                    ),
                }) + "\n")
        # A small observed start offset intentionally shifts the nominal frame
        # indices.  The contract is 600 unique scheduled frames, not literals.
        start_ns = int((origin_s + 15.00001) * 1e9)
        result = calibration.evaluate_tail_service(
            sender,
            events,
            start_wall_ns=start_ns,
            end_wall_ns=start_ns + 60_000_100_000,
            fps=10.0,
            expected_tail_frames=600,
            expected_source_ip="192.168.70.134",
            structural_integrity=True,
            gates=self.config["transport_gates"],
        )
        self.assertTrue(result["exact_frozen_frame_set_pass"])
        self.assertEqual(result["expected_frames"], 600)
        self.assertEqual(result["expected_frame_indices"][0], 151)
        self.assertEqual(result["expected_frame_indices"][-1], 750)
        self.assertTrue(result["primary_99_pass"])

    def test_repetition_writer_seals_valid_scientific_surprise(self):
        clean = n3a.verify_predecessor(self.config, "clean_control")
        search = n3a.verify_predecessor(self.config, "command_search")
        directory = self.root / "rep"
        runner = n3a.ReplicationRunner(
            CONFIG_PATH,
            directory,
            condition_index=0,
            repetition_index=1,
            clean_control_proof=clean,
            command_search_proof=search,
        )
        summary = sustained_summary(median=6.6)
        runner.application_count = 1
        runner.restored = True
        runner.write_manifest_terminal(calibration.RUNG_CAPTURED, summary)
        terminal_path = directory / f"{n3a.VALID_SURPRISE_CAPTURED}.json"
        terminal = json.loads(terminal_path.read_text())
        manifest = json.loads((directory / "manifest.json").read_text())
        self.assertTrue(terminal["evidence_valid_for_aggregation"])
        self.assertFalse(terminal["matches_expected_outcome"])
        self.assertEqual(
            terminal["manifest_sha256"], n3a.n2.sha256(directory / "manifest.json")
        )
        n3a._verify_manifest_inventory(directory, manifest)

    def test_config_rejects_narrowed_loss_contract(self):
        changed = copy.deepcopy(self.config)
        changed["campaign"]["conditions"][1]["accepted_hard_loss_reasons"] = [
            "CURRENT_RNTI_PUSCH_SILENCE"
        ]
        with self.assertRaises(n3a.SustainReplicationFailure):
            n3a.validate_config(changed, verify_hashes=False)

    def test_valid_mixed_results_aggregate_to_unstable_review(self):
        rows = []
        for repetition in range(1, 4):
            rows.extend([
                {
                    "condition_id": "SUSTAIN_CANDIDATE_MINUS2P5",
                    "repetition_index": repetition,
                    "evidence_valid_for_aggregation": True,
                    "matches_expected_outcome": repetition != 2,
                    "tail": {"achieved_pusch_snr_db_median": 6.0},
                },
                {
                    "condition_id": "ADJACENT_HARD_LOSS_MINUS2P0",
                    "repetition_index": repetition,
                    "evidence_valid_for_aggregation": True,
                    "matches_expected_outcome": True,
                    "tail": None,
                },
            ])
        result = n3a.classify_campaign_results(rows)
        self.assertEqual(result["status"], n3a.UNSTABLE_REVIEW_REQUIRED)
        self.assertEqual(result["sustain_candidate_expected_matches"], 2)
        self.assertTrue(result["valid_mixed_outcomes_retained"])

    def test_prepare_validation_failure_gets_hashed_failure_terminal(self):
        changed = copy.deepcopy(self.config)
        changed["runtime_seals"][0]["sha256"] = "0" * 64
        config_path = self.root / "bad_config.json"
        config_path.write_text(json.dumps(changed), encoding="utf-8")
        output = self.root / "failed_plan"
        runner = n3a.CampaignRunner(config_path, output)
        self.assertEqual(runner.prepare(), 1)
        terminal = json.loads((output / "FAILED.json").read_text())
        manifest_path = output / "manifest.json"
        self.assertFalse(terminal["runtime_executed"])
        self.assertEqual(terminal["manifest_sha256"], n3a.n2.sha256(manifest_path))


if __name__ == "__main__":
    unittest.main()
