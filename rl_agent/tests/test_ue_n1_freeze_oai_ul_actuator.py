from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import rl_agent.ue_n1_freeze_oai_ul_actuator as n1
from rl_agent.ue_n1_freeze_oai_ul_actuator import (
    DEFAULT_CONFIG,
    EXPECTED_TELEMETRY,
    FROZEN_STATUS,
    InterfaceFreezeError,
    assemble,
    load_config,
    validate_bundle,
    validate_channel_command_literal,
)


ROOT = Path(__file__).resolve().parents[2]
FIXED_NOW = "2026-08-21T05:00:00+00:00"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_mutated_config(temp: Path, mutate) -> Path:
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["repository_root"] = str(ROOT)
    mutate(config)
    path = temp / "mutated.json"
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


class UEN1FreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = Path(tempfile.mkdtemp(prefix="ue-n1-interface-tests-"))
        cls.output = cls.temp / "bundle_a"
        assemble(DEFAULT_CONFIG, cls.output, now=FIXED_NOW)
        cls.config = load_config(DEFAULT_CONFIG)
        cls.manifest = validate_bundle(cls.output)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp, ignore_errors=True)

    def test_shipped_config_loads_and_all_nested_contracts_are_exact(self) -> None:
        self.assertEqual(self.config["schema"], n1.CONFIG_SCHEMA)
        self.assertEqual(set(self.config), n1.EXPECTED_TOP_LEVEL_KEYS)
        self.assertEqual(self.config["authority"], n1.EXPECTED_AUTHORITY)
        self.assertEqual(
            {event["id"] for event in self.config["telemetry"]["events"]},
            set(EXPECTED_TELEMETRY),
        )
        self.assertEqual(self.config["output"]["terminal_status"], FROZEN_STATUS)
        self.assertEqual(self.config["output"]["next_checklist_item"], "UE-N2")

    def test_bundle_is_interface_only_and_next_is_n2(self) -> None:
        terminal = json.loads(
            (self.output / "UE_N1_INTERFACE_FROZEN.json").read_text(encoding="utf-8")
        )
        self.assertEqual(terminal["status"], "FROZEN_INTERFACE_ONLY")
        self.assertEqual(terminal["next_checklist_item"], "UE-N2")
        self.assertEqual(terminal["numeric_calibration_status"], "NOT_PERFORMED")
        self.assertEqual(terminal["numeric_bounds_status"], "NOT_DEFINED")
        self.assertEqual(terminal["direct_ul_bler_status"], "UNAVAILABLE_UNRESOLVED")
        self.assertFalse(terminal["runtime_executed"])
        self.assertFalse(terminal["socket_executed"])
        self.assertFalse(terminal["oai_run"])
        self.assertFalse(terminal["carla_run"])

    def test_exact_single_ue_actuator_and_clean_attach_are_frozen(self) -> None:
        actuator = self.config["actuator"]
        self.assertEqual(self.config["scope"]["ue_count"], 1)
        self.assertEqual(actuator["channel_model_name"], "rfsimu_channel_ue0")
        self.assertEqual(actuator["model_index_binding"], "RESOLVE_EXACT_NAME_EACH_GNB_SESSION")
        self.assertFalse(actuator["hardcoded_model_index_authorized"])
        self.assertEqual(actuator["mutable_parameter"], "noise_power_dB")
        self.assertEqual(actuator["fixed_path_loss_db"], 0)
        self.assertEqual(actuator["global_noise_requirement"], "UNSET")
        lifecycle = self.config["attach_lifecycle"]
        self.assertEqual(lifecycle["initial_and_restore_channel_command_db"], -50)
        self.assertEqual(lifecycle["source_template_initial_channel_command_db"], -10)
        self.assertFalse(lifecycle["source_template_is_effective_runtime_config"])

    def test_persistent_monotonic_no_catch_up_and_ack_semantics_are_frozen(self) -> None:
        control = self.config["control_transport"]
        self.assertEqual(control["n2_connection_lifecycle"], "ONE_PERSISTENT_CONNECTION_PER_TRACE")
        self.assertFalse(control["reconnect_per_command_authorized"])
        self.assertEqual(control["connection_loss_trace_result"], "FAILED")
        self.assertFalse(control["cleanup_success_can_change_failed_to_pass"])
        schedule = self.config["schedule"]
        self.assertEqual(schedule["clock"], "time.monotonic_ns")
        self.assertEqual(schedule["period_ms"], 100)
        self.assertEqual(schedule["catch_up_policy"], "NEVER_BURST_OBSOLETE_COMMANDS")
        timing = self.config["command_timing"]
        self.assertIn("desired_achieved_pusch_snr_db", timing["required_fields"])
        self.assertNotIn("target_snr_db", timing["required_fields"])
        self.assertIn("control_session_id", timing["required_fields"])
        self.assertIn("resolved_model_name", timing["required_fields"])
        self.assertIn("echoed_owner", timing["required_fields"])
        self.assertNotIn("echoed_model_name", timing["required_fields"])
        self.assertIn("ACK_UPPER_BOUND", timing["response_received_semantics"])
        self.assertEqual(timing["prohibited_fields"], ["command_applied_at", "command_applied_at_ns"])

    def test_signal_and_bler_observation_semantics_are_not_conflated(self) -> None:
        signal = self.config["signal_contract"]
        self.assertEqual(
            set(signal),
            {
                "desired_achieved_pusch_snr_db",
                "channel_command_db",
                "instantaneous_mac_normalized_pusch_snr_db",
                "cqi_domain_pusch_snr_db",
                "scheduler_ema_snr_db",
                "selected_mcs",
                "final_mcs",
            },
        )
        self.assertIn("NOT_RAW_PHY_SNR", signal["instantaneous_mac_normalized_pusch_snr_db"])
        scheduler = self.config["scheduler"]
        self.assertEqual(scheduler["power_control_target_parameter"], "pusch_TargetSNRx10")
        self.assertIn("NOT_DESIRED", scheduler["power_control_target_semantics"])
        telemetry = self.config["telemetry"]
        self.assertEqual(
            telemetry["ul_outcome_contract"]["direct_ul_bler_status"],
            "UNAVAILABLE_UNRESOLVED_CURRENT_SINR_TRACE",
        )
        self.assertFalse(telemetry["ul_outcome_contract"]["missing_direct_bler_is_zero"])
        self.assertEqual(
            telemetry["ul_outcome_contract"]["ue_grant_round_semantics"],
            "RETRANSMISSION_PROXY_ONLY",
        )
        self.assertEqual(
            telemetry["observation_semantics"]["missing_pusch_snr"],
            "UNAVAILABLE_OR_DTX_NEVER_ZERO",
        )

    def test_event_time_availability_and_join_contract_are_causal(self) -> None:
        telemetry = self.config["telemetry"]
        self.assertEqual(telemetry["source_event_timestamp_clock"], "CLOCK_REALTIME")
        self.assertIn("NOT_RECORDER_AVAILABILITY", telemetry["csv_time_semantics"])
        self.assertEqual(
            telemetry["live_ingest_availability_fields"],
            ["ingest_available_wall_time_ns", "ingest_available_monotonic_ns"],
        )
        join = telemetry["join_contract"]
        self.assertFalse(join["frame_slot_alone_authorized"])
        self.assertTrue(join["decision_and_scheduled_frame_slot_distinct"])
        self.assertIn("control_session_id", join["required_identity_and_time"])

    def test_a4_and_failed_two_ue_evidence_are_pinned_honestly(self) -> None:
        self.assertEqual(
            self.config["predecessor"]["manifest_sha256"],
            "ea044dcc31632f3729f9ddae11311ab980598c6120f1aacad09034bd32698128",
        )
        evidence = self.config["mechanism_evidence"]
        self.assertEqual(evidence["claim"], "TWO_UE_CHANNELMOD_READ_MODIFY_READ_MECHANISM_ONLY")
        self.assertEqual(evidence["overall_experiment_status"], "FAILED_HOLD")
        self.assertFalse(evidence["single_ue_calibration_claimed"])
        self.assertFalse(evidence["cadence_claimed"])
        self.assertFalse(evidence["numeric_bound_claimed"])
        self.assertEqual(
            self.manifest["gates"]["two_ue_mechanism_evidence"],
            "PASS_MECHANISM_ONLY_OVERALL_RUN_FAILED_HOLD",
        )

    def test_current_runtime_artifacts_are_sealed_but_recheck_is_deferred(self) -> None:
        runtime = self.config["runtime_artifacts"]
        self.assertFalse(runtime["execution_claimed"])
        self.assertIn("REVERIFY_AT_UE_N2_PREFLIGHT", runtime["authority"])
        self.assertEqual(
            set(runtime["files"]),
            {"gnb_softmodem", "ue_softmodem", "telnet_server_library"},
        )
        kinds = {(row["kind"], row.get("label")) for row in self.manifest["inputs"]}
        self.assertIn(("runtime_artifact", "gnb_softmodem"), kinds)
        self.assertIn(("runtime_artifact", "telnet_server_library"), kinds)

    def test_command_literal_validation_is_lexical_not_a_numeric_bound(self) -> None:
        for valid in ("0", "-50", "12", "-0.5", "24.25"):
            with self.subTest(valid=valid):
                self.assertEqual(str(validate_channel_command_literal(valid)), valid)
        for invalid in (
            True, 1.0, "", "-0", "00", "01", "1.0", "1e2", "+1", " nan",
            "NaN", "Infinity", "-inf", "1 ", " 1", "1junk", "1\nshow current",
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(InterfaceFreezeError):
                    validate_channel_command_literal(invalid)

    def test_create_only_and_deterministic_fixed_time(self) -> None:
        with self.assertRaisesRegex(InterfaceFreezeError, "refusing to overwrite"):
            assemble(DEFAULT_CONFIG, self.output, now=FIXED_NOW)
        other = self.temp / "bundle_b"
        assemble(DEFAULT_CONFIG, other, now=FIXED_NOW)
        for name in ("resolved_config.json", "REPORT.md", "manifest.json", "UE_N1_INTERFACE_FROZEN.json"):
            self.assertEqual(sha256(self.output / name), sha256(other / name), name)

    def test_failed_temp_validation_publishes_nothing(self) -> None:
        target = self.temp / "must_not_publish"
        with mock.patch.object(n1, "validate_bundle", side_effect=InterfaceFreezeError("injected")):
            with self.assertRaisesRegex(InterfaceFreezeError, "injected"):
                assemble(DEFAULT_CONFIG, target, now=FIXED_NOW)
        self.assertFalse(target.exists())

    def test_mutations_extra_keys_and_output_escape_fail_before_write(self) -> None:
        cases = [
            (lambda c: c.__setitem__("extra_calibration", {"mapping": [1, 2]}), "top-level"),
            (lambda c: c["actuator"].__setitem__("hardcoded_index", 2), "physical actuator"),
            (lambda c: c["schedule"].__setitem__("period_ms", 101), "100-ms"),
            (lambda c: c["telemetry"]["ul_outcome_contract"].__setitem__("missing_direct_bler_is_zero", True), "UL outcome"),
            (lambda c: c["output"].__setitem__("report_md", "../escape.md"), "output/next-item"),
        ]
        for index, (mutate, message) in enumerate(cases):
            case_dir = self.temp / f"mutate_{index}"
            case_dir.mkdir()
            path = write_mutated_config(case_dir, mutate)
            with mock.patch.object(n1, "FROZEN_CONFIG_SHA256", sha256(path)):
                with self.subTest(message=message):
                    with self.assertRaisesRegex(InterfaceFreezeError, message):
                        load_config(path)
            self.assertFalse((self.temp / "escape.md").exists())

    def test_alternate_self_consistent_config_is_rejected_by_canonical_seal(self) -> None:
        case_dir = self.temp / "retarget"
        case_dir.mkdir()
        path = write_mutated_config(
            case_dir,
            lambda c: c["predecessor"].__setitem__("manifest_sha256", "0" * 64),
        )
        with self.assertRaisesRegex(InterfaceFreezeError, "frozen config seal"):
            load_config(path)

    def test_tamper_and_self_consistent_report_reseal_are_rejected(self) -> None:
        tampered = self.temp / "tampered_report"
        shutil.copytree(self.output, tampered)
        report = tampered / "REPORT.md"
        report.write_text(report.read_text(encoding="utf-8") + "false claim\n", encoding="utf-8")
        manifest_path = tampered / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        record = next(row for row in manifest["outputs"] if row["path"] == "REPORT.md")
        record.update({"sha256": sha256(report), "bytes": report.stat().st_size})
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        terminal_path = tampered / "UE_N1_INTERFACE_FROZEN.json"
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        terminal["manifest_sha256"] = sha256(manifest_path)
        terminal_path.write_text(json.dumps(terminal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(InterfaceFreezeError, "deterministic report"):
            validate_bundle(tampered)

    def test_self_consistent_deferred_or_duplicate_output_reseal_is_rejected(self) -> None:
        for name, mutation, message in (
            (
                "deferred",
                lambda manifest: manifest["deferred"].append("NUMERIC_MAPPING_ALREADY_DONE"),
                "deferred-work",
            ),
            (
                "duplicate",
                lambda manifest: manifest["outputs"].append(dict(manifest["outputs"][0])),
                "output seal set",
            ),
        ):
            tampered = self.temp / f"tampered_{name}"
            shutil.copytree(self.output, tampered)
            manifest_path = tampered / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            mutation(manifest)
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            terminal_path = tampered / "UE_N1_INTERFACE_FROZEN.json"
            terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
            terminal["manifest_sha256"] = sha256(manifest_path)
            terminal_path.write_text(json.dumps(terminal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            with self.subTest(name=name):
                with self.assertRaisesRegex(InterfaceFreezeError, message):
                    validate_bundle(tampered)

    def test_direct_cli_uses_no_runtime_or_socket(self) -> None:
        output = self.temp / "direct_cli"
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "rl_agent/ue_n1_freeze_oai_ul_actuator.py"),
                "--output-dir",
                str(output),
            ],
            cwd=self.temp,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        terminal = json.loads((output / "UE_N1_INTERFACE_FROZEN.json").read_text())
        self.assertFalse(terminal["runtime_executed"])
        self.assertFalse(terminal["socket_executed"])


if __name__ == "__main__":
    unittest.main()
