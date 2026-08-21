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

import rl_agent.ue_n1_freeze_oai_ul_actuator_v2 as n2
from rl_agent.ue_n1_freeze_oai_ul_actuator_v2 import (
    DEFAULT_CONFIG,
    InterfaceV2Error,
    assemble,
    load_config,
    validate_bundle,
    validate_commanded_noise_power_literal,
    validate_created_at,
)


ROOT = Path(__file__).resolve().parents[2]
FIXED_NOW = "2026-08-21T06:00:00.000000Z"
V1_BUNDLE = ROOT / "rl_agent/registries/ue_n1_oai_ul_actuator_interface_v1"
V1_HASHES = {
    "manifest.json": "75d6a380bcd52e5ddc1a4d2c52154d94f42b8471b778c60a7929ff966e0136d6",
    "UE_N1_INTERFACE_FROZEN.json": (
        "f1ec2f1989368d63fb062f21c0f62f044fcd77f3bc9e26c7f437d6cb397222d3"
    ),
    "resolved_config.json": (
        "00aed91aed19c3f71fd759dc8f74a6af863b94e8b9f0f6242a5c9921b56866bf"
    ),
    "REPORT.md": "83bd7a11076c9ef069849cfc9c0e7f9e39195df8f24b13fbef6e75fbaeaaa327",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mutate_config(temp: Path, mutate) -> Path:
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["repository_root"] = str(ROOT)
    mutate(config)
    path = temp / "mutated_v2.json"
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


class UEN1V2FreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = Path(tempfile.mkdtemp(prefix="ue-n1-v2-tests-"))
        cls.output = cls.temp / "bundle_a"
        assemble(DEFAULT_CONFIG, cls.output, now=FIXED_NOW)
        cls.config = load_config(DEFAULT_CONFIG)
        cls.manifest = validate_bundle(cls.output)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp, ignore_errors=True)

    def test_v2_supersedes_but_does_not_mutate_v1(self) -> None:
        supersedes = self.config["supersedes"]
        self.assertEqual(supersedes["authority_status"], "SUPERSEDED_PRE_FINAL_OBSERVATION_AUDIT")
        self.assertFalse(supersedes["v1_bytes_mutable"])
        self.assertFalse(supersedes["v1_remains_execution_authority"])
        for name, expected in V1_HASHES.items():
            with self.subTest(name=name):
                self.assertEqual(sha256(V1_BUNDLE / name), expected)
        self.assertEqual(
            self.manifest["supersession"]["supersedes_manifest_sha256"],
            V1_HASHES["manifest.json"],
        )

    def test_v2_terminal_is_final_interface_only_and_next_n2(self) -> None:
        terminal = json.loads(
            (self.output / "UE_N1_INTERFACE_V2_FROZEN.json").read_text(encoding="utf-8")
        )
        self.assertEqual(terminal["status"], "FROZEN_INTERFACE_ONLY")
        self.assertEqual(terminal["next_checklist_item"], "UE-N2")
        self.assertEqual(terminal["canonical_command_field"], "commanded_noise_power_db")
        self.assertEqual(terminal["oai_mutable_parameter"], "noise_power_dB")
        self.assertEqual(terminal["policy_observation_binding"], "UNBOUND_UNTIL_MEASURED_UE_VISIBLE_FEEDBACK_PATH")
        self.assertEqual(terminal["direct_ul_bler_status"], "UNAVAILABLE_UNRESOLVED")
        self.assertFalse(terminal["runtime_executed"])
        self.assertFalse(terminal["socket_executed"])

    def test_canonical_command_is_renamed_and_excluded_from_policy(self) -> None:
        serialized = json.dumps(self.config, sort_keys=True)
        self.assertNotIn("channel_command_db", serialized)
        self.assertNotIn("target_snr_db", serialized)
        actuator = self.config["actuator"]
        self.assertEqual(actuator["oai_mutable_parameter"], "noise_power_dB")
        self.assertEqual(actuator["canonical_command_field"], "commanded_noise_power_db")
        self.assertIn("{commanded_noise_power_db}", actuator["modify_command_template"])
        causal = self.config["causal_classification"]
        self.assertEqual(
            causal["control_and_evaluation_only_fields"],
            ["desired_achieved_pusch_snr_db", "commanded_noise_power_db"],
        )
        self.assertFalse(causal["control_fields_in_policy_state_authorized"])

    def test_policy_availability_is_not_collector_ingest(self) -> None:
        policy = self.config["policy_availability"]
        self.assertFalse(policy["collector_ingest_is_policy_availability"])
        self.assertEqual(
            policy["admission_predicate"],
            "policy_observation_available_monotonic_ns <= decision_cutoff_monotonic_ns "
            "AND observation.ran_epoch_id == decision.ran_epoch_id AND "
            "observation.control_session_id == decision.control_session_id",
        )
        self.assertTrue(policy["availability_must_be_measured_non_null"])
        self.assertTrue(policy["feedback_path_must_be_ue_visible_and_measured"])
        self.assertFalse(
            self.config["causal_classification"]["collector_evidence_is_ue_policy_observation"]
        )

    def test_raw_event_envelope_and_missingness_are_explicit(self) -> None:
        envelope = self.config["raw_event_envelope"]
        for field in (
            "ran_epoch_id", "source_event_index", "source_event_realtime_sec",
            "source_event_realtime_nsec", "source_event_timestamp_ns",
            "unwrapped_absolute_slot", "collector_ingest_wall_time_ns",
            "collector_ingest_monotonic_ns", "raw_event_sha256", "missing_reason_code",
        ):
            self.assertIn(field, envelope["required_fields"])
        self.assertIsNone(envelope["missing_numeric_value"])
        self.assertTrue(envelope["missing_reason_required_when_value_absent"])
        self.assertEqual(
            envelope["missing_pusch_semantics"],
            "MISSING_UNRESOLVED_NOT_DTX_WITHOUT_DTX_EVIDENCE",
        )
        self.assertFalse(envelope["zero_fill_authorized"])
        self.assertFalse(envelope["forward_fill_authorized"])

    def test_ack_bracket_and_first_effect_never_claim_application_timestamp(self) -> None:
        timing = self.config["command_timing"]
        self.assertIn("ACK_UPPER_BOUND", timing["response_received_semantics"])
        self.assertEqual(
            timing["first_effect_semantics"],
            "MEASURED_STEP_RESPONSE_ESTIMATE_NOT_COMMAND_APPLICATION_TIMESTAMP",
        )
        self.assertIn("first_effect_lag_estimate_ns", timing["n2_first_effect_fields"])
        self.assertEqual(
            timing["prohibited_fields"],
            ["command_applied_at", "command_applied_at_ns", "application_timestamp_ns"],
        )

    def test_librfsimulator_and_other_runtime_artifacts_are_pinned(self) -> None:
        runtime = self.config["runtime_artifacts"]
        self.assertFalse(runtime["execution_claimed"])
        self.assertIn("REVERIFY_AT_UE_N2_PREFLIGHT", runtime["authority"])
        self.assertEqual(
            runtime["files"]["rfsimulator_library"],
            {
                "path": "OAI/openairinterface5g/cmake_targets/ran_build/build/librfsimulator.so",
                "sha256": "e61a78176a4c42097183bf4ff89aead1943469a7f0110afa0ec1d0171ae79bfd",
            },
        )
        input_labels = {(row["kind"], row.get("label")) for row in self.manifest["inputs"]}
        self.assertIn(("runtime_artifact", "rfsimulator_library"), input_labels)

    def test_strict_created_at_accepts_only_canonical_utc(self) -> None:
        self.assertEqual(validate_created_at(FIXED_NOW), FIXED_NOW)
        for invalid in (
            "2026-08-21T06:00:00Z",
            "2026-08-21T06:00:00.000000+00:00",
            "2026-08-21T06:00:00.000000-07:00",
            "2026-08-21 06:00:00.000000Z",
            "2026-02-30T06:00:00.000000Z",
            "2026-08-21T06:00:00.00000Z",
            None,
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(InterfaceV2Error):
                    validate_created_at(invalid)

    def test_invalid_created_at_publishes_nothing(self) -> None:
        target = self.temp / "invalid_time"
        with self.assertRaises(InterfaceV2Error):
            assemble(DEFAULT_CONFIG, target, now="2026-08-21T06:00:00+00:00")
        self.assertFalse(target.exists())

    def test_command_literal_is_lexically_safe_without_numeric_bounds(self) -> None:
        for valid in ("0", "-50", "12", "-0.5", "24.25"):
            self.assertEqual(str(validate_commanded_noise_power_literal(valid)), valid)
        for invalid in (
            True, 1.0, "", "-0", "00", "01", "1.0", "1e2", "+1",
            "NaN", "Infinity", " 1", "1 ", "1junk", "1\nshow current",
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(InterfaceV2Error):
                    validate_commanded_noise_power_literal(invalid)
        with self.assertRaisesRegex(InterfaceV2Error, "OAI binary64"):
            validate_commanded_noise_power_literal("9" * 400)
        with self.assertRaisesRegex(InterfaceV2Error, "OAI binary32 storage"):
            validate_commanded_noise_power_literal("9" * 100)
        self.assertEqual(
            self.config["attach_lifecycle"][
                "initial_and_restore_commanded_noise_power_db"
            ],
            "-50",
        )
        self.assertEqual(
            self.manifest["interface"][
                "clean_attach_and_restore_commanded_noise_power_db"
            ],
            "-50",
        )
        self.assertEqual(
            self.config["attach_lifecycle"][
                "pre_attach_show_current_required_noise_power_db"
            ],
            -50,
        )

    def test_superseded_v1_directory_is_closed_to_unexpected_entries(self) -> None:
        copied_v1 = self.temp / "v1_with_unexpected_entry"
        shutil.copytree(V1_BUNDLE, copied_v1)
        (copied_v1 / "unexpected.json").write_text("{}\n", encoding="utf-8")
        original_repo_path = n2._repo_path

        def redirect_v1(relative: str) -> Path:
            if relative == self.config["supersedes"]["bundle_dir"]:
                return copied_v1
            return original_repo_path(relative)

        with mock.patch.object(n2, "_repo_path", side_effect=redirect_v1):
            with self.assertRaisesRegex(InterfaceV2Error, "v1 bundle entry set"):
                n2._verify_superseded_v1(self.config)

    def test_direct_bler_is_unresolved_and_grant_round_is_only_proxy(self) -> None:
        telemetry = self.config["telemetry"]
        self.assertEqual(
            telemetry["direct_ul_bler_status"],
            "UNAVAILABLE_UNRESOLVED_CURRENT_SINR_TRACE",
        )
        self.assertFalse(telemetry["missing_direct_bler_is_zero"])
        self.assertEqual(telemetry["ue_grant_round_semantics"], "RETRANSMISSION_PROXY_ONLY")
        events = {event["id"]: event for event in telemetry["events"]}
        self.assertIn("NOT_DIRECT_UL_BLER", events["GNB_MAC_BLER_MCS_DECISION"]["semantics"])

    def test_bundle_is_create_only_and_deterministic(self) -> None:
        with self.assertRaisesRegex(InterfaceV2Error, "refusing to overwrite"):
            assemble(DEFAULT_CONFIG, self.output, now=FIXED_NOW)
        other = self.temp / "bundle_b"
        assemble(DEFAULT_CONFIG, other, now=FIXED_NOW)
        for name in (
            "resolved_config.json", "REPORT.md", "manifest.json",
            "UE_N1_INTERFACE_V2_FROZEN.json",
        ):
            self.assertEqual(sha256(self.output / name), sha256(other / name), name)

    def test_failed_temp_validation_does_not_publish(self) -> None:
        target = self.temp / "must_not_publish"
        with mock.patch.object(n2, "validate_bundle", side_effect=InterfaceV2Error("injected")):
            with self.assertRaisesRegex(InterfaceV2Error, "injected"):
                assemble(DEFAULT_CONFIG, target, now=FIXED_NOW)
        self.assertFalse(target.exists())

    def test_atomic_publish_does_not_replace_racing_target(self) -> None:
        target = self.temp / "racing_target"
        original = n2._rename_noreplace

        def inject_competitor(source: Path, destination: Path) -> None:
            destination.mkdir()
            original(source, destination)

        with mock.patch.object(n2, "_rename_noreplace", side_effect=inject_competitor):
            with self.assertRaisesRegex(InterfaceV2Error, "refusing to overwrite"):
                assemble(DEFAULT_CONFIG, target, now=FIXED_NOW)
        self.assertTrue(target.is_dir())
        self.assertEqual(list(target.iterdir()), [])

    def test_canonical_seal_rejects_alternate_self_consistent_retarget(self) -> None:
        case = self.temp / "retarget"
        case.mkdir()
        path = mutate_config(
            case,
            lambda config: config["runtime_artifacts"]["files"]["rfsimulator_library"].__setitem__(
                "sha256", "0" * 64
            ),
        )
        with self.assertRaisesRegex(InterfaceV2Error, "canonical seal"):
            load_config(path)

    def test_semantic_mutations_fail_even_with_mutated_config_seal(self) -> None:
        cases = (
            (lambda c: c.__setitem__("extra", {}), "top-level"),
            (lambda c: c["actuator"].__setitem__("canonical_command_field", "channel_command_db"), "actuator"),
            (lambda c: c["causal_classification"].__setitem__("control_fields_in_policy_state_authorized", True), "causal"),
            (lambda c: c["policy_availability"].__setitem__("collector_ingest_is_policy_availability", True), "policy availability"),
            (lambda c: c["raw_event_envelope"].__setitem__("missing_pusch_semantics", "DTX"), "raw event"),
            (lambda c: c["runtime_artifacts"]["files"].pop("rfsimulator_library"), "runtime artifact"),
            (lambda c: c["output"].__setitem__("report_md", "../escape.md"), "output"),
        )
        for index, (mutate, message) in enumerate(cases):
            case = self.temp / f"semantic_{index}"
            case.mkdir()
            path = mutate_config(case, mutate)
            with mock.patch.object(n2, "FROZEN_CONFIG_SHA256", sha256(path)):
                with self.subTest(message=message):
                    with self.assertRaisesRegex(InterfaceV2Error, message):
                        load_config(path)
            self.assertFalse((self.temp / "escape.md").exists())

    def test_tampered_created_at_or_report_is_rejected_even_if_resealed(self) -> None:
        for name, mutate, message in (
            (
                "time",
                lambda manifest, _: manifest.__setitem__(
                    "created_at", "2026-08-21T06:00:00.000000-00:00"
                ),
                "created_at",
            ),
            (
                "report",
                lambda manifest, bundle: _reseal_changed_report(manifest, bundle),
                "deterministic report",
            ),
        ):
            bundle = self.temp / f"tampered_{name}"
            shutil.copytree(self.output, bundle)
            manifest_path = bundle / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            mutate(manifest, bundle)
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            terminal_path = bundle / "UE_N1_INTERFACE_V2_FROZEN.json"
            terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
            terminal["manifest_sha256"] = sha256(manifest_path)
            terminal_path.write_text(json.dumps(terminal, indent=2, sort_keys=True) + "\n")
            with self.subTest(name=name):
                with self.assertRaisesRegex(InterfaceV2Error, message):
                    validate_bundle(bundle)

    def test_direct_cli_is_offline(self) -> None:
        output = self.temp / "direct_cli"
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "rl_agent/ue_n1_freeze_oai_ul_actuator_v2.py"),
                "--output-dir",
                str(output),
            ],
            cwd=self.temp,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        terminal = json.loads((output / "UE_N1_INTERFACE_V2_FROZEN.json").read_text())
        self.assertFalse(terminal["runtime_executed"])
        self.assertFalse(terminal["socket_executed"])


def _reseal_changed_report(manifest: dict, bundle: Path) -> None:
    report = bundle / "REPORT.md"
    report.write_text(report.read_text(encoding="utf-8") + "false claim\n", encoding="utf-8")
    record = next(row for row in manifest["outputs"] if row["path"] == "REPORT.md")
    record.update({"sha256": sha256(report), "bytes": report.stat().st_size})


if __name__ == "__main__":
    unittest.main()
