from __future__ import annotations

import copy
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import rl_agent.ue_n3_oai_ul_live_stage as n3live


def receiver_summary(*, complete_frames: int = 600, malformed: int = 0,
                     gaps: int = 0, streams: int = 1,
                     stop_reason: str = "DURATION_COMPLETE",
                     stream_limit_exceeded: int = 0,
                     stream_id: str = "192.168.70.134:44000") -> dict:
    rows = []
    if streams == 1:
        rows = [{
            "stream_id": stream_id,
            "expected_frames": 600,
            "complete_frames": complete_frames,
            "contract_mismatch_datagrams": 0,
            "outside_expected_range_datagrams": 0,
            "interarrival_gaps_over_one_second": gaps,
            "unique_datagram_goodput_mbps": 0.92,
            "unique_payload_goodput_mbps": 0.918,
        }]
    return {
        "status": "CAPTURED",
        "clean_shutdown": True,
        "stop_reason": stop_reason,
        "valid_stream_count": streams,
        "stream_limit_exceeded_datagrams": stream_limit_exceeded,
        "malformed_datagrams": malformed,
        "streams": rows,
    }


class FakeProcess:
    def __init__(self) -> None:
        self.returncode = None

    def poll(self):
        return self.returncode


class LiveStageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(n3live.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        self.temp = tempfile.TemporaryDirectory(prefix="ue-n3-live-stage-")
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_config_freezes_control_and_future_search_but_no_promotion(self) -> None:
        n3live.validate_config(self.config, require_live_authority=False)
        clean = self.config["modes"][n3live.MODE]
        self.assertEqual(clean["sender_frames"], 600)
        self.assertEqual(clean["service_duration_s"], 60.0)
        self.assertEqual(
            self.config["modes"]["COMMAND_CALIBRATION_SEARCH"]["commanded_noise_power_db"],
            [-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0],
        )
        self.assertFalse(self.config["authority"]["target_mapping_promotion_authorized"])
        self.assertFalse(self.config["authority"]["numeric_bound_promotion_authorized"])

    def test_live_authority_is_explicit(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["authority"]["oai_run_authorized"] = False
        changed["authority"]["socket_execution_authorized"] = False
        with self.assertRaisesRegex(n3live.LiveStageFailure, "live OAI authority"):
            n3live.validate_config(changed, require_live_authority=True)

    def test_primary_and_sensitivity_gates_are_distinct(self) -> None:
        primary = n3live.classify_receiver_gate(
            receiver_summary=receiver_summary(complete_frames=594),
            sender_frames=600,
            gates=self.config["gates"],
        )
        self.assertTrue(primary["primary_usable_service_pass"])
        self.assertTrue(primary["delivery_95_pass"])
        self.assertTrue(primary["delivery_90_pass"])

        sensitivity = n3live.classify_receiver_gate(
            receiver_summary=receiver_summary(complete_frames=570),
            sender_frames=600,
            gates=self.config["gates"],
        )
        self.assertFalse(sensitivity["primary_usable_service_pass"])
        self.assertTrue(sensitivity["delivery_95_pass"])
        self.assertTrue(sensitivity["delivery_90_pass"])

        low = n3live.classify_receiver_gate(
            receiver_summary=receiver_summary(complete_frames=539),
            sender_frames=600,
            gates=self.config["gates"],
        )
        self.assertFalse(low["delivery_90_pass"])

    def test_structural_error_dominates_delivery(self) -> None:
        for summary in (
            receiver_summary(malformed=1),
            receiver_summary(gaps=1),
            receiver_summary(streams=0),
            receiver_summary(stop_reason="SIGNAL_SIGTERM"),
            receiver_summary(stream_limit_exceeded=1),
            receiver_summary(stream_id="192.168.70.99:44000"),
        ):
            with self.subTest(summary=summary):
                result = n3live.classify_receiver_gate(
                    receiver_summary=summary,
                    sender_frames=600,
                    gates=self.config["gates"],
                )
                self.assertFalse(result["structural_pass"])
                self.assertFalse(result["primary_usable_service_pass"])

    def test_receiver_ready_precedes_sender_spawn(self) -> None:
        runner = n3live.CleanControlRunner(n3live.DEFAULT_CONFIG, self.root / "run")
        runner.ue_ip = "10.0.0.2"
        order: list[str] = []

        def spawn(name, _argv, _log_name, **_kwargs):
            order.append(name)
            managed = SimpleNamespace(name=name, process=FakeProcess())
            if name == "structured_receiver":
                ready = runner.output_dir / "traffic/receiver_ready.json"
                ready.parent.mkdir(parents=True, exist_ok=True)
                ready.write_text(json.dumps({"status": "READY", "port": 56130}),
                                 encoding="utf-8")
            return managed

        inspected = SimpleNamespace(stdout="1234\n")
        with mock.patch.object(n3live.n2, "run_checked", return_value=inspected), \
             mock.patch.object(runner, "spawn", side_effect=spawn):
            runner.start_traffic()
        self.assertEqual(order, ["structured_receiver", "structured_sender"])
        self.assertTrue(runner.receiver_ready_observed)

    def test_create_only_output_is_preserved(self) -> None:
        output = self.root / "existing"
        output.mkdir()
        with self.assertRaisesRegex(n3live.n2.SmokeFailure, "create-only"):
            n3live.CleanControlRunner(n3live.DEFAULT_CONFIG, output)

    def test_cleanup_rejects_leaked_ext_dn_udp_listener(self) -> None:
        runner = n3live.CleanControlRunner(n3live.DEFAULT_CONFIG, self.root / "cleanup")
        inspect = SimpleNamespace(stdout="1234\n")
        busy = SimpleNamespace(stdout="UNCONN 0 0 192.168.70.135:56130 0.0.0.0:*\n")
        with mock.patch.object(n3live.n2.Runner, "cleanup", return_value=[]), \
             mock.patch.object(n3live.n2, "run_checked", side_effect=[inspect, busy]):
            errors = runner.cleanup(strict=False)
        self.assertTrue(any("UDP port survived" in error for error in errors))
        report = json.loads((runner.output_dir / "cleanup_report.json").read_text())
        self.assertTrue(report["ext_dn_structured_udp_port_busy"])
        self.assertFalse(report["clean"])

    def test_command_search_mode_is_not_executable_in_this_runner(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                n3live.parse_args([
                    "--output-dir", str(self.root / "unused"),
                    "--mode", "COMMAND_CALIBRATION_SEARCH",
                ])

    def test_runtime_seals_match_current_files(self) -> None:
        for entry in self.config["runtime_seals"]:
            path = n3live.resolve(entry["path"])
            self.assertTrue(path.is_file(), entry["path"])
            self.assertEqual(n3live.sha256(path), entry["sha256"], entry["path"])


if __name__ == "__main__":
    unittest.main()
