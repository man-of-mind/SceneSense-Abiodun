from __future__ import annotations

import contextlib
import copy
import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from rl_agent import ue_n3_oai_ul_command_calibration_v1 as n3


class FakeTelnet:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def command(self, command: str) -> tuple[int, int, int, int, str]:
        self.commands.append(command)
        state = (
            "model 0 rfsimu_channel_ue0 type AWGN:\n"
            "path loss: 0 noise: -4.0\n"
            "model owner: rfsimulator\nsoftmodem_gnb> "
        )
        return (
            100,
            200,
            300,
            400,
            state,
        )


def pusch_row(monotonic_ns: int, rnti: int = 0x1234, snr_x10: int = 40):
    return (1, monotonic_ns, f"12:00:00.000000,{rnti},1,2,{snr_x10}")


def mcs_row(monotonic_ns: int, rnti: int = 0x1234):
    fields = [
        "12:00:00.000000", str(rnti), "1", "2", "1", "3", "40", "0",
        "9", "8", "8", "8", "7", "12500", "12500", "100", "10",
        "106", "96", "20", "23", "10", "12500", "-1",
    ]
    return (1, monotonic_ns, ",".join(fields))


class CommandCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = n3.load_json(n3.DEFAULT_CONFIG)
        self.temp = tempfile.TemporaryDirectory(prefix="ue-n3-command-calibration-")
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_config_is_offline_safe_and_frozen(self) -> None:
        n3.validate_config(self.config, verify_hashes=True)
        self.assertFalse(self.config["authority"]["live_oai_run_authorized"])
        self.assertFalse(self.config["authority"]["live_socket_execution_authorized"])
        self.assertFalse(self.config["authority"]["target_mapping_promotion_authorized"])
        self.assertFalse(self.config["authority"]["numeric_bound_promotion_authorized"])
        rows = n3.campaign_plan_rows(self.config)
        self.assertEqual(len(rows), 9)
        self.assertEqual([row["commanded_noise_power_db"] for row in rows],
                         [-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0])
        self.assertTrue(all(row["fresh_ran_epoch_required"] for row in rows))
        self.assertTrue(all(row["candidate_application_count"] == 1 for row in rows))
        self.assertTrue(all(row["sender_frames"] == 250 for row in rows))

    def test_live_authority_requires_the_frozen_basis(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["authority"]["live_oai_run_authorized"] = True
        changed["authority"]["live_socket_execution_authorized"] = True
        with self.assertRaisesRegex(n3.CalibrationFailure, "authority basis"):
            n3.validate_config(changed, verify_hashes=False)

    def test_rnti_change_is_not_mislabeled_as_detachment(self) -> None:
        self.assertEqual(
            n3.classify_service_loss_reason("UE_TUNNEL_IDENTITY_LOST"),
            n3.RUNG_DETACHED,
        )
        self.assertEqual(
            n3.classify_service_loss_reason("RNTI_CHANGED"),
            n3.RUNG_IDENTITY_DISCONTINUITY,
        )
        self.assertEqual(
            n3.classify_service_loss_reason("CURRENT_RNTI_PUSCH_SILENCE"),
            n3.RUNG_HARD_LOSS,
        )

    def test_prepare_only_is_create_only_and_never_promotes(self) -> None:
        output = self.root / "prepared"
        with mock.patch.object(n3.subprocess, "Popen") as popen, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(n3.CampaignRunner(n3.DEFAULT_CONFIG, output).prepare(), 0)
        popen.assert_not_called()
        terminal = n3.load_json(output / f"{n3.PLAN_FROZEN}.json")
        summary = n3.load_json(output / "campaign_summary.json")
        self.assertEqual(terminal["status"], n3.PLAN_FROZEN)
        self.assertFalse(summary["runtime_executed"])
        self.assertFalse(summary["socket_executed"])
        self.assertFalse(terminal["target_mapping_promoted"])
        self.assertFalse(terminal["numeric_bound_promoted"])
        with self.assertRaisesRegex(n3.CalibrationFailure, "create-only"):
            n3.CampaignRunner(n3.DEFAULT_CONFIG, output)

    def test_candidate_command_can_be_applied_exactly_once(self) -> None:
        runner = n3.RungRunner(
            n3.DEFAULT_CONFIG, self.root / "one-apply", rung_index=0, command_db=-4.0,
        )
        telnet = FakeTelnet()
        runner.telnet = telnet
        with mock.patch.object(runner, "assert_carla_absent"):
            row = runner.apply_candidate_once(0)
            self.assertEqual(row["status"], "ACK_AND_POST_STATE_VALIDATED_ONCE")
            self.assertEqual(runner.application_count, 1)
            self.assertEqual(telnet.commands, [
                "channelmod modify 0 noise_power_dB -4.0", "channelmod show current",
            ])
            with self.assertRaisesRegex(n3.CalibrationFailure, "only once"):
                runner.apply_candidate_once(0)
        self.assertEqual(len([value for value in telnet.commands if " modify " in value]), 1)

    def test_carla_gate_does_not_precreate_n2_runtime_directory(self) -> None:
        runner = n3.RungRunner(
            n3.DEFAULT_CONFIG, self.root / "carla-gate-layout", rung_index=0,
            command_db=-4.0,
        )
        process_result = SimpleNamespace(returncode=0, stdout="")
        with (
            mock.patch.object(n3.subprocess, "run", return_value=process_result),
            mock.patch.object(runner, "strict_port_free", return_value=True),
        ):
            runner.assert_carla_absent()
        self.assertTrue((runner.output_dir / "carla_absent_gate.json").is_file())
        self.assertFalse((runner.output_dir / "runtime").exists())

    def test_post_send_timeout_is_conservatively_an_application_attempt(self) -> None:
        runner = n3.RungRunner(
            n3.DEFAULT_CONFIG, self.root / "send-timeout", rung_index=0, command_db=-4.0,
        )

        class TimeoutTelnet:
            def command(self, _command):
                raise TimeoutError("ACK timeout after send")

        runner.telnet = TimeoutTelnet()
        with mock.patch.object(runner, "assert_carla_absent"):
            with self.assertRaisesRegex(TimeoutError, "ACK timeout"):
                runner.apply_candidate_once(0)
        self.assertTrue(runner.nonclean_applied)
        self.assertEqual(runner.application_count, 1)
        self.assertEqual(
            runner.command_rows[0]["status"],
            "SEND_ATTEMPT_STARTED_ACK_UNCONFIRMED",
        )

    def test_receiver_silence_does_not_erase_live_pusch_mapping(self) -> None:
        runner = n3.RungRunner(
            n3.DEFAULT_CONFIG, self.root / "receiver-outage", rung_index=0,
            command_db=-4.0,
        )
        alive = SimpleNamespace(poll=lambda: None)
        alive_thread = SimpleNamespace(is_alive=lambda: True)
        runner.processes = [
            SimpleNamespace(name="gnb", process=alive),
            SimpleNamespace(name="ue", process=alive),
        ]
        runner.receiver = SimpleNamespace(process=alive)
        runner.sender = SimpleNamespace(process=alive)
        runner.live_csv = SimpleNamespace(process=alive, thread=alive_thread)
        runner.live_mcs = SimpleNamespace(process=alive, thread=alive_thread)
        runner.ue_ip = "10.0.0.2"
        runner.current_rnti = 0x1234
        runner.traffic_start_ns = time.monotonic_ns() - 3_000_000_000
        runner.last_carla_check_monotonic_ns = time.monotonic_ns()
        runner.event_tail = SimpleNamespace(
            poll=lambda: time.monotonic_ns() - 3_000_000_000,
        )
        with mock.patch.object(runner, "tunnel_ip", return_value=runner.ue_ip), \
             mock.patch.object(runner, "observed_rntis", return_value={runner.current_rnti}), \
             mock.patch.object(runner, "latest_pusch_ns", return_value=time.monotonic_ns()):
            runner.check_health(enforce_silence=True)
        self.assertTrue(runner.receiver_service_outage_detected)

    def test_collector_exit_is_an_instrumentation_failure(self) -> None:
        runner = n3.RungRunner(
            n3.DEFAULT_CONFIG, self.root / "collector-exit", rung_index=0,
            command_db=-4.0,
        )
        alive = SimpleNamespace(poll=lambda: None)
        exited = SimpleNamespace(poll=lambda: 1)
        alive_thread = SimpleNamespace(is_alive=lambda: True)
        runner.processes = [
            SimpleNamespace(name="gnb", process=alive),
            SimpleNamespace(name="ue", process=alive),
        ]
        runner.receiver = SimpleNamespace(process=alive)
        runner.sender = SimpleNamespace(process=alive)
        runner.live_csv = SimpleNamespace(process=exited, thread=alive_thread)
        runner.live_mcs = SimpleNamespace(process=alive, thread=alive_thread)
        runner.last_carla_check_monotonic_ns = time.monotonic_ns()
        with self.assertRaisesRegex(n3.CalibrationFailure, "PUSCH collector"):
            runner.check_health(enforce_silence=False)

        runner.live_csv = SimpleNamespace(process=alive, thread=alive_thread)
        runner.live_mcs = SimpleNamespace(
            process=alive,
            thread=SimpleNamespace(is_alive=lambda: False),
        )
        with self.assertRaisesRegex(n3.CalibrationFailure, "MCS collector drain"):
            runner.check_health(enforce_silence=False)

    def test_fresh_receiver_with_stale_pusch_is_instrumentation_failure(self) -> None:
        runner = n3.RungRunner(
            n3.DEFAULT_CONFIG, self.root / "contradictory-freshness", rung_index=0,
            command_db=-4.0,
        )
        alive = SimpleNamespace(poll=lambda: None)
        alive_thread = SimpleNamespace(is_alive=lambda: True)
        runner.processes = [
            SimpleNamespace(name="gnb", process=alive),
            SimpleNamespace(name="ue", process=alive),
        ]
        runner.receiver = SimpleNamespace(process=alive)
        runner.sender = SimpleNamespace(process=alive)
        runner.live_csv = SimpleNamespace(process=alive, thread=alive_thread)
        runner.live_mcs = SimpleNamespace(process=alive, thread=alive_thread)
        runner.ue_ip = "10.0.0.2"
        runner.current_rnti = 0x1234
        runner.traffic_start_ns = time.monotonic_ns() - 3_000_000_000
        runner.last_carla_check_monotonic_ns = time.monotonic_ns()
        runner.event_tail = SimpleNamespace(poll=time.monotonic_ns)
        with (
            mock.patch.object(runner, "tunnel_ip", return_value=runner.ue_ip),
            mock.patch.object(runner, "observed_rntis", return_value={runner.current_rnti}),
            mock.patch.object(
                runner, "latest_pusch_ns",
                return_value=time.monotonic_ns() - 3_000_000_000,
            ),
        ):
            with self.assertRaisesRegex(
                n3.CalibrationFailure, "telemetry stale"
            ):
                runner.check_health(enforce_silence=True)

    def test_recovery_allows_one_stable_replacement_rnti_and_requires_delivery(self) -> None:
        runner = n3.RungRunner(
            n3.DEFAULT_CONFIG, self.root / "replacement-rnti", rung_index=0,
            command_db=-4.0,
        )
        runner.config["rung"]["clean_recovery_s"] = 0.02
        runner.config["rung"]["minimum_recovery_pusch_samples"] = 2
        runner.config["rung"]["minimum_recovery_receiver_frames"] = 1
        alive = SimpleNamespace(poll=lambda: None)
        alive_thread = SimpleNamespace(is_alive=lambda: True)

        class FakeLive:
            process = alive
            thread = alive_thread

            @staticmethod
            def snapshot():
                now = time.monotonic_ns()
                return [pusch_row(now - 2, 0x5678), pusch_row(now - 1, 0x5678)]

        runner.processes = [
            SimpleNamespace(name="gnb", process=alive),
            SimpleNamespace(name="ue", process=alive),
        ]
        runner.receiver = SimpleNamespace(process=alive)
        runner.live_csv = FakeLive()
        runner.live_mcs = SimpleNamespace(process=alive, thread=alive_thread)
        events_path = runner.output_dir / "receiver_events.jsonl"
        events_path.write_text("\n".join(
            json.dumps({
                "event_type": "datagram",
                "status": "ACCEPTED_UNIQUE",
                "source_ip": "192.168.70.134",
                "receiver_monotonic_ns": time.monotonic_ns(),
            })
            for _ in range(10)
        ) + "\n", encoding="utf-8")
        runner.event_tail = n3.EventTail(
            events_path, expected_source_ip="192.168.70.134"
        )
        runner.event_tail.poll()
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "event_type": "datagram",
                "status": "ACCEPTED_UNIQUE",
                "source_ip": "192.168.70.134",
                "receiver_monotonic_ns": time.monotonic_ns(),
            }) + "\n")
        runner.current_rnti = 0x1234
        runner.ue_ip = "10.0.0.2"
        runner.last_carla_check_monotonic_ns = time.monotonic_ns()
        with mock.patch.object(runner, "tunnel_ip", return_value=runner.ue_ip):
            result = runner.verify_recovery(0, 10, required=False)
        self.assertTrue(result["passed"])
        self.assertTrue(result["application_delivery_passed"])
        self.assertTrue(result["rnti_replaced_after_restore"])
        self.assertEqual(result["recovered_rnti"], 0x5678)
        self.assertEqual(runner.current_rnti, 0x5678)

        receiver_baseline = runner.event_tail.accepted_count
        with events_path.open("a", encoding="utf-8") as handle:
            for _ in range(5):
                handle.write(json.dumps({
                    "event_type": "datagram",
                    "status": "ACCEPTED_UNIQUE",
                    "source_ip": "203.0.113.99",
                    "receiver_monotonic_ns": time.monotonic_ns(),
                }) + "\n")
        with mock.patch.object(runner, "tunnel_ip", return_value=runner.ue_ip):
            missing_delivery = runner.verify_recovery(
                0, receiver_baseline, required=False
            )
        self.assertFalse(missing_delivery["passed"])
        self.assertTrue(missing_delivery["radio_recovery_passed"])
        self.assertFalse(missing_delivery["application_delivery_passed"])

        runner.live_mcs = SimpleNamespace(
            process=alive,
            thread=SimpleNamespace(is_alive=lambda: False),
        )
        with mock.patch.object(runner, "tunnel_ip", return_value=runner.ue_ip):
            with self.assertRaisesRegex(
                n3.CalibrationFailure, "MCS collector drain.*recovery"
            ):
                runner.verify_recovery(
                    0, runner.event_tail.accepted_count, required=False
                )

    def test_partial_sender_is_retained_only_for_service_loss_evidence(self) -> None:
        runner = n3.RungRunner(
            n3.DEFAULT_CONFIG, self.root / "partial-sender", rung_index=0,
            command_db=-4.0,
        )

        class DoneProcess:
            def __init__(self, returncode):
                self.returncode = returncode

            def poll(self):
                return self.returncode

        class DelayedReceiverProcess:
            def __init__(self):
                self.returncode = None
                self.wait_timeout = None

            def poll(self):
                return self.returncode

            def wait(self, timeout):
                self.wait_timeout = timeout
                self.returncode = 0

        runner.sender = SimpleNamespace(process=DoneProcess(1))
        delayed_receiver = DelayedReceiverProcess()
        runner.receiver = SimpleNamespace(process=delayed_receiver)
        runner.traffic_start_ns = time.monotonic_ns() - 12_000_000_000
        traffic = runner.output_dir / "traffic"
        traffic.mkdir(parents=True)
        n3.write_csv(traffic / "sender.csv", [
            {"frame_index": index, "chunk_index": 0}
            for index in range(10)
        ])
        n3.n2.atomic_json(traffic / "receiver_summary.json", {
            "schema": "scenesense.ue_n3_structured_udp_receiver_summary.v1",
            "status": "CAPTURED",
            "clean_shutdown": True,
            "stop_reason": "DURATION_COMPLETE",
            "valid_stream_count": 1,
            "malformed_datagrams": 0,
            "stream_limit_exceeded_datagrams": 0,
            "streams": [{
                "stream_id": "192.168.70.134:12345",
                "complete_frames": 10,
                "contract_mismatch_datagrams": 0,
                "outside_expected_range_datagrams": 0,
                "interarrival_gaps_over_one_second": 0,
                "max_interarrival_gap_s": 0.1,
            }],
        })
        retained = runner.finish_probe(allow_partial_sender=True)
        self.assertFalse(retained["integrity_gate"])
        self.assertTrue(
            retained["sender_completion"][
                "partial_sender_allowed_after_service_loss"
            ]
        )
        self.assertGreater(delayed_receiver.wait_timeout, 10.0)
        with self.assertRaisesRegex(n3.CalibrationFailure, "sender exited"):
            runner.finish_probe(allow_partial_sender=False)

    def test_repeated_cleanup_cannot_erase_an_earlier_dirty_result(self) -> None:
        runner = n3.RungRunner(
            n3.DEFAULT_CONFIG, self.root / "sticky-cleanup", rung_index=0,
            command_db=-4.0,
        )
        report_path = runner.output_dir / "cleanup_report.json"
        n3.n2.atomic_json(report_path, {
            "clean": False,
            "errors": ["first cleanup failure"],
        })

        def superficially_clean(base_runner, *, strict=False):
            n3.n2.atomic_json(base_runner.output_dir / "cleanup_report.json", {
                "clean": True,
                "errors": [],
            })
            return []

        with mock.patch.object(
            n3.n2.Runner, "cleanup", autospec=True,
            side_effect=superficially_clean,
        ):
            first = runner.cleanup(strict=False)
            second = runner.cleanup(strict=False)
        report = n3.load_json(report_path)
        self.assertFalse(report["clean"])
        self.assertIn("first cleanup failure", report["errors"])
        self.assertIn("first cleanup failure", first)
        self.assertIn("first cleanup failure", second)
        self.assertTrue(report["prior_dirty_state_preserved"])

    def test_transport_integrity_uses_frozen_upf_source_and_complete_stop(self) -> None:
        receiver = {
            "schema": "scenesense.ue_n3_structured_udp_receiver_summary.v1",
            "status": "CAPTURED", "clean_shutdown": True,
            "stop_reason": "DURATION_COMPLETE", "valid_stream_count": 1,
            "malformed_datagrams": 0, "stream_limit_exceeded_datagrams": 0,
            "streams": [{
                "stream_id": "192.168.70.134:40000", "complete_frames": 248,
                "contract_mismatch_datagrams": 0,
                "outside_expected_range_datagrams": 0,
                "interarrival_gaps_over_one_second": 0,
                "max_interarrival_gap_s": 0.11,
            }],
        }
        sender = {"complete": True}
        accepted = n3.evaluate_transport(
            receiver, sender, expected_frames=250, gates=self.config["transport_gates"],
        )
        self.assertTrue(accepted["integrity_gate"])
        self.assertTrue(accepted["primary_99_pass"])
        self.assertEqual(accepted["expected_source_ip"], "192.168.70.134")
        for changed in (
            {"stop_reason": "SIGNAL_SIGTERM"},
            {"stream_limit_exceeded_datagrams": 1},
        ):
            bad = copy.deepcopy(receiver)
            bad.update(changed)
            self.assertFalse(n3.evaluate_transport(
                bad, sender, expected_frames=250, gates=self.config["transport_gates"],
            )["integrity_gate"])
        wrong_source = copy.deepcopy(receiver)
        wrong_source["streams"][0]["stream_id"] = "10.0.0.2:40000"
        self.assertFalse(n3.evaluate_transport(
            wrong_source, sender, expected_frames=250,
            gates=self.config["transport_gates"],
        )["integrity_gate"])

    def test_tail_requires_current_rnti_pusch_and_scheduler_mcs(self) -> None:
        pusch = [pusch_row(1_000 + index) for index in range(120)]
        mcs = [mcs_row(1_000 + index) for index in range(30)]
        accepted = n3.summarize_tail(
            [*pusch, pusch_row(2_000, snr_x10=999)],
            [*mcs, mcs_row(2_000)],
            start_ns=1_000, end_ns=2_000, expected_rnti=0x1234,
            minimum_pusch=120, minimum_mcs=30, required_mcs_table=0,
            required_force_mcs=-1,
        )
        self.assertEqual(accepted["status"], "TAIL_ACCEPTED")
        self.assertEqual(accepted["pusch_samples"], 120)
        self.assertEqual(accepted["mcs_samples"], 30)
        self.assertEqual(accepted["achieved_pusch_snr_db_median"], 4.0)
        self.assertEqual(accepted["achieved_pusch_snr_db_p05"], 4.0)
        self.assertEqual(accepted["achieved_pusch_snr_db_p95"], 4.0)
        self.assertEqual(accepted["selected_mcs_median"], 8)
        self.assertEqual(accepted["final_mcs_median"], 7)

        mixed = n3.summarize_tail(
            [*pusch, pusch_row(1_500, rnti=0x9999)], mcs,
            start_ns=1_000, end_ns=2_000, expected_rnti=0x1234,
            minimum_pusch=120, minimum_mcs=30, required_mcs_table=0,
            required_force_mcs=-1,
        )
        self.assertEqual(mixed["status"], "TAIL_UNCONFIRMED")

    def test_mapping_integrity_is_orthogonal_to_service_outage(self) -> None:
        sender = {"complete": True}

        def summary(gaps: int) -> dict:
            return {
                "schema": "scenesense.ue_n3_structured_udp_receiver_summary.v1",
                "status": "CAPTURED",
                "clean_shutdown": True,
                "stop_reason": "DURATION_COMPLETE",
                "valid_stream_count": 1,
                "malformed_datagrams": 0,
                "stream_limit_exceeded_datagrams": 0,
                "streams": [{
                    "stream_id": "192.168.70.134:44000",
                    "complete_frames": 200,
                    "contract_mismatch_datagrams": 0,
                    "outside_expected_range_datagrams": 0,
                    "interarrival_gaps_over_one_second": gaps,
                    "max_interarrival_gap_s": 1.2 if gaps else 0.1,
                }],
            }

        poor_delivery = n3.evaluate_transport(
            summary(0), sender, expected_frames=250,
            gates=self.config["transport_gates"],
        )
        self.assertTrue(poor_delivery["integrity_gate"])
        self.assertTrue(poor_delivery["no_one_second_outage_pass"])
        self.assertFalse(poor_delivery["primary_99_pass"])

        outage = n3.evaluate_transport(
            summary(1), sender, expected_frames=250,
            gates=self.config["transport_gates"],
        )
        self.assertTrue(outage["integrity_gate"])
        self.assertFalse(outage["no_one_second_outage_pass"])
        self.assertFalse(outage["sensitivity_90_pass"])

    def test_command_tail_service_is_not_diluted_by_clean_frames(self) -> None:
        sender_path = self.root / "sender.csv"
        sender_rows = [
            {
                "wall_time_s": f"{1000.0 + index * 0.1:.6f}",
                "elapsed_s": f"{index * 0.1:.6f}",
                "frame_index": index,
                "chunk_index": 0,
                "scheduled_frame_time_s": f"{index * 0.1:.6f}",
            }
            for index in range(250)
        ]
        n3.write_csv(sender_path, sender_rows)
        events_path = self.root / "receiver.jsonl"
        accepted = []
        for index in range(150, 200):
            if index in {160, 170}:
                continue
            accepted.append(json.dumps({
                "event_type": "datagram",
                "status": "ACCEPTED_UNIQUE",
                "source_ip": "192.168.70.134",
                "frame_index": index,
                "receiver_wall_time_ns": int((1000.01 + index * 0.1) * 1e9),
            }))
        # A scheduler overrun exposes frame 200 before the health loop returns,
        # but the frozen five-second statistical unit remains frames 150..199.
        accepted.append(json.dumps({
            "event_type": "datagram",
            "status": "ACCEPTED_UNIQUE",
            "source_ip": "192.168.70.134",
            "frame_index": 200,
            "receiver_wall_time_ns": int(1020.05 * 1e9),
        }))
        # A missing expected frame arriving after the observed window must not
        # be credited to command-conditioned service.
        accepted.append(json.dumps({
            "event_type": "datagram",
            "status": "ACCEPTED_UNIQUE",
            "source_ip": "192.168.70.134",
            "frame_index": 160,
            "receiver_wall_time_ns": int(1020.000001 * 1e9),
        }))
        events_path.write_text("\n".join(accepted) + "\n", encoding="utf-8")
        result = n3.evaluate_tail_service(
            sender_path, events_path,
            start_wall_ns=int(1015.0 * 1e9),
            end_wall_ns=int(1020.2 * 1e9),
            fps=10.0,
            expected_tail_frames=50,
            expected_source_ip="192.168.70.134",
            structural_integrity=True,
            gates=self.config["transport_gates"],
        )
        self.assertEqual(result["expected_frames"], 50)
        self.assertEqual(result["received_frames"], 48)
        self.assertAlmostEqual(result["complete_frame_ratio"], 0.96)
        self.assertFalse(result["primary_99_pass"])
        self.assertTrue(result["sensitivity_95_pass"])
        self.assertTrue(result["exact_frozen_frame_set_pass"])
        self.assertEqual(result["observed_window_overrun_ns"], 200_000_000)

    def test_tail_frame_set_is_stable_under_boundary_jitter(self) -> None:
        sender_path = self.root / "jitter_sender.csv"
        n3.write_csv(sender_path, [
            {
                "wall_time_s": f"{1000.0 + index * 0.1:.6f}",
                "elapsed_s": f"{index * 0.1:.6f}",
                "frame_index": index,
                "chunk_index": 0,
                "scheduled_frame_time_s": f"{index * 0.1:.6f}",
            }
            for index in range(250)
        ])
        start_ns = int(1015.05 * 1e9)
        events_path = self.root / "jitter_receiver.jsonl"
        events_path.write_text("\n".join(
            json.dumps({
                "event_type": "datagram",
                "status": "ACCEPTED_UNIQUE",
                "source_ip": "192.168.70.134",
                "frame_index": index,
                "receiver_wall_time_ns": int((1000.01 + index * 0.1) * 1e9),
            })
            for index in range(151, 201)
        ) + "\n", encoding="utf-8")
        result = n3.evaluate_tail_service(
            sender_path, events_path,
            start_wall_ns=start_ns,
            end_wall_ns=start_ns + 5_100_000_000,
            fps=10.0,
            expected_tail_frames=50,
            expected_source_ip="192.168.70.134",
            structural_integrity=True,
            gates=self.config["transport_gates"],
        )
        self.assertEqual(result["expected_frame_indices"], list(range(151, 201)))
        self.assertEqual(result["expected_frames"], 50)
        self.assertEqual(result["received_frames"], 50)
        self.assertTrue(result["primary_99_pass"])

    def test_mapping_proposals_prefer_measured_then_bracket_with_source_seals(self) -> None:
        config_seal = "a" * 64
        runner_seal = "b" * 64

        def row(index, command, achieved):
            return {
                "status": n3.RUNG_CAPTURED,
                "rung_index": index,
                "commanded_noise_power_db": command,
                "achieved_pusch_snr_db_median": achieved,
                "tail_service": {
                    "complete_frame_ratio": 0.96,
                    "primary_99_pass": False,
                    "sensitivity_95_pass": True,
                    "sensitivity_90_pass": True,
                    "exact_frozen_frame_set_pass": True,
                    "required_expected_frames": 50,
                },
                "tail": {
                    "status": "TAIL_ACCEPTED",
                    "start_monotonic_ns": 100,
                    "end_monotonic_ns": 5_000_000_100,
                    "pusch_samples": 120,
                    "mcs_samples": 30,
                    "achieved_pusch_snr_db_p05": achieved - 0.5,
                    "achieved_pusch_snr_db_p95": achieved + 0.5,
                },
                "clean_recovery": {
                    "status": "CLEAN_RECOVERY_PASSED",
                    "passed": True,
                },
                "rung_evidence": {
                    "manifest_sha256": "c" * 64,
                    "terminal_sha256": "d" * 64,
                    "rung_summary_sha256": "e" * 64,
                    "ran_epoch_id": f"ran-{index}",
                    "control_session_id": f"control-{index}",
                    "config_sha256": config_seal,
                    "runner_sha256": runner_seal,
                },
            }

        result = n3.propose_mappings(
            [row(0, -4.0, 6.0), row(1, -3.0, 4.5), row(2, -2.0, 3.5)],
            [6.0, 4.0, 2.0],
            0.1,
            config_sha256=config_seal,
            runner_sha256=runner_seal,
            monotonicity={"status": "MONOTONE_WITHIN_TOLERANCE"},
        )
        exact, bracket, absent = result["proposals"]
        self.assertEqual(
            exact["status"],
            "DIRECT_MEASURED_WITHIN_TOLERANCE_REPLICATION_REQUIRED",
        )
        self.assertEqual(exact["candidate_commanded_noise_power_db"], -4.0)
        self.assertIsNone(exact["proposed_commanded_noise_power_db"])
        self.assertTrue(exact["requires_independent_replication"])
        self.assertEqual(
            exact["sources"][0]["rung_manifest_sha256"], "c" * 64
        )
        self.assertEqual(
            bracket["status"],
            "ADJACENT_MEASURED_BRACKET_REPLICATION_REQUIRED",
        )
        self.assertEqual(
            bracket["commanded_noise_power_db_interval"], [-3.0, -2.0]
        )
        self.assertNotIn("interpolated_command_candidate_db", bracket)
        self.assertTrue(bracket["requires_independent_replication"])
        self.assertIsNone(bracket["proposed_commanded_noise_power_db"])
        self.assertEqual(absent["status"], "UNBRACKETED_NO_MEASURED_CANDIDATE")
        self.assertFalse(result["target_mapping_promoted"])
        self.assertFalse(result["numeric_bound_promoted"])

        ambiguous = n3.propose_mappings(
            [row(0, -4.0, 6.0), row(1, -3.5, 6.05)],
            [6.0], 0.1,
            config_sha256=config_seal,
            runner_sha256=runner_seal,
            monotonicity={"status": "MONOTONE_WITHIN_TOLERANCE"},
        )["proposals"][0]
        self.assertEqual(
            ambiguous["status"],
            "AMBIGUOUS_MULTIPLE_DIRECT_MATCHES_REVIEW_REQUIRED",
        )

        gap = n3.propose_mappings(
            [row(0, -4.0, 6.0), row(2, -2.0, 4.0)],
            [5.0], 0.1,
            config_sha256=config_seal,
            runner_sha256=runner_seal,
            monotonicity={"status": "MONOTONE_WITHIN_TOLERANCE"},
        )["proposals"][0]
        self.assertEqual(gap["status"], "UNBRACKETED_NO_MEASURED_CANDIDATE")

        malformed = row(0, -4.0, 6.0)
        malformed["rung_evidence"]["manifest_sha256"] = "not-a-seal"
        with self.assertRaisesRegex(n3.CalibrationFailure, "malformed"):
            n3.propose_mappings(
                [malformed], [6.0], 0.1,
                config_sha256=config_seal,
                runner_sha256=runner_seal,
                monotonicity={"status": "MONOTONE_WITHIN_TOLERANCE"},
            )

    def test_rung_verifier_binds_plan_identity_and_clean_cleanup(self) -> None:
        rung = self.root / "sealed-rung"
        rung.mkdir()
        status = n3.RUNG_CAPTURED
        shared = {
            "status": status,
            "rung_index": 2,
            "commanded_noise_power_db": -3.0,
            "candidate_application_count": 1,
            "ran_epoch_id": "ran-epoch-unique",
            "control_session_id": "control-session-unique",
            "target_mapping_promoted": False,
            "numeric_bound_promoted": False,
        }
        summary_path = rung / "rung_summary.json"
        cleanup_path = rung / "cleanup_report.json"
        manifest_path = rung / "manifest.json"

        def reseal(payload, cleanup):
            n3.n2.atomic_json(summary_path, payload)
            n3.n2.atomic_json(cleanup_path, cleanup)
            n3.n2.atomic_json(manifest_path, {
                **payload,
                "config_sha256": "config-seal",
                "runner_sha256": "runner-seal",
                "outputs": [
                    {
                        "path": path.name,
                        "bytes": path.stat().st_size,
                        "sha256": n3.n2.sha256(path),
                    }
                    for path in (summary_path, cleanup_path)
                ],
            })
            n3.n2.atomic_json(rung / f"{status}.json", {
                **payload,
                "clean_restore_verified": True,
                "manifest_sha256": n3.n2.sha256(manifest_path),
            })

        reseal(shared, {"clean": True, "errors": []})
        proof = n3.verify_rung_evidence(
            rung,
            expected_status=status,
            expected_rung_index=2,
            expected_command_db=-3.0,
            expected_candidate_applications=1,
            expected_config_sha256="config-seal",
            expected_runner_sha256="runner-seal",
            require_clean_restore=True,
        )
        self.assertEqual(proof["ran_epoch_id"], "ran-epoch-unique")
        with self.assertRaisesRegex(n3.CalibrationFailure, "campaign plan"):
            n3.verify_rung_evidence(
                rung,
                expected_status=status,
                expected_rung_index=2,
                expected_command_db=-2.5,
                expected_candidate_applications=1,
                expected_config_sha256="config-seal",
                expected_runner_sha256="runner-seal",
                require_clean_restore=True,
            )

        same_identity = {
            **shared,
            "control_session_id": shared["ran_epoch_id"],
        }
        reseal(same_identity, {"clean": True, "errors": []})
        with self.assertRaisesRegex(
            n3.CalibrationFailure, "distinct fresh-RAN"
        ):
            n3.verify_rung_evidence(
                rung,
                expected_status=status,
                expected_rung_index=2,
                expected_command_db=-3.0,
                expected_candidate_applications=1,
                expected_config_sha256="config-seal",
                expected_runner_sha256="runner-seal",
                require_clean_restore=True,
            )

        reseal(shared, {"clean": False, "errors": ["leftover port"]})
        with self.assertRaisesRegex(n3.CalibrationFailure, "cleanup evidence"):
            n3.verify_rung_evidence(
                rung,
                expected_status=status,
                expected_rung_index=2,
                expected_command_db=-3.0,
                expected_candidate_applications=1,
                expected_config_sha256="config-seal",
                expected_runner_sha256="runner-seal",
                require_clean_restore=True,
            )

    def test_direct_rung_preflight_requires_clean_control_proof(self) -> None:
        runner = n3.RungRunner(
            n3.DEFAULT_CONFIG, self.root / "no-proof", rung_index=0, command_db=-4.0,
        )
        with self.assertRaisesRegex(n3.CalibrationFailure, "clean-control proof"):
            runner.verify_dependencies()

    def test_direct_rung_rejects_forged_minimal_clean_control_proof(self) -> None:
        evidence = self.root / "forged-control"
        evidence.mkdir()
        manifest = evidence / "manifest.json"
        manifest.write_text("{}\n", encoding="utf-8")
        terminal = evidence / "UE_N3_CLEAN_RECEIVER_CONTROL_PASSED.json"
        terminal.write_text(json.dumps({
            "status": "UE_N3_CLEAN_RECEIVER_CONTROL_PASSED",
            "manifest_sha256": n3.n2.sha256(manifest),
        }) + "\n", encoding="utf-8")
        proof = {
            "directory": str(evidence),
            "terminal": terminal.name,
            "terminal_sha256": n3.n2.sha256(terminal),
            "manifest": manifest.name,
            "manifest_sha256": n3.n2.sha256(manifest),
        }
        runner = n3.RungRunner(
            n3.DEFAULT_CONFIG, self.root / "forged-rung", rung_index=0,
            command_db=-4.0, clean_control_proof=proof,
        )
        with self.assertRaisesRegex(n3.CalibrationFailure, "primary service pass"):
            runner.verify_dependencies()

    def test_live_campaign_authority_failure_has_terminal(self) -> None:
        output = self.root / "blocked-live"
        runner = n3.CampaignRunner(n3.DEFAULT_CONFIG, output)
        self.assertEqual(runner.execute(), 1)
        terminal = n3.load_json(output / "FAILED.json")
        self.assertEqual(terminal["status"], "FAILED")
        self.assertIn("live OAI authority", terminal["error"])

    def test_campaign_stops_after_first_hard_loss(self) -> None:
        live = copy.deepcopy(self.config)
        live["authority"]["live_oai_run_authorized"] = True
        live["authority"]["live_socket_execution_authorized"] = True
        live["authority"]["live_authority_basis"] = (
            "USER_REQUEST_2026-08-21_CONTINUE_LOWER_OAI_SNR_SEARCH_"
            "AFTER_CLEAN_CONTROL_PASS"
        )
        config_path = self.root / "live-authorized-for-mocked-test.json"
        config_path.write_text(json.dumps(live), encoding="utf-8")
        clean_control = self.root / "clean-control"
        clean_control.mkdir()
        approved_clean_config = n3.resolve_repo_path(
            live["live_prerequisites"]["clean_control_config_path"]
        )
        clean_outputs = {
            "summary.json": {
                "status": "UE_N3_CLEAN_RECEIVER_CONTROL_PASSED",
                "receiver_gate": {"primary_usable_service_pass": True},
                "restored_to_clean_minus50": True,
                "cleanup_clean": True,
                "connectivity_gate": {"status": "PASSED"},
            },
            "receiver_gate.json": {"primary_usable_service_pass": True},
            "cleanup_report.json": {
                "clean": True, "ext_dn_structured_udp_port_busy": False,
            },
            "preflight.json": {"status": "PASSED"},
            "connectivity_gate.json": {"status": "PASSED"},
            "mcs_summary.json": {"mcs_table": 0, "force_ul_mcs": -1},
            "resolved_config.json": n3.load_json(approved_clean_config),
        }
        for name, payload in clean_outputs.items():
            (clean_control / name).write_text(
                json.dumps(payload) + "\n", encoding="utf-8",
            )
        manifest = clean_control / "manifest.json"
        clean_runner_hash = next(
            seal["sha256"] for seal in live["runtime_seals"]
            if seal["path"] == live["live_prerequisites"]["clean_control_runner_path"]
        )
        manifest.write_text(json.dumps({
            "status": "UE_N3_CLEAN_RECEIVER_CONTROL_PASSED",
            "mode": "CLEAN_RECEIVER_CONTROL",
            "runner_sha256": clean_runner_hash,
            "config_path": str(approved_clean_config),
            "config_sha256": n3.n2.sha256(approved_clean_config),
            "outputs": [
                {
                    "path": name,
                    "bytes": (clean_control / name).stat().st_size,
                    "sha256": n3.n2.sha256(clean_control / name),
                }
                for name in clean_outputs
            ],
        }) + "\n", encoding="utf-8")
        (clean_control / "UE_N3_CLEAN_RECEIVER_CONTROL_PASSED.json").write_text(
            json.dumps({
                "status": "UE_N3_CLEAN_RECEIVER_CONTROL_PASSED",
                "manifest_sha256": n3.n2.sha256(manifest),
                "primary_usable_service_pass": True,
                "clean_restore_verified": True,
                "mapping_promoted": False,
                "numeric_bound_promoted": False,
            }),
            encoding="utf-8",
        )
        calls: list[float] = []

        class FakeRungRunner:
            def __init__(
                self, _config_path, output_dir, *, rung_index, command_db,
                clean_control_proof,
            ):
                self.output_dir = Path(output_dir)
                self.output_dir.mkdir(parents=True)
                self.rung_index = rung_index
                self.command_db = command_db
                self.clean_control_proof = clean_control_proof

            def run(self):
                calls.append(self.command_db)
                n3.n2.atomic_json(self.output_dir / "rung_summary.json", {
                    "status": n3.RUNG_HARD_LOSS,
                    "rung_index": self.rung_index,
                    "commanded_noise_power_db": self.command_db,
                    "candidate_application_count": 1,
                    "achieved_pusch_snr_db_median": None,
                    "target_mapping_promoted": False,
                    "numeric_bound_promoted": False,
                })
                return 0

        output = self.root / "mocked-live"
        verified = {
            "status": "VERIFIED_RUNG_EVIDENCE",
            "ran_epoch_id": "fresh-ran-epoch-0",
            "control_session_id": "fresh-control-session-0",
        }
        with (
            mock.patch.object(n3, "RungRunner", FakeRungRunner),
            mock.patch.object(n3, "verify_rung_evidence", return_value=verified),
        ):
            self.assertEqual(n3.CampaignRunner(
                config_path, output, clean_control_evidence=clean_control,
            ).execute(), 0)
        self.assertEqual(calls, [-4.0])
        summary = n3.load_json(output / "campaign_summary.json")
        self.assertEqual(summary["rungs_executed"], 1)
        self.assertTrue(summary["stopped_after_hard_loss"])
        self.assertEqual(summary["status"], n3.CAMPAIGN_UNRESOLVED)
        self.assertFalse(summary["target_mapping_promoted"])
        self.assertFalse(summary["numeric_bound_promoted"])
        self.assertFalse(summary["cold_attach_bound_evaluated"])
        self.assertEqual(summary["clean_control_predecessor"]["status"],
                         "VERIFIED_READ_ONLY_PREDECESSOR")

        class FakeCapturedRunner(FakeRungRunner):
            def run(self):
                calls.append(self.command_db)
                n3.n2.atomic_json(self.output_dir / "rung_summary.json", {
                    "status": n3.RUNG_CAPTURED,
                    "rung_index": self.rung_index,
                    "commanded_noise_power_db": self.command_db,
                    "candidate_application_count": 1,
                    "achieved_pusch_snr_db_median": 6.0,
                    "target_mapping_promoted": False,
                    "numeric_bound_promoted": False,
                })
                return 0

        reused_output = self.root / "reused-ran-identity"
        calls.clear()
        with (
            mock.patch.object(n3, "RungRunner", FakeCapturedRunner),
            mock.patch.object(
                n3, "verify_rung_evidence", return_value=verified
            ),
        ):
            self.assertEqual(n3.CampaignRunner(
                config_path, reused_output,
                clean_control_evidence=clean_control,
            ).execute(), 1)
        reused_terminal = n3.load_json(reused_output / "FAILED.json")
        self.assertIn("epoch identity was reused", reused_terminal["error"])
        self.assertEqual(len(calls), 2)

        class FakePreflightFailureRunner(FakeRungRunner):
            def run(self):
                calls.append(self.command_db)
                n3.n2.atomic_json(self.output_dir / "rung_summary.json", {
                    "status": "FAILED",
                    "rung_index": self.rung_index,
                    "commanded_noise_power_db": self.command_db,
                    "candidate_application_count": 0,
                    "error_type": "FileExistsError",
                    "error": "root preflight failure",
                    "target_mapping_promoted": False,
                    "numeric_bound_promoted": False,
                })
                return 1

        failed_output = self.root / "preserved-preflight-failure"
        calls.clear()
        with (
            mock.patch.object(n3, "RungRunner", FakePreflightFailureRunner),
            mock.patch.object(
                n3, "verify_rung_evidence", return_value={
                    **verified,
                    "ran_epoch_id": "failed-ran-epoch",
                    "control_session_id": "failed-control-session",
                },
            ),
        ):
            self.assertEqual(n3.CampaignRunner(
                config_path, failed_output,
                clean_control_evidence=clean_control,
            ).execute(), 1)
        failed_summary = n3.load_json(failed_output / "campaign_summary.json")
        self.assertEqual(failed_summary["status"], "FAILED")
        self.assertEqual(
            failed_summary["failed_rung"]["error"], "root preflight failure"
        )
        self.assertEqual(
            failed_summary["failed_rung"]["candidate_application_count"], 0
        )

        altered_resolved = n3.load_json(clean_control / "resolved_config.json")
        altered_resolved["actuator"][
            "clean_and_restore_commanded_noise_power_db"
        ] = "-40"
        n3.n2.atomic_json(clean_control / "resolved_config.json", altered_resolved)
        altered_manifest = n3.load_json(manifest)
        for artifact in altered_manifest["outputs"]:
            if artifact["path"] == "resolved_config.json":
                artifact["bytes"] = (
                    clean_control / "resolved_config.json"
                ).stat().st_size
                artifact["sha256"] = n3.n2.sha256(
                    clean_control / "resolved_config.json"
                )
        n3.n2.atomic_json(manifest, altered_manifest)
        terminal_path = (
            clean_control / "UE_N3_CLEAN_RECEIVER_CONTROL_PASSED.json"
        )
        altered_terminal = n3.load_json(terminal_path)
        altered_terminal["manifest_sha256"] = n3.n2.sha256(manifest)
        n3.n2.atomic_json(terminal_path, altered_terminal)
        altered_output = self.root / "altered-clean-config"
        self.assertEqual(n3.CampaignRunner(
            config_path, altered_output, clean_control_evidence=clean_control,
        ).execute(), 1)
        self.assertIn(
            "resolved config differs",
            n3.load_json(altered_output / "FAILED.json")["error"],
        )

    def test_failed_terminal_preserves_diagnostic_details(self) -> None:
        runner = n3.RungRunner(
            n3.DEFAULT_CONFIG, self.root / "failure", rung_index=0, command_db=-4.0,
        )
        runner.write_manifest_terminal("FAILED", {
            "status": "FAILED", "error_type": "CalibrationFailure",
            "error": "specific diagnostic", "cleanup_errors": ["specific cleanup"],
        })
        terminal = n3.load_json(runner.output_dir / "FAILED.json")
        self.assertEqual(terminal["error"], "specific diagnostic")
        self.assertEqual(terminal["cleanup_errors"], ["specific cleanup"])


if __name__ == "__main__":
    unittest.main()
