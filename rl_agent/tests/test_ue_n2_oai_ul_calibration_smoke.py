from __future__ import annotations

import csv
import json
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import rl_agent.ue_n2_oai_ul_calibration_smoke as n2


SHOW = """model 0 rfsimu_channel_enB0 type AWGN:
model owner: not
max Doppler: 0 path loss: 0.000000  noise: -50.000000 rchannel offset: 0
model 2 rfsimu_channel_ue0 type AWGN:
model owner: rfsimulator
max Doppler: 0 path loss: 0.000000  noise: -50.000000 rchannel offset: 0
softmodem_gnb> """


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class FakeTelnet:
    def __init__(self, period_ns: int, *, late: bool = False):
        self.period_ns = period_ns
        self.late = late
        self.calls: list[str] = []

    def command(self, command: str):
        self.calls.append(command)
        index = len(self.calls) - 1
        target = command.rsplit(" ", 1)[-1]
        send = 500_000_000 + index * self.period_ns
        response = send + (self.period_ns + 1 if self.late else 1)
        payload = (
            "model owner: rfsimulator\nmax Doppler: 0 path loss: 0.000000  "
            f"noise: {float(target):.6f} rchannel offset: 0\nsoftmodem_gnb> "
        )
        return send, send, response, response, payload


class FakeSocket:
    def __init__(self, responses: list[bytes]):
        self.responses = list(responses)
        self.sent: list[bytes] = []
        self.closed = False

    def sendall(self, value: bytes) -> None:
        self.sent.append(value)

    def recv(self, _: int) -> bytes:
        return self.responses.pop(0)

    def settimeout(self, _: float) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class UEN2OwnedRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(n2.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        self.temp = tempfile.TemporaryDirectory(prefix="ue-n2-tests-")
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def runner(self, name: str = "run") -> n2.Runner:
        return n2.Runner(n2.DEFAULT_CONFIG, self.root / name)

    def test_config_freezes_bounded_partial_smoke(self) -> None:
        self.assertEqual(self.config["schedule"]["commanded_noise_plateaus_db"], ["-10", "-8", "-5", "-4"])
        self.assertEqual(self.config["schedule"]["period_ns"], 100_000_000)
        self.assertEqual(self.config["schedule"]["commands_per_plateau"], 30)
        self.assertEqual(self.config["traffic"]["method"], "CARLA_SHAPED_UDP_10HZ")
        self.assertEqual(self.config["traffic"]["frame_bytes"], 25_000)
        self.assertEqual(self.config["traffic"]["remote_port"], 56_120)
        self.assertEqual(self.config["telemetry"]["direct_ul_bler_status"], "UNAVAILABLE_UNRESOLVED")
        self.assertFalse(self.config["analysis"]["direct_ul_bler_zero_fill_authorized"])

    def test_materialized_configs_are_clean_and_source_is_unchanged(self) -> None:
        runner = self.runner()
        source = runner.path(self.config["paths"]["oai_ran_conf"]) / self.config["paths"]["channel_config"]
        before = n2.sha256(source)
        gnb, ue = runner.materialize_configs()
        self.assertEqual(n2.sha256(source), before)
        self.assertEqual(gnb.read_text().count("noise_power_dB = -50;"), 3)
        self.assertEqual(ue.read_text().count("noise_power_dB = -50;"), 3)
        self.assertNotIn("noise_power_dBFS", gnb.read_text() + ue.read_text())
        hashes = json.loads((runner.output_dir / "runtime/config_hashes.json").read_text())
        self.assertEqual(hashes["gnb_sha256"], n2.sha256(gnb))
        self.assertEqual(hashes["ue_sha256"], n2.sha256(ue))

    def test_channel_parser_preserves_dynamic_index_owner_and_clean_state(self) -> None:
        models = n2.parse_channel_models(SHOW)
        self.assertEqual(models["rfsimu_channel_ue0"]["model_index"], 2)
        self.assertEqual(models["rfsimu_channel_ue0"]["owner"], "rfsimulator")
        self.assertEqual(models["rfsimu_channel_ue0"]["path_loss_db"], 0)
        self.assertEqual(models["rfsimu_channel_ue0"]["noise_power_db"], -50)

    def test_telnet_session_reuses_one_socket_for_multiple_commands(self) -> None:
        response = b"model owner: rfsimulator\npath loss: 0 noise: -10\nsoftmodem_gnb> "
        sock = FakeSocket([b"softmodem_gnb> ", response, response])
        with mock.patch.object(n2.socket, "create_connection", return_value=sock) as connect:
            session = n2.TelnetSession("127.0.0.1", 9090, 2, 10000)
            session.command("channelmod modify 2 noise_power_dB -10")
            session.command("channelmod modify 2 noise_power_dB -10")
            session.close()
        connect.assert_called_once()
        self.assertEqual(sock.sent[0], b"\n")
        self.assertEqual(sock.sent[1:], [
            b"channelmod modify 2 noise_power_dB -10\n",
            b"channelmod modify 2 noise_power_dB -10\n",
        ])
        self.assertTrue(sock.closed)

    def test_modify_response_rejects_owner_path_or_noise_mismatch(self) -> None:
        valid = "model owner: rfsimulator\npath loss: 0 noise: -8\nsoftmodem_gnb> "
        n2.Runner.validate_modify_response(valid, "-8")
        for invalid in (
            valid.replace("rfsimulator", "notset"),
            valid.replace("path loss: 0", "path loss: 1"),
            valid.replace("noise: -8", "noise: -7"),
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(n2.SmokeFailure):
                    n2.Runner.validate_modify_response(invalid, "-8")

    @staticmethod
    def trace_clock(period: int):
        values = [0]
        for index in range(120):
            scheduled = 500_000_000 + index * period
            values.extend([scheduled, scheduled])
        return iter(values)

    def test_trace_has_120_acks_and_only_four_value_transitions(self) -> None:
        runner = self.runner()
        period = int(runner.config["schedule"]["period_ns"])
        fake = FakeTelnet(period)
        runner.telnet = fake  # type: ignore[assignment]
        clock = self.trace_clock(period)
        with mock.patch.object(n2.time, "monotonic_ns", side_effect=lambda: next(clock)), mock.patch.object(n2.time, "sleep", return_value=None):
            runner.run_trace(2)
        self.assertEqual(len(runner.command_rows), 120)
        self.assertTrue(all(row["status"] == "ACK_VALIDATED" for row in runner.command_rows))
        self.assertEqual(sum(bool(row["is_value_transition"]) for row in runner.command_rows), 4)
        self.assertTrue(all(row["response_before_next_boundary"] for row in runner.command_rows))
        self.assertEqual(len(fake.calls), 120)

    def test_trace_fails_if_any_ack_crosses_next_boundary(self) -> None:
        runner = self.runner()
        period = int(runner.config["schedule"]["period_ns"])
        runner.telnet = FakeTelnet(period, late=True)  # type: ignore[assignment]
        clock = self.trace_clock(period)
        with mock.patch.object(n2.time, "monotonic_ns", side_effect=lambda: next(clock)), mock.patch.object(n2.time, "sleep", return_value=None):
            with self.assertRaisesRegex(n2.SmokeFailure, "crossed"):
                runner.run_trace(2)

    def test_analysis_is_per_plateau_and_never_promotes_causal_evidence(self) -> None:
        runner = self.runner()
        runner.current_rnti = 4660
        base_ns = time.time_ns()
        commands = []
        for index, command in enumerate(("-10", "-8", "-5", "-4")):
            start = base_ns + index * 3_000_000_000
            commands.append({
                "trace_index": index * 30, "plateau_index": index,
                "is_value_transition": True, "commanded_noise_power_db": command,
                "send_wall_time_ns": start, "handler_bracket_ms": 1.0,
                "response_received_wall_time_ns": start + 1_000_000,
                "response_before_next_boundary": True, "status": "ACK_VALIDATED",
            })
        runner.command_rows = commands
        power_rows, mcs_rows = [], []
        for index, (snr, ema, final_mcs) in enumerate(((20, 19, 24), (16, 16, 19), (10, 11, 12), (8, 9, 9))):
            event_ns = base_ns + index * 3_000_000_000 + 2_500_000_000
            stamp = datetime.fromtimestamp(event_ns / 1e9).astimezone().strftime("%H:%M:%S.%f")
            power_rows.append({"time": stamp, "rnti": "4660", "frame": str(index), "slot": "1", "snrx10": str(snr * 10), "mcs": str(final_mcs)})
            mcs_rows.append({
                "time": stamp, "rnti": "4660", "avg_snr_x10": str(ema * 10),
                "mcs_table": "0", "final_mcs": str(final_mcs), "force_ul_mcs": "-1",
            })
        base = runner.output_dir / "ttracer/gnb/csv"
        write_csv(base / "GNB_MAC_PUSCH_POWER_CONTROL.csv", list(power_rows[0]), power_rows)
        write_csv(base / "GNB_MAC_UL_MCS_DECISION.csv", list(mcs_rows[0]), mcs_rows)
        write_csv(runner.output_dir / "traffic/sender.csv", ["frame_index"], [{"frame_index": i} for i in range(120)])
        write_csv(runner.output_dir / "traffic/sink_packets.csv", ["size"], [{"size": 25000} for _ in range(120)])
        summary = runner.analyze()
        self.assertEqual(summary["status"], n2.SUCCESS_STATUS)
        self.assertEqual(len(summary["plateaus"]), 4)
        self.assertTrue(summary["descriptive_monotone_response"])
        self.assertFalse(summary["full_raw_event_envelope_satisfied"])
        self.assertEqual(summary["causal_first_effect_status"], "UNAVAILABLE_STOCK_TRACER_LIMITATION")
        self.assertEqual(summary["direct_ul_bler_status"], "UNAVAILABLE_UNRESOLVED")
        self.assertTrue(all(row["ema_settling_status"] == "NOT_SETTLED_INSUFFICIENT_COUNT" for row in summary["plateaus"]))

    def test_raw_limit_record_is_explicit(self) -> None:
        runner = self.runner()
        for source in ("gnb", "ue"):
            raw = runner.output_dir / "ttracer" / source / f"{source}.raw"
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_bytes(source.encode())
        runner.write_raw_limit_record()
        path = runner.output_dir / runner.config["output"]["raw_radio_events"]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual({row["source"] for row in rows}, {"gnb", "ue"})
        self.assertTrue(all(row["raw_event_envelope_status"] == "UNAVAILABLE_WITH_STOCK_EXPORTER" for row in rows))


if __name__ == "__main__":
    unittest.main()
