#!/usr/bin/env python3
"""Offline regression tests for the bounded live quality-feedback harness."""

from __future__ import annotations

import json
import socket
import struct
import csv
import tempfile
import unittest
from pathlib import Path

from rl_agent.splitfusion_quality_feedback_probe_v1.live_cell_child import (
    RouteBudgetReached,
    build_bounded_collector_class,
)
from rl_agent.splitfusion_quality_feedback_probe_v1.live_probe import (
    DEFAULT_CONFIG,
    _parse_csv_ints,
    _parse_csv_strings,
    _probe_campaign,
    _registered_cells,
    build_parser,
)
from rl_agent.splitfusion_quality_feedback_probe_v1 import protocol
from rl_agent.splitfusion_quality_feedback_probe_v1.analyze_live_probe import (
    analyze_attempt,
    summarize,
)
from rl_agent.splitfusion_quality_feedback_probe_v1.packet_evidence import (
    QualityAckCapture,
    _source_endpoint_matches,
    parse_quality_pcap,
)


class _Counters:
    def __init__(self) -> None:
        self.values: dict[str, int] = {}

    def bump(self, name: str, amount: int = 1) -> None:
        self.values[name] = self.values.get(name, 0) + int(amount)

    def snapshot(self) -> dict[str, int]:
        return dict(self.values)


class _FakeCollector:
    def __init__(self, **keywords: object) -> None:
        self.attempt_dir = Path(str(keywords["attempt_dir"]))
        self.sent = 0
        self.dropped = 0
        self.failures: list[str] = []
        self.cleanup_ok = False
        self.transport_counters = _Counters()
        # These sentinels prove the wrapper does not install the timing
        # diagnostic's discarding queues.
        self.segmentation_queue = object()
        self.evaluation_queue = object()
        self.exact_retrieval_queue = object()
        self.per_frame_written = False
        self.perception_written = False

    def on_world_tick(self, frame_id: int, route_tick: int) -> None:
        del frame_id, route_tick

    def _process_token(self, token: object) -> None:
        del token
        self.sent += 1

    def finish(self) -> bool:
        self.cleanup_ok = True
        return True

    def write_per_frame(self) -> None:
        self.per_frame_written = True
        (self.attempt_dir / "per_frame_metrics.csv").write_text(
            "frame_id\n", encoding="utf-8"
        )

    def write_perception(self) -> None:
        self.perception_written = True
        (self.attempt_dir / "perception_metrics.csv").write_text(
            "frame_id\n", encoding="utf-8"
        )


class BoundedCollectorTests(unittest.TestCase):
    def test_exact_budget_and_non_discarding_queues(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            bounded = build_bounded_collector_class(
                _FakeCollector,
                transmitted_budget=3,
                safety_timeout_s=10.0,
                artifacts_dir=artifacts,
            )
            collector = bounded(attempt_dir=root)
            original_queues = (
                collector.segmentation_queue,
                collector.evaluation_queue,
                collector.exact_retrieval_queue,
            )
            for ordinal in range(3):
                collector._process_token({"ordinal": ordinal})
            self.assertEqual(collector.sent, 3)
            self.assertEqual(
                collector.probe_stop_reason, "TRANSMITTED_BUDGET_REACHED"
            )
            collector._process_token({"ordinal": 4})
            self.assertEqual(collector.sent, 3)
            with self.assertRaises(RouteBudgetReached):
                collector.on_world_tick(4, 4)
            self.assertTrue(collector.finish())
            self.assertEqual(
                original_queues,
                (
                    collector.segmentation_queue,
                    collector.evaluation_queue,
                    collector.exact_retrieval_queue,
                ),
            )
            summary = json.loads(
                (artifacts / "collector_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["transmitted_frames"], 3)
            self.assertTrue(summary["quality_and_evaluation_drain_complete"])
            self.assertFalse(summary["evaluation_queues_discarded"])
            self.assertTrue(collector.per_frame_written)
            self.assertTrue(collector.perception_written)


class ParentContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))

    def test_default_matrix_is_two_pinned_action_50_cells(self) -> None:
        cells = _registered_cells(
            self.config,
            action_ids=(50,),
            profile_ids=("FAVORABLE_STABLE", "ADVERSE_STABLE"),
        )
        self.assertEqual(len(cells), 2)
        self.assertEqual([cell.action_id for cell in cells], [50, 50])
        self.assertEqual(
            [cell.network_profile_id for cell in cells],
            ["FAVORABLE_STABLE", "ADVERSE_STABLE"],
        )

    def test_unpinned_action_fails_closed(self) -> None:
        with self.assertRaisesRegex(Exception, "not pinned"):
            _registered_cells(
                self.config,
                action_ids=(52, 70),
                profile_ids=("FAVORABLE_STABLE",),
            )

    def test_cli_defaults(self) -> None:
        args = build_parser().parse_args(["--output-root", "/tmp/probe"])
        self.assertEqual(_parse_csv_ints(args.actions), (50,))
        self.assertEqual(
            _parse_csv_strings(args.network_profiles),
            ("FAVORABLE_STABLE", "ADVERSE_STABLE"),
        )
        self.assertEqual(args.transmitted_budget, 300)

    def test_probe_campaign_cannot_activate_reserved_qualification_mode(self) -> None:
        inherited = {**self.config, "_qualification": {"action_ids": [50]}}
        campaign = _probe_campaign(inherited, run_id="test")
        self.assertNotIn("_qualification", campaign)
        self.assertEqual(
            campaign["campaign_id"],
            "splitfusion_quality_feedback_probe_v1/test",
        )


class PacketAndAnalysisTests(unittest.TestCase):
    def test_udp_wildcard_local_ip_defers_to_packet_but_port_stays_exact(self) -> None:
        packet = {"source_ip": "192.168.70.140", "source_port": 44798}
        self.assertTrue(
            _source_endpoint_matches(packet, ("0.0.0.0", 44798))
        )
        self.assertTrue(
            _source_endpoint_matches(packet, ("192.168.70.140", 44798))
        )
        self.assertFalse(
            _source_endpoint_matches(packet, ("192.168.70.141", 44798))
        )
        self.assertFalse(
            _source_endpoint_matches(packet, ("0.0.0.0", 44799))
        )

    def test_capture_is_explicitly_on_host_owned_ue_tunnel(self) -> None:
        capture = QualityAckCapture(
            Path("/tmp/unused"),
            interface="oaitun_ue1",
            ue_host="10.0.0.2",
            ue_port=51014,
        )
        self.assertEqual(
            capture.link_check_argv()[:4],
            ("sudo", "-n", "ip", "link"),
        )
        argv = capture.capture_argv()
        self.assertEqual(argv[:3], ("sudo", "-n", "tcpdump"))
        self.assertIn("oaitun_ue1", argv)
        self.assertIn("51014", argv)

    def test_classic_pcap_decodes_compact_quality_ack(self) -> None:
        identity = {
            "run_id": "r", "cell_id": "c", "stream_id": "s",
            "frame_id": 7, "action_id": 50, "profile_id": "p",
            "capture_timestamp_ns": 1_000_000_000,
        }
        timing = {name: None for name in protocol.TIMING_FIELDS}
        timing["ack_emit_start_wall_ns"] = 1_100_000_000
        message = protocol.build_ack(
            identity_fields=identity,
            frozen_carla_frame_id=7,
            timing=timing,
            quality=None,
            evaluator_mode="test",
            detail_sha256="a" * 64,
            failure_reason="TEST_FAILURE",
        )
        payload = protocol.canonical_bytes(message)
        udp = struct.pack("!HHHH", 40000, 51014, 8 + len(payload), 0) + payload
        ip = bytearray(20)
        ip[0] = 0x45
        ip[2:4] = struct.pack("!H", 20 + len(udp))
        ip[8] = 64
        ip[9] = 17
        ip[12:16] = socket.inet_aton("192.168.70.140")
        ip[16:20] = socket.inet_aton("10.0.0.2")
        ethernet = b"\0" * 12 + b"\x08\x00" + bytes(ip) + udp
        global_header = struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)
        record = struct.pack("<IIII", 2, 3, len(ethernet), len(ethernet)) + ethernet
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "one.pcap"
            path.write_bytes(global_header + record)
            packets, rows = parse_quality_pcap(path)
        self.assertEqual(packets, 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["destination_port"], 51014)
        self.assertEqual(rows[0]["message_sha256"], protocol.digest(message))

    def test_percentiles_and_denominators_are_explicit(self) -> None:
        result = summarize([10.0, 20.0, 200.0], denominator=5)
        self.assertEqual(result["available"], 3)
        self.assertEqual(result["denominator_sent"], 5)
        self.assertEqual(result["within_140ms_count"], 2)
        self.assertAlmostEqual(result["within_140ms_fraction_of_sent"], 0.4)

    def test_analyzer_keeps_not_eligible_out_of_ack_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "direct_edge_map").mkdir()
            fields = [
                "run_id", "cell_id", "stream_id", "frame_id", "action_id",
                "profile_id", "capture_timestamp_ns", "sensor_ready_wall_ns",
                "model_action_start_wall_ns", "first_feature_datagram_send_wall_ns",
                "final_prediction_ready_wall_ns", "evaluation_enqueued_wall_ns",
                "evaluation_started_wall_ns", "evaluation_completed_wall_ns",
                "ack_emit_start_wall_ns", "quality_ack_received_wall_ns",
                "quality_feedback_event",
            ]
            base = {
                "run_id": "r", "cell_id": "c", "stream_id": "s",
                "action_id": 50, "profile_id": "p",
                "sensor_ready_wall_ns": 1_010_000_000,
                "model_action_start_wall_ns": 1_020_000_000,
                "first_feature_datagram_send_wall_ns": 1_030_000_000,
            }
            rows = [
                {
                    **base, "frame_id": 1, "capture_timestamp_ns": 1_000_000_000,
                    "final_prediction_ready_wall_ns": 1_080_000_000,
                    "evaluation_enqueued_wall_ns": 1_081_000_000,
                    "evaluation_started_wall_ns": 1_082_000_000,
                    "evaluation_completed_wall_ns": 1_090_000_000,
                    "ack_emit_start_wall_ns": 1_091_000_000,
                    "quality_ack_received_wall_ns": 1_100_000_000,
                    "quality_feedback_event": "QUALITY_EVALUATED",
                },
                {
                    **base, "frame_id": 2, "capture_timestamp_ns": 2_000_000_000,
                    "sensor_ready_wall_ns": 2_010_000_000,
                    "model_action_start_wall_ns": 2_020_000_000,
                    "first_feature_datagram_send_wall_ns": 2_030_000_000,
                    "final_prediction_ready_wall_ns": "",
                    "evaluation_enqueued_wall_ns": "",
                    "evaluation_started_wall_ns": "",
                    "evaluation_completed_wall_ns": "",
                    "ack_emit_start_wall_ns": "",
                    "quality_ack_received_wall_ns": "",
                    "quality_feedback_event": "",
                },
            ]
            with (root / "quality_policy_timing.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with (root / "quality_feedback.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["message_sha256"])
                writer.writeheader()
                writer.writerow({"message_sha256": "a" * 64})
                writer.writerow({"message_sha256": ""})
            (root / "direct_edge_map/quality_edge_report.json").write_text(
                json.dumps(
                    {
                        "messages": [
                            {
                                "sha256": "a" * 64,
                                "identity": ["r", "c", "s", 1, 50, "p", 1_000_000_000],
                                "socket_send_call_wall_ns": 1_092_000_000,
                            }
                        ]
                    }
                ) + "\n",
                encoding="utf-8",
            )
            (root / "direct_edge_map/direct_edge_counters.json").write_text(
                json.dumps({"counters": {"feature_messages_reassembled": 1}}) + "\n",
                encoding="utf-8",
            )
            (root / "quality_ack_packet_evidence.json").write_text(
                json.dumps({"quality_ack_packets": 1}) + "\n", encoding="utf-8"
            )
            result = analyze_attempt(root)
            population = result["population_denominators"]
            self.assertEqual(population["sent"], 2)
            self.assertEqual(population["quality_ack_received"], 1)
            self.assertEqual(population["not_eligible_no_final_prediction"], 1)


if __name__ == "__main__":
    unittest.main()
