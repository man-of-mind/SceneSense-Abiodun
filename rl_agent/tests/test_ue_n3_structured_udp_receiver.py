from __future__ import annotations

import json
import socket
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from oai_layer_latency.carla_shaped_udp_burst_sender import make_payload
from rl_agent.ue_n3_structured_udp_receiver import (
    HEADER,
    PacketContractError,
    ReceiverAccounting,
    StructuredUdpReceiver,
    parse_ssburst_datagram,
)
import rl_agent.ue_n3_structured_udp_receiver as receiver_module


class FakeDatagramSocket:
    def __init__(self, packets: list[tuple[bytes, tuple[str, int]]]) -> None:
        self.packets = list(packets)
        self.closed = False

    def setsockopt(self, *_args: object) -> None:
        pass

    def bind(self, _address: tuple[str, int]) -> None:
        pass

    def settimeout(self, _timeout: float) -> None:
        pass

    def getsockname(self) -> tuple[str, int]:
        return ("127.0.0.1", 56130)

    def recvfrom(self, _maximum: int) -> tuple[bytes, tuple[str, int]]:
        if self.packets:
            return self.packets.pop(0)
        time.sleep(0.001)
        raise socket.timeout

    def close(self) -> None:
        self.closed = True


class SSBurstContractTests(unittest.TestCase):
    def test_parser_consumes_existing_sender_header_exactly(self) -> None:
        packet = make_payload(12_500, frame_index=17, chunk_index=0, chunks_per_frame=1)
        parsed = parse_ssburst_datagram(packet, max_chunks_per_frame=8)
        self.assertEqual(HEADER.format, "!8sIIII")
        self.assertEqual(parsed.frame_index, 17)
        self.assertEqual(parsed.chunk_index, 0)
        self.assertEqual(parsed.chunks_per_frame, 1)
        self.assertEqual(parsed.declared_datagram_bytes, 12_500)

    def test_parser_rejects_each_ambiguous_header_case(self) -> None:
        valid = make_payload(100, frame_index=0, chunk_index=0, chunks_per_frame=1)
        cases = {
            "DATAGRAM_SHORTER_THAN_HEADER": valid[: HEADER.size - 1],
            "MAGIC_MISMATCH": b"BADMAGIC" + valid[8:],
            "ZERO_CHUNKS_PER_FRAME": HEADER.pack(b"SSBURST", 0, 0, 0, HEADER.size),
            "CHUNK_INDEX_OUT_OF_RANGE": HEADER.pack(b"SSBURST", 0, 1, 1, HEADER.size),
            "DECLARED_DATAGRAM_SIZE_MISMATCH": HEADER.pack(b"SSBURST", 0, 0, 1, 999),
        }
        for expected_reason, packet in cases.items():
            with self.subTest(expected_reason=expected_reason):
                with self.assertRaises(PacketContractError) as captured:
                    parse_ssburst_datagram(packet, max_chunks_per_frame=8)
                self.assertEqual(captured.exception.reason, expected_reason)


class ReceiverAccountingTests(unittest.TestCase):
    @staticmethod
    def accounting(*, duration_s: float = 4.0, expected_frames: int = 4,
                   max_streams: int = 1, reorder_window_frames: int = 4) -> ReceiverAccounting:
        return ReceiverAccounting(
            measurement_start_monotonic_ns=1_000_000_000,
            duration_s=duration_s,
            expected_first_frame=0,
            expected_frames=expected_frames,
            expected_chunks_per_frame=1,
            max_streams=max_streams,
            reorder_window_frames=reorder_window_frames,
            max_chunks_per_frame=8,
        )

    @staticmethod
    def ingest(accounting: ReceiverAccounting, frame: int, *,
               monotonic_ns: int, address: tuple[str, int] = ("10.0.0.2", 44000)) -> dict:
        packet = make_payload(100, frame_index=frame, chunk_index=0, chunks_per_frame=1)
        return accounting.ingest(
            packet,
            address,
            wall_time_ns=1_700_000_000_000_000_000 + monotonic_ns,
            monotonic_ns=monotonic_ns,
        )

    def test_in_order_delivery_has_no_sequence_loss(self) -> None:
        accounting = self.accounting(duration_s=2.0, expected_frames=2)
        self.ingest(accounting, 0, monotonic_ns=1_100_000_000)
        self.ingest(accounting, 1, monotonic_ns=1_200_000_000)
        summary = accounting.finalize(
            end_monotonic_ns=3_000_000_000,
            stop_reason="DURATION_COMPLETE",
            clean_shutdown=True,
        )
        stream = summary["streams"][0]
        self.assertEqual(stream["complete_frames"], 2)
        self.assertEqual(stream["lost_chunks"], 0)
        self.assertEqual(stream["chunk_delivery_ratio"], 1.0)
        self.assertEqual(stream["bounded_pending_frames_after_finalize"], 0)

    def test_duplicate_reordering_loss_and_one_second_gaps_remain_distinct(self) -> None:
        accounting = self.accounting()
        self.assertEqual(
            self.ingest(accounting, 0, monotonic_ns=1_100_000_000)["status"],
            "ACCEPTED_UNIQUE",
        )
        self.ingest(accounting, 2, monotonic_ns=3_100_000_000)
        self.assertEqual(
            self.ingest(accounting, 2, monotonic_ns=3_200_000_000)["status"],
            "DUPLICATE_CHUNK",
        )
        reordered = self.ingest(accounting, 1, monotonic_ns=3_300_000_000)
        self.assertTrue(reordered["out_of_order"])
        summary = accounting.finalize(
            end_monotonic_ns=5_000_000_000,
            stop_reason="DURATION_COMPLETE",
            clean_shutdown=True,
        )
        stream = summary["streams"][0]
        self.assertEqual(stream["duplicate_chunks"], 1)
        self.assertEqual(stream["out_of_order_unique_chunks"], 1)
        self.assertEqual(stream["wholly_missing_frames"], 1)
        self.assertEqual(stream["lost_chunks"], 1)
        self.assertEqual(stream["empty_one_second_bins"], 2)
        self.assertEqual(stream["max_consecutive_empty_one_second_bins"], 1)
        self.assertGreaterEqual(stream["max_interarrival_gap_s"], 2.0)

    def test_far_sequence_jump_does_not_expand_pending_memory(self) -> None:
        accounting = self.accounting(
            duration_s=1.0,
            expected_frames=1_000,
            reorder_window_frames=4,
        )
        self.ingest(accounting, 999, monotonic_ns=1_100_000_000)
        tracker = next(iter(accounting.streams.values()))
        self.assertLessEqual(len(tracker.pending), 4)
        self.assertEqual(tracker.next_finalize_frame, 996)
        summary = accounting.finalize(
            end_monotonic_ns=2_000_000_000,
            stop_reason="DURATION_COMPLETE",
            clean_shutdown=True,
        )
        stream = summary["streams"][0]
        self.assertEqual(stream["expected_chunks"], 1_000)
        self.assertEqual(stream["lost_chunks"], 999)

    def test_stream_limit_is_fail_visible_without_unbounded_state(self) -> None:
        accounting = self.accounting(max_streams=1)
        self.ingest(accounting, 0, monotonic_ns=1_100_000_000, address=("10.0.0.2", 44000))
        rejected = self.ingest(
            accounting, 0, monotonic_ns=1_200_000_000,
            address=("10.0.0.3", 44001),
        )
        self.assertEqual(rejected["status"], "STREAM_LIMIT_EXCEEDED")
        self.assertEqual(len(accounting.streams), 1)
        self.assertEqual(accounting.stream_limit_exceeded_datagrams, 1)

    def test_header_contract_mismatch_is_not_counted_as_delivery(self) -> None:
        accounting = self.accounting(expected_frames=1)
        packet = make_payload(100, frame_index=0, chunk_index=0, chunks_per_frame=2)
        event = accounting.ingest(
            packet,
            ("10.0.0.2", 44000),
            wall_time_ns=1_700_000_000_000_000_000,
            monotonic_ns=1_100_000_000,
        )
        self.assertEqual(event["status"], "CHUNKS_PER_FRAME_CONTRACT_MISMATCH")
        summary = accounting.finalize(
            end_monotonic_ns=2_000_000_000,
            stop_reason="DURATION_COMPLETE",
            clean_shutdown=True,
        )
        stream = summary["streams"][0]
        self.assertEqual(stream["received_unique_chunks"], 0)
        self.assertEqual(stream["lost_chunks"], 1)


class LiveReceiverTests(unittest.TestCase):
    def test_invalid_bounds_fail_before_socket_creation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ue-n3-receiver-") as temp:
            root = Path(temp)
            with mock.patch.object(receiver_module.socket, "socket") as socket_factory:
                with self.assertRaisesRegex(ValueError, "max_streams"):
                    StructuredUdpReceiver(
                        bind_host="127.0.0.1",
                        port=56_130,
                        events_jsonl=root / "events.jsonl",
                        summary_json=root / "summary.json",
                        ready_json=None,
                        duration_s=60.0,
                        expected_first_frame=0,
                        expected_frames=600,
                        expected_chunks_per_frame=1,
                        max_streams=17,
                        reorder_window_frames=64,
                        max_chunks_per_frame=1_024,
                        socket_receive_buffer_bytes=1_048_576,
                    )
            socket_factory.assert_not_called()
            self.assertFalse(root.joinpath("events.jsonl").exists())
            self.assertFalse(root.joinpath("summary.json").exists())

    def test_duration_shutdown_writes_jsonl_and_atomic_summary(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ue-n3-receiver-") as temp:
            root = Path(temp)
            fake_socket = FakeDatagramSocket([
                (make_payload(100, frame, 0, 1), ("10.0.0.2", 44000))
                for frame in range(3)
            ])
            with mock.patch.object(receiver_module.socket, "socket", return_value=fake_socket):
                receiver = StructuredUdpReceiver(
                    bind_host="127.0.0.1",
                    port=0,
                    events_jsonl=root / "events.jsonl",
                    summary_json=root / "summary.json",
                    ready_json=root / "ready.json",
                    duration_s=0.03,
                    expected_first_frame=0,
                    expected_frames=3,
                    expected_chunks_per_frame=1,
                    max_streams=1,
                    reorder_window_frames=4,
                    max_chunks_per_frame=8,
                    socket_receive_buffer_bytes=1_048_576,
                    poll_timeout_s=0.002,
                )
                result = receiver.run(install_signal_handlers=False)
            self.assertTrue(fake_socket.closed)
            self.assertTrue((root / "ready.json").exists())
            self.assertEqual(result["status"], "CAPTURED")
            self.assertEqual(result["stop_reason"], "DURATION_COMPLETE")
            self.assertTrue(result["clean_shutdown"])
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["streams"][0]["received_unique_chunks"], 3)
            events = [
                json.loads(line)
                for line in (root / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                sum(row.get("status") == "ACCEPTED_UNIQUE" for row in events), 3
            )
            self.assertTrue(any(row["event_type"] == "one_second_interval" for row in events))

    def test_outputs_are_create_only(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ue-n3-receiver-") as temp:
            root = Path(temp)
            existing = root / "events.jsonl"
            existing.write_text("owned\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                StructuredUdpReceiver(
                    bind_host="127.0.0.1", port=0,
                    events_jsonl=existing,
                    summary_json=root / "summary.json",
                    ready_json=None,
                    duration_s=1.0,
                    expected_first_frame=0,
                    expected_frames=1,
                    expected_chunks_per_frame=1,
                    max_streams=1,
                    reorder_window_frames=1,
                    max_chunks_per_frame=1,
                    socket_receive_buffer_bytes=65_536,
                )


if __name__ == "__main__":
    unittest.main()
