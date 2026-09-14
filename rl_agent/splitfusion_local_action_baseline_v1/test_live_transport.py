#!/usr/bin/env python3
"""CPU-only tests for the LOCAL transport measurement helpers."""

from __future__ import annotations

from rl_agent.splitfusion_local_action_baseline_v1.compute_baseline import build_local_payload
from rl_agent.splitfusion_local_action_baseline_v1.live_transport import (
    chunk_payload,
    merge_sender_sink_rows,
    rewrite_payload,
)
from rl_agent.splitfusion_local_action_baseline_v1.map_sink import CHUNK_HEADER, decode_and_validate


def main() -> int:
    source = build_local_payload(
        run_id="old", frame_id=9, capture_timestamp_ns=100,
        local_result_available_ns=150,
        records=[{"frame_id": 9, "stream_id": "old/local", "capture_timestamp_ns": 100, "sample_id": "old:9", "class_id": 1, "score": 0.8, "world_x": 1.0}],
        checkpoint_sha256="a" * 64,
    )
    payload = rewrite_payload(source, run_id="new", stream_id="new/favorable", frame_id=3, capture_ns=1_000, available_ns=1_050)
    decoded = decode_and_validate(payload, 3)
    assert decoded["stream_id"] == "new/favorable"
    assert decoded["objects"][0]["frame_id"] == 3
    packets = chunk_payload(payload, 3, 37)
    rebuilt = b"".join(packet[CHUNK_HEADER.size:] for packet in packets)
    assert rebuilt == payload
    command = {"frame_id": 3, "schedule_status": "SENT_ONCE", "capture_raw_ns": 1_000, "local_result_available_raw_ns": 1_050, "application_send_attempts": 1}
    ack = {3: {"status": "ACK_INSTALLED", "ack_receive_raw_ns": 1_210}}
    sink = {3: {"first_datagram_raw_ns": 1_100, "complete_raw_ns": 1_150, "edge_install_raw_ns": 1_170, "feedback_emit_raw_ns": 1_180}}
    row = merge_sender_sink_rows([command], ack, sink)[0]
    assert row["local_to_map_install_ms"] == 0.00012
    assert row["capture_to_map_install_ms"] == 0.00017
    assert row["fresh_150ms"] is True
    print("LOCAL live transport tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
