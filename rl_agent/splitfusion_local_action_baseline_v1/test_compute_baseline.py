#!/usr/bin/env python3
"""CPU-only LOCAL payload contract tests."""

from __future__ import annotations

from rl_agent.splitfusion_local_action_baseline_v1.compute_baseline import (
    LOCAL_PROFILE_ID,
    build_local_payload,
    decode_local_payload,
)


def main() -> int:
    record = {
        "stream_id": "run/local",
        "capture_timestamp_ns": 100,
        "sample_id": "run/local:7",
        "frame_id": 7,
        "class_id": 1,
        "score": 0.9,
        "world_x": 1.0,
        "world_y": 2.0,
    }
    payload = build_local_payload(
        run_id="run",
        frame_id=7,
        capture_timestamp_ns=100,
        local_result_available_ns=150,
        records=[record],
        checkpoint_sha256="a" * 64,
    )
    decoded = decode_local_payload(payload)
    assert decoded["profile_id"] == LOCAL_PROFILE_ID
    assert decoded["frame_id"] == 7
    assert decoded["objects"] == [record]
    assert decoded["segmentation"]["transported"] is False
    assert decoded["segmentation"]["edge_map_credit"] is False
    print("LOCAL compute contract tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
