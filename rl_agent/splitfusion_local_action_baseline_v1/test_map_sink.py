#!/usr/bin/env python3
"""CPU-only LOCAL map-sink validation tests."""

from __future__ import annotations

from rl_agent.splitfusion_local_action_baseline_v1.compute_baseline import build_local_payload
from rl_agent.splitfusion_local_action_baseline_v1.map_sink import decode_and_validate


def record(frame: int, timestamp: int) -> dict[str, object]:
    return {
        "stream_id": "run/local",
        "capture_timestamp_ns": timestamp,
        "sample_id": f"run/local:{frame}",
        "frame_id": frame,
        "class_id": 1,
        "score": 0.9,
        "world_x": 1.0,
        "world_y": 2.0,
    }


def main() -> int:
    payload = build_local_payload(
        run_id="run",
        frame_id=7,
        capture_timestamp_ns=100,
        local_result_available_ns=150,
        records=[record(7, 100)],
        checkpoint_sha256="a" * 64,
    )
    assert decode_and_validate(payload, 7)["objects"] == [record(7, 100)]
    try:
        decode_and_validate(payload, 8)
    except ValueError as exc:
        assert str(exc) == "frame_identity"
    else:
        raise AssertionError("message/frame substitution was accepted")
    corrupt = bytearray(payload)
    corrupt[-1] ^= 1
    try:
        decode_and_validate(bytes(corrupt), 7)
    except Exception:
        pass
    else:
        raise AssertionError("corrupt compressed payload was accepted")
    print("LOCAL map sink tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
