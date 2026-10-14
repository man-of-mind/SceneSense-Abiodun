"""M3.3 route-triggered GOOD_TO_BAD target-SNR controller.

The retained 100-ms controller, inverse mapping, telnet validation, ACK
publication, catch-up policy and -50 dB restoration are unchanged. This module
replaces only its sequence source with a causal, one-way route trigger.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable

from . import contract as C

SCHEMA = "splitfusion.m3_route_b_v13.good_to_bad_controller.v1"


def _argument(argv, name):
    hits = [i for i, value in enumerate(argv) if value == name]
    if len(hits) != 1 or hits[0] + 1 >= len(argv):
        raise ValueError(f"expected exactly one {name} argument")
    return Path(argv[hits[0] + 1])


class RouteTriggeredSequence:
    """The retained ``next_sample`` interface with a permanent BAD latch."""

    def __init__(self, schedule: dict[str, Any], progress_path: Path, trace_path: Path,
                 *, monotonic_ns: Callable[[], int] = time.monotonic_ns,
                 raw_ns: Callable[[], int] = lambda: time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)):
        self.schedule = schedule
        self.progress_path, self.trace_path = Path(progress_path), Path(trace_path)
        self.monotonic_ns, self.raw_ns = monotonic_ns, raw_ns
        self.started_ns = monotonic_ns()
        self.seen_active = self.latched_bad = False
        self.transition: dict[str, Any] | None = None
        self.samples, self.last_frame_id = 0, -1
        self.trace = self.trace_path.open("x", encoding="utf-8")
        phases = schedule["phases"]
        if [row["phase"] for row in phases] != ["GOOD", "BAD"]:
            raise ValueError("registered schedule must contain ordered GOOD then BAD phases")
        self.good, self.bad = phases

    def _read(self, now: int) -> tuple[dict[str, Any] | None, str, float | None]:
        if not self.progress_path.exists():
            age = (now - self.started_ns) / 1e6
            if self.seen_active or age > self.schedule["startup_grace_ms"]:
                raise RuntimeError("route progress missing outside registered startup grace")
            return None, "STARTUP_WAITING_FOR_ROUTE", None
        try:
            value = json.loads(self.progress_path.read_bytes())
        except (OSError, ValueError) as exc:
            raise RuntimeError("route progress unreadable: " + str(exc)) from exc
        required = {"schema", "status", "source", "source_monotonic_ns", "route_length_m",
                    "threshold_distance_m", "cumulative_distance_m"}
        if not required <= set(value):
            raise RuntimeError("route progress omits registered fields")
        expected = self.schedule["route_transition"]
        if (value["schema"] != expected["schema"] or value["source"] != expected["source"]
                or not math.isclose(float(value["route_length_m"]), expected["route_length_m"], abs_tol=1e-9)
                or not math.isclose(float(value["threshold_distance_m"]), expected["threshold_distance_m"], abs_tol=1e-9)):
            raise RuntimeError("route progress identity/geometry mismatch")
        source_ns = int(value["source_monotonic_ns"])
        if source_ns > now:
            raise RuntimeError("route progress source is in the future")
        age_ms, status = (now - source_ns) / 1e6, str(value["status"])
        if status == "ACTIVE":
            self.seen_active = True
            frame = int(value["source_frame_id"])
            if frame < self.last_frame_id:
                raise RuntimeError("route progress frame order moved backwards")
            self.last_frame_id = frame
            if age_ms > self.schedule["max_progress_sample_age_ms"]:
                raise RuntimeError(f"route progress stale: {age_ms:.3f} ms")
            return value, status, age_ms
        if status in ("ROUTE_COMPLETED", "ROUTE_INCOMPLETE") and self.seen_active and self.latched_bad:
            return value, status, age_ms
        raise RuntimeError("route progress ended/failed before a valid latched transition: " + status)

    def next_sample(self):
        now, raw = self.monotonic_ns(), self.raw_ns()
        progress, status, age_ms = self._read(now)
        cumulative = None if progress is None else progress["cumulative_distance_m"]
        threshold = self.schedule["route_transition"]["threshold_distance_m"]
        crossed_now = bool(not self.latched_bad and cumulative is not None and float(cumulative) >= threshold)
        if crossed_now:
            self.latched_bad = True
            self.transition = dict(source_frame_id=int(progress["source_frame_id"]),
                                   crossing_monotonic_ns=int(progress["source_monotonic_ns"]),
                                   crossing_monotonic_raw_ns=int(progress["source_monotonic_raw_ns"]),
                                   crossing_wall_ns=int(progress["source_wall_ns"]),
                                   selected_monotonic_ns=now, selected_monotonic_raw_ns=raw,
                                   cumulative_distance_m=float(cumulative), threshold_distance_m=threshold,
                                   progress_sample_age_ms=age_ms)
        phase = self.bad if self.latched_bad else self.good
        row = dict(schema=SCHEMA, sequence_index=self.samples, selected_monotonic_ns=now,
                   selected_monotonic_raw_ns=raw, progress_status=status, progress_sample_age_ms=age_ms,
                   source_frame_id=(None if progress is None else progress.get("source_frame_id")),
                   cumulative_distance_m=cumulative, threshold_distance_m=threshold,
                   threshold_crossed_now=crossed_now, latched_phase=phase["phase"],
                   state_index=phase["state_index"], target_snr_db=phase["target_snr_db"])
        if crossed_now:
            row.update(crossing_monotonic_ns=int(progress["source_monotonic_ns"]),
                       crossing_monotonic_raw_ns=int(progress["source_monotonic_raw_ns"]),
                       crossing_wall_ns=int(progress["source_wall_ns"]))
        self.trace.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        self.trace.flush()
        self.samples += 1
        return int(phase["state_index"]), float(phase["target_snr_db"])

    def close(self):
        if not self.trace.closed:
            self.trace.write(json.dumps(dict(schema=SCHEMA, kind="summary", samples=self.samples,
                                             seen_active=self.seen_active, latched_bad=self.latched_bad,
                                             transition=self.transition), sort_keys=True) + "\n")
            self.trace.flush()
            self.trace.close()


def build_prepare(original, registration, progress_path, trace_path, holder):
    schedule = registration["schedule"]

    def prepare_sequence(*, campaign, profile_id):
        if profile_id != schedule["schedule_id"]:
            raise ValueError("GOOD_TO_BAD controller received a foreign profile id: " + repr(profile_id))
        original(campaign=campaign, profile_id=C.ORIGINAL_PROFILE)
        if int(campaign["network"]["prefix_samples"]) != schedule["carrier_validation_prefix_samples"]:
            raise ValueError("carrier validation prefix length drift")
        if C.schedule_definition_sha256(Path(__file__).resolve().parents[2]) != schedule["trace_sha256"]:
            raise ValueError("GOOD_TO_BAD schedule definition digest drift")
        sequence = RouteTriggeredSequence(schedule, progress_path, trace_path)
        holder.append(sequence)
        frozen = dict(profile_id=schedule["schedule_id"], trace_id=schedule["trace_id"], seed=0,
                      trace_sha256=schedule["trace_sha256"])
        return sequence, [], frozen
    return prepare_sequence


def rewrite_profile(argv):
    argv = list(argv)
    hits = [i for i, value in enumerate(argv) if value == "--profile-id"]
    if len(hits) != 1 or hits[0] + 1 >= len(argv) or argv[hits[0] + 1] != C.ORIGINAL_PROFILE:
        raise ValueError("expected exactly one --profile-id FAVORABLE_STABLE carrier argument")
    argv[hits[0] + 1] = C.SCHEDULE["schedule_id"]
    return argv


def main(argv=None):
    from rl_agent import ue_target_snr_cell_runtime_v1 as base
    from rl_agent.splitfusion_run5_live_avo_v1 import snr_live
    root = Path(__file__).resolve().parents[2]
    registration = C.channel_registration(root)
    argv = rewrite_profile(sys.argv[1:] if argv is None else argv)
    audit = _argument(argv, "--snr-audit")
    progress_path = audit.parent.parent / C.PROGRESS_SNAPSHOT_RELATIVE
    trace_path = audit.parent / "ROUTE_CHANNEL_TRANSITION.jsonl"
    original, sequences = base.prepare_sequence, []
    base.prepare_sequence = build_prepare(original, registration, progress_path, trace_path, sequences)
    print(json.dumps(dict(schema=SCHEMA, wall_ns=time.time_ns(),
                          schedule=registration["schedule"]["schedule_id"],
                          trace_id=registration["schedule"]["trace_id"],
                          progress_path=str(progress_path), trace_path=str(trace_path))), flush=True)
    try:
        return snr_live.controller_main(argv)
    finally:
        base.prepare_sequence = original
        for sequence in sequences:
            sequence.close()


if __name__ == "__main__":
    raise SystemExit(main())
