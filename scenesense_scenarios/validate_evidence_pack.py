#!/usr/bin/env python3
"""Validate a SceneSense scenario evidence pack.

This is a lightweight gate for canonical occlusion runs. It checks that the
run folder contains event traces, actor ground-truth evidence, sampled camera
frames, and a target pedestrian that actually started moving along the planned
crossing vector.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a SceneSense evidence-pack run folder.")
    parser.add_argument("run_dir", help="Scenario run directory containing scenario_event_summary.json.")
    parser.add_argument(
        "--min-progress-ratio",
        type=float,
        default=0.45,
        help="Minimum target crossing progress ratio required when a crossing vector is available.",
    )
    parser.add_argument(
        "--require-danger",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require target collision or near-miss danger event.",
    )
    parser.add_argument(
        "--require-collision",
        action="store_true",
        help="Require an actual ego collision with the target actor, not just a near miss.",
    )
    parser.add_argument(
        "--require-evidence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require evidence/evidence_summary.json and non-empty actor trace rows.",
    )
    parser.add_argument(
        "--forbid-ai-controller",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if the target used CARLA ai_controller navigation for the crossing.",
    )
    parser.add_argument(
        "--min-ego-frames",
        type=int,
        default=1,
        help="Minimum buffered ego RGB frames required in the evidence pack.",
    )
    parser.add_argument(
        "--min-helper-frames",
        type=int,
        default=-1,
        help=(
            "Minimum helper RGB frames required. Default -1 means require one helper "
            "frame only when the scenario layout enabled a helper vehicle."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return data


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def location_xy(row: object) -> Optional[Tuple[float, float]]:
    if not isinstance(row, dict):
        return None
    x = as_float(row.get("x"))
    y = as_float(row.get("y"))
    if x is None or y is None:
        return None
    return x, y


def horizontal_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def trace_target_points(rows: Iterable[Dict[str, str]]) -> Iterable[Tuple[float, float]]:
    for row in rows:
        x = as_float(row.get("target_x"))
        y = as_float(row.get("target_y"))
        if x is None or y is None:
            continue
        yield x, y


def max_trace_progress(
    rows: Sequence[Dict[str, str]],
    start_xy: Optional[Tuple[float, float]],
    end_xy: Optional[Tuple[float, float]],
) -> Tuple[Optional[float], Optional[float]]:
    explicit_progress = [
        value
        for value in (as_float(row.get("target_crossing_progress_ratio")) for row in rows)
        if value is not None
    ]
    explicit_distance_to_end = [
        value
        for value in (as_float(row.get("target_crossing_distance_to_end_m")) for row in rows)
        if value is not None
    ]
    max_progress = max(explicit_progress) if explicit_progress else None
    min_distance_to_end = min(explicit_distance_to_end) if explicit_distance_to_end else None
    if max_progress is not None:
        return max_progress, min_distance_to_end

    if start_xy is None or end_xy is None:
        return None, min_distance_to_end
    initial_distance = horizontal_distance(start_xy, end_xy)
    if initial_distance <= 0.05:
        return None, min_distance_to_end

    computed_progress: List[float] = []
    computed_distance_to_end: List[float] = []
    for point in trace_target_points(rows):
        distance_to_end = horizontal_distance(point, end_xy)
        computed_distance_to_end.append(distance_to_end)
        computed_progress.append(max(0.0, min(1.0, (initial_distance - distance_to_end) / initial_distance)))
    if computed_progress:
        max_progress = max(computed_progress)
    if computed_distance_to_end:
        min_distance_to_end = min(computed_distance_to_end)
    return max_progress, min_distance_to_end


def camera_frames(evidence_summary: Dict[str, object], label: str) -> int:
    exports = evidence_summary.get("camera_exports")
    if not isinstance(exports, list):
        return 0
    for export in exports:
        if not isinstance(export, dict):
            continue
        if str(export.get("camera_label", "")).lower() == label.lower():
            return int(as_float(export.get("frames_written")) or 0)
    return 0


def first_started_row(rows: Sequence[Dict[str, str]]) -> Optional[Dict[str, str]]:
    for row in rows:
        if as_bool(row.get("target_started")):
            return row
    return None


def first_target_collision_elapsed_s(summary: Dict[str, object]) -> Optional[float]:
    events = summary.get("collision_events")
    if not isinstance(events, list):
        return None
    target_actor_id = as_float(summary.get("target_actor_id"))
    for event in events:
        if not isinstance(event, dict):
            continue
        if target_actor_id is not None:
            other_actor_id = as_float(event.get("other_actor_id"))
            if other_actor_id is None or int(other_actor_id) != int(target_actor_id):
                continue
        elapsed_s = as_float(event.get("elapsed_s"))
        if elapsed_s is not None:
            return elapsed_s
    return None


def print_metric(name: str, value: object) -> None:
    print(f"{name}={value}")


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    failures: List[str] = []

    summary_path = run_dir / "scenario_event_summary.json"
    trace_path = run_dir / "scenario_event_trace.csv"
    manifest_path = run_dir / "scenario_manifest.json"
    evidence_summary_path = run_dir / "evidence" / "evidence_summary.json"

    if not summary_path.exists():
        print(f"Missing {summary_path}", file=sys.stderr)
        return 2
    if not trace_path.exists():
        print(f"Missing {trace_path}", file=sys.stderr)
        return 2

    summary = load_json(summary_path)
    trace_rows = load_csv(trace_path)
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    layout = manifest.get("occlusion_layout") if isinstance(manifest.get("occlusion_layout"), dict) else {}

    evidence_summary: Dict[str, object] = {}
    if evidence_summary_path.exists():
        evidence_summary = load_json(evidence_summary_path)
    elif args.require_evidence:
        failures.append("missing evidence/evidence_summary.json")

    target_motion_mode = str(summary.get("target_motion_mode", ""))
    target_started = as_bool(summary.get("target_started"))
    target_collision_count = int(as_float(summary.get("target_collision_count")) or 0)
    target_danger_event = as_bool(summary.get("target_danger_event"))
    min_target_distance_m = as_float(summary.get("min_target_distance_m"))
    target_crossing_trigger_min_ego_speed_mps = as_float(
        summary.get("target_crossing_trigger_min_ego_speed_mps")
    )

    target_start_xy = location_xy(layout.get("target_crossing_start_location"))
    target_end_xy = location_xy(layout.get("target_crossing_end_location"))
    max_progress, min_distance_to_end = max_trace_progress(trace_rows, target_start_xy, target_end_xy)
    started_row = first_started_row(trace_rows)
    target_started_at_s = as_float(summary.get("target_started_at_s"))
    first_target_collision_s = first_target_collision_elapsed_s(summary)
    target_start_to_first_collision_s = None
    if target_started_at_s is not None and first_target_collision_s is not None:
        target_start_to_first_collision_s = first_target_collision_s - target_started_at_s

    actor_trace_rows = int(as_float(evidence_summary.get("actor_trace_rows")) or 0)
    event_window_rows = int(as_float(evidence_summary.get("event_trace_window_rows")) or 0)
    ego_frames = camera_frames(evidence_summary, "ego")
    helper_frames = camera_frames(evidence_summary, "helper")
    helper_enabled = bool(layout.get("helper_vehicle_enabled")) if isinstance(layout, dict) else False
    min_helper_frames = int(args.min_helper_frames)
    if min_helper_frames < 0:
        min_helper_frames = 1 if helper_enabled else 0

    if not target_started:
        failures.append("target crossing never started")
    if args.forbid_ai_controller and target_motion_mode == "ai_controller":
        failures.append("target used ai_controller navigation instead of controlled crossing")
    if args.require_danger and not (target_danger_event or target_collision_count > 0):
        failures.append("no target collision or near-miss danger event was recorded")
    if args.require_collision and target_collision_count <= 0:
        failures.append("no target collision was recorded")
    if max_progress is None:
        failures.append("target crossing progress could not be computed")
    elif max_progress < float(args.min_progress_ratio):
        failures.append(
            f"target max crossing progress {max_progress:.3f} < required {float(args.min_progress_ratio):.3f}"
        )
    if args.require_evidence and actor_trace_rows <= 0:
        failures.append("evidence actor trace has no rows")
    if args.require_evidence and event_window_rows <= 0:
        failures.append("evidence event-window trace has no rows")
    if int(args.min_ego_frames) > 0 and ego_frames < int(args.min_ego_frames):
        failures.append(f"ego RGB evidence frames {ego_frames} < required {int(args.min_ego_frames)}")
    if min_helper_frames > 0 and helper_frames < min_helper_frames:
        failures.append(f"helper RGB evidence frames {helper_frames} < required {min_helper_frames}")

    print_metric("run_dir", run_dir)
    print_metric("target_motion_mode", target_motion_mode)
    print_metric("target_started", target_started)
    print_metric("target_start_reason", summary.get("target_start_reason", ""))
    print_metric(
        "target_crossing_trigger_min_ego_speed_mps",
        ""
        if target_crossing_trigger_min_ego_speed_mps is None
        else f"{target_crossing_trigger_min_ego_speed_mps:.3f}",
    )
    print_metric("target_started_at_s", "" if target_started_at_s is None else f"{target_started_at_s:.3f}")
    print_metric(
        "first_target_collision_s",
        "" if first_target_collision_s is None else f"{first_target_collision_s:.3f}",
    )
    print_metric(
        "target_start_to_first_collision_s",
        "" if target_start_to_first_collision_s is None else f"{target_start_to_first_collision_s:.3f}",
    )
    print_metric("target_collision_count", target_collision_count)
    print_metric("target_danger_event", target_danger_event)
    print_metric("min_target_distance_m", min_target_distance_m)
    print_metric("target_max_progress_ratio", "" if max_progress is None else f"{max_progress:.3f}")
    print_metric("target_min_distance_to_end_m", "" if min_distance_to_end is None else f"{min_distance_to_end:.3f}")
    if started_row is not None:
        print_metric(
            "target_start_ego_route_distance_to_trigger_m",
            started_row.get("ego_route_distance_to_trigger_m", ""),
        )
        print_metric(
            "target_start_ego_route_progress_m",
            started_row.get("ego_route_progress_m", ""),
        )
        print_metric(
            "target_start_ego_speed_mps",
            started_row.get("ego_speed_mps", ""),
        )
        print_metric(
            "target_start_ego_ttc_to_conflict_s",
            started_row.get("ego_ttc_to_conflict_s", ""),
        )
        print_metric(
            "target_start_progress_ratio",
            started_row.get("target_crossing_progress_ratio", ""),
        )
    print_metric("evidence_actor_trace_rows", actor_trace_rows)
    print_metric("evidence_event_window_rows", event_window_rows)
    print_metric("ego_rgb_frames", ego_frames)
    print_metric("helper_rgb_frames", helper_frames)

    if failures:
        print("validation=FAIL")
        for failure in failures:
            print(f"failure={failure}")
        return 1

    print("validation=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
