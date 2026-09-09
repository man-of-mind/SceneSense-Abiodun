#!/usr/bin/env python3
"""Render an ego-following raw-versus-associated two-UE map demonstration."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, Polygon  # noqa: E402

from .render_action50_evidence_v1 import (
    DEFAULT_STATIC_GEOMETRY,
    _load_and_verify,
    _load_static_geometry,
    _sha256,
    _write_png,
)


COLORS = {"ue-a": "#00d1ff", "ue-b": "#ff9f43"}
MULTI_SOURCE_COLOR = "#6ee787"
BACKGROUND = "#080b10"
CANONICAL_SIZES = {
    "Vehicle": (4.6, 2.0),
    "Pedestrian": (0.8, 0.8),
    "Cyclist": (1.8, 0.7),
}
MAX_DISPLAY_SPEED_MPS = {"Vehicle": 25.0, "Pedestrian": 4.0, "Cyclist": 12.0}


def _canonical_class(value: object) -> str | None:
    lowered = str(value or "").strip().lower().replace("_", "")
    return {
        "vehicle": "Vehicle",
        "movingvehicle": "Vehicle",
        "parkedvehicle": "Vehicle",
        "car": "Vehicle",
        "person": "Pedestrian",
        "pedestrian": "Pedestrian",
        "walker": "Pedestrian",
        "cyclist": "Cyclist",
        "bicycle": "Cyclist",
    }.get(lowered)


def _angle_delta(first: float, second: float) -> float:
    return (second - first + 180.0) % 360.0 - 180.0


def _object_xy(item: Mapping[str, Any]) -> tuple[float, float]:
    if "world_xyz" in item:
        return float(item["world_xyz"][0]), float(item["world_xyz"][1])
    location = item.get("location")
    if not isinstance(location, Mapping):
        location = {}
    return (
        float(item.get("world_x", location.get("x"))),
        float(item.get("world_y", location.get("y"))),
    )


def _object_yaw(item: Mapping[str, Any]) -> float:
    if "yaw_deg" in item:
        return float(item["yaw_deg"])
    if "yaw_sin" in item and "yaw_cos" in item:
        return math.degrees(math.atan2(float(item["yaw_sin"]), float(item["yaw_cos"])))
    return 0.0


def _object_class(item: Mapping[str, Any]) -> str | None:
    return _canonical_class(item.get("class_name", item.get("type")))


class _LaneGuide:
    """Nearest Town10HD road tangent for display orientation only."""

    def __init__(self, static_geometry: Mapping[str, Any]) -> None:
        starts: list[tuple[float, float]] = []
        deltas: list[tuple[float, float]] = []
        for road in static_geometry["roads"]:
            points = [(float(point[0]), float(point[1])) for point in road]
            for first, second in zip(points[:-1], points[1:]):
                dx, dy = second[0] - first[0], second[1] - first[1]
                if dx * dx + dy * dy <= 1.0e-8:
                    continue
                starts.append(first)
                deltas.append((dx, dy))
        if not starts:
            raise RuntimeError("static geometry contains no usable road segments")
        self.starts = np.asarray(starts, dtype=np.float64)
        self.deltas = np.asarray(deltas, dtype=np.float64)
        self.length_squared = np.sum(self.deltas * self.deltas, axis=1)

    def display_yaw(
        self, x: float, y: float, measured_yaw: float, *, maximum_distance_m: float = 8.0
    ) -> tuple[float, bool]:
        point = np.asarray((x, y), dtype=np.float64)
        fractions = np.sum((point - self.starts) * self.deltas, axis=1) / self.length_squared
        fractions = np.clip(fractions, 0.0, 1.0)
        projections = self.starts + fractions[:, None] * self.deltas
        distance_squared = np.sum((projections - point) ** 2, axis=1)
        index = int(np.argmin(distance_squared))
        if float(distance_squared[index]) > maximum_distance_m * maximum_distance_m:
            return measured_yaw, False
        dx, dy = self.deltas[index]
        tangent = math.degrees(math.atan2(float(dy), float(dx)))
        alternatives = (tangent, tangent + 180.0)
        guided = min(alternatives, key=lambda value: abs(_angle_delta(measured_yaw, value)))
        return guided, True


def _validate_presentation_rows(
    summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, int]:
    session_id = str(summary["session_id"])
    total_raw_records = 0
    total_members = 0
    for index, row in enumerate(rows):
        poses = row.get("ego_poses")
        observations = row.get("source_observations")
        associations = row.get("associations")
        if not isinstance(poses, list) or {pose.get("ue_id") for pose in poses} != {
            "ue-a",
            "ue-b",
        }:
            raise RuntimeError(
                f"pair {index} lacks exact ue-a/ue-b ego poses; collect fresh presentation evidence"
            )
        if not isinstance(observations, list) or {
            source.get("ue_id") for source in observations
        } != {"ue-a", "ue-b"}:
            raise RuntimeError(
                f"pair {index} lacks exact ue-a/ue-b model reports; collect fresh presentation evidence"
            )
        if not isinstance(associations, list):
            raise RuntimeError(f"pair {index} lacks association identities")
        raw_identities: set[tuple[str, str, str, int, str]] = set()
        for source in observations:
            records = source.get("records")
            if not isinstance(records, list):
                raise RuntimeError(f"pair {index} has invalid source records")
            total_raw_records += len(records)
            for record_index, record in enumerate(records):
                if not isinstance(record, Mapping):
                    raise RuntimeError(f"pair {index} has a non-object model record")
                observation_id = str(
                    record.get("candidate_identity", record.get("id", f"observation_{record_index}"))
                )
                raw_identities.add(
                    (
                        str(source["ue_id"]),
                        session_id,
                        str(source["stream_id"]),
                        int(source["frame_id"]),
                        observation_id,
                    )
                )
        for association in associations:
            members = association.get("member_identities")
            if not isinstance(members, list):
                raise RuntimeError(f"pair {index} has invalid association membership")
            total_members += len(members)
            for member in members:
                if tuple(member) not in raw_identities:
                    raise RuntimeError(
                        f"pair {index} association member is absent from retained model reports"
                    )
    return {"compact_model_records": total_raw_records, "association_members": total_members}


def _clamped_velocity(
    class_name: str | None, delta_xy: np.ndarray, delta_s: float
) -> np.ndarray:
    if delta_s <= 0.0:
        return np.zeros(2, dtype=np.float64)
    velocity = delta_xy / delta_s
    speed = float(np.linalg.norm(velocity))
    maximum = MAX_DISPLAY_SPEED_MPS.get(class_name or "", 15.0)
    if speed > maximum:
        velocity *= maximum / speed
    return velocity


def _annotate_kinematics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    track_history: dict[str, tuple[int, np.ndarray, np.ndarray]] = {}
    ego_history: dict[str, tuple[int, np.ndarray, float, np.ndarray, float]] = {}
    result: list[dict[str, Any]] = []
    for source_row in rows:
        row = dict(source_row)
        tracks: list[dict[str, Any]] = []
        for source_track in source_row["tracks"]:
            track = dict(source_track)
            track_id = str(track["track_id"])
            timestamp_ns = int(track["capture_timestamp_ns"])
            position = np.asarray(track["world_xyz"][:2], dtype=np.float64)
            previous = track_history.get(track_id)
            velocity = np.zeros(2, dtype=np.float64) if previous is None else previous[2]
            if previous is not None and timestamp_ns > previous[0]:
                velocity = _clamped_velocity(
                    _object_class(track),
                    position - previous[1],
                    (timestamp_ns - previous[0]) / 1_000_000_000.0,
                )
            track["_display_velocity_xy"] = velocity.tolist()
            tracks.append(track)
            if previous is None or timestamp_ns > previous[0]:
                track_history[track_id] = (timestamp_ns, position, velocity)
        row["tracks"] = tracks

        poses: list[dict[str, Any]] = []
        for source_pose in source_row["ego_poses"]:
            pose = json.loads(json.dumps(source_pose))
            ue_id = str(pose["ue_id"])
            timestamp_ns = int(pose["capture_timestamp_ns"])
            position = np.asarray(
                (float(pose["location"]["x"]), float(pose["location"]["y"])),
                dtype=np.float64,
            )
            yaw = float(pose["rotation"]["yaw"])
            previous = ego_history.get(ue_id)
            velocity = np.zeros(2, dtype=np.float64) if previous is None else previous[3]
            yaw_rate = 0.0 if previous is None else previous[4]
            if previous is not None and timestamp_ns > previous[0]:
                delta_s = (timestamp_ns - previous[0]) / 1_000_000_000.0
                velocity = _clamped_velocity("Vehicle", position - previous[1], delta_s)
                yaw_rate = max(-60.0, min(60.0, _angle_delta(previous[2], yaw) / delta_s))
            pose["_display_velocity_xy"] = velocity.tolist()
            pose["_display_yaw_rate_dps"] = yaw_rate
            poses.append(pose)
            ego_history[ue_id] = (timestamp_ns, position, yaw, velocity, yaw_rate)
        row["ego_poses"] = poses
        row["_display_prediction_s"] = 0.0
        result.append(row)
    return result


def _predict_row(source_row: Mapping[str, Any], elapsed_s: float) -> dict[str, Any]:
    """Causal, bounded constant-velocity display prediction between measurements."""

    row = dict(source_row)
    extrapolation_s = min(max(0.0, float(elapsed_s)), 0.20)
    tracks: list[dict[str, Any]] = []
    for source_track in source_row["tracks"]:
        track = dict(source_track)
        x, y = _object_xy(track)
        velocity = track.get("_display_velocity_xy", (0.0, 0.0))
        world_xyz = list(track["world_xyz"])
        world_xyz[0] = x + float(velocity[0]) * extrapolation_s
        world_xyz[1] = y + float(velocity[1]) * extrapolation_s
        track["world_xyz"] = world_xyz
        tracks.append(track)
    row["tracks"] = tracks
    poses: list[dict[str, Any]] = []
    for source_pose in source_row["ego_poses"]:
        pose = json.loads(json.dumps(source_pose))
        velocity = pose.get("_display_velocity_xy", (0.0, 0.0))
        pose["location"]["x"] += float(velocity[0]) * extrapolation_s
        pose["location"]["y"] += float(velocity[1]) * extrapolation_s
        pose["rotation"]["yaw"] += (
            float(pose.get("_display_yaw_rate_dps", 0.0)) * extrapolation_s
        )
        poses.append(pose)
    row["ego_poses"] = poses
    row["_display_prediction_s"] = extrapolation_s
    return row


def _display_rows(
    rows: Sequence[Mapping[str, Any]], prediction_steps: int
) -> list[dict[str, Any]]:
    annotated = _annotate_kinematics(rows)
    result: list[dict[str, Any]] = []
    for index, row in enumerate(annotated[:-1]):
        result.append(row)
        interval_s = max(
            0.0,
            (int(annotated[index + 1]["snapshot_timestamp_ns"])
             - int(row["snapshot_timestamp_ns"]))
            / 1_000_000_000.0,
        )
        for step in range(1, prediction_steps):
            result.append(_predict_row(row, interval_s * step / prediction_steps))
    result.append(annotated[-1])
    return result


def _footprint(
    item: Mapping[str, Any], lane_guide: _LaneGuide
) -> tuple[list[tuple[float, float]], bool]:
    x, y = _object_xy(item)
    class_name = _object_class(item)
    length, width = CANONICAL_SIZES.get(class_name or "", (1.2, 1.2))
    yaw = _object_yaw(item)
    lane_guided = False
    if class_name == "Vehicle":
        yaw, lane_guided = lane_guide.display_yaw(x, y, yaw)
    radians = math.radians(yaw)
    cosine, sine = math.cos(radians), math.sin(radians)
    corners = []
    for local_x, local_y in (
        (-length / 2.0, -width / 2.0),
        (length / 2.0, -width / 2.0),
        (length / 2.0, width / 2.0),
        (-length / 2.0, width / 2.0),
    ):
        corners.append(
            (x + local_x * cosine - local_y * sine, y + local_x * sine + local_y * cosine)
        )
    return corners, lane_guided


def _ego_triangle(pose: Mapping[str, Any]) -> list[tuple[float, float]]:
    x, y = float(pose["location"]["x"]), float(pose["location"]["y"])
    yaw = math.radians(float(pose["rotation"]["yaw"]))
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return [
        (x + 3.0 * cosine, y + 3.0 * sine),
        (x - 2.2 * cosine - 1.2 * sine, y - 2.2 * sine + 1.2 * cosine),
        (x - 2.2 * cosine + 1.2 * sine, y - 2.2 * sine - 1.2 * cosine),
    ]


def _focus_bounds(
    row: Mapping[str, Any], radius_m: float, forward_bias: float
) -> tuple[float, float, float, float]:
    ego_a = next(pose for pose in row["ego_poses"] if pose["ue_id"] == "ue-a")
    x, y = float(ego_a["location"]["x"]), float(ego_a["location"]["y"])
    yaw = math.radians(float(ego_a["rotation"]["yaw"]))
    center_x = x + radius_m * forward_bias * math.cos(yaw)
    center_y = y + radius_m * forward_bias * math.sin(yaw)
    return (
        center_x - radius_m,
        center_x + radius_m,
        center_y - radius_m,
        center_y + radius_m,
    )


def _draw_background(
    axis: Any, static_geometry: Mapping[str, Any], bounds: tuple[float, float, float, float]
) -> None:
    x_min, x_max, y_min, y_max = bounds
    for building in static_geometry["buildings"]:
        xs = [float(point[0]) for point in building]
        ys = [float(point[1]) for point in building]
        if max(xs) < x_min or min(xs) > x_max or max(ys) < y_min or min(ys) > y_max:
            continue
        axis.add_patch(
            Polygon(
                [(float(x), float(y)) for x, y in building],
                closed=True,
                facecolor="#20242b",
                edgecolor="#333a44",
                linewidth=0.5,
                zorder=0,
            )
        )
    for road in static_geometry["roads"]:
        xs = [float(point[0]) for point in road]
        ys = [float(point[1]) for point in road]
        if max(xs) < x_min or min(xs) > x_max or max(ys) < y_min or min(ys) > y_max:
            continue
        axis.plot(xs, ys, color="#515964", linewidth=0.9, alpha=0.82, zorder=1)


def _draw_egos(axis: Any, row: Mapping[str, Any]) -> None:
    for pose in row["ego_poses"]:
        ue_id = str(pose["ue_id"])
        color = COLORS[ue_id]
        axis.add_patch(
            Polygon(
                _ego_triangle(pose),
                closed=True,
                facecolor=color,
                edgecolor="#ffffff",
                linewidth=1.2,
                alpha=0.95,
                zorder=7,
            )
        )
        axis.text(
            float(pose["location"]["x"]),
            float(pose["location"]["y"]) + 3.0,
            "A" if ue_id == "ue-a" else "B",
            color="#ffffff",
            fontsize=9,
            ha="center",
            weight="bold",
            zorder=8,
        )


def _draw_raw(axis: Any, row: Mapping[str, Any], lane_guide: _LaneGuide) -> int:
    count = 0
    for source in row["source_observations"]:
        color = COLORS[str(source["ue_id"])]
        for record in source["records"]:
            if _object_class(record) is None:
                continue
            polygon, _guided = _footprint(record, lane_guide)
            axis.add_patch(
                Polygon(
                    polygon,
                    closed=True,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=1.0,
                    alpha=0.30,
                    zorder=3,
                )
            )
            count += 1
    return count


def _track_sources(track: Mapping[str, Any]) -> set[str]:
    return {str(source[0]) for source in track.get("contributing_sources", [])}


def _track_color(track: Mapping[str, Any]) -> str:
    sources = _track_sources(track)
    if {"ue-a", "ue-b"}.issubset(sources):
        return MULTI_SOURCE_COLOR
    if "ue-a" in sources:
        return COLORS["ue-a"]
    if "ue-b" in sources:
        return COLORS["ue-b"]
    return "#cccccc"


def _draw_tracks(
    axis: Any, row: Mapping[str, Any], lane_guide: _LaneGuide
) -> dict[str, int]:
    counts = {"ue-a": 0, "ue-b": 0, "both": 0, "lane_guided_vehicles": 0}
    for track in row["tracks"]:
        if _object_class(track) is None:
            continue
        sources = _track_sources(track)
        if {"ue-a", "ue-b"}.issubset(sources):
            category = "both"
        elif "ue-a" in sources:
            category = "ue-a"
        elif "ue-b" in sources:
            category = "ue-b"
        else:
            continue
        counts[category] += 1
        polygon, lane_guided = _footprint(track, lane_guide)
        counts["lane_guided_vehicles"] += int(lane_guided)
        color = _track_color(track)
        axis.add_patch(
            Polygon(
                polygon,
                closed=True,
                facecolor=color,
                edgecolor=color,
                linewidth=2.0 if category == "both" else 1.1,
                alpha=0.48 if category == "both" else 0.30,
                zorder=4,
            )
        )
    return counts


def _style_axis(
    axis: Any, bounds: tuple[float, float, float, float], title: str
) -> None:
    x_min, x_max, y_min, y_max = bounds
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
    axis.set_aspect("equal", adjustable="box")
    axis.set_facecolor(BACKGROUND)
    axis.grid(color="#27313b", linewidth=0.5, alpha=0.55)
    axis.tick_params(colors="#9aa4af", labelsize=8)
    for spine in axis.spines.values():
        spine.set_color("#4a525c")
    axis.set_title(title, color="#f3f6f9", fontsize=12, weight="bold")


def _render(
    row: Mapping[str, Any],
    summary: Mapping[str, Any],
    static_geometry: Mapping[str, Any],
    lane_guide: _LaneGuide,
    *,
    radius_m: float,
    forward_bias: float,
) -> np.ndarray:
    bounds = _focus_bounds(row, radius_m, forward_bias)
    figure, axes = plt.subplots(1, 2, figsize=(15, 7.2), dpi=100)
    figure.patch.set_facecolor(BACKGROUND)
    for axis in axes:
        _draw_background(axis, static_geometry, bounds)
        _draw_egos(axis, row)
    raw_count = _draw_raw(axes[0], row, lane_guide)
    counts = _draw_tracks(axes[1], row, lane_guide)
    _style_axis(axes[0], bounds, "BEFORE: two raw UE reports\n(no cross-UE association)")
    _style_axis(axes[1], bounds, "AFTER: cooperative spatial map\n(one track per associated object)")
    axes[0].text(
        0.02,
        0.02,
        f"raw model detections: {raw_count}\ncyan = UE-A   orange = UE-B",
        transform=axes[0].transAxes,
        color="#f3f6f9",
        fontsize=8.5,
        va="bottom",
        bbox={"facecolor": "#111820", "edgecolor": "#56616c", "alpha": 0.88},
    )
    axes[1].text(
        0.02,
        0.02,
        f"A only: {counts['ue-a']}   B only: {counts['ue-b']}   A+B: {counts['both']}\n"
        f"lane-guided vehicle yaw: {counts['lane_guided_vehicles']}",
        transform=axes[1].transAxes,
        color="#f3f6f9",
        fontsize=8.5,
        va="bottom",
        bbox={"facecolor": "#111820", "edgecolor": "#56616c", "alpha": 0.88},
    )
    profile = summary["profile"]
    prediction_ms = 1000.0 * float(row.get("_display_prediction_s", 0.0))
    figure.suptitle(
        "Two live UEs → one conservative spatial map\n"
        f"action 50: AE64 / UINT4 / q={float(profile['q']):.2f}  ·  "
        f"pair {int(row['pair_index']) + 1}/20  ·  display prediction +{prediction_ms:.0f} ms",
        color="#f3f6f9",
        fontsize=14,
        weight="bold",
    )
    figure.legend(
        handles=[
            Patch(facecolor=COLORS["ue-a"], label="UE-A only"),
            Patch(facecolor=COLORS["ue-b"], label="UE-B only"),
            Patch(facecolor=MULTI_SOURCE_COLOR, label="UE-A + UE-B associated"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=3,
        framealpha=0.85,
        fontsize=9,
    )
    figure.text(
        0.5,
        0.012,
        "Model detections only—no CARLA actor ground truth. Lane tangent changes display yaw, never XY. "
        "Motion prediction is visualization-only; measured map state is unchanged.",
        color="#cbd3dc",
        fontsize=8.5,
        ha="center",
    )
    figure.tight_layout(rect=(0.0, 0.13, 1.0, 0.91))
    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba())
    rgb = np.ascontiguousarray(rgba[:, :, :3]).copy()
    plt.close(figure)
    return rgb


def _recommended_index(rows: Sequence[Mapping[str, Any]]) -> int:
    return max(
        range(len(rows)),
        key=lambda index: (
            int(rows[index]["multi_source_associations"]),
            -abs(index - (len(rows) // 2)),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--prediction-steps", type=int, default=8)
    parser.add_argument("--focus-radius-m", type=float, default=40.0)
    parser.add_argument("--forward-bias", type=float, default=0.35)
    parser.add_argument("--static-geometry", default=str(DEFAULT_STATIC_GEOMETRY))
    args = parser.parse_args()
    if args.fps <= 0.0 or args.prediction_steps < 1 or args.focus_radius_m <= 0.0:
        parser.error("--fps, --prediction-steps and --focus-radius-m must be positive")
    if not 0.0 <= args.forward_bias <= 0.75:
        parser.error("--forward-bias must lie in [0, 0.75]")

    run_dir = Path(args.run_dir).resolve(strict=True)
    static_path = Path(args.static_geometry).resolve(strict=True)
    summary, source_rows = _load_and_verify(run_dir)
    evidence_counts = _validate_presentation_rows(summary, source_rows)
    static_geometry = _load_static_geometry(static_path)
    lane_guide = _LaneGuide(static_geometry)
    display_rows = _display_rows(source_rows, int(args.prediction_steps))
    recommended_index = _recommended_index(source_rows)
    output_dir = Path(args.output).resolve()
    if output_dir.exists():
        raise RuntimeError(f"create-only rendering output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    recommended_still_path = output_dir / "two_ue_before_after_recommended.png"
    video_path = output_dir / "two_ue_before_after_ego_follow.mp4"
    writer: cv2.VideoWriter | None = None
    rendered_frames = 0
    still_written = False
    try:
        for row in display_rows:
            rgb = _render(
                row,
                summary,
                static_geometry,
                lane_guide,
                radius_m=float(args.focus_radius_m),
                forward_bias=float(args.forward_bias),
            )
            if writer is None:
                height, width = rgb.shape[:2]
                writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    float(args.fps),
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError("OpenCV could not open the MP4 video writer")
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            rendered_frames += 1
            if (
                not still_written
                and int(row["pair_index"]) == recommended_index
                and float(row.get("_display_prediction_s", 0.0)) == 0.0
            ):
                _write_png(recommended_still_path, rgb)
                still_written = True
    finally:
        if writer is not None:
            writer.release()
    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise RuntimeError("visualization MP4 was not created")
    if not recommended_still_path.is_file():
        raise RuntimeError("recommended visualization still was not created")

    manifest = {
        "schema": "scenesense.live_two_ue_action50_before_after_visualization.v2",
        "source_manifest_sha256": _sha256(run_dir / "manifest.json"),
        "source_summary_sha256": _sha256(run_dir / "SUMMARY.json"),
        "source_snapshots_sha256": _sha256(run_dir / "snapshots.jsonl"),
        "source_terminal_verified": True,
        "source_static_geometry_sha256": _sha256(static_path),
        "source_static_map": static_geometry["map_name"],
        "evidence_counts": evidence_counts,
        "measured_keyframes": len(source_rows),
        "rendered_video_frames": rendered_frames,
        "display_fps": float(args.fps),
        "prediction_steps_between_measurements": int(args.prediction_steps),
        "focus_ue": "ue-a",
        "focus_radius_m": float(args.focus_radius_m),
        "focus_forward_bias": float(args.forward_bias),
        "before_panel": "TWO_UNASSOCIATED_MODEL_REPORTS",
        "after_panel": "ONE_TRACK_PER_ASSOCIATED_OBJECT_WITH_UNMATCHED_TRACKS_RETAINED",
        "colors": {
            "ue_a_only": COLORS["ue-a"],
            "ue_b_only": COLORS["ue-b"],
            "ue_a_and_ue_b": MULTI_SOURCE_COLOR,
        },
        "motion_display_model": {
            "kind": "CAUSAL_BOUNDED_CONSTANT_VELOCITY_ZERO_LEARNING",
            "maximum_extrapolation_s": 0.20,
            "measured_state_corrects_next_display_anchor": True,
            "changes_scientific_map_state": False,
        },
        "lane_geometry_use": {
            "vehicle_display_yaw_guided_by_nearest_road_tangent": True,
            "maximum_tangent_distance_m": 8.0,
            "world_xy_snapped_to_lane": False,
            "used_in_scientific_association": False,
        },
        "mp4": {
            "name": video_path.name,
            "bytes": video_path.stat().st_size,
            "sha256": _sha256(video_path),
        },
        "recommended_still": {
            "name": recommended_still_path.name,
            "pair_index": recommended_index,
            "bytes": recommended_still_path.stat().st_size,
            "sha256": _sha256(recommended_still_path),
        },
        "visualization_only": True,
        "scientific_measurements_modified": False,
        "carla_actor_ground_truth_used": False,
        "raw_sensor_frames_used": 0,
    }
    manifest_path = output_dir / "visualization_manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
