#!/usr/bin/env python3
"""Render the hash-verified two-UE/action-50 evidence without CARLA or CUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, Polygon  # noqa: E402


COMPLETE_TERMINAL = "SPLITFUSION_LIVE_TWO_UE_ACTION50_DEMO_COMPLETE"
COLORS = {"ue-a": "#00d1ff", "ue-b": "#ff9f43"}
MULTI_SOURCE_COLOR = "#7CFC80"
DEFAULT_STATIC_GEOMETRY = (
    Path(__file__).resolve().parents[1]
    / "recordings"
    / "two_ego_live.jsonl.static.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_and_verify(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = run_dir / name
        if not path.is_file():
            raise RuntimeError(f"source evidence file is missing: {name}")
        if path.stat().st_size != int(expected["bytes"]):
            raise RuntimeError(f"source evidence size mismatch: {name}")
        if _sha256(path) != str(expected["sha256"]):
            raise RuntimeError(f"source evidence SHA-256 mismatch: {name}")
    terminal = run_dir / COMPLETE_TERMINAL
    expected_terminal = f"{COMPLETE_TERMINAL} {manifest['summary_sha256']}"
    if not terminal.is_file() or terminal.read_text(encoding="utf-8").strip() != expected_terminal:
        raise RuntimeError("source completion terminal is absent or invalid")
    summary = json.loads((run_dir / "SUMMARY.json").read_text(encoding="utf-8"))
    if summary.get("status") != "COMPLETE" or int(summary.get("paired_frames", -1)) != 20:
        raise RuntimeError("source summary is not the completed 20-pair demonstration")
    rows = [
        json.loads(line)
        for line in (run_dir / "snapshots.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 20 or [row.get("pair_index") for row in rows] != list(range(20)):
        raise RuntimeError("snapshot inventory is not the exact contiguous 20-pair run")
    return summary, rows


def _bounds(rows: list[Mapping[str, Any]]) -> tuple[float, float, float, float]:
    points = [track["world_xyz"][:2] for row in rows for track in row["tracks"]]
    if not points:
        raise RuntimeError("source evidence contains no map tracks")
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    margin = 6.0
    return min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin


def _load_static_geometry(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("map_name") != "Carla/Maps/Town10HD_Opt":
        raise RuntimeError("static geometry is not registered Town10HD_Opt")
    if not isinstance(document.get("roads"), list) or not isinstance(
        document.get("buildings"), list
    ):
        raise RuntimeError("static geometry has an invalid schema")
    return document


def _interpolate_angle(first: float, second: float, fraction: float) -> float:
    delta = (second - first + 180.0) % 360.0 - 180.0
    return first + fraction * delta


def _interpolate_rows(
    first: Mapping[str, Any], second: Mapping[str, Any], fraction: float
) -> dict[str, Any]:
    """Interpolate stable tracks and gently fade inventory transitions."""

    first_by_id = {str(track["track_id"]): track for track in first["tracks"]}
    second_by_id = {str(track["track_id"]): track for track in second["tracks"]}
    tracks = []
    for track_id in sorted(first_by_id.keys() | second_by_id.keys()):
        before = first_by_id.get(track_id)
        after = second_by_id.get(track_id)
        base = dict(after if after is not None else before)
        if before is not None and after is not None:
            base["world_xyz"] = [
                (1.0 - fraction) * float(a) + fraction * float(b)
                for a, b in zip(before["world_xyz"], after["world_xyz"])
            ]
            base["size_lwh"] = [
                (1.0 - fraction) * float(a) + fraction * float(b)
                for a, b in zip(before["size_lwh"], after["size_lwh"])
            ]
            base["yaw_deg"] = _interpolate_angle(
                float(before.get("yaw_deg", 0.0)),
                float(after.get("yaw_deg", 0.0)),
                fraction,
            )
            base["_display_alpha"] = 1.0
        elif after is not None:
            base["_display_alpha"] = fraction
        else:
            base["_display_alpha"] = 1.0 - fraction
        if float(base["_display_alpha"]) > 0.0:
            tracks.append(base)
    row = dict(second)
    row["tracks"] = tracks
    row["_display_fraction"] = fraction
    return row


def _footprint(track: Mapping[str, Any]) -> list[tuple[float, float]]:
    x, y = (float(value) for value in track["world_xyz"][:2])
    length, width = (max(0.2, float(value)) for value in track["size_lwh"][:2])
    yaw = math.radians(float(track.get("yaw_deg", 0.0)))
    cosine, sine = math.cos(yaw), math.sin(yaw)
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
    return corners


def _render(
    row: Mapping[str, Any],
    summary: Mapping[str, Any],
    bounds: tuple[float, float, float, float],
    static_geometry: Mapping[str, Any],
) -> np.ndarray:
    figure, axis = plt.subplots(figsize=(9, 7), dpi=120)
    figure.patch.set_facecolor("#080b10")
    axis.set_facecolor("#080b10")
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
                linewidth=0.6,
                zorder=0,
            )
        )
    for road in static_geometry["roads"]:
        xs = [float(point[0]) for point in road]
        ys = [float(point[1]) for point in road]
        if max(xs) < x_min or min(xs) > x_max or max(ys) < y_min or min(ys) > y_max:
            continue
        axis.plot(xs, ys, color="#4a525c", linewidth=1.0, alpha=0.8, zorder=1)
    multi_source_tracks = 0
    for track in row["tracks"]:
        sources = {str(source[0]) for source in track.get("contributing_sources", [])}
        selected_ue = str(track.get("selected_ue_id", ""))
        multi_source = len(sources) >= 2
        multi_source_tracks += int(multi_source)
        color = MULTI_SOURCE_COLOR if multi_source else COLORS.get(selected_ue, "#cccccc")
        display_alpha = float(track.get("_display_alpha", 1.0))
        polygon = Polygon(
            _footprint(track),
            closed=True,
            facecolor=color,
            edgecolor=color,
            linewidth=2.2 if multi_source else 1.0,
            alpha=(0.42 if multi_source else 0.22) * display_alpha,
            zorder=2,
        )
        axis.add_patch(polygon)
        x, y = (float(value) for value in track["world_xyz"][:2])
        axis.scatter(
            x,
            y,
            s=18 if multi_source else 8,
            color=color,
            alpha=display_alpha,
            zorder=3,
        )
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(color="#27313b", linewidth=0.6, alpha=0.65)
    axis.tick_params(colors="#9aa4af")
    for spine in axis.spines.values():
        spine.set_color("#4a525c")
    axis.set_xlabel("CARLA world X (m)", color="#cbd3dc")
    axis.set_ylabel("CARLA world Y (m)", color="#cbd3dc")
    profile = summary["profile"]
    axis.set_title(
        "Two-UE cooperative spatial map — learned action 50\n"
        f"AE64 / UINT4 / q={profile['q']:.2f}  ·  pair {int(row['pair_index']) + 1}/20  "
        f"·  frame {row['frame_id']}",
        color="#f3f6f9",
        fontsize=13,
    )
    axis.legend(
        handles=[
            Patch(facecolor=COLORS["ue-a"], label="UE-A selected observation"),
            Patch(facecolor=COLORS["ue-b"], label="UE-B selected observation"),
            Patch(facecolor=MULTI_SOURCE_COLOR, label="Track supported by both UEs"),
        ],
        loc="upper right",
        framealpha=0.82,
        fontsize=9,
    )
    axis.text(
        0.01,
        0.01,
        f"active tracks: {len(row['tracks'])}   "
        f"two-UE tracks: {multi_source_tracks}   "
        f"current two-source associations: {row['multi_source_associations']}\n"
        "Class/time/distance/size gating → Hungarian assignment → freshest-source state\n"
        "No position averaging; provenance retained for every contributor",
        transform=axis.transAxes,
        color="#f3f6f9",
        fontsize=8.5,
        va="bottom",
        bbox={"facecolor": "#111820", "edgecolor": "#56616c", "alpha": 0.88},
    )
    figure.tight_layout()
    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba())
    rgb = np.ascontiguousarray(rgba[:, :, :3]).copy()
    plt.close(figure)
    return rgb


def _write_png(path: Path, rgb: np.ndarray) -> None:
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to write visualization PNG: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--interpolation-steps", type=int, default=5)
    parser.add_argument("--static-geometry", default=str(DEFAULT_STATIC_GEOMETRY))
    args = parser.parse_args()
    if args.fps <= 0.0 or args.interpolation_steps < 1:
        parser.error("--fps and --interpolation-steps must be positive")
    run_dir = Path(args.run_dir).resolve(strict=True)
    static_path = Path(args.static_geometry).resolve(strict=True)
    output_dir = Path(args.output).resolve()
    if output_dir.exists():
        raise RuntimeError(f"create-only rendering output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    summary, rows = _load_and_verify(run_dir)
    static_geometry = _load_static_geometry(static_path)
    bounds = _bounds(rows)
    recommended_index = max(
        range(len(rows)), key=lambda index: int(rows[index]["multi_source_associations"])
    )
    key_frames: dict[int, Path] = {}
    video_path = output_dir / "two_ue_action50_cooperative_map.mp4"
    writer: cv2.VideoWriter | None = None
    rendered_frames = 0
    display_rows: list[Mapping[str, Any]] = [rows[0]]
    for first, second in zip(rows[:-1], rows[1:]):
        display_rows.extend(
            _interpolate_rows(first, second, step / args.interpolation_steps)
            for step in range(1, args.interpolation_steps + 1)
        )
    try:
        for display_row in display_rows:
            rgb = _render(display_row, summary, bounds, static_geometry)
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
            fraction = float(display_row.get("_display_fraction", 1.0))
            if fraction == 1.0:
                pair_index = int(display_row["pair_index"])
                path = output_dir / f"pair_{pair_index:02d}.png"
                _write_png(path, rgb)
                key_frames[pair_index] = path
    finally:
        if writer is not None:
            writer.release()
    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise RuntimeError("visualization MP4 was not created")
    still = key_frames[recommended_index]
    evidence = {
        "schema": "scenesense.live_two_ue_action50_visualization.v1",
        "source_manifest_sha256": _sha256(run_dir / "manifest.json"),
        "source_summary_sha256": _sha256(run_dir / "SUMMARY.json"),
        "source_terminal_verified": True,
        "source_static_geometry_sha256": _sha256(static_path),
        "source_static_map": static_geometry["map_name"],
        "measured_key_frames": len(key_frames),
        "rendered_video_frames": rendered_frames,
        "display_fps": float(args.fps),
        "interpolation_steps": int(args.interpolation_steps),
        "mp4": {
            "name": video_path.name,
            "bytes": video_path.stat().st_size,
            "sha256": _sha256(video_path),
        },
        "recommended_still": {
            "name": still.name,
            "bytes": still.stat().st_size,
            "sha256": _sha256(still),
        },
        "visualization_only": True,
        "interpolation_changes_scientific_track_inventory": False,
        "track_appearance_and_expiry_are_display_faded": True,
        "scientific_measurements_modified": False,
        "raw_sensor_frames_used": 0,
    }
    with (output_dir / "visualization_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(evidence, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
