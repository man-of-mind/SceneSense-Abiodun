#!/usr/bin/env python3
"""Bounded offline legacy-vs-fast radar rasterizer comparison.

Replays real saved radar point clouds from a collected episode through both
``rasterize_radar_channels`` (legacy) and ``rasterize_radar_channels_fast`` and
compares tensor values, differing element count, maximum absolute difference and
runtime. Offline only: no CARLA, no model, nothing written into the episode.

Decision rule: adopt the fast rasterizer only if outputs are exact on every
tested frame, or every difference is confined to the already-documented
equal-magnitude signed-velocity tie in the velocity channel (index 2).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

AB = Path(__file__).resolve().parents[2]
if str(AB) not in sys.path:
    sys.path.insert(0, str(AB))

import numpy as np  # noqa: E402

from pole_lraspp_multimodal_fusion.pole_lraspp_multimodal_fusion.radar_fusion import (  # noqa: E402
    rasterize_radar_channels,
    rasterize_radar_channels_fast,
)

CHANNEL_NAMES = ("occupancy", "inverse_range", "radial_velocity", "stationary_age")
VELOCITY_CHANNEL = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=432)
    parser.add_argument("--point-radius-px", type=int, default=4)
    parser.add_argument("--max-range-m", type=float, default=120.0)
    parser.add_argument("--max-abs-velocity-mps", type=float, default=20.0)
    parser.add_argument("--parked-threshold-s", type=float, default=5.0)
    parser.add_argument("--report-json", type=Path,
                        default=Path(__file__).resolve().parent / "rasterizer_comparison_v1.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    point_files = sorted((Path(args.episode_dir) / "radar_points").glob("*.npz"))
    if not point_files:
        print(f"no radar point clouds under {args.episode_dir}", file=sys.stderr)
        return 2
    # Evenly spaced across the episode so the sample is representative of the
    # whole route rather than one stretch of it.
    if len(point_files) > int(args.frames):
        step = len(point_files) / float(args.frames)
        point_files = [point_files[int(i * step)] for i in range(int(args.frames))]

    rows: list[dict[str, Any]] = []
    legacy_seconds = 0.0
    fast_seconds = 0.0
    for path in point_files:
        with np.load(path) as payload:
            kwargs = dict(
                width=int(args.width), height=int(args.height),
                u=payload["u"], v=payload["v"],
                depth_m=payload["camera_depth_m"],
                velocity_mps=payload["velocity_mps"],
                stationary_age_s=payload["stationary_age_s"],
                valid_mask=payload["valid_projection"].astype(bool),
                max_range_m=float(args.max_range_m),
                max_abs_velocity_mps=float(args.max_abs_velocity_mps),
                parked_threshold_s=float(args.parked_threshold_s),
                point_radius_px=int(args.point_radius_px),
            )
            points = int(payload["u"].shape[0])

        started = time.perf_counter()
        legacy = rasterize_radar_channels(**kwargs)
        legacy_s = time.perf_counter() - started
        started = time.perf_counter()
        fast = rasterize_radar_channels_fast(**kwargs)
        fast_s = time.perf_counter() - started
        legacy_seconds += legacy_s
        fast_seconds += fast_s

        differing = legacy != fast
        row: dict[str, Any] = {
            "sample": path.stem,
            "points": points,
            "legacy_s": legacy_s,
            "fast_s": fast_s,
            "differing_elements": int(np.count_nonzero(differing)),
            "max_abs_difference": float(np.max(np.abs(legacy - fast))) if legacy.size else 0.0,
            "per_channel_differing": {
                name: int(np.count_nonzero(differing[index]))
                for index, name in enumerate(CHANNEL_NAMES)
            },
        }
        # A tolerated difference is a velocity-channel pixel where the two
        # implementations picked opposite signs of the SAME magnitude.
        non_velocity = int(np.count_nonzero(
            np.delete(differing, VELOCITY_CHANNEL, axis=0)
        ))
        velocity_diff = differing[VELOCITY_CHANNEL]
        tie_only = bool(np.all(
            np.isclose(np.abs(legacy[VELOCITY_CHANNEL][velocity_diff]),
                       np.abs(fast[VELOCITY_CHANNEL][velocity_diff]), rtol=0.0, atol=1e-6)
        )) if np.any(velocity_diff) else True
        row["non_velocity_differing_elements"] = non_velocity
        row["velocity_differences_are_equal_magnitude_ties"] = tie_only
        rows.append(row)

    exact = all(r["differing_elements"] == 0 for r in rows)
    tie_only_all = all(
        r["non_velocity_differing_elements"] == 0
        and r["velocity_differences_are_equal_magnitude_ties"]
        for r in rows
    )
    decision = "fast" if (exact or tie_only_all) else "legacy"
    report = {
        "schema": "route_b_perception_v2.rasterizer_comparison.v1",
        "episode_dir": str(Path(args.episode_dir).resolve()),
        "frames_compared": len(rows),
        "points_per_frame": {
            "min": min(r["points"] for r in rows),
            "mean": sum(r["points"] for r in rows) / len(rows),
            "max": max(r["points"] for r in rows),
        },
        "exact_on_every_frame": exact,
        "differences_only_equal_magnitude_velocity_ties": tie_only_all,
        "total_differing_elements": sum(r["differing_elements"] for r in rows),
        "max_abs_difference": max(r["max_abs_difference"] for r in rows),
        "runtime_s": {
            "legacy_total": legacy_seconds,
            "fast_total": fast_seconds,
            "legacy_mean_per_frame": legacy_seconds / len(rows),
            "fast_mean_per_frame": fast_seconds / len(rows),
            "speedup": (legacy_seconds / fast_seconds) if fast_seconds > 0 else None,
        },
        "decision": decision,
        "decision_rule": "fast only if exact everywhere, or every difference is an "
                         "equal-magnitude signed-velocity tie in the velocity channel",
        "per_frame": rows,
    }
    Path(args.report_json).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    printable = {k: v for k, v in report.items() if k != "per_frame"}
    print(json.dumps(printable, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
