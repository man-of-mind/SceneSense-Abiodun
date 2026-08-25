#!/usr/bin/env python3
"""Measure - not assume - CARLA's radar radial-velocity sign convention.

Drives an ego forward, then in reverse, through otherwise static geometry and
compares the radar's reported radial velocity for boresight returns against the
ego's own measured forward speed. Static world geometry has no velocity of its
own, so the only relative motion is the ego's: while closing on it the true
range rate is negative, while receding it is positive.

Reports which sign CARLA uses and whether the magnitude tracks the ego speed.
Nothing is persisted into the dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import sys
from pathlib import Path
from typing import Any

AB = Path(__file__).resolve().parents[2]
if str(AB) not in sys.path:
    sys.path.insert(0, str(AB))

import numpy as np  # noqa: E402
import carla  # noqa: E402

from pole_lraspp_multimodal_fusion.pole_lraspp_multimodal_fusion.radar_fusion import (  # noqa: E402
    radar_raw_to_alt_az_depth_velocity,
)

BORESIGHT_RAD = math.radians(4.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--ticks-per-phase", type=int, default=80)
    parser.add_argument("--report-json", type=Path,
                        default=Path(__file__).resolve().parent / "radar_velocity_sign_diagnostic_v1.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0)
    world = client.load_world("Town10HD_Opt", True)
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    blueprints = world.get_blueprint_library()
    ego = None
    radar = None
    samples: dict[str, list[dict[str, float]]] = {"forward": [], "reverse": []}
    try:
        spawn_point = world.get_map().get_spawn_points()[0]
        ego = world.spawn_actor(blueprints.find("vehicle.lincoln.mkz"), spawn_point)
        radar_bp = blueprints.find("sensor.other.radar")
        radar_bp.set_attribute("range", "120.0")
        radar_bp.set_attribute("horizontal_fov", "120.0")
        radar_bp.set_attribute("vertical_fov", "30.0")
        radar_bp.set_attribute("points_per_second", "200000")
        radar_bp.set_attribute("sensor_tick", "0.0")
        radar_queue: "queue.Queue[Any]" = queue.Queue()
        radar = world.spawn_actor(
            radar_bp,
            carla.Transform(carla.Location(x=2.0, z=1.0)),
            attach_to=ego,
        )
        radar.listen(radar_queue.put)

        for phase, control in (
            ("forward", carla.VehicleControl(throttle=0.8, reverse=False)),
            ("reverse", carla.VehicleControl(throttle=0.6, reverse=True)),
        ):
            if phase == "reverse":
                ego.apply_control(carla.VehicleControl(brake=1.0))
                for _ in range(40):
                    world.tick()
                    try:
                        radar_queue.get(timeout=5.0)
                    except queue.Empty:
                        pass
            for _ in range(int(args.ticks_per_phase)):
                ego.apply_control(control)
                frame = int(world.tick())
                try:
                    measurement = radar_queue.get(timeout=10.0)
                except queue.Empty:
                    continue
                if int(measurement.frame) != frame:
                    continue
                velocity = ego.get_velocity()
                forward = ego.get_transform().get_forward_vector()
                forward_speed = (
                    velocity.x * forward.x + velocity.y * forward.y + velocity.z * forward.z
                )
                if abs(forward_speed) < 2.0:
                    continue
                detections = radar_raw_to_alt_az_depth_velocity(bytes(measurement.raw_data))
                if detections.size == 0:
                    continue
                boresight = (
                    (np.abs(detections[:, 0]) < BORESIGHT_RAD)
                    & (np.abs(detections[:, 1]) < BORESIGHT_RAD)
                )
                if not np.any(boresight):
                    continue
                selected = detections[boresight]
                samples[phase].append({
                    "ego_forward_speed_mps": float(forward_speed),
                    "median_radar_velocity_mps": float(np.median(selected[:, 3])),
                    "returns": int(selected.shape[0]),
                })
    finally:
        for actor in (radar, ego):
            if actor is None:
                continue
            try:
                if actor.type_id.startswith("sensor."):
                    actor.stop()
                actor.destroy()
            except RuntimeError:
                pass
        try:
            world.tick()
        except RuntimeError:
            pass
        world.apply_settings(original)

    report: dict[str, Any] = {
        "schema": "route_b_perception_v2.radar_velocity_sign.v1",
        "boresight_half_angle_deg": math.degrees(BORESIGHT_RAD),
        "raw_layout": "CARLA raw float32 order is (velocity, azimuth, altitude, depth); "
                      "radar_raw_to_alt_az_depth_velocity reorders to "
                      "[altitude, azimuth, depth, velocity]",
    }
    all_rows = [row for rows in samples.values() for row in rows]
    closing_rows = [r for r in all_rows if r["ego_forward_speed_mps"] > 2.0]
    receding_rows = [r for r in all_rows if r["ego_forward_speed_mps"] < -2.0]
    for label, rows in (("ego_moving_forward", closing_rows), ("ego_moving_backward", receding_rows)):
        if not rows:
            report[label] = {"samples": 0}
            continue
        speeds = np.array([r["ego_forward_speed_mps"] for r in rows])
        velocities = np.array([r["median_radar_velocity_mps"] for r in rows])
        report[label] = {
            "samples": len(rows),
            "ego_forward_speed_mps": {"min": float(speeds.min()), "mean": float(speeds.mean()),
                                      "max": float(speeds.max())},
            "radar_velocity_mps": {"min": float(velocities.min()), "mean": float(velocities.mean()),
                                   "max": float(velocities.max())},
            "fraction_negative": float(np.mean(velocities < 0.0)),
            "mean_ratio_radar_over_ego_speed": float(np.mean(velocities / speeds)),
        }

    for phase, rows in samples.items():
        if not rows:
            report[phase] = {"samples": 0}
            continue
        speeds = np.array([r["ego_forward_speed_mps"] for r in rows])
        velocities = np.array([r["median_radar_velocity_mps"] for r in rows])
        report[phase] = {
            "samples": len(rows),
            "ego_forward_speed_mps": {"min": float(speeds.min()), "mean": float(speeds.mean()),
                                      "max": float(speeds.max())},
            "radar_velocity_mps": {"min": float(velocities.min()), "mean": float(velocities.mean()),
                                   "max": float(velocities.max())},
            "fraction_negative": float(np.mean(velocities < 0.0)),
            "mean_ratio_radar_over_ego_speed": float(np.mean(velocities / speeds)),
        }

    closing = report.get("ego_moving_forward", {})
    receding = report.get("ego_moving_backward", {})
    convention = "undetermined"
    if closing.get("samples") and receding.get("samples"):
        if closing["fraction_negative"] > 0.95 and receding["fraction_negative"] < 0.05:
            convention = ("negative = closing (range decreasing); "
                          "positive = receding; value is the range rate")
        elif closing["fraction_negative"] < 0.05 and receding["fraction_negative"] > 0.95:
            convention = ("positive = closing (range decreasing); "
                          "negative = receding")
    report["convention"] = convention
    ratios = [r["median_radar_velocity_mps"] / r["ego_forward_speed_mps"]
              for r in all_rows if abs(r["ego_forward_speed_mps"]) > 2.0]
    report["mean_ratio_all_samples"] = float(np.mean(ratios)) if ratios else None
    report["magnitude_tracks_ego_speed"] = bool(
        ratios and abs(abs(float(np.mean(ratios))) - 1.0) < 0.25
    )
    report["status"] = "MEASURED" if convention != "undetermined" else "UNDETERMINED"
    Path(args.report_json).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["status"] == "MEASURED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
