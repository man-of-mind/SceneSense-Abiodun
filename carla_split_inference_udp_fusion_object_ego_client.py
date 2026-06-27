#!/usr/bin/env python3
"""Parked ego-vehicle RGB+radar fusion client.

This is a small entrypoint for the ego-mounted version of the SceneSense fusion
runtime. It reuses the pole/OAI split implementation and selects the
`ego_vehicle` sensor platform by default, so the model/UDP/metrics path stays
identical while the RGB + radar sensors are attached to a parked vehicle.
"""

from __future__ import annotations

import sys

import carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai as fusion_runtime


def _has_option(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:])


def main() -> None:
    if not _has_option("--sensor-platform"):
        sys.argv[1:1] = ["--sensor-platform", "ego_vehicle"]
    if not _has_option("--spatial-map-stream-id"):
        sys.argv.extend(["--spatial-map-stream-id", "fusion_ego_front"])
    if not _has_option("--npc-vehicles"):
        sys.argv.extend(["--npc-vehicles", "0"])
    if not _has_option("--npc-pedestrians"):
        sys.argv.extend(["--npc-pedestrians", "0"])
    if not _has_option("--enable-semantic-gt") and not _has_option("--disable-semantic-gt"):
        sys.argv.append("--enable-semantic-gt")
    # The projected-3D-OBB box uses deprioritized dims/yaw and is unreliable; default the
    # ego deploy to the learned 2D box only (override with --draw-projected-obb-box).
    if not _has_option("--draw-projected-obb-box") and not _has_option("--no-draw-projected-obb-box"):
        sys.argv.append("--no-draw-projected-obb-box")
    fusion_runtime.main()


if __name__ == "__main__":
    main()
