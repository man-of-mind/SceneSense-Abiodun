#!/usr/bin/env python3
"""Epic-quality rendering provenance and the collection preflight gate.

Every canonical Route B perception episode must be rendered at CARLA ``Epic``
quality with GPU rendering enabled. Quality level is a server launch flag and is
not exposed on ``carla.WorldSettings``, so it is confirmed here from the
*controlled launch configuration*: the command line of the running
``CarlaUnreal-Linux-Shipping`` server bound to the RPC port in use.

``-RenderOffScreen`` is permitted - it is headless GPU rendering, not
no-rendering mode. What is rejected is ``-quality-level=Low`` (or any non-Epic
level), a missing explicit ``-quality-level``, any flag that disables the RHI
(``-nullrhi``, ``-nullRHI``, ``-norender``, ``-disablerendering``), and a world
whose ``no_rendering_mode`` is true.

The resulting :func:`render_provenance` payload is embedded verbatim in every
episode manifest so that a dataset consumer can prove the renderer condition.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

REQUIRED_QUALITY_LEVEL = "Epic"
FORBIDDEN_QUALITY_LEVELS = ("low", "medium", "high")
RENDER_DISABLING_FLAGS = ("-nullrhi", "-norender", "-disablerendering", "-noRHI")
SERVER_BINARY_MARKER = "CarlaUnreal-Linux-Shipping"
DEFAULT_RPC_PORT = 2000


class RenderProvenanceError(RuntimeError):
    """The renderer preflight gate rejected the running server or world."""


def _cmdline(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [part for part in raw.decode("utf-8", "replace").split("\0") if part]


def _is_server_process(pid: int, argv: list[str]) -> bool:
    if not argv:
        return False
    try:
        executable = os.readlink(f"/proc/{pid}/exe")
    except OSError:
        executable = ""
    return SERVER_BINARY_MARKER in executable or SERVER_BINARY_MARKER in argv[0]


def _rpc_port(argv: list[str]) -> int:
    for token in argv:
        if token.startswith("-carla-rpc-port="):
            try:
                return int(token.split("=", 1)[1])
            except ValueError:
                continue
    return DEFAULT_RPC_PORT


def find_server_process(port: int = DEFAULT_RPC_PORT) -> tuple[int, list[str]]:
    """Return ``(pid, argv)`` of the CARLA server serving ``port``."""
    candidates: list[tuple[int, list[str]]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        argv = _cmdline(pid)
        if _is_server_process(pid, argv):
            candidates.append((pid, argv))
    matching = [(pid, argv) for pid, argv in candidates if _rpc_port(argv) == int(port)]
    if not matching:
        raise RenderProvenanceError(
            f"no running {SERVER_BINARY_MARKER} process serves RPC port {port}; "
            "the renderer preflight cannot confirm the launch configuration"
        )
    if len(matching) > 1:
        pids = ", ".join(str(pid) for pid, _ in matching)
        raise RenderProvenanceError(
            f"multiple CARLA servers claim RPC port {port} (pids {pids}); refusing to guess"
        )
    return matching[0]


def quality_level_from_argv(argv: list[str]) -> str | None:
    level: str | None = None
    for index, token in enumerate(argv):
        if token.startswith("-quality-level="):
            level = token.split("=", 1)[1]
        elif token == "-quality-level" and index + 1 < len(argv):
            level = argv[index + 1]
    return level


def inspect_launch(port: int = DEFAULT_RPC_PORT) -> dict[str, Any]:
    pid, argv = find_server_process(port)
    level = quality_level_from_argv(argv)
    lowered = [token.lower() for token in argv]
    return {
        "server_pid": pid,
        "launch_command": " ".join(argv),
        "launch_argv": argv,
        "rpc_port": _rpc_port(argv),
        "quality_level": level,
        "quality_level_explicit": level is not None,
        "render_offscreen": "-renderoffscreen" in lowered,
        "render_disabling_flags": [
            token for token in argv
            if token.lower() in {flag.lower() for flag in RENDER_DISABLING_FLAGS}
        ],
    }


def check_frames(rgb_image: Any, semantic_image: Any, *, width: int, height: int) -> dict[str, Any]:
    """Confirm the RGB and segmentation frames are non-empty, sized and synchronized."""
    import numpy as np

    report: dict[str, Any] = {}
    problems: list[str] = []
    for name, image in (("rgb", rgb_image), ("semantic", semantic_image)):
        observed = (int(image.width), int(image.height))
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        expected_bytes = int(width) * int(height) * 4
        report[name] = {
            "size": list(observed),
            "bytes": int(array.size),
            "frame": int(image.frame),
            "timestamp_s": float(image.timestamp),
        }
        if observed != (int(width), int(height)):
            problems.append(f"{name} frame size {observed} != ({width}, {height})")
        if int(array.size) != expected_bytes:
            problems.append(f"{name} payload {array.size} bytes != expected {expected_bytes}")
        if array.size == 0 or not bool(np.any(array)):
            problems.append(f"{name} frame is empty (all-zero payload)")
    if int(rgb_image.frame) != int(semantic_image.frame):
        problems.append(
            f"rgb frame {int(rgb_image.frame)} != semantic frame {int(semantic_image.frame)}"
        )
    if abs(float(rgb_image.timestamp) - float(semantic_image.timestamp)) > 1e-9:
        problems.append("rgb and semantic timestamps are not synchronized")
    report["problems"] = problems
    report["ok"] = not problems
    return report


def render_provenance(world: Any, client: Any, *, port: int = DEFAULT_RPC_PORT,
                      camera_width: int, camera_height: int, camera_fov: float) -> dict[str, Any]:
    """Full renderer provenance payload for an episode manifest."""
    launch = inspect_launch(port)
    settings = world.get_settings()
    weather = world.get_weather()
    weather_fields = (
        "cloudiness", "precipitation", "precipitation_deposits", "wind_intensity",
        "sun_azimuth_angle", "sun_altitude_angle", "fog_density", "fog_distance",
        "fog_falloff", "wetness", "scattering_intensity", "mie_scattering_scale",
        "rayleigh_scattering_scale", "dust_storm",
    )
    return {
        "launch": launch,
        "server_version": str(client.get_server_version()),
        "client_version": str(client.get_client_version()),
        "map": str(world.get_map().name),
        "no_rendering_mode": bool(settings.no_rendering_mode),
        "synchronous_mode": bool(settings.synchronous_mode),
        "fixed_delta_seconds": float(settings.fixed_delta_seconds or 0.0),
        "camera_resolution": [int(camera_width), int(camera_height)],
        "camera_fov_deg": float(camera_fov),
        "weather": {
            name: float(getattr(weather, name))
            for name in weather_fields if hasattr(weather, name)
        },
    }


def assert_epic_rendering(provenance: dict[str, Any]) -> None:
    """Raise unless the provenance proves Epic quality with rendering enabled."""
    launch = provenance["launch"]
    level = launch.get("quality_level")
    problems: list[str] = []
    if not launch.get("quality_level_explicit"):
        problems.append(
            "CARLA was launched without an explicit -quality-level; "
            f"collection requires -quality-level={REQUIRED_QUALITY_LEVEL}"
        )
    elif str(level).strip().lower() != REQUIRED_QUALITY_LEVEL.lower():
        problems.append(
            f"CARLA quality level is {level!r}; collection requires "
            f"{REQUIRED_QUALITY_LEVEL!r} and never a reduced level"
        )
    if str(level).strip().lower() in FORBIDDEN_QUALITY_LEVELS:
        problems.append(f"reduced quality level {level!r} is not admissible for canonical data")
    if launch.get("render_disabling_flags"):
        problems.append(
            f"rendering-disabling launch flags present: {launch['render_disabling_flags']}"
        )
    if provenance.get("no_rendering_mode"):
        problems.append("world no_rendering_mode is true; camera rendering is disabled")
    if problems:
        raise RenderProvenanceError("; ".join(problems))
