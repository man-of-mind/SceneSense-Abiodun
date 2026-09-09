"""Shared identity, binding and statistics for the SplitFusion timing diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]

SCHEMA = "scenesense.splitfusion_timing_diagnostic.v1"
MANIFEST_SCHEMA = "scenesense.splitfusion_timing_diagnostic_manifest.v1"
EDGE_RECORD_SCHEMA = "scenesense.splitfusion_timing_diagnostic_edge_record.v1"
EDGE_SUMMARY_SCHEMA = "scenesense.splitfusion_timing_diagnostic_edge_summary.v1"
EDGE_READY_SCHEMA = "scenesense.splitfusion_timing_diagnostic_edge_ready.v1"
TERMINAL_SUCCESS = "SPLITFUSION_TIMING_DIAGNOSTIC_V1_COMPLETE"
TERMINAL_FAILURE = "SPLITFUSION_TIMING_DIAGNOSTIC_V1_FAILED"

OUTPUT_RELPATH = "experiments/splitfusion_timing_diagnostic_v1/20260909_actions30_15_50_71_favorable_stable"

# Exactly the four authorized catalog actions, in declaration order.
DIAGNOSTIC_ACTIONS: tuple[tuple[int, str], ...] = (
    (30, "split_ae128_uint4_q0000"),
    (15, "split_noae_uint4_q7000"),
    (50, "split_ae64_uint4_q5000"),
    (71, "split_ae32_uint4_q9800"),
)
FRAMES = 300
SCHEDULE_PERIOD_NS = 100_000_000
NETWORK_PROFILE_ID = "FAVORABLE_STABLE"
DEVICE_NAME = "NVIDIA GeForce RTX 5090"
CLEAN_NOISE_POWER_DB = -50.0

# The registered Phase-13C fit-only 300-frame sample, bound by its own two
# digests as published in the completed Phase-13C run manifest.
PHASE13C_EVIDENCE_RELPATH = (
    "experiments/splitfusion_live_dispatch_v1/20260904_phase13c_36x300_localhost_measurement"
)
PHASE13C_SAMPLE_MANIFEST_SHA256 = (
    "ad2cb25ea8d64d8d774c9c455c09e169f9a3eb1b51bf6143e82dfacdecf10ee9"
)

# Previously published reference points this diagnostic is asked to decompose.
PHASE13C_CONTROLLED_TAIL_MS = 73.4
PHASE15_LIVE_FROZEN_TAIL_MS = 111.6

# Every implementation, configuration and binding input, pinned by SHA-256.
BOUND_INPUTS: Mapping[str, str] = {
    "rl_agent/splitfusion_live_dispatch_v1/context_tail.py": "1781f3013967c464aa8f8de0eb6b46bccb5f3ff00a305c6f6d56012383859fe1",
    "rl_agent/splitfusion_live_dispatch_v1/edge_runtime.py": "4b5f89eb49b11300cd6762e356a599affa32158479a76f1ad91a8c404d2d6125",
    "rl_agent/splitfusion_live_dispatch_v1/ue_runtime.py": "a72f6ab4b122adb232e4b5927fadd6e0e5381c1fe0201bbf502b28cf2ade1029",
    "rl_agent/splitfusion_live_dispatch_v1/envelope.py": "e49a70e1f92097cd42735f2aea2cd1171a6e4be2a361ab6834e518bfb213b418",
    "rl_agent/splitfusion_live_dispatch_v1/transport.py": "5c8bd3d1bb18818334a28266822b30d475bc67dc016dcf04df7db595e159222f",
    "rl_agent/splitfusion_live_dispatch_v1/frame_context.py": "845c3886f8862ce3f1cd618323b5b7c8baba05b043e5709ff0cd83d36fc33676",
    "rl_agent/splitfusion_live_dispatch_v1/live_pilot_runtime.py": "ce568a89fa262916e7b6f737a5d1f7fc310428de96ff170eb56eca303411f894",
    "rl_agent/splitfusion_live_dispatch_v1/live_pilot_target_snr_runtime.py": "b43b71dfdeb12d51d6eae9d1d8d59e3b44ddbe3c18a587b4f851b275fd5b9018",
    "rl_agent/splitfusion_live_dispatch_v1/phase13c_measurement.py": "fa04e483dcc38801b34f989f5cf58eac1a7d91b9641c1a98052c50640484ad29",
    "rl_agent/splitfusion_live_dispatch_v1/runtime_binding.json": "604450c7f0f791a480ba4e5ed6b286a2b6f68497735372a7a63ed92e93585551",
    "rl_agent/configs/splitfusion_16_cell_live_carla_oai_pilot_v1.json": "603d3c211fbe70934a3537ba090d16e66a0ac2ce1cb62015389c1e406cec2520",
    "rl_agent/configs/splitfusion_phase14a_100mhz_calibration_v1.json": "ad541f71f5659e1bb08d7d2c48a45086dfbd5bf45e669b32adfb320ddcab5cd9",
    "rl_agent/configs/splitfusion_phase14a_campaign_binding_v1.json": "103aeda31a37594c89e820daf1794d28af0440f5e4324450cac01266f5540004",
    "uplink_only_spatial_map_pipeline/run_splitfusion_oai_100mhz_4d5u_v1.sh": "8e02f0913338a187bff1de24bfea09d7eca03607da0e383ef8ad4a005b64ce61",
    "phase2_map_sharing/transport.py": "c8d8d0b253356c11776e9c35b7d6b1bef009bfc980e4cecfcbba84ae95734a6e",
    "scripts/receiver_container_fusion_back_up.sh": "abf532c88d27fcf101dcecbfcf55f4994fd44cd287d5d1a9245ce0ca3b333610",
    "receiver_container/docker-compose.yaml": "aa45477d63d40799a45115f6db5f07a1eb2440217f43b32173c8536f759c0857",
    "receiver_container/docker-compose.fusion-back.yaml": "6d00149c7d9a9502af1ea701e3494aed0a7bbf9dbb1412f0c2b1e1b7b0f9061a",
}

PILOT_CONFIG_RELPATH = "rl_agent/configs/splitfusion_16_cell_live_carla_oai_pilot_v1.json"

# Preserve, never touch, the operator's own dirty worktree paths.
EXPECTED_DIRTY_PATHS = (
    "OAI/openairinterface5g",
    "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
    "lraspp_to_splitfusion_fcos_report_v1/FULL_TECHNICAL_REPORT_AVO_V2.md",
)


class DiagnosticError(RuntimeError):
    """A fatal diagnostic contract violation."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticError(message)


def repo_path(relative: str) -> Path:
    return (ROOT / relative).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def seal(document: Mapping[str, Any], field: str) -> dict[str, Any]:
    body = {key: value for key, value in document.items() if key != field}
    return {**body, field: digest_bytes(canonical_bytes(body))}


def load_json(path: Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_create_json(path: Path, value: Any) -> str:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    return atomic_create_text(path, payload)


def atomic_create_text(path: Path, text: str) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
    return sha256_file(path)


def nearest_rank(values: Sequence[float], probability: float) -> float | None:
    """Nearest-rank percentile, matching the Phase-13C/Phase-15 convention."""

    finite = sorted(float(value) for value in values if _is_finite(value))
    if not finite:
        return None
    if probability <= 0.0:
        return finite[0]
    index = max(1, math.ceil(probability * len(finite))) - 1
    return finite[min(index, len(finite) - 1)]


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def summarize(values: Sequence[float]) -> dict[str, Any]:
    """Count, median, p90, p95, minimum and maximum for one timing stage."""

    finite = sorted(float(value) for value in values if _is_finite(value))
    if not finite:
        return {
            "count": 0,
            "median": None,
            "p90": None,
            "p95": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "count": len(finite),
        "median": nearest_rank(finite, 0.5),
        "p90": nearest_rank(finite, 0.90),
        "p95": nearest_rank(finite, 0.95),
        "minimum": finite[0],
        "maximum": finite[-1],
    }


def clock_anchor(label: str) -> dict[str, Any]:
    """One paired wall/monotonic anchor used to prove a shared clock domain."""

    import time

    wall = time.time_ns()
    monotonic = time.monotonic_ns()
    return {
        "label": str(label),
        "wall_ns": int(wall),
        "monotonic_ns": int(monotonic),
        "wall_minus_monotonic_ns": int(wall - monotonic),
    }
