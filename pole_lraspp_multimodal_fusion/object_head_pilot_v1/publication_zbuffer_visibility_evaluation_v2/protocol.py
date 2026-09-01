"""Fail-closed loader for the registered protocol-v2 control files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data_collection.route_b_publication_zbuffer_visibility_v2.core import sha256


PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parents[2]
LOCK_DIR = PACKAGE_ROOT.parent / "splitfusion_fcos_r50_fpn_p2_p7_service_candidate_v1"
LOCK_PATH = LOCK_DIR / "PERCEPTION_BASELINE_LOCK_V1.json"
PROTOCOL_PATH = LOCK_DIR / "PUBLICATION_EVALUATION_PROTOCOL_V2.json"
AMENDMENT_PATH = LOCK_DIR / "PUBLICATION_EVALUATION_PROTOCOL_V2_AMENDMENT_001.json"
BLOCKED_EVIDENCE_PATH = (
    ROOT
    / "data_collection/experiments/route_b_publication_instance_visibility_v1"
    / "qualification_20260901_005600/CONTROLLED_RENDERER_BLOCKED_EVIDENCE.json"
)
PREVIOUS_FAILURE_PATH = (
    ROOT
    / "data_collection/experiments/route_b_publication_zbuffer_visibility_v2"
    / "qualification_20260901_020000/controlled_qualification"
    / "controlled_qualification_failure.json"
)
EXPECTED_LOCK_SHA256 = "e3d15610978a11acf1d4da2d98608bb6a7bfcae79130ab64e6592b78a4facf6b"
EXPECTED_PROTOCOL_SHA256 = "361bd50f9f94f18689a67fbb2cfb3ed0ac02f668677daf3ab861405210adb728"
EXPECTED_AMENDMENT_SHA256 = "20f2fd616ab1e498c1c859dcee5c57b8a233cbd80d25e149f721cc2d3f911228"
EXPECTED_BLOCKED_EVIDENCE_SHA256 = (
    "0ab0c5a68b8130a8f385294a2b86fc38d3fee716386434faaf121846aed9c371"
)
EXPECTED_PREVIOUS_FAILURE_SHA256 = (
    "25c5a8c713d865fb264dde45f1a9abda74d05fb610031bed81723d9239f2fc11"
)


class FrozenProtocolError(RuntimeError):
    pass


def _require_hash(path: Path, expected: str, label: str) -> None:
    try:
        actual = sha256(path)
    except OSError as exc:
        raise FrozenProtocolError(f"missing bound {label}: {path}") from exc
    if actual != expected:
        raise FrozenProtocolError(
            f"bound {label} SHA-256 drift: {actual} != {expected}: {path}"
        )


def load_registered_protocol(
    *,
    root: Path = ROOT,
    lock_path: Path = LOCK_PATH,
    protocol_path: Path = PROTOCOL_PATH,
    amendment_path: Path = AMENDMENT_PATH,
    blocked_evidence_path: Path = BLOCKED_EVIDENCE_PATH,
    previous_failure_path: Path = PREVIOUS_FAILURE_PATH,
    expected_lock_sha256: str = EXPECTED_LOCK_SHA256,
    expected_protocol_sha256: str = EXPECTED_PROTOCOL_SHA256,
    expected_amendment_sha256: str = EXPECTED_AMENDMENT_SHA256,
    expected_blocked_evidence_sha256: str = EXPECTED_BLOCKED_EVIDENCE_SHA256,
    expected_previous_failure_sha256: str = EXPECTED_PREVIOUS_FAILURE_SHA256,
) -> dict[str, Any]:
    _require_hash(lock_path, expected_lock_sha256, "baseline lock")
    _require_hash(protocol_path, expected_protocol_sha256, "publication protocol v2")
    _require_hash(amendment_path, expected_amendment_sha256, "protocol-v2 amendment 001")
    _require_hash(
        blocked_evidence_path,
        expected_blocked_evidence_sha256,
        "blocked protocol-v1 evidence",
    )
    _require_hash(
        previous_failure_path,
        expected_previous_failure_sha256,
        "triggering controlled-v2 failure evidence",
    )
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenProtocolError(f"invalid frozen registration JSON: {exc}") from exc
    if lock.get("schema") != "splitfusion_fcos_perception_baseline_lock_v1":
        raise FrozenProtocolError("baseline lock schema drift")
    if protocol.get("schema") != "route_b_publication_renderer_zbuffer_visibility_protocol_v2":
        raise FrozenProtocolError("publication protocol-v2 schema drift")
    if amendment.get("schema") != "route_b_publication_renderer_zbuffer_visibility_protocol_v2_amendment_001":
        raise FrozenProtocolError("publication protocol-v2 amendment schema drift")
    baseline = protocol.get("baseline_lock", {})
    if baseline.get("path") != str(lock_path.relative_to(root)):
        raise FrozenProtocolError("protocol baseline-lock path drift")
    if baseline.get("sha256") != expected_lock_sha256:
        raise FrozenProtocolError("protocol baseline-lock hash binding drift")
    blocked = protocol.get("blocked_evidence", {})
    if blocked.get("path") != str(blocked_evidence_path.relative_to(root)):
        raise FrozenProtocolError("protocol blocked-evidence path drift")
    if blocked.get("sha256") != expected_blocked_evidence_sha256:
        raise FrozenProtocolError("protocol blocked-evidence hash binding drift")
    amendment_base = amendment.get("base_protocol", {})
    if amendment_base.get("path") != str(protocol_path.relative_to(root)):
        raise FrozenProtocolError("amendment base-protocol path drift")
    if amendment_base.get("sha256") != expected_protocol_sha256:
        raise FrozenProtocolError("amendment base-protocol hash binding drift")
    triggering = amendment.get("triggering_evidence", {})
    if triggering.get("path") != str(previous_failure_path.relative_to(root)):
        raise FrozenProtocolError("amendment triggering-evidence path drift")
    if triggering.get("sha256") != expected_previous_failure_sha256:
        raise FrozenProtocolError("amendment triggering-evidence hash binding drift")
    visibility = protocol.get("visibility", {})
    if visibility.get("tau_empty_m") != 0.02 or visibility.get("tau_match_m") != 0.02:
        raise FrozenProtocolError("registered depth tolerances drift")
    if visibility.get("thresholds") != [0.1, 0.25, 0.5, 0.7, 0.85]:
        raise FrozenProtocolError("visibility thresholds drift")
    gates = protocol.get("qualification_before_collection", {})
    if gates.get("clear_visibility_minimum") != 0.98:
        raise FrozenProtocolError("clear qualification threshold drift")
    if gates.get("full_visibility_maximum") != 0.02:
        raise FrozenProtocolError("full qualification threshold drift")
    if gates.get("vehicle_depth_mask_vs_instance_mask_iou_minimum") != 0.98:
        raise FrozenProtocolError("vehicle diagnostic IoU threshold drift")
    change = amendment.get("single_change", {})
    if change != {
        "field": "qualification_before_collection.vehicle_depth_mask_vs_instance_mask_iou_minimum",
        "old": 0.98,
        "new": None,
        "new_role": "optional_nonblocking_diagnostic",
        "missing_component_behavior": (
            "record instance_diagnostic_unavailable with raw evidence and continue primary z-buffer qualification"
        ),
    }:
        raise FrozenProtocolError("amendment single-change binding drift")
    return {
        "lock": lock,
        "protocol": protocol,
        "amendment": amendment,
        "lock_path": str(lock_path),
        "protocol_path": str(protocol_path),
        "amendment_path": str(amendment_path),
        "blocked_evidence_path": str(blocked_evidence_path),
        "previous_failure_path": str(previous_failure_path),
        "lock_sha256": expected_lock_sha256,
        "protocol_sha256": expected_protocol_sha256,
        "amendment_sha256": expected_amendment_sha256,
        "blocked_evidence_sha256": expected_blocked_evidence_sha256,
        "previous_failure_sha256": expected_previous_failure_sha256,
        "vehicle_instance_diagnostic_required": False,
        "registered_controls_verified": True,
    }
