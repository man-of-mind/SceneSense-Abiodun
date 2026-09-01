"""Fail-closed loader for the immutable publication registration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parents[2]
LOCK_DIR = PACKAGE_ROOT.parent / "splitfusion_fcos_r50_fpn_p2_p7_service_candidate_v1"
LOCK_PATH = LOCK_DIR / "PERCEPTION_BASELINE_LOCK_V1.json"
PROTOCOL_PATH = LOCK_DIR / "PUBLICATION_EVALUATION_PROTOCOL_V1.json"
EXPECTED_LOCK_SHA256 = "e3d15610978a11acf1d4da2d98608bb6a7bfcae79130ab64e6592b78a4facf6b"
EXPECTED_PROTOCOL_SHA256 = "5f63a9415c33cac237d52faf5a45a60e4c5fd10ff5ba4e64bf407ba110c956ed"


class FrozenProtocolError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
    expected_lock_sha256: str = EXPECTED_LOCK_SHA256,
    expected_protocol_sha256: str = EXPECTED_PROTOCOL_SHA256,
    verify_bound_files: bool = True,
) -> dict[str, Any]:
    _require_hash(lock_path, expected_lock_sha256, "baseline lock")
    _require_hash(protocol_path, expected_protocol_sha256, "publication protocol")
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenProtocolError(f"invalid frozen registration JSON: {exc}") from exc
    if lock.get("schema") != "splitfusion_fcos_perception_baseline_lock_v1":
        raise FrozenProtocolError("baseline lock schema drift")
    if protocol.get("schema") != "route_b_publication_instance_visibility_protocol_v1":
        raise FrozenProtocolError("publication protocol schema drift")
    if protocol.get("baseline_lock") != str(LOCK_PATH.relative_to(ROOT)):
        raise FrozenProtocolError("publication protocol baseline-lock binding drift")
    visibility = protocol.get("visibility", {})
    if visibility.get("thresholds") != [0.1, 0.25, 0.5, 0.7, 0.85]:
        raise FrozenProtocolError("visibility thresholds drift")
    if protocol.get("evaluation", {}).get("fixed_score_views") != [0.2, 0.02]:
        raise FrozenProtocolError("score views drift")
    if protocol.get("non_visibility_eligibility", {}).get("maximum_range_m") != 40.0:
        raise FrozenProtocolError("40 m eligibility drift")
    if verify_bound_files:
        items = [(lock["base_checkpoint"]["path"], lock["base_checkpoint"]["sha256"], "primary checkpoint")]
        frozen = protocol["frozen_models"]
        for item in [frozen["primary"], *frozen["comparators"]]:
            items.append((item["checkpoint"], item["checkpoint_sha256"], f"checkpoint {item['name']}"))
        items.extend((
            (lock["service_candidate"]["locked_config_path"], lock["service_candidate"]["locked_config_sha256"], "service configuration"),
            (lock["postprocessing"]["person"]["feasibility_result_path"], lock["postprocessing"]["person"]["feasibility_result_sha256"], "person consolidation result"),
        ))
        for relative, expected, label in items:
            _require_hash(root / relative, expected, label)
    return {
        "lock": lock,
        "protocol": protocol,
        "lock_path": str(lock_path),
        "protocol_path": str(protocol_path),
        "lock_sha256": expected_lock_sha256,
        "protocol_sha256": expected_protocol_sha256,
        "bound_files_verified": bool(verify_bound_files),
    }
