"""Create-only ground-truth handoff from CARLA host to edge evaluator."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .scoring import immutable_mask, immutable_rows


GT_OBJECT_SCHEMA = "splitfusion_privileged_object_gt.v1"
GT_SEMANTIC_SCHEMA = "splitfusion_privileged_semantic_gt.v1"


class GroundTruthEvidenceError(RuntimeError):
    pass


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stem(stream_id: str, frame_id: int) -> str:
    digest = hashlib.sha256(str(stream_id).encode("utf-8")).hexdigest()[:16]
    return f"quality_gt_{digest}_{int(frame_id)}"


def _create_identical_or_fail(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise GroundTruthEvidenceError(f"conflicting evidence: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _identity(document: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(
        document[name]
        for name in (
            "run_id",
            "cell_id",
            "stream_id",
            "frame_id",
            "action_id",
            "profile_id",
            "capture_timestamp_ns",
        )
    )


def write_object_ground_truth(
    directory: Path,
    *,
    identity: Mapping[str, Any],
    frozen_carla_frame_id: int,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    if int(frozen_carla_frame_id) != int(identity["frame_id"]):
        raise GroundTruthEvidenceError(
            "world snapshot frame differs from synchronized sensor frame"
        )
    owned = immutable_rows(rows)
    ready_wall_ns, ready_monotonic_ns = time.time_ns(), time.monotonic_ns()
    document = {
        "schema": GT_OBJECT_SCHEMA,
        **dict(identity),
        "frozen_carla_frame_id": int(frozen_carla_frame_id),
        "gt_ready_wall_ns": ready_wall_ns,
        "gt_ready_monotonic_ns": ready_monotonic_ns,
        "objects": list(owned),
    }
    path = Path(directory) / f"{_stem(str(identity['stream_id']), int(identity['frame_id']))}.objects.json"
    _create_identical_or_fail(path, _canonical(document))
    return path


def write_semantic_ground_truth(
    directory: Path,
    *,
    identity: Mapping[str, Any],
    frozen_carla_frame_id: int,
    mask: np.ndarray,
) -> tuple[Path, Path]:
    if int(frozen_carla_frame_id) != int(identity["frame_id"]):
        raise GroundTruthEvidenceError(
            "semantic frame differs from synchronized sensor frame"
        )
    owned = immutable_mask(mask)
    payload_path = Path(directory) / f"{_stem(str(identity['stream_id']), int(identity['frame_id']))}.semantic.npy"
    temporary = payload_path.with_name(
        f".{payload_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with temporary.open("xb") as handle:
            np.save(handle, owned, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        payload = temporary.read_bytes()
    finally:
        temporary.unlink(missing_ok=True)
    _create_identical_or_fail(payload_path, payload)
    ready_wall_ns, ready_monotonic_ns = time.time_ns(), time.monotonic_ns()
    document = {
        "schema": GT_SEMANTIC_SCHEMA,
        **dict(identity),
        "frozen_carla_frame_id": int(frozen_carla_frame_id),
        "gt_ready_wall_ns": ready_wall_ns,
        "gt_ready_monotonic_ns": ready_monotonic_ns,
        "shape": list(owned.shape),
        "dtype": str(owned.dtype),
        "sha256": hashlib.sha256(owned.tobytes()).hexdigest(),
    }
    sidecar = payload_path.with_suffix(".json")
    _create_identical_or_fail(sidecar, _canonical(document))
    return payload_path, sidecar


def read_ground_truth(
    directory: Path,
    *,
    expected_identity: Mapping[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    stem = _stem(
        str(expected_identity["stream_id"]), int(expected_identity["frame_id"])
    )
    root = Path(directory)
    objects_path = root / f"{stem}.objects.json"
    mask_path = root / f"{stem}.semantic.npy"
    mask_sidecar_path = root / f"{stem}.semantic.json"
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        if objects_path.is_file() and mask_path.is_file() and mask_sidecar_path.is_file():
            break
        time.sleep(0.001)
    else:
        raise GroundTruthEvidenceError(
            f"ground truth missing after {timeout_s:.3f}s for {stem}"
        )
    objects_bytes = objects_path.read_bytes()
    semantic_sidecar_bytes = mask_sidecar_path.read_bytes()
    objects = json.loads(objects_bytes.decode("utf-8"))
    semantic = json.loads(semantic_sidecar_bytes.decode("utf-8"))
    if objects.get("schema") != GT_OBJECT_SCHEMA:
        raise GroundTruthEvidenceError("object ground-truth schema drift")
    if semantic.get("schema") != GT_SEMANTIC_SCHEMA:
        raise GroundTruthEvidenceError("semantic ground-truth schema drift")
    expected = _identity(expected_identity)
    if _identity(objects) != expected or _identity(semantic) != expected:
        raise GroundTruthEvidenceError("ground-truth identity mismatch")
    frame_id = int(expected_identity["frame_id"])
    if (
        int(objects.get("frozen_carla_frame_id", -1)) != frame_id
        or int(semantic.get("frozen_carla_frame_id", -1)) != frame_id
    ):
        raise GroundTruthEvidenceError("ground-truth snapshot/frame mismatch")
    raw_mask = np.load(mask_path, allow_pickle=False)
    if str(raw_mask.dtype) != str(semantic.get("dtype")) or raw_mask.dtype != np.uint8:
        raise GroundTruthEvidenceError("semantic ground-truth dtype drift")
    if list(raw_mask.shape) != list(semantic.get("shape") or ()):
        raise GroundTruthEvidenceError("semantic ground-truth shape drift")
    mask = immutable_mask(raw_mask)
    if hashlib.sha256(mask.tobytes()).hexdigest() != semantic.get("sha256"):
        raise GroundTruthEvidenceError("semantic ground-truth digest mismatch")
    return {
        "objects": immutable_rows(objects["objects"]),
        "semantic": mask,
        "frozen_carla_frame_id": frame_id,
        "gt_ready_wall_ns": max(
            int(objects["gt_ready_wall_ns"]), int(semantic["gt_ready_wall_ns"])
        ),
        "gt_ready_monotonic_ns": max(
            int(objects["gt_ready_monotonic_ns"]),
            int(semantic["gt_ready_monotonic_ns"]),
        ),
        "object_gt_sha256": hashlib.sha256(objects_bytes).hexdigest(),
        "semantic_gt_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
        "semantic_sidecar_sha256": hashlib.sha256(semantic_sidecar_bytes).hexdigest(),
        "semantic_npy_sha256": _sha256_file(mask_path),
    }


def read_semantic_ground_truth(
    directory: Path,
    *,
    expected_identity: Mapping[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    """Read semantic GT without waiting for the independent object-GT branch."""

    stem = _stem(
        str(expected_identity["stream_id"]), int(expected_identity["frame_id"])
    )
    root = Path(directory)
    mask_path = root / f"{stem}.semantic.npy"
    sidecar_path = root / f"{stem}.semantic.json"
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        if mask_path.is_file() and sidecar_path.is_file():
            break
        time.sleep(0.001)
    else:
        raise GroundTruthEvidenceError(
            f"semantic ground truth missing after {timeout_s:.3f}s for {stem}"
        )
    sidecar_bytes = sidecar_path.read_bytes()
    sidecar = json.loads(sidecar_bytes.decode("utf-8"))
    if sidecar.get("schema") != GT_SEMANTIC_SCHEMA:
        raise GroundTruthEvidenceError("semantic ground-truth schema drift")
    if _identity(sidecar) != _identity(expected_identity):
        raise GroundTruthEvidenceError("semantic ground-truth identity mismatch")
    frame_id = int(expected_identity["frame_id"])
    if int(sidecar.get("frozen_carla_frame_id", -1)) != frame_id:
        raise GroundTruthEvidenceError("semantic ground-truth frame mismatch")
    raw_mask = np.load(mask_path, allow_pickle=False)
    if raw_mask.dtype != np.uint8 or str(raw_mask.dtype) != str(sidecar.get("dtype")):
        raise GroundTruthEvidenceError("semantic ground-truth dtype drift")
    if list(raw_mask.shape) != list(sidecar.get("shape") or ()):
        raise GroundTruthEvidenceError("semantic ground-truth shape drift")
    mask = immutable_mask(raw_mask)
    if hashlib.sha256(mask.tobytes()).hexdigest() != sidecar.get("sha256"):
        raise GroundTruthEvidenceError("semantic ground-truth digest mismatch")
    return {
        "semantic": mask,
        "frozen_carla_frame_id": frame_id,
        "gt_ready_wall_ns": int(sidecar["gt_ready_wall_ns"]),
        "gt_ready_monotonic_ns": int(sidecar["gt_ready_monotonic_ns"]),
        "semantic_gt_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
        "semantic_sidecar_sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
        "semantic_npy_sha256": _sha256_file(mask_path),
    }
