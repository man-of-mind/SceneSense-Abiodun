"""Small, dependency-light invariants for true instance visibility evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


INSTANCE_ENCODING = "CARLA raw BGRA: semantic=R, rendered_instance_id=B+256*G"
VISIBILITY_DEFINITION = (
    "count(visible_actor_mask AND unoccluded_actor_mask) / "
    "count(in_frame_unoccluded_actor_mask)"
)


class VisibilityGroundTruthError(RuntimeError):
    """Fail-closed publication ground-truth error."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_instance_bgra(raw_bgra: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode CARLA's raw instance image without assuming actor-ID equality."""
    raw = np.asarray(raw_bgra)
    if raw.ndim != 3 or raw.shape[2] != 4 or raw.dtype != np.uint8:
        raise VisibilityGroundTruthError(
            f"instance image must be uint8 HxWx4 BGRA, got {raw.dtype} {raw.shape}"
        )
    semantic = raw[:, :, 2].copy()
    rendered_id = raw[:, :, 0].astype(np.uint32)
    rendered_id += raw[:, :, 1].astype(np.uint32) << np.uint32(8)
    return semantic, rendered_id


def image_bgra(image: Any) -> np.ndarray:
    raw = np.frombuffer(image.raw_data, dtype=np.uint8)
    expected = int(image.width) * int(image.height) * 4
    if raw.size != expected:
        raise VisibilityGroundTruthError(
            f"instance image byte count {raw.size} != {expected}"
        )
    return raw.reshape((int(image.height), int(image.width), 4)).copy()


def instance_mask(rendered_ids: np.ndarray, rendered_instance_id: int) -> np.ndarray:
    ids = np.asarray(rendered_ids)
    if ids.ndim != 2:
        raise VisibilityGroundTruthError(f"rendered IDs must be HxW, got {ids.shape}")
    value = int(rendered_instance_id)
    if value <= 0 or value > 65535:
        raise VisibilityGroundTruthError(f"rendered instance ID outside 16-bit contract: {value}")
    return ids == value


def measure_visibility(
    visible_actor_mask: np.ndarray,
    unoccluded_actor_mask: np.ndarray,
) -> dict[str, int | float]:
    """Apply the registered equation and retain mismatch evidence."""
    visible = np.asarray(visible_actor_mask, dtype=bool)
    reference = np.asarray(unoccluded_actor_mask, dtype=bool)
    if visible.ndim != 2 or reference.ndim != 2 or visible.shape != reference.shape:
        raise VisibilityGroundTruthError(
            f"visible/reference mask shape mismatch: {visible.shape} vs {reference.shape}"
        )
    denominator = int(np.count_nonzero(reference))
    if denominator <= 0:
        raise VisibilityGroundTruthError("unoccluded in-frame actor mask has zero area")
    visible_pixels = int(np.count_nonzero(visible))
    overlap = int(np.count_nonzero(visible & reference))
    outside = int(np.count_nonzero(visible & ~reference))
    value = overlap / denominator
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise VisibilityGroundTruthError(f"invalid visibility ratio: {value}")
    ys, xs = np.nonzero(reference)
    return {
        "visible_pixels": visible_pixels,
        "unoccluded_pixels": denominator,
        "overlap_pixels": overlap,
        "visible_outside_reference_pixels": outside,
        "visibility": float(value),
        "unoccluded_bbox_x0": int(xs.min()),
        "unoccluded_bbox_y0": int(ys.min()),
        "unoccluded_bbox_x1": int(xs.max()) + 1,
        "unoccluded_bbox_y1": int(ys.max()) + 1,
    }


def prove_actor_id_mapping(
    observations: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Require a controlled one-to-one actor/rendered-token correspondence."""
    rows = list(observations)
    if not rows:
        raise VisibilityGroundTruthError("actor-ID mapping proof has no observations")
    actor_ids: set[int] = set()
    rendered_ids: set[int] = set()
    normalized = []
    for row in rows:
        actor_id = int(row["actor_id"])
        candidates = sorted({int(value) for value in row["rendered_instance_ids"] if int(value) > 0})
        if len(candidates) != 1:
            raise VisibilityGroundTruthError(
                f"actor {actor_id} rendered instance IDs {candidates}; unique mapping not proven"
            )
        rendered_id = candidates[0]
        if actor_id in actor_ids or rendered_id in rendered_ids:
            raise VisibilityGroundTruthError(
                f"duplicate controlled actor/rendered token: {actor_id}/{rendered_id}"
            )
        actor_ids.add(actor_id)
        rendered_ids.add(rendered_id)
        normalized.append({"actor_id": actor_id, "rendered_instance_id": rendered_id})
    return {
        "encoding": INSTANCE_ENCODING,
        "observations": normalized,
        "unique_actor_ids": len(actor_ids),
        "unique_rendered_ids": len(rendered_ids),
        "actor_id_equals_rendered_instance_id": all(
            row["actor_id"] == row["rendered_instance_id"] for row in normalized
        ),
        "bijection_proven": len(actor_ids) == len(rendered_ids) == len(normalized),
        "actor_to_rendered_instance_id": {
            str(row["actor_id"]): row["rendered_instance_id"] for row in normalized
        },
    }


def transform_matrix(value: Any) -> np.ndarray:
    matrix = np.asarray(value.get_matrix(), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise VisibilityGroundTruthError(f"invalid transform matrix: {matrix.shape}")
    return matrix


def inverse_transform_matrix(value: Any) -> np.ndarray:
    matrix = np.asarray(value.get_inverse_matrix(), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise VisibilityGroundTruthError(f"invalid inverse transform matrix: {matrix.shape}")
    return matrix


def relative_transform_matrix(camera_transform: Any, actor_transform: Any) -> np.ndarray:
    return inverse_transform_matrix(camera_transform) @ transform_matrix(actor_transform)


def reproduce_transform_matrix(
    reference_camera_transform: Any,
    camera_relative_actor_matrix: np.ndarray,
) -> np.ndarray:
    relative = np.asarray(camera_relative_actor_matrix, dtype=np.float64)
    if relative.shape != (4, 4) or not np.all(np.isfinite(relative)):
        raise VisibilityGroundTruthError("invalid camera-relative actor matrix")
    return transform_matrix(reference_camera_transform) @ relative


def matrix_to_transform_payload(matrix: np.ndarray) -> dict[str, dict[str, float]]:
    """Invert CARLA's yaw-pitch-roll matrix convention without importing CARLA."""
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise VisibilityGroundTruthError("invalid 4x4 transform")
    pitch = math.asin(max(-1.0, min(1.0, float(value[2, 0]))))
    cp = math.cos(pitch)
    if abs(cp) > 1e-8:
        yaw = math.atan2(float(value[1, 0]), float(value[0, 0]))
        roll = math.atan2(float(-value[2, 1]), float(value[2, 2]))
    else:
        yaw = math.atan2(float(-value[0, 1]), float(value[1, 1]))
        roll = 0.0
    return {
        "location": {"x": float(value[0, 3]), "y": float(value[1, 3]), "z": float(value[2, 3])},
        "rotation": {
            "pitch": math.degrees(pitch),
            "yaw": math.degrees(yaw),
            "roll": math.degrees(roll),
        },
    }


def transform_payload(value: Any) -> dict[str, dict[str, float]]:
    return {
        "location": {axis: float(getattr(value.location, axis)) for axis in "xyz"},
        "rotation": {
            key: float(getattr(value.rotation, key)) for key in ("pitch", "yaw", "roll")
        },
    }


def write_json_x(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_png_x(path: Path, mask: np.ndarray) -> str:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    binary = np.where(np.asarray(mask, dtype=bool), 255, 0).astype(np.uint8)
    if not cv2.imwrite(str(path), binary, [int(cv2.IMWRITE_PNG_COMPRESSION), 3]):
        raise VisibilityGroundTruthError(f"failed to write {path}")
    return sha256(path)


def require_renderer_proof(proof: Mapping[str, Any]) -> None:
    required = {
        "actor_id_mapping_proven": True,
        "reference_intrinsics_equal": True,
        "reference_coordinates_equal": True,
        "external_geometry_absent": True,
        "walker_bone_pose_copy_proven": True,
    }
    failures = [key for key, expected in required.items() if proof.get(key) is not expected]
    if failures:
        raise VisibilityGroundTruthError(
            "PUBLICATION_VISIBILITY_GROUND_TRUTH_BLOCKED: renderer proof failed "
            + ",".join(failures)
        )
