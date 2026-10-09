"""Strict source synchronization plus durable, bounded rejection diagnostics.

Frame number alone is insufficient: a camera's delayed callback can carry a
later timestamp/pose. Never relabel such input or silently relax the guard.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .quality_adapter import LiveGtContractError

SYNC_SCHEMA = "splitfusion.run5.exact_sensor_source_synchronization.v2"
POSE_ATOL = 1e-5  # unchanged from the first live producer; not fitted to failures


class SourceSynchronizationError(LiveGtContractError):
    def __init__(self, status: str, detail: str):
        self.status = status
        super().__init__(f"{status}: {detail}")


def _stamp(value):
    if value is None:
        return {"timestamp": None, "timestamp_repr": None, "timestamp_hex": None}
    value = float(value)
    return {"timestamp": value if math.isfinite(value) else None,
            "timestamp_repr": repr(value), "timestamp_hex": value.hex()}


def sensor_metadata(image: Any, receipt_raw_ns: int | None = None) -> dict:
    """Copy header metadata immediately in the callback; no device operation."""
    matrix = np.asarray(image.transform.get_matrix(), dtype=np.float64)
    return {"frame_id": int(image.frame), **_stamp(image.timestamp),
            "width": None if not hasattr(image, "width") else int(image.width),
            "height": None if not hasattr(image, "height") else int(image.height),
            "camera_matrix": matrix.tolist(),
            "callback_receipt_raw_ns": (time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                                        if receipt_raw_ns is None else int(receipt_raw_ns))}


def _sensor_copy(value: Mapping[str, Any]) -> dict:
    output = {key: value.get(key) for key in ("frame_id", "width", "height", "callback_receipt_raw_ns")}
    output.update(_stamp(value.get("timestamp")))
    output["camera_matrix"] = np.asarray(value["camera_matrix"], dtype=np.float64).tolist()
    return output


def source_alignment_record(*, identity: Mapping[str, Any], rgb: Mapping[str, Any],
                            semantic_image: Any, depth_image: Any, frozen_world: Any,
                            policy_timestamp_s: float, radar: Mapping[str, Any] | None = None) -> dict:
    sources = {"rgb": _sensor_copy(rgb),
               "semantic": _sensor_copy(rgb.get("semantic_callback") or sensor_metadata(semantic_image)),
               "depth": _sensor_copy(rgb.get("depth_callback") or sensor_metadata(depth_image))}
    radar = radar or rgb.get("radar_callback")
    sources["radar"] = None if radar is None else _sensor_copy(radar)
    sources["policy_radar"] = {"frame_id": int(identity["frame_id"]), **_stamp(policy_timestamp_s)}
    sources["frozen_world"] = {"frame_id": getattr(frozen_world, "frozen_frame_id", None),
                               **_stamp(getattr(frozen_world, "frozen_timestamp_s", None)),
                               "snapshot_observed_raw_ns": getattr(frozen_world, "frozen_observed_raw_ns", None)}
    reference = sources["rgb"]["timestamp"]
    deltas = {name: (None if source is None or reference is None or source.get("timestamp") is None
                     else float(source["timestamp"]) - reference) for name, source in sources.items()}
    pose_deltas = {}
    for name in ("semantic", "depth"):
        delta = np.asarray(sources[name]["camera_matrix"]) - np.asarray(sources["rgb"]["camera_matrix"])
        pose_deltas[name] = {"max_abs_matrix_delta": float(np.max(np.abs(delta))),
                             "translation_delta_m": float(np.linalg.norm(delta[:3, 3]))}
    return {"schema": SYNC_SCHEMA, "identity": dict(identity), "sources": sources,
            "timestamp_deltas_vs_rgb_s": deltas, "pose_deltas_vs_rgb": pose_deltas,
            "same_frame_required": True, "timestamp_tolerance_s": 0., "pose_matrix_atol": POSE_ATOL,
            "observed_raw_ns": time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)}


def validate_source_alignment(record: Mapping[str, Any]) -> None:
    frame_id = int(record["identity"]["frame_id"])
    sources = record["sources"]
    for role in ("rgb", "semantic", "depth", "radar", "policy_radar", "frozen_world"):
        source = sources.get(role)
        if source is None or source.get("timestamp") is None:
            raise SourceSynchronizationError("SOURCE_METADATA_MISSING_OR_NONFINITE", role)
        if int(source.get("frame_id", -1)) != frame_id:
            raise SourceSynchronizationError("SOURCE_FRAME_MISMATCH", f"{role}={source.get('frame_id')} expected={frame_id}")
    reference = sources["rgb"]["timestamp"]
    for role, source in sources.items():
        if source["timestamp"] != reference:
            raise SourceSynchronizationError("SOURCE_TIMESTAMP_MISMATCH",
                f"{role}={source['timestamp_repr']} ({source['timestamp_hex']}) "
                f"RGB={sources['rgb']['timestamp_repr']} ({sources['rgb']['timestamp_hex']})")
    shape = (sources["rgb"]["width"], sources["rgb"]["height"])
    matrix = np.asarray(sources["rgb"]["camera_matrix"], dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise SourceSynchronizationError("SOURCE_POSE_INVALID", "rgb")
    for role in ("semantic", "depth"):
        source = sources[role]
        if (source["width"], source["height"]) != shape:
            raise SourceSynchronizationError("SOURCE_IMAGE_GEOMETRY_MISMATCH", role)
        other = np.asarray(source["camera_matrix"], dtype=np.float64)
        if other.shape != (4, 4) or not np.allclose(other, matrix, rtol=0, atol=POSE_ATOL):
            raise SourceSynchronizationError("SOURCE_POSE_MISMATCH", f"{role}: {record['pose_deltas_vs_rgb'][role]}")


class LiveSyncEvidence:
    """Every triple keeps metadata; only the first bounded rejects keep raw BGRA.

    All writes are create-only. This runs on the existing GT worker, never in
    the sensor callback or UE action path. The bound is not outcome selection:
    every rejected frame/status remains in the compact evidence stream.
    """
    def __init__(self, directory: Path, raw_bundle_limit: int = 8,
                 raw_byte_limit: int = 128 * 1024 * 1024):
        self.directory = Path(directory)
        self.raw_bundle_limit = int(raw_bundle_limit)
        self.raw_byte_limit = int(raw_byte_limit)
        self.raw_bundles = self.raw_bytes = self.accepted = self.rejected = 0
        self.records = []
        self._lock = threading.Lock()

    def _write(self, path, document):
        from rl_agent.splitfusion_quality_feedback_probe_v1 import gt_evidence as G
        payload = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        G._create_identical_or_fail(path, payload)
        return hashlib.sha256(payload).hexdigest()

    def check_and_record(self, record, *, raw_sources: Mapping[str, Any] | None = None):
        frame = int(record["identity"]["frame_id"])
        stem = f"frame_{frame:010d}"
        # First durable record is written BEFORE any source qualification.
        observed_path = self.directory / f"{stem}.observed.json"
        observed_sha = self._write(observed_path, {"status": "OBSERVED_BEFORE_VALIDATION", **record})
        try:
            validate_source_alignment(record)
        except SourceSynchronizationError as exc:
            self.record_rejection(record, exc, raw_sources=raw_sources, observed_sha=observed_sha)
            raise
        self.accepted += 1
        accepted = {"schema": SYNC_SCHEMA, "frame_id": frame, "status": "EXACT_SOURCE_SYNCHRONIZATION_ACCEPTED",
                    "observed_sha256": observed_sha}
        self._write(self.directory / f"{stem}.accepted.json", accepted)
        self.records.append(accepted)
        return accepted

    def record_rejection(self, record, error, *, raw_sources=None, observed_sha=None):
        """Preserve failures detected by the shared all-tick pose/barrier guard."""
        frame = int(record["identity"]["frame_id"])
        stem = f"frame_{frame:010d}"
        if observed_sha is None:
            observed_sha = self._write(self.directory / f"{stem}.observed.json",
                                       {"status": "OBSERVED_BEFORE_VALIDATION", **record})
        raw_manifest = {}
        elision = "RAW_CAPTURE_LIMIT_REACHED"
        with self._lock:
            if self.raw_bundles < self.raw_bundle_limit and raw_sources:
                images = {role: image for role, image in raw_sources.items() if image is not None}
                required = sum(len(image.raw_data) for image in images.values())
                if self.raw_bytes + required <= self.raw_byte_limit:
                    from rl_agent.splitfusion_quality_feedback_probe_v1 import gt_evidence as G
                    for role, image in images.items():
                        value = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
                        buffer = io.BytesIO()
                        np.save(buffer, value, allow_pickle=False)
                        payload = buffer.getvalue()
                        path = self.directory / f"{stem}.{role}.bgra.npy"
                        G._create_identical_or_fail(path, payload)
                        raw_manifest[role] = {"filename": path.name, "sha256": hashlib.sha256(payload).hexdigest(),
                                              "bytes": len(payload)}
                    self.raw_bundles += 1
                    self.raw_bytes += required
                    elision = None
                else:
                    elision = "RAW_BYTE_LIMIT_REACHED"
            self.rejected += 1
        rejected = {"schema": SYNC_SCHEMA, "frame_id": frame, "status": error.status,
                    "error": str(error), "observed_sha256": observed_sha,
                    "raw_payloads": raw_manifest, "raw_elision": elision}
        self._write(self.directory / f"{stem}.rejected.json", rejected)
        self.records.append(rejected)
        return rejected

    def summary(self):
        return {"schema": SYNC_SCHEMA, "directory": str(self.directory), "accepted": self.accepted,
                "rejected": self.rejected, "raw_bundles": self.raw_bundles, "raw_bytes": self.raw_bytes,
                "raw_bundle_limit": self.raw_bundle_limit, "raw_byte_limit": self.raw_byte_limit,
                "records": list(self.records)}
