"""Causal, quota-bounded instrumentation for the paired Phase-2 pilot.

This module contains no CARLA launch logic.  The derived collector entrypoint
calls it at explicit pre-capture, inference, tracking, and truth-completion
boundaries.  Runtime state and evaluation truth are intentionally written to
different directories.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from phase2_map_sharing.causal_contract import (
    CausalDecisionAudit,
    CausalAuditWriter,
    CausalField,
    DecisionRecord,
)
from phase2_map_sharing.retention import (
    RawRetentionBudget,
    RetentionLimits,
    RetentionQuotaExceeded,
)


RUNTIME_SCHEMA = "scenesense.phase2_causal_capture_runtime.v1"
TRACKER_VERSION = "source_local_nearest_cv.v1"
ROLE_NAMES = frozenset({"helper", "recipient"})


def _finite(value: object, fallback: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return result if math.isfinite(result) else float(fallback)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exclusive_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


class _CreateOnlyCsv:
    def __init__(self, path: Path, fields: Sequence[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fields = tuple(str(field) for field in fields)
        self._stream = self.path.open("x", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._stream, fieldnames=self.fields)
        self._writer.writeheader()
        self._stream.flush()

    def write(self, row: Mapping[str, object]) -> None:
        unexpected = set(row) - set(self.fields)
        if unexpected:
            raise ValueError(f"unexpected CSV fields for {self.path.name}: {sorted(unexpected)}")
        self._writer.writerow({field: row.get(field, "") for field in self.fields})
        self._stream.flush()

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()


@dataclass(frozen=True)
class Phase2RuntimeConfig:
    role: str
    trajectory_id: str
    scenario_role: str
    run_dir: Path
    ready_sentinel: Path
    capture_start_sentinel: Path
    tick_ready_path: Path
    heartbeat_path: Path
    contract_config_path: Path
    start_timeout_s: float = 180.0
    association_gate_m: float = 5.0
    maximum_missed_frames: int = 3
    placement_action: str = "SPLIT_FEATURE"
    publication_action: str = "PUBLISH_ALL"
    retention_start_offset_s: float = 0.0
    retention_frame_count: Optional[int] = None
    retention_tier: str = "inputs_plus_logits_window"

    def validate(self) -> None:
        if self.role not in ROLE_NAMES:
            raise ValueError("Phase-2 UE role must be helper or recipient")
        if not self.trajectory_id.strip() or not self.scenario_role.strip():
            raise ValueError("trajectory_id and scenario_role are required")
        if self.start_timeout_s <= 0.0:
            raise ValueError("capture-start timeout must be positive")
        if self.association_gate_m <= 0.0 or self.maximum_missed_frames < 0:
            raise ValueError("causal tracker limits are invalid")
        if self.retention_start_offset_s < 0.0:
            raise ValueError("raw-retention start offset must be non-negative")
        if self.retention_frame_count is not None and self.retention_frame_count <= 0:
            raise ValueError("raw-retention frame count must be positive when set")
        if self.retention_tier not in {
            "inputs_only_window",
            "inputs_plus_logits_window",
        }:
            raise ValueError("unsupported Phase-2 raw-retention tier")


@dataclass
class _Track:
    track_id: str
    class_name: str
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    score: float
    last_timestamp_s: float
    last_frame_id: int
    missed_frames: int = 0


class SourceLocalCausalTracker:
    """Small class-consistent nearest-CV tracker with no truth interface."""

    def __init__(
        self,
        source_role: str,
        *,
        association_gate_m: float = 5.0,
        maximum_missed_frames: int = 3,
    ) -> None:
        if source_role not in ROLE_NAMES:
            raise ValueError("source_role must be helper or recipient")
        self.source_role = source_role
        self.association_gate_m = float(association_gate_m)
        self.maximum_missed_frames = int(maximum_missed_frames)
        self._next_id = 1
        self._tracks: Dict[str, _Track] = {}

    def update(
        self,
        *,
        frame_id: int,
        timestamp_s: float,
        detections: Sequence[Mapping[str, object]],
    ) -> Tuple[list[dict], list[dict]]:
        timestamp = float(timestamp_s)
        normalized: list[dict] = []
        for index, detection in enumerate(detections):
            class_name = str(detection.get("class_name", "object"))
            x = _finite(detection.get("world_x"), float("nan"))
            y = _finite(detection.get("world_y"), float("nan"))
            z = _finite(detection.get("world_z"), 0.0)
            if not math.isfinite(x) or not math.isfinite(y):
                continue
            normalized.append(
                {
                    "detection_index": int(index),
                    "class_name": class_name,
                    "x": x,
                    "y": y,
                    "z": z,
                    "score": _finite(detection.get("score"), 0.0),
                }
            )

        candidates: list[tuple[float, str, int]] = []
        for track_id, track in self._tracks.items():
            dt = max(0.0, timestamp - track.last_timestamp_s)
            predicted_x = track.x + track.vx * dt
            predicted_y = track.y + track.vy * dt
            for detection_index, detection in enumerate(normalized):
                if detection["class_name"] != track.class_name:
                    continue
                distance = math.hypot(
                    float(detection["x"]) - predicted_x,
                    float(detection["y"]) - predicted_y,
                )
                if distance <= self.association_gate_m:
                    candidates.append((distance, track_id, detection_index))

        assignments: Dict[int, tuple[str, float]] = {}
        used_tracks: set[str] = set()
        for distance, track_id, detection_index in sorted(candidates):
            if track_id in used_tracks or detection_index in assignments:
                continue
            used_tracks.add(track_id)
            assignments[detection_index] = (track_id, float(distance))

        associations: list[dict] = []
        updated: Dict[str, _Track] = {}
        for detection_index, detection in enumerate(normalized):
            if detection_index in assignments:
                track_id, distance = assignments[detection_index]
                previous = self._tracks[track_id]
                dt = timestamp - previous.last_timestamp_s
                if dt > 1e-9:
                    vx = (float(detection["x"]) - previous.x) / dt
                    vy = (float(detection["y"]) - previous.y) / dt
                    vz = (float(detection["z"]) - previous.z) / dt
                else:
                    vx, vy, vz = previous.vx, previous.vy, previous.vz
                lifecycle = "matched"
            else:
                track_id = f"{self.source_role}:track:{self._next_id:06d}"
                self._next_id += 1
                distance = float("nan")
                vx = vy = vz = 0.0
                lifecycle = "birth"
            track = _Track(
                track_id=track_id,
                class_name=str(detection["class_name"]),
                x=float(detection["x"]),
                y=float(detection["y"]),
                z=float(detection["z"]),
                vx=float(vx),
                vy=float(vy),
                vz=float(vz),
                score=float(detection["score"]),
                last_timestamp_s=timestamp,
                last_frame_id=int(frame_id),
            )
            updated[track_id] = track
            associations.append(
                {
                    "frame_id": int(frame_id),
                    "timestamp_s": timestamp,
                    "detection_index": int(detection["detection_index"]),
                    "source_track_id": track_id,
                    "association": lifecycle,
                    "association_distance_m": distance,
                    "class_name": track.class_name,
                }
            )

        for track_id, track in self._tracks.items():
            if track_id in used_tracks:
                continue
            track.missed_frames += 1
            if track.missed_frames <= self.maximum_missed_frames:
                updated[track_id] = track
                associations.append(
                    {
                        "frame_id": int(frame_id),
                        "timestamp_s": timestamp,
                        "detection_index": "",
                        "source_track_id": track_id,
                        "association": "missed",
                        "association_distance_m": "",
                        "class_name": track.class_name,
                    }
                )
            else:
                associations.append(
                    {
                        "frame_id": int(frame_id),
                        "timestamp_s": timestamp,
                        "detection_index": "",
                        "source_track_id": track_id,
                        "association": "death",
                        "association_distance_m": "",
                        "class_name": track.class_name,
                    }
                )
        self._tracks = updated
        outputs = [
            {
                "source_track_id": track.track_id,
                "source_role": self.source_role,
                "tracker_version": TRACKER_VERSION,
                "class_name": track.class_name,
                "world_x": track.x,
                "world_y": track.y,
                "world_z": track.z,
                "velocity_x": track.vx,
                "velocity_y": track.vy,
                "velocity_z": track.vz,
                "score": track.score,
                "last_observed_timestamp_s": track.last_timestamp_s,
                "last_observed_frame_id": track.last_frame_id,
                "missed_frames": track.missed_frames,
            }
            for track in sorted(self._tracks.values(), key=lambda item: item.track_id)
        ]
        return outputs, associations


class Phase2CaptureRuntime:
    """Own all Phase-2 runtime writers for one UE process."""

    def __init__(self, config: Phase2RuntimeConfig, retention: Mapping[str, object]) -> None:
        config.validate()
        self.config = config
        self.runtime_dir = config.run_dir / "runtime"
        self.truth_dir = config.run_dir / "evaluation_truth"
        self.shadow_dir = config.run_dir / "evaluation_shadow"
        self.raw_dir = config.run_dir / "retained_inputs"
        for path in (self.runtime_dir, self.truth_dir, self.shadow_dir, self.raw_dir):
            path.mkdir(parents=True, exist_ok=True)
        self._closed = False
        self._capture_started = False
        self._capture_start_clock_s: Optional[float] = None
        self._retention_window_started_at_s: Optional[float] = None
        self._armed_after_frame_id: Optional[int] = None
        self._last_completed_frame_id: Optional[int] = None
        self._last_tracks: list[dict] = []
        self._radar_points: Dict[int, Dict[str, np.ndarray]] = {}
        self._frame_timestamps: Dict[int, float] = {}
        self._write_lock = threading.Lock()
        self._quota_stop_reason: Optional[str] = None
        self._raw_files_written = 0
        self._logits_files_written = 0
        self._retained_input_frames: set[int] = set()

        role_limit = int(retention["maximum_raw_bytes_per_trajectory"]) // 2
        role_limits = RetentionLimits(
            maximum_window_seconds_per_trajectory=float(
                retention["maximum_window_seconds_per_trajectory"]
            ),
            maximum_raw_bytes_per_trajectory=role_limit,
            maximum_raw_bytes_pilot_total=role_limit,
            minimum_free_bytes_after_reservation=int(
                retention["minimum_free_bytes_after_reservation"]
            ),
        )
        self.retention = RawRetentionBudget(config.run_dir, role_limits)
        self.retention_preflight = self.retention.preflight(1)

        self.audit = CausalAuditWriter(self.runtime_dir / "causal_decisions.jsonl")
        self.ego_writer = _CreateOnlyCsv(
            self.runtime_dir / "ego_states.csv",
            (
                "trajectory_id", "source_role", "frame_id", "carla_timestamp",
                "world_x", "world_y", "world_z", "velocity_x", "velocity_y",
                "velocity_z", "yaw_deg", "observed_at_s", "available_at_s",
            ),
        )
        self.detection_writer = _CreateOnlyCsv(
            self.runtime_dir / "final_detections.csv",
            (
                "trajectory_id", "source_role", "frame_id", "carla_timestamp",
                "detection_index", "class_name", "score", "world_x", "world_y",
                "world_z", "inference_done_at_s",
            ),
        )
        self.track_writer = _CreateOnlyCsv(
            self.runtime_dir / "causal_tracks.csv",
            (
                "trajectory_id", "source_role", "frame_id", "carla_timestamp",
                "source_track_id", "tracker_version", "class_name", "score",
                "world_x", "world_y", "world_z", "velocity_x", "velocity_y",
                "velocity_z", "last_observed_frame_id", "missed_frames",
                "last_observed_timestamp_s", "available_at_s",
            ),
        )
        self.association_writer = _CreateOnlyCsv(
            self.runtime_dir / "tracker_associations.csv",
            (
                "trajectory_id", "source_role", "frame_id", "carla_timestamp",
                "detection_index", "source_track_id", "association",
                "association_distance_m", "class_name",
            ),
        )
        self.publication_writer = _CreateOnlyCsv(
            self.runtime_dir / "publication_events.csv",
            (
                "trajectory_id", "source_role", "frame_id", "carla_timestamp",
                "decision_id", "placement_action", "publication_action",
                "track_count", "decision_at_s", "evaluation_only",
            ),
        )
        self.inference_inventory_writer = _CreateOnlyCsv(
            self.runtime_dir / "raw_inference_inventory.csv",
            (
                "trajectory_id", "source_role", "frame_id", "output_shapes_json",
                "object_heatmap_channels", "object_heatmap_cells", "retained_logits",
                "available_at_s",
            ),
        )
        self.tracker = SourceLocalCausalTracker(
            config.role,
            association_gate_m=config.association_gate_m,
            maximum_missed_frames=config.maximum_missed_frames,
        )
        _exclusive_json(
            self.runtime_dir / "runtime_contract.json",
            {
                "schema": RUNTIME_SCHEMA,
                "trajectory_id": config.trajectory_id,
                "scenario_role": config.scenario_role,
                "source_role": config.role,
                "tracker_version": TRACKER_VERSION,
                "tracker": {
                    "association_gate_m": config.association_gate_m,
                    "maximum_missed_frames": config.maximum_missed_frames,
                },
                "placement_action": config.placement_action,
                "publication_action": config.publication_action,
                "retention_start_offset_s": config.retention_start_offset_s,
                "retention_frame_count": config.retention_frame_count,
                "retention_tier": config.retention_tier,
                "contract_config_path": str(config.contract_config_path),
                "retention_preflight": self.retention_preflight,
            },
        )
        self._write_ready_sentinel()

    def _write_ready_sentinel(self) -> None:
        _exclusive_json(
            self.config.ready_sentinel,
            {
                "schema": RUNTIME_SCHEMA,
                "status": "ready_waiting_for_capture_start",
                "trajectory_id": self.config.trajectory_id,
                "source_role": self.config.role,
                "pid": os.getpid(),
                "written_at_s": time.time(),
            },
        )

    def _await_capture_start(self) -> None:
        if self._capture_started:
            return
        deadline = time.monotonic() + self.config.start_timeout_s
        while time.monotonic() < deadline:
            if self.config.capture_start_sentinel.is_file():
                self._capture_started = True
                return
            time.sleep(0.01)
        raise RuntimeError(
            f"timed out waiting for capture-start sentinel: {self.config.capture_start_sentinel}"
        )

    def await_capture_start(self) -> None:
        """Wait until the external owner freezes the first capture boundary."""

        self._await_capture_start()

    @staticmethod
    def _ego_state(anchor_actor: object) -> dict:
        transform = anchor_actor.get_transform()
        velocity = anchor_actor.get_velocity()
        return {
            "world_x": float(transform.location.x),
            "world_y": float(transform.location.y),
            "world_z": float(transform.location.z),
            "velocity_x": float(velocity.x),
            "velocity_y": float(velocity.y),
            "velocity_z": float(velocity.z),
            "yaw_deg": float(transform.rotation.yaw),
        }

    def on_pre_capture(self, *, world: object, anchor_actor: object, previous_frame_id: int) -> None:
        self._await_capture_start()
        after_frame_id = int(previous_frame_id)
        if self._armed_after_frame_id is not None:
            expected_completed_frame = self._armed_after_frame_id + 1
            if self._last_completed_frame_id != expected_completed_frame:
                raise RuntimeError(
                    "collector entered another pre-capture decision before completing "
                    f"armed CARLA frame {expected_completed_frame}"
                )
            if after_frame_id != self._last_completed_frame_id:
                raise RuntimeError(
                    "non-consecutive external capture boundary: "
                    f"completed={self._last_completed_frame_id}, after={after_frame_id}"
                )
        decision_at_s = time.perf_counter()
        snapshot = world.get_snapshot()
        if int(snapshot.frame) != after_frame_id:
            raise RuntimeError(
                "CARLA advanced during the pre-capture decision: "
                f"expected frame {after_frame_id}, observed {int(snapshot.frame)}"
            )
        carla_timestamp = float(snapshot.timestamp.elapsed_seconds)
        if self._capture_start_clock_s is None:
            self._capture_start_clock_s = carla_timestamp
        state = self._ego_state(anchor_actor)
        state_field_name = "helper_state" if self.config.role == "helper" else "recipient_state"
        state_source = (
            "helper_localization" if self.config.role == "helper" else "recipient_localization"
        )
        decision_id = (
            f"{self.config.trajectory_id}:{self.config.role}:placement:"
            f"after-{int(previous_frame_id)}"
        )
        decision = DecisionRecord(
            trajectory_id=self.config.trajectory_id,
            arm_id="selected_runtime",
            decision_id=decision_id,
            decision_stage="placement",
            decision_at_s=decision_at_s,
            clock_id="host_perf_counter",
            action=self.config.placement_action,
        )
        fields = [
            CausalField(
                field_name=state_field_name,
                value={**state, "frame_id": int(snapshot.frame)},
                source_stage=state_source,
                observed_at_s=decision_at_s,
                available_at_s=decision_at_s,
                consuming_decision_id=decision_id,
                consuming_decision_stage="placement",
                clock_id="host_perf_counter",
                arm_id="selected_runtime",
            )
        ]
        if self._last_tracks:
            fields.append(
                CausalField(
                    field_name="prior_source_track_summary",
                    value={
                        "track_count": len(self._last_tracks),
                        "source_track_ids": [
                            str(track["source_track_id"]) for track in self._last_tracks
                        ],
                    },
                    source_stage="causal_tracker",
                    observed_at_s=decision_at_s,
                    available_at_s=decision_at_s,
                    consuming_decision_id=decision_id,
                    consuming_decision_stage="placement",
                    clock_id="host_perf_counter",
                    arm_id="selected_runtime",
                )
            )
        self.audit.write(CausalDecisionAudit(decision=decision, fields=tuple(fields)))
        self.ego_writer.write(
            {
                "trajectory_id": self.config.trajectory_id,
                "source_role": self.config.role,
                "frame_id": int(snapshot.frame),
                "carla_timestamp": carla_timestamp,
                **state,
                "observed_at_s": decision_at_s,
                "available_at_s": decision_at_s,
            }
        )
        self._write_tick_ready(
            after_frame_id=after_frame_id,
            decision_id=decision_id,
        )

    def _write_tick_ready(self, *, after_frame_id: int, decision_id: str) -> None:
        payload = {
            "schema": "scenesense.phase2_capture_barrier.v2",
            "status": "armed_for_next_frame",
            "trajectory_id": self.config.trajectory_id,
            "source_role": self.config.role,
            "after_frame_id": int(after_frame_id),
            "minimum_capture_frame": int(after_frame_id) + 1,
            "placement_decision_id": str(decision_id),
            "updated_at_s": time.time(),
        }
        temporary = self.config.tick_ready_path.with_suffix(".tmp")
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(payload, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.config.tick_ready_path)
        self._armed_after_frame_id = int(after_frame_id)

    def remember_radar_points(self, frame_id: int, points: Mapping[str, np.ndarray]) -> None:
        if not self._capture_started:
            return
        self._radar_points[int(frame_id)] = {
            str(key): np.asarray(value).copy() for key, value in points.items()
        }
        while len(self._radar_points) > 4:
            oldest = min(self._radar_points)
            del self._radar_points[oldest]

    def _quota_write(self, path: Path, payload: bytes, at_s: float) -> bool:
        with self._write_lock:
            if self._quota_stop_reason is not None:
                return False
            trajectory_key = f"{self.config.trajectory_id}:{self.config.role}"
            try:
                permit = self.retention.authorize_write(
                    trajectory_key, len(payload), at_s
                )
            except RetentionQuotaExceeded as exc:
                self._quota_stop_reason = exc.reason
                return False
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
            except Exception:
                self.retention.cancel_write(permit)
                raise
            self.retention.record_write_complete(permit)
            return True

    def _retain_frame(self, frame_id: int, carla_timestamp: float) -> bool:
        """Select the registered bounded window without dropping light logs.

        The capture barrier defines time zero.  Heavy retention begins at the
        first sensor frame on or after the configured offset and is bounded by
        an exact frame count.  This avoids the inclusive-endpoint ambiguity of
        treating a 4 s, 10 Hz window as a floating-time interval (40 vs 41
        frames).
        """

        if self._capture_start_clock_s is None:
            return False
        if int(frame_id) in self._retained_input_frames:
            return True
        if (
            self.config.retention_frame_count is not None
            and len(self._retained_input_frames) >= self.config.retention_frame_count
        ):
            return False
        requested_start_s = (
            self._capture_start_clock_s + self.config.retention_start_offset_s
        )
        if float(carla_timestamp) + 1e-9 < requested_start_s:
            return False
        if self._retention_window_started_at_s is None:
            self._retention_window_started_at_s = float(carla_timestamp)
            self.retention.start_window(
                f"{self.config.trajectory_id}:{self.config.role}",
                self._retention_window_started_at_s,
            )
        return True

    def record_inputs(
        self,
        *,
        frame_id: int,
        carla_timestamp: float,
        frame_bgr: np.ndarray,
        radar_tensor: np.ndarray,
        camera_matrix: np.ndarray,
        camera_intrinsics_input: np.ndarray,
    ) -> None:
        if not self._capture_started or self._capture_start_clock_s is None:
            return
        self._frame_timestamps[int(frame_id)] = float(carla_timestamp)
        points = self._radar_points.pop(int(frame_id), {})
        if not self._retain_frame(int(frame_id), float(carla_timestamp)):
            return
        buffer = io.BytesIO()
        arrays: Dict[str, np.ndarray] = {
            "frame_bgr": np.asarray(frame_bgr),
            "radar_tensor": np.asarray(radar_tensor),
            "camera_matrix": np.asarray(camera_matrix),
            "camera_intrinsics_input": np.asarray(camera_intrinsics_input),
            "frame_id": np.asarray([int(frame_id)], dtype=np.int64),
            "carla_timestamp": np.asarray([float(carla_timestamp)], dtype=np.float64),
        }
        for key, value in points.items():
            arrays[f"radar_points__{key}"] = np.asarray(value)
        np.savez_compressed(buffer, **arrays)
        if self._quota_write(
            self.raw_dir / f"frame_{int(frame_id):08d}_inputs.npz",
            buffer.getvalue(),
            float(carla_timestamp),
        ):
            self._raw_files_written += 1
            self._retained_input_frames.add(int(frame_id))

    def record_logits(self, frame_id: int, outputs: Mapping[str, object]) -> None:
        if not self._capture_started:
            return
        retained_input = int(frame_id) in self._retained_input_frames
        arrays: Dict[str, np.ndarray] = {}

        def visit(prefix: str, value: object) -> None:
            if hasattr(value, "detach") and hasattr(value, "cpu"):
                arrays[prefix] = value.detach().cpu().numpy()
            elif isinstance(value, np.ndarray):
                arrays[prefix] = value
            elif isinstance(value, Mapping):
                for key, item in value.items():
                    visit(f"{prefix}__{key}" if prefix else str(key), item)

        visit("", outputs)
        if not arrays:
            raise RuntimeError("inference output contained no retainable logits")
        object_tensor = arrays.get("object")
        heatmap_channels = 0
        heatmap_cells = 0
        if object_tensor is not None and object_tensor.ndim in (3, 4):
            channel_axis = 1 if object_tensor.ndim == 4 else 0
            heatmap_channels = min(2, int(object_tensor.shape[channel_axis]))
            height, width = int(object_tensor.shape[-2]), int(object_tensor.shape[-1])
            batch = int(object_tensor.shape[0]) if object_tensor.ndim == 4 else 1
            heatmap_cells = batch * heatmap_channels * height * width
        retained = False
        if retained_input and self.config.retention_tier == "inputs_plus_logits_window":
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **arrays)
            retained = self._quota_write(
                self.raw_dir / f"frame_{int(frame_id):08d}_logits.npz",
                buffer.getvalue(),
                float(self._frame_timestamps[int(frame_id)]),
            )
        self.inference_inventory_writer.write(
            {
                "trajectory_id": self.config.trajectory_id,
                "source_role": self.config.role,
                "frame_id": int(frame_id),
                "output_shapes_json": json.dumps(
                    {key: list(value.shape) for key, value in sorted(arrays.items())},
                    sort_keys=True,
                ),
                "object_heatmap_channels": heatmap_channels,
                "object_heatmap_cells": heatmap_cells,
                "retained_logits": int(retained),
                "available_at_s": time.perf_counter(),
            }
        )
        if retained:
            self._logits_files_written += 1

    def record_predictions(
        self,
        *,
        frame_id: int,
        carla_timestamp: float,
        objects: Sequence[Mapping[str, object]],
    ) -> None:
        inference_done_at_s = time.perf_counter()
        for index, obj in enumerate(objects):
            self.detection_writer.write(
                {
                    "trajectory_id": self.config.trajectory_id,
                    "source_role": self.config.role,
                    "frame_id": int(frame_id),
                    "carla_timestamp": float(carla_timestamp),
                    "detection_index": int(index),
                    "class_name": str(obj.get("class_name", "object")),
                    "score": _finite(obj.get("score"), 0.0),
                    "world_x": _finite(obj.get("world_x"), 0.0),
                    "world_y": _finite(obj.get("world_y"), 0.0),
                    "world_z": _finite(obj.get("world_z"), 0.0),
                    "inference_done_at_s": inference_done_at_s,
                }
            )
        tracks, associations = self.tracker.update(
            frame_id=int(frame_id),
            timestamp_s=float(carla_timestamp),
            detections=objects,
        )
        for association in associations:
            self.association_writer.write(
                {
                    "trajectory_id": self.config.trajectory_id,
                    "source_role": self.config.role,
                    "frame_id": int(frame_id),
                    "carla_timestamp": float(carla_timestamp),
                    "detection_index": association["detection_index"],
                    "source_track_id": association["source_track_id"],
                    "association": association["association"],
                    "association_distance_m": association["association_distance_m"],
                    "class_name": association["class_name"],
                }
            )
        for track in tracks:
            self.track_writer.write(
                {
                    "trajectory_id": self.config.trajectory_id,
                    "source_role": self.config.role,
                    "frame_id": int(frame_id),
                    "carla_timestamp": float(carla_timestamp),
                    **track,
                    "available_at_s": inference_done_at_s,
                }
            )
        self._last_tracks = tracks
        decision_at_s = time.perf_counter()
        decision_id = (
            f"{self.config.trajectory_id}:{self.config.role}:publication:{int(frame_id)}"
        )
        decision = DecisionRecord(
            trajectory_id=self.config.trajectory_id,
            arm_id="selected_runtime",
            decision_id=decision_id,
            decision_stage="publication",
            decision_at_s=decision_at_s,
            clock_id="host_perf_counter",
            action=self.config.publication_action,
        )
        fields = (
            CausalField(
                field_name="current_inference_result",
                value={"final_detection_count": len(objects), "frame_id": int(frame_id)},
                source_stage="selected_inference",
                observed_at_s=inference_done_at_s,
                available_at_s=inference_done_at_s,
                consuming_decision_id=decision_id,
                consuming_decision_stage="publication",
                clock_id="host_perf_counter",
                arm_id="selected_runtime",
            ),
            CausalField(
                field_name="current_causal_tracks",
                value={
                    "track_count": len(tracks),
                    "source_track_ids": [str(track["source_track_id"]) for track in tracks],
                },
                source_stage="causal_tracker",
                observed_at_s=inference_done_at_s,
                available_at_s=inference_done_at_s,
                consuming_decision_id=decision_id,
                consuming_decision_stage="publication",
                clock_id="host_perf_counter",
                arm_id="selected_runtime",
            ),
        )
        self.audit.write(CausalDecisionAudit(decision=decision, fields=fields))
        self.publication_writer.write(
            {
                "trajectory_id": self.config.trajectory_id,
                "source_role": self.config.role,
                "frame_id": int(frame_id),
                "carla_timestamp": float(carla_timestamp),
                "decision_id": decision_id,
                "placement_action": self.config.placement_action,
                "publication_action": self.config.publication_action,
                "track_count": len(tracks),
                "decision_at_s": decision_at_s,
                "evaluation_only": 0,
            }
        )

    def mark_frame_complete(self, frame_id: int, carla_timestamp: float) -> None:
        if self._armed_after_frame_id is None:
            raise RuntimeError("frame completed before the collector armed a capture tick")
        expected_frame_id = self._armed_after_frame_id + 1
        if int(frame_id) != expected_frame_id:
            raise RuntimeError(
                "collector completed an unexpected CARLA frame: "
                f"armed={expected_frame_id}, completed={int(frame_id)}"
            )
        if self._last_completed_frame_id is not None and int(frame_id) <= self._last_completed_frame_id:
            raise RuntimeError(
                f"collector completed CARLA frame {int(frame_id)} more than once"
            )
        payload = {
            "schema": RUNTIME_SCHEMA,
            "status": "frame_complete",
            "trajectory_id": self.config.trajectory_id,
            "source_role": self.config.role,
            "frame_id": int(frame_id),
            "carla_timestamp": float(carla_timestamp),
            "updated_at_s": time.time(),
        }
        temporary = self.config.heartbeat_path.with_suffix(".tmp")
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(payload, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.config.heartbeat_path)
        self._last_completed_frame_id = int(frame_id)

    def close(self, *, status: str, error: Optional[str] = None) -> None:
        if self._closed:
            return
        self._closed = True
        if self._capture_started:
            key = f"{self.config.trajectory_id}:{self.config.role}"
            window = self.retention.windows.get(key)
            if window is not None and window.status == "active":
                self.retention.finish_window(key, "collector_closed")
        self.audit.close()
        for writer in (
            self.ego_writer,
            self.detection_writer,
            self.track_writer,
            self.association_writer,
            self.publication_writer,
            self.inference_inventory_writer,
        ):
            writer.close()
        summary_path = self.config.run_dir / "phase2_runtime_summary.json"
        _exclusive_json(
            summary_path,
            {
                "schema": RUNTIME_SCHEMA,
                "status": str(status),
                "error": error,
                "trajectory_id": self.config.trajectory_id,
                "scenario_role": self.config.scenario_role,
                "source_role": self.config.role,
                "raw_input_files_written": self._raw_files_written,
                "logits_files_written": self._logits_files_written,
                "retained_frame_ids": sorted(self._retained_input_frames),
                "retention_window_started_at_s": self._retention_window_started_at_s,
                "retention_tier": self.config.retention_tier,
                "quota_stop_reason": self._quota_stop_reason,
                "retention": self.retention.summary(),
            },
        )
        artifact_paths = sorted(
            path
            for path in self.config.run_dir.rglob("*")
            if path.is_file()
            and path.name != "artifact_manifest.json"
            and path != self.config.heartbeat_path
        )
        _exclusive_json(
            self.config.run_dir / "artifact_manifest.json",
            {
                "schema": "scenesense.phase2_artifact_manifest.v1",
                "trajectory_id": self.config.trajectory_id,
                "source_role": self.config.role,
                "files": [
                    {
                        "path": str(path.relative_to(self.config.run_dir)),
                        "bytes": path.stat().st_size,
                        "sha256": _hash_file(path),
                    }
                    for path in artifact_paths
                ],
            },
        )
