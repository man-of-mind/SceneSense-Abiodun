#!/usr/bin/env python3
"""Retained-input wrapper for the pedestrian sensor-contract diagnostic.

This entry point deliberately reuses the policy-corpus collector and changes
neither its model path nor its decoder.  When explicitly enabled it retains
the lossless RGB image, the exact radar tensor consumed by the front half,
the projected raw radar points, camera calibration, and the object logits
produced by the live split path.  The retained files make an identical-input
offline replay possible without widening the scope of the halted corpus run.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# This file is launched by absolute path from the batch runner.  In that mode
# Python places ``data_collection/`` rather than the repository root on
# ``sys.path``; register the root before importing the shared package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_collection import carla_fusion_policy_corpus_collector as policy


base = policy.base
_BUILD_RADAR_TENSOR = base.PoleRadarPipeline.build_tensor
_FRONT_PROCESS = base.CameraSideFusionInference.process
_POLICY_DECODE = policy.decode_objects_with_diagnostics
_POLICY_RUN_BACK_HALF = policy.run_back_half_with_diagnostics
_LOGIT_CAPTURE = threading.local()
_RADAR_LOCK = threading.Lock()
_LATEST_RADAR: Optional[Tuple[int, Dict[str, np.ndarray]]] = None


@dataclass(frozen=True)
class RetentionOptions:
    enabled: bool = False
    root: Optional[Path] = None
    every: int = 1
    max_frames: int = 0


class DiagnosticRetainer:
    """Thread-safe writer for auditable, frame-aligned diagnostic inputs."""

    def __init__(self, options: RetentionOptions) -> None:
        if not options.enabled or options.root is None:
            raise ValueError("diagnostic retainer requires an enabled output root")
        self.options = options
        self.root = options.root
        self.rgb_dir = self.root / "rgb"
        self.radar_dir = self.root / "radar"
        self.logits_dir = self.root / "live_logits"
        for directory in (self.rgb_dir, self.radar_dir, self.logits_dir):
            directory.mkdir(parents=True, exist_ok=False)
        self._lock = threading.Lock()
        self._seen = 0
        self._selected: set[int] = set()
        self._front_rows: Dict[int, Dict[str, object]] = {}
        self._logit_rows: Dict[int, Dict[str, object]] = {}

    def wants_front(self, frame_id: int) -> bool:
        with self._lock:
            self._seen += 1
            if (self._seen - 1) % self.options.every != 0:
                return False
            if self.options.max_frames > 0 and len(self._selected) >= self.options.max_frames:
                return False
            self._selected.add(int(frame_id))
            return True

    def wants_logits(self, frame_id: int) -> bool:
        with self._lock:
            return int(frame_id) in self._selected

    def save_front(
        self,
        *,
        frame_id: int,
        frame_bgr: np.ndarray,
        radar_tensor: np.ndarray,
        radar_frame_id: int,
        radar_points: Mapping[str, np.ndarray],
        camera_matrix: np.ndarray,
        camera_intrinsics_input: np.ndarray,
        display_size: Tuple[int, int],
        model_size: Tuple[int, int],
        carla_timestamp: float,
    ) -> None:
        token = f"frame_{int(frame_id):08d}"
        rgb_path = self.rgb_dir / f"{token}.png"
        radar_path = self.radar_dir / f"{token}.npz"
        if not base.cv2.imwrite(
            str(rgb_path),
            frame_bgr,
            [base.cv2.IMWRITE_PNG_COMPRESSION, 3],
        ):
            raise RuntimeError(f"failed to retain RGB frame {frame_id}")
        with radar_path.open("wb") as stream:
            np.savez_compressed(
                stream,
                radar=np.asarray(radar_tensor, dtype=np.float32),
                camera_matrix=np.asarray(camera_matrix, dtype=np.float64),
                camera_intrinsics_input=np.asarray(
                    camera_intrinsics_input, dtype=np.float64
                ),
                display_size=np.asarray(display_size, dtype=np.int32),
                model_size=np.asarray(model_size, dtype=np.int32),
                camera_frame_id=np.asarray(int(frame_id), dtype=np.int64),
                radar_frame_id=np.asarray(int(radar_frame_id), dtype=np.int64),
                carla_timestamp=np.asarray(float(carla_timestamp), dtype=np.float64),
                **{
                    f"points_{name}": np.asarray(value)
                    for name, value in radar_points.items()
                },
            )
        with self._lock:
            self._front_rows[int(frame_id)] = {
                "frame_id": int(frame_id),
                "radar_frame_id": int(radar_frame_id),
                "carla_timestamp": float(carla_timestamp),
                "rgb_path": str(rgb_path.relative_to(self.root)),
                "radar_path": str(radar_path.relative_to(self.root)),
                "display_width": int(display_size[0]),
                "display_height": int(display_size[1]),
                "model_width": int(model_size[0]),
                "model_height": int(model_size[1]),
                "raw_radar_points": int(
                    len(np.asarray(radar_points.get("u", np.zeros(0))))
                ),
            }

    def save_logits(self, frame_id: int, logits: np.ndarray) -> None:
        if not self.wants_logits(frame_id):
            return
        token = f"frame_{int(frame_id):08d}"
        logits_path = self.logits_dir / f"{token}.npz"
        with logits_path.open("wb") as stream:
            np.savez_compressed(stream, object_logits=np.asarray(logits, dtype=np.float32))
        with self._lock:
            self._logit_rows[int(frame_id)] = {
                "logits_path": str(logits_path.relative_to(self.root)),
                "logits_shape": "x".join(str(value) for value in logits.shape),
            }

    def finalize(self) -> None:
        with self._lock:
            frame_ids = sorted(self._selected)
            rows: List[Dict[str, object]] = []
            for frame_id in frame_ids:
                front = self._front_rows.get(frame_id, {"frame_id": frame_id})
                logits = self._logit_rows.get(frame_id, {})
                rows.append({**front, **logits})
        fieldnames = (
            "frame_id",
            "radar_frame_id",
            "carla_timestamp",
            "rgb_path",
            "radar_path",
            "logits_path",
            "logits_shape",
            "display_width",
            "display_height",
            "model_width",
            "model_height",
            "raw_radar_points",
        )
        with (self.root / "frames.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row.get(name, "") for name in fieldnames})
        complete = sum(
            bool(row.get("rgb_path") and row.get("radar_path") and row.get("logits_path"))
            for row in rows
        )
        manifest = {
            "schema": "pedestrian_on_contract_retained_inputs.v1",
            "status": "complete" if complete == len(rows) else "incomplete",
            "selected_frames": len(rows),
            "complete_aligned_frames": int(complete),
            "retention_every": int(self.options.every),
            "retention_max_frames": int(self.options.max_frames),
            "artifacts": {
                "frame_index": "frames.csv",
                "rgb": "rgb/*.png",
                "radar_and_calibration": "radar/*.npz",
                "live_object_logits": "live_logits/*.npz",
            },
        }
        (self.root / "retention_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


_RETAINER: Optional[DiagnosticRetainer] = None


def _parse_retention_args(
    argv: Sequence[str],
) -> Tuple[RetentionOptions, List[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--retain-diagnostic-inputs", action="store_true")
    parser.add_argument("--retained-input-every", type=int, default=1)
    parser.add_argument("--retained-input-max-frames", type=int, default=0)
    parsed, remaining = parser.parse_known_args(list(argv))
    if parsed.retained_input_every <= 0:
        raise ValueError("retained input interval must be positive")
    if parsed.retained_input_max_frames < 0:
        raise ValueError("retained input maximum must be non-negative")
    metrics_parser = argparse.ArgumentParser(add_help=False)
    metrics_parser.add_argument("--metrics-run-dir", default="")
    inherited, _unknown = metrics_parser.parse_known_args(remaining)
    root = None
    if parsed.retain_diagnostic_inputs:
        if not inherited.metrics_run_dir:
            raise ValueError("retained diagnostic inputs require --metrics-run-dir")
        root = Path(inherited.metrics_run_dir).expanduser().resolve() / "retained_inputs"
    return (
        RetentionOptions(
            enabled=bool(parsed.retain_diagnostic_inputs),
            root=root,
            every=int(parsed.retained_input_every),
            max_frames=int(parsed.retained_input_max_frames),
        ),
        remaining,
    )


def build_tensor_with_retention(
    pipeline: "base.PoleRadarPipeline", **kwargs: object
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    global _LATEST_RADAR
    tensor, points = _BUILD_RADAR_TENSOR(pipeline, **kwargs)
    measurement = kwargs["measurement"]
    with _RADAR_LOCK:
        _LATEST_RADAR = (
            int(measurement.frame),
            {str(name): np.asarray(value).copy() for name, value in points.items()},
        )
    return tensor, points


def process_with_retention(
    inference: "base.CameraSideFusionInference", **kwargs: object
) -> Dict[str, object]:
    if _RETAINER is not None and _RETAINER.wants_front(int(kwargs["frame_id"])):
        with _RADAR_LOCK:
            latest = _LATEST_RADAR
        if latest is None:
            raise RuntimeError("no aligned radar measurement available for retained frame")
        radar_frame_id, radar_points = latest
        _RETAINER.save_front(
            frame_id=int(kwargs["frame_id"]),
            frame_bgr=np.asarray(kwargs["frame_bgr"]),
            radar_tensor=np.asarray(kwargs["radar_tensor"]),
            radar_frame_id=radar_frame_id,
            radar_points=radar_points,
            camera_matrix=np.asarray(kwargs["camera_matrix"]),
            camera_intrinsics_input=np.asarray(kwargs["camera_intrinsics_input"]),
            display_size=tuple(int(value) for value in kwargs["display_size"]),
            model_size=(int(inference.model_w), int(inference.model_h)),
            carla_timestamp=float(kwargs.get("carla_timestamp", 0.0)),
        )
    return _FRONT_PROCESS(inference, **kwargs)


def decode_with_logit_capture(
    object_output: "base.torch.Tensor", **kwargs: object
) -> List[Dict[str, float]]:
    _LOGIT_CAPTURE.current = (
        object_output.detach().to("cpu", dtype=base.torch.float32).numpy().copy()
    )
    return _POLICY_DECODE(object_output, **kwargs)


def run_back_half_with_retention(
    worker: "base.FusionRemoteInferenceWorker", payload: Dict[str, object]
) -> Dict[str, object]:
    _LOGIT_CAPTURE.current = None
    result = _POLICY_RUN_BACK_HALF(worker, payload)
    logits = getattr(_LOGIT_CAPTURE, "current", None)
    if _RETAINER is not None and isinstance(logits, np.ndarray):
        _RETAINER.save_logits(int(result["frame_id"]), logits)
    return result


def main() -> None:
    global _RETAINER
    options, remaining = _parse_retention_args(sys.argv[1:])
    original_argv = list(sys.argv)
    sys.argv = [sys.argv[0], *remaining]
    if options.enabled:
        _RETAINER = DiagnosticRetainer(options)
    base.PoleRadarPipeline.build_tensor = build_tensor_with_retention
    base.CameraSideFusionInference.process = process_with_retention
    policy.decode_objects_with_diagnostics = decode_with_logit_capture
    policy.run_back_half_with_diagnostics = run_back_half_with_retention
    # The inherited manifest should name this diagnostic entry point.
    policy.__file__ = __file__
    try:
        policy.main()
    finally:
        if _RETAINER is not None:
            _RETAINER.finalize()
        sys.argv = original_argv


if __name__ == "__main__":
    main()
