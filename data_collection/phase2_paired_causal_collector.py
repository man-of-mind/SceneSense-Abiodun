#!/usr/bin/env python3
"""Derived one-UE collector for the paired Phase-2 causal pilot.

Two instances (helper and recipient) attach passively to one externally ticked
CARLA world.  This entrypoint reuses the validated policy detector and adds
only causal/runtime instrumentation.  It is not a launcher.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

import yaml

from data_collection import carla_fusion_policy_corpus_collector as policy
from data_collection.phase2_causal_runtime import (
    Phase2CaptureRuntime,
    Phase2RuntimeConfig,
)
from phase2_map_sharing.pilot_contract import load_and_validate_pilot_config


base = policy.base
_RUNTIME: Optional[Phase2CaptureRuntime] = None
_PHASE2_ARGS: Optional[argparse.Namespace] = None
_CONTRACT_CONFIG: Optional[Mapping[str, object]] = None


def _parse_phase2_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--phase2-role", required=True, choices=("helper", "recipient"))
    parser.add_argument("--phase2-trajectory-id", required=True)
    parser.add_argument(
        "--phase2-scenario-role",
        required=True,
        choices=("controlled_positive_occlusion", "matched_benign_negative"),
    )
    parser.add_argument("--phase2-contract-config", required=True)
    parser.add_argument("--phase2-geometry-id", required=True)
    parser.add_argument(
        "--phase2-motion-owner",
        required=True,
        choices=("external_orchestrator",),
    )
    parser.add_argument("--phase2-ready-sentinel", required=True)
    parser.add_argument("--phase2-capture-start-sentinel", required=True)
    parser.add_argument("--phase2-tick-ready", required=True)
    parser.add_argument("--phase2-heartbeat", required=True)
    parser.add_argument("--phase2-start-timeout-s", type=float, default=180.0)
    parser.add_argument("--phase2-tracker-association-gate-m", type=float, default=5.0)
    parser.add_argument("--phase2-tracker-maximum-missed-frames", type=int, default=3)
    parsed, remaining = parser.parse_known_args(list(argv))
    if not parsed.phase2_trajectory_id.strip():
        raise ValueError("--phase2-trajectory-id is required")
    return parsed, remaining


def _load_contract(path: Path) -> Mapping[str, object]:
    load_and_validate_pilot_config(path)
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, Mapping):
        raise ValueError("Phase-2 contract config root must be a mapping")
    return payload


def _require_inherited_contract(argv: Sequence[str]) -> None:
    """Catch dangerous command drift before CARLA/model startup."""

    required_pairs = {
        "--role": "loopback",
        "--async-world": None,
        "--external-sync-ticker": None,
        "--sensor-platform": "ego_vehicle",
        "--ego-spawn-require-exact": None,
        "--ego-freeze": None,
        "--npc-vehicles": "0",
        "--npc-pedestrians": "0",
        "--fps": "10.0",
        "--world-tick-hz": "10.0",
        "--camera-width": "1280",
        "--camera-height": "720",
        "--camera-fov": "120.0",
        "--radar-points-per-second": "200000",
        "--radar-hfov": "120",
        "--radar-rasterizer": "legacy",
        "--radar-raster-radius-px": "4",
        "--radar-temporal-window-frames": "2",
        "--object-score-threshold": "0.05",
        "--object-nms-radius-px": "2",
        "--topk-objects": "120",
        "--quantization-mode": "per_channel_uint8",
        "--entropy-coder": "zlib",
        "--sensor-every-tick": None,
        "--no-spatial-map-stream": None,
        "--disable-semantic-gt": None,
        "--enable-run-logging": None,
        "--headless": None,
    }
    values = list(argv)
    for flag, expected in required_pairs.items():
        if flag not in values:
            raise ValueError(f"Phase-2 collector command is missing {flag}")
        if expected is not None:
            index = values.index(flag)
            if index + 1 >= len(values) or values[index + 1] != expected:
                raise ValueError(
                    f"Phase-2 collector requires {flag} {expected}, got "
                    f"{values[index + 1] if index + 1 < len(values) else '<missing>'}"
                )
    if "--capture-pipeline" in values:
        raise ValueError("Phase-2 paired collector forbids --capture-pipeline")
    if "--no-ego-freeze" in values:
        raise ValueError("paired egos must remain frozen until the shared start barrier")
    if "--ego-route-control" in values:
        index = values.index("--ego-route-control")
        if index + 1 >= len(values) or values[index + 1] != "traffic_manager":
            raise ValueError("paired external-ticker capture requires traffic_manager ego route")


def _runtime() -> Phase2CaptureRuntime:
    if _RUNTIME is None:
        raise RuntimeError("Phase-2 runtime has not been initialized")
    return _RUNTIME


def _move_truth_writer(logger: object) -> None:
    """Relocate the newly created header-only GT stream before any frame."""

    old_path = Path(logger.ground_truth_path)
    logger._ground_truth_file.close()
    truth_dir = Path(logger.run_dir) / "evaluation_truth"
    truth_dir.mkdir(parents=True, exist_ok=True)
    new_path = truth_dir / old_path.name
    if new_path.exists():
        raise FileExistsError(f"evaluation truth stream already exists: {new_path}")
    old_path.replace(new_path)
    logger.ground_truth_path = new_path
    logger._ground_truth_file = new_path.open("a", newline="", encoding="utf-8")
    logger._ground_truth_writer = csv.DictWriter(
        logger._ground_truth_file,
        fieldnames=base.FUSION_OBJECT_GROUND_TRUTH_FIELDS,
    )


def install_phase2_hooks() -> None:
    global _RUNTIME
    if _PHASE2_ARGS is None or _CONTRACT_CONFIG is None:
        raise RuntimeError("Phase-2 arguments/config must be loaded before installing hooks")

    original_logger_init = base.FusionRunLogger.__init__
    original_write_manifest = base.FusionRunLogger.write_manifest
    original_append_predictions = base.FusionRunLogger.append_object_predictions
    original_append_truth = base.FusionRunLogger.append_object_ground_truth
    original_pre_capture = base.on_pre_sensor_capture
    original_build_tensor = base.PoleRadarPipeline.build_tensor
    original_front_process = base.CameraSideFusionInference.process
    original_back_half = base.FusionRemoteInferenceWorker._run_back_half

    def logger_init(logger: object, *args: object, **kwargs: object) -> None:
        original_logger_init(logger, *args, **kwargs)
        _move_truth_writer(logger)
        logger._phase2_pending_frame = None

    def write_manifest(logger: object, *args: object, **kwargs: object) -> None:
        global _RUNTIME
        original_write_manifest(logger, *args, **kwargs)
        if _RUNTIME is not None:
            raise RuntimeError("Phase-2 runtime initialized more than once")
        phase2 = _PHASE2_ARGS
        assert phase2 is not None
        _RUNTIME = Phase2CaptureRuntime(
            Phase2RuntimeConfig(
                role=str(phase2.phase2_role),
                trajectory_id=str(phase2.phase2_trajectory_id),
                scenario_role=str(phase2.phase2_scenario_role),
                run_dir=Path(logger.run_dir),
                ready_sentinel=Path(phase2.phase2_ready_sentinel),
                capture_start_sentinel=Path(phase2.phase2_capture_start_sentinel),
                tick_ready_path=Path(phase2.phase2_tick_ready),
                heartbeat_path=Path(phase2.phase2_heartbeat),
                contract_config_path=Path(phase2.phase2_contract_config).resolve(),
                start_timeout_s=float(phase2.phase2_start_timeout_s),
                association_gate_m=float(phase2.phase2_tracker_association_gate_m),
                maximum_missed_frames=int(
                    phase2.phase2_tracker_maximum_missed_frames
                ),
            ),
            _CONTRACT_CONFIG["raw_retention"],
        )
        manifest_path = Path(logger.manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["phase2_paired_causal"] = {
            "schema": "scenesense.phase2_causal_capture_runtime.v1",
            "trajectory_id": str(phase2.phase2_trajectory_id),
            "scenario_role": str(phase2.phase2_scenario_role),
            "source_role": str(phase2.phase2_role),
            "clock_owner": "external_orchestrator",
            "motion_owner": str(phase2.phase2_motion_owner),
            "geometry_id": str(phase2.phase2_geometry_id),
            "warnings_actuated": False,
            "ground_truth_namespace": "evaluation_truth",
            "runtime_namespace": "runtime",
            "shadow_namespace": "evaluation_shadow",
        }
        if isinstance(manifest.get("output_files"), dict):
            manifest["output_files"]["object_ground_truth_csv"] = str(
                logger.ground_truth_path
            )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def on_pre_capture(*, world: object, anchor_actor: object, args: object, previous_frame_id: int) -> None:
        original_pre_capture(
            world=world,
            anchor_actor=anchor_actor,
            args=args,
            previous_frame_id=previous_frame_id,
        )
        runtime = _runtime()
        # The base loop can enter this hook before the parent creates the
        # shared start sentinel.  Wait first, then sample the stable boundary;
        # the parent is forbidden to tick again until both collectors arm it.
        runtime.await_capture_start()
        stable_previous_frame_id = int(world.get_snapshot().frame)
        # Discard any callback buffered behind the shared start barrier.  Set
        # the filter before publishing tick-ready so the parent cannot race it.
        args.external_capture_minimum_frame = stable_previous_frame_id + 1
        runtime.on_pre_capture(
            world=world,
            anchor_actor=anchor_actor,
            previous_frame_id=stable_previous_frame_id,
        )

    def build_tensor(pipeline: object, *args: object, **kwargs: object):
        tensor, points = original_build_tensor(pipeline, *args, **kwargs)
        measurement = kwargs.get("measurement")
        if measurement is not None and _RUNTIME is not None:
            _RUNTIME.remember_radar_points(int(measurement.frame), points)
        return tensor, points

    def front_process(inference: object, *args: object, **kwargs: object):
        _runtime().record_inputs(
            frame_id=int(kwargs["frame_id"]),
            carla_timestamp=float(kwargs["carla_timestamp"]),
            frame_bgr=kwargs["frame_bgr"],
            radar_tensor=kwargs["radar_tensor"],
            camera_matrix=kwargs["camera_matrix"],
            camera_intrinsics_input=kwargs["camera_intrinsics_input"],
        )
        return original_front_process(inference, *args, **kwargs)

    def back_half(worker: object, payload: dict):
        original_decode = worker.model.decode_outputs

        def capture_decode(*args: object, **kwargs: object):
            outputs = original_decode(*args, **kwargs)
            _runtime().record_logits(int(payload["frame_id"]), outputs)
            return outputs

        worker.model.decode_outputs = capture_decode
        try:
            return original_back_half(worker, payload)
        finally:
            worker.model.decode_outputs = original_decode

    def append_predictions(
        logger: object,
        *,
        elapsed_s: float,
        frame_id: int,
        objects: Sequence[Mapping[str, object]],
    ) -> None:
        original_append_predictions(
            logger,
            elapsed_s=elapsed_s,
            frame_id=frame_id,
            objects=objects,
        )
        # The base loop's elapsed_s uses the same origin for every stream only
        # within a process, so use the captured CARLA timestamp retained by the
        # front hook for paired replay.
        carla_timestamp = _runtime_frame_timestamp(int(frame_id))
        _runtime().record_predictions(
            frame_id=int(frame_id),
            carla_timestamp=carla_timestamp,
            objects=objects,
        )
        logger._phase2_pending_frame = (int(frame_id), float(carla_timestamp))

    def append_truth(logger: object, rows: Sequence[dict]) -> None:
        original_append_truth(logger, rows)
        logger._ground_truth_file.flush()
        pending = getattr(logger, "_phase2_pending_frame", None)
        if pending is None:
            raise RuntimeError("truth append occurred without a pending Phase-2 frame")
        _runtime().mark_frame_complete(int(pending[0]), float(pending[1]))
        logger._phase2_pending_frame = None

    base.FusionRunLogger.__init__ = logger_init
    base.FusionRunLogger.write_manifest = write_manifest
    base.FusionRunLogger.append_object_predictions = append_predictions
    base.FusionRunLogger.append_object_ground_truth = append_truth
    base.on_pre_sensor_capture = on_pre_capture
    base.PoleRadarPipeline.build_tensor = build_tensor
    base.CameraSideFusionInference.process = front_process
    base.FusionRemoteInferenceWorker._run_back_half = back_half
    base.__file__ = __file__


def _runtime_frame_timestamp(frame_id: int) -> float:
    runtime = _runtime()
    try:
        return float(runtime._frame_timestamps.pop(int(frame_id)))
    except (AttributeError, KeyError) as exc:
        raise RuntimeError(f"missing retained CARLA timestamp for frame {frame_id}") from exc


def main(argv: Optional[Sequence[str]] = None) -> None:
    global _PHASE2_ARGS, _CONTRACT_CONFIG
    original_argv = list(sys.argv)
    arguments = list(sys.argv[1:] if argv is None else argv)
    _PHASE2_ARGS, policy_and_base = _parse_phase2_args(arguments)
    _CONTRACT_CONFIG = _load_contract(Path(_PHASE2_ARGS.phase2_contract_config).resolve())
    policy._PEDESTRIAN_OVERLAY, inherited = policy._parse_overlay_args(policy_and_base)
    _require_inherited_contract(inherited)
    sys.argv = [sys.argv[0], *inherited]
    policy._CONTROLLED_TARGET_INFO = None
    policy.install_policy_overlay_hooks()
    install_phase2_hooks()
    status = "failed"
    error: Optional[str] = None
    try:
        base.main()
        status = "complete"
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if _RUNTIME is not None:
            _RUNTIME.close(status=status, error=error)
        policy._destroy_overlay_actors()
        sys.argv = original_argv


if __name__ == "__main__":
    main()
