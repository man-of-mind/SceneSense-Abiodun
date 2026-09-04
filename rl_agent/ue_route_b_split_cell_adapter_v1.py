#!/usr/bin/env python3
"""Qualified Route B adapter for one fixed UE split-inference campaign cell.

Route B owns the ego, traffic, controller, Traffic Manager, and every CARLA
tick.  This module only replaces ``drive_one_loop_with_traffic`` long enough
to attach two passive sensors and pass a ``SamplingWorld`` facade to the
unchanged density runner.  Prepared frames are handed to a bounded worker;
the route thread never waits for model or transport work.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ADAPTER_SCHEMA = "scenesense.ue_route_b_split_cell_adapter.v1"
EXPECTED_OUTPUTS = (
    "per_frame_metrics.csv",
    "radio_trace.csv",
    "map_feedback.csv",
    "perception_metrics.csv",
    "resolved_config.yaml",
    "RESULTS_SUMMARY.json",
    "manifest.json",
)
PER_FRAME_FIELDS = (
    "cell_id", "action_id", "network_profile_id", "stream_id", "capture_id",
    "frame_id", "route_tick", "carla_timestamp", "capture_wall_s",
    "service_deadline_at", "prepare_status", "processing_late", "queue_depth",
    "queue_wait_ms", "front_ms", "payload_bytes", "payload_bytes_uncompressed",
    "payload_chunks", "window_sweeps", "window_callbacks", "window_returns",
    "window_span_s", "raw_radar_return_count", "raw_radar_valid_range_count",
    "raw_radar_closing_count", "raw_radar_receding_count",
    "raw_radar_stationary_count", "raw_radar_min_range_m",
    "raw_radar_mean_range_m", "radar_projected_points", "ego_speed_mps",
    "error",
)
EDGE_SEGMENTATION_EVIDENCE_FLAG = "--edge-segmentation-evidence-dir"


class AdapterError(RuntimeError):
    """A fixed cell contract or runtime stage failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AdapterError(message)


def repo_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"YAML root must be a mapping: {path}")
    return value


def write_json_create_only(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def action_row(campaign: Mapping[str, Any], action_id: str) -> dict[str, str]:
    registry = repo_path(str(campaign["actions"]["technical_registry_csv"]))
    require(registry.is_file(), f"technical action registry missing: {registry}")
    require(
        sha256_file(registry) == str(campaign["actions"]["technical_registry_sha256"]),
        "technical action registry hash drift",
    )
    with registry.open(newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    matches = [row for row in rows if row.get("profile_id") == action_id]
    require(len(matches) == 1, f"resolved action is not unique in registry: {action_id}")
    row = matches[0]
    require(
        row.get("certification_status") == campaign["actions"]["required_certification_status"],
        f"resolved action is not certified: {action_id}",
    )
    return row


def validate_resolved_contract(
    resolved_path: Path,
    attempt_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    resolved = load_yaml(resolved_path)
    require(resolved.get("schema") == "scenesense.ue_288_cell_resolved.v1", "resolved cell schema drift")
    campaign = resolved.get("campaign")
    cell = resolved.get("cell")
    require(isinstance(campaign, dict) and isinstance(cell, dict), "resolved cell lacks campaign/cell mappings")
    contract = campaign.get("measurement_contract")
    require(isinstance(contract, dict), "resolved cell lacks measurement contract")
    require(resolved.get("measurement_contract") == contract, "resolved measurement contract stamp drift")
    require(float(contract["match_distance_m"]) == 3.0, "primary match distance must be 3.0 m")
    require(float(contract["max_gt_distance_m"]) == 40.0, "GT distance gate must be 40.0 m")
    require(float(contract["min_gt_area_px"]) == 12.0, "GT area gate must be 12.0 px")
    require(Path(str(resolved.get("attempt_dir"))).resolve() == attempt_dir, "attempt directory mismatch")
    require((attempt_dir / "resolved_config.yaml").resolve() == resolved_path, "resolved config must be in attempt directory")
    route = campaign["route_b"]
    require(route["density"] == "traffic_50_50", "adapter accepts only traffic_50_50")
    require(route["hybrid_physics"] is False, "hybrid physics is forbidden")
    require(route["loops_per_process"] == 1, "adapter accepts one Route B loop only")
    require(route["allow_roadblock_clearing"] is True, "stationary-roadblock clearing must be enabled")
    require(route["forced_overtaking"] is False and route["maximum_overtakes"] == 0, "forced overtaking is forbidden")
    require(route["carla_quality"] == "Epic" and route["no_rendering_mode"] is False, "Epic rendering contract drift")
    require(campaign["network"]["catch_up_policy"] == "SKIP_OBSOLETE_NEVER_BURST", "target-SNR catch-up drift")
    require(float(campaign["network"]["clean_restore_noise_power_db"]) == -50.0, "RFsim restore drift")
    require(tuple(campaign["cell"]["expected_outputs"]) == EXPECTED_OUTPUTS, "registered output set drift")
    row = action_row(campaign, str(cell["action_id"]))
    require(row["model_family"] == str(cell["model_family"]), "resolved model family/action mismatch")
    require(row["entropy_coder"] == "zstd", "certified action transport must remain zstd")
    return resolved, campaign, row


def launcher_binding(campaign: Mapping[str, Any], row: Mapping[str, str]) -> dict[str, Any]:
    launcher = repo_path(str(campaign["runtime"]["oai_registered_profile_launcher"]))
    registry = ROOT / "rl_agent/registries/ue_split_profile_registry_v1/ue_split_profile_registry.csv"
    env = os.environ.copy()
    env.update(
        {
            "UE_SPLIT_PROFILE_ID": str(row["profile_id"]),
            "UE_SPLIT_PROFILE_REGISTRY_CSV": str(registry),
            "UE_PROFILE_BINDING_ONLY": "1",
        }
    )
    completed = subprocess.run(
        [str(launcher)], cwd=str(ROOT), env=env, check=False,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    require(completed.returncode == 0, f"registered launcher binding failed: {completed.stderr.strip()}")
    binding = json.loads(completed.stdout)
    require(binding.get("profile_id") == row["profile_id"], "launcher resolved a different action")
    require(binding.get("profile_identity", {}).get("checkpoint_sha256") == row["checkpoint_sha256"], "launcher checkpoint binding drift")
    require(binding.get("profile_identity", {}).get("quantization_mode") == row["quantization_mode"], "launcher quantizer binding drift")
    require(str(binding.get("profile_identity", {}).get("roi_drop_fraction")) == str(row["roi_drop_fraction"]), "launcher q binding drift")
    return binding


def attach_oai(campaign: Mapping[str, Any], row: Mapping[str, str]) -> None:
    launcher = repo_path(str(campaign["runtime"]["oai_registered_profile_launcher"]))
    registry = ROOT / "rl_agent/registries/ue_split_profile_registry_v1/ue_split_profile_registry.csv"
    env = os.environ.copy()
    env.update(
        {
            "UE_SPLIT_PROFILE_ID": str(row["profile_id"]),
            "UE_SPLIT_PROFILE_REGISTRY_CSV": str(registry),
            "ATTACH_ONLY": "1",
            "RFSIM_CHANMOD": "1",
            "ENABLE_SOFTMODEM_TTRACER": "0",
            "RECORD_GNB": "0",
        }
    )
    completed = subprocess.run(
        [str(launcher)], cwd=str(ROOT), env=env, check=False,
        stdin=subprocess.DEVNULL,
    )
    require(completed.returncode == 0, f"registered OAI attach failed rc={completed.returncode}")


def arg_value(argv: Sequence[str], flag: str) -> str:
    try:
        index = list(argv).index(flag)
        return str(argv[index + 1])
    except (ValueError, IndexError) as exc:
        raise AdapterError(f"registered binding lacks {flag}") from exc


def start_tail(
    campaign: Mapping[str, Any],
    binding: Mapping[str, Any],
    *,
    stream_id: str,
    spatial_map_port: int,
    edge_evidence_container_dir: Path,
) -> None:
    edge_args = [str(value) for value in binding["edge_args"]]
    filtered: list[str] = []
    skip_flags = {"--fusion-checkpoint", "--quantization-mode", "--entropy-coder"}
    index = 0
    while index < len(edge_args):
        if edge_args[index] in skip_flags:
            index += 2
        else:
            filtered.append(edge_args[index])
            index += 1
    filtered.extend(
        [
            "--uplink-only-spatial-map", "--edge-result-mode", "none",
            "--edge-receive-queue-size", "32", "--spatial-map-stream",
            "--spatial-map-host", "127.0.0.1", "--spatial-map-port", str(spatial_map_port),
            "--spatial-map-stream-id", stream_id, "--camera-resolution", "custom",
            "--camera-width", "1280", "--camera-height", "720", "--camera-fov", "120",
            EDGE_SEGMENTATION_EVIDENCE_FLAG, str(edge_evidence_container_dir),
        ]
    )
    env = os.environ.copy()
    env.update(
        {
            "FUSION_BACK_REMOTE_HOST": "10.0.0.2",
            "FUSION_BACK_REMOTE_HOST_1": "10.0.0.2",
            "FUSION_BACK_DUAL": "0",
            # The wrapper imports and runs the unchanged certified runtime. Its
            # only hook queues the already-decoded mask after normal map
            # publication; registered feature/map packets remain unchanged.
            "FUSION_BACK_SCRIPT": "/work/abiodun/rl_agent/ue_route_b_split_cell_adapter_v1.py",
            "FUSION_BACK_CHECKPOINT": str(binding["checkpoint_paths"]["container"]),
            "FUSION_QUANTIZATION_MODE": arg_value(edge_args, "--quantization-mode"),
            "FUSION_ENTROPY_CODER": arg_value(edge_args, "--entropy-coder"),
            "FUSION_BACK_LOG_EVERY": "50",
            "FUSION_REMOTE_PORT_1": "51002",
            "FUSION_REMOTE_SOURCE_PORT_1": "51013",
            "FUSION_CAMERA_RESULT_PORT_1": "51004",
            "FUSION_BACK_EXTRA_ARGS": " ".join(filtered),
        }
    )
    helper = ROOT / "scripts/receiver_container_fusion_back_up.sh"
    completed = subprocess.run([str(helper)], cwd=str(ROOT), env=env, check=False, stdin=subprocess.DEVNULL)
    require(completed.returncode == 0, f"registered tail helper failed rc={completed.returncode}")


def stop_tail() -> bool:
    completed = subprocess.run(
        [str(ROOT / "scripts/receiver_container_down.sh")], cwd=str(ROOT), check=False,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return completed.returncode == 0


def tail_running() -> bool:
    completed = subprocess.run(
        ["sudo", "docker", "inspect", "-f", "{{.State.Running}}", "oai-perception-rx"],
        cwd=str(ROOT), check=False, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    return completed.returncode == 0 and completed.stdout.strip() == "true"


def start_map_process(
    campaign: Mapping[str, Any],
    *,
    temporary_dir: Path,
    action_id: str,
    carla_host: str,
    carla_port: int,
    api_port: int,
    udp_port: int,
    feedback_port: int,
) -> subprocess.Popen[bytes]:
    runtime = repo_path(str(campaign["runtime"]["map_install_runtime"]))
    argv = [
        sys.executable, str(runtime), "--api-host", "127.0.0.1", "--api-port", str(api_port),
        "--udp-host", "127.0.0.1", "--udp-port", str(udp_port),
        "--install-feedback-host", "127.0.0.1", "--install-feedback-port", str(feedback_port),
        "--default-action-id", action_id, "--carla-host", carla_host,
        "--carla-port", str(carla_port), "--output-dir", str(temporary_dir / "map"),
        "--focus-follow-stream-id", "unused",
        "--installed-frame-history-size",
        str(int(campaign["measurement_contract"]["installed_frame_history_size"])),
    ]
    process = subprocess.Popen(argv, cwd=str(ROOT), stdin=subprocess.DEVNULL)
    deadline = time.monotonic() + 30.0
    url = f"http://127.0.0.1:{api_port}/healthz"
    while time.monotonic() < deadline:
        require(process.poll() is None, "per-cell map process exited during startup")
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                if response.status == 200:
                    return process
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.5)
    process.terminate()
    raise AdapterError("per-cell map process did not become ready")


def stop_process(process: subprocess.Popen[Any] | None, timeout_s: float = 15.0) -> bool:
    if process is None:
        return True
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
    return process.poll() is not None


def start_target_snr(
    campaign: Mapping[str, Any],
    *,
    campaign_path: Path,
    profile_id: str,
    temporary_dir: Path,
) -> tuple[subprocess.Popen[bytes], Path, Path]:
    runtime = repo_path(str(campaign["runtime"]["target_snr_runtime"]))
    output = temporary_dir / "radio_trace.csv"
    stop_file = temporary_dir / "stop_target_snr"
    process = subprocess.Popen(
        [sys.executable, str(runtime), "--campaign", str(campaign_path),
         "--profile-id", profile_id, "--output", str(output), "--stop-file", str(stop_file)],
        cwd=str(ROOT), stdin=subprocess.DEVNULL,
    )
    return process, output, stop_file


def stop_target_snr(
    process: subprocess.Popen[Any] | None,
    output: Path,
    stop_file: Path,
    destination: Path,
) -> bool:
    if process is None:
        return False
    stop_file.touch(exist_ok=False)
    try:
        rc = process.wait(timeout=15.0)
    except subprocess.TimeoutExpired:
        process.send_signal(signal.SIGINT)
        try:
            rc = process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            process.kill()
            rc = process.wait(timeout=5.0)
    if output.is_file() and not destination.exists():
        os.replace(output, destination)
    summary = output.with_suffix(output.suffix + ".summary.json")
    restored = False
    if summary.is_file():
        value = json.loads(summary.read_text(encoding="utf-8"))
        restored = bool(value.get("clean_restore_verified"))
    return rc == 0 and restored and destination.is_file()


def split_args(front_args: Sequence[str], runtime: Any) -> argparse.Namespace:
    argv = [
        "split-runtime", "--role", "front", "--bind-host", "10.0.0.2",
        "--remote-host", "192.168.70.140", "--camera-source-port", "51001",
        "--remote-port", "51002", "--remote-source-port", "51013",
        "--camera-result-port", "51004", "--front-device", "cuda",
        "--camera-resolution", "custom", "--camera-width", "1280",
        "--camera-height", "720", "--camera-fov", "120", "--fps", "10",
        "--world-tick-hz", "20", "--sensor-every-tick", "--headless",
        *[str(value) for value in front_args],
    ]
    previous = sys.argv
    try:
        sys.argv = argv
        parsed = runtime.parse_args()
    finally:
        sys.argv = previous
    return parsed


def mean_or_nan(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def _yaw_deg(row: Mapping[str, Any]) -> float:
    if row.get("model_yaw_deg") not in (None, ""):
        return float(row["model_yaw_deg"])
    if row.get("yaw_deg") not in (None, ""):
        return float(row["yaw_deg"])
    return math.degrees(
        math.atan2(float(row.get("yaw_sin", 0.0)), float(row.get("yaw_cos", 1.0)))
    )


def oriented_footprint_iou(prediction: Mapping[str, Any], truth: Mapping[str, Any]) -> float:
    """Intersection-over-union of two oriented world-XY footprints."""
    import cv2

    def corners(row: Mapping[str, Any]) -> np.ndarray:
        length = float(row["size_x"])
        width = float(row["size_y"])
        require(length > 0.0 and width > 0.0, "footprint dimensions must be positive")
        yaw = math.radians(_yaw_deg(row))
        local = np.asarray(
            [
                [-0.5 * length, -0.5 * width],
                [0.5 * length, -0.5 * width],
                [0.5 * length, 0.5 * width],
                [-0.5 * length, 0.5 * width],
            ],
            dtype=np.float64,
        )
        rotation = np.asarray(
            [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
            dtype=np.float64,
        )
        center = np.asarray([float(row["world_x"]), float(row["world_y"])])
        return (local @ rotation.T + center).astype(np.float32)

    pred_corners = corners(prediction)
    truth_corners = corners(truth)
    intersection, _polygon = cv2.intersectConvexConvex(pred_corners, truth_corners)
    pred_area = float(prediction["size_x"]) * float(prediction["size_y"])
    truth_area = float(truth["size_x"]) * float(truth["size_y"])
    union = pred_area + truth_area - float(intersection)
    return max(0.0, min(1.0, float(intersection) / union)) if union > 0.0 else float("nan")


def segmentation_evidence_name(stream_id: str, frame_id: int) -> str:
    stream_digest = hashlib.sha256(str(stream_id).encode("utf-8")).hexdigest()[:16]
    return f"{stream_digest}_{int(frame_id)}.npy"


class DecodedSegmentationEvidenceWriter:
    """Bounded, non-blocking edge-side sink for evaluation-only decoded masks."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.queue: "queue.Queue[tuple[str, int, np.ndarray] | None]" = queue.Queue(maxsize=32)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="decoded-seg-evidence", daemon=True)
        self.thread.start()

    def submit(self, stream_id: str, frame_id: int, mask: np.ndarray) -> None:
        try:
            # The inference result owns this array and never mutates it after
            # publication. Retaining the reference avoids synchronous copying.
            self.queue.put_nowait((str(stream_id), int(frame_id), mask))
        except queue.Full:
            print(
                f"[SegEval] evidence queue full; frame {frame_id} was not retained",
                file=sys.stderr,
            )

    def _run(self) -> None:
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                item = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                self.queue.task_done()
                break
            stream_id, frame_id, mask = item
            path = self.output_dir / segmentation_evidence_name(stream_id, frame_id)
            temporary = path.with_suffix(path.suffix + ".tmp")
            try:
                with temporary.open("xb") as handle:
                    np.save(handle, mask.astype(np.uint8, copy=False), allow_pickle=False)
                os.replace(temporary, path)
            except Exception as exc:
                print(f"[SegEval] frame {frame_id} evidence write failed: {exc}", file=sys.stderr)
            finally:
                self.queue.task_done()

    def close(self) -> None:
        self.stop_event.set()
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        self.thread.join(timeout=5.0)


def run_edge_evaluation_wrapper(argv: Sequence[str]) -> int:
    """Run the certified tail with an out-of-band post-publish mask hook."""
    values = list(argv)
    try:
        index = values.index(EDGE_SEGMENTATION_EVIDENCE_FLAG)
        evidence_dir = Path(values[index + 1])
    except (ValueError, IndexError) as exc:
        raise AdapterError("edge evaluation wrapper lacks its evidence directory") from exc
    del values[index:index + 2]
    require(evidence_dir.is_absolute(), "edge segmentation evidence directory must be absolute")

    from uplink_only_spatial_map_pipeline import carla_fusion_staleness_scenario_uplink_only_v2 as split

    writer = DecodedSegmentationEvidenceWriter(evidence_dir)
    original_publish = split.SpatialMapResultPublisher.publish_from_payload

    def publish_with_evidence(
        publisher: Any,
        *,
        source_payload: dict[str, object],
        result: dict[str, object],
        timing: dict[str, object],
    ) -> None:
        # Normal spatial publication is invoked first and without modification.
        original_publish(
            publisher,
            source_payload=source_payload,
            result=result,
            timing=timing,
        )
        mask = result.get("mask")
        if isinstance(mask, np.ndarray):
            writer.submit(
                str(source_payload.get("stream_id") or publisher.stream_id),
                int(source_payload.get("frame_id", result.get("frame_id", -1))),
                mask,
            )

    split.SpatialMapResultPublisher.publish_from_payload = publish_with_evidence
    previous = sys.argv
    try:
        sys.argv = [previous[0], *values]
        split.main()
    finally:
        split.SpatialMapResultPublisher.publish_from_payload = original_publish
        writer.close()
        sys.argv = previous
    return 0


class PassiveSplitCollector:
    """Certified split sensors plus evaluation-only GT on Route B's exact ego."""

    def __init__(
        self,
        *,
        world: Any,
        ego: Any,
        cell: Mapping[str, Any],
        campaign: Mapping[str, Any],
        row: Mapping[str, str],
        binding: Mapping[str, Any],
        attempt_dir: Path,
        map_api_port: int,
        feedback_port: int,
        edge_evidence_dir: Path,
    ) -> None:
        import carla_collect_parked_ego_fusion_training_data as parked
        from data_collection.radar_sweep_aggregator_v1 import RadarSweepAggregator
        from rl_agent.ue_map_install_feedback_v1 import InstallFeedbackLedger
        from uplink_only_spatial_map_pipeline import carla_fusion_staleness_scenario_uplink_only_v2 as split

        self.world = world
        self.ego = ego
        self.cell = cell
        self.campaign = campaign
        self.row = row
        self.binding = binding
        self.attempt_dir = attempt_dir
        self.map_api_port = int(map_api_port)
        self.feedback_port = int(feedback_port)
        self.edge_evidence_dir = Path(edge_evidence_dir)
        self.parked = parked
        self.split = split
        self.stream_id = f"ue288_{cell['cell_id']}"
        contract = campaign["measurement_contract"]
        self.match_distance_m = float(contract["match_distance_m"])
        self.max_gt_distance_m = float(contract["max_gt_distance_m"])
        self.min_gt_area_px = float(contract["min_gt_area_px"])
        self.expected_prepared_hz = float(contract["expected_prepared_hz"])
        self.minimum_preparation_coverage = float(
            contract["minimum_sensor_preparation_coverage"]
        )
        self.segmentation_evidence_retention_s = float(
            contract["segmentation_evidence_retention_s"]
        )
        self.service_deadline_s = float(campaign["cell"]["service_deadline_ms"]) / 1000.0
        self.ack_timeout_s = float(campaign["cell"]["ack_timeout_ms"]) / 1000.0
        self.sensor_condition = threading.Condition()
        self.images: "OrderedDict[int, tuple[Any, float, float]]" = OrderedDict()
        self.semantic_images: "OrderedDict[int, Any]" = OrderedDict()
        self.radars: "OrderedDict[int, Any]" = OrderedDict()
        self.aggregator = RadarSweepAggregator(keep_sweeps=12)
        self.aggregator_error = ""
        self.prepared_queue: "queue.Queue[dict[str, Any] | None]" = queue.Queue(maxsize=4)
        self.segmentation_queue: "queue.Queue[int | None]" = queue.Queue(maxsize=32)
        self.stop_event = threading.Event()
        self.segmentation_stop_event = threading.Event()
        self.rows_lock = threading.Lock()
        self.rows: list[dict[str, Any]] = []
        self.gt_lock = threading.Lock()
        self.source_gt: dict[int, list[dict[str, Any]]] = {}
        self.aligned_gt: dict[int, list[dict[str, Any]]] = {}
        self.installed_predictions: dict[int, list[dict[str, Any]]] = {}
        self.segmentation_quality: dict[int, dict[str, object]] = {}
        self.segmentation_evidence_errors: dict[int, str] = {}
        self.ack_installed_frames: set[int] = set()
        self.sent_frames: set[int] = set()
        self.sensors: list[Any] = []
        self.failures: list[str] = []
        self.dropped = 0
        self.sent = 0
        self.route_ticks = 0
        self.first_frame: int | None = None
        self.last_frame: int | None = None
        self.cleanup_ok = False

        args = split_args(binding["front_args"], split)
        self.split_runtime_args = args
        device = split.od_demo.resolve_device(args.front_device)
        model, model_size = split.load_fusion_model(args, device)
        self.model_size = model_size
        transport = split._transport_config_from_args(args)
        self.sender = split.od_collect.UDPMessageSocket(
            bind_port=int(args.camera_source_port), remote_port=int(args.remote_port),
            chunk_bytes=int(args.chunk_bytes), socket_timeout=float(args.socket_timeout),
            host=str(args.bind_host), remote_host=str(args.remote_host),
            entropy_coder=transport.make_entropy_coder(),
        )
        registered = getattr(args, "_ue_registered_profile", None)
        self.head = split.CameraSideFusionInference(
            model=model, sender=self.sender, transport=transport, device=device,
            model_input_size=model_size, registered_profile=registered,
        )
        self.tracker = parked.StationaryTrackAccumulator(
            stationary_velocity_mps=0.35, parked_threshold_s=5.0,
            association_grid_m=1.5, max_stale_s=2.0,
        )
        self.actor_tracker = parked.ActorStationaryTracker(0.35, 5.0)
        self.aligned_actor_tracker = parked.ActorStationaryTracker(0.35, 5.0)
        self.intrinsics = split.intrinsics_at(int(model_size[0]), int(model_size[1]), 120.0)
        self.feedback = InstallFeedbackLedger(
            output_csv=attempt_dir / "map_feedback.csv",
            experiment_id=str(campaign["campaign_id"]), cell_id=str(cell["cell_id"]),
            bind_host="127.0.0.1", bind_port=self.feedback_port,
        )
        self._spawn_sensors()
        self.worker = threading.Thread(target=self._worker, name="route-b-split-front", daemon=True)
        self.segmentation_worker = threading.Thread(
            target=self._segmentation_worker,
            name="route-b-segmentation-evaluation",
            daemon=True,
        )
        self.feedback_worker = threading.Thread(target=self._feedback_worker, name="route-b-map-feedback", daemon=True)
        self.worker.start()
        self.segmentation_worker.start()
        self.feedback_worker.start()

    def _spawn_sensors(self) -> None:
        bp_lib = self.world.get_blueprint_library()
        args = SimpleNamespace(
            ego_camera_x=self.split.DEFAULT_EGO_CAMERA_X,
            ego_camera_y=self.split.DEFAULT_EGO_CAMERA_Y,
            ego_camera_z=self.split.DEFAULT_EGO_CAMERA_Z,
            ego_camera_pitch=self.split.DEFAULT_EGO_CAMERA_PITCH,
            ego_camera_yaw=self.split.DEFAULT_EGO_CAMERA_YAW,
            ego_camera_roll=self.split.DEFAULT_EGO_CAMERA_ROLL,
            ego_radar_x=self.split.DEFAULT_EGO_RADAR_X,
            ego_radar_y=self.split.DEFAULT_EGO_RADAR_Y,
            ego_radar_z=self.split.DEFAULT_EGO_RADAR_Z,
            ego_radar_pitch=self.split.DEFAULT_EGO_RADAR_PITCH,
            ego_radar_yaw=self.split.DEFAULT_EGO_RADAR_YAW,
            ego_radar_roll=self.split.DEFAULT_EGO_RADAR_ROLL,
        )
        rgb_bp = bp_lib.find("sensor.camera.rgb")
        rgb_bp.set_attribute("image_size_x", "1280")
        rgb_bp.set_attribute("image_size_y", "720")
        rgb_bp.set_attribute("fov", "120")
        rgb_bp.set_attribute("sensor_tick", "0.0")
        semantic_bp = bp_lib.find("sensor.camera.semantic_segmentation")
        semantic_bp.set_attribute("image_size_x", "1280")
        semantic_bp.set_attribute("image_size_y", "720")
        semantic_bp.set_attribute("fov", "120")
        semantic_bp.set_attribute("sensor_tick", "0.0")
        radar_bp = bp_lib.find("sensor.other.radar")
        radar_bp.set_attribute("range", "120")
        radar_bp.set_attribute("horizontal_fov", "120")
        radar_bp.set_attribute("vertical_fov", "30")
        radar_bp.set_attribute("points_per_second", "200000")
        radar_bp.set_attribute("sensor_tick", "0.0")
        self.camera = self.world.spawn_actor(rgb_bp, self.split._ego_camera_transform(args), attach_to=self.ego)
        self.semantic_camera = self.world.spawn_actor(
            semantic_bp,
            self.split._ego_camera_transform(args),
            attach_to=self.ego,
        )
        self.radar = self.world.spawn_actor(radar_bp, self.split._ego_radar_transform(args), attach_to=self.ego)
        self.sensors = [self.camera, self.semantic_camera, self.radar]
        self.camera.listen(self._on_rgb)
        self.semantic_camera.listen(self._on_semantic)
        self.radar.listen(self._on_radar)

    def _prune(self, values: OrderedDict[int, Any]) -> None:
        while len(values) > 24:
            values.popitem(last=False)

    def _on_rgb(self, image: Any) -> None:
        capture_wall = time.time()
        capture_perf = time.perf_counter()
        with self.sensor_condition:
            self.images[int(image.frame)] = (image, capture_wall, capture_perf)
            self._prune(self.images)
            self.sensor_condition.notify_all()

    def _on_radar(self, measurement: Any) -> None:
        with self.sensor_condition:
            try:
                self.aggregator.ingest(measurement)
            except Exception as exc:
                self.aggregator_error = f"{type(exc).__name__}: {exc}"
            self.radars[int(measurement.frame)] = measurement
            self._prune(self.radars)
            self.sensor_condition.notify_all()

    def _on_semantic(self, image: Any) -> None:
        with self.sensor_condition:
            self.semantic_images[int(image.frame)] = image
            self._prune(self.semantic_images)
            self.sensor_condition.notify_all()

    def on_world_tick(self, frame_id: int, route_tick: int) -> None:
        """Called by SamplingWorld after Route B advances its one owned tick."""
        self.route_ticks = int(route_tick)
        self.last_frame = int(frame_id)
        if self.first_frame is None:
            self.first_frame = int(frame_id)
        if (int(route_tick) - 1) % 2:
            return
        token = {
            "frame_id": int(frame_id), "route_tick": int(route_tick),
            "scheduled_perf": time.perf_counter(), "scheduled_wall": time.time(),
            "queue_depth": self.prepared_queue.qsize(),
        }
        try:
            self.prepared_queue.put_nowait(token)
        except queue.Full:
            self.dropped += 1
            self._append_row({**token, "prepare_status": "DROPPED_QUEUE_FULL", "processing_late": 1})

    def _records_for(self, frame_id: int, timeout_s: float = 0.25) -> tuple[Any, float, float, Any] | None:
        deadline = time.monotonic() + timeout_s
        with self.sensor_condition:
            while frame_id not in self.images or frame_id not in self.radars:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0 or self.stop_event.is_set():
                    return None
                self.sensor_condition.wait(timeout=remaining)
            image, capture_wall, capture_perf = self.images[frame_id]
            radar = self.radars[frame_id]
            return image, capture_wall, capture_perf, radar

    def _semantic_for(self, frame_id: int, timeout_s: float = 0.25) -> Any | None:
        deadline = time.monotonic() + timeout_s
        with self.sensor_condition:
            while frame_id not in self.semantic_images:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self.sensor_condition.wait(timeout=remaining)
            return self.semantic_images.pop(frame_id)

    def _append_row(self, row: Mapping[str, Any]) -> None:
        base = {
            "cell_id": self.cell["cell_id"], "action_id": self.cell["action_id"],
            "network_profile_id": self.cell["network_profile_id"], "stream_id": self.stream_id,
        }
        with self.rows_lock:
            self.rows.append({**base, **dict(row)})

    def _radar_activity(self, meta: Mapping[str, Any], radar_summary: Mapping[str, Any]) -> dict[str, Any]:
        provenance = meta.get("raw_provenance", {})
        ranges = np.asarray(provenance.get("original_range_m", []), dtype=np.float64)
        velocity = np.asarray(provenance.get("radial_velocity_mps", []), dtype=np.float64)
        valid = np.isfinite(ranges) & (ranges > 0.0) & (ranges <= 120.0)
        finite_velocity = velocity[np.isfinite(velocity)]
        return {
            "raw_radar_return_count": int(ranges.size),
            "raw_radar_valid_range_count": int(np.count_nonzero(valid)),
            "raw_radar_closing_count": int(np.count_nonzero(finite_velocity < -0.35)),
            "raw_radar_receding_count": int(np.count_nonzero(finite_velocity > 0.35)),
            "raw_radar_stationary_count": int(np.count_nonzero(np.abs(finite_velocity) <= 0.35)),
            "raw_radar_min_range_m": float(np.min(ranges[valid])) if np.any(valid) else "",
            "raw_radar_mean_range_m": float(np.mean(ranges[valid])) if np.any(valid) else "",
            "radar_projected_points": int(radar_summary.get("radar_points", 0)),
        }

    def _ground_truth(
        self,
        *,
        frame_id: int,
        timestamp: float,
        camera_matrix: np.ndarray,
        camera_inverse: np.ndarray,
        radar_points: Mapping[str, Any],
        stationary_tracker: Any | None = None,
    ) -> list[dict[str, Any]]:
        from pole_lraspp_multimodal_fusion.pole_lraspp_multimodal_fusion.object_targets import valid_localization_objects

        rows = self.parked.build_object_rows(
            world=self.world, ego_vehicle=self.ego,
            sample_base={"timestamp": float(timestamp), "frame_id": int(frame_id)},
            camera_location=self.camera.get_transform().location,
            camera_matrix=camera_matrix, camera_inverse_matrix=camera_inverse,
            intrinsics=self.intrinsics, width=int(self.model_size[0]), height=int(self.model_size[1]),
            max_distance_m=140.0,
            radar_world_xyz=np.asarray(radar_points.get("world_xyz", np.zeros((0, 3)))),
            stationary_tracker=stationary_tracker or self.actor_tracker,
            include_pedestrians=True,
            radar_support_margin_m=1.0, radar_person_support_mode="radius",
            radar_person_support_radius_m=1.5, radar_person_support_z_down_m=0.5,
            radar_person_support_z_up_m=2.0,
        )
        return valid_localization_objects(
            rows, image_width=int(self.model_size[0]), image_height=int(self.model_size[1]),
            min_area_px=self.min_gt_area_px, max_distance_m=self.max_gt_distance_m,
        )

    def _worker(self) -> None:
        while True:
            try:
                token = self.prepared_queue.get(timeout=0.1)
            except queue.Empty:
                if self.stop_event.is_set():
                    return
                continue
            if token is None:
                self.prepared_queue.task_done()
                return
            try:
                self._process_token(token)
            except Exception as exc:
                message = f"frame {token['frame_id']}: {type(exc).__name__}: {exc}"
                self.failures.append(message)
                self._append_row({**token, "prepare_status": "SPLIT_PROCESSING_FAILED", "error": message})
            finally:
                self.prepared_queue.task_done()

    def _process_token(self, token: Mapping[str, Any]) -> None:
        frame_id = int(token["frame_id"])
        records = self._records_for(frame_id)
        if records is None:
            self.dropped += 1
            self._append_row({**token, "prepare_status": "DROPPED_SENSOR_LATE_OR_MISSING", "processing_late": 1})
            return
        require(not self.aggregator_error, self.aggregator_error)
        image, capture_wall, capture_perf, radar_measurement = records
        with self.sensor_condition:
            if self.aggregator.anchor_s is None:
                self.aggregator.set_anchor(float(radar_measurement.timestamp))
            sweep_index = self.aggregator.sweep_index_for(float(radar_measurement.timestamp))
            if not self.aggregator.has_window(sweep_index):
                self._append_row({**token, "carla_timestamp": radar_measurement.timestamp, "prepare_status": "WARMUP_NO_COMPLETE_RADAR_WINDOW"})
                return
            radar_inverse = self.split.actor_world_inverse_matrix(self.radar)
            detections, window_meta = self.aggregator.window_detections(
                sweep_index, sensor_inverse_matrix=radar_inverse,
                reference_timestamp_s=float(radar_measurement.timestamp),
            )
        require(window_meta["callbacks"] == 4, f"accepted window requires four radar callbacks, got {window_meta['callbacks']}")
        camera_matrix = self.split.actor_world_matrix(self.camera)
        camera_inverse = self.split.actor_world_inverse_matrix(self.camera)
        radar_matrix = self.split.actor_world_matrix(self.radar)
        radar_tensor, radar_points, radar_summary = self.parked.build_radar_sample(
            detections=detections, sensor_matrix=radar_matrix,
            camera_inverse_matrix=camera_inverse, camera_intrinsics=self.intrinsics,
            width=int(self.model_size[0]), height=int(self.model_size[1]),
            frame_time_s=float(radar_measurement.timestamp), tracker=self.tracker,
            max_range_m=120.0, max_abs_velocity_mps=20.0,
            parked_threshold_s=5.0, point_radius_px=4, rasterizer="fast",
        )
        frame_bgr = self.split.od_demo.camera_image_to_bgr(image)
        capture_id = f"{self.stream_id}:{frame_id}"
        deadline = float(capture_wall) + self.service_deadline_s
        timeout_at = float(capture_wall) + self.ack_timeout_s
        self.feedback.register_capture(
            stream_id=self.stream_id, capture_id=capture_id, frame_id=frame_id,
            capture_at=float(capture_wall), action_id=str(self.cell["action_id"]),
            service_deadline_at=deadline, ack_timeout_at=timeout_at,
        )
        queue_wait_ms = (time.perf_counter() - float(token["scheduled_perf"])) * 1000.0
        front = self.head.process(
            frame_id=frame_id, frame_bgr=frame_bgr, radar_tensor=radar_tensor,
            camera_matrix=camera_matrix, camera_intrinsics_input=self.intrinsics,
            display_size=(1280, 720), carla_timestamp=float(radar_measurement.timestamp),
            camera_transform_payload=self.split._carla_transform_payload(self.camera.get_transform()),
            stream_id=self.stream_id, capture_perf=float(capture_perf),
            capture_wall_s=float(capture_wall),
            prep_timing={"capture_pipeline_queue_wait_ms": queue_wait_ms,
                         "capture_pipeline_queue_depth": int(token["queue_depth"])},
        )
        self.sent += 1
        self.sent_frames.add(frame_id)
        try:
            self.segmentation_queue.put_nowait(frame_id)
        except queue.Full:
            self.segmentation_evidence_errors[frame_id] = "GT_EVALUATION_QUEUE_FULL"
        velocity = self.ego.get_velocity()
        ego_speed = math.sqrt(float(velocity.x) ** 2 + float(velocity.y) ** 2 + float(velocity.z) ** 2)
        activity = self._radar_activity(window_meta, radar_summary)
        self._append_row(
            {
                **token, "capture_id": capture_id, "carla_timestamp": float(radar_measurement.timestamp),
                "capture_wall_s": float(capture_wall), "service_deadline_at": deadline,
                "prepare_status": "SENT", "processing_late": int(time.time() > deadline),
                "queue_wait_ms": queue_wait_ms, "front_ms": front.get("front_ms", ""),
                "payload_bytes": front.get("payload_bytes", ""),
                "payload_bytes_uncompressed": front.get("payload_bytes_uncompressed", ""),
                "payload_chunks": front.get("payload_chunks", ""),
                "window_sweeps": "|".join(str(v) for v in window_meta["sweep_indices"]),
                "window_callbacks": window_meta["callbacks"], "window_returns": window_meta["returns"],
                "window_span_s": window_meta["window_span_s"], "ego_speed_mps": ego_speed,
                **activity,
            }
        )
        # Evaluation-only GT is acquired after encode/send, so it cannot affect
        # action handling, feature construction, transport, decoding, or install.
        gt = self._ground_truth(
            frame_id=frame_id, timestamp=float(radar_measurement.timestamp),
            camera_matrix=camera_matrix, camera_inverse=camera_inverse,
            radar_points=radar_points,
        )
        with self.gt_lock:
            self.source_gt[frame_id] = gt

    def _segmentation_worker(self) -> None:
        pending: dict[int, tuple[np.ndarray, float]] = {}
        stop_started: float | None = None
        while True:
            try:
                frame_id = self.segmentation_queue.get(timeout=0.02)
            except queue.Empty:
                frame_id = None
            if frame_id is not None:
                try:
                    semantic_image = self._semantic_for(int(frame_id))
                    if semantic_image is None:
                        self.segmentation_evidence_errors[int(frame_id)] = (
                            "SEMANTIC_GT_EXACT_FRAME_MISSING"
                        )
                    else:
                        gt_tags = self.split.trained_seg_demo.carla_semantic_image_to_tags(
                            semantic_image
                        )
                        gt_3class = self.split.trained_seg_demo.map_carla_tags_to_3class(
                            gt_tags
                        )
                        pending[int(frame_id)] = (gt_3class, time.monotonic())
                except Exception as exc:
                    self.segmentation_evidence_errors[int(frame_id)] = (
                        f"SEMANTIC_GT_DECODE_FAILED:{type(exc).__name__}:{exc}"
                    )
                finally:
                    self.segmentation_queue.task_done()

            now = time.monotonic()
            for candidate, (gt_3class, observed_at) in list(pending.items()):
                evidence_path = self.edge_evidence_dir / segmentation_evidence_name(
                    self.stream_id, candidate
                )
                if evidence_path.is_file():
                    try:
                        predicted = np.load(evidence_path, allow_pickle=False)
                        quality = self.split._segmentation_quality_columns(
                            predicted, gt_3class
                        )
                        quality["prediction_vehicle_pixels"] = int(
                            np.count_nonzero(
                                predicted
                                == self.split.trained_seg_demo.CLASS_ID_VEHICLE
                            )
                        )
                        quality["prediction_person_pixels"] = int(
                            np.count_nonzero(
                                predicted
                                == self.split.trained_seg_demo.CLASS_ID_PERSON
                            )
                        )
                        with self.gt_lock:
                            self.segmentation_quality[candidate] = quality
                    except Exception as exc:
                        self.segmentation_evidence_errors[candidate] = (
                            f"DECODED_MASK_EVALUATION_FAILED:{type(exc).__name__}:{exc}"
                        )
                    pending.pop(candidate, None)
                    continue
                if now - observed_at > self.segmentation_evidence_retention_s:
                    self.segmentation_evidence_errors.setdefault(
                        candidate, "DECODED_MASK_NOT_OBSERVED_WITHIN_RETENTION"
                    )
                    pending.pop(candidate, None)

            if self.segmentation_stop_event.is_set():
                if stop_started is None:
                    stop_started = now
                with self.gt_lock:
                    required = set(self.ack_installed_frames)
                for candidate in list(pending):
                    if candidate not in required:
                        pending.pop(candidate, None)
                if not pending or now - stop_started >= 2.0:
                    for candidate in pending:
                        self.segmentation_evidence_errors.setdefault(
                            candidate, "DECODED_MASK_MISSING_FOR_ACK_INSTALLED_FRAME"
                        )
                    return

    def _map_snapshot(self, expected_frame: int) -> list[dict[str, Any]] | None:
        encoded_stream = urllib.parse.quote(self.stream_id, safe="")
        url = (
            f"http://127.0.0.1:{self.map_api_port}"
            f"/api/fusion_streams/installed/{encoded_stream}/{int(expected_frame)}"
        )
        try:
            with urllib.request.urlopen(url, timeout=0.5) as response:
                value = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            return None
        record = value.get("record") if isinstance(value.get("record"), dict) else {}
        if (
            value.get("status") != "INSTALLED"
            or str(record.get("stream_id") or "") != self.stream_id
            or int(record.get("frame_id") or -1) != int(expected_frame)
        ):
            return None
        predictions: list[dict[str, Any]] = []
        for obj in record.get("objects", []):
            location = obj.get("location", {})
            dimensions = obj.get("dimensions", {})
            type_name = str(obj.get("type", "")).lower()
            predictions.append(
                {
                    "class_name": "person" if type_name in {"person", "pedestrian", "walker"} else "vehicle",
                    "world_x": float(location.get("x", 0.0)), "world_y": float(location.get("y", 0.0)),
                    "world_z": float(location.get("z", 0.0)),
                    "size_x": float(dimensions.get("length", 0.0)),
                    "size_y": float(dimensions.get("width", 0.0)),
                    "size_z": float(dimensions.get("height", 0.0)),
                    "model_yaw_deg": float(
                        obj.get("model_yaw_deg", obj.get("yaw_deg", 0.0))
                    ),
                    "score": float(obj.get("score", 0.0)),
                }
            )
        return predictions

    def _feedback_worker(self) -> None:
        while not self.stop_event.is_set():
            before = set(self.feedback.pending)
            try:
                received = self.feedback.receive_once()
                self.feedback.record_expired()
            except Exception as exc:
                self.failures.append(f"feedback contract: {type(exc).__name__}: {exc}")
                self.stop_event.set()
                return
            if received is None:
                continue
            completed = before - set(self.feedback.pending)
            for capture_id in completed:
                try:
                    frame_id = int(capture_id.rsplit(":", 1)[1])
                except (IndexError, ValueError):
                    continue
                status = str(received.get("status") or "")
                with self.gt_lock:
                    if status == "ACK_INSTALLED":
                        self.ack_installed_frames.add(frame_id)
                if status != "ACK_INSTALLED":
                    continue
                predictions = self._map_snapshot(frame_id)
                if predictions is None:
                    self.failures.append(
                        f"exact installed map record missing after ACK for frame {frame_id}"
                    )
                    continue
                try:
                    aligned_timestamp = float(
                        self.world.get_snapshot().timestamp.elapsed_seconds
                    )
                    aligned = self._ground_truth(
                        frame_id=frame_id,
                        timestamp=aligned_timestamp,
                        camera_matrix=self.split.actor_world_matrix(self.camera),
                        camera_inverse=self.split.actor_world_inverse_matrix(self.camera),
                        radar_points={"world_xyz": np.zeros((0, 3), dtype=np.float32)},
                        stationary_tracker=self.aligned_actor_tracker,
                    )
                except Exception as exc:
                    self.failures.append(
                        f"aligned GT frame {frame_id}: {type(exc).__name__}: {exc}"
                    )
                    aligned = []
                with self.gt_lock:
                    self.installed_predictions[frame_id] = predictions
                    # Current GT is evaluation-only and is never fed back to
                    # the front, edge, map, action, or route controller.
                    self.aligned_gt[frame_id] = aligned

    def finish(self) -> bool:
        deadline = time.monotonic() + 10.0
        while self.prepared_queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.1)
        ack_deadline = time.monotonic() + 2.0
        while self.feedback.pending and time.monotonic() < ack_deadline:
            time.sleep(0.1)
        self.feedback.record_expired(time.time() + self.ack_timeout_s + 1.0)
        self.stop_event.set()
        try:
            self.prepared_queue.put_nowait(None)
        except queue.Full:
            pass
        with self.sensor_condition:
            self.sensor_condition.notify_all()
        self.worker.join(timeout=5.0)
        self.feedback_worker.join(timeout=3.0)
        evaluation_deadline = time.monotonic() + 5.0
        while self.segmentation_queue.unfinished_tasks and time.monotonic() < evaluation_deadline:
            time.sleep(0.05)
        self.segmentation_stop_event.set()
        self.segmentation_worker.join(timeout=3.0)
        sensor_ok = True
        for sensor in self.sensors:
            try:
                sensor.stop()
            except Exception:
                sensor_ok = False
            try:
                sensor_ok = bool(sensor.destroy()) and sensor_ok
            except Exception:
                sensor_ok = False
        try:
            self.sender.close()
        except Exception:
            sensor_ok = False
        try:
            self.feedback.close()
        except Exception:
            sensor_ok = False
        self.cleanup_ok = (
            sensor_ok
            and not self.worker.is_alive()
            and not self.feedback_worker.is_alive()
            and not self.segmentation_worker.is_alive()
        )
        return self.cleanup_ok

    def write_per_frame(self) -> None:
        path = self.attempt_dir / "per_frame_metrics.csv"
        with path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(PER_FRAME_FIELDS))
            writer.writeheader()
            with self.rows_lock:
                for row in sorted(self.rows, key=lambda item: (int(item.get("route_tick", 0)), str(item.get("prepare_status", "")))):
                    writer.writerow({field: row.get(field, "") for field in PER_FRAME_FIELDS})

    def write_perception(self) -> None:
        from pole_lraspp_multimodal_fusion.pole_lraspp_multimodal_fusion.object_targets import greedy_match_predictions

        terminal_status: dict[int, str] = {}
        with (self.attempt_dir / "map_feedback.csv").open(newline="", encoding="utf-8") as handle:
            for feedback_row in csv.DictReader(handle):
                if str(feedback_row.get("terminal", "")).lower() in {"1", "true"}:
                    terminal_status[int(feedback_row["frame_id"])] = str(
                        feedback_row.get("status") or ""
                    )

        fields = list(self.campaign["cell"]["perception_metric_fields"])
        path = self.attempt_dir / "perception_metrics.csv"
        with path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            with self.gt_lock:
                frames = sorted(self.sent_frames)
                for frame_id in frames:
                    predictions = self.installed_predictions.get(frame_id)
                    object_gt_available = frame_id in self.source_gt
                    frame_gt = self.source_gt.get(frame_id, [])
                    exact_available = predictions is not None
                    segmentation = self.segmentation_quality.get(frame_id)
                    for class_name in ("vehicle", "person"):
                        gt = [
                            row for row in frame_gt
                            if row.get("class_name") == class_name
                        ]
                        preds = [row for row in (predictions or []) if row.get("class_name") == class_name]
                        object_evidence = exact_available and object_gt_available
                        matches = (
                            greedy_match_predictions(
                                preds,
                                gt,
                                max_distance_m=self.match_distance_m,
                                class_aware=True,
                            )
                            if object_evidence
                            else []
                        )
                        tp: int | str = len(matches) if object_evidence else ""
                        fp: int | str = len(preds) - len(matches) if object_evidence else ""
                        fn: int | str = len(gt) - len(matches) if object_evidence else ""
                        source_errors = [distance for _pred, _gt, distance in matches]
                        aligned_truth = [
                            row for row in self.aligned_gt.get(frame_id, [])
                            if row.get("class_name") == class_name
                        ]
                        aligned_matches = (
                            greedy_match_predictions(
                                preds,
                                aligned_truth,
                                max_distance_m=self.match_distance_m,
                                class_aware=True,
                            )
                            if exact_available and aligned_truth
                            else []
                        )
                        aligned_errors = [
                            distance for _pred, _gt, distance in aligned_matches
                        ]
                        dimension_errors = []
                        footprint_ious = []
                        for pred_index, gt_index, _distance in matches:
                            pred, truth = preds[pred_index], gt[gt_index]
                            dimension_errors.append(float(np.mean(np.abs(np.asarray(
                                [pred["size_x"], pred["size_y"], pred["size_z"]]
                            ) - np.asarray([truth["size_x"], truth["size_y"], truth["size_z"]])))))
                            footprint_ious.append(oriented_footprint_iou(pred, truth))
                        denom_pred = len(preds)
                        denom_gt = len(gt)
                        if object_evidence:
                            precision: float | str = (
                                len(matches) / denom_pred if denom_pred else (1.0 if not gt else 0.0)
                            )
                            recall: float | str = (
                                len(matches) / denom_gt if denom_gt else (1.0 if not preds else 0.0)
                            )
                            valid_empty: int | str = int(not preds and not gt)
                            coverage: float | str = recall
                        else:
                            precision = recall = valid_empty = coverage = ""
                        segmentation_iou = float("nan")
                        if segmentation is not None:
                            key = (
                                "miou_vehicle_iou"
                                if class_name == "vehicle"
                                else "miou_person_iou"
                            )
                            segmentation_iou = float(segmentation[key])
                        writer.writerow(
                            {
                                "frame_id": frame_id, "class_name": class_name, "tp": tp, "fp": fp, "fn": fn,
                                "precision": precision,
                                "recall": recall,
                                "valid_empty": valid_empty,
                                "coverage": coverage,
                                "feedback_status": terminal_status.get(frame_id, ""),
                                "exact_frame_prediction_available": int(exact_available),
                                "object_gt_evidence_available": int(object_gt_available),
                                "segmentation_evidence_available": int(segmentation is not None),
                                "source_time_world_xy_error_m": mean_or_nan(source_errors),
                                "aligned_world_xy_error_m": mean_or_nan(aligned_errors),
                                "dimension_error_m": mean_or_nan(dimension_errors),
                                "footprint_iou": mean_or_nan(footprint_ious),
                                "segmentation_iou": segmentation_iou,
                            }
                        )

    def structural_acceptance(self) -> dict[str, Any]:
        failures: list[str] = []
        with self.rows_lock:
            rows = list(self.rows)
        scheduled_ticks = [int(row.get("route_tick", 0)) for row in rows]
        expected_ticks = list(range(1, int(self.route_ticks) + 1, 2))
        schedule_ok = sorted(scheduled_ticks) == expected_ticks
        if not schedule_ok:
            failures.append("10 Hz prepared-input scheduling phase/count contract failed")

        sent_rows = [row for row in rows if row.get("prepare_status") == "SENT"]
        eligible_rows = [
            row for row in rows
            if row.get("prepare_status") != "WARMUP_NO_COMPLETE_RADAR_WINDOW"
        ]
        preparation_coverage = (
            len(sent_rows) / len(eligible_rows) if eligible_rows else 0.0
        )
        if not sent_rows:
            failures.append("no prepared split frame was sent")
        if preparation_coverage < self.minimum_preparation_coverage:
            failures.append(
                "sensor/preparation coverage below campaign minimum: "
                f"{preparation_coverage:.6f} < {self.minimum_preparation_coverage:.6f}"
            )
        if any(int(row.get("window_callbacks", 0)) != 4 for row in sent_rows):
            failures.append("one or more sent frames lack the accepted four-callback radar window")

        feedback_path = self.attempt_dir / "map_feedback.csv"
        with feedback_path.open(newline="", encoding="utf-8") as handle:
            feedback_rows = list(csv.DictReader(handle))
        sent_captures = {str(row.get("capture_id") or "") for row in sent_rows}
        terminal_counts: dict[str, int] = {}
        outcome_counts: dict[str, int] = {}
        ack_frames: set[int] = set()
        for feedback_row in feedback_rows:
            status = str(feedback_row.get("status") or "")
            if status == "ACK_INSTALLED":
                ack_frames.add(int(feedback_row["frame_id"]))
            if str(feedback_row.get("terminal", "")).lower() in {"1", "true"}:
                capture_id = str(feedback_row.get("capture_id") or "")
                terminal_counts[capture_id] = terminal_counts.get(capture_id, 0) + 1
                outcome_counts[status] = outcome_counts.get(status, 0) + 1
        invalid_terminal_counts = {
            capture_id: terminal_counts.get(capture_id, 0)
            for capture_id in sent_captures
            if terminal_counts.get(capture_id, 0) != 1
        }
        unexpected_terminals = sorted(set(terminal_counts) - sent_captures)
        if invalid_terminal_counts or unexpected_terminals:
            failures.append(
                "exactly-one terminal feedback contract failed: "
                f"sent_counts={invalid_terminal_counts} unexpected={unexpected_terminals}"
            )

        exact_frames = set(self.installed_predictions)
        missing_exact = sorted(ack_frames - exact_frames)
        if missing_exact:
            failures.append(f"ACK-installed frames lack exact map records: {missing_exact[:8]}")
        exact_coverage = (
            len(ack_frames & exact_frames) / len(ack_frames) if ack_frames else None
        )
        missing_segmentation = sorted(ack_frames - set(self.segmentation_quality))
        if missing_segmentation:
            reasons = {
                frame_id: self.segmentation_evidence_errors.get(frame_id, "UNSPECIFIED")
                for frame_id in missing_segmentation[:8]
            }
            failures.append(f"ACK-installed frames lack exact segmentation IoU: {reasons}")

        perception_path = self.attempt_dir / "perception_metrics.csv"
        with perception_path.open(newline="", encoding="utf-8") as handle:
            perception_rows = list(csv.DictReader(handle))
        for metric_row in perception_rows:
            frame_id = int(metric_row["frame_id"])
            class_name = str(metric_row["class_name"])
            has_objects = metric_row["exact_frame_prediction_available"] == "1" and metric_row[
                "object_gt_evidence_available"
            ] == "1"
            if has_objects:
                required = ("tp", "fp", "fn", "precision", "recall", "valid_empty", "coverage")
                missing = [field for field in required if metric_row.get(field, "") == ""]
                if missing:
                    failures.append(
                        f"frame {frame_id} {class_name} missing object metrics: {missing}"
                    )
                if int(metric_row["tp"] or 0) > 0:
                    for field in (
                        "source_time_world_xy_error_m", "dimension_error_m", "footprint_iou"
                    ):
                        try:
                            value = float(metric_row[field])
                        except (TypeError, ValueError):
                            value = float("nan")
                        if not math.isfinite(value):
                            failures.append(
                                f"frame {frame_id} {class_name} missing matched {field}"
                            )
            if metric_row["segmentation_evidence_available"] == "1":
                quality = self.segmentation_quality.get(frame_id, {})
                gt_key = "gt_vehicle_pixels" if class_name == "vehicle" else "gt_person_pixels"
                pred_key = (
                    "prediction_vehicle_pixels"
                    if class_name == "vehicle"
                    else "prediction_person_pixels"
                )
                union_has_evidence = int(quality.get(gt_key, 0)) + int(
                    quality.get(pred_key, 0)
                ) > 0
                try:
                    value = float(metric_row["segmentation_iou"])
                except (TypeError, ValueError):
                    value = float("nan")
                if union_has_evidence and not math.isfinite(value):
                    failures.append(
                        f"frame {frame_id} {class_name} missing required segmentation_iou"
                    )

        return {
            "status": "PASS" if not failures else "FAIL",
            "expected_prepared_hz": self.expected_prepared_hz,
            "route_ticks": int(self.route_ticks),
            "scheduled_frames": len(rows),
            "expected_scheduled_frames": len(expected_ticks),
            "schedule_phase_and_count_ok": schedule_ok,
            "eligible_preparation_frames": len(eligible_rows),
            "sent_frames": len(sent_rows),
            "minimum_sensor_preparation_coverage": self.minimum_preparation_coverage,
            "sensor_preparation_coverage": preparation_coverage,
            "terminal_feedback_records": sum(terminal_counts.values()),
            "terminal_feedback_outcomes": dict(sorted(outcome_counts.items())),
            "ack_installed_frames": len(ack_frames),
            "exact_frame_perception_records": len(ack_frames & exact_frames),
            "exact_frame_perception_coverage": exact_coverage,
            "exact_frame_segmentation_records": len(ack_frames & set(self.segmentation_quality)),
            "failures": failures,
        }


def run_route_b(
    *,
    campaign: Mapping[str, Any],
    cell: Mapping[str, Any],
    row: Mapping[str, str],
    binding: Mapping[str, Any],
    attempt_dir: Path,
    carla_host: str,
    carla_port: int,
    map_api_port: int,
    feedback_port: int,
    edge_evidence_dir: Path,
    maximum_loop_sim_s: float,
) -> tuple[bool, dict[str, Any], PassiveSplitCollector | None]:
    import data_collection.run_route_b_density_loop as density
    from data_collection.run_route_b_perception_collection_v2 import (
        ClientProxy, SamplingWorld, intervention_policy,
    )

    route = campaign["route_b"]
    with tempfile.TemporaryDirectory(prefix="ue_route_b_metrics_") as raw_tmp:
        temporary = Path(raw_tmp)
        density_argv = [
            "--density", "traffic_50_50", "--vehicles", "50", "--pedestrians", "50",
            "--loops", "1", "--seed", str(route["scenario_seed"]),
            "--host", carla_host, "--port", str(carla_port), "--tm-port", "8010",
            "--route-config", str(repo_path(str(route["route_json"]))),
            "--lane-offset-m", "-0.5", "--target-speed-kph", "25.0",
            "--walker-brake-distance-m", "10.0", "--fixed-delta-seconds", "0.05",
            "--maximum-loop-sim-s", str(maximum_loop_sim_s), "--replenish-interval-s", "2.0",
            "--real-time-tick-period-s", "0.05", "--no-spectator", "--no-hybrid-physics",
            "--allow-scenario-interventions", "--maximum-overtakes", "0",
            "--out-csv", str(temporary / "route_metrics.csv"),
            "--summary-json", str(temporary / "route_metrics_summary.json"),
        ]
        density_args = density.build_parser().parse_args(density_argv)
        real_client_class = density.carla.Client
        density.carla.Client = lambda *values, **keywords: ClientProxy(
            real_client_class, int(route["traffic_manager_seed"]), *values, **keywords
        )
        original_drive = density.drive_one_loop_with_traffic
        holder: dict[str, Any] = {}

        def collecting_drive(
            world: Any, vehicle: Any, agent: Any, route_value: dict[str, Any], collisions: Any,
            run_args: argparse.Namespace, loop_index: int, maintain: Any, janitor: Any,
        ) -> dict[str, Any]:
            collector = PassiveSplitCollector(
                world=world, ego=vehicle, cell=cell, campaign=campaign, row=row,
                binding=binding, attempt_dir=attempt_dir, map_api_port=map_api_port,
                feedback_port=feedback_port, edge_evidence_dir=edge_evidence_dir,
            )
            holder["collector"] = collector
            result: dict[str, Any] | None = None
            try:
                result = original_drive(
                    SamplingWorld(world, collector, getattr(maintain, "population", None)),
                    vehicle, agent, route_value, collisions, run_args, loop_index, maintain, janitor,
                )
                return result
            finally:
                cleanup_ok = collector.finish()
                if result is not None and not cleanup_ok:
                    result["completed"] = False
                    result["abort_reason"] = "split adapter cleanup failure"
                holder["route_result"] = result

        density.drive_one_loop_with_traffic = collecting_drive
        route_rc = 2
        error = ""
        try:
            route_rc = int(density.run(density_args))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            density.drive_one_loop_with_traffic = original_drive
            density.carla.Client = real_client_class
        density_summary: dict[str, Any] = {}
        summary_path = temporary / "route_metrics_summary.json"
        if summary_path.is_file():
            density_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        result = holder.get("route_result")
        policy = intervention_policy(result, True)
        density_status = str(density_summary.get("status", ""))
        route_ok = route_rc == 0 or (
            density_status == "INTERVENED" and bool(policy["interventions_permitted_and_expected"])
        )
        collector = holder.get("collector")
        accepted = bool(
            route_ok and result and result.get("completed")
            and policy["interventions_permitted_and_expected"]
            and collector is not None and not collector.failures and collector.cleanup_ok
        )
        return accepted, {
            "route_runner_returncode": route_rc, "density_status": density_status,
            "route_completed": bool(result and result.get("completed")),
            "route_abort_reason": str((result or {}).get("abort_reason", "")),
            "intervention_policy": policy, "error": error,
        }, collector


def write_manifest(attempt_dir: Path, summary: Mapping[str, Any]) -> None:
    entries = []
    for name in EXPECTED_OUTPUTS:
        path = attempt_dir / name
        if name == "manifest.json":
            continue
        require(path.is_file(), f"required output missing before manifest: {name}")
        entries.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_json_create_only(
        attempt_dir / "manifest.json",
        {
            "schema": "scenesense.ue_288_cell_manifest.v1",
            "terminal_status": summary["terminal_status"],
            "measurement_contract": dict(summary["measurement_contract"]),
            "structural_acceptance_status": summary["structural_acceptance"]["status"],
            "registered_outputs": list(EXPECTED_OUTPUTS), "files": entries,
            "git_commit_at_launch": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=str(ROOT), check=False,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            ).stdout.strip(),
        },
    )


def validate_outputs(attempt_dir: Path, campaign: Mapping[str, Any]) -> None:
    missing = [name for name in EXPECTED_OUTPUTS if not (attempt_dir / name).is_file()]
    require(not missing, f"required outputs missing: {missing}")
    extras = sorted(
        path.name for path in attempt_dir.iterdir()
        if path.is_file() and path.name not in EXPECTED_OUTPUTS
    )
    require(not extras, f"unregistered cell output files: {extras}")
    with (attempt_dir / "radio_trace.csv").open(newline="", encoding="utf-8") as handle:
        radio_fields = set(csv.DictReader(handle).fieldnames or [])
    require(set(campaign["cell"]["radio_trace_fields"]).issubset(radio_fields), "radio trace schema drift")
    with (attempt_dir / "map_feedback.csv").open(newline="", encoding="utf-8") as handle:
        feedback_fields = set(csv.DictReader(handle).fieldnames or [])
    require(set(campaign["cell"]["map_feedback_fields"]).issubset(feedback_fields), "map feedback schema drift")
    with (attempt_dir / "perception_metrics.csv").open(newline="", encoding="utf-8") as handle:
        perception_fields = set(csv.DictReader(handle).fieldnames or [])
    require(set(campaign["cell"]["perception_metric_fields"]) == perception_fields, "perception schema drift")
    summary = json.loads((attempt_dir / "RESULTS_SUMMARY.json").read_text(encoding="utf-8"))
    require(
        summary.get("measurement_contract") == campaign["measurement_contract"],
        "results summary measurement-contract stamp drift",
    )
    structural_status = str(summary.get("structural_acceptance", {}).get("status") or "")
    require(structural_status in {"PASS", "FAIL"}, "structural acceptance status missing")
    if summary.get("terminal_status") == "PASSED":
        require(structural_status == "PASS", "PASSED summary has failed structural acceptance")
    manifest = json.loads((attempt_dir / "manifest.json").read_text(encoding="utf-8"))
    require(
        manifest.get("measurement_contract") == campaign["measurement_contract"],
        "manifest measurement-contract stamp drift",
    )


def run(args: argparse.Namespace) -> int:
    attempt_dir = args.attempt_dir.resolve()
    resolved_path = args.resolved_config.resolve()
    resolved, campaign, row = validate_resolved_contract(resolved_path, attempt_dir)
    cell = resolved["cell"]
    binding = launcher_binding(campaign, row)
    map_process: subprocess.Popen[Any] | None = None
    target_process: subprocess.Popen[Any] | None = None
    target_output = Path()
    target_stop = Path()
    collector: PassiveSplitCollector | None = None
    route_detail: dict[str, Any] = {}
    structural_acceptance: dict[str, Any] = {
        "status": "FAIL",
        "failures": ["Route B split collector did not reach structural validation"],
    }
    failures: list[str] = []
    cleanup = {"target_snr_restored": False, "map_process_stopped": False, "tail_stopped": False}
    started = time.time()
    edge_cache_root = ROOT / "torch_cache"
    require(edge_cache_root.is_dir(), f"shared tail cache mount missing: {edge_cache_root}")
    with (
        tempfile.TemporaryDirectory(prefix="ue_288_cell_runtime_") as raw_tmp,
        tempfile.TemporaryDirectory(
            prefix="ue_288_seg_eval_", dir=str(edge_cache_root)
        ) as edge_evidence_raw,
    ):
        temporary = Path(raw_tmp)
        edge_evidence_dir = Path(edge_evidence_raw)
        edge_evidence_container_dir = Path("/work/torch_cache") / edge_evidence_dir.name
        try:
            attach_oai(campaign, row)
            map_process = start_map_process(
                campaign, temporary_dir=temporary, action_id=str(cell["action_id"]),
                carla_host=args.carla_host, carla_port=args.carla_port,
                api_port=args.map_api_port, udp_port=args.spatial_map_port,
                feedback_port=args.feedback_port,
            )
            start_tail(
                campaign, binding, stream_id=f"ue288_{cell['cell_id']}",
                spatial_map_port=args.spatial_map_port,
                edge_evidence_container_dir=edge_evidence_container_dir,
            )
            # The target runtime reads the campaign root, not the resolved-cell
            # wrapper. Supply an isolated copy containing exactly that mapping.
            campaign_copy = temporary / "campaign.yaml"
            campaign_copy.write_text(yaml.safe_dump(campaign, sort_keys=False), encoding="utf-8")
            target_process, target_output, target_stop = start_target_snr(
                campaign, campaign_path=campaign_copy,
                profile_id=str(cell["network_profile_id"]), temporary_dir=temporary,
            )
            time.sleep(1.0)
            require(target_process.poll() is None, "target-SNR runtime exited during startup")
            route_ok, route_detail, collector = run_route_b(
                campaign=campaign, cell=cell, row=row, binding=binding,
                attempt_dir=attempt_dir, carla_host=args.carla_host,
                carla_port=args.carla_port, map_api_port=args.map_api_port,
                feedback_port=args.feedback_port,
                edge_evidence_dir=edge_evidence_dir,
                maximum_loop_sim_s=float(args.maximum_loop_sim_s),
            )
            if map_process.poll() is not None:
                failures.append("per-cell map process exited before cell cleanup")
            if target_process.poll() is not None:
                failures.append("target-SNR runtime exited before cell cleanup")
            if not tail_running():
                failures.append("registered tail container exited before cell cleanup")
            if not route_ok:
                failures.append("qualified Route B did not complete with a clean split adapter")
            if collector is None:
                failures.append("Route B never entered drive_one_loop_with_traffic")
            else:
                failures.extend(collector.failures)
                collector.write_per_frame()
                collector.write_perception()
                structural_acceptance = collector.structural_acceptance()
                failures.extend(structural_acceptance["failures"])
        except KeyboardInterrupt:
            failures.append("operator interrupt")
        except Exception as exc:
            failures.append(f"{type(exc).__name__}: {exc}")
        finally:
            if target_process is not None:
                try:
                    cleanup["target_snr_restored"] = stop_target_snr(
                        target_process, target_output, target_stop,
                        attempt_dir / "radio_trace.csv",
                    )
                except Exception as exc:
                    failures.append(f"target-SNR cleanup: {type(exc).__name__}: {exc}")
            cleanup["map_process_stopped"] = stop_process(map_process)
            cleanup["tail_stopped"] = stop_tail()
            if not all(cleanup.values()):
                failures.append("one or more adapter-owned runtime resources failed cleanup")

    # If setup failed before the sensor collector existed, still materialize the
    # registered CSV schemas so the failed cell remains inspectable.
    if not (attempt_dir / "per_frame_metrics.csv").exists():
        with (attempt_dir / "per_frame_metrics.csv").open("x", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=list(PER_FRAME_FIELDS)).writeheader()
    if not (attempt_dir / "perception_metrics.csv").exists():
        with (attempt_dir / "perception_metrics.csv").open("x", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=list(campaign["cell"]["perception_metric_fields"])).writeheader()
    if not (attempt_dir / "map_feedback.csv").exists():
        from rl_agent.ue_map_install_feedback_v1 import FIELDS as feedback_fields
        with (attempt_dir / "map_feedback.csv").open("x", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=list(feedback_fields)).writeheader()
    if not (attempt_dir / "radio_trace.csv").exists():
        from rl_agent.ue_target_snr_cell_runtime_v1 import FIELDS as radio_fields
        with (attempt_dir / "radio_trace.csv").open("x", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=list(radio_fields)).writeheader()

    terminal_status = "PASSED" if not failures else "FAILED"
    summary = {
        "schema": "scenesense.ue_288_cell_results_summary.v1",
        "status": terminal_status, "terminal_status": terminal_status,
        "cell_id": cell["cell_id"], "action_id": cell["action_id"],
        "network_profile_id": cell["network_profile_id"],
        "registered_feature_wire_codec": row["entropy_coder"],
        "spatial_map_packet_codec": "zlib",
        "one_ego_owner": "qualified_route_b_density_runner",
        "one_clock_owner": "qualified_route_b_density_runner_via_SamplingWorld",
        "measurement_contract": dict(campaign["measurement_contract"]),
        "structural_acceptance": structural_acceptance,
        "route": route_detail, "split_frames_sent": collector.sent if collector else 0,
        "split_frames_dropped": collector.dropped if collector else 0,
        "cleanup": cleanup, "failures": failures,
        "started_at_unix_s": started, "finished_at_unix_s": time.time(),
    }
    write_json_create_only(attempt_dir / "RESULTS_SUMMARY.json", summary)
    write_manifest(attempt_dir, summary)
    try:
        validate_outputs(attempt_dir, campaign)
    except Exception as exc:
        print(f"adapter output validation failed: {exc}", file=sys.stderr)
        return 2
    return 0 if terminal_status == "PASSED" else 1


def contract_check(configs: Sequence[Path]) -> int:
    source = Path(__file__).read_text(encoding="utf-8")
    runtime_source = source[:source.index("def contract_check(")]
    map_source = (
        ROOT
        / "uplink_only_spatial_map_pipeline/spatial_map_server_moving_ego_uplink_only_baseline.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = ""
        if isinstance(node.func, ast.Attribute):
            name = node.func.attr
        elif isinstance(node.func, ast.Name):
            name = node.func.id
        if name in {"tick", "apply_settings", "set_synchronous_mode", "run_client"}:
            forbidden_calls.append({"name": name, "line": node.lineno})
    require(not forbidden_calls, f"adapter contains forbidden clock/top-level calls: {forbidden_calls}")
    require(
        "/api/spatial_map/latest" not in runtime_source,
        "adapter still reads the racy latest-map endpoint",
    )
    require(
        "/api/fusion_streams/installed/" in source,
        "adapter lacks exact installed-frame lookup",
    )
    require(
        "installed_frame_history[history_key] = normalized" in map_source
        and "get_fusion_stream_installed_frame" in map_source,
        "map runtime lacks bounded exact installed-frame history",
    )
    require(
        source.index("original_publish(") < source.index("writer.submit("),
        "decoded-mask evidence must be queued only after normal map publication",
    )
    identical = {
        "world_x": 2.0,
        "world_y": -1.0,
        "size_x": 4.0,
        "size_y": 2.0,
        "yaw_deg": 37.0,
    }
    require(
        math.isclose(oriented_footprint_iou(identical, identical), 1.0, abs_tol=1e-6),
        "offline oriented-footprint IoU identity check failed",
    )
    reports = []
    for config_path in configs:
        campaign = load_yaml(config_path.resolve())
        require(campaign["runtime"]["required_route_b_split_cell_adapter"] == "rl_agent/ue_route_b_split_cell_adapter_v1.py", "campaign is not bound to this adapter")
        require(campaign.get("stop_on_first_failure") is True, "campaign fail-fast default is disabled")
        contract = campaign.get("measurement_contract", {})
        require(float(contract.get("match_distance_m", -1.0)) == 3.0, "primary match distance is not 3.0 m")
        require(float(contract.get("max_gt_distance_m", -1.0)) == 40.0, "GT max distance is not 40.0 m")
        require(float(contract.get("min_gt_area_px", -1.0)) == 12.0, "GT min area is not 12.0 px")
        require(float(contract.get("expected_prepared_hz", -1.0)) == 10.0, "prepared cadence is not 10 Hz")
        radio = campaign.get("network", {}).get("radio_baseline", {})
        require(
            radio.get("profile_id") == "OAI_N78_100MHZ_273PRB_4D5U_V1"
            and radio.get("selection_status") == "LOCKED",
            "campaign is not bound to the locked 100-MHz/4D5U radio baseline",
        )
        require("terminal" in campaign["cell"]["map_feedback_fields"], "terminal feedback marker is missing")
        expected = int(campaign["actions"]["expected_count"]) * 4
        require(int(campaign["cell"]["count"]) == expected, "campaign Cartesian count drift")
        reports.append({
            "config": str(config_path),
            "cells": expected,
            "radio_profile_id": radio["profile_id"],
            "target_snr_mapping_status": radio["target_snr_mapping_status"],
            "radio_runtime_binding_status": campaign["runtime"]["oai_radio_runtime_binding_status"],
        })
    registry = repo_path(str(load_yaml(configs[0].resolve())["actions"]["technical_registry_csv"]))
    with registry.open(newline="", encoding="utf-8") as handle:
        runtime_hashes = {row["certified_runtime_sha256"] for row in csv.DictReader(handle)}
    certified_runtime = ROOT / "uplink_only_spatial_map_pipeline/carla_fusion_staleness_scenario_uplink_only_v2.py"
    require(
        runtime_hashes == {sha256_file(certified_runtime)},
        "certified split runtime hash changed while adding evaluation evidence",
    )
    print(json.dumps({
        "status": "ADAPTER_CONTRACT_DRY_RUN_PASS", "configs": reports,
        "external_processes_started": 0, "ego_owner": "Route B",
        "clock_owner": "Route B through imported SamplingWorld",
        "adapter_forbidden_clock_calls": forbidden_calls,
        "certified_split_runtime_sha256": sha256_file(certified_runtime),
        "exact_installed_frame_history": "PASS",
        "primary_match_distance_m": 3.0,
        "oriented_footprint_iou": "PASS",
        "segmentation_evaluation_path": "POST_MAP_PUBLISH_OUT_OF_BAND",
        "selected_oai_radio_profile": "OAI_N78_100MHZ_273PRB_4D5U_V1",
        "real_launch_status": "BLOCKED_UNTIL_SPLITFUSION_MODELS_RADIO_RUNTIME_AND_SNR_MAPPING_ARE_BOUND",
    }, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolved-config", type=Path)
    parser.add_argument("--attempt-dir", type=Path)
    parser.add_argument("--carla-host", default="127.0.0.1")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--map-api-port", type=int, default=35001)
    parser.add_argument("--spatial-map-port", type=int, default=39310)
    parser.add_argument("--feedback-port", type=int, default=39401)
    parser.add_argument("--maximum-loop-sim-s", type=float, default=600.0)
    parser.add_argument("--contract-check", action="store_true")
    parser.add_argument("--campaign", type=Path, action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if EDGE_SEGMENTATION_EVIDENCE_FLAG in values:
        try:
            return run_edge_evaluation_wrapper(values)
        except (AdapterError, OSError, ValueError, KeyError) as exc:
            print(f"Route B edge evaluation wrapper error: {exc}", file=sys.stderr)
            return 2
    args = build_parser().parse_args(values)
    try:
        if args.contract_check:
            configs = args.campaign or [
                ROOT / "rl_agent/configs/ue_288_campaign_v1.yaml",
                ROOT / "rl_agent/configs/ue_16_cell_integration_pilot_v1.yaml",
            ]
            return contract_check(configs)
        require(args.resolved_config is not None and args.attempt_dir is not None, "live run requires --resolved-config and --attempt-dir")
        return run(args)
    except (AdapterError, OSError, ValueError, KeyError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(f"Route B split-cell adapter error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
