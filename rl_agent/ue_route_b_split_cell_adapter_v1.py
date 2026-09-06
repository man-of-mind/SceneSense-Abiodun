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
import shutil
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
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_agent.splitfusion_live_dispatch_v1.live_pilot_runtime import (  # noqa: E402
    DEADLINE_STAGES,
    UE_STAGE_AFTER_PREPARATION,
    DeadlineExpired,
    FastStationaryTrackAccumulator,
    LatestFramePendingSlot,
    LivePilotCellRuntime,
    _Counters,
    ack_timeout_s,
    check_deadline,
    service_deadline_s,
)

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
    "service_deadline_at", "ack_timeout_at", "prepare_status", "processing_late", "queue_depth",
    "queue_wait_ms", "sensor_wait_ms", "radar_window_ms", "radar_prepare_ms",
    "rgb_convert_ms", "scene_snapshot_ms", "pre_front_compute_ms", "front_ms",
    "payload_bytes", "payload_bytes_uncompressed",
    "payload_chunks", "window_sweeps", "window_callbacks", "window_returns",
    "window_span_s", "raw_radar_return_count", "raw_radar_valid_range_count",
    "raw_radar_closing_count", "raw_radar_receding_count",
    "raw_radar_stationary_count", "raw_radar_min_range_m",
    "raw_radar_mean_range_m", "radar_projected_points", "ego_speed_mps",
    "profile_id", "model_family", "quantizer", "q_e4", "routing_tag",
    "scientific_inner_bytes", "sfd1_overhead_bytes", "sfd1_bytes", "datagrams",
    "udp_application_bytes", "estimated_wire_bytes", "front_timing_ns",
    "capture_started_ns", "ue_prepare_finished_ns", "send_finished_ns",
    "edge_result_received_ns", "edge_timing_ns", "edge_result_datagrams",
    "feature_received_datagrams", "feature_duplicate_datagrams",
    "decoded", "finite", "decoder_identity", "reconstructed_device",
    "frame_context_valid", "camera_pose_reconstruct_ns",
    "finite_output_tensor_count", "service_record_count",
    "edge_call_ledger", "edge_counters",
    # Phase-15 real-time recovery accounting. FEATURE_RECEIVED is diagnostic
    # only; MAP_INSTALLED is the terminal service-success event and the one
    # measured against the authoritative capture-based deadline.
    "deadline_expiry_stage", "deadline_expiry_age_ms", "replaced_by_frame_id",
    "feature_received_at", "map_publication_status", "map_installed_at",
    "install_aoi_ms", "edge_receipt_wall_s", "edge_tail_complete_wall_s",
    "edge_evidence_install_wall_s", "edge_evidence_installation_status",
    "edge_evidence_sha256", "edge_terminal_reason", "evaluation_gt_status",
    "error",
)
EDGE_SEGMENTATION_EVIDENCE_FLAG = "--edge-segmentation-evidence-dir"
# The evaluation label map is persisted by the edge on its own per-cell mount
# and never base64-encoded into the radio return path.
EDGE_EVIDENCE_LEAF = "segmentation_evidence"
PRESERVED_EVIDENCE_LEAF = "segmentation_evidence"
# Registered preservation quota for raw label maps. The complete hash manifest
# is always retained; raw arrays are preserved for installed frames up to this
# many bytes per cell so a cell can never exhaust the host filesystem.
PRESERVED_EVIDENCE_QUOTA_BYTES = 1 << 30
CAMERA_MOUNT = (1.8, 0.0, 1.55, -4.0, 0.0, 0.0)
RADAR_MOUNT = (2.0, 0.0, 1.0, 0.0, 0.0, 0.0)
CLASS_ID_BACKGROUND, CLASS_ID_VEHICLE, CLASS_ID_PERSON = 0, 1, 2
_CARLA_3CLASS_LUT = np.zeros(256, dtype=np.uint8)
for _tag in (1, 2, 4, 6, 7, 14, 19):
    _CARLA_3CLASS_LUT[_tag] = CLASS_ID_VEHICLE
_CARLA_3CLASS_LUT[15] = CLASS_ID_PERSON


class AdapterError(RuntimeError):
    """A fixed cell contract or runtime stage failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AdapterError(message)


def camera_intrinsics(width: int, height: int, fov_deg: float) -> np.ndarray:
    focal = (float(width) / 2.0) / math.tan(math.radians(float(fov_deg)) / 2.0)
    return np.asarray(((focal, 0.0, width / 2.0), (0.0, focal, height / 2.0), (0.0, 0.0, 1.0)), dtype=np.float64)


def sensor_transform(values: tuple[float, float, float, float, float, float]) -> Any:
    import carla

    x, y, z, pitch, yaw, roll = values
    return carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(pitch=pitch, yaw=yaw, roll=roll))


def actor_world_matrix(actor: Any) -> np.ndarray:
    return np.asarray(actor.get_transform().get_matrix(), dtype=np.float64)


def actor_world_inverse_matrix(actor: Any) -> np.ndarray:
    return np.asarray(actor.get_transform().get_inverse_matrix(), dtype=np.float64)


def carla_image_to_bgr(image: Any) -> np.ndarray:
    return np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3].copy()


def semantic_gt_3class(image: Any) -> np.ndarray:
    tags = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, 2]
    return _CARLA_3CLASS_LUT[tags]


def segmentation_quality_columns(predicted: np.ndarray, ground_truth: np.ndarray) -> dict[str, object]:
    import cv2

    if predicted.shape != ground_truth.shape:
        ground_truth = cv2.resize(ground_truth, (predicted.shape[1], predicted.shape[0]), interpolation=cv2.INTER_NEAREST)
    ious: dict[int, float] = {}
    present: list[float] = []
    for class_id in (CLASS_ID_BACKGROUND, CLASS_ID_VEHICLE, CLASS_ID_PERSON):
        union = int(np.logical_or(predicted == class_id, ground_truth == class_id).sum())
        value = float("nan") if union == 0 else int(np.logical_and(predicted == class_id, ground_truth == class_id).sum()) / union
        ious[class_id] = value
        if math.isfinite(value):
            present.append(value)
    fg_union = int(np.logical_or(predicted != 0, ground_truth != 0).sum())
    return {
        "gt_camera_available": 1, "miou_binary": float("nan") if fg_union == 0 else int(np.logical_and(predicted != 0, ground_truth != 0).sum()) / fg_union,
        "miou_3class_macro": float(np.mean(present)) if present else float("nan"),
        "miou_vehicle_iou": ious[CLASS_ID_VEHICLE], "miou_person_iou": ious[CLASS_ID_PERSON],
        "gt_vehicle_pixels": int(np.count_nonzero(ground_truth == CLASS_ID_VEHICLE)),
        "gt_person_pixels": int(np.count_nonzero(ground_truth == CLASS_ID_PERSON)),
    }


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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_create_only(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def action_row(campaign: Mapping[str, Any], action_id: int | str) -> dict[str, Any]:
    actions = campaign["actions"]
    catalog_path = repo_path(str(actions["catalog_json"]))
    require(catalog_path.is_file(), f"locked action catalog missing: {catalog_path}")
    require(
        sha256_file(catalog_path) == str(actions["catalog_sha256"]),
        "locked action catalog hash drift",
    )
    catalog = load_json(catalog_path)
    require(catalog.get("schema") == actions["catalog_schema"], "action catalog schema drift")
    rows = catalog.get("profiles")
    require(isinstance(rows, list) and len(rows) == 72, "action catalog inventory drift")
    matches = [row for row in rows if str(row.get("action_id")) == str(action_id)]
    require(len(matches) == 1, f"resolved action is not unique in catalog: {action_id}")
    profile = matches[0]
    require(
        profile.get("execution_mode") == "SPLIT"
        and profile.get("capabilities", {}).get("transport_valid") is True
        and profile.get("capabilities", {}).get("agent_action_enabled") is True,
        f"resolved action is not enabled and transport-valid: {action_id}",
    )
    checkpoint = profile.get("ae_checkpoint") or profile.get("perception_checkpoint")
    require(isinstance(checkpoint, dict), f"resolved action lacks a checkpoint binding: {action_id}")
    quantizer = str(profile["quantizer"]).lower()
    return {
        **profile,
        "model_family": str(profile["family"]).lower(),
        "checkpoint_path": str(checkpoint["path"]),
        "checkpoint_sha256": str(checkpoint["sha256"]),
        "quantization_mode": f"per_channel_{quantizer}",
        "roi_drop_fraction": format(float(profile["q"]), "g"),
        "entropy_coder": "zstd",
        "zstd_level": int(profile["zstd_level"]),
    }


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
    radio = campaign.get("network", {}).get("radio_baseline", {})
    require(
        radio.get("profile_id") == "OAI_N78_100MHZ_273PRB_4D5U_V1"
        and radio.get("selection_status") == "LOCKED",
        "real adapter launch refused: radio profile is not the locked n78 100-MHz/273-PRB baseline",
    )
    require(
        radio.get("target_snr_mapping_status") == "QUALIFIED_ON_OAI_N78_100MHZ_273PRB_4D5U_V1"
        and campaign["network"].get("mapping_calibration_radio_profile_id")
        == "OAI_N78_100MHZ_273PRB_4D5U_V1",
        "real adapter launch refused: legacy 40-MHz/106-PRB RFsim mapping is not valid for this campaign",
    )
    require(
        campaign["runtime"].get("oai_radio_runtime_binding_status") == "BOUND_SPLITFUSION_100MHZ_4D5U"
        and campaign["runtime"].get("oai_registered_profile_launcher_status")
        == "QUALIFIED_SPLITFUSION_100MHZ_4D5U",
        "real adapter launch refused: 100-MHz/273-PRB runtime/launcher is not qualified",
    )
    runtime = campaign["runtime"]
    require(
        runtime.get("split_inference_runtime")
        == "rl_agent/splitfusion_live_dispatch_v1/runtime_binding.json"
        and int(runtime.get("sfd1_protocol_version", 0)) == 2
        and runtime.get("frame_context_required") is True,
        "real adapter launch refused: Phase-13 SFD1-v2 dispatcher is not bound",
    )
    require(
        runtime.get("udp_fragment_header") == "!IHH"
        and int(runtime.get("udp_chunk_bytes", 0)) == 12_500
        and int(runtime.get("socket_buffer_request_bytes", 0)) == 8 * 1024 * 1024
        and int(runtime.get("edge_receive_port", 0)) == 51002
        and int(runtime.get("camera_result_port", 0)) == 51004,
        "real adapter launch refused: qualified UDP buffer/fragment contract drift",
    )
    row = action_row(campaign, cell["action_id"])
    require(row["profile_id"] == str(cell["profile_id"]), "resolved catalog profile/action mismatch")
    require(row["model_family"] == str(cell["model_family"]), "resolved model family/action mismatch")
    require(row["entropy_coder"] == "zstd", "certified action transport must remain zstd")
    return resolved, campaign, row


def stop_tail() -> bool:
    completed = subprocess.run(
        [
            "sudo", "-n", "docker", "compose", "-f", "docker-compose.yaml",
            "-f", "docker-compose.fusion-back.yaml", "down", "--remove-orphans",
        ], cwd=str(ROOT / "receiver_container"), check=False,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return completed.returncode == 0 and not tail_running()


def tail_running() -> bool:
    completed = subprocess.run(
        ["sudo", "docker", "inspect", "-f", "{{.State.Running}}", "oai-perception-rx"],
        cwd=str(ROOT), check=False, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    return completed.returncode == 0 and completed.stdout.strip() == "true"


def create_cell_edge_state_root(temporary_dir: Path) -> Path:
    """Create the one writable mount owned by this cell's live edge."""

    owner = Path(temporary_dir).resolve(strict=True)
    edge_state = owner / "splitfusion_edge_state"
    try:
        edge_state.mkdir(parents=False, exist_ok=False, mode=0o700)
    except FileExistsError as exc:
        raise AdapterError(f"cell edge-state path already exists: {edge_state}") from exc
    resolved = edge_state.resolve(strict=True)
    require(resolved.parent == owner, "cell edge-state path escaped its runtime directory")
    require(os.access(resolved, os.W_OK | os.X_OK), "cell edge-state root is not writable")
    return resolved

def seed_cell_edge_state(campaign: Mapping[str, Any], edge_state: Path) -> None:
    """Seed immutable constructor weights into an otherwise fresh cell cache."""

    record = campaign["deployment"]["fcos_constructor_weights"]
    source = repo_path(str(record["path"]))
    require(source.name == "fcos_resnet50_fpn_coco-99b0c9b7.pth", "FCOS cache filename drift")
    require(sha256_file(source) == str(record["sha256"]), "FCOS cache seed hash drift")
    checkpoint_dir = edge_state / "hub/checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    destination = checkpoint_dir / source.name
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    require(sha256_file(destination) == str(record["sha256"]), "seeded FCOS cache hash drift")




def _bounded_log_tail(path: Path, limit: int = 8192) -> str:
    if not path.is_file():
        return ""
    return path.read_bytes()[-limit:].decode("utf-8", errors="replace")


def start_live_edge(
    campaign: Mapping[str, Any], cell: Mapping[str, Any], temporary_dir: Path
) -> Path:
    """Start exactly the qualified SFD1-v2 edge in the OAI-network GPU container."""

    runtime = campaign["runtime"]
    require(not tail_running(), "a previous phase-owned edge container is still running")
    edge_scratch = create_cell_edge_state_root(temporary_dir)
    seed_cell_edge_state(campaign, edge_scratch)
    # The edge container runs as root while this directory tree is owned by the
    # host user, so the evaluation-evidence leaf is created here with group and
    # other write permission. Otherwise the host could not unlink the
    # root-created label maps during cold teardown.
    evidence_host = edge_scratch / EDGE_EVIDENCE_LEAF
    evidence_host.mkdir(parents=False, exist_ok=False, mode=0o777)
    os.chmod(evidence_host, 0o777)
    evidence_container = Path("/work/torch_cache") / EDGE_EVIDENCE_LEAF
    ready_host = edge_scratch / "ready.json"
    ready_container = Path("/work/torch_cache/ready.json")
    config_container = Path("/work/abiodun") / "rl_agent/configs/splitfusion_16_cell_live_carla_oai_pilot_v1.json"
    env = os.environ.copy()
    env.update(
        {
            "FUSION_BACK_DUAL": "0",
            "FUSION_BACK_BIND_HOST": "0.0.0.0",
            "FUSION_BACK_REMOTE_HOST": str(runtime["ue_bind_host"]),
            "FUSION_BACK_REMOTE_HOST_1": str(runtime["ue_bind_host"]),
            "FUSION_BACK_DEVICE": "cuda",
            "SPLITFUSION_EDGE_STATE_ROOT": str(edge_scratch),
            "SPLITFUSION_FCOS_WEIGHT_PATH": str(repo_path(str(campaign["deployment"]["fcos_constructor_weights"]["path"]))),
            "FUSION_BACK_SCRIPT": "-m rl_agent.splitfusion_live_dispatch_v1.live_pilot_runtime",
            "FUSION_REMOTE_PORT_1": str(runtime["edge_receive_port"]),
            "FUSION_REMOTE_SOURCE_PORT_1": str(runtime["edge_source_port"]),
            "FUSION_CAMERA_RESULT_PORT_1": str(runtime["camera_result_port"]),
            "FUSION_BACK_EXTRA_ARGS": " ".join(
                (
                    "--edge", "--config", str(config_container), "--action-id", str(cell["action_id"]),
                    "--allowed-action-ids", ",".join(
                        str(value)
                        for value in campaign.get("_qualification", {}).get(
                            "action_ids", [cell["action_id"]]
                        )
                    ),
                    "--ready-file", str(ready_container), "--edge-port", str(runtime["edge_receive_port"]),
                    "--result-host", str(runtime["ue_bind_host"]), "--result-port", str(runtime["camera_result_port"]),
                    EDGE_SEGMENTATION_EVIDENCE_FLAG, str(evidence_container),
                    "--run-id", str(campaign["campaign_id"]),
                    "--cell-id", str(cell["cell_id"]),
                )
            ),
        }
    )
    launcher_log = Path(temporary_dir) / "edge_launcher.log"
    try:
        with launcher_log.open("xb") as stream:
            completed = subprocess.run(
                [str(ROOT / "scripts/receiver_container_fusion_back_up.sh")], cwd=str(ROOT), env=env,
                check=False, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, timeout=180.0,
            )
        require(
            completed.returncode == 0,
            f"qualified edge container startup failed rc={completed.returncode}; "
            f"launcher_tail={_bounded_log_tail(launcher_log)!r}",
        )
        deadline = time.monotonic() + 180.0
        while time.monotonic() < deadline:
            require(tail_running(), "qualified edge container exited before preload completed")
            if ready_host.is_file():
                ready = json.loads(ready_host.read_text(encoding="utf-8"))
                require(
                    ready.get("schema") == "splitfusion_live_edge_ready.v1"
                    and ready.get("allowed_action_ids") == [
                        int(value)
                        for value in campaign.get("_qualification", {}).get(
                            "action_ids", [cell["action_id"]]
                        )
                    ]
                    and ready.get("tail_device") == "cuda:0"
                    and ready.get("dense_label_map_on_radio") is False
                    and str(ready.get("evaluation_evidence_dir") or "")
                    == str(evidence_container),
                    "edge preload ready record identity/device drift",
                )
                return edge_scratch
            time.sleep(0.25)
        raise AdapterError("qualified edge did not complete preload readiness")
    except Exception as exc:
        container_logs = subprocess.run(
            ["sudo", "docker", "logs", "--tail", "80", "oai-perception-rx"],
            cwd=str(ROOT), check=False, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        ).stdout[-8192:]
        stop_tail()
        shutil.rmtree(edge_scratch, ignore_errors=True)
        raise AdapterError(
            f"{type(exc).__name__}: {exc}; edge_container_tail={container_logs!r}"
        ) from exc


def stop_live_edge(edge_scratch: Path | None) -> bool:
    """Stop only this phase-owned container and its newly-created readiness leaf."""

    stopped = stop_tail()
    if edge_scratch is not None:
        shutil.rmtree(edge_scratch, ignore_errors=True)
    return stopped and not tail_running()


def inspect_live_edge_mounts(edge_scratch: Path) -> dict[str, Any]:
    """Bind the running consumer to the exact read-only code and cell state."""

    completed = subprocess.run(
        ["sudo", "docker", "inspect", "-f", "{{json .Mounts}}", "oai-perception-rx"],
        cwd=str(ROOT), check=False, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    require(completed.returncode == 0, "cannot inspect qualified edge mounts")
    mounts = json.loads(completed.stdout)

    def selected(destination: str) -> dict[str, Any]:
        rows = [row for row in mounts if row.get("Destination") == destination]
        require(len(rows) == 1, f"edge container mount is not unique: {destination}")
        return rows[0]

    state = selected("/work/torch_cache")
    repository = selected("/work/abiodun")
    require(
        Path(str(state["Source"])).resolve(strict=True) == edge_scratch.resolve(strict=True)
        and bool(state.get("RW")) is True,
        "edge state mount source/mode drift",
    )
    require(
        Path(str(repository["Source"])).resolve(strict=True) == ROOT.resolve(strict=True)
        and bool(repository.get("RW")) is False,
        "edge repository mount source/mode drift",
    )
    return {
        "state": {"source": str(state["Source"]), "destination": "/work/torch_cache", "rw": True},
        "repository": {"source": str(repository["Source"]), "destination": "/work/abiodun", "rw": False},
    }


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
    try:
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
        raise AdapterError("per-cell map process did not become ready")
    except BaseException:
        stop_process(process)
        raise


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
    start_file: Path,
) -> tuple[subprocess.Popen[bytes], Path, Path]:
    runtime = repo_path(str(campaign["runtime"]["target_snr_runtime"]))
    try:
        runtime_module = ".".join(runtime.relative_to(ROOT).with_suffix("").parts)
    except ValueError as exc:
        raise AdapterError("target-SNR runtime is outside the repository") from exc
    require(
        runtime_module == "rl_agent.splitfusion_live_dispatch_v1.live_pilot_target_snr_runtime",
        "target-SNR module identity drift",
    )
    output = temporary_dir / "radio_trace.csv"
    stop_file = temporary_dir / "stop_target_snr"
    process = subprocess.Popen(
        [sys.executable, "-m", runtime_module, "--campaign", str(campaign_path),
         "--profile-id", profile_id, "--output", str(output), "--stop-file", str(stop_file),
         "--start-file", str(start_file)],
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


def verify_clean_rfsim_without_runtime(campaign: Mapping[str, Any]) -> bool:
    """Read back the clean channel when no target runtime ever mutated it."""

    from rl_agent import splitfusion_phase14a_100mhz_calibration_v1 as phase14a
    from rl_agent import splitfusion_phase14b_corrected_four_profile_replay_v1 as phase14b

    config = phase14a.load_json(repo_path(str(campaign["runtime"]["phase14a_config"])))
    report = phase14b.restore_interrupted_radio(config)
    return bool(
        report is not None
        and report.get("verified") is True
        and math.isclose(float(report.get("noise_power_db", math.nan)), -50.0, abs_tol=1e-6)
    )


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


def _monotonic_fraction(series: Sequence[tuple[float, float]]) -> float | None:
    """Fraction of consecutive installed frames whose AoI did not decrease.

    A repaired data plane holds AoI near the service period; a value close to
    1.0 is the signature of the backlog the retry4 audit measured, where every
    admitted frame was served from an ever-older queue.
    """

    ordered = [value for _key, value in sorted(series, key=lambda item: item[0])]
    if len(ordered) < 2:
        return None
    increases = sum(
        1 for index in range(1, len(ordered)) if ordered[index] >= ordered[index - 1]
    )
    return increases / (len(ordered) - 1)


class FrozenActor:
    """An immutable actor view pinned to one synchronized CARLA frame.

    Evaluation-only ground truth is no longer built on the real-time
    preparation worker, so it can no longer read live actor state: by the time
    it runs, the actors have moved. This proxy exposes exactly the surface the
    unmodified ``carla_collect_parked_ego_fusion_training_data`` row builder
    uses, backed by the transform and velocity recorded in the world snapshot
    taken at the frame the features were actually derived from. Static geometry
    (``type_id``, ``bounding_box``) is immutable for an actor's lifetime and is
    carried by reference.
    """

    __slots__ = ("id", "type_id", "bounding_box", "_transform", "_velocity")

    def __init__(self, *, actor_id: int, type_id: str, bounding_box: Any,
                 transform: Any, velocity: Any) -> None:
        self.id = int(actor_id)
        self.type_id = str(type_id)
        self.bounding_box = bounding_box
        self._transform = transform
        self._velocity = velocity

    def get_transform(self) -> Any:
        return self._transform

    def get_location(self) -> Any:
        return self._transform.location

    def get_velocity(self) -> Any:
        return self._velocity


class FrozenActorList:
    """A snapshot-backed stand-in for ``carla.ActorList`` with ``filter``."""

    __slots__ = ("_actors",)

    def __init__(self, actors: Sequence[FrozenActor]) -> None:
        self._actors = tuple(actors)

    def filter(self, pattern: str) -> tuple[FrozenActor, ...]:
        import fnmatch

        return tuple(
            actor for actor in self._actors
            if fnmatch.fnmatch(actor.type_id, str(pattern))
        )

    def __iter__(self):
        return iter(self._actors)

    def __len__(self) -> int:
        return len(self._actors)


class FrozenWorld:
    """Expose only ``get_actors`` so the shared row builder stays unmodified."""

    __slots__ = ("_actors",)

    def __init__(self, actors: FrozenActorList) -> None:
        self._actors = actors

    def get_actors(self) -> FrozenActorList:
        return self._actors


class SceneSnapshotSource:
    """Build immutable per-frame actor snapshots without live world queries.

    ``world.get_actors()`` is an RPC and actor geometry never changes, so the
    static registry is refreshed on the evaluation thread only. The real-time
    worker pays for one already-cached ``world.get_snapshot()`` plus a dict of
    transforms.
    """

    def __init__(self, world: Any, *, ego_id: int, refresh_interval_s: float = 2.0) -> None:
        self.world = world
        self.ego_id = int(ego_id)
        self.refresh_interval_s = float(refresh_interval_s)
        self._static: dict[int, tuple[str, Any]] = {}
        self._refreshed_at = 0.0
        self._lock = threading.Lock()

    def refresh_static(self, *, force: bool = False) -> None:
        """Refresh immutable actor identity/geometry off the real-time path."""

        now = time.monotonic()
        with self._lock:
            if not force and now - self._refreshed_at < self.refresh_interval_s:
                return
        registry: dict[int, tuple[str, Any]] = {}
        for actor in self.world.get_actors():
            type_id = str(getattr(actor, "type_id", ""))
            if not (type_id.startswith("vehicle.") or type_id.startswith("walker.pedestrian.")):
                continue
            try:
                registry[int(actor.id)] = (type_id, actor.bounding_box)
            except RuntimeError:
                continue
        with self._lock:
            self._static = registry
            self._refreshed_at = now

    def capture(self, world_snapshot: Any) -> FrozenWorld:
        """Freeze the scene exactly as of ``world_snapshot``."""

        with self._lock:
            static = dict(self._static)
        actors: list[FrozenActor] = []
        for actor_id, (type_id, bounding_box) in static.items():
            if actor_id == self.ego_id:
                continue
            item = world_snapshot.find(int(actor_id))
            if item is None:
                continue
            actors.append(
                FrozenActor(
                    actor_id=actor_id, type_id=type_id, bounding_box=bounding_box,
                    transform=item.get_transform(), velocity=item.get_velocity(),
                )
            )
        return FrozenWorld(FrozenActorList(actors))


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
        self.stream_id = f"ue288_{cell['cell_id']}"
        qualification = campaign.get("_qualification")
        self.qualification_action_ids: tuple[int, ...] = ()
        self.qualification_transmit_order: tuple[int, ...] = ()
        self.qualification_interframe_drain_s = 0.0
        self.qualification_capture_limit: int | None = None
        if qualification is not None:
            require(isinstance(qualification, dict), "qualification contract is not a mapping")
            self.qualification_action_ids = tuple(
                int(value) for value in qualification.get("action_ids", ())
            )
            self.qualification_capture_limit = int(
                qualification.get("capture_limit", 0)
            )
            self.qualification_transmit_order = tuple(
                int(value) for value in qualification.get("transmit_order", ())
            )
            self.qualification_interframe_drain_s = float(
                qualification.get("interframe_drain_s", 0.0)
            )
            require(
                self.qualification_action_ids == (0, 20, 46, 71)
                and self.qualification_transmit_order == (71, 46, 20, 0)
                and self.qualification_interframe_drain_s == 3.0
                and self.qualification_capture_limit == 20,
                "live qualification action/order/drain/capture contract drift",
            )
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
        # The 100 ms service target classifies on-time installs.  The separate
        # 500 ms ACK timeout bounds useful processing and missing feedback.
        self.service_deadline_s = service_deadline_s(campaign)
        self.ack_timeout_s = ack_timeout_s(campaign)
        self.sensor_condition = threading.Condition()
        self.images: "OrderedDict[int, tuple[Any, float, float]]" = OrderedDict()
        self.semantic_images: "OrderedDict[int, Any]" = OrderedDict()
        self.radars: "OrderedDict[int, Any]" = OrderedDict()
        self.aggregator = RadarSweepAggregator(keep_sweeps=12)
        self.aggregator_error = ""
        # Bounded latest-frame-first preparation: at most one pending
        # unprepared frame per stream, a newer opportunity replaces an older
        # pending one, and a frame already executing is never interrupted.
        self.prepared_queue = LatestFramePendingSlot()
        self.segmentation_queue: "queue.Queue[int | None]" = queue.Queue(maxsize=32)
        # Evaluation-only work (ground truth, mask evidence, diagnostics) runs
        # here so it can never consume the real-time preparation period.
        self.evaluation_queue: "queue.Queue[dict[str, Any] | None]" = queue.Queue(maxsize=64)
        self.transport_counters = _Counters()
        self.scene_source: SceneSnapshotSource | None = None
        self.evaluation_errors: dict[int, str] = {}
        self.installed_at: dict[int, float] = {}
        self.evidence_manifest: dict[int, dict[str, Any]] = {}
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
        self.live_summary: dict[str, Any] = {}

        # The historical LR-ASPP/M-prime camera transport is deliberately not
        # instantiated here.  Route B provides only live RGB/radar preparation;
        # the qualified SFD1-v2 SplitFusion stack owns front, ranker, optional
        # AE, decode, frozen tail, p025 serialization, and map publication.
        self.model_size = (768, 448)
        self.live = LivePilotCellRuntime(
            campaign=campaign, cell=cell, attempt_dir=attempt_dir,
            map_host="127.0.0.1", map_port=int(campaign["runtime"]["map_ingest_port"]),
            evidence_dir=self.edge_evidence_dir,
        )
        self.target_start_file = Path(str(campaign["_target_start_file"])).resolve()
        self.profile_activated = False
        self.tracker = FastStationaryTrackAccumulator(
            stationary_velocity_mps=0.35, parked_threshold_s=5.0,
            association_grid_m=1.5, max_stale_s=2.0,
        )
        self.actor_tracker = parked.ActorStationaryTracker(0.35, 5.0)
        self.aligned_actor_tracker = parked.ActorStationaryTracker(0.35, 5.0)
        self.intrinsics = camera_intrinsics(
            int(self.model_size[0]), int(self.model_size[1]), 120.0
        )
        self.feedback = InstallFeedbackLedger(
            output_csv=attempt_dir / "map_feedback.csv",
            experiment_id=str(campaign["campaign_id"]), cell_id=str(cell["cell_id"]),
            bind_host="127.0.0.1", bind_port=self.feedback_port,
        )
        self._spawn_sensors()
        self.scene_source = SceneSnapshotSource(self.world, ego_id=int(self.ego.id))
        self.scene_source.refresh_static(force=True)
        self.worker = threading.Thread(target=self._worker, name="route-b-split-front", daemon=True)
        self.segmentation_worker = threading.Thread(
            target=self._segmentation_worker,
            name="route-b-segmentation-evaluation",
            daemon=True,
        )
        self.evaluation_worker = threading.Thread(
            target=self._evaluation_worker,
            name="route-b-object-gt-evaluation",
            daemon=True,
        )
        self.feedback_worker = threading.Thread(target=self._feedback_worker, name="route-b-map-feedback", daemon=True)
        self.worker.start()
        self.segmentation_worker.start()
        self.evaluation_worker.start()
        self.feedback_worker.start()

    def _spawn_sensors(self) -> None:
        bp_lib = self.world.get_blueprint_library()
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
        self.camera = self.world.spawn_actor(rgb_bp, sensor_transform(CAMERA_MOUNT), attach_to=self.ego)
        self.semantic_camera = self.world.spawn_actor(
            semantic_bp,
            sensor_transform(CAMERA_MOUNT),
            attach_to=self.ego,
        )
        self.radar = self.world.spawn_actor(radar_bp, sensor_transform(RADAR_MOUNT), attach_to=self.ego)
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
        if (
            self.qualification_capture_limit is not None
            and self.sent >= self.qualification_capture_limit
        ):
            return
        if (int(route_tick) - 1) % 2:
            return
        token = {
            "frame_id": int(frame_id), "route_tick": int(route_tick),
            "scheduled_perf": time.perf_counter(), "scheduled_wall": time.time(),
            "queue_depth": self.prepared_queue.depth(),
        }
        # Latest-frame-first: a newer complete opportunity displaces an older
        # pending one so the worker always serves the freshest frame instead of
        # draining an ever-older FIFO backlog.
        admitted, displaced = self.prepared_queue.offer(
            self.stream_id, token, sequence=int(frame_id)
        )
        self.transport_counters.bump("preparation_offers")
        if not admitted:
            self.dropped += 1
            self.transport_counters.bump("preparation_offer_refused")
            self._append_row({
                **token,
                "prepare_status": "DROPPED_PREPARATION_SLOT_CLOSED",
                "processing_late": 1,
            })
            return
        if displaced is not None:
            self.dropped += 1
            self.transport_counters.bump("preparation_pending_replacements")
            self._append_row({
                **displaced,
                "prepare_status": "DROPPED_REPLACED_BY_NEWER_FRAME",
                "replaced_by_frame_id": int(frame_id),
                "processing_late": 1,
            })

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
        world: Any | None = None,
        camera_location: Any | None = None,
    ) -> list[dict[str, Any]]:
        from pole_lraspp_multimodal_fusion.pole_lraspp_multimodal_fusion.object_targets import valid_localization_objects

        rows = self.parked.build_object_rows(
            world=self.world if world is None else world, ego_vehicle=self.ego,
            sample_base={"timestamp": float(timestamp), "frame_id": int(frame_id)},
            camera_location=(
                self.camera.get_transform().location
                if camera_location is None
                else camera_location
            ),
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
        """The real-time preparation worker.

        This thread contains only work required to acquire the synchronized
        sensor input, run the front/ranker/AE path, encode, send and record
        minimal timing/accounting metadata. Evaluation-only ground truth, mask
        evidence handling and diagnostic formatting run on other threads.
        """

        while True:
            taken = self.prepared_queue.take(timeout=0.1)
            if taken is None:
                if self.stop_event.is_set():
                    return
                continue
            _stream_id, token = taken
            try:
                self._process_token(token)
            except Exception as exc:
                message = f"frame {token['frame_id']}: {type(exc).__name__}: {exc}"
                self.failures.append(message)
                self._append_row({**token, "prepare_status": "SPLIT_PROCESSING_FAILED", "error": message})

    def _process_token(self, token: Mapping[str, Any]) -> None:
        if (
            self.qualification_capture_limit is not None
            and self.sent >= self.qualification_capture_limit
        ):
            return
        frame_id = int(token["frame_id"])
        process_started_perf = time.perf_counter()
        records = self._records_for(frame_id)
        sensor_ready_perf = time.perf_counter()
        sensor_wait_ms = (sensor_ready_perf - process_started_perf) * 1000.0
        if records is None:
            self.dropped += 1
            self._append_row({
                **token,
                "prepare_status": "DROPPED_SENSOR_LATE_OR_MISSING",
                "processing_late": 1,
                "sensor_wait_ms": sensor_wait_ms,
            })
            return
        require(not self.aggregator_error, self.aggregator_error)
        image, capture_wall, capture_perf, radar_measurement = records
        radar_window_started_perf = time.perf_counter()
        with self.sensor_condition:
            if self.aggregator.anchor_s is None:
                self.aggregator.set_anchor(float(radar_measurement.timestamp))
            sweep_index = self.aggregator.sweep_index_for(float(radar_measurement.timestamp))
            if not self.aggregator.has_window(sweep_index):
                self._append_row({**token, "carla_timestamp": radar_measurement.timestamp, "prepare_status": "WARMUP_NO_COMPLETE_RADAR_WINDOW"})
                return
            radar_inverse = actor_world_inverse_matrix(self.radar)
            detections, window_meta = self.aggregator.window_detections(
                sweep_index, sensor_inverse_matrix=radar_inverse,
                reference_timestamp_s=float(radar_measurement.timestamp),
            )
        radar_window_ms = (time.perf_counter() - radar_window_started_perf) * 1000.0
        if int(window_meta["callbacks"]) != 4:
            # CARLA sensor callbacks are asynchronous. A short sweep is an
            # unavailable preparation opportunity, not corruption of the
            # model/transport path and not a reason to invalidate the entire
            # live cell. It remains explicit in the 10 Hz denominator.
            self.dropped += 1
            self._append_row({
                **token,
                "carla_timestamp": float(radar_measurement.timestamp),
                "prepare_status": "DROPPED_INCOMPLETE_RADAR_WINDOW",
                "window_callbacks": int(window_meta["callbacks"]),
                "processing_late": 1,
            })
            return
        # The synchronized sensor input is now prepared.  The 100 ms target is
        # a classification boundary; the later ACK timeout is the processing
        # horizon after which work can no longer provide useful feedback.
        try:
            check_deadline(
                UE_STAGE_AFTER_PREPARATION,
                int(round(float(capture_wall) * 1_000_000_000)),
                self.ack_timeout_s,
            )
        except DeadlineExpired as expired:
            self.dropped += 1
            self.transport_counters.bump("stale_before_send")
            self.transport_counters.bump(f"deadline_drop_{expired.stage}")
            self._append_row({
                **token,
                "carla_timestamp": float(radar_measurement.timestamp),
                "capture_wall_s": float(capture_wall),
                "service_deadline_at": float(capture_wall) + self.service_deadline_s,
                "ack_timeout_at": float(capture_wall) + self.ack_timeout_s,
                "prepare_status": "STALE_BEFORE_SEND",
                "processing_late": 1,
                "deadline_expiry_stage": expired.stage,
                "deadline_expiry_age_ms": expired.age_ms,
                "queue_wait_ms": (
                    time.perf_counter() - float(token["scheduled_perf"])
                ) * 1000.0,
                "sensor_wait_ms": sensor_wait_ms,
                "radar_window_ms": radar_window_ms,
            })
            return
        camera_matrix = actor_world_matrix(self.camera)
        camera_inverse = actor_world_inverse_matrix(self.camera)
        radar_matrix = actor_world_matrix(self.radar)
        radar_prepare_started_perf = time.perf_counter()
        radar_tensor, radar_points, radar_summary = self.parked.build_radar_sample(
            detections=detections, sensor_matrix=radar_matrix,
            camera_inverse_matrix=camera_inverse, camera_intrinsics=self.intrinsics,
            width=int(self.model_size[0]), height=int(self.model_size[1]),
            frame_time_s=float(radar_measurement.timestamp), tracker=self.tracker,
            max_range_m=120.0, max_abs_velocity_mps=20.0,
            parked_threshold_s=5.0, point_radius_px=4, rasterizer="fast",
        )
        radar_prepare_ms = (time.perf_counter() - radar_prepare_started_perf) * 1000.0
        rgb_convert_started_perf = time.perf_counter()
        frame_bgr = carla_image_to_bgr(image)
        rgb_convert_ms = (time.perf_counter() - rgb_convert_started_perf) * 1000.0
        capture_id = f"{self.stream_id}:{frame_id}"
        action_id = (
            self.qualification_transmit_order[self.sent % len(self.qualification_transmit_order)]
            if self.qualification_transmit_order
            else int(self.cell["action_id"])
        )
        capture_timestamp_ns = int(round(float(capture_wall) * 1_000_000_000))
        service_deadline_at = float(capture_wall) + self.service_deadline_s
        ack_timeout_at = float(capture_wall) + self.ack_timeout_s
        queue_wait_ms = (time.perf_counter() - float(token["scheduled_perf"])) * 1000.0
        if not self.profile_activated:
            self.target_start_file.touch(exist_ok=False)
            self.profile_activated = True
        transform = self.ego.get_transform()
        location, rotation = transform.location, transform.rotation
        # Freeze the scene at exactly this synchronized frame so evaluation can
        # run off the real-time path and still use the correct scene state.
        scene = None
        scene_snapshot_started_perf = time.perf_counter()
        if self.scene_source is not None:
            try:
                scene = self.scene_source.capture(self.world.get_snapshot())
            except Exception as exc:
                self.evaluation_errors[frame_id] = (
                    f"SCENE_SNAPSHOT_FAILED:{type(exc).__name__}:{exc}"
                )
        scene_snapshot_ms = (time.perf_counter() - scene_snapshot_started_perf) * 1000.0
        pre_front_compute_ms = (
            time.perf_counter() - sensor_ready_perf
        ) * 1000.0
        # The capture's feedback obligation is created at the moment the UE
        # commits to transmitting -- after the pre-send deadline gate and
        # before the first datagram leaves. A frame refused as stale therefore
        # never registers, so no registration is ever retracted, and an ACK can
        # never arrive for an unregistered capture.
        def register() -> None:
            self.feedback.register_capture(
                stream_id=self.stream_id, capture_id=capture_id, frame_id=frame_id,
                capture_at=float(capture_wall), action_id=str(action_id),
                service_deadline_at=service_deadline_at,
                ack_timeout_at=ack_timeout_at,
            )

        front = self.live.submit(
            frame_bgr=frame_bgr, radar_tensor=radar_tensor, frame_id=frame_id,
            capture_timestamp_ns=capture_timestamp_ns,
            ego_pose=(float(location.x), float(location.y), float(location.z),
                      float(rotation.pitch), float(rotation.yaw), float(rotation.roll)),
            stream_id=self.stream_id, carla_timestamp=float(radar_measurement.timestamp),
            capture_id=capture_id, action_id=action_id, on_commit=register,
        )
        velocity = self.ego.get_velocity()
        ego_speed = math.sqrt(float(velocity.x) ** 2 + float(velocity.y) ** 2 + float(velocity.z) ** 2)
        activity = self._radar_activity(window_meta, radar_summary)
        common = {
            **token, "capture_id": capture_id,
            "carla_timestamp": float(radar_measurement.timestamp),
            "action_id": action_id, "capture_wall_s": float(capture_wall),
            "service_deadline_at": service_deadline_at,
            "ack_timeout_at": ack_timeout_at,
            "queue_wait_ms": queue_wait_ms,
            "sensor_wait_ms": sensor_wait_ms,
            "radar_window_ms": radar_window_ms,
            "radar_prepare_ms": radar_prepare_ms,
            "rgb_convert_ms": rgb_convert_ms,
            "scene_snapshot_ms": scene_snapshot_ms,
            "pre_front_compute_ms": pre_front_compute_ms,
            "window_sweeps": "|".join(str(v) for v in window_meta["sweep_indices"]),
            "window_callbacks": window_meta["callbacks"],
            "window_returns": window_meta["returns"],
            "window_span_s": window_meta["window_span_s"],
            "ego_speed_mps": ego_speed, **activity,
        }
        if not front.get("sent", False):
            # Expired between preparation and transmission: the radio never
            # carried it and it never registered a feedback obligation.
            self.dropped += 1
            self.transport_counters.bump("stale_before_send")
            self._append_row({
                **common,
                "prepare_status": str(front.get("prepare_status") or "STALE_BEFORE_SEND"),
                "processing_late": 1,
                "deadline_expiry_stage": str(front.get("stale_stage") or ""),
                "deadline_expiry_age_ms": front.get("stale_age_ms", ""),
            })
            return
        self.sent += 1
        self.sent_frames.add(frame_id)
        try:
            self.segmentation_queue.put_nowait(frame_id)
        except queue.Full:
            self.segmentation_evidence_errors[frame_id] = "GT_EVALUATION_QUEUE_FULL"
        # Evaluation-only ground truth is handed to its own bounded queue with
        # the frozen scene, so it can never consume the real-time period.
        try:
            self.evaluation_queue.put_nowait({
                "frame_id": frame_id,
                "timestamp": float(radar_measurement.timestamp),
                "camera_matrix": camera_matrix,
                "camera_inverse": camera_inverse,
                "camera_location": self.camera.get_transform().location,
                "radar_points": radar_points,
                "scene": scene,
            })
            self.transport_counters.bump("evaluation_tickets_queued")
        except queue.Full:
            self.transport_counters.bump("evaluation_tickets_dropped_queue_full")
            self.evaluation_errors[frame_id] = "OBJECT_GT_EVALUATION_QUEUE_FULL"
        self._append_row({
            **common,
            "prepare_status": "SENT",
            "processing_late": int(time.time() > service_deadline_at),
            "front_ms": front.get("front_ms", ""),
            "payload_bytes": front.get("payload_bytes", ""),
            "payload_bytes_uncompressed": front.get("payload_bytes_uncompressed", ""),
            "payload_chunks": front.get("payload_chunks", ""),
        })
        if self.qualification_interframe_drain_s > 0.0:
            drain_deadline = time.monotonic() + self.qualification_interframe_drain_s
            while time.monotonic() < drain_deadline and not self.stop_event.is_set():
                metric = self.live.take_metric(frame_id)
                if metric and metric.get("edge_result_received_ns") not in (None, ""):
                    break
                time.sleep(0.02)

    def _recovery_accounting(
        self,
        rows: Sequence[Mapping[str, Any]],
        feedback_rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Reconcile the Phase-15 real-time counters into one auditable block."""

        ue = self.transport_counters.snapshot()
        transport = dict(self.live_summary.get("transport_counters") or {})
        if not transport:
            transport = self.live.counters.snapshot()
        edge = self._read_edge_counters()
        status_counts: dict[str, int] = {}
        for row in rows:
            status = str(row.get("prepare_status") or "")
            status_counts[status] = status_counts.get(status, 0) + 1
        expiry_stages: dict[str, int] = {}
        for row in rows:
            stage = str(row.get("deadline_expiry_stage") or "")
            if stage:
                expiry_stages[stage] = expiry_stages.get(stage, 0) + 1
        with self.gt_lock:
            installed = dict(self.installed_at)
            manifest = {int(key): dict(value) for key, value in self.evidence_manifest.items()}
        aoi_ms: list[float] = []
        for row in rows:
            frame_id = int(row.get("frame_id") or -1)
            capture_wall = row.get("capture_wall_s")
            if frame_id in installed and capture_wall not in (None, ""):
                aoi_ms.append(
                    (float(installed[frame_id]) - float(capture_wall)) * 1000.0
                )
        aoi_ms.sort()

        def quantile(values: list[float], fraction: float) -> float | None:
            if not values:
                return None
            index = min(len(values) - 1, max(0, int(round(fraction * (len(values) - 1)))))
            return float(values[index])

        service_on_time = [
            value for value in aoi_ms
            if value <= self.service_deadline_s * 1000.0
        ]
        ack_within_timeout = [
            row for row in feedback_rows
            if str(row.get("status") or "") == "ACK_INSTALLED"
            and str(row.get("terminal", "")).lower() in {"1", "true"}
            and row.get("feedback_received_at") not in (None, "")
            and row.get("ack_timeout_at") not in (None, "")
            and float(row["feedback_received_at"]) <= float(row["ack_timeout_at"])
        ]
        edge_deadline_drops = {
            key.replace("deadline_drop_", ""): int(value)
            for key, value in {**edge.get("counters", {}), **transport}.items()
            if key.startswith("deadline_drop_")
        }
        for key, value in ue.items():
            if key.startswith("deadline_drop_"):
                stage = key.replace("deadline_drop_", "")
                edge_deadline_drops[stage] = edge_deadline_drops.get(stage, 0) + int(value)
        return {
            "service_deadline_s": self.service_deadline_s,
            "ack_timeout_s": self.ack_timeout_s,
            "processing_expiry_s": self.ack_timeout_s,
            "service_deadline_and_ack_timeout_are_distinct": True,
            "dense_label_map_on_radio": False,
            "prepare_status_counts": status_counts,
            "deadline_expiry_stage_counts_ue_rows": expiry_stages,
            "deadline_drops_by_stage": dict(sorted(edge_deadline_drops.items())),
            "registered_deadline_stages": list(DEADLINE_STAGES),
            "ue_counters": dict(sorted(ue.items())),
            "ue_transport_counters": dict(sorted(transport.items())),
            "edge_counters_file": edge,
            "maps_installed": len(installed),
            "install_aoi_ms_median": quantile(aoi_ms, 0.5),
            "install_aoi_ms_p95": quantile(aoi_ms, 0.95),
            "install_aoi_ms_max": (max(aoi_ms) if aoi_ms else None),
            # Backward-compatible names now retain their contract meaning:
            # "timely" is the 100 ms service target, never the 500 ms ACK
            # observation timeout.
            "timely_installations": len(service_on_time),
            "timely_installation_fraction": (
                len(service_on_time) / len(aoi_ms) if aoi_ms else None
            ),
            "service_on_time_installations": len(service_on_time),
            "service_on_time_fraction": (
                len(service_on_time) / len(aoi_ms) if aoi_ms else None
            ),
            "ack_within_timeout_installations": len(ack_within_timeout),
            "ack_within_timeout_fraction": (
                len(ack_within_timeout) / len(aoi_ms) if aoi_ms else None
            ),
            "install_aoi_monotonic_fraction": _monotonic_fraction(
                [
                    (float(row.get("capture_wall_s") or 0.0),
                     (float(installed[int(row["frame_id"])]) - float(row["capture_wall_s"])) * 1000.0)
                    for row in rows
                    if int(row.get("frame_id") or -1) in installed
                    and row.get("capture_wall_s") not in (None, "")
                ]
            ),
            "evaluation_masks_manifest_records": len(manifest),
            "evaluation_masks_hash_verified": sum(
                1 for value in manifest.values() if value.get("hash_verified")
            ),
            "evaluation_masks_hash_mismatched": sum(
                1 for value in manifest.values() if not value.get("hash_verified")
            ),
            "terminal_feedback_rows": sum(
                1 for row in feedback_rows
                if str(row.get("terminal", "")).lower() in {"1", "true"}
            ),
            "late_feedback_rows": sum(
                1 for row in feedback_rows
                if str(row.get("late", "")).lower() in {"1", "true"}
            ),
            "late_nonterminal_feedback_rows": sum(
                1 for row in feedback_rows
                if str(row.get("late", "")).lower() in {"1", "true"}
                and str(row.get("terminal", "")).lower() not in {"1", "true"}
            ),
            "duplicate_result_messages": int(transport.get("duplicate_result_messages", 0)),
            "counter_reconciliation": self._reconcile_counters(
                rows, ue, transport, edge, len(installed)
            ),
        }

    def _reconcile_counters(
        self,
        rows: Sequence[Mapping[str, Any]],
        ue: Mapping[str, int],
        transport: Mapping[str, int],
        edge: Mapping[str, Any],
        installed: int,
    ) -> dict[str, Any]:
        """State each funnel identity and whether the counters satisfy it."""

        edge_counts = dict(edge.get("counters", {})) if edge.get("available") else {}
        sent_rows = sum(1 for row in rows if str(row.get("prepare_status")) == "SENT")

        def drops(prefix: str, source: Mapping[str, int]) -> int:
            return sum(
                int(value) for key, value in source.items()
                if key.startswith(f"deadline_drop_{prefix}")
            )

        checks: dict[str, Any] = {
            "ue_sent_rows_equal_transmitted_messages": {
                "sent_rows": sent_rows,
                "transmitted_messages": int(transport.get("feature_messages_transmitted", 0)),
                "holds": sent_rows == int(transport.get("feature_messages_transmitted", 0)),
            },
            "installed_maps_never_exceed_published_results": {
                "installed": installed,
                "published": int(transport.get("results_published_to_map", 0)),
                "holds": installed <= int(transport.get("results_published_to_map", 0)),
            },
            "published_results_never_exceed_reassembled_results": {
                "published": int(transport.get("results_published_to_map", 0)),
                "reassembled": int(transport.get("result_messages_reassembled", 0)),
                "holds": int(transport.get("results_published_to_map", 0))
                <= int(transport.get("result_messages_reassembled", 0)),
            },
        }
        if edge_counts:
            reassembled = int(edge_counts.get("feature_messages_reassembled", 0))
            admitted = int(edge_counts.get("edge_queue_admissions", 0))
            after_reassembly_drops = drops("EDGE_AFTER_REASSEMBLY", edge_counts)
            rejected = sum(
                int(edge_counts.get(name, 0))
                for name in (
                    "feature_envelope_rejected", "feature_action_outside_allowlist",
                    "feature_missing_frame_context", "edge_admission_refused_not_freshest",
                )
            )
            process_starts = int(edge_counts.get("edge_process_starts", 0))
            tail_starts = int(edge_counts.get("tail_starts", 0))
            tail_completions = int(edge_counts.get("tail_completions", 0))
            transmitted = int(edge_counts.get("compact_results_transmitted", 0))
            checks["edge_reassembled_equals_admitted_plus_dropped_plus_rejected"] = {
                "reassembled": reassembled,
                "admitted": admitted,
                "after_reassembly_deadline_drops": after_reassembly_drops,
                "rejected": rejected,
                "holds": reassembled == admitted + after_reassembly_drops + rejected,
            }
            checks["edge_tail_completions_never_exceed_starts"] = {
                "process_starts": process_starts,
                "starts": tail_starts, "completions": tail_completions,
                "pre_tail_deadline_refusals": drops("EDGE_BEFORE_TAIL", edge_counts),
                "holds": tail_completions <= tail_starts <= process_starts,
            }
            checks["edge_results_transmitted_never_exceed_tail_completions"] = {
                "transmitted": transmitted, "completions": tail_completions,
                "holds": transmitted <= tail_completions,
            }
            checks["edge_masks_persisted_match_results_transmitted"] = {
                "persisted": int(edge_counts.get("evaluation_masks_persisted", 0)),
                "transmitted": transmitted,
                "holds": int(edge_counts.get("evaluation_masks_persisted", 0)) <= transmitted,
            }
        checks["all_identities_hold"] = all(
            bool(value.get("holds")) for value in checks.values()
            if isinstance(value, dict) and "holds" in value
        )
        return checks

    def _read_edge_counters(self) -> dict[str, Any]:
        """Read the edge counters the edge persists outside the result path."""

        path = self.edge_evidence_dir.parent / "edge_counters.json"
        if not path.is_file():
            return {"available": False, "reason": "edge counters file absent"}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
        value["available"] = True
        return value

    def _verify_evidence_hash(
        self, frame_id: int, evidence_path: Path, predicted: np.ndarray
    ) -> None:
        """Verify one persisted label map against its edge-written sidecar."""

        sidecar = evidence_path.with_suffix(".json")
        if not sidecar.is_file():
            self.transport_counters.bump("evaluation_masks_missing_sidecar")
            self.segmentation_evidence_errors.setdefault(
                int(frame_id), "SEGMENTATION_EVIDENCE_SIDECAR_MISSING"
            )
            return
        try:
            record = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.transport_counters.bump("evaluation_masks_unreadable_sidecar")
            return
        digest = hashlib.sha256(
            np.ascontiguousarray(predicted).tobytes()
        ).hexdigest()
        expected = str(record.get("sha256") or "")
        matches = (
            digest == expected
            and int(record.get("frame_id", -1)) == int(frame_id)
            and str(record.get("stream_id") or "") == self.stream_id
        )
        with self.gt_lock:
            self.evidence_manifest[int(frame_id)] = {
                "evidence_name": evidence_path.name,
                "sha256": digest,
                "declared_sha256": expected,
                "hash_verified": bool(matches),
                "bytes": int(evidence_path.stat().st_size),
                "shape": [int(value) for value in predicted.shape],
                "dtype": str(predicted.dtype),
                "action_id": record.get("action_id", ""),
                "capture_timestamp_ns": record.get("capture_timestamp_ns", ""),
            }
        if matches:
            self.transport_counters.bump("evaluation_masks_hash_verified")
        else:
            self.transport_counters.bump("evaluation_masks_hash_mismatched")
            self.segmentation_evidence_errors.setdefault(
                int(frame_id), "SEGMENTATION_EVIDENCE_HASH_MISMATCH"
            )

    def preserve_evidence(self, destination: Path) -> dict[str, Any]:
        """Preserve required label maps under a registered per-cell quota.

        The complete hash manifest is always retained. Raw arrays are preserved
        for the frames evaluation actually requires -- the ACK-installed ones --
        up to the registered byte quota, so one cell can never exhaust the host
        filesystem while leaving the accounting incomplete.
        """

        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        with self.gt_lock:
            manifest = {int(key): dict(value) for key, value in self.evidence_manifest.items()}
            required = sorted(self.ack_installed_frames)
        preserved = 0
        preserved_bytes = 0
        elided = 0
        failed = 0
        for frame_id in required:
            record = manifest.get(int(frame_id))
            if record is None:
                continue
            source = self.edge_evidence_dir / str(record["evidence_name"])
            if not source.is_file():
                continue
            size = int(record.get("bytes", 0))
            if preserved_bytes + size > PRESERVED_EVIDENCE_QUOTA_BYTES:
                elided += 1
                record["preserved"] = False
                record["preservation_status"] = "ELIDED_BY_REGISTERED_QUOTA"
                continue
            try:
                shutil.copy2(source, destination / source.name)
                sidecar = source.with_suffix(".json")
                if sidecar.is_file():
                    shutil.copy2(sidecar, destination / sidecar.name)
                preserved += 1
                preserved_bytes += size
                record["preserved"] = True
                record["preservation_status"] = "PRESERVED"
            except OSError as exc:
                failed += 1
                record["preserved"] = False
                record["preservation_status"] = f"PRESERVATION_FAILED:{exc.__class__.__name__}"
        write_json_create_only(
            destination / "segmentation_evidence_manifest.json",
            {
                "schema": "scenesense.segmentation_evidence_manifest.v1",
                "cell_id": str(self.cell["cell_id"]),
                "action_id": int(self.cell["action_id"]),
                "stream_id": self.stream_id,
                "quota_bytes": PRESERVED_EVIDENCE_QUOTA_BYTES,
                "required_frames": len(required),
                "preserved_masks": preserved,
                "preserved_bytes": preserved_bytes,
                "quota_elided_masks": elided,
                "preservation_failures": failed,
                "hash_verified_masks": sum(
                    1 for value in manifest.values() if value.get("hash_verified")
                ),
                "hash_mismatched_masks": sum(
                    1 for value in manifest.values() if not value.get("hash_verified")
                ),
                "masks": [manifest[key] for key in sorted(manifest)],
            },
        )
        return {
            "required_frames": len(required),
            "preserved_masks": preserved,
            "preserved_bytes": preserved_bytes,
            "quota_elided_masks": elided,
            "preservation_failures": failed,
            "manifest_records": len(manifest),
            "hash_verified_masks": sum(
                1 for value in manifest.values() if value.get("hash_verified")
            ),
            "hash_mismatched_masks": sum(
                1 for value in manifest.values() if not value.get("hash_verified")
            ),
        }

    def _evaluation_worker(self) -> None:
        """Build evaluation-only object ground truth off the real-time path.

        Each ticket carries the scene frozen at the synchronized CARLA frame the
        features came from, so deferring this work does not change which scene
        state the ground truth describes. Nothing here is ever fed back to the
        front, edge, map, action or route controller.
        """

        while True:
            try:
                ticket = self.evaluation_queue.get(timeout=0.05)
            except queue.Empty:
                if self.stop_event.is_set() and self.evaluation_queue.empty():
                    return
                continue
            if ticket is None:
                self.evaluation_queue.task_done()
                return
            frame_id = int(ticket["frame_id"])
            try:
                if self.scene_source is not None:
                    self.scene_source.refresh_static()
                gt = self._ground_truth(
                    frame_id=frame_id, timestamp=float(ticket["timestamp"]),
                    camera_matrix=ticket["camera_matrix"],
                    camera_inverse=ticket["camera_inverse"],
                    radar_points=ticket["radar_points"],
                    world=ticket.get("scene"),
                    camera_location=ticket.get("camera_location"),
                )
                with self.gt_lock:
                    self.source_gt[frame_id] = gt
                self.transport_counters.bump("evaluation_tickets_completed")
            except Exception as exc:
                self.evaluation_errors[frame_id] = (
                    f"OBJECT_GT_EVALUATION_FAILED:{type(exc).__name__}:{exc}"
                )
                self.transport_counters.bump("evaluation_tickets_failed")
            finally:
                self.evaluation_queue.task_done()

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
                        gt_3class = semantic_gt_3class(semantic_image)
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
                        # The edge persisted this label map on its own mount and
                        # bound it to a SHA-256 in a sidecar published first, so
                        # the evidence is verified here rather than trusted.
                        self._verify_evidence_hash(candidate, evidence_path, predicted)
                        quality = segmentation_quality_columns(predicted, gt_3class)
                        quality["prediction_vehicle_pixels"] = int(
                            np.count_nonzero(
                                predicted
                                == CLASS_ID_VEHICLE
                            )
                        )
                        quality["prediction_person_pixels"] = int(
                            np.count_nonzero(
                                predicted
                                == CLASS_ID_PERSON
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
                        install_at = received.get("install_timestamp", "")
                        if install_at not in (None, ""):
                            # MAP_INSTALLED is the terminal service-success
                            # event; a receipt ACK never reaches this branch.
                            self.installed_at[frame_id] = float(install_at)
                            self.transport_counters.bump("maps_installed")
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
                        camera_matrix=actor_world_matrix(self.camera),
                        camera_inverse=actor_world_inverse_matrix(self.camera),
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
        # The bounded slot holds at most one pending frame, so draining is
        # bounded by one preparation period rather than by a queue depth.
        deadline = time.monotonic() + 10.0
        while self.prepared_queue.depth() and time.monotonic() < deadline:
            time.sleep(0.1)
        ack_deadline = time.monotonic() + 2.0
        while self.feedback.pending and time.monotonic() < ack_deadline:
            time.sleep(0.1)
        self.feedback.record_expired(time.time() + self.ack_timeout_s + 1.0)
        self.stop_event.set()
        for abandoned in self.prepared_queue.close():
            self.dropped += 1
            self.transport_counters.bump("preparation_abandoned_at_teardown")
            self._append_row({
                **abandoned,
                "prepare_status": "DROPPED_PREPARATION_SLOT_CLOSED",
                "processing_late": 1,
            })
        with self.sensor_condition:
            self.sensor_condition.notify_all()
        self.worker.join(timeout=5.0)
        self.feedback_worker.join(timeout=3.0)
        object_gt_deadline = time.monotonic() + 5.0
        while (
            self.evaluation_queue.unfinished_tasks
            and time.monotonic() < object_gt_deadline
        ):
            time.sleep(0.05)
        try:
            self.evaluation_queue.put_nowait(None)
        except queue.Full:
            pass
        self.evaluation_worker.join(timeout=5.0)
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
            self.live_summary = self.live.close()
            sensor_ok = not bool(self.live_summary.get("errors")) and sensor_ok
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
            and not self.evaluation_worker.is_alive()
        )
        return self.cleanup_ok

    def write_per_frame(self) -> None:
        path = self.attempt_dir / "per_frame_metrics.csv"
        with path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(PER_FRAME_FIELDS))
            writer.writeheader()
            with self.rows_lock:
                rows = sorted(
                    self.rows,
                    key=lambda item: (
                        int(item.get("route_tick", 0)),
                        str(item.get("prepare_status", "")),
                    ),
                )
            with self.gt_lock:
                installed = dict(self.installed_at)
                evaluation_errors = dict(self.evaluation_errors)
                object_gt = set(self.source_gt)
            for row in rows:
                frame_id = int(row["frame_id"])
                metric = (
                    self.live.take_metric(frame_id)
                    if row.get("prepare_status") == "SENT"
                    else None
                )
                merged: dict[str, Any] = {**row, **(metric or {})}
                install_at = installed.get(frame_id)
                if install_at is not None:
                    # MAP_INSTALLED, measured from the original CARLA capture.
                    merged["map_installed_at"] = install_at
                    capture_wall = merged.get("capture_wall_s")
                    if capture_wall not in (None, ""):
                        merged["install_aoi_ms"] = (
                            float(install_at) - float(capture_wall)
                        ) * 1000.0
                if row.get("prepare_status") == "SENT":
                    merged["evaluation_gt_status"] = evaluation_errors.get(
                        frame_id, "OK" if frame_id in object_gt else "PENDING"
                    )
                writer.writerow(
                    {field: merged.get(field, "") for field in PER_FRAME_FIELDS}
                )

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
        performance_warnings: list[str] = []
        with self.rows_lock:
            rows = list(self.rows)
        scheduled_ticks = [int(row.get("route_tick", 0)) for row in rows]
        expected_last_tick = (
            max(scheduled_ticks, default=0)
            if self.qualification_capture_limit is not None
            else int(self.route_ticks)
        )
        expected_ticks = list(range(1, expected_last_tick + 1, 2))
        schedule_ok = sorted(scheduled_ticks) == expected_ticks
        if not schedule_ok and self.qualification_capture_limit is None:
            failures.append("10 Hz prepared-input scheduling phase/count contract failed")
        elif not schedule_ok:
            performance_warnings.append(
                "bounded qualification stopped after its capture quota; full-route 10 Hz count is diagnostic only"
            )

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
        preparation_coverage_met = preparation_coverage >= self.minimum_preparation_coverage
        if not preparation_coverage_met:
            performance_warnings.append(
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
        ack_actions: set[int] = set()
        for feedback_row in feedback_rows:
            status = str(feedback_row.get("status") or "")
            if status == "ACK_INSTALLED":
                ack_frames.add(int(feedback_row["frame_id"]))
                ack_actions.add(int(feedback_row["action_id"]))
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
        action_counts: dict[int, int] = {}
        for sent_row in sent_rows:
            action_id = int(sent_row["action_id"])
            action_counts[action_id] = action_counts.get(action_id, 0) + 1
        if self.qualification_capture_limit is not None:
            expected_action_counts = {
                action_id: self.qualification_capture_limit
                // len(self.qualification_action_ids)
                for action_id in self.qualification_action_ids
            }
            if len(sent_rows) != self.qualification_capture_limit:
                failures.append(
                    "live qualification did not produce exactly "
                    f"{self.qualification_capture_limit} sent captures"
                )
            if action_counts != expected_action_counts:
                failures.append(
                    "live qualification action counts drift: "
                    f"observed={action_counts} expected={expected_action_counts}"
                )
            missing_ack_actions = sorted(set(self.qualification_action_ids) - ack_actions)
            if not ack_actions:
                failures.append("live qualification has no end-to-end ACK_INSTALLED result")
            if missing_ack_actions:
                performance_warnings.append(
                    "live radio did not install every qualification action; "
                    f"uninstalled actions={missing_ack_actions}"
                )
        else:
            missing_ack_actions = []

        exact_frames = set(self.installed_predictions)
        missing_exact = sorted(ack_frames - exact_frames)
        if missing_exact:
            failures.append(f"ACK-installed frames lack exact map records: {missing_exact[:8]}")
        exact_coverage = (
            len(ack_frames & exact_frames) / len(ack_frames) if ack_frames else None
        )
        missing_segmentation = sorted(ack_frames - set(self.segmentation_quality))
        missing_segmentation_reasons = {
            frame_id: self.segmentation_evidence_errors.get(frame_id, "UNSPECIFIED")
            for frame_id in missing_segmentation
        }
        retention_expired = sorted(
            frame_id for frame_id, reason in missing_segmentation_reasons.items()
            if reason == "DECODED_MASK_NOT_OBSERVED_WITHIN_RETENTION"
        )
        fatal_missing_segmentation = {
            frame_id: reason for frame_id, reason in missing_segmentation_reasons.items()
            if frame_id not in retention_expired
        }
        if retention_expired:
            performance_warnings.append(
                "segmentation IoU unavailable after the registered evidence-retention window; "
                f"late installed frames={retention_expired[:8]}"
            )
        if fatal_missing_segmentation:
            failures.append(
                "ACK-installed frames lack exact segmentation IoU for a non-retention reason: "
                f"{dict(list(fatal_missing_segmentation.items())[:8])}"
            )

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
            "realtime_recovery": self._recovery_accounting(rows, feedback_rows),
            "expected_prepared_hz": self.expected_prepared_hz,
            "route_ticks": int(self.route_ticks),
            "scheduled_frames": len(rows),
            "expected_scheduled_frames": len(expected_ticks),
            "schedule_phase_and_count_ok": schedule_ok,
            "eligible_preparation_frames": len(eligible_rows),
            "sent_frames": len(sent_rows),
            "minimum_sensor_preparation_coverage": self.minimum_preparation_coverage,
            "sensor_preparation_coverage": preparation_coverage,
            "sensor_preparation_coverage_met": preparation_coverage_met,
            "terminal_feedback_records": sum(terminal_counts.values()),
            "terminal_feedback_outcomes": dict(sorted(outcome_counts.items())),
            "ack_installed_frames": len(ack_frames),
            "action_capture_counts": dict(sorted(action_counts.items())),
            "ack_installed_actions": sorted(ack_actions),
            "qualification_transmit_order": list(self.qualification_transmit_order),
            "qualification_interframe_drain_s": self.qualification_interframe_drain_s,
            "qualification_actions_without_live_install": missing_ack_actions,
            "exact_frame_perception_records": len(ack_frames & exact_frames),
            "exact_frame_perception_coverage": exact_coverage,
            "exact_frame_segmentation_records": len(ack_frames & set(self.segmentation_quality)),
            "segmentation_evidence_install_coverage": (
                len(ack_frames & set(self.segmentation_quality)) / len(ack_frames)
                if ack_frames else None
            ),
            "segmentation_retention_expired_installed_frames": retention_expired,
            "segmentation_evidence_unavailability_is_measured_not_structurally_invalid": True,
            "performance_warnings": performance_warnings,
            "low_preparation_or_delivery_is_measured_not_structurally_invalid": True,
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
    import pole_lraspp_multimodal_fusion as fusion_namespace
    legacy_package = ROOT / "pole_lraspp_multimodal_fusion" / "pole_lraspp_multimodal_fusion"
    require(legacy_package.is_dir(), "legacy fusion package is missing")
    namespace_paths = {Path(value).resolve() for value in fusion_namespace.__path__}
    if legacy_package.resolve() not in namespace_paths:
        fusion_namespace.__path__.append(str(legacy_package.resolve()))

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
    binding: dict[str, Any] = {"dispatcher": "phase13_sfd1_v2"}
    map_process: subprocess.Popen[Any] | None = None
    target_process: subprocess.Popen[Any] | None = None
    target_output = Path()
    target_stop = Path()
    edge_scratch: Path | None = None
    edge_startup: dict[str, Any] = {}
    edge_mounts: dict[str, Any] = {}
    collector: PassiveSplitCollector | None = None
    route_detail: dict[str, Any] = {}
    structural_acceptance: dict[str, Any] = {
        "status": "FAIL",
        "failures": ["Route B split collector did not reach structural validation"],
    }
    failures: list[str] = []
    cleanup: dict[str, Any] = {
        "target_snr_restored": False,
        "target_snr_restore_status": "not_evaluated",
        "map_process_stopped": False,
        "live_dispatch_stopped": False,
        "edge_stopped": False,
    }
    started = time.time()
    evidence_preservation: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="ue_288_cell_runtime_") as raw_tmp:
        temporary = Path(raw_tmp)
        # The evaluation label map is written by the edge on its own per-cell
        # mount, so the consumer reads the edge evidence leaf directly instead
        # of a UE-side directory fed by the radio payload.
        edge_evidence_dir = Path(temporary) / "unstarted_edge_evidence"
        try:
            map_process = start_map_process(
                campaign, temporary_dir=temporary, action_id=str(cell["action_id"]),
                carla_host=args.carla_host, carla_port=args.carla_port,
                api_port=args.map_api_port, udp_port=args.spatial_map_port,
                feedback_port=args.feedback_port,
            )
            edge_scratch = start_live_edge(campaign, cell, temporary)
            edge_evidence_dir = edge_scratch / EDGE_EVIDENCE_LEAF
            require(
                edge_evidence_dir.is_dir(),
                "edge evaluation-evidence mount leaf is missing",
            )
            edge_startup = load_json(edge_scratch / "ready.json")
            edge_mounts = inspect_live_edge_mounts(edge_scratch)
            # The target runtime reads the campaign root, not the resolved-cell
            # wrapper. Supply an isolated copy containing exactly that mapping.
            campaign_copy = temporary / "campaign.yaml"
            target_start = temporary / "target_snr_start"
            campaign["_target_start_file"] = str(target_start)
            campaign_copy.write_text(yaml.safe_dump(campaign, sort_keys=False), encoding="utf-8")
            target_process, target_output, target_stop = start_target_snr(
                campaign, campaign_path=campaign_copy,
                profile_id=str(cell["network_profile_id"]), temporary_dir=temporary,
                start_file=target_start,
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
                    cleanup["target_snr_restore_status"] = (
                        "restored_and_read_back"
                        if cleanup["target_snr_restored"]
                        else "restore_or_read_back_failed"
                    )
                except Exception as exc:
                    failures.append(f"target-SNR cleanup: {type(exc).__name__}: {exc}")
            else:
                try:
                    cleanup["target_snr_restored"] = verify_clean_rfsim_without_runtime(campaign)
                    cleanup["target_snr_restore_status"] = (
                        "restoration_not_required_clean_state_verified"
                        if cleanup["target_snr_restored"]
                        else "clean_state_read_back_failed"
                    )
                except Exception as exc:
                    failures.append(f"clean RFsim read-back: {type(exc).__name__}: {exc}")
            cleanup["map_process_stopped"] = stop_process(map_process)
            cleanup["live_dispatch_stopped"] = bool(
                collector is not None and collector.cleanup_ok
            )
            if collector is not None and edge_scratch is not None:
                try:
                    evidence_preservation = collector.preserve_evidence(
                        attempt_dir / PRESERVED_EVIDENCE_LEAF
                    )
                except Exception as exc:
                    failures.append(
                        f"segmentation evidence preservation: {type(exc).__name__}: {exc}"
                    )
            cleanup["edge_stopped"] = stop_live_edge(edge_scratch)
            if not all(
                bool(cleanup[name])
                for name in (
                    "target_snr_restored", "map_process_stopped",
                    "live_dispatch_stopped", "edge_stopped",
                )
            ):
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
        from rl_agent.splitfusion_live_dispatch_v1.live_pilot_target_snr_runtime import FIELDS as radio_fields
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
        "live_dispatch": collector.live_summary if collector else {},
        "edge_startup": edge_startup,
        "edge_mounts": edge_mounts,
        "segmentation_evidence_preservation": evidence_preservation,
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
    campaign = load_yaml(configs[0].resolve())
    certified_runtime = repo_path(str(campaign["runtime"]["split_inference_runtime"]))
    require(
        sha256_file(certified_runtime) == str(campaign["runtime"]["split_inference_runtime_sha256"]),
        "configured split runtime hash changed while adding evaluation evidence",
    )
    print(json.dumps({
        "status": "ADAPTER_CONTRACT_DRY_RUN_PASS", "configs": reports,
        "external_processes_started": 0, "ego_owner": "Route B",
        "clock_owner": "Route B through imported SamplingWorld",
        "adapter_forbidden_clock_calls": forbidden_calls,
        "configured_split_runtime_sha256": sha256_file(certified_runtime),
        "exact_installed_frame_history": "PASS",
        "primary_match_distance_m": 3.0,
        "oriented_footprint_iou": "PASS",
        "segmentation_evaluation_path": "EDGE_PERSISTED_EVIDENCE_MOUNT_HASH_VERIFIED",
        "selected_oai_radio_profile": "OAI_N78_100MHZ_273PRB_4D5U_V1",
        "real_launch_status": "QUALIFIED_CONTRACT_PENDING_GUARDED_LIVE_PREFLIGHT",
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
