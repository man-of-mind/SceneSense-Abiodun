#!/usr/bin/env python3
"""Live CARLA/OAI SplitFusion timing diagnostic: four short bounded cells.

Amendment scope. Each action runs one fresh CARLA lifecycle on the qualified
live Route-B 50-vehicle / 50-pedestrian configuration -- the same route file,
scenario seed 31, traffic-manager seed 31, Epic quality, sensor rig and
traffic contract as the qualified adapter -- behind one fresh
CN5G/gNB/UE/edge lifecycle with FAVORABLE_STABLE restarted from its identical
first network-profile sample. One action is fixed for the whole cell. The cell
collects exactly ``TRANSMITTED_BUDGET`` successfully transmitted SplitFusion
frames and stops immediately after the last one, under a hard
``SAFETY_TIMEOUT_S`` guard. The full Route-B loop is deliberately not
completed. CARLA and OAI are torn down completely before the next action.

Nothing here replays a stored dataset frame. Every measured payload is built
from live CARLA RGB + radar. Dataset access is limited to zero frames; the
only synthetic input in the process is the edge's CUDA warm-up, which uses a
seeded random 7-channel tensor and is excluded from every statistic.

Sensor preparation and CARLA waiting cannot enter an uplink interval:
application-level uplink timing starts at the wall clock taken immediately
before the first UDP datagram of a frame is handed to the socket.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

from rl_agent.splitfusion_live_dispatch_v1.frame_context import build_frame_context_v1
from rl_agent.splitfusion_live_dispatch_v1.registry import SplitActionRegistry
from rl_agent.splitfusion_live_dispatch_v1.timing import EDGE_STAGES

from . import diagnostic_common as common
from . import runner
from .diagnostic_common import ROOT, DiagnosticError, repo_path, require
from .edge_preload import preload_ue
from .instrumented_tail import SERIALIZE_STAGE, TAIL_STAGES, TOTAL_STAGE
from .live_capture import (
    GpuSampler,
    RouteBudgetReached,
    build_collector_class,
    build_runtime_class,
)


LIVE_EXECUTE_TOKEN = "SPLITFUSION_TIMING_DIAGNOSTIC_LIVE_V1_EXECUTE"
LIVE_SCHEMA = "scenesense.splitfusion_timing_diagnostic_live.v1"
LIVE_MANIFEST_SCHEMA = "scenesense.splitfusion_timing_diagnostic_live_manifest.v1"
LIVE_OUTPUT_RELPATH = (
    "experiments/splitfusion_timing_diagnostic_v1/20260909_live_carla_actions30_15_50_71"
)
TERMINAL_SUCCESS = "SPLITFUSION_TIMING_DIAGNOSTIC_LIVE_V1_COMPLETE"
TERMINAL_FAILURE = "SPLITFUSION_TIMING_DIAGNOSTIC_LIVE_V1_FAILED"

TRANSMITTED_BUDGET = 300
SAFETY_TIMEOUT_S = 90.0
WARMUP_ITERATIONS = 12
CARLA_RPC_PORT = 2000
MAP_API_PORT = 35001
SPATIAL_MAP_PORT = 39310
FEEDBACK_PORT = 39401
CARLA_RPC_TIMEOUT_S = 240.0

ROUTE_B_BOUND_INPUTS = (
    ("route_json", "route_json_sha256"),
    ("progress_csv", "progress_csv_sha256"),
    ("qualified_density_runner", "qualified_density_runner_sha256"),
)


def _adapter() -> Any:
    from rl_agent import ue_route_b_split_cell_adapter_v1 as adapter

    return adapter


def _lifecycle(campaign: Mapping[str, Any]) -> Any:
    import importlib.util

    path = repo_path(str(campaign["runtime"]["carla_lifecycle_helper"]))
    spec = importlib.util.spec_from_file_location("route_b_carla_lifecycle", path)
    require(spec is not None and spec.loader is not None, "cannot import the CARLA lifecycle helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_route_b_bindings(campaign: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the live Route-B route, progress and density runner by SHA-256."""

    route = campaign["route_b"]
    bindings: dict[str, str] = {}
    for path_key, hash_key in ROUTE_B_BOUND_INPUTS:
        path = repo_path(str(route[path_key]))
        require(path.is_file(), f"Route-B input is missing: {route[path_key]}")
        observed = common.sha256_file(path)
        require(
            observed == str(route[hash_key]),
            f"Route-B input hash drift: {route[path_key]} {observed}",
        )
        bindings[str(route[path_key])] = observed
    adapter_path = repo_path(str(campaign["runtime"]["required_route_b_split_cell_adapter"]))
    observed = common.sha256_file(adapter_path)
    require(
        observed == str(campaign["runtime"]["required_route_b_split_cell_adapter_sha256"]),
        f"qualified Route-B adapter hash drift: {observed}",
    )
    bindings[str(campaign["runtime"]["required_route_b_split_cell_adapter"])] = observed
    require(
        str(route["density"]) == "traffic_50_50"
        and int(route["scenario_seed"]) == 31
        and int(route["traffic_manager_seed"]) == 31
        and str(route["carla_quality"]) == "Epic"
        and bool(route["fresh_carla_process_and_world_per_cell"]) is True
        and int(route["loops_per_process"]) == 1,
        "live Route-B traffic/seed/quality contract drift",
    )
    return {
        "bound_inputs": bindings,
        "density": str(route["density"]),
        "vehicles": 50,
        "pedestrians": 50,
        "scenario_seed": int(route["scenario_seed"]),
        "traffic_manager_seed": int(route["traffic_manager_seed"]),
        "carla_quality": str(route["carla_quality"]),
        "route_json": str(route["route_json"]),
        "fresh_carla_per_action": True,
        "full_route_loop_completed": False,
    }


def synthetic_warmup_payloads(
    *, ue: Any, profile: Any, stream_id: str, count: int
) -> list[bytes]:
    """Seeded synthetic 7-channel warm-up inputs; never a dataset frame.

    The amendment forbids replaying stored dataset frames, and the original
    instruction to warm every resident model and CUDA path before recording
    still stands. These payloads exist only to compile kernels and populate
    caches at the edge; they carry their own stream identity so the edge marks
    them WARMUP and excludes them from every statistic.
    """

    generator = torch.Generator(device="cpu").manual_seed(20260909)
    base = torch.randn((1, 7, 448, 768), generator=generator, dtype=torch.float32)
    payloads: list[bytes] = []
    anchor_ns = time.time_ns()
    for index in range(int(count)):
        sequence_id = index + 1
        capture_ns = anchor_ns + index * 1_000_000
        with torch.inference_mode():
            prepared = ue.prepare(
                profile.action_id,
                base.to(ue.device),
                sequence_id=sequence_id,
                capture_timestamp_ns=capture_ns,
                frame_context=build_frame_context_v1(
                    stream_id=stream_id,
                    frame_id=sequence_id,
                    sequence_id=sequence_id,
                    capture_timestamp_ns=capture_ns,
                    ego_world_x=0.0, ego_world_y=0.0, ego_world_z=0.0,
                    ego_world_pitch=0.0, ego_world_yaw=0.0, ego_world_roll=0.0,
                ),
            )
        payloads.append(bytes(prepared.wire_bytes))
    return payloads


def _install_live_wrappers(adapter: Any) -> dict[str, Any]:
    """Swap in the instrumented runtime and the bounded collector at run time.

    Both are subclasses of the qualified classes, installed by assignment, so
    no SHA-256 pinned file is edited and the deep SFD1 authority is untouched.
    """

    original = {
        "PassiveSplitCollector": adapter.PassiveSplitCollector,
        "LivePilotCellRuntime": adapter.LivePilotCellRuntime,
    }
    runtime_class = build_runtime_class(adapter.LivePilotCellRuntime)
    collector_class = build_collector_class(adapter.PassiveSplitCollector)
    collector_class.transmitted_budget = TRANSMITTED_BUDGET
    collector_class.safety_timeout_s = SAFETY_TIMEOUT_S
    adapter.LivePilotCellRuntime = runtime_class
    adapter.PassiveSplitCollector = collector_class
    return original


def _restore_live_wrappers(adapter: Any, original: Mapping[str, Any]) -> None:
    adapter.PassiveSplitCollector = original["PassiveSplitCollector"]
    adapter.LivePilotCellRuntime = original["LivePilotCellRuntime"]


def join_live_records(
    *,
    collector_rows: Sequence[Mapping[str, Any]],
    send_boundaries: Mapping[int, Mapping[str, int]],
    edge_records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join UE preparation, UE send boundaries and edge records by frame id."""

    edge_by_frame = {int(record["frame_id"]): record for record in edge_records}
    require(
        len(edge_by_frame) == len(edge_records),
        "edge records contain duplicate frame identities",
    )
    sent_rows = [
        row for row in collector_rows if str(row.get("prepare_status")) == "SENT"
    ]
    drops: dict[str, int] = {}
    for row in collector_rows:
        status = str(row.get("prepare_status") or "UNKNOWN")
        drops[status] = drops.get(status, 0) + 1

    joined: list[dict[str, Any]] = []
    for row in sent_rows:
        frame_id = int(row["frame_id"])
        boundary = send_boundaries.get(frame_id, {})
        edge = edge_by_frame.get(frame_id)
        first = int(boundary.get("ue_first_send_wall_ns", 0)) or None
        final = int(boundary.get("ue_final_send_wall_ns", 0)) or None
        entry: dict[str, Any] = {
            "frame_id": frame_id,
            "capture_id": str(row.get("capture_id", "")),
            "action_id": int(row.get("action_id", -1)),
            "route_tick": row.get("route_tick"),
            "carla_timestamp": row.get("carla_timestamp"),
            "capture_wall_s": row.get("capture_wall_s"),
            "ego_speed_mps": row.get("ego_speed_mps"),
            # Preparation is reported but is structurally outside every uplink
            # interval: uplink timing starts at the first datagram send.
            "prep_queue_wait_ms": row.get("queue_wait_ms"),
            "prep_sensor_wait_ms": row.get("sensor_wait_ms"),
            "prep_radar_window_ms": row.get("radar_window_ms"),
            "prep_radar_prepare_ms": row.get("radar_prepare_ms"),
            "prep_rgb_convert_ms": row.get("rgb_convert_ms"),
            "prep_scene_snapshot_ms": row.get("scene_snapshot_ms"),
            "prep_pre_front_compute_ms": row.get("pre_front_compute_ms"),
            "window_callbacks": row.get("window_callbacks"),
            "window_returns": row.get("window_returns"),
            "ue_front_ms": row.get("front_ms"),
            "payload_bytes": row.get("payload_bytes"),
            "payload_bytes_uncompressed": row.get("payload_bytes_uncompressed"),
            "payload_chunks": row.get("payload_chunks"),
            "ue_first_send_wall_ns": first,
            "ue_final_send_wall_ns": final,
            "ue_send_loop_ms": (final - first) / 1e6 if first and final else None,
            "ue_datagrams_observed": boundary.get("ue_datagrams_observed"),
            "delivered": edge is not None,
        }
        if edge is not None:
            reassembled = int(edge["edge_complete_reassembly_wall_ns"])
            first_datagram = int(edge["edge_first_datagram_wall_ns"])
            worker_start = int(edge["edge_worker_start_wall_ns"])
            published = int(edge.get("edge_result_published_wall_ns", 0))
            entry.update(
                {
                    "edge_first_datagram_wall_ns": first_datagram,
                    "edge_complete_reassembly_wall_ns": reassembled,
                    "edge_admitted_wall_ns": int(edge["edge_admitted_wall_ns"]),
                    "edge_worker_start_wall_ns": worker_start,
                    "edge_tail_finished_wall_ns": int(edge["edge_tail_finished_wall_ns"]),
                    "edge_result_published_wall_ns": published or None,
                    "edge_feature_datagrams": int(edge["feature_datagrams"]),
                    "edge_duplicate_datagrams": int(edge["duplicate_datagrams"]),
                    "detection_count": int(edge["service_record_count"]),
                    "service_record_bytes": int(edge["service_record_bytes"]),
                    "service_target_met": bool(edge["service_target_met"]),
                    "processing_horizon_met": bool(edge["processing_horizon_met"]),
                    "edge_first_to_complete_reassembly_ms": (
                        reassembled - first_datagram
                    ) / 1e6,
                    "edge_queue_wait_ms": (worker_start - reassembled) / 1e6,
                    "edge_service_wall_ms": float(edge["edge_service_wall_ms"]),
                    "deployed_tail_service_ms": edge.get("deployed_tail_service_ms"),
                    "decode_tail_cuda_ms": (edge.get("tail_stage_cuda_ms") or {}).get(
                        "decode_tail_cuda"
                    ),
                }
            )
            if first:
                entry["application_feature_uplink_ms"] = (reassembled - first) / 1e6
            if final:
                entry["post_send_to_reassembly_ms"] = (reassembled - final) / 1e6
            for stage in EDGE_STAGES:
                value = (edge.get("edge_stage_ns") or {}).get(stage)
                entry[f"edge_{stage}_ms"] = None if value is None else float(value) / 1e6
            for stage in (*TAIL_STAGES, SERIALIZE_STAGE, TOTAL_STAGE):
                value = (edge.get("tail_stage_wall_ns") or {}).get(stage)
                entry[f"tail_{stage}_ms"] = None if value is None else float(value) / 1e6
                entry[f"tail_{stage}_cuda_ms"] = (
                    edge.get("tail_stage_cuda_ms") or {}
                ).get(stage)
            launch = entry.get("tail_decode_tail_launch_ms")
            settle = entry.get("tail_finite_check_outputs_ms")
            if launch is not None and settle is not None:
                entry["decode_tail_inference_block_ms"] = launch + settle
            postprocess = entry.get("tail_camera_aware_postprocess_ms")
            p025 = entry.get("tail_p025_service_filter_ms")
            if postprocess is not None and p025 is not None:
                entry["post_processing_ms"] = postprocess + p025
        for name in (
            "ue_send_loop_ms", "application_feature_uplink_ms",
            "post_send_to_reassembly_ms", "edge_first_to_complete_reassembly_ms",
            "edge_queue_wait_ms", "deployed_tail_service_ms",
        ):
            value = entry.get(name)
            require(
                value is None or float(value) >= 0.0,
                f"negative derived interval {name} on frame {frame_id}",
            )
        joined.append(entry)
    return joined, {
        "collector_rows": len(collector_rows),
        "transmitted_rows": len(sent_rows),
        "prepare_status_counts": drops,
        "delivered_rows": sum(1 for row in joined if row["delivered"]),
    }


def run_live_action(
    *,
    action_id: int,
    campaign: dict[str, Any],
    campaign_path: Path,
    ue: Any,
    registry: SplitActionRegistry,
    run_id: str,
) -> dict[str, Any]:
    """One action: fresh CARLA, fresh radio, bounded live capture, full teardown."""

    adapter = _adapter()
    lifecycle = _lifecycle(campaign)
    profile = registry.resolve(int(action_id))
    cell_id = f"live_a{action_id:02d}__{common.NETWORK_PROFILE_ID.lower()}"
    cell = {
        "cell_id": cell_id,
        "action_id": int(profile.action_id),
        "profile_id": profile.profile_id,
        "model_family": profile.family,
        "network_profile_id": common.NETWORK_PROFILE_ID,
    }
    # Bind the cell's identity from the catalog row keyed by action_id, never
    # from the order of a config list. The catalog spells the family in lower
    # case ("ae128", "noae") where the registry uses the canonical "AE128" /
    # "noAE", so the family is compared case-insensitively and the cell carries
    # the catalog's own spelling for anything downstream that checks it.
    row = adapter.action_row(campaign, int(action_id))
    require(
        str(row["profile_id"]) == profile.profile_id,
        f"action {action_id} catalog profile mismatch: "
        f"{row['profile_id']!r} != {profile.profile_id!r}",
    )
    require(
        str(row["model_family"]).casefold() == profile.family.casefold(),
        f"action {action_id} catalog family mismatch: "
        f"{row['model_family']!r} vs registry {profile.family!r}",
    )
    require(
        str(row["entropy_coder"]) == "zstd",
        f"action {action_id} entropy codec drift: {row['entropy_coder']!r}",
    )
    cell["model_family"] = str(row["model_family"])
    started_at = time.time()
    print(f"[{cell_id}] cold preflight", flush=True)
    report: dict[str, Any] = {
        "cell_id": cell_id,
        "action_id": int(action_id),
        "profile_id": profile.profile_id,
        "network_profile_id": common.NETWORK_PROFILE_ID,
        "started_at_unix_s": started_at,
        "cold_before": runner.verify_cold_host(campaign, label=f"{cell_id}/before"),
    }

    temporary = Path(tempfile.mkdtemp(prefix=f"splitfusion_timing_diagnostic_{cell_id}_"))
    attempt_dir = temporary / "attempt"
    attempt_dir.mkdir(parents=False, exist_ok=False)
    radio_namespace: Path | None = None
    radio_state: Path | None = None
    radio_base: Mapping[str, Any] | None = None
    attached: Mapping[str, Any] | None = None
    telemetry: Any = None
    server: Any = None
    pgid: int | None = None
    map_process: Any = None
    target_process: Any = None
    target_output = Path()
    target_stop = Path()
    edge_meta: dict[str, Any] = {}
    collector: Any = None
    gpu = GpuSampler()
    edge_records: list[dict[str, Any]] = []
    ue_clock_start = common.clock_anchor("ue_process_start")
    try:
        print(f"[{cell_id}] starting the qualified 100 MHz/273 PRB/4D5U radio", flush=True)
        radio_namespace, radio_state, attached, radio_base = runner.start_radio(
            campaign, cell_id=cell_id, service_log_dir=temporary
        )
        report["radio_attachment"] = {
            "status": attached.get("status"),
            "clean_noise_preflight": attached.get("clean_noise_preflight"),
        }
        telemetry = runner.RadioTelemetry(radio_base, temporary)
        telemetry.start()

        print(f"[{cell_id}] starting a fresh Epic CARLA lifecycle", flush=True)
        server, pgid = lifecycle.start_carla(CARLA_RPC_PORT, temporary / "carla_server.log")
        version = lifecycle.wait_for_rpc(CARLA_RPC_PORT, CARLA_RPC_TIMEOUT_S)
        require(version is not None, "fresh Epic CARLA did not become RPC-ready")
        report["carla"] = {"server_version": str(version), "rpc_port": CARLA_RPC_PORT}

        print(f"[{cell_id}] starting the deployment map/install path", flush=True)
        map_process = adapter.start_map_process(
            campaign, temporary_dir=temporary, action_id=str(action_id),
            carla_host="127.0.0.1", carla_port=CARLA_RPC_PORT,
            api_port=MAP_API_PORT, udp_port=SPATIAL_MAP_PORT,
            feedback_port=FEEDBACK_PORT,
        )

        warmup_stream = f"diagwarmup_{cell_id}"
        print(f"[{cell_id}] starting the instrumented edge and warming it up", flush=True)
        warmup_payloads = synthetic_warmup_payloads(
            ue=ue, profile=profile, stream_id=warmup_stream, count=WARMUP_ITERATIONS
        )
        edge_state, edge_meta = runner.start_edge_container(
            campaign=campaign, campaign_path=campaign_path, profile=profile,
            warmup_payloads=warmup_payloads, temporary_dir=temporary,
            run_id=run_id, cell_id=cell_id, warmup_stream_id=warmup_stream,
        )
        del warmup_payloads
        report["edge_ready"] = edge_meta["ready"]
        report["edge_mounts"] = runner.inspect_edge_mounts(edge_state)
        report["warmup"] = {
            "source": "SEEDED_SYNTHETIC_7CH_TENSOR",
            "dataset_frames_used": 0,
            "iterations": WARMUP_ITERATIONS,
            "stream_id": warmup_stream,
            "wall_ms": edge_meta["ready"].get("warmup_wall_ms"),
            "excluded_from_statistics": True,
        }
        evidence_dir = edge_state / adapter.EDGE_EVIDENCE_LEAF
        evidence_dir.mkdir(parents=False, exist_ok=True)

        campaign_copy = temporary / "campaign.yaml"
        target_start = temporary / "target_snr_start"
        campaign["_target_start_file"] = str(target_start)
        campaign_copy.write_text(
            yaml.safe_dump(campaign, sort_keys=False), encoding="utf-8"
        )
        target_process, target_output, target_stop = adapter.start_target_snr(
            campaign, campaign_path=campaign_copy,
            profile_id=common.NETWORK_PROFILE_ID, temporary_dir=temporary,
            start_file=target_start,
        )
        time.sleep(1.0)
        require(target_process.poll() is None, "target-SNR runtime exited during startup")

        gpu.start()
        original = _install_live_wrappers(adapter)
        print(
            f"[{cell_id}] live capture: {TRANSMITTED_BUDGET} transmitted frames "
            f"or {SAFETY_TIMEOUT_S:.0f} s",
            flush=True,
        )
        try:
            route_ok, route_detail, collector = adapter.run_route_b(
                campaign=campaign, cell=cell, row=row,
                binding={"dispatcher": "phase13_sfd1_v2"},
                attempt_dir=attempt_dir, carla_host="127.0.0.1",
                carla_port=CARLA_RPC_PORT, map_api_port=MAP_API_PORT,
                feedback_port=FEEDBACK_PORT, edge_evidence_dir=evidence_dir,
                maximum_loop_sim_s=SAFETY_TIMEOUT_S,
            )
        finally:
            _restore_live_wrappers(adapter, original)
        gpu.stop()
        report["gpu"] = gpu.summary()
        report["route_detail"] = {
            key: route_detail.get(key)
            for key in (
                "route_runner_returncode", "density_status", "route_completed",
                "route_abort_reason", "error", "route_summary_identity_ok",
                "route_summary_identity_mismatches",
            )
        }
        report["route_accepted_by_campaign_gate"] = bool(route_ok)
        require(collector is not None, "the route never entered its drive loop")
        diagnostic = collector.diagnostic_summary()
        report["capture"] = {
            key: value for key, value in diagnostic.items() if key != "send_boundaries"
        }
        # A bounded cell deliberately does not complete the Route-B loop, so the
        # campaign's own route/coverage gate cannot pass and is not used here.
        require(
            str(diagnostic["stop_reason"])
            in {"TRANSMITTED_BUDGET_REACHED", "SAFETY_TIMEOUT_EXPIRED"},
            f"live cell ended for an unregistered reason: {diagnostic['stop_reason']!r}; "
            f"route_error={report['route_detail'].get('error')!r}",
        )
        require(not diagnostic["failures"], f"collector failures: {diagnostic['failures']}")

        require(runner._edge_running(), "the instrumented edge exited during capture")
        report["edge_shutdown"] = runner.request_edge_shutdown(
            Path(edge_meta["stop_host"]), Path(edge_meta["summary_host"])
        )
        require(
            bool(report["edge_shutdown"]["graceful_shutdown_observed"]),
            "the edge did not publish its final summary before shutdown",
        )
        require(runner._stop_edge_container(), "the edge container did not stop")
        edge_summary = common.load_json(Path(edge_meta["summary_host"]))
        require(bool(edge_summary.get("final")), "the edge summary is not the final one")
        require(
            not edge_summary.get("failures"),
            f"the edge reported failures: {edge_summary.get('failures')}",
        )
        records_path = Path(edge_meta["records_host"])
        if records_path.is_file():
            for line in records_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    edge_records.append(json.loads(line))
        require(
            all(row.get("phase") == "MEASURED" for row in edge_records),
            "a warm-up record leaked into the measured edge records",
        )
        require(
            all(int(row["action_id"]) == int(action_id) for row in edge_records),
            "an edge record carries a foreign action",
        )
        report["edge_summary"] = edge_summary
        skew = max(
            abs(
                int(anchor["wall_minus_monotonic_ns"])
                - int(ue_clock_start["wall_minus_monotonic_ns"])
            )
            for anchor in (
                edge_summary["start_clock_anchor"], edge_summary["clock_anchor"]
            )
        )
        require(
            skew < 5_000_000,
            f"UE host and edge container do not share one wall clock: {skew} ns skew",
        )
        report["clock_domain"] = {
            "ue_anchors": [ue_clock_start, common.clock_anchor("ue_process_end")],
            "edge_anchors": [
                edge_summary["start_clock_anchor"], edge_summary["clock_anchor"]
            ],
            "shared_wall_clock_domain_verified": True,
            "maximum_wall_minus_monotonic_skew_ns": int(skew),
        }
        rows, accounting = join_live_records(
            collector_rows=collector.rows,
            send_boundaries=diagnostic["send_boundaries"],
            edge_records=edge_records,
        )
        require(
            accounting["transmitted_rows"] == int(diagnostic["transmitted_frames"]),
            "transmitted-frame accounting mismatch: "
            f"{accounting['transmitted_rows']} rows vs "
            f"{diagnostic['transmitted_frames']} counted sends",
        )
        report["per_frame_rows"] = rows
        report["accounting"] = accounting
    finally:
        gpu.stop()
        if target_process is not None:
            try:
                report["radio_actuation_restored"] = adapter.stop_target_snr(
                    target_process, target_output, target_stop,
                    temporary / "radio_trace_final.csv",
                )
            except Exception as exc:
                report["radio_actuation_error"] = f"{type(exc).__name__}: {exc}"
        if telemetry is not None:
            telemetry.stop()
            report["radio_telemetry"] = telemetry.summary()
        if runner._edge_running():
            report["edge_forced_stop"] = runner._stop_edge_container()
        report["map_process_stopped"] = adapter.stop_process(map_process)
        if server is not None and pgid is not None:
            report["carla_teardown"] = lifecycle.stop_carla(server, pgid, CARLA_RPC_PORT)
        if radio_base is not None and radio_namespace is not None and radio_state is not None:
            report["radio_teardown"] = runner.stop_radio(
                radio_base, radio_namespace, radio_state, attached,
                actuator_restore_verified=bool(report.get("radio_actuation_restored")),
            )
        shutil.rmtree(temporary, ignore_errors=True)
        report["cell_scratch_removed"] = not temporary.exists()

    require(
        bool(report.get("cell_scratch_removed")),
        "the cell scratch directory survived cleanup",
    )
    require(
        bool(report.get("carla_teardown", {}).get("shutdown_verified")),
        "fresh CARLA process group or RPC port survived teardown",
    )
    report["cold_after"] = runner.verify_cold_host(campaign, label=f"{cell_id}/after")
    report["finished_at_unix_s"] = time.time()
    report["wall_seconds"] = report["finished_at_unix_s"] - started_at
    return report


# --------------------------------------------------------------------------
# statistics and evidence
# --------------------------------------------------------------------------

PREPARATION_METRICS: tuple[str, ...] = (
    "prep_queue_wait_ms", "prep_sensor_wait_ms", "prep_radar_window_ms",
    "prep_radar_prepare_ms", "prep_rgb_convert_ms", "prep_scene_snapshot_ms",
    "prep_pre_front_compute_ms", "ue_front_ms",
)
UPLINK_METRICS: tuple[str, ...] = (
    "ue_send_loop_ms", "application_feature_uplink_ms",
    "post_send_to_reassembly_ms", "edge_first_to_complete_reassembly_ms",
)
EDGE_METRICS: tuple[str, ...] = (
    "edge_queue_wait_ms",
    *(f"edge_{stage}_ms" for stage in EDGE_STAGES),
    "edge_service_wall_ms", "deployed_tail_service_ms",
)
TAIL_METRICS: tuple[str, ...] = (
    "decode_tail_cuda_ms", "decode_tail_inference_block_ms", "post_processing_ms",
    *(f"tail_{stage}_ms" for stage in (*TAIL_STAGES, SERIALIZE_STAGE, TOTAL_STAGE)),
    *(f"tail_{stage}_cuda_ms" for stage in (*TAIL_STAGES, SERIALIZE_STAGE, TOTAL_STAGE)),
)
SCENE_METRICS: tuple[str, ...] = (
    "detection_count", "ego_speed_mps", "window_returns",
    "payload_bytes", "payload_bytes_uncompressed", "payload_chunks",
)
LIVE_METRICS: tuple[str, ...] = (
    *PREPARATION_METRICS, *UPLINK_METRICS, *EDGE_METRICS, *TAIL_METRICS, *SCENE_METRICS,
)

PER_FRAME_FIELDS: tuple[str, ...] = (
    "cell_id", "action_id", "profile_id", "network_profile_id",
    "frame_id", "capture_id", "route_tick", "carla_timestamp", "capture_wall_s",
    "delivered", "service_target_met", "processing_horizon_met",
    "window_callbacks", "service_record_bytes",
    "ue_first_send_wall_ns", "ue_final_send_wall_ns", "ue_datagrams_observed",
    "edge_first_datagram_wall_ns", "edge_complete_reassembly_wall_ns",
    "edge_admitted_wall_ns", "edge_worker_start_wall_ns",
    "edge_tail_finished_wall_ns", "edge_result_published_wall_ns",
    "edge_feature_datagrams", "edge_duplicate_datagrams",
    *LIVE_METRICS,
)


def _live_group_totals(row: Mapping[str, Any]) -> dict[str, float | None]:
    totals: dict[str, float | None] = {}
    for name, stages in runner.STAGE_GROUPS:
        values = [row.get(f"tail_{stage}_ms") for stage in stages]
        totals[name] = (
            sum(float(value) for value in values)
            if all(value is not None for value in values)
            else None
        )
    return totals


def summarize_live_action(report: Mapping[str, Any]) -> dict[str, Any]:
    rows = report["per_frame_rows"]
    delivered = [row for row in rows if row.get("delivered")]
    capture = report["capture"]
    edge_summary = report.get("edge_summary", {})
    counters = dict(edge_summary.get("counters", {}))
    metrics = {
        name: common.summarize([row[name] for row in rows if row.get(name) is not None])
        for name in LIVE_METRICS
    }
    groups = {
        name: common.summarize(
            [
                value
                for row in delivered
                if (value := _live_group_totals(row)[name]) is not None
            ]
        )
        for name, _stages in runner.STAGE_GROUPS
    }
    group_total = sum(
        float(value["median"]) for value in groups.values() if value["median"] is not None
    )
    return {
        "cell_id": report["cell_id"],
        "action_id": int(report["action_id"]),
        "profile_id": report["profile_id"],
        "network_profile_id": report["network_profile_id"],
        "wall_seconds": report["wall_seconds"],
        "counts": {
            "transmitted_frames": int(capture["transmitted_frames"]),
            "transmitted_budget": int(capture["transmitted_budget"]),
            "reached_budget": bool(capture["reached_budget"]),
            "stop_reason": str(capture["stop_reason"]),
            "route_wall_seconds": capture.get("route_wall_seconds"),
            "route_ticks_observed": int(capture["route_ticks_observed"]),
            "preparation_opportunities_dropped": int(
                capture["preparation_opportunities_dropped"]
            ),
            "prepare_status_counts": report["accounting"]["prepare_status_counts"],
            "delivered_and_measured": len(delivered),
            "feature_datagrams_received_edge": int(
                counters.get("feature_datagrams_received", 0)
            ),
            "complete_reassemblies": int(counters.get("feature_messages_reassembled", 0)),
            "incomplete_reassemblies_expired": int(
                edge_summary.get("incomplete_reassemblies_expired", 0)
            ),
            "queue_admissions": int(counters.get("edge_queue_admissions", 0)),
            "queue_replacements": int(counters.get("edge_pending_replacements", 0)),
            "tail_completions": int(counters.get("tail_completions", 0)),
            "compact_results_transmitted": int(
                counters.get("compact_results_transmitted", 0)
            ),
            "service_target_met": sum(
                1 for row in delivered if row.get("service_target_met")
            ),
            "processing_horizon_met": sum(
                1 for row in delivered if row.get("processing_horizon_met")
            ),
        },
        "timing": metrics,
        "live_stage_groups_ms": groups,
        "live_stage_group_total_median_ms": group_total,
        "gpu": report.get("gpu", {}),
        "radio_telemetry": report.get("radio_telemetry", {}),
        "radio_actuation_restored": bool(report.get("radio_actuation_restored")),
        "clock_domain": report.get("clock_domain", {}),
        "warmup": report.get("warmup", {}),
        "evaluation_scaffolding": capture.get("evaluation_scaffolding", {}),
        "route_detail": report.get("route_detail", {}),
    }


def build_live_comparisons(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def median(summary: Mapping[str, Any], name: str) -> float | None:
        value = summary["timing"].get(name, {})
        return None if value.get("median") is None else float(value["median"])

    per_action = []
    for summary in summaries:
        cuda = median(summary, "decode_tail_cuda_ms")
        deployed = median(summary, "deployed_tail_service_ms")
        total = float(summary["live_stage_group_total_median_ms"])
        group_medians = {
            name: (
                None
                if summary["live_stage_groups_ms"][name]["median"] is None
                else float(summary["live_stage_groups_ms"][name]["median"])
            )
            for name, _stages in runner.STAGE_GROUPS
        }
        inference = group_medians["tail_inference_block"]
        per_action.append(
            {
                "action_id": summary["action_id"],
                "profile_id": summary["profile_id"],
                "payload_median_bytes": median(summary, "payload_bytes"),
                "datagrams_per_message_median": median(summary, "payload_chunks"),
                "detection_count": summary["timing"]["detection_count"],
                "decode_tail_cuda_ms": summary["timing"]["decode_tail_cuda_ms"],
                "deployed_tail_service_ms": summary["timing"]["deployed_tail_service_ms"],
                "post_processing_ms": summary["timing"]["post_processing_ms"],
                "live_frozen_tail_ms": summary["timing"]["edge_frozen_tail_ms"],
                "live_stage_group_medians_ms": group_medians,
                "live_tail_span_ms_median": total,
                "non_inference_overhead_ms_median": (
                    None if inference is None else total - inference
                ),
                "non_inference_overhead_fraction": (
                    None if inference is None or total <= 0 else (total - inference) / total
                ),
                "delta_vs_phase13c_controlled_tail_ms": (
                    None if cuda is None else cuda - common.PHASE13C_CONTROLLED_TAIL_MS
                ),
                "delta_vs_phase15_live_frozen_tail_ms": (
                    None if cuda is None else cuda - common.PHASE15_LIVE_FROZEN_TAIL_MS
                ),
                "deployed_over_cuda_ratio": (
                    None if not cuda or deployed is None else deployed / cuda
                ),
                "application_feature_uplink_ms": summary["timing"][
                    "application_feature_uplink_ms"
                ],
                "ue_send_loop_ms": summary["timing"]["ue_send_loop_ms"],
                "edge_queue_wait_ms": summary["timing"]["edge_queue_wait_ms"],
                "transmitted_frames": summary["counts"]["transmitted_frames"],
                "complete_reassemblies": summary["counts"]["complete_reassemblies"],
                "gpu_utilization_percent": summary["gpu"].get("utilization_gpu_percent", {}),
                "concurrent_compute_processes": summary["gpu"].get(
                    "concurrent_compute_processes", {}
                ),
            }
        )

    queue_pairs = [
        (
            row["edge_queue_wait_ms"]["median"],
            row["deployed_tail_service_ms"]["median"],
        )
        for row in per_action
        if row["edge_queue_wait_ms"]["median"] is not None
        and row["deployed_tail_service_ms"]["median"] is not None
    ]
    ratios = [wait / service for wait, service in queue_pairs if service]
    uplink = sorted(
        (
            {
                "action_id": row["action_id"],
                "profile_id": row["profile_id"],
                "payload_median_bytes": row["payload_median_bytes"],
                "datagrams_per_message_median": row["datagrams_per_message_median"],
                "transmitted_frames": row["transmitted_frames"],
                "complete_reassemblies": row["complete_reassemblies"],
                "complete_reassembly_fraction": (
                    None
                    if not row["transmitted_frames"]
                    else row["complete_reassemblies"] / row["transmitted_frames"]
                ),
                "application_feature_uplink_ms_median": row[
                    "application_feature_uplink_ms"
                ]["median"],
                "application_feature_uplink_ms_p95": row[
                    "application_feature_uplink_ms"
                ]["p95"],
                "ue_send_loop_ms_median": row["ue_send_loop_ms"]["median"],
                "detection_count_median": row["detection_count"]["median"],
            }
            for row in per_action
        ),
        key=lambda item: -(item["payload_median_bytes"] or 0),
    )
    return {
        "reference_points": {
            "phase13c_controlled_tail_gpu_ms_median": common.PHASE13C_CONTROLLED_TAIL_MS,
            "phase13c_scope": (
                "CUDA-event span over the whole frozen tail adapter call, 36 "
                "profiles x 300 fit frames, localhost, no OAI, no CARLA"
            ),
            "phase15_live_frozen_tail_ms_median": common.PHASE15_LIVE_FROZEN_TAIL_MS,
            "phase15_scope": (
                "monotonic wall span of the edge runtime's frozen_tail stage, "
                "live CARLA/OAI Route-B cells"
            ),
        },
        "per_action": per_action,
        "uplink_versus_payload": uplink,
        "queue_wait_versus_deployed_service": {
            "paired_actions": len(queue_pairs),
            "queue_wait_ms_medians": [pair[0] for pair in queue_pairs],
            "deployed_tail_service_ms_medians": [pair[1] for pair in queue_pairs],
            "queue_wait_over_service_ratio": ratios,
            "queue_wait_tracks_service_time": (
                bool(ratios and min(ratios) >= 0.5 and max(ratios) <= 2.0)
                if len(queue_pairs) >= 2
                else None
            ),
        },
        "scene_confound_warning": (
            "Detection counts differ per live scene, and post-processing and p025 "
            "filtering scale with detection count, so differences between actions "
            "must not be read as payload effects alone. Detection counts are "
            "reported beside post-processing latency for every action."
        ),
    }


def write_live_per_frame_csv(path: Path, report: Mapping[str, Any]) -> str:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PER_FRAME_FIELDS))
        writer.writeheader()
        for row in report["per_frame_rows"]:
            enriched = {
                **row,
                "cell_id": report["cell_id"],
                "action_id": int(report["action_id"]),
                "profile_id": report["profile_id"],
                "network_profile_id": report["network_profile_id"],
            }
            writer.writerow(
                {field: enriched.get(field, "") for field in PER_FRAME_FIELDS}
            )
    return common.sha256_file(path)


def write_live_summary_csv(path: Path, summaries: Sequence[Mapping[str, Any]]) -> str:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        row: dict[str, Any] = {
            "action_id": summary["action_id"],
            "profile_id": summary["profile_id"],
            "cell_id": summary["cell_id"],
            "network_profile_id": summary["network_profile_id"],
            "wall_seconds": round(float(summary["wall_seconds"]), 3),
        }
        for key, value in summary["counts"].items():
            if isinstance(value, Mapping):
                for status, count in sorted(value.items()):
                    row[f"count_{key}_{status}"] = count
            else:
                row[f"count_{key}"] = "" if value is None else value
        for name, value in summary["timing"].items():
            row.update(runner._flatten(name, value))
        for name, value in summary["live_stage_groups_ms"].items():
            row.update(runner._flatten(f"live_group_{name}", value))
        gpu = summary.get("gpu", {})
        row["gpu_sample_count"] = gpu.get("sample_count", "")
        row["gpu_throttle_reasons"] = "|".join(gpu.get("throttle_reasons_observed", []))
        for name in (
            "utilization_gpu_percent", "utilization_memory_percent",
            "memory_used_mib", "temperature_c", "concurrent_compute_processes",
        ):
            if isinstance(gpu.get(name), Mapping):
                row.update(runner._flatten(f"gpu_{name}", gpu[name]))
        telemetry = summary.get("radio_telemetry", {})
        row["radio_telemetry_status"] = telemetry.get("status", "")
        for name in (
            "achieved_pusch_snr_db", "achieved_pusch_mcs",
            "scheduler_selected_ul_mcs", "scheduler_final_ul_mcs",
        ):
            if isinstance(telemetry.get(name), Mapping):
                row.update(runner._flatten(f"radio_{name}", telemetry[name]))
        row["radio_actuation_restored"] = summary["radio_actuation_restored"]
        rows.append(row)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return common.sha256_file(path)


def write_live_figure(
    base_path: Path,
    summaries: Sequence[Mapping[str, Any]],
    comparisons: Mapping[str, Any],
) -> dict[str, str]:
    """Live tail decomposition and live uplink-versus-payload, print artifact."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    ordered = sorted(
        summaries,
        key=lambda item: -(item["timing"]["payload_bytes"]["median"] or 0),
    )
    labels = [
        f"a{item['action_id']:02d}  {item['profile_id'].replace('split_', '')}"
        for item in ordered
    ]
    positions = list(range(len(ordered)))[::-1]
    figure, (upper, lower) = plt.subplots(
        2, 1, figsize=(12.0, 9.2), gridspec_kw={"height_ratios": [1.2, 1.0]}
    )
    figure.patch.set_facecolor(runner.SURFACE)
    for axis in (upper, lower):
        axis.set_facecolor(runner.SURFACE)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(runner.GRID)
        axis.tick_params(colors=runner.TEXT_SECONDARY, labelsize=9, length=3, width=0.8)
        axis.xaxis.grid(True, color=runner.GRID, linewidth=0.8)
        axis.set_axisbelow(True)

    totals = [float(item["live_stage_group_total_median_ms"]) for item in ordered]
    span = max([*totals, 1.0])
    gap = span * 0.004
    for index, (name, _stages) in enumerate(runner.STAGE_GROUPS):
        left, widths = [], []
        for item in ordered:
            offset = sum(
                float(item["live_stage_groups_ms"][earlier]["median"] or 0.0) + gap
                for earlier, _s in runner.STAGE_GROUPS[:index]
            )
            left.append(offset)
            widths.append(float(item["live_stage_groups_ms"][name]["median"] or 0.0))
        bars = upper.barh(
            positions, widths, left=left, height=0.52,
            color=runner.GROUP_COLORS[index], edgecolor=runner.SURFACE, linewidth=0.0,
        )
        for bar, value in zip(bars, widths):
            if value >= span * 0.055:
                upper.text(
                    bar.get_x() + value / 2.0,
                    bar.get_y() + bar.get_height() / 2.0,
                    f"{value:.1f}", ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if index in (0, 5) else runner.TEXT_PRIMARY,
                )
    for position, item, total in zip(positions, ordered, totals):
        cuda = item["timing"]["decode_tail_cuda_ms"]["median"]
        deployed = item["timing"]["deployed_tail_service_ms"]["median"]
        detections = item["timing"]["detection_count"]["median"]
        annotation = f"decode_tail CUDA {float(cuda):.1f}" if cuda is not None else "CUDA n/a"
        if deployed is not None:
            annotation += f"  ·  deployed service {float(deployed):.1f}"
        if detections is not None:
            annotation += f"  ·  {int(detections)} det"
        upper.text(
            total + span * 0.025, position, annotation,
            ha="left", va="center", fontsize=8, color=runner.TEXT_SECONDARY,
        )
    for reference, text in (
        (common.PHASE13C_CONTROLLED_TAIL_MS, "Phase-13C tail_gpu 73.4"),
        (common.PHASE15_LIVE_FROZEN_TAIL_MS, "Phase-15 live frozen_tail 111.6"),
    ):
        upper.axvline(reference, color=runner.TEXT_MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
        upper.text(
            reference, len(ordered) - 0.42, f"  {text}",
            fontsize=8, color=runner.TEXT_MUTED, ha="left", va="bottom",
        )
    upper.set_yticks(positions)
    upper.set_yticklabels(labels, fontsize=9, color=runner.TEXT_PRIMARY)
    upper.set_xlim(0, span * 1.62)
    upper.set_ylim(-0.55, len(ordered) - 0.2)
    upper.set_xlabel(
        "milliseconds (median over live delivered frames)",
        fontsize=9, color=runner.TEXT_SECONDARY,
    )
    upper.set_title(
        "Live CARLA/OAI frozen p025 tail service span, decomposed",
        fontsize=11, color=runner.TEXT_PRIMARY, loc="left", pad=10,
    )
    upper.legend(
        handles=[
            Patch(facecolor=runner.GROUP_COLORS[index], edgecolor=runner.SURFACE,
                  label=runner.GROUP_LABELS[index])
            for index in range(len(runner.STAGE_GROUPS))
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.20), frameon=False,
        fontsize=8.5, ncol=3, labelcolor=runner.TEXT_SECONDARY,
        handlelength=1.4, columnspacing=1.6,
    )

    uplink = comparisons["uplink_versus_payload"]
    medians = [row["application_feature_uplink_ms_median"] for row in uplink]
    p95s = [row["application_feature_uplink_ms_p95"] for row in uplink]
    scale = max(
        [*[value for value in medians if value is not None],
         *[value for value in p95s if value is not None], 1.0]
    )
    positions_b = list(range(len(uplink)))[::-1]
    labels_b = [
        f"a{row['action_id']:02d}  {(row['payload_median_bytes'] or 0) / 1024:,.0f} KiB"
        f"  ·  {int(row['datagrams_per_message_median'] or 0)} dgram"
        for row in uplink
    ]
    lower.barh(
        positions_b,
        [value if value is not None else 0.0 for value in medians],
        height=0.46, color=runner.GROUP_COLORS[0],
        edgecolor=runner.SURFACE, linewidth=0.0,
    )
    for position, row, value, p95 in zip(positions_b, uplink, medians, p95s):
        if value is None:
            lower.text(
                scale * 0.01, position,
                "no complete reassembly arrived  ·  uplink-limited",
                ha="left", va="center", fontsize=9, color="#e34948",
            )
            continue
        lower.text(
            value + scale * 0.014, position,
            f"{value:,.0f} ms  ·  {row['complete_reassemblies']}/"
            f"{row['transmitted_frames']} complete",
            ha="left", va="center", fontsize=9, color=runner.TEXT_PRIMARY,
        )
        if p95 is not None:
            lower.plot(
                [p95], [position], marker="o", markersize=6.5,
                markerfacecolor=runner.SURFACE, markeredgecolor=runner.GROUP_COLORS[0],
                markeredgewidth=2.0, linestyle="none",
            )
            lower.text(
                p95 + scale * 0.014, position - 0.29, f"p95 {p95:,.0f}",
                ha="left", va="center", fontsize=8, color=runner.TEXT_SECONDARY,
            )
    lower.set_yticks(positions_b)
    lower.set_yticklabels(labels_b, fontsize=9, color=runner.TEXT_PRIMARY)
    lower.set_xlim(0, scale * 1.42)
    lower.set_ylim(-0.6, len(uplink) - 0.3)
    lower.set_xlabel(
        "milliseconds, UE first datagram send -> edge complete message reassembly",
        fontsize=9, color=runner.TEXT_SECONDARY,
    )
    lower.set_title(
        "Application-level feature-uplink handling through live OAI, by payload"
        "  ·  hollow dot = p95",
        fontsize=11, color=runner.TEXT_PRIMARY, loc="left", pad=10,
    )
    figure.tight_layout(pad=1.4, h_pad=4.2)
    digests: dict[str, str] = {}
    for suffix in ("pdf", "png"):
        path = base_path.with_suffix(f".{suffix}")
        options: dict[str, Any] = {
            "format": suffix, "facecolor": runner.SURFACE, "bbox_inches": "tight",
        }
        if suffix == "png":
            options["dpi"] = 200
        figure.savefig(path, **options)
        digests[path.name] = common.sha256_file(path)
    plt.close(figure)
    return digests


LIVE_TABLE = (
    ("ue_send_loop_ms", "UE send loop (first->final datagram)"),
    ("application_feature_uplink_ms", "application feature uplink (first send->edge reassembled)"),
    ("post_send_to_reassembly_ms", "post-send to reassembly (final send->edge reassembled)"),
    ("edge_first_to_complete_reassembly_ms", "edge first datagram->complete reassembly"),
    ("edge_queue_wait_ms", "edge queue wait (reassembled->worker start)"),
    ("edge_zstd_decompression_ms", "zstd decompression"),
    ("edge_unpack_dequantize_ms", "unpack / dequantization"),
    ("edge_ae_decode_ms", "AE decode"),
    ("tail_camera_pose_reconstruct_ms", "camera-pose / calibration reconstruction"),
    ("decode_tail_cuda_ms", "decode_tail CUDA (device, pure)"),
    ("decode_tail_inference_block_ms", "decode_tail inference block (launch + completion)"),
    ("tail_camera_aware_postprocess_ms", "camera-aware post-processing"),
    ("tail_p025_service_filter_ms", "p025 service filtering"),
    ("post_processing_ms", "post-processing + p025 combined"),
    ("tail_segmentation_upsample_argmax_ms", "720x1280 segmentation interpolate + argmax"),
    ("tail_output_serialization_ms", "compact object-result serialization"),
    ("edge_frozen_tail_ms", "frozen_tail stage (deployed definition)"),
    ("edge_total_edge_processing_ms", "total edge processing"),
    ("deployed_tail_service_ms", "deployed tail service (worker start->result published)"),
    ("detection_count", "detections per frame"),
)
PREPARATION_TABLE = (
    ("prep_queue_wait_ms", "preparation queue wait"),
    ("prep_sensor_wait_ms", "CARLA sensor wait"),
    ("prep_radar_window_ms", "radar logical-sweep window"),
    ("prep_radar_prepare_ms", "radar rasterisation"),
    ("prep_rgb_convert_ms", "RGB conversion"),
    ("prep_scene_snapshot_ms", "scene snapshot"),
    ("prep_pre_front_compute_ms", "total pre-front preparation"),
    ("ue_front_ms", "UE front/ranker/AE/quantize/zstd"),
)


def write_live_report(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    summaries: Sequence[Mapping[str, Any]],
    comparisons: Mapping[str, Any],
    runtime_seconds: float,
) -> str:
    ms = runner._ms
    lines: list[str] = []
    lines.append("# SplitFusion live CARLA/OAI timing diagnostic — measured decomposition")
    lines.append("")
    lines.append(
        f"Run `{manifest['run_id']}` · implementation commit "
        f"`{manifest['git']['head']}` · {runtime_seconds / 60.0:.1f} min wall · "
        f"{manifest['environment']['device_name']}."
    )
    lines.append("")
    lines.append(
        "Four actions, four fresh CARLA lifecycles and four fresh "
        "CN5G/gNB/UE/edge lifecycles. Each cell drives the qualified live "
        "Route-B configuration — Town10HD_Opt "
        f"`{manifest['route_b']['route_json']}`, 50 vehicles / 50 pedestrians, "
        f"scenario seed {manifest['route_b']['scenario_seed']}, traffic-manager "
        f"seed {manifest['route_b']['traffic_manager_seed']}, Epic quality — with "
        "one action fixed throughout, FAVORABLE_STABLE restarted from its "
        f"identical first sample, a {TRANSMITTED_BUDGET}-transmitted-frame budget "
        f"and a {SAFETY_TIMEOUT_S:.0f} s safety timeout. The full Route-B loop is "
        "deliberately not completed."
    )
    lines.append("")
    lines.append("## What each number is, and is not")
    lines.append("")
    lines.append(
        "- **No stored dataset frame is replayed.** Every measured payload is "
        "built from live CARLA RGB + radar. The only synthetic input is the "
        "edge's CUDA warm-up (a seeded random 7-channel tensor on its own "
        "stream), which the edge marks WARMUP and which is excluded from every "
        "statistic."
    )
    lines.append(
        "- **`application_feature_uplink_ms` = edge complete reassembly − UE "
        "first datagram send.** Application-level feature-uplink handling "
        "through OAI, **not** PHY/RLC latency: it includes the host UDP send "
        "path, the UE tunnel, the core and RAN transport, edge socket receipt, "
        "fragmentation and message reassembly. Sensor preparation and CARLA "
        "waiting are structurally excluded — the clock starts at the wall time "
        "taken immediately before the first datagram reaches the socket."
    )
    lines.append(
        "- **Two tail measurements, reported separately.** "
        "`decode_tail_cuda_ms` is a dedicated CUDA event pair with nothing but "
        "`model.decode_tail(batch, dense=False)` between the two records. "
        "`deployed_tail_service_ms` is the wall span from edge worker start to "
        "compact-result publication, so it carries live contention, "
        "post-processing, p025 filtering, segmentation construction, "
        "serialization and publication."
    )
    lines.append(
        "- The earlier round-trip residual is **not** reused or relabelled. "
        "Every uplink quantity is a one-way difference between two same-host "
        "`time.time_ns()` boundaries."
    )
    lines.append("")
    clock = summaries[0].get("clock_domain", {})
    lines.append(
        "Clock domain: UE host and edge container verified to share one "
        "wall-clock domain from paired `time.time_ns()` / `time.monotonic_ns()` "
        "anchors at both ends of both processes (worst wall−monotonic offset "
        f"skew {clock.get('maximum_wall_minus_monotonic_skew_ns', 'n/a')} ns). No "
        "derived interval in any artifact is negative."
    )
    lines.append("")
    lines.append(
        "Collector configuration: the qualified collector runs with its map "
        "install and install-feedback deployment functions **enabled** and its "
        "segmentation-quality, object-ground-truth and exact-installed-record "
        "**evaluation scaffolding disabled**, so the deployed service span "
        "carries deployment contention without offline scoring work."
    )
    lines.append("")

    lines.append("## Commissioned comparisons")
    lines.append("")
    reference = comparisons["reference_points"]
    lines.append(
        "**1 & 2 — pure `decode_tail` CUDA time versus the two published spans.** "
        f"The ~{reference['phase13c_controlled_tail_gpu_ms_median']} ms Phase-13C "
        f"`tail_gpu_ms` and the ~{reference['phase15_live_frozen_tail_ms_median']} ms "
        "Phase-15 live `frozen_tail` are both spans over the *whole* tail adapter "
        "call, not over `decode_tail`."
    )
    lines.append("")
    lines.append(
        "| action | profile | decode_tail CUDA (ms) | deployed service (ms) "
        "| live frozen_tail (ms) | Δ CUDA vs 73.4 | Δ CUDA vs 111.6 | deployed / CUDA |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in comparisons["per_action"]:
        ratio = row["deployed_over_cuda_ratio"]
        lines.append(
            f"| {row['action_id']} | `{row['profile_id']}` | "
            f"{ms(row['decode_tail_cuda_ms']['median'])} | "
            f"{ms(row['deployed_tail_service_ms']['median'])} | "
            f"{ms(row['live_frozen_tail_ms']['median'])} | "
            f"{ms(row['delta_vs_phase13c_controlled_tail_ms'])} | "
            f"{ms(row['delta_vs_phase15_live_frozen_tail_ms'])} | "
            + ("n/a" if ratio is None else f"{ratio:.1f}x")
            + " |"
        )
    lines.append("")
    lines.append(
        "**3 — where the difference goes.** Wall-clock stage groups partition "
        "the live tail service span (medians, ms), with detection count beside "
        "post-processing so the two are never conflated:"
    )
    lines.append("")
    lines.append(
        "| action | " + " | ".join(name for name, _s in runner.STAGE_GROUPS)
        + " | span | non-inference share | detections |"
    )
    lines.append("|---" * (len(runner.STAGE_GROUPS) + 4) + "|")
    for row in comparisons["per_action"]:
        cells = " | ".join(
            ms(row["live_stage_group_medians_ms"][name]) for name, _s in runner.STAGE_GROUPS
        )
        share = row["non_inference_overhead_fraction"]
        detections = row["detection_count"]["median"]
        lines.append(
            f"| {row['action_id']} | {cells} | {ms(row['live_tail_span_ms_median'])} | "
            + ("n/a" if share is None else f"{share * 100:.1f}%")
            + " | "
            + ("n/a" if detections is None else f"{int(detections)}")
            + " |"
        )
    lines.append("")
    lines.append("**4 — application uplink latency versus payload** (descending payload):")
    lines.append("")
    lines.append(
        "| action | payload (B) | datagrams/msg | transmitted | complete "
        "reassemblies | complete fraction | uplink median (ms) | uplink p95 (ms) "
        "| UE send loop median (ms) | detections |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in comparisons["uplink_versus_payload"]:
        fraction = row["complete_reassembly_fraction"]
        lines.append(
            f"| {row['action_id']} | {int(row['payload_median_bytes'] or 0):,} | "
            f"{int(row['datagrams_per_message_median'] or 0)} | "
            f"{row['transmitted_frames']} | {row['complete_reassemblies']} | "
            + ("n/a" if fraction is None else f"{fraction * 100:.1f}%")
            + f" | {ms(row['application_feature_uplink_ms_median'], 1)} | "
            f"{ms(row['application_feature_uplink_ms_p95'], 1)} | "
            f"{ms(row['ue_send_loop_ms_median'])} | "
            + ("n/a" if row["detection_count_median"] is None
               else f"{int(row['detection_count_median'])}")
            + " |"
        )
    lines.append("")
    queue = comparisons["queue_wait_versus_deployed_service"]
    lines.append(
        "**5 — does edge queue wait track the deployed tail service time?** "
        f"Paired actions: {queue['paired_actions']}. Queue-wait medians "
        f"{[None if v is None else round(float(v), 2) for v in queue['queue_wait_ms_medians']]} ms "
        "against deployed-service medians "
        f"{[None if v is None else round(float(v), 2) for v in queue['deployed_tail_service_ms_medians']]} ms. "
        f"Tracks service time: {queue['queue_wait_tracks_service_time']}."
    )
    lines.append("")
    lines.append(f"> {comparisons['scene_confound_warning']}")
    lines.append("")

    for summary in summaries:
        counts = summary["counts"]
        lines.append(f"## Action {summary['action_id']} — `{summary['profile_id']}`")
        lines.append("")
        lines.append(
            f"Transmitted {counts['transmitted_frames']}/"
            f"{counts['transmitted_budget']} frames "
            f"(reached budget: {counts['reached_budget']}, stop reason "
            f"`{counts['stop_reason']}`) over "
            + (
                f"{float(counts['route_wall_seconds']):.1f} s"
                if counts["route_wall_seconds"] is not None else "n/a"
            )
            + f" of route across {counts['route_ticks_observed']} ticks. "
            f"Preparation opportunities dropped: "
            f"{counts['preparation_opportunities_dropped']} "
            f"({counts['prepare_status_counts']}). Edge: "
            f"{counts['feature_datagrams_received_edge']:,} datagrams, "
            f"{counts['complete_reassemblies']} complete reassemblies, "
            f"{counts['incomplete_reassemblies_expired']} incomplete expiries, "
            f"{counts['queue_admissions']} admissions "
            f"({counts['queue_replacements']} replacements), "
            f"{counts['tail_completions']} tail completions, "
            f"{counts['compact_results_transmitted']} compact results. "
            f"On-time against the 100 ms service target: "
            f"{counts['service_target_met']}/{counts['delivered_and_measured']}; "
            f"within the 500 ms horizon: {counts['processing_horizon_met']}/"
            f"{counts['delivered_and_measured']}."
        )
        lines.append("")
        gpu = summary.get("gpu", {})
        telemetry = summary.get("radio_telemetry", {})
        snr = telemetry.get("achieved_pusch_snr_db", {})
        mcs = telemetry.get("scheduler_final_ul_mcs", {})
        lines.append(
            f"GPU: utilization median {ms(gpu.get('utilization_gpu_percent', {}).get('median'), 1)}% "
            f"(p95 {ms(gpu.get('utilization_gpu_percent', {}).get('p95'), 1)}%), memory "
            f"{ms(gpu.get('memory_used_mib', {}).get('median'), 0)} MiB, concurrent compute "
            f"processes median "
            f"{ms(gpu.get('concurrent_compute_processes', {}).get('median'), 1)}, "
            f"{gpu.get('sample_count', 0)} samples"
            + (
                f", throttle reasons observed: {gpu['throttle_reasons_observed']}"
                if gpu.get("throttle_reasons_observed") else ", no throttling flagged"
            )
            + f". Radio telemetry `{telemetry.get('status', 'n/a')}`"
            + (
                f": PUSCH SNR median {ms(snr.get('median'))} dB "
                f"({snr.get('count', 0)} samples), final UL MCS median "
                f"{ms(mcs.get('median'), 1)}."
                if telemetry.get("status") == "COLLECTED"
                else f" ({telemetry.get('error', '') or 'no samples'})."
            )
            + f" Clean −50 dB restore verified: {summary['radio_actuation_restored']}."
        )
        lines.append("")
        lines.append("### Live path through OAI and the deployed edge")
        lines.append("")
        lines.append(runner._stage_table(summary, LIVE_TABLE))
        lines.append("")
        lines.append(
            "### Sensor preparation (reported; structurally outside every uplink interval)"
        )
        lines.append("")
        lines.append(runner._stage_table(summary, PREPARATION_TABLE))
        lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- **Live scenes are not matched across actions.** Each cell drives the "
        "same route from the same start with the same seeds, but a bounded "
        f"{TRANSMITTED_BUDGET}-frame window covers a different stretch of road "
        "depending on how fast frames were transmitted, and detection counts "
        "differ. Post-processing and p025 filtering scale with detection count, "
        "so between-action differences are **not** attributable to payload "
        "alone. Detection counts are reported beside post-processing latency."
    )
    lines.append(
        "- **The Route-B loop is not completed**, so no route-completion, "
        "preparation-coverage or age-of-information claim can be read off this "
        "run, and the campaign's own route/coverage acceptance gate is "
        "inapplicable and unused."
    )
    lines.append(
        "- **The diagnostic edge classifies deadlines instead of dropping.** "
        "Each frame carries `service_target_met` / `processing_horizon_met` and "
        "is still processed, because dropping late frames would delete the "
        "decomposition being measured. Model, codec, thresholds, action "
        "definitions and wire bytes are unchanged, and the instrumented tail "
        "was proved bit-identical to the production tail before measurement."
    )
    lines.append(
        "- **Per-stage CUDA and wall times overlap by construction.** "
        "`decode_tail_cuda_ms` is device time for the tail kernels; most of "
        "that same time appears as wall time in the immediately following "
        "finite check, where the deployed path first synchronizes. The additive "
        "wall partition is the stage-group table; the CUDA columns are the "
        "device-side attribution of the same work."
    )
    lines.append(
        "- Warm-up used a seeded synthetic tensor rather than a live or stored "
        "frame, so its detection count and therefore its post-processing cost "
        "differ from the live scenes; warm-up frames are excluded from every "
        "statistic."
    )
    lines.append(
        "- One bounded execution per action; medians are within-run, so no "
        "run-to-run variance is characterized. **No optimization was attempted.**"
    )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(
        "`LIVE_RUN_MANIFEST.json` (immutable pre-run binding), "
        "`per_frame/action_*.csv` (one per action), `action_summary.csv`, "
        "`LIVE_DIAGNOSTIC_RESULTS.json`, `REPORT.md`, "
        "`live_timing_breakdown.pdf`, `live_timing_breakdown.png`, "
        "`ARTIFACT_MANIFEST.json`, terminal file. No RGB or radar frame, C2 "
        "tensor, compressed payload blob, prediction, segmentation label map or "
        "raw OAI tracer log is retained."
    )
    lines.append("")
    return common.atomic_create_text(path, "\n".join(lines) + "\n")


def build_live_manifest(
    *,
    run_id: str,
    output: Path,
    git_state: Mapping[str, Any],
    bindings: Mapping[str, Any],
    route_b: Mapping[str, Any],
    cuda: Mapping[str, Any],
    container: Mapping[str, Any],
    cold: Mapping[str, Any],
    actions: Mapping[str, Any],
    campaign: Mapping[str, Any],
    synthetic_tests: Mapping[str, Any],
) -> dict[str, Any]:
    return common.seal(
        {
            "schema": LIVE_MANIFEST_SCHEMA,
            "run_id": run_id,
            "execution_token": LIVE_EXECUTE_TOKEN,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "objective": (
                "Decompose, without conflation, application-level one-way "
                "feature-uplink handling through the live OAI 5G path and every "
                "edge stage from datagram receipt through compact result "
                "publication, on live CARLA Route-B scenes, reporting pure "
                "model.decode_tail CUDA time and the deployed tail service span "
                "separately."
            ),
            "amendment": (
                "Supersedes the earlier no-CARLA instruction and the Phase-13C "
                "fit-frame replay. Live CARLA capture only; no stored dataset "
                "frame is replayed and the full Route-B loop is not completed."
            ),
            "scope": {
                "actions": [action_id for action_id, _profile in common.DIAGNOSTIC_ACTIONS],
                "transmitted_frames_per_action": TRANSMITTED_BUDGET,
                "safety_timeout_s": SAFETY_TIMEOUT_S,
                "network_profile_id": common.NETWORK_PROFILE_ID,
                "carla_lifecycles": len(common.DIAGNOSTIC_ACTIONS),
                "radio_lifecycles": len(common.DIAGNOSTIC_ACTIONS),
                "action_fixed_per_cell": True,
                "full_route_b_loop_completed": False,
                "dataset_frames_replayed": 0,
                "training_or_tuning": False,
                "holdout_or_test_frames_read": False,
                "sixteen_cell_pilot": False,
                "two_hundred_eighty_eight_cell_campaign": False,
                "optimization_attempted": False,
            },
            "output_relpath": str(output.relative_to(ROOT)),
            "git": git_state,
            "bindings": bindings,
            "route_b": route_b,
            "environment": {
                **cuda,
                "platform": platform.platform(),
                "python": platform.python_version(),
                "hostname": platform.node(),
                **container,
            },
            "cold_host_preflight": cold,
            "action_binding": actions,
            "collector_configuration": {
                "base_class": "ue_route_b_split_cell_adapter_v1.PassiveSplitCollector",
                "map_install": "ENABLED_DEPLOYMENT_FUNCTION",
                "install_feedback": "ENABLED_DEPLOYMENT_FUNCTION",
                "segmentation_quality_scoring": "DISABLED_FOR_DIAGNOSTIC",
                "object_ground_truth": "DISABLED_FOR_DIAGNOSTIC",
                "exact_installed_record_retrieval": "DISABLED_FOR_DIAGNOSTIC",
                "rationale": (
                    "the deployed service span must carry deployment contention "
                    "without offline scoring work the deployed system never runs"
                ),
            },
            "transport_contract": {
                "sfd1_protocol_version": int(campaign["runtime"]["sfd1_protocol_version"]),
                "udp_fragment_header": campaign["runtime"]["udp_fragment_header"],
                "udp_chunk_bytes": int(campaign["runtime"]["udp_chunk_bytes"]),
                "retransmission": bool(campaign["runtime"]["retransmission"]),
                "entropy_coder": "zstd",
                "inner_bytes_modified": False,
            },
            "radio_contract": {
                "radio_profile_id": "OAI_N78_100MHZ_273PRB_4D5U_V1",
                "launcher": campaign["runtime"]["oai_registered_profile_launcher"],
                "cn5g_owner": "qualified_launcher",
                "clean_restore_noise_power_db": common.CLEAN_NOISE_POWER_DB,
                "network_profile_restart_per_action": (
                    "identical_first_sample_at_first_transmitted_frame"
                ),
                "prefix_samples": int(campaign["network"]["prefix_samples"]),
                "forbidden_after_prefix": list(campaign["network"]["forbidden_after_prefix"]),
            },
            "instrumentation": {
                "production_tail_source": "rl_agent/splitfusion_live_dispatch_v1/context_tail.py",
                "approach": (
                    "run-time subclasses of the qualified tail adapter, UE "
                    "runtime and Route-B collector, installed by assignment; no "
                    "SHA-256 pinned file is edited and the SFD1 authority is "
                    "untouched"
                ),
                "hash_bound_production_files_modified": [],
                "uplink_clock_start": "wall clock immediately before the first UDP datagram send",
                "decode_tail_cuda": "dedicated CUDA event pair around model.decode_tail only",
                "deployed_tail_service": "edge worker start -> compact result published",
                "edge_deadline_policy": "CLASSIFY_AND_RECORD_NEVER_DROP",
                "warmup_input": "seeded synthetic 7-channel tensor, own stream, excluded",
            },
            "synthetic_tests": synthetic_tests,
            "reference_points": {
                "phase13c_controlled_tail_gpu_ms": common.PHASE13C_CONTROLLED_TAIL_MS,
                "phase15_live_frozen_tail_ms": common.PHASE15_LIVE_FROZEN_TAIL_MS,
            },
        },
        "run_manifest_sha256",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", required=True)
    parser.add_argument("--output", default=LIVE_OUTPUT_RELPATH)
    parser.add_argument(
        "--actions",
        default=",".join(str(value) for value, _p in common.DIAGNOSTIC_ACTIONS),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    require(args.execute == LIVE_EXECUTE_TOKEN, "execution token mismatch")

    started = time.time()
    output = (ROOT / args.output).resolve()
    require(not output.exists(), f"create-only output already exists: {output}")

    campaign_path = repo_path(common.PILOT_CONFIG_RELPATH)
    campaign = common.load_json(campaign_path)

    print("preflight: git, bindings, Route-B, container runtime, cold host", flush=True)
    git_state = runner.verify_git_state()
    bindings = runner.verify_bound_inputs()
    route_b = verify_route_b_bindings(campaign)
    container = runner.verify_container_runtime()
    cold = runner.verify_cold_host(campaign, label="run/preflight")
    synthetic = runner.run_synthetic_tests()
    require(bool(synthetic.get("all_passed")), f"synthetic tests failed: {synthetic}")
    registry = SplitActionRegistry.from_runtime_binding()
    actions = runner.verify_actions(registry)
    print("preflight: CUDA device identity", flush=True)
    cuda = runner.verify_cuda()

    run_id = output.name
    output.mkdir(parents=True, exist_ok=False)
    (output / "per_frame").mkdir(parents=False, exist_ok=False)
    manifest = build_live_manifest(
        run_id=run_id, output=output, git_state=git_state, bindings=bindings,
        route_b=route_b, cuda=cuda, container=container, cold=cold, actions=actions,
        campaign=campaign, synthetic_tests=synthetic,
    )
    common.atomic_create_json(output / "LIVE_RUN_MANIFEST.json", manifest)
    print(f"manifest sealed: {manifest['run_manifest_sha256']}", flush=True)

    device = torch.device("cuda:0")
    print("loading one resident UE for synthetic edge warm-up payloads", flush=True)
    ue, _ledger, _models, _base, _registry = preload_ue(device)

    reports: list[dict[str, Any]] = []
    status = TERMINAL_FAILURE
    failure = ""
    selected = [int(value) for value in str(args.actions).split(",")]
    try:
        for action_id in selected:
            reports.append(
                run_live_action(
                    action_id=action_id, campaign=campaign,
                    campaign_path=campaign_path, ue=ue, registry=registry,
                    run_id=run_id,
                )
            )
        status = TERMINAL_SUCCESS
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if reports:
            summaries = [summarize_live_action(report) for report in reports]
            comparisons = build_live_comparisons(summaries)
            for report in reports:
                name = f"action_{report['action_id']:02d}_{report['profile_id']}.csv"
                write_live_per_frame_csv(output / "per_frame" / name, report)
            write_live_summary_csv(output / "action_summary.csv", summaries)
            figure_error = ""
            try:
                write_live_figure(output / "live_timing_breakdown", summaries, comparisons)
            except Exception as exc:
                figure_error = f"{type(exc).__name__}: {exc}"
            runtime_seconds = time.time() - started
            common.atomic_create_json(
                output / "LIVE_DIAGNOSTIC_RESULTS.json",
                {
                    "schema": LIVE_SCHEMA,
                    "run_id": run_id,
                    "status": status,
                    "failure": failure,
                    "figure_error": figure_error,
                    "run_manifest_sha256": manifest["run_manifest_sha256"],
                    "runtime_seconds": runtime_seconds,
                    "actions_executed": [summary["action_id"] for summary in summaries],
                    "comparisons": comparisons,
                    "action_summaries": summaries,
                    "lifecycle": [
                        {
                            "cell_id": report["cell_id"],
                            "action_id": report["action_id"],
                            "cold_before": report["cold_before"],
                            "cold_after": report.get("cold_after"),
                            "carla": report.get("carla"),
                            "carla_shutdown_verified": report.get(
                                "carla_teardown", {}
                            ).get("shutdown_verified"),
                            "radio_attachment": report.get("radio_attachment"),
                            "radio_teardown_gates_passed": report.get(
                                "radio_teardown", {}
                            ).get("teardown", {}).get("all_lifecycle_gates_passed"),
                            "final_restore_noise_power_db": report.get(
                                "radio_teardown", {}
                            ).get("final_restore", {}).get("noise_power_db"),
                            "map_process_stopped": report.get("map_process_stopped"),
                            "cell_scratch_removed": report.get("cell_scratch_removed"),
                            "capture": report.get("capture"),
                            "wall_seconds": report.get("wall_seconds"),
                        }
                        for report in reports
                    ],
                },
            )
            write_live_report(
                output / "REPORT.md", manifest=manifest, summaries=summaries,
                comparisons=comparisons, runtime_seconds=runtime_seconds,
            )
            runner.write_artifact_manifest(output)
            common.atomic_create_json(
                output / status,
                {
                    "terminal": status, "run_id": run_id,
                    "actions": [summary["action_id"] for summary in summaries],
                    "failure": failure,
                    "finished_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
        else:
            common.atomic_create_json(
                output / "NO_SCIENTIFIC_ROWS.json",
                {
                    "run_id": run_id, "failure": failure,
                    "reason": "no action completed, so no scientific row exists",
                    "finished_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            runner.write_artifact_manifest(output)
            common.atomic_create_json(
                output / TERMINAL_FAILURE,
                {
                    "terminal": TERMINAL_FAILURE, "run_id": run_id, "actions": [],
                    "failure": failure,
                    "finished_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
    print(f"terminal: {status}", flush=True)
    return 0 if status == TERMINAL_SUCCESS else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DiagnosticError as error:
        print(f"live diagnostic contract error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
