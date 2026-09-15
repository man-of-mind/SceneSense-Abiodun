#!/usr/bin/env python3
"""Run one create-only action-50 sensor-preparation diagnostic cell.

This runner deliberately reuses the qualified direct-map configuration and
the Phase-15 single-cell lifecycle.  It adds only profiling metadata and an
additive adapter wrapper; the route, radio, action, model, sensor, queue,
deadline, and map-installation contracts remain the registered ones.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_agent import splitfusion_direct_edge_map_live_validation_v1 as direct  # noqa: E402
from rl_agent import ue_288_campaign_supervisor as supervisor  # noqa: E402


SCHEMA = "scenesense.splitfusion.sensor_preparation_live_cell.v1"
TOKEN = "SPLITFUSION_SENSOR_PREPARATION_LIVE_CELL"
BASELINE_MODE = "INSTRUMENTED_PRODUCTION_EQUIVALENT"
ACTION_ID = 50
PROFILE_ID = "split_ae64_uint4_q5000"
NETWORK_PROFILE = "FAVORABLE_STABLE"
SAMPLE_TARGET = 500
WARMUP_SENT_FRAMES = 20
EQUIVALENCE_FRAMES = 8
STARTING_HEAD = "b846c484b6de7a633d502c0261ca4561c1a7e55c"
DEFAULT_CONFIG = (
    ROOT / "rl_agent/configs/splitfusion_direct_edge_map_live_validation_v1.json"
)
PROFILED_ADAPTER = (
    ROOT
    / "rl_agent/splitfusion_supervisor_analysis_v1/profiled_direct_adapter_v1.py"
)
PROFILER = (
    ROOT / "rl_agent/splitfusion_supervisor_analysis_v1/profiled_sensor_stages.py"
)
SUCCESS_TERMINAL = "SPLITFUSION_SENSOR_PREPARATION_BASELINE_COMPLETE"


STAGES = (
    ("P01_rgb_callback", "profile_rgb_callback_ms", "callback"),
    ("P02_radar_callback", "profile_radar_callback_ms", "callback"),
    ("P03_semantic_callback", "profile_semantic_callback_ms", "callback"),
    ("P04_worker_schedule_wait", "profile_worker_schedule_wait_ms", "wait"),
    ("P05_rgb_callback_to_worker", "profile_rgb_callback_to_worker_ms", "wait"),
    ("P06_sensor_wait", "sensor_wait_ms", "wait"),
    ("P07_radar_window_extraction", "radar_window_ms", "compute"),
    ("P08_radar_spherical_to_world", "profile_radar_spherical_to_world_ms", "compute"),
    ("P09_stationary_track_update", "profile_radar_stationary_tracking_ms", "compute"),
    ("P10_world_to_camera", "profile_radar_world_to_camera_ms", "compute"),
    ("P11_projection_and_bounds", "profile_radar_projection_ms", "compute"),
    ("P12_radar_rasterization", "profile_radar_rasterization_ms", "compute"),
    ("P13_radar_evidence_packaging", "profile_radar_packaging_ms", "compute"),
    ("P14_carla_bgra_to_bgr", "rgb_convert_ms", "compute"),
    ("P15_bgr_to_rgb", "profile_camera_bgr_to_rgb_ms", "compute"),
    ("P16_rgb_resize", "profile_camera_resize_ms", "compute"),
    ("P17_rgb_tensor_pack", "profile_camera_tensor_pack_ms", "compute"),
    ("P18_rgb_h2d", "profile_camera_h2d_ms", "compute_cuda_event"),
    ("P19_normalization_constants", "profile_camera_normalization_constants_wall_ms", "compute"),
    ("P20_rgb_normalize", "profile_camera_normalize_ms", "compute_cuda_event"),
    ("P21_radar_resize_pack", "profile_radar_resize_pack_ms", "compute"),
    ("P22_radar_h2d", "profile_radar_h2d_ms", "compute_cuda_event"),
    ("P23_seven_channel_concatenate", "profile_seven_channel_concatenate_ms", "compute_cuda_event"),
    ("P24_immutable_evaluation_snapshot", "scene_snapshot_ms", "evaluation_only"),
    ("P25_unattributed_pre_front", "profile_unattributed_pre_front_ms", "compute"),
    ("P26_diagnostic_final_sync_wait", "profile_seven_channel_diagnostic_sync_wait_ms", "diagnostic_overhead"),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise supervisor.CampaignError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"required CSV is absent: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(value: Any) -> float | None:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def quantile(values: Sequence[float], fraction: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def distribution(values: Sequence[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None}
    return {
        "count": len(finite),
        "p50_ms": quantile(finite, 0.50),
        "p95_ms": quantile(finite, 0.95),
        "p99_ms": quantile(finite, 0.99),
        "mean_ms": statistics.fmean(finite),
        "maximum_ms": max(finite),
        "sum_ms": math.fsum(finite),
    }


def correlation(rows: Sequence[Mapping[str, str]], left: str, right: str) -> dict[str, Any]:
    pairs = [
        (a, b)
        for row in rows
        if (a := number(row.get(left))) is not None
        and (b := number(row.get(right))) is not None
    ]
    if len(pairs) < 3:
        return {"count": len(pairs), "pearson_r": None}
    left_values, right_values = zip(*pairs)
    left_mean = statistics.fmean(left_values)
    right_mean = statistics.fmean(right_values)
    covariance = math.fsum(
        (a - left_mean) * (b - right_mean) for a, b in pairs
    )
    left_energy = math.fsum((a - left_mean) ** 2 for a in left_values)
    right_energy = math.fsum((b - right_mean) ** 2 for b in right_values)
    denominator = math.sqrt(left_energy * right_energy)
    return {
        "count": len(pairs),
        "pearson_r": covariance / denominator if denominator else None,
    }


def wall_ns(value: Any) -> int:
    parsed = number(value)
    require(parsed is not None, f"invalid wall-clock value: {value!r}")
    return int(round(parsed * 1_000_000_000))


def clock_bridge(sent: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    anchors = []
    for row in sent:
        if row.get("feature_received_at") and row.get("edge_result_received_ns"):
            anchors.append(
                wall_ns(row["feature_received_at"])
                - int(row["edge_result_received_ns"])
            )
    require(anchors, "no same-event wall/monotonic clock anchors")
    offset = int(statistics.median(anchors))
    deviations = [abs(value - offset) / 1e6 for value in anchors]
    return {
        "definition": "feature_received_at_wall_ns - edge_result_received_ns",
        "same_event_pair": True,
        "same_host": True,
        "anchors": len(anchors),
        "median_offset_ns": offset,
        "absolute_deviation_ms": distribution(deviations),
    }


def keyed(rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, str], Mapping[str, str]]:
    output: dict[tuple[str, str], Mapping[str, str]] = {}
    for row in rows:
        key = (str(row.get("stream_id")), str(row.get("frame_id")))
        require(key not in output, f"duplicate identity in ledger: {key}")
        output[key] = row
    return output


def full_path_metrics(
    window: Sequence[Mapping[str, str]],
    ingest: Sequence[Mapping[str, str]],
    bridge: Mapping[str, Any],
) -> dict[str, Any]:
    ingest_by_key = keyed(
        [row for row in ingest if row.get("outcome") == "RESULT_INSTALLED"]
    )
    offset = int(bridge["median_offset_ns"])
    samples: dict[str, list[float]] = {
        "ue_action": [],
        "feature_uplink": [],
        "edge_compute": [],
        "map_service": [],
        "capture_to_install_aoi": [],
        "action_start_to_install": [],
    }
    joined = 0
    for row in window:
        key = (str(row.get("stream_id")), str(row.get("frame_id")))
        installed = ingest_by_key.get(key)
        if installed is None:
            continue
        joined += 1
        action_start_ns = int(row["capture_started_ns"]) + offset
        send_start_ns = int(row["ue_prepare_finished_ns"]) + offset
        capture_ns = wall_ns(row["capture_wall_s"])
        install_ns = wall_ns(installed["map_install_at"])
        edge_arrival_ns = wall_ns(row["edge_receipt_wall_s"])
        edge_start_ns = wall_ns(installed["edge_compute_start_wall_s"])
        edge_finish_ns = wall_ns(installed["edge_compute_finish_wall_s"])
        map_ingest_ns = wall_ns(installed["map_ingest_at"])
        boundaries = (capture_ns, action_start_ns, send_start_ns, edge_arrival_ns)
        require(
            all(right + 1_000_000 >= left for left, right in zip(boundaries, boundaries[1:])),
            f"non-causal UE/uplink boundaries for {key}",
        )
        samples["ue_action"].append((send_start_ns - action_start_ns) / 1e6)
        samples["feature_uplink"].append((edge_arrival_ns - send_start_ns) / 1e6)
        samples["edge_compute"].append((edge_finish_ns - edge_start_ns) / 1e6)
        samples["map_service"].append((install_ns - map_ingest_ns) / 1e6)
        samples["capture_to_install_aoi"].append((install_ns - capture_ns) / 1e6)
        samples["action_start_to_install"].append((install_ns - action_start_ns) / 1e6)
    return {
        "window_installed_joined": joined,
        "metrics": {name: distribution(values) for name, values in samples.items()},
    }


def complete_publication_join(
    ingest: Sequence[Mapping[str, str]], publication: Sequence[Mapping[str, str]]
) -> dict[str, Any]:
    installed = [row for row in ingest if row.get("outcome") == "RESULT_INSTALLED"]
    publication_by_key = keyed(publication)
    missing: list[str] = []
    incomplete: list[str] = []
    negative: list[str] = []
    bad_keys: set[tuple[str, str]] = set()
    for row in installed:
        key = (str(row.get("stream_id")), str(row.get("frame_id")))
        edge = publication_by_key.get(key)
        if edge is None:
            missing.append(f"{key[0]}:{key[1]}")
            continue
        merged = dict(edge)
        merged.update({name: value for name, value in row.items() if str(value) != ""})
        for stage, start, finish in direct.DIRECT_STAGE_INTERVALS:
            left, right = number(merged.get(start)), number(merged.get(finish))
            if left is None or right is None:
                incomplete.append(f"{key[0]}:{key[1]}:{stage}")
                bad_keys.add(key)
            elif right < left:
                negative.append(f"{key[0]}:{key[1]}:{stage}")
                bad_keys.add(key)
    complete = len(installed) - len(bad_keys) - len(missing)
    return {
        "installed_frames": len(installed),
        "publication_rows": len(publication),
        "joined_frames": len(installed) - len(missing),
        "complete_frames": complete,
        "join_fraction": ((len(installed) - len(missing)) / len(installed)) if installed else None,
        "complete_fraction": complete / len(installed) if installed else None,
        "missing_examples": missing[:10],
        "incomplete_examples": incomplete[:10],
        "negative_examples": negative[:10],
    }


def find_attempt(campaign_root: Path, cell_id: str) -> Path:
    attempts = sorted((campaign_root / "cells" / cell_id / "attempts").glob("attempt_*"))
    require(len(attempts) == 1, f"expected exactly one create-only attempt, found {len(attempts)}")
    return attempts[0]


def analyse(
    attempt: Path, cell: supervisor.Cell, sample_target: int,
    warmup: int, run_result: Mapping[str, Any], final_cold: Mapping[str, Any],
) -> dict[str, Any]:
    per_frame = read_csv(attempt / "per_frame_metrics.csv")
    sent = sorted(
        [row for row in per_frame if row.get("prepare_status") == "SENT"],
        key=lambda row: number(row.get("capture_wall_s")) or 0.0,
    )
    require(len(sent) >= warmup + sample_target, "fewer than the registered profiling frames were sent")
    window = sent[warmup : warmup + sample_target]
    direct_dir = attempt / "direct_edge_map"
    publication_path = direct_dir / "direct_edge_publication.csv"
    ingest = read_csv(direct_dir / "direct_map_ingest.csv")
    publication = read_csv(publication_path)
    bridge = clock_bridge(sent)

    totals = [
        value for row in window
        if (value := number(row.get("profile_sensor_compute_production_estimate_ms")))
        is not None
    ]
    total_sum = math.fsum(totals)
    stages: dict[str, Any] = {}
    for label, field, category in STAGES:
        values = [
            value for row in window if (value := number(row.get(field))) is not None
        ]
        stage = distribution(values)
        stage.update(
            {
                "field": field,
                "category": category,
                "percentage_of_total_sensor_compute": (
                    100.0 * math.fsum(values) / total_sum if total_sum and values else None
                ),
            }
        )
        stages[label] = stage

    profile_exact = [
        row for row in sent if str(row.get("profile_equivalence_checked")).lower() in {"1", "true"}
    ]
    exact_holds = all(
        str(row.get(field)).lower() in {"1", "true"}
        for row in profile_exact
        for field in (
            "profile_radar_tensor_exact",
            "profile_radar_evidence_exact",
            "profile_model_input_exact",
        )
    )
    join = complete_publication_join(ingest, publication)
    direct_result = direct.evaluate_cell(attempt, cell, sample_target)
    ready = json.loads((direct_dir / "direct_map_ready.json").read_text(encoding="utf-8"))
    summary = json.loads((attempt / "RESULTS_SUMMARY.json").read_text(encoding="utf-8"))
    structural = dict(summary.get("structural_acceptance") or {})
    full_path = full_path_metrics(window, ingest, bridge)

    correlation_fields = {
        "radar_points": "raw_radar_return_count",
        "visible_actors": "profile_visible_actor_count",
        "ego_speed": "ego_speed_mps",
        "ego_acceleration": "profile_ego_acceleration_mps2",
        "ego_yaw_rate": "profile_ego_yaw_rate_deg_s",
        "route_tick": "route_tick",
        "frame_id": "frame_id",
    }
    gates = {
        "single_registered_action50_cell": (
            cell.action_id == ACTION_ID
            and cell.profile_id == PROFILE_ID
            and cell.network_profile_id == NETWORK_PROFILE
        ),
        "analysis_window_exactly_500": len(window) == sample_target,
        "exact_equivalence_frames_complete": (
            len(profile_exact) == EQUIVALENCE_FRAMES and exact_holds
        ),
        "frame_action_context_identity_exact": (
            all(
                int(row["action_id"]) == ACTION_ID
                and row["profile_id"] == PROFILE_ID
                and str(row.get("stream_id") or "") != ""
                and row["capture_id"]
                == f"{row['stream_id']}:{int(row['frame_id'])}"
                and str(row.get("frame_context_valid")).lower() in {"1", "true"}
                for row in sent
            )
            and int(direct_result["identity_mismatches"]) == 0
        ),
        "one_diagnostic_cuda_sync_per_profiled_frame": all(
            number(row.get("profile_cuda_substage_synchronizations")) == 1.0
            for row in sent
        ),
        "publication_ledger_survived_teardown": publication_path.is_file() and bool(publication),
        "all_installed_frames_join_completely": (
            join["installed_frames"] > 0
            and join["joined_frames"] == join["installed_frames"]
            and join["complete_frames"] == join["installed_frames"]
        ),
        "renderer_disabled": ready.get("render_thread_started") is False,
        "install_always_precedes_ack": (
            int(direct_result["ack_before_install_frames"]) == 0
            and int(direct_result["install_before_ack_frames"]) == int(join["installed_frames"])
        ),
        "no_object_records_traversed_radio": (
            direct_result["edge_object_records_on_radio"] is False
            and int(direct_result["ue_record_bearing_messages"]) == 0
        ),
        "exact_terminal_accounting": (
            int(direct_result["captures_without_terminal"]) == 0
            and int(direct_result["unexpected_terminals"]) == 0
            and int(direct_result["captures_with_multiple_terminals"]) == 0
        ),
        "cell_runtime_passed": run_result.get("status") == "PASSED",
        "final_host_cold": bool(final_cold),
    }
    return {
        "schema": SCHEMA,
        "mode": BASELINE_MODE,
        "cell": {
            "cell_id": cell.cell_id,
            "action_id": cell.action_id,
            "profile_id": cell.profile_id,
            "family": "AE64",
            "quantizer": "UINT4",
            "q": 0.50,
            "network_profile": cell.network_profile_id,
        },
        "diagnostic_timing_semantics": {
            "cpu": "perf_counter_ns",
            "cuda": "CUDA event pairs read after one final synchronization per prepared frame",
            "diagnostic_only_because_of_final_cuda_sync": True,
            "production_enqueue_wall_time_has_no_per_substage_sync": True,
        },
        "frames": {
            "prepared_rows": len(per_frame),
            "sent": len(sent),
            "warmup_sent_excluded": warmup,
            "analysis_sample": len(window),
            "analysis_first_frame": int(window[0]["frame_id"]),
            "analysis_last_frame": int(window[-1]["frame_id"]),
        },
        "stages": stages,
        "total_sensor_compute": distribution(totals),
        "sensor_compute_definition": (
            "pre_front_compute_ms excluding immutable evaluation snapshot, plus "
            "seven-channel production enqueue wall time; sensor wait and worker "
            "scheduling are excluded"
        ),
        "complete_path": full_path,
        "clock_bridge": bridge,
        "correlations": {
            name: correlation(
                window, "profile_sensor_compute_production_estimate_ms", field
            )
            for name, field in correlation_fields.items()
        },
        "preparation": {
            "route_ticks": structural.get("route_ticks"),
            "scheduled_frames": structural.get("scheduled_frames"),
            "eligible_preparation_frames": structural.get("eligible_preparation_frames"),
            "sent_frames": structural.get("sent_frames"),
            "coverage": structural.get("sensor_preparation_coverage"),
            "minimum_coverage": structural.get("minimum_sensor_preparation_coverage"),
            "split_frames_dropped": summary.get("split_frames_dropped"),
            "status_counts": (
                (structural.get("realtime_recovery") or {}).get("prepare_status_counts")
            ),
        },
        "equivalence": {
            "checked_frames": len(profile_exact),
            "radar_tensor_exact": exact_holds,
            "radar_evidence_exact": exact_holds,
            "model_input_exact": exact_holds,
            "declared_tolerance": {"numpy": "exact", "torch": "exact"},
        },
        "publication_to_install": join,
        "direct_map": direct_result,
        "final_host_cold": dict(final_cold),
        "gates": gates,
        "status": "PASS" if all(gates.values()) else "FAIL",
    }


def write_outputs(root: Path, result: Mapping[str, Any]) -> None:
    result_path = root / "SENSOR_PROFILE_RESULT.json"
    supervisor.atomic_json(result_path, result)
    lines = [
        "# Live sensor-preparation baseline",
        "",
        f"Status: **{result['status']}**",
        "",
        "| Stage | Category | N | P50 ms | P95 ms | P99 ms | % sensor compute |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in result["stages"].items():
        show = lambda value: "n/a" if value is None else f"{float(value):.3f}"
        lines.append(
            f"| {name} | {row['category']} | {row['count']} | "
            f"{show(row['p50_ms'])} | {show(row['p95_ms'])} | "
            f"{show(row['p99_ms'])} | {show(row['percentage_of_total_sensor_compute'])} |"
        )
    lines.extend(["", "## Gates", ""])
    for name, holds in result["gates"].items():
        lines.append(f"- {name}: {'PASS' if holds else 'FAIL'}")
    report = root / "REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    hashes = {
        path.name: supervisor.sha256_file(path) for path in (result_path, report)
    }
    supervisor.atomic_json(
        root / "artifact_manifest.json",
        {"schema": f"{SCHEMA}.artifacts", "sha256": hashes},
    )
    if result["status"] == "PASS":
        supervisor.write_create_only(
            root / SUCCESS_TERMINAL,
            json.dumps(
                {"schema": f"{SCHEMA}.terminal", "status": "PASS", "sha256": hashes},
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--execute", required=True)
    value.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    value.add_argument("--output-root", type=Path, required=True)
    value.add_argument("--carla-port", type=int, default=2000)
    value.add_argument("--maximum-loop-sim-s", type=float, default=600.0)
    value.add_argument("--sample-target", type=int, default=SAMPLE_TARGET)
    value.add_argument("--warmup-sent-frames", type=int, default=WARMUP_SENT_FRAMES)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    require(args.execute == TOKEN, f"exact execution token required: {TOKEN}")
    require(args.sample_target == SAMPLE_TARGET, "profiling sample target is locked to 500")
    require(args.warmup_sent_frames == WARMUP_SENT_FRAMES, "profiling warmup is locked to 20 sent frames")
    config_path = args.config.resolve(strict=True)
    config, cells, _trace_hashes = supervisor.validate_static(config_path)
    candidates = [
        cell for cell in cells
        if cell.action_id == ACTION_ID and cell.network_profile_id == NETWORK_PROFILE
    ]
    require(len(candidates) == 1, "registered action-50/FAVORABLE_STABLE cell is not unique")
    cell = candidates[0]
    require(cell.profile_id == PROFILE_ID, "action 50 profile identity drift")
    require(
        config["authorization"]["another_288_live_campaign_authorized"] is False
        and config["authorization"]["campaign_288_authorized"] is False,
        "the completed 288-cell campaign must remain unauthorized",
    )
    placement = config["direct_edge_map"]["cpu_reservation"]
    require(
        placement["map_render"] == "off"
        and all(
            placement[name] == ""
            for name in (
                "edge_compute_cpus", "edge_receive_cpus",
                "map_ingest_cpus", "map_receive_cpus",
            )
        ),
        "renderer-off/unpinned direct-map contract drift",
    )
    require(PROFILED_ADAPTER.is_file() and PROFILER.is_file(), "profiling sources are absent")
    ancestor = subprocess.run(
        ("git", "merge-base", "--is-ancestor", STARTING_HEAD, "HEAD"),
        cwd=str(ROOT), check=False, stdin=subprocess.DEVNULL,
    )
    require(ancestor.returncode == 0, "required starting HEAD is not an ancestor")
    worktree = supervisor.verify_live_pilot_worktree()
    gpu = supervisor._phase15_gpu_audit()
    cold_before = supervisor._require_phase15_application_cold(config)
    supervisor.verify_real_launch_readiness(config)
    supervisor.verify_resolved_models(config, supervisor.read_catalog(config))

    output = args.output_root.resolve(strict=False)
    experiments = (ROOT / "experiments").resolve(strict=True)
    try:
        output.relative_to(experiments)
    except ValueError as exc:
        raise supervisor.CampaignError("output must remain beneath experiments") from exc
    require(not output.exists(), f"create-only output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=False, exist_ok=False)
    config["_maximum_loop_sim_s_override"] = float(args.maximum_loop_sim_s)
    config["_sensor_preparation_diagnostic"] = {
        "mode": BASELINE_MODE,
        "sample_target": SAMPLE_TARGET,
        "warmup_sent_frames": WARMUP_SENT_FRAMES,
        "equivalence_frames": EQUIVALENCE_FRAMES,
        "numpy_equivalence": "EXACT",
        "torch_equivalence": "EXACT",
        "cuda_timing": "EVENTS_WITH_ONE_FINAL_SYNCHRONIZATION_PER_FRAME",
        "production_wall_measurement_has_per_stage_synchronization": False,
    }
    manifest = {
        "schema": f"{SCHEMA}.manifest",
        "mode": BASELINE_MODE,
        "git": worktree,
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": supervisor.sha256_file(config_path),
        "profiled_adapter_sha256": supervisor.sha256_file(PROFILED_ADAPTER),
        "profiled_stages_sha256": supervisor.sha256_file(PROFILER),
        "cell_id": cell.cell_id,
        "action_id": ACTION_ID,
        "profile_id": PROFILE_ID,
        "network_profile": NETWORK_PROFILE,
        "sample_target": SAMPLE_TARGET,
        "warmup_sent_frames": WARMUP_SENT_FRAMES,
        "gpu": gpu,
        "cold_before": cold_before,
        "started_at_unix_s": time.time(),
        "another_288_campaign_authorized": False,
    }
    supervisor.write_create_only(
        output / "run_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    ledger: list[dict[str, Any]] = []
    run_result = supervisor.run_one_cell(
        config=config,
        cell=cell,
        adapter=PROFILED_ADAPTER,
        campaign_root=output,
        ledger_rows=ledger,
        port=int(args.carla_port),
    )
    supervisor.atomic_json(
        output / "campaign_ledger.json",
        {
            "schema": "scenesense.sensor_preparation_single_cell_ledger.v1",
            "cell_id": cell.cell_id,
            "attempts": [run_result],
        },
    )
    cold_after = supervisor._require_phase15_application_cold(config)
    attempt = find_attempt(output, cell.cell_id)
    if run_result.get("status") != "PASSED":
        terminal_path = output / str(run_result["terminal"])
        require(terminal_path.is_file(), "failed cell has no durable terminal")
        failure = {
            "schema": f"{SCHEMA}.runtime_failure",
            "status": str(run_result.get("status")),
            "cell_id": cell.cell_id,
            "attempt": dict(run_result),
            "terminal_sha256": supervisor.sha256_file(terminal_path),
            "terminal": json.loads(terminal_path.read_text(encoding="utf-8")),
            "final_host_cold": dict(cold_after),
            "analysis_not_started": True,
        }
        supervisor.atomic_json(output / "RUNTIME_FAILURE.json", failure)
        print(json.dumps({"status": failure["status"], "output": str(output)}, sort_keys=True))
        return 1
    result = analyse(
        attempt, cell, SAMPLE_TARGET, WARMUP_SENT_FRAMES,
        run_result, cold_after,
    )
    write_outputs(output, result)
    print(json.dumps({"status": result["status"], "output": str(output)}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
