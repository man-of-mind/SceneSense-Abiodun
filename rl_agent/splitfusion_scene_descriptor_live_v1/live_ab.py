#!/usr/bin/env python3
"""Create-only bounded A/B timing for SI/P40 in the live sensor path."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from rl_agent import splitfusion_direct_edge_map_live_validation_v1 as direct_eval
from rl_agent import ue_288_campaign_supervisor as supervisor
from rl_agent.splitfusion_hybrid_sac_v1.scene_descriptors import (
    SCHEMA_ID as DESCRIPTOR_SCHEMA_ID,
    SCHEMA_SHA256 as DESCRIPTOR_SCHEMA_SHA256,
)
from rl_agent.splitfusion_quality_feedback_probe_v1.live_probe import (
    _bounded_log_record,
    _stop_child,
)
from rl_agent.splitfusion_supervisor_analysis_v1.run_sensor_preparation_cell_v1 import (
    OPTIMIZED_V2_MODE,
    V2_EQUIVALENCE_FRAMES,
    complete_publication_join,
    distribution,
    number,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    ROOT / "rl_agent/configs/splitfusion_direct_edge_map_live_validation_v1.json"
)
EXECUTE_TOKEN = "SPLITFUSION_SCENE_DESCRIPTOR_LIVE_AB_V1_EXECUTE"
SUCCESS_TERMINAL = "SPLITFUSION_SCENE_DESCRIPTOR_LIVE_AB_V1_COMPLETE"
FAILURE_TERMINAL = "SPLITFUSION_SCENE_DESCRIPTOR_LIVE_AB_V1_FAILED"
CHILD_MODULE = "rl_agent.splitfusion_scene_descriptor_live_v1.live_cell_child"
ACTION_ID = 50
PROFILE_ID = "split_ae64_uint4_q5000"
NETWORK_PROFILE = "FAVORABLE_STABLE"
TRANSMITTED_BUDGET = 520
WARMUP_SENT = 20
ANALYSIS_FRAMES = 500
SAFETY_TIMEOUT_S = 90.0
CHILD_TIMEOUT_S = 900.0
VARIANTS = (("descriptor_off", "OFF"), ("descriptor_on", "SI_P40_V1"))


class SceneDescriptorABError(RuntimeError):
    """The bounded A/B contract is not satisfied."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SceneDescriptorABError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    require(not path.exists(), f"create-only output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(dict(payload), sort_keys=True, indent=1, default=str) + "\n",
        encoding="utf-8",
    )
    os.link(temporary, path)
    temporary.unlink()


def _read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"required evidence is absent: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _selected_cell(config: Mapping[str, Any]) -> supervisor.Cell:
    cells = [
        cell
        for cell in supervisor.enumerate_cells(config)
        if cell.action_id == ACTION_ID
        and cell.network_profile_id == NETWORK_PROFILE
        and cell.profile_id == PROFILE_ID
    ]
    require(len(cells) == 1, "registered action-50/FAVORABLE cell is not unique")
    return cells[0]


def _campaign(
    config: Mapping[str, Any], *, run_id: str, descriptor_mode: str
) -> dict[str, Any]:
    campaign = deepcopy(dict(config))
    campaign.pop("_qualification", None)
    campaign["campaign_id"] = f"scene_descriptor_live_v1/{run_id}/{descriptor_mode}"
    campaign["_sensor_preparation_diagnostic"] = {
        "mode": OPTIMIZED_V2_MODE,
        "sample_target": ANALYSIS_FRAMES,
        "warmup_sent_frames": WARMUP_SENT,
        "equivalence_frames": V2_EQUIVALENCE_FRAMES,
        "numpy_equivalence": "EXACT",
        "torch_equivalence": "EXACT",
        "scene_descriptor_mode": descriptor_mode,
        "scene_descriptor_schema_id": DESCRIPTOR_SCHEMA_ID,
        "scene_descriptor_schema_sha256": DESCRIPTOR_SCHEMA_SHA256,
    }
    return campaign


def _analyse_attempt(
    attempt: Path,
    cell: supervisor.Cell,
    *,
    variant: str,
    descriptor_mode: str,
    child_result: Mapping[str, Any],
) -> dict[str, Any]:
    rows = _read_csv(attempt / "per_frame_metrics.csv")
    sent = sorted(
        [row for row in rows if row.get("prepare_status") == "SENT"],
        key=lambda row: number(row.get("capture_wall_s")) or 0.0,
    )
    require(
        len(sent) == TRANSMITTED_BUDGET,
        f"expected exactly {TRANSMITTED_BUDGET} sent rows, found {len(sent)}",
    )
    window = sent[WARMUP_SENT : WARMUP_SENT + ANALYSIS_FRAMES]
    require(len(window) == ANALYSIS_FRAMES, "analysis window is incomplete")

    def values(field: str) -> list[float]:
        return [
            value
            for row in window
            if (value := number(row.get(field))) is not None
        ]

    descriptor_totals = values("profile_scene_descriptor_total_ms")
    sensor_totals = values("profile_sensor_compute_production_estimate_ms")
    pre_front = values("pre_front_compute_ms")
    seven_channel = values("profile_seven_channel_production_enqueue_wall_ms")
    luma_times = values("profile_scene_luma_ms")
    si_times = values("profile_scene_si_ms")
    p40_times = values("profile_scene_p40_ms")
    require(
        all(
            len(series) == len(window)
            for series in (
                descriptor_totals,
                sensor_totals,
                pre_front,
                seven_channel,
                luma_times,
                si_times,
                p40_times,
            )
        ),
        "sensor/descriptor timing evidence is incomplete",
    )
    sensor_without_descriptor = [
        total - descriptor
        for total, descriptor in zip(sensor_totals, descriptor_totals)
    ]
    require(
        all(value >= -1e-6 for value in sensor_without_descriptor),
        "descriptor timing exceeds its containing sensor interval",
    )

    checked = [
        row
        for row in sent
        if str(row.get("profile_equivalence_checked")).casefold()
        in {"1", "true"}
        and row.get("prepare_status") == "SENT"
    ]
    exact = all(
        str(row.get(field)).casefold() in {"1", "true"}
        for row in checked
        for field in (
            "profile_radar_tensor_exact",
            "profile_radar_evidence_exact",
            "profile_model_input_exact",
        )
    )
    direct = direct_eval.evaluate_cell(attempt, cell, TRANSMITTED_BUDGET)
    publication_path = attempt / "direct_edge_map/direct_edge_publication.csv"
    ingest_path = attempt / "direct_edge_map/direct_map_ingest.csv"
    publication = _read_csv(publication_path)
    ingest = _read_csv(ingest_path)
    publication_join = complete_publication_join(ingest, publication)
    descriptor_valid = all(
        row.get("scene_descriptor_status") == "VALID"
        and row.get("scene_descriptor_camera_status") == "VALID"
        and row.get("scene_descriptor_radar_status") == "VALID"
        and row.get("scene_descriptor_schema_id") == DESCRIPTOR_SCHEMA_ID
        and row.get("scene_descriptor_schema_sha256")
        == DESCRIPTOR_SCHEMA_SHA256
        and number(row.get("camera_si")) is not None
        and number(row.get("radar_p40")) is not None
        and 0.0 <= float(row["radar_p40"]) <= 1.0
        and int(row.get("scene_descriptor_current_sweep_raw_returns") or 0) > 0
        and int(row.get("scene_descriptor_current_sweep_valid_returns") or 0) > 0
        for row in window
    )
    descriptor_disabled = all(
        row.get("scene_descriptor_status") == "DISABLED"
        and (number(row.get("profile_scene_descriptor_total_ms")) or 0.0) == 0.0
        and str(row.get("camera_si") or "") == ""
        and str(row.get("radar_p40") or "") == ""
        for row in window
    )
    collector = dict(child_result.get("collector") or {})
    gates = {
        "exact_transmitted_budget": len(sent) == TRANSMITTED_BUDGET,
        "exact_analysis_window": len(window) == ANALYSIS_FRAMES,
        "action_profile_identity": all(
            int(row["action_id"]) == ACTION_ID
            and row["profile_id"] == PROFILE_ID
            for row in sent
        ),
        "descriptor_contract": (
            descriptor_valid if descriptor_mode == "SI_P40_V1" else descriptor_disabled
        ),
        "exact_tensor_equivalence": (
            len(checked) == V2_EQUIVALENCE_FRAMES and exact
        ),
        "non_discarding_evaluation_drain": (
            collector.get("evaluation_queues_discarded") is False
            and collector.get("quality_and_evaluation_drain_complete") is True
            and collector.get("collector_cleanup_ok") is True
            and not collector.get("collector_failures")
        ),
        "direct_identity_exact": int(direct["identity_mismatches"]) == 0,
        "real_feature_uplink_reached_edge": int(
            direct["edge_feature_messages_reassembled"]
        )
        > 0,
        "edge_tail_and_direct_publication_observed": (
            int(direct["edge_tail_completions"]) > 0
            and int(direct["edge_direct_map_publications"]) > 0
        ),
        "map_install_observed": (
            int(direct["map_updates_reassembled"]) > 0
            and int(direct["map_updates_installed"]) > 0
        ),
        "publication_ledger_join_complete": (
            publication_path.is_file()
            and bool(publication)
            and publication_join["installed_frames"] > 0
            and publication_join["joined_frames"]
            == publication_join["installed_frames"]
            and publication_join["complete_frames"]
            == publication_join["installed_frames"]
        ),
        "object_records_not_on_radio": (
            direct["edge_object_records_on_radio"] is False
            and int(direct["ue_record_bearing_messages"]) == 0
        ),
        "install_precedes_ack": (
            int(direct["ack_before_install_frames"]) == 0
            and int(direct["install_before_ack_frames"])
            == int(direct["map_updates_installed"])
        ),
        "terminal_accounting_exact": (
            int(direct["captures_without_terminal"]) == 0
            and int(direct["unexpected_terminals"]) == 0
            and int(direct["captures_with_multiple_terminals"]) == 0
        ),
    }
    return {
        "schema": "scenesense.scene_descriptor_live_cell_analysis.v1",
        "variant": variant,
        "descriptor_mode": descriptor_mode,
        "cell_id": cell.cell_id,
        "action_id": cell.action_id,
        "profile_id": cell.profile_id,
        "network_profile_id": cell.network_profile_id,
        "full_route_completion_claimed": False,
        "frames": {
            "sent": len(sent),
            "warmup_sent_excluded": WARMUP_SENT,
            "analysis": len(window),
            "prepared_rows": len(rows),
        },
        "timing_ms": {
            "scene_luma": distribution(luma_times),
            "camera_si": distribution(si_times),
            "radar_p40": distribution(p40_times),
            "scene_descriptor_measured_kernels": distribution(descriptor_totals),
            "sensor_compute_with_descriptor_setting": distribution(sensor_totals),
            "sensor_compute_without_descriptor_arithmetic": distribution(
                sensor_without_descriptor
            ),
            "pre_front": distribution(pre_front),
            "seven_channel_prepare": distribution(seven_channel),
        },
        "scene_values": {
            "camera_si": distribution(values("camera_si")),
            "radar_p40": distribution(values("radar_p40")),
            "current_sweep_raw_returns": distribution(
                values("scene_descriptor_current_sweep_raw_returns")
            ),
            "current_sweep_valid_returns": distribution(
                values("scene_descriptor_current_sweep_valid_returns")
            ),
            "current_sweep_invalid_returns": distribution(
                values("scene_descriptor_current_sweep_invalid_returns")
            ),
        },
        "equivalence": {
            "checked_frames": len(checked),
            "exact": exact,
        },
        "direct_edge_map": direct,
        "publication_to_install_join": publication_join,
        "gates": gates,
        "status": "PASS" if all(gates.values()) else "FAIL",
    }


def _run_variant(
    *,
    base_config: Mapping[str, Any],
    registered: supervisor.Cell,
    output_root: Path,
    run_id: str,
    variant: str,
    descriptor_mode: str,
    ordinal: int,
    carla_port: int,
) -> dict[str, Any]:
    unique_cell_id = f"scene_{variant}_{run_id}_{registered.cell_id}"
    cell = supervisor.Cell(
        cell_id=unique_cell_id,
        action_index=registered.action_index,
        action_id=registered.action_id,
        profile_id=registered.profile_id,
        model_family=registered.model_family,
        network_profile_id=registered.network_profile_id,
        trace_id=registered.trace_id,
        seed=registered.seed,
    )
    attempt = output_root / "cells" / variant
    attempt.mkdir(parents=True, exist_ok=False)
    artifacts = attempt / "child_artifacts"
    artifacts.mkdir(parents=False, exist_ok=False)
    scratch = Path(tempfile.mkdtemp(prefix=f"splitfusion_scene_{variant}_"))
    lifecycle = supervisor.import_lifecycle_helper(base_config)
    campaign = _campaign(
        base_config, run_id=run_id, descriptor_mode=descriptor_mode
    )
    campaign_json = scratch / "campaign.json"
    cell_json = scratch / "cell.json"
    campaign_json.write_text(
        json.dumps(campaign, sort_keys=True, indent=1) + "\n", encoding="utf-8"
    )
    cell_json.write_text(
        json.dumps(supervisor.cell_to_dict(cell), sort_keys=True, indent=1) + "\n",
        encoding="utf-8",
    )
    report: dict[str, Any] = {
        "schema": "scenesense.scene_descriptor_live_parent_cell.v1",
        "variant": variant,
        "descriptor_mode": descriptor_mode,
        "cell_id": unique_cell_id,
        "action_id": ACTION_ID,
        "ordinal": ordinal,
        "started_at_unix_s": time.time(),
        "status": "FAILED",
        "full_route_completion_claimed": False,
        "cleanup": {},
        "error": "",
    }
    radio_namespace: Path | None = None
    radio_state: Path | None = None
    attached: Mapping[str, Any] | None = None
    server: Any = None
    pgid: int | None = None
    child: subprocess.Popen[Any] | None = None
    child_pgid: int | None = None
    child_log = attempt / "child_stdout_stderr.log"
    try:
        report["cold_before"] = supervisor._require_phase15_application_cold(
            campaign
        )
        radio_namespace, radio_state, attached = supervisor._start_live_radio(
            campaign, cell, 1, scratch
        )
        report["radio_attachment"] = {
            "status": attached.get("status"),
            "clean_noise_preflight": attached.get("clean_noise_preflight"),
        }
        server, pgid = lifecycle.start_carla(
            carla_port, scratch / "carla_server.log"
        )
        version = lifecycle.wait_for_rpc(carla_port, 180.0)
        require(version is not None, "fresh Epic CARLA did not become RPC-ready")
        report["carla"] = {
            "rpc_port": carla_port,
            "server_version": str(version),
        }
        argv = [
            sys.executable,
            "-m",
            CHILD_MODULE,
            "--campaign-json",
            str(campaign_json),
            "--cell-json",
            str(cell_json),
            "--attempt-dir",
            str(attempt),
            "--temporary-dir",
            str(scratch),
            "--artifacts-dir",
            str(artifacts),
            "--carla-port",
            str(carla_port),
            "--transmitted-budget",
            str(TRANSMITTED_BUDGET),
            "--safety-timeout-s",
            str(SAFETY_TIMEOUT_S),
        ]
        with child_log.open("xb") as stream:
            child = subprocess.Popen(
                argv,
                cwd=str(ROOT),
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=lifecycle.child_env(),
                start_new_session=True,
            )
            child_pgid = os.getpgid(child.pid)
            try:
                child_returncode = int(child.wait(timeout=CHILD_TIMEOUT_S))
            except subprocess.TimeoutExpired as exc:
                _stop_child(child, child_pgid)
                raise SceneDescriptorABError("bounded child exceeded 900 s") from exc
        child_result_path = artifacts / "child_result.json"
        require(child_result_path.is_file(), "child left no durable result")
        child_result = json.loads(child_result_path.read_text(encoding="utf-8"))
        report["child_returncode"] = child_returncode
        report["child"] = child_result
        require(child_returncode == 0, f"child failed: {child_result.get('error')}")
        collector = dict(child_result.get("collector") or {})
        require(
            int(collector.get("transmitted_frames", -1)) == TRANSMITTED_BUDGET
            and collector.get("stop_reason") == "TRANSMITTED_BUDGET_REACHED",
            "child did not stop at the exact transmitted budget",
        )
        report["analysis"] = _analyse_attempt(
            attempt,
            cell,
            variant=variant,
            descriptor_mode=descriptor_mode,
            child_result=child_result,
        )
        require(report["analysis"]["status"] == "PASS", "analysis gates failed")
        report["status"] = "PASSED"
    except KeyboardInterrupt:
        report["status"] = "INTERRUPTED"
        report["error"] = "operator interrupt"
        raise
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        _stop_child(child, child_pgid)
        cleanup = report["cleanup"]
        try:
            cleanup["application"] = supervisor._stop_phase15_application(campaign)
        except BaseException as exc:
            cleanup["application_error"] = f"{type(exc).__name__}: {exc}"
        if server is not None and pgid is not None:
            try:
                cleanup["carla"] = lifecycle.stop_carla(server, pgid, carla_port)
            except BaseException as exc:
                cleanup["carla_error"] = f"{type(exc).__name__}: {exc}"
        if radio_namespace is not None and radio_state is not None:
            try:
                cleanup["radio"] = supervisor._stop_live_radio(
                    campaign, radio_namespace, radio_state, attached
                )
                cleanup["radio_shutdown_verified"] = True
            except BaseException as exc:
                cleanup["radio_error"] = f"{type(exc).__name__}: {exc}"
                cleanup["radio_shutdown_verified"] = False
        report["service_logs"] = [
            _bounded_log_record(scratch / name)
            for name in ("oai_launcher.log", "carla_server.log", "edge_launcher.log")
        ]
        shutil.rmtree(scratch, ignore_errors=True)
        cleanup["service_scratch_removed"] = not scratch.exists()
        try:
            cleanup["cold_after"] = supervisor._require_phase15_application_cold(
                campaign
            )
        except BaseException as exc:
            cleanup["cold_after_error"] = f"{type(exc).__name__}: {exc}"
        cleanup_ok = bool(
            not cleanup.get("application_error")
            and not cleanup.get("carla_error")
            and cleanup.get("radio_shutdown_verified")
            and cleanup.get("service_scratch_removed")
            and "cold_after" in cleanup
            and (
                server is None
                or bool((cleanup.get("carla") or {}).get("shutdown_verified"))
            )
        )
        cleanup["all_gates_passed"] = cleanup_ok
        if not cleanup_ok and report["status"] == "PASSED":
            report["status"] = "FAILED"
            report["error"] = "parent lifecycle cleanup/cold-postflight failed"
        report["finished_at_unix_s"] = time.time()
        report["wall_seconds"] = report["finished_at_unix_s"] - report["started_at_unix_s"]
        _atomic_json(attempt / "CELL_RESULT.json", report)
        terminal = {
            "PASSED": "PASSED.json",
            "INTERRUPTED": "INTERRUPTED.json",
        }.get(report["status"], "FAILED.json")
        _atomic_json(
            attempt / terminal,
            {
                "status": report["status"],
                "cell_id": unique_cell_id,
                "cell_result_sha256": _sha256(attempt / "CELL_RESULT.json"),
            },
        )
    return report


def _source_binding() -> dict[str, str]:
    paths = (
        ROOT / "rl_agent/splitfusion_hybrid_sac_v1/scene_descriptors.py",
        ROOT / "rl_agent/splitfusion_supervisor_analysis_v1/profiled_sensor_stages.py",
        ROOT / "rl_agent/splitfusion_supervisor_analysis_v1/profiled_direct_adapter_v1.py",
        Path(__file__).resolve(),
        Path(__file__).with_name("live_cell_child.py").resolve(),
    )
    return {str(path.relative_to(ROOT)): _sha256(path) for path in paths}


def _comparison(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(len(reports) == 2, "A/B comparison requires exactly two cells")
    by_variant = {str(row["variant"]): row["analysis"] for row in reports}
    off = by_variant["descriptor_off"]["timing_ms"]
    on = by_variant["descriptor_on"]["timing_ms"]
    deltas: dict[str, Any] = {}
    for quantile in ("p50_ms", "p95_ms", "p99_ms"):
        deltas[quantile] = (
            float(on["sensor_compute_with_descriptor_setting"][quantile])
            - float(off["sensor_compute_with_descriptor_setting"][quantile])
        )
    return {
        "primary_same_frame_descriptor_kernel_time_ms": on[
            "scene_descriptor_measured_kernels"
        ],
        "secondary_between_cell_sensor_compute_delta_ms": deltas,
        "interpretation": (
            "same-frame luma/SI/P40 kernel timing is the primary measured "
            "descriptor cost but excludes wrapper/recorder overhead; "
            "the fresh-cell A/B delta is secondary because route scenes and "
            "host scheduling are not paired"
        ),
    }


def _artifact_manifest(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.name.endswith(".partial")
    ]


def run(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve(strict=True)
    output = Path(args.output_root).resolve()
    require(not output.exists(), f"create-only output exists: {output}")
    config, _cells, _trace_hashes = supervisor.validate_static(config_path)
    registered = _selected_cell(config)
    require(
        str(config["runtime"]["architecture"]) == "DIRECT_EDGE_TO_MAP_V1"
        and config["runtime"]["object_records_on_radio"] is False,
        "direct edge-to-map runtime contract drift",
    )
    placement = dict((config.get("direct_edge_map") or {}).get("cpu_reservation") or {})
    require(
        str(placement.get("map_render") or "").casefold() == "off",
        "scene-descriptor timing requires the qualified renderer-off map path",
    )
    supervisor.verify_file_hashes(config)
    supervisor.verify_radio_baseline(config)
    supervisor.verify_real_launch_readiness(config)
    supervisor.verify_route_contract(config)
    if args.preflight:
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "external_processes_started": 0,
                    "cell": supervisor.cell_to_dict(registered),
                    "variants": VARIANTS,
                    "transmitted_budget": TRANSMITTED_BUDGET,
                    "analysis_frames": ANALYSIS_FRAMES,
                    "source_sha256": _source_binding(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    require(args.execute == EXECUTE_TOKEN, f"exact token required: {EXECUTE_TOKEN}")
    worktree = supervisor.verify_live_pilot_worktree()
    gpu = supervisor._phase15_gpu_audit()
    cold = supervisor._require_phase15_application_cold(config)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "scenesense.scene_descriptor_live_ab_manifest.v1",
        "run_id": output.name,
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "git": worktree,
        "gpu": gpu,
        "cold_before": cold,
        "cell": supervisor.cell_to_dict(registered),
        "variants": VARIANTS,
        "transmitted_budget_per_cell": TRANSMITTED_BUDGET,
        "warmup_sent_excluded": WARMUP_SENT,
        "analysis_frames_per_cell": ANALYSIS_FRAMES,
        "full_route_completion_claimed": False,
        "evaluation_queues_discarded": False,
        "source_sha256": _source_binding(),
        "started_at_unix_s": time.time(),
    }
    _atomic_json(output / "RUN_MANIFEST.json", manifest)
    reports: list[dict[str, Any]] = []
    error = ""
    try:
        for ordinal, (variant, descriptor_mode) in enumerate(VARIANTS, start=1):
            print(f"[{variant}] starting bounded live cell", flush=True)
            report = _run_variant(
                base_config=config,
                registered=registered,
                output_root=output,
                run_id=output.name,
                variant=variant,
                descriptor_mode=descriptor_mode,
                ordinal=ordinal,
                carla_port=int(args.carla_port),
            )
            reports.append(report)
            require(report["status"] == "PASSED", f"{variant} failed: {report['error']}")
        result = {
            "schema": "scenesense.scene_descriptor_live_ab_result.v1",
            "status": "PASS",
            "cells": [report["analysis"] for report in reports],
            "comparison": _comparison(reports),
            "finished_at_unix_s": time.time(),
        }
        _atomic_json(output / "RESULT.json", result)
        terminal = SUCCESS_TERMINAL
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        _atomic_json(
            output / "FAILURE.json",
            {
                "schema": "scenesense.scene_descriptor_live_ab_failure.v1",
                "status": "FAIL",
                "error": error,
                "completed_cells": [report.get("variant") for report in reports],
                "finished_at_unix_s": time.time(),
            },
        )
        terminal = FAILURE_TERMINAL
    _atomic_json(
        output / terminal,
        {"terminal": terminal, "error": error, "finished_at_unix_s": time.time()},
    )
    _atomic_json(
        output / "ARTIFACT_MANIFEST.json",
        {"artifacts": _artifact_manifest(output)},
    )
    return 0 if terminal == SUCCESS_TERMINAL else 1


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--preflight", action="store_true")
    value.add_argument("--execute", default="")
    value.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    value.add_argument("--output-root", type=Path)
    value.add_argument("--carla-port", type=int, default=2000)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(list(argv) if argv is not None else None)
    if args.preflight:
        if args.output_root is None:
            args.output_root = ROOT / "experiments/_preflight_unused"
        return run(args)
    require(args.output_root is not None, "--output-root is required for live execution")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
