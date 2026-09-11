#!/usr/bin/env python3
"""Matched live payoff comparison for the v2 tail and predicted scheduler."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from rl_agent.splitfusion_timing_diagnostic_v1 import diagnostic_common as common
from rl_agent.splitfusion_timing_diagnostic_v1 import live_runner as baseline

from . import live_runner as freshness
from .pipeline import CandidatePolicy


EXECUTE_TOKEN = "SPLITFUSION_EDGE_V2_PREDICTED_LIVE_PAYOFF_V1"
SUCCESS_TERMINAL = "SPLITFUSION_EDGE_V2_PREDICTED_LIVE_PAYOFF_COMPLETE"
DEFAULT_OUTPUT = (
    "experiments/splitfusion_edge_freshness_scheduler_v1/"
    "20260910_live_v1_fixed25_vs_v2_predicted"
)
CHILD_MODULE = "rl_agent.splitfusion_edge_freshness_scheduler_v1.live_cell_child"
ACTIONS = (30, 15, 50, 71)
VARIANTS = (
    {
        "id": "V1_FIXED_25_MS",
        "policy": CandidatePolicy.LATEST_ONLY_25_MS,
        "edge_module": (
            "rl_agent.splitfusion_edge_freshness_scheduler_v1.edge_service_25ms"
        ),
        "tail": "SYNCHRONIZATION_LIGHT_V1",
    },
    {
        "id": "V2_PREDICTED_INSTALL_HORIZON",
        "policy": CandidatePolicy.PREDICTED_INSTALL_HORIZON,
        "edge_module": (
            "rl_agent.splitfusion_edge_freshness_scheduler_v1."
            "edge_service_v2_predicted"
        ),
        "tail": "SYNCHRONIZATION_LIGHT_V2",
    },
)
SOURCES = (
    "rl_agent/splitfusion_edge_freshness_scheduler_v1/scheduler.py",
    "rl_agent/splitfusion_edge_freshness_scheduler_v1/pipeline.py",
    "rl_agent/splitfusion_edge_freshness_scheduler_v1/live_edge_service.py",
    "rl_agent/splitfusion_edge_freshness_scheduler_v1/edge_service_25ms.py",
    "rl_agent/splitfusion_edge_freshness_scheduler_v1/edge_service_v2_predicted.py",
    "rl_agent/splitfusion_edge_freshness_scheduler_v1/live_capture.py",
    "rl_agent/splitfusion_edge_freshness_scheduler_v1/live_cell_child.py",
    "rl_agent/splitfusion_edge_freshness_scheduler_v1/live_runner.py",
    "rl_agent/splitfusion_edge_freshness_scheduler_v1/live_payoff_runner.py",
    "rl_agent/splitfusion_edge_optimization_v1/optimized_tail.py",
    "rl_agent/splitfusion_edge_optimization_v1/optimized_tail_v2.py",
    "rl_agent/splitfusion_edge_optimization_v1/detached_tail.py",
    "rl_agent/splitfusion_edge_optimization_v1/detached_tail_v2.py",
    "rl_agent/splitfusion_edge_optimization_v1/detached_runtime.py",
    "rl_agent/splitfusion_edge_optimization_v1/detached_edge_preload.py",
    "rl_agent/splitfusion_edge_optimization_v1/detached_edge_preload_v2.py",
    "experiments/splitfusion_edge_freshness_scheduler_v1/"
    "20260910_predicted_horizon_288_v2/comparison.json",
)


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite timing value: {value!r}")
    return result


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _manifest_builder(
    variant: Mapping[str, Any], policy: CandidatePolicy, edge_module: str
):
    def build(**kwargs: Any) -> dict[str, Any]:
        document = dict(freshness._build_manifest(**kwargs))
        document.pop("run_manifest_sha256", None)
        document["schema"] = "scenesense.splitfusion_live_payoff_manifest.v1"
        document["execution_token"] = EXECUTE_TOKEN
        document["objective"] = (
            "Matched live CARLA/OAI measurement of v2 edge-tail payoff, causal "
            "predicted-install scheduling, map utility and exact end-to-end timing."
        )
        document["comparison_variant"] = {
            "id": str(variant["id"]),
            "tail": str(variant["tail"]),
            "scheduler_policy": policy.value,
            "edge_module": edge_module,
        }
        document["freshness_scheduler"] = {
            "policy": policy.value,
            "queue_wait_budget_ms": (
                None
                if policy.queue_wait_budget_ns is None
                else policy.queue_wait_budget_ns / 1e6
            ),
            "processing_horizon_ms": 500.0,
            "active_cuda_preemption": False,
            "compute_workers": 1,
            "publication_workers": 1,
            "compute_pending_depth": 1,
            "publication_pending_depth": 1,
            "predicted_policy_is_causal": (
                policy is CandidatePolicy.PREDICTED_INSTALL_HORIZON
            ),
            "sources": {
                path: common.sha256_file(common.repo_path(path)) for path in SOURCES
            },
        }
        return common.seal(document, "run_manifest_sha256")

    return build


def _configure(variant: Mapping[str, Any]) -> CandidatePolicy:
    policy = variant["policy"]
    if not isinstance(policy, CandidatePolicy):
        raise RuntimeError("comparison variant has an invalid policy")
    edge_module = str(variant["edge_module"])
    baseline.runner.EDGE_MODULE = edge_module
    baseline.CHILD_MODULE = CHILD_MODULE
    baseline.LIVE_EXECUTE_TOKEN = EXECUTE_TOKEN
    baseline.LIVE_SCHEMA = "scenesense.splitfusion_live_payoff_cell.v1"
    baseline.LIVE_MANIFEST_SCHEMA = "scenesense.splitfusion_live_payoff_manifest.v1"
    baseline.TERMINAL_SUCCESS = "SPLITFUSION_LIVE_PAYOFF_CELL_COMPLETE"
    baseline.TERMINAL_FAILURE = "SPLITFUSION_LIVE_PAYOFF_CELL_FAILED"
    baseline.build_live_manifest = _manifest_builder(variant, policy, edge_module)
    baseline.summarize_live_action = freshness._summary
    baseline.write_live_report = freshness._report_writer
    return policy


def _installed_breakdown(
    *,
    variant_id: str,
    action_id: int,
    summary: Mapping[str, Any],
    per_frame_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _read_rows(per_frame_path)
    row_by_frame = {int(row["frame_id"]): row for row in rows}
    common.require(
        len(row_by_frame) == len(rows), "per-frame timing rows contain duplicates"
    )
    installs = {
        int(row["frame_id"]): row
        for row in summary.get("map_install_records", [])
    }
    common.require(
        len(installs) == int(summary["map_utility"]["ack_installed_frames"]),
        "per-frame install timing inventory does not match ACK-installed count",
    )
    detailed: list[dict[str, Any]] = []
    maximum_reconciliation_error_ms = 0.0
    for frame_id, install in sorted(installs.items()):
        common.require(frame_id in row_by_frame, "installed frame lacks timing row")
        source = row_by_frame[frame_id]
        capture_ns = int(round(float(install["capture_wall_s"]) * 1e9))
        first_send_ns = int(source["ue_first_send_wall_ns"])
        reassembled_ns = int(source["edge_complete_reassembly_wall_ns"])
        worker_ns = int(source["edge_worker_start_wall_ns"])
        result_ns = int(source["edge_result_published_wall_ns"])
        install_ns = int(round(float(install["install_timestamp_s"]) * 1e9))
        boundaries = (
            capture_ns,
            first_send_ns,
            reassembled_ns,
            worker_ns,
            result_ns,
            install_ns,
        )
        common.require(
            all(right >= left for left, right in zip(boundaries, boundaries[1:])),
            "end-to-end timing boundaries are not monotonic",
        )
        capture_to_send = (first_send_ns - capture_ns) / 1e6
        feature_uplink = (reassembled_ns - first_send_ns) / 1e6
        edge_queue = (worker_ns - reassembled_ns) / 1e6
        edge_service = (result_ns - worker_ns) / 1e6
        result_to_install = (install_ns - result_ns) / 1e6
        install_aoi = (install_ns - capture_ns) / 1e6
        component_sum = (
            capture_to_send
            + feature_uplink
            + edge_queue
            + edge_service
            + result_to_install
        )
        error = abs(component_sum - install_aoi)
        maximum_reconciliation_error_ms = max(
            maximum_reconciliation_error_ms, error
        )
        ue_front = _float(source.get("ue_front_ms"))
        preparation_to_send = (
            None if ue_front is None else capture_to_send - ue_front
        )
        if preparation_to_send is not None:
            common.require(
                preparation_to_send >= -0.05,
                "reported UE front exceeds capture-to-send span",
            )
        detailed.append(
            {
                "variant": variant_id,
                "action_id": action_id,
                "frame_id": frame_id,
                "payload_bytes": int(float(source["payload_bytes"])),
                "capture_to_first_send_ms": capture_to_send,
                "preparation_to_front_residual_ms": preparation_to_send,
                "ue_front_dispatch_ms": ue_front,
                "feature_uplink_ms": feature_uplink,
                "edge_queue_wait_ms": edge_queue,
                "edge_service_to_result_ms": edge_service,
                "result_to_map_install_ms": result_to_install,
                "install_aoi_ms": install_aoi,
                "component_sum_ms": component_sum,
                "reconciliation_error_ms": error,
                "decode_tail_cuda_ms": _float(source.get("decode_tail_cuda_ms")),
                "tail_inference_block_ms": _float(
                    source.get("decode_tail_inference_block_ms")
                ),
                "camera_aware_postprocess_ms": _float(
                    source.get("tail_camera_aware_postprocess_ms")
                ),
                "p025_service_filter_ms": _float(
                    source.get("tail_p025_service_filter_ms")
                ),
                "segmentation_upsample_argmax_ms": _float(
                    source.get("tail_segmentation_upsample_argmax_ms")
                ),
                "compact_output_serialization_ms": _float(
                    source.get("tail_output_serialization_ms")
                ),
                "detections": int(float(source["detection_count"])),
            }
        )
    common.require(
        maximum_reconciliation_error_ms <= 0.001,
        "end-to-end component reconciliation exceeds one microsecond",
    )
    metric_names = tuple(
        name
        for name in detailed[0]
        if name.endswith("_ms") and name not in {"reconciliation_error_ms"}
    ) if detailed else ()
    timing = {
        name: common.summarize(
            [float(row[name]) for row in detailed if row.get(name) is not None]
        )
        for name in metric_names
    }
    return detailed, {
        "installed_frames": len(detailed),
        "maximum_reconciliation_error_ms": maximum_reconciliation_error_ms,
        "timing": timing,
    }


def _cell_row(
    *,
    variant: Mapping[str, Any],
    summary: Mapping[str, Any],
    breakdown: Mapping[str, Any],
) -> dict[str, Any]:
    utility = summary["map_utility"]
    scheduler = summary["freshness_scheduler"]
    pipeline = summary["edge_pipeline"]
    timing = breakdown["timing"]
    source_timing = summary["timing"]
    result: dict[str, Any] = {
        "variant": str(variant["id"]),
        "tail": str(variant["tail"]),
        "scheduler_policy": str(variant["policy"].value),
        "action_id": int(summary["action_id"]),
        "profile_id": str(summary["profile_id"]),
        "payload_bytes_median": source_timing["payload_bytes"]["median"],
        "sent_frames": utility["sent_frames"],
        "ack_installed_frames": utility["ack_installed_frames"],
        "useful_newer_map_installations": utility[
            "useful_newer_map_installations"
        ],
        "installed_within_100ms": utility["installed_within_100ms"],
        "installed_within_500ms": utility["installed_within_500ms"],
        "time_weighted_map_aoi_ms": utility[
            "time_weighted_map_aoi_ms_after_first_install"
        ],
        "superseded_pending": scheduler["terminal_reason_counts"].get(
            "SUPERSEDED_PENDING", 0
        ),
        "predicted_horizon_drops": scheduler["terminal_reason_counts"].get(
            "PREDICTED_MAP_INSTALL_HORIZON_EXCEEDED", 0
        ),
        "fixed_queue_budget_drops": scheduler["terminal_reason_counts"].get(
            "QUEUE_WAIT_BUDGET_EXCEEDED", 0
        ),
        "prediction_admission_evaluations": pipeline.get(
            "prediction_admission_evaluations", 0
        ),
        "prediction_admission_rejections": pipeline.get(
            "prediction_admission_rejections", 0
        ),
    }
    for name, values in timing.items():
        result[f"{name}_median"] = values["median"]
        result[f"{name}_p95"] = values["p95"]
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    common.require(bool(rows), f"cannot write empty table: {path.name}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    return common.sha256_file(path)


def _report(root: Path, rows: Sequence[Mapping[str, Any]], runtime_s: float) -> None:
    lines = [
        "# SplitFusion live v1/fixed-25 versus v2/predicted payoff",
        "",
        "Each cell used 300 transmitted frames from live CARLA Route B and a "
        "fresh `FAVORABLE_STABLE` OAI/RFsim lifecycle. Map utility is credited "
        "only by the authoritative `ACK_INSTALLED` path.",
        "",
        "The end-to-end decomposition uses same-host wall-clock boundaries and "
        "reconciles per installed frame: capture→first send + feature uplink + "
        "edge queue + edge service/result send + result→map install = install AoI.",
        "",
        "| action | variant | sent | installed | useful | ≤100 ms | ≤500 ms | "
        "queue med | edge service med | install AoI med | time-weighted AoI |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def show(name: str) -> str:
            value = row.get(name)
            return "—" if value in (None, "") else f"{float(value):.2f}"

        lines.append(
            f"| {row['action_id']} | `{row['variant']}` | {row['sent_frames']} | "
            f"{row['ack_installed_frames']} | "
            f"{row['useful_newer_map_installations']} | "
            f"{row['installed_within_100ms']} | {row['installed_within_500ms']} | "
            f"{show('edge_queue_wait_ms_median')} | "
            f"{show('edge_service_to_result_ms_median')} | "
            f"{show('install_aoi_ms_median')} | "
            f"{show('time_weighted_map_aoi_ms')} |"
        )
    lines.extend(
        [
            "",
            f"Total wall time: {runtime_s / 60.0:.1f} minutes.",
            "",
            "The predicted policy uses only registered initial estimates and "
            "causal EWMA updates from previously completed frames. It never "
            "interrupts a running CUDA kernel and never reads a current frame's "
            "future realized service time.",
            "",
        ]
    )
    common.atomic_create_text(root / "REPORT.md", "\n".join(lines))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", required=True)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args(list(argv) if argv is not None else None)
    common.require(args.execute == EXECUTE_TOKEN, "execution token mismatch")
    root = (common.ROOT / args.output).resolve()
    common.require(not root.exists(), f"create-only output already exists: {root}")
    started = time.time()
    cells: list[dict[str, Any]] = []
    detailed: list[dict[str, Any]] = []
    subruns: list[dict[str, Any]] = []
    for action_id in ACTIONS:
        for variant in VARIANTS:
            policy = _configure(variant)
            leaf = root / f"action{action_id:02d}__{str(variant['id']).lower()}"
            relative = str(leaf.relative_to(common.ROOT))
            rc = baseline.main(
                [
                    "--execute",
                    EXECUTE_TOKEN,
                    "--output",
                    relative,
                    "--actions",
                    str(action_id),
                ]
            )
            common.require(
                rc == 0,
                f"live payoff cell failed: action={action_id} "
                f"variant={variant['id']} rc={rc}",
            )
            result_path = leaf / "LIVE_DIAGNOSTIC_RESULTS.json"
            result = common.load_json(result_path)
            summary = result["action_summaries"][0]
            frame_path = next((leaf / "per_frame").glob("*.csv"))
            frame_rows, breakdown = _installed_breakdown(
                variant_id=str(variant["id"]),
                action_id=action_id,
                summary=summary,
                per_frame_path=frame_path,
            )
            detailed.extend(frame_rows)
            cells.append(
                _cell_row(variant=variant, summary=summary, breakdown=breakdown)
            )
            subruns.append(
                {
                    "action_id": action_id,
                    "variant": str(variant["id"]),
                    "policy": policy.value,
                    "path": str(leaf.relative_to(root)),
                    "results_sha256": common.sha256_file(result_path),
                    "run_manifest_sha256": result["run_manifest_sha256"],
                    "maximum_timing_reconciliation_error_ms": breakdown[
                        "maximum_reconciliation_error_ms"
                    ],
                }
            )
    runtime_s = time.time() - started
    result_document = {
        "schema": "scenesense.splitfusion_live_payoff_comparison.v1",
        "status": "COMPLETE",
        "started_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": runtime_s,
        "actions": list(ACTIONS),
        "variants": [
            {
                "id": str(row["id"]),
                "tail": str(row["tail"]),
                "policy": row["policy"].value,
                "edge_module": str(row["edge_module"]),
            }
            for row in VARIANTS
        ],
        "cells": cells,
        "subruns": subruns,
        "timing_semantics": {
            "sensor_preparation_included_in_install_aoi": True,
            "feature_uplink_starts_at_first_udp_datagram": True,
            "feature_uplink_ends_at_complete_edge_reassembly": True,
            "edge_queue_ends_at_compute_worker_start": True,
            "edge_service_ends_after_compact_result_send": True,
            "result_to_map_install_is_separately_measured": True,
            "per_installed_frame_components_reconcile_exactly": True,
        },
        "not_claimed": [
            "100_MS_SERVICE_READY",
            "PHY_ONLY_UPLINK_LATENCY",
            "ROUTE_B_COMPLETION",
            "RUN_TO_RUN_VARIANCE_CHARACTERIZED",
        ],
    }
    common.atomic_create_json(root / "LIVE_PAYOFF_RESULTS.json", result_document)
    _write_csv(root / "cell_payoff_summary.csv", cells)
    _write_csv(root / "installed_frame_latency_breakdown.csv", detailed)
    _report(root, cells, runtime_s)
    baseline.runner.write_artifact_manifest(root)
    common.atomic_create_json(
        root / SUCCESS_TERMINAL,
        {
            "terminal": SUCCESS_TERMINAL,
            "cells": len(cells),
            "installed_frame_records": len(detailed),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(SUCCESS_TERMINAL, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
