#!/usr/bin/env python3
"""Four short live cells for freshness-first edge scheduling qualification."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from rl_agent.splitfusion_timing_diagnostic_v1 import diagnostic_common as common
from rl_agent.splitfusion_timing_diagnostic_v1 import live_runner as baseline

from .pipeline import CandidatePolicy


EXECUTE_TOKEN = "SPLITFUSION_FRESHNESS_SCHEDULER_FOUR_CELL_LIVE_V1"
SUCCESS_TERMINAL = "SPLITFUSION_FRESHNESS_SCHEDULER_FOUR_CELL_LIVE_COMPLETE"
DEFAULT_OUTPUT = (
    "experiments/splitfusion_edge_freshness_scheduler_v1/"
    "20260910_live_actions50_71_two_policies"
)
CHILD_MODULE = (
    "rl_agent.splitfusion_edge_freshness_scheduler_v1.live_cell_child"
)
EDGE_MODULES = {
    CandidatePolicy.LATEST_ONLY_NO_EXPIRY: (
        "rl_agent.splitfusion_edge_freshness_scheduler_v1.edge_service_no_expiry"
    ),
    CandidatePolicy.LATEST_ONLY_25_MS: (
        "rl_agent.splitfusion_edge_freshness_scheduler_v1.edge_service_25ms"
    ),
}
SOURCES = (
    "rl_agent/splitfusion_edge_freshness_scheduler_v1/scheduler.py",
    "rl_agent/splitfusion_edge_freshness_scheduler_v1/pipeline.py",
    "rl_agent/splitfusion_edge_freshness_scheduler_v1/live_edge_service.py",
    "rl_agent/splitfusion_edge_freshness_scheduler_v1/edge_service_no_expiry.py",
    "rl_agent/splitfusion_edge_freshness_scheduler_v1/edge_service_25ms.py",
    "rl_agent/splitfusion_edge_freshness_scheduler_v1/live_capture.py",
    "rl_agent/splitfusion_edge_freshness_scheduler_v1/live_cell_child.py",
    "rl_agent/splitfusion_edge_freshness_scheduler_v1/live_runner.py",
    "rl_agent/splitfusion_edge_freshness_scheduler_v1/qualify_live_candidate.py",
    "rl_agent/splitfusion_edge_optimization_v1/optimized_tail.py",
    "rl_agent/splitfusion_edge_optimization_v1/detached_tail.py",
    "rl_agent/splitfusion_edge_optimization_v1/detached_runtime.py",
    "rl_agent/splitfusion_edge_optimization_v1/detached_edge_preload.py",
)


_build_manifest = baseline.build_live_manifest
_summarize = baseline.summarize_live_action
_write_report = baseline.write_live_report


def _manifest_builder(policy: CandidatePolicy, edge_module: str):
    def build(**kwargs: Any) -> dict[str, Any]:
        document = dict(_build_manifest(**kwargs))
        document.pop("run_manifest_sha256", None)
        document["schema"] = "scenesense.splitfusion_freshness_live_manifest.v1"
        document["execution_token"] = EXECUTE_TOKEN
        document["objective"] = (
            "Measure latest-only edge scheduling with explicit intentional "
            "supersession feedback and authoritative map-install ACKs."
        )
        document["freshness_scheduler"] = {
            "policy": policy.value,
            "queue_wait_budget_ms": policy.queue_wait_budget_ns / 1e6
            if policy.queue_wait_budget_ns is not None
            else None,
            "queue_wait_budget_semantics": (
                "EXPIRY_CEILING_NEVER_AN_INTENTIONAL_HOLD"
            ),
            "active_cuda_preemption": False,
            "compute_workers": 1,
            "publication_workers": 1,
            "compute_pending_depth": 1,
            "publication_pending_depth": 1,
            "edge_module": edge_module,
            "child_module": CHILD_MODULE,
            "sources": {
                path: common.sha256_file(common.repo_path(path))
                for path in SOURCES
            },
        }
        return common.seal(document, "run_manifest_sha256")

    return build


def _summary(report: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(_summarize(report))
    capture = report.get("capture", {})
    edge = report.get("edge_summary", {})
    value["freshness_scheduler"] = dict(
        capture.get("freshness_scheduler") or {}
    )
    value["map_utility"] = dict(capture.get("map_utility") or {})
    value["edge_pipeline"] = dict(edge.get("pipeline") or {})
    value["edge_pipeline_terminal_reason_counts"] = dict(
        edge.get("pipeline_terminal_reason_counts") or {}
    )
    value["edge_pipeline_outcome_class_counts"] = dict(
        edge.get("pipeline_outcome_class_counts") or {}
    )
    return value


def _report_writer(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    summaries: Sequence[Mapping[str, Any]],
    comparisons: Mapping[str, Any],
    runtime_seconds: float,
) -> str:
    _write_report(
        path,
        manifest=manifest,
        summaries=summaries,
        comparisons=comparisons,
        runtime_seconds=runtime_seconds,
    )
    policy = str(manifest["freshness_scheduler"]["policy"])
    summary = summaries[0]
    utility = summary.get("map_utility", {})
    scheduler = summary.get("freshness_scheduler", {})
    lines = [
        "",
        "## Freshness scheduling outcome",
        "",
        f"Policy: `{policy}`. The 25 ms value, when present, is an expiry "
        "ceiling checked only when a worker becomes available; it is never a hold.",
        "",
        f"- Sent frames: {utility.get('sent_frames')}",
        f"- Authoritative ACK-installed frames: {utility.get('ack_installed_frames')}",
        f"- Useful newer-map installations: {utility.get('useful_newer_map_installations')}",
        f"- Intentional non-install terminals: {utility.get('intentional_non_install_terminals')}",
        f"- True timeouts after excluding explicit scheduler outcomes: "
        f"{utility.get('true_timeout_without_scheduler_or_install')}",
        f"- Time-weighted map AoI after first install: "
        f"{utility.get('time_weighted_map_aoi_ms_after_first_install')} ms",
        f"- Scheduler reasons: `{json.dumps(scheduler.get('terminal_reason_counts', {}), sort_keys=True)}`",
        "",
        "`RESULT_PUBLISHED` is an edge-publication event only. It is not map "
        "utility credit; only the existing `ACK_INSTALLED` path proves installation.",
        "",
    ]
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        path.read_text(encoding="utf-8").rstrip() + "\n" + "\n".join(lines),
        encoding="utf-8",
    )
    temporary.replace(path)
    return common.sha256_file(path)


def _configure(policy: CandidatePolicy) -> None:
    edge_module = EDGE_MODULES[policy]
    baseline.runner.EDGE_MODULE = edge_module
    baseline.CHILD_MODULE = CHILD_MODULE
    baseline.LIVE_EXECUTE_TOKEN = EXECUTE_TOKEN
    baseline.LIVE_SCHEMA = "scenesense.splitfusion_freshness_live.v1"
    baseline.LIVE_MANIFEST_SCHEMA = (
        "scenesense.splitfusion_freshness_live_manifest.v1"
    )
    baseline.TERMINAL_SUCCESS = (
        "SPLITFUSION_FRESHNESS_SINGLE_CELL_LIVE_COMPLETE"
    )
    baseline.TERMINAL_FAILURE = (
        "SPLITFUSION_FRESHNESS_SINGLE_CELL_LIVE_FAILED"
    )
    baseline.build_live_manifest = _manifest_builder(policy, edge_module)
    baseline.summarize_live_action = _summary
    baseline.write_live_report = _report_writer


def _flatten_cell(
    *, policy: CandidatePolicy, result: Mapping[str, Any]
) -> dict[str, Any]:
    summary = result["action_summaries"][0]
    utility = summary.get("map_utility", {})
    scheduler = summary.get("freshness_scheduler", {})
    timing = summary.get("timing", {})
    return {
        "action_id": int(summary["action_id"]),
        "profile_id": str(summary["profile_id"]),
        "network_profile_id": str(summary["network_profile_id"]),
        "scheduler_policy": policy.value,
        "sent_frames": utility.get("sent_frames"),
        "ack_installed_frames": utility.get("ack_installed_frames"),
        "useful_newer_map_installations": utility.get(
            "useful_newer_map_installations"
        ),
        "intentional_non_install_terminals": utility.get(
            "intentional_non_install_terminals"
        ),
        "true_timeout_without_scheduler_or_install": utility.get(
            "true_timeout_without_scheduler_or_install"
        ),
        "map_nack_without_scheduler": utility.get("map_nack_without_scheduler"),
        "installed_within_100ms": utility.get("installed_within_100ms"),
        "installed_within_500ms": utility.get("installed_within_500ms"),
        "install_aoi_ms_median": utility.get("install_aoi_ms_median"),
        "install_aoi_ms_p95": utility.get("install_aoi_ms_p95"),
        "time_weighted_map_aoi_ms": utility.get(
            "time_weighted_map_aoi_ms_after_first_install"
        ),
        "superseded_pending": scheduler.get("terminal_reason_counts", {}).get(
            "SUPERSEDED_PENDING", 0
        ),
        "queue_wait_expired": scheduler.get("terminal_reason_counts", {}).get(
            "QUEUE_WAIT_BUDGET_EXCEEDED", 0
        ),
        "wasted_feature_bytes": scheduler.get("wasted_feature_bytes"),
        "edge_queue_wait_ms_median": (
            timing.get("edge_queue_wait_ms") or {}
        ).get("median"),
        "deployed_tail_service_ms_median": (
            timing.get("deployed_tail_service_ms") or {}
        ).get("median"),
        "application_feature_uplink_ms_median": (
            timing.get("application_feature_uplink_ms") or {}
        ).get("median"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", required=True)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.execute != EXECUTE_TOKEN:
        raise RuntimeError("freshness live execution token mismatch")
    root = (common.ROOT / args.output).resolve()
    if root.exists():
        raise RuntimeError(f"create-only output already exists: {root}")
    started = time.time()
    cells: list[dict[str, Any]] = []
    subruns: list[dict[str, Any]] = []
    for action_id in (50, 71):
        for policy in (
            CandidatePolicy.LATEST_ONLY_NO_EXPIRY,
            CandidatePolicy.LATEST_ONLY_25_MS,
        ):
            _configure(policy)
            leaf = (
                root
                / f"action{action_id:02d}__{policy.value.lower()}"
            )
            relative = str(leaf.relative_to(common.ROOT))
            result_code = baseline.main(
                [
                    "--execute",
                    EXECUTE_TOKEN,
                    "--output",
                    relative,
                    "--actions",
                    str(action_id),
                ]
            )
            if result_code != 0:
                raise RuntimeError(
                    f"live scheduler cell failed: action={action_id} "
                    f"policy={policy.value} rc={result_code}"
                )
            result_path = leaf / "LIVE_DIAGNOSTIC_RESULTS.json"
            result = common.load_json(result_path)
            cells.append(_flatten_cell(policy=policy, result=result))
            subruns.append(
                {
                    "action_id": action_id,
                    "policy": policy.value,
                    "path": str(leaf.relative_to(root)),
                    "results_sha256": common.sha256_file(result_path),
                    "run_manifest_sha256": result["run_manifest_sha256"],
                }
            )

    aggregate = {
        "schema": "scenesense.splitfusion_freshness_four_cell_live.v1",
        "status": "COMPLETE",
        "started_utc": datetime.fromtimestamp(
            started, tz=timezone.utc
        ).isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.time() - started,
        "cells": cells,
        "subruns": subruns,
        "selection_rule": (
            "Prefer lower time-weighted map AoI and more useful newer-map "
            "installations; treat intentional supersession separately from "
            "transport/structural failure while retaining byte and compute cost."
        ),
        "not_claimed": [
            "100_MS_SERVICE_READY",
            "POLICY_SELECTED_BEFORE_COMPARATIVE_RESULTS",
            "MULTI_WORKER_GPU_SCALING",
        ],
    }
    common.atomic_create_json(root / "LIVE_FRESHNESS_SCHEDULER_RESULTS.json", aggregate)
    with (root / "action_policy_summary.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    report_lines = [
        "# SplitFusion freshness-first edge scheduling — four live cells",
        "",
        "Each cell uses fresh live CARLA/OAI lifecycle and 300 transmitted "
        "frames under `FAVORABLE_STABLE`. Actual map installation is proven "
        "only by the existing UE-side `ACK_INSTALLED` feedback.",
        "",
        "| action | policy | sent | installed | useful installs | intentional drops | true timeouts | median AoI | time-weighted AoI |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cells:
        report_lines.append(
            "| {action_id} | {scheduler_policy} | {sent_frames} | "
            "{ack_installed_frames} | {useful_newer_map_installations} | "
            "{intentional_non_install_terminals} | "
            "{true_timeout_without_scheduler_or_install} | "
            "{install_aoi_ms_median} | {time_weighted_map_aoi_ms} |".format(
                **row
            )
        )
    report_lines.extend(
        [
            "",
            "The 25 ms candidate is a pre-compute expiry ceiling, never an "
            "intentional wait. Running CUDA work is never interrupted; a "
            "single newest pending frame replaces older pending work.",
            "",
        ]
    )
    common.atomic_create_text(root / "REPORT.md", "\n".join(report_lines))
    baseline.runner.write_artifact_manifest(root)
    common.atomic_create_json(
        root / SUCCESS_TERMINAL,
        {
            "terminal": SUCCESS_TERMINAL,
            "cells": len(cells),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(SUCCESS_TERMINAL, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
