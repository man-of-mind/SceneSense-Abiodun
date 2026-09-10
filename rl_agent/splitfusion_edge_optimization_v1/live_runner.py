#!/usr/bin/env python3
"""Four-action live CARLA/OAI before/after qualification for edge optimization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from rl_agent.splitfusion_timing_diagnostic_v1 import diagnostic_common as common
from rl_agent.splitfusion_timing_diagnostic_v1 import live_runner as baseline_runner


EXECUTE_TOKEN = "SPLITFUSION_EDGE_OPTIMIZATION_FOUR_ACTION_LIVE_V1"
SUCCESS_TERMINAL = "SPLITFUSION_EDGE_OPTIMIZATION_FOUR_ACTION_LIVE_COMPLETE"
FAILURE_TERMINAL = "SPLITFUSION_EDGE_OPTIMIZATION_FOUR_ACTION_LIVE_FAILED"
EDGE_MODULE = "rl_agent.splitfusion_edge_optimization_v1.edge_service"
DEFAULT_OUTPUT = (
    "experiments/splitfusion_edge_optimization_v1/"
    "20260909_live_actions30_15_50_71"
)
BASELINE_RESULTS_RELPATH = (
    "experiments/splitfusion_timing_diagnostic_v1/"
    "20260909_live_carla_actions30_15_50_71_retry3/"
    "LIVE_DIAGNOSTIC_RESULTS.json"
)
OPTIMIZATION_SOURCES = (
    "rl_agent/splitfusion_edge_optimization_v1/optimized_tail.py",
    "rl_agent/splitfusion_edge_optimization_v1/edge_preload.py",
    "rl_agent/splitfusion_edge_optimization_v1/edge_service.py",
    "rl_agent/splitfusion_edge_optimization_v1/live_runner.py",
)


_original_build_manifest = baseline_runner.build_live_manifest
_original_write_report = baseline_runner.write_live_report


def _baseline_results() -> dict[str, Any]:
    path = common.repo_path(BASELINE_RESULTS_RELPATH)
    return json.loads(path.read_text(encoding="utf-8"))


def _optimized_manifest(**kwargs: Any) -> dict[str, Any]:
    document = dict(_original_build_manifest(**kwargs))
    document.pop("run_manifest_sha256", None)
    document["schema"] = "scenesense.splitfusion_edge_optimization_live_manifest.v1"
    document["execution_token"] = EXECUTE_TOKEN
    document["objective"] = (
        "Measure the output-preserving edge optimization against the immutable "
        "four-action live CARLA/OAI baseline, retaining the original feature "
        "transport and full per-stage timing decomposition."
    )
    document["scope"] = {
        **dict(document["scope"]),
        "optimization_attempted": True,
        "optimization_changes_feature_transport": False,
        "baseline_actions": [30, 15, 50, 71],
    }
    baseline_path = common.repo_path(BASELINE_RESULTS_RELPATH)
    document["optimization_candidate"] = {
        "schema": "scenesense.splitfusion_edge_optimization_candidate.v1",
        "geometry_policy": "DECODE_GEOMETRY_FOR_ORDERED_POST_NMS_SURVIVORS",
        "world_coordinate_exactness": (
            "PRESERVE_ORIGINAL_PER_LEVEL_MATRIX_BATCH_SHAPE"
        ),
        "p025_policy": "FROZEN_GRID27_AND_PERSON_P025_EXACT",
        "serialization": "BATCHED_TENSOR_CPU_TRANSFER_BEFORE_FROZEN_ROW_BUILDER",
        "required_parity": (
            "BIT_IDENTICAL_TENSORS_INDICES_SEGMENTATION_AND_SERVICE_BYTES"
        ),
        "edge_module": EDGE_MODULE,
        "baseline_results": BASELINE_RESULTS_RELPATH,
        "baseline_results_sha256": common.sha256_file(baseline_path),
        "sources": {
            path: common.sha256_file(common.repo_path(path))
            for path in OPTIMIZATION_SOURCES
        },
    }
    document["instrumentation"] = {
        **dict(document["instrumentation"]),
        "approach": (
            "the established live timing harness with a separate optimized "
            "edge preload; the frozen production model, action registry, wire "
            "format, UE path and radio path remain untouched"
        ),
        "optimized_edge_module": EDGE_MODULE,
    }
    return common.seal(document, "run_manifest_sha256")


def _metric(summary: Mapping[str, Any], name: str) -> float:
    return float(summary["timing"][name]["median"])


def _optimized_report(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    summaries: Sequence[Mapping[str, Any]],
    comparisons: Mapping[str, Any],
    runtime_seconds: float,
) -> str:
    _original_write_report(
        path,
        manifest=manifest,
        summaries=summaries,
        comparisons=comparisons,
        runtime_seconds=runtime_seconds,
    )
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "# SplitFusion live CARLA/OAI timing diagnostic — measured decomposition",
        "# SplitFusion edge optimization — four-action live before/after",
        1,
    ).replace(
        "**No optimization was attempted.**",
        "The output-preserving edge optimization candidate was exercised.",
    )
    baseline = {
        int(item["action_id"]): item
        for item in _baseline_results()["action_summaries"]
    }
    lines = [
        "",
        "## Before/after edge optimization",
        "",
        "The baseline and candidate use the same four actions, live Route-B "
        "contract and FAVORABLE_STABLE process. Scenes are separate live "
        "realizations, so timing distributions—not individual frames—are "
        "compared. Every cell separately proves exact output parity during "
        "warm-up before accepting measured traffic.",
        "",
        "| action | deployed service before | after | saving | postprocess before | after | p025 before | after | serialization before | after | uplink before | after |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        action_id = int(summary["action_id"])
        prior = baseline[action_id]
        before_service = _metric(prior, "deployed_tail_service_ms")
        after_service = _metric(summary, "deployed_tail_service_ms")
        values = (
            action_id,
            before_service,
            after_service,
            before_service - after_service,
            _metric(prior, "tail_camera_aware_postprocess_ms"),
            _metric(summary, "tail_camera_aware_postprocess_ms"),
            _metric(prior, "tail_p025_service_filter_ms"),
            _metric(summary, "tail_p025_service_filter_ms"),
            _metric(prior, "tail_output_serialization_ms"),
            _metric(summary, "tail_output_serialization_ms"),
            _metric(prior, "application_feature_uplink_ms"),
            _metric(summary, "application_feature_uplink_ms"),
        )
        lines.append(
            "| %d | %.2f ms | %.2f ms | %.2f ms | %.2f ms | %.2f ms | "
            "%.2f ms | %.2f ms | %.2f ms | %.2f ms | %.2f ms | %.2f ms |"
            % values
        )
    lines.extend(
        (
            "",
            "Feature-uplink timing is reported to verify transport stability; "
            "it is not part of the optimized code. End-to-end AoI and queue "
            "effects are nonlinear and are not inferred by subtracting the "
            "service saving from the 288-cell measurements.",
            "",
        )
    )
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(text.rstrip() + "\n" + "\n".join(lines), encoding="utf-8")
    temporary.replace(path)
    return common.sha256_file(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", required=True)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--actions", default="30,15,50,71")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.execute != EXECUTE_TOKEN:
        raise RuntimeError("edge-optimization execution token mismatch")

    baseline_runner.runner.EDGE_MODULE = EDGE_MODULE
    baseline_runner.LIVE_EXECUTE_TOKEN = EXECUTE_TOKEN
    baseline_runner.LIVE_SCHEMA = "scenesense.splitfusion_edge_optimization_live.v1"
    baseline_runner.LIVE_MANIFEST_SCHEMA = (
        "scenesense.splitfusion_edge_optimization_live_manifest.v1"
    )
    baseline_runner.TERMINAL_SUCCESS = SUCCESS_TERMINAL
    baseline_runner.TERMINAL_FAILURE = FAILURE_TERMINAL
    baseline_runner.build_live_manifest = _optimized_manifest
    baseline_runner.write_live_report = _optimized_report
    return baseline_runner.main(
        [
            "--execute",
            EXECUTE_TOKEN,
            "--output",
            args.output,
            "--actions",
            args.actions,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
