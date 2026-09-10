#!/usr/bin/env python3
"""Build corrected, compact evidence from the immutable four-cell live run."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from rl_agent.splitfusion_timing_diagnostic_v1 import diagnostic_common as common


SCHEMA = "scenesense.splitfusion_freshness_four_cell_analysis.v1"
TERMINAL = "SPLITFUSION_FRESHNESS_SCHEDULER_LIVE_ANALYSIS_COMPLETE"


def _verify_manifest(root: Path, manifest_path: Path) -> int:
    document = common.load_json(manifest_path)
    verified = 0
    for relative, expected in document["sha256"].items():
        path = root / relative
        if not path.is_file() or common.sha256_file(path) != str(expected):
            raise RuntimeError(f"artifact hash mismatch: {path}")
        verified += 1
    return verified


def _row(policy: str, result: Mapping[str, Any]) -> dict[str, Any]:
    summary = result["action_summaries"][0]
    utility = summary["map_utility"]
    scheduler = summary["freshness_scheduler"]
    pipeline = summary["edge_pipeline"]
    reasons = summary["edge_pipeline_terminal_reason_counts"]
    sent = int(utility["sent_frames"])
    published = int(reasons.get("RESULT_PUBLISHED", 0))
    explicit = int(scheduler["terminal_feedback_records"])
    if sent != published + explicit:
        raise RuntimeError("sent/publication/scheduler accounting mismatch")
    if int(pipeline["offered_frames"]) != sent or int(pipeline["terminal_frames"]) != sent:
        raise RuntimeError("pipeline terminal accounting mismatch")
    if int(utility["ack_installed_frames"]) > published:
        raise RuntimeError("map installs exceed published edge results")
    if int(utility["map_nack_without_scheduler"]) != 0:
        raise RuntimeError("unexpected non-scheduler map rejection")
    if pipeline.get("fatal_error") is not None:
        raise RuntimeError("pipeline reports a structural failure")
    if not bool(pipeline["stage_overlap_observed"]):
        raise RuntimeError("compute/publication stage overlap was not observed")
    timing = summary["timing"]
    return {
        "action_id": int(summary["action_id"]),
        "profile_id": str(summary["profile_id"]),
        "policy": str(policy),
        "sent": sent,
        "edge_results_published": published,
        "ack_installed": int(utility["ack_installed_frames"]),
        "useful_newer_map_installations": int(
            utility["useful_newer_map_installations"]
        ),
        "intentional_freshness_drops": int(
            scheduler["intentional_freshness_drop_frames"]
        ),
        "expired_work": int(scheduler["expired_work_frames"]),
        "explicit_scheduler_non_install_terminals": explicit,
        "true_timeout_without_scheduler_or_install": int(
            utility["true_timeout_without_scheduler_or_install"]
        ),
        "installed_within_100ms": int(utility["installed_within_100ms"]),
        "installed_within_500ms": int(utility["installed_within_500ms"]),
        "install_aoi_ms_median": float(utility["install_aoi_ms_median"]),
        "install_aoi_ms_p95": float(utility["install_aoi_ms_p95"]),
        "time_weighted_map_aoi_ms": float(
            utility["time_weighted_map_aoi_ms_after_first_install"]
        ),
        "application_feature_uplink_ms_median": float(
            timing["application_feature_uplink_ms"]["median"]
        ),
        "edge_queue_wait_ms_median": float(
            timing["edge_queue_wait_ms"]["median"]
        ),
        "deployed_tail_service_ms_median": float(
            timing["deployed_tail_service_ms"]["median"]
        ),
        "wasted_feature_bytes": int(scheduler["wasted_feature_bytes"]),
        "stage_overlap_observed": True,
        "compute_owner_thread_id": int(pipeline["compute_owner_thread_id"]),
        "publication_owner_thread_id": int(
            pipeline["publication_owner_thread_id"]
        ),
    }


def run(source: Path, output: Path) -> dict[str, Any]:
    source = source.resolve(strict=True)
    if output.exists():
        raise RuntimeError(f"create-only analysis output exists: {output}")
    aggregate_path = source / "LIVE_FRESHNESS_SCHEDULER_RESULTS.json"
    aggregate = common.load_json(aggregate_path)
    if aggregate.get("status") != "COMPLETE" or len(aggregate.get("subruns", [])) != 4:
        raise RuntimeError("source four-cell run is not complete")
    verified = _verify_manifest(source, source / "ARTIFACT_MANIFEST.json")
    rows: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    for item in aggregate["subruns"]:
        leaf = source / str(item["path"])
        result_path = leaf / "LIVE_DIAGNOSTIC_RESULTS.json"
        if common.sha256_file(result_path) != str(item["results_sha256"]):
            raise RuntimeError("subrun result binding mismatch")
        verified += _verify_manifest(leaf, leaf / "ARTIFACT_MANIFEST.json")
        result = common.load_json(result_path)
        rows.append(_row(str(item["policy"]), result))
        bindings.append(
            {
                **dict(item),
                "artifact_manifest_sha256": common.sha256_file(
                    leaf / "ARTIFACT_MANIFEST.json"
                ),
            }
        )
    inventory = {(row["action_id"], row["policy"]) for row in rows}
    expected = {
        (action, policy)
        for action in (50, 71)
        for policy in ("LATEST_ONLY_NO_EXPIRY", "LATEST_ONLY_25_MS")
    }
    if inventory != expected:
        raise RuntimeError("four-cell action/policy inventory mismatch")
    comparisons: list[dict[str, Any]] = []
    for action in (50, 71):
        by_policy = {row["policy"]: row for row in rows if row["action_id"] == action}
        base = by_policy["LATEST_ONLY_NO_EXPIRY"]
        expiry = by_policy["LATEST_ONLY_25_MS"]
        comparisons.append(
            {
                "action_id": action,
                "profile_id": base["profile_id"],
                "time_weighted_map_aoi_reduction_ms": (
                    base["time_weighted_map_aoi_ms"]
                    - expiry["time_weighted_map_aoi_ms"]
                ),
                "time_weighted_map_aoi_reduction_fraction": (
                    1.0
                    - expiry["time_weighted_map_aoi_ms"]
                    / base["time_weighted_map_aoi_ms"]
                ),
                "median_install_aoi_reduction_ms": (
                    base["install_aoi_ms_median"]
                    - expiry["install_aoi_ms_median"]
                ),
                "ack_installed_delta": expiry["ack_installed"] - base["ack_installed"],
                "useful_installation_delta": (
                    expiry["useful_newer_map_installations"]
                    - base["useful_newer_map_installations"]
                ),
                "expired_work_delta": expiry["expired_work"] - base["expired_work"],
                "wasted_feature_bytes_delta": (
                    expiry["wasted_feature_bytes"] - base["wasted_feature_bytes"]
                ),
            }
        )
    output.mkdir(parents=True, exist_ok=False)
    analysis = {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(source.relative_to(common.ROOT)),
            "aggregate_sha256": common.sha256_file(aggregate_path),
            "artifact_manifest_sha256": common.sha256_file(
                source / "ARTIFACT_MANIFEST.json"
            ),
            "verified_artifact_hashes": verified,
            "subruns": bindings,
        },
        "cells": rows,
        "comparisons": comparisons,
        "provisional_policy_selection": {
            "policy": "LATEST_ONLY_25_MS",
            "basis": (
                "lower time-weighted and median map AoI in both action-50 and "
                "action-71 cells under the preregistered freshness-first rule"
            ),
            "tradeoff": (
                "fewer completed installations and more expiry/wasted already-sent "
                "bytes, most strongly for action 50"
            ),
            "scope": (
                "provisional single-run live selection; not a claim of run-to-run "
                "variance or 100 ms service readiness"
            ),
        },
        "reporting_correction": {
            "source_field": "map_utility.intentional_non_install_terminals",
            "correct_meaning": "all explicit scheduler non-install terminals",
            "correct_replacement": "explicit_scheduler_non_install_terminals",
            "scientific_counters_changed": False,
            "live_cells_rerun": False,
        },
    }
    common.atomic_create_json(output / "analysis.json", analysis)
    with (output / "action_policy_summary.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = [
        "# Freshness-first edge scheduling: corrected live analysis",
        "",
        "The four immutable source cells completed with 300 transmitted frames "
        "each. Edge publication is distinct from authoritative map installation.",
        "",
        "| action | policy | installed | intentional supersession | expired | true timeout | median AoI | time-weighted AoI |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['action_id']} | {row['policy']} | {row['ack_installed']} | "
            f"{row['intentional_freshness_drops']} | {row['expired_work']} | "
            f"{row['true_timeout_without_scheduler_or_install']} | "
            f"{row['install_aoi_ms_median']:.1f} ms | "
            f"{row['time_weighted_map_aoi_ms']:.1f} ms |"
        )
    report.extend(
        [
            "",
            "The 25 ms expiry ceiling reduced time-weighted AoI by "
            f"{comparisons[0]['time_weighted_map_aoi_reduction_ms']:.1f} ms "
            f"({comparisons[0]['time_weighted_map_aoi_reduction_fraction']:.1%}) "
            "for action 50 and "
            f"{comparisons[1]['time_weighted_map_aoi_reduction_ms']:.1f} ms "
            f"({comparisons[1]['time_weighted_map_aoi_reduction_fraction']:.1%}) "
            "for action 71.",
            "",
            "It is selected provisionally because freshness is the primary map "
            "objective. The cost is fewer installations and more expired, already-"
            "transmitted work. No cell installed within 100 ms, so this is not a "
            "100 ms service-readiness result.",
            "",
            "Correction: the source aggregate label `intentional_non_install_terminals` "
            "included deadline expiry as well as intentional supersession. This "
            "analysis separates the unchanged raw counters; no live record was rewritten.",
            "",
        ]
    )
    common.atomic_create_text(output / "REPORT.md", "\n".join(report))
    hashes = {
        name: common.sha256_file(output / name)
        for name in ("analysis.json", "action_policy_summary.csv", "REPORT.md")
    }
    common.atomic_create_json(
        output / "artifact_manifest.json",
        {"schema": f"{SCHEMA}.artifacts", "sha256": hashes},
    )
    common.atomic_create_json(
        output / TERMINAL,
        {"terminal": TERMINAL, "artifact_manifest_sha256": common.sha256_file(output / "artifact_manifest.json")},
    )
    return analysis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    analysis = run(Path(args.source), Path(args.output))
    print(json.dumps(analysis["provisional_policy_selection"], indent=2, sort_keys=True))
    print(TERMINAL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
