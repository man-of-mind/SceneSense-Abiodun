#!/usr/bin/env python3
"""Corrected 288-cell counterfactual: direct edge-to-map installation.

The measured campaign installed every map update through a detour --
edge -> UE at 10.0.0.2 over the radio -> UE re-serialisation -> map on
localhost. This replay removes exactly that detour and nothing else.

Held fixed from the immutable campaign, per original cell:
capture times, action/profile identity, payload bytes, measured
reassembly/admission outcomes, the registered causal edge-arrival model, the
final optimized family-specific edge service model, and the frozen validation
quality. The scheduler remains strict latest-only with the same causal
predicted-install horizon.

Changed: ``post_publication_install_ns`` -- the span from edge tail completion
to map installation. In the measured campaign that quantity was
``map_installed_at - edge_tail_complete_wall_s`` and contained the full
edge->UE radio hop plus the UE re-serialisation. Here it is drawn only from the
new live direct edge-to-map validation, which measures the same span on the
corrected container-to-host path. No constant is guessed or subtracted.

Agent-feedback arrival is deliberately absent from this model: it is a separate
controller-observation delay and is never part of physical map-installation AoI.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from scipy import stats

from . import counterfactual_288 as source
from . import counterfactual_288_final_v3 as final_v3
from .two_stage_simulator import TwoStageConfig, TwoStageReason, simulate_two_stage


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "scenesense.splitfusion.direct_edge_map_288_counterfactual.v1"
TERMINAL = "SPLITFUSION_DIRECT_EDGE_MAP_288_COUNTERFACTUAL_COMPLETE"
DEFAULT_OUTPUT = ROOT / (
    "experiments/splitfusion_direct_edge_map_v1/20260914_direct_map_288_counterfactual"
)
BUDGETS_MS = (150, 200, 250, 300)
# Pre-registered decision rule for how narrowly the live direct-map delay may
# be pooled. Chosen before the live samples were inspected.
POOLING_ALPHA = 0.05
FAMILY_BY_ACTION = {15: "noAE", 30: "AE128", 50: "AE64", 71: "AE32"}


def _require(condition: bool, message: str) -> None:
    source._require(condition, message)


# ---------------------------------------------------------------------------
# Live direct-map delay


def load_direct_delay_samples(
    validation_root: Path,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    """Per-family live samples of the edge-tail -> map-install span, in ns."""

    results_path = validation_root / "DIRECT_VALIDATION_RESULTS.json"
    manifest_path = validation_root / "artifact_manifest.json"
    _require(results_path.is_file(), f"live validation results absent: {results_path}")
    _require(manifest_path.is_file(), f"live validation manifest absent: {manifest_path}")
    manifest = source._load_json(manifest_path)
    recorded = (manifest.get("sha256") or {}).get("DIRECT_VALIDATION_RESULTS.json")
    _require(
        recorded == source._sha256(results_path),
        "live validation results hash drift against its own manifest",
    )
    document = source._load_json(results_path)
    _require(
        document.get("gates", {}).get("status") == "PASS",
        "live direct-map validation did not pass its gates",
    )
    _require(
        bool(document.get("is_full_registered_matrix", False)),
        "live direct-map validation is a subset run, not the registered matrix",
    )

    samples: dict[str, list[int]] = {}
    per_cell: dict[str, Any] = {}
    for row in document["cells"]:
        if row.get("error"):
            continue
        action_id = int(row["action_id"])
        family = FAMILY_BY_ACTION[action_id]
        attempt = ROOT / str(row["attempt_dir"])
        ingest_csv = attempt / "direct_edge_map" / "direct_map_ingest.csv"
        _require(ingest_csv.is_file(), f"direct map ingest evidence absent: {ingest_csv}")
        values: list[int] = []
        for record in source._read_csv(ingest_csv):
            if record.get("outcome") != "RESULT_INSTALLED":
                continue
            value = source._f(record.get("install_latency_from_tail_ms", ""))
            if value is None or value < 0:
                continue
            values.append(max(1, int(round(value * 1_000_000))))
        _require(bool(values), f"{family}: live direct-map delay pool is empty")
        samples[family] = values
        per_cell[family] = {
            "action_id": action_id,
            "cell_id": row["cell_id"],
            "network_profile_id": row["network_profile_id"],
            "samples": len(values),
            "median_ms": statistics.median(values) / 1e6,
            "mean_ms": statistics.fmean(values) / 1e6,
            "p95_ms": source._percentile([v / 1e6 for v in values], 0.95),
            "ingest_csv_sha256": source._sha256(ingest_csv),
        }
    _require(
        set(samples) == set(FAMILY_BY_ACTION.values()),
        f"live validation lacks a family pool: {sorted(samples)}",
    )
    provenance = {
        "validation_root": str(validation_root.relative_to(ROOT)),
        "results_sha256": source._sha256(results_path),
        "manifest_sha256": source._sha256(manifest_path),
        "measured_quantity": (
            "map install_timestamp minus edge tail_complete_wall_s, on the "
            "direct container-to-host path"
        ),
        "matches_replaced_quantity": (
            "identical definition to the campaign's map_installed_at minus "
            "edge_tail_complete_wall_s, so the replacement is like-for-like"
        ),
        "per_family": per_cell,
    }
    return samples, provenance


def choose_pooling(samples: Mapping[str, list[int]]) -> dict[str, Any]:
    """Use the narrowest pooling the live evidence actually supports."""

    families = sorted(samples)
    groups = [samples[family] for family in families]
    test = stats.kruskal(*groups)
    family_dependent = bool(test.pvalue < POOLING_ALPHA)
    pairwise = {}
    for index, left in enumerate(families):
        for right in families[index + 1 :]:
            result = stats.mannwhitneyu(
                samples[left], samples[right], alternative="two-sided"
            )
            pairwise[f"{left}_vs_{right}"] = {
                "u": float(result.statistic),
                "p": float(result.pvalue),
                "median_delta_ms": (
                    statistics.median(samples[left])
                    - statistics.median(samples[right])
                )
                / 1e6,
            }
    pooled = [value for family in families for value in samples[family]]
    return {
        "rule": "PER_FAMILY" if family_dependent else "SINGLE_POOLED",
        "preregistered_alpha": POOLING_ALPHA,
        "kruskal_h": float(test.statistic),
        "kruskal_p": float(test.pvalue),
        "family_dependent": family_dependent,
        "pairwise_mannwhitney": pairwise,
        "family_medians_ms": {
            family: statistics.median(samples[family]) / 1e6 for family in families
        },
        "family_sample_counts": {family: len(samples[family]) for family in families},
        "pooled_median_ms": statistics.median(pooled) / 1e6,
        "pooled_samples": len(pooled),
        "interpretation": (
            "per-family pools retained because the live medians differ"
            if family_dependent
            else "a single pooled delay is used because the live families are "
            "statistically indistinguishable"
        ),
    }


def delay_pool_for(
    family: str, samples: Mapping[str, list[int]], pooling: Mapping[str, Any]
) -> list[int]:
    if pooling["rule"] == "PER_FAMILY":
        return list(samples[family])
    return [value for key in sorted(samples) for value in samples[key]]


# ---------------------------------------------------------------------------
# Replay


def _direct_frames(
    frames: Sequence[Any],
    *,
    cell_id: str,
    delay_pool: Sequence[int],
) -> tuple[list[Any], int]:
    """Replace only the post-edge installation span of each admitted frame.

    Frames that never arrived keep their zero placeholder, so transport and
    admission outcomes are bit-identical to the baseline replay.
    """

    replaced = 0
    rebuilt = []
    for frame in frames:
        if frame.arrival_ns is None:
            rebuilt.append(frame)
            continue
        delay = source._deterministic_sample(
            delay_pool, f"direct-install:{cell_id}:{frame.frame_id}"
        )
        rebuilt.append(
            dataclasses.replace(
                frame,
                post_publication_install_ns=int(delay),
                # The campaign never observed this path, so the flag is honest:
                # the install delay is modelled from the live validation.
                install_delay_observed=False,
            )
        )
        replaced += 1
    return rebuilt, replaced


def _fresh_fractions(result: Any, budgets_ms: Sequence[int]) -> dict[str, float | None]:
    """Fraction of observed route time the physical map age is within budget."""

    outcomes = [
        item
        for item in result.outcomes
        if getattr(item, "install_ns", None) is not None
    ]
    if not outcomes:
        return {f"fresh_map_fraction_{value}ms": None for value in budgets_ms}
    installs = sorted(
        ((int(item.install_ns), int(item.frame.capture_ns)) for item in outcomes)
    )
    useful: list[tuple[int, int]] = []
    newest = -1
    for install_ns, capture_ns in installs:
        if capture_ns > newest:
            useful.append((install_ns, capture_ns))
            newest = capture_ns
    captures = [int(item.frame.capture_ns) for item in result.outcomes]
    end_ns = max(captures) + source.HORIZON_NS
    relevant = [item for item in useful if item[0] < end_ns]
    if not relevant:
        return {f"fresh_map_fraction_{value}ms": None for value in budgets_ms}
    start_ns = max(min(captures), relevant[0][0])
    total_ns = end_ns - start_ns
    fresh = {value: 0 for value in budgets_ms}
    for index, (install_ns, capture_ns) in enumerate(relevant):
        interval_start = max(start_ns, install_ns)
        interval_end = (
            min(end_ns, relevant[index + 1][0])
            if index + 1 < len(relevant)
            else end_ns
        )
        if interval_end <= interval_start:
            continue
        for budget in budgets_ms:
            crossing = capture_ns + budget * 1_000_000
            covered = max(0, min(interval_end, crossing) - interval_start)
            fresh[budget] += covered
    return {
        f"fresh_map_fraction_{budget}ms": fresh[budget] / total_ns
        for budget in budgets_ms
    }


def run(output: Path, validation_root: Path) -> dict[str, Any]:
    _require(not output.exists(), f"create-only output exists: {output}")
    provenance = final_v3._verify_final_sources()
    calibration, publication_samples = final_v3._final_calibration()
    samples, delay_provenance = load_direct_delay_samples(validation_root)
    pooling = choose_pooling(samples)

    cells = source._read_csv(source.CONSOLIDATION / "campaign_288_cell_table.csv")
    _require(len(cells) == 288, "campaign table is not 288 cells")
    quality = source._quality_by_action()
    (
        action_services,
        family_services,
        profile_delays,
        action_profile_arrivals,
        profile_arrivals,
        source_hashes,
    ) = source._build_empirical_pools(cells)

    rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    predictions: dict[str, dict[str, Any]] = {}
    total_replaced = 0

    for number, cell in enumerate(cells, start=1):
        attempt = source._attempt(cell)
        sent = source._sent_rows(attempt)
        measured = source._measured_install_summary(sent)
        family = cell["family"]
        frames, counters = source._candidate_frames(
            cell=cell,
            rows=sent,
            family_calibration=calibration[family],
            publication_samples=publication_samples[family],
            action_service_pool=action_services.get(int(cell["action_id"]), ()),
            family_service_pool=family_services[family],
            profile_delay_pool=profile_delays[cell["network_profile"]],
            action_profile_arrival_pool=action_profile_arrivals.get(
                (int(cell["action_id"]), cell["network_profile"]), ()
            ),
            profile_arrival_pool=profile_arrivals[cell["network_profile"]],
        )
        publication_ns = int(
            round(calibration[family]["optimized_publication_ms_median"] * 1e6)
        )
        total_ns = int(
            round(calibration[family]["optimized_total_edge_processing_ms_median"] * 1e6)
        )
        compute_ns = max(1, total_ns - publication_ns)

        # -- baseline (edge -> UE -> map), reproduced exactly ---------------
        baseline_install_ns = int(statistics.median(profile_delays[cell["network_profile"]]))
        baseline_result = simulate_two_stage(
            frames,
            config=TwoStageConfig(
                queue_wait_budget_ns=None,
                processing_horizon_ns=source.HORIZON_NS,
                service_target_ns=source.SERVICE_TARGET_NS,
                predicted_compute_ns=compute_ns,
                predicted_publication_ns=publication_ns,
                predicted_post_publication_install_ns=baseline_install_ns,
            ),
        )
        baseline_summary = baseline_result.summary()
        baseline_row = source._flatten_cell(
            cell, baseline_summary, counters, quality[int(cell["action_id"])],
            measured, source_hashes[cell["cell_id"]],
        )
        baseline_row.update(_fresh_fractions(baseline_result, BUDGETS_MS))
        baseline_rows.append(baseline_row)

        # -- corrected (direct edge -> map) ---------------------------------
        delay_pool = delay_pool_for(family, samples, pooling)
        direct_frames, replaced = _direct_frames(
            frames, cell_id=cell["cell_id"], delay_pool=delay_pool
        )
        total_replaced += replaced
        direct_install_ns = int(statistics.median(delay_pool))
        predictions[f"{family}/{cell['network_profile']}"] = {
            "edge_variant": calibration[family]["target_variant"],
            "compute_ms": compute_ns / 1e6,
            "publication_ms": publication_ns / 1e6,
            "baseline_post_publication_install_ms": baseline_install_ns / 1e6,
            "direct_post_publication_install_ms": direct_install_ns / 1e6,
            "install_path_delta_ms": (baseline_install_ns - direct_install_ns) / 1e6,
        }
        result = simulate_two_stage(
            direct_frames,
            config=TwoStageConfig(
                queue_wait_budget_ns=None,
                processing_horizon_ns=source.HORIZON_NS,
                service_target_ns=source.SERVICE_TARGET_NS,
                predicted_compute_ns=compute_ns,
                predicted_publication_ns=publication_ns,
                predicted_post_publication_install_ns=direct_install_ns,
            ),
        )
        summary = result.summary()
        _require(
            sum(summary["reason_counts"].values()) == len(sent),
            f"{cell['cell_id']}: terminal accounting drift",
        )
        row = source._flatten_cell(
            cell, summary, counters, quality[int(cell["action_id"])],
            measured, source_hashes[cell["cell_id"]],
        )
        row.update(_fresh_fractions(result, BUDGETS_MS))
        row["edge_target_variant"] = calibration[family]["target_variant"]
        row["edge_target_anchor_action_id"] = calibration[family]["anchor_action_id"]
        row["install_path"] = "DIRECT_EDGE_TO_MAP"
        row["direct_post_publication_install_ms"] = direct_install_ns / 1e6
        row["baseline_post_publication_install_ms"] = baseline_install_ns / 1e6
        row["direct_install_frames_replaced"] = replaced
        for key in (
            "reason_TRANSPORT_INCOMPLETE",
            "reason_MEASURED_PRE_QUEUE_REJECTION",
            "edge_scheduler_input_frames",
            "measured_reassemblies",
            "measured_edge_admissions",
            "input_frames",
        ):
            _require(
                row[key] == baseline_row[key],
                f"{cell['cell_id']}: upstream quantity {key} changed",
            )
        rows.append(row)
        if number % 24 == 0:
            print(f"direct-map simulation: {number}/288 cells", flush=True)

    aggregate = source._aggregate(rows)
    baseline_aggregate = source._aggregate(baseline_rows)
    overall = aggregate["overall"]
    _require(overall["sent"] == 896_856, "source sent total drift")
    _require(overall["source_ack_installed"] == 334_174, "source installed total drift")
    _require(overall["transport_incomplete"] == 184_124, "transport total changed")
    _require(
        overall["measured_pre_queue_rejections"] == 110_417,
        "edge-admission total changed",
    )
    _require(
        baseline_aggregate["overall"]["transport_incomplete"]
        == overall["transport_incomplete"]
        and baseline_aggregate["overall"]["measured_pre_queue_rejections"]
        == overall["measured_pre_queue_rejections"],
        "upstream transport/admission totals differ between the two replays",
    )

    document = {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "scientific_status": "COUNTERFACTUAL_MODEL_NOT_LIVE_REMEASUREMENT",
        "architecture": "DIRECT_EDGE_TO_MAP_V1",
        "policy": {
            "scheduler": "PREDICTED_INSTALL_HORIZON_LATEST_ONLY_DEPTH_ONE",
            "prediction_uses_current_frame_future": False,
            "processing_horizon_ms": 500,
            "service_reporting_reference_ms": 100,
            "install_path": "DIRECT_EDGE_TO_MAP",
        },
        "provenance": provenance,
        "direct_delay_provenance": delay_provenance,
        "pooling_decision": pooling,
        "family_calibration": calibration,
        "prediction_inputs": predictions,
        "inventory": {
            "cells": len(rows),
            "actions": len({row["action_id"] for row in rows}),
            "network_profiles": sorted({row["network_profile"] for row in rows}),
            "frames_with_replaced_install_path": total_replaced,
            "source_per_frame_hashes_verified": len(source_hashes),
        },
        "aggregate": aggregate,
        "baseline_aggregate": baseline_aggregate,
        "budgets_ms": list(BUDGETS_MS),
        "extrapolations": [
            "the live direct-map delay was measured only under FAVORABLE_STABLE "
            "and is applied to all four network profiles; the justification is "
            "physical -- the direct path is a container-to-host datagram on the "
            "CN5G bridge that never traverses the radio -- but it is an "
            "extrapolation and is not independently confirmed per profile",
            "the live validation covers actions 15/30/50/71; every other action "
            "in a family inherits that family's measured delay",
            "the edge service model, arrival model and quality are unchanged "
            "from the final optimized-edge analysis and are not re-measured",
        ],
        "limitations": [
            "counterfactual model, not a live remeasurement of 288 cells",
            "only the post-edge installation path changed; transport, admission "
            "and edge service are bit-identical to the baseline replay",
            "agent-feedback arrival is excluded from physical map AoI by design",
            "no 100-ms service-readiness claim",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    source._write_csv(output / "direct_map_288_cell_summary.csv", rows)
    source._write_csv(output / "baseline_288_cell_summary.csv", baseline_rows)
    source._atomic_json(output / "counterfactual_results.json", document)
    source._atomic_text(output / "REPORT.md", _report(document))
    primary = (
        "direct_map_288_cell_summary.csv",
        "baseline_288_cell_summary.csv",
        "counterfactual_results.json",
        "REPORT.md",
    )
    hashes = {name: source._sha256(output / name) for name in primary}
    source._atomic_json(
        output / "artifact_manifest.json",
        {"schema": f"{SCHEMA}.artifacts", "status": "COMPLETE", "sha256": hashes},
    )
    source._atomic_text(
        output / TERMINAL,
        json.dumps(
            {
                "schema": f"{SCHEMA}.terminal",
                "status": "COMPLETE",
                "results_sha256": hashes["counterfactual_results.json"],
            },
            sort_keys=True,
        )
        + "\n",
    )
    return document


def _report(document: Mapping[str, Any]) -> str:
    overall = document["aggregate"]["overall"]
    baseline = document["baseline_aggregate"]["overall"]
    pooling = document["pooling_decision"]
    lines = [
        "# Corrected 288-cell counterfactual: direct edge-to-map installation",
        "",
        "The measured 288-cell arrival/reassembly/admission surface and the final",
        "optimized edge service are held fixed. Only the post-edge installation",
        "path changed: the edge -> UE -> map detour is removed and replaced by the",
        "direct edge-to-map delay measured in the live validation.",
        "",
        "**Counterfactual model, not a live remeasurement.**",
        "",
        f"Pooling rule: `{pooling['rule']}` "
        f"(Kruskal-Wallis H={pooling['kruskal_h']:.3f}, p={pooling['kruskal_p']:.3g}, "
        f"pre-registered alpha={pooling['preregistered_alpha']}).",
        "",
        "## Aggregate: old detour versus corrected direct path",
        "",
        "| Quantity | Measured campaign | Old edge->UE->map replay | Corrected direct edge->map |",
        "|---|---:|---:|---:|",
        f"| Installed/sent | {overall['source_rate_installed_per_sent']:.4f} | "
        f"{baseline['rate_installed_per_sent']:.4f} | {overall['rate_installed_per_sent']:.4f} |",
        f"| Installed frames | {overall['source_ack_installed']:,} | "
        f"{baseline['ack_installed']:,} | {overall['ack_installed']:,} |",
        f"| Useful installations | {overall['source_useful_newer_map_installations']:,} | "
        f"{baseline['useful_newer_map_installations']:,} | "
        f"{overall['useful_newer_map_installations']:,} |",
        f"| Median cell install AoI | {overall['source_install_aoi_ms_cell_median']:.1f} ms | "
        f"{baseline['install_aoi_ms_cell_median']:.1f} ms | "
        f"{overall['install_aoi_ms_cell_median']:.1f} ms |",
        f"| Median cell map AoI | {overall['source_time_weighted_map_aoi_ms_cell_median']:.1f} ms | "
        f"{baseline['time_weighted_map_aoi_ms_cell_median']:.1f} ms | "
        f"{overall['time_weighted_map_aoi_ms_cell_median']:.1f} ms |",
        "",
        "## Extrapolations",
        "",
    ]
    for item in document["extrapolations"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Limitations", ""])
    for item in document["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    document = run(args.output.resolve(), args.validation_root.resolve(strict=True))
    print(json.dumps(document["pooling_decision"], indent=2, sort_keys=True))
    print(TERMINAL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
