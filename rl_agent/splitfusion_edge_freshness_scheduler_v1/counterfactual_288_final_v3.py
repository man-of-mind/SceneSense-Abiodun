#!/usr/bin/env python3
"""Apply the final qualified edge service to all 288 measured cells.

Measured capture times, payloads, complete-reassembly counts, edge-admission
counts and available arrival times stay fixed.  Only edge service changes, and
the causal predicted-install, depth-one latest-only scheduler is replayed.

NoAE retains the live-valid v2 edge path because action 15 rejected v3
fail-closed on a non-finite geometry output.  The three AE families use v3.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import counterfactual_288 as source
from .two_stage_simulator import TwoStageConfig, TwoStageReason, simulate_two_stage


ROOT = Path(__file__).resolve().parents[2]
FINAL_LIVE = ROOT / (
    "experiments/splitfusion_edge_optimization_v3/"
    "20260910_live_v2_vs_v3_actions30_50_71_retry1"
)
NOAE_LIVE = ROOT / (
    "experiments/splitfusion_edge_optimization_v3/"
    "20260910_live_v2_vs_v3_actions30_15_50_71/"
    "action15__v2_predicted_install_horizon"
)
ACTION15_FAILURE = ROOT / (
    "experiments/splitfusion_edge_optimization_v3/"
    "20260910_action15_v3_failure_reproduction_retry1/"
    "action15__v3_overlapped_final/NO_SCIENTIFIC_ROWS.json"
)
PREDECESSOR = ROOT / (
    "experiments/splitfusion_edge_freshness_scheduler_v1/"
    "20260910_predicted_horizon_288_v2/comparison.json"
)
PREDECESSOR_MANIFEST = PREDECESSOR.with_name("artifact_manifest.json")
DEFAULT_OUTPUT = ROOT / (
    "experiments/splitfusion_edge_freshness_scheduler_v1/"
    "20260911_final_v3_predicted_horizon_288"
)
SCHEMA = "scenesense.splitfusion.final_v3_predicted_horizon_288.v1"
TERMINAL = "SPLITFUSION_FINAL_V3_PREDICTED_HORIZON_288_COMPLETE"
EXPECTED_MANIFESTS = {
    FINAL_LIVE / "ARTIFACT_MANIFEST.json": (
        "96fffa9d0f821af6fea571425f8190d482651d294d4354beaac0ca04057c675d"
    ),
    NOAE_LIVE / "ARTIFACT_MANIFEST.json": (
        "d8cf19d50a87c85a3847701dbab45f6231d4fcc23d920a050a1a1dbcbeeb64bd"
    ),
}
ACTION15_FAILURE_SHA256 = (
    "aca5a6eb5ea1e6e26156fd95ba739a4dd17a0f706c4075f27e415290a001c352"
)
PREDECESSOR_MANIFEST_SHA256 = (
    "648fa97cc5e5adec73e60defb0f4822fa72d9c6a04f453531bb52cad9612339d"
)
TARGETS = {
    "noAE": (15, NOAE_LIVE, "SYNCHRONIZATION_LIGHT_V2"),
    "AE128": (
        30,
        FINAL_LIVE / "action30__v3_overlapped_final",
        "OVERLAPPED_OUTPUT_PRESERVING_V3",
    ),
    "AE64": (
        50,
        FINAL_LIVE / "action50__v3_overlapped_final",
        "OVERLAPPED_OUTPUT_PRESERVING_V3",
    ),
    "AE32": (
        71,
        FINAL_LIVE / "action71__v3_overlapped_final",
        "OVERLAPPED_OUTPUT_PRESERVING_V3",
    ),
}


def _verify_sha(path: Path, expected: str, label: str) -> None:
    source._require(path.is_file(), f"{label} is absent: {path}")
    source._require(source._sha256(path) == expected, f"{label} hash drift")


def _verify_final_sources() -> dict[str, Any]:
    provenance = source._verify_sources()
    for path, expected in EXPECTED_MANIFESTS.items():
        _verify_sha(path, expected, "final live manifest")
    verified = source._verify_manifest(FINAL_LIVE)
    verified += source._verify_manifest(NOAE_LIVE)
    _verify_sha(
        ACTION15_FAILURE,
        ACTION15_FAILURE_SHA256,
        "action-15 v3 failure record",
    )
    failure = source._load_json(ACTION15_FAILURE)
    source._require(
        "non-finite" in str(failure.get("failure", "")),
        "action-15 v3 exclusion reason drift",
    )
    _verify_sha(
        PREDECESSOR_MANIFEST,
        PREDECESSOR_MANIFEST_SHA256,
        "predecessor counterfactual manifest",
    )
    predecessor_manifest = source._load_json(PREDECESSOR_MANIFEST)
    predecessor_hashes = predecessor_manifest.get("files", {})
    source._require(
        predecessor_hashes.get(PREDECESSOR.name) == source._sha256(PREDECESSOR),
        "predecessor comparison hash drift",
    )
    provenance["final_edge"] = {
        "final_live_manifest_sha256": source._sha256(
            FINAL_LIVE / "ARTIFACT_MANIFEST.json"
        ),
        "noae_live_manifest_sha256": source._sha256(
            NOAE_LIVE / "ARTIFACT_MANIFEST.json"
        ),
        "action15_v3_failure_sha256": source._sha256(ACTION15_FAILURE),
        "verified_live_artifacts": verified,
        "predecessor_manifest_sha256": source._sha256(PREDECESSOR_MANIFEST),
        "predecessor_comparison_sha256": source._sha256(PREDECESSOR),
        "noae_policy": (
            "retain live-valid v2 edge service; do not extrapolate through the "
            "v3 non-finite action-15 result"
        ),
    }
    provenance["implementation"][str(Path(__file__).resolve().relative_to(ROOT))] = (
        source._sha256(Path(__file__).resolve())
    )
    return provenance


def _single_action_summary(root: Path, action_id: int) -> Mapping[str, Any]:
    document = source._load_json(root / "LIVE_DIAGNOSTIC_RESULTS.json")
    rows = [
        row
        for row in document["action_summaries"]
        if int(row["action_id"]) == action_id
    ]
    source._require(len(rows) == 1, f"action {action_id} lacks one live summary")
    return rows[0]


def _final_calibration() -> tuple[dict[str, dict[str, Any]], dict[str, list[int]]]:
    baseline = source._action_summaries(
        source.BASELINE / "LIVE_DIAGNOSTIC_RESULTS.json"
    )
    calibration: dict[str, dict[str, Any]] = {}
    publication: dict[str, list[int]] = {}
    for family, (action_id, target_root, variant) in TARGETS.items():
        target = _single_action_summary(target_root, action_id)
        old_ms = float(
            baseline[action_id]["timing"]["edge_total_edge_processing_ms"]["median"]
        )
        target_ms = float(
            target["timing"]["edge_total_edge_processing_ms"]["median"]
        )
        source._require(target_ms > 0 and old_ms > target_ms, f"{family} target drift")
        files = sorted((target_root / "per_frame").glob(f"action_{action_id}_*.csv"))
        source._require(len(files) == 1, f"{family} lacks one per-frame target file")
        samples = []
        for row in source._read_csv(files[0]):
            value = source._f(row.get("edge_output_serialization_ms", ""))
            if value is not None and value > 0:
                samples.append(max(1, int(round(value * 1_000_000))))
        source._require(bool(samples), f"{family} publication pool is empty")
        publication[family] = samples
        calibration[family] = {
            "anchor_action_id": action_id,
            "target_variant": variant,
            "target_profile": "FAVORABLE_STABLE",
            "baseline_total_edge_processing_ms_median": old_ms,
            "target_total_edge_processing_ms_median": target_ms,
            "optimized_total_edge_processing_ms_median": target_ms,
            "total_edge_processing_reduction_ms": old_ms - target_ms,
            "optimized_publication_ms_median": statistics.median(samples) / 1e6,
            "optimized_publication_samples": len(samples),
            "application": (
                "subtract the family median reduction from original per-frame "
                "edge service, retain residual variance, then causally reschedule"
            ),
        }
    source._require(set(calibration) == set(TARGETS), "family calibration incomplete")
    return calibration, publication


def _report(document: Mapping[str, Any]) -> str:
    overall = document["aggregate"]["overall"]
    predecessor = document["predecessor_comparison"]
    lines = [
        "# Final-edge 288-cell counterfactual",
        "",
        "The measured 288-cell arrival/reassembly/admission surface was replayed",
        "through the causal predicted-install, depth-one latest-only scheduler.",
        "AE128/AE64/AE32 use final v3 edge calibration. NoAE conservatively",
        "retains live-valid v2 because action 15 rejected v3 fail-closed.",
        "",
        "This is a counterfactual model, not a live remeasurement.",
        "",
        "## Aggregate",
        "",
        "| Quantity | Measured | Previous simulator | Final-edge simulator |",
        "|---|---:|---:|---:|",
        f"| Installed/sent | {overall['source_rate_installed_per_sent']:.4f} | {predecessor['rate_installed_per_sent']:.4f} | {overall['rate_installed_per_sent']:.4f} |",
        f"| Installed frames | {overall['source_ack_installed']:,} | {predecessor['ack_installed']:,} | {overall['ack_installed']:,} |",
        f"| Useful installations | {overall['source_useful_newer_map_installations']:,} | {predecessor['useful_newer_map_installations']:,} | {overall['useful_newer_map_installations']:,} |",
        f"| Median cell install AoI | {overall['source_install_aoi_ms_cell_median']:.1f} ms | {predecessor['install_aoi_ms_cell_median']:.1f} ms | {overall['install_aoi_ms_cell_median']:.1f} ms |",
        f"| Median cell map AoI | {overall['source_time_weighted_map_aoi_ms_cell_median']:.1f} ms | {predecessor['time_weighted_map_aoi_ms_cell_median']:.1f} ms | {overall['time_weighted_map_aoi_ms_cell_median']:.1f} ms |",
        "",
        "## Network profiles",
        "",
        "| Profile | Source install/sent | Final install/sent | Final install AoI | Final map AoI |",
        "|---|---:|---:|---:|---:|",
    ]
    for profile, block in document["aggregate"]["by_network_profile"].items():
        lines.append(
            f"| {profile} | {block['source_rate_installed_per_sent']:.4f} | "
            f"{block['rate_installed_per_sent']:.4f} | "
            f"{block['install_aoi_ms_cell_median']:.1f} ms | "
            f"{block['time_weighted_map_aoi_ms_cell_median']:.1f} ms |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Payload, transport completion and edge admission stay fixed per cell.",
            "- Missing admitted identities/timestamps retain deterministic imputations.",
            "- Family calibration is extrapolated from one favorable anchor.",
            "- NoAE does not use v3 and is marked separately in every row.",
            "- This supports simulator construction, not a 100-ms readiness claim.",
            "",
        ]
    )
    return "\n".join(lines)


def run(output: Path) -> dict[str, Any]:
    source._require(not output.exists(), f"create-only output exists: {output}")
    provenance = _verify_final_sources()
    calibration, publication_samples = _final_calibration()
    cells = source._read_csv(
        source.CONSOLIDATION / "campaign_288_cell_table.csv"
    )
    source._require(len(cells) == 288, "campaign table is not 288 cells")
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
    predictions: dict[str, dict[str, Any]] = {}

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
        install_ns = int(statistics.median(profile_delays[cell["network_profile"]]))
        predictions[f"{family}/{cell['network_profile']}"] = {
            "edge_variant": calibration[family]["target_variant"],
            "compute_ms": compute_ns / 1e6,
            "publication_ms": publication_ns / 1e6,
            "post_publication_install_ms": install_ns / 1e6,
        }
        result = simulate_two_stage(
            frames,
            config=TwoStageConfig(
                queue_wait_budget_ns=None,
                processing_horizon_ns=source.HORIZON_NS,
                service_target_ns=source.SERVICE_TARGET_NS,
                predicted_compute_ns=compute_ns,
                predicted_publication_ns=publication_ns,
                predicted_post_publication_install_ns=install_ns,
            ),
        )
        summary = result.summary()
        source._require(
            sum(summary["reason_counts"].values()) == len(sent),
            f"{cell['cell_id']}: terminal accounting drift",
        )
        row = source._flatten_cell(
            cell,
            summary,
            counters,
            quality[int(cell["action_id"])],
            measured,
            source_hashes[cell["cell_id"]],
        )
        row["edge_target_variant"] = calibration[family]["target_variant"]
        row["edge_target_anchor_action_id"] = calibration[family]["anchor_action_id"]
        source._require(
            row["edge_scheduler_input_frames"] == row["measured_edge_admissions"],
            f"{cell['cell_id']}: admission count drift",
        )
        source._require(
            row["reason_TRANSPORT_INCOMPLETE"]
            == row["input_frames"] - row["measured_reassemblies"],
            f"{cell['cell_id']}: transport count drift",
        )
        source._require(
            row["reason_MEASURED_PRE_QUEUE_REJECTION"]
            == row["measured_reassemblies"] - row["measured_edge_admissions"],
            f"{cell['cell_id']}: admission rejection drift",
        )
        rows.append(row)
        if number % 24 == 0:
            print(f"final-edge simulation: {number}/288 cells", flush=True)

    aggregate = source._aggregate(rows)
    predecessor = source._load_json(PREDECESSOR)["predicted_horizon"]
    overall = aggregate["overall"]
    source._require(overall["sent"] == 896_856, "source sent total drift")
    source._require(
        overall["source_ack_installed"] == 334_174,
        "source installed total drift",
    )
    source._require(
        overall["transport_incomplete"] == 184_124,
        "transport total changed",
    )
    source._require(
        overall["measured_pre_queue_rejections"] == 110_417,
        "edge-admission total changed",
    )
    document = {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "scientific_status": "COUNTERFACTUAL_MODEL_NOT_LIVE_REMEASUREMENT",
        "policy": {
            "scheduler": "PREDICTED_INSTALL_HORIZON_LATEST_ONLY_DEPTH_ONE",
            "prediction_uses_current_frame_future": False,
            "processing_horizon_ms": 500,
            "service_reporting_reference_ms": 100,
        },
        "provenance": provenance,
        "family_calibration": calibration,
        "prediction_inputs": predictions,
        "inventory": {
            "cells": len(rows),
            "actions": len({row["action_id"] for row in rows}),
            "network_profiles": sorted({row["network_profile"] for row in rows}),
            "terminal_reconciled_cells": sum(
                sum(row[f"reason_{reason.value}"] for reason in TwoStageReason)
                == row["input_frames"]
                for row in rows
            ),
            "source_per_frame_hashes_verified": len(source_hashes),
            "v3_families": ["AE128", "AE64", "AE32"],
            "v2_retained_families": ["noAE"],
        },
        "aggregate": aggregate,
        "predecessor_comparison": predecessor,
        "limitations": [
            "counterfactual model, not live remeasurement",
            "final service calibration extrapolated from one favorable anchor per family",
            "noAE retains live-valid v2 because action-15 v3 failed closed",
            "missing frame identities and timing fields retain deterministic imputations",
            "no 100-ms service-readiness claim",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    source._write_csv(output / "final_v3_288_cell_summary.csv", rows)
    source._atomic_json(output / "counterfactual_results.json", document)
    source._atomic_text(output / "REPORT.md", _report(document))
    primary = (
        "final_v3_288_cell_summary.csv",
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
                "cell_summary_sha256": hashes["final_v3_288_cell_summary.csv"],
            },
            sort_keys=True,
        )
        + "\n",
    )
    return document


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    document = run(args.output.resolve())
    print(json.dumps(document["aggregate"], indent=2, sort_keys=True))
    print(TERMINAL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
