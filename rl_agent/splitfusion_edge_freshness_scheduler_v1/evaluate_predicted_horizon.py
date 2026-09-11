#!/usr/bin/env python3
"""Compare fixed 25-ms expiry with a causal predicted-install horizon.

This is an offline policy hypothesis over the same reconstructed 288-cell
inputs used by the registered counterfactual. It does not rewrite that result.
The candidate uses only an action-family service median and network-profile
post-publication median available before a frame begins; it never reads the
current frame's future duration to decide whether to start it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import counterfactual_288 as source
from .two_stage_simulator import TwoStageConfig, simulate_two_stage


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / (
    "experiments/splitfusion_edge_freshness_scheduler_v1/"
    "20260910_predicted_horizon_288_v2"
)
FIXED_RESULT = ROOT / (
    "experiments/splitfusion_edge_freshness_scheduler_v1/"
    "20260910_counterfactual_288_v4/counterfactual_results.json"
)
SCHEMA = "scenesense.splitfusion.predicted_horizon_288.v2"
SUCCESS_TERMINAL = "SPLITFUSION_PREDICTED_HORIZON_288_COUNTERFACTUAL_COMPLETE"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, document: Any) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(rows[0])
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _report(document: Mapping[str, Any]) -> str:
    fixed = document["fixed_25_ms"]
    predicted = document["predicted_horizon"]
    return "\n".join(
        [
            "# Latest-only scheduler: fixed expiry versus predicted usefulness",
            "",
            "This is a counterfactual model over the same 288 reconstructed cells,",
            "not a live remeasurement. The predicted policy never uses the current",
            "frame's realized service time. It uses frozen action-family and network-",
            "profile medians known before the frame starts.",
            "",
            "| Metric | Fixed 25 ms | Predicted 500 ms install horizon |",
            "|---|---:|---:|",
            f"| Installed/sent | {fixed['rate_installed_per_sent']:.4f} | {predicted['rate_installed_per_sent']:.4f} |",
            f"| Install AoI, median of cell medians | {fixed['install_aoi_ms_cell_median']:.1f} ms | {predicted['install_aoi_ms_cell_median']:.1f} ms |",
            f"| Time-weighted map AoI, cell median | {fixed['time_weighted_map_aoi_ms_cell_median']:.1f} ms | {predicted['time_weighted_map_aoi_ms_cell_median']:.1f} ms |",
            f"| Useful installations | {fixed['useful_newer_map_installations']:,} | {predicted['useful_newer_map_installations']:,} |",
            f"| Fixed queue expiries | {fixed['queue_expiry_count']:,} | 0 |",
            f"| Predicted-obsolete drops | 0 | {predicted['predicted_obsolete_count']:,} |",
            "",
            "The candidate processes the only pending frame even after 25 ms when its",
            "predicted map-install age still fits 500 ms. It drops that frame only",
            "when the causally predicted completion would already exceed the horizon.",
            "The 100 ms value remains a reporting target, not this admission horizon.",
            "",
            "## Limits",
            "",
            "- Service estimates are fixed family medians from prior live measurements.",
            "- Install-delay estimates are fixed network-profile medians from the completed campaign.",
            "- A deployment should update those estimates causally (for example, an EWMA) and retain the frozen fallback.",
            "- This comparison does not include the new v2 post-processing saving, which is not yet live-qualified.",
            "",
        ]
    )


def run(output: Path) -> dict[str, Any]:
    source._require(not output.exists(), f"create-only output exists: {output}")
    provenance = source._verify_sources()
    family, publication_samples = source._calibration()
    cells = source._read_csv(source.CONSOLIDATION / "campaign_288_cell_table.csv")
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
    fixed_rows: list[dict[str, Any]] = []
    predicted_rows: list[dict[str, Any]] = []
    estimates: dict[str, dict[str, float]] = {}

    for number, cell in enumerate(cells, start=1):
        attempt = source._attempt(cell)
        sent = source._sent_rows(attempt)
        measured_install = source._measured_install_summary(sent)
        family_name = cell["family"]
        frames, counters = source._candidate_frames(
            cell=cell,
            rows=sent,
            family_calibration=family[family_name],
            publication_samples=publication_samples[family_name],
            action_service_pool=action_services.get(int(cell["action_id"]), ()),
            family_service_pool=family_services[family_name],
            profile_delay_pool=profile_delays[cell["network_profile"]],
            action_profile_arrival_pool=action_profile_arrivals.get(
                (int(cell["action_id"]), cell["network_profile"]), ()
            ),
            profile_arrival_pool=profile_arrivals[cell["network_profile"]],
        )
        fixed_result = simulate_two_stage(
            frames,
            config=TwoStageConfig(
                queue_wait_budget_ns=source.WAIT_BUDGET_NS,
                processing_horizon_ns=source.HORIZON_NS,
                service_target_ns=source.SERVICE_TARGET_NS,
            ),
        )
        publication_ns = int(
            round(float(family[family_name]["optimized_publication_ms_median"]) * 1e6)
        )
        total_ns = int(
            round(
                float(family[family_name]["optimized_total_edge_processing_ms_median"])
                * 1e6
            )
        )
        compute_ns = max(1, total_ns - publication_ns)
        install_ns = int(statistics.median(profile_delays[cell["network_profile"]]))
        estimates[f"{family_name}/{cell['network_profile']}"] = {
            "compute_ms": compute_ns / 1e6,
            "publication_ms": publication_ns / 1e6,
            "post_publication_install_ms": install_ns / 1e6,
        }
        predicted_result = simulate_two_stage(
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
        for destination, result in (
            (fixed_rows, fixed_result),
            (predicted_rows, predicted_result),
        ):
            summary = result.summary()
            source._require(
                sum(summary["reason_counts"].values()) == len(sent),
                f"{cell['cell_id']}: terminal accounting drift",
            )
            destination.append(
                source._flatten_cell(
                    cell,
                    summary,
                    counters,
                    quality[int(cell["action_id"])],
                    measured_install,
                    source_hashes[cell["cell_id"]],
                )
            )
        if number % 24 == 0:
            print(f"scheduler comparison: {number}/288 cells", flush=True)

    fixed_aggregate = source._aggregate(fixed_rows)["overall"]
    predicted_aggregate = source._aggregate(predicted_rows)["overall"]
    fixed_aggregate["queue_expiry_count"] = sum(
        int(row["reason_QUEUE_WAIT_BUDGET_EXCEEDED"]) for row in fixed_rows
    )
    fixed_aggregate["predicted_obsolete_count"] = 0
    predicted_aggregate["queue_expiry_count"] = 0
    predicted_aggregate["predicted_obsolete_count"] = sum(
        int(row["reason_PREDICTED_MAP_INSTALL_HORIZON_EXCEEDED"])
        for row in predicted_rows
    )
    registered = json.loads(FIXED_RESULT.read_text(encoding="utf-8"))["aggregate"][
        "overall"
    ]
    for key in (
        "ack_installed",
        "rate_installed_per_sent",
        "install_aoi_ms_cell_median",
        "time_weighted_map_aoi_ms_cell_median",
    ):
        source._require(
            fixed_aggregate[key] == registered[key],
            f"fixed 25-ms reproduction drift: {key}",
        )

    output.mkdir(parents=True, exist_ok=False)
    comparison_rows = []
    for fixed, predicted in zip(fixed_rows, predicted_rows):
        source._require(fixed["cell_id"] == predicted["cell_id"], "cell order drift")
        comparison_rows.append(
            {
                "cell_id": fixed["cell_id"],
                "action_id": fixed["action_id"],
                "family": fixed["family"],
                "network_profile": fixed["network_profile"],
                "fixed_25_installed": fixed["ack_installed_frames"],
                "predicted_horizon_installed": predicted["ack_installed_frames"],
                "fixed_25_install_aoi_ms_median": fixed["install_aoi_ms_median"],
                "predicted_horizon_install_aoi_ms_median": predicted[
                    "install_aoi_ms_median"
                ],
                "fixed_25_time_weighted_map_aoi_ms": fixed[
                    "time_weighted_map_aoi_ms"
                ],
                "predicted_horizon_time_weighted_map_aoi_ms": predicted[
                    "time_weighted_map_aoi_ms"
                ],
                "fixed_25_expired": fixed["reason_QUEUE_WAIT_BUDGET_EXCEEDED"],
                "predicted_obsolete": predicted[
                    "reason_PREDICTED_MAP_INSTALL_HORIZON_EXCEEDED"
                ],
            }
        )
    document = {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "scientific_status": "COUNTERFACTUAL_POLICY_HYPOTHESIS_NOT_LIVE_MEASUREMENT",
        "implementation_sha256": _sha256(Path(__file__).resolve()),
        "source_provenance": provenance,
        "fixed_result_sha256": _sha256(FIXED_RESULT),
        "prediction_inputs": estimates,
        "fixed_25_ms": fixed_aggregate,
        "predicted_horizon": predicted_aggregate,
        "delta_predicted_minus_fixed": {
            "ack_installed": (
                predicted_aggregate["ack_installed"]
                - fixed_aggregate["ack_installed"]
            ),
            "useful_newer_map_installations": (
                predicted_aggregate["useful_newer_map_installations"]
                - fixed_aggregate["useful_newer_map_installations"]
            ),
            "install_aoi_ms_cell_median": (
                predicted_aggregate["install_aoi_ms_cell_median"]
                - fixed_aggregate["install_aoi_ms_cell_median"]
            ),
            "time_weighted_map_aoi_ms_cell_median": (
                predicted_aggregate["time_weighted_map_aoi_ms_cell_median"]
                - fixed_aggregate["time_weighted_map_aoi_ms_cell_median"]
            ),
        },
        "policy": {
            "pending_slot": "LATEST_ONLY_DEPTH_ONE",
            "current_frame_future_service_used": False,
            "predicted_map_install_horizon_ms": 500,
            "service_reporting_target_ms": 100,
        },
    }
    _write_csv(output / "cell_comparison.csv", comparison_rows)
    _write_json(output / "comparison.json", document)
    _write_text(output / "REPORT.md", _report(document))
    primary_files = ("cell_comparison.csv", "comparison.json", "REPORT.md")
    terminal = {
        "schema": "scenesense.splitfusion.predicted_horizon_terminal.v1",
        "terminal": SUCCESS_TERMINAL,
        "status": "COMPLETE",
        "artifacts": {
            name: _sha256(output / name) for name in primary_files
        },
    }
    _write_json(output / f"{SUCCESS_TERMINAL}.json", terminal)
    registered = primary_files + (f"{SUCCESS_TERMINAL}.json",)
    _write_json(
        output / "artifact_manifest.json",
        {
            "schema": "scenesense.splitfusion.predicted_horizon_manifest.v1",
            "status": "COMPLETE",
            "files": {name: _sha256(output / name) for name in registered},
        },
    )
    return document


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    document = run(args.output.resolve())
    print(
        json.dumps(
            {
                "fixed_25_ms": document["fixed_25_ms"],
                "predicted_horizon": document["predicted_horizon"],
            },
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
