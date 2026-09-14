#!/usr/bin/env python3
"""Build presentation tables for freshness budgets, quality, and latency.

This analysis extends the immutable dual-clock 288-cell counterfactual with a
300 ms physical-map freshness budget.  It uses the final optimized edge
calibration and strict latest-only/no-expiry scheduling already qualified by
the source analysis.  No live experiment or source measurement is modified.

"Meets a budget" is deliberately defined as keeping physical map age at or
below that budget for at least half of the complete observation interval.  A
separate sensitivity table retains the 25%, 50%, and 75% route-time counts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rl_agent.splitfusion_edge_freshness_scheduler_v1 import (
    counterfactual_288 as source,
)
from rl_agent.splitfusion_edge_freshness_scheduler_v1 import (
    counterfactual_288_final_v3 as final_v3,
)
from rl_agent.splitfusion_map_freshness_analysis_v1 import analyze_288 as prior
from rl_agent.splitfusion_map_freshness_analysis_v1 import (
    analyze_timing_boundaries as timing,
)
from rl_agent.splitfusion_map_freshness_analysis_v1.queue_models import (
    QueueOutcome,
    QueuePolicy,
    QueueSimulation,
    simulate_queue_policy,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / (
    "experiments/splitfusion_map_freshness_policy_analysis_v1/"
    "20260914_capture_vs_action_clock_v4"
)
DEFAULT_OUTPUT = ROOT / (
    "experiments/splitfusion_map_freshness_policy_analysis_v1/"
    "20260914_budget_quality_latency_v2"
)
SCHEMA = "scenesense.splitfusion.budget_quality_latency.v1"
TERMINAL = "SPLITFUSION_BUDGET_QUALITY_LATENCY_ANALYSIS_COMPLETE"
BUDGETS_MS = (150, 200, 250, 300)
THRESHOLDS = (0.25, 0.50, 0.75)
PRIMARY_THRESHOLD = 0.50
PROFILE_ORDER = (
    "FAVORABLE_STABLE",
    "MID_VARIABLE",
    "ADVERSE_STABLE",
    "FADE_RECOVERY",
)
PROFILE_LABELS = {
    "FAVORABLE_STABLE": "Favorable Stable",
    "MID_VARIABLE": "Mid Variable",
    "ADVERSE_STABLE": "Adverse Stable",
    "FADE_RECOVERY": "Fade Recovery",
}
QUALITY_FIELDS = (
    "val_vehicle_precision",
    "val_vehicle_recall",
    "val_vehicle_f1",
    "val_vehicle_xy_mae_m",
    "val_vehicle_iou",
    "val_canonical_person_precision",
    "val_canonical_person_recall",
    "val_canonical_person_f1",
    "val_canonical_person_xy_mae_m",
    "val_person_box_mask_iou",
    "val_segmentation_miou",
)


class PresentationAnalysisError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PresentationAnalysisError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    require(bool(rows), f"refusing to write empty CSV: {path}")
    fields = list(rows[0])
    require(all(list(row) == fields for row in rows), f"column drift: {path}")
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def finite(value: Any) -> float | None:
    if value in (None, ""):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def median(values: Iterable[float]) -> float | None:
    numbers = [float(value) for value in values if math.isfinite(float(value))]
    return None if not numbers else float(statistics.median(numbers))


def percentile(values: Iterable[float], probability: float) -> float | None:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    index = min(len(ordered) - 1, max(0, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def generalized_map_freshness(
    simulation: QueueSimulation,
    useful: Sequence[QueueOutcome],
) -> dict[str, Any]:
    """Integrate physical map age over the whole observation interval."""

    start = int(simulation.observation_start_ns)
    end = int(simulation.observation_end_ns)
    duration = end - start
    require(duration > 0, "non-positive observation duration")
    installed = [
        item
        for item in useful
        if item.install_ns is not None and int(item.install_ns) < end
    ]
    output: dict[str, Any] = {
        "map_observation_duration_s": duration / 1e9,
        "map_available_fraction": 0.0,
        "time_weighted_map_aoi_ms_when_available": None,
    }
    for budget in BUDGETS_MS:
        output[f"fresh_map_time_ms_le_{budget}_fraction"] = 0.0
    if not installed:
        return output

    first_install = max(start, int(installed[0].install_ns))
    available = max(0, end - first_install)
    output["map_available_fraction"] = available / duration
    area_ns2 = 0.0
    fresh_ns = {budget: 0 for budget in BUDGETS_MS}
    for index, item in enumerate(installed):
        interval_start = max(first_install, int(item.install_ns))
        interval_end = (
            min(end, int(installed[index + 1].install_ns))
            if index + 1 < len(installed)
            else end
        )
        if interval_end <= interval_start:
            continue
        capture_ns = int(item.frame.capture_ns)
        age_start = interval_start - capture_ns
        require(age_start >= 0, "map installation predates capture")
        interval = interval_end - interval_start
        area_ns2 += float(age_start) * interval + 0.5 * float(interval) ** 2
        for budget in BUDGETS_MS:
            crossing = capture_ns + budget * 1_000_000
            fresh_ns[budget] += max(
                0, min(interval_end, crossing) - interval_start
            )
    output["time_weighted_map_aoi_ms_when_available"] = (
        area_ns2 / available / 1e6 if available else None
    )
    for budget in BUDGETS_MS:
        output[f"fresh_map_time_ms_le_{budget}_fraction"] = (
            fresh_ns[budget] / duration
        )
    return output


def installed_stage_summary(
    simulation: QueueSimulation,
    sent: Sequence[Mapping[str, str]],
    bridge_ns: int,
) -> dict[str, Any]:
    """Return medians for additive capture-to-install latency boundaries."""

    useful = prior.useful_installations(simulation.outcomes)
    relevant = [
        item
        for item in useful
        if item.install_ns is not None
        and int(item.install_ns) < int(simulation.observation_end_ns)
    ]
    values: dict[str, list[float]] = defaultdict(list)
    observed_arrivals = 0
    imputed_arrivals = 0
    for item in relevant:
        sequence = int(item.frame.sequence_id)
        row = sent[sequence]
        capture_ns = int(item.frame.capture_ns)
        action_start_ns = int(row["capture_started_ns"]) + bridge_ns
        send_start_ns = int(row["ue_prepare_finished_ns"]) + bridge_ns
        arrival_ns = int(item.frame.arrival_ns)
        compute_start_ns = int(item.compute_start_ns)
        compute_finish_ns = int(item.compute_finish_ns)
        publication_start_ns = int(item.publication_start_ns)
        publication_finish_ns = int(item.publication_finish_ns)
        install_ns = int(item.install_ns)
        boundaries = (
            capture_ns,
            action_start_ns,
            send_start_ns,
            arrival_ns,
            compute_start_ns,
            compute_finish_ns,
            publication_start_ns,
            publication_finish_ns,
            install_ns,
        )
        require(
            all(right >= left for left, right in zip(boundaries, boundaries[1:])),
            "non-causal installed-frame latency boundary",
        )
        component_ns = (
            action_start_ns - capture_ns,
            send_start_ns - action_start_ns,
            arrival_ns - send_start_ns,
            compute_start_ns - arrival_ns,
            compute_finish_ns - compute_start_ns,
            publication_start_ns - compute_finish_ns,
            publication_finish_ns - publication_start_ns,
            install_ns - publication_finish_ns,
        )
        require(
            sum(component_ns) == install_ns - capture_ns,
            "installed-frame latency components do not reconcile",
        )
        values["sensor_pre_action_ms"].append((action_start_ns - capture_ns) / 1e6)
        values["ue_action_path_ms"].append((send_start_ns - action_start_ns) / 1e6)
        values["feature_transfer_reassembly_ms"].append((arrival_ns - send_start_ns) / 1e6)
        values["edge_queue_ms"].append((compute_start_ns - arrival_ns) / 1e6)
        values["edge_compute_ms"].append((compute_finish_ns - compute_start_ns) / 1e6)
        values["publication_queue_ms"].append((publication_start_ns - compute_finish_ns) / 1e6)
        values["edge_publication_ms"].append((publication_finish_ns - publication_start_ns) / 1e6)
        values["edge_processing_ms"].append(
            (
                compute_finish_ns
                - compute_start_ns
                + publication_finish_ns
                - publication_start_ns
            )
            / 1e6
        )
        values["result_return_map_install_ms"].append((install_ns - publication_finish_ns) / 1e6)
        values["edge_to_map_ms"].append((install_ns - arrival_ns) / 1e6)
        values["capture_to_install_ms"].append((install_ns - capture_ns) / 1e6)
        if row.get("edge_receipt_wall_s") not in (None, ""):
            observed_arrivals += 1
        else:
            imputed_arrivals += 1

    output: dict[str, Any] = {
        "useful_installs": len(relevant),
        "observed_arrivals": observed_arrivals,
        "imputed_arrivals": imputed_arrivals,
    }
    for key in (
        "sensor_pre_action_ms",
        "ue_action_path_ms",
        "feature_transfer_reassembly_ms",
        "edge_queue_ms",
        "edge_compute_ms",
        "publication_queue_ms",
        "edge_publication_ms",
        "edge_processing_ms",
        "result_return_map_install_ms",
        "edge_to_map_ms",
        "capture_to_install_ms",
    ):
        output[f"{key}_median"] = median(values[key])
        output[f"{key}_p95"] = percentile(values[key], 0.95)
    return output


def verify_source() -> tuple[dict[tuple[str, str], dict[str, str]], dict[str, str]]:
    hashes = prior.verify_manifest(SOURCE_ROOT)
    rows = read_csv(SOURCE_ROOT / "action_network_dual_clock_freshness.csv")
    keyed = {(row["cell_id"], row["queue_policy"]): row for row in rows}
    require(len(rows) == 576 and len(keyed) == 576, "dual-clock source is not 288 x 2")
    latest = [row for row in rows if row["queue_policy"] == QueuePolicy.LATEST_ONLY_NO_EXPIRY.value]
    require(len(latest) == 288, "dual-clock source lacks 288 latest-only rows")
    return keyed, hashes


def build_cell_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_rows, source_hashes = verify_source()
    cells = source._read_csv(source.CONSOLIDATION / "campaign_288_cell_table.csv")
    require(len(cells) == 288, "campaign table is not 288 cells")
    bridge_ns, clock_audit, per_frame_hashes = timing.collect_clock_bridge(cells)
    quality = source._quality_by_action()
    provenance = final_v3._verify_final_sources()
    calibration, publication_samples = final_v3._final_calibration()
    (
        action_services,
        family_services,
        profile_delays,
        action_profile_arrivals,
        profile_arrivals,
        pool_hashes,
    ) = source._build_empirical_pools(cells)
    require(pool_hashes == per_frame_hashes, "per-frame hash verification drift")

    rows: list[dict[str, Any]] = []
    for number, cell in enumerate(cells, start=1):
        attempt = source._attempt(cell)
        sent = sorted(
            source._sent_rows(attempt), key=lambda row: float(row["capture_wall_s"])
        )
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
        corrected, floored, floor_delta_ns = timing.enforce_imputed_arrival_causality(
            frames, sent, bridge_ns, cell["cell_id"]
        )
        simulation = simulate_queue_policy(
            corrected,
            policy=QueuePolicy.LATEST_ONLY_NO_EXPIRY,
            observation_tail_ns=prior.OBSERVATION_TAIL_NS,
        )
        useful = prior.useful_installations(simulation.outcomes)
        freshness = generalized_map_freshness(simulation, useful)
        stages = installed_stage_summary(simulation, sent, bridge_ns)
        existing = source_rows[
            (cell["cell_id"], QueuePolicy.LATEST_ONLY_NO_EXPIRY.value)
        ]
        for budget in (150, 200, 250):
            key = f"fresh_map_time_ms_le_{budget}_fraction"
            require(
                abs(
                    float(freshness[key])
                    - float(existing[f"physical_{key}"])
                )
                < 1e-9,
                f"{cell['cell_id']}: physical freshness reproduction drift at {budget} ms",
            )
        for key, existing_key in (
            ("map_available_fraction", "physical_map_available_fraction"),
            ("time_weighted_map_aoi_ms_when_available", "physical_map_aoi_ms_when_available"),
        ):
            left = freshness[key]
            right = finite(existing[existing_key])
            if left is None or right is None:
                require(left is None and right is None, f"{cell['cell_id']}: null map metric drift")
            else:
                require(abs(float(left) - float(right)) < 1e-9, f"{cell['cell_id']}: map metric drift")

        row: dict[str, Any] = {
            "cell_id": cell["cell_id"],
            "action_id": int(cell["action_id"]),
            "profile_id": cell["profile_id"],
            "network_profile": cell["network_profile"],
            "family": cell["family"],
            "quantizer": cell["quantizer"],
            "q": float(cell["q"]),
            "median_feature_bytes": finite(existing["median_feature_bytes"]),
            **quality[int(cell["action_id"])],
            **freshness,
            **stages,
            "arrival_observations_floored": floored,
            "arrival_floor_total_ms": floor_delta_ns / 1e6,
            "source_per_frame_sha256": per_frame_hashes[cell["cell_id"]],
            "source_arrival_observed": counters["arrival_observed"],
        }
        rows.append(row)
        if number % 24 == 0:
            print(f"budget/latency simulation: {number}/288 cells", flush=True)

    require(len(rows) == 288, "analysis does not contain 288 rows")
    require(
        {(int(row["action_id"]), row["network_profile"]) for row in rows}
        == {(action, profile) for action in range(72) for profile in PROFILE_ORDER},
        "action/profile inventory drift",
    )
    for row in rows:
        fractions = [
            float(row[f"fresh_map_time_ms_le_{budget}_fraction"])
            for budget in BUDGETS_MS
        ]
        require(
            all(right >= left for left, right in zip(fractions, fractions[1:])),
            f"{row['cell_id']}: freshness is not monotonic in budget",
        )
    audit = {
        "source_manifest_sha256": sha256(SOURCE_ROOT / "artifact_manifest.json"),
        "source_artifacts_verified": len(source_hashes),
        "source_rows_reproduced_at_150_200_250_ms": 288,
        "source_per_frame_hashes_verified": len(per_frame_hashes),
        "clock_bridge": clock_audit,
        "source_verification": provenance,
    }
    return rows, audit


def build_budget_rows(
    cells: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    counts: list[dict[str, Any]] = []
    eligible_detail: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for budget in BUDGETS_MS:
        fresh_key = f"fresh_map_time_ms_le_{budget}_fraction"
        for profile in PROFILE_ORDER:
            profile_rows = sorted(
                (row for row in cells if row["network_profile"] == profile),
                key=lambda row: int(row["action_id"]),
            )
            require(len(profile_rows) == 72, f"{profile}: not 72 actions")
            eligible_by_threshold = {
                threshold: [
                    row
                    for row in profile_rows
                    if float(row[fresh_key]) >= threshold
                ]
                for threshold in THRESHOLDS
            }
            primary = eligible_by_threshold[PRIMARY_THRESHOLD]
            best = max(
                profile_rows,
                key=lambda row: (
                    float(row[fresh_key]),
                    -float(row["median_feature_bytes"] or math.inf),
                ),
            )
            counts.append(
                {
                    "freshness_budget_ms": budget,
                    "network_profile": profile,
                    "total_actions": 72,
                    "actions_fresh_at_least_25pct_route": len(eligible_by_threshold[0.25]),
                    "actions_meeting_budget_majority_route": len(primary),
                    "actions_fresh_at_least_75pct_route": len(eligible_by_threshold[0.75]),
                    "eligible_action_ids_majority_route": "|".join(
                        str(int(row["action_id"])) for row in primary
                    ),
                    "best_raw_freshness_action_id": int(best["action_id"]),
                    "best_raw_freshness_profile_id": best["profile_id"],
                    "best_raw_fresh_map_fraction": float(best[fresh_key]),
                    "best_vehicle_f1": best["val_vehicle_f1"],
                    "best_person_f1": best["val_canonical_person_f1"],
                    "best_segmentation_miou": best["val_segmentation_miou"],
                }
            )
            for row in primary:
                eligible_detail.append(
                    {
                        "freshness_budget_ms": budget,
                        "network_profile": profile,
                        "fresh_map_fraction": row[fresh_key],
                        "action_id": row["action_id"],
                        "profile_id": row["profile_id"],
                        "family": row["family"],
                        "quantizer": row["quantizer"],
                        "q": row["q"],
                        "median_feature_bytes": row["median_feature_bytes"],
                        **{field: row[field] for field in QUALITY_FIELDS},
                    }
                )
            for row in profile_rows:
                vehicle_f1 = float(row["val_vehicle_f1"] or 0.0)
                person_f1 = float(row["val_canonical_person_f1"] or 0.0)
                balanced_f1 = 0.5 * (vehicle_f1 + person_f1)
                candidates.append(
                    {
                        "freshness_budget_ms": budget,
                        "network_profile": profile,
                        "action_id": row["action_id"],
                        "profile_id": row["profile_id"],
                        "median_feature_bytes": row["median_feature_bytes"],
                        "fresh_map_fraction": row[fresh_key],
                        "meets_budget_majority_route": float(row[fresh_key]) >= PRIMARY_THRESHOLD,
                        "vehicle_f1": vehicle_f1,
                        "person_f1": person_f1,
                        "balanced_f1": balanced_f1,
                        "vehicle_f1_x_freshness": vehicle_f1 * float(row[fresh_key]),
                        "person_f1_x_freshness": person_f1 * float(row[fresh_key]),
                        "balanced_f1_x_freshness": balanced_f1 * float(row[fresh_key]),
                    }
                )
    candidates.sort(
        key=lambda row: (
            BUDGETS_MS.index(int(row["freshness_budget_ms"])),
            PROFILE_ORDER.index(str(row["network_profile"])),
            -float(row["balanced_f1_x_freshness"]),
            int(row["action_id"]),
        )
    )
    ranked: list[dict[str, Any]] = []
    group_counts: dict[tuple[int, str], int] = defaultdict(int)
    for row in candidates:
        key = (int(row["freshness_budget_ms"]), str(row["network_profile"]))
        group_counts[key] += 1
        if group_counts[key] <= 5:
            ranked.append({"rank": group_counts[key], **row})
    return counts, eligible_detail, ranked


def build_latency_rows(
    cells: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fields = (
        "sensor_pre_action_ms_median",
        "ue_action_path_ms_median",
        "feature_transfer_reassembly_ms_median",
        "edge_queue_ms_median",
        "edge_compute_ms_median",
        "publication_queue_ms_median",
        "edge_publication_ms_median",
        "edge_processing_ms_median",
        "result_return_map_install_ms_median",
        "edge_to_map_ms_median",
        "capture_to_install_ms_median",
    )
    action_rows: list[dict[str, Any]] = []
    for row in cells:
        action_rows.append(
            {
                "network_profile": row["network_profile"],
                "action_id": row["action_id"],
                "profile_id": row["profile_id"],
                "median_feature_bytes": row["median_feature_bytes"],
                "useful_installs": row["useful_installs"],
                "observed_arrivals": row["observed_arrivals"],
                "imputed_arrivals": row["imputed_arrivals"],
                **{field: row[field] for field in fields},
            }
        )
    profile_rows: list[dict[str, Any]] = []
    for profile in PROFILE_ORDER:
        selected = [
            row
            for row in action_rows
            if row["network_profile"] == profile and int(row["useful_installs"]) > 0
        ]
        require(bool(selected), f"{profile}: no useful installations")
        aggregate = {
            "network_profile": profile,
            "actions_with_useful_installs": len(selected),
            "useful_installs": sum(int(row["useful_installs"]) for row in selected),
            "observed_arrivals": sum(int(row["observed_arrivals"]) for row in selected),
            "imputed_arrivals": sum(int(row["imputed_arrivals"]) for row in selected),
        }
        for field in fields:
            aggregate[field.replace("_median", "_action_balanced_median_ms")] = median(
                float(row[field]) for row in selected if row[field] is not None
            )
        aggregate["additive_component_sum_ms"] = sum(
            float(aggregate[key])
            for key in (
                "sensor_pre_action_ms_action_balanced_median_ms",
                "ue_action_path_ms_action_balanced_median_ms",
                "feature_transfer_reassembly_ms_action_balanced_median_ms",
                "edge_queue_ms_action_balanced_median_ms",
                "edge_processing_ms_action_balanced_median_ms",
                "result_return_map_install_ms_action_balanced_median_ms",
            )
        )
        profile_rows.append(aggregate)
    return profile_rows, action_rows


def configure_axes(ax: Any) -> None:
    ax.tick_params(axis="both", labelsize=11, width=1.4)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


def plot_budget_counts(output: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    matrix = np.zeros((len(PROFILE_ORDER), len(BUDGETS_MS)), dtype=float)
    best = np.zeros_like(matrix)
    for row in rows:
        i = PROFILE_ORDER.index(str(row["network_profile"]))
        j = BUDGETS_MS.index(int(row["freshness_budget_ms"]))
        matrix[i, j] = int(row["actions_meeting_budget_majority_route"])
        best[i, j] = float(row["best_raw_fresh_map_fraction"])
    figure, ax = plt.subplots(figsize=(10.8, 5.7), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=72, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] >= 38 else "black"
            ax.text(
                j,
                i,
                f"{int(matrix[i, j])}/72\nbest {100 * best[i, j]:.1f}%",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=color,
            )
    ax.set_xticks(range(len(BUDGETS_MS)), [f"{value} ms" for value in BUDGETS_MS])
    ax.set_yticks(range(len(PROFILE_ORDER)), [PROFILE_LABELS[value] for value in PROFILE_ORDER])
    ax.set_xlabel("Physical map-freshness budget", fontsize=12, fontweight="bold")
    ax.set_ylabel("Network profile", fontsize=12, fontweight="bold")
    ax.set_title(
        "Actions keeping map age within budget for at least 50% of route time",
        fontsize=14,
        fontweight="bold",
    )
    colorbar = figure.colorbar(image, ax=ax, label="Qualifying actions (of 72)")
    colorbar.ax.tick_params(labelsize=10, width=1.2)
    configure_axes(ax)
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"01_budget_action_counts.{suffix}", dpi=300)
    plt.close(figure)


def plot_latency_breakdown(output: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    labels = [PROFILE_LABELS[str(row["network_profile"])] for row in rows]
    components = (
        ("Sensor/pre-action", "sensor_pre_action_ms_action_balanced_median_ms", "#4C78A8"),
        ("UE action path", "ue_action_path_ms_action_balanced_median_ms", "#F58518"),
        ("Feature transfer + reassembly", "feature_transfer_reassembly_ms_action_balanced_median_ms", "#54A24B"),
        ("Edge queue", "edge_queue_ms_action_balanced_median_ms", "#B279A2"),
        ("Edge compute + publication", "edge_processing_ms_action_balanced_median_ms", "#E45756"),
        ("Result return + map install", "result_return_map_install_ms_action_balanced_median_ms", "#72B7B2"),
    )
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    figure, ax = plt.subplots(figsize=(11.5, 6.4), constrained_layout=True)
    for label, key, color in components:
        values = np.array([float(row[key]) for row in rows])
        ax.bar(x, values, bottom=bottom, label=label, color=color, width=0.68)
        for index, (base, value) in enumerate(zip(bottom, values)):
            if value >= 13:
                ax.text(
                    index,
                    base + value / 2,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="white" if color in {"#4C78A8", "#B279A2", "#E45756"} else "black",
                )
        bottom += values
    direct = np.array([float(row["capture_to_install_ms_action_balanced_median_ms"]) for row in rows])
    ax.scatter(x, direct, marker="D", s=70, color="black", label="Direct capture-to-install median", zorder=5)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Latency (ms)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Network profile", fontsize=12, fontweight="bold")
    ax.set_title("Optimized latest-only end-to-end latency decomposition", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, ncols=2, frameon=False, loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    configure_axes(ax)
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"02_latency_breakdown_by_network_profile.{suffix}", dpi=300)
    plt.close(figure)


def quality_range(rows: Sequence[Mapping[str, Any]], key: str) -> str:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return "—"
    return f"{min(values):.3f}–{max(values):.3f}"


def presentation_tables(
    counts: Sequence[Mapping[str, Any]],
    eligible: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    latency: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# Map-freshness budget and latency presentation tables",
        "",
        "## Definition used",
        "",
        "An action **meets a physical map-freshness budget** here when its",
        "counterfactual map age is no greater than the budget for at least 50%",
        "of the complete Route-B observation time. This is majority-time",
        "compliance, not a hard per-frame or worst-case guarantee. The simulator",
        "uses measured 288-cell transport/reassembly behavior, final optimized",
        "edge calibration, and strict latest-only/no-expiry scheduling.",
        "",
        "## Number of eligible actions",
        "",
        "Each entry is `actions / 72`; the parenthesis gives the best fraction",
        "of route time achieved by any action at that budget.",
        "",
        "| Budget | Favorable Stable | Mid Variable | Adverse Stable | Fade Recovery |",
        "|---:|---:|---:|---:|---:|",
    ]
    by_key = {(int(row["freshness_budget_ms"]), str(row["network_profile"])): row for row in counts}
    for budget in BUDGETS_MS:
        values = []
        for profile in PROFILE_ORDER:
            row = by_key[(budget, profile)]
            values.append(
                f"{row['actions_meeting_budget_majority_route']}/72 "
                f"({100 * float(row['best_raw_fresh_map_fraction']):.1f}%)"
            )
        lines.append(f"| {budget} ms | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Sensitivity to the required route-time fraction",
            "",
            "| Budget | Network profile | ≥25% of time | ≥50% of time (primary) | ≥75% of time | Best action | Best fresh time |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for budget in BUDGETS_MS:
        for profile in PROFILE_ORDER:
            row = by_key[(budget, profile)]
            lines.append(
                f"| {budget} ms | {PROFILE_LABELS[profile]} | "
                f"{row['actions_fresh_at_least_25pct_route']} | "
                f"{row['actions_meeting_budget_majority_route']} | "
                f"{row['actions_fresh_at_least_75pct_route']} | "
                f"{row['best_raw_freshness_action_id']} | "
                f"{100 * float(row['best_raw_fresh_map_fraction']):.1f}% |"
            )

    lines.extend(
        [
            "",
            "## Model performance of majority-time-eligible actions",
            "",
            "The exact action-by-action values are in",
            "`eligible_action_model_performance.csv`. These rows summarize the",
            "quality range across the eligible action set.",
            "",
            "| Budget | Network profile | Actions | Action IDs | Vehicle F1 | Person F1 | Segmentation mIoU | Vehicle XY MAE (m) | Person XY MAE (m) |",
            "|---:|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for budget in BUDGETS_MS:
        for profile in PROFILE_ORDER:
            selected = [
                row
                for row in eligible
                if int(row["freshness_budget_ms"]) == budget
                and row["network_profile"] == profile
            ]
            ids = ", ".join(str(row["action_id"]) for row in selected) or "—"
            lines.append(
                f"| {budget} ms | {PROFILE_LABELS[profile]} | {len(selected)} | {ids} | "
                f"{quality_range(selected, 'val_vehicle_f1')} | "
                f"{quality_range(selected, 'val_canonical_person_f1')} | "
                f"{quality_range(selected, 'val_segmentation_miou')} | "
                f"{quality_range(selected, 'val_vehicle_xy_mae_m')} | "
                f"{quality_range(selected, 'val_canonical_person_xy_mae_m')} |"
            )

    lines.extend(
        [
            "",
            "## Best balanced quality–freshness candidate",
            "",
            "This is a diagnostic ranking by",
            "`0.5 × (vehicle F1 + person F1) × fresh-map fraction`; it is not",
            "yet the final RL reward. A `no` in the final column means the best",
            "trade-off candidate still does not satisfy the primary majority-time",
            "rule.",
            "",
            "| Budget | Network profile | Action | Fresh time | Vehicle F1 | Person F1 | Quality × freshness | Meets majority-time rule |",
            "|---:|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for budget in BUDGETS_MS:
        for profile in PROFILE_ORDER:
            row = next(
                item
                for item in candidates
                if int(item["freshness_budget_ms"]) == budget
                and item["network_profile"] == profile
                and int(item["rank"]) == 1
            )
            lines.append(
                f"| {budget} ms | {PROFILE_LABELS[profile]} | {row['action_id']} | "
                f"{100 * float(row['fresh_map_fraction']):.1f}% | "
                f"{float(row['vehicle_f1']):.3f} | {float(row['person_f1']):.3f} | "
                f"{float(row['balanced_f1_x_freshness']):.3f} | "
                f"{'yes' if row['meets_budget_majority_route'] else 'no'} |"
            )

    lines.extend(
        [
            "",
            "## End-to-end latency by network profile",
            "",
            "Values are medians across the per-action cell medians, so each",
            "action has equal weight. Only actions with at least one useful map",
            "installation contribute. The direct end-to-end median is computed",
            "independently; therefore it need not equal the sum of marginal",
            "component medians exactly.",
            "",
            "| Network profile | Actions | Sensor/pre-action | UE action path | Feature transfer + reassembly | Edge queue | Edge compute + publication | Result return + map install | Direct capture → install |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in latency:
        lines.append(
            f"| {PROFILE_LABELS[str(row['network_profile'])]} | "
            f"{row['actions_with_useful_installs']} | "
            f"{row['sensor_pre_action_ms_action_balanced_median_ms']:.1f} ms | "
            f"{row['ue_action_path_ms_action_balanced_median_ms']:.1f} ms | "
            f"{row['feature_transfer_reassembly_ms_action_balanced_median_ms']:.1f} ms | "
            f"{row['edge_queue_ms_action_balanced_median_ms']:.1f} ms | "
            f"{row['edge_processing_ms_action_balanced_median_ms']:.1f} ms | "
            f"{row['result_return_map_install_ms_action_balanced_median_ms']:.1f} ms | "
            f"{row['capture_to_install_ms_action_balanced_median_ms']:.1f} ms |"
        )

    lines.extend(
        [
            "",
            "### What the six stages mean",
            "",
            "- **Sensor/pre-action:** RGB capture to entry into seven-channel",
            "  tensor assembly. It includes radar-window extraction and other",
            "  work before the selected split action can influence the frame.",
            "- **UE action path:** tensor-assembly entry to transmission start.",
            "  This includes final input assembly, the selected front/ranker/AE",
            "  path, feature serialization, and dispatch preparation; it is not",
            "  the pure backbone CUDA time alone.",
            "- **Feature transfer + reassembly:** UE transmission start to a",
            "  complete feature at the edge. This includes the sender loop, OAI/",
            "  RFsim uplink, and application reassembly; it is not PHY-only time.",
            "- **Edge queue:** wait after complete feature reassembly before edge",
            "  work starts. It is zero for the installed frames under strict",
            "  latest-only scheduling; superseded pending frames are not installs.",
            "- **Edge compute + publication:** reconstruction, optimized tail",
            "  service/post-processing, and compact-result serialization. This is",
            "  the edge-processing column; it excludes return transport and map",
            "  installation.",
            "- **Result return + map install:** compact result delivery after edge",
            "  publication plus receiver/map installation. The retained timestamps",
            "  do not split those two contributions further.",
            "",
            "## Discussion sequence for the supervisor",
            "",
            "1. Start with the budget-count table: tighter freshness budgets",
            "   sharply restrict the feasible action set, especially in adverse",
            "   conditions.",
            "2. Use the eligible-quality table to show the policy problem: among",
            "   actions that are fresh often enough, choose the one providing the",
            "   best person/vehicle utility—not simply the smallest payload.",
            "3. Show the latency breakdown. Network-sensitive transfer changes by",
            "   profile; sensor/pre-action and much of edge compute are outside the",
            "   action's immediate control but still determine physical map age.",
            "4. Explain that the agent must observe input age at decision time, so",
            "   it is not blamed for already-consumed preparation time. Reward the",
            "   resulting physical map utility, while using action-clock timing for",
            "   attribution and diagnosis.",
            "5. Ask whether the control objective should require 50% route-time",
            "   compliance, a stricter fraction, or a soft freshness reward. The",
            "   tables deliberately expose that choice rather than hiding it.",
            "",
            "## Limits",
            "",
            "This is an offline counterfactual, not a new 288-cell live run.",
            "Observed and deterministically imputed edge arrivals are both used as",
            "in the bound source simulator. Model validation quality is action-",
            "specific and therefore repeats across network profiles; network",
            "conditions alter freshness and feasibility, not the frozen validation",
            "score itself.",
            "",
        ]
    )
    return "\n".join(lines)


def run(output: Path) -> dict[str, Any]:
    require(not output.exists(), f"create-only output already exists: {output}")
    cells, provenance = build_cell_rows()
    counts, eligible, candidates = build_budget_rows(cells)
    latency_profile, latency_action = build_latency_rows(cells)
    require(len(counts) == 16, "budget-count table is not 4 x 4")
    require(len(candidates) == 80, "top-candidate table is not 4 x 4 x 5")
    require(len(latency_profile) == 4 and len(latency_action) == 288, "latency inventory drift")

    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "budget_action_counts.csv", counts)
    if eligible:
        write_csv(output / "eligible_action_model_performance.csv", eligible)
    else:
        # The file remains a usable, explicit table even if the primary rule
        # produces no eligible action anywhere.
        fields = [
            "freshness_budget_ms",
            "network_profile",
            "fresh_map_fraction",
            "action_id",
            "profile_id",
            "family",
            "quantizer",
            "q",
            "median_feature_bytes",
            *QUALITY_FIELDS,
        ]
        temporary = output / "eligible_action_model_performance.csv.partial"
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fields, lineterminator="\n").writeheader()
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output / "eligible_action_model_performance.csv")
    write_csv(output / "top5_quality_freshness_candidates.csv", candidates)
    write_csv(output / "latency_breakdown_by_network_profile.csv", latency_profile)
    write_csv(output / "latency_breakdown_by_action_network.csv", latency_action)
    atomic_text(
        output / "PRESENTATION_TABLES.md",
        presentation_tables(counts, eligible, candidates, latency_profile),
    )
    plot_budget_counts(output, counts)
    plot_latency_breakdown(output, latency_profile)

    document: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "scientific_status": "OFFLINE_COUNTERFACTUAL_NOT_LIVE_REMEASUREMENT",
        "budgets_ms": list(BUDGETS_MS),
        "primary_budget_rule": {
            "clock": "PHYSICAL_CAPTURE_CLOCK",
            "minimum_fraction_of_complete_observation": PRIMARY_THRESHOLD,
            "meaning": "physical map age is within budget for at least half of route time",
            "not_a_claim": "not a per-frame, worst-case, or always-fresh guarantee",
        },
        "queue_policy": QueuePolicy.LATEST_ONLY_NO_EXPIRY.value,
        "cell_count": len(cells),
        "budget_count_rows": len(counts),
        "eligible_action_rows": len(eligible),
        "top_candidate_rows": len(candidates),
        "latency_profile_rows": len(latency_profile),
        "latency_action_rows": len(latency_action),
        "provenance": provenance,
        "interpretation_limits": [
            "counterfactual replay over measured 288-cell transport and calibrated optimized edge service",
            "model validation quality is frozen per action and does not vary by network profile",
            "feature transfer includes UE sender and application reassembly and is not PHY-only latency",
            "edge receipt to map install includes queue, compute, publication, and installation",
            "latency profile rows are action-balanced medians of per-cell medians",
        ],
    }
    atomic_json(output / "analysis.json", document)
    return document


def finalize(output: Path, document: Mapping[str, Any]) -> None:
    names = (
        "budget_action_counts.csv",
        "eligible_action_model_performance.csv",
        "top5_quality_freshness_candidates.csv",
        "latency_breakdown_by_network_profile.csv",
        "latency_breakdown_by_action_network.csv",
        "PRESENTATION_TABLES.md",
        "01_budget_action_counts.png",
        "01_budget_action_counts.pdf",
        "02_latency_breakdown_by_network_profile.png",
        "02_latency_breakdown_by_network_profile.pdf",
        "analysis.json",
    )
    hashes = {name: sha256(output / name) for name in names}
    atomic_json(
        output / "artifact_manifest.json",
        {"schema": f"{SCHEMA}.artifacts", "status": "COMPLETE", "sha256": hashes},
    )
    atomic_text(
        output / TERMINAL,
        json.dumps(
            {
                "schema": f"{SCHEMA}.terminal",
                "status": document["status"],
                "analysis_sha256": hashes["analysis.json"],
                "budget_table_sha256": hashes["budget_action_counts.csv"],
                "latency_table_sha256": hashes["latency_breakdown_by_network_profile.csv"],
            },
            sort_keys=True,
        )
        + "\n",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output.resolve()
    document = run(output)
    finalize(output, document)
    print(TERMINAL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
