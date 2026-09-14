#!/usr/bin/env python3
"""Presentation artifacts for the corrected direct edge-to-map architecture.

Six deliverables, each as PNG + PDF plus a CSV, with a Markdown table sheet:

1. actions meeting each 150/200/250/300 ms freshness budget, by network profile
   (primary rule: physical map age within budget for >=50% of route time, with
   25% and 75% sensitivity);
2. exact model quality for every eligible action;
3. matched-action analysis -- the same action across all four network profiles;
4. conditional uplink/reassembly latency, never shown without its success
   probability and zero-delivery indication;
5. the corrected end-to-end decomposition, with map-feedback arrival drawn
   outside physical map AoI;
6. before/after -- the old edge->UE->map detour versus the corrected path.

Simulator results are labelled counterfactual; short-run values are labelled
live measurements. The two are never merged into one number.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rl_agent.splitfusion_edge_freshness_scheduler_v1 import counterfactual_288 as source


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "scenesense.splitfusion.direct_edge_map_presentation.v1"
TERMINAL = "SPLITFUSION_DIRECT_EDGE_MAP_PRESENTATION_COMPLETE"
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
UPSTREAM_LATENCY_CSV = ROOT / (
    "experiments/splitfusion_map_freshness_policy_analysis_v1/"
    "20260914_budget_quality_latency_v2/latency_breakdown_by_network_profile.csv"
)
UPSTREAM_LATENCY_SHA256 = (
    "" # resolved and recorded at run time; the file is upstream and unmodified
)
PALETTE = {
    "sensor": "#4C78A8",
    "ue": "#F58518",
    "uplink": "#54A24B",
    "queue": "#B279A2",
    "edge": "#E45756",
    "install": "#72B7B2",
    "feedback": "#9D755D",
    "old": "#E45756",
    "new": "#4C78A8",
}


def require(condition: bool, message: str) -> None:
    source._require(condition, message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    require(bool(rows), f"refusing to write an empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def f(value: Any) -> float | None:
    return source._f(value)


def configure_axes(ax: Any) -> None:
    ax.tick_params(axis="both", labelsize=11, width=1.4)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


def save(figure: Any, output: Path, stem: str) -> None:
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"{stem}.{suffix}", dpi=300)
    plt.close(figure)


# ---------------------------------------------------------------------------
# inputs


def load_counterfactual(root: Path) -> tuple[
    list[dict[str, str]], list[dict[str, str]], dict[str, Any]
]:
    manifest = source._load_json(root / "artifact_manifest.json")
    document = source._load_json(root / "counterfactual_results.json")
    for name in ("direct_map_288_cell_summary.csv", "baseline_288_cell_summary.csv"):
        path = root / name
        require(path.is_file(), f"counterfactual artifact absent: {path}")
        require(
            manifest["sha256"][name] == source._sha256(path),
            f"counterfactual artifact hash drift: {name}",
        )
    direct = read_csv(root / "direct_map_288_cell_summary.csv")
    baseline = read_csv(root / "baseline_288_cell_summary.csv")
    require(len(direct) == 288 and len(baseline) == 288, "counterfactual is not 288 cells")
    return direct, baseline, document


def load_live(root: Path) -> dict[str, Any]:
    manifest = source._load_json(root / "artifact_manifest.json")
    results = root / "DIRECT_VALIDATION_RESULTS.json"
    require(
        manifest["sha256"]["DIRECT_VALIDATION_RESULTS.json"]
        == source._sha256(results),
        "live validation hash drift",
    )
    return source._load_json(results)


# ---------------------------------------------------------------------------
# 1. freshness budgets


def build_budget_rows(direct: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for budget in BUDGETS_MS:
        key = f"fresh_map_time_ms_le_{budget}_fraction"
        for profile in PROFILE_ORDER:
            cells = [row for row in direct if row["network_profile"] == profile]
            require(len(cells) == 72, f"{profile} does not hold 72 actions")
            fractions = {int(row["action_id"]): (f(row[key]) or 0.0) for row in cells}
            best_action = max(fractions, key=lambda item: fractions[item])
            entry: dict[str, Any] = {
                "freshness_budget_ms": budget,
                "network_profile": profile,
                "network_profile_label": PROFILE_LABELS[profile],
                "actions_total": len(cells),
                "best_action_id": best_action,
                "best_fresh_map_fraction": fractions[best_action],
            }
            for threshold in THRESHOLDS:
                qualifying = [
                    action for action, value in fractions.items() if value >= threshold
                ]
                suffix = f"{int(threshold * 100)}pct_route"
                entry[f"actions_meeting_budget_{suffix}"] = len(qualifying)
                if threshold == PRIMARY_THRESHOLD:
                    entry["actions_meeting_budget_primary"] = len(qualifying)
                    entry["qualifying_action_ids_primary"] = ";".join(
                        str(value) for value in sorted(qualifying)
                    )
            rows.append(entry)
    return rows


def plot_budget_counts(output: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    matrix = np.zeros((len(PROFILE_ORDER), len(BUDGETS_MS)))
    best = np.zeros_like(matrix)
    for row in rows:
        i = PROFILE_ORDER.index(str(row["network_profile"]))
        j = BUDGETS_MS.index(int(row["freshness_budget_ms"]))
        matrix[i, j] = int(row["actions_meeting_budget_primary"])
        best[i, j] = float(row["best_fresh_map_fraction"])
    figure, ax = plt.subplots(figsize=(10.8, 5.7), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=72, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i,
                f"{int(matrix[i, j])}/72\nbest {100 * best[i, j]:.1f}%",
                ha="center", va="center", fontsize=11, fontweight="bold",
                color="white" if matrix[i, j] >= 38 else "black",
            )
    ax.set_xticks(range(len(BUDGETS_MS)), [f"{value} ms" for value in BUDGETS_MS])
    ax.set_yticks(
        range(len(PROFILE_ORDER)), [PROFILE_LABELS[value] for value in PROFILE_ORDER]
    )
    ax.set_xlabel("Physical map-freshness budget", fontsize=12, fontweight="bold")
    ax.set_ylabel("Network profile", fontsize=12, fontweight="bold")
    ax.set_title(
        "Direct edge-to-map: actions keeping map age within budget\n"
        "for at least 50% of route time (counterfactual)",
        fontsize=13, fontweight="bold",
    )
    colorbar = figure.colorbar(image, ax=ax, label="Qualifying actions (of 72)")
    colorbar.ax.tick_params(labelsize=10, width=1.2)
    configure_axes(ax)
    save(figure, output, "01_budget_action_counts")


def plot_budget_sensitivity(output: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    figure, axes = plt.subplots(
        1, len(THRESHOLDS), figsize=(15.0, 4.8), constrained_layout=True, sharey=True
    )
    for ax, threshold in zip(axes, THRESHOLDS):
        key = f"actions_meeting_budget_{int(threshold * 100)}pct_route"
        width = 0.2
        x = np.arange(len(BUDGETS_MS))
        for index, profile in enumerate(PROFILE_ORDER):
            values = [
                int(row[key])
                for budget in BUDGETS_MS
                for row in rows
                if int(row["freshness_budget_ms"]) == budget
                and row["network_profile"] == profile
            ]
            ax.bar(
                x + (index - 1.5) * width, values, width=width,
                label=PROFILE_LABELS[profile],
            )
        ax.set_xticks(x, [f"{value}" for value in BUDGETS_MS])
        ax.set_xlabel("Budget (ms)", fontsize=11, fontweight="bold")
        ax.set_title(
            f"{int(threshold * 100)}% of route time"
            + (" (primary)" if threshold == PRIMARY_THRESHOLD else ""),
            fontsize=12, fontweight="bold",
        )
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        configure_axes(ax)
    axes[0].set_ylabel("Qualifying actions (of 72)", fontsize=11, fontweight="bold")
    axes[-1].legend(fontsize=9, frameon=False)
    figure.suptitle(
        "Freshness-budget sensitivity to the route-time threshold (counterfactual)",
        fontsize=13, fontweight="bold",
    )
    save(figure, output, "01b_budget_threshold_sensitivity")


# ---------------------------------------------------------------------------
# 2. model quality


def build_quality_rows(
    direct: Sequence[Mapping[str, str]], budget_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    eligible: dict[int, set[str]] = {}
    for row in budget_rows:
        ids = str(row.get("qualifying_action_ids_primary") or "")
        for value in (item for item in ids.split(";") if item):
            eligible.setdefault(int(value), set()).add(
                f"{row['network_profile']}@{row['freshness_budget_ms']}ms"
            )
    by_action: dict[int, dict[str, str]] = {}
    for row in direct:
        by_action.setdefault(int(row["action_id"]), row)
    rows: list[dict[str, Any]] = []
    for action_id in sorted(by_action):
        cell = by_action[action_id]
        entry: dict[str, Any] = {
            "action_id": action_id,
            "profile_id": cell["profile_id"],
            "family": cell["family"],
            "quantizer": cell["quantizer"],
            "q": f(cell["q"]),
            "keep_count": cell["keep_count"],
            "eligible_any_budget": int(action_id in eligible),
            "eligible_contexts": len(eligible.get(action_id, ())),
            "eligible_context_list": ";".join(sorted(eligible.get(action_id, ()))),
        }
        for field in QUALITY_FIELDS:
            entry[field] = f(cell.get(field, ""))
        rows.append(entry)
    return rows


def plot_quality(output: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    eligible = [row for row in rows if int(row["eligible_any_budget"]) == 1]
    require(bool(eligible), "no action is eligible at any budget")
    eligible = sorted(eligible, key=lambda row: -(row["val_vehicle_f1"] or 0.0))
    figure, axes = plt.subplots(1, 3, figsize=(16.0, 6.2), constrained_layout=True)
    x = np.arange(len(eligible))
    labels = [str(row["action_id"]) for row in eligible]

    ax = axes[0]
    ax.bar(x - 0.2, [row["val_vehicle_f1"] or 0.0 for row in eligible], width=0.4,
           label="Vehicle F1", color=PALETTE["new"])
    ax.bar(x + 0.2, [row["val_canonical_person_f1"] or 0.0 for row in eligible],
           width=0.4, label="Person F1", color=PALETTE["ue"])
    ax.set_ylabel("F1", fontsize=11, fontweight="bold")
    ax.set_title("Detection F1", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, frameon=False)

    ax = axes[1]
    ax.bar(x - 0.2, [row["val_vehicle_xy_mae_m"] or 0.0 for row in eligible], width=0.4,
           label="Vehicle XY MAE", color=PALETTE["new"])
    ax.bar(x + 0.2, [row["val_canonical_person_xy_mae_m"] or 0.0 for row in eligible],
           width=0.4, label="Person XY MAE", color=PALETTE["ue"])
    ax.set_ylabel("Localization error (m)", fontsize=11, fontweight="bold")
    ax.set_title("Localization error", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, frameon=False)

    ax = axes[2]
    ax.bar(x - 0.2, [row["val_vehicle_iou"] or 0.0 for row in eligible], width=0.4,
           label="Vehicle IoU", color=PALETTE["new"])
    ax.bar(x + 0.2, [row["val_segmentation_miou"] or 0.0 for row in eligible],
           width=0.4, label="Segmentation mIoU", color=PALETTE["uplink"])
    ax.set_ylabel("Overlap", fontsize=11, fontweight="bold")
    ax.set_title("Overlap and segmentation", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, frameon=False)

    for ax in axes:
        step = max(1, len(labels) // 24)
        ax.set_xticks(x[::step], labels[::step], rotation=90)
        ax.set_xlabel("Action ID", fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        configure_axes(ax)
    figure.suptitle(
        "Exact frozen model quality for every budget-eligible action "
        "(quality is action-intrinsic and does not vary with the network)",
        fontsize=13, fontweight="bold",
    )
    save(figure, output, "02_eligible_action_model_quality")


# ---------------------------------------------------------------------------
# 3. matched-action network analysis


def build_matched_rows(direct: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    by_action: dict[int, dict[str, Mapping[str, str]]] = {}
    for row in direct:
        by_action.setdefault(int(row["action_id"]), {})[row["network_profile"]] = row
    rows: list[dict[str, Any]] = []
    for action_id in sorted(by_action):
        cells = by_action[action_id]
        require(
            set(cells) == set(PROFILE_ORDER),
            f"action {action_id} lacks all four network profiles",
        )
        entry: dict[str, Any] = {
            "action_id": action_id,
            "profile_id": cells[PROFILE_ORDER[0]]["profile_id"],
            "family": cells[PROFILE_ORDER[0]]["family"],
        }
        aois: list[float] = []
        for profile in PROFILE_ORDER:
            cell = cells[profile]
            aoi = f(cell.get("time_weighted_map_aoi_ms", ""))
            entry[f"{profile}_time_weighted_map_aoi_ms"] = aoi
            entry[f"{profile}_rate_installed_per_sent"] = f(
                cell.get("rate_installed_per_sent", "")
            )
            entry[f"{profile}_fresh_map_fraction_200ms"] = f(
                cell.get("fresh_map_time_ms_le_200_fraction", "")
            )
            entry[f"{profile}_complete_reassembly_per_sent"] = f(
                cell.get("complete_reassembly_per_sent", "")
            )
            if aoi is not None:
                aois.append(aoi)
        if len(aois) == len(PROFILE_ORDER):
            entry["aoi_spread_ms"] = max(aois) - min(aois)
            entry["aoi_worst_profile"] = PROFILE_ORDER[
                max(range(len(aois)), key=lambda index: aois[index])
            ]
        rows.append(entry)
    return rows


def plot_matched_actions(output: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda row: int(row["action_id"]))
    x = np.arange(len(ordered))
    figure, axes = plt.subplots(2, 1, figsize=(15.5, 8.6), constrained_layout=True, sharex=True)
    for profile in PROFILE_ORDER:
        axes[0].plot(
            x,
            [row.get(f"{profile}_time_weighted_map_aoi_ms") or np.nan for row in ordered],
            marker="o", markersize=3, linewidth=1.3, label=PROFILE_LABELS[profile],
        )
        axes[1].plot(
            x,
            [row.get(f"{profile}_rate_installed_per_sent") or np.nan for row in ordered],
            marker="o", markersize=3, linewidth=1.3, label=PROFILE_LABELS[profile],
        )
    axes[0].set_ylabel("Time-weighted map AoI (ms)", fontsize=11, fontweight="bold")
    axes[0].set_title(
        "Matched action across all four network profiles (counterfactual)",
        fontsize=13, fontweight="bold",
    )
    axes[1].set_ylabel("Installed / sent", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Action ID", fontsize=11, fontweight="bold")
    step = max(1, len(ordered) // 24)
    axes[1].set_xticks(x[::step], [str(row["action_id"]) for row in ordered][::step], rotation=90)
    for ax in axes:
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        ax.legend(fontsize=9, ncols=4, frameon=False)
        configure_axes(ax)
    save(figure, output, "03_matched_action_across_profiles")


# ---------------------------------------------------------------------------
# 4. conditional latency paired with delivery


def build_conditional_rows(direct: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for profile in PROFILE_ORDER:
        cells = [row for row in direct if row["network_profile"] == profile]
        for cell in cells:
            rows.append(
                {
                    "network_profile": profile,
                    "network_profile_label": PROFILE_LABELS[profile],
                    "action_id": int(cell["action_id"]),
                    "family": cell["family"],
                    "conditional_uplink_reassembly_ms_median": f(
                        cell.get("stage_uplink_and_reassembly_ms_median", "")
                    ),
                    "conditional_uplink_reassembly_ms_p95": f(
                        cell.get("stage_uplink_and_reassembly_ms_p95", "")
                    ),
                    "conditional_sample_count": f(
                        cell.get("stage_uplink_and_reassembly_count", "")
                    ),
                    "complete_reassembly_per_sent": f(
                        cell.get("complete_reassembly_per_sent", "")
                    ),
                    "edge_admission_per_sent": f(cell.get("edge_admission_per_sent", "")),
                    "zero_delivery": int(f(cell.get("zero_delivery", "")) or 0),
                    "installed_per_sent": f(cell.get("rate_installed_per_sent", "")),
                }
            )
    return rows


def plot_conditional_latency(output: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15.0, 6.2), constrained_layout=True)
    for profile in PROFILE_ORDER:
        subset = [row for row in rows if row["network_profile"] == profile]
        survivors = [row for row in subset if not row["zero_delivery"]]
        zero = [row for row in subset if row["zero_delivery"]]
        axes[0].scatter(
            [row["complete_reassembly_per_sent"] for row in survivors],
            [row["conditional_uplink_reassembly_ms_median"] for row in survivors],
            s=26, alpha=0.75, label=f"{PROFILE_LABELS[profile]} (delivering)",
        )
        if zero:
            axes[0].scatter(
                [row["complete_reassembly_per_sent"] for row in zero],
                [row["conditional_uplink_reassembly_ms_median"] for row in zero],
                s=60, marker="x", color="black",
                label="Zero delivery" if profile == PROFILE_ORDER[0] else None,
            )
        axes[1].scatter(
            [row["edge_admission_per_sent"] for row in subset],
            [row["conditional_uplink_reassembly_ms_median"] for row in subset],
            s=26, alpha=0.75, label=PROFILE_LABELS[profile],
        )
    axes[0].set_xlabel("Complete reassembly / sent", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Edge admission / sent", fontsize=11, fontweight="bold")
    for ax in axes:
        ax.set_ylabel(
            "Conditional uplink + reassembly latency (ms, median)",
            fontsize=11, fontweight="bold",
        )
        ax.grid(linestyle=":", alpha=0.35)
        ax.legend(fontsize=8, frameon=False)
        configure_axes(ax)
    figure.suptitle(
        "Conditional survivor latency is only interpretable beside its success "
        "probability\n(each point is one action/profile cell; crosses mark zero-delivery cells)",
        fontsize=12, fontweight="bold",
    )
    save(figure, output, "04_conditional_latency_with_delivery")


# ---------------------------------------------------------------------------
# 5/6. corrected E2E decomposition and before/after


def build_e2e_rows(
    direct: Sequence[Mapping[str, str]],
    baseline: Sequence[Mapping[str, str]],
    upstream: Sequence[Mapping[str, str]],
    live: Mapping[str, Any],
) -> list[dict[str, Any]]:
    upstream_by_profile = {row["network_profile"]: row for row in upstream}
    direct_by_profile: dict[str, list[Mapping[str, str]]] = {}
    baseline_by_profile: dict[str, list[Mapping[str, str]]] = {}
    for row in direct:
        direct_by_profile.setdefault(row["network_profile"], []).append(row)
    for row in baseline:
        baseline_by_profile.setdefault(row["network_profile"], []).append(row)

    ack_medians = [
        (cell.get("ack_observation_delay_ms") or {}).get("median")
        for cell in live["cells"]
        if not cell.get("error")
    ]
    ack_medians = [value for value in ack_medians if value is not None]
    ack_median = statistics.median(ack_medians) if ack_medians else None

    rows: list[dict[str, Any]] = []
    for profile in PROFILE_ORDER:
        up = upstream_by_profile[profile]
        direct_install = [
            f(cell.get("stage_direct_map_publish_install_ms_median", ""))
            for cell in direct_by_profile[profile]
        ]
        direct_install = [value for value in direct_install if value is not None]
        baseline_install = float(up["result_return_map_install_ms_action_balanced_median_ms"])
        direct_capture_install = [
            f(cell.get("stage_capture_to_install_ms_median", ""))
            for cell in direct_by_profile[profile]
        ]
        direct_capture_install = [v for v in direct_capture_install if v is not None]
        baseline_capture_install = [
            f(cell.get("stage_capture_to_install_ms_median", ""))
            for cell in baseline_by_profile[profile]
        ]
        baseline_capture_install = [v for v in baseline_capture_install if v is not None]
        rows.append(
            {
                "network_profile": profile,
                "network_profile_label": PROFILE_LABELS[profile],
                "sensor_pre_action_ms": float(
                    up["sensor_pre_action_ms_action_balanced_median_ms"]
                ),
                "ue_action_preparation_ms": float(
                    up["ue_action_path_ms_action_balanced_median_ms"]
                ),
                "feature_uplink_and_edge_reassembly_ms": float(
                    up["feature_transfer_reassembly_ms_action_balanced_median_ms"]
                ),
                "edge_queue_ms": float(up["edge_queue_ms_action_balanced_median_ms"]),
                "edge_processing_ms": float(
                    up["edge_compute_ms_action_balanced_median_ms"]
                )
                + float(up["edge_publication_ms_action_balanced_median_ms"]),
                "old_result_return_and_map_install_ms": baseline_install,
                "direct_map_publication_and_install_ms": (
                    statistics.median(direct_install) if direct_install else None
                ),
                "install_path_saving_ms": (
                    baseline_install - statistics.median(direct_install)
                    if direct_install
                    else None
                ),
                "map_feedback_arrival_ms_live_median": ack_median,
                "old_capture_to_install_ms_cell_median": (
                    statistics.median(baseline_capture_install)
                    if baseline_capture_install
                    else None
                ),
                "direct_capture_to_install_ms_cell_median": (
                    statistics.median(direct_capture_install)
                    if direct_capture_install
                    else None
                ),
            }
        )
    return rows


def plot_e2e(output: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    labels = [row["network_profile_label"] for row in rows]
    components = (
        ("Sensor / pre-action", "sensor_pre_action_ms", PALETTE["sensor"]),
        ("UE action preparation", "ue_action_preparation_ms", PALETTE["ue"]),
        ("Feature uplink + edge reassembly", "feature_uplink_and_edge_reassembly_ms", PALETTE["uplink"]),
        ("Edge queue", "edge_queue_ms", PALETTE["queue"]),
        ("Edge processing", "edge_processing_ms", PALETTE["edge"]),
        ("Direct edge-map publication + install", "direct_map_publication_and_install_ms", PALETTE["install"]),
    )
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    figure, ax = plt.subplots(figsize=(12.0, 6.8), constrained_layout=True)
    for label, key, color in components:
        values = np.array([float(row[key] or 0.0) for row in rows])
        ax.bar(x, values, bottom=bottom, label=label, color=color, width=0.62)
        for index, (base, value) in enumerate(zip(bottom, values)):
            if value >= 12:
                ax.text(
                    index, base + value / 2, f"{value:.1f}",
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    color="white" if color in {PALETTE["sensor"], PALETTE["queue"], PALETTE["edge"]} else "black",
                )
        bottom += values
    feedback = np.array(
        [float(row["map_feedback_arrival_ms_live_median"] or 0.0) for row in rows]
    )
    ax.bar(
        x, feedback, bottom=bottom + 14.0, width=0.62, color=PALETTE["feedback"],
        alpha=0.85, hatch="//",
        label="Map-feedback arrival at UE (separate; NOT part of map AoI)",
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Latency (ms)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Network profile", fontsize=12, fontweight="bold")
    ax.set_title(
        "Corrected end-to-end decomposition, direct edge-to-map (counterfactual)\n"
        "physical map freshness ends at map install; the hatched bar is a "
        "detached controller-observation delay",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9, ncols=2, frameon=False, loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    configure_axes(ax)
    save(figure, output, "05_corrected_e2e_decomposition")


def plot_before_after(output: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    labels = [row["network_profile_label"] for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(14.5, 6.0), constrained_layout=True)

    ax = axes[0]
    old = [float(row["old_result_return_and_map_install_ms"]) for row in rows]
    new = [float(row["direct_map_publication_and_install_ms"] or 0.0) for row in rows]
    ax.bar(x - width / 2, old, width=width, color=PALETTE["old"],
           label="Old: edge -> UE -> map")
    ax.bar(x + width / 2, new, width=width, color=PALETTE["new"],
           label="Corrected: direct edge -> map")
    for index, (left, right) in enumerate(zip(old, new)):
        ax.text(index - width / 2, left, f"{left:.1f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
        ax.text(index + width / 2, right, f"{right:.1f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    ax.set_ylabel("Post-edge installation path (ms)", fontsize=11, fontweight="bold")
    ax.set_title("Installation path", fontsize=12, fontweight="bold")

    ax = axes[1]
    old_total = [float(row["old_capture_to_install_ms_cell_median"] or 0.0) for row in rows]
    new_total = [float(row["direct_capture_to_install_ms_cell_median"] or 0.0) for row in rows]
    ax.bar(x - width / 2, old_total, width=width, color=PALETTE["old"],
           label="Old: edge -> UE -> map")
    ax.bar(x + width / 2, new_total, width=width, color=PALETTE["new"],
           label="Corrected: direct edge -> map")
    for index, (left, right) in enumerate(zip(old_total, new_total)):
        ax.text(index - width / 2, left, f"{left:.0f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
        ax.text(index + width / 2, right, f"{right:.0f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    ax.set_ylabel("Capture-to-install (ms, cell median)", fontsize=11, fontweight="bold")
    ax.set_title("End-to-end capture to map install", fontsize=12, fontweight="bold")

    for ax in axes:
        ax.set_xticks(x, labels, rotation=12)
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        ax.legend(fontsize=9, frameon=False)
        configure_axes(ax)
    figure.suptitle(
        "Before / after: removing the edge -> UE -> map detour (counterfactual replay "
        "of the same immutable 288-cell transport)",
        fontsize=13, fontweight="bold",
    )
    save(figure, output, "06_before_after_install_path")


# ---------------------------------------------------------------------------


def presentation_tables(
    budget_rows: Sequence[Mapping[str, Any]],
    quality_rows: Sequence[Mapping[str, Any]],
    e2e_rows: Sequence[Mapping[str, Any]],
    document: Mapping[str, Any],
    live: Mapping[str, Any],
) -> str:
    pooling = document["pooling_decision"]
    lines = [
        "# Direct edge-to-map presentation tables",
        "",
        "Simulator rows are **counterfactual**: the immutable 288-cell transport,",
        "admission and edge service are replayed with only the post-edge",
        "installation path changed. Rows marked **live** are measurements from the",
        "short four-action validation run.",
        "",
        "## 1. Actions meeting each freshness budget (counterfactual)",
        "",
        "Primary rule: physical map age within budget for at least 50% of route time.",
        "",
        "| Budget | Profile | 25% route | 50% route (primary) | 75% route | Best action | Best fraction |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in budget_rows:
        lines.append(
            f"| {row['freshness_budget_ms']} ms | {row['network_profile_label']} | "
            f"{row['actions_meeting_budget_25pct_route']}/72 | "
            f"**{row['actions_meeting_budget_primary']}/72** | "
            f"{row['actions_meeting_budget_75pct_route']}/72 | "
            f"{row['best_action_id']} | {100 * row['best_fresh_map_fraction']:.1f}% |"
        )
    eligible = [row for row in quality_rows if int(row["eligible_any_budget"]) == 1]
    lines.extend(
        [
            "",
            f"## 2. Model quality for the {len(eligible)} budget-eligible actions",
            "",
            "Quality is action-intrinsic: it is a property of the frozen split profile",
            "and does not vary with the network profile or the installation path.",
            "",
            "| Action | Family | Veh P | Veh R | Veh F1 | Veh XY MAE | Veh IoU | "
            "Per P | Per R | Per F1 | Per XY MAE | Seg mIoU |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    def show(value: Any, digits: int = 3) -> str:
        return "n/a" if value is None else f"{float(value):.{digits}f}"

    for row in sorted(eligible, key=lambda item: -(item["val_vehicle_f1"] or 0.0))[:20]:
        lines.append(
            f"| {row['action_id']} | {row['family']} | "
            f"{show(row['val_vehicle_precision'])} | {show(row['val_vehicle_recall'])} | "
            f"{show(row['val_vehicle_f1'])} | {show(row['val_vehicle_xy_mae_m'])} | "
            f"{show(row['val_vehicle_iou'])} | "
            f"{show(row['val_canonical_person_precision'])} | "
            f"{show(row['val_canonical_person_recall'])} | "
            f"{show(row['val_canonical_person_f1'])} | "
            f"{show(row['val_canonical_person_xy_mae_m'])} | "
            f"{show(row['val_segmentation_miou'])} |"
        )
    lines.append("")
    lines.append(
        f"(Top 20 of {len(eligible)} shown; the complete table is in "
        "`eligible_action_model_performance.csv`.)"
    )
    lines.extend(
        [
            "",
            "## 5/6. Corrected decomposition and before/after",
            "",
            "| Profile | Sensor | UE prep | Uplink+reassembly | Edge queue | Edge proc | "
            "Old return+install | Direct publish+install | Saving |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in e2e_rows:
        lines.append(
            f"| {row['network_profile_label']} | {row['sensor_pre_action_ms']:.1f} | "
            f"{row['ue_action_preparation_ms']:.1f} | "
            f"{row['feature_uplink_and_edge_reassembly_ms']:.1f} | "
            f"{row['edge_queue_ms']:.1f} | {row['edge_processing_ms']:.1f} | "
            f"{row['old_result_return_and_map_install_ms']:.1f} | "
            f"{show(row['direct_map_publication_and_install_ms'], 1)} | "
            f"{show(row['install_path_saving_ms'], 1)} |"
        )
    ack = e2e_rows[0]["map_feedback_arrival_ms_live_median"]
    lines.extend(
        [
            "",
            f"Map-feedback arrival at the UE (**live**, median of the four actions): "
            f"{show(ack, 2)} ms. This is a controller-observation delay measured "
            "separately and is **not** part of physical map-installation AoI.",
            "",
            "## Direct-map delay provenance",
            "",
            f"- Pooling rule: `{pooling['rule']}` "
            f"(Kruskal-Wallis H={pooling['kruskal_h']:.3f}, p={pooling['kruskal_p']:.3g}, "
            f"pre-registered alpha={pooling['preregistered_alpha']}).",
            f"- Live family medians (ms): "
            + ", ".join(
                f"{key} {value:.3f}"
                for key, value in sorted(pooling["family_medians_ms"].items())
            ),
            "",
            "### Extrapolations",
            "",
        ]
    )
    for item in document["extrapolations"]:
        lines.append(f"- {item}")
    lines.extend(["", "### Limitations", ""])
    for item in document["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def run(output: Path, counterfactual_root: Path, validation_root: Path) -> dict[str, Any]:
    require(not output.exists(), f"create-only output exists: {output}")
    direct, baseline, document = load_counterfactual(counterfactual_root)
    live = load_live(validation_root)
    require(UPSTREAM_LATENCY_CSV.is_file(), f"upstream latency table absent: {UPSTREAM_LATENCY_CSV}")
    upstream = read_csv(UPSTREAM_LATENCY_CSV)

    budget_rows = build_budget_rows(direct)
    quality_rows = build_quality_rows(direct, budget_rows)
    matched_rows = build_matched_rows(direct)
    conditional_rows = build_conditional_rows(direct)
    e2e_rows = build_e2e_rows(direct, baseline, upstream, live)

    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "budget_action_counts.csv", budget_rows)
    write_csv(output / "eligible_action_model_performance.csv", quality_rows)
    write_csv(output / "matched_action_across_profiles.csv", matched_rows)
    write_csv(output / "conditional_latency_with_delivery.csv", conditional_rows)
    write_csv(output / "corrected_e2e_decomposition.csv", e2e_rows)
    write_csv(output / "before_after_install_path.csv", e2e_rows)

    plot_budget_counts(output, budget_rows)
    plot_budget_sensitivity(output, budget_rows)
    plot_quality(output, quality_rows)
    plot_matched_actions(output, matched_rows)
    plot_conditional_latency(output, conditional_rows)
    plot_e2e(output, e2e_rows)
    plot_before_after(output, e2e_rows)

    tables = presentation_tables(budget_rows, quality_rows, e2e_rows, document, live)
    (output / "PRESENTATION_TABLES.md").write_text(tables, encoding="utf-8")

    analysis = {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "scientific_status": "COUNTERFACTUAL_SIMULATOR_PLUS_LIVE_SHORT_RUN",
        "counterfactual_root": str(counterfactual_root.relative_to(ROOT)),
        "validation_root": str(validation_root.relative_to(ROOT)),
        "upstream_latency_csv": str(UPSTREAM_LATENCY_CSV.relative_to(ROOT)),
        "upstream_latency_sha256": source._sha256(UPSTREAM_LATENCY_CSV),
        "budgets_ms": list(BUDGETS_MS),
        "thresholds": list(THRESHOLDS),
        "primary_threshold": PRIMARY_THRESHOLD,
        "primary_budget_rule": {
            "clock": "PHYSICAL_CAPTURE_CLOCK",
            "meaning": "physical map age is within budget for at least half of route time",
            "not_a_claim": "not a per-frame, worst-case, or always-fresh guarantee",
        },
        "budget_rows": len(budget_rows),
        "quality_rows": len(quality_rows),
        "eligible_actions": sum(
            1 for row in quality_rows if int(row["eligible_any_budget"]) == 1
        ),
        "matched_rows": len(matched_rows),
        "conditional_rows": len(conditional_rows),
        "e2e_rows": e2e_rows,
        "pooling_decision": document["pooling_decision"],
        "extrapolations": document["extrapolations"],
        "limitations": document["limitations"]
        + [
            "figures mixing simulator and live values label each explicitly and "
            "never merge them into one number",
        ],
    }
    source._atomic_json(output / "analysis.json", analysis)
    names = sorted(
        path.name
        for path in output.iterdir()
        if path.suffix in {".csv", ".json", ".md", ".png", ".pdf"}
    )
    hashes = {name: source._sha256(output / name) for name in names}
    source._atomic_json(
        output / "artifact_manifest.json",
        {"schema": f"{SCHEMA}.artifacts", "status": "COMPLETE", "sha256": hashes},
    )
    source._atomic_text(
        output / TERMINAL,
        json.dumps(
            {"schema": f"{SCHEMA}.terminal", "status": "COMPLETE", "artifacts": len(hashes)},
            sort_keys=True,
        )
        + "\n",
    )
    return analysis


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counterfactual-root", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    analysis = run(
        args.output.resolve(),
        args.counterfactual_root.resolve(strict=True),
        args.validation_root.resolve(strict=True),
    )
    print(json.dumps({k: v for k, v in analysis.items() if k != "e2e_rows"}, indent=2, sort_keys=True))
    print(TERMINAL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
