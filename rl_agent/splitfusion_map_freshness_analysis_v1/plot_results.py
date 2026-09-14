#!/usr/bin/env python3
"""Create presentation figures and a policy-oriented findings report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/splitfusion-map-freshness-mpl")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rl_agent.splitfusion_map_freshness_analysis_v1.analyze_288 import (
    BUDGETS_MS,
    DEFAULT_OUTPUT,
    PROFILE_ORDER,
    QueuePolicy,
    atomic_json,
    atomic_text,
    sha256,
    verify_manifest,
)


PROFILE_LABELS = {
    "FAVORABLE_STABLE": "Favorable",
    "MID_VARIABLE": "Mid-variable",
    "FADE_RECOVERY": "Fade/recovery",
    "ADVERSE_STABLE": "Adverse",
}
POLICY_LABELS = {
    "FIFO_NO_DISCARD": "FIFO, no discard",
    "LATEST_ONLY_NO_EXPIRY": "Strict latest-only",
}
FAMILY_COLORS = {
    "noAE": "#4c78a8",
    "AE128": "#f58518",
    "AE64": "#54a24b",
    "AE32": "#e45756",
}
PDF_METADATA = {
    "Creator": "SplitFusion map freshness analysis",
    "Producer": "matplotlib",
    "CreationDate": None,
    "ModDate": None,
}


def style_axis(axis: Any) -> None:
    axis.grid(True, alpha=0.22, linewidth=0.8)
    axis.tick_params(axis="both", labelsize=9, width=1.1)
    for label in axis.get_xticklabels() + axis.get_yticklabels():
        label.set_fontweight("bold")
    axis.xaxis.label.set_fontweight("bold")
    axis.yaxis.label.set_fontweight("bold")
    axis.title.set_fontweight("bold")


def save_figure(figure: Any, root: Path, stem: str) -> list[Path]:
    paths = [root / f"{stem}.png", root / f"{stem}.pdf"]
    figure.savefig(paths[0], dpi=220, bbox_inches="tight")
    figure.savefig(paths[1], metadata=PDF_METADATA, bbox_inches="tight")
    plt.close(figure)
    return paths


def load(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    verify_manifest(root)
    cells = pd.read_csv(root / "action_network_policy_freshness.csv")
    pareto = pd.read_csv(root / "quality_freshness_pareto.csv")
    aligned = pd.read_csv(root / "aligned_localization_by_profile_aoi_band.csv")
    analysis = json.loads((root / "analysis.json").read_text(encoding="utf-8"))
    return cells, pareto, aligned, analysis


def queue_policy_figure(analysis: dict[str, Any], figures: Path) -> list[Path]:
    aggregate = analysis["aggregate"]
    profiles = list(PROFILE_ORDER)
    x = np.arange(len(profiles))
    width = 0.36
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.5))
    colors = ("#8f8f8f", "#2a6fbb")

    specs = (
        ("queue_wait_ms_p95_cell_median", "Queue wait p95 (ms)", False),
        ("queue_wait_ms_max", "Worst queue wait (ms, log scale)", True),
        ("map_aoi_ms_cell_median", "Time-weighted map AoI (ms)", False),
        ("rate_useful_install_per_sent", "Useful installations / sent", False),
    )
    for axis, (field, ylabel, log_scale) in zip(axes.flat, specs):
        for offset, policy in enumerate(("FIFO_NO_DISCARD", "LATEST_ONLY_NO_EXPIRY")):
            values = [
                aggregate[policy]["by_network_profile"][profile][field]
                for profile in profiles
            ]
            axis.bar(
                x + (offset - 0.5) * width,
                values,
                width,
                label=POLICY_LABELS[policy],
                color=colors[offset],
            )
        axis.set_xticks(x, [PROFILE_LABELS[p] for p in profiles], rotation=15)
        axis.set_ylabel(ylabel)
        if log_scale:
            axis.set_yscale("log")
        style_axis(axis)
    axes[0, 0].legend(frameon=False, fontsize=9)
    figure.suptitle("Final edge service: FIFO backlog versus strict latest-only", fontweight="bold")
    figure.tight_layout()
    return save_figure(figure, figures, "01_fifo_vs_latest_queue_and_freshness")


def heatmap_figures(cells: pd.DataFrame, figures: Path) -> list[Path]:
    latest = cells[cells.queue_policy == "LATEST_ONLY_NO_EXPIRY"]
    paths: list[Path] = []
    for budget in BUDGETS_MS:
        key = f"fresh_map_time_ms_le_{budget}_fraction"
        pivot = (
            latest.pivot(index="action_id", columns="network_profile", values=key)
            .reindex(index=range(72), columns=PROFILE_ORDER)
        )
        figure, axis = plt.subplots(figsize=(7.4, 15.5))
        image = axis.imshow(
            100 * pivot.to_numpy(),
            aspect="auto",
            cmap="viridis",
            vmin=0,
            vmax=max(1.0, float(100 * pivot.max().max())),
        )
        axis.set_xticks(
            range(4), [PROFILE_LABELS[p] for p in PROFILE_ORDER], rotation=18
        )
        axis.set_yticks(range(0, 72, 3), range(0, 72, 3))
        axis.set_xlabel("Network profile")
        axis.set_ylabel("Action ID")
        axis.set_title(f"Strict latest-only: map fresh within {budget} ms")
        colorbar = figure.colorbar(image, ax=axis, pad=0.02)
        colorbar.set_label("Route time with fresh map (%)", fontweight="bold")
        colorbar.ax.tick_params(labelsize=9, width=1.1)
        for label in colorbar.ax.get_yticklabels():
            label.set_fontweight("bold")
        style_axis(axis)
        paths += save_figure(
            figure, figures, f"02_fresh_map_action_network_heatmap_{budget}ms"
        )
    return paths


def frontier_figures(pareto: pd.DataFrame, figures: Path, cls: str) -> list[Path]:
    paths: list[Path] = []
    for budget in BUDGETS_MS:
        selected = pareto[
            (pareto.freshness_budget_ms == budget) & (pareto.class_name == cls)
        ]
        figure, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), sharex=True)
        for axis, profile in zip(axes.flat, PROFILE_ORDER):
            block = selected[selected.network_profile == profile]
            for family, family_block in block.groupby("family"):
                axis.scatter(
                    family_block.median_feature_bytes / 1024.0,
                    family_block.quality_weighted_fresh_map_score,
                    s=24,
                    alpha=0.68,
                    color=FAMILY_COLORS[family],
                    label=family,
                )
            for row in block[block.pareto == 1].itertuples():
                axis.annotate(
                    str(int(row.action_id)),
                    (row.median_feature_bytes / 1024.0, row.quality_weighted_fresh_map_score),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=7,
                    fontweight="bold",
                )
            axis.set_xscale("log")
            axis.set_title(PROFILE_LABELS[profile])
            axis.set_xlabel("Median feature payload (KiB, log scale)")
            axis.set_ylabel(f"{cls.title()} F1 × fresh-map fraction")
            style_axis(axis)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=4,
            frameon=False,
        )
        figure.suptitle(
            f"{cls.title()} quality–freshness frontier at {budget} ms",
            fontweight="bold",
            y=0.998,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.895))
        paths += save_figure(
            figure, figures, f"03_{cls}_quality_freshness_frontier_{budget}ms"
        )
    return paths


def fixed_action_figure(cells: pd.DataFrame, figures: Path) -> list[Path]:
    latest = cells[cells.queue_policy == "LATEST_ONLY_NO_EXPIRY"]
    actions = (20, 46, 50, 58, 70, 71)
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 8.7), sharey=True)
    x = np.arange(4)
    width = 0.24
    colors = ("#4c78a8", "#f58518", "#54a24b")
    for axis, action in zip(axes.flat, actions):
        block = latest[latest.action_id == action].set_index("network_profile")
        for index, budget in enumerate(BUDGETS_MS):
            values = [
                100
                * block.loc[profile, f"fresh_map_time_ms_le_{budget}_fraction"]
                for profile in PROFILE_ORDER
            ]
            axis.bar(
                x + (index - 1) * width,
                values,
                width,
                color=colors[index],
                label=f"≤{budget} ms",
            )
        first = block.iloc[0]
        axis.set_title(
            f"Action {action}: {first.family}/{first.quantizer}, q={first.q:g}"
        )
        axis.set_xticks(
            x, [PROFILE_LABELS[p] for p in PROFILE_ORDER], rotation=20
        )
        axis.set_ylabel("Route time with fresh map (%)")
        style_axis(axis)
    axes[0, 0].legend(frameon=False, fontsize=9)
    figure.suptitle(
        "Hold the action fixed: network-conditioned map freshness",
        fontweight="bold",
    )
    figure.tight_layout()
    return save_figure(figure, figures, "04_fixed_actions_across_network_profiles")


def aligned_localization_figure(aligned: pd.DataFrame, figures: Path) -> list[Path]:
    vehicle = aligned[
        (aligned.class_name == "vehicle")
        & (aligned.rows_with_both_localization_errors > 0)
    ].copy()
    order = (
        "LE_150",
        "GT_150_LE_200",
        "GT_200_LE_250",
        "GT_250_LE_500",
        "GT_500",
    )
    labels = ("≤150", "150–200", "200–250", "250–500", ">500")
    minimum_rows = 100
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharex=True)
    colors = ("#4c78a8", "#f58518", "#54a24b", "#e45756")
    for profile, color in zip(PROFILE_ORDER, colors):
        block = vehicle[vehicle.network_profile == profile].set_index(
            "install_aoi_band_ms"
        )
        x = []
        delta_values = []
        counts = []
        for index, band in enumerate(order):
            if band not in block.index:
                continue
            row = block.loc[band]
            x.append(index)
            count = int(row.rows_with_both_localization_errors)
            counts.append(count)
            delta_values.append(
                row.aligned_minus_source_xy_error_m_median
                if count >= minimum_rows
                else np.nan
            )
        axes[0].plot(
            x,
            delta_values,
            marker="o",
            color=color,
            label=PROFILE_LABELS[profile],
        )
        axes[1].plot(x, counts, marker="o", color=color)
    axes[0].axhline(0.0, color="#333333", linewidth=1.0, linestyle="--")
    axes[0].set_ylabel("Median aligned-minus-source XY error (m)")
    axes[0].set_title(f"Vehicle error change (shown only when n ≥ {minimum_rows})")
    axes[1].set_ylabel("Matched vehicle rows (log scale)")
    axes[1].set_yscale("log")
    axes[1].set_title("Evidence support by age band")
    for axis in axes:
        axis.set_xticks(range(len(order)), labels, rotation=15)
        axis.set_xlabel("Capture-to-install AoI band (ms)")
        style_axis(axis)
    axes[0].legend(frameon=False, fontsize=9)
    figure.suptitle(
        "Secondary live check: vehicle localization versus installed-frame age",
        fontweight="bold",
    )
    figure.tight_layout()
    return save_figure(figure, figures, "05_aligned_vehicle_localization_by_aoi")


def best_actions(
    cells: pd.DataFrame, *, budget: int, cls: str, count: int = 5
) -> list[dict[str, Any]]:
    latest = cells[cells.queue_policy == "LATEST_ONLY_NO_EXPIRY"]
    key = f"{cls}_f1_x_fresh_map_fraction_ms_le_{budget}"
    result: list[dict[str, Any]] = []
    for profile in PROFILE_ORDER:
        block = latest[latest.network_profile == profile].sort_values(
            key, ascending=False
        )
        for rank, row in enumerate(block.head(count).itertuples(), start=1):
            result.append(
                {
                    "network_profile": profile,
                    "budget_ms": budget,
                    "class_name": cls,
                    "rank": rank,
                    "action_id": int(row.action_id),
                    "profile_id": row.profile_id,
                    "payload_kib": row.median_feature_bytes / 1024.0,
                    "validation_f1": getattr(
                        row,
                        "val_vehicle_f1"
                        if cls == "vehicle"
                        else "val_canonical_person_f1",
                    ),
                    "fresh_map_fraction": getattr(
                        row, f"fresh_map_time_ms_le_{budget}_fraction"
                    ),
                    "quality_weighted_fresh_map_score": getattr(row, key),
                }
            )
    return result


def findings_text(
    cells: pd.DataFrame,
    aligned: pd.DataFrame,
    analysis: dict[str, Any],
) -> str:
    aggregate = analysis["aggregate"]
    fifo = aggregate["FIFO_NO_DISCARD"]["overall"]
    latest = aggregate["LATEST_ONLY_NO_EXPIRY"]["overall"]
    latest_cells = cells[cells.queue_policy == "LATEST_ONLY_NO_EXPIRY"]
    lines = [
        "# Scientific findings for policy design",
        "",
        "## Decisive queue result",
        "",
        f"With final edge service, FIFO has a cell-median p95 queue wait of "
        f"{fifo['queue_wait_ms_p95_cell_median']:.1f} ms, a worst wait of "
        f"{fifo['queue_wait_ms_max'] / 1000:.2f} s and a maximum compute backlog "
        f"of {fifo['compute_queue_high_water_max']} frames. Its median-of-cell "
        "queue medians is 0 ms because many heavy-payload cells admit frames",
        "too sparsely to queue; that zero must not be presented as absence of",
        "FIFO congestion.",
        "",
        f"Strict latest-only caps pending compute at one frame and lowers the "
        f"median cell map AoI from {fifo['map_aoi_ms_cell_median']:.1f} to "
        f"{latest['map_aoi_ms_cell_median']:.1f} ms. It processes fewer eventual "
        "updates, but improves timely useful-update yield and fresh-map time at",
        "all three budgets. This is exactly the intended trade: discard obsolete",
        "pending work, not the latest available pending frame.",
        "",
        "## Freshness budgets",
        "",
        "| Budget | Best favorable fresh-map time | Best adverse fresh-map time | Interpretation |",
        "|---:|---:|---:|---|",
    ]
    for budget in BUDGETS_MS:
        key = f"fresh_map_time_ms_le_{budget}_fraction"
        favorable = latest_cells[
            latest_cells.network_profile == "FAVORABLE_STABLE"
        ][key].max()
        adverse = latest_cells[
            latest_cells.network_profile == "ADVERSE_STABLE"
        ][key].max()
        interpretation = (
            "No split action is a credible hard-budget fallback"
            if budget <= 200
            else "Useful split region appears, but still not continuously reliable"
        )
        lines.append(
            f"| {budget} ms | {100 * favorable:.1f}% | {100 * adverse:.1f}% | {interpretation} |"
        )
    lines.extend(
        [
            "",
            "The 150 ms budget is an aggressive diagnostic. The best split action",
            "keeps the map inside it for less than one tenth of favorable route",
            "time and about one percent of adverse route time. At 200 ms the best",
            "actions remain below 40% favorable and near 10% adverse. At 250 ms",
            "the best favorable action reaches roughly two thirds, but the best",
            "adverse action remains near one third. A local-inference action is",
            "therefore empirically motivated if any of these is a hard safety",
            "budget; split inference remains useful for softer cooperative-map",
            "objectives.",
            "",
            "## Action-conditioned network behavior",
            "",
            "Intrinsic validation quality is held fixed for an action. The network",
            "changes whether and when that quality becomes available. Hence the",
            "policy model must use P(outcome | action, network state), not a network",
            "average over all actions. The heatmaps and fixed-action figure retain",
            "all 72 action × four profile cells.",
            "",
            "The quality-weighted scores are decision surrogates:",
            "",
            "```math",
            "M^{(c)}_{a,n}(B)=Q^{(c)}_a\,C_{a,n}(B)",
            "```",
            "",
            "They are not claims of newly measured network-specific model accuracy.",
            "Person and vehicle scores remain separate; no unsupported weighting",
            "between them is introduced.",
            "",
            "## Aligned localization limitation",
            "",
        ]
    )
    person_supported = int(
        aligned.loc[aligned.class_name == "person", "rows_with_both_localization_errors"].sum()
    )
    vehicle_supported = int(
        aligned.loc[aligned.class_name == "vehicle", "rows_with_both_localization_errors"].sum()
    )
    lines.extend(
        [
            f"The retained live rows provide {vehicle_supported:,} vehicle rows with",
            "both source-time and aligned-retrieval localization error, but",
            f"{person_supported:,} corresponding person rows. Person localization",
            "versus staleness is therefore unsupported by this campaign and is not",
            "silently inferred. Vehicle aligned error generally increases in the",
            "older AoI bands, but the result is secondary because source/aligned",
            "matching support can differ and aligned truth is sampled during exact",
            "record retrieval after ACK rather than at an exact install interrupt.",
            "",
            "## Consequences for the controller",
            "",
            "The causal state should include lagged network state, current map AoI,",
            "last useful installation time, edge busy/pending state, recent service",
            "EWMA, previous action and terminal outcome. `SUPERSEDED_PENDING` is not",
            "radio loss: it earns no installation utility, while bytes and compute",
            "already spent remain charged. This gives PPO the correct delayed credit",
            "signal without inventing a discard penalty.",
            "",
            "Before policy training, a LOCAL action needs a matched measurement of",
            "full local compute, quality, compact object-result bytes and map-install",
            "latency. Local perception and local-to-cooperative-map publication must",
            "remain distinct boundaries.",
            "",
        ]
    )
    return "\n".join(lines)


def policy_guidance_text(
    cells: pd.DataFrame,
    analysis: dict[str, Any],
) -> str:
    latest = cells[cells.queue_policy == "LATEST_ONLY_NO_EXPIRY"]
    fifo = analysis["aggregate"]["FIFO_NO_DISCARD"]["overall"]
    strict = analysis["aggregate"]["LATEST_ONLY_NO_EXPIRY"]["overall"]
    lines = [
        "# Policy-design guidance from the 288-cell freshness analysis",
        "",
        "## What the controller is actually optimizing",
        "",
        "`installed / sent` is a throughput statistic. It does not say whether",
        "the installed information was still fresh. The controller-facing",
        "outcomes in this analysis are:",
        "",
        "- **fresh-map fraction:** fraction of route time for which the newest",
        "  installed contribution has capture-to-current-time AoI within the",
        "  selected 150, 200 or 250 ms budget;",
        "- **timely useful-update yield:** sent frames that install within the",
        "  budget and advance the map to a newer capture;",
        "- **quality-weighted freshness:** frozen validation F1 for the selected",
        "  action multiplied by its network-conditioned fresh-map fraction.",
        "",
        "The first metric is the primary map-service outcome. The second",
        "explains how efficiently an action creates those fresh intervals. The",
        "third exposes the quality/freshness trade rather than rewarding a tiny",
        "but inaccurate payload solely for arriving quickly.",
        "",
        "## Queue policy decision",
        "",
        f"FIFO reaches a worst observed counterfactual wait of "
        f"{fifo['queue_wait_ms_max'] / 1000:.2f} s and a backlog of "
        f"{fifo['compute_queue_high_water_max']} frames. Strict latest-only caps",
        "the pending compute slot at one frame. When active work finishes, the",
        "newest pending frame runs and all older pending frames receive an",
        "explicit `SUPERSEDED_PENDING_COMPUTE` terminal.",
        "",
        f"This lowers median cell map AoI from {fifo['map_aoi_ms_cell_median']:.1f}",
        f"to {strict['map_aoi_ms_cell_median']:.1f} ms while intentionally reducing",
        "the number of eventual installations. That reduction is not failure:",
        "old work is being exchanged for newer map state. There is no 25 ms",
        "expiry and no reason to discard the only pending frame merely because",
        "it waited.",
        "",
        "## Best observed split regions",
        "",
        "The table reports the action with the largest *raw* fresh-map fraction",
        "for each network and budget. It must not be read as the best final",
        "policy action because it does not yet include person/vehicle quality.",
        "",
        "| Budget | Network | Action | Payload | Fresh-map time |",
        "|---:|---|---:|---:|---:|",
    ]
    for budget in BUDGETS_MS:
        key = f"fresh_map_time_ms_le_{budget}_fraction"
        for profile in PROFILE_ORDER:
            row = latest[latest.network_profile == profile].sort_values(
                key, ascending=False
            ).iloc[0]
            lines.append(
                f"| {budget} ms | {PROFILE_LABELS[profile]} | "
                f"{int(row.action_id)} | {row.median_feature_bytes / 1024:.1f} KiB | "
                f"{100 * row[key]:.1f}% |"
            )
    lines.extend(
        [
            "",
            "No split action keeps the map continuously inside any tested",
            "budget. The 150 ms surface is an aggressive diagnostic; even the",
            "best favorable result is below 10%. At 200 ms, the best favorable",
            "result remains below 40%. At 250 ms, small-payload actions become",
            "useful in favorable/mid/fade conditions, but the best adverse result",
            "is only about 30%. This creates a real decision problem rather than",
            "one globally dominant split action.",
            "",
            "## Causal controller inputs",
            "",
            "A recurrent policy should receive only values available before the",
            "next choice: lagged SNR/MCS or capacity estimate, current installed",
            "map AoI, time since last useful install, edge busy/pending flags,",
            "recent service-time EWMA, previous action, and its terminal outcome.",
            "The LSTM hidden state may learn channel trend and predict the next",
            "condition implicitly; future profile labels and future ACKs remain",
            "forbidden.",
            "",
            "A compact starting reward is:",
            "",
            "```math",
            "r_t = w_p Q_p(a_t)F_B(\\mathrm{AoI}_{t+1})",
            "    + w_v Q_v(a_t)F_B(\\mathrm{AoI}_{t+1})",
            "    - \\lambda_b \\frac{b(a_t)}{B_{\\max}}",
            "    - \\lambda_c C(a_t) - \\lambda_s I[a_t \\ne a_{t-1}]",
            "```",
            "",
            "where `Q_p` and `Q_v` are frozen person/vehicle quality, `F_B` is a",
            "hard or soft freshness utility at budget `B`, `b(a)` is transmitted",
            "feature bytes, and `C(a)` is measured compute cost. A superseded",
            "frame earns no installation utility, but its already-spent bytes and",
            "compute remain charged. It should not receive an extra arbitrary",
            "discard penalty, because replacement is the scheduler's correct",
            "map-first behavior.",
            "",
            "## LOCAL decision boundary",
            "",
            "The split results trigger—not satisfy—the LOCAL baseline step. The",
            "older LR-ASPP local measurements are not interchangeable with the",
            "current frozen FCOS service. Before `LOCAL_INFER` enters the action",
            "set, measure the current FCOS full-local compute distribution and",
            "sustainable rate, compact object-result bytes versus object count,",
            "identical-input quality, and compact-result delivery/map-install",
            "latency over all four OAI profiles. Record local completion, edge",
            "installation and ACK as distinct times. Until then, train and report",
            "the present surface as **split-only** and do not manufacture a local",
            "transition from old measurements.",
            "",
            "## What not to conclude",
            "",
            "- Network profiles do not change intrinsic model quality; they change",
            "  whether and when that quality reaches the map.",
            "- A high install ratio does not establish freshness.",
            "- Person localization degradation with age is not identified here;",
            "  the retained campaign has no matched person rows for that check.",
            "- The counterfactual predicts the qualified final implementation; it",
            "  is not a second 288-cell live campaign.",
            "",
        ]
    )
    return "\n".join(lines)


def run(root: Path) -> dict[str, Any]:
    cells, pareto, aligned, analysis = load(root)
    figures = root / "figures"
    # This presentation-only builder may replace its own derived figures. It
    # never overwrites source measurements or the primary analysis artifacts.
    figures.mkdir(exist_ok=True)
    paths: list[Path] = []
    paths += queue_policy_figure(analysis, figures)
    paths += heatmap_figures(cells, figures)
    paths += frontier_figures(pareto, figures, "person")
    paths += frontier_figures(pareto, figures, "vehicle")
    paths += fixed_action_figure(cells, figures)
    paths += aligned_localization_figure(aligned, figures)

    rankings: list[dict[str, Any]] = []
    for budget in BUDGETS_MS:
        rankings += best_actions(cells, budget=budget, cls="person")
        rankings += best_actions(cells, budget=budget, cls="vehicle")
    rankings_path = root / "top_actions_by_profile_budget.csv"
    pd.DataFrame(rankings).to_csv(rankings_path, index=False, lineterminator="\n")
    findings_path = root / "SCIENTIFIC_FINDINGS.md"
    atomic_text(findings_path, findings_text(cells, aligned, analysis))
    guidance_path = root / "POLICY_DESIGN_GUIDANCE.md"
    atomic_text(guidance_path, policy_guidance_text(cells, analysis))

    all_paths = paths + [rankings_path, findings_path, guidance_path]
    manifest = {
        "schema": "scenesense.splitfusion.map_freshness_policy_analysis.plots.v2",
        "status": "COMPLETE",
        "source_analysis_sha256": sha256(root / "analysis.json"),
        "sha256": {
            str(path.relative_to(root)): sha256(path) for path in sorted(all_paths)
        },
    }
    atomic_json(root / "plot_manifest.json", manifest)
    atomic_text(
        root / "SPLITFUSION_MAP_FRESHNESS_POLICY_PLOTS_COMPLETE",
        json.dumps(
            {
                "status": "COMPLETE",
                "plot_manifest_sha256": sha256(root / "plot_manifest.json"),
            },
            sort_keys=True,
        )
        + "\n",
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    root = parse_args(argv).analysis_root.resolve()
    manifest = run(root)
    print(json.dumps({"artifacts": len(manifest["sha256"])}, indent=2))
    print("SPLITFUSION_MAP_FRESHNESS_POLICY_PLOTS_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
