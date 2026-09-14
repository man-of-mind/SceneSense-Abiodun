#!/usr/bin/env python3
"""Plot the 288-cell capture-clock versus action-clock analysis."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/splitfusion-dual-clock-mpl")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rl_agent.splitfusion_map_freshness_analysis_v1.analyze_timing_boundaries import (
    BUDGETS_MS,
    DEFAULT_OUTPUT,
    PROFILE_ORDER,
    QueuePolicy,
    atomic_json,
    atomic_text,
    prior,
    require,
)


PROFILE_LABELS = {
    "FAVORABLE_STABLE": "Favorable",
    "MID_VARIABLE": "Mid-variable",
    "FADE_RECOVERY": "Fade/recovery",
    "ADVERSE_STABLE": "Adverse",
}
COLORS = {
    "prior": "#9a9a9a",
    "physical": "#2a6fbb",
    "action": "#e68613",
}
PDF_METADATA = {
    "Creator": "SplitFusion dual-clock freshness analysis",
    "Producer": "matplotlib",
    "CreationDate": None,
    "ModDate": None,
}
TERMINAL = "SPLITFUSION_MAP_FRESHNESS_DUAL_CLOCK_PLOTS_COMPLETE"


def style(axis: Any) -> None:
    axis.grid(True, axis="y", alpha=0.22, linewidth=0.8)
    axis.tick_params(axis="both", labelsize=9, width=1.1)
    for label in axis.get_xticklabels() + axis.get_yticklabels():
        label.set_fontweight("bold")
    axis.xaxis.label.set_fontweight("bold")
    axis.yaxis.label.set_fontweight("bold")
    axis.title.set_fontweight("bold")


def save(figure: Any, root: Path, stem: str) -> list[Path]:
    paths = [root / f"{stem}.png", root / f"{stem}.pdf"]
    figure.savefig(paths[0], dpi=220, bbox_inches="tight")
    figure.savefig(paths[1], metadata=PDF_METADATA, bbox_inches="tight")
    plt.close(figure)
    return paths


def timing_figure(analysis: dict[str, Any], root: Path) -> list[Path]:
    latest = analysis["aggregate"][QueuePolicy.LATEST_ONLY_NO_EXPIRY.value]
    x = np.arange(len(PROFILE_ORDER))
    width = 0.25
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))
    specs = (
        (
            "Install latency",
            (
                ("prior_physical_install_aoi_ms_cell_median", "Prior capture clock", "prior"),
                ("physical_install_aoi_ms_cell_median", "Corrected capture clock", "physical"),
                ("action_service_install_latency_ms_cell_median", "Action-start clock", "action"),
            ),
        ),
        (
            "Time-weighted map age",
            (
                ("prior_physical_map_aoi_ms_cell_median", "Prior capture clock", "prior"),
                ("physical_map_aoi_ms_cell_median", "Corrected capture clock", "physical"),
                ("action_clock_map_age_ms_cell_median", "Action-start clock", "action"),
            ),
        ),
    )
    for axis, (title, fields) in zip(axes, specs):
        for index, (field, label, color) in enumerate(fields):
            values = [
                latest["by_network_profile"][profile][field]
                for profile in PROFILE_ORDER
            ]
            axis.bar(
                x + (index - 1) * width,
                values,
                width,
                label=label,
                color=COLORS[color],
            )
        axis.set_xticks(x, [PROFILE_LABELS[p] for p in PROFILE_ORDER], rotation=15)
        axis.set_ylabel("Milliseconds")
        axis.set_title(title)
        style(axis)
    axes[0].legend(frameon=False, fontsize=9)
    figure.suptitle(
        "288 cells: original analysis, causal correction, and action boundary",
        fontweight="bold",
    )
    figure.tight_layout()
    return save(figure, root, "01_old_vs_new_timing_boundaries")


def aggregate_freshness_figure(analysis: dict[str, Any], root: Path) -> list[Path]:
    latest = analysis["aggregate"][QueuePolicy.LATEST_ONLY_NO_EXPIRY.value]
    x = np.arange(len(PROFILE_ORDER))
    width = 0.25
    figure, axes = plt.subplots(1, 3, figsize=(17.2, 5.2), sharey=True)
    for axis, budget in zip(axes, BUDGETS_MS):
        fields = (
            (f"prior_physical_fresh_map_time_ms_le_{budget}_fraction", "Prior capture clock", "prior"),
            (f"physical_fresh_map_time_ms_le_{budget}_fraction", "Corrected capture clock", "physical"),
            (f"action_clock_fresh_map_time_ms_le_{budget}_fraction", "Action-start clock", "action"),
        )
        for index, (field, label, color) in enumerate(fields):
            values = [
                100 * latest["by_network_profile"][profile][field]
                for profile in PROFILE_ORDER
            ]
            axis.bar(
                x + (index - 1) * width,
                values,
                width,
                label=label,
                color=COLORS[color],
            )
        axis.set_xticks(x, [PROFILE_LABELS[p] for p in PROFILE_ORDER], rotation=18)
        axis.set_ylabel("Route time within budget (%)")
        axis.set_title(f"Map-age budget: {budget} ms")
        style(axis)
    axes[0].legend(frameon=False, fontsize=9)
    figure.suptitle(
        "Strict latest-only: physical freshness versus action-attributable age",
        fontweight="bold",
    )
    figure.tight_layout()
    return save(figure, root, "02_old_vs_new_fresh_map_time")


def winner_figure(winners: pd.DataFrame, root: Path) -> list[Path]:
    selected = winners[winners.objective == "RAW_FRESHNESS"]
    x = np.arange(len(PROFILE_ORDER))
    width = 0.36
    figure, axes = plt.subplots(1, 3, figsize=(16.7, 5.3), sharey=True)
    for axis, budget in zip(axes, BUDGETS_MS):
        block = selected[selected.freshness_budget_ms == budget]
        for index, (clock, label, color) in enumerate(
            (
                ("PHYSICAL_CAPTURE_CLOCK", "Physical capture clock", "physical"),
                ("ACTION_START_CLOCK", "Action-start clock", "action"),
            )
        ):
            rows = [
                block[
                    (block.network_profile == profile) & (block.clock == clock)
                ].iloc[0]
                for profile in PROFILE_ORDER
            ]
            values = [100 * float(row.fresh_map_fraction) for row in rows]
            bars = axis.bar(
                x + (index - 0.5) * width,
                values,
                width,
                label=label,
                color=COLORS[color],
            )
            for bar, row in zip(bars, rows):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.0,
                    f"a{int(row.action_id)}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )
        axis.set_xticks(x, [PROFILE_LABELS[p] for p in PROFILE_ORDER], rotation=18)
        axis.set_ylabel("Best route time within budget (%)")
        axis.set_title(f"Budget: {budget} ms")
        style(axis)
    axes[0].legend(frameon=False, fontsize=9)
    figure.suptitle(
        "Best fixed action under each timing interpretation",
        fontweight="bold",
    )
    figure.tight_layout()
    return save(figure, root, "03_best_action_by_clock_and_network")


def queue_policy_figure(analysis: dict[str, Any], root: Path) -> list[Path]:
    aggregate = analysis["aggregate"]
    x = np.arange(len(PROFILE_ORDER))
    width = 0.36
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.4))
    specifications = (
        ("physical_map_aoi_ms_cell_median", "Physical map age", "Milliseconds"),
        ("action_clock_map_age_ms_cell_median", "Action-clock map age", "Milliseconds"),
        ("physical_fresh_map_time_ms_le_200_fraction", "Physical map age ≤200 ms", "Route time (%)"),
        ("action_clock_fresh_map_time_ms_le_200_fraction", "Action-clock map age ≤200 ms", "Route time (%)"),
    )
    for axis, (field, title, ylabel) in zip(axes.flat, specifications):
        for index, (policy, label, color) in enumerate(
            (
                (QueuePolicy.FIFO_NO_DISCARD.value, "FIFO, no discard", "#8f8f8f"),
                (QueuePolicy.LATEST_ONLY_NO_EXPIRY.value, "Strict latest-only", "#2a6fbb"),
            )
        ):
            values = [
                aggregate[policy]["by_network_profile"][profile][field]
                for profile in PROFILE_ORDER
            ]
            if "fraction" in field:
                values = [100 * value for value in values]
            axis.bar(
                x + (index - 0.5) * width,
                values,
                width,
                label=label,
                color=color,
            )
        axis.set_xticks(x, [PROFILE_LABELS[p] for p in PROFILE_ORDER], rotation=18)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        style(axis)
    axes[0, 0].legend(frameon=False, fontsize=9)
    figure.suptitle(
        "Causality-corrected FIFO versus strict latest-only",
        fontweight="bold",
    )
    figure.tight_layout()
    return save(figure, root, "04_fifo_vs_latest_dual_clock")


def heatmap_figures(cells: pd.DataFrame, root: Path) -> list[Path]:
    latest = cells[cells.queue_policy == QueuePolicy.LATEST_ONLY_NO_EXPIRY.value]
    paths: list[Path] = []
    for budget in BUDGETS_MS:
        figure, axes = plt.subplots(1, 2, figsize=(12.0, 15.2), sharey=True)
        specifications = (
            (f"physical_fresh_map_time_ms_le_{budget}_fraction", "Physical capture clock"),
            (f"action_clock_fresh_map_time_ms_le_{budget}_fraction", "Action-start clock"),
        )
        maximum = 100 * max(float(latest[field].max()) for field, _ in specifications)
        for axis, (field, title) in zip(axes, specifications):
            pivot = (
                latest.pivot(index="action_id", columns="network_profile", values=field)
                .reindex(index=range(72), columns=PROFILE_ORDER)
            )
            image = axis.imshow(
                100 * pivot.to_numpy(),
                aspect="auto",
                cmap="viridis",
                vmin=0,
                vmax=max(1.0, maximum),
            )
            axis.set_xticks(
                range(4),
                [PROFILE_LABELS[profile] for profile in PROFILE_ORDER],
                rotation=18,
            )
            axis.set_yticks(range(0, 72, 3), range(0, 72, 3))
            axis.set_xlabel("Network profile")
            axis.set_ylabel("Action ID")
            axis.set_title(title)
            axis.grid(False)
            style(axis)
        colorbar = figure.colorbar(image, ax=axes, pad=0.02, shrink=0.82)
        colorbar.set_label("Route time within budget (%)", fontweight="bold")
        colorbar.ax.tick_params(labelsize=9, width=1.1)
        for label in colorbar.ax.get_yticklabels():
            label.set_fontweight("bold")
        figure.suptitle(
            f"Action × network behavior at {budget} ms",
            fontweight="bold",
        )
        paths += save(figure, root, f"05_action_network_heatmap_{budget}ms")
    return paths


def run(analysis_root: Path) -> list[Path]:
    prior.verify_manifest(analysis_root)
    figure_root = analysis_root / "figures"
    require(not figure_root.exists(), f"create-only figure directory exists: {figure_root}")
    figure_root.mkdir(parents=False, exist_ok=False)
    cells = pd.read_csv(analysis_root / "action_network_dual_clock_freshness.csv")
    winners = pd.read_csv(analysis_root / "top_actions_by_clock_profile_budget.csv")
    analysis = json.loads((analysis_root / "analysis.json").read_text(encoding="utf-8"))
    require(len(cells) == 576, "dual-clock table is not 576 rows")
    paths: list[Path] = []
    paths += timing_figure(analysis, figure_root)
    paths += aggregate_freshness_figure(analysis, figure_root)
    paths += winner_figure(winners, figure_root)
    paths += queue_policy_figure(analysis, figure_root)
    paths += heatmap_figures(cells, figure_root)
    hashes = {path.relative_to(analysis_root).as_posix(): prior.sha256(path) for path in paths}
    atomic_json(
        analysis_root / "plot_manifest.json",
        {
            "schema": "scenesense.splitfusion.map_freshness_dual_clock_plots.v1",
            "status": "COMPLETE",
            "sha256": hashes,
        },
    )
    atomic_text(analysis_root / TERMINAL, TERMINAL + "\n")
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = run(args.analysis_root.resolve())
    print(f"{TERMINAL}: {len(paths)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
