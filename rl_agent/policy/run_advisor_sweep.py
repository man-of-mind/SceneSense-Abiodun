"""Run the advisor-facing epsilon x preferred-core x range Track A sweep."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_matplotlib_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .channel import ChannelSurface
from .config import REPO_ROOT, load_config
from .replay import registry_frame
from .reporting import new_run_directory, summarize_frames, write_run_files
from .sweep_support import markdown_table, prepare_replays, run_cell, selected_manifest, source_hashes


Cell = tuple[float, int, float]


def _cell_id(epsilon_m: float, preferred_core_kib: int, range_m: float) -> str:
    return f"eps{epsilon_m:.1f}__core{preferred_core_kib}__range{range_m:.0f}"


def _grid_cells(config: dict) -> list[Cell]:
    section = config["advisor_sweep"]
    return [
        (float(epsilon), int(core), float(range_m))
        for epsilon in section["epsilon_m_values"]
        for core in section["preferred_core_kib_values"]
        for range_m in section["range_m_values"]
    ]


def _plot_frontier(summary: pd.DataFrame, path_prefix: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14.0, 5.2))
    colors = {25.0: "#4472C4", 40.0: "#ED7D31"}
    markers = {90: "o", 129: "s"}
    linestyles = {"shielded": "-", "clairvoyant": "--"}
    for (controller, core, range_m), group in summary.groupby(
        ["controller", "preferred_core_kib", "range_m"], sort=True
    ):
        group = group.sort_values("epsilon_m")
        label = f"{controller}, core={core}, range={range_m:g}m"
        axes[0].plot(
            group["epsilon_m"],
            group["over_budget_pct"],
            marker=markers[int(core)],
            linestyle=linestyles[str(controller)],
            color=colors[float(range_m)],
            label=label,
        )
        axes[1].plot(
            group["epsilon_m"],
            group["split_pct"],
            marker=markers[int(core)],
            linestyle=linestyles[str(controller)],
            color=colors[float(range_m)],
            label=label,
        )
    axes[0].set_ylabel("Over-budget frames (%)")
    axes[0].set_title("Achievability / feasibility frontier")
    axes[1].set_ylabel("SPLIT schedule (%)")
    axes[1].set_title("Mode response")
    for axis in axes:
        axis.set_xlabel("Localization target epsilon (m)")
        axis.set_xticks([1.5, 2.0, 2.5])
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    figure.suptitle("Track A advisor sweep (solid=shielded, dashed=clairvoyant)")
    figure.tight_layout()
    figure.savefig(path_prefix.with_suffix(".png"), dpi=300)
    figure.savefig(path_prefix.with_suffix(".pdf"))
    plt.close(figure)


def run(config_path: Path | None = None, smoke: bool = False) -> Path:
    config = load_config(config_path)
    section = config["advisor_sweep"]
    fixed_shield = section["fixed_shield"]
    config["safety"]["ucb_k"] = float(fixed_shield["ucb_k"])
    config["safety"]["c1_pessimism_factor"] = float(fixed_shield["c1_pessimism_factor"])
    cells = _grid_cells(config)
    if smoke:
        cells = [(1.5, 90, 25.0), (2.5, 129, 40.0)]

    ranges = sorted({range_m for _, _, range_m in cells})
    surface = ChannelSurface(config)
    registry, by_range = prepare_replays(config, ranges)
    run_dir = new_run_directory("advisor_sweep_smoke" if smoke else "advisor_sweep")
    registry_frame(registry).to_csv(run_dir / "replay_registry.csv", index=False)

    rows: list[dict] = []
    for index, (epsilon_m, preferred_core_kib, range_m) in enumerate(cells, start=1):
        cell_config = copy.deepcopy(config)
        cell_config["safety"]["epsilon_m"] = epsilon_m
        cell_config["safety"]["range_m"] = range_m
        cell_config["actions"]["preferred_core_kib"] = preferred_core_kib
        cell_name = _cell_id(epsilon_m, preferred_core_kib, range_m)
        print(f"[{index}/{len(cells)}] {cell_name}", flush=True)
        rows.extend(
            run_cell(
                cell_config,
                surface,
                by_range[range_m],
                ["shielded", "clairvoyant"],
                bool(section["common_random_latency_by_tick"]),
                {
                    "cell_id": cell_name,
                    "epsilon_m": epsilon_m,
                    "preferred_core_kib": preferred_core_kib,
                    "range_m": range_m,
                    "range_extrapolative": range_m > 25.0,
                },
            )
        )

    metrics = pd.DataFrame(rows)
    group_keys = [
        "cell_id",
        "epsilon_m",
        "preferred_core_kib",
        "range_m",
        "range_extrapolative",
        "controller",
    ]
    summary = summarize_frames(metrics, group_keys=group_keys).sort_values(
        ["epsilon_m", "preferred_core_kib", "range_m", "controller"]
    )
    summary["feasible_pct"] = 100.0 - summary["over_budget_pct"]
    per_epsilon = summarize_frames(metrics, group_keys=["epsilon_m", "controller"]).sort_values(
        ["epsilon_m", "controller"]
    )
    per_epsilon["feasible_pct"] = 100.0 - per_epsilon["over_budget_pct"]
    per_replay = summarize_frames(metrics, group_keys=[*group_keys[:-1], "scenario", "controller"])
    per_replay["feasible_pct"] = 100.0 - per_replay["over_budget_pct"]
    per_epsilon.to_csv(run_dir / "per_epsilon_summary.csv", index=False)
    per_replay.to_csv(run_dir / "per_replay_summary.csv", index=False)
    _plot_frontier(summary, run_dir / "figures" / "advisor_achievability_frontier")

    shielded_metrics = metrics[metrics["controller"] == "shielded"]
    range_findings = []
    for range_m, group in shielded_metrics.groupby("range_m", sort=True):
        admitted = group["selected_admitted_split"].astype(bool)
        false_admits = admitted & group["false_admit_selected_matched"].astype(bool)
        admitted_count = int(admitted.sum())
        range_findings.append(
            {
                "range_m": float(range_m),
                "matched_false_admit_count": int(false_admits.sum()),
                "admitted_send_count": admitted_count,
                "matched_false_admit_conditional_pct": (
                    100.0 * float(false_admits.sum()) / admitted_count
                    if admitted_count
                    else float("nan")
                ),
                "over_budget_pct": 100.0 * float(group["shield_over_budget"].mean()),
            }
        )
    range_findings_frame = pd.DataFrame(range_findings)

    headline_columns = [
        "epsilon_m",
        "controller",
        "frames",
        "over_budget_pct",
        "feasible_pct",
        "split_pct",
        "capture_attempt_pct",
        "false_reject_conditional_pct",
        "matched_false_reject_conditional_pct",
        "matched_false_admit_count",
        "admitted_send_count",
        "matched_false_admit_conditional_pct",
        "matched_false_admit_ci95_high_pct",
    ]
    cell_columns = [
        "cell_id",
        "controller",
        "over_budget_pct",
        "feasible_pct",
        "split_pct",
        "capture_attempt_pct",
        "degraded_tier_pct",
        "mean_prb_cost",
        "matched_false_admit_count",
        "admitted_send_count",
        "matched_false_admit_conditional_pct",
        "matched_false_admit_ci95_high_pct",
        "false_reject_count",
        "true_feasible_frame_count",
        "false_reject_conditional_pct",
        "matched_false_reject_conditional_pct",
        "mean_matched_true_scored_reward_finite",
        "observation_coverage_pct",
    ]
    report_lines = [
        "# Track A advisor sweep" + (" — smoke" if smoke else ""),
        "",
        "**Status:** advisor-facing characterization complete; no epsilon, preferred-core, or range value is "
        "selected by this report.",
        "",
        ("Scope is the 3 epsilon x 2 preferred-core x 2 range grid" if not smoke else "Scope is a two-corner smoke test")
        + " at fixed `ucb_k=0` and C1=0.70, using "
        "the same three held-out vehicle replays, paired channel seeds, and per-tick latency common random "
        "numbers. Both the deployable shielded oracle and non-deployable clairvoyant upper bound are shown. "
        "No CARLA, OAI, LOCAL, RL, or model training was run.",
        "",
        "## Headline: per-epsilon feasibility",
        "",
        markdown_table(per_epsilon[headline_columns].round(4)),
        "",
        "`over_budget_pct` is the direct achievability signal: no action in that controller's raw-safe set met "
        "the frame target. It is reported before interpreting reward or mode mix.",
        "",
        "## Range boundary diagnostic (shielded controller, pooled across executed epsilon/core cells)",
        "",
        markdown_table(range_findings_frame.round(4)),
        "",
        "The 40 m result is not merely less feasible: its matched/tracked false-admit rate is materially larger. "
        "This supports retaining 25 m as the headline operating region and treating 40 m only as a diagnostic "
        "until the observation/risk residual is understood and live-validated.",
        "",
        "## All cells",
        "",
        markdown_table(summary[cell_columns].round(4)),
        "",
        "## Interpretation",
        "",
        "- `aggregation=max` is a worst-object, per-frame bottleneck: one in-scope object's risk can make every "
        "whole-frame action infeasible. Object-selective transmission/scheduling is the explicit phase-2 relief, "
        "not an unmodeled fix applied here.",
        "- Range<=25 m is the headline measured-validity operating region. The 40 m cells are labelled "
        "extrapolative sensitivity and must not silently replace the 25 m result.",
        "- The preferred-core value is a quality preference tier, not a hard safety floor; degraded profiles "
        "remain available under graceful degradation.",
        "- Shielded-versus-clairvoyant gaps measure observability/estimation cost inside the surrogate; the "
        "clairvoyant controller is not deployable.",
        "- Vehicle-only replay, thin admitted-SPLIT denominators, and 90-KiB-anchored payload/FPS projections "
        "remain. Zero point estimates require their counts and descriptive Wilson intervals.",
        "",
        "## Decision boundary",
        "",
        "This run supplies evidence for the advisor discussion only. It does not rank or lock epsilon, the "
        "preferred segmentation core, or range, and it does not authorize LOCAL or RL training.",
        "",
        "## Artifacts",
        "",
        f"Run directory: `{run_dir.relative_to(REPO_ROOT)}`",
        "",
        "See `per_frame_metrics.csv`, `summary.csv`, `per_epsilon_summary.csv`, `per_replay_summary.csv`, "
        "`replay_registry.csv`, `resolved_config.yaml`, `manifest.json`, and "
        "`figures/advisor_achievability_frontier.{png,pdf}`.",
        "",
    ]
    report = "\n".join(report_lines)
    report_name = "ADVISOR_SWEEP_SMOKE_RESULTS.md" if smoke else "ADVISOR_SWEEP_RESULTS.md"
    (run_dir / report_name).write_text(report, encoding="utf-8")
    if not smoke:
        (REPO_ROOT / "rl_agent" / "policy" / "ADVISOR_SWEEP_RESULTS.md").write_text(
            report, encoding="utf-8"
        )
    write_run_files(
        run_dir,
        config,
        metrics,
        summary,
        {
            "run_type": "advisor_sweep_smoke" if smoke else "advisor_sweep_3x2x2",
            "controllers": ["shielded", "clairvoyant"],
            "grid_cells": [
                {
                    "epsilon_m": epsilon,
                    "preferred_core_kib": core,
                    "range_m": range_m,
                    "range_extrapolative": range_m > 25.0,
                }
                for epsilon, core, range_m in cells
            ],
            "fixed_shield": dict(fixed_shield),
            "common_random_latency_by_tick": bool(section["common_random_latency_by_tick"]),
            "selected_replays": selected_manifest(by_range),
            "source_hashes": source_hashes(surface),
            "selection_made": False,
            "headline_metric": "per-epsilon over_budget_pct and feasible_pct",
            "limitations": [
                "synthetic Markov channel composed with real CARLA scene replay",
                "vehicle-only replay ground truth",
                "three held-out replay episodes",
                "40 m cells are extrapolative sensitivity",
                "channel payload/FPS projection outside measured anchors",
                "surrogate validation only",
            ],
        },
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(run(args.config, smoke=args.smoke))


if __name__ == "__main__":
    main()
