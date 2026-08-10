"""Run the pre-registered Track A reward one-at-a-time robustness study."""

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


Cell = tuple[str, str, float]


def _grid_cells(config: dict) -> list[Cell]:
    cells: list[Cell] = [("baseline", "baseline", 0.0)]
    for knob in ("w_error", "lambda_prb", "w_task"):
        low, high = [float(value) for value in config["reward"]["one_at_a_time_sensitivity"][knob]]
        cells.extend([(f"{knob}_low", knob, low), (f"{knob}_high", knob, high)])
    return cells


def _add_behavior_diagnostics(metrics: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    aggregate = metrics.groupby("cell_id").agg(
        mean_expected_task_utility=("shield_expected_task_utility", "mean"),
        mean_expected_localization_m=("shield_expected_g_m", "mean"),
    )
    result = result.merge(aggregate, left_on="cell_id", right_index=True, how="left")
    baseline = metrics[metrics["cell_id"] == "baseline"].set_index(["scenario", "step_index"])
    action_changes = {}
    safe_set_changes = {}
    for cell_id, group in metrics.groupby("cell_id"):
        indexed = group.set_index(["scenario", "step_index"]).reindex(baseline.index)
        action_changes[cell_id] = int((indexed["action_id"] != baseline["action_id"]).sum())
        safe_set_changes[cell_id] = int(
            (indexed["raw_safe_action_ids"] != baseline["raw_safe_action_ids"]).sum()
        )
    result["selected_action_changes_vs_baseline"] = result["cell_id"].map(action_changes)
    result["raw_safe_set_changes_vs_baseline"] = result["cell_id"].map(safe_set_changes)
    baseline_frames = int(len(baseline))
    result["selected_action_changes_vs_baseline_pct"] = (
        100.0 * result["selected_action_changes_vs_baseline"] / baseline_frames
    )
    return result


def _plot_behavior(summary: pd.DataFrame, path_prefix: Path) -> None:
    ordered = summary.set_index("cell_id").loc[
        [
            "baseline",
            "w_error_low",
            "w_error_high",
            "lambda_prb_low",
            "lambda_prb_high",
            "w_task_low",
            "w_task_high",
        ]
    ]
    labels = [value.replace("lambda_", "lam_").replace("_", "\n") for value in ordered.index]
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))
    specifications = [
        ("split_pct", "SPLIT schedule (%)", "#4472C4"),
        ("capture_attempt_pct", "Capture attempts (%)", "#70AD47"),
        ("mean_expected_task_utility", "Expected map task utility", "#ED7D31"),
        ("mean_prb_cost", "Mean realized PRB cost", "#8064A2"),
    ]
    for axis, (column, title, color) in zip(axes.flat, specifications):
        axis.bar(labels, ordered[column], color=color)
        axis.set_title(title)
        axis.tick_params(axis="x", labelsize=8)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Track A reward one-at-a-time robustness (all safety settings fixed)")
    figure.tight_layout()
    figure.savefig(path_prefix.with_suffix(".png"), dpi=300)
    figure.savefig(path_prefix.with_suffix(".pdf"))
    plt.close(figure)


def run(config_path: Path | None = None) -> Path:
    config = load_config(config_path)
    section = config["reward_sensitivity"]
    fixed = section["fixed_point"]
    config["safety"].update(
        {
            "epsilon_m": float(fixed["epsilon_m"]),
            "range_m": float(fixed["range_m"]),
            "ucb_k": float(fixed["ucb_k"]),
            "c1_pessimism_factor": float(fixed["c1_pessimism_factor"]),
        }
    )
    config["actions"]["preferred_core_kib"] = int(fixed["preferred_core_kib"])
    cells = _grid_cells(config)
    surface = ChannelSurface(config)
    registry, by_range = prepare_replays(config, [float(fixed["range_m"])])
    selected = by_range[float(fixed["range_m"])]
    run_dir = new_run_directory("reward_sensitivity")
    registry_frame(registry).to_csv(run_dir / "replay_registry.csv", index=False)

    rows: list[dict] = []
    baseline_values = {name: float(config["reward"][name]) for name in ("w_error", "lambda_prb", "w_task")}
    for index, (cell_id, knob, value) in enumerate(cells, start=1):
        cell_config = copy.deepcopy(config)
        if knob != "baseline":
            cell_config["reward"][knob] = value
        print(f"[{index}/{len(cells)}] {cell_id}", flush=True)
        rows.extend(
            run_cell(
                cell_config,
                surface,
                selected,
                ["shielded"],
                bool(section["common_random_latency_by_tick"]),
                {
                    "cell_id": cell_id,
                    "varied_knob": knob,
                    "varied_value": value if knob != "baseline" else 0.0,
                    "w_error": float(cell_config["reward"]["w_error"]),
                    "lambda_prb": float(cell_config["reward"]["lambda_prb"]),
                    "w_task": float(cell_config["reward"]["w_task"]),
                },
            )
        )

    metrics = pd.DataFrame(rows)
    summary = summarize_frames(
        metrics,
        group_keys=[
            "cell_id",
            "varied_knob",
            "varied_value",
            "w_error",
            "lambda_prb",
            "w_task",
            "controller",
        ],
    )
    summary = _add_behavior_diagnostics(metrics, summary)
    order = {cell_id: index for index, (cell_id, _, _) in enumerate(cells)}
    summary["display_order"] = summary["cell_id"].map(order)
    summary = summary.sort_values("display_order").drop(columns="display_order")
    per_replay = summarize_frames(
        metrics,
        group_keys=["cell_id", "varied_knob", "varied_value", "scenario", "controller"],
    )
    per_replay["display_order"] = per_replay["cell_id"].map(order)
    per_replay = per_replay.sort_values(["display_order", "scenario"]).drop(columns="display_order")
    per_replay.to_csv(run_dir / "per_replay_summary.csv", index=False)
    _plot_behavior(summary, run_dir / "figures" / "reward_oat_behavior")

    split_span = float(summary["split_pct"].max() - summary["split_pct"].min())
    over_budget_span = float(summary["over_budget_pct"].max() - summary["over_budget_pct"].min())
    max_action_change = int(summary["selected_action_changes_vs_baseline"].max())
    baseline_frame_count = int((metrics["cell_id"] == "baseline").sum())
    display_columns = [
        "cell_id",
        "w_error",
        "lambda_prb",
        "w_task",
        "split_pct",
        "capture_attempt_pct",
        "degraded_tier_pct",
        "over_budget_pct",
        "matched_false_admit_count",
        "admitted_send_count",
        "matched_false_admit_conditional_pct",
        "false_reject_conditional_pct",
        "mean_expected_task_utility",
        "mean_expected_localization_m",
        "mean_prb_cost",
        "mean_matched_true_scored_reward_finite",
        "selected_action_changes_vs_baseline",
        "selected_action_changes_vs_baseline_pct",
        "raw_safe_set_changes_vs_baseline",
    ]
    report_lines = [
        "# Track A reward one-at-a-time sensitivity",
        "",
        "**Scope:** the pre-registered seven cells (baseline plus low/high `w_error`, `lambda_prb`, and "
        "`w_task`) at epsilon=2.0 m, preferred core=90 KiB, range<=25 m, `ucb_k=0`, and C1=0.70. "
        "Replay, channel seeds, and per-tick latency shocks are paired. No CARLA, OAI, LOCAL, RL, or model "
        "training was run.",
        "",
        "## Robustness result",
        "",
        f"Across the seven cells, SPLIT scheduling spans {split_span:.3f} percentage points and shield "
        f"over-budget spans {over_budget_span:.3f} points. The largest paired action change from baseline is "
        f"{max_action_change}/{baseline_frame_count} frames.",
        "",
        "Absolute reward values are not compared across cells because changing a reward weight changes the "
        "units of the scalar objective. The defensible comparison is behavior and physical components: mode, "
        "capture rate, task utility, localization, PRB cost, feasibility, and paired action changes.",
        "",
        "## All cells",
        "",
        markdown_table(summary[display_columns].round(4)),
        "",
        "## Guardrails",
        "",
        "- Reward weights do not directly alter C1/C2 equations, but changed actions can alter later map state "
        "and therefore later raw-safe sets; paired raw-safe-set changes are reported rather than assumed zero.",
        "- Safety rates retain their conditional denominators and should not be read from unconditional frame "
        "percentages.",
        "- Vehicle-only replay, thin admitted-SPLIT support, and 90-KiB-anchored payload/FPS projections remain.",
        "",
        "## Artifacts",
        "",
        f"Run directory: `{run_dir.relative_to(REPO_ROOT)}`",
        "",
        "See `per_frame_metrics.csv`, `summary.csv`, `per_replay_summary.csv`, `replay_registry.csv`, "
        "`resolved_config.yaml`, `manifest.json`, and `figures/reward_oat_behavior.{png,pdf}`.",
        "",
    ]
    report = "\n".join(report_lines)
    (run_dir / "REWARD_SENSITIVITY_RESULTS.md").write_text(report, encoding="utf-8")
    (REPO_ROOT / "rl_agent" / "policy" / "REWARD_SENSITIVITY_RESULTS.md").write_text(
        report, encoding="utf-8"
    )
    write_run_files(
        run_dir,
        config,
        metrics,
        summary,
        {
            "run_type": "reward_one_at_a_time_7_cell",
            "controllers": ["shielded"],
            "cells": [
                {"cell_id": cell_id, "varied_knob": knob, "varied_value": value}
                for cell_id, knob, value in cells
            ],
            "baseline_reward_weights": baseline_values,
            "fixed_point": dict(fixed),
            "common_random_latency_by_tick": bool(section["common_random_latency_by_tick"]),
            "selected_replays": selected_manifest(by_range),
            "source_hashes": source_hashes(surface),
            "comparison_guardrail": "absolute scalar rewards are not comparable across changed reward weights",
            "limitations": [
                "synthetic Markov channel composed with real CARLA scene replay",
                "vehicle-only replay ground truth",
                "three held-out replay episodes",
                "channel payload/FPS projection outside measured anchors",
                "surrogate validation only",
            ],
        },
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    print(run(args.config))


if __name__ == "__main__":
    main()
