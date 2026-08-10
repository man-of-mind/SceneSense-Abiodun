"""Run the fixed-point Track A safety-conservativeness calibration grid."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_matplotlib_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .catalog import flatten_actions, load_profile_catalog
from .channel import ChannelProcess, ChannelSurface
from .config import REPO_ROOT, load_config
from .env import SurrogateEnv
from .oracles import run_oracle
from .replay import TraceRecord, discover_trace_registry, registry_frame
from .reporting import new_run_directory, summarize_frames, write_run_files
from .run_pilot import _select_episodes


Cell = Tuple[float, float]


def _markdown_table(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return "```text\n" + frame.to_string(index=False) + "\n```"


def _cell_id(ucb_k: float, pessimism: float) -> str:
    return f"ucb{ucb_k:.1f}__c1{pessimism:.1f}"


def _grid_cells(config: dict) -> List[Cell]:
    calibration = config["safety_calibration"]
    return [
        (float(ucb_k), float(pessimism))
        for pessimism in calibration["c1_pessimism_factor_values"]
        for ucb_k in calibration["ucb_k_values"]
    ]


def _pareto_mask(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    values = frame[list(columns)].to_numpy(dtype=float)
    keep = np.ones(len(frame), dtype=bool)
    for index, candidate in enumerate(values):
        if not np.isfinite(candidate).all():
            keep[index] = False
            continue
        dominated = np.all(values <= candidate, axis=1) & np.any(values < candidate, axis=1)
        dominated[index] = False
        keep[index] = not bool(dominated.any())
    return pd.Series(keep, index=frame.index)


def _axis_effect_diagnostics(metrics: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    definitions = [
        ("ucb_k", "c1_pessimism_factor"),
        ("c1_pessimism_factor", "ucb_k"),
    ]
    frame_index = ["scenario", "step_index"]
    span_columns = [
        "split_pct",
        "capture_attempt_pct",
        "over_budget_pct",
        "matched_false_admit_conditional_pct",
        "false_reject_conditional_pct",
        "c1_estimate_miss_pct_attempted",
    ]
    for varied, fixed in definitions:
        for fixed_value, group in metrics.groupby(fixed, sort=True):
            action_pivot = group.pivot(index=frame_index, columns=varied, values="action_id")
            safe_set_pivot = group.pivot(index=frame_index, columns=varied, values="raw_safe_action_ids")
            summary_group = summary[np.isclose(summary[fixed], float(fixed_value))]
            row = {
                "varied_knob": varied,
                "fixed_knob": fixed,
                "fixed_value": float(fixed_value),
                "cell_count": int(group[varied].nunique()),
                "frames": int(len(action_pivot)),
                "selected_action_variation_count": int((action_pivot.nunique(axis=1) > 1).sum()),
                "selected_action_variation_pct": 100.0
                * float((action_pivot.nunique(axis=1) > 1).mean()),
                "raw_safe_set_variation_count": int((safe_set_pivot.nunique(axis=1) > 1).sum()),
                "raw_safe_set_variation_pct": 100.0
                * float((safe_set_pivot.nunique(axis=1) > 1).mean()),
                "max_selected_risk_sigma_m": float(summary_group["max_risk_sigma_m"].max()),
            }
            for column in span_columns:
                finite = summary_group[column].replace([np.inf, -np.inf], np.nan).dropna()
                row[f"{column}_span_pp"] = float(finite.max() - finite.min()) if len(finite) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _plot_tradeoffs(summary: pd.DataFrame, path_prefix: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, summary["c1_pessimism_factor"].nunique()))
    for color, (pessimism, group) in zip(
        colors, summary.groupby("c1_pessimism_factor", sort=True)
    ):
        group = group.sort_values("ucb_k")
        x_c2 = group["matched_false_admit_conditional_pct"].to_numpy(dtype=float)
        y = group["false_reject_conditional_pct"].to_numpy(dtype=float)
        axes[0].plot(x_c2, y, marker="o", color=color, label=f"C1 factor={pessimism:.1f}")
        x_c1 = group["c1_estimate_miss_pct_attempted"].to_numpy(dtype=float)
        axes[1].plot(x_c1, y, marker="o", color=color, label=f"C1 factor={pessimism:.1f}")
        for _, row in group.iterrows():
            axes[0].annotate(
                f"k={row['ucb_k']:.1f}",
                (row["matched_false_admit_conditional_pct"], row["false_reject_conditional_pct"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
            axes[1].annotate(
                f"k={row['ucb_k']:.1f}",
                (row["c1_estimate_miss_pct_attempted"], row["false_reject_conditional_pct"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
    axes[0].set_xlabel("C2 matched false admit | admitted SPLIT (%)")
    axes[0].set_ylabel("False reject | truly feasible frame (%)")
    axes[0].set_title("C2 localization calibration")
    axes[1].set_xlabel("C1 estimate miss | capture attempt (%)")
    axes[1].set_ylabel("False reject | truly feasible frame (%)")
    axes[1].set_title("C1 congestion calibration")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Track A safety calibration (fixed epsilon=2 m, core=90 KiB, range<=25 m)")
    figure.tight_layout()
    figure.savefig(path_prefix.with_suffix(".png"), dpi=300)
    figure.savefig(path_prefix.with_suffix(".pdf"))
    plt.close(figure)


def _heatmap(axis, summary: pd.DataFrame, value: str, title: str) -> None:
    pivot = summary.pivot(index="ucb_k", columns="c1_pessimism_factor", values=value).sort_index(
        ascending=False
    )
    image = axis.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="cividis")
    axis.set_xticks(range(len(pivot.columns)), [f"{value:.1f}" for value in pivot.columns])
    axis.set_yticks(range(len(pivot.index)), [f"{value:.1f}" for value in pivot.index])
    axis.set_xlabel("C1 pessimism factor")
    axis.set_ylabel("UCB k")
    axis.set_title(title)
    for row_index in range(len(pivot.index)):
        for column_index in range(len(pivot.columns)):
            value_at_cell = float(pivot.iloc[row_index, column_index])
            axis.text(
                column_index,
                row_index,
                f"{value_at_cell:.1f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black",
                bbox={"facecolor": "white", "alpha": 0.55, "edgecolor": "none", "pad": 1},
            )
    return image


def _plot_operations(summary: pd.DataFrame, path_prefix: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    specifications = [
        ("split_pct", "SPLIT schedule (%)"),
        ("capture_attempt_pct", "Actual capture attempts (%)"),
        ("over_budget_pct", "Shield over-budget frames (%)"),
    ]
    for axis, (column, title) in zip(axes, specifications):
        image = _heatmap(axis, summary, column, title)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("Operating behavior across the safety-calibration grid")
    figure.tight_layout()
    figure.savefig(path_prefix.with_suffix(".png"), dpi=300)
    figure.savefig(path_prefix.with_suffix(".pdf"))
    plt.close(figure)


def _selected_manifest(selected: Iterable[Tuple[TraceRecord, list]]) -> list:
    rows = []
    for record, frames in selected:
        rows.append(
            {
                "episode_id": record.episode_id,
                "run_group": record.run_group,
                "scenario_family": record.scenario_family,
                "split": record.split,
                "ground_truth_path": str(record.ground_truth_path.relative_to(REPO_ROOT)),
                "ground_truth_sha256": record.ground_truth_sha256,
                "prediction_path": str(record.prediction_path.relative_to(REPO_ROOT)),
                "prediction_sha256": record.prediction_sha256,
                "frame_count": len(frames),
            }
        )
    return rows


def run(config_path: Path | None = None, smoke: bool = False) -> Path:
    config = load_config(config_path)
    calibration = config["safety_calibration"]
    fixed = calibration["fixed_point"]
    config["safety"]["epsilon_m"] = float(fixed["epsilon_m"])
    config["safety"]["range_m"] = float(fixed["range_m"])
    config["actions"]["preferred_core_kib"] = int(fixed["preferred_core_kib"])
    cells = _grid_cells(config)
    if smoke:
        cells = [(1.0, 0.7), (0.0, 1.0), (2.0, 0.6)]

    profiles = load_profile_catalog(config["actions"]["catalog_csv"])
    actions = flatten_actions(profiles, config["actions"]["fps"], config["actions"]["preferred_core_kib"])
    surface = ChannelSurface(config)
    registry = discover_trace_registry(config)
    selected = _select_episodes(config, registry)
    run_kind = "safety_calibration_smoke" if smoke else "safety_calibration"
    run_dir = new_run_directory(run_kind)
    registry_frame(registry).to_csv(run_dir / "replay_registry.csv", index=False)

    rows = []
    channel_seeds = [int(value) for value in config["pilot"]["channel_seeds"]]
    for cell_index, (ucb_k, pessimism) in enumerate(cells, start=1):
        cell_config = copy.deepcopy(config)
        cell_config["safety"]["ucb_k"] = ucb_k
        cell_config["safety"]["c1_pessimism_factor"] = pessimism
        cell_name = _cell_id(ucb_k, pessimism)
        print(f"[{cell_index}/{len(cells)}] {cell_name}", flush=True)
        for episode_index, (record, frames) in enumerate(selected):
            seed = channel_seeds[episode_index % len(channel_seeds)]
            channel = ChannelProcess(cell_config, surface, seed)
            env = SurrogateEnv(
                cell_config,
                frames,
                actions,
                channel,
                surface,
                seed + 10_000,
                latency_mode="sample",
                latency_crn_by_tick=bool(calibration["common_random_latency_by_tick"]),
            )
            result = run_oracle(env, "shielded")
            for row in result.rows:
                row["scenario"] = record.run_group
                row["scenario_family"] = record.scenario_family
                row["replay_split"] = record.split
                row["cell_id"] = cell_name
                row["ucb_k"] = ucb_k
                row["c1_pessimism_factor"] = pessimism
                rows.append(row)

    metrics = pd.DataFrame(rows)
    summary = summarize_frames(
        metrics,
        group_keys=["cell_id", "ucb_k", "c1_pessimism_factor", "controller"],
    ).sort_values(["c1_pessimism_factor", "ucb_k"])
    per_replay = summarize_frames(
        metrics,
        group_keys=[
            "cell_id",
            "ucb_k",
            "c1_pessimism_factor",
            "scenario",
            "controller",
        ],
    ).sort_values(["c1_pessimism_factor", "ucb_k", "scenario"])
    summary["roc_nondominated"] = _pareto_mask(
        summary,
        ["matched_false_admit_conditional_pct", "false_reject_conditional_pct"],
    )
    axis_diagnostics = _axis_effect_diagnostics(metrics, summary)
    per_replay.to_csv(run_dir / "per_replay_summary.csv", index=False)
    axis_diagnostics.to_csv(run_dir / "axis_effect_diagnostics.csv", index=False)
    _plot_tradeoffs(summary, run_dir / "figures" / "safety_tradeoff")
    _plot_operations(summary, run_dir / "figures" / "operating_surface")

    display_columns = [
        "cell_id",
        "matched_false_admit_count",
        "admitted_send_count",
        "matched_false_admit_conditional_pct",
        "matched_false_admit_ci95_high_pct",
        "false_reject_count",
        "true_feasible_frame_count",
        "false_reject_conditional_pct",
        "split_pct",
        "capture_attempt_pct",
        "over_budget_pct",
        "c1_estimate_miss_count",
        "attempts",
        "c1_estimate_miss_pct_attempted",
        "mean_true_scored_reward",
        "mean_matched_true_scored_reward",
        "mean_matched_true_scored_reward_finite",
        "matched_true_reward_finite_frame_count",
        "mean_predicted_reward",
        "mean_prb_cost",
        "max_risk_sigma_m",
        "oracle_action_set_mismatch_pct",
        "shield_skip_clairvoyant_split_pct",
        "roc_nondominated",
    ]
    anchor = summary[
        np.isclose(summary["ucb_k"], 1.0)
        & np.isclose(summary["c1_pessimism_factor"], 0.7)
    ]
    report_lines = [
        "# Track A safety calibration",
        "",
        "**Status:** fixed-point safety characterization complete; no operating point selected.",
        "",
        "Scope is fixed at epsilon=2.0 m, preferred core=90 KiB, range<=25 m, the same three held-out "
        "vehicle replay episodes, and channel seeds `[1101, 2202, 3303]`. CARLA, OAI, LOCAL, RL, reward-weight "
        "sensitivity, and the 3x2x2 advisor sweep were not run.",
        "",
        "## Metric contract",
        "",
        "- Raw shield-safe actions `{B<=epsilon}` are recorded before preferred-core/reward narrowing.",
        "- C2 false-admit rate is conditional on selected raw-safe SPLIT schedules; counts and a descriptive "
        "95% Wilson interval are reported.",
        "- False-reject rate is conditional on frames where the full-GT clairvoyant raw-safe set is nonempty; "
        "the numerator is zero overlap with the deployable raw-safe set.",
        "- C1 estimate-miss rate is conditional on actual capture attempts and remains separate from C2.",
        "- `split_pct` is schedule selection; `capture_attempt_pct` is the actual target-FPS send rate.",
        "- `mean_matched_true_scored_reward` evaluates the selected action with tracked-object hidden truth; "
        "`mean_true_scored_reward` is the strict end-to-end GT score and may be dominated by unobserved objects; "
        "`mean_matched_true_scored_reward_finite` excludes the explicit unobserved sentinel for interpretability; "
        "`mean_predicted_reward` is the deployable model's score.",
        "- Latency shocks are common random numbers indexed by episode and control tick, so policy-dependent "
        "capture counts do not desynchronize cells.",
        "",
        "## Current-pilot configuration anchor",
        "",
        _markdown_table(anchor[display_columns].round(4)),
        "",
        "## All calibration cells",
        "",
        _markdown_table(summary[display_columns].round(4)),
        "",
        "## Axis identifiability diagnostics",
        "",
        _markdown_table(axis_diagnostics.round(4)),
        "",
        "## Interpretation guardrails",
        "",
        "- `roc_nondominated` is descriptive over conditional C2 false admission and full-GT false rejection; "
        "it is not an automatic operating-point recommendation.",
        "- A zero action/safe-set variation span means that knob is not identifiable under this surrogate "
        "construction; do not choose its value from a flat curve.",
        "- Wilson intervals treat frames as binomial trials and are descriptive only because replay frames "
        "are temporally correlated.",
        "- Matched/tracked C2 and strict end-to-end GT exposure remain distinct; observation coverage is not "
        "reinterpreted as shield error.",
        "- The corpus is vehicle-only and all payload/FPS projection caveats from the pilot remain in force.",
        "- Stop here for Abiodun/advisor review; do not launch the reward or 3x2x2 sweeps yet.",
        "",
        "## Artifacts",
        "",
        f"Run directory: `{run_dir.relative_to(REPO_ROOT)}`",
        "",
        "See `per_frame_metrics.csv`, `summary.csv`, `per_replay_summary.csv`, "
        "`axis_effect_diagnostics.csv`, `replay_registry.csv`, "
        "`resolved_config.yaml`, `manifest.json`, and `figures/*.{png,pdf}`.",
        "",
    ]
    report = "\n".join(report_lines)
    report_name = "SAFETY_CALIBRATION_SMOKE_RESULTS.md" if smoke else "SAFETY_CALIBRATION_RESULTS.md"
    (run_dir / report_name).write_text(report, encoding="utf-8")
    if not smoke:
        (REPO_ROOT / "rl_agent" / "policy" / "SAFETY_CALIBRATION_RESULTS.md").write_text(
            report, encoding="utf-8"
        )

    catalog_meta_path = REPO_ROOT / "rl_agent" / "policy" / "data" / "action_catalog.meta.json"
    catalog_meta = json.loads(catalog_meta_path.read_text())
    write_run_files(
        run_dir,
        config,
        metrics,
        summary,
        {
            "run_type": "safety_calibration_smoke" if smoke else "safety_calibration_5x5",
            "controllers": ["shielded"],
            "grid_cells": [
                {"ucb_k": ucb_k, "c1_pessimism_factor": pessimism}
                for ucb_k, pessimism in cells
            ],
            "fixed_point": dict(fixed),
            "common_random_latency_by_tick": bool(calibration["common_random_latency_by_tick"]),
            "selected_replays": _selected_manifest(selected),
            "source_hashes": {**surface.source_hashes, **catalog_meta},
            "metric_denominators": {
                "matched_false_admit_conditional_pct": "selected raw-safe SPLIT schedules",
                "false_reject_conditional_pct": "full-GT clairvoyant-feasible frames",
                "c1_estimate_miss_pct_attempted": "actual capture attempts",
            },
            "limitations": [
                "synthetic Markov channel composed with real CARLA scene replay",
                "vehicle-only replay ground truth",
                "temporally correlated frame-level Wilson intervals",
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
