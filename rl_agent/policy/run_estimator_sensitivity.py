"""Run the identifiable Track A telemetry-lag x estimate-noise sensitivity."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_matplotlib_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .channel import ChannelSurface
from .config import REPO_ROOT, load_config
from .replay import registry_frame
from .reporting import new_run_directory, summarize_frames, write_run_files
from .sweep_support import markdown_table, prepare_replays, run_cell, selected_manifest, source_hashes


Cell = tuple[int, float]


def _cell_id(lag_steps: int, noise_fraction: float) -> str:
    return f"lag{lag_steps}__noise{noise_fraction:.2f}"


def _grid_cells(config: dict) -> list[Cell]:
    section = config["estimator_sensitivity"]
    return [
        (int(lag), float(noise))
        for lag in section["telemetry_lag_steps_values"]
        for noise in section["estimate_noise_fraction_values"]
    ]


def _add_paired_diagnostics(metrics: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    baseline_id = _cell_id(2, 0.05)
    baseline_summary = result[result["cell_id"] == baseline_id]
    if len(baseline_summary) != 1:
        raise AssertionError("estimator grid must contain exactly one lag=2/noise=0.05 baseline")
    baseline = baseline_summary.iloc[0]
    result["false_reject_recovered_pp"] = (
        float(baseline["false_reject_conditional_pct"]) - result["false_reject_conditional_pct"]
    )
    result["matched_false_reject_recovered_pp"] = (
        float(baseline["matched_false_reject_conditional_pct"])
        - result["matched_false_reject_conditional_pct"]
    )
    result["finite_matched_reward_delta"] = (
        result["mean_matched_true_scored_reward_finite"]
        - float(baseline["mean_matched_true_scored_reward_finite"])
    )
    result["capture_attempt_delta_pp"] = (
        result["capture_attempt_pct"] - float(baseline["capture_attempt_pct"])
    )

    baseline_frames = metrics[metrics["cell_id"] == baseline_id].set_index(["scenario", "step_index"])
    action_changes = {}
    safe_set_changes = {}
    for cell_id, group in metrics.groupby("cell_id"):
        indexed = group.set_index(["scenario", "step_index"])
        if not indexed.index.equals(baseline_frames.index):
            indexed = indexed.reindex(baseline_frames.index)
        action_changes[cell_id] = int((indexed["action_id"] != baseline_frames["action_id"]).sum())
        safe_set_changes[cell_id] = int(
            (indexed["raw_safe_action_ids"] != baseline_frames["raw_safe_action_ids"]).sum()
        )
    result["selected_action_changes_vs_baseline"] = result["cell_id"].map(action_changes)
    result["raw_safe_set_changes_vs_baseline"] = result["cell_id"].map(safe_set_changes)
    return result


def _heatmap(axis, summary: pd.DataFrame, value: str, title: str, digits: int = 2) -> None:
    pivot = summary.pivot(
        index="telemetry_lag_steps", columns="estimate_noise_fraction", values=value
    ).sort_index(ascending=False)
    image = axis.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="cividis")
    axis.set_xticks(range(len(pivot.columns)), [f"{value:.2f}" for value in pivot.columns])
    axis.set_yticks(range(len(pivot.index)), [str(int(value)) for value in pivot.index])
    axis.set_xlabel("Estimate noise fraction")
    axis.set_ylabel("Telemetry lag (steps)")
    axis.set_title(title)
    for row_index in range(len(pivot.index)):
        for column_index in range(len(pivot.columns)):
            cell_value = float(pivot.iloc[row_index, column_index])
            label = "NA" if not np.isfinite(cell_value) else f"{cell_value:.{digits}f}"
            axis.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.6, "edgecolor": "none", "pad": 1},
            )
    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def _plot_surface(summary: pd.DataFrame, path_prefix: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    specifications = [
        ("false_reject_conditional_pct", "Full-GT false reject | feasible (%)", 2),
        ("matched_false_reject_conditional_pct", "Tracked false reject | feasible (%)", 2),
        ("mean_matched_true_scored_reward_finite", "Finite tracked true-scored reward", 3),
        ("capture_attempt_pct", "Capture attempts (%)", 2),
    ]
    for axis, (column, title, digits) in zip(axes.flat, specifications):
        _heatmap(axis, summary, column, title, digits)
    figure.suptitle("Track A estimator-quality sensitivity (paired channel and latency randomness)")
    figure.tight_layout()
    figure.savefig(path_prefix.with_suffix(".png"), dpi=300)
    figure.savefig(path_prefix.with_suffix(".pdf"))
    plt.close(figure)


def run(config_path: Path | None = None, smoke: bool = False) -> Path:
    config = load_config(config_path)
    section = config["estimator_sensitivity"]
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
    if smoke:
        cells = [(0, 0.0), (2, 0.05), (4, 0.10)]

    surface = ChannelSurface(config)
    registry, by_range = prepare_replays(config, [float(fixed["range_m"])])
    selected = by_range[float(fixed["range_m"])]
    run_dir = new_run_directory("estimator_sensitivity_smoke" if smoke else "estimator_sensitivity")
    registry_frame(registry).to_csv(run_dir / "replay_registry.csv", index=False)
    rows: list[dict] = []
    for index, (lag_steps, noise_fraction) in enumerate(cells, start=1):
        cell_config = copy.deepcopy(config)
        cell_config["channel"]["telemetry_lag_steps"] = lag_steps
        cell_config["channel"]["estimate_noise_fraction"] = noise_fraction
        cell_name = _cell_id(lag_steps, noise_fraction)
        print(f"[{index}/{len(cells)}] {cell_name}", flush=True)
        rows.extend(
            run_cell(
                cell_config,
                surface,
                selected,
                ["shielded"],
                bool(section["common_random_latency_by_tick"]),
                {
                    "cell_id": cell_name,
                    "telemetry_lag_steps": lag_steps,
                    "estimate_noise_fraction": noise_fraction,
                },
            )
        )

    metrics = pd.DataFrame(rows)
    summary = summarize_frames(
        metrics,
        group_keys=["cell_id", "telemetry_lag_steps", "estimate_noise_fraction", "controller"],
    ).sort_values(["telemetry_lag_steps", "estimate_noise_fraction"])
    summary = _add_paired_diagnostics(metrics, summary)
    per_replay = summarize_frames(
        metrics,
        group_keys=[
            "cell_id",
            "telemetry_lag_steps",
            "estimate_noise_fraction",
            "scenario",
            "controller",
        ],
    ).sort_values(["telemetry_lag_steps", "estimate_noise_fraction", "scenario"])
    per_replay.to_csv(run_dir / "per_replay_summary.csv", index=False)
    _plot_surface(summary, run_dir / "figures" / "estimator_quality_surface")

    display_columns = [
        "cell_id",
        "false_reject_count",
        "true_feasible_frame_count",
        "false_reject_conditional_pct",
        "false_reject_recovered_pp",
        "matched_false_reject_conditional_pct",
        "matched_false_reject_recovered_pp",
        "matched_false_admit_count",
        "admitted_send_count",
        "matched_false_admit_conditional_pct",
        "split_pct",
        "capture_attempt_pct",
        "over_budget_pct",
        "mean_matched_true_scored_reward_finite",
        "finite_matched_reward_delta",
        "selected_action_changes_vs_baseline",
        "raw_safe_set_changes_vs_baseline",
    ]
    baseline = summary[summary["cell_id"] == _cell_id(2, 0.05)].iloc[0]
    ideal = summary[summary["cell_id"] == _cell_id(0, 0.0)].iloc[0]
    recovered = float(ideal["false_reject_recovered_pp"])
    residual = float(ideal["false_reject_conditional_pct"])
    false_reject_span = float(
        summary["false_reject_conditional_pct"].max()
        - summary["false_reject_conditional_pct"].min()
    )
    max_action_changes = int(summary["selected_action_changes_vs_baseline"].max())
    max_safe_set_changes = int(summary["raw_safe_set_changes_vs_baseline"].max())
    baseline_frame_count = int(baseline["frames"])
    finding = (
        "The tested estimator settings do not explain the headline false-reject gap: the idealized estimator "
        f"recovers {recovered:.2f} percentage points and the entire tested grid spans only "
        f"{false_reject_span:.2f} points. This falsifies the prior hypothesis that lag/noise drives the "
        "approximately 42% rate in this fixed surrogate."
        if abs(recovered) < 0.10 and false_reject_span < 0.25
        else (
            f"The idealized estimator recovers {recovered:.2f} percentage points and the tested grid spans "
            f"{false_reject_span:.2f} points."
        )
    )
    report_lines = [
        "# Track A estimator-quality sensitivity" + (" — smoke" if smoke else ""),
        "",
        "**Scope:** fixed epsilon=2.0 m, preferred core=90 KiB, range<=25 m, `ucb_k=0`, and "
        "`c1_pessimism_factor=0.70`; shielded oracle over the same three held-out vehicle replays. "
        "Only telemetry lag and estimate noise vary. No CARLA, OAI, LOCAL, RL, or model training was run.",
        "",
        "## Paired finding",
        "",
        f"The baseline lag=2/noise=0.05 cell has {baseline['false_reject_conditional_pct']:.2f}% full-GT "
        f"conditional false rejection. The idealized lag=0/noise=0 cell has {residual:.2f}%, a paired "
        f"recovery of {recovered:.2f} percentage points in this surrogate.",
        "",
        finding,
        "",
        f"Estimator settings still change as many as {max_safe_set_changes}/{baseline_frame_count} raw-safe "
        f"sets, but only {max_action_changes}/{baseline_frame_count} selected actions. Reward/preference "
        "narrowing and map-state dynamics "
        "absorb most availability changes at this operating point.",
        "",
        "The residual at lag=0/noise=0 is not labelled irreducible: speed uncertainty, observation mismatch, "
        "worst-object aggregation, and map-state trajectory remain mixed in this three-episode vehicle-only "
        "corpus. Those mechanisms need a separate attribution diagnostic before changing the shield or reward.",
        "",
        "## Grid",
        "",
        markdown_table(summary[display_columns].round(4)),
        "",
        "## Guardrails",
        "",
        "- False-reject percentages are conditional on a nonempty clairvoyant raw-safe set; counts are shown.",
        "- Matched/tracked and strict full-GT metrics remain separate; perception misses are not shield errors.",
        "- Any zero false-admit estimate is denominator-limited and must be read with its counts/Wilson interval.",
        "- This sensitivity diagnoses a deterministic table-composed surrogate; it does not calibrate a live "
        "residual/conformal uncertainty model.",
        "",
        "## Artifacts",
        "",
        f"Run directory: `{run_dir.relative_to(REPO_ROOT)}`",
        "",
        "See `per_frame_metrics.csv`, `summary.csv`, `per_replay_summary.csv`, `replay_registry.csv`, "
        "`resolved_config.yaml`, `manifest.json`, and `figures/estimator_quality_surface.{png,pdf}`.",
        "",
    ]
    report = "\n".join(report_lines)
    report_name = "ESTIMATOR_SENSITIVITY_SMOKE_RESULTS.md" if smoke else "ESTIMATOR_SENSITIVITY_RESULTS.md"
    (run_dir / report_name).write_text(report, encoding="utf-8")
    if not smoke:
        (REPO_ROOT / "rl_agent" / "policy" / "ESTIMATOR_SENSITIVITY_RESULTS.md").write_text(
            report, encoding="utf-8"
        )
    write_run_files(
        run_dir,
        config,
        metrics,
        summary,
        {
            "run_type": "estimator_sensitivity_smoke" if smoke else "estimator_sensitivity_4x3",
            "controllers": ["shielded"],
            "grid_cells": [
                {"telemetry_lag_steps": lag, "estimate_noise_fraction": noise}
                for lag, noise in cells
            ],
            "fixed_point": dict(fixed),
            "baseline_cell": {"telemetry_lag_steps": 2, "estimate_noise_fraction": 0.05},
            "common_random_latency_by_tick": bool(section["common_random_latency_by_tick"]),
            "selected_replays": selected_manifest(by_range),
            "source_hashes": source_hashes(surface),
            "metric_denominators": {
                "matched_false_admit_conditional_pct": "selected raw-safe SPLIT schedules",
                "false_reject_conditional_pct": "full-GT clairvoyant-feasible frames",
                "matched_false_reject_conditional_pct": "tracked-object clairvoyant-feasible frames",
            },
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
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(run(args.config, smoke=args.smoke))


if __name__ == "__main__":
    main()
