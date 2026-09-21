#!/usr/bin/env python3
"""Render the preliminary Hybrid-SAC training and fit-validation results.

This is a read-only, deterministic presentation adapter.  It does not load a
checkpoint or execute a policy.  Training diagnostics come from the three
``metrics.csv`` files, while evaluation quantities come from the already
materialized frozen fit-validation tables.  The two populations are never
mixed or relabelled as online measurements.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SEEDS = (17, 29, 43)
UPDATES = tuple(range(0, 5001, 500))
PROFILES = (
    "FAVORABLE_STABLE",
    "MID_VARIABLE",
    "ADVERSE_STABLE",
    "FADE_RECOVERY",
)
PROFILE_LABELS = {
    "FAVORABLE_STABLE": "Favorable stable",
    "MID_VARIABLE": "Mid variable",
    "ADVERSE_STABLE": "Adverse stable",
    "FADE_RECOVERY": "Fade recovery",
}
PROFILE_COLORS = {
    "FAVORABLE_STABLE": "#0072B2",
    "MID_VARIABLE": "#E69F00",
    "ADVERSE_STABLE": "#D55E00",
    "FADE_RECOVERY": "#009E73",
}
SEED_COLORS = {17: "#0072B2", 29: "#D55E00", 43: "#009E73"}
MODE_LABELS = (
    "M0\nnoAE/U8",
    "M1\nnoAE/U6",
    "M2\nnoAE/U4",
    "M3\nAE128/U8",
    "M4\nAE128/U6",
    "M5\nAE128/U4",
    "M6\nAE64/U8",
    "M7\nAE64/U6",
    "M8\nAE64/U4",
    "M9\nAE32/U8",
    "M10\nAE32/U6",
    "M11\nAE32/U4",
)
ROLLING_WINDOW = 100
RESULT_SCHEMA = "splitfusion.hybrid_sac.preliminary_results.v1"


class ResultsError(RuntimeError):
    """Raised when an input result violates the expected evidence shape."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def rolling_mean(values: Sequence[float], window: int = ROLLING_WINDOW) -> np.ndarray:
    """Return only complete trailing windows; never label a short prefix as full."""
    if window <= 0:
        raise ValueError("window must be positive")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError("values must be a non-empty one-dimensional sequence")
    if len(array) < window:
        raise ValueError("window cannot exceed the number of values")
    prefix = np.concatenate(([0.0], np.cumsum(array, dtype=np.float64)))
    totals = prefix[window:] - prefix[:-window]
    return totals / float(window)


def quantile(values: Sequence[float], probability: float) -> float:
    """Linear quantile with an explicit implementation for reproducibility."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot take a quantile of an empty sequence")
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _float(row: Mapping[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ResultsError(f"non-finite {key}: {row[key]!r}")
    return value


def _int(row: Mapping[str, str], key: str) -> int:
    value = int(row[key])
    return value


@dataclass(frozen=True)
class Inputs:
    baseline_root: Path
    validation_root: Path
    training: Mapping[int, Tuple[Mapping[str, str], ...]]
    validation_aggregate: Tuple[Mapping[str, str], ...]
    validation_context: Tuple[Mapping[str, str], ...]
    source_hashes: Mapping[str, str]


def load_inputs(baseline_root: Path, validation_root: Path) -> Inputs:
    training: Dict[int, Tuple[Mapping[str, str], ...]] = {}
    hashes: Dict[str, str] = {}
    for seed in SEEDS:
        path = baseline_root / f"seed_{seed}" / "metrics.csv"
        rows = tuple(read_csv(path))
        if len(rows) != 5000:
            raise ResultsError(f"seed {seed} has {len(rows)} metrics rows, expected 5000")
        indices = [_int(row, "update_index") for row in rows]
        if indices != list(range(1, 5001)):
            raise ResultsError(f"seed {seed} update indexes are not exactly 1..5000")
        for row in rows:
            _float(row, "reward_mean")
        training[seed] = rows
        hashes[str(path)] = sha256_file(path)

    aggregate_path = validation_root / "fit_validation_aggregate.csv"
    context_path = validation_root / "fit_validation_per_context.csv"
    manifest_path = validation_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if type(manifest) is not dict or manifest.get("status") != "COMPLETE":
        raise ResultsError("fit-validation manifest is not COMPLETE")
    if manifest.get("aggregate_row_count") != 165:
        raise ResultsError("fit-validation manifest aggregate row count differs")
    if manifest.get("per_context_row_count") != 11220:
        raise ResultsError("fit-validation manifest per-context row count differs")
    declared_outputs = manifest.get("output_files")
    if type(declared_outputs) is not dict:
        raise ResultsError("fit-validation manifest output_files is not an object")
    for source_path in (aggregate_path, context_path):
        declared_hash = declared_outputs.get(source_path.name)
        if declared_hash != sha256_file(source_path):
            raise ResultsError(f"fit-validation hash mismatch: {source_path.name}")
    aggregate = tuple(read_csv(aggregate_path))
    contexts = tuple(read_csv(context_path))
    if len(aggregate) != len(SEEDS) * len(UPDATES) * (len(PROFILES) + 1):
        raise ResultsError(f"unexpected aggregate row count: {len(aggregate)}")
    if len(contexts) != len(SEEDS) * len(UPDATES) * 340:
        raise ResultsError(f"unexpected per-context row count: {len(contexts)}")

    aggregate_keys = {
        (_int(row, "seed"), _int(row, "update_index"), row["network_profile"])
        for row in aggregate
    }
    expected_keys = {
        (seed, update, profile)
        for seed in SEEDS
        for update in UPDATES
        for profile in ("ALL_PROFILES",) + PROFILES
    }
    if aggregate_keys != expected_keys or len(aggregate_keys) != len(aggregate):
        raise ResultsError("aggregate seed/update/profile keys are incomplete or duplicated")

    context_keys = {
        (_int(row, "seed"), _int(row, "update_index"), _int(row, "panel_index"))
        for row in contexts
    }
    if len(context_keys) != len(contexts):
        raise ResultsError("per-context seed/update/panel keys are duplicated")
    for seed in SEEDS:
        for update in UPDATES:
            rows = [
                row
                for row in contexts
                if _int(row, "seed") == seed and _int(row, "update_index") == update
            ]
            if len(rows) != 340:
                raise ResultsError(f"seed {seed}, update {update} has {len(rows)} panel rows")
            counts = Counter(row["network_profile"] for row in rows)
            if counts != Counter({profile: 85 for profile in PROFILES}):
                raise ResultsError(
                    f"seed {seed}, update {update} profile counts differ: {counts}"
                )

    for path in (aggregate_path, context_path, manifest_path):
        hashes[str(path)] = sha256_file(path)
    return Inputs(
        baseline_root=baseline_root,
        validation_root=validation_root,
        training=training,
        validation_aggregate=aggregate,
        validation_context=contexts,
        source_hashes=hashes,
    )


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 15,
            "figure.titleweight": "bold",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _bold_ticks(axis: plt.Axes) -> None:
    for label in axis.get_xticklabels() + axis.get_yticklabels():
        label.set_fontweight("bold")


def _save(fig: plt.Figure, output_stem: Path) -> Tuple[Path, Path]:
    fig.tight_layout()
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight", metadata={"Software": RESULT_SCHEMA})
    fig.savefig(
        pdf,
        bbox_inches="tight",
        metadata={
            "Title": output_stem.name,
            "Creator": RESULT_SCHEMA,
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)
    return png, pdf


def _aggregate_row(inputs: Inputs, seed: int, update: int, profile: str) -> Mapping[str, str]:
    matches = [
        row
        for row in inputs.validation_aggregate
        if _int(row, "seed") == seed
        and _int(row, "update_index") == update
        and row["network_profile"] == profile
    ]
    if len(matches) != 1:
        raise ResultsError(f"expected one aggregate row for {(seed, update, profile)}")
    return matches[0]


def _seed_series(inputs: Inputs, field: str, profile: str) -> Dict[int, np.ndarray]:
    return {
        seed: np.asarray(
            [_float(_aggregate_row(inputs, seed, update, profile), field) for update in UPDATES],
            dtype=np.float64,
        )
        for seed in SEEDS
    }


def plot_training_reward(inputs: Inputs, output_dir: Path) -> Tuple[Path, Path]:
    fig, axis = plt.subplots(figsize=(11.5, 6.4))
    for seed in SEEDS:
        rows = inputs.training[seed]
        x = np.asarray([_int(row, "update_index") for row in rows])
        y = np.asarray([_float(row, "reward_mean") for row in rows])
        smooth = rolling_mean(y, ROLLING_WINDOW)
        smooth_x = x[ROLLING_WINDOW - 1 :]
        color = SEED_COLORS[seed]
        axis.plot(x, y, color=color, alpha=0.11, linewidth=0.65)
        axis.plot(
            smooth_x,
            smooth,
            color=color,
            linewidth=2.1,
            label=f"Seed {seed}: {ROLLING_WINDOW}-update rolling mean",
        )
    axis.set_title("Training diagnostic: replay-batch reward (not online reward)")
    axis.set_xlabel("SAC gradient updates")
    axis.set_ylabel("Replay-batch reward mean")
    axis.set_xlim(1, 5000)
    axis.legend(frameon=False, ncol=1)
    _bold_ticks(axis)
    return _save(fig, output_dir / "01_training_replay_batch_reward")


def plot_validation_reward(inputs: Inputs, output_dir: Path) -> Tuple[Path, Path]:
    fig, (overall_axis, profile_axis) = plt.subplots(1, 2, figsize=(15.5, 6.1))
    overall = _seed_series(inputs, "reward_mean", "ALL_PROFILES")
    stack = np.vstack([overall[seed] for seed in SEEDS])
    for seed in SEEDS:
        overall_axis.plot(
            UPDATES,
            overall[seed],
            marker="o",
            markersize=3.5,
            linewidth=1.1,
            alpha=0.45,
            color=SEED_COLORS[seed],
            label=f"Seed {seed}",
        )
    overall_axis.fill_between(
        UPDATES,
        stack.min(axis=0),
        stack.max(axis=0),
        color="#6C757D",
        alpha=0.16,
        label="Min–max across 3 seeds",
    )
    overall_axis.plot(
        UPDATES,
        stack.mean(axis=0),
        color="#111111",
        marker="o",
        linewidth=2.7,
        label="3-seed mean",
    )
    overall_axis.set_title("All network profiles")
    overall_axis.set_xlabel("Checkpoint update")
    overall_axis.set_ylabel("Frozen fit-validation expected reward")
    overall_axis.legend(frameon=False)

    for profile in PROFILES:
        series = _seed_series(inputs, "reward_mean", profile)
        values = np.vstack([series[seed] for seed in SEEDS])
        profile_axis.plot(
            UPDATES,
            values.mean(axis=0),
            color=PROFILE_COLORS[profile],
            marker="o",
            markersize=4,
            linewidth=2.0,
            label=PROFILE_LABELS[profile],
        )
    profile_axis.set_title("Cross-seed mean by network profile")
    profile_axis.set_xlabel("Checkpoint update")
    profile_axis.set_ylabel("Frozen fit-validation expected reward")
    profile_axis.legend(frameon=False)
    fig.suptitle("Deterministic policy on the frozen fit-validation panel")
    for axis in (overall_axis, profile_axis):
        axis.set_xticks(UPDATES[::2])
        _bold_ticks(axis)
    return _save(fig, output_dir / "02_fit_validation_expected_reward")


def plot_validation_decomposition(inputs: Inputs, output_dir: Path) -> Tuple[Path, Path]:
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.2), sharex=True)

    specifications = (
        (axes[0, 0], "q_perc_mean", "Expected perception quality, $Q_{perc}$", 1.0),
        (axes[0, 1], "latency_proxy_ms_mean", "Modeled P50 latency proxy (ms)", 1.0),
        (
            axes[1, 0],
            "p_edge_admission_given_sent_mean",
            "Modeled edge-admission probability (%)",
            100.0,
        ),
    )
    for axis, field, label, scale in specifications:
        series = _seed_series(inputs, field, "ALL_PROFILES")
        values = np.vstack([series[seed] for seed in SEEDS]) * scale
        axis.fill_between(
            UPDATES,
            values.min(axis=0),
            values.max(axis=0),
            color="#0072B2",
            alpha=0.17,
            label="Min–max across 3 seeds",
        )
        axis.plot(
            UPDATES,
            values.mean(axis=0),
            color="#0072B2",
            marker="o",
            linewidth=2.3,
            label="3-seed mean",
        )
        axis.set_ylabel(label)
        axis.set_xticks(UPDATES[::2])
        axis.legend(frameon=False)
        _bold_ticks(axis)

    deadline_axis = axes[1, 1]
    for field, label, color, linestyle in (
        ("budget_miss_rate_p50", "Conditional P50", "#0072B2", "-"),
        ("budget_miss_rate_p95", "Conditional P95", "#D55E00", "--"),
        ("budget_miss_rate_p99", "Conditional P99", "#000000", ":"),
    ):
        series = _seed_series(inputs, field, "ALL_PROFILES")
        values = np.vstack([series[seed] for seed in SEEDS]) * 100.0
        deadline_axis.plot(
            UPDATES,
            values.mean(axis=0),
            color=color,
            linestyle=linestyle,
            marker="o",
            linewidth=2.3,
            label=label,
        )
    deadline_axis.set_ylabel("Validation contexts above 200 ms (%)")
    deadline_axis.set_xticks(UPDATES[::2])
    deadline_axis.set_ylim(-3.0, 103.0)
    deadline_axis.legend(frameon=False)
    _bold_ticks(deadline_axis)

    for axis in axes[-1, :]:
        axis.set_xlabel("Checkpoint update")
    fig.suptitle("Frozen fit-validation decomposition — modeled expected quantities")
    return _save(fig, output_dir / "03_fit_validation_decomposition")


def _final_rows(inputs: Inputs, seed: int, profile: str) -> List[Mapping[str, str]]:
    rows = [
        row
        for row in inputs.validation_context
        if _int(row, "update_index") == 5000
        and _int(row, "seed") == seed
        and row["network_profile"] == profile
    ]
    if len(rows) != 85:
        raise ResultsError(f"final {(seed, profile)} has {len(rows)} rows, expected 85")
    return rows


def plot_final_action_behavior(inputs: Inputs, output_dir: Path) -> Tuple[Path, Path]:
    fig = plt.figure(figsize=(16.0, 8.8))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.65, 1.0), wspace=0.25)
    heat_axis = fig.add_subplot(grid[0, 0])
    q_axis = fig.add_subplot(grid[0, 1])

    heat_rows: List[List[float]] = []
    labels: List[str] = []
    for profile in PROFILES:
        for seed in SEEDS:
            rows = _final_rows(inputs, seed, profile)
            counts = Counter(_int(row, "executed_mode_id") for row in rows)
            heat_rows.append([100.0 * counts[mode] / len(rows) for mode in range(12)])
            labels.append(f"{PROFILE_LABELS[profile]} · s{seed}")
    heat = np.asarray(heat_rows)
    image = heat_axis.imshow(heat, aspect="auto", cmap="Blues", vmin=0.0, vmax=100.0)
    heat_axis.grid(False)
    heat_axis.set_xticks(
        range(12), MODE_LABELS, fontsize=7.5, rotation=45, ha="right"
    )
    heat_axis.set_yticks(range(len(labels)), labels)
    heat_axis.set_xlabel("Executed discrete mode")
    heat_axis.set_ylabel("Network profile and seed")
    heat_axis.set_title("Mode-selection frequency (%)")
    for row_index in range(heat.shape[0]):
        for mode in range(heat.shape[1]):
            value = heat[row_index, mode]
            if value >= 4.0:
                heat_axis.text(
                    mode,
                    row_index,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    fontweight="bold",
                    color="white" if value >= 52 else "#111111",
                )
    colorbar = fig.colorbar(image, ax=heat_axis, fraction=0.035, pad=0.02)
    colorbar.set_label("Decisions (%)", fontweight="bold")

    offsets = {17: -0.22, 29: 0.0, 43: 0.22}
    for profile_index, profile in enumerate(PROFILES):
        for seed in SEEDS:
            rows = _final_rows(inputs, seed, profile)
            values = [_int(row, "executed_q_e4") / 10000.0 for row in rows]
            q25 = quantile(values, 0.25)
            median = quantile(values, 0.5)
            q75 = quantile(values, 0.75)
            x = profile_index + offsets[seed]
            q_axis.vlines(x, q25, q75, color=SEED_COLORS[seed], linewidth=4, alpha=0.72)
            q_axis.scatter(
                [x],
                [median],
                s=58,
                color=SEED_COLORS[seed],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
                label=f"Seed {seed}" if profile_index == 0 else None,
            )
    q_axis.set_xticks(range(len(PROFILES)), [PROFILE_LABELS[p].replace(" ", "\n") for p in PROFILES])
    q_axis.set_ylabel("Executed continuous drop fraction, q")
    q_axis.set_xlabel("Network profile")
    q_axis.set_ylim(-0.02, 1.0)
    q_axis.set_title("Executed q: median and interquartile range")
    q_axis.legend(frameon=False, ncol=3, loc="upper center")
    fig.suptitle("Final checkpoint (update 5000): hybrid action behavior by seed")
    _bold_ticks(heat_axis)
    _bold_ticks(q_axis)
    return _save(fig, output_dir / "04_final_hybrid_action_behavior")


def _cross_seed_checkpoint_rows(inputs: Inputs) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    fields = (
        "reward_mean",
        "q_perc_mean",
        "payload_bytes_mean",
        "p_edge_admission_given_sent_mean",
        "latency_proxy_ms_mean",
        "budget_miss_rate_p50",
        "budget_miss_rate_p95",
        "budget_miss_rate_p99",
    )
    for update in UPDATES:
        for profile in ("ALL_PROFILES",) + PROFILES:
            source = [_aggregate_row(inputs, seed, update, profile) for seed in SEEDS]
            row: Dict[str, object] = {
                "update_index": update,
                "network_profile": profile,
                "seed_count": len(SEEDS),
            }
            for field in fields:
                values = [_float(item, field) for item in source]
                row[f"{field}_cross_seed_mean"] = float(np.mean(values))
                row[f"{field}_seed_min"] = min(values)
                row[f"{field}_seed_max"] = max(values)
            output.append(row)
    return output


def _final_action_rows(inputs: Inputs) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for profile in PROFILES:
        for seed in SEEDS:
            rows = _final_rows(inputs, seed, profile)
            modes = Counter(_int(row, "executed_mode_id") for row in rows)
            values = [_int(row, "executed_q_e4") / 10000.0 for row in rows]
            record: Dict[str, object] = {
                "network_profile": profile,
                "seed": seed,
                "decision_count": len(rows),
                "q_p25": quantile(values, 0.25),
                "q_median": quantile(values, 0.5),
                "q_p75": quantile(values, 0.75),
                "q_min": min(values),
                "q_max": max(values),
            }
            for mode in range(12):
                record[f"mode_{mode}_fraction"] = modes[mode] / len(rows)
            output.append(record)
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ResultsError(f"refusing to write empty CSV {path}")
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _training_summary(inputs: Inputs) -> Dict[str, object]:
    per_seed: Dict[str, object] = {}
    for seed in SEEDS:
        values = [_float(row, "reward_mean") for row in inputs.training[seed]]
        smooth = rolling_mean(values)
        per_seed[str(seed)] = {
            "first_100_update_reward_mean": float(np.mean(values[:100])),
            "last_100_update_reward_mean": float(np.mean(values[-100:])),
            "last_100_update_rolling_mean": float(smooth[-1]),
            "absolute_change_first_100_to_last_100": float(
                np.mean(values[-100:]) - np.mean(values[:100])
            ),
        }
    return {
        "metric_identity": "REPLAY_BATCH_REWARD_MEAN_NOT_ONLINE_REWARD",
        "rolling_window_updates": ROLLING_WINDOW,
        "per_seed": per_seed,
    }


def _validation_summary(inputs: Inputs) -> Dict[str, object]:
    means: Dict[int, float] = {}
    seed_values: Dict[int, List[float]] = {}
    for update in UPDATES:
        values = [
            _float(_aggregate_row(inputs, seed, update, "ALL_PROFILES"), "reward_mean")
            for seed in SEEDS
        ]
        seed_values[update] = values
        means[update] = float(np.mean(values))
    best_update = max(UPDATES, key=lambda update: means[update])
    plateau_updates = (3000, 3500, 4000, 4500, 5000)
    x = np.asarray(plateau_updates, dtype=np.float64)
    y = np.asarray([means[update] for update in plateau_updates], dtype=np.float64)
    slope_per_update = float(np.polyfit(x, y, 1)[0])

    decomposition: Dict[str, object] = {}
    for field in (
        "q_perc_mean",
        "latency_proxy_ms_mean",
        "p_edge_admission_given_sent_mean",
        "budget_miss_rate_p50",
        "budget_miss_rate_p95",
        "budget_miss_rate_p99",
    ):
        start = float(
            np.mean(
                [
                    _float(_aggregate_row(inputs, seed, 0, "ALL_PROFILES"), field)
                    for seed in SEEDS
                ]
            )
        )
        final = float(
            np.mean(
                [
                    _float(_aggregate_row(inputs, seed, 5000, "ALL_PROFILES"), field)
                    for seed in SEEDS
                ]
            )
        )
        decomposition[field] = {"update_0": start, "update_5000": final, "change": final - start}

    per_profile_final: Dict[str, object] = {}
    for profile in PROFILES:
        values = [
            _float(_aggregate_row(inputs, seed, 5000, profile), "reward_mean")
            for seed in SEEDS
        ]
        per_profile_final[profile] = {
            "cross_seed_reward_mean": float(np.mean(values)),
            "seed_min": min(values),
            "seed_max": max(values),
        }
    return {
        "metric_identity": "MODELED_EXPECTED_UTILITY_ON_FROZEN_FIT_VALIDATION_PANEL",
        "panel_contexts_per_seed_checkpoint": 340,
        "update_0_cross_seed_mean": means[0],
        "update_5000_cross_seed_mean": means[5000],
        "change_update_0_to_5000": means[5000] - means[0],
        "best_observed_cross_seed_mean_update": best_update,
        "best_observed_cross_seed_mean": means[best_update],
        "final_update_seed_min": min(seed_values[5000]),
        "final_update_seed_max": max(seed_values[5000]),
        "last_five_checkpoint_mean_range": float(y.max() - y.min()),
        "last_five_checkpoint_linear_slope_per_1000_updates": slope_per_update * 1000.0,
        "plateau_interpretation": (
            "The cross-seed mean varies within a narrow band over updates 3000–5000, "
            "but this finite fit-validation run does not establish final convergence or "
            "authorize checkpoint selection."
        ),
        "decomposition": decomposition,
        "per_profile_final": per_profile_final,
    }


def write_report(output_dir: Path, summary: Mapping[str, object], artifact_hashes: Mapping[str, str]) -> Path:
    validation = summary["validation"]
    assert isinstance(validation, Mapping)
    lines = [
        "# Preliminary Hybrid-SAC training and fit-validation results",
        "",
        "## Scope",
        "",
        "This pack separates two evidence populations:",
        "",
        "- Figure 01 is a **replay-batch training diagnostic**. It is not online reward, episodic return, cumulative reward, or a deployment measurement.",
        "- Figures 02–04 use the deterministic policy on the frozen 340-context-per-checkpoint **fit-validation panel**. Their reward, latency, delivery and deadline quantities are modeled expectations derived from the registered empirical surrogates.",
        "",
        "The fit-validation scenes were excluded from the training partition, but the fit procedure and normalization used the wider fit evidence. This is therefore useful for optimization diagnosis, not an untouched final generalization test.",
        "",
        "Replay-batch reward and frozen-panel reward are different populations and their numerical levels must not be compared as if they were the same measurement.",
        "",
        "## Quantitative reading",
        "",
        f"- Cross-seed fit-validation expected reward changes from `{validation['update_0_cross_seed_mean']:.6f}` at update 0 to `{validation['update_5000_cross_seed_mean']:.6f}` at update 5000.",
        f"- The highest observed cross-seed mean is `{validation['best_observed_cross_seed_mean']:.6f}` at update `{validation['best_observed_cross_seed_mean_update']}`. This is reported descriptively; no checkpoint is selected.",
        f"- Over updates 3000–5000, the cross-seed mean spans `{validation['last_five_checkpoint_mean_range']:.6f}` and its fitted slope is `{validation['last_five_checkpoint_linear_slope_per_1000_updates']:.6f}` reward per 1000 updates.",
        f"- At update 5000, the three seed means span `{validation['final_update_seed_min']:.6f}` to `{validation['final_update_seed_max']:.6f}`.",
        f"- The modeled 200-ms context-miss rate at update 5000 is `{100.0 * validation['decomposition']['budget_miss_rate_p50']['update_5000']:.2f}%` for conditional P50, `{100.0 * validation['decomposition']['budget_miss_rate_p95']['update_5000']:.2f}%` for conditional P95, and `{100.0 * validation['decomposition']['budget_miss_rate_p99']['update_5000']:.2f}%` for conditional P99.",
        "- This is plateau evidence, not proof of final convergence. Training should not be extended or the reward changed solely from replay-batch reward.",
        "- The current registered reward uses the conditional P50 latency proxy. It can improve median-path utility without enforcing P95/P99 reliability.",
        "",
        "## Artifacts",
        "",
        "- `01_training_replay_batch_reward.{png,pdf}`",
        "- `02_fit_validation_expected_reward.{png,pdf}`",
        "- `03_fit_validation_decomposition.{png,pdf}`",
        "- `04_final_hybrid_action_behavior.{png,pdf}`",
        "- `checkpoint_summary.csv`",
        "- `final_action_summary.csv`",
        "- `summary.json`",
        "",
        "Mode labels M0–M11 are stable discrete mode IDs from the frozen action contract: M0–M2 are noAE U8/U6/U4, M3–M5 are AE128 U8/U6/U4, M6–M8 are AE64 U8/U6/U4, and M9–M11 are AE32 U8/U6/U4. The q panel reports the executed continuous drop fraction, with each seed displayed separately.",
        "",
        "## Limitations",
        "",
        "- No CARLA, radio, CUDA, network, or online-policy execution was performed for this pack.",
        "- The training schedule has no episode boundary: each gradient update samples terminal contextual transitions. Therefore an episode-return plot would be false for this experiment.",
        "- Modeled admission probability and latency proxy are not live measurements.",
        "- A deadline curve is the fraction of frozen validation contexts whose modeled conditional P50/P95/P99 latency exceeds 200 ms; it is not an empirical per-frame timeout or packet-loss probability.",
        "- Three seeds characterize preliminary variability but do not establish statistical generalization.",
        "- SHAP/state-attribution analysis is deliberately deferred until the policy/reward choice is stable.",
        "",
        "## Output hashes",
        "",
    ]
    for name, digest in sorted(artifact_hashes.items()):
        lines.append(f"- `{name}`: `{digest}`")
    path = output_dir / "REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def render(inputs: Inputs, output_dir: Path) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _style()
    figure_paths: List[Path] = []
    for renderer in (
        plot_training_reward,
        plot_validation_reward,
        plot_validation_decomposition,
        plot_final_action_behavior,
    ):
        figure_paths.extend(renderer(inputs, output_dir))

    checkpoint_csv = output_dir / "checkpoint_summary.csv"
    action_csv = output_dir / "final_action_summary.csv"
    _write_csv(checkpoint_csv, _cross_seed_checkpoint_rows(inputs))
    _write_csv(action_csv, _final_action_rows(inputs))

    summary: Dict[str, object] = {
        "schema": RESULT_SCHEMA,
        "training": _training_summary(inputs),
        "validation": _validation_summary(inputs),
        "source_sha256": dict(inputs.source_hashes),
        "claims": {
            "learning_interpretation": (
                "PRELIMINARY_REWARD_HELD_LEARNING_SIGNAL_AND_PLATEAU_EVIDENCE"
            ),
            "final_convergence_claimed": False,
            "checkpoint_selected": False,
            "online_reward_claimed": False,
            "generalization_claimed": False,
        },
    }
    pre_summary_artifacts = figure_paths + [checkpoint_csv, action_csv]
    artifact_hashes = {path.name: sha256_file(path) for path in pre_summary_artifacts}
    summary["artifact_sha256"] = artifact_hashes
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    artifact_hashes[summary_path.name] = sha256_file(summary_path)
    report_path = write_report(output_dir, summary, artifact_hashes)
    return {
        "output_dir": str(output_dir),
        "summary": summary,
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    inputs = load_inputs(args.baseline_root, args.validation_root)
    result = render(inputs, args.output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
