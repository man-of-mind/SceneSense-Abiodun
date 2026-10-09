#!/usr/bin/env python3
"""Render a presentation pack from the completed exact-P95 Run-2 evaluation.

This module is deliberately read-only with respect to the training and
evaluation evidence.  It verifies the two source tables against their pinned
SHA-256 digests, checks the registered table shapes, and only then renders
figures.  It does not load a checkpoint, execute a policy, initialize CUDA, or
reinterpret the registered reward.

The pack compares policies only under the common V2 scoring rule.  In
particular, the Run-1 point is the evaluator's V2 rescore, never Run-1's native
reward.  The P95 quantity is the registered modeled conditional
retained-survivor latency proxy; it is not a live end-to-end SLA.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/splitfusion_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .presentation_quality_components import quality_components_for_rows


SCHEMA = "splitfusion.hybrid_sac.run2_presentation.v1"
SOURCE_AGGREGATE_SHA256 = (
    "82588c5d88b370a92fe954733dc82a03decbac19c700c34c4ba669b4931ed13c"
)
SOURCE_CONTEXT_SHA256 = (
    "6b32edf48ffb5616c9644ac84e7193b6e6b2c75fc424bc15910eb5e50d982948"
)
SOURCE_TIERS_SHA256 = (
    "27eca031810038950f86790c2c188f22e48163c1e091ceb2430553c5cc3c9651"
)
SOURCE_ADAPTATION_SHA256 = (
    "1000bfc5649e271e6447220f44d4c333bc7e1567bd039626a2a74aa13f80e05c"
)

RUN2 = "RUN2_V2_ACTOR"
RUN1 = "RUN1_UPDATE5000_RESCORED_V2"
RANDOM = "REGISTERED_IDENTITY_RANDOM"
FIXED = "FROZEN_TRAIN_SELECTED_FIXED"
ORACLE = "CONTEXTUAL_P95_CONSTRAINED_ORACLE"

SEEDS = (17, 29, 43)
UPDATES = tuple(range(0, 5001, 500))
PROFILES = (
    "FAVORABLE_STABLE",
    "MID_VARIABLE",
    "ADVERSE_STABLE",
    "FADE_RECOVERY",
)
PROFILE_LABELS = {
    "FAVORABLE_STABLE": "Favorable\nstable",
    "MID_VARIABLE": "Mid\nvariable",
    "ADVERSE_STABLE": "Adverse\nstable",
    "FADE_RECOVERY": "Fade\nrecovery",
}
PROFILE_COLORS = {
    "FAVORABLE_STABLE": "#0072B2",
    "MID_VARIABLE": "#E69F00",
    "ADVERSE_STABLE": "#D55E00",
    "FADE_RECOVERY": "#009E73",
}
SEED_COLORS = {17: "#0072B2", 29: "#D55E00", 43: "#009E73"}
ROLLING_WINDOW = 100
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN2_TRAINING_ROOT = (
    PROJECT_ROOT
    / "experiments/splitfusion_hybrid_sac_run2_training_v2/20260921_exact_p95_run2_v2"
)

POLICY_ORDER = (RUN1, RUN2, FIXED, ORACLE)
POLICY_LABELS = {
    RUN1: "Run 1\n(V2 rescore)",
    RUN2: "Run 2\n(update 5000)",
    FIXED: "Fixed\nbaseline",
    ORACLE: "Contextual\noracle",
}
POLICY_SHORT = {RUN1: "Run 1", RUN2: "Run 2", FIXED: "Fixed", ORACLE: "Oracle"}
POLICY_COLORS = {
    RUN1: "#CC79A7",
    RUN2: "#0072B2",
    FIXED: "#E69F00",
    ORACLE: "#009E73",
}
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


class PresentationError(RuntimeError):
    """The source evidence or requested presentation is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> Tuple[Mapping[str, str], ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _finite(row: Mapping[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise PresentationError(f"non-finite {field}: {row[field]!r}")
    return value


def _integer(row: Mapping[str, str], field: str) -> int:
    return int(row[field])


def _rolling_mean(values: Sequence[float], window: int = ROLLING_WINDOW) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < window or window <= 0:
        raise PresentationError("invalid rolling-mean input")
    prefix = np.concatenate(([0.0], np.cumsum(array, dtype=np.float64)))
    return (prefix[window:] - prefix[:-window]) / float(window)


@dataclass(frozen=True)
class Inputs:
    source_root: Path
    aggregate: Tuple[Mapping[str, str], ...]
    contexts: Tuple[Mapping[str, str], ...]
    training: Mapping[int, Tuple[Mapping[str, str], ...]]
    tiers: Mapping[str, object]
    adaptation: Mapping[str, object]
    source_hashes: Mapping[str, str]


def load_inputs(source_root: Path) -> Inputs:
    paths = {
        "aggregate.csv": source_root / "aggregate.csv",
        "per_context.csv": source_root / "per_context.csv",
        "success_tiers.json": source_root / "success_tiers.json",
        "adaptation.json": source_root / "adaptation.json",
    }
    expected = {
        "aggregate.csv": SOURCE_AGGREGATE_SHA256,
        "per_context.csv": SOURCE_CONTEXT_SHA256,
        "success_tiers.json": SOURCE_TIERS_SHA256,
        "adaptation.json": SOURCE_ADAPTATION_SHA256,
    }
    hashes: Dict[str, str] = {}
    for name, path in paths.items():
        actual = sha256_file(path)
        if actual != expected[name]:
            raise PresentationError(
                f"source hash mismatch for {name}: expected {expected[name]}, got {actual}"
            )
        hashes[name] = actual

    training: Dict[int, Tuple[Mapping[str, str], ...]] = {}
    for seed in SEEDS:
        path = RUN2_TRAINING_ROOT / f"seed_{seed}" / "metrics.csv"
        rows = _read_csv(path)
        if len(rows) != 5000:
            raise PresentationError(
                f"Run-2 seed {seed} has {len(rows)} training rows, expected 5000"
            )
        if [_integer(row, "update_index") for row in rows] != list(range(1, 5001)):
            raise PresentationError(f"Run-2 seed {seed} update indexes differ")
        for row in rows:
            _finite(row, "reward_mean")
        training[seed] = rows
        hashes[f"training_metrics_seed_{seed}.csv"] = sha256_file(path)

    aggregate = _read_csv(paths["aggregate.csv"])
    contexts = _read_csv(paths["per_context.csv"])
    if len(aggregate) != 270 or len(contexts) != 13940:
        raise PresentationError(
            f"source shape mismatch: aggregate={len(aggregate)}, contexts={len(contexts)}"
        )

    aggregate_keys = {
        (
            row["source_kind"],
            _integer(row, "seed"),
            _integer(row, "update_index"),
            row["network_profile"],
        )
        for row in aggregate
    }
    if len(aggregate_keys) != len(aggregate):
        raise PresentationError("aggregate keys are duplicated")

    context_keys = {
        (
            row["source_kind"],
            _integer(row, "seed"),
            _integer(row, "update_index"),
            _integer(row, "panel_index"),
        )
        for row in contexts
    }
    if len(context_keys) != len(contexts):
        raise PresentationError("per-context keys are duplicated")

    required_run2 = {
        (seed, update, profile)
        for seed in (-1,) + SEEDS
        for update in UPDATES
        for profile in ("ALL_PROFILES",) + PROFILES
    }
    observed_run2 = {
        (_integer(row, "seed"), _integer(row, "update_index"), row["network_profile"])
        for row in aggregate
        if row["source_kind"] == RUN2
    }
    if observed_run2 != required_run2:
        raise PresentationError("Run-2 aggregate grid is incomplete or foreign")

    for source, update, expected_seeds in (
        (RUN1, 5000, {-1, *SEEDS}),
        (RANDOM, -1, {-1, *SEEDS}),
        (FIXED, -1, {-1}),
        (ORACLE, -1, {-1}),
    ):
        observed = {
            (_integer(row, "seed"), row["network_profile"])
            for row in aggregate
            if row["source_kind"] == source and _integer(row, "update_index") == update
        }
        required = {
            (seed, profile)
            for seed in expected_seeds
            for profile in ("ALL_PROFILES",) + PROFILES
        }
        if observed != required:
            raise PresentationError(f"aggregate grid differs for {source}")

    for row in aggregate:
        for field in (
            "v2_utility_mean",
            "q_perc_mean",
            "p_edge_admission_given_sent_mean",
            "latency_p50_ms_mean",
            "latency_p95_ms_mean",
            "latency_p99_ms_mean",
            "p95_margin_ms_mean",
            "p95_miss_rate",
        ):
            _finite(row, field)

    tiers = json.loads(paths["success_tiers.json"].read_text(encoding="utf-8"))
    adaptation = json.loads(paths["adaptation.json"].read_text(encoding="utf-8"))
    if tiers.get("tier0", {}).get("passed") is not True:
        raise PresentationError("source Tier 0 integrity gate did not pass")
    if tiers.get("tier1", {}).get("passed") is not True:
        raise PresentationError("source Tier 1 learning gate did not pass")
    return Inputs(source_root, aggregate, contexts, training, tiers, adaptation, hashes)


def aggregate_row(
    inputs: Inputs, source: str, seed: int, update: int, profile: str
) -> Mapping[str, str]:
    rows = [
        row
        for row in inputs.aggregate
        if row["source_kind"] == source
        and _integer(row, "seed") == seed
        and _integer(row, "update_index") == update
        and row["network_profile"] == profile
    ]
    if len(rows) != 1:
        raise PresentationError(
            f"expected one aggregate row for {(source, seed, update, profile)}, got {len(rows)}"
        )
    return rows[0]


def context_rows(
    inputs: Inputs, source: str, seed: int, update: int, profile: str
) -> Tuple[Mapping[str, str], ...]:
    rows = tuple(
        row
        for row in inputs.contexts
        if row["source_kind"] == source
        and _integer(row, "seed") == seed
        and _integer(row, "update_index") == update
        and row["network_profile"] == profile
    )
    expected = 85
    if len(rows) != expected:
        raise PresentationError(
            f"expected {expected} context rows for {(source, seed, update, profile)}, got {len(rows)}"
        )
    return rows


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


def _save(fig: plt.Figure, stem: Path) -> Tuple[Path, Path]:
    layout_rect = getattr(fig, "_splitfusion_tight_layout_rect", None)
    fig.tight_layout(rect=layout_rect)
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight", metadata={"Software": SCHEMA})
    fig.savefig(
        pdf,
        bbox_inches="tight",
        metadata={"Title": stem.name, "Creator": SCHEMA, "CreationDate": None},
    )
    plt.close(fig)
    return png, pdf


def _pooled_run2_series(inputs: Inputs, field: str) -> np.ndarray:
    return np.asarray(
        [
            _finite(aggregate_row(inputs, RUN2, -1, update, "ALL_PROFILES"), field)
            for update in UPDATES
        ],
        dtype=np.float64,
    )


def plot_utility_learning(inputs: Inputs, output_dir: Path) -> Tuple[Path, Path]:
    fig, axis = plt.subplots(figsize=(11.8, 6.6))
    for seed in SEEDS:
        values = [
            _finite(aggregate_row(inputs, RUN2, seed, update, "ALL_PROFILES"), "v2_utility_mean")
            for update in UPDATES
        ]
        axis.plot(
            UPDATES,
            values,
            color=SEED_COLORS[seed],
            alpha=0.42,
            linewidth=1.3,
            marker="o",
            markersize=3.5,
            label=f"Run 2 seed {seed}",
        )
    pooled = _pooled_run2_series(inputs, "v2_utility_mean")
    axis.plot(
        UPDATES,
        pooled,
        color="#111111",
        linewidth=3.0,
        marker="o",
        markersize=5.0,
        label="Run 2 pooled",
        zorder=5,
    )
    references = (
        (RANDOM, "Random", "#777777", ":"),
        (RUN1, "Run 1 (V2 rescore)", POLICY_COLORS[RUN1], "--"),
        (FIXED, "Fixed baseline", POLICY_COLORS[FIXED], "-."),
        (ORACLE, "Contextual oracle", POLICY_COLORS[ORACLE], (0, (5, 2))),
    )
    for source, label, color, linestyle in references:
        update = 5000 if source == RUN1 else -1
        value = _finite(
            aggregate_row(inputs, source, -1, update, "ALL_PROFILES"), "v2_utility_mean"
        )
        axis.axhline(value, color=color, linewidth=1.8, linestyle=linestyle, label=label)
    axis.set_title("Run 2 learns the deadline constraint, then plateaus")
    axis.set_xlabel("SAC gradient updates")
    axis.set_ylabel("V2 validation utility")
    axis.set_xticks(UPDATES[::2])
    axis.set_xlim(0, 5000)
    axis.axhline(0.0, color="#444444", linewidth=0.8)
    axis.legend(frameon=False, ncol=2, loc="lower right")
    _bold_ticks(axis)
    return _save(fig, output_dir / "01_run2_learning_utility")


def plot_deadline_learning(inputs: Inputs, output_dir: Path) -> Tuple[Path, Path]:
    fig, axis = plt.subplots(figsize=(11.8, 6.5))
    for seed in SEEDS:
        values = [
            100.0
            * _finite(aggregate_row(inputs, RUN2, seed, update, "ALL_PROFILES"), "p95_miss_rate")
            for update in UPDATES
        ]
        axis.plot(
            UPDATES,
            values,
            color=SEED_COLORS[seed],
            alpha=0.45,
            linewidth=1.3,
            marker="o",
            markersize=3.5,
            label=f"Seed {seed}",
        )
    pooled = 100.0 * _pooled_run2_series(inputs, "p95_miss_rate")
    axis.plot(
        UPDATES,
        pooled,
        color="#111111",
        linewidth=3.0,
        marker="o",
        markersize=5.0,
        label="Pooled",
        zorder=5,
    )
    run1 = 100.0 * _finite(
        aggregate_row(inputs, RUN1, -1, 5000, "ALL_PROFILES"), "p95_miss_rate"
    )
    axis.axhline(
        run1,
        color=POLICY_COLORS[RUN1],
        linestyle="--",
        linewidth=1.8,
        label="Run 1 (V2 rescore)",
    )
    axis.axhline(5.0, color="#009E73", linestyle=":", linewidth=1.6, label="5% guide")
    axis.set_title("Run 2 sharply reduces modeled 200-ms P95 deadline misses")
    axis.set_xlabel("SAC gradient updates")
    axis.set_ylabel("Validation contexts above 200 ms (%)")
    axis.set_xticks(UPDATES[::2])
    axis.set_xlim(0, 5000)
    axis.set_ylim(-2, 82)
    axis.legend(frameon=False, ncol=2)
    _bold_ticks(axis)
    return _save(fig, output_dir / "02_run2_deadline_learning")


def _policy_row(inputs: Inputs, source: str, profile: str = "ALL_PROFILES") -> Mapping[str, str]:
    update = 5000 if source in (RUN1, RUN2) else -1
    return aggregate_row(inputs, source, -1, update, profile)


def _bar_labels(axis: plt.Axes, bars: Iterable[object], fmt: str) -> None:
    for bar in bars:
        height = float(bar.get_height())
        axis.annotate(
            fmt.format(height),
            (bar.get_x() + bar.get_width() / 2.0, height),
            xytext=(0, 4 if height >= 0 else -14),
            textcoords="offset points",
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=8.5,
            fontweight="bold",
        )


def plot_final_comparator_summary(inputs: Inputs, output_dir: Path) -> Tuple[Path, Path]:
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.2))
    entries = (
        (RUN2, 0, "Run 2\ninitial", "#999999"),
        (RUN1, 5000, POLICY_LABELS[RUN1], POLICY_COLORS[RUN1]),
        (RUN2, 5000, POLICY_LABELS[RUN2], POLICY_COLORS[RUN2]),
        (FIXED, -1, POLICY_LABELS[FIXED], POLICY_COLORS[FIXED]),
        (ORACLE, -1, POLICY_LABELS[ORACLE], POLICY_COLORS[ORACLE]),
    )
    x = np.arange(len(entries))
    colors = [entry[3] for entry in entries]
    specs = (
        (axes[0, 0], "v2_utility_mean", 1.0, "V2 utility", "{:.3f}"),
        (
            axes[0, 1],
            "q_perc_mean",
            1.0,
            "Composite perception quality",
            "{:.3f}",
        ),
        (
            axes[1, 0],
            "latency_p95_ms_mean",
            1.0,
            "Mean modeled P95 latency (ms)",
            "{:.1f}",
        ),
        (
            axes[1, 1],
            "p95_miss_rate",
            100.0,
            "Contexts above 200 ms (%)",
            "{:.1f}%",
        ),
    )
    for axis, field, scale, ylabel, fmt in specs:
        values = [
            scale * _finite(aggregate_row(inputs, source, -1, update, "ALL_PROFILES"), field)
            for source, update, _label, _color in entries
        ]
        bars = axis.bar(x, values, color=colors, edgecolor="#222222", linewidth=0.65, width=0.68)
        axis.set_xticks(x, [entry[2] for entry in entries])
        axis.set_ylabel(ylabel)
        _bar_labels(axis, bars, fmt)
        if field == "v2_utility_mean":
            axis.axhline(0.0, color="#333333", linewidth=0.8)
        if field == "latency_p95_ms_mean":
            axis.axhline(200.0, color="#D55E00", linestyle="--", linewidth=1.5, label="200-ms budget")
            axis.legend(frameon=False)
        _bold_ticks(axis)
    fig.suptitle("Final policy comparison under one common V2 scoring rule")
    return _save(fig, output_dir / "03_final_policy_comparison")


def plot_profile_miss_comparison(inputs: Inputs, output_dir: Path) -> Tuple[Path, Path]:
    fig, axis = plt.subplots(figsize=(13.2, 6.8))
    policies = (
        (RUN2, 0, "Run 2 update 0", "#999999"),
        (RUN1, 5000, "Run 1 (V2 rescore)", POLICY_COLORS[RUN1]),
        (RUN2, 5000, "Run 2 update 5000", POLICY_COLORS[RUN2]),
        (FIXED, -1, "Fixed baseline", POLICY_COLORS[FIXED]),
        (ORACLE, -1, "Contextual oracle", POLICY_COLORS[ORACLE]),
    )
    x = np.arange(len(PROFILES), dtype=np.float64)
    width = 0.16
    offsets = (np.arange(len(policies)) - (len(policies) - 1) / 2.0) * width
    for offset, (source, update, label, color) in zip(offsets, policies):
        values = [
            100.0
            * _finite(aggregate_row(inputs, source, -1, update, profile), "p95_miss_rate")
            for profile in PROFILES
        ]
        bars = axis.bar(
            x + offset,
            values,
            width=width,
            label=label,
            color=color,
            edgecolor="#222222",
            linewidth=0.5,
        )
        for bar, value in zip(bars, values):
            if value > 0:
                axis.annotate(
                    f"{value:.1f}",
                    (bar.get_x() + bar.get_width() / 2.0, value),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    fontweight="bold",
                    rotation=90 if value < 12 else 0,
                )
    axis.set_xticks(x, [PROFILE_LABELS[p] for p in PROFILES])
    axis.set_ylabel("Contexts above 200 ms (%)")
    axis.set_xlabel("Network profile")
    axis.set_title("Deadline learning by network profile")
    axis.legend(frameon=False, ncol=3, loc="upper center")
    _bold_ticks(axis)
    return _save(fig, output_dir / "04_p95_deadline_misses_by_profile")


def _mode_matrix(
    inputs: Inputs, source: str, seeds: Sequence[int], update: int
) -> np.ndarray:
    matrix = np.zeros((len(PROFILES), len(MODE_LABELS)), dtype=np.float64)
    for profile_index, profile in enumerate(PROFILES):
        rows: List[Mapping[str, str]] = []
        for seed in seeds:
            rows.extend(context_rows(inputs, source, seed, update, profile))
        counts = Counter(_integer(row, "executed_mode_id") for row in rows)
        matrix[profile_index, :] = [100.0 * counts[index] / len(rows) for index in range(12)]
    return matrix


def plot_mode_selection(inputs: Inputs, output_dir: Path) -> Tuple[Path, Path]:
    matrices = (
        _mode_matrix(inputs, RUN2, SEEDS, 5000),
        _mode_matrix(inputs, ORACLE, (-1,), -1),
    )
    fig, axes = plt.subplots(1, 2, figsize=(17.2, 5.4), sharey=True)
    image = None
    for axis, matrix, title in zip(
        axes, matrices, ("Run 2, update 5000", "Contextual oracle")
    ):
        image = axis.imshow(matrix, aspect="auto", cmap="Blues", vmin=0.0, vmax=100.0)
        axis.grid(False)
        axis.set_xticks(range(12), MODE_LABELS, rotation=50, ha="right", fontsize=7.8)
        axis.set_yticks(range(len(PROFILES)), [PROFILE_LABELS[p].replace("\n", " ") for p in PROFILES])
        axis.set_xlabel("Discrete mode")
        axis.set_title(title)
        for row_index in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row_index, column]
                if value >= 4.0:
                    axis.text(
                        column,
                        row_index,
                        f"{value:.0f}",
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        fontweight="bold",
                        color="white" if value >= 52.0 else "#111111",
                    )
        _bold_ticks(axis)
    axes[0].set_ylabel("Network profile")
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    colorbar.set_label("Decisions (%)", fontweight="bold")
    for label in colorbar.ax.get_yticklabels():
        label.set_fontweight("bold")
    fig.suptitle("Run 2 collapses to AE32/UINT4; the oracle uses multiple modes")
    fig.subplots_adjust(bottom=0.25, right=0.92, wspace=0.12)
    png = output_dir / "05_mode_selection_run2_vs_oracle.png"
    pdf = output_dir / "05_mode_selection_run2_vs_oracle.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", metadata={"Software": SCHEMA})
    fig.savefig(pdf, bbox_inches="tight", metadata={"Title": "mode selection", "Creator": SCHEMA})
    plt.close(fig)
    return png, pdf


def _q_values(
    inputs: Inputs, source: str, seeds: Sequence[int], update: int, profile: str
) -> List[float]:
    values: List[float] = []
    for seed in seeds:
        values.extend(
            _integer(row, "executed_q_e4") / 10000.0
            for row in context_rows(inputs, source, seed, update, profile)
        )
    return values


def plot_q_adaptation(inputs: Inputs, output_dir: Path) -> Tuple[Path, Path]:
    fig, axis = plt.subplots(figsize=(12.8, 6.8))
    policies = (
        (RUN2, SEEDS, 5000, "Run 2", POLICY_COLORS[RUN2]),
        (FIXED, (-1,), -1, "Fixed", POLICY_COLORS[FIXED]),
        (ORACLE, (-1,), -1, "Oracle", POLICY_COLORS[ORACLE]),
    )
    base = np.arange(len(PROFILES), dtype=np.float64)
    offsets = (-0.24, 0.0, 0.24)
    for offset, (source, seeds, update, label, color) in zip(offsets, policies):
        positions = base + offset
        values = [_q_values(inputs, source, seeds, update, profile) for profile in PROFILES]
        boxes = axis.boxplot(
            values,
            positions=positions,
            widths=0.19,
            patch_artist=True,
            showfliers=False,
            whis=(5, 95),
            manage_ticks=False,
        )
        for patch in boxes["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.72)
            patch.set_edgecolor("#222222")
        for median in boxes["medians"]:
            median.set_color("#111111")
            median.set_linewidth(1.8)
        for element in ("whiskers", "caps"):
            for artist in boxes[element]:
                artist.set_color(color)
                artist.set_linewidth(1.2)
        axis.plot([], [], color=color, linewidth=8, alpha=0.72, label=label)
    axis.set_xticks(base, [PROFILE_LABELS[p] for p in PROFILES])
    axis.set_ylabel("Executed spatial drop fraction, q")
    axis.set_xlabel("Network profile")
    axis.set_ylim(0.25, 1.0)
    axis.set_title("Run 2 adapts continuous compression to channel severity")
    axis.legend(frameon=False, ncol=3)
    _bold_ticks(axis)
    return _save(fig, output_dir / "06_continuous_q_by_profile")


def plot_profile_quality_latency(inputs: Inputs, output_dir: Path) -> Tuple[Path, Path]:
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.4), sharex=True, sharey=True)
    annotation_offsets = {
        RUN1: (5, 5),
        RUN2: (5, -13),
        FIXED: (5, 7),
        ORACLE: (5, 5),
    }
    for axis, profile in zip(axes.flat, PROFILES):
        for source in POLICY_ORDER:
            row = _policy_row(inputs, source, profile)
            x = _finite(row, "latency_p95_ms_mean")
            y = _finite(row, "q_perc_mean")
            axis.scatter(
                x,
                y,
                s=105,
                color=POLICY_COLORS[source],
                edgecolor="#222222",
                linewidth=0.7,
                label=POLICY_SHORT[source],
                zorder=3,
            )
            axis.annotate(
                POLICY_SHORT[source],
                (x, y),
                xytext=annotation_offsets[source],
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
            )
        axis.axvline(200.0, color="#D55E00", linestyle="--", linewidth=1.3)
        axis.set_title(PROFILE_LABELS[profile].replace("\n", " "))
        axis.set_xlim(165, 235)
        axis.set_ylim(0.46, 0.72)
        _bold_ticks(axis)
    for axis in axes[-1, :]:
        axis.set_xlabel("Mean modeled P95 latency (ms) — lower is better")
    for axis in axes[:, 0]:
        axis.set_ylabel("Composite perception quality — higher is better")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Perception quality versus P95 latency by network profile", y=1.01)
    return _save(fig, output_dir / "07_quality_vs_p95_by_profile")


def plot_p95_margin(inputs: Inputs, output_dir: Path) -> Tuple[Path, Path]:
    fig, axis = plt.subplots(figsize=(12.8, 6.8))
    policies = (
        (RUN2, SEEDS, 5000, "Run 2", POLICY_COLORS[RUN2]),
        (FIXED, (-1,), -1, "Fixed", POLICY_COLORS[FIXED]),
        (ORACLE, (-1,), -1, "Oracle", POLICY_COLORS[ORACLE]),
    )
    base = np.arange(len(PROFILES), dtype=np.float64)
    offsets = (-0.24, 0.0, 0.24)
    for offset, (source, seeds, update, label, color) in zip(offsets, policies):
        distributions: List[List[float]] = []
        for profile in PROFILES:
            values: List[float] = []
            for seed in seeds:
                values.extend(
                    _finite(row, "p95_margin_ms")
                    for row in context_rows(inputs, source, seed, update, profile)
                )
            distributions.append(values)
        boxes = axis.boxplot(
            distributions,
            positions=base + offset,
            widths=0.19,
            patch_artist=True,
            showfliers=False,
            whis=(5, 95),
            manage_ticks=False,
        )
        for patch in boxes["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.72)
            patch.set_edgecolor("#222222")
        for median in boxes["medians"]:
            median.set_color("#111111")
            median.set_linewidth(1.8)
        for element in ("whiskers", "caps"):
            for artist in boxes[element]:
                artist.set_color(color)
                artist.set_linewidth(1.2)
        axis.plot([], [], color=color, linewidth=8, alpha=0.72, label=label)
    axis.axhline(0.0, color="#D55E00", linestyle="--", linewidth=1.7, label="200-ms boundary")
    axis.set_xticks(base, [PROFILE_LABELS[p] for p in PROFILES])
    axis.set_ylabel("P95 budget margin: 200 ms − latency (ms)")
    axis.set_xlabel("Network profile")
    axis.set_title("Positive margin meets the modeled 200-ms P95 budget")
    axis.legend(frameon=False, ncol=4)
    _bold_ticks(axis)
    return _save(fig, output_dir / "08_p95_budget_margin_by_profile")


def plot_adaptation_lift(inputs: Inputs, output_dir: Path) -> Tuple[Path, Path]:
    adaptation = inputs.adaptation
    seed_lift = adaptation["seed_mean_lift"]
    labels = [f"Seed {seed}" for seed in SEEDS] + ["Pooled"]
    values = [float(seed_lift[str(seed)]) for seed in SEEDS] + [float(adaptation["pooled_mean_lift"])]
    colors = [SEED_COLORS[seed] for seed in SEEDS] + ["#111111"]
    fig, axis = plt.subplots(figsize=(10.5, 6.3))
    bars = axis.bar(labels, values, color=colors, edgecolor="#222222", linewidth=0.65, width=0.65)
    _bar_labels(axis, bars, "{:+.3f}")
    lower = float(adaptation["lower_95"])
    upper = float(adaptation["upper_95"])
    axis.errorbar(
        [3],
        [values[-1]],
        yerr=[[values[-1] - lower], [upper - values[-1]]],
        fmt="none",
        ecolor="#D55E00",
        elinewidth=2.0,
        capsize=6,
        label="95% scene-cluster bootstrap CI",
    )
    axis.axhline(0.0, color="#333333", linewidth=1.0)
    axis.set_ylabel("Run 2 utility − fixed-baseline utility")
    axis.set_title("Run 2 does not yet outperform the fixed baseline")
    axis.legend(frameon=False)
    _bold_ticks(axis)
    return _save(fig, output_dir / "09_run2_minus_fixed_utility")


def plot_training_replay_reward(inputs: Inputs, output_dir: Path) -> Tuple[Path, Path]:
    """Render the raw and trailing-mean Run-2 replay-minibatch rewards."""

    fig, axis = plt.subplots(figsize=(11.5, 6.4))
    for seed in SEEDS:
        rows = inputs.training[seed]
        x = np.asarray([_integer(row, "update_index") for row in rows])
        y = np.asarray([_finite(row, "reward_mean") for row in rows])
        smooth = _rolling_mean(y)
        color = SEED_COLORS[seed]
        axis.plot(x, y, color=color, alpha=0.11, linewidth=0.65)
        axis.plot(
            x[ROLLING_WINDOW - 1 :],
            smooth,
            color=color,
            linewidth=2.1,
            label=f"Seed {seed}: {ROLLING_WINDOW}-update rolling mean",
        )
    axis.set_title("Run 2 training diagnostic: replay-batch reward")
    axis.set_xlabel("SAC gradient updates")
    axis.set_ylabel("Replay-batch reward mean")
    axis.set_xlim(1, 5000)
    axis.legend(frameon=False)
    _bold_ticks(axis)
    return _save(fig, output_dir / "10_run2_training_replay_batch_reward")


def plot_validation_latency_trajectory(
    inputs: Inputs, output_dir: Path
) -> Tuple[Path, Path]:
    """Show Run-2 P50/P95/P99 checkpoint trajectories by profile."""

    fields = (
        ("latency_p50_ms_mean", "P50", "#0072B2", "-"),
        ("latency_p95_ms_mean", "P95", "#D55E00", "--"),
        ("latency_p99_ms_mean", "P99", "#111111", ":"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.2), sharex=True)
    for axis, profile in zip(axes.flat, PROFILES):
        for field, label, color, linestyle in fields:
            means = []
            lows = []
            highs = []
            for update in UPDATES:
                values = [
                    _finite(aggregate_row(inputs, RUN2, seed, update, profile), field)
                    for seed in SEEDS
                ]
                means.append(float(np.mean(values)))
                lows.append(min(values))
                highs.append(max(values))
            axis.fill_between(UPDATES, lows, highs, color=color, alpha=0.08)
            axis.plot(
                UPDATES,
                means,
                color=color,
                linestyle=linestyle,
                marker="o",
                markersize=3.6,
                linewidth=2.1,
                label=label,
            )
        axis.axhline(
            200.0,
            color="#CC0000",
            linestyle="-.",
            linewidth=1.5,
            label="200-ms budget",
        )
        axis.set_title(PROFILE_LABELS[profile].replace("\n", " "))
        axis.set_xticks(UPDATES[::2])
        _bold_ticks(axis)
    for axis in axes[-1, :]:
        axis.set_xlabel("SAC gradient updates")
    for axis in axes[:, 0]:
        axis.set_ylabel("Modeled end-to-feedback latency (ms)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center right",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )
    fig.suptitle("Run 2: validation latency across training checkpoints")
    setattr(fig, "_splitfusion_tight_layout_rect", (0.0, 0.0, 0.88, 0.96))
    return _save(fig, output_dir / "11_run2_validation_latency_by_profile")


def plot_validation_quality_components(
    inputs: Inputs, output_dir: Path
) -> Tuple[Path, Path]:
    """Rejoin Run-2 actions to the bound quality-component surface."""

    rows = tuple(
        row
        for row in inputs.contexts
        if row["source_kind"] == RUN2
        and _integer(row, "seed") in SEEDS
        and _integer(row, "update_index") in UPDATES
    )
    if len(rows) != len(SEEDS) * len(UPDATES) * len(PROFILES) * 85:
        raise PresentationError(f"unexpected Run-2 quality row count: {len(rows)}")
    derived = quality_components_for_rows(rows, project_root=PROJECT_ROOT)
    enriched = tuple(zip(rows, derived))
    specifications = (
        ("q_loc", "Localization quality", "#0072B2", "-"),
        ("q_seg", "Segmentation quality (diagnostic)", "#D55E00", "--"),
        ("q_perc", "Joint perception quality", "#009E73", "-."),
    )
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.2), sharex=True, sharey=True)
    for axis, profile in zip(axes.flat, PROFILES):
        for field, label, color, linestyle in specifications:
            means = []
            for update in UPDATES:
                values = [
                    components[field]
                    for row, components in enriched
                    if _integer(row, "update_index") == update
                    and row["network_profile"] == profile
                ]
                if len(values) != len(SEEDS) * 85:
                    raise PresentationError(
                        f"expected 255 quality rows for {(update, profile, field)}"
                    )
                means.append(float(np.mean(values)))
            axis.plot(
                UPDATES,
                means,
                color=color,
                linestyle=linestyle,
                marker="o",
                markersize=3.6,
                linewidth=2.1,
                label=label,
            )
        axis.set_title(PROFILE_LABELS[profile].replace("\n", " "))
        axis.set_xticks(UPDATES[::2])
        axis.set_ylim(0.0, 1.0)
        _bold_ticks(axis)
    for axis in axes[-1, :]:
        axis.set_xlabel("SAC gradient updates")
    for axis in axes[:, 0]:
        axis.set_ylabel("Quality score")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center right",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )
    fig.suptitle("Run 2: perception-quality behavior across checkpoints")
    setattr(fig, "_splitfusion_tight_layout_rect", (0.0, 0.025, 0.86, 0.96))
    fig.text(
        0.5,
        0.005,
        "Qloc and Qperc passed surface qualification; off-anchor Qseg interpolation is diagnostic only.",
        ha="center",
        fontsize=9,
    )
    return _save(fig, output_dir / "12_run2_validation_quality_components")


def _headline_summary(inputs: Inputs) -> Dict[str, object]:
    run2 = _policy_row(inputs, RUN2)
    run1 = _policy_row(inputs, RUN1)
    fixed = _policy_row(inputs, FIXED)
    oracle = _policy_row(inputs, ORACLE)
    update0 = aggregate_row(inputs, RUN2, -1, 0, "ALL_PROFILES")
    final_miss_latencies: List[float] = []
    for seed in SEEDS:
        for row in context_rows(inputs, RUN2, seed, 5000, "ADVERSE_STABLE"):
            if _integer(row, "p95_miss") == 1:
                final_miss_latencies.append(_finite(row, "latency_proxy_p95_ms"))
    return {
        "scope": {
            "evaluation_population": "DEVELOPMENT_FIT_VALIDATION_NOT_FINAL_TEST",
            "latency_semantics": (
                "MODELED_CONDITIONAL_RETAINED_SURVIVOR_P95_PROXY_NOT_LIVE_END_TO_END_SLA"
            ),
            "run1_comparison": "RUN1_RESCORED_UNDER_COMMON_V2_REWARD",
        },
        "utility": {
            "run2_update0": _finite(update0, "v2_utility_mean"),
            "run1_rescored": _finite(run1, "v2_utility_mean"),
            "run2_update5000": _finite(run2, "v2_utility_mean"),
            "fixed": _finite(fixed, "v2_utility_mean"),
            "oracle": _finite(oracle, "v2_utility_mean"),
            "normalized_random_to_oracle_progress": _finite(
                run2, "normalized_random_to_oracle_progress_unclamped"
            ),
        },
        "run2_final": {
            "p95_miss_count": _integer(run2, "p95_miss_count"),
            "context_count": _integer(run2, "row_count"),
            "p95_miss_rate": _finite(run2, "p95_miss_rate"),
            "mean_modeled_p95_ms": _finite(run2, "latency_p95_ms_mean"),
            "composite_perception_quality": _finite(run2, "q_perc_mean"),
            "admission_probability": _finite(run2, "p_edge_admission_given_sent_mean"),
            "miss_latency_min_ms": min(final_miss_latencies),
            "miss_latency_median_ms": float(np.median(final_miss_latencies)),
            "miss_latency_max_ms": max(final_miss_latencies),
            "all_misses_profile": "ADVERSE_STABLE",
        },
        "adaptation": dict(inputs.adaptation),
        "tiers": dict(inputs.tiers),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise PresentationError(f"refusing to write empty CSV {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_tables(inputs: Inputs, output_dir: Path) -> Tuple[Path, Path]:
    comparator_rows: List[Mapping[str, object]] = []
    entries = (
        ("Run 2 initial", RUN2, 0),
        ("Run 1 (V2 rescore)", RUN1, 5000),
        ("Run 2", RUN2, 5000),
        ("Fixed", FIXED, -1),
        ("Oracle", ORACLE, -1),
    )
    for label, source, update in entries:
        row = aggregate_row(inputs, source, -1, update, "ALL_PROFILES")
        comparator_rows.append(
            {
                "policy": label,
                "source_kind": source,
                "v2_utility": _finite(row, "v2_utility_mean"),
                "composite_perception_quality": _finite(row, "q_perc_mean"),
                "mean_modeled_p95_ms": _finite(row, "latency_p95_ms_mean"),
                "p95_miss_count": _integer(row, "p95_miss_count"),
                "context_count": _integer(row, "row_count"),
                "p95_miss_rate": _finite(row, "p95_miss_rate"),
                "admission_probability": _finite(row, "p_edge_admission_given_sent_mean"),
            }
        )
    comparator_path = output_dir / "policy_comparison.csv"
    _write_csv(comparator_path, comparator_rows)

    trajectory_rows: List[Mapping[str, object]] = []
    for update in UPDATES:
        row = aggregate_row(inputs, RUN2, -1, update, "ALL_PROFILES")
        trajectory_rows.append(
            {
                "update_index": update,
                "v2_utility": _finite(row, "v2_utility_mean"),
                "normalized_random_to_oracle_progress": _finite(
                    row, "normalized_random_to_oracle_progress_unclamped"
                ),
                "composite_perception_quality": _finite(row, "q_perc_mean"),
                "mean_modeled_p95_ms": _finite(row, "latency_p95_ms_mean"),
                "p95_miss_count": _integer(row, "p95_miss_count"),
                "p95_miss_rate": _finite(row, "p95_miss_rate"),
            }
        )
    trajectory_path = output_dir / "run2_checkpoint_trajectory.csv"
    _write_csv(trajectory_path, trajectory_rows)
    return comparator_path, trajectory_path


def _write_guide(output_dir: Path, summary: Mapping[str, object]) -> Path:
    utility = summary["utility"]
    final = summary["run2_final"]
    adaptation = summary["adaptation"]
    tiers = summary["tiers"]
    assert isinstance(utility, Mapping)
    assert isinstance(final, Mapping)
    assert isinstance(adaptation, Mapping)
    assert isinstance(tiers, Mapping)
    lines = [
        "# Run-2 presentation figure guide",
        "",
        "## Scientific scope",
        "",
        "These figures use the completed frozen development fit-validation panel. They are evidence for preliminary learning behavior, not untouched final testing or a live end-to-end SLA. All policies are compared under the same V2 scoring rule; Run 1 is rescored rather than plotted using its incompatible native reward.",
        "",
        "## Suggested talking points",
        "",
        "1. **Learning trajectory.** Run 2 moves from negative utility at initialization to positive utility quickly, peaks around update 1500, and then plateaus. Update 5000 remains the pre-registered primary endpoint; the earlier peak is not cherry-picked.",
        "2. **Deadline learning.** The modeled P95 miss rate falls from 75.0% at initialization and 76.0% for Run 1 to 2.25% at Run 2's final checkpoint. This is the clearest evidence that the revised reward taught the intended constraint.",
        "3. **What improved and what did not.** Run 2 is far better than Run 1, but it remains below both the fixed baseline and contextual oracle in V2 utility. Its perception quality reaches 82.1% of oracle quality, below the registered 90% Tier-2 gate.",
        "4. **Where the residual misses occur.** All 23 final misses are in the adverse profile and are near-boundary misses: their modeled P95 values span about 200.2–205.2 ms.",
        "5. **Mode collapse diagnosis.** Run 2 always selects AE32/UINT4 and adapts only the continuous drop fraction q. The oracle uses several discrete modes. This motivates the next bounded diagnosis of discrete-mode learning; it does not justify claiming policy superiority yet.",
        "6. **Continuous adaptation is directionally sensible.** Run 2 increases q as the channel becomes worse, meaning it drops more low-saliency spatial cells to reduce payload. The fixed comparator cannot make this adjustment.",
        "7. **Honest adaptation result.** The learned policy is 0.0221 utility below the fixed comparator, with the entire 95% scene-cluster bootstrap interval below zero. This is a useful negative result that identifies the remaining training problem.",
        "",
        "## Figure map",
        "",
        "- `01_run2_learning_utility`: learning curve plus fair common-score references.",
        "- `02_run2_deadline_learning`: modeled P95 deadline misses versus updates.",
        "- `03_final_policy_comparison`: utility, composite quality, P95 latency and miss rate together.",
        "- `04_p95_deadline_misses_by_profile`: where deadline behavior changes across channel profiles.",
        "- `05_mode_selection_run2_vs_oracle`: the discrete-mode collapse compared with oracle diversity.",
        "- `06_continuous_q_by_profile`: continuous compression adaptation across network profiles.",
        "- `07_quality_vs_p95_by_profile`: the quality–latency trade-off in each network profile.",
        "- `08_p95_budget_margin_by_profile`: distance from the 200-ms modeled P95 boundary.",
        "- `09_run2_minus_fixed_utility`: per-seed and pooled adaptation result with bootstrap CI.",
        "- `10_run2_training_replay_batch_reward`: raw and 100-update-smoothed replay-minibatch reward.",
        "- `11_run2_validation_latency_by_profile`: modeled P50/P95/P99 latency trajectories against 200 ms.",
        "- `12_run2_validation_quality_components`: localization, diagnostic segmentation and joint quality trajectories.",
        "",
        "## Exact headline values",
        "",
        f"- Run 2 final utility: `{float(utility['run2_update5000']):.6f}`.",
        f"- Fixed utility: `{float(utility['fixed']):.6f}`; oracle utility: `{float(utility['oracle']):.6f}`.",
        f"- Normalized random-to-oracle progress: `{100.0 * float(utility['normalized_random_to_oracle_progress']):.2f}%`.",
        f"- Final misses: `{int(final['p95_miss_count'])}/{int(final['context_count'])}` (`{100.0 * float(final['p95_miss_rate']):.2f}%`).",
        f"- Run 2 minus fixed: `{float(adaptation['pooled_mean_lift']):.6f}`, 95% CI `[{float(adaptation['lower_95']):.6f}, {float(adaptation['upper_95']):.6f}]`.",
        f"- Registered gates: Tier 0 `{tiers['tier0']['passed']}`, Tier 1 `{tiers['tier1']['passed']}`, Tier 2 `{tiers['tier2']['passed']}`, Tier 3 `{tiers['tier3']['passed']}`.",
        "",
        "Composite perception quality is the registered localization-first quality coordinate modulated by segmentation. It is dimensionless on [0,1], but it is not a probability and need not approach 1 for learning to be valid.",
    ]
    path = output_dir / "FIGURE_GUIDE.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def render(inputs: Inputs, output_dir: Path) -> Mapping[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise PresentationError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _style()
    produced: List[Path] = []
    for renderer in (
        plot_utility_learning,
        plot_deadline_learning,
        plot_final_comparator_summary,
        plot_profile_miss_comparison,
        plot_mode_selection,
        plot_q_adaptation,
        plot_profile_quality_latency,
        plot_p95_margin,
        plot_adaptation_lift,
        plot_training_replay_reward,
        plot_validation_latency_trajectory,
        plot_validation_quality_components,
    ):
        produced.extend(renderer(inputs, output_dir))

    produced.extend(_write_tables(inputs, output_dir))
    summary = _headline_summary(inputs)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    produced.append(summary_path)
    produced.append(_write_guide(output_dir, summary))

    artifact_hashes = {path.name: sha256_file(path) for path in sorted(produced)}
    manifest = {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "source_root": str(inputs.source_root.resolve()),
        "source_hashes": dict(inputs.source_hashes),
        "artifact_hashes": artifact_hashes,
        "figure_count": 12,
        "figure_formats": ["PNG_300_DPI", "VECTOR_PDF"],
        "scope": summary["scope"],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=project_root
        / "experiments/splitfusion_hybrid_sac_run2_evaluation_v2/20260922_exact_p95_run2_v2",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    inputs = load_inputs(args.source.resolve())
    manifest = render(inputs, args.output.resolve())
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print("SPLITFUSION_HYBRID_SAC_RUN2_PRESENTATION_V1_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
