#!/usr/bin/env python3
"""Generate fairer speed-vs-latency presentation plots.

Why this exists
---------------
The direct empirical plot can make the ~23 mph bin look worse than the ~28 mph
bin. That is not a legend bug; it is a bin-composition artifact:

* ~23 mph is thin in the baseline sweep.
* it is mostly junction samples.
* direct GT(t+L) errors can be lower/higher than a monotonic kinematic model
  depending on whether the model's instantaneous localization error points
  with or against target motion.

This script therefore produces two slide-safe companions:

1. a common-floor modeled budget plot that isolates the effect of speed and L;
2. an equal-N empirical bootstrap plot that uses the same number of samples per
   speed band, so sample count is not the explanation.

The modeled plot is the recommended main-slide figure. The equal-N plot is best
kept as a backup/appendix figure because it still cannot balance road state.
"""

from __future__ import annotations

import csv
import glob
import math
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-uplink-fair-speed")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = RESULTS / "plots" / "presentation"
RUNS_GLOB = str(HERE.parent / "metrics_logs" / "scenesense_runs" / "*")

MODEL_FLOOR_M = 1.10
MPH_PER_MS = 2.236936
NEAR_M = 25.0
MATCH_GATE_M = 2.0
SCORE_GATE = 0.20

LATENCY_REFERENCE_LINES = [
    (67, "67 ms", "#16A34A"),
    (93, "93 ms", "#2563EB"),
    (212, "212 ms", "#DC2626"),
]

BANDS = [
    (0, 4, "~walk/slow"),
    (4, 8, "~6 mph"),
    (8, 12, "~10 mph"),
    (12, 16, "~14 mph"),
    (16, 20, "~18 mph"),
    (20, 26, "~23 mph"),
    (26, 30, "~28 mph"),
    (30, 40, "~32 mph"),
]

# Keep all vehicle-speed bands; omit walk/slow from the equal-N plot so the
# policy story is focused on moving vehicles.
SPEED_ORDER = ["~6 mph", "~10 mph", "~14 mph", "~18 mph", "~23 mph", "~28 mph", "~32 mph"]

COLORS = {
    "~walk/slow": "#64748B",
    "~6 mph": "#0891B2",
    "~10 mph": "#16A34A",
    "~14 mph": "#84CC16",
    "~18 mph": "#F59E0B",
    "~23 mph": "#F97316",
    "~28 mph": "#DC2626",
    "~32 mph": "#7C3AED",
}


def style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 13,
            "axes.labelweight": "bold",
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10.0,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def latency_columns(df: pd.DataFrame) -> list[tuple[int, str]]:
    cols = []
    for col in df.columns:
        if col.startswith("err_m_L") and col.endswith("ms"):
            try:
                ms = int(col.replace("err_m_L", "").replace("ms", ""))
            except ValueError:
                continue
            cols.append((ms, col))
    return sorted(cols)


def add_reference_lines(ax: plt.Axes, ymin: float = 1.04) -> None:
    for x, label, color in LATENCY_REFERENCE_LINES:
        ax.axvline(x, color=color, linestyle=":", linewidth=2.4, alpha=0.95)
        ax.text(
            x + 2,
            ymin,
            label,
            rotation=90,
            va="bottom",
            ha="left",
            color=color,
            fontweight="bold",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=color, alpha=0.92),
        )


def plot_common_floor_model() -> None:
    """Clean policy plot: isolate speed and latency using one model floor."""
    df = pd.read_csv(RESULTS / "error_vs_L_by_speed.csv")
    latencies = np.arange(0, 301, 5, dtype=float)

    fig, ax = plt.subplots(figsize=(12.0, 6.4))
    for band in SPEED_ORDER:
        sub = df[df["speed_band"].eq(band)]
        if sub.empty:
            continue
        row = sub.iloc[0]
        speed_ms = float(row["mean_speed_ms"])
        mean_mph = float(row["mean_speed_mph"])
        y = np.sqrt(MODEL_FLOOR_M**2 + (speed_ms * latencies / 1000.0) ** 2)
        ax.plot(
            latencies,
            y,
            linewidth=3.0,
            marker="o",
            markevery=10,
            markersize=5.5,
            color=COLORS[band],
            label=f"{band} ({mean_mph:.1f} mph)",
        )

    add_reference_lines(ax)
    for y, label, color in [(2.5, "2.5 m budget", "#92400E"), (4.0, "4.0 m budget", "#4338CA")]:
        ax.axhline(y, color=color, linestyle="--", linewidth=1.9, alpha=0.82)
        ax.text(302, y, label, va="center", ha="left", color=color, fontweight="bold")

    ax.set_xlim(0, 310)
    ax.set_ylim(0.95, 4.65)
    ax.set_xlabel("Capture→map latency L (ms)")
    ax.set_ylabel("Localization error budget (m)")
    ax.set_title("Clean speed/latency budget: common 1.1 m model floor")
    ax.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.96)
    ax.text(
        0.01,
        -0.17,
        "Modeled curve: sqrt(1.1² + (target speed × L)²). This isolates speed and latency; it removes sample-count and road-state bin artifacts.",
        transform=ax.transAxes,
        fontsize=10.4,
        color="#374151",
        fontweight="bold",
    )
    save(fig, "presentation_error_vs_latency_by_speed_common_floor")


def truthy(v: object) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def gt_at(sm: list[tuple[float, float, float]], t: float) -> tuple[float, float]:
    if t <= sm[0][0]:
        return sm[0][1], sm[0][2]
    if t >= sm[-1][0]:
        (t0, x0, y0), (t1, x1, y1) = sm[-2], sm[-1]
        dt = t1 - t0
        if dt <= 1e-6:
            return sm[-1][1], sm[-1][2]
        k = (t - t1) / dt
        return x1 + (x1 - x0) * k, y1 + (y1 - y0) * k
    for i in range(1, len(sm)):
        if sm[i][0] >= t:
            (t0, x0, y0), (t1, x1, y1) = sm[i - 1], sm[i]
            k = (t - t0) / max(1e-6, t1 - t0)
            return x0 + (x1 - x0) * k, y0 + (y1 - y0) * k
    return sm[-1][1], sm[-1][2]


def speed_band(v_mph: float) -> str | None:
    for lo, hi, label in BANDS:
        if lo <= v_mph < hi:
            return label
    return None


def is_sweep_run(run: str) -> bool:
    for metrics in glob.glob(run + "/streams/*metrics.csv"):
        try:
            with open(metrics, newline="") as fh:
                rows = csv.DictReader(fh)
                first = next(rows, None)
            if first and str(first.get("run_group", "")).startswith("speedsweep_"):
                return True
        except Exception:
            continue
    return False


def collect_observations() -> dict[str, list[tuple[float, list[tuple[float, float, float]], tuple[float, float], float]]]:
    """Return band -> [(t, trajectory, pred_xy, speed_ms), ...]."""
    grouped: dict[str, list[tuple[float, list[tuple[float, float, float]], tuple[float, float], float]]] = defaultdict(list)
    runs = [r for r in sorted(glob.glob(RUNS_GLOB)) if is_sweep_run(r)]
    if not runs:
        raise RuntimeError(f"No speed-sweep runs found under {RUNS_GLOB}")

    for run in runs:
        gt_files = glob.glob(run + "/streams/*ground_truth.csv")
        pred_files = glob.glob(run + "/streams/*predictions.csv")
        if not gt_files or not pred_files:
            continue
        with open(gt_files[0], newline="") as fh:
            gt_rows = list(csv.DictReader(fh))
        with open(pred_files[0], newline="") as fh:
            pred_rows = list(csv.DictReader(fh))

        traj: dict[str, list[tuple[float, float, float, int, bool, float]]] = defaultdict(list)
        for r in gt_rows:
            if r.get("origin_x") in (None, "") or r.get("origin_y") in (None, ""):
                continue
            try:
                traj[r["actor_id"]].append(
                    (
                        float(r["carla_timestamp"]),
                        float(r["origin_x"]),
                        float(r["origin_y"]),
                        int(r["frame_id"]),
                        truthy(r.get("in_camera_frustum", "")),
                        float(r.get("distance_m", 999.0)),
                    )
                )
            except (KeyError, ValueError):
                continue
        for aid in traj:
            traj[aid].sort()

        preds_by_frame: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for r in pred_rows:
            try:
                if float(r.get("score", 0.0)) >= SCORE_GATE:
                    preds_by_frame[int(r["frame_id"])].append((float(r["world_x"]), float(r["world_y"])))
            except (KeyError, ValueError):
                continue

        for _aid, sm_full in traj.items():
            sm = [(x[0], x[1], x[2]) for x in sm_full]
            for i, (t, x, y, fid, in_frustum, dist_m) in enumerate(sm_full):
                if not (in_frustum and dist_m <= NEAR_M):
                    continue
                preds = preds_by_frame.get(fid, [])
                if not preds:
                    continue
                pred = min(preds, key=lambda p: math.hypot(p[0] - x, p[1] - y))
                if math.hypot(pred[0] - x, pred[1] - y) > MATCH_GATE_M:
                    continue
                j = min(max(1, i), len(sm_full) - 1)
                t0, x0, y0 = sm_full[j - 1][:3]
                t1, x1, y1 = sm_full[j][:3]
                speed_ms = math.hypot(x1 - x0, y1 - y0) / max(1e-6, t1 - t0)
                band = speed_band(speed_ms * MPH_PER_MS)
                if band is not None:
                    grouped[band].append((t, sm, pred, speed_ms))
    return grouped


def plot_empirical_equal_n() -> None:
    """Equal-sample-count empirical check.

    The expectation of a down-sampled mean is still the full-bin mean, so this
    plot is not expected to "fix" the ~23 mph curve. Its role is to make clear
    that sample-count alone is not the whole confound.
    """
    df = pd.read_csv(RESULTS / "error_vs_L_by_speed.csv")
    lat_cols = latency_columns(df)
    latencies = np.array([x for x, _ in lat_cols], dtype=float)
    grouped = collect_observations()
    usable = [band for band in SPEED_ORDER if len(grouped.get(band, [])) > 0]
    n_equal = min(len(grouped[band]) for band in usable)
    rng = np.random.default_rng(17)
    n_boot = 1000

    summary_rows = []
    fig, ax = plt.subplots(figsize=(12.0, 6.4))
    for band in usable:
        obs = grouped[band]
        # Precompute per-observation error curves.
        errors = np.zeros((len(obs), len(latencies)), dtype=float)
        speeds_ms = []
        for i, (t, sm, pred, speed_ms) in enumerate(obs):
            speeds_ms.append(speed_ms)
            for j, L_ms in enumerate(latencies):
                gx, gy = gt_at(sm, t + L_ms / 1000.0)
                errors[i, j] = math.hypot(pred[0] - gx, pred[1] - gy)

        boot = np.zeros((n_boot, len(latencies)), dtype=float)
        for b in range(n_boot):
            idx = rng.choice(len(obs), size=n_equal, replace=False)
            boot[b, :] = errors[idx, :].mean(axis=0)
        mean = boot.mean(axis=0)
        lo = np.percentile(boot, 10, axis=0)
        hi = np.percentile(boot, 90, axis=0)

        full_n = len(obs)
        mean_mph = float(np.mean(speeds_ms) * MPH_PER_MS)
        ax.plot(
            latencies,
            mean,
            linewidth=2.8,
            marker="o",
            markersize=5.0,
            color=COLORS[band],
            label=f"{band} ({mean_mph:.1f} mph, n={n_equal})",
        )
        ax.fill_between(latencies, lo, hi, color=COLORS[band], alpha=0.10, linewidth=0)
        for L_ms, m, l, h in zip(latencies, mean, lo, hi):
            summary_rows.append(
                {
                    "speed_band": band,
                    "full_n": full_n,
                    "equal_n": n_equal,
                    "mean_speed_mph": round(mean_mph, 2),
                    "latency_ms": int(L_ms),
                    "equal_n_mean_error_m": round(float(m), 3),
                    "equal_n_p10_error_m": round(float(l), 3),
                    "equal_n_p90_error_m": round(float(h), 3),
                }
            )

    add_reference_lines(ax)
    for y, label, color in [(2.5, "2.5 m budget", "#92400E"), (4.0, "4.0 m budget", "#4338CA")]:
        ax.axhline(y, color=color, linestyle="--", linewidth=1.8, alpha=0.75)
        ax.text(302, y, label, va="center", ha="left", color=color, fontweight="bold")

    ax.set_xlim(0, 310)
    ax.set_ylim(0.95, 4.70)
    ax.set_xlabel("Capture→map latency L (ms)")
    ax.set_ylabel("Localization error (m)")
    ax.set_title(f"Empirical equal-N check: {n_equal} samples per speed band")
    ax.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.96)
    ax.text(
        0.01,
        -0.18,
        "Equal-N bootstrap balances sample count only. It does not balance road state; the ~23 mph baseline bin is still mostly junction.",
        transform=ax.transAxes,
        fontsize=10.4,
        color="#374151",
        fontweight="bold",
    )
    save(fig, "presentation_error_vs_latency_by_speed_empirical_equal_n")

    pd.DataFrame(summary_rows).to_csv(OUT / "presentation_error_vs_latency_by_speed_empirical_equal_n_summary.csv", index=False)


def main() -> None:
    style()
    plot_common_floor_model()
    plot_empirical_equal_n()
    print(f"Wrote fair speed-latency plots to {OUT}")


if __name__ == "__main__":
    main()
