#!/usr/bin/env python3
"""Make slide-ready plots for the uplink-only staleness-budget analysis.

These figures are intentionally simpler than the analysis plots in
``results/plots``. They use a small number of high-contrast colors and labels
that explain the policy story:

  frame capture -> sensor prep -> split/uplink/tail -> map update

The latency-breakdown plot uses additive per-frame components. In particular,
the map term is ``tail_done -> map_update_done``. Do not add
``edge_to_map_publish_ms`` to the map term; that column starts at edge receive
and therefore includes tail work.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-uplink-staleness")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path("abiodun/staleness/uplink_only_latency_budget")
RESULTS = ROOT / "results"
OUT = RESULTS / "plots" / "presentation"
MODEL_FLOOR_M = 1.1
MPH_PER_MS = 2.236936
FPS_LIST = [1, 5, 10, 15, 20, 25, 30]

FAST_ANCHOR_MAP = Path(
    "abiodun/uplink_only_spatial_map_pipeline/runs/live_front_prep_profile_fast_50f/map_ingest_metrics.csv"
)
FRESH_ROOT = ROOT / "fresh_run_20260730_000257"

COLORS = {
    "Sensor prep": "#2563EB",  # blue
    "Front encode": "#F97316",  # orange
    "Loopback uplink": "#16A34A",  # green
    "Edge tail": "#7C3AED",  # purple
    "Map insert/update": "#0891B2",  # cyan
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
            "legend.fontsize": 10.5,
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


def q(s: pd.Series, quantile: float) -> float:
    vals = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return float("nan")
    return float(vals.quantile(quantile))


def load_map_csv(path: Path, warmup: int = 10) -> pd.DataFrame:
    df = pd.read_csv(path)
    if len(df) > warmup:
        df = df.iloc[warmup:].copy()
    return add_additive_components(df)


def add_additive_components(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-frame additive staleness components."""
    out = df.copy()
    out["Sensor prep"] = pd.to_numeric(out["capture_to_backbone_input_ms"], errors="coerce")
    out["Front encode"] = pd.to_numeric(out["backbone_input_to_front_send_ms"], errors="coerce")
    out["Loopback uplink"] = (
        pd.to_numeric(out["backbone_input_to_edge_recv_ms"], errors="coerce")
        - pd.to_numeric(out["backbone_input_to_front_send_ms"], errors="coerce")
    ).clip(lower=0.0)
    out["Edge tail"] = (
        pd.to_numeric(out["backbone_input_to_tail_done_ms"], errors="coerce")
        - pd.to_numeric(out["backbone_input_to_edge_recv_ms"], errors="coerce")
    ).clip(lower=0.0)
    out["Map insert/update"] = (
        pd.to_numeric(out["capture_to_map_update_done_ms"], errors="coerce")
        - out["Sensor prep"]
        - out["Front encode"]
        - out["Loopback uplink"]
        - out["Edge tail"]
    ).clip(lower=0.0)
    return out


def fresh_pooled() -> pd.DataFrame:
    frames = []
    for path in sorted(FRESH_ROOT.glob("L_*/map_ingest_metrics.csv")):
        df = load_map_csv(path)
        df["source"] = path.parent.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No fresh L map_ingest_metrics.csv under {FRESH_ROOT}")
    return pd.concat(frames, ignore_index=True)


def breakdown_summary() -> pd.DataFrame:
    rows = []
    specs = [
        ("Optimized Track-1\nL = 93 ms run", load_map_csv(FAST_ANCHOR_MAP)),
        ("Optimized speed-sweep\nL = 67 ms run", fresh_pooled()),
    ]
    components = list(COLORS)
    for label, df in specs:
        row = {"condition": label, "L_p50": q(df["capture_to_map_update_done_ms"], 0.5), "L_p95": q(df["capture_to_map_update_done_ms"], 0.95)}
        for comp in components:
            row[f"{comp}_p50"] = q(df[comp], 0.5)
            row[f"{comp}_p95"] = q(df[comp], 0.95)
        row["map_insert_mean"] = float(pd.to_numeric(df["Map insert/update"], errors="coerce").mean())
        row["map_insert_p95"] = q(df["Map insert/update"], 0.95)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "presentation_latency_breakdown_summary.csv", index=False)
    return out


def _latency_columns(df: pd.DataFrame) -> list[tuple[int, str]]:
    cols = []
    for col in df.columns:
        if col.startswith("err_m_L") and col.endswith("ms"):
            try:
                latency = int(col.removeprefix("err_m_L").removesuffix("ms"))
            except ValueError:
                continue
            cols.append((latency, col))
    return sorted(cols)


def _error_from_speed_floor(speed_ms: np.ndarray | float, floor_m: np.ndarray | float, total_staleness_ms: float) -> np.ndarray:
    """Kinematic staleness model: sqrt(floor^2 + (v * age)^2)."""
    return np.sqrt(np.asarray(floor_m, dtype=float) ** 2 + (np.asarray(speed_ms, dtype=float) * total_staleness_ms / 1000.0) ** 2)


def plot_latency_breakdown() -> None:
    df = breakdown_summary()
    components = list(COLORS)
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(12.0, 5.4))
    left = np.zeros(len(df))
    for comp in components:
        vals = df[f"{comp}_p50"].to_numpy(dtype=float)
        ax.barh(
            y,
            vals,
            left=left,
            color=COLORS[comp],
            edgecolor="white",
            linewidth=1.2,
            label=comp,
        )
        for yi, v, lft in zip(y, vals, left):
            if v >= 5:
                ax.text(lft + v / 2, yi, f"{v:.0f}", ha="center", va="center", color="white", fontweight="bold", fontsize=10)
        left += vals

    for yi, total, p95 in zip(y, df["L_p50"], df["L_p95"]):
        ax.scatter(total, yi, marker="D", s=64, color="#111827", zorder=5)
        ax.text(total + 4, yi - 0.08, f"L p50 {total:.0f} ms", ha="left", va="center", fontweight="bold", color="#111827")
        ax.plot([total, p95], [yi + 0.28, yi + 0.28], color="#111827", linewidth=2.1)
        ax.plot([p95, p95], [yi + 0.20, yi + 0.36], color="#111827", linewidth=2.1)
        ax.text(p95 + 3, yi + 0.28, f"p95 {p95:.0f}", ha="left", va="center", fontsize=10.5, color="#111827")

    ax.set_yticks(y)
    ax.set_yticklabels(df["condition"], fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Freshness age from frame capture to map update (ms)")
    ax.set_title("Uplink-only staleness budget: optimized pipeline")
    ax.legend(loc="upper right", frameon=True, framealpha=0.96)
    ax.set_xlim(0, max(df["L_p95"]) * 1.32)
    ax.text(
        0.01,
        -0.22,
        "Map insert/update is tail_done→map_update_done. Radar tensor build is folded into Sensor prep.",
        transform=ax.transAxes,
        fontsize=10.5,
        color="#374151",
        fontweight="bold",
    )
    save(fig, "presentation_staleness_budget_breakdown")


def plot_error_vs_speed() -> None:
    df = pd.read_csv(RESULTS / "error_vs_L_by_speed.csv")
    x = df["mean_speed_mph"].to_numpy(float)
    lines = [
        ("L=0 ms (model floor)", "err_m_L0ms", "#64748B", "--"),
        ("L=67 ms", "err_m_L67ms", "#16A34A", "-"),
        ("L=93 ms", "err_m_L93ms", "#2563EB", "-"),
        ("L=136 ms", "err_m_L136ms", "#7C3AED", "-"),
    ]
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    for label, col, color, ls in lines:
        ax.plot(x, df[col], marker="o", linewidth=3.0, markersize=7, color=color, linestyle=ls, label=label)
    for eps, color in [(1.5, "#B45309"), (2.0, "#7C3AED")]:
        ax.axhline(eps, color=color, linestyle=":", linewidth=2.2)
        ax.text(x.max() + 0.5, eps, f"ε={eps:.1f} m", va="center", ha="left", color=color, fontweight="bold")
    ax.set_xlim(0, 35.5)
    ax.set_ylim(0.9, 2.65)
    ax.set_xlabel("Observed object speed (mph)")
    ax.set_ylabel("Localization error (m)")
    ax.set_title("Latency hurts most when the observed target is moving fast")
    ax.legend(loc="upper left", frameon=True, framealpha=0.96)
    ax.annotate(
        "32 mph: 1.67 m @67 ms\nvs 1.94 m @93 ms",
        xy=(33.0, float(df.loc[df["speed_band"].eq("~32 mph"), "err_m_L93ms"].iloc[0])),
        xytext=(15.8, 2.07),
        arrowprops=dict(arrowstyle="->", color="#111827", linewidth=1.6),
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#CBD5E1", alpha=0.98),
        fontweight="bold",
    )
    save(fig, "presentation_error_vs_speed_by_staleness")


def plot_error_vs_latency_by_speed() -> None:
    """Direct story plot: latency on x-axis, localization error on y-axis."""
    df = pd.read_csv(RESULTS / "error_vs_L_by_speed.csv")
    latency_cols = _latency_columns(df)
    latencies = np.array([x for x, _ in latency_cols], dtype=float)
    speed_order = ["~walk/slow", "~6 mph", "~10 mph", "~18 mph", "~23 mph", "~28 mph", "~32 mph"]
    color_map = {
        "~walk/slow": "#64748B",
        "~6 mph": "#0891B2",
        "~10 mph": "#16A34A",
        "~18 mph": "#F59E0B",
        "~23 mph": "#F97316",
        "~28 mph": "#DC2626",
        "~32 mph": "#7C3AED",
    }
    fig, ax = plt.subplots(figsize=(12.0, 6.5))
    for band in speed_order:
        sub = df[df["speed_band"].eq(band)]
        if sub.empty:
            continue
        row = sub.iloc[0]
        y = np.array([float(row[col]) for _, col in latency_cols])
        ax.plot(
            latencies,
            y,
            marker="o",
            markersize=5.5,
            linewidth=2.7,
            color=color_map.get(band, "#111827"),
            label=f"{band} target",
        )

    for x, label, color in [
        (67, "67 ms", "#16A34A"),
        (100, "100 ms", "#2563EB"),
        (212, "212 ms", "#DC2626"),
    ]:
        ax.axvline(x, color=color, linestyle=":", linewidth=2.6, alpha=0.95)
        ax.text(
            x + 2,
            1.06,
            label,
            rotation=90,
            va="bottom",
            ha="left",
            color=color,
            fontweight="bold",
            fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor=color, alpha=0.90),
        )

    ax.axhline(1.5, color="#92400E", linestyle="--", linewidth=1.8, alpha=0.80)
    ax.axhline(2.0, color="#4338CA", linestyle="--", linewidth=1.8, alpha=0.80)
    ax.text(301, 1.5, "1.5 m", ha="left", va="center", color="#92400E", fontweight="bold")
    ax.text(301, 2.0, "2.0 m", ha="left", va="center", color="#4338CA", fontweight="bold")
    ax.set_xlim(0, 310)
    ax.set_ylim(0.95, 4.25)
    ax.set_xlabel("Capture→map latency L (ms)")
    ax.set_ylabel("Localization error (m)")
    ax.set_title("Localization error grows with latency and target speed")
    ax.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.96)
    ax.text(
        0.01,
        -0.18,
        "Vertical dotted lines are measured/reference latency levels. Curves use the post-hoc GT(t+L) speed-sweep results.",
        transform=ax.transAxes,
        fontsize=10.5,
        color="#374151",
        fontweight="bold",
    )
    save(fig, "presentation_error_vs_latency_by_speed")


def plot_error_vs_speed_by_fps_and_latency() -> None:
    """Direct map-staleness plot: fixed L panels, FPS curves, target speed on x-axis."""
    df = pd.read_csv(RESULTS / "error_vs_L_by_speed.csv").sort_values("mean_speed_mph")
    speeds_mph = df["mean_speed_mph"].to_numpy(float)
    speeds_ms = df["mean_speed_ms"].to_numpy(float)
    floor = df["err_m_L0ms"].to_numpy(float)
    L_values = [0, 93, 150, 200]
    colors = {
        1: "#991B1B",
        5: "#DC2626",
        10: "#F97316",
        15: "#F59E0B",
        20: "#16A34A",
        25: "#0891B2",
        30: "#2563EB",
    }
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.0), sharex=True, sharey=True)
    for ax, L_ms in zip(axes.flat, L_values):
        for fps in FPS_LIST:
            total_ms = L_ms + 1000.0 / fps
            y = _error_from_speed_floor(speeds_ms, floor, total_ms)
            lw = 3.0 if fps in (10, 20, 30) else 2.1
            ax.plot(speeds_mph, y, marker="o", markersize=4.8, linewidth=lw, color=colors[fps], label=f"{fps} FPS")
        ax.axhline(2.5, color="#92400E", linestyle=":", linewidth=1.8, alpha=0.85)
        ax.axhline(4.0, color="#4338CA", linestyle=":", linewidth=1.8, alpha=0.85)
        ax.set_title(f"Fixed latency L = {L_ms} ms", fontweight="bold")
        ax.set_xlim(0, 35)
        ax.set_ylim(1.0, 16.8)
        ax.text(
            29.8,
            2.50,
            "2.5 m",
            ha="left",
            va="bottom",
            color="#92400E",
            fontsize=9.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="#92400E", alpha=0.90),
        )
        ax.text(
            29.8,
            4.00,
            "4.0 m",
            ha="left",
            va="bottom",
            color="#4338CA",
            fontsize=9.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="#4338CA", alpha=0.90),
        )
        ax.tick_params(axis="y", labelleft=True)
        ax.tick_params(axis="x", labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel("Localization error (m)")
    for ax in axes[-1, :]:
        ax.set_xlabel("Observed target speed (mph)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.01, 0.5), frameon=True, framealpha=0.96)
    fig.suptitle("FPS controls map-hold staleness; latency shifts every curve upward", fontweight="bold", y=0.995)
    fig.text(
        0.5,
        0.015,
        "Assumption shown: worst-case map hold. Total age = L + 1/FPS, i.e., just before the next map update.",
        ha="center",
        va="bottom",
        fontsize=11,
        color="#374151",
        fontweight="bold",
    )
    fig.subplots_adjust(right=0.88, top=0.91, bottom=0.09, hspace=0.22, wspace=0.12)
    save(fig, "presentation_error_vs_speed_fps_by_latency")


ROADSTATE_SPEED_ORDER = ["walk/slow", "~6 mph", "~10 mph", "~14 mph", "~18 mph", "~23 mph", "~28-32 mph"]
ROADSTATE_COLORS = {
    "walk/slow": "#64748B",
    "~6 mph": "#0891B2",
    "~10 mph": "#16A34A",
    "~14 mph": "#F59E0B",
    "~18 mph": "#F97316",
    "~23 mph": "#DC2626",
    "~28-32 mph": "#7C3AED",
}


def _plot_one_roadstate(
    ax: plt.Axes,
    df: pd.DataFrame,
    state: str,
    title: str,
    show_ylabel: bool = True,
    include_counts: bool = True,
) -> None:
    sub = df[df["road_state"].eq(state)].copy()
    for band in ROADSTATE_SPEED_ORDER:
        b = sub[sub["speed_band"].eq(band)].sort_values("L_ms")
        if b.empty:
            continue
        n = int(b["n"].iloc[0])
        label = f"{band} (n={n})" if include_counts else band
        ax.plot(
            b["L_ms"],
            b["loc_error_m"],
            marker="o",
            markersize=5.8,
            linewidth=2.8,
            color=ROADSTATE_COLORS[band],
            label=label,
        )
    for x, color in [(67, "#16A34A"), (93, "#2563EB")]:
        ax.axvline(x, color=color, linestyle=":", linewidth=2.0, alpha=0.95)
        ax.text(
            x + 1.8,
            0.93,
            f"L={x} ms",
            rotation=90,
            va="bottom",
            ha="left",
            color=color,
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor=color, alpha=0.88),
        )
    for y, label, color in [(1.5, "1.5 m", "#92400E"), (2.0, "2.0 m", "#4338CA")]:
        ax.axhline(y, color=color, linestyle="--", linewidth=1.7, alpha=0.75)
        ax.text(
            135,
            y + 0.025,
            label,
            ha="right",
            va="bottom",
            color=color,
            fontsize=9.2,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor=color, alpha=0.88),
        )
    ax.set_title(title, fontweight="bold")
    ax.set_xlim(0, 138)
    ax.set_ylim(0.85, 2.38)
    ax.set_xlabel("Capture→map latency L (ms)")
    if show_ylabel:
        ax.set_ylabel("Localization error (m)")
    ax.tick_params(axis="y", labelleft=True)


def plot_roadstate_speed_presentation() -> None:
    df = pd.read_csv(RESULTS / "roadstate_error_by_speed.csv")
    state_specs = [
        ("straight", "Straight road", "presentation_uplink_roadstate_straight_speed"),
        ("curve", "Curve", "presentation_uplink_roadstate_curve_speed"),
        ("junction", "Intersection", "presentation_uplink_roadstate_intersection_speed"),
    ]

    for state, title, stem in state_specs:
        fig, ax = plt.subplots(figsize=(10.8, 6.4))
        _plot_one_roadstate(ax, df, state, title, show_ylabel=True)
        ax.set_title(f"{title}: localization error vs latency by target speed", fontweight="bold")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, framealpha=0.96)
        ax.text(
            0.01,
            -0.18,
            "Road state from Town10 waypoint: junction vs curve/straight by lane yaw change. Use speed-matched comparisons; curve bins are sparse/confounded.",
            transform=ax.transAxes,
            fontsize=9.6,
            color="#374151",
            fontweight="bold",
        )
        save(fig, stem)

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.6), sharey=True)
    for ax, (state, title, _) in zip(axes, state_specs):
        _plot_one_roadstate(ax, df, state, title, show_ylabel=(ax is axes[0]), include_counts=False)
    present_bands = [b for b in ROADSTATE_SPEED_ORDER if df["speed_band"].eq(b).any()]
    handles = [
        Line2D([0], [0], color=ROADSTATE_COLORS[b], marker="o", linewidth=2.8, markersize=5.8, label=b)
        for b in present_bands
    ]
    labels = present_bands
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.005, 0.5), frameon=True, framealpha=0.96)
    fig.suptitle("Road state split: speed and latency dominate localization error", fontweight="bold", y=0.995)
    fig.text(
        0.5,
        0.015,
        "Read within the same speed band. Road-state mix is speed-confounded, especially curves, so this is a robustness split rather than proof of a road-type causal effect.",
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#374151",
        fontweight="bold",
    )
    fig.subplots_adjust(right=0.82, top=0.86, bottom=0.17, wspace=0.16)
    save(fig, "presentation_uplink_roadstate_speed_summary")


def plot_max_staleness_budget() -> None:
    df = pd.read_csv(RESULTS / "budget_latency_upper.csv")
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    colors = {1.5: "#B45309", 2.0: "#2563EB", 2.5: "#16A34A", 3.0: "#7C3AED"}
    for eps in [1.5, 2.0, 2.5, 3.0]:
        sub = df[df["eps_m"].eq(eps)].copy()
        mph = sub["v_ms"].astype(float) * 2.236936
        y = pd.to_numeric(sub["max_L_ms_closedform"], errors="coerce").clip(upper=320)
        ax.plot(mph, y, marker="o", linewidth=2.8, markersize=6.5, color=colors[eps], label=f"ε≤{eps:.1f} m")
    for val, label, color in [(67.5, "L = 67 ms", "#16A34A"), (93.3, "L = 93 ms", "#2563EB")]:
        ax.axhline(val, color=color, linestyle="--", linewidth=2.2, alpha=0.85)
        ax.text(
            1.0,
            val + 4,
            label,
            ha="left",
            va="bottom",
            color=color,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=color, alpha=0.92),
        )
    ax.set_xlim(0, 35)
    ax.set_ylim(0, 320)
    ax.set_xlabel("Object speed (mph)")
    ax.set_ylabel("Max allowed capture→map staleness L (ms)")
    ax.set_title("Latency-only budget: how much capture→map staleness can the agent afford?")
    ax.legend(loc="upper right", frameon=True, framealpha=0.96)
    ax.annotate(
        "Example: 20 mph, ε=1.5 m\nLmax ≈ 114 ms (latency only)",
        xy=(20, math.sqrt(1.5**2 - 1.1**2) / (20 / 2.236936) * 1000),
        xytext=(8.5, 168),
        arrowprops=dict(arrowstyle="->", color="#111827", linewidth=1.6),
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#CBD5E1", alpha=0.98),
        fontweight="bold",
    )
    ax.text(
        0.01,
        -0.17,
        "This plot budgets L only. If the map holds detections between updates, add map-hold age separately: worst case total age = L + 1/FPS.",
        transform=ax.transAxes,
        fontsize=10.3,
        color="#374151",
        fontweight="bold",
    )
    save(fig, "presentation_max_staleness_budget_by_speed")


def _fps_value(x: object) -> float:
    if isinstance(x, str) and x.upper().startswith("INFEAS"):
        return float("nan")
    return float(x)


def plot_fps_requirement() -> None:
    df = pd.read_csv(RESULTS / "budget_fps_lower.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), sharey=True)
    fig.subplots_adjust(right=0.86, top=0.82)
    for ax, eps in zip(axes, [1.5, 2.0]):
        for L, label, color in [(67.5, "L=67 ms", "#16A34A"), (93.3, "L=93 ms", "#2563EB")]:
            sub = df[df["eps_m"].eq(eps) & df["L_ms"].eq(L)].copy()
            mph = sub["v_ms"].astype(float) * 2.236936
            vals = sub["fps_min_worstcase"].map(_fps_value).astype(float)
            capped = vals.clip(upper=60)
            ax.plot(mph, capped, marker="o", linewidth=2.8, markersize=6.5, color=color, label=label)
            over_cap = vals > 60
            if over_cap.any():
                ax.scatter(mph[over_cap], np.full(int(over_cap.sum()), 59.0), marker="^", s=92, color=color, zorder=5)
            for x, y, raw in zip(mph, capped, vals):
                if np.isnan(raw):
                    ax.scatter([x], [58], marker="x", s=80, color=color, linewidths=2.5)
            if np.isnan(vals).any() or over_cap.any():
                ax.text(
                    1.1,
                    57.2,
                    "top markers = >60 FPS",
                    ha="left",
                    va="bottom",
                    color="#374151",
                    fontsize=8.8,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#D1D5DB", alpha=0.92),
                )
        ax.set_title(f"ε≤{eps:.1f} m", fontweight="bold")
        ax.set_xlabel("Object speed (mph)")
        ax.set_ylim(0, 62)
        ax.set_xlim(0, 35)
        ax.axhline(10, color="#6B7280", linestyle=":", linewidth=1.8)
        ax.axhline(20, color="#6B7280", linestyle=":", linewidth=1.8)
        ax.text(1, 10.8, "10 FPS", fontsize=9.5, color="#4B5563", fontweight="bold")
        ax.text(1, 20.8, "20 FPS", fontsize=9.5, color="#4B5563", fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(0.985, 0.5), frameon=True, framealpha=0.96)
    axes[0].set_ylabel("Minimum FPS, worst-case map hold")
    fig.suptitle("FPS is a freshness lever: fast objects need higher update rate", fontweight="bold", y=0.98)
    save(fig, "presentation_fps_requirement_by_speed")


def plot_20mph_example() -> None:
    fps = np.array([5, 10, 15, 20, 25, 30], dtype=float)
    v = 20 / 2.236936
    floor = 1.1
    eps = 1.5
    B = math.sqrt(eps**2 - floor**2)
    allowed_worst = (B / v - 1.0 / fps) * 1000
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.plot(fps, allowed_worst, marker="o", linewidth=3.0, markersize=7, color="#DC2626", label="Worst-case map hold: L + 1/FPS")
    ax.axhline(67.5, color="#16A34A", linestyle="--", linewidth=2.2, label="L = 67 ms")
    ax.axhline(93.3, color="#7C3AED", linestyle="--", linewidth=2.2, label="L = 93 ms")
    ax.axhline(150, color="#F97316", linestyle="--", linewidth=2.0, label="L = 150 ms")
    ax.axhline(200, color="#991B1B", linestyle="--", linewidth=2.0, label="L = 200 ms")
    ax.fill_between(fps, 0, np.maximum(allowed_worst, 0), color="#DC2626", alpha=0.08)
    ax.set_ylim(0, 220)
    ax.set_xlim(4, 31)
    ax.set_xlabel("Map update FPS")
    ax.set_ylabel("Max allowed capture→map L (ms)")
    ax.set_title("20 mph example: FPS changes the latency budget")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, framealpha=0.96)
    ax.text(
        0.03,
        0.93,
        "At 20 mph, ε=1.5 m: latency budget shrinks by 1/FPS map hold",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#CBD5E1", alpha=0.98),
    )
    save(fig, "presentation_20mph_e15_worked_example")


def write_notes() -> None:
    txt = """# Presentation plot notes

Generated by `make_presentation_plots.py` and the companion fairness check
`make_fair_speed_latency_plots.py`.

- `presentation_staleness_budget_breakdown`: additive p50 components from frame capture to map update for
  the optimized pipeline only. Radar tensor build is folded into Sensor prep. Map insert/update is
  `tail_done -> map_update_done`.
- `presentation_error_vs_speed_by_staleness`: why fast objects are sensitive to `L` while slow objects sit
  near the model floor.
- `presentation_error_vs_latency_by_speed`: direct 0-300 ms latency sweep at different target speeds, with
  67/100/212 ms reference lines.
- `presentation_error_vs_latency_by_speed_common_floor`: recommended main-slide version for the speed/latency
  budget story. It uses a common 1.1 m model floor and isolates speed/latency from sample-count and road-state
  artifacts.
- `presentation_error_vs_latency_by_speed_empirical_equal_n`: equal-sample-count empirical check. Useful as
  backup/appendix evidence; it balances sample count only and does not fix the road-state mix in the sparse
  ~23 mph bin.
- `presentation_error_vs_speed_fps_by_latency`: direct FPS × speed × latency plot. Uses worst-case map hold:
  total age = `L + 1/FPS`.
- `presentation_uplink_roadstate_{straight,curve,intersection}_speed`: road-state splits of localization
  error vs latency by speed band.
- `presentation_uplink_roadstate_speed_summary`: compact 3-panel road-state summary.
- `presentation_max_staleness_budget_by_speed`: policy lookup for max allowed capture→map staleness.
- `presentation_fps_requirement_by_speed`: FPS lower bound once map-hold staleness is included.
- `presentation_20mph_e15_worked_example`: concrete 20 mph / 1.5 m budget example using worst-case map hold.

Map timing caveat: `edge_to_map_publish_ms` is not a standalone map-insert term; it starts at edge receive
and includes edge-tail work. For the staleness budget, use `tail_done -> map_update_done`. The current map
packet is small JSON/zlib, but the measured map term still includes Python UDP receive scheduling,
decompression/JSON parsing, normalization, queue admission, and update apply. It does not include future
association/occlusion/cooperative fusion logic.
"""
    (OUT / "README.md").write_text(txt)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    plot_latency_breakdown()
    plot_error_vs_speed()
    plot_error_vs_latency_by_speed()
    plot_error_vs_speed_by_fps_and_latency()
    plot_roadstate_speed_presentation()
    plot_max_staleness_budget()
    plot_fps_requirement()
    plot_20mph_example()
    write_notes()
    print(f"Wrote presentation plots to {OUT}")


if __name__ == "__main__":
    main()
