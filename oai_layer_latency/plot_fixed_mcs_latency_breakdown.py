#!/usr/bin/env python3
"""Slide-ready latency breakdown: loopback vs OAI adaptive vs fixed MCS28.

Uses the corrected drivable-route no-AE/zstd/~1 MB 10 FPS summaries:
  - ideal loopback floor
  - 273PRB OAI adaptive low-MCS
  - 273PRB OAI fixed MCS28 diagnostic

The stacked bars use the Step-1 component definitions:
  front_ms + uplink_payload_handling_ms + edge_tail_ms + downlink_ms
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 11,
        "figure.titlesize": 22,
        "figure.dpi": 220,
        "savefig.dpi": 420,
        "axes.linewidth": 1.35,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_ROOT = ROOT / "downlink_latency_fps/runs"
OUT_DIR = ROOT / "oai_layer_latency/plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


RUNS = [
    {
        "label": "Ideal loopback\nsoftware floor",
        "short": "Loopback",
        "path": SUMMARY_ROOT / "downlink_fps_summary_drivable_rerun_20260722_loopback.csv",
        "selector": lambda df: df[df["fps"].eq(10)].iloc[0],
    },
    {
        "label": "OAI 273PRB\nadaptive low MCS",
        "short": "Adaptive",
        "path": SUMMARY_ROOT / "downlink_fps_summary_drivable_rerun_20260722_bw273.csv",
        "selector": lambda df: df.iloc[0],
    },
    {
        "label": "OAI 273PRB\nfixed MCS28",
        "short": "Fixed MCS28",
        "path": SUMMARY_ROOT / "downlink_fps_summary_forcemcs28_bw273_20260723.csv",
        "selector": lambda df: df.iloc[0],
    },
]


COMPONENTS = [
    ("front_p50_ms", "Front feature prep", "#64748B"),
    ("feature_upload_payload_handling_p50_ms", "Uplink handling", "#D1495B"),
    ("back_p50_ms", "Edge tail inference", "#2E86AB"),
    ("result_send_to_recv_wall_p50_ms", "Downlink result", "#10B981"),
]


def load_rows() -> pd.DataFrame:
    rows = []
    for spec in RUNS:
        df = pd.read_csv(spec["path"])
        row = spec["selector"](df)
        out = {
            "run": spec["short"],
            "label": spec["label"],
            "delivery_pct": float(row["delivery"]) * 100.0,
            "payload_kb": float(row["feature_kb_p50"]),
            "result_kb": float(row["result_kb_p50"]),
            "capture_to_result_ms": float(row["capture_to_result_est_p50_ms"]),
            "rtt_ms": float(row["rtt_recv_p50_ms"]),
        }
        for key, _, _ in COMPONENTS:
            out[key] = float(row[key])
        out["component_sum_ms"] = sum(out[key] for key, _, _ in COMPONENTS)
        rows.append(out)
    return pd.DataFrame(rows)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", color="#CBD5E1", linewidth=1.0, alpha=0.55)
    ax.tick_params(axis="both", which="major", labelsize=12, width=1.3, length=5)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.35)


def main() -> None:
    df = load_rows()

    fig, ax = plt.subplots(figsize=(12.8, 7.1))
    fig.subplots_adjust(left=0.095, right=0.975, top=0.83, bottom=0.18)
    fig.suptitle("System latency breakdown: fixed MCS28 pulls OAI close to the loopback floor", y=0.965, fontweight="bold")

    x = np.arange(len(df))
    width = 0.56
    bottom = np.zeros(len(df))
    for key, name, color in COMPONENTS:
        vals = df[key].to_numpy()
        bars = ax.bar(x, vals, width, bottom=bottom, label=name, color=color, edgecolor="white", linewidth=1.2)
        for i, (bar, v, b) in enumerate(zip(bars, vals, bottom)):
            if v >= 7:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    b + v / 2,
                    f"{v:.0f}",
                    ha="center",
                    va="center",
                    fontsize=10.5,
                    fontweight="bold",
                    color="white",
                )
        bottom += vals

    totals = df["capture_to_result_ms"].to_numpy()
    for i, total in enumerate(totals):
        ax.text(
            x[i],
            bottom[i] + 7,
            f"{total:.1f} ms",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color="#0F172A",
            bbox={"facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.95, "pad": 3},
        )

    loop_total = float(df.loc[df["run"].eq("Loopback"), "capture_to_result_ms"].iloc[0])
    adaptive_total = float(df.loc[df["run"].eq("Adaptive"), "capture_to_result_ms"].iloc[0])
    fixed_total = float(df.loc[df["run"].eq("Fixed MCS28"), "capture_to_result_ms"].iloc[0])
    adaptive_uplink = float(df.loc[df["run"].eq("Adaptive"), "feature_upload_payload_handling_p50_ms"].iloc[0])
    fixed_uplink = float(df.loc[df["run"].eq("Fixed MCS28"), "feature_upload_payload_handling_p50_ms"].iloc[0])

    ax.axhline(loop_total, color="#0F172A", linestyle="--", linewidth=1.8, alpha=0.62)

    ax.text(
        1.36,
        158,
        f"Uplink handling collapses\n{adaptive_uplink:.0f} → {fixed_uplink:.0f} ms",
        ha="left",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color="#0F172A",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#D1495B", "alpha": 0.94},
    )
    ax.text(
        1.36,
        102,
        f"Total p50: {adaptive_total:.0f} → {fixed_total:.0f} ms\n"
        f"fixed is +{fixed_total - loop_total:.0f} ms over loopback",
        ha="left",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color="#0F172A",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#2E86AB", "alpha": 0.94},
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"])
    ax.set_ylabel("p50 capture → result latency (ms)")
    ax.set_ylim(0, max(bottom) * 1.23)
    style_axis(ax)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=4,
        frameon=True,
        facecolor="white",
        framealpha=0.94,
        fontsize=11.5,
    )

    fig.text(
        0.5,
        0.065,
        "Same corrected drivable route, no-AE/zstd/~1 MB payload, 10 FPS. Fixed MCS28 is a diagnostic control, not a deployment policy for fading channels.",
        ha="center",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color="#475569",
    )

    summary = df[
        [
            "run",
            "delivery_pct",
            "payload_kb",
            "result_kb",
            "capture_to_result_ms",
            "rtt_ms",
            *[key for key, _, _ in COMPONENTS],
        ]
    ].copy()
    summary.to_csv(OUT_DIR / "fixed_mcs_latency_breakdown_summary.csv", index=False)

    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fixed_mcs_latency_breakdown_vs_loopback.{ext}", bbox_inches="tight")
    plt.close(fig)

    print(OUT_DIR / "fixed_mcs_latency_breakdown_vs_loopback.pdf")
    print(OUT_DIR / "fixed_mcs_latency_breakdown_summary.csv")


if __name__ == "__main__":
    main()
