#!/usr/bin/env python3
"""Plot the first TRACTOR-over-OAI vanilla replay baselines."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


AB = Path(__file__).resolve().parents[1]
OUT = AB / "bursty_traffic" / "plots"
ANALYSIS = AB / "bursty_traffic" / "analysis"

RUNS = [
    (
        "OneDrive eMBB\n6.1 Mbps",
        "tractor_replay_bw273_vanilla_embb0303a_off100_60s_tcpdump_20260727_210512",
    ),
    (
        "Google Meet\n1.1 Mbps",
        "tractor_replay_bw273_vanilla_urllc0303_off240_60s_tcpdump_20260727_210842",
    ),
]


def load() -> pd.DataFrame:
    rows = []
    for label, rg in RUNS:
        p = AB / "metrics_logs" / "tractor_replay" / rg / "tractor_oai_summary.csv"
        row = pd.read_csv(p).iloc[0].to_dict()
        row["label"] = label
        rows.append(row)
    df = pd.DataFrame(rows)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    df.to_csv(ANALYSIS / "tractor_oai_vanilla_baseline_summary.csv", index=False)
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    colors = ["#4c78a8", "#54a24b"]

    fig, axs = plt.subplots(1, 4, figsize=(14.0, 4.0))
    panels = [
        ("packet_delivery", "Packet delivery", "Delivery is clean", lambda v: f"{v*100:.0f}%"),
        ("ul_avg_mcs_p50", "Median UL MCS", "MCS stays high", lambda v: f"{v:.1f}"),
        ("rlc_queue_wait_mean_ms", "Mean UE RLC wait (ms)", "No RLC backlog", lambda v: f"{v:.1f}"),
        ("branch_decrease_few_samples_pct", "Few-sample branch (%)", "Few-sample branch alone\nis not sufficient", lambda v: f"{v:.0f}%"),
    ]
    for ax, (col, ylabel, title, fmt) in zip(axs, panels):
        ax.bar(df["label"], df[col], color=colors, width=0.62)
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        for i, v in enumerate(df[col]):
            ax.text(i, v * 1.03 if v > 0 else 0.02, fmt(float(v)), ha="center", fontweight="bold", fontsize=9)
    fig.suptitle("TRACTOR real bursty traffic over ideal OAI 273PRB: vanilla scheduler baseline", fontweight="bold", y=1.05)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"tractor_oai_vanilla_baseline_summary.{ext}", dpi=240, bbox_inches="tight")


if __name__ == "__main__":
    main()
