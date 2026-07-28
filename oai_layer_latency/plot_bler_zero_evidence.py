#!/usr/bin/env python3
"""Plot UL BLER evidence from the direct OAI BLER/OLLA trace.

This figure is intentionally small and slide-friendly: it shows the filtered
UL BLER value used by OAI's MCS selector at each actual update window, with
the default 15% high-BLER decrement threshold as a reference.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / "oai_layer_latency" / "plots"

RUNS = {
    "Observed CARLA-paced burst": ROOT
    / "metrics_logs"
    / "scenesense_ttracer"
    / "carla_shape_udp_bw273_blertrace_observed_20260723_173640"
    / "gnb"
    / "csv"
    / "GNB_MAC_BLER_MCS_DECISION.csv",
    "Open-loop 10 FPS burst": ROOT
    / "metrics_logs"
    / "scenesense_ttracer"
    / "carla_shape_udp_bw273_blertrace_openloop_20260723_174042"
    / "gnb"
    / "csv"
    / "GNB_MAC_BLER_MCS_DECISION.csv",
}


def load_ul_updates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[(df["direction"] == 1) & (df["updated"] == 1)].copy()
    # The trace stores only clock time. These runs do not cross midnight, so a
    # dummy date is sufficient for elapsed-time plotting.
    ts = pd.to_datetime("2026-07-23 " + df["time"].astype(str), errors="coerce")
    df["elapsed_s"] = (ts - ts.iloc[0]).dt.total_seconds()
    df["bler_pct"] = df["bler_after_ppm"] / 10000.0
    return df


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    colors = ["#1f77b4", "#ff7f0e"]

    summary_rows = []
    for (label, path), color in zip(RUNS.items(), colors):
        df = load_ul_updates(path)
        ax.plot(
            df["elapsed_s"],
            df["bler_pct"],
            lw=3.0,
            color=color,
            alpha=0.9,
            label=f"{label} (max={df['bler_pct'].max():.1f}%)",
        )
        summary_rows.append(
            {
                "label": label,
                "rows": len(df),
                "bler_median_pct": df["bler_pct"].median(),
                "bler_p95_pct": df["bler_pct"].quantile(0.95),
                "bler_max_pct": df["bler_pct"].max(),
                "high_bler_branch_rows": int((df["branch"] == 2).sum()),
                "few_sample_branch_rows": int((df["branch"] == 3).sum()),
            }
        )

    ax.axhline(15.0, color="#d62728", lw=2.0, ls="--", label="OAI high-BLER threshold = 15%")
    ax.set_title("UL BLER stayed at 0% during OAI MCS-selector updates", weight="bold", pad=12)
    ax.set_xlabel("Elapsed time in trace (s)", weight="bold")
    ax.set_ylabel("Filtered UL BLER after update (%)", weight="bold")
    ax.set_ylim(-0.45, 16.5)
    ax.grid(True, axis="y", alpha=0.28)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=True, fontsize=10.5)

    ax.text(
        0.03,
        0.17,
        "All uplink BLER-update samples: median=0%, p95=0%, max=0%\n"
        "High-BLER decrement branch: 0 rows",
        transform=ax.transAxes,
        fontsize=10.0,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#bbbbbb", alpha=0.92),
    )

    fig.subplots_adjust(left=0.12, right=0.98, top=0.86, bottom=0.16)
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS / f"bler_zero_evidence.{ext}", dpi=220)
    pd.DataFrame(summary_rows).to_csv(PLOTS / "bler_zero_evidence_summary.csv", index=False)


if __name__ == "__main__":
    main()
