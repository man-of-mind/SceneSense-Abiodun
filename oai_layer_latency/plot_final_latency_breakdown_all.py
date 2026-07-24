#!/usr/bin/env python3
"""Final slide: stacked p50 latency breakdown across main mitigation runs.

Compares:
  - ideal loopback software floor
  - OAI 273PRB adaptive, full no-AE payload
  - OAI 273PRB fixed MCS28 diagnostic
  - OAI 273PRB adaptive, reduced no-AE uint4 payload
  - OAI 106PRB adaptive, reduced AE128/u6/ROI0.5 payload

The bar height is the p50 capture->result estimate from the run summaries.
To make the stacked bars close exactly, "uplink" is derived as:

  capture_to_result_p50 - front_p50 - back_p50 - downlink_p50

This folds the small independent-median residual into the uplink segment, which
is reasonable here because the residual is tiny relative to the OAI uplink queue.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12.5,
        "axes.labelsize": 14,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.titlesize": 17,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "figure.titlesize": 21,
        "figure.dpi": 220,
        "savefig.dpi": 430,
        "axes.linewidth": 1.35,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


AB = Path(__file__).resolve().parents[1]
RUNS = AB / "downlink_latency_fps" / "runs"
OUT = AB / "oai_layer_latency" / "plots"
OUT.mkdir(parents=True, exist_ok=True)


RUN_SPECS = [
    {
        "label": "Loopback\nfloor",
        "condition": "Ideal loopback",
        "path": RUNS / "downlink_fps_summary_drivable_rerun_20260722_loopback.csv",
        "selector": lambda df: df[df["fps"].eq(10)].iloc[0],
        "note": "1.05 MB, no OAI",
    },
    {
        "label": "OAI 273\nadaptive\nfull payload",
        "condition": "OAI 273PRB adaptive, no-AE uint8",
        "path": RUNS / "downlink_fps_summary_drivable_rerun_20260722_bw273.csv",
        "selector": lambda df: df.iloc[0],
        "note": "1.05 MB",
    },
    {
        "label": "OAI 273\nfixed\nMCS28",
        "condition": "OAI 273PRB fixed MCS28, no-AE uint8",
        "path": RUNS / "downlink_fps_summary_forcemcs28_bw273_20260723.csv",
        "selector": lambda df: df.iloc[0],
        "note": "1.06 MB",
    },
    {
        "label": "OAI 273\nadaptive\nuint4",
        "condition": "OAI 273PRB adaptive, no-AE uint4",
        "path": RUNS / "downlink_fps_summary_int4_adaptive_20260723.csv",
        "selector": lambda df: df.iloc[0],
        "note": "395 KB",
    },
    {
        "label": "OAI 106\nadaptive\nAE128 u6",
        "condition": "OAI 106PRB adaptive, AE128 uint6 ROI0.5",
        "path": RUNS / "downlink_fps_summary_ae128_u6_roi05_default106_20260723.csv",
        "selector": lambda df: df.iloc[0],
        "note": "153 KB",
    },
]


COMPONENTS = [
    ("front_ms", "Front feature prep", "#64748B"),
    ("uplink_ms", "OAI uplink queue/transport", "#D1495B"),
    ("edge_ms", "Edge tail inference", "#2E86AB"),
    ("downlink_ms", "Downlink result", "#10B981"),
]


def load_rows() -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for spec in RUN_SPECS:
        df = pd.read_csv(spec["path"])
        row = spec["selector"](df)
        total = float(row["capture_to_result_est_p50_ms"])
        front = float(row["front_p50_ms"])
        edge = float(row["back_p50_ms"])
        downlink = float(row["result_send_to_recv_wall_p50_ms"])
        uplink = max(total - front - edge - downlink, 0.0)
        direct_uplink = float(row["feature_upload_payload_handling_p50_ms"])
        rows.append(
            {
                "label": spec["label"],
                "condition": spec["condition"],
                "note": spec["note"],
                "frames": int(row["frames"]),
                "received": int(row["received"]),
                "delivery_pct": float(row["delivery"]) * 100.0,
                "feature_kb_p50": float(row["feature_kb_p50"]),
                "result_kb_p50": float(row["result_kb_p50"]),
                "capture_to_result_p50_ms": total,
                "rtt_p50_ms": float(row["rtt_recv_p50_ms"]),
                "front_ms": front,
                "uplink_ms": uplink,
                "direct_uplink_p50_ms": direct_uplink,
                "edge_ms": edge,
                "downlink_ms": downlink,
                "median_residual_folded_into_uplink_ms": uplink - direct_uplink,
            }
        )
    return pd.DataFrame(rows)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", color="#CBD5E1", linewidth=1.0, alpha=0.55)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", labelsize=11.5, width=1.3, length=5)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.35)


def main() -> int:
    df = load_rows()
    df.to_csv(OUT / "final_latency_breakdown_all_conditions_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(14.0, 7.6))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.82, bottom=0.17)
    fig.suptitle(
        "Latency breakdown across OAI mitigation runs",
        y=0.965,
        fontweight="bold",
    )

    x = np.arange(len(df))
    width = 0.62
    bottom = np.zeros(len(df))

    for key, name, color in COMPONENTS:
        vals = df[key].to_numpy(dtype=float)
        bars = ax.bar(
            x,
            vals,
            width,
            bottom=bottom,
            label=name,
            color=color,
            edgecolor="white",
            linewidth=1.15,
        )
        for bar, val, base in zip(bars, vals, bottom):
            if val >= 8.0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    base + val / 2,
                    f"{val:.0f}",
                    ha="center",
                    va="center",
                    fontsize=10.5,
                    fontweight="bold",
                    color="white",
                )
        bottom += vals

    totals = df["capture_to_result_p50_ms"].to_numpy(dtype=float)
    for i, total in enumerate(totals):
        ax.text(
            x[i],
            total + 6,
            f"{total:.0f} ms\n{df.loc[i, 'delivery_pct']:.1f}% del.",
            ha="center",
            va="bottom",
            fontsize=12.0,
            fontweight="bold",
            color="#0F172A",
            bbox={"facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.95, "pad": 2.8},
        )
        ax.text(
            x[i],
            3.2,
            str(df.loc[i, "note"]),
            ha="center",
            va="bottom",
            fontsize=10.4,
            fontweight="bold",
            color="white",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "#334155", "edgecolor": "white", "alpha": 0.82},
        )

    loopback_total = float(df.loc[0, "capture_to_result_p50_ms"])
    ax.axhline(loopback_total, color="#0F172A", linestyle="--", linewidth=1.7, alpha=0.58)

    full_uplink = float(df.loc[1, "uplink_ms"])
    fixed_uplink = float(df.loc[2, "uplink_ms"])
    uint4_uplink = float(df.loc[3, "uplink_ms"])
    ae_uplink = float(df.loc[4, "uplink_ms"])
    ax.text(
        3.05,
        202,
        "Takeaway\n"
        f"• Fixed MCS28: uplink {full_uplink:.0f} → {fixed_uplink:.0f} ms\n"
        f"• Smaller adaptive bursts: {uint4_uplink:.0f} / {ae_uplink:.0f} ms uplink\n"
        "• Payload reduction also reduces latency",
        ha="left",
        va="top",
        fontsize=11.7,
        fontweight="bold",
        color="#0F172A",
        linespacing=1.22,
        bbox={"boxstyle": "round,pad=0.42", "facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.96},
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"])
    ax.set_ylabel("p50 capture → result latency (ms)")
    ax.set_ylim(0, max(totals) * 1.24)
    style_axis(ax)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.045),
        ncol=4,
        frameon=True,
        facecolor="white",
        framealpha=0.96,
        fontsize=11.4,
    )

    fig.text(
        0.5,
        0.055,
        "Corrected drivable-route 10 FPS summaries. Fixed MCS28 is diagnostic; reduced-payload adaptive runs show payload reduction also reduces latency.",
        ha="center",
        va="center",
        fontsize=11.2,
        fontweight="bold",
        color="#475569",
    )

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"final_latency_breakdown_all_conditions.{ext}", bbox_inches="tight")
    plt.close(fig)

    print(OUT / "final_latency_breakdown_all_conditions.pdf")
    print(OUT / "final_latency_breakdown_all_conditions_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
