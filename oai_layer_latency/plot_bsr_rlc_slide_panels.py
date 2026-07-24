#!/usr/bin/env python3
"""Generate slide-ready UE BSR / RLC queue panels.

These figures support the post-scheduler part of the OAI latency story:

1. iperf smooth UDP keeps UE BSR/RLC backlog tiny, while CARLA split-feature
   bursts create whole-frame UE RLC backlog.
2. For the same CARLA ~1 MB payload, forcing MCS28 collapses the persistent RLC
   queue compared with adaptive low-MCS behavior.

The plotted traces use 1-second p95 windows so sparse burst backlogs are visible
without dumping millions of per-slot samples into a slide.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
        "axes.titlesize": 15,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.titlesize": 22,
        "figure.dpi": 220,
        "savefig.dpi": 420,
        "axes.linewidth": 1.35,
        "lines.linewidth": 2.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


ROOT = Path(__file__).resolve().parents[1]
TRACE_ROOT = ROOT / "metrics_logs/scenesense_ttracer"
OUT_DIR = ROOT / "oai_layer_latency/plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class TraceSpec:
    key: str
    label: str
    run_dir: Path
    color: str
    window_s: Optional[float] = None

    @property
    def ue_csv_dir(self) -> Path:
        return self.run_dir / "ue/csv"


IPERF = TraceSpec(
    key="iperf",
    label="iperf smooth UDP\nMCS28, small steady backlog",
    run_dir=TRACE_ROOT / "validate_20260722_175810_iperf_ul",
    color="#2563EB",
)

CARLA_ADAPTIVE = TraceSpec(
    key="carla_adaptive",
    label="CARLA split tensor\nadaptive low MCS, ~1 MB bursts",
    run_dir=TRACE_ROOT / "downlink_oai_bw273_mu1_ttracer_fps10_layerinstr_20260723_145921",
    color="#D1495B",
    window_s=250.0,
)

CARLA_FIXED = TraceSpec(
    key="carla_fixed_mcs28",
    label="CARLA split tensor\nforced MCS28 diagnostic",
    run_dir=TRACE_ROOT / "downlink_oai_bw273_mu1_ttracer_forcemcs28_fps10_forcemcs28_bw273_20260723",
    color="#2E86AB",
    window_s=250.0,
)


def parse_clock(series: pd.Series) -> pd.Series:
    """Convert HH:MM:SS.ffffff to elapsed seconds, handling midnight wrap."""
    parts = series.astype(str).str.split(":", expand=True)
    if parts.shape[1] != 3:
        raise ValueError("time column is not HH:MM:SS.ffffff")
    sec = (
        pd.to_numeric(parts[0], errors="coerce") * 3600
        + pd.to_numeric(parts[1], errors="coerce") * 60
        + pd.to_numeric(parts[2], errors="coerce")
    )
    # Very defensive midnight wrap handling.
    jumps = sec.diff().lt(-43200).fillna(False).cumsum() * 86400
    elapsed = sec + jumps
    return elapsed - elapsed.min()


def aggregate_p95(df: pd.DataFrame, value_col: str, window_s: Optional[float], bin_s: float = 1.0) -> pd.DataFrame:
    out = df.copy()
    out["t"] = parse_clock(out["time"])
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.dropna(subset=["t", value_col])
    if window_s is not None:
        out = out[out["t"].le(window_s)]
    if out.empty:
        return pd.DataFrame(columns=["t", "p50_kb", "p95_kb", "max_kb"])
    out["bin"] = np.floor(out["t"] / bin_s).astype(int)
    agg = (
        out.groupby("bin", as_index=False)
        .agg(
            t=("bin", lambda x: float(x.iloc[0]) * bin_s),
            p50_kb=(value_col, lambda x: np.nanpercentile(x, 50) / 1024.0),
            p95_kb=(value_col, lambda x: np.nanpercentile(x, 95) / 1024.0),
            max_kb=(value_col, lambda x: np.nanmax(x) / 1024.0),
        )
        .sort_values("t")
    )
    return agg


def load_bsr(spec: TraceSpec) -> pd.DataFrame:
    path = spec.ue_csv_dir / "NRUE_MAC_BSR_STATUS.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, usecols=["time", "lcg1_bytes"])
    return aggregate_p95(df, "lcg1_bytes", spec.window_s)


def load_rlc(spec: TraceSpec) -> pd.DataFrame:
    path = spec.ue_csv_dir / "NRUE_MAC_RLC_BUFFER_STATUS.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    chunks = []
    for chunk in pd.read_csv(path, usecols=["time", "lcid", "bytes_in_buffer"], chunksize=750_000):
        chunk["lcid"] = pd.to_numeric(chunk["lcid"], errors="coerce")
        chunks.append(chunk[chunk["lcid"].eq(4)][["time", "bytes_in_buffer"]])
    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=["time", "bytes_in_buffer"])
    return aggregate_p95(df, "bytes_in_buffer", spec.window_s)


def summarize(name: str, metric: str, df: pd.DataFrame) -> dict[str, float | str]:
    if df.empty:
        return {"run": name, "metric": metric, "p50_kb": np.nan, "p95_kb": np.nan, "max_kb": np.nan}
    return {
        "run": name,
        "metric": metric,
        "p50_kb": float(np.nanpercentile(df["p50_kb"], 50)),
        "p95_kb": float(np.nanpercentile(df["p95_kb"], 95)),
        "max_kb": float(np.nanmax(df["max_kb"])),
    }


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="both", color="#CBD5E1", linewidth=1.0, alpha=0.55)
    ax.tick_params(axis="both", which="major", labelsize=11, width=1.3, length=5)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.35)


def add_stat_box(ax: plt.Axes, df: pd.DataFrame, color: str) -> None:
    if df.empty:
        text = "no data"
    else:
        text = (
            f"p95-window median: {np.nanmedian(df['p95_kb']):.1f} KB\n"
            f"peak window max: {np.nanmax(df['max_kb']):.1f} KB"
        )
    ax.text(
        0.985,
        0.82,
        text,
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color="#0F172A",
        bbox={"facecolor": "white", "edgecolor": color, "linewidth": 1.0, "alpha": 0.88, "pad": 4},
    )


def plot_single_trace(ax: plt.Axes, df: pd.DataFrame, spec: TraceSpec, ylabel: str) -> None:
    ax.plot(df["t"], df["p95_kb"], color=spec.color, linewidth=3.0, solid_capstyle="round")
    ax.fill_between(df["t"], 0, df["p95_kb"], color=spec.color, alpha=0.12)
    ax.set_title(spec.label, loc="left", pad=8)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Elapsed time (s)")
    add_stat_box(ax, df, spec.color)
    style_axis(ax)


def make_iperf_vs_carla(bsr: dict[str, pd.DataFrame], rlc: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 9.1), sharex=False)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.855, bottom=0.115, hspace=0.53, wspace=0.22)
    fig.suptitle("UE backlog formation: smooth iperf stays small; CARLA queues whole feature bursts", y=0.965, fontweight="bold")

    plot_single_trace(axes[0, 0], bsr[IPERF.key], IPERF, "BSR LCG1 backlog\np95 / 1s (KB)")
    plot_single_trace(axes[0, 1], bsr[CARLA_ADAPTIVE.key], CARLA_ADAPTIVE, "BSR LCG1 backlog\np95 / 1s (KB)")
    plot_single_trace(axes[1, 0], rlc[IPERF.key], IPERF, "RLC LCID4 occupancy\np95 / 1s (KB)")
    plot_single_trace(axes[1, 1], rlc[CARLA_ADAPTIVE.key], CARLA_ADAPTIVE, "RLC LCID4 occupancy\np95 / 1s (KB)")

    axes[0, 0].set_ylim(0, max(12, axes[0, 0].get_ylim()[1]))
    axes[1, 0].set_ylim(0, max(12, axes[1, 0].get_ylim()[1]))
    for ax in (axes[0, 1], axes[1, 1]):
        ax.axhline(1024, color="#475569", linestyle="--", linewidth=1.7, alpha=0.75)
        ax.text(
            0.985,
            0.18,
            "~1 MB feature frame",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color="#475569",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2},
        )
        ax.set_ylim(0, 1160)

    fig.text(
        0.5,
        0.035,
        "UE-side metrics from T-tracer. BSR is LCG1 reported backlog; RLC is data-bearer LCID4 occupancy. CARLA panels show first 250 s.",
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color="#475569",
    )
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"bsr_rlc_iperf_vs_carla_timeseries.{ext}", bbox_inches="tight")
    plt.close(fig)


def make_adaptive_vs_fixed(bsr: dict[str, pd.DataFrame], rlc: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(15.7, 8.7), sharex=True)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.845, bottom=0.13, hspace=0.38)
    fig.suptitle("Fixed MCS28 diagnostic: higher spectral efficiency drains the UE queue", y=0.97, fontweight="bold")

    panels = [
        (axes[0], bsr, "BSR backlog\np95 / 1s (KB)", "A. Offered CARLA bursts still appear in BSR"),
        (axes[1], rlc, "RLC occupancy\np95 / 1s (KB)", "B. Fixed MCS28 drains RLC faster, so persistent queueing falls"),
    ]
    for ax, dataset, ylabel, title in panels:
        for spec in (CARLA_ADAPTIVE, CARLA_FIXED):
            df = dataset[spec.key]
            ax.plot(df["t"], df["p95_kb"], color=spec.color, linewidth=3.0, label=spec.label.replace("\n", " — "))
        ax.set_title(title, loc="left", pad=8)
        ax.axhline(1024, color="#475569", linestyle="--", linewidth=1.7, alpha=0.70)
        ax.text(
            0.985,
            0.83,
            "~1 MB feature frame",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10.5,
            fontweight="bold",
            color="#475569",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.80, "pad": 2},
        )
        ax.set_ylim(0, 1160)
        ax.set_ylabel(ylabel)
        style_axis(ax)
    axes[1].set_xlabel("Elapsed time (s)")
    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=2,
        frameon=True,
        facecolor="white",
        framealpha=0.92,
        fontsize=11.5,
    )
    fig.text(
        0.5,
        0.04,
        "Same ~1 MB CARLA payload on 273PRB RFsim. Adaptive low MCS leaves persistent backlog; fixed MCS28 turns it into short drain spikes.",
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color="#475569",
    )
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"bsr_rlc_adaptive_vs_fixed_mcs_timeseries.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    specs = [IPERF, CARLA_ADAPTIVE, CARLA_FIXED]
    bsr = {spec.key: load_bsr(spec) for spec in specs}
    rlc = {spec.key: load_rlc(spec) for spec in specs}

    rows = []
    for spec in specs:
        rows.append(summarize(spec.key, "bsr_lcg1_p95_1s", bsr[spec.key]))
        rows.append(summarize(spec.key, "rlc_lcid4_p95_1s", rlc[spec.key]))
    pd.DataFrame(rows).to_csv(OUT_DIR / "bsr_rlc_slide_panel_summary.csv", index=False)

    make_iperf_vs_carla(bsr, rlc)
    make_adaptive_vs_fixed(bsr, rlc)

    for path in [
        OUT_DIR / "bsr_rlc_iperf_vs_carla_timeseries.pdf",
        OUT_DIR / "bsr_rlc_adaptive_vs_fixed_mcs_timeseries.pdf",
        OUT_DIR / "bsr_rlc_slide_panel_summary.csv",
    ]:
        print(path)


if __name__ == "__main__":
    main()
