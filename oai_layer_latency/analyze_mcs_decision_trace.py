#!/usr/bin/env python3
"""Analyze SceneSense gNB UL MCS decision trace.

This reads the custom GNB_MAC_UL_MCS_DECISION T-tracer CSV and writes a compact
markdown summary plus presentation-safe plots.  The key distinction is whether
MCS is already low before nr_ue_max_mcs_min_rb(), or whether that function
reduces an SNR/BLER-selected high MCS after the fact.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


AB_ROOT = Path(__file__).resolve().parents[1]
TTRACER_ROOT = AB_ROOT / "metrics_logs" / "scenesense_ttracer"


def pct(series: pd.Series, q: float) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return float("nan")
    return float(np.percentile(s.to_numpy(), q))


def stat_line(df: pd.DataFrame, col: str, scale: float = 1.0, unit: str = "") -> str:
    s = pd.to_numeric(df[col], errors="coerce").dropna() / scale
    if s.empty:
        return f"| {col} | n/a | n/a | n/a | n/a | n/a |"
    return (
        f"| {col}{unit} | {s.mean():.2f} | {np.percentile(s, 50):.2f} | "
        f"{np.percentile(s, 95):.2f} | {s.min():.2f} | {s.max():.2f} |"
    )


def add_elapsed_seconds(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    t = pd.to_datetime(out["time"], format="%H:%M:%S.%f", errors="coerce")
    if t.isna().all():
        out["elapsed_s"] = np.arange(len(out), dtype=float)
    else:
        out["elapsed_s"] = (t - t.iloc[0]).dt.total_seconds()
        # Handle rare midnight wrap.
        out.loc[out["elapsed_s"] < 0, "elapsed_s"] += 24 * 3600
    return out


def write_summary(run_group: str, df: pd.DataFrame, out_dir: Path, window_csv: Path, plot_paths: list[Path]) -> Path:
    data = df[df["B"] > 0].copy()
    idle = df[df["B"] <= 0].copy()
    phr_reduced = data[data["post_phr_mcs"] < data["pre_phr_mcs"]]
    mcs_changed_after_selector = data[data["final_mcs"] != data["selected_mcs"]]
    pre_to_post_drop = data["pre_phr_mcs"] - data["post_phr_mcs"]

    md = out_dir / "mcs_decision_summary.md"
    lines: list[str] = []
    lines.append("# gNB UL MCS decision trace summary")
    lines.append("")
    lines.append(f"Run group: `{run_group}`")
    lines.append("")
    lines.append("## Main result")
    lines.append("")
    if len(data):
        lines.append(
            f"- Scheduler rows with data queued: **{len(data):,}** of {len(df):,}; "
            f"idle/min-grant rows: {len(idle):,}."
        )
        lines.append(
            f"- Median SNR seen by the gNB trace is **{pct(data['avg_snr_x10'], 50) / 10:.1f} dB**, "
            f"but median selected/final MCS is only **{pct(data['selected_mcs'], 50):.0f}/{pct(data['final_mcs'], 50):.0f}**."
        )
        lines.append(
            f"- `nr_ue_max_mcs_min_rb()` reduced MCS in **{len(phr_reduced):,} rows "
            f"({100 * len(phr_reduced) / len(data):.2f}% of data rows)**."
        )
        lines.append(
            f"- Final MCS differed from selected MCS in **{len(mcs_changed_after_selector):,} rows "
            f"({100 * len(mcs_changed_after_selector) / len(data):.2f}% of data rows)**."
        )
        lines.append("")
        if len(phr_reduced) == 0 and len(mcs_changed_after_selector) == 0:
            lines.append(
                "**Interpretation:** in this observed-CARLA-cadence run, the low MCS is already present "
                "at the MCS-selection stage. The PHR-normalized RB/MCS helper did not lower it further."
            )
        else:
            lines.append(
                "**Interpretation:** at least part of the MCS reduction happens after initial selection; "
                "inspect the pre/post columns before attributing the bottleneck."
            )
    else:
        lines.append("- No scheduler rows with `B > 0` were found.")
    lines.append("")
    lines.append("## Data-row metric summary")
    lines.append("")
    lines.append("| Metric | mean | p50 | p95 | min | max |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for col, scale, unit in [
        ("avg_snr_x10", 10.0, " /10 dB"),
        ("selected_mcs", 1.0, ""),
        ("pre_phr_mcs", 1.0, ""),
        ("post_phr_mcs", 1.0, ""),
        ("final_mcs", 1.0, ""),
        ("B", 1024.0, " KB"),
        ("available_rb_before", 1.0, ""),
        ("available_rb_after", 1.0, ""),
        ("rb_size_final", 1.0, ""),
        ("tbs_final", 1.0, " B"),
        ("ph", 1.0, ""),
    ]:
        lines.append(stat_line(data, col, scale, unit))
    lines.append("")
    lines.append("## Drop checks")
    lines.append("")
    lines.append(
        f"- Pre-PHR minus post-PHR MCS drop: mean={pre_to_post_drop.mean() if len(pre_to_post_drop) else float('nan'):.2f}, "
        f"p95={pct(pre_to_post_drop, 95):.2f}, max={pre_to_post_drop.max() if len(pre_to_post_drop) else 'n/a'}."
    )
    lines.append(
        "- If the drop metrics are zero, the bottleneck is not this PHR helper; it is upstream in initial MCS selection "
        "or in the BLER/OLLA state that feeds selection."
    )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Windowed CSV: `{window_csv}`")
    for p in plot_paths:
        lines.append(f"- Plot: `{p}`")
    lines.append("")
    md.write_text("\n".join(lines))
    return md


def make_plots(df: pd.DataFrame, windows: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths: list[Path] = []
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.plot(windows["elapsed_s"], windows["selected_mcs_p50"], color="#1f77b4", lw=2.0, label="Selected MCS p50")
    ax.plot(windows["elapsed_s"], windows["final_mcs_p50"], color="#d62728", lw=1.8, ls="--", label="Final MCS p50")
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("MCS index")
    ax.set_ylim(-0.5, max(10, windows[["selected_mcs_p95", "final_mcs_p95"]].max().max() + 2))
    ax2 = ax.twinx()
    ax2.fill_between(
        windows["elapsed_s"],
        0,
        windows["B_p95_kB"],
        color="#7f7f7f",
        alpha=0.18,
        label="Backlog B p95 (KB)",
    )
    ax2.set_ylabel("Scheduler backlog B p95 (KB)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", frameon=True)
    ax.set_title("Observed-CARLA cadence: selected/final MCS stays low while backlog bursts")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = out_dir / f"mcs_decision_timeseries.{ext}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)

    data = df[df["B"] > 0].copy()
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    drops = data["pre_phr_mcs"] - data["post_phr_mcs"]
    bins = np.arange(-0.5, max(1, int(drops.max()) + 1.5), 1)
    ax.hist(drops, bins=bins, color="#2ca02c", edgecolor="white")
    ax.set_xlabel("MCS drop from PHR helper (pre_phr_mcs - post_phr_mcs)")
    ax.set_ylabel("Scheduler rows")
    ax.set_title("PHR helper did not reduce MCS in this run" if drops.max() == 0 else "PHR helper MCS-drop distribution")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = out_dir / f"mcs_decision_phr_drop_hist.{ext}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)

    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-group", required=True)
    ap.add_argument("--input-csv", type=Path)
    ap.add_argument("--output-dir", type=Path)
    args = ap.parse_args()

    csv_path = args.input_csv or TTRACER_ROOT / args.run_group / "gnb" / "csv" / "GNB_MAC_UL_MCS_DECISION.csv"
    if not csv_path.exists():
        raise SystemExit(f"missing {csv_path}")
    out_dir = args.output_dir or TTRACER_ROOT / args.run_group / "gnb" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    numeric_cols = [c for c in df.columns if c != "time"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = add_elapsed_seconds(df)

    df["window_s"] = np.floor(df["elapsed_s"]).astype(int)
    grouped = df.groupby("window_s", as_index=False)
    windows = grouped.agg(
        elapsed_s=("elapsed_s", "median"),
        rows=("time", "count"),
        selected_mcs_p50=("selected_mcs", "median"),
        selected_mcs_p95=("selected_mcs", lambda s: np.percentile(s, 95)),
        final_mcs_p50=("final_mcs", "median"),
        final_mcs_p95=("final_mcs", lambda s: np.percentile(s, 95)),
        post_phr_mcs_p50=("post_phr_mcs", "median"),
        B_p50_kB=("B", lambda s: np.percentile(s, 50) / 1024),
        B_p95_kB=("B", lambda s: np.percentile(s, 95) / 1024),
        avg_snr_db_p50=("avg_snr_x10", lambda s: np.percentile(s, 50) / 10),
        rb_size_p50=("rb_size_final", "median"),
        tbs_p50=("tbs_final", "median"),
    )
    window_csv = out_dir / "mcs_decision_windows.csv"
    windows.to_csv(window_csv, index=False)
    plot_paths = make_plots(df, windows, out_dir)
    md = write_summary(args.run_group, df, out_dir, window_csv, plot_paths)
    print(f"[analyze_mcs_decision_trace] wrote {md}")
    print(f"[analyze_mcs_decision_trace] wrote {window_csv}")
    for p in plot_paths:
        print(f"[analyze_mcs_decision_trace] wrote {p}")


if __name__ == "__main__":
    main()
