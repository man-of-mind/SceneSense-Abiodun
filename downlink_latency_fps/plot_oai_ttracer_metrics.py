#!/usr/bin/env python3
"""Presentation plots for corrected CARLA/OAI T-tracer runs.

This script intentionally discovers corrected t-tracer artifacts dynamically
instead of relying on fixed dates. Expected per-run inputs are produced by
`prepare_ttracer_grant_artifacts.py` under:

  metrics_logs/carla_oai_ttracer/<run_group>/

It plots grant PRB/MCS/scheduled-rate, tunnel TX/RX rate, and optional PHY
quality metrics (UE RSRP/SNR/CQI and gNB PUSCH SNR) when OAI emits them.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
ARTIFACT_ROOT = REPO / "metrics_logs" / "carla_oai_ttracer"
PLOTS = ROOT / "plots" / "oai_ttracer"
RUNS = ROOT / "runs"

COLORS = {
    "blue": "#4C78A8",
    "orange": "#F58518",
    "green": "#54A24B",
    "red": "#E45756",
    "purple": "#B279A2",
    "grey": "#666666",
    "light_grey": "#B8B8B8",
}


def num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def rolling(series: pd.Series, window: int = 7) -> pd.Series:
    return num(series).rolling(window=window, min_periods=1, center=True).median()


def finite_phy(series: pd.Series, kind: str) -> pd.Series:
    vals = num(series)
    if kind == "rsrp":
        vals = vals[(vals > -1000) & (vals < 100)]
    elif kind in {"snr", "cqi"}:
        vals = vals[(vals > -40) & (vals < 100)]
    return vals


def save(fig: plt.Figure, stem: str) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(PLOTS / f"{stem}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def safe_stem(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text).strip("_")


def discover_runs() -> list[Path]:
    runs = []
    if not ARTIFACT_ROOT.exists():
        return runs
    for path in sorted(ARTIFACT_ROOT.iterdir(), key=lambda p: p.stat().st_mtime):
        if not path.is_dir():
            continue
        if not (path / "CARLA10_OAI_TTRACER_SUMMARY.csv").exists():
            continue
        if not (path / "nrue_ul_grant_windows_compact.csv").exists():
            continue
        runs.append(path)
    return runs


def run_label(path: Path) -> tuple[str, str, int]:
    name = path.name
    if "bw273" in name or "273" in name:
        return "273PRB wider bandwidth", "bw273", 273
    if "ulheavy" in name:
        return "106PRB UL-heavy 4DL/5UL", "ulheavy106", 106
    if "default" in name:
        return "106PRB default", "default106", 106
    return name, safe_stem(name), 106


def find_frontend_metrics(run_group: str) -> Path | None:
    matches = sorted(RUNS.glob(f"**/streams/{run_group}_metrics.csv"))
    return matches[-1] if matches else None


def frontend_rate_1s(metrics_path: Path) -> pd.DataFrame:
    app = pd.read_csv(metrics_path)
    app["wall_time"] = pd.to_datetime(app["wall_time_iso"], errors="coerce")
    app = app[app["wall_time"].notna()].copy()
    app["feature_payload_bytes"] = num(app["feature_payload_bytes"]).fillna(0.0)
    app["result_payload_bytes_estimate"] = num(app.get("result_payload_bytes_estimate", 0)).fillna(0.0)
    t0 = app["wall_time"].min()
    app["sec"] = np.floor((app["wall_time"] - t0).dt.total_seconds()).astype(int)
    out = (
        app.groupby("sec", as_index=False)
        .agg(
            feature_payload_bytes_1s=("feature_payload_bytes", "sum"),
            result_payload_bytes_1s=("result_payload_bytes_estimate", "sum"),
            frames=("frame_id", "count"),
            received=("result_received", "sum"),
        )
        .sort_values("sec")
    )
    out["app_offered_mbps"] = out["feature_payload_bytes_1s"] * 8.0 / 1e6
    out["app_result_mbps"] = out["result_payload_bytes_1s"] * 8.0 / 1e6
    return out


def trim_active(df: pd.DataFrame, x_col: str, cols: list[str], threshold: float = 1.0, pad_s: float = 3.0) -> pd.DataFrame:
    activity = pd.Series(False, index=df.index)
    for col in cols:
        if col in df.columns:
            activity = activity | (num(df[col]).fillna(0.0).abs() > threshold)
    if not activity.any():
        return df.copy()
    x = num(df[x_col])
    lo = float(x[activity].min()) - pad_s
    hi = float(x[activity].max()) + pad_s
    out = df[(x >= lo) & (x <= hi)].copy()
    out[x_col] = num(out[x_col]) - lo
    return out


def plot_grants(run_dir: Path) -> None:
    title, stem_key, prb_ceiling = run_label(run_dir)
    grants = pd.read_csv(run_dir / "nrue_ul_grant_windows_compact.csv")
    grants["t_norm"] = num(grants["t_norm"])
    grants = trim_active(grants, "t_norm", ["scheduled_mbps"], threshold=1.0, pad_s=0.0)
    grants["bin_s"] = (np.floor(num(grants["t_norm"]) / 10.0) * 10).astype(int)
    binned = (
        grants.groupby("bin_s", as_index=False)
        .agg(
            avg_rb_size=("avg_rb_size", "mean"),
            scheduled_mbps=("scheduled_mbps", "mean"),
            avg_mcs=("avg_mcs", "mean"),
        )
        .sort_values("bin_s")
    )

    x = binned["bin_s"]
    fig, axes = plt.subplots(3, 1, figsize=(11.2, 7.2), sharex=True, gridspec_kw={"hspace": 0.14})
    axes[0].plot(x, num(binned["avg_rb_size"]), color=COLORS["blue"], marker="o", markersize=2.8, linewidth=2.0)
    axes[0].axhline(prb_ceiling, color=COLORS["light_grey"], linestyle=":", linewidth=1.3)
    axes[0].set_ylim(0, prb_ceiling * 1.06)
    axes[0].set_ylabel("RB allocation\n10s bins")
    axes[0].grid(axis="y", alpha=0.22)

    axes[1].plot(x, num(binned["scheduled_mbps"]), color=COLORS["green"], marker="o", markersize=2.6, linewidth=1.9)
    axes[1].set_ylabel("Scheduled\nUL Mbps")
    axes[1].grid(axis="y", alpha=0.22)

    axes[2].plot(x, num(binned["avg_mcs"]), color=COLORS["orange"], marker="o", markersize=2.6, linewidth=1.9)
    axes[2].set_ylabel("Average MCS")
    axes[2].set_xlabel("Active CARLA/OAI time (s)")
    axes[2].grid(axis="y", alpha=0.22)

    summary = pd.read_csv(run_dir / "CARLA10_OAI_TTRACER_SUMMARY.csv").iloc[0]
    fig.suptitle(f"{title}: UL grants over corrected drivable-scene run", fontsize=13)
    fig.text(
        0.5,
        0.905,
        f"delivery={float(summary['delivery'])*100:.1f}%, RTT p50={float(summary['rtt_recv_ms_p50']):.1f} ms, "
        f"scheduled UL p50={float(summary['ul_sched_mbps_p50']):.1f} Mbps, "
        f"RB p50/window={float(summary['ul_prb_p50_window']):.1f}",
        ha="center",
        fontsize=8.7,
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "white", "edgecolor": "#DDDDDD", "alpha": 0.96},
    )
    save(fig, f"ttracer_ul_mcs_prb_timeseries_{stem_key}")


def plot_tunnel(run_dir: Path) -> None:
    run_group = run_dir.name
    metrics_path = find_frontend_metrics(run_group)
    net_path = run_dir / "network_timeseries.csv"
    summary_path = run_dir / "network_summary.csv"
    if metrics_path is None or not net_path.exists():
        return
    app_rate = frontend_rate_1s(metrics_path)
    net = pd.read_csv(net_path)
    net["wall_time"] = pd.to_datetime(net["wall_time_iso"], errors="coerce")
    net = net[net["wall_time"].notna()].copy()
    app_first_wall = pd.to_datetime(pd.read_csv(metrics_path, usecols=["wall_time_iso"])["wall_time_iso"]).min()
    net["sec"] = np.floor((net["wall_time"] - app_first_wall).dt.total_seconds()).astype(int)
    merged = pd.merge(
        app_rate[["sec", "app_offered_mbps", "app_result_mbps"]],
        net[["sec", "tx_bitrate_mbps", "rx_bitrate_mbps"]],
        on="sec",
        how="outer",
    ).sort_values("sec")
    merged = trim_active(merged, "sec", ["app_offered_mbps", "tx_bitrate_mbps"], threshold=1.0, pad_s=5.0)

    title, stem_key, _ = run_label(run_dir)
    fig, ax = plt.subplots(figsize=(11.0, 4.7))
    ax.plot(merged["sec"], rolling(merged["app_offered_mbps"], 7), color=COLORS["purple"], linewidth=2.2, label="App feature bytes offered, 7s median")
    ax.plot(merged["sec"], rolling(merged["tx_bitrate_mbps"], 7), color=COLORS["blue"], linewidth=2.2, label="UE tunnel TX/uplink, 7s median")
    ax.plot(merged["sec"], rolling(merged["rx_bitrate_mbps"], 7), color=COLORS["green"], linewidth=2.2, label="UE tunnel RX/downlink results, 7s median")
    if summary_path.exists():
        ns = pd.read_csv(summary_path).iloc[0]
        ax.text(
            0.012,
            0.06,
            f"oaitun_ue1 mean: TX={float(ns['avg_tx_mbps']):.2f} Mbps, RX={float(ns['avg_rx_mbps']):.3f} Mbps",
            transform=ax.transAxes,
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.32", "facecolor": "white", "edgecolor": "#DDDDDD", "alpha": 0.92},
        )
    ax.set_title(f"{title}: app offered load vs UE tunnel TX/RX")
    ax.set_xlabel("Time since first frontend frame (s)")
    ax.set_ylabel("Rate (Mbps)")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    save(fig, f"ttracer_tunnel_tx_rx_timeseries_{stem_key}")


def plot_phy_quality(run_dir: Path) -> None:
    ue_path = run_dir / "ue_phy_meas_compact.csv"
    gnb_path = run_dir / "gnb_pusch_power_compact.csv"
    if not ue_path.exists() and not gnb_path.exists():
        return
    title, stem_key, _ = run_label(run_dir)
    fig, axes = plt.subplots(3, 1, figsize=(11.2, 7.2), sharex=True, gridspec_kw={"hspace": 0.14})
    plotted = False
    if ue_path.exists():
        ue = pd.read_csv(ue_path)
        if "t_norm" in ue.columns and "rsrp" in ue.columns:
            vals = finite_phy(ue["rsrp"], "rsrp")
            if vals.notna().sum() > 10:
                axes[0].plot(num(ue.loc[vals.index, "t_norm"]), rolling(vals, 15), color=COLORS["blue"], linewidth=1.8, label="UE RSRP")
                plotted = True
        if "t_norm" in ue.columns and "snr" in ue.columns:
            vals = finite_phy(ue["snr"], "snr")
            if vals.notna().sum() > 10:
                axes[1].plot(num(ue.loc[vals.index, "t_norm"]), rolling(vals, 15), color=COLORS["green"], linewidth=1.8, label="UE SNR")
                plotted = True
        if "t_norm" in ue.columns and "w_cqi" in ue.columns:
            vals = finite_phy(ue["w_cqi"], "cqi")
            if vals.notna().sum() > 10:
                axes[2].plot(num(ue.loc[vals.index, "t_norm"]), rolling(vals, 15), color=COLORS["orange"], linewidth=1.8, label="UE wideband CQI")
                plotted = True
    if gnb_path.exists():
        gnb = pd.read_csv(gnb_path)
        if "t_norm" in gnb.columns and "snr_db" in gnb.columns:
            axes[1].plot(num(gnb["t_norm"]), rolling(gnb["snr_db"], 15), color=COLORS["red"], linewidth=1.5, alpha=0.85, label="gNB PUSCH SNR")
            plotted = True
    if not plotted:
        plt.close(fig)
        return
    axes[0].set_ylabel("RSRP")
    axes[1].set_ylabel("SNR / dB")
    axes[2].set_ylabel("CQI")
    axes[2].set_xlabel("T-tracer time (s)")
    for ax in axes:
        ax.grid(axis="y", alpha=0.22)
        ax.legend(frameon=False, loc="upper left")
    fig.suptitle(f"{title}: optional PHY quality traces over corrected run", fontsize=13)
    save(fig, f"ttracer_phy_quality_timeseries_{stem_key}")


def main() -> int:
    runs = discover_runs()
    if not runs:
        raise SystemExit(f"No corrected t-tracer artifacts found under {ARTIFACT_ROOT}")
    for run_dir in runs:
        plot_grants(run_dir)
        plot_tunnel(run_dir)
        # UE_PHY_MEAS in the current RFsim traces emits placeholder/sentinel
        # values for RSRP/SNR/CQI, and gNB PUSCH power-control traces were not
        # emitted. Keep the valid grant and tunnel plots; do not generate
        # misleading RF-quality plots from placeholder fields.
        # plot_phy_quality(run_dir)
    print(f"Wrote corrected t-tracer plots for {len(runs)} run(s) to {PLOTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
