#!/usr/bin/env python3
"""Clean presentation plots for CARLA/OAI T-tracer and queue-probe metrics.

The figures generated here are evidence plots for the OAI transport bottleneck
discussion.  They intentionally keep the conclusion modest: the traces show
uplink pressure, high PRB use, low/moderate MCS, and tiny downlink/result
traffic, but they are not a definitive layer-by-layer BSR/RLC proof.
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
PLOTS = ROOT / "plots" / "oai_ttracer"

TTRACER_DIR = (
    REPO
    / "metrics_logs"
    / "carla_oai_ttracer"
    / "downlink_oai_default_fps10_oai_default_20260720_10fps_ttracer"
)
QUEUEPROBE_DIR = (
    REPO
    / "metrics_logs"
    / "carla_oai_queueprobe"
    / "downlink_oai_default_fps10_oai_default_20260720_10fps_queueprobe"
)
NETWORK_DIR = (
    REPO
    / "metrics_logs"
    / "scenesense_network"
    / "downlink_oai_default_fps10_oai_default_20260720_10fps_ttracer"
)
APP_METRICS = (
    ROOT
    / "runs"
    / "oai_default"
    / "fps_10_oai_default_20260720_10fps_ttracer"
    / "streams"
    / "downlink_oai_default_fps10_oai_default_20260720_10fps_ttracer_metrics.csv"
)
LOOPBACK_APP_METRICS = (
    ROOT
    / "runs"
    / "ideal_loopback"
    / "fps_10_20260717_ideal_one_loop"
    / "streams"
    / "downlink_ideal_loopback_fps10_20260717_ideal_one_loop_metrics.csv"
)


COLORS = {
    "blue": "#4C78A8",
    "orange": "#F58518",
    "green": "#54A24B",
    "red": "#E45756",
    "purple": "#B279A2",
    "grey": "#666666",
    "light_grey": "#B8B8B8",
}


def save(fig: plt.Figure, stem: str) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(PLOTS / f"{stem}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def rolling(series: pd.Series, window: int = 7) -> pd.Series:
    return num(series).rolling(window=window, min_periods=1, center=True).median()


def trim_active_window(
    df: pd.DataFrame,
    x_col: str,
    activity_cols: list[str],
    threshold: float = 1.0,
    pad_s: float = 3.0,
) -> pd.DataFrame:
    activity = pd.Series(False, index=df.index)
    for col in activity_cols:
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


def find_compact_grant_csv(run_dir: Path) -> Path:
    candidates = [
        run_dir / "carla10_nrue_ul_grant_windows_compact.csv",
        run_dir / "nrue_ul_grant_windows_compact.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"no compact UE grant CSV found under {run_dir}")


def plot_ul_mcs_prb_timeseries_for(
    run_dir: Path,
    stem: str,
    title_prefix: str,
    prb_ceiling: int,
) -> None:
    grants = pd.read_csv(find_compact_grant_csv(run_dir))
    summary_path = run_dir / "CARLA10_OAI_TTRACER_SUMMARY.csv"
    summary = pd.read_csv(summary_path).iloc[0] if summary_path.exists() else None
    grants["t_norm"] = num(grants["t_norm"])
    grants = trim_active_window(
        grants,
        x_col="t_norm",
        activity_cols=["scheduled_mbps"],
        threshold=1.0,
        pad_s=0.0,
    )

    bin_s = 10
    grants["bin_s"] = (np.floor(num(grants["t_norm"]) / bin_s) * bin_s).astype(int)
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
    avg_prb = num(binned["avg_rb_size"])
    scheduled = num(binned["scheduled_mbps"])
    mcs = num(binned["avg_mcs"])

    fig, (ax_prb, ax_sched, ax_mcs) = plt.subplots(
        3,
        1,
        figsize=(11.2, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 0.9, 0.9], "hspace": 0.14},
    )

    ax_prb.plot(
        x,
        avg_prb,
        color=COLORS["blue"],
        linewidth=2.0,
        marker="o",
        markersize=2.8,
        alpha=0.95,
        label="Average RB allocation, 10s bins",
    )
    ax_prb.axhline(prb_ceiling, color=COLORS["light_grey"], linestyle=":", linewidth=1.4)
    ax_prb.text(
        x.max() * 0.995,
        prb_ceiling * 1.01,
        f"{prb_ceiling} RB config ceiling",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=COLORS["grey"],
    )

    ax_sched.plot(
        x,
        scheduled,
        color=COLORS["green"],
        linewidth=1.9,
        marker="o",
        markersize=2.6,
        alpha=0.95,
        label="Scheduled UL capacity from TBS grants, 10s bins",
    )

    ax_mcs.plot(
        x,
        mcs,
        color=COLORS["orange"],
        linewidth=1.9,
        marker="o",
        markersize=2.6,
        alpha=0.95,
        label="Average MCS, 10s bins",
    )

    full_avg_rb_p50 = float(num(grants["avg_rb_size"]).median())
    full_avg_rb_mean = float(num(grants["avg_rb_size"]).mean())
    ul_prb_p50 = float(summary["ul_prb_p50_window"]) if summary is not None else full_avg_rb_p50
    avg_mcs = float(summary["ul_avg_mcs_p50_window"]) if summary is not None else float(num(grants["avg_mcs"]).median())
    scheduled_mbps = (
        float(summary["ul_sched_mbps_p50"])
        if summary is not None
        else float(num(grants["scheduled_mbps"]).median())
    )
    key_text = (
        f"summary: avg-RB p50={full_avg_rb_p50:.1f}, avg-RB mean={full_avg_rb_mean:.1f}; "
        f"1s median-RB p50={ul_prb_p50:.0f}; "
        f"avg MCS={avg_mcs:.1f}, "
        f"scheduled={scheduled_mbps:.1f} Mbps"
    )
    fig.text(
        0.5,
        0.902,
        key_text,
        ha="center",
        fontsize=8.7,
        color="#333333",
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "white", "edgecolor": "#DDDDDD", "alpha": 0.96},
    )

    fig.suptitle(
        f"{title_prefix}: UL grants show PRB pressure and MCS over active feature-stream time",
        y=0.97,
        fontsize=13,
    )
    ax_prb.set_ylabel("RB allocation\n10s bins")
    ax_sched.set_ylabel("Scheduled\nUL Mbps")
    ax_mcs.set_ylabel("Average MCS")
    ax_mcs.set_xlabel("Active CARLA/OAI time (s)")
    ax_prb.set_ylim(0, prb_ceiling * 1.06)
    ax_sched.set_ylim(0, max(30, float(np.nanpercentile(scheduled, 99)) * 1.08))
    ax_mcs.set_ylim(0, 16)
    ax_prb.grid(axis="y", alpha=0.22)
    ax_sched.grid(axis="y", alpha=0.22)
    ax_mcs.grid(axis="y", alpha=0.22)
    ax_prb.legend(frameon=False, loc="upper left")
    ax_sched.legend(frameon=False, loc="upper left")
    ax_mcs.legend(frameon=False, loc="upper left")
    fig.text(
        0.5,
        -0.02,
        "Source: UE/gNB T-tracer grant windows. Plot uses 10-second bins computed from the extracted 1-second windows.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    save(fig, stem)


def latest_bw273_ttracer_dir() -> Path | None:
    root = REPO / "metrics_logs" / "carla_oai_ttracer"
    candidates = sorted(
        p
        for p in root.glob("downlink_oai_bw273_mu1_ttracer_fps10_*")
        if p.is_dir()
        and (p / "VALIDATED_273PRB_TTRACER.ok").exists()
        and any((p / name).exists() for name in ("nrue_ul_grant_windows_compact.csv", "carla10_nrue_ul_grant_windows_compact.csv"))
    )
    return candidates[-1] if candidates else None


def plot_ul_mcs_prb_timeseries() -> None:
    plot_ul_mcs_prb_timeseries_for(
        TTRACER_DIR,
        stem="ttracer_ul_mcs_prb_timeseries",
        title_prefix="Default OAI 10 FPS / 106PRB",
        prb_ceiling=106,
    )
    bw273_dir = latest_bw273_ttracer_dir()
    if bw273_dir is not None:
        plot_ul_mcs_prb_timeseries_for(
            bw273_dir,
            stem="ttracer_ul_mcs_prb_timeseries_bw273",
            title_prefix="Validated wider-bandwidth OAI 10 FPS / 273PRB",
            prb_ceiling=273,
        )


def frontend_rate_1s(metrics_path: Path) -> pd.DataFrame:
    app = pd.read_csv(metrics_path)
    app["wall_time"] = pd.to_datetime(app["wall_time_iso"], errors="coerce")
    app["feature_payload_bytes"] = num(app["feature_payload_bytes"]).fillna(0.0)
    app["result_payload_bytes_estimate"] = num(app["result_payload_bytes_estimate"]).fillna(0.0)
    app = app[app["wall_time"].notna()].copy()
    t0 = app["wall_time"].min()
    app["sec"] = np.floor((app["wall_time"] - t0).dt.total_seconds()).astype(int)
    offered = (
        app.groupby("sec", as_index=False)
        .agg(
            feature_payload_bytes_1s=("feature_payload_bytes", "sum"),
            result_payload_bytes_1s=("result_payload_bytes_estimate", "sum"),
            frames=("frame_id", "count"),
            received=("result_received", "sum"),
        )
    )
    offered["app_offered_mbps"] = offered["feature_payload_bytes_1s"] * 8.0 / 1e6
    offered["app_result_mbps"] = offered["result_payload_bytes_1s"] * 8.0 / 1e6
    return offered


def app_offered_rate_1s() -> pd.DataFrame:
    return frontend_rate_1s(APP_METRICS)


def frontend_run_summary(metrics_path: Path) -> dict[str, float]:
    app = pd.read_csv(metrics_path)
    wall = pd.to_datetime(app["wall_time_iso"], errors="coerce")
    elapsed = float((wall.max() - wall.min()).total_seconds())
    frames = float(len(app))
    payload_mbit = float(num(app["feature_payload_bytes"]).median() * 8.0 / 1e6)
    result_mbit = float(num(app["result_payload_bytes_estimate"]).median() * 8.0 / 1e6)
    delivery = float(num(app["result_received"]).mean())
    return {
        "elapsed_s": elapsed,
        "frames": frames,
        "actual_fps": frames / elapsed if elapsed > 0 else float("nan"),
        "payload_mbit": payload_mbit,
        "result_mbit": result_mbit,
        "delivery": delivery,
        "mean_feature_mbps": float(num(app["feature_payload_bytes"]).sum() * 8.0 / elapsed / 1e6),
    }


def plot_tunnel_tx_rx_timeseries() -> None:
    net = pd.read_csv(NETWORK_DIR / "network_timeseries.csv")
    net_summary = pd.read_csv(NETWORK_DIR / "network_summary.csv").iloc[0]
    app_rate = app_offered_rate_1s()

    net["wall_time"] = pd.to_datetime(net["wall_time_iso"], errors="coerce")
    net = net[net["wall_time"].notna()].copy()
    app_first_wall = pd.to_datetime(pd.read_csv(APP_METRICS, usecols=["wall_time_iso"])["wall_time_iso"]).min()
    net["sec"] = np.floor((net["wall_time"] - app_first_wall).dt.total_seconds()).astype(int)

    merged = pd.merge(
        app_rate[["sec", "app_offered_mbps"]],
        net[["sec", "tx_bitrate_mbps", "rx_bitrate_mbps"]],
        on="sec",
        how="outer",
    ).sort_values("sec")
    merged = trim_active_window(
        merged,
        x_col="sec",
        activity_cols=["app_offered_mbps", "tx_bitrate_mbps"],
        threshold=1.0,
        pad_s=5.0,
    )
    x = merged["sec"]

    fig, ax = plt.subplots(figsize=(11.0, 4.7))
    ax.plot(
        x,
        rolling(merged["app_offered_mbps"], 7),
        color=COLORS["purple"],
        linewidth=2.2,
        label="App feature bytes offered, 7s median",
    )
    ax.plot(
        x,
        rolling(merged["tx_bitrate_mbps"], 7),
        color=COLORS["blue"],
        linewidth=2.2,
        label="UE tunnel TX, 7s median",
    )
    ax.plot(
        x,
        rolling(merged["rx_bitrate_mbps"], 7),
        color=COLORS["green"],
        linewidth=2.2,
        label="UE tunnel RX/downlink results, 7s median",
    )

    ax.text(
        0.012,
        0.06,
        f"oaitun_ue1 mean: TX={float(net_summary['avg_tx_mbps']):.2f} Mbps, "
        f"RX={float(net_summary['avg_rx_mbps']):.3f} Mbps",
        transform=ax.transAxes,
        fontsize=9,
        color="#333333",
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "white", "edgecolor": "#DDDDDD", "alpha": 0.92},
    )

    ax.set_title("Default OAI 10 FPS: tunnel TX carries the heavy feature stream; RX is tiny")
    ax.set_xlabel("Time since first CARLA frontend frame (s)")
    ax.set_ylabel("Rate (Mbps)")
    ax.set_ylim(0, max(35, float(np.nanpercentile(num(merged["app_offered_mbps"]), 99)) * 1.15))
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.text(
        0.5,
        -0.02,
        "Source: frontend feature payload timestamps + oaitun_ue1 network sampler. RX is result/downlink traffic, not camera features.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    save(fig, "ttracer_tunnel_tx_rx_timeseries")


def plot_loopback_payload_tx_rx_timeseries() -> None:
    rates = frontend_rate_1s(LOOPBACK_APP_METRICS)
    summary = frontend_run_summary(LOOPBACK_APP_METRICS)
    max_sec = int(rates["sec"].max())
    rates = rates.set_index("sec").reindex(range(max_sec + 1), fill_value=0.0).reset_index()
    x = rates["sec"]

    target_10fps_mbps = summary["payload_mbit"] * 10.0

    fig, ax = plt.subplots(figsize=(11.0, 4.7))
    ax.plot(
        x,
        rolling(rates["app_offered_mbps"], 7),
        color=COLORS["purple"],
        linewidth=2.2,
        label="Frontend feature payload TX, 7s median",
    )
    ax.plot(
        x,
        rolling(rates["app_result_mbps"], 7),
        color=COLORS["green"],
        linewidth=2.2,
        label="Frontend result payload RX, 7s median",
    )
    ax.axhline(
        target_10fps_mbps,
        color=COLORS["grey"],
        linestyle=":",
        linewidth=1.5,
        label=f"10 wall-FPS feature rate ≈ {target_10fps_mbps:.1f} Mbps",
    )

    ax.text(
        0.012,
        0.07,
        f"actual wall-clock rate={summary['actual_fps']:.2f} FPS; "
        f"mean feature TX={summary['mean_feature_mbps']:.1f} Mbps; delivery={summary['delivery']*100:.0f}%",
        transform=ax.transAxes,
        fontsize=9,
        color="#333333",
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "white", "edgecolor": "#DDDDDD", "alpha": 0.92},
    )
    ax.set_title("Ideal loopback 10 FPS: app-derived feature TX and result RX rates")
    ax.set_xlabel("Time since first CARLA frontend frame (s)")
    ax.set_ylabel("Rate (Mbps)")
    ax.set_ylim(0, max(target_10fps_mbps * 1.12, float(np.nanpercentile(num(rates["app_offered_mbps"]), 99)) * 1.2))
    ax.grid(axis="y", alpha=0.22)
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#DDDDDD",
        framealpha=0.88,
        ncol=2,
        loc="upper left",
    )
    fig.text(
        0.5,
        -0.02,
        "No loopback interface sampler was found for this run, so this uses frontend payload timestamps: feature bytes sent and result bytes received.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    save(fig, "loopback_app_payload_tx_rx_timeseries")


def plot_queueprobe_layer_rates() -> None:
    rates = pd.read_csv(QUEUEPROBE_DIR / "QUEUEPROBE_LAYER_RATES_1S.csv")
    rates["sec"] = num(rates["sec"])
    rates = trim_active_window(
        rates,
        x_col="sec",
        activity_cols=["offered_mbps", "scheduled_mbps", "phy_ul_mbps", "lcid_ul_mbps"],
        threshold=1.0,
        pad_s=4.0,
    )
    x = rates["sec"]

    fig, ax = plt.subplots(figsize=(11.0, 4.7))
    series = [
        ("offered_mbps", "App offered feature load", COLORS["purple"], 2.4),
        ("scheduled_mbps", "UE scheduled UL capacity", COLORS["orange"], 2.1),
        ("phy_ul_mbps", "gNB PHY UL payload RX", COLORS["blue"], 2.1),
        ("lcid_ul_mbps", "gNB MAC LCID UL payload", COLORS["green"], 1.8),
    ]
    for col, label, color, width in series:
        if col not in rates.columns:
            continue
        ax.plot(
            x,
            rolling(rates[col], 11),
            linewidth=width,
            color=color,
            label=f"{label}, 11s median",
        )

    ax.set_title("Queue-probe run: app load vs OAI UL drain-rate traces")
    ax.set_xlabel("Active queue-probe time (s)")
    ax.set_ylabel("Rate (Mbps)")
    ax.set_ylim(0, max(30, float(np.nanpercentile(num(rates["offered_mbps"]), 99)) * 1.18))
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.text(
        0.5,
        -0.02,
        "Source: 300-frame CARLA/OAI queue-probe with expanded OAI layer counters. Use as diagnostic evidence, not final deployment headline.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    save(fig, "queueprobe_app_vs_ran_rate_timeseries")


def plot_carla_vs_iperf_radio_summary() -> None:
    comparison = pd.read_csv(TTRACER_DIR / "CARLA10_VS_IPERF_LAYER_COMPARISON.csv")
    wanted = {
        "UL PRB p50/window": "PRB p50",
        "UL avg MCS p50/window": "Avg MCS p50",
        "offered/app Mbps p50 1s": "Offered Mbps p50",
        "UL scheduled p50/window": "UL scheduled Mbps p50",
    }
    rows = comparison[comparison["metric"].isin(wanted.keys())].copy()
    rows["label"] = rows["metric"].map(wanted)
    labels = rows["label"].tolist()
    configs = [
        ("constant_iperf", "small-datagram iperf", COLORS["green"]),
        ("largeblock_iperf", "60KB-block iperf", COLORS["orange"]),
        ("carla_10fps", "CARLA 10 FPS", COLORS["blue"]),
    ]

    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for i, (col, label, color) in enumerate(configs):
        vals = num(rows[col]).to_numpy()
        ax.bar(x + (i - 1) * width, vals, width=width, label=label, color=color)

    ax.set_xticks(x, labels)
    ax.set_title("Radio-side comparison: CARLA behaves more like large-block traffic than small datagrams")
    ax.set_ylabel("Metric value")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, loc="upper left")
    fig.text(
        0.5,
        -0.02,
        "Source: T-tracer comparison table. Mixed units by metric; use as a qualitative diagnostic summary.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    save(fig, "ttracer_carla_vs_iperf_radio_summary")


def main() -> None:
    plot_ul_mcs_prb_timeseries()
    plot_tunnel_tx_rx_timeseries()
    plot_loopback_payload_tx_rx_timeseries()
    plot_queueprobe_layer_rates()
    plot_carla_vs_iperf_radio_summary()
    print(f"Wrote plots to {PLOTS}")


if __name__ == "__main__":
    main()
