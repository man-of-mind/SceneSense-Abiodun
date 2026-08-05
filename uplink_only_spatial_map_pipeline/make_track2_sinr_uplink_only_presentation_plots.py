#!/usr/bin/env python3
"""Presentation plots for the Track-2 SINR uplink-only OAI ladder.

The batch is the uplink-only spatial-map path using 106PRB OAI, no-AE/ROI0,
zstd, 10FPS target, and SINR-driven MCS selection.  Plots are aligned to the
first actual feature-send event, not the CARLA process start line, so the
time-series show the real application traffic window.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "uplink_only_spatial_map_pipeline" / "results"
OUT_DIR = RESULTS_DIR / "presentation_sinr_uplink_only"
BIN_S = 1.0


@dataclass(frozen=True)
class RunSpec:
    label: str
    display: str
    short: str
    color: str


RUNS: List[RunSpec] = [
    RunSpec("clear_sinr", "Clean\n50.3 dB", "Clean", "#009E73"),
    RunSpec("mild_sinr", "Mild\n19.5 dB", "Mild", "#0072B2"),
    RunSpec("mid15_sinr", "Mid\n15.6 dB", "Mid", "#E69F00"),
    RunSpec("strong_sinr", "Strong\n8.2 dB", "Strong", "#D55E00"),
]


def q(series: pd.Series, p: float) -> float:
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return float("nan")
    return float(vals.quantile(p))


def safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def hms_to_seconds(value: str) -> float:
    hour, minute, second = value.strip().split(":")
    return int(hour) * 3600.0 + int(minute) * 60.0 + float(second)


def wall_seconds(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    return (
        dt.dt.hour * 3600.0
        + dt.dt.minute * 60.0
        + dt.dt.second
        + dt.dt.microsecond / 1_000_000.0
    )


def trace_seconds(series: pd.Series, t0_s: float) -> pd.Series:
    vals = series.astype(str).map(hms_to_seconds).astype(float)
    vals = vals.where(vals >= t0_s - 12.0 * 3600.0, vals + 24.0 * 3600.0)
    return vals - t0_s


def summary_path(base_batch: str) -> Path:
    return RESULTS_DIR / f"track2_sinr_uplink_only_ladder_{base_batch}.csv"


def find_send_events(run_group: str) -> Path:
    candidates = sorted(
        (ROOT / "uplink_only_spatial_map_pipeline" / "runs").glob(
            f"*/fps_10_*/front_metrics/streams/{run_group}_queue_probe_send_events.csv"
        )
    )
    if not candidates:
        raise FileNotFoundError(f"send events not found for {run_group}")
    return candidates[-1]


def find_edge_metrics(run_group: str) -> Path:
    candidates = sorted(
        (ROOT / "uplink_only_spatial_map_pipeline" / "runs").glob(
            f"*/fps_10_*/edge_uplink_metrics.csv"
        )
    )
    matches = []
    for path in candidates:
        try:
            head = pd.read_csv(path, nrows=1)
        except Exception:
            continue
        if not head.empty and str(head.get("stream_id", pd.Series([""])).iloc[0]) == run_group:
            matches.append(path)
    if not matches:
        raise FileNotFoundError(f"edge metrics not found for {run_group}")
    return matches[-1]


def ttracer_dir(run_group: str) -> Path:
    path = ROOT / "metrics_logs" / "scenesense_ttracer" / run_group
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def active_bins(duration_s: float, extra_s: float = 30.0) -> pd.DataFrame:
    n = max(1, int(math.ceil(duration_s + extra_s)))
    bins = pd.DataFrame({"bin": np.arange(n, dtype=int)})
    bins["t"] = bins["bin"].astype(float) * BIN_S
    return bins


def load_context(run_group: str) -> Dict[str, object]:
    send_path = find_send_events(run_group)
    send = pd.read_csv(send_path)
    if send.empty:
        raise ValueError(f"empty send events: {send_path}")
    wall_abs = wall_seconds(send["wall_time_iso"])
    t0_s = float(wall_abs.iloc[0])
    perf = pd.to_numeric(send["camera_sent_perf"], errors="coerce")
    t0_perf = float(perf.dropna().iloc[0]) if not perf.dropna().empty else float("nan")
    send = send.copy()
    if math.isfinite(t0_perf):
        send["_t"] = pd.to_numeric(send["camera_sent_perf"], errors="coerce") - t0_perf
    else:
        send["_t"] = wall_abs.where(wall_abs >= t0_s - 12.0 * 3600.0, wall_abs + 24.0 * 3600.0) - t0_s
    duration_s = max(float(send["_t"].iloc[-1]), 1e-9)
    return {
        "send_path": send_path,
        "send": send,
        "t0_s": t0_s,
        "t0_perf": t0_perf,
        "send_duration_s": duration_s,
    }


def load_app_bins(ctx: Dict[str, object], extra_s: float = 30.0) -> pd.DataFrame:
    send = ctx["send"].copy()
    duration_s = float(ctx["send_duration_s"])
    bins = active_bins(duration_s, extra_s=extra_s)
    send["bin"] = np.floor(send["_t"] / BIN_S).astype(int)
    send["payload"] = pd.to_numeric(send["feature_payload_bytes"], errors="coerce").fillna(0.0)
    agg = send.groupby("bin", as_index=False).agg(
        app_bytes=("payload", "sum"),
        app_frames=("payload", "size"),
        send_lag_ms=("send_lag_ms", "median"),
    )
    out = bins.merge(agg, on="bin", how="left")
    for col in ["app_bytes", "app_frames"]:
        out[col] = out[col].fillna(0.0)
    out["app_mbps"] = out["app_bytes"] * 8.0 / BIN_S / 1_000_000.0
    return out


def load_edge_bins(run_group: str, ctx: Dict[str, object], extra_s: float = 30.0) -> pd.DataFrame:
    path = find_edge_metrics(run_group)
    edge = safe_read_csv(path)
    duration_s = float(ctx["send_duration_s"])
    bins = active_bins(duration_s, extra_s=extra_s)
    if edge.empty:
        return bins.assign(edge_frames=0.0, edge_fps=0.0)
    edge = edge.copy()
    if "t_edge_recv_perf" in edge and math.isfinite(float(ctx.get("t0_perf", float("nan")))):
        edge["_t"] = pd.to_numeric(edge["t_edge_recv_perf"], errors="coerce") - float(ctx["t0_perf"])
    else:
        edge["_t"] = wall_seconds(edge["wall_time_iso"]) - float(ctx["t0_s"])
    edge = edge[(edge["_t"] >= 0.0) & (edge["_t"] <= duration_s + extra_s)]
    if edge.empty:
        return bins.assign(edge_frames=0.0, edge_fps=0.0)
    edge["bin"] = np.floor(edge["_t"] / BIN_S).astype(int)
    agg = edge.groupby("bin", as_index=False).agg(edge_frames=("frame_id", "size"))
    out = bins.merge(agg, on="bin", how="left")
    out["edge_frames"] = out["edge_frames"].fillna(0.0)
    out["edge_fps"] = out["edge_frames"] / BIN_S
    return out


def load_grant_bins(run_group: str, ctx: Dict[str, object], extra_s: float = 30.0) -> pd.DataFrame:
    path = ttracer_dir(run_group) / "ue" / "csv" / "NRUE_MAC_DCI_GRANT.csv"
    usecols = lambda c: c in {"time", "direction", "tbs", "mcs", "rb_size", "rv", "round"}
    df = safe_read_csv(path, usecols=usecols)
    duration_s = float(ctx["send_duration_s"])
    bins = active_bins(duration_s, extra_s=extra_s)
    if df.empty:
        return bins.assign(
            scheduled_mbps=0.0,
            first_tx_mbps=0.0,
            retx_mbps=0.0,
            grant_rate_hz=0.0,
            retx_rate_pct=0.0,
            mcs_p50=np.nan,
            mcs_avg=np.nan,
            tbs_avg_bytes=np.nan,
        )
    for col in ["direction", "tbs", "mcs", "rb_size", "rv", "round"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["_t"] = trace_seconds(df["time"], float(ctx["t0_s"]))
    ul = df[(df["direction"] == 1) & (df["_t"] >= 0.0) & (df["_t"] <= duration_s + extra_s)].copy()
    if ul.empty:
        return bins.assign(
            scheduled_mbps=0.0,
            first_tx_mbps=0.0,
            retx_mbps=0.0,
            grant_rate_hz=0.0,
            retx_rate_pct=0.0,
            mcs_p50=np.nan,
            mcs_avg=np.nan,
            tbs_avg_bytes=np.nan,
        )
    ul["bin"] = np.floor(ul["_t"] / BIN_S).astype(int)
    ul["tbs"] = pd.to_numeric(ul["tbs"], errors="coerce").fillna(0.0)
    rv = pd.to_numeric(ul.get("rv", 0), errors="coerce").fillna(0.0)
    harq_round = pd.to_numeric(ul.get("round", 0), errors="coerce").fillna(0.0)
    ul["is_retx"] = (rv > 0) | (harq_round > 0)
    ul["first_tbs"] = ul["tbs"].where(~ul["is_retx"], 0.0)
    ul["retx_tbs"] = ul["tbs"].where(ul["is_retx"], 0.0)
    agg = ul.groupby("bin", as_index=False).agg(
        grants=("tbs", "size"),
        total_tbs=("tbs", "sum"),
        first_tbs=("first_tbs", "sum"),
        retx_tbs=("retx_tbs", "sum"),
        retx_rate=("is_retx", "mean"),
        mcs_p50=("mcs", "median"),
        mcs_avg=("mcs", "mean"),
        tbs_avg_bytes=("tbs", "mean"),
        rb_p50=("rb_size", "median"),
    )
    out = bins.merge(agg, on="bin", how="left")
    for col in ["grants", "total_tbs", "first_tbs", "retx_tbs", "retx_rate"]:
        out[col] = out[col].fillna(0.0)
    out["scheduled_mbps"] = out["total_tbs"] * 8.0 / BIN_S / 1_000_000.0
    out["first_tx_mbps"] = out["first_tbs"] * 8.0 / BIN_S / 1_000_000.0
    out["retx_mbps"] = out["retx_tbs"] * 8.0 / BIN_S / 1_000_000.0
    out["grant_rate_hz"] = out["grants"] / BIN_S
    out["retx_rate_pct"] = out["retx_rate"] * 100.0
    return out


def load_snr_bins(run_group: str, ctx: Dict[str, object], extra_s: float = 30.0) -> pd.DataFrame:
    path = ttracer_dir(run_group) / "gnb" / "csv" / "GNB_MAC_UL_MCS_DECISION.csv"
    df = safe_read_csv(path, usecols=lambda c: c in {"time", "avg_snr_x10"})
    duration_s = float(ctx["send_duration_s"])
    bins = active_bins(duration_s, extra_s=extra_s)
    if df.empty:
        return bins.assign(snr_db=np.nan)
    df["_t"] = trace_seconds(df["time"], float(ctx["t0_s"]))
    df["snr_db"] = pd.to_numeric(df["avg_snr_x10"], errors="coerce") / 10.0
    df = df[(df["_t"] >= 0.0) & (df["_t"] <= duration_s + extra_s)].copy()
    if df.empty:
        return bins.assign(snr_db=np.nan)
    df["bin"] = np.floor(df["_t"] / BIN_S).astype(int)
    agg = df.groupby("bin", as_index=False).agg(snr_db=("snr_db", "median"))
    return bins.merge(agg, on="bin", how="left")


def load_queue_bins(run_group: str, ctx: Dict[str, object], extra_s: float = 30.0) -> pd.DataFrame:
    base = ttracer_dir(run_group) / "ue" / "csv"
    duration_s = float(ctx["send_duration_s"])
    bins = active_bins(duration_s, extra_s=extra_s)

    rlc = safe_read_csv(
        base / "NRUE_MAC_RLC_BUFFER_STATUS.csv",
        usecols=lambda c: c in {"time", "lcid", "bytes_in_buffer"},
    )
    if not rlc.empty:
        rlc["lcid"] = pd.to_numeric(rlc["lcid"], errors="coerce")
        rlc["bytes_in_buffer"] = pd.to_numeric(rlc["bytes_in_buffer"], errors="coerce").fillna(0.0)
        rlc["_t"] = trace_seconds(rlc["time"], float(ctx["t0_s"]))
        rlc = rlc[(rlc["lcid"] == 4) & (rlc["_t"] >= 0.0) & (rlc["_t"] <= duration_s + extra_s)].copy()
        if not rlc.empty:
            rlc["bin"] = np.floor(rlc["_t"] / BIN_S).astype(int)
            agg = rlc.groupby("bin", as_index=False).agg(
                rlc_p50_mib=("bytes_in_buffer", lambda s: q(s, 0.50) / 1024.0 / 1024.0),
                rlc_p95_mib=("bytes_in_buffer", lambda s: q(s, 0.95) / 1024.0 / 1024.0),
            )
            bins = bins.merge(agg, on="bin", how="left")

    bsr = safe_read_csv(
        base / "NRUE_MAC_BSR_STATUS.csv",
        usecols=lambda c: c in {"time", "lcg1_bytes", "sdu_bytes"},
    )
    if not bsr.empty:
        bsr["lcg1_bytes"] = pd.to_numeric(bsr["lcg1_bytes"], errors="coerce").fillna(0.0)
        bsr["sdu_bytes"] = pd.to_numeric(bsr["sdu_bytes"], errors="coerce").fillna(0.0)
        bsr["_t"] = trace_seconds(bsr["time"], float(ctx["t0_s"]))
        bsr = bsr[(bsr["_t"] >= 0.0) & (bsr["_t"] <= duration_s + extra_s)].copy()
        if not bsr.empty:
            bsr["bin"] = np.floor(bsr["_t"] / BIN_S).astype(int)
            agg = bsr.groupby("bin", as_index=False).agg(
                bsr_lcg1_p50_mib=("lcg1_bytes", lambda s: q(s, 0.50) / 1024.0 / 1024.0),
                bsr_lcg1_p95_mib=("lcg1_bytes", lambda s: q(s, 0.95) / 1024.0 / 1024.0),
                rlc_sdu_drain_bytes=("sdu_bytes", "sum"),
            )
            bins = bins.merge(agg, on="bin", how="left")

    for col in ["rlc_p50_mib", "rlc_p95_mib", "bsr_lcg1_p50_mib", "bsr_lcg1_p95_mib", "rlc_sdu_drain_bytes"]:
        if col not in bins:
            bins[col] = 0.0
        bins[col] = bins[col].fillna(0.0)
    bins["rlc_sdu_drain_mbps"] = bins["rlc_sdu_drain_bytes"] * 8.0 / BIN_S / 1_000_000.0
    return bins


def p50_edge_component(run_group: str, col: str) -> float:
    path = find_edge_metrics(run_group)
    df = safe_read_csv(path, usecols=lambda c: c in {col})
    if df.empty or col not in df:
        return float("nan")
    return q(df[col], 0.50)


def active_summary(row: pd.Series, spec: RunSpec, ctx: Dict[str, object]) -> Dict[str, object]:
    duration_s = float(ctx["send_duration_s"])
    run_group = str(row["run_group"])
    app = load_app_bins(ctx, extra_s=0.0)
    grants = load_grant_bins(run_group, ctx, extra_s=0.0)
    queue = load_queue_bins(run_group, ctx, extra_s=0.0)
    snr = load_snr_bins(run_group, ctx, extra_s=0.0)
    edge = load_edge_bins(run_group, ctx, extra_s=30.0)

    send = ctx["send"]
    payload = pd.to_numeric(send["feature_payload_bytes"], errors="coerce").fillna(0.0)
    out: Dict[str, object] = {
        "label": spec.label,
        "display": spec.display.replace("\n", " "),
        "run_group": run_group,
        "send_frames": int(len(send)),
        "send_duration_s": duration_s,
        "send_fps": float(len(send) / duration_s) if duration_s > 0 else float("nan"),
        "app_offered_mbps_send": float(payload.sum() * 8.0 / duration_s / 1_000_000.0),
        "edge_frames_30s_drain": int(edge["edge_frames"].sum()) if "edge_frames" in edge else int(row.get("edge_frames", 0)),
        "edge_delivery_pct_30s_drain": float(row.get("edge_delivery_pct", float("nan"))),
        "snr_p50_db_active": q(snr["snr_db"], 0.50),
        "mcs_p50_active": q(grants["mcs_p50"], 0.50),
        "mcs_p95_active": q(grants["mcs_p50"], 0.95),
        "scheduled_mbps_send": float(grants["scheduled_mbps"].mean()),
        "first_tx_mbps_send": float(grants["first_tx_mbps"].mean()),
        "grant_rate_hz_send": float(grants["grant_rate_hz"].mean()),
        "retx_rate_pct_send": float(grants["retx_rate_pct"].mean()),
        "tbs_avg_kib_send": q(grants["tbs_avg_bytes"], 0.50) / 1024.0,
        "bsr_lcg1_p95_mib_send": q(queue["bsr_lcg1_p95_mib"], 0.95),
        "rlc_p95_mib_send": q(queue["rlc_p95_mib"], 0.95),
        "rlc_drain_mbps_send": float(queue["rlc_sdu_drain_mbps"].mean()),
        "map_publish_p50_ms": p50_edge_component(run_group, "edge_to_map_publish_ms"),
        "spatial_publisher_dropped_max": float(row.get("spatial_publisher_dropped_max", float("nan"))),
        "udp_partial_messages_dropped_max": float(row.get("udp_partial_messages_dropped_max", float("nan"))),
    }
    for col in [
        "payload_p50_kib",
        "front_build_p50_ms",
        "front_to_edge_p50_ms",
        "front_to_edge_p95_ms",
        "tail_p50_ms",
        "backbone_to_tail_p50_ms",
        "capture_to_map_publish_p50_ms",
        "pdcp_to_gnb_p50_ms",
        "pdcp_to_gnb_p95_ms",
    ]:
        out[col] = float(row.get(col, float("nan")))
    return out


def fmt(value: object) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(f):
        return "N/A"
    return f"{f:,.3f}"


def fmt_bar_label(value: float) -> str:
    if not math.isfinite(value):
        return ""
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:.0f}"
    return f"{value:.1f}"


def to_markdown(df: pd.DataFrame) -> str:
    lines = [
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join("---" for _ in df.columns) + " |",
    ]
    for _, row in df.iterrows():
        vals = [fmt(row[col]) if pd.api.types.is_numeric_dtype(df[col]) else str(row[col]) for col in df.columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", alpha=0.24, linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10, width=1.2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def savefig(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight", dpi=280)
    plt.close(fig)


def plot_summary_bars(df: pd.DataFrame) -> None:
    metrics = [
        ("snr_p50_db_active", "Measured SNR", "dB"),
        ("mcs_p50_active", "MCS", "p50"),
        ("app_offered_mbps_send", "Observed app send rate", "Mbps"),
        ("scheduled_mbps_send", "Scheduled UL", "Mbps"),
        ("front_to_edge_p50_ms", "Front→edge latency", "p50 ms"),
        ("capture_to_map_publish_p50_ms", "Capture→map publish", "p50 ms"),
        ("edge_delivery_pct_30s_drain", "Complete tensors at edge", "% sent"),
        ("bsr_lcg1_p95_mib_send", "UE BSR backlog", "p95 MiB"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16.2, 7.9))
    x = np.arange(len(RUNS))
    colors = [spec.color for spec in RUNS]
    labels = [spec.display for spec in RUNS]
    for ax, (col, title, ylabel) in zip(axes.flat, metrics):
        vals = [float(df[df["label"].eq(spec.label)][col].iloc[0]) for spec in RUNS]
        bars = ax.bar(x, vals, color=colors, width=0.68)
        for bar, val in zip(bars, vals):
            if math.isfinite(val):
                txt = fmt_bar_label(val)
                ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), txt, ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
        ax.set_ylabel(ylabel, fontsize=10.5, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9.5, fontweight="bold")
        style_axes(ax)
    fig.tight_layout()
    savefig(fig, "uplink_only_sinr_ladder_summary_bars")


def plot_latency_breakdown(df: pd.DataFrame) -> None:
    rows = df.copy()
    rows["y_label"] = rows["display"].str.replace(" ", "\n", n=1)
    cols = [
        ("front_build_p50_ms", "front build", "#9E9E9E"),
        ("front_to_edge_p50_ms", "front→edge", "#56B4E9"),
        ("tail_p50_ms", "edge tail", "#CC79A7"),
        ("map_publish_p50_ms", "map publish", "#F0E442"),
    ]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(13.0, 5.8))
    left = np.zeros(len(rows))
    for col, label, color in cols:
        vals = pd.to_numeric(rows[col], errors="coerce").fillna(0.0).to_numpy()
        ax.barh(y, vals, left=left, height=0.62, color=color, label=label)
        for idx, val in enumerate(vals):
            if val >= 300.0:
                ax.text(left[idx] + val / 2.0, idx, f"{val:,.0f}", ha="center", va="center", fontsize=8.5, fontweight="bold")
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(rows["y_label"], fontsize=10, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("p50 latency component (ms)", fontsize=12, fontweight="bold")
    ax.set_title("Uplink-only SINR ladder: latency breakdown", fontsize=15, fontweight="bold", pad=10)
    ax.legend(loc="lower right", frameon=False, ncol=4, prop={"weight": "bold", "size": 10})
    style_axes(ax)
    savefig(fig, "uplink_only_sinr_ladder_latency_breakdown")


def legend_handles() -> List[Line2D]:
    return [Line2D([0], [0], color=spec.color, linewidth=2.8, label=spec.short) for spec in RUNS]


def plot_radio_rate_timeseries(summary_df: pd.DataFrame, contexts: Dict[str, Dict[str, object]], max_t: float = 180.0) -> None:
    fig, axes = plt.subplots(6, 1, figsize=(13.2, 13.6), sharex=True)
    for spec in RUNS:
        row = summary_df[summary_df["label"].eq(spec.label)].iloc[0]
        rg = str(row["run_group"])
        ctx = contexts[spec.label]
        app = load_app_bins(ctx)
        grants = load_grant_bins(rg, ctx)
        snr = load_snr_bins(rg, ctx)
        edge = load_edge_bins(rg, ctx)
        app = app[app["t"] <= max_t]
        grants = grants[grants["t"] <= max_t]
        snr = snr[snr["t"] <= max_t]
        edge = edge[edge["t"] <= max_t]
        axes[0].plot(snr["t"], snr["snr_db"], color=spec.color, linewidth=2.5)
        axes[1].plot(grants["t"], grants["mcs_p50"], color=spec.color, linewidth=2.7)
        axes[2].plot(app["t"], app["app_mbps"], color=spec.color, linewidth=2.4)
        axes[3].plot(grants["t"], grants["scheduled_mbps"], color=spec.color, linewidth=2.6)
        axes[4].plot(grants["t"], grants["grant_rate_hz"], color=spec.color, linewidth=2.4)
        axes[5].plot(edge["t"], edge["edge_fps"], color=spec.color, linewidth=2.4)

    titles = [
        "Measured uplink SNR",
        "SINR-selected uplink MCS",
        "Observed feature send rate into OAI",
        "Scheduled uplink service rate",
        "Uplink grants per second",
        "Complete feature frames reaching edge",
    ]
    ylabels = ["dB", "MCS", "Mbps", "Mbps", "grants/s", "frames/s"]
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_title(title, fontsize=12.6, fontweight="bold", loc="left", pad=7)
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
        style_axes(ax)
        ax.set_ylim(bottom=0)
    axes[-1].set_xlabel("Time from first feature send (s)", fontsize=12, fontweight="bold")
    fig.legend(
        handles=legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.998),
        ncol=4,
        frameon=False,
        prop={"weight": "bold", "size": 11},
    )
    axes[0].text(
        0.0,
        1.30,
        "Uplink-only SINR ladder: radio and traffic rates",
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    savefig(fig, "uplink_only_sinr_ladder_radio_rate_timeseries")


def plot_backlog_drain_timeseries(summary_df: pd.DataFrame, contexts: Dict[str, Dict[str, object]], max_t: float = 180.0) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(13.2, 11.6), sharex=True)
    for spec in RUNS:
        row = summary_df[summary_df["label"].eq(spec.label)].iloc[0]
        rg = str(row["run_group"])
        ctx = contexts[spec.label]
        queue = load_queue_bins(rg, ctx)
        grants = load_grant_bins(rg, ctx)
        queue = queue[queue["t"] <= max_t]
        grants = grants[grants["t"] <= max_t]
        axes[0].plot(queue["t"], queue["bsr_lcg1_p95_mib"], color=spec.color, linewidth=2.6)
        axes[1].plot(queue["t"], queue["rlc_p95_mib"], color=spec.color, linewidth=2.6)
        axes[2].plot(queue["t"], queue["rlc_sdu_drain_mbps"], color=spec.color, linewidth=2.5)
        axes[3].plot(grants["t"], grants["first_tx_mbps"], color=spec.color, linewidth=2.5)
        axes[4].plot(grants["t"], grants["retx_rate_pct"], color=spec.color, linewidth=2.4)

    titles = [
        "UE BSR LCG1 backlog reported to gNB",
        "UE RLC LCID4 buffer occupancy",
        "RLC SDU drain rate",
        "First-transmission uplink service rate",
        "Retransmission rate",
    ]
    ylabels = ["MiB", "MiB", "Mbps", "Mbps", "% grants"]
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_title(title, fontsize=12.6, fontweight="bold", loc="left", pad=7)
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
        style_axes(ax)
        ax.set_ylim(bottom=0)
    axes[-1].set_xlabel("Time from first feature send (s)", fontsize=12, fontweight="bold")
    fig.legend(
        handles=legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.998),
        ncol=4,
        frameon=False,
        prop={"weight": "bold", "size": 11},
    )
    axes[0].text(
        0.0,
        1.30,
        "Uplink-only SINR ladder: backlog and drain behavior",
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    savefig(fig, "uplink_only_sinr_ladder_backlog_drain_timeseries")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-batch", default="track2_sinr_uplink_only_20260803")
    parser.add_argument("--max-t", type=float, default=180.0, help="timeseries x-axis limit in seconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.linewidth": 1.1,
        }
    )
    summary_csv = summary_path(args.base_batch)
    if not summary_csv.exists():
        raise SystemExit(f"summary not found: {summary_csv}")
    source = pd.read_csv(summary_csv)
    rows: List[Dict[str, object]] = []
    contexts: Dict[str, Dict[str, object]] = {}
    for spec in RUNS:
        matches = source[source["label"].eq(spec.label)]
        if matches.empty:
            raise SystemExit(f"missing summary row for {spec.label}")
        row = matches.iloc[0]
        ctx = load_context(str(row["run_group"]))
        contexts[spec.label] = ctx
        rows.append(active_summary(row, spec, ctx))

    df = pd.DataFrame(rows)
    preferred = [
        "label",
        "display",
        "send_frames",
        "send_duration_s",
        "send_fps",
        "app_offered_mbps_send",
        "scheduled_mbps_send",
        "first_tx_mbps_send",
        "edge_frames_30s_drain",
        "edge_delivery_pct_30s_drain",
        "snr_p50_db_active",
        "mcs_p50_active",
        "mcs_p95_active",
        "tbs_avg_kib_send",
        "grant_rate_hz_send",
        "retx_rate_pct_send",
        "front_build_p50_ms",
        "front_to_edge_p50_ms",
        "front_to_edge_p95_ms",
        "tail_p50_ms",
        "map_publish_p50_ms",
        "capture_to_map_publish_p50_ms",
        "bsr_lcg1_p95_mib_send",
        "rlc_p95_mib_send",
        "rlc_drain_mbps_send",
        "udp_partial_messages_dropped_max",
        "spatial_publisher_dropped_max",
        "run_group",
    ]
    df = df[[c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "uplink_only_sinr_presentation_summary.csv"
    md_path = OUT_DIR / "uplink_only_sinr_presentation_summary.md"
    df.to_csv(csv_path, index=False)
    md = "# Uplink-only SINR ladder presentation summary\n\n"
    md += f"Base batch: `{args.base_batch}`\n\n"
    md += "All active-window traffic metrics use first actual feature send as t=0.\n\n"
    md += to_markdown(df) + "\n"
    md_path.write_text(md, encoding="utf-8")

    plot_summary_bars(df)
    plot_latency_breakdown(df)
    plot_radio_rate_timeseries(df, contexts, max_t=args.max_t)
    plot_backlog_drain_timeseries(df, contexts, max_t=args.max_t)

    print(OUT_DIR)
    print(csv_path)
    print(md_path)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
