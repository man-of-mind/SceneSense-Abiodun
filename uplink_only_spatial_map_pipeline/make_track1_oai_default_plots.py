#!/usr/bin/env python3
"""Summarize and plot the Track-1 uplink-only default-OAI run.

The main latency summary reads Track-1 artifacts.  The 100 ms traffic-shape
figure also overlays the optimized closed-loop OAI 106PRB no-AE run as an
observed baseline for the return-wait deployment.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-track1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_GROUP = "track1_track1_oai_default106_ttracer_fps10_track1_default106_20260729_204536"
DEFAULT_CLOSED_LOOP_RUN_GROUP = "downlink_oai_default106_ttracer_fps10_drivable_fast_20260730_default106_noae"
CLOSED_LOOP_RUN_GROUP = os.environ.get("CLOSED_LOOP_RUN_GROUP", DEFAULT_CLOSED_LOOP_RUN_GROUP)
OAI_RUN_DIR = Path(
    "abiodun/uplink_only_spatial_map_pipeline/runs/"
    "track1_oai_default106_ttracer/fps_10_track1_default106_20260729_204536"
)
LOOPBACK_RUN_DIR = Path(
    "abiodun/uplink_only_spatial_map_pipeline/runs/"
    "track1_ideal_loopback_matrix_20260729_fast_pipeline_10fps/"
    "ideal_none_fps10_map0_fast_pipeq2"
)
TTRACER_DIR = Path("abiodun/metrics_logs/scenesense_ttracer") / RUN_GROUP
NETWORK_DIR = Path("abiodun/metrics_logs/scenesense_network") / RUN_GROUP
CLOSED_LOOP_TTRACER_DIR = Path("abiodun/metrics_logs/scenesense_ttracer") / CLOSED_LOOP_RUN_GROUP
CLOSED_LOOP_NETWORK_DIR = Path("abiodun/metrics_logs/scenesense_network") / CLOSED_LOOP_RUN_GROUP
OUT_DIR = Path("abiodun/uplink_only_spatial_map_pipeline/plots/track1_oai_default106")
CLOSED_LOOP_OAI_DEFAULT106_METRICS = Path(
    os.environ.get(
        "CLOSED_LOOP_METRICS_CSV",
        "abiodun/downlink_latency_fps/runs/oai_default106_ttracer/"
        "fps_10_drivable_fast_20260730_default106_noae/streams/"
        "downlink_oai_default106_ttracer_fps10_drivable_fast_20260730_default106_noae_metrics.csv",
    )
)


def _q(series: pd.Series, q: float) -> float:
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return float("nan")
    return float(vals.quantile(q))


def _mean(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return float("nan")
    return float(vals.mean())


def _front_events(run_dir: Path) -> pd.DataFrame:
    streams = run_dir / "front_metrics" / "streams"
    candidates = sorted(streams.glob("*send_events.csv"))
    if not candidates:
        raise FileNotFoundError(f"no front send-events CSV under {streams}")
    return pd.read_csv(candidates[-1])


def _edge_metrics(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "edge_uplink_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _merged_metrics(run_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    edge = _edge_metrics(run_dir)
    front = _front_events(run_dir)
    keep = [
        "frame_id",
        "send_call_ms",
        "feature_payload_bytes",
        "feature_payload_chunks",
        "camera_sent_perf",
        "wall_time_iso",
    ]
    keep = [c for c in keep if c in front.columns]
    merged = edge.merge(front[keep], on="frame_id", how="left", suffixes=("", "_front"))
    merged = merged[merged["send_call_ms"].notna()].copy()
    merged["sensor_prep_ms"] = merged["capture_to_backbone_input_ms"] - merged["model_preprocess_ms"]
    merged["front_model_ms"] = merged["model_preprocess_ms"] + merged["front_backbone_ms"]
    merged["uplink_transport_only_ms"] = (
        merged["front_to_edge_ms"] - merged["send_call_ms"]
    ).clip(lower=0.0)
    merged["edge_tail_total_ms"] = merged["edge_queue_ms"] + merged["tail_ms"]
    return merged, edge, front


def _summarize_path(label: str, run_dir: Path) -> Dict[str, float]:
    merged, edge, front = _merged_metrics(run_dir)
    send_span_s = float(front["camera_sent_perf"].max() - front["camera_sent_perf"].min())
    actual_fps = (len(front) - 1) / send_span_s if send_span_s > 0 else float("nan")
    return {
        "condition": label,
        "sent_frames": float(len(front)),
        "processed_frames": float(len(edge)),
        "delivery_pct": 100.0 * len(edge) / len(front) if len(front) else float("nan"),
        "actual_send_fps": actual_fps,
        "uplink_payload_p50_kib": _q(front["feature_payload_bytes"], 0.50) / 1024.0,
        "uplink_payload_p95_kib": _q(front["feature_payload_bytes"], 0.95) / 1024.0,
        "uplink_chunks_p50": _q(front["feature_payload_chunks"], 0.50),
        "sensor_prep_p50_ms": _q(merged["sensor_prep_ms"], 0.50),
        "sensor_prep_p95_ms": _q(merged["sensor_prep_ms"], 0.95),
        "front_model_p50_ms": _q(merged["front_model_ms"], 0.50),
        "front_model_p95_ms": _q(merged["front_model_ms"], 0.95),
        "serialize_p50_ms": _q(merged["feature_serialize_ms"], 0.50),
        "serialize_p95_ms": _q(merged["feature_serialize_ms"], 0.95),
        "send_call_p50_ms": _q(merged["send_call_ms"], 0.50),
        "send_call_p95_ms": _q(merged["send_call_ms"], 0.95),
        "uplink_transport_p50_ms": _q(merged["uplink_transport_only_ms"], 0.50),
        "uplink_transport_p95_ms": _q(merged["uplink_transport_only_ms"], 0.95),
        "edge_queue_p50_ms": _q(merged["edge_queue_ms"], 0.50),
        "edge_queue_p95_ms": _q(merged["edge_queue_ms"], 0.95),
        "tail_p50_ms": _q(merged["tail_ms"], 0.50),
        "tail_p95_ms": _q(merged["tail_ms"], 0.95),
        "capture_to_tail_p50_ms": _q(merged["capture_to_tail_done_ms"], 0.50),
        "capture_to_tail_p95_ms": _q(merged["capture_to_tail_done_ms"], 0.95),
        "capture_to_map_plus30_p50_ms": _q(merged["capture_to_tail_done_ms"], 0.50) + 30.0,
        "capture_to_map_plus30_p95_ms": _q(merged["capture_to_tail_done_ms"], 0.95) + 30.0,
        "backbone_to_tail_p50_ms": _q(merged["backbone_input_to_tail_done_ms"], 0.50),
        "backbone_to_tail_p95_ms": _q(merged["backbone_input_to_tail_done_ms"], 0.95),
        "sync_tick_p50_ms": _q(merged["sync_world_tick_ms"], 0.50),
        "camera_wait_p50_ms": _q(merged["camera_frame_wait_ms"], 0.50),
        "udp_partial_messages_last": float(
            pd.to_numeric(edge.get("udp_partial_messages_dropped", pd.Series([0])), errors="coerce")
            .dropna()
            .max()
        ),
        "edge_queue_drops_last": float(
            pd.to_numeric(edge.get("edge_receive_queue_dropped", pd.Series([0])), errors="coerce")
            .dropna()
            .max()
        ),
    }


def _time_to_seconds(series: pd.Series) -> pd.Series:
    # t-tracer has HH:MM:SS.ffffff without date. These runs do not cross midnight.
    td = pd.to_timedelta(series.astype(str))
    return td.dt.total_seconds()


def _first_ttracer_second(ttracer_dir: Path = TTRACER_DIR) -> float:
    candidates = [
        ttracer_dir / "ue" / "csv" / "NRUE_MAC_DCI_GRANT.csv",
        ttracer_dir / "gnb" / "csv" / "GNB_MAC_UL_MCS_DECISION.csv",
    ]
    vals = []
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path, usecols=["time"], nrows=1)
            if len(df):
                vals.append(float(_time_to_seconds(df["time"]).iloc[0]))
    if not vals:
        return 0.0
    return min(vals)


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.7,
        }
    )


def _save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def _save_multi(fig: plt.Figure, stems: Iterable[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        for ext in ("png", "pdf"):
            fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_latency(summary: pd.DataFrame) -> None:
    """Audience-facing p50 latency breakdown.

    Uplink-only runs have richer front-side instrumentation, so their
    `front_feature_build` term folds together sensor prep, front model,
    serialization, and UDP send-call time.  The closed-loop run uses its native
    `front_ms` + RTT decomposition from the downlink-latency experiment.
    """
    loop = summary[summary["condition"].eq("Ideal loopback")].iloc[0]
    oai = summary[summary["condition"].eq("OAI default 106PRB")].iloc[0]
    closed = _optimized_closed_loop_oai_summary()

    rows = [
        {
            "condition": "Ideal loopback\nuplink-only",
            "front_feature_build_ms": loop.sensor_prep_p50_ms
            + loop.front_model_p50_ms
            + loop.serialize_p50_ms,
            "feature_uplink_ms": loop.send_call_p50_ms + loop.uplink_transport_p50_ms,
            "edge_tail_ms": loop.edge_queue_p50_ms + loop.tail_p50_ms,
            "result_downlink_ms": 0.0,
            "measured_total_ms": loop.capture_to_tail_p50_ms,
            "total_label": "capture→tail",
        },
        {
            "condition": "OAI 106PRB\nuplink-only",
            "front_feature_build_ms": oai.sensor_prep_p50_ms
            + oai.front_model_p50_ms
            + oai.serialize_p50_ms,
            "feature_uplink_ms": oai.send_call_p50_ms + oai.uplink_transport_p50_ms,
            "edge_tail_ms": oai.edge_queue_p50_ms + oai.tail_p50_ms,
            "result_downlink_ms": 0.0,
            "measured_total_ms": oai.capture_to_tail_p50_ms,
            "total_label": "capture→tail",
        },
        {
            "condition": "OAI 106PRB\nclosed-loop",
            "front_feature_build_ms": closed["front_p50_ms"],
            "feature_uplink_ms": closed["feature_uplink_p50_ms"],
            "edge_tail_ms": closed["back_p50_ms"],
            "result_downlink_ms": closed["downlink_p50_ms"],
            "measured_total_ms": closed["capture_to_result_p50_ms"],
            "total_label": "capture→result",
        },
    ]
    plot_df = pd.DataFrame(rows)
    plot_df.to_csv(OUT_DIR / "track1_latency_breakdown_loopback_uplink_closedloop_summary.csv", index=False)

    components = [
        ("front_feature_build_ms", "Front feature build"),
        ("feature_uplink_ms", "Feature uplink handling"),
        ("edge_tail_ms", "Edge tail"),
        ("result_downlink_ms", "Result downlink"),
    ]
    colors = ["#4C78A8", "#E45756", "#72B7B2", "#F58518"]
    x = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(10.6, 5.4))
    bottom = np.zeros(len(plot_df))
    for (col, name), color in zip(components, colors):
        vals = plot_df[col].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, label=name, color=color, edgecolor="white", linewidth=0.8)
        bottom += vals

    total = plot_df["measured_total_ms"].to_numpy(dtype=float)
    ax.scatter(x, total, marker="D", s=64, color="black", label="Measured p50 total", zorder=5)
    for xi, val, label in zip(x, total, plot_df["total_label"]):
        ax.text(xi, val + 6.5, f"{val:.0f} ms\n{label}", ha="center", va="bottom", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["condition"], fontweight="bold")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency breakdown: loopback, uplink-only OAI, and closed-loop OAI")
    ax.set_ylim(0, max(total) * 1.30)
    ax.legend(ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, prop={"weight": "bold", "size": 9})
    ax.text(
        0.01,
        -0.18,
        "Feature uplink handling uses the reportable front-send/payload-ready → edge-receive convention. Closed-loop includes result downlink; uplink-only stops at edge-tail output for map ingestion.",
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        color="#374151",
    )
    _save_multi(fig, ["track1_latency_breakdown_loopback_vs_oai", "track1_latency_breakdown_loopback_uplink_closedloop"])


def _app_rate_1s(front: pd.DataFrame, t0_abs_s: float) -> pd.DataFrame:
    ts = pd.to_datetime(front["wall_time_iso"])
    abs_s = ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second + ts.dt.microsecond / 1e6
    elapsed = abs_s - t0_abs_s
    bins = np.floor(elapsed).astype(int)
    out = (
        pd.DataFrame({"t": bins, "bytes": front["feature_payload_bytes"], "frames": 1})
        .groupby("t", as_index=False)
        .sum()
    )
    out["app_offered_mbps"] = out["bytes"] * 8.0 / 1e6
    return out


def _closed_loop_send_seconds(metrics: pd.DataFrame) -> pd.Series:
    """Feature-send times for the optimized closed-loop comparator.

    Returned rows do have `t_front_send_wall_s`, but timed-out rows often do
    not.  `elapsed_s - result_wait_ms` recovers the send instant for both,
    so use it as the primary source.
    """
    elapsed = pd.to_numeric(metrics["elapsed_s"], errors="coerce")
    wait_s = pd.to_numeric(metrics["result_wait_ms"], errors="coerce") / 1000.0
    send_s = elapsed - wait_s
    wall = pd.to_numeric(metrics.get("t_front_send_wall_s"), errors="coerce")
    if send_s.notna().sum() < len(metrics) * 0.8 and wall.notna().any():
        send_s = wall - float(wall.dropna().min())
    return send_s


def _closed_loop_app_rate_1s() -> pd.DataFrame:
    metrics = pd.read_csv(CLOSED_LOOP_OAI_DEFAULT106_METRICS)
    send_s = _closed_loop_send_seconds(metrics)
    bins = np.floor(send_s).astype("Int64")
    out = (
        pd.DataFrame(
            {
                "t": bins,
                "bytes": pd.to_numeric(metrics["feature_payload_bytes"], errors="coerce").fillna(0.0),
                "frames": 1,
            }
        )
        .dropna(subset=["t"])
        .groupby("t", as_index=False)
        .sum()
    )
    out["t"] = out["t"].astype(int)
    out["app_offered_mbps"] = out["bytes"] * 8.0 / 1e6
    return out


def _rlc_sdu_rate_1s(ttracer_dir: Path) -> pd.DataFrame:
    queue = ttracer_dir / "ue" / "analysis" / "nrue_queue_windows.csv"
    if queue.exists():
        qdf = pd.read_csv(queue)
        if "sdu_mbps" in qdf.columns:
            return qdf[["window_start_s", "sdu_mbps"]].copy()
    raw = ttracer_dir / "ue" / "csv" / "NR_RLC_TX_DEQUEUE.csv"
    if not raw.exists():
        return pd.DataFrame(columns=["window_start_s", "sdu_mbps"])
    df = pd.read_csv(raw, usecols=["time", "lcid", "pdu_bytes"])
    df = df[pd.to_numeric(df["lcid"], errors="coerce").eq(4)].copy()
    if df.empty:
        return pd.DataFrame(columns=["window_start_s", "sdu_mbps"])
    sec = _time_to_seconds(df["time"]) - _first_ttracer_second(ttracer_dir)
    bins = np.floor(sec).astype(int)
    out = (
        pd.DataFrame({"window_start_s": bins, "bytes": pd.to_numeric(df["pdu_bytes"], errors="coerce").fillna(0.0)})
        .groupby("window_start_s", as_index=False)
        .sum()
    )
    out["sdu_mbps"] = out["bytes"] * 8.0 / 1e6
    return out


def _traffic_rate_inputs(
    label: str,
    app: pd.DataFrame,
    ttracer_dir: Path,
    network_dir: Path,
) -> Dict[str, pd.DataFrame | str]:
    grant = pd.read_csv(ttracer_dir / "ue" / "analysis" / "nrue_grant_windows.csv")
    grant = grant[grant["direction_label"] == "ul"].copy()
    queue = _rlc_sdu_rate_1s(ttracer_dir)
    net = pd.read_csv(network_dir / "network_timeseries.csv")
    net = net[(net["iface"] == "oaitun_ue1") & (net["iface_up"] == True)].copy()
    return {"label": label, "app": app, "grant": grant, "queue": queue, "net": net}


def _plot_one_rate_panel(ax: plt.Axes, data: Dict[str, pd.DataFrame | str], title: str) -> None:
    app = data["app"]
    grant = data["grant"]
    queue = data["queue"]
    net = data["net"]
    assert isinstance(app, pd.DataFrame)
    assert isinstance(grant, pd.DataFrame)
    assert isinstance(queue, pd.DataFrame)
    assert isinstance(net, pd.DataFrame)
    ax.plot(app["t"], app["app_offered_mbps"], color="#4C78A8", linewidth=2.25, label="App feature offered")
    ax.plot(net["elapsed_s"], net["tx_bitrate_mbps"], color="#F58518", linewidth=2.0, label="UE tunnel TX")
    ax.plot(grant["window_start_s"], grant["scheduled_mbps"], color="#E45756", linewidth=1.95, label="MAC scheduled UL")
    if not queue.empty:
        ax.plot(queue["window_start_s"], queue["sdu_mbps"], color="#54A24B", linewidth=1.95, label="RLC SDU drain")
    ax.set_xlim(0, 250)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Rate (Mbps)\n1 s window")
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, ncol=2, prop={"weight": "bold", "size": 9})


def plot_traffic_rates(oai_front: pd.DataFrame) -> None:
    track_app = _app_rate_1s(oai_front, _first_ttracer_second(TTRACER_DIR))
    closed_app = _closed_loop_app_rate_1s()
    track = _traffic_rate_inputs("Uplink-only", track_app, TTRACER_DIR, NETWORK_DIR)
    closed = _traffic_rate_inputs("Closed-loop", closed_app, CLOSED_LOOP_TTRACER_DIR, CLOSED_LOOP_NETWORK_DIR)

    fig, axes = plt.subplots(2, 1, figsize=(12.2, 8.2), sharex=True, sharey=True)
    _plot_one_rate_panel(axes[0], track, "A. Uplink-only OAI: denser feature stream")
    _plot_one_rate_panel(axes[1], closed, "B. Closed-loop OAI: return-wait creates sparse bursts")
    axes[0].set_ylim(0, 95)
    axes[1].set_ylim(0, 95)
    axes[1].set_xlabel("Time since OAI trace start (s)")
    fig.suptitle("OAI traffic rate over time: uplink-only vs closed-loop", y=0.98, fontweight="bold")
    fig.subplots_adjust(hspace=0.25, top=0.91)
    _save(fig, "track1_oai_traffic_rates_1s")


def _aggregate_raw_timeseries(
    path: Path,
    value_cols: Iterable[str],
    filters: Dict[str, int] | None = None,
    ttracer_dir: Path | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    if filters:
        for col, val in filters.items():
            df = df[df[col] == val]
    if df.empty:
        return pd.DataFrame()
    sec = _time_to_seconds(df["time"])
    t0 = _first_ttracer_second(ttracer_dir or path.parents[2])
    df = df.copy()
    df["window_start_s"] = np.floor(sec - t0).astype(int)
    agg = df.groupby("window_start_s").agg({col: ["median", "mean", "max"] for col in value_cols})
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    return agg


def _ul_grant_windows(ttracer_dir: Path) -> pd.DataFrame:
    grant = pd.read_csv(ttracer_dir / "ue" / "analysis" / "nrue_grant_windows.csv")
    return grant[grant["direction_label"] == "ul"].copy()


def _snr_windows(ttracer_dir: Path) -> pd.DataFrame:
    return _aggregate_raw_timeseries(
        ttracer_dir / "gnb" / "csv" / "GNB_MAC_PUSCH_POWER_CONTROL.csv",
        ["snrx10"],
        ttracer_dir=ttracer_dir,
    )


def plot_radio_backlog() -> None:
    track_ul = _ul_grant_windows(TTRACER_DIR)
    closed_ul = _ul_grant_windows(CLOSED_LOOP_TTRACER_DIR)
    track_snr = _snr_windows(TTRACER_DIR)
    closed_snr = _snr_windows(CLOSED_LOOP_TTRACER_DIR)

    def _active_stop_s(df: pd.DataFrame) -> float:
        active = df[
            (pd.to_numeric(df["scheduled_mbps"], errors="coerce").fillna(0.0) > 1.0)
            | (pd.to_numeric(df["avg_mcs"], errors="coerce").fillna(0.0) > 0.5)
            | (pd.to_numeric(df["avg_rb_size"], errors="coerce").fillna(0.0) > 10.0)
        ].copy()
        if active.empty:
            return 250.0
        return float(active["window_start_s"].max())

    common_stop_s = min(250.0, _active_stop_s(track_ul), _active_stop_s(closed_ul))
    track_ul = track_ul[track_ul["window_start_s"].between(0.0, common_stop_s)].copy()
    closed_ul = closed_ul[closed_ul["window_start_s"].between(0.0, common_stop_s)].copy()
    if not track_snr.empty:
        track_snr = track_snr[track_snr["window_start_s"].between(0.0, common_stop_s)].copy()
    if not closed_snr.empty:
        closed_snr = closed_snr[closed_snr["window_start_s"].between(0.0, common_stop_s)].copy()

    colors = {"Uplink-only": "#2563EB", "Closed-loop": "#DC2626"}
    fig, axes = plt.subplots(4, 1, figsize=(12.2, 9.2), sharex=True)
    fig.suptitle("OAI scheduler comparison: uplink-only vs closed-loop", y=0.985, fontweight="bold")

    axes[0].plot(
        track_ul["window_start_s"],
        track_ul["scheduled_mbps"],
        color=colors["Uplink-only"],
        linewidth=2.2,
        label="Uplink-only",
    )
    axes[0].plot(
        closed_ul["window_start_s"],
        closed_ul["scheduled_mbps"],
        color=colors["Closed-loop"],
        linewidth=2.0,
        label="Closed-loop",
        alpha=0.88,
    )
    axes[0].set_ylabel("Scheduled UL\n(Mbps)")
    axes[0].set_title("A. Scheduled uplink throughput")
    axes[0].legend(loc="upper right", frameon=True, framealpha=0.95, prop={"weight": "bold", "size": 9})

    axes[1].plot(
        track_ul["window_start_s"],
        track_ul["avg_mcs"],
        color=colors["Uplink-only"],
        linewidth=2.2,
        label="Uplink-only avg MCS",
    )
    axes[1].plot(
        closed_ul["window_start_s"],
        closed_ul["avg_mcs"],
        color=colors["Closed-loop"],
        linewidth=2.0,
        label="Closed-loop avg MCS",
        alpha=0.88,
    )
    axes[1].set_ylabel("Avg MCS")
    axes[1].set_title("B. MCS assigned by OAI link adaptation")
    axes[1].legend(loc="upper right", frameon=True, framealpha=0.95, prop={"weight": "bold", "size": 9})

    axes[2].plot(
        track_ul["window_start_s"],
        track_ul["avg_rb_size"],
        color=colors["Uplink-only"],
        linewidth=2.2,
        label="Uplink-only avg PRB",
    )
    axes[2].plot(
        closed_ul["window_start_s"],
        closed_ul["avg_rb_size"],
        color=colors["Closed-loop"],
        linewidth=2.0,
        label="Closed-loop avg PRB",
        alpha=0.88,
    )
    axes[2].set_ylabel("Allocated\nPRB")
    axes[2].set_title("C. PRB allocation stays near the 106PRB ceiling")
    axes[2].legend(loc="lower right", frameon=True, framealpha=0.95, prop={"weight": "bold", "size": 9})

    if not track_snr.empty:
        axes[3].plot(
            track_snr["window_start_s"],
            track_snr["snrx10_median"] / 10.0,
            color=colors["Uplink-only"],
            linewidth=2.2,
            label="Uplink-only SNR",
        )
    if not closed_snr.empty:
        axes[3].plot(
            closed_snr["window_start_s"],
            closed_snr["snrx10_median"] / 10.0,
            color=colors["Closed-loop"],
            linewidth=2.0,
            label="Closed-loop SNR",
            alpha=0.88,
        )
    axes[3].set_ylabel("SNR (dB)")
    axes[3].set_title("D. Reported SNR is high in both runs")
    axes[3].set_xlabel("Time since t-tracer start (s)")
    axes[3].legend(loc="upper right", frameon=True, framealpha=0.95, prop={"weight": "bold", "size": 9})

    for ax in axes:
        ax.set_xlim(0, common_stop_s)
    fig.subplots_adjust(hspace=0.35, top=0.92)
    _save_multi(fig, ["track1_oai_radio_backlog_timeseries", "track1_oai_radio_scheduler_comparison"])


def plot_delivery(front: pd.DataFrame, edge: pd.DataFrame) -> None:
    # Use perf_counter timestamps here.  The front wall-time logs are local-time
    # while edge/container wall-time may be UTC, but perf_counter is consistent
    # for this same-host RFsim setup and is the timestamp used for latency.
    t0 = float(front["camera_sent_perf"].min())
    sent_t = np.floor(pd.to_numeric(front["camera_sent_perf"], errors="coerce") - t0).astype(int)
    sent = pd.DataFrame({"t": sent_t, "frames": 1}).groupby("t", as_index=False).sum()
    edge_t = np.floor(pd.to_numeric(edge["t_edge_recv_perf"], errors="coerce") - t0).astype(int)
    processed = pd.DataFrame({"t": edge_t, "processed": 1}).groupby("t", as_index=False).sum()
    drops = edge[["t_edge_recv_perf", "udp_partial_messages_dropped"]].copy()
    drops["t"] = np.floor(pd.to_numeric(drops["t_edge_recv_perf"], errors="coerce") - t0).astype(int)
    drops = drops.groupby("t", as_index=False)["udp_partial_messages_dropped"].max()
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.plot(sent["t"], sent["frames"].cumsum(), color="#4C78A8", linewidth=2.2, label="front frames sent")
    ax.plot(processed["t"], processed["processed"].cumsum(), color="#54A24B", linewidth=2.2, label="edge frames processed")
    ax2 = ax.twinx()
    ax2.plot(drops["t"], drops["udp_partial_messages_dropped"], color="#E45756", linewidth=1.8, label="UDP partial messages dropped")
    ax.set_xlim(0, 250)
    ax.set_xlabel("Time since t-tracer start (s)")
    ax.set_ylabel("Cumulative frames")
    ax2.set_ylabel("Cumulative partial messages", fontweight="bold", color="#E45756")
    ax2.tick_params(axis="y", labelcolor="#E45756")
    ax.set_title("Track-1 OAI delivery/reassembly behavior")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True, framealpha=0.95)
    _save(fig, "track1_oai_delivery_reassembly")


def _bin_100ms_from_hms(df: pd.DataFrame, bytes_col: str) -> pd.DataFrame:
    t0 = _first_ttracer_second()
    sec = _time_to_seconds(df["time"]) - t0
    bins = np.floor(sec * 10.0).astype(int)
    out = (
        pd.DataFrame({"bin": bins, "bytes": pd.to_numeric(df[bytes_col], errors="coerce").fillna(0.0)})
        .groupby("bin", as_index=False)
        .sum()
    )
    out["t_s"] = out["bin"] / 10.0
    out["mbit_per_100ms"] = out["bytes"] * 8.0 / 1e6
    out["mbps_equiv"] = out["mbit_per_100ms"] / 0.1
    return out


def _bin_100ms_front(front: pd.DataFrame) -> pd.DataFrame:
    t0 = _first_ttracer_second()
    ts = pd.to_datetime(front["wall_time_iso"])
    abs_s = ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second + ts.dt.microsecond / 1e6
    bins = np.floor((abs_s - t0) * 10.0).astype(int)
    out = (
        pd.DataFrame(
            {
                "bin": bins,
                "bytes": pd.to_numeric(front["feature_payload_bytes"], errors="coerce").fillna(0.0),
                "frames": 1,
            }
        )
        .groupby("bin", as_index=False)
        .sum()
    )
    out["t_s"] = out["bin"] / 10.0
    out["mbit_per_100ms"] = out["bytes"] * 8.0 / 1e6
    out["mbps_equiv"] = out["mbit_per_100ms"] / 0.1
    return out


def _bin_100ms_front_relative(front: pd.DataFrame) -> pd.DataFrame:
    """100 ms app offered-load bins using same-process perf timestamps."""
    t = pd.to_numeric(front["camera_sent_perf"], errors="coerce")
    t0 = float(t.dropna().min())
    bins = np.floor((t - t0) * 10.0).astype("Int64")
    out = (
        pd.DataFrame(
            {
                "bin": bins,
                "bytes": pd.to_numeric(front["feature_payload_bytes"], errors="coerce").fillna(0.0),
                "frames": 1,
            }
        )
        .dropna(subset=["bin"])
        .groupby("bin", as_index=False)
        .sum()
    )
    out["bin"] = out["bin"].astype(int)
    out["t_s"] = out["bin"] / 10.0
    out["mbit_per_100ms"] = out["bytes"] * 8.0 / 1e6
    out["mbps_equiv"] = out["mbit_per_100ms"] / 0.1
    return out


def _bin_100ms_closed_loop_app() -> pd.DataFrame:
    """Closed-loop feature-send bins from the OAI 106PRB no-AE run.

    Use actual front-send wall timestamps when available. This is the correct
    OAI-vs-OAI comparator for the traffic-shape panel. It uses the same fast
    radar-rasterizer path as the Track-1 optimized frontend, so the remaining
    sparse cadence is the closed-loop result-wait/OAI return effect.
    """
    df = pd.read_csv(CLOSED_LOOP_OAI_DEFAULT106_METRICS)
    send = _closed_loop_send_seconds(df)
    send_rel = send - float(send.dropna().min())

    bins = np.floor(send_rel * 10.0).astype("Int64")
    out = (
        pd.DataFrame(
            {
                "bin": bins,
                "bytes": pd.to_numeric(df["feature_payload_bytes"], errors="coerce").fillna(0.0),
                "frames": 1,
            }
        )
        .dropna(subset=["bin"])
        .groupby("bin", as_index=False)
        .sum()
    )
    out["bin"] = out["bin"].astype(int)
    out["t_s"] = out["bin"] / 10.0
    out["mbit_per_100ms"] = out["bytes"] * 8.0 / 1e6
    out["mbps_equiv"] = out["mbit_per_100ms"] / 0.1
    return out


def _complete_100ms_grid(dfs: Iterable[pd.DataFrame], end_s: float = 250.0) -> pd.DataFrame:
    max_bin = int(end_s * 10.0)
    grid = pd.DataFrame({"bin": np.arange(0, max_bin + 1)})
    grid["t_s"] = grid["bin"] / 10.0
    return grid


def _track1_rlc_lcid4_occupancy() -> pd.DataFrame:
    """Load changed LCID4 RLC occupancy points for the Track-1 OAI run."""
    path = TTRACER_DIR / "ue" / "csv" / "NRUE_MAC_RLC_BUFFER_STATUS.csv"
    chunks = []
    for chunk in pd.read_csv(
        path,
        usecols=["time", "lcid", "bytes_in_buffer"],
        chunksize=750_000,
    ):
        chunk["lcid"] = pd.to_numeric(chunk["lcid"], errors="coerce")
        chunk = chunk[chunk["lcid"].eq(4)][["time", "bytes_in_buffer"]].copy()
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        return pd.DataFrame(columns=["t_s", "kb"])
    df = pd.concat(chunks, ignore_index=True)
    df["t_s"] = _time_to_seconds(df["time"]) - _first_ttracer_second()
    df["kb"] = pd.to_numeric(df["bytes_in_buffer"], errors="coerce") / 1024.0
    df = df[df["kb"].ne(df["kb"].shift())].reset_index(drop=True)
    return df[["t_s", "kb"]]


def _extract_track1_rlc_decay(target_ms: float) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Find a clean ~1 MB RLC occupancy decay window.

    This mirrors the older `carla_low_mcs_observed_rlc_drain` figure: choose a
    window where one feature-frame-sized backlog enters LCID4 and drains down.
    """
    df = _track1_rlc_lcid4_occupancy()
    t = df["t_s"].to_numpy(dtype=float)
    b = df["kb"].to_numpy(dtype=float)
    starts = np.where((b > 950.0) & (np.r_[True, b[:-1] < 500.0]))[0]
    best = None
    best_score = None
    for start in starts:
        end = start
        while end < len(b) - 1 and t[end] - t[start] < 0.50 and b[end] > 120.0:
            end += 1
        if end <= start or b[end] > 140.0:
            continue
        segment = b[start : end + 1]
        large_increases = int(np.sum(np.diff(segment) > 50.0))
        duration_ms = float((t[end] - t[start]) * 1000.0)
        # Prefer a clean, mostly monotonic decay near the run-median drain time.
        score = (large_increases, abs(duration_ms - target_ms), start)
        if best_score is None or score < best_score:
            best_score = score
            best = (start, end)
    if best is None:
        raise RuntimeError("Could not find a clean Track-1 RLC decay window")

    start, end = best
    t_start = float(t[start])
    t0 = t_start - 0.010
    t1 = float(t[end]) + 0.040
    out = df[df["t_s"].between(t0, t1)].copy()
    out["since_queued_ms"] = (out["t_s"] - t_start) * 1000.0
    drain_s = float(t[end] - t[start])
    stats = {
        "start_kb": float(b[start]),
        "end_kb": float(b[end]),
        "drain_ms": float(drain_s * 1000.0),
        "burst_slope_mbps": float((b[start] - b[end]) * 1024.0 * 8.0 / drain_s / 1e6)
        if drain_s > 0
        else float("nan"),
        "window_start_s": t_start,
    }
    return out, stats


def plot_100ms_volume_and_drain(oai_front: pd.DataFrame) -> Dict[str, float]:
    app = _bin_100ms_front_relative(oai_front)
    closed_app = _bin_100ms_closed_loop_app()

    grant = pd.read_csv(TTRACER_DIR / "ue" / "csv" / "NRUE_MAC_DCI_GRANT.csv")
    grant = grant[grant["direction"] == 1].copy()
    mac = _bin_100ms_from_hms(grant, "tbs")

    rlc = pd.read_csv(TTRACER_DIR / "ue" / "csv" / "NR_RLC_TX_DEQUEUE.csv")
    rlc = rlc[rlc["lcid"] == 4].copy()
    rlc_drain = _bin_100ms_from_hms(rlc, "pdu_bytes")

    bsr = pd.read_csv(TTRACER_DIR / "ue" / "csv" / "NRUE_MAC_BSR_STATUS.csv")
    bsr["sec"] = _time_to_seconds(bsr["time"]) - _first_ttracer_second()
    bsr["bin"] = np.floor(bsr["sec"] * 10.0).astype(int)
    bsr["lcg1_kib"] = pd.to_numeric(bsr["lcg1_bytes"], errors="coerce").fillna(0.0) / 1024.0
    bsr_bin = bsr.groupby("bin", as_index=False).agg(lcg1_p50_kib=("lcg1_kib", "median"), lcg1_p95_kib=("lcg1_kib", lambda x: x.quantile(0.95)))
    bsr_bin["t_s"] = bsr_bin["bin"] / 10.0
    bsr["bin1s"] = np.floor(bsr["sec"]).astype(int)
    bsr_1s = bsr.groupby("bin1s", as_index=False).agg(
        lcg1_p50_kib=("lcg1_kib", "median"),
        lcg1_p95_kib=("lcg1_kib", lambda x: x.quantile(0.95)),
        lcg1_max_kib=("lcg1_kib", "max"),
    )
    bsr_1s["t_s"] = bsr_1s["bin1s"].astype(float)

    rlc_occ = pd.read_csv(TTRACER_DIR / "ue" / "csv" / "NRUE_MAC_RLC_BUFFER_STATUS.csv")
    rlc_occ = rlc_occ[rlc_occ["lcid"] == 4].copy()
    rlc_occ["sec"] = _time_to_seconds(rlc_occ["time"]) - _first_ttracer_second()
    rlc_occ["bin"] = np.floor(rlc_occ["sec"] * 10.0).astype(int)
    rlc_occ["lcid4_kib"] = pd.to_numeric(rlc_occ["bytes_in_buffer"], errors="coerce").fillna(0.0) / 1024.0
    rlc_occ_bin = rlc_occ.groupby("bin", as_index=False).agg(lcid4_p50_kib=("lcid4_kib", "median"), lcid4_p95_kib=("lcid4_kib", lambda x: x.quantile(0.95)))
    rlc_occ_bin["t_s"] = rlc_occ_bin["bin"] / 10.0
    rlc_occ["bin1s"] = np.floor(rlc_occ["sec"]).astype(int)
    rlc_occ_1s = rlc_occ.groupby("bin1s", as_index=False).agg(
        lcid4_p50_kib=("lcid4_kib", "median"),
        lcid4_p95_kib=("lcid4_kib", lambda x: x.quantile(0.95)),
        lcid4_max_kib=("lcid4_kib", "max"),
    )
    rlc_occ_1s["t_s"] = rlc_occ_1s["bin1s"].astype(float)

    grid = _complete_100ms_grid([app, mac, rlc_drain], end_s=250.0)
    for name, df in [("app", app), ("mac", mac), ("rlc", rlc_drain)]:
        grid = grid.merge(
            df[["bin", "mbit_per_100ms", "mbps_equiv"]].rename(
                columns={
                    "mbit_per_100ms": f"{name}_mbit_100ms",
                    "mbps_equiv": f"{name}_mbps",
                }
            ),
            on="bin",
            how="left",
        )
    grid = grid.fillna(0.0)

    frame_mbit = _q(oai_front["feature_payload_bytes"], 0.50) * 8.0 / 1e6
    active = grid[(grid["t_s"] >= 20.0) & (grid["t_s"] <= 205.0)].copy()
    closed_grid = _complete_100ms_grid([closed_app], end_s=max(250.0, float(closed_app["t_s"].max()) + 1.0))
    closed_grid = closed_grid.merge(
        closed_app[["bin", "mbit_per_100ms", "mbps_equiv"]].rename(
            columns={"mbit_per_100ms": "app_mbit_100ms", "mbps_equiv": "app_mbps"}
        ),
        on="bin",
        how="left",
    ).fillna(0.0)
    closed_active = closed_grid[(closed_grid["t_s"] >= 0.0) & (closed_grid["t_s"] <= 205.0)].copy()
    app_active_bins_pct = 100.0 * float((active["app_mbit_100ms"] > 0).mean())
    closed_app_active_bins_pct = 100.0 * float((closed_active["app_mbit_100ms"] > 0).mean())
    mac_active_bins_pct = 100.0 * float((active["mac_mbit_100ms"] > 0).mean())
    rlc_active_bins_pct = 100.0 * float((active["rlc_mbit_100ms"] > 0).mean())
    rlc_mbit_100ms_p50_all = _q(active["rlc_mbit_100ms"], 0.50)
    nominal_drain_ms = 100.0 * frame_mbit / rlc_mbit_100ms_p50_all if rlc_mbit_100ms_p50_all > 0 else float("nan")
    decay_win, decay_stats = _extract_track1_rlc_decay(nominal_drain_ms)

    # Representative one-frame drain window.  This window is chosen because its
    # 100 ms drain is close to the active-window median, avoiding cherry-picking
    # the fastest spike.
    drain_t0_s = 150.0
    drain_window_s = 0.25
    rlc_deq = pd.read_csv(TTRACER_DIR / "ue" / "csv" / "NR_RLC_TX_DEQUEUE.csv")
    rlc_deq = rlc_deq[rlc_deq["lcid"] == 4].copy()
    rlc_deq["t_s"] = _time_to_seconds(rlc_deq["time"]) - _first_ttracer_second()
    rlc_win = rlc_deq[
        (rlc_deq["t_s"] >= drain_t0_s) & (rlc_deq["t_s"] <= drain_t0_s + drain_window_s)
    ].sort_values("t_s")
    rlc_win["elapsed_ms"] = (rlc_win["t_s"] - drain_t0_s) * 1000.0
    rlc_win["cum_mbit"] = pd.to_numeric(rlc_win["pdu_bytes"], errors="coerce").fillna(0.0).cumsum() * 8.0 / 1e6

    mac_raw = pd.read_csv(TTRACER_DIR / "ue" / "csv" / "NRUE_MAC_DCI_GRANT.csv")
    mac_raw = mac_raw[mac_raw["direction"] == 1].copy()
    mac_raw["t_s"] = _time_to_seconds(mac_raw["time"]) - _first_ttracer_second()
    mac_win = mac_raw[
        (mac_raw["t_s"] >= drain_t0_s) & (mac_raw["t_s"] <= drain_t0_s + drain_window_s)
    ].sort_values("t_s")
    mac_win["elapsed_ms"] = (mac_win["t_s"] - drain_t0_s) * 1000.0
    mac_win["cum_mbit"] = pd.to_numeric(mac_win["tbs"], errors="coerce").fillna(0.0).cumsum() * 8.0 / 1e6

    def _cross_time_ms(df: pd.DataFrame) -> float:
        hit = df[df["cum_mbit"] >= frame_mbit]
        if hit.empty:
            return float("nan")
        return float(hit["elapsed_ms"].iloc[0])

    rlc_cross_ms = _cross_time_ms(rlc_win)
    mac_cross_ms = _cross_time_ms(mac_win)

    fig, axes = plt.subplots(2, 2, figsize=(15.2, 8.8))
    fig.subplots_adjust(hspace=0.44, wspace=0.24)
    barw = 0.075
    ax_app = axes[0, 0]
    ax_closed = axes[0, 1]
    ax_cum = axes[1, 0]
    ax_backlog = axes[1, 1]

    zoom_start_s = 0.0
    zoom_end_s = 20.0
    zoom = grid[(grid["t_s"] >= zoom_start_s) & (grid["t_s"] <= zoom_end_s)].copy()
    zoom_app_active_pct = 100.0 * float((zoom["app_mbit_100ms"] > 0).mean()) if len(zoom) else float("nan")

    ax_app.bar(
        zoom["t_s"],
        zoom["app_mbit_100ms"],
        width=barw,
        color="#4C78A8",
        alpha=0.40,
        label="App offered",
        align="edge",
    )
    ax_app.axhline(
        frame_mbit,
        color="#333333",
        linewidth=1.4,
        linestyle="--",
        alpha=0.75,
        label=f"one feature frame ≈ {frame_mbit:.1f} Mbit",
    )
    ax_app.set_xlim(zoom_start_s, zoom_end_s)
    ax_app.set_ylim(0, max(18.0, zoom["app_mbit_100ms"].quantile(0.995) * 1.08))
    ax_app.set_title("A. Uplink-only app bursts, 100 ms bins")
    ax_app.set_xlabel("Time since first feature send (s)")
    ax_app.set_ylabel("Mbits offered per 100 ms\n(×10 = Mbps)")
    ax_app.legend(loc="upper right", frameon=True, framealpha=0.95)
    ax_app.text(
        0.02,
        0.90,
        f"One nonzero bin ≈ one frame\n20 s window idle: {100.0 - zoom_app_active_pct:.0f}% of 100 ms bins",
        transform=ax_app.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#B8C7D9", alpha=0.95),
    )

    closed_zoom = closed_grid[(closed_grid["t_s"] >= zoom_start_s) & (closed_grid["t_s"] <= zoom_end_s)].copy()
    closed_zoom_active_pct = (
        100.0 * float((closed_zoom["app_mbit_100ms"] > 0).mean()) if len(closed_zoom) else float("nan")
    )
    ax_closed.bar(
        closed_zoom["t_s"],
        closed_zoom["app_mbit_100ms"],
        width=barw,
        color="#F58518",
        alpha=0.48,
        label="Closed-loop app offered",
        align="edge",
    )
    ax_closed.axhline(
        frame_mbit,
        color="#333333",
        linewidth=1.4,
        linestyle="--",
        alpha=0.75,
        label=f"one feature frame ≈ {frame_mbit:.1f} Mbit",
    )
    ax_closed.set_xlim(zoom_start_s, zoom_end_s)
    ax_closed.set_ylim(0, max(18.0, closed_zoom["app_mbit_100ms"].quantile(0.995) * 1.08))
    ax_closed.set_title("B. Closed-loop OAI app bursts, 100 ms bins")
    ax_closed.set_xlabel("Time since first feature send (s)")
    ax_closed.set_ylabel("Mbits offered per 100 ms\n(×10 = Mbps)")
    ax_closed.legend(loc="upper right", frameon=True, framealpha=0.95)
    ax_closed.text(
        0.02,
        0.90,
        f"Return-wait cadence\n20 s window idle: {100.0 - closed_zoom_active_pct:.0f}% of 100 ms bins",
        transform=ax_closed.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#B8C7D9", alpha=0.95),
    )

    ax_cum.plot(
        decay_win["since_queued_ms"],
        decay_win["kb"],
        color="#D1495B",
        linewidth=3.2,
        solid_capstyle="round",
        label="UE RLC LCID4 occupancy",
    )
    ax_cum.fill_between(decay_win["since_queued_ms"], 0, decay_win["kb"], color="#D1495B", alpha=0.10)
    ax_cum.axhline(1024.0, color="#475569", linewidth=1.5, linestyle="--", alpha=0.75, label="~1 MB feature frame")
    ax_cum.annotate(
        f"Observed drain: {decay_stats['drain_ms']:.0f} ms\n"
        f"{decay_stats['start_kb']:.0f} KB → {decay_stats['end_kb']:.0f} KB\n"
        f"burst-slope ≈ {decay_stats['burst_slope_mbps']:.0f} Mbps",
        xy=(decay_stats["drain_ms"] * 0.55, (decay_stats["start_kb"] + decay_stats["end_kb"]) * 0.45),
        xytext=(decay_stats["drain_ms"] * 0.78, max(260.0, decay_stats["start_kb"] * 0.58)),
        arrowprops=dict(arrowstyle="->", color="#D1495B", linewidth=1.7),
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#D1495B", alpha=0.95),
    )
    ax_cum.set_xlim(-10, max(125.0, decay_stats["drain_ms"] + 25.0))
    ax_cum.set_ylim(0, max(1200.0, decay_win["kb"].max() * 1.08))
    ax_cum.set_title("C. RLC occupancy drain: one feature burst sits in UE RLC")
    ax_cum.set_xlabel(f"Time within selected burst window near t={decay_stats['window_start_s']:.0f}s (ms)")
    ax_cum.set_ylabel("UE RLC LCID4\noccupancy (KiB)")
    ax_cum.legend(loc="lower right", frameon=True, framealpha=0.95)
    ax_cum.text(
        0.02,
        0.92,
        f"Run-median estimate: {frame_mbit:.1f} Mbit / {rlc_mbit_100ms_p50_all:.1f} Mbit per 100 ms ≈ {nominal_drain_ms:.0f} ms",
        transform=ax_cum.transAxes,
        va="top",
        ha="left",
        fontsize=10.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#B8C7D9", alpha=0.95),
    )

    ax_backlog.plot(
        rlc_occ_1s["t_s"],
        rlc_occ_1s["lcid4_p95_kib"],
        color="#374151",
        linewidth=2.0,
        label="RLC LCID4 buffer p95",
    )
    ax_backlog.plot(
        rlc_occ_1s["t_s"],
        rlc_occ_1s["lcid4_max_kib"],
        color="#111827",
        linewidth=1.2,
        linestyle=":",
        alpha=0.80,
        label="RLC LCID4 buffer max",
    )
    ax_backlog.plot(
        bsr_1s["t_s"],
        bsr_1s["lcg1_p95_kib"],
        color="#B279A2",
        linewidth=2.0,
        label="UE BSR LCG1 p95",
    )
    ax_backlog.plot(
        bsr_1s["t_s"],
        bsr_1s["lcg1_max_kib"],
        color="#7C3F73",
        linewidth=1.2,
        linestyle=":",
        alpha=0.80,
        label="UE BSR LCG1 max",
    )
    ax_backlog.set_xlim(20, 205)
    ax_backlog.set_title("D. Backlog context: UE RLC buffer and BSR")
    ax_backlog.set_xlabel("Time since t-tracer start (s)")
    ax_backlog.set_ylabel("Backlog / buffer (KiB)")
    ax_backlog.set_ylim(bottom=0)
    ax_backlog.legend(loc="upper right", frameon=True, framealpha=0.95, ncol=2)
    _save(fig, "track1_oai_100ms_volume_drain_backlog")

    return {
        "frame_mbit_p50": frame_mbit,
        "app_active_bins_pct": app_active_bins_pct,
        "closed_loop_app_active_bins_pct": closed_app_active_bins_pct,
        "track1_example20s_app_active_pct": zoom_app_active_pct,
        "closed_loop_example20s_app_active_pct": closed_zoom_active_pct,
        "app_nonzero_bin_mbit_p50": _q(active.loc[active["app_mbit_100ms"] > 0, "app_mbit_100ms"], 0.50),
        "app_nonzero_bin_mbit_p95": _q(active.loc[active["app_mbit_100ms"] > 0, "app_mbit_100ms"], 0.95),
        "app_nonzero_bin_mbit_max": _q(active.loc[active["app_mbit_100ms"] > 0, "app_mbit_100ms"], 1.00),
        "mac_active_bins_pct": mac_active_bins_pct,
        "rlc_active_bins_pct": rlc_active_bins_pct,
        "mac_mbit_100ms_p50_all": _q(active["mac_mbit_100ms"], 0.50),
        "rlc_mbit_100ms_p50_all": _q(active["rlc_mbit_100ms"], 0.50),
        "mac_mbit_100ms_p50_active": _q(active.loc[active["mac_mbit_100ms"] > 0, "mac_mbit_100ms"], 0.50),
        "rlc_mbit_100ms_p50_active": _q(active.loc[active["rlc_mbit_100ms"] > 0, "rlc_mbit_100ms"], 0.50),
        "mac_mbps_equiv_p50_all": _q(active["mac_mbps"], 0.50),
        "rlc_mbps_equiv_p50_all": _q(active["rlc_mbps"], 0.50),
        "nominal_one_frame_drain_ms": nominal_drain_ms,
        "representative_rlc_one_frame_drain_ms": decay_stats["drain_ms"],
        "representative_mac_one_frame_drain_ms": mac_cross_ms,
        "representative_rlc_burst_slope_mbps": decay_stats["burst_slope_mbps"],
        "representative_rlc_start_kib": decay_stats["start_kb"],
        "representative_rlc_end_kib": decay_stats["end_kb"],
    }


def plot_track1_observed_rlc_drain(burst_summary: Dict[str, float]) -> None:
    """Standalone old-style RLC occupancy drain plot for slides."""
    target_ms = float(burst_summary.get("nominal_one_frame_drain_ms", 127.0))
    decay_win, stats = _extract_track1_rlc_decay(target_ms)
    grant_summary = pd.read_csv(TTRACER_DIR / "ue" / "analysis" / "nrue_grant_summary.csv")
    ul = grant_summary[grant_summary["direction_label"].eq("ul")].iloc[0]
    grant_raw = pd.read_csv(TTRACER_DIR / "ue" / "csv" / "NRUE_MAC_DCI_GRANT.csv")
    grant_raw = grant_raw[grant_raw["direction"].eq(1)].copy()
    tbs_p50 = _q(grant_raw["tbs"], 0.50)

    fig, ax = plt.subplots(figsize=(12.4, 6.3))
    fig.subplots_adjust(left=0.105, right=0.975, top=0.84, bottom=0.16)
    fig.suptitle(
        "Uplink-only OAI RLC drain: one feature burst sits in UE RLC",
        y=0.965,
        fontweight="bold",
    )

    ax.plot(
        decay_win["since_queued_ms"],
        decay_win["kb"],
        color="#D1495B",
        linewidth=3.2,
        solid_capstyle="round",
    )
    ax.fill_between(decay_win["since_queued_ms"], 0, decay_win["kb"], color="#D1495B", alpha=0.12)
    ax.axhline(1024.0, color="#475569", linestyle="--", linewidth=1.7, alpha=0.75)
    ax.text(
        0.985,
        0.91,
        "~1 MB feature frame",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#475569",
    )
    ax.annotate(
        f"Observed drain: {stats['drain_ms']:.0f} ms\n"
        f"{stats['start_kb']:.0f} KB → {stats['end_kb']:.0f} KB\n"
        f"burst-slope ≈ {stats['burst_slope_mbps']:.0f} Mbps",
        xy=(stats["drain_ms"] * 0.55, (stats["start_kb"] + stats["end_kb"]) * 0.45),
        xytext=(stats["drain_ms"] * 0.76, max(350.0, stats["start_kb"] * 0.56)),
        arrowprops=dict(arrowstyle="->", color="#D1495B", linewidth=1.8),
        fontsize=13,
        fontweight="bold",
        color="#0F172A",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#D1495B", alpha=0.95),
    )
    ax.text(
        0.03,
        0.12,
        f"Scheduler context: MCS p50={ul.p50_mcs:.0f}, p95={ul.p95_mcs:.0f}; "
        f"PRB p50={ul.p50_rb_size:.0f}; TBS p50={tbs_p50:.0f} B",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color="#0F172A",
        bbox=dict(boxstyle="square,pad=0.35", facecolor="white", edgecolor="#CBD5E1", alpha=0.98),
    )

    ax.set_xlim(-10, max(125.0, stats["drain_ms"] + 25.0))
    ax.set_ylim(0, max(1200.0, decay_win["kb"].max() * 1.12))
    ax.set_xlabel("Time within selected burst window (ms)")
    ax.set_ylabel("UE RLC LCID4 occupancy (KiB)")
    _save(fig, "track1_oai_observed_rlc_drain")


def _optimized_closed_loop_oai_summary() -> Dict[str, float]:
    """Summary for the optimized closed-loop OAI comparator."""
    metrics = pd.read_csv(CLOSED_LOOP_OAI_DEFAULT106_METRICS)
    received = pd.to_numeric(metrics["result_received"], errors="coerce").fillna(0.0)
    send_wall = pd.to_numeric(metrics["t_front_send_wall_s"], errors="coerce")
    send = send_wall.dropna()
    actual_fps = float((len(send) - 1) / (send.max() - send.min())) if len(send) > 1 else float("nan")
    summary_path = (
        Path("abiodun/metrics_logs/carla_oai_ttracer")
        / CLOSED_LOOP_RUN_GROUP
        / "CARLA10_OAI_TTRACER_SUMMARY.csv"
    )
    summary_csv = pd.read_csv(summary_path).iloc[0] if summary_path.exists() else None
    grant_path = (
        Path("abiodun/metrics_logs/scenesense_ttracer")
        / CLOSED_LOOP_RUN_GROUP
        / "ue/analysis/nrue_grant_summary.csv"
    )
    if grant_path.exists():
        grant_summary = pd.read_csv(grant_path)
        ul = grant_summary[grant_summary["direction_label"].eq("ul")].iloc[0]
    else:
        ul = pd.Series(dtype=float)

    detailed_cols = {
        "capture_to_backbone_input_ms",
        "model_preprocess_ms",
        "front_backbone_ms",
        "feature_serialize_ms",
        "send_call_ms",
        "t_front_send_wall_s",
        "t_edge_recv_wall_s",
    }
    has_detailed_timing = detailed_cols.issubset(metrics.columns) and _q(
        metrics["capture_to_backbone_input_ms"], 0.50
    ) == _q(metrics["capture_to_backbone_input_ms"], 0.50)
    if has_detailed_timing:
        capture_to_backbone = pd.to_numeric(
            metrics["capture_to_backbone_input_ms"], errors="coerce"
        )
        front_backbone = pd.to_numeric(metrics["front_backbone_ms"], errors="coerce")
        feature_serialize = pd.to_numeric(metrics["feature_serialize_ms"], errors="coerce")
        front_feature_build = capture_to_backbone + front_backbone + feature_serialize
        uplink_wall = (
            pd.to_numeric(metrics["t_edge_recv_wall_s"], errors="coerce")
            - pd.to_numeric(metrics["t_front_send_wall_s"], errors="coerce")
        ) * 1000.0
        uplink_wall = uplink_wall.where(uplink_wall >= 0.0)
        downlink = pd.to_numeric(metrics["result_send_to_recv_ms_perf"], errors="coerce")
        back = pd.to_numeric(metrics["back_ms"], errors="coerce")
        rtt = pd.to_numeric(metrics["round_trip_result_recv_ms"], errors="coerce")
        capture_to_result = (
            pd.to_numeric(metrics["capture_to_front_send_ms"], errors="coerce") + rtt
        )
        timing_schema = "detailed"
        front_p50_ms = _q(front_feature_build, 0.50)
        feature_uplink_p50_ms = _q(uplink_wall, 0.50)
        back_p50_ms = _q(back, 0.50)
        downlink_p50_ms = _q(downlink, 0.50)
        rtt_p50_ms = _q(rtt, 0.50)
        rtt_p95_ms = _q(rtt, 0.95)
        capture_to_result_p50_ms = _q(capture_to_result, 0.50)
    else:
        if summary_csv is None:
            raise FileNotFoundError(
                f"closed-loop detailed timing is absent and summary CSV is missing: {summary_path}"
            )
        timing_schema = "legacy"
        front_p50_ms = float(summary_csv["front_ms_p50"])
        feature_uplink_p50_ms = float(summary_csv["feature_upload_payload_handling_ms_p50"])
        back_p50_ms = float(summary_csv["back_ms_p50"])
        downlink_p50_ms = float(summary_csv["downlink_ms_p50"])
        rtt_p50_ms = float(summary_csv["rtt_recv_ms_p50"])
        rtt_p95_ms = float(summary_csv["rtt_recv_ms_p95"])
        capture_to_result_p50_ms = front_p50_ms + rtt_p50_ms

    return {
        "sent_frames": float(len(metrics)),
        "received_frames": float(received.sum()),
        "delivery_pct": float(100.0 * received.sum() / len(metrics)),
        "actual_send_fps": actual_fps,
        "payload_kib_p50": float(_q(metrics["feature_payload_bytes"], 0.50) / 1024.0),
        "rtt_p50_ms": float(rtt_p50_ms),
        "rtt_p95_ms": float(rtt_p95_ms),
        "front_p50_ms": float(front_p50_ms),
        "back_p50_ms": float(back_p50_ms),
        "feature_uplink_p50_ms": float(feature_uplink_p50_ms),
        "downlink_p50_ms": float(downlink_p50_ms),
        "capture_to_result_p50_ms": float(capture_to_result_p50_ms),
        "timing_schema": timing_schema,
        "scheduled_mbps": float(ul.get("scheduled_mbps", float("nan"))),
        "mcs_avg": float(ul.get("avg_mcs", float("nan"))),
        "mcs_p50": float(ul.get("p50_mcs", float("nan"))),
        "mcs_p95": float(ul.get("p95_mcs", float("nan"))),
        "prb_p50": float(ul.get("p50_rb_size", float("nan"))),
    }


def write_markdown(summary: pd.DataFrame, burst_summary: Dict[str, float] | None = None) -> None:
    out = Path("abiodun/uplink_only_spatial_map_pipeline/TRACK1_OAI_DEFAULT106_RESULTS.md")
    oai = summary[summary["condition"] == "OAI default 106PRB"].iloc[0]
    loop = summary[summary["condition"] == "Ideal loopback"].iloc[0]
    grant_summary = pd.read_csv(TTRACER_DIR / "ue" / "analysis" / "nrue_grant_summary.csv")
    ul = grant_summary[grant_summary["direction_label"] == "ul"].iloc[0]
    closed = _optimized_closed_loop_oai_summary()
    layer_md = (TTRACER_DIR / "layer_latency" / "uplink_layer_latency.md").read_text()
    layer_lines = []
    for needle in [
        "LCID 4 (data bearer)",
        "SDU drain:",
        "RLC mean queueing delay",
        "grant PRB:",
        "grant MCS:",
        "BSR reports:",
        "SNR dB:",
        "UE PDCP-ingress -> gNB PDCP-deliver",
        "RLC queue-wait is",
    ]:
        for line in layer_md.splitlines():
            if needle in line:
                layer_lines.append(line)
                break
    md = f"""# Track 1 OAI default-106PRB uplink-only results

Date: 2026-07-29/30

Run group: `{RUN_GROUP}`

This run uses the Track-1 uplink-only pipeline: CARLA/front split features go to the edge tail and are published toward the spatial-map side. No detection result is returned to the car.

## Configuration

- OAI: default adaptive MCS, 106 PRB, default 7DL/2UL TDD setup from `gnb.sa.band78.fr1.106PRB.usrpb210.conf`
- UE launch: `-r 106 -C 3619200000`
- Model/knob: no-AE baseline, ROI 0, 200k radar PPS, zstd feature transport, fast radar rasterizer
- Target FPS: 10 FPS, duration budget 130 s
- Traffic: normal Town10 drivable route with the corrected traffic count
- Map compute: not measured in this run; for reporting we add an explicit `+30 ms assumed map compute` row

## Main comparison

| Condition | Sent | Processed | Delivery | Actual send FPS | Uplink payload p50 | Chunks p50 | Sensor prep p50/p95 | Front model p50/p95 | UDP send p50/p95 | Uplink transport p50/p95 | Tail p50/p95 | Capture→tail p50/p95 | Capture→map with +30 ms p50/p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ideal loopback | {int(loop.sent_frames)} | {int(loop.processed_frames)} | {loop.delivery_pct:.1f}% | {loop.actual_send_fps:.2f} | {loop.uplink_payload_p50_kib:.1f} KiB | {loop.uplink_chunks_p50:.0f} | {loop.sensor_prep_p50_ms:.1f}/{loop.sensor_prep_p95_ms:.1f} ms | {loop.front_model_p50_ms:.1f}/{loop.front_model_p95_ms:.1f} ms | {loop.send_call_p50_ms:.1f}/{loop.send_call_p95_ms:.1f} ms | {loop.uplink_transport_p50_ms:.1f}/{loop.uplink_transport_p95_ms:.1f} ms | {loop.tail_p50_ms:.1f}/{loop.tail_p95_ms:.1f} ms | {loop.capture_to_tail_p50_ms:.1f}/{loop.capture_to_tail_p95_ms:.1f} ms | {loop.capture_to_map_plus30_p50_ms:.1f}/{loop.capture_to_map_plus30_p95_ms:.1f} ms |
| OAI default 106PRB | {int(oai.sent_frames)} | {int(oai.processed_frames)} | {oai.delivery_pct:.1f}% | {oai.actual_send_fps:.2f} | {oai.uplink_payload_p50_kib:.1f} KiB | {oai.uplink_chunks_p50:.0f} | {oai.sensor_prep_p50_ms:.1f}/{oai.sensor_prep_p95_ms:.1f} ms | {oai.front_model_p50_ms:.1f}/{oai.front_model_p95_ms:.1f} ms | {oai.send_call_p50_ms:.1f}/{oai.send_call_p95_ms:.1f} ms | {oai.uplink_transport_p50_ms:.1f}/{oai.uplink_transport_p95_ms:.1f} ms | {oai.tail_p50_ms:.1f}/{oai.tail_p95_ms:.1f} ms | {oai.capture_to_tail_p50_ms:.1f}/{oai.capture_to_tail_p95_ms:.1f} ms | {oai.capture_to_map_plus30_p50_ms:.1f}/{oai.capture_to_map_plus30_p95_ms:.1f} ms |

Notes:

- `Sensor prep` is measured from the frame timestamp available at the front process to backbone input, excluding model preprocessing. It includes the front-side sensor packaging/rasterization path that contributes to spatial-map staleness.
- `Uplink transport` is `front_to_edge_ms - send_call_ms`; the raw `front_to_edge_ms` includes the UDP send call because the timestamp is taken immediately before the send.
- CARLA producer cadence (`sync_world_tick_ms`, `camera_frame_wait_ms`) is tracked separately. It constrains actual FPS, but it is not folded into `capture→tail` because the current capture timestamp is placed after camera/radar data are available to the client.
- The OAI edge no-return CSV records `uplink_payload_bytes=0`; therefore payload is taken from the front send-events CSV.

## OAI radio / queue summary

- Scheduled UL throughput: {ul.scheduled_mbps:.1f} Mbps
- Average / p50 / p95 MCS: {ul.avg_mcs:.1f} / {ul.p50_mcs:.0f} / {ul.p95_mcs:.0f}
- Average / p50 / p95 PRB allocation: {ul.avg_rb_size:.1f} / {ul.p50_rb_size:.0f} / {ul.p95_rb_size:.0f}
- Retransmission grant rate in this trace: {ul.retx_rate:.3f}

Cross-layer notes from `uplink_layer_latency.md`:

{chr(10).join(layer_lines)}

## Optimized closed-loop OAI comparator

Both rows below use default OAI 106PRB / 7DL-2UL, no-AE, ROI 0, per-channel uint8, zstd, 200k radar PPS, corrected drivable route, and the fast radar rasterizer.

| Path | Vehicle waits for result? | Sent / received | Delivery | Actual send FPS | Payload p50 | 20 s idle bins | Uplink/feature handling p50 | Result RTT p50/p95 | UL MCS avg / p50 / p95 | Scheduled UL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Track-1 uplink-only OAI | no | {int(oai.sent_frames)} / {int(oai.processed_frames)} | {oai.delivery_pct:.1f}% | {oai.actual_send_fps:.2f} | {oai.uplink_payload_p50_kib:.1f} KiB | {100.0 - burst_summary["track1_example20s_app_active_pct"]:.0f}% | {oai.uplink_transport_p50_ms:.1f} ms | n/a | {ul.avg_mcs:.1f} / {ul.p50_mcs:.0f} / {ul.p95_mcs:.0f} | {ul.scheduled_mbps:.1f} Mbps |
| Optimized closed-loop OAI | yes | {int(closed["sent_frames"])} / {int(closed["received_frames"])} | {closed["delivery_pct"]:.1f}% | {closed["actual_send_fps"]:.2f} | {closed["payload_kib_p50"]:.1f} KiB | {100.0 - burst_summary["closed_loop_example20s_app_active_pct"]:.0f}% | {closed["feature_uplink_p50_ms"]:.1f} ms | {closed["rtt_p50_ms"]:.1f}/{closed["rtt_p95_ms"]:.1f} ms | {closed["mcs_avg"]:.1f} / {closed["mcs_p50"]:.0f} / {closed["mcs_p95"]:.0f} | {closed["scheduled_mbps"]:.1f} Mbps |

## 100 ms traffic-shape check

{(
f'''- Median compressed feature frame: {burst_summary["frame_mbit_p50"]:.2f} Mbit. One full frame in a 100 ms bin is therefore {burst_summary["frame_mbit_p50"] / 0.1:.1f} Mbps equivalent.
- Track-1 OAI uplink-only app offered data appears in {burst_summary["app_active_bins_pct"]:.1f}% of active 100 ms bins. The optimized closed-loop **OAI** no-AE run appears in {burst_summary["closed_loop_app_active_bins_pct"]:.1f}% of comparable 100 ms bins.
- In the displayed 20 s zoom window, Track-1 OAI is active in {burst_summary["track1_example20s_app_active_pct"]:.1f}% of 100 ms bins, while closed-loop OAI is active in {burst_summary["closed_loop_example20s_app_active_pct"]:.1f}%.
- Nonzero Track-1 app bins are usually one frame ({burst_summary["app_nonzero_bin_mbit_p50"]:.2f} Mbit p50), with occasional two-frame bins ({burst_summary["app_nonzero_bin_mbit_max"]:.2f} Mbit max in the active window).
- MAC scheduling is active in {burst_summary["mac_active_bins_pct"]:.1f}% of active bins and RLC dequeue in {burst_summary["rlc_active_bins_pct"]:.1f}%. Median drain over all active-window bins is {burst_summary["rlc_mbit_100ms_p50_all"]:.2f} Mbit/100 ms, or {burst_summary["rlc_mbps_equiv_p50_all"]:.1f} Mbps equivalent.
- RLC occupancy-drain view: one clean observed burst drains from {burst_summary["representative_rlc_start_kib"]:.0f} KiB to {burst_summary["representative_rlc_end_kib"]:.0f} KiB in {burst_summary["representative_rlc_one_frame_drain_ms"]:.0f} ms, with burst-slope about {burst_summary["representative_rlc_burst_slope_mbps"]:.0f} Mbps.
- Interpretation: this is now an optimized OAI-vs-OAI traffic-shape comparison. Track-1 removes the result-return wait from the vehicle, while the optimized closed-loop OAI run still shows the return-wait/timeout cadence that makes feature bursts sparse.
''' if burst_summary else '- See the 100 ms plot for app/radio microburst behavior.'
)}

## Interpretation

Track 1 behaves differently from the earlier OAI closed-loop return-to-car deployment. The OAI traffic-shape panel now compares Track-1 uplink-only OAI against closed-loop OAI, so it isolates the effect of removing result-return waiting from the vehicle-side pacing. The median Track-1 OAI front-to-edge transport-only time is now about {oai.uplink_transport_p50_ms:.1f} ms, not the ~200 ms closed-loop symptom.

However, the 1 MB no-AE feature stream is still close to or above the sustained uplink drain rate. The front offers roughly one 1 MB feature frame every ~140 ms in this run, while the measured RLC/air drain is about 38--42 Mbps. That creates BSR/RLC backlog bursts and explains why capture→tail rises from loopback's {loop.capture_to_tail_p50_ms:.1f} ms p50 to OAI's {oai.capture_to_tail_p50_ms:.1f} ms p50.

Reliability is the main caveat: edge processed {int(oai.processed_frames)} of {int(oai.sent_frames)} frames ({oai.delivery_pct:.1f}%). Edge queue drops were 0, but UDP partial-message drops reached {oai.udp_partial_messages_last:.0f}; that points to incomplete multi-chunk UDP reassembly over OAI rather than tail/map compute saturation.

## Plots

- `plots/track1_oai_default106/track1_latency_breakdown_loopback_vs_oai.pdf`
- `plots/track1_oai_default106/track1_oai_traffic_rates_1s.pdf`
- `plots/track1_oai_default106/track1_oai_radio_scheduler_comparison.pdf`
- `plots/track1_oai_default106/track1_oai_delivery_reassembly.pdf`
- `plots/track1_oai_default106/track1_oai_100ms_volume_drain_backlog.pdf`
- `plots/track1_oai_default106/track1_oai_observed_rlc_drain.pdf`

## Next actions

1. Keep this default-OAI uplink-only result as the Track-1 baseline.
2. Repeat Track 1 with reduced payload knobs to test whether backlog and UDP partial-frame loss improve.
3. Add a real map-worker timing path later; until then report map compute as an explicit assumed add-on, not a measured latency.
"""
    out.write_text(md)


def main() -> None:
    global OUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    OUT_DIR = Path(args.out_dir)
    _style()

    rows = [
        _summarize_path("Ideal loopback", LOOPBACK_RUN_DIR),
        _summarize_path("OAI default 106PRB", OAI_RUN_DIR),
    ]
    summary = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_DIR / "track1_oai_default106_summary.csv", index=False)

    oai_merged, oai_edge, oai_front = _merged_metrics(OAI_RUN_DIR)
    plot_latency(summary)
    plot_traffic_rates(oai_front)
    plot_radio_backlog()
    plot_delivery(oai_front, oai_edge)
    burst_summary = plot_100ms_volume_and_drain(oai_front)
    plot_track1_observed_rlc_drain(burst_summary)
    pd.DataFrame([burst_summary]).to_csv(OUT_DIR / "track1_oai_100ms_burst_summary.csv", index=False)
    write_markdown(summary, burst_summary)
    print(summary.to_string(index=False))
    print(f"Wrote plots to {OUT_DIR}")
    print("Wrote abiodun/uplink_only_spatial_map_pipeline/TRACK1_OAI_DEFAULT106_RESULTS.md")


if __name__ == "__main__":
    main()
