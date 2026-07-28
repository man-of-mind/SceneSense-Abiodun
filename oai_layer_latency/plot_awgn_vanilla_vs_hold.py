#!/usr/bin/env python3
"""Summarize the RFsim AWGN vanilla-vs-hold MCS diagnostic runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


AB = Path(__file__).resolve().parents[1]
PLOTS = AB / "oai_layer_latency" / "plots"

RUNS = {
    "Vanilla\nOAI": AB
    / "metrics_logs"
    / "carla_oai_ttracer"
    / "downlink_oai_bw273_awgn_vanilla_fps10_awgn273_short_20260727_194554_vanilla",
    "Hold few\nsamples": AB
    / "metrics_logs"
    / "carla_oai_ttracer"
    / "downlink_oai_bw273_awgn_hold_fps10_awgn273_short_20260727_194554_hold",
}

SUMMARY_CSV = PLOTS / "awgn_vanilla_vs_hold_summary.csv"

APP_METRICS = {
    "Vanilla\nOAI": AB
    / "downlink_latency_fps"
    / "runs"
    / "oai_bw273_awgn_vanilla"
    / "fps_10_awgn273_short_20260727_194554_vanilla"
    / "streams"
    / "downlink_oai_bw273_awgn_vanilla_fps10_awgn273_short_20260727_194554_vanilla_metrics.csv",
    "Hold few\nsamples": AB
    / "downlink_latency_fps"
    / "runs"
    / "oai_bw273_awgn_hold"
    / "fps_10_awgn273_short_20260727_194554_hold"
    / "streams"
    / "downlink_oai_bw273_awgn_hold_fps10_awgn273_short_20260727_194554_hold_metrics.csv",
}


def load_summary() -> pd.DataFrame:
    rows = []
    for label, cap in RUNS.items():
        s = pd.read_csv(cap / "CARLA10_OAI_TTRACER_SUMMARY.csv").iloc[0]
        rows.append(
            {
                "label": label,
                "delivery_pct": 100 * s["delivery"],
                "capture_to_result_ms": s["front_ms_p50"] + s["rtt_recv_ms_p50"],
                "uplink_ms": s["feature_upload_payload_handling_ms_p50"],
                "rtt_ms": s["rtt_recv_ms_p50"],
                "mcs_p50": s["ul_avg_mcs_p50_window"],
                "mcs_p95": s["ul_p95_mcs_p50_window"],
                "sched_mbps_p50": s["ul_sched_mbps_p50"],
                "sched_mbps_p95": s["ul_sched_mbps_p95"],
                "retx_pct": 100 * s["ul_retx_rate_mean"],
                "snr_db_p50": s["gnb_pusch_snr_db_p50"],
                "feature_kb": s["feature_kb_p50"],
            }
        )
    return pd.DataFrame(rows)


def make_summary_plot(df: pd.DataFrame) -> None:
    colors = ["#e45756", "#2ca02c"]
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 4.1))

    ax = axs[0]
    x = range(len(df))
    ax.bar([i - 0.18 for i in x], df["capture_to_result_ms"], width=0.36, color="#4c78a8", label="capture→result")
    ax.bar([i + 0.18 for i in x], df["uplink_ms"], width=0.36, color="#f58518", label="uplink handling")
    ax.set_xticks(list(x), df["label"], fontweight="bold")
    ax.set_ylabel("Median latency (ms)", fontweight="bold")
    ax.set_title("Latency improved", fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)

    ax = axs[1]
    ax.bar(df["label"], df["mcs_p50"], color=colors)
    ax.set_ylabel("Median MCS index", fontweight="bold")
    ax.set_title("MCS stayed higher", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    for i, v in enumerate(df["mcs_p50"]):
        ax.text(i, v + 0.6, f"{v:.1f}", ha="center", fontweight="bold")
    ax.set_ylim(0, max(df["mcs_p50"]) + 5)

    ax = axs[2]
    ax.bar([i - 0.18 for i in x], df["sched_mbps_p50"], width=0.36, color="#54a24b", label="scheduled Mbps")
    ax2 = ax.twinx()
    ax2.bar([i + 0.18 for i in x], df["retx_pct"], width=0.36, color="#b279a2", label="retx rate")
    ax.set_xticks(list(x), df["label"], fontweight="bold")
    ax.set_ylabel("Median scheduled UL Mbps", fontweight="bold")
    ax2.set_ylabel("Mean retransmission rate (%)", fontweight="bold")
    ax.set_title("Throughput rose, retx also rose", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, fontsize=9, loc="upper left")

    fig.suptitle("RFsim AWGN: vanilla OAI vs hold-few-samples MCS logic", fontweight="bold", y=1.04)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS / f"awgn_vanilla_vs_hold_summary.{ext}", dpi=220, bbox_inches="tight")


def make_timeseries_plot() -> None:
    fig, axs = plt.subplots(2, 1, figsize=(10.8, 6.0), sharex=True)
    styles = {
        "Vanilla\nOAI": ("#e45756", "Vanilla OAI"),
        "Hold few\nsamples": ("#2ca02c", "Hold few samples"),
    }

    for label, cap in RUNS.items():
        color, clean = styles[label]
        w = pd.read_csv(cap / "nrue_ul_grant_windows_compact.csv")
        w = w[w["scheduled_mbps"] > 0.1].copy()
        axs[0].plot(w["window_start_s"], w["avg_mcs"], color=color, lw=2.0, label=clean)
        axs[1].plot(w["window_start_s"], w["retx_rate"] * 100, color=color, lw=2.0, label=clean)

    axs[0].set_ylabel("Avg MCS / 1 s window", fontweight="bold")
    axs[0].set_title("AWGN active-window MCS", fontweight="bold")
    axs[1].set_ylabel("Retx rate / 1 s window (%)", fontweight="bold")
    axs[1].set_title("AWGN active-window retransmission pressure", fontweight="bold")
    axs[1].set_xlabel("Elapsed time (s)", fontweight="bold")
    for ax in axs:
        ax.grid(True, axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS / f"awgn_vanilla_vs_hold_timeseries.{ext}", dpi=220, bbox_inches="tight")


def _window_lcid4_mbps(cap: Path) -> pd.DataFrame:
    """Return 1-second successful LCID4 receive bytes at the gNB as Mbps."""

    path = AB / "metrics_logs" / "scenesense_ttracer" / cap.name / "gnb" / "csv" / "GNB_MAC_LCID_UL.csv"
    df = pd.read_csv(path)
    df = df[df["lcid"] == 4].copy()
    ts = pd.to_datetime("2026-07-27 " + df["time"].astype(str), errors="coerce")
    start = ts.iloc[0]
    df["elapsed_s"] = (ts - start).dt.total_seconds()
    df["window_s"] = df["elapsed_s"].astype(int)
    win = df.groupby("window_s", as_index=False)["data_size"].sum()
    # GNB_MAC_LCID_UL.data_size is already in bits; it matches
    # ENB_RLC_MAC_UL.length_bytes * 8 for LCID 4.
    win["lcid4_rx_mbps"] = win["data_size"] / 1e6
    return win


def make_snr_mcs_retx_plot() -> None:
    fig, axs = plt.subplots(3, 1, figsize=(11.2, 8.0), sharex=True)
    styles = {
        "Vanilla\nOAI": ("#e45756", "Vanilla OAI"),
        "Hold few\nsamples": ("#2ca02c", "Hold few samples"),
    }

    for label, cap in RUNS.items():
        color, clean = styles[label]
        snr = pd.read_csv(cap / "gnb_pusch_power_compact.csv")
        snr["window_s"] = snr["t_norm"].astype(int)
        snr_win = (
            snr.groupby("window_s")["snr_db"]
            .agg(snr_p10=lambda s: s.quantile(0.10), snr_p50="median", snr_p90=lambda s: s.quantile(0.90))
            .reset_index()
        )
        win = pd.read_csv(cap / "nrue_ul_grant_windows_compact.csv")
        active = win[win["scheduled_mbps"] > 0.1].copy()

        axs[0].plot(snr_win["window_s"], snr_win["snr_p50"], lw=2.2, color=color, alpha=0.95, label=f"{clean}: median")
        axs[0].fill_between(
            snr_win["window_s"].to_numpy(),
            snr_win["snr_p10"].to_numpy(),
            snr_win["snr_p90"].to_numpy(),
            color=color,
            alpha=0.12,
            linewidth=0,
        )
        axs[1].plot(active["window_start_s"], active["avg_mcs"], lw=2.2, color=color, label=clean)
        axs[2].plot(active["window_start_s"], active["retx_rate"] * 100, lw=2.0, color=color, label=clean)

    axs[0].set_title("AWGN gNB PUSCH SNR over time (1 s median, p10–p90 band)", fontweight="bold")
    axs[0].set_ylabel("SNR (dB)", fontweight="bold")
    axs[1].set_title("AWGN scheduled MCS over active UL windows", fontweight="bold")
    axs[1].set_ylabel("Avg MCS / 1 s", fontweight="bold")
    axs[2].set_title("AWGN retransmission pressure over time", fontweight="bold")
    axs[2].set_ylabel("Retx rate / 1 s (%)", fontweight="bold")
    axs[2].set_xlabel("Elapsed time (s)", fontweight="bold")

    for ax in axs:
        ax.grid(True, axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS / f"awgn_snr_mcs_retx_timeseries.{ext}", dpi=220, bbox_inches="tight")


def make_drain_latency_plot() -> None:
    fig, axs = plt.subplots(3, 1, figsize=(11.2, 8.4), sharex=False)
    styles = {
        "Vanilla\nOAI": ("#e45756", "Vanilla OAI"),
        "Hold few\nsamples": ("#2ca02c", "Hold few samples"),
    }

    delivered_rows = []
    for label, cap in RUNS.items():
        color, clean = styles[label]
        grant = pd.read_csv(cap / "nrue_ul_grant_windows_compact.csv")
        active = grant[grant["scheduled_mbps"] > 0.1].copy()
        lcid = _window_lcid4_mbps(cap)
        app = pd.read_csv(APP_METRICS[label])
        app["received_bool"] = app["result_received"].astype(str).str.lower().eq("true")
        rec = app[app["received_bool"]].copy()
        miss = app[~app["received_bool"]].copy()

        axs[0].plot(active["window_start_s"], active["scheduled_mbps"], color=color, lw=1.8, alpha=0.65, ls="--", label=f"{clean}: scheduled TBS")
        axs[0].plot(lcid["window_s"], lcid["lcid4_rx_mbps"], color=color, lw=2.2, label=f"{clean}: decoded LCID4")
        axs[1].scatter(rec["elapsed_s"], rec["round_trip_result_recv_ms"], s=13, color=color, alpha=0.8, label=f"{clean}: delivered")
        if not miss.empty:
            axs[1].scatter(miss["elapsed_s"], [1500] * len(miss), s=10, color=color, alpha=0.22, marker="x", label=f"{clean}: timeout")
        axs[2].plot(active["window_start_s"], active["retx_rate"] * 100, color=color, lw=2.0, label=clean)

        delivered_rows.append(
            {
                "label": clean,
                "decoded_lcid4_mbps_p50": lcid["lcid4_rx_mbps"].median(),
                "decoded_lcid4_mbps_p95": lcid["lcid4_rx_mbps"].quantile(0.95),
                "scheduled_mbps_active_p50": active["scheduled_mbps"].median(),
                "scheduled_mbps_active_p95": active["scheduled_mbps"].quantile(0.95),
                "delivered_rtt_ms_p50": rec["round_trip_result_recv_ms"].median(),
                "delivered_rtt_ms_p95": rec["round_trip_result_recv_ms"].quantile(0.95),
                "result_wait_ms_p50": rec["result_wait_ms"].median(),
                "downlink_ms_p50": rec["result_send_to_recv_ms_perf"].median(),
                "timeouts": int(len(miss)),
            }
        )

    axs[0].set_title("AWGN drain proxy: scheduled TBS vs decoded LCID4 bits", fontweight="bold")
    axs[0].set_ylabel("Mbps / 1 s", fontweight="bold")
    axs[1].set_title("Application latency over time: delivered frames and timeouts", fontweight="bold")
    axs[1].set_ylabel("Post-send RTT / timeout (ms)", fontweight="bold")
    axs[1].set_ylim(0, 1600)
    axs[2].set_title("Retransmissions align with unstable effective drain", fontweight="bold")
    axs[2].set_ylabel("Retx rate / 1 s (%)", fontweight="bold")
    axs[2].set_xlabel("Elapsed time (s)", fontweight="bold")

    for ax in axs:
        ax.grid(True, axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS / f"awgn_drain_latency_retx_timeseries.{ext}", dpi=220, bbox_inches="tight")
    pd.DataFrame(delivered_rows).to_csv(PLOTS / "awgn_drain_latency_retx_summary.csv", index=False)


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    df = load_summary()
    df.to_csv(SUMMARY_CSV, index=False)
    make_summary_plot(df)
    make_timeseries_plot()
    make_snr_mcs_retx_plot()
    make_drain_latency_plot()


if __name__ == "__main__":
    main()
