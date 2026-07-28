#!/usr/bin/env python3
"""Plot the AWGN all-metrics rerun for vanilla OAI vs hold-few-samples MCS.

This is intentionally separate from plot_awgn_vanilla_vs_hold.py because the
all-metrics rerun records a richer UE trace profile (`all`) and gNB latency
profile. The purpose is to answer: why does high median MCS under AWGN still
leave >200 ms application latency?
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


AB = Path(__file__).resolve().parents[1]
OUT = AB / "oai_layer_latency" / "plots"

RUNS = {
    "Vanilla OAI": {
        "color": "#e45756",
        "rg": "downlink_oai_bw273_awgn_vanilla_fps10_awgn273_allmetrics_20260727_202430_vanilla",
        "app": AB
        / "downlink_latency_fps/runs/oai_bw273_awgn_vanilla/fps_10_awgn273_allmetrics_20260727_202430_vanilla/streams/downlink_oai_bw273_awgn_vanilla_fps10_awgn273_allmetrics_20260727_202430_vanilla_metrics.csv",
    },
    "Hold few samples": {
        "color": "#2ca02c",
        "rg": "downlink_oai_bw273_awgn_hold_fps10_awgn273_allmetrics_20260727_202430_hold",
        "app": AB
        / "downlink_latency_fps/runs/oai_bw273_awgn_hold/fps_10_awgn273_allmetrics_20260727_202430_hold/streams/downlink_oai_bw273_awgn_hold_fps10_awgn273_allmetrics_20260727_202430_hold_metrics.csv",
    },
}

BRANCH_NAMES = {
    1: "increase\nlow BLER",
    2: "decrease\nhigh BLER",
    3: "decrease\nfew samples",
    4: "hold\ntarget",
}


def run_root(rg: str) -> Path:
    return AB / "metrics_logs" / "scenesense_ttracer" / rg


def cap_root(rg: str) -> Path:
    return AB / "metrics_logs" / "carla_oai_ttracer" / rg


def elapsed_from_time(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime("2026-07-27 " + series.astype(str), errors="coerce")
    return (ts - ts.dropna().iloc[0]).dt.total_seconds()


def one_sec_p95(df: pd.DataFrame, time_col: str, value_col: str, out_col: str) -> pd.DataFrame:
    d = df[[time_col, value_col]].copy()
    d["elapsed_s"] = elapsed_from_time(d[time_col])
    d = d.dropna(subset=["elapsed_s"])
    d["window_s"] = d["elapsed_s"].astype(int)
    return d.groupby("window_s")[value_col].quantile(0.95).reset_index(name=out_col)


def app_metrics(path: Path) -> pd.DataFrame:
    app = pd.read_csv(path)
    app["received_bool"] = app["result_received"].astype(str).str.lower().eq("true")
    return app


def summary_rows() -> pd.DataFrame:
    rows = []
    for label, spec in RUNS.items():
        rg = spec["rg"]
        csum = pd.read_csv(cap_root(rg) / "CARLA10_OAI_TTRACER_SUMMARY.csv").iloc[0]
        layer_md = run_root(rg) / "layer_latency/uplink_layer_latency.md"
        # Parse only the specific analyzer values needed for the plot/table.
        text = layer_md.read_text() if layer_md.exists() else ""
        rlc_wait = float("nan")
        ran_mean = float("nan")
        for line in text.splitlines():
            if "RLC queue-wait (mean" in line and "~" in line:
                rlc_wait = float(line.split("~", 1)[1].split()[0])
            if "UE PDCP-ingress -> gNB PDCP-deliver" in line and "mean=" in line:
                ran_mean = float(line.split("mean=", 1)[1].split()[0])
        bler = pd.read_csv(run_root(rg) / "gnb/csv/GNB_MAC_BLER_MCS_DECISION.csv")
        upd = bler[(bler["direction"] == 1) & (bler["updated"] == 1)]
        high_bler_pct = 100 * (upd["branch"] == 2).mean()
        few_sample_pct = 100 * (upd["branch"] == 3).mean()
        retx_update_pct = 100 * (upd["num_retx"] > 0).mean()
        app = app_metrics(spec["app"])
        rec = app[app["received_bool"]]
        rows.append(
            {
                "label": label,
                "delivery_pct": 100 * csum["delivery"],
                "app_rtt_p50": csum["rtt_recv_ms_p50"],
                "capture_to_result_p50": csum["front_ms_p50"] + csum["rtt_recv_ms_p50"],
                "feature_upload_p50": csum["feature_upload_payload_handling_ms_p50"],
                "downlink_p50": csum["downlink_ms_p50"],
                "mcs_p50": csum["ul_avg_mcs_p50_window"],
                "retx_mean_pct": 100 * csum["ul_retx_rate_mean"],
                "snr_p50": csum["gnb_pusch_snr_db_p50"],
                "rlc_wait_mean_ms": rlc_wait,
                "ran_transit_mean_ms": ran_mean,
                "result_wait_p50": rec["result_wait_ms"].median(),
                "timeouts": int((~app["received_bool"]).sum()),
                "high_bler_update_pct": high_bler_pct,
                "few_sample_update_pct": few_sample_pct,
                "retx_update_pct": retx_update_pct,
            }
        )
    return pd.DataFrame(rows)


def plot_summary_bars(df: pd.DataFrame) -> None:
    colors = [RUNS[x]["color"] for x in df["label"]]
    fig, axs = plt.subplots(1, 4, figsize=(14.5, 4.0))
    panels = [
        ("capture_to_result_p50", "Capture→result p50\n(ms)", "Latency remains high"),
        ("mcs_p50", "Median MCS", "Hold raises MCS"),
        ("retx_update_pct", "BLER updates with\nretx (%)", "Retx evidence rises"),
        ("rlc_wait_mean_ms", "Mean UE RLC\nqueue wait (ms)", "Queue improves but remains"),
    ]
    for ax, (col, ylabel, title) in zip(axs, panels):
        ax.bar(df["label"], df[col], color=colors, width=0.62)
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        for i, v in enumerate(df[col]):
            ax.text(i, v * 1.02 if v > 0 else 0.5, f"{v:.1f}", ha="center", fontweight="bold", fontsize=9)
    fig.suptitle("AWGN all-metrics rerun: high MCS helps, but retransmissions and RLC wait remain", fontweight="bold", y=1.04)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"awgn_allmetrics_summary_bars.{ext}", dpi=240, bbox_inches="tight")


def plot_scheduler_timeseries() -> None:
    fig, axs = plt.subplots(4, 1, figsize=(12.2, 9.0), sharex=True)
    for label, spec in RUNS.items():
        color = spec["color"]
        rg = spec["rg"]
        cap = cap_root(rg)
        root = run_root(rg)

        snr = pd.read_csv(cap / "gnb_pusch_power_compact.csv")
        snr["window_s"] = snr["t_norm"].astype(int)
        snr_w = snr.groupby("window_s")["snr_db"].agg(
            p10=lambda s: s.quantile(0.10),
            p50="median",
            p90=lambda s: s.quantile(0.90),
        ).reset_index()
        grant = pd.read_csv(cap / "nrue_ul_grant_windows_compact.csv")
        active = grant[grant["scheduled_mbps"] > 0.1].copy()
        bler = pd.read_csv(root / "gnb/csv/GNB_MAC_BLER_MCS_DECISION.csv")
        ul_upd = bler[(bler["direction"] == 1) & (bler["updated"] == 1)].copy()
        ul_upd["elapsed_s"] = elapsed_from_time(ul_upd["time"])
        ul_upd["window_s"] = ul_upd["elapsed_s"].astype(int)
        branch = ul_upd.groupby("window_s").agg(
            high_bler_pct=("branch", lambda s: 100 * (s == 2).mean()),
            few_sample_pct=("branch", lambda s: 100 * (s == 3).mean()),
        ).reset_index()
        branch = branch.set_index("window_s").sort_index()
        full_idx = range(int(branch.index.min()), int(branch.index.max()) + 1) if len(branch) else []
        branch = branch.reindex(full_idx).fillna(0.0).rolling(10, min_periods=1).mean().reset_index(names="window_s")

        axs[0].plot(snr_w["window_s"], snr_w["p50"], color=color, lw=2.1, label=label)
        axs[0].fill_between(snr_w["window_s"].to_numpy(), snr_w["p10"].to_numpy(), snr_w["p90"].to_numpy(), color=color, alpha=0.12)
        axs[1].plot(active["window_start_s"], active["avg_mcs"], color=color, lw=2.1, label=label)
        axs[2].plot(active["window_start_s"], active["retx_rate"] * 100, color=color, lw=1.9, label=label)
        axs[3].plot(branch["window_s"], branch["high_bler_pct"], color=color, lw=2.2, label=f"{label}: high-BLER decreases")
        axs[3].plot(branch["window_s"], branch["few_sample_pct"], color=color, lw=1.5, ls="--", alpha=0.72, label=f"{label}: few-sample decreases")

    titles = [
        "gNB PUSCH SNR is comparable in both AWGN runs",
        "Hold-few-samples keeps scheduled MCS much higher",
        "Higher MCS also creates more retransmission pressure",
        "BLER/OLLA update branches (10 s rolling): high-BLER decreases appear under hold-MCS",
    ]
    ylabels = ["SNR (dB)", "Avg MCS / 1 s", "Retx rate / 1 s (%)", "Updated decisions (%)"]
    for ax, title, ylabel in zip(axs, titles, ylabels):
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    axs[-1].set_xlabel("Elapsed time (s)", fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"awgn_allmetrics_scheduler_timeseries.{ext}", dpi=240, bbox_inches="tight")


def plot_queue_grant_timeseries() -> None:
    fig, axs = plt.subplots(4, 1, figsize=(12.2, 9.2), sharex=True)
    grant_ax2 = axs[2].twinx()
    for label, spec in RUNS.items():
        color = spec["color"]
        rg = spec["rg"]
        root = run_root(rg)
        cap = cap_root(rg)
        bsr = pd.read_csv(root / "ue/csv/NRUE_MAC_BSR_STATUS.csv")
        rlc = pd.read_csv(root / "ue/csv/NRUE_MAC_RLC_BUFFER_STATUS.csv")
        rlc4 = rlc[rlc["lcid"] == 4].copy()
        bsr_w = one_sec_p95(bsr, "time", "lcg1_bytes", "bsr_p95")
        rlc_w = one_sec_p95(rlc4, "time", "bytes_in_buffer", "rlc_p95")
        grant = pd.read_csv(cap / "nrue_ul_grant_windows_compact.csv")
        active = grant[grant["scheduled_mbps"] > 0.1].copy()
        app = app_metrics(spec["app"])
        rec = app[app["received_bool"]]
        miss = app[~app["received_bool"]]

        axs[0].plot(bsr_w["window_s"], bsr_w["bsr_p95"] / 1024, color=color, lw=1.9, label=label)
        axs[1].plot(rlc_w["window_s"], rlc_w["rlc_p95"] / 1024, color=color, lw=1.9, label=label)
        axs[2].plot(active["window_start_s"], active["grant_rate_hz"], color=color, lw=1.8, label=f"{label}: grant rate")
        grant_ax2.plot(active["window_start_s"], active["scheduled_mbps"], color=color, lw=1.8, ls="--", alpha=0.78, label=f"{label}: scheduled Mbps")
        axs[3].scatter(rec["elapsed_s"], rec["round_trip_result_recv_ms"], color=color, s=13, alpha=0.75, label=f"{label}: delivered")
        if not miss.empty:
            axs[3].scatter(miss["elapsed_s"], [1500] * len(miss), color=color, s=12, marker="x", alpha=0.20, label=f"{label}: timeout")

    titles = [
        "UE BSR reports whole-frame backlogs (~1 MB peaks)",
        "UE RLC LCID4 occupancy: hold-MCS reduces mean wait but not burst peaks",
        "Grant cadence/effective scheduled drain, not MCS alone, controls queue drain",
        "Application post-send RTT: frame completion waits for tail chunks/retransmissions",
    ]
    ylabels = ["BSR LCG1 p95 / 1 s (KB)", "RLC LCID4 p95 / 1 s (KB)", "Grant rate / 1 s", "RTT / timeout (ms)"]
    for ax, title, ylabel in zip(axs, titles, ylabels):
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        if ax is axs[2]:
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = grant_ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, frameon=False, fontsize=8.2, loc="upper right")
        else:
            ax.legend(frameon=False, fontsize=8.2, loc="upper right")
    grant_ax2.set_ylabel("Scheduled Mbps / 1 s", fontweight="bold")
    grant_ax2.spines["top"].set_visible(False)
    axs[3].set_ylim(0, 1600)
    axs[-1].set_xlabel("Elapsed time (s)", fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"awgn_allmetrics_queue_grant_timeseries.{ext}", dpi=240, bbox_inches="tight")


def write_summary(df: pd.DataFrame) -> None:
    df.to_csv(OUT / "awgn_allmetrics_summary.csv", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = summary_rows()
    write_summary(df)
    plot_summary_bars(df)
    plot_scheduler_timeseries()
    plot_queue_grant_timeseries()


if __name__ == "__main__":
    main()
