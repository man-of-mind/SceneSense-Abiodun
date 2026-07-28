#!/usr/bin/env python3
"""Generate focused plots for the 2026-07-27 OAI BLER/OLLA hold-MCS experiment.

This compares the same no-AE / uint8 / ROI0 / zstd / ~1 MB CARLA payload across:

- ideal loopback,
- vanilla adaptive OAI 273PRB,
- patched adaptive OAI 273PRB with SCENESENSE_HOLD_MCS_FEW_SAMPLES=1,
- fixed MCS28 diagnostic OAI 273PRB.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "downlink_latency_fps" / "runs"
TRACE = ROOT / "metrics_logs" / "scenesense_ttracer"
PLOTS = ROOT / "oai_layer_latency" / "plots"


@dataclass(frozen=True)
class Condition:
    key: str
    label: str
    metrics: Path
    trace_group: str | None = None
    color: str = "#4c78a8"


CONDITIONS = [
    Condition(
        "loopback",
        "Ideal loopback",
        RUNS
        / "ideal_loopback/fps_10_drivable_rerun_20260722_loopback/streams/"
        / "downlink_ideal_loopback_fps10_drivable_rerun_20260722_loopback_metrics.csv",
        None,
        "#6f6f6f",
    ),
    Condition(
        "oai_vanilla",
        "OAI 273PRB\nvanilla adaptive",
        RUNS
        / "oai_bw273_mu1_ttracer/fps_10_drivable_rerun_20260722_bw273/streams/"
        / "downlink_oai_bw273_mu1_ttracer_fps10_drivable_rerun_20260722_bw273_metrics.csv",
        "downlink_oai_bw273_mu1_ttracer_fps10_drivable_rerun_20260722_bw273",
        "#e45756",
    ),
    Condition(
        "oai_hold",
        "OAI 273PRB\nhold few-sample MCS",
        RUNS
        / "oai_bw273_mu1_ttracer/fps_10_holdmcs_noae_bw273_fulltrace_20260727_1755/streams/"
        / "downlink_oai_bw273_mu1_ttracer_fps10_holdmcs_noae_bw273_fulltrace_20260727_1755_metrics.csv",
        "downlink_oai_bw273_mu1_ttracer_fps10_holdmcs_noae_bw273_fulltrace_20260727_1755",
        "#2ca02c",
    ),
    Condition(
        "oai_fixed28",
        "OAI 273PRB\nfixed MCS28",
        RUNS
        / "oai_bw273_mu1_ttracer/fps_10_forcemcs28_20260722_201150/streams/"
        / "downlink_oai_bw273_mu1_ttracer_fps10_forcemcs28_20260722_201150_metrics.csv",
        "downlink_oai_bw273_mu1_ttracer_fps10_forcemcs28_20260722_201150",
        "#7b61b3",
    ),
]


def save(fig: plt.Figure, stem: str) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(PLOTS / f"{stem}.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def received_mask(df: pd.DataFrame) -> pd.Series:
    return df["result_received"].astype(str).str.lower().isin({"true", "1", "yes"})


def metric_summary(cond: Condition) -> dict[str, float | str]:
    df = pd.read_csv(cond.metrics)
    recv = received_mask(df)
    got = df[recv].copy()
    med = lambda c: float(pd.to_numeric(got[c], errors="coerce").median()) if c in got else np.nan
    mean = lambda c: float(pd.to_numeric(got[c], errors="coerce").mean()) if c in got else np.nan

    rtt = med("round_trip_result_recv_ms")
    back = med("back_ms")
    down = med("result_send_to_recv_ms_wall")
    front = med("front_ms")
    uplink = rtt - back - down
    return {
        "key": cond.key,
        "label": cond.label.replace("\n", " "),
        "rows": len(df),
        "received": int(recv.sum()),
        "delivery_pct": 100.0 * float(recv.mean()),
        "front_ms": front,
        "uplink_ms": uplink,
        "back_ms": back,
        "downlink_ms": down,
        "rtt_ms": rtt,
        "capture_to_result_ms": front + rtt,
        "uplink_payload_kb": mean("feature_payload_bytes") / 1024.0,
        "downlink_payload_kb": mean("result_payload_bytes_estimate") / 1024.0,
    }


def grant_summary(cond: Condition) -> dict[str, float | str]:
    out: dict[str, float | str] = {"key": cond.key}
    if not cond.trace_group:
        return out
    grant_path = TRACE / cond.trace_group / "ue" / "csv" / "NRUE_MAC_DCI_GRANT.csv"
    if grant_path.exists():
        grant = pd.read_csv(grant_path)
        ul = grant[grant["direction"].eq(1)].copy()
        out.update(
            {
                "ul_grants": len(ul),
                "ul_mcs_mean": float(ul["mcs"].mean()) if len(ul) else np.nan,
                "ul_mcs_median": float(ul["mcs"].median()) if len(ul) else np.nan,
                "ul_mcs28_pct": 100.0 * float(ul["mcs"].eq(28).mean()) if len(ul) else np.nan,
                "ul_rb_mean": float(ul["rb_size"].mean()) if len(ul) else np.nan,
                "ul_tbs_kb_mean": float(ul["tbs"].mean()) / 1024.0 if len(ul) else np.nan,
            }
        )
    pusch_path = TRACE / cond.trace_group / "gnb" / "csv" / "GNB_MAC_PUSCH_POWER_CONTROL.csv"
    if pusch_path.exists():
        pusch = pd.read_csv(pusch_path)
        out["gnb_pusch_snr_db_mean"] = float(pusch["snrx10"].mean()) / 10.0
        out["gnb_pusch_phr_mean"] = float(pusch["phr"].mean())
    return out


def load_window(cond: Condition) -> pd.DataFrame | None:
    if not cond.trace_group:
        return None
    path = TRACE / cond.trace_group / "ue" / "analysis" / "nrue_grant_windows.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df["direction_label"].astype(str).str.lower().eq("ul")].copy()
    return df.sort_values("window_start_s")


def make_latency_plot(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    comps = ["front_ms", "uplink_ms", "back_ms", "downlink_ms"]
    labels = ["front", "uplink / RAN", "edge tail", "downlink"]
    colors = ["#8ecae6", "#fb8500", "#90be6d", "#577590"]
    x = np.arange(len(summary))
    bottom = np.zeros(len(summary))
    for comp, label, color in zip(comps, labels, colors):
        vals = summary[comp].to_numpy(float)
        ax.bar(x, vals, bottom=bottom, label=label, color=color, edgecolor="white", linewidth=0.7)
        bottom += vals
    for i, row in summary.iterrows():
        ax.text(
            i,
            row["capture_to_result_ms"] + 9,
            f"{row['capture_to_result_ms']:.0f} ms",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(summary["label"], fontsize=10)
    ax.set_ylim(0, summary["capture_to_result_ms"].max() + 45)
    ax.set_ylabel("Median latency component (ms)", fontweight="bold")
    ax.set_title("No-AE ~1 MB CARLA payload: OAI hold-MCS removes the high uplink latency", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    fig.tight_layout()
    save(fig, "hold_mcs_latency_breakdown")


def make_scheduler_efficiency_plot(summary: pd.DataFrame) -> None:
    rows = summary[summary["key"].isin(["oai_vanilla", "oai_hold", "oai_fixed28"])].copy()
    rows["grant_k"] = rows["ul_grants"] / 1000.0
    short = ["Vanilla\nadaptive", "Hold-MCS\nadaptive", "Fixed\nMCS28"]
    panels = [
        ("ul_mcs_median", "Median UL MCS", ""),
        ("ul_tbs_kb_mean", "Mean TBS per UL grant", "KB/grant"),
        ("grant_k", "UL grant count over run", "k grants"),
        ("uplink_ms", "Median uplink/RAN component", "ms"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.4))
    axes = axes.ravel()
    for ax, (col, title, unit) in zip(axes, panels):
        vals = rows[col].to_numpy(float)
        x = np.arange(len(rows))
        ax.bar(x, vals, color=rows["color"], edgecolor="white", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(short, fontsize=10)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, max(vals) * 1.22 if max(vals) > 0 else 1.0)
        for xx, v in zip(x, vals):
            ax.text(xx, v + max(vals) * 0.035, f"{v:.1f} {unit}".strip(), ha="center", va="bottom", fontsize=9, fontweight="bold")
    fig.suptitle("Scheduler efficiency: hold-MCS behaves like high-MCS without forcing MCS", fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "hold_mcs_scheduler_efficiency")


def make_mcs_timeseries() -> None:
    selected = [c for c in CONDITIONS if c.key in {"oai_vanilla", "oai_hold"}]
    active_by_key: dict[str, tuple[Condition, pd.DataFrame]] = {}
    common_end_s: float | None = None
    for cond in selected:
        df = load_window(cond)
        if df is None or df.empty:
            continue
        active = df[df["scheduled_mbps"] > 0.1].copy()
        if active.empty:
            continue
        active_by_key[cond.key] = (cond, active)
        end_s = float(active["window_start_s"].max())
        common_end_s = end_s if common_end_s is None else min(common_end_s, end_s)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True)
    for cond, active in active_by_key.values():
        if common_end_s is not None:
            active = active[active["window_start_s"] <= common_end_s].copy()
        axes[0].plot(active["window_start_s"], active["avg_mcs"], lw=2.0, color=cond.color, label=cond.label.replace("\n", " "))
        axes[1].plot(active["window_start_s"], active["avg_tbs_bytes"] / 1024.0, lw=2.0, color=cond.color, label=cond.label.replace("\n", " "))
    axes[0].set_ylabel("Avg UL MCS\n(1s window)", fontweight="bold")
    axes[1].set_ylabel("Avg TBS/grant\n(KB)", fontweight="bold")
    axes[1].set_xlabel("Run time (s)", fontweight="bold")
    axes[0].set_ylim(-1, 30)
    if common_end_s is not None:
        axes[1].set_xlim(0, common_end_s)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", frameon=True)
    fig.suptitle("Adaptive scheduler behavior over common active window: vanilla low MCS vs hold-MCS near 28", fontweight="bold")
    fig.tight_layout()
    save(fig, "hold_mcs_mcs_tbs_timeseries")


def main() -> None:
    metric_rows = [metric_summary(c) for c in CONDITIONS]
    grant_rows = [grant_summary(c) for c in CONDITIONS]
    summary = pd.DataFrame(metric_rows).merge(pd.DataFrame(grant_rows), on="key", how="left")
    color_map = {c.key: c.color for c in CONDITIONS}
    summary["color"] = summary["key"].map(color_map)
    summary.to_csv(PLOTS / "hold_mcs_experiment_summary.csv", index=False)
    make_latency_plot(summary)
    make_scheduler_efficiency_plot(summary)
    make_mcs_timeseries()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
