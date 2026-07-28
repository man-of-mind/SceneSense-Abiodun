#!/usr/bin/env python3
"""Compare UE RLC drain/queue behavior for TRACTOR vs CARLA split traffic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd


AB = Path(__file__).resolve().parents[1]
OUT = AB / "bursty_traffic" / "plots"
ANALYSIS = AB / "bursty_traffic" / "analysis"


@dataclass(frozen=True)
class RunSpec:
    key: str
    title: str
    run_group: str
    dequeue_path: Path
    rlc_buffer_path: Path
    grant_path: Path


TRACTOR_EMBB = "tractor_replay_bw273_vanilla_embb0303a_off100_60s_tcpdump_20260727_210512"
TRACTOR_URLLC = "tractor_replay_bw273_vanilla_urllc0303_off240_60s_tcpdump_20260727_210842"
CARLA_LAYER = "downlink_oai_bw273_mu1_ttracer_fps10_layerinstr_20260723_145921"

RUNS = [
    RunSpec(
        key="tractor_onedrive",
        title="TRACTOR OneDrive\nreal bursty replay",
        run_group=TRACTOR_EMBB,
        dequeue_path=AB / "metrics_logs" / "scenesense_ttracer" / TRACTOR_EMBB / "ue/csv/NR_RLC_TX_DEQUEUE.csv",
        rlc_buffer_path=AB / "metrics_logs" / "scenesense_ttracer" / TRACTOR_EMBB / "ue/csv/NRUE_MAC_RLC_BUFFER_STATUS.csv",
        grant_path=AB / "metrics_logs" / "scenesense_ttracer" / TRACTOR_EMBB / "ue/analysis/nrue_grant_windows.csv",
    ),
    RunSpec(
        key="tractor_meet",
        title="TRACTOR Google Meet\nreal bursty replay",
        run_group=TRACTOR_URLLC,
        dequeue_path=AB / "metrics_logs" / "scenesense_ttracer" / TRACTOR_URLLC / "ue/csv/NR_RLC_TX_DEQUEUE.csv",
        rlc_buffer_path=AB / "metrics_logs" / "scenesense_ttracer" / TRACTOR_URLLC / "ue/csv/NRUE_MAC_RLC_BUFFER_STATUS.csv",
        grant_path=AB / "metrics_logs" / "scenesense_ttracer" / TRACTOR_URLLC / "ue/analysis/nrue_grant_windows.csv",
    ),
    RunSpec(
        key="carla_split",
        title="CARLA split inference\n≈1 MB feature bursts",
        run_group=CARLA_LAYER,
        dequeue_path=AB / "metrics_logs" / "scenesense_ttracer" / CARLA_LAYER / "ue/csv/NR_RLC_TX_DEQUEUE.csv",
        rlc_buffer_path=AB / "metrics_logs" / "scenesense_ttracer" / CARLA_LAYER / "ue/csv/NRUE_MAC_RLC_BUFFER_STATUS.csv",
        grant_path=AB / "metrics_logs" / "scenesense_ttracer" / CARLA_LAYER / "ue/analysis/nrue_grant_windows.csv",
    ),
]


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def smooth(y: pd.Series, window: int = 3) -> pd.Series:
    if len(y) < window:
        return y
    return y.rolling(window=window, min_periods=1, center=True).median()


def align_active(df: pd.DataFrame, value_col: str, threshold: float) -> pd.DataFrame:
    if df.empty or value_col not in df.columns:
        return df.copy()
    out = df.copy()
    active = out[num(out[value_col]).fillna(0) > threshold]
    if len(active):
        out["t_s"] = out["t_s"] - float(active["t_s"].min())
    return out


def complete_rate_bins(df: pd.DataFrame, duration_s: float, bin_s: float) -> pd.DataFrame:
    n = int(duration_s / bin_s)
    if df.empty:
        return pd.DataFrame({"t_s": [i * bin_s for i in range(n)], "drain_mbps": [0.0] * n})
    idx = (df["t_s"] / bin_s).round().astype(int)
    vals = df.set_index(idx)["drain_mbps"].groupby(level=0).sum().reindex(range(n), fill_value=0.0)
    return pd.DataFrame({"t_s": vals.index.astype(float) * bin_s, "drain_mbps": vals.to_numpy()})


def load_dequeue_rate(path: Path, bin_s: float = 1.0) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["lcid"].astype(str).eq("4")].copy()
    df["pdu_bytes"] = num(df["pdu_bytes"])
    df["t_raw"] = num(df["mono_sec"]) + num(df["mono_nsec"]) / 1e9
    df = df.dropna(subset=["t_raw", "pdu_bytes"]).copy()
    df["t_s"] = df["t_raw"] - df["t_raw"].min()
    df["bin"] = (df["t_s"] / bin_s).astype(int)
    out = df.groupby("bin", as_index=False)["pdu_bytes"].sum()
    out["t_s"] = out["bin"].astype(float) * bin_s
    out["drain_mbps"] = out["pdu_bytes"] * 8.0 / bin_s / 1e6
    return out[["t_s", "drain_mbps"]]


def elapsed_from_clock_strings(time_col: pd.Series) -> pd.Series:
    t = pd.to_datetime("2026-07-27 " + time_col.astype(str), errors="coerce")
    return (t - t.min()).dt.total_seconds()


def load_rlc_buffer(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["lcid"].astype(str).eq("4")].copy()
    df["t_s"] = elapsed_from_clock_strings(df["time"])
    df["bytes_in_buffer"] = num(df["bytes_in_buffer"])
    df = df.dropna(subset=["t_s", "bytes_in_buffer"]).copy()
    df["sec"] = df["t_s"].astype(int)
    out = df.groupby("sec", as_index=False)["bytes_in_buffer"].quantile(0.95)
    out["t_s"] = out["sec"].astype(float)
    out["buffer_kb"] = out["bytes_in_buffer"] / 1024.0
    return out[["t_s", "buffer_kb"]]


def load_mcs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "direction_label" in df.columns:
        df = df[df["direction_label"].astype(str).str.lower().eq("ul")].copy()
    elif "direction" in df.columns:
        d = df["direction"].astype(str).str.lower()
        df = df[d.eq("1") | d.eq("ul")].copy()
    t_col = "t_norm" if "t_norm" in df.columns else "window_start_s"
    df["t_s"] = num(df[t_col])
    df["scheduled_mbps"] = num(df["scheduled_mbps"])
    df["avg_mcs"] = num(df["avg_mcs"])
    df = df.dropna(subset=["t_s", "avg_mcs"]).copy()
    return df[["t_s", "scheduled_mbps", "avg_mcs"]]


def style(ax: plt.Axes, ylabel: str, ylim: tuple[float, float] | None = None) -> None:
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.26, linewidth=0.8)
    ax.grid(True, axis="x", alpha=0.10, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlim(0, 70)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "savefig.dpi": 300,
        }
    )

    loaded: dict[str, dict[str, pd.DataFrame]] = {}
    rows: list[dict[str, object]] = []
    for spec in RUNS:
        # Ignore tiny keepalive/control trickles when aligning presentation time.
        # We want t=0 to mean the first meaningful data-bearer drain/backlog.
        drain = align_active(load_dequeue_rate(spec.dequeue_path), "drain_mbps", 0.1)
        buf = align_active(load_rlc_buffer(spec.rlc_buffer_path), "buffer_kb", 1.0)
        mcs = align_active(load_mcs(spec.grant_path), "scheduled_mbps", 0.1)
        drain70 = complete_rate_bins(drain[drain["t_s"].between(0, 70)].copy(), duration_s=70.0, bin_s=1.0)
        buf70 = buf[buf["t_s"].between(0, 70)].copy()
        mcs70 = mcs[mcs["t_s"].between(0, 70)].copy()
        active_mcs = mcs70[mcs70["scheduled_mbps"].fillna(0) > 0.1]
        loaded[spec.key] = {"drain": drain70, "buffer": buf70, "mcs_all": mcs70, "mcs_active": active_mcs}
        active_drain = drain70[drain70["drain_mbps"] > 0]["drain_mbps"]
        all_drain = drain70["drain_mbps"]
        rows.append(
            {
                "key": spec.key,
                "run_group": spec.run_group,
                "rlc_drain_mbps_p50": float(all_drain.quantile(0.50)) if len(all_drain) else float("nan"),
                "rlc_drain_mbps_p95": float(all_drain.quantile(0.95)) if len(all_drain) else float("nan"),
                "rlc_drain_mbps_max": float(active_drain.max()) if len(active_drain) else float("nan"),
                "rlc_drain_active_mbps_p50": float(active_drain.quantile(0.50)) if len(active_drain) else float("nan"),
                "rlc_drain_active_mbps_p95": float(active_drain.quantile(0.95)) if len(active_drain) else float("nan"),
                "rlc_buffer_kb_p95": float(buf70["buffer_kb"].quantile(0.95)) if len(buf70) else float("nan"),
                "rlc_buffer_kb_max": float(buf70["buffer_kb"].max()) if len(buf70) else float("nan"),
                "mcs_p50": float(active_mcs["avg_mcs"].quantile(0.50)) if len(active_mcs) else float("nan"),
                "mcs_p95": float(active_mcs["avg_mcs"].quantile(0.95)) if len(active_mcs) else float("nan"),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(ANALYSIS / "tractor_vs_carla_rlc_drain_summary.csv", index=False)

    fig, axs = plt.subplots(3, len(RUNS), figsize=(15.3, 7.5), sharex=True, constrained_layout=True)
    for col, spec in enumerate(RUNS):
        row = summary[summary["key"].eq(spec.key)].iloc[0]
        axs[0, col].set_title(
            f"{spec.title}\nfirst 70 s drain p50/p95={row['rlc_drain_mbps_p50']:.1f}/{row['rlc_drain_mbps_p95']:.1f} Mbps",
            fontsize=10.8,
            pad=8,
        )
        drain = loaded[spec.key]["drain"]
        buf = loaded[spec.key]["buffer"]
        mcs_all = loaded[spec.key]["mcs_all"]
        mcs_active = loaded[spec.key]["mcs_active"]

        axs[0, col].fill_between(drain["t_s"], drain["drain_mbps"], color="#F59E0B", alpha=0.25, linewidth=0)
        axs[0, col].plot(drain["t_s"], drain["drain_mbps"], color="#C56A00", linewidth=2.0)
        style(axs[0, col], "RLC dequeue\nMbps", (0, None))

        axs[1, col].fill_between(buf["t_s"], buf["buffer_kb"], color="#D62728", alpha=0.16, linewidth=0)
        axs[1, col].plot(buf["t_s"], buf["buffer_kb"], color="#D62728", linewidth=2.0)
        style(axs[1, col], "RLC LCID4\np95 KB", (0, None))

        idle = mcs_all[mcs_all["scheduled_mbps"].fillna(0) <= 0.1].copy()
        if len(idle):
            axs[2, col].scatter(idle["t_s"], idle["avg_mcs"], color="#B9B9B9", s=10, alpha=0.55, label="idle/low data")
        axs[2, col].plot(mcs_active["t_s"], smooth(mcs_active["avg_mcs"]), color="#218C3A", linewidth=2.8, label="active data")
        style(axs[2, col], "UL MCS index\n(all windows)", (0, 30))
        axs[2, col].set_xlabel("time since first active sample (s)", fontweight="bold")
        axs[2, col].text(
            0.98,
            0.08,
            f"MCS p50={row['mcs_p50']:.1f}",
            transform=axs[2, col].transAxes,
            ha="right",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#218C3A",
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#218C3A", "alpha": 0.88},
        )

    fig.suptitle(
        "UE RLC drain and queue formation: TRACTOR drains cleanly, CARLA queues behind low MCS",
        fontsize=14.2,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.01,
        "RLC dequeue is bytes leaving UE RLC toward MAC, binned at 1 s with zeros included; gray/green: idle/active MCS windows. Each series starts at first active sample.",
        ha="center",
        fontsize=9.3,
        color="#444444",
    )
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"tractor_vs_carla_rlc_drain_queue_mcs_70s.{ext}", bbox_inches="tight")


if __name__ == "__main__":
    main()
