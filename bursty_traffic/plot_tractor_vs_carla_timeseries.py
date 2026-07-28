#!/usr/bin/env python3
"""Time-series comparison: TRACTOR replay traffic vs CARLA split inference.

The earlier summary bars were useful for totals, but they hid the shape of the
traffic.  This plot keeps the time axis visible: application traffic rate,
scheduler MCS, UE BSR backlog, and UE RLC queue occupancy.
"""

from __future__ import annotations

import re
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
    kind: str
    run_group: str
    traffic_path: Path
    grant_path: Path
    bsr_path: Path
    rlc_path: Path
    summary_path: Path | None = None
    layer_latency_path: Path | None = None


TRACTOR_EMBB = "tractor_replay_bw273_vanilla_embb0303a_off100_60s_tcpdump_20260727_210512"
TRACTOR_URLLC = "tractor_replay_bw273_vanilla_urllc0303_off240_60s_tcpdump_20260727_210842"
CARLA_273 = "downlink_oai_bw273_mu1_ttracer_fps10_layerbaseline_20260722_183259"

RUNS = [
    RunSpec(
        key="tractor_onedrive",
        title="TRACTOR OneDrive eMBB\nreal trace replay, 273PRB",
        kind="tractor",
        run_group=TRACTOR_EMBB,
        traffic_path=AB / "metrics_logs" / "tractor_replay" / TRACTOR_EMBB / "udp_sink_packets.txt",
        grant_path=AB / "metrics_logs" / "scenesense_ttracer" / TRACTOR_EMBB / "ue/analysis/nrue_grant_windows.csv",
        bsr_path=AB / "metrics_logs" / "scenesense_ttracer" / TRACTOR_EMBB / "ue/csv/NRUE_MAC_BSR_STATUS.csv",
        rlc_path=AB / "metrics_logs" / "scenesense_ttracer" / TRACTOR_EMBB / "ue/csv/NRUE_MAC_RLC_BUFFER_STATUS.csv",
        summary_path=AB / "metrics_logs" / "tractor_replay" / TRACTOR_EMBB / "tractor_oai_summary.csv",
        layer_latency_path=AB / "metrics_logs" / "scenesense_ttracer" / TRACTOR_EMBB / "layer_latency/uplink_layer_latency.md",
    ),
    RunSpec(
        key="tractor_meet",
        title="TRACTOR Google Meet URLLC\nreal trace replay, 273PRB",
        kind="tractor",
        run_group=TRACTOR_URLLC,
        traffic_path=AB / "metrics_logs" / "tractor_replay" / TRACTOR_URLLC / "udp_sink_packets.txt",
        grant_path=AB / "metrics_logs" / "scenesense_ttracer" / TRACTOR_URLLC / "ue/analysis/nrue_grant_windows.csv",
        bsr_path=AB / "metrics_logs" / "scenesense_ttracer" / TRACTOR_URLLC / "ue/csv/NRUE_MAC_BSR_STATUS.csv",
        rlc_path=AB / "metrics_logs" / "scenesense_ttracer" / TRACTOR_URLLC / "ue/csv/NRUE_MAC_RLC_BUFFER_STATUS.csv",
        summary_path=AB / "metrics_logs" / "tractor_replay" / TRACTOR_URLLC / "tractor_oai_summary.csv",
        layer_latency_path=AB / "metrics_logs" / "scenesense_ttracer" / TRACTOR_URLLC / "layer_latency/uplink_layer_latency.md",
    ),
    RunSpec(
        key="carla_split",
        title="CARLA split inference\nno-AE ≈1 MB/frame, 273PRB",
        kind="carla",
        run_group=CARLA_273,
        traffic_path=AB
        / "downlink_latency_fps/runs/oai_bw273_mu1_ttracer/fps_10_layerbaseline_20260722_183259/streams"
        / f"{CARLA_273}_metrics.csv",
        grant_path=AB / "metrics_logs" / "carla_oai_ttracer" / CARLA_273 / "nrue_ul_grant_windows_compact.csv",
        bsr_path=AB / "metrics_logs" / "scenesense_ttracer" / CARLA_273 / "ue/csv/NRUE_MAC_BSR_STATUS.csv",
        rlc_path=AB / "metrics_logs" / "scenesense_ttracer" / CARLA_273 / "ue/csv/NRUE_MAC_RLC_BUFFER_STATUS.csv",
        summary_path=AB / "metrics_logs" / "carla_oai_ttracer" / CARLA_273 / "CARLA10_OAI_TTRACER_SUMMARY.csv",
        layer_latency_path=AB / "metrics_logs" / "scenesense_ttracer" / CARLA_273 / "layer_latency/uplink_layer_latency.md",
    ),
]

XLIMS = {
    "tractor_onedrive": 70.0,
    "tractor_meet": 70.0,
    "carla_split": 250.0,
}


def _coerce_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _smooth(y: pd.Series, window: int = 3) -> pd.Series:
    if len(y) < window:
        return y
    return y.rolling(window=window, min_periods=1, center=True).median()


def load_tractor_rate(path: Path, bin_s: float = 1.0) -> pd.DataFrame:
    """Parse tcpdump UDP packet lengths into delivered Mbps bins."""
    pat = re.compile(r"^([0-9.]+).*UDP, length ([0-9]+)")
    rows: list[tuple[float, int]] = []
    with path.open(errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if m:
                rows.append((float(m.group(1)), int(m.group(2))))
    if not rows:
        return pd.DataFrame({"t_s": [], "mbps": []})
    df = pd.DataFrame(rows, columns=["ts", "bytes"])
    df["t_s"] = (df["ts"] - df["ts"].min()).astype(float)
    df["bin"] = (df["t_s"] / bin_s).astype(int)
    out = df.groupby("bin", as_index=False)["bytes"].sum()
    out["t_s"] = out["bin"].astype(float) * bin_s
    out["mbps"] = out["bytes"] * 8.0 / bin_s / 1e6
    return out[["t_s", "mbps"]]


def load_carla_rate(path: Path, bin_s: float = 1.0) -> pd.DataFrame:
    """Use front-side feature payload bytes as offered uplink application rate."""
    df = pd.read_csv(path)
    df["elapsed_s"] = _coerce_num(df["elapsed_s"])
    df["feature_payload_bytes"] = _coerce_num(df["feature_payload_bytes"])
    df = df.dropna(subset=["elapsed_s", "feature_payload_bytes"]).copy()
    if df.empty:
        return pd.DataFrame({"t_s": [], "mbps": []})
    df["t_s"] = df["elapsed_s"] - df["elapsed_s"].min()
    df["bin"] = (df["t_s"] / bin_s).astype(int)
    out = df.groupby("bin", as_index=False)["feature_payload_bytes"].sum()
    out["t_s"] = out["bin"].astype(float) * bin_s
    out["mbps"] = out["feature_payload_bytes"] * 8.0 / bin_s / 1e6
    return out[["t_s", "mbps"]]


def load_grants(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "direction_label" in df.columns:
        mask = df["direction_label"].astype(str).str.lower().eq("ul")
        df = df[mask].copy()
    elif "direction" in df.columns:
        d = df["direction"].astype(str).str.lower()
        df = df[d.eq("ul") | d.eq("1")].copy()
    t_col = "t_norm" if "t_norm" in df.columns else "window_start_s"
    df["t_s"] = _coerce_num(df[t_col])
    for col in ["avg_mcs", "scheduled_mbps"]:
        if col in df.columns:
            df[col] = _coerce_num(df[col])
    df = df.dropna(subset=["t_s"]).sort_values("t_s")
    return df[["t_s", "avg_mcs", "scheduled_mbps"]].copy()


def _elapsed_from_clock_strings(time_col: pd.Series) -> pd.Series:
    # The T-tracer CSVs store HH:MM:SS.ffffff. Use an arbitrary date and
    # normalize to the first sample within each file.
    t = pd.to_datetime("2026-07-27 " + time_col.astype(str), errors="coerce")
    return (t - t.min()).dt.total_seconds()


def load_bsr(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "time" not in df.columns or "lcg1_bytes" not in df.columns:
        return pd.DataFrame({"t_s": [], "kb": []})
    df["t_s"] = _elapsed_from_clock_strings(df["time"])
    df["lcg1_bytes"] = _coerce_num(df["lcg1_bytes"])
    df = df.dropna(subset=["t_s", "lcg1_bytes"]).copy()
    df["sec"] = df["t_s"].astype(int)
    out = df.groupby("sec", as_index=False)["lcg1_bytes"].quantile(0.95)
    out["t_s"] = out["sec"].astype(float)
    out["kb"] = out["lcg1_bytes"] / 1024.0
    return out[["t_s", "kb"]]


def load_rlc(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"time", "lcid", "bytes_in_buffer"}
    if not needed.issubset(set(df.columns)):
        return pd.DataFrame({"t_s": [], "kb": []})
    df = df[df["lcid"].astype(str).eq("4")].copy()
    df["t_s"] = _elapsed_from_clock_strings(df["time"])
    df["bytes_in_buffer"] = _coerce_num(df["bytes_in_buffer"])
    df = df.dropna(subset=["t_s", "bytes_in_buffer"]).copy()
    df["sec"] = df["t_s"].astype(int)
    out = df.groupby("sec", as_index=False)["bytes_in_buffer"].quantile(0.95)
    out["t_s"] = out["sec"].astype(float)
    out["kb"] = out["bytes_in_buffer"] / 1024.0
    return out[["t_s", "kb"]]


def summarize_run(spec: RunSpec, rate: pd.DataFrame, grants: pd.DataFrame, bsr: pd.DataFrame, rlc: pd.DataFrame) -> dict[str, object]:
    active_grants = grants[grants["scheduled_mbps"].fillna(0) > 0.1].copy()
    row: dict[str, object] = {
        "key": spec.key,
        "run_group": spec.run_group,
        "traffic_mbps_p50": rate["mbps"].quantile(0.50) if len(rate) else float("nan"),
        "traffic_mbps_p95": rate["mbps"].quantile(0.95) if len(rate) else float("nan"),
        "mcs_p50": active_grants["avg_mcs"].quantile(0.50) if len(active_grants) else float("nan"),
        "mcs_p95": active_grants["avg_mcs"].quantile(0.95) if len(active_grants) else float("nan"),
        "scheduled_mbps_p50": active_grants["scheduled_mbps"].quantile(0.50) if len(active_grants) else float("nan"),
        "scheduled_mbps_p95": active_grants["scheduled_mbps"].quantile(0.95) if len(active_grants) else float("nan"),
        "bsr_lcg1_kb_p95": bsr["kb"].quantile(0.95) if len(bsr) else float("nan"),
        "bsr_lcg1_kb_max": bsr["kb"].max() if len(bsr) else float("nan"),
        "rlc_lcid4_kb_p95": rlc["kb"].quantile(0.95) if len(rlc) else float("nan"),
        "rlc_lcid4_kb_max": rlc["kb"].max() if len(rlc) else float("nan"),
    }
    if spec.summary_path and spec.summary_path.exists():
        summary = pd.read_csv(spec.summary_path).iloc[0].to_dict()
        for src, dst in [
            ("rlc_queue_wait_mean_ms", "rlc_queue_wait_mean_ms"),
            ("ran_ul_p50_ms", "ran_ul_p50_ms"),
            ("ran_ul_p95_ms", "ran_ul_p95_ms"),
            ("delivery_pct", "delivery_pct"),
            ("delivery_rate", "delivery_rate"),
            ("packet_delivery", "packet_delivery"),
        ]:
            if src in summary:
                row[dst] = summary[src]
    if spec.layer_latency_path and spec.layer_latency_path.exists():
        text = spec.layer_latency_path.read_text(errors="replace")
        m = re.search(r"RLC mean queueing delay .*:\*\* ([0-9.]+) ms", text)
        if m and ("rlc_queue_wait_mean_ms" not in row or pd.isna(row.get("rlc_queue_wait_mean_ms"))):
            row["rlc_queue_wait_mean_ms"] = float(m.group(1))
    return row


def style_ax(ax: plt.Axes, ylabel: str, ylim: tuple[float, float] | None = None) -> None:
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.28, linewidth=0.8)
    ax.grid(True, axis="x", alpha=0.12, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9, width=1)
    if ylim:
        ax.set_ylim(*ylim)


def window(df: pd.DataFrame, key: str) -> pd.DataFrame:
    xmax = XLIMS.get(key)
    if xmax is None or "t_s" not in df.columns:
        return df
    return df[df["t_s"].between(0, xmax)].copy()


def align_to_first_active(df: pd.DataFrame, value_col: str, threshold: float) -> pd.DataFrame:
    """Return a copy with t_s zeroed at the first active sample.

    This is for presentation plots: it removes startup/idling trace time so the
    visual comparison starts where the actual traffic/scheduler response begins.
    """
    if df.empty or "t_s" not in df.columns or value_col not in df.columns:
        return df.copy()
    active = df[_coerce_num(df[value_col]).fillna(0) > threshold]
    out = df.copy()
    if len(active):
        out["t_s"] = out["t_s"] - float(active["t_s"].min())
    return out


def complete_rate_bins(df: pd.DataFrame, duration_s: float, bin_s: float) -> pd.DataFrame:
    """Fill missing traffic bins with explicit zeros for full-timeline plots."""
    n = int(duration_s / bin_s)
    if df.empty:
        return pd.DataFrame({"t_s": [i * bin_s for i in range(n)], "mbps": [0.0] * n})
    idx = (df["t_s"] / bin_s).round().astype(int)
    vals = df.set_index(idx)["mbps"].groupby(level=0).sum().reindex(range(n), fill_value=0.0)
    return pd.DataFrame({"t_s": vals.index.astype(float) * bin_s, "mbps": vals.to_numpy()})


def plot_rate_and_mcs(ax_rate: plt.Axes, ax_mcs: plt.Axes, spec: RunSpec, data: dict[str, pd.DataFrame]) -> None:
    rate = window(data["rate"], spec.key)
    grants = window(data["grants"], spec.key)
    colors = {
        "rate": "#1f77b4",
        "scheduled": "#ff7f0e",
        "mcs": "#2ca02c",
    }
    if len(rate):
        ax_rate.fill_between(rate["t_s"], rate["mbps"], color=colors["rate"], alpha=0.22, linewidth=0)
        ax_rate.plot(rate["t_s"], rate["mbps"], color=colors["rate"], linewidth=2.1, label="app/data rate")
    if len(grants):
        ax_rate.plot(
            grants["t_s"],
            _smooth(grants["scheduled_mbps"]),
            color=colors["scheduled"],
            linewidth=1.8,
            linestyle="--",
            label="scheduled UL rate",
        )
        active = grants[grants["scheduled_mbps"].fillna(0) > 0.1].copy()
        ax_mcs.scatter(grants["t_s"], grants["avg_mcs"], color="#b8b8b8", s=8, alpha=0.45, label="idle/low data")
        ax_mcs.plot(active["t_s"], _smooth(active["avg_mcs"]), color=colors["mcs"], linewidth=2.4, label="active UL MCS")
    style_ax(ax_rate, "Mbps", (0, None))
    style_ax(ax_mcs, "MCS index", (-1, 30))
    ax_rate.set_xlim(0, XLIMS.get(spec.key, ax_rate.get_xlim()[1]))
    ax_mcs.set_xlim(0, XLIMS.get(spec.key, ax_mcs.get_xlim()[1]))
    ax_rate.legend(loc="upper right", frameon=False, fontsize=8)
    ax_mcs.legend(loc="lower right", frameon=False, fontsize=8)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ANALYSIS.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )

    loaded: dict[str, dict[str, pd.DataFrame]] = {}
    summary_rows: list[dict[str, object]] = []
    for spec in RUNS:
        rate = load_tractor_rate(spec.traffic_path) if spec.kind == "tractor" else load_carla_rate(spec.traffic_path)
        grants = load_grants(spec.grant_path)
        bsr = load_bsr(spec.bsr_path)
        rlc = load_rlc(spec.rlc_path)
        loaded[spec.key] = {"rate": rate, "grants": grants, "bsr": bsr, "rlc": rlc}
        summary_rows.append(summarize_run(spec, rate, grants, bsr, rlc))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(ANALYSIS / "tractor_vs_carla_timeseries_summary.csv", index=False)

    colors = {"bsr": "#9467bd", "rlc": "#d62728"}

    fig, axs = plt.subplots(
        4,
        len(RUNS),
        figsize=(16.0, 10.4),
        sharex=False,
        constrained_layout=True,
    )

    for col, spec in enumerate(RUNS):
        data = loaded[spec.key]
        srow = summary_df[summary_df["key"].eq(spec.key)].iloc[0]
        subtitle = (
            f"{spec.title}\n"
            f"traffic p50/p95={srow['traffic_mbps_p50']:.1f}/{srow['traffic_mbps_p95']:.1f} Mbps, "
            f"MCS p50={srow['mcs_p50']:.1f}"
        )
        axs[0, col].set_title(subtitle, fontsize=10.5, pad=10)

        plot_rate_and_mcs(axs[0, col], axs[1, col], spec, data)

        bsr = window(data["bsr"], spec.key)
        rlc = window(data["rlc"], spec.key)
        if len(bsr):
            axs[2, col].fill_between(bsr["t_s"], bsr["kb"], color=colors["bsr"], alpha=0.18, linewidth=0)
            axs[2, col].plot(bsr["t_s"], bsr["kb"], color=colors["bsr"], linewidth=2.0)
        style_ax(axs[2, col], "BSR LCG1\np95 KB", (0, None))
        axs[2, col].set_xlim(0, XLIMS.get(spec.key, axs[2, col].get_xlim()[1]))

        if len(rlc):
            axs[3, col].fill_between(rlc["t_s"], rlc["kb"], color=colors["rlc"], alpha=0.16, linewidth=0)
            axs[3, col].plot(rlc["t_s"], rlc["kb"], color=colors["rlc"], linewidth=2.0)
        style_ax(axs[3, col], "RLC LCID4\np95 KB", (0, None))
        axs[3, col].set_xlim(0, XLIMS.get(spec.key, axs[3, col].get_xlim()[1]))
        axs[3, col].set_xlabel("elapsed time in run (s)", fontweight="bold")

    fig.suptitle(
        "Real bursty replay vs CARLA split inference over OAI 273PRB: time-series evidence",
        fontsize=15,
        fontweight="bold",
    )

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"tractor_vs_carla_timeseries_rate_mcs_queue.{ext}", bbox_inches="tight")

    focused, faxs = plt.subplots(
        2,
        len(RUNS),
        figsize=(16.0, 5.8),
        sharex=False,
        constrained_layout=True,
    )
    for col, spec in enumerate(RUNS):
        data = loaded[spec.key]
        srow = summary_df[summary_df["key"].eq(spec.key)].iloc[0]
        faxs[0, col].set_title(
            f"{spec.title}\ntraffic p50/p95={srow['traffic_mbps_p50']:.1f}/{srow['traffic_mbps_p95']:.1f} Mbps, "
            f"MCS p50={srow['mcs_p50']:.1f}",
            fontsize=10.5,
            pad=10,
        )
        plot_rate_and_mcs(faxs[0, col], faxs[1, col], spec, data)
        faxs[1, col].set_xlabel("elapsed time in run (s)", fontweight="bold")
    focused.suptitle(
        "Traffic burst shape and MCS response: TRACTOR real traces vs CARLA split inference",
        fontsize=15,
        fontweight="bold",
    )
    for ext in ("png", "pdf"):
        focused.savefig(OUT / f"tractor_vs_carla_rate_mcs_timeseries.{ext}", bbox_inches="tight")

    # Matched-window view: keep all three traffic families on the same 70 s
    # horizon. This avoids visually over-weighting the longer CARLA run.
    matched, maxs = plt.subplots(
        2,
        len(RUNS),
        figsize=(16.0, 5.8),
        sharex=False,
        constrained_layout=True,
    )
    old_xlims = dict(XLIMS)
    try:
        for spec in RUNS:
            XLIMS[spec.key] = 70.0
        for col, spec in enumerate(RUNS):
            data = loaded[spec.key]
            srow = summary_df[summary_df["key"].eq(spec.key)].iloc[0]
            maxs[0, col].set_title(
                f"{spec.title}\nfirst 70 s, 1 s bins; MCS p50={srow['mcs_p50']:.1f}",
                fontsize=10.5,
                pad=10,
            )
            plot_rate_and_mcs(maxs[0, col], maxs[1, col], spec, data)
            maxs[1, col].set_xlabel("elapsed time in run (s)", fontweight="bold")
        matched.suptitle(
            "Matched 70-second view: traffic rate and MCS response",
            fontsize=15,
            fontweight="bold",
        )
        for ext in ("png", "pdf"):
            matched.savefig(OUT / f"tractor_vs_carla_rate_mcs_timeseries_70s.{ext}", bbox_inches="tight")
    finally:
        XLIMS.clear()
        XLIMS.update(old_xlims)

    fine_bin_s = 0.1
    fine_loaded: dict[str, dict[str, pd.DataFrame]] = {}
    for spec in RUNS:
        fine_rate = (
            load_tractor_rate(spec.traffic_path, bin_s=fine_bin_s)
            if spec.kind == "tractor"
            else load_carla_rate(spec.traffic_path, bin_s=fine_bin_s)
        )
        fine_loaded[spec.key] = {"rate": fine_rate, "grants": loaded[spec.key]["grants"]}

    # Slide-clean version: only traffic shape and active MCS, with each series
    # zeroed at the first active sample. This avoids CARLA appearing to "start"
    # late because the scheduler trace includes pre-traffic idle windows.
    clean, caxs = plt.subplots(
        2,
        len(RUNS),
        figsize=(15.2, 5.25),
        sharex=True,
        constrained_layout=True,
    )
    clean_activity_rows: list[dict[str, object]] = []
    for col, spec in enumerate(RUNS):
        fine_rate = align_to_first_active(fine_loaded[spec.key]["rate"], "mbps", 0.0)
        fine_rate = fine_rate[fine_rate["t_s"].between(0, 70.0)].copy()
        rate_full = complete_rate_bins(fine_rate, duration_s=70.0, bin_s=fine_bin_s)
        rate_1s = rate_full["mbps"].rolling(window=int(1.0 / fine_bin_s), min_periods=1).mean()
        grants = align_to_first_active(loaded[spec.key]["grants"], "scheduled_mbps", 0.1)
        grants = grants[grants["t_s"].between(0, 70.0)].copy()
        active_grants = grants[grants["scheduled_mbps"].fillna(0) > 0.1].copy()

        vals = rate_full["mbps"]
        active_vals = vals[vals > 0]
        active_frac = float((vals > 0).mean()) if len(vals) else float("nan")
        sustained_mbps = float(vals.mean()) if len(vals) else float("nan")
        mcs_p50 = float(active_grants["avg_mcs"].quantile(0.50)) if len(active_grants) else float("nan")
        clean_activity_rows.append(
            {
                "key": spec.key,
                "run_group": spec.run_group,
                "active_bin_fraction": active_frac,
                "sustained_mbps_70s": sustained_mbps,
                "active_mbps_p50": float(active_vals.quantile(0.50)) if len(active_vals) else float("nan"),
                "active_mcs_p50": mcs_p50,
            }
        )

        title = {
            "tractor_onedrive": "TRACTOR OneDrive\nreal bursty replay",
            "tractor_meet": "TRACTOR Google Meet\nreal bursty replay",
            "carla_split": "CARLA split inference\n≈1 MB feature bursts",
        }[spec.key]
        caxs[0, col].set_title(
            f"{title}\nsustained={sustained_mbps:.1f} Mbps, active bins={100*active_frac:.0f}%",
            fontsize=11.0,
            fontweight="bold",
            pad=8,
        )
        caxs[0, col].bar(
            rate_full["t_s"],
            rate_full["mbps"],
            width=fine_bin_s * 0.92,
            color="#3B82C4",
            alpha=0.80,
            align="edge",
            linewidth=0,
        )
        caxs[0, col].plot(
            rate_full["t_s"],
            rate_1s,
            color="#111111",
            linewidth=1.9,
            alpha=0.88,
            label="1 s moving average",
        )
        caxs[0, col].set_xlim(0, 70)
        caxs[0, col].grid(True, axis="y", alpha=0.25, linewidth=0.8)
        caxs[0, col].grid(True, axis="x", alpha=0.10, linewidth=0.6)
        caxs[0, col].spines[["top", "right"]].set_visible(False)
        caxs[0, col].yaxis.set_major_locator(MaxNLocator(5))
        caxs[0, col].tick_params(labelsize=9)
        if col == 0:
            caxs[0, col].set_ylabel("Traffic rate\n(Mbps, 100 ms bins)", fontweight="bold")

        idle = grants[grants["scheduled_mbps"].fillna(0) <= 0.1].copy()
        if len(idle):
            caxs[1, col].scatter(
                idle["t_s"],
                idle["avg_mcs"],
                color="#B9B9B9",
                s=10,
                alpha=0.55,
                label="idle/low data",
            )
        caxs[1, col].plot(
            active_grants["t_s"],
            _smooth(active_grants["avg_mcs"]),
            color="#218C3A",
            linewidth=2.8,
            label="active data",
        )
        caxs[1, col].set_ylim(0, 30)
        caxs[1, col].set_xlim(0, 70)
        caxs[1, col].grid(True, axis="y", alpha=0.25, linewidth=0.8)
        caxs[1, col].grid(True, axis="x", alpha=0.10, linewidth=0.6)
        caxs[1, col].spines[["top", "right"]].set_visible(False)
        caxs[1, col].yaxis.set_major_locator(MaxNLocator(6))
        caxs[1, col].tick_params(labelsize=9)
        caxs[1, col].set_xlabel("time since first active sample (s)", fontweight="bold")
        if col == 0:
            caxs[1, col].set_ylabel("UL MCS index\n(all scheduler windows)", fontweight="bold")
        caxs[1, col].text(
            0.98,
            0.08,
            f"MCS p50={mcs_p50:.1f}",
            transform=caxs[1, col].transAxes,
            ha="right",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#218C3A",
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#218C3A", "alpha": 0.88},
        )
    clean_activity_df = pd.DataFrame(clean_activity_rows)
    clean_activity_df.to_csv(ANALYSIS / "tractor_vs_carla_clean_aligned_70s_summary.csv", index=False)
    clean.suptitle(
        "Traffic shape vs OAI MCS response: CARLA is sparse burst/wait, TRACTOR is mostly continuous",
        fontsize=14.0,
        fontweight="bold",
    )
    clean.text(
        0.5,
        -0.01,
        "Blue bars: 100 ms traffic bins with zeros included; black line: 1 s moving average; gray/green: idle/active MCS windows. Each series starts at first active sample.",
        ha="center",
        fontsize=9.2,
        color="#444444",
    )
    for ext in ("png", "pdf"):
        clean.savefig(OUT / f"tractor_vs_carla_clean_aligned_traffic_mcs_70s.{ext}", bbox_inches="tight")

    # Fine-bin traffic shape view: the CARLA idle/wait periods are sub-second,
    # so 1 s aggregation hides them. This plot uses 100 ms bins for traffic
    # injection/delivery rate while keeping MCS as the 1 s scheduler window.
    activity_rows: list[dict[str, object]] = []
    for spec in RUNS:
        fine_rate = fine_loaded[spec.key]["rate"]
        vals = (
            fine_rate[fine_rate["t_s"].between(0, 70.0)]
            .set_index((fine_rate[fine_rate["t_s"].between(0, 70.0)]["t_s"] / fine_bin_s).round().astype(int))["mbps"]
            .reindex(range(int(70.0 / fine_bin_s)), fill_value=0.0)
        )
        active = vals[vals > 0]
        activity_rows.append(
            {
                "key": spec.key,
                "run_group": spec.run_group,
                "bin_s": fine_bin_s,
                "window_s": 70.0,
                "active_bin_fraction": float((vals > 0).mean()),
                "active_bin_count": int((vals > 0).sum()),
                "total_bins": int(len(vals)),
                "active_mbps_p50": float(active.quantile(0.50)) if len(active) else float("nan"),
                "all_bins_mbps_p95": float(vals.quantile(0.95)) if len(vals) else float("nan"),
            }
        )
    activity_df = pd.DataFrame(activity_rows)
    activity_df.to_csv(ANALYSIS / "tractor_vs_carla_100ms_activity_summary.csv", index=False)

    fine, faxs = plt.subplots(
        2,
        len(RUNS),
        figsize=(16.0, 5.9),
        sharex=False,
        constrained_layout=True,
    )
    old_xlims = dict(XLIMS)
    try:
        for spec in RUNS:
            XLIMS[spec.key] = 70.0
        for col, spec in enumerate(RUNS):
            rate = window(fine_loaded[spec.key]["rate"], spec.key)
            grants = window(fine_loaded[spec.key]["grants"], spec.key)
            arow = activity_df[activity_df["key"].eq(spec.key)].iloc[0]
            faxs[0, col].set_title(
                f"{spec.title}\n100 ms bins active={100*arow['active_bin_fraction']:.0f}%, "
                f"active-bin p50={arow['active_mbps_p50']:.1f} Mbps",
                fontsize=10.2,
                pad=10,
            )
            if len(rate):
                faxs[0, col].bar(
                    rate["t_s"],
                    rate["mbps"],
                    width=fine_bin_s * 0.92,
                    color="#1f77b4",
                    alpha=0.72,
                    align="edge",
                )
            style_ax(faxs[0, col], "Mbps per\n100 ms bin", (0, None))
            faxs[0, col].set_xlim(0, 70)

            if len(grants):
                active = grants[grants["scheduled_mbps"].fillna(0) > 0.1].copy()
                faxs[1, col].scatter(grants["t_s"], grants["avg_mcs"], color="#b8b8b8", s=8, alpha=0.45)
                faxs[1, col].plot(active["t_s"], _smooth(active["avg_mcs"]), color="#2ca02c", linewidth=2.4)
            style_ax(faxs[1, col], "MCS index", (-1, 30))
            faxs[1, col].set_xlim(0, 70)
            faxs[1, col].set_xlabel("elapsed time in run (s)", fontweight="bold")
        fine.suptitle(
            "Fine-bin traffic shape: CARLA has burst/wait gaps that 1-second bins hide",
            fontsize=15,
            fontweight="bold",
        )
        for ext in ("png", "pdf"):
            fine.savefig(OUT / f"tractor_vs_carla_traffic_100ms_mcs_70s.{ext}", bbox_inches="tight")
    finally:
        XLIMS.clear()
        XLIMS.update(old_xlims)


if __name__ == "__main__":
    main()
