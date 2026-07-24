#!/usr/bin/env python3
"""Generate slide-ready scheduler/throughput time-series panels.

Each output figure has four vertical panels:
  1. default OAI iperf3
  2. CARLA default 106PRB
  3. CARLA 106PRB UL-heavy
  4. CARLA 273PRB

The intent is to make the scheduler story visually obvious before the
per-layer latency breakdown slide: iperf receives high MCS under the same
RFsim channel, while CARLA split-tensor traffic receives low MCS even when
the scheduler allocates near-max PRBs.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.labelsize": 13,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.titlesize": 15,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.titlesize": 22,
        "figure.dpi": 220,
        "savefig.dpi": 420,
        "axes.linewidth": 1.4,
        "lines.linewidth": 2.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


ROOT = Path(__file__).resolve().parents[1]
TRACE_ROOT = ROOT / "metrics_logs/scenesense_ttracer"
NETWORK_ROOT = ROOT / "metrics_logs/scenesense_network"
COMPACT_ROOT = ROOT / "metrics_logs/carla_oai_ttracer"
OUT_DIR = ROOT / "oai_layer_latency/plots/scheduler_panels"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class RunSpec:
    label: str
    grant_path: Path
    network_path: Path
    max_prb: int
    snr_path: Optional[Path] = None
    snr_note: str = ""
    summary_path: Optional[Path] = None
    min_feature_kb_p50: Optional[float] = None
    grant_has_direction: bool = False
    mcs_col: str = "avg_mcs"
    prb_col: str = "avg_rb_size"
    throughput_col: str = "tx_bitrate_mbps"


RUNS: list[RunSpec] = [
    RunSpec(
        label="iperf3 UDP uplink 9 Mbps\n106PRB, 7DL/2UL",
        grant_path=TRACE_ROOT
        / "oai_iperf_default_udp9m_20260720/ue/analysis/nrue_grant_windows.csv",
        network_path=NETWORK_ROOT / "oai_iperf_default_udp9m_20260720/network_timeseries.csv",
        snr_path=TRACE_ROOT
        / "oai_iperf_default_udp9m_20260720/gnb/csv/GNB_MAC_PUSCH_POWER_CONTROL.csv",
        max_prb=106,
        snr_note="exact",
        grant_has_direction=True,
    ),
    RunSpec(
        label="CARLA default OAI\n106PRB, 7DL/2UL",
        grant_path=COMPACT_ROOT
        / "downlink_oai_default106_ttracer_fps10_drivable_rerun_20260724_default106_noae/nrue_ul_grant_windows_compact.csv",
        network_path=COMPACT_ROOT
        / "downlink_oai_default106_ttracer_fps10_drivable_rerun_20260724_default106_noae/network_timeseries.csv",
        snr_path=COMPACT_ROOT
        / "downlink_oai_default106_ttracer_fps10_drivable_rerun_20260724_default106_noae/gnb_pusch_power_compact.csv",
        max_prb=106,
        snr_note="exact",
        summary_path=COMPACT_ROOT
        / "downlink_oai_default106_ttracer_fps10_drivable_rerun_20260724_default106_noae/CARLA10_OAI_TTRACER_SUMMARY.csv",
        min_feature_kb_p50=900.0,
    ),
    RunSpec(
        label="CARLA UL-heavy OAI\n106PRB, 4DL/5UL",
        grant_path=COMPACT_ROOT
        / "downlink_oai_ulheavy_106_ttracer_fps10_drivable_rerun_20260722_ulheavy106/nrue_ul_grant_windows_compact.csv",
        network_path=COMPACT_ROOT
        / "downlink_oai_ulheavy_106_ttracer_fps10_drivable_rerun_20260722_ulheavy106/network_timeseries.csv",
        snr_path=COMPACT_ROOT
        / "downlink_oai_ulheavy_106_ttracer_forcemcs28_fps10_forcemcs28_ulheavy106_20260723/gnb_pusch_power_compact.csv",
        max_prb=106,
        snr_note="same-config RFsim control",
        summary_path=COMPACT_ROOT
        / "downlink_oai_ulheavy_106_ttracer_fps10_drivable_rerun_20260722_ulheavy106/CARLA10_OAI_TTRACER_SUMMARY.csv",
        min_feature_kb_p50=900.0,
    ),
    RunSpec(
        label="CARLA wider BW OAI\n273PRB, 7DL/2UL",
        grant_path=COMPACT_ROOT
        / "downlink_oai_bw273_mu1_ttracer_fps10_drivable_rerun_20260722_bw273/nrue_ul_grant_windows_compact.csv",
        network_path=COMPACT_ROOT
        / "downlink_oai_bw273_mu1_ttracer_fps10_drivable_rerun_20260722_bw273/network_timeseries.csv",
        snr_path=COMPACT_ROOT
        / "downlink_oai_bw273_mu1_ttracer_fps10_layerinstr_20260723_145921/gnb_pusch_power_compact.csv",
        max_prb=273,
        snr_note="273PRB RFsim control",
        summary_path=COMPACT_ROOT
        / "downlink_oai_bw273_mu1_ttracer_fps10_drivable_rerun_20260722_bw273/CARLA10_OAI_TTRACER_SUMMARY.csv",
        min_feature_kb_p50=900.0,
    ),
]


COLORS = {
    "mcs": "#2563EB",
    "prb": "#059669",
    "throughput": "#D97706",
    "snr": "#7C3AED",
    "grid": "#CBD5E1",
    "note": "#475569",
    "maxline": "#334155",
}

METRIC_NAMES = {
    "mcs": "MCS",
    "prb": "PRB",
    "tx_mbps": "TX",
    "snr_db": "SNR",
}


def _ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def validate_run_specs() -> None:
    """Fail fast if a CARLA comparison row is not a full-payload no-AE trace."""
    errors: list[str] = []
    for spec in RUNS:
        if spec.summary_path is None:
            continue
        if not spec.summary_path.exists():
            errors.append(f"{spec.label}: missing summary {spec.summary_path}")
            continue
        try:
            summary = pd.read_csv(spec.summary_path)
            feature_kb = float(summary["feature_kb_p50"].iloc[0])
        except Exception as exc:
            errors.append(f"{spec.label}: could not read feature_kb_p50 from {spec.summary_path}: {exc}")
            continue
        if spec.min_feature_kb_p50 is not None and feature_kb < spec.min_feature_kb_p50:
            errors.append(
                f"{spec.label}: feature_kb_p50={feature_kb:.1f} KB is below "
                f"required full-payload threshold {spec.min_feature_kb_p50:.1f} KB"
            )
    if errors:
        joined = "\n  - ".join(errors)
        raise RuntimeError(
            "Refusing to generate mixed-payload scheduler plots. Missing or invalid full-payload rows:\n"
            f"  - {joined}"
        )


def load_grants(spec: RunSpec) -> pd.DataFrame:
    _ensure_exists(spec.grant_path)
    df = pd.read_csv(spec.grant_path)
    if spec.grant_has_direction:
        if "direction_label" in df.columns:
            df = df[df["direction_label"].astype(str).str.lower().eq("ul")]
        elif "direction" in df.columns:
            df = df[df["direction"].astype(str).eq("1")]
    # Keep the real time base, but normalize the left edge to zero for plotting.
    if "window_start_s" not in df.columns:
        raise ValueError(f"{spec.grant_path} has no window_start_s column")
    out = pd.DataFrame(
        {
            "t": pd.to_numeric(df["window_start_s"], errors="coerce"),
            "mcs": pd.to_numeric(df[spec.mcs_col], errors="coerce"),
            "prb": pd.to_numeric(df[spec.prb_col], errors="coerce"),
            "scheduled_mbps": pd.to_numeric(df["scheduled_mbps"], errors="coerce"),
        }
    ).dropna(subset=["t"])
    out["t"] = out["t"] - out["t"].min()
    return out.sort_values("t")


def load_network(spec: RunSpec) -> pd.DataFrame:
    _ensure_exists(spec.network_path)
    df = pd.read_csv(spec.network_path)
    if "iface_label" in df.columns:
        df = df[df["iface_label"].astype(str).eq("ue1")]
    if "iface_up" in df.columns:
        # pandas may read booleans as bool or strings depending on mixed rows.
        up = df["iface_up"].astype(str).str.lower().isin(["true", "1", "yes"])
        df = df[up]
    out = pd.DataFrame(
        {
            "t": pd.to_numeric(df["elapsed_s"], errors="coerce"),
            "tx_mbps": pd.to_numeric(df[spec.throughput_col], errors="coerce"),
        }
    ).dropna(subset=["t", "tx_mbps"])
    if out.empty:
        return out
    out["t"] = out["t"] - out["t"].min()
    return out.sort_values("t")


def _seconds_from_clock(value: str) -> float:
    # Handles HH:MM:SS.ffffff from T-tracer CSVs.
    try:
        hh, mm, ss = value.split(":")
        return int(hh) * 3600 + int(mm) * 60 + float(ss)
    except Exception:
        return math.nan


def load_snr(spec: RunSpec) -> pd.DataFrame:
    if spec.snr_path is None or not spec.snr_path.exists():
        return pd.DataFrame(columns=["t", "snr_db"])
    df = pd.read_csv(spec.snr_path)
    if "snr_db" in df.columns:
        t = pd.to_numeric(df.get("t_norm"), errors="coerce")
        snr = pd.to_numeric(df["snr_db"], errors="coerce")
    else:
        # Raw GNB_MAC_PUSCH_POWER_CONTROL.csv.
        t = df["time"].astype(str).map(_seconds_from_clock)
        t = pd.to_numeric(t, errors="coerce")
        snr = pd.to_numeric(df["snrx10"], errors="coerce") * 0.1
    out = pd.DataFrame({"t": t, "snr_db": snr}).dropna()
    if out.empty:
        return out
    out["t"] = out["t"] - out["t"].min()
    # Window to 1-second medians so the plot is readable and consistent with
    # the UE grant windowing.
    out["window"] = np.floor(out["t"]).astype(int)
    win = (
        out.groupby("window", as_index=False)
        .agg(t=("window", "first"), snr_db=("snr_db", "median"))
        .sort_values("t")
    )
    return win[["t", "snr_db"]]


def robust_ylim(values: np.ndarray, pad_frac: float = 0.12) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    lo = float(np.nanpercentile(values, 1))
    hi = float(np.nanpercentile(values, 99))
    if math.isclose(lo, hi):
        lo -= 1.0
        hi += 1.0
    pad = (hi - lo) * pad_frac
    return max(0.0, lo - pad), hi + pad


def summarize_series(values: pd.Series) -> str:
    v = pd.to_numeric(values, errors="coerce").dropna()
    if v.empty:
        return "no data"
    return f"median {np.nanpercentile(v, 50):.1f}   p95 {np.nanpercentile(v, 95):.1f}"


def make_metric_plot(
    metric: str,
    title: str,
    y_label: str,
    file_stem: str,
    datasets: dict[str, pd.DataFrame],
    source_notes: Optional[list[str]] = None,
    carla_window_s: Optional[float] = None,
) -> None:
    fig, axes = plt.subplots(len(RUNS), 1, figsize=(15.2, 10.2), sharex=False)
    fig.subplots_adjust(left=0.095, right=0.985, top=0.895, bottom=0.075, hspace=0.72)
    fig.suptitle(title, fontsize=22, fontweight="bold", y=0.982)

    for idx, (ax, spec) in enumerate(zip(axes, RUNS)):
        df = datasets[spec.label]
        if df.empty or metric not in df.columns:
            ax.text(
                0.5,
                0.5,
                "not captured in this trace",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=13,
                fontweight="bold",
                color=COLORS["note"],
            )
            ax.set_xlim(0, 1)
        else:
            ax.plot(
                df["t"],
                df[metric],
                color=COLORS.get(metric, COLORS["mcs"]),
                linewidth=2.8,
                solid_capstyle="round",
            )
            if carla_window_s is not None and spec.summary_path is not None:
                ax.set_xlim(0, carla_window_s)
            else:
                ax.set_xlim(float(df["t"].min()), float(df["t"].max()))
            lo, hi = robust_ylim(df[metric].to_numpy(dtype=float))
            if metric == "mcs":
                ax.axhspan(0, 9, color="#FEF3C7", alpha=0.45, linewidth=0)
                ax.set_ylim(0, max(30, hi))
                ax.text(
                    0.985,
                    0.82,
                    "QPSK-dominant\nlow-MCS region",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=11,
                    fontweight="bold",
                    color="#92400E",
                )
            elif metric == "prb":
                ax.axhline(
                    spec.max_prb,
                    color=COLORS["maxline"],
                    linestyle="--",
                    linewidth=1.8,
                    alpha=0.65,
                )
                ax.text(
                    0.985,
                    0.82,
                    f"max {spec.max_prb} PRB",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=11,
                    fontweight="bold",
                    color=COLORS["maxline"],
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.8},
                )
                ax.set_ylim(0, spec.max_prb * 1.10)
            elif metric == "snr_db":
                ax.axhline(
                    24.5,
                    color=COLORS["maxline"],
                    linestyle="--",
                    linewidth=1.8,
                    alpha=0.70,
                )
                ax.text(
                    0.985,
                    0.80,
                    "OAI MCS28 threshold ≈24.5 dB",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=11,
                    fontweight="bold",
                    color=COLORS["maxline"],
                )
                ax.set_ylim(20, max(55, hi))
            else:
                ax.set_ylim(0, max(1.0, hi))

        stat = summarize_series(df[metric]) if metric in df.columns else "no data"
        metric_name = METRIC_NAMES.get(metric, metric)
        ax.set_title(
            f"{spec.label}     {metric_name}: {stat}",
            loc="left",
            fontsize=15,
            fontweight="bold",
            pad=7,
        )
        ax.set_ylabel(y_label, fontsize=13, fontweight="bold", labelpad=8)
        ax.grid(True, axis="both", color=COLORS["grid"], alpha=0.50, linewidth=1.0)
        ax.tick_params(axis="both", which="major", labelsize=11, width=1.4, length=5)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")
        for spine in ax.spines.values():
            spine.set_linewidth(1.4)
        if source_notes and source_notes[idx]:
            ax.text(
                0.01,
                0.35,
                source_notes[idx],
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=10.5,
                fontweight="bold",
                color=COLORS["note"],
            )

    axes[-1].set_xlabel("Elapsed time in run / trace (s)", fontsize=14, fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{file_stem}.{ext}", dpi=420, bbox_inches="tight")
    plt.close(fig)


def write_source_notes(
    grants: dict[str, pd.DataFrame],
    nets: dict[str, pd.DataFrame],
    snrs: dict[str, pd.DataFrame],
) -> None:
    out = OUT_DIR / "scheduler_panel_sources.md"
    rows = []
    for spec in RUNS:
        g = grants[spec.label]
        n = nets[spec.label]
        s = snrs[spec.label]
        rows.append(
            {
                "run": spec.label.replace("\n", " "),
                "grant_source": str(spec.grant_path.relative_to(ROOT)),
                "network_source": str(spec.network_path.relative_to(ROOT)),
                "snr_source": str(spec.snr_path.relative_to(ROOT)) if spec.snr_path else "not captured",
                "snr_note": spec.snr_note,
                "mcs_p50": np.nanpercentile(g["mcs"], 50) if not g.empty else np.nan,
                "prb_p50": np.nanpercentile(g["prb"], 50) if not g.empty else np.nan,
                "scheduled_mbps_p50": np.nanpercentile(g["scheduled_mbps"], 50)
                if not g.empty
                else np.nan,
                "ue_tunnel_tx_mbps_p50": np.nanpercentile(n["tx_mbps"], 50)
                if not n.empty
                else np.nan,
                "snr_db_p50": np.nanpercentile(s["snr_db"], 50) if not s.empty else np.nan,
            }
        )

    with out.open("w") as f:
        f.write("# Scheduler panel plot sources\n\n")
        f.write(
            "These are the exact files used for the 4-row slide plots. MCS/PRB are "
            "from UE-visible uplink DCI grant windows; throughput is UE tunnel TX; "
            "SNR is gNB PUSCH SNR where captured. The corrected drivable UL-heavy "
            "106PRB and drivable 273PRB runs did not capture gNB PUSCH SNR, so their "
            "SNR rows use same-configuration RFsim controls only to show the channel "
            "remained high-SNR. MCS, PRB, and UE tunnel throughput for those rows still "
            "come from the corrected drivable CARLA runs.\n\n"
        )
        f.write(
            "| Run | MCS p50 | PRB p50 | Scheduled Mbps p50 | UE tunnel TX Mbps p50 | SNR p50 | SNR source note |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---|\n")
        for r in rows:
            f.write(
                f"| {r['run']} | {r['mcs_p50']:.1f} | {r['prb_p50']:.1f} | "
                f"{r['scheduled_mbps_p50']:.1f} | {r['ue_tunnel_tx_mbps_p50']:.1f} | "
                f"{r['snr_db_p50']:.1f} | {r['snr_note']} |\n"
            )
        f.write("\n## Full source paths\n\n")
        for r in rows:
            f.write(f"### {r['run']}\n\n")
            f.write(f"- Grant source: `{r['grant_source']}`\n")
            f.write(f"- Network source: `{r['network_source']}`\n")
            f.write(f"- SNR source: `{r['snr_source']}`\n\n")

    csv_out = OUT_DIR / "scheduler_panel_summary.csv"
    with csv_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def crop_carla_window(datasets: dict[str, pd.DataFrame], window_s: float) -> dict[str, pd.DataFrame]:
    """Crop CARLA rows to the first N seconds; leave the short iperf row intact."""
    out: dict[str, pd.DataFrame] = {}
    carla_labels = {spec.label for spec in RUNS if spec.summary_path is not None}
    for label, df in datasets.items():
        if label in carla_labels:
            out[label] = df[df["t"].le(window_s)].copy()
        else:
            out[label] = df.copy()
    return out


def make_cropped_plots(
    window_s: float,
    grants: dict[str, pd.DataFrame],
    nets: dict[str, pd.DataFrame],
    snrs: dict[str, pd.DataFrame],
) -> None:
    suffix = f"first{int(window_s)}s"
    make_metric_plot(
        metric="mcs",
        title=f"Uplink scheduler MCS continuous trace — first {int(window_s)}s of CARLA runs",
        y_label="Avg MCS / 1s",
        file_stem=f"scheduler_panel_mcs_timeseries_{suffix}",
        datasets=crop_carla_window(grants, window_s),
        carla_window_s=window_s,
    )
    make_metric_plot(
        metric="prb",
        title=f"Uplink PRB allocation continuous trace — first {int(window_s)}s of CARLA runs",
        y_label="Avg PRBs / 1s",
        file_stem=f"scheduler_panel_prb_timeseries_{suffix}",
        datasets=crop_carla_window(grants, window_s),
        carla_window_s=window_s,
    )
    make_metric_plot(
        metric="tx_mbps",
        title=f"UE tunnel uplink throughput — first {int(window_s)}s of CARLA runs",
        y_label="TX (Mbps)",
        file_stem=f"scheduler_panel_uplink_throughput_timeseries_{suffix}",
        datasets=crop_carla_window(nets, window_s),
        carla_window_s=window_s,
    )
    make_metric_plot(
        metric="snr_db",
        title=f"gNB-observed uplink SNR — first {int(window_s)}s of CARLA runs",
        y_label="SNR (dB)",
        file_stem=f"scheduler_panel_snr_timeseries_{suffix}",
        datasets=crop_carla_window(snrs, window_s),
        source_notes=[spec.snr_note for spec in RUNS],
        carla_window_s=window_s,
    )


def main() -> None:
    validate_run_specs()
    grants = {spec.label: load_grants(spec) for spec in RUNS}
    nets = {spec.label: load_network(spec) for spec in RUNS}
    snrs = {spec.label: load_snr(spec) for spec in RUNS}

    make_metric_plot(
        metric="mcs",
        title="Uplink scheduler MCS over the full trace",
        y_label="Avg MCS / 1s",
        file_stem="scheduler_panel_mcs_timeseries",
        datasets=grants,
    )
    make_metric_plot(
        metric="prb",
        title="Uplink PRB allocation over the full trace",
        y_label="Avg PRBs / 1s",
        file_stem="scheduler_panel_prb_timeseries",
        datasets=grants,
    )
    make_metric_plot(
        metric="tx_mbps",
        title="UE tunnel uplink throughput over the full trace",
        y_label="TX (Mbps)",
        file_stem="scheduler_panel_uplink_throughput_timeseries",
        datasets=nets,
    )
    make_metric_plot(
        metric="snr_db",
        title="gNB-observed uplink SNR stays high in RFsim controls",
        y_label="SNR (dB)",
        file_stem="scheduler_panel_snr_timeseries",
        datasets=snrs,
        source_notes=[spec.snr_note for spec in RUNS],
    )

    make_cropped_plots(250.0, grants, nets, snrs)

    write_source_notes(grants, nets, snrs)
    for p in sorted(OUT_DIR.glob("scheduler_panel_*")):
        print(p)


if __name__ == "__main__":
    main()
