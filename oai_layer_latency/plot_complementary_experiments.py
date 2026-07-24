#!/usr/bin/env python3
"""Comparison plots for the CARLA/OAI layer-latency complementary experiments.

Inputs:
  - frontend metrics under downlink_latency_fps/runs/**/streams/<run_group>_metrics.csv
  - latency-profile T-tracer CSVs under metrics_logs/scenesense_ttracer/<run_group>/{ue,gnb}/csv

Outputs:
  - oai_layer_latency/plots/complementary_latency_summary.{pdf,png}
  - oai_layer_latency/plots/complementary_mcs_prb_summary.{pdf,png}
  - oai_layer_latency/plots/complementary_rlc_buffer_timeseries.{pdf,png}
  - oai_layer_latency/plots/complementary_gnb_snr_timeseries.{pdf,png}
"""
from __future__ import annotations

import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


AB = Path(__file__).resolve().parents[1]
TT = AB / "metrics_logs" / "scenesense_ttracer"
RUNS = AB / "downlink_latency_fps" / "runs"
OUT = AB / "oai_layer_latency" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

RUNS_TO_COMPARE = [
    {
        "label": "273 adaptive\nuint8",
        "short": "273-adapt-u8",
        "run_group": "downlink_oai_bw273_mu1_ttracer_fps10_layerinstr_20260722_191024",
        "color": "#D1495B",
        "prb_ceiling": 273,
    },
    {
        "label": "273 adaptive\nuint4",
        "short": "273-adapt-u4",
        "run_group": "downlink_oai_bw273_mu1_ttracer_int4_adaptive_fps10_int4_adaptive_20260723",
        "color": "#F6AE2D",
        "prb_ceiling": 273,
    },
    {
        "label": "273 fixed\nMCS28",
        "short": "273-fixed28-u8",
        "run_group": "downlink_oai_bw273_mu1_ttracer_forcemcs28_fps10_forcemcs28_bw273_20260723",
        "color": "#2E86AB",
        "prb_ceiling": 273,
    },
    {
        "label": "106 fixed\nMCS28",
        "short": "106-fixed28-u8",
        "run_group": "downlink_oai_ulheavy_106_ttracer_forcemcs28_fps10_forcemcs28_ulheavy106_20260723",
        "color": "#54A24B",
        "prb_ceiling": 106,
    },
    {
        "label": "106 default\nAE128 u6 r0.5",
        "short": "106-adapt-ae128-u6-r05",
        "run_group": "downlink_oai_default106_ttracer_ae128_u6_roi05_fps10_ae128_u6_roi05_default106_20260723",
        "color": "#8A60B0",
        "prb_ceiling": 106,
    },
]


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def pct(values, q):
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return float("nan")
    return float(np.percentile(vals, q))


def clock_to_seconds(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series.astype(str), format="%H:%M:%S.%f", errors="coerce")
    seconds = (
        parsed.dt.hour * 3600.0
        + parsed.dt.minute * 60.0
        + parsed.dt.second
        + parsed.dt.microsecond / 1_000_000.0
    )
    out = []
    offset = 0.0
    prev = None
    for value in seconds:
        if pd.isna(value):
            out.append(float("nan"))
            continue
        v = float(value) + offset
        if prev is not None and v + 12 * 3600 < prev:
            offset += 24 * 3600
            v = float(value) + offset
        out.append(v)
        prev = v
    return pd.Series(out, index=series.index, dtype="float64")


def find_metrics(run_group: str) -> Path:
    matches = sorted(RUNS.glob(f"**/streams/{run_group}_metrics.csv"))
    if not matches:
        raise FileNotFoundError(run_group)
    return matches[-1]


def frontend_summary(run_group: str) -> dict[str, float]:
    p = find_metrics(run_group)
    df = pd.read_csv(p)
    rec = df[num(df["result_received"]).fillna(0).astype(bool)].copy()
    upload = pd.Series(dtype=float)
    if not rec.empty and {"t_edge_recv_wall_s", "t_front_send_wall_s"}.issubset(rec.columns):
        upload = (num(rec["t_edge_recv_wall_s"]) - num(rec["t_front_send_wall_s"])) * 1000.0
    return {
        "frames": float(len(df)),
        "received": float(num(df["result_received"]).fillna(0).sum()),
        "delivery": float(num(df["result_received"]).fillna(0).mean()),
        "feature_kb": pct(num(df["feature_payload_bytes"]) / 1024.0, 50),
        "feature_chunks": pct(num(df["feature_payload_chunks"]), 50),
        "front_ms": pct(num(df["front_ms"]), 50),
        "rtt_ms": pct(num(rec["round_trip_result_recv_ms"]), 50) if not rec.empty else float("nan"),
        "rtt_p95_ms": pct(num(rec["round_trip_result_recv_ms"]), 95) if not rec.empty else float("nan"),
        "back_ms": pct(num(rec["back_ms"]), 50) if not rec.empty else float("nan"),
        "downlink_ms": pct(num(rec["result_send_to_recv_ms_wall"]), 50) if not rec.empty else float("nan"),
        "uplink_ms": pct(upload, 50),
    }


def layer_summary(run_group: str) -> dict[str, float]:
    ue = TT / run_group / "ue" / "csv"
    gnb = TT / run_group / "gnb" / "csv"
    out = {
        "rlc_occ_mean_kb": float("nan"),
        "rlc_occ_p95_kb": float("nan"),
        "rlc_occ_max_kb": float("nan"),
        "rlc_queue_ms": float("nan"),
        "mcs_p50": float("nan"),
        "mcs_p95": float("nan"),
        "prb_p50": float("nan"),
        "tbs_p50": float("nan"),
        "ran_p50_ms": float("nan"),
        "ran_p95_ms": float("nan"),
        "snr_p50_db": float("nan"),
    }

    rlc_p = ue / "NRUE_MAC_RLC_BUFFER_STATUS.csv"
    bsr_p = ue / "NRUE_MAC_BSR_STATUS.csv"
    if rlc_p.exists():
        rlc = pd.read_csv(rlc_p, usecols=["time", "lcid", "bytes_in_buffer"])
        rlc = rlc[num(rlc["lcid"]) == 4]
        occ = num(rlc["bytes_in_buffer"]) / 1024.0
        out["rlc_occ_mean_kb"] = float(occ.mean())
        out["rlc_occ_p95_kb"] = pct(occ, 95)
        out["rlc_occ_max_kb"] = float(occ.max())
        if bsr_p.exists():
            bsr = pd.read_csv(bsr_p, usecols=["time", "sdu_bytes"])
            t = clock_to_seconds(bsr["time"])
            dur = float(t.max() - t.min()) if len(t.dropna()) > 1 else float("nan")
            sdu_total = float(num(bsr["sdu_bytes"]).sum())
            drain_bps = sdu_total / dur if dur and math.isfinite(dur) and dur > 0 else float("nan")
            if math.isfinite(drain_bps) and drain_bps > 0:
                out["rlc_queue_ms"] = (out["rlc_occ_mean_kb"] * 1024.0) / drain_bps * 1000.0

    grant_p = ue / "NRUE_MAC_DCI_GRANT.csv"
    if grant_p.exists():
        grant = pd.read_csv(grant_p, usecols=["direction", "mcs", "rb_size", "tbs"])
        grant = grant[num(grant["direction"]) == 1]
        out["mcs_p50"] = pct(num(grant["mcs"]), 50)
        out["mcs_p95"] = pct(num(grant["mcs"]), 95)
        out["prb_p50"] = pct(num(grant["rb_size"]), 50)
        out["tbs_p50"] = pct(num(grant["tbs"]), 50)

    snr_p = gnb / "GNB_MAC_PUSCH_POWER_CONTROL.csv"
    if snr_p.exists():
        snr = pd.read_csv(snr_p, usecols=["snrx10"])
        out["snr_p50_db"] = pct(num(snr["snrx10"]) / 10.0, 50)

    pin_p = ue / "NR_PDCP_TX_SDU.csv"
    gout_p = gnb / "GNB_PDCP_RX_DELIVER.csv"
    if pin_p.exists() and gout_p.exists():
        # Stream via csv to avoid carrying several million timestamp rows in memory.
        trans = []
        with open(pin_p) as f1, open(gout_p) as f2:
            r1, r2 = csv.DictReader(f1), csv.DictReader(f2)
            for a, b in zip(r1, r2):
                try:
                    ba, bb = int(a["sdu_bytes"]), int(b["sdu_bytes"])
                    if ba != bb or ba <= 1000:
                        continue
                    ta = int(a["mono_sec"]) + int(a["mono_nsec"]) / 1e9
                    tb = int(b["mono_sec"]) + int(b["mono_nsec"]) / 1e9
                    dt = (tb - ta) * 1000.0
                    if 0 < dt < 10000:
                        trans.append(dt)
                except (ValueError, KeyError):
                    continue
        out["ran_p50_ms"] = pct(trans, 50)
        out["ran_p95_ms"] = pct(trans, 95)
    return out


def rlc_timeseries(run_group: str) -> pd.DataFrame:
    p = TT / run_group / "ue" / "csv" / "NRUE_MAC_RLC_BUFFER_STATUS.csv"
    df = pd.read_csv(p, usecols=["time", "lcid", "bytes_in_buffer"])
    df = df[num(df["lcid"]) == 4].copy()
    df["t_s"] = clock_to_seconds(df["time"])
    df = df[df["t_s"].notna()]
    df["t_norm"] = df["t_s"] - float(df["t_s"].min())
    df["bin_s"] = (np.floor(num(df["t_norm"]) / 5.0) * 5).astype(int)
    out = df.groupby("bin_s", as_index=False).agg(
        occ_p95_kb=("bytes_in_buffer", lambda x: pct(num(x) / 1024.0, 95)),
        occ_max_kb=("bytes_in_buffer", lambda x: float(num(x).max()) / 1024.0),
    )
    # Trim to active region with a small context.
    active = out["occ_max_kb"] > 1.0
    if active.any():
        lo = max(0, int(out.loc[active, "bin_s"].min()) - 20)
        hi = int(out.loc[active, "bin_s"].max()) + 20
        out = out[(out["bin_s"] >= lo) & (out["bin_s"] <= hi)].copy()
        out["bin_s"] -= lo
    return out


def snr_timeseries(run_group: str) -> pd.DataFrame:
    p = TT / run_group / "gnb" / "csv" / "GNB_MAC_PUSCH_POWER_CONTROL.csv"
    df = pd.read_csv(p, usecols=["time", "snrx10", "mcs", "rbSize"])
    df["t_s"] = clock_to_seconds(df["time"])
    df = df[df["t_s"].notna()].copy()
    df["t_norm"] = df["t_s"] - float(df["t_s"].min())
    df["bin_s"] = (np.floor(num(df["t_norm"]) / 5.0) * 5).astype(int)
    out = df.groupby("bin_s", as_index=False).agg(
        snr_db=("snrx10", lambda x: pct(num(x) / 10.0, 50)),
        mcs=("mcs", lambda x: pct(num(x), 50)),
        rb=("rbSize", lambda x: pct(num(x), 50)),
    )
    return out


def save(fig, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    rows = []
    for spec in RUNS_TO_COMPARE:
        fs = frontend_summary(spec["run_group"])
        ls = layer_summary(spec["run_group"])
        rows.append({**spec, **fs, **ls})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "complementary_experiment_summary.csv", index=False)

    x = np.arange(len(summary))
    labels = summary["label"].tolist()

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.8))
    width = 0.34
    ax[0].bar(x - width / 2, summary["rtt_ms"], width, color="#8D99AE", label="App RTT p50")
    ax[0].bar(x + width / 2, summary["ran_p50_ms"], width, color="#2E86AB", label="RAN uplink p50")
    for i, row in summary.iterrows():
        ax[0].text(i - width / 2, row["rtt_ms"] + 3, f"{row['rtt_ms']:.0f}", ha="center", fontsize=9)
        ax[0].text(i + width / 2, row["ran_p50_ms"] + 3, f"{row['ran_p50_ms']:.0f}", ha="center", fontsize=9)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels)
    ax[0].set_ylabel("Latency (ms)")
    ax[0].set_title("A. Application RTT and true RAN uplink transit")
    ax[0].legend(frameon=False)

    ax[1].bar(x - width / 2, summary["rlc_queue_ms"], width, color="#F6AE2D", label="RLC mean queue")
    ax[1].bar(x + width / 2, summary["feature_kb"], width, color="#B8B8B8", label="Feature payload KB")
    for i, row in summary.iterrows():
        ax[1].text(i - width / 2, row["rlc_queue_ms"] + 3, f"{row['rlc_queue_ms']:.0f}", ha="center", fontsize=9)
        ax[1].text(i + width / 2, row["feature_kb"] + 25, f"{row['feature_kb']:.0f}", ha="center", fontsize=9)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels)
    ax[1].set_ylabel("ms / KB")
    ax[1].set_title("B. Payload relief reduces queue; fixed MCS collapses it")
    ax[1].legend(frameon=False)
    fig.suptitle("CARLA/OAI complementary layer-latency experiments", fontsize=13)
    fig.tight_layout()
    save(fig, "complementary_latency_summary")

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.8))
    ax[0].bar(x, summary["mcs_p50"], color=[c for c in summary["color"]])
    for i, row in summary.iterrows():
        ax[0].text(i, row["mcs_p50"] + 0.6, f"{row['mcs_p50']:.0f}", ha="center", fontsize=9)
    ax[0].axhspan(0, 9, color="#999999", alpha=0.08)
    ax[0].text(0.02, 0.86, "QPSK region\n(MCS 0–9)", transform=ax[0].transAxes, color="dimgray", fontsize=9)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels)
    ax[0].set_ylabel("UL MCS p50")
    ax[0].set_title("A. Adaptive CARLA stays low-MCS; forced control pins 28")

    ax1b = ax[1].twinx()
    ax[1].bar(x - width / 2, summary["prb_p50"], width, color="#4C78A8", label="PRB p50")
    ax1b.bar(x + width / 2, summary["tbs_p50"], width, color="#54A24B", alpha=0.88, label="TBS p50 (bytes)")
    for i, row in summary.iterrows():
        ax[1].text(i - width / 2, row["prb_p50"] + 8, f"{row['prb_p50']:.0f}", ha="center", fontsize=9)
        ax1b.text(i + width / 2, row["tbs_p50"] + 400, f"{row['tbs_p50']:.0f}", ha="center", fontsize=9, color="#2D6B2D")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels)
    ax[1].set_ylabel("PRB p50", color="#4C78A8")
    ax1b.set_ylabel("TBS p50 (bytes)", color="#54A24B")
    ax[1].set_ylim(0, max(float(np.nanmax(summary["prb_p50"])) * 1.16, 120.0))
    ax1b.set_ylim(0, max(float(np.nanmax(summary["tbs_p50"])) * 1.20, 1000.0))
    ax[1].set_title("B. Fixed MCS gives much larger grant payloads")
    h1, l1 = ax[1].get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax[1].legend(
        h1 + h2,
        l1 + l2,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
    )
    fig.suptitle("UL scheduling summary", fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    save(fig, "complementary_mcs_prb_summary")

    fig, ax = plt.subplots(figsize=(12.5, 5.0))
    for spec in RUNS_TO_COMPARE:
        ts = rlc_timeseries(spec["run_group"])
        ax.plot(ts["bin_s"], ts["occ_p95_kb"], color=spec["color"], linewidth=2.0, label=spec["label"].replace("\n", " "))
    ax.set_title("UE RLC data-bearer buffer occupancy over active run (5s bins, p95)")
    ax.set_xlabel("Active-run time, normalized per run (s)")
    ax.set_ylabel("RLC buffer occupancy p95 (KB)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    save(fig, "complementary_rlc_buffer_timeseries")

    fig, axes = plt.subplots(len(RUNS_TO_COMPARE), 1, figsize=(12.5, 7.2), sharex=True, gridspec_kw={"hspace": 0.12})
    for axis, spec in zip(axes, RUNS_TO_COMPARE):
        ts = snr_timeseries(spec["run_group"])
        axis.plot(ts["bin_s"], ts["snr_db"], color=spec["color"], linewidth=1.8)
        axis.set_ylim(49.4, 51.6)
        axis.set_ylabel(spec["short"], rotation=0, ha="right", va="center", labelpad=58, fontsize=9)
        axis.grid(axis="y", alpha=0.25)
        axis.text(
            0.985, 0.72,
            f"p50={summary.loc[summary['short'] == spec['short'], 'snr_p50_db'].iloc[0]:.1f} dB",
            transform=axis.transAxes,
            ha="right",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#DDDDDD", "alpha": 0.9},
        )
    axes[0].set_title("gNB PUSCH SNR over run (RFsim ideal channel; flat by design)")
    axes[-1].set_xlabel("Run time since trace start (s)")
    fig.text(0.018, 0.5, "SNR (dB), 5s p50", va="center", rotation=90)
    save(fig, "complementary_gnb_snr_timeseries")

    print(summary[[
        "short", "delivery", "feature_kb", "rtt_ms", "rtt_p95_ms", "ran_p50_ms", "ran_p95_ms",
        "rlc_queue_ms", "rlc_occ_p95_kb", "mcs_p50", "mcs_p95", "prb_p50", "tbs_p50", "snr_p50_db",
    ]].to_string(index=False))
    print(f"wrote plots to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
