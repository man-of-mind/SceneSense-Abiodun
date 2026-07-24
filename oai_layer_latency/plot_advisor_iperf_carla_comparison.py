#!/usr/bin/env python3
"""Clean advisor-facing comparison: iperf vs CARLA split tensor uplink.

Question being answered:
  Is CARLA getting low MCS because it is classified as a stricter traffic class,
  or because its large/sparse BSR backlog pattern drives OAI's adaptive
  BLER/OLLA MCS selector into a low-MCS state?

This script uses only the reportable T-tracer CSVs:
  - iperf validation: smooth UDP traffic
  - CARLA layer-instrumented run: ~1 MB split-feature bursts
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRACE_ROOT = ROOT / "metrics_logs/scenesense_ttracer"
OUT_DIR = ROOT / "oai_layer_latency/plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = {
    "iperf\nsmooth UDP": TRACE_ROOT / "validate_20260722_175810_iperf_ul",
    "CARLA\n1 MB bursts": TRACE_ROOT / "downlink_oai_bw273_mu1_ttracer_fps10_layerinstr_20260723_145921",
}


def read_column(path: Path, column: str, scale: float = 1.0) -> np.ndarray:
    values = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                values.append(float(row[column]) * scale)
            except (KeyError, TypeError, ValueError):
                pass
    return np.asarray(values, dtype=float)


def summarize_run(run_dir: Path) -> dict[str, float]:
    bsr = run_dir / "ue/csv/NRUE_MAC_BSR_STATUS.csv"
    pusch = run_dir / "gnb/csv/GNB_MAC_PUSCH_POWER_CONTROL.csv"

    lcg1_kb = read_column(bsr, "lcg1_bytes", 1.0 / 1024.0)
    snr_db = read_column(pusch, "snrx10", 0.1)
    phr = read_column(pusch, "phr")
    rb = read_column(pusch, "rbSize")
    mcs = read_column(pusch, "mcs")
    tbs = read_column(pusch, "tb_size")

    return {
        "bsr_p50_kb": float(np.percentile(lcg1_kb, 50)),
        "bsr_p95_kb": float(np.percentile(lcg1_kb, 95)),
        "bsr_max_kb": float(np.max(lcg1_kb)),
        "snr_p50_db": float(np.percentile(snr_db, 50)),
        "snr_p05_db": float(np.percentile(snr_db, 5)),
        "snr_p95_db": float(np.percentile(snr_db, 95)),
        "phr_p50": float(np.percentile(phr, 50)),
        "rb_p50": float(np.percentile(rb, 50)),
        "rb_p95": float(np.percentile(rb, 95)),
        "mcs_p50": float(np.percentile(mcs, 50)),
        "mcs_p95": float(np.percentile(mcs, 95)),
        "tbs_p50_b": float(np.percentile(tbs, 50)),
        "tbs_p95_b": float(np.percentile(tbs, 95)),
        "samples_bsr": int(lcg1_kb.size),
        "samples_pusch": int(mcs.size),
    }


def write_summary(stats: dict[str, dict[str, float]]) -> None:
    out = OUT_DIR / "advisor_iperf_carla_summary.csv"
    cols = [
        "label",
        "samples_bsr",
        "samples_pusch",
        "bsr_p50_kb",
        "bsr_p95_kb",
        "bsr_max_kb",
        "snr_p50_db",
        "snr_p05_db",
        "snr_p95_db",
        "phr_p50",
        "rb_p50",
        "rb_p95",
        "mcs_p50",
        "mcs_p95",
        "tbs_p50_b",
        "tbs_p95_b",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for label, row in stats.items():
            writer.writerow({"label": label, **row})


def make_plot(stats: dict[str, dict[str, float]]) -> None:
    labels = list(stats)
    colors = ["#2E86AB", "#D1495B"]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(14.2, 8.1))
    fig.subplots_adjust(left=0.065, right=0.985, top=0.875, bottom=0.125, hspace=0.42, wspace=0.25)
    fig.suptitle(
        "Advisor check: same bearer and high SNR, but CARLA reports huge sparse BSR backlog",
        fontsize=16,
        fontweight="bold",
    )

    # A. BSR backlog
    ax = axes[0, 0]
    w = 0.34
    bsr_p50 = [stats[l]["bsr_p50_kb"] for l in labels]
    bsr_p95 = [stats[l]["bsr_p95_kb"] for l in labels]
    ax.bar(x - w / 2, bsr_p50, width=w, color=colors, alpha=0.92, label="p50")
    ax.bar(x + w / 2, bsr_p95, width=w, color=colors, alpha=0.45, hatch="//", label="p95")
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_ylabel("BSR LCG1 backlog (KB, log scale)")
    ax.set_title("A. UE BSR: CARLA backlog is ~500× larger", fontsize=12.5)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    for i, v in enumerate(bsr_p50):
        ax.text(i - w / 2, v * 1.12, f"{v:.1f} KB", ha="center", va="bottom", fontsize=9)
    ax.text(1 + w / 2, bsr_p95[1] * 1.12, f"{bsr_p95[1]:.0f} KB", ha="center", va="bottom", fontsize=9)

    # B. MCS
    ax = axes[0, 1]
    mcs_p50 = [stats[l]["mcs_p50"] for l in labels]
    mcs_p95 = [stats[l]["mcs_p95"] for l in labels]
    ax.bar(x - w / 2, mcs_p50, width=w, color=colors, alpha=0.92, label="p50")
    ax.bar(x + w / 2, mcs_p95, width=w, color=colors, alpha=0.45, hatch="//", label="p95")
    ax.axhspan(0, 9.5, color="#FFF2CC", alpha=0.65, label="QPSK region")
    ax.axhline(28, color="#374151", linestyle="--", linewidth=1.2)
    ax.text(0.05, 28.45, "MCS28 / 64QAM", fontsize=9, color="#374151")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 31)
    ax.set_ylabel("UL MCS")
    ax.set_title("B. Adaptive MCS collapses only for CARLA bursts", fontsize=12.5)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    for i, v in enumerate(mcs_p50):
        ax.text(i - w / 2, v + 0.7, f"{v:.0f}", ha="center", va="bottom", fontsize=10)

    # C. SNR/PHR sanity
    ax = axes[1, 0]
    snr = [stats[l]["snr_p50_db"] for l in labels]
    ax.bar(x, snr, width=0.48, color=colors, alpha=0.9)
    ax.axhline(24.5, color="#111827", linestyle="--", linewidth=1.2)
    ax.text(0.04, 25.5, "OAI MCS28 SNR threshold ≈24.5 dB", fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 56)
    ax.set_ylabel("gNB PUSCH SNR p50 (dB)")
    ax.set_title("C. Channel quality is not the differentiator", fontsize=12.5)
    ax.grid(True, axis="y", alpha=0.25)
    for i, l in enumerate(labels):
        ax.text(i, snr[i] + 1.0, f"SNR {snr[i]:.1f} dB\nPHR {stats[l]['phr_p50']:.0f}", ha="center", fontsize=9)

    # D. Grant payload: more PRBs but less TBS under low MCS
    ax = axes[1, 1]
    tbs = [stats[l]["tbs_p50_b"] for l in labels]
    ax.bar(x, tbs, width=0.48, color=colors, alpha=0.9)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(tbs) * 1.30)
    ax.set_ylabel("PUSCH grant TBS p50 (bytes)")
    ax.set_title("D. CARLA gets max PRBs but smaller TBS due to low MCS", fontsize=12.5)
    ax.grid(True, axis="y", alpha=0.25)
    for i, l in enumerate(labels):
        ax.text(i, tbs[i] + max(tbs) * 0.035, f"TBS {tbs[i]:.0f} B\nPRB {stats[l]['rb_p50']:.0f}", ha="center", fontsize=9)

    fig.text(
        0.5,
        0.035,
        "Both flows are observed on data bearer LCID 4 / LCG 1; current evidence supports BLER/OLLA cadence/backlog-driven low MCS, not traffic-class-specific PER.",
        ha="center",
        fontsize=10.5,
        color="#334155",
    )

    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"advisor_iperf_vs_carla_bsr_mcs.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    stats = {label: summarize_run(path) for label, path in RUNS.items()}
    write_summary(stats)
    make_plot(stats)
    print(f"wrote {OUT_DIR / 'advisor_iperf_vs_carla_bsr_mcs.png'}")
    for label, row in stats.items():
        print(
            f"{label}: BSR p50={row['bsr_p50_kb']:.1f}KB p95={row['bsr_p95_kb']:.1f}KB, "
            f"MCS p50/p95={row['mcs_p50']:.0f}/{row['mcs_p95']:.0f}, "
            f"SNR p50={row['snr_p50_db']:.1f}dB, TBS p50={row['tbs_p50_b']:.0f}B"
        )


if __name__ == "__main__":
    main()
