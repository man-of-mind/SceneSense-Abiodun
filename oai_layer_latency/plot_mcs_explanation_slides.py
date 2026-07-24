#!/usr/bin/env python3
"""Generate cleaner MCS explanation plots for the OAI latency deck.

Outputs:
  - observed CARLA low-MCS RLC drain for one ~1 MB feature burst;
  - MCS table + theoretical throughput ceiling from OAI's NR MCS table 0.

The goal is slide clarity, not an exhaustive PHY simulator. The throughput
ceiling uses the same PRB/time-resource approximation as the earlier plot, but
the MCS entries are taken directly from OAI's copy of 3GPP 38.214 table
5.1.3.1-1 (`Table_51311`).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

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
        "axes.labelsize": 14,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.titlesize": 16,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.titlesize": 22,
        "figure.dpi": 220,
        "savefig.dpi": 420,
        "axes.linewidth": 1.35,
        "lines.linewidth": 3.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


ROOT = Path(__file__).resolve().parents[1]
TRACE_ROOT = ROOT / "metrics_logs/scenesense_ttracer"
OUT_DIR = ROOT / "oai_layer_latency/plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CARLA_ADAPTIVE_RUN = TRACE_ROOT / "downlink_oai_bw273_mu1_ttracer_fps10_layerinstr_20260723_145921"
OAI_MCS_SOURCE = ROOT / "OAI/openairinterface5g/openair2/LAYER2/NR_MAC_COMMON/nr_mac_common.c"

CARLA_RED = "#D1495B"
BLUE = "#2E86AB"
ORANGE = "#EDAE49"
GREEN = "#059669"
GRID = "#CBD5E1"
TEXT = "#0F172A"
NOTE = "#475569"


def parse_clock(series: pd.Series) -> pd.Series:
    parts = series.astype(str).str.split(":", expand=True)
    sec = (
        pd.to_numeric(parts[0], errors="coerce") * 3600
        + pd.to_numeric(parts[1], errors="coerce") * 60
        + pd.to_numeric(parts[2], errors="coerce")
    )
    wraps = sec.diff().lt(-43200).fillna(False).cumsum() * 86400
    elapsed = sec + wraps
    return elapsed - elapsed.min()


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="both", color=GRID, linewidth=1.0, alpha=0.55)
    ax.tick_params(axis="both", which="major", labelsize=11, width=1.3, length=5)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.35)


def load_rlc_lcid4(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "ue/csv/NRUE_MAC_RLC_BUFFER_STATUS.csv"
    chunks = []
    for chunk in pd.read_csv(path, usecols=["time", "lcid", "bytes_in_buffer"], chunksize=750_000):
        chunk["lcid"] = pd.to_numeric(chunk["lcid"], errors="coerce")
        chunks.append(chunk[chunk["lcid"].eq(4)][["time", "bytes_in_buffer"]])
    df = pd.concat(chunks, ignore_index=True)
    df["t"] = parse_clock(df["time"])
    df["kb"] = pd.to_numeric(df["bytes_in_buffer"], errors="coerce") / 1024.0
    # Keep only points where the buffer value changes, otherwise the curve is
    # visually overplotted with millions of repeated samples.
    df = df[df["kb"].ne(df["kb"].shift())].reset_index(drop=True)
    return df[["t", "kb"]]


def find_clean_decay(df: pd.DataFrame) -> pd.DataFrame:
    t = df["t"].to_numpy()
    b = df["kb"].to_numpy()
    starts = np.where((b > 950) & (np.r_[True, b[:-1] < 500]))[0]
    best = None
    best_score = None
    for start in starts:
        end = start
        while end < len(b) - 1 and t[end] - t[start] < 0.45 and b[end] > 100:
            end += 1
        if end <= start or b[end] > 120:
            continue
        segment = b[start : end + 1]
        large_increases = int(np.sum(np.diff(segment) > 50))
        duration_ms = (t[end] - t[start]) * 1000
        # Prefer a simple monotonic-looking decay around 150 ms.
        score = (large_increases, abs(duration_ms - 150), start)
        if best_score is None or score < best_score:
            best_score = score
            best = (start, end)
    if best is None:
        raise RuntimeError("Could not find a clean RLC decay window")
    start, end = best
    # Include a short pre/post margin so the step into the queue is visible.
    t0 = t[start] - 0.025
    t1 = t[end] + 0.045
    out = df[df["t"].between(t0, t1)].copy()
    out["window_ms"] = (out["t"] - t0) * 1000
    return out


def load_grant_summary(run_dir: Path) -> dict[str, float]:
    md = run_dir / "layer_latency/uplink_layer_latency.md"
    text = md.read_text()
    out: dict[str, float] = {}
    for key, pattern in {
        "mcs_p50": r"grant MCS: p50=([0-9.]+)",
        "mcs_p95": r"grant MCS: p50=[0-9.]+\\s+p95=([0-9.]+)",
        "prb_p50": r"grant PRB: p50=([0-9.]+)",
        "tbs_p50": r"grant TBS: p50=([0-9.]+) B",
    }.items():
        m = re.search(pattern, text)
        if m:
            out[key] = float(m.group(1))
    return out


def plot_observed_low_mcs_drain() -> None:
    df = load_rlc_lcid4(CARLA_ADAPTIVE_RUN)
    win = find_clean_decay(df)
    grants = load_grant_summary(CARLA_ADAPTIVE_RUN)

    above = win[win["kb"].gt(950)]
    below = win[win["kb"].lt(100)]
    t_start = float(above["window_ms"].iloc[0])
    kb_start = float(above["kb"].iloc[0])
    t_end = float(below["window_ms"].iloc[0])
    kb_end = float(below["kb"].iloc[0])
    drain_ms = t_end - t_start
    drained_kb = kb_start - kb_end
    inst_mbps = drained_kb * 1024 * 8 / (drain_ms / 1000) / 1e6

    fig, ax = plt.subplots(figsize=(12.4, 6.3))
    fig.subplots_adjust(left=0.105, right=0.975, top=0.84, bottom=0.16)
    fig.suptitle("Observed CARLA low-MCS drain: one feature burst sits in UE RLC", y=0.965, fontweight="bold")

    ax.plot(win["window_ms"], win["kb"], color=CARLA_RED, linewidth=3.2, solid_capstyle="round")
    ax.fill_between(win["window_ms"], 0, win["kb"], color=CARLA_RED, alpha=0.12)
    ax.axhline(1024, color=NOTE, linestyle="--", linewidth=1.7, alpha=0.75)
    ax.text(
        0.985,
        0.91,
        "~1 MB feature frame",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=NOTE,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2},
    )
    ax.annotate(
        f"Observed drain: {drain_ms:.0f} ms\\n"
        f"{kb_start:.0f} KB → {kb_end:.0f} KB\\n"
        f"burst-slope ≈ {inst_mbps:.0f} Mbps",
        xy=((t_start + t_end) / 2, (kb_start + kb_end) / 2),
        xytext=(0.56, 0.58),
        textcoords="axes fraction",
        ha="left",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=TEXT,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": CARLA_RED, "alpha": 0.92},
        arrowprops={"arrowstyle": "->", "color": CARLA_RED, "linewidth": 2.0},
    )
    ax.text(
        0.03,
        0.12,
        f"Scheduler context: MCS p50={grants.get('mcs_p50', float('nan')):.0f}, "
        f"p95={grants.get('mcs_p95', float('nan')):.0f}; "
        f"PRB p50={grants.get('prb_p50', float('nan')):.0f}; "
        f"TBS p50={grants.get('tbs_p50', float('nan')):.0f} B",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color=TEXT,
        bbox={"facecolor": "white", "edgecolor": GRID, "alpha": 0.9, "pad": 4},
    )
    ax.set_xlabel("Time within selected burst window (ms)")
    ax.set_ylabel("UE RLC LCID4 occupancy (KB)")
    ax.set_ylim(0, 1160)
    style_axis(ax)
    fig.text(
        0.5,
        0.045,
        "This is one observed CARLA burst. Run-average RLC wait is ~100 ms because the closed-loop workload includes idle gaps and repeated bursts.",
        ha="center",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color=NOTE,
    )
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"carla_low_mcs_observed_rlc_drain.{ext}", bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(
        [
            {
                "run": CARLA_ADAPTIVE_RUN.name,
                "start_kb": kb_start,
                "end_kb": kb_end,
                "drain_ms": drain_ms,
                "burst_slope_mbps": inst_mbps,
                **grants,
            }
        ]
    ).to_csv(OUT_DIR / "carla_low_mcs_observed_rlc_drain_summary.csv", index=False)


def load_oai_mcs_table() -> pd.DataFrame:
    text = OAI_MCS_SOURCE.read_text()
    m = re.search(r"Table_51311\[32\]\[2\]\s*=\s*\{(.*?)\};", text, re.S)
    if not m:
        raise RuntimeError("Could not locate Table_51311 in OAI source")
    pairs = re.findall(r"\{\s*(\d+)\s*,\s*(\d+)\s*\}", m.group(1))
    rows = []
    for idx, (qm, r10x) in enumerate(pairs):
        qm_i = int(qm)
        r10x_i = int(r10x)
        if r10x_i == 0:
            mod = "reserved"
            code_rate = 0.0
            se = 0.0
        else:
            mod = {2: "QPSK", 4: "16QAM", 6: "64QAM", 8: "256QAM"}.get(qm_i, f"Qm={qm_i}")
            # OAI stores 10x the 38.214 code-rate numerator. Convert to R in [0,1].
            code_rate = r10x_i / 10240.0
            se = qm_i * code_rate
        rows.append(
            {
                "mcs": idx,
                "modulation": mod,
                "Qm": qm_i,
                "code_rate": code_rate,
                "spectral_eff_bits_per_re": se,
            }
        )
    return pd.DataFrame(rows)


def plot_mcs_table_and_ceiling() -> None:
    table = load_oai_mcs_table()
    selected = table[table["mcs"].isin([4, 7, 8, 13, 16, 28])].copy()
    selected["role"] = [
        "CARLA 273 p50",
        "CARLA 106 p50",
        "QPSK high",
        "mid 16QAM",
        "16QAM high",
        "iperf/fixed",
    ]

    # Same approximation as the previous plot, now using OAI's actual MCS table.
    # This is a PRB-time ceiling, useful for ratios; deployment TDD/control
    # overhead scales the absolute wall-clock rate down.
    n_prb = 273
    re_per_s = n_prb * 12 * 12 * 2000 * 0.85
    selected["ceiling_mbps"] = re_per_s * selected["spectral_eff_bits_per_re"] / 1e6
    colors = [CARLA_RED, "#F97316", "#F59E0B", ORANGE, GREEN, BLUE]

    fig, (ax_table, ax_bar) = plt.subplots(
        1,
        2,
        figsize=(15.6, 6.25),
        gridspec_kw={"width_ratios": [1.28, 1.0]},
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.84, bottom=0.16, wspace=0.19)
    fig.suptitle("Why MCS matters: it sets bits per resource element, so it caps drain rate", y=0.965, fontweight="bold")

    ax_table.axis("off")
    table_rows = []
    for _, row in selected.iterrows():
        table_rows.append(
            [
                f"{int(row['mcs'])}",
                row["modulation"],
                f"{int(row['Qm'])}",
                f"{row['code_rate']:.3f}",
                f"{row['spectral_eff_bits_per_re']:.2f}",
                row["role"],
            ]
        )
    tbl = ax_table.table(
        cellText=table_rows,
        colLabels=["MCS", "Mod.", "Qm", "Code rate", "bits/RE", "Why shown"],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.10, 0.17, 0.10, 0.18, 0.15, 0.30],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    tbl.scale(1.0, 1.68)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.8)
        cell.set_edgecolor("#CBD5E1")
        if r == 0:
            cell.set_facecolor("#E2E8F0")
            cell.set_text_props(weight="bold", color=TEXT)
        else:
            cell.set_text_props(weight="bold", color=TEXT)
            if int(selected.iloc[r - 1]["mcs"]) in (4, 7):
                cell.set_facecolor("#FEF2F2")
            elif int(selected.iloc[r - 1]["mcs"]) == 28:
                cell.set_facecolor("#E0F2FE")
            else:
                cell.set_facecolor("white")
    ax_table.set_title("A. OAI NR MCS table 0 entries used in this run", loc="left", pad=12)
    ax_table.text(
        0.0,
        0.055,
        "Formula: bits/RE = Qm × code-rate. Low MCS lowers both modulation order and coding rate.",
        transform=ax_table.transAxes,
        ha="left",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color=NOTE,
    )

    x = np.arange(len(selected))
    ax_bar.bar(x, selected["ceiling_mbps"], color=colors, width=0.68)
    for i, v in enumerate(selected["ceiling_mbps"]):
        ax_bar.text(i, v + 8, f"{v:.0f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"MCS {int(m)}" for m in selected["mcs"]], rotation=0)
    ax_bar.set_ylabel("Estimated UL ceiling at same PRB-time (Mbps)")
    ax_bar.set_title("B. Same 273 PRBs, different MCS → different drain ceiling", loc="left", pad=12)
    ax_bar.annotate(
        f"MCS28 is {selected.loc[selected['mcs'].eq(28), 'spectral_eff_bits_per_re'].iloc[0] / selected.loc[selected['mcs'].eq(4), 'spectral_eff_bits_per_re'].iloc[0]:.1f}× "
        "MCS4\\nin bits/RE",
        xy=(5, selected["ceiling_mbps"].iloc[-1]),
        xytext=(2.65, selected["ceiling_mbps"].iloc[-1] * 0.77),
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=TEXT,
        arrowprops={"arrowstyle": "->", "color": TEXT, "linewidth": 2},
        bbox={"facecolor": "white", "edgecolor": GRID, "alpha": 0.92, "pad": 4},
    )
    style_axis(ax_bar)
    ax_bar.set_ylim(0, max(selected["ceiling_mbps"]) * 1.22)
    fig.text(
        0.5,
        0.045,
        "Absolute wall-clock throughput also depends on TDD pattern, grants, DMRS/control overhead, and retransmissions; the key point is the MCS-driven capacity ratio.",
        ha="center",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color=NOTE,
    )

    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"mcs_table_throughput_ceiling.{ext}", bbox_inches="tight")
    plt.close(fig)
    selected.to_csv(OUT_DIR / "mcs_table_throughput_ceiling.csv", index=False)


def main() -> None:
    plot_observed_low_mcs_drain()
    plot_mcs_table_and_ceiling()
    for path in [
        OUT_DIR / "carla_low_mcs_observed_rlc_drain.pdf",
        OUT_DIR / "mcs_table_throughput_ceiling.pdf",
        OUT_DIR / "mcs_table_throughput_ceiling.csv",
    ]:
        print(path)


if __name__ == "__main__":
    main()
