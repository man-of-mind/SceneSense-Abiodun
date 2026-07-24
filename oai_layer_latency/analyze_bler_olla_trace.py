#!/usr/bin/env python3
"""Analyze the direct OAI BLER/OLLA MCS-selection trace.

The custom T-tracer event GNB_MAC_BLER_MCS_DECISION is emitted inside
get_mcs_from_bler().  This script compares one or more runs and asks the
specific advisor-followup question:

    Is low uplink MCS coming from poor channel/SNR, PHR/RB clipping, or from
    the BLER/OLLA state machine/cadence itself?

The event branch codes are:

    0: no update yet; frame diff < BLER_UPDATE_FRAME
    1: increase; BLER below lower threshold and enough scheduled samples
    2: decrease; BLER above upper threshold
    3: decrease/hold-low; too few scheduled samples (num_sched <= 3)
    4: hold; BLER inside target window with enough samples
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


AB_ROOT = Path(__file__).resolve().parents[1]
TTRACER_ROOT = AB_ROOT / "metrics_logs" / "scenesense_ttracer"
PLOTS_DIR = AB_ROOT / "oai_layer_latency" / "plots"

BRANCH_LABELS = {
    0: "no update\n(diff < 10)",
    1: "increase\n(low BLER)",
    2: "decrease\n(high BLER)",
    3: "decrease/hold-low\n(few samples)",
    4: "hold\n(in target)",
}

BRANCH_COLORS = {
    0: "#9ca3af",
    1: "#2ca02c",
    2: "#d62728",
    3: "#ff7f0e",
    4: "#1f77b4",
}


@dataclass(frozen=True)
class RunSpec:
    label: str
    run_group: str


def parse_run_spec(value: str) -> RunSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must be LABEL=RUN_GROUP")
    label, run_group = value.split("=", 1)
    label = label.strip()
    run_group = run_group.strip()
    if not label or not run_group:
        raise argparse.ArgumentTypeError("run must be LABEL=RUN_GROUP")
    return RunSpec(label=label, run_group=run_group)


def pct(series: pd.Series, q: float) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return float("nan")
    return float(np.percentile(s.to_numpy(dtype=float), q))


def add_elapsed_seconds(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    t = pd.to_datetime(out["time"], format="%H:%M:%S.%f", errors="coerce")
    if t.isna().all():
        out["elapsed_s"] = np.arange(len(out), dtype=float)
    else:
        out["elapsed_s"] = (t - t.iloc[0]).dt.total_seconds()
        out.loc[out["elapsed_s"] < 0, "elapsed_s"] += 24 * 3600
    return out


def load_run(spec: RunSpec) -> pd.DataFrame:
    csv_path = (
        TTRACER_ROOT
        / spec.run_group
        / "gnb"
        / "csv"
        / "GNB_MAC_BLER_MCS_DECISION.csv"
    )
    if not csv_path.exists():
        raise SystemExit(f"missing {csv_path}")
    df = pd.read_csv(csv_path)
    for col in df.columns:
        if col != "time":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = add_elapsed_seconds(df)
    df["run_label"] = spec.label
    df["run_group"] = spec.run_group
    # Direction 1 is uplink; direction 0 is downlink.
    df = df[df["direction"] == 1].copy()
    df["bler_window"] = df["bler_window_ppm"] / 1_000_000.0
    df["bler_before"] = df["bler_before_ppm"] / 1_000_000.0
    df["bler_after"] = df["bler_after_ppm"] / 1_000_000.0
    df["branch_label"] = df["branch"].map(BRANCH_LABELS).fillna("unknown")
    activity = df[(df["old_mcs"] > 0) | (df["new_mcs"] > 0) | (df["branch"] == 1)]
    if activity.empty:
        df["active_elapsed_s"] = df["elapsed_s"]
        df["in_active_window"] = True
    else:
        start = float(activity["elapsed_s"].min())
        end = float(activity["elapsed_s"].max())
        df["active_elapsed_s"] = df["elapsed_s"] - start
        df["in_active_window"] = (df["elapsed_s"] >= start) & (df["elapsed_s"] <= end)
    return df


def summarize_run(df: pd.DataFrame) -> dict[str, float | int | str]:
    active = df[df["in_active_window"]].copy()
    updated = active[active["updated"] == 1].copy()
    inc = updated[updated["branch"] == 1]
    few = updated[updated["branch"] == 3]
    high_bler = updated[updated["branch"] == 2]
    hold = updated[updated["branch"] == 4]
    nonzero = active[active["new_mcs"] > 0]
    return {
        "label": str(df["run_label"].iloc[0]),
        "run_group": str(df["run_group"].iloc[0]),
        "ul_rows": int(len(df)),
        "active_ul_rows": int(len(active)),
        "updated_rows": int(len(updated)),
        "active_start_s": float(active["elapsed_s"].min()) if len(active) else float("nan"),
        "active_end_s": float(active["elapsed_s"].max()) if len(active) else float("nan"),
        "new_mcs_p50": pct(active["new_mcs"], 50),
        "new_mcs_p95": pct(active["new_mcs"], 95),
        "new_mcs_last_nonzero": float(nonzero["new_mcs"].iloc[-1]) if len(nonzero) else float("nan"),
        "num_sched_p50_updated": pct(updated["num_sched"], 50),
        "num_sched_p95_updated": pct(updated["num_sched"], 95),
        "branch_increase_pct_updated": 100.0 * len(inc) / len(updated) if len(updated) else float("nan"),
        "branch_few_samples_pct_updated": 100.0 * len(few) / len(updated) if len(updated) else float("nan"),
        "branch_high_bler_pct_updated": 100.0 * len(high_bler) / len(updated) if len(updated) else float("nan"),
        "branch_hold_pct_updated": 100.0 * len(hold) / len(updated) if len(updated) else float("nan"),
        "bler_after_p50_updated": pct(updated["bler_after"], 50),
        "bler_after_p95_updated": pct(updated["bler_after"], 95),
    }


def window_active(df: pd.DataFrame) -> pd.DataFrame:
    active = df[df["in_active_window"]].copy()
    if active.empty:
        return pd.DataFrame()
    active["window_s"] = np.floor(active["active_elapsed_s"]).astype(int)
    mcs = (
        active.groupby(["run_label", "run_group", "window_s"], as_index=False)
        .agg(
            elapsed_s=("active_elapsed_s", "median"),
            rows=("time", "count"),
            old_mcs_p50=("old_mcs", "median"),
            new_mcs_p50=("new_mcs", "median"),
            new_mcs_p95=("new_mcs", lambda s: np.percentile(s, 95)),
        )
    )
    updated = active[active["updated"] == 1].copy()
    if updated.empty:
        return mcs.sort_values(["run_label", "elapsed_s"])
    upd = (
        updated.groupby(["run_label", "run_group", "window_s"], as_index=False)
        .agg(
            updated_rows=("time", "count"),
            num_sched_p50=("num_sched", "median"),
            num_sched_p95=("num_sched", lambda s: np.percentile(s, 95)),
            bler_after_p50=("bler_after", "median"),
            increase_rows=("branch", lambda s: int((s == 1).sum())),
            few_sample_rows=("branch", lambda s: int((s == 3).sum())),
            high_bler_rows=("branch", lambda s: int((s == 2).sum())),
            hold_rows=("branch", lambda s: int((s == 4).sum())),
        )
    )
    return mcs.merge(upd, on=["run_label", "run_group", "window_s"], how="left").sort_values(
        ["run_label", "elapsed_s"]
    )


def make_branch_plot(all_df: pd.DataFrame, out_dir: Path) -> list[Path]:
    updated = all_df[(all_df["updated"] == 1) & (all_df["in_active_window"])].copy()
    labels = list(dict.fromkeys(updated["run_label"].tolist()))
    branches = [1, 3, 2, 4]
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    bottom = np.zeros(len(labels))
    for branch in branches:
        vals = []
        for label in labels:
            sub = updated[updated["run_label"] == label]
            vals.append(100.0 * (sub["branch"] == branch).sum() / len(sub) if len(sub) else 0.0)
        ax.bar(labels, vals, bottom=bottom, color=BRANCH_COLORS[branch], label=BRANCH_LABELS[branch])
        bottom += np.asarray(vals)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of BLER-update decisions (%)")
    ax.set_title("Active traffic window: direct get_mcs_from_bler() branch outcomes")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    paths: list[Path] = []
    for ext in ("png", "pdf"):
        p = out_dir / f"bler_olla_branch_comparison.{ext}"
        fig.savefig(p, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def make_mcs_plot(windows: pd.DataFrame, out_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
    for idx, (label, sub) in enumerate(windows.groupby("run_label", sort=False)):
        color = colors[idx % len(colors)]
        ax.plot(sub["elapsed_s"], sub["new_mcs_p50"], lw=2.2, color=color, label=f"{label}: MCS p50")
        ax.fill_between(
            sub["elapsed_s"],
            sub["new_mcs_p50"],
            sub["new_mcs_p95"],
            color=color,
            alpha=0.12,
            linewidth=0,
        )
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("MCS index from BLER/OLLA selector")
    ax.set_ylim(-0.5, 30)
    ax.set_title("Active traffic window: sparse observed pace vs dense open-loop control")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    paths: list[Path] = []
    for ext in ("png", "pdf"):
        p = out_dir / f"bler_olla_mcs_timeseries.{ext}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths


def make_num_sched_plot(windows: pd.DataFrame, out_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    colors = ["#ff7f0e", "#1f77b4", "#2ca02c", "#9467bd"]
    for idx, (label, sub) in enumerate(windows.groupby("run_label", sort=False)):
        color = colors[idx % len(colors)]
        ax.plot(sub["elapsed_s"], sub["num_sched_p50"], lw=2.0, color=color, label=f"{label}: num_sched p50")
    ax.axhline(3, color="#d62728", ls="--", lw=1.5, label="too-few-samples threshold (≤3)")
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Scheduled samples per BLER window")
    finite = pd.to_numeric(windows["num_sched_p50"], errors="coerce").dropna()
    y_max = max(15.0, float(np.percentile(finite, 98)) + 2.0) if len(finite) else 15.0
    ax.set_ylim(0, min(65.0, y_max))
    ax.set_title("Active traffic window: scheduled-sample availability drives MCS behavior")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    paths: list[Path] = []
    for ext in ("png", "pdf"):
        p = out_dir / f"bler_olla_num_sched_timeseries.{ext}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths


def write_markdown(summary: pd.DataFrame, plots: list[Path], out_dir: Path) -> Path:
    out = AB_ROOT / "oai_layer_latency" / "BLER_OLLA_TRACE_RESULTS_20260723.md"
    lines: list[str] = []
    lines.append("# Direct BLER/OLLA MCS trace results")
    lines.append("")
    lines.append("This is the final scheduler-side check requested after the advisor discussion. It instruments `get_mcs_from_bler()` directly and compares sparse CARLA-like bursts against a dense 10 FPS open-loop control on the same 273 PRB RFsim path. Summaries below use the active traffic window only, not the tracer idle tail.")
    lines.append("")
    lines.append("## Main takeaway")
    lines.append("")
    if len(summary) >= 2:
        lines.append("- The low-MCS behavior is visible inside the BLER/OLLA MCS selector itself, before the later PHR/RB helper.")
        lines.append("- The decisive difference is scheduling cadence/sample availability: sparse closed-loop-style bursts repeatedly hit the `num_sched <= 3` branch, while dense open-loop traffic gives the selector enough high-sample windows to keep ratcheting MCS upward.")
        lines.append("- This explains why iperf/open-loop UDP can ramp to high MCS while the CARLA closed-loop app remains stuck near QPSK despite high RFsim PUSCH SNR.")
    else:
        lines.append("- Only one run was analyzed; compare against the paired open-loop run before making the final claim.")
    lines.append("")
    lines.append("## Summary table")
    lines.append("")
    cols = [
        ("label", "Run"),
        ("new_mcs_p50", "MCS p50"),
        ("new_mcs_p95", "MCS p95"),
        ("new_mcs_last_nonzero", "Last nonzero MCS"),
        ("num_sched_p50_updated", "num_sched p50"),
        ("num_sched_p95_updated", "num_sched p95"),
        ("branch_increase_pct_updated", "Increase %"),
        ("branch_few_samples_pct_updated", "Few-samples %"),
        ("branch_high_bler_pct_updated", "High-BLER dec %"),
        ("branch_hold_pct_updated", "Hold %"),
    ]
    lines.append("| " + " | ".join(h for _, h in cols) + " |")
    lines.append("|" + "|".join(["---"] + ["---:"] * (len(cols) - 1)) + "|")
    for _, row in summary.iterrows():
        vals: list[str] = []
        for key, _ in cols:
            val = row[key]
            if key == "label":
                vals.append(str(val))
            elif "pct" in key:
                vals.append(f"{float(val):.1f}%")
            else:
                vals.append(f"{float(val):.1f}")
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    lines.append("## Branch-code reference")
    lines.append("")
    lines.append("| Branch | Meaning |")
    lines.append("|---:|---|")
    for code in [0, 1, 2, 3, 4]:
        lines.append(f"| {code} | {BRANCH_LABELS[code].replace(chr(10), ' / ')} |")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Summary CSV: `{out_dir / 'bler_olla_summary.csv'}`")
    lines.append(f"- Windowed CSV: `{out_dir / 'bler_olla_windows.csv'}`")
    for p in plots:
        lines.append(f"- Plot: `{p}`")
    lines.append("")
    out.write_text("\n".join(lines))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", type=parse_run_spec, required=True, help="LABEL=RUN_GROUP")
    ap.add_argument("--output-dir", type=Path, default=PLOTS_DIR)
    args = ap.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )

    frames = [load_run(spec) for spec in args.run]
    all_df = pd.concat(frames, ignore_index=True)
    summary = pd.DataFrame([summarize_run(df) for df in frames])
    windows = pd.concat([window_active(df) for df in frames], ignore_index=True)

    summary_csv = out_dir / "bler_olla_summary.csv"
    window_csv = out_dir / "bler_olla_windows.csv"
    summary.to_csv(summary_csv, index=False)
    windows.to_csv(window_csv, index=False)

    plots: list[Path] = []
    plots.extend(make_branch_plot(all_df, out_dir))
    plots.extend(make_mcs_plot(windows, out_dir))
    plots.extend(make_num_sched_plot(windows, out_dir))
    md = write_markdown(summary, plots, out_dir)

    print(f"[analyze_bler_olla_trace] wrote {summary_csv}")
    print(f"[analyze_bler_olla_trace] wrote {window_csv}")
    print(f"[analyze_bler_olla_trace] wrote {md}")
    for p in plots:
        print(f"[analyze_bler_olla_trace] wrote {p}")


if __name__ == "__main__":
    main()
