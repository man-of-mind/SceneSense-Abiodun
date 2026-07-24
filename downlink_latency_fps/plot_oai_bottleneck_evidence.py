#!/usr/bin/env python3
"""Corrected Step-1 presentation plots for loopback/OAI transport evidence."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
PLOTS = ROOT / "plots" / "oai_bottleneck"

COLORS = {
    "ideal": "#72B7B2",
    "default": "#4C78A8",
    "ulheavy": "#F58518",
    "bw273": "#59A14F",
    "front": "#B279A2",
    "uplink": "#E45756",
    "back": "#F58518",
    "downlink": "#54A24B",
    "grey": "#666666",
}


def load_all_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(RUNS.glob("downlink_fps_summary_*.csv"), key=lambda p: p.stat().st_mtime):
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["_summary_path"] = str(path)
                row["_summary_mtime"] = str(path.stat().st_mtime)
                rows.append(row)
    return rows


def f(row: dict[str, str], key: str) -> float:
    value = str(row.get(key, "")).strip()
    try:
        return float(value) if value else float("nan")
    except ValueError:
        return float("nan")


def save(fig: plt.Figure, stem: str) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(PLOTS / f"{stem}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def latest_row(
    rows: list[dict[str, str]],
    condition_contains: str,
    fps: str = "10",
    source_contains: str | None = None,
) -> dict[str, str] | None:
    matches = [
        r
        for r in rows
        if condition_contains in str(r.get("condition", ""))
        and str(r.get("fps", "")).split(".")[0] == str(fps)
        and (source_contains is None or source_contains in str(r.get("source", "")))
    ]
    if not matches:
        return None
    return sorted(matches, key=lambda r: float(r.get("_summary_mtime", "0")))[-1]


def all_loopback_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = [r for r in rows if "ideal_loopback" in str(r.get("condition", ""))]
    out.sort(key=lambda r: f(r, "fps"))
    return out


def corrected_comparison_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    specs = [
        (
            "ideal_loopback",
            "drivable_rerun_20260722_loopback",
            "Ideal loopback\nzstd",
            COLORS["ideal"],
        ),
        (
            "oai_default_noae_zstd_drivable",
            "drivable_ab_20260721_233016",
            "Default OAI\n106PRB",
            COLORS["default"],
        ),
        (
            "oai_ulheavy_106_noae_ttracer",
            "drivable_rerun_20260722_ulheavy106",
            "UL-heavy OAI\n106PRB",
            COLORS["ulheavy"],
        ),
        (
            "oai_bw273_mu1_noae_ttracer",
            "drivable_rerun_20260722_bw273",
            "Wider BW OAI\n273PRB",
            COLORS["bw273"],
        ),
    ]
    out: list[dict[str, object]] = []
    for key, source_key, label, color in specs:
        row = latest_row(rows, key, "10", source_key)
        if row is None:
            continue
        out.append({"key": key, "label": label, "color": color, "row": row})
    return out


def annotate_bars(ax: plt.Axes, bars, fmt: str = "{:.1f}") -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_reliability_latency(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    labels = [str(r["label"]) for r in rows]
    colors = [str(r["color"]) for r in rows]
    delivery = [100.0 * f(r["row"], "delivery") for r in rows]  # type: ignore[arg-type]
    rtt_p50 = [f(r["row"], "rtt_recv_p50_ms") for r in rows]  # type: ignore[arg-type]
    rtt_p95 = [f(r["row"], "rtt_recv_p95_ms") for r in rows]  # type: ignore[arg-type]
    x = list(range(len(rows)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.6), gridspec_kw={"width_ratios": [0.95, 1.05]})
    bars = ax1.bar(x, delivery, color=colors, width=0.62)
    annotate_bars(ax1, bars, "{:.1f}%")
    ax1.set_xticks(x, labels)
    ax1.set_ylim(0, 105)
    ax1.set_ylabel("Returned-result delivery (%)")
    ax1.set_title("Reliability")
    ax1.grid(axis="y", alpha=0.22)

    ax2.plot(x, rtt_p50, marker="o", linewidth=2.2, color="#222222", label="RTT p50")
    ax2.plot(x, rtt_p95, marker="s", linewidth=2.2, color="#777777", label="RTT p95")
    ax2.set_xticks(x, labels)
    ax2.set_ylabel("Result RTT (ms)")
    ax2.set_title("Latency")
    ax2.grid(axis="y", alpha=0.22)
    ax2.legend(frameon=False, loc="upper left")

    fig.suptitle("Corrected drivable scene: loopback/default/UL-heavy/273PRB reliability and RTT", fontsize=13)
    save(fig, "corrected_transport_reliability_rtt")


def plot_latency_breakdown(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    labels = [str(r["label"]) for r in rows]
    front = [f(r["row"], "front_p50_ms") for r in rows]  # type: ignore[arg-type]
    uplink = [f(r["row"], "feature_upload_payload_handling_p50_ms") for r in rows]  # type: ignore[arg-type]
    back = [f(r["row"], "back_p50_ms") for r in rows]  # type: ignore[arg-type]
    downlink = [f(r["row"], "result_send_to_recv_wall_p50_ms") for r in rows]  # type: ignore[arg-type]
    capture_to_result = [f(r["row"], "capture_to_result_est_p50_ms") for r in rows]  # type: ignore[arg-type]
    x = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=(10.8, 5.3))
    ax.bar(x, front, color=COLORS["front"], label="front compute/codec")
    bottom = front[:]
    ax.bar(x, uplink, bottom=bottom, color=COLORS["uplink"], label="feature uplink handling")
    bottom = [a + b for a, b in zip(bottom, uplink)]
    ax.bar(x, back, bottom=bottom, color=COLORS["back"], label="edge tail compute")
    bottom = [a + b for a, b in zip(bottom, back)]
    ax.bar(x, downlink, bottom=bottom, color=COLORS["downlink"], label="result downlink")
    totals = [a + b + c + d for a, b, c, d in zip(front, uplink, back, downlink)]
    ax.plot(
        x,
        capture_to_result,
        marker="D",
        markersize=5,
        linewidth=1.6,
        color="#222222",
        label="capture→result p50",
    )
    for i, (stack_total, capture_total) in enumerate(zip(totals, capture_to_result)):
        ax.text(
            i,
            max(stack_total, capture_total) + 4,
            f"{capture_total:.0f} ms",
            ha="center",
            fontsize=9,
            color="#222222",
        )
    ax.set_xticks(x, labels)
    ax.set_ylabel("p50 latency components (ms)")
    ax.set_title("Corrected drivable scene: feature uplink dominates OAI latency")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=9)
    ax.text(
        0.99,
        0.01,
        "Bar height = sum of component p50s; black markers/labels = capture→result p50.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=COLORS["grey"],
    )
    ax.set_ylim(0, max(max(totals), max(capture_to_result)) + 35)
    save(fig, "corrected_transport_latency_breakdown")


def plot_loopback_fps(rows: list[dict[str, str]]) -> None:
    loop = all_loopback_rows(rows)
    if not loop:
        return
    x = [f(r, "fps") for r in loop]
    delivery = [100.0 * f(r, "delivery") for r in loop]
    capture = [f(r, "capture_to_result_est_p50_ms") for r in loop]
    rtt = [f(r, "rtt_recv_p50_ms") for r in loop]
    fig, ax1 = plt.subplots(figsize=(8.8, 4.5))
    ax1.plot(x, delivery, marker="o", linewidth=2.2, color=COLORS["ideal"], label="delivery")
    ax1.set_xlabel("Requested FPS")
    ax1.set_ylabel("Delivery (%)")
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", alpha=0.22)
    ax2 = ax1.twinx()
    ax2.plot(x, capture, marker="s", linewidth=2.0, color=COLORS["front"], label="capture→result p50")
    ax2.plot(x, rtt, marker="D", linewidth=2.0, color=COLORS["grey"], label="RTT p50")
    ax2.set_ylabel("Latency (ms)")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, loc="center right")
    ax1.set_title("Corrected ideal loopback FPS sweep")
    save(fig, "corrected_ideal_loopback_fps_sweep")


def main() -> int:
    rows = load_all_rows()
    comparison = corrected_comparison_rows(rows)
    plot_reliability_latency(comparison)
    plot_latency_breakdown(comparison)
    plot_loopback_fps(rows)
    print(f"Wrote corrected bottleneck plots to {PLOTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
