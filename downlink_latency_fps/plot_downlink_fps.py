#!/usr/bin/env python3
"""Build presentation plots for the downlink/result-return FPS study."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
PLOTS = ROOT / "plots"


def load_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: float(r["fps"]))
    return rows


def newest_summary(pattern: str) -> Path:
    matches = sorted(RUNS.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"no summary CSV matched {RUNS / pattern}")
    return matches[-1]


def f(row: dict[str, str], key: str) -> float:
    value = str(row.get(key, "")).strip()
    return float(value) if value else float("nan")


def save(fig: plt.Figure, stem: str) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS / f"{stem}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_ideal_latency_breakdown(rows: list[dict[str, str]]) -> None:
    fps = [str(int(float(r["fps"]))) for r in rows]
    uplink = [f(r, "feature_upload_payload_handling_p50_ms") for r in rows]
    back = [f(r, "back_p50_ms") for r in rows]
    downlink = [f(r, "result_send_to_recv_wall_p50_ms") for r in rows]
    capture_to_result = [f(r, "capture_to_result_est_p50_ms") for r in rows]
    front = [f(r, "front_p50_ms") for r in rows]
    post_send = [f(r, "rtt_recv_p50_ms") for r in rows]

    x = range(len(rows))
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8.8, 6.2), sharex=True, height_ratios=[1.1, 0.9])
    ax.bar(x, uplink, label="feature-upload payload handling", color="#4C78A8")
    ax.bar(x, back, bottom=uplink, label="edge tail compute", color="#F58518")
    ax.bar(
        x,
        downlink,
        bottom=[u + b for u, b in zip(uplink, back)],
        label="result downlink",
        color="#54A24B",
    )
    ax.plot(x, post_send, marker="o", color="#222222", label="post-send RTT")
    ax2.plot(
        x,
        capture_to_result,
        marker="D",
        linestyle="--",
        color="#B279A2",
        label="capture→result estimate",
    )
    ax2.plot(
        x,
        front,
        marker="s",
        linestyle=":",
        color="#E45756",
        label="front processing",
    )
    ax.set_ylabel("Post-send subpath p50 latency (ms)")
    ax2.set_ylabel("Front/capture-side p50 latency (ms)")
    ax2.set_xlabel("Requested CARLA FPS")
    ax2.set_xticks(list(x), fps)
    ax.set_ylim(0, 50)
    ax2.set_ylim(0, 100)
    ax.set_title("Ideal loopback no-AE 200k: result-return latency split")
    ax.grid(axis="y", alpha=0.25)
    ax2.grid(axis="y", alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper right")
    ax2.legend(handles2, labels2, loc="upper right")
    save(fig, "ideal_loopback_latency_breakdown")


def plot_delivery_comparison(ideal_rows: list[dict[str, str]], bounded_rows: list[dict[str, str]]) -> None:
    labels = [f"ideal {int(float(r['fps']))} FPS" for r in ideal_rows]
    delivery = [f(r, "delivery") for r in ideal_rows]
    if bounded_rows:
        labels.append("bounded 10 FPS")
        delivery.append(f(bounded_rows[0], "delivery"))

    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    colors = ["#54A24B"] * len(ideal_rows) + (["#E45756"] if bounded_rows else [])
    ax.bar(range(len(labels)), delivery, color=colors)
    ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Result delivery rate")
    ax.set_title("Raised-buffer loopback vs bounded-buffer calibration")
    ax.grid(axis="y", alpha=0.25)
    for i, v in enumerate(delivery):
        ax.text(i, v + 0.025, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    save(fig, "delivery_ideal_vs_bounded")


def plot_payloads(rows: list[dict[str, str]]) -> None:
    fps = [str(int(float(r["fps"]))) for r in rows]
    feature_kb = [f(r, "feature_kb_p50") for r in rows]
    result_kb = [f(r, "result_kb_p50") for r in rows]
    x = range(len(rows))

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.plot(x, feature_kb, marker="o", label="feature payload KB", color="#4C78A8")
    ax2 = ax.twinx()
    ax2.plot(x, result_kb, marker="s", label="result payload KB", color="#54A24B")
    ax.set_xticks(list(x), fps)
    ax.set_xlabel("Requested CARLA FPS")
    ax.set_ylabel("Feature payload p50 (KB)")
    ax2.set_ylabel("Result payload p50 (KB)")
    ax.set_title("Payload sizes remain stable across FPS")
    ax.grid(axis="y", alpha=0.25)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    save(fig, "ideal_loopback_payloads")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ideal-summary",
        default="",
        help="Corrected ideal-loopback summary CSV. Defaults to newest downlink_fps_summary_*loopback*.csv.",
    )
    parser.add_argument(
        "--bounded-summary",
        default="",
        help="Optional bounded-loopback summary CSV. Omit to skip bounded comparison.",
    )
    args = parser.parse_args()

    ideal_path = Path(args.ideal_summary) if args.ideal_summary else newest_summary("downlink_fps_summary_*loopback*.csv")
    bounded_path = Path(args.bounded_summary) if args.bounded_summary else None

    ideal = load_summary(ideal_path)
    bounded = load_summary(bounded_path) if bounded_path and bounded_path.exists() else []
    plot_ideal_latency_breakdown(ideal)
    plot_delivery_comparison(ideal, bounded)
    plot_payloads(ideal)
    print(f"Wrote plots to {PLOTS} using ideal_summary={ideal_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
