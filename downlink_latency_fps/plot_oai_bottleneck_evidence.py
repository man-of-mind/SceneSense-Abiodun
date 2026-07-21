#!/usr/bin/env python3
"""Presentation plots for OAI transport-bottleneck discussion.

These plots intentionally separate:
  - closed-loop live CARLA frontend evidence, which is the deployable Step-1 result;
  - replay/open-loop diagnostics, which are useful but not the headline result.
"""

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
    "default": "#4C78A8",
    "ulheavy": "#F58518",
    "bw273": "#59A14F",
    "ideal": "#72B7B2",
    "front": "#B279A2",
    "uplink": "#E45756",
    "back": "#F58518",
    "downlink": "#54A24B",
    "grey": "#666666",
}


def load_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_replay_md(path: Path) -> dict[str, float]:
    """Parse the simple markdown metric table from replay outputs."""
    out: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 2:
            continue
        key, value = cells
        if key in {"Metric", "---"}:
            continue
        try:
            out[key] = float(value.replace("%", "").replace("bytes", "").strip())
        except ValueError:
            continue
    return out


def f(row: dict[str, str], key: str) -> float:
    value = str(row.get(key, "")).strip()
    return float(value) if value else float("nan")


def save(fig: plt.Figure, stem: str) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(PLOTS / f"{stem}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


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


def live_rows() -> list[dict[str, object]]:
    default = [
        r
        for r in load_summary(RUNS / "downlink_fps_summary_oai_default_20260720_one_loop.csv")
        if str(r["fps"]) == "10"
    ][0]
    ulheavy = load_summary(
        RUNS / "downlink_fps_summary_oai_ulheavy106_carla_10fps_full_20260720.csv"
    )[0]
    bw273 = load_summary(
        RUNS / "downlink_fps_summary_oai_bw273_mu1_carla_10fps_full_20260720.csv"
    )[0]
    # Include the manual/validated 273PRB CARLA run.  Do not confuse this with
    # the earlier automated prb_273 sweep attempt, which failed because the UE
    # was launched with mismatched center-frequency/SSB parameters.  The
    # bw273_mu1_carla_live RAN logs prove -r 273, N_RB_DL 273, --ssb 516, and
    # UE tunnel IP 10.0.0.5, matching the CARLA frontend bind host.
    return [
        {
            "label": "Default\n106 PRB",
            "short": "Default",
            "color": COLORS["default"],
            "row": default,
        },
        {
            "label": "UL-heavy\n106 PRB",
            "short": "UL-heavy",
            "color": COLORS["ulheavy"],
            "row": ulheavy,
        },
        {
            "label": "Wider BW\n273 PRB",
            "short": "273 PRB",
            "color": COLORS["bw273"],
            "row": bw273,
        },
    ]


def plot_live_config_summary(rows: list[dict[str, object]]) -> None:
    labels = [str(r["label"]) for r in rows]
    colors = [str(r["color"]) for r in rows]
    delivery = [100.0 * f(r["row"], "delivery") for r in rows]  # type: ignore[arg-type]
    rtt_p50 = [f(r["row"], "rtt_recv_p50_ms") for r in rows]  # type: ignore[arg-type]
    rtt_p95 = [f(r["row"], "rtt_recv_p95_ms") for r in rows]  # type: ignore[arg-type]
    x = list(range(len(rows)))

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(9.8, 4.5),
        gridspec_kw={"width_ratios": [0.95, 1.05]},
    )
    bars = ax1.bar(x, delivery, color=colors, width=0.62)
    annotate_bars(ax1, bars, "{:.1f}%")
    ax1.set_xticks(x, labels)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Returned-result delivery (%)")
    ax1.set_title("Delivery barely moves")
    ax1.grid(axis="y", alpha=0.22)

    ax2.plot(x, rtt_p50, marker="o", linewidth=2.2, color="#222222", label="RTT p50")
    ax2.plot(x, rtt_p95, marker="s", linewidth=2.2, color="#777777", label="RTT p95")
    ax2.set_xticks(x, labels)
    ax2.set_ylabel("Result RTT (ms)")
    ax2.set_title("Latency shifts by config")
    ax2.set_ylim(0, max(rtt_p95) * 1.25)
    ax2.grid(axis="y", alpha=0.22)
    ax2.legend(frameon=False, loc="upper left")

    fig.suptitle("Live CARLA frontend: validated OAI config sensitivity at 10 FPS", fontsize=13)
    fig.text(
        0.5,
        -0.02,
        "Closed-loop frontend waits for result/timeout; 273PRB row is the manual validated bw273_mu1 run, not the failed automated prb_273 sweep.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    save(fig, "oai_live_config_delivery_rtt")


def plot_live_latency_breakdown(rows: list[dict[str, object]]) -> None:
    labels = [str(r["label"]) for r in rows]
    front = [f(r["row"], "front_p50_ms") for r in rows]  # type: ignore[arg-type]
    uplink = [f(r["row"], "feature_upload_payload_handling_p50_ms") for r in rows]  # type: ignore[arg-type]
    back = [f(r["row"], "back_p50_ms") for r in rows]  # type: ignore[arg-type]
    downlink = [f(r["row"], "result_send_to_recv_wall_p50_ms") for r in rows]  # type: ignore[arg-type]
    x = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.bar(x, front, color=COLORS["front"], label="front compute/codec")
    bottom = front[:]
    ax.bar(x, uplink, bottom=bottom, color=COLORS["uplink"], label="feature uplink handling")
    bottom = [a + b for a, b in zip(bottom, uplink)]
    ax.bar(x, back, bottom=bottom, color=COLORS["back"], label="edge tail compute")
    bottom = [a + b for a, b in zip(bottom, back)]
    ax.bar(x, downlink, bottom=bottom, color=COLORS["downlink"], label="result downlink")

    total = [a + b + c + d for a, b, c, d in zip(front, uplink, back, downlink)]
    for i, t in enumerate(total):
        ax.text(i, t + 5, f"{t:.0f} ms", ha="center", fontsize=9)

    ax.set_xticks(x, labels)
    ax.set_ylabel("p50 latency components (ms)")
    ax.set_title("Live CARLA latency budget: feature uplink dominates")
    ax.set_ylim(0, max(total) * 1.2)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    save(fig, "oai_live_latency_breakdown")


def plot_loopback_vs_oai_10fps(rows: list[dict[str, object]]) -> None:
    ideal = [
        r
        for r in load_summary(RUNS / "downlink_fps_summary_20260717_ideal_one_loop.csv")
        if str(r["fps"]) == "10"
    ][0]
    default = rows[0]["row"]  # type: ignore[index]
    labels = ["Ideal loopback\n8MB buffer", "Default OAI\n106PRB 7DL/2UL"]
    colors = [COLORS["ideal"], COLORS["default"]]
    delivery = [100.0 * f(ideal, "delivery"), 100.0 * f(default, "delivery")]  # type: ignore[arg-type]
    uplink = [
        f(ideal, "feature_upload_payload_handling_p50_ms"),
        f(default, "feature_upload_payload_handling_p50_ms"),  # type: ignore[arg-type]
    ]
    downlink = [
        f(ideal, "result_send_to_recv_wall_p50_ms"),
        f(default, "result_send_to_recv_wall_p50_ms"),  # type: ignore[arg-type]
    ]
    rtt = [
        f(ideal, "rtt_recv_p50_ms"),
        f(default, "rtt_recv_p50_ms"),  # type: ignore[arg-type]
    ]
    x = [0, 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.2))
    bars = ax1.bar(x, delivery, color=colors, width=0.58)
    annotate_bars(ax1, bars, "{:.1f}%")
    ax1.set_xticks(x, labels)
    ax1.set_ylim(0, 110)
    ax1.set_ylabel("Delivery (%)")
    ax1.set_title("Delivery drop appears on OAI")
    ax1.grid(axis="y", alpha=0.22)

    w = 0.25
    ax2.bar([i - w for i in x], uplink, width=w, color=COLORS["uplink"], label="feature uplink handling")
    ax2.bar(x, rtt, width=w, color=COLORS["grey"], label="post-send RTT")
    ax2.bar([i + w for i in x], downlink, width=w, color=COLORS["downlink"], label="result downlink")
    ax2.set_xticks(x, labels)
    ax2.set_ylabel("p50 latency (ms)")
    ax2.set_title("OAI inflation is uplink-side")
    ax2.grid(axis="y", alpha=0.22)
    ax2.legend(frameon=False, fontsize=9)

    fig.suptitle("10 FPS no-AE: ideal loopback vs OAI closed-loop", fontsize=13)
    save(fig, "loopback_vs_oai_10fps")


def plot_oai_default_fps_sweep() -> None:
    rows = load_summary(RUNS / "downlink_fps_summary_oai_default_20260720_one_loop.csv")
    x = [f(r, "fps") for r in rows]
    delivery = [100.0 * f(r, "delivery") for r in rows]
    rtt50 = [f(r, "rtt_recv_p50_ms") for r in rows]
    rtt95 = [f(r, "rtt_recv_p95_ms") for r in rows]
    moving = [100.0 * f(r, "moving_gt0p5_frac") for r in rows]

    fig, ax1 = plt.subplots(figsize=(8.6, 4.4))
    ax1.plot(x, delivery, marker="o", linewidth=2.2, color=COLORS["default"], label="delivery")
    ax1.plot(
        x,
        moving,
        marker="^",
        linewidth=1.8,
        linestyle="--",
        color="#999999",
        label="moving-frame fraction",
    )
    ax1.set_xlabel("Requested FPS")
    ax1.set_ylabel("Percent (%)")
    ax1.set_ylim(0, 100)
    ax1.grid(axis="y", alpha=0.22)

    ax2 = ax1.twinx()
    ax2.plot(x, rtt50, marker="s", linewidth=2.1, color="#222222", label="RTT p50")
    ax2.plot(x, rtt95, marker="D", linewidth=2.1, color="#777777", label="RTT p95")
    ax2.set_ylabel("Result RTT (ms)")
    ax2.set_ylim(0, max(rtt95) * 1.25)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, ncol=2, loc="lower center")
    ax1.set_title("Default OAI FPS sweep: delivery stays ~72%, RTT stays ~200–280 ms")
    fig.text(
        0.5,
        -0.02,
        "Closed-loop harness: missed results slow wall-clock progress, so this is not a true open-loop offered-load sweep.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    save(fig, "oai_default_fps_sweep")


def plot_replay_vs_carla(rows: list[dict[str, object]]) -> None:
    replay_paths = [
        ROOT.parent / "metrics_logs/oai_payload_replay/oai_payload_replay_10fps_20260720_quick/REPLAY_RESULTS.md",
        ROOT.parent / "metrics_logs/oai_payload_replay/oai_cfg_bw273_mu1_replay_10fps_20260720/REPLAY_RESULTS.md",
        ROOT.parent / "metrics_logs/oai_payload_replay/oai_cfg_ulheavy106_replay_10fps_20260720/REPLAY_RESULTS.md",
    ]
    replay = [load_replay_md(p) for p in replay_paths]
    labels = ["Default", "273 PRB", "UL-heavy"]
    carla_delivery = [100.0 * f(r["row"], "delivery") for r in rows]  # type: ignore[arg-type]
    replay_delivery = [r["delivery"] for r in replay]
    x = list(range(len(labels)))
    w = 0.34

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    b1 = ax.bar([i - w / 2 for i in x], replay_delivery, width=w, color="#9ecae9", label="Replay/open-loop")
    b2 = ax.bar([i + w / 2 for i in x], carla_delivery, width=w, color="#3182bd", label="CARLA/closed-loop")
    annotate_bars(ax, b1, "{:.1f}%")
    annotate_bars(ax, b2, "{:.1f}%")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Delivery (%)")
    ax.set_title("Replay diagnostic vs live CARLA frontend")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, loc="upper left")
    fig.text(
        0.5,
        -0.02,
        "Replay forces ~92 Mbps offered load; normal CARLA frontend waits for result/timeout before advancing.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    save(fig, "replay_vs_carla_delivery")


def main() -> int:
    rows = live_rows()
    plot_live_config_summary(rows)
    plot_live_latency_breakdown(rows)
    plot_loopback_vs_oai_10fps(rows)
    plot_oai_default_fps_sweep()
    plot_replay_vs_carla(rows)
    print(f"Wrote OAI bottleneck plots to {PLOTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
