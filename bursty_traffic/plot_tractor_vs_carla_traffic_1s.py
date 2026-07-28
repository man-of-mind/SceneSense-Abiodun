#!/usr/bin/env python3
"""Simple 1-second traffic-rate comparison for TRACTOR vs CARLA.

This intentionally avoids 100 ms active-bin language. It answers the simpler
question: "how many Mbps were offered/delivered in each 1-second window?"
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
    path: Path


TRACTOR_EMBB = "tractor_replay_bw273_vanilla_embb0303a_off100_60s_tcpdump_20260727_210512"
TRACTOR_URLLC = "tractor_replay_bw273_vanilla_urllc0303_off240_60s_tcpdump_20260727_210842"
CARLA_273 = "downlink_oai_bw273_mu1_ttracer_fps10_layerbaseline_20260722_183259"

RUNS = [
    RunSpec(
        "tractor_onedrive",
        "TRACTOR OneDrive\nreal bursty replay",
        "tractor",
        AB / "metrics_logs" / "tractor_replay" / TRACTOR_EMBB / "udp_sink_packets.txt",
    ),
    RunSpec(
        "tractor_meet",
        "TRACTOR Google Meet\nreal bursty replay",
        "tractor",
        AB / "metrics_logs" / "tractor_replay" / TRACTOR_URLLC / "udp_sink_packets.txt",
    ),
    RunSpec(
        "carla_split",
        "CARLA split inference\n≈1 MB feature frames",
        "carla",
        AB
        / "downlink_latency_fps/runs/oai_bw273_mu1_ttracer/fps_10_layerbaseline_20260722_183259/streams"
        / f"{CARLA_273}_metrics.csv",
    ),
]


def load_tractor_1s(path: Path) -> pd.DataFrame:
    pat = re.compile(r"^([0-9.]+).*UDP, length ([0-9]+)")
    rows: list[tuple[float, int]] = []
    with path.open(errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if m:
                rows.append((float(m.group(1)), int(m.group(2))))
    if not rows:
        return pd.DataFrame({"t_s": [], "mbps": [], "frames_or_packets": []})
    df = pd.DataFrame(rows, columns=["ts", "bytes"])
    df["t_s"] = df["ts"] - df["ts"].min()
    df["sec"] = df["t_s"].astype(int)
    out = df.groupby("sec", as_index=False).agg(bytes=("bytes", "sum"), frames_or_packets=("bytes", "count"))
    out["t_s"] = out["sec"].astype(float)
    out["mbps"] = out["bytes"] * 8.0 / 1e6
    return out[["t_s", "mbps", "frames_or_packets"]]


def load_carla_1s(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["elapsed_s"] = pd.to_numeric(df["elapsed_s"], errors="coerce")
    df["feature_payload_bytes"] = pd.to_numeric(df["feature_payload_bytes"], errors="coerce")
    df = df.dropna(subset=["elapsed_s", "feature_payload_bytes"]).copy()
    df["t_s"] = df["elapsed_s"] - df["elapsed_s"].min()
    df["sec"] = df["t_s"].astype(int)
    out = df.groupby("sec", as_index=False).agg(
        bytes=("feature_payload_bytes", "sum"),
        frames_or_packets=("feature_payload_bytes", "count"),
    )
    out["t_s"] = out["sec"].astype(float)
    out["mbps"] = out["bytes"] * 8.0 / 1e6
    return out[["t_s", "mbps", "frames_or_packets"]]


def align_and_fill(df: pd.DataFrame, duration_s: int = 70) -> pd.DataFrame:
    """Zero time at first nonzero second and fill missing seconds with zeros."""
    out = df.copy()
    active = out[out["mbps"] > 0]
    if len(active):
        out["t_s"] = out["t_s"] - float(active["t_s"].min())
    out = out[out["t_s"].between(0, duration_s - 1)].copy()
    idx = out["t_s"].round().astype(int)
    out = out.set_index(idx)[["mbps", "frames_or_packets"]].groupby(level=0).sum()
    out = out.reindex(range(duration_s), fill_value=0.0)
    out["t_s"] = out.index.astype(float)
    return out.reset_index(drop=True)[["t_s", "mbps", "frames_or_packets"]]


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

    loaded: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []
    for spec in RUNS:
        raw = load_tractor_1s(spec.path) if spec.kind == "tractor" else load_carla_1s(spec.path)
        data = align_and_fill(raw, duration_s=70)
        loaded[spec.key] = data
        active = data[data["mbps"] > 0]
        rows.append(
            {
                "key": spec.key,
                "mean_mbps_70s": float(data["mbps"].mean()),
                "p50_mbps_1s": float(data["mbps"].quantile(0.50)),
                "p95_mbps_1s": float(data["mbps"].quantile(0.95)),
                "max_mbps_1s": float(data["mbps"].max()),
                "active_seconds": int((data["mbps"] > 0).sum()),
                "active_second_fraction": float((data["mbps"] > 0).mean()),
                "p50_frames_or_packets_per_s": float(data["frames_or_packets"].quantile(0.50)),
                "p95_frames_or_packets_per_s": float(data["frames_or_packets"].quantile(0.95)),
                "active_p50_mbps_1s": float(active["mbps"].quantile(0.50)) if len(active) else float("nan"),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(ANALYSIS / "tractor_vs_carla_traffic_1s_summary.csv", index=False)

    fig, axs = plt.subplots(1, len(RUNS), figsize=(15.2, 3.9), sharey=False, constrained_layout=True)
    for ax, spec in zip(axs, RUNS):
        data = loaded[spec.key]
        row = summary[summary["key"].eq(spec.key)].iloc[0]
        ax.set_title(
            f"{spec.title}\nmean={row['mean_mbps_70s']:.1f} Mbps, 1s p50/p95={row['p50_mbps_1s']:.1f}/{row['p95_mbps_1s']:.1f}",
            fontsize=10.7,
            pad=8,
        )
        ax.fill_between(data["t_s"], data["mbps"], step="mid", color="#3B82C4", alpha=0.20, linewidth=0)
        ax.plot(data["t_s"], data["mbps"], color="#2F6FAE", linewidth=2.1)
        ax.scatter(data["t_s"], data["mbps"], color="#2F6FAE", s=11, alpha=0.55)
        ax.set_xlim(0, 69)
        ax.set_xlabel("time since first traffic second (s)", fontweight="bold")
        ax.set_ylabel("Traffic rate (Mbps)\nper 1-second window", fontweight="bold")
        ax.grid(True, axis="y", alpha=0.28, linewidth=0.8)
        ax.grid(True, axis="x", alpha=0.10, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.tick_params(labelsize=9)

    fig.suptitle(
        "Traffic rate over time using simple 1-second windows",
        fontsize=14.2,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.02,
        "Each point is total bytes observed in that 1-second window converted to Mbps; zeros are included.",
        ha="center",
        fontsize=9.2,
        color="#444444",
    )
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"tractor_vs_carla_traffic_rate_1s_70s.{ext}", bbox_inches="tight")


if __name__ == "__main__":
    main()
