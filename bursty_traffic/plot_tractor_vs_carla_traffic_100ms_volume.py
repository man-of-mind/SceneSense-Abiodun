#!/usr/bin/env python3
"""100 ms traffic-volume comparison for TRACTOR vs CARLA.

This plot intentionally uses *Mbits per 100 ms window* instead of Mbps.  That
keeps the y-axis as data amount per bin and avoids the common confusion where a
shorter averaging window inflates the equivalent rate by dividing by 0.1 s.
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


def load_tractor(path: Path, bin_s: float) -> pd.DataFrame:
    pat = re.compile(r"^([0-9.]+).*UDP, length ([0-9]+)")
    rows: list[tuple[float, int]] = []
    with path.open(errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if m:
                rows.append((float(m.group(1)), int(m.group(2))))
    if not rows:
        return pd.DataFrame({"t_s": [], "mbits": [], "count": []})
    df = pd.DataFrame(rows, columns=["ts", "bytes"])
    df["t_s"] = df["ts"] - df["ts"].min()
    df["bin"] = (df["t_s"] / bin_s).astype(int)
    out = df.groupby("bin", as_index=False).agg(bytes=("bytes", "sum"), count=("bytes", "count"))
    out["t_s"] = out["bin"].astype(float) * bin_s
    out["mbits"] = out["bytes"] * 8.0 / 1e6
    return out[["t_s", "mbits", "count"]]


def load_carla(path: Path, bin_s: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["elapsed_s"] = pd.to_numeric(df["elapsed_s"], errors="coerce")
    df["feature_payload_bytes"] = pd.to_numeric(df["feature_payload_bytes"], errors="coerce")
    df = df.dropna(subset=["elapsed_s", "feature_payload_bytes"]).copy()
    df["t_s"] = df["elapsed_s"] - df["elapsed_s"].min()
    df["bin"] = (df["t_s"] / bin_s).astype(int)
    out = df.groupby("bin", as_index=False).agg(bytes=("feature_payload_bytes", "sum"), count=("feature_payload_bytes", "count"))
    out["t_s"] = out["bin"].astype(float) * bin_s
    out["mbits"] = out["bytes"] * 8.0 / 1e6
    return out[["t_s", "mbits", "count"]]


def align_and_fill(df: pd.DataFrame, duration_s: float, bin_s: float) -> pd.DataFrame:
    out = df.copy()
    active = out[out["mbits"] > 0]
    if len(active):
        out["t_s"] = out["t_s"] - float(active["t_s"].min())
    out = out[out["t_s"].between(0, duration_s - bin_s)].copy()
    idx = (out["t_s"] / bin_s).round().astype(int)
    out = out.set_index(idx)[["mbits", "count"]].groupby(level=0).sum()
    n = int(duration_s / bin_s)
    out = out.reindex(range(n), fill_value=0.0)
    out["t_s"] = out.index.astype(float) * bin_s
    return out.reset_index(drop=True)[["t_s", "mbits", "count"]]


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

    bin_s = 0.1
    duration_s = 70.0
    loaded: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []
    for spec in RUNS:
        raw = load_tractor(spec.path, bin_s) if spec.kind == "tractor" else load_carla(spec.path, bin_s)
        data = align_and_fill(raw, duration_s, bin_s)
        loaded[spec.key] = data
        active = data[data["mbits"] > 0]
        rows.append(
            {
                "key": spec.key,
                "mean_mbits_per_100ms": float(data["mbits"].mean()),
                "p50_mbits_per_100ms": float(data["mbits"].quantile(0.50)),
                "p95_mbits_per_100ms": float(data["mbits"].quantile(0.95)),
                "max_mbits_per_100ms": float(data["mbits"].max()),
                "active_bins": int((data["mbits"] > 0).sum()),
                "active_bin_fraction": float((data["mbits"] > 0).mean()),
                "active_p50_mbits_per_100ms": float(active["mbits"].quantile(0.50)) if len(active) else float("nan"),
                "equiv_mean_mbps": float(data["mbits"].mean() / bin_s),
                "equiv_active_p50_mbps": float(active["mbits"].quantile(0.50) / bin_s) if len(active) else float("nan"),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(ANALYSIS / "tractor_vs_carla_traffic_100ms_volume_summary.csv", index=False)

    fig, axs = plt.subplots(1, len(RUNS), figsize=(15.2, 3.9), sharey=False, constrained_layout=True)
    for ax, spec in zip(axs, RUNS):
        data = loaded[spec.key]
        row = summary[summary["key"].eq(spec.key)].iloc[0]
        ax.set_title(
            f"{spec.title}\nmean={row['mean_mbits_per_100ms']:.2f} Mbits/100ms, "
            f"active bins={100*row['active_bin_fraction']:.0f}%",
            fontsize=10.7,
            pad=8,
        )
        ax.bar(
            data["t_s"],
            data["mbits"],
            width=bin_s * 0.9,
            color="#3B82C4",
            alpha=0.78,
            align="edge",
            linewidth=0,
        )
        ax.set_xlim(0, duration_s)
        ax.set_xlabel("time since first traffic sample (s)", fontweight="bold")
        ax.set_ylabel("Traffic volume\n(Mbits per 100 ms window)", fontweight="bold")
        ax.grid(True, axis="y", alpha=0.28, linewidth=0.8)
        ax.grid(True, axis="x", alpha=0.10, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.tick_params(labelsize=9)

    fig.suptitle(
        "Traffic volume over time using 100 ms windows",
        fontsize=14.2,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.02,
        "Each bar is total data in that 100 ms window. Equivalent Mbps = Mbits per bar ÷ 0.1 s, so a 1 MB CARLA frame appears as ~8.6 Mbits/bar or ~86 Mbps equivalent.",
        ha="center",
        fontsize=9.0,
        color="#444444",
    )
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"tractor_vs_carla_traffic_volume_100ms_70s.{ext}", bbox_inches="tight")


if __name__ == "__main__":
    main()
