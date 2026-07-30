#!/usr/bin/env python3
"""Track-1 closeout plots: payload reduction, reliability, and FPS capacity.

This script summarizes the reportable Track-1 OAI uplink-only runs from their
raw front/edge/T-tracer artifacts.  It intentionally excludes the failed
external-AE attempt; the reportable AE result is the integrated-AE checkpoint
run, matching the per-model knob matrix convention.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-track1-closeout")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


AB = Path("abiodun")
OUT = AB / "uplink_only_spatial_map_pipeline" / "plots" / "track1_oai_reduced_payload"
MD_OUT = AB / "uplink_only_spatial_map_pipeline" / "TRACK1_CLOSEOUT_RESULTS.md"


RUNS: Dict[str, Tuple[str, Path, str, int]] = {
    "1 MB no-AE\nu8 ROI0\n10 FPS target": (
        "track1_track1_oai_default106_ttracer_fps10_track1_default106_20260729_204536",
        AB
        / "uplink_only_spatial_map_pipeline/runs/track1_oai_default106_ttracer/"
        / "fps_10_track1_default106_20260729_204536",
        "baseline_u8_roi0_1MB_10fps",
        10,
    ),
    "394 KB no-AE\nu4 ROI0\n10 FPS target": (
        "track1_track1_oai_default106_ttracer_noae_u4_roi0_fps10_20260730_track1_noae_u4_roi0",
        AB
        / "uplink_only_spatial_map_pipeline/runs/track1_oai_default106_ttracer_noae_u4_roi0/"
        / "fps_10_20260730_track1_noae_u4_roi0",
        "noae_u4_roi0_394KB_10fps",
        10,
    ),
    "157 KB AE-128\nu6 ROI0.5\n10 FPS target": (
        "track1_track1_oai_default106_ttracer_ae128_u6_roi05_integrated_fps10_20260730_track1_ae128_u6_roi05_integrated",
        AB
        / "uplink_only_spatial_map_pipeline/runs/track1_oai_default106_ttracer_ae128_u6_roi05_integrated/"
        / "fps_10_20260730_track1_ae128_u6_roi05_integrated",
        "ae128_u6_roi05_157KB_10fps",
        10,
    ),
    "157 KB AE-128\nu6 ROI0.5\n20 FPS probe": (
        "track1_track1_oai_default106_ttracer_ae128_u6_roi05_integrated_fps20probe_fps20_20260730_track1_ae128_u6_roi05_fps20probe",
        AB
        / "uplink_only_spatial_map_pipeline/runs/track1_oai_default106_ttracer_ae128_u6_roi05_integrated_fps20probe/"
        / "fps_20_20260730_track1_ae128_u6_roi05_fps20probe",
        "ae128_u6_roi05_157KB_20fps_probe",
        20,
    ),
}


def q(series: pd.Series, pct: float) -> float:
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return float("nan")
    return float(vals.quantile(pct))


def front_events(run_dir: Path) -> pd.DataFrame:
    candidates = sorted((run_dir / "front_metrics" / "streams").glob("*send_events.csv"))
    if not candidates:
        raise FileNotFoundError(f"no send_events CSV under {run_dir}")
    return pd.read_csv(candidates[-1])


def summarize_one(plot_label: str, run_group: str, run_dir: Path, run_id: str, target_fps: int) -> dict:
    edge = pd.read_csv(run_dir / "edge_uplink_metrics.csv")
    front = front_events(run_dir)
    keep = ["frame_id", "send_call_ms", "feature_payload_bytes", "feature_payload_chunks", "camera_sent_perf"]
    merged = edge.merge(front[keep], on="frame_id", how="left")
    merged = merged[merged["send_call_ms"].notna()].copy()
    merged["sensor_prep_ms"] = merged["capture_to_backbone_input_ms"] - merged["model_preprocess_ms"]
    merged["front_model_ms"] = merged["model_preprocess_ms"] + merged["front_backbone_ms"]
    merged["uplink_transport_only_ms"] = (merged["front_to_edge_ms"] - merged["send_call_ms"]).clip(lower=0)

    send_span_s = float(front["camera_sent_perf"].max() - front["camera_sent_perf"].min())
    actual_fps = (len(front) - 1) / send_span_s if send_span_s > 0 else float("nan")

    row = {
        "plot_label": plot_label,
        "short_label": run_id.replace("_10fps", "").replace("_20fps_probe", " 20FPS").replace("baseline_u8_roi0_1MB", "1MB u8").replace("noae_u4_roi0_394KB", "394KB u4").replace("ae128_u6_roi05_157KB", "157KB AE"),
        "run_id": run_id,
        "run_group": run_group,
        "target_fps": target_fps,
        "sent": len(front),
        "processed": len(edge),
        "delivery_pct": 100.0 * len(edge) / len(front),
        "actual_fps": actual_fps,
        "payload_p50_kib": q(front["feature_payload_bytes"], 0.50) / 1024.0,
        "payload_p95_kib": q(front["feature_payload_bytes"], 0.95) / 1024.0,
        "chunks_p50": q(front["feature_payload_chunks"], 0.50),
        "udp_partial_drops": float(
            pd.to_numeric(edge.get("udp_partial_messages_dropped", pd.Series([0])), errors="coerce")
            .dropna()
            .max()
        ),
        "edge_queue_drops": float(
            pd.to_numeric(edge.get("edge_receive_queue_dropped", pd.Series([0])), errors="coerce")
            .dropna()
            .max()
        ),
        "sensor_p50_ms": q(merged["sensor_prep_ms"], 0.50),
        "sensor_p95_ms": q(merged["sensor_prep_ms"], 0.95),
        "front_model_p50_ms": q(merged["front_model_ms"], 0.50),
        "front_model_p95_ms": q(merged["front_model_ms"], 0.95),
        "send_call_p50_ms": q(merged["send_call_ms"], 0.50),
        "send_call_p95_ms": q(merged["send_call_ms"], 0.95),
        "uplink_transport_p50_ms": q(merged["uplink_transport_only_ms"], 0.50),
        "uplink_transport_p95_ms": q(merged["uplink_transport_only_ms"], 0.95),
        "tail_p50_ms": q(merged["tail_ms"], 0.50),
        "tail_p95_ms": q(merged["tail_ms"], 0.95),
        "capture_tail_p50_ms": q(merged["capture_to_tail_done_ms"], 0.50),
        "capture_tail_p95_ms": q(merged["capture_to_tail_done_ms"], 0.95),
        "backbone_tail_p50_ms": q(merged["backbone_input_to_tail_done_ms"], 0.50),
        "backbone_tail_p95_ms": q(merged["backbone_input_to_tail_done_ms"], 0.95),
    }

    tt = AB / "metrics_logs" / "scenesense_ttracer" / run_group
    grant = tt / "ue" / "analysis" / "nrue_grant_summary.csv"
    if grant.exists():
        ul = pd.read_csv(grant).query("direction_label == 'ul'").iloc[0]
        row.update(
            {
                "scheduled_mbps": float(ul.scheduled_mbps),
                "avg_mcs": float(ul.avg_mcs),
                "p50_mcs": float(ul.p50_mcs),
                "p95_mcs": float(ul.p95_mcs),
                "p50_prb": float(ul.p50_rb_size),
                "retx_rate": float(ul.retx_rate),
            }
        )
    queue = tt / "ue" / "analysis" / "nrue_queue_summary.csv"
    if queue.exists():
        qu = pd.read_csv(queue).iloc[0]
        row.update(
            {
                "bsr_p50_kib": float(qu.bsr_total_lcg_p50_bytes) / 1024.0,
                "bsr_p95_kib": float(qu.bsr_total_lcg_p95_bytes) / 1024.0,
                "sdu_mbps": float(qu.sdu_mbps),
            }
        )
    layer = tt / "layer_latency" / "uplink_layer_latency.md"
    if layer.exists():
        m = re.search(r"RLC mean queueing delay.*?:\*\* ([0-9.]+) ms", layer.read_text())
        if m:
            row["rlc_mean_queue_ms"] = float(m.group(1))
    return row


def style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "font.size": 10.5,
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "legend.fontsize": 9.5,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_payload_latency(df: pd.DataFrame) -> None:
    x = np.arange(len(df))
    labels = df["short_label"].tolist()
    colors = ["#4C78A8", "#54A24B", "#2E86AB", "#6C5CE7"]
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.2))
    fig.subplots_adjust(hspace=0.48, wspace=0.25)
    axes = axes.ravel()

    ax = axes[0]
    ax.bar(x, df["payload_p50_kib"], color=colors, edgecolor="white")
    for xi, payload, chunks in zip(x, df["payload_p50_kib"], df["chunks_p50"]):
        ax.text(xi, payload + 42, f"{payload:.0f} KiB\n{chunks:.0f} chunks", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.set_ylim(0, max(df["payload_p50_kib"]) * 1.22)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Payload p50 (KiB)")
    ax.set_title("A. Feature payload shrinks")

    ax = axes[1]
    ax.bar(x, df["delivery_pct"], color=colors, edgecolor="white")
    for xi, delivery, partial, edge_drop in zip(x, df["delivery_pct"], df["udp_partial_drops"], df["edge_queue_drops"]):
        ax.text(
            xi,
            delivery + 0.35,
            f"{delivery:.1f}%\npartial {partial:.0f}\nedge q {edge_drop:.0f}",
            ha="center",
            va="bottom",
            color="#111827",
            fontweight="bold",
            fontsize=9.2,
            bbox=dict(facecolor="white", edgecolor="#CBD5E1", alpha=0.92, boxstyle="round,pad=0.25"),
        )
    ax.set_ylim(90, 103.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Processed / sent (%)")
    ax.set_title("B. Reliability/reassembly")

    ax = axes[2]
    comps = [
        ("sensor_p50_ms", "Sensor prep", "#4C78A8"),
        ("front_model_p50_ms", "Front model", "#F58518"),
        ("send_call_p50_ms", "UDP send", "#B279A2"),
        ("uplink_transport_p50_ms", "OAI uplink", "#E45756"),
        ("tail_p50_ms", "Edge tail", "#72B7B2"),
    ]
    bottom = np.zeros(len(df))
    for col, name, color in comps:
        vals = df[col].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, label=name, color=color, edgecolor="white", linewidth=0.5)
        bottom += vals
    ax.scatter(x, df["capture_tail_p50_ms"], color="black", marker="D", s=38, zorder=4, label="measured capture→tail")
    for xi, val in zip(x, df["capture_tail_p50_ms"]):
        ax.text(xi, val + 5, f"{val:.0f} ms", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("p50 latency (ms)")
    ax.set_title("C. Capture→edge-tail latency")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False, ncol=3)

    ax = axes[3]
    ax.bar(x, df["actual_fps"], color=colors, edgecolor="white")
    ax.scatter(x, df["target_fps"], color="#111827", marker="_", s=260, linewidths=2.5, label="target FPS")
    for xi, actual, target in zip(x, df["actual_fps"], df["target_fps"]):
        ax.text(xi, actual + 0.25, f"{actual:.1f}", ha="center", va="bottom", fontweight="bold")
        ax.text(xi, target + 0.35, f"target {target}", ha="center", va="bottom", fontsize=8.5, color="#111827")
    ax.set_ylim(0, max(df["target_fps"].max() + 3, 12))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Actual sent FPS")
    ax.set_title("D. Producer/FPS ceiling")
    ax.legend(frameon=False, loc="upper left")

    save(fig, "track1_oai_payload_latency_reliability")


def plot_radio(df: pd.DataFrame) -> None:
    x = np.arange(len(df))
    labels = df["short_label"].tolist()
    colors = ["#4C78A8", "#54A24B", "#2E86AB", "#6C5CE7"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))
    fig.subplots_adjust(wspace=0.34)

    ax = axes[0]
    ax.bar(x, df["bsr_p95_kib"], color=colors, edgecolor="white")
    for xi, bsr in zip(x, df["bsr_p95_kib"]):
        ax.text(xi, bsr + 28, f"{bsr:.0f} KiB", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=16, ha="right")
    ax.set_ylabel("UE BSR LCG1 p95 (KiB)")
    ax.set_title("A. Backlog scales with payload")

    ax = axes[1]
    ax.bar(x, df["scheduled_mbps"], color=colors, edgecolor="white", label="MAC scheduled")
    ax.plot(x, df["sdu_mbps"], color="#111827", marker="o", linewidth=2.2, label="RLC SDU drain")
    for xi, sched in zip(x, df["scheduled_mbps"]):
        ax.text(xi, sched + 1.2, f"{sched:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=16, ha="right")
    ax.set_ylabel("Mbps")
    ax.set_title("B. Radio/RLC load falls")
    ax.legend(frameon=False)

    ax = axes[2]
    ax.bar(x, df["p50_mcs"], color=colors, edgecolor="white", label="p50 MCS")
    ax.plot(x, df["p95_mcs"], color="#111827", marker="o", linewidth=2.2, label="p95 MCS")
    for xi, mcs in zip(x, df["p50_mcs"]):
        ax.text(xi, mcs + 0.45, f"{mcs:.0f}", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=16, ha="right")
    ax.set_ylabel("MCS index")
    ax.set_title("C. Lower load gets lower MCS but still drains")
    ax.legend(frameon=False)

    save(fig, "track1_oai_radio_backlog_reduced_payload")


def write_markdown(df: pd.DataFrame) -> None:
    def row_for(run_id: str) -> pd.Series:
        return df[df["run_id"].eq(run_id)].iloc[0]

    base = row_for("baseline_u8_roi0_1MB_10fps")
    u4 = row_for("noae_u4_roi0_394KB_10fps")
    ae10 = row_for("ae128_u6_roi05_157KB_10fps")
    ae20 = row_for("ae128_u6_roi05_157KB_20fps_probe")

    table_rows = []
    for _, r in df.iterrows():
        table_rows.append(
            "| {label} | {target:.0f} | {fps:.2f} | {payload:.1f} | {chunks:.0f} | {deliv:.1f}% | {partial:.0f} | {edge:.0f} | {uplink:.1f} | {cap:.1f} | {backbone:.1f} | {mcs:.0f} | {bsr:.1f} |".format(
                label=r["run_id"].replace("_", " "),
                target=r["target_fps"],
                fps=r["actual_fps"],
                payload=r["payload_p50_kib"],
                chunks=r["chunks_p50"],
                deliv=r["delivery_pct"],
                partial=r["udp_partial_drops"],
                edge=r["edge_queue_drops"],
                uplink=r["uplink_transport_p50_ms"],
                cap=r["capture_tail_p50_ms"],
                backbone=r["backbone_tail_p50_ms"],
                mcs=r["p50_mcs"],
                bsr=r["bsr_p95_kib"],
            )
        )

    md = f"""# Track 1 closeout: uplink-only spatial-map pipeline

Date: 2026-07-30

Track 1 tests the project-relevant path: CARLA/front split features go uplink to the edge tail and are published toward the spatial-map side. The car does **not** wait for returned detections.

## Reportable runs

| Run | Target FPS | Actual FPS | Payload p50 KiB | Chunks p50 | Delivery | UDP partial drops | Edge queue drops | Uplink p50 ms | Capture→tail p50 ms | Backbone→tail p50 ms | MCS p50 | BSR p95 KiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_rows)}

## Main conclusions

1. **Payload reduction fixes the UDP/reassembly problem.** The 1 MB baseline had {base.udp_partial_drops:.0f} UDP partial-message drops and {base.delivery_pct:.1f}% delivery. The 394 KiB no-AE/uint4 point reached {u4.delivery_pct:.1f}% delivery with zero UDP partial drops. The 157 KiB AE-128 point also had zero UDP partial drops.
2. **Latency improves, but not proportionally to payload.** Capture→tail p50 moved from {base.capture_tail_p50_ms:.1f} ms to {u4.capture_tail_p50_ms:.1f} ms to {ae10.capture_tail_p50_ms:.1f} ms. The OAI uplink term improves, but sensor/front production still dominates the map staleness budget.
3. **Actual FPS is still front/CARLA limited.** At 10 FPS target, actual send rate stayed around {base.actual_fps:.1f}--{u4.actual_fps:.1f} FPS. The AE-128 20 FPS probe reached {ae20.actual_fps:.1f} FPS, not 20 FPS, while keeping UDP partial drops at zero. This matches the ideal-loopback capacity observation that increasing target FPS raises actual FPS sublinearly.
4. **Residual AE delivery misses are edge-queue, not UDP/OAI partial reassembly.** AE-128 had {ae10.edge_queue_drops:.0f} edge queue drops at 10 FPS and {ae20.edge_queue_drops:.0f} at the 20 FPS probe. These are startup/drain-side application queue drops; they are separate from the old 1 MB UDP partial-message loss.
5. **Accuracy should be reported from the per-model knob matrix, not inferred from OAI transport.** Network transport changes frame availability/staleness; it does not change the decoded-frame model accuracy. The matching offline matrix entries are in `rl_agent/PERMODEL_KNOB_MATRIX_GROUPED.md` / `PERMODEL_KNOB_MATRIX_ZSTD.md`.

## Implementation note

During the AE run, the first attempt failed because the copied Track-1 live script still treated AE as an external standalone split-AE checkpoint. The current reportable AE run uses the **integrated AE checkpoint as the main fusion checkpoint**, matching the per-model matrix. The Track-1 runner now allows checkpoint override, and the copied live script attaches the integrated `feature_ae` before loading checkpoint weights.

## What this closes

Track 1 can be wrapped with this framing:

- The project-relevant uplink-only pipeline removes the closed-loop result-wait idle pattern.
- With 1 MB no-AE features, OAI default 106PRB still shows partial-message loss and ~155 ms capture→tail p50.
- Reducing feature payload to ~394 KiB or ~157 KiB removes UDP partial loss and lowers p50 staleness.
- The remaining ceiling is no longer mainly radio reassembly; it is a combination of CARLA/sensor/front production cadence plus small edge queue behavior.

## Plots

- `plots/track1_oai_reduced_payload/track1_oai_payload_latency_reliability.pdf`
- `plots/track1_oai_reduced_payload/track1_oai_radio_backlog_reduced_payload.pdf`
- `plots/track1_oai_reduced_payload/track1_oai_payload_comparison_summary.csv`

## Recommended next work after Track 1

1. Add real spatial-map worker timing instead of the current assumed +30 ms map compute.
2. Increase/warm the edge receive queue before starting timed captures if we want delivery accounting to exclude startup drops.
3. Carry the reduced-payload findings into the RL/action policy: payload reduction is a reliability and staleness lever, but FPS target alone cannot overcome the CARLA/front production ceiling.
"""
    MD_OUT.write_text(md)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    rows = [summarize_one(label, *spec) for label, spec in RUNS.items()]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "track1_oai_payload_comparison_summary.csv", index=False)
    plot_payload_latency(df)
    plot_radio(df)
    write_markdown(df)
    print(df[[
        "run_id",
        "target_fps",
        "actual_fps",
        "payload_p50_kib",
        "delivery_pct",
        "udp_partial_drops",
        "edge_queue_drops",
        "uplink_transport_p50_ms",
        "capture_tail_p50_ms",
    ]].to_string(index=False))
    print(f"Wrote {OUT}")
    print(f"Wrote {MD_OUT}")


if __name__ == "__main__":
    main()
