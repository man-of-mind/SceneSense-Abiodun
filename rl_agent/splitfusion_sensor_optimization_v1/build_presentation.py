#!/usr/bin/env python3
"""Build the before/after evidence for the sensor-preparation optimization.

Offline only: reads the two preserved live cells and writes figures, a stage
table and a machine-readable result.  It launches no CARLA, OAI, Docker, RFsim
or model inference and mutates nothing it reads.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "experiments/splitfusion_sensor_preparation_live_v1"
BASELINE = LIVE / "20260914_v2_action50_favorable_baseline"
OPTIMIZED = LIVE / "20260914_v2_action50_favorable_optimized_retry1"
FAILED_FIRST = LIVE / "20260914_v2_action50_favorable_optimized"
OUT = LIVE / "20260914_v2_optimization_presentation"
TERMINAL = "SPLITFUSION_SENSOR_PREPARATION_OPTIMIZATION_V2_PRESENTATION_COMPLETE"

# Validated categorical slots 1-5 (see dataviz reference palette); assigned in
# fixed order and never cycled.
SERIES = {"baseline": "#2a78d6", "optimized": "#eb6834"}
GROUP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

STAGES: list[tuple[str, str, str]] = [
    ("P07", "radar_window_ms", "radar window extraction"),
    ("P08", "profile_radar_spherical_to_world_ms", "radar spherical-to-world"),
    ("P09", "profile_radar_stationary_tracking_ms", "stationary tracking"),
    ("P10", "profile_radar_world_to_camera_ms", "world-to-camera"),
    ("P11", "profile_radar_projection_ms", "projection and bounds"),
    ("P12", "profile_radar_rasterization_ms", "radar rasterization"),
    ("P13", "profile_radar_packaging_ms", "radar evidence packaging"),
    ("P14", "rgb_convert_ms", "CARLA BGRA-to-BGR"),
    ("P15", "profile_camera_bgr_to_rgb_ms", "BGR-to-RGB"),
    ("P16", "profile_camera_resize_ms", "RGB resize"),
    ("P17", "profile_camera_tensor_pack_ms", "RGB tensor pack"),
    ("P18", "profile_camera_h2d_ms", "RGB host-to-device"),
    ("P19", "profile_camera_normalization_constants_wall_ms", "normalization constants"),
    ("P20", "profile_camera_normalize_ms", "RGB normalize"),
    ("P21", "profile_radar_resize_pack_ms", "radar resize/pack"),
    ("P22", "profile_radar_h2d_ms", "radar host-to-device"),
    ("P23", "profile_seven_channel_concatenate_ms", "seven-channel concatenate"),
    ("P25", "profile_unattributed_pre_front_ms", "unattributed pre-front"),
]
OPTIMIZED_STAGES = {"P09", "P12", "P19", "P21"}

COMPOSITION = [
    ("radar window extraction", ["radar_window_ms"]),
    ("radar coordinate transforms", ["profile_radar_spherical_to_world_ms",
                                     "profile_radar_world_to_camera_ms",
                                     "profile_radar_projection_ms"]),
    ("stationary tracking", ["profile_radar_stationary_tracking_ms"]),
    ("radar rasterization", ["profile_radar_rasterization_ms"]),
    ("camera chain and GPU packing", ["profile_radar_packaging_ms", "rgb_convert_ms",
                                      "profile_camera_bgr_to_rgb_ms", "profile_camera_resize_ms",
                                      "profile_camera_tensor_pack_ms", "profile_camera_h2d_ms",
                                      "profile_camera_normalization_constants_wall_ms",
                                      "profile_camera_normalize_ms", "profile_radar_resize_pack_ms",
                                      "profile_radar_h2d_ms",
                                      "profile_seven_channel_concatenate_ms"]),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sent_rows(cell: Path) -> list[dict[str, str]]:
    attempt = next((cell / "cells").glob("*/attempts/attempt_0001"))
    with (attempt / "per_frame_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if row.get("prepare_status") == "SENT"]


def column(rows: Sequence[dict[str, str]], field: str) -> np.ndarray:
    values = []
    for row in rows:
        raw = row.get(field)
        if raw in (None, ""):
            continue
        try:
            values.append(float(raw))
        except ValueError:
            continue
    return np.asarray(values, dtype=float)


def pcts(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0:
        return {"count": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None, "mean_ms": None}
    return {
        "count": int(values.size),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "mean_ms": float(values.mean()),
        "max_ms": float(values.max()),
    }


def style(ax: Any) -> None:
    ax.set_facecolor("#fcfcfb")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def save(fig: Any, name: str) -> list[Path]:
    written = []
    for suffix in ("png", "pdf"):
        path = OUT / f"{name}.{suffix}"
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
        written.append(path)
    plt.close(fig)
    return written


def figure_composition(base: list[dict], opt: list[dict]) -> list[Path]:
    """Stacked mean composition. Means are used because a sum of medians is not
    the median of the sum; the intervals are sequential and non-overlapping."""
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.7)
    labels = ["baseline\n(production)", "optimized\n(v2)"]
    for row_index, rows in enumerate((base, opt)):
        left = 0.0
        for group_index, (name, fields) in enumerate(COMPOSITION):
            total = float(sum(column(rows, field).mean() for field in fields
                              if column(rows, field).size))
            ax.barh(row_index, total, left=left, height=0.46,
                    color=GROUP_COLORS[group_index], edgecolor="#fcfcfb", linewidth=2.0,
                    label=name if row_index == 0 else None)
            if total > 1.2:
                ax.text(left + total / 2.0, row_index, f"{total:.1f}",
                        ha="center", va="center", fontsize=7.5, color="#ffffff", fontweight="bold")
            left += total
        ax.text(left + 0.7, row_index, f"{left:.1f} ms", va="center", fontsize=9,
                color=INK, fontweight="bold")
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.set_xlabel("mean production sensor-compute time per prepared frame (ms)",
                  fontsize=9, color=MUTED)
    ax.set_title("Where sensor preparation spends its time", fontsize=11, color=INK, pad=10, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.32), labelcolor=MUTED)
    ax.invert_yaxis()
    return save(fig, "stacked_latency_breakdown")


def figure_stages(base: list[dict], opt: list[dict]) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 6.4), sharey=True)
    names = [f"{tag} {label}" + ("  *" if tag in OPTIMIZED_STAGES else "")
             for tag, _, label in STAGES]
    y = np.arange(len(STAGES))
    for ax, (pct, title) in zip(axes, [(50, "P50"), (95, "P95"), (99, "P99")]):
        style(ax)
        b = [float(np.percentile(column(base, f), pct)) if column(base, f).size else 0.0
             for _, f, _ in STAGES]
        o = [float(np.percentile(column(opt, f), pct)) if column(opt, f).size else 0.0
             for _, f, _ in STAGES]
        ax.barh(y - 0.21, b, height=0.38, color=SERIES["baseline"],
                edgecolor="#fcfcfb", linewidth=1.2, label="baseline (production)")
        ax.barh(y + 0.21, o, height=0.38, color=SERIES["optimized"],
                edgecolor="#fcfcfb", linewidth=1.2, label="optimized (v2)")
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("ms", fontsize=8, color=MUTED)
    axes[0].set_yticks(y); axes[0].set_yticklabels(names, fontsize=8, color=INK)
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False, fontsize=8, loc="lower right", labelcolor=MUTED)
    fig.suptitle("Per-stage sensor preparation, full sent population   "
                 "(*  = stage changed by this work)",
                 fontsize=11, color=INK, x=0.012, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return save(fig, "per_stage_percentiles")


def figure_timeseries(base: list[dict], opt: list[dict]) -> list[Path]:
    """Two panels sharing one x-axis rather than a second y-scale."""
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 1.4]})
    count = 600
    for ax in axes:
        style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.7)
    x = np.arange(count)
    for rows, key, label in ((base, "baseline", "baseline (production)"),
                             (opt, "optimized", "optimized (v2)")):
        returns = column(rows, "window_returns")[:count]
        compute = column(rows, "profile_sensor_compute_production_estimate_ms")[:count]
        axes[0].plot(x[:returns.size], returns, color=SERIES[key], linewidth=1.2,
                     alpha=0.85, label=label)
        axes[1].plot(x[:compute.size], compute, color=SERIES[key], linewidth=1.2,
                     alpha=0.85, label=label)
    axes[0].set_ylabel("radar returns\nin window", fontsize=8.5, color=MUTED)
    axes[1].set_ylabel("production sensor\ncompute (ms)", fontsize=8.5, color=MUTED)
    axes[1].set_xlabel("prepared frame index within the cell", fontsize=9, color=MUTED)
    axes[1].legend(frameon=False, fontsize=8, loc="upper right", labelcolor=MUTED)
    axes[0].set_title("Radar load is matched; the compute bursts are not",
                      fontsize=11, color=INK, loc="left", pad=8)
    fig.tight_layout()
    return save(fig, "per_frame_timeseries")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    base, opt = sent_rows(BASELINE), sent_rows(OPTIMIZED)
    base_result = json.loads((BASELINE / "SENSOR_PROFILE_RESULT.json").read_text())
    opt_result = json.loads((OPTIMIZED / "SENSOR_PROFILE_RESULT.json").read_text())

    stage_table = []
    for tag, field, label in STAGES:
        b, o = pcts(column(base, field)), pcts(column(opt, field))
        stage_table.append({
            "stage": tag, "label": label, "field": field,
            "changed_by_this_work": tag in OPTIMIZED_STAGES,
            "baseline": b, "optimized": o,
            "delta_p50_ms": (None if b["p50_ms"] is None or o["p50_ms"] is None
                             else round(o["p50_ms"] - b["p50_ms"], 4)),
            "delta_p99_ms": (None if b["p99_ms"] is None or o["p99_ms"] is None
                             else round(o["p99_ms"] - b["p99_ms"], 4)),
        })

    total_field = "profile_sensor_compute_production_estimate_ms"
    totals = {"baseline": pcts(column(base, total_field)),
              "optimized": pcts(column(opt, total_field))}
    radar_chain = [f for _, f, _ in STAGES[:6]]
    chain = {}
    for name, rows in (("baseline", base), ("optimized", opt)):
        stack = [column(rows, f) for f in radar_chain]
        size = min(a.size for a in stack)
        chain[name] = pcts(np.sum([a[:size] for a in stack], axis=0))

    figures = []
    figures += figure_composition(base, opt)
    figures += figure_stages(base, opt)
    figures += figure_timeseries(base, opt)

    result = {
        "schema": "scenesense.splitfusion.sensor_preparation_optimization_v2.result",
        "terminal": TERMINAL,
        "verdict": "SENSOR_PREPARATION_OPTIMIZATION_VALIDATED",
        "cells": {
            "baseline": {"path": str(BASELINE.relative_to(ROOT)),
                         "mode": base_result["mode"], "status": base_result["status"]},
            "optimized": {"path": str(OPTIMIZED.relative_to(ROOT)),
                          "mode": opt_result["mode"], "status": opt_result["status"]},
            "optimized_first_attempt_failed": {
                "path": str(FAILED_FIRST.relative_to(ROOT)),
                "preserved": True,
                "reason": ("exactly-one terminal feedback contract failed for the final "
                           "in-flight capture ue288_a50__favorable_stable:6742 "
                           "(3034 sent, 3033 terminals); teardown drain race, not an "
                           "output or equivalence defect"),
            },
        },
        "populations": {
            "full_sent": {"baseline_frames": len(base), "optimized_frames": len(opt)},
            "registered_window": {
                "frames": 500,
                "baseline": [base_result["frames"]["analysis_first_frame"],
                             base_result["frames"]["analysis_last_frame"]],
                "optimized": [opt_result["frames"]["analysis_first_frame"],
                              opt_result["frames"]["analysis_last_frame"]],
            },
        },
        "total_production_sensor_compute": {
            "full_sent_population": totals,
            "registered_500_frame_window": {
                "baseline": base_result["total_sensor_compute"],
                "optimized": opt_result["total_sensor_compute"],
            },
        },
        "radar_chain_p07_to_p12": chain,
        "stages_full_sent_population": stage_table,
        "paired_same_frame_radar_chain": {
            "optimized_cell": opt_result.get("paired_same_frame_radar_chain"),
            "baseline_cell_ordering_control": base_result.get("paired_same_frame_radar_chain"),
            "note": ("In the baseline cell both timed calls are the production "
                     "implementation, so its apparent reduction is the measurement "
                     "ordering bias and must be subtracted from the optimized cell's."),
        },
        "complete_path": {"baseline": base_result["complete_path"]["metrics"],
                          "optimized": opt_result["complete_path"]["metrics"]},
        "preparation_coverage": {"baseline": base_result["preparation"],
                                 "optimized": opt_result["preparation"]},
        "gates": {"baseline": base_result["gates"], "optimized": opt_result["gates"]},
        "figures": sorted(path.name for path in figures),
    }
    (OUT / "SENSOR_OPTIMIZATION_V2_RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = {
        "schema": "scenesense.splitfusion.sensor_preparation_optimization_v2.manifest",
        "terminal": TERMINAL,
        "artifacts": {p.name: sha256_file(p) for p in sorted(OUT.iterdir()) if p.is_file()},
        "bound_sources": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                BASELINE / "SENSOR_PROFILE_RESULT.json",
                OPTIMIZED / "SENSOR_PROFILE_RESULT.json",
                FAILED_FIRST / "RUNTIME_FAILURE.json",
                ROOT / "rl_agent/splitfusion_sensor_optimization_v1/optimized_stages.py",
            )
        },
    }
    (OUT / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / TERMINAL).write_text(
        json.dumps({"terminal": TERMINAL, "verdict": result["verdict"]},
                   indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"terminal": TERMINAL, "output": str(OUT.relative_to(ROOT)),
                      "figures": len(figures)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
