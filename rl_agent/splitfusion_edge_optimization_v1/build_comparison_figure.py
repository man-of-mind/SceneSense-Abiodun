#!/usr/bin/env python3
"""Build the presentation comparison for the qualified edge optimization."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ACTION_ORDER = (30, 15, 50, 71)
ACTION_LABELS = {
    30: "A30\nAE128\nq=0",
    15: "A15\nnoAE\nq=0.70",
    50: "A50\nAE64\nq=0.50",
    71: "A71\nAE32\nq=0.98",
}
STAGES = (
    ("Tail inference", "tail_inference_block"),
    ("Camera-aware\npostprocess", "camera_aware_postprocess"),
    ("p025 filter", "p025_service_filter"),
    ("Serialization", "compact_result_serialization"),
    ("Edge queue", "edge_queue_wait_ms"),
)


def _load_actions(path: Path) -> dict[int, dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(row["action_id"]): row
        for row in document["comparisons"]["per_action"]
    }


def _stage(row: dict, key: str) -> float:
    if key == "edge_queue_wait_ms":
        return float(row[key]["median"])
    return float(row["live_stage_group_medians_ms"][key])


def _bold_axes(axis) -> None:
    axis.tick_params(axis="both", labelsize=10, width=1.2)
    for label in axis.get_xticklabels() + axis.get_yticklabels():
        label.set_fontweight("bold")
    axis.xaxis.label.set_fontweight("bold")
    axis.yaxis.label.set_fontweight("bold")
    axis.title.set_fontweight("bold")
    axis.grid(axis="y", alpha=0.22, linewidth=0.8)


def build(baseline_path: Path, optimized_path: Path, output_dir: Path) -> None:
    baseline = _load_actions(baseline_path)
    optimized = _load_actions(optimized_path)
    missing = set(ACTION_ORDER) - baseline.keys() | set(ACTION_ORDER) - optimized.keys()
    if missing:
        raise RuntimeError(f"missing actions: {sorted(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "edge_optimization_before_after.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "action_id",
                "profile_id",
                "baseline_edge_service_ms_median",
                "optimized_edge_service_ms_median",
                "direct_saving_ms",
                "baseline_edge_queue_ms_median",
                "optimized_edge_queue_ms_median",
                "tail_inference_baseline_ms_median",
                "tail_inference_optimized_ms_median",
                "postprocess_baseline_ms_median",
                "postprocess_optimized_ms_median",
                "p025_baseline_ms_median",
                "p025_optimized_ms_median",
                "serialization_baseline_ms_median",
                "serialization_optimized_ms_median",
            ]
        )
        for action_id in ACTION_ORDER:
            before = baseline[action_id]
            after = optimized[action_id]
            before_service = float(before["deployed_tail_service_ms"]["median"])
            after_service = float(after["deployed_tail_service_ms"]["median"])
            writer.writerow(
                [
                    action_id,
                    after["profile_id"],
                    f"{before_service:.6f}",
                    f"{after_service:.6f}",
                    f"{before_service - after_service:.6f}",
                    f"{_stage(before, 'edge_queue_wait_ms'):.6f}",
                    f"{_stage(after, 'edge_queue_wait_ms'):.6f}",
                    f"{_stage(before, 'tail_inference_block'):.6f}",
                    f"{_stage(after, 'tail_inference_block'):.6f}",
                    f"{_stage(before, 'camera_aware_postprocess'):.6f}",
                    f"{_stage(after, 'camera_aware_postprocess'):.6f}",
                    f"{_stage(before, 'p025_service_filter'):.6f}",
                    f"{_stage(after, 'p025_service_filter'):.6f}",
                    f"{_stage(before, 'compact_result_serialization'):.6f}",
                    f"{_stage(after, 'compact_result_serialization'):.6f}",
                ]
            )

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 1.2})
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.5), constrained_layout=True)
    width = 0.36

    x = np.arange(len(ACTION_ORDER))
    before_service = np.array(
        [float(baseline[a]["deployed_tail_service_ms"]["median"]) for a in ACTION_ORDER]
    )
    after_service = np.array(
        [float(optimized[a]["deployed_tail_service_ms"]["median"]) for a in ACTION_ORDER]
    )
    axes[0].bar(x - width / 2, before_service, width, label="Before", color="#a7b6c8")
    axes[0].bar(x + width / 2, after_service, width, label="After", color="#176b87")
    for index, (before, after) in enumerate(zip(before_service, after_service)):
        axes[0].text(
            index,
            max(before, after) + 4,
            f"−{before - after:.1f} ms",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    axes[0].set_xticks(x, [ACTION_LABELS[a] for a in ACTION_ORDER])
    axes[0].set_ylabel("Median edge service time (ms)")
    axes[0].set_title("Deployed edge service (all UINT4)")
    axes[0].set_ylim(0, max(before_service) * 1.22)
    axes[0].legend(frameon=False, prop={"weight": "bold"})
    _bold_axes(axes[0])

    sx = np.arange(len(STAGES))
    before_stages = np.array(
        [np.median([_stage(baseline[a], key) for a in ACTION_ORDER]) for _, key in STAGES]
    )
    after_stages = np.array(
        [np.median([_stage(optimized[a], key) for a in ACTION_ORDER]) for _, key in STAGES]
    )
    axes[1].bar(sx - width / 2, before_stages, width, label="Before", color="#a7b6c8")
    axes[1].bar(sx + width / 2, after_stages, width, label="After", color="#176b87")
    axes[1].set_xticks(sx, [label for label, _ in STAGES])
    axes[1].set_ylabel("Median stage time across actions (ms)")
    axes[1].set_title("Where the saving comes from")
    axes[1].set_ylim(0, max(before_stages) * 1.24)
    axes[1].legend(frameon=False, prop={"weight": "bold"})
    _bold_axes(axes[1])

    fig.suptitle(
        "SplitFusion Edge Optimization — Before vs After\n"
        "Edge-side stages only; sensor preparation and radio transport are excluded",
        fontsize=15,
        fontweight="bold",
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"09_edge_optimization_before_after.{suffix}", dpi=240)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.baseline, args.optimized, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
