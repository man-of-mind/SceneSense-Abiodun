#!/usr/bin/env python3
"""Summarize moving-ego fusion tuning trials against the current baseline.

The script is intentionally useful before and after tuning results exist:

* before remote tuning finishes, it reports the current 8-loop baseline and
  per-density bottlenecks;
* after tuning results are copied back, it scans the tuning experiment folders,
  ranks trials, and plots the deltas against the same baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "analysis_outputs/moving_ego_fusion_tuning"
DENSITIES = ("overall", "low", "medium", "crowded")
TARGET_VEHICLE_IOU = 0.90
TARGET_MIOU = 0.85
TARGET_PERSON_IOU = 0.70

BASELINE_PATHS = {
    "overall": ROOT
    / "experiments/moving_ego_tl16_spawn80_fixedroute_speed60_fusion_train_20260617/eval_moving_model_on_moving/metrics/test_fusion_evaluation_metrics.json",
    "low": ROOT
    / "experiments/moving_ego_tl16_spawn80_fixedroute_speed60_fusion_train_20260617/eval_moving_model_on_moving_low/metrics/test_fusion_evaluation_metrics.json",
    "medium": ROOT
    / "experiments/moving_ego_tl16_spawn80_fixedroute_speed60_fusion_train_20260617/eval_moving_model_on_moving_medium/metrics/test_fusion_evaluation_metrics.json",
    "crowded": ROOT
    / "experiments/moving_ego_tl16_spawn80_fixedroute_speed60_fusion_train_20260617/eval_moving_model_on_moving_crowded/metrics/test_fusion_evaluation_metrics.json",
}

MORE_DATA_PATH = ROOT / (
    "experiments/moving_ego_tl16_spawn80_fixedroute_speed60_fusion_train_20260618_moredata/"
    "eval_moving_model_on_moving/metrics/test_fusion_evaluation_metrics.json"
)


def to_float(value: object, default: float = float("nan")) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_row(
    *,
    experiment: str,
    trial: str,
    density: str,
    path: Path,
    baseline_by_density: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    payload = read_json(path)
    baseline = baseline_by_density.get(density, {})
    vehicle_iou = to_float(payload.get("vehicle_iou"))
    miou = to_float(payload.get("miou"))
    baseline_vehicle = to_float(baseline.get("vehicle_iou"))
    baseline_miou = to_float(baseline.get("miou"))
    return {
        "experiment": experiment,
        "trial": trial,
        "density": density,
        "samples": payload.get("samples", ""),
        "miou": miou,
        "vehicle_iou": vehicle_iou,
        "person_iou": to_float(payload.get("person_iou")),
        "background_iou": to_float(payload.get("background_iou")),
        "pixel_accuracy": to_float(payload.get("pixel_accuracy")),
        "baseline_rgb_miou": to_float(payload.get("baseline_rgb_miou")),
        "baseline_rgb_vehicle_iou": to_float(payload.get("baseline_rgb_vehicle_iou")),
        "fusion_miou_delta_vs_rgb": to_float(payload.get("fusion_miou_delta_vs_rgb")),
        "learned_object_f1": to_float(payload.get("learned_object_f1")),
        "learned_global_xy_mae_m": to_float(payload.get("learned_global_xy_mae_m")),
        "vehicle_gap_to_0p90": max(0.0, TARGET_VEHICLE_IOU - vehicle_iou) if math.isfinite(vehicle_iou) else float("nan"),
        "miou_gap_to_0p85": max(0.0, TARGET_MIOU - miou) if math.isfinite(miou) else float("nan"),
        "vehicle_iou_delta_vs_8loop": vehicle_iou - baseline_vehicle
        if math.isfinite(vehicle_iou) and math.isfinite(baseline_vehicle)
        else float("nan"),
        "miou_delta_vs_8loop": miou - baseline_miou
        if math.isfinite(miou) and math.isfinite(baseline_miou)
        else float("nan"),
        "device": payload.get("device", ""),
        "device_name": payload.get("device_name", ""),
        "metrics_path": str(path),
    }


def load_baseline_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    baseline_payloads = {density: read_json(path) for density, path in BASELINE_PATHS.items() if path.exists()}
    for density, path in BASELINE_PATHS.items():
        if path.exists():
            rows.append(
                metric_row(
                    experiment="baseline_8loop",
                    trial="baseline_8loop",
                    density=density,
                    path=path,
                    baseline_by_density=baseline_payloads,
                )
            )
    if MORE_DATA_PATH.exists():
        rows.append(
            metric_row(
                experiment="moredata_12loop",
                trial="moredata_12loop",
                density="overall",
                path=MORE_DATA_PATH,
                baseline_by_density=baseline_payloads,
            )
        )
    return rows


def parse_density_from_eval_dir(name: str) -> tuple[str, str] | None:
    if not name.startswith("eval_"):
        return None
    stem = name[len("eval_") :]
    for density in DENSITIES:
        suffix = f"_{density}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)], density
    return None


def load_tuning_rows(tuning_glob: str, baseline_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    baseline_by_density = {
        str(row["density"]): row
        for row in baseline_rows
        if row.get("experiment") == "baseline_8loop"
    }
    rows: List[Dict[str, object]] = []
    for exp_dir in sorted(ROOT.glob(tuning_glob)):
        if not exp_dir.is_dir():
            continue
        for metrics_path in sorted(exp_dir.glob("eval_*/metrics/test_fusion_evaluation_metrics.json")):
            parsed = parse_density_from_eval_dir(metrics_path.parents[1].name)
            if parsed is None:
                continue
            trial, density = parsed
            rows.append(
                metric_row(
                    experiment=exp_dir.name,
                    trial=trial,
                    density=density,
                    path=metrics_path,
                    baseline_by_density=baseline_by_density,
                )
            )
    return rows


def ordered_trials(rows: Sequence[Mapping[str, object]]) -> List[str]:
    preferred = ["baseline_8loop", "moredata_12loop"]
    found = []
    for row in rows:
        trial = str(row["trial"])
        if trial not in found:
            found.append(trial)
    return [trial for trial in preferred if trial in found] + [trial for trial in found if trial not in preferred]


def plot_metric_by_density(
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    *,
    field: str,
    title: str,
    ylabel: str,
    target: float | None = None,
    filename: str,
) -> None:
    trials = ordered_trials(rows)
    densities = [density for density in DENSITIES if any(row["density"] == density for row in rows)]
    if not trials or not densities:
        return
    by_key = {(str(row["trial"]), str(row["density"])): row for row in rows}
    x = np.arange(len(densities), dtype=np.float64)
    width = min(0.16, 0.74 / max(1, len(trials)))
    cmap = plt.get_cmap("tab10")
    fig_width = max(10.5, 1.2 * len(densities) + 1.45 * len(trials))
    fig, ax = plt.subplots(figsize=(fig_width, 6.2), constrained_layout=True)
    for idx, trial in enumerate(trials):
        values = [to_float(by_key.get((trial, density), {}).get(field)) for density in densities]
        positions = x + (idx - (len(trials) - 1) / 2.0) * width
        bars = ax.bar(positions, values, width=width, label=trial, color=cmap(idx % 10))
        for bar, value in zip(bars, values):
            if math.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90 if len(trials) > 4 else 0,
                )
    if target is not None:
        ax.axhline(target, color="#444444", linestyle="--", linewidth=1.2, label=f"target {target:.2f}")
    ax.set_xticks(x, [density.capitalize() for density in densities])
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.04 if target is None or target <= 1.0 else None)
    ax.set_title(title, weight="bold", fontsize=15)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(output_dir / f"{filename}.png", dpi=220)
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def plot_delta(rows: Sequence[Mapping[str, object]], output_dir: Path) -> None:
    tuning_rows = [
        row for row in rows if row.get("experiment") not in {"baseline_8loop"} and row.get("density") == "overall"
    ]
    if not tuning_rows:
        return
    tuning_rows = sorted(tuning_rows, key=lambda row: to_float(row.get("vehicle_iou_delta_vs_8loop")), reverse=True)
    labels = [str(row["trial"]) for row in tuning_rows]
    vehicle_delta = [to_float(row.get("vehicle_iou_delta_vs_8loop")) for row in tuning_rows]
    miou_delta = [to_float(row.get("miou_delta_vs_8loop")) for row in tuning_rows]
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(9.0, 1.5 * len(labels)), 5.6), constrained_layout=True)
    ax.bar(x - width / 2, vehicle_delta, width=width, label="Vehicle IoU delta", color="#2f9e75")
    ax.bar(x + width / 2, miou_delta, width=width, label="mIoU delta", color="#3b6fb6")
    ax.axhline(0, color="#333333", linewidth=1.0)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Delta vs 8-loop baseline")
    ax.set_title("Tuning Trial Improvement over Current Moving Baseline", weight="bold")
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.savefig(output_dir / "moving_fusion_tuning_delta_vs_baseline.png", dpi=220)
    fig.savefig(output_dir / "moving_fusion_tuning_delta_vs_baseline.pdf")
    plt.close(fig)


def best_rows(rows: Sequence[Mapping[str, object]]) -> Dict[str, Mapping[str, object]]:
    best: MutableMapping[str, Mapping[str, object]] = {}
    for density in DENSITIES:
        candidates = [
            row for row in rows if row.get("density") == density and math.isfinite(to_float(row.get("vehicle_iou")))
        ]
        if candidates:
            best[density] = max(candidates, key=lambda row: to_float(row.get("vehicle_iou")))
    return dict(best)


def fmt(value: object, digits: int = 3) -> str:
    number = to_float(value)
    return "n/a" if not math.isfinite(number) else f"{number:.{digits}f}"


def write_markdown(path: Path, rows: Sequence[Mapping[str, object]], tuning_count: int) -> None:
    by_key = {(str(row["trial"]), str(row["density"])): row for row in rows}
    best_by_density = best_rows(rows)
    baseline_low = by_key.get(("baseline_8loop", "low"), {})
    baseline_medium = by_key.get(("baseline_8loop", "medium"), {})
    baseline_crowded = by_key.get(("baseline_8loop", "crowded"), {})
    baseline_overall = by_key.get(("baseline_8loop", "overall"), {})

    lines = [
        "# Moving-Ego Fusion Tuning Summary",
        "",
        "## Current Baseline",
        "",
        f"- Overall 8-loop moving model: mIoU `{fmt(baseline_overall.get('miou'))}`, vehicle IoU `{fmt(baseline_overall.get('vehicle_iou'))}`, person IoU `{fmt(baseline_overall.get('person_iou'))}`.",
        f"- Low density is the main vehicle-IoU bottleneck: vehicle IoU `{fmt(baseline_low.get('vehicle_iou'))}`, gap to 0.90 target `{fmt(baseline_low.get('vehicle_gap_to_0p90'))}`.",
        f"- Medium density is also below target: vehicle IoU `{fmt(baseline_medium.get('vehicle_iou'))}`, gap `{fmt(baseline_medium.get('vehicle_gap_to_0p90'))}`.",
        f"- Crowded density already reaches the vehicle target: vehicle IoU `{fmt(baseline_crowded.get('vehicle_iou'))}`.",
        "",
        "## Interpretation",
        "",
        "- More crowded-only data is not the cleanest next move because crowded traffic already performs best. It may improve the easiest bucket while leaving low/medium weak.",
        "- The useful test is whether tuning the objective raises low/medium vehicle IoU without sacrificing crowded performance.",
        "- With person IoU near `0.63`, mIoU `0.85` is mathematically hard unless vehicle IoU gets very high or person IoU also improves. For now, vehicle IoU > `0.90` is the practical near-term gate.",
        "",
        "## Tuning Results",
        "",
    ]
    if tuning_count == 0:
        lines.extend(
            [
                "- No tuning-trial evaluation folders were found yet.",
                "- Copy the remote `experiments/moving_ego_tl16_spawn80_fixedroute_speed60_seg_tuning_*/` folder back and rerun this script; it will automatically rank the trials.",
            ]
        )
    else:
        for density in DENSITIES:
            row = best_by_density.get(density)
            if not row:
                continue
            lines.append(
                f"- Best `{density}` vehicle IoU: `{fmt(row.get('vehicle_iou'))}` from `{row.get('trial')}` "
                f"(delta vs 8-loop `{fmt(row.get('vehicle_iou_delta_vs_8loop'))}`)."
            )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `moving_fusion_tuning_summary.csv`",
            "- `moving_fusion_tuning_vehicle_iou_by_density.png`",
            "- `moving_fusion_tuning_person_iou_by_density.png`",
            "- `moving_fusion_tuning_miou_by_density.png`",
            "- `moving_fusion_tuning_delta_vs_baseline.png`, created once tuning rows exist",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--tuning-glob",
        default="experiments/moving_ego_tl16_spawn80_fixedroute_speed60_seg_tuning_*",
        help="Glob, relative to abiodun/, for tuning experiment directories.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows = load_baseline_rows()
    tuning_rows = load_tuning_rows(args.tuning_glob, baseline_rows)
    rows = baseline_rows + tuning_rows
    if not rows:
        raise FileNotFoundError("No baseline or tuning metrics were found.")

    write_rows(output_dir / "moving_fusion_tuning_summary.csv", rows)
    plot_metric_by_density(
        rows,
        output_dir,
        field="vehicle_iou",
        title="Moving-Ego Fusion Vehicle IoU by Density",
        ylabel="Vehicle IoU",
        target=TARGET_VEHICLE_IOU,
        filename="moving_fusion_tuning_vehicle_iou_by_density",
    )
    plot_metric_by_density(
        rows,
        output_dir,
        field="person_iou",
        title="Moving-Ego Fusion Person IoU by Density",
        ylabel="Person IoU",
        target=TARGET_PERSON_IOU,
        filename="moving_fusion_tuning_person_iou_by_density",
    )
    plot_metric_by_density(
        rows,
        output_dir,
        field="miou",
        title="Moving-Ego Fusion mIoU by Density",
        ylabel="3-class mIoU",
        target=TARGET_MIOU,
        filename="moving_fusion_tuning_miou_by_density",
    )
    plot_delta(rows, output_dir)
    write_markdown(output_dir / "moving_fusion_tuning_summary.md", rows, len(tuning_rows))
    print(f"Wrote {len(rows)} rows ({len(tuning_rows)} tuning rows) to {output_dir}")


if __name__ == "__main__":
    main()
