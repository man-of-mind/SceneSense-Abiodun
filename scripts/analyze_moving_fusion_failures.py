#!/usr/bin/env python3
"""Failure and data-quality analysis for moving-ego RGB+radar fusion models."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DEFAULT = ROOT / "analysis_outputs/moving_ego_fusion_failure_analysis"

RUNS = {
    "moving8": {
        "label": "Moving model, 8 loops",
        "metrics": ROOT
        / "experiments/moving_ego_tl16_spawn80_fixedroute_speed60_fusion_train_20260617/eval_moving_model_on_moving/metrics/test_fusion_evaluation_metrics.json",
        "objects": ROOT
        / "experiments/moving_ego_tl16_spawn80_fixedroute_speed60_fusion_train_20260617/eval_moving_model_on_moving/metrics/test_learned_object_metrics.csv",
        "training": ROOT
        / "experiments/moving_ego_tl16_spawn80_fixedroute_speed60_fusion_train_20260617/metrics/moving_fixedroute_8loops_cap6000_768x432_lr1e-4_bs2_metrics.csv",
    },
    "moving12": {
        "label": "Moving model, 12 loops",
        "metrics": ROOT
        / "experiments/moving_ego_tl16_spawn80_fixedroute_speed60_fusion_train_20260618_moredata/eval_moving_model_on_moving/metrics/test_fusion_evaluation_metrics.json",
        "objects": ROOT
        / "experiments/moving_ego_tl16_spawn80_fixedroute_speed60_fusion_train_20260618_moredata/eval_moving_model_on_moving/metrics/test_learned_object_metrics.csv",
        "training": ROOT
        / "experiments/moving_ego_tl16_spawn80_fixedroute_speed60_fusion_train_20260618_moredata/metrics/moving_fixedroute_12loops_cap9000_768x432_lr1e-4_bs2_metrics.csv",
    },
    "radar12k_pilot": {
        "label": "Moving radar-12k pilot, 2 loops",
        "metrics": ROOT
        / "experiments/moving_ego_radarpps12000_classaware_2loops_cap2200_fusion_train_20260622/eval_pilot_on_pilot_test/metrics/test_fusion_evaluation_metrics.json",
        "objects": ROOT
        / "experiments/moving_ego_radarpps12000_classaware_2loops_cap2200_fusion_train_20260622/eval_pilot_on_pilot_test/metrics/test_learned_object_metrics.csv",
        "training": ROOT
        / "experiments/moving_ego_radarpps12000_classaware_2loops_cap2200_fusion_train_20260622/metrics/moving_ego_radarpps12000_classaware_2loops_cap2200_768x432_lr1e-4_bs2_metrics.csv",
    },
}

PER_DENSITY_SEGMENTATION = {
    "low": ROOT
    / "experiments/moving_ego_tl16_spawn80_fixedroute_speed60_fusion_train_20260617/eval_moving_model_on_moving_low/metrics/test_fusion_evaluation_metrics.json",
    "medium": ROOT
    / "experiments/moving_ego_tl16_spawn80_fixedroute_speed60_fusion_train_20260617/eval_moving_model_on_moving_medium/metrics/test_fusion_evaluation_metrics.json",
    "crowded": ROOT
    / "experiments/moving_ego_tl16_spawn80_fixedroute_speed60_fusion_train_20260617/eval_moving_model_on_moving_crowded/metrics/test_fusion_evaluation_metrics.json",
}

DENSITIES = ("low", "medium", "crowded")
CLASSES = ("vehicle", "person", "all")
STATUSES = ("tp", "fp", "fn")
RUN_ORDER = ("moving8", "moving12", "radar12k_pilot")


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def to_float(value: object, default: float = float("nan")) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def density_from_sample_id(sample_id: str) -> str:
    for density in DENSITIES:
        if f"_{density}_" in sample_id:
            return density
    return "unknown"


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def analyze_object_rows(rows: Sequence[Dict[str, str]]) -> Dict[str, object]:
    counts: MutableMapping[Tuple[str, str], Counter[str]] = defaultdict(Counter)
    xy_errors: MutableMapping[Tuple[str, str], List[float]] = defaultdict(list)
    fp_scores: MutableMapping[Tuple[str, str], List[float]] = defaultdict(list)
    sample_counts: MutableMapping[Tuple[str, str], Counter[str]] = defaultdict(Counter)
    samples_by_density: MutableMapping[str, set[str]] = defaultdict(set)

    for row in rows:
        sample_id = row.get("sample_id", "")
        density = density_from_sample_id(sample_id)
        status = row.get("match_status", "")
        class_name = row.get("class_name") or row.get("gt_class_name") or row.get("pred_class_name") or "unknown"
        samples_by_density[density].add(sample_id)

        for key in ((density, class_name), (density, "all"), ("all", class_name), ("all", "all")):
            counts[key][status] += 1
            sample_counts[key][sample_id] += 1

        if status == "tp":
            err = to_float(row.get("global_xy_error_m"))
            if math.isfinite(err):
                for key in ((density, class_name), (density, "all"), ("all", class_name), ("all", "all")):
                    xy_errors[key].append(err)
        elif status == "fp":
            score = to_float(row.get("score"))
            if math.isfinite(score):
                for key in ((density, class_name), (density, "all"), ("all", class_name), ("all", "all")):
                    fp_scores[key].append(score)

    summary_rows: List[Dict[str, object]] = []
    for density in (*DENSITIES, "all", "unknown"):
        for class_name in CLASSES:
            key = (density, class_name)
            counter = counts.get(key, Counter())
            if not counter and key not in xy_errors and key not in fp_scores:
                continue
            tp = int(counter.get("tp", 0))
            fp = int(counter.get("fp", 0))
            fn = int(counter.get("fn", 0))
            precision, recall, f1 = prf(tp, fp, fn)
            errs = xy_errors.get(key, [])
            scores = fp_scores.get(key, [])
            summary_rows.append(
                {
                    "density": density,
                    "class_name": class_name,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "xy_error_mean_m": float(np.mean(errs)) if errs else float("nan"),
                    "xy_error_median_m": float(np.median(errs)) if errs else float("nan"),
                    "xy_error_p90_m": percentile(errs, 90),
                    "fp_score_mean": float(np.mean(scores)) if scores else float("nan"),
                    "fp_score_p90": percentile(scores, 90),
                    "unique_samples_with_object_rows": len({sid for sid, count in sample_counts.get(key, {}).items() if count}),
                }
            )

    sample_failure: Counter[str] = Counter()
    sample_density: Dict[str, str] = {}
    for row in rows:
        sample_id = row.get("sample_id", "")
        status = row.get("match_status", "")
        if status in ("fp", "fn"):
            sample_failure[sample_id] += 1
            sample_density[sample_id] = density_from_sample_id(sample_id)
    top_samples = [
        {"sample_id": sample_id, "density": sample_density.get(sample_id, "unknown"), "fp_plus_fn": count}
        for sample_id, count in sample_failure.most_common(25)
    ]

    return {
        "summary_rows": summary_rows,
        "top_samples": top_samples,
        "samples_by_density": {density: len(samples) for density, samples in samples_by_density.items()},
    }


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_all() -> Dict[str, Dict[str, object]]:
    runs: Dict[str, Dict[str, object]] = {}
    for run_id, paths in RUNS.items():
        if not paths["metrics"].exists() or not paths["objects"].exists() or not paths["training"].exists():
            continue
        object_rows = read_csv(paths["objects"])
        runs[run_id] = {
            "label": paths["label"],
            "metrics": read_json(paths["metrics"]),
            "objects": object_rows,
            "object_analysis": analyze_object_rows(object_rows),
            "training": read_csv(paths["training"]),
        }
    return runs


def load_per_density_segmentation() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for density, path in PER_DENSITY_SEGMENTATION.items():
        if not path.exists():
            continue
        payload = read_json(path)
        rows.append(
            {
                "density": density,
                "samples": payload.get("samples", ""),
                "miou": payload.get("miou", ""),
                "vehicle_iou": payload.get("vehicle_iou", ""),
                "person_iou": payload.get("person_iou", ""),
                "background_iou": payload.get("background_iou", ""),
                "pixel_accuracy": payload.get("pixel_accuracy", ""),
                "baseline_rgb_miou": payload.get("baseline_rgb_miou", ""),
                "baseline_rgb_vehicle_iou": payload.get("baseline_rgb_vehicle_iou", ""),
                "baseline_rgb_person_iou": payload.get("baseline_rgb_person_iou", ""),
                "fusion_miou_delta_vs_rgb": payload.get("fusion_miou_delta_vs_rgb", ""),
                "learned_object_f1": payload.get("learned_object_f1", ""),
                "learned_global_xy_mae_m": payload.get("learned_global_xy_mae_m", ""),
                "device": payload.get("device", ""),
                "device_name": payload.get("device_name", ""),
                "metrics_path": str(path),
            }
        )
    return rows


def save_per_density_segmentation_plot(rows: Sequence[Mapping[str, object]], output_dir: Path) -> None:
    if not rows:
        return
    ordered = [row for density in DENSITIES for row in rows if row.get("density") == density]
    labels = [str(row["density"]).capitalize() for row in ordered]
    x = np.arange(len(labels))
    width = 0.24
    fields = [
        ("miou", "3-class mIoU", "#3b6fb6"),
        ("vehicle_iou", "Vehicle IoU", "#2f9e75"),
        ("person_iou", "Person IoU", "#d08a1d"),
    ]
    fig, ax = plt.subplots(figsize=(10.8, 5.6), constrained_layout=True)
    for idx, (field, label, color) in enumerate(fields):
        values = [to_float(row.get(field)) for row in ordered]
        positions = x + (idx - 1) * width
        bars = ax.bar(positions, values, width, label=label, color=color)
        for bar in bars:
            value = bar.get_height()
            if math.isfinite(value):
                ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("IoU")
    ax.set_title("Moving-Ego Fusion Segmentation by Traffic Density", weight="bold", fontsize=15)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3)
    fig.savefig(output_dir / "moving_fusion_segmentation_by_density.png", dpi=220)
    fig.savefig(output_dir / "moving_fusion_segmentation_by_density.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.6, 5.2), constrained_layout=True)
    values = [to_float(row.get("fusion_miou_delta_vs_rgb")) for row in ordered]
    bars = ax.bar(labels, values, color="#5b7f95", width=0.58)
    for bar in bars:
        value = bar.get_height()
        if math.isfinite(value):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"+{value:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, max(values) * 1.25 if values else 1.0)
    ax.set_ylabel("mIoU gain")
    ax.set_title("RGB+Radar Fusion Gain over RGB Baseline by Density", weight="bold", fontsize=15)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.savefig(output_dir / "moving_fusion_gain_vs_rgb_by_density.png", dpi=220)
    fig.savefig(output_dir / "moving_fusion_gain_vs_rgb_by_density.pdf")
    plt.close(fig)


def save_training_curves(runs: Dict[str, Dict[str, object]], output_dir: Path) -> None:
    fields = [
        ("miou", "Validation mIoU"),
        ("vehicle_iou", "Validation vehicle IoU"),
        ("person_iou", "Validation person IoU"),
        ("loc_loss", "Localization loss"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.8), constrained_layout=True)
    colors = {"moving8": "#3b6fb6", "moving12": "#d08a1d", "radar12k_pilot": "#6b5ca5"}
    for ax, (field, title) in zip(axes.flat, fields):
        for run_id, payload in runs.items():
            training = payload["training"]
            epochs = [int(float(row["epoch"])) for row in training]
            values = [to_float(row.get(field)) for row in training]
            ax.plot(epochs, values, label=str(payload["label"]), linewidth=2.0, color=colors.get(run_id))
        ax.set_title(title, weight="bold")
        ax.set_xlabel("Epoch")
        ax.grid(color="#dddddd", linewidth=0.8)
        ax.set_axisbelow(True)
    axes.flat[0].legend(frameon=False, fontsize=10)
    fig.suptitle("Moving-Ego Fusion Training Signals", weight="bold", fontsize=16)
    fig.savefig(output_dir / "moving_fusion_training_failure_signals.png", dpi=220)
    fig.savefig(output_dir / "moving_fusion_training_failure_signals.pdf")
    plt.close(fig)


def save_class_compare(runs: Dict[str, Dict[str, object]], output_dir: Path) -> None:
    run_ids = [run_id for run_id in RUN_ORDER if run_id in runs]
    labels = {
        "moving8": "8 loops",
        "moving12": "12 loops",
        "radar12k_pilot": "12k radar pilot",
    }
    colors = {"moving8": "#3b6fb6", "moving12": "#d08a1d", "radar12k_pilot": "#6b5ca5"}
    metrics_by_class = {
        run_id: {
            (row["class_name"], row["density"]): row
            for row in runs[run_id]["object_analysis"]["summary_rows"]
        }
        for run_id in run_ids
    }

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.8), constrained_layout=True)
    fields = [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("xy_error_mean_m", "XY error (m)"),
    ]
    class_labels = ["vehicle", "person"]
    x = np.arange(len(class_labels))
    width = 0.72 / max(1, len(run_ids))
    for ax, (field, title) in zip(axes.flat, fields):
        for idx, run_id in enumerate(run_ids):
            values = [
                float(metrics_by_class[run_id].get((class_name, "all"), {}).get(field, float("nan")))
                for class_name in class_labels
            ]
            offset = (idx - (len(run_ids) - 1) / 2) * width
            bars = ax.bar(x + offset, values, width, color=colors[run_id], label=labels[run_id])
            for bar in bars:
                value = bar.get_height()
                if math.isfinite(value):
                    ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x, class_labels)
        ax.set_title(title, weight="bold")
        if field != "xy_error_mean_m":
            ax.set_ylim(0, 1.0)
        ax.grid(axis="y", color="#dddddd", linewidth=0.8)
        ax.set_axisbelow(True)
    axes.flat[0].legend(frameon=False)
    fig.suptitle("Moving-Ego Object-Head Failure by Class", weight="bold", fontsize=16)
    fig.savefig(output_dir / "moving_fusion_object_failures_by_class.png", dpi=220)
    fig.savefig(output_dir / "moving_fusion_object_failures_by_class.pdf")
    plt.close(fig)


def save_density_status_plot(runs: Dict[str, Dict[str, object]], output_dir: Path, run_id: str) -> None:
    if run_id not in runs:
        return
    rows = runs[run_id]["object_analysis"]["summary_rows"]
    by_density = {row["density"]: row for row in rows if row["class_name"] == "all"}
    densities = [density for density in DENSITIES if density in by_density]
    x = np.arange(len(densities))
    bottom = np.zeros(len(densities))
    colors = {"tp": "#2f9e75", "fp": "#d08a1d", "fn": "#c76c5b"}
    fig, ax = plt.subplots(figsize=(9.4, 5.4), constrained_layout=True)
    for status in STATUSES:
        values = np.array([float(by_density[density].get(status, 0.0)) for density in densities])
        ax.bar(x, values, bottom=bottom, color=colors[status], label=status.upper())
        bottom += values
    ax.set_xticks(x, [density.capitalize() for density in densities])
    ax.set_ylabel("Object-head rows")
    ax.set_title(f"{runs[run_id]['label']}: Localization TP/FP/FN by Density", weight="bold")
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3)
    fig.savefig(output_dir / f"{run_id}_localization_status_by_density.png", dpi=220)
    fig.savefig(output_dir / f"{run_id}_localization_status_by_density.pdf")
    plt.close(fig)


def best_training_row(training: Sequence[Dict[str, str]], field: str) -> Dict[str, str]:
    return max(training, key=lambda row: to_float(row.get(field), default=-math.inf))


def write_markdown(
    runs: Dict[str, Dict[str, object]],
    output_dir: Path,
    per_density_segmentation: Sequence[Mapping[str, object]],
) -> None:
    lines: List[str] = [
        "# Moving-Ego Fusion Failure and Data-Quality Analysis",
        "",
        "## What We Can Inspect Locally",
        "",
        "- Evaluation metrics JSON, learned-object TP/FP/FN CSVs, training curves, and the remote GPU per-density segmentation metrics are available locally.",
        "",
        "## Key Findings",
        "",
    ]
    if "moving8" in runs and "moving12" in runs:
        m8 = runs["moving8"]["metrics"]
        m12 = runs["moving12"]["metrics"]
        lines.extend(
            [
                f"- The 8-loop model remains better for segmentation: mIoU `{to_float(m8.get('miou')):.3f}`, vehicle IoU `{to_float(m8.get('vehicle_iou')):.3f}`, person IoU `{to_float(m8.get('person_iou')):.3f}`.",
                f"- The 12-loop model is worse on segmentation despite more samples: mIoU `{to_float(m12.get('miou')):.3f}`, vehicle IoU `{to_float(m12.get('vehicle_iou')):.3f}`, person IoU `{to_float(m12.get('person_iou')):.3f}`.",
                f"- The 12-loop model improves localization slightly: object F1 `{to_float(m8.get('learned_object_f1')):.3f}` -> `{to_float(m12.get('learned_object_f1')):.3f}`, XY MAE `{to_float(m8.get('learned_global_xy_mae_m')):.3f}m` -> `{to_float(m12.get('learned_global_xy_mae_m')):.3f}m`.",
                "- The improvement is class-skewed: vehicle localization improves, while person localization slightly degrades.",
            ]
        )
    if "radar12k_pilot" in runs:
        pilot = runs["radar12k_pilot"]["metrics"]
        lines.append(
            f"- The 12k-radar/class-aware pilot did not beat the 8-loop model yet: mIoU `{to_float(pilot.get('miou')):.3f}`, vehicle IoU `{to_float(pilot.get('vehicle_iou')):.3f}`, person IoU `{to_float(pilot.get('person_iou')):.3f}`, object F1 `{to_float(pilot.get('learned_object_f1')):.3f}`."
        )
    if per_density_segmentation:
        by_density = {str(row["density"]): row for row in per_density_segmentation}
        lines.extend(["", "## Per-Density Segmentation"])
        for density in DENSITIES:
            row = by_density.get(density)
            if not row:
                continue
            lines.append(
                f"- `{density}`: mIoU `{to_float(row.get('miou')):.3f}`, vehicle IoU `{to_float(row.get('vehicle_iou')):.3f}`, person IoU `{to_float(row.get('person_iou')):.3f}`, fusion gain over RGB baseline `+{to_float(row.get('fusion_miou_delta_vs_rgb')):.3f}`."
            )
        lines.append(
            "- Segmentation does not simply collapse in crowded traffic. Vehicle IoU improves with density because there are more visible vehicle pixels, while person IoU is slightly best in medium density and lowest in crowded scenes."
        )
    for run_id in RUN_ORDER:
        if run_id not in runs:
            continue
        lines.extend(["", f"## {runs[run_id]['label']}"])
        summary = {
            (row["density"], row["class_name"]): row
            for row in runs[run_id]["object_analysis"]["summary_rows"]
        }
        for class_name in ("vehicle", "person", "all"):
            row = summary.get(("all", class_name), {})
            if not row:
                continue
            lines.append(
                f"- `{class_name}` localization: precision `{float(row['precision']):.3f}`, recall `{float(row['recall']):.3f}`, F1 `{float(row['f1']):.3f}`, XY MAE `{float(row['xy_error_mean_m']):.3f}m`."
            )
        lines.append("- Density-level localization pressure:")
        for density in DENSITIES:
            row = summary.get((density, "all"), {})
            if row:
                lines.append(
                    f"  - `{density}`: TP `{int(row['tp'])}`, FP `{int(row['fp'])}`, FN `{int(row['fn'])}`, F1 `{float(row['f1']):.3f}`."
                )
        best_miou = best_training_row(runs[run_id]["training"], "miou")
        best_person = best_training_row(runs[run_id]["training"], "person_iou")
        lines.append(
            f"- Training peaked at validation mIoU `{to_float(best_miou.get('miou')):.3f}` on epoch `{best_miou.get('epoch')}` and person IoU `{to_float(best_person.get('person_iou')):.3f}` on epoch `{best_person.get('epoch')}`."
        )
    lines.extend(
        [
            "",
            "## Diagnosis",
            "",
            "- Repeating the same route more times mostly adds near-neighbor views. It increases object-head training examples, but does not add enough new visual geometry to improve segmentation.",
            "- Pixel segmentation is reasonably stable across density; the bigger weakness is object/localization reliability, especially false negatives in medium/crowded scenes.",
            "- The moving model's segmentation ceiling is now likely caused by route/view diversity, class imbalance, and/or label difficulty rather than insufficient epochs alone.",
            "- The object head is still the weakest piece. False positives are high in every density bucket, and false negatives dominate medium/crowded scenes.",
            "- Person localization is not fixed by more route loops; this supports the LiDAR/radar-processing investigation for sparse pedestrian returns.",
            "",
            "## Recommended Next Experiment",
            "",
            "1. Keep the 8-loop moving checkpoint as the current best segmentation checkpoint.",
            "2. Stop adding repeated loops on the same route as the first fix; per-density segmentation shows the model already handles the three density levels fairly consistently.",
            "3. Try a targeted training recipe instead of another repeated-route data run: stronger class weighting/person sampling, lower object score threshold sweep, and radar-processing changes for pedestrian support.",
            "4. Add route diversity only if it changes viewpoint geometry, not just loop count.",
        ]
    )
    (output_dir / "moving_fusion_failure_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_all()
    if not runs:
        raise FileNotFoundError("No moving-fusion metrics/object CSVs were found.")

    all_summary_rows: List[Dict[str, object]] = []
    all_top_rows: List[Dict[str, object]] = []
    for run_id, payload in runs.items():
        for row in payload["object_analysis"]["summary_rows"]:
            all_summary_rows.append({"run_id": run_id, "run_label": payload["label"], **row})
        for row in payload["object_analysis"]["top_samples"]:
            all_top_rows.append({"run_id": run_id, "run_label": payload["label"], **row})
    write_rows(output_dir / "moving_fusion_localization_failure_summary.csv", all_summary_rows)
    write_rows(output_dir / "moving_fusion_top_failure_samples.csv", all_top_rows)
    per_density_segmentation = load_per_density_segmentation()
    write_rows(output_dir / "moving_fusion_segmentation_by_density.csv", per_density_segmentation)
    save_per_density_segmentation_plot(per_density_segmentation, output_dir)
    save_training_curves(runs, output_dir)
    save_class_compare(runs, output_dir)
    save_density_status_plot(runs, output_dir, "moving12")
    save_density_status_plot(runs, output_dir, "radar12k_pilot")
    write_markdown(runs, output_dir, per_density_segmentation)
    print(f"Wrote moving-fusion failure analysis to {output_dir}")


if __name__ == "__main__":
    main()
