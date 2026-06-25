#!/usr/bin/env python3
"""Summarize the r100k/r4/tw2 moving-fusion focused checks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METRICS_NAME = "metrics/test_fusion_evaluation_metrics.json"
DENSITIES = ("overall", "low", "medium", "crowded")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-exp",
        type=Path,
        default=ROOT
        / "experiments/moving_ego_radarpps100000_bboxsupport_r4_tw2_2loops_cap3200_fusion_train_20260624_r100k_r4_tw2_pilot",
    )
    parser.add_argument(
        "--tuning-exp",
        type=Path,
        default=ROOT / "experiments/moving_ego_radarpps100000_bboxsupport_r4_tw2_person_tuning_20260624",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "analysis_outputs/r100k_r4_tw2_person_focus",
    )
    return parser.parse_args()


def to_float(value: object, default: float = float("nan")) -> float:
    try:
        if value in ("", None):
            return default
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def base_metric_path(exp: Path, density: str) -> Path:
    if density == "overall":
        return exp / "eval_pilot_on_pilot_test" / METRICS_NAME
    return exp / f"eval_pilot_on_{density}_test" / METRICS_NAME


def row_from_metrics(label: str, density: str, path: Path) -> Dict[str, object]:
    payload = read_json(path)
    return {
        "run": label,
        "density": density,
        "samples": payload.get("samples", ""),
        "miou": to_float(payload.get("miou")),
        "vehicle_iou": to_float(payload.get("vehicle_iou")),
        "person_iou": to_float(payload.get("person_iou")),
        "learned_object_f1": to_float(payload.get("learned_object_f1")),
        "learned_vehicle_object_f1": to_float(payload.get("learned_vehicle_object_f1")),
        "learned_person_object_f1": to_float(payload.get("learned_person_object_f1")),
        "learned_global_xy_mae_m": to_float(payload.get("learned_global_xy_mae_m")),
        "device": payload.get("device", ""),
        "metrics_path": str(path),
    }


def load_rows(base_exp: Path, tuning_exp: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for density in DENSITIES:
        path = base_metric_path(base_exp, density)
        if path.exists():
            rows.append(row_from_metrics("r100k_r4_tw2_base", density, path))

    if tuning_exp.exists():
        for path in sorted(tuning_exp.glob(f"eval_*/{METRICS_NAME}")):
            eval_name = path.parents[1].name
            if not eval_name.startswith("eval_"):
                continue
            stem = eval_name[len("eval_") :]
            density = ""
            for candidate in DENSITIES:
                suffix = f"_{candidate}"
                if stem.endswith(suffix):
                    density = candidate
                    trial = stem[: -len(suffix)]
                    break
            else:
                continue
            rows.append(row_from_metrics(trial, density, path))
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(rows: Sequence[Mapping[str, object]], output_dir: Path, field: str, title: str, filename: str) -> None:
    runs = []
    for row in rows:
        run = str(row["run"])
        if run not in runs:
            runs.append(run)
    densities = [density for density in DENSITIES if any(row["density"] == density for row in rows)]
    if not runs or not densities:
        return

    by_key = {(str(row["run"]), str(row["density"])): row for row in rows}
    x = np.arange(len(densities), dtype=np.float64)
    width = min(0.16, 0.78 / max(1, len(runs)))
    fig, ax = plt.subplots(figsize=(max(10.5, 1.25 * len(densities) + 1.45 * len(runs)), 6.0), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    for idx, run in enumerate(runs):
        values = [to_float(by_key.get((run, density), {}).get(field)) for density in densities]
        positions = x + (idx - (len(runs) - 1) / 2.0) * width
        bars = ax.bar(positions, values, width=width, label=run, color=cmap(idx % 10))
        for bar, value in zip(bars, values):
            if math.isfinite(value):
                ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8, rotation=90 if len(runs) > 4 else 0)

    ax.set_xticks(x, [density.capitalize() for density in densities])
    ax.set_ylabel(field.replace("_", " ").title())
    ax.set_ylim(0, 1.04)
    ax.set_title(title, weight="bold")
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(output_dir / f"{filename}.png", dpi=220)
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def fmt(value: object) -> str:
    number = to_float(value)
    return "n/a" if not math.isfinite(number) else f"{number:.3f}"


def write_markdown(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    lines = [
        "# R100k/R4/TW2 Person-Focused Checks",
        "",
        "| Run | Density | mIoU | Vehicle IoU | Person IoU | Person F1 | Vehicle F1 | XY MAE (m) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run']} | {row['density']} | {fmt(row.get('miou'))} | {fmt(row.get('vehicle_iou'))} | "
            f"{fmt(row.get('person_iou'))} | {fmt(row.get('learned_person_object_f1'))} | "
            f"{fmt(row.get('learned_vehicle_object_f1'))} | {fmt(row.get('learned_global_xy_mae_m'))} |"
        )

    overall = [row for row in rows if row.get("density") == "overall"]
    if overall:
        best_person = max(overall, key=lambda row: to_float(row.get("person_iou"), -math.inf))
        best_vehicle = max(overall, key=lambda row: to_float(row.get("vehicle_iou"), -math.inf))
        lines.extend(
            [
                "",
                "## Quick Read",
                "",
                f"- Best overall person IoU: `{fmt(best_person.get('person_iou'))}` from `{best_person.get('run')}`.",
                f"- Best overall vehicle IoU: `{fmt(best_vehicle.get('vehicle_iou'))}` from `{best_vehicle.get('run')}`.",
                "- If person-focused selection improves person IoU but hurts vehicle IoU, that confirms a model-objective tradeoff rather than only a radar-representation issue.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.base_exp.expanduser().resolve(), args.tuning_exp.expanduser().resolve())
    if not rows:
        raise FileNotFoundError("No focused-check metrics were found.")
    write_csv(output_dir / "r100k_r4_tw2_person_focus_summary.csv", rows)
    write_markdown(output_dir / "r100k_r4_tw2_person_focus_summary.md", rows)
    plot_metric(rows, output_dir, "person_iou", "Person IoU by Density", "person_iou_by_density")
    plot_metric(rows, output_dir, "vehicle_iou", "Vehicle IoU by Density", "vehicle_iou_by_density")
    plot_metric(rows, output_dir, "miou", "mIoU by Density", "miou_by_density")
    print(f"Wrote {len(rows)} focused-check rows to {output_dir}")


if __name__ == "__main__":
    main()
