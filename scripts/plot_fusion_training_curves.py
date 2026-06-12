#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def _float_or_none(value: str) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _load_rows(metrics_csv: Path) -> List[Dict[str, str]]:
    with metrics_csv.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"No metric rows found in {metrics_csv}")
    return rows


def _series(rows: Sequence[Dict[str, str]], key: str) -> Tuple[List[int], List[float]]:
    xs: List[int] = []
    ys: List[float] = []
    for idx, row in enumerate(rows):
        value = _float_or_none(row.get(key, ""))
        if value is None:
            continue
        epoch = _float_or_none(row.get("epoch", ""))
        xs.append(int(epoch) if epoch is not None else idx)
        ys.append(value)
    return xs, ys


def _available(rows: Sequence[Dict[str, str]], keys: Iterable[str]) -> List[str]:
    return [key for key in keys if any(_float_or_none(row.get(key, "")) is not None for row in rows)]


def _plot_group(ax, rows: Sequence[Dict[str, str]], keys: Sequence[str], title: str, ylabel: str) -> bool:
    plotted = False
    for key in keys:
        xs, ys = _series(rows, key)
        if not ys:
            continue
        ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=3.5, label=key)
        plotted = True
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.28)
    if plotted:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No metrics", ha="center", va="center", transform=ax.transAxes)
    return plotted


def _summary(rows: Sequence[Dict[str, str]], metrics_csv: Path) -> Dict[str, object]:
    numeric_keys = [
        key
        for key in rows[0].keys()
        if key not in {"trial", "timestamp"} and any(_float_or_none(row.get(key, "")) is not None for row in rows)
    ]
    best: Dict[str, Dict[str, float | int]] = {}
    for key in numeric_keys:
        values: List[Tuple[int, float]] = []
        for idx, row in enumerate(rows):
            value = _float_or_none(row.get(key, ""))
            if value is None:
                continue
            epoch = _float_or_none(row.get("epoch", ""))
            values.append((int(epoch) if epoch is not None else idx, value))
        if values:
            best_epoch, best_value = max(values, key=lambda item: item[1])
            min_epoch, min_value = min(values, key=lambda item: item[1])
            best[key] = {
                "max_epoch": best_epoch,
                "max_value": best_value,
                "min_epoch": min_epoch,
                "min_value": min_value,
            }
    return {
        "metrics_csv": str(metrics_csv),
        "rows": len(rows),
        "trial": rows[0].get("trial", ""),
        "epoch_start": rows[0].get("epoch", ""),
        "epoch_end": rows[-1].get("epoch", ""),
        "best": best,
    }


def plot_training_curves(metrics_csv: Path, output_dir: Path, prefix: str) -> Dict[str, Path]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-scensesense")
    warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _load_rows(metrics_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_groups = [
        (
            ["train_loss", "val_loss"],
            "Training and Validation Loss",
            "Loss",
        ),
        (
            ["miou", "vehicle_iou", "person_iou", "pixel_accuracy", "selection_score"],
            "Segmentation / Selection Metrics",
            "Score",
        ),
        (
            ["object_loss", "center_loss", "loc_loss", "dim_loss", "yaw_loss"],
            "Localization Head Losses",
            "Loss",
        ),
        (
            ["parked_loss", "radar_support_loss", "gt_objects"],
            "Auxiliary Localization Signals",
            "Value",
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.0), constrained_layout=True)
    for ax, (keys, title, ylabel) in zip(axes.ravel(), metric_groups):
        _plot_group(ax, rows, _available(rows, keys), title, ylabel)
    fig.suptitle(f"Fusion Training Curves: {rows[0].get('trial', metrics_csv.stem)}", fontsize=14)

    png_path = output_dir / f"{prefix}_training_curves.png"
    pdf_path = output_dir / f"{prefix}_training_curves.pdf"
    summary_path = output_dir / f"{prefix}_training_curves_summary.json"
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)

    summary = _summary(rows, metrics_csv)
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    return {"png": png_path, "pdf": pdf_path, "summary": summary_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot fusion SEG/localization training curves from a metrics CSV.")
    parser.add_argument("metrics_csv", help="Path to a train_fusion metrics CSV.")
    parser.add_argument("--output-dir", default="", help="Directory for plots. Defaults to sibling figures/ folder.")
    parser.add_argument("--prefix", default="", help="Output filename prefix. Defaults to the metrics CSV stem.")
    args = parser.parse_args()

    metrics_csv = Path(args.metrics_csv).expanduser().resolve()
    if not metrics_csv.exists():
        raise FileNotFoundError(metrics_csv)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else metrics_csv.parent.parent / "figures"
    prefix = args.prefix.strip() or metrics_csv.stem
    paths = plot_training_curves(metrics_csv, output_dir, prefix)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
