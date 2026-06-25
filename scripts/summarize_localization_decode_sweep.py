#!/usr/bin/env python3
"""Summarize localization decoder sweeps into one CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


METRIC_KEYS = [
    "learned_object_precision",
    "learned_object_recall",
    "learned_object_f1",
    "learned_global_xy_mae_m",
    "learned_global_xy_rmse_m",
    "learned_vehicle_object_precision",
    "learned_vehicle_object_recall",
    "learned_vehicle_object_f1",
    "learned_vehicle_global_xy_mae_m",
    "learned_person_object_precision",
    "learned_person_object_recall",
    "learned_person_object_f1",
    "learned_person_global_xy_mae_m",
    "learned_object_tp",
    "learned_object_fp",
    "learned_object_fn",
    "miou",
    "vehicle_iou",
    "person_iou",
]


def as_float(value: object) -> object:
    if isinstance(value, (int, float)):
        return value
    try:
        return float(str(value))
    except Exception:
        return value


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    sweep_dir = args.sweep_dir.expanduser().resolve()
    rows: List[Dict[str, object]] = []
    for metrics_path in sorted(sweep_dir.glob("*/metrics/test_fusion_evaluation_metrics.json")):
        run_dir = metrics_path.parents[1]
        config_path = run_dir / "decode_config.json"
        metrics = read_json(metrics_path)
        config = read_json(config_path) if config_path.exists() else {}
        row: Dict[str, object] = {
            "setting": run_dir.name,
            "object_score_threshold": config.get("object_score_threshold", ""),
            "object_nms_radius_px": config.get("object_nms_radius_px", ""),
            "topk_objects": config.get("topk_objects", ""),
            "match_distance_m": metrics.get("match_distance_m", config.get("match_distance_m", "")),
            "samples": metrics.get("samples", ""),
        }
        for key in METRIC_KEYS:
            row[key] = as_float(metrics.get(key, ""))
        rows.append(row)

    if not rows:
        raise SystemExit(f"No evaluation metrics found under {sweep_dir}")

    out = args.out or (sweep_dir / "decode_sweep_summary.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
