#!/usr/bin/env python3
"""Evaluate one checkpoint on the Route B validation view with the fixed decoder.

Wraps the production evaluator (``pole_lraspp_multimodal_fusion.evaluate_fusion``)
without editing it. The decoder is fixed for every checkpoint in this pilot -
score threshold 0.20, top-k 120, image-space NMS radius 2 px, no world
suppression (the production decoder has none), 40 m GT eligibility and a 3.0 m
class-aware match radius - so no epoch, arm, density or class ever gets its own
threshold.

The production evaluator writes ``metrics/<split>_*`` inside the experiment
directory, which would collide across checkpoints, so each run is moved into a
create-only ``eval/<tag>/`` directory.

Derived metrics are computed from the *same* per-detection rows that inference
already produced - the collision-window sensitivity in particular never re-runs
inference, it only drops the recorded collision-window sample ids and recomputes.

Duplicate-FP definition (registered once, applied identically everywhere):
a prediction is a *duplicate* when another prediction of the same class, in the
same frame, with a strictly higher score, lies within ``--duplicate-radius-m``
(default 3.0 m, the match radius) of it in predicted world XY. This uses
predicted positions only - ground truth never enters the duplicate test.
``duplicate_fp_fraction`` is the share of false positives that are duplicates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent

FIXED_DECODER = {
    "object_score_threshold": 0.20,
    "topk_objects": 120,
    "object_nms_radius_px": 2,
    "match_distance_m": 3.0,
    "max_gt_distance_m": 40.0,
    "world_suppression": "none",
}


def _f(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def summarize(
    rows: Iterable[Dict[str, str]],
    *,
    frame_ids: Set[str],
    duplicate_radius_m: float,
    label: str,
) -> Dict[str, Any]:
    """Recompute every reported metric from the evaluator's per-detection rows."""
    rows = [row for row in rows if str(row.get("sample_id", "")) in frame_ids]
    by_frame: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_frame[str(row["sample_id"])].append(row)

    # Duplicate test: predictions only (tp + fp rows carry pred_world_x/y and score).
    duplicate_flags: Dict[int, bool] = {}
    predictions_total = 0
    duplicates_total = 0
    for _, frame_rows in by_frame.items():
        preds = [row for row in frame_rows if str(row.get("match_status")) in {"tp", "fp"}]
        predictions_total += len(preds)
        for row in preds:
            cls = str(row.get("pred_class_name") or row.get("class_name", ""))
            px, py, score = _f(row.get("pred_world_x")), _f(row.get("pred_world_y")), _f(row.get("score"))
            is_duplicate = False
            for other in preds:
                if other is row:
                    continue
                other_cls = str(other.get("pred_class_name") or other.get("class_name", ""))
                if other_cls != cls:
                    continue
                if _f(other.get("score")) <= score:
                    continue
                if math.hypot(px - _f(other.get("pred_world_x")), py - _f(other.get("pred_world_y"))) <= duplicate_radius_m:
                    is_duplicate = True
                    break
            duplicate_flags[id(row)] = is_duplicate
            duplicates_total += int(is_duplicate)

    out: Dict[str, Any] = {
        "subset": label,
        "frames": len(frame_ids),
        "frames_with_detections_or_gt": len(by_frame),
        "duplicate_radius_m": duplicate_radius_m,
    }

    classes = sorted({str(row.get("class_name", "")) for row in rows if row.get("class_name")})
    totals = {"tp": 0, "fp": 0, "fn": 0, "dup_fp": 0}
    xy_all: List[float] = []
    dim_all: List[float] = []
    centroid_all: List[float] = []
    for cls in classes + ["__all__"]:
        subset = rows if cls == "__all__" else [row for row in rows if str(row.get("class_name")) == cls]
        tp = sum(1 for row in subset if row.get("match_status") == "tp")
        fp = sum(1 for row in subset if row.get("match_status") == "fp")
        fn = sum(1 for row in subset if row.get("match_status") == "fn")
        dup_fp = sum(
            1 for row in subset
            if row.get("match_status") == "fp" and duplicate_flags.get(id(row), False)
        )
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2.0 * precision * recall / max(1e-9, precision + recall)
        xy = [_f(row.get("global_xy_error_m")) for row in subset if row.get("match_status") == "tp"]
        dim = [_f(row.get("dimension_mae_m")) for row in subset if row.get("match_status") == "tp"]
        centroid: List[float] = []
        for row in subset:
            if row.get("match_status") != "tp":
                continue
            x0, x1 = _f(row.get("pred_bbox_x0")), _f(row.get("pred_bbox_x1"))
            y0, y1 = _f(row.get("pred_bbox_y0")), _f(row.get("pred_bbox_y1"))
            iw, ih = _f(row.get("input_w")), _f(row.get("input_h"))
            ow, oh = _f(row.get("orig_w")), _f(row.get("orig_h"))
            gx, gy = _f(row.get("gt_center_x")), _f(row.get("gt_center_y"))
            if any(math.isnan(v) for v in (x0, x1, y0, y1, iw, ih, ow, oh, gx, gy)) or iw <= 0 or ih <= 0:
                continue
            # Predicted 2D centre is in input pixels; GT centre is in original pixels.
            centroid.append(math.hypot(0.5 * (x0 + x1) * ow / iw - gx, 0.5 * (y0 + y1) * oh / ih - gy))
        prefix = "overall" if cls == "__all__" else cls
        out[f"{prefix}_tp"] = tp
        out[f"{prefix}_fp"] = fp
        out[f"{prefix}_fn"] = fn
        out[f"{prefix}_precision"] = precision
        out[f"{prefix}_recall"] = recall
        out[f"{prefix}_f1"] = f1
        out[f"{prefix}_fp_per_frame"] = fp / max(1, len(frame_ids))
        out[f"{prefix}_duplicate_fp"] = dup_fp
        out[f"{prefix}_duplicate_fp_fraction"] = dup_fp / max(1, fp)
        out[f"{prefix}_xy_mae_m"] = sum(xy) / len(xy) if xy else float("nan")
        out[f"{prefix}_dimension_mae_m"] = sum(dim) / len(dim) if dim else float("nan")
        out[f"{prefix}_centroid_2d_error_px"] = sum(centroid) / len(centroid) if centroid else float("nan")
        if cls != "__all__":
            for key, value in (("tp", tp), ("fp", fp), ("fn", fn), ("dup_fp", dup_fp)):
                totals[key] += value
            xy_all.extend(xy)
            dim_all.extend(dim)
            centroid_all.extend(centroid)
    # Ground-truth-free cross-check: duplicate share over *all* predictions.
    out["all_predictions"] = predictions_total
    out["all_prediction_duplicates"] = duplicates_total
    out["all_prediction_duplicate_fraction"] = duplicates_total / max(1, predictions_total)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--duplicate-radius-m", type=float, default=3.0)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)

    exp_dir = args.experiment_dir.resolve()
    out_dir = exp_dir / "eval" / args.tag
    if out_dir.exists():
        print(f"refusing to overwrite an existing evaluation: {out_dir}", file=sys.stderr)
        return 2
    checkpoint = args.checkpoint.resolve(strict=True)

    command = [
        args.python, "-m", "pole_lraspp_multimodal_fusion.evaluate_fusion",
        "--config", str(args.config),
        "--experiment-dir", str(exp_dir),
        "--checkpoint", str(checkpoint),
        "--split", args.split,
        "--require-cuda",
        "--object-score-threshold", str(FIXED_DECODER["object_score_threshold"]),
        "--topk-objects", str(FIXED_DECODER["topk_objects"]),
        "--object-nms-radius-px", str(FIXED_DECODER["object_nms_radius_px"]),
        "--match-distance-m", str(FIXED_DECODER["match_distance_m"]),
        "--max-gt-distance-m", str(FIXED_DECODER["max_gt_distance_m"]),
    ]
    print(" ".join(command), flush=True)
    result = subprocess.run(command, cwd=str(PKG_ROOT))
    if result.returncode != 0:
        return result.returncode

    out_dir.mkdir(parents=True)
    metrics_dir = exp_dir / "metrics"
    produced = {
        "evaluator_metrics.json": metrics_dir / f"{args.split}_fusion_evaluation_metrics.json",
        "detections.csv": metrics_dir / f"{args.split}_learned_object_metrics.csv",
    }
    for name, path in produced.items():
        shutil.move(str(path), str(out_dir / name))
    figure = exp_dir / "figures" / f"{args.split}_fusion_confusion.png"
    if figure.exists():
        shutil.move(str(figure), str(out_dir / "confusion.png"))

    manifest_rows = load_rows(exp_dir / "dataset" / "manifest.csv")
    split_ids = {str(row["sample_id"]) for row in manifest_rows if row.get("split") == args.split}
    collision_ids = {
        str(row["sample_id"])
        for row in load_rows(exp_dir / "provenance" / "collision_window_samples.csv")
        if row.get("split") == args.split and str(row.get("retained_in_dataset")) == "1"
    }
    detections = load_rows(out_dir / "detections.csv")

    derived = {
        "tag": args.tag,
        "checkpoint": str(checkpoint),
        "split": args.split,
        "fixed_decoder": FIXED_DECODER,
        "duplicate_fp_definition": (
            "a prediction is a duplicate when another prediction of the same class, in the "
            "same frame, with a strictly higher score, lies within duplicate_radius_m of it "
            "in predicted world XY; predicted positions only, no ground truth"
        ),
        "primary": summarize(
            detections, frame_ids=split_ids,
            duplicate_radius_m=args.duplicate_radius_m, label="all_val_frames",
        ),
        "collision_window_excluded": summarize(
            detections, frame_ids=split_ids - collision_ids,
            duplicate_radius_m=args.duplicate_radius_m, label="collision_window_excluded",
        ),
        "collision_window_frames_excluded": len(collision_ids & split_ids),
        "segmentation_note": (
            "segmentation IoU/mIoU come from the evaluator's aggregate confusion matrix over "
            "all evaluated frames and are therefore reported for the primary subset only; "
            "a per-frame confusion breakdown would require a second inference pass"
        ),
    }
    (out_dir / "derived_metrics.json").write_text(
        json.dumps(derived, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(derived["primary"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
