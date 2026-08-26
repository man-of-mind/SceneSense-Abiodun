#!/usr/bin/env python3
"""Route B decoder post-processing: optional vehicle-only predicted-world NMS.

The production decoder (score 0.20, top-k 120, image NMS 2 px, 3.0 m class-aware
match, 40 m GT eligibility) is unchanged. This module adds ONE optional stage on
top of it - suppression of vehicle predictions that sit within ``radius_m`` of a
strictly higher-scoring vehicle prediction in predicted world XY - and rescores
from the evaluator's retained per-detection rows.

Why rescoring offline is exact
------------------------------
``detections.csv`` retains, per frame, every prediction (the ``tp`` and ``fp``
rows carry ``pred_world_x/y``, ``score`` and the predicted class) and every
eligible ground-truth object (the ``tp`` and ``fn`` rows carry ``gt_world_x/y``
and the GT class). The production matcher
(``object_targets.greedy_match_predictions``) is greedy over candidate pairs
sorted by distance, class-aware, with a 3.0 m cap - a pure function of those
world coordinates and classes. So the prediction set, the GT set and the matcher
can all be reconstructed exactly, and ``verify_parity`` proves it by rescoring
the *unsuppressed* rows and comparing against the recorded metrics.

No ground truth enters the suppression decision: NMS ranks by predicted score
and measures distance between predicted positions only. Person predictions are
never touched.

Metric definitions are not redefined here. The registered ``summarize`` from
``evaluate_route_b_checkpoint_v1`` is imported and used verbatim, so
duplicate-FP/frame, recall, F1, XY MAE and the collision-window-excluded
sensitivity subset are computed by exactly the same code that produced the
control numbers.

Predicted 3D sizes are recorded only for predictions the *original* matcher made
into a true positive, so a prediction that was a false positive and becomes a
true positive after suppression has no recorded size. Those pairs are counted and
reported as ``dimension_coverage`` rather than silently averaged; dimension MAE is
not one of the selection gates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

HERE = Path(__file__).resolve().parent
PILOT_ROOT = HERE.parent
PKG_ROOT = PILOT_ROOT.parent
for _p in (str(PKG_ROOT), str(PKG_ROOT.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from object_head_pilot_v1.evaluate_route_b_checkpoint_v1 import (  # noqa: E402
    FIXED_DECODER,
    load_rows,
    summarize,
)

MATCH_DISTANCE_M = float(FIXED_DECODER["match_distance_m"])
DUPLICATE_RADIUS_M = 3.0
NMS_CLASSES = ("vehicle",)

# Metrics that must reproduce exactly when rescoring the unsuppressed rows.
PARITY_KEYS = (
    "overall_tp", "overall_fp", "overall_fn",
    "vehicle_tp", "vehicle_fp", "vehicle_fn",
    "person_tp", "person_fp", "person_fn",
    "overall_duplicate_fp", "vehicle_duplicate_fp", "person_duplicate_fp",
    "overall_recall", "vehicle_recall", "person_recall",
    "overall_xy_mae_m", "vehicle_xy_mae_m", "person_xy_mae_m",
    "overall_f1", "vehicle_f1", "person_f1",
)


def _f(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _has(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


# ---------------------------------------------------------------------------
# reconstruct the decoder's prediction and ground-truth sets
# ---------------------------------------------------------------------------

def build_frames(rows: Iterable[Dict[str, str]]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Split the retained detection rows back into per-frame predictions and GT."""
    frames: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: {"preds": [], "gts": []})
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        status = str(row.get("match_status", ""))
        frame = frames[sample_id]
        if status in {"tp", "fp"}:
            frame["preds"].append({
                "class_name": str(row.get("pred_class_name") or row.get("class_name", "")),
                "world_x": _f(row.get("pred_world_x")),
                "world_y": _f(row.get("pred_world_y")),
                "score": _f(row.get("score")),
                # Predicted geometry, present only on rows the original matcher made a tp.
                "size_x": _f(row.get("pred_size_x")),
                "size_y": _f(row.get("pred_size_y")),
                "size_z": _f(row.get("pred_size_z")),
                "has_sizes": all(_has(row.get(k)) for k in ("pred_size_x", "pred_size_y", "pred_size_z")),
                "bbox": {k: row.get(k, "") for k in (
                    "pred_bbox_x0", "pred_bbox_y0", "pred_bbox_x1", "pred_bbox_y1",
                    "input_w", "input_h", "orig_w", "orig_h",
                )},
                "parked_correct": row.get("parked_correct", ""),
                "yaw_error_deg": row.get("yaw_error_deg", ""),
                "source_status": status,
            })
        if status in {"tp", "fn"}:
            frame["gts"].append({
                "class_name": str(row.get("gt_class_name") or row.get("class_name", "")),
                "world_x": _f(row.get("gt_world_x")),
                "world_y": _f(row.get("gt_world_y")),
                "size_x": _f(row.get("gt_size_x")),
                "size_y": _f(row.get("gt_size_y")),
                "size_z": _f(row.get("gt_size_z")),
                "gt_center_x": row.get("gt_center_x", ""),
                "gt_center_y": row.get("gt_center_y", ""),
                "gt_bbox_w": row.get("gt_bbox_w", ""),
                "gt_bbox_h": row.get("gt_bbox_h", ""),
                "split": row.get("split", ""),
                "frame_id": row.get("frame_id", ""),
                "traffic_light_id": row.get("traffic_light_id", ""),
            })
    return frames


def frame_context(frames: Dict[str, Dict[str, List[Dict[str, Any]]]], rows: Iterable[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    """Per-frame split / frame_id / traffic_light_id, carried through unchanged."""
    ctx: Dict[str, Dict[str, str]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if sample_id not in ctx:
            ctx[sample_id] = {
                "split": row.get("split", ""),
                "frame_id": row.get("frame_id", ""),
                "traffic_light_id": row.get("traffic_light_id", ""),
            }
    return ctx


# ---------------------------------------------------------------------------
# the one new decoder stage
# ---------------------------------------------------------------------------

def vehicle_world_nms(preds: Sequence[Dict[str, Any]], radius_m: float) -> List[Dict[str, Any]]:
    """Suppress a vehicle prediction inside ``radius_m`` of a kept, higher-scoring vehicle.

    Ranking is by predicted score, distance is between predicted world positions.
    Ground truth never participates. Non-vehicle predictions pass through
    untouched and keep their relative order.
    """
    if radius_m <= 0.0:
        return list(preds)
    targets = [(i, p) for i, p in enumerate(preds) if p["class_name"] in NMS_CLASSES]
    # Descending score; index breaks exact score ties deterministically.
    targets.sort(key=lambda item: (-item[1]["score"], item[0]))
    suppressed: Set[int] = set()
    kept_idx: List[int] = []
    for idx, pred in targets:
        if idx in suppressed:
            continue
        kept_idx.append(idx)
        for other_idx, other in targets:
            if other_idx == idx or other_idx in suppressed:
                continue
            if math.hypot(pred["world_x"] - other["world_x"], pred["world_y"] - other["world_y"]) <= radius_m:
                suppressed.add(other_idx)
    return [p for i, p in enumerate(preds) if i not in suppressed]


def greedy_match(
    preds: Sequence[Dict[str, Any]],
    gts: Sequence[Dict[str, Any]],
    max_distance_m: float = MATCH_DISTANCE_M,
) -> List[Tuple[int, int, float]]:
    """Reimplementation of object_targets.greedy_match_predictions (class-aware)."""
    candidates: List[Tuple[float, int, int]] = []
    for pred_idx, pred in enumerate(preds):
        for gt_idx, gt in enumerate(gts):
            if pred["class_name"] != gt["class_name"]:
                continue
            dist = math.hypot(pred["world_x"] - gt["world_x"], pred["world_y"] - gt["world_y"])
            if dist <= max_distance_m:
                candidates.append((dist, pred_idx, gt_idx))
    candidates.sort(key=lambda item: item[0])
    used_pred: Set[int] = set()
    used_gt: Set[int] = set()
    matches: List[Tuple[int, int, float]] = []
    for dist, pred_idx, gt_idx in candidates:
        if pred_idx in used_pred or gt_idx in used_gt:
            continue
        used_pred.add(pred_idx)
        used_gt.add(gt_idx)
        matches.append((pred_idx, gt_idx, dist))
    return matches


def rescore_rows(
    frames: Dict[str, Dict[str, List[Dict[str, Any]]]],
    ctx: Dict[str, Dict[str, str]],
    radius_m: float,
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """Apply the NMS stage, rematch, and re-emit rows in the detections.csv schema."""
    out: List[Dict[str, str]] = []
    stats = {"suppressed": 0, "predictions_before": 0, "predictions_after": 0,
             "reassigned_tp": 0, "tp_without_pred_sizes": 0}
    for sample_id, frame in frames.items():
        base = ctx.get(sample_id, {})
        preds_all = frame["preds"]
        gts = frame["gts"]
        stats["predictions_before"] += len(preds_all)
        preds = vehicle_world_nms(preds_all, radius_m)
        stats["predictions_after"] += len(preds)
        stats["suppressed"] += len(preds_all) - len(preds)

        matches = greedy_match(preds, gts)
        matched_pred = {p for p, _, _ in matches}
        matched_gt = {g for _, g, _ in matches}

        for pred_idx, gt_idx, dist in matches:
            pred, gt = preds[pred_idx], gts[gt_idx]
            if pred["source_status"] != "tp":
                stats["reassigned_tp"] += 1
            row: Dict[str, str] = {
                "split": base.get("split", ""),
                "sample_id": sample_id,
                "frame_id": base.get("frame_id", ""),
                "traffic_light_id": base.get("traffic_light_id", ""),
                "match_status": "tp",
                "class_name": gt["class_name"],
                "pred_class_name": pred["class_name"],
                "gt_class_name": gt["class_name"],
                "score": repr(pred["score"]),
                "global_xy_error_m": repr(dist),
                "yaw_error_deg": pred.get("yaw_error_deg", ""),
                "parked_correct": pred.get("parked_correct", ""),
                "pred_world_x": repr(pred["world_x"]),
                "pred_world_y": repr(pred["world_y"]),
                "gt_world_x": repr(gt["world_x"]),
                "gt_world_y": repr(gt["world_y"]),
                "gt_size_x": repr(gt["size_x"]),
                "gt_size_y": repr(gt["size_y"]),
                "gt_size_z": repr(gt["size_z"]),
                "gt_center_x": gt.get("gt_center_x", ""),
                "gt_center_y": gt.get("gt_center_y", ""),
                "gt_bbox_w": gt.get("gt_bbox_w", ""),
                "gt_bbox_h": gt.get("gt_bbox_h", ""),
            }
            row.update({k: (v if v is not None else "") for k, v in pred["bbox"].items()})
            if pred["has_sizes"]:
                row["pred_size_x"] = repr(pred["size_x"])
                row["pred_size_y"] = repr(pred["size_y"])
                row["pred_size_z"] = repr(pred["size_z"])
                row["dimension_mae_m"] = repr(
                    (abs(pred["size_x"] - gt["size_x"])
                     + abs(pred["size_y"] - gt["size_y"])
                     + abs(pred["size_z"] - gt["size_z"])) / 3.0
                )
            else:
                # Predicted sizes were never recorded for this prediction.
                stats["tp_without_pred_sizes"] += 1
                for key in ("pred_size_x", "pred_size_y", "pred_size_z", "dimension_mae_m"):
                    row[key] = ""
            out.append(row)

        for pred_idx, pred in enumerate(preds):
            if pred_idx in matched_pred:
                continue
            out.append({
                "split": base.get("split", ""),
                "sample_id": sample_id,
                "match_status": "fp",
                "class_name": pred["class_name"],
                "pred_class_name": pred["class_name"],
                "score": repr(pred["score"]),
                "pred_world_x": repr(pred["world_x"]),
                "pred_world_y": repr(pred["world_y"]),
            })
        for gt_idx, gt in enumerate(gts):
            if gt_idx in matched_gt:
                continue
            out.append({
                "split": base.get("split", ""),
                "sample_id": sample_id,
                "match_status": "fn",
                "class_name": gt["class_name"],
                "gt_class_name": gt["class_name"],
                "gt_world_x": repr(gt["world_x"]),
                "gt_world_y": repr(gt["world_y"]),
            })
    return out, stats


# ---------------------------------------------------------------------------
# scoring entry points
# ---------------------------------------------------------------------------

def split_and_collision_ids(experiment_dir: Path, split: str) -> Tuple[Set[str], Set[str]]:
    manifest_rows = load_rows(experiment_dir / "dataset" / "manifest.csv")
    split_ids = {str(r["sample_id"]) for r in manifest_rows if r.get("split") == split}
    collision_ids = {
        str(r["sample_id"])
        for r in load_rows(experiment_dir / "provenance" / "collision_window_samples.csv")
        if r.get("split") == split and str(r.get("retained_in_dataset")) == "1"
    }
    return split_ids, collision_ids


def score(
    rows: List[Dict[str, str]],
    split_ids: Set[str],
    collision_ids: Set[str],
) -> Dict[str, Any]:
    """Primary and collision-window-excluded metrics via the registered summarize()."""
    return {
        "primary": summarize(rows, frame_ids=split_ids,
                             duplicate_radius_m=DUPLICATE_RADIUS_M, label="all_val_frames"),
        "collision_window_excluded": summarize(rows, frame_ids=split_ids - collision_ids,
                                               duplicate_radius_m=DUPLICATE_RADIUS_M,
                                               label="collision_window_excluded"),
        "collision_window_frames_excluded": len(collision_ids & split_ids),
    }


def _close(a: float, b: float, tol: float) -> bool:
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    if math.isnan(a) and math.isnan(b):
        return True
    return abs(a - b) <= tol


def verify_parity(
    detections_csv: Path,
    recorded: Dict[str, Any],
    split_ids: Set[str],
    collision_ids: Set[str],
    tol: float = 1e-9,
) -> Dict[str, Any]:
    """Rescore the *unsuppressed* rows and require the recorded metrics to reproduce.

    This is the guard that makes the suppressed variants trustworthy: if the
    reconstructed prediction/GT sets and the reimplemented matcher reproduce the
    control exactly, the only thing the variants change is the suppression.
    """
    rows = load_rows(detections_csv)
    frames = build_frames(rows)
    ctx = frame_context(frames, rows)
    rebuilt, stats = rescore_rows(frames, ctx, radius_m=0.0)
    scored = score(rebuilt, split_ids, collision_ids)
    mismatches = []
    for subset in ("primary", "collision_window_excluded"):
        for key in PARITY_KEYS:
            got, want = scored[subset].get(key), recorded[subset].get(key)
            if got is None or want is None:
                mismatches.append({"subset": subset, "key": key, "got": got, "want": want})
                continue
            if not _close(float(got), float(want), tol):
                mismatches.append({"subset": subset, "key": key, "got": got, "want": want})
    return {
        "status": "PASS" if not mismatches else "FAIL",
        "keys_checked": len(PARITY_KEYS) * 2,
        "mismatches": mismatches,
        "tolerance": tol,
        "rebuild_stats": stats,
        "rescored": scored,
    }


def evaluate_radius(
    detections_csv: Path,
    radius_m: float,
    split_ids: Set[str],
    collision_ids: Set[str],
) -> Dict[str, Any]:
    rows = load_rows(detections_csv)
    frames = build_frames(rows)
    ctx = frame_context(frames, rows)
    rebuilt, stats = rescore_rows(frames, ctx, radius_m=radius_m)
    scored = score(rebuilt, split_ids, collision_ids)
    scored["primary"].update(dimension_mae_covered(rebuilt, split_ids))
    scored["collision_window_excluded"].update(
        dimension_mae_covered(rebuilt, split_ids - collision_ids)
    )
    scored["nms"] = {
        "stage": "vehicle_only_predicted_world_nms",
        "radius_m": radius_m,
        "classes": list(NMS_CLASSES),
        "ranking": "predicted score, descending",
        "ground_truth_used": False,
        **stats,
    }
    return scored


def dimension_mae_covered(rows: Iterable[Dict[str, str]], frame_ids: Set[str]) -> Dict[str, float]:
    """Dimension MAE over the true positives that actually have a recorded predicted size.

    A prediction that was a false positive under the control decoder and becomes a
    true positive after suppression carries no recorded predicted size, so the
    registered ``summarize`` returns NaN for the whole subset. Reporting the mean
    over covered pairs plus the coverage fraction is honest; silently dropping the
    gap would not be. Dimension MAE is a reported metric, never a selection gate.
    """
    vals: List[float] = []
    total = 0
    for row in rows:
        if str(row.get("sample_id", "")) not in frame_ids or row.get("match_status") != "tp":
            continue
        total += 1
        if _has(row.get("dimension_mae_m")):
            vals.append(_f(row["dimension_mae_m"]))
    return {
        "dimension_mae_m_covered": (sum(vals) / len(vals)) if vals else float("nan"),
        "dimension_covered_tp": len(vals),
        "dimension_total_tp": total,
        "dimension_coverage": (len(vals) / total) if total else float("nan"),
    }


def headline(scored: Dict[str, Any], subset: str = "primary") -> Dict[str, float]:
    s = scored[subset]
    frames = max(1, int(s["frames"]))
    return {
        "duplicate_fp_per_frame": s["overall_duplicate_fp"] / frames,
        "overall_recall": s["overall_recall"],
        "vehicle_recall": s["vehicle_recall"],
        "person_recall": s["person_recall"],
        "vehicle_xy_mae_m": s["vehicle_xy_mae_m"],
        "person_xy_mae_m": s["person_xy_mae_m"],
        "mean_xy_mae_m": 0.5 * (s["vehicle_xy_mae_m"] + s["person_xy_mae_m"]),
        "vehicle_f1": s["vehicle_f1"],
        "person_f1": s["person_f1"],
        "mean_f1": 0.5 * (s["vehicle_f1"] + s["person_f1"]),
        "fp_per_frame": s["overall_fp"] / frames,
        "dimension_mae_m": s["overall_dimension_mae_m"],
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--detections", required=True, type=Path)
    parser.add_argument("--recorded-metrics", type=Path,
                        help="derived_metrics.json of the control, for the parity guard")
    parser.add_argument("--radius-m", type=float, default=0.0)
    parser.add_argument("--split", default="val")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    split_ids, collision_ids = split_and_collision_ids(args.experiment_dir.resolve(), args.split)
    if args.recorded_metrics:
        recorded = json.loads(args.recorded_metrics.read_text(encoding="utf-8"))
        parity = verify_parity(args.detections, recorded, split_ids, collision_ids)
        print(json.dumps({"parity": {k: v for k, v in parity.items() if k != "rescored"}}, indent=2))
        if parity["status"] != "PASS":
            return 3
    scored = evaluate_radius(args.detections, args.radius_m, split_ids, collision_ids)
    print(json.dumps({"radius_m": args.radius_m, "headline": headline(scored)}, indent=2, sort_keys=True))
    if args.out:
        args.out.write_text(json.dumps(scored, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
