#!/usr/bin/env python3
"""Validation selection for the Route B AE families under the frozen selected decoder.

Feasibility reuses the registered ``select_route_b_pilot_v1.GUARDS`` and
``feasibility()`` unchanged. Only the *ranking* differs, because this task
registers its own rule:

  1. highest mean(vehicle_f1, person_f1)
  2. lower mean(vehicle_xy_mae_m, person_xy_mae_m)
  3. lower duplicate_fp_per_frame
  4. earlier epoch (final tie-break only)

Both the candidate and the baseline are scored through the *same* frozen decoder
(production decoder plus the selected vehicle-only predicted-world NMS radius),
so the guard deltas compare like with like. Segmentation IoU/mIoU come from the
evaluator's own confusion matrix and are unaffected by object-level suppression,
so they are carried through from ``evaluator_metrics.json`` unchanged.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
PILOT_ROOT = HERE.parent
PKG_ROOT = PILOT_ROOT.parent
for _p in (str(HERE), str(PKG_ROOT), str(PKG_ROOT.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import route_b_postprocess_v1 as pp  # noqa: E402
from object_head_pilot_v1.select_route_b_pilot_v1 import GUARDS, feasibility  # noqa: E402

SELECTION_RULE = [
    "1. highest mean(vehicle_f1, person_f1)",
    "2. lower mean(vehicle_xy_mae_m, person_xy_mae_m)",
    "3. lower duplicate_fp_per_frame",
    "4. earlier epoch (final tie-break only)",
]


def _subset_record(s: Dict[str, Any]) -> Dict[str, Any]:
    frames = max(1, int(s["frames"]))
    return {
        "frames": frames,
        "vehicle_precision": s["vehicle_precision"],
        "vehicle_recall": s["vehicle_recall"],
        "vehicle_f1": s["vehicle_f1"],
        "person_precision": s["person_precision"],
        "person_recall": s["person_recall"],
        "person_f1": s["person_f1"],
        "overall_precision": s["overall_precision"],
        "overall_recall": s["overall_recall"],
        "overall_f1": s["overall_f1"],
        "mean_f1": 0.5 * (s["vehicle_f1"] + s["person_f1"]),
        "vehicle_xy_mae_m": s["vehicle_xy_mae_m"],
        "person_xy_mae_m": s["person_xy_mae_m"],
        "overall_xy_mae_m": s["overall_xy_mae_m"],
        "mean_xy_mae_m": 0.5 * (s["vehicle_xy_mae_m"] + s["person_xy_mae_m"]),
        "dimension_mae_m": s["overall_dimension_mae_m"],
        "dimension_mae_m_covered": s.get("dimension_mae_m_covered"),
        "dimension_coverage": s.get("dimension_coverage"),
        "centroid_2d_error_px": s["overall_centroid_2d_error_px"],
        "fp_per_frame": s["overall_fp"] / frames,
        "duplicate_fp": s["overall_duplicate_fp"],
        "duplicate_fp_per_frame": s["overall_duplicate_fp"] / frames,
        "duplicate_fp_fraction": s["overall_duplicate_fp_fraction"],
        "non_duplicate_fp_per_frame": (s["overall_fp"] - s["overall_duplicate_fp"]) / frames,
        "vehicle_duplicate_fp_fraction": s["vehicle_duplicate_fp_fraction"],
        "person_duplicate_fp_fraction": s["person_duplicate_fp_fraction"],
        "total_fp": s["overall_fp"],
        "overall_tp": s["overall_tp"],
        "overall_fn": s["overall_fn"],
    }


def candidate_record(
    eval_dir: Path,
    experiment_dir: Path,
    radius_m: float,
    split: str = "val",
    epoch: Optional[int] = None,
    split_ids=None,
    collision_ids=None,
) -> Dict[str, Any]:
    """One decoded checkpoint, scored raw (control decoder) and postprocessed."""
    derived = json.loads((eval_dir / "derived_metrics.json").read_text(encoding="utf-8"))
    metrics = json.loads((eval_dir / "evaluator_metrics.json").read_text(encoding="utf-8"))
    if split_ids is None or collision_ids is None:
        split_ids, collision_ids = pp.split_and_collision_ids(experiment_dir, split)

    post = pp.evaluate_radius(eval_dir / "detections.csv", radius_m, split_ids, collision_ids)

    record: Dict[str, Any] = {
        "tag": eval_dir.name,
        "epoch": epoch,
        "checkpoint": derived["checkpoint"],
        # Segmentation is decoder-invariant: object NMS cannot change the pixel confusion matrix.
        "vehicle_iou": metrics["vehicle_iou"],
        "person_iou": metrics["person_iou"],
        "miou": metrics["miou"],
        "ae_bottleneck": metrics.get("ae_bottleneck", 0),
        "raw": {
            "primary": _subset_record(derived["primary"]),
            "collision_window_excluded": _subset_record(derived["collision_window_excluded"]),
        },
        "postprocessed": {
            "primary": _subset_record(post["primary"]),
            "collision_window_excluded": _subset_record(post["collision_window_excluded"]),
        },
        "nms": post["nms"],
        "collision_window_frames_excluded": derived["collision_window_frames_excluded"],
    }
    return record


def guard_view(record: Dict[str, Any], decoder: str = "postprocessed") -> Dict[str, Any]:
    """Flatten a record into the field names select_route_b_pilot_v1.feasibility expects."""
    primary = record[decoder]["primary"]
    return {
        "vehicle_recall": primary["vehicle_recall"],
        "person_recall": primary["person_recall"],
        "vehicle_xy_mae_m": primary["vehicle_xy_mae_m"],
        "person_xy_mae_m": primary["person_xy_mae_m"],
        "vehicle_iou": record["vehicle_iou"],
        "person_iou": record["person_iou"],
        "dimension_mae_m": primary["dimension_mae_m"],
    }


def _rank_key(record: Dict[str, Any], decoder: str = "postprocessed") -> Tuple[float, float, float, int]:
    p = record[decoder]["primary"]
    return (-p["mean_f1"], p["mean_xy_mae_m"], p["duplicate_fp_per_frame"], int(record["epoch"]))


def select_family(
    records: List[Dict[str, Any]],
    baseline: Dict[str, Any],
    decoder: str = "postprocessed",
) -> Dict[str, Any]:
    """Registered selection over the decoded checkpoints of one family."""
    base_view = guard_view(baseline, decoder)
    for record in records:
        # dimension_mae_m can be NaN when reassigned pairs have no recorded predicted size;
        # substitute the coverage-restricted mean so the guard is evaluated on real numbers.
        view = guard_view(record, decoder)
        if math.isnan(float(view["dimension_mae_m"])):
            covered = record[decoder]["primary"].get("dimension_mae_m_covered")
            if covered is not None and not math.isnan(float(covered)):
                view["dimension_mae_m"] = float(covered)
                record["dimension_guard_used_covered_mean"] = True
        record["feasibility"] = feasibility(view, base_view)

    feasible = [r for r in records if r["feasibility"]["feasible"]]
    if feasible:
        selected = min(feasible, key=lambda r: _rank_key(r, decoder))
        status = "SELECTED"
    elif records:
        # Highest mean F1 retained as a diagnostic candidate; never called deployment-ready.
        selected = min(records, key=lambda r: (-r[decoder]["primary"]["mean_f1"], int(r["epoch"])))
        status = "VALIDATION_GATE_FAILED"
    else:
        selected = None
        status = "NO_DECODED_CHECKPOINT"
    return {
        "status": status,
        "selection_rule": SELECTION_RULE,
        "decoder_used": decoder,
        "guards": GUARDS,
        "guards_source": "select_route_b_pilot_v1.GUARDS (unchanged)",
        "baseline_tag": baseline["tag"],
        "epochs_decoded": [r["epoch"] for r in records],
        "feasible_epochs": [r["epoch"] for r in feasible],
        "infeasible_epochs": {
            str(r["epoch"]): r["feasibility"]["failed"] for r in records if not r["feasibility"]["feasible"]
        },
        "selected": selected,
        "all_decoded": records,
    }
