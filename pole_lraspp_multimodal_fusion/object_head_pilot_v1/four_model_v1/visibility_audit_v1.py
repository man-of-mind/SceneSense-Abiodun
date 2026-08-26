#!/usr/bin/env python3
"""Read-only visibility-aware <=40 m evaluation and evaluator-parity audit (W10275).

Nothing here trains, tunes or touches the locked test split. It reads the retained
Route B validation predictions (``eval/<tag>/detections.csv``) and the recorded GT
table (``dataset/object_boxes.csv``) and rescores under three GT-eligibility rules.

Eligibility rules
-----------------
A  EXISTING: exactly ``object_targets.valid_localization_objects`` -
   ``gt_source == "actor"``, ``gt_bbox_area_px >= min_gt_area_px`` (12.0),
   ``gt_distance_m <= 40``, and the projected CENTRE inside the original image.
   Rule A has no ignore concept: GT inside 40 m that fails it disappears, so a
   prediction landing on such an object is charged as a false positive.

B  CAMERA-FRUSTUM-VISIBLE: within 40 m, positive camera depth, and the projected
   box clipped to the image retains at least ``min_gt_area_px`` of area. Objects
   within 40 m that fail become IGNORE, not false negatives.

C  SENSOR-OBSERVABLE: camera-visible (B) OR recorded ``radar_support_points`` > 0,
   i.e. the fused-sensor observability contract the model is actually trained on.

The pixel-area floor is NOT invented here - it is the repository's own
``object_heads.min_gt_area_px`` (12.0), reused unchanged. No threshold is chosen
to improve recall, and no model score or outcome ever enters the visibility test.

Ignore semantics
----------------
1. predictions are greedily matched to EVALUABLE GT (same 3.0 m class-aware rule)
2. still-unmatched predictions are matched to IGNORE GT and then dropped
3. only predictions matched to neither count as false positives
4. unmatched EVALUABLE GT are false negatives; IGNORE GT never are
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

HERE = Path(__file__).resolve().parent
PILOT_ROOT = HERE.parent
PKG_ROOT = PILOT_ROOT.parent
ABIODUN = PKG_ROOT.parent
for _p in (str(HERE), str(PKG_ROOT), str(ABIODUN)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MATCH_DISTANCE_M = 3.0
MAX_GT_DISTANCE_M = 40.0
MIN_GT_AREA_PX = 12.0          # object_heads.min_gt_area_px, reused unchanged
CLASSES = ("vehicle", "person")
ROUTE_FILE = ABIODUN / "data_collection/routes/town10hd_opt_route_b_full_map_loop_v1.json"

DIST_BANDS = ((0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 40.0))
# Projected-height bands in original-image pixels, reported as recorded, not tuned.
SIZE_BANDS = ((0.0, 16.0), (16.0, 32.0), (32.0, 64.0), (64.0, 128.0), (128.0, 1e9))


def _f(row: Dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# GT classification
# ---------------------------------------------------------------------------

def classify_gt(row: Dict[str, str], width: float, height: float) -> Dict[str, Any]:
    """Geometry and sensor-support facts for one recorded GT object, plus A/B/C class."""
    label = str(row.get("label", ""))
    dist = _f(row, "gt_distance_m")
    depth = _f(row, "gt_depth_m")
    area_recorded = _f(row, "gt_bbox_area_px")
    bx, by = _f(row, "gt_bbox_x"), _f(row, "gt_bbox_y")
    bw, bh = _f(row, "gt_bbox_w"), _f(row, "gt_bbox_h")
    cx, cy = _f(row, "gt_center_x"), _f(row, "gt_center_y")
    radar = _f(row, "radar_support_points", 0.0)

    # Clip the projected box to the image; truncation is the share of area lost.
    x0, y0 = max(0.0, bx), max(0.0, by)
    x1, y1 = min(width, bx + bw), min(height, by + bh)
    clipped_w, clipped_h = max(0.0, x1 - x0), max(0.0, y1 - y0)
    clipped_area = clipped_w * clipped_h
    full_area = max(0.0, bw) * max(0.0, bh)
    truncation = (1.0 - clipped_area / full_area) if full_area > 0 else 1.0
    centre_in_image = (0.0 <= cx < width) and (0.0 <= cy < height)
    intersects = clipped_area > 0.0
    depth_ok = depth > 0.0 if not math.isnan(depth) else False
    radar_ok = (not math.isnan(radar)) and radar > 0.0

    eligible_class = label in CLASSES and str(row.get("gt_source", "")) == "actor"
    has_pose = str(row.get("object_sensor_x", "")) != "" and str(row.get("object_world_x", "")) != ""
    in_range = dist <= MAX_GT_DISTANCE_M

    # A: the existing rule, reproduced exactly.
    rule_a = (eligible_class and has_pose and area_recorded >= MIN_GT_AREA_PX
              and in_range and centre_in_image)
    # B: camera frustum, on the CLIPPED support.
    visible_b = (eligible_class and has_pose and in_range and depth_ok
                 and intersects and clipped_area >= MIN_GT_AREA_PX)
    # C: camera-visible OR recorded radar support.
    visible_c = visible_b or (eligible_class and has_pose and in_range and radar_ok)

    def bucket(is_eval: bool) -> str:
        if not (eligible_class and has_pose):
            return "EXCLUDED_NON_TARGET"
        if not in_range:
            return "OUT_OF_RANGE"
        return "EVALUABLE" if is_eval else "IGNORE"

    return {
        "label": label,
        "sample_id": str(row.get("sample_id", "")),
        "world_x": _f(row, "object_world_x"),
        "world_y": _f(row, "object_world_y"),
        "size_x": max(0.01, _f(row, "gt_size_x_m")),
        "size_y": max(0.01, _f(row, "gt_size_y_m")),
        "size_z": max(0.01, _f(row, "gt_size_z_m")),
        "distance_m": dist,
        "depth_m": depth,
        "area_recorded_px": area_recorded,
        "clipped_area_px": clipped_area,
        "clipped_h_px": clipped_h,
        "proj_h_px": bh,
        "truncation": truncation,
        "centre_in_image": centre_in_image,
        "intersects_image": intersects,
        "depth_positive": depth_ok,
        "radar_support_points": 0.0 if math.isnan(radar) else radar,
        "radar_ok": radar_ok,
        "eligible_class": eligible_class and has_pose,
        "in_range": in_range,
        "A": bucket(rule_a), "B": bucket(visible_b), "C": bucket(visible_c),
    }


# ---------------------------------------------------------------------------
# matching with ignore semantics
# ---------------------------------------------------------------------------

def greedy(preds: Sequence[Dict[str, Any]], gts: Sequence[Dict[str, Any]],
           pred_free: Set[int], gt_free: Set[int]) -> List[Tuple[int, int, float]]:
    cand: List[Tuple[float, int, int]] = []
    for pi in pred_free:
        p = preds[pi]
        for gi in gt_free:
            g = gts[gi]
            if p["class_name"] != g["label"]:
                continue
            d = math.hypot(p["world_x"] - g["world_x"], p["world_y"] - g["world_y"])
            if d <= MATCH_DISTANCE_M:
                cand.append((d, pi, gi))
    cand.sort(key=lambda t: t[0])
    used_p: Set[int] = set()
    used_g: Set[int] = set()
    out: List[Tuple[int, int, float]] = []
    for d, pi, gi in cand:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        out.append((pi, gi, d))
    return out


def score_rule(
    preds_by_frame: Dict[str, List[Dict[str, Any]]],
    gts_by_frame: Dict[str, List[Dict[str, Any]]],
    rule: str,
    frame_ids: Set[str],
    honour_ignore: bool,
) -> Dict[str, Any]:
    """TP / FP / FN with ignore semantics, plus per-slice recall accumulators."""
    agg: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    slices: Dict[str, Dict[str, Dict[str, float]]] = {
        "region": defaultdict(lambda: defaultdict(float)),
        "distance": defaultdict(lambda: defaultdict(float)),
        "size": defaultdict(lambda: defaultdict(float)),
    }
    ignored_preds = 0
    for sid in frame_ids:
        preds = preds_by_frame.get(sid, [])
        allg = gts_by_frame.get(sid, [])
        evaluable = [g for g in allg if g[rule] == "EVALUABLE"]
        ignore = [g for g in allg if g[rule] == "IGNORE"] if honour_ignore else []

        m = greedy(preds, evaluable, set(range(len(preds))), set(range(len(evaluable))))
        matched_p = {pi for pi, _, _ in m}
        matched_g = {gi for _, gi, _ in m}

        if ignore:
            rest = set(range(len(preds))) - matched_p
            mi = greedy(preds, ignore, rest, set(range(len(ignore))))
            ignored_preds += len(mi)
            matched_p |= {pi for pi, _, _ in mi}

        for pi, gi, d in m:
            g = evaluable[gi]
            cls = g["label"]
            agg[cls]["tp"] += 1
            agg[cls]["xy_sum"] += d
            agg["__all__"]["tp"] += 1
            agg["__all__"]["xy_sum"] += d
            _slice_add(slices, g, "tp")
        for gi, g in enumerate(evaluable):
            if gi in matched_g:
                continue
            agg[g["label"]]["fn"] += 1
            agg["__all__"]["fn"] += 1
            _slice_add(slices, g, "fn")
        for pi, p in enumerate(preds):
            if pi in matched_p:
                continue
            agg[p["class_name"]]["fp"] += 1
            agg["__all__"]["fp"] += 1

    out: Dict[str, Any] = {"rule": rule, "frames": len(frame_ids),
                           "ignored_predictions": ignored_preds,
                           "honour_ignore": honour_ignore}
    for cls in list(CLASSES) + ["__all__"]:
        a = agg[cls]
        tp, fp, fn = a["tp"], a["fp"], a["fn"]
        prec = tp / max(1.0, tp + fp)
        rec = tp / max(1.0, tp + fn)
        name = "overall" if cls == "__all__" else cls
        out[f"{name}_tp"] = int(tp)
        out[f"{name}_fp"] = int(fp)
        out[f"{name}_fn"] = int(fn)
        out[f"{name}_precision"] = prec
        out[f"{name}_recall"] = rec
        out[f"{name}_f1"] = 2 * prec * rec / max(1e-9, prec + rec)
        out[f"{name}_xy_mae_m"] = (a["xy_sum"] / tp) if tp else float("nan")
        out[f"{name}_fp_per_frame"] = fp / max(1, len(frame_ids))
    out["slices"] = {
        kind: {k: {"tp": int(v["tp"]), "fn": int(v["fn"]),
                   "recall": v["tp"] / max(1.0, v["tp"] + v["fn"])}
               for k, v in sorted(vals.items())}
        for kind, vals in slices.items()
    }
    return out


def _slice_add(slices, g: Dict[str, Any], key: str) -> None:
    slices["region"][f"{g.get('region', 'unknown')}|{g['label']}"][key] += 1
    d = g["distance_m"]
    for lo, hi in DIST_BANDS:
        if lo <= d < hi:
            slices["distance"][f"{int(lo)}-{int(hi)}m|{g['label']}"][key] += 1
            break
    h = g["clipped_h_px"] if g["clipped_h_px"] > 0 else g["proj_h_px"]
    for lo, hi in SIZE_BANDS:
        if lo <= h < hi:
            tag = f"h{int(lo)}-{int(hi) if hi < 1e8 else 'inf'}px|{g['label']}"
            slices["size"][tag][key] += 1
            break


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_predictions(detections_csv: Path) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in load_csv(detections_csv):
        if str(row.get("match_status")) not in {"tp", "fp"}:
            continue
        out[str(row.get("sample_id", ""))].append({
            "class_name": str(row.get("pred_class_name") or row.get("class_name", "")),
            "world_x": _f(row, "pred_world_x"),
            "world_y": _f(row, "pred_world_y"),
            "score": _f(row, "score"),
        })
    return out


def region_assigner():
    route = json.loads(ROUTE_FILE.read_text(encoding="utf-8"))
    labels = route["route_b_regions"]
    pts = [(float(w["x"]), float(w["y"])) for w in route["intermediate_waypoints"]]

    def assign(x: float, y: float) -> str:
        if math.isnan(x) or math.isnan(y):
            return "unknown"
        best, best_d = "unknown", float("inf")
        for (wx, wy), lab in zip(pts, labels):
            d = math.hypot(x - wx, y - wy)
            if d < best_d:
                best_d, best = d, lab
        return best
    return assign
