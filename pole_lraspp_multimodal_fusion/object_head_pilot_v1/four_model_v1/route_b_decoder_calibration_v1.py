#!/usr/bin/env python3
"""Offline predicted-only decoder calibration over one saved Route B validation pass.

Reads the retained detections of a single inference pass decoded at score
threshold 0.02 (image NMS 2 px, top-k 120, 40 m range gate) and re-scores a
preregistered operating-point grid entirely offline. Inference is never re-run.

Reused unchanged from ``visibility_audit_v1``: the GT eligibility rule A
(``object_targets.valid_localization_objects``), the class-aware greedy matcher
and the distance bands. Reused unchanged from
``evaluate_route_b_checkpoint_v1``: the registered duplicate-FP definition
(same class, same frame, strictly higher score, within 3.0 m in predicted world
XY; predictions only, no ground truth).

Post-processing applied per setting, predicted-only:
  1. per-class score threshold
  2. per-class predicted-world greedy NMS (highest score kept, lower-score
     predictions of the same class within the radius dropped). Radius 0 = off.
No ground-truth-dependent suppression is ever applied.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from visibility_audit_v1 import (  # noqa: E402  - reused unchanged
    CLASSES,
    DIST_BANDS,
    MAX_GT_DISTANCE_M,
    classify_gt,
    load_csv,
)

DUPLICATE_RADIUS_M = 3.0
PRIMARY_MATCH_M = 3.0
RECALL_RADII = (1.0, 2.0, 3.0)

VEHICLE_THRESHOLDS = (0.02, 0.05, 0.10, 0.15, 0.20)
PERSON_THRESHOLDS = (0.02, 0.05, 0.10, 0.15, 0.20)
VEHICLE_WORLD_NMS_M = (0.0, 0.5, 1.0, 2.0)
PERSON_WORLD_NMS_M = (0.0,)  # disabled unless duplicate evidence justifies it


def _f(row: Dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------
# predicted-only post-processing
# --------------------------------------------------------------------------

def world_nms(preds: Sequence[Dict[str, Any]], radius_by_class: Dict[str, float]) -> List[Dict[str, Any]]:
    """Greedy predicted-world NMS, per class. Ground truth never enters."""
    kept: List[Dict[str, Any]] = []
    for p in sorted(preds, key=lambda r: -r["score"]):
        r = radius_by_class.get(p["class_name"], 0.0)
        if r > 0.0 and any(
            k["class_name"] == p["class_name"]
            and math.hypot(k["world_x"] - p["world_x"], k["world_y"] - p["world_y"]) <= r
            for k in kept
        ):
            continue
        kept.append(p)
    return kept


def mark_duplicates(preds: Sequence[Dict[str, Any]], radius_m: float) -> List[bool]:
    flags: List[bool] = []
    for p in preds:
        dup = False
        for o in preds:
            if o is p or o["class_name"] != p["class_name"] or o["score"] <= p["score"]:
                continue
            if math.hypot(p["world_x"] - o["world_x"], p["world_y"] - o["world_y"]) <= radius_m:
                dup = True
                break
        flags.append(dup)
    return flags


def greedy_match(preds: Sequence[Dict[str, Any]], gts: Sequence[Dict[str, Any]],
                 radius_m: float) -> List[Tuple[int, int, float]]:
    """Class-aware nearest-first greedy matching (same rule as visibility_audit_v1.greedy)."""
    cand: List[Tuple[float, int, int]] = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(gts):
            if p["class_name"] != g["label"]:
                continue
            d = math.hypot(p["world_x"] - g["world_x"], p["world_y"] - g["world_y"])
            if d <= radius_m:
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


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def score_setting(
    preds_by_frame: Dict[str, List[Dict[str, Any]]],
    gts_by_frame: Dict[str, List[Dict[str, Any]]],
    frame_ids: Sequence[str],
    thresholds: Dict[str, float],
    nms_radius: Dict[str, float],
    *,
    match_m: float = PRIMARY_MATCH_M,
    with_bands: bool = False,
) -> Dict[str, Any]:
    agg: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    bands: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    n_frames = len(frame_ids)
    for sid in frame_ids:
        raw = preds_by_frame.get(sid, [])
        preds = [p for p in raw if p["score"] >= thresholds.get(p["class_name"], 1.1)]
        preds = world_nms(preds, nms_radius)
        dup_flags = mark_duplicates(preds, DUPLICATE_RADIUS_M)
        gts = gts_by_frame.get(sid, [])

        m = greedy_match(preds, gts, match_m)
        matched_p = {pi for pi, _, _ in m}
        matched_g = {gi for _, gi, _ in m}
        for pi, gi, d in m:
            cls = gts[gi]["label"]
            agg[cls]["tp"] += 1
            agg[cls]["xy_sum"] += d
            if with_bands:
                _band_add(bands, gts[gi], "tp")
        for gi, g in enumerate(gts):
            if gi not in matched_g:
                agg[g["label"]]["fn"] += 1
                if with_bands:
                    _band_add(bands, g, "fn")
        for pi, p in enumerate(preds):
            agg[p["class_name"]]["n_pred"] += 1
            if dup_flags[pi]:
                agg[p["class_name"]]["dup_pred"] += 1
            if pi in matched_p:
                continue
            agg[p["class_name"]]["fp"] += 1
            if dup_flags[pi]:
                agg[p["class_name"]]["dup_fp"] += 1

    out: Dict[str, Any] = {
        "frames": n_frames,
        "vehicle_score_threshold": thresholds["vehicle"],
        "person_score_threshold": thresholds["person"],
        "vehicle_world_nms_m": nms_radius["vehicle"],
        "person_world_nms_m": nms_radius["person"],
        "match_distance_m": match_m,
    }
    tot = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "dup_fp": 0.0, "xy_sum": 0.0}
    for cls in CLASSES:
        a = agg[cls]
        tp, fp, fn = a["tp"], a["fp"], a["fn"]
        prec = tp / max(1.0, tp + fp)
        rec = tp / max(1.0, tp + fn)
        out[f"{cls}_tp"] = int(tp)
        out[f"{cls}_fp"] = int(fp)
        out[f"{cls}_fn"] = int(fn)
        out[f"{cls}_precision"] = prec
        out[f"{cls}_recall"] = rec
        out[f"{cls}_f1"] = 2 * prec * rec / max(1e-9, prec + rec)
        out[f"{cls}_fp_per_frame"] = fp / max(1, n_frames)
        out[f"{cls}_duplicate_fp"] = int(a["dup_fp"])
        out[f"{cls}_duplicate_fp_per_frame"] = a["dup_fp"] / max(1, n_frames)
        out[f"{cls}_duplicate_fp_fraction"] = a["dup_fp"] / max(1.0, fp)
        out[f"{cls}_predictions"] = int(a["n_pred"])
        out[f"{cls}_duplicate_predictions"] = int(a["dup_pred"])
        out[f"{cls}_duplicate_prediction_fraction"] = a["dup_pred"] / max(1.0, a["n_pred"])
        out[f"{cls}_xy_mae_m"] = (a["xy_sum"] / tp) if tp else float("nan")
        for k in ("tp", "fp", "fn", "dup_fp", "xy_sum"):
            tot[k] += a[k]
    prec = tot["tp"] / max(1.0, tot["tp"] + tot["fp"])
    rec = tot["tp"] / max(1.0, tot["tp"] + tot["fn"])
    out["overall_precision"] = prec
    out["overall_recall"] = rec
    out["overall_f1"] = 2 * prec * rec / max(1e-9, prec + rec)
    out["overall_xy_mae_m"] = tot["xy_sum"] / tot["tp"] if tot["tp"] else float("nan")
    out["fp_per_frame"] = tot["fp"] / max(1, n_frames)
    out["duplicate_fp_per_frame"] = tot["dup_fp"] / max(1, n_frames)
    out["mean_f1"] = 0.5 * (out["vehicle_f1"] + out["person_f1"])
    out["mean_xy_mae_m"] = 0.5 * (out["vehicle_xy_mae_m"] + out["person_xy_mae_m"])
    if with_bands:
        out["distance_bands"] = {
            k: {"tp": int(v["tp"]), "fn": int(v["fn"]),
                "recall": v["tp"] / max(1.0, v["tp"] + v["fn"])}
            for k, v in sorted(bands.items())
        }
    return out


def _band_add(bands, g: Dict[str, Any], key: str) -> None:
    d = g["distance_m"]
    for lo, hi in DIST_BANDS:
        if lo <= d < hi:
            bands[f"{int(lo)}-{int(hi)}m|{g['label']}"][key] += 1
            break


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_predictions(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in load_csv(path):
        if str(row.get("match_status")) not in {"tp", "fp"}:
            continue
        out[str(row.get("sample_id", ""))].append({
            "class_name": str(row.get("pred_class_name") or row.get("class_name", "")),
            "world_x": _f(row, "pred_world_x"),
            "world_y": _f(row, "pred_world_y"),
            "score": _f(row, "score"),
        })
    return out


def load_gt(boxes_csv: Path, manifest_csv: Path, split: str):
    dims: Dict[str, Tuple[float, float]] = {}
    split_ids: List[str] = []
    for row in load_csv(manifest_csv):
        if row.get("split") != split:
            continue
        sid = str(row["sample_id"])
        split_ids.append(sid)
        dims[sid] = (_f(row, "camera_width", 0.0), _f(row, "camera_height", 0.0))
    gts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in load_csv(boxes_csv):
        sid = str(row.get("sample_id", ""))
        if sid not in dims:
            continue
        w, h = dims[sid]
        info = classify_gt(row, w, h)
        if info["A"] == "EVALUABLE":
            gts[sid].append(info)
    return split_ids, gts


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment-dir", required=True, type=Path)
    ap.add_argument("--detections", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--split", default="val")
    args = ap.parse_args(argv)

    exp = args.experiment_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    split_ids, gts = load_gt(exp / "dataset" / "object_boxes.csv",
                             exp / "dataset" / "manifest.csv", args.split)
    preds = load_predictions(args.detections)
    collision_ids = {
        str(r["sample_id"]) for r in load_csv(exp / "provenance" / "collision_window_samples.csv")
        if r.get("split") == args.split and str(r.get("retained_in_dataset")) == "1"
    }
    non_collision_ids = [s for s in split_ids if s not in collision_ids]

    gt_counts = defaultdict(int)
    for rows in gts.values():
        for g in rows:
            gt_counts[g["label"]] += 1
    pred_counts = defaultdict(int)
    for rows in preds.values():
        for p in rows:
            pred_counts[p["class_name"]] += 1

    grid: List[Dict[str, Any]] = []
    for vt in VEHICLE_THRESHOLDS:
        for pt in PERSON_THRESHOLDS:
            for vn in VEHICLE_WORLD_NMS_M:
                for pn in PERSON_WORLD_NMS_M:
                    rec = score_setting(
                        preds, gts, split_ids,
                        {"vehicle": vt, "person": pt},
                        {"vehicle": vn, "person": pn},
                    )
                    grid.append(rec)
                    print(
                        f"v_thr={vt:.2f} p_thr={pt:.2f} v_nms={vn:.1f} "
                        f"vR={rec['vehicle_recall']:.4f} vP={rec['vehicle_precision']:.4f} "
                        f"pR={rec['person_recall']:.4f} pP={rec['person_precision']:.4f} "
                        f"meanF1={rec['mean_f1']:.4f}", flush=True)

    with (out_dir / "operating_point_grid.csv").open("w", newline="", encoding="utf-8") as fh:
        keys = [k for k in grid[0].keys() if k != "distance_bands"]
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        for rec in grid:
            wr.writerow({k: rec[k] for k in keys})

    summary = {
        "detections_source": str(args.detections),
        "inference_decode": {
            "object_score_threshold": 0.02, "topk_objects": 120,
            "object_nms_radius_px": 2, "max_gt_distance_m": MAX_GT_DISTANCE_M,
            "match_distance_m": PRIMARY_MATCH_M,
        },
        "gt_rule": "A (object_targets.valid_localization_objects), <=40 m, geometrically evaluable",
        "gt_evaluable": dict(gt_counts),
        "gt_evaluable_total": sum(gt_counts.values()),
        "predictions_retained_at_0p02": dict(pred_counts),
        "frames_primary": len(split_ids),
        "frames_collision_window_excluded": len(non_collision_ids),
        "grid": grid,
    }
    (out_dir / "operating_point_grid.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    # Recall-vs-radius and distance bands + collision sensitivity for the ceiling
    # setting and for the best per-target candidates are produced by the caller
    # via score_setting(); here we emit the ceiling for orientation.
    ceiling = {}
    for r in RECALL_RADII:
        ceiling[f"recall_at_{r:g}m"] = score_setting(
            preds, gts, split_ids, {"vehicle": 0.02, "person": 0.02},
            {"vehicle": 0.0, "person": 0.0}, match_m=r)
    (out_dir / "ceiling_recall_vs_radius.json").write_text(
        json.dumps(ceiling, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: {"vehicle_recall": v["vehicle_recall"],
                          "person_recall": v["person_recall"]}
                      for k, v in ceiling.items()}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
