"""Fit the class-specific ridge range correction on TRAIN-subset frozen-model predictions.

GT is read only here (fit time).  Grouped-by-episode 5-fold CV selects lambda and gates each
class (identity if not helpful).  Final parameters are refit on all training-subset pairs and
frozen to JSON with its sha256.  Validation is never opened by this module.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from . import score as S
from .apply import correct_frame, frame_origin
from .features import FEATURE_NAMES, record_features

FOLDS = (
    ("canonical_v3_01_train_30_30_s501_tm1501", "canonical_v3_02_train_50_50_s502_tm1502"),
    ("canonical_v3_03_train_30_30_s503_tm1503", "canonical_v3_04_train_50_50_s504_tm1504"),
    ("extra_v3_09_train_30_30_s801_tm1801", "extra_v3_10_train_50_50_s802_tm1802"),
    ("extra_v3_11_train_30_30_s803_tm1803", "extra_v3_12_train_50_50_s804_tm1804"),
    ("extra_v3_13_train_30_30_s805_tm1805", "extra_v3_14_train_50_50_s806_tm1806"),
)
LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, "intercept_only")
TARGET_CLIP = 0.5
OUTPUT_CLIP = 0.3
IOU_MIN = 0.5
MAX_FOLD_DROP_PP = 0.5


def iou(a, b) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def associate(preds, gts, cls):
    """Greedy one-to-one by 2-D IoU >= 0.5 (fit-time only; includes gate-crossing over-ranged)."""
    p = [x for x in preds if x["class_name"] == cls and x["score"] >= S.THRESH[cls]]
    cand = []
    for i, x in enumerate(p):
        pb = (x["bbox_x0"], x["bbox_y0"], x["bbox_x1"], x["bbox_y1"])
        for j, g in enumerate(gts):
            v = iou(pb, g["box"])
            if v >= IOU_MIN:
                cand.append((-v, i, j))
    up, ug, out = set(), set(), []
    for _, i, j in sorted(cand):
        if i in up or j in ug:
            continue
        up.add(i); ug.add(j); out.append((p[i], gts[j]))
    return out


def target(pred, gt, origin) -> float | None:
    px, py = pred["world_x"] - origin[0], pred["world_y"] - origin[1]
    gx, gy = gt["world_x"] - origin[0], gt["world_y"] - origin[1]
    den = px * px + py * py
    s = (px * gx + py * gy) / den if den > 1e-9 else -1.0
    if s <= 0:
        return None
    return min(max(math.log(s), -TARGET_CLIP), TARGET_CLIP)


def ridge(X: np.ndarray, y: np.ndarray, lam) -> dict[str, Any]:
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Z = (X - mu) / sd
    b = float(y.mean())
    if lam == "intercept_only":
        w = np.zeros(X.shape[1])
    else:
        n = len(y)
        w = np.linalg.solve(Z.T @ Z + lam * n * np.eye(X.shape[1]), Z.T @ (y - b))
    return {"enabled": True, "intercept": b, "coef": w.tolist(), "mean": mu.tolist(), "std": sd.tolist()}


def params_for(cls_models: dict[str, Any]) -> dict[str, Any]:
    return {"clip_log_scale": OUTPUT_CLIP, "features": list(FEATURE_NAMES), "classes": cls_models}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--diagnostic-ungated", action="store_true",
                    help="also freeze the CV-best-lambda refit with the gate bypassed (diagnostic only, never retainable)")
    a = ap.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    det = S.read_detections(a.detections)
    sids = sorted(det)
    if any("_val_" in s for s in sids):
        raise RuntimeError("validation sample in training detections")
    gts = {"vehicle": S.load_vehicle_gt("train"), "person": S.load_v010_person_gt("train")}
    avo_hold = S.load_avo_table(S.HOLDOUT_AVO)
    manifest = {r["sample_id"]: r for r in csv.DictReader((S.DS / "dataset/manifest.csv").open()) if r["split"] == "train"}
    origins, origin_err, no_origin = {}, [], 0
    for sid in sids:
        o = frame_origin(det[sid])
        if o is None:
            no_origin += 1; continue
        origins[sid] = o
        cam = np.array(json.loads(manifest[sid]["camera_matrix_json"]))[:3, 3]
        origin_err.append(float(np.abs(o - cam).max()))
    ep_of = {sid: S.episode_of(sid) for sid in sids}
    fold_of = {ep: k for k, pair in enumerate(FOLDS) for ep in pair}
    # Fit pairs
    pairs = {c: [] for c in ("vehicle", "person")}
    for sid in sids:
        if sid not in origins:
            continue
        for c in pairs:
            for p, g in associate(det[sid], gts[c].get(sid, []), c):
                y = target(p, g, origins[sid])
                if y is not None:
                    pairs[c].append((fold_of[ep_of[sid]], record_features(p), y, g["distance_m"]))
    summary: dict[str, Any] = {
        "frames": len(sids), "frames_without_origin": no_origin,
        "origin_vs_camera_matrix_max_abs_m": max(origin_err) if origin_err else None,
        "episodes": sorted(set(ep_of.values())), "pairs": {c: len(v) for c, v in pairs.items()},
        "target_stats": {}, "cv": {}, "selection": {}}
    for c, v in pairs.items():
        y = np.array([t[2] for t in v]); d = np.array([t[3] for t in v])
        summary["target_stats"][c] = {
            "mean_log_scale": float(y.mean()), "median_log_scale": float(np.median(y)),
            "sd": float(y.std()), "clipped_share": float(np.mean(np.abs(y) >= TARGET_CLIP)),
            "by_gt_bin_mean": {b: float(y[(d >= lo) & (d < hi)].mean()) if np.any((d >= lo) & (d < hi)) else None
                               for b, lo, hi in S.BINS}}

    def evaluate_fold(c, k, model) -> dict[str, Any]:
        params = params_for({c: model} if model else {})
        held = [s for s in sids if fold_of[ep_of[s]] == k]
        el = tp = w = 0; avo_el = avo_w = 0
        for sid in held:
            preds = correct_frame(det[sid], params, origins.get(sid)) if (model and sid in origins) else det[sid]
            g = gts[c].get(sid, [])
            errs = S.simple_frame(c, preds, g)
            el += len(errs); tp += sum(e is not None for e in errs); w += sum(e is not None and e <= S.TOL[c] for e in errs)
            if c == "person" and k == 1:
                elig = [r for r in avo_hold.get(sid, []) if r["avo"] >= S.AVO_T]
                pp = sorted([dict(x) for x in preds if x["class_name"] == "person" and x["score"] >= S.THRESH["person"]],
                            key=lambda it: -it["score"])
                m, _ = S.quality_adapter.greedy_match(pp, elig)
                avo_el += len(elig)
                avo_w += sum(math.hypot(pp[pi]["world_x"] - elig[gi]["world_x"], pp[pi]["world_y"] - elig[gi]["world_y"]) <= S.TOL[c]
                             for pi, gi in m.items())
        return {"eligible": el, "tp": tp, "within": w, "avo_eligible": avo_el, "avo_within": avo_w}

    models_final, models_ungated = {}, {}
    for c, v in pairs.items():
        folds = np.array([t[0] for t in v]); X = np.array([t[1] for t in v]); y = np.array([t[2] for t in v])
        base = {k: evaluate_fold(c, k, None) for k in range(len(FOLDS))}
        res = {"identity": base}
        for lam in LAMBDAS:
            per = {}
            for k in range(len(FOLDS)):
                tr = folds != k
                per[k] = evaluate_fold(c, k, ridge(X[tr], y[tr], lam))
            res[str(lam)] = per
        table = {}
        for key, per in res.items():
            el = sum(p["eligible"] for p in per.values()); w = sum(p["within"] for p in per.values())
            fold_delta_pp = [100 * (per[k]["within"] - base[k]["within"]) / max(1, base[k]["eligible"]) for k in per]
            table[key] = {"eligible": el, "within": w, "tp": sum(p["tp"] for p in per.values()),
                          "within_recall": w / el, "fold_within_delta_pp": fold_delta_pp,
                          "avo_0304_within": per[1]["avo_within"], "avo_0304_eligible": per[1]["avo_eligible"]}
        summary["cv"][c] = table
        rank = {str(l): i for i, l in enumerate(LAMBDAS)}  # ties -> stronger regularization
        best = max((k for k in table if k != "identity"), key=lambda k: (table[k]["within"], rank[k]))
        gain = table[best]["within"] - table["identity"]["within"]
        worst_fold = min(table[best]["fold_within_delta_pp"])
        avo_ok = c != "person" or table[best]["avo_0304_within"] >= table["identity"]["avo_0304_within"]
        enabled = gain > 0 and worst_fold >= -MAX_FOLD_DROP_PP and avo_ok
        lam = best if best == "intercept_only" else float(best)
        summary["selection"][c] = {"best_lambda": best, "oof_within_gain": gain, "worst_fold_delta_pp": worst_fold,
                                   "person_avo_0304_ok": avo_ok, "enabled": enabled}
        models_ungated[c] = ridge(X, y, lam)
        models_ungated[c].update({"lambda": best, "n_pairs": int(len(y)), "gate_bypassed": True})
        if enabled:
            models_final[c] = ridge(X, y, lam)
            models_final[c]["lambda"] = best
            models_final[c]["n_pairs"] = int(len(y))
        else:
            models_final[c] = {"enabled": False, "reason": "holdout gate not passed -> identity"}
    params = params_for(models_final)
    params.update({"schema": "splitfusion_range_correction_v1.frozen_params", "target_clip": TARGET_CLIP,
                   "iou_min": IOU_MIN, "training_detections_sha256": hashlib.sha256(Path(a.detections).read_bytes()).hexdigest(),
                   "validation_accessed": False, "folds": [list(f) for f in FOLDS]})
    blob = (json.dumps(params, indent=1, sort_keys=True) + "\n").encode()
    (out / "FROZEN_PARAMS.json").write_bytes(blob)
    digest = hashlib.sha256(blob).hexdigest()
    (out / "FROZEN_PARAMS.sha256").write_text(digest + "  FROZEN_PARAMS.json\n")
    summary["frozen_params_sha256"] = digest
    if a.diagnostic_ungated:
        diag = params_for(models_ungated)
        diag.update({"schema": "splitfusion_range_correction_v1.frozen_params.DIAGNOSTIC_UNGATED",
                     "retainable": False, "validation_accessed": False,
                     "training_detections_sha256": params["training_detections_sha256"]})
        blob = (json.dumps(diag, indent=1, sort_keys=True) + "\n").encode()
        (out / "FROZEN_PARAMS_DIAGNOSTIC_UNGATED.json").write_bytes(blob)
        summary["diagnostic_ungated_sha256"] = hashlib.sha256(blob).hexdigest()
        (out / "FROZEN_PARAMS_DIAGNOSTIC_UNGATED.sha256").write_text(summary["diagnostic_ungated_sha256"] + "  FROZEN_PARAMS_DIAGNOSTIC_UNGATED.json\n")
    (out / "FIT_SUMMARY.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"pairs": summary["pairs"], "target_stats": summary["target_stats"],
                      "selection": summary["selection"], "sha256": digest,
                      "cv": {c: {k: (v["within"], v["eligible"], round(v["within_recall"], 4), [round(x, 2) for x in v["fold_within_delta_pp"]],
                                     v["avo_0304_within"]) for k, v in t.items()} for c, t in summary["cv"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
