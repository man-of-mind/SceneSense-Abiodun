#!/usr/bin/env python3
"""Apply the registered family-relative gate (REGISTERED_GATE_AE_FAMILIES.md).

Stages:
  shortlist -- rank the predeclared sparse set on {q=0.00, q=0.98}; top 2 by the
               WORST of those two mean F1 values, tie-broken by clean mean F1.
  final     -- apply the registered PASS conditions over the six anchors and rank.

Every candidate cell is compared against the matched family baseline decoded at the
SAME q. Missing cells produce INCOMPLETE_EVALUATION, never a silent selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ANCHORS = [0.0, 0.3, 0.5, 0.7, 0.9, 0.98]
MIDPOINTS = [0.15, 0.40, 0.60, 0.80, 0.94]
SPARSE_Q = [0.0, 0.98]

TOL_MEAN_F1 = 0.005
TOL_CLASS_F1 = 0.010
TOL_XY = 0.05
TOL_SEG = 0.01
TOL_DIM = 0.02
MIN_WORST_ANCHOR_GAIN = 0.03

ADVISORY = {"clean_vehicle_recall": 0.60, "clean_person_recall": 0.50, "clean_mean_f1": 0.60}


def qtag(q: float) -> str:
    return f"q{int(round(q * 100)):03d}"


def load_eval(d: Path) -> Optional[Dict[str, Any]]:
    dm = d / "derived_metrics.json"
    if not dm.is_file():
        return None
    j = json.loads(dm.read_text())
    m = dict(j["primary"])
    em = d / "evaluator_metrics.json"
    if em.is_file():
        for k, v in json.loads(em.read_text()).items():
            if isinstance(v, (int, float)) and any(t in k.lower() for t in ("iou", "pixel_accuracy")):
                m[f"seg_{k}"] = v
    m["_q"] = j.get("feature_drop_fraction", 0.0)
    m["_checkpoint"] = j.get("checkpoint", "")
    return m


def seg(m: Dict[str, Any], name: str) -> Optional[float]:
    for k in m:
        if k.startswith("seg_") and k[4:].lower() == name:
            return float(m[k])
    return None


def mf1(m: Dict[str, Any]) -> float:
    return 0.5 * (float(m.get("vehicle_f1", 0.0)) + float(m.get("person_f1", 0.0)))


def dupf(m: Dict[str, Any]) -> float:
    return float(m.get("overall_duplicate_fp", 0.0)) / max(1, int(m.get("frames", 1)))


def scan(root: Path, prefix: str) -> Dict[int, Dict[float, Dict[str, Any]]]:
    out: Dict[int, Dict[float, Dict[str, Any]]] = {}
    if not (root / "eval").is_dir():
        return out
    for d in sorted((root / "eval").iterdir()):
        if not d.name.startswith(prefix):
            continue
        try:
            ep = int(d.name.split("_ep")[1].split("_")[0])
            q = int(d.name.split("_q")[1]) / 100.0
        except (IndexError, ValueError):
            continue
        m = load_eval(d)
        if m is not None:
            out.setdefault(ep, {})[q] = m
    return out


def scan_baseline(root: Path, prefix: str) -> Dict[float, Dict[str, Any]]:
    out: Dict[float, Dict[str, Any]] = {}
    if not (root / "eval").is_dir():
        return out
    for d in sorted((root / "eval").iterdir()):
        if not d.name.startswith(prefix):
            continue
        try:
            q = int(d.name.split("_q")[1]) / 100.0
        except (IndexError, ValueError):
            continue
        m = load_eval(d)
        if m is not None:
            out[q] = m
    return out


def gate_reasons(c0: Dict[str, Any], b0: Dict[str, Any]) -> List[str]:
    r: List[str] = []
    if mf1(c0) < mf1(b0) - TOL_MEAN_F1:
        r.append(f"clean mean F1 {mf1(c0):.4f} < {mf1(b0):.4f}-{TOL_MEAN_F1}")
    for cls in ("vehicle", "person"):
        cv, bv = float(c0.get(f"{cls}_f1", 0)), float(b0.get(f"{cls}_f1", 0))
        if cv < bv - TOL_CLASS_F1:
            r.append(f"clean {cls} F1 {cv:.4f} < {bv:.4f}-{TOL_CLASS_F1}")
    cv, bv = float(c0.get("overall_xy_mae_m", 9e9)), float(b0.get("overall_xy_mae_m", 0))
    if cv > bv + TOL_XY:
        r.append(f"clean XY MAE {cv:.4f} > {bv:.4f}+{TOL_XY}")
    cv, bv = float(c0.get("overall_dimension_mae_m", 9e9)), float(b0.get("overall_dimension_mae_m", 0))
    if cv > bv + TOL_DIM:
        r.append(f"clean dim MAE {cv:.4f} > {bv:.4f}+{TOL_DIM}")
    for nm in ("miou", "vehicle_iou", "person_iou"):
        a, b = seg(c0, nm), seg(b0, nm)
        if a is not None and b is not None and a < b - TOL_SEG:
            r.append(f"clean {nm} {a:.4f} < {b:.4f}-{TOL_SEG}")
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--experiment-dir", required=True, type=Path)
    ap.add_argument("--baseline-dir", required=True, type=Path)
    ap.add_argument("--baseline-prefix", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--stage", choices=("shortlist", "final"), required=True)
    ap.add_argument("--sparse-epochs", default="5,9,13,17,20,23")
    args = ap.parse_args()

    exp = args.experiment_dir.resolve()
    base = scan_baseline(args.baseline_dir.resolve(), args.baseline_prefix)
    cands = scan(exp, args.prefix)
    sparse = [int(x) for x in args.sparse_epochs.split(",")]

    if args.stage == "shortlist":
        rows = []
        missing = []
        for ep in sparse:
            per = cands.get(ep, {})
            gap = [q for q in SPARSE_Q if q not in per]
            if gap:
                missing.append({"epoch": ep, "missing_q": gap})
                continue
            rows.append({"epoch": ep,
                         "mean_f1_q000": mf1(per[0.0]), "mean_f1_q098": mf1(per[0.98]),
                         "worst_of_two": min(mf1(per[0.0]), mf1(per[0.98]))})
        res = {"stage": "shortlist", "family": args.family,
               "sparse_epochs_declared": sparse, "missing_cells": missing,
               "ranked": sorted(rows, key=lambda r: (-r["worst_of_two"], -r["mean_f1_q000"]))}
        res["shortlist_epochs"] = [r["epoch"] for r in res["ranked"][:2]]
        out = exp / f"{args.family}_shortlist.json"
    else:
        base_gap = [q for q in ANCHORS if q not in base]
        rows, cand_gap = [], []
        shortlist = json.loads((exp / f"{args.family}_shortlist.json").read_text())["shortlist_epochs"]
        for ep in shortlist:
            per = cands.get(ep, {})
            gap = [q for q in ANCHORS if q not in per]
            if gap:
                cand_gap.append({"epoch": ep, "missing_q": gap})
                continue
            c0, b0 = per[0.0], base[0.0]
            reasons = gate_reasons(c0, b0)
            worst = min(mf1(per[q]) for q in ANCHORS)
            bworst = min(mf1(base[q]) for q in ANCHORS)
            gain = worst - bworst
            if gain < MIN_WORST_ANCHOR_GAIN:
                reasons.append(f"worst-anchor gain {gain:+.4f} < required +{MIN_WORST_ANCHOR_GAIN}")
            rows.append({
                "epoch": ep, "checkpoint": c0.get("_checkpoint", ""),
                "clean_mean_f1": mf1(c0), "clean_vehicle_f1": c0.get("vehicle_f1"),
                "clean_person_f1": c0.get("person_f1"),
                "clean_vehicle_recall": c0.get("vehicle_recall"),
                "clean_person_recall": c0.get("person_recall"),
                "clean_xy_mae_m": c0.get("overall_xy_mae_m"),
                "clean_dimension_mae_m": c0.get("overall_dimension_mae_m"),
                "clean_miou": seg(c0, "miou"), "clean_vehicle_iou": seg(c0, "vehicle_iou"),
                "clean_person_iou": seg(c0, "person_iou"),
                "worst_anchor_mean_f1": worst, "worst_anchor_q": min(ANCHORS, key=lambda q: mf1(per[q])),
                "baseline_worst_anchor_mean_f1": bworst, "worst_anchor_gain": gain,
                "mean_xy_mae_over_anchors": sum(float(per[q]["overall_xy_mae_m"]) for q in ANCHORS) / len(ANCHORS),
                "mean_dup_fp_per_frame_over_anchors": sum(dupf(per[q]) for q in ANCHORS) / len(ANCHORS),
                "passes": not reasons, "fail_reasons": reasons,
                "per_q": {qtag(q): {
                    "q": q, "is_registered_anchor": q in ANCHORS,
                    "vehicle_f1": per[q].get("vehicle_f1"), "person_f1": per[q].get("person_f1"),
                    "vehicle_recall": per[q].get("vehicle_recall"), "person_recall": per[q].get("person_recall"),
                    "vehicle_precision": per[q].get("vehicle_precision"), "person_precision": per[q].get("person_precision"),
                    "mean_f1": mf1(per[q]), "overall_xy_mae_m": per[q].get("overall_xy_mae_m"),
                    "overall_dimension_mae_m": per[q].get("overall_dimension_mae_m"),
                    "duplicate_fp_per_frame": dupf(per[q]),
                    "miou": seg(per[q], "miou"), "vehicle_iou": seg(per[q], "vehicle_iou"),
                    "person_iou": seg(per[q], "person_iou"),
                    "baseline_at_q": ({"vehicle_f1": base[q].get("vehicle_f1"),
                                       "person_f1": base[q].get("person_f1"),
                                       "mean_f1": mf1(base[q]),
                                       "overall_xy_mae_m": base[q].get("overall_xy_mae_m")}
                                      if q in base else None),
                    "delta_vs_baseline_at_q": ({
                        "vehicle_f1": float(per[q].get("vehicle_f1", 0)) - float(base[q].get("vehicle_f1", 0)),
                        "person_f1": float(per[q].get("person_f1", 0)) - float(base[q].get("person_f1", 0)),
                        "mean_f1": mf1(per[q]) - mf1(base[q]),
                        "overall_xy_mae_m": float(per[q].get("overall_xy_mae_m", 0)) - float(base[q].get("overall_xy_mae_m", 0)),
                    } if q in base else None),
                    "baseline_missing_at_q": q not in base,
                } for q in sorted(per)},
            })
        if base_gap or cand_gap:
            res = {"stage": "final", "family": args.family,
                   "verdict": "INCOMPLETE_EVALUATION",
                   "missing_baseline_q": base_gap, "missing_candidate_cells": cand_gap,
                   "note": "refusing to select from missing results"}
        else:
            passing = sorted([r for r in rows if r["passes"]],
                             key=lambda r: (-r["worst_anchor_mean_f1"],
                                            r["mean_xy_mae_over_anchors"],
                                            r["mean_dup_fp_per_frame_over_anchors"]))
            sel = passing[0] if passing else None
            res = {"stage": "final", "family": args.family,
                   "verdict": "PASS" if sel else "FAIL",
                   "baseline": {
                       "checkpoint": base[0.0].get("_checkpoint", ""),
                       "clean_mean_f1": mf1(base[0.0]),
                       "clean_vehicle_f1": base[0.0].get("vehicle_f1"),
                       "clean_person_f1": base[0.0].get("person_f1"),
                       "clean_xy_mae_m": base[0.0].get("overall_xy_mae_m"),
                       "clean_dimension_mae_m": base[0.0].get("overall_dimension_mae_m"),
                       "clean_miou": seg(base[0.0], "miou"),
                       "clean_vehicle_iou": seg(base[0.0], "vehicle_iou"),
                       "clean_person_iou": seg(base[0.0], "person_iou"),
                       "worst_anchor_mean_f1": min(mf1(base[q]) for q in ANCHORS),
                       "per_q": {qtag(q): {"mean_f1": mf1(base[q]),
                                           "vehicle_f1": base[q].get("vehicle_f1"),
                                           "person_f1": base[q].get("person_f1")}
                                 for q in sorted(base)}},
                   "selected": sel, "candidates": rows,
                   "advisory_targets": ({k: {"value": sel.get(k), "target": v,
                                             "met": sel.get(k) is not None and sel[k] >= v}
                                         for k, v in ADVISORY.items()} if sel else
                                        {k: {"target": v, "value": None, "met": False} for k, v in ADVISORY.items()}),
                   "completeness": {"complete": True, "anchors": ANCHORS, "shortlist": shortlist}}
        out = exp / f"{args.family}_final.json"

    out.write_text(json.dumps(res, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in res.items() if k not in ("candidates",)}, indent=2, sort_keys=True)[:3500], flush=True)
    print(f"\nwritten: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
