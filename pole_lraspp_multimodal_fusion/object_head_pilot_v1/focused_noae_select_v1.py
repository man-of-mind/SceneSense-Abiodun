#!/usr/bin/env python3
"""Select the focused_noae_v1 checkpoint from decoded validation service metrics.

Registered rule (fixed before any candidate was scored):

  ELIGIBILITY - clean q=0.00 non-regression vs the matched epoch-13 baseline,
    all of which must hold:
      vehicle_f1     >= baseline - 0.005
      person_f1      >= baseline - 0.005
      overall_xy_mae_m       <= baseline + 0.10 m    (localization)
      overall_dimension_mae_m<= baseline + 0.05 m    (dimensions)
      miou / vehicle_iou / person_iou >= baseline - 0.005  (segmentation)

  IMPROVEMENT - must CLEARLY beat the matched baseline, else FAILED:
      clean q=0.00 mean(vehicle_f1, person_f1) >= baseline + 0.010

  RANKING among eligible candidates, in strict order:
      1. maximize the WORST-ANCHOR mean of (vehicle_f1, person_f1) over the six
         registered anchors {0.00,0.30,0.50,0.70,0.90,0.98}
      2. minimize mean overall_xy_mae_m across those anchors
      3. minimize mean duplicate FP per frame across those anchors

  Interval midpoints {0.15,0.40,0.60,0.80,0.94} are REPORTED for the shortlist
  (they show interpolation between anchors) but never enter the ranking.

  loc_dim_loss and every in-training selection_score are ignored here by design.

Absolute service-quality targets are advisory: printed and recorded, never gating.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ANCHORS = [0.0, 0.3, 0.5, 0.7, 0.9, 0.98]
MIDPOINTS = [0.15, 0.40, 0.60, 0.80, 0.94]

ADVISORY_TARGETS = {
    "clean_vehicle_recall": 0.60,
    "clean_person_recall": 0.50,
    "clean_mean_f1": 0.60,
}

TOL_F1 = 0.005
TOL_XY = 0.10
TOL_DIM = 0.05
TOL_SEG = 0.005
IMPROVE_MARGIN = 0.010


def qtag(q: float) -> str:
    return f"q{int(round(q * 100)):03d}"


def load_eval(eval_dir: Path) -> Optional[Dict[str, Any]]:
    dm = eval_dir / "derived_metrics.json"
    em = eval_dir / "evaluator_metrics.json"
    if not dm.is_file():
        return None
    d = json.loads(dm.read_text())
    p = dict(d["primary"])
    if em.is_file():
        e = json.loads(em.read_text())
        for k, v in e.items():
            if isinstance(v, (int, float)) and any(
                t in k.lower() for t in ("iou", "pixel_accuracy", "miou")
            ):
                p[f"seg_{k}"] = v
    p["_feature_drop_fraction"] = d.get("feature_drop_fraction", 0.0)
    p["_checkpoint"] = d.get("checkpoint", "")
    return p


def seg(m: Dict[str, Any], *names: str) -> Optional[float]:
    for n in names:
        for k in m:
            if k.startswith("seg_") and k.lower().endswith(n):
                return float(m[k])
    return None


def mean_f1(m: Dict[str, Any]) -> float:
    return 0.5 * (float(m.get("vehicle_f1", 0.0)) + float(m.get("person_f1", 0.0)))


def dup_per_frame(m: Dict[str, Any]) -> float:
    return float(m.get("overall_duplicate_fp", 0.0)) / max(1, int(m.get("frames", 1)))


def collect(exp: Path, prefix: str) -> Dict[int, Dict[float, Dict[str, Any]]]:
    out: Dict[int, Dict[float, Dict[str, Any]]] = {}
    root = exp / "eval"
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir()):
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-dir", required=True, type=Path)
    ap.add_argument("--baseline-dir", required=True, type=Path)
    ap.add_argument("--baseline-tag-prefix", default="baseline_epoch13")
    ap.add_argument("--prefix", default="focused_ep")
    ap.add_argument("--stage", choices=("shortlist", "final"), default="final")
    ap.add_argument("--shortlist-size", type=int, default=3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    exp = args.experiment_dir.resolve()
    base_evals = {}
    for d in sorted((args.baseline_dir.resolve() / "eval").iterdir()):
        if d.name.startswith(args.baseline_tag_prefix):
            try:
                q = int(d.name.split("_q")[1]) / 100.0
            except (IndexError, ValueError):
                continue
            m = load_eval(d)
            if m is not None:
                base_evals[q] = m
    if 0.0 not in base_evals:
        print("FATAL: no clean q=0.00 baseline decode found", flush=True)
        return 2
    b0 = base_evals[0.0]
    cands = collect(exp, args.prefix)
    if not cands:
        print("FATAL: no candidate decodes found", flush=True)
        return 2

    rows: List[Dict[str, Any]] = []
    for ep in sorted(cands):
        per_q = cands[ep]
        if 0.0 not in per_q:
            continue
        c0 = per_q[0.0]
        reasons: List[str] = []
        if float(c0.get("vehicle_f1", 0)) < float(b0["vehicle_f1"]) - TOL_F1:
            reasons.append(f"clean vehicle_f1 {c0['vehicle_f1']:.4f} < {b0['vehicle_f1']:.4f}-{TOL_F1}")
        if float(c0.get("person_f1", 0)) < float(b0["person_f1"]) - TOL_F1:
            reasons.append(f"clean person_f1 {c0['person_f1']:.4f} < {b0['person_f1']:.4f}-{TOL_F1}")
        if float(c0.get("overall_xy_mae_m", 9e9)) > float(b0["overall_xy_mae_m"]) + TOL_XY:
            reasons.append(f"clean xy_mae {c0['overall_xy_mae_m']:.4f} > {b0['overall_xy_mae_m']:.4f}+{TOL_XY}")
        if float(c0.get("overall_dimension_mae_m", 9e9)) > float(b0["overall_dimension_mae_m"]) + TOL_DIM:
            reasons.append(
                f"clean dim_mae {c0['overall_dimension_mae_m']:.4f} > "
                f"{b0['overall_dimension_mae_m']:.4f}+{TOL_DIM}")
        for nm, keys in (("miou", ("miou",)), ("vehicle_iou", ("vehicle_iou",)),
                         ("person_iou", ("person_iou",))):
            cv, bv = seg(c0, *keys), seg(b0, *keys)
            if cv is not None and bv is not None and cv < bv - TOL_SEG:
                reasons.append(f"clean {nm} {cv:.4f} < {bv:.4f}-{TOL_SEG}")

        have = [q for q in ANCHORS if q in per_q]
        worst = min((mean_f1(per_q[q]) for q in have), default=float("nan"))
        worst_q = min(have, key=lambda q: mean_f1(per_q[q])) if have else None
        rows.append({
            "epoch": ep,
            "checkpoint": c0.get("_checkpoint", ""),
            "anchors_scored": have,
            "clean_mean_f1": mean_f1(c0),
            "clean_vehicle_f1": float(c0.get("vehicle_f1", float("nan"))),
            "clean_person_f1": float(c0.get("person_f1", float("nan"))),
            "clean_vehicle_recall": float(c0.get("vehicle_recall", float("nan"))),
            "clean_person_recall": float(c0.get("person_recall", float("nan"))),
            "clean_xy_mae_m": float(c0.get("overall_xy_mae_m", float("nan"))),
            "clean_dimension_mae_m": float(c0.get("overall_dimension_mae_m", float("nan"))),
            "clean_miou": seg(c0, "miou"),
            "clean_vehicle_iou": seg(c0, "vehicle_iou"),
            "clean_person_iou": seg(c0, "person_iou"),
            "worst_anchor_mean_f1": worst,
            "worst_anchor_q": worst_q,
            "mean_xy_mae_over_anchors": (
                sum(float(per_q[q].get("overall_xy_mae_m", 0.0)) for q in have) / len(have)
                if have else float("nan")),
            "mean_dup_fp_per_frame_over_anchors": (
                sum(dup_per_frame(per_q[q]) for q in have) / len(have) if have else float("nan")),
            "eligible": not reasons,
            "ineligible_reasons": reasons,
            "improves": mean_f1(c0) >= mean_f1(b0) + IMPROVE_MARGIN,
            "per_q": {qtag(q): {
                "q": q,
                "vehicle_f1": per_q[q].get("vehicle_f1"),
                "person_f1": per_q[q].get("person_f1"),
                "vehicle_recall": per_q[q].get("vehicle_recall"),
                "person_recall": per_q[q].get("person_recall"),
                "vehicle_precision": per_q[q].get("vehicle_precision"),
                "person_precision": per_q[q].get("person_precision"),
                "mean_f1": mean_f1(per_q[q]),
                "overall_xy_mae_m": per_q[q].get("overall_xy_mae_m"),
                "overall_dimension_mae_m": per_q[q].get("overall_dimension_mae_m"),
                "duplicate_fp_per_frame": dup_per_frame(per_q[q]),
                "miou": seg(per_q[q], "miou"),
                "vehicle_iou": seg(per_q[q], "vehicle_iou"),
                "person_iou": seg(per_q[q], "person_iou"),
                "is_registered_anchor": any(abs(q - a) < 1e-9 for a in ANCHORS),
            } for q in sorted(per_q)},
        })

    if args.stage == "shortlist":
        ranked = sorted(rows, key=lambda r: (-r["clean_mean_f1"], r["clean_xy_mae_m"]))
        picks = [r for r in ranked if r["eligible"]][: args.shortlist_size]
        if not picks:
            picks = ranked[: args.shortlist_size]
        result = {
            "stage": "shortlist",
            "baseline_clean_mean_f1": mean_f1(b0),
            "shortlist_epochs": [r["epoch"] for r in picks],
            "ranked_clean": [
                {k: r[k] for k in ("epoch", "clean_mean_f1", "clean_vehicle_f1", "clean_person_f1",
                                   "clean_xy_mae_m", "eligible", "improves", "ineligible_reasons")}
                for r in ranked],
        }
    else:
        pool = [r for r in rows if r["eligible"] and r["improves"] and len(r["anchors_scored"]) == len(ANCHORS)]
        ranked = sorted(pool, key=lambda r: (-r["worst_anchor_mean_f1"],
                                             r["mean_xy_mae_over_anchors"],
                                             r["mean_dup_fp_per_frame_over_anchors"]))
        verdict = "FOCUSED_NOAE_TRAINING_FAILED" if not ranked else "SELECTED"
        sel = ranked[0] if ranked else None
        advisory = {}
        if sel:
            advisory = {
                "clean_vehicle_recall": {"value": sel["clean_vehicle_recall"],
                                         "target": ADVISORY_TARGETS["clean_vehicle_recall"],
                                         "met": sel["clean_vehicle_recall"] >= ADVISORY_TARGETS["clean_vehicle_recall"]},
                "clean_person_recall": {"value": sel["clean_person_recall"],
                                        "target": ADVISORY_TARGETS["clean_person_recall"],
                                        "met": sel["clean_person_recall"] >= ADVISORY_TARGETS["clean_person_recall"]},
                "clean_mean_f1": {"value": sel["clean_mean_f1"],
                                  "target": ADVISORY_TARGETS["clean_mean_f1"],
                                  "met": sel["clean_mean_f1"] >= ADVISORY_TARGETS["clean_mean_f1"]},
            }
        result = {
            "stage": "final",
            "verdict": verdict,
            "note": ("advisory targets are reported, never gating; ranking is "
                     "worst-anchor mean F1 -> mean XY MAE -> mean duplicate FP/frame"),
            "baseline": {
                "checkpoint": b0.get("_checkpoint", ""),
                "clean_mean_f1": mean_f1(b0),
                "clean_vehicle_f1": b0.get("vehicle_f1"),
                "clean_person_f1": b0.get("person_f1"),
                "clean_vehicle_recall": b0.get("vehicle_recall"),
                "clean_person_recall": b0.get("person_recall"),
                "clean_xy_mae_m": b0.get("overall_xy_mae_m"),
                "clean_dimension_mae_m": b0.get("overall_dimension_mae_m"),
                "clean_miou": seg(b0, "miou"),
                "clean_vehicle_iou": seg(b0, "vehicle_iou"),
                "clean_person_iou": seg(b0, "person_iou"),
                "per_q": {qtag(q): {"mean_f1": mean_f1(m), "vehicle_f1": m.get("vehicle_f1"),
                                    "person_f1": m.get("person_f1"),
                                    "overall_xy_mae_m": m.get("overall_xy_mae_m")}
                          for q, m in sorted(base_evals.items())},
            },
            "selected": sel,
            "advisory_targets": advisory,
            "all_candidates": rows,
        }

    out = Path(args.out) if args.out else (exp / f"focused_noae_selection_{args.stage}.json")
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in result.items() if k not in ("all_candidates", "ranked_clean")},
                     indent=2, sort_keys=True)[:6000], flush=True)
    print(f"\nwritten: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
