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
            "baseline_worst_anchor_mean_f1": (
                min((mean_f1(base_evals[q]) for q in have if q in base_evals), default=None)
                if have else None),
            "anchors_beating_matched_baseline": [
                q for q in have if q in base_evals and mean_f1(per_q[q]) > mean_f1(base_evals[q])],
            "anchors_below_matched_baseline": [
                q for q in have if q in base_evals and mean_f1(per_q[q]) <= mean_f1(base_evals[q])],
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
                # Matched-q comparison: this candidate at q vs the epoch-13 baseline
                # decoded at the SAME q. Never against the clean baseline only.
                "baseline_at_q": ({
                    "vehicle_f1": base_evals[q].get("vehicle_f1"),
                    "person_f1": base_evals[q].get("person_f1"),
                    "vehicle_recall": base_evals[q].get("vehicle_recall"),
                    "person_recall": base_evals[q].get("person_recall"),
                    "mean_f1": mean_f1(base_evals[q]),
                    "overall_xy_mae_m": base_evals[q].get("overall_xy_mae_m"),
                    "duplicate_fp_per_frame": dup_per_frame(base_evals[q]),
                } if q in base_evals else None),
                "delta_vs_baseline_at_q": ({
                    "vehicle_f1": float(per_q[q].get("vehicle_f1", 0.0)) - float(base_evals[q].get("vehicle_f1", 0.0)),
                    "person_f1": float(per_q[q].get("person_f1", 0.0)) - float(base_evals[q].get("person_f1", 0.0)),
                    "vehicle_recall": float(per_q[q].get("vehicle_recall", 0.0)) - float(base_evals[q].get("vehicle_recall", 0.0)),
                    "person_recall": float(per_q[q].get("person_recall", 0.0)) - float(base_evals[q].get("person_recall", 0.0)),
                    "mean_f1": mean_f1(per_q[q]) - mean_f1(base_evals[q]),
                    "overall_xy_mae_m": float(per_q[q].get("overall_xy_mae_m", 0.0)) - float(base_evals[q].get("overall_xy_mae_m", 0.0)),
                    "duplicate_fp_per_frame": dup_per_frame(per_q[q]) - dup_per_frame(base_evals[q]),
                } if q in base_evals else None),
                "baseline_missing_at_q": q not in base_evals,
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
        # ---- Completeness audit: never silently select from missing results. ----
        expected_full = sorted(ANCHORS + MIDPOINTS)
        shortlist_file = exp / "focused_noae_selection_shortlist.json"
        if shortlist_file.is_file():
            shortlist_epochs = list(json.loads(shortlist_file.read_text())["shortlist_epochs"])
        else:
            shortlist_epochs = [r["epoch"] for r in rows if len(r["per_q"]) > 1]
        missing: Dict[str, List[Any]] = {}
        for ep in shortlist_epochs:
            row = next((r for r in rows if r["epoch"] == ep), None)
            if row is None:
                missing[f"epoch_{ep}"] = ["no decode at all"]
                continue
            have_q = {round(v["q"], 4) for v in row["per_q"].values()}
            gap = [q for q in expected_full if round(q, 4) not in have_q]
            if gap:
                missing[f"epoch_{ep}"] = gap
        base_gap = [q for q in expected_full if q not in base_evals]
        if base_gap:
            missing["baseline_epoch13"] = base_gap
        clean_gap = [r["epoch"] for r in rows if 0.0 not in {v["q"] for v in r["per_q"].values()}]
        if clean_gap:
            missing["candidates_without_clean_decode"] = clean_gap

        completeness = {
            "expected_q_grid": expected_full,
            "shortlist_epochs": shortlist_epochs,
            "candidate_epochs_decoded_clean": [r["epoch"] for r in rows],
            "missing_cells": missing,
            "complete": not missing,
        }
        if missing:
            result = {
                "stage": "final",
                "verdict": "FOCUSED_NOAE_INCOMPLETE_EVALUATION",
                "note": ("evaluation lanes did not all complete; refusing to select from "
                         "missing results. Re-run the decode sweep for the missing cells."),
                "completeness": completeness,
                "all_candidates": rows,
            }
            out = Path(args.out) if args.out else (exp / f"focused_noae_selection_{args.stage}.json")
            out.write_text(json.dumps(result, indent=2, sort_keys=True))
            print(json.dumps({k: v for k, v in result.items() if k != "all_candidates"},
                             indent=2, sort_keys=True), flush=True)
            print(f"\nwritten: {out}", flush=True)
            return 3

        pool = [r for r in rows if r["eligible"] and r["improves"] and len(r["anchors_scored"]) == len(ANCHORS)]
        dropped_incomplete = [r["epoch"] for r in rows
                              if r["eligible"] and r["improves"]
                              and len(r["anchors_scored"]) != len(ANCHORS)]
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
            "completeness": completeness,
            "dropped_for_incomplete_anchor_coverage": dropped_incomplete,
            "advisory_targets": advisory,
            "all_candidates": rows,
        }

    out = Path(args.out) if args.out else (exp / f"focused_noae_selection_{args.stage}.json")
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in result.items() if k not in ("all_candidates", "ranked_clean")},
                     indent=2, sort_keys=True)[:6000], flush=True)
    print(f"\nwritten: {out}", flush=True)

    if args.stage == "final" and result.get("selected"):
        (exp / "FOCUSED_NOAE_DECISION.md").write_text(decision_md(result))
        print(f"written: {exp / 'FOCUSED_NOAE_DECISION.md'}", flush=True)
    return 0


def decision_md(res: Dict[str, Any]) -> str:
    sel = res["selected"]
    b = res["baseline"]
    L = ["# FOCUSED_NOAE decision", "",
         f"**Verdict:** `{res['verdict']}`", "",
         f"- Selected: epoch **{sel['epoch']}** — `{sel['checkpoint']}`",
         f"- Baseline: epoch 13 — `{b['checkpoint']}`",
         f"- Ranking: worst-anchor mean(vehicle_f1, person_f1) -> mean XY MAE -> mean duplicate FP/frame",
         f"- Worst anchor: q={sel['worst_anchor_q']}, mean F1 {sel['worst_anchor_mean_f1']:.4f} "
         f"(baseline at its own worst anchor: {sel['baseline_worst_anchor_mean_f1']:.4f})",
         f"- Evaluation completeness: **{'COMPLETE' if res['completeness']['complete'] else 'INCOMPLETE'}**",
         "",
         "## Per-q metrics, vehicle and person separately",
         "",
         "Every candidate value is compared against the epoch-13 baseline decoded at the **same q**.",
         "`A` = registered training anchor, `M` = unseen interval midpoint (reported, never ranked).",
         "",
         "| q | kind | veh F1 | veh F1 vs base | per F1 | per F1 vs base | veh rec | per rec | "
         "mean F1 | XY MAE m | dupFP/fr |",
         "|---|------|--------|----------------|--------|----------------|---------|---------|"
         "---------|----------|----------|"]
    for tag in sorted(sel["per_q"], key=lambda t: sel["per_q"][t]["q"]):
        e = sel["per_q"][tag]
        d = e.get("delta_vs_baseline_at_q")
        kind = "A" if e["is_registered_anchor"] else "M"
        def f(x, n=4):
            return "n/a" if x is None else f"{x:.{n}f}"
        def sgn(x):
            return "n/a" if x is None else f"{x:+.4f}"
        L.append(
            f"| {e['q']:.2f} | {kind} | {f(e['vehicle_f1'])} | {sgn(d and d['vehicle_f1'])} | "
            f"{f(e['person_f1'])} | {sgn(d and d['person_f1'])} | {f(e['vehicle_recall'])} | "
            f"{f(e['person_recall'])} | {f(e['mean_f1'])} | {f(e['overall_xy_mae_m'])} | "
            f"{f(e['duplicate_fp_per_frame'])} |")
    L += ["", "## Clean q=0.00 guards (vs matched baseline)", "",
          "| metric | selected | baseline | delta |", "|---|---|---|---|"]
    for k, bk in (("clean_vehicle_f1", "clean_vehicle_f1"), ("clean_person_f1", "clean_person_f1"),
                  ("clean_vehicle_recall", "clean_vehicle_recall"),
                  ("clean_person_recall", "clean_person_recall"),
                  ("clean_xy_mae_m", "clean_xy_mae_m"),
                  ("clean_dimension_mae_m", "clean_dimension_mae_m"),
                  ("clean_miou", "clean_miou"), ("clean_vehicle_iou", "clean_vehicle_iou"),
                  ("clean_person_iou", "clean_person_iou")):
        sv, bv = sel.get(k), b.get(bk)
        dv = "n/a" if (sv is None or bv is None) else f"{sv - bv:+.4f}"
        L.append(f"| {k} | {'n/a' if sv is None else f'{sv:.4f}'} | "
                 f"{'n/a' if bv is None else f'{bv:.4f}'} | {dv} |")
    L += ["", "## Advisory service targets (visible, never gating)", "",
          "| target | value | threshold | met |", "|---|---|---|---|"]
    for k, v in res.get("advisory_targets", {}).items():
        L.append(f"| {k} | {v['value']:.4f} | {v['target']:.2f} | {'yes' if v['met'] else 'no'} |")
    L += ["", "## Exclusions", "",
          "- Aborted pure-categorical run isolated at "
          "`_ABORTED_pure_categorical_20260826_focused_noae_v1/` and excluded from selection.",
          f"- Candidates dropped for incomplete anchor coverage: "
          f"{res.get('dropped_for_incomplete_anchor_coverage', [])}", ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
