#!/usr/bin/env python3
"""Gates and selection for the hybrid noAE pilot.

Three stages, each reading only decoded artifacts (never a training log):

``parity``  Phase C part 2. The warm start's decoded validation metrics must
            reproduce the frozen baseline inside the tolerances registered in
            ``HYBRID_NOAE_PILOT_PLAN.md``.
``early``   The six-epoch early continuation gate.
``final``   Selection over the evaluated epochs plus the service-target report.

Every threshold is a module constant. None is an argument, so none can be
loosened from a command line after a number is known.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------------------------------------------- registered
PARITY_TOLERANCE = {
    "precision": 0.005, "recall": 0.005, "f1": 0.005,
    "xy_mae_m": 0.01, "iou": 0.002,
}

EARLY_GATE = {
    "recall_gain_at_002": 0.05,      # both classes, absolute, >= baseline + this
    "precision_drop_at_020": 0.03,   # both classes, absolute, <= this
    "miou_drop": 0.02,               # <= this
}

SERVICE_TARGETS = {
    "vehicle_recall": 0.85, "person_recall": 0.80,
    "vehicle_precision": 0.80, "person_precision": 0.80,
    "vehicle_xy_mae_m": 1.0, "person_xy_mae_m": 1.2,
    "vehicle_iou": 0.85, "person_iou": 0.50, "miou": 0.80,
}
SERVICE_LOWER_IS_BETTER = {"vehicle_xy_mae_m", "person_xy_mae_m"}


# ------------------------------------------------------------------- loading
def _seg_metrics(eval_dir: Path) -> Dict[str, float]:
    evaluator = json.loads((eval_dir / "evaluator_metrics.json").read_text(encoding="utf-8"))
    return {key: float(evaluator[key]) for key in ("miou", "vehicle_iou", "person_iou")
            if key in evaluator}


def load_eval(eval_dir: Path) -> Dict[str, Any]:
    derived = json.loads((eval_dir / "derived_metrics.json").read_text(encoding="utf-8"))
    primary = derived["primary"]
    record: Dict[str, Any] = {
        "tag": derived.get("tag", eval_dir.name),
        "checkpoint": derived.get("checkpoint", ""),
        "score_threshold": float(derived["fixed_decoder"]["object_score_threshold"]),
        "frames": int(primary.get("frames", 0)),
    }
    for cls in ("vehicle", "person"):
        for metric in ("precision", "recall", "f1", "xy_mae_m"):
            record[f"{cls}_{metric}"] = float(primary[f"{cls}_{metric}"])
        record[f"{cls}_tp"] = int(primary[f"{cls}_tp"])
        record[f"{cls}_fp"] = int(primary[f"{cls}_fp"])
        record[f"{cls}_fn"] = int(primary[f"{cls}_fn"])
        record[f"{cls}_duplicate_fp_fraction"] = float(primary[f"{cls}_duplicate_fp_fraction"])
    record.update(_seg_metrics(eval_dir))
    return record


def _find(root: Path, tag: str) -> Path:
    path = root / "eval" / tag
    if not (path / "derived_metrics.json").is_file():
        raise FileNotFoundError(f"missing decoded evaluation: {path}")
    return path


# -------------------------------------------------------------------- stages
def stage_parity(exp_dir: Path, baseline_dir: Path, baseline_tag: str) -> Dict[str, Any]:
    base20 = load_eval(_find(baseline_dir, baseline_tag))
    base02 = load_eval(_find(exp_dir, "baseline_epoch013_s002"))
    warm20 = load_eval(_find(exp_dir, "warm_start_s020"))
    warm02 = load_eval(_find(exp_dir, "warm_start_s002"))

    failures: List[str] = []
    comparisons: List[Dict[str, Any]] = []

    def compare(name: str, hybrid_value: float, base_value: float, tolerance: float) -> None:
        delta = float(hybrid_value) - float(base_value)
        ok = abs(delta) <= tolerance
        comparisons.append({"metric": name, "warm_start": hybrid_value, "baseline": base_value,
                            "delta": delta, "tolerance": tolerance, "ok": ok})
        if not ok:
            failures.append(f"{name}: |{delta:+.4f}| > {tolerance}")

    for cls in ("vehicle", "person"):
        for metric, tolerance in (("precision", PARITY_TOLERANCE["precision"]),
                                  ("recall", PARITY_TOLERANCE["recall"]),
                                  ("f1", PARITY_TOLERANCE["f1"]),
                                  ("xy_mae_m", PARITY_TOLERANCE["xy_mae_m"])):
            compare(f"s020.{cls}_{metric}", warm20[f"{cls}_{metric}"],
                    base20[f"{cls}_{metric}"], tolerance)
        compare(f"s002.{cls}_recall", warm02[f"{cls}_recall"], base02[f"{cls}_recall"],
                PARITY_TOLERANCE["recall"])
    for metric in ("miou", "vehicle_iou", "person_iou"):
        compare(f"s020.{metric}", warm20[metric], base20[metric], PARITY_TOLERANCE["iou"])

    return {
        "stage": "parity",
        "status": "PASS" if not failures else "WARM_START_PARITY_FAILED",
        "failures": failures,
        "tolerances": PARITY_TOLERANCE,
        "baseline_s020": base20,
        "baseline_s002": base02,
        "warm_start_s020": warm20,
        "warm_start_s002": warm02,
        "comparisons": comparisons,
    }


def stage_early(exp_dir: Path, baseline_dir: Path, baseline_tag: str) -> Dict[str, Any]:
    base20 = load_eval(_find(baseline_dir, baseline_tag))
    base02 = load_eval(_find(exp_dir, "baseline_epoch013_s002"))
    hyb20 = load_eval(_find(exp_dir, "hybrid_ep006_s020"))
    hyb02 = load_eval(_find(exp_dir, "hybrid_ep006_s002"))

    checks: List[Dict[str, Any]] = []

    def check(name: str, value: float, threshold: float, ok: bool, detail: str) -> None:
        checks.append({"criterion": name, "value": value, "threshold": threshold,
                       "ok": bool(ok), "detail": detail})

    for cls in ("vehicle", "person"):
        gain = hyb02[f"{cls}_recall"] - base02[f"{cls}_recall"]
        check(f"s002_{cls}_recall_gain", gain, EARLY_GATE["recall_gain_at_002"],
              gain >= EARLY_GATE["recall_gain_at_002"],
              f"{base02[f'{cls}_recall']:.4f} -> {hyb02[f'{cls}_recall']:.4f}")
    for cls in ("vehicle", "person"):
        drop = base20[f"{cls}_precision"] - hyb20[f"{cls}_precision"]
        check(f"s020_{cls}_precision_drop", drop, EARLY_GATE["precision_drop_at_020"],
              drop <= EARLY_GATE["precision_drop_at_020"],
              f"{base20[f'{cls}_precision']:.4f} -> {hyb20[f'{cls}_precision']:.4f}")
    miou_drop = base20["miou"] - hyb20["miou"]
    check("miou_drop", miou_drop, EARLY_GATE["miou_drop"], miou_drop <= EARLY_GATE["miou_drop"],
          f"{base20['miou']:.4f} -> {hyb20['miou']:.4f}")

    finite = all(math.isfinite(float(v)) for record in (hyb20, hyb02)
                 for v in record.values() if isinstance(v, float))
    check("no_nan_or_collapse", 1.0 if finite else 0.0, 1.0, finite,
          "all decoded epoch-6 metrics are finite")

    passed = all(item["ok"] for item in checks)
    return {
        "stage": "early_continuation_gate",
        "status": "PASS" if passed else "HYBRID_NOAE_PILOT_NO_GAIN",
        "thresholds": EARLY_GATE,
        "checks": checks,
        "baseline_s020": base20, "baseline_s002": base02,
        "epoch6_s020": hyb20, "epoch6_s002": hyb02,
    }


def _selection_key(record20: Dict[str, Any]) -> Tuple[float, float, float]:
    mean_f1 = 0.5 * (record20["vehicle_f1"] + record20["person_f1"])
    min_recall = min(record20["vehicle_recall"], record20["person_recall"])
    mean_xy = 0.5 * (record20["vehicle_xy_mae_m"] + record20["person_xy_mae_m"])
    return (mean_f1, min_recall, -mean_xy)


def stage_final(exp_dir: Path, baseline_dir: Path, baseline_tag: str,
                epochs: List[int]) -> Dict[str, Any]:
    base20 = load_eval(_find(baseline_dir, baseline_tag))
    base02 = load_eval(_find(exp_dir, "baseline_epoch013_s002"))

    rows: List[Dict[str, Any]] = []
    for epoch in epochs:
        record20 = load_eval(_find(exp_dir, f"hybrid_ep{epoch:03d}_s020"))
        record02 = load_eval(_find(exp_dir, f"hybrid_ep{epoch:03d}_s002"))
        mean_f1, min_recall, neg_xy = _selection_key(record20)
        rows.append({
            "epoch": epoch, "s020": record20, "s002": record02,
            "mean_f1": mean_f1, "min_class_recall": min_recall, "mean_xy_mae_m": -neg_xy,
        })
    best = max(rows, key=lambda row: _selection_key(row["s020"]))

    service: Dict[str, Any] = {}
    for name, target in SERVICE_TARGETS.items():
        value = float(best["s020"][name])
        met = value <= target if name in SERVICE_LOWER_IS_BETTER else value >= target
        service[name] = {"value": value, "target": target,
                         "direction": "<=" if name in SERVICE_LOWER_IS_BETTER else ">=",
                         "met": bool(met)}
    service_ready = all(item["met"] for item in service.values())

    improved = (
        best["s020"]["vehicle_f1"] > base20["vehicle_f1"]
        and best["s020"]["person_f1"] > base20["person_f1"]
    )
    verdict = ("HYBRID_NOAE_SERVICE_READY" if service_ready
               else ("HYBRID_NOAE_IMPROVED_NOT_SERVICE_READY" if improved
                     else "HYBRID_NOAE_PILOT_NO_GAIN"))
    return {
        "stage": "final_selection",
        "verdict": verdict,
        "selection_rule": "highest mean(vehicle_f1, person_f1); tie -> higher min class recall; "
                          "tie -> lower mean XY MAE; all at score 0.20",
        "selected_epoch": best["epoch"],
        "selected_checkpoint": best["s020"]["checkpoint"],
        "service_targets": service,
        "service_ready": service_ready,
        "baseline_s020": base20, "baseline_s002": base02,
        "epochs": rows,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--baseline-tag", default="curriculum_stage2_joint_v1_epoch_013")
    parser.add_argument("--stage", required=True, choices=("parity", "early", "final"))
    parser.add_argument("--epochs", default="6,10,14,18,22,24")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    exp_dir = args.experiment_dir.resolve()
    baseline_dir = args.baseline_dir.resolve()
    if args.stage == "parity":
        result = stage_parity(exp_dir, baseline_dir, args.baseline_tag)
    elif args.stage == "early":
        result = stage_early(exp_dir, baseline_dir, args.baseline_tag)
    else:
        result = stage_final(exp_dir, baseline_dir, args.baseline_tag,
                             [int(v) for v in args.epochs.split(",") if v.strip()])

    output = args.output or (exp_dir / f"gate_{args.stage}_v1.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result.get("status", "PASS") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
