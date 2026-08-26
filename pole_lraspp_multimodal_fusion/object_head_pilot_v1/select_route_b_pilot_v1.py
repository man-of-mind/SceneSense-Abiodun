#!/usr/bin/env python3
"""Precision-aware epoch selection and the three-way pilot decision.

Selection is post-hoc and never touches the differentiable training loss. Every
saved epoch of both arms is evaluated on the same Route B validation view with
the same fixed decoder; this script then applies feasibility first and only
ranks among the feasible epochs.

Feasibility (versus the frozen old noAE baseline, evaluated on the same view):

* vehicle and person recall may not fall by more than 0.02 absolute;
* vehicle XY MAE may not worsen by more than 0.05 m;
* person XY MAE may not worsen by more than 0.10 m;
* vehicle and person segmentation IoU may not fall by more than 0.02;
* dimension MAE may not worsen by more than 10 %.

Among feasible epochs: maximise vehicle precision, then minimise the duplicate-FP
fraction, then use vehicle XY MAE as the final tie-break.

The ``loc_dim_loss`` selection that the trainer itself made is recorded for
comparison but does not determine the promoted checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

GUARDS = {
    "vehicle_recall_max_drop": 0.02,
    "person_recall_max_drop": 0.02,
    "vehicle_xy_mae_max_increase_m": 0.05,
    "person_xy_mae_max_increase_m": 0.10,
    "vehicle_iou_max_drop": 0.02,
    "person_iou_max_drop": 0.02,
    "dimension_mae_max_relative_increase": 0.10,
}


def load_eval(exp_dir: Path, tag: str) -> Optional[Dict[str, Any]]:
    root = exp_dir / "eval" / tag
    derived_path, metrics_path = root / "derived_metrics.json", root / "evaluator_metrics.json"
    if not derived_path.is_file() or not metrics_path.is_file():
        return None
    derived = json.loads(derived_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    primary = derived["primary"]
    return {
        "tag": tag,
        "checkpoint": derived["checkpoint"],
        "vehicle_precision": primary["vehicle_precision"],
        "vehicle_recall": primary["vehicle_recall"],
        "vehicle_f1": primary["vehicle_f1"],
        "person_precision": primary["person_precision"],
        "person_recall": primary["person_recall"],
        "person_f1": primary["person_f1"],
        "overall_precision": primary["overall_precision"],
        "overall_recall": primary["overall_recall"],
        "overall_f1": primary["overall_f1"],
        "fp_per_frame": primary["overall_fp_per_frame"],
        "duplicate_fp_fraction": primary["overall_duplicate_fp_fraction"],
        "vehicle_duplicate_fp_fraction": primary["vehicle_duplicate_fp_fraction"],
        "person_duplicate_fp_fraction": primary["person_duplicate_fp_fraction"],
        "vehicle_xy_mae_m": primary["vehicle_xy_mae_m"],
        "person_xy_mae_m": primary["person_xy_mae_m"],
        "overall_xy_mae_m": primary["overall_xy_mae_m"],
        "dimension_mae_m": primary["overall_dimension_mae_m"],
        "centroid_2d_error_px": primary["overall_centroid_2d_error_px"],
        "vehicle_iou": metrics["vehicle_iou"],
        "person_iou": metrics["person_iou"],
        "miou": metrics["miou"],
        "collision_window_excluded": derived["collision_window_excluded"],
        "collision_window_frames_excluded": derived["collision_window_frames_excluded"],
    }


def feasibility(candidate: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
    checks = {
        "vehicle_recall": (
            candidate["vehicle_recall"] >= base["vehicle_recall"] - GUARDS["vehicle_recall_max_drop"],
            candidate["vehicle_recall"] - base["vehicle_recall"],
        ),
        "person_recall": (
            candidate["person_recall"] >= base["person_recall"] - GUARDS["person_recall_max_drop"],
            candidate["person_recall"] - base["person_recall"],
        ),
        "vehicle_xy_mae_m": (
            candidate["vehicle_xy_mae_m"] <= base["vehicle_xy_mae_m"] + GUARDS["vehicle_xy_mae_max_increase_m"],
            candidate["vehicle_xy_mae_m"] - base["vehicle_xy_mae_m"],
        ),
        "person_xy_mae_m": (
            candidate["person_xy_mae_m"] <= base["person_xy_mae_m"] + GUARDS["person_xy_mae_max_increase_m"],
            candidate["person_xy_mae_m"] - base["person_xy_mae_m"],
        ),
        "vehicle_iou": (
            candidate["vehicle_iou"] >= base["vehicle_iou"] - GUARDS["vehicle_iou_max_drop"],
            candidate["vehicle_iou"] - base["vehicle_iou"],
        ),
        "person_iou": (
            candidate["person_iou"] >= base["person_iou"] - GUARDS["person_iou_max_drop"],
            candidate["person_iou"] - base["person_iou"],
        ),
        "dimension_mae_m": (
            candidate["dimension_mae_m"]
            <= base["dimension_mae_m"] * (1.0 + GUARDS["dimension_mae_max_relative_increase"]),
            candidate["dimension_mae_m"] / base["dimension_mae_m"] - 1.0
            if base["dimension_mae_m"] else float("nan"),
        ),
    }
    return {
        "feasible": all(passed for passed, _ in checks.values()),
        "checks": {name: {"pass": bool(passed), "delta": float(delta)} for name, (passed, delta) in checks.items()},
        "failed": [name for name, (passed, _) in checks.items() if not passed],
    }


def rank_key(row: Dict[str, Any]) -> Tuple[float, float, float]:
    # maximise vehicle precision, then minimise duplicate-FP fraction, then minimise vehicle XY MAE
    return (-row["vehicle_precision"], row["duplicate_fp_fraction"], row["vehicle_xy_mae_m"])


def loc_dim_selection(exp_dir: Path, trial: str) -> Dict[str, Any]:
    path = exp_dir / "metrics" / f"{trial}_metrics.csv"
    rows = list(csv.DictReader(path.open("r", newline="", encoding="utf-8")))
    best = max(rows, key=lambda row: float(row["selection_score"]))
    return {
        "epoch": int(best["epoch"]),
        "selection_score": float(best["selection_score"]),
        "loc_loss": float(best["loc_loss"]),
        "dim_loss": float(best["dim_loss"]),
        "criterion": "selection_score_mode=loc_dim_loss  ->  -(loc_loss + 0.25 * dim_loss)",
        "per_epoch": [
            {
                "epoch": int(row["epoch"]),
                "train_loss": float(row["train_loss"]),
                "val_loss": float(row["val_loss"]),
                "selection_score": float(row["selection_score"]),
                "loc_loss": float(row["loc_loss"]),
                "dim_loss": float(row["dim_loss"]),
                "miou": float(row["miou"]),
                "epoch_seconds": float(row.get("epoch_seconds", "nan") or "nan"),
                "cuda_max_memory_allocated_mib": float(row.get("cuda_max_memory_allocated_mib", "nan") or "nan"),
                "cuda_max_memory_reserved_mib": float(row.get("cuda_max_memory_reserved_mib", "nan") or "nan"),
            }
            for row in rows
        ],
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--baseline-tag", default="baseline_frozen_noae")
    parser.add_argument("--control-trial", default="pilotA_control_objhead_smoke_v1")
    parser.add_argument("--candidate-trial", default="pilotB_capped_objhead_smoke_v1")
    parser.add_argument("--epochs", type=int, default=6)
    args = parser.parse_args(argv)

    exp_dir = args.experiment_dir.resolve()
    base = load_eval(exp_dir, args.baseline_tag)
    if base is None:
        raise SystemExit(f"missing baseline evaluation {args.baseline_tag}")

    arms: Dict[str, Any] = {}
    for role, trial in (("control", args.control_trial), ("candidate", args.candidate_trial)):
        epochs = []
        for epoch in range(args.epochs):
            row = load_eval(exp_dir, f"{trial}_epoch_{epoch:03d}")
            if row is None:
                continue
            row["epoch"] = epoch
            row["feasibility"] = feasibility(row, base)
            epochs.append(row)
        feasible = [row for row in epochs if row["feasibility"]["feasible"]]
        selected = min(feasible, key=rank_key) if feasible else None
        arms[role] = {
            "trial": trial,
            "epochs": epochs,
            "feasible_epochs": [row["epoch"] for row in feasible],
            "precision_aware_selection": (
                {
                    "epoch": selected["epoch"],
                    "tag": selected["tag"],
                    "checkpoint": selected["checkpoint"],
                    "vehicle_precision": selected["vehicle_precision"],
                    "duplicate_fp_fraction": selected["duplicate_fp_fraction"],
                    "vehicle_xy_mae_m": selected["vehicle_xy_mae_m"],
                }
                if selected else None
            ),
            "loc_dim_loss_selection": loc_dim_selection(exp_dir, trial),
        }

    control = arms["control"]["precision_aware_selection"]
    candidate = arms["candidate"]["precision_aware_selection"]
    control_row = next((row for row in arms["control"]["epochs"]
                        if control and row["epoch"] == control["epoch"]), None)
    candidate_row = next((row for row in arms["candidate"]["epochs"]
                          if candidate and row["epoch"] == candidate["epoch"]), None)

    reasons: List[str] = []
    if candidate_row is None:
        decision = "NO_PILOT_ADVANCES" if control_row is None else "CONTROL_ONLY_ADVANCES"
        reasons.append("no capped-arm epoch satisfied every non-inferiority guard")
        if control_row is not None:
            reasons.append("a control epoch is feasible, so Route B retraining itself is viable")
    else:
        cap_beats_control_precision = candidate_row["vehicle_precision"] > (
            control_row["vehicle_precision"] if control_row else base["vehicle_precision"]
        )
        cap_beats_control_duplicates = candidate_row["duplicate_fp_fraction"] < (
            control_row["duplicate_fp_fraction"] if control_row else base["duplicate_fp_fraction"]
        )
        cap_beats_base_precision = candidate_row["vehicle_precision"] > base["vehicle_precision"]
        cap_beats_base_duplicates = candidate_row["duplicate_fp_fraction"] < base["duplicate_fp_fraction"]
        if (cap_beats_control_precision and cap_beats_control_duplicates
                and cap_beats_base_precision and cap_beats_base_duplicates):
            decision = "CAP_CANDIDATE_ADVANCES"
            reasons.append("the capped candidate improves vehicle precision and reduces the "
                           "duplicate-FP fraction versus both the frozen baseline and the "
                           "retrained control, under every non-inferiority guard")
        elif control_row is not None and (
            control_row["vehicle_precision"] > base["vehicle_precision"]
            or control_row["duplicate_fp_fraction"] < base["duplicate_fp_fraction"]
        ):
            decision = "CONTROL_ONLY_ADVANCES"
            reasons.append("Route B retraining helps, but the cap adds no further benefit "
                           "on vehicle precision and duplicate-FP fraction together")
        else:
            decision = "NO_PILOT_ADVANCES"
            reasons.append("neither arm improves on the frozen baseline under the registered criteria")

    report = {
        "experiment_dir": str(exp_dir),
        "guards": GUARDS,
        "ranking": "feasibility first; then max vehicle precision, min duplicate-FP fraction, min vehicle XY MAE",
        "baseline": base,
        "arms": arms,
        "three_way_comparison": {
            "frozen_old_noae": base,
            "retrained_control": control_row,
            "capped_candidate": candidate_row,
        },
        "terminal_decision": decision,
        "decision_reasons": reasons,
        "scope": "pilot decision only; not deployment approval",
    }
    out_dir = exp_dir / "decision"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pilot_decision_v1.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
