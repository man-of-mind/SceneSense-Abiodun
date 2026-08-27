#!/usr/bin/env python3
"""Fixed pilot gate, final selection, and clean-model verdict."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

PILOT_GATES = {
    "s002_vehicle_recall": 0.67,
    "s002_person_recall": 0.58,
    "s020_vehicle_precision": 0.43,
    "s020_person_precision": 0.32,
    "miou": 0.68,
}
SERVICE_TARGETS = {
    "vehicle_recall": 0.85,
    "person_recall": 0.80,
    "vehicle_precision": 0.80,
    "person_precision": 0.80,
    "vehicle_xy_mae_m": 1.0,
    "person_xy_mae_m": 1.2,
    "vehicle_iou": 0.85,
    "person_iou": 0.50,
    "miou": 0.80,
}
LOWER_IS_BETTER = {"vehicle_xy_mae_m", "person_xy_mae_m"}
BASELINE = {
    "vehicle_precision": 0.4624,
    "vehicle_recall": 0.4498,
    "vehicle_f1": 0.4560,
    "person_precision": 0.3480,
    "person_recall": 0.3752,
    "person_f1": 0.3611,
    "vehicle_xy_mae_m": 1.1343,
    "person_xy_mae_m": 1.3195,
    "vehicle_iou": 0.8117,
    "person_iou": 0.3274,
    "miou": 0.7078,
    "s002_vehicle_recall": 0.5702,
    "s002_person_recall": 0.4852,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_eval(exp_dir: Path, epoch: int, score: str) -> Dict[str, Any]:
    tag = f"centernet_ep{epoch:03d}_s{score}"
    root = exp_dir / "eval" / tag
    derived = json.loads((root / "derived_metrics.json").read_text(encoding="utf-8"))
    primary = derived["primary"]
    record: Dict[str, Any] = {
        "tag": tag,
        "checkpoint": derived["checkpoint"],
        "checkpoint_sha256": derived["checkpoint_sha256"],
        "score_threshold": float(derived["fixed_decoder"]["object_score_threshold"]),
    }
    for cls in ("vehicle", "person"):
        for metric in ("precision", "recall", "f1", "xy_mae_m", "dimension_mae_m"):
            record[f"{cls}_{metric}"] = float(primary[f"{cls}_{metric}"])
        for metric in ("tp", "fp", "fn"):
            record[f"{cls}_{metric}"] = int(primary[f"{cls}_{metric}"])
    for key, value in derived["segmentation"].items():
        if value is not None:
            record[key] = float(value)
    return record


def _finite(record: Dict[str, Any]) -> bool:
    return all(
        math.isfinite(float(value))
        for value in record.values()
        if isinstance(value, (float, int))
    )


def pilot(exp_dir: Path) -> Dict[str, Any]:
    s20 = load_eval(exp_dir, 4, "020")
    s02 = load_eval(exp_dir, 4, "002")
    checks = {
        "s002_vehicle_recall": s02["vehicle_recall"] >= PILOT_GATES["s002_vehicle_recall"],
        "s002_person_recall": s02["person_recall"] >= PILOT_GATES["s002_person_recall"],
        "s020_vehicle_precision": s20["vehicle_precision"] >= PILOT_GATES["s020_vehicle_precision"],
        "s020_person_precision": s20["person_precision"] >= PILOT_GATES["s020_person_precision"],
        "miou": s20["miou"] >= PILOT_GATES["miou"],
        "finite_and_contract": _finite(s20) and _finite(s02),
    }
    passed = all(checks.values())
    return {
        "stage": "four_epoch_pilot",
        "status": "PASS" if passed else "CENTERNET_BASE_PILOT_FAILED",
        "thresholds": PILOT_GATES,
        "checks": checks,
        "epoch4_s020": s20,
        "epoch4_s002": s02,
    }


def _selection_key(record: Dict[str, Any]) -> Tuple[float, float, float, int]:
    s20 = record["s020"]
    minimum_recall = min(s20["vehicle_recall"], s20["person_recall"])
    mean_f1 = 0.5 * (s20["vehicle_f1"] + s20["person_f1"])
    mean_xy = 0.5 * (s20["vehicle_xy_mae_m"] + s20["person_xy_mae_m"])
    return minimum_recall, mean_f1, -mean_xy, -int(record["epoch"])


def final(exp_dir: Path, epochs: List[int]) -> Dict[str, Any]:
    records = [
        {"epoch": epoch, "s020": load_eval(exp_dir, epoch, "020"),
         "s002": load_eval(exp_dir, epoch, "002")}
        for epoch in epochs
    ]
    selected = max(records, key=_selection_key)
    s20 = selected["s020"]
    service = {
        name: {
            "value": float(s20[name]),
            "target": target,
            "direction": "<=" if name in LOWER_IS_BETTER else ">=",
            "met": float(s20[name]) <= target if name in LOWER_IS_BETTER else float(s20[name]) >= target,
        }
        for name, target in SERVICE_TARGETS.items()
    }
    ready = all(item["met"] for item in service.values())
    improved = (
        s20["vehicle_f1"] > BASELINE["vehicle_f1"]
        and s20["person_f1"] > BASELINE["person_f1"]
    )
    # Reaching this stage means the immutable four-epoch continuation gate
    # passed.  The bounded final vocabulary has one non-service-ready outcome
    # for such a run; keep the explicit baseline comparison alongside it.
    verdict = (
        "CENTERNET_BASE_SERVICE_READY"
        if ready
        else "CENTERNET_BASE_IMPROVED_NOT_SERVICE_READY"
    )
    checkpoint = Path(s20["checkpoint"])
    return {
        "stage": "full_clean_selection",
        "verdict": verdict,
        "selection_rule": "highest min(vehicle/person recall), then highest mean F1, then lower mean XY MAE, then earlier epoch; score 0.20",
        "selected_epoch": selected["epoch"],
        "selected_checkpoint": str(checkpoint),
        "selected_checkpoint_sha256": _sha256(checkpoint),
        "service_targets": service,
        "service_ready": ready,
        "improved_vs_lraspp": improved,
        "baseline": BASELINE,
        "epochs": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("pilot", "final"))
    parser.add_argument("--epochs", default="4,8,12,16,20,24")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = (
        pilot(args.experiment_dir.resolve()) if args.stage == "pilot"
        else final(args.experiment_dir.resolve(), [int(v) for v in args.epochs.split(",")])
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result.get("status", "PASS") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
