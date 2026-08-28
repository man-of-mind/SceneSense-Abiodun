#!/usr/bin/env python3
"""Phase C: fixed validation evaluation of the class-balanced continuation.

Evaluates exactly epochs 4, 8 and 12 under the preregistered arms, applies the
registered non-regression / material-gain contract against the epoch-20 baseline,
and emits the terminal verdict. No calibration, no threshold selection, no test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
BASE_PKG = ROOT / "pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_clean_base_v1"
for path in (str(ROOT), str(BASE_PKG)):
    if path not in sys.path:
        sys.path.insert(0, path)

from score_contract_v1 import score_segmentation  # noqa: E402  (frozen v3.1 scorer, unchanged)

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.route_b_v3_1_targeted_refinement_v1.audit_v1 import (  # noqa: E402
    CLASSES, load_gt, load_predictions, read_csv, score_arm, sha256,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.route_b_v3_1_targeted_refinement_v1.postprocess_v1 import (  # noqa: E402
    apply_arm,
)

EPOCHS = (4, 8, 12)
TRIAL_NAME = "route_b_v3_1_targeted_refinement_cb_v1"
PRIMARY = "v010"
SENSITIVITY = "v025"

BASELINE = {
    "epoch": 20,
    "checkpoint_sha256": "88b34a69eeec7bf2f6444e70a0e346c365b979e6936d277cb0c75e8cd747aa1d",
    "vehicle_f1": 0.571607411116675, "person_f1": 0.43010752688172044,
    "mean_class_f1": 0.5008574689991977,
    "vehicle_recall": 0.70426735218509, "person_recall": 0.40289256198347106,
    "vehicle_recall_002": 0.7461182519280206, "person_recall_002": 0.45609504132231404,
    "vehicle_xy_mae_m": 0.9980633837411443, "person_xy_mae_m": 1.4087968615412807,
    "vehicle_precision": 0.4810028794156893, "person_precision": 0.4612655233589592,
    "foreground_miou": 0.6513838982870985,
    "vehicle_iou": 0.8639556911420896, "person_box_mask_iou": 0.4388121054321076,
}

SERVICE_TARGETS = {
    "vehicle_precision_ge_0_80": lambda m: m["vehicle_precision"] >= 0.80,
    "vehicle_recall_ge_0_85": lambda m: m["vehicle_recall"] >= 0.85,
    "person_precision_ge_0_80": lambda m: m["person_precision"] >= 0.80,
    "person_recall_ge_0_80": lambda m: m["person_recall"] >= 0.80,
    "vehicle_xy_mae_le_1_0m": lambda m: m["vehicle_xy_mae_m"] <= 1.0,
    "person_xy_mae_le_1_2m": lambda m: m["person_xy_mae_m"] <= 1.2,
    "vehicle_iou_ge_0_85": lambda m: m["vehicle_iou"] >= 0.85,
    "person_box_mask_iou_ge_0_50": lambda m: m["person_box_mask_iou"] >= 0.50,
    "foreground_miou_ge_0_675": lambda m: m["foreground_miou"] >= 0.675,
}


def write_json_x(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def flatten(primary_020: dict, primary_002: dict, segmentation: dict) -> dict[str, Any]:
    vehicle, person = primary_020["classes"]["vehicle"], primary_020["classes"]["person"]
    return {
        "vehicle_tp": vehicle["tp"], "vehicle_fp": vehicle["fp"], "vehicle_fn": vehicle["fn"],
        "person_tp": person["tp"], "person_fp": person["fp"], "person_fn": person["fn"],
        "vehicle_precision": vehicle["precision"], "vehicle_recall": vehicle["recall"],
        "vehicle_f1": vehicle["f1"], "vehicle_xy_mae_m": vehicle["xy_mae_m"],
        "person_precision": person["precision"], "person_recall": person["recall"],
        "person_f1": person["f1"], "person_xy_mae_m": person["xy_mae_m"],
        "vehicle_recall_002": primary_002["classes"]["vehicle"]["recall"],
        "person_recall_002": primary_002["classes"]["person"]["recall"],
        "mean_class_f1": (vehicle["f1"] + person["f1"]) / 2.0,
        "minimum_class_recall": min(vehicle["recall"], person["recall"]),
        "mean_xy_mae_m": (vehicle["xy_mae_m"] + person["xy_mae_m"]) / 2.0,
        "foreground_miou": segmentation["foreground_miou"],
        "vehicle_iou": segmentation["vehicle_iou"],
        "person_box_mask_iou": segmentation["person_box_mask_iou"],
    }


def non_regression(metrics: dict) -> dict[str, bool]:
    return {
        "vehicle_f1_delta_ge_-0.005": (metrics["vehicle_f1"] - BASELINE["vehicle_f1"]) >= -0.005,
        "person_f1_delta_ge_-0.005": (metrics["person_f1"] - BASELINE["person_f1"]) >= -0.005,
        "vehicle_recall_delta_ge_-0.01": (metrics["vehicle_recall"] - BASELINE["vehicle_recall"]) >= -0.01,
        "person_recall_002_delta_ge_0": (metrics["person_recall_002"] - BASELINE["person_recall_002"]) >= 0.0,
        "foreground_miou_delta_ge_-0.005": (metrics["foreground_miou"] - BASELINE["foreground_miou"]) >= -0.005,
        "vehicle_xy_mae_increase_le_0.05": (metrics["vehicle_xy_mae_m"] - BASELINE["vehicle_xy_mae_m"]) <= 0.05,
        "person_xy_mae_increase_le_0.05": (metrics["person_xy_mae_m"] - BASELINE["person_xy_mae_m"]) <= 0.05,
    }


def material_gain(metrics: dict, guards: dict[str, bool]) -> dict[str, bool]:
    return {
        "mean_class_f1_ge_baseline_plus_0.02": metrics["mean_class_f1"] >= BASELINE["mean_class_f1"] + 0.02,
        "person_f1_ge_baseline_plus_0.02": metrics["person_f1"] >= BASELINE["person_f1"] + 0.02,
        "person_recall_002_ge_baseline_plus_0.05": metrics["person_recall_002"] >= BASELINE["person_recall_002"] + 0.05,
        "no_non_regression_failure": all(guards.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--infer-script", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    started = time.monotonic()

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    nms_eligible = audit["vehicle_world_nms_2m"]["verdict"] == "VEHICLE_WORLD_NMS_2M_ELIGIBLE"
    arms = ["RAW_FIXED_DECODER"] + (["VEHICLE_WORLD_NMS_2M"] if nms_eligible else [])

    checkpoint_dir = experiment / "checkpoints" / TRIAL_NAME
    manifest = read_csv(experiment / "dataset/manifest.csv")
    frame_ids = [row["sample_id"] for row in manifest if row["split"] == "val"]
    gt = {contract: load_gt(experiment, contract)[0] for contract in (PRIMARY, SENSITIVITY)}
    ignore_cache: dict[str, Any] = {}

    records: list[dict[str, Any]] = []
    for epoch in EPOCHS:
        checkpoint = checkpoint_dir / f"epoch_{epoch:03d}.pt"
        checkpoint_hash = sha256(checkpoint)
        tag = f"trained_epoch_{epoch:03d}"
        prediction_root = experiment / "predictions" / tag
        if not (prediction_root / "INFERENCE_COMPLETE").is_file():
            command = [
                sys.executable, str(args.infer_script.resolve()),
                "--experiment", str(experiment), "--checkpoint", str(checkpoint),
                "--checkpoint-sha256", checkpoint_hash, "--tag", tag,
            ]
            print(f"[phase C] inference epoch={epoch}", flush=True)
            if subprocess.run(command).returncode != 0:
                raise RuntimeError(f"inference failed for epoch {epoch}")
        inference = json.loads((prediction_root / "inference_manifest.json").read_text(encoding="utf-8"))
        if sha256(prediction_root / "detections.csv") != inference["detections_sha256"]:
            raise RuntimeError(f"detection hash drift: {tag}")
        if inference["checkpoint_sha256"] != checkpoint_hash:
            raise RuntimeError(f"checkpoint provenance mismatch: {tag}")

        predictions, missing = load_predictions(prediction_root / "detections.csv")
        if missing:
            raise RuntimeError(f"missing prediction fields: {tag}")
        print(f"[phase C] segmentation epoch={epoch}", flush=True)
        segmentation = score_segmentation(
            experiment, PRIMARY, frame_ids, prediction_root,
            prediction_root / "segmentation_manifest.csv",
        )
        record: dict[str, Any] = {
            "epoch": epoch, "tag": tag, "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "prediction_set_sha256": inference["prediction_set_sha256"],
            "peak_allocated_mib": inference["peak_allocated_mib"],
            "peak_reserved_mib": inference["peak_reserved_mib"],
            "arms": {},
        }
        for arm in arms:
            arm_predictions = apply_arm(predictions, arm)
            scored = {}
            for contract in (PRIMARY, SENSITIVITY):
                scored[contract] = {
                    f"{threshold:.2f}": score_arm(
                        experiment=experiment, contract=contract, frame_ids=frame_ids,
                        predictions=arm_predictions, gt=gt[contract], threshold=threshold,
                        ignore_cache=ignore_cache,
                    ) for threshold in (0.20, 0.02)
                }
            metrics = flatten(scored[PRIMARY]["0.20"], scored[PRIMARY]["0.02"], segmentation)
            guards = non_regression(metrics)
            gain = material_gain(metrics, guards)
            recall_guard_ok = True
            if arm == "VEHICLE_WORLD_NMS_2M":
                raw = record["arms"]["RAW_FIXED_DECODER"]["metrics"]
                recall_guard_ok = (raw["vehicle_recall"] - metrics["vehicle_recall"]) <= 0.01
            record["arms"][arm] = {
                "metrics": metrics,
                "primary_v010": scored[PRIMARY],
                "sensitivity_v025": scored[SENSITIVITY],
                "segmentation_v010": segmentation,
                "non_regression_guards": guards,
                "non_regression_pass": all(guards.values()),
                "material_gain_gates": gain,
                "material_gain_pass": all(gain.values()),
                "arm_recall_loss_guard_satisfied": recall_guard_ok,
                "selectable": recall_guard_ok,
                "service_targets": {name: bool(test(metrics)) for name, test in SERVICE_TARGETS.items()},
            }
            record["arms"][arm]["service_ready"] = all(record["arms"][arm]["service_targets"].values())
        records.append(record)

    candidates = [
        (record, arm, payload)
        for record in records for arm, payload in record["arms"].items()
        if payload["selectable"]
    ]
    ranked = sorted(
        candidates,
        key=lambda item: (
            -item[2]["metrics"]["mean_class_f1"],
            -item[2]["metrics"]["minimum_class_recall"],
            item[2]["metrics"]["mean_xy_mae_m"],
            -item[2]["metrics"]["foreground_miou"],
            item[0]["epoch"],
        ),
    )
    passing = [item for item in ranked if item[2]["material_gain_pass"]]
    selected = passing[0] if passing else None

    if selected is None:
        terminal = "LRASPP_TARGETED_REFINEMENT_NO_GAIN"
    elif selected[2]["service_ready"]:
        terminal = "LRASPP_TARGETED_REFINEMENT_SERVICE_READY"
    else:
        terminal = "LRASPP_TARGETED_REFINEMENT_MATERIAL_GAIN_NOT_SERVICE_READY"

    result: dict[str, Any] = {
        "schema": "route_b_v3_1_targeted_refinement_evaluation_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_epoch20": BASELINE,
        "arms_evaluated": arms,
        "vehicle_world_nms_2m_verdict": audit["vehicle_world_nms_2m"]["verdict"],
        "evaluated_epochs": list(EPOCHS),
        "records": records,
        "ranking": [
            {"epoch": record["epoch"], "arm": arm,
             "mean_class_f1": payload["metrics"]["mean_class_f1"],
             "material_gain_pass": payload["material_gain_pass"],
             "non_regression_pass": payload["non_regression_pass"]}
            for record, arm, payload in ranked
        ],
        "terminal": terminal,
        "wall_seconds": time.monotonic() - started,
    }
    if selected is not None:
        record, arm, payload = selected
        result["selected"] = {
            "epoch": record["epoch"], "arm": arm,
            "checkpoint": record["checkpoint"], "checkpoint_sha256": record["checkpoint_sha256"],
            "metrics": payload["metrics"],
            "deltas_vs_baseline": {
                key: payload["metrics"][key] - BASELINE[key]
                for key in payload["metrics"] if key in BASELINE
            },
            "service_targets": payload["service_targets"],
        }
    else:
        result["selected"] = None

    write_json_x(experiment / "PHASE_C_EVALUATION.json", result)
    (experiment / "TERMINAL_VERDICT.txt").write_text(terminal + "\n", encoding="utf-8")
    (experiment / "PHASE_C_COMPLETE").write_text(terminal + "\n", encoding="utf-8")
    print(json.dumps({"terminal": terminal, "ranking": result["ranking"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
