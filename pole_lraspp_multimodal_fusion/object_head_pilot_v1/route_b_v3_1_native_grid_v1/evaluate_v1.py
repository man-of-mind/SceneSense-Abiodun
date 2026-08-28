#!/usr/bin/env python3
"""Fixed validation evaluation of the native stride-4 correction.

Evaluates exactly epochs 3, 6, 9, 12 and 15 under the pre-registered selection
contract, runs the FP/FN taxonomy for the baseline and the selected checkpoint only,
adds the v0.25 sensitivity pass for the selected checkpoint only, and emits the
terminal verdict. No calibration, no threshold selection, no test split.

Matching semantics are the frozen v3.1 ones, reused verbatim from the clean-base
scorer via the targeted-refinement audit helpers.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parents[2]
FUSION_ROOT = ROOT / "pole_lraspp_multimodal_fusion"
BASE_PKG = FUSION_ROOT / "object_head_pilot_v1/route_b_v3_1_clean_base_v1"
REFINE_PKG = FUSION_ROOT / "object_head_pilot_v1/route_b_v3_1_targeted_refinement_v1"
# FUSION_ROOT is deliberately NOT on sys.path here. abiodun/pole_lraspp_multimodal_fusion
# (namespace) and abiodun/pole_lraspp_multimodal_fusion/pole_lraspp_multimodal_fusion
# (regular package) share a name; adding FUSION_ROOT makes the regular package win and
# breaks audit_v1's pole_lraspp_multimodal_fusion.object_head_pilot_v1.* import. This
# script needs only the frozen scorers, and runs inference in a subprocess.
for _path in (str(BASE_PKG), str(REFINE_PKG), str(ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from score_contract_v1 import score_segmentation  # noqa: E402  (frozen v3.1 scorer)
from audit_v1 import (  # noqa: E402  (frozen v3.1 matching + registered taxonomy)
    decompose_person_fn, decompose_vehicle_fp, load_gt, load_predictions, read_csv,
    score_arm, sha256,
)

EPOCHS = (3, 6, 9, 12, 15)
TRIAL_NAME = "route_b_v3_1_native_grid_v1"
PRIMARY = "v010"
SENSITIVITY = "v025"
THRESHOLDS = (0.20, 0.02)

BASELINE_EXPERIMENT = "experiments/route_b_v3_1_clean_base_v1/20260828_012309"
BASELINE_CHECKPOINT = "checkpoints/route_b_v3_1_clean_noae_stage2_v1/epoch_020.pt"
BASELINE_PREDICTIONS = "predictions/trained_epoch_020"


def write_json_x(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def flatten(primary_020: dict, primary_002: dict, segmentation: dict) -> dict[str, Any]:
    vehicle, person = primary_020["classes"]["vehicle"], primary_020["classes"]["person"]
    return {
        "vehicle_tp": vehicle["tp"], "vehicle_fp": vehicle["fp"], "vehicle_fn": vehicle["fn"],
        "vehicle_ignored": vehicle["ignored_predictions"],
        "person_tp": person["tp"], "person_fp": person["fp"], "person_fn": person["fn"],
        "person_ignored": person["ignored_predictions"],
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


def eligibility(metrics: dict, contract: dict) -> dict[str, bool]:
    gate = contract["eligibility_non_regression"]
    return {
        "vehicle_f1_ge_0.5666": metrics["vehicle_f1"] >= gate["vehicle_f1_ge"],
        "person_f1_ge_0.4251": metrics["person_f1"] >= gate["person_f1_ge"],
        "vehicle_recall_ge_0.6943": metrics["vehicle_recall"] >= gate["vehicle_recall_ge"],
        "vehicle_xy_mae_le_1.02": metrics["vehicle_xy_mae_m"] <= gate["vehicle_xy_mae_le_m"],
        "person_xy_mae_le_1.43": metrics["person_xy_mae_m"] <= gate["person_xy_mae_le_m"],
        "vehicle_iou_ge_0.854": metrics["vehicle_iou"] >= gate["vehicle_iou_ge"],
        "person_box_mask_iou_ge_0.429": metrics["person_box_mask_iou"] >= gate["person_box_mask_iou_ge"],
        "foreground_miou_ge_0.641": metrics["foreground_miou"] >= gate["foreground_miou_ge"],
    }


def material_gain(metrics: dict, duplicate_reduction: float | None, contract: dict) -> dict[str, bool]:
    gate = contract["material_gain_all_required"]
    person_gate = gate["person_improvement_either"]
    return {
        "mean_class_f1_ge_0.5309": metrics["mean_class_f1"] >= gate["mean_class_f1_ge"],
        "person_f1_ge_0.4501_or_person_recall_002_ge_0.5061": (
            metrics["person_f1"] >= person_gate["person_f1_ge"]
            or metrics["person_recall_002"] >= person_gate["person_recall_002_ge"]),
        "vehicle_precision_ge_0.5310": metrics["vehicle_precision"] >= gate["vehicle_precision_ge"],
        "vehicle_recall_ge_0.6943": metrics["vehicle_recall"] >= gate["vehicle_recall_ge"],
        "vehicle_duplicate_fp_reduction_ge_30pct": (
            duplicate_reduction is not None
            and duplicate_reduction >= gate["vehicle_duplicate_fp_reduction_ge_fraction"]),
    }


def service_targets(metrics: dict, contract: dict) -> dict[str, bool]:
    gate = contract["service_targets_advisory"]
    return {
        "vehicle_precision_ge_0.80": metrics["vehicle_precision"] >= gate["vehicle_precision_ge"],
        "vehicle_recall_ge_0.85": metrics["vehicle_recall"] >= gate["vehicle_recall_ge"],
        "person_precision_ge_0.80": metrics["person_precision"] >= gate["person_precision_ge"],
        "person_recall_ge_0.80": metrics["person_recall"] >= gate["person_recall_ge"],
        "vehicle_xy_mae_le_1.0m": metrics["vehicle_xy_mae_m"] <= gate["vehicle_xy_mae_le_m"],
        "person_xy_mae_le_1.2m": metrics["person_xy_mae_m"] <= gate["person_xy_mae_le_m"],
        "vehicle_iou_ge_0.85": metrics["vehicle_iou"] >= gate["vehicle_iou_ge"],
        "person_box_mask_iou_ge_0.50": metrics["person_box_mask_iou"] >= gate["person_box_mask_iou_ge"],
        "foreground_miou_ge_0.675": metrics["foreground_miou"] >= gate["foreground_miou_ge"],
    }


def run_taxonomy(experiment: Path, frame_ids, predictions, gt, ignore_cache) -> dict[str, Any]:
    """The registered v3.1 FP/FN taxonomy: vehicle FP at 0.20, person FN at 0.02."""
    at_020 = score_arm(experiment=experiment, contract=PRIMARY, frame_ids=frame_ids,
                       predictions=predictions, gt=gt, threshold=0.20,
                       ignore_cache=ignore_cache, collect=True)
    at_002 = score_arm(experiment=experiment, contract=PRIMARY, frame_ids=frame_ids,
                       predictions=predictions, gt=gt, threshold=0.02,
                       ignore_cache=ignore_cache, collect=True)
    vehicle_fp = decompose_vehicle_fp(at_020["_detail"]["vehicle_fp"], gt)
    person_fn = decompose_person_fn(at_002["_detail"]["person_fn"])
    vehicle_denominator = at_020["classes"]["vehicle"]["fp"]
    person_denominator = at_002["classes"]["person"]["fn"]
    return {
        "vehicle_fp_at_0_20": {
            **vehicle_fp, "denominator": vehicle_denominator,
            "total_labelled": sum(vehicle_fp["counts"].values()),
            "labels_sum_to_denominator": sum(vehicle_fp["counts"].values()) == vehicle_denominator,
            "percentages": {key: 100.0 * value / max(1, vehicle_denominator)
                            for key, value in vehicle_fp["counts"].items()},
        },
        "person_fn_at_0_02": {
            **person_fn, "denominator": person_denominator,
            "total_labelled": sum(person_fn["counts"].values()),
            "labels_sum_to_denominator": sum(person_fn["counts"].values()) == person_denominator,
            "percentages": {key: 100.0 * value / max(1, person_denominator)
                            for key, value in person_fn["counts"].items()},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    baseline_experiment = (ROOT / BASELINE_EXPERIMENT).resolve()
    infer_script = PACKAGE_ROOT / "infer_native_v1.py"
    started = time.monotonic()

    manifest = read_csv(experiment / "dataset/manifest.csv")
    frame_ids = [row["sample_id"] for row in manifest if row["split"] == "val"]
    if len(frame_ids) != contract["evaluation_view"]["validation_frames"]:
        raise RuntimeError(f"validation frame count {len(frame_ids)} != registered view")
    gt = {name: load_gt(experiment, name)[0] for name in (PRIMARY, SENSITIVITY)}
    # The v0.10 and v0.25 ignore masks are NOT identical, so the cache is per contract.
    ignore_caches: dict[str, dict[str, Any]] = {PRIMARY: {}, SENSITIVITY: {}}

    checkpoint_dir = experiment / "checkpoints" / TRIAL_NAME
    records: list[dict[str, Any]] = []
    for epoch in EPOCHS:
        checkpoint = checkpoint_dir / f"epoch_{epoch:03d}.pt"
        checkpoint_hash = sha256(checkpoint)
        tag = f"trained_epoch_{epoch:03d}"
        prediction_root = experiment / "predictions" / tag
        if not (prediction_root / "INFERENCE_COMPLETE").is_file():
            print(f"[eval] inference epoch={epoch}", flush=True)
            command = [sys.executable, str(infer_script), "--experiment", str(experiment),
                       "--checkpoint", str(checkpoint), "--checkpoint-sha256", checkpoint_hash,
                       "--tag", tag]
            if subprocess.run(command).returncode != 0:
                raise RuntimeError(f"inference failed for epoch {epoch}")
        inference = json.loads((prediction_root / "inference_manifest.json").read_text(encoding="utf-8"))
        if sha256(prediction_root / "detections.csv") != inference["detections_sha256"]:
            raise RuntimeError(f"detection hash drift: {tag}")
        if inference["checkpoint_sha256"] != checkpoint_hash:
            raise RuntimeError(f"checkpoint provenance mismatch: {tag}")
        if inference["native_object_grid"] != [108, 192]:
            raise RuntimeError(f"non-native object grid in {tag}: {inference['native_object_grid']}")

        predictions, missing = load_predictions(prediction_root / "detections.csv")
        if missing:
            raise RuntimeError(f"missing prediction fields: {tag}")
        print(f"[eval] segmentation epoch={epoch}", flush=True)
        segmentation = score_segmentation(experiment, PRIMARY, frame_ids, prediction_root,
                                          prediction_root / "segmentation_manifest.csv")
        scored = {f"{threshold:.2f}": score_arm(
            experiment=experiment, contract=PRIMARY, frame_ids=frame_ids, predictions=predictions,
            gt=gt[PRIMARY], threshold=threshold, ignore_cache=ignore_caches[PRIMARY],
        ) for threshold in THRESHOLDS}
        metrics = flatten(scored["0.20"], scored["0.02"], segmentation)
        guards = eligibility(metrics, contract)
        records.append({
            "epoch": epoch, "tag": tag, "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "prediction_set_sha256": inference["prediction_set_sha256"],
            "detection_predictions": inference["detection_predictions"],
            "inference_wall_seconds": inference["wall_seconds"],
            "peak_allocated_mib": inference["peak_allocated_mib"],
            "peak_reserved_mib": inference["peak_reserved_mib"],
            "metrics": metrics, "primary_v010": scored, "segmentation_v010": segmentation,
            "eligibility_guards": guards, "eligible": all(guards.values()),
            "service_targets": service_targets(metrics, contract),
        })

    # Registered ranking, applied to eligible checkpoints only.
    eligible = [record for record in records if record["eligible"]]
    ranked = sorted(eligible, key=lambda record: (
        -record["metrics"]["mean_class_f1"],
        -record["metrics"]["minimum_class_recall"],
        record["metrics"]["mean_xy_mae_m"],
        record["epoch"],
    ))
    selected = ranked[0] if ranked else None

    # Taxonomy: baseline and selected only.
    baseline_predictions, baseline_missing = load_predictions(
        baseline_experiment / BASELINE_PREDICTIONS / "detections.csv")
    if baseline_missing:
        raise RuntimeError("missing baseline prediction fields")
    baseline_gt, _ = load_gt(baseline_experiment, PRIMARY)
    print("[eval] baseline taxonomy", flush=True)
    baseline_taxonomy = run_taxonomy(baseline_experiment, frame_ids, baseline_predictions,
                                     baseline_gt, {})
    baseline_duplicates = baseline_taxonomy["vehicle_fp_at_0_20"]["counts"]["PREDICTED_DUPLICATE"]

    selected_taxonomy = None
    duplicate_reduction = None
    sensitivity = None
    if selected is not None:
        prediction_root = experiment / "predictions" / selected["tag"]
        selected_predictions, _ = load_predictions(prediction_root / "detections.csv")
        print(f"[eval] selected taxonomy epoch={selected['epoch']}", flush=True)
        selected_taxonomy = run_taxonomy(experiment, frame_ids, selected_predictions,
                                         gt[PRIMARY], {})
        selected_duplicates = selected_taxonomy["vehicle_fp_at_0_20"]["counts"]["PREDICTED_DUPLICATE"]
        duplicate_reduction = (baseline_duplicates - selected_duplicates) / max(1, baseline_duplicates)
        print(f"[eval] v025 sensitivity epoch={selected['epoch']}", flush=True)
        sensitivity = {f"{threshold:.2f}": score_arm(
            experiment=experiment, contract=SENSITIVITY, frame_ids=frame_ids,
            predictions=selected_predictions, gt=gt[SENSITIVITY], threshold=threshold,
            ignore_cache=ignore_caches[SENSITIVITY],
        ) for threshold in THRESHOLDS}

    gain = (material_gain(selected["metrics"], duplicate_reduction, contract)
            if selected is not None else None)
    if selected is None or not all(gain.values()):
        terminal = "LRASPP_NATIVE_GRID_NO_GAIN"
    elif all(selected["service_targets"].values()):
        terminal = "LRASPP_NATIVE_GRID_SERVICE_READY"
    else:
        terminal = "LRASPP_NATIVE_GRID_MATERIAL_GAIN_NOT_SERVICE_READY"

    result: dict[str, Any] = {
        "schema": "route_b_v3_1_native_grid_evaluation_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_contract": contract,
        "baseline_experiment": str(baseline_experiment),
        "baseline_checkpoint_sha256": sha256(baseline_experiment / BASELINE_CHECKPOINT),
        "evaluated_epochs": list(EPOCHS),
        "validation_frames": len(frame_ids),
        "records": records,
        "ranking": [{"epoch": record["epoch"],
                     "mean_class_f1": record["metrics"]["mean_class_f1"],
                     "minimum_class_recall": record["metrics"]["minimum_class_recall"],
                     "mean_xy_mae_m": record["metrics"]["mean_xy_mae_m"]} for record in ranked],
        "eligible_epochs": [record["epoch"] for record in eligible],
        "baseline_taxonomy": baseline_taxonomy,
        "selected_taxonomy": selected_taxonomy,
        "vehicle_duplicate_fp": {
            "baseline": baseline_duplicates,
            "selected": (selected_taxonomy["vehicle_fp_at_0_20"]["counts"]["PREDICTED_DUPLICATE"]
                         if selected_taxonomy else None),
            "reduction_fraction": duplicate_reduction,
        },
        "material_gain_gates": gain,
        "material_gain_pass": bool(gain and all(gain.values())),
        "terminal": terminal,
        "wall_seconds": time.monotonic() - started,
    }
    if selected is not None:
        result["selected"] = {
            "epoch": selected["epoch"], "checkpoint": selected["checkpoint"],
            "checkpoint_sha256": selected["checkpoint_sha256"],
            "metrics": selected["metrics"],
            "deltas_vs_baseline": {
                key: selected["metrics"][key] - contract["baseline_epoch20"][key]
                for key in selected["metrics"] if key in contract["baseline_epoch20"]},
            "service_targets": selected["service_targets"],
            "service_ready": all(selected["service_targets"].values()),
            "sensitivity_v025": sensitivity,
        }
    else:
        result["selected"] = None

    write_json_x(experiment / "EVALUATION.json", result)
    (experiment / "TERMINAL_VERDICT.txt").write_text(terminal + "\n", encoding="utf-8")
    (experiment / "EVALUATION_COMPLETE").write_text(terminal + "\n", encoding="utf-8")
    print(json.dumps({
        "terminal": terminal,
        "epoch_metrics": [{"epoch": record["epoch"], "eligible": record["eligible"],
                           **{key: record["metrics"][key] for key in
                              ("vehicle_precision", "vehicle_recall", "vehicle_f1",
                               "person_precision", "person_recall", "person_f1",
                               "mean_class_f1", "vehicle_xy_mae_m", "person_xy_mae_m",
                               "foreground_miou")}} for record in records],
        "selected_epoch": selected["epoch"] if selected else None,
        "vehicle_duplicate_fp": result["vehicle_duplicate_fp"],
        "material_gain_gates": gain,
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
