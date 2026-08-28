#!/usr/bin/env python3
"""Evaluate exactly epochs 4/8/12 under the registered amended-baseline gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parents[2]
BASE_PKG = ROOT / "pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_clean_base_v1"
REFINE_PKG = ROOT / "pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_targeted_refinement_v1"
for path in (str(PACKAGE_ROOT), str(BASE_PKG), str(REFINE_PKG), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from audit_v1 import (  # noqa: E402
    decompose_person_fn, decompose_vehicle_fp, load_gt, load_predictions, match_frame,
    read_csv, score_arm,
)

EPOCHS = (4, 8, 12)
THRESHOLDS = (0.20, 0.02)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_x(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def flatten(primary: Mapping[str, Any], ceiling: Mapping[str, Any], segmentation: Mapping[str, Any]) -> dict[str, Any]:
    vehicle, person = primary["classes"]["vehicle"], primary["classes"]["person"]
    return {
        "vehicle_tp": vehicle["tp"], "vehicle_fp": vehicle["fp"], "vehicle_fn": vehicle["fn"],
        "vehicle_ignored_predictions": vehicle["ignored_predictions"],
        "vehicle_precision": vehicle["precision"], "vehicle_recall": vehicle["recall"],
        "vehicle_f1": vehicle["f1"], "vehicle_recall_002": ceiling["classes"]["vehicle"]["recall"],
        "vehicle_xy_mae_m": vehicle["xy_mae_m"],
        "person_tp": person["tp"], "person_fp": person["fp"], "person_fn": person["fn"],
        "person_ignored_predictions": person["ignored_predictions"],
        "person_precision": person["precision"], "person_recall": person["recall"],
        "person_f1": person["f1"], "person_recall_002": ceiling["classes"]["person"]["recall"],
        "person_xy_mae_m": person["xy_mae_m"],
        "vehicle_iou": segmentation["vehicle_iou"],
        "person_box_mask_iou": segmentation["person_box_mask_iou"],
        "foreground_miou": segmentation["foreground_miou"],
        "background_iou": segmentation["background_iou"],
        "mean_class_f1": (vehicle["f1"] + person["f1"]) / 2.0,
    }


def evaluate_primary(
    contract_experiment: Path, frame_ids: Sequence[str], predictions,
    gt: Mapping[str, Sequence[Mapping[str, Any]]], segmentation: Mapping[str, Any],
) -> dict[str, Any]:
    # This cache belongs only to v0.10 and is discarded before any v0.25 scoring.
    ignore_cache: dict[str, Any] = {}
    primary = score_arm(
        experiment=contract_experiment, contract="v010", frame_ids=frame_ids,
        predictions=predictions, gt=gt, threshold=0.20, ignore_cache=ignore_cache, collect=True,
    )
    ceiling = score_arm(
        experiment=contract_experiment, contract="v010", frame_ids=frame_ids,
        predictions=predictions, gt=gt, threshold=0.02, ignore_cache=ignore_cache, collect=True,
    )
    vehicle = decompose_vehicle_fp(primary["_detail"]["vehicle_fp"], gt)
    person = decompose_person_fn(ceiling["_detail"]["person_fn"])
    taxonomy = {
        "vehicle_fp_at_0_20": {
            **vehicle, "denominator": primary["classes"]["vehicle"]["fp"],
            "labels_sum_to_denominator": sum(vehicle["counts"].values())
            == primary["classes"]["vehicle"]["fp"],
        },
        "person_fn_at_0_02": {
            **person, "denominator": ceiling["classes"]["person"]["fn"],
            "labels_sum_to_denominator": sum(person["counts"].values())
            == ceiling["classes"]["person"]["fn"],
        },
    }
    primary.pop("_detail")
    ceiling.pop("_detail")
    return {
        "thresholds": {"0.20": primary, "0.02": ceiling},
        "flat": flatten(primary, ceiling, segmentation), "taxonomy": taxonomy,
        "ignore_cache_contract_key": "v010",
    }


def evaluate_sensitivity(
    contract_experiment: Path, frame_ids: Sequence[str], predictions,
    gt: Mapping[str, Sequence[Mapping[str, Any]]], segmentation: Mapping[str, Any],
) -> dict[str, Any]:
    # Independent object and mask cache: never shared with the v0.10 scorer above.
    ignore_cache: dict[str, Any] = {}
    arms = {
        f"{threshold:.2f}": score_arm(
            experiment=contract_experiment, contract="v025", frame_ids=frame_ids,
            predictions=predictions, gt=gt, threshold=threshold,
            ignore_cache=ignore_cache, collect=False,
        ) for threshold in THRESHOLDS
    }
    return {
        "thresholds": arms, "flat": flatten(arms["0.20"], arms["0.02"], segmentation),
        "ignore_cache_contract_key": "v025",
    }


def eligibility(metrics: Mapping[str, Any], taxonomy: Mapping[str, Any],
                baseline: Mapping[str, Any], invariant: Mapping[str, Any]) -> dict[str, bool]:
    deltas = {
        key: float(metrics[key]) - float(baseline[key])
        for key in ("vehicle_precision", "vehicle_recall", "vehicle_f1",
                    "person_precision", "person_recall", "person_f1")
    }
    vehicle_tax = taxonomy["vehicle_fp_at_0_20"]["counts"]
    person_tax = taxonomy["person_fn_at_0_02"]["counts"]
    return {
        **{f"{key}_delta_ge_minus_0_01": value >= -0.01 for key, value in deltas.items()},
        "vehicle_iou_bit_identical_1e_6": abs(metrics["vehicle_iou"] - baseline["vehicle_iou"]) <= 1e-6,
        "person_iou_bit_identical_1e_6": abs(
            metrics["person_box_mask_iou"] - baseline["person_box_mask_iou"]
        ) <= 1e-6,
        "foreground_iou_bit_identical_1e_6": abs(
            metrics["foreground_miou"] - baseline["foreground_miou"]
        ) <= 1e-6,
        "all_non_xyz_fields_bit_identical": bool(invariant["all_non_xyz_detection_fields_bit_identical"]),
        "segmentation_outputs_bit_identical": bool(invariant["segmentation_outputs_bit_identical"]),
        "vehicle_duplicate_candidate_count_unchanged": (
            vehicle_tax["PREDICTED_DUPLICATE"] == baseline["vehicle_duplicate_fp"]
        ),
        "person_heatmap_center_miss_count_unchanged": (
            person_tax["HEATMAP_CENTER_MISS"] == baseline["person_heatmap_center_miss"]
        ),
    }


def service_targets(metrics: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "vehicle_precision_ge_0_80": metrics["vehicle_precision"] >= 0.80,
        "vehicle_recall_ge_0_85": metrics["vehicle_recall"] >= 0.85,
        "person_precision_ge_0_80": metrics["person_precision"] >= 0.80,
        "person_recall_ge_0_80": metrics["person_recall"] >= 0.80,
        "vehicle_xy_mae_le_1_0m": metrics["vehicle_xy_mae_m"] <= 1.0,
        "person_xy_mae_le_1_2m": metrics["person_xy_mae_m"] <= 1.2,
        "vehicle_iou_ge_0_85": metrics["vehicle_iou"] >= 0.85,
        "person_box_mask_iou_ge_0_50": metrics["person_box_mask_iou"] >= 0.50,
        "foreground_miou_ge_0_675": metrics["foreground_miou"] >= 0.675,
    }


def material_gates(metrics: Mapping[str, Any], taxonomy: Mapping[str, Any],
                   baseline: Mapping[str, Any], sensitivity: Mapping[str, Any],
                   baseline_v025: Mapping[str, Any]) -> dict[str, bool]:
    vehicle_tax = taxonomy["vehicle_fp_at_0_20"]["counts"]
    person_tax = taxonomy["person_fn_at_0_02"]["counts"]
    sensitivity_flat = sensitivity["flat"]
    return {
        "vehicle_two_d_correct_world_wrong_reduced_ge_30pct": (
            vehicle_tax["TWO_D_CORRECT_WORLD_WRONG"]
            <= 0.70 * baseline["vehicle_two_d_correct_world_wrong"]
        ),
        "person_center_present_world_wrong_reduced_ge_30pct": (
            person_tax["CENTER_PRESENT_WORLD_WRONG"]
            <= 0.70 * baseline["person_center_present_world_wrong"]
        ),
        "vehicle_xy_mae_le_0_95m": metrics["vehicle_xy_mae_m"] <= 0.95,
        "person_xy_mae_le_1_25m": metrics["person_xy_mae_m"] <= 1.25,
        "vehicle_f1_delta_ge_0_02": metrics["vehicle_f1"] >= baseline["vehicle_f1"] + 0.02,
        "person_f1_delta_ge_0_05": metrics["person_f1"] >= baseline["person_f1"] + 0.05,
        "vehicle_duplicate_candidates_no_increase": (
            vehicle_tax["PREDICTED_DUPLICATE"] <= baseline["vehicle_duplicate_fp"]
        ),
        "person_heatmap_center_miss_no_increase": (
            person_tax["HEATMAP_CENTER_MISS"] <= baseline["person_heatmap_center_miss"]
        ),
        "v025_vehicle_f1_no_reversal": (
            sensitivity_flat["vehicle_f1"] >= baseline_v025["vehicle_f1"] - 0.01
        ),
        "v025_vehicle_xy_no_reversal": (
            sensitivity_flat["vehicle_xy_mae_m"] <= baseline_v025["vehicle_xy_mae_m"]
        ),
        "v025_person_f1_no_reversal": (
            sensitivity_flat["person_f1"] >= baseline_v025["person_f1"] - 0.01
        ),
        "v025_person_xy_no_reversal": (
            sensitivity_flat["person_xy_mae_m"] <= baseline_v025["person_xy_mae_m"]
        ),
    }


def load_radar_gt(contract_experiment: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sx, sy = 768.0 / 1280.0, 432.0 / 720.0
    for row in read_csv(contract_experiment / "contracts/v010/val/object_boxes.csv"):
        x0, y0 = float(row["gt_bbox_x"]), float(row["gt_bbox_y"])
        grouped[row["sample_id"]].append({
            "class_name": row["label"], "world_x": float(row["object_world_x"]),
            "world_y": float(row["object_world_y"]),
            "box": (x0 * sx, y0 * sy, (x0 + float(row["gt_bbox_w"])) * sx,
                    (y0 + float(row["gt_bbox_h"])) * sy),
            "radar_supported": float(row.get("radar_support_points", 0) or 0) > 0.0,
        })
    return grouped


def radar_stratified(frame_ids: Sequence[str], predictions, gt) -> dict[str, Any]:
    buckets = {
        class_name: {support: [] for support in ("supported", "unsupported")}
        for class_name in ("vehicle", "person")
    }
    denominators = {
        class_name: {support: 0 for support in ("supported", "unsupported")}
        for class_name in ("vehicle", "person")
    }
    for sample_id in frame_ids:
        frame_gt = list(gt.get(sample_id, []))
        frame_predictions = [item for item in predictions.get(sample_id, [])
                             if float(item["score"]) >= 0.20]
        _used_pred, _used_gt, pred_to_gt = match_frame(frame_predictions, frame_gt)
        for target in frame_gt:
            support = "supported" if target["radar_supported"] else "unsupported"
            denominators[target["class_name"]][support] += 1
        for pred_index, gt_index in pred_to_gt.items():
            prediction, target = frame_predictions[pred_index], frame_gt[gt_index]
            support = "supported" if target["radar_supported"] else "unsupported"
            buckets[target["class_name"]][support].append(math.hypot(
                float(prediction["world_x"]) - float(target["world_x"]),
                float(prediction["world_y"]) - float(target["world_y"]),
            ))
    return {
        class_name: {
            support: {
                "eligible_gt": denominators[class_name][support],
                "matched": len(buckets[class_name][support]),
                "recall": len(buckets[class_name][support]) / max(1, denominators[class_name][support]),
                "xy_mae_m": (sum(buckets[class_name][support]) / len(buckets[class_name][support])
                             if buckets[class_name][support] else None),
            } for support in ("supported", "unsupported")
        } for class_name in ("vehicle", "person")
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--contract-experiment", required=True, type=Path)
    parser.add_argument("--selection-contract", required=True, type=Path)
    parser.add_argument("--infer-script", required=True, type=Path)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    contract_experiment = args.contract_experiment.resolve()
    started = time.monotonic()
    if not (experiment / "TRAINING_COMPLETE").is_file():
        raise RuntimeError("12-epoch training completion gate absent")
    selection_path = args.selection_contract.resolve(strict=True)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    amended = json.loads((contract_experiment / "AMENDED_BASELINE.json").read_text(encoding="utf-8"))
    baseline = dict(selection["amended_baseline"])
    baseline_v025 = dict(selection["amended_v025_baseline"])
    if baseline != {key: amended["amended"]["v010"]["flat"].get(key,
                   amended["amended_taxonomy"].get(key)) for key in baseline}:
        # The registered contract also contains taxonomy aliases, checked directly below.
        flat = amended["amended"]["v010"]["flat"]
        taxonomy = amended["amended_taxonomy"]
        aliases = {
            **{key: flat[key] for key in baseline if key in flat},
            "vehicle_two_d_correct_world_wrong": taxonomy["vehicle_fp_at_0_20"]["counts"]["TWO_D_CORRECT_WORLD_WRONG"],
            "person_center_present_world_wrong": taxonomy["person_fn_at_0_02"]["counts"]["CENTER_PRESENT_WORLD_WRONG"],
            "vehicle_duplicate_fp": taxonomy["vehicle_fp_at_0_20"]["counts"]["PREDICTED_DUPLICATE"],
            "person_heatmap_center_miss": taxonomy["person_fn_at_0_02"]["counts"]["HEATMAP_CENTER_MISS"],
        }
        if aliases != baseline:
            raise RuntimeError("registered amended baseline drift")

    manifest = read_csv(contract_experiment / "dataset/manifest.csv")
    frame_ids = [row["sample_id"] for row in manifest if row["split"] == "val"]
    gt_v010, _ = load_gt(contract_experiment, "v010")
    gt_v025, _ = load_gt(contract_experiment, "v025")
    segmentation_v010 = amended["amended"]["v010"]["segmentation"]
    segmentation_v025 = amended["amended"]["v025"]["segmentation"]
    checkpoint_dir = experiment / "checkpoints/route_b_v3_1_factorized_localization_v2"
    records: list[dict[str, Any]] = []
    for epoch in EPOCHS:
        checkpoint = checkpoint_dir / f"epoch_{epoch:03d}.pt"
        checkpoint_hash = sha256(checkpoint)
        tag = f"factorized_epoch_{epoch:03d}"
        prediction_root = experiment / "predictions" / tag
        if prediction_root.exists():
            raise RuntimeError(f"create-only inference path already exists: {prediction_root}")
        command = [
            sys.executable, str(args.infer_script.resolve()),
            "--experiment", str(experiment), "--contract-experiment", str(contract_experiment),
            "--checkpoint", str(checkpoint), "--checkpoint-sha256", checkpoint_hash,
            "--tag", tag,
        ]
        print(f"[factorized evaluation] one inference pass for epoch {epoch}", flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"inference failed for epoch {epoch}: {completed.returncode}")
        inference = json.loads((prediction_root / "inference_manifest.json").read_text(encoding="utf-8"))
        predictions, missing = load_predictions(prediction_root / "detections.csv")
        if missing:
            raise RuntimeError(f"missing candidate prediction fields at epoch {epoch}: {len(missing)}")
        primary = evaluate_primary(
            contract_experiment, frame_ids, predictions, gt_v010, segmentation_v010
        )
        gates = eligibility(primary["flat"], primary["taxonomy"], baseline, inference)
        record = {
            "epoch": epoch, "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
            "prediction_root": str(prediction_root),
            "detections_sha256": inference["detections_sha256"],
            "prediction_set_sha256": inference["prediction_set_sha256"],
            "inference": inference, "primary_v010": primary,
            "eligibility_gates": gates, "eligible": all(gates.values()),
            "rank_values": {
                "mean_normalized_xy_mae": (
                    primary["flat"]["vehicle_xy_mae_m"] / baseline["vehicle_xy_mae_m"]
                    + primary["flat"]["person_xy_mae_m"] / baseline["person_xy_mae_m"]
                ) / 2.0,
                "combined_world_error_taxonomy": (
                    primary["taxonomy"]["vehicle_fp_at_0_20"]["counts"]["TWO_D_CORRECT_WORLD_WRONG"]
                    + primary["taxonomy"]["person_fn_at_0_02"]["counts"]["CENTER_PRESENT_WORLD_WRONG"]
                ),
                "mean_class_f1": primary["flat"]["mean_class_f1"],
            },
            "service_targets": service_targets(primary["flat"]),
        }
        records.append(record)

    ranked_all = sorted(records, key=lambda record: (
        record["rank_values"]["mean_normalized_xy_mae"],
        record["rank_values"]["combined_world_error_taxonomy"],
        -record["rank_values"]["mean_class_f1"], record["epoch"],
    ))
    ranked_eligible = [record for record in ranked_all if record["eligible"]]
    selected = ranked_eligible[0] if ranked_eligible else None
    best_ranked = ranked_all[0]

    selected_sensitivity = None
    selected_material = None
    selected_radar = None
    baseline_predictions, baseline_missing = load_predictions(Path(amended["retained_predictions"]))
    if baseline_missing:
        raise RuntimeError("retained baseline became unreadable")
    radar_gt = load_radar_gt(contract_experiment)
    baseline_radar = radar_stratified(frame_ids, baseline_predictions, radar_gt)
    if selected is not None:
        selected_predictions, missing = load_predictions(Path(selected["prediction_root"]) / "detections.csv")
        if missing:
            raise RuntimeError("selected predictions became unreadable")
        selected_sensitivity = evaluate_sensitivity(
            contract_experiment, frame_ids, selected_predictions, gt_v025, segmentation_v025
        )
        selected_material = material_gates(
            selected["primary_v010"]["flat"], selected["primary_v010"]["taxonomy"],
            baseline, selected_sensitivity, baseline_v025,
        )
        selected_radar = radar_stratified(frame_ids, selected_predictions, radar_gt)
    material_pass = bool(selected is not None and selected_material and all(selected_material.values()))
    terminal = ("LRASPP_FACTORIZED_LOCALIZATION_MATERIAL_GAIN" if material_pass
                else "LRASPP_FACTORIZED_LOCALIZATION_NO_GAIN")
    result = {
        "schema": "route_b_v3_1_factorized_localization_evaluation_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "registered_selection_contract": str(selection_path),
        "registered_selection_contract_sha256": sha256(selection_path),
        "selection_registered_before_candidate_evaluation": True,
        "amended_baseline": amended, "evaluated_epochs": list(EPOCHS),
        "inference_passes_per_epoch": {str(epoch): 1 for epoch in EPOCHS},
        "records": records,
        "ranking": [{
            "epoch": record["epoch"], "eligible": record["eligible"],
            **record["rank_values"], "checkpoint_sha256": record["checkpoint_sha256"],
            "detections_sha256": record["detections_sha256"],
        } for record in ranked_all],
        "best_ranked_epoch_regardless_of_eligibility": best_ranked["epoch"],
        "selected": ({
            "epoch": selected["epoch"], "checkpoint": selected["checkpoint"],
            "checkpoint_sha256": selected["checkpoint_sha256"],
            "metrics_v010": selected["primary_v010"]["flat"],
            "taxonomy_v010": selected["primary_v010"]["taxonomy"],
            "sensitivity_v025": selected_sensitivity,
            "material_gain_gates": selected_material,
            "material_gain_pass": material_pass,
            "service_targets": selected["service_targets"],
        } if selected is not None else None),
        "radar_stratified_localization": {
            "amended_baseline": baseline_radar, "selected": selected_radar,
            "score_threshold": 0.20, "matching_radius_m": 3.0,
        },
        "terminal": terminal, "wall_seconds": time.monotonic() - started,
    }
    write_json_x(experiment / "SELECTION.json", result)
    print(json.dumps({
        "terminal": terminal, "ranking": result["ranking"], "selected": result["selected"],
        "wall_seconds": result["wall_seconds"],
    }, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
