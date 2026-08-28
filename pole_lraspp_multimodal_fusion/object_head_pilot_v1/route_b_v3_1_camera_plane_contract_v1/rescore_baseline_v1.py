#!/usr/bin/env python3
"""Re-score retained native epoch-15 predictions under the camera-plane contract."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parents[2]
BASE_PKG = ROOT / "pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_clean_base_v1"
REFINE_PKG = ROOT / "pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_targeted_refinement_v1"
for path in (str(BASE_PKG), str(REFINE_PKG), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from score_contract_v1 import score_segmentation  # noqa: E402
from audit_v1 import (  # noqa: E402
    decompose_person_fn, decompose_vehicle_fp, load_gt, load_predictions, read_csv,
    score_arm, sha256,
)

CONTRACTS = ("v010", "v025")
THRESHOLDS = (0.20, 0.02)
EXPECTED_WARM_SHA = "1245b2028372d486ed0b25b8a6b8a3e8b341257d542ec57cfdabf3b543d7c9ed"


def write_json_x(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_text_x(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value)


def flatten(primary: Mapping[str, Any], ceiling: Mapping[str, Any], segmentation: Mapping[str, Any]) -> dict[str, Any]:
    vehicle = primary["classes"]["vehicle"]
    person = primary["classes"]["person"]
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
        "mean_normalized_xy_mae": (
            vehicle["xy_mae_m"] / 0.95 + person["xy_mae_m"] / 1.25
        ) / 2.0,
    }


def taxonomy(
    experiment: Path, frame_ids: Sequence[str], predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    gt: Mapping[str, Sequence[Mapping[str, Any]]], ignore_cache: dict[str, Any],
) -> dict[str, Any]:
    at_020 = score_arm(experiment=experiment, contract="v010", frame_ids=frame_ids,
                       predictions=predictions, gt=gt, threshold=0.20,
                       ignore_cache=ignore_cache, collect=True)
    at_002 = score_arm(experiment=experiment, contract="v010", frame_ids=frame_ids,
                       predictions=predictions, gt=gt, threshold=0.02,
                       ignore_cache=ignore_cache, collect=True)
    vehicle = decompose_vehicle_fp(at_020["_detail"]["vehicle_fp"], gt)
    person = decompose_person_fn(at_002["_detail"]["person_fn"])
    vehicle_denominator = at_020["classes"]["vehicle"]["fp"]
    person_denominator = at_002["classes"]["person"]["fn"]
    return {
        "vehicle_fp_at_0_20": {
            **vehicle, "denominator": vehicle_denominator,
            "labels_sum_to_denominator": sum(vehicle["counts"].values()) == vehicle_denominator,
        },
        "person_fn_at_0_02": {
            **person, "denominator": person_denominator,
            "labels_sum_to_denominator": sum(person["counts"].values()) == person_denominator,
        },
    }


def score_contract(
    experiment: Path, contract: str, frame_ids: Sequence[str], predictions, gt,
    prediction_root: Path, segmentation_manifest: Path,
) -> dict[str, Any]:
    # Deliberately local to this contract: v0.10 and v0.25 never share ignore caches.
    ignore_cache: dict[str, Any] = {}
    thresholds = {
        f"{threshold:.2f}": score_arm(
            experiment=experiment, contract=contract, frame_ids=frame_ids,
            predictions=predictions, gt=gt, threshold=threshold,
            ignore_cache=ignore_cache,
        )
        for threshold in THRESHOLDS
    }
    segmentation = score_segmentation(
        experiment, contract, frame_ids, prediction_root, segmentation_manifest
    )
    return {
        "thresholds": thresholds,
        "segmentation": segmentation,
        "flat": flatten(thresholds["0.20"], thresholds["0.02"], segmentation),
        "ignore_cache_contract_key": contract,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived-experiment", required=True, type=Path)
    parser.add_argument("--native-experiment", required=True, type=Path)
    args = parser.parse_args()
    derived = args.derived_experiment.resolve()
    native = args.native_experiment.resolve()
    started = time.monotonic()

    checkpoint = native / "checkpoints/route_b_v3_1_native_grid_v1/epoch_015.pt"
    prediction_root = native / "predictions/trained_epoch_015"
    inference = json.loads((prediction_root / "inference_manifest.json").read_text(encoding="utf-8"))
    checkpoint_sha = sha256(checkpoint)
    detections_sha = sha256(prediction_root / "detections.csv")
    if checkpoint_sha != EXPECTED_WARM_SHA or inference["checkpoint_sha256"] != EXPECTED_WARM_SHA:
        raise RuntimeError("warm-start provenance mismatch")
    if detections_sha != inference["detections_sha256"]:
        raise RuntimeError("retained prediction hash mismatch")
    predictions, missing = load_predictions(prediction_root / "detections.csv")
    if missing:
        raise RuntimeError(f"retained predictions have missing fields: {len(missing)}")
    manifest = read_csv(derived / "dataset/manifest.csv")
    frame_ids = [row["sample_id"] for row in manifest if row["split"] == "val"]
    if len(frame_ids) != 3345 or any(row["split"] == "test" for row in manifest):
        raise RuntimeError("derived validation view mismatch")

    amended: dict[str, Any] = {}
    original: dict[str, Any] = {}
    for contract in CONTRACTS:
        amended_gt, _ = load_gt(derived, contract)
        original_gt, _ = load_gt(native, contract)
        amended[contract] = score_contract(
            derived, contract, frame_ids, predictions, amended_gt,
            prediction_root, prediction_root / "segmentation_manifest.csv",
        )
        original[contract] = score_contract(
            native, contract, frame_ids, predictions, original_gt,
            prediction_root, prediction_root / "segmentation_manifest.csv",
        )

    amended_gt_v010, _ = load_gt(derived, "v010")
    original_gt_v010, _ = load_gt(native, "v010")
    amended_taxonomy = taxonomy(derived, frame_ids, predictions, amended_gt_v010, {})
    original_taxonomy = taxonomy(native, frame_ids, predictions, original_gt_v010, {})

    provenance = [row for row in read_csv(derived / "provenance/camera_plane_exclusions.csv")
                  if row["contract"] == "v010" and row["split"] == "val"]
    affected = {row["sample_id"] for row in provenance}
    unaffected = [sample_id for sample_id in frame_ids if sample_id not in affected]
    amended_unaffected = score_arm(
        experiment=derived, contract="v010", frame_ids=unaffected, predictions=predictions,
        gt=amended_gt_v010, threshold=0.20, ignore_cache={},
    )
    original_unaffected = score_arm(
        experiment=native, contract="v010", frame_ids=unaffected, predictions=predictions,
        gt=original_gt_v010, threshold=0.20, ignore_cache={},
    )
    explanation_gates = {
        "retained_predictions_reused_without_inference": True,
        "retained_prediction_hash_verified": True,
        "exactly_34_v010_validation_gt_removed_from_localization_denominator": (
            original["v010"]["flat"]["vehicle_tp"] + original["v010"]["flat"]["vehicle_fn"]
            - amended["v010"]["flat"]["vehicle_tp"] - amended["v010"]["flat"]["vehicle_fn"]
        ) == 34,
        "person_denominator_unchanged": (
            original["v010"]["flat"]["person_tp"] + original["v010"]["flat"]["person_fn"]
            == amended["v010"]["flat"]["person_tp"] + amended["v010"]["flat"]["person_fn"]
        ),
        "unaffected_frame_metrics_exact": amended_unaffected == original_unaffected,
        "segmentation_v010_bit_identical": (
            amended["v010"]["segmentation"] == original["v010"]["segmentation"]
        ),
        "segmentation_v025_bit_identical": (
            amended["v025"]["segmentation"] == original["v025"]["segmentation"]
        ),
        "v010_v025_ignore_caches_independently_keyed": (
            amended["v010"]["ignore_cache_contract_key"] == "v010"
            and amended["v025"]["ignore_cache_contract_key"] == "v025"
        ),
    }
    if not all(explanation_gates.values()):
        raise RuntimeError(f"amended baseline explanation failure: {explanation_gates}")
    result = {
        "schema": "route_b_v3_1_camera_plane_amended_native_baseline_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
        "retained_predictions": str(prediction_root / "detections.csv"),
        "retained_detections_sha256": detections_sha,
        "prediction_set_sha256": inference["prediction_set_sha256"],
        "new_inference_passes": 0, "validation_frames": len(frame_ids),
        "original": original, "amended": amended,
        "original_taxonomy": original_taxonomy, "amended_taxonomy": amended_taxonomy,
        "affected_validation_frames": len(affected),
        "explanation_gates": explanation_gates,
        "wall_seconds": time.monotonic() - started,
    }
    write_json_x(derived / "AMENDED_BASELINE.json", result)
    write_text_x(derived / "BASELINE_RESCORE_COMPLETE", "AMENDED_BASELINE_READY\n")
    print(json.dumps({
        "terminal": "AMENDED_BASELINE_READY",
        "amended_v010": amended["v010"]["flat"],
        "amended_v025": amended["v025"]["flat"],
        "amended_taxonomy": amended_taxonomy,
        "explanation_gates": explanation_gates,
        "wall_seconds": result["wall_seconds"],
    }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
