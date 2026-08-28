#!/usr/bin/env python3
"""Phase A: retained-prediction audit for the Route B v3.1 clean epoch-20 baseline.

Uses only the retained validation predictions. No inference, no training, no test split.
Matching semantics are reproduced exactly from the frozen v3.1 scorer
(``route_b_v3_1_clean_base_v1/score_contract_v1.py``) and are reconciled against the
published epoch-20 TP/FP/FN counts as a hard gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.route_b_v3_1_targeted_refinement_v1.postprocess_v1 import (  # noqa: E402
    VEHICLE_WORLD_NMS_RADIUS_M,
    apply_arm,
)

MODEL_SIZE = (768, 432)
SOURCE_SIZE = (1280, 720)
CLASSES = ("vehicle", "person")
MATCH_RADIUS_M = 3.0
DUPLICATE_RADIUS_M = 2.0
PRIMARY_CONTRACT = "v010"

PUBLISHED_EPOCH20 = {
    "vehicle": {"0.20": {"tp": 6849, "fp": 7390, "fn": 2876, "ignored": 2744},
                "0.02": {"tp": 7256, "fp": 10611, "fn": 2469, "ignored": 3841}},
    "person": {"0.20": {"tp": 1560, "fp": 1822, "fn": 2312, "ignored": 925},
               "0.02": {"tp": 1766, "fp": 5651, "fn": 2106, "ignored": 1892}},
}
EXPECTED_CHECKPOINT_SHA = "88b34a69eeec7bf2f6444e70a0e346c365b979e6936d277cb0c75e8cd747aa1d"

PREDICTION_FIELDS = (
    "score", "world_x", "world_y", "world_z", "size_x", "size_y", "size_z",
    "yaw_sin", "yaw_cos", "center_x_px", "center_y_px",
    "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json_x(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def load_gt(experiment: Path, contract: str) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Ground truth grouped by frame, carrying the 2D image box in model pixel space."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    states: dict[str, int] = defaultdict(int)
    sx = MODEL_SIZE[0] / SOURCE_SIZE[0]
    sy = MODEL_SIZE[1] / SOURCE_SIZE[1]
    for row in read_csv(experiment / f"contracts/{contract}/val/object_boxes.csv"):
        states[row["contract_state"]] += 1
        x0, y0 = float(row["gt_bbox_x"]), float(row["gt_bbox_y"])
        grouped[row["sample_id"]].append({
            "class_name": row["label"],
            "world_x": float(row["object_world_x"]), "world_y": float(row["object_world_y"]),
            "box": (x0 * sx, y0 * sy, (x0 + float(row["gt_bbox_w"])) * sx,
                    (y0 + float(row["gt_bbox_h"])) * sy),
            "area_px": float(row["gt_bbox_area_px"]),
            "source_identity": row["source_identity"],
        })
    return grouped, dict(states)


def load_predictions(path: Path) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing: list[str] = []
    for row in read_csv(path):
        item: dict[str, Any] = {"class_name": row["class_name"]}
        for key in PREDICTION_FIELDS:
            raw = row.get(key, "")
            if raw == "" or raw is None:
                missing.append(f"{row['sample_id']}:{row.get('prediction_index')}:{key}")
                item[key] = float("nan")
            else:
                item[key] = float(raw)
            if not math.isfinite(float(item[key])):
                missing.append(f"{row['sample_id']}:{row.get('prediction_index')}:{key}:nonfinite")
        grouped[row["sample_id"]].append(item)
    for values in grouped.values():
        values.sort(key=lambda item: (-float(item["score"]), str(item["class_name"])))
    return grouped, missing


def prediction_center(prediction: Mapping[str, Any]) -> tuple[int, int]:
    """Identical to the frozen scorer: predictions are already in model pixel space."""
    cx, cy = float(prediction["center_x_px"]), float(prediction["center_y_px"])
    if not math.isfinite(cx) or not math.isfinite(cy):
        cx = (float(prediction["bbox_x0"]) + float(prediction["bbox_x1"])) / 2.0
        cy = (float(prediction["bbox_y0"]) + float(prediction["bbox_y1"])) / 2.0
    return int(round(cx)), int(round(cy))


def boxes_overlap(pred: Mapping[str, Any], box: tuple[float, float, float, float]) -> bool:
    x0 = max(float(pred["bbox_x0"]), box[0])
    y0 = max(float(pred["bbox_y0"]), box[1])
    x1 = min(float(pred["bbox_x1"]), box[2])
    y1 = min(float(pred["bbox_y1"]), box[3])
    return x1 > x0 and y1 > y0


def center_inside(pred: Mapping[str, Any], box: tuple[float, float, float, float]) -> bool:
    cx, cy = float(pred["center_x_px"]), float(pred["center_y_px"])
    if not (math.isfinite(cx) and math.isfinite(cy)):
        return False
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def two_d_support(pred: Mapping[str, Any], box: tuple[float, float, float, float]) -> bool:
    return center_inside(pred, box) or boxes_overlap(pred, box)


def match_frame(
    frame_predictions: Sequence[Mapping[str, Any]], frame_gt: Sequence[Mapping[str, Any]]
) -> tuple[set[int], set[int], dict[int, int]]:
    """Greedy nearest-first same-class matching within 3.0 m, exactly as the frozen scorer."""
    candidates: list[tuple[float, int, int]] = []
    for pred_index, prediction in enumerate(frame_predictions):
        for gt_index, target in enumerate(frame_gt):
            if prediction["class_name"] != target["class_name"]:
                continue
            distance = math.hypot(
                float(prediction["world_x"]) - float(target["world_x"]),
                float(prediction["world_y"]) - float(target["world_y"]),
            )
            if distance <= MATCH_RADIUS_M:
                candidates.append((distance, pred_index, gt_index))
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    pred_to_gt: dict[int, int] = {}
    for _distance, pred_index, gt_index in sorted(candidates):
        if pred_index in used_pred or gt_index in used_gt:
            continue
        used_pred.add(pred_index)
        used_gt.add(gt_index)
        pred_to_gt[pred_index] = gt_index
    return used_pred, used_gt, pred_to_gt


def load_ignore(experiment: Path, contract: str, sample_id: str):
    path = experiment / f"contracts/{contract}/val/object_ignore_masks/{sample_id}.png"
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None or mask.shape != (MODEL_SIZE[1], MODEL_SIZE[0]):
        raise RuntimeError(f"invalid object ignore mask: {sample_id}")
    return mask


def is_neutral(prediction: Mapping[str, Any], ignore) -> bool:
    cx, cy = prediction_center(prediction)
    return 0 <= cx < ignore.shape[1] and 0 <= cy < ignore.shape[0] and int(ignore[cy, cx]) != 0


def score_arm(
    *, experiment: Path, contract: str, frame_ids: Sequence[str],
    predictions: Mapping[str, Sequence[Mapping[str, Any]]], gt: Mapping[str, Sequence[Mapping[str, Any]]],
    threshold: float, ignore_cache: dict[str, Any], collect: bool = False,
) -> dict[str, Any]:
    buckets = {name: {"tp": 0, "fp": 0, "fn": 0, "neutral": 0, "xy": []} for name in CLASSES}
    eligible = {name: 0 for name in CLASSES}
    detail: dict[str, Any] = {"vehicle_fp": [], "person_fn": []} if collect else {}
    for sample_id in frame_ids:
        frame_gt = list(gt.get(sample_id, []))
        frame_predictions = [
            item for item in predictions.get(sample_id, []) if float(item["score"]) >= threshold
        ]
        used_pred, used_gt, pred_to_gt = match_frame(frame_predictions, frame_gt)
        for pred_index, gt_index in pred_to_gt.items():
            prediction, target = frame_predictions[pred_index], frame_gt[gt_index]
            bucket = buckets[str(target["class_name"])]
            bucket["tp"] += 1
            bucket["xy"].append(math.hypot(
                float(prediction["world_x"]) - float(target["world_x"]),
                float(prediction["world_y"]) - float(target["world_y"]),
            ))
        if sample_id not in ignore_cache:
            ignore_cache[sample_id] = load_ignore(experiment, contract, sample_id)
        ignore = ignore_cache[sample_id]
        fp_indices: list[int] = []
        for pred_index, prediction in enumerate(frame_predictions):
            if pred_index in used_pred:
                continue
            bucket = buckets[str(prediction["class_name"])]
            if is_neutral(prediction, ignore):
                bucket["neutral"] += 1
            else:
                bucket["fp"] += 1
                fp_indices.append(pred_index)
        for gt_index, target in enumerate(frame_gt):
            eligible[str(target["class_name"])] += 1
        if collect:
            detail["vehicle_fp"].append((sample_id, frame_predictions, frame_gt, fp_indices, used_pred, pred_to_gt))
            detail["person_fn"].append((sample_id, frame_predictions, frame_gt, used_gt, used_pred, pred_to_gt))
        for gt_index, target in enumerate(frame_gt):
            if gt_index not in used_gt:
                buckets[str(target["class_name"])]["fn"] += 1
    output: dict[str, Any] = {}
    for class_name in CLASSES:
        bucket = buckets[class_name]
        tp, fp, fn = int(bucket["tp"]), int(bucket["fp"]), int(bucket["fn"])
        if tp + fn != eligible[class_name]:
            raise RuntimeError(f"TP+FN denominator failure: {contract}/{threshold}/{class_name}")
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        output[class_name] = {
            "eligible_gt": eligible[class_name], "tp": tp, "fp": fp, "fn": fn,
            "ignored_predictions": int(bucket["neutral"]),
            "precision": precision, "recall": recall,
            "f1": 2.0 * precision * recall / max(1e-12, precision + recall),
            "xy_mae_m": (sum(bucket["xy"]) / len(bucket["xy"])) if bucket["xy"] else None,
        }
    result: dict[str, Any] = {"threshold": threshold, "classes": output}
    if collect:
        result["_detail"] = detail
    return result


def decompose_vehicle_fp(detail: Sequence[tuple], gt_all: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Registered, mutually exclusive vehicle FP labels at score 0.20 (priority ordered)."""
    counts = {"PREDICTED_DUPLICATE": 0, "TWO_D_CORRECT_WORLD_WRONG": 0, "BACKGROUND_OR_OTHER": 0}
    raw = {"duplicate_any": 0, "two_d_any": 0, "both": 0}
    for sample_id, frame_predictions, frame_gt, fp_indices, _used_pred, _pred_to_gt in detail:
        vehicle_gt = [target for target in frame_gt if target["class_name"] == "vehicle"]
        for pred_index in fp_indices:
            prediction = frame_predictions[pred_index]
            if str(prediction["class_name"]) != "vehicle":
                continue
            x, y = float(prediction["world_x"]), float(prediction["world_y"])
            score = float(prediction["score"])
            duplicate = False
            for other_index, other in enumerate(frame_predictions):
                if other_index == pred_index or str(other["class_name"]) != "vehicle":
                    continue
                other_score = float(other["score"])
                higher = other_score > score or (
                    other_score == score and other_index < pred_index
                )
                if not higher:
                    continue
                if math.hypot(x - float(other["world_x"]), y - float(other["world_y"])) <= DUPLICATE_RADIUS_M:
                    duplicate = True
                    break
            two_d = any(
                two_d_support(prediction, target["box"])
                and math.hypot(x - float(target["world_x"]), y - float(target["world_y"])) > MATCH_RADIUS_M
                for target in vehicle_gt
            )
            raw["duplicate_any"] += int(duplicate)
            raw["two_d_any"] += int(two_d)
            raw["both"] += int(duplicate and two_d)
            if duplicate:
                counts["PREDICTED_DUPLICATE"] += 1
            elif two_d:
                counts["TWO_D_CORRECT_WORLD_WRONG"] += 1
            else:
                counts["BACKGROUND_OR_OTHER"] += 1
    return {"counts": counts, "unprioritised_overlap": raw}


def decompose_person_fn(detail: Sequence[tuple]) -> dict[str, Any]:
    """Registered, mutually exclusive person FN labels at score 0.02 (priority ordered)."""
    counts = {"MATCHING_CONTENTION": 0, "CENTER_PRESENT_WORLD_WRONG": 0, "HEATMAP_CENTER_MISS": 0}
    for sample_id, frame_predictions, frame_gt, used_gt, used_pred, pred_to_gt in detail:
        for gt_index, target in enumerate(frame_gt):
            if gt_index in used_gt or str(target["class_name"]) != "person":
                continue
            supporters = [
                pred_index for pred_index, prediction in enumerate(frame_predictions)
                if str(prediction["class_name"]) == "person" and two_d_support(prediction, target["box"])
            ]
            if not supporters:
                counts["HEATMAP_CENTER_MISS"] += 1
            elif any(
                pred_index in used_pred and pred_to_gt.get(pred_index) != gt_index
                for pred_index in supporters
            ):
                counts["MATCHING_CONTENTION"] += 1
            else:
                counts["CENTER_PRESENT_WORLD_WRONG"] += 1
    return {"counts": counts}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-experiment", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    baseline = args.baseline_experiment.resolve()
    output_dir = args.output.resolve()
    started = time.monotonic()

    checkpoint = baseline / "checkpoints/route_b_v3_1_clean_noae_stage2_v1/epoch_020.pt"
    prediction_root = baseline / "predictions/trained_epoch_020"
    inference = json.loads((prediction_root / "inference_manifest.json").read_text(encoding="utf-8"))
    checkpoint_hash = sha256(checkpoint)
    detections_hash = sha256(prediction_root / "detections.csv")
    hash_gate = (
        checkpoint_hash == EXPECTED_CHECKPOINT_SHA
        and inference["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA
        and detections_hash == inference["detections_sha256"]
        and tuple(inference["input_size"]) == MODEL_SIZE
    )

    manifest = read_csv(baseline / "dataset/manifest.csv")
    frame_ids = [row["sample_id"] for row in manifest if row["split"] == "val"]
    gt, gt_states = load_gt(baseline, PRIMARY_CONTRACT)
    predictions, missing_fields = load_predictions(prediction_root / "detections.csv")
    ignore_cache: dict[str, Any] = {}

    raw_020 = score_arm(experiment=baseline, contract=PRIMARY_CONTRACT, frame_ids=frame_ids,
                        predictions=predictions, gt=gt, threshold=0.20,
                        ignore_cache=ignore_cache, collect=True)
    raw_002 = score_arm(experiment=baseline, contract=PRIMARY_CONTRACT, frame_ids=frame_ids,
                        predictions=predictions, gt=gt, threshold=0.02,
                        ignore_cache=ignore_cache, collect=True)

    reconciliation = {}
    reconciled = True
    for class_name in CLASSES:
        for label, arm in (("0.20", raw_020), ("0.02", raw_002)):
            got = arm["classes"][class_name]
            want = PUBLISHED_EPOCH20[class_name][label]
            ok = (got["tp"] == want["tp"] and got["fp"] == want["fp"]
                  and got["fn"] == want["fn"] and got["ignored_predictions"] == want["ignored"])
            reconciled = reconciled and ok
            reconciliation[f"{class_name}@{label}"] = {
                "published": want,
                "recomputed": {"tp": got["tp"], "fp": got["fp"], "fn": got["fn"],
                               "ignored": got["ignored_predictions"]},
                "exact": ok,
            }

    vehicle_fp = decompose_vehicle_fp(raw_020["_detail"]["vehicle_fp"], gt)
    person_fn = decompose_person_fn(raw_002["_detail"]["person_fn"])
    vehicle_fp_total = sum(vehicle_fp["counts"].values())
    person_fn_total = sum(person_fn["counts"].values())
    vehicle_fp_denominator = raw_020["classes"]["vehicle"]["fp"]
    person_fn_denominator = raw_002["classes"]["person"]["fn"]

    gates = {
        "exact_reconciliation": reconciled,
        "no_missing_prediction_fields": not missing_fields,
        "checkpoint_and_prediction_hashes_verified": hash_gate,
        "ignored_predictions_neutral": (
            raw_020["classes"]["vehicle"]["ignored_predictions"] == PUBLISHED_EPOCH20["vehicle"]["0.20"]["ignored"]
            and raw_020["classes"]["person"]["ignored_predictions"] == PUBLISHED_EPOCH20["person"]["0.20"]["ignored"]
        ),
        "vehicle_fp_labels_sum_to_denominator": vehicle_fp_total == vehicle_fp_denominator,
        "person_fn_labels_sum_to_denominator": person_fn_total == person_fn_denominator,
    }

    # Registered predicted-only postprocessor arm, evaluated on the baseline predictions.
    nms_predictions = apply_arm(predictions, "VEHICLE_WORLD_NMS_2M")
    nms_020 = score_arm(experiment=baseline, contract=PRIMARY_CONTRACT, frame_ids=frame_ids,
                        predictions=nms_predictions, gt=gt, threshold=0.20, ignore_cache=ignore_cache)
    nms_002 = score_arm(experiment=baseline, contract=PRIMARY_CONTRACT, frame_ids=frame_ids,
                        predictions=nms_predictions, gt=gt, threshold=0.02, ignore_cache=ignore_cache)

    base_v, nms_v = raw_020["classes"]["vehicle"], nms_020["classes"]["vehicle"]
    base_p, nms_p = raw_020["classes"]["person"], nms_020["classes"]["person"]
    person_unchanged = all(
        base_p[key] == nms_p[key] for key in ("tp", "fp", "fn", "ignored_predictions")
    ) and raw_002["classes"]["person"]["recall"] == nms_002["classes"]["person"]["recall"]
    nms_gates = {
        "vehicle_precision_gain_ge_0_05": (nms_v["precision"] - base_v["precision"]) >= 0.05,
        "vehicle_recall_loss_le_0_01": (base_v["recall"] - nms_v["recall"]) <= 0.01,
        "person_metrics_unchanged": person_unchanged,
        "suppression_reads_predictions_only": True,
    }
    nms_eligible = all(nms_gates.values())

    licensed = (
        gates["exact_reconciliation"]
        and person_fn["counts"]["HEATMAP_CENTER_MISS"] >= 0.50 * person_fn_denominator
    )

    for arm in (raw_020, raw_002):
        arm.pop("_detail", None)

    result = {
        "schema": "route_b_v3_1_targeted_refinement_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_experiment": str(baseline),
        "checkpoint_sha256": checkpoint_hash,
        "detections_sha256": detections_hash,
        "prediction_set_sha256": inference["prediction_set_sha256"],
        "contract": PRIMARY_CONTRACT,
        "validation_frames": len(frame_ids),
        "gt_contract_states": gt_states,
        "missing_prediction_fields": missing_fields[:20],
        "missing_prediction_field_count": len(missing_fields),
        "baseline": {"0.20": raw_020, "0.02": raw_002},
        "vehicle_fp_decomposition_at_0_20": {
            **vehicle_fp,
            "denominator": vehicle_fp_denominator,
            "total_labelled": vehicle_fp_total,
            "ignore_neutral_excluded_from_fp": raw_020["classes"]["vehicle"]["ignored_predictions"],
            "percentages": {
                key: 100.0 * value / max(1, vehicle_fp_denominator)
                for key, value in vehicle_fp["counts"].items()
            },
        },
        "person_fn_decomposition_at_0_02": {
            **person_fn,
            "denominator": person_fn_denominator,
            "total_labelled": person_fn_total,
            "percentages": {
                key: 100.0 * value / max(1, person_fn_denominator)
                for key, value in person_fn["counts"].items()
            },
        },
        "vehicle_world_nms_2m": {
            "radius_m": VEHICLE_WORLD_NMS_RADIUS_M,
            "score_threshold": 0.20,
            "person_suppression_enabled": False,
            "baseline_0_20": {"vehicle": base_v, "person": base_p},
            "nms_0_20": {"vehicle": nms_v, "person": nms_p},
            "nms_0_02_recall": {
                "vehicle": nms_002["classes"]["vehicle"]["recall"],
                "person": nms_002["classes"]["person"]["recall"],
            },
            "gates": nms_gates,
            "verdict": "VEHICLE_WORLD_NMS_2M_ELIGIBLE" if nms_eligible else "VEHICLE_WORLD_NMS_2M_REJECTED",
        },
        "audit_gates": gates,
        "class_balanced_training_licensed": bool(licensed),
        "license_rule": "HEATMAP_CENTER_MISS >= 50% of person FN at score 0.02",
        "wall_seconds": time.monotonic() - started,
    }

    if not all(gates.values()):
        result["terminal"] = "LRASPP_TARGETED_AUDIT_FAILED"
    elif not licensed:
        result["terminal"] = "LRASPP_CLASS_BALANCE_NOT_LICENSED"
    else:
        result["terminal"] = "AUDIT_LICENSES_CLASS_BALANCED_CONTINUATION"

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_x(output_dir / "PHASE_A_AUDIT.json", result)
    (output_dir / "PHASE_A_COMPLETE").write_text(result["terminal"] + "\n", encoding="utf-8")
    print(json.dumps({
        "terminal": result["terminal"],
        "gates": gates,
        "vehicle_fp": result["vehicle_fp_decomposition_at_0_20"]["counts"],
        "vehicle_fp_pct": result["vehicle_fp_decomposition_at_0_20"]["percentages"],
        "person_fn": result["person_fn_decomposition_at_0_02"]["counts"],
        "person_fn_pct": result["person_fn_decomposition_at_0_02"]["percentages"],
        "nms": result["vehicle_world_nms_2m"]["verdict"],
        "nms_gates": nms_gates,
        "licensed": bool(licensed),
    }, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
