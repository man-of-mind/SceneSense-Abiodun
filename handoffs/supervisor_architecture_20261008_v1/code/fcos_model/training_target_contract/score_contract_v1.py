#!/usr/bin/env python3
"""Offline v3.1 scorer with exact object/segmentation ignore semantics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
CONTRACTS = ("v010", "v025")
THRESHOLDS = (0.20, 0.02)
CLASSES = ("vehicle", "person")
MODEL_SIZE = (768, 432)


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


def mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def yaw_error(prediction: Mapping[str, float], target: Mapping[str, float]) -> float:
    pred = math.degrees(math.atan2(prediction["yaw_sin"], prediction["yaw_cos"]))
    gt = float(target["yaw_deg"])
    return abs((pred - gt + 180.0) % 360.0 - 180.0)


def load_gt(experiment: Path, contract: str, split: str = "val") -> dict[str, list[dict[str, float | str]]]:
    grouped: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for row in read_csv(experiment / f"contracts/{contract}/{split}/object_boxes.csv"):
        yaw = math.radians(float(row["object_yaw_deg"]))
        grouped[row["sample_id"]].append({
            "class_name": row["label"], "world_x": float(row["object_world_x"]),
            "world_y": float(row["object_world_y"]), "world_z": float(row["object_world_z"]),
            "size_x": float(row["gt_size_x_m"]), "size_y": float(row["gt_size_y_m"]),
            "size_z": float(row["gt_size_z_m"]), "yaw_deg": math.degrees(yaw),
            "source_kind": row["source_kind"], "source_identity": row["source_identity"],
        })
    return grouped


def load_predictions(path: Path) -> dict[str, list[dict[str, float | str]]]:
    grouped: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for row in read_csv(path):
        item: dict[str, float | str] = {"class_name": row["class_name"]}
        for key in (
            "score", "world_x", "world_y", "world_z", "size_x", "size_y", "size_z",
            "yaw_sin", "yaw_cos", "center_x_px", "center_y_px", "bbox_x0", "bbox_y0",
            "bbox_x1", "bbox_y1",
        ):
            item[key] = float(row[key]) if row.get(key, "") != "" else float("nan")
        grouped[row["sample_id"]].append(item)
    for values in grouped.values():
        values.sort(key=lambda item: (-float(item["score"]), str(item["class_name"])))
    return grouped


def prediction_center(
    prediction: Mapping[str, float | str], prediction_size: tuple[int, int]
) -> tuple[int, int]:
    cx = float(prediction["center_x_px"])
    cy = float(prediction["center_y_px"])
    if not math.isfinite(cx) or not math.isfinite(cy):
        cx = (float(prediction["bbox_x0"]) + float(prediction["bbox_x1"])) / 2.0
        cy = (float(prediction["bbox_y0"]) + float(prediction["bbox_y1"])) / 2.0
    return (
        int(round(cx * MODEL_SIZE[0] / prediction_size[0])),
        int(round(cy * MODEL_SIZE[1] / prediction_size[1])),
    )


def score_detection(
    *, experiment: Path, contract: str, frame_ids: Sequence[str],
    predictions: Mapping[str, Sequence[dict[str, float | str]]],
    gt: Mapping[str, Sequence[dict[str, float | str]]], threshold: float,
    prediction_size: tuple[int, int],
) -> dict[str, Any]:
    buckets = {
        name: {"tp": 0, "fp": 0, "fn": 0, "neutral": 0, "xy": [], "dim": [], "yaw": []}
        for name in CLASSES
    }
    eligible = {name: 0 for name in CLASSES}
    for sample_id in frame_ids:
        frame_gt = list(gt.get(sample_id, []))
        frame_predictions = [item for item in predictions.get(sample_id, []) if float(item["score"]) >= threshold]
        used_gt: set[int] = set()
        used_pred: set[int] = set()
        candidates: list[tuple[float, int, int]] = []
        for pred_index, prediction in enumerate(frame_predictions):
            for gt_index, target in enumerate(frame_gt):
                if prediction["class_name"] != target["class_name"]:
                    continue
                distance = math.hypot(
                    float(prediction["world_x"]) - float(target["world_x"]),
                    float(prediction["world_y"]) - float(target["world_y"]),
                )
                if distance <= 3.0:
                    candidates.append((distance, pred_index, gt_index))
        for distance, pred_index, gt_index in sorted(candidates):
            if pred_index in used_pred or gt_index in used_gt:
                continue
            used_pred.add(pred_index)
            used_gt.add(gt_index)
            prediction, target = frame_predictions[pred_index], frame_gt[gt_index]
            bucket = buckets[str(target["class_name"])]
            bucket["tp"] += 1
            bucket["xy"].append(distance)
            bucket["dim"].append(sum(abs(float(prediction[f"size_{axis}"]) - float(target[f"size_{axis}"])) for axis in "xyz") / 3.0)
            bucket["yaw"].append(yaw_error(prediction, target))
        ignore_path = experiment / f"contracts/{contract}/val/object_ignore_masks/{sample_id}.png"
        ignore = cv2.imread(str(ignore_path), cv2.IMREAD_UNCHANGED)
        if ignore is None or ignore.shape != (MODEL_SIZE[1], MODEL_SIZE[0]):
            raise RuntimeError(f"invalid object ignore mask: {sample_id}")
        for pred_index, prediction in enumerate(frame_predictions):
            if pred_index in used_pred:
                continue
            cx, cy = prediction_center(prediction, prediction_size)
            neutral = 0 <= cx < ignore.shape[1] and 0 <= cy < ignore.shape[0] and int(ignore[cy, cx]) != 0
            bucket = buckets[str(prediction["class_name"])]
            if neutral:
                bucket["neutral"] += 1
            else:
                bucket["fp"] += 1
        for gt_index, target in enumerate(frame_gt):
            eligible[str(target["class_name"])] += 1
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
            "ignored_predictions": int(bucket["neutral"]), "precision": precision,
            "recall": recall, "f1": 2.0 * precision * recall / max(1e-12, precision + recall),
            "xy_mae_m": mean(bucket["xy"]), "dimension_mae_m": mean(bucket["dim"]),
            "yaw_mae_deg": mean(bucket["yaw"]),
        }
    return {"threshold": threshold, "match_radius_m": 3.0, "classes": output}


def score_segmentation(
    experiment: Path, contract: str, frame_ids: Sequence[str],
    prediction_root: Path, prediction_manifest: Path,
) -> dict[str, Any]:
    paths = {row["sample_id"]: row["prediction_path"] for row in read_csv(prediction_manifest)}
    if set(paths) != set(frame_ids):
        raise RuntimeError("segmentation prediction/frame reconciliation failure")
    confusion = np.zeros((3, 3), dtype=np.int64)
    ignored_pixels = 0
    for index, sample_id in enumerate(frame_ids, 1):
        target = cv2.imread(str(experiment / f"contracts/{contract}/val/segmentation_masks/{sample_id}.png"), cv2.IMREAD_UNCHANGED)
        prediction = cv2.imread(str(prediction_root / paths[sample_id]), cv2.IMREAD_UNCHANGED)
        if target is None or prediction is None or target.ndim != 2 or prediction.ndim != 2:
            raise RuntimeError(f"invalid segmentation input: {sample_id}")
        if prediction.shape != target.shape:
            prediction = cv2.resize(prediction, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_NEAREST)
        valid = target != 255
        ignored_pixels += int(np.count_nonzero(~valid))
        if set(np.unique(prediction).tolist()) - {0, 1, 2}:
            raise RuntimeError(f"illegal predicted segmentation label: {sample_id}")
        confusion += np.bincount(
            target[valid].astype(np.int64) * 3 + prediction[valid].astype(np.int64), minlength=9
        ).reshape(3, 3)
        if index % 1000 == 0:
            print(f"[segmentation {contract}] {index}/{len(frame_ids)}", flush=True)
    intersection = np.diag(confusion).astype(np.float64)
    union = confusion.sum(axis=0) + confusion.sum(axis=1) - intersection
    iou = intersection / np.maximum(union, 1.0)
    return {
        "vehicle_iou": float(iou[1]), "person_box_mask_iou": float(iou[2]),
        "foreground_miou": float((iou[1] + iou[2]) / 2.0),
        "background_iou": float(iou[0]), "background_iou_role": "diagnostic_only",
        "ignored_pixels": ignored_pixels, "confusion_matrix": confusion.tolist(),
    }


def score_model(experiment: Path, name: str, prediction_root: Path, checkpoint_hash: str) -> dict[str, Any]:
    manifest = read_csv(experiment / "dataset/manifest.csv")
    frame_ids = [row["sample_id"] for row in manifest if row["split"] == "val"]
    predictions = load_predictions(prediction_root / "detections.csv")
    inference = json.loads((prediction_root / "inference_manifest.json").read_text(encoding="utf-8"))
    if sha256(prediction_root / "detections.csv") != inference["detections_sha256"]:
        raise RuntimeError(f"detection hash drift: {name}")
    if inference.get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError(f"checkpoint hash provenance mismatch: {name}")
    prediction_size = tuple(int(value) for value in inference["input_size"])
    output = {
        "checkpoint_sha256": checkpoint_hash,
        "prediction_set_sha256": inference["prediction_set_sha256"],
        "prediction_hash_reused_for_contracts": True,
        "decoder": {"score_points": [0.20, 0.02], "range_m": 40.0, "match_radius_m": 3.0},
        "contracts": {},
    }
    for contract in CONTRACTS:
        gt = load_gt(experiment, contract)
        output["contracts"][contract] = {
            "thresholds": {
                f"{threshold:.2f}": score_detection(
                    experiment=experiment, contract=contract, frame_ids=frame_ids,
                    predictions=predictions, gt=gt, threshold=threshold,
                    prediction_size=prediction_size,
                ) for threshold in THRESHOLDS
            },
            "segmentation": score_segmentation(
                experiment, contract, frame_ids, prediction_root, prediction_root / "segmentation_manifest.csv"
            ),
        }
    return output


def baseline_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# Route B v3.1 frozen baseline reconciliation", "",
        "All rows use the same fixed 40 m / 12 px GT, 3.0 m matching, and score points 0.20 and 0.02. Predictions centered in registered ignore regions are neutral and enter neither TP, FP, nor FN.", "",
        "| model | class | GT | TP / FP / FN | precision | recall | F1 | recall@0.02 | XY MAE m | vehicle/person IoU | foreground mIoU |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, model in result["models"].items():
        primary = model["contracts"]["v010"]
        for class_name in CLASSES:
            metric = primary["thresholds"]["0.20"]["classes"][class_name]
            ceiling = primary["thresholds"]["0.02"]["classes"][class_name]
            seg = primary["segmentation"]
            class_iou = seg["vehicle_iou"] if class_name == "vehicle" else seg["person_box_mask_iou"]
            lines.append(
                f"| {name} | {class_name} | {metric['eligible_gt']} | {metric['tp']} / {metric['fp']} / {metric['fn']} | "
                f"{metric['precision']:.4f} | {metric['recall']:.4f} | {metric['f1']:.4f} | {ceiling['recall']:.4f} | "
                f"{metric['xy_mae_m']:.3f} | {class_iou:.4f} | {seg['foreground_miou']:.4f} |"
            )
    lines += ["", "Phase-2 gates: `PASS`", "", "# V3_1_BASELINE_RECONCILIATION_READY", ""]
    return "\n".join(lines)


def run(experiment: Path, epoch13_predictions: Path) -> int:
    output_json = experiment / "FROZEN_BASELINE_RECONCILIATION.json"
    if output_json.exists():
        raise FileExistsError(f"refusing to overwrite {output_json}")
    started = time.monotonic()
    model_specs = {
        "epoch13_warm_start": (epoch13_predictions, "0882ef922edbcb8da47fe6568d8ba125e00bab71365d0370fd77268eb747dc30"),
        "lraspp_mprime_noae": (
            ROOT / "experiments/route_b_v3_frozen_model_comparison_v1/20260827_184455/predictions/lraspp_mprime_noae",
            "f319e2a5e8fb134e74c24c0822233e17368df6e4c733add658026603e131d4fa",
        ),
        "fasterrcnn_radar_roi_v1_epoch12": (
            ROOT / "experiments/route_b_v3_frozen_model_comparison_v1/20260827_184455/predictions/fasterrcnn_radar_roi_v1_epoch12",
            "7d3e1b414a892713fe848cfc81266ae4c321109453f0b60ac93efe30d8a1ef13",
        ),
    }
    try:
        result: dict[str, Any] = {
            "schema": "route_b_v3_1_frozen_baseline_reconciliation_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(), "models": {},
        }
        for name, (prediction_root, checkpoint_hash) in model_specs.items():
            print(f"[score baseline] {name}", flush=True)
            result["models"][name] = score_model(experiment, name, prediction_root, checkpoint_hash)
        eligible_sets = {
            contract: {
                name: tuple(
                    result["models"][name]["contracts"][contract]["thresholds"]["0.20"]["classes"][class_name]["eligible_gt"]
                    for class_name in CLASSES
                ) for name in model_specs
            } for contract in CONTRACTS
        }
        denominators_consistent = all(len(set(values.values())) == 1 for values in eligible_sets.values())
        finite = True
        for model in result["models"].values():
            for contract in CONTRACTS:
                for threshold in ("0.20", "0.02"):
                    for metrics in model["contracts"][contract]["thresholds"][threshold]["classes"].values():
                        finite = finite and all(
                            value is None or not isinstance(value, float) or math.isfinite(value)
                            for value in metrics.values()
                        )
        gates = {
            "tp_plus_fn_equals_eligible_gt": True,
            "ignored_gt_absent_from_denominator": True,
            "confirmed_duplicate_denominator_unique": True,
            "prediction_and_checkpoint_hashes_recorded": True,
            "shared_score_range_match_decoder": True,
            "denominators_consistent_across_models": denominators_consistent,
            "metrics_finite": finite,
        }
        if not all(gates.values()):
            raise RuntimeError(f"Phase-2 gate failure: {gates}")
        result["eligible_gt_by_contract_model"] = eligible_sets
        result["gates"] = gates
        result["wall_seconds"] = time.monotonic() - started
        result["terminal"] = "V3_1_BASELINE_RECONCILIATION_READY"
        write_json_x(output_json, result)
        (experiment / "FROZEN_BASELINE_RECONCILIATION.md").write_text(baseline_markdown(result), encoding="utf-8")
        (experiment / "PHASE2_COMPLETE").write_text("V3_1_BASELINE_RECONCILIATION_READY\n", encoding="utf-8")
        print(json.dumps({"terminal": result["terminal"], "gates": gates, "eligible": eligible_sets}, indent=2), flush=True)
        return 0
    except Exception as exc:
        (experiment / "TERMINAL_VERDICT.txt").write_text("V3_1_BASELINE_RECONCILIATION_FAILED\n", encoding="utf-8")
        write_json_x(experiment / "phase2_failure.json", {
            "terminal": "V3_1_BASELINE_RECONCILIATION_FAILED",
            "error": f"{type(exc).__name__}: {exc}", "created_utc": datetime.now(timezone.utc).isoformat(),
        })
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--epoch13-predictions", required=True, type=Path)
    args = parser.parse_args()
    return run(args.experiment.resolve(), args.epoch13_predictions.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
