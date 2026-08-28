#!/usr/bin/env python3
"""One-pass retained-prediction diagnostic for exact epoch-10 and epoch-40 bases."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parents[2]
TARGETED = PACKAGE_ROOT.parent / "route_b_v3_1_targeted_refinement_v1"
for path in (str(TARGETED), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from audit_v1 import (  # noqa: E402
    MATCH_RADIUS_M, decompose_person_fn, decompose_vehicle_fp, is_neutral,
    load_gt, load_ignore, load_predictions, match_frame, read_csv, score_arm, sha256,
)

CLASSES = ("vehicle", "person")


def write_json_x(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def full_pr_curve(experiment: Path, frame_ids: Sequence[str],
                  predictions: Mapping[str, Sequence[Mapping[str, Any]]],
                  gt: Mapping[str, Sequence[Mapping[str, Any]]], class_name: str,
                  ignore_cache: dict[str, Any]) -> dict[str, Any]:
    ordered = sorted(
        ((float(item["score"]), sample_id, item)
         for sample_id in frame_ids for item in predictions.get(sample_id, [])
         if item["class_name"] == class_name),
        key=lambda value: -value[0],
    )
    eligible = sum(1 for sample_id in frame_ids for item in gt.get(sample_id, [])
                   if item["class_name"] == class_name)
    active: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    frame_state: dict[str, tuple[int, int, int, int, float]] = {}
    total = [0, 0, eligible, 0, 0.0]  # tp, fp, fn, neutral, xy sum
    points: list[dict[str, Any]] = []
    index = 0
    while index < len(ordered):
        threshold = ordered[index][0]
        affected: set[str] = set()
        while index < len(ordered) and ordered[index][0] == threshold:
            _score, sample_id, item = ordered[index]
            active[sample_id].append(item)
            affected.add(sample_id)
            index += 1
        for sample_id in affected:
            old = frame_state.get(sample_id, (0, 0, sum(
                item["class_name"] == class_name for item in gt.get(sample_id, [])
            ), 0, 0.0))
            for slot in range(5):
                total[slot] -= old[slot]
            frame_pred = active[sample_id]
            frame_gt = [item for item in gt.get(sample_id, []) if item["class_name"] == class_name]
            used_pred, used_gt, mapping = match_frame(frame_pred, frame_gt)
            if sample_id not in ignore_cache:
                ignore_cache[sample_id] = load_ignore(experiment, "v010", sample_id)
            ignore = ignore_cache[sample_id]
            neutral = sum(
                pred_index not in used_pred and is_neutral(prediction, ignore)
                for pred_index, prediction in enumerate(frame_pred)
            )
            fp = len(frame_pred) - len(used_pred) - neutral
            xy_sum = sum(math.hypot(
                float(frame_pred[pred_index]["world_x"]) - float(frame_gt[gt_index]["world_x"]),
                float(frame_pred[pred_index]["world_y"]) - float(frame_gt[gt_index]["world_y"]),
            ) for pred_index, gt_index in mapping.items())
            state = (len(used_pred), fp, len(frame_gt) - len(used_gt), neutral, xy_sum)
            frame_state[sample_id] = state
            for slot in range(5):
                total[slot] += state[slot]
        tp, fp, fn, neutral, xy_sum = total
        precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
        points.append({
            "threshold": threshold, "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "ignored_predictions": int(neutral), "precision": precision, "recall": recall,
            "f1": 2.0 * precision * recall / max(1e-12, precision + recall),
            "xy_mae_m": xy_sum / max(1, tp),
        })
    return {
        "class_name": class_name, "eligible_gt": eligible,
        "persisted_score_floor": min((value[0] for value in ordered), default=None),
        "distinct_thresholds": len(points), "points": points,
    }


def person_fp_taxonomy(detail: Sequence[tuple]) -> dict[str, int]:
    counts = Counter({"PREDICTED_DUPLICATE": 0, "TWO_D_CORRECT_WORLD_WRONG": 0, "BACKGROUND_OR_OTHER": 0})
    for _sample_id, frame_predictions, frame_gt, fp_indices, _used, _mapping in detail:
        people = [item for item in frame_gt if item["class_name"] == "person"]
        for pred_index in fp_indices:
            prediction = frame_predictions[pred_index]
            if prediction["class_name"] != "person":
                continue
            duplicate = any(
                other_index != pred_index and other["class_name"] == "person"
                and (float(other["score"]) > float(prediction["score"])
                     or (float(other["score"]) == float(prediction["score"]) and other_index < pred_index))
                and math.hypot(float(other["world_x"]) - float(prediction["world_x"]),
                               float(other["world_y"]) - float(prediction["world_y"])) <= 2.0
                for other_index, other in enumerate(frame_predictions)
            )
            support_wrong = any(
                not (float(prediction["bbox_x1"]) <= target["box"][0]
                     or float(prediction["bbox_x0"]) >= target["box"][2]
                     or float(prediction["bbox_y1"]) <= target["box"][1]
                     or float(prediction["bbox_y0"]) >= target["box"][3])
                and math.hypot(float(prediction["world_x"]) - float(target["world_x"]),
                               float(prediction["world_y"]) - float(target["world_y"])) > MATCH_RADIUS_M
                for target in people
            )
            category = "PREDICTED_DUPLICATE" if duplicate else (
                "TWO_D_CORRECT_WORLD_WRONG" if support_wrong else "BACKGROUND_OR_OTHER"
            )
            counts[category] += 1
    return dict(counts)


def bucket(value: float, edges: Sequence[float], labels: Sequence[str]) -> str:
    for edge, label in zip(edges, labels):
        if value < edge:
            return label
    return labels[-1]


def strata_report(raw_gt: Sequence[dict[str, str]], primary_detail: Sequence[tuple]) -> dict[str, Any]:
    matched: dict[tuple[str, str], float] = {}
    for sample_id, predictions, frame_gt, _fp, _used_pred, mapping in primary_detail:
        for pred_index, gt_index in mapping.items():
            target = frame_gt[gt_index]
            if target["class_name"] == "person":
                matched[(sample_id, target["source_identity"])] = math.hypot(
                    float(predictions[pred_index]["world_x"]) - float(target["world_x"]),
                    float(predictions[pred_index]["world_y"]) - float(target["world_y"]),
                )
    track_lengths = Counter((row["experiment_id"], row["gt_actor_id"]) for row in raw_gt
                            if row["label"] == "person")
    aggregates: dict[str, dict[str, list[float]]] = {
        name: defaultdict(list) for name in
        ("distance", "area", "radar", "visibility", "occlusion_proxy", "episode", "track_length")
    }
    totals: dict[str, Counter[str]] = {name: Counter() for name in aggregates}
    for row in raw_gt:
        if row["label"] != "person":
            continue
        distance = float(row["gt_distance_m"])
        area = float(row["gt_bbox_area_px"])
        radar = int(float(row["radar_support_points"]))
        x, y, w, h = (float(row[key]) for key in ("gt_bbox_x", "gt_bbox_y", "gt_bbox_w", "gt_bbox_h"))
        strata = {
            "distance": bucket(distance, (10.0, 20.0, 30.0, math.inf), ("near_<10", "10_20", "20_30", "far_30_40")),
            "area": bucket(area, (400.0, 1200.0, math.inf), ("small_<400", "medium_400_1200", "large_>=1200")),
            "radar": "none" if radar == 0 else ("sparse_1_4" if radar < 5 else "supported_5_plus"),
            "visibility": "v010_visible_authoritative" if row["contract_state"] == "POSITIVE" else row["contract_state"],
            "occlusion_proxy": "border_truncated" if (x <= 1 or y <= 1 or x + w >= 1279 or y + h >= 719) else (
                "small_or_likely_occluded" if area < 400 else "interior_visible"
            ),
            "episode": row["experiment_id"],
            "track_length": bucket(float(track_lengths[(row["experiment_id"], row["gt_actor_id"])]),
                                   (10, 50, 150, math.inf), ("short_<10", "10_49", "50_149", "long_150_plus")),
        }
        key = (row["sample_id"], row["source_identity"])
        for name, label in strata.items():
            totals[name][label] += 1
            if key in matched:
                aggregates[name][label].append(matched[key])
    result: dict[str, Any] = {}
    for name in aggregates:
        result[name] = {
            label: {
                "eligible_gt": count, "matched": len(aggregates[name][label]),
                "recall": len(aggregates[name][label]) / max(1, count),
                "xy_mae_m": (sum(aggregates[name][label]) / len(aggregates[name][label]))
                            if aggregates[name][label] else None,
            } for label, count in sorted(totals[name].items())
        }
    result["visibility_note"] = "v010 encodes authoritative visible positives; area is reported separately."
    result["occlusion_note"] = "No oracle occlusion field exists; border/small-area labels are a registered diagnostic proxy only."
    return result


def analyse(experiment: Path, prediction_root: Path, frame_ids: Sequence[str],
            gt: Mapping[str, Sequence[Mapping[str, Any]]], raw_gt: Sequence[dict[str, str]]) -> dict[str, Any]:
    inference = json.loads((prediction_root / "inference_manifest.json").read_text())
    detections = prediction_root / "detections.csv"
    if sha256(detections) != inference["detections_sha256"]:
        raise RuntimeError(f"prediction hash mismatch: {prediction_root}")
    predictions, missing = load_predictions(detections)
    if missing:
        raise RuntimeError(f"missing/nonfinite prediction fields: {missing[:5]}")
    ignore_cache: dict[str, Any] = {}
    primary = score_arm(
        experiment=experiment, contract="v010", frame_ids=frame_ids,
        predictions=predictions, gt=gt, threshold=0.20, ignore_cache=ignore_cache, collect=True,
    )
    low = score_arm(
        experiment=experiment, contract="v010", frame_ids=frame_ids,
        predictions=predictions, gt=gt, threshold=0.02, ignore_cache=ignore_cache, collect=True,
    )
    curves = {
        name: full_pr_curve(experiment, frame_ids, predictions, gt, name, ignore_cache)
        for name in CLASSES
    }
    result = {
        "prediction_root": str(prediction_root), "inference_manifest": inference,
        "detections_sha256": sha256(detections),
        "primary_0_20": {key: value for key, value in primary.items() if key != "_detail"},
        "low_0_02": {key: value for key, value in low.items() if key != "_detail"},
        "full_precision_recall_from_persisted_score_floor": curves,
        "person_strata_at_0_20": strata_report(raw_gt, primary["_detail"]["person_fn"]),
        "taxonomy": {
            "vehicle_fp_at_0_20": decompose_vehicle_fp(primary["_detail"]["vehicle_fp"], gt),
            "person_fp_at_0_20": person_fp_taxonomy(primary["_detail"]["vehicle_fp"]),
            "person_fn_at_0_02": decompose_person_fn(low["_detail"]["person_fn"]),
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--epoch10-predictions", type=Path)
    parser.add_argument("--epoch40-predictions", type=Path)
    parser.add_argument("--single-predictions", type=Path)
    parser.add_argument("--single-label", default="selected")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    experiment = args.experiment.resolve()
    manifest = read_csv(experiment / "dataset/manifest.csv")
    frame_ids = [row["sample_id"] for row in manifest if row["split"] == "val"]
    if len(frame_ids) != 3345:
        raise RuntimeError("validation frame count drift")
    gt, _states = load_gt(experiment, "v010")
    raw_gt = read_csv(experiment / "contracts/v010/val/object_boxes.csv")
    if args.single_predictions is not None:
        if args.epoch10_predictions is not None or args.epoch40_predictions is not None:
            raise ValueError("single diagnostic cannot be combined with base diagnostic inputs")
        bases = {
            args.single_label: analyse(
                experiment, args.single_predictions.resolve(strict=True), frame_ids, gt, raw_gt,
            )
        }
    else:
        if args.epoch10_predictions is None or args.epoch40_predictions is None:
            raise ValueError("base diagnostic requires both epoch-10 and epoch-40 predictions")
        bases = {
            "epoch_010": analyse(experiment, args.epoch10_predictions.resolve(strict=True), frame_ids, gt, raw_gt),
            "epoch_040": analyse(experiment, args.epoch40_predictions.resolve(strict=True), frame_ids, gt, raw_gt),
        }
    result = {
        "schema": "route_b_v3_1_person_refinement_base_diagnostic_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "validation_frames": len(frame_ids),
        "inference_passes": {name: 1 for name in bases},
        "validation_used_for_training_mining_or_sampler": False,
        "bases": bases, "wall_seconds": time.monotonic() - started,
    }
    output = args.output.resolve() if args.output is not None else experiment / "BASE_DIAGNOSTIC.json"
    write_json_x(output, result)
    sentinel = (
        experiment / "SELECTED_DIAGNOSTIC_COMPLETE"
        if args.single_predictions is not None else experiment / "BASE_DIAGNOSTIC_COMPLETE"
    )
    sentinel.write_text("PASS\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output), "wall_seconds": result["wall_seconds"],
        "threshold_points": {base: {name: payload["distinct_thresholds"] for name, payload in detail["full_precision_recall_from_persisted_score_floor"].items()} for base, detail in bases.items()},
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
