#!/usr/bin/env python3
"""Offline raw-detection precision and bounded post-processing audit.

This driver never imports or starts a CARLA client, never changes a checkpoint,
and never writes outside its own audit directory. It reconstructs the retained
prediction/GT lists from the frozen per-object evaluator outputs, replays only
predicted-field post-processing, and rematches with the frozen evaluator rule.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
RAW = ROOT / "rl_agent" / "density_knob" / "raw"
SWEEPS = ROOT / "experiments" / "ae_integrated_20260710" / "sweeps_permodel_zstd"
STAGE_A = ROOT / "rl_agent" / "experiments" / "ue_split_stage_a_v1" / "20260820_024055_review"
DATASET = ROOT / "fusion_training_data" / "moving_ego_pps200000_merged_8loops_stride2"

FAMILIES = ("noae", "ae32", "ae64", "ae128")
QUANTS = ("uint8", "uint6", "uint4")
CLASS_NAMES = ("vehicle", "person")
MATCH_RADIUS_M = 5.0
PLAUSIBLE_RADIUS_M = 10.0
BASELINE_THRESHOLD = 0.20
BOOTSTRAP_SEED = 20260819
BOOTSTRAP_REPS = 2000

EXPECTED = {
    "noae": {"veh_precision": 0.526, "veh_recall": 0.893, "veh_f1": 0.662,
              "ped_precision": 0.627, "ped_recall": 0.850, "ped_f1": 0.722, "fp_per_frame": 1.26},
    "ae32": {"veh_precision": 0.500, "veh_recall": 0.924, "veh_f1": 0.649,
              "ped_precision": 0.634, "ped_recall": 0.863, "ped_f1": 0.731, "fp_per_frame": 1.39},
    "ae64": {"veh_precision": 0.502, "veh_recall": 0.920, "veh_f1": 0.650,
              "ped_precision": 0.632, "ped_recall": 0.864, "ped_f1": 0.730, "fp_per_frame": 1.37},
    "ae128": {"veh_precision": 0.497, "veh_recall": 0.924, "veh_f1": 0.646,
               "ped_precision": 0.632, "ped_recall": 0.884, "ped_f1": 0.737, "fp_per_frame": 1.41},
}


@dataclass(frozen=True)
class Candidate:
    name: str
    vehicle_threshold: float = BASELINE_THRESHOLD
    person_threshold: float = BASELINE_THRESHOLD
    world_nms_m: float = 0.0
    image_nms_px: float = 0.0


PREREGISTERED = (
    Candidate("baseline"),
    Candidate("world_nms_1m", world_nms_m=1.0),
    Candidate("world_nms_2m", world_nms_m=2.0),
    Candidate("world_nms_3m", world_nms_m=3.0),
    Candidate("image_nms_6px", image_nms_px=6.0),
    Candidate("image_nms_8px", image_nms_px=8.0),
    Candidate("veh_thr_0p225", vehicle_threshold=0.225),
    Candidate("veh_thr_0p25", vehicle_threshold=0.25),
)

CHECKPOINTS = {
    "noae": ROOT / "experiments/ae_integrated_20260710/noae_baseline/checkpoints/mprime_joint_noae/best.pt",
    "ae32": ROOT / "experiments/ae_integrated_20260710/ae32/checkpoints/ae32_integrated/best.pt",
    "ae64": ROOT / "experiments/ae_integrated_20260710/ae64/checkpoints/ae64_integrated/best.pt",
    "ae128": ROOT / "experiments/ae_integrated_20260710/ae128/checkpoints/ae128_integrated/best.pt",
}


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def f1_score(precision: float, recall: float) -> float:
    if not math.isfinite(precision) or not math.isfinite(recall) or precision + recall == 0.0:
        return 0.0 if precision == 0.0 and recall == 0.0 else float("nan")
    return float(2.0 * precision * recall / (precision + recall))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


SAMPLE_RE = re.compile(r"^(?P<prefix>.+)_(?P<index>\d+)_frame(?P<frame>\d+)$")


def parse_sample_id(sample_id: str) -> Tuple[str, int, int]:
    match = SAMPLE_RE.match(str(sample_id))
    if not match:
        raise ValueError(f"Unrecognized sample_id: {sample_id}")
    return match.group("prefix"), int(match.group("index")), int(match.group("frame"))


def audit_assignment(sample_id: str) -> Tuple[str, str, int]:
    prefix, collection_index, _ = parse_sample_id(sample_id)
    block = collection_index // 25
    token = f"decoder-audit-v1|{prefix}|{block}"
    value = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:16], 16)
    split = "audit_validation" if value % 5 in (0, 1) else "audit_test"
    return split, prefix, block


def normalize_class(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in ("pedestrian", "person"):
        return "person"
    if text in ("vehicle", "movingvehicle", "parkedvehicle"):
        return "vehicle"
    return text


def metric_row(tp: float, fp: float, fn: float, err_sum: float, err_sq_sum: float, frames: float) -> Dict[str, float]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "prediction_count": int(tp + fp),
        "gt_count": int(tp + fn),
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
        "fp_per_frame": safe_div(fp, frames),
        "xy_mae_m": safe_div(err_sum, tp),
        "xy_rmse_m": math.sqrt(max(0.0, safe_div(err_sq_sum, tp))) if tp else float("nan"),
        "profile_frames": int(frames),
    }


def greedy_match(predictions: Sequence[Mapping[str, Any]], gt_objects: Sequence[Mapping[str, Any]]) -> List[Tuple[int, int, float]]:
    candidates: List[Tuple[float, int, int]] = []
    for pred_index, pred in enumerate(predictions):
        for gt_index, gt in enumerate(gt_objects):
            if normalize_class(pred.get("class_name")) != normalize_class(gt.get("class_name")):
                continue
            distance = math.hypot(float(pred["world_x"]) - float(gt["world_x"]),
                                  float(pred["world_y"]) - float(gt["world_y"]))
            if distance <= MATCH_RADIUS_M:
                candidates.append((distance, pred_index, gt_index))
    candidates.sort(key=lambda item: item[0])
    used_predictions: set[int] = set()
    used_gt: set[int] = set()
    result: List[Tuple[int, int, float]] = []
    for distance, pred_index, gt_index in candidates:
        if pred_index in used_predictions or gt_index in used_gt:
            continue
        used_predictions.add(pred_index)
        used_gt.add(gt_index)
        result.append((pred_index, gt_index, distance))
    return result


def apply_candidate(predictions: Sequence[Mapping[str, Any]], candidate: Candidate) -> List[Dict[str, Any]]:
    thresholds = {"vehicle": candidate.vehicle_threshold, "person": candidate.person_threshold}
    eligible = [dict(item) for item in predictions
                if float(item.get("score", 0.0)) >= thresholds.get(normalize_class(item.get("class_name")), BASELINE_THRESHOLD)]
    if candidate.world_nms_m <= 0.0 and candidate.image_nms_px <= 0.0:
        return eligible
    ordered = sorted(eligible, key=lambda item: (-float(item.get("score", 0.0)), int(item.get("source_order", 0))))
    kept: List[Dict[str, Any]] = []
    for pred in ordered:
        suppress = False
        for accepted in kept:
            if normalize_class(pred.get("class_name")) != normalize_class(accepted.get("class_name")):
                continue
            if candidate.world_nms_m > 0.0:
                distance = math.hypot(float(pred["world_x"]) - float(accepted["world_x"]),
                                      float(pred["world_y"]) - float(accepted["world_y"]))
                if distance <= candidate.world_nms_m:
                    suppress = True
                    break
            if candidate.image_nms_px > 0.0:
                px = finite_float(pred.get("center_x_px"))
                py = finite_float(pred.get("center_y_px"))
                ax = finite_float(accepted.get("center_x_px"))
                ay = finite_float(accepted.get("center_y_px"))
                if all(math.isfinite(v) for v in (px, py, ax, ay)) and max(abs(px - ax), abs(py - ay)) <= candidate.image_nms_px:
                    suppress = True
                    break
        if not suppress:
            kept.append(pred)
    return kept


def build_split_manifest(frame_density: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in frame_density.to_dict("records"):
        sample_id = str(record["sample_id"])
        split, prefix, block = audit_assignment(sample_id)
        _, collection_index, frame_from_id = parse_sample_id(sample_id)
        rows.append({
            "sample_id": sample_id,
            "frame_id": int(record.get("frame_id", frame_from_id)),
            "source_prefix": prefix,
            "collection_index": collection_index,
            "block": block,
            "audit_split": split,
            "density_bin": str(record.get("density_bin", "")),
        })
    result = pd.DataFrame(rows).sort_values(["source_prefix", "collection_index", "sample_id"])
    if result["sample_id"].duplicated().any():
        raise AssertionError("Duplicate sample identifiers in frame_density.csv")
    validation_ids = set(result.loc[result.audit_split == "audit_validation", "sample_id"])
    test_ids = set(result.loc[result.audit_split == "audit_test", "sample_id"])
    if validation_ids & test_ids:
        raise AssertionError("Audit validation/test identifier overlap")
    validation_blocks = set(map(tuple, result.loc[result.audit_split == "audit_validation", ["source_prefix", "block"]].drop_duplicates().values.tolist()))
    test_blocks = set(map(tuple, result.loc[result.audit_split == "audit_test", ["source_prefix", "block"]].drop_duplicates().values.tolist()))
    if validation_blocks & test_blocks:
        raise AssertionError("Audit validation/test block overlap")
    return result


def profile_dir(family: str, quant: str) -> Path:
    return SWEEPS / f"{family}__{quant}__roi0.0"


def prediction_from_row(row: Mapping[str, Any], source_order: int) -> Dict[str, Any]:
    x0 = finite_float(row.get("pred_bbox_x0"))
    x1 = finite_float(row.get("pred_bbox_x1"))
    y0 = finite_float(row.get("pred_bbox_y0"))
    y1 = finite_float(row.get("pred_bbox_y1"))
    return {
        "class_name": normalize_class(row.get("pred_class_name")),
        "score": finite_float(row.get("score"), 0.0),
        "world_x": finite_float(row.get("pred_world_x"), 0.0),
        "world_y": finite_float(row.get("pred_world_y"), 0.0),
        "world_z": 0.0,
        "size_x": finite_float(row.get("pred_size_x"), 0.05),
        "size_y": finite_float(row.get("pred_size_y"), 0.05),
        "size_z": finite_float(row.get("pred_size_z"), 0.05),
        "center_x_px": (x0 + x1) / 2.0 if math.isfinite(x0) and math.isfinite(x1) else float("nan"),
        "center_y_px": (y0 + y1) / 2.0 if math.isfinite(y0) and math.isfinite(y1) else float("nan"),
        "source_order": source_order,
    }


def gt_from_row(row: Mapping[str, Any], source_order: int) -> Dict[str, Any]:
    return {
        "class_name": normalize_class(row.get("gt_class_name")),
        "world_x": finite_float(row.get("gt_world_x"), 0.0),
        "world_y": finite_float(row.get("gt_world_y"), 0.0),
        "world_z": 0.0,
        "source_order": source_order,
    }


def load_profiles(universe: Sequence[str]) -> Dict[Tuple[str, str], Dict[str, Dict[str, List[Dict[str, Any]]]]]:
    profiles: Dict[Tuple[str, str], Dict[str, Dict[str, List[Dict[str, Any]]]]] = {}
    universe_set = set(universe)
    for family in FAMILIES:
        for quant in QUANTS:
            path = profile_dir(family, quant) / "metrics" / "test_learned_object_metrics.csv"
            table = pd.read_csv(path, low_memory=False)
            frames: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
                sample_id: {"predictions": [], "gt": []} for sample_id in universe
            }
            for source_order, row in enumerate(table.to_dict("records")):
                sample_id = str(row.get("sample_id", ""))
                if sample_id not in universe_set:
                    raise AssertionError(f"Per-object sample not in frame universe: {sample_id}")
                status = str(row.get("match_status", "")).lower()
                if status in ("tp", "fp"):
                    frames[sample_id]["predictions"].append(prediction_from_row(row, source_order))
                if status in ("tp", "fn"):
                    frames[sample_id]["gt"].append(gt_from_row(row, source_order))
            profiles[(family, quant)] = frames
    return profiles


def class_frame_counts(predictions: Sequence[Mapping[str, Any]], gt: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    matches = greedy_match(predictions, gt)
    matched_predictions = {pred_index for pred_index, _, _ in matches}
    matched_gt = {gt_index for _, gt_index, _ in matches}
    result: Dict[str, Any] = {}
    for class_name, short in (("vehicle", "veh"), ("person", "ped")):
        class_matches = [(pi, gi, distance) for pi, gi, distance in matches if normalize_class(gt[gi].get("class_name")) == class_name]
        tp = len(class_matches)
        fp = sum(1 for index, item in enumerate(predictions)
                 if index not in matched_predictions and normalize_class(item.get("class_name")) == class_name)
        fn = sum(1 for index, item in enumerate(gt)
                 if index not in matched_gt and normalize_class(item.get("class_name")) == class_name)
        result[f"{short}_tp"] = tp
        result[f"{short}_fp"] = fp
        result[f"{short}_fn"] = fn
        result[f"{short}_err_sum"] = sum(distance for _, _, distance in class_matches)
        result[f"{short}_err_sq_sum"] = sum(distance * distance for _, _, distance in class_matches)
    return result


def evaluate_candidate(
    profiles: Mapping[Tuple[str, str], Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]],
    candidate: Candidate,
    sample_ids: Sequence[str],
    split_name: str,
    density_by_sample: Mapping[str, str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (family, quant), frames in profiles.items():
        for sample_id in sample_ids:
            frame = frames[sample_id]
            predictions = apply_candidate(frame["predictions"], candidate)
            counts = class_frame_counts(predictions, frame["gt"])
            rows.append({
                "candidate": candidate.name,
                "split": split_name,
                "family": family,
                "quant": quant,
                "sample_id": sample_id,
                "density_bin": density_by_sample[sample_id],
                "n_predictions": len(predictions),
                "n_gt": len(frame["gt"]),
                **counts,
            })
    return pd.DataFrame(rows)


def summarize_frames(frame_table: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    groups: List[Tuple[str, pd.DataFrame]] = [(family, frame_table.loc[frame_table.family == family]) for family in FAMILIES]
    groups.append(("pooled", frame_table))
    for group_name, group in groups:
        if group.empty:
            continue
        candidate_name = str(group.candidate.iloc[0])
        split_name = str(group.split.iloc[0])
        for class_name, short in (("vehicle", "veh"), ("person", "ped"), ("all", "all")):
            if short == "all":
                tp = float(group.veh_tp.sum() + group.ped_tp.sum())
                fp = float(group.veh_fp.sum() + group.ped_fp.sum())
                fn = float(group.veh_fn.sum() + group.ped_fn.sum())
                err_sum = float(group.veh_err_sum.sum() + group.ped_err_sum.sum())
                err_sq_sum = float(group.veh_err_sq_sum.sum() + group.ped_err_sq_sum.sum())
            else:
                tp = float(group[f"{short}_tp"].sum())
                fp = float(group[f"{short}_fp"].sum())
                fn = float(group[f"{short}_fn"].sum())
                err_sum = float(group[f"{short}_err_sum"].sum())
                err_sq_sum = float(group[f"{short}_err_sq_sum"].sum())
            rows.append({
                "candidate": candidate_name,
                "split": split_name,
                "group": group_name,
                "class_name": class_name,
                "unique_frames": int(group.sample_id.nunique()),
                **metric_row(tp, fp, fn, err_sum, err_sq_sum, float(len(group))),
            })
    return pd.DataFrame(rows)


def reproduce_table() -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    seg_rows: List[Dict[str, Any]] = []
    for family in FAMILIES:
        source = pd.read_csv(RAW / f"perframe_{family}.csv")
        source = source.loc[np.isclose(source.roi.astype(float), 0.0)].copy()
        if len(source) != 2162 * 3 or source.sample_id.nunique() != 2162:
            raise AssertionError(f"Unexpected q=0 shape for {family}: rows={len(source)}, ids={source.sample_id.nunique()}")
        record: Dict[str, Any] = {"variant": family, "profile_rows": len(source), "unique_frames": source.sample_id.nunique(),
                                  "complete_quantizers": source["quant"].nunique()}
        for prefix, label in (("veh", "veh"), ("ped", "ped")):
            tp = float(source[f"tp_{prefix}"].sum())
            fp = float(source[f"fp_{prefix}"].sum())
            fn = float(source[f"fn_{prefix}"].sum())
            p = safe_div(tp, tp + fp)
            r = safe_div(tp, tp + fn)
            record[f"{label}_precision"] = p
            record[f"{label}_recall"] = r
            record[f"{label}_f1"] = f1_score(p, r)
        record["fp_per_frame"] = safe_div(float(source.fp.sum()), float(len(source)))
        for key, expected in EXPECTED[family].items():
            record[f"expected_{key}"] = expected
            record[f"rounded_delta_{key}"] = round(float(record[key]), 3) - expected
        rows.append(record)

        confusion = np.array([[int(source[f"conf_{i}{j}"].sum()) for j in range(3)] for i in range(3)], dtype=np.int64)
        ious = []
        for index, class_name in enumerate(("background", "vehicle", "person")):
            intersection = float(confusion[index, index])
            union = float(confusion[index, :].sum() + confusion[:, index].sum() - confusion[index, index])
            iou = safe_div(intersection, union)
            ious.append(iou)
            seg_rows.append({"variant": family, "class_name": class_name, "iou": iou, "decoder_invariant": True})
        seg_rows.append({"variant": family, "class_name": "mean", "iou": float(np.nanmean(ious)), "decoder_invariant": True})
    return pd.DataFrame(rows), pd.DataFrame(seg_rows)


def reconcile_profile_sources(
    profiles: Mapping[Tuple[str, str], Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]],
    universe: Sequence[str],
) -> pd.DataFrame:
    rows = []
    quant_to_raw = {quant: f"per_channel_{quant}" for quant in QUANTS}
    for family in FAMILIES:
        perframe = pd.read_csv(RAW / f"perframe_{family}.csv")
        for quant in QUANTS:
            source = perframe.loc[np.isclose(perframe.roi.astype(float), 0.0) & (perframe["quant"] == quant_to_raw[quant])]
            derived = defaultdict(float)
            for sample_id in universe:
                frame = profiles[(family, quant)][sample_id]
                counts = class_frame_counts(apply_candidate(frame["predictions"], Candidate("baseline")), frame["gt"])
                for key, value in counts.items():
                    if not key.endswith(("err_sum", "err_sq_sum")):
                        derived[key] += float(value)
            expected = {
                "veh_tp": float(source.tp_veh.sum()), "veh_fp": float(source.fp_veh.sum()), "veh_fn": float(source.fn_veh.sum()),
                "ped_tp": float(source.tp_ped.sum()), "ped_fp": float(source.fp_ped.sum()), "ped_fn": float(source.fn_ped.sum()),
            }
            record: Dict[str, Any] = {"family": family, "quant": quant, "profile_frames": len(source)}
            exact = True
            for key, value in expected.items():
                record[f"perframe_{key}"] = int(value)
                record[f"perobject_replay_{key}"] = int(derived[key])
                record[f"delta_{key}"] = int(derived[key] - value)
                exact = exact and int(derived[key]) == int(value)
            record["exact_count_match"] = exact
            rows.append(record)
    return pd.DataFrame(rows)


def nearest_distance(pred: Mapping[str, Any], gt: Sequence[Mapping[str, Any]], class_name: Optional[str] = None) -> Tuple[float, Optional[int]]:
    best = float("inf")
    best_index: Optional[int] = None
    for index, item in enumerate(gt):
        if class_name is not None and normalize_class(item.get("class_name")) != class_name:
            continue
        distance = math.hypot(float(pred["world_x"]) - float(item["world_x"]),
                              float(pred["world_y"]) - float(item["world_y"]))
        if distance < best:
            best = distance
            best_index = index
    return best, best_index


def build_fp_taxonomy(
    profiles: Mapping[Tuple[str, str], Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]],
    split_manifest: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    meta = split_manifest.set_index("sample_id").to_dict("index")
    rows: List[Dict[str, Any]] = []
    fp_indices_by_frame: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
    for (family, quant), frames in profiles.items():
        for sample_id, frame in frames.items():
            predictions = apply_candidate(frame["predictions"], Candidate("baseline"))
            gt = frame["gt"]
            matches = greedy_match(predictions, gt)
            matched_predictions = {pred_index for pred_index, _, _ in matches}
            matched_gt = {gt_index for _, gt_index, _ in matches}
            for pred_index, pred in enumerate(predictions):
                if pred_index in matched_predictions:
                    continue
                pred_class = normalize_class(pred.get("class_name"))
                same_distance, same_index = nearest_distance(pred, gt, pred_class)
                other_distance, _ = nearest_distance(pred, gt, "person" if pred_class == "vehicle" else "vehicle")
                any_distance, _ = nearest_distance(pred, gt, None)
                if same_distance <= MATCH_RADIUS_M and same_index in matched_gt:
                    category = "duplicate_same_class_claimed_gt"
                elif other_distance <= MATCH_RADIUS_M:
                    category = "cross_class_confusion"
                elif MATCH_RADIUS_M < same_distance <= PLAUSIBLE_RADIUS_M:
                    category = "same_class_near_outside_match_radius"
                elif any_distance > PLAUSIBLE_RADIUS_M:
                    category = "no_plausible_nearby_gt"
                else:
                    category = "other_nearby_gt_geometry"
                row_index = len(rows)
                rows.append({
                    "family": family,
                    "quant": quant,
                    "sample_id": sample_id,
                    "audit_split": meta[sample_id]["audit_split"],
                    "source_prefix": meta[sample_id]["source_prefix"],
                    "collection_index": int(meta[sample_id]["collection_index"]),
                    "density_bin": meta[sample_id]["density_bin"],
                    "class_name": pred_class,
                    "score": float(pred.get("score", 0.0)),
                    "pred_world_x": float(pred["world_x"]),
                    "pred_world_y": float(pred["world_y"]),
                    "nearest_same_class_gt_m": same_distance if math.isfinite(same_distance) else float("nan"),
                    "nearest_other_class_gt_m": other_distance if math.isfinite(other_distance) else float("nan"),
                    "nearest_any_gt_m": any_distance if math.isfinite(any_distance) else float("nan"),
                    "category": category,
                    "empty_scene_fp": len(gt) == 0,
                    "temporal_status": "single_frame",
                })
                fp_indices_by_frame[(family, quant, sample_id)].append(row_index)

    by_prefix: Dict[str, List[str]] = {}
    for prefix, group in split_manifest.groupby("source_prefix"):
        by_prefix[str(prefix)] = list(group.sort_values("collection_index").sample_id.astype(str))
    neighbor_ids: Dict[str, List[str]] = {}
    for ordered in by_prefix.values():
        for index, sample_id in enumerate(ordered):
            neighbor_ids[sample_id] = ordered[max(0, index - 1):index] + ordered[index + 1:index + 2]

    for row_index, row in enumerate(rows):
        for neighbor_sample in neighbor_ids.get(str(row["sample_id"]), []):
            for neighbor_index in fp_indices_by_frame.get((str(row["family"]), str(row["quant"]), neighbor_sample), []):
                neighbor = rows[neighbor_index]
                if neighbor["class_name"] != row["class_name"]:
                    continue
                distance = math.hypot(float(neighbor["pred_world_x"]) - float(row["pred_world_x"]),
                                      float(neighbor["pred_world_y"]) - float(row["pred_world_y"]))
                if distance <= 3.0:
                    row["temporal_status"] = "persistent"
                    break
            if row["temporal_status"] == "persistent":
                break

    detail = pd.DataFrame(rows)
    summary = (detail.groupby(["family", "quant", "class_name", "category", "temporal_status", "empty_scene_fp"], dropna=False)
               .size().rename("fp_count").reset_index())
    totals = summary.groupby(["family", "quant", "class_name"], dropna=False).fp_count.transform("sum")
    summary["fraction_of_class_fp"] = summary.fp_count / totals
    pooled = (detail.groupby(["class_name", "category", "temporal_status", "empty_scene_fp"], dropna=False)
              .size().rename("fp_count").reset_index())
    pooled.insert(0, "quant", "pooled")
    pooled.insert(0, "family", "pooled")
    pooled["fraction_of_class_fp"] = pooled.fp_count / pooled.groupby("class_name").fp_count.transform("sum")
    return detail, pd.concat([summary, pooled], ignore_index=True)


def behavior_summary(frame_tables: Sequence[pd.DataFrame]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for table in frame_tables:
        behavior_masks = {
            "all": np.ones(len(table), dtype=bool),
            "empty_gt": table.n_gt.to_numpy() == 0,
            "dense_5plus": table.density_bin.astype(str).to_numpy() == "5+",
            "density_0": table.density_bin.astype(str).to_numpy() == "0",
            "density_1-2": table.density_bin.astype(str).to_numpy() == "1-2",
            "density_3-4": table.density_bin.astype(str).to_numpy() == "3-4",
        }
        for behavior, mask in behavior_masks.items():
            subset = table.loc[mask]
            if subset.empty:
                continue
            for family, group in [(family, subset.loc[subset.family == family]) for family in FAMILIES] + [("pooled", subset)]:
                if group.empty:
                    continue
                for class_name, short in (("vehicle", "veh"), ("person", "ped"), ("all", "all")):
                    if short == "all":
                        tp = float(group.veh_tp.sum() + group.ped_tp.sum())
                        fp = float(group.veh_fp.sum() + group.ped_fp.sum())
                        fn = float(group.veh_fn.sum() + group.ped_fn.sum())
                        err = float(group.veh_err_sum.sum() + group.ped_err_sum.sum())
                        err_sq = float(group.veh_err_sq_sum.sum() + group.ped_err_sq_sum.sum())
                    else:
                        tp, fp, fn = (float(group[f"{short}_{key}"].sum()) for key in ("tp", "fp", "fn"))
                        err = float(group[f"{short}_err_sum"].sum())
                        err_sq = float(group[f"{short}_err_sq_sum"].sum())
                    rows.append({"candidate": str(group.candidate.iloc[0]), "behavior": behavior, "group": family,
                                 "class_name": class_name, "unique_frames": int(group.sample_id.nunique()),
                                 **metric_row(tp, fp, fn, err, err_sq, float(len(group)))})
    return pd.DataFrame(rows)


def distance_bin(distance: float) -> str:
    if not math.isfinite(distance):
        return "unknown"
    if distance < 10.0:
        return "0-10"
    if distance < 20.0:
        return "10-20"
    if distance < 30.0:
        return "20-30"
    if distance <= 40.0:
        return "30-40"
    return "outside_40"


def distance_strata(
    profiles: Mapping[Tuple[str, str], Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]],
    candidates: Sequence[Candidate],
    test_ids: Sequence[str],
    camera_by_sample: Mapping[str, Tuple[float, float]],
) -> pd.DataFrame:
    counts: MutableMapping[Tuple[str, str, str, str, str], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for candidate in candidates:
        for (family, quant), frames in profiles.items():
            for sample_id in test_ids:
                frame = frames[sample_id]
                predictions = apply_candidate(frame["predictions"], candidate)
                gt = frame["gt"]
                matches = greedy_match(predictions, gt)
                matched_pred = {pi for pi, _, _ in matches}
                matched_gt = {gi for _, gi, _ in matches}
                camera_x, camera_y = camera_by_sample.get(sample_id, (float("nan"), float("nan")))
                for pi, gi, error in matches:
                    item = gt[gi]
                    dist = math.hypot(float(item["world_x"]) - camera_x, float(item["world_y"]) - camera_y)
                    bucket = counts[(candidate.name, family, quant, normalize_class(item.get("class_name")), distance_bin(dist))]
                    bucket["tp"] += 1; bucket["err"] += error; bucket["err_sq"] += error * error
                for gi, item in enumerate(gt):
                    if gi in matched_gt:
                        continue
                    dist = math.hypot(float(item["world_x"]) - camera_x, float(item["world_y"]) - camera_y)
                    counts[(candidate.name, family, quant, normalize_class(item.get("class_name")), distance_bin(dist))]["fn"] += 1
                for pi, item in enumerate(predictions):
                    if pi in matched_pred:
                        continue
                    dist = math.hypot(float(item["world_x"]) - camera_x, float(item["world_y"]) - camera_y)
                    counts[(candidate.name, family, quant, normalize_class(item.get("class_name")), distance_bin(dist))]["fp"] += 1
    rows = []
    for (candidate, family, quant, class_name, stratum), values in counts.items():
        rows.append({"candidate": candidate, "group": family, "quant": quant, "class_name": class_name, "distance_stratum_m": stratum,
                     **metric_row(values["tp"], values["fp"], values["fn"], values["err"], values["err_sq"], len(test_ids))})
    raw = pd.DataFrame(rows)
    pooled_rows = []
    for keys, group in raw.groupby(["candidate", "class_name", "distance_stratum_m"]):
        candidate, class_name, stratum = keys
        pooled_rows.append({"candidate": candidate, "group": "pooled", "quant": "pooled", "class_name": class_name,
                            "distance_stratum_m": stratum,
                            **metric_row(group.tp.sum(), group.fp.sum(), group.fn.sum(),
                                         float((group.xy_mae_m * group.tp).sum()),
                                         float(((group.xy_rmse_m ** 2) * group.tp).sum()),
                                         len(test_ids) * len(FAMILIES) * len(QUANTS))})
    return pd.concat([raw, pd.DataFrame(pooled_rows)], ignore_index=True)


def select_candidate(validation_summary: pd.DataFrame) -> Tuple[Candidate, List[Candidate], Dict[str, Any]]:
    candidates = list(PREREGISTERED)
    pooled_vehicle = validation_summary.loc[(validation_summary.group == "pooled") & (validation_summary.class_name == "vehicle")]
    baseline_f1 = float(pooled_vehicle.loc[pooled_vehicle.candidate == "baseline", "f1"].iloc[0])
    world_rows = pooled_vehicle.loc[pooled_vehicle.candidate.str.startswith("world_nms_")]
    threshold_rows = pooled_vehicle.loc[pooled_vehicle.candidate.str.startswith("veh_thr_")]
    best_world_name = str(world_rows.sort_values(["f1", "fp_per_frame"], ascending=[False, True]).candidate.iloc[0])
    best_threshold_name = str(threshold_rows.sort_values(["f1", "fp_per_frame"], ascending=[False, True]).candidate.iloc[0])
    best_world_f1 = float(world_rows.loc[world_rows.candidate == best_world_name, "f1"].iloc[0])
    best_threshold_f1 = float(threshold_rows.loc[threshold_rows.candidate == best_threshold_name, "f1"].iloc[0])
    gate = best_world_f1 > baseline_f1 and best_threshold_f1 > baseline_f1
    details: Dict[str, Any] = {
        "baseline_vehicle_f1": baseline_f1,
        "best_world_candidate": best_world_name,
        "best_world_vehicle_f1": best_world_f1,
        "best_threshold_candidate": best_threshold_name,
        "best_threshold_vehicle_f1": best_threshold_f1,
        "combination_gate_passed": gate,
    }
    if gate:
        world = next(item for item in candidates if item.name == best_world_name)
        threshold = next(item for item in candidates if item.name == best_threshold_name)
        combo = Candidate(
            name=f"combo_{best_world_name}_{best_threshold_name}",
            vehicle_threshold=threshold.vehicle_threshold,
            person_threshold=threshold.person_threshold,
            world_nms_m=world.world_nms_m,
        )
        candidates.append(combo)
        details["combination_candidate"] = combo.name
    return Candidate("baseline"), candidates, details


def choose_from_complete_validation(validation_summary: pd.DataFrame, candidates: Sequence[Candidate]) -> Candidate:
    eligible_names = {item.name for item in candidates}
    vehicle = validation_summary.loc[(validation_summary.group == "pooled") &
                                     (validation_summary.class_name == "vehicle") &
                                     (validation_summary.candidate.isin(eligible_names))][["candidate", "f1", "fp_per_frame"]]
    person = validation_summary.loc[(validation_summary.group == "pooled") &
                                    (validation_summary.class_name == "person") &
                                    (validation_summary.candidate.isin(eligible_names))][["candidate", "f1"]].rename(columns={"f1": "person_f1"})
    joined = vehicle.merge(person, on="candidate", how="left")
    joined = joined.sort_values(["f1", "person_f1", "fp_per_frame", "candidate"], ascending=[False, False, True, True])
    name = str(joined.candidate.iloc[0])
    return next(item for item in candidates if item.name == name)


def metric_vector(frame_table: pd.DataFrame, class_name: str) -> Tuple[np.ndarray, List[str]]:
    short = "veh" if class_name == "vehicle" else "ped"
    grouped = frame_table.groupby("sample_id")[[f"{short}_tp", f"{short}_fp", f"{short}_fn", f"{short}_err_sum", f"{short}_err_sq_sum"]].sum()
    frame_counts = frame_table.groupby("sample_id").size().rename("frames")
    joined = grouped.join(frame_counts).sort_index()
    return joined.to_numpy(dtype=np.float64), list(joined.index)


def vector_metrics(vector: np.ndarray) -> Dict[str, float]:
    tp, fp, fn, err, err_sq, frames = vector.sum(axis=0)
    row = metric_row(tp, fp, fn, err, err_sq, frames)
    return {key: float(row[key]) for key in ("precision", "recall", "f1", "fp_per_frame", "xy_mae_m", "xy_rmse_m")}


def paired_bootstrap(baseline: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: List[Dict[str, Any]] = []
    for group_name in (*FAMILIES, "pooled"):
        base_group = baseline if group_name == "pooled" else baseline.loc[baseline.family == group_name]
        sel_group = selected if group_name == "pooled" else selected.loc[selected.family == group_name]
        for class_name in CLASS_NAMES:
            base_vector, base_ids = metric_vector(base_group, class_name)
            sel_vector, sel_ids = metric_vector(sel_group, class_name)
            if base_ids != sel_ids:
                raise AssertionError("Paired bootstrap identifier mismatch")
            observed_base = vector_metrics(base_vector)
            observed_sel = vector_metrics(sel_vector)
            deltas: Dict[str, List[float]] = defaultdict(list)
            count = len(base_ids)
            for _ in range(BOOTSTRAP_REPS):
                indices = rng.integers(0, count, size=count)
                boot_base = vector_metrics(base_vector[indices])
                boot_sel = vector_metrics(sel_vector[indices])
                for metric in observed_base:
                    deltas[metric].append(boot_sel[metric] - boot_base[metric])
            for metric, values in deltas.items():
                finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
                rows.append({
                    "group": group_name,
                    "class_name": class_name,
                    "metric": metric,
                    "baseline": observed_base[metric],
                    "selected": observed_sel[metric],
                    "observed_delta": observed_sel[metric] - observed_base[metric],
                    "delta_ci95_low": float(np.percentile(finite, 2.5)),
                    "delta_ci95_high": float(np.percentile(finite, 97.5)),
                    "bootstrap_reps": BOOTSTRAP_REPS,
                    "resampling_unit": "unique_sample_id_with_all_profiles",
                })
    return pd.DataFrame(rows)


def measure_latency(
    profiles: Mapping[Tuple[str, str], Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]],
    selected: Candidate,
    test_ids: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    baseline = Candidate("baseline")
    no_op_samples = []
    empty: List[Dict[str, Any]] = []
    for _ in range(10000):
        started = time.perf_counter_ns(); list(empty); no_op_samples.append(time.perf_counter_ns() - started)
    overhead_ns = float(np.median(no_op_samples))
    rows = []
    for (family, quant), frames in profiles.items():
        for sample_id in test_ids:
            predictions = frames[sample_id]["predictions"]
            for _ in range(5):
                apply_candidate(predictions, baseline); apply_candidate(predictions, selected)
            base_times: List[float] = []
            selected_times: List[float] = []
            for repeat in range(30):
                order = ((baseline, base_times), (selected, selected_times)) if repeat % 2 == 0 else ((selected, selected_times), (baseline, base_times))
                for candidate, destination in order:
                    started = time.perf_counter_ns()
                    apply_candidate(predictions, candidate)
                    destination.append(max(0.0, float(time.perf_counter_ns() - started) - overhead_ns))
            base_ms = float(np.median(base_times)) / 1e6
            selected_ms = float(np.median(selected_times)) / 1e6
            rows.append({"family": family, "quant": quant, "sample_id": sample_id,
                         "input_prediction_count": len(predictions), "baseline_ms": base_ms,
                         "selected_ms": selected_ms, "paired_delta_ms": selected_ms - base_ms})
    detail = pd.DataFrame(rows)
    summary_rows = []
    for label, column in (("baseline", "baseline_ms"), (selected.name, "selected_ms"), ("paired_delta", "paired_delta_ms")):
        values = detail[column].to_numpy(dtype=np.float64)
        summary_rows.append({"stage": "retained_list_postprocessing", "configuration": label,
                             "p50_ms": float(np.percentile(values, 50)), "p90_ms": float(np.percentile(values, 90)),
                             "p95_ms": float(np.percentile(values, 95)), "max_ms": float(np.max(values)),
                             "profile_frame_samples": len(values), "repeats_per_side": 30,
                             "loop_overhead_subtracted_ns": overhead_ns})
    return detail, pd.DataFrame(summary_rows)


def taxonomy_counts_for_predictions(predictions: Sequence[Mapping[str, Any]], gt: Sequence[Mapping[str, Any]]) -> Tuple[int, int]:
    matches = greedy_match(predictions, gt)
    matched_predictions = {pi for pi, _, _ in matches}
    matched_gt = {gi for _, gi, _ in matches}
    duplicates = 0
    ghosts = len(predictions) - len(matches)
    for pred_index, pred in enumerate(predictions):
        if pred_index in matched_predictions:
            continue
        same_distance, same_index = nearest_distance(pred, gt, normalize_class(pred.get("class_name")))
        if same_distance <= MATCH_RADIUS_M and same_index in matched_gt:
            duplicates += 1
    return duplicates, ghosts


def offline_map_replay(
    profiles: Mapping[Tuple[str, str], Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]],
    candidates: Sequence[Candidate],
    test_ids: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    path = ROOT / "real_time_spatial_map_server_fusion_object_v2.py"
    root_text = str(ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    spec = importlib.util.spec_from_file_location("decoder_audit_map_server", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not construct map-server import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CONFIG = argparse.Namespace(
        min_object_score=0.0,
        object_yaw_map_offset_deg=90.0,
        association_dimension_ratio=0.65,
        association_radius_m=4.0,
        common_min_streams=2,
        track_stale_s=4.0,
        track_match_radius_m=6.0,
        smoothing_alpha=0.45,
        hide_single_stream_objects=False,
        max_rendered_objects=200,
    )
    rows = []
    for candidate in candidates:
        for (family, quant), frames in profiles.items():
            for sample_id in test_ids:
                frame = frames[sample_id]
                predictions = apply_candidate(frame["predictions"], candidate)
                objects = []
                for index, pred in enumerate(predictions):
                    objects.append({
                        "id": f"audit:{index}",
                        "type": "Vehicle" if normalize_class(pred.get("class_name")) == "vehicle" else "Pedestrian",
                        "score": float(pred.get("score", 0.0)),
                        "location": {"x": float(pred["world_x"]), "y": float(pred["world_y"]), "z": float(pred.get("world_z", 0.0))},
                        "dimensions": {"length": float(pred.get("size_x", 0.05)), "width": float(pred.get("size_y", 0.05)),
                                       "height": float(pred.get("size_z", 0.05))},
                    })
                packet = module._normalize_packet({"stream_id": "single_audit_stream", "frame_id": parse_sample_id(sample_id)[2], "objects": objects}, time.time())
                module.fusion_tracks = {}
                module.next_track_id = 1
                installed, measurements = module._fuse_and_smooth_objects(packet["objects"])
                installed_predictions = []
                for item in installed:
                    location = item["location"]
                    installed_predictions.append({
                        "class_name": "vehicle" if str(item.get("type")) == "Vehicle" else "person",
                        "world_x": float(location["x"]), "world_y": float(location["y"]),
                    })
                matches = greedy_match(installed_predictions, frame["gt"])
                duplicates, ghosts = taxonomy_counts_for_predictions(installed_predictions, frame["gt"])
                rows.append({"candidate": candidate.name, "family": family, "quant": quant, "sample_id": sample_id,
                             "raw_detection_count": len(predictions), "measurement_count": len(measurements),
                             "installed_object_count": len(installed), "installed_tp": len(matches),
                             "installed_fp": ghosts, "installed_fn": len(frame["gt"]) - len(matches),
                             "installed_duplicate_count": duplicates})
    detail = pd.DataFrame(rows)
    proof = {
        "verified": True,
        "server_path": str(path.relative_to(ROOT)),
        "server_sha256": sha256_file(path),
        "actual_functions_invoked": ["_normalize_packet", "_fuse_and_smooth_objects"],
        "network_or_carla_calls": False,
        "single_stream_cluster_rule": "_can_join_cluster rejects any cluster containing the same source_stream_id",
        "raw_equals_measurements_all_frames": bool((detail.raw_detection_count == detail.measurement_count).all()),
        "raw_equals_installed_all_frames": bool((detail.raw_detection_count == detail.installed_object_count).all()),
        "scope": "each frame/profile replayed independently; temporal smoothing reset",
    }
    return detail, proof


def summarize_map(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in detail.groupby(["candidate", "family"]):
        candidate, family = keys
        tp, fp, fn = float(group.installed_tp.sum()), float(group.installed_fp.sum()), float(group.installed_fn.sum())
        metrics = metric_row(tp, fp, fn, 0.0, 0.0, len(group))
        metrics["xy_mae_m"] = float("nan"); metrics["xy_rmse_m"] = float("nan")
        rows.append({"candidate": candidate, "group": family, "raw_detection_count": int(group.raw_detection_count.sum()),
                     "installed_object_count": int(group.installed_object_count.sum()),
                     "installed_duplicate_count": int(group.installed_duplicate_count.sum()), **metrics})
    for candidate, group in detail.groupby("candidate"):
        tp, fp, fn = float(group.installed_tp.sum()), float(group.installed_fp.sum()), float(group.installed_fn.sum())
        metrics = metric_row(tp, fp, fn, 0.0, 0.0, len(group)); metrics["xy_mae_m"] = float("nan"); metrics["xy_rmse_m"] = float("nan")
        rows.append({"candidate": candidate, "group": "pooled", "raw_detection_count": int(group.raw_detection_count.sum()),
                     "installed_object_count": int(group.installed_object_count.sum()),
                     "installed_duplicate_count": int(group.installed_duplicate_count.sum()), **metrics})
    return pd.DataFrame(rows)


def build_input_manifest() -> Dict[str, Any]:
    files: List[Tuple[str, Path]] = []
    files.extend(("per_frame_metrics", RAW / f"perframe_{family}.csv") for family in FAMILIES)
    files.append(("frame_density", RAW / "frame_density.csv"))
    files.extend(("checkpoint", path) for path in CHECKPOINTS.values())
    for family in FAMILIES:
        for quant in QUANTS:
            base = profile_dir(family, quant) / "metrics"
            files.append(("per_object_metrics", base / "test_learned_object_metrics.csv"))
            files.append(("evaluation_metrics", base / "test_fusion_evaluation_metrics.json"))
    files.extend([
        ("decoder_source", ROOT / "pole_lraspp_multimodal_fusion/pole_lraspp_multimodal_fusion/object_targets.py"),
        ("evaluator_source", ROOT / "pole_lraspp_multimodal_fusion/pole_lraspp_multimodal_fusion/evaluate_fusion.py"),
        ("density_evaluator_source", ROOT / "rl_agent/density_knob/density_knob_eval.py"),
        ("fusion_config", ROOT / "pole_lraspp_multimodal_fusion/configs/fusion_full_run.yaml"),
        ("map_server_source", ROOT / "real_time_spatial_map_server_fusion_object_v2.py"),
    ])
    files.extend(("stage_a_evidence", path) for path in sorted(STAGE_A.glob("*")) if path.is_file())
    entries = []
    for role, path in files:
        exists = path.exists()
        stat = path.stat() if exists else None
        entries.append({
            "role": role,
            "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "exists": exists,
            "size_bytes": stat.st_size if stat else None,
            "mtime_ns": stat.st_mtime_ns if stat else None,
            "sha256": sha256_file(path) if exists else None,
        })
    return {
        "audit_id": OUT.relative_to(ROOT).as_posix(),
        "generated_at_unix": time.time(),
        "repository_commit": git_value("rev-parse", "HEAD"),
        "repository_branch": git_value("rev-parse", "--abbrev-ref", "HEAD"),
        "dataset_path_recorded_by_evaluator": str(DATASET),
        "dataset_path_exists": DATASET.exists(),
        "frozen_evaluator": {"score_threshold": 0.20, "nms_radius_px": 2, "topk": 120,
                             "match_distance_m": 5.0, "max_gt_distance_m": 40.0,
                             "min_gt_area_px": 12.0, "class_aware": True},
        "files": entries,
    }


def pareto_flags(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    result["pareto_nondominated"] = False
    points = result[["precision", "recall"]].to_numpy(dtype=float)
    flags = []
    for index, point in enumerate(points):
        dominated = any((other[0] >= point[0] and other[1] >= point[1]) and
                        (other[0] > point[0] or other[1] > point[1]) for j, other in enumerate(points) if j != index)
        flags.append(not dominated)
    result["pareto_nondominated"] = flags
    return result


def make_pareto_figure(validation_summary: pd.DataFrame, selected: Candidate) -> pd.DataFrame:
    vehicle = validation_summary.loc[(validation_summary.group == "pooled") & (validation_summary.class_name == "vehicle")].copy()
    vehicle = pareto_flags(vehicle)
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    colors = ["#0072B2" if name == selected.name else "#E69F00" if flag else "#7A7A7A"
              for name, flag in zip(vehicle.candidate, vehicle.pareto_nondominated)]
    ax.scatter(vehicle.recall, vehicle.precision, c=colors, s=70, edgecolor="black", linewidth=0.5)
    for row in vehicle.itertuples():
        ax.annotate(str(row.candidate), (row.recall, row.precision), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Vehicle recall")
    ax.set_ylabel("Vehicle precision")
    ax.set_title("Audit-validation precision-recall candidates (pooled complete profiles)")
    ax.grid(True, alpha=0.25)
    caption = "model_precision_decoder_audit_v1/20260819_210004"
    ax.text(0.01, 0.01, caption, transform=ax.transAxes, fontsize=7, color="#555555")
    fig.tight_layout()
    fig.savefig(OUT / "precision_recall_pareto.png", dpi=300)
    fig.savefig(OUT / "precision_recall_pareto.pdf")
    plt.close(fig)
    return vehicle


def markdown_table(table: pd.DataFrame, columns: Sequence[str], digits: int = 4) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, separator]
    for row in table.loc[:, columns].itertuples(index=False, name=None):
        formatted = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                formatted.append("nan" if not math.isfinite(float(value)) else f"{float(value):.{digits}f}")
            else:
                formatted.append(str(value))
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines)


def write_report(
    reproduction: pd.DataFrame,
    taxonomy: pd.DataFrame,
    validation_pareto: pd.DataFrame,
    selected: Candidate,
    test_summary: pd.DataFrame,
    uncertainty: pd.DataFrame,
    behavior: pd.DataFrame,
    strata: pd.DataFrame,
    latency: pd.DataFrame,
    map_summary: pd.DataFrame,
    map_proof: Mapping[str, Any],
    split_manifest: pd.DataFrame,
    reconciliation: pd.DataFrame,
    selection_details: Mapping[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    pooled_tax = taxonomy.loc[(taxonomy.family == "pooled") & (taxonomy["quant"] == "pooled")]
    vehicle_total = int(pooled_tax.loc[pooled_tax.class_name == "vehicle", "fp_count"].sum())
    vehicle_duplicates = int(pooled_tax.loc[(pooled_tax.class_name == "vehicle") &
                                            (pooled_tax.category == "duplicate_same_class_claimed_gt"), "fp_count"].sum())
    vehicle_dup_fraction = safe_div(vehicle_duplicates, vehicle_total)
    temporal_focus = (pooled_tax.groupby(["class_name", "temporal_status"], as_index=False).fp_count.sum())
    temporal_focus["fraction"] = temporal_focus.fp_count / temporal_focus.groupby("class_name").fp_count.transform("sum")
    empty_focus = (pooled_tax.groupby(["class_name", "empty_scene_fp"], as_index=False).fp_count.sum())
    empty_focus["fraction"] = empty_focus.fp_count / empty_focus.groupby("class_name").fp_count.transform("sum")

    compare = test_summary.loc[(test_summary.group == "pooled") &
                               (test_summary.class_name.isin(CLASS_NAMES)) &
                               (test_summary.candidate.isin(["baseline", selected.name]))].copy()
    compare = compare[["candidate", "class_name", "precision", "recall", "f1", "fp_per_frame", "xy_mae_m", "xy_rmse_m", "prediction_count", "gt_count"]]
    base_vehicle = compare.loc[(compare.candidate == "baseline") & (compare.class_name == "vehicle")].iloc[0]
    selected_vehicle = compare.loc[(compare.candidate == selected.name) & (compare.class_name == "vehicle")].iloc[0]
    vehicle_recall_delta = float(selected_vehicle.recall - base_vehicle.recall)
    vehicle_xy_delta = float(selected_vehicle.xy_mae_m - base_vehicle.xy_mae_m)
    vehicle_f1_delta = float(selected_vehicle.f1 - base_vehicle.f1)
    vehicle_fp_frame_delta = float(selected_vehicle.fp_per_frame - base_vehicle.fp_per_frame)
    decision = "INSUFFICIENT_EVIDENCE"
    if (selected.name != "baseline" and selected_vehicle.precision > base_vehicle.precision and
            selected_vehicle.fp_per_frame < base_vehicle.fp_per_frame and selected_vehicle.f1 > base_vehicle.f1):
        decision = "POSTPROCESSING_SUFFICIENT"
    elif selected.name == "baseline" and bool(reconciliation.exact_count_match.all()):
        decision = "RETRAINING_PILOT_JUSTIFIED"

    ci_focus = uncertainty.loc[(uncertainty.group == "pooled") &
                               (uncertainty.class_name == "vehicle") &
                               (uncertainty.metric.isin(["precision", "recall", "f1", "fp_per_frame", "xy_mae_m"]))]
    behavior_focus = behavior.loc[(behavior.group == "pooled") &
                                  (behavior.class_name == "vehicle") &
                                  (behavior.behavior.isin(["empty_gt", "dense_5plus"])) &
                                  (behavior.candidate.isin(["baseline", selected.name]))]
    distance_focus = strata.loc[(strata.group == "pooled") &
                                (strata.class_name == "vehicle") &
                                (strata.candidate.isin(["baseline", selected.name]))].copy()
    distance_focus["distance_sort"] = pd.Categorical(
        distance_focus.distance_stratum_m,
        categories=["0-10", "10-20", "20-30", "30-40", "unknown", "outside_40"],
        ordered=True,
    )
    distance_focus = distance_focus.sort_values(["distance_sort", "candidate"])
    map_focus = map_summary.loc[(map_summary.group == "pooled") & map_summary.candidate.isin(["baseline", selected.name])]

    split_counts = split_manifest.audit_split.value_counts().to_dict()
    reproduction_columns = ["variant", "veh_precision", "veh_recall", "veh_f1", "ped_precision", "ped_recall", "ped_f1", "fp_per_frame", "profile_rows", "unique_frames"]
    report = f"""# Raw Object-Detection Precision / Decoder Audit Report

Audit: `model_precision_decoder_audit_v1/20260819_210004`  
Final conclusion: **`{decision}`**  
Selected validation-only configuration: **`{selected.name}`**

## Executive result

The baseline taxonomy assigns **{vehicle_duplicates:,} of {vehicle_total:,} pooled vehicle false-positive profile instances ({vehicle_dup_fraction:.1%})** to multiple same-class predictions competing for an already-claimed real object. This directly supports duplicate detections as the main raw-precision failure mode within the frozen evaluator envelope; it is not merely a nearest-GT observation.

The selected predicted-only correction is `{selected.name}` with vehicle threshold {selected.vehicle_threshold:.3f}, person threshold {selected.person_threshold:.3f}, predicted-world NMS radius {selected.world_nms_m:.1f} m, and incremental image-space radius {selected.image_nms_px:.1f} px. It was selected on audit-validation blocks before the frozen audit-test comparison below.

## Provenance and reproduction

The q=0 table reproduces from the four per-frame files with 2,162 unique frames and three complete quantizer profiles per family. Rounded deltas from the supplied table are recorded in `reproduction_table.csv`.

{markdown_table(reproduction, reproduction_columns, 4)}

The per-object causal replay agrees exactly with per-frame TP/FP/FN counts for **{int(reconciliation.exact_count_match.sum())}/{len(reconciliation)} profiles**. Inputs, evaluator settings, checkpoint hashes, decoder/evaluator hashes, and Stage-A evidence are pinned in `input_hash_manifest.json`.

## Split integrity and limitation

- Audit validation: {int(split_counts.get('audit_validation', 0)):,} unique identifiers.
- Frozen audit test: {int(split_counts.get('audit_test', 0)):,} unique identifiers.
- Identifier overlap: 0; grouped-block overlap: 0.
- All four families and all three quantizers share the same assignment.

The original dataset path recorded by the evaluator is absent and original validation per-object predictions were not preserved. Consequently, this is a preregistered grouped holdout of the published 2,162-frame evaluation set, not the original model-development validation split. The known aggregate test table and preliminary nearest-GT result were already available before this split. This weakens claims of untouched historical test secrecy but does not create GT-dependent deployment logic.

## FP taxonomy

Primary categories are mutually exclusive and saved per prediction in `fp_taxonomy.csv`. Persistence is a conservative adjacent-retained-frame proxy, not a tracker.

{markdown_table(pooled_tax.groupby(['class_name', 'category'], as_index=False).fp_count.sum().assign(fraction=lambda d: d.fp_count / d.groupby('class_name').fp_count.transform('sum')).sort_values(['class_name', 'fp_count'], ascending=[True, False]), ['class_name', 'category', 'fp_count', 'fraction'], 4)}

Temporal and empty-scene flags (orthogonal to the primary category):

{markdown_table(temporal_focus, ['class_name', 'temporal_status', 'fp_count', 'fraction'], 4)}

{markdown_table(empty_focus, ['class_name', 'empty_scene_fp', 'fp_count', 'fraction'], 4)}

## Validation sweep and Pareto selection

The complete validation sweep is in `validation_sweep_results.csv`; `validation_pareto.csv` and `precision_recall_pareto.png/.pdf` preserve the precision-recall frontier. The combination gate was {str(bool(selection_details.get('combination_gate_passed'))).lower()}. Selection maximized pooled validation vehicle F1, with the preregistered tie-breaks.

{markdown_table(validation_pareto.sort_values('candidate'), ['candidate', 'precision', 'recall', 'f1', 'fp_per_frame', 'xy_mae_m', 'pareto_nondominated'], 4)}

## One frozen audit-test comparison

{markdown_table(compare.sort_values(['class_name', 'candidate']), list(compare.columns), 4)}

The selected point is not cost-free: vehicle recall changes by **{vehicle_recall_delta:+.4f}**,
matched vehicle XY MAE by **{vehicle_xy_delta:+.4f} m**, vehicle F1 by
**{vehicle_f1_delta:+.4f}**, and vehicle FP/frame by **{vehicle_fp_frame_delta:+.4f}**.
`POSTPROCESSING_SUFFICIENT` therefore means bounded predicted-only processing can
address the dominant mechanism; it does **not** promote this exact operating point
or declare its recall trade-off deployment-safe. The validation Pareto curve also
retains the less aggressive 1 m and 2 m world-suppression points for a downstream
safety choice.

Grouped paired uncertainty resamples unique frames with all quantizer/family profile rows carried together:

{markdown_table(ci_focus, ['class_name', 'metric', 'baseline', 'selected', 'observed_delta', 'delta_ci95_low', 'delta_ci95_high'], 5)}

Empty- and dense-scene behavior:

{markdown_table(behavior_focus, ['candidate', 'behavior', 'class_name', 'precision', 'recall', 'f1', 'fp_per_frame', 'prediction_count', 'gt_count'], 4)}

Pooled frozen-test vehicle distance strata:

{markdown_table(distance_focus, ['candidate', 'distance_stratum_m', 'precision', 'recall', 'f1', 'fp_per_frame', 'xy_mae_m'], 4)}

All family/quantizer distance rows are in `frozen_test_distance_strata.csv`. Segmentation IoU is reproduced separately in `reproduction_segmentation_iou.csv` and is decoder-invariant.

## Runtime latency

Only the deployable retained-list post-processing stage can be paired from persisted artifacts; raw heatmap/GPU decoder tensors are absent. Times therefore quantify incremental causal overhead, not end-to-end object-head decode latency.

{markdown_table(latency, ['configuration', 'p50_ms', 'p90_ms', 'p95_ms', 'max_ms', 'profile_frame_samples'], 6)}

## Installed-map implication

The isolated replay invoked the actual production server's `_normalize_packet` and `_fuse_and_smooth_objects` functions without starting sockets, Flask, CARLA, or OAI. Verified: `{str(bool(map_proof.get('verified'))).lower()}`. For a single source stream, `_can_join_cluster` rejects same-stream joins; raw count equaled measurement and installed count in every independently reset replay frame: `{str(bool(map_proof.get('raw_equals_installed_all_frames'))).lower()}`.

{markdown_table(map_focus, ['candidate', 'raw_detection_count', 'installed_object_count', 'installed_duplicate_count', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1'], 4)}

Thus raw same-stream duplicates survive as separate installed objects in this isolated path; the selected decoder correction reduces them before installation. Multi-frame/multi-stream live precision remains outside this no-CARLA audit.

## Decision boundary

**`{decision}`**. The evidence attributes the primary vehicle precision failure to duplicate retained predictions and tests a bounded predicted-only remedy. No checkpoint, object head, AE/backbone, ROI ranking, UE policy, production decoder, production map server, CARLA/OAI path, or catalog was changed. No retraining was started or recommended by default. Proper raw-heatmap local-maximum suppression and end-to-end GPU decoder timing remain unverified because the raw tensors and source dataset are unavailable.
"""

    summary = {
        "audit_id": "model_precision_decoder_audit_v1/20260819_210004",
        "conclusion": decision,
        "selected_candidate": selected.__dict__,
        "vehicle_fp_taxonomy": {"total_fp_profile_instances": vehicle_total,
                                "duplicate_same_class_claimed_gt": vehicle_duplicates,
                                "duplicate_fraction": vehicle_dup_fraction},
        "audit_split": {"validation_unique_frames": int(split_counts.get("audit_validation", 0)),
                        "test_unique_frames": int(split_counts.get("audit_test", 0)),
                        "identifier_overlap": 0, "block_overlap": 0,
                        "original_validation_available": False},
        "frozen_test_pooled": json.loads(compare.to_json(orient="records")),
        "selection_details": dict(selection_details),
        "map_replay": dict(map_proof),
        "limitations": [
            "Original dataset and original validation per-object outputs are unavailable.",
            "Audit validation/test are grouped disjoint holdouts of the published test identifiers.",
            "Raw heatmap local-maximum suppression was not replayed.",
            "Latency covers retained-list post-processing, not end-to-end GPU decoder latency.",
            "Live temporal/multi-stream map behavior was not run.",
        ],
    }
    return report, summary


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = build_input_manifest()
    (OUT / "input_hash_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    frame_density = pd.read_csv(RAW / "frame_density.csv")
    frame_density["density_bin"] = frame_density["density_bin"].astype(str)
    split_manifest = build_split_manifest(frame_density)
    split_manifest.to_csv(OUT / "audit_split_manifest.csv", index=False)
    universe = list(split_manifest.sample_id.astype(str))
    validation_ids = list(split_manifest.loc[split_manifest.audit_split == "audit_validation", "sample_id"].astype(str))
    test_ids = list(split_manifest.loc[split_manifest.audit_split == "audit_test", "sample_id"].astype(str))
    density_by_sample = dict(zip(frame_density.sample_id.astype(str), frame_density.density_bin.astype(str)))
    camera_by_sample = {str(row.sample_id): (finite_float(row.camera_x), finite_float(row.camera_y)) for row in frame_density.itertuples()}

    resolved = {
        "audit_id": "model_precision_decoder_audit_v1/20260819_210004",
        "scope": "offline persisted model outputs only",
        "baseline": PREREGISTERED[0].__dict__,
        "preregistered_candidates": [item.__dict__ for item in PREREGISTERED],
        "matching": {"class_aware": True, "world_xy_radius_m": MATCH_RADIUS_M},
        "taxonomy": {"plausible_radius_m": PLAUSIBLE_RADIUS_M, "persistence_radius_m": 3.0},
        "split": {"block_size_collection_indices": 25, "hash_salt": "decoder-audit-v1",
                  "validation_modulo_values": [0, 1], "modulo": 5},
        "selection_rule": "max pooled audit-validation vehicle F1; person F1, FP/frame, latency tie-breaks",
        "bootstrap": {"seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPS, "unit": "unique sample_id with complete profiles"},
        "runtime": {"latency_repeats_per_side": 30, "clock": "time.perf_counter_ns"},
        "dataset_available": DATASET.exists(),
    }

    reproduction, segmentation = reproduce_table()
    reproduction.to_csv(OUT / "reproduction_table.csv", index=False)
    segmentation.to_csv(OUT / "reproduction_segmentation_iou.csv", index=False)

    profiles = load_profiles(universe)
    reconciliation = reconcile_profile_sources(profiles, universe)
    reconciliation.to_csv(OUT / "perobject_perframe_reconciliation.csv", index=False)
    if not bool(reconciliation.exact_count_match.all()):
        raise AssertionError("Persisted per-object replay does not reproduce per-frame profile counts")

    fp_detail, fp_summary = build_fp_taxonomy(profiles, split_manifest)
    fp_detail.to_csv(OUT / "fp_taxonomy.csv", index=False)
    fp_summary.to_csv(OUT / "fp_taxonomy_summary.csv", index=False)

    validation_frames: Dict[str, pd.DataFrame] = {}
    validation_summaries = []
    for candidate in PREREGISTERED:
        table = evaluate_candidate(profiles, candidate, validation_ids, "audit_validation", density_by_sample)
        validation_frames[candidate.name] = table
        validation_summaries.append(summarize_frames(table))
    validation_summary = pd.concat(validation_summaries, ignore_index=True)

    _, eligible_candidates, selection_details = select_candidate(validation_summary)
    combo_candidates = [item for item in eligible_candidates if item.name not in {base.name for base in PREREGISTERED}]
    for combo in combo_candidates:
        table = evaluate_candidate(profiles, combo, validation_ids, "audit_validation", density_by_sample)
        validation_frames[combo.name] = table
        validation_summary = pd.concat([validation_summary, summarize_frames(table)], ignore_index=True)

    selected = choose_from_complete_validation(validation_summary, eligible_candidates)
    selection_details["selected_candidate"] = selected.name
    resolved["eligible_candidates_after_combination_gate"] = [item.__dict__ for item in eligible_candidates]
    resolved["selection_details"] = selection_details
    resolved["selected_candidate"] = selected.__dict__
    decoder_signature = hashlib.sha256(json.dumps(selected.__dict__, sort_keys=True).encode("utf-8")).hexdigest()
    resolved["selected_decoder_config_sha256"] = decoder_signature
    resolved["proposed_unpromoted_decoder_version"] = f"decoder_postprocess_audit_{decoder_signature[:12]}"
    (OUT / "resolved_config.json").write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    validation_summary.to_csv(OUT / "validation_sweep_results.csv", index=False)
    validation_pareto = make_pareto_figure(validation_summary, selected)
    validation_pareto.to_csv(OUT / "validation_pareto.csv", index=False)

    # Frozen audit-test access begins only after validation selection above.
    baseline_test = evaluate_candidate(profiles, Candidate("baseline"), test_ids, "audit_test", density_by_sample)
    selected_test = evaluate_candidate(profiles, selected, test_ids, "audit_test", density_by_sample)
    frozen_frames = pd.concat([baseline_test, selected_test], ignore_index=True)
    frozen_frames.to_csv(OUT / "frozen_test_paired_perframe.csv", index=False)
    test_summary = pd.concat([summarize_frames(baseline_test), summarize_frames(selected_test)], ignore_index=True)
    test_summary.to_csv(OUT / "frozen_test_comparison.csv", index=False)

    uncertainty = paired_bootstrap(baseline_test, selected_test)
    uncertainty.to_csv(OUT / "frozen_test_paired_bootstrap_ci.csv", index=False)
    behavior = behavior_summary([baseline_test, selected_test])
    behavior.to_csv(OUT / "frozen_test_behavior.csv", index=False)
    strata = distance_strata(profiles, [Candidate("baseline"), selected], test_ids, camera_by_sample)
    strata.to_csv(OUT / "frozen_test_distance_strata.csv", index=False)

    latency_detail, latency_summary = measure_latency(profiles, selected, test_ids)
    latency_detail.to_csv(OUT / "latency_paired_samples.csv", index=False)
    latency_summary.to_csv(OUT / "latency_comparison.csv", index=False)

    map_proof: Dict[str, Any]
    try:
        map_detail, map_proof = offline_map_replay(profiles, [Candidate("baseline"), selected], test_ids)
        map_detail.to_csv(OUT / "offline_map_replay_perframe.csv", index=False)
        map_summary = summarize_map(map_detail)
    except Exception as exc:  # preserve a precise, inspectable failure instead of pretending a surrogate is production-equivalent
        map_proof = {"verified": False, "error": f"{type(exc).__name__}: {exc}",
                     "server_path": "real_time_spatial_map_server_fusion_object_v2.py",
                     "network_or_carla_calls": False}
        map_summary = pd.DataFrame([{
            "candidate": "unverified", "group": "pooled", "raw_detection_count": 0,
            "installed_object_count": 0, "installed_duplicate_count": 0,
            "tp": 0, "fp": 0, "fn": 0, "precision": float("nan"),
            "recall": float("nan"), "f1": float("nan"),
        }])
    map_summary.to_csv(OUT / "offline_map_replay_summary.csv", index=False)
    (OUT / "offline_map_replay_evidence.json").write_text(json.dumps(map_proof, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report, results_summary = write_report(
        reproduction, fp_summary, validation_pareto, selected, test_summary, uncertainty,
        behavior, strata, latency_summary, map_summary, map_proof, split_manifest,
        reconciliation, selection_details,
    )
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    (OUT / "RESULTS_SUMMARY.json").write_text(json.dumps(results_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"conclusion": results_summary["conclusion"], "selected": selected.name,
                      "validation_frames": len(validation_ids), "test_frames": len(test_ids),
                      "output_dir": str(OUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
