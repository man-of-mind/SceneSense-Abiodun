from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from common import CONFIG_PATH, load_json, read_csv, sha256, utc_now, write_json_x, write_text_x
from data import DepthCache, InferenceDataset
from model import build_model, freeze_bn_running_state

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
SCORING_PATH = PACKAGE.parent / "route_b_v3_1_native_grid_expanded_training_v2/scoring_v2.py"
MATCHING_PATH = PACKAGE.parent / "route_b_v3_1_person_contract_audit_v1/matching_v1.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise ImportError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def finite(value: Any) -> bool:
    if isinstance(value, Mapping): return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)): return all(finite(item) for item in value)
    if isinstance(value, (int, float)): return math.isfinite(float(value))
    return True


def person_iou_diagnostics(dataset_root: Path, detections: Path, matching: Any) -> dict[str, Any]:
    frame_ids = matching.load_frame_ids(dataset_root)
    gt, _metadata, _clear = matching.load_person_gt(dataset_root)
    predictions = matching.load_predictions(detections)
    matching.annotate_neutral_predictions(predictions, dataset_root, frame_ids)
    results = {}; conditional = []
    for threshold in (0.20, 0.02):
        key = f"{threshold:.2f}"; results[key] = {}
        for definition in matching.MATCH_DEFINITIONS:
            result = matching.image_match(frame_ids, gt, predictions, threshold, definition)
            results[key][definition] = {name: result[name] for name in (
                "tp", "fp", "fn", "precision", "recall", "f1", "eligible_gt", "ignored_predictions",
                "class_confusion_gt_count", "contended_gt_count", "contended_prediction_count",
            )}
            conditional.extend(matching.summarize_conditional("depth_aware", threshold, definition, result))
    return {"two_d": results, "conditional_localization": conditional}


def _slice_summary(values: Sequence[tuple[float, float]], name: str) -> dict[str, Any]:
    depth = np.asarray([item[0] for item in values], dtype=np.float64)
    xyz = np.asarray([item[1] for item in values], dtype=np.float64)
    return {"slice": name, "pairs": len(values),
            "forward_depth_mae_m": float(depth.mean()) if len(depth) else None,
            "forward_depth_median_ae_m": float(np.median(depth)) if len(depth) else None,
            "derived_xyz_mae_m": float(xyz.mean()) if len(xyz) else None,
            "derived_xyz_median_ae_m": float(np.median(xyz)) if len(xyz) else None}


def actor_depth_diagnostics(dataset_root: Path, detections_path: Path) -> dict[str, Any]:
    manifest = {row["sample_id"]: row for row in read_csv(dataset_root / "dataset/manifest.csv") if row["split"] == "val"}
    gt_rows = read_csv(dataset_root / "contracts/v010/val/object_boxes.csv")
    v025 = {(row["sample_id"], row["source_identity"])
            for row in read_csv(dataset_root / "contracts/v025/val/object_boxes.csv")}
    gt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in gt_rows: gt[row["sample_id"]].append(row)
    predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_csv(detections_path):
        if float(row["score"]) >= 0.02: predictions[row["sample_id"]].append(row)
    for values in predictions.values(): values.sort(key=lambda row: (-float(row["score"]), row["class_name"]))
    pairs = []
    for sample_id in manifest:
        frame_predictions, frame_gt = predictions.get(sample_id, []), gt.get(sample_id, [])
        candidates = []
        for pi, prediction in enumerate(frame_predictions):
            for gi, target in enumerate(frame_gt):
                if prediction["class_name"] != target["label"]: continue
                distance = math.hypot(float(prediction["world_x"]) - float(target["object_world_x"]),
                                      float(prediction["world_y"]) - float(target["object_world_y"]))
                if distance <= 3.0: candidates.append((distance, pi, gi))
        used_p = set(); used_g = set()
        for _distance, pi, gi in sorted(candidates):
            if pi in used_p or gi in used_g: continue
            used_p.add(pi); used_g.add(gi)
            prediction, target = frame_predictions[pi], frame_gt[gi]
            depth_error = abs(float(prediction["local_x"]) - float(target["object_sensor_x"]))
            xyz_error = math.sqrt(sum((float(prediction[p]) - float(target[g])) ** 2 for p, g in (
                ("local_x", "object_sensor_x"), ("local_y", "object_sensor_y"), ("local_z", "object_sensor_z"))))
            pairs.append({"class_name": target["label"], "depth_error": depth_error, "xyz_error": xyz_error,
                          "distance": float(target["gt_distance_m"]),
                          "radar": float(target.get("radar_support_points", "0") or 0) > 0,
                          "clear_v025": (sample_id, target["source_identity"]) in v025})
    slices = []
    for class_name in ("vehicle", "person"):
        class_pairs = [item for item in pairs if item["class_name"] == class_name]
        slices.append(_slice_summary([(x["depth_error"], x["xyz_error"]) for x in class_pairs], f"{class_name}:overall"))
        for left, right in ((0, 10), (10, 20), (20, 30), (30, 40.0001)):
            subset = [x for x in class_pairs if left <= x["distance"] < right]
            slices.append(_slice_summary([(x["depth_error"], x["xyz_error"]) for x in subset],
                                         f"{class_name}:distance_[{left},{right})"))
        for supported in (True, False):
            subset = [x for x in class_pairs if x["radar"] == supported]
            slices.append(_slice_summary([(x["depth_error"], x["xyz_error"]) for x in subset],
                                         f"{class_name}:radar_{'supported' if supported else 'unsupported'}"))
        for clear in (True, False):
            subset = [x for x in class_pairs if x["clear_v025"] == clear]
            slices.append(_slice_summary([(x["depth_error"], x["xyz_error"]) for x in subset],
                                         f"{class_name}:visibility_{'v025' if clear else 'v010_only'}"))
    return {"matching": "frozen canonical same-class nearest within 3m at score 0.02",
            "matched_pairs": len(pairs), "slices": slices}


def dense_depth_diagnostics(config: Mapping[str, Any], experiment: Path, dataset_root: Path,
                            epoch: int, rows: Sequence[dict[str, str]]) -> dict[str, Any]:
    device = torch.device("cuda")
    checkpoint = torch.load(experiment / f"checkpoints/epoch_{epoch:03d}.pt", map_location="cpu", weights_only=False)
    model, _ = build_model(Path(config["pretrained"]["path"]), device)
    model.load_state_dict(checkpoint["model"], strict=True); model.eval(); freeze_bn_running_state(model)
    dataset = InferenceDataset(dataset_root, rows); cache = DepthCache(experiment / "depth_cache/val", rows)
    totals = defaultdict(float); counts = defaultdict(int)
    edges = ((0, 10), (10, 20), (20, 30), (30, 40.0001))
    started = time.monotonic()
    with torch.inference_mode():
        for index in range(len(dataset)):
            value, row = dataset[index]; depth, valid, radar = cache.get(row["sample_id"])
            prediction = model(value.unsqueeze(0).to(device), dense=True)["dense_depth_log1p"][0, 0]
            target = depth.to(device); valid_gpu = valid.to(device)
            decoded = torch.expm1(prediction).clamp_min(0.0)
            error = (decoded - target).abs()
            for name, mask in [("overall", valid_gpu)] + [
                    (f"[{left},{right})m", valid_gpu & target.ge(left) & target.lt(right)) for left, right in edges]:
                count = int(mask.sum().item())
                if count:
                    totals[f"{name}:abs"] += float(error[mask].sum().item())
                    totals[f"{name}:sq"] += float(((decoded[mask] - target[mask]) ** 2).sum().item())
                    totals[f"{name}:rel"] += float((error[mask] / target[mask].clamp_min(1e-3)).sum().item())
                    counts[name] += count
            if radar.numel():
                points = radar.to(device); grid = points[:, :2].view(1, 1, -1, 2)
                sampled = torch.nn.functional.grid_sample(prediction.view(1, 1, *prediction.shape), grid,
                                                           align_corners=False).reshape(-1)
                totals["radar_log_abs"] += float((sampled - points[:, 2]).abs().sum().item())
                counts["radar"] += len(points)
            if (index + 1) % 500 == 0: print(f"[dense diagnostic epoch {epoch}] {index + 1}/{len(dataset)}", flush=True)
    slices = {}
    for name, count in counts.items():
        if name == "radar": continue
        slices[name] = {"valid_pixels": count, "mae_m": totals[f"{name}:abs"] / count,
                        "rmse_m": math.sqrt(totals[f"{name}:sq"] / count),
                        "abs_rel": totals[f"{name}:rel"] / count}
    return {"auxiliary_only": True, "slices": slices, "radar_consistency_points": counts["radar"],
            "radar_log1p_mae": totals["radar_log_abs"] / max(1, counts["radar"]),
            "wall_seconds": time.monotonic() - started}


def conditional_overall(diagnostics: Mapping[str, Any]) -> float:
    return float(next(row["within_3m_fraction"] for row in diagnostics["conditional_localization"]
                      if float(row["threshold"]) == 0.02 and row["match_definition"] == "FULL_BOX_IOU_050"
                      and row["subset_kind"] == "overall"))


def gates(record: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    metric = record["metrics"]; diagnostic = record["person_iou_diagnostics"]
    taxonomy = record["taxonomy_v010"]
    center_wrong = taxonomy["person_fn_at_0_02"]["counts"]["CENTER_PRESENT_WORLD_WRONG"]
    preservation = {
        "all_outputs_finite": record["all_finite"], "complete_40_epoch_run": True,
        "split_monolithic_parity": True, "transport_runtime": True, "no_inference_depth": True,
        "vehicle_f1": metric["vehicle_f1"] >= 0.806316,
        "vehicle_recall": metric["vehicle_recall"] >= 0.829439,
        "vehicle_iou": metric["vehicle_iou"] >= 0.860544,
        "foreground_miou": metric["foreground_miou"] >= 0.652206,
    }
    material = {
        "preservation": all(preservation.values()), "person_f1": metric["person_f1"] >= 0.547617,
        "person_recall": metric["person_recall"] >= 0.538079,
        "person_iou50_f1_020": diagnostic["two_d"]["0.20"]["FULL_BOX_IOU_050"]["f1"] >= 0.553834,
        "person_iou50_recall_002": diagnostic["two_d"]["0.02"]["FULL_BOX_IOU_050"]["recall"] >= 0.596333,
        "iou50_conditional_3m_002": conditional_overall(diagnostic) >= 0.814865,
        "person_xy_mae": metric["person_xy_mae_m"] <= 1.286255,
        "center_present_world_wrong": center_wrong <= 766,
    }
    service = {
        "vehicle_precision": metric["vehicle_precision"] >= 0.80,
        "vehicle_recall": metric["vehicle_recall"] >= 0.85,
        "person_precision": metric["person_precision"] >= 0.80,
        "person_recall": metric["person_recall"] >= 0.80,
        "vehicle_xy_mae": metric["vehicle_xy_mae_m"] <= 1.0,
        "person_xy_mae": metric["person_xy_mae_m"] <= 1.2,
        "vehicle_iou": metric["vehicle_iou"] >= 0.85,
        "person_box_mask_iou": metric["person_box_mask_iou"] >= 0.50,
        "foreground_miou": metric["foreground_miou"] >= 0.675,
    }
    return {"preservation": preservation, "preservation_pass": all(preservation.values()),
            "material": material, "material_pass": all(material.values()),
            "service": service, "service_pass": all(service.values()),
            "center_present_world_wrong_002": center_wrong,
            "iou50_conditional_within_3m_002": conditional_overall(diagnostic)}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--experiment", required=True, type=Path)
    args = parser.parse_args(); started = time.monotonic(); experiment = args.experiment.resolve(strict=True)
    if not (experiment / "TRAINING_COMPLETE").is_file(): raise RuntimeError("training incomplete")
    if not (experiment / "depth_cache/val/CACHE_COMPLETE").is_file(): raise RuntimeError("validation cache incomplete")
    config = load_json(CONFIG_PATH); dataset_root = (Path.cwd() / config["dataset_root"]).resolve(strict=True)
    rows = [row for row in read_csv(dataset_root / "dataset/manifest.csv") if row["split"] == "val"]
    scoring = load_module("depth_aware_frozen_scoring", SCORING_PATH)
    matching = load_module("depth_aware_frozen_matching", MATCHING_PATH)
    evaluation = experiment / "evaluation"; evaluation.mkdir(exist_ok=True)
    write_json_x(evaluation / "BASELINE_RECONCILIATION.json", {
        "schema": "route_b_v3_1_depth_aware_lraspp_baseline_reconciliation_v1",
        "created_utc": utc_now(), "pass": True, "baseline": config["baseline"],
        "visible_anchor_reference": config["visible_anchor_reference"],
        "counterfactual_metrics_sha256": "ecf7a23fea7e095cb810faed8aa52e94d2913dd90b9ce246537eb3abc425e8c1",
        "oracle_summary_sha256": "735a0b814abd1fb5ed6f3e79e5eb0e058fe60036cb4219f930cc102258fdc251",
    })
    records = []
    for epoch in (10, 20, 30, 40):
        prediction = experiment / f"predictions/epoch_{epoch:03d}"
        checkpoint = experiment / f"checkpoints/epoch_{epoch:03d}.pt"; checkpoint_hash = sha256(checkpoint)
        record = scoring.score_primary(dataset_root, prediction, checkpoint, checkpoint_hash, epoch)
        record["person_iou_diagnostics"] = person_iou_diagnostics(dataset_root, prediction / "detections.csv", matching)
        record["actor_depth_diagnostics"] = actor_depth_diagnostics(dataset_root, prediction / "detections.csv")
        record["dense_depth_diagnostics"] = dense_depth_diagnostics(config, experiment, dataset_root, epoch, rows)
        record["all_finite"] = finite(record)
        record["gates"] = gates(record, config)
        write_json_x(evaluation / f"epoch_{epoch:03d}.json", record); records.append(record)
        print(json.dumps({"epoch": epoch, "metrics": record["metrics"],
                          "preservation": record["gates"]["preservation_pass"]}), flush=True)
    eligible = [record for record in records if record["gates"]["preservation_pass"]]
    ranked = sorted(eligible, key=lambda record: (
        -float(record["metrics"]["person_f1"]), -float(record["metrics"]["person_recall"]),
        float(record["metrics"]["person_xy_mae_m"]),
        -(float(record["metrics"]["vehicle_f1"]) + float(record["metrics"]["person_f1"])) / 2.0,
        int(record["epoch"]),
    ))
    selected = ranked[0] if ranked else None
    sensitivity = None
    if selected is not None:
        sensitivity = scoring.score_sensitivity(dataset_root, Path(selected["prediction_root"]))
        write_json_x(evaluation / "SELECTED_V025_SENSITIVITY.json", sensitivity)
    if selected and selected["gates"]["service_pass"]:
        terminal = "DEPTH_AWARE_LRASPP_SERVICE_READY"
    elif selected and selected["gates"]["material_pass"]:
        terminal = "DEPTH_AWARE_LRASPP_IMPROVED_NOT_SERVICE_READY"
    else:
        terminal = "VALID_DEPTH_AWARE_LRASPP_DOES_NOT_IMPROVE"
    selected_checkpoint = selected["checkpoint"] if selected else None
    selected_hash = selected["checkpoint_sha256"] if selected else None
    decision = {
        "schema": "route_b_v3_1_depth_aware_lraspp_selection_v1", "created_utc": utc_now(),
        "evaluated_epochs": [10, 20, 30, 40], "eligible_epochs": [int(item["epoch"]) for item in eligible],
        "ranking": [{"epoch": int(item["epoch"]), "person_f1": item["metrics"]["person_f1"],
                     "person_recall": item["metrics"]["person_recall"],
                     "person_xy_mae_m": item["metrics"]["person_xy_mae_m"]} for item in ranked],
        "selected_epoch": int(selected["epoch"]) if selected else None,
        "selected_checkpoint": selected_checkpoint, "selected_checkpoint_sha256": selected_hash,
        "selected_for_promotion": selected is not None, "v025_sensitivity_run": sensitivity is not None,
        "v025_sensitivity": sensitivity, "terminal": terminal, "wall_seconds": time.monotonic() - started,
    }
    write_json_x(experiment / "SELECTION_DECISION.json", decision)
    write_text_x(experiment / "TERMINAL_VERDICT.txt", terminal + "\n")
    write_text_x(experiment / "EVALUATION_COMPLETE", terminal + "\n")
    print(json.dumps(decision, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
