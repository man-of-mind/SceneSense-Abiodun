from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2

from common import CONFIG_PATH, PACKAGE, ROOT, atomic_json, atomic_text, desktop_notify, load_json, read_csv, sha256, utc_now

EPOCHS = (3, 8, 16, 22, 26)
LEVELS = ("p2", "p3", "p4", "p5", "p6", "p7")
SCORING_PATH = PACKAGE.parent / "route_b_v3_1_native_grid_expanded_training_v2/scoring_v2.py"


def load_scoring() -> Any:
    spec = importlib.util.spec_from_file_location("splitfusion_frozen_route_b_scoring", SCORING_PATH)
    if spec is None or spec.loader is None: raise ImportError(SCORING_PATH)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]); area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(1e-12, area_a + area_b - intersection)


def diagnostic_taxonomy(dataset_root: Path, detections_path: Path, threshold: float) -> dict[str, Any]:
    manifest = {row["sample_id"]: row for row in read_csv(dataset_root / "dataset/manifest.csv") if row["split"] == "val"}
    v025 = {(row["sample_id"], row["source_identity"])
            for row in read_csv(dataset_root / "contracts/v025/val/object_boxes.csv")}
    gt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_csv(dataset_root / "contracts/v010/val/object_boxes.csv"):
        frame = manifest[row["sample_id"]]; sx, sy = 768 / float(frame["camera_width"]), 432 / float(frame["camera_height"])
        gt[row["sample_id"]].append({"class_name": row["label"], "world_x": float(row["object_world_x"]),
            "world_y": float(row["object_world_y"]), "distance": float(row["gt_distance_m"]),
            "radar": float(row.get("radar_support_points", "0") or 0) > 0,
            "clear_v025": (row["sample_id"], row["source_identity"]) in v025,
            "bbox": (float(row["gt_bbox_x"]) * sx, float(row["gt_bbox_y"]) * sy,
                     (float(row["gt_bbox_x"]) + float(row["gt_bbox_w"])) * sx,
                     (float(row["gt_bbox_y"]) + float(row["gt_bbox_h"])) * sy),
            "dimensions": tuple(float(row[key]) for key in ("gt_size_x_m", "gt_size_y_m", "gt_size_z_m")),
            "yaw_rad": math.radians(float(row["object_yaw_deg"])),
            "source_identity": row["source_identity"]})
    predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    identity_count = 0
    for stable, row in enumerate(read_csv(detections_path)):
        if float(row["score"]) < threshold: continue
        identity = tuple(int(value) for value in row["candidate_identity"].split(":"))
        expected_class = 0 if row["class_name"] == "vehicle" else 1
        if (len(identity) != 4 or identity[0] != 0 or not 0 <= identity[1] < len(LEVELS)
                or LEVELS[identity[1]] != row["fpn_level"] or identity[2] != int(row["point_index"])
                or identity[3] != int(row["internal_class"]) or identity[3] != expected_class):
            raise RuntimeError(f"geometry candidate-identity drift: {row['sample_id']} {identity}")
        identity_count += 1
        predictions[row["sample_id"]].append({**row, "stable": stable, "identity": identity,
            "score_f": float(row["score"]),
            "world_x_f": float(row["world_x"]), "world_y_f": float(row["world_y"]),
            "bbox": tuple(float(row[name]) for name in ("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"))})
    for rows in predictions.values(): rows.sort(key=lambda row: (-row["score_f"], row["class_name"], row["stable"]))
    level = {name: Counter() for name in ("vehicle", "person")}; duplicate = Counter(); background = Counter()
    two_d_world_wrong = Counter(); person_miss = Counter(); slices = defaultdict(lambda: Counter(tp=0, fn=0))
    totals = {name: Counter(tp=0, fp=0, fn=0, ignored=0) for name in ("vehicle", "person")}
    geometry = {name: {"dimension_x": [], "dimension_y": [], "dimension_z": [], "yaw_degrees": []}
                for name in ("vehicle", "person")}
    fp_records = []
    for sample_id in manifest:
        targets, preds = gt.get(sample_id, []), predictions.get(sample_id, [])
        if len({pred["identity"] for pred in preds}) != len(preds):
            raise RuntimeError(f"duplicate candidate identity after NMS: {sample_id}")
        candidates = []
        for pi, pred in enumerate(preds):
            for gi, target in enumerate(targets):
                if pred["class_name"] != target["class_name"]: continue
                distance = math.hypot(pred["world_x_f"] - target["world_x"], pred["world_y_f"] - target["world_y"])
                if distance <= 3.0: candidates.append((distance, pi, gi))
        used_p, used_g = set(), set()
        for distance, pi, gi in sorted(candidates):
            if pi in used_p or gi in used_g: continue
            used_p.add(pi); used_g.add(gi); pred, target = preds[pi], targets[gi]
            totals[target["class_name"]]["tp"] += 1; level[target["class_name"]][f"tp_{pred['fpn_level']}"] += 1
            class_geometry = geometry[target["class_name"]]
            for axis, prediction_key, target_value in zip(("x", "y", "z"), ("size_x", "size_y", "size_z"),
                                                           target["dimensions"]):
                class_geometry[f"dimension_{axis}"].append(abs(float(pred[prediction_key]) - target_value))
            predicted_yaw = math.atan2(float(pred["yaw_sin"]), float(pred["yaw_cos"]))
            delta = math.atan2(math.sin(predicted_yaw - target["yaw_rad"]),
                               math.cos(predicted_yaw - target["yaw_rad"]))
            class_geometry["yaw_degrees"].append(abs(math.degrees(delta)))
            for key in (f"distance_{int(target['distance']//10)*10}_{min(40,int(target['distance']//10)*10+10)}",
                        f"radar_{'supported' if target['radar'] else 'unsupported'}",
                        f"visibility_{'v025' if target['clear_v025'] else 'v010_only'}"):
                slices[(target["class_name"], key)]["tp"] += 1
        ignore = cv2.imread(str(dataset_root / f"contracts/v010/val/object_ignore_masks/{sample_id}.png"), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(dataset_root / f"contracts/v010/val/segmentation_masks/{sample_id}.png"), cv2.IMREAD_UNCHANGED)
        for pi, pred in enumerate(preds):
            if pi in used_p: continue
            cx = int(round((pred["bbox"][0] + pred["bbox"][2]) / 2)); cy = int(round((pred["bbox"][1] + pred["bbox"][3]) / 2))
            neutral = 0 <= cx < 768 and 0 <= cy < 432 and int(ignore[cy, cx]) != 0
            if neutral:
                totals[pred["class_name"]]["ignored"] += 1; continue
            totals[pred["class_name"]]["fp"] += 1; level[pred["class_name"]][f"fp_{pred['fpn_level']}"] += 1
            is_background = not (0 <= cx < 768 and 0 <= cy < 432) or int(mask[cy, cx]) == 0
            background[pred["class_name"]] += int(is_background)
            correct_2d = any(target["class_name"] == pred["class_name"] and box_iou(pred["bbox"], target["bbox"]) >= 0.5
                             for target in targets)
            two_d_world_wrong[pred["class_name"]] += int(correct_2d)
            fp_records.append((sample_id, pi, pred, preds))
        for gi, target in enumerate(targets):
            if gi in used_g: continue
            totals[target["class_name"]]["fn"] += 1
            for key in (f"distance_{int(target['distance']//10)*10}_{min(40,int(target['distance']//10)*10+10)}",
                        f"radar_{'supported' if target['radar'] else 'unsupported'}",
                        f"visibility_{'v025' if target['clear_v025'] else 'v010_only'}"):
                slices[(target["class_name"], key)]["fn"] += 1
            if target["class_name"] == "person":
                x0, y0, x1, y1 = target["bbox"]
                has_point = any(pred["class_name"] == "person" and x0 <= (pred["bbox"][0] + pred["bbox"][2]) / 2 < x1
                                and y0 <= (pred["bbox"][1] + pred["bbox"][3]) / 2 < y1 for pred in preds)
                person_miss["centre_present_world_wrong_or_contention" if has_point else "centre_point_miss"] += 1
    for _sample_id, prediction_index, pred, all_predictions in fp_records:
        higher = [other for other in all_predictions[:prediction_index]
                  if other["class_name"] == pred["class_name"]
                  and math.hypot(other["world_x_f"] - pred["world_x_f"],
                                 other["world_y_f"] - pred["world_y_f"]) <= 3.0]
        if higher:
            duplicate[pred["class_name"]] += 1
            if any(other["fpn_level"] != pred["fpn_level"] for other in higher):
                duplicate[f"{pred['class_name']}_cross_level"] += 1
    reconciliation = {name: totals[name]["tp"] + totals[name]["fn"] for name in totals}
    eligible = {name: sum(target["class_name"] == name for values in gt.values() for target in values) for name in totals}
    if reconciliation != eligible: raise RuntimeError(f"TP+FN denominator drift {reconciliation} != {eligible}")
    slice_rows = []
    for (class_name, name), values in sorted(slices.items()):
        slice_rows.append({"class_name": class_name, "slice": name, **values,
                           "recall": values["tp"] / max(1, values["tp"] + values["fn"])})
    geometry_errors = {}
    for class_name, fields in geometry.items():
        axis_values = fields["dimension_x"] + fields["dimension_y"] + fields["dimension_z"]
        geometry_errors[class_name] = {
            "matched": len(fields["yaw_degrees"]),
            "dimension_mae_m": sum(axis_values) / len(axis_values) if axis_values else None,
            "dimension_x_mae_m": sum(fields["dimension_x"]) / len(fields["dimension_x"]) if fields["dimension_x"] else None,
            "dimension_y_mae_m": sum(fields["dimension_y"]) / len(fields["dimension_y"]) if fields["dimension_y"] else None,
            "dimension_z_mae_m": sum(fields["dimension_z"]) / len(fields["dimension_z"]) if fields["dimension_z"] else None,
            "yaw_mae_deg": sum(fields["yaw_degrees"]) / len(fields["yaw_degrees"]) if fields["yaw_degrees"] else None,
        }
    return {"threshold": threshold, "totals": {name: dict(value) for name, value in totals.items()},
            "eligible_gt": eligible, "tp_plus_fn_reconciles": True,
            "fpn_level_attribution": {name: dict(value) for name, value in level.items()},
            "duplicate_fp": dict(duplicate), "cross_level_duplicate_fp": {name: duplicate[f"{name}_cross_level"] for name in totals},
            "background_fp": dict(background), "two_d_correct_world_wrong": dict(two_d_world_wrong),
            "person_centre_point_miss": dict(person_miss), "slices": slice_rows,
            "geometry_errors": geometry_errors,
            "geometry_identity_fields_verified": ["image", "level", "flattened_point", "internal_class"],
            "geometry_identity_predictions_verified": identity_count,
            "geometry_candidate_identities_unique_within_frame": True}


def install_undefined_localization_adapter(scoring: Any) -> None:
    """Preserve frozen scoring while representing no-match MAE as undefined.

    The frozen flattener assumes both classes have at least one matched
    detection and adds their localization MAEs. A valid zero-detection
    checkpoint instead returns ``None`` for both MAEs. Keep every registered
    metric unchanged and emit ``None`` for their mean so the existing service
    and ranking code treats the checkpoint as failing those finite targets.
    """
    native = scoring.native_evaluator()
    frozen_flatten = native.flatten

    def flatten(primary_020: dict[str, Any], primary_002: dict[str, Any],
                segmentation: dict[str, Any]) -> dict[str, Any]:
        vehicle = primary_020["classes"]["vehicle"]
        person = primary_020["classes"]["person"]
        if vehicle["xy_mae_m"] is not None and person["xy_mae_m"] is not None:
            return frozen_flatten(primary_020, primary_002, segmentation)
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
            "mean_xy_mae_m": None,
            "foreground_miou": segmentation["foreground_miou"],
            "vehicle_iou": segmentation["vehicle_iou"],
            "person_box_mask_iou": segmentation["person_box_mask_iou"],
        }

    native.flatten = flatten


def metric(record: Mapping[str, Any], name: str) -> float:
    try:
        value = float(record["metrics"][name])
        return value if math.isfinite(value) else float("nan")
    except Exception:
        return float("nan")


def service(record: Mapping[str, Any]) -> dict[str, Any]:
    values = {
        "vehicle_precision": (metric(record, "vehicle_precision"), 0.80, "higher"),
        "vehicle_recall": (metric(record, "vehicle_recall"), 0.85, "higher"),
        "person_precision": (metric(record, "person_precision"), 0.80, "higher"),
        "person_recall": (metric(record, "person_recall"), 0.80, "higher"),
        "vehicle_xy_mae_m": (metric(record, "vehicle_xy_mae_m"), 1.0, "lower"),
        "person_xy_mae_m": (metric(record, "person_xy_mae_m"), 1.2, "lower"),
        "vehicle_iou": (metric(record, "vehicle_iou"), 0.85, "higher"),
        "person_box_mask_iou": (metric(record, "person_box_mask_iou"), 0.50, "higher"),
        "foreground_miou": (metric(record, "foreground_miou"), 0.675, "higher"),
    }
    rows, ratios = {}, []
    for name, (value, target, direction) in values.items():
        finite = math.isfinite(value)
        passed = finite and (value >= target if direction == "higher" else value <= target)
        ratio = (value / target if direction == "higher" else target / max(value, 1e-12)) if finite else float("-inf")
        rows[name] = {"value": value if finite else None, "target": target, "direction": direction,
                      "passed": passed, "attainment_ratio": ratio if finite else None}
        ratios.append(ratio if finite else None)
    return {"targets": rows, "pass_count": sum(row["passed"] for row in rows.values()),
            "all_pass": all(row["passed"] for row in rows.values()),
            "minimum_attainment_ratio": min(ratios) if all(value is not None for value in ratios) else None}


def rank_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    gate = record["service"]
    mean_f1 = (metric(record, "vehicle_f1") + metric(record, "person_f1")) / 2
    normalized_localization = (metric(record, "vehicle_xy_mae_m") / 1.0 + metric(record, "person_xy_mae_m") / 1.2) / 2
    def safe(value: Any, worst: float) -> float:
        try:
            return float(value) if math.isfinite(float(value)) else worst
        except (TypeError, ValueError):
            return worst
    return (-int(gate["all_pass"]), -gate["pass_count"], -safe(gate["minimum_attainment_ratio"], -1e99),
            -safe(mean_f1, -1e99), -safe(metric(record, "person_f1"), -1e99),
            safe(normalized_localization, 1e99), -safe(metric(record, "foreground_miou"), -1e99), int(record["epoch"]))


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--experiment", required=True, type=Path)
    args = parser.parse_args(); experiment = args.experiment.resolve(strict=True)
    if not (experiment / "TRAINING_COMPLETE").is_file(): raise RuntimeError("fixed validation forbidden before complete training")
    config = load_json(CONFIG_PATH); dataset_root = (ROOT / config["dataset_root"]).resolve(strict=True)
    scoring = load_scoring(); install_undefined_localization_adapter(scoring)
    evaluation = experiment / "evaluation"; evaluation.mkdir(exist_ok=True)
    records = []
    for epoch in EPOCHS:
        prediction = experiment / f"predictions/epoch_{epoch:03d}"; checkpoint = experiment / f"checkpoints/epoch_{epoch:03d}.pt"
        if not (prediction / "INFERENCE_COMPLETE").is_file(): raise RuntimeError(f"inference incomplete epoch {epoch}")
        record_path = evaluation / f"epoch_{epoch:03d}.json"
        if record_path.is_file():
            result = load_json(record_path)
            inference = load_json(prediction / "inference_manifest.json")
            if (int(result["epoch"]) != epoch or result["checkpoint_sha256"] != sha256(checkpoint)
                    or result["prediction_set_sha256"] != inference["prediction_set_sha256"]):
                raise RuntimeError(f"existing evaluation provenance drift at epoch {epoch}")
            records.append(result)
            continue
        result = scoring.score_primary(dataset_root, prediction, checkpoint, sha256(checkpoint), epoch)
        undefined = sorted(name for name, value in result["metrics"].items() if value is None)
        result["undefined_metrics"] = undefined
        result["all_metrics_finite"] = not undefined and all(
            not isinstance(value, (int, float)) or math.isfinite(float(value))
            for value in result["metrics"].values())
        result["taxonomy_0_20"] = diagnostic_taxonomy(dataset_root, prediction / "detections.csv", 0.20)
        result["taxonomy_0_02"] = diagnostic_taxonomy(dataset_root, prediction / "detections.csv", 0.02)
        result["service"] = service(result)
        result["class_detail"] = {threshold: result["primary_v010"][threshold]["classes"] for threshold in ("0.20", "0.02")}
        if not result["taxonomy_0_20"]["tp_plus_fn_reconciles"] or not result["taxonomy_0_02"]["tp_plus_fn_reconciles"]:
            raise RuntimeError("evaluation denominator drift")
        atomic_json(record_path, result, overwrite=False); records.append(result)
        print(json.dumps({"evaluated_epoch": epoch, "metrics": result["metrics"], "service": result["service"]}), flush=True)
    ranked = sorted(records, key=rank_key); selected = ranked[0]
    sensitivity = scoring.score_sensitivity(dataset_root, Path(selected["prediction_root"]))
    atomic_json(evaluation / "SELECTED_V025_SENSITIVITY.json", sensitivity, overwrite=False)
    terminal = "SPLITFUSION_FCOS_CLEAN_BASE_SERVICE_READY" if selected["service"]["all_pass"] else \
               "SPLITFUSION_FCOS_CLEAN_BASE_NOT_SERVICE_READY"
    decision = {"schema": "splitfusion_fcos_fixed_selection_v1", "created_utc": utc_now(),
                "evaluated_epochs": list(EPOCHS), "ranking": [{"epoch": row["epoch"], "rank_key": rank_key(row),
                    "service": row["service"], "mean_class_f1": (metric(row, "vehicle_f1") + metric(row, "person_f1")) / 2}
                    for row in ranked], "selected_epoch": selected["epoch"],
                "selected_checkpoint": selected["checkpoint"], "selected_checkpoint_sha256": selected["checkpoint_sha256"],
                "selected_prediction_root": selected["prediction_root"], "selected_v025_sensitivity": sensitivity,
                "terminal": terminal, "locked_test_accessed": False}
    atomic_json(experiment / "SELECTION_DECISION.json", decision, overwrite=False)
    atomic_text(experiment / "TERMINAL_VERDICT.txt", terminal + "\n", overwrite=False)
    atomic_text(experiment / "EVALUATION_COMPLETE", "FIXED_FIVE_CHECKPOINT_VALIDATION_COMPLETE\n", overwrite=False)
    atomic_json(experiment / "STATUS.json", {"phase": "D", "state": "complete", "created_utc": utc_now(),
                                              "selected_epoch": selected["epoch"], "terminal": terminal})
    atomic_json(experiment / "NOTIFICATION_EVALUATION_COMPLETE.json", desktop_notify(
        "SplitFusion FCOS", f"Fixed validation complete; selected epoch {selected['epoch']}: {terminal}."), overwrite=False)
    print(json.dumps(decision, indent=2, default=str)); return 0


if __name__ == "__main__": raise SystemExit(main())
