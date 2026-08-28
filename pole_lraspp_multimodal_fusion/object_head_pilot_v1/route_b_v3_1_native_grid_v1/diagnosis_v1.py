#!/usr/bin/env python3
"""Phase A: verify the structural stride-8 / full-resolution-interpolation diagnosis.

Reads the unchanged shared model and target/decoder code, records file+line evidence,
and empirically confirms the four expected consequences. No training, no test split.

Emits LRASPP_NATIVE_GRID_DIAGNOSIS_NOT_CONFIRMED if the diagnosis is materially false.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
FUSION_ROOT = ROOT / "pole_lraspp_multimodal_fusion"
for _path in (str(ROOT), str(FUSION_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from pole_lraspp_multimodal_fusion import model as model_module  # noqa: E402
from pole_lraspp_multimodal_fusion import object_targets as target_module  # noqa: E402
from pole_lraspp_multimodal_fusion.model import build_multitask_fusion_lraspp  # noqa: E402
from pole_lraspp_multimodal_fusion.object_targets import decode_objects  # noqa: E402

MODEL_SIZE = (768, 432)
SOURCE_SIZE = (1280, 720)
BASELINE_STRIDE = 8
NATIVE_STRIDE = 4
BASELINE_NMS_RADIUS_PX = 2


def write_json_x(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def cite(path: Path, needle: str) -> dict[str, Any]:
    """Locate an exact source line so the diagnosis carries file/line evidence."""
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines, 1):
        if needle in line:
            rel = path.relative_to(ROOT)
            return {"file": str(rel), "line": index, "source": line.strip()}
    raise RuntimeError(f"diagnosis evidence line not found in {path}: {needle!r}")


def measure_grid() -> dict[str, Any]:
    """Consequence 0: the object head predicts at stride 8 and is then enlarged."""
    model = build_multitask_fusion_lraspp(
        num_classes=3, radar_channels=4, pretrained=False, object_channels=14,
        object_hidden_channels=128, fuse_low_into_object_head=True,
        head_arch="shared", head_depth=3, predict_bbox2d=True,
    ).eval()
    sample = torch.zeros(1, 7, MODEL_SIZE[1], MODEL_SIZE[0])
    with torch.no_grad():
        features = model.backbone(sample)
        object_input = model._object_input(features)
        native = model.object_head(object_input)
        forward = model(sample)
    return {
        "backbone_features": {
            name: {"shape": list(value.shape), "stride": MODEL_SIZE[0] // int(value.shape[-1])}
            for name, value in features.items()
        },
        "object_input_shape": list(object_input.shape),
        "native_object_logits_shape": list(native.shape),
        "native_object_stride": MODEL_SIZE[0] // int(native.shape[-1]),
        "forward_object_shape": list(forward["object"].shape),
        "forward_object_is_full_resolution": tuple(forward["object"].shape[-2:]) == (MODEL_SIZE[1], MODEL_SIZE[0]),
    }


def measure_interpolated_patch_duplicates(median_box_px: tuple[float, float]) -> dict[str, Any]:
    """Consequences 1 and 2: one native response becomes a broad full-resolution patch
    from which the top-k + 2 px image NMS decoder emits repeated detections, and the
    regression read at those pixels interpolates neighbouring native cells.

    The probe uses a single median-sized vehicle response drawn with the unchanged
    shared target code at native stride-8 resolution, i.e. what a well-fit head emits.
    """
    height, width = MODEL_SIZE[1] // BASELINE_STRIDE, MODEL_SIZE[0] // BASELINE_STRIDE
    box_w, box_h = median_box_px
    radius = int(max(1, round(target_module.gaussian_radius(box_h / BASELINE_STRIDE, box_w / BASELINE_STRIDE))))
    heat = np.zeros((height, width), dtype=np.float32)
    target_module.draw_gaussian(heat, 48.3, 27.4, radius)
    heat[27, 48] = 1.0
    clipped = np.clip(heat, 1e-6, 1.0 - 1e-6)

    native = torch.full((1, 14, height, width), -12.0)
    native[0, 0] = torch.from_numpy(np.log(clipped / (1.0 - clipped)))
    # Two adjacent native column bands carry clearly different metric regression
    # values, so any mixed read is attributable only to the interpolation.
    native[0, 2, :, :], native[0, 2, :, 49:] = 20.0, 40.0
    native[0, 3:8, :, :] = 0.0
    native[0, 7, :, :] = 1.0  # yaw_cos

    enlarged = torch.nn.functional.interpolate(
        native, size=(MODEL_SIZE[1], MODEL_SIZE[0]), mode="bilinear", align_corners=False
    )
    above = int((torch.sigmoid(enlarged[0, 0]) >= 0.20).sum().item())
    predictions = decode_objects(
        enlarged, camera_matrix=np.eye(4), topk=120, score_threshold=0.20,
        nms_radius_px=BASELINE_NMS_RADIUS_PX, predict_bbox2d=True,
    )
    local_x = sorted(float(item["local_x"]) for item in predictions)
    return {
        "probe_object_box_px": [box_w, box_h],
        "native_gaussian_radius_cells": radius,
        "single_object_native_response_cells": int((heat > 0.0).sum()),
        "full_resolution_pixels_above_0_20": above,
        "decoded_detections_from_one_object_response": len(predictions),
        "duplicate_emission_confirmed": len(predictions) > 1,
        "native_regression_values": [20.0, 40.0],
        "decoded_local_x_values": local_x,
        "interpolated_regression_mixing_confirmed": any(
            20.0 + 1e-3 < value < 40.0 - 1e-3 for value in local_x
        ),
    }


def measure_observed_duplicate_separation(detections_csv: Path) -> dict[str, Any]:
    """Empirical confirmation on the retained epoch-20 validation predictions.

    For every vehicle prediction that lies within 2 m of a higher-scoring vehicle
    prediction in the same frame (the registered PREDICTED_DUPLICATE condition),
    measure the IMAGE separation to that partner. If duplicates originate inside one
    interpolated native patch, the separation collapses onto the decoder's own 2 px
    NMS floor of 5 px and sits well inside a single stride-8 cell.
    """
    from collections import defaultdict

    frames: dict[str, list[dict[str, str]]] = defaultdict(list)
    with detections_csv.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["class_name"] == "vehicle" and float(row["score"]) >= 0.20:
                frames[row["sample_id"]].append(row)
    separations: list[float] = []
    for rows in frames.values():
        rows.sort(key=lambda item: -float(item["score"]))
        for index, row in enumerate(rows):
            for other in rows[:index]:  # strictly higher score
                world = float(np.hypot(float(row["world_x"]) - float(other["world_x"]),
                                       float(row["world_y"]) - float(other["world_y"])))
                if world <= 2.0:
                    separations.append(float(np.hypot(
                        float(row["center_x_px"]) - float(other["center_x_px"]),
                        float(row["center_y_px"]) - float(other["center_y_px"]))))
                    break
    array = np.asarray(separations)
    nms_floor_px = 2.0 * BASELINE_NMS_RADIUS_PX + 1.0
    return {
        "vehicle_world_duplicate_pairs_at_0_20": int(array.size),
        "decoder_nms_floor_px": nms_floor_px,
        "image_separation_px": {
            f"p{percent}": float(np.percentile(array, percent)) for percent in (10, 25, 50, 75, 90)
        },
        "median_separation_in_stride8_cells": float(np.median(array) / BASELINE_STRIDE),
        "fraction_at_nms_floor": float(np.mean(array <= nms_floor_px + 1e-6)),
        "fraction_within_one_stride8_cell": float(np.mean(array <= BASELINE_STRIDE)),
        "fraction_within_two_stride8_cells": float(np.mean(array <= 2 * BASELINE_STRIDE)),
        "duplicates_originate_inside_one_interpolated_patch": bool(
            np.median(array) <= 2 * BASELINE_STRIDE and float(np.mean(array <= BASELINE_STRIDE)) >= 0.50
        ),
    }


def measure_object_scale(boxes_csv: Path) -> dict[str, Any]:
    """Consequence 3: small pedestrians occupy about one stride-8 cell or less."""
    scale_x, scale_y = MODEL_SIZE[0] / SOURCE_SIZE[0], MODEL_SIZE[1] / SOURCE_SIZE[1]
    with boxes_csv.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result: dict[str, Any] = {}
    for label in ("vehicle", "person"):
        widths, heights = [], []
        for row in rows:
            if row["label"] != label:
                continue
            widths.append(float(row["gt_bbox_w"]) * scale_x)
            heights.append(float(row["gt_bbox_h"]) * scale_y)
        if not widths:
            continue
        widths_a, heights_a = np.asarray(widths), np.asarray(heights)
        entry: dict[str, Any] = {
            "count": len(widths),
            "median_width_px": float(np.median(widths_a)),
            "median_height_px": float(np.median(heights_a)),
        }
        for stride in (BASELINE_STRIDE, NATIVE_STRIDE):
            cells = (widths_a / stride) * (heights_a / stride)
            entry[f"stride{stride}"] = {
                "median_cell_area": float(np.median(cells)),
                "p10_cell_area": float(np.percentile(cells, 10)),
                "fraction_at_or_below_one_cell": float(np.mean(cells <= 1.0)),
                "fraction_at_or_below_four_cells": float(np.mean(cells <= 4.0)),
            }
        result[label] = entry
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--baseline-experiment", required=True, type=Path)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    started = time.monotonic()

    model_path = Path(model_module.__file__).resolve()
    target_path = Path(target_module.__file__).resolve()
    evidence = {
        "object_logits_enlarged_to_input_resolution": cite(
            model_path, 'object_logits = F.interpolate(object_logits, size=x.shape[-2:]'
        ),
        "enlargement_condition": cite(
            model_path, "if tuple(object_logits.shape[-2:]) != tuple(x.shape[-2:]):"
        ),
        "high_feature_resampled_to_low_grid": cite(
            model_path, 'high = F.interpolate(high, size=low.shape[-2:]'
        ),
        "targets_allocated_at_input_resolution": cite(
            target_path, "heatmap = np.zeros((object_class_count, input_height, input_width)"
        ),
        "regression_allocated_at_input_resolution": cite(
            target_path, "regression = np.zeros((reg_channels, input_height, input_width)"
        ),
        "target_centre_rounded_to_input_pixel": cite(target_path, "ix = int(round(cx))"),
        "decoder_topk_over_enlarged_map": cite(target_path, "scores, indices = torch.topk(flat, k=k)"),
        "decoder_box_nms_not_local_maximum": cite(
            target_path, "if occupied[class_index, y0:y1, x0:x1].any():"
        ),
        "decoder_reads_regression_at_enlarged_pixel": cite(target_path, "local = regs[REG_LOCAL_XYZ, y, x]"),
    }

    baseline = args.baseline_experiment.resolve()
    grid = measure_grid()
    scale = measure_object_scale(baseline / "contracts/v010/val/object_boxes.csv")
    duplicates = measure_interpolated_patch_duplicates(
        (scale["vehicle"]["median_width_px"], scale["vehicle"]["median_height_px"])
    )
    observed = measure_observed_duplicate_separation(
        baseline / "predictions/trained_epoch_020/detections.csv"
    )

    person8 = scale["person"]["stride8"]
    checks = {
        "object_head_predicts_at_stride_8": grid["native_object_stride"] == BASELINE_STRIDE,
        "native_grid_is_96x54": grid["native_object_logits_shape"][-2:] == [54, 96],
        "predictions_enlarged_to_768x432": grid["forward_object_is_full_resolution"],
        "targets_and_decoding_at_enlarged_resolution": True,
        "one_native_response_becomes_broad_patch": duplicates["full_resolution_pixels_above_0_20"] > 1,
        "topk_plus_2px_nms_emits_repeated_detections": duplicates["duplicate_emission_confirmed"],
        "interpolated_regression_mixes_neighbouring_cells": duplicates["interpolated_regression_mixing_confirmed"],
        "observed_duplicates_sit_inside_one_interpolated_patch": observed[
            "duplicates_originate_inside_one_interpolated_patch"
        ],
        "small_pedestrians_occupy_about_one_stride8_cell_or_less": (
            person8["median_cell_area"] <= 4.0 and person8["fraction_at_or_below_one_cell"] >= 0.10
        ),
    }
    confirmed = all(checks.values())

    result = {
        "schema": "route_b_v3_1_native_grid_diagnosis_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_experiment": str(baseline),
        "source_evidence": evidence,
        "measured_grid": grid,
        "interpolated_patch_experiment": duplicates,
        "observed_duplicate_separation_epoch20": observed,
        "gt_object_scale_in_grid_cells": scale,
        "diagnosis_checks": checks,
        "diagnosis_confirmed": confirmed,
        "measured_failure_decomposition_of_record": {
            "vehicle_fp_predicted_duplicate_pct": 48.44,
            "vehicle_fp_two_d_correct_world_wrong_pct": 34.49,
            "person_fn_heatmap_center_miss_pct_at_0_02": 69.66,
        },
        "wall_seconds": time.monotonic() - started,
    }
    if not confirmed:
        result["terminal"] = "LRASPP_NATIVE_GRID_DIAGNOSIS_NOT_CONFIRMED"

    experiment.mkdir(parents=True, exist_ok=True)
    write_json_x(experiment / "PHASE_A_DIAGNOSIS.json", result)
    (experiment / "PHASE_A_COMPLETE").write_text(
        ("DIAGNOSIS_CONFIRMED" if confirmed else "LRASPP_NATIVE_GRID_DIAGNOSIS_NOT_CONFIRMED") + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"diagnosis_confirmed": confirmed, "checks": checks,
                      "native_object_logits_shape": grid["native_object_logits_shape"],
                      "forward_object_shape": grid["forward_object_shape"],
                      "decoded_from_one_object_response": duplicates["decoded_detections_from_one_object_response"],
                      "observed_duplicate_separation": observed["image_separation_px"],
                      "observed_fraction_at_nms_floor": observed["fraction_at_nms_floor"],
                      "person_stride8": person8}, indent=2), flush=True)
    return 0 if confirmed else 2


if __name__ == "__main__":
    raise SystemExit(main())
