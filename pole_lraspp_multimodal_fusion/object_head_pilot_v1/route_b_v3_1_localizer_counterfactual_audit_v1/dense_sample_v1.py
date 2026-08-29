#!/usr/bin/env python3
"""One frozen epoch-40 traversal that persists only registered native-cell samples."""

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
from typing import Any, Mapping

import numpy as np
import torch

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
NATIVE = PACKAGE.parent / "route_b_v3_1_native_grid_v1"
FUSION = ROOT / "pole_lraspp_multimodal_fusion"
for path in (str(PACKAGE), str(NATIVE), str(FUSION), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from common_v1 import read_csv, sha256, utc_now, write_csv_x, write_json_x, write_text_x  # noqa: E402
from pole_lraspp_multimodal_fusion.common import read_manifest  # noqa: E402
from pole_lraspp_multimodal_fusion.evaluate_fusion import load_fused_tensor  # noqa: E402
from pole_lraspp_multimodal_fusion.object_targets import (  # noqa: E402
    REG_DIMS, REG_LOCAL_XYZ, REG_YAW, parse_matrix, transform_point,
)


def load_native_model() -> Any:
    spec = importlib.util.spec_from_file_location(
        "localizer_counterfactual_native_model_v1", NATIVE / "model_v1.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("unable to load native-grid model")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


native = load_native_model()

FIELDS = (
    "sample_kind", "sample_id", "candidate_stable_row", "source_identity",
    "requested_center_x_px", "requested_center_y_px", "cell_x", "cell_y", "cell_valid",
    "base_person_heatmap_score", "base_offset_x", "base_offset_y",
    "base_local_x", "base_local_y", "base_local_z", "base_world_x", "base_world_y",
    "base_world_z", "base_size_x", "base_size_y", "base_size_z", "base_yaw_sin",
    "base_yaw_cos",
)


def grouped_candidate_rows(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as stream:
        for stable, row in enumerate(csv.DictReader(stream)):
            if row["class_name"] != "person":
                continue
            grouped[row["sample_id"]].append({
                "stable_row": stable, "sample_id": row["sample_id"],
                "center_x": float(row["center_x_px"]), "center_y": float(row["center_y_px"]),
            })
    return grouped


def grouped_gt_rows(dataset: Path) -> dict[str, list[dict[str, Any]]]:
    manifest = {row["sample_id"]: row for row in read_csv(dataset / "dataset/manifest.csv")
                if row["split"] == "val"}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_csv(dataset / "contracts/v010/val/object_boxes.csv"):
        if row["label"] != "person":
            continue
        frame = manifest[row["sample_id"]]
        sx = 768.0 / float(frame["camera_width"])
        sy = 432.0 / float(frame["camera_height"])
        grouped[row["sample_id"]].append({
            "sample_id": row["sample_id"], "source_identity": row["source_identity"],
            "center_x": (float(row["gt_bbox_x"]) + 0.5 * float(row["gt_bbox_w"])) * sx,
            "center_y": (float(row["gt_bbox_y"]) + 0.5 * float(row["gt_bbox_h"])) * sy,
        })
    return grouped


def sample_field(object_output: torch.Tensor, *, sample_kind: str, sample_id: str,
                 candidate_stable_row: int | str, source_identity: str,
                 center_x: float, center_y: float, camera_matrix: np.ndarray) -> dict[str, Any]:
    cell_x = math.floor(float(center_x) / 4.0)
    cell_y = math.floor(float(center_y) / 4.0)
    height, width = int(object_output.shape[-2]), int(object_output.shape[-1])
    valid = 0 <= cell_x < width and 0 <= cell_y < height
    row: dict[str, Any] = {
        "sample_kind": sample_kind, "sample_id": sample_id,
        "candidate_stable_row": candidate_stable_row, "source_identity": source_identity,
        "requested_center_x_px": center_x, "requested_center_y_px": center_y,
        "cell_x": cell_x, "cell_y": cell_y, "cell_valid": int(valid),
    }
    if not valid:
        row.update({field: "" for field in FIELDS if field not in row})
        return row
    values = object_output[0].float()
    regs = values[native.SL_REG]
    offsets = values[native.SL_OFFSET].clamp(0.0, 1.0)
    local = regs[REG_LOCAL_XYZ, cell_y, cell_x].detach().cpu().numpy().astype(np.float64)
    dims = torch.clamp(regs[REG_DIMS, cell_y, cell_x], min=0.0).detach().cpu().numpy()
    yaw = regs[REG_YAW, cell_y, cell_x].detach().cpu().numpy()
    yaw_norm = max(1e-6, float(np.hypot(yaw[0], yaw[1])))
    world = transform_point(camera_matrix, local)
    row.update({
        "base_person_heatmap_score": float(torch.sigmoid(values[1, cell_y, cell_x]).item()),
        "base_offset_x": float(offsets[0, cell_y, cell_x].item()),
        "base_offset_y": float(offsets[1, cell_y, cell_x].item()),
        "base_local_x": float(local[0]), "base_local_y": float(local[1]),
        "base_local_z": float(local[2]), "base_world_x": float(world[0]),
        "base_world_y": float(world[1]), "base_world_z": float(world[2]),
        "base_size_x": float(dims[0]), "base_size_y": float(dims[1]),
        "base_size_z": float(dims[2]), "base_yaw_sin": float(yaw[0] / yaw_norm),
        "base_yaw_cos": float(yaw[1] / yaw_norm),
    })
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    args = parser.parse_args()
    experiment = args.experiment.resolve(strict=True)
    if not (experiment / "RECONCILIATION_COMPLETE").is_file():
        raise RuntimeError("dense traversal requires passing reconciliation")
    config = json.loads((experiment / "RESOLVED_CONFIG.json").read_text())
    registration = json.loads((experiment / "REGISTERED_AUDIT_PLAN.json").read_text())
    if registration["inference_allowance"] != config["inference_allowance"]:
        raise RuntimeError("inference allowance drift")
    if config["inference_allowance"]["new_candidate_traversals"] != 0:
        raise RuntimeError("candidate traversal must remain forbidden")
    if sys.executable != "/usr/bin/python3" or not torch.cuda.is_available():
        raise RuntimeError("required /usr/bin/python3 CUDA runtime unavailable")
    checkpoint_path = (ROOT / config["base_checkpoint"]).resolve(strict=True)
    if sha256(checkpoint_path) != config["base_checkpoint_sha256"]:
        raise RuntimeError("epoch-40 checkpoint hash drift")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint["epoch"]) != 40 or len(checkpoint["model"]) != 351:
        raise RuntimeError("unexpected epoch-40 checkpoint contract")
    device = torch.device("cuda")
    model = native.build_native_grid_model(
        radar_channels=int(checkpoint["radar_channels"]),
        hidden_channels=int(checkpoint["object_hidden_channels"]),
        head_depth=int(checkpoint["object_head_depth"]), device=device,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    dataset = (ROOT / config["dataset_root"]).resolve(strict=True)
    rows = [row for row in read_manifest(dataset / "dataset/manifest.csv") if row["split"] == "val"]
    if len(rows) != 3345 or any("canonical_v3_07" in row["sample_id"]
                                or "canonical_v3_08" in row["sample_id"] for row in rows):
        raise RuntimeError("validation/test contract drift")
    candidates = grouped_candidate_rows(
        (ROOT / config["candidate_predictions"] / "detections.csv").resolve(strict=True),
    )
    gt = grouped_gt_rows(dataset)
    write_json_x(experiment / "DENSE_TRAVERSAL_STARTED.json", {
        "schema": "route_b_v3_1_epoch40_dense_field_traversal_started_v1",
        "created_utc": utc_now(), "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": config["base_checkpoint_sha256"],
        "base_traversal_number": 1, "candidate_traversals": 0,
        "validation_frames": len(rows), "persist_dense_maps": False,
        "sampler": config["counterfactual_contract"]["dense_sampler"],
    })
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    samples: list[dict[str, Any]] = []
    with torch.inference_mode():
        for index, frame in enumerate(rows, 1):
            fused, _output_hw, _original = load_fused_tensor(
                frame, dataset / "dataset", tuple(checkpoint["input_size"]), device,
            )
            output = model(fused, feature_drop_fraction=0.0)["object"]
            if tuple(output.shape) != (1, 16, 108, 192) or not torch.isfinite(output).all().item():
                raise RuntimeError(f"invalid epoch-40 dense output: {frame['sample_id']}")
            matrix = parse_matrix(frame["camera_matrix_json"])
            if matrix is None:
                raise RuntimeError(f"missing camera matrix: {frame['sample_id']}")
            camera_matrix = np.asarray(matrix, dtype=np.float64)
            for item in candidates.get(frame["sample_id"], ()):
                samples.append(sample_field(
                    output, sample_kind="candidate_predicted_full_box_cell",
                    sample_id=frame["sample_id"], candidate_stable_row=item["stable_row"],
                    source_identity="", center_x=item["center_x"], center_y=item["center_y"],
                    camera_matrix=camera_matrix,
                ))
            for item in gt.get(frame["sample_id"], ()):
                samples.append(sample_field(
                    output, sample_kind="gt_full_box_center_cell", sample_id=frame["sample_id"],
                    candidate_stable_row="", source_identity=item["source_identity"],
                    center_x=item["center_x"], center_y=item["center_y"],
                    camera_matrix=camera_matrix,
                ))
            if index % 500 == 0:
                print(f"[epoch40 compact dense sampling] {index}/{len(rows)}", flush=True)
    output_path = experiment / "DENSE_FIELD_SAMPLES.csv"
    write_csv_x(output_path, samples, FIELDS)
    candidate_samples = [row for row in samples if row["sample_kind"].startswith("candidate")]
    gt_samples = [row for row in samples if row["sample_kind"].startswith("gt_")]
    result = {
        "schema": "route_b_v3_1_epoch40_dense_field_traversal_v1",
        "created_utc": utc_now(), "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": config["base_checkpoint_sha256"],
        "validation_frames": len(rows), "base_traversal_count": 1,
        "candidate_traversal_count": 0, "segmentation_outputs_created": 0,
        "threshold_passes": 0, "optimizer_steps": 0, "dense_maps_persisted": 0,
        "candidate_cell_samples": len(candidate_samples), "gt_cell_samples": len(gt_samples),
        "invalid_candidate_cells": sum(not row["cell_valid"] for row in candidate_samples),
        "invalid_gt_cells": sum(not row["cell_valid"] for row in gt_samples),
        "samples_sha256": sha256(output_path), "wall_seconds": time.monotonic() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
    }
    write_json_x(experiment / "DENSE_TRAVERSAL.json", result)
    write_text_x(experiment / "DENSE_TRAVERSAL_COMPLETE", "ONE_BASE_TRAVERSAL_COMPLETE\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
