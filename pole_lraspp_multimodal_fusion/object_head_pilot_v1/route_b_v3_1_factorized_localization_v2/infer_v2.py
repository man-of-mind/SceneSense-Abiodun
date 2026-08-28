#!/usr/bin/env python3
"""One-pass validation inference replacing only native decoded XYZ."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parents[2]
FUSION_ROOT = ROOT / "pole_lraspp_multimodal_fusion"
for path in (str(PACKAGE_ROOT), str(FUSION_ROOT), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from pole_lraspp_multimodal_fusion.common import read_manifest  # noqa: E402
from pole_lraspp_multimodal_fusion.evaluate_fusion import load_fused_tensor  # noqa: E402
from pole_lraspp_multimodal_fusion.object_targets import parse_matrix  # noqa: E402
from decode_v2 import decode_factorized_objects  # noqa: E402
from model_v2 import build_factorized_model  # noqa: E402

FIELDS = (
    "sample_id", "frame_id", "prediction_index", "class_name", "score",
    "world_x", "world_y", "world_z", "local_x", "local_y", "local_z",
    "size_x", "size_y", "size_z", "yaw_sin", "yaw_cos", "parked_score",
    "radar_support_score", "center_x_px", "center_y_px", "bbox_x0", "bbox_y0",
    "bbox_x1", "bbox_y1",
)
XYZ_FIELDS = {"world_x", "world_y", "world_z", "local_x", "local_y", "local_z"}
INVARIANT_FIELDS = tuple(
    field for field in FIELDS
    if field not in XYZ_FIELDS and field not in {"sample_id", "frame_id", "prediction_index"}
)
SCORE_FLOOR = 0.02
TOPK = 120
RANGE_M = 40.0


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--contract-experiment", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    contract_experiment = args.contract_experiment.resolve()
    checkpoint_path = args.checkpoint.resolve(strict=True)
    if sha256(checkpoint_path) != args.checkpoint_sha256:
        raise RuntimeError("factorized checkpoint SHA mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["position_decoder"] != "factorized_xyz":
        raise RuntimeError("checkpoint decoder provenance mismatch")
    config = checkpoint["config"]
    device = torch.device("cuda")
    model = build_factorized_model(
        radar_channels=int(checkpoint["radar_channels"]),
        hidden_channels=int(checkpoint["object_hidden_channels"]),
        head_depth=int(checkpoint["object_head_depth"]),
        localization_hidden=int(config["localization_hidden_channels"]), device=device,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    amended = json.loads((contract_experiment / "AMENDED_BASELINE.json").read_text(encoding="utf-8"))
    baseline_csv = Path(amended["retained_predictions"])
    baseline_root = baseline_csv.parent
    baseline_manifest = json.loads((baseline_root / "inference_manifest.json").read_text(encoding="utf-8"))
    if sha256(baseline_csv) != amended["retained_detections_sha256"]:
        raise RuntimeError("retained baseline detection hash drift")
    baseline_by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(baseline_csv):
        baseline_by_sample[row["sample_id"]].append(row)

    dataset = contract_experiment / "dataset"
    rows = [row for row in read_manifest(dataset / "manifest.csv") if row.get("split") == "val"]
    if len(rows) != 3345 or len({row["sample_id"] for row in rows}) != len(rows):
        raise RuntimeError("derived validation manifest count/uniqueness failure")
    output = experiment / "predictions" / args.tag
    output.mkdir(parents=True, exist_ok=False)
    os.symlink(str((baseline_root / "segmentation").resolve()), output / "segmentation")
    os.symlink(str((baseline_root / "segmentation_manifest.csv").resolve()),
               output / "segmentation_manifest.csv")
    write_json_x(output / "INFERENCE_STARTED.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "tag": args.tag,
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": args.checkpoint_sha256,
        "position_decoder": "factorized_xyz", "score_floor": SCORE_FLOOR,
    })
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    detections_path = output / "detections.csv"
    detection_count = 0
    segmentation_equal = True
    invariant_equal = True
    mismatch_examples: list[str] = []
    with detections_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        with torch.inference_mode():
            for index, row in enumerate(rows, 1):
                fused, output_hw, _original = load_fused_tensor(
                    row, dataset, tuple(config["input_size"]), device
                )
                outputs = model(fused, feature_drop_fraction=0.0)

                seg_logits = F.interpolate(
                    outputs["out"], size=output_hw, mode="bilinear", align_corners=False
                )
                labels = seg_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                baseline_seg = cv2.imread(
                    str(baseline_root / "segmentation" / f"{row['sample_id']}.png"),
                    cv2.IMREAD_UNCHANGED,
                )
                if baseline_seg is None or not np.array_equal(labels, baseline_seg):
                    segmentation_equal = False
                    if len(mismatch_examples) < 10:
                        mismatch_examples.append(f"segmentation:{row['sample_id']}")

                camera_matrix = parse_matrix(row["camera_matrix_json"])
                if camera_matrix is None:
                    predictions = []
                else:
                    scale_x = config["input_size"][0] / float(row["camera_width"])
                    scale_y = config["input_size"][1] / float(row["camera_height"])
                    intrinsic_model = np.asarray([
                        [float(row["camera_fx"]) * scale_x, 0.0, float(row["camera_cx"]) * scale_x],
                        [0.0, float(row["camera_fy"]) * scale_y, float(row["camera_cy"]) * scale_y],
                        [0.0, 0.0, 1.0],
                    ], dtype=np.float64)
                    predictions = decode_factorized_objects(
                        outputs["object"], outputs["localization"],
                        camera_matrix=camera_matrix, intrinsic_model=intrinsic_model,
                        score_threshold=SCORE_FLOOR, topk=TOPK,
                        stride=int(checkpoint["native_stride"]),
                        model_size=config["input_size"],
                        object_class_names=checkpoint["object_class_names"],
                    )
                    camera = np.asarray(camera_matrix)[:3, 3]
                    predictions = [
                        item for item in predictions
                        if math.hypot(float(item["legacy_world_x"]) - camera[0],
                                      float(item["legacy_world_y"]) - camera[1]) <= RANGE_M
                    ]
                baseline_rows = baseline_by_sample.get(row["sample_id"], [])
                if len(predictions) != len(baseline_rows):
                    invariant_equal = False
                    if len(mismatch_examples) < 10:
                        mismatch_examples.append(
                            f"count:{row['sample_id']}:{len(predictions)}!={len(baseline_rows)}"
                        )
                for prediction_index, (prediction, baseline_row) in enumerate(
                    zip(predictions, baseline_rows)
                ):
                    for field in INVARIANT_FIELDS:
                        candidate = prediction.get(field)
                        reference = baseline_row[field]
                        equal = (str(candidate) == reference if field == "class_name"
                                 else float(candidate) == float(reference))
                        if not equal:
                            invariant_equal = False
                            if len(mismatch_examples) < 10:
                                mismatch_examples.append(
                                    f"{field}:{row['sample_id']}:{prediction_index}"
                                )
                    for field in (*XYZ_FIELDS, "predicted_depth_m"):
                        if not math.isfinite(float(prediction[field])):
                            raise RuntimeError(
                                f"non-finite factorized {field}: {row['sample_id']}:{prediction_index}"
                            )
                    output_row = dict(baseline_row)
                    for field in XYZ_FIELDS:
                        output_row[field] = prediction[field]
                    writer.writerow({field: output_row[field] for field in FIELDS})
                detection_count += len(predictions)
                if index % 500 == 0:
                    print(f"[factorized inference {args.tag}] {index}/{len(rows)}", flush=True)

    if not segmentation_equal or not invariant_equal:
        raise RuntimeError(f"hard non-XYZ invariance failure: {mismatch_examples}")
    seg_manifest = baseline_root / "segmentation_manifest.csv"
    result = {
        "schema": "route_b_v3_1_factorized_localization_inference_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(), "tag": args.tag,
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": args.checkpoint_sha256,
        "checkpoint_epoch": int(checkpoint["epoch"]), "validation_frames": len(rows),
        "inference_pass_count": 1, "one_pass_supplies_thresholds": [0.20, 0.02],
        "score_floor": SCORE_FLOOR, "topk_per_class": TOPK,
        "native_local_maximum_kernel": 3, "q": 0, "ae": False,
        "position_decoder": "factorized_xyz",
        "legacy_xyz_retained_but_not_decoded": True,
        "range_filter_position_decoder": "legacy_xyz_for_candidate_structure_invariance",
        "all_non_xyz_detection_fields_bit_identical": invariant_equal,
        "segmentation_outputs_bit_identical": segmentation_equal,
        "segmentation_payload_reused_by_symlink": True,
        "retained_baseline_detections_sha256": amended["retained_detections_sha256"],
        "detection_predictions": detection_count,
        "detections_sha256": sha256(detections_path),
        "segmentation_manifest_sha256": sha256(seg_manifest),
        "prediction_set_sha256": hashlib.sha256(
            (sha256(detections_path) + sha256(seg_manifest)).encode("ascii")
        ).hexdigest(),
        "wall_seconds": time.monotonic() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024.0 ** 2),
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024.0 ** 2),
    }
    write_json_x(output / "inference_manifest.json", result)
    (output / "INFERENCE_COMPLETE").write_text("ONE_FACTORIZED_INFERENCE_PASS_COMPLETE\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
