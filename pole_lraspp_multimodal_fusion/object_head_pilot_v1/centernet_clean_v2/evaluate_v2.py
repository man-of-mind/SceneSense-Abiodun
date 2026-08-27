#!/usr/bin/env python3
"""CenterNet v2 validation evaluator - frozen native decoder, create-only output.

Evaluation contract, unchanged from the v1 audit except for the decoder:
score thresholds 0.20 (operating point) and 0.02 (permissive recall diagnostic
only), class-aware matching, 3.0 m match distance, 40 m max GT distance and
prediction range gate, 12 px min GT area, validation split only, greedy-distance
matching as the primary metric, and the legacy ``summarize`` function verbatim
so v1 and v2 numbers are directly comparable.

Segmentation: vehicle IoU is reported normally; the person channel is reported
as ``person_box_mask_iou`` because its ground truth is a filled projected box,
not a silhouette (verified in the v1 audit: mean person-class fill inside person
GT boxes = 0.955, and the corpus contains zero walker semantic pixels).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
for path in (HERE, HERE.parent, HERE.parent.parent, HERE.parent.parent.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from object_head_pilot_v1.evaluate_route_b_checkpoint_v1 import summarize  # noqa: E402
from pole_lraspp_multimodal_fusion.common import (  # noqa: E402
    class_iou_from_confusion,
    load_config,
    read_manifest,
    update_confusion,
)
from pole_lraspp_multimodal_fusion.evaluate_fusion import yaw_error_deg  # noqa: E402
from pole_lraspp_multimodal_fusion.object_targets import (  # noqa: E402
    greedy_match_predictions,
    load_object_boxes,
    parse_matrix,
    valid_localization_objects,
)

from centernet_model_v2 import build_centernet_v2  # noqa: E402
from decode_v2 import DECODER_NAME, TOPK_PER_BRANCH, decode_objects_v2, range_gate  # noqa: E402
from train_v2 import sha256  # noqa: E402

THRESHOLDS = (0.20, 0.02)
DISTANCE_BANDS = ((0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 40.0))
SIZE_BANDS = ((0.0, 8.0), (8.0, 16.0), (16.0, 32.0), (32.0, 64.0), (64.0, float("inf")))
FIXED_CONTRACT = {
    "operating_score_threshold": 0.20,
    "diagnostic_score_threshold": 0.02,
    "topk_objects_per_branch": TOPK_PER_BRANCH,
    "peak_suppression": "per-class 3x3 native local maxima before top-k",
    "match_distance_m": 3.0,
    "max_gt_distance_m": 40.0,
    "min_gt_area_px": 12.0,
    "class_aware_matching": True,
    "matcher": "greedy_distance",
    "decoder": DECODER_NAME,
}

RGB_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
RGB_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


class EvalDataset(Dataset):
    def __init__(self, dataset_dir: Path, rows, input_size: Tuple[int, int]) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.rows = list(rows)
        self.input_width, self.input_height = int(input_size[0]), int(input_size[1])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(self.dataset_dir / row["rgb_path"]).convert("RGB")
        ow, oh = image.size
        image = image.resize((self.input_width, self.input_height), Image.Resampling.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = (torch.from_numpy(arr).permute(2, 0, 1) - RGB_MEAN) / RGB_STD
        payload = np.load(self.dataset_dir / row["radar_tensor_path"])
        try:
            radar = (
                payload["radar"].astype(np.float32)
                if isinstance(payload, np.lib.npyio.NpzFile)
                else np.asarray(payload, dtype=np.float32)
            )
        finally:
            if hasattr(payload, "close"):
                payload.close()
        if radar.shape[2] != self.input_width or radar.shape[1] != self.input_height:
            channels = [
                cv2.resize(
                    channel,
                    (self.input_width, self.input_height),
                    interpolation=cv2.INTER_NEAREST if idx == 0 else cv2.INTER_LINEAR,
                )
                for idx, channel in enumerate(radar)
            ]
            radar = np.stack(channels, axis=0).astype(np.float32)
        fused = torch.cat([image_tensor, torch.from_numpy(np.ascontiguousarray(radar))], dim=0)
        mask = torch.from_numpy(
            np.asarray(Image.open(self.dataset_dir / row["mask_path"]).convert("L"), dtype=np.int64)
        )
        return fused, mask, index, oh, ow


def band_label(value: float, bands) -> str:
    for lo, hi in bands:
        if lo <= value < hi:
            return f"{lo:g}-{hi:g}" if math.isfinite(hi) else f"{lo:g}+"
    return "out_of_range"


def load_model(checkpoint: Dict, device: torch.device) -> torch.nn.Module:
    model = build_centernet_v2(
        num_classes=int(checkpoint.get("num_classes", 3)),
        radar_channels=int(checkpoint.get("radar_channels", 4)),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def evaluate_checkpoint(
    checkpoint_path: Path, loader, rows, object_boxes, out_dir: Path, config, device
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = load_model(checkpoint, device)
    input_size = tuple(int(v) for v in checkpoint["input_size"])
    class_names = tuple(checkpoint.get("object_class_names") or ("vehicle", "person"))
    num_classes = int(checkpoint.get("num_classes", 3))
    min_gt_area = float(config.get("object_heads", {}).get("min_gt_area_px", 12.0))
    max_gt_distance = float(config.get("evaluation", {}).get("max_gt_distance_m", 40.0))
    match_distance = float(config.get("evaluation", {}).get("match_distance_m", 3.0))

    confusion = np.zeros((num_classes, num_classes), dtype=np.float64)
    detection_rows: Dict[float, List[Dict[str, object]]] = {t: [] for t in THRESHOLDS}
    gt_band_rows: Dict[float, List[Dict[str, object]]] = {t: [] for t in THRESHOLDS}

    processed = 0
    with torch.inference_mode():
        for fused, masks, indices, ohs, ows in loader:
            fused = fused.to(device, non_blocking=True)
            outputs = model(fused)
            seg = outputs["out"]
            for b in range(fused.shape[0]):
                index = int(indices[b].item())
                row = rows[index]
                oh, ow = int(ohs[b].item()), int(ows[b].item())
                iw, ih = int(input_size[0]), int(input_size[1])
                logits = F.interpolate(
                    seg[b : b + 1].float(), size=(oh, ow), mode="bilinear", align_corners=False
                )
                update_confusion(
                    confusion,
                    logits.argmax(dim=1)[0].cpu().numpy().astype(np.int64),
                    masks[b].numpy().astype(np.int64),
                    num_classes,
                )
                matrix = parse_matrix(row.get("camera_matrix_json", ""))
                gt_objects = valid_localization_objects(
                    object_boxes.get(row["sample_id"], []),
                    image_width=ow,
                    image_height=oh,
                    min_area_px=min_gt_area,
                    object_class_names=class_names,
                    max_distance_m=max_gt_distance,
                )
                all_preds: List[Dict[str, float]] = []
                if matrix is not None:
                    all_preds = decode_objects_v2(
                        outputs,
                        camera_matrix=matrix,
                        input_size=input_size,
                        score_threshold=min(THRESHOLDS),
                        topk=TOPK_PER_BRANCH,
                        sample_index=b,
                    )
                cam_c = np.asarray(matrix)[:3, 3] if matrix is not None else np.zeros(3)
                sx, sy = float(iw) / max(1.0, float(ow)), float(ih) / max(1.0, float(oh))

                for thr in THRESHOLDS:
                    preds = (
                        range_gate([p for p in all_preds if p["score"] >= thr], matrix, max_gt_distance)
                        if matrix is not None
                        else []
                    )
                    matches = greedy_match_predictions(
                        preds, gt_objects, max_distance_m=match_distance, class_aware=True
                    )
                    matched_pred = {p for p, _, _ in matches}
                    matched_gt = {g for _, g, _ in matches}
                    sink = detection_rows[thr]
                    for p_i, g_i, dist in matches:
                        p, g = preds[p_i], gt_objects[g_i]
                        dim_err = float(
                            np.mean(
                                np.abs(
                                    np.array([p["size_x"], p["size_y"], p["size_z"]])
                                    - np.array([g["size_x"], g["size_y"], g["size_z"]])
                                )
                            )
                        )
                        sink.append(
                            {
                                "split": "val",
                                "sample_id": row["sample_id"],
                                "frame_id": row.get("frame_id", ""),
                                "match_status": "tp",
                                "class_name": str(g["class_name"]),
                                "pred_class_name": p.get("class_name", ""),
                                "gt_class_name": str(g["class_name"]),
                                "score": p["score"],
                                "global_xy_error_m": dist,
                                "dimension_mae_m": dim_err,
                                "yaw_error_deg": yaw_error_deg(p, g),
                                "parked_correct": int(
                                    (float(p["parked_score"]) >= 0.5) == (float(g["parked"]) >= 0.5)
                                ),
                                "branch_stride": p.get("branch_stride", ""),
                                "pred_world_x": p["world_x"],
                                "pred_world_y": p["world_y"],
                                "pred_size_x": p["size_x"],
                                "pred_size_y": p["size_y"],
                                "pred_size_z": p["size_z"],
                                "gt_world_x": g["world_x"],
                                "gt_world_y": g["world_y"],
                                "gt_size_x": g["size_x"],
                                "gt_size_y": g["size_y"],
                                "gt_size_z": g["size_z"],
                                "pred_bbox_x0": p.get("bbox_x0", float("nan")),
                                "pred_bbox_y0": p.get("bbox_y0", float("nan")),
                                "pred_bbox_x1": p.get("bbox_x1", float("nan")),
                                "pred_bbox_y1": p.get("bbox_y1", float("nan")),
                                "input_w": iw,
                                "input_h": ih,
                                "gt_center_x": g.get("center_x", float("nan")),
                                "gt_center_y": g.get("center_y", float("nan")),
                                "gt_bbox_w": g.get("bbox_w", float("nan")),
                                "gt_bbox_h": g.get("bbox_h", float("nan")),
                                "orig_w": ow,
                                "orig_h": oh,
                            }
                        )
                    for p_i, p in enumerate(preds):
                        if p_i not in matched_pred:
                            sink.append(
                                {
                                    "split": "val",
                                    "sample_id": row["sample_id"],
                                    "match_status": "fp",
                                    "class_name": str(p.get("class_name", "object")),
                                    "pred_class_name": str(p.get("class_name", "object")),
                                    "score": p["score"],
                                    "branch_stride": p.get("branch_stride", ""),
                                    "pred_world_x": p["world_x"],
                                    "pred_world_y": p["world_y"],
                                }
                            )
                    for g_i, g in enumerate(gt_objects):
                        if g_i not in matched_gt:
                            sink.append(
                                {
                                    "split": "val",
                                    "sample_id": row["sample_id"],
                                    "match_status": "fn",
                                    "class_name": str(g["class_name"]),
                                    "gt_class_name": str(g["class_name"]),
                                    "gt_world_x": g["world_x"],
                                    "gt_world_y": g["world_y"],
                                }
                            )
                    for g_i, g in enumerate(gt_objects):
                        distance = float(
                            math.hypot(
                                float(g["world_x"]) - float(cam_c[0]),
                                float(g["world_y"]) - float(cam_c[1]),
                            )
                        )
                        resized = max(
                            float(g.get("bbox_w", 0.0)) * sx, float(g.get("bbox_h", 0.0)) * sy
                        )
                        gt_band_rows[thr].append(
                            {
                                "sample_id": row["sample_id"],
                                "class_name": str(g["class_name"]),
                                "distance_m": distance,
                                "distance_band": band_label(distance, DISTANCE_BANDS),
                                "resized_max_dim_px": resized,
                                "size_band": band_label(resized, SIZE_BANDS),
                                "detected": int(g_i in matched_gt),
                            }
                        )
                processed += 1
            if processed % 800 < int(fused.shape[0]):
                print(f"[eval {checkpoint_path.name}] {processed}/{len(rows)}", flush=True)

    miou, ious, pixel_acc = class_iou_from_confusion(confusion)
    split_ids = {r["sample_id"] for r in rows}
    det_fields = [
        "split", "sample_id", "frame_id", "match_status", "class_name", "pred_class_name",
        "gt_class_name", "score", "global_xy_error_m", "dimension_mae_m", "yaw_error_deg",
        "parked_correct", "branch_stride", "pred_world_x", "pred_world_y", "pred_size_x",
        "pred_size_y", "pred_size_z", "gt_world_x", "gt_world_y", "gt_size_x", "gt_size_y",
        "gt_size_z", "pred_bbox_x0", "pred_bbox_y0", "pred_bbox_x1", "pred_bbox_y1",
        "input_w", "input_h", "gt_center_x", "gt_center_y", "gt_bbox_w", "gt_bbox_h",
        "orig_w", "orig_h",
    ]
    result: Dict[str, object] = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "epoch": int(checkpoint.get("epoch", -1)),
        "split": "val",
        "frames": len(rows),
        "fixed_contract": FIXED_CONTRACT,
        "segmentation": {
            "miou": float(miou),
            "background_iou": float(ious[0]),
            "vehicle_iou": float(ious[1]),
            "person_box_mask_iou": float(ious[2]),
            "person_box_mask_iou_note": (
                "person GT is a filled projected box, not a silhouette; this is box-mask IoU "
                "and must not be compared as silhouette-quality IoU"
            ),
            "pixel_accuracy": float(pixel_acc),
        },
        "by_threshold": {},
    }
    for thr in THRESHOLDS:
        tag = "s020" if thr == 0.20 else "s002"
        with (out_dir / f"detections_{tag}.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=det_fields, extrasaction="ignore")
            writer.writeheader()
            for r in detection_rows[thr]:
                writer.writerow(r)
        with (out_dir / f"gt_bands_{tag}.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "sample_id", "class_name", "distance_m", "distance_band",
                    "resized_max_dim_px", "size_band", "detected",
                ],
            )
            writer.writeheader()
            for r in gt_band_rows[thr]:
                writer.writerow(r)
        bands: Dict[str, Dict[str, Dict[str, float]]] = {"distance": {}, "size": {}}
        for kind, key in (("distance", "distance_band"), ("size", "size_band")):
            agg: Dict[Tuple[str, str], List[int]] = defaultdict(list)
            for r in gt_band_rows[thr]:
                agg[(str(r["class_name"]), str(r[key]))].append(int(r["detected"]))
            for (cls, band), values in sorted(agg.items()):
                bands[kind].setdefault(cls, {})[band] = {
                    "gt": len(values),
                    "detected": int(sum(values)),
                    "recall": float(sum(values)) / max(1, len(values)),
                }
        result["by_threshold"][f"{thr:.2f}"] = {
            "primary_greedy": summarize(
                detection_rows[thr], frame_ids=split_ids, duplicate_radius_m=3.0,
                label="all_val_frames",
            ),
            "recall_by_band": bands,
        }
    (out_dir / "metrics_v2.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoints", required=True, nargs="+", type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    if args.split != "val":
        raise SystemExit("only the validation split is authorized; the test split stays locked")

    config = load_config(str(args.config))
    exp_dir = args.experiment_dir.resolve()
    dataset_dir = exp_dir / "dataset"
    manifest = read_manifest(dataset_dir / "manifest.csv")
    if any(r.get("split") == "test" for r in manifest):
        raise SystemExit("test split present in manifest; refusing to run")
    rows = [r for r in manifest if r.get("split") == args.split]
    object_boxes = load_object_boxes(dataset_dir / "object_boxes.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    first = torch.load(args.checkpoints[0], map_location="cpu", weights_only=False)
    input_size = tuple(int(v) for v in first["input_size"])
    del first
    dataset = EvalDataset(dataset_dir, rows, input_size)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )

    summary = {}
    for checkpoint_path in args.checkpoints:
        checkpoint_path = checkpoint_path.resolve()
        out_dir = args.out_root.resolve() / checkpoint_path.stem
        print(f"[eval] {checkpoint_path} -> {out_dir}", flush=True)
        result = evaluate_checkpoint(
            checkpoint_path, loader, rows, object_boxes, out_dir, config, device
        )
        summary[checkpoint_path.stem] = result
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "all_epochs_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print("[eval] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
