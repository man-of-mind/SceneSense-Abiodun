#!/usr/bin/env python3
"""Corrected CenterNet-only evaluator (v1).  Create-only; the legacy evaluator is untouched.

Identical to the fixed Route B contract in every respect except the object decoder:

  legacy     bilinearly enlarge the 14 object maps to 432x768 -> global top-k=120
             -> 2 px occupancy suppression AFTER top-k
  corrected  3x3 per-class local-maximum suppression on the native 108x192 grid
             -> global top-k=120 AFTER local maxima -> read score and every regression
             channel from the interpolated map at the arg-max full-resolution pixel
             inside that peak's own 4x4 block (the trained read location; this
             checkpoint's regression was supervised at full-resolution target pixels)

Everything else is byte-identical to the legacy path: score thresholds 0.20 / 0.02,
match distance 3.0 m, max GT distance 40.0 m, min GT area 12 px, class-aware matching,
greedy-distance matching as the primary metric, validation split, model weights.
Per-detection rows use the legacy schema and the reported metrics are produced by the
legacy ``summarize`` function verbatim, so old and corrected numbers are comparable.

Also emits, as clearly-labelled secondary columns, the maximum-cardinality bipartite
matching result.  The primary metric remains greedy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PILOT_ROOT = HERE.parent
PKG_ROOT = PILOT_ROOT.parent
for path in (HERE, PKG_ROOT, PKG_ROOT.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from centernet_model_v1 import install  # noqa: E402

install()

from object_head_pilot_v1.evaluate_route_b_checkpoint_v1 import FIXED_DECODER, summarize  # noqa: E402
from pole_lraspp_multimodal_fusion.common import (  # noqa: E402
    class_iou_from_confusion,
    load_config,
    read_manifest,
    update_confusion,
)
from pole_lraspp_multimodal_fusion.evaluate_fusion import (  # noqa: E402
    load_fused_tensor,
    load_mask,
    yaw_error_deg,
)
from pole_lraspp_multimodal_fusion.object_targets import (  # noqa: E402
    OBJECT_CLASS_NAMES,
    greedy_match_predictions,
    load_object_boxes,
    parse_matrix,
    valid_localization_objects,
)

from audit_decoder_contract_v1 import build_model, range_gate, sha256  # noqa: E402
from native_decode_v1 import decode_objects_hybrid, max_cardinality_match  # noqa: E402

THRESHOLDS = (0.20, 0.02)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit-rows", type=int, default=0)
    args = parser.parse_args()
    if args.split != "val":
        raise SystemExit("only the validation split is authorized; the test split stays locked")

    out_dir = args.out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite evaluation {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(str(args.config))
    object_cfg = config.get("object_heads", {})
    eval_cfg = dict(config.get("evaluation", {}))
    topk = int(eval_cfg.get("topk_objects", 120))
    match_distance = float(eval_cfg.get("match_distance_m", 3.0))
    min_gt_area = float(object_cfg.get("min_gt_area_px", 12.0))
    max_gt_distance = 40.0

    exp_dir = args.experiment_dir.resolve()
    dataset_dir = exp_dir / "dataset"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model, num_classes = build_model(config, checkpoint, device)
    input_size = tuple(int(v) for v in checkpoint["input_size"])
    class_names = tuple(checkpoint.get("object_class_names") or OBJECT_CLASS_NAMES)
    predict_bbox2d = bool(checkpoint.get("object_predict_bbox2d"))

    manifest = read_manifest(dataset_dir / "manifest.csv")
    rows = [r for r in manifest if r.get("split") == args.split]
    assert not any(r.get("split") == "test" for r in manifest), "test split must be absent"
    if int(args.limit_rows or 0) > 0:
        rows = rows[: int(args.limit_rows)]
    object_boxes = load_object_boxes(dataset_dir / "object_boxes.csv")
    print(f"[corrected] frames={len(rows)}", flush=True)

    confusion = np.zeros((num_classes, num_classes), dtype=np.float64)
    detection_rows: Dict[float, List[Dict[str, object]]] = {t: [] for t in THRESHOLDS}
    bipartite_tp: Dict[float, int] = {t: 0 for t in THRESHOLDS}
    greedy_tp: Dict[float, int] = {t: 0 for t in THRESHOLDS}

    with torch.inference_mode():
        for f_i, row in enumerate(rows):
            fused, output_hw, original_size = load_fused_tensor(row, dataset_dir, input_size, device)
            ow, oh = int(original_size[0]), int(original_size[1])
            iw, ih = int(input_size[0]), int(input_size[1])
            bundle = model.encode_front(fused[:, :3], fused[:, 3 : 3 + int(model.radar_channels)])
            primary = model.object_head(bundle["rgb_p2"])
            native = primary + model.refinement_head(
                bundle["rgb_p2"], bundle["radar_p2"], primary
            )
            full = F.interpolate(native, size=(ih, iw), mode="bilinear", align_corners=False)
            seg = model.classifier(
                model.fusion_projection(torch.cat((bundle["rgb_p2"], bundle["radar_p2"]), dim=1))
            )
            seg_logits = F.interpolate(seg, size=output_hw, mode="bilinear", align_corners=False)
            update_confusion(
                confusion,
                seg_logits.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.int64),
                load_mask(dataset_dir / row["mask_path"]),
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
                all_preds, _ = decode_objects_hybrid(
                    native,
                    full,
                    camera_matrix=matrix,
                    topk=topk,
                    score_threshold=min(THRESHOLDS),
                    object_class_names=class_names,
                    predict_bbox2d=predict_bbox2d,
                )

            for thr in THRESHOLDS:
                preds = (
                    range_gate([p for p in all_preds if p["score"] >= thr], matrix, max_gt_distance)
                    if matrix is not None
                    else []
                )
                matches = greedy_match_predictions(
                    preds, gt_objects, max_distance_m=match_distance, class_aware=True
                )
                greedy_tp[thr] += len(matches)
                bipartite_tp[thr] += len(
                    max_cardinality_match(
                        preds, gt_objects, max_distance_m=match_distance, class_aware=True
                    )
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
                    pred_parked = float(p["parked_score"]) >= 0.5
                    gt_parked = float(g["parked"]) >= 0.5
                    sink.append(
                        {
                            "split": args.split,
                            "sample_id": row["sample_id"],
                            "frame_id": row.get("frame_id", ""),
                            "traffic_light_id": row.get("traffic_light_id", ""),
                            "match_status": "tp",
                            "class_name": str(g["class_name"]),
                            "pred_class_name": p.get("class_name", ""),
                            "gt_class_name": str(g["class_name"]),
                            "score": p["score"],
                            "global_xy_error_m": dist,
                            "dimension_mae_m": dim_err,
                            "yaw_error_deg": yaw_error_deg(p, g),
                            "parked_correct": int(pred_parked == gt_parked),
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
                                "split": args.split,
                                "sample_id": row["sample_id"],
                                "match_status": "fp",
                                "class_name": str(p.get("class_name", "object")),
                                "pred_class_name": str(p.get("class_name", "object")),
                                "score": p["score"],
                                "pred_world_x": p["world_x"],
                                "pred_world_y": p["world_y"],
                            }
                        )
                for g_i, g in enumerate(gt_objects):
                    if g_i not in matched_gt:
                        sink.append(
                            {
                                "split": args.split,
                                "sample_id": row["sample_id"],
                                "match_status": "fn",
                                "class_name": str(g["class_name"]),
                                "gt_class_name": str(g["class_name"]),
                                "gt_world_x": g["world_x"],
                                "gt_world_y": g["world_y"],
                            }
                        )
            if (f_i + 1) % 500 == 0:
                print(f"[corrected] {f_i + 1}/{len(rows)}", flush=True)

    miou, ious, pixel_acc = class_iou_from_confusion(confusion)
    split_ids = {r["sample_id"] for r in rows}
    fields = [
        "split", "sample_id", "frame_id", "traffic_light_id", "match_status", "class_name",
        "pred_class_name", "gt_class_name", "score", "global_xy_error_m", "dimension_mae_m",
        "yaw_error_deg", "parked_correct", "pred_world_x", "pred_world_y", "pred_size_x",
        "pred_size_y", "pred_size_z", "gt_world_x", "gt_world_y", "gt_size_x", "gt_size_y",
        "gt_size_z", "pred_bbox_x0", "pred_bbox_y0", "pred_bbox_x1", "pred_bbox_y1",
        "input_w", "input_h", "gt_center_x", "gt_center_y", "gt_bbox_w", "gt_bbox_h",
        "orig_w", "orig_h",
    ]
    out: Dict[str, object] = {
        "decoder": "centernet_clean_v1_corrected_native_localmax_before_topk",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint.resolve()),
        "config": str(args.config.resolve()),
        "split": args.split,
        "frames": len(rows),
        "segmentation": {
            "miou": float(miou),
            "background_iou": float(ious[0]),
            "vehicle_iou": float(ious[1]),
            "person_iou": float(ious[2]),
            "pixel_accuracy": float(pixel_acc),
        },
        "by_threshold": {},
    }
    for thr in THRESHOLDS:
        tag = "s020" if thr == 0.20 else "s002"
        path = out_dir / f"detections_{tag}.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in detection_rows[thr]:
                w.writerow(r)
        out["by_threshold"][f"{thr:.2f}"] = {
            "fixed_decoder": {
                **FIXED_DECODER,
                "object_score_threshold": float(thr),
                "min_gt_area_px": 12.0,
                "class_aware": True,
                "decode_grid": "native_108x192_localmax_before_topk",
                "object_nms_radius_px": "n/a (3x3 native local maxima replace post-hoc suppression)",
            },
            "primary_greedy": summarize(
                detection_rows[thr],
                frame_ids=split_ids,
                duplicate_radius_m=3.0,
                label="all_val_frames",
            ),
            "secondary_bipartite_tp": bipartite_tp[thr],
            "primary_greedy_tp_crosscheck": greedy_tp[thr],
        }
    (out_dir / "derived_metrics_corrected.json").write_text(
        json.dumps(out, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(json.dumps(out, indent=2, sort_keys=True, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
