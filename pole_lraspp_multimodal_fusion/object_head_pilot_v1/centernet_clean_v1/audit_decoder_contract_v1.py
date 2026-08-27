#!/usr/bin/env python3
"""CenterNet evaluation-contract audit (create-only, CenterNet-clean-v1 scope).

One inference pass over the Route B validation view.  For every frame it computes, in
parallel and from the same forward pass:

  Phase 2  current full-resolution decoder vs native stride-4 local-max decoder,
           plus the top-k saturation / interpolated-duplicate budget accounting;
  Phase 3  greedy-distance matching vs maximum-cardinality bipartite matching;
  Phase 4  heatmap-target vs evaluation-GT parity, native-cell collisions, regression
           overwrites, resized GT box geometry, distance/size bands;
  Phase 5  vehicle semantic-tag pixel support inside each projected GT box, and a
           deterministic stratified sample of score-0.02 person false negatives.

Nothing is trained.  No model weights, thresholds, gates or splits are changed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
PILOT_ROOT = HERE.parent
PKG_ROOT = PILOT_ROOT.parent
for path in (HERE, PKG_ROOT, PKG_ROOT.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from centernet_model_v1 import install  # noqa: E402

install()

from pole_lraspp_multimodal_fusion.common import (  # noqa: E402
    CLASS_NAMES,
    PERSON_TAGS,
    VEHICLE_TAGS,
    class_iou_from_confusion,
    load_config,
    read_manifest,
    update_confusion,
)
from pole_lraspp_multimodal_fusion.evaluate_fusion import (  # noqa: E402
    load_fused_tensor,
    load_mask,
)
from pole_lraspp_multimodal_fusion.model import (  # noqa: E402
    OBJECT_HEAD_CHANNELS,
    build_multitask_fusion_lraspp,
)
from pole_lraspp_multimodal_fusion.object_targets import (  # noqa: E402
    OBJECT_CLASS_NAMES,
    decode_objects,
    greedy_match_predictions,
    load_object_boxes,
    object_reg_channels,
    parse_matrix,
    valid_localization_objects,
)

from native_decode_v1 import (  # noqa: E402
    decode_objects_hybrid,
    decode_objects_native,
    local_maxima_mask,
    max_cardinality_match,
    native_object_maps,
)

STRIDE = 4
THRESHOLDS = (0.20, 0.02)
DISTANCE_BANDS = ((0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 40.0))
SIZE_BANDS = ((0.0, 8.0), (8.0, 16.0), (16.0, 32.0), (32.0, 64.0), (64.0, 1e9))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def band_label(value: float, bands) -> str:
    for lo, hi in bands:
        if lo <= value < hi:
            return f"{lo:g}-{hi:g}" if hi < 1e8 else f"{lo:g}+"
    return "out"


def build_model(config: dict, checkpoint: dict, device: torch.device):
    train_cfg = config["training"]
    object_cfg = config.get("object_heads", {})
    fusion_cfg = config.get("fusion", {})
    num_classes = int(train_cfg.get("num_classes", 3))
    model = build_multitask_fusion_lraspp(
        num_classes=num_classes,
        radar_channels=int(checkpoint.get("radar_channels") or fusion_cfg.get("radar_channels", 4)),
        pretrained=False,
        object_channels=int(checkpoint.get("object_channels") or OBJECT_HEAD_CHANNELS),
        object_hidden_channels=int(object_cfg.get("hidden_channels", 128)),
        fuse_low_into_object_head=bool(checkpoint.get("fuse_low_into_object_head")),
        head_arch=str(checkpoint.get("object_head_arch") or object_cfg.get("head_arch", "shared")),
        use_coordconv=bool(checkpoint.get("object_use_coordconv")),
        head_depth=int(checkpoint.get("object_head_depth") or object_cfg.get("head_depth", 2)),
        predict_bbox2d=bool(checkpoint.get("object_predict_bbox2d")),
        use_groundplane_prior=bool(checkpoint.get("object_use_groundplane_prior")),
        groundplane_params=dict(checkpoint.get("object_groundplane_params") or {}),
        device=device,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, num_classes


def instrument_current_topk(
    object_full: torch.Tensor,
    *,
    heatmap_channels: int,
    topk: int,
    score_threshold: float,
    nms_radius_px: int,
) -> Dict[str, object]:
    """Replay the legacy global-top-k / post-hoc-suppression order and account for it.

    Byte-for-byte the same traversal as ``object_targets.decode_objects`` so the survivor
    count is exactly the legacy prediction count; the extra bookkeeping is the audit.
    """
    center = torch.sigmoid(object_full[:heatmap_channels]).detach().float().cpu()
    height, width = int(center.shape[1]), int(center.shape[2])
    flat = center.reshape(-1)
    k = min(int(topk), int(flat.numel()))
    scores, indices = torch.topk(flat, k=k)
    occupied = np.zeros((heatmap_channels, height, width), dtype=bool)
    per_class = {
        c: {
            "raw_above_thr": 0,
            "distinct_native_cells": set(),
            "discarded_neighbour_dup": 0,
            "survivors": 0,
            "survivor_native_cells": set(),
        }
        for c in range(heatmap_channels)
    }
    min_score_in_topk = float(scores[-1].item()) if k else 0.0
    for score_t, index_t in zip(scores, indices):
        score = float(score_t.item())
        if score < float(score_threshold):
            continue
        idx = int(index_t.item())
        class_index, rem = divmod(idx, height * width)
        y, x = divmod(rem, width)
        rec = per_class[class_index]
        rec["raw_above_thr"] += 1
        rec["distinct_native_cells"].add((y // STRIDE, x // STRIDE))
        y0, y1 = max(0, y - int(nms_radius_px)), min(height, y + int(nms_radius_px) + 1)
        x0, x1 = max(0, x - int(nms_radius_px)), min(width, x + int(nms_radius_px) + 1)
        if occupied[class_index, y0:y1, x0:x1].any():
            rec["discarded_neighbour_dup"] += 1
            continue
        occupied[class_index, y0:y1, x0:x1] = True
        rec["survivors"] += 1
        rec["survivor_native_cells"].add((y // STRIDE, x // STRIDE))
    out: Dict[str, object] = {
        "topk_budget": int(k),
        "topk_saturated": int(min_score_in_topk >= float(score_threshold)),
        "topk_min_score": min_score_in_topk,
    }
    for c, rec in per_class.items():
        out[f"c{c}_raw_above_thr"] = int(rec["raw_above_thr"])
        out[f"c{c}_distinct_native_cells"] = int(len(rec["distinct_native_cells"]))
        out[f"c{c}_discarded_neighbour_dup"] = int(rec["discarded_neighbour_dup"])
        out[f"c{c}_survivors"] = int(rec["survivors"])
        out[f"c{c}_survivor_native_cells"] = int(len(rec["survivor_native_cells"]))
    return out


def range_gate(predictions, camera_matrix, max_distance_m: float):
    cam_c = np.asarray(camera_matrix)[:3, 3]
    return [
        p
        for p in predictions
        if math.hypot(float(p["world_x"]) - cam_c[0], float(p["world_y"]) - cam_c[1])
        <= float(max_distance_m)
    ]


def score_variant(predictions, gt_objects, *, match_distance_m: float, matcher: str):
    fn_matcher = greedy_match_predictions if matcher == "greedy" else max_cardinality_match
    matches = fn_matcher(
        predictions, gt_objects, max_distance_m=float(match_distance_m), class_aware=True
    )
    return matches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--diagnostic-topk-cap", type=int, default=512)
    args = parser.parse_args()
    if args.split != "val":
        raise SystemExit("only the validation split is authorized; the test split stays locked")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(str(args.config))
    object_cfg = config.get("object_heads", {})
    eval_cfg = dict(config.get("evaluation", {}))
    topk = int(eval_cfg.get("topk_objects", 120))
    nms_radius = int(eval_cfg.get("object_nms_radius_px", 2))
    match_distance = float(eval_cfg.get("match_distance_m", 3.0))
    min_gt_area = float(object_cfg.get("min_gt_area_px", 12.0))
    max_gt_distance = float(object_cfg.get("max_gt_distance_m", 40.0) or 40.0)
    if "max_gt_distance_m" not in object_cfg:
        max_gt_distance = 40.0
    max_objects_per_frame = int(object_cfg.get("max_objects_per_frame", 64))
    heatmap_radius_px = int(object_cfg.get("heatmap_radius_px", 4))

    exp_dir = args.experiment_dir.resolve()
    dataset_dir = exp_dir / "dataset"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model, num_classes = build_model(config, checkpoint, device)
    input_size = tuple(int(v) for v in checkpoint["input_size"])  # (w, h)
    class_names = tuple(checkpoint.get("object_class_names") or OBJECT_CLASS_NAMES)
    predict_bbox2d = bool(checkpoint.get("object_predict_bbox2d"))
    heatmap_channels = len(class_names)
    n_cls = heatmap_channels

    rows = [r for r in read_manifest(dataset_dir / "manifest.csv") if r.get("split") == args.split]
    splits_present = Counter(
        r.get("split") for r in read_manifest(dataset_dir / "manifest.csv")
    )
    if int(args.limit_rows or 0) > 0:
        rows = rows[: int(args.limit_rows)]
    object_boxes = load_object_boxes(dataset_dir / "object_boxes.csv")
    print(
        f"[audit] split={args.split} frames={len(rows)} splits_in_manifest={dict(splits_present)}",
        flush=True,
    )

    # ---- accumulators -------------------------------------------------------
    variants = [
        ("current_greedy", "current", "greedy"),
        ("current_bipartite", "current", "bipartite"),
        ("native_greedy", "native", "greedy"),
        ("native_bipartite", "native", "bipartite"),
        ("native_cap512_greedy", "native_cap", "greedy"),
        ("hybrid_greedy", "hybrid", "greedy"),
        ("hybrid_bipartite", "hybrid", "bipartite"),
        ("hybrid_cap512_greedy", "hybrid_cap", "greedy"),
    ]
    stats = {
        (name, thr): {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "n_pred": 0,
            "loc_err": [],
            "per_class": {c: {"tp": 0, "fp": 0, "fn": 0, "loc_err": [], "dim_err": []} for c in range(n_cls)},
        }
        for name, _, _ in variants
        for thr in THRESHOLDS
    }
    greedy_loss_frames = {thr: 0 for thr in THRESHOLDS}
    greedy_loss_objects = {thr: 0 for thr in THRESHOLDS}
    greedy_loss_frames_native = {thr: 0 for thr in THRESHOLDS}
    greedy_loss_objects_native = {thr: 0 for thr in THRESHOLDS}
    greedy_loss_frames_hybrid = {thr: 0 for thr in THRESHOLDS}
    greedy_loss_objects_hybrid = {thr: 0 for thr in THRESHOLDS}

    confusion = np.zeros((num_classes, num_classes), dtype=np.float64)

    frame_rows: List[Dict[str, object]] = []
    grid_rows: List[Dict[str, object]] = []
    object_rows: List[Dict[str, object]] = []
    person_fn_records: List[Dict[str, object]] = []

    sx = None
    sy = None

    with torch.inference_mode():
        for f_i, row in enumerate(rows):
            sample_id = row["sample_id"]
            fused, output_hw, original_size = load_fused_tensor(
                row, dataset_dir, input_size, device
            )
            ow, oh = int(original_size[0]), int(original_size[1])
            iw, ih = int(input_size[0]), int(input_size[1])
            sx, sy = iw / float(ow), ih / float(oh)

            # ONE forward pass: encode_front once, then read native object maps and
            # segmentation off the same feature bundle (exactly decode_tail's algebra).
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
            pred_mask = seg_logits.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.int64)
            gt_mask = load_mask(dataset_dir / row["mask_path"])
            update_confusion(confusion, pred_mask, gt_mask, num_classes)

            matrix = parse_matrix(row.get("camera_matrix_json", ""))
            gt_objects = valid_localization_objects(
                object_boxes.get(sample_id, []),
                image_width=ow,
                image_height=oh,
                min_area_px=min_gt_area,
                object_class_names=class_names,
                max_distance_m=max_gt_distance,
            )
            gt_by_class = Counter(int(g["class_index"]) for g in gt_objects)
            # radar_support_points is dropped by valid_localization_objects; recover it by
            # (label, rounded projected centre), which is unique within a sample here.
            support_lut = {}
            for b in object_boxes.get(sample_id, []):
                try:
                    key = (
                        str(b.get("label")),
                        round(float(b.get("gt_center_x") or 0.0), 2),
                        round(float(b.get("gt_center_y") or 0.0), 2),
                    )
                except ValueError:
                    continue
                support_lut[key] = int(float(b.get("radar_support_points") or 0.0))

            # ---------------- Phase 4: target-grid parity ----------------
            # Replicate build_object_targets placement exactly (full-resolution peaks).
            peak_cells: Dict[Tuple[int, int, int], int] = {}
            native_cells: Dict[Tuple[int, int, int], int] = {}
            native_cells_anyclass: Dict[Tuple[int, int], int] = {}
            reg_pixel_owner: Dict[Tuple[int, int], int] = {}
            reg_overwrites = 0
            peak_collision_same = 0
            targets_placed = 0
            dropped_out_of_range = 0
            subcell_w = {c: 0 for c in range(n_cls)}
            subcell_h = {c: 0 for c in range(n_cls)}
            subcell_either = {c: 0 for c in range(n_cls)}
            ordered = sorted(gt_objects, key=lambda item: float(item.get("area", 0.0)), reverse=True)
            for rank, obj in enumerate(ordered):
                ci = int(obj["class_index"])
                cx = float(obj["center_x"]) * sx
                cy = float(obj["center_y"]) * sy
                ix, iy = int(round(cx)), int(round(cy))
                if ix < 0 or iy < 0 or ix >= iw or iy >= ih:
                    dropped_out_of_range += 1
                    obj["_target_placed"] = False
                    continue
                obj["_target_placed"] = True
                obj["_ix"], obj["_iy"] = ix, iy
                obj["_rank"] = rank
                targets_placed += 1
                key = (ci, iy, ix)
                if key in peak_cells:
                    peak_collision_same += 1
                peak_cells[key] = peak_cells.get(key, 0) + 1
                nkey = (ci, iy // STRIDE, ix // STRIDE)
                native_cells[nkey] = native_cells.get(nkey, 0) + 1
                native_cells_anyclass[(iy // STRIDE, ix // STRIDE)] = (
                    native_cells_anyclass.get((iy // STRIDE, ix // STRIDE), 0) + 1
                )
                # reg_mask is class-agnostic: first writer at a full-res pixel wins.
                if (iy, ix) in reg_pixel_owner:
                    reg_overwrites += 1
                    obj["_reg_written"] = False
                else:
                    reg_pixel_owner[(iy, ix)] = ci
                    obj["_reg_written"] = True
                bw_in = float(obj.get("bbox_w", 0.0)) * sx
                bh_in = float(obj.get("bbox_h", 0.0)) * sy
                obj["_bw_in"], obj["_bh_in"] = bw_in, bh_in
                if bw_in < STRIDE:
                    subcell_w[ci] += 1
                if bh_in < STRIDE:
                    subcell_h[ci] += 1
                if bw_in < STRIDE or bh_in < STRIDE:
                    subcell_either[ci] += 1

            native_collide_same = {c: 0 for c in range(n_cls)}
            for (ci, ny, nx), cnt in native_cells.items():
                if cnt > 1:
                    native_collide_same[ci] += cnt - 1
            native_collide_cross = 0
            for (ny, nx), cnt in native_cells_anyclass.items():
                classes_here = sum(
                    1 for c in range(n_cls) if native_cells.get((c, ny, nx), 0) > 0
                )
                if classes_here > 1:
                    native_collide_cross += 1

            # ---------------- Phase 5: vehicle semantic support ----------------
            tags = np.asarray(Image.open(dataset_dir / row["instance_raw_path"]))
            if tags.ndim == 3:
                tags = tags[:, :, 2]
            veh_tag_mask = np.isin(tags, list(VEHICLE_TAGS))
            person_tag_px = int(np.isin(tags, list(PERSON_TAGS)).sum())

            # ---------------- Phase 2/3: decode + match ----------------
            frame_record: Dict[str, object] = {
                "sample_id": sample_id,
                "split": args.split,
                "eligible_gt_total": len(gt_objects),
            }
            for c, name in enumerate(class_names):
                frame_record[f"eligible_gt_{name}"] = int(gt_by_class.get(c, 0))

            preds_by_variant: Dict[Tuple[str, float], List[Dict[str, float]]] = {}
            if matrix is not None:
                cur_all = decode_objects(
                    full,
                    camera_matrix=matrix,
                    topk=topk,
                    score_threshold=min(THRESHOLDS),
                    nms_radius_px=nms_radius,
                    object_class_names=class_names,
                    predict_bbox2d=predict_bbox2d,
                )
                nat_all, nat_diag = decode_objects_native(
                    native,
                    camera_matrix=matrix,
                    topk=topk,
                    score_threshold=min(THRESHOLDS),
                    input_size=input_size,
                    object_class_names=class_names,
                    predict_bbox2d=predict_bbox2d,
                )
                hyb_all, hyb_diag = decode_objects_hybrid(
                    native,
                    full,
                    camera_matrix=matrix,
                    topk=topk,
                    score_threshold=min(THRESHOLDS),
                    object_class_names=class_names,
                    predict_bbox2d=predict_bbox2d,
                )
                hybcap_all, _ = decode_objects_hybrid(
                    native,
                    full,
                    camera_matrix=matrix,
                    topk=int(args.diagnostic_topk_cap),
                    score_threshold=min(THRESHOLDS),
                    object_class_names=class_names,
                    predict_bbox2d=predict_bbox2d,
                )
                cap_all, cap_diag = decode_objects_native(
                    native,
                    camera_matrix=matrix,
                    topk=int(args.diagnostic_topk_cap),
                    score_threshold=min(THRESHOLDS),
                    input_size=input_size,
                    object_class_names=class_names,
                    predict_bbox2d=predict_bbox2d,
                )
                if f_i < 25:
                    # The legacy decoder walks candidates in descending score and skips
                    # sub-threshold ones before they can occupy, so decoding at 0.02 and
                    # filtering to >=0.20 must equal decoding at 0.20 directly. Verify.
                    direct = decode_objects(
                        full,
                        camera_matrix=matrix,
                        topk=topk,
                        score_threshold=0.20,
                        nms_radius_px=nms_radius,
                        object_class_names=class_names,
                        predict_bbox2d=predict_bbox2d,
                    )
                    filtered = [p for p in cur_all if p["score"] >= 0.20]
                    assert len(direct) == len(filtered) and all(
                        abs(a["score"] - b["score"]) < 1e-12
                        and abs(a["world_x"] - b["world_x"]) < 1e-9
                        for a, b in zip(direct, filtered)
                    ), f"threshold-prefix equivalence failed on {sample_id}"
                for thr in THRESHOLDS:
                    preds_by_variant[("current", thr)] = range_gate(
                        [p for p in cur_all if p["score"] >= thr], matrix, max_gt_distance
                    )
                    preds_by_variant[("native", thr)] = range_gate(
                        [p for p in nat_all if p["score"] >= thr], matrix, max_gt_distance
                    )
                    preds_by_variant[("native_cap", thr)] = range_gate(
                        [p for p in cap_all if p["score"] >= thr], matrix, max_gt_distance
                    )
                    preds_by_variant[("hybrid", thr)] = range_gate(
                        [p for p in hyb_all if p["score"] >= thr], matrix, max_gt_distance
                    )
                    preds_by_variant[("hybrid_cap", thr)] = range_gate(
                        [p for p in hybcap_all if p["score"] >= thr], matrix, max_gt_distance
                    )
                for thr in THRESHOLDS:
                    tag = "020" if thr == 0.20 else "002"
                    instr = instrument_current_topk(
                        full[0] if full.ndim == 4 else full,
                        heatmap_channels=heatmap_channels,
                        topk=topk,
                        score_threshold=thr,
                        nms_radius_px=nms_radius,
                    )
                    frame_record[f"cur_topk_saturated_{tag}"] = instr["topk_saturated"]
                    frame_record[f"cur_topk_min_score_{tag}"] = round(float(instr["topk_min_score"]), 6)
                    for c, name in enumerate(class_names):
                        for field in (
                            "raw_above_thr",
                            "distinct_native_cells",
                            "discarded_neighbour_dup",
                            "survivors",
                            "survivor_native_cells",
                        ):
                            frame_record[f"cur_{name}_{field}_{tag}"] = instr[f"c{c}_{field}"]
                        frame_record[f"native_localmax_{name}_{tag}"] = nat_diag[
                            f"native_localmax_c{c}_ge{tag}"
                        ]
                    frame_record[f"native_topk_saturated_{tag}"] = nat_diag["native_topk_saturated"]
                    frame_record[f"native_cap_topk_saturated_{tag}"] = cap_diag[
                        "native_topk_saturated"
                    ]
                    frame_record[f"hybrid_topk_saturated_{tag}"] = hyb_diag[
                        "hybrid_topk_saturated"
                    ]
            else:
                for thr in THRESHOLDS:
                    for kind in ("current", "native", "native_cap", "hybrid", "hybrid_cap"):
                        preds_by_variant[(kind, thr)] = []

            for vname, kind, matcher in variants:
                for thr in THRESHOLDS:
                    preds = preds_by_variant[(kind, thr)]
                    matches = score_variant(
                        preds, gt_objects, match_distance_m=match_distance, matcher=matcher
                    )
                    st = stats[(vname, thr)]
                    st["tp"] += len(matches)
                    st["fp"] += max(0, len(preds) - len(matches))
                    st["fn"] += max(0, len(gt_objects) - len(matches))
                    st["n_pred"] += len(preds)
                    mgt = set()
                    for p_i, g_i, d in matches:
                        st["loc_err"].append(d)
                        ci = int(gt_objects[g_i]["class_index"])
                        st["per_class"][ci]["tp"] += 1
                        st["per_class"][ci]["loc_err"].append(d)
                        st["per_class"][ci]["dim_err"].append(
                            float(
                                np.mean(
                                    np.abs(
                                        np.array(
                                            [
                                                preds[p_i]["size_x"] - gt_objects[g_i]["size_x"],
                                                preds[p_i]["size_y"] - gt_objects[g_i]["size_y"],
                                                preds[p_i]["size_z"] - gt_objects[g_i]["size_z"],
                                            ]
                                        )
                                    )
                                )
                            )
                        )
                        mgt.add(g_i)
                    for g_i, g in enumerate(gt_objects):
                        if g_i not in mgt:
                            st["per_class"][int(g["class_index"])]["fn"] += 1
                    mp = {p_i for p_i, _, _ in matches}
                    for p_i, p in enumerate(preds):
                        if p_i not in mp:
                            st["per_class"][int(p["class_index"])]["fp"] += 1
                    frame_record[f"{vname}_tp_{'020' if thr==0.20 else '002'}"] = len(matches)
                    frame_record[f"{vname}_npred_{'020' if thr==0.20 else '002'}"] = len(preds)
                    if vname == "current_greedy":
                        frame_record[f"_cg_matched_{thr}"] = mgt
                    if vname == "native_greedy":
                        frame_record[f"_ng_matched_{thr}"] = mgt
                    if vname == "hybrid_greedy":
                        frame_record[f"_hg_matched_{thr}"] = mgt

            for thr in THRESHOLDS:
                g = frame_record[f"current_greedy_tp_{'020' if thr==0.20 else '002'}"]
                b = frame_record[f"current_bipartite_tp_{'020' if thr==0.20 else '002'}"]
                if b > g:
                    greedy_loss_frames[thr] += 1
                    greedy_loss_objects[thr] += b - g
                gn = frame_record[f"native_greedy_tp_{'020' if thr==0.20 else '002'}"]
                bn = frame_record[f"native_bipartite_tp_{'020' if thr==0.20 else '002'}"]
                if bn > gn:
                    greedy_loss_frames_native[thr] += 1
                    greedy_loss_objects_native[thr] += bn - gn
                gh = frame_record[f"hybrid_greedy_tp_{'020' if thr==0.20 else '002'}"]
                bh = frame_record[f"hybrid_bipartite_tp_{'020' if thr==0.20 else '002'}"]
                if bh > gh:
                    greedy_loss_frames_hybrid[thr] += 1
                    greedy_loss_objects_hybrid[thr] += bh - gh

            # per-object rows (recall by band / observability / target status)
            cg002 = frame_record.pop("_cg_matched_0.02", set())
            cg020 = frame_record.pop("_cg_matched_0.2", set())
            ng002 = frame_record.pop("_ng_matched_0.02", set())
            ng020 = frame_record.pop("_ng_matched_0.2", set())
            hg002 = frame_record.pop("_hg_matched_0.02", set())
            hg020 = frame_record.pop("_hg_matched_0.2", set())
            for g_i, obj in enumerate(gt_objects):
                ci = int(obj["class_index"])
                cname = str(obj["class_name"])
                bw_in = float(obj.get("bbox_w", 0.0)) * sx
                bh_in = float(obj.get("bbox_h", 0.0)) * sy
                x0 = int(round(float(obj["center_x"]) - float(obj["bbox_w"]) / 2.0))
                y0 = int(round(float(obj["center_y"]) - float(obj["bbox_h"]) / 2.0))
                x1 = int(round(float(obj["center_x"]) + float(obj["bbox_w"]) / 2.0))
                y1 = int(round(float(obj["center_y"]) + float(obj["bbox_h"]) / 2.0))
                x0c, y0c = max(0, x0), max(0, y0)
                x1c, y1c = min(ow, x1), min(oh, y1)
                sem_px = 0
                if cname == "vehicle" and x1c > x0c and y1c > y0c:
                    sem_px = int(veh_tag_mask[y0c:y1c, x0c:x1c].sum())
                cam_c = np.asarray(matrix)[:3, 3] if matrix is not None else np.zeros(3)
                dist_m = float(
                    math.hypot(float(obj["world_x"]) - cam_c[0], float(obj["world_y"]) - cam_c[1])
                )
                rec = {
                    "sample_id": sample_id,
                    "class_name": cname,
                    "gt_index": g_i,
                    "distance_m": round(dist_m, 4),
                    "distance_band": band_label(dist_m, DISTANCE_BANDS),
                    "gt_bbox_w_orig": round(float(obj.get("bbox_w", 0.0)), 3),
                    "gt_bbox_h_orig": round(float(obj.get("bbox_h", 0.0)), 3),
                    "gt_area_orig_px": round(float(obj.get("area", 0.0)), 3),
                    "gt_bbox_w_input": round(bw_in, 4),
                    "gt_bbox_h_input": round(bh_in, 4),
                    "gt_area_input_px": round(bw_in * bh_in, 4),
                    "input_size_band": band_label(max(bw_in, bh_in), SIZE_BANDS),
                    "subcell_w": int(bw_in < STRIDE),
                    "subcell_h": int(bh_in < STRIDE),
                    "subcell_either": int(bw_in < STRIDE or bh_in < STRIDE),
                    "target_placed": int(bool(obj.get("_target_placed", False))),
                    "reg_target_written": int(bool(obj.get("_reg_written", False))),
                    "area_rank": int(obj.get("_rank", -1)),
                    "beyond_max_objects_array": int(int(obj.get("_rank", 0)) >= max_objects_per_frame),
                    "vehicle_semantic_px_in_box": sem_px,
                    "radar_support_points": support_lut.get(
                        (cname, round(float(obj["center_x"]), 2), round(float(obj["center_y"]), 2)), -1
                    ),
                    "tp_current_greedy_002": int(g_i in cg002),
                    "tp_current_greedy_020": int(g_i in cg020),
                    "tp_native_greedy_002": int(g_i in ng002),
                    "tp_native_greedy_020": int(g_i in ng020),
                    "tp_hybrid_greedy_002": int(g_i in hg002),
                    "tp_hybrid_greedy_020": int(g_i in hg020),
                }
                object_rows.append(rec)
                if cname == "person" and g_i not in cg002 and g_i not in hg002:
                    person_fn_records.append(
                        {
                            **rec,
                            "rgb_path": row["rgb_path"],
                            "box_orig": (x0, y0, x1, y1),
                            "center_orig": (float(obj["center_x"]), float(obj["center_y"])),
                            "person_tag_px_frame": person_tag_px,
                        }
                    )

            grid_rows.append(
                {
                    "sample_id": sample_id,
                    "eligible_gt_total": len(gt_objects),
                    **{f"eligible_gt_{n}": int(gt_by_class.get(c, 0)) for c, n in enumerate(class_names)},
                    "targets_placed": targets_placed,
                    "targets_dropped_out_of_input": dropped_out_of_range,
                    "gt_array_capped": int(max(0, len(gt_objects) - max_objects_per_frame)),
                    "fullres_peak_collisions_same_class": peak_collision_same,
                    **{
                        f"native_cell_collisions_same_class_{n}": native_collide_same[c]
                        for c, n in enumerate(class_names)
                    },
                    "native_cells_with_two_classes": native_collide_cross,
                    "reg_target_overwrites": reg_overwrites,
                    **{f"subcell_w_{n}": subcell_w[c] for c, n in enumerate(class_names)},
                    **{f"subcell_h_{n}": subcell_h[c] for c, n in enumerate(class_names)},
                    **{f"subcell_either_{n}": subcell_either[c] for c, n in enumerate(class_names)},
                    "person_semantic_tag_px": person_tag_px,
                }
            )
            frame_rows.append(frame_record)
            if (f_i + 1) % 250 == 0:
                print(f"[audit] {f_i + 1}/{len(rows)} frames", flush=True)

    def write_csv(path: Path, records: List[Dict[str, object]]) -> None:
        if not records:
            path.write_text("", encoding="utf-8")
            return
        keys: List[str] = []
        for r in records:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in records:
                w.writerow(r)

    write_csv(out_dir / "per_frame_decoder_audit.csv", frame_rows)
    write_csv(out_dir / "target_grid_audit.csv", grid_rows)
    write_csv(out_dir / "gt_object_audit.csv", object_rows)

    # ---- summary -------------------------------------------------------------
    def summarize(st) -> Dict[str, object]:
        out = {
            "tp": st["tp"],
            "fp": st["fp"],
            "fn": st["fn"],
            "n_pred": st["n_pred"],
            "precision": st["tp"] / max(1, st["tp"] + st["fp"]),
            "recall": st["tp"] / max(1, st["tp"] + st["fn"]),
            "xy_mae_m": float(np.mean(st["loc_err"])) if st["loc_err"] else None,
        }
        out["f1"] = (
            2 * out["precision"] * out["recall"] / max(1e-12, out["precision"] + out["recall"])
        )
        for c, name in enumerate(class_names):
            pc = st["per_class"][c]
            p = pc["tp"] / max(1, pc["tp"] + pc["fp"])
            r = pc["tp"] / max(1, pc["tp"] + pc["fn"])
            out[f"{name}_tp"] = pc["tp"]
            out[f"{name}_fp"] = pc["fp"]
            out[f"{name}_fn"] = pc["fn"]
            out[f"{name}_precision"] = p
            out[f"{name}_recall"] = r
            out[f"{name}_f1"] = 2 * p * r / max(1e-12, p + r)
            out[f"{name}_xy_mae_m"] = float(np.mean(pc["loc_err"])) if pc["loc_err"] else None
            out[f"{name}_dimension_mae_m"] = (
                float(np.mean(pc["dim_err"])) if pc["dim_err"] else None
            )
        return out

    miou_value, ious, pixel_acc = class_iou_from_confusion(confusion)
    summary: Dict[str, object] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint.resolve()),
        "config": str(args.config.resolve()),
        "split": args.split,
        "frames": len(rows),
        "manifest_split_counts": dict(splits_present),
        "input_size_wh": list(input_size),
        "native_grid_hw": [input_size[1] // STRIDE, input_size[0] // STRIDE],
        "fixed_contract": {
            "topk_objects": topk,
            "object_nms_radius_px": nms_radius,
            "match_distance_m": match_distance,
            "max_gt_distance_m": max_gt_distance,
            "min_gt_area_px": min_gt_area,
            "class_aware": True,
            "heatmap_radius_px": heatmap_radius_px,
            "max_objects_per_frame": max_objects_per_frame,
        },
        "segmentation": {
            "miou": float(miou_value),
            "pixel_accuracy": float(pixel_acc),
            **{f"{n}_iou": float(ious[i]) for i, n in enumerate(CLASS_NAMES[:num_classes])},
        },
        "greedy_vs_bipartite": {
            f"{thr}": {
                "current_frames_greedy_loses": greedy_loss_frames[thr],
                "current_objects_greedy_loses": greedy_loss_objects[thr],
                "native_frames_greedy_loses": greedy_loss_frames_native[thr],
                "native_objects_greedy_loses": greedy_loss_objects_native[thr],
                "hybrid_frames_greedy_loses": greedy_loss_frames_hybrid[thr],
                "hybrid_objects_greedy_loses": greedy_loss_objects_hybrid[thr],
            }
            for thr in THRESHOLDS
        },
        "variants": {
            f"{vname}@{thr}": summarize(stats[(vname, thr)])
            for vname, _, _ in variants
            for thr in THRESHOLDS
        },
    }
    (out_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    with (out_dir / "person_fn_records.json").open("w", encoding="utf-8") as fh:
        json.dump(person_fn_records, fh, indent=1, default=str)
    print(json.dumps(summary["variants"], indent=2, default=str), flush=True)
    print(json.dumps(summary["greedy_vs_bipartite"], indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
