#!/usr/bin/env python3
"""Density-adaptive knob selection -- PER-FRAME offline eval (DENSITY_ADAPTIVE_KNOB_PLAN.md step 2).

Why a new driver instead of `evaluate_fusion`: the packaged eval only emits *aggregate*
`payload_bytes_mean` + a per-object CSV, so payload cannot be joined to a frame's scene density.
This driver emits ONE ROW PER (profile x frame) with:
    payload bytes (post-entropy, zstd)  |  in-view GT count (the density label)
    tp / fp / fn and loc-error sums, split by class
so density binning is a pure post-hoc group-by (the same trick the road-state analysis used).

It reuses the packaged model/codec/matching code verbatim (`split_runtime`, `decode_objects`,
`greedy_match_predictions`, `valid_localization_objects`) and the SAME eval knobs the zstd knob
matrix was built with (score 0.20 / nms 2 / topk 120 / match 5.0 m / max-gt 40 m / zstd-3), so the
numbers are directly comparable to PERMODEL_KNOB_MATRIX_ZSTD.md. Reproducing that matrix's
`noae__uint8__roi0.0` row is the GATE (see gate_density_eval.py).

Speed: the backbone encode + objectness map are computed ONCE per frame and reused across all
(quant x ROI) profiles of that model -- only the AE/codec/head-decode part is per profile.

GT convention: `object_world_x/y` from object_boxes.csv. On THIS dataset (abiodun moving-ego
collector) that column is the bbox-centre-in-world, and train_fusion.py regresses that exact
column, so it is the self-consistent target the model actually learned. See CONVENTION note in
DENSITY_KNOB_RESULTS.md -- the origin-vs-bbox-centre XY delta is measured, not assumed.

Usage (needs PYTHONPATH for the package -- this is an OFFLINE eval, not a CARLA client):
  export PYTHONPATH=$AB/pole_lraspp_multimodal_fusion:$AB:$AB/rl_agent/feature_ae
  python3 rl_agent/density_knob/density_knob_eval.py --models noae,ae32,ae64,ae128 \
      --rois 0.0,0.3,0.5,0.7,0.9,0.98 --quants uint8,uint6,uint4 --out-dir rl_agent/density_knob/raw
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

AB = Path("/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun")
for p in (str(AB / "pole_lraspp_multimodal_fusion"), str(AB), str(AB / "rl_agent" / "feature_ae")):
    if p not in sys.path:
        sys.path.insert(0, p)

from pole_lraspp_multimodal_fusion.common import (  # noqa: E402
    CLASS_NAMES,
    class_iou_from_confusion,
    load_config,
    read_manifest,
    update_confusion,
)
from pole_lraspp_multimodal_fusion.evaluate_fusion import load_fused_tensor, load_mask  # noqa: E402
from pole_lraspp_multimodal_fusion.model import OBJECT_HEAD_CHANNELS, build_multitask_fusion_lraspp  # noqa: E402
from pole_lraspp_multimodal_fusion.object_targets import (  # noqa: E402
    OBJECT_CLASS_NAMES,
    decode_objects,
    greedy_match_predictions,
    load_object_boxes,
    parse_matrix,
    valid_localization_objects,
)
from pole_lraspp_multimodal_fusion.split_runtime import (  # noqa: E402
    MultimodalLRASPPSplitModel,
    deserialize_backbone_features,
    serialize_backbone_features,
)

CFG = AB / "pole_lraspp_multimodal_fusion" / "configs" / "fusion_full_run.yaml"
DS = AB / "fusion_training_data" / "moving_ego_pps200000_merged_8loops_stride2"
CKPT = {
    "noae": AB / "experiments/ae_integrated_20260710/noae_baseline/checkpoints/mprime_joint_noae/best.pt",
    "ae32": AB / "experiments/ae_integrated_20260710/ae32/checkpoints/ae32_integrated/best.pt",
    "ae64": AB / "experiments/ae_integrated_20260710/ae64/checkpoints/ae64_integrated/best.pt",
    "ae128": AB / "experiments/ae_integrated_20260710/ae128/checkpoints/ae128_integrated/best.pt",
}
# Eval knobs: IDENTICAL to run_zstd_full_overnight.sh (so the gate row can reproduce the matrix).
EVAL_KNOBS = dict(object_score_threshold=0.20, object_nms_radius_px=2, topk_objects=120,
                  match_distance_m=5.0, max_gt_distance_m=40.0)
ENTROPY_CODER, ZSTD_LEVEL = "zstd", 3

FIELDS = ["model", "ae_bottleneck", "quant", "roi", "sample_id", "frame_id",
          "n_inview", "n_inview_veh", "n_inview_ped", "payload_bytes", "n_pred",
          "tp", "fp", "fn", "tp_veh", "fp_veh", "fn_veh", "tp_ped", "fp_ped", "fn_ped",
          "loc_err_sum", "loc_err_sq_sum", "loc_err_sum_veh", "loc_err_sum_ped",
          # --- segmentation: per-frame 3x3 confusion (row=GT class, col=pred class), so mIoU and
          # per-class (background/vehicle/person) IoU are a pure post-hoc SUM within each density bin,
          # identical to how PERMODEL_KNOB_MATRIX computes mIoU (class_iou_from_confusion on the sum).
          "conf_00", "conf_01", "conf_02", "conf_10", "conf_11", "conf_12",
          "conf_20", "conf_21", "conf_22"]


def build_model(ckpt_path: Path, device: torch.device, config: dict):
    """Model construction copied from evaluate_fusion.evaluate_checkpoint (lines ~200-293)."""
    train_cfg, fusion_cfg = config["training"], config.get("fusion", {})
    object_cfg = config.get("object_heads", {})
    num_classes = int(train_cfg.get("num_classes", 3))
    checkpoint = torch.load(ckpt_path, map_location=device)
    ckpt = checkpoint if isinstance(checkpoint, dict) else {}
    object_class_names = tuple(ckpt.get("object_class_names") or object_cfg.get("object_classes", OBJECT_CLASS_NAMES))
    input_size = tuple(int(v) for v in (ckpt.get("input_size") or train_cfg.get("input_size", [512, 288])))
    model = build_multitask_fusion_lraspp(
        num_classes=num_classes,
        radar_channels=int(ckpt.get("radar_channels") or fusion_cfg.get("radar_channels", 4)),
        pretrained=False,
        object_channels=int(ckpt.get("object_channels") or object_cfg.get("output_channels", OBJECT_HEAD_CHANNELS)),
        object_hidden_channels=int(object_cfg.get("hidden_channels", 128)),
        fuse_low_into_object_head=bool(ckpt.get("fuse_low_into_object_head")) or bool(object_cfg.get("fuse_low_feature", False)),
        head_arch=str(ckpt.get("object_head_arch") or object_cfg.get("head_arch", "shared")),
        use_coordconv=bool(ckpt.get("object_use_coordconv")) or bool(object_cfg.get("use_coordconv", False)),
        head_depth=int(ckpt.get("object_head_depth") or object_cfg.get("head_depth", 2)),
        predict_bbox2d=bool(ckpt.get("object_predict_bbox2d")) or bool(object_cfg.get("predict_bbox2d", False)),
        use_groundplane_prior=bool(ckpt.get("object_use_groundplane_prior")) or bool(object_cfg.get("use_groundplane_prior", False)),
        groundplane_params=dict(ckpt.get("object_groundplane_params") or object_cfg.get("groundplane_params", {}) or {}),
        device=device,
    ).to(device)
    ae_bn = int((ckpt.get("trial") or {}).get("ae_bottleneck", 0))
    if ae_bn > 0:
        from ae_model import build_ae
        hi = int(model.classifier.cbr[0].in_channels)
        model.feature_ae = build_ae(str((ckpt.get("trial") or {}).get("ae_arch", "v2")), hi, ae_bn).to(device)
    model.load_state_dict(checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint)
    model.eval()
    return model, input_size, object_class_names, ae_bn, bool(ckpt.get("object_predict_bbox2d")) or bool(object_cfg.get("predict_bbox2d", False))


def roi_gate(feats, keep_order_cache, q):
    """Rank-based front-side ROI drop -- byte-for-byte the same rule as evaluate_fusion._roi_gate
    and the training-time model._objectness_drop (drop the k=round(q*N) LOWEST-objectness cells).
    `keep_order_cache[shape] = argsort(pooled objectness)` is precomputed once per frame."""
    if q <= 0:
        return feats
    gated = OrderedDict()
    for name, feat in feats.items():
        order = keep_order_cache[feat.shape[-2:]]
        n = order.numel()
        k = int(round(float(q) * n))
        keep = torch.ones(n, device=feat.device, dtype=feat.dtype)
        if k > 0:
            keep[order[:k]] = 0.0
        gated[name] = feat * keep.reshape(1, 1, feat.shape[-2], feat.shape[-1])
    return gated


def payload_of(serialized) -> int:
    """On-wire payload = entropy-coded size of the quantized buffers (same as evaluate_fusion)."""
    blob = b"".join(b for entry in serialized.values()
                    for b in (entry.values() if isinstance(entry, dict) else [entry])
                    if isinstance(b, (bytes, bytearray)))
    import zstandard as zstd
    return len(zstd.ZstdCompressor(level=ZSTD_LEVEL).compress(blob))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="noae,ae32,ae64,ae128")
    ap.add_argument("--quants", default="uint8,uint6,uint4")
    ap.add_argument("--rois", default="0.0,0.3,0.5,0.7,0.9,0.98")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit-rows", type=int, default=0)
    ap.add_argument("--out-dir", default=str(AB / "rl_agent" / "density_knob" / "raw"))
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(str(CFG))
    object_cfg = config.get("object_heads", {})
    min_gt_area_px = float(object_cfg.get("min_gt_area_px", 24.0))
    num_classes = int(config["training"].get("num_classes", 3))  # seg classes: background/vehicle/person
    seg_class_names = list(CLASS_NAMES[:num_classes])

    rows = [r for r in read_manifest(DS / "manifest.csv") if r.get("split") == a.split]
    if a.limit_rows:
        rows = rows[: a.limit_rows]
    boxes = load_object_boxes(DS / "object_boxes.csv")
    print(f"[density-eval] device={device} split={a.split} frames={len(rows)} "
          f"min_gt_area_px={min_gt_area_px}", flush=True)

    import carla_split_inference_udp_data_collect as od_collect

    quants = [f"per_channel_{q}" for q in a.quants.split(",") if q]
    rois = [float(r) for r in a.rois.split(",") if r]

    for model_name in [m for m in a.models.split(",") if m]:
        ck = CKPT[model_name]
        out_csv = out_dir / f"perframe_{model_name}.csv"
        if out_csv.exists():
            print(f"[density-eval] skip {model_name} (exists: {out_csv.name})", flush=True)
            continue
        t_model = time.time()
        model, input_size, object_class_names, ae_bn, predict_bbox2d = build_model(ck, device, config)
        split = MultimodalLRASPPSplitModel(model, device, input_size=input_size)
        out_hw = (int(input_size[1]), int(input_size[0]))
        n_heat = int(getattr(model, "heatmap_channels", len(object_class_names)))
        ae = getattr(model, "feature_ae", None)
        print(f"[density-eval] {model_name}: input={input_size} ae_bottleneck={ae_bn} "
              f"profiles={len(quants)*len(rois)}", flush=True)

        # one TransportConfig + codec cache per profile (quantization mode is baked into the codec)
        profiles = []
        for q in quants:
            for r in rois:
                profiles.append({
                    "quant": q, "roi": r,
                    "transport": od_collect.TransportConfig(
                        quantization_mode=q, entropy_coder_name=ENTROPY_CODER, zstd_level=ZSTD_LEVEL,
                        roi_objectness_threshold=0.0, bypass_rcnn_transform=False),
                    "codecs": {},
                })

        tmp_csv = out_csv.with_suffix(".partial")
        fh = tmp_csv.open("w", newline="")
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        with torch.inference_mode():
            for i, row in enumerate(rows):
                fused, output_hw, original_size = load_fused_tensor(row, DS, input_size, device)
                matrix = parse_matrix(row.get("camera_matrix_json", ""))
                if matrix is None:
                    continue
                gt_objects = valid_localization_objects(
                    boxes.get(row["sample_id"], []),
                    image_width=int(original_size[0]), image_height=int(original_size[1]),
                    min_area_px=min_gt_area_px, object_class_names=object_class_names,
                    max_distance_m=EVAL_KNOBS["max_gt_distance_m"])
                n_veh = sum(1 for o in gt_objects if o["class_name"] == "vehicle")
                n_ped = len(gt_objects) - n_veh
                cam_c = np.asarray(matrix)[:3, 3]

                # ---- shared per-frame work: backbone encode + objectness ranking + GT seg mask ----
                feats = split.encode(fused)
                obj_maps = split.decode_object_maps(feats, out_hw)
                objness = torch.sigmoid(obj_maps[:, :n_heat]).amax(dim=1, keepdim=True)
                keep_order = {}
                for feat in feats.values():
                    if feat.shape[-2:] not in keep_order:
                        pooled = F.adaptive_max_pool2d(objness, feat.shape[-2:]).reshape(-1).float()
                        keep_order[feat.shape[-2:]] = pooled.argsort()
                gt_mask = load_mask(DS / row["mask_path"])  # (H, W) at original resolution, once per frame

                for prof in profiles:
                    gated = roi_gate(feats, keep_order, prof["roi"])
                    if ae is not None:
                        gated = OrderedDict((k, (ae.encode(v) if k == "high" else v)) for k, v in gated.items())
                    serialized, _ = serialize_backbone_features(gated, prof["transport"], prof["codecs"])
                    nbytes = payload_of(serialized)
                    feats_rt = deserialize_backbone_features(serialized, device=device,
                                                             transport=prof["transport"],
                                                             feature_codecs=prof["codecs"])
                    if ae is not None:
                        feats_rt = OrderedDict((k, (ae.decode(v) if k == "high" else v)) for k, v in feats_rt.items())
                    outputs = split.decode_outputs(feats_rt, out_hw)

                    # ---- segmentation: same recipe as evaluate_fusion (interp to full res, argmax,
                    # confusion vs GT mask). outputs["out"] was already produced above, just unused. ----
                    seg_logits = F.interpolate(outputs["out"], size=output_hw, mode="bilinear",
                                               align_corners=False)
                    seg_pred = seg_logits.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.int64)
                    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
                    update_confusion(conf, seg_pred, gt_mask, num_classes)

                    preds = decode_objects(
                        outputs["object"], camera_matrix=matrix,
                        topk=EVAL_KNOBS["topk_objects"],
                        score_threshold=EVAL_KNOBS["object_score_threshold"],
                        nms_radius_px=EVAL_KNOBS["object_nms_radius_px"],
                        object_class_names=object_class_names,
                        predict_bbox2d=predict_bbox2d)
                    preds = [p for p in preds
                             if math.hypot(float(p["world_x"]) - cam_c[0],
                                           float(p["world_y"]) - cam_c[1]) <= EVAL_KNOBS["max_gt_distance_m"]]
                    matches = greedy_match_predictions(preds, gt_objects,
                                                       max_distance_m=EVAL_KNOBS["match_distance_m"],
                                                       class_aware=True)
                    rec = {"model": model_name, "ae_bottleneck": ae_bn, "quant": prof["quant"],
                           "roi": prof["roi"], "sample_id": row["sample_id"],
                           "frame_id": row.get("frame_id", ""), "n_inview": len(gt_objects),
                           "n_inview_veh": n_veh, "n_inview_ped": n_ped,
                           "payload_bytes": nbytes, "n_pred": len(preds),
                           "tp": len(matches), "fp": max(0, len(preds) - len(matches)),
                           "fn": max(0, len(gt_objects) - len(matches)),
                           "tp_veh": 0, "fp_veh": 0, "fn_veh": 0, "tp_ped": 0, "fp_ped": 0, "fn_ped": 0,
                           "loc_err_sum": 0.0, "loc_err_sq_sum": 0.0,
                           "loc_err_sum_veh": 0.0, "loc_err_sum_ped": 0.0}
                    for ci in range(num_classes):
                        for cj in range(num_classes):
                            rec[f"conf_{ci}{cj}"] = int(conf[ci, cj])
                    mp, mg = set(), set()
                    for pi, gi, dist in matches:
                        mp.add(pi); mg.add(gi)
                        cls = "veh" if gt_objects[gi]["class_name"] == "vehicle" else "ped"
                        rec[f"tp_{cls}"] += 1
                        rec["loc_err_sum"] += float(dist)
                        rec["loc_err_sq_sum"] += float(dist) ** 2
                        rec[f"loc_err_sum_{cls}"] += float(dist)
                    for pi, p in enumerate(preds):
                        if pi not in mp:
                            rec["fp_veh" if p.get("class_name") == "vehicle" else "fp_ped"] += 1
                    for gi, g in enumerate(gt_objects):
                        if gi not in mg:
                            rec["fn_veh" if g["class_name"] == "vehicle" else "fn_ped"] += 1
                    w.writerow(rec)

                if (i + 1) % 200 == 0:
                    el = time.time() - t_model
                    print(f"  [{model_name}] {i+1}/{len(rows)} frames  {el/60:.1f} min  "
                          f"eta {el/(i+1)*(len(rows)-i-1)/60:.1f} min", flush=True)
                    fh.flush()
        fh.close()
        tmp_csv.rename(out_csv)
        print(f"[density-eval] {model_name} DONE in {(time.time()-t_model)/60:.1f} min -> {out_csv}", flush=True)
        del model, split, ae
        torch.cuda.empty_cache()

    (out_dir / "eval_settings.json").write_text(json.dumps({
        "eval_knobs": EVAL_KNOBS, "entropy_coder": ENTROPY_CODER, "zstd_level": ZSTD_LEVEL,
        "min_gt_area_px": min_gt_area_px, "dataset": str(DS), "split": a.split,
        "models": a.models, "quants": a.quants, "rois": a.rois,
        "gt_convention": "object_world_x/y (bbox-centre-in-world) == the column train_fusion.py regresses",
        "seg_num_classes": num_classes, "seg_class_names": seg_class_names,
        "seg_recipe": "outputs['out'] -> interp full-res -> argmax -> update_confusion vs GT mask "
                      "(identical to evaluate_fusion); per-frame 3x3 confusion summed per density bin",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
