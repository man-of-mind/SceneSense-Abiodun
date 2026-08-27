#!/usr/bin/env python3
"""Bounded false-negative decomposition for CenterNet v2 (Route B, val 3,588 frames).

Stage `dump`  : one authorized validation inference pass with the FROZEN v2 native
                decoder at score 0.02.  Writes every native local maximum above 0.02
                (pre-matching) plus every eligible GT object with its context fields.
Stage `taxon` : pure-CSV taxonomy / breakdowns / transitions / contact sheet.

Nothing in centernet_clean_v2/ is imported-and-modified: the decoder, the GT
eligibility rule and the greedy matcher are used verbatim.

Fixed contract: score 0.02 (decomposition) and 0.20 (reconciliation only),
local maximum before top-k, top-k 120 per branch, match distance 3.0 m,
max GT distance 40.0 m, min GT area 12 px, class-aware, unchanged GT denominator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ABI = Path(__file__).resolve().parents[3]
V2 = ABI / "pole_lraspp_multimodal_fusion" / "object_head_pilot_v1" / "centernet_clean_v2"
for p in (ABI, ABI / "pole_lraspp_multimodal_fusion", ABI / "pole_lraspp_multimodal_fusion" / "object_head_pilot_v1", V2):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

THRESHOLDS = (0.20, 0.02)
VEHICLE_SEM_CLASS = 1


# --------------------------------------------------------------------------- dump
def stage_dump(args: argparse.Namespace) -> None:
    from torch.utils.data import DataLoader

    from pole_lraspp_multimodal_fusion.common import load_config, read_manifest
    from pole_lraspp_multimodal_fusion.object_targets import (
        greedy_match_predictions,
        load_object_boxes,
        parse_matrix,
        valid_localization_objects,
    )
    from evaluate_v2 import EvalDataset, load_model  # frozen v2 evaluator
    from decode_v2 import TOPK_PER_BRANCH, decode_objects_v2, range_gate  # frozen v2 decoder

    exp = Path(args.experiment_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = load_config(Path(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_dir = (exp / "dataset").resolve()
    # manifest order, unsorted, exactly as evaluate_v2.main builds it
    manifest = read_manifest(dataset_dir / "manifest.csv")
    if any(r.get("split") == "test" for r in manifest):
        raise SystemExit("test split present in manifest; refusing to run")
    rows = [r for r in manifest if r.get("split") == "val"]
    val_ids = {line.strip() for line in (exp / "splits" / "val.txt").read_text().split() if line.strip()}
    assert len(rows) == len(val_ids), f"val rows {len(rows)} != split {len(val_ids)}"
    object_boxes = load_object_boxes(dataset_dir / "object_boxes.csv")

    # raw radar-support counts / actor ids, keyed by (sample_id, center_x, center_y)
    raw_by_sample: dict[str, list[dict]] = {}
    with open(dataset_dir / "object_boxes.csv") as fh:
        for r in csv.DictReader(fh):
            raw_by_sample.setdefault(r["sample_id"], []).append(r)

    ckpt_path = Path(args.checkpoint)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = load_model(checkpoint, device)
    input_size = tuple(int(v) for v in checkpoint["input_size"])
    class_names = tuple(checkpoint.get("object_class_names") or ("vehicle", "person"))
    min_gt_area = float(config.get("object_heads", {}).get("min_gt_area_px", 12.0))
    max_gt_distance = float(config.get("evaluation", {}).get("max_gt_distance_m", 40.0))
    match_distance = float(config.get("evaluation", {}).get("match_distance_m", 3.0))

    loader = DataLoader(
        EvalDataset(dataset_dir, rows, input_size),
        batch_size=int(args.batch_size), shuffle=False,
        num_workers=int(args.num_workers), pin_memory=True,
    )

    peak_fh = open(out / "peaks_s002.csv", "w", newline="")
    gt_fh = open(out / "gt_objects.csv", "w", newline="")
    peak_w = csv.writer(peak_fh)
    gt_w = csv.writer(gt_fh)
    peak_w.writerow([
        "sample_id", "peak_idx", "class_name", "score", "branch_stride", "native_x", "native_y",
        "center_x_in", "center_y_in", "center_x_full", "center_y_full",
        "bbox_w_in", "bbox_h_in", "pred_world_x", "pred_world_y", "pred_world_z",
        "pred_size_x", "pred_size_y", "pred_size_z", "radar_support_score",
        "range_gated_in_s002", "matched_gt_idx_s002", "matched_dist_s002",
        "range_gated_in_s020", "matched_gt_idx_s020", "matched_dist_s020",
    ])
    gt_w.writerow([
        "sample_id", "experiment_id", "traffic_density", "pedestrian_density", "frame_id",
        "gt_idx", "gt_actor_id", "class_name", "gt_world_x", "gt_world_y", "gt_world_z",
        "gt_center_x_full", "gt_center_y_full", "gt_bbox_w_full", "gt_bbox_h_full", "gt_area_full",
        "gt_center_x_res", "gt_center_y_res", "gt_bbox_w_res", "gt_bbox_h_res", "gt_area_res",
        "distance_m", "radar_support_points", "nearer_box_overlap_frac", "veh_semantic_frac",
        "orig_w", "orig_h", "input_w", "input_h",
        "detected_s002", "matched_dist_s002", "detected_s020", "matched_dist_s020",
    ])

    n_saturated = {"vehicle": 0, "person": 0}
    n_peaks = 0
    n_gt = 0
    processed = 0
    with torch.inference_mode():
        for fused, masks, indices, ohs, ows in loader:
            fused = fused.to(device, non_blocking=True)
            outputs = model(fused)
            for b in range(fused.shape[0]):
                index = int(indices[b].item())
                row = rows[index]
                oh, ow = int(ohs[b].item()), int(ows[b].item())
                iw, ih = int(input_size[0]), int(input_size[1])
                sx, sy = float(iw) / max(1.0, float(ow)), float(ih) / max(1.0, float(oh))
                fx, fy = float(ow) / float(iw), float(oh) / float(ih)
                matrix = parse_matrix(row.get("camera_matrix_json", ""))
                gt_objects = valid_localization_objects(
                    object_boxes.get(row["sample_id"], []),
                    image_width=ow, image_height=oh, min_area_px=min_gt_area,
                    object_class_names=class_names, max_distance_m=max_gt_distance,
                )
                all_preds = []
                if matrix is not None:
                    all_preds = decode_objects_v2(
                        outputs, camera_matrix=matrix, input_size=input_size,
                        score_threshold=min(THRESHOLDS), topk=TOPK_PER_BRANCH, sample_index=b,
                    )
                for cn in ("vehicle", "person"):
                    if sum(1 for p in all_preds if p["class_name"] == cn) >= TOPK_PER_BRANCH:
                        n_saturated[cn] += 1

                cam_c = np.asarray(matrix)[:3, 3] if matrix is not None else np.zeros(3)
                pid = {id(p): i for i, p in enumerate(all_preds)}
                pinfo = {i: {} for i in range(len(all_preds))}
                ginfo = {i: {} for i in range(len(gt_objects))}
                for thr in THRESHOLDS:
                    tag = "s020" if thr == 0.20 else "s002"
                    preds = (range_gate([p for p in all_preds if p["score"] >= thr], matrix, max_gt_distance)
                             if matrix is not None else [])
                    kept = [pid[id(p)] for p in preds]
                    for i in kept:
                        pinfo[i][f"gated_{tag}"] = 1
                    matches = greedy_match_predictions(
                        preds, gt_objects, max_distance_m=match_distance, class_aware=True)
                    for p_i, g_i, dist in matches:
                        pinfo[kept[p_i]][f"m_{tag}"] = (g_i, dist)
                        ginfo[g_i][f"m_{tag}"] = dist

                # ---- GT context: nearer-box overlap + vehicle semantic support
                order = sorted(range(len(gt_objects)), key=lambda i: _dist(gt_objects[i], cam_c))
                rank = {g: r for r, g in enumerate(order)}
                mask_np = masks[b].numpy()
                for g_i, g in enumerate(gt_objects):
                    x0 = int(max(0, math.floor(g["center_x"] - g["bbox_w"] / 2.0)))
                    y0 = int(max(0, math.floor(g["center_y"] - g["bbox_h"] / 2.0)))
                    x1 = int(min(ow, math.ceil(g["center_x"] + g["bbox_w"] / 2.0)))
                    y1 = int(min(oh, math.ceil(g["center_y"] + g["bbox_h"] / 2.0)))
                    w, h = max(0, x1 - x0), max(0, y1 - y0)
                    if w <= 0 or h <= 0:
                        ginfo[g_i]["ovl"] = 0.0
                        ginfo[g_i]["sem"] = 0.0
                        continue
                    cov = np.zeros((h, w), dtype=bool)
                    for o_i, o in enumerate(gt_objects):
                        if o_i == g_i or rank[o_i] >= rank[g_i]:
                            continue
                        ox0 = int(max(x0, math.floor(o["center_x"] - o["bbox_w"] / 2.0)))
                        oy0 = int(max(y0, math.floor(o["center_y"] - o["bbox_h"] / 2.0)))
                        ox1 = int(min(x1, math.ceil(o["center_x"] + o["bbox_w"] / 2.0)))
                        oy1 = int(min(y1, math.ceil(o["center_y"] + o["bbox_h"] / 2.0)))
                        if ox1 > ox0 and oy1 > oy0:
                            cov[oy0 - y0:oy1 - y0, ox0 - x0:ox1 - x0] = True
                    ginfo[g_i]["ovl"] = float(cov.mean())
                    sub = mask_np[y0:y1, x0:x1]
                    ginfo[g_i]["sem"] = float((sub == VEHICLE_SEM_CLASS).mean()) if sub.size else 0.0

                # raw radar-support count matched by projected centre
                raw = raw_by_sample.get(row["sample_id"], [])
                for g_i, g in enumerate(gt_objects):
                    best, bd = None, 1e9
                    for r in raw:
                        if r.get("label") != g["class_name"] or r.get("gt_source") != "actor":
                            continue
                        try:
                            d = abs(float(r["gt_center_x"]) - g["center_x"]) + abs(float(r["gt_center_y"]) - g["center_y"])
                        except (ValueError, KeyError):
                            continue
                        if d < bd:
                            best, bd = r, d
                    if best is not None and bd < 1e-6:
                        ginfo[g_i]["rsp"] = best.get("radar_support_points", "")
                        ginfo[g_i]["aid"] = best.get("gt_actor_id", "")
                    else:
                        ginfo[g_i]["rsp"] = ""
                        ginfo[g_i]["aid"] = ""

                for i, p in enumerate(all_preds):
                    m2 = pinfo[i].get("m_s002"); m20 = pinfo[i].get("m_s020")
                    peak_w.writerow([
                        row["sample_id"], i, p["class_name"], f'{p["score"]:.6f}', int(p["branch_stride"]),
                        int(p["native_x"]), int(p["native_y"]),
                        f'{p["center_x_px"]:.4f}', f'{p["center_y_px"]:.4f}',
                        f'{p["center_x_px"] * fx:.4f}', f'{p["center_y_px"] * fy:.4f}',
                        f'{p["bbox_w_px"]:.4f}', f'{p["bbox_h_px"]:.4f}',
                        f'{p["world_x"]:.6f}', f'{p["world_y"]:.6f}', f'{p["world_z"]:.6f}',
                        f'{p["size_x"]:.4f}', f'{p["size_y"]:.4f}', f'{p["size_z"]:.4f}',
                        f'{p["radar_support_score"]:.6f}',
                        pinfo[i].get("gated_s002", 0), "" if m2 is None else m2[0], "" if m2 is None else f"{m2[1]:.6f}",
                        pinfo[i].get("gated_s020", 0), "" if m20 is None else m20[0], "" if m20 is None else f"{m20[1]:.6f}",
                    ])
                n_peaks += len(all_preds)
                for g_i, g in enumerate(gt_objects):
                    d2 = ginfo[g_i].get("m_s002"); d20 = ginfo[g_i].get("m_s020")
                    gt_w.writerow([
                        row["sample_id"], row.get("experiment_id", ""), row.get("traffic_density", ""),
                        row.get("pedestrian_density", ""), row.get("frame_id", ""),
                        g_i, ginfo[g_i]["aid"], g["class_name"],
                        f'{g["world_x"]:.6f}', f'{g["world_y"]:.6f}', f'{g["world_z"]:.6f}',
                        f'{g["center_x"]:.4f}', f'{g["center_y"]:.4f}', f'{g["bbox_w"]:.4f}', f'{g["bbox_h"]:.4f}',
                        f'{g["area"]:.4f}',
                        f'{g["center_x"] * sx:.4f}', f'{g["center_y"] * sy:.4f}',
                        f'{g["bbox_w"] * sx:.4f}', f'{g["bbox_h"] * sy:.4f}', f'{g["bbox_w"] * sx * g["bbox_h"] * sy:.4f}',
                        f'{_dist(g, cam_c):.6f}', ginfo[g_i]["rsp"],
                        f'{ginfo[g_i]["ovl"]:.6f}', f'{ginfo[g_i]["sem"]:.6f}',
                        ow, oh, iw, ih,
                        0 if d2 is None else 1, "" if d2 is None else f"{d2:.6f}",
                        0 if d20 is None else 1, "" if d20 is None else f"{d20:.6f}",
                    ])
                n_gt += len(gt_objects)
                processed += 1
            if processed % 480 < int(args.batch_size):
                print(f"  {processed}/{len(rows)} frames", flush=True)

    peak_fh.close()
    gt_fh.close()
    meta = {
        "checkpoint": str(ckpt_path), "checkpoint_sha256": _sha(ckpt_path),
        "frames": processed, "peaks_above_0.02": n_peaks, "eligible_gt": n_gt,
        "topk_saturated_frames": n_saturated, "topk_per_branch": TOPK_PER_BRANCH,
        "score_thresholds": list(THRESHOLDS), "match_distance_m": match_distance,
        "max_gt_distance_m": max_gt_distance, "min_gt_area_px": min_gt_area,
        "class_aware_matching": True, "decoder": "frozen decode_v2.decode_objects_v2",
        "dataset_dir": str(dataset_dir),
    }
    (out / "dump_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    print(json.dumps(meta, indent=2, sort_keys=True))


def _dist(g, cam_c) -> float:
    return float(math.hypot(float(g["world_x"]) - float(cam_c[0]), float(g["world_y"]) - float(cam_c[1])))


def _sha(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()




# -------------------------------------------------------------------------- taxon
DIST_BANDS = [(0, 10), (10, 20), (20, 30), (30, 40)]
WIDTH_BANDS = [(0, 2), (2, 4), (4, 8), (8, 16), (16, 1e9)]
AREA_BANDS = [(0, 8), (8, 32), (32, 128), (128, 512), (512, 1e18)]
OVL_BANDS = [(0, .25), (.25, .5), (.5, .75), (.75, 1.01)]
EXPAND_FULL_PX = {"vehicle": 4.0, "person": 2.0}   # pre-registered, full-resolution px


def _band(v, bands, fmt="{:g}-{:g}"):
    for lo, hi in bands:
        if lo <= v < hi:
            return fmt.format(lo, hi) if hi < 1e8 else f"{lo:g}+"
    return "out_of_range"


def stage_taxon(args: argparse.Namespace) -> None:
    import pandas as pd

    out = Path(args.out_dir)
    pk = pd.read_csv(out / "peaks_s002.csv")
    gt = pd.read_csv(out / "gt_objects.csv")

    # index peaks by sample
    pk_by = {s: d for s, d in pk.groupby("sample_id", sort=False)}
    recs = []
    for sid, gsub in gt.groupby("sample_id", sort=False):
        P = pk_by.get(sid)
        px = P["center_x_full"].to_numpy() if P is not None else np.zeros(0)
        py = P["center_y_full"].to_numpy() if P is not None else np.zeros(0)
        pc = P["class_name"].to_numpy() if P is not None else np.zeros(0, dtype=object)
        pw = P["pred_world_x"].to_numpy() if P is not None else np.zeros(0)
        pwy = P["pred_world_y"].to_numpy() if P is not None else np.zeros(0)
        ps = P["score"].to_numpy() if P is not None else np.zeros(0)
        pg = P["range_gated_in_s002"].to_numpy() if P is not None else np.zeros(0)
        pm = P["matched_gt_idx_s002"].to_numpy() if P is not None else np.zeros(0)
        pst = P["branch_stride"].to_numpy() if P is not None else np.zeros(0)
        pnx = P["native_x"].to_numpy() if P is not None else np.zeros(0)
        pny = P["native_y"].to_numpy() if P is not None else np.zeros(0)
        psz = P[["pred_size_x", "pred_size_y", "pred_size_z"]].to_numpy() if P is not None else np.zeros((0, 3))
        prs = P["radar_support_score"].to_numpy() if P is not None else np.zeros(0)

        for _, g in gsub.iterrows():
            cls = g["class_name"]
            e = EXPAND_FULL_PX[cls]
            hw, hh = g["gt_bbox_w_full"] / 2.0, g["gt_bbox_h_full"] / 2.0
            cx, cy = g["gt_center_x_full"], g["gt_center_y_full"]
            strict = (px >= cx - hw) & (px <= cx + hw) & (py >= cy - hh) & (py <= cy + hh)
            exp = (px >= cx - hw - e) & (px <= cx + hw + e) & (py >= cy - hh - e) & (py <= cy + hh + e)
            same = pc == cls
            err = np.hypot(pw - g["gt_world_x"], pwy - g["gt_world_y"]) if len(px) else np.zeros(0)

            rec = {
                "sample_id": sid, "gt_idx": int(g["gt_idx"]), "gt_actor_id": g["gt_actor_id"],
                "class_name": cls, "experiment_id": g["experiment_id"],
                "traffic_density": g["traffic_density"], "pedestrian_density": g["pedestrian_density"],
                "frame_id": g["frame_id"],
                "gt_world_x": g["gt_world_x"], "gt_world_y": g["gt_world_y"], "gt_world_z": g["gt_world_z"],
                "gt_center_x_full": cx, "gt_center_y_full": cy,
                "gt_bbox_w_full": g["gt_bbox_w_full"], "gt_bbox_h_full": g["gt_bbox_h_full"],
                "gt_bbox_w_res": g["gt_bbox_w_res"], "gt_bbox_h_res": g["gt_bbox_h_res"],
                "gt_area_res": g["gt_area_res"],
                "distance_m": g["distance_m"], "radar_support_points": g["radar_support_points"],
                "nearer_box_overlap_frac": g["nearer_box_overlap_frac"],
                "veh_semantic_frac": g["veh_semantic_frac"],
                "detected_s002": int(g["detected_s002"]), "detected_s020": int(g["detected_s020"]),
                "matched_dist_s002": g["matched_dist_s002"],
                "n_peaks_strict": int(strict.sum()), "n_peaks_expanded": int(exp.sum()),
                "n_sameclass_peaks_strict": int((strict & same).sum()),
                "n_sameclass_peaks_expanded": int((exp & same).sum()),
                "n_wrongclass_peaks_expanded": int((exp & ~same).sum()),
            }
            # nearest same-class peak overall (for the contact sheet / diagnostics)
            if len(px) and same.any():
                j = int(np.argmin(np.hypot(px - cx, py - cy) + np.where(same, 0.0, 1e9)))
                rec.update(nearest_same_score=float(ps[j]), nearest_same_xy_err_m=float(err[j]),
                           nearest_same_cx_full=float(px[j]), nearest_same_cy_full=float(py[j]),
                           nearest_same_stride=int(pst[j]), nearest_same_native_x=int(pnx[j]),
                           nearest_same_native_y=int(pny[j]),
                           nearest_same_size_x=float(psz[j, 0]), nearest_same_size_y=float(psz[j, 1]),
                           nearest_same_size_z=float(psz[j, 2]),
                           nearest_same_radar_support_score=float(prs[j]))
            else:
                rec.update(nearest_same_score="", nearest_same_xy_err_m="", nearest_same_cx_full="",
                           nearest_same_cy_full="", nearest_same_stride="", nearest_same_native_x="",
                           nearest_same_native_y="", nearest_same_size_x="", nearest_same_size_y="",
                           nearest_same_size_z="", nearest_same_radar_support_score="")

            if g["detected_s002"] == 1:
                rec["fn_reason"] = "TP"
                rec["fn_reason_strict"] = "TP"
            else:
                claimed = same & (err <= 3.0) & (pg == 1) & (~pd.isna(pm))
                eligible_unclaimed = same & (err <= 3.0) & (pg == 1) & (pd.isna(pm))
                rec["fn_reason"] = _reason(claimed.any(), eligible_unclaimed.any(),
                                           (exp & same).any(), (exp & same & (err > 3.0)).any(),
                                           (exp & ~same).any(), exp.any())
                rec["fn_reason_strict"] = _reason(claimed.any(), eligible_unclaimed.any(),
                                                  (strict & same).any(), (strict & same & (err > 3.0)).any(),
                                                  (strict & ~same).any(), strict.any())
                rec["contention_peak_count"] = int(claimed.sum())
            rec["sameclass_peak_in_strict_box"] = int((strict & same).any())
            rec["distance_band"] = _band(float(g["distance_m"]), DIST_BANDS)
            rec["width_band_res_px"] = _band(float(g["gt_bbox_w_res"]), WIDTH_BANDS)
            rec["area_band_res_px2"] = _band(float(g["gt_area_res"]), AREA_BANDS)
            rec["overlap_band"] = _band(float(g["nearer_box_overlap_frac"]), OVL_BANDS)
            rec["radar_support_band"] = "positive" if float(g["radar_support_points"] or 0) > 0 else "zero"
            recs.append(rec)

    df = pd.DataFrame(recs)
    df.to_csv(out / "fn_taxonomy_per_gt.csv", index=False)
    print("wrote", out / "fn_taxonomy_per_gt.csv", len(df))
    return df


def _reason(has_claimed, has_eligible_unclaimed, has_same_in_box, has_same_far, has_wrong_in_box, has_any_in_box):
    if has_claimed:
        return "MATCHING_CONTENTION"
    if has_eligible_unclaimed:
        return "MATCHING_CONTENTION"      # peak within 3 m existed but greedy order left it unused
    if has_same_far:
        return "XYZ_LOCALIZATION_FAILURE"
    if has_same_in_box:
        return "XYZ_LOCALIZATION_FAILURE"
    if has_wrong_in_box:
        return "CLASS_CONFUSION"
    return "HEATMAP_CENTER_MISS"




# -------------------------------------------------------------------------- summary
def stage_summary(args):
    import pandas as pd
    out = Path(args.out_dir)
    df = pd.read_csv(out / "fn_taxonomy_per_gt.csv")
    fn = df[df.fn_reason != "TP"]
    REASONS = ["MATCHING_CONTENTION", "XYZ_LOCALIZATION_FAILURE", "CLASS_CONFUSION", "HEATMAP_CENTER_MISS"]
    rows = []

    def emit(dim, cls, key, sub):
        if not len(sub):
            return
        for r in REASONS:
            c = int((sub.fn_reason == r).sum())
            rows.append(dict(breakdown=dim, class_name=cls, bucket=key, fn_reason=r,
                             unique_gt_fn=c, pct_of_bucket_fn=round(100 * c / len(sub), 3),
                             bucket_fn_total=len(sub)))
    DIMS = [("distance_m", "distance_band", ["0-10", "10-20", "20-30", "30-40"]),
            ("resized_width_px", "width_band_res_px", ["0-2", "2-4", "4-8", "8-16", "16+"]),
            ("resized_area_px2", "area_band_res_px2", ["0-8", "8-32", "32-128", "128-512", "512+"]),
            ("radar_support", "radar_support_band", ["zero", "positive"]),
            ("nearer_box_overlap", "overlap_band", ["0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1.01"]),
            ("traffic_density", "traffic_density", None), ("episode", "experiment_id", None)]
    for cls in ("vehicle", "person"):
        s = fn[fn.class_name == cls]
        emit("overall", cls, "all", s)
        for dim, col, order in DIMS:
            for k in (order or sorted(s[col].dropna().astype(str).unique())):
                emit(dim, cls, str(k), s[s[col].astype(str) == str(k)])
    pd.DataFrame(rows).to_csv(out / "fn_taxonomy_summary.csv", index=False)

    # v1 -> v2 transitions on identical GT keys (v1 corrected/hybrid decoder, retained audit table)
    if args.v1_gt_audit and Path(args.v1_gt_audit).exists():
        v1 = pd.read_csv(args.v1_gt_audit)
        m = df.merge(v1[["sample_id", "gt_index", "class_name", "tp_hybrid_greedy_002", "tp_hybrid_greedy_020"]],
                     left_on=["sample_id", "gt_idx", "class_name"],
                     right_on=["sample_id", "gt_index", "class_name"], how="inner")
        for tag in ("s002", "s020"):
            a, b = m[f"tp_hybrid_greedy_{tag[1:]}" if False else f"tp_hybrid_greedy_{tag.replace('s','')}"], m[f"detected_{tag}"]
            m[f"transition_{tag}"] = np.select(
                [(a == 1) & (b == 1), (a == 0) & (b == 1), (a == 1) & (b == 0), (a == 0) & (b == 0)],
                ["UNCHANGED_TP", "V1_FN_TO_V2_TP", "V1_TP_TO_V2_FN", "UNCHANGED_FN"], default="NA")
        m[["sample_id", "gt_idx", "gt_actor_id", "class_name", "distance_m", "gt_bbox_w_res",
           "radar_support_points", "nearer_box_overlap_frac", "tp_hybrid_greedy_002", "detected_s002",
           "transition_s002", "tp_hybrid_greedy_020", "detected_s020", "transition_s020",
           "fn_reason", "fn_reason_strict"]].to_csv(out / "v1_v2_gt_transitions.csv", index=False)
    print("summary + transitions written")


# -------------------------------------------------------------------------- sheet
def stage_sheet(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.patches import Rectangle
    from PIL import Image
    O=str(Path(args.out_dir))+'/'
    DS=str(Path(args.dataset_dir))+'/'
    df=pd.read_csv(O+'fn_taxonomy_per_gt.csv')
    fn=df[(df.class_name=='person')&(df.fn_reason!='TP')].copy()
    
    man={}
    with open(DS+'manifest.csv') as fh:
        for r in csv.DictReader(fh):
            man[r['sample_id']]=r['rgb_path']
    
    # deterministic stratification: reason x range x width x overlap, proportional, seedless
    fn['stratum']=fn.fn_reason+'|'+fn.distance_band+'|'+fn.width_band_res_px+'|'+fn.overlap_band
    counts=fn.stratum.value_counts()
    N=32
    alloc={}; rem=N
    for s,c in counts.items():
        alloc[s]=int(round(N*c/len(fn)))
    strata=[s for s in counts.index if alloc.get(s,0)>0]
    # force coverage of every taxonomy reason present
    for r in fn.fn_reason.unique():
        if not any(s.startswith(r+'|') for s in strata):
            strata.append(counts[counts.index.str.startswith(r+'|')].index[0]); alloc[strata[-1]]=1
    picks=[]
    for s in strata:
        k=max(1,alloc[s])
        sub=fn[fn.stratum==s].sort_values(['sample_id','gt_idx'])
        idx=np.linspace(0,len(sub)-1,min(k,len(sub))).round().astype(int)
        picks.append(sub.iloc[np.unique(idx)])
    sel=pd.concat(picks).drop_duplicates(['sample_id','gt_idx'])
    sel=sel.sort_values(['fn_reason','distance_m','gt_bbox_w_res','sample_id','gt_idx'])
    if len(sel)>N:
        sel=sel.iloc[np.linspace(0,len(sel)-1,N).round().astype(int)].drop_duplicates(['sample_id','gt_idx'])
    if len(sel)<N:
        extra=fn[~fn.set_index(['sample_id','gt_idx']).index.isin(sel.set_index(['sample_id','gt_idx']).index)]
        extra=extra.sort_values(['fn_reason','distance_m','sample_id','gt_idx'])
        sel=pd.concat([sel,extra.iloc[np.linspace(0,len(extra)-1,N-len(sel)).round().astype(int)]])
    sel=sel.sort_values(['fn_reason','distance_m','gt_bbox_w_res','sample_id','gt_idx']).head(N).reset_index(drop=True)
    print("panels",len(sel)); print(sel.fn_reason.value_counts().to_dict())
    
    CROP=320
    fig,axes=plt.subplots(4,8,figsize=(30,20))
    recs=[]
    for i,(_,g) in enumerate(sel.iterrows()):
        ax=axes[i//8,i%8]
        img=Image.open(DS+man[g.sample_id]).convert('RGB')
        cx,cy=g.gt_center_x_full,g.gt_center_y_full
        w,h=g.gt_bbox_w_full,g.gt_bbox_h_full
        # fixed-aspect square crop, clamped inside the frame, then rescaled to CROP px
        side=max(140.0, 2.2*max(w,h)); side=min(side, float(min(img.width,img.height)))
        x0=min(max(0.0,cx-side/2), img.width-side); y0=min(max(0.0,cy-side/2), img.height-side)
        crop=img.crop((int(x0),int(y0),int(x0+side),int(y0+side))).resize((CROP,CROP),Image.Resampling.LANCZOS)
        k=CROP/side
        ax.imshow(np.asarray(crop))
        ax.add_patch(Rectangle(((cx-w/2-x0)*k,(cy-h/2-y0)*k),w*k,h*k,fill=False,ec='lime',lw=1.6))
        ax.plot((cx-x0)*k,(cy-y0)*k,'+',color='lime',ms=9,mew=1.8)
        px=g.nearest_same_cx_full; sc=g.nearest_same_score
        if not pd.isna(px):
            qx,qy=(px-x0)*k,(g.nearest_same_cy_full-y0)*k
            if -20<=qx<=CROP+20 and -20<=qy<=CROP+20:
                ax.plot(min(max(qx,3),CROP-3),min(max(qy,3),CROP-3),'x',color='red',ms=9,mew=1.8)
            else:
                ax.plot(CROP-10,10,'x',color='orange',ms=9,mew=1.8)
        ax.set_xlim(0,CROP); ax.set_ylim(CROP,0)
        ax.set_xticks([]); ax.set_yticks([])
        xye = "n/a" if pd.isna(sc) else f"{g.nearest_same_xy_err_m:.2f}m"
        scs = "none" if pd.isna(sc) else f"{sc:.3f}"
        ax.set_title(f"P{i+1:02d}  {g.fn_reason}", fontsize=8.5, fontfamily='monospace', loc='left', pad=3)
        ax.set_xlabel(f"score={scs}  xyErr={xye}  rng={g.distance_m:.1f}m\n"
                      f"resized={g.gt_bbox_w_res:.1f}x{g.gt_bbox_h_res:.1f}px  radar={int(g.radar_support_points)}  "
                      f"nearerOvl={g.nearer_box_overlap_frac*100:.0f}%\n"
                      f"VERDICT: [ ]visible [ ]partial [ ]occluded\n"
                      f"         [ ]not-visible [ ]ambiguous",
                      fontsize=7.2, fontfamily='monospace', loc='left', labelpad=3)
        recs.append(dict(panel=f"P{i+1:02d}", sample_id=g.sample_id, gt_idx=int(g.gt_idx),
                         gt_actor_id=g.gt_actor_id, fn_reason=g.fn_reason,
                         distance_m=round(float(g.distance_m),2),
                         resized_w_px=round(float(g.gt_bbox_w_res),2), resized_h_px=round(float(g.gt_bbox_h_res),2),
                         nearer_box_overlap_pct=round(float(g.nearer_box_overlap_frac)*100,1),
                         radar_support_points=int(g.radar_support_points),
                         nearest_pred_score=("" if pd.isna(sc) else round(float(sc),4)),
                         nearest_pred_xy_err_m=("" if pd.isna(sc) else round(float(g.nearest_same_xy_err_m),3)),
                         manual_verdict="", verdict_choices="clearly visible|partially visible|heavily occluded|not visible|ambiguous"))
    fig.suptitle("CenterNet v2 epoch-012 — person false negatives @score 0.02 — 32-panel manual observability review\n"
                 "green = GT box/centre   red x = nearest same-class native peak   VERDICT LEFT BLANK BY DESIGN",
                 fontsize=15)
    fig.subplots_adjust(left=.01,right=.99,top=.925,bottom=.02,wspace=.10,hspace=.62)
    fig.savefig(O+'person_fn_taxonomy_contact_sheet.png',dpi=105)
    pd.DataFrame(recs).to_csv(O+'person_fn_contact_sheet_manual_review.csv',index=False)
    print("saved")
    


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["dump", "taxon", "summary", "sheet"])
    ap.add_argument("--experiment-dir", default="")
    ap.add_argument("--config", default="")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dataset-dir", default="")
    ap.add_argument("--v1-gt-audit", default="")
    ap.add_argument("--batch-size", default=8)
    ap.add_argument("--num-workers", default=8)
    a = ap.parse_args()
    if a.stage == "dump":
        stage_dump(a)
    elif a.stage == "taxon":
        stage_taxon(a)
    elif a.stage == "summary":
        stage_summary(a)
    elif a.stage == "sheet":
        stage_sheet(a)