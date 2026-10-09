"""Unchanged original matchers/scorers, imported (not copied) from their sources.

vehicle: audit_v1.match_frame (greedy nearest-first same-class 3 m) on predictions sorted
         like audit_v1.load_predictions (-score, class), threshold 0.20, v010 POSITIVE GT;
         FP = unmatched & not is_neutral (ignore mask at rounded centre pixel).
person:  quality_adapter.person_avo_measurement (eligible AVO>=0.65 -> AVO-ignored ->
         structural-ignored), score >= 0.25, predictions sorted by -score as in
         run_comparison.load_person_predictions; structural GT from run_comparison.
"""
from __future__ import annotations

import csv
import importlib
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

audit_v1 = importlib.import_module(
    "pole_lraspp_multimodal_fusion.object_head_pilot_v1.route_b_v3_1_targeted_refinement_v1.audit_v1")
quality_adapter = importlib.import_module("rl_agent.splitfusion_run5_live_avo_v1.quality_adapter")

DS = ROOT / "experiments/route_b_v3_1_expanded_train_camera_plane_v1/20260828_094151"
DET = ROOT / "experiments/splitfusion_fcos_service_candidate_v1/predictions/detections.csv"
DET_SHA = "a682a1fc5eabb2e59e07449a8c6b5fc604077b40ef094b57dc30c5a18d7ec260"
AVO = ROOT / "experiments/actor_volume_observability_model_comparison_v1/20260901_repaired_tolerance_cpu_once/actor_volume_observability_table.csv"
AVO_SHA = "abb976f388ad33e8806d080750e9e7fbe1b1eb60e7e18ea55bedc60dce011386"
HOLDOUT_AVO = ROOT / "experiments/splitfusion_fcos_person_p025_calibration_v1/holdout_actor_volume_observability_table.csv"
THRESH = {"vehicle": 0.20, "person": 0.25}
TOL = {"vehicle": 1.0, "person": 1.2}
AVO_T = 0.65
BINS = (("00_10m", 0.0, 10.0), ("10_20m", 10.0, 20.0), ("20_30m", 20.0, 30.0), ("30_40m", 30.0, 40.0000000001))


def bin_of(d: float) -> str:
    for name, lo, hi in BINS:
        if lo <= d < hi:
            return name
    return ">40m"


def episode_of(sample_id: str) -> str:
    return sample_id.rsplit("_", 2)[0]


def read_detections(path: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    num = ("score", "world_x", "world_y", "world_z", "local_x", "local_y", "local_z", "center_x_px",
           "center_y_px", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1", "depth_residual", "actor_forward_depth_m")
    for r in csv.DictReader(Path(path).open()):
        d = dict(r)
        for k in num:
            d[k] = float(r[k])
        d["level_index"] = int(r["level_index"])
        out[r["sample_id"]].append(d)
    return out


def load_vehicle_gt(split: str) -> dict[str, list[dict[str, Any]]]:
    sx, sy = 768 / 1280, 432 / 720
    g: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in csv.DictReader((DS / f"contracts/v010/{split}/object_boxes.csv").open()):
        if r["label"] != "vehicle" or r["contract_state"] != "POSITIVE":
            continue
        x0, y0 = float(r["gt_bbox_x"]), float(r["gt_bbox_y"])
        g[r["sample_id"]].append({
            "class_name": "vehicle", "world_x": float(r["object_world_x"]), "world_y": float(r["object_world_y"]),
            "distance_m": float(r["gt_distance_m"]), "source_identity": r["source_identity"],
            "box": (x0 * sx, y0 * sy, (x0 + float(r["gt_bbox_w"])) * sx, (y0 + float(r["gt_bbox_h"])) * sy)})
    return g


def load_v010_person_gt(split: str) -> dict[str, list[dict[str, Any]]]:
    sx, sy = 768 / 1280, 432 / 720
    g: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in csv.DictReader((DS / f"contracts/v010/{split}/object_boxes.csv").open()):
        if r["label"] != "person" or r["contract_state"] != "POSITIVE":
            continue
        x0, y0 = float(r["gt_bbox_x"]), float(r["gt_bbox_y"])
        g[r["sample_id"]].append({
            "class_name": "person", "world_x": float(r["object_world_x"]), "world_y": float(r["object_world_y"]),
            "distance_m": float(r["gt_distance_m"]), "gt_actor_id": r["gt_actor_id"],
            "box": (x0 * sx, y0 * sy, (x0 + float(r["gt_bbox_w"])) * sx, (y0 + float(r["gt_bbox_h"])) * sy)})
    return g


def load_avo_table(path: Path) -> dict[str, list[dict[str, Any]]]:
    g: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in csv.DictReader(Path(path).open()):
        wx = r["object_world_x"] if "object_world_x" in r else r["world_x"]  # holdout table names them world_x/y
        wy = r["object_world_y"] if "object_world_y" in r else r["world_y"]
        g[r["sample_id"]].append({"world_x": float(wx), "world_y": float(wy),
                                  "avo": float(r["actor_volume_observability"]), "distance_m": float(r["distance_m"]),
                                  "distance_bin": r["distance_bin"], "gt_actor_id": r["gt_actor_id"]})
    return g


def load_val_structural() -> dict[str, list[dict[str, Any]]]:
    rc = importlib.import_module(
        "data_collection.route_b_publication_actor_volume_observability_model_comparison_v1.run_comparison")
    return rc.structural_gt(rc.load_raw_sources())


def vehicle_frame(preds: Sequence[Mapping[str, Any]], gts: Sequence[Mapping[str, Any]], ignore=None) -> dict[str, Any]:
    """Returns per-GT matched error (None = FN), fp, neutral."""
    p = sorted([x for x in preds if x["class_name"] == "vehicle" and float(x["score"]) >= THRESH["vehicle"]],
               key=lambda it: (-float(it["score"]), str(it["class_name"])))
    used_pred, used_gt, p2g = audit_v1.match_frame(p, gts)
    err = [None] * len(gts)
    for pi, gi in p2g.items():
        err[gi] = math.hypot(float(p[pi]["world_x"]) - gts[gi]["world_x"], float(p[pi]["world_y"]) - gts[gi]["world_y"])
    fp = neutral = 0
    fp_ids = []
    for pi, x in enumerate(p):
        if pi in used_pred:
            continue
        if ignore is not None and audit_v1.is_neutral(x, ignore):
            neutral += 1
        else:
            fp += 1
            fp_ids.append(x["prediction_index"])
    return {"err": err, "fp": fp, "neutral": neutral, "fp_ids": fp_ids,
            "matched_pred_ids": {gi: p[pi]["prediction_index"] for pi, gi in p2g.items()}}


def person_frame(preds: Sequence[Mapping[str, Any]], qualified: Sequence[Mapping[str, Any]],
                 structural: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    p = sorted([dict(x) for x in preds if x["class_name"] == "person" and float(x["score"]) >= THRESH["person"]],
               key=lambda it: -float(it["score"]))
    m = quality_adapter.person_avo_measurement(p, {"qualified": list(qualified), "structural_ignored": list(structural)})
    eligible = [r for r in qualified if float(r["avo"]) >= AVO_T]
    matched, _ = quality_adapter.greedy_match(p, eligible)
    err = [None] * len(eligible)
    for pi, gi in matched.items():
        err[gi] = math.hypot(float(p[pi]["world_x"]) - eligible[gi]["world_x"], float(p[pi]["world_y"]) - eligible[gi]["world_y"])
    assert sum(e is not None for e in err) == m["loc_person_tp"]
    return {"err": err, "eligible": eligible, "fp": m["loc_person_fp"],
            "neutral": m["loc_person_ignored_predictions"], "avo_ignored": m["loc_person_avo_ignored_predictions"],
            "structural_ignored": m["loc_person_structural_ignored_predictions"],
            "matched_pred_ids": {gi: p[pi]["prediction_index"] for pi, gi in matched.items()}}


def simple_frame(cls: str, preds: Sequence[Mapping[str, Any]], gts: Sequence[Mapping[str, Any]]) -> list:
    """Recall-only greedy (same matcher) used for train-side CV; no ignore semantics."""
    p = sorted([x for x in preds if x["class_name"] == cls and float(x["score"]) >= THRESH[cls]],
               key=lambda it: -float(it["score"]))
    _, _, p2g = audit_v1.match_frame(p, gts)
    err = [None] * len(gts)
    for pi, gi in p2g.items():
        err[gi] = math.hypot(float(p[pi]["world_x"]) - gts[gi]["world_x"], float(p[pi]["world_y"]) - gts[gi]["world_y"])
    return err
