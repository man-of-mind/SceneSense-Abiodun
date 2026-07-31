#!/usr/bin/env python3
"""Post-hoc per-frame DENSITY LABEL for the test split (DENSITY_ADAPTIVE_KNOB_PLAN.md step 1).

The ego drives the route continuously and density varies naturally; nothing is controlled during
the drive. Each frame is labelled AFTER the fact from GT -- exactly the trick the road-state
analysis used to tag frames curve/straight/intersection -- with the in-view GT object count:
in the camera frustum, <= 40 m, above the eval's min GT area. That is the SAME object set the
accuracy is scored against, so density label and accuracy denominator cannot disagree.

Also emits the confound controls the plan asks for (density correlates with location: crowded ==
intersections): mean/min GT distance, mean GT speed, fraction parked, per frame.

Out: frame_density.csv  (one row per test frame)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

AB = Path("/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun")
for p in (str(AB / "pole_lraspp_multimodal_fusion"), str(AB)):
    if p not in sys.path:
        sys.path.insert(0, p)

from pole_lraspp_multimodal_fusion.common import load_config, read_manifest  # noqa: E402
from pole_lraspp_multimodal_fusion.object_targets import load_object_boxes, valid_localization_objects  # noqa: E402

DS = AB / "fusion_training_data" / "moving_ego_pps200000_merged_8loops_stride2"
CFG = AB / "pole_lraspp_multimodal_fusion" / "configs" / "fusion_full_run.yaml"
MAX_GT_DIST_M = 40.0
OUT = AB / "rl_agent" / "density_knob" / "raw" / "frame_density.csv"

BINS = [(0, 0, "0"), (1, 2, "1-2"), (3, 4, "3-4"), (5, 10 ** 6, "5+")]


def density_bin(n: int) -> str:
    for lo, hi, name in BINS:
        if lo <= n <= hi:
            return name
    return "?"


def main() -> int:
    cfg = load_config(str(CFG))
    object_cfg = cfg.get("object_heads", {})
    min_area = float(object_cfg.get("min_gt_area_px", 24.0))
    classes = tuple(object_cfg.get("object_classes", ("vehicle", "person")))
    rows = [r for r in read_manifest(DS / "manifest.csv") if r.get("split") == "test"]
    boxes = load_object_boxes(DS / "object_boxes.csv")

    # index the raw GT rows so distance/speed can be read for exactly the IN-VIEW objects
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sample_id", "frame_id", "scenario_id", "view_id", "traffic_light_id",
              "n_inview", "n_inview_veh", "n_inview_ped", "density_bin",
              "gt_dist_mean_m", "gt_dist_min_m", "gt_speed_mean_mps", "gt_speed_max_mps",
              "frac_parked", "frac_moving", "camera_x", "camera_y"]
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            raw = boxes.get(r["sample_id"], [])
            objs = valid_localization_objects(
                raw, image_width=int(r["camera_width"]), image_height=int(r["camera_height"]),
                min_area_px=min_area, object_class_names=classes, max_distance_m=MAX_GT_DIST_M)
            # re-derive the same in-view subset on the RAW rows to read distance/speed/parked
            keep = []
            for b in raw:
                if b.get("label") not in classes or b.get("gt_source") != "actor":
                    continue
                if b.get("object_sensor_x", "") == "" or b.get("object_world_x", "") == "":
                    continue
                try:
                    if float(b.get("gt_bbox_area_px") or 0.0) < min_area:
                        continue
                    if float(b.get("gt_distance_m") or 0.0) > MAX_GT_DIST_M:
                        continue
                    cx, cy = float(b.get("gt_center_x") or -1), float(b.get("gt_center_y") or -1)
                except ValueError:
                    continue
                if not (0.0 <= cx < float(r["camera_width"]) and 0.0 <= cy < float(r["camera_height"])):
                    continue
                keep.append(b)
            assert len(keep) == len(objs), f"in-view subset mismatch {len(keep)} != {len(objs)} on {r['sample_id']}"
            n = len(objs)
            nv = sum(1 for o in objs if o["class_name"] == "vehicle")
            dists = [float(b["gt_distance_m"]) for b in keep]
            speeds = [float(b.get("object_speed_mps") or 0.0) for b in keep]
            parked = [float(b.get("parked_label") or 0.0) for b in keep]
            w.writerow({
                "sample_id": r["sample_id"], "frame_id": r.get("frame_id", ""),
                "scenario_id": r.get("scenario_id", ""), "view_id": r.get("view_id", ""),
                "traffic_light_id": r.get("traffic_light_id", ""),
                "n_inview": n, "n_inview_veh": nv, "n_inview_ped": n - nv,
                "density_bin": density_bin(n),
                "gt_dist_mean_m": round(sum(dists) / n, 3) if n else "",
                "gt_dist_min_m": round(min(dists), 3) if n else "",
                "gt_speed_mean_mps": round(sum(speeds) / n, 3) if n else "",
                "gt_speed_max_mps": round(max(speeds), 3) if n else "",
                "frac_parked": round(sum(parked) / n, 3) if n else "",
                "frac_moving": round(sum(1 for s in speeds if s > 0.5) / n, 3) if n else "",
                "camera_x": r.get("camera_x", ""), "camera_y": r.get("camera_y", ""),
            })
    print(f"wrote {OUT} ({len(rows)} test frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
