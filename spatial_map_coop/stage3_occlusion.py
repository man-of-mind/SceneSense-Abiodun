#!/usr/bin/env python3
"""Stage-3 GROUNDWORK (autonomous, synthetic-GT verifiable): bridge our scenes into the reusable
`spatial_map_geometry` scaffold, run its EXISTING occlusion reasoner, verify against known ground truth,
and render an overlay. This deliberately uses only the baseline FoV-membership reasoner — the novel
ray/visibility-grid disambiguation is HELD for collaboration (see STAGE3_NOTES.md).

  python3 stage3_occlusion.py --outdir autonomous_run/figs
"""
from __future__ import annotations
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# make the sibling scaffold package importable
_ABIODUN = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
if _ABIODUN not in sys.path:
    sys.path.insert(0, _ABIODUN)
from spatial_map_geometry import geometry, occlusion_reasoner  # noqa: E402
from spatial_map_geometry.schemas import LocalSensorMap, SensorPose2D, SpatialObject  # noqa: E402

SRC_COLOR = {"fusion_ego": "#00d1ff", "fusion_ego_b": "#ff9f43"}


def scene_to_local_maps(streams):
    """Our synthetic stream dicts -> scaffold LocalSensorMap list (FoV polygon from pose+fov+range)."""
    maps = []
    for i, s in enumerate(streams):
        pose = SensorPose2D(x=s["pose"]["x"], y=s["pose"]["y"], yaw_deg=s["pose"]["yaw_deg"])
        fov_poly = geometry.sensor_fov_polygon(pose, float(s["fov_deg"]), float(s["range_m"]))
        objs = [
            SpatialObject(object_id=f"{s['stream_id']}:{j}", class_name=o["type"],
                          x=o["x"], y=o["y"], length=o["length"], width=o["width"],
                          yaw_deg=o.get("yaw_deg", 0.0), confidence=0.9, source_stream_id=s["stream_id"])
            for j, o in enumerate(s["objects"])
        ]
        maps.append(LocalSensorMap(stream_id=s["stream_id"], pose=pose, fov_polygon=fov_poly,
                                   objects=objs, fov_deg=float(s["fov_deg"]), range_m=float(s["range_m"])))
    return maps


def snapshot_to_local_maps(snap, range_m=50.0):
    """Real /latest snapshot -> LocalSensorMap list, using per-stream sensor_pose+fov_deg (needs the
    updated server) and raw_spatial_map_objects grouped by source_stream_id. Only streams with a pose
    and >=1 object are included."""
    active = snap.get("active_streams") or []
    objs_by_src = {}
    for o in snap.get("raw_spatial_map_objects") or []:
        objs_by_src.setdefault(o.get("source_stream_id"), []).append(o)
    maps = []
    for s in active:
        sid = s.get("stream_id")
        pose_d = s.get("sensor_pose") or {}
        if "x" not in pose_d:
            continue  # old server without per-stream pose
        pose = SensorPose2D(x=float(pose_d.get("x", 0)), y=float(pose_d.get("y", 0)),
                            yaw_deg=float(pose_d.get("yaw_deg", 0)))
        fov = float(s.get("fov_deg", 90.0)) or 90.0
        objs = []
        for j, o in enumerate(objs_by_src.get(sid, [])):
            loc, dim = o.get("location") or {}, o.get("dimensions") or {}
            objs.append(SpatialObject(object_id=f"{sid}:{j}", class_name=o.get("type", "Vehicle"),
                                      x=float(loc.get("x", 0)), y=float(loc.get("y", 0)),
                                      length=float(dim.get("length", 1)), width=float(dim.get("width", 1)),
                                      confidence=float(o.get("score", 0.8)), source_stream_id=sid))
        maps.append(LocalSensorMap(stream_id=sid, pose=pose,
                                   fov_polygon=geometry.sensor_fov_polygon(pose, fov, range_m),
                                   objects=objs, fov_deg=fov, range_m=range_m))
    return maps


def run_on_trace(trace_path, outdir):
    """Find the most recent both-egos-fresh snapshot in a trace and run the baseline occlusion reasoner."""
    rows = [__import__("json").loads(l) for l in open(trace_path) if l.strip()]
    best = None
    for r in rows:
        maps = snapshot_to_local_maps(r.get("snap") or {})
        if len(maps) >= 2 and sum(len(m.objects) for m in maps) > 0:
            best = (r, maps)
    if not best:
        print("no snapshot with >=2 posed streams found — is the server updated (sensor_pose)? "
              "and were both egos fresh together?")
        return
    r, maps = best
    hyps = occlusion_reasoner.infer_overlap_disagreements(maps, distance_threshold_m=3.0,
                                                          min_overlap_area_m2=10.0)
    out = os.path.join(outdir, "stage3_occlusion_LIVE.png")
    render(maps, hyps, out,
           title=f"Stage-3 LIVE — baseline FoV reasoner (real 2-ego, {len(hyps)} flagged, unverified)")
    print("streams:", [(m.stream_id, len(m.objects)) for m in maps])
    print("occlusion hypotheses (baseline FoV-membership reasoner):")
    for h in hyps:
        print(f"  {h.class_name} seen by {h.source_stream_id}, missing from {h.missing_from_stream_id} "
              f"@ ({h.x:.1f},{h.y:.1f}) p={h.confidence:.2f} overlap={h.overlap_area_m2:.0f}m2")
    print(f"{len(hyps)} hypotheses. figure: {out}")


def verify(hypotheses, ground_truth, tol_m=3.0):
    """Precision/recall of flagged occlusions vs known GT (match by class + XY within tol)."""
    def match(h, g):
        return (h.class_name == g["class_name"] and h.missing_from_stream_id == g["occluded_from"]
                and abs(h.x - g["x"]) <= tol_m and abs(h.y - g["y"]) <= tol_m)
    tp = sum(1 for g in ground_truth if any(match(h, g) for h in hypotheses))
    matched_h = sum(1 for h in hypotheses if any(match(h, g) for g in ground_truth))
    fp = len(hypotheses) - matched_h
    fn = len(ground_truth) - tp
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec}


def render(local_maps, hypotheses, out_png, title="Stage-3 groundwork — cooperative occlusion (synthetic GT)"):
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor("#080b10"); fig.patch.set_facecolor("#080b10")
    for m in local_maps:
        col = SRC_COLOR.get(m.stream_id, "#8aff80")
        ax.add_patch(Polygon(m.fov_polygon, closed=True, facecolor=col, edgecolor=col, alpha=0.08, lw=1, zorder=2))
        ax.scatter([m.pose.x], [m.pose.y], c=col, marker="s", s=90, edgecolors="white", zorder=6,
                   label=f"{m.stream_id} sensor")
        for o in m.objects:
            ped = o.class_name == "Pedestrian"
            ax.add_patch(Polygon(geometry.object_footprint_polygon(o), closed=True, fill=ped,
                                 facecolor=col, edgecolor=col, alpha=0.4 if ped else 1.0, lw=2, zorder=5))
    for h in hypotheses:
        ax.scatter([h.x], [h.y], s=340, facecolors="none", edgecolors="#ff3b3b", lw=2.5, zorder=8)
        src = next((m for m in local_maps if m.stream_id == h.source_stream_id), None)
        tgt = next((m for m in local_maps if m.stream_id == h.missing_from_stream_id), None)
        if tgt:
            ax.plot([tgt.pose.x, h.x], [tgt.pose.y, h.y], color="#ff3b3b", ls="--", lw=1.4, zorder=7)
        ax.annotate(f"occluded from {h.missing_from_stream_id}\n(seen by {h.source_stream_id}, p={h.confidence:.2f})",
                    (h.x, h.y), textcoords="offset points", xytext=(10, 12), color="#ff8a8a", fontsize=9)
    ax.set_title("Stage-3 groundwork — cooperative occlusion (synthetic GT)", color="#e6edf3", fontsize=13)
    ax.set_xlabel("world X (m)", color="#9aa4af"); ax.set_ylabel("world Y (m)", color="#9aa4af")
    ax.tick_params(colors="#5a626c"); ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", framealpha=0.3, fontsize=9)
    ax.set_ylim(ax.get_ylim()[::-1])  # y down
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", facecolor="#080b10"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="autonomous_run/figs")
    ap.add_argument("--trace", help="run the baseline reasoner on a real recorded trace instead of synthetic")
    a = ap.parse_args()
    if a.trace:
        run_on_trace(a.trace, a.outdir)
        return
    from synthetic_scenes import occlusion_scene
    streams, gt = occlusion_scene()
    local_maps = scene_to_local_maps(streams)
    hyps = occlusion_reasoner.infer_overlap_disagreements(local_maps, distance_threshold_m=3.0,
                                                          min_overlap_area_m2=10.0)
    metrics = verify(hyps, gt)
    out = os.path.join(a.outdir, "stage3_occlusion_synthetic.png")
    render(local_maps, hyps, out)
    print("hypotheses:", [(h.class_name, h.source_stream_id, "->missing_from", h.missing_from_stream_id,
                           round(h.x, 1), round(h.y, 1)) for h in hyps])
    print("ground_truth:", [(g["class_name"], g["occluded_from"]) for g in gt])
    print("metrics:", metrics)
    assert metrics["recall"] == 1.0 and metrics["fn"] == 0, "reasoner missed a known occlusion"
    assert metrics["precision"] == 1.0, f"unexpected false-positive occlusions: {metrics}"
    print("PASS — reasoner flagged the known occlusion with no false positives. figure:", out)


if __name__ == "__main__":
    main()
