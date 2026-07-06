#!/usr/bin/env python3
"""Offline renderer for spatial-map snapshots (CARLA-free). Mirrors the live canvas viewer's Stage-2
logic (color-by-SOURCE when >1 stream, else by type) in headless matplotlib so figures can be inspected.

Works from a recorded trace (record_trace.py output) or a synthetic scene:
  python3 replay_trace.py --synthetic --outdir autonomous_run/figs
  python3 replay_trace.py --trace recordings/two_ego.jsonl --static recordings/two_ego.jsonl.static.json --last --outdir autonomous_run/figs
  python3 replay_trace.py --trace recordings/two_ego.jsonl --every 25 --outdir autonomous_run/figs
"""
from __future__ import annotations
import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

PALETTE = ["#00d1ff", "#ff9f43", "#8aff80", "#c780ff", "#ffd166", "#ff5fd1"]
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})

# Canonical footprints (metres) by class — the model's regressed dims are de-prioritized/unreliable
# (like its yaw), so we draw standard sizes for clean, consistent boxes. length x width.
CANONICAL_SIZE = {"Vehicle": (4.6, 2.0), "Pedestrian": (0.8, 0.8), "Cyclist": (1.8, 0.7)}


def box_size_for(o, mode="canonical"):
    d = o.get("dimensions") or {}
    model_lw = (_f(d.get("length"), 1.0), _f(d.get("width"), 1.0))
    if mode == "model":
        return model_lw
    return CANONICAL_SIZE.get(o.get("type"), model_lw)


def _f(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def _oriented_corners(cx, cy, length, width, yaw_deg):
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    hl, hw = max(0.05, length) / 2.0, max(0.05, width) / 2.0
    pts = [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)]
    return [(cx + px * c - py * s, cy + px * s + py * c) for px, py in pts]


def nearest_road_heading(x, y, roads, max_dist_m=15.0):
    """Orientation (deg) of the nearest road-centreline segment to (x,y), or (None, None) if no road is
    within max_dist_m. Box orientation only needs this mod 180 (a rectangle is symmetric), so the road's
    direction sign doesn't matter. Robust for stopped vehicles where velocity heading is unavailable."""
    best_d2, best_h = float("inf"), None
    for pl in roads or []:
        for i in range(len(pl) - 1):
            ax, ay = pl[i][0], pl[i][1]
            bx, by = pl[i + 1][0], pl[i + 1][1]
            dx, dy = bx - ax, by - ay
            seg2 = dx * dx + dy * dy
            if seg2 < 1e-9:
                continue
            t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / seg2))
            px, py = ax + t * dx, ay + t * dy
            d2 = (x - px) ** 2 + (y - py) ** 2
            if d2 < best_d2:
                best_d2, best_h = d2, math.degrees(math.atan2(dy, dx))
    if best_h is None or best_d2 > max_dist_m ** 2:
        return None, None
    return best_h, math.sqrt(best_d2)


def object_heading(o, roads, mode="road"):
    """Heading (deg) to draw an object's box. mode='model' uses the (unreliable) model yaw;
    'road' snaps vehicles to the nearest road direction, falling back to model yaw when no road is near.
    Pedestrians always keep model yaw (box is tiny; orientation irrelevant)."""
    model_yaw = _f(o.get("map_yaw_deg", o.get("yaw_deg")))
    if mode == "model" or o.get("type") == "Pedestrian":
        return model_yaw, "model"
    loc = o.get("location") or {}
    rh, rd = nearest_road_heading(_f(loc.get("x")), _f(loc.get("y")), roads)
    return (rh, "road") if rh is not None else (model_yaw, "model")


def render_snapshot(snap: Dict, static: Optional[Dict], out_png: str, title: str = "",
                    heading_mode: str = "road", box_size_mode: str = "canonical") -> Dict:
    """Render one snapshot to a PNG. Returns a small dict of what it drew (for logging/asserts).
    heading_mode: 'road' (snap vehicle boxes to nearest road orientation; the default, fixes model-yaw
    slant) or 'model' (raw model yaw)."""
    fv = (snap.get("metadata") or {}).get("focus_view") or {}
    ego = fv.get("ego_pose")
    bounds = fv.get("bounds")
    objs = snap.get("raw_spatial_map_objects") or []
    active = snap.get("active_streams") or []
    srcs = sorted({s for s in ({st.get("stream_id") for st in active}
                               | {o.get("source_stream_id") for o in objs}) if s})
    by_source = len(srcs) > 1
    src_color = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(srcs)}

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("#080b10")
    fig.patch.set_facecolor("#080b10")

    static = static or {}
    for b in static.get("buildings") or []:
        if len(b) >= 3:
            ax.add_patch(Polygon(b, closed=True, facecolor="#20242b", edgecolor="#333a44", lw=0.8, zorder=2))
    for pl in static.get("roads") or []:
        if len(pl) >= 2:
            ax.plot([p[0] for p in pl], [p[1] for p in pl], color="#4a525c", lw=1.6, zorder=3,
                    solid_capstyle="round", solid_joinstyle="round")

    for a in snap.get("traffic_light_anchors") or []:
        loc = a.get("location") or {}
        ax.scatter([_f(loc.get("x"))], [_f(loc.get("y"))], c="#8a8f98", marker="^", s=36, zorder=4)

    roads = static.get("roads") or []
    counts = {}
    nv = np_ = 0
    heading_methods = {"road": 0, "model": 0}
    for o in objs:
        loc, dim = o.get("location") or {}, o.get("dimensions") or {}
        ped = o.get("type") == "Pedestrian"
        nv += (not ped)
        np_ += ped
        sid = o.get("source_stream_id")
        counts[sid] = counts.get(sid, 0) + 1
        col = src_color.get(sid, "#ffffff") if by_source else ("#ff5fd1" if ped else "#00d1ff")
        cx, cy = _f(loc.get("x")), _f(loc.get("y"))
        yaw, method = object_heading(o, roads, heading_mode)
        heading_methods[method] = heading_methods.get(method, 0) + 1
        bL, bW = box_size_for(o, box_size_mode)
        corners = _oriented_corners(cx, cy, bL, bW, yaw)
        ax.add_patch(Polygon(corners, closed=True, fill=ped, facecolor=col, edgecolor=col,
                             alpha=0.35 if ped else 1.0, lw=2, zorder=6))
        ax.scatter([cx], [cy], c=col, s=22, edgecolors="white", linewidths=0.5, zorder=7)

    if bounds:
        ax.add_patch(Rectangle((bounds["x_min"], bounds["y_min"]),
                               bounds["x_max"] - bounds["x_min"], bounds["y_max"] - bounds["y_min"],
                               fill=False, edgecolor="#78b4ff", alpha=0.4, lw=1.2, zorder=5))
        ax.set_xlim(bounds["x_min"], bounds["x_max"])
        ax.set_ylim(bounds["y_max"], bounds["y_min"])  # y down (CARLA top-down)

    # ego markers — draw every streaming car at its own sensor pose, colored by source
    ego_drawn = 0
    for st in active:
        sp = st.get("sensor_pose") or {}
        if sp.get("x") is None:
            continue
        ex, ey, eyaw = _f(sp.get("x")), _f(sp.get("y")), _f(sp.get("yaw_deg"))
        col = src_color.get(st.get("stream_id"), "#ffcf66")
        c, s = math.cos(math.radians(eyaw)), math.sin(math.radians(eyaw))
        ax.plot([ex - 3 * c, ex + 5 * c], [ey - 3 * s, ey + 5 * s], color=col, lw=3, zorder=8)
        ax.scatter([ex], [ey], c=col, s=90, marker="o", edgecolors="white", linewidths=1.2, zorder=9)
        ax.annotate(st.get("stream_id", ""), (ex, ey), textcoords="offset points", xytext=(8, 8),
                    color=col, fontsize=9, fontweight="bold")
        ego_drawn += 1
    if not ego_drawn and ego:  # fallback: followed ego only
        ex, ey, eyaw = _f(ego.get("x")), _f(ego.get("y")), _f(ego.get("yaw_deg"))
        c, s = math.cos(math.radians(eyaw)), math.sin(math.radians(eyaw))
        ax.plot([ex - 3 * c, ex + 4 * c], [ey - 3 * s, ey + 4 * s], color="#ffcf66", lw=3, zorder=8)
        ax.scatter([ex], [ey], c="#ffcf66", s=60, marker="o", edgecolors="black", zorder=9)

    if by_source:
        handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=src_color[s],
                              markersize=10, label=f"{s} ({counts.get(s, 0)})") for s in srcs]
        ax.legend(handles=handles, loc="upper right", framealpha=0.3, fontsize=9, title="source")

    ax.set_title(title or f"frame {snap.get('frame_id')} | {'by source' if by_source else 'by type'} | "
                          f"{nv} veh {np_} ped", color="#e6edf3", fontsize=12)
    ax.set_xlabel("world X (m)", color="#9aa4af")
    ax.set_ylabel("world Y (m)", color="#9aa4af")
    ax.tick_params(colors="#5a626c")
    ax.set_aspect("equal", adjustable="box")
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", facecolor="#080b10")
    plt.close(fig)
    return {"streams": srcs, "by_source": by_source, "n_veh": nv, "n_ped": np_,
            "per_source": counts, "has_ego": ego is not None, "heading_methods": heading_methods,
            "png": out_png}


def _load_trace(path: str) -> List[Dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace")
    ap.add_argument("--static")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--outdir", default="autonomous_run/figs")
    ap.add_argument("--frame", type=int, default=-1, help="row index in the trace")
    ap.add_argument("--last", action="store_true")
    ap.add_argument("--every", type=int, default=0, help="render every Kth row")
    ap.add_argument("--heading", choices=("road", "model"), default="road",
                    help="box orientation source: 'road' snaps to nearest road (fixes model-yaw slant)")
    ap.add_argument("--box-size", choices=("canonical", "model"), default="canonical", dest="box_size",
                    help="box footprint: 'canonical' uses standard class sizes (fixes noisy model dims)")
    a = ap.parse_args()

    if a.synthetic:
        from synthetic_scenes import two_ego_scene
        snap, static = two_ego_scene()
        info = render_snapshot(snap, static,
                               os.path.join(a.outdir, f"stage2_synthetic_two_ego_{a.heading}.png"),
                               title=f"Stage 2 (synthetic) — color by source, heading={a.heading}",
                               heading_mode=a.heading, box_size_mode=a.box_size)
        print("rendered synthetic:", json.dumps(info))
        return

    if not a.trace:
        ap.error("provide --trace or --synthetic")
    static = None
    static_path = a.static or (a.trace + ".static.json")
    if os.path.exists(static_path):
        with open(static_path) as f:
            static = json.load(f)
    rows = _load_trace(a.trace)
    if not rows:
        print("no rows in trace:", a.trace)
        return
    idxs = ([len(rows) - 1] if a.last else
            list(range(0, len(rows), a.every)) if a.every else
            [a.frame if a.frame >= 0 else len(rows) - 1])
    for i in idxs:
        snap = rows[i].get("snap") or {}
        info = render_snapshot(snap, static, os.path.join(a.outdir, f"replay_{i:05d}.png"),
                               title=f"replay row {i} — frame {snap.get('frame_id')} "
                                     f"(heading={a.heading}, size={a.box_size})",
                               heading_mode=a.heading, box_size_mode=a.box_size)
        print(f"row {i}:", json.dumps(info))


if __name__ == "__main__":
    main()
