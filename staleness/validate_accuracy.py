#!/usr/bin/env python3
"""Validate LIVE deployment accuracy vs offline eval. Matches predictions to GT in WORLD space (both are world
coords from the SAME decode_objects the offline eval uses), with the eval's conventions: score>=0.2, greedy
1-1 association, 5 m gate. Reports live loc-MAE (matched TPs) + per-class + sanity ranges. Compare to offline.
Usage: validate_accuracy.py <run_dir> [score=0.2] [gate=5.0]"""
import csv, sys, math, statistics
from collections import defaultdict
from pathlib import Path

run = Path(sys.argv[1]); SCORE = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2; GATE = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
s = run / "streams"
gt = list(csv.DictReader(open(next(s.glob("*object_ground_truth.csv")))))
pr = list(csv.DictReader(open(next(s.glob("*object_predictions.csv")))))

def cls(c):
    c = (c or "").lower()
    return "person" if ("ped" in c or "person" in c or "walker" in c) else "vehicle"

def truthy(v): return str(v).strip().lower() in ("true", "1", "yes")
# Match the TRAINING GT convention: collect_dataset.py regresses actor.get_location() (the actor ORIGIN),
# so the offline eval measures error vs the origin. Compare live predictions against origin_x/y when present
# (falls back to world_x/y = bbox center for pre-fix runs, which carry a spurious ~1 m convention offset).
def gx(r): return float(r.get("origin_x") or r["world_x"])
def gy(r): return float(r.get("origin_y") or r["world_y"])
USING_ORIGIN = any((r.get("origin_x") not in (None, "")) for r in gt)
print(f"  GT position convention: {'ORIGIN (matches training)' if USING_ORIGIN else 'bbox-center (legacy, +~1m bias)'}")
gtf = defaultdict(list); prf = defaultdict(list)
n_frustum = 0
for r in gt:
    try:
        # reliable in-view gate: in_camera_frustum + within detection range (40 m)
        if not truthy(r.get("in_camera_frustum", "")): continue
        if float(r.get("distance_m", 999)) > 40.0: continue
        n_frustum += 1
        gtf[int(r["frame_id"])].append((gx(r), gy(r), cls(r.get("class_name"))))
    except: pass
print(f"  GT in-frustum & <40m: {n_frustum}")
for r in pr:
    try:
        if float(r.get("score", 0)) >= SCORE:
            prf[int(r["frame_id"])].append((float(r["world_x"]), float(r["world_y"]), cls(r.get("class_name"))))
    except: pass

errs = []; errs_cls = defaultdict(list); matched = 0; ngt = 0; npred = 0; samples = []
gx_all=[]; px_all=[]
for f in set(list(gtf) + list(prf)):
    G = gtf.get(f, []); P = prf.get(f, [])
    ngt += len(G); npred += len(P)
    gx_all += [(g[0],g[1]) for g in G]; px_all += [(p[0],p[1]) for p in P]
    pairs = sorted((math.hypot(g[0]-p[0], g[1]-p[1]), gi, pi)
                   for gi, g in enumerate(G) for pi, p in enumerate(P)
                   if g[2] == p[2] and math.hypot(g[0]-p[0], g[1]-p[1]) <= GATE)
    uG=set(); uP=set()
    for d, gi, pi in pairs:
        if gi in uG or pi in uP: continue
        uG.add(gi); uP.add(pi); errs.append(d); errs_cls[G[gi][2]].append(d); matched += 1
        if len(samples) < 5: samples.append((f, G[gi], P[pi], d))

print(f"run={run.name}  score>={SCORE} gate={GATE}m")
print(f"  matched TPs={matched}  GT(score-any)={ngt}  pred(score>={SCORE})={npred}")
if errs:
    print(f"  LIVE loc-MAE (matched) = {statistics.mean(errs):.3f} m   (median {statistics.median(errs):.3f})")
    for c in ("vehicle","person"):
        if errs_cls[c]: print(f"    {c}: MAE={statistics.mean(errs_cls[c]):.3f}m (n={len(errs_cls[c])})")
if gx_all and px_all:
    print(f"  sanity: GT world x[{min(a for a,_ in gx_all):.0f},{max(a for a,_ in gx_all):.0f}] "
          f"y[{min(b for _,b in gx_all):.0f},{max(b for _,b in gx_all):.0f}] | "
          f"PRED x[{min(a for a,_ in px_all):.0f},{max(a for a,_ in px_all):.0f}] "
          f"y[{min(b for _,b in px_all):.0f},{max(b for _,b in px_all):.0f}]")
print("  sample matches (frame: GT(x,y,cls) vs PRED(x,y,cls) dist):")
for f,g,p,d in samples:
    print(f"    f{f}: GT({g[0]:.1f},{g[1]:.1f},{g[2]}) PRED({p[0]:.1f},{p[1]:.1f},{p[2]}) d={d:.2f}m")
print("\n  OFFLINE reference (no-AE u8): loc 0.95m, obj-recall 0.878, ped-recall 0.853")
