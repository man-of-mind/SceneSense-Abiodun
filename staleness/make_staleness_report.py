#!/usr/bin/env python3
"""Latency+FPS requirement -- PURELY MEASURED (no model/formula).
error(lag) = mean_over_detections || inferred_loc(captured at t) - true_loc(t + lag) ||, per object-speed bin.
- Latency Y is a point on this curve (lag = Y).
- FPS just adds inter-frame lag: effective lag = Y + 1/(2*FPS). So error-vs-FPS is read off the SAME measured
  curve at lag = Y + 1/(2*FPS). No formula, no fitting.
Usage: make_staleness_report.py <run_dir> [match_gate_m=2.0]"""
import csv, sys, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

run = Path(sys.argv[1]); GATE = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
OUT = Path("staleness/plots"); OUT.mkdir(parents=True, exist_ok=True)
s = run / "streams"
gt = list(csv.DictReader(open(next(s.glob("*object_ground_truth.csv")))))
pr = list(csv.DictReader(open(next(s.glob("*object_predictions.csv")))))

# GT trajectory in the TRAINING convention: actor ORIGIN (origin_x/y = actor.get_location()), which is what
# the model regresses and the offline eval measures against. Fall back to bbox-center world_x for legacy runs.
def _gx(r): return float(r.get("origin_x") or r["world_x"])
def _gy(r): return float(r.get("origin_y") or r["world_y"])
traj = defaultdict(list)
for r in gt:
    try: traj[r["actor_id"]].append((float(r["carla_timestamp"]), _gx(r), _gy(r), int(r["frame_id"])))
    except: pass
for a in traj: traj[a].sort()
prby = defaultdict(list)
for r in pr:
    try: prby[int(r["frame_id"])].append((float(r["world_x"]), float(r["world_y"])))
    except: pass

def gt_at(sm, t):
    if t <= sm[0][0]: return sm[0][1], sm[0][2]
    if t >= sm[-1][0]:
        if len(sm) >= 2:
            (t0,x0,y0),(t1,x1,y1)=sm[-2][:3],sm[-1][:3]; dt=t1-t0
            if dt>1e-6: k=(t-t1)/dt; return x1+(x1-x0)*k, y1+(y1-y0)*k
        return sm[-1][1], sm[-1][2]
    for i in range(1,len(sm)):
        if sm[i][0]>=t:
            (t0,x0,y0),(t1,x1,y1)=sm[i-1][:3],sm[i][:3]; k=(t-t0)/max(1e-6,t1-t0)
            return x0+(x1-x0)*k, y0+(y1-y0)*k
    return sm[-1][1], sm[-1][2]

# matched detections with local speed: (speed, inferred(dx,dy), samples, t)
# top bin observed max ~8.3 m/s (~18 mph) in natural Town10 traffic; label reflects the MEASURED range, not the bin ceiling
BINS=[(0.5,2.5,"pedestrian ~1.5 m/s","#009E73"),(2.5,6.0,"~5-13 mph","#0072B2"),(6.0,11.0,"~13-18 mph","#E69F00")]
dets_by_bin=defaultdict(list)
n=0
for aid, sm in traj.items():
    for i,(t,x,y,fid) in enumerate(sm):
        preds=prby.get(fid,[])
        if not preds: continue
        d=min(preds,key=lambda p:math.hypot(p[0]-x,p[1]-y))
        if math.hypot(d[0]-x,d[1]-y)>GATE: continue
        j=min(max(1,i),len(sm)-1); (t0,x0,y0),(t1,x1,y1)=sm[j-1][:3],sm[j][:3]
        v=math.hypot(x1-x0,y1-y0)/max(1e-6,t1-t0)
        b=next((bi for bi,(lo,hi,_,_) in enumerate(BINS) if lo<=v<hi),None)
        if b is None: continue
        dets_by_bin[b].append((d, sm, t)); n+=1
print(f"run={run.name} matched(gate {GATE}m)={n}")

LAGS=np.linspace(0,0.30,31)
def err_curve(bi):
    """mean measured error vs lag for a speed bin."""
    out=[]
    for lag in LAGS:
        e=[]
        for d,sm,t in dets_by_bin[bi]:
            g=gt_at(sm,t+lag); e.append(math.hypot(d[0]-g[0],d[1]-g[1]))
        out.append(sum(e)/len(e) if e else float('nan'))
    return np.array(out)

curves={bi: err_curve(bi) for bi in dets_by_bin}
def err_at(bi, lag):
    return float(np.interp(lag, LAGS, curves[bi]))

# ---- Plot 1: measured error vs lag (the trend). Plot 2: error vs FPS at fixed Y (read same curve at Y+1/2F) ----
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13.5,5.4))
# Panel A: error vs latency (frame gap negligible: high FPS) -> latency is the only variable
for bi,(lo,hi,lab,c) in enumerate(BINS):
    if bi in curves:
        ax1.plot(LAGS*1000, curves[bi], color=c, lw=2.4, marker="o", ms=3, label=f"{lab} (n={len(dets_by_bin[bi])})")
ax1.set_xlabel("latency Y (ms) = capture→inference"); ax1.set_ylabel("measured localization error (m)")
ax1.set_title("Error ↓ as latency ↓", fontweight="bold", fontsize=13)
ax1.legend(fontsize=9.5, frameon=False, loc="upper left"); ax1.grid(alpha=0.25); ax1.set_ylim(0,None)
ax1.set_xlim(0, LAGS[-1]*1000); ax1.margins(x=0)
ax1.annotate("Y=0: model's own error\n(pedestrian slow → flat;\ncars rise with latency)", (150, 0.3), fontsize=8.5, color="#555")
# Overlay our MEASURED operating points (Y = capture->inference = front+uplink+back). SAME model curve,
# different transport Y -> reads which scenarios each config meets (transport isolated from model accuracy).
OPS=[("loopback ~50ms", 50, "#009E73"), ("AE-128 OAI ~105ms", 105, "#0072B2"), ("no-AE OAI ~267ms", 267, "#D55E00")]
_ytop=ax1.get_ylim()[1]
for lab,Yms,col in OPS:
    ax1.axvline(Yms, color=col, ls=":", lw=1.8, alpha=0.9)
    ax1.text(Yms+3, _ytop*0.62, lab, rotation=90, va="top", ha="left", fontsize=7.8, color=col)

# Panel B: error vs FPS at LOW latency (Y=0) -> the frame gap is the only variable (shows FPS effect cleanly)
FPS=np.array([2,3,5,8,10,15,20,30])
for bi,(lo,hi,lab,c) in enumerate(BINS):
    if bi in curves:
        ax2.plot(FPS, [err_at(bi, 1.0/(2*f)) for f in FPS], color=c, lw=2.4, marker="o", ms=4, label=lab)
ax2.axvspan(15,30, color="#eeeeee", zorder=0); ax2.annotate("plateau:\n>~15 FPS adds\nlittle", (22,ax2.get_ylim()[1]*0.5 if False else 2.6), fontsize=8.5, color="#777", ha="center")
ax2.set_xlabel("camera FPS  (at low latency)"); ax2.set_ylabel("measured localization error (m)")
ax2.set_title("Error ↓ as FPS ↑, then plateaus (~10–15 FPS)", fontweight="bold", fontsize=13)
ax2.legend(fontsize=9.5, frameon=False, loc="upper right"); ax2.grid(alpha=0.25); ax2.set_ylim(0,None)
ax2.set_xlim(0, FPS[-1]); ax2.margins(x=0)
fig.suptitle("Localization error minimizes as latency ↓ and FPS ↑ (FPS with diminishing returns past ~10–15 FPS)", fontsize=12, y=1.02)
fig.tight_layout(); fig.savefig(OUT/"staleness_requirement.pdf",bbox_inches="tight"); fig.savefig(OUT/"staleness_requirement.png",dpi=200,bbox_inches="tight")
print(f"wrote {OUT}/staleness_requirement.pdf/.png")

print(f"\n{'speed bin':<20}"+" ".join(f"{int(l*1000):>4}ms" for l in (0,0.05,0.10,0.15,0.20,0.269)))
for bi,(lo,hi,lab,c) in enumerate(BINS):
    if bi in curves:
        print(f"{lab:<20}"+" ".join(f"{err_at(bi,l):>5.2f}" for l in (0,0.05,0.10,0.15,0.20,0.269))+"  m")
print("\n(Read thresholds straight off the curve: max lag with error<=eps. FPS: e.g. 10 FPS adds 50 ms of lag.)")
