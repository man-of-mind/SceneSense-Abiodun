#!/usr/bin/env python3
"""Single-frame vs TRACKED localization error, across REAL CARLA FPS captures (run-groups fps_5/10/20/30).
Per GT actor: match detections, run a constant-velocity Kalman (smooths noise + estimates velocity), then at
each detection time predict FORWARD by Y (velocity extrapolation) to cancel staleness. Compare:
  single-frame error(Y) = ||raw_detection(t) - GT(t+Y)||   (memoryless; grows with Y)
  tracked error(Y)      = ||KF_predict(t+Y)   - GT(t+Y)||   (fused + predicted; should beat single-frame,
                                                             and IMPROVE with FPS via more updates)
Also reports per-frame accuracy (Y=0, single-frame) vs FPS -> verifies the model is FPS-robust (trained @10)."""
import csv, sys, math, glob, os, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

GATE = 4.0
Y_SWEEP = [0.0, 0.05, 0.10, 0.15, 0.20, 0.269]
BINS = [(0.5, 2.5, "pedestrian"), (2.5, 6.0, "~5-13 mph"), (6.0, 12.0, "~13-27 mph")]
OUT = Path("staleness/plots"); OUT.mkdir(parents=True, exist_ok=True)

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

def analyze(run):
    s=Path(run)/"streams"
    gt=list(csv.DictReader(open(next(s.glob("*object_ground_truth.csv")))))
    pr=list(csv.DictReader(open(next(s.glob("*object_predictions.csv")))))
    # GT trajectory = actor ORIGIN (training convention), fallback to bbox-center world_x for legacy runs.
    def _gx(r): return float(r.get("origin_x") or r["world_x"])
    def _gy(r): return float(r.get("origin_y") or r["world_y"])
    traj=defaultdict(list)
    for r in gt:
        try: traj[r["actor_id"]].append((float(r["carla_timestamp"]),_gx(r),_gy(r),int(r["frame_id"])))
        except: pass
    for a in traj: traj[a].sort()
    prby=defaultdict(list)
    for r in pr:
        try: prby[int(r["frame_id"])].append((float(r["world_x"]),float(r["world_y"])))
        except: pass
    R=np.diag([2.5**2,2.5**2]); qa=3.0**2  # meas noise ~ floor; accel process noise
    H=np.array([[1,0,0,0],[0,1,0,0]],float)
    agg=defaultdict(lambda: defaultdict(lambda: {"sf":[], "tr":[]}))
    for aid, sm in traj.items():
        dets=[]
        for i,(t,x,y,fid) in enumerate(sm):
            preds=prby.get(fid,[])
            if not preds: continue
            z=min(preds,key=lambda p:math.hypot(p[0]-x,p[1]-y))
            if math.hypot(z[0]-x,z[1]-y)>GATE: continue
            j=min(max(1,i),len(sm)-1); (t0,x0,y0),(t1,x1,y1)=sm[j-1][:3],sm[j][:3]
            v=math.hypot(x1-x0,y1-y0)/max(1e-6,t1-t0)
            dets.append((t,z[0],z[1],v))
        if len(dets)<2: continue
        # constant-velocity Kalman over this actor's detections
        x=np.array([dets[0][1],dets[0][2],0,0],float); P=np.diag([R[0,0],R[1,1],25,25]); tprev=dets[0][0]
        for k,(t,zx,zy,v) in enumerate(dets):
            dt=t-tprev; tprev=t
            if dt>0:
                F=np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]],float)
                G=np.array([dt*dt/2,dt*dt/2,dt,dt]); Q=np.outer(G,G)*qa
                x=F@x; P=F@P@F.T+Q
            z=np.array([zx,zy]); S=H@P@H.T+R; K=P@H.T@np.linalg.inv(S)
            x=x+K@(z-H@x); P=(np.eye(4)-K@H)@P
            b=next((bi for bi,(lo,hi,_) in enumerate(BINS) if lo<=v<hi),None)
            if b is None: continue
            for Y in Y_SWEEP:
                gx,gy=gt_at(sm,t+Y)
                agg[b][Y]["sf"].append(math.hypot(zx-gx,zy-gy))
                px,py=x[0]+x[2]*Y, x[1]+x[3]*Y
                agg[b][Y]["tr"].append(math.hypot(px-gx,py-gy))
    return agg

# find fps runs
runs={}
for d in sorted(glob.glob("staleness/metrics_logs/scenesense_runs/2026*fusion_tl_14"), key=os.path.getmtime):
    mc=glob.glob(d+"/streams/*metrics.csv")
    if not mc: continue
    rows=list(csv.DictReader(open(mc[0])))
    if rows:
        m=re.match(r"fps_(\d+)", rows[0].get("run_group",""))
        if m: runs[int(m.group(1))]=d
print("FPS runs found:", sorted(runs))
data={f: analyze(runs[f]) for f in sorted(runs)}

# FPS-robustness: per-frame single-frame Y=0 error per FPS (fast-car bin)
print("\n--- FPS-robustness: per-frame model error (Y=0, single-frame), fast-car bin ---")
for f in sorted(data):
    b=2
    v=data[f].get(b,{}).get(0.0,{}).get("sf",[])
    print(f"  {f:>2} FPS: floor={sum(v)/len(v):.2f}m (n={len(v)})" if v else f"  {f} FPS: no data")

# plot: single-frame vs tracked error(Y), fast-car bin, per FPS
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13.5,5.4))
cmap={5:"#9ecae1",10:"#4292c6",20:"#08519c",30:"#08306b"}
b=2
for f in sorted(data):
    row=data[f].get(b,{})
    sf=[np.mean(row[Y]["sf"]) if row.get(Y,{}).get("sf") else np.nan for Y in Y_SWEEP]
    tr=[np.mean(row[Y]["tr"]) if row.get(Y,{}).get("tr") else np.nan for Y in Y_SWEEP]
    ax1.plot([Y*1000 for Y in Y_SWEEP], sf, color=cmap.get(f,"#888"), ls="--", lw=1.6, marker="s", ms=3)
    ax1.plot([Y*1000 for Y in Y_SWEEP], tr, color=cmap.get(f,"#888"), ls="-", lw=2.2, marker="o", ms=4, label=f"{f} FPS (tracked)")
ax1.plot([],[],color="#555",ls="--",label="single-frame (any FPS)")
ax1.set_xlabel("latency Y (ms)"); ax1.set_ylabel("localization error (m)"); ax1.set_ylim(0,None); ax1.grid(alpha=0.25)
ax1.set_title("Tracked (solid) vs single-frame (dashed) — fast car", fontweight="bold", fontsize=12); ax1.legend(fontsize=8.5, frameon=False, ncol=2)
# tracked error vs FPS at fixed Y=100ms
Yi=Y_SWEEP.index(0.10)
for bi,(lo,hi,lab) in enumerate(BINS):
    fps=sorted(data); sfv=[np.mean(data[f][bi][0.10]["sf"]) if data[f].get(bi,{}).get(0.10,{}).get("sf") else np.nan for f in fps]
    trv=[np.mean(data[f][bi][0.10]["tr"]) if data[f].get(bi,{}).get(0.10,{}).get("tr") else np.nan for f in fps]
    ax2.plot(fps, trv, lw=2.2, marker="o", ms=4, label=f"{lab} (tracked)")
ax2.set_xlabel("camera FPS (real CARLA capture)"); ax2.set_ylabel("tracked error @ Y=100ms (m)"); ax2.set_ylim(0,None); ax2.grid(alpha=0.25)
ax2.set_title("Tracked error ↓ as FPS ↑ (more updates)", fontweight="bold", fontsize=12); ax2.legend(fontsize=9, frameon=False)
fig.suptitle("Temporal fusion (Kalman): tracking predicts forward to cancel staleness; more FPS = better velocity",fontsize=11.5,y=1.02)
fig.tight_layout(); fig.savefig(OUT/"tracked_vs_singleframe.pdf",bbox_inches="tight"); fig.savefig(OUT/"tracked_vs_singleframe.png",dpi=200,bbox_inches="tight")
print(f"\nwrote {OUT}/tracked_vs_singleframe.pdf/.png")

print("\n--- fast-car bin: single-frame vs tracked error(Y), per FPS ---")
for f in sorted(data):
    row=data[f].get(2,{})
    print(f"  {f:>2} FPS:  " + "  ".join(f"Y{int(Y*1000)}:{np.mean(row[Y]['sf']):.2f}/{np.mean(row[Y]['tr']):.2f}" for Y in Y_SWEEP if row.get(Y,{}).get('sf')) + "   (single-frame/tracked m)")
