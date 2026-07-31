#!/usr/bin/env python3
"""Road-state (straight / curve / intersection) x speed error breakdown for the UPLINK-ONLY lag.
Same opportunity-window data + origin-GT + classifier as the original staleness Result 1a
(make_roadstate_speed_plots_with_curves.py), but evaluated at the uplink-only capture->map lag L
(67 / 93 ms), not the retired OAI network anchors. Reuses the 829-obs speedsweep captures; no re-capture.
Requires CARLA up (Town10). Guards the known speed-confound by printing the road-state x speed count matrix.
"""
import csv, glob, math, statistics
from collections import defaultdict, Counter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AB = "/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"
OUT = f"{AB}/staleness/uplink_only_latency_budget/results"
NEAR = 25.0; GATE = 2.0; SCORE = 0.2
Y_SWEEP = [0.0, 0.067, 0.093, 0.136]                 # uplink-only capture->map lag anchors (fresh p50 / design / p95)
OPS = [("fresh p50 67ms", 67, "#009E73"), ("design 93ms", 93, "#D55E00")]
BANDS = [(0,4,"walk/slow"),(4,8,"~6 mph"),(8,12,"~10 mph"),(12,16,"~14 mph"),
         (16,20,"~18 mph"),(20,26,"~23 mph"),(26,40,"~28-32 mph")]
CURVE_DEG_PER_5M = 4.0
MIN_N = 10

def truthy(v): return str(v).strip().lower() in ("true","1","yes")
def gt_at(sm,t):
    if t<=sm[0][0]: return sm[0][1],sm[0][2]
    if t>=sm[-1][0]:
        (t0,x0,y0),(t1,x1,y1)=sm[-2][:3],sm[-1][:3]; dt=t1-t0
        return (x1+(x1-x0)*((t-t1)/dt),y1+(y1-y0)*((t-t1)/dt)) if dt>1e-6 else (sm[-1][1],sm[-1][2])
    for i in range(1,len(sm)):
        if sm[i][0]>=t:
            (t0,x0,y0),(t1,x1,y1)=sm[i-1][:3],sm[i][:3]; k=(t-t0)/max(1e-6,t1-t0)
            return x0+(x1-x0)*k,y0+(y1-y0)*k
    return sm[-1][1],sm[-1][2]
def band(mph):
    for lo,hi,lab in BANDS:
        if lo<=mph<hi: return lab
    return None
def is_sweep(run):
    for m in glob.glob(run+"/streams/*metrics.csv"):
        try:
            r=list(csv.DictReader(open(m)))
            if r and str(r[0].get("run_group","")).startswith("speedsweep_"): return True
        except: pass
    return False
runs=[r for r in sorted(glob.glob(f"{AB}/staleness/metrics_logs/scenesense_runs/*")) if is_sweep(r)]

obs=[]
using_origin_rows=0; total_gt_rows=0
for run in runs:
    gt=list(csv.DictReader(open(glob.glob(run+"/streams/*ground_truth.csv")[0])))
    pr=list(csv.DictReader(open(glob.glob(run+"/streams/*predictions.csv")[0])))
    traj=defaultdict(list)
    for r in gt:
        total_gt_rows+=1
        if r.get("origin_x") not in (None,""): using_origin_rows+=1
        try: traj[r["actor_id"]].append((float(r["carla_timestamp"]),float(r.get("origin_x") or r["world_x"]),
                                         float(r.get("origin_y") or r["world_y"]),int(r["frame_id"]),
                                         truthy(r.get("in_camera_frustum","")),float(r.get("distance_m",999))))
        except: pass
    for a in traj: traj[a].sort()
    prby=defaultdict(list)
    for r in pr:
        try:
            if float(r.get("score",0))>=SCORE: prby[int(r["frame_id"])].append((float(r["world_x"]),float(r["world_y"])))
        except: pass
    for aid,sm in traj.items():
        for i,(t,x,y,fid,infr,d) in enumerate(sm):
            if not (infr and d<=NEAR): continue
            P=prby.get(fid,[])
            if not P: continue
            dd=min(P,key=lambda p:math.hypot(p[0]-x,p[1]-y))
            if math.hypot(dd[0]-x,dd[1]-y)>GATE: continue
            j=min(max(1,i),len(sm)-1);(t0,x0,y0),(t1,x1,y1)=sm[j-1][:3],sm[j][:3]
            v=math.hypot(x1-x0,y1-y0)/max(1e-6,t1-t0)*2.237
            errs={Y: math.hypot(dd[0]-gt_at(sm,t+Y)[0],dd[1]-gt_at(sm,t+Y)[1]) for Y in Y_SWEEP}
            obs.append((v,x,y,errs))

# GUARDRAIL: GT origin convention
assert using_origin_rows>0, "no origin_x rows -- refusing to run on bbox-center (would re-introduce the ~1 m bug)"
print(f"GT origin convention: origin rows={using_origin_rows}/{total_gt_rows} (using ORIGIN, not bbox-center)")

import carla
client=carla.Client("127.0.0.1",2000); client.set_timeout(20.0); cmap=client.get_world().get_map()
def road_state(x,y):
    wp=cmap.get_waypoint(carla.Location(x=x,y=y,z=0.0),project_to_road=True,lane_type=carla.LaneType.Driving)
    if wp is None: return "unknown"
    if wp.is_junction: return "junction"
    nxt=wp.next(5.0)
    if not nxt: return "straight"
    dyaw=abs((nxt[0].transform.rotation.yaw - wp.transform.rotation.yaw + 180)%360 - 180)
    return "curve" if dyaw>CURVE_DEG_PER_5M else "straight"
tagged=[(v, band(v), road_state(x,y), errs) for (v,x,y,errs) in obs]
print(f"observations: {len(tagged)}  " + str(Counter(t[2] for t in tagged)))

# CONFOUND GUARD: road-state x speed-band count matrix (the original curve claim was withdrawn because
# the curve bin was 70% ~6 mph). Print so the confound is explicit before reading any road-state effect.
STATES=["straight","curve","junction"]
print("\n=== road-state x speed-band OBSERVATION COUNTS (confound check) ===")
print(f"  {'band':12s} " + " ".join(f"{s:>10s}" for s in STATES))
band_counts=defaultdict(lambda: Counter())
for v,bd,rs,_ in tagged:
    if bd: band_counts[bd][rs]+=1
for lo,hi,lab in BANDS:
    print(f"  {lab:12s} " + " ".join(f"{band_counts[lab][s]:>10d}" for s in STATES))

# per road-state plot + CSV
rows_csv=[]
for rs,fname,title in [("straight","uplink_roadstate_straight_speed","Straight road"),
                       ("curve","uplink_roadstate_curve_speed","Curve"),
                       ("junction","uplink_roadstate_intersection_speed","Intersection")]:
    sub=[(v,errs) for (v,bd,r,errs) in tagged if r==rs]
    fig,ax=plt.subplots(figsize=(8.0,5.6))
    cmap_v=plt.cm.viridis(np.linspace(0,0.92,len(BANDS)))
    print(f"\n=== {title} (n={len(sub)}), error(L) per speed at L(ms)={[int(y*1000) for y in Y_SWEEP]} ===")
    for bi,(lo,hi,lab) in enumerate(BANDS):
        cell=[errs for (v,errs) in sub if lo<=v<hi]
        if len(cell)<MIN_N:
            if cell: print(f"  {lab:12s} n={len(cell)} (skip, <{MIN_N})")
            continue
        curve=[statistics.mean(e[Y] for e in cell) for Y in Y_SWEEP]
        ax.plot([Y*1000 for Y in Y_SWEEP],curve,color=cmap_v[bi],lw=2.3,marker="o",ms=5,label=f"{lab} (n={len(cell)})")
        print(f"  {lab:12s} n={len(cell):3d}  " + " ".join(f"{c:.2f}" for c in curve))
        for Y,c in zip(Y_SWEEP,curve): rows_csv.append([rs,lab,len(cell),int(Y*1000),round(c,3)])
    yt=ax.get_ylim()[1] or 1.0
    for lab,Yms,col in OPS:
        ax.axvline(Yms,color=col,ls=":",lw=1.5,alpha=0.85); ax.text(Yms+2,yt*0.32,lab,rotation=90,va="top",ha="left",fontsize=8,color=col)
    ax.set_xlim(0,Y_SWEEP[-1]*1000); ax.margins(x=0); ax.set_ylim(0,None); ax.grid(alpha=0.25)
    ax.set_xlabel("uplink-only lag L (ms) = capture → map"); ax.set_ylabel("localization error (m)")
    ax.set_title(f"{title}: loc error vs uplink-only lag, per speed (Town10, ≤25 m)", fontweight="bold", fontsize=12)
    ax.legend(fontsize=8.5,frameon=False,loc="upper left")
    fig.tight_layout()
    for ext in ("pdf","png"): fig.savefig(f"{OUT}/plots/{fname}.{ext}",bbox_inches="tight",dpi=200)
    print(f"  wrote results/plots/{fname}.pdf/.png")

with open(f"{OUT}/roadstate_error_by_speed.csv","w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["road_state","speed_band","n","L_ms","loc_error_m"]); w.writerows(rows_csv)
print(f"\nwrote results/roadstate_error_by_speed.csv ({len(rows_csv)} rows)")
