#!/usr/bin/env python3
"""Post-hoc breakdown of the staleness opportunity-window observations by (a) ROAD STATE (junction / curve /
straight, from the CARLA Town10 map at each target's GT position) and (b) DETECTION DISTANCE (ego->NPC range).
No re-capture: reads the existing speed-sweep runs. Reports distance distribution + road-state mix + error(Y)
split by road state. Requires CARLA up on 127.0.0.1:2000 (Town10 loaded)."""
import csv, glob, math, statistics, sys
from collections import defaultdict, Counter

AB = "/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"
NEAR = 25.0; GATE = 2.0; SCORE = 0.2
Y_KEYS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.269]   # full latency sweep for the split plot
OPS = [("loopback ~50ms", 50, "#009E73"), ("AE-128 ~105ms", 105, "#0072B2"), ("no-AE ~267ms", 267, "#D55E00")]
CURVE_DEG_PER_5M = 8.0   # |yaw change| over 5 m above this = "curve"; below = "straight" (non-junction)

def truthy(v): return str(v).strip().lower() in ("true","1","yes")
def gt_at(sm, t):
    if t <= sm[0][0]: return sm[0][1], sm[0][2]
    if t >= sm[-1][0]:
        (t0,x0,y0),(t1,x1,y1)=sm[-2][:3],sm[-1][:3]; dt=t1-t0
        return (x1+(x1-x0)*((t-t1)/dt), y1+(y1-y0)*((t-t1)/dt)) if dt>1e-6 else (sm[-1][1],sm[-1][2])
    for i in range(1,len(sm)):
        if sm[i][0]>=t:
            (t0,x0,y0),(t1,x1,y1)=sm[i-1][:3],sm[i][:3]; k=(t-t0)/max(1e-6,t1-t0)
            return x0+(x1-x0)*k, y0+(y1-y0)*k
    return sm[-1][1], sm[-1][2]

# ---- collect opportunity-window observations from the speed-sweep runs ----
def is_sweep(run):
    for m in glob.glob(run+"/streams/*metrics.csv"):
        try:
            r=list(csv.DictReader(open(m)))
            if r and str(r[0].get("run_group","")).startswith("speedsweep_"): return True
        except: pass
    return False
runs=[r for r in sorted(glob.glob(f"{AB}/staleness/metrics_logs/scenesense_runs/*")) if is_sweep(r)]
print(f"speed-sweep runs: {len(runs)}")

obs=[]           # (dist_m, gt_x, gt_y, {Y:err})
all_inview=[]    # distances of ALL in-frustum GT (context: full range the ego sees NPCs)
for run in runs:
    gt=list(csv.DictReader(open(glob.glob(run+"/streams/*ground_truth.csv")[0])))
    pr=list(csv.DictReader(open(glob.glob(run+"/streams/*predictions.csv")[0])))
    traj=defaultdict(list)
    for r in gt:
        try:
            infr=truthy(r.get("in_camera_frustum","")); d=float(r.get("distance_m",999))
            traj[r["actor_id"]].append((float(r["carla_timestamp"]),float(r.get("origin_x") or r["world_x"]),
                                        float(r.get("origin_y") or r["world_y"]),int(r["frame_id"]),infr,d))
            if infr and d<=60: all_inview.append(d)
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
            errs={Y: math.hypot(dd[0]-gt_at(sm,t+Y)[0], dd[1]-gt_at(sm,t+Y)[1]) for Y in Y_KEYS}
            obs.append((d, x, y, errs))
print(f"matched opportunity-window observations (<{NEAR:.0f}m): {len(obs)}")

# ---- (b) DISTANCE distribution ----
def pct(v,p): v=sorted(v); return v[min(len(v)-1,int(p/100*len(v)))]
dm=[o[0] for o in obs]
print("\n=== DETECTION DISTANCE (ego camera -> NPC), matched observations used in the plot ===")
print(f"  min={min(dm):.1f}  p10={pct(dm,10):.1f}  p25={pct(dm,25):.1f}  MEDIAN={statistics.median(dm):.1f}  "
      f"p75={pct(dm,75):.1f}  p90={pct(dm,90):.1f}  max={max(dm):.1f} m   (n={len(dm)})")
print(f"  (analysis gate = in camera view AND <= {NEAR:.0f} m)")
if all_inview:
    print(f"  context — ALL in-view NPCs (no near gate): median={statistics.median(all_inview):.1f}m, "
          f"90% within {pct(all_inview,90):.0f}m, max seen {max(all_inview):.0f}m")

# ---- (a) ROAD STATE from CARLA map (classify each observation's target position) ----
try:
    import carla
    client=carla.Client("127.0.0.1",2000); client.set_timeout(20.0)
    cmap=client.get_world().get_map()
    def road_state(x,y):
        wp=cmap.get_waypoint(carla.Location(x=x,y=y,z=0.0), project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None: return "unknown"
        if wp.is_junction: return "junction"
        nxt=wp.next(5.0)
        if not nxt: return "straight"
        dyaw=abs((nxt[0].transform.rotation.yaw - wp.transform.rotation.yaw + 180)%360 - 180)
        return "curve" if dyaw>CURVE_DEG_PER_5M else "straight"
    by_state=defaultdict(list)   # state -> list of errs dicts
    counts=Counter()
    for d,x,y,errs in obs:
        st=road_state(x,y); counts[st]+=1; by_state[st].append(errs)
    total=sum(counts.values())
    print("\n=== ROAD STATE mix of the observations (target's location on Town10 map) ===")
    for st in ("junction","curve","straight","unknown"):
        if counts[st]: print(f"  {st:9s}: {counts[st]:4d}  ({100*counts[st]/total:.0f}%)")
    def at(L,Y): return statistics.mean(e[Y] for e in L)
    print("\n=== error(Y) split by road state (m) ===")
    print(f"  {'road state':12s} {'n':>4s}  " + " ".join(f"{int(Y*1000):>4d}ms" for Y in Y_KEYS))
    curves={}
    for st in ("straight","curve","junction"):
        L=by_state.get(st,[])
        if len(L)<10:
            if L: print(f"  {st:12s} {len(L):>4d}  (too few for a stable split)")
            continue
        curves[st]=[at(L,Y) for Y in Y_KEYS]
        print(f"  {st:12s} {len(L):>4d}  " + " ".join(f"{v:>5.2f}" for v in curves[st]))
    print("\n(Note: 'curve' vs 'straight' from lane yaw-change > 8 deg / 5 m; 'junction' = waypoint.is_junction.)")

    # ---- plot: error(Y) for straight vs intersection ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    STYLE={"straight":("#0072B2","straight road"),"junction":("#D55E00","intersection"),"curve":("#009E73","curve")}
    fig,ax=plt.subplots(figsize=(7.6,5.4))
    for st,c in curves.items():
        col,lab=STYLE.get(st,("#555",st))
        ax.plot([Y*1000 for Y in Y_KEYS], c, color=col, lw=2.6, marker="o", ms=6, label=f"{lab} (n={len(by_state[st])})")
    yt=ax.get_ylim()[1]
    for lab,Yms,col in OPS:
        ax.axvline(Yms,color=col,ls=":",lw=1.5,alpha=0.8); ax.text(Yms+3,yt*0.30,lab,rotation=90,va="top",ha="left",fontsize=7.5,color=col)
    ax.set_xlim(0,Y_KEYS[-1]*1000); ax.margins(x=0); ax.set_ylim(0,None); ax.grid(alpha=0.25)
    ax.set_xlabel("latency Y (ms) = capture→inference"); ax.set_ylabel("localization error (m)")
    ax.set_title("Localization error vs latency, by road state (Town10, ≤25 m)", fontweight="bold")
    ax.legend(fontsize=9.5,frameon=False,loc="upper left")
    fig.tight_layout()
    for ext in ("pdf","png"): fig.savefig(f"staleness/plots/roadstate_error_latency.{ext}", bbox_inches="tight", dpi=200)
    print("\nwrote staleness/plots/roadstate_error_latency.pdf/.png")
except Exception as e:
    print(f"\n[road-state] CARLA query unavailable ({e}). Distance stats above still valid.")
