#!/usr/bin/env python3
"""CPU package energy per frame (RAPL), FULL vs FRONT(split), 1 thread — the cell E2 could not measure
(RAPL was root-only; read here via `sudo -n cat`). torch runs as the user; only the counter reads use sudo."""
import torch, time, subprocess, json
from pathlib import Path
from common_setup import FrontWrapper, build_full_model, get_real_input

RAPL = "/sys/class/powercap/intel-rapl:0/energy_uj"
RNG_P = "/sys/class/powercap/intel-rapl:0/max_energy_range_uj"
def _read(p): return int(subprocess.run(["sudo","-n","cat",p],capture_output=True,text=True).stdout.strip())
RNG = _read(RNG_P)
def uj(): return _read(RAPL)
def dlt(a,b): d=b-a; return d+RNG if d<0 else d   # RAPL counter wraps

torch.set_num_threads(1)
dev = torch.device("cpu")
model, input_size, _ = build_full_model(dev)
x, _, _ = get_real_input(dev, input_size)
front = FrontWrapper(model).eval()
mods = {"FULL (whole model)": model, "FRONT (split: backbone only)": front}

# idle baseline
with torch.no_grad():
    t0=time.time(); e0=uj(); time.sleep(10); e1=uj(); t1=time.time()
idle_W = dlt(e0,e1)/1e6/(t1-t0)

def run(mod, secs=15):
    with torch.no_grad():
        for _ in range(5): mod(x)                     # warmup
        n=0; t0=time.time(); e0=uj()
        while time.time()-t0 < secs:
            mod(x); n+=1
        e1=uj(); t1=time.time()
    w=t1-t0; J=dlt(e0,e1)/1e6
    lat=w/n
    return dict(frames=n, wall_s=round(w,1), pkg_J=round(J,1), busy_W=round(J/w,1),
                latency_ms=round(lat*1000,1), fps=round(n/w,2),
                J_per_frame=round(J/n,3), active_J_per_frame=round(J/n - idle_W*lat,3))

res = {name: run(mod) for name,mod in mods.items()}
print(f"idle_W = {idle_W:.1f}")
for k,v in res.items(): print(k,v)
Path("results").mkdir(exist_ok=True)
json.dump({"domain":"package-0","idle_W":round(idle_W,2),"threads":1,"results":res},
          open("results/E2b_cpu_energy.json","w"), indent=1)
print("wrote results/E2b_cpu_energy.json")
