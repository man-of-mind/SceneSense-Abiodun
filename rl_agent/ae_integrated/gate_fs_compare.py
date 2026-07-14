#!/usr/bin/env python3
"""Decide whether the AE-from-phase-1 (from-scratch) model is CLEARLY better than the
current warm-started model of the same bottleneck. Prints a verdict and exits:
  0 = CLEARLY BETTER  (proceed to train the other bottlenecks)
  1 = NOT clearly better (AE-from-phase-1 buys nothing -> warm-started version stands)
  2 = missing metrics

Rule ("clearly better", conservative so we only spend more compute on a real win):
  - loc improves by >= 0.10 m  OR  mIoU improves by >= 0.015 (1.5 pts)
  - AND no regression worse than 1% (absolute 0.01) on ped-recall or obj-recall
  - AND seg (mIoU) not worse by more than 1%
"""
import sys, json
from pathlib import Path

def load(p):
    p = Path(p)
    if not p.exists(): return None
    return json.load(open(p))

def row(m):
    return dict(miou=m["miou"], veh=m["vehicle_iou"],
               ped=m["learned_person_object_recall"], obj=m["learned_object_recall"],
               loc=m["learned_global_xy_mae_m"])

def main():
    fs_p, cur_p = sys.argv[1], sys.argv[2]
    fs, cur = load(fs_p), load(cur_p)
    if fs is None or cur is None:
        print(f"MISSING metrics (fs={fs_p} exists={fs is not None}; cur={cur_p} exists={cur is not None})")
        sys.exit(2)
    f, c = row(fs), row(cur)
    d = {k: f[k] - c[k] for k in f}
    print("            miou     veh     ped     obj     loc")
    print(f"  fs      {f['miou']:.3f}  {f['veh']:.3f}  {f['ped']:.3f}  {f['obj']:.3f}  {f['loc']:.2f}m")
    print(f"  current {c['miou']:.3f}  {c['veh']:.3f}  {c['ped']:.3f}  {c['obj']:.3f}  {c['loc']:.2f}m")
    print(f"  delta   {d['miou']:+.3f}  {d['veh']:+.3f}  {d['ped']:+.3f}  {d['obj']:+.3f}  {(-d['loc']):+.2f}m(lower=better)")

    loc_gain = -d["loc"]          # positive = better (lower loc)
    miou_gain = d["miou"]
    improved = (loc_gain >= 0.10) or (miou_gain >= 0.015)
    no_regress = (d["ped"] >= -0.01) and (d["obj"] >= -0.01) and (d["miou"] >= -0.01)
    if improved and no_regress:
        print("VERDICT: CLEARLY BETTER -> proceed to other bottlenecks")
        sys.exit(0)
    print("VERDICT: NOT clearly better -> AE-from-phase-1 adds no advantage; warm-started stands")
    sys.exit(1)

if __name__ == "__main__":
    main()
