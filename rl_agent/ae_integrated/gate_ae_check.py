#!/usr/bin/env python3
"""GATE for an integrated-AE model: does it retain BOTH segmentation AND localization (the bolt-on failed
loc)? Targets vs the M' baseline, pedestrian recall raised to >=0.80. Exit 0 iff the loc-recovery
HYPOTHESIS holds (loc<=1.6m AND mIoU>=0.80) -> the orchestrator only proceeds to 32/128 if AE-64 recovers.
Usage: gate_ae_check.py <metrics.json>"""
import json, sys
m = json.load(open(sys.argv[1]))
V = {
    "mIoU":        (m.get("miou"),                          0.82,  ">="),
    "vehicle_IoU": (m.get("vehicle_iou"),                   0.91,  ">="),
    "person_IoU":  (m.get("person_iou"),                    0.55,  ">="),
    "ped_recall":  (m.get("learned_person_object_recall"),  0.80,  ">="),   # RAISED target
    "obj_recall":  (m.get("learned_object_recall"),         0.80,  ">="),
    "loc_m":       (m.get("learned_global_xy_mae_m"),       1.50,  "<="),
    "ped_loc_m":   (m.get("learned_person_global_xy_mae_m"),1.70,  "<="),
    "dim_m":       (m.get("learned_dimension_mae_m"),       0.24,  "<="),
}
print("GATE (integrated AE) vs targets  [M' ref: mIoU 0.841 / veh 0.933 / ped-rec 0.787 / loc 1.21m]")
allok = True
for k, (v, t, op) in V.items():
    if v is None:
        print(f"  {k:12} MISSING"); allok = False; continue
    ok = (v >= t) if op == ">=" else (v <= t)
    allok = allok and ok
    print(f"  {k:12} {v:.3f} {op} {t}  -> {'PASS' if ok else 'FAIL'}")
loc = m.get("learned_global_xy_mae_m", 9); miou = m.get("miou", 0)
recovered = (loc <= 1.6) and (miou >= 0.80)      # the make-or-break: loc recovered (vs bolt-on 2.15m)
print(f"\n  ALL-TARGETS: {'PASS' if allok else 'REVIEW (some target missed)'}")
print(f"  HYPOTHESIS (loc recovered): {'RECOVERED' if recovered else 'FAILED'}  "
      f"(loc={loc:.2f}m miou={miou:.3f})")
sys.exit(0 if recovered else 1)
