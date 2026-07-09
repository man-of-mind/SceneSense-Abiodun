#!/usr/bin/env python3
"""GATE A: does drop-aware M' (evaluated at q=0, clean) match the 200k baseline?

Compares M' test metrics against the actual det_pps200000_v2 eval (same eval config:
test split, thr 0.20, nms 2, topk 120, match 5m, range-gate 40m). The gate is
"no meaningful regression vs baseline" rather than absolute targets, so it stays correct
even as the exact baseline numbers evolve. IoU/recall may drop by at most REL_TOL;
localization MAE may rise by at most REL_TOL.

Usage: gate_a_check.py <mprime_metrics.json> [baseline_metrics.json]
"""
import json, sys

BASELINE = ("/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/"
            "neu_collab/abiodun/experiments/autonomous_arch_runs_20260625/det_pps200000_v2/"
            "eval_best_thr020/metrics/test_fusion_evaluation_metrics.json")

REL_TOL = 0.01  # allow 1% relative slack (drop-aware robustness may cost a hair at q=0)

# name -> (metric key, direction)  higher_is_better=True for iou/recall, False for MAE
CHECKS = [
    ("mIoU",                    "miou",                          True),
    ("vehicle_IoU",             "vehicle_iou",                   True),
    ("person_IoU",              "person_iou",                    True),
    ("object_recall",           "learned_object_recall",         True),
    ("person_object_recall",    "learned_person_object_recall",  True),
    ("global_xy_mae_m",         "learned_global_xy_mae_m",       False),
    ("person_xy_mae_m",         "learned_person_global_xy_mae_m", False),
    ("dimension_mae_m",         "learned_dimension_mae_m",        False),
]

def main():
    mp = json.load(open(sys.argv[1]))
    base = json.load(open(sys.argv[2] if len(sys.argv) > 2 else BASELINE))
    print(f"GATE A: M'(q=0) vs 200k baseline det_pps200000_v2  (rel_tol={REL_TOL:.0%})")
    print(f"{'metric':24} {'M-prime':>10} {'baseline':>10} {'allowed':>10}  verdict")
    ok = True
    for name, key, higher in CHECKS:
        v, b = mp.get(key), base.get(key)
        if v is None or b is None:
            print(f"{name:24} {'--':>10} {'--':>10} {'--':>10}  MISSING({key})"); ok = False; continue
        if higher:
            allowed = b * (1 - REL_TOL)
            passed = v >= allowed
        else:
            allowed = b * (1 + REL_TOL)
            passed = v <= allowed
        ok = ok and passed
        print(f"{name:24} {v:>10.4f} {b:>10.4f} {allowed:>10.4f}  {'PASS' if passed else 'FAIL'}")
    print(f"\nGATE A RESULT: {'PASS -> build sweeps/AE on M-prime' if ok else 'FAIL/REVIEW -> inspect before proceeding'}")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
