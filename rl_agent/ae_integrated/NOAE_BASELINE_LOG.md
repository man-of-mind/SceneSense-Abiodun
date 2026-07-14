[2026-07-13 14:01:03] ===== NO-AE BASELINE (joint recipe, fair control) START =====
[2026-07-13 15:38:10] === GATE-eval no-AE baseline (clean) ===
GATE (integrated AE) vs targets  [M' ref: mIoU 0.841 / veh 0.933 / ped-rec 0.787 / loc 1.21m]
  mIoU         0.839 >= 0.82  -> PASS
  vehicle_IoU  0.931 >= 0.91  -> PASS
  person_IoU   0.590 >= 0.55  -> PASS
  ped_recall   0.853 >= 0.8  -> PASS
  obj_recall   0.878 >= 0.8  -> PASS
  loc_m        0.948 <= 1.5  -> PASS
  ped_loc_m    1.079 <= 1.7  -> PASS
  dim_m        0.149 <= 0.24  -> PASS

  ALL-TARGETS: PASS
  HYPOTHESIS (loc recovered): RECOVERED  (loc=0.95m miou=0.839)
  noae_baseline: mIoU 0.839 veh 0.931 ped-rec 0.853 obj-rec 0.878 loc 0.95m
[2026-07-13 15:40:45] ===== NO-AE BASELINE END =====
NOAE_BASELINE_DONE OK
