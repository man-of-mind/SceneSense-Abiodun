wrote /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/rl_agent/ae_integrated/ae64_integrated.json
wrote /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/rl_agent/ae_integrated/ae32_integrated.json
wrote /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/rl_agent/ae_integrated/ae128_integrated.json
[2026-07-10 20:06:20] ===== INTEGRATED-AE AUTONOMOUS RUN START =====
[2026-07-10 20:06:20] === TRAIN ae64_integrated (warm-start M', AE end-to-end, drop-aware) ===
[2026-07-10 21:47:13] === GATE-eval ae64 (clean; AE runs inside forward) ===
GATE (integrated AE) vs targets  [M' ref: mIoU 0.841 / veh 0.933 / ped-rec 0.787 / loc 1.21m]
  mIoU         0.825 >= 0.82  -> PASS
  vehicle_IoU  0.916 >= 0.91  -> PASS
  person_IoU   0.562 >= 0.55  -> PASS
  ped_recall   0.864 >= 0.8  -> PASS
  obj_recall   0.900 >= 0.8  -> PASS
  loc_m        0.865 <= 1.5  -> PASS
  ped_loc_m    1.037 <= 1.7  -> PASS
  dim_m        0.148 <= 0.24  -> PASS

  ALL-TARGETS: PASS
  HYPOTHESIS (loc recovered): RECOVERED  (loc=0.87m miou=0.825)
[2026-07-10 21:49:57] AE-64 RECOVERED localization -> proceeding to AE-32 and AE-128
[2026-07-10 21:49:57] === TRAIN ae32_integrated (warm-start M', AE end-to-end, drop-aware) ===
[2026-07-10 23:31:19] === GATE-eval ae32 (clean; AE runs inside forward) ===
GATE (integrated AE) vs targets  [M' ref: mIoU 0.841 / veh 0.933 / ped-rec 0.787 / loc 1.21m]
  mIoU         0.822 >= 0.82  -> PASS
  vehicle_IoU  0.916 >= 0.91  -> PASS
  person_IoU   0.554 >= 0.55  -> PASS
  ped_recall   0.863 >= 0.8  -> PASS
  obj_recall   0.902 >= 0.8  -> PASS
  loc_m        0.878 <= 1.5  -> PASS
  ped_loc_m    1.060 <= 1.7  -> PASS
  dim_m        0.146 <= 0.24  -> PASS

  ALL-TARGETS: PASS
  HYPOTHESIS (loc recovered): RECOVERED  (loc=0.88m miou=0.822)
[2026-07-10 23:34:04] === TRAIN ae128_integrated (warm-start M', AE end-to-end, drop-aware) ===
[2026-07-11 01:15:26] === GATE-eval ae128 (clean; AE runs inside forward) ===
GATE (integrated AE) vs targets  [M' ref: mIoU 0.841 / veh 0.933 / ped-rec 0.787 / loc 1.21m]
  mIoU         0.819 >= 0.82  -> FAIL
  vehicle_IoU  0.914 >= 0.91  -> PASS
  person_IoU   0.546 >= 0.55  -> FAIL
  ped_recall   0.883 >= 0.8  -> PASS
  obj_recall   0.908 >= 0.8  -> PASS
  loc_m        0.871 <= 1.5  -> PASS
  ped_loc_m    1.057 <= 1.7  -> PASS
  dim_m        0.146 <= 0.24  -> PASS

  ALL-TARGETS: REVIEW (some target missed)
  HYPOTHESIS (loc recovered): RECOVERED  (loc=0.87m miou=0.819)
[2026-07-11 01:18:12] --- SUMMARY (clean accuracy of integrated-AE models) ---
  ae64: mIoU 0.825 veh 0.916 ped-rec 0.864 obj-rec 0.900 loc 0.87m
  ae32: mIoU 0.822 veh 0.916 ped-rec 0.863 obj-rec 0.902 loc 0.88m
  ae128: mIoU 0.819 veh 0.914 ped-rec 0.883 obj-rec 0.908 loc 0.87m
[2026-07-11 01:18:12] ===== INTEGRATED-AE RUN END =====
AE_INTEGRATED_DONE OK
