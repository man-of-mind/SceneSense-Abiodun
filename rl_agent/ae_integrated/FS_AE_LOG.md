[2026-07-13 23:58:10] ===== AE-FROM-PHASE-1 AUTONOMOUS RUN START =====
[2026-07-13 23:58:10] ===== AE-32 FROM-PHASE-1 BUILD =====
wrote /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/rl_agent/ae_integrated/fs_stage1_ae32.json
wrote /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/rl_agent/ae_integrated/fs_stage2_ae32.json
wrote /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/rl_agent/ae_integrated/fs_phase3_ae32.json
[2026-07-13 23:58:10]   TRAIN fs_stage1_ae32 (budget 3.0h)
extracted 12 AE tensors -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/ae_integrated_fs_20260713/ae32/stage1/ae_extracted.pt
[2026-07-14 01:41:46]   TRAIN fs_stage2_ae32 (budget 3.0h)
extracted 12 AE tensors -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/ae_integrated_fs_20260713/ae32/stage2/ae_extracted.pt
[2026-07-14 03:19:10]   TRAIN fs_phase3_ae32 (budget 3.0h)
[2026-07-14 03:51:10]   GATE-eval fs_phase3_ae32 (clean; AE in forward)
  fs_ae32: mIoU 0.758 veh 0.850 ped-rec 0.874 obj-rec 0.899 loc 0.95m
[2026-07-14 03:53:48] --- COMPARE fs AE-32 vs current warm-started AE-32 ---
            miou     veh     ped     obj     loc
  fs      0.758  0.850  0.874  0.899  0.95m
  current 0.822  0.916  0.863  0.902  0.88m
  delta   -0.064  -0.066  +0.010  -0.002  -0.07m(lower=better)
VERDICT: NOT clearly better -> AE-from-phase-1 adds no advantage; warm-started stands
[2026-07-14 03:53:48] AE-32 from-phase-1 NOT clearly better (cmp rc=1) -> per plan, NOT training 64/128.
[2026-07-14 03:53:48]   Conclusion: AE-from-phase-1 adds no advantage; the warm-started (phase-3) models stand.
[2026-07-14 03:53:48] ===== AE-FROM-PHASE-1 RUN END =====
FS_AE_DONE OK
