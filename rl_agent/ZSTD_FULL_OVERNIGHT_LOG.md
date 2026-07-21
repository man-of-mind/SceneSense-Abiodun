[2026-07-21 13:12:49] ===================== ZSTD FULL OVERNIGHT START =====================
[2026-07-21 13:12:49] --- Stage 1: zstd offline per-model eval ---
[2026-07-21 13:12:49] skip noae__clean (done)
[2026-07-21 13:12:49] skip noae__uint8__roi0.0 (done)
[2026-07-21 13:12:49] skip noae__uint8__roi0.3 (done)
[2026-07-21 13:12:49] skip noae__uint8__roi0.5 (done)
[2026-07-21 13:12:49] skip noae__uint6__roi0.0 (done)
[2026-07-21 13:12:49] skip noae__uint6__roi0.3 (done)
[2026-07-21 13:12:49] skip noae__uint6__roi0.5 (done)
[2026-07-21 13:12:49] skip noae__uint4__roi0.0 (done)
[2026-07-21 13:12:49] skip noae__uint4__roi0.3 (done)
[2026-07-21 13:12:49] skip noae__uint4__roi0.5 (done)
[2026-07-21 13:12:49] skip ae32__clean (done)
[2026-07-21 13:12:49] skip ae32__uint8__roi0.0 (done)
[2026-07-21 13:12:49] skip ae32__uint8__roi0.3 (done)
[2026-07-21 13:12:49] skip ae32__uint8__roi0.5 (done)
[2026-07-21 13:12:49] skip ae32__uint6__roi0.0 (done)
[2026-07-21 13:12:49] skip ae32__uint6__roi0.3 (done)
[2026-07-21 13:12:49] skip ae32__uint6__roi0.5 (done)
[2026-07-21 13:12:49] skip ae32__uint4__roi0.0 (done)
[2026-07-21 13:12:49] skip ae32__uint4__roi0.3 (done)
[2026-07-21 13:12:49] skip ae32__uint4__roi0.5 (done)
[2026-07-21 13:12:49] skip ae64__clean (done)
[2026-07-21 13:12:49] skip ae64__uint8__roi0.0 (done)
[2026-07-21 13:12:49] skip ae64__uint8__roi0.3 (done)
[2026-07-21 13:12:49] skip ae64__uint8__roi0.5 (done)
[2026-07-21 13:12:49] skip ae64__uint6__roi0.0 (done)
[2026-07-21 13:12:49] skip ae64__uint6__roi0.3 (done)
[2026-07-21 13:12:49] skip ae64__uint6__roi0.5 (done)
[2026-07-21 13:12:49] skip ae64__uint4__roi0.0 (done)
[2026-07-21 13:12:49] skip ae64__uint4__roi0.3 (done)
[2026-07-21 13:12:49] skip ae64__uint4__roi0.5 (done)
[2026-07-21 13:12:49] skip ae128__clean (done)
[2026-07-21 13:12:49] skip ae128__uint8__roi0.0 (done)
[2026-07-21 13:12:49] skip ae128__uint8__roi0.3 (done)
[2026-07-21 13:12:49] skip ae128__uint8__roi0.5 (done)
[2026-07-21 13:12:49] skip ae128__uint6__roi0.0 (done)
[2026-07-21 13:12:49] skip ae128__uint6__roi0.3 (done)
[2026-07-21 13:12:49] skip ae128__uint6__roi0.5 (done)
[2026-07-21 13:12:49] skip ae128__uint4__roi0.0 (done)
[2026-07-21 13:12:49] skip ae128__uint4__roi0.3 (done)
[2026-07-21 13:12:49] skip ae128__uint4__roi0.5 (done)
[2026-07-21 13:12:49] Stage 1 done: 40/40 eval metrics present
[2026-07-21 13:12:49] --- Stage 2: zstd full latency sweep ---
[2026-07-21 13:12:49] reusing existing CARLA
sweep 'loopback_ideal_zstd_FULL': 36 job(s), max_parallel=1, cwd=/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun
  [OK] noae_u4_roi0.0_zstd (50.4s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zstd_full/noae_u4_roi0.0_zstd
  [OK] noae_u4_roi0.3_zstd (47.9s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zstd_full/noae_u4_roi0.3_zstd
  [OK] noae_u4_roi0.5_zstd (47.7s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zstd_full/noae_u4_roi0.5_zstd
  [OK] noae_u6_roi0.0_zstd (48.2s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zstd_full/noae_u6_roi0.0_zstd
  [OK] noae_u6_roi0.3_zstd (47.0s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zstd_full/noae_u6_roi0.3_zstd
  [OK] noae_u6_roi0.5_zstd (48.3s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zstd_full/noae_u6_roi0.5_zstd
  [OK] noae_u8_roi0.0_zstd (46.8s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zstd_full/noae_u8_roi0.0_zstd
  [OK] noae_u8_roi0.3_zstd (48.5s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zstd_full/noae_u8_roi0.3_zstd
  [OK] noae_u8_roi0.5_zstd (46.2s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zstd_full/noae_u8_roi0.5_zstd
  [OK] ae_b32_u4_roi0.0_zstd (45.6s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zstd_full/ae_b32_u4_roi0.0_zstd
  [OK] ae_b32_u4_roi0.3_zstd (45.6s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zstd_full/ae_b32_u4_roi0.3_zstd
  [OK] ae_b32_u4_roi0.5_zstd (46.3s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zstd_full/ae_b32_u4_roi0.5_zstd
  [OK] ae_b32_u6_roi0.0_zstd (45.4s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zstd_full/ae_b32_u6_roi0.0_zstd
  [OK] ae_b32_u6_roi0.3_zstd (46.5s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zstd_full/ae_b32_u6_roi0.3_zstd
  [OK] ae_b32_u6_roi0.5_zstd (47.9s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zstd_full/ae_b32_u6_roi0.5_zstd
  [OK] ae_b32_u8_roi0.0_zstd (46.2s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zstd_full/ae_b32_u8_roi0.0_zstd
  [OK] ae_b32_u8_roi0.3_zstd (46.4s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zstd_full/ae_b32_u8_roi0.3_zstd
  [OK] ae_b32_u8_roi0.5_zstd (46.6s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zstd_full/ae_b32_u8_roi0.5_zstd
  [OK] ae_b64_u4_roi0.0_zstd (46.2s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zstd_full/ae_b64_u4_roi0.0_zstd
  [OK] ae_b64_u4_roi0.3_zstd (44.8s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zstd_full/ae_b64_u4_roi0.3_zstd
  [OK] ae_b64_u4_roi0.5_zstd (43.5s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zstd_full/ae_b64_u4_roi0.5_zstd
  [OK] ae_b64_u6_roi0.0_zstd (45.1s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zstd_full/ae_b64_u6_roi0.0_zstd
  [OK] ae_b64_u6_roi0.3_zstd (45.9s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zstd_full/ae_b64_u6_roi0.3_zstd
  [OK] ae_b64_u6_roi0.5_zstd (45.3s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zstd_full/ae_b64_u6_roi0.5_zstd
  [OK] ae_b64_u8_roi0.0_zstd (44.7s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zstd_full/ae_b64_u8_roi0.0_zstd
  [OK] ae_b64_u8_roi0.3_zstd (45.0s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zstd_full/ae_b64_u8_roi0.3_zstd
  [OK] ae_b64_u8_roi0.5_zstd (44.3s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zstd_full/ae_b64_u8_roi0.5_zstd
  [OK] ae_b128_u4_roi0.0_zstd (47.1s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zstd_full/ae_b128_u4_roi0.0_zstd
  [OK] ae_b128_u4_roi0.3_zstd (44.6s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zstd_full/ae_b128_u4_roi0.3_zstd
  [OK] ae_b128_u4_roi0.5_zstd (45.9s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zstd_full/ae_b128_u4_roi0.5_zstd
  [OK] ae_b128_u6_roi0.0_zstd (43.8s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zstd_full/ae_b128_u6_roi0.0_zstd
  [OK] ae_b128_u6_roi0.3_zstd (44.4s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zstd_full/ae_b128_u6_roi0.3_zstd
  [OK] ae_b128_u6_roi0.5_zstd (45.6s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zstd_full/ae_b128_u6_roi0.5_zstd
  [OK] ae_b128_u8_roi0.0_zstd (45.1s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zstd_full/ae_b128_u8_roi0.0_zstd
  [OK] ae_b128_u8_roi0.3_zstd (44.7s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zstd_full/ae_b128_u8_roi0.3_zstd
  [OK] ae_b128_u8_roi0.5_zstd (44.9s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zstd_full/ae_b128_u8_roi0.5_zstd

36/36 OK. manifest: /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zstd_full/loopback_ideal_zstd_FULL_manifest.json
[agg_loopback] 36 profiles -> rl_agent/LOOPBACK_LATENCY_ZSTD.md
[2026-07-21 13:40:27] Stage 2 done: 36 zstd latency profiles
[2026-07-21 13:40:27] --- Stage 3: build PERMODEL_KNOB_MATRIX_ZSTD.md ---
wrote rl_agent/PERMODEL_KNOB_MATRIX_ZSTD.md  (42 profiles, 11 within tol)
[2026-07-21 13:40:27] --- Stage 4: banner + BYMODEL + A/B ---
[apply_matrix_banner] zstd banner applied to rl_agent/PERMODEL_KNOB_MATRIX_ZSTD.md
[make_bymodel_grouped] 36 rows (zstd) -> rl_agent/PERMODEL_KNOB_MATRIX_ZSTD_BYMODEL.md
[make_codec_ab] 36 overlapping profiles -> CODEC_LATENCY_AB.md
[2026-07-21 13:40:27] interpolated markers in zstd matrix: 4 (expect small: header note + fp16 anchor only)
[2026-07-21 13:40:27] ===================== ZSTD FULL OVERNIGHT END =====================
ZSTD_FULL_OVERNIGHT_DONE
