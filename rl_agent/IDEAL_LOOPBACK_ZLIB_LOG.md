[2026-07-20 14:13:17] ===== IDEAL LOOPBACK ZLIB re-run START (rmem=8MB, deployed codec) =====
[2026-07-20 14:13:17] rmem_max=8388608 OK
[2026-07-20 14:13:17] reusing existing CARLA on :2000 (will NOT kill it on exit)
sweep 'loopback_ideal_quantroi_zlib': 6 job(s), max_parallel=1, cwd=/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun
  [OK] q_u8_zlib (51.5s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zlib/q_u8_zlib
  [OK] q_u6_zlib (49.7s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zlib/q_u6_zlib
  [OK] q_u4_zlib (45.3s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zlib/q_u4_zlib
  [OK] roi0.1_u8_zlib (49.5s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zlib/roi0.1_u8_zlib
  [OK] roi0.3_u8_zlib (48.3s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zlib/roi0.3_u8_zlib
  [OK] roi0.5_u8_zlib (47.1s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zlib/roi0.5_u8_zlib

6/6 OK. manifest: /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zlib/loopback_ideal_quantroi_zlib_manifest.json
sweep 'loopback_ideal_ae_zlib': 2 job(s), max_parallel=1, cwd=/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun
  [OK] ae_b128_u8_zlib (44.1s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zlib/ae_b128_u8_zlib
  [OK] ae_b64_u8_zlib (43.8s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zlib/ae_b64_u8_zlib

2/2 OK. manifest: /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zlib/loopback_ideal_ae_zlib_manifest.json
[agg_loopback] 8 profiles -> rl_agent/LOOPBACK_LATENCY_ZLIB.md
[2026-07-20 14:19:36] zlib latency aggregated -> loopback_latency_zlib.json
[2026-07-20 14:19:36] ===== IDEAL LOOPBACK ZLIB re-run END =====
IDEAL_LOOPBACK_ZLIB_DONE
