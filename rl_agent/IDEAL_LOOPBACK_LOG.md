[2026-07-09 20:07:05] ===== IDEAL LOOPBACK re-run START (rmem=8MB) =====
[2026-07-09 20:07:05] CARLA launch attempt 1
[2026-07-09 20:07:30] CARLA up
sweep 'loopback_ideal_quantroi': 7 job(s), max_parallel=1, cwd=/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun
  [OK] q_u8_zstd (40.7s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal/q_u8_zstd
  [OK] q_u8_zlib (50.0s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal/q_u8_zlib
  [OK] q_u6_zstd (44.4s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal/q_u6_zstd
  [OK] q_u4_zstd (42.3s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal/q_u4_zstd
  [OK] roi0.1_u8_zstd (42.8s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal/roi0.1_u8_zstd
  [OK] roi0.3_u8_zstd (42.8s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal/roi0.3_u8_zstd
  [OK] roi0.5_u8_zstd (42.9s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal/roi0.5_u8_zstd

7/7 OK. manifest: /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal/loopback_ideal_quantroi_manifest.json
sweep 'loopback_ideal_ae': 2 job(s), max_parallel=1, cwd=/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun
  [OK] ae_b128_u8_zstd (41.3s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal/ae_b128_u8_zstd
  [OK] ae_b64_u8_zstd (41.7s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal/ae_b64_u8_zstd

2/2 OK. manifest: /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal/loopback_ideal_ae_manifest.json
[agg_loopback] 9 profiles -> rl_agent/LOOPBACK_LATENCY.md
wrote rl_agent/COMPLETE_KNOB_MATRIX.md  (17 profiles, 9 within tol)
[2026-07-09 20:14:04] matrix refreshed with measured ideal-transport latency
[2026-07-09 20:14:04] ===== IDEAL LOOPBACK re-run END =====
IDEAL_LOOPBACK_DONE
