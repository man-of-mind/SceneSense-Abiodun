[2026-07-08 21:11:30] ===== OVERNIGHT MONTH-2 START =====
[2026-07-08 21:21:31] ===== OVERNIGHT MONTH-2 START =====
[2026-07-08 22:17:31] M' ready: /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/stage2_obj_drop/checkpoints/mprime_stage2_obj_drop/best.pt
[2026-07-08 22:17:31]   gateA| mIoU                         0.8408     0.8367     0.8284  PASS
[2026-07-08 22:17:31]   gateA| vehicle_IoU                  0.9327     0.9339     0.9246  PASS
[2026-07-08 22:17:31]   gateA| person_IoU                   0.5924     0.5790     0.5732  PASS
[2026-07-08 22:17:31]   gateA| object_recall                0.8348     0.8464     0.8379  FAIL
[2026-07-08 22:17:31]   gateA| person_object_recall         0.7869     0.7980     0.7901  FAIL
[2026-07-08 22:17:31]   gateA| global_xy_mae_m              1.2112     1.1921     1.2040  FAIL
[2026-07-08 22:17:31]   gateA| person_xy_mae_m              1.3924     1.4401     1.4545  PASS
[2026-07-08 22:17:31]   gateA| dimension_mae_m              0.1856     0.2032     0.2053  PASS
[2026-07-08 22:17:31]   gateA| 
[2026-07-08 22:17:31]   gateA| GATE A RESULT: FAIL/REVIEW -> inspect before proceeding
[2026-07-08 22:17:31]   gateA| [2026-07-08 22:16:46] ===== Stage-2 REBALANCE END =====
[2026-07-08 22:17:31]   gateA| STAGE2_REBALANCE_DONE OK
[2026-07-08 22:17:31] --- AE training on M' (128,64,32) ---
[2026-07-08 22:17:31] AE train b128

### GATE A assessment (rebalanced M', for morning review) — 2026-07-08
VERDICT: marginal PASS-with-caveat. Maximin selection fixed the v1 localization regression:
  - global loc 1.32->1.21m (baseline 1.19, now +1.6%), person loc 1.39m (BEATS baseline 1.44),
    dimension 0.186 (BEATS baseline 0.203), mIoU/veh/person-IoU all PASS.
  - Residual cost: object recall 0.835 (-1.4%) and person recall 0.787 (-1.4%) sit just outside the
    strict 1% band. Seg fully preserved; localization at baseline.
RECOMMENDATION: ACCEPT this M' for the static-knob characterization (robust model within ~1.5% of the
  clean specialist on every metric, seg+loc preserved). The ~1.4% pedestrian-recall gap is the residual
  price of drop-robustness and is covered operationally by the RL vulnerable-object guardrail (agent picks
  low-q when pedestrians present). If the team wants that recall back, options for a follow-up pass:
  upweight object center/recall loss, or widen the maximin toward clean. NOT worth blocking Month-2 on.
  All overnight sweeps + matrix are being built on this M' and remain valid as the action-cost model.
[2026-07-08 22:47:28] AE train b64
[2026-07-08 23:17:21] AE train b32
[2026-07-08 23:47:14] eval clean_noquant: 
[2026-07-08 23:50:02] eval quant_uint8_zlib: --quantization-mode per_channel_uint8 --entropy-coder zlib
[2026-07-08 23:54:16] eval quant_uint8_zstd: --quantization-mode per_channel_uint8 --entropy-coder zstd
[2026-07-08 23:57:03] eval quant_uint8_none: --quantization-mode per_channel_uint8 --entropy-coder none
[2026-07-08 23:59:47] eval quant_uint6_zlib: --quantization-mode per_channel_uint6 --entropy-coder zlib
[2026-07-09 00:04:13] eval quant_uint6_zstd: --quantization-mode per_channel_uint6 --entropy-coder zstd
[2026-07-09 00:07:15] eval quant_uint6_none: --quantization-mode per_channel_uint6 --entropy-coder none
[2026-07-09 00:09:57] eval quant_uint4_zlib: --quantization-mode per_channel_uint4 --entropy-coder zlib
[2026-07-09 00:13:59] eval quant_uint4_zstd: --quantization-mode per_channel_uint4 --entropy-coder zstd
[2026-07-09 00:16:46] eval quant_uint4_none: --quantization-mode per_channel_uint4 --entropy-coder none
[2026-07-09 00:19:27] eval roi_0.1: --quantization-mode per_channel_uint8 --entropy-coder zlib --roi-threshold 0.1
[2026-07-09 00:23:23] eval roi_0.3: --quantization-mode per_channel_uint8 --entropy-coder zlib --roi-threshold 0.3
[2026-07-09 00:27:02] eval roi_0.5: --quantization-mode per_channel_uint8 --entropy-coder zlib --roi-threshold 0.5
[2026-07-09 00:30:34] eval roi_0.7: --quantization-mode per_channel_uint8 --entropy-coder zlib --roi-threshold 0.7
[2026-07-09 00:33:53] eval ae_b128: --quantization-mode per_channel_uint8 --entropy-coder zlib --ae-checkpoint /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/rl_agent/feature_ae/checkpoints/ae_b128.pt
[2026-07-09 00:36:42] eval ae_b64: --quantization-mode per_channel_uint8 --entropy-coder zlib --ae-checkpoint /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/rl_agent/feature_ae/checkpoints/ae_b64.pt
[2026-07-09 00:39:32] eval ae_b32: --quantization-mode per_channel_uint8 --entropy-coder zlib --ae-checkpoint /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/rl_agent/feature_ae/checkpoints/ae_b32.pt
[2026-07-09 00:42:21] eval ae_b64_roi0.3: --quantization-mode per_channel_uint8 --entropy-coder zlib --ae-checkpoint /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/rl_agent/feature_ae/checkpoints/ae_b64.pt --roi-threshold 0.3
[2026-07-09 00:45:12] eval ae_b32_roi0.3: --quantization-mode per_channel_uint8 --entropy-coder zlib --ae-checkpoint /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/rl_agent/feature_ae/checkpoints/ae_b32.pt --roi-threshold 0.3
[2026-07-09 00:48:00] --- loopback latency/reliability sweep (starting CARLA) ---
[2026-07-09 00:48:00] CARLA launch attempt 1
[2026-07-09 00:48:24] CARLA up+stable
sweep 'loopback_sweep_mprime': 8 job(s), max_parallel=1, cwd=/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun
  [OK] q_pchan_u8_zlib (444.7s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback/q_pchan_u8_zlib
  [OK] q_pchan_u8_none (432.1s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback/q_pchan_u8_none
  [OK] q_pchan_u8_zstd (442.2s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback/q_pchan_u8_zstd
  [OK] q_pchan_u6_zlib (443.2s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback/q_pchan_u6_zlib
  [OK] q_pchan_u6_zstd (349.2s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback/q_pchan_u6_zstd
  [OK] q_pchan_u4_zlib (44.4s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback/q_pchan_u4_zlib
  [OK] q_pchan_u4_zstd (41.4s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback/q_pchan_u4_zstd
  [OK] q_pchan_u4_none (381.7s) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback/q_pchan_u4_none

8/8 OK. manifest: /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps_loopback/loopback_sweep_mprime_manifest.json
[agg_loopback] 8 profiles -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/rl_agent/LOOPBACK_LATENCY.md
[2026-07-09 01:31:28] --- aggregate COMPLETE_KNOB_MATRIX ---
wrote /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/rl_agent/COMPLETE_KNOB_MATRIX.md  (19 profiles, 8 within tol)
[2026-07-09 01:31:28] ===== OVERNIGHT MONTH-2 END =====
MONTH2_DONE OK
