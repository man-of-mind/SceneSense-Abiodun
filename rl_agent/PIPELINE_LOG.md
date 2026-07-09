[2026-07-08 17:10:55] ===== M' drop-aware pipeline START (pid 3817291) =====
[2026-07-08 17:10:55] STAGE start: mprime_stage1_seg_drop  (budget 6.0h)  -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/stage1_seg_drop
[2026-07-08 17:11:04] STAGE mprime_stage1_seg_drop exited rc=1 (see /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/stage1_seg_drop/train.log)
[2026-07-08 17:11:04] GATE FAIL: no checkpoint at /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/stage1_seg_drop/checkpoints/mprime_stage1_seg_drop/best.pt. Halting pipeline.
[2026-07-08 17:13:17] ===== M' drop-aware pipeline START (pid 3841138) =====
[2026-07-08 17:13:17] STAGE start: mprime_stage1_seg_drop  (budget 6.0h)  -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/stage1_seg_drop

### Design notes (recorded 2026-07-08)
- **Stage-1 (drop-aware seg):** full-model fine-tune (backbone+seg TRAINABLE) warm-started from
  seg_pps200000. The trained det_pps200000_v2 object head is loaded **frozen** (freeze_object_head=true,
  object_total=0.0) purely as the *objectness oracle* so the objectness-guided drop is meaningful during
  seg training (a randomly-initialized head gives constant objectness -> drop no-op). Output M'_seg =
  drop-robust backbone + seg head.
- **Stage-2 (drop-aware object):** backbone+seg FROZEN (init_rgb=M'_seg), object head trainable
  (init_object=det_pps200000_v2). Output M' = drop-robust object head on the drop-robust frozen backbone.
- Both stages: feature_drop_max=0.8 => per-batch q~U(0,0.8); q=0 path is a structural no-op so validation
  (clean) measures true q=0 accuracy => GATE A.
- **Bug caught + fixed at first training step (why we monitor launches):** under AMP autocast the pooled
  objectness is float16 and torch.quantile rejects half dtype. Fixed model.py gate() to compute the
  quantile in fp32. Re-verified under CUDA+autocast before relaunch.
[2026-07-08 17:38:47] ===== M' drop-aware pipeline START (pid 4018378) =====
[2026-07-08 17:38:47] STAGE start: mprime_stage1_seg_drop  (budget 6.0h)  -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/stage1_seg_drop
[2026-07-08 17:46:17] ===== M' drop-aware pipeline START (pid 4092534) =====
[2026-07-08 17:46:17] STAGE start: mprime_stage1_seg_drop  (budget 6.0h)  -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/stage1_seg_drop

### CRITICAL FIX (2026-07-08): rank-based objectness drop + drop-aware validation
- **Bug:** the objectness-guided drop used a *quantile value threshold* (`keep = objness >= quantile(objness,q)`).
  The focal-biased object heatmap is FLOOR-DOMINATED (most cells ~sigmoid(-4.6)=0.01), so the q-quantile lands
  ON the floor and `>=` keeps ~everything => the drop was a NO-OP for q<~0.85. Drop-aware training would have
  trained on undropped features (pointless). Verified: q=0.4 zeroed 0 extra cells, seg logits identical.
- **Fix:** drop by RANK — zero the lowest-objectness `round(q*N)` cells (argsort + scatter). Now q controls the
  dropped fraction exactly (verified zero-frac 32/49/67% at q=0.2/0.4/0.6). model.py `_objectness_drop.gate`.
- **Drop-aware validation:** validation now runs at q=feature_drop_max/2 (=0.4) so selection rewards robustness;
  clean q=0 is guarded separately at GATE A. (train_fusion evaluate_model + feature_drop_val.)
- **Finding (research):** objectness-guided drop preserves SEG background via context+low-skip and preserves
  object-center cells, so degradation is graceful; but minority object-class IoU (vehicle/person) IS sensitive
  at q>=0.4 (few pixels flip, but they're the object pixels) -> drop-aware training is needed and is what recovers
  vehicle/person IoU under drop. Epoch 0 val@q0.4: miou 0.766 veh 0.759 (pre-training); expect climb over epochs.
- **TODO before ROI sweep:** align evaluate_fusion `_roi_gate` (inference ROI action) to the SAME rank-based
  semantics so inference matches training. The earlier plain-200k ROI-sweep "30% free/50% mild" finding used the
  quantile-threshold gate and should be RE-RUN on M' with rank-based drop (its low-q results were likely near-no-op).
[2026-07-08 19:10:07] STAGE done: mprime_stage1_seg_drop  best=/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/stage1_seg_drop/checkpoints/mprime_stage1_seg_drop/best.pt
[2026-07-08 19:10:07] STAGE start: mprime_stage2_obj_drop  (budget 5.0h)  -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/stage2_obj_drop
[2026-07-08 20:04:10] STAGE done: mprime_stage2_obj_drop  best=/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/stage2_obj_drop/checkpoints/mprime_stage2_obj_drop/best.pt
[2026-07-08 20:04:10] M' READY: /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/stage2_obj_drop/checkpoints/mprime_stage2_obj_drop/best.pt
[2026-07-08 20:04:10] GATE A eval start -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/stage2_obj_drop/gateA_eval_best_thr020
[2026-07-08 20:06:55] GATE A eval done. Metrics: /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/stage2_obj_drop/gateA_eval_best_thr020/metrics/test_fusion_evaluation_metrics.json
[GATE A] q=0 acceptance check vs 200k targets:
   mIoU: 0.8408 >= 0.837  -> PASS
   vehicle_IoU: 0.9327 >= 0.934  -> FAIL
   obj_recall: (metric key not found - inspect json)
   ped_loc_m: (metric key not found - inspect json)
[GATE A] RESULT: REVIEW - some metric off; inspect before building on M-prime
[2026-07-08 20:06:55] ===== M' drop-aware pipeline END =====
[2026-07-08 20:36:20] ===== Stage-2 REBALANCE (maximin selection) START =====
[2026-07-08 22:13:54] M' (rebalanced) READY: /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/stage2_obj_drop/checkpoints/mprime_stage2_obj_drop/best.pt
[2026-07-08 22:13:54] GATE A eval (rebalanced) -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/stage2_obj_drop/gateA_rebalance_best_thr020
GATE A: M'(q=0) vs 200k baseline det_pps200000_v2  (rel_tol=1%)
metric                      M-prime   baseline    allowed  verdict
mIoU                         0.8408     0.8367     0.8284  PASS
vehicle_IoU                  0.9327     0.9339     0.9246  PASS
person_IoU                   0.5924     0.5790     0.5732  PASS
object_recall                0.8348     0.8464     0.8379  FAIL
person_object_recall         0.7869     0.7980     0.7901  FAIL
global_xy_mae_m              1.2112     1.1921     1.2040  FAIL
person_xy_mae_m              1.3924     1.4401     1.4545  PASS
dimension_mae_m              0.1856     0.2032     0.2053  PASS

GATE A RESULT: FAIL/REVIEW -> inspect before proceeding
[2026-07-08 22:16:46] ===== Stage-2 REBALANCE END =====
STAGE2_REBALANCE_DONE OK
