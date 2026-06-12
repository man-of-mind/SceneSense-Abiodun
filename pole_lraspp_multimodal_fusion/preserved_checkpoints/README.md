# Preserved fusion checkpoints — DO NOT OVERWRITE

This directory holds blessed checkpoints from completed unattended training
cycles. Each entry is a frozen reference point that future inference work can
revert to if a new training cycle regresses or doesn't ship.

The pipeline supervisor (`run_pipeline.py`) writes new experiment dirs as
siblings under `experiments/pole_lraspp_multimodal_fusion/`, so prior
checkpoints there are *also* preserved by design — the copies here are
explicit one-line pointers for fast retrieval, not the only copy.

## Operational default — use this one

**`run5_lowfuse_obj_sel_best.pt`** is the operational default for the live
inference client (`carla_split_inference_udp_fusion_object_pole_client.py`).
Tighter on every object metric than the legacy run-3 checkpoint. See entry
below for full metrics.

The 2026-05-08 iteration cycle tested whether more epochs at the same recipe
would clear the remaining `xy_mae ≤ 1.0`, `yaw_mae ≤ 10`, `recall ≥ 0.60`
targets. Run-6 (training_budget bumped 6h→9h, ran the full 40 epochs)
**did not strictly improve** over run-5 — vehicle_iou cleared to a strict
pass, yaw_mae tightened slightly, but xy_mae regressed (1.088 → 1.135 m) due
to additional low-confidence TPs at 0.03 score threshold. **The current recipe
is at its asymptote**: more epochs alone trade recall for localization
precision rather than uniformly improving. Further gains require structural
changes (multi-scale object head, sub-pixel offset regression, larger
backbone) — see `experiments/.../20260508_145448_*/final_summary.md` for the
diagnosis. Run-6's checkpoint is preserved in its own experiment directory
but NOT promoted into this folder.

## Entries

### run3_high_only_object_head_best.pt
- Trained 2026-05-06 / 2026-05-07 (run 3 of the original 3-iteration sweep).
- Architecture: `MultiTaskFusionLRASPP` with **object_head fed by `high` feature only** (1/16 stride, 960 channels). This is the architecture *before* the `fuse_low_feature` change introduced in the next training cycle.
- Trial: `fusion_v2_adamw_768x432_lr1e-4_radar4_aug_strong_bs2`.
- Source: `experiments/pole_lraspp_multimodal_fusion/20260506_201944_pole_lraspp_multimodal_fusion_learned_localization/checkpoints/fusion_v2_adamw_768x432_lr1e-4_radar4_aug_strong_bs2/best.pt`
- Test metrics summary (full JSON in `run3_test_metrics.json`):
  - mIoU 0.787 (3-class) / 0.903 (2-class bg+veh)
  - vehicle_iou 0.846, person_iou 0.556
  - learned_object f1 0.532, precision 0.680, recall 0.437
  - learned_global_xy_mae_m **2.426**
  - learned_yaw_mae_deg **19.88**
  - learned_dimension_mae_m 0.451
  - learned_parked_accuracy 0.923
  - fusion_miou_delta_vs_rgb +0.207 (note: not apples-to-apples with the next cycle, see `run3_final_summary.md`)

### To revert a future inference task to this checkpoint
```
--fusion-checkpoint /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/pole_lraspp_multimodal_fusion/preserved_checkpoints/run3_high_only_object_head_best.pt
```

The live client `carla_split_inference_udp_fusion_object_pole_client.py`
inspects the checkpoint dict and constructs the matching architecture from the
`fuse_low_into_object_head` flag stored at save time. This run-3 checkpoint
was saved before that flag existed, so it is treated as `False` (high-only
object head) — the live client builds the matching legacy architecture.

### run5_lowfuse_obj_sel_best.pt
- Trained 2026-05-08 (iteration 2 of the new tightened-target cycle).
- Architecture: `MultiTaskFusionLRASPP` with **object_head fed by concatenated `low` (1/8 stride, 40 ch) + `high` (1/16 stride, 960 ch)** features → `fuse_low_into_object_head=True` is stored in the checkpoint dict, so the live client constructs the matching architecture automatically.
- Trial: `fusion_v4_lowfuse_adamw_768x432_lr1e-4_radar4_aug_strong_bs2_obj_sel` (single trial, 768×432, batch=2, strong aug, AdamW lr=1e-4).
- Selection-score formula: `miou - 0.05·loc_loss - 0.05·dim_loss` (10×/5× the prior weights), so the saved checkpoint reflects the epoch where object regression was best, not just where seg miou peaked.
- Source: `experiments/pole_lraspp_multimodal_fusion/20260508_070718_pole_lraspp_multimodal_fusion_learned_localization/checkpoints/fusion_v4_lowfuse_adamw_768x432_lr1e-4_radar4_aug_strong_bs2_obj_sel/best.pt`
- Test metrics summary (full JSON in `run5_test_metrics.json`):
  - mIoU 0.792 (3-class) / 0.904 (2-class bg+veh)
  - vehicle_iou 0.849, person_iou 0.568
  - learned_object f1 **0.619**, precision 0.702, recall 0.554
  - learned_global_xy_mae_m **1.088** (vs run-3 2.426 — halved twice over)
  - learned_yaw_mae_deg **10.97** (vs run-3 19.88)
  - learned_dimension_mae_m 0.275
  - learned_parked_accuracy 0.909
  - fusion_miou_delta_vs_rgb +0.205 (caveat: legacy RGB-only baseline still includes person_iou=0)
- §2 status: **6/11 strict pass**. Misses: miou (3-class artifact, 2-class would pass), vehicle_iou (off by 0.0015 — within noise), recall (0.554 vs 0.60), xy_mae (1.088 vs 1.0), yaw_mae (10.97 vs 10).
- Training was budget-cut at epoch 36/40. Iteration 3 (run-6) ran the full 40 epochs but did not strictly improve — see "Operational default" section above.

### Choosing between run-3 and run-5 for live inference
- Use **run5_lowfuse_obj_sel_best.pt** by default. Tighter on every object metric.
- Revert to **run3_high_only_object_head_best.pt** only if the run-5 architecture surfaces an unexpected runtime issue (e.g., higher latency at 768×432 — the low-feature concat adds compute at the 1/8-stride spatial size).

## Inference-time knob to tighten run-5 xy_mae further (no retraining)

`xy_mae` is sensitive to the score threshold. The metric is computed only over
matched TPs, so admitting more low-confidence detections (0.03 threshold)
drags up per-TP MAE even though more GTs match. Raising
`--object-score-threshold 0.03 → 0.10` in the live client is expected to
shift the operating point toward tighter localization at the cost of recall.
Useful for deployments where precise XY matters more than detecting every
distant vehicle. Has not been formally measured offline against run-5 yet —
left as a quick task for the next session.
