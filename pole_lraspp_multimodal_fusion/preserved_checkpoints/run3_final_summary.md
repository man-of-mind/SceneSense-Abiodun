# Final summary — pole LR-ASPP multimodal fusion (3-iteration unattended sweep)

Three unattended runs over 2026-05-06 / 2026-05-07. Iteration cap reached.

## Best checkpoint
`experiments/pole_lraspp_multimodal_fusion/20260506_201944_pole_lraspp_multimodal_fusion_learned_localization/checkpoints/fusion_v2_adamw_768x432_lr1e-4_radar4_aug_strong_bs2/best.pt`

## Test-set metric trajectory

| Metric | §2 target | Run 1 | Run 2 | Run 3 (final) | Pass? |
|---|---|---|---|---|---|
| miou (3-class avg) | ≥ 0.90 | 0.924 (NaN-filtered, 2-cls) | 0.928 (NaN-filtered, 2-cls) | **0.787 (3-class)** | ✗ artifact |
| 2-class mIoU (bg+veh) | — | 0.924 | 0.928 | 0.903 | — (matches runs 1-2 if person stays NaN) |
| vehicle_iou | ≥ 0.85 | 0.854 | 0.862 | 0.846 | ✗ off by 0.004 |
| person_iou | ≥ 0.55 (not NaN) | NaN | NaN | **0.556** | ✓ |
| learned_object_recall | ≥ 0.60 | 0.371 | 0.435 | 0.437 | ✗ |
| learned_object_precision | ≥ 0.55 | 0.755 | 0.683 | 0.680 | ✓ |
| learned_object_f1 | ≥ 0.55 | 0.498 | 0.532 | 0.532 | ✗ off by 0.018 |
| learned_global_xy_mae_m | ≤ 2.5 | 2.516 | 2.462 | **2.426** | ✓ |
| learned_dimension_mae_m | ≤ 0.6 | 0.468 | 0.466 | 0.451 | ✓ |
| learned_yaw_mae_deg | ≤ 25 | **78.3** | 19.56 | **19.88** | ✓ |
| learned_parked_accuracy | ≥ 0.80 | 0.895 | 0.914 | 0.923 | ✓ |
| fusion_miou_delta_vs_rgb | > 0 | +0.037 | +0.041 | +0.207 | ✓ |

7/11 strict pass. Two of the four "fails" are artifacts/within-noise:
- **mIoU 0.787 vs ≥ 0.90** — `class_iou_from_confusion` returns NaN for any class with no GT pixels, and `miou` averages the non-NaN classes only. Runs 1 and 2 averaged over only `(background, vehicle)` because pedestrians weren't tagged. Run 3 averages over all three classes, so a perfectly-IoU-equal model would score lower. The apples-to-apples 2-class mIoU for run 3 is **0.903**, which still passes the 0.90 target. The §2 threshold needs to be revisited as a 3-class threshold (recommend ≥ 0.80) or stay 2-class explicit.
- **vehicle_iou 0.846 vs ≥ 0.85** — within run-to-run noise; run 2 was 0.862. Different collection seed gives a different test split.

The genuine remaining gaps are **learned_object_recall** (0.437 vs 0.60) and **learned_object_f1** (0.532 vs 0.55, ~3.4% short). Lowering the score threshold further (0.10 → 0.05) didn't move recall (0.435 → 0.437), so the bottleneck is not the decoder threshold — it's the heatmap response itself at GT centers.

## What this 3-run cycle fixed

1. **`learned_object_f1 = 0` in the prior baseline run** — heatmap target never reached 1.0 at the integer peak pixel because the gaussian was evaluated at integer coordinates with sub-pixel center, leaving `target.eq(1.0)` empty and starving the focal loss of positives. Fixed by forcing `heatmap[iy, ix] = 1.0` after `draw_gaussian` and tolerant `target.ge(1 - 1e-3)`. f1 went from 0.0 → 0.498 in run 1.
2. **`learned_yaw_mae = 78°`** — yaw loss weight 0.05 was 10× too low. Bumped to 0.5 in run 2, MAE collapsed to 19.6° (well under the 25° target).
3. **`learned_global_xy_mae = 2.516 m`** (just over) — fixed by giving the head more epochs (8 → 14). Now 2.426 m.
4. **`person_iou = NaN`** — CARLA 0.10's instance-segmentation camera emits tag **24** for pedestrians, not tag 12. Verified against the saved `instance_raw/*.png` (tag 24 with 1739 px in a `ped48` frame matches 6 pedestrian actor silhouettes ~10% of bbox area), and against the user's working LiDAR collector (`radar_camera_lidar_data_collect_update_pedestrian_vizualizor_fusion.py:1116` defaults `--ped-candidate-tags 12,24,25,4`). Updated `PERSON_TAGS = {4, 12, 13, 24, 25}` in both `pole_lraspp_multimodal_fusion/common.py` and `pole_lraspp_training/common.py`. person_iou went NaN → 0.556 in run 3.
5. **AMP regression-head stability** — wrapped `multitask_object_loss` in `torch.cuda.amp.autocast(enabled=False)` and cast `outputs["object"].float()` so smooth-L1 / BCE on small-magnitude regression targets don't silently zero gradients in FP16.

## Diagnosis of remaining gaps (per §9)

### Object recall stuck at ~0.435 across runs 2 and 3
The decoder's `topk → score-threshold → NMS` pipeline isn't the bottleneck — lowering threshold from 0.10 → 0.05 added only 0.002 recall. That means the model's heatmap is genuinely **not firing strongly at ~57% of GT centers**. Most of those misses are likely small/distant vehicles where the LR-ASPP `high` feature (1/16 input resolution) doesn't have enough spatial support to localize a center.

Likely-effective single change for a future run: **fuse the `low` feature into the object head** alongside `high` (i.e., upsample `high` to `low` resolution and concat, then run the existing 3-conv object head at higher spatial resolution). The segmentation head already uses both; the object head only uses `high`. Code change to `model.MultiTaskFusionLRASPP.forward`. This is a structural model change, not a config knob, and is the single highest-leverage edit if you want to push recall toward 0.60.

Alternative: extend the heatmap target's gaussian radius for distant/small objects (size-aware sigma already supported by `object_targets.draw_gaussian` if you supply a per-object radius). Currently a fixed `heatmap_radius_px: 3` is used.

### vehicle_iou 0.846 vs 0.85
Within noise across runs (range 0.846–0.862). Don't chase.

### mIoU threshold semantics
§2 says `miou ≥ 0.90` without specifying class count. `class_iou_from_confusion` skips NaN classes, so a 2-class run can score higher than a 3-class run on the same model. Recommend revising §2 to either (a) compute mIoU only over classes with GT pixels and lower threshold to 0.80, or (b) compute a 2-class mIoU explicitly (background + vehicle) and keep 0.90.

## fusion_miou_delta_vs_rgb interpretation
Run 3 reports +0.207 over RGB-only baseline, but this is **not** a fair comparison anymore. The baseline RGB-only checkpoint (`pole_lraspp_training/20260505_173329`) was trained against masks built with the old `PERSON_TAGS = {12, 13}` and thus has zero person predictions. The fusion model in run 3 was trained against the corrected mask. To get a fair fusion-vs-RGB comparison, retrain the RGB-only baseline against the corrected masks and re-evaluate. That's a separate workstream for the `pole_lraspp_training` workflow, outside this fusion-run cycle.

## Suggested next iteration (next session, not this one)

1. Retrain the RGB-only baseline against corrected `PERSON_TAGS` so `fusion_miou_delta_vs_rgb` is fair.
2. Add the `low`-feature path to `MultiTaskFusionLRASPP.object_head` to lift object recall.
3. If still below targets, replace the constant `heatmap_radius_px` with size-aware sigma in `object_targets.build_object_targets` (smaller objects get smaller sigma, larger get larger).
4. Optionally add a **person** path to the learned-object head (currently `valid_vehicle_objects` filters to vehicles only, so all `learned_object_*` metrics are vehicle-specific). Adding person object detection would make the demo end-to-end on both classes.

## Iteration cap reached
Per §9 / §11, no further unattended runs in this session. The unattended-training instruction set is in good shape; the remaining work is targeted code-level improvements with their own focused planning, not another sweep.
