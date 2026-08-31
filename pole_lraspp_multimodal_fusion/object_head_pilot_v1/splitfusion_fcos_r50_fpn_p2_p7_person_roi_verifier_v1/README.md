# SplitFusion FCOS person ROI verifier v1

This package adds one minimal residual score verifier for existing post-original-NMS **person** candidates from the frozen recovered epoch-26 SplitFusion FCOS model. It creates, removes, reorders, or changes no candidate or geometry. Vehicle records and scores remain bit-identical, and segmentation is emitted from the unchanged base output.

The frozen checkpoint is
`experiments/route_b_v3_1_splitfusion_fcos_r50_fpn_p2_p7_v1_numerical_recovery_v1/20260830_recovered_epoch10_gate_v1/checkpoints/epoch_026.pt`
with required SHA-256 `da14d21edbd374c1c3abce02ca4674b9f4097becfba9759aba945cea160a297f`.

For each frame, one vectorized `MultiScaleRoIAlign` call gathers all person boxes from frozen P2/P3 at `7x7` with sampling ratio 2. Adaptive average pooling produces 1,024 values; ten fixed scalars follow in the order defined by `SCALAR_FEATURE_NAMES`. Only `LayerNorm(1034) -> Linear(1034,128) -> ReLU -> Linear(128,1)` is trainable. The reviewed canonical match-before-ignore labeler is reused only for cache labels; the prior candidate-quality head is neither loaded nor stacked.

The cache split is train-only and episode-disjoint. The two fixed holdouts are `canonical_v3_03_train_30_30_s503_tm1503` and `canonical_v3_04_train_50_50_s504_tm1504`; the remaining eight training episodes are fit episodes. Training is fixed at five epochs, deterministic 1:3 positive/negative sampling, BCE, Adam `1e-3`, batch size 1,024, and seed 20260830. Validation inference is refused unless the exact untouched-holdout frontier contains a joint 0.80 precision/0.80 recall interval.

Future commands (not run during implementation), from the repository root:

```bash
python3 -m pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_roi_verifier_v1.build_train_cache \
  --output experiments/person_roi_verifier_v1/train_cache --device cuda:0

python3 -m pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_roi_verifier_v1.train_verifier \
  --cache experiments/person_roi_verifier_v1/train_cache \
  --output experiments/person_roi_verifier_v1/person_roi_verifier.pt --device cuda:0

python3 -m pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_roi_verifier_v1.infer_verified \
  --verifier-checkpoint experiments/person_roi_verifier_v1/person_roi_verifier.pt \
  --output experiments/person_roi_verifier_v1/predictions --device cuda:0

python3 -m pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_roi_verifier_v1.evaluate_verified \
  --prediction-dir experiments/person_roi_verifier_v1/predictions
```

Compact CPU synthetic checks:

```bash
python3 -m unittest pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_roi_verifier_v1.tests.test_synthetic -v
```
