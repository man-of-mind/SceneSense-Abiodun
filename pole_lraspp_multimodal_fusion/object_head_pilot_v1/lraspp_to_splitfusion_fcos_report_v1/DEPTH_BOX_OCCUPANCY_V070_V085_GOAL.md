/goal Compute read-only `0.70` and `0.85` depth-consistent projected-box occupancy sensitivity results for the locked SplitFusion-FCOS service candidate and the two frozen LR-ASPP comparators. Do not run model inference or change any scientific configuration.

This is not a new visibility-ground-truth attempt. The renderer z-buffer branch is closed. Use only the original registered Route B depth-consistent projected-box occupancy calculation and describe it accurately.

## Exact interpretation

For each geometrically qualified actor within 40 m:

```text
depth-box occupancy =
  pixels inside the clipped projected 3D actor box whose synchronized scene depth
  lies inside the actor camera-depth near/far interval with the registered ±0.25 m tolerance
  / pixels in the clipped projected box
```

Evaluate minimum occupancy `>=0.70` and `>=0.85`.

Never call these values anatomical visibility, actor silhouette coverage, or the fraction of the body visible. They are diagnostic depth-consistent projected-box occupancy views. Do not use or modify any uncommitted instance/z-buffer implementation.

## Frozen prediction sets

Evaluate exactly these three completed models using their existing prediction files:

1. Locked SplitFusion-FCOS epoch-26 service candidate:
   - predictions: `experiments/splitfusion_fcos_service_candidate_v1/predictions/`
   - checkpoint SHA-256: `da14d21edbd374c1c3abce02ca4674b9f4097becfba9759aba945cea160a297f`
   - prediction-set SHA-256: `8c2d0ae02912204a7d24bcd6924b540ecb1a4d048dcec8ddf6df9209bb72e295`
2. Depth-aware LR-ASPP joint epoch 10:
   - experiment: `experiments/route_b_v3_1_depth_aware_lraspp_v1/20260829_060656/`
   - checkpoint SHA-256: `f58b3e71c60ea5f225105de63e5be910f9d6d18330d2df33dbda13470589354e`
3. Depth-aware LR-ASPP two-stage epoch 30:
   - experiment: `experiments/route_b_v3_1_depth_aware_lraspp_two_stage_v1/20260829_184743/`
   - checkpoint SHA-256: `ae118744f9b312320bfc10c726982708a7d225747bf67d30748086739001dbde`

Reuse the exact frozen scorer/matching/segmentation path previously used for the completed `0.25` and `0.50` sensitivity evaluations. Verify prediction and canonical artifact hashes before scoring. Do not load a model checkpoint into memory, run CUDA, regenerate predictions or rescore canonical `0.10/0.25/0.50` results.

## Scope

For each model and each of the two occupancy thresholds, run exactly one scoring pass and write create-only sensitivity JSON artifacts beside the existing sensitivity results.

Report:

- eligible and ignored GT counts by vehicle/person;
- TP/FP/FN and ignored predictions at the fixed service score threshold `0.20`;
- precision, recall and F1 by class;
- vehicle/person XY MAE;
- recall at `0.02` as the existing diagnostic, if the frozen scorer normally emits it;
- vehicle IoU, person box-mask IoU and foreground mIoU under the same threshold-specific ignore policy;
- all nine historical gate decisions and pass count, diagnostic only;
- exact model/prediction/evaluation hashes;
- explicit warning if any class has a small or empty denominator.

Create one compact comparison table covering all three models at `0.70` and `0.85`. Do not select a model, tune a score, alter a service decision, create a new checkpoint, or describe better high-threshold scores as general accuracy improvement. Higher occupancy changes the evaluated population.

No test data, training, inference, CUDA, CARLA, OAI mutation, compression, q/AE work, deployment or 288 measurements. No source changes or commit should be necessary. Preserve the existing dirty OAI submodule and uncommitted failed visibility packages.

Terminal:

`DEPTH_BOX_OCCUPANCY_V070_V085_SENSITIVITY_COMPLETE`
