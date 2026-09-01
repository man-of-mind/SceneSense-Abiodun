# Publication instance-visibility evaluation protocol v1

This is a prospective held-out evaluation. The perception models and post-processing were frozen before the data were collected, so these episodes cannot be used for model selection, threshold selection, calibration, or retraining.

## Visibility definition

For each actor, CARLA supplies an exact visible instance mask from the normal scene. A second renderer pass reproduces the same actor blueprint, pose, camera-relative transform, and camera intrinsics without external occluders. The metric is:

`visibility = pixels(visible mask ∩ unoccluded mask) / pixels(in-frame unoccluded mask)`

Self-occlusion remains present in both masks. Pixels outside the camera image are not counted as occlusion. The fixed minimum-visibility views are `0.10`, `0.25`, `0.50`, `0.70`, and `0.85`.

The implementation must not substitute projected boxes, semantic-class pixels, depth intervals, ellipses, or learned amodal masks.

## Prospective held-out episodes

| episode | traffic | scenario seed | Traffic Manager seed |
|---|---|---:|---:|
| publication_v1_01 | 30/30 | 801 | 1801 |
| publication_v1_02 | 50/50 | 802 | 1802 |
| publication_v1_03 | 30/30 | 803 | 1803 |
| publication_v1_04 | 50/50 | 804 | 1804 |

All other route, camera, radar, cadence, weather, and runtime settings remain those of Route B perception v3. Existing canonical test episodes are prohibited.

## Models

The primary model is the locked SplitFusion-FCOS epoch-26 service candidate. The two frozen LR-ASPP comparators are joint epoch 10 and task-separated stage-2 epoch 30. All three must use the same ground truth and metric implementation, and none may be changed in response to these data.

## Reporting

Report fixed-score service precision/recall/F1 using the historical 3 m world-XY matching rule, plus AP50 and AP50–95 using tight unoccluded actor-mask boxes. Report actual visible-pixel segmentation IoU and world-centre localization mean/median/p90, range error, dimensions, and yaw.

Every metric is reported by class, minimum visibility, range (`0–10`, `10–20`, `20–30`, `30–40 m`), episode, and aggregate. Use fixed-seed temporal-block bootstrap confidence intervals. Counts and empty strata must always be shown.

The complete machine-readable registration is `PUBLICATION_EVALUATION_PROTOCOL_V1.json`.
