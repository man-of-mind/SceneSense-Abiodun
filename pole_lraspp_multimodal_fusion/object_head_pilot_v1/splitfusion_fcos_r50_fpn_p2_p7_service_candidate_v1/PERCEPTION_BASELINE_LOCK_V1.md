# SplitFusion-FCOS perception baseline lock v1

Lock date: 2026-08-31

## Decision

The locked `SplitFusion-FCOS-R50-FPN-P2-P7` epoch-26 service candidate is the perception baseline for the compression and system-evaluation phase. The user reported that the supervisor accepted its measured 7/9 result as service-ready for this project phase. This is a service-scope decision, not a claim that all nine original gates passed.

The two original misses were person precision and person recall under the historical v0.10 evaluator. No further detector, verifier, relational-selector, score-threshold, NMS, or calibration tuning is permitted on validation data.

## Exact baseline

- Input: one tensor containing RGB (3 channels) concatenated with radar raster (4 channels).
- Split: raw fused ResNet C2 tensor `Z`, shape `[256,112,192]` and 22,020,096 bytes in FP32.
- Checkpoint: `experiments/route_b_v3_1_splitfusion_fcos_r50_fpn_p2_p7_v1_numerical_recovery_v1/20260830_recovered_epoch10_gate_v1/checkpoints/epoch_026.pt`.
- Checkpoint SHA-256: `da14d21edbd374c1c3abce02ca4674b9f4097becfba9759aba945cea160a297f`.
- Service package source commit: `bb7edac2fbd98b2f9dec616311e9f79d957ee192`.
- Locked configuration SHA-256: `cd4db04d97ff47492cb20ae454491a02b69dc00351918ba789ea37076f7e2d79`.
- Vehicle score logit bias: `-1.476162131187961`; canonical score threshold: `0.20`.
- Person consolidation: grid 27, semantic support `0.10`, group-box IoU `0.20`.
- FCOS centerness: the original Torchvision FCOS centerness score remains part of 2D candidate scoring. It does not explicitly modify radar or the custom 3D centroid calculation.
- Geometry: the custom head predicts physical-centre direction and actor depth from fused RGB-radar features; camera calibration and pose convert these into local and world coordinates.

The candidate-quality MLP, ROI verifier, and relational selector are explicitly not part of this baseline.

## Frozen behavior

The checkpoint, model architecture, seven-channel input, C2 split, FCOS/centerness logic, semantic and geometry heads, class mapping, score calibration, person consolidation, score thresholds, NMS, and output schema are frozen.

Permitted next work is restricted to compression and transport at `Z`—zstd, fixed INT8, hybrid-q/ROI, and AE128/64/32—plus payload/latency measurement and a corrected evaluator applied equally to frozen models.

## Historical result and terminology

The existing 3,345-frame prediction set is fixed at SHA-256 `8c2d0ae02912204a7d24bcd6924b540ecb1a4d048dcec8ddf6df9209bb72e295`.

| historical view | vehicle P/R | person P/R | vehicle/person XY MAE |
|---|---:|---:|---:|
| 0.10 | 0.9316 / 0.8684 | 0.7307 / 0.6005 | 0.4787 / 0.8436 m |
| 0.25 | 0.9521 / 0.9515 | 0.7604 / 0.6685 | 0.4431 / 0.8338 m |
| 0.50 | 0.9646 / 0.9778 | 0.7687 / 0.7644 | 0.4030 / 0.7311 m |

These are **depth-consistent projected-box occupancy** sensitivity results. They must not be described in the paper as true anatomical visibility fractions.

## Publication evaluation boundary

A new prospective held-out evaluation will use actual actor-instance pixels and actual renderer-derived unoccluded actor silhouettes. Its fixed visibility thresholds are `0.10`, `0.25`, `0.50`, `0.70`, and `0.85`, within 40 m. The baseline may be evaluated but not tuned on this data.

Machine-readable provenance and all artifact hashes are in `PERCEPTION_BASELINE_LOCK_V1.json`.
