# Parked-Ego RGB+Radar Fusion V1 Results And Next Steps

Last updated: 2026-06-12

## Purpose

This document freezes the first parked-ego RGB+radar fusion training result as
`v1`, records the exact configuration and metrics, and explains why the next
iteration should focus on data balance plus object-head improvements rather
than simply continuing to train the same run.

The model task is **segmentation plus learned localization**, not true Faster
R-CNN-style object detection.

## V1 Dataset

Dataset:

```text
fusion_training_data/parked_ego_tl16_spawn80_right7_fwd4_merged_12000_stride2
```

Collection view:

- Town10HD, TL16 area, spawn `80`
- forward offset `4.0 m`
- right offset `7.0 m`
- yaw offset `-28.414 deg`
- RGB `1280x720`, model input `768x432`
- radar HFoV `120 deg`, VFoV `30 deg`, range `120 m`

Dataset composition:

- `12000` samples
- train/val/test: `8620 / 1666 / 1714`
- `114582` object rows
- `58066` vehicle rows
- `56516` person rows
- Low/medium/crowded traffic profiles, `4000` samples each

## V1 Training Configuration

Experiment:

```text
experiments/parked_ego_tl16_right7_fusion_train_20260612
```

Trial:

```text
parked_right7_lowmedcrowd_768x432_lr1e-4_bs2
```

Training:

- 40 epochs at `lr=1e-4`
- second-stage fine-tune to 80 total epochs with `resume_lr=5e-5`
- batch size `2`
- AdamW, weight decay `0.0002`
- strong photometric augmentation
- RGB warm start from the pole-trained LR-ASPP checkpoint
- class-aware localization head for `vehicle` and `person`

Best validation checkpoint:

- epoch `78`
- selection score `0.7101`
- validation mIoU `0.7849`
- validation vehicle IoU `0.9074`
- validation person IoU `0.4728`
- validation loc loss `1.3177`

Training curves:

```text
experiments/parked_ego_tl16_right7_fusion_train_20260612/figures/parked_right7_lowmedcrowd_768x432_lr1e-4_bs2_training_curves.png
```

## Held-Out Test Metrics

Strict evaluation uses:

- object score threshold `0.03`
- class-aware matching
- `3.0 m` match distance

Segmentation:

| Metric | Value |
| --- | ---: |
| mIoU | `0.7882` |
| background IoU | `0.9748` |
| vehicle IoU | `0.9180` |
| person IoU | `0.4719` |
| pixel accuracy | `0.9759` |
| RGB baseline mIoU | `0.6065` |
| fusion mIoU gain vs RGB baseline | `+0.1817` |

Learned localization:

| Metric | Value |
| --- | ---: |
| precision | `0.4750` |
| recall | `0.3806` |
| F1 | `0.4226` |
| vehicle F1 | `0.4205` |
| person F1 | `0.4251` |
| overall XY MAE | `1.1556 m` |
| vehicle XY MAE | `0.9729 m` |
| person XY MAE | `1.3727 m` |
| dimension MAE | `0.2231 m` |
| yaw MAE | `11.39 deg` |
| parked accuracy | `0.9606` |

V1 conclusion:

- Segmentation is clearly useful and fusion beats the RGB baseline.
- Vehicle segmentation and vehicle localization are close to usable.
- Person segmentation and learned localization recall/F1 are not final.
- V1 should be treated as a baseline and qualitative demo checkpoint, not as the
  final parked-ego fusion model.

## Failure Analysis

Analysis command:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache \
python3 scripts/analyze_fusion_localization_failures.py \
  --experiment-dir experiments/parked_ego_tl16_right7_fusion_train_20260612 \
  --output-dir analysis_outputs/parked_ego_fusion_v1
```

Outputs:

```text
analysis_outputs/parked_ego_fusion_v1/fusion_localization_failure_summary.json
analysis_outputs/parked_ego_fusion_v1/fusion_localization_gt_enriched.csv
analysis_outputs/parked_ego_fusion_v1/localization_f1_by_density.png
analysis_outputs/parked_ego_fusion_v1/localization_recall_by_distance.png
analysis_outputs/parked_ego_fusion_v1/localization_recall_by_bbox_area.png
analysis_outputs/parked_ego_fusion_v1/localization_recall_by_radar_support.png
```

By density:

| Density | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Low | `0.451` | `0.544` | `0.493` |
| Medium | `0.517` | `0.415` | `0.460` |
| Crowded | `0.446` | `0.307` | `0.363` |

By class:

| Class | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Vehicle | `0.474` | `0.378` | `0.420` |
| Person | `0.476` | `0.384` | `0.425` |

Important patterns:

- Crowded frames are the largest recall bottleneck.
- Small/far objects dominate misses.
- Radar support improves recall:
  - vehicle recall `0.284` without radar support vs `0.498` with radar support
  - person recall `0.375` without radar support vs `0.529` with radar support
- Person radar support is sparse, so RGB still carries most pedestrian burden.
- A looser `7.5 m` match distance raises F1 to about `0.58`, which means many
  predictions are near the right actor but too spatially loose for the strict
  `3 m` metric.

## What Not To Do Next

Do not keep training V1 indefinitely. The fine-tune improved cheap metrics, but
validation mIoU and localization F1 have mostly plateaued. More epochs alone are
unlikely to close the gap to a strong final model.

Also do not report the looser `7.5 m` match result as the primary metric. It is
useful as diagnosis, but the strict `3 m` metric is the honest target.

## V2 Plan

### Data

Collect a larger and more balanced parked-ego dataset:

- increase from `12k` to `24k-36k` samples
- keep the current TL16 right-lane view as anchor 1
- add 1-2 nearby parked viewpoints at the same intersection
- oversample crosswalk/person-heavy scenes
- oversample medium density and controlled crowded scenes
- avoid overly blocked/fully occluded crowded frames
- keep stride `2` unless a short event needs denser capture

Target split composition:

- not just equal frame counts by density
- ensure each split has enough visible pedestrian pixels, radar-supported
  vehicles, and mid-range objects

### Training

First V2 recipe:

- keep `768x432`, batch size `2`, AdamW
- initialize from V1 best checkpoint or the pole RGB checkpoint depending on
  whether the new radar channels and object head should be preserved
- increase person segmentation weight modestly
- increase center/location loss weight carefully
- consider training longer only after data balance improves

Potential structural changes if V2 data alone does not fix recall:

- sub-pixel center offset regression
- larger object head hidden channels
- stronger multi-scale object head
- class-specific regression heads if person/vehicle localization conflict

### Evaluation

Report V1 and V2 side by side:

- segmentation mIoU, vehicle IoU, person IoU
- object precision, recall, F1 at strict `3 m`
- vehicle/person XY MAE
- breakdown by density
- recall by distance and bbox area
- qualitative crowded/crosswalk examples

This preserves the research story:

1. V1 proved parked-ego RGB+radar fusion improves segmentation.
2. V1 exposed localization recall limits under crowding and small/far objects.
3. V2 addresses those limits with targeted data and training changes.
