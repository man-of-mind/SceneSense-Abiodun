# Hybrid-q Phase 6 — validation accuracy–payload curve (stable epoch-4 ranker)

Terminal: `HYBRID_Q_PHASE6_VALIDATION_CURVE_COMPLETE`  
Generated: 2026-09-02T18:59:37.760638+00:00  
Artifact: `experiments/splitfusion_fcos_hybrid_q_v1/20260902_182401_phase6_validation_curve`

## What this phase is

Measurement of the validation accuracy-payload curve; not checkpoint selection and not model development.

ranker_epoch_04.pt is the stable distillation-only checkpoint: it is taken at the end of the four distillation epochs, before the q-aware stage. The Phase-5 q-aware training failure is unchanged by this measurement phase; epochs 8 and 12 were neither loaded nor evaluated.

Bound inputs, all verified by exact SHA-256 before any inference:

- Stable ranker: `experiments/splitfusion_fcos_hybrid_q_v1/20260901_185725_phase5_ranker_training/checkpoints/ranker_epoch_04.pt`  
  `07781c56a4c0f306f16d332f64627ce6b9458e154f40ab9fef89f89909b79cb5` (epoch 4, distillation_only)
- Frozen perception checkpoint: `da14d21edbd374c1c3abce02ca4674b9f4097becfba9759aba945cea160a297f`
- p025 forward lock: `86d6f13ae9168b33b697df5b785c5f7c320afc52cfdcded5b632d94a6d943fe1`
- Hybrid-q locked config: `b2b0d8427bd867f46058ebba49ac6a183eb89413b4d69326fef93b150ebfcde6`

Validation split: 3345 frames, episodes `canonical_v3_05_val_30_30_s601_tm1601`, `canonical_v3_06_val_50_50_s602_tm1602`.
q=0 inference rerun: **False** — the frozen p025 q=0 validation result is reused verbatim. Test split accessed: **False**.

The Phase-6 scoring path reproduces all 15 published frozen q=0 validation values exactly, so every q>0 row below is measured on the same scoring semantics as the reference row.

## Payload

| q | retained cells | dropped cells | framed FP32 payload B | ratio vs q=0 |
|---|---:|---:|---:|---:|
| 0.00 | 21,504 | 0 | 22,020,140 | 1.000000 |
| 0.30 | 15,053 | 6,451 | 15,417,004 | 0.700132 |
| 0.50 | 10,752 | 10,752 | 11,012,780 | 0.500123 |
| 0.70 | 6,451 | 15,053 | 6,608,556 | 0.300114 |
| 0.90 | 2,150 | 19,354 | 2,204,332 | 0.100105 |
| 0.98 | 430 | 21,074 | 443,052 | 0.020120 |

Framed q=0 denominator = 22,020,140 B (44-byte header + dense FP32 payload). The unframed raw FP32 tensor reference is 22,020,096 B and is reported separately.

## Detection accuracy

| q | veh P | veh R | veh F1 | person AVO≥0.65 P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.9316 | 0.8684 | 0.8989 | 0.7042 | 0.7132 | 0.7087 |
| 0.30 | 0.9344 | 0.8664 | 0.8991 | 0.6404 | 0.7299 | 0.6823 |
| 0.50 | 0.9345 | 0.8630 | 0.8973 | 0.6305 | 0.7313 | 0.6772 |
| 0.70 | 0.9179 | 0.8551 | 0.8854 | 0.6310 | 0.7209 | 0.6729 |
| 0.90 | 0.8874 | 0.7612 | 0.8195 | 0.5625 | 0.6270 | 0.5930 |
| 0.98 | 0.8273 | 0.3999 | 0.5391 | 0.5476 | 0.3757 | 0.4457 |

## Localization, long range and segmentation

| q | veh XY MAE m | person XY MAE m | person recall 20–40 m | veh IoU | person box-mask IoU | foreground mIoU |
|---|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.4787 | 0.8122 | 0.5777 | 0.8990 | 0.5279 | 0.7135 |
| 0.30 | 0.4860 | 0.8099 | 0.6017 | 0.8749 | 0.5012 | 0.6881 |
| 0.50 | 0.4999 | 0.8054 | 0.6034 | 0.8214 | 0.4903 | 0.6559 |
| 0.70 | 0.5343 | 0.8193 | 0.5954 | 0.7036 | 0.4596 | 0.5816 |
| 0.90 | 0.6683 | 0.8360 | 0.4854 | 0.4034 | 0.3666 | 0.3850 |
| 0.98 | 0.9445 | 0.9561 | 0.2269 | 0.0792 | 0.1785 | 0.1288 |

## Absolute change from q=0

Signed difference candidate − q=0. Negative is worse for precision, recall, F1 and IoU; positive is worse for the XY MAE columns.

| q | veh P | veh R | veh F1 | person P | person R | person F1 | veh XY | person XY | R 20–40 m | veh IoU | box-mask IoU | fg mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.30 | +0.0029 | -0.0021 | +0.0002 | -0.0637 | +0.0167 | -0.0264 | +0.0073 | -0.0023 | +0.0241 | -0.0241 | -0.0267 | -0.0254 |
| 0.50 | +0.0029 | -0.0055 | -0.0016 | -0.0737 | +0.0181 | -0.0315 | +0.0212 | -0.0068 | +0.0258 | -0.0776 | -0.0376 | -0.0576 |
| 0.70 | -0.0137 | -0.0133 | -0.0135 | -0.0732 | +0.0076 | -0.0357 | +0.0556 | +0.0071 | +0.0178 | -0.1954 | -0.0682 | -0.1318 |
| 0.90 | -0.0442 | -0.1072 | -0.0794 | -0.1417 | -0.0862 | -0.1157 | +0.1896 | +0.0238 | -0.0923 | -0.4956 | -0.1613 | -0.3285 |
| 0.98 | -0.1043 | -0.4686 | -0.3598 | -0.1566 | -0.3375 | -0.2630 | +0.4658 | +0.1439 | -0.3507 | -0.8198 | -0.3494 | -0.5846 |

## Gate counts and action profile

| q | absolute service gates passed (of 9) | near-lossless preservation gates passed (of 12) | classification |
|---|---:|---:|---|
| 0.00 | 7/9 | 12/12 (reference) | **accuracy-first** |
| 0.30 | 7/9 | 7/12 | **localization-preserving/segmentation-reduced** |
| 0.50 | 4/9 | 7/12 | **localization-preserving/segmentation-reduced** |
| 0.70 | 4/9 | 3/12 | **localization-preserving/segmentation-reduced** |
| 0.90 | 3/9 | 1/12 | **unusable** |
| 0.98 | 3/9 | 0/12 | **unusable** |

The nine absolute service targets are the original deployment targets from the registered evaluator; the q=0 reference itself passes 7/9, failing `person_precision` and `person_recall`, exactly as the p025 forward lock records. The preservation gates are the registered near-lossless bounds, reported as one characterization column; a q is never rejected here for failing the earlier near-lossless gates.

Which gates fail, per q:

| q | failed absolute service targets | failed preservation gates |
|---|---|---|
| 0.00 | `person_precision`, `person_recall` | none (reference) |
| 0.30 | `person_precision`, `person_recall` | `foreground_miou`, `person_avo_f1`, `person_avo_precision`, `person_box_mask_iou`, `vehicle_iou` |
| 0.50 | `foreground_miou`, `person_box_mask_iou`, `person_precision`, `person_recall`, `vehicle_iou` | `foreground_miou`, `person_avo_f1`, `person_avo_precision`, `person_box_mask_iou`, `vehicle_iou` |
| 0.70 | `foreground_miou`, `person_box_mask_iou`, `person_precision`, `person_recall`, `vehicle_iou` | `foreground_miou`, `person_avo_f1`, `person_avo_precision`, `person_box_mask_iou`, `vehicle_f1`, `vehicle_iou`, `vehicle_precision`, `vehicle_recall`, `vehicle_xy_mae_m` |
| 0.90 | `foreground_miou`, `person_box_mask_iou`, `person_precision`, `person_recall`, `vehicle_iou`, `vehicle_recall` | `foreground_miou`, `person_avo_f1`, `person_avo_precision`, `person_avo_recall`, `person_avo_recall_20_40m`, `person_box_mask_iou`, `vehicle_f1`, `vehicle_iou`, `vehicle_precision`, `vehicle_recall`, `vehicle_xy_mae_m` |
| 0.98 | `foreground_miou`, `person_box_mask_iou`, `person_precision`, `person_recall`, `vehicle_iou`, `vehicle_recall` | `foreground_miou`, `person_avo_f1`, `person_avo_precision`, `person_avo_recall`, `person_avo_recall_20_40m`, `person_avo_xy_mae_m`, `person_box_mask_iou`, `vehicle_f1`, `vehicle_iou`, `vehicle_precision`, `vehicle_recall`, `vehicle_xy_mae_m` |

Classification cascade, registered before the measurement (`contract.VALIDATION_PROFILE_CASCADE`), first match wins:

1. **unusable** — person AVO F1 or vehicle F1 collapses by more than 0.20 absolute, or at most three of the nine absolute service targets survive.
1. **accuracy-first** — every registered near-lossless preservation gate passes.
1. **balanced** — person AVO F1 loss <= 0.05, vehicle F1 loss <= 0.02 and foreground mIoU loss <= 0.02.
1. **localization-preserving/segmentation-reduced** — both XY MAE increases stay within 0.10 m while foreground mIoU loss exceeds 0.02.
1. **emergency-bandwidth** — still finite and scientifically usable, but detection or segmentation quality is materially reduced.

Scientifically usable q settings, all of which remain available as agent actions: `0.00`, `0.30`, `0.50`, `0.70`.

Measured but not scientifically usable: `0.90`, `0.98`.

## Reading the curve

- **Vehicle detection is the most drop-tolerant head.** Vehicle F1 is within 0.002 of the q=0 reference all the way to q=0.50 at half the payload, and vehicle precision is very slightly *higher* at q=0.30 and q=0.50 than at q=0.
- **Segmentation is the first-order casualty, well before detection.** Vehicle IoU loses 0.078 by q=0.50 and 0.195 by q=0.70, and foreground mIoU loses 0.058 and 0.132 respectively, while detection F1 is still nearly intact. This is the same ordering the earlier density-knob study found, where ROI drop destroyed segmentation before it hurt detection.
- **Localization is preserved much further than quality is.** Vehicle XY MAE rises only 0.056 m and person XY MAE only 0.007 m out to q=0.70; that is why q=0.30-0.70 land in the localization-preserving band rather than the balanced one.
- **The person head shifts its operating point rather than degrading symmetrically.** Person AVO precision falls immediately (-0.064 at q=0.30) while person AVO recall *rises* (+0.017), and 20-40 m person recall rises above the q=0 reference at q=0.30, 0.50 and 0.70 (+0.024, +0.026, +0.018). Sparsification is therefore not a uniform accuracy tax on people: it trades precision for recall, including at long range.
- **The usable ladder ends between q=0.70 and q=0.90.** Every metric falls off sharply at q=0.90 (vehicle IoU 0.403, foreground mIoU 0.385, 20-40 m person recall 0.485) and collapses at q=0.98 (vehicle recall 0.400, vehicle IoU 0.079). Both cross the registered collapse condition and are labelled unusable.
- **The 7/9 -> 4/9 service step happens between q=0.30 and q=0.50**, and it is driven entirely by the three segmentation-side targets (`vehicle_iou`, `person_box_mask_iou`, `foreground_miou`), not by detection or localization.

No accuracy-first or balanced rung exists on this curve: the smallest measured drop, q=0.30, already exceeds the near-lossless segmentation bounds. Measuring a denser ladder below q=0.30 is the way to find one, and this phase does not assume it exists.

## Future continuous-q readiness

- The ranker consumes detached fused C2 only and never sees q (`ranker_sees_q: False`). Over 16 validation frames, every registered q selection is a prefix of **one** q-independent spatial ordering: `True`.
- Registered q masks are nested: `True` (keep-set at a larger q is a subset of every smaller q).
- Deterministic keep-count convention for an arbitrary future q: **K(q) = round((1 − q) × 21,504)**, bounded to the supported range. It reproduces every registered keep count: `True`.
- An arbitrary cutoff such as q=0.55 is constructible from the same ordering without retraining: `True` (K=9,677, nested between q=0.50 and q=0.70). It was never encoded or transported and its accuracy was not measured.
- Unmeasured-q accuracy is **not** interpolated and **not** validated (`True`).
- Recommendation: snap a requested continuous q down to the nearest validated, less-aggressive q (contract.snap_continuous_q) until a denser validation sweep is completed.
- The ranker was not retrained and the discrete production contract is unchanged (`production_contract_changed: False`).

## What this phase did not do

- training: `False`
- tuning or recalibration: `False`
- threshold changes: `False`
- ranker modification: `False`
- teacher-map recomputation: `False`
- new cache creation: `False`
- zstd or INT8 measurement: `False`
- test-set access: `False`
- CARLA launch: `False`
- excluded ranker epochs: [8, 12] — the Phase-5 q-aware stage diverged; those checkpoints are not reopened and the training failure is unchanged by this measurement phase

Wall time: 35.6 min. Frozen perception and ranker state unchanged at end: `True`.

HYBRID_Q_PHASE6_VALIDATION_CURVE_COMPLETE
