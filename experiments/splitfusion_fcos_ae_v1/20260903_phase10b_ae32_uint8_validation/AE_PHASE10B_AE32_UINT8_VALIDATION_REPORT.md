# Phase 10B — selected AE32 UINT8 + mandatory-zstd validation

Generated 2026-09-04T01:10:42.855019+00:00 · terminal `SPLITFUSION_AE32_PHASE10B_UINT8_VALIDATION_COMPLETE`

One frozen measurement of the six registered q anchors on the 3,345 registered validation frames, one inference/evaluation pass per q. Nothing was trained, tuned, recalibrated, reselected or removed; no threshold, NMS setting, scorer or geometry evaluator changed; test data and CARLA were never opened. Component latency below is current-host diagnostic evidence only — no Raspberry Pi and no OAI latency is claimed.

## Deployment path measured

```text
original FP32 C2 -> AE32 encoder (complete frame)
  -> ranges from the complete latent -> per-channel UINT8
  -> stable per-frame top-K (q>0) -> family-labelled sparse wire
  -> mandatory zstd-1 -> received raw bytes
  -> exactly one decompression
  -> decoder selected from header family/bottleneck/routing tag
  -> dequantize / zero scatter -> AE32 decoder
  -> unchanged frozen perception tail and p025 service policy
```

Selected checkpoint `experiments/splitfusion_fcos_ae_v1/20260903_phase10_ae32_training/checkpoints/ae32_epoch_08.pt` (sha256 `e2f867757e8db0620316c092264ac7eb53d12bb5ef66ed14475eb40693d1f271`), epoch 8, 32-channel latent, routing tag `0xe2f86775` derived from that full digest. The 32-bit tag routes a frame to the decoder that produced it; it is not the checkpoint's identity.

## Payload

| q | keep | pre-zstd mean B | median | p95 | zstd mean B | median | p95 | vs framed FP32 noAE q0 | vs noAE UINT8+zstd same q | vs AE32 UINT8+zstd q0 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 21,504 | 688,434 | 688,434 | 688,434 | 601,546 | 602,242 | 609,642 | 0.027350 | 0.168774 | 1.000000 |
| 0.30 | 15,053 | 484,690 | 484,690 | 484,690 | 428,306 | 428,462 | 432,569 | 0.019458 | 0.170449 | 0.711445 |
| 0.50 | 10,752 | 347,058 | 347,058 | 347,058 | 308,477 | 308,425 | 311,046 | 0.014006 | 0.172530 | 0.512128 |
| 0.70 | 6,451 | 209,426 | 209,426 | 209,426 | 186,886 | 186,913 | 188,275 | 0.008488 | 0.175804 | 0.310362 |
| 0.90 | 2,150 | 71,794 | 71,794 | 71,794 | 63,342 | 63,351 | 63,793 | 0.002877 | 0.181359 | 0.105192 |
| 0.98 | 430 | 16,754 | 16,754 | 16,754 | 13,246 | 13,246 | 13,365 | 0.000602 | 0.186225 | 0.021994 |

Ratios use median bytes on both sides. The frozen noAE UINT8+zstd reference publishes no mean compressed size, so no mean-vs-mean ratio against it is reported.

## Accuracy

| q | vehicle P/R/F1/XY | canonical-p025 person P/R/F1/XY | AVO>=0.65 person P/R/F1/XY | person 20–40 m recall (historical) | vehicle IoU | person box-mask IoU | foreground mIoU | service gates | same-q gates | profile |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.00 | 0.930716/0.863585/0.895895/0.510952 | 0.728412/0.555527/0.630330/0.849992 | 0.648703/0.669447/0.658912/0.827598 | 0.519771 | 0.889349 | 0.480285 | 0.684817 | 6/9 | 6/12 | primary profile |
| 0.30 | 0.926405/0.863791/0.894003/0.512467 | 0.699048/0.568698/0.627172/0.850551 | 0.613317/0.681960/0.645820/0.824964 | 0.536390 | 0.877135 | 0.461722 | 0.669429 | 5/9 | 6/12 | primary profile |
| 0.50 | 0.914273/0.867196/0.890113/0.517675 | 0.680305/0.575413/0.623478/0.853926 | 0.593037/0.686827/0.636495/0.824976 | 0.543266 | 0.844628 | 0.451665 | 0.648146 | 4/9 | 6/12 | primary profile |
| 0.70 | 0.885953/0.864926/0.875313/0.551153 | 0.626816/0.579545/0.602254/0.870096 | 0.539341/0.688564/0.604885/0.842411 | 0.545559 | 0.763400 | 0.428376 | 0.595888 | 4/9 | 6/12 | primary profile |
| 0.90 | 0.812336/0.767826/0.789454/0.741265 | 0.542842/0.517045/0.529630/0.891658 | 0.448452/0.609315/0.516652/0.851706 | 0.453868 | 0.445378 | 0.325370 | 0.385374 | 3/9 | 4/12 | stress/emergency profile |
| 0.98 | 0.609243/0.394490/0.478893/1.045700 | 0.453729/0.260847/0.331256/0.934786 | 0.352411/0.312478/0.331245/0.901493 | 0.181662 | 0.142451 | 0.119582 | 0.131016 | 1/9 | 4/12 | stress/emergency profile |

Canonical-p025 person metrics are diagnostics. The twelve preservation gates and the secondary localization-priority classification both use the AVO>=0.65 visible-object person view.

## Pedestrian range stratification

Primary operating range: `0 <= gt_distance_m < 30`. Extended diagnostic range: `30 <= gt_distance_m <= 40`. Only `person_avo_recall_0_30m >= 0.70` is an absolute tier gate; every other row below is reported and never gated.

The 30 m boundary is evaluation-only. It does not filter, suppress, relabel, rescore or otherwise change any runtime detection. Deployment continues to emit every detection accepted by the frozen p025 pipeline throughout its existing range. Ground-truth distance is available only to the evaluator, so the boundary is not runtime-computable and is never applied to model output.

| q | 0-10 m R | 10-20 m R | 20-30 m R | **0-30 m R (gate)** | 30-40 m R (stress) | 20-40 m R (historical) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.919355 | 0.897817 | 0.703187 | **0.807584** | 0.271255 | 0.519771 |
| 0.30 | 0.911290 | 0.905754 | 0.719124 | **0.818352** | 0.288799 | 0.536390 |
| 0.50 | 0.895161 | 0.909722 | 0.719124 | **0.819288** | 0.304993 | 0.543266 |
| 0.70 | 0.879032 | 0.912698 | 0.708167 | **0.814607** | 0.325236 | 0.545559 |
| 0.90 | 0.806452 | 0.854167 | 0.617530 | **0.740169** | 0.232119 | 0.453868 |
| 0.98 | 0.451613 | 0.521825 | 0.283865 | **0.405899** | 0.043185 | 0.181662 |

20-30 m is shown on its own so the cumulative 0-30 m result cannot hide boundary-band behaviour, and 30-40 m is retained as extended-range stress. The frozen AVO scorer publishes each distance bin as a recall slice (eligible_gt / tp / fn) only: a false positive is not attributed to a range, because doing so would require binning predictions by predicted distance, which is new matching logic this correction does not introduce. Per-band precision is therefore not derivable from the frozen artifacts, and aggregate AVO precision remains the precision gate.

Range provenance:

> The 0-30 m primary operating range was selected from frozen noAE range-stratified analysis and literature context before Phase-10B AE64/AE32 validation. The 30-40 m results remain reported as extended-range stress. Independent test-set confirmation has not been performed.

## Failed gates and exact degradations

| q | failed same-q preservation gates | degradation / bound | failed absolute service gates |
| ---: | --- | --- | --- |
| 0.00 | foreground_miou, person_avo_f1, person_avo_precision, person_avo_recall, person_avo_recall_20_40m, person_box_mask_iou | foreground_miou +0.028591 / 0.01; person_avo_f1 +0.049306 / 0.015; person_avo_precision +0.054899 / 0.015; person_avo_recall +0.043448 / 0.015; person_avo_recall_20_40m +0.057880 / 0.03; person_box_mask_iou +0.047534 / 0.01 | person_box_mask_iou, person_precision, person_recall |
| 0.30 | foreground_miou, person_avo_f1, person_avo_precision, person_avo_recall, person_avo_recall_20_40m, person_box_mask_iou | foreground_miou +0.018643 / 0.01; person_avo_f1 +0.035674 / 0.015; person_avo_precision +0.026038 / 0.015; person_avo_recall +0.047619 / 0.015; person_avo_recall_20_40m +0.064183 / 0.03; person_box_mask_iou +0.039513 / 0.01 | foreground_miou, person_box_mask_iou, person_precision, person_recall |
| 0.50 | person_avo_f1, person_avo_precision, person_avo_recall, person_avo_recall_20_40m, person_box_mask_iou, vehicle_precision | person_avo_f1 +0.039706 / 0.015; person_avo_precision +0.035773 / 0.015; person_avo_recall +0.044491 / 0.015; person_avo_recall_20_40m +0.060172 / 0.03; person_box_mask_iou +0.038623 / 0.01; vehicle_precision +0.020327 / 0.01 | foreground_miou, person_box_mask_iou, person_precision, person_recall, vehicle_iou |
| 0.70 | person_avo_f1, person_avo_precision, person_avo_recall, person_avo_recall_20_40m, person_box_mask_iou, vehicle_precision | person_avo_f1 +0.067833 / 0.015; person_avo_precision +0.090975 / 0.015; person_avo_recall +0.032673 / 0.015; person_avo_recall_20_40m +0.050430 / 0.03; person_box_mask_iou +0.031289 / 0.01; vehicle_precision +0.031646 / 0.01 | foreground_miou, person_box_mask_iou, person_precision, person_recall, vehicle_iou |
| 0.90 | person_avo_f1, person_avo_precision, person_avo_recall, person_avo_recall_20_40m, person_box_mask_iou, vehicle_f1, vehicle_precision, vehicle_xy_mae_m | person_avo_f1 +0.076780 / 0.015; person_avo_precision +0.113950 / 0.015; person_avo_recall +0.018770 / 0.015; person_avo_recall_20_40m +0.032665 / 0.03; person_box_mask_iou +0.041183 / 0.01; vehicle_f1 +0.029959 / 0.01; vehicle_precision +0.074762 / 0.01; vehicle_xy_mae_m +0.072703 / 0.05 | foreground_miou, person_box_mask_iou, person_precision, person_recall, vehicle_iou, vehicle_recall |
| 0.98 | person_avo_f1, person_avo_precision, person_avo_recall, person_avo_recall_20_40m, person_box_mask_iou, vehicle_f1, vehicle_precision, vehicle_xy_mae_m | person_avo_f1 +0.114572 / 0.015; person_avo_precision +0.194883 / 0.015; person_avo_recall +0.063608 / 0.015; person_avo_recall_20_40m +0.045272 / 0.03; person_box_mask_iou +0.058959 / 0.01; vehicle_f1 +0.059735 / 0.01; vehicle_precision +0.216557 / 0.01; vehicle_xy_mae_m +0.102696 / 0.05 | foreground_miou, person_box_mask_iou, person_precision, person_recall, vehicle_iou, vehicle_precision, vehicle_recall, vehicle_xy_mae_m |

## Primary preregistered interpretation (relative, 12 gates)

AE32 UINT8+zstd deployment is accepted if and only if both hold: (1) q=0 passes all 12 same-q preservation gates against the frozen noAE UINT8+zstd validation result and retains at least the baseline 7/9 absolute service gates; and (2) at least one of q in {0.30, 0.50, 0.70} passes all 12 same-q preservation gates without reducing the absolute service-gate count below the frozen noAE UINT8+zstd count at that same q. q=0.90 and q=0.98 are stress/emergency profiles regardless of their results and cannot make or break acceptance. Every q is reported independently, and no setting is tuned or removed after observing a result.

- q=0 condition: **not met** (6/12 same-q gates, 6/9 absolute service gates against a 7/9 baseline)
- qualifying primary q: none
- **decision: AE32_UINT8_ZSTD_DEPLOYMENT_NOT_ACCEPTED**
- q=0.90 and q=0.98 are stress/emergency profiles regardless of their results and did not enter the decision
- this decision is a *relative* preservation result against the frozen noAE UINT8+zstd row at the same q. It is not an absolute service claim, and it does not by itself authorize replacing the spatial-map segmentation layer
- every measured q is reported above whatever this decision was: a failed acceptance suppressed 0 rows

## Secondary prospective classification (absolute AVO/object)

Holdout-informed thresholds frozen before AE64/AE32 held-out deployment validation. The validation frames were not used for AE training or checkpoint selection.

This is not an independent or untouched test-set confirmation. It changed no checkpoint selection, no primary acceptance terminal, and no threshold, NMS setting, model or scorer, and it neither erases nor reinterprets any preservation failure recorded above.

| requirement | target |
| --- | ---: |
| `vehicle_precision` | >= 0.8 |
| `vehicle_recall` | >= 0.85 |
| `vehicle_xy_mae_m` | <= 1.0 |
| `person_avo_precision` | >= 0.7 |
| `person_avo_recall` | >= 0.7 |
| `person_avo_f1` | >= 0.7 |
| `person_avo_xy_mae_m` | <= 1.2 |
| `person_avo_recall_0_30m` | >= 0.7 |

The three segmentation outputs — vehicle IoU, person box-mask IoU and foreground mIoU — are measured and reported above and decide segmentation installability, but do not enter this classification.

| q | tier | object requirements | failed | segmentation installable | segmentation action | 9/9 service ready |
| ---: | --- | ---: | --- | ---: | --- | ---: |
| 0.00 | `EMERGENCY_ONLY` | 5/8 | person_avo_f1, person_avo_precision, person_avo_recall | False | `retain_previous_segmentation_layer_with_original_timestamp` | False |
| 0.30 | `EMERGENCY_ONLY` | 5/8 | person_avo_f1, person_avo_precision, person_avo_recall | False | `retain_previous_segmentation_layer_with_original_timestamp` | False |
| 0.50 | `EMERGENCY_ONLY` | 5/8 | person_avo_f1, person_avo_precision, person_avo_recall | False | `retain_previous_segmentation_layer_with_original_timestamp` | False |
| 0.70 | `EMERGENCY_ONLY` | 5/8 | person_avo_f1, person_avo_precision, person_avo_recall | False | `retain_previous_segmentation_layer_with_original_timestamp` | False |
| 0.90 | `EMERGENCY_ONLY` | 4/8 | person_avo_f1, person_avo_precision, person_avo_recall, vehicle_recall | False | `retain_previous_segmentation_layer_with_original_timestamp` | False |
| 0.98 | `EMERGENCY_ONLY` | 1/8 | person_avo_f1, person_avo_precision, person_avo_recall, person_avo_recall_0_30m, vehicle_precision, vehicle_recall, vehicle_xy_mae_m | False | `retain_previous_segmentation_layer_with_original_timestamp` | False |

segmentation_installable = vehicle_iou >= 0.85 and person_box_mask_iou >= 0.50 and foreground_miou >= 0.675. A 12/12 relative-preservation result does not by itself authorize replacing the spatial-map segmentation layer. Install new segmentation only when segmentation_installable is true; otherwise retain the previous segmentation layer with its original timestamp.

SERVICE_READY is a separate absolute result: all nine registered absolute service gates pass at this profile. It is never derived from, implied by or substituted for a 12/12 relative preservation result.

perception degradation changes a profile's tier and therefore its reward; it never permanently masks the action. Only technical invalidity (INVALID) or a hard state-dependent resource constraint (STATE_INFEASIBLE) may mask an action.

`STATE_INFEASIBLE` is reserved for a runtime action that a hard state-dependent resource constraint makes unavailable in the current state -- for example a payload that does not fit the instantaneous transport budget. It is a runtime availability verdict about a state, not a measurement outcome about a profile, so this offline validation never assigns it: every registered q is measured and reported here.

## Integrity

- family: AE32, family id 3, 32 transported latent channels
- validation frames per q: 3,345
- q settings completed exactly once: 6/6
- every frame carried the AE32 family id, a 32-channel latent and the bound routing tag in its own header
- every frame was decompressed exactly once, and the decoder was discovered from the received header bytes alone
- retained UINT8 cells were exactly the selected cells; dropped cells scattered to exact zero before reconstruction
- q=0 invoked the ranker zero times and AE32 every time; no q produced an identity reconstruction
- frozen perception, stable ranker and selected AE32 parameters and buffers were unchanged
- per q the setting JSON was fsynced into place first, its predictions were removed only afterwards, and the cleanup marker was written last, so an interruption could only lose scratch predictions
- only compact evidence is retained; no prediction directory survives
