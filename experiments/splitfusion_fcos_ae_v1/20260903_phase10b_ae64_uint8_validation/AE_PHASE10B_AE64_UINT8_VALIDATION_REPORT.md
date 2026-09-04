# Phase 10B — selected AE64 UINT8 + mandatory-zstd validation

Generated 2026-09-04T00:46:17.914287+00:00 · terminal `SPLITFUSION_AE64_PHASE10B_UINT8_VALIDATION_COMPLETE`

One frozen measurement of the six registered q anchors on the 3,345 registered validation frames, one inference/evaluation pass per q. Nothing was trained, tuned, recalibrated, reselected or removed; no threshold, NMS setting, scorer or geometry evaluator changed; test data and CARLA were never opened. Component latency below is current-host diagnostic evidence only — no Raspberry Pi and no OAI latency is claimed.

## Deployment path measured

```text
original FP32 C2 -> AE64 encoder (complete frame)
  -> ranges from the complete latent -> per-channel UINT8
  -> stable per-frame top-K (q>0) -> family-labelled sparse wire
  -> mandatory zstd-1 -> received raw bytes
  -> exactly one decompression
  -> decoder selected from header family/bottleneck/routing tag
  -> dequantize / zero scatter -> AE64 decoder
  -> unchanged frozen perception tail and p025 service policy
```

Selected checkpoint `experiments/splitfusion_fcos_ae_v1/20260903_phase10_ae64_training/checkpoints/ae64_epoch_12.pt` (sha256 `dd7c5124e27114584ab2083e59160a3ff2a2d040d0a37d22564ac98c838aa8e0`), epoch 12, 64-channel latent, routing tag `0xdd7c5124` derived from that full digest. The 32-bit tag routes a frame to the decoder that produced it; it is not the checkpoint's identity.

## Payload

| q | keep | pre-zstd mean B | median | p95 | zstd mean B | median | p95 | vs framed FP32 noAE q0 | vs noAE UINT8+zstd same q | vs AE64 UINT8+zstd q0 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 21,504 | 1,376,818 | 1,376,818 | 1,376,818 | 1,215,518 | 1,216,049 | 1,227,762 | 0.055224 | 0.340790 | 1.000000 |
| 0.30 | 15,053 | 966,642 | 966,642 | 966,642 | 861,951 | 862,189 | 867,327 | 0.039155 | 0.342992 | 0.709008 |
| 0.50 | 10,752 | 691,378 | 691,378 | 691,378 | 619,274 | 619,563 | 622,664 | 0.028136 | 0.346578 | 0.509489 |
| 0.70 | 6,451 | 416,114 | 416,114 | 416,114 | 374,155 | 374,264 | 376,063 | 0.016996 | 0.352020 | 0.307770 |
| 0.90 | 2,150 | 140,850 | 140,850 | 140,850 | 126,216 | 126,237 | 126,785 | 0.005733 | 0.361388 | 0.103809 |
| 0.98 | 430 | 30,770 | 30,770 | 30,770 | 26,101 | 26,102 | 26,263 | 0.001185 | 0.366967 | 0.021465 |

Ratios use median bytes on both sides. The frozen noAE UINT8+zstd reference publishes no mean compressed size, so no mean-vs-mean ratio against it is reported.

## Accuracy

| q | vehicle P/R/F1/XY | canonical-p025 person P/R/F1/XY | AVO>=0.65 person P/R/F1/XY | person 20–40 m recall (historical) | vehicle IoU | person box-mask IoU | foreground mIoU | service gates | same-q gates | profile |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.00 | 0.930382/0.867403/0.897789/0.493256 | 0.772332/0.573864/0.658468/0.842778 | 0.691693/0.691693/0.691693/0.818532 | 0.547851 | 0.896174 | 0.521451 | 0.708812 | 7/9 | 10/12 | primary profile |
| 0.30 | 0.928272/0.868022/0.897136/0.497788 | 0.745974/0.586260/0.656544/0.838668 | 0.658735/0.702468/0.679899/0.814125 | 0.561032 | 0.885935 | 0.511463 | 0.698699 | 7/9 | 10/12 | primary profile |
| 0.50 | 0.913580/0.870498/0.891519/0.499260 | 0.744307/0.590909/0.658796/0.835027 | 0.652917/0.704206/0.677592/0.803960 | 0.563897 | 0.851795 | 0.501417 | 0.676606 | 7/9 | 9/12 | primary profile |
| 0.70 | 0.894206/0.869570/0.881716/0.535287 | 0.693442/0.595300/0.640634/0.850361 | 0.595022/0.706291/0.645900/0.813417 | 0.572493 | 0.766854 | 0.464610 | 0.615732 | 4/9 | 9/12 | primary profile |
| 0.90 | 0.817257/0.771128/0.793523/0.724005 | 0.561447/0.525052/0.542640/0.882412 | 0.469729/0.625652/0.536593/0.840396 | 0.472779 | 0.428602 | 0.353591 | 0.391096 | 3/9 | 6/12 | stress/emergency profile |
| 0.98 | 0.629267/0.424208/0.506780/1.048471 | 0.488732/0.285640/0.360554/0.966264 | 0.392219/0.343413/0.366197/0.943572 | 0.206304 | 0.145745 | 0.138233 | 0.141989 | 1/9 | 5/12 | stress/emergency profile |

Canonical-p025 person metrics are diagnostics. The twelve preservation gates and the secondary localization-priority classification both use the AVO>=0.65 visible-object person view.

## Pedestrian range stratification

Primary operating range: `0 <= gt_distance_m < 30`. Extended diagnostic range: `30 <= gt_distance_m <= 40`. Only `person_avo_recall_0_30m >= 0.70` is an absolute tier gate; every other row below is reported and never gated.

The 30 m boundary is evaluation-only. It does not filter, suppress, relabel, rescore or otherwise change any runtime detection. Deployment continues to emit every detection accepted by the frozen p025 pipeline throughout its existing range. Ground-truth distance is available only to the evaluator, so the boundary is not runtime-computable and is never applied to model output.

| q | 0-10 m R | 10-20 m R | 20-30 m R | **0-30 m R (gate)** | 30-40 m R (stress) | 20-40 m R (historical) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.919355 | 0.912698 | 0.709163 | **0.817416** | 0.329285 | 0.547851 |
| 0.30 | 0.927419 | 0.919643 | 0.717131 | **0.824906** | 0.349528 | 0.561032 |
| 0.50 | 0.919355 | 0.920635 | 0.714143 | **0.823502** | 0.360324 | 0.563897 |
| 0.70 | 0.879032 | 0.916667 | 0.711155 | **0.817884** | 0.384615 | 0.572493 |
| 0.90 | 0.822581 | 0.866071 | 0.629482 | **0.752341** | 0.260459 | 0.472779 |
| 0.98 | 0.467742 | 0.565476 | 0.311753 | **0.440543** | 0.063428 | 0.206304 |

20-30 m is shown on its own so the cumulative 0-30 m result cannot hide boundary-band behaviour, and 30-40 m is retained as extended-range stress. The frozen AVO scorer publishes each distance bin as a recall slice (eligible_gt / tp / fn) only: a false positive is not attributed to a range, because doing so would require binning predictions by predicted distance, which is new matching logic this correction does not introduce. Per-band precision is therefore not derivable from the frozen artifacts, and aggregate AVO precision remains the precision gate.

Range provenance:

> The 0-30 m primary operating range was selected from frozen noAE range-stratified analysis and literature context before Phase-10B AE64/AE32 validation. The 30-40 m results remain reported as extended-range stress. Independent test-set confirmation has not been performed.

## Failed gates and exact degradations

| q | failed same-q preservation gates | degradation / bound | failed absolute service gates |
| ---: | --- | --- | --- |
| 0.00 | person_avo_f1, person_avo_recall | person_avo_f1 +0.016525 / 0.015; person_avo_recall +0.021203 / 0.015 | person_precision, person_recall |
| 0.30 | person_avo_recall, person_avo_recall_20_40m | person_avo_recall +0.027112 / 0.015; person_avo_recall_20_40m +0.039542 / 0.03 | person_precision, person_recall |
| 0.50 | person_avo_recall, person_avo_recall_20_40m, vehicle_precision | person_avo_recall +0.027112 / 0.015; person_avo_recall_20_40m +0.039542 / 0.03; vehicle_precision +0.021020 / 0.01 | person_precision, person_recall |
| 0.70 | person_avo_f1, person_avo_precision, vehicle_precision | person_avo_f1 +0.026819 / 0.015; person_avo_precision +0.035294 / 0.015; vehicle_precision +0.023393 / 0.01 | foreground_miou, person_box_mask_iou, person_precision, person_recall, vehicle_iou |
| 0.90 | person_avo_f1, person_avo_precision, person_box_mask_iou, vehicle_f1, vehicle_precision, vehicle_xy_mae_m | person_avo_f1 +0.056839 / 0.015; person_avo_precision +0.092674 / 0.015; person_box_mask_iou +0.012963 / 0.01; vehicle_f1 +0.025891 / 0.01; vehicle_precision +0.069841 / 0.01; vehicle_xy_mae_m +0.055442 / 0.05 | foreground_miou, person_box_mask_iou, person_precision, person_recall, vehicle_iou, vehicle_recall |
| 0.98 | person_avo_f1, person_avo_precision, person_avo_recall, person_box_mask_iou, vehicle_f1, vehicle_precision, vehicle_xy_mae_m | person_avo_f1 +0.079621 / 0.015; person_avo_precision +0.155075 / 0.015; person_avo_recall +0.032673 / 0.015; person_box_mask_iou +0.040308 / 0.01; vehicle_f1 +0.031847 / 0.01; vehicle_precision +0.196533 / 0.01; vehicle_xy_mae_m +0.105467 / 0.05 | foreground_miou, person_box_mask_iou, person_precision, person_recall, vehicle_iou, vehicle_precision, vehicle_recall, vehicle_xy_mae_m |

## Primary preregistered interpretation (relative, 12 gates)

AE64 UINT8+zstd deployment is accepted if and only if both hold: (1) q=0 passes all 12 same-q preservation gates against the frozen noAE UINT8+zstd validation result and retains at least the baseline 7/9 absolute service gates; and (2) at least one of q in {0.30, 0.50, 0.70} passes all 12 same-q preservation gates without reducing the absolute service-gate count below the frozen noAE UINT8+zstd count at that same q. q=0.90 and q=0.98 are stress/emergency profiles regardless of their results and cannot make or break acceptance. Every q is reported independently, and no setting is tuned or removed after observing a result.

- q=0 condition: **not met** (10/12 same-q gates, 7/9 absolute service gates against a 7/9 baseline)
- qualifying primary q: none
- **decision: AE64_UINT8_ZSTD_DEPLOYMENT_NOT_ACCEPTED**
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
| 0.00 | `EMERGENCY_ONLY` | 5/8 | person_avo_f1, person_avo_precision, person_avo_recall | True | `install_new_segmentation` | False |
| 0.30 | `EMERGENCY_ONLY` | 6/8 | person_avo_f1, person_avo_precision | True | `install_new_segmentation` | False |
| 0.50 | `EMERGENCY_ONLY` | 6/8 | person_avo_f1, person_avo_precision | True | `install_new_segmentation` | False |
| 0.70 | `EMERGENCY_ONLY` | 6/8 | person_avo_f1, person_avo_precision | False | `retain_previous_segmentation_layer_with_original_timestamp` | False |
| 0.90 | `EMERGENCY_ONLY` | 4/8 | person_avo_f1, person_avo_precision, person_avo_recall, vehicle_recall | False | `retain_previous_segmentation_layer_with_original_timestamp` | False |
| 0.98 | `EMERGENCY_ONLY` | 1/8 | person_avo_f1, person_avo_precision, person_avo_recall, person_avo_recall_0_30m, vehicle_precision, vehicle_recall, vehicle_xy_mae_m | False | `retain_previous_segmentation_layer_with_original_timestamp` | False |

segmentation_installable = vehicle_iou >= 0.85 and person_box_mask_iou >= 0.50 and foreground_miou >= 0.675. A 12/12 relative-preservation result does not by itself authorize replacing the spatial-map segmentation layer. Install new segmentation only when segmentation_installable is true; otherwise retain the previous segmentation layer with its original timestamp.

SERVICE_READY is a separate absolute result: all nine registered absolute service gates pass at this profile. It is never derived from, implied by or substituted for a 12/12 relative preservation result.

perception degradation changes a profile's tier and therefore its reward; it never permanently masks the action. Only technical invalidity (INVALID) or a hard state-dependent resource constraint (STATE_INFEASIBLE) may mask an action.

`STATE_INFEASIBLE` is reserved for a runtime action that a hard state-dependent resource constraint makes unavailable in the current state -- for example a payload that does not fit the instantaneous transport budget. It is a runtime availability verdict about a state, not a measurement outcome about a profile, so this offline validation never assigns it: every registered q is measured and reported here.

## Integrity

- family: AE64, family id 2, 64 transported latent channels
- validation frames per q: 3,345
- q settings completed exactly once: 6/6
- every frame carried the AE64 family id, a 64-channel latent and the bound routing tag in its own header
- every frame was decompressed exactly once, and the decoder was discovered from the received header bytes alone
- retained UINT8 cells were exactly the selected cells; dropped cells scattered to exact zero before reconstruction
- q=0 invoked the ranker zero times and AE64 every time; no q produced an identity reconstruction
- frozen perception, stable ranker and selected AE64 parameters and buffers were unchanged
- per q the setting JSON was fsynced into place first, its predictions were removed only afterwards, and the cleanup marker was written last, so an interruption could only lose scratch predictions
- only compact evidence is retained; no prediction directory survives
