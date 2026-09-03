# Phase 9D — selected AE128 UINT8 + mandatory-zstd validation

Generated 2026-09-03T17:44:31.349277+00:00 · terminal `SPLITFUSION_AE128_UINT8_VALIDATION_COMPLETE`

One frozen measurement of the six registered q anchors on the 3,345 registered validation frames, one inference/evaluation pass per q. Nothing was trained, tuned, recalibrated or removed; no threshold, NMS setting, scorer or geometry evaluator changed; test data and CARLA were never opened. Component latency below is current-host diagnostic evidence only — no Raspberry Pi and no OAI latency is claimed.

## Deployment path measured

```text
original FP32 C2 -> AE128 encoder (complete frame) -> per-channel UINT8
  -> sparse AE wire -> mandatory zstd-1 -> received raw bytes
  -> exactly one decompression -> header-driven AE128 decoder selection
  -> dequantize / zero scatter -> AE128 decoder -> frozen perception tail
```

Selected checkpoint `experiments/splitfusion_fcos_ae_v1/20260902_220623_phase9c_ae128_training/checkpoints/ae128_epoch_08.pt`
(sha256 `0c2ba3a495684c0f8222492f554eb3de7c7a76181e0bd4b4a83529897db30f72`), routing tag `0x0c2ba3a4` derived from that full digest. The 32-bit tag routes a frame to the decoder that produced it; it is not the checkpoint's identity.

## Payload

| q | keep | pre-zstd mean B | median | p95 | zstd mean B | median | p95 | vs framed FP32 noAE q0 | vs noAE UINT8+zstd same q | vs AE128 UINT8+zstd q0 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 21,504 | 2,753,586 | 2,753,586 | 2,753,586 | 2,342,255 | 2,345,384 | 2,374,485 | 0.106511 | 0.657279 | 1.000000 |
| 0.30 | 15,053 | 1,930,546 | 1,930,546 | 1,930,546 | 1,673,603 | 1,673,599 | 1,689,775 | 0.076003 | 0.665783 | 0.713571 |
| 0.50 | 10,752 | 1,380,018 | 1,380,018 | 1,380,018 | 1,208,834 | 1,209,045 | 1,218,748 | 0.054906 | 0.676329 | 0.515500 |
| 0.70 | 6,451 | 829,490 | 829,490 | 829,490 | 733,069 | 733,074 | 737,450 | 0.033291 | 0.689504 | 0.312560 |
| 0.90 | 2,150 | 278,962 | 278,962 | 278,962 | 248,477 | 248,558 | 249,885 | 0.011288 | 0.711564 | 0.105978 |
| 0.98 | 430 | 58,802 | 58,802 | 58,802 | 51,426 | 51,422 | 51,832 | 0.002335 | 0.722940 | 0.021925 |

Ratios use median bytes on both sides. The frozen noAE UINT8+zstd reference publishes no mean compressed size, so no mean-vs-mean ratio against it is reported.

## Accuracy

| q | vehicle P/R/F1/XY | canonical-p025 person P/R/F1/XY | AVO>=0.65 person P/R/F1/XY | person 20–40 m recall | vehicle IoU | person box-mask IoU | foreground mIoU | service gates | same-q gates | profile |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.00 | 0.935434/0.867093/0.899968/0.481347 | 0.790779/0.584711/0.672309/0.839936 | 0.701110/0.702815/0.701961/0.815183 | 0.564470 | 0.897821 | 0.521265 | 0.709543 | 7/9 | 12/12 | primary profile |
| 0.30 | 0.930287/0.867506/0.897800/0.481959 | 0.751388/0.594008/0.663493/0.833387 | 0.652866/0.712548/0.681403/0.810999 | 0.577650 | 0.888288 | 0.510079 | 0.699183 | 7/9 | 11/12 | primary profile |
| 0.50 | 0.925406/0.870498/0.897113/0.489397 | 0.746129/0.597366/0.663511/0.826176 | 0.651429/0.713243/0.680936/0.799479 | 0.582235 | 0.854455 | 0.503997 | 0.679226 | 7/9 | 11/12 | primary profile |
| 0.70 | 0.905790/0.870086/0.887579/0.532106 | 0.696979/0.601756/0.645877/0.851848 | 0.595065/0.712548/0.648529/0.814280 | 0.582235 | 0.771622 | 0.472061 | 0.621842 | 4/9 | 9/12 | primary profile |
| 0.90 | 0.831281/0.773295/0.801240/0.732571 | 0.566538/0.529959/0.547638/0.875727 | 0.471728/0.626347/0.538151/0.838133 | 0.472206 | 0.443568 | 0.365945 | 0.404756 | 3/9 | 7/12 | stress/emergency profile |
| 0.98 | 0.624733/0.423279/0.504644/1.048120 | 0.510929/0.289773/0.369809/0.950103 | 0.410057/0.348627/0.376855/0.922459 | 0.215473 | 0.152215 | 0.143690 | 0.147953 | 1/9 | 5/12 | stress/emergency profile |

## Failed gates and exact degradations

| q | failed same-q preservation gates | degradation / bound | failed absolute service gates |
| ---: | --- | --- | --- |
| 0.00 | — | — | person_precision, person_recall |
| 0.30 | person_avo_recall | person_avo_recall +0.017032 / 0.015 | person_precision, person_recall |
| 0.50 | person_avo_recall | person_avo_recall +0.018074 / 0.015 | person_precision, person_recall |
| 0.70 | person_avo_f1, person_avo_precision, vehicle_precision | person_avo_f1 +0.024189 / 0.015; person_avo_precision +0.035251 / 0.015; vehicle_precision +0.011809 / 0.01 | foreground_miou, person_box_mask_iou, person_precision, person_recall, vehicle_iou |
| 0.90 | person_avo_f1, person_avo_precision, vehicle_f1, vehicle_precision, vehicle_xy_mae_m | person_avo_f1 +0.055280 / 0.015; person_avo_precision +0.090675 / 0.015; vehicle_f1 +0.018173 / 0.01; vehicle_precision +0.055818 / 0.01; vehicle_xy_mae_m +0.064008 / 0.05 | foreground_miou, person_box_mask_iou, person_precision, person_recall, vehicle_iou, vehicle_recall |
| 0.98 | person_avo_f1, person_avo_precision, person_avo_recall, person_box_mask_iou, vehicle_f1, vehicle_precision, vehicle_xy_mae_m | person_avo_f1 +0.068963 / 0.015; person_avo_precision +0.137237 / 0.015; person_avo_recall +0.027459 / 0.015; person_box_mask_iou +0.034851 / 0.01; vehicle_f1 +0.033983 / 0.01; vehicle_precision +0.201066 / 0.01; vehicle_xy_mae_m +0.105116 / 0.05 | foreground_miou, person_box_mask_iou, person_precision, person_recall, vehicle_iou, vehicle_precision, vehicle_recall, vehicle_xy_mae_m |

## Preregistered interpretation

AE128 UINT8+zstd deployment is accepted if and only if both hold: (1) q=0 passes all 12 same-q preservation gates against the frozen noAE UINT8+zstd validation result and retains at least the baseline 7/9 absolute service gates; and (2) at least one of q in {0.30, 0.50, 0.70} passes all 12 same-q preservation gates without reducing the absolute service-gate count below the frozen noAE UINT8+zstd count at that same q. q=0.90 and q=0.98 are stress/emergency profiles regardless of their results and cannot make or break acceptance. Every q is reported independently, and no setting is tuned or removed after observing a result.

- q=0 condition: **met** (12/12 same-q gates, 7/9 absolute service gates against a 7/9 baseline)
- qualifying primary q: none
- **decision: AE128_UINT8_ZSTD_DEPLOYMENT_NOT_ACCEPTED**
- q=0.90 and q=0.98 are stress/emergency profiles regardless of their results and did not enter the decision

## Integrity

- validation frames per q: 3,345
- q settings completed exactly once: 6/6
- every frame carried the AE128 family id, a 128-channel latent and the bound routing tag in its own header
- every frame was decompressed exactly once, and the decoder was discovered from the received header bytes alone
- retained UINT8 cells were exactly the selected cells; dropped cells scattered to exact zero before reconstruction
- q=0 invoked the ranker zero times and AE128 every time; no q produced an identity reconstruction
- frozen perception, stable ranker and selected AE128 parameters and buffers were unchanged
- per q the setting JSON was fsynced into place first, its predictions were removed only afterwards, and the cleanup marker was written last, so an interruption could only lose scratch predictions
- only compact evidence is retained; no prediction directory survives
