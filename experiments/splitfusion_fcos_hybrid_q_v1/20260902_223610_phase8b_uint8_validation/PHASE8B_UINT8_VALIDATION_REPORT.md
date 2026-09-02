# Phase 8B — frozen noAE UINT8 + zstd validation

Generated 2026-09-02T23:18:38.211690+00:00 · terminal `HYBRID_Q_UINT8_VALIDATION_COMPLETE`

This is one frozen-validation measurement. No training, tuning, threshold
change, test access, CARLA access, Raspberry Pi claim, or OAI latency claim was made.

## Locked pipeline

```text
FP32 C2 -> q selection -> per-channel UINT8 framing -> mandatory zstd-1
          -> zstd decode -> dequantize/zero-scatter -> FP32 C2 -> frozen tail
```

## Payload and host codec cost

| q | analytical pre-zstd B | measured pre-zstd B | zstd median B | p95 | min | max | vs framed FP32 q0 | vs compressed UINT8 q0 | range ms | quant/frame ms | zstd comp ms | zstd decomp ms | dequant/scatter ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 5,507,116 | 5,507,116 | 3,568,326 | 3,618,122 | 3,484,576 | 3,682,378 | 0.162048 | 1.000000 | 0.312 | 16.434 | 6.820 | 2.977 | 49.391 |
| 0.30 | 3,858,348 | 3,858,348 | 2,513,732 | 2,552,431 | 2,451,084 | 2,586,529 | 0.114156 | 0.704457 | 0.313 | 8.053 | 4.907 | 2.068 | 42.702 |
| 0.50 | 2,757,292 | 2,757,292 | 1,787,658 | 1,812,519 | 1,741,128 | 1,838,245 | 0.081183 | 0.500979 | 0.311 | 5.961 | 3.478 | 1.483 | 38.480 |
| 0.70 | 1,656,236 | 1,656,236 | 1,063,190 | 1,078,546 | 1,036,023 | 1,090,566 | 0.048283 | 0.297952 | 0.307 | 3.666 | 2.219 | 0.900 | 31.974 |
| 0.90 | 555,180 | 555,180 | 349,312 | 355,346 | 339,039 | 364,301 | 0.015863 | 0.097892 | 0.306 | 1.606 | 0.848 | 0.316 | 29.110 |
| 0.98 | 114,860 | 114,860 | 71,129 | 72,249 | 69,048 | 73,175 | 0.003230 | 0.019933 | 0.302 | 0.688 | 0.197 | 0.086 | 28.219 |

Ratios use median compressed bytes. Codec latency excludes backbone, ranker/selection, C2 upload, frozen tail, postprocessing, and scoring; it is current-host evidence only.

## Accuracy and independent decisions

| q | vehicle P/R/F1/XY | canonical-p025 person P/R/F1/XY | AVO>=0.65 person P/R/F1/XY | person 20–40 m recall | vehicle IoU | person box-mask IoU | foreground mIoU | service gates | quantization gates vs same-q FP32 | emergency status | finite |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 0.00 | 0.931393/0.868538/0.898868/0.478550 | 0.796616/0.595816/0.681738/0.839959 | 0.703602/0.712895/0.708218/0.813232 | 0.577650 | 0.898997 | 0.527819 | 0.713408 | 7/9 | 12/12 | primary anchor | yes |
| 0.30 | 0.934246/0.866474/0.899085/0.485661 | 0.741228/0.611054/0.669875/0.840235 | 0.639354/0.729579/0.681494/0.808642 | 0.600573 | 0.874908 | 0.501236 | 0.688072 | 7/9 | 12/12 | primary anchor | yes |
| 0.50 | 0.934600/0.862656/0.897188/0.498985 | 0.733560/0.613636/0.668260/0.837415 | 0.628811/0.731317/0.676201/0.805067 | 0.603438 | 0.821445 | 0.490288 | 0.655867 | 4/9 | 12/12 | primary anchor | yes |
| 0.70 | 0.917599/0.854917/0.885150/0.533809 | 0.741445/0.604339/0.665908/0.863608 | 0.630316/0.721237/0.672718/0.819462 | 0.595989 | 0.703595 | 0.459665 | 0.581630 | 4/9 | 12/12 | primary anchor | yes |
| 0.90 | 0.887099/0.761325/0.819414/0.668562 | 0.661206/0.521178/0.582900/0.869219 | 0.562403/0.628085/0.593432/0.835011 | 0.486533 | 0.403398 | 0.366553 | 0.384975 | 3/9 | 12/12 | emergency-mode anchor | yes |
| 0.98 | 0.825800/0.399649/0.538627/0.943004 | 0.630181/0.306302/0.412235/0.974020 | 0.547294/0.376086/0.445818/0.955582 | 0.226934 | 0.079219 | 0.178541 | 0.128880 | 3/9 | 12/12 | emergency-mode anchor | yes |

Quantization preservation, absolute service-gate attainment, and emergency-anchor designation are separate. No executable q was removed for missing service gates.

## Absolute protected-metric deltas from same-q FP32

Values below are `abs(UINT8 - FP32)` at the same q.

| q | vehicle_precision | vehicle_recall | vehicle_f1 | person_avo_precision | person_avo_recall | person_avo_f1 | vehicle_xy_mae_m | person_avo_xy_mae_m | vehicle_iou | person_box_mask_iou | foreground_miou | person_avo_recall_20_40m |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.000198603 | 0.000103189 | 0.000037202 | 0.000584627 | 0.000347584 | 0.000467660 | 0.000125293 | 0.001050246 | 0.000015642 | 0.000074803 | 0.000045222 | 0.000000000 |
| 0.30 | 0.000200639 | 0.000103189 | 0.000037334 | 0.001084909 | 0.000347584 | 0.000767702 | 0.000354189 | 0.001274693 | 0.000004721 | 0.000040422 | 0.000017851 | 0.001146132 |
| 0.50 | 0.000082512 | 0.000309566 | 0.000129359 | 0.001695923 | 0.000000000 | 0.000979371 | 0.000923629 | 0.000336000 | 0.000021818 | 0.000035669 | 0.000006925 | 0.000000000 |
| 0.70 | 0.000323172 | 0.000206377 | 0.000260973 | 0.000654572 | 0.000347584 | 0.000221219 | 0.000459636 | 0.000185260 | 0.000000160 | 0.000018317 | 0.000009239 | 0.000573066 |
| 0.90 | 0.000306555 | 0.000103189 | 0.000070965 | 0.000116750 | 0.001042753 | 0.000400955 | 0.000310242 | 0.001011516 | 0.000009492 | 0.000036675 | 0.000013591 | 0.001146132 |
| 0.98 | 0.001484799 | 0.000206377 | 0.000503079 | 0.000325168 | 0.000347584 | 0.000136579 | 0.001518866 | 0.000547419 | 0.000047673 | 0.000045854 | 0.000046763 | 0.000000000 |

## Integrity

- validation frames per q: 3,345
- q settings completed exactly once: 6/6
- every UINT8 sparse payload was zstd-wrapped and restored byte-for-byte
- all ranges, reconstructed C2 tensors, model outputs, and reported metrics were finite
- q=0 bypassed the ranker but remained UINT8-quantized
- model and stable-ranker state remained unchanged
- predictions were removed after scoring; only compact evidence is retained
