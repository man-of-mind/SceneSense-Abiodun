# Actor-volume observability frozen-model comparison

Run: `20260901_repaired_tolerance_cpu_once`  
Immutable AVO table SHA-256: `abb976f388ad33e8806d080750e9e7fbe1b1eb60e7e18ea55bedc60dce011386`

## Scope and qualification

This is the single registered CPU-only, read-only retrospective comparison. It used the original unnormalized actor-volume implementation from commit `dc5238d`; it did not train, run inference, load a checkpoint, import torch, use CUDA, invoke CARLA/Epic, open test data, tune a threshold, select a model, or change a service verdict.

The frozen evaluator universe contains 3,345 frames from the two raw validation episodes. The immutable table contains 5,276 person actor-frames satisfying distance ≤40 m, projected-box center inside the 1280×720 image, projected area ≥12 px, positive camera-forward geometry, and exact synchronized depth. 20,780 other raw person actor-frames are structural ignores. Truncation remains a separate diagnostic.

Before any prediction was opened, all 3200 registered pilot comparisons passed: identities, counts, flags, bands, and other discrete fields were exact; projected/visible-box coordinates and areas, unnormalized AVO, and truncation satisfied `math.isclose(rel_tol=1e-12, abs_tol=1e-12)`. The table has 736 no-support records.

`actor_volume_observability = area(B_visible) / area(B_full_clipped)`. It uses unchanged depth back-projection, oriented actor-volume containment, 0.05 m containment tolerance, bottom +0.03 m ground rejection, deterministic overlap assignment, and a no-support score of 0. It is not an exact visible-silhouette percentage.

## Complete model × AVO-threshold comparison

Detection score is fixed at 0.20. `R@.02` is diagnostic. `Ign pred` includes matches to AVO-below-cutoff and structural-ignore person GT. The six AVO thresholds are supplementary sensitivity views, not literal percentages of the pedestrian silhouette.

| Model | AVO≥ | Obs GT | AVO-ignored GT | Obs no-support GT | TP | FP | FN | Ign pred | Precision | Recall | F1 | XY MAE m | R@.02 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SplitFusion-FCOS | 0.10 | 4228 | 1048 | 0 | 2332 | 1217 | 1896 | 28 | 0.6571 | 0.5516 | 0.5997 | 0.845 | 0.5516 |
| SplitFusion-FCOS | 0.25 | 3995 | 1281 | 0 | 2325 | 1217 | 1670 | 35 | 0.6564 | 0.5820 | 0.6170 | 0.844 | 0.5820 |
| SplitFusion-FCOS | 0.50 | 3354 | 1922 | 0 | 2235 | 1216 | 1119 | 126 | 0.6476 | 0.6664 | 0.6569 | 0.830 | 0.6664 |
| SplitFusion-FCOS | 0.65 **(human-supported)** | 2877 | 2399 | 0 | 2062 | 1215 | 815 | 300 | 0.6292 | 0.7167 | 0.6701 | 0.813 | 0.7167 |
| SplitFusion-FCOS | 0.70 | 2606 | 2670 | 0 | 1894 | 1216 | 712 | 467 | 0.6090 | 0.7268 | 0.6627 | 0.803 | 0.7268 |
| SplitFusion-FCOS | 0.85 | 1080 | 4196 | 0 | 777 | 1215 | 303 | 1585 | 0.3901 | 0.7194 | 0.5059 | 0.778 | 0.7194 |
| Joint LR-ASPP | 0.10 | 4228 | 1048 | 0 | 2120 | 3409 | 2108 | 60 | 0.3834 | 0.5014 | 0.4346 | 1.203 | 0.7133 |
| Joint LR-ASPP | 0.25 | 3995 | 1281 | 0 | 2105 | 3409 | 1890 | 75 | 0.3818 | 0.5269 | 0.4427 | 1.200 | 0.7264 |
| Joint LR-ASPP | 0.50 | 3354 | 1922 | 0 | 1998 | 3409 | 1356 | 182 | 0.3695 | 0.5957 | 0.4561 | 1.189 | 0.7606 |
| Joint LR-ASPP | 0.65 **(human-supported)** | 2877 | 2399 | 0 | 1817 | 3408 | 1060 | 364 | 0.3478 | 0.6316 | 0.4485 | 1.168 | 0.7817 |
| Joint LR-ASPP | 0.70 | 2606 | 2670 | 0 | 1684 | 3407 | 922 | 498 | 0.3308 | 0.6462 | 0.4376 | 1.160 | 0.7913 |
| Joint LR-ASPP | 0.85 | 1080 | 4196 | 0 | 736 | 3410 | 344 | 1443 | 0.1775 | 0.6815 | 0.2817 | 1.170 | 0.8185 |
| Two-stage LR-ASPP | 0.10 | 4228 | 1048 | 0 | 2224 | 5457 | 2004 | 73 | 0.2895 | 0.5260 | 0.3735 | 1.315 | 0.7826 |
| Two-stage LR-ASPP | 0.25 | 3995 | 1281 | 0 | 2200 | 5457 | 1795 | 97 | 0.2873 | 0.5507 | 0.3776 | 1.311 | 0.7870 |
| Two-stage LR-ASPP | 0.50 | 3354 | 1922 | 0 | 2056 | 5457 | 1298 | 241 | 0.2737 | 0.6130 | 0.3784 | 1.292 | 0.8053 |
| Two-stage LR-ASPP | 0.65 **(human-supported)** | 2877 | 2399 | 0 | 1883 | 5456 | 994 | 415 | 0.2566 | 0.6545 | 0.3686 | 1.289 | 0.8203 |
| Two-stage LR-ASPP | 0.70 | 2606 | 2670 | 0 | 1733 | 5456 | 873 | 565 | 0.2411 | 0.6650 | 0.3539 | 1.290 | 0.8231 |
| Two-stage LR-ASPP | 0.85 | 1080 | 4196 | 0 | 744 | 5458 | 336 | 1552 | 0.1200 | 0.6889 | 0.2043 | 1.246 | 0.8241 |

## Highlighted AVO≥0.65 view and frozen references

AVO≥0.65 is highlighted because it was independently compared with the 100-person human pilot and achieved 0.8523 balanced accuracy. It is the only human-supported binary AVO operating point here. The existing human-band recall is a separate target-stratified reference and is not part of the AVO calculation.

| Model | View | GT/N | TP | FP | FN | Precision | Recall | F1 | XY MAE m | R@.02 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SplitFusion-FCOS | canonical v0.10 | 3872 | 2325 | 857 | 1547 | 0.7307 | 0.6005 | 0.6592 | 0.844 | 0.6005 |
| SplitFusion-FCOS | **AVO≥0.65** | 2877 | 2062 | 1215 | 815 | 0.6292 | 0.7167 | 0.6701 | 0.813 | 0.7167 |
| SplitFusion-FCOS | human bands ≥65, non-severe | 44 | 31 | NA | 13 | NA | 0.7045 | NA | 0.632 | NA |
| Joint LR-ASPP | canonical v0.10 | 3872 | 2091 | 2257 | 1781 | 0.4809 | 0.5400 | 0.5088 | 1.200 | 0.7342 |
| Joint LR-ASPP | **AVO≥0.65** | 2877 | 1817 | 3408 | 1060 | 0.3478 | 0.6316 | 0.4485 | 1.168 | 0.7817 |
| Joint LR-ASPP | human bands ≥65, non-severe | 44 | 23 | NA | 21 | NA | 0.5227 | NA | 0.709 | NA |
| Two-stage LR-ASPP | canonical v0.10 | 3872 | 2181 | 3971 | 1691 | 0.3545 | 0.5633 | 0.4352 | 1.307 | 0.8022 |
| Two-stage LR-ASPP | **AVO≥0.65** | 2877 | 1883 | 5456 | 994 | 0.2566 | 0.6545 | 0.3686 | 1.289 | 0.8203 |
| Two-stage LR-ASPP | human bands ≥65, non-severe | 44 | 25 | NA | 19 | NA | 0.5682 | NA | 1.295 | NA |

## Per-model precision-recall trends

These are denominator changes under successively stricter supplementary eligibility views, not model improvements and not a threshold-selection exercise.

- SplitFusion-FCOS: AVO≥0.10: n=4228, P=0.6571, R=0.5516; AVO≥0.25: n=3995, P=0.6564, R=0.5820; AVO≥0.50: n=3354, P=0.6476, R=0.6664; AVO≥0.65: n=2877, P=0.6292, R=0.7167; AVO≥0.70: n=2606, P=0.6090, R=0.7268; AVO≥0.85: n=1080, P=0.3901, R=0.7194.
- Joint LR-ASPP: AVO≥0.10: n=4228, P=0.3834, R=0.5014; AVO≥0.25: n=3995, P=0.3818, R=0.5269; AVO≥0.50: n=3354, P=0.3695, R=0.5957; AVO≥0.65: n=2877, P=0.3478, R=0.6316; AVO≥0.70: n=2606, P=0.3308, R=0.6462; AVO≥0.85: n=1080, P=0.1775, R=0.6815.
- Two-stage LR-ASPP: AVO≥0.10: n=4228, P=0.2895, R=0.5260; AVO≥0.25: n=3995, P=0.2873, R=0.5507; AVO≥0.50: n=3354, P=0.2737, R=0.6130; AVO≥0.65: n=2877, P=0.2566, R=0.6545; AVO≥0.70: n=2606, P=0.2411, R=0.6650; AVO≥0.85: n=1080, P=0.1200, R=0.6889.

## Direct comparison with each frozen canonical v0.10 result

Canonical values below are reused artifacts, not rescored values. Differences reflect eligibility denominators and ignore assignment; they are not model changes.

| Model | AVO≥ | Canonical GT | Canonical P/R/F1 | AVO GT | AVO P/R/F1 | Canonical/AVO XY MAE m |
|---|---:|---:|---:|---:|---:|---:|
| SplitFusion-FCOS | 0.10 | 3872 | 0.7307/0.6005/0.6592 | 4228 | 0.6571/0.5516/0.5997 | 0.844/0.845 |
| SplitFusion-FCOS | 0.25 | 3872 | 0.7307/0.6005/0.6592 | 3995 | 0.6564/0.5820/0.6170 | 0.844/0.844 |
| SplitFusion-FCOS | 0.50 | 3872 | 0.7307/0.6005/0.6592 | 3354 | 0.6476/0.6664/0.6569 | 0.844/0.830 |
| SplitFusion-FCOS | 0.65 | 3872 | 0.7307/0.6005/0.6592 | 2877 | 0.6292/0.7167/0.6701 | 0.844/0.813 |
| SplitFusion-FCOS | 0.70 | 3872 | 0.7307/0.6005/0.6592 | 2606 | 0.6090/0.7268/0.6627 | 0.844/0.803 |
| SplitFusion-FCOS | 0.85 | 3872 | 0.7307/0.6005/0.6592 | 1080 | 0.3901/0.7194/0.5059 | 0.844/0.778 |
| Joint LR-ASPP | 0.10 | 3872 | 0.4809/0.5400/0.5088 | 4228 | 0.3834/0.5014/0.4346 | 1.200/1.203 |
| Joint LR-ASPP | 0.25 | 3872 | 0.4809/0.5400/0.5088 | 3995 | 0.3818/0.5269/0.4427 | 1.200/1.200 |
| Joint LR-ASPP | 0.50 | 3872 | 0.4809/0.5400/0.5088 | 3354 | 0.3695/0.5957/0.4561 | 1.200/1.189 |
| Joint LR-ASPP | 0.65 | 3872 | 0.4809/0.5400/0.5088 | 2877 | 0.3478/0.6316/0.4485 | 1.200/1.168 |
| Joint LR-ASPP | 0.70 | 3872 | 0.4809/0.5400/0.5088 | 2606 | 0.3308/0.6462/0.4376 | 1.200/1.160 |
| Joint LR-ASPP | 0.85 | 3872 | 0.4809/0.5400/0.5088 | 1080 | 0.1775/0.6815/0.2817 | 1.200/1.170 |
| Two-stage LR-ASPP | 0.10 | 3872 | 0.3545/0.5633/0.4352 | 4228 | 0.2895/0.5260/0.3735 | 1.307/1.315 |
| Two-stage LR-ASPP | 0.25 | 3872 | 0.3545/0.5633/0.4352 | 3995 | 0.2873/0.5507/0.3776 | 1.307/1.311 |
| Two-stage LR-ASPP | 0.50 | 3872 | 0.3545/0.5633/0.4352 | 3354 | 0.2737/0.6130/0.3784 | 1.307/1.292 |
| Two-stage LR-ASPP | 0.65 | 3872 | 0.3545/0.5633/0.4352 | 2877 | 0.2566/0.6545/0.3686 | 1.307/1.289 |
| Two-stage LR-ASPP | 0.70 | 3872 | 0.3545/0.5633/0.4352 | 2606 | 0.2411/0.6650/0.3539 | 1.307/1.290 |
| Two-stage LR-ASPP | 0.85 | 3872 | 0.3545/0.5633/0.4352 | 1080 | 0.1200/0.6889/0.2043 | 1.307/1.246 |

## Reused canonical vehicle and segmentation evidence

These frozen canonical v0.10 values were copied from each existing evaluation artifact; vehicle and segmentation were not rescored.

| Model | Vehicle GT | Vehicle TP/FP/FN | Vehicle P/R/F1 | Vehicle XY MAE m | Vehicle IoU | Person box-mask IoU | Foreground mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| SplitFusion-FCOS | 9691 | 8416/618/1275 | 0.9316/0.8684/0.8989 | 0.479 | 0.8990 | 0.5279 | 0.7135 |
| Joint LR-ASPP | 9691 | 7425/4268/2266 | 0.6350/0.7662/0.6944 | 0.767 | 0.8186 | 0.3783 | 0.5984 |
| Two-stage LR-ASPP | 9691 | 7580/17357/2111 | 0.3040/0.7822/0.4378 | 0.969 | 0.9084 | 0.5733 | 0.7408 |

## Results by validation episode

| Model | AVO≥ | Episode | Obs GT | AVO-ignored GT | TP | FP | FN | Ign pred | Precision | Recall | F1 | XY MAE m | R@.02 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SplitFusion-FCOS | 0.10 | canonical_v3_05_val_30_30_s601_tm1601 | 1185 | 205 | 633 | 260 | 552 | 7 | 0.7088 | 0.5342 | 0.6092 | 0.824 | 0.5342 |
| SplitFusion-FCOS | 0.10 | canonical_v3_06_val_50_50_s602_tm1602 | 3043 | 843 | 1699 | 957 | 1344 | 21 | 0.6397 | 0.5583 | 0.5962 | 0.853 | 0.5583 |
| SplitFusion-FCOS | 0.25 | canonical_v3_05_val_30_30_s601_tm1601 | 1131 | 259 | 631 | 260 | 500 | 9 | 0.7082 | 0.5579 | 0.6241 | 0.823 | 0.5579 |
| SplitFusion-FCOS | 0.25 | canonical_v3_06_val_50_50_s602_tm1602 | 2864 | 1022 | 1694 | 957 | 1170 | 26 | 0.6390 | 0.5915 | 0.6143 | 0.851 | 0.5915 |
| SplitFusion-FCOS | 0.50 | canonical_v3_05_val_30_30_s601_tm1601 | 945 | 445 | 614 | 260 | 331 | 26 | 0.7025 | 0.6497 | 0.6751 | 0.813 | 0.6497 |
| SplitFusion-FCOS | 0.50 | canonical_v3_06_val_50_50_s602_tm1602 | 2409 | 1477 | 1621 | 956 | 788 | 100 | 0.6290 | 0.6729 | 0.6502 | 0.837 | 0.6729 |
| SplitFusion-FCOS | 0.65 | canonical_v3_05_val_30_30_s601_tm1601 | 853 | 537 | 591 | 259 | 262 | 50 | 0.6953 | 0.6928 | 0.6941 | 0.807 | 0.6928 |
| SplitFusion-FCOS | 0.65 | canonical_v3_06_val_50_50_s602_tm1602 | 2024 | 1862 | 1471 | 956 | 553 | 250 | 0.6061 | 0.7268 | 0.6610 | 0.816 | 0.7268 |
| SplitFusion-FCOS | 0.70 | canonical_v3_05_val_30_30_s601_tm1601 | 787 | 603 | 546 | 260 | 241 | 94 | 0.6774 | 0.6938 | 0.6855 | 0.798 | 0.6938 |
| SplitFusion-FCOS | 0.70 | canonical_v3_06_val_50_50_s602_tm1602 | 1819 | 2067 | 1348 | 956 | 471 | 373 | 0.5851 | 0.7411 | 0.6539 | 0.805 | 0.7411 |
| SplitFusion-FCOS | 0.85 | canonical_v3_05_val_30_30_s601_tm1601 | 344 | 1046 | 219 | 260 | 125 | 421 | 0.4572 | 0.6366 | 0.5322 | 0.729 | 0.6366 |
| SplitFusion-FCOS | 0.85 | canonical_v3_06_val_50_50_s602_tm1602 | 736 | 3150 | 558 | 955 | 178 | 1164 | 0.3688 | 0.7582 | 0.4962 | 0.797 | 0.7582 |
| Joint LR-ASPP | 0.10 | canonical_v3_05_val_30_30_s601_tm1601 | 1185 | 205 | 600 | 1326 | 585 | 16 | 0.3115 | 0.5063 | 0.3857 | 1.138 | 0.7629 |
| Joint LR-ASPP | 0.10 | canonical_v3_06_val_50_50_s602_tm1602 | 3043 | 843 | 1520 | 2083 | 1523 | 44 | 0.4219 | 0.4995 | 0.4574 | 1.229 | 0.6941 |
| Joint LR-ASPP | 0.25 | canonical_v3_05_val_30_30_s601_tm1601 | 1131 | 259 | 598 | 1326 | 533 | 18 | 0.3108 | 0.5287 | 0.3915 | 1.139 | 0.7745 |
| Joint LR-ASPP | 0.25 | canonical_v3_06_val_50_50_s602_tm1602 | 2864 | 1022 | 1507 | 2083 | 1357 | 57 | 0.4198 | 0.5262 | 0.4670 | 1.224 | 0.7074 |
| Joint LR-ASPP | 0.50 | canonical_v3_05_val_30_30_s601_tm1601 | 945 | 445 | 570 | 1326 | 375 | 46 | 0.3006 | 0.6032 | 0.4013 | 1.125 | 0.8053 |
| Joint LR-ASPP | 0.50 | canonical_v3_06_val_50_50_s602_tm1602 | 2409 | 1477 | 1428 | 2083 | 981 | 136 | 0.4067 | 0.5928 | 0.4824 | 1.214 | 0.7430 |
| Joint LR-ASPP | 0.65 | canonical_v3_05_val_30_30_s601_tm1601 | 853 | 537 | 531 | 1326 | 322 | 85 | 0.2859 | 0.6225 | 0.3919 | 1.117 | 0.8077 |
| Joint LR-ASPP | 0.65 | canonical_v3_06_val_50_50_s602_tm1602 | 2024 | 1862 | 1286 | 2082 | 738 | 279 | 0.3818 | 0.6354 | 0.4770 | 1.189 | 0.7708 |
| Joint LR-ASPP | 0.70 | canonical_v3_05_val_30_30_s601_tm1601 | 787 | 603 | 498 | 1326 | 289 | 118 | 0.2730 | 0.6328 | 0.3815 | 1.109 | 0.8158 |
| Joint LR-ASPP | 0.70 | canonical_v3_06_val_50_50_s602_tm1602 | 1819 | 2067 | 1186 | 2081 | 633 | 380 | 0.3630 | 0.6520 | 0.4664 | 1.181 | 0.7806 |
| Joint LR-ASPP | 0.85 | canonical_v3_05_val_30_30_s601_tm1601 | 344 | 1046 | 225 | 1326 | 119 | 391 | 0.1451 | 0.6541 | 0.2375 | 1.054 | 0.8547 |
| Joint LR-ASPP | 0.85 | canonical_v3_06_val_50_50_s602_tm1602 | 736 | 3150 | 511 | 2084 | 225 | 1052 | 0.1969 | 0.6943 | 0.3068 | 1.221 | 0.8016 |
| Two-stage LR-ASPP | 0.10 | canonical_v3_05_val_30_30_s601_tm1601 | 1185 | 205 | 646 | 1886 | 539 | 14 | 0.2551 | 0.5451 | 0.3476 | 1.329 | 0.8228 |
| Two-stage LR-ASPP | 0.10 | canonical_v3_06_val_50_50_s602_tm1602 | 3043 | 843 | 1578 | 3571 | 1465 | 59 | 0.3065 | 0.5186 | 0.3853 | 1.309 | 0.7670 |
| Two-stage LR-ASPP | 0.25 | canonical_v3_05_val_30_30_s601_tm1601 | 1131 | 259 | 639 | 1886 | 492 | 21 | 0.2531 | 0.5650 | 0.3496 | 1.326 | 0.8249 |
| Two-stage LR-ASPP | 0.25 | canonical_v3_06_val_50_50_s602_tm1602 | 2864 | 1022 | 1561 | 3571 | 1303 | 76 | 0.3042 | 0.5450 | 0.3904 | 1.305 | 0.7720 |
| Two-stage LR-ASPP | 0.50 | canonical_v3_05_val_30_30_s601_tm1601 | 945 | 445 | 613 | 1886 | 332 | 47 | 0.2453 | 0.6487 | 0.3560 | 1.300 | 0.8402 |
| Two-stage LR-ASPP | 0.50 | canonical_v3_06_val_50_50_s602_tm1602 | 2409 | 1477 | 1443 | 3571 | 966 | 194 | 0.2878 | 0.5990 | 0.3888 | 1.289 | 0.7916 |
| Two-stage LR-ASPP | 0.65 | canonical_v3_05_val_30_30_s601_tm1601 | 853 | 537 | 582 | 1885 | 271 | 79 | 0.2359 | 0.6823 | 0.3506 | 1.295 | 0.8453 |
| Two-stage LR-ASPP | 0.65 | canonical_v3_06_val_50_50_s602_tm1602 | 2024 | 1862 | 1301 | 3571 | 723 | 336 | 0.2670 | 0.6428 | 0.3773 | 1.286 | 0.8098 |
| Two-stage LR-ASPP | 0.70 | canonical_v3_05_val_30_30_s601_tm1601 | 787 | 603 | 545 | 1885 | 242 | 116 | 0.2243 | 0.6925 | 0.3388 | 1.289 | 0.8475 |
| Two-stage LR-ASPP | 0.70 | canonical_v3_06_val_50_50_s602_tm1602 | 1819 | 2067 | 1188 | 3571 | 631 | 449 | 0.2496 | 0.6531 | 0.3612 | 1.290 | 0.8125 |
| Two-stage LR-ASPP | 0.85 | canonical_v3_05_val_30_30_s601_tm1601 | 344 | 1046 | 251 | 1886 | 93 | 409 | 0.1175 | 0.7297 | 0.2023 | 1.239 | 0.8634 |
| Two-stage LR-ASPP | 0.85 | canonical_v3_06_val_50_50_s602_tm1602 | 736 | 3150 | 493 | 3572 | 243 | 1143 | 0.1213 | 0.6698 | 0.2054 | 1.250 | 0.8057 |

## Distance-bin recall and localization

| Model | AVO≥ | Distance | Obs GT | TP/FN | Recall | XY MAE m | R@.02 |
|---|---:|---|---:|---:|---:|---:|---:|
| SplitFusion-FCOS | 0.10 | 00_10m | 156 | 140/16 | 0.8974 | 0.404 | 0.8974 |
| SplitFusion-FCOS | 0.10 | 10_20m | 1241 | 1018/223 | 0.8203 | 0.630 | 0.8203 |
| SplitFusion-FCOS | 0.10 | 20_30m | 1605 | 847/758 | 0.5277 | 1.037 | 0.5277 |
| SplitFusion-FCOS | 0.10 | 30_40m | 1226 | 327/899 | 0.2667 | 1.206 | 0.2667 |
| SplitFusion-FCOS | 0.25 | 00_10m | 152 | 138/14 | 0.9079 | 0.370 | 0.9079 |
| SplitFusion-FCOS | 0.25 | 10_20m | 1197 | 1015/182 | 0.8480 | 0.630 | 0.8480 |
| SplitFusion-FCOS | 0.25 | 20_30m | 1487 | 846/641 | 0.5689 | 1.038 | 0.5689 |
| SplitFusion-FCOS | 0.25 | 30_40m | 1159 | 326/833 | 0.2813 | 1.207 | 0.2813 |
| SplitFusion-FCOS | 0.50 | 00_10m | 139 | 129/10 | 0.9281 | 0.346 | 0.9281 |
| SplitFusion-FCOS | 0.50 | 10_20m | 1104 | 983/121 | 0.8904 | 0.617 | 0.8904 |
| SplitFusion-FCOS | 0.50 | 20_30m | 1191 | 806/385 | 0.6767 | 1.017 | 0.6767 |
| SplitFusion-FCOS | 0.50 | 30_40m | 920 | 317/603 | 0.3446 | 1.211 | 0.3446 |
| SplitFusion-FCOS | 0.65 | 00_10m | 124 | 115/9 | 0.9274 | 0.352 | 0.9274 |
| SplitFusion-FCOS | 0.65 | 10_20m | 1008 | 932/76 | 0.9246 | 0.597 | 0.9246 |
| SplitFusion-FCOS | 0.65 | 20_30m | 1004 | 735/269 | 0.7321 | 1.014 | 0.7321 |
| SplitFusion-FCOS | 0.65 | 30_40m | 741 | 280/461 | 0.3779 | 1.194 | 0.3779 |
| SplitFusion-FCOS | 0.70 | 00_10m | 111 | 104/7 | 0.9369 | 0.340 | 0.9369 |
| SplitFusion-FCOS | 0.70 | 10_20m | 941 | 874/67 | 0.9288 | 0.583 | 0.9288 |
| SplitFusion-FCOS | 0.70 | 20_30m | 886 | 659/227 | 0.7438 | 1.018 | 0.7438 |
| SplitFusion-FCOS | 0.70 | 30_40m | 668 | 257/411 | 0.3847 | 1.185 | 0.3847 |
| SplitFusion-FCOS | 0.85 | 00_10m | 43 | 39/4 | 0.9070 | 0.362 | 0.9070 |
| SplitFusion-FCOS | 0.85 | 10_20m | 413 | 398/15 | 0.9637 | 0.586 | 0.9637 |
| SplitFusion-FCOS | 0.85 | 20_30m | 299 | 229/70 | 0.7659 | 0.967 | 0.7659 |
| SplitFusion-FCOS | 0.85 | 30_40m | 325 | 111/214 | 0.3415 | 1.222 | 0.3415 |
| Joint LR-ASPP | 0.10 | 00_10m | 156 | 127/29 | 0.8141 | 0.703 | 0.9936 |
| Joint LR-ASPP | 0.10 | 10_20m | 1241 | 903/338 | 0.7276 | 1.083 | 0.8687 |
| Joint LR-ASPP | 0.10 | 20_30m | 1605 | 718/887 | 0.4474 | 1.322 | 0.6467 |
| Joint LR-ASPP | 0.10 | 30_40m | 1226 | 372/854 | 0.3034 | 1.438 | 0.6077 |
| Joint LR-ASPP | 0.25 | 00_10m | 152 | 126/26 | 0.8289 | 0.688 | 0.9934 |
| Joint LR-ASPP | 0.25 | 10_20m | 1197 | 900/297 | 0.7519 | 1.083 | 0.8805 |
| Joint LR-ASPP | 0.25 | 20_30m | 1487 | 712/775 | 0.4788 | 1.317 | 0.6617 |
| Joint LR-ASPP | 0.25 | 30_40m | 1159 | 367/792 | 0.3167 | 1.433 | 0.6152 |
| Joint LR-ASPP | 0.50 | 00_10m | 139 | 117/22 | 0.8417 | 0.624 | 1.0000 |
| Joint LR-ASPP | 0.50 | 10_20m | 1104 | 874/230 | 0.7917 | 1.071 | 0.8958 |
| Joint LR-ASPP | 0.50 | 20_30m | 1191 | 660/531 | 0.5542 | 1.316 | 0.6919 |
| Joint LR-ASPP | 0.50 | 30_40m | 920 | 347/573 | 0.3772 | 1.435 | 0.6511 |
| Joint LR-ASPP | 0.65 | 00_10m | 124 | 105/19 | 0.8468 | 0.571 | 1.0000 |
| Joint LR-ASPP | 0.65 | 10_20m | 1008 | 829/179 | 0.8224 | 1.054 | 0.9167 |
| Joint LR-ASPP | 0.65 | 20_30m | 1004 | 584/420 | 0.5817 | 1.308 | 0.7022 |
| Joint LR-ASPP | 0.65 | 30_40m | 741 | 299/442 | 0.4035 | 1.421 | 0.6694 |
| Joint LR-ASPP | 0.70 | 00_10m | 111 | 95/16 | 0.8559 | 0.558 | 1.0000 |
| Joint LR-ASPP | 0.70 | 10_20m | 941 | 782/159 | 0.8310 | 1.050 | 0.9245 |
| Joint LR-ASPP | 0.70 | 20_30m | 886 | 530/356 | 0.5982 | 1.300 | 0.7099 |
| Joint LR-ASPP | 0.70 | 30_40m | 668 | 277/391 | 0.4147 | 1.407 | 0.6766 |
| Joint LR-ASPP | 0.85 | 00_10m | 43 | 38/5 | 0.8837 | 0.427 | 1.0000 |
| Joint LR-ASPP | 0.85 | 10_20m | 413 | 358/55 | 0.8668 | 1.044 | 0.9346 |
| Joint LR-ASPP | 0.85 | 20_30m | 299 | 193/106 | 0.6455 | 1.378 | 0.7258 |
| Joint LR-ASPP | 0.85 | 30_40m | 325 | 147/178 | 0.4523 | 1.395 | 0.7323 |
| Two-stage LR-ASPP | 0.10 | 00_10m | 156 | 123/33 | 0.7885 | 1.034 | 0.9615 |
| Two-stage LR-ASPP | 0.10 | 10_20m | 1241 | 863/378 | 0.6954 | 1.161 | 0.8711 |
| Two-stage LR-ASPP | 0.10 | 20_30m | 1605 | 833/772 | 0.5190 | 1.390 | 0.7807 |
| Two-stage LR-ASPP | 0.10 | 30_40m | 1226 | 405/821 | 0.3303 | 1.575 | 0.6729 |
| Two-stage LR-ASPP | 0.25 | 00_10m | 152 | 122/30 | 0.8026 | 1.021 | 0.9605 |
| Two-stage LR-ASPP | 0.25 | 10_20m | 1197 | 862/335 | 0.7201 | 1.160 | 0.8789 |
| Two-stage LR-ASPP | 0.25 | 20_30m | 1487 | 819/668 | 0.5508 | 1.383 | 0.7801 |
| Two-stage LR-ASPP | 0.25 | 30_40m | 1159 | 397/762 | 0.3425 | 1.582 | 0.6782 |
| Two-stage LR-ASPP | 0.50 | 00_10m | 139 | 115/24 | 0.8273 | 0.973 | 0.9712 |
| Two-stage LR-ASPP | 0.50 | 10_20m | 1104 | 841/263 | 0.7618 | 1.149 | 0.8904 |
| Two-stage LR-ASPP | 0.50 | 20_30m | 1191 | 734/457 | 0.6163 | 1.367 | 0.7901 |
| Two-stage LR-ASPP | 0.50 | 30_40m | 920 | 366/554 | 0.3978 | 1.570 | 0.6978 |
| Two-stage LR-ASPP | 0.65 | 00_10m | 124 | 105/19 | 0.8468 | 0.961 | 0.9677 |
| Two-stage LR-ASPP | 0.65 | 10_20m | 1008 | 805/203 | 0.7986 | 1.148 | 0.9107 |
| Two-stage LR-ASPP | 0.65 | 20_30m | 1004 | 660/344 | 0.6574 | 1.374 | 0.7998 |
| Two-stage LR-ASPP | 0.65 | 30_40m | 741 | 313/428 | 0.4224 | 1.585 | 0.7004 |
| Two-stage LR-ASPP | 0.70 | 00_10m | 111 | 95/16 | 0.8559 | 0.959 | 0.9640 |
| Two-stage LR-ASPP | 0.70 | 10_20m | 941 | 761/180 | 0.8087 | 1.149 | 0.9129 |
| Two-stage LR-ASPP | 0.70 | 20_30m | 886 | 591/295 | 0.6670 | 1.386 | 0.8014 |
| Two-stage LR-ASPP | 0.70 | 30_40m | 668 | 286/382 | 0.4281 | 1.576 | 0.7021 |
| Two-stage LR-ASPP | 0.85 | 00_10m | 43 | 37/6 | 0.8605 | 1.007 | 0.9535 |
| Two-stage LR-ASPP | 0.85 | 10_20m | 413 | 349/64 | 0.8450 | 1.153 | 0.9128 |
| Two-stage LR-ASPP | 0.85 | 20_30m | 299 | 202/97 | 0.6756 | 1.240 | 0.7793 |
| Two-stage LR-ASPP | 0.85 | 30_40m | 325 | 156/169 | 0.4800 | 1.521 | 0.7354 |

## Interpretation

AVO≥0.65 achieved 0.8523 balanced accuracy against the human pilot. This is a binary observability sensitivity analysis. The actor-volume score failed fine-grained four-band agreement (exact agreement 0.4416; linear-weighted kappa 0.4581, below 0.60), so human bands remain the fine-grained visibility reference. AVO is a bounding-box extent statistic derived from actor-volume-supported depth points; it is not an exact visible-silhouette percentage.

The comparison is retrospective and supplementary. It does not alter checkpoint selection, the supervisor-approved SplitFusion-FCOS service decision, canonical v0.10 results, vehicle results, segmentation results, service gates, or model selection. Historical depth-consistent occupancy is retained only as an internal sensitivity. Increasing precision or recall under a stricter AVO eligibility view must not be interpreted as model improvement.

The six AVO thresholds are supplementary sensitivity views. AVO≥0.65 is highlighted because it was independently compared with the human pilot; none of the thresholds changes the canonical or supervisor-approved service result.

ACTOR_VOLUME_OBSERVABILITY_MODEL_COMPARISON_COMPLETE
