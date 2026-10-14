| Arm | rc | Verdict | Completed / dec. | ≤170 | Later | None | P50/P99 ms | Payload P50/P99 kB | Feedback ≤170 |
|---|---:|---|---:|---:|---:|---:|---|---|---:|
| Agent (deployed actor) | 0 | VALID_MEASUREMENT | 493/493 | 479 | 14 | 0 | 134.0 / 204.0 | 117.3 / 165.5 | 390/493 |
| Mode 8 · AE64/UINT4 · q=0 | 1 | VALID_MEASUREMENT | 500/500 | 210 | 290 | 0 | 173.1 / 430.9 | 469.1 / 511.5 | 30/500 |
| Mode 11 · AE32/UINT4 · q=0 | 1 | VALID_MEASUREMENT | 500/500 | 453 | 47 | 0 | 143.1 / 216.7 | 234.2 / 250.8 | 304/500 |
| Mode 3 · AE128/UINT8 · q=0 | 1 | VALID_MEASUREMENT | 4/500 | 0 | 4 | 496 | 429.3 / 471.6 | 2339.1 / 2379.3 | 0/500 |
| Mode 2 · noAE/UINT4 · q=0 | 1 | VALID_MEASUREMENT | 153/500 | 0 | 153 | 347 | 292.9 / 470.7 | 1437.7 / 1600.2 | 0/500 |
| Mode 1 · noAE/UINT6 · q=0 | 1 | VALID_MEASUREMENT | 1/500 | 0 | 1 | 499 | 482.8 / 482.8 | 2726.0 / 2812.2 | 0/500 |
| Mode 10 · AE32/UINT6 · q=0 | 1 | VALID_MEASUREMENT | 500/500 | 292 | 208 | 0 | 166.6 / 339.6 | 469.5 / 483.2 | 47/500 |
| Mode 7 · AE64/UINT6 · q=0 | 1 | VALID_MEASUREMENT | 413/500 | 2 | 411 | 87 | 218.7 / 451.3 | 959.5 / 975.7 | 0/500 |
| Mode 0 · noAE/UINT8 · q=0 | 1 | VALID_MEASUREMENT · ZERO_COMPLETION | 0/500 | 0 | 0 | 500 | — (no completion) | 3558.9 / 3629.0 | 0/500 |
| Mode 6 · AE64/UINT8 · q=0 | 1 | VALID_MEASUREMENT | 316/500 | 0 | 316 | 184 | 237.8 / 474.2 | 1215.3 / 1233.0 | 0/500 |
| Mode 4 · AE128/UINT6 · q=0 | 1 | VALID_MEASUREMENT | 27/500 | 0 | 27 | 473 | 314.5 / 405.3 | 1865.0 / 1895.4 | 0/500 |
| Mode 9 · AE32/UINT8 · q=0 | 1 | VALID_MEASUREMENT | 476/500 | 123 | 353 | 24 | 179.2 / 397.4 | 600.2 / 609.9 | 8/500 |
| Mode 5 · AE128/UINT4 · q=0 | 1 | VALID_MEASUREMENT | 424/500 | 0 | 424 | 76 | 228.2 / 450.0 | 876.3 / 953.8 | 0/500 |

≥ = denominator counts only decisions with GT objects; decisions without GT (NO_GT_PRODUCER_CAPACITY) are listed in gt_coverage and never counted as zero objects.

| Arm | Class | Recall all | Recall ≤170 (bound) | Precision all / timely | Seg IoU all → ≤170 (evaluable) |
|---|---|---|---|---|---|
| Agent (deployed actor) | vehicle | 587/906 = 0.648 | 573/906 = 0.632 (EXACT) | 0.898 / 0.898 | 0.769 → 0.733 (EXACT, 493/493) |
| Agent (deployed actor) | person | 399/702 = 0.568 | 381/702 = 0.543 (EXACT) | 0.611 / 0.608 | 0.487 → 0.474 (EXACT, 493/493) |
| Mode 8 · AE64/UINT4 · q=0 | vehicle | 715/942 = 0.759 | 301/942 = 0.320 (EXACT) | 0.930 / 0.929 | 0.893 → 0.365 (EXACT, 500/500) |
| Mode 8 · AE64/UINT4 · q=0 | person | 423/725 = 0.583 | 187/725 = 0.258 (EXACT) | 0.794 / 0.816 | 0.537 → 0.266 (EXACT, 500/500) |
| Mode 11 · AE32/UINT4 · q=0 | vehicle | 849/1179 = 0.720 | 796/1179 = 0.675 (EXACT) | 0.914 / 0.913 | 0.889 → 0.868 (EXACT, 500/500) |
| Mode 11 · AE32/UINT4 · q=0 | person | 318/692 = 0.460 | 278/692 = 0.402 (EXACT) | 0.763 / 0.757 | 0.449 → 0.402 (EXACT, 500/500) |
| Mode 3 · AE128/UINT8 · q=0 | vehicle | 11/14 = 0.786 | 0/≥392 = 0.000 (EXACT_ZERO_NO_TIMELY_PREDICTION) | 1.000 / n/a | 0.835 → 0 by construction (no timely prediction; intersection 0) |
| Mode 3 · AE128/UINT8 · q=0 | person | n/a (no eligible GT on the completed frames) | 0/≥246 = 0.000 (EXACT_ZERO_NO_TIMELY_PREDICTION) | n/a / n/a | n/a → 0 wherever GT foreground exists (no timely prediction); union UNMEASURED |
| Mode 2 · noAE/UINT4 · q=0 | vehicle | 207/291 = 0.711 | 0/≥470 = 0.000 (EXACT_ZERO_NO_TIMELY_PREDICTION) | 0.962 / n/a | 0.842 → 0 by construction (no timely prediction; intersection 0) |
| Mode 2 · noAE/UINT4 · q=0 | person | 107/179 = 0.598 | 0/≥413 = 0.000 (EXACT_ZERO_NO_TIMELY_PREDICTION) | 0.774 / n/a | 0.610 → 0 by construction (no timely prediction; intersection 0) |
| Mode 1 · noAE/UINT6 · q=0 | vehicle | 0/2 = 0.000 | 0/≥475 = 0.000 (EXACT_ZERO_NO_TIMELY_PREDICTION) | 1.000 / n/a | 0.517 → 0 by construction (no timely prediction; intersection 0) |
| Mode 1 · noAE/UINT6 · q=0 | person | n/a (no eligible GT on the completed frames) | 0/≥246 = 0.000 (EXACT_ZERO_NO_TIMELY_PREDICTION) | n/a / n/a | n/a → 0 wherever GT foreground exists (no timely prediction); union UNMEASURED |
| Mode 10 · AE32/UINT6 · q=0 | vehicle | 709/953 = 0.744 | 455/953 = 0.477 (EXACT) | 0.934 / 0.935 | 0.873 → 0.468 (EXACT, 500/500) |
| Mode 10 · AE32/UINT6 · q=0 | person | 377/635 = 0.594 | 219/635 = 0.345 (EXACT) | 0.697 / 0.706 | 0.513 → 0.347 (EXACT, 500/500) |
| Mode 7 · AE64/UINT6 · q=0 | vehicle | 763/983 = 0.776 | 4/≥1025 = 0.004 (UPPER_BOUND_UNKNOWN_GT_ON_FAILED_OPPORTUNITIES) | 0.947 / 1.000 | 0.908 → 0.003 (UPPER_BOUND_GT_PIXELS_UNKNOWN_ON_FAILED_OPPORTUNITIES, 413/500) |
| Mode 7 · AE64/UINT6 · q=0 | person | 290/505 = 0.574 | 0/≥555 = 0.000 (EXACT_ZERO_NO_TIMELY_PREDICTION) | 0.816 / 1.000 | 0.559 → 0.000 (UPPER_BOUND_GT_PIXELS_UNKNOWN_ON_FAILED_OPPORTUNITIES, 413/500) |
| Mode 0 · noAE/UINT8 · q=0 | vehicle | n/a (0 completed) | 0/≥422 = 0.000 (EXACT_ZERO_NO_TIMELY_PREDICTION) | n/a / n/a | n/a → 0 wherever GT foreground exists (no timely prediction); union UNMEASURED |
| Mode 0 · noAE/UINT8 · q=0 | person | n/a (0 completed) | 0/≥213 = 0.000 (EXACT_ZERO_NO_TIMELY_PREDICTION) | n/a / n/a | n/a → 0 wherever GT foreground exists (no timely prediction); union UNMEASURED |
| Mode 6 · AE64/UINT8 · q=0 | vehicle | 736/881 = 0.835 | 0/≥1018 = 0.000 (EXACT_ZERO_NO_TIMELY_PREDICTION) | 0.945 / n/a | 0.921 → 0 by construction (no timely prediction; intersection 0) |
| Mode 6 · AE64/UINT8 · q=0 | person | 149/287 = 0.519 | 0/≥402 = 0.000 (EXACT_ZERO_NO_TIMELY_PREDICTION) | 0.789 / n/a | 0.487 → 0 by construction (no timely prediction; intersection 0) |
| Mode 4 · AE128/UINT6 · q=0 | vehicle | 64/76 = 0.842 | 0/≥408 = 0.000 (EXACT_ZERO_NO_TIMELY_PREDICTION) | 0.837 / n/a | 0.925 → 0 by construction (no timely prediction; intersection 0) |
| Mode 4 · AE128/UINT6 · q=0 | person | 0/7 = 0.000 | 0/≥243 = 0.000 (EXACT_ZERO_NO_TIMELY_PREDICTION) | n/a / n/a | 0.000 → 0 by construction (no timely prediction; intersection 0) |
| Mode 9 · AE32/UINT8 · q=0 | vehicle | 639/937 = 0.682 | 174/≥955 = 0.182 (UPPER_BOUND_UNKNOWN_GT_ON_FAILED_OPPORTUNITIES) | 0.925 / 0.902 | 0.877 → 0.281 (UPPER_BOUND_GT_PIXELS_UNKNOWN_ON_FAILED_OPPORTUNITIES, 476/500) |
| Mode 9 · AE32/UINT8 · q=0 | person | 333/574 = 0.580 | 90/≥597 = 0.151 (UPPER_BOUND_UNKNOWN_GT_ON_FAILED_OPPORTUNITIES) | 0.778 / 0.759 | 0.505 → 0.147 (UPPER_BOUND_GT_PIXELS_UNKNOWN_ON_FAILED_OPPORTUNITIES, 476/500) |
| Mode 5 · AE128/UINT4 · q=0 | vehicle | 594/818 = 0.726 | 0/≥875 = 0.000 (EXACT_ZERO_NO_TIMELY_PREDICTION) | 0.948 / n/a | 0.896 → 0 by construction (no timely prediction; intersection 0) |
| Mode 5 · AE128/UINT4 · q=0 | person | 289/555 = 0.521 | 0/≥610 = 0.000 (EXACT_ZERO_NO_TIMELY_PREDICTION) | 0.807 / n/a | 0.534 → 0 by construction (no timely prediction; intersection 0) |
