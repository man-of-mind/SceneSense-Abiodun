# Range-aware person semantic-support gate — train-holdout feasibility

One bounded feasibility study. No training, no cache rebuild, no model forward pass,
no CUDA, no validation or test access, and no deployed artifact was modified.

## Rule under test

- Candidates with predicted radial distance < 30 m keep the deployed
  `semantic_support >= 0.1` gate, unchanged.
- Candidates at or beyond 30 m use a relaxed threshold: A = 0.075,
  B = 0.050, C = 0.025. No other threshold, range boundary, score threshold, grouping
  parameter, or p025 value was varied.
- `score >= 0.20` before consolidation, the frozen grouping configuration, and the final
  p025 threshold are unchanged. The support gate is applied per candidate and the frozen
  `consolidate_person_candidates` then runs on the admitted subset under the
  preregistered `(None, 0.20)` configuration, so no selection logic is re-implemented.
- The boundary uses predicted radial distance, a runtime-computable candidate property.
  Ground-truth distance is used only to bin recall, exactly as in the completed audit.

## Validity

- Every frozen input verified against its registered SHA-256, plus the raw holdout
  metadata hashes against the frozen p025 `INPUT_HASHES.json`.
- Per frame, the baseline path is **bitwise equal** to the frozen p025 selection, and
  every relaxed policy admits a superset of the baseline candidates. Retention is
  deliberately *not* asserted to be a superset: under the frozen grouping rule a newly
  admitted higher-scoring candidate can merge with a baseline retention and win the
  group, replacing it. Those displacements are counted below rather than rejected.
- Each scored baseline reproduces the frozen per-episode p025 record exactly (observable
  GT, TP, FP, FN, precision, recall, XY MAE).
- Per-band counts recombine to the aggregate the frozen scorer computed, including XY MAE.
- Recall bins by `gt_distance_m`; precision bins by predicted radial distance. Matching
  order preserved: observable GT, then AVO-ignore, then structural-ignore, greedy inside
  3.0 m at AVO >= 0.65. The `gte_40m` row holds predictions beyond 40 m;
  it carries no eligible GT but does contribute to aggregate precision.

## Development on episode 03

Episode: `canonical_v3_03_train_30_30_s503_tm1503`, 1486 frames.

### baseline — unchanged 0.10 everywhere (>= 30 m support 0.1)

Retained p025 person predictions: 1041.

| scope | eligible GT | TP(gt) | TP(pred) | FP | FN | precision | recall | F1 | XY MAE m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 00_10m | 45 | 45 | 45 | 4 | 0 | 0.918367 | 1.000000 | 0.957447 | 0.170192 |
| 10_20m | 237 | 237 | 240 | 41 | 0 | 0.854093 | 1.000000 | 0.921305 | 0.329223 |
| 20_30m | 219 | 206 | 206 | 17 | 13 | 0.923767 | 0.940639 | 0.932127 | 0.619151 |
| 30_35m | 97 | 83 | 83 | 14 | 14 | 0.855670 | 0.855670 | 0.855670 | 0.731890 |
| 35_40m | 100 | 72 | 66 | 13 | 28 | 0.835443 | 0.720000 | 0.773438 | 0.884131 |
| gte_40m | 0 | 0 | 3 | 6 | 0 | 0.333333 | 0.000000 | 0.000000 | n/a |
| cumulative <=30 m | 501 | 488 | 491 | 62 | 13 | 0.887884 | 0.974052 | 0.928974 | 0.436946 |
| cumulative 30-40 m | 197 | 155 | 149 | 27 | 42 | 0.846591 | 0.786802 | 0.815602 | 0.802608 |
| aggregate AVO | 698 | 643 | 643 | 95 | 55 | 0.871274 | 0.921203 | 0.895543 | 0.525092 |

### A — >= 30 m support 0.075 (>= 30 m support 0.075)

Retained p025 person predictions: 1049.

| scope | eligible GT | TP(gt) | TP(pred) | FP | FN | precision | recall | F1 | XY MAE m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 00_10m | 45 | 45 | 45 | 4 | 0 | 0.918367 | 1.000000 | 0.957447 | 0.170192 |
| 10_20m | 237 | 237 | 240 | 41 | 0 | 0.854093 | 1.000000 | 0.921305 | 0.329223 |
| 20_30m | 219 | 206 | 206 | 17 | 13 | 0.923767 | 0.940639 | 0.932127 | 0.619151 |
| 30_35m | 97 | 83 | 83 | 16 | 14 | 0.838384 | 0.855670 | 0.846939 | 0.731890 |
| 35_40m | 100 | 73 | 67 | 16 | 27 | 0.807229 | 0.730000 | 0.766675 | 0.886513 |
| gte_40m | 0 | 0 | 3 | 7 | 0 | 0.300000 | 0.000000 | 0.000000 | n/a |
| cumulative <=30 m | 501 | 488 | 491 | 62 | 13 | 0.887884 | 0.974052 | 0.928974 | 0.436946 |
| cumulative 30-40 m | 197 | 156 | 150 | 32 | 41 | 0.824176 | 0.791878 | 0.807704 | 0.804246 |
| aggregate AVO | 698 | 644 | 644 | 101 | 54 | 0.864430 | 0.922636 | 0.892585 | 0.525919 |

Relative to the unchanged baseline:

| scope | recovered GT | additional FP | precision delta | recall delta |
|---|---:|---:|---:|---:|
| cumulative_30_40m | +1 | +5 | -0.022415 | +0.005076 |
| cumulative_le_30m | +0 | +0 | +0.000000 | +0.000000 |
| aggregate_avo | +1 | +6 | -0.006844 | +0.001433 |

Retained prediction delta: +8.

### B — >= 30 m support 0.050 (>= 30 m support 0.05)

Retained p025 person predictions: 1067.

| scope | eligible GT | TP(gt) | TP(pred) | FP | FN | precision | recall | F1 | XY MAE m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 00_10m | 45 | 45 | 45 | 4 | 0 | 0.918367 | 1.000000 | 0.957447 | 0.170192 |
| 10_20m | 237 | 237 | 240 | 41 | 0 | 0.854093 | 1.000000 | 0.921305 | 0.329223 |
| 20_30m | 219 | 206 | 205 | 18 | 13 | 0.919283 | 0.940639 | 0.929838 | 0.618389 |
| 30_35m | 97 | 83 | 84 | 23 | 14 | 0.785047 | 0.855670 | 0.818838 | 0.731890 |
| 35_40m | 100 | 76 | 70 | 20 | 24 | 0.777778 | 0.760000 | 0.768786 | 0.895862 |
| gte_40m | 0 | 0 | 3 | 8 | 0 | 0.272727 | 0.000000 | 0.000000 | n/a |
| cumulative <=30 m | 501 | 488 | 490 | 63 | 13 | 0.886076 | 0.974052 | 0.927983 | 0.436624 |
| cumulative 30-40 m | 197 | 159 | 154 | 43 | 38 | 0.781726 | 0.807107 | 0.794214 | 0.810266 |
| aggregate AVO | 698 | 647 | 647 | 114 | 51 | 0.850197 | 0.926934 | 0.886909 | 0.528446 |

Relative to the unchanged baseline:

| scope | recovered GT | additional FP | precision delta | recall delta |
|---|---:|---:|---:|---:|
| cumulative_30_40m | +4 | +16 | -0.064865 | +0.020305 |
| cumulative_le_30m | +0 | +1 | -0.001808 | +0.000000 |
| aggregate_avo | +4 | +19 | -0.021077 | +0.005731 |

Retained prediction delta: +26.

### C — >= 30 m support 0.025 (>= 30 m support 0.025)

Retained p025 person predictions: 1083.

| scope | eligible GT | TP(gt) | TP(pred) | FP | FN | precision | recall | F1 | XY MAE m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 00_10m | 45 | 45 | 45 | 4 | 0 | 0.918367 | 1.000000 | 0.957447 | 0.170192 |
| 10_20m | 237 | 237 | 240 | 41 | 0 | 0.854093 | 1.000000 | 0.921305 | 0.329223 |
| 20_30m | 219 | 206 | 205 | 18 | 13 | 0.919283 | 0.940639 | 0.929838 | 0.618389 |
| 30_35m | 97 | 83 | 84 | 30 | 14 | 0.736842 | 0.855670 | 0.791823 | 0.731890 |
| 35_40m | 100 | 76 | 70 | 25 | 24 | 0.736842 | 0.760000 | 0.748242 | 0.895181 |
| gte_40m | 0 | 0 | 3 | 9 | 0 | 0.250000 | 0.000000 | 0.000000 | n/a |
| cumulative <=30 m | 501 | 488 | 490 | 63 | 13 | 0.886076 | 0.974052 | 0.927983 | 0.436624 |
| cumulative 30-40 m | 197 | 159 | 154 | 55 | 38 | 0.736842 | 0.807107 | 0.770375 | 0.809941 |
| aggregate AVO | 698 | 647 | 647 | 127 | 51 | 0.835917 | 0.926934 | 0.879076 | 0.528366 |

Relative to the unchanged baseline:

| scope | recovered GT | additional FP | precision delta | recall delta |
|---|---:|---:|---:|---:|
| cumulative_30_40m | +4 | +28 | -0.109749 | +0.020305 |
| cumulative_le_30m | +0 | +1 | -0.001808 | +0.000000 |
| aggregate_avo | +4 | +32 | -0.035356 | +0.005731 |

Retained prediction delta: +42.

### Feasibility frontier on episode 03

Gates: 30-40 m precision >= 0.70, 30-40 m recall >= 0.70, aggregate AVO precision
>= 0.70, aggregate AVO recall >= 0.70, and cumulative <=30 m precision and recall each
degrading by at most 0.01 absolute from the unchanged baseline.

| policy | >=30 m support | 30-40 m P | 30-40 m R | 30-40 m F1 | agg P | agg R | <=30 m P loss | <=30 m R loss | passed | failed conditions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| A | 0.075 | 0.824176 | 0.791878 | 0.807704 | 0.864430 | 0.922636 | +0.000000 | +0.000000 | yes | — |
| B | 0.05 | 0.781726 | 0.807107 | 0.794214 | 0.850197 | 0.926934 | +0.001808 | +0.000000 | yes | — |
| C | 0.025 | 0.736842 | 0.807107 | 0.770375 | 0.835917 | 0.926934 | +0.001808 | +0.000000 | yes | — |

### Grouping displacement on episode 03

| policy | frames affected | newly retained | baseline retentions displaced |
|---|---:|---:|---:|
| A | 9 | 9 | 1 |
| B | 27 | 27 | 1 |
| C | 42 | 43 | 1 |

## Confirmation on episode 04

Frozen policy `A` was chosen on episode 03 alone and scored
exactly once on `canonical_v3_04_train_50_50_s504_tm1504` (1798 frames). The choice
was not revised afterwards.

### baseline — unchanged 0.10 everywhere (>= 30 m support 0.1)

Retained p025 person predictions: 2176.

| scope | eligible GT | TP(gt) | TP(pred) | FP | FN | precision | recall | F1 | XY MAE m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 00_10m | 103 | 101 | 98 | 3 | 2 | 0.970297 | 0.980583 | 0.975413 | 0.225390 |
| 10_20m | 534 | 526 | 526 | 38 | 8 | 0.932624 | 0.985019 | 0.958106 | 0.328595 |
| 20_30m | 755 | 684 | 649 | 53 | 71 | 0.924501 | 0.905960 | 0.915137 | 0.633086 |
| 30_35m | 246 | 186 | 227 | 34 | 60 | 0.869732 | 0.756098 | 0.808944 | 0.757502 |
| 35_40m | 220 | 109 | 100 | 24 | 111 | 0.806452 | 0.495455 | 0.613808 | 0.874456 |
| gte_40m | 0 | 0 | 6 | 6 | 0 | 0.500000 | 0.000000 | 0.000000 | n/a |
| cumulative <=30 m | 1392 | 1311 | 1273 | 94 | 81 | 0.931236 | 0.941810 | 0.936493 | 0.479509 |
| cumulative 30-40 m | 466 | 295 | 327 | 58 | 171 | 0.849351 | 0.633047 | 0.725418 | 0.800715 |
| aggregate AVO | 1858 | 1606 | 1606 | 158 | 252 | 0.910431 | 0.864370 | 0.886803 | 0.538510 |

### A — >= 30 m support 0.075 (>= 30 m support 0.075)

Retained p025 person predictions: 2203.

| scope | eligible GT | TP(gt) | TP(pred) | FP | FN | precision | recall | F1 | XY MAE m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 00_10m | 103 | 101 | 98 | 3 | 2 | 0.970297 | 0.980583 | 0.975413 | 0.225390 |
| 10_20m | 534 | 526 | 526 | 38 | 8 | 0.932624 | 0.985019 | 0.958106 | 0.328595 |
| 20_30m | 755 | 684 | 649 | 53 | 71 | 0.924501 | 0.905960 | 0.915137 | 0.633086 |
| 30_35m | 246 | 189 | 230 | 41 | 57 | 0.848708 | 0.768293 | 0.806501 | 0.751904 |
| 35_40m | 220 | 112 | 103 | 32 | 108 | 0.762963 | 0.509091 | 0.610693 | 0.869012 |
| gte_40m | 0 | 0 | 6 | 7 | 0 | 0.461538 | 0.000000 | 0.000000 | n/a |
| cumulative <=30 m | 1392 | 1311 | 1273 | 94 | 81 | 0.931236 | 0.941810 | 0.936493 | 0.479509 |
| cumulative 30-40 m | 466 | 301 | 333 | 73 | 165 | 0.820197 | 0.645923 | 0.722702 | 0.795479 |
| aggregate AVO | 1858 | 1612 | 1612 | 174 | 246 | 0.902576 | 0.867600 | 0.884742 | 0.538508 |

Relative to the unchanged baseline:

| scope | recovered GT | additional FP | precision delta | recall delta |
|---|---:|---:|---:|---:|
| cumulative_30_40m | +6 | +15 | -0.029154 | +0.012876 |
| cumulative_le_30m | +0 | +0 | +0.000000 | +0.000000 |
| aggregate_avo | +6 | +16 | -0.007855 | +0.003229 |

Retained prediction delta: +27.

### Whether the episode-03 conditions also hold on episode 04

| condition | holds |
|---|---|
| aggregate_precision_gte_0_70 | yes |
| aggregate_recall_gte_0_70 | yes |
| long_range_precision_gte_0_70 | yes |
| long_range_recall_gte_0_70 | no |
| near_precision_degradation_lte_0_01 | yes |
| near_recall_degradation_lte_0_01 | yes |

**The frozen policy does not satisfy every condition on episode 04.** The choice stands as made on episode 03; this disagreement is the finding, not a reason to retune.

## Reading

The protocol's terminal is decided on episode 03, and by that rule the answer is
`RANGE_AWARE_PERSON_SUPPORT_HOLDOUT_FEASIBLE`. Three caveats belong with it.

**The 30-40 m recall gate did not discriminate on episode 03.** The unchanged baseline
already scores 0.786802 there, above the 0.70 requirement, before any policy is
applied. On episode 03 the gates therefore mostly test whether relaxation damages
precision, not whether it delivers recall.

**The recovery is small in absolute terms.** Against 42 baseline long-range misses on episode 03, policy A recovers 1 and adds 5 false positives; even C, the most permissive threshold tested, recovers only 4.

**Episode 04 does not corroborate.** Its unchanged baseline sits at 0.633047 long-range recall, already below the 0.70 gate, and the frozen policy reaches only 0.645923. The gate is not reachable on that episode by this family of policies at all, so the episode-03 pass reflects episode difficulty more than policy strength.

**Threshold relaxation is the wrong lever for the audited headroom.** The completed
distance audit put the 30-40 m candidate recall ceiling at 0.6878 for the deployed p025
candidate set and 0.9638 when only `score >= 0.20` is applied with no support gate.
Dropping the long-range support threshold all the way to 0.025 moves measured recall
very little, so nearly all of that headroom sits in candidates whose semantic support is
below 0.025 — the person segmentation yields no usable component at range rather than a
merely weak one. Recovering it needs support-independent evidence, not a lower threshold.

## Scope limits

- Feasibility only, on two train-holdout episodes. No validation or test claim, and
  nothing here authorizes a runtime, threshold, checkpoint, forward-lock, segmentation,
  AE, or Phase-10B change.
- Only the three registered alternatives were evaluated. The range boundary, score
  threshold, grouping parameters, and p025 value were not searched.

RANGE_AWARE_PERSON_SUPPORT_HOLDOUT_FEASIBLE
