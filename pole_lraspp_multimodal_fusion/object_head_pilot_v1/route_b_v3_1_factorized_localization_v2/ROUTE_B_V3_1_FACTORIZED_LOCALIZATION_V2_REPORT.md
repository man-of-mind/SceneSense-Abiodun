# Route B v3.1 factorized localization v2 numerical audit

Terminal: `LRASPP_FACTORIZED_LOCALIZATION_NUMERICAL_CAUSE_UNRESOLVED`

Reproduction evidence: `experiments/route_b_v3_1_factorized_localization_v2/numerical_reproduction_activation_20260828_064700`

## Decision

The epoch-2/batch-134 failure was reproduced twice from the original epoch-15 warm start with the committed seed, manifest order, implicit PyTorch DataLoader generator, eight workers, batch 16, AdamW configuration, LR, cosine schedule, and AMP settings.

The first non-finite operation is not unconstrained depth exponentiation, its endpoint gradient, or projected-offset scaling. It is the first localization-trunk `Conv2d` executed under FP16 autocast. This operation is outside the two authorized repair branches, so no repair qualification or candidate training was allowed.

## Exact first non-finite tensor and operation

The frozen native stride-4 input to `localization_trunk.0` was finite: range `[0, 15272]`, mean `265.878`. All trainable parameters were finite. Under explicit fp32 the first convolution was finite with range `[-65844.9609, 31632.5801]`. FP16's minimum finite value is `-65504`; under autocast the same convolution produced one `-inf` as its first non-finite value.

GroupNorm propagated that single infinity into 165,888 NaNs. Both output heads then became non-finite at the affected region: seven positive-cell raw depth logits were NaN and fourteen projected-offset values were NaN. The downstream `exp`, camera/world geometry, and all three losses propagated those NaNs but did not originate them.

The full explicit-fp32 failing-batch path was finite:

| Quantity | fp32 result |
|---|---:|
| Raw depth-logit range | 1.319609 to 3.581608 |
| Decoded depth range | 3.741957 to 35.931282 m |
| Target depth range | 3.648628 to 37.250889 m |
| Predicted projected-offset range | -2.089510 to 0.261687 grid units |
| Target projected-offset range | -18.004744 to 8.225213 grid units |
| Log-depth loss | 0.006895 |
| Projected-offset loss | 0.564491 |
| Local-XY endpoint loss | 1.130124 |
| Total loss | 1.701510 |

The AMP path instead produced NaN total loss. This proves an FP16 convolution range overflow before depth decoding, rather than a target, parameter, optimizer-state, `exp`, or offset-normalization failure.

## Failing sample IDs

1. `canonical_v3_04_train_50_50_s504_tm1504_001638_frame6778`
2. `canonical_v3_04_train_50_50_s504_tm1504_000943_frame3998`
3. `canonical_v3_03_train_30_30_s503_tm1503_000405_frame1834`
4. `canonical_v3_04_train_50_50_s504_tm1504_001315_frame5486`
5. `canonical_v3_01_train_30_30_s501_tm1501_000508_frame2246`
6. `canonical_v3_03_train_30_30_s503_tm1503_001352_frame5622`
7. `canonical_v3_03_train_30_30_s503_tm1503_000395_frame1794`
8. `canonical_v3_04_train_50_50_s504_tm1504_000567_frame2494`
9. `canonical_v3_02_train_50_50_s502_tm1502_001036_frame4362`
10. `canonical_v3_01_train_30_30_s501_tm1501_001486_frame6158`
11. `canonical_v3_02_train_50_50_s502_tm1502_000530_frame2338`
12. `canonical_v3_04_train_50_50_s504_tm1504_001500_frame6226`
13. `canonical_v3_04_train_50_50_s504_tm1504_000493_frame2198`
14. `canonical_v3_03_train_30_30_s503_tm1503_000823_frame3506`
15. `canonical_v3_01_train_30_30_s501_tm1501_000225_frame1114`
16. `canonical_v3_01_train_30_30_s501_tm1501_000285_frame1354`

## Preceding gradients and state

The GradScaler scale remained `1024` before and after batches 130–133 and was `1024` on entry to batch 134. All parameters remained finite before and after each preceding optimizer step.

| Batch | Localization trunk norm | Log-depth head norm | Offset head norm |
|---:|---:|---:|---:|
| 130 | 26.3322 | 20.0492 | 0.1494 |
| 131 | 54.1300 | 34.2228 | 0.3022 |
| 132 | 31.1866 | 21.0938 | 0.6380 |
| 133 | 38.7749 | 30.8105 | 0.5917 |

These gradients were finite. No backward or optimizer step was executed for batch 134 because the harness stopped at the first non-finite forward/loss value.

## Repair gate

No repair was applied:

- Bounded-depth parameterization: not authorized because `exp` was not the first non-finite operation.
- Bounded-depth interval and median-bias initialization: not applicable.
- Projected-offset normalization: not authorized because target and fp32 offset paths were finite.
- Gradient clipping: not applied because it was conditional on an authorized repair.
- Forcing the localization trunk convolution to fp32 would address the observed operation, but it was not an authorized repair and was therefore not implemented.

Consequently the bounded repair-qualification suite, fresh 12-epoch resume, checkpoints 4/8/12, and three candidate validation inference passes were not run.

## Baseline and unavailable candidate results

The amended retained baseline remains unchanged:

| Contract | Class | Precision | Recall | F1 | XY MAE m |
|---|---|---:|---:|---:|---:|
| v0.10 | vehicle | 0.712543 | 0.807760 | 0.757170 | 0.984324 |
| v0.10 | person | 0.495587 | 0.464101 | 0.479328 | 1.396104 |
| v0.25 | vehicle | 0.721978 | 0.882648 | 0.794269 | 0.943158 |
| v0.25 | person | 0.497530 | 0.507109 | 0.502274 | 1.394697 |

Baseline taxonomy remains vehicle duplicate `979`, vehicle `TWO_D_CORRECT_WORLD_WRONG=1694`, person `CENTER_PRESENT_WORLD_WRONG=854`, and person `HEATMAP_CENTER_MISS=685`.

There is no repaired candidate, epoch table, world-error taxonomy, radar-supported/unsupported candidate result, selected checkpoint/SHA, selected v0.10 result, or selected-checkpoint v0.25 sensitivity result.

## Service targets

| Target | Amended baseline | Candidate |
|---|:---:|:---:|
| Vehicle precision >= 0.80 | fail | not trained |
| Vehicle recall >= 0.85 | fail | not trained |
| Person precision >= 0.80 | fail | not trained |
| Person recall >= 0.80 | fail | not trained |
| Vehicle XY MAE <= 1.0 m | pass | not trained |
| Person XY MAE <= 1.2 m | fail | not trained |
| Vehicle IoU >= 0.85 | pass | not trained |
| Person box-mask IoU >= 0.50 | fail | not trained |
| Foreground mIoU >= 0.675 | fail | not trained |

## Runtime, resources, and safety

The activation-instrumented reproduction took `36.488 s`; the first reproduction took `37.619 s`. The reproduction harness did not record a new peak-memory counter; the immediately preceding unchanged CUDA run recorded `928.482 MiB` allocated and `1350.0 MiB` reserved.

The camera-plane contract and dataset were not rebuilt. Test, CARLA, OAI, q/AE, and 288 measurements remained untouched. No branch, remote, dependency, threshold, NMS, loss weight, optimizer, schedule, batch, selection gate, service target, or candidate metric was changed. No follow-up experiment or push occurred.
