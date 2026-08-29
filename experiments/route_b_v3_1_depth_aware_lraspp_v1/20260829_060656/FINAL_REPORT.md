# Route B v3.1 depth-aware LR-ASPP — implementation recovery and first valid evaluation

**Registered terminal: `VALID_DEPTH_AWARE_LRASPP_DOES_NOT_IMPROVE`**

One bounded implementation recovery and one clean-lineage 40-epoch scientific candidate were completed. All pre-scientific contract gates passed, but none of epochs 10, 20, 30, or 40 passed the frozen preservation gate. No checkpoint was selected or promoted, and v0.25 sensitivity was therefore not licensed.

## Accepted diagnosis and corrections

The accepted prior evidence was reproduced twice from identical official initialization, sampler order, sample IDs, augmentations, inputs, and targets through batches 1–15. The first non-finite operation was `vehicle log_dimensions -> FP32 exp` for `canonical_v3_02_train_50_50_s502_tm1502:actor:82` at cell `(53,87)`, with finite raw log-dimension near 95.5. Depth `expm1` and every preceding operation remained finite. Small replay trajectory drift was accepted as normal for CUDA cross-entropy and grid-sampler backward.

Only the predeclared corrections were applied:

1. Dimension supervision is now direct `SmoothL1(predicted_log_dimensions, log(target_dimensions))`, guarded by strict target positivity. This restores nonzero direct log-space supervision outside the defective expression's safe interval. It is Class-A, safe-domain equivalent with zero measured loss delta, and explicitly not claimed globally equivalent where the old lower clamp was active or FP32 `exp` overflowed.
2. Every final vehicle/person field-convolution weight starts at exact zero while all registered biases are preserved. The shared neck and object trunks retain nonzero Kaiming initialization. This repairs the ineffective priors that previously produced update-1 loss about 391,407, 99.976% heatmap pressure, and batch-14 raw log-dimensions near 95.5, -43, and 31.
3. Deployable dimension `exp`, actor-depth `expm1`, and dense diagnostic `expm1` run unbounded in float64. No clamp, upper physical bound, overflow class, or prediction filter was introduced; the stale >40 m world-distance filter was removed. Any scored non-finite field raises explicitly.
4. The unregistered eight-hour stop was replaced with the authorized 16-hour operational ceiling without changing the 40-epoch schedule.

The original registered design remains authoritative: 32 log1p anchors over 0–40 m, one unbounded residual per anchor, probability-weighted continuous log-depth decoding, last-bin extrapolation, lower depth guard only, CE plus Lovász, and unchanged targets, weights, optimizer, schedule, sampler, augmentation, batch, and evaluation gates.

## Unit, replay, and optimization qualification

- Direct log-dimension tests were finite forward/backward for `[-120,-72,-43,-13.8,0,31,80,95.5,120]`; eight non-optimal inputs had nonzero gradients and the exact optimum correctly had zero gradient.
- All 192,996 eligible train dimension components across 64,332 objects were strictly positive; minimum target was 0.375358 m.
- All field outputs equaled their biases spatially. Every applicable field head had a finite nonzero update-1 weight gradient. `vehicle.box_center_delta` was explicitly proven structurally at its exact zero target/prior optimum; its mathematically required zero gradient was not hidden or injected. Object trunks and the shared neck had finite nonzero object-task gradients by update 2.
- Float64 decode matched the old safe-domain decoder within FP32 tolerance. `exp(95.5)` decoded finitely as `2.9862284022825254e41`; a synthetic remaining non-finite scored detection raised explicitly.
- Both repaired batches 1–15 replays reached `REPAIRED_EXECUTION_FINITE`. Initial states and all 15 input/target batches were identical. Batch-14 totals were 34.396336 and 34.396122 (absolute delta 0.000214, within rtol 1e-3/atol 1e-5). Diagnostic model drift had maximum absolute 1.494e-4 and relative L2 3.399e-10; AdamW drift had maximum absolute 3.234e-4 and relative L2 1.475e-5. State drift was not a gate.
- The unchanged qualification suite passed at physical batch 16, accumulation 1. The pristine -4.6 heatmap prior intentionally yields no score ≥0.02; scored split parity stayed unchanged, while a separate threshold-zero probe verified record schema.

Initial and end-of-disposable-epoch weighted shares were:

| Component | Initial | End epoch 1 |
|---|---:|---:|
| segmentation | 10.5423% | 1.6383% |
| heatmap | 47.5688% | 61.5617% |
| subcell | 0.0343% | 0.0939% |
| box center delta | 0.0281% | 0.1094% |
| box width/height | 11.6319% | 10.0226% |
| physical ray | 0.9812% | 1.7991% |
| depth bin | 13.5838% | 21.7993% |
| depth residual | 0.1891% | 0.6022% |
| endpoint | 0.8268% | 0.3708% |
| dimensions | 0.1407% | 0.0356% |
| yaw | 0.2352% | 0.6821% |
| parked | 0.3622% | 0.7288% |
| radar support | 0.1811% | 0.2213% |
| dense depth | 8.9029% | 0.2327% |
| radar consistency | 4.7917% | 0.1023% |

The initial detection and actor-depth group shares were 61.1634% and 14.5997%; the maximum individual initial share was 47.5688%. Thus all 50%/5%/5% pressure gates passed without weight tuning. The end shares were required to remain finite, not to retain the initialization-only balance.

The disposable epoch visited all 16,827 train frames exactly once in 1,052 finite updates, exercised every required module, decoded 182 finite records, reserved 9,836 MiB, completed in 191.1 s, and was discarded.

## Durability and runtime

The measured uninstrumented means were 116.45 ms/batch for Stage A and 141.07 ms/batch for Stage B. The conservative projection was 3.614 h including 2 h for evaluation/diagnostics, leaving 77.4% margin inside 16 h. Actual training took 6,018.9 s (1.672 h). Measured training, four inference traversals, validation-cache construction, and evaluation totaled 7,309.4 s (2.030 h), excluding orchestration gaps.

Before update 1, atomic `epoch_000.pt` was written with pristine corrected model, empty optimizer, scheduler/update counters, RNG and sampler state, config hash, and code provenance. Its 16,940,621 bytes and SHA-256 `c8fda39a179c2c5047a8a45acad91cf0f746da3aeed013c392f8a888264a1756` were independently verified. All 41 epoch 000–040 checkpoint/sidecar pairs subsequently passed complete/byte/SHA verification. Every training epoch visited all 16,827 frames once; no validation ran during optimization.

## Frozen evaluation

| Epoch | Vehicle P/R/F1 | Vehicle XY MAE | Person P/R/F1 | Person R@.02 | Person XY MAE | Vehicle IoU | Person box IoU | fg mIoU | Preserve |
|---:|---|---:|---|---:|---:|---:|---:|---:|:---:|
| 10 | .6350/.7662/.6944 | .7671 | .4809/.5400/.5088 | .7342 | 1.2004 | .8186 | .3783 | .5984 | no |
| 20 | .5580/.7956/.6559 | .7060 | .6546/.4112/.5051 | .5958 | 1.1542 | .8349 | .4056 | .6202 | no |
| 30 | .5563/.7723/.6468 | .6752 | .6389/.3915/.4855 | .5323 | 1.0994 | .8411 | .4186 | .6298 | no |
| 40 | .6130/.7483/.6739 | .6632 | .6635/.3809/.4840 | .5093 | 1.0933 | .8417 | .4200 | .6308 | no |

All outputs and diagnostics were finite. Because the preservation gate failed at every epoch, material and service gates also failed by registered precedence. Selected checkpoint: **none**. Registered terminal: `VALID_DEPTH_AWARE_LRASPP_DOES_NOT_IMPROVE`.

## Attempts, provenance, and scope

- Scientific attempt 1 (`20260829_042423`) aborted due the dimension-loss implementation defect.
- Numerical recovery `20260829_051209` was diagnostic only and was not a scientific attempt.
- Scientific attempt 2 (`20260829_060656`) is the first valid 40-epoch evaluation reported here.
- Complete repair-code tree: `d5741eee70ff4a776f6417f2b5ae7be9da85ad4a` (foundation `3829518fbbdc34ca770159ce1e587c72aeefdf67`).
- Scientific launch commit: `b47cfdf3076b9b854c1c933f9a92d2a032f3ea83`.
- Final-report commit is recorded after this create-only report commit in `FINAL_REPORT_PROVENANCE.json`, avoiding a self-reference.
- The original failed and numerical-recovery experiment directories were preserved. The train depth cache was reused read-only after every cache hash was verified. No test, CARLA, OAI contents, q/AE, hybrid-q, live runtime, or 288-measurement access occurred; no branch was created and nothing was pushed.
- Worktree at launch and completion had only the pre-existing dirty `OAI/openairinterface5g` submodule. Its contents were not accessed or modified.
- Desktop notification returned 0 and was recorded as delivered. `COMPLETION_SENTINEL` contains the exact registered terminal.
