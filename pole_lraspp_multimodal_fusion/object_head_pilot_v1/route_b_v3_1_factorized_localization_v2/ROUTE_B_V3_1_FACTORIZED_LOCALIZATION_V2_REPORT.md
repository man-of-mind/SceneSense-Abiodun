# Route B v3.1 camera-plane contract and factorized localization v2 report

Terminal: `LRASPP_FACTORIZED_LOCALIZATION_RUNTIME_FAILURE`

Experiment: `experiments/route_b_v3_1_factorized_localization_v2/20260827_232621`

## Phase A — camera-plane localization contract

The deterministic derived rule moves a localization-positive object whose physical-centre camera-forward depth is `<= 0.0` to localization-ignore with reason `CAMERA_PLANE_STRADDLING_CENTER_NONPOSITIVE_DEPTH`. Its source provenance and segmentation pixels remain, its visible region is neutral under the existing ignore-assignment semantics, and it is never converted to background.

| Visibility | Split | Exclusions | Vehicle | Person | Actor | Static environment | Identities |
|---|---|---:|---:|---:|---:|---:|---:|
| v0.10 | train | 100 | 100 | 0 | 29 | 71 | 24 |
| v0.10 | validation | 34 | 34 | 0 | 26 | 8 | 11 |
| v0.25 | train | 10 | 10 | 0 | 3 | 7 | 5 |
| v0.25 | validation | 1 | 1 | 0 | 0 | 1 | 1 |

All nine contract gates passed: all remaining localization depths are positive; segmentation masks, counts, and hashes are unchanged; no ignore became background; train/validation IDs remain disjoint; test rows/payloads are absent; and the derived view copied zero raw corpus files.

## Amended retained epoch-15 baseline

No inference was run. The retained detections SHA-256 is `265e68dc0bc6e1b5a851cf7254be45918a23d20ed60dbb040f60c607fd3ae1ba`, and v0.10/v0.25 used independently keyed ignore caches.

| Contract | Class | Precision | Recall | F1 | Recall @ 0.02 | XY MAE m |
|---|---|---:|---:|---:|---:|---:|
| v0.10 | vehicle | 0.712543 | 0.807760 | 0.757170 | 0.845527 | 0.984324 |
| v0.10 | person | 0.495587 | 0.464101 | 0.479328 | 0.560692 | 1.396104 |
| v0.25 | vehicle | 0.721978 | 0.882648 | 0.794269 | 0.903757 | 0.943158 |
| v0.25 | person | 0.497530 | 0.507109 | 0.502274 | 0.592417 | 1.394697 |

v0.10 segmentation: vehicle IoU `0.865511`, person box-mask IoU `0.443745`, foreground mIoU `0.654628`, background IoU `0.994015`.

Amended v0.10 taxonomy:

- Vehicle FP at 0.20: 979 `PREDICTED_DUPLICATE`, 1,694 `TWO_D_CORRECT_WORLD_WRONG`, 485 `BACKGROUND_OR_OTHER`.
- Person FN at 0.02: 162 `MATCHING_CONTENTION`, 854 `CENTER_PRESENT_WORLD_WRONG`, 685 `HEATMAP_CENTER_MISS`.

## Factorized localization v2 implementation

The new tail-side path reads the frozen native stride-4 128-channel fused feature. It uses two `3x3 Conv(64) + GroupNorm + ReLU` blocks and emits only one `log_depth` channel and two projected-physical-centre offset channels. It decodes positive depth with `exp(log_depth)`, unprojects with per-frame intrinsics, converts through the verified camera-to-world transform, and replaces only local/world XYZ. The legacy XYZ channels remain in checkpoints for compatibility and are frozen.

The fixed class-macro loss is the sum of Smooth-L1 log-depth, projected-centre offset, and local XY endpoint losses, each weight 1.0 and endpoint beta 1.0 m. The configured optimizer is AdamW at `3e-4` with a fixed 12-epoch cosine schedule, batch 16, q=0, and no AE.

Source-deterministic parameter counts are 111,171 trainable new parameters and 4,931,198 frozen inherited parameters (5,042,369 total). Runtime parameter reporting was not reached because CUDA was unavailable.

The source boundary continues to transport only `{low, high}`; the localization path is implemented wholly in the tail and accepts only those features plus static/per-frame camera calibration metadata. No raw RGB, radar, or depth side channel was added. The mandatory runtime split-parity proof could not execute after the CUDA-dependent real-batch check failed, so bit-identical runtime parity is not claimed.

## Launch result and fail-closed stop

| Check | Result |
|---|:---:|
| 1. `py_compile` on 10 new files | pass |
| 2. Parse resolved configs | pass |
| 3. Verify epoch-15 SHA-256 | pass |
| 4. Verify all Phase-A gates | pass |
| 5. Synthetic positive-depth projection/unprojection | pass |
| 6. One real v3.1 q=0 AMP batch | **fail — CUDA unavailable** |
| 7. Runtime split check | not executable because check 6 could not instantiate the CUDA model |
| 8. Retained-prediction legacy parity | pass |

The supervisor emitted the required runtime-failure terminal and stopped. Training epochs completed: 0. Candidate inference passes: 0. Epoch 4/8/12 metrics, candidate taxonomy, radar-supported/unsupported candidate localization, selected v0.10/v0.25 results, and selected checkpoint are therefore unavailable. No checkpoint was created or selected.

## Service targets

| Target | Amended baseline | Candidate |
|---|:---:|:---:|
| Vehicle precision >= 0.80 | fail | not evaluated |
| Vehicle recall >= 0.85 | fail | not evaluated |
| Person precision >= 0.80 | fail | not evaluated |
| Person recall >= 0.80 | fail | not evaluated |
| Vehicle XY MAE <= 1.0 m | pass | not evaluated |
| Person XY MAE <= 1.2 m | fail | not evaluated |
| Vehicle IoU >= 0.85 | pass | not evaluated |
| Person box-mask IoU >= 0.50 | fail | not evaluated |
| Foreground mIoU >= 0.675 | fail | not evaluated |

Full service readiness is not claimed; the localization-only experiment could not run and frozen segmentation could not improve.

## Resources and safety

Measured wall time was approximately 99.6 seconds across contract construction, retained-prediction re-score, and the failed v2 pipeline launch. VRAM usage is unavailable because CUDA was unavailable.

Test stayed unopened. CARLA, OAI, containers, q/AE, and 288 measurements were not run. No q/AE phase, retry, second experiment, or remote operation was started. Canonical data, the epoch-15 warm start, retained predictions, prior experiments, and the existing dirty OAI submodule were untouched. Desktop notification was attempted once and was unavailable because the session could not connect to the notification service.
