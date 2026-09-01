# SplitFusion-FCOS forward perception lock: person p025

**Decision date:** 2026-09-01

**Status:** locked for hybrid-q, quantization, zstd, AE and system evaluation

## Forward baseline

The forward perception baseline is the frozen epoch-26
`SplitFusion-FCOS-R50-FPN-P2-P7` model plus the already qualified service
post-processing and a final person-only FP32 score floor of `0.25`.

The earlier `0.20` service candidate remains the historical, supervisor-
accepted 7/9 result. It is not deleted or reinterpreted. This lock promotes
the `0.25` wrapper for future transport experiments because it:

- runs the accepted `0.20` pipeline unchanged first;
- retains only an exact ordered subset of its person outputs;
- leaves every vehicle output, segmentation output, score and geometry field
  unchanged;
- improved canonical person precision from `0.730673` to `0.796686`;
- achieved person precision/recall `0.704187/0.713243` on the supporting
  `AVO >= 0.65` validation view.

The original nine-gate canonical result remains 7/9: person precision is
`0.003314` below `0.80` and person recall remains below `0.80`. Locking the
model is a project decision to stop perception tuning and begin the transport
study, not a claim that those gates passed.

The original `locked_config.json`, validation code and validation artifact
retain the pre-decision wording “awaiting final acceptance” and remain
hash-identical. This later lock supersedes only that status field; it does not
rewrite the implementation or evidence. Because validation threshold behavior
had previously been explored, the p025 validation result is confirmation, not
an untouched selection estimate. The reserved test set is still required for
independent publication confirmation.

## Frozen architecture and split

- Input: one tensor containing RGB `(3)` concatenated with radar raster `(4)`.
- Backbone/detector: ResNet-50 + FPN P2-P7 + FCOS.
- Split tensor: raw fused `C2`, shape `[256,112,192]`.
- Clean FP32 split payload: `22,020,096` bytes per frame.
- Detector confidence: original FCOS class/centerness score equation.
- Localization: custom class-specific depth, physical-centre ray, dimensions
  and yaw heads gathered at the retained FCOS candidate location.
- Checkpoint: epoch 26, SHA-256
  `da14d21edbd374c1c3abce02ca4674b9f4097becfba9759aba945cea160a297f`.

## Frozen post-processing

- Vehicle: accepted monotonic score calibration; no candidate filtering or
  rerun of NMS.
- Person stage 1: parameter-free semantic instance consolidation at input
  score `0.20`, grid index `27`, semantic support `0.10`, group-box IoU `0.20`.
- Person stage 2: retain consolidated rows satisfying
  `score_fp32 >= 0.25`.
- No candidate creation, score rewriting, geometry rewriting or reordering.

## Evidence used for the lock

| View | Person precision | Person recall | F1 | XY MAE |
|---|---:|---:|---:|---:|
| Train holdout, `AVO >= 0.65` | .898881 | .879890 | .889284 | .534674 m |
| Validation, `AVO >= 0.65` | .704187 | .713243 | .708686 | .812181 m |
| Validation, canonical v0.10 | .796686 | .596074 | .681932 | .839516 m |

At `AVO >= 0.65`, validation recall by distance is `.9274` at 0-10 m,
`.9216` at 10-20 m, `.7281` at 20-30 m and `.3738` at 30-40 m. The decline
at long range remains a documented limitation.

## Rules for all following variants

Hybrid-q, fixed quantization, zstd and AE variants must begin from this exact
perception baseline. They may change only transport encoding/decoding at the
frozen `Z=C2` boundary. They must not change:

- checkpoint weights or architecture;
- seven-channel input preparation;
- FCOS centerness or class scoring;
- semantic or geometry heads;
- vehicle calibration;
- person consolidation or the final `0.25` threshold;
- class mapping, output schema, matching or evaluation rules.

The untouched test set remains reserved for independent publication
confirmation. Compression variants must be compared against the same locked
noAE p025 baseline on identical data and service metrics.

## Bound records

- Machine-readable lock: `PERCEPTION_FORWARD_LOCK_P025_V1.json`
- p025 implementation commit:
  `86b49eaa30ac59cea9b2a467e447525feb3e8ca0`
- Train qualification SHA-256:
  `3d403dd481235aa50353747104dcb90339dcde322373975fba4498925f86b405`
- Validation confirmation SHA-256:
  `ce1bc88736064d8dba59a3bb578ab47db2d0400857f704af2c591701f4dd403b`
- Frozen validation detections SHA-256:
  `a682a1fc5eabb2e59e07449a8c6b5fc604077b40ef094b57dc30c5a18d7ec260`
- Frozen validation AVO table SHA-256:
  `abb976f388ad33e8806d080750e9e7fbe1b1eb60e7e18ea55bedc60dce011386`
