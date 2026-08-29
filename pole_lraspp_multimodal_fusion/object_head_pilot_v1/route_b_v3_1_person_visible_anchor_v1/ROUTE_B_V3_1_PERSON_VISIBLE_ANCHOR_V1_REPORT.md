# Route B v3.1 person-private visible-anchor experiment

Terminal: `LRASPP_VISIBLE_ANCHOR_NO_GAIN`

The single licensed scientific attempt completed the registered 24-epoch schedule. Corrected visible-anchor supervision improved person 2D centre/box diagnostics and reduced canonical XY MAE, but canonical F1, recall, low-score recall, and conditional localization did not meet either registered material-gain route. The inherited vehicle and segmentation paths were preserved exactly.

## Provenance and scope

- Starting repository commit: `40357a6133842ef8cf8b7287eb7a1ae1e0a4de9c`.
- Frozen warm start: `experiments/route_b_v3_1_person_refinement_v1/20260828_163100/checkpoints/route_b_v3_1_person_refinement_v1/epoch_040.pt`.
- Warm-start SHA-256: `5c6bb268b43f4dd84bd7a283ff483ec4e87366a50ea51dfacee44979df2bf6e8`.
- Authoritative run: `experiments/route_b_v3_1_person_visible_anchor_v1/20260829_021120/`.
- Dataset split: 16,827 train / 3,345 validation / 0 test; all 10 train and 2 validation episodes remained disjoint.
- Locked test data remained absent and unopened.
- CARLA, OAI, q/quant/AE/zstd, live runtime, and the 288-measurement campaign were untouched. No COCO distillation, teacher, raw tail-side sensor channel, geometry-changing augmentation, threshold/NMS sweep, or v0.25 sensitivity run occurred.

Four earlier timestamp directories contain only preflight/supervisor artifacts. They repaired, in order, preflight-log creation order, whitespace normalization in the repository-status comparison, missing camera-model dimension constants, and a provenance variable-shadowing defect. No optimizer step or scientific inference occurred in them, and the passing authoritative preflight records zero optimizer steps and zero scientific attempts consumed. The `20260829_021120` directory is the only scientific attempt.

## Registered implementation

Every inherited parameter and BatchNorm state was frozen. A person-private tower was attached to detached copies of the existing fused `{low, high}` bundle and warm-initialized from compatible epoch-40 tensors without modifying the source modules. The private decoder emits distinct visible heatmap, visible subcell offset, visible-to-box-centre offset, full box size, visible-to-physical-ray offset, bounded positive forward depth, dimensions, yaw, and radar-support outputs. Camera calibration is used only for geometric decoding; the external object-record schema and split boundary are unchanged.

The registered numerical policy was full FP32. Both BF16-feature/private-FP32 and full-FP32 candidates qualified finite, full FP32 was chosen before epoch 1, and FP16 was never used on the private localization path. The peak learning rate was `3e-5` with one warmup epoch and cosine decay to `3e-6`; AdamW, batch size 16, strong photometric-only augmentation, and the complete loss/sampling contract were frozen before training.

## Target and geometry proofs

- Derived person target rows: 21,459 (17,587 train / 3,872 validation).
- Own-visible anchor pixels: 21,459/21,459 (100%).
- Own-visible stride-4 anchor cells: 21,459/21,459 (100%).
- Anchor rule counts: 21,346 nearest own-visible pixels in the centroid cell; 113 deterministic global nearest-visible fallbacks.
- Reference Gaussian audit reproduction: 21,459/21,459 exact raw and integer radii; zero mismatches.
- Visible-box radius counts were no longer collapsed to one value. Train counts began `r1=15058, r2=1724, r3=489, r4=196`; validation began `r1=3307, r2=418, r3=84, r4=34`.
- Train-derived normalization scales: 4 grid cells for visible-to-box-centre offsets and 5 grid cells for visible-to-physical-ray offsets; validation influenced neither training targets nor parameters.
- Physical projection/unprojection maximum absolute error: `1.0658141036401503e-14 m`.
- Visible anchor, full-box centre, and physical-centre ray remained separate targets.
- Monolithic/split deltas were zero for every compared output.
- Transport remained `low=[1,40,54,96]` (207,360 elements) and `high=[1,960,27,48]` (1,244,160 elements), FP32, with unchanged semantics.
- Inherited tensor-state drift was zero; audit hash `1c65369ccc7fb1f694f663aea40a0c564ab074fdfad699c056df1c7213362f25`.

## Parameter and numerical qualification

| Scope | Total | Trainable | Frozen |
|---|---:|---:|---:|
| Complete model | 6,643,342 | 1,711,120 | 4,932,222 |
| Person-private branch | 1,712,144 | 1,711,120 | 1,024 |
| Backbone | 2,972,528 | 0 | 2,972,528 |
| Segmentation classifier | 246,526 | 0 | 246,526 |
| Native shared object trunk | 1,447,680 | 0 | 1,447,680 |
| Native upsampler | 262,400 | 0 | 262,400 |

All nine private outputs produced finite, nonzero gradients on the real batch-16 qualification batch. The largest weighted loss share was `0.5024726644`, below the registered 0.60 cap. Full-FP32 qualification peaked at 1,469.63 MiB allocated and 2,562 MiB reserved, below 12 GiB.

## Training and checkpoints

Exactly one scientific attempt ran 24 epochs and 25,248 optimizer steps. The epoch-12 catastrophic gate passed all four conditions, so training continued without retry. Final-run peak VRAM was 1,461.16 MiB allocated / 2,738 MiB reserved. End-to-end pipeline wall time was 2,378.43 seconds.

| Epoch | Checkpoint SHA-256 |
|---:|---|
| 6 | `267b569424aff04ae06c7a9850a28171a73d5e42d427802e42241ee9aaa9dd14` |
| 12 | `05d65554f347af18f721cf97e73e3c43481414134bcf2f61383dfab7d41d913e` |
| 18 | `62330263c90ae8d71c2d44b7e0cf164b08dd3f928bee6966f023c619b629b5fc` |
| 24 | `eb0e799e372fe8c15012a68bed1e0ccd5d989726b1b344a6ff68895c50d36091` |

Each checkpoint received exactly one inference pass at score floor 0.02. Score 0.20 results were derived offline from those predictions. Eligible epochs were 18 and 24; the frozen rank selected epoch 18.

## Base versus selected epoch 18

| Person metric | Epoch-40 base | Epoch 18 | Delta |
|---|---:|---:|---:|
| Canonical precision @0.20 | 0.537513 | 0.543137 | +0.005624 |
| Canonical recall @0.20 | 0.518079 | 0.500775 | -0.017304 |
| Canonical F1 @0.20 | 0.527617 | 0.521096 | -0.006521 |
| Canonical recall @0.02 | 0.615186 | 0.595816 | -0.019370 |
| Canonical XY MAE (m) | 1.341153 | 1.286255 | -0.054898 |
| Full-box-centre F1 @0.20 | 0.626772 | 0.642829 | +0.016056 |
| IoU50 F1 @0.20 | 0.529474 | 0.553834 | +0.024360 |
| IoU50 recall @0.02 | 0.566374 | 0.596333 | +0.029959 |
| IoU50 conditional within 3 m @0.02 | 0.814865 | 0.760069 | -0.054796 |

The result supports the supervision diagnosis: visible anchors and corrected radii improved 2D proposal/extent quality, and the private geometry path improved XY error among canonical matches. It did not recover enough people, and localization conditioned on valid IoU50 matches regressed. Route A failed canonical F1, recall, and low-score recall; route B failed canonical F1, IoU50-F1 gain, low-score recall, and the conditional-localization guard. The v0.25 selected-only sensitivity therefore was not licensed and was not run.

## Conditional localization slices

IoU50 matches at score 0.02 for selected epoch 18:

| Slice | Matches | Within 1 m | Within 2 m | Within 3 m | Within 5 m | Median m | P90 m |
|---|---:|---:|---:|---:|---:|---:|---:|
| Overall | 2,309 | 0.3265 | 0.5834 | 0.7601 | 0.9138 | 1.6419 | 4.6569 |
| 0-10 m | 135 | 0.3778 | 0.7037 | 0.9111 | 1.0000 | 1.3126 | 2.7981 |
| 10-20 m | 948 | 0.4072 | 0.6751 | 0.8407 | 0.9451 | 1.2656 | 3.8335 |
| 20-30 m | 871 | 0.2560 | 0.4994 | 0.6831 | 0.8749 | 2.0064 | 5.4017 |
| 30-40 m | 355 | 0.2648 | 0.4986 | 0.6761 | 0.8930 | 2.0033 | 5.0935 |
| Clear v0.25 visibility | 2,145 | 0.3347 | 0.5963 | 0.7748 | 0.9226 | 1.5722 | 4.5296 |
| Primary-v0.10-only visibility | 164 | 0.2195 | 0.4146 | 0.5671 | 0.7988 | 2.3936 | 6.1609 |
| Radar supported | 2,186 | 0.3317 | 0.5906 | 0.7681 | 0.9190 | 1.6087 | 4.5662 |
| Radar unsupported | 123 | 0.2358 | 0.4553 | 0.6179 | 0.8211 | 2.3165 | 6.3722 |

## Frozen-path preservation and service status

For each of epochs 6, 12, 18, and 24, all 25,246 vehicle detection rows were bit-identical after excluding artifact-only prediction indices, all 3,345 segmentation PNG hashes were bit-identical, and vehicle/segmentation metric deltas were zero. Vehicle P/R/F1 remained `0.79443359375 / 0.83943865442 / 0.81631629120`, R@0.02 `0.87194303993`, XY MAE `0.83048259021 m`, IoU `0.87054405600`, and duplicate FP `644`. Foreground mIoU remained `0.66220551335` and person box-mask IoU `0.45386697069`.

Only 2/9 service gates passed (vehicle XY and vehicle IoU); full service readiness is not claimed. Vehicle precision/recall, person box-mask IoU, and foreground mIoU were structurally unreachable because their paths were intentionally frozen.

Selected checkpoint: `experiments/route_b_v3_1_person_visible_anchor_v1/20260829_021120/checkpoints/route_b_v3_1_person_visible_anchor_v1/epoch_018.pt`

Selected SHA-256: `62330263c90ae8d71c2d44b7e0cf164b08dd3f928bee6966f023c619b629b5fc`

The run directory contains the resolved config, registered design, provenance and hashes, target/Gaussian audit, warm-start mapping, parameter and split-parity audits, numerical qualification, per-epoch metrics, checkpoint hashes, four evaluation records, selection decision, completion sentinel, and successful desktop-notification record.
