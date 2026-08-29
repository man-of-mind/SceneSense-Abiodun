# PERSON_CONTRACT_AUDIT_DETECTION_DOMINANT

Low reported person recall is primarily image-space detection/box-support failure under the preregistered IoU≥0.50 diagnostic. At the retained score floor 0.02, 1,092/1,490 (73.289%) base canonical joint FNs lack a valid one-to-one IoU50 person assignment, while 346/1,490 (23.221%) have valid 2D support but paired world error above 3 m. The distilled diagnostic has the same classification: 889/1,295 (68.649%) lack a 2D assignment and 357/1,295 (27.568%) are localization failures.

The canonical class-aware, one-to-one 3 m world-XY metric remains authoritative for deployment. Image-space results are diagnostic and do not establish service readiness or promote any threshold.

## Input and canonical parity

- Frozen validation population: 3,345 frames and 3,872 primary-v0.10 person GT; 16,827/3,345/0 train/validation/test frames.
- Ten train episodes and two validation episodes are sample-ID and episode-disjoint; zero locked-test rows or references were opened.
- Both retained prediction sets have floor 0.02 and their checkpoint/detection hashes match their immutable inference manifests.
- Base at 0.20: P/R/F1 `0.537513397642 / 0.518078512397 / 0.527617043661`; R@0.02 `0.615185950413`.
- Distilled epoch 12 at 0.20: P/R/F1 `0.428195792880 / 0.546745867769 / 0.480263157895`; R@0.02 `0.665547520661`.

## Deterministic 2D diagnostics

| model | score | definition | TP / FP / FN | precision | recall | F1 |
|---|---:|---|---:|---:|---:|---:|
| base e40 | .20 | FULL_BOX_CENTER | 2409 / 1406 / 1463 | .631455 | .622159 | .626772 |
| base e40 | .20 | FULL_BOX_IOU_050 | 2012 / 1716 / 1860 | .539700 | .519628 | .529474 |
| base e40 | .20 | FULL_BOX_IOU_030 | 2387 / 1426 / 1485 | .626016 | .616477 | .621210 |
| base e40 | .02 | FULL_BOX_CENTER | 2762 / 13091 / 1110 | .174226 | .713326 | .280051 |
| base e40 | .02 | FULL_BOX_IOU_050 | 2193 / 13552 / 1679 | .139282 | .566374 | .223582 |
| base e40 | .02 | FULL_BOX_IOU_030 | 2709 / 13139 / 1163 | .170936 | .699638 | .274746 |
| distilled e12 | .20 | FULL_BOX_CENTER | 2526 / 2494 / 1346 | .503187 | .652376 | .568151 |
| distilled e12 | .20 | FULL_BOX_IOU_050 | 2101 / 2825 / 1771 | .426512 | .542614 | .477609 |
| distilled e12 | .20 | FULL_BOX_IOU_030 | 2505 / 2514 / 1367 | .499103 | .646952 | .563491 |
| distilled e12 | .02 | FULL_BOX_CENTER | 2904 / 32797 / 968 | .081342 | .750000 | .146767 |
| distilled e12 | .02 | FULL_BOX_IOU_050 | 2277 / 33290 / 1595 | .064020 | .588068 | .115469 |
| distilled e12 | .02 | FULL_BOX_IOU_030 | 2837 / 32853 / 1035 | .079490 | .732696 | .143420 |

Assignment differences, ignored-neutral counts, class-confusion support, and one-prediction/multiple-GT contention are recorded in `person_2d_metrics.csv`. The IoU50 definition above is primary; no best-definition selection occurred.

For primary IoU50 pairs at 0.02, base conditional localization succeeds within 1/2/3/5 m for `31.69% / 64.93% / 81.49% / 94.25%`; median/p90/p95 errors are `1.481 / 4.018 / 5.166 m`. Distilled values are `31.93% / 64.08% / 79.89% / 93.98%` and `1.497 / 4.092 / 5.248 m`. Full distance, area, visibility-contract, and radar slices are in `conditional_localization.csv`.

## Heatmap-target visibility defect

`HEATMAP_TARGET_VISIBILITY_DEFECT_CONFIRMED`

| contract | split | denominator | own visible | closer occluder | inside box, not visible | off-own total |
|---|---|---:|---:|---:|---:|---:|
| v0.10 | train | 17,587 | 15,323 (87.127%) | 2,159 (12.276%) | 105 (.597%) | 2,264 (12.873%) |
| v0.10 | val | 3,872 | 3,376 (87.190%) | 481 (12.423%) | 15 (.387%) | 496 (12.810%) |
| v0.25 | train | 14,986 | 14,162 (94.502%) | 763 (5.091%) | 61 (.407%) | 824 (5.498%) |
| v0.25 | val | 3,376 | 3,194 (94.609%) | 173 (5.124%) | 9 (.267%) | 182 (5.391%) |

All reconstructed nearest-visible points and visible centroids land in an in-bounds stride-4 cell. The centroid cell contains an own-visible pixel for 3,861/3,872 v0.10 validation rows; the nearest-visible cell does so for 3,872/3,872. Counts and percentages by tier, range, area, radar, split, and episode are materialized in `target_visibility_summary.csv`; all continuous offsets and cell checks remain in `target_visibility_audit.csv`. The 24 individual panels have exactly 8 closer-occluder, 8 inside-box-not-visible, 4 clean, and 4 largest-offset review roles.

## Gaussian implementation mismatch

`GAUSSIAN_RADIUS_IMPLEMENTATION_MISMATCH_CONFIRMED`

All independent square, tall, tiny, half-cell, boundary, and paired-population tests pass. On validation people, current versus standard CornerNet/CenterNet reference values are:

- radius 1: `3,853/3,872 (99.509%)` versus `3,283/3,872 (84.788%)`;
- raw radius mean: `.2911` versus `1.0073`;
- mean support cells: `9.122` versus `14.068`.

The production function uses different quadratic roots/divisors from the standard reference. Forced floor-cell peaks also create the measured left/right and up/down half-cell asymmetry. Full distance/area summaries are in `gaussian_radius_summary.csv`.

## Ranking/calibration and source findings

`DISTILLATION_CALIBRATION_SHIFT_CONFIRMED` is false. The requested observations reproduce (`base F1=.5808985 @ .40`, `distilled F1=.5823529 @ .50`), but the denser curve peaks at nearly the same score (`base .5817484 @ .440`, `distilled .5856737 @ .435`). Joint dense AUPRC rises from `.477120` to `.499958`; therefore the evidence includes a modest ranking change and does not isolate a calibration-only shift.

Source inspection also establishes:

- `CONFIRMED_DEFECT`: the earlier diagnostic slices key metadata by `source_identity` alone instead of `(sample_id, source_identity)`.
- `CONFIRMED_DESIGN_LIMIT`: visibility eligibility does not relocate actor positives away from the full projected-box centre; native targets consume that centre and force its floor cell to one.
- `CONFIRMED_DESIGN_LIMIT`: COCO distillation freezes the regression output head while updating its person heatmap, shared trunk, and upstream backbone.
- `CONFIRMED_DESIGN_LIMIT`: augmentation transforms intrinsics, but model forward/localization loss never consumes them.
- `CONFIRMED_DESIGN_LIMIT`: the physical stride-16 high tensor is explicitly average-pooled into a lossy nominal stride-32 train-only ROI level; this is not a silent raw-scale mismatch.
- `UNTESTED_HYPOTHESIS`: the odd 27→14 ceil-pooled terminal row may not share the nominal uniform stride-32 geometry. The existing coordinate probe does not propagate coordinates through that pooling operation.
- `NOT_A_DEFECT`: canonical world matching, frozen depth-visibility reconstruction, and absence of the train-only adapter from deployable state match their registered contracts.

## Exactly one next design

A factorized visible-person centre-and-range head: supervise the person heatmap at the depth-consistent visible-mask centroid, then predict camera-ray/metric range separately from the same seven-channel RGB+radar LR-ASPP features. Preserve the existing `{low, high}` split transport and q/quant/AE/zstd attachment point, with no raw RGB/radar tail side channel. This recommendation is not implemented or trained here.

Authoritative create-only artifacts: `experiments/route_b_v3_1_person_contract_audit_v1/20260829_011329/`. Wall time was 240.822 s. No training, optimizer step, new inference, threshold calibration/promotion, test access, CARLA, OAI, q/AE, live split runtime, or 288 work occurred.
