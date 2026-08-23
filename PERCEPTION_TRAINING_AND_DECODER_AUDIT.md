# Perception training and decoder audit

Date: 2026-08-22  
Scope: read-only inspection. No CARLA process, collection, training, checkpoint write, production edit, or final-test evaluation was performed.

## Finding

The smallest justified AE64 training pilot is a **vehicle-only cap on the adaptive center-heatmap radius**, with the architecture, split-inference tensors, regression heads, losses, and pedestrian targets otherwise unchanged. This directly addresses a code-supported duplicate mechanism: adaptive vehicle targets are often much broader than the configured 4 px radius, their Gaussian shoulders receive deliberately weak negative focal-loss weight, and the decoder can retain score-ranked peaks separated by only 5 px in image-grid Chebyshev distance. More full-route data is needed for coverage and leakage-safe validation, but data alone does not remove this mechanism.

A 1 m class-aware predicted-world **cluster representative** is the only decoder alternative worth carrying as a frozen secondary comparison. It must not be promoted from this audit. The prior winner-take-all 1 m and 2 m settings did not establish a globally eligible decoder.

## Exact training provenance

All four families used the same code and merged dataset:

- Dataset: `fusion_training_data/moving_ego_pps200000_merged_8loops_stride2`
- Base config: `pole_lraspp_multimodal_fusion/configs/fusion_full_run.yaml`
- Trainer/model/targets: `pole_lraspp_multimodal_fusion/pole_lraspp_multimodal_fusion/{train_fusion.py,model.py,object_targets.py}`
- Evaluator: `pole_lraspp_multimodal_fusion/pole_lraspp_multimodal_fusion/evaluate_fusion.py`
- Launchers: `rl_agent/ae_integrated/{run_noae_baseline.sh,run_ae_integrated.sh}`
- Common warm start: `experiments/mprime_dropaware_20260708/stage2_obj_drop/checkpoints/mprime_stage2_obj_drop/best.pt`

| Family | Trial config | AE initialization | Training seed derived by code | Frozen checkpoint (best epoch; SHA-256) |
|---|---|---|---:|---|
| no-AE | `rl_agent/ae_integrated/mprime_joint_noae.json` | none | 2115118955 | `experiments/ae_integrated_20260710/noae_baseline/checkpoints/mprime_joint_noae/best.pt` (39; `f319e2a5e8fb134e74c24c0822233e17368df6e4c733add658026603e131d4fa`) |
| AE32 | `rl_agent/ae_integrated/ae32_integrated.json` | `rl_agent/feature_ae/checkpoints/ae_b32_v2clean.pt` | 170365354 | `experiments/ae_integrated_20260710/ae32/checkpoints/ae32_integrated/best.pt` (38; `10cebbeede4da992e68850d8f38358e89000b62524be25b68c88517d7b58f9b2`) |
| AE64 | `rl_agent/ae_integrated/ae64_integrated.json` | `rl_agent/feature_ae/checkpoints/ae_b64_v2clean.pt` | 2163670643 | `experiments/ae_integrated_20260710/ae64/checkpoints/ae64_integrated/best.pt` (33; `c6a2362c7c2d72ff31825508ae7532c0796ec063a8556317d47d8d30fad99480`) |
| AE128 | `rl_agent/ae_integrated/ae128_integrated.json` | `rl_agent/feature_ae/checkpoints/ae_b128_v2clean.pt` | 4112758664 | `experiments/ae_integrated_20260710/ae128/checkpoints/ae128_integrated/best.pt` (36; `601984b96d85e1d72e9521f8a796103ab630834991d0957b62990116524cd62c`) |

Each checkpoint embeds the same 768x432, four-radar-channel, shared-head configuration and the same 10,911/2,110/2,162 train/validation/test identifier lists; only the integrated high-feature AE bottleneck differs (0/32/64/128). The launchers explicitly contain a historical `--split test` gate evaluation and corresponding gate artifacts exist. The conservative decoder study did not run a new test evaluation, but the old 2,162-frame split should not be described as historically unseen. It was not accessed in this audit and must not be used for the Route B pilot.

## Verified model outputs

The model is a MobileNetV3 LR-ASPP early-fusion network over normalized RGB plus four radar raster channels. The high feature passes through the integrated AE for AE families; high and 1/8-scale low features feed a three-layer shared object head. Both segmentation and object logits are bilinearly resized to the input image size.

- `out`: three segmentation logits: background, vehicle, person.
- `object`, 14 channels: two class-specific center-heatmap logits (vehicle/person), local camera-frame XYZ (3), physical XYZ dimensions (3), yaw sine/cosine (2), parked logit (1), radar-support logit (1), and normalized 2D-box width/height (2).
- There is no separate objectness head, class softmax, temperature/calibration output, uncertainty output, or learned confidence calibrator. Detection confidence is the sigmoid of the selected class heatmap. Parked and radar-support scores are auxiliary sigmoid outputs.

## Training path

1. **GT filtering and assignment.** Actor-derived vehicle/person rows require a projected center inside the original image, 2D box area at least 12 px, and sensor/world coordinates within 40 m. Rows are area-sorted; one regression target is installed at the rounded center pixel, with the largest-area object winning an exact center-cell collision. Up to 64 GT objects are retained.
2. **Center target.** A class-specific CenterNet Gaussian uses minimum overlap 0.7 and radius `max(2, round(adaptive_radius))`; because adaptive mode is on, `heatmap_radius_px: 4` is not a cap. On all eligible historical GT, vehicle radii were p50=3, p90=14, p95=17, maximum=32 px; 37.6% exceeded 4 px and 23.4% exceeded 8 px. Person radii were p50/p90=2 and p95=3 px. The rounded center is forced to exactly 1.
3. **Losses.** Center focal loss uses alpha=2, beta=4. Only target=1 cells are positives; Gaussian shoulders are negatives whose contribution is reduced by `(1-target)^4`. Center/location/dimension/yaw/parked/radar/2D-box weights are 4.0/1.5/0.6/0.3/0.2/0.1/1.0. XYZ and dimensions use Smooth-L1, normalized yaw uses Smooth-L1, parked/radar use BCE, and 2D size uses GIoU. Segmentation uses class-weighted cross entropy `[0.5,1,4]` plus 0.5 Lovasz; those weights do not balance the object heatmap classes. Segmentation and total object loss each have weight 1.
4. **Optimization and augmentation.** AdamW, LR 1.5e-4, weight decay 1e-4, batch 16, 40 epochs, three warm-up epochs, cosine decay, patience 12, AMP, all modules trainable except frozen batch-normalization state. Strong RGB-only brightness `[0.65,1.35]` and contrast `[0.7,1.3]` are each applied with probability 0.35. Geometric augmentation is disabled because it would misalign object targets. Drop-aware training samples feature-drop fraction uniformly from 0 to 0.8; validation selection is maximin over clean and q=0.4.
5. **Checkpoint selection.** `loc_dim_loss` selects `-(localization loss + 0.25 * dimension loss)`. Decoded precision, recall, F1, FP/frame, duplicates, calibration, and latency do not participate. There is no explicit hard-negative mining beyond focal background loss.

## Decoder and map installation

`object_targets.py::decode_objects` applies sigmoid, takes a global top-120 over both classes and all pixels, then applies score threshold 0.20. In descending score order it accepts a cell unless a same-class occupied square of radius 2 px is already marked. This is greedy spacing, not a heatmap local-maximum operator and not metric/world NMS. Regression is sampled at every accepted cell; local XYZ is transformed by the supplied camera matrix to CARLA world XYZ. Dimensions are clamped nonnegative, yaw is normalized, and 2D width/height use softplus.

The evaluator then range-filters and performs one-to-one, class-aware greedy world-XY matching within 5 m. The studied 1 m/2 m world suppression is external post-processing, not part of the trained model or current production decoder.

The non-OAI spatial clients `carla_split_inference_udp_fusion_object_pole_client_spatial_stream.py` and `_2.py` discard decoded `class_name` while packaging results and publish every object as `type: Vehicle`. This does not explain the offline class-aware duplicate result, but it would corrupt pedestrian/vehicle identity in `object_map_v2`. The map server `real_time_spatial_map_server_fusion_object_v2.py` normalizes the packet and `_can_join_cluster` rejects a cluster containing the same `source_stream_id`; therefore same-frame, same-stream duplicates remain separate map measurements/objects. Both are production-review findings only, not authorization to edit runtime.

## Why one vehicle produces several predictions

The strongest code-and-data-supported chain is:

1. Large nearby vehicle boxes generate broad adaptive Gaussians; their shoulders sharply reduce the penalty for noncentral high scores.
2. The object head operates on a 1/8 low-feature grid and its logits are bilinearly enlarged, which further couples neighboring output pixels.
3. The decoder does not first isolate true local maxima; a 2 px exclusion is much smaller than many vehicle target footprints, so several samples from one basin survive.
4. Regression is supervised only at the one rounded center cell, yet is read at every retained neighboring cell. Nearby retained cells can consequently map to slightly different world positions around the same GT.
5. Model selection never measures duplicates or precision. The prior audit's 25,453 of 26,491 vehicle FPs (96.1%) classified as same-class duplicates is consistent with this mechanism.

Single-seed, repetitive training data, frame leakage, no explicit hard-negative mining, and uncalibrated heatmap scores can amplify the effect. They are secondary explanations; the near-equal raw vehicle/person actor counts do not support simple class-count imbalance as the primary cause.

## Original dataset audit

| Property | Observed evidence |
|---|---|
| Collection diversity | Three collection runs (low/medium/crowded), all `Town10HD_Opt`, scenario `moving_ego_training`, seed 31, spawn 80; 8 repetitions per run of the same approximately 268.7 m loop (24 laps total), not full-map coverage. Requested actors were 8/10, 20/25, and 28/35 vehicle/pedestrian. |
| Frames/rate | 4,921 + 5,115 + 5,147 = 15,183 frames. Sensors ran at 10 Hz with stride 2, hence 5 Hz saved and median adjacent delta 0.2 s. Roughly one third of adjacent positions moved less than 0.05 m, showing substantial temporal redundancy. |
| Spatial extent | Camera positions occupy only about x `[-110.36,-45.05]`, y `[-65.42,24.73]`; this is a small Town10HD region. |
| Object counts | 59,813 vehicle and 54,786 person actor rows. Under the actual training filters: 17,004 vehicles and 10,235 persons; 3,393 frames have no eligible object (5,253 no eligible vehicle; 8,564 no eligible person). |
| Segmentation/background | A 300-frame mask read showed 96.57% background, 3.32% vehicle, and 0.11% person pixels. Exact person pixels cannot be recovered from the manifest field because it is written as zero before pedestrian rasterization; the masks do contain class 2. Pedestrian masks are filled projected 2D boxes rather than actor silhouettes, so person segmentation IoU has different semantics from vehicle segmentation. |
| Split | Frame/sample-ID hashing with seed 23 produced 10,911 train, 2,110 validation, 2,162 test identifiers and no identifier overlap. It did **not** split episodes, seeds, laps, or route regions. |
| Leakage | 6,801/15,180 (44.8%) adjacent saved-frame pairs cross a split boundary. Using 10 m camera-position cells, train/validation share 31 cells, train/test share all 32 test cells, and validation/test share 31 cells. Historical validation therefore overstates independence. |
| Storage | Referenced payload is 96,230,483,040 bytes (about 89.6 GiB), about 6.04 MiB/frame; dense float radar tensors dominate. |

## Bounded AE64 options

### Option 1 — recommended: vehicle adaptive-radius cap

- **Mechanism:** cap the vehicle center Gaussian at 4 input-image pixels at 768x432; retain the current adaptive rule for persons and all center/regression targets otherwise. This strengthens the negative signal surrounding large vehicle centers without changing output tensors or inference code.
- **Later files/parameters:** a versioned AE64 trial derived from `rl_agent/ae_integrated/ae64_integrated.json`; class-aware `vehicle_heatmap_radius_cap_px: 4` consumed in `object_targets.py::build_object_targets`; service-aware decoded validation selection in `train_fusion.py` (feasibility first, then precision/duplicate evidence). Preserve the AE64 v1 initialization and split interface.
- **Expected benefit:** fewer broad vehicle score basins and fewer same-object peaks, improving vehicle precision and duplicate-FP fraction. No numeric gain is claimed before the paired Route B validation.
- **Risks:** the sharper target can reduce vehicle recall or alter score calibration; fewer retained neighboring candidates can change XY error. Dimension heads are unchanged but are jointly trained, so dimension non-inferiority must be measured. Person targets are unchanged, while shared-backbone drift still requires pedestrian and segmentation gates. Tensor shapes and decoder compute are unchanged, so payload and inference latency should be neutral.
- **Post-processing:** keep the existing baseline and frozen 1 m comparison; do not assume training eliminates the need for conservative post-processing.

### Option 2 — decoder fallback: 1 m score/XY cluster representative

- **Mechanism:** cluster same-class predictions using only predicted class, score, and world XY within 1 m, then retain the score/XY-selected medoid prediction rather than blindly retaining the maximum-score member. This targets duplicates while reducing the localization risk of winner-take-all suppression.
- **Later files/parameters:** candidate-only post-processing adjacent to `decode_objects`, radius 1.0 m, class-aware, no GT input, fixed globally before comparison. Do not edit the production map server during the pilot.
- **Expected benefit:** precision close to the prior 1 m result with a chance of lower XY degradation; prior list-level 1 m suppression cost only about 0.003 ms p95.
- **Risks:** two genuinely distinct same-class objects inside 1 m can merge, reducing recall; the underlying multi-peak heatmap and calibration remain. The retained prediction supplies unchanged dimensions, segmentation is unaffected, and decoder latency should remain negligible but must be measured.
- **Post-processing:** this option is itself post-processing. It is a frozen fallback/ablation, not evidence for decoder-only promotion.

A decoupled or larger object head is not recommended now. Although the code supports a decoupled head, it is less directly tied to the observed mechanism, changes warm-start behavior, and adds training/latency risk without prior evidence.

## Route B data required by this pilot

The accepted qualified Route B route file, progress CSV, and supplied SHA-256 are prerequisites. Do not invent coordinates. The task-specified actor requests **5/5, 15/15, and 25/25 vehicle/pedestrian** supersede the older provisional 20/10, 45/25, 75/50 values still visible in `perception_full_route_collection_v1.json`; achieved local and in-view counts must be reported per frame.

- Retain the scaffold's nine seed bundles across all three densities: 27 complete two-lap episodes after approval (15 train, 6 validation, 6 locked final-test episodes). First stage is only three episodes—one accepted episode per density—followed by review. Splits are by complete seed bundle/episode and Route B region, never by frame; no final-test artifact is evaluated.
- Capture at **2 Hz** (every tenth 20 Hz tick), not 20 Hz and not the historical 5 Hz. This gives useful motion spacing while retaining stops and cuts deterministic near-duplicates without motion-based sampling bias.
- Let `T` be the qualified one-lap expected duration in minutes. With the scaffold's two laps, one episode is about `240T` frames at 2 Hz. The three-episode stage is `720T` frames, `6T` simulated minutes, and conservatively `4.25T` GiB at the old 6.04 MiB/frame format. The 27-episode matrix is `6,480T` frames, `54T` simulated minutes, and `38.2T` GiB. For illustration only, if `T=10`, that is about 7,200 frames/42.5 GiB/1 hour for the stage and 64,800 frames/382 GiB/9 hours for the full matrix, plus warm-up, spawn, cleanup, and I/O overhead. Replace the illustration with the qualified route duration before collection.
- GT must include synchronized episode/seed/density/route/region/progress/lap/frame/timestamps; ego, camera, radar, and world transforms plus camera intrinsics; sensor frame IDs; every in-scope actor's stable ID, class/type, world and sensor XYZ, 3D center/extents or L/W/H, yaw, velocity/parked state, distance, visibility/occlusion/truncation, projected centroid and 2D box, and radar support; class and instance segmentation masks with generation provenance; achieved local/in-view vehicle/person counts. These fields support object matching, XY/centroid error, dimensions, and secondary segmentation evaluation.

## AE64 old-versus-new validation contract

Evaluate AE64 v1 and each retrained AE64 candidate on **exactly the same six complete validation episodes** (two validation seed bundles x three densities), with identical artifacts, ordering, Route B region definitions, quantization/ROI profiles, and decoder parameters. Preserve the split-feature `low`/`high` tensor/interface.

- Primary decoder: score 0.20, image NMS radius 2 px, top-k 120, no world suppression, for both checkpoints. Secondary ablation: the same preregistered 1 m setting for both. Never tune by density, region, scene, or checkpoint.
- Representative AE64 transport cases: clean plus the frozen uint4 q=0, q=0.5, and q=0.7 quality/compact/high-q cases. Report payload and latency for the same cases; do not multiply collection by these profiles.
- Overall, per density, and by preregistered Route B region: vehicle/person precision, recall, F1; FP/frame; duplicate-FP fraction (same-class unmatched predictions within the matching radius of an already-claimed GT divided by all FPs); world-XY MAE; 2D centroid error; per-axis and aggregate dimension MAE; vehicle/person IoU and mIoU; payload; feature-to-list GPU decoder latency; incremental list-processing latency; and end-to-end inference latency.
- Use paired per-frame results and episode/region-aware paired uncertainty. Require vehicle precision superiority and duplicate-FP reduction, all preserved normal floors (vehicle/person recall 0.90/0.85, precision 0.49/0.61, XY MAE 0.90/1.20 m, FP/frame 1.45), and preregistered non-inferiority for pedestrian performance, dimensions, segmentation, payload, and compute. No final-test access without `TEST_EVALUATION_AUTHORIZED`.
- Expansion to AE32, AE128, or no-AE requires AE64 precision superiority, floor compliance, non-inferiority above, stability across the three frozen training seeds and densities/regions, unchanged split interface, and human review. It does not require strict improvement of every scalar.

## Exact minimal next step

1. Wait for the qualified Route B route file, progress CSV, and external SHA-256; validate them before any CARLA startup.
2. After collection authorization, run only one two-lap episode at each of 5/5, 15/15, and 25/25, sampled at 2 Hz. Report duration, frames, object counts, route/segment coverage, storage, GT alignment, and actor cleanup; then stop for approval.
3. If approved, collect the remaining frozen seed bundles and make complete-episode/seed/region splits, leaving the final-test episodes locked.
4. Train only the three-seed AE64 Option 1 candidates from AE64 v1 with versioned checkpoint names. Do not change the split interface or overwrite `best.pt`.
5. Run the paired old-versus-new validation contract above. Review before any final-test evaluation, family expansion, decoder promotion, or production change.

`AE64_TRAINING_CHANGE_PILOT_RECOMMENDED`
