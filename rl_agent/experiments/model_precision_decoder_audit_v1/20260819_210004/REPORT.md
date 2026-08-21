# Raw Object-Detection Precision / Decoder Audit Report

Audit: `model_precision_decoder_audit_v1/20260819_210004`  
Final conclusion: **`POSTPROCESSING_SUFFICIENT`**  
Selected validation-only configuration: **`combo_world_nms_3m_veh_thr_0p25`**

## Executive result

The baseline taxonomy assigns **25,453 of 26,491 pooled vehicle false-positive profile instances (96.1%)** to multiple same-class predictions competing for an already-claimed real object. This directly supports duplicate detections as the main raw-precision failure mode within the frozen evaluator envelope; it is not merely a nearest-GT observation.

The selected predicted-only correction is `combo_world_nms_3m_veh_thr_0p25` with vehicle threshold 0.250, person threshold 0.200, predicted-world NMS radius 3.0 m, and incremental image-space radius 0.0 px. It was selected on audit-validation blocks before the frozen audit-test comparison below.

## Provenance and reproduction

The q=0 table reproduces from the four per-frame files with 2,162 unique frames and three complete quantizer profiles per family. Rounded deltas from the supplied table are recorded in `reproduction_table.csv`.

| variant | veh_precision | veh_recall | veh_f1 | ped_precision | ped_recall | ped_f1 | fp_per_frame | profile_rows | unique_frames |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| noae | 0.5256 | 0.8934 | 0.6619 | 0.6269 | 0.8500 | 0.7216 | 1.2553 | 6486 | 2162 |
| ae32 | 0.4998 | 0.9240 | 0.6487 | 0.6344 | 0.8626 | 0.7311 | 1.3847 | 6486 | 2162 |
| ae64 | 0.5025 | 0.9204 | 0.6501 | 0.6322 | 0.8642 | 0.7302 | 1.3731 | 6486 | 2162 |
| ae128 | 0.4968 | 0.9237 | 0.6461 | 0.6321 | 0.8845 | 0.7373 | 1.4086 | 6486 | 2162 |

The per-object causal replay agrees exactly with per-frame TP/FP/FN counts for **12/12 profiles**. Inputs, evaluator settings, checkpoint hashes, decoder/evaluator hashes, and Stage-A evidence are pinned in `input_hash_manifest.json`.

## Split integrity and limitation

- Audit validation: 822 unique identifiers.
- Frozen audit test: 1,340 unique identifiers.
- Identifier overlap: 0; grouped-block overlap: 0.
- All four families and all three quantizers share the same assignment.

The original dataset path recorded by the evaluator is absent and original validation per-object predictions were not preserved. Consequently, this is a preregistered grouped holdout of the published 2,162-frame evaluation set, not the original model-development validation split. The known aggregate test table and preliminary nearest-GT result were already available before this split. This weakens claims of untouched historical test secrecy but does not create GT-dependent deployment logic.

## FP taxonomy

Primary categories are mutually exclusive and saved per prediction in `fp_taxonomy.csv`. Persistence is a conservative adjacent-retained-frame proxy, not a tracker.

| class_name | category | fp_count | fraction |
| --- | --- | --- | --- |
| person | duplicate_same_class_claimed_gt | 7951 | 0.9166 |
| person | no_plausible_nearby_gt | 465 | 0.0536 |
| person | same_class_near_outside_match_radius | 179 | 0.0206 |
| person | other_nearby_gt_geometry | 49 | 0.0056 |
| person | cross_class_confusion | 30 | 0.0035 |
| vehicle | duplicate_same_class_claimed_gt | 25453 | 0.9608 |
| vehicle | no_plausible_nearby_gt | 462 | 0.0174 |
| vehicle | same_class_near_outside_match_radius | 447 | 0.0169 |
| vehicle | cross_class_confusion | 96 | 0.0036 |
| vehicle | other_nearby_gt_geometry | 33 | 0.0012 |

Temporal and empty-scene flags (orthogonal to the primary category):

| class_name | temporal_status | fp_count | fraction |
| --- | --- | --- | --- |
| person | persistent | 4693 | 0.5410 |
| person | single_frame | 3981 | 0.4590 |
| vehicle | persistent | 17894 | 0.6755 |
| vehicle | single_frame | 8597 | 0.3245 |

| class_name | empty_scene_fp | fp_count | fraction |
| --- | --- | --- | --- |
| person | False | 8517 | 0.9819 |
| person | True | 157 | 0.0181 |
| vehicle | False | 26381 | 0.9958 |
| vehicle | True | 110 | 0.0042 |

## Validation sweep and Pareto selection

The complete validation sweep is in `validation_sweep_results.csv`; `validation_pareto.csv` and `precision_recall_pareto.png/.pdf` preserve the precision-recall frontier. The combination gate was true. Selection maximized pooled validation vehicle F1, with the preregistered tie-breaks.

| candidate | precision | recall | f1 | fp_per_frame | xy_mae_m | pareto_nondominated |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.5152 | 0.9261 | 0.6621 | 0.9361 | 0.7856 | True |
| combo_world_nms_3m_veh_thr_0p25 | 0.9259 | 0.8743 | 0.8994 | 0.0751 | 0.9171 | True |
| image_nms_6px | 0.5147 | 0.9232 | 0.6609 | 0.9350 | 0.7786 | False |
| image_nms_8px | 0.5148 | 0.9232 | 0.6610 | 0.9347 | 0.7790 | False |
| veh_thr_0p225 | 0.5471 | 0.9064 | 0.6823 | 0.8061 | 0.8017 | False |
| veh_thr_0p25 | 0.5790 | 0.8851 | 0.7001 | 0.6914 | 0.8149 | False |
| world_nms_1m | 0.7328 | 0.9243 | 0.8175 | 0.3620 | 0.8368 | True |
| world_nms_2m | 0.8302 | 0.9223 | 0.8738 | 0.2027 | 0.8845 | True |
| world_nms_3m | 0.8838 | 0.9151 | 0.8992 | 0.1293 | 0.9230 | True |

## One frozen audit-test comparison

| candidate | class_name | precision | recall | f1 | fp_per_frame | xy_mae_m | xy_rmse_m | prediction_count | gt_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | person | 0.6269 | 0.8561 | 0.7238 | 0.3244 | 1.0502 | 1.3434 | 13979 | 10236 |
| combo_world_nms_3m_veh_thr_0p25 | person | 0.8823 | 0.8408 | 0.8610 | 0.0714 | 1.1781 | 1.4540 | 9754 | 10236 |
| baseline | vehicle | 0.5006 | 0.9094 | 0.6457 | 1.0732 | 0.8192 | 1.1399 | 34554 | 19020 |
| combo_world_nms_3m_veh_thr_0p25 | vehicle | 0.9187 | 0.8537 | 0.8850 | 0.0893 | 0.9507 | 1.2680 | 17673 | 19020 |

The selected point is not cost-free: vehicle recall changes by **-0.0557**,
matched vehicle XY MAE by **+0.1315 m**, vehicle F1 by
**+0.2393**, and vehicle FP/frame by **-0.9839**.
`POSTPROCESSING_SUFFICIENT` therefore means bounded predicted-only processing can
address the dominant mechanism; it does **not** promote this exact operating point
or declare its recall trade-off deployment-safe. The validation Pareto curve also
retains the less aggressive 1 m and 2 m world-suppression points for a downstream
safety choice.

Grouped paired uncertainty resamples unique frames with all quantizer/family profile rows carried together:

| class_name | metric | baseline | selected | observed_delta | delta_ci95_low | delta_ci95_high |
| --- | --- | --- | --- | --- | --- | --- |
| vehicle | precision | 0.50058 | 0.91875 | 0.41817 | 0.40446 | 0.43163 |
| vehicle | recall | 0.90941 | 0.85368 | -0.05573 | -0.06381 | -0.04842 |
| vehicle | f1 | 0.64572 | 0.88502 | 0.23930 | 0.22792 | 0.25081 |
| vehicle | fp_per_frame | 1.07320 | 0.08930 | -0.98389 | -1.04883 | -0.92351 |
| vehicle | xy_mae_m | 0.81923 | 0.95070 | 0.13148 | 0.11362 | 0.14899 |

Empty- and dense-scene behavior:

| candidate | behavior | class_name | precision | recall | f1 | fp_per_frame | prediction_count | gt_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | empty_gt | vehicle | 0.0000 | nan | nan | 0.0153 | 57 | 0 |
| baseline | dense_5plus | vehicle | 0.5484 | 0.8545 | 0.6680 | 2.0383 | 4712 | 3024 |
| combo_world_nms_3m_veh_thr_0p25 | empty_gt | vehicle | 0.0000 | nan | nan | 0.0054 | 20 | 0 |
| combo_world_nms_3m_veh_thr_0p25 | dense_5plus | vehicle | 0.9049 | 0.7894 | 0.8432 | 0.2404 | 2638 | 3024 |

Pooled frozen-test vehicle distance strata:

| candidate | distance_stratum_m | precision | recall | f1 | fp_per_frame | xy_mae_m |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 0-10 | 0.3857 | 0.9425 | 0.5474 | 0.5053 | 0.2745 |
| combo_world_nms_3m_veh_thr_0p25 | 0-10 | 0.9855 | 0.9069 | 0.9446 | 0.0045 | 0.3815 |
| baseline | 10-20 | 0.4976 | 0.9409 | 0.6509 | 0.1340 | 0.7229 |
| combo_world_nms_3m_veh_thr_0p25 | 10-20 | 0.9160 | 0.9277 | 0.9218 | 0.0120 | 0.8871 |
| baseline | 20-30 | 0.6005 | 0.9314 | 0.7302 | 0.1650 | 1.2373 |
| combo_world_nms_3m_veh_thr_0p25 | 20-30 | 0.8731 | 0.8609 | 0.8669 | 0.0333 | 1.4122 |
| baseline | 30-40 | 0.5841 | 0.8605 | 0.6959 | 0.2688 | 1.0359 |
| combo_world_nms_3m_veh_thr_0p25 | 30-40 | 0.8971 | 0.7847 | 0.8372 | 0.0395 | 1.1720 |

All family/quantizer distance rows are in `frozen_test_distance_strata.csv`. Segmentation IoU is reproduced separately in `reproduction_segmentation_iou.csv` and is decoder-invariant.

## Runtime latency

Only the deployable retained-list post-processing stage can be paired from persisted artifacts; raw heatmap/GPU decoder tensors are absent. Times therefore quantify incremental causal overhead, not end-to-end object-head decode latency.

| configuration | p50_ms | p90_ms | p95_ms | max_ms | profile_frame_samples |
| --- | --- | --- | --- | --- | --- |
| baseline | 0.000778 | 0.001563 | 0.001803 | 0.003534 | 16080 |
| combo_world_nms_3m_veh_thr_0p25 | 0.001318 | 0.004899 | 0.006622 | 0.019519 | 16080 |
| paired_delta | 0.000642 | 0.003361 | 0.004788 | 0.016172 | 16080 |

## Installed-map implication

The isolated replay invoked the actual production server's `_normalize_packet` and `_fuse_and_smooth_objects` functions without starting sockets, Flask, CARLA, or OAI. Verified: `true`. For a single source stream, `_can_join_cluster` rejects same-stream joins; raw count equaled measurement and installed count in every independently reset replay frame: `true`.

| candidate | raw_detection_count | installed_object_count | installed_duplicate_count | tp | fp | fn | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 48533 | 48533 | 21339 | 26060 | 22473 | 3196 | 0.5370 | 0.8908 | 0.6700 |
| combo_world_nms_3m_veh_thr_0p25 | 27427 | 27427 | 1773 | 24843 | 2584 | 4413 | 0.9058 | 0.8492 | 0.8766 |

Thus raw same-stream duplicates survive as separate installed objects in this isolated path; the selected decoder correction reduces them before installation. Multi-frame/multi-stream live precision remains outside this no-CARLA audit.

## Decision boundary

**`POSTPROCESSING_SUFFICIENT`**. The evidence attributes the primary vehicle precision failure to duplicate retained predictions and tests a bounded predicted-only remedy. No checkpoint, object head, AE/backbone, ROI ranking, UE policy, production decoder, production map server, CARLA/OAI path, or catalog was changed. No retraining was started or recommended by default. Proper raw-heatmap local-maximum suppression and end-to-end GPU decoder timing remain unverified because the raw tensors and source dataset are unavailable.
