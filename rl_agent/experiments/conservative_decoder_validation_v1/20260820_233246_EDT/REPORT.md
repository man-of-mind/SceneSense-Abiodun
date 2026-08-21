# Conservative decoder validation report

Audit: `conservative_decoder_validation_v1/20260820_233246_EDT`  
Conclusion: **`RETRAINING_PILOT_JUSTIFIED`**  
Frozen global setting: **`none`**

## Outcome

Neither 1 m nor 2 m met every frozen validation floor and paired material-improvement requirement.
This is an offline promotion-review result, not deployment approval. No production file, checkpoint, CARLA/OAI path, registry, runtime, launcher, controller, or map server was changed.

## Provenance and split integrity

The original manifest contains 10,911 train, 2,110 validation, and 2,162 test identifiers. Pairwise overlap is zero, and all four checkpoint families' saved split files exactly match the manifest in identity and order. The frozen validation identifier SHA-256 is `91204710264ec63f53e4cdbae3cec8e9526c7f99bb329c4817346de281b1ba9c`.

All source/checkpoint/data hashes are in `input_hash_manifest.json`; preregistration completion precedes both inference markers.

## Validation per-profile/class metrics

| profile_id | candidate | class_name | precision | recall | xy_mae_m | fp_per_frame_all |
| --- | --- | --- | --- | --- | --- | --- |
| ae128_u4_q0p3_quality | baseline | vehicle | 0.4963 | 0.9279 | 0.7995 | 1.4294 |
| ae128_u4_q0p3_quality | baseline | person | 0.6214 | 0.8980 | 1.0303 | 1.4294 |
| ae128_u4_q0p3_quality | world_suppression_1m | vehicle | 0.7228 | 0.9270 | 0.8561 | 0.6336 |
| ae128_u4_q0p3_quality | world_suppression_1m | person | 0.7215 | 0.8945 | 1.0661 | 0.6336 |
| ae128_u4_q0p3_quality | world_suppression_2m | vehicle | 0.8187 | 0.9258 | 0.8988 | 0.3682 |
| ae128_u4_q0p3_quality | world_suppression_2m | person | 0.8131 | 0.8847 | 1.1256 | 0.3682 |
| ae32_u4_q0p5_compact_fallback | baseline | vehicle | 0.4951 | 0.9283 | 0.8523 | 1.4422 |
| ae32_u4_q0p5_compact_fallback | baseline | person | 0.6147 | 0.8896 | 1.0971 | 1.4422 |
| ae32_u4_q0p5_compact_fallback | world_suppression_1m | vehicle | 0.7321 | 0.9266 | 0.9165 | 0.6251 |
| ae32_u4_q0p5_compact_fallback | world_suppression_1m | person | 0.7103 | 0.8826 | 1.1199 | 0.6251 |
| ae32_u4_q0p5_compact_fallback | world_suppression_2m | vehicle | 0.8232 | 0.9232 | 0.9491 | 0.3659 |
| ae32_u4_q0p5_compact_fallback | world_suppression_2m | person | 0.8057 | 0.8749 | 1.1869 | 0.3659 |
| ae32_u4_q0p9_high_q_rescue | baseline | vehicle | 0.4908 | 0.9216 | 0.8507 | 1.4360 |
| ae32_u4_q0p9_high_q_rescue | baseline | person | 0.6247 | 0.8875 | 1.1380 | 1.4360 |
| ae32_u4_q0p9_high_q_rescue | world_suppression_1m | vehicle | 0.7249 | 0.9203 | 0.9155 | 0.6213 |
| ae32_u4_q0p9_high_q_rescue | world_suppression_1m | person | 0.7232 | 0.8819 | 1.1649 | 0.6213 |
| ae32_u4_q0p9_high_q_rescue | world_suppression_2m | vehicle | 0.8231 | 0.9186 | 0.9570 | 0.3550 |
| ae32_u4_q0p9_high_q_rescue | world_suppression_2m | person | 0.8163 | 0.8728 | 1.2242 | 0.3550 |
| ae64_u4_q0p7_compact | baseline | vehicle | 0.4928 | 0.9194 | 0.8224 | 1.4213 |
| ae64_u4_q0p7_compact | baseline | person | 0.6266 | 0.8854 | 1.1263 | 1.4213 |
| ae64_u4_q0p7_compact | world_suppression_1m | vehicle | 0.7327 | 0.9178 | 0.8797 | 0.6095 |
| ae64_u4_q0p7_compact | world_suppression_1m | person | 0.7190 | 0.8798 | 1.1540 | 0.6095 |
| ae64_u4_q0p7_compact | world_suppression_2m | vehicle | 0.8227 | 0.9161 | 0.9252 | 0.3531 |
| ae64_u4_q0p7_compact | world_suppression_2m | person | 0.8175 | 0.8672 | 1.2053 | 0.3531 |
| noae_u8_q0_reference | baseline | vehicle | 0.5211 | 0.8895 | 0.9173 | 1.2602 |
| noae_u8_q0_reference | baseline | person | 0.6342 | 0.8735 | 1.0792 | 1.2602 |
| noae_u8_q0_reference | world_suppression_1m | vehicle | 0.7187 | 0.8878 | 0.9571 | 0.6161 |
| noae_u8_q0_reference | world_suppression_1m | person | 0.7236 | 0.8707 | 1.1039 | 0.6161 |
| noae_u8_q0_reference | world_suppression_2m | vehicle | 0.8108 | 0.8840 | 1.0036 | 0.3555 |
| noae_u8_q0_reference | world_suppression_2m | person | 0.8251 | 0.8602 | 1.1596 | 0.3555 |

## Validation pooled-normal paired uncertainty

| candidate | resampling_unit | metric | observed_delta | delta_ci95_low | delta_ci95_high |
| --- | --- | --- | --- | --- | --- |
| world_suppression_1m | frame | vehicle_precision | 0.2344 | 0.2239 | 0.2447 |
| world_suppression_1m | frame | vehicle_recall | -0.0014 | -0.0025 | -0.0006 |
| world_suppression_1m | frame | vehicle_xy_mae_m | 0.0594 | 0.0539 | 0.0645 |
| world_suppression_1m | frame | person_precision | 0.0961 | 0.0879 | 0.1045 |
| world_suppression_1m | frame | person_recall | -0.0054 | -0.0086 | -0.0024 |
| world_suppression_1m | frame | person_xy_mae_m | 0.0287 | 0.0209 | 0.0359 |
| world_suppression_1m | frame | fp_per_frame_all | -0.8082 | -0.8493 | -0.7632 |
| world_suppression_1m | scenario | vehicle_precision | 0.2344 | 0.2105 | 0.2595 |
| world_suppression_1m | scenario | vehicle_recall | -0.0014 | -0.0024 | -0.0005 |
| world_suppression_1m | scenario | vehicle_xy_mae_m | 0.0594 | 0.0498 | 0.0707 |
| world_suppression_1m | scenario | person_precision | 0.0961 | 0.0871 | 0.1040 |
| world_suppression_1m | scenario | person_recall | -0.0054 | -0.0109 | -0.0005 |
| world_suppression_1m | scenario | person_xy_mae_m | 0.0287 | 0.0215 | 0.0362 |
| world_suppression_1m | scenario | fp_per_frame_all | -0.8082 | -0.9785 | -0.6375 |
| world_suppression_2m | frame | vehicle_precision | 0.3269 | 0.3145 | 0.3391 |
| world_suppression_2m | frame | vehicle_recall | -0.0035 | -0.0053 | -0.0020 |
| world_suppression_2m | frame | vehicle_xy_mae_m | 0.0996 | 0.0903 | 0.1082 |
| world_suppression_2m | frame | person_precision | 0.1912 | 0.1799 | 0.2036 |
| world_suppression_2m | frame | person_recall | -0.0154 | -0.0204 | -0.0105 |
| world_suppression_2m | frame | person_xy_mae_m | 0.0880 | 0.0738 | 0.1016 |
| world_suppression_2m | frame | fp_per_frame_all | -1.0686 | -1.1216 | -1.0169 |
| world_suppression_2m | scenario | vehicle_precision | 0.3269 | 0.3022 | 0.3518 |
| world_suppression_2m | scenario | vehicle_recall | -0.0035 | -0.0056 | -0.0016 |
| world_suppression_2m | scenario | vehicle_xy_mae_m | 0.0996 | 0.0881 | 0.1127 |
| world_suppression_2m | scenario | person_precision | 0.1912 | 0.1749 | 0.2120 |
| world_suppression_2m | scenario | person_recall | -0.0154 | -0.0255 | -0.0054 |
| world_suppression_2m | scenario | person_xy_mae_m | 0.0880 | 0.0694 | 0.1111 |
| world_suppression_2m | scenario | fp_per_frame_all | -1.0686 | -1.2706 | -0.8552 |

## One-shot untouched-test evidence

No test metrics were produced because no candidate was eligible or required evidence was unavailable.

## Latency

GPU end-to-end decoder latency is CUDA-synchronized feature-to-retained-list time. Incremental list latency measures only the predicted-only suppression on an already-retained list. These scopes are intentionally not combined.

| split | profile_id | candidate | latency_scope | samples | p50_ms | p95_ms | max_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| val | ae128_u4_q0p3_quality | decoder_envelope_shared | gpu_decoder_end_to_end | 2085 | 5.106659 | 10.823429 | 19.927622 |
| val | ae32_u4_q0p5_compact_fallback | decoder_envelope_shared | gpu_decoder_end_to_end | 2085 | 5.947559 | 12.518842 | 16.627972 |
| val | ae32_u4_q0p9_high_q_rescue | decoder_envelope_shared | gpu_decoder_end_to_end | 2085 | 5.809106 | 12.568427 | 16.277543 |
| val | ae64_u4_q0p7_compact | decoder_envelope_shared | gpu_decoder_end_to_end | 2085 | 4.016594 | 9.489069 | 13.096269 |
| val | noae_u8_q0_reference | decoder_envelope_shared | gpu_decoder_end_to_end | 2085 | 4.167325 | 5.703516 | 12.337820 |
| val | ae128_u4_q0p3_quality | baseline | incremental_retained_list | 42200 | 0.000091 | 0.000130 | 0.061340 |
| val | ae128_u4_q0p3_quality | world_suppression_1m | incremental_retained_list | 42200 | 0.000602 | 0.003219 | 0.012700 |
| val | ae128_u4_q0p3_quality | world_suppression_2m | incremental_retained_list | 42200 | 0.000585 | 0.002911 | 0.009220 |
| val | ae32_u4_q0p5_compact_fallback | baseline | incremental_retained_list | 42200 | 0.000092 | 0.000130 | 0.004270 |
| val | ae32_u4_q0p5_compact_fallback | world_suppression_1m | incremental_retained_list | 42200 | 0.000568 | 0.003094 | 0.040553 |
| val | ae32_u4_q0p5_compact_fallback | world_suppression_2m | incremental_retained_list | 42200 | 0.000554 | 0.002807 | 0.009192 |
| val | ae32_u4_q0p9_high_q_rescue | baseline | incremental_retained_list | 42200 | 0.000092 | 0.000129 | 0.116093 |
| val | ae32_u4_q0p9_high_q_rescue | world_suppression_1m | incremental_retained_list | 42200 | 0.000555 | 0.003055 | 0.010420 |
| val | ae32_u4_q0p9_high_q_rescue | world_suppression_2m | incremental_retained_list | 42200 | 0.000541 | 0.002814 | 0.009127 |
| val | ae64_u4_q0p7_compact | baseline | incremental_retained_list | 42200 | 0.000092 | 0.000129 | 0.017168 |
| val | ae64_u4_q0p7_compact | world_suppression_1m | incremental_retained_list | 42200 | 0.000548 | 0.003032 | 0.011953 |
| val | ae64_u4_q0p7_compact | world_suppression_2m | incremental_retained_list | 42200 | 0.000535 | 0.002786 | 0.146991 |
| val | noae_u8_q0_reference | baseline | incremental_retained_list | 42200 | 0.000092 | 0.000130 | 0.002826 |
| val | noae_u8_q0_reference | world_suppression_1m | incremental_retained_list | 42200 | 0.000520 | 0.002867 | 0.010854 |
| val | noae_u8_q0_reference | world_suppression_2m | incremental_retained_list | 42200 | 0.000510 | 0.002640 | 0.009994 |

## Secondary evidence and catalog implication

Retained-list suppression does not alter feature serialization, payload, or segmentation logits. Their per-profile measurements are preserved in `secondary_payload_segmentation_{val,test}.csv` and versioned separately. If a later review promotes this decoder, all affected detection/localization catalog rows must be regenerated globally; payload/segmentation rows may remain only with this invariance link.

## Decision boundary

**`RETRAINING_PILOT_JUSTIFIED`**. Neither 1 m nor 2 m met every frozen validation floor and paired material-improvement requirement. Human review remains mandatory; deployment is neither performed nor approved here.
