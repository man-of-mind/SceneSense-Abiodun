# Frozen person p025 service-candidate confirmation

The fixed post-consolidation person threshold 0.25 qualified on the two
registered train-only holdout episodes under AVO >= 0.65. The thin candidate
wrapper therefore runs the accepted p020 service unchanged and then removes
only consolidated person outputs whose FP32 score is below 0.25. It never
filters, reorders, or changes a vehicle field.

## Train-only holdout

| view | episode | precision | recall | F1 | XY MAE m |
|---|---|---:|---:|---:|---:|
| p020 | aggregate | 0.822928 | 0.881847 | 0.851369 | 0.536021 |
| p020 | canonical_v3_03_train_30_30_s503_tm1503 | 0.801242 | 0.924069 | 0.858283 | 0.525700 |
| p020 | canonical_v3_04_train_50_50_s504_tm1504 | 0.831954 | 0.865985 | 0.848629 | 0.540158 |
| p025 | aggregate | 0.898881 | 0.879890 | 0.889284 | 0.534674 |
| p025 | canonical_v3_03_train_30_30_s503_tm1503 | 0.871274 | 0.921203 | 0.895543 | 0.525092 |
| p025 | canonical_v3_04_train_50_50_s504_tm1504 | 0.910431 | 0.864370 | 0.886803 | 0.538510 |

All seven registered gates passed. The p025 set contains 3,217 of the 3,460
p020 consolidated person outputs as an exact ordered subset, with scores and
all non-score fields unchanged. The policy's bitwise runtime invariant and the
single synthetic regression verify that every vehicle output is unchanged.

The AVO table contains 4,703 canonically eligible train-holdout actor-frames.
It reuses 4,591 exact-identity rows from the existing train-only reference and
computes only the 112 missing rows from 106 saved holdout depth frames.

## Frozen validation at person threshold 0.25

| view | episode | precision | recall | F1 | XY MAE m |
|---|---|---:|---:|---:|---:|
| AVO>=0.65 | aggregate | 0.704187 | 0.713243 | 0.708686 | 0.812181 |
| AVO>=0.65 | canonical_v3_05_val_30_30_s601_tm1601 | 0.758710 | 0.689332 | 0.722359 | 0.811478 |
| AVO>=0.65 | canonical_v3_06_val_50_50_s602_tm1602 | 0.684432 | 0.723320 | 0.703339 | 0.812464 |
| canonical v0.10 | aggregate | 0.796686 | 0.596074 | 0.681932 | 0.839516 |
| canonical v0.10 | canonical_v3_05_val_30_30_s601_tm1601 | 0.841121 | 0.572727 | 0.681449 | 0.828203 |
| canonical v0.10 | canonical_v3_06_val_50_50_s602_tm1602 | 0.781192 | 0.605339 | 0.682114 | 0.843763 |

Validation threshold behavior was previously explored. The untouched test set
remains necessary for independent publication confirmation. The
supervisor-approved p020 service is unchanged; p025 is a proposed deployment
candidate awaiting final acceptance.

## Frozen inputs and execution

- Feasibility result SHA-256: `a1bb8b2b7062abc2d0ef4c5cbc715154c5a4e9f1da64e050547de14c56bdddde`
- Cache manifest SHA-256: `6e9386a6ee1d87cb19685ae0afb1c54cc6b9406bfae4ccf01d9e804578ddcc4c`
- Cache shard hash-map SHA-256: `b599e883affc13ec1fd723c42e3901423af6062c2ac495e433254d2bdeef0d4b`
- Training support records SHA-256: `8755b1904c821e6942197a3d41abb18806d049131a764ccb9f6100ab80493faf`
- Training reference JSON SHA-256: `a825cffac4a060ee422951bb7d5af0b10d15eb39a347c081af836de35e6c1fff`
- Frozen validation detections SHA-256: `a682a1fc5eabb2e59e07449a8c6b5fc604077b40ef094b57dc30c5a18d7ec260`
- Frozen validation AVO table SHA-256: `abb976f388ad33e8806d080750e9e7fbe1b1eb60e7e18ea55bedc60dce011386`
- Runtime: 7.997 seconds train qualification + 5.397 seconds validation = 13.394 seconds total.
- Compact CPU checks: 8 passed, including exactly one new combined subset/vehicle-invariance regression.

No training, cache rebuild, model inference, CUDA, CARLA, visibility-method
change, or test access occurred. Generated evidence is under
`experiments/splitfusion_fcos_person_p025_calibration_v1/`.

PERSON_P025_TRAIN_HOLDOUT_QUALIFIED_VALIDATION_CONFIRMED
