# Relational p070 implementation review packet

## Frozen inputs and historical status

- FCOS epoch-26 SHA-256: `da14d21edbd374c1c3abce02ca4674b9f4097becfba9759aba945cea160a297f`
- Relational-selector SHA-256: `af7e8016dbbab41b4edf9ef30f3780bc504b07efe6772a45ac04f5f10df4555a`
- Historical selector status remains `train_infeasible` under its original
  0.80 precision/recall contract. The checkpoint was not modified.
- The revised objective is post-hoc and requires precision and recall at least
  0.70 across the aggregate and each of the two untouched train-holdout episodes.

## Changed files

All changed files are confined to this package:

- `__init__.py`
- `locked_config.json`
- `contract.py`
- `runtime.py`
- `infer_relational_p070.py`
- `evaluate_relational_p070.py`
- `verify_contract.py`
- `smoke_one_train_frame.py`
- `tests/__init__.py`
- `tests/test_synthetic.py`
- `HOLDOUT_VERIFICATION.json`
- `CUDA_SMOKE_RESULT.json`
- `README.md`
- `REVIEW_PACKET.md`

## Exact calibration and service behavior

All operands and operations below are FP32 at deployment:

```text
person_score = sigmoid(
    logit(clamp(base_score, 1e-6, 1 - 1e-6))
    + selector_residual
    + float32(-2.064300755242339)
)
retain person iff person_score >= float32(0.20)
```

The raw relational boundary `0.6632936000823975` maps bit-exactly to FP32
`0.20`. Vehicles are neither filtered nor reordered and use the accepted
service-candidate FP32 calibrator. Every retained non-score field is selected
unchanged from its original post-NMS index. Consolidation contributes one
feature and is never a hard person filter.

## Reproduced train-holdout frontier point

Canonical frame rematching reproduced identical counts at the raw relational
threshold and at calibrated deployment threshold 0.20:

| Scope | TP | FP | FN | Ignored | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| Aggregate | 2879 | 735 | 777 | 429 | 0.7966242390702822 | 0.7874726477024070 |
| Episode 03 | 898 | 259 | 200 | 175 | 0.7761452031114953 | 0.8178506375227687 |
| Episode 04 | 1981 | 476 | 577 | 254 | 0.8062678062678063 | 0.7744331508991400 |

The minimum across aggregate and per-episode precision/recall is
`0.77443315089914`, so the revised train-only 0.70 gate passes. The immutable
historical checkpoint verifies its 218,742 tied-score frontier boundaries and
still reports that no original 0.80/0.80 joint point exists.

## Minimal verification

The original wrapper's two CPU synthetic test cases pass:

```text
test_calibration_maps_selected_boundary_to_fp32_point_20 ... ok
test_vehicle_behavior_and_non_score_fields_are_unchanged ... ok

Ran 2 tests in 0.001s

OK
```

The one approved real-frame CUDA smoke used train frame index 0 and exactly one
frozen-model forward:

- sample: `canonical_v3_01_train_30_30_s501_tm1501_000000_frame214`
- base / retained candidates: `100 / 20`
- base / retained vehicles: `20 / 20`
- base / retained persons: `80 / 0`
- finite outputs: yes
- original candidate alignment: verified
- all retained non-score fields unchanged: verified
- vehicle candidate behavior unchanged: verified
- segmentation logits unchanged: verified
- validation or test accessed: no
- peak allocated / reserved CUDA memory: `374.681640625 / 444.0 MiB`

Machine-readable details are in `HOLDOUT_VERIFICATION.json` and
`CUDA_SMOKE_RESULT.json`.

## Future single validation command (not run)

```bash
python3 -m \
  pole_lraspp_multimodal_fusion.object_head_pilot_v1.\
splitfusion_fcos_r50_fpn_p2_p7_relational_p070_v1.infer_relational_p070 \
  --device cuda:0 \
  --output experiments/relational_p070_v1/<create-only-validation-run>
```

No validation inference/evaluation, training, test access, sensitivity run,
human-band rescoring, hybrid-q work, quantization, zstd, or autoencoder work was
performed.

## Future canonical v0.10 evaluation command (not run)

Run only after the future inference command has produced a completed immutable
prediction directory:

```bash
python3 -m \
  pole_lraspp_multimodal_fusion.object_head_pilot_v1.\
splitfusion_fcos_r50_fpn_p2_p7_relational_p070_v1.evaluate_relational_p070 \
  --prediction-dir experiments/relational_p070_v1/<completed-validation-run>
```

The evaluator delegates unchanged scoring to the frozen canonical v0.10
scorer. It preserves the original nine gates and adds a clearly separate p070
decision that changes only person precision and recall targets from 0.80 to
0.70. The accepted service-candidate comparison is recorded as person P/R
`0.730673 / 0.600465`. One additional CPU synthetic test covers acceptance of
the relational completion contract and rejection of the old sentinel/schema or
an altered calibration. The evaluator itself has not been run.
