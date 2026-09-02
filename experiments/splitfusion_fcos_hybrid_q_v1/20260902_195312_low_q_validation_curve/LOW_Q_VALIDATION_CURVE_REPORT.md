# Fixed low-q validation sweep: is there a near-lossless operating point below q=0.30?

Terminal: `HYBRID_Q_LOW_Q_VALIDATION_CURVE_COMPLETE`
Artifact: `experiments/splitfusion_fcos_hybrid_q_v1/20260902_195312_low_q_validation_curve`
Schema: `splitfusion_fcos_hybrid_q_low_q_validation_curve_v1`
Generated: 2026-09-02T20:31:39Z · wall 2306.7 s · RTX 5090 · torch 2.10.0.dev20251114+cu128 / CUDA 12.8

## Answer

**No q passes all 12 registered near-lossless preservation gates.** `largest_q_passing_all_gates = null`.

`person_avo_precision` fails at *every* measured setting, including the smallest step off dense:
q=0.05 already loses 0.01884 against a 0.015 bound (1.26x the bound), and the loss grows
monotonically to 0.05371 at q=0.25. It is the single binding gate over the whole interval.

The interval 0 < q < 0.05 is unmeasured, so a 12/12 point there is not excluded -- but any such
point retains more than 20,429 of 21,504 cells and therefore saves under 5 % of framed payload.
No accuracy value is interpolated into that interval here.

### What *is* near-lossless below 0.30

The three IoU/segmentation gates are genuinely preserved up to q=0.15 and all three cross the
0.010 bound together between q=0.15 and q=0.20:

| gate | bound | q=0.15 | q=0.20 |
|---|---|---|---|
| `foreground_miou` | 0.010 | +0.00848 PASS | +0.01359 FAIL |
| `person_box_mask_iou` | 0.010 | +0.00877 PASS | +0.01443 FAIL |
| `vehicle_iou` | 0.010 | +0.00820 PASS | +0.01275 FAIL |

So a *segmentation*-near-lossless operating point does exist below q=0.30 (q<=0.15, 11/12 gates,
0.8501 framed payload), while the full 12-gate near-lossless set does not. These are different
claims and are not interchangeable.

All four vehicle-detection gates pass at every measured q, several with negative degradation
(q>0 slightly better than dense). Person AVO *recall* also improves at every q. The drop therefore
trades person precision for person recall; it does not uniformly degrade detection.

## Curve

`retained_cells` of 21,504; framed FP32 payload against the 22,020,140 B framed q=0 denominator.

| q | source | cells | framed bytes | ratio | preservation | service |
|---|---|---|---|---|---|---|
| 0.00 | frozen p025, reused | 21,504 | 22,020,140 | 1.000000 | 12/12 (identity) | 7/9 |
| 0.05 | measured | 20,429 | 20,922,028 | 0.950131 | 11/12 | 7/9 |
| 0.10 | measured | 19,354 | 19,821,228 | 0.900141 | 11/12 | 7/9 |
| 0.15 | measured | 18,278 | 18,719,404 | 0.850104 | 11/12 | 7/9 |
| 0.20 | measured | 17,203 | 17,618,604 | 0.800113 | 7/12 | 7/9 |
| 0.25 | measured | 16,128 | 16,517,804 | 0.750123 | 7/12 | 7/9 |
| 0.30 | Phase 6, reused | 15,053 | 15,417,004 | 0.700132 | 7/12 | 7/9 |

### Vehicle detection (canonical 0.20 service point)

| q | P | R | F1 | XY MAE (m) |
|---|---|---|---|---|
| 0.00 | 0.931592 | 0.868435 | 0.898905 | 0.478675 |
| 0.05 | 0.931665 | 0.868022 | 0.898718 | 0.476918 |
| 0.10 | 0.931619 | 0.867403 | 0.898365 | 0.476970 |
| 0.15 | 0.933444 | 0.866887 | 0.898935 | 0.477222 |
| 0.20 | 0.934311 | 0.867403 | 0.899615 | 0.480801 |
| 0.25 | 0.934862 | 0.866371 | 0.899314 | 0.482172 |
| 0.30 | 0.934446 | 0.866371 | 0.899122 | 0.486015 |

### Canonical person (v010 view, locked p025 output threshold)

| q | P | R | F1 | XY MAE (m) |
|---|---|---|---|---|
| 0.00 | 0.796686 | 0.596074 | 0.681932 | 0.839516 |
| 0.05 | 0.779940 | 0.600465 | 0.678535 | 0.841426 |
| 0.10 | 0.771685 | 0.599690 | 0.674902 | 0.843093 |
| 0.15 | 0.766090 | 0.605630 | 0.676475 | 0.846035 |
| 0.20 | 0.758920 | 0.609762 | 0.676214 | 0.845080 |
| 0.25 | 0.750871 | 0.612603 | 0.674726 | 0.846986 |
| 0.30 | 0.742320 | 0.611570 | 0.670632 | 0.842400 |

### Person, AVO >= 0.65 observable view

| q | P | R | F1 | XY MAE (m) | recall 20-40 m |
|---|---|---|---|---|---|
| 0.00 | 0.704187 | 0.713243 | 0.708686 | 0.812181 | 0.577650 |
| 0.05 | 0.685345 | 0.718457 | 0.701510 | 0.816331 | 0.585100 |
| 0.10 | 0.677388 | 0.717414 | 0.696826 | 0.817517 | 0.586819 |
| 0.15 | 0.670106 | 0.722280 | 0.695216 | 0.819034 | 0.593123 |
| 0.20 | 0.661295 | 0.727494 | 0.692817 | 0.816043 | 0.599427 |
| 0.25 | 0.650479 | 0.730970 | 0.688380 | 0.815796 | 0.603438 |
| 0.30 | 0.640439 | 0.729927 | 0.682261 | 0.809916 | 0.601719 |

20-40 m person recall *rises* monotonically to q=0.25 and its gate (bound 0.030) passes everywhere.

### Segmentation

| q | vehicle IoU | person box-mask IoU | foreground mIoU |
|---|---|---|---|
| 0.00 | 0.899013 | 0.527894 | 0.713453 |
| 0.05 | 0.897248 | 0.525499 | 0.711373 |
| 0.10 | 0.893441 | 0.519614 | 0.706528 |
| 0.15 | 0.890812 | 0.519125 | 0.704969 |
| 0.20 | 0.886262 | 0.513461 | 0.699861 |
| 0.25 | 0.880960 | 0.509706 | 0.695333 |
| 0.30 | 0.874913 | 0.501195 | 0.688054 |

## All 12 preservation gates

Signed degradation; positive is worse than the q=0 baseline. `*` marks a failure. Gates are the
registered `contract.HOLDOUT_PRESERVATION_GATES`; none was invented, retuned or relaxed.

| gate | dir | bound | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 |
|---|---|---|---|---|---|---|---|---|
| `vehicle_precision` | loss | 0.010 | -0.00007 | -0.00003 | -0.00185 | -0.00272 | -0.00327 | -0.00285 |
| `vehicle_recall` | loss | 0.010 | +0.00041 | +0.00103 | +0.00155 | +0.00103 | +0.00206 | +0.00206 |
| `vehicle_f1` | loss | 0.010 | +0.00019 | +0.00054 | -0.00003 | -0.00071 | -0.00041 | -0.00022 |
| `person_avo_precision` | loss | 0.015 | +0.01884* | +0.02680* | +0.03408* | +0.04289* | +0.05371* | +0.06375* |
| `person_avo_recall` | loss | 0.015 | -0.00521 | -0.00417 | -0.00904 | -0.01425 | -0.01773 | -0.01668 |
| `person_avo_f1` | loss | 0.015 | +0.00718 | +0.01186 | +0.01347 | +0.01587* | +0.02031* | +0.02642* |
| `vehicle_xy_mae_m` | increase | 0.050 | -0.00176 | -0.00171 | -0.00145 | +0.00213 | +0.00350 | +0.00734 |
| `person_avo_xy_mae_m` | increase | 0.050 | +0.00415 | +0.00534 | +0.00685 | +0.00386 | +0.00361 | -0.00227 |
| `vehicle_iou` | loss | 0.010 | +0.00176 | +0.00557 | +0.00820 | +0.01275* | +0.01805* | +0.02410* |
| `person_box_mask_iou` | loss | 0.010 | +0.00240 | +0.00828 | +0.00877 | +0.01443* | +0.01819* | +0.02670* |
| `foreground_miou` | loss | 0.010 | +0.00208 | +0.00693 | +0.00848 | +0.01359* | +0.01812* | +0.02540* |
| `person_avo_recall_20_40m` | loss | 0.030 | -0.00745 | -0.00917 | -0.01547 | -0.02178 | -0.02579 | -0.02407 |

Ordering of failure as q grows: `person_avo_precision` (already at 0.05), then
`person_avo_f1` + `vehicle_iou` + `person_box_mask_iou` + `foreground_miou` together at 0.20.
Every degradation is monotone in q except the two XY MAE terms, which stay far inside their bounds.

## Absolute service gates

7 of 9 at every q from 0.00 to 0.30, failing the *same* two targets as the frozen q=0 baseline:
`person_precision` (0.80) and `person_recall` (0.80). The drop does not change service-gate status
anywhere in this interval; both failures are inherited from the dense model, not caused by transport.

| target | bound | 0.00 | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 |
|---|---|---|---|---|---|---|---|---|
| `vehicle_precision` | >=0.800 | 0.9316 | 0.9317 | 0.9316 | 0.9334 | 0.9343 | 0.9349 | 0.9344 |
| `vehicle_recall` | >=0.850 | 0.8684 | 0.8680 | 0.8674 | 0.8669 | 0.8674 | 0.8664 | 0.8664 |
| `person_precision` | >=0.800 | 0.7967* | 0.7799* | 0.7717* | 0.7661* | 0.7589* | 0.7509* | 0.7423* |
| `person_recall` | >=0.800 | 0.5961* | 0.6005* | 0.5997* | 0.6056* | 0.6098* | 0.6126* | 0.6116* |
| `vehicle_xy_mae_m` | <=1.000 | 0.4787 | 0.4769 | 0.4770 | 0.4772 | 0.4808 | 0.4822 | 0.4860 |
| `person_xy_mae_m` | <=1.200 | 0.8395 | 0.8414 | 0.8431 | 0.8460 | 0.8451 | 0.8470 | 0.8424 |
| `vehicle_iou` | >=0.850 | 0.8990 | 0.8972 | 0.8934 | 0.8908 | 0.8863 | 0.8810 | 0.8749 |
| `person_box_mask_iou` | >=0.500 | 0.5279 | 0.5255 | 0.5196 | 0.5191 | 0.5135 | 0.5097 | 0.5012 |
| `foreground_miou` | >=0.675 | 0.7135 | 0.7114 | 0.7065 | 0.7050 | 0.6999 | 0.6953 | 0.6881 |

`person_box_mask_iou` and `foreground_miou` are the closest to their targets at q=0.30
(0.5012 vs 0.500 and 0.6881 vs 0.675) -- both would fail before q=0.50, which Phase 6 already
measured as 4/9.

## Per-episode diagnostic

Projection of the per-episode counts the frozen AVO person scorer already returns. No extra
inference pass, no bootstrap, no new evaluation machinery. Diagnostic only: the curve and the
verdict above are the split-level numbers.

`canonical_v3_05_val_30_30_s601_tm1601` (853 observable GT)

| q | P | R | F1 | XY MAE (m) |
|---|---|---|---|---|
| 0.00 | 0.7587 | 0.6893 | 0.7224 | 0.8115 |
| 0.05 | 0.7427 | 0.6905 | 0.7157 | 0.8016 |
| 0.10 | 0.7356 | 0.6882 | 0.7111 | 0.7930 |
| 0.15 | 0.7366 | 0.6917 | 0.7134 | 0.7907 |
| 0.20 | 0.7345 | 0.6940 | 0.7137 | 0.7901 |
| 0.25 | 0.7278 | 0.6928 | 0.7099 | 0.7811 |
| 0.30 | 0.7202 | 0.6940 | 0.7069 | 0.7825 |

`canonical_v3_06_val_50_50_s602_tm1602` (2,024 observable GT)

| q | P | R | F1 | XY MAE (m) |
|---|---|---|---|---|
| 0.00 | 0.6844 | 0.7233 | 0.7033 | 0.8125 |
| 0.05 | 0.6649 | 0.7302 | 0.6960 | 0.8222 |
| 0.10 | 0.6567 | 0.7297 | 0.6913 | 0.8273 |
| 0.15 | 0.6470 | 0.7352 | 0.6883 | 0.8303 |
| 0.20 | 0.6363 | 0.7416 | 0.6849 | 0.8263 |
| 0.25 | 0.6245 | 0.7470 | 0.6803 | 0.8294 |
| 0.30 | 0.6138 | 0.7451 | 0.6731 | 0.8207 |

The precision loss is concentrated in the denser 50/50 episode (-0.0706 from q=0 to q=0.30) versus
the 30/30 episode (-0.0385), and the low-density episode's XY MAE actually improves. Two episodes
is not a population; this is a direction, not an estimate.

## Per-q transport confirmation

Every new q was served through `continuous_q.transport()` directly. `contract.snap_continuous_q`
was never called -- snapping would have served q=0 for all five settings and measured nothing.

| q | wire q | q_e4 | snapped | registered | keep count | framed bytes | bit-exact | nested |
|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.05 | 500 | false | false | 20,429 = expected | 20,922,028 (single value) | yes, 8 frames | 21,504 > 20,429 > 19,354 |
| 0.10 | 0.10 | 1000 | false | false | 19,354 = expected | 19,821,228 (single value) | yes, 8 frames | 20,429 > 19,354 > 18,278 |
| 0.15 | 0.15 | 1500 | false | false | 18,278 = expected | 18,719,404 (single value) | yes, 8 frames | 19,354 > 18,278 > 17,203 |
| 0.20 | 0.20 | 2000 | false | false | 17,203 = expected | 17,618,604 (single value) | yes, 8 frames | 18,278 > 17,203 > 16,128 |
| 0.25 | 0.25 | 2500 | false | false | 16,128 = expected | 16,517,804 (single value) | yes, 8 frames | 17,203 > 16,128 > 15,053 |

* **requested q == wire q** on all 3,345 frames of every pass (checked per frame, not per pass).
* **`snapped == false`** on every frame; the observed set of `snapped` values is exactly `{False}`.
* **Exact keep count** on every frame: the observed keep-count set is a single value equal to
  `contract.keep_count(q)`, and the framed payload length is likewise a single value per q.
* **Bit-exact encode/decode**, in the three-part sense: retained C2 values bit-identical to source,
  dropped cells decoding to exact zero, and the decoded wire bitmask equal to the selected indices.
  The dense decoded C2 is *not* claimed equal to the original dense C2 -- at q>0 it cannot be.
* **Nesting**, measured on 16 frames per pass from the same q-independent score map: the keep set at
  each q is a strict subset of its less-aggressive neighbour's and a strict superset of its
  more-aggressive neighbour's. The q=0.30 neighbour mask for the q=0.25 check was recomputed from
  those same ranker scores; q=0.30 edge inference and evaluation were **not** rerun.
  For q=0.05 the less-aggressive neighbour is the dense q=0 bypass, which retains all 21,504 cells
  by construction, so containment there is a strict cardinality bound and no ranking is run.

## Scope, provenance and what this does not establish

Bound inputs, all verified by exact sha256 before any inference:

* stable ranker `experiments/.../20260901_185725_phase5_ranker_training/checkpoints/ranker_epoch_04.pt`
  = `07781c56a4c0f306f16d332f64627ce6b9458e154f40ab9fef89f89909b79cb5` (epoch 4, distillation-only)
* frozen perception checkpoint = `da14d21edbd374c1c3abce02ca4674b9f4097becfba9759aba945cea160a297f`
* p025 forward lock = `86d6f13ae9168b33b697df5b785c5f7c320afc52cfdcded5b632d94a6d943fe1`
* hybrid-q locked config = `b2b0d8427bd867f46058ebba49ac6a183eb89413b4d69326fef93b150ebfcde6`
* reused Phase-6 curve = `54987920a7430564425664e82511d1121e77935beabfbd4cf2f34bee5cadfc74`

Frozen perception and ranker module state were snapshotted and re-verified unchanged after every
pass. Epochs 8 and 12 were never loaded.

* **Exactly the registered 3,345 validation frames** per setting, in the registered order, over the
  two registered validation episodes. Frame-id order and coverage were asserted per pass.
* **q=0 reused, not rerun.** The frozen p025 prediction set was re-scored through this same path and
  reproduced all 15 published metrics exactly, plus the 1008/1745 20-40 m split and the 7/9 service
  pass count with the identical two failures. That is what makes the q>0 rows comparable. Evaluating
  the gates on that row against itself is the identity: 12/12, zero degradation.
* **q=0.30 reused, not rerun**, verbatim from the hash-bound Phase-6 artifact.
* Test split not opened; CARLA not launched; no teacher map read (the Phase-4 shards were
  hash-verified as bound inputs only, exactly as Phase 6 did).
* No training, tuning, recalibration, checkpoint selection, threshold change, model-parameter change,
  architecture change or wire-format change. p025, AVO>=0.65, canonical evaluation, geometry and
  segmentation semantics are the frozen Phase-6 ones.
* **Any q reported as passing 12/12 would be a validation-selected engineering operating point, not
  independent unseen-test confirmation.** Here the point is moot: none does.
* Accuracy at a q not listed above is neither measured nor interpolated. The five new rows say
  nothing about q values between them.
* This phase only expands the measured accuracy anchors of the already-mechanically-ready
  continuous-q interface. q=0.90 and q=0.98 remain emergency RL actions on the Phase-6 evidence.

## Reproduce

```bash
python3 -m pole_lraspp_multimodal_fusion.object_head_pilot_v1.\
splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.low_q_validation \
  --execute HYBRID_Q_LOW_Q_VALIDATION_CURVE \
  --output experiments/splitfusion_fcos_hybrid_q_v1/<timestamp>_low_q_validation_curve
```

Settings run sequentially; each is written to `settings/q<e4>.json` with `os.replace` after its own
scoring completes, so a later failure never invalidates or reruns an earlier one. Re-invoking with
the same `--output` reuses completed settings and clears only an interrupted setting's partial
predictions. Prediction sets (297 MB) stay untracked; segmentation masks are removed after scoring.
