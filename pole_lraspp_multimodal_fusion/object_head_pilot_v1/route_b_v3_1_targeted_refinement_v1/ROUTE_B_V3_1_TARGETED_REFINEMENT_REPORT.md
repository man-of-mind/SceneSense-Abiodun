# Route B v3.1 targeted LR-ASPP refinement — audit + one class-balanced continuation

Terminal: **`LRASPP_TARGETED_REFINEMENT_NO_GAIN`**

Experiment: `experiments/route_b_v3_1_targeted_refinement_v1/20260828_032520/`
Baseline: `experiments/route_b_v3_1_clean_base_v1/20260828_012309/` epoch 20,
SHA-256 `88b34a69eeec7bf2f6444e70a0e346c365b979e6936d277cb0c75e8cd747aa1d` (verified read-only).

All numbers below are contract `v010`, match radius 3.0 m, range 40 m, minimum GT area 12 px,
fixed score points 0.20 and 0.02. `v025` was computed as sensitivity only and selected nothing.
Predictions centred in registered ignore regions stay neutral and enter neither TP, FP nor FN.

## Phase A — retained-prediction audit

No inference. Matching semantics reproduced from the frozen scorer and reconciled exactly.

| class @ score | published TP / FP / FN | recomputed | ignored-neutral | exact |
|---|---|---|---|---|
| vehicle @0.20 | 6849 / 7390 / 2876 | 6849 / 7390 / 2876 | 2744 | yes |
| person @0.20 | 1560 / 1822 / 2312 | 1560 / 1822 / 2312 | 925 | yes |
| vehicle @0.02 | 7256 / 10611 / 2469 | 7256 / 10611 / 2469 | 3841 | yes |
| person @0.02 | 1766 / 5651 / 2106 | 1766 / 5651 / 2106 | 1892 | yes |

Audit gates: exact reconciliation, no missing/non-finite prediction fields, checkpoint and
prediction hashes verified, ignored predictions confirmed neutral, and both label sets summing
exactly to their denominators — **all pass**.

### Vehicle FP decomposition @0.20 (denominator 7390)

Labels are priority-ordered and mutually exclusive: duplicate first, then 2D-correct/world-wrong,
then background. These GT-referenced labels are diagnostic only.

| label | count | share |
|---|---:|---:|
| `PREDICTED_DUPLICATE` | 3580 | 48.44% |
| `TWO_D_CORRECT_WORLD_WRONG` | 2549 | 34.49% |
| `BACKGROUND_OR_OTHER` | 1261 | 17.06% |
| `IGNORE_NEUTRAL` (excluded from FP) | 2744 | — |

Unprioritised overlap (before priority ordering): 3580 duplicate-any, 4337 two-D-any, 1788 both.

### Person FN decomposition @0.02 (denominator 2106)

| label | count | share |
|---|---:|---:|
| `HEATMAP_CENTER_MISS` | 1467 | 69.66% |
| `CENTER_PRESENT_WORLD_WRONG` | 534 | 25.36% |
| `MATCHING_CONTENTION` | 105 | 4.99% |

`HEATMAP_CENTER_MISS` at 69.66% clears the registered 50% bar, so the class-balanced centre loss
is the licensed mechanism: most person misses are the heatmap never firing, not a localisation or
matching artefact.

### Vehicle postprocessor — `VEHICLE_WORLD_NMS_2M_REJECTED`

One predicted-only candidate, no sweep: vehicle-only world NMS at 2.0 m, keep highest score,
person suppression disabled, threshold 0.20. Suppression reads only `class_name`, `score`,
`world_x`, `world_y`.

| metric | baseline | + 2 m NMS | gate | result |
|---|---:|---:|---|---|
| vehicle precision | 0.4810 | 0.7654 | ≥ +0.05 | pass (+0.2844) |
| vehicle recall | 0.7043 | 0.6790 | ≥ -0.01 | **fail (-0.0253)** |
| person TP/FP/FN | 1560/1822/2312 | 1560/1822/2312 | unchanged | pass |

The duplicate mechanism is real and large — NMS removes 5366 of 7390 vehicle FPs — but it also
consumes 246 true positives, so the registered recall guard rejects it. No radius or threshold
retune was permitted, so Phase C ran the `RAW_FIXED_DECODER` arm only.

## Phase B — one 12-epoch clean-q continuation

Warm start epoch 20 (verified). Sole training change:
`loss_weights.object.class_balanced_center = true`, `pos_weight_enable = false`.
Architecture, optimizer, augmentation, BN policy, target geometry and ignore semantics unchanged;
`MultiTaskFusionLRASPP` / MobileNetV3 LR-ASPP, 7-channel input, shared object head, existing
`encode_front` / `decode_tail` split. q=0, feature dropping off, no AE, AdamW, LR 3e-5, weight
decay 1e-4, cosine with one warm-up epoch, batch 16, workers 8, seed 20260825, 12 epochs exactly,
checkpoints 4/8/12 create-only.

Launch check `PASS`: loss 5.1741 finite; gradient norms backbone 16.641, segmentation classifier
0.0263, object head 35.731, all finite and nonzero; warm start loaded 20/20 tensors; AMP autocast
`cache_enabled=False`. The class-balanced path is proven live rather than merely configured —
with `pos_weight_enable=false` the per-class statistics are emitted only on the macro-average
branch, and it reported `pos_classes_present=2.0` with 53 vehicle and 19 person positive cells
(2.79:1, consistent with the measured 2.68:1 imbalance).

## Phase C — fixed validation evaluation

| epoch | vehicle F1 | person F1 | mean F1 | veh P | veh R | per P | per R | veh R@.02 | per R@.02 | veh XY | per XY | fg mIoU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base 20 | 0.5716 | 0.4301 | 0.5009 | 0.4810 | 0.7043 | 0.4613 | 0.4029 | 0.7461 | 0.4561 | 0.998 | 1.409 | 0.6514 |
| 4 | 0.5717 | 0.4118 | 0.4917 | 0.4946 | 0.6772 | 0.3831 | 0.4450 | 0.7300 | 0.4959 | 0.986 | 1.400 | 0.6542 |
| 8 | 0.5757 | 0.4173 | 0.4965 | 0.4978 | 0.6825 | 0.3903 | 0.4483 | 0.7374 | 0.5018 | 0.982 | 1.387 | 0.6554 |
| 12 | 0.5787 | 0.4147 | 0.4967 | 0.5015 | 0.6841 | 0.3848 | 0.4496 | 0.7386 | 0.5049 | 0.986 | 1.372 | 0.6557 |

No checkpoint or arm passed the material-gain contract, so **no candidate was selected**.
Every epoch failed the same two non-regression guards (`person_f1_delta ≥ -0.005`,
`vehicle_recall_delta ≥ -0.01`) and three of the four material-gain gates.

Best-ranked candidate (epoch 12, `RAW_FIXED_DECODER`,
SHA-256 `44843543126d1b95…`) versus baseline:

| metric | delta | note |
|---|---:|---|
| person recall @0.02 | +0.0488 | below the +0.05 material bar |
| person recall @0.20 | +0.0467 | |
| person precision | -0.0764 | offsets the recall gain |
| person F1 | -0.0154 | guard breach (≥ -0.005) |
| vehicle recall | -0.0202 | guard breach (≥ -0.01) |
| vehicle F1 | +0.0071 | |
| mean class F1 | -0.0041 | |
| person XY MAE | -0.0367 m | improvement |
| vehicle XY MAE | -0.0120 m | improvement |
| foreground mIoU | +0.0044 | |

**Interpretation.** The class-balanced centre loss did exactly what the audit predicted
mechanically: it raised person heatmap firing, converting `HEATMAP_CENTER_MISS` into detections
(+0.0488 recall@0.02) and improving both XY MAE and foreground mIoU. But the extra person centres
are largely low-precision, so person precision drops 0.0764 and person F1 falls. It also costs
vehicle recall. Under the registered contract this is a redistribution along the
precision/recall curve, not a gain.

## Clean-service targets (reported separately, unchanged)

Evaluated on the best-ranked epoch-12 candidate; the baseline fails the same set.

| target | required | epoch 12 | pass |
|---|---:|---:|---|
| vehicle precision | ≥0.80 | 0.5015 | no |
| vehicle recall | ≥0.85 | 0.6841 | no |
| person precision | ≥0.80 | 0.3848 | no |
| person recall | ≥0.80 | 0.4496 | no |
| vehicle XY MAE | ≤1.0 m | 0.986 | yes |
| person XY MAE | ≤1.2 m | 1.372 | no |
| vehicle IoU | ≥0.85 | 0.8657 | yes |
| person box-mask IoU | ≥0.50 | 0.4458 | no |
| foreground mIoU | ≥0.675 | 0.6557 | no |

## Provenance

- Audit 3.1 s; training 1430.1 s (12 epochs, ~119 s/epoch); Phase C 400.8 s including three
  inference passes. Session wall time ~41 min against a 90 min box.
- VRAM: training peak 5819.0 MiB allocated / 7680.0 MiB reserved; inference 110.4 / 154.0 MiB.
- v3/v3.1 dataset, contracts and payload reused by symlink — nothing rebuilt or copied.
- Test split never opened; no q/AE training, threshold sweep, CARLA, OAI or container work.
- Production decoding untouched; the 2 m NMS exists only as a rejected evaluation-arm candidate.
