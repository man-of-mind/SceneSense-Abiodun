# Route B visibility-aware ≤40 m evaluation and match-radius parity audit (W10275)

Read-only audit over retained artifacts. No training, no inference, no CARLA/OAI, no test-split
access, no decoder or threshold changes, no edits to the canonical evaluator.

Reusable script: `visibility_audit_v1.py`.
Concise result: `visibility_and_match_radius_audit_v1.json` (this directory).
Full artifact including region/distance/size slices:
`experiments/route_b_noae_precision_full_v1/20260825_195301/decision/visibility_and_match_radius_audit_v1.json`
(under the gitignored `experiments/` tree).

Central question: why Route B recall is ~0.45 vehicle / ~0.38 person (Stage-2 selected) against a
historical M-prime figure of ~0.89 / ~0.85.

---

## 1. Visibility — terminal `TRUE_OCCLUSION_UNRESOLVED_FROM_RETAINED_FIELDS`

Three eligibility rules were applied to every GT object within 40 m: **A** the existing
`object_targets.valid_localization_objects` rule, **B** camera-frustum-visible, **C** camera-visible
OR recorded radar support. All three produced **identical** metrics, with an empty IGNORE
population, so the specified ignore semantics had nothing to act on.

**That null result must not be read as proof of observability.**

- **Rule B measures geometric camera-frustum inclusion only** — positive camera depth plus a
  projected box with support inside the image. It is not a test of actual visibility and not a
  test of occlusion.
- **The stored boxes were already clipped to the image during collection.** The measured zero
  truncation, and the A == B equality, are therefore partly consequences of the stored schema
  rather than independent evidence that no target is truncated or occluded in the world.
- **Positive depth, projected box support and recorded radar support do not prove that an actor
  was visually unobstructed.** No retained per-object field distinguishes a clearly observed
  actor from one hidden behind another object.

Measured facts (unchanged, Route B val):

| Quantity | Value |
|---|---|
| target-class actor GT within 40 m | 13,545 (8,908 vehicle / 4,637 person) |
| GT beyond 40 m (OUT_OF_RANGE) | 50,833 (79.0%) |
| IGNORE population under A / B / C | 0 |
| within-40 m GT with `radar_support_points == 0` | 13.1% |
| parity with canonical evaluator | `tp+fn` = 13,545 for all three checkpoints |

**True occlusion remains unresolved from the retained fields.**

## 2. Match-radius parity — verified from the saved command

[`rl_agent/ae_integrated/run_permodel_sweep.sh:31-32`](../../../rl_agent/ae_integrated/run_permodel_sweep.sh)
produced the file holding the historical vehicle 0.8926 / person 0.8532:

```
--object-score-threshold 0.20 --object-nms-radius-px 2 --topk-objects 120 \
--match-distance-m 5.0 --max-gt-distance-m 40
```

Score threshold 0.20, NMS 2 px, top-k 120, 40 m range gate, `min_gt_area_px` 12.0, classes and the
prediction range gate are **identical** to Route B. The sole registered difference is the match
radius: **5.0 m historically, 3.0 m on Route B**. Note that `configs/fusion_full_run.yaml` itself
records `match_distance_m: 3.0` — the historical 0.9 came from a command-line override.

## 3. Results at 3 m and 5 m (preserved unchanged)

Same retained predictions, same eligible GT, only `match_distance_m` varying.

**Stage-2 selected (epoch 13)**

| class | R | precision | recall | F1 | TP | FP | FN | XY MAE (m) |
|---|---|---|---|---|---|---|---|---|
| vehicle | 3.0 m | 0.4624 | 0.4498 | 0.4560 | 4007 | 4658 | 4901 | 1.1343 |
| vehicle | 5.0 m | 0.5415 | 0.5267 | 0.5340 | 4692 | 3973 | 4216 | 1.5452 |
| person | 3.0 m | 0.3480 | 0.3752 | 0.3611 | 1740 | 3260 | 2897 | 1.3195 |
| person | 5.0 m | 0.4220 | 0.4550 | 0.4379 | 2110 | 2890 | 2527 | 1.7643 |

Matches newly accepted in the 3–5 m band: vehicle **685** (+0.0769 recall), person **370**
(+0.0798). Frozen noAE: 850 / 349. Stage-1: 738 / 385.

**Recall vs match radius (Stage-2)** — still climbing at 5 m, no plateau:

| class | 1 m | 2 m | 3 m | 4 m | 5 m |
|---|---|---|---|---|---|
| vehicle | 0.2292 | 0.3882 | 0.4498 | 0.4894 | 0.5267 |
| person | 0.1551 | 0.2935 | 0.3752 | 0.4220 | 0.4550 |
| overall | 0.2038 | 0.3558 | 0.4243 | 0.4664 | 0.5022 |

Detections recovered only by the 5 m gate are localization errors of 3–5 m. This is an
**EVALUATION_TOLERANCE_EFFECT, not a model-quality recovery**. Precision also rises at 5 m purely
because previously unmatched predictions become matches — the same artifact, not a precision gain.

## 4. Gap attribution

- **~17% of the historical→Route B recall gap is explained by 3 m → 5 m** (vehicle 17.4%,
  person 16.7%, overall 17.2%) — evaluation tolerance.
- **Approximately 83% remains a genuine model/input/domain gap.**
- The failure is **strongly associated with small, distant and dense Route B objects**:
  Stage-2 at 3 m gives recall 0.8164 vehicle / 0.7694 person on 64–128 px objects but
  0.2352 / 0.2014 on 16–32 px; 0.6681 / 0.6127 at 10–20 m but 0.2302 / 0.1659 at 30–40 m.
  Route B val carries 3.78 GT/frame against 1.80 historically.
- **Causality cannot be assigned solely to corpus composition, because historical per-object GT is
  unavailable.** The M-prime corpus GT table
  (`fusion_training_data/moving_ego_pps200000_merged_8loops_stride2`) is not retained — only its
  2,162 split ids survive. The association above is therefore a Route B-internal observation, not a
  measured comparison of the two corpora.

The historical corpus was collected with a **moving ego vehicle using RGB + radar**, per its
retained split identifiers. It was not a pole-mounted-camera domain.

## 5. Operational conclusion

**Neither the 3 m Route B result nor the historical-parity 5 m result meets the new full-map
quality objective.** Stage-2 selected reaches 0.4498 / 0.3752 at 3 m and 0.5267 / 0.4550 at 5 m,
against the historical reference of 0.8926 / 0.8532. Re-scoring at the historical tolerance does
not close the gap, and adopting 5 m would only relabel 3–5 m localization errors as successes.

No threshold was selected and the canonical evaluator is unchanged.
