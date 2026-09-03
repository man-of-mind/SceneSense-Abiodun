# Train-only distance failure audit — frozen person p025 path

One bounded diagnostic answering a single question: do visible 30-40 m pedestrians
already exist among the raw FCOS person candidates, and which stage removes them?

No training, no cache rebuild, no model forward pass, no CUDA, and no validation or
test access. Source: `distance_failure_audit.py`. Full evidence:
`experiments/splitfusion_fcos_person_distance_failure_audit_v1/`. Compact companion:
`DISTANCE_FAILURE_AUDIT_V1.json`.

## Answer

**Yes — the candidates are there, and the semantic-support filter removes them.** Among
the 663 observable 30-40 m GT actor-frames, 658 (99.25%) have a raw post-NMS person
candidate within the 3 m match radius, so the detector is not blind at range. Of the 202
reachable GT lost across the pipeline, **178 (88.1%) are lost at the semantic-support
filter** (`semantic_support >= 0.10`), against 19 at the 0.20 score filter, 1 at instance
grouping, and 4 at the final p025 threshold. In the 30-35 m band the support filter is
responsible for 70 of 71 losses (98.6%).

## Inputs and validity

- Train-holdout only: the two registered episodes, 3,284 frames, 222,096 cached person
  candidates, 2,556 observable GT actor-frames at AVO >= 0.65.
- Every frozen input verified against its registered SHA-256 (consolidation feasibility
  result, cache manifest, cache shard hash-map, training support records, training
  reference JSON, the 4,703-row holdout AVO table), plus exact re-verification of the raw
  holdout metadata hashes against the frozen p025 `INPUT_HASHES.json`.
- The AVO table joins to the raw GT on the exact `(episode_id, sample_id, gt_actor_id)`
  identity with bitwise agreement on `world_x`, `world_y`, and `distance_m`, and every
  AVO sample is present in the candidate cache.
- **Stages 4 and 5 reproduce the frozen p020 and p025 train-holdout views exactly** —
  observable GT, TP, FP, FN, ignored predictions, precision, recall, and XY MAE, plus the
  3,460 / 3,217 retained-output counts. All four reproduction gates pass, which is what
  makes the staged decomposition trustworthy.
- Stages 2-4 are expressed as preregistered consolidation-grid configurations and
  evaluated by the frozen `consolidate_person_candidates`, so no selection logic is
  re-implemented. Stage nesting (S5 ⊆ S4 ⊆ S3 ⊆ S2 ⊆ S1) is asserted per frame.

## Conventions

- Recall bands GT by `gt_distance_m` (3D camera-origin radial distance). Precision bands
  predictions by predicted radial distance, measured on the world plane from
  `(camera_x, camera_y)` because the cache stores only `world_xyz[:, :2]` for candidates.
  Over the 4,703 eligible GT rows the plane-vs-3D convention gap is at most 0.1940 m
  (mean 0.0067 m). Per-band F1 mixes the two assignments; the two TP counts differ per
  band and are equal in total.
- Matching order preserved: observable GT first, then AVO-ignore, then structural-ignore,
  greedy by ascending world distance inside 3.0 m.
- `candidate recall ceiling` = share of eligible GT with at least one surviving candidate
  in radius. `max_matching_ceiling` is the same under one maximum bipartite matching, so
  it is the attainable recall of a perfect downstream selector over that candidate set.

## Long-range recall by stage

Observable-GT recall and candidate recall ceiling (in parentheses):

| stage | retained preds | 20-30 m | 30-35 m | 35-40 m | overall |
|---|---:|---:|---:|---:|---:|
| S1 raw FCOS (post-NMS) | 222,096 | 0.9990 (1.0000) | 1.0000 (1.0000) | 0.9844 (0.9844) | 0.9977 (0.9980) |
| S2 score >= 0.20 | 50,616 | 0.9969 (1.0000) | 0.9942 (0.9971) | 0.9281 (0.9281) | 0.9890 (0.9906) |
| S3 semantic support >= 0.10 | 7,330 | 0.9271 (0.9302) | 0.7901 (0.7930) | 0.5844 (0.5906) | 0.8893 (0.8916) |
| S4 grouping IoU >= 0.20 | 3,460 | 0.9148 (0.9302) | 0.7843 (0.7930) | 0.5781 (0.5875) | 0.8818 (0.8908) |
| S5 p025 >= 0.25 | 3,217 | 0.9138 (0.9292) | 0.7843 (0.7930) | 0.5656 (0.5750) | 0.8799 (0.8889) |

Corresponding per-band precision at S5: 0.9243 (20-30 m), 0.8659 (30-35 m), 0.8177
(35-40 m); overall 0.8989 with XY MAE 0.5347 m. At S2 the same bands sit at 0.0513,
0.0401, and 0.0412, so the support filter is buying most of the operating precision.

## Reachable GT lost per stage transition, 30-40 m

| transition | 30-35 m | 35-40 m | 30-40 m union | share of union losses |
|---|---:|---:|---:|---:|
| S1 → S2 (score >= 0.20) | 1 | 18 | 19 | 9.4% |
| S2 → S3 (support >= 0.10) | 70 | 108 | 178 | 88.1% |
| S3 → S4 (grouping) | 0 | 1 | 1 | 0.5% |
| S4 → S5 (p025) | 0 | 4 | 4 | 2.0% |

## Is a range-aware post-processing attempt worth attempting?

Two different answers, and the distinction matters:

- **Post-processing the p025 output set: no.** The S5 candidate recall ceiling in 30-40 m
  is 0.6878 against a realized 0.6787 — about 0.9 recall points of headroom. Rescoring,
  reordering, or range-conditioned thresholding *after* consolidation cannot recover the
  long-range misses, because the candidates are already gone.
- **Revisiting the semantic-support gate: yes, that is where the recall lives.** Relative
  to the raw ceiling the recoverable headroom is +21.6 recall points in 30-35 m and
  +41.9 in 35-40 m (+31.4 for the union, 208 actor-frames). The raw-candidate diagnostic
  localizes the mechanism: in 35-40 m, 297 of 320 observable GT have a raw candidate at
  score >= 0.20, but only 189 have one that also clears `semantic_support >= 0.10` — the
  person segmentation mask, not the detector score, is the binding constraint at range.
  This is consistent with the standing density+segmentation finding.

That headroom is a ceiling, not a projection: the S2 precision numbers above show a naive
relaxation would be very expensive, so any range-aware rule has to buy support-independent
evidence rather than simply drop the gate.

## Scope limits

- Stage 1 is the earliest cached stage: post-NMS candidates (head score > 0.02, per-level
  top-1000, class-wise NMS at IoU 0.60, top-100 detections per image). Pre-NMS candidates
  are not cached, so loss inside NMS or the top-100 cap is upstream of this audit and is
  not measured. The measured 0.9925 raw ceiling is therefore a lower bound on what the
  head actually produced.
- Fit-side episodes were **not** audited. A complete, hash-registered AVO table exists only
  for the two holdout episodes; the train reference covers all ten episodes but is a strict
  subset of the canonically eligible actor-frames (112 of 4,703 holdout rows were absent
  from it), and closing that gap would mean regenerating AVO records, which this phase
  forbids. The holdout support (663 observable 30-40 m actor-frames) is ample for the
  question asked.
- Train-holdout numbers only. No validation or test claim, and no policy was tuned.

PERSON_P025_DISTANCE_FAILURE_AUDIT_COMPLETE
