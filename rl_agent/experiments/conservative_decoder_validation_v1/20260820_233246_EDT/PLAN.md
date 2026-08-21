# Conservative predicted-world decoder validation plan

Audit: `conservative_decoder_validation_v1/20260820_233246_EDT`  
Status at freeze: preregistered before validation inference

## Scope

This is an offline, validation-first perception-quality study. It may read the
original dataset and existing checkpoints. It writes only inside this new
experiment directory. It does not run CARLA or OAI, train a model, edit a
checkpoint, or touch any UE/production component.

The deployable operation is greedy, score-ordered, class-aware predicted-world
suppression. A prediction is suppressed when a higher-scoring retained
prediction of the same predicted class lies within the frozen radius. The
operation consumes only predicted class, score, and predicted world XY. GT is
loaded only after prediction generation for offline one-to-one scoring.

## Frozen inputs and integrity gate

Before validation inference, `freeze_inputs.py` must:

1. extract the manifest's original train/val/test sample identifiers;
2. assert counts `10911/2110/2162`, uniqueness, and zero pairwise overlap;
3. assert exact set and order agreement with every historical checkpoint's
   saved split files;
4. save the three identifier lists and their SHA-256 hashes;
5. hash checkpoints, trial summaries, evaluator, decoder, split runtime,
   training/data configuration, dataset manifests, the prior audit, the
   intended-catalog evidence, and all preregistered scripts/configuration here;
6. record nested-repository commit/dirty state and the compute environment.

Any failed assertion ends the study without inference.

## Profiles and candidates

Five profiles span a no-AE reference, two compact normal profiles, one
quality-oriented normal profile, and one high-q degraded-rescue diagnostic.
Only the three rows marked `normal_gate=true` decide normal-service eligibility;
the reference and degraded-rescue rows remain visible in every output.

All candidates keep score thresholds at `.20`, image NMS at `2 px`, and top-k at
`120`. Validation evaluates exactly `baseline`, `world_suppression_1m`, and
`world_suppression_2m`. No threshold, per-profile, per-scene, or test tuning is
permitted.

## Metrics and uncertainty

For each split/profile/candidate/class, report TP/FP/FN, precision, recall, F1,
prediction count, GT count, XY MAE/RMSE, and FP/frame. Segmentation confusion and
payload are captured per profile and explicitly versioned as decoder-invariant.

Paired deltas versus baseline are bootstrapped two ways with 2,000 fixed-seed
replicates: by individual `sample_id`, and by trajectory scenario block
(`experiment_id` by one of the declared eight collection loops). Profile and
pooled-normal estimates retain candidate pairing.

GPU decoder timing and incremental retained-list timing are different evidence:
the former is CUDA-synchronized feature-to-list latency; the latter times only
the candidate's CPU list operation. Raw paired samples and percentile summaries
must both be retained.

## Frozen floors, selection, and no-retraining rule

The exact floors, material-effect thresholds, uncertainty requirements, and
selection rule are machine-readable in `resolved_config.json`. In brief, every
normal profile must pass every absolute floor. Both frame- and scenario-paired
evidence must show vehicle/person precision improvement and FP/frame reduction,
with minimum point effects of +.02, +.01, and -.10 respectively.

The conservative order is 1 m first, then 2 m only if 1 m is ineligible. At
most one setting is frozen. Only that setting and baseline may be scored on the
untouched test split; test results cannot change the setting.

If no conservative setting qualifies with complete evidence, the report may
justify a bounded AE64 retraining pilot, but this experiment will not train it.

## Outputs and terminal conclusions

Required artifacts are provenance and split-integrity JSON, frozen identifiers,
raw per-frame predictions, per-profile/class metrics, paired per-frame results,
frame/scenario bootstrap intervals, latency samples/summary, frozen selection,
manifest, report, and `REVIEW_REQUIRED`.

The report returns exactly one of:

- `CONSERVATIVE_POSTPROCESSING_READY_FOR_PROMOTION_REVIEW`;
- `RETRAINING_PILOT_JUSTIFIED`; or
- `INSUFFICIENT_EVIDENCE`.

No conclusion is deployment approval. If promotion is later authorized, every
detection/localization catalog row affected by the global decoder must be
regenerated. Payload and segmentation may remain separately versioned only
where their decoder invariance is demonstrated.
