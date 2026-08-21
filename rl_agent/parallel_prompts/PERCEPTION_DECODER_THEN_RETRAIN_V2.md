# Prompt for a separate Codex session — decoder validation before retraining

Work on the machine that contains the original training/validation dataset.
This is a parallel offline perception-quality track. Do not edit the current
UE-A1/UE-A2 registry, runtime, launcher, controller, production map server, or
existing checkpoints. Do not run CARLA or OAI, and never overwrite `best.pt`.

## Starting evidence

The prior offline audit concluded `POSTPROCESSING_SUFFICIENT` at mechanism
level: about 96.1% of vehicle false positives were same-class duplicate
predictions near an already matched object. The selected 3 m predicted-world
suppression plus vehicle threshold .25 raised vehicle precision strongly but
failed the current normal-service recall/localization floors. It is not
deployment-approved. Less aggressive 1 m and 2 m predicted-world suppression
remain the candidates.

## Phase 1 — validation-only decoder study

Before running anything, hash and freeze:

- the original 2,110 validation identifiers and prove train/validation/test
  disjointness;
- checkpoint, evaluator, decoder, and data manifests;
- candidate post-processing definitions;
- class-aware metrics and paired frame/scenario uncertainty;
- normal experimental floors: vehicle recall >= .90, pedestrian recall >= .85,
  vehicle precision >= .49, pedestrian precision >= .61, vehicle XY MAE <=
  .90 m, pedestrian XY MAE <= 1.20 m, FP/frame <= 1.45;
- a no-retraining decision rule.

Evaluate baseline, 1 m, and 2 m class-aware predicted-world suppression with
the score thresholds held at .20 first. GT may be used only for offline scoring;
the deployable suppression must use predicted class, score, and world XY only.
Use the current evidence decoder envelope (score .20, image NMS radius 2 px,
top-k 120) and report incremental list-level latency separately from end-to-end
GPU decoder latency.

Start with representative action profiles spanning the intended catalog,
including compact/quality/high-q cases if their predictions can be regenerated.
Choose at most one global decoder setting on validation; do not tune per test
profile or per scene. Evaluate that frozen setting once on the untouched test
split. If promoted later, all detection/localization catalog rows affected by
the decoder must be regenerated; payload and segmentation evidence may remain
separately versioned where mathematically unchanged.

## Phase 2 — retrain only if necessary

Do not retrain if a conservative decoder setting materially improves precision
while preserving the frozen recall/localization floors with paired evidence.
If it cannot, propose a bounded pilot on one central family (AE64 is a sensible
candidate), with a service-aware selection score. Promotion requires precision
superiority and non-inferiority—not strict improvement of every scalar—for
vehicle/person recall, world XY, secondary segmentation, payload, and compute.
Save any new model as a versioned candidate with checkpoint/config/training-code
hashes; never replace v1.

## Deliverable

Create a new timestamped, immutable experiment directory with plan, provenance,
per-profile/class metrics, paired per-frame results, latency, report, manifest,
and `REVIEW_REQUIRED`. Return one conclusion:

- `CONSERVATIVE_POSTPROCESSING_READY_FOR_PROMOTION_REVIEW`;
- `RETRAINING_PILOT_JUSTIFIED`; or
- `INSUFFICIENT_EVIDENCE`.

Do not make production edits or claim deployment approval.
