# Task A v1 — argmax-stability / rank-reversal preregistration

**Frozen before execution:** 2026-08-14. This is an offline analysis of existing artifacts. It does not run
CARLA, OAI, model inference, controller replay, or RL.

## Question and claim boundary

Holding the payload-feasible profile set fixed, does the reward-v5 task-utility maximizer change with scene
context, and does a context-keyed lookup improve held-out utility by a practically useful amount?

This screen tests whether the current context-free profile-quality table hides a scene-conditioned opportunity.
It does not test sequential control and cannot by itself justify RL. A positive result unlocks the already-defined
global/contextual/clairvoyant lookup ladder; the simplest lookup that captures the gain remains preferred.

## Frozen inputs

- The 36 published zstd profiles: 4 AE choices x 3 quantizers x ROI q in {0, 0.3, 0.5}.
- The intersection of sample IDs present in all 36 per-object evaluation CSVs. The expected count is exactly
  1,683; fail closed otherwise.
- Per-frame detection counts and 3x3 segmentation confusion matrices in
  `rl_agent/density_knob/raw/perframe_*.csv`.
- Context is derived once from the full-quality reference profile `noae__uint8__roi0.0` and the frozen
  `frame_density.csv`; candidate-profile outcomes never define their own context.
- Static payload costs are the published zstd knob-matrix payloads. Per-frame content-dependent byte variation is
  deliberately not used to change the feasible set, because budget-driven changes must be conditioned out.

The checked-in per-frame confusion matrices mean the registered segmentation re-evaluation is already available.
The incremental GPU cost for Task A is therefore zero. For provenance, a clean regeneration of all 72 density
profiles took about 78 minutes on L10319; regenerating only the registered 36 profiles is estimated at 35–45 GPU
minutes plus about one minute for gates and analysis. This estimate is reported even if no rerun is needed.

## Utility

For each frame and profile, compute reward-v5 task utility from per-frame measurements:

`U = weighted mean of {mIoU/0.840, ped_recall/0.887, vehicle_recall/0.927}`

with weights 0.35/0.40/0.25. Segmentation is always active. A class-recall term is active only when that class is
present in GT; active weights are renormalized. This makes the class-presence hypothesis testable without assigning
an arbitrary recall to an absent class. A detection-only companion result is diagnostic; the seg-inclusive result
is primary.

## Budgets, contexts, and grouped split

- Evaluate every unique profile-payload breakpoint. At each breakpoint, all compared context cells see exactly the
  same feasible set `{p: payload(p) <= budget}`.
- Primary context: `class_mix` in {empty, vehicle_only, pedestrian_only, mixed}.
- Secondary confirmatory context: nearest-object range in {empty, <=12 m, 12–25 m, >25 m}.
- Supporting diagnostics: vulnerable-object presence, reference low-confidence/miss state, COCO-style small-object
  presence (GT box area < 32^2 pixels), and image-edge truncation. Confidence/miss is outcome-derived and is not
  claimed deployable. Edge truncation is not called occlusion: the artifacts contain no true occlusion label.
- Density is excluded by design; it was already falsified as a useful seg-aware selector.
- The dataset names declare eight loops per low/medium/crowded acquisition. Each regime is divided into its eight
  ordered acquisition windows. Even windows are discovery; odd windows are confirmation. No frame from a window
  appears in both partitions. Bootstrap units are these trajectory windows, never individual frames.

## Frozen positive gate and asymmetric interpretation

For each budget and context family, fit the global winner and context lookup on discovery windows, then evaluate
both unchanged on confirmation windows. A confirmed practical reversal requires all of:

1. the lookup changes profile on at least 5% of confirmation frames;
2. mean held-out normalized utility lift is at least +0.010 absolute;
3. the trajectory-window bootstrap 95% CI lower bound is >0; and
4. the one-sided cluster-bootstrap p-value survives Holm correction across tested budgets in that family at 0.05.

The primary verdict is `POSITIVE_CONTEXTUAL_OPPORTUNITY` if class mix or nearest range passes. Supporting-only
axes cannot independently unlock the ladder.

If no primary family passes, the verdict is `NO_PRACTICAL_REVERSAL_ON_AVAILABLE_CONTEXTS`. Because per-frame
segmentation is present, this is no longer the previously anticipated detection-only null. It still scopes only to
the available contexts and 36 measured profiles; missing true occlusion labels, cyclists, and broader scenes remain
limitations. If the segmentation columns fail validation, fall back to detection-only and force the verdict to
`INCONCLUSIVE_DETECTION_ONLY`, regardless of apparent null results.

