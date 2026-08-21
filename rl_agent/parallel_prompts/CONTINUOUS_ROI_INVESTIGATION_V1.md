# Prompt for a separate Codex session — continuous ROI investigation

Work in this repository as an isolated, offline research track. Do not edit the
current UE-A1 registry, UE-A2 runtime/launcher, controller checklist, production
map server, or existing experiment artifacts. Do not run CARLA or OAI.

## Question

Is the rank-based ROI drop fraction `q` suitable as a continuous control within
each fixed `{model family, quantizer}` branch, or should the UE controller keep
using measured discrete q anchors?

The four integrated model families are `{noae, ae32, ae64, ae128}` and the three
quantizers are `{uint8, uint6, uint4}`. The current 72-profile evidence uses six
q anchors `{0, .3, .5, .7, .9, .98}`. The trained checkpoints sampled q from
`Uniform(0, .8)`, so treat `[0, .8]` as the primary in-distribution interval.
Values `.9/.98` are measured extrapolation references, not proof that continuous
control is valid there. `q=1` is forbidden.

## Design-first gate

Before running inference, write and freeze a short plan covering:

1. exact q grid and why it is dense enough to test smoothness (start with
   existing anchors plus `.05/.10/.15/.20` and regular points through `.8`);
2. frozen validation and test roles, grouped by complete sample/trajectory;
3. metrics: feature payload bytes, vehicle/person precision and recall,
   world-XY MAE/RMSE, FP/frame, and secondary segmentation IoUs;
4. paired uncertainty method using complete frames, not raw repeated rows;
5. latency measurement for q gating and serialization;
6. criteria for `CONTINUOUS_SUPPORTED`, `DISCRETE_ONLY`, or
   `INSUFFICIENT_EVIDENCE`;
7. compute/time budget and stop rule.

GT is evaluation-only. Do not use same-frame GT or post-tail output as a
deployable policy input.

## Required analysis

- Verify the production q semantics: it discards `round(q*N)` lowest-ranked
  feature cells; it is not a score threshold. Quantify the resulting piecewise
  nature of the action.
- Evaluate q curves separately for all 12 family/quantizer branches. Do not
  assume separability or pool repeated quantizers as independent frames.
- Test monotonicity and local smoothness of payload, detection/localization,
  and segmentation outcomes. Identify crossings and discontinuities.
- Determine whether interpolation between measured q anchors predicts held-out
  q outcomes well enough for policy training.
- Explain the eventual hybrid action honestly: 12 categorical branches plus
  one bounded continuous q. Standard SAC/TD3 do not directly solve this mixed
  action without an explicit architecture or hierarchy.
- Keep object-map quality primary and segmentation secondary; do not add an
  explicit reward for selecting a particular q.

## Deliverable

Write a new timestamped, create-only experiment directory containing the
frozen plan, resolved config, per-q results, paired summaries, figures, report,
manifest with hashes, and a `REVIEW_REQUIRED` terminal. Do not promote q values,
change the 72-action registry, or implement an RL agent. End with the smallest
recommended next experiment and the evidence required before promotion.

