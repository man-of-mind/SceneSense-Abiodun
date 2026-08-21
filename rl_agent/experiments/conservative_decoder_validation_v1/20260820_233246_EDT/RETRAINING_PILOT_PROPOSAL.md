# Bounded AE64 retraining pilot proposal (not started)

This proposal is review-only. It creates no checkpoint and authorizes no training.

## Bounds

- Central family: AE64 only; initialize from the frozen v1 AE64 checkpoint.
- One preregistered training configuration, at most three fixed seeds, the existing
  train/validation split, and one final untouched-test evaluation after seed/config
  selection on validation.
- Candidate outputs must use a new versioned directory and filename; `best.pt` and
  every v1 checkpoint remain read-only.
- Freeze checkpoint, config, training/evaluator/decoder source, dependency, split,
  and dataset-manifest hashes for each candidate.

## Service-aware selection and promotion rule

First apply a feasibility filter. Vehicle and person precision must be superior
to AE64-v1 with paired 95% lower bounds above zero. Vehicle/person recall, each
class's world-XY MAE, secondary segmentation, payload, and compute must be
non-inferior within preregistered margins; no requirement says every scalar must
strictly improve. Suggested validation margins are recall delta >= -0.01,
XY-MAE delta <= +0.05 m, mIoU delta >= -0.01, payload P95 <= +2%, GPU decoder
P95 <= +5%, and total inference P95 <= +5%.

Among feasible candidates only, rank by the frozen service score:
`0.35*vehicle_precision_gain + 0.20*person_precision_gain +
0.15*minimum_recall_margin + 0.15*minimum_XY_margin +
0.05*segmentation_margin + 0.05*payload_margin + 0.05*compute_margin`,
with each term normalized by its preregistered margin. Use seed-stability and
lower compute as tie-breaks. A later promotion still requires human review and
regeneration of affected detection/localization catalog rows.
