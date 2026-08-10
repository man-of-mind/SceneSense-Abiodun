# Track A safety calibration

**Status:** fixed-point safety characterization complete; no operating point selected.

Scope is fixed at epsilon=2.0 m, preferred core=90 KiB, range<=25 m, the same three held-out vehicle replay episodes, and channel seeds `[1101, 2202, 3303]`. CARLA, OAI, LOCAL, RL, reward-weight sensitivity, and the 3x2x2 advisor sweep were not run.

## Metric contract

- Raw shield-safe actions `{B<=epsilon}` are recorded before preferred-core/reward narrowing.
- C2 false-admit rate is conditional on selected raw-safe SPLIT schedules; counts and a descriptive 95% Wilson interval are reported.
- False-reject rate is conditional on frames where the full-GT clairvoyant raw-safe set is nonempty; the numerator is zero overlap with the deployable raw-safe set.
- C1 estimate-miss rate is conditional on actual capture attempts and remains separate from C2.
- `split_pct` is schedule selection; `capture_attempt_pct` is the actual target-FPS send rate.
- `mean_true_scored_reward` evaluates the selected action with hidden truth; `mean_predicted_reward` is the deployable model's score.
- Latency shocks are common random numbers indexed by episode and control tick, so policy-dependent capture counts do not desynchronize cells.

## Current-pilot configuration anchor

| cell_id       |   matched_false_admit_count |   admitted_send_count |   matched_false_admit_conditional_pct |   matched_false_admit_ci95_high_pct |   false_reject_count |   true_feasible_frame_count |   false_reject_conditional_pct |   split_pct |   capture_attempt_pct |   over_budget_pct |   c1_estimate_miss_count |   attempts |   c1_estimate_miss_pct_attempted |   mean_true_scored_reward |   mean_predicted_reward |   mean_prb_cost |   oracle_action_set_mismatch_pct |   shield_skip_clairvoyant_split_pct | roc_nondominated   |
|:--------------|----------------------------:|----------------------:|--------------------------------------:|------------------------------------:|---------------------:|----------------------------:|-------------------------------:|------------:|----------------------:|------------------:|-------------------------:|-----------:|---------------------------------:|--------------------------:|------------------------:|----------------:|---------------------------------:|------------------------------------:|:-------------------|
| ucb1.0__c10.7 |                           0 |                    15 |                                     0 |                             20.3883 |                  414 |                         986 |                        41.9878 |       5.827 |                4.7087 |           56.5627 |                        1 |         80 |                             1.25 |                  -6547.48 |                  0.3189 |          0.0276 |                          28.7228 |                             26.6039 | False              |

## All calibration cells

| cell_id       |   matched_false_admit_count |   admitted_send_count |   matched_false_admit_conditional_pct |   matched_false_admit_ci95_high_pct |   false_reject_count |   true_feasible_frame_count |   false_reject_conditional_pct |   split_pct |   capture_attempt_pct |   over_budget_pct |   c1_estimate_miss_count |   attempts |   c1_estimate_miss_pct_attempted |   mean_true_scored_reward |   mean_predicted_reward |   mean_prb_cost |   oracle_action_set_mismatch_pct |   shield_skip_clairvoyant_split_pct | roc_nondominated   |
|:--------------|----------------------------:|----------------------:|--------------------------------------:|------------------------------------:|---------------------:|----------------------------:|-------------------------------:|------------:|----------------------:|------------------:|-------------------------:|-----------:|---------------------------------:|--------------------------:|------------------------:|----------------:|---------------------------------:|------------------------------------:|:-------------------|
| ucb2.0__c10.6 |                           0 |                    15 |                                     0 |                             20.3883 |                  413 |                         984 |                        41.9715 |       5.827 |                4.6498 |           56.5627 |                        1 |         79 |                           1.2658 |                  -6547.49 |                  0.3151 |          0.0259 |                          28.8405 |                             26.6039 | True               |
| ucb1.0__c10.7 |                           0 |                    15 |                                     0 |                             20.3883 |                  414 |                         986 |                        41.9878 |       5.827 |                4.7087 |           56.5627 |                        1 |         80 |                           1.25   |                  -6547.48 |                  0.3189 |          0.0276 |                          28.7228 |                             26.6039 | False              |
| ucb0.0__c11.0 |                           0 |                    15 |                                     0 |                             20.3883 |                  414 |                         986 |                        41.9878 |       5.827 |                4.7087 |           56.5627 |                        1 |         80 |                           1.25   |                  -6547.48 |                  0.3189 |          0.0276 |                          28.7228 |                             26.6039 | False              |

## Interpretation guardrails

- `roc_nondominated` is descriptive over conditional C2 false admission and full-GT false rejection; it is not an automatic operating-point recommendation.
- Wilson intervals treat frames as binomial trials and are descriptive only because replay frames are temporally correlated.
- Matched/tracked C2 and strict end-to-end GT exposure remain distinct; observation coverage is not reinterpreted as shield error.
- The corpus is vehicle-only and all payload/FPS projection caveats from the pilot remain in force.
- Stop here for Abiodun/advisor review; do not launch the reward or 3x2x2 sweeps yet.

## Artifacts

Run directory: `rl_agent/policy/experiments/safety_calibration_smoke/20260810_191325`

See `per_frame_metrics.csv`, `summary.csv`, `per_replay_summary.csv`, `replay_registry.csv`, `resolved_config.yaml`, `manifest.json`, and `figures/*.{png,pdf}`.
