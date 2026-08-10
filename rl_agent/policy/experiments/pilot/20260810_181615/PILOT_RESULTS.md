# Track A policy results — gated one-configuration pilot

**Scope:** table-driven SPLIT+SKIP only; epsilon=2.0 m; 90 KiB preferred core; objects within 25 m; real CARLA vehicle replay composed with a synthetic Markov channel. No CARLA, OAI, LOCAL, or RL run.

## Gate status

- Canonical seven-profile / 36-action catalog: PASS
- Contract tests: PASS
- Four deterministic acceptance episodes: PASS
- One-config real-replay pilot: COMPLETE
- Twelve-condition advisor sweep: NOT STARTED (still gated on review of this pilot)

## Pre-sweep verdict

The implementation and pilot gates pass, but the 12-condition advisor sweep remains intentionally paused until the pre-registered weight sensitivity and metric-scope review are complete.

- Matched/tracked-object false admission: 0.00%.
- Matched/tracked-object false rejection: 6.47%.
- Strict end-to-end GT false admission: 11.42% with 45.18% observation coverage; this includes upstream perception misses and must not be attributed solely to the channel/AoI shield.
- Frames flagged over budget by the shield: 56.74%.
- C1 estimate misses among attempted captures: 2.74%.

## Overall pilot summary

| controller   |   frames |   split_pct |   skip_pct |   capture_attempt_pct |   delivery_pct_attempted |   c1_estimate_miss_pct_attempted |   over_budget_pct |   selected_true_safe_pct |   selected_matched_true_safe_pct |   false_admit_selected_pct |   false_admit_selected_matched_pct |   false_reject_frame_pct |   mean_prb_cost |   mean_oracle_reward_gap_safe_only |   oracle_action_set_mismatch_pct |   observation_coverage_pct |
|:-------------|---------:|------------:|-----------:|----------------------:|-------------------------:|---------------------------------:|------------------:|-------------------------:|---------------------------------:|---------------------------:|-----------------------------------:|-------------------------:|----------------:|-----------------------------------:|---------------------------------:|---------------------------:|
| clairvoyant  |     1699 |     22.7781 |    77.2219 |                5.7092 |                 100      |                           0      |           39.4938 |                  60.5062 |                          70.5121 |                     0      |                                  0 |                   0      |          0.0368 |                             0      |                           0      |                    45.1805 |
| shielded     |     1699 |      5.3561 |    94.6439 |                4.2966 |                  97.2603 |                           2.7397 |           56.7393 |                  49.2054 |                          68.6875 |                    11.4185 |                                  0 |                   6.4744 |          0.0258 |                             0.0051 |                          16.8923 |                    45.1805 |

## Per-replay summary

| scenario            | controller   |   frames |   split_pct |   skip_pct |   degraded_tier_pct |   over_budget_pct |   shield_ood_pct |   selected_true_safe_pct |   selected_matched_true_safe_pct |   false_admit_selected_pct |   false_admit_selected_matched_pct |   false_reject_frame_pct |   mean_bound_m |   p95_true_risk_m |   p95_matched_true_risk_m |   mean_reward |   mean_prb_cost |   mean_oracle_reward_gap_safe_only |   oracle_action_set_mismatch_pct |   attempts |   capture_attempt_pct |   delivery_pct_attempted |   c1_estimate_miss_count |   c1_estimate_miss_pct_attempted |   truth_objects |   observed_objects |   unobserved_gt_objects |   observation_coverage_pct |
|:--------------------|:-------------|---------:|------------:|-----------:|--------------------:|------------------:|-----------------:|-------------------------:|---------------------------------:|---------------------------:|-----------------------------------:|-------------------------:|---------------:|------------------:|--------------------------:|--------------:|----------------:|-----------------------------------:|---------------------------------:|-----------:|----------------------:|-------------------------:|-------------------------:|---------------------------------:|----------------:|-------------------:|------------------------:|---------------------------:|
| ctrl_busylight_200k | clairvoyant  |      600 |     29.8333 |    70.1667 |              1.5    |           33      |                0 |                  67      |                          83.3333 |                     0      |                                  0 |                   0      |         3.9138 |           15.5354 |                    6.6255 |        0.7939 |          0.0416 |                             0      |                           0      |         32 |                5.3333 |                 100      |                        0 |                           0      |            1590 |                466 |                    1124 |                    29.3082 |
| ctrl_busylight_200k | shielded     |      600 |      5.8333 |    94.1667 |              4      |           58.3333 |                0 |                  52.5    |                          80.3333 |                    20.5    |                                  0 |                   5.6667 |        13.5395 |            1e+06  |                   56.6151 |        0.2212 |          0.0299 |                             0.0038 |                          21.1667 |         22 |                3.6667 |                  90.9091 |                        2 |                           9.0909 |            1590 |                466 |                    1124 |                    29.3082 |
| ctrl_obeylight_200k | clairvoyant  |      600 |     29.5    |    70.5    |              2.3333 |           51.3333 |                0 |                  48.6667 |                          53.5    |                     0      |                                  0 |                   0      |         5.6328 |           21.0359 |                   21.0359 |        0.6482 |          0.0423 |                             0      |                           0      |         40 |                6.6667 |                 100      |                        0 |                           0      |            1355 |                808 |                     547 |                    59.631  |
| ctrl_obeylight_200k | shielded     |      600 |      5.5    |    94.5    |              4      |           79.8333 |                0 |                  31.5    |                          51.6667 |                     4.6667 |                                  0 |                  12.6667 |        12.9477 |            1e+06  |                   23.5154 |        0.4399 |          0.0256 |                             0.0073 |                          23.5    |         30 |                5      |                 100      |                        0 |                           0      |            1355 |                808 |                     547 |                    59.631  |
| oppwin_fast_test    | clairvoyant  |      499 |      6.2124 |    93.7876 |              4.008  |           33.0661 |                0 |                  66.9339 |                          75.5511 |                     0      |                                  0 |                   0      |         2.0558 |            9.9732 |                    7.1478 |        0.2484 |          0.0245 |                             0      |                           0      |         25 |                5.01   |                 100      |                        0 |                           0      |             240 |                165 |                      75 |                    68.75   |
| oppwin_fast_test    | shielded     |      499 |      4.6092 |    95.3908 |              3.4068 |           27.0541 |                0 |                  66.5331 |                          75.1503 |                     8.6172 |                                  0 |                   0      |         1.8517 |           11.4376 |                    6.7834 |        0.172  |          0.0209 |                             0.0044 |                           3.8076 |         21 |                4.2084 |                 100      |                        0 |                           0      |             240 |                165 |                      75 |                    68.75   |

## Projection and interpretation guardrails

- Attempted transmitted frames: 170.
- Payload-projected attempts: 57.65%.
- FPS-projected attempts: 100.00%.
- `split_pct` is the fraction of 20 Hz control ticks for which a SPLIT schedule was active; `capture_attempt_pct` is the actual transmitted-frame fraction.
- Shield false-admit/reject values are surrogate validation against replay GT + synthetic channel truth, not live safety validation.
- `false_admit_selected_matched_pct` isolates localization-shield failures on matched/tracked objects, matching the staleness study's C2 domain. `false_admit_selected_pct` is the stricter end-to-end GT exposure and includes upstream perception misses. Observation coverage is reported separately.
- The replay GT contains vehicles only. Pedestrian conclusions require the separately labelled synthetic stress extension and cannot be claimed from this pilot.
- 25-40 m remains extrapolative and was not used in this pilot.

## Artifacts

Run directory: `rl_agent/policy/experiments/pilot/20260810_181615`

See `per_frame_metrics.csv`, `summary.csv`, `replay_registry.csv`, `resolved_config.yaml`, `manifest.json`, and `figures/pilot_mode_and_risk.{png,pdf}` in that directory.
