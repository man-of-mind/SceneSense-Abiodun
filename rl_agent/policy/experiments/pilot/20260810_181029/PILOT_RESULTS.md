# Track A policy results — gated one-configuration pilot

**Scope:** table-driven SPLIT+SKIP only; epsilon=2.0 m; 90 KiB preferred core; objects within 25 m; real CARLA vehicle replay composed with a synthetic Markov channel. No CARLA, OAI, LOCAL, or RL run.

## Gate status

- Canonical seven-profile / 36-action catalog: PASS
- Contract tests: PASS
- Four deterministic acceptance episodes: PASS
- One-config real-replay pilot: COMPLETE
- Twelve-condition advisor sweep: NOT STARTED (still gated on review of this pilot)

## Overall pilot summary

| controller   |   frames |   split_pct |   skip_pct |   capture_attempt_pct |   delivery_pct_attempted |   over_budget_pct |   selected_true_safe_pct |   selected_matched_true_safe_pct |   false_admit_selected_pct |   false_admit_selected_matched_pct |   false_reject_frame_pct |   mean_prb_cost |   mean_oracle_reward_gap_safe_only |   oracle_action_set_mismatch_pct |   observation_coverage_pct |
|:-------------|---------:|------------:|-----------:|----------------------:|-------------------------:|------------------:|-------------------------:|---------------------------------:|---------------------------:|-----------------------------------:|-------------------------:|----------------:|-----------------------------------:|---------------------------------:|---------------------------:|
| clairvoyant  |     1699 |     72.3955 |    27.6045 |                6.0624 |                      100 |             7.063 |                  92.937  |                          95.3502 |                     0      |                              0     |                   0      |          0.0523 |                             0      |                           0      |                    45.1805 |
| shielded     |     1699 |     73.6315 |    26.3685 |                3.4138 |                      100 |            31.548 |                  83.5197 |                          92.8193 |                    10.1236 |                              0.824 |                   2.1778 |          0.0545 |                             0.0077 |                          34.2554 |                    45.1805 |

## Per-replay summary

| scenario            | controller   |   frames |   split_pct |   skip_pct |   degraded_tier_pct |   over_budget_pct |   shield_ood_pct |   selected_true_safe_pct |   selected_matched_true_safe_pct |   false_admit_selected_pct |   false_admit_selected_matched_pct |   false_reject_frame_pct |   mean_bound_m |   p95_true_risk_m |   p95_matched_true_risk_m |   mean_reward |   mean_prb_cost |   mean_oracle_reward_gap_safe_only |   oracle_action_set_mismatch_pct |   attempts |   capture_attempt_pct |   delivery_pct_attempted |   truth_objects |   observed_objects |   unobserved_gt_objects |   observation_coverage_pct |
|:--------------------|:-------------|---------:|------------:|-----------:|--------------------:|------------------:|-----------------:|-------------------------:|---------------------------------:|---------------------------:|-----------------------------------:|-------------------------:|---------------:|------------------:|--------------------------:|--------------:|----------------:|-----------------------------------:|---------------------------------:|-----------:|----------------------:|-------------------------:|----------------:|-------------------:|------------------------:|---------------------------:|
| ctrl_busylight_200k | clairvoyant  |      600 |     61.1667 |    38.8333 |              4.8333 |            0      |                0 |                 100      |                         100      |                     0      |                             0      |                   0      |         1.2758 |            1.9492 |                    1.8189 |        0.8411 |          0.0582 |                             0      |                           0      |         30 |                5      |                      100 |            1590 |                466 |                    1124 |                    29.3082 |
| ctrl_busylight_200k | shielded     |      600 |     69.6667 |    30.3333 |             26.6667 |           28.5    |                0 |                  82.6667 |                         100      |                    17.3333 |                             0      |                   4.8333 |         1.3611 |           30.6956 |                    1.7909 |        0.4048 |          0.0636 |                             0.0054 |                          44      |         19 |                3.1667 |                      100 |            1590 |                466 |                    1124 |                    29.3082 |
| ctrl_obeylight_200k | clairvoyant  |      600 |     89.6667 |    10.3333 |              9.8333 |            0      |                0 |                 100      |                         100      |                     0      |                             0      |                   0      |         1.3845 |            1.9914 |                    1.9914 |        0.7073 |          0.0594 |                             0      |                           0      |         47 |                7.8333 |                      100 |            1355 |                808 |                     547 |                    59.631  |
| ctrl_obeylight_200k | shielded     |      600 |     88.1667 |    11.8333 |             40.6667 |           43.5    |                0 |                  91      |                          92.8333 |                     3.5    |                             1.6667 |                   1.3333 |         1.9215 |            2.0253 |                    2.0237 |        0.541  |          0.0609 |                             0.0033 |                          41.5    |         21 |                3.5    |                      100 |            1355 |                808 |                     547 |                    59.631  |
| oppwin_fast_test    | clairvoyant  |      499 |     65.1303 |    34.8697 |             23.8477 |           24.0481 |                0 |                  75.9519 |                          84.1683 |                     0      |                             0      |                   0      |         0.7523 |            2.3973 |                    2.3832 |        0.2101 |          0.0368 |                             0      |                           0      |         26 |                5.2104 |                      100 |             240 |                165 |                      75 |                    68.75   |
| oppwin_fast_test    | shielded     |      499 |     60.9218 |    39.0782 |             18.4369 |           20.8417 |                0 |                  75.5511 |                          84.1683 |                     9.4188 |                             0.8016 |                   0      |         0.7836 |            2.452  |                    2.3786 |        0.1391 |          0.0359 |                             0.0132 |                          13.8277 |         18 |                3.6072 |                      100 |             240 |                165 |                      75 |                    68.75   |

## Projection and interpretation guardrails

- Attempted transmitted frames: 161.
- Payload-projected attempts: 13.04%.
- FPS-projected attempts: 100.00%.
- `split_pct` is the fraction of 20 Hz control ticks for which a SPLIT schedule was active; `capture_attempt_pct` is the actual transmitted-frame fraction.
- Shield false-admit/reject values are surrogate validation against replay GT + synthetic channel truth, not live safety validation.
- `false_admit_selected_matched_pct` isolates localization-shield failures on matched/tracked objects, matching the staleness study's C2 domain. `false_admit_selected_pct` is the stricter end-to-end GT exposure and includes upstream perception misses. Observation coverage is reported separately.
- The replay GT contains vehicles only. Pedestrian conclusions require the separately labelled synthetic stress extension and cannot be claimed from this pilot.
- 25-40 m remains extrapolative and was not used in this pilot.

## Artifacts

Run directory: `rl_agent/policy/experiments/pilot/20260810_181029`

See `per_frame_metrics.csv`, `summary.csv`, `replay_registry.csv`, `resolved_config.yaml`, `manifest.json`, and `figures/pilot_mode_and_risk.{png,pdf}` in that directory.
