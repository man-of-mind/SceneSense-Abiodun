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
- Matched/tracked-object false rejection: 8.77%.
- Strict end-to-end GT false admission: 11.83% with 45.18% observation coverage; this includes upstream perception misses and must not be attributed solely to the channel/AoI shield.
- Frames flagged over budget by the shield: 56.74%.
- C1 estimate misses among attempted captures: 1.22%.

## Overall pilot summary

| controller   |   frames |   split_pct |   skip_pct |   capture_attempt_pct |   delivery_pct_attempted |   c1_estimate_miss_pct_attempted |   over_budget_pct |   selected_true_safe_pct |   selected_matched_true_safe_pct |   false_admit_selected_pct |   false_admit_selected_matched_pct |   false_reject_frame_pct |   mean_prb_cost |   mean_oracle_reward_gap_safe_only |   oracle_action_set_mismatch_pct |   observation_coverage_pct |
|:-------------|---------:|------------:|-----------:|----------------------:|-------------------------:|---------------------------------:|------------------:|-------------------------:|---------------------------------:|---------------------------:|-----------------------------------:|-------------------------:|----------------:|-----------------------------------:|---------------------------------:|---------------------------:|
| clairvoyant  |     1699 |     51.03   |    48.97   |               19.3643 |                 100      |                           0      |           39.1995 |                  60.8005 |                          71.3361 |                     0      |                                  0 |                   0      |          0.1391 |                              0     |                           0      |                    45.1805 |
| shielded     |     1699 |      6.1212 |    93.8788 |                4.8264 |                  98.7805 |                           1.2195 |           56.7393 |                  47.2042 |                          68.9818 |                    11.8305 |                                  0 |                   8.7699 |          0.029  |                              0.007 |                          28.6639 |                    45.1805 |

## Per-replay summary

| scenario            | controller   |   frames |   split_pct |   skip_pct |   degraded_tier_pct |   over_budget_pct |   shield_ood_pct |   selected_true_safe_pct |   selected_matched_true_safe_pct |   false_admit_selected_pct |   false_admit_selected_matched_pct |   false_reject_frame_pct |   mean_bound_m |   p95_true_risk_m |   p95_matched_true_risk_m |   mean_reward |   mean_prb_cost |   mean_oracle_reward_gap_safe_only |   oracle_action_set_mismatch_pct |   attempts |   capture_attempt_pct |   delivery_pct_attempted |   c1_estimate_miss_count |   c1_estimate_miss_pct_attempted |   truth_objects |   observed_objects |   unobserved_gt_objects |   observation_coverage_pct |
|:--------------------|:-------------|---------:|------------:|-----------:|--------------------:|------------------:|-----------------:|-------------------------:|---------------------------------:|---------------------------:|-----------------------------------:|-------------------------:|---------------:|------------------:|--------------------------:|--------------:|----------------:|-----------------------------------:|---------------------------------:|-----------:|----------------------:|-------------------------:|-------------------------:|---------------------------------:|----------------:|-------------------:|------------------------:|---------------------------:|
| ctrl_busylight_200k | clairvoyant  |      600 |     78.3333 |    21.6667 |             15.8333 |           30.5    |                0 |                  69.5    |                          85      |                     0      |                                  0 |                   0      |         2.4393 |            9.3036 |                    3.2773 |        0.6112 |          0.2087 |                             0      |                           0      |        155 |               25.8333 |                 100      |                        0 |                           0      |            1590 |                466 |                    1124 |                    29.3082 |
| ctrl_busylight_200k | shielded     |      600 |      6.3333 |    93.6667 |              4      |           58.3333 |                0 |                  51.1667 |                          81.3333 |                    20.5    |                                  0 |                   9      |        10.1448 |            1e+06  |                    6.3127 |        0.3026 |          0.0316 |                             0.0073 |                          44.1667 |         24 |                4      |                  95.8333 |                        1 |                           4.1667 |            1590 |                466 |                    1124 |                    29.3082 |
| ctrl_obeylight_200k | clairvoyant  |      600 |     43.8333 |    56.1667 |             12.5    |           53.6667 |                0 |                  46.3333 |                          53.5    |                     0      |                                  0 |                   0      |         4.7354 |           19.0841 |                   19.0841 |        0.5385 |          0.1354 |                             0      |                           0      |        115 |               19.1667 |                 100      |                        0 |                           0      |            1355 |                808 |                     547 |                    59.631  |
| ctrl_obeylight_200k | shielded     |      600 |      7.1667 |    92.8333 |              5.5    |           79.8333 |                0 |                  27.1667 |                          51.5    |                     5.8333 |                                  0 |                  15.8333 |        11.2409 |            1e+06  |                   19.0841 |        0.4516 |          0.0329 |                             0.0092 |                          31.1667 |         37 |                6.1667 |                 100      |                        0 |                           0      |            1355 |                808 |                     547 |                    59.631  |
| oppwin_fast_test    | clairvoyant  |      499 |     26.8537 |    73.1463 |              8.6172 |           32.2645 |                0 |                  67.7355 |                          76.3527 |                     0      |                                  0 |                   0      |         1.8211 |            8.8893 |                    5.9069 |        0.205  |          0.0597 |                             0      |                           0      |         59 |               11.8236 |                 100      |                        0 |                           0      |             240 |                165 |                      75 |                    68.75   |
| oppwin_fast_test    | shielded     |      499 |      4.6092 |    95.3908 |              3.6072 |           27.0541 |                0 |                  66.5331 |                          75.1503 |                     8.6172 |                                  0 |                   0      |         1.8519 |            1e+06  |                    6.7834 |        0.1707 |          0.0214 |                             0.0049 |                           7.014  |         21 |                4.2084 |                 100      |                        0 |                           0      |             240 |                165 |                      75 |                    68.75   |

## Projection and interpretation guardrails

- Attempted transmitted frames: 411.
- Payload-projected attempts: 58.39%.
- FPS-projected attempts: 100.00%.
- `split_pct` is the fraction of 20 Hz control ticks for which a SPLIT schedule was active; `capture_attempt_pct` is the actual transmitted-frame fraction.
- Shield false-admit/reject values are surrogate validation against replay GT + synthetic channel truth, not live safety validation.
- `false_admit_selected_matched_pct` isolates localization-shield failures on matched/tracked objects, matching the staleness study's C2 domain. `false_admit_selected_pct` is the stricter end-to-end GT exposure and includes upstream perception misses. Observation coverage is reported separately.
- The replay GT contains vehicles only. Pedestrian conclusions require the separately labelled synthetic stress extension and cannot be claimed from this pilot.
- 25-40 m remains extrapolative and was not used in this pilot.

## Artifacts

Run directory: `rl_agent/policy/experiments/pilot/20260810_182039`

See `per_frame_metrics.csv`, `summary.csv`, `replay_registry.csv`, `resolved_config.yaml`, `manifest.json`, and `figures/pilot_mode_and_risk.{png,pdf}` in that directory.
