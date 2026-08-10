# Track A policy results — gated one-configuration pilot

**Scope:** table-driven SPLIT+SKIP only; epsilon=2.0 m; 90 KiB preferred core; objects within 25 m; real CARLA vehicle replay composed with a synthetic Markov channel. No CARLA, OAI, LOCAL, or RL run.

## Gate status

- Canonical seven-profile / 36-action catalog: PASS
- Contract tests: PASS
- Four deterministic acceptance episodes: PASS
- One-config real-replay pilot: COMPLETE
- Twelve-condition advisor sweep: NOT STARTED (still gated on review of this pilot)

## Overall pilot summary

| controller   |   frames |   split_pct |   skip_pct |   capture_attempt_pct |   delivery_pct_attempted |   over_budget_pct |   selected_true_safe_pct |   false_admit_selected_pct |   false_reject_frame_pct |   mean_prb_cost |   mean_oracle_reward_gap_safe_only |   oracle_action_set_mismatch_pct |   observation_coverage_pct |
|:-------------|---------:|------------:|-----------:|----------------------:|-------------------------:|------------------:|-------------------------:|---------------------------:|-------------------------:|----------------:|-----------------------------------:|---------------------------------:|---------------------------:|
| clairvoyant  |     1699 |     87.9929 |    12.0071 |                7.1807 |                      100 |           12.4191 |                  87.5809 |                     0      |                   0      |          0.0651 |                             0      |                           0      |                    24.5957 |
| shielded     |     1699 |     77.163  |    22.837  |                2.884  |                      100 |           39.847  |                  70.2178 |                    22.9547 |                   4.3555 |          0.0574 |                             0.0109 |                          53.2078 |                    24.5957 |

## Per-replay summary

| scenario            | controller   |   frames |   split_pct |   skip_pct |   degraded_tier_pct |   over_budget_pct |   shield_ood_pct |   selected_true_safe_pct |   false_admit_selected_pct |   false_reject_frame_pct |   mean_bound_m |   p95_true_risk_m |   mean_reward |   mean_prb_cost |   mean_oracle_reward_gap_safe_only |   oracle_action_set_mismatch_pct |   attempts |   capture_attempt_pct |   delivery_pct_attempted |   truth_objects |   observed_objects |   observation_coverage_pct |
|:--------------------|:-------------|---------:|------------:|-----------:|--------------------:|------------------:|-----------------:|-------------------------:|---------------------------:|-------------------------:|---------------:|------------------:|--------------:|----------------:|-----------------------------------:|---------------------------------:|-----------:|----------------------:|-------------------------:|----------------:|-------------------:|---------------------------:|
| ctrl_busylight_200k | clairvoyant  |      600 |     76.6667 |    23.3333 |             13.3333 |            0      |                0 |                 100      |                     0      |                   0      |         1.476  |            1.9918 |        0.7983 |          0.0741 |                             0      |                           0      |         36 |                6      |                      100 |            3684 |                647 |                    17.5624 |
| ctrl_busylight_200k | shielded     |      600 |     78.8333 |    21.1667 |             40.1667 |           42.1667 |                0 |                  76      |                    22.8333 |                  10.3333 |         1.7763 |            1e+06  |        0.4311 |          0.0745 |                             0.007  |                          61.1667 |         11 |                1.8333 |                      100 |            3684 |                647 |                    17.5624 |
| ctrl_obeylight_200k | clairvoyant  |      600 |     98.1667 |     1.8333 |             11.1667 |            0      |                0 |                 100      |                     0      |                   0      |         1.6227 |            1.9924 |        0.8072 |          0.0672 |                             0      |                           0      |         52 |                8.6667 |                      100 |            2677 |                866 |                    32.3496 |
| ctrl_obeylight_200k | shielded     |      600 |     88.5    |    11.5    |             47.5    |           50      |                0 |                  83.6667 |                    11.8333 |                   2      |         2.1287 |           13.2393 |        0.5044 |          0.0582 |                             0.0007 |                          54.5    |         22 |                3.6667 |                      100 |            2677 |                866 |                    32.3496 |
| oppwin_fast_test    | clairvoyant  |      499 |     89.3788 |    10.6212 |             41.8838 |           42.2846 |                0 |                  57.7154 |                     0      |                   0      |         1.4091 |            2.5936 |        0.4085 |          0.0518 |                             0      |                           0      |         34 |                6.8136 |                      100 |             502 |                175 |                    34.8606 |
| oppwin_fast_test    | shielded     |      499 |     61.523  |    38.477  |             20.2405 |           24.8497 |                0 |                  47.0942 |                    36.4729 |                   0      |         0.86   |            1e+06  |        0.1515 |          0.0359 |                             0.0237 |                          42.0842 |         16 |                3.2064 |                      100 |             502 |                175 |                    34.8606 |

## Projection and interpretation guardrails

- Attempted transmitted frames: 171.
- Payload-projected attempts: 18.13%.
- FPS-projected attempts: 100.00%.
- `split_pct` is the fraction of 20 Hz control ticks for which a SPLIT schedule was active; `capture_attempt_pct` is the actual transmitted-frame fraction.
- Shield false-admit/reject values are surrogate validation against replay GT + synthetic channel truth, not live safety validation.
- The replay GT contains vehicles only. Pedestrian conclusions require the separately labelled synthetic stress extension and cannot be claimed from this pilot.
- 25-40 m remains extrapolative and was not used in this pilot.

## Artifacts

Run directory: `rl_agent/policy/experiments/pilot/20260810_180620`

See `per_frame_metrics.csv`, `summary.csv`, `replay_registry.csv`, `resolved_config.yaml`, `manifest.json`, and `figures/pilot_mode_and_risk.{png,pdf}` in that directory.
