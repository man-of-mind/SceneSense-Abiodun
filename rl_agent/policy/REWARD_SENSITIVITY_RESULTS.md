# Track A reward one-at-a-time sensitivity

> **Causal-audit scope (2026-08-14).** This stateful Phase-1 result uses the legacy noncausal matched-support
> replay. It remains a reproducible sensitivity study, not a basis for Phase-2 reward tuning.

**Scope:** the pre-registered seven cells (baseline plus low/high `w_error`, `lambda_prb`, and `w_task`) at epsilon=2.0 m, preferred core=90 KiB, range<=25 m, `ucb_k=0`, and C1=0.70. Replay, channel seeds, and per-tick latency shocks are paired. No CARLA, OAI, LOCAL, RL, or model training was run.

## Robustness result

Across the seven cells, SPLIT scheduling spans 0.000 percentage points and shield over-budget spans 0.000 points. The largest paired action change from baseline is 17/1699 frames.

Absolute reward values are not compared across cells because changing a reward weight changes the units of the scalar objective. The defensible comparison is behavior and physical components: mode, capture rate, task utility, localization, PRB cost, feasibility, and paired action changes.

## All cells

| cell_id         |   w_error |   lambda_prb |   w_task |   split_pct |   capture_attempt_pct |   degraded_tier_pct |   over_budget_pct |   matched_false_admit_count |   admitted_send_count |   matched_false_admit_conditional_pct |   false_reject_conditional_pct |   mean_expected_task_utility |   mean_expected_localization_m |   mean_prb_cost |   mean_matched_true_scored_reward_finite |   selected_action_changes_vs_baseline |   selected_action_changes_vs_baseline_pct |   raw_safe_set_changes_vs_baseline |
|:----------------|----------:|-------------:|---------:|------------:|----------------------:|--------------------:|------------------:|----------------------------:|----------------------:|--------------------------------------:|-------------------------------:|-----------------------------:|-------------------------------:|----------------:|-----------------------------------------:|--------------------------------------:|------------------------------------------:|-----------------------------------:|
| baseline        |     0.05  |          1   |      1   |       5.827 |                4.7087 |              4.2966 |           56.5627 |                           0 |                    15 |                                     0 |                        41.9878 |                       0.5538 |                         7.4412 |          0.0276 |                                   0.4396 |                                     0 |                                    0      |                                  0 |
| w_error_low     |     0.025 |          1   |      1   |       5.827 |                4.7087 |              4.2966 |           56.5627 |                           0 |                    15 |                                     0 |                        41.9878 |                       0.5538 |                         7.4412 |          0.0276 |                                   0.4732 |                                     0 |                                    0      |                                  0 |
| w_error_high    |     0.1   |          1   |      1   |       5.827 |                4.7087 |              4.2966 |           56.5627 |                           0 |                    15 |                                     0 |                        41.9878 |                       0.5538 |                         7.4412 |          0.0276 |                                   0.3724 |                                     0 |                                    0      |                                  0 |
| lambda_prb_low  |     0.05  |          0.5 |      1   |       5.827 |                4.7087 |              4.2966 |           56.5627 |                           0 |                    15 |                                     0 |                        41.9878 |                       0.5539 |                         7.4411 |          0.0277 |                                   0.4525 |                                     3 |                                    0.1766 |                                  0 |
| lambda_prb_high |     0.05  |          2   |      1   |       5.827 |                4.7087 |              4.3555 |           56.5627 |                           0 |                    15 |                                     0 |                        41.9125 |                       0.5488 |                         7.4418 |          0.0261 |                                   0.4094 |                                    17 |                                    1.0006 |                                  1 |
| w_task_low      |     0.05  |          1   |      0.5 |       5.827 |                4.7087 |              4.2966 |           56.5627 |                           0 |                    15 |                                     0 |                        41.9878 |                       0.5538 |                         7.4412 |          0.0276 |                                   0.1628 |                                     0 |                                    0      |                                  0 |
| w_task_high     |     0.05  |          1   |      2   |       5.827 |                4.7087 |              4.2966 |           56.5627 |                           0 |                    15 |                                     0 |                        41.9878 |                       0.5539 |                         7.4411 |          0.0277 |                                   0.9933 |                                     3 |                                    0.1766 |                                  0 |

## Guardrails

- Reward weights do not directly alter C1/C2 equations, but changed actions can alter later map state and therefore later raw-safe sets; paired raw-safe-set changes are reported rather than assumed zero.
- Safety rates retain their conditional denominators and should not be read from unconditional frame percentages.
- Vehicle-only replay, thin admitted-SPLIT support, and 90-KiB-anchored payload/FPS projections remain.

## Artifacts

Run directory: `rl_agent/policy/experiments/reward_sensitivity/20260810_210255`

See `per_frame_metrics.csv`, `summary.csv`, `per_replay_summary.csv`, `replay_registry.csv`, `resolved_config.yaml`, `manifest.json`, and `figures/reward_oat_behavior.{png,pdf}`.
