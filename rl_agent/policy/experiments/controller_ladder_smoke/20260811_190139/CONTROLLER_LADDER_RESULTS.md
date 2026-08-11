# Track A controller ladder results

**Implementation status:** `scaffold_validation`.

This is a table-driven SPLIT+SKIP surrogate comparison on the explicitly pinned corrected-vehicle corpus. It is not a CARLA/OAI run, LOCAL evaluation, live safety validation, or RL result.

## Shared comparison contract

- Fixed, threshold rule, one-step greedy, fitted LinUCB, and shielded MPC use the identical canonical action catalog and live `A_m -> A_safe` implementation.
- LinUCB trains only on the grouped training split, using matched/tracked environment reward feedback, and is frozen on the evaluation split.
- MPC replans each tick from observable state with declared Markov-expected capacity, modal-rung latency, and constant-kinematics projections; it receives neither future replay frames nor true channel capacity.
- DQN/SAC/PPO are intentionally absent until the simpler ladder is reviewed.

## Evaluation summary

| controller   |   frames |   split_pct |   skip_pct |   capture_attempt_pct |   over_budget_pct |   selected_matched_true_safe_pct |   matched_false_admit_conditional_pct |   matched_false_reject_conditional_pct |   mean_predicted_reward |   mean_matched_true_scored_reward_finite |   mean_prb_cost |
|:-------------|---------:|------------:|-----------:|----------------------:|------------------:|---------------------------------:|--------------------------------------:|---------------------------------------:|------------------------:|-----------------------------------------:|----------------:|
| fixed        |      600 |      8.5    |    91.5    |                5.3333 |           92.5    |                          86.6667 |                                     0 |                                91.3462 |                  0.6648 |                                   0.8173 |          0.0314 |
| greedy       |      600 |      2.1667 |    97.8333 |                2.1667 |           93.6667 |                          86.6667 |                                     0 |                                92.6923 |                  0.6859 |                                   0.8481 |          0.0127 |
| linucb       |      600 |      3      |    97      |                2.1667 |           94      |                          86.6667 |                                     0 |                                93.0769 |                  0.6289 |                                   0.7906 |          0.0144 |
| mpc          |      600 |      2.1667 |    97.8333 |                2.1667 |           93.6667 |                          86.6667 |                                     0 |                                92.6923 |                  0.6859 |                                   0.8481 |          0.0127 |
| rule         |      600 |      3.1667 |    96.8333 |                2.8333 |           92.3333 |                          86.6667 |                                     0 |                                91.1538 |                  0.6514 |                                   0.8047 |          0.0148 |

## Interpretation

A smoke artifact validates plumbing only. A non-smoke artifact is a completed surrogate controller evaluation, but adoption still requires held-out anticipatory traces and comparable safety; these results alone do not justify RL.

## Artifacts

Run directory: `rl_agent/policy/experiments/controller_ladder_smoke/20260811_190139`

See `per_frame_metrics.csv`, `training_metrics.csv`, `summary.csv`, `controller_states.json`, `replay_registry.csv`, `resolved_config.yaml`, `manifest.json`, and the figure files.
