# Track A controller ladder results

**Implementation status:** `completed_surrogate_controller_evaluation`.

This is a table-driven SPLIT+SKIP surrogate comparison on the explicitly pinned corrected-vehicle corpus. It is not a CARLA/OAI run, LOCAL evaluation, live safety validation, or RL result.

## Shared comparison contract

- Fixed, threshold rule, one-step greedy, fitted LinUCB, and shielded MPC use the identical canonical action catalog and live `A_m -> A_safe` implementation.
- LinUCB trains only on the grouped training split, using matched/tracked environment reward feedback, and is frozen on the evaluation split.
- MPC replans each tick from observable state with declared Markov-expected capacity, modal-rung latency, and constant-kinematics projections; it receives neither future replay frames nor true channel capacity.
- DQN/SAC/PPO are intentionally absent until the simpler ladder is reviewed.

## Evaluation summary

| controller   |   frames |   split_pct |   skip_pct |   capture_attempt_pct |   over_budget_pct |   selected_matched_true_safe_pct |   matched_false_admit_conditional_pct |   matched_false_reject_conditional_pct |   mean_predicted_reward |   mean_matched_true_scored_reward_finite |   mean_prb_cost |
|:-------------|---------:|------------:|-----------:|----------------------:|------------------:|---------------------------------:|--------------------------------------:|---------------------------------------:|------------------------:|-----------------------------------------:|----------------:|
| fixed        |     3998 |     25.3877 |    74.6123 |               13.1816 |           72.3862 |                          79.7399 |                                     0 |                                65.4449 |                  0.1525 |                                   0.3961 |          0.087  |
| greedy       |     3998 |      2.5513 |    97.4487 |                1.5758 |           72.6113 |                          79.8399 |                                     0 |                                65.7384 |                  0.2411 |                                   0.4872 |          0.0087 |
| linucb       |     3998 |      4.9525 |    95.0475 |                2.2011 |           72.6363 |                          79.7899 |                                     0 |                                65.7796 |                  0.2299 |                                   0.4755 |          0.0126 |
| mpc          |     3998 |      2.1511 |    97.8489 |                1.5508 |           72.6113 |                          79.8399 |                                     0 |                                65.7384 |                  0.2414 |                                   0.4875 |          0.0084 |
| rule         |     3998 |      2.4512 |    97.5488 |                2.051  |           72.2611 |                          79.7399 |                                     0 |                                65.2991 |                  0.1993 |                                   0.4426 |          0.0109 |

## Interpretation

A smoke artifact validates plumbing only. A non-smoke artifact is a completed surrogate controller evaluation, but adoption still requires held-out anticipatory traces and comparable safety; these results alone do not justify RL.

## Artifacts

Run directory: `rl_agent/policy/experiments/controller_ladder/20260811_190816`

See `per_frame_metrics.csv`, `training_metrics.csv`, `summary.csv`, `controller_states.json`, `replay_registry.csv`, `resolved_config.yaml`, `manifest.json`, and the figure files.
