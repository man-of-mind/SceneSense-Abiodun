# Pre-RL controller ladder results

> **Causal-audit scope (2026-08-14).** This replay supplies same-frame post-tail detections and GT-assisted
> matched tracks before action selection. The comparison is internally paired but **noncausal matched-support**;
> it is not deployable-controller evidence and does not close the full dynamic RL question.

**Implementation status:** `completed_surrogate_controller_evaluation`.

This is a table-driven SPLIT+SKIP surrogate comparison on the explicitly pinned `policy_corpus_advisor_rich_v5` corpus. It is not a CARLA/OAI run, LOCAL evaluation, live safety validation, or RL result.

## Shared comparison contract

- Fixed, threshold rule, one-step greedy, fitted LinUCB, and shielded MPC use the identical canonical action catalog and live `A_m -> A_safe` implementation.
- LinUCB trains only on the grouped training split, using matched/tracked environment reward feedback, and is frozen on the evaluation split.
- MPC replans each tick from observable state with declared Markov-expected capacity, modal-rung latency, and constant-kinematics projections; it receives neither future replay frames nor true channel capacity.
- DQN/SAC/PPO are intentionally absent until the simpler ladder is reviewed.

## Evaluation summary

| controller   |   frames |   split_pct |   skip_pct |   capture_attempt_pct |   over_budget_pct |   selected_matched_true_safe_pct |   matched_false_admit_conditional_pct |   matched_false_reject_conditional_pct |   mean_predicted_reward |   mean_matched_true_scored_reward_finite |   mean_prb_cost |
|:-------------|---------:|------------:|-----------:|----------------------:|------------------:|---------------------------------:|--------------------------------------:|---------------------------------------:|------------------------:|-----------------------------------------:|----------------:|
| fixed        |     2638 |     54.0561 |    45.9439 |               27.1418 |           27.4829 |                          91.1296 |                                     0 |                                20.4574 |                 -0.0254 |                                   0.0047 |          0.199  |
| greedy       |     2638 |      3.9803 |    96.0197 |                1.5542 |           27.4829 |                          91.1296 |                                     0 |                                20.4574 |                  0.1627 |                                   0.1965 |          0.0089 |
| linucb       |     2638 |      5.9136 |    94.0864 |                1.4784 |           27.2934 |                          91.1296 |                                     0 |                                20.2495 |                  0.1577 |                                   0.1906 |          0.0119 |
| mpc          |     2638 |      1.4405 |    98.5595 |                1.3268 |           27.4829 |                          91.1296 |                                     0 |                                20.4574 |                  0.1645 |                                   0.1983 |          0.0071 |
| rule         |     2638 |      1.5163 |    98.4837 |                1.4026 |           27.2934 |                          91.1296 |                                     0 |                                20.2495 |                  0.1587 |                                   0.1918 |          0.0078 |

## Interpretation

A smoke artifact validates plumbing only. A non-smoke artifact is a completed surrogate controller evaluation, but adoption still requires held-out anticipatory traces and comparable safety; these results alone do not justify RL.

## Artifacts

Run directory: `rl_agent/policy/experiments/controller_ladder/20260813_063514`

See `per_frame_metrics.csv`, `training_metrics.csv`, `summary.csv`, `controller_states.json`, `replay_registry.csv`, `resolved_config.yaml`, `manifest.json`, and the figure files.
