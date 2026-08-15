# Pre-RL controller ladder results

> **Causal-audit scope (2026-08-14).** This replay supplies same-frame post-tail detections and GT-assisted
> matched tracks before action selection. The runtime comparison is **noncausal matched-support**. Static
> full-36-profile Task C results are evaluated separately and remain valid.

**Implementation status:** `completed_surrogate_controller_evaluation`.

This is a table-driven SPLIT+SKIP surrogate comparison on the explicitly pinned `policy_corpus_advisor_rich_v5` corpus. It is not a CARLA/OAI run, LOCAL evaluation, live safety validation, or RL result.

## Shared comparison contract

- Enabled baselines: exact finite expected-reward enumerator over shield candidates; measured-profile lambda-RDO supported-hull lookup; AoI-index-inspired freshness-risk-per-PRB heuristic (not Whittle).
- Every baseline uses the identical canonical action catalog and live `A_m -> A_safe` shield; the runner rejects candidate-set bypasses.
- If enabled, LinUCB trains only on the grouped training split and is frozen for evaluation; MPC uses only its declared observable-state projection.
- DQN/SAC/PPO are intentionally absent until the simpler ladder is reviewed.

## Evaluation summary

| controller          |   frames |   split_pct |   skip_pct |   capture_attempt_pct |   over_budget_pct |   selected_matched_true_safe_pct |   matched_false_admit_conditional_pct |   matched_false_reject_conditional_pct |   mean_predicted_reward |   mean_matched_true_scored_reward_finite |   mean_prb_cost |
|:--------------------|---------:|------------:|-----------:|----------------------:|------------------:|---------------------------------:|--------------------------------------:|---------------------------------------:|------------------------:|-----------------------------------------:|----------------:|
| aoi_index           |     2638 |     21.721  |    78.279  |                8.3776 |           14.5186 |                          92.7976 |                                     0 |                                 8.0033 |                  0.1352 |                                   0.1419 |          0.0779 |
| budgeted_enumerator |     2638 |     29.3783 |    70.6217 |                8.9462 |           14.1774 |                          92.7597 |                                     0 |                                 7.6358 |                  0.1427 |                                   0.1489 |          0.0682 |
| lambda_rdo          |     2638 |     29.3783 |    70.6217 |                8.9462 |           14.1774 |                          92.7597 |                                     0 |                                 7.6358 |                  0.1427 |                                   0.1489 |          0.0682 |

## Interpretation

A smoke artifact validates plumbing only. A non-smoke artifact is a completed surrogate controller evaluation, but adoption still requires held-out anticipatory traces and comparable safety; these results alone do not justify RL.

## Artifacts

Run directory: `rl_agent/policy/experiments/controller_ladder/20260814_220006`

See `per_frame_metrics.csv`, `training_metrics.csv`, `summary.csv`, `controller_states.json`, `replay_registry.csv`, `resolved_config.yaml`, `manifest.json`, and the figure files.
