# Expanded action gate results (scenesense.policy.expanded_action_gate.v3)

Verdict: **`EXPANDED_SURROGATE_NO_GO_STOP`**.

This was a desk-only run over immutable accepted reward-v5 replay. It launched no OAI or CARLA and includes
neither LOCAL, MPC, nor RL. The oracle is a joint true-state **one-step** upper bound. The replay comparison is
queue-free and must not be presented as a shared-queue or real-radio result.

## Registered reward gate

- Expanded decentralized greedy: 0.192625
- Expanded joint oracle: 0.195290
- Absolute lift: 0.002665
- Relative lift: 1.383%
- Group-cluster bootstrap 95% interval: [0.001929, 0.003452]
- Lift by UE count: {'2': 0.0028136389220889684, '4': 0.001620545157906957}
- Minimum paired worst-UE lift: 0.000009
- Maximum decentralized aggregate true-C1 miss fraction: 0.840%

Frozen checks: `{"absolute_lift_pass": false, "bootstrap_lower_pass": true, "n2_and_n4_positive_pass": true, "queue_free_c1_validity_pass": true, "relative_lift_pass": false, "worst_ue_regression_pass": true}`.

## Deadline-feasibility frontier

Across the equal-C1-share rows, 487/1600 payload × N × rung × deadline × FPS cells meet both the
on-wire rate condition and the queue-free p95 necessary condition. Feasible does **not** mean queue-sufficient.

- 49.4 KiB: 89/200 necessary-feasible cells
- 61.3 KiB: 75/200 necessary-feasible cells
- 64.1 KiB: 75/200 necessary-feasible cells
- 75.2 KiB: 73/200 necessary-feasible cells
- 79.2 KiB: 67/200 necessary-feasible cells
- 90 KiB: 58/200 necessary-feasible cells
- 129.2 KiB: 43/200 necessary-feasible cells
- 400 KiB: 7/200 necessary-feasible cells

Detailed per-cell, per-frame, and per-group outputs are in `feasibility_frontier.csv`,
`per_frame_metrics.csv`, and `group_seed_summary.csv`. Source/code hashes are in `manifest.json`.
