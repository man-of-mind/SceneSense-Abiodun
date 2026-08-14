# Expanded action gate v1 results

Verdict: **`EXPANDED_SURROGATE_NO_GO_STOP`**.

This was a desk-only run over immutable accepted reward-v5 replay. It launched no OAI or CARLA and includes
neither LOCAL, MPC, nor RL. The oracle is a joint true-state **one-step** upper bound. The replay comparison is
queue-free and must not be presented as a shared-queue or real-radio result.

## Registered reward gate

- Expanded decentralized greedy: 0.192667
- Expanded joint oracle: -47.950017
- Absolute lift: -48.142684
- Relative lift: -24987.525%
- Group-cluster bootstrap 95% interval: [-144.428055, 0.002378]
- Lift by UE count: {'2': -55.01987944595771, '4': -0.0023174571597657048}
- Minimum paired worst-UE lift: -1260.460765
- Maximum decentralized aggregate true-C1 miss fraction: 0.840%

Frozen checks: `{"absolute_lift_pass": false, "bootstrap_lower_pass": false, "n2_and_n4_positive_pass": false, "queue_free_c1_validity_pass": true, "relative_lift_pass": false, "worst_ue_regression_pass": false}`.

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
