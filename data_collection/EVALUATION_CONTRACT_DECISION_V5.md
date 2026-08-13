# Advisor-rich v5 acceptance and pre-RL decision

**Decision: accept the 23-run corpus; do not re-collect. The current
SPLIT+SKIP surrogate gives a `NO-GO` for RL training because greedy and MPC
remain effectively tied under reward v5.**

The immutable source batch is
`experiments/policy_corpus_advisor_rich_v5/20260813_045142_full`. Collection
completed 24/24 runs and 8,480/8,480 frames. The accepted inventory excludes
only `pcarv5_mixed_va01`, whose ego/walker impact is independently invalid;
the remaining 23 trajectories are not rewritten.

## Acceptance contract

The authoritative acceptance is
`verification/20260813_061952` with status `PASS`. It is based on structural
controller-corpus gates:

- native 10 Hz world/sensor sampling, 1280x720 RGB, FOV 120 degrees, radar
  raster radius 4, temporal window 2, and 200,000 radar points/s;
- median radar support **19,404.5 returns/frame** over accepted runs, or
  **104.37%** of the retained 18,591.5/frame reference;
- complete runs, zero logged traffic collisions, zero persistent gridlock,
  zero actor leaks, and both pedestrian and vehicle observations populated;
- all run-level timing, decoder-telemetry, scenario-realization, and cleanup
  checks passing.

Perception recall is report-only for this controller-training corpus. Score
thresholds were selected on complete validation trajectories by maximum F1
and frozen at **0.165 pedestrian / 0.205 vehicle** before the test split.

| Test diagnostic | Trajectories | Matched / eligible | Recall | Trajectory-bootstrap 95% CI | Matched localization median / p95 |
|---|---:|---:|---:|---:|---:|
| Pedestrian, <=12 m | 4 | 264 / 389 | **67.87%** | **43.75--69.30%** | **0.575 / 2.732 m** |
| Vehicle, <=25 m | 3 | 84 / 125 | **67.20%** | **0.00--70.00%** | **1.270 / 2.249 m** |

The wide vehicle interval reflects sparse, trajectory-grouped support and is
not promoted to a detector-quality claim. The separate retained-input
diagnostic remains the model-contract check: pedestrian recall was 82.84%
(111/134) with 0.666 m median localization on identical on-contract tensors.
The diverse-corpus numbers above describe the imperfect observations that the
controller must actually handle; they are not collection failure gates.

## Freshness re-score

The accepted split was re-scored without CARLA at
`freshness_rescore/20260813_062203`. All 23 accepted episodes were consumed and
there were no QC exclusions. Its `HUMAN_REVIEW_REQUIRED` disposition is the
analysis tool's intentional handoff state, not a failed corpus gate.

- Pedestrians: 81 tracks / 6,962 object-frames, median speed 1.569 m/s, with
  slow-regime support in 19 runs.
- Vehicles: 9 tracks / 962 object-frames; 98.96% of their frames are at least
  10 m/s, with 5.95 s sustained-fast dwell in each of eight fast runs.
- Freshness pressure is material: **54.43%** of mapped GT-seeded object-frames
  exceed epsilon, and **52.27%** of mapped detection-seeded object-frames do.

The richer corpus therefore fixes the old corpus's missing dynamics. Whether
those dynamics create useful temporal control headroom is answered by the
baseline comparison below.

## Reward-v5 baseline ladder

The authoritative immutable run is
`rl_agent/policy/experiments/controller_ladder/20260813_063514`. It evaluates
six held-out test trajectories (2,638 frames per controller) with paired
channel/latency randomness. LinUCB trains only on the 12 grouped training
trajectories and is frozen for test. The resolved reward is exactly v5:

`U_task = 0.35 * segmentation + 0.40 * pedestrian_recall + 0.25 * vehicle_recall`

There is no explicit ROI cost. All controllers use the same shielded candidate
set and action catalog.

| Controller | Finite matched reward | Matched-safe | Matched false-admit | Matched false-reject | SPLIT rate | Mean PRB cost |
|---|---:|---:|---:|---:|---:|---:|
| Rule | 0.19176 | 91.13% | 0% | 20.25% | 1.52% | 0.00784 |
| Greedy | 0.19655 | 91.13% | 0% | 20.46% | 3.98% | 0.00895 |
| LinUCB | 0.19056 | 91.13% | 0% | 20.25% | 5.91% | 0.01193 |
| MPC | **0.19834** | 91.13% | 0% | 20.46% | 1.44% | **0.00714** |

MPC minus greedy is only **+0.001795 reward/frame (+0.91%)**. An equal-weight
trajectory bootstrap gives a mean difference of +0.001315 with 95% interval
**[0.000000, 0.003833]**. They choose different actions on only **2.54%** of
finite frames: 0% in both exact-fast trajectories, 0% in both pedestrian
crossing trajectories, and 5.59% in mixed urban. Four of six test trajectories
have exactly zero mean reward difference; nearly all improvement is
concentrated in `pcarv5_mixed_te02`.

The fixed-action controller was retained as an extra diagnostic and scores
0.00467, confirming that adaptive control matters in general. The relevant
result, however, is that short-horizon planning does not materially improve
over the much simpler one-step greedy controller.

## RL go/no-go

**NO-GO for SAC/DQN/PPO training on the current surrogate.** The richer corpus
contains feasible/dynamic frames and substantial freshness pressure, yet MPC
still ties greedy on held-out reward and safety. An RL policy using the same
state, shield, action catalog, and reward is therefore more likely to reproduce
that tie than create defensible additional value.

This is conditional rather than universal: the evaluation is table-driven,
uses a synthetic Markov channel over real accepted-corpus perception replay,
and currently supports SPLIT+SKIP only. The LOCAL action table remains pending.
If LOCAL is later calibrated and added, or if a new scenario/channel family
introduces genuine delayed consequences, rerun the non-RL ladder before
reconsidering RL. Under the present contract, the honest research result is
that simple shielded control suffices; RL training should remain stopped.

## Reproducible artifacts

- Acceptance report: `verification/20260813_061952/CORPUS_VERIFICATION.md`
- Recall CI: `verification/20260813_061952/diagnostic_recall_trajectory_bootstrap_ci.csv`
- Localization diagnostic: `verification/20260813_061952/diagnostic_localization_error.csv`
- Freshness report: `freshness_rescore/20260813_062203/FRESHNESS_RESCORE.md`
- Baseline report: `rl_agent/policy/experiments/controller_ladder/20260813_063514/CONTROLLER_LADDER_RESULTS.md`
- Baseline per-frame evidence: `rl_agent/policy/experiments/controller_ladder/20260813_063514/per_frame_metrics.csv`
