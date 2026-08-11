# Track A — table-driven SPLIT+SKIP surrogate and controller ladder

This directory contains the gated Track A implementation. It uses existing CARLA replay files and measured
tables only; it does not launch CARLA/OAI, LOCAL inference, or RL.

## Reproduce

From the repository root:

```bash
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  -m rl_agent.policy.build_action_catalog

/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  -m unittest discover -s rl_agent/policy/tests -v

/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  -m rl_agent.policy.run_acceptance

/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  -m rl_agent.policy.run_pilot

/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  -m rl_agent.policy.run_safety_calibration

/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  -m rl_agent.policy.run_estimator_sensitivity

/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  -m rl_agent.policy.run_reward_sensitivity

/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  -m rl_agent.policy.run_advisor_sweep
```

Each experiment writes a new timestamped directory containing the resolved config, per-frame CSV, summary,
figures, and a SHA-256 manifest. Do not overwrite a prior experiment directory.

## Controller ladder on the corrected vehicle-v2 corpus

`configs/controller_ladder.yaml` deliberately does not point at the legacy `staleness` replay. Pin the final
vehicle-v2 root and its episode-level grouped split manifest in that file, or pass both explicitly:

```bash
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  -m rl_agent.policy.run_controller_ladder \
  --config rl_agent/policy/configs/controller_ladder.yaml \
  --replay-root <verified-corrected-vehicle-v2-root> \
  --split-manifest <verification-dir>/replay_split_manifest.csv \
  --verification-manifest <verification-dir>/verification_manifest.json
```

For a one-train-episode/one-test-episode plumbing check, append `--scaffold-smoke`. Smoke manifests are marked
`implementation_status=scaffold_validation`; a complete configured run is marked
`completed_surrogate_controller_evaluation`. Neither label denotes a live or RL result. The runner refuses a
headline run against the legacy replay or without an explicit split manifest.
It also verifies `status=PASS`, the full-batch mode, corpus identity, and the batch/config/split hashes before
loading an episode.

## Layout

- `IMPLEMENTATION_CONTRACT.md` — frozen clock/reward/channel/replay/tier semantics.
- `configs/track_a_pilot.yaml` — resolved pilot defaults and pre-registered weight variants.
- `data/action_catalog.csv` + metadata — seven measured profiles; 35 SPLIT actions + SKIP at runtime.
- `channel.py`, `latency.py`, `replay.py` — measured input adapters and declared projections.
- `env.py` — fixed-20-Hz scheduler, in-flight event queue, and per-object contribution map.
- `shield.py`, `oracles.py` — one shared observation-based shield and the true-state upper bound.
- `controllers.py`, `ladder.py` — common deployable interface plus fixed, rule, greedy, LinUCB, and MPC rungs;
  every selected action must belong to the shared shield's candidate set.
- `run_controller_ladder.py` — grouped train/evaluation runner with frozen bandit evaluation, common channel/
  latency seeds, fitted-state serialization, and explicit scaffold-vs-result status.
- `configs/controller_ladder.yaml` — controller thresholds, bandit fit settings, MPC projection, and verified-
  corpus contract.
- `run_safety_calibration.py` — paired 5x5 UCB/C1 fixed-point characterization with conditional metrics.
- `run_estimator_sensitivity.py` — paired 4x3 telemetry-lag/noise causal sensitivity at the pilot point.
- `run_reward_sensitivity.py` — paired seven-cell one-at-a-time reward robustness study.
- `run_advisor_sweep.py` — paired 3x2x2 epsilon/core/range characterization; it never selects a setting.
- `sweep_support.py` — shared replay identity, source provenance, and paired execution helpers.
- `tests/` — standard-library unit/contract tests; no pytest dependency.
- `POLICY_RESULTS.md` — latest gated pilot report.
- `SAFETY_CALIBRATION_RESULTS.md` — latest fixed-point calibration report and identifiability verdict.
- `ESTIMATOR_SENSITIVITY_RESULTS.md` — latest lag/noise sensitivity report.
- `REWARD_SENSITIVITY_RESULTS.md` — latest reward robustness report.
- `ADVISOR_SWEEP_RESULTS.md` — latest advisor-facing achievability report.

## Current gate

The fixed-point 5x5 safety characterization was a valid negative result. Phase 1 therefore uses the reviewed
engineering convention `ucb_k=0`, C1 factor `0.70`; this is not a statistically calibrated optimum. The
authorized follow-on runs are complete. Estimator lag/noise did not explain the approximately 42% baseline
false-reject rate; reward one-at-a-time behavior was stable; and the 3x2x2 advisor grid exposed a strong 40 m
false-admission/feasibility boundary. No advisor-pending value has been selected. LOCAL, RL, CARLA, and OAI
remain out of scope. The corrected vehicle-v2 corpus passed verification at
`data_collection/experiments/policy_corpus_vehicle_v2/20260811_110400_full/verification/20260811_185731`, and
the completed pre-RL ladder is under `experiments/controller_ladder/20260811_190816`. Greedy and MPC are
effectively tied on held-out replay, so the locked sequential-value gate does not yet justify an RL result.
Track B pedestrian evidence remains held by its separate failed unchanged-gate re-smoke; see
`collab/REVIEW_NOTES.md` for the separate disposition.
