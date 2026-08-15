# Table-driven SPLIT+SKIP surrogate and controller ladder

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

## Controller ladder on a verified corpus

`configs/controller_ladder.yaml` deliberately does not point at the legacy `staleness` replay. Pin a verified
corpus root and its episode-level grouped split manifest in that file, or pass all three artifacts explicitly:

```bash
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  -m rl_agent.policy.run_controller_ladder \
  --config rl_agent/policy/configs/controller_ladder.yaml \
  --replay-root <verified-corpus-root> \
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
- `controllers.py`, `ladder.py` — common controller interface plus fixed, rule,
  greedy, LinUCB, MPC, exact-enumerator, lambda-RDO, and AoI-index-inspired
  rungs; every selected action must belong to the shared shield's candidate set.
  The legacy replay supplying that interface is noncausal and is not deployable evidence.
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

The accepted multiclass corpus is
`data_collection/experiments/policy_corpus_advisor_rich_v5/20260813_045142_full`;
its structural verifier is `verification/20260813_061952`, with impact run
`pcarv5_mixed_va01` excluded. The completed reward-v5 Phase-1 ladder artifact is
`experiments/controller_ladder/20260813_063514`.

The corpus remains valid for perception QA, workload characterization, and the
legacy matched-support analyses. It is **not** a paired helper-recipient causal
corpus and cannot measure cooperative warning lead.

On six held-out trajectories, finite matched reward is 0.19655 for greedy and
0.19834 for MPC (+0.91%), with the same 91.13% matched-safe rate. They disagree
on only 2.54% of finite frames. A later causal audit showed that the replay
provides current-frame post-tail detections and GT-assisted tracks before the
action. The result is therefore a **noncausal matched-support upper-bound study**,
not a dynamic-controller NO-GO. The static measured-profile selection result
remains valid. A new causal gap—not a retune of this replay—is required before
reconsidering SAC/DQN/PPO. Full evidence and limitations are in
`../RL_JOURNEY_REPORT.md` and `data_collection/EVALUATION_CONTRACT_DECISION_V5.md`.

Tasks A-C are complete:

- scene-context stability: `../contextual_knob/experiments/20260814_214749`;
- observed-vulnerable guardrail ablation:
  `experiments/vulnerable_guardrail/20260814_215337`;
- exact-enumerator/lambda-RDO/AoI baseline audit:
  `experiments/task_c/20260814_220006`.

The full 36-profile scalar RDO problem is not exactly represented by its
supported hull, while lambda-RDO is exactly equivalent to full enumeration on
the retained catalog only inside the current noncausal replay. The static result
is a conditional design rule, not a universal convex-hull or deployable-runtime claim.

Phase-2 collection/evaluation is governed by
`../../phase2_map_sharing/PHASE2_PAIRED_CAUSAL_CORPUS_SPEC.md`. Until its
two-trajectory pilot is reviewed and passes, do not launch the full corpus,
OAI evaluation, a three-action ladder, or RL training.
