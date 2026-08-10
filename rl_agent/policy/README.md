# Track A — table-driven SPLIT+SKIP surrogate and oracles

This directory contains the gated Track A implementation. It uses existing CARLA replay files and measured
tables only; it does not launch CARLA/OAI, model training, LOCAL inference, or RL.

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
```

Each experiment writes a new timestamped directory containing the resolved config, per-frame CSV, summary,
figures, and a SHA-256 manifest. Do not overwrite a prior experiment directory.

## Layout

- `IMPLEMENTATION_CONTRACT.md` — frozen clock/reward/channel/replay/tier semantics.
- `configs/track_a_pilot.yaml` — resolved pilot defaults and pre-registered weight variants.
- `data/action_catalog.csv` + metadata — seven measured profiles; 35 SPLIT actions + SKIP at runtime.
- `channel.py`, `latency.py`, `replay.py` — measured input adapters and declared projections.
- `env.py` — fixed-20-Hz scheduler, in-flight event queue, and per-object contribution map.
- `shield.py`, `oracles.py` — one shared observation-based shield and the true-state upper bound.
- `run_safety_calibration.py` — paired 5x5 UCB/C1 fixed-point characterization with conditional metrics.
- `tests/` — standard-library unit/contract tests; no pytest dependency.
- `POLICY_RESULTS.md` — latest gated pilot report.
- `SAFETY_CALIBRATION_RESULTS.md` — latest fixed-point calibration report and identifiability verdict.

## Current gate

The deterministic acceptance, one-configuration pilot, and fixed-point 5x5 safety characterization are
complete. The requested calibration surface is degenerate under the current surrogate, so no operating point
was selected and the 12-condition advisor sweep remains deliberately paused.
