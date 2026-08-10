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
- `tests/` — standard-library unit/contract tests; no pytest dependency.
- `POLICY_RESULTS.md` — latest gated pilot report.

## Current gate

The deterministic acceptance and one-configuration pilot are complete. The 12-condition advisor sensitivity
sweep has deliberately not started. The current pilot must be reviewed first, especially the distinction
between tracked-object C2 safety and end-to-end GT exposure from upstream perception misses.
