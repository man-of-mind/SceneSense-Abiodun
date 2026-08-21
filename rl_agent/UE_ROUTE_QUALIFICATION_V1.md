# UE route qualification v1 — operator guide

The authoritative frozen thresholds are in
[`UE_ROUTE_QUALIFICATION_CONTRACT_V1.md`](UE_ROUTE_QUALIFICATION_CONTRACT_V1.md).
This file only explains how to validate, run, inspect, and review that contract.

This is the deliberately small first implementation gate for the UE
experiment. It answers only one question: **can the same ego vehicle drive the
same closed Town10HD route three times with stable, measurable, clean motion?**

It does not spawn an RGB camera, radar, NPC, pedestrian, blocker, perception
model, OAI process, or spatial-map connection. Those are later gates after the
motion substrate is visually and mechanically accepted.

## Frozen contract

- Town10HD_Opt, spawn index 55;
- `town10hd_opt_advisor_safe_perimeter_loop_v3.progress.csv`, verified by hash;
- direct closed-route controller at a 6 m/s target;
- synchronous CARLA control at 20 Hz, with every second tick merely labelled as
  a future 10-Hz decision slot;
- monotonic real-time pacing at 50 ms per control tick so the chase view is
  watchable; wall-clock pacing remains diagnostic and never enters the
  simulation-time acceptance gates;
- one fresh Lincoln MKZ and one attached collision sensor per trial;
- three independent one-lap trials;
- chase spectator enabled for manual viewing; and
- create-only, atomic evidence under a new timestamped experiment directory.

After the measured lap stops, the ego is held on its brake while the collision
sensor remains alive for exactly one bounded, unmeasured flush tick. The runner
then drains the collision mailbox before destroying either actor. That tick is
recorded in the machine review but cannot change route progress, lap count, or
the route trace.

The detector cannot complete at startup. It must leave the 15 m start gate,
prove ordered progress through at least 95% of the frozen route, wrap from the
final route region to the initial region exactly once after arming, and return
within 4.0 m and 15 degrees of the start approach.

## Offline validation

This command does not connect to CARLA:

```bash
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  -m rl_agent.ue_route_qualification --validate-only
```

## Run the three visible laps

Start a fresh CARLA server with rendering, then run:

```bash
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  -m rl_agent.ue_route_qualification
```

The runner reloads Town10HD_Opt once, verifies that the world is empty, and
then runs three trials sequentially. Each trial destroys its ego and collision
sensor and verifies their absence before the next ego is spawned.

The new timestamped run directory contains exactly the machine evidence named
by the contract:

- `resolved_config.yaml`;
- `manifest.json` and `route_contract.json`;
- the combined `route_trace.csv` and `route_events.csv`;
- `ROUTE_MACHINE_REVIEW.json` with all per-trial and duration-spread gates;
- `manual_review_template.json`; and
- exactly one machine terminal: `REVIEW_REQUIRED.json` or `FAILED.json`.

Machine success is intentionally named `REVIEW_REQUIRED`, never `PASS`. Watch
all three laps. Copy `manual_review_template.json` to `manual_review.json`,
enter the review time, record all eight checks for each trial, add any anomaly's
trial/time/route region, and set consistent per-trial and overall verdicts.

After all three reviews exist, finalize without rewriting any earlier output:

```bash
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  -m rl_agent.ue_route_qualification \
  --finalize-reviewed-run rl_agent/experiments/ue_route_qualification_v1/<run_id>
```

Only the separate create-only `ROUTE_QUALIFIED.json` may contain the final
`PASS`; a manual failure instead creates `FAILED_MANUAL.json`.
