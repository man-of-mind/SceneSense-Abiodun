# Phase-2 paired pilot implementation readiness

**Status (2026-08-17): TWO-TRAJECTORY PILOT CAPTURE + VERSIONED STRUCTURAL VERIFICATION PASS. FULL COLLECTION, OAI EVALUATION, CONTROLLER EVALUATION, AND RL REMAIN DISABLED.**

This record prevents a structural pilot PASS from being mistaken for C2 performance evidence. CARLA capture is
complete for exactly the reviewed positive/benign pair; no OAI evaluation, full collection, controller evaluation,
or RL run is authorized by that result.

## Implemented and offline-tested

- immutable v1 plus strict `scenesense.map_contribution.v2` objects with source-local track provenance, full
  placement-to-publication timing, exact serialized byte/chunk accounting, covariance/process-noise/validity
  metadata, and unknown-field/GT-key rejection;
- a CV recipient-map baseline that propagates both object and recipient state covariance, rejects unaligned clocks,
  rejects unsupported motion models, and uses relative uncertainty in closest-approach warnings;
- separate placement/publication action sets and a strict per-stage causal state allowlist with mandatory
  observation/availability/decision/clock/arm metadata;
- create-only JSONL causal audit records; evaluation-truth and shadow-inference sources are rejected at the runtime
  boundary;
- independent, deep-copied, revision-guarded counterfactual arm states for `ego_only`, `send_everything`, and
  `hazard_only`;
- controlled-window raw-retention accounting with pre-write permits, duration/per-trajectory/pilot-total/free-space
  limits, pending-write overbooking protection, and no deletion path; and
- an offline validator for the exact sensor, action, non-actuation, storage, and minimal network-transition design
  contract.

Offline command:

```bash
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  -m phase2_map_sharing.run_pilot_contract_preflight \
  --config phase2_map_sharing/configs/paired_causal_pilot_v1.yaml \
  --disk-root .
```

The command checks logical headroom (`pilot maximum + protected free-space floor`); it does not launch or reserve a
live experiment. The eventual writer must own the returned budget and obtain a permit before each heavy write.

## Collector/replay/verifier integration now implemented

- `data_collection/phase2_paired_causal_collector.py` derives from the accepted v5 perception path, moves actor
  truth into `evaluation_truth/`, and writes source-local tracker/runtime evidence without a GT join;
- passive `--external-sync-ticker` support makes the paired orchestrator the only CARLA clock owner. A shared start
  barrier discards pre-decision buffered frames, exact-spawn mode forbids silent helper/recipient geometry swaps,
  and a two-phase per-frame barrier requires both role-specific pre-action `tick_ready` records before one tick and
  both exact-frame completion heartbeats before the next tick;
- raw RGB, radar tensor/point records, and logits use pre-write permits on the CARLA simulation clock. Runtime logs
  continue if the raw quota stops, and the role writer produces a completion/failure summary plus SHA manifest;
- `replay_paired_pilot.py` gives ego-only, send-everything, and hazard-only independent v2 recipient maps, exact
  application/on-wire byte accounting, and an evaluation-only truth join. Its uncertainty/deadline parameters are
  labelled provisional-for-computability, never confirmatory performance settings; and
- `verify_paired_pilot.py` implements the nine gates in order and stops at the first failure. It includes raw/final
  count preservation, synthetic unmatched-detection injection, artifact hash recovery, arm isolation, C2 metric
  computability, integrity, and sensor-density/cadence checks. A positive warning-lead result is deliberately not a
  pilot gate.

The 2026-08-17 integration audit removed the old spawn-53/55/advisor-population placeholders. The candidate now:

- uses the manually accepted Town10HD_Opt legal curbside pair (`helper` lane `+1`, `recipient` lane `-2`) and keeps
  the lane-ID/native-heading assertion live;
- keeps both collector-owned egos frozen through model/sensor startup, verifies their exact realized pose, then
  gives the paired orchestrator the same non-looping direct controllers used by the accepted visual instrument;
- reloads Town10HD_Opt before each matched arm, owns the Sprinter and optional pedestrian directly, and excludes
  ambient NPCs from this capture/computability pilot. NPC density and seed variation belong to the full suites
  after pilot PASS;
- captures 120 frames (12 seconds) per trajectory, enough for the reviewed 3-second-delay crossing without the
  unused 20-second raw tail; and
- labels the shared `cuda:0` assignment correctness-only. Its inference timings are not citable.

## Historical authorization evidence and completed controlled action

The host check with CARLA running observed one RTX 5090 Laptop GPU with 24,463 MiB total, 7,251 MiB used, and
16,748 MiB free. CARLA was the only heavy GPU process (6,362 MiB). The 96% instantaneous utilization is acceptable
only because the runner is synchronous and this pilot tests correctness; it is further evidence that its inference
latency is non-citable.

`paired_causal_pilot_reviewed_v1.yaml` and
`phase2_paired_causal_pilot_reviewed_v1.yaml` separately authorize only the two-trajectory CARLA pilot. The offline
configs remain immutable and unauthorized. Validation fails if OAI, a full collection, controller evaluation, or
RL is enabled, if the GPU/geometry evidence differs between configs, or if a loopback UDP port is already occupied.

The detached launcher below was the reviewed command used to produce the accepted batch. Do **not** rerun it merely
because it remains documented: the pilot gate is complete. The child wrote a sibling run log, per-10-frame
`progress.jsonl`, `COMPLETED.json`, and `RESULTS_SUMMARY.json`; replay and verification were then run separately.

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun

/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  -m data_collection.launch_phase2_paired_causal_pilot \
  --launch-detached | tee /tmp/phase2_pilot_launch.json
```

## Verification result and current boundary

Accepted capture: `data_collection/experiments/phase2_paired_causal_v1/20260817_181354_pilot`. Its authoritative
create-only derived outputs are `evaluation_v3/` and `verification_v3/`; all nine gates PASS. The target-specific
recovery chain covers both helper and recipient evidence at frame 156300 and matches the registered controlled
pedestrian. `evaluation_v2/` and `verification_v2/` are superseded audit intermediates and are not citable.

`python -m unittest discover -s phase2_map_sharing/tests -v`: **48/48 PASS**. The data-collection suite is
**65/65 PASS**. `py_compile`, config resolution, and the storage preflight pass. Free space at launch was
1,273,659,449,344 bytes; the role quotas cap the four pilot streams at 64 GB, below the 80 GB global design cap and
above the protected 500 GB floor. The pilot is a real structural/computability PASS but not C2 performance evidence.
The next gates are the evaluation-only future-trajectory hazard adjudicator and the powered grouped Suite A/B
inventory in `WARNING_EVALUATION_DESIGN_FREEZE.md`; full collection remains HOLD until both are reviewed.

The first detached attempt (`20260817_175134_pilot`) produced no frames: both collectors failed before readiness
because they were invoked by file path and could not resolve the `data_collection` package. Cleanup left zero
dynamic actors. Child launch now uses module mode, the entrypoint is pinned, and all four resolved child commands
pass an import/argument-parser smoke. The failed directory is provenance only, not pilot data.

The second detached attempt (`20260817_175758_pilot`) reached both ready sentinels but produced no retained inputs
or logits. The parent ticked frame 90087 before both collectors had finished their pre-action placement hooks; the
hooks then correctly rejected buffered frames and required frame 90088, while the parent waited for completion of
90087. This exposed a one-sided barrier race that could have recurred on every frame. The repaired protocol is
two-phase: both collectors atomically arm the same stable boundary, the parent advances exactly one tick, and both
collectors must atomically complete that exact frame before another tick. Non-consecutive completion, repeated
pre-capture entry, wrong-role sentinels, and an advanced/skipped frame now fail closed with explicit diagnostics.
Postflight after this failed attempt again showed asynchronous Town10HD_Opt with zero vehicles, walkers, sensors,
or walker controllers. This failed directory is provenance only and must not enter replay or evaluation.

The third detached attempt (`20260817_180751_pilot`) stopped before capture because the unchanged exact-pose gate
found the helper 0.463 m below its requested spawn Z. The orchestrator was legitimately ticking for child/sensor
startup while the child transitioned the newly spawned ego from default physics to frozen; the number of gravity
ticks in that RPC interval was nondeterministic. Frozen spawn now restores the requested transform and zeros both
velocities only after physics is disabled, while exact-mode RPC failure destroys the actor and fails immediately.
A live one-second asynchronous-world smoke held the same helper transform to `3.8e-6 m`; cleanup returned to zero
dynamic actors. No data from the failed attempt is accepted, and the 0.25 m pose gate was not weakened.

The fourth attempt (`20260817_181354_pilot`) completed both 120-frame trajectories for helper and recipient with
zero unintended collisions, stable 10 Hz cadence, on-contract radar density, exact legal geometry, complete actor
cleanup, and immutable manifests. Offline replay and the target-specific multi-source v3 verifier pass all nine
gates. Pilot warning thresholds, warning-lead magnitude, byte advantage, and shared-GPU timing remain provisional;
the capture is accepted for design/calibration planning, not as a paper performance result.
