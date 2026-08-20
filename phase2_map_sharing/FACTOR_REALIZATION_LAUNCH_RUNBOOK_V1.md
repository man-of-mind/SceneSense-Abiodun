# Exact-16 factor-realization launch runbook v1

Status: **build/review only. Do not launch the 16-row stage until the offline
validator reports all four runtime adapters ready and the durable corner
acceptance exists.** No command in the review-plan step starts CARLA.

## 0. Confirm the final pins and adapters before collecting review evidence

The final code/config hash repin and runtime-readiness review must happen first;
acceptance deliberately binds those hashes and a later repin invalidates it.
With CARLA still off, first run:

```bash
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  -m data_collection.validate_phase2_factor_realization_smoke \
  --require-runtime-ready
```

Then run the command in Step 4. Before corner review, its
status must be `validated_blocked_pending_manual_corner_acceptance`,
`runtime_ready` must be true, and `runtime_blockers` must be empty.

## 1. Generate the immutable eight-corner review plan

Run from `abiodun/` with CARLA still off:

```bash
export FACTOR_REVIEW_ROOT="/tmp/phase2_factor_corner_final_$(date -u +%Y%m%d_%H%M%S)"
export FACTOR_REVIEW_PLAN="$FACTOR_REVIEW_ROOT/review_plan.json"
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  -m data_collection.launch_phase2_factor_realization_smoke \
  --write-corner-review-plan \
  --review-root "$FACTOR_REVIEW_ROOT" \
  --review-plan-output "$FACTOR_REVIEW_PLAN"
```

This validates the pinned v2 manifest and emits eight positive commands: two
geometries by low/high radial closing speed by short/long typed proximity
horizon. Use a new, empty review root for the final pass: the acceptance gate
requires exactly eight summaries, so a failed/retried corner left in a reused
directory is intentionally rejected. It does not start CARLA or OAI.

## 2. Run the eight geometry-only checks manually

Start a clean `Town10HD_Opt` server with `-quality-level=Epic`. Display the
commands with:

```bash
jq -r '.commands[] | [.trajectory_id, .command_shell] | @tsv' \
  "$FACTOR_REVIEW_PLAN"
```

Run each `command_shell` one at a time. Each run is 12 simulated seconds and
uses no sensors, detector, endpoint replay, OAI, controller ladder, or RL. For
every cell, the geometry reviewer must use the exact runtime factor monitor and
write a passing closing-speed, proximity-horizon, and predicted-surface-
clearance gate. Acceptance also cadence-rounds the later 40-sample/10 Hz raw
window from the same row/config and requires the reviewed physical onset to
leave at least 2.9 s through its expected last sample (one tick beyond the
2.8 s postflight minimum). Also inspect the helper and recipient views. Confirm
that:

- low/high ego motion is visibly distinct and plausible;
- short/long realized proximity-horizon conditions are visibly distinct and
  plausible;
- the intended occlusion and legal routes remain intact; and
- there is no collision, overlap, U-turn, floating actor, or visual anomaly.

Any failed physical cell stops the process. Do not weaken a band or silently
substitute another row.

## 3. Archive and accept the reviewed evidence

Only after all eight runs pass and the human checks above are true:

```bash
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  -m data_collection.launch_phase2_factor_realization_smoke \
  --record-corner-acceptance \
  --review-root "$FACTOR_REVIEW_ROOT" \
  --operator Abiodun \
  --confirm-all-listed-checks
```

This copies the source summaries, traces, and screenshots create-only into
`phase2_map_sharing/geometry_reviews/factor_realization_corner_v1/`, verifies
all eight exact-metric gates, and writes a hash-bound acceptance record. It
never deletes or edits the `/tmp` source. Existing archive/acceptance paths are
a hard error rather than an overwrite.

## 4. Validate launch readiness without launching

```bash
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  -m data_collection.launch_phase2_factor_realization_smoke \
  --validate-launch \
  --operator-quality Epic
```

The only launch-ready status is `validated_ready_not_started`. A runtime-
adapter blocker, missing/stale acceptance, hash drift, mixed row selection,
insufficient storage, or non-Epic declaration blocks launch before CARLA is
touched.

The launch manifest fingerprints the base runner, postflight, validator, and
the complete relevant Python/YAML source trees. The detached child recomputes
that source-tree fingerprint before capture and after postflight; any edit to
result-defining code or configuration while the stage runs fails the atomic
batch instead of mixing implementations.

The exact resolved config also enables the pinned Car/Truck/Bus static-
environment truth registry. It is captured after each fresh Town10HD_Opt reload
and before dynamic actor spawn, so the postflight can adjudicate static
occluders rather than failing after an otherwise complete long capture.

## 5. Detached hand-off (only after human authorization)

```bash
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  -m data_collection.launch_phase2_factor_realization_smoke \
  --launch-detached \
  --operator-quality Epic
```

The printed launch record gives exact progress, run-log, summary, and sentinel
paths. The generic audit's inner `COMPLETED.json` means raw capture only. The
outer stage writes `COMPLETED.json` only after the immutable result bundle
passes the all-16 factor, same-consumer-boundary endpoint, causal-availability,
loader-leakage, one-frozen-checkpoint, and structural-integrity gates and the
validator returns the exact registered verdict
`PASS_ATOMIC_EXACT_16_ADMITTED`. Returning without an
exception or writing the inner generic `COMPLETED.json` is not a scientific
PASS. Any failure writes
`FAILED.json`, retains all 16 as an excluded diagnostic fixture, and stops.
Nothing chains to the old 15-row audit, additional collection, OAI, baselines,
or RL.

To monitor without attaching to the process, substitute the paths printed by
the launch record:

```bash
tail -F <outer-progress.jsonl> <raw-capture/progress.jsonl> <run-log>
```

Stop monitoring whenever convenient; the detached stage continues and writes
exactly one outer `COMPLETED.json` or `FAILED.json` plus
`RESULTS_SUMMARY.json`.
