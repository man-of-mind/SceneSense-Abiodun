# UE repeatable-route qualification contract v1

**Status:** PRE-RUN CONTRACT — manual and machine qualification pending

**Decision date:** 2026-08-20

**Owners:** Abiodun and Codex

**Parent authority:**
[`UE_AGENT_EXECUTION_CHECKLIST_V2.md`](UE_AGENT_EXECUTION_CHECKLIST_V2.md)

## 1. What UE-R1 through UE-R3 mean

`UE` identifies the current UE-controller workstream and `R` means **route
qualification**. These are checklist identifiers, not three agents or three
different driving policies.

These are the three decisions that must be written before watching the car:

- **UE-R1 — route contract:** exactly which map, route, spawn, controller,
  speed, and clocks the ego uses;
- **UE-R2 — qualification scene:** exactly which actors are present while the
  route itself is tested; and
- **UE-R3 — acceptance contract:** what Abiodun checks visually and what the
  logger must prove before we call the route repeatable.

The three observed trials happen afterward. They are evidence for the frozen
contract, not a way to invent the contract after seeing the result.

## 2. UE-R1 — frozen route contract

| Field | Frozen value |
|---|---|
| Route ID | `town10hd_opt_safe_perimeter_loop_v3` |
| CARLA map | `Carla/Maps/Town10HD_Opt` |
| Route JSON | `data_collection/routes/town10hd_opt_advisor_safe_perimeter_loop_v3.json` |
| Route JSON SHA-256 | `0d3cceeb30d603e258cc61c00bb51e8d0ca29c176e7fccb38ec1e10692233860` |
| Progress CSV | `data_collection/routes/town10hd_opt_advisor_safe_perimeter_loop_v3.progress.csv` |
| Progress CSV SHA-256 | `f3dc2f4d8c59905801fdfad2df7a19f2b427459d4039ed3a8cdec3535e818ce1` |
| Progress points | 85 |
| Open path length | 330.775 m |
| Closing seam | 7.248 m |
| Closed reference length | 338.023 m |
| Ego spawn index | 55, exact |
| Ego blueprint | `vehicle.lincoln.mkz` (verified in the CARLA 0.10 server) |
| Controller | lightweight direct-route controller, no Traffic Manager path |
| Target speed | 6.0 m/s, reduced through turns by the controller |
| World/control clock | synchronous 20 Hz |
| Manual-view pacing | 50 ms monotonic wall-clock schedule per tick; catch up if late |
| Qualification trials | 3 independent one-lap trials |
| Viewer | CARLA chase spectator following the ego |

The direct controller is chosen because it already follows the route indices
cyclically. The existing base runner's `--ego-fixed-path-loop` is not used as
proof of looping: it only appends the first point once to a Traffic Manager
path and has no lap detector.

### Valid lap event

A lap counts only after the detector is armed. It must:

1. leave the start gate;
2. pass the ordered route-progress sequence and reach at least 95% of the
   reference closed length;
3. wrap from the final route region to the initial region exactly once; and
4. return within 4.0 m of the start approach with heading error no greater than
   15 degrees.

This prevents an immediate false completion because the spawn and route seam
are geometrically close.

## 3. UE-R2 — qualification scene

The first route qualification deliberately tests movement only:

- one ego vehicle;
- one attached collision sensor;
- the spectator camera;
- zero ambient vehicles;
- zero walkers;
- zero blockers;
- no RGB/radar inference sensors;
- no model, OAI, map publication, or ACK path.

This isolates a route/controller defect from perception, traffic, or scripted
actor behavior. After the route passes, the fixed blocker and deterministic
pedestrian are added and checked in the later representative integration
stage. They are not allowed to obscure whether the basic loop itself works.

Each trial starts from a clean exact spawn and destroys every owned ego/sensor
actor before the next trial. The same CARLA world may remain open for Abiodun's
visual review, but owned state cannot carry across trials.

## 4. UE-R3 — machine acceptance

All thresholds below are frozen before the CARLA trials.

| Measurement | Pass rule |
|---|---:|
| Map and route hashes | exact match |
| Trial count | exactly 3 independent trials |
| Completed laps per trial | exactly 1 |
| Route wrap per trial | exactly 1, after arming |
| Unwrapped progress | 95–110% of 338.023 m |
| False/startup completion | 0 |
| Return position error | <= 4.0 m |
| Return heading error | <= 15 degrees |
| Cross-track error p95 | <= 1.5 m |
| Persistent divergence | no cross-track > 2.0 m for >= 0.5 s |
| Absolute cross-track maximum | <= 3.0 m |
| Ego collision events | 0 |
| Unexplained stall after warm-up | no speed < 0.5 m/s for > 5 s |
| Per-trial simulator duration | < 120 s |
| Three-trial duration spread | max–min <= 5% of median |
| Control cadence | 20 Hz in CARLA simulation time |
| Duplicate/non-monotonic world frames | 0 |
| Final collision drain | exactly 1 unmeasured flush tick before sensor cleanup |
| Owned ego/sensor leaks | 0 within 5.0 s of simulated cleanup time |

Wall-clock runtime is diagnostic only. Machine load must not make a
geometrically correct route fail when the CARLA simulation-time contract
passes. The 50 ms pacer exists only so Abiodun can observe the chase view at
approximately real time. After the measured lap stops, the ego brakes and the
collision sensor remains alive for one unmeasured tick plus a bounded callback
settle before zero-collision classification and cleanup.

## 5. Abiodun's manual review

Abiodun watches every complete trial in the chase view and records `PASS` or
`FAIL` for each item:

1. the ego spawns in the intended lane and heading without a visible physics
   jump;
2. it follows the expected perimeter road and makes every intended turn;
3. it does not climb a curb, use a sidewalk, enter the wrong lane, cut across a
   corner, oscillate, reverse, or make an unnatural U-turn;
4. steering, acceleration, braking, and turn speed look stable;
5. the final-to-first seam looks like one smooth route rather than a teleport
   or abrupt correction;
6. it completes the full route before being counted as returned;
7. trials 2 and 3 look materially the same as trial 1; and
8. the previous ego and collision sensor disappear before the next trial.

Any anomaly gets a trial number, approximate simulation time/route region, and
short comment. A machine pass plus any manual fail is an overall fail pending
diagnosis.

## 6. Output contract

The qualifier writes a new immutable timestamped directory:

```text
resolved_config.yaml
manifest.json
route_contract.json
route_trace.csv
route_events.csv
ROUTE_MACHINE_REVIEW.json
manual_review_template.json
REVIEW_REQUIRED.json or FAILED.json
```

`route_trace.csv` includes at least:

```text
experiment_id,trial_id,frame_id,sim_time_s,
ego_x,ego_y,ego_z,ego_yaw_deg,ego_speed_mps,
route_index,unwrapped_progress_m,lap_count,cross_track_m,heading_error_deg,
throttle,steer,brake,collision_count,stall_s,divergence_s
```

`route_events.csv` includes at least:

```text
experiment_id,trial_id,event_id,event_type,frame_id,sim_time_s,
route_index,unwrapped_progress_m,status,details
```

Machine success emits `REVIEW_REQUIRED.json`, not `COMPLETED.json`. Only after
Abiodun records the eight visual checks for all three trials may a separate
review step emit the final route-qualification decision. The raw trial
directory is never modified or overwritten.

## 7. Stop rules

Stop the active trial immediately and preserve evidence on:

- any collision;
- persistent divergence or an absolute cross-track breach;
- a false lap completion;
- a stall longer than the frozen limit;
- missing/non-monotonic world frames;
- loss of the ego actor;
- map/route/config drift; or
- user abort after visible unsafe or abnormal behavior.

Do not begin model loading, OAI actuation, scene actors, or the 4x4 pilot during
this task.
