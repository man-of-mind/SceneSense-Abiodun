# Route B mid-episode population lifecycle — Phase A causal audit (2026-08-24)

Verdict: **hidden ticks CONFIRMED.** Every one of the 59 reported missing frame IDs is a CARLA
world frame advanced by `client.apply_batch_sync(batch, True)` inside `TrafficPopulationManager`,
outside `SamplingWorld.tick()`. Zero of them are dropped sensor callbacks.

The prior "controller started before a world tick" diagnosis is **false** and is refuted below.

## A1. Tick semantics, measured on this build (not assumed)

`TrafficPopulationManager` is constructed with `synchronous_master=True`
(`run_route_b_density_loop.py:1092-1101`) and `args.asynch=False` (`traffic_args`), so
`_batch_ticks_world` is True and `_apply_batch_sync()` calls
`client.apply_batch_sync(batch, True)` (`generate_traffic_v11.py:119-122`).

Probe against the live CARLA 0.10.0 server (sync mode, `fixed_delta_seconds=0.05`),
measuring `world.get_snapshot().frame` either side of each call:

| call | frame delta |
|---|---|
| `apply_batch_sync(batch, False)` | **0** |
| `apply_batch_sync(batch, True)` | **1** |
| `apply_batch_sync([], True)` | **1** (unreachable: `_apply_batch_sync` returns early on an empty batch) |
| walker body spawn batch, `do_tick=True` | **1** |
| walker controller spawn batch, `do_tick=True` | **1** |
| `world.spawn_actor(...)` | 0 |
| `actor.destroy()` | 0 |
| `controller.start()` / `go_to_location()` / `set_max_speed()` | 0 |
| `controller.stop()` | 0 |
| `world.get_actors(...)` | 0 |

Consequences:

1. **`apply_batch_sync(..., True)` is the only frame-advancing call anywhere in the population
   or janitor path.** `RoadblockJanitor.sweep()` / `clear_blocker()` use `actor.destroy()` and
   `set_transform()` and advance nothing. The manager is the sole hidden-tick source.
2. **The controller spawn batch itself ticks the world.** There *is* a world tick between body
   spawn and controller spawn, and between controller spawn and `controller.start()`. The prior
   diagnosis had the sign backwards: the defect is not a missing tick, it is an *unobserved* one.

## A2. Every `apply_batch_sync` reachable from `maintain_population()`

`maintain_population()` → `population.reconcile()` (`run_route_b_density_loop.py:1143-1159`).
All of the following pass `do_tick=True` and each tick the world exactly once when its batch is
non-empty (an empty batch is short-circuited and ticks nothing):

| # | site | call | condition |
|---|---|---|---|
| 1 | `_reap_orphan_controllers` `generate_traffic_v11.py:419` | DestroyActor × live orphans | ≥1 live orphan controller |
| 2 | `_spawn_missing_walker_controllers(self.walkers)` `:284` | SpawnActor(controller, parent) | ≥1 surviving body with `con is None` |
| 3 | `_spawn_vehicles(missing)` `:226` | SpawnActor+SetAutopilot | ≥1 vehicle deficit |
| 4 | `_spawn_walker_bodies_once` `:284`, up to `WALKER_SPAWN_ROUNDS=3` rounds | SpawnActor(walker) | ≥1 walker deficit; one tick **per round attempted** |
| 5 | `_spawn_missing_walker_controllers(new_walkers)` `:284` | SpawnActor(controller, parent) | ≥1 new body |

`_initialize_walker_controllers` (`start`/`go_to_location`/`set_max_speed`) ticks nothing.

Predicted hidden ticks for one lost walker:
- 2 = body round + controller attach (dead body took its controller with it)
- 3 = orphan reap + body round + controller attach
- 4 = orphan reap + 2 body rounds + controller attach
- 5 = orphan reap + 3 body rounds + controller attach

Predicted hidden ticks for a controller-only repair (no body loss, therefore **no**
"Detected population loss" log line): exactly **1** per reconcile — either the orphan reap or the
controller respawn.

## A3. Predicted vs observed — exact match

Run: `experiments/route_b_perception_v2/20260825_smoke3_traffic_30_30_fast` (30/30, Epic, fast
rasterizer, 200k PPS). `route_summary.json` reports `cadence.dropped_callback_frames = 59`.

`dropped_callback_frames` is incremented in `radar_sweep_aggregator_v1.py:182-183` as
`frame - last_frame - 1` over *ingested* radar callbacks, and the collector only ingests the
callback whose frame equals the frame returned by `SamplingWorld.tick()`
(`_drain_exact`, `run_route_b_perception_collection_v2.py:583-599`). So the metric counts
world frames that were advanced but never observed.

Independent reconstruction from `per_frame_density_counts`, which carries both `frame_id` and
`route_tick` per saved frame: `frame_id - route_tick` is constant unless a frame advances
without a route tick.

```
base offset 715 -> final offset 774   =>  total hidden ticks = 59
```

**59 predicted-mechanism hidden ticks = 59 reported dropped callback frames.** 42 discrete events:

| event size | count | ticks | interpretation |
|---|---|---|---|
| +2 | 2 | 4 | body + controller attach |
| +3 | 3 | 9 | orphan reap + body + controller attach |
| +4 | 3 | 12 | orphan reap + extra body round + controller attach |
| +1 | 34 | 34 | steady-state controller repair, one non-empty batch per reconcile |
| | **42** | **59** | |

The eight multi-tick events line up one-for-one with the eight
`WARNING:root:Detected population loss: vehicles=0 walkers=1` lines and with the eight
`population.deficit_spans` entries. Converting span start times to frames
(`frame = 716 + (t - 7.3655)/0.05`):

| deficit span start (sim s) | derived frame | observed hidden-tick event | size |
|---|---|---|---|
| 71.17 | 1992 | 2004 → 2012 | +3 |
| 97.57 | 2520 | 2536 → 2544 | +2 |
| 109.37 | 2756 | 2760 → 2768 | +3 |
| 118.17 | 2932 | 2948 → 2956 | +2 |
| 142.77 | 3424 | 3424 → 3432 | +3 |
| (sub-sample, between saved frames) | ~4434 | 4434 → 4444 | +4 |
| 198.47 | 4538 | 4542 → 4552 | +4 |
| 208.97 | 4748 | 4748 → 4758 | +4 |

The 34 single-tick events are spaced ~31 route ticks apart, which at the measured
152 ms/prepared-input wall cost is ~5 s wall — exactly `replenish_interval_s = 5.0`. They are
reconciles in which one and only one batch was non-empty, i.e. a walker whose controller kept
being orphaned and respawned. They produce **no** log line at all today.

Cross-check on the previous run
(`20260825_smoke2_traffic_30_30_CRASHED_SIGSEGV_diagnostic_only`, the client that exited 139):
manifest frame deltas are `{4: 1376, 8: 3}` → 3 events × 4 hidden frames = 12, and its log has
exactly 3 `Detected population loss` lines. Same mechanism, same arithmetic.

## A4. Failure-mode separation

**Hidden internal ticks — 59, all of them.** As above.

**Genuinely dropped sensor callbacks — zero.** `duplicate_callbacks = 0`,
`out_of_order_callbacks = 0`, `timestamp_reversals = 0`,
`sensor_alignment.max_timestamp_delta_s = 0.0`, `frame_content_failures = []`. A real dropped
radar callback would raise `missing radar callback at world frame N` in `on_world_tick`; it never
fired. The failed gates `callbacks_per_sweep_exact` / `window_callbacks_exact` /
`no_dropped_duplicate_or_reordered_callbacks` are all downstream of the same 59 hidden ticks —
the hidden frames' radar callbacks are discarded by `_drain_exact` as `observed < frame_id`, which
leaves the affected 100 ms sweeps holding 1 callback instead of 2.

**Native failure — client-side, both runs. The CARLA server did not die.**
- smoke3: the server process (PID 158162, `-quality-level=Epic -carla-rpc-port=2000`) is **still
  running and responsive** now, and it answered `get_server_version`, `get_actor`, `get_actors`
  and `apply_batch_sync` throughout the run's own cleanup path
  (`run_route_b_density_loop.py:1235-1240` gated cleanup on a live RPC and proceeded).
  What failed was `world.tick()` specifically, raising `RuntimeError("std::exception")` — first
  inside the route loop (`route_result` is `null`, so `drive_one_loop_with_traffic` raised), then
  again in `stop_sensors` (`sensor_cleanup.cleanup_tick = "error: std::exception"`). Meanwhile
  `apply_batch_sync(..., True)`, which ticks *server-side*, kept working — that is how the sensors
  and NPCs actually got destroyed afterwards. A tick path that fails while every other RPC
  succeeds is a **client-side** fault in `libcarla`'s tick, which includes client-side pedestrian
  navigation.
- The kernel log has exactly one fault in the whole window:
  `Mon Aug 24 19:46:32 2026 python3[139427]: segfault at 7ffd59f92fc8 ip ... sp 00007ffd59f92fb0
  error 6 in carla.cpython-310-x86_64-linux-gnu.so`. That is smoke2's **client** process, in the
  CARLA client extension module; the faulting address sits 0x18 above `rsp` immediately after a
  `sub $0x98,%rsp` prologue, i.e. a stack guard-page hit. There is **no** kernel fault at 20:06
  (smoke3) and none for the server at any time.
- Related, measured: destroying a *started* `controller.ai.walker` without `stop()` first returns
  `failed to destroy actor N : std::exception`; `stop()` then destroy succeeds. Destroying a
  walker body and leaving one started orphan controller did **not**, on its own, break 20
  subsequent ticks — so a single orphan is not sufficient; the fault correlates with sustained
  churn (8 replacements + 34 controller repairs) rather than one bad pair.

  Both native faults are in the CARLA **client** library and both correlate with mid-episode
  walker body/controller churn. The precise `libcarla` frame is not proven and is not claimed.

**Slow wall-clock execution — real, but not the cause and not hardware.**
`prepare_wall_clock_s` mean 152 ms / max 352 ms per 10 Hz input, so the episode runs at ~0.33×
real time. In synchronous mode slow wall-clock cannot create a frame gap. It does have one real
side effect worth recording: `replenish_interval_s` is gated on `time.monotonic()`, so 5 s wall
is only ~1.55 s simulated, and reconcile fires ~3× more often per simulated second than intended.
No hardware attribution is made or warranted.

## A5. Provenance gap found while auditing

When the route loop raises, `rows` is empty and `write_outputs()` returns at
`run_route_b_density_loop.py:1346-1351` without writing anything. The entire
`PopulationLedger.population_events` list — every LOST/REPLENISHED actor id — is discarded.
smoke3 has no `route_metrics_summary.json` at all; the eight stderr warnings are the only trace
that any replacement happened. Phase D must persist population events incrementally.

---

# Phase B — repair

Applied to `rl_agent/advisor_helper_scripts/codes/generate_traffic_v1.py`, which is the module
`run_route_b_density_loop.py:45` actually imports. `data_collection/generate_traffic_v11.py` is a
stale orphan copy that nothing imports and that **lacks the `_live_actor_map()` fix**; it was left
byte-identical to how it was found. Do not use it.

1. **One authoritative tick owner.** `begin_route_mode()` / `end_route_mode()` bracket the drive
   loop. In route mode `_batch_ticks_world` is forced False, `_apply_batch_sync` raises if it ever
   computes `do_tick=True`, and `_wait_for_actor_update()` is a no-op so nothing blocks waiting for
   a tick from the thread that owns ticking. `spawn_initial_population()` is unchanged, still ticks,
   and now refuses to run in route mode.
2. **Phased replacement.** Walker records carry `phase` + `phase_tick`. `note_route_tick()` runs
   once per observed route tick from `maintain_population()` and advances each pending record by at
   most one phase: body submitted → observed tick → controller attached → observed tick → controller
   started. Controller repair for a surviving body enters the same machine. Every transition is
   recorded with its body and controller id.
3. **Immediate invariant.** `SamplingWorld.tick()` requires `frame_id == previous + 1` and raises
   `TickOwnershipError` naming both frames, the gap, the route tick and the most recent population
   event. A gap is never reinterpreted as reduced effective Hz.
4. **Provenance.** Population events are written and flushed per event to
   `<experiment_dir>_population_events.jsonl`, a *sibling* of the output directory so the
   collector's create-only `mkdir` still guards the dataset.

# Phase C — lifecycle smoke: PASSED (16/16)

`route_b_perception_v2/population_lifecycle_smoke_v1.py`, Epic, full rig (3x 1280x720 cameras +
200,000 PPS radar, fast rasterizer), 8 walkers + 4 vehicles, 560 ticks, three walkers destroyed at
ticks 90 / 250 / 410.

```
raw_callbacks 580 == 20 warmup + 560 route ticks;  dropped 0  duplicate 0  out_of_order 0
332: body 121 -> controller 122 -> start 123
334: body 281 -> controller 282 -> start 283
336: body 441 -> controller 442 -> start 443
population 8/8, controllers_ready 8, pending_phases 0, orphans 0, clean teardown
```

# Phase D — one supervised traffic_30_30 collection: PASSED

`run_route_b_collection_supervised_v1.py` → `20260825_smoke5_traffic_30_30_tickfix`. Fresh Epic
server, no hybrid physics, 25 km/h, 200,000 PPS, fast rasterizer, 1029.6 s wall, client exit 0,
no native fault, server alive throughout.

```
status COLLECTION_EPISODE_PASSED       all 20 gates true      route completed, 1251.7 m
world_tick 20.0 Hz (6034)   logical_sweeps 10.0 Hz (3018)   prepared 10.0 Hz (3017)   saved 5.0 Hz (1509)
raw_callbacks 6054 == 20 warmup + 6034 route ticks
dropped 0   duplicate 0   out_of_order 0   timestamp_reversals 0
callbacks_per_sweep min=mean=max=2.0    window_callbacks min=mean=max=4.0
population 30/30, min 29, 6 losses / 6 replenishments, max deficit span 1.2 s
```

Independent reconstruction, the same statistic that exposed the 59 hidden ticks:
`frame_id - route_tick` takes **exactly one value (162)** across all 1509 saved frames →
**0 hidden ticks**.

All six mid-episode replacements were phased with exactly one observed route tick between stages:

```
body 146 -> 447 / 448 / 449      body 177 -> 3071 / 3072 / 3073
body 159 -> 2009 / 2010 / 2011   body 181 -> 4179 / 4180 / 4181
body 173 -> 2529 / 2530 / 2531   body 185 -> 5447 / 5448 / 5449
```

Zero deferrals: no attach or start ever had to be retried on a later tick.

## Residual, not introduced here

The controller-orphaning churn that produced the 34 single-tick reconciles in smoke3 still happens:
30 `controller_disowned` events, all reason `controller_absent`, 21 of them on one walker (body 99).
Each now costs one tick-free RPC round instead of a hidden world frame, so it no longer damages the
frame stream or the sweep accounting - but a walker whose `controller.ai.walker` repeatedly vanishes
is a separate, still-unexplained CARLA behaviour and is worth its own bounded investigation.

## Aborted attempt retained

`20260825_smoke4_ABORTED_output_dir_precreated` holds the 8.2 s first attempt, aborted before the
route began by a defect introduced in this repair: the population event stream was initially opened
inside the output directory, which pre-created it and tripped the collector's create-only `mkdir`.
Fixed by moving the ledger to a sibling path. Retained as provenance; it contains no route data.

---

# Follow-up (2026-08-24, post-Phase-D)

## 1. Reconcile schedule moved to simulated time

`run_route_b_density_loop.py` now gates reconciliation on the route loop's own
`sim_now_s` via `SimTimeReconcileSchedule`, so `replenish_interval_s = 5` means five *CARLA
simulated* seconds. The wall-clock gate made the real rate a function of machine speed: at the
measured ~0.17 s wall per 0.05 s tick, the 5 s wall gate fired roughly every 1.55 simulated
seconds, and that rate moved with load. The schedule advances by whole intervals so a stall cannot
queue a burst, and resets when simulated time restarts on a new loop. Each firing is written to the
population ledger as a `reconcile` event with its simulated time.

`log_population` remains wall-clock gated, unchanged, as requested. The two-tick phased replacement
mechanism is untouched.

## 2. Per-saved-frame controller-health telemetry

New per-saved-frame fields (in `route_summary.json` under `per_frame_controller_health` and
`per_frame_density_counts`, **never** in `manifest.csv`, the masks, or any model tensor):

`managed_walker_bodies_alive`, `live_attached_walker_controllers` (liveness *and* parent verified
against a world snapshot), `controllers_marked_ready`, `pending_body_phase`,
`pending_controller_phase`, `orphan_controllers`.

Episode summary `controller_health` adds `ready_controllers_min`, `frames_below_ready_floor` (95%
of bodies alive in that frame), `first_frame_below_ready_floor`, `deficit_spans` and
`max_controller_deficit_span_s`, plus two gates on the same `replenish_interval + 2 s` simulated
bound already used for body population:
`controllers_ready_95pct_every_saved_frame` and `no_controller_deficit_beyond_replenish_plus_2s`.

Registry-side counts come from a new read-only `TrafficPopulationManager.phase_summary()`, which
mutates nothing and issues no RPC.

## 3. Controller churn analysis — concentrated, not systematic

`route_b_perception_v2/controller_churn_analysis_v1.py` over the Phase D ledger
(`controller_churn_analysis_v1.json`):

```
managed bodies seen 36      bodies with zero disowns 30      affected fraction 0.167
controller_disowned events 30, all reason controller_absent
  body 99 -> 21   body 86 -> 2   body 98 -> 2   body 105 -> 2   body 146 -> 2   body 109 -> 1
body 99 share of all disowns: 0.70
controller lifetime (observed ticks): disowned  min 27 / median 29 / max 5241
                                      survived  count 30 / median 6014 (whole episode)
```

Body 99's timeline is the whole story. Its original controller lived 1741 ticks (87 s) normally.
It then entered a repair loop from tick 1741 to 2529: a freshly attached controller vanished after
27-31 ticks - almost exactly one reconcile period - 21 times in a row. The body itself was then
lost at tick 2529 and replaced cleanly through the normal phased path, which ended the loop. The
condition is self-terminating.

Excluding body 99: 9 disowns across 5 bodies, 1.8 each, over a 300 s episode with 36 bodies, and
five of those nine ended controllers that had already lived 249-5241 ticks. That is low-rate
background noise, not a systematic failure.

**Therefore no controller behaviour was changed**, per the stated condition. The new telemetry makes
the condition observable per frame if it recurs: a body in this state shows
`live_attached_walker_controllers` below `managed_walker_bodies_alive` and a growing
`controller_health.deficit_spans` entry.
