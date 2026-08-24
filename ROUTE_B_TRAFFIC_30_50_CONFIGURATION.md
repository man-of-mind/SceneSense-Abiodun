# Route B — traffic-light inspection and the 30/30 and 50/50 traffic profiles

Date: 2026-08-24. Scope: inspection and configuration only. No perception inference,
no retraining, no OAI, no campaign was run.

Accepted evidence this builds on:
`data_collection/route_b_density_validation/20260824_manual_very_dense_50_50_seed31/`
— Route B completed in 419.85 s, 50 vehicles / 50 pedestrians requested and **50 alive
at completion for both**, 0 roadblock removals, 0 interventions, 0 collisions, B1/B2/B3
complete. CARLA did not remove the population; that is confirmed below, not assumed.

Inspection artifacts (read-only, nothing in the map was modified):
`data_collection/route_b_density_validation/20260824_traffic_light_inspection/`

---

## 1. Original traffic-light timings (recorded before any change)

`Town10HD_Opt` contains **15 traffic lights in total**, and every one of them lies within
30 m of the driven Route B path — Route B is a full-map loop, so "lights associated with
Route B" is the whole set. Distances below are to the Route B `planned_path` polyline.

Configured values are `get_red_time()` / `get_yellow_time()` / `get_green_time()`.
"Observed red" is the measured dwell in `Red`, from a 180 simulated-second synchronous
readback of an **empty** world (no ego, no NPCs), which is the effective red an approach
sees once CARLA's group cycling is included.

| actor ID | group (key / poles) | OpenDRIVE junction | pole | world x, y, z | red (s) | yellow (s) | green (s) | observed red (s) | dist to Route B path (m) | nearest ordered wp | long-queue approach |
|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 34 | 34 / 4 | 943 | 0 | -46.17, -73.56, 0.25 | 2.00 | 3.00 | 10.00 | 32.25 | 5.08 | 14 | no |
| 35 | 34 / 4 | 944 | 2 | -34.45, -51.02, 0.25 | 2.00 | 3.00 | 10.00 | 32.25 | 17.33 | 14 | no |
| 36 | 36 / 4 | 958 | 1 | 114.45, 21.20, 0.25 | 2.00 | 3.00 | 10.00 | 32.25 | 4.73 | 1 | **yes — leg-11, 96 s** |
| 37 | 36 / 4 | 957 | 0 | 89.79, 20.92, 0.25 | 2.00 | 3.00 | 10.00 | 32.25 | 7.60 | 1 | no |
| 38 | 38 / 5 | 950 | 2 | -31.93, 20.30, 0.25 | 2.00 | 3.00 | 10.00 | 47.35 | 7.79 | 17 | no |
| 39 | 38 / 5 | 952 | 0 | -64.26, 7.06, 0.25 | 2.00 | 3.00 | 10.00 | 47.35 | 9.44 | 16 | no |
| 40 | 38 / 5 | 951 | 1 | -62.35, 20.20, 0.25 | 2.00 | 3.00 | 10.00 | 47.35 | 3.57 | 16 | no |
| 41 | 41 / 4 | 961 | 2 | -94.93, 20.33, 0.25 | 2.00 | 3.00 | 10.00 | 32.25 | 3.84 | 4 | no |
| 42 | 34 / 4 | 945 | 1 | -59.24, -51.50, 0.25 | 2.00 | 3.00 | 10.00 | 32.25 | 17.11 | 14 | no |
| 43 | 41 / 4 | 962 | 1 | -119.24, 19.00, 0.25 | 2.00 | 3.00 | 10.00 | 32.25 | 9.81 | 4 | no |
| 44 | 44 / 3 | 953 | 0 | -58.72, 145.82, 0.25 | 2.00 | 3.00 | 10.00 | 17.15 | 5.47 | 8 | no |
| 45 | 44 / 3 | 954 | 1 | -62.10, 123.76, 0.28 | 2.00 | 3.00 | 10.00 | 17.15 | 16.57 | 8 | no |
| 46 | 38 / 5 | 949 | 3 | -31.64, 33.78, 0.25 | 2.00 | 3.00 | 10.00 | 47.35 | 5.68 | 17 | **yes — leg-16, 48 s** |
| 47 | 41 / 4 | 960 | 0 | -119.23, 5.09, 0.26 | 2.00 | 3.00 | 10.00 | 32.25 | 8.93 | 4 | no (stop line 15.6 m from the leg-16 block but on road 18, a different approach) |
| 48 | 36 / 4 | 959 | 2 | 115.45, 35.04, 0.22 | 2.00 | 3.00 | 10.00 | 32.25 | 5.82 | 1 | no |

### Min / median / max across the Route B lights (= all 15)

| quantity | min | median | max |
|---|---:|---:|---:|
| configured red | 2.00 | 2.00 | 2.00 |
| configured yellow | 3.00 | 3.00 | 3.00 |
| configured green | 10.00 | 10.00 | 10.00 |
| configured cycle (R+Y+G) | 15.00 | 15.00 | 15.00 |
| **observed effective red** | 17.15 | 32.25 | 47.35 |
| observed green | 10.00 | 10.00 | 10.00 |

**Configured timing variance is exactly zero.** All 15 lights are 2 / 3 / 10 s. The
observed red spread is entirely explained by group size — there are five groups, of
3, 4 and 5 poles (keys 44, {34, 36, 41}, 38), and effective red is
`15 x (poles - 1) - 13`: 17.15 s, 32.25 s, 47.35 s. That is stock CARLA group cycling
over identical per-pole values, not per-light misconfiguration.

### The two approaches where the ego actually waited

The accepted 50/50 run recorded 12 ego block events at exactly two locations, totalling
~144 s — 34% of the 419.85 s episode. Mean speed over the loop was 2.98 m/s.

| block | location | duration | in-lane signal ahead | that signal's effective red | ratio |
|---|---|---:|---|---:|---:|
| leg 11 | (109.39, 69.66), inside junction road 675 | 96 s (8 x 12 s, sim 171.7–267.7) | light **36**, group 36, 4 poles, stop line 21.2 m ahead | 32.25 s | **~3.0 cycles** |
| leg 16 | (-72.86, 27.98), road 19 lane -2 | 48 s (4 x 12 s, sim 363.9–411.9) | light **46**, group 38, 5 poles, stop line **2.77 m** ahead | 47.35 s | **~1.01 cycles** |

- **leg 16** is a textbook single red phase: the ego stopped 2.77 m behind the stop line
  and waited 48 s against a 47.35 s effective red.
- **leg 11** spans roughly three consecutive cycles of a 32.25 s red. The immediate
  obstruction was NPC **197** (`vehicle.dodgecop.charger`), stationary 2.7 m ahead of the
  ego. The roadblock janitor detected it at sim 144.4 s and **deliberately left it in
  place** because `--allow-scenario-interventions` is off. 33 such stationary-NPC
  observations were logged and 0 were cleared.

---

## 2. Which rule applied

**Rule 2 applies. Traffic-light timings are not changed.**

Rule 1's trigger — a Route B light whose red exceeds 1.5x the Route B median red —
**cannot fire**: the median configured red is 2.00 s and the maximum configured red is
also 2.00 s, so the ratio is 1.00 for every light. There is no anomalous light to replace.
Applying Rule 1 anyway would mean inventing a "fixed replacement cycle" identical to the
values already present, which is a code change with no effect.

The long waits are **ordinary queue saturation across multiple signal cycles**, plus one
uncleared stationary NPC at leg 11 that the runner is configured by design not to remove.
No fixed timing was selected and no `set_red_time` / `set_yellow_time` / `set_green_time`
call was added anywhere.

---

## 3. Population-accounting change

Confirms from actor evidence — rather than assuming — whether CARLA removed anything.

Changed file: `data_collection/run_route_b_density_loop.py` (+150 / -6). No new framework,
no schema, no manifest, no unit tests. `generate_traffic_v1.py` is **untouched**; the new
accounting observes the existing population manager from the caller side.

- New `PopulationLedger` class. It snapshots the owned vehicle-ID and walker-body-ID sets
  either side of every `population.reconcile()` call, so a replacement is never miscounted
  as a survivor. The minimum live count is read from the surviving set **before** the
  refill, which is the true trough rather than the post-refill number.
- `maintain_population()` now takes the episode simulated time, so every loss and
  replenishment carries a `sim_s` timestamp and the actor ID.
- New per-loop CSV columns (`route_b_density_validation/<run>/<name>.csv`):
  `npc_vehicles_spawned`, `npc_pedestrians_spawned`, `npc_vehicles_live_min`,
  `npc_pedestrians_live_min`, `npc_vehicles_lost`, `npc_pedestrians_lost`,
  `npc_vehicles_replenished`, `npc_pedestrians_replenished`, `respawn_dormant_enabled`,
  `target_speed_kph`. The existing `npc_vehicles_requested` / `npc_pedestrians_requested`
  and `npc_vehicles_live` / `npc_pedestrians_live` (alive at completion) are unchanged.
- New summary-JSON fields: the same totals plus a `population_events` list of
  `{sim_s, actor_kind, action: LOST|REPLENISHED, actor_id}`. Empty list means no churn.
- Roadblock removals and interventions were already reported
  (`npc_roadblocks_cleared`, `roadblock_relocations`, `roadblock_destructions`,
  `intervention_count`, `intervention_events`, `roadblock_observations`) and are unchanged.
- **Dormant respawning stays disabled.** `--respawn-dormant` default is still `False`;
  it is now recorded explicitly in both outputs. There is no evidence it is needed — the
  accepted 50/50 run finished with 50/50 alive.
- Replenishment is reported, not suppressed: a non-zero `npc_vehicles_lost` means the
  episode did not run on the actors it started with, and that fact reaches the artifact.

Traffic profiles added to `DENSITY_PROFILES`:

| `--density` value | vehicles | pedestrians |
|---|---:|---:|
| `traffic_30_30` | 30 | 30 |
| `traffic_50_50` | 50 | 50 |

These names state the **requested actor counts only**. Actual scene density must be
measured per frame as local and in-view counts; a spawn request cannot assert it. The
historical `low` / `medium` / `dense` names are retained solely so the accepted
2026-08-22 qualifications stay reproducible, and must not be used as labels in new
outputs. The `density` column in new runs carries the value `traffic_30_30` or
`traffic_50_50`.

Verified: `py_compile` passes; the ledger's loss / replenishment / trough arithmetic and
event log were exercised against a synthetic population; `--target-speed-kph` default is
still **25.0** and `--respawn-dormant` default is still **False**.

---

## 4. Visual qualification commands (one loop each, 30 km/h)

Ego target speed is passed **explicitly** as `--target-speed-kph 30` (= 8.33 m/s). The
runner's global default stays 25 km/h so historical Route B evidence remains reproducible.

Everything else is the accepted default and is deliberately not passed: seed 31,
lane offset -0.5 m, walker brake distance 10 m, NPC hardening on, safe-vehicle filtering
on, interventions disabled, hybrid physics on, TM port 8010, spectator on, default weather.
Each command reloads a fresh `Town10HD_Opt` world and runs exactly one loop.

Start CARLA **with rendering** (not `-RenderOffScreen`) so the loop can be watched:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping
setsid ./CarlaUnreal.sh -nosound -carla-rpc-port=2000 >/tmp/carla_route_b_visual.log 2>&1 &
```

Then, in another shell:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun
V=/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3
```

**Loop 1 — traffic_30_30 at 30 km/h**

```bash
$V data_collection/run_route_b_density_loop.py \
  --density traffic_30_30 \
  --target-speed-kph 30 \
  --loops 1 \
  --seed 31 \
  --out-csv       data_collection/route_b_density_validation/20260824_traffic_30_30_seed31_30kph/traffic_30_30_seed31_30kph.csv \
  --summary-json  data_collection/route_b_density_validation/20260824_traffic_30_30_seed31_30kph/traffic_30_30_seed31_30kph.json
```

**Loop 2 — traffic_50_50 at 30 km/h** (fresh world; run after loop 1 finishes)

```bash
$V data_collection/run_route_b_density_loop.py \
  --density traffic_50_50 \
  --target-speed-kph 30 \
  --loops 1 \
  --seed 31 \
  --out-csv       data_collection/route_b_density_validation/20260824_traffic_50_50_seed31_30kph/traffic_50_50_seed31_30kph.csv \
  --summary-json  data_collection/route_b_density_validation/20260824_traffic_50_50_seed31_30kph/traffic_50_50_seed31_30kph.json
```

### Exact new output paths

```
data_collection/route_b_density_validation/20260824_traffic_30_30_seed31_30kph/traffic_30_30_seed31_30kph.csv
data_collection/route_b_density_validation/20260824_traffic_30_30_seed31_30kph/traffic_30_30_seed31_30kph.json
data_collection/route_b_density_validation/20260824_traffic_50_50_seed31_30kph/traffic_50_50_seed31_30kph.csv
data_collection/route_b_density_validation/20260824_traffic_50_50_seed31_30kph/traffic_50_50_seed31_30kph.json
```

Both directories are new. Output paths are create-only — the runner refuses to append to
or overwrite an existing file, so every accepted Route B artifact, including
`20260824_manual_very_dense_50_50_seed31/`, is preserved untouched.

A pedestrian contact should be logged but is not automatically blocking for this
perception-oriented qualification, provided the route completes, sensors and actors stay
valid, no intervention occurs, and cleanup succeeds. Pedestrian-controller logic was not
modified.

---

## 5. Remaining risk

1. **30 km/h buys less than it looks like.** Only the moving fraction shortens. At the
   accepted 1251.7 m driven distance, the free-flow floor moves from 180 s (25 km/h) to
   150 s (30 km/h) — about 30 s, ~7% of the 419.85 s episode — because roughly 240 s was
   waiting, and signal and queue waiting is unchanged. Do not expect a large speed-up.
2. **The leg-11 obstruction can recur.** The 96 s block was a stationary NPC the janitor
   detected and was configured not to remove. Seed 31 makes it repeatable, and 30 vs 50
   NPCs changes which vehicle lands there. Watch `roadblock_observations` and
   `ego_block_events` in both runs.
3. **The 900 s per-loop simulated budget is the real ceiling.** The accepted run used
   419.85 s of it. 30/30 should be well inside; if 50/50 plus a bad leg-11 obstruction
   pushes past 900 s the loop aborts cleanly with `watchdog_aborted`. This is a reporting
   risk, not a silent one.
4. **50 pedestrians raises contact likelihood.** `BasicAgent` has no native pedestrian
   awareness; the 10 m walker-brake wrapper is what protects the ego, and it was
   deliberately left unmodified. Expect `walker_brake_ticks > 0` and treat a logged
   contact per the non-blocking rule above.
5. **The dwell readback was taken on an empty world.** It measures the signal program
   correctly, but real queue discharge under 50 NPCs can leave a vehicle stopped across
   more than one green — that is precisely the saturation Rule 2 names, and it is not
   fixable by retiming.
6. **`traffic_30_30` / `traffic_50_50` are requested counts, not measured density.** No
   per-frame local or in-view density measurement is added by this task; the profile name
   must not be quoted as a scene-density result.
7. **Not yet run.** These two loops have not been executed. Nothing here is qualification
   evidence until both complete and their artifacts exist.
