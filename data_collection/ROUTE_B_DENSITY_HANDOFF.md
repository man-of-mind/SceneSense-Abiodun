# Route B density runs — handoff

Status updated 2026-08-22 after the final seed-31 visual qualifications. Written
for whoever picks this up next. The final closure below supersedes conflicting
historical status/open-item text retained later in this document for provenance.

## Files

| File | State |
|---|---|
| `data_collection/routes/town10hd_opt_route_b_full_map_loop_v1.json` | **Accepted, frozen.** Do not regenerate. |
| `data_collection/routes/town10hd_opt_route_b_full_map_loop_v1.progress.csv` | **Accepted, frozen.** |
| `data_collection/run_route_b_ego_loop.py` | **Accepted, unchanged.** An `on_tick` hook was added mid-session and then reverted; the file is byte-identical to the approved version. |
| `data_collection/run_route_b_density_loop.py` | **Seed-31 density qualification complete — see final closure below.** |
| `rl_agent/advisor_helper_scripts/codes/generate_traffic_v1.py` | Untouched. |

## Final qualification closure — 2026-08-22

The accepted operating profiles for the planned measurement study are:

| density | NPC vehicles / pedestrians | recorded terminal | route | collision incidents | walker-brake ticks | ego block events | interventions | watchdog | cleanup | simulated duration |
|---|---:|---|---|---:|---:|---:|---:|---|---|---:|
| low | 5 / 5 | `PEDESTRIAN_BRAKING_NOT_EXERCISED` | 19/19, complete | 0 | 0 | 6 | 0 | no abort | succeeded | 359.25 s |
| medium | 15 / 15 | `PASS` | 19/19, complete | 0 | 22 | 7 | 0 | no abort | succeeded | 358.70 s |
| dense | 25 / 25 | `PASS` | 19/19, complete | 0 | 13 | 11 | 0 | no abort | succeeded | 419.90 s |

All three were fresh `Town10HD_Opt` worlds, one loop only, seed 31, lane offset
`-0.5 m`, walker detection distance 10 m, safe-car filtering, hardened NPC
Traffic Manager settings, and hybrid physics. Roadblock relocation and forced
ego overtaking were disabled. The ego recovered from observed traffic waits
without scenario intervention.

The low result must not be relabelled as a pedestrian-braking validation: no
pedestrian entered its detection path. It is a successful low-density route and
traffic qualification. Pedestrian braking was exercised and passed in the
medium and dense episodes.

Final runner behavior relevant to reuse:

- Every invocation reloads a fresh world and requires exactly `--loops 1`.
- Output CSV/JSON paths are explicit, create-only, and never appended to or
  overwritten.
- The no-progress watchdog is based on ego distance progress; a moving actor
  farther ahead cannot indefinitely reset it. Red-light waits remain exempt.
- Scenario interventions are disabled by default and would produce
  `INTERVENED`, not a clean pass.
- Collision logging retains actor type, simulation time, location, ego speed,
  brake activation, raw callback count, and distinct incident count.
- BasicAgent emergency braking uses `max_brake=1.0`. The qualified medium and
  dense runs with this setting had zero incidents after contacts were observed
  in earlier runs using the 0.5 setting.
- Cleanup first probes CARLA with a short timeout, restores the normal timeout
  when the server is alive, and uses the verified per-actor cleanup check.

Qualification artifacts:

- Low: `data_collection/route_b_density_validation/20260822_final_low_5_5_seed31/`
- Medium: `data_collection/route_b_density_validation/20260822_manual_medium_15_15_seed31/`
- Dense: `data_collection/route_b_density_validation/20260822_manual_dense_25_25_seed31/`

### Planned 864-run measurement study

The primary study is a controlled full factorial:

`72 AE/ROI/quantization profiles x 4 network profiles x 3 densities = 864`
fresh one-loop episodes.

Use `scenario_seed=31` for every primary-study cell. The seed is a fixed
scenario control, not another primary factor: changing it across individual
AE/network cells would confound traffic realization with the knobs being
measured. Every cell must still reload a fresh world. Pass the population counts
explicitly (`5/5`, `15/15`, or `25/25`) because the medium and dense accepted
counts override the runner's historical defaults. Record at least the AE/ROI/
quantization identifier, network-profile identifier, density, scenario seed,
and unique episode/output identifier with each result.

The 864-run matrix supports controlled metric-surface and reward-design work
conditional on the seed-31 scenario. Do not present it as cross-scenario
generalization evidence. After reward design and RL-agent shortlisting, reserve
seeds 32 and 33 for the finalists only; there is no present requirement to
repeat the complete 864-cell matrix for those seeds.

### Known non-blocking visual observation

Across low, medium, and dense visual runs, the ego sometimes waits at an
intersection slightly slanted within its lane after a lane transition rather
than settling perfectly straight. This did not cause a collision, route-order
failure, watchdog abort, intervention, or excessive final pose error in the
qualified episodes. It is consistent with the BasicAgent/local-planner
transition into the turn while using the accepted `-0.5 m` lane offset.

Do not tune the controller, route waypoints, or lane offset solely for this
cosmetic posture before the matched measurement study: doing so would change
the qualified trajectory. Revisit it only as a separately qualified controller
change if later camera-pose or intersection-orientation measurements prove
sensitive to the transient yaw.

The sections below are retained as historical evidence from before the final
runner and qualifications; the closure above is authoritative where they
conflict.

## Confirmed finding: the traffic helper is not the problem

`PythonAPI/examples/generate_traffic.py` (stock CARLA) and
`rl_agent/advisor_helper_scripts/codes/generate_traffic_v1.py` (advisor fork) have
**identical Traffic Manager configuration**: `set_global_distance_to_leading_vehicle(2.5)`,
`global_percentage_speed_difference(30.0)`, plain `SetAutopilot`, nothing per-vehicle.
The fork additionally does population maintenance. Switching to the stock script
would change nothing. Neither disables TM auto lane change, which is the main
source of NPC-on-NPC side-swipes; a shunted NPC never recovers and becomes a
permanent roadblock.

Separately confirmed: **`BasicAgent` has no pedestrian awareness at all** — the
strings `walker`/`pedestrian` do not appear in `basic_agent.py`. It checks only
vehicles and traffic lights. `BehaviorAgent` has a `pedestrian_avoid_manager`;
`BasicAgent` does not. The ego therefore drives through pedestrians.

## Changes made to `run_route_b_density_loop.py`

Verified working (medium 3/3 loops, exit 0):

1. **Safe-vehicle filter** (`--no-safe-vehicles` to disable). Keeps the 6
   `base_type == car` blueprints; excludes carlacola/firetruck/ambulance (trucks),
   fuso (bus), sprinter (van). `carlacola` was the truck in the reported deadlock.
2. **Per-NPC TM hardening** (`--no-npc-hardening` to disable): `auto_lane_change`
   off, random lane-change percentages 0, 4 m leading distance, speed −35%.
3. **Hybrid physics** (`--no-hybrid-physics`), ego `role_name = hero`, radius 70 m.
   NPCs beyond the radius cannot collide.
4. **Roadblock clearing** — a stationary NPC blocking the ego is relocated
   (`actor.destroy()` is unreliable; CARLA throws a bare `std::exception` on some
   actors, and dropping ownership after a failed destroy orphans a live wreck —
   that was the original permanent deadlock). Population manager replenishes.
5. **Progress watchdog** — aborts the loop cleanly with a full diagnostic
   (blocker, speed, throttle/brake, at_light, nearby actors) instead of hanging.
   Queuing behind moving traffic and red-light waits are excluded.

Implemented but **NOT validated** (added last, run was interrupted before any
loop completed):

6. **Pedestrian braking** (`--no-brake-for-walkers`, `--walker-brake-distance-m`,
   default 10 m). Uses `agent._vehicle_obstacle_detected(walker_list, reach)` —
   the same call `BehaviorAgent.pedestrian_avoid_manager` uses — and applies
   `agent.add_emergency_stop`. Counted per loop as `walker_brake_ticks`.
7. **Metric fix**: `npc_roadblocks_cleared` is a running total carried across
   loops, so the campaign figure is now the last value, not the sum of per-loop
   snapshots. The previously reported "110" for 3 medium loops was double-counted;
   the true figure was 42.

## Measured results

All headless, `--real-time-tick-period-s 0.002`, seed 31, `--lane-offset-m -0.5`.

**Medium, 3 loops, before item 6/7 (exit 0, all completed):**

| loop | driven m | sim s | collisions | ego blocks | roadblocks (cumulative) |
|---|---|---|---|---|---|
| 1 | 1252.0 | 539.1 | 2 | 16 | 31 |
| 2 | 1251.4 | 301.3 | 5 | 1 | 37 |
| 3 | 1251.9 | 240.1 | 0 | 1 | 42 |

Collisions across those 3 loops: `walker.pedestrian.0038` ×3,
`vehicle.dodge.charger` ×2, `static.road` ×2. The pedestrian hits are what
motivated item 6.

**Dense, 1 loop only** (loops 2–3 not run): completed, 1252.5 m, 357.7 s sim,
**27 collisions**, 6 ego blocks. Not trustworthy for collection yet.

**Low:** passed 1 loop earlier in the session, but that was before items 1–7.
Not re-verified since.

## Historical open items (superseded)

These items describe the pre-final runner and are closed or superseded by the
final qualification closure above.

1. **Validate pedestrian braking (item 6).** Never ran a single complete loop.
   Risk: an emergency stop for a stationary pedestrian could stall the ego until
   the 180 s watchdog. Run medium 2 loops and check `walker_brake_ticks > 0`,
   zero `walker.pedestrian.*` collisions, and no watchdog aborts.
2. **Re-verify low and medium** with the final code (items 6–7 changed the loop).
3. **Dense is unresolved.** 27 collisions in the one observed loop. Needs
   3+ loops and a look at what it is colliding with before dense is usable.
4. **Intervention rate is high** — 42 roadblock clears over 3 medium loops. The
   scenario is being actively managed. This is a data-provenance fact: per-loop
   counts are in the CSV (`npc_roadblocks_cleared`, `ego_block_events`,
   `ego_replans`, `walker_brake_ticks`). Decide whether that level of
   intervention is acceptable for the collection, or whether NPC counts should
   be lowered instead.
5. **`static.road` collisions ×2** in medium were not investigated.

## Historical commands (do not reuse)

These commands predate fresh-world enforcement, mandatory create-only output
paths, and the exactly-one-loop rule.

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun
V=/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3

# visual, one loop per density (CARLA must already be on Town10HD_Opt)
$V data_collection/run_route_b_density_loop.py --density low    --loops 1
$V data_collection/run_route_b_density_loop.py --density medium --loops 1
$V data_collection/run_route_b_density_loop.py --density dense  --loops 1

# headless fast validation (what the numbers above used)
$V data_collection/run_route_b_density_loop.py --density medium --loops 2 \
   --no-spectator --real-time-tick-period-s 0.002 \
   --out-csv /tmp/rb_med.csv --summary-json /tmp/rb_med.json
```

Headless CARLA for validation:
```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping
setsid ./CarlaUnreal.sh -RenderOffScreen -nosound -carla-rpc-port=2000 >/tmp/carla.log 2>&1 &
```

## Historical notes

- The runner appends to `route_b_density_loops.csv`, so the three profiles
  accumulate in one table. Delete it between campaigns if you want a clean run.
- Do not use `pgrep -f`/`pkill -f` with a pattern that appears in your own
  command line — it matches the invoking shell and kills it. Cost several
  wasted runs this session.
- Environment was left clean: no CARLA, no runner processes.
