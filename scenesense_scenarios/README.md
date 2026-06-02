# SceneSense Scenario Harness

This folder is Step 1 of the SceneSense scenario track.

The goal is to create repeatable CARLA scenes before touching fusion models,
training, or RL. A scenario run should answer:

- Which town, seed, anchor, actors, and weather were used?
- Where were vehicles and pedestrians spawned?
- Which pole/ego sensor placements should later be attached?
- Can we rerun the same scene and get the same layout?

## Scenarios

Current starter battery:

- `clear_low_density`: low object count, clear line of sight.
- `crowded_intersection`: more vehicles and pedestrians near the same anchor.
- `occlusion_static`: target and occluder candidates near the anchor.
- `occlusion_crossing_ego`: ego-facing blind-spot setup with a parked occluder and hidden target pedestrian.
- `intersection_truck_pedestrian_occlusion`: clean intersection occlusion with a parked truck/van surrogate and crossing pedestrian.
- `right_turn_truck_pedestrian_occlusion`: right-turn yield failure with a stopped truck/van queue hiding the crosswalk approach.
- `visible_crossing_failure`: positive-control failure where the crossing pedestrian is intentionally visible.
- `curbside_parked_vehicle_pedestrian_occlusion`: mid-block hidden pedestrian emerging from behind parked curbside vehicles.

## Quick Start

Start CARLA first, then run one scenario:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun

python3 scenesense_scenarios/scenesense_scenario_harness.py \
  --scenario clear_low_density \
  --seed 7 \
  --duration-s 60
```

List scenarios:

```bash
python3 scenesense_scenarios/scenesense_scenario_harness.py --list
```

Run the other starters:

```bash
python3 scenesense_scenarios/scenesense_scenario_harness.py \
  --scenario crowded_intersection \
  --seed 7 \
  --duration-s 60

python3 scenesense_scenarios/scenesense_scenario_harness.py \
  --scenario occlusion_static \
  --seed 7 \
  --duration-s 60
```

Use `--duration-s 0` to hold until Ctrl+C. Use `--keep-actors` only when you
want to inspect the scene after the script exits.

Run with ego-mounted front RGB/radar smoke-test sensors:

```bash
python3 scenesense_scenarios/scenesense_scenario_harness.py \
  --scenario occlusion_static \
  --seed 7 \
  --duration-s 60 \
  --ego-sensors \
  --ego-camera-preview
```

For a moving ego-view smoke test:

```bash
python3 scenesense_scenarios/scenesense_scenario_harness.py \
  --scenario crowded_intersection \
  --seed 7 \
  --duration-s 60 \
  --background-autopilot \
  --ego-autopilot \
  --move-pedestrians \
  --ego-sensors \
  --ego-camera-preview
```

Run the ego-facing occlusion crossing setup without scripted motion:

```bash
python3 scenesense_scenarios/scenesense_scenario_harness.py \
  --scenario occlusion_crossing_ego \
  --seed 7 \
  --duration-s 60 \
  --ego-sensors \
  --ego-camera-preview
```

Run the first failure-case motion pass:

```bash
python3 scenesense_scenarios/scenesense_scenario_harness.py \
  --scenario occlusion_crossing_ego \
  --seed 7 \
  --duration-s 60 \
  --ego-sensors \
  --ego-camera-preview \
  --scripted-ego-drive \
  --ego-drive-mode waypoint \
  --ego-route-choice left \
  --target-crossing
```

`--scripted-ego-drive` defaults to waypoint mode, which follows CARLA lane
waypoints. `--ego-drive-mode straight` is available only as a crude debug mode
and can drive into static roadside objects.

Run the cleaner intersection truck/pedestrian occlusion scene:

```bash
python3 scenesense_scenarios/scenesense_scenario_harness.py \
  --scenario intersection_truck_pedestrian_occlusion \
  --traffic-light-id 11 \
  --ego-route-choice left \
  --seed 7 \
  --duration-s 60 \
  --ego-sensors \
  --ego-camera-preview
```

Run the visible crossing failure-control pass:

```bash
python3 scenesense_scenarios/scenesense_scenario_harness.py \
  --scenario visible_crossing_failure \
  --traffic-light-id 11 \
  --seed 7 \
  --duration-s 60 \
  --ego-sensors \
  --ego-camera-preview \
  --scripted-ego-drive \
  --ego-drive-mode waypoint \
  --ego-route-choice left \
  --ego-target-speed 4.5 \
  --target-crossing \
  --target-crossing-delay-s 1.0 \
  --target-crossing-speed 1.8 \
  --target-crossing-trigger-distance-m 18.0 \
  --spectator-focus conflict
```

Run the occluded intersection failure-case pass:

```bash
python3 scenesense_scenarios/scenesense_scenario_harness.py \
  --scenario intersection_truck_pedestrian_occlusion \
  --traffic-light-id 11 \
  --seed 7 \
  --duration-s 60 \
  --ego-sensors \
  --ego-camera-preview \
  --scripted-ego-drive \
  --ego-drive-mode waypoint \
  --ego-route-choice left \
  --ego-target-speed 4.5 \
  --target-crossing \
  --target-crossing-delay-s 1.0 \
  --target-crossing-speed 2.2 \
  --target-crossing-trigger-distance-m 18.0 \
  --stop-on-target-collision \
  --spectator-focus conflict
```

Run the cleaner right-turn hidden-pedestrian failure pass:

```bash
python3 scenesense_scenarios/scenesense_scenario_harness.py \
  --scenario right_turn_truck_pedestrian_occlusion \
  --traffic-light-id 11 \
  --seed 7 \
  --duration-s 60 \
  --ego-sensors \
  --ego-camera-preview \
  --scripted-ego-drive \
  --ego-drive-mode waypoint \
  --ego-route-choice right \
  --ego-target-speed 4.5 \
  --target-crossing \
  --target-crossing-delay-s 1.0 \
  --target-crossing-speed 2.2 \
  --target-crossing-trigger-distance-m 12.0 \
  --stop-on-target-collision \
  --spectator-focus conflict
```

Run the curbside parked-vehicle hidden-pedestrian candidate:

```bash
python3 scenesense_scenarios/scenesense_scenario_harness.py \
  --scenario curbside_parked_vehicle_pedestrian_occlusion \
  --anchor-source spawn_point \
  --anchor-spawn-index 152 \
  --ego-spawn-index 152 \
  --seed 7 \
  --duration-s 60 \
  --ego-sensors \
  --ego-camera-preview \
  --scripted-ego-drive \
  --ego-drive-mode waypoint \
  --ego-route-choice straight \
  --ego-target-speed 4.2 \
  --target-crossing \
  --target-crossing-delay-s 1.0 \
  --target-crossing-speed 1.8 \
  --target-crossing-control-speed 12.0 \
  --target-crossing-trigger-distance-m 14.0 \
  --curbside-target-start-lateral-offset-m 4.2 \
  --curbside-target-end-lateral-offset-m 0.4 \
  --curbside-heavy-occluder-first \
  --helper-vehicle \
  --helper-drive \
  --helper-target-speed 1.5 \
  --helper-stop-distance-to-conflict-m 5.0 \
  --helper-camera-preview \
  --evidence-pack \
  --stop-on-target-collision \
  --post-target-collision-hold-s 3.0 \
  --spectator-focus conflict
```

For Month 1, treat this as the canonical hidden-pedestrian dart-out
scenario. The prewalk flags exist for later demo polish, but the current
CARLA pedestrian controller can make sidewalk prewalks look awkward or route
through nearby parked vehicles. The optional helper vehicle is an opposite-lane
observer camera for checking whether another viewpoint can see the hidden
pedestrian earlier than the ego camera; `--helper-drive` makes that viewpoint
move slowly through the opposite lane and past the scene instead of remaining
parked or participating in the ego-pedestrian collision.
`--evidence-pack` adds an `evidence/` folder with actor ground-truth traces,
event-window CSVs, and buffered ego/helper RGB frames around the collision.

Scout cleaner non-intersection curbside anchors first:

```bash
python3 scenesense_scenarios/scout_curbside_spawn_anchors.py \
  --top 20
```

The scout writes `curbside_spawn_candidates.csv` and a Markdown file with trial
commands. Try the first few candidates visually and prefer the one that looks
like a neighborhood/rural curbside road rather than an intersection.
If an additional town is installed, add `--load-town --town Town07_Opt` (or the
desired map name). If CARLA cannot load that map, the scout falls back to the
current world and prints the reason.

For both crossing-failure scenarios, the target pedestrian is no longer started by timer alone.
After the delay, it starts moving when the ego vehicle is close to the
computed conflict point. The visible control confirms that the route/timing
can produce a collision. The occluded version tries to use a bus/truck/van as
a stopped queue, as if that approach has a red light, between the ego camera
and the pedestrian start point so the same failure becomes a hidden-hazard
case. If the CARLA build does not expose those blueprints, it falls back to
regular vehicles and requests a denser stopped-queue layout. Check
`occluder_blueprint_id`, `occluder_heavy_blueprint_available`, and
`occluder_heavy_candidate_ids` in `scenario_manifest.json` to confirm whether
the run used a large vehicle or the multi-car fallback. For the right-turn
scenario, interpret the occluder as a stopped queue or service vehicle in an
adjacent lane, not as curb parking; this avoids the no-parking-zone ambiguity.
The right-turn layout also tries to choose a conflict point near CARLA
crosswalk geometry and records `conflict_crosswalk_gap_m` when available.
The occluded failure
commands intentionally use a later crossing trigger than the visible control:
the target should stay hidden until the ego is near the conflict point, not
finish crossing while the ego is still far away. The right-turn scenario is the
cleaner final-demo candidate because it matches a vehicle turning right while
yielding to a crossing pedestrian hidden by a stopped truck or queue.
If you are copying commands from an older scout output, replace
old right-turn commands that used slow walker control with the updated command
above. The harness records both `target_crossing_speed_mps` and
`target_crossing_control_speed` so pedestrian-control tuning remains visible in
the run metadata.
The curbside scenario uses a short deterministic crossing rather than CARLA's
walker navigation controller, because the navigation controller can route the
pedestrian along the sidewalk instead of straight across the lane. Treat it as
the stronger reviewer-facing candidate if the intersection geometry continues
to look artificial.

Confirm the run using `scenario_event_summary.json`:

- `target_collision_count > 0`: ego hit the target pedestrian.
- `target_danger_event == true`: ego came within the near-miss threshold.
- `min_target_distance_m <= target_near_miss_threshold_m`: numerical near miss.
- `ego_route_index == ego_preplanned_route_points - 1` with no danger event:
  ego probably finished the scripted route and stopped normally.

The event monitor also writes `scenario_event_trace.csv`, which contains
ego-target distance, ego-conflict distance, target-start reason, and ego route
index per tick.

If the selected traffic-light anchor produces an unnatural layout, scout better
intersection anchors:

```bash
python3 scenesense_scenarios/scout_intersection_anchors.py --top 20
```

For the right-turn hidden-pedestrian demo, use the more targeted scout:

```bash
python3 scenesense_scenarios/scout_right_turn_occlusion_anchors.py --top 20
```

This scout ranks anchors by crosswalk proximity, whether the truck/queue stays
on a straight approach, and whether the occluder lands near a driving lane. It
writes `right_turn_occlusion_candidates.csv` and a Markdown file with trial
commands for the top candidates.

The scout writes ranked candidates under `metrics_logs/scenesense_scenarios/`
and prints traffic-light ids plus route choices. Use a good candidate with
`--traffic-light-id` and `--ego-route-choice`.

## Outputs

Each run writes:

```text
metrics_logs/scenesense_scenarios/<timestamp>_<scenario>_seed<seed>/
  scenario_manifest.json
  actors.json
  summary.txt
  ego_sensor_summary.json   # only when --ego-sensors is used
  scenario_event_summary.json # only for occlusion event runs
  evidence/                 # only when --evidence-pack is used
```

The manifest records:

- scenario name and seed
- CARLA map/town and weather
- traffic-light/static anchor
- requested vs spawned actor counts
- actor ids, type ids, roles, transforms, and bounding boxes
- optional ego front RGB/radar smoke-test configuration and frame counts
- optional occlusion event layout, target crossing, closest distance, and collision events
- optional evidence pack with actor ground truth and sampled ego/helper RGB frames
- suggested pole sensor placements for later fusion runs

## Step Boundary

This harness does not run the model, collect training data, or evaluate task
quality yet. Next steps are:

1. Attach RGB/radar/semantic sensors. Basic ego RGB/radar smoke-test support is available with `--ego-sensors`.
2. Save synchronized data and CARLA ground truth.
3. Run the existing fusion model as a baseline.
4. Create the parked ego variant.
5. Fine-tune/retrain only after the new data path is verified.
