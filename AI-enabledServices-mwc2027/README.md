# AI-enabled Services MWC 2027 CARLA demo

This directory contains the self-authored scripts, configuration, and saved
routes used for the Town10HD Physical AI demo. It deliberately does **not**
vendor the CARLA simulator, CARLA Python wheel, map assets, or CARLA's stock
navigation agents. Place this checkout directly below CARLA's `PythonAPI/`
directory so the scripts can resolve the adjacent `PythonAPI/carla/agents`
package.

## Included files

| File | Purpose |
| --- | --- |
| `pedestrian_head_camera_client_v13.py` | Keyboard-controlled ego pedestrian, head camera, route record/replay, warnings, boxes, virtual spatial-sensor map, network-zone visualization, and metrics |
| `manual_control_ar_v13.py` | Current keyboard-controlled ego vehicle with saved-route guidance, virtual camera/radar FoVs, 2-D cooperative occlusion reasoning, multi-site recovery, gated rogue-pedestrian warnings, network-zone behavior, and metrics |
| `cooperative_occlusion_v1.py` | Pure geometry helper for bounded FoVs, occlusion shadows, target-footprint line-of-sight tests, and ego/peer visibility fusion |
| `test_manual_control_ar_v13_occlusion.py` | Offline regressions for FoV geometry, partial-footprint visibility, occluders, cooperative reveals, warning authorization, and virtual-site selection |
| `spatial_map_coop/MANUAL_CONTROL_AR_V13_COOPERATIVE_OCCLUSION.md` | Detailed design and calculation guide for the v13 cooperative-occlusion demonstration |
| `manual_control_ar_v12.py` | Retained previous ego-vehicle client without the v13 cooperative visibility gate |
| `spawn_blocker_v5.py` | Static blockers, reactive pedestrians, recurring lane-following rogue vehicle, and shared network-degradation metadata; it does not spawn spatial-map RGB/radar sensors |
| `test_spawn_blocker_network_profile.py` | Offline fake-CARLA regression tests for transactional profile publication, rollback, reuse, and safe residue recovery |
| `network_degradation_profile_v1.py` | Shared transactional world-metadata schema and discovery helper for the two configurable network-degradation zones |
| `physical_ai_scenario_controller_ui_v2.py` | Vehicle route authoring UI |
| `physical_ai_scenario_controller_ui_v3.py` | Vehicle and pedestrian route authoring UI |
| `physical_ai_scenario_controller_ui_v1.py` | Shared base required by both route-authoring UIs |
| `traffic_light_pole_camera_ui_client_v1.py` | Shared traffic-light/pole map helper required by the UIs |
| `ego_route_config.py` | Vehicle route schema, validation, and atomic I/O |
| `pedestrian_route_config.py` | Pedestrian route schema, validation, and atomic I/O |
| `physical_ai_scenario_config_v2.yaml` | Town10HD scenario-controller preset |
| `traffic_lights_data.json` | Town10HD traffic-light/pole metadata |
| `my_ego_route.json` | Saved ego-vehicle route used by the demo command |
| `ego_pedestrian_route_v1.json` | Saved pedestrian route authored by the v3 UI |
| `recorded_pedestrian_route_1.json` | Keyboard-recorded pedestrian route used by replay |
| `SOURCE_MANIFEST.json` | Repository destinations, source mappings, and SHA-256 hashes for this import |
| `archive/v9/`, `archive/v11/`, `archive/v5/` | Deprecated clients and the prior physical-spatial-sensor blocker snapshot, retained for source-history comparison only |

## Prerequisites

- CARLA 0.10.0 with `Carla/Maps/Town10HD_Opt` loaded.
- Python 3.10 and the CARLA 0.10.0 Python module.
- A graphical desktop for the Pygame RGB view and OpenCV map/metrics windows.
- The stock CARLA `PythonAPI/carla/agents` tree. The repository should be at:

  ```text
  Carla-0.10.0-Linux-Shipping/
  `-- PythonAPI/
      |-- carla/
      `-- AI-enabledServices-mwc2027/
  ```

The tested interpreter is:

```bash
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python
```

The system `/usr/bin/python3` on the source host does not have the CARLA module
or all navigation dependencies. Use the CARLA virtual environment:

```bash
source /home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` pins the versions verified on the source host. CARLA itself
is intentionally an external prerequisite because its wheel must match the
simulator release and Python ABI.

## Clock ownership

Only one process may own a synchronous CARLA world clock.

- `pedestrian_head_camera_client_v13.py` and `spawn_blocker_v5.py` are passive:
  they never load a map, change world settings, or call `world.tick()`.
- `manual_control_ar_v13.py` owns the clock only when launched with `--sync`.
  The command below omits it; an existing master must advance a synchronous
  world.
- The supplied scenario-controller preset has `world.master_clock: true` and
  makes either v2 or v3 the sole 20 Hz master while that UI is running. Use the
  route-authoring UIs one at a time and do not run another tick owner beside
  them.

The route files and traffic-light metadata are map-specific. The documented
workflows require `Town10HD_Opt`; the route validators intentionally reject a
different map, including `Town10HD`.

## Run the blocker scenario

Run this in a separate terminal. In asynchronous mode CARLA advances frames
itself, so the blocker does not require another client. If the world is already
synchronous, an existing clock owner must advance snapshots; this passive
script never changes world settings or calls `world.tick()`:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/AI-enabledServices-mwc2027

/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python \
  spawn_blocker_v5.py \
  --vehicle-blueprint vehicle.nissan.patrol \
  --impact-target center \
  --reactive-impact-settle-time 0.25 \
  --reactive-contact-hold-time 5.0 \
  --reactive-contact-stop-speed 0.25 \
  --reactive-launch-margin 0.50 \
  --reactive-approach-hold-distance 1.50 \
  --reactive-approach-hold-timeout 30.0
```

The rogue vehicle follows a preplanned Driving-lane route, dynamically latches
the pedestrian's crossing station, stages upstream when the pedestrian ETA is
late or unavailable, and continues through the route after a miss. A confirmed
contact is allowed to settle physically and remain visible for the configured
hold before the actor is retired and respawned.

For physical pedestrian ragdolls, the process that owns CARLA world settings
must set `deterministic_ragdolls=False`. The blocker is intentionally passive
and only warns when deterministic ragdolls are enabled.

The blocker publishes a shared display-model network-degradation profile. Its
two zones default to:

- Ego vehicle: `(8.827, 62.216)`, radius `18.0 m`.
- Ego pedestrian: `(90.390, 43.290)`, radius `18.0 m`.

Append any of these controls to the blocker launch command:

```bash
# Customize both network degraded zones
--ego-vehicle-network-degradation-zone X Y RADIUS
--ego-pedestrian-network-degradation-zone X Y RADIUS

# Disable independently
--disable-ego-vehicle-network-degradation-zone
--disable-ego-pedestrian-network-degradation-zone

# Disable both
--disable-network-degradation

# Opt in to real subscriptions (legacy no-op in this version)
--start-active-spatial-map-sensors
```

Disabling both zones still publishes a zero-zone manifest so display clients
receive an explicit shared disabled state. The profile uses invisible,
parentless `sensor.other.gnss` metadata actors. They are never listened to and
do not collect sensor data. Zone records are created first and a versioned
manifest is created last; owned teardown removes the manifest before its zones.

At startup, an existing complete profile with exactly the requested settings
is reused without taking ownership. Manifest-free residue is removed
automatically only when every visible record exactly matches a requested zone,
all records use one session token, and the actor set remains stable for at
least 1.5 seconds across at least three externally driven snapshots. Other
partial, incompatible, or ambiguous profiles are preserved and startup fails
closed. Use `--replace-existing-network-profile` only when those metadata
actors are known to be stale. In a paused synchronous world with no external
snapshots, automatic residue cleanup is not attempted.

Despite its historical name, `--start-active-spatial-map-sensors` does not opt
in to real subscriptions in this virtual-sensor revision. It is a deprecated
compatibility no-op: no camera/radar actors are spawned, no `listen()` callback
is registered, and the shared profile always publishes streaming as disabled.
Legacy traffic-light physical-sensor options are likewise accepted as no-ops.

## Run the pedestrian demo

From the repository root:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/AI-enabledServices-mwc2027

/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python \
  pedestrian_head_camera_client_v13.py \
  --walk-speed 30.0 \
  --run-speed 40.0 \
  --show-bboxes \
  --npc-vehicles 0 \
  --topdown-zoom-radius 50 \
  --camera-height-reduction 0.60 \
  --spawn-x 90.39 \
  --spawn-y 43.29 \
  --replay-route recorded_pedestrian_route_1.json \
  --rogue-vehicle-warning-radius 12
```

Controls:

- `W/S`: walk forward/backward; hold either Shift key to use run speed.
- `A/D`: turn the pedestrian.
- Arrow keys: move the pedestrian head camera.
- `B`: toggle ground-truth bounding boxes.
- `U`: toggle boxes, route arrows, the top-down spatial map, and metrics.
- `Y`: respawn at the resolved start pose.
- `R`: recenter the head camera.
- `Space`: jump; `Esc` or `Q`: exit.

To record a new keyboard-driven route instead of replaying one:

```bash
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python \
  pedestrian_head_camera_client_v13.py \
  --walk-speed 30.0 \
  --run-speed 40.0 \
  --npc-vehicles 0 \
  --spawn-x 90.39 \
  --spawn-y 43.29 \
  --record-route recorded_pedestrian_route_1.json
```

The recorder checkpoints atomically and finalizes the route on a normal exit.
It refuses to overwrite an existing output unless
`--record-route-overwrite` is supplied.

## Run the ego-vehicle demo

Start the blocker first and wait for its `INFO: Ready ...` message. In terminal
1:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/AI-enabledServices-mwc2027

/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python \
  spawn_blocker_v5.py \
  --vehicle-blueprint vehicle.nissan.patrol \
  --impact-target center
```

Then launch the cooperative-occlusion ego vehicle in terminal 2:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/AI-enabledServices-mwc2027

/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python \
  manual_control_ar_v13.py \
  --topdown-zoom-radius 50 \
  --ego-spawn-x 73.63 \
  --ego-spawn-y 66.36 \
  --vehicle-blueprint vehicle.lincoln.mkz \
  --route-config ./my_ego_route.json \
  --rogue-pedestrian-warning-radius 45 \
  --rogue-pedestrian-brake-radius 12 \
  --spatial-map-active-sensor-pairs 5 \
  --cooperative-sensor-pair-horizontal-fov 120
```

This configuration selects five active virtual sites in normal coverage. The
ego pair counts as one site, leaving up to four active peer sites. The
pair-wide 120-degree override applies to both the virtual RGB camera and radar
at every selected site. Press `U` to enable the spatial map, sensor FoVs,
remaining-occlusion layer, route, boxes, metrics, and cooperative warning
gate.

A blocker pedestrian is withheld from the v13 spatial map and RGB warning
overlay until an active ego or peer modality sees at least one sampled point
on its ground footprint through an unobstructed 2-D line of sight. Direct ego
visibility uses the ordinary pedestrian map color. A peer-only observation is
drawn red and labelled `OCCLUDED PEDESTRIAN`. Either direct or cooperative
visibility authorizes the existing proximity warning; the 45 m warning
radius, 12 m brake radius, and passed-target gate still apply. Inside a shared
network-degraded zone, the same calculation uses the selected radar-only
subset. These virtual spatial-map sensors remain metadata and never collect
data.

Controls:

- `W/S`: throttle/brake; `A/D`: steer.
- Arrow keys: move the active camera.
- `U`: toggle route arrows, actor boxes, top-down map, and metrics.
- `[` or `-`: reduce the active virtual-site budget by one; hold Shift to
  reduce it by five.
- `]` or `=`: increase the active virtual-site budget by one; hold Shift to
  increase it by five.
- `Y`: respawn at the configured or saved-route start.
- `F1`: HUD; `H` or `?`: full in-application help; `Esc`: exit.

The saved route is visual guidance until a route/autonomy control is explicitly
enabled. See `python manual_control_ar_v13.py --help` and the in-application
help for the complete controls.

For the FoV equations, shadow construction, footprint probes, multi-site
occlusion removal, object states, display behavior, and current limitations,
see [Cooperative Occlusion Reasoning in `manual_control_ar_v13.py`](spatial_map_coop/MANUAL_CONTROL_AR_V13_COOPERATIVE_OCCLUSION.md).

## Occluded-hazard visualization

The current vehicle and pedestrian clients recognize the stable role names assigned by
`spawn_blocker_v5.py`; they do not depend on actor IDs, which change after a
blocker respawns.

- In the v13 ego-vehicle client, a `pedestrian_blocker_v5_<index>` must first
  be visible to an active ego or peer virtual modality. When the pedestrian
  also passes the distance and passed-target gates, it gets a red
  `PEDESTRIAN` box and a `SLOW DOWN` or `APPLY BRAKES` warning. Configure the
  two thresholds with `--rogue-pedestrian-warning-radius` and
  `--rogue-pedestrian-brake-radius`.
- In the ego-pedestrian RGB view, a nearby
  `reactive_blocker_v5_<index>` gets a red `VEHICLE` box and an
  `APPROACHING VEHICLE / STOP` warning. Configure its threshold with
  `--rogue-vehicle-warning-radius`.
- In the v13 vehicle top-down map, an unresolved rogue target is withheld to
  prevent ground-truth leakage. A target visible only to a peer is red and
  labelled `OCCLUDED PEDESTRIAN` or `OCCLUDED VEHICLE`; a directly ego-visible
  target uses its ordinary class color. Blocker pedestrians use a larger
  circular marker. The parked `static_blocker_v5_<index>` occluders retain the
  ordinary vehicle color.
- The pedestrian client's occluded-vehicle highlighting remains its existing
  scenario-role/proximity presentation rather than the v13 vehicle client's
  footprint line-of-sight gate.

Press `U` to show the grouped boxes, route markers, top-down map, and live
metrics. `--topdown-zoom-radius` controls the spatial-map coverage. In
`manual_control_ar_v13.py`, the occluded/revealed state is calculated from
CARLA ground-truth 2-D footprints and virtual sensor geometry for strict
blocker roles. It is not measured RGB/radar inference or a validated safety
classifier.

## Virtual spatial sensors and degraded-network behavior

The spatial-map camera and radar symbols are display-only metadata. Camera and
radar symbols in a pair share the same simulated mount position; their small
screen offsets exist only so both shapes remain visible. Marking a symbol
`active` changes its map color and policy state but does not create a CARLA
sensor actor, call `listen()`, capture a frame, or stream data.

- The v13 vehicle client derives virtual pairs for the ego vehicle, every
  nearby live vehicle and pedestrian, and configured traffic-light IDs
  `14,24,11` by default. It displays sensor sites only within the default
  40-metre, +/-45-degree region ahead of the ego vehicle. Use
  `--spatial-map-sensor-forward-range`,
  `--spatial-map-sensor-forward-half-angle`, and
  `--infrastructure-sensor-traffic-light-ids` to customize this inventory.
- In normal coverage, `--spatial-map-active-sensor-pairs` selects the nearest
  complete camera/radar sites with the ego pair pinned first. The target's own
  mounted pair cannot detect its host. Camera and radar FoV/range can be set
  separately, or `--cooperative-sensor-pair-horizontal-fov` can apply one
  horizontal FoV to both modalities.
- The pedestrian client derives virtual pairs for its ego pedestrian, the same
  exact blocker roles, and live traffic-light roots inside its full top-down
  map radius.
- In normal coverage, the nearest configured subset is shown as active
  camera-and-radar pairs. Inside either degraded-network zone, active cameras
  are suppressed and the nearest configured subset of radars is prioritized.
  Spatial-map latency, spatial-map accuracy error, and Sense-to-Act latency are
  increased by the existing deterministic zone-strength model.

The ego vehicle's driving-view camera and the pedestrian's head-view camera
remain real because they render the interactive windows. The vehicle client's
optional `G`-key debug radar and non-RGB safety sensors are also independent of
the virtual spatial-map inventory.

## Author route files

These are authoring workflows, not additional live-demo clients. Each supplied
UI becomes the synchronous 20 Hz master under the default YAML configuration.

Vehicle route:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/AI-enabledServices-mwc2027

/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python \
  physical_ai_scenario_controller_ui_v2.py \
  --route-config ./my_ego_route.json
```

Select the vehicle start, ordered vehicle waypoints, and vehicle end. Click
`Save` or press `Ctrl+S`.

Pedestrian route:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/AI-enabledServices-mwc2027

/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python \
  physical_ai_scenario_controller_ui_v3.py \
  --pedestrian-route-config ./ego_pedestrian_route_v1.json
```

Select the pedestrian start, ordered pedestrian waypoints, and pedestrian end.
Click `Save ped` or press `Ctrl+S` while the pedestrian selector is active.

The YAML's unused default vehicle route name is
`ego_vehicle_route_v1.json`; that file is not part of the source directory.
Passing `./my_ego_route.json`, as above, avoids relying on that absent default.

## Metrics shown with `U`

Both current clients render the six-column window without blocking the CARLA
data path. The fields have these meanings:

| Metric | Source |
| --- | --- |
| Sense: events detected | Current non-ego CARLA ground-truth boxes successfully projected into the RGB frame; not a cumulative detector count |
| Spatial map latency | Local top-down actor query/drawing/composition time; not sensor-network or server end-to-end latency |
| Spatial map accuracy | **DEMO** clipped normal value: mean 2.3 cm, standard deviation 0.45 cm, bounds 1.0-4.0 cm |
| AI reasoning | **DEMO** clipped normal value: mean 45 ms, standard deviation 5 ms, bounds 30-60 ms |
| Sense-to-Act latency | Local camera-callback receipt to the next client control submission; not human reaction or physics/network end-to-end latency |
| Outcome | Collision-based proxy in the vehicle client; dynamic-actor bounding-box-overlap proxy in the pedestrian client |

The two synthetic fields update every 0.5 seconds, pass through an EWMA with
alpha 0.25, and use a dedicated seeded RNG so they do not perturb scenario
randomness. No collision/overlap only means none was observed during the
current ego lifetime; it does not prove that a hazard existed and was avoided.

## Generated files

- `R` in the vehicle client writes image captures beneath `_out/`.
- CARLA recorder output uses `manual_recording.rec`.
- Scenario-controller runs write beneath `experiments/physical_ai_demo/`.
- Route recording writes the requested JSON path and atomic temporary files.

These generated paths are ignored by Git. Saved route JSON files intentionally
included in this repository remain tracked.

## Maintain the GitHub demo branch

Ongoing demo development belongs on `demo/carla-import-20260817`; keep `main`
as the reviewed baseline. Before editing, authenticate with an InterDigital
GitHub account that has repository access and any required organization SSO,
then fast-forward the existing branch:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/AI-enabledServices-mwc2027
gh auth status
git switch demo/carla-import-20260817
git pull --ff-only origin demo/carla-import-20260817
```

Stage only intended demo paths, review the staged diff, and push without force:

```bash
git add -- README.md SOURCE_MANIFEST.json archive/ \
  manual_control_ar_v13.py cooperative_occlusion_v1.py \
  test_manual_control_ar_v13_occlusion.py \
  spatial_map_coop/MANUAL_CONTROL_AR_V13_COOPERATIVE_OCCLUSION.md
git diff --cached --check
git diff --cached --stat
git commit -m "Add cooperative occlusion vehicle client v13"
git push origin demo/carla-import-20260817
```

When the branch has been verified and is ready for `main`, open a new pull
request; a previously merged pull request does not automatically reopen for
later commits on the same branch.

Never place a token in a command, script, README, Git remote URL, or shell
history, and never force-push this shared development branch.

## Validation boundary

The import is checked offline with the CARLA virtual environment: source/copy
hash equality, Python compilation, CLI `--help`, JSON/YAML parsing, local import
closure, and route-schema loading. A live CARLA server run is still required
to validate actor spawning, lane tracking, collision/ragdoll rendering, and GUI
window placement on a specific machine.

Run the offline network-profile transaction regressions with:

```bash
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python \
  -m unittest -v test_spawn_blocker_network_profile.py
```

Run the v13 cooperative-occlusion regressions with:

```bash
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python \
  -m unittest -v test_manual_control_ar_v13_occlusion.py
```
