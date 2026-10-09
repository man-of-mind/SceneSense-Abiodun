# L10319 cooperative-occlusion demo runbook

Status: reproduction guide for the CARLA ground-truth/virtual-sensor v13 demo.
It is not a launch contract for the learned SplitFusion pipeline.

## What this demonstration proves

The demonstration uses CARLA actor poses and 2-D footprints to show how a peer
view can reveal a role-labelled hazard hidden from the ego view. The virtual
camera/radar icons are geometry metadata: they do not capture frames, run the
perception model, or transmit packets.

The authoritative object test samples the target footprint and checks 2-D
line of sight from each active site. The UI reports:

- `EGO_VISIBLE`: an active ego modality has a clear target-footprint probe;
- `COOPERATIVELY_REVEALED`: a peer has a clear probe and the ego does not; or
- `UNRESOLVED_OCCLUDED`: neither ego nor peer supplies fresh visibility.

The purple area is a lower-resolution display approximation of the remaining
ego blind region. It is not the object-state test. Network-degraded zones only
switch the virtual selection policy to radar-only and add labelled demo
metrics; they do not emulate the OAI data plane.

## 1. Select the checkout and interpreter

Run these commands from the graphical desktop session on L10319. Adapt
`DEMO_ROOT` only if the checkout is nested inside `neu_collab/abiodun`.

```bash
export CARLA_ROOT=/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping
export CARLA_PY=/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python
export DEMO_ROOT="$CARLA_ROOT/PythonAPI/AI-enabledServices-mwc2027"

# Nested-tree alternative:
# export DEMO_ROOT="$CARLA_ROOT/PythonAPI/neu_collab/abiodun/AI-enabledServices-mwc2027"

test -x "$CARLA_ROOT/CarlaUnreal.sh"
test -x "$CARLA_PY"
test -f "$DEMO_ROOT/manual_control_ar_v13.py"
export PYTHONPATH="$CARLA_ROOT/PythonAPI/carla:$DEMO_ROOT:${PYTHONPATH:-}"
```

Use the desktop's existing `DISPLAY`; do not launch the GUI through a shell
that cannot access the active display.

## 2. Start a fresh CARLA server and load the required map

Close any previous CARLA demo cleanly first. In terminal 1:

```bash
cd "$CARLA_ROOT"
./CarlaUnreal.sh -quality-level=Epic -carla-rpc-port=2000
```

In another terminal, load and verify the exact map:

```bash
"$CARLA_PY" -c 'import carla; c=carla.Client("127.0.0.1",2000); c.set_timeout(30); w=c.load_world("Town10HD_Opt"); print(w.get_map().name)'
```

The printed name must end in `Town10HD_Opt`, not `Town10HD`.

## 3. Optional short offline geometry check

This is useful after moving the checkout or changing Python environments:

```bash
cd "$DEMO_ROOT"
"$CARLA_PY" -m unittest -v test_manual_control_ar_v13_occlusion.py
```

It checks geometry and mocked integration only; it is not a substitute for the
live visual run.

## 4. Start the blocker scenario

In terminal 2:

```bash
cd "$DEMO_ROOT"
"$CARLA_PY" spawn_blocker_v5.py \
  --vehicle-blueprint vehicle.nissan.patrol \
  --impact-target center \
  --reactive-impact-settle-time 0.25 \
  --reactive-contact-hold-time 5.0 \
  --reactive-contact-stop-speed 0.25 \
  --reactive-launch-margin 0.50 \
  --reactive-approach-hold-distance 1.50 \
  --reactive-approach-hold-timeout 30.0
```

Wait for its `INFO: Ready ...` line before starting the ego client.

## 5. Start the cooperative-occlusion ego view

In terminal 3:

```bash
cd "$DEMO_ROOT"
"$CARLA_PY" manual_control_ar_v13.py \
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

Then:

1. Press `U` to show the map, FoVs, remaining-occlusion layer, labels and
   metrics.
2. Drive with `W/A/S/D`, or press `P` to toggle the available route/autopilot
   behavior.
3. Use `[`/`-` and `]`/`=` to reduce/increase the virtual-site budget.
4. Observe whether a hidden role-labelled pedestrian is withheld, appears red
   as `OCCLUDED PEDESTRIAN` after a peer reveal, and becomes ordinarily colored
   when directly visible.
5. Compare the exact object label with the purple approximate area near shadow
   boundaries; small boundary differences are expected.

The metrics labelled `DEMO` are synthetic presentation values. Do not record
them as learned-model, OAI, or service-latency evidence.

## 6. Optional route-authoring UI

The scenario-controller UI is a separate route/scenario authoring view, not
the v13 cooperative-occlusion evaluator. Stop the manual client and blocker
before using it because it owns the 20 Hz synchronous CARLA clock:

```bash
cd "$DEMO_ROOT"
"$CARLA_PY" physical_ai_scenario_controller_ui_v3.py \
  --route-config ./my_ego_route.json \
  --pedestrian-route-config ./ego_pedestrian_route_v1.json
```

## 7. Teardown

Exit the ego client first, interrupt `spawn_blocker_v5.py` second, and stop the
CARLA server last. Confirm no demo client remains before another experiment.
Starting the next reproduction from a fresh CARLA process avoids inherited
actors, clock mode, weather, and scenario state.

## Production integration boundary

The reusable part is the pure geometry in `cooperative_occlusion_v1.py` and its
explicit visibility-state vocabulary. A production implementation must replace
CARLA actor truth and virtual visibility with timestamped model observations,
associated tracks, pose/calibration provenance, uncertainty, and real per-UE
delivery state. The demo is best retained as an oracle/reference evaluator and
UI prototype.
