# SceneSense Curbside Accident Demo

This folder contains the motivating hidden-pedestrian scenario. An ego vehicle
approaches parked curbside vehicles while a pedestrian emerges from behind the
occluder. A helper vehicle travels in the opposite lane and provides a second
camera viewpoint. The run opens separate ego and helper RGB preview windows.

## Start the demo

1. Start CARLA in one terminal:

   ```bash
   cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping
   ./CarlaUnreal.sh
   ```

2. In a second terminal, activate the CARLA Python environment and run the
   launcher:

   ```bash
   source /home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/activate
   cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun
   bash curbside_accident_demo/run_curbside_far_sidewalk_demo.sh
   ```

The launcher loads `Town10HD_Opt` and runs for approximately 25 seconds. Press
`q` or `Esc` in a preview window to stop early. A graphical desktop or working
X display is required for the two OpenCV preview windows.

Results and buffered ego/helper camera evidence are written under:

```text
metrics_logs/scenesense_scenarios/<timestamp>_curbside_parked_vehicle_pedestrian_occlusion_seed7/
```

The shell launcher and `scenesense_scenario_harness.py` must remain together in
this folder because the launcher invokes the harness by relative path.
