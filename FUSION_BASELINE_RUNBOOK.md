# RGB+Radar Fusion Baseline Runbook

This runbook records the local baseline commands for the copied `abiodun/` RGB+radar fusion pipeline.

## Prerequisite

Start CARLA 0.10 with Town10HD/Town10HD_Opt loaded.

Quick connection check:

```bash
source /home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/activate
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun

python - <<'PY'
import carla
client = carla.Client("127.0.0.1", 2000)
client.set_timeout(10.0)
world = client.get_world()
print("connected")
print(world.get_map().name)
print("actors", len(world.get_actors()))
PY
```

## Terminal 1: Spatial-Map Server

```bash
source /home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/activate
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun

export MPLCONFIGDIR=/tmp/fusion_mplconfig

python real_time_spatial_map_server_fusion_object_v2.py \
  --object-yaw-map-offset-deg 10.0 \
  --focus-traffic-light-ids 14 \
  --focus-radius-m 20
```

Viewer:

```text
http://127.0.0.1:35011/api/spatial_map/viewer
```

Health:

```bash
curl http://127.0.0.1:35011/healthz
```

Latest JSON:

```bash
curl http://127.0.0.1:35011/api/spatial_map/latest
```

## Terminal 2: Pole Stream 1

Stream 1 owns synchronous CARLA ticking.

For metrics runs, prefer the OAI-capable script in `--role loopback` mode so
local and OAI runs share the same CSV schema.

No shared setup command is required for normal logging. Each stream prints its
own metrics folder, and related streams can be paired later by the CSV/manifest
`run_group` field. Add the same `--run-group <label>` to both stream commands
only when you want an exact manual experiment label.

For smoke tests, the automatic group is enough. For official experiment runs,
add a unique manual group to both stream commands, for example:
`--run-group exp01_loopback_clear`.

```bash
source /home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/activate
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun

python carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai.py \
  --role loopback \
  --bind-host 127.0.0.1 \
  --remote-host 127.0.0.1 \
  --sync-world \
  --traffic-light-id 14 \
  --camera-x 9 \
  --camera-y 2 \
  --camera-pitch -30 \
  --camera-yaw-offset 50 \
  --camera-roll 0 \
  --camera-fov 100 \
  --fusion-checkpoint checkpoints/fusion_object_best.pt \
  --entropy-coder zlib \
  --spatial-map-stream-id fusion_tl_14 \
  --spatial-map-port 39201 \
  --camera-source-port 51001 \
  --remote-port 51002 \
  --remote-source-port 51003 \
  --camera-result-port 51004 \
  --transport-label loopback \
  --result-timeout 1.5 \
  --headless
```

For a short smoke test, add:

```bash
--max-frames 120
```

## Terminal 3: Pole Stream 2

Stream 2 must stay asynchronous so it does not fight stream 1 for world ticks.

```bash
source /home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/activate
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun

python carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai.py \
  --role loopback \
  --bind-host 127.0.0.1 \
  --remote-host 127.0.0.1 \
  --async-world \
  --traffic-light-id 14 \
  --camera-x 11 \
  --camera-y 2 \
  --camera-pitch -30 \
  --camera-yaw-offset 120 \
  --camera-roll 0 \
  --camera-fov 100 \
  --fusion-checkpoint checkpoints/fusion_object_best.pt \
  --entropy-coder zlib \
  --spatial-map-stream-id fusion_tl_14_view_2 \
  --spatial-map-port 39201 \
  --camera-source-port 51101 \
  --remote-port 51102 \
  --remote-source-port 51103 \
  --camera-result-port 51104 \
  --npc-vehicles 0 \
  --npc-pedestrians 0 \
  --transport-label loopback \
  --result-timeout 1.5 \
  --headless
```

For a short smoke test, add:

```bash
--max-frames 120
```

## Expected Success

- Stream clients load `checkpoints/fusion_object_best.pt`.
- Stream 1 runs with `--sync-world`.
- Stream 2 runs with `--async-world`.
- Pole client receives segmentation masks and object/localization outputs from the split-inference back half.
- Spatial-map server receives both stream IDs and updates the viewer.
