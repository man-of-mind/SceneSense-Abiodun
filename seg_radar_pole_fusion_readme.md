# Remote Runbook: Fusion Object Spatial Map Streams

This runbook is for moving the two pole RGB+radar fusion clients plus the fused spatial-map server to another Ubuntu machine that already has CARLA 0.10 and a working Python virtual environment.

## Entry Points

Run these three primary scripts on the remote machine:

- `real_time_spatial_map_server_fusion_object_v2.py`
- `carla_split_inference_udp_fusion_object_pole_client_spatial_stream.py`
- `carla_split_inference_udp_fusion_object_pole_client_spatial_stream_2.py`

## Additional Files To Copy

Copy these files and folders in addition to the three entry-point scripts:

```text
neu_collab/
  extract_traffic_lights.py
  traffic_lights_data.json

  carla_split_inference_udp_demo.py
  carla_split_inference_udp_data_collect.py
  carla_split_inference_udp_segmentation_demo.py
  carla_split_inference_udp_segmentation_trained_lraspp_demo.py
  carla_split_inference_udp_segmentation_trained_lraspp_pole_client.py

  pole_lraspp_multimodal_fusion/
    pole_lraspp_multimodal_fusion/
      __init__.py
      model.py
      object_targets.py
      radar_fusion.py
      split_runtime.py
      common.py
```

Also copy the trained fusion checkpoint:

```text
experiments/pole_lraspp_multimodal_fusion/
  20260508_070718_pole_lraspp_multimodal_fusion_learned_localization/
    checkpoints/
      fusion_v4_lowfuse_adamw_768x432_lr1e-4_radar4_aug_strong_bs2_obj_sel/
        best.pt
```

Do not copy the full `20260508_070718...` experiment directory unless you need the original dataset, figures, metrics, and logs. The full directory is large; inference only needs `best.pt`. If you want to keep using `--fusion-experiment-dir`, copy `manifest.json` too and either preserve the same absolute directory path on the remote machine or edit the manifest's `best_checkpoint` path. The safer remote option is to pass `--fusion-checkpoint /remote/path/to/best.pt` directly.

## Python Prerequisites

Use the CARLA 0.10 Python virtual environment on the remote machine. The Python version must match the CARLA Python API wheel or installed `carla` module.

Required Python packages:

```bash
pip install numpy torch torchvision matplotlib flask
pip install opencv-python
```

For a purely SSH/headless machine, `opencv-python-headless` is also acceptable in place of `opencv-python` as long as you keep the client commands on `--headless`.

Optional:

```bash
pip install zstandard
```

`zstandard` is only needed if you run with `--entropy-coder zstd`. The commands below use `zlib`, so it is not required.

The remote environment must be able to import CARLA:

```bash
python3 -c "import carla; print('carla import ok')"
```

If that fails, add the CARLA Python API wheel or egg to `PYTHONPATH`, or install the CARLA API into the virtual environment.

## CARLA Prerequisites

Start CARLA 0.10 before launching the server and clients. These clients attach to the already loaded CARLA world; they do not load a town themselves. Use the same Town10HD/Town10HD_Opt setup used when `traffic_lights_data.json` was generated, otherwise traffic-light ID `14` may not refer to the expected pole.

Example:

```bash
source /path/to/carla_0_10_venv/bin/activate
cd /path/to/Carla-0.10.0-Linux-Shipping
./CarlaUnreal.sh
```

In another terminal, confirm that traffic light `14` can be resolved:

```bash
source /path/to/carla_0_10_venv/bin/activate
cd /path/to/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab
python3 carla_split_inference_udp_fusion_object_pole_client_spatial_stream.py \
  --list-traffic-lights
```

If live traffic-light actors are not visible, the clients can fall back to `traffic_lights_data.json`. That fallback only makes sense if the remote CARLA map matches the map used to generate the JSON file.

## Recommended Remote Layout

Keep the copied files under the remote CARLA `PythonAPI/neu_collab` directory:

```text
/path/to/Carla-0.10.0-Linux-Shipping/
  PythonAPI/
    carla/
    examples/
    neu_collab/
      real_time_spatial_map_server_fusion_object_v2.py
      carla_split_inference_udp_fusion_object_pole_client_spatial_stream.py
      carla_split_inference_udp_fusion_object_pole_client_spatial_stream_2.py
      ...
      checkpoints/
        fusion_object_best.pt
```

Using a short checkpoint path such as `checkpoints/fusion_object_best.pt` makes the run commands portable.

## Execution Steps

Open four terminals on the remote machine: one for CARLA, one for the spatial-map server, and one for each stream.

### Terminal 1: CARLA

```bash
source /path/to/carla_0_10_venv/bin/activate
cd /path/to/Carla-0.10.0-Linux-Shipping
./CarlaUnreal.sh
```

### Terminal 2: Fused Spatial-Map Server

```bash
source /path/to/carla_0_10_venv/bin/activate
cd /path/to/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab

python3 real_time_spatial_map_server_fusion_object_v2.py \
  --object-yaw-map-offset-deg 10.0 \
  --focus-traffic-light-ids 14 \
  --focus-radius-m 20
```

The server listens for UDP fusion-object packets on port `39201` and exposes the map API on port `35011`. Open this viewer from the remote machine or through an SSH tunnel:

```text
http://127.0.0.1:35011/api/spatial_map/viewer
```

### Terminal 3: Stream 1, Synchronous World Owner

Start Stream 1 first. It owns synchronous CARLA ticking.

```bash
source /path/to/carla_0_10_venv/bin/activate
cd /path/to/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab

python3 carla_split_inference_udp_fusion_object_pole_client_spatial_stream.py \
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
  --result-timeout 1.5 \
  --headless
```

Remove `--headless` only if the remote machine has a working display and you want the OpenCV camera overlay window.

### Terminal 4: Stream 2, Async Viewer

Start Stream 2 after Stream 1 is running. It must remain `--async-world` so it does not fight the synchronous owner.

```bash
source /path/to/carla_0_10_venv/bin/activate
cd /path/to/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab

python3 carla_split_inference_udp_fusion_object_pole_client_spatial_stream_2.py \
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
  --result-timeout 1.5 \
  --headless
```

## Health Checks

Check that the Flask server is alive:

```bash
curl http://127.0.0.1:35011/healthz
```

Check the latest fused spatial-map JSON:

```bash
curl http://127.0.0.1:35011/api/spatial_map/latest
```

Check the live PNG directly:

```text
http://127.0.0.1:35011/api/spatial_map/live.png
```

The server also writes `latest_fusion_object_spatial_map.png` in the server output directory by default.

## Common Issues

- `Fusion checkpoint not found`: use `--fusion-checkpoint` with the remote path to `best.pt`. Do not rely on a copied manifest unless its absolute `best_checkpoint` path matches the remote machine.
- `Unable to import CARLA`: activate the CARLA virtual environment and verify the CARLA Python API is importable with `python3 -c "import carla"`.
- No objects appear in the server map: confirm both clients are sending to the same `--spatial-map-port 39201` and each stream has a unique `--spatial-map-stream-id`.
- Stream 2 disrupts Stream 1: confirm Stream 1 uses `--sync-world` and Stream 2 uses `--async-world`.
- Traffic-light ID `14` cannot be resolved: run `--list-traffic-lights` against the loaded remote world, or regenerate/copy a matching `traffic_lights_data.json`.
- Headless server has display errors: run the clients with `--headless` and do not pass `--display` to the server.
