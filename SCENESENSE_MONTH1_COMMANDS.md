# SceneSense Month 1 Reproducible Commands

Last updated: 2026-06-05

Purpose: one compact command sheet for the Month 1 Definition of Done. These
commands point to the reproducible local loopback and OAI 5G paths for:

- camera-only object detection
- camera-only semantic segmentation
- RGB+radar fusion evaluated as OD and SEG

Detailed setup remains in `scripts/README.md`, `receiver_container/README.md`,
`FUSION_BASELINE_RUNBOOK.md`, `FUSION_OAI_RUNBOOK.md`,
`FUSION_OAI_MULTI_UE_RUNBOOK.md`, and `seg_radar_pole_fusion_readme.md`.

## Sensor/Input Boundary

Camera-only OD and camera-only SEG use only the CARLA RGB camera as model input.

- OD scripts spawn/read `sensor.camera.rgb` and send Faster R-CNN/FPN features.
- SEG scripts spawn/read `sensor.camera.rgb` and send LR-ASPP backbone features.
- SEG variants may spawn a co-located CARLA semantic-segmentation camera for
  ground truth, but that camera is evaluation-only and is not model input.
- RGB+radar fusion is the separate multimodal route. It builds a 7-channel
  model input by concatenating RGB with a 4-channel radar tensor.

## Common Setup

Run from the editable project folder:

```bash
source /home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/activate
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun
```

Assumptions:

- CARLA 0.10 is running and reachable at `127.0.0.1:2000`.
- OAI commands assume CN/gNB/UE are already up and UE1 has `10.0.0.2`.
- OAI back-half container IP is `192.168.70.140`.
- For headless SSH sessions, keep `--headless`.
- For official fusion/OAI runs, use one shared `--run-group` label across
  front-half clients, tunnel sampler, and T-tracer commands.

## 0. OAI 5G Stack Bring-Up

Use these before any `--role front` OAI command. The scripts live under
`abiodun/scripts/`.

Single UE:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/scripts

./cn_start.sh
./cn_status.sh

# terminal 2
./gnb_start.sh

# terminal 3
./ue_start.sh

# terminal 4, after the UE registers
./ue_check.sh
```

Expected single-UE tunnel:

```text
oaitun_ue1 -> 10.0.0.2
```

Two UEs:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/scripts

./cn_start.sh

# terminal 2
./gnb_start.sh

# terminal 3
./ue_multi_start.sh

# terminal 4
./ue_multi_check.sh
```

Expected two-UE tunnels:

```text
oaitun_ue1 -> 10.0.0.2
oaitun_ue2 -> 10.0.0.3
```

Useful teardown:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/scripts

./receiver_container_down.sh
./cn_stop.sh
```

## 1. Camera-Only OD

Model/input: Faster R-CNN split route, RGB camera only.

### Local Loopback

Single process, both halves on one host:

```bash
python3 carla_split_inference_udp_oai.py \
  --role loopback \
  --bind-host 127.0.0.1 \
  --remote-host 127.0.0.1 \
  --camera-resolution 720p \
  --front-device cuda \
  --back-device cuda \
  --metrics-log-prefix month1_camera_od_loopback \
  --metrics-warmup-frames 0 \
  --enable-od-gt \
  --od-gt-iou-threshold 0.5 \
  --od-gt-min-area-px 64 \
  --disable-live-plot \
  --headless
```

CPU smoke option:

```bash
python3 carla_split_inference_udp_oai.py \
  --role loopback \
  --bind-host 127.0.0.1 \
  --remote-host 127.0.0.1 \
  --camera-resolution 720p \
  --disable-pretrained \
  --front-device cpu \
  --back-device cpu \
  --metrics-log-prefix month1_camera_od_loopback_cpu_smoke \
  --metrics-warmup-frames 0 \
  --disable-od-gt \
  --disable-live-plot \
  --headless
```

### OAI 5G

Terminal A, start OD back half in the perception container:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/scripts
./receiver_container_od_back_up.sh
sudo docker logs -f oai-perception-rx
```

Terminal B, run OD front half on the UE-bound host:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun

python3 carla_split_inference_udp_oai.py \
  --role front \
  --bind-host 10.0.0.2 \
  --remote-host 192.168.70.140 \
  --camera-resolution 720p \
  --front-device cuda \
  --metrics-log-prefix month1_camera_od_oai \
  --metrics-warmup-frames 0 \
  --enable-od-gt \
  --od-gt-iou-threshold 0.5 \
  --od-gt-min-area-px 64 \
  --disable-live-plot \
  --headless
```

### Analyze OD Quality

Run this after one or both camera-only OD runs finish:

```bash
python3 scripts/analyze_camera_od_metrics.py \
  'metrics_logs/month1_camera_od_loopback_*.csv' \
  'metrics_logs/month1_camera_od_oai_*.csv' \
  --output-dir metrics_logs/month1_camera_od_analysis \
  --label month1_camera_od_quality
```

If an older CSV was collected without `--enable-od-gt`, the analyzer will mark
it as `quality_columns_present=false`; that file is transport-only and should
not be used for OD recall/precision conclusions.

## 2. Camera-Only SEG

Model/input: LR-ASPP split route, RGB camera only.
The `--enable-semantic-gt` flag spawns a co-located CARLA semantic camera only
for evaluation; it does not change the RGB-only model input. Use it when the
run needs mIoU / foreground IoU / vehicle-person IoU columns.

### Local Loopback

Single process, both halves on one host:

```bash
python3 carla_split_inference_udp_segmentation_oai.py \
  --role loopback \
  --bind-host 127.0.0.1 \
  --remote-host 127.0.0.1 \
  --camera-resolution 1080p \
  --seg-input-width 512 \
  --seg-input-height 288 \
  --mask-output-size model \
  --front-device cuda \
  --back-device cuda \
  --metrics-log-prefix month1_camera_seg_loopback \
  --metrics-warmup-frames 0 \
  --enable-semantic-gt \
  --headless
```

CPU smoke option:

```bash
python3 carla_split_inference_udp_segmentation_oai.py \
  --role loopback \
  --bind-host 127.0.0.1 \
  --remote-host 127.0.0.1 \
  --camera-resolution 1080p \
  --seg-input-width 512 \
  --seg-input-height 288 \
  --mask-output-size model \
  --seg-disable-pretrained \
  --front-device cpu \
  --back-device cpu \
  --metrics-log-prefix month1_camera_seg_loopback_cpu_smoke \
  --metrics-warmup-frames 0 \
  --disable-semantic-gt \
  --headless
```

### OAI 5G

Terminal A, start SEG back half in the perception container:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/scripts
./receiver_container_seg_back_up.sh
sudo docker logs -f oai-perception-rx
```

Terminal B, run SEG front half on the UE-bound host:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun

python3 carla_split_inference_udp_segmentation_oai.py \
  --role front \
  --bind-host 10.0.0.2 \
  --remote-host 192.168.70.140 \
  --camera-resolution 1080p \
  --seg-input-width 512 \
  --seg-input-height 288 \
  --mask-output-size model \
  --front-device cuda \
  --metrics-log-prefix month1_camera_seg_oai \
  --metrics-warmup-frames 0 \
  --enable-semantic-gt \
  --headless
```

### Analyze SEG Quality

Run this after one or both camera-only SEG runs finish:

```bash
python3 scripts/analyze_camera_seg_metrics.py \
  'metrics_logs/month1_camera_seg_loopback_*.csv' \
  'metrics_logs/month1_camera_seg_oai_*.csv' \
  --output-dir metrics_logs/month1_camera_seg_analysis \
  --label month1_camera_seg_quality
```

If an older CSV was collected without `--enable-semantic-gt`, the analyzer will
mark it as `quality_columns_present=false`; that file is transport-only and
should not be used for mIoU conclusions.

## 3. Camera-Only OD-vs-SEG Latency Presentation

Use latency-only traces when comparing transport behavior. Keep GT disabled so
the runs measure the split-inference path rather than evaluation overhead:

- OD: `--disable-od-gt`
- SEG: `--disable-semantic-gt`
- Recommended duration wrapper: `timeout --signal=INT --kill-after=10s 210s`
  for about 180 s of steady-state samples.

After the four traces and the OD/SEG analyzers finish, generate the summary
plots and PowerPoint deck:

```bash
python3 scripts/create_camera_latency_comparison_deck.py \
  --od-json metrics_logs/month1_camera_latency_analysis/<month1_latency_od_*.json> \
  --seg-json metrics_logs/month1_camera_latency_analysis/<month1_latency_seg_*.json>
```

Default output:

```text
metrics_logs/scenesense_analysis/camera_od_seg_latency_20260604/
SceneSense_Camera_OD_SEG_Latency_Comparison.pptx
```

If the traces were produced on the remote machine, pull the raw CSVs and
analyzer outputs first:

```bash
rsync -avh \
  shr_aisvcs@L10319.idcc.lab:/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/metrics_logs/month1_latency_od_loopback_*.csv \
  shr_aisvcs@L10319.idcc.lab:/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/metrics_logs/month1_latency_od_oai_*.csv \
  shr_aisvcs@L10319.idcc.lab:/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/metrics_logs/month1_latency_seg_loopback_*.csv \
  shr_aisvcs@L10319.idcc.lab:/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/metrics_logs/month1_latency_seg_oai_*.csv \
  metrics_logs/

rsync -avh \
  shr_aisvcs@L10319.idcc.lab:/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/metrics_logs/month1_camera_latency_analysis/ \
  metrics_logs/month1_camera_latency_analysis/
```

## 4. RGB+Radar Fusion

Model/input: LR-ASPP RGB+radar fusion model. Both `fusion_as_od` and
`fusion_as_seg` metrics come from the same RGB+radar runtime, but they must be
analyzed separately.

### Local Loopback

Terminal A, spatial-map server:

```bash
python3 real_time_spatial_map_server_fusion_object_v2.py \
  --object-yaw-map-offset-deg 10.0 \
  --focus-traffic-light-ids 14 \
  --focus-radius-m 20
```

Terminal B, pole stream 1. This stream owns synchronous CARLA ticking:

```bash
python3 carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai.py \
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
  --run-group month1_fusion_loopback \
  --result-timeout 1.5 \
  --headless
```

Terminal C, pole stream 2. This stream stays asynchronous:

```bash
python3 carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai.py \
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
  --run-group month1_fusion_loopback \
  --result-timeout 1.5 \
  --headless
```

### Local Loopback, Parked-Ego Transfer Variant

Run this after the two pole streams above when collecting pole-vs-parked-ego
fusion transferability evidence. Stop the pole streams first, then start the
parked-ego pair with the same `--run-group` so the SEG/OD analyzers can compare
`fusion_tl_*` and `fusion_ego_*` streams together. For parked-ego map viewing,
restart the spatial-map server without the TL14 focus crop:

```bash
python3 real_time_spatial_map_server_fusion_object_v2.py \
  --object-yaw-map-offset-deg 10.0
```

Terminal B, parked-ego stream 1. This stream owns synchronous CARLA ticking and
spawns the shared NPC scene:

```bash
python3 carla_split_inference_udp_fusion_object_ego_client.py \
  --sync-world \
  --ego-vehicle-blueprint vehicle.lincoln.mkz \
  --ego-spawn-index 152 \
  --ego-spawn-forward-offset-m 0.0 \
  --ego-spawn-right-offset-m 3.0 \
  --ego-spawn-z-offset-m 0.15 \
  --fusion-checkpoint checkpoints/fusion_object_best.pt \
  --entropy-coder zlib \
  --npc-vehicles 20 \
  --npc-pedestrians 10 \
  --run-group month1_fusion_loopback \
  --transport-label loopback \
  --spatial-map-stream-id fusion_ego_front \
  --spatial-map-port 39201 \
  --camera-source-port 51201 \
  --remote-port 51202 \
  --remote-source-port 51203 \
  --camera-result-port 51204 \
  --result-timeout 1.5 \
  --headless
```

Terminal C, parked-ego stream 2. Start this after stream 1 is running; it uses
the nearby face-to-face pose and stays asynchronous:

```bash
python3 carla_split_inference_udp_fusion_object_ego_client.py \
  --async-world \
  --ego-vehicle-blueprint vehicle.dodge.charger \
  --ego-spawn-index 152 \
  --ego-spawn-forward-offset-m 8.0 \
  --ego-spawn-right-offset-m 3.0 \
  --ego-spawn-z-offset-m 0.15 \
  --ego-spawn-yaw-offset-deg 180.0 \
  --ego-camera-yaw 0.0 \
  --fusion-checkpoint checkpoints/fusion_object_best.pt \
  --entropy-coder zlib \
  --npc-vehicles 0 \
  --npc-pedestrians 0 \
  --run-group month1_fusion_loopback \
  --transport-label loopback \
  --spatial-map-stream-id fusion_ego_front_view_2 \
  --spatial-map-port 39201 \
  --camera-source-port 51301 \
  --remote-port 51302 \
  --remote-source-port 51303 \
  --camera-result-port 51304 \
  --result-timeout 1.5 \
  --headless
```

### OAI 5G

Terminal A, start fusion back halves in the perception container:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/scripts
./receiver_container_fusion_back_up.sh
sudo docker logs -f oai-perception-rx
```

Terminal B, spatial-map server:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun

python3 real_time_spatial_map_server_fusion_object_v2.py \
  --object-yaw-map-offset-deg 10.0 \
  --focus-traffic-light-ids 14 \
  --focus-radius-m 20
```

Terminal C, pole stream 1 over UE1:

```bash
python3 carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai.py \
  --role front \
  --bind-host 10.0.0.2 \
  --remote-host 192.168.70.140 \
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
  --front-device cuda \
  --transport-label oai_5g \
  --run-group month1_fusion_oai \
  --result-timeout 1.5 \
  --headless
```

Terminal D, pole stream 2 over UE2:

```bash
python3 carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai.py \
  --role front \
  --bind-host 10.0.0.3 \
  --remote-host 192.168.70.140 \
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
  --front-device cuda \
  --transport-label oai_5g \
  --run-group month1_fusion_oai \
  --result-timeout 1.5 \
  --headless
```

### OAI 5G, Two UE Variant

Use this when the two fusion streams should traverse separate UE tunnels.

Terminal A, verify two UE tunnels:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/scripts
./ue_multi_check.sh
```

Terminal B, start fusion back halves with separate return addresses:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/scripts

FUSION_BACK_REMOTE_HOST_1=10.0.0.2 \
FUSION_BACK_REMOTE_HOST_2=10.0.0.3 \
FUSION_BACK_LOG_EVERY=30 \
./receiver_container_fusion_back_up.sh

sudo docker logs -f oai-perception-rx
```

Terminal C, spatial-map server:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun

python3 real_time_spatial_map_server_fusion_object_v2.py \
  --object-yaw-map-offset-deg 10.0 \
  --focus-traffic-light-ids 14 \
  --focus-radius-m 20
```

Terminal D, stream 1 over UE1:

```bash
python3 carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai.py \
  --role front \
  --bind-host 10.0.0.2 \
  --remote-host 192.168.70.140 \
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
  --front-device cuda \
  --transport-label multi_ue_oai \
  --run-group month1_fusion_multiue_oai \
  --result-timeout 1.5 \
  --headless
```

Terminal E, stream 2 over UE2:

```bash
python3 carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai.py \
  --role front \
  --bind-host 10.0.0.3 \
  --remote-host 192.168.70.140 \
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
  --front-device cuda \
  --transport-label multi_ue_oai \
  --run-group month1_fusion_multiue_oai \
  --result-timeout 1.5 \
  --headless
```

### Analyze Fusion Task Quality

Fusion runtime/app/network metrics are summarized in Section 5. The commands
below are for task quality only.

`fusion_as_seg` uses the semantic-GT IoU columns in each fusion stream metrics
CSV:

```bash
python3 scripts/analyze_camera_seg_metrics.py \
  'metrics_logs/scenesense_runs/<run_group>/*/streams/fusion_*_metrics.csv' \
  --output-dir metrics_logs/scenesense_analysis/<run_group>_fusion_seg_quality \
  --label <run_group>_fusion_as_seg
```

`fusion_as_od` uses CARLA actor object truth saved by the fusion streams and
matches predicted spatial-map objects by XY distance:

```bash
python3 scripts/analyze_fusion_object_transfer.py \
  --run-group <run_group> \
  --match-distance-m 2.0 \
  --distance-thresholds-m 1,2,3 \
  --min-gt-bbox-area-px 12 \
  --min-gt-bbox-width-px 4 \
  --min-gt-bbox-height-px 4 \
  --max-gt-distance-m 45 \
  --require-gt-center-in-image \
  --output-dir metrics_logs/scenesense_analysis/<run_group>_fusion_od_strict
```

Sensitivity view, useful for checking whether detections are roughly present
but poorly localized:

```bash
python3 scripts/analyze_fusion_object_transfer.py \
  --run-group <run_group> \
  --match-distance-m 5.0 \
  --distance-thresholds-m 1,2,3,5 \
  --min-gt-bbox-area-px 12 \
  --min-gt-bbox-width-px 4 \
  --min-gt-bbox-height-px 4 \
  --max-gt-distance-m 45 \
  --require-gt-center-in-image \
  --output-dir metrics_logs/scenesense_analysis/<run_group>_fusion_od_sensitivity
```

Regenerate the transferability deck after the pole-vs-ego SEG summary and
fusion OD summaries are refreshed:

```bash
python3 scripts/create_fusion_transferability_deck.py
```

## 5. Canonical Curbside Accident Scenario

The Month 1 hidden-hazard scenario is the animated curbside parked-vehicle
pedestrian dart-out. It uses a helper vehicle camera and writes an evidence
pack with actor traces plus ego/helper RGB frames.

Run the validated demo helper:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun

bash scenesense_scenarios/run_curbside_far_sidewalk_demo.sh
```

The helper defaults reproduce the locked Month 1 demo:

```text
TARGET_START_LAT=5.2
TARGET_FORWARD=-6.5
TARGET_SPEED=21.0
ROUTE_LEAD=24.0
EGO_TARGET_SPEED=6.0
EGO_THROTTLE=0.45
OCCLUDER_BP=vehicle.sprinter.mercedes
```

Optional explicit override pattern:

```bash
ROUTE_LEAD=24.0 TARGET_START_LAT=5.2 TARGET_SPEED=21.0 EGO_THROTTLE=0.45 \
  bash scenesense_scenarios/run_curbside_far_sidewalk_demo.sh
```

Validate the generated evidence folder:

```bash
python3 scenesense_scenarios/validate_evidence_pack.py \
  metrics_logs/scenesense_scenarios/<timestamp>_curbside_parked_vehicle_pedestrian_occlusion_seed7 \
  --require-collision
```

List available scenarios:

```bash
python3 scenesense_scenarios/scenesense_scenario_harness.py --list
```

Scout right-turn occlusion anchors, if that deferred scenario is resumed:

```bash
python3 scenesense_scenarios/scout_right_turn_occlusion_anchors.py --top 20
```

## 6. OAI Network, Application, and RAN Metrics

Use the same label everywhere for an official OAI run. Examples below use
`month1_fusion_oai` for single UE and `month1_fusion_multiue_oai` for two UEs.

### UE Tunnel Network Sampler

Single UE:

```bash
python3 scripts/sample_oai_network_metrics.py \
  --run-group month1_fusion_oai \
  --interface oaitun_ue1:ue1 \
  --ping-host 192.168.70.135
```

Two UEs:

```bash
python3 scripts/sample_oai_network_metrics.py \
  --run-group month1_fusion_multiue_oai \
  --interface oaitun_ue1:ue1 \
  --interface oaitun_ue2:ue2 \
  --ping-host 192.168.70.135
```

Stop the sampler with Ctrl+C after the front-half clients stop.

### Application + Network Summary

List discovered application run groups:

```bash
python3 scripts/analyze_scenesense_app_metrics.py --list-groups
```

Analyze one fusion run group and automatically join matching tunnel metrics:

```bash
python3 scripts/analyze_scenesense_app_metrics.py \
  --run-group month1_fusion_oai \
  --output-dir metrics_logs/scenesense_analysis/month1_fusion_oai
```

If the application and network sampler labels differ:

```bash
python3 scripts/analyze_scenesense_app_metrics.py \
  --run-group <application_run_group> \
  --network-run-group <network_run_group>
```

### OAI Snapshot Bundle

After a fusion OAI run, collect container/network snapshots into one printed
run folder:

```bash
scripts/collect_oai_run_logs.sh \
  metrics_logs/scenesense_runs/<printed_run_folder>
```

### T-Tracer / RAN Metrics

Build the T-tracer tools once:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/scripts
./ttracer_build_tools.sh
```

For a RAN-instrumented OAI run, start OAI with the T-enabled launchers:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/scripts

./cn_start.sh

# terminal 2
./gnb_start_ttracer.sh

# terminal 3, two-UE mode
./ue_multi_start_ttracer.sh

# terminal 4
./ue_multi_check.sh
```

Record gNB and UE T-tracer files while the application traffic is active:

```bash
./ttracer_record_smoke.sh \
  --run-group month1_fusion_multiue_oai \
  --source gnb \
  --duration-s 60
```

```bash
./ttracer_record_smoke.sh \
  --run-group month1_fusion_multiue_oai \
  --source ue \
  --duration-s 60
```

Extract CSVs:

```bash
./ttracer_extract_csv_smoke.sh \
  --run-group month1_fusion_multiue_oai \
  --source gnb

./ttracer_extract_csv_smoke.sh \
  --run-group month1_fusion_multiue_oai \
  --source ue \
  --clean-output
```

Analyze UE decoded grants:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun

python3 scripts/analyze_nrue_grant_metrics.py \
  --run-group month1_fusion_multiue_oai \
  --window-s 1.0
```

Run the full logging-validation bundle:

```bash
scripts/run_logging_validation_analysis.sh \
  --run-group month1_fusion_multiue_oai \
  --window-s 1.0
```

If gNB stdout was saved with `tee`, parse MAC summary blocks:

```bash
python3 scripts/parse_oai_gnb_mac_stats.py \
  --input metrics_logs/scenesense_ttracer/<run_group>/gnb/stdout/gnb_stdout.log \
  --output-dir metrics_logs/scenesense_ttracer/<run_group>/gnb/stdout_parsed
```

## Outputs To Check

Camera-only OD/SEG CSVs are written under:

```text
metrics_logs/
```

RGB+radar fusion run folders are written under:

```text
metrics_logs/scenesense_runs/<timestamp>_<run_group>/
```

Scenario evidence folders are written under:

```text
metrics_logs/scenesense_scenarios/<timestamp>_<scenario>_seed<seed>/
```

UE tunnel network samples are written under:

```text
metrics_logs/scenesense_network/<run_group>/
```

Application/network analysis summaries are written under:

```text
metrics_logs/scenesense_analysis/<analysis_label>/
```

T-tracer/RAN files are written under:

```text
metrics_logs/scenesense_ttracer/<run_group>/
```

Useful checks:

```bash
tail -n 5 metrics_logs/month1_camera_od_loopback_*.csv
tail -n 5 metrics_logs/month1_camera_seg_loopback_*.csv
curl http://127.0.0.1:35011/api/fusion_streams/latest | python3 -m json.tool
curl http://127.0.0.1:35011/api/spatial_map/latest | python3 -m json.tool
python3 scripts/analyze_scenesense_app_metrics.py --list-groups
```

## Interpretation Notes

- OD/SEG traffic traces provide bytes, chunks, front/back time, RTT, and timeout
  behavior.
- OD AP/mAP and SEG mIoU/foreground IoU require the corresponding GT/evaluation
  path; do not infer task quality from payload CSVs alone.
- For fusion, keep `fusion_as_od` and `fusion_as_seg` separate even though they
  come from the same runtime.
