# RGB+Radar Fusion OAI 5G Runbook

This runbook uses the editable `abiodun/` fusion pipeline and moves the
split-inference back half into the `oai-perception-rx` container at
`192.168.70.140`.

This is a transport baseline only: no intentional low-SNR, bandwidth throttling,
or resource stress is applied here.

This file is the single-UE transport baseline. For separate UE tunnels per
stream, use [FUSION_OAI_MULTI_UE_RUNBOOK.md](FUSION_OAI_MULTI_UE_RUNBOOK.md).

## Expected Flow

```text
pole front half on UE host
  bind 10.0.0.2
  send feature tensors to 192.168.70.140

OAI 5G path
  UE tunnel -> gNB -> UPF -> OAI docker bridge

fusion back half in oai-perception-rx
  bind 0.0.0.0
  return mask + object results to 10.0.0.2

pole front half
  draw mask/object overlay
  publish object stream to spatial-map server
```

## Terminal 1: OAI Core, gNB, and UE

Use the existing OAI bring-up from [scripts/README.md](scripts/README.md).

Confirm the UE tunnel is up:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/scripts
./ue_check.sh
```

Expected UE IP:

```text
10.0.0.2
```

## Terminal 2: Fusion Back Half in the Container

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/scripts

./receiver_container_fusion_back_up.sh
sudo docker logs -f oai-perception-rx
```

Expected log lines:

```text
[fusion-back] device=cuda recv 0.0.0.0:51002, send -> 10.0.0.2:51004
[fusion-back] device=cuda recv 0.0.0.0:51102, send -> 10.0.0.2:51104
```

For a single-worker smoke test:

```bash
FUSION_BACK_DUAL=0 ./receiver_container_fusion_back_up.sh
```

## Terminal 3: Spatial-Map Server

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

## Terminal 4: Network Metrics Logger

For OAI runs that should feed analysis, start this after the UE tunnel is up and
before launching front halves. Use the same `--run-group` that the front-half
commands use.

```bash
source /home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/activate
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun

python3 scripts/sample_oai_network_metrics.py \
  --run-group exp01_oai_clear_singleue \
  --interface oaitun_ue1:ue1 \
  --ping-host 192.168.70.135
```

Stop it with Ctrl+C after the front halves stop.

## Terminal 5: Stream 1 Front Half

```bash
source /home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/activate
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun

python carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai.py \
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
  --run-group exp01_oai_clear_singleue \
  --result-timeout 1.5 \
  --headless
```

For a short smoke test, add:

```bash
--max-frames 120
```

## Terminal 6: Stream 2 Front Half

```bash
source /home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/activate
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun

python carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai.py \
  --role front \
  --bind-host 10.0.0.2 \
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
  --run-group exp01_oai_clear_singleue \
  --result-timeout 1.5 \
  --headless
```

For a short smoke test, add:

```bash
--max-frames 120
```

## Success Criteria

- Container logs show one or two `[fusion-back]` workers.
- Stream clients print `Role: front | bind-host: 10.0.0.2 | remote-host: 192.168.70.140`.
- Pole clients receive segmentation masks and object/localization outputs.
- Spatial-map viewer updates from the active stream IDs.
- No intentional network impairment is applied in this baseline.
