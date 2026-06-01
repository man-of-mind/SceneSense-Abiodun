# RGB+Radar Fusion OAI Multi-UE Runbook

This is the two-UE version of the fusion OAI transport baseline.

The goal is to stop treating both pole/car clients as if they share one UE.
Instead:

- Stream 1 binds to UE1: `10.0.0.2` / `oaitun_ue1`.
- Stream 2 binds to UE2: `10.0.0.3` / `oaitun_ue2`.
- The fusion back half still runs in `oai-perception-rx` at `192.168.70.140`.

## Why This Matters

The previous fusion OAI runbook used one simulated UE tunnel for both streams.
That was fine as a transport smoke test, but it does not represent separate
cars/poles. Multi-UE makes the network path more faithful: each client has its
own subscriber identity, tunnel IP, and return address.

## Terminal 1: Core and gNB

Start the OAI core as usual:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/scripts
./cn_start.sh
```

Start the gNB as usual in another terminal:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/scripts
./gnb_start.sh
```

## Terminal 2: Two UEs

Use the multi-UE start script instead of the old single-UE script:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/scripts
./ue_multi_start.sh
```

In another terminal, confirm both tunnels:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/scripts
./ue_multi_check.sh
```

Expected tunnel mapping:

```text
oaitun_ue1 -> 10.0.0.2
oaitun_ue2 -> 10.0.0.3
```

## Terminal 3: Fusion Back Half

Start the receiver container with separate return addresses for the two workers:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/scripts

FUSION_BACK_REMOTE_HOST_1=10.0.0.2 \
FUSION_BACK_REMOTE_HOST_2=10.0.0.3 \
FUSION_BACK_LOG_EVERY=30 \
./receiver_container_fusion_back_up.sh

sudo docker logs -f oai-perception-rx
```

If `oaitun_ue2` is already up, the launcher should now infer worker 2's return
address automatically. The explicit environment variables above are kept so the
mapping is obvious in logs and repeatable in notes.

Expected back-half mapping:

```text
worker 1: recv 0.0.0.0:51002, send result to 10.0.0.2:51004
worker 2: recv 0.0.0.0:51102, send result to 10.0.0.3:51104
```

For packet-level debugging, use `FUSION_BACK_LOG_EVERY=1`. If the logs print
`waiting for feature tensors...`, the front-to-back path is not reaching the
container. If the logs print frame/result lines but the front UI still waits,
the back-to-front return path is the likely issue.

## Terminal 4: Spatial-Map Server

```bash
source /home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/activate
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun

export MPLCONFIGDIR=/tmp/fusion_mplconfig

python3 real_time_spatial_map_server_fusion_object_v2.py \
  --object-yaw-map-offset-deg 10.0 \
  --focus-traffic-light-ids 14 \
  --focus-radius-m 20
```

Viewer:

```text
http://127.0.0.1:35011/api/spatial_map/viewer
```

## Metrics Grouping

No shared setup command is required for normal logging. Each stream writes its
own metrics folder and prints:

```text
[Metrics] Run directory: ...
[Metrics] Run group: ...
[Metrics] Stream CSV: ...
```

Streams started around the same time with the same `--transport-label` get the
same automatic `run_group`, so plotting can pair them by `run_group` +
`stream_id` even when the folders are different. Add the same
`--run-group <label>` to both stream commands only when you want an exact manual
experiment label.

For smoke tests, the automatic group is enough. For official experiment runs,
add a unique manual group to both stream commands, for example:

```bash
--run-group exp01_oai_clear_multiue
```

## Terminal 5: Network Metrics Logger

For OAI runs that should feed analysis, start this after both UE tunnels are up
and before launching the front halves. Use the same `--run-group` that the two
front-half commands use.

```bash
source /home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/activate
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun

python3 scripts/sample_oai_network_metrics.py \
  --run-group exp01_oai_clear_multiue \
  --ping-host 192.168.70.135
```

Stop it with Ctrl+C after both front halves stop. The analysis helper will load
the matching network CSV automatically.

For radio-side OAI metrics beyond tunnel counters, use the T-tracer smoke-test
workflow in [`TTRACER_SMOKE_RUNBOOK.md`](TTRACER_SMOKE_RUNBOOK.md). Use the same
manual `--run-group` label across the fusion front halves, tunnel sampler, and
T-tracer recorder so later plots can join the run cleanly.

## Terminal 6: Stream 1 Front Half on UE1

```bash
source /home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/activate
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun

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
  --run-group exp01_oai_clear_multiue \
  --result-timeout 1.5 \
  --headless
```

## Terminal 7: Stream 2 Front Half on UE2

```bash
source /home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/activate
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun

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
  --run-group exp01_oai_clear_multiue \
  --result-timeout 1.5 \
  --headless
```

After the run, optionally collect OAI/network snapshots into one of the printed
run folders:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/scripts

./collect_oai_run_logs.sh \
  /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/metrics_logs/scenesense_runs/<printed_run_folder>
```

## Success Criteria

- `ue_multi_check.sh` shows both `oaitun_ue1` and `oaitun_ue2`.
- UE1 has `10.0.0.2`; UE2 has `10.0.0.3`.
- The receiver logs show worker 1 returning to `10.0.0.2` and worker 2 returning to `10.0.0.3`.
- Both front clients receive mask/object outputs.
- The spatial-map viewer updates from both stream IDs.

## Notes

The official OAI Docker examples use a different demo core namespace
(`192.168.71.x`) and UE subnet (`12.1.1.x`). Our existing working setup uses
`192.168.70.x` for the core/container network and `10.0.0.x` for UE tunnels, so
this runbook keeps the current working addressing instead of swapping to the
stock Docker demo network.

The official Docker multi-UE route is still useful, but it puts each UE tunnel
inside Docker/container networking. Our CARLA front clients currently run on the
host and bind directly to UE tunnel IPs. For that reason, the first multi-UE
bridge uses host-side `nr-uesoftmodem --num-ues 2`, which should expose
`oaitun_ue1` and `oaitun_ue2` to the same host namespace as the CARLA clients.
