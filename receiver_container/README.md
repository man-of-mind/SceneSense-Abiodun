# oai-perception-rx

A dedicated receiver-side container for the cooperative-perception demo.
Sits on the OAI core network as a peer of `oai-ext-dn`, so UE-originated
RTP traffic traverses the real 5G data plane (UE → gNB → UPF → bridge →
this container) instead of loopback.

## Topology

```
host (CARLA + sender)        OAI core (docker)              this container
─────────────────────        ────────────────────           ────────────────
sender                       gNB ─── UPF ─── bridge ─────── eth0 192.168.70.140
  source=10.0.0.2     ──►      (oai-cn5g-public-net)        receiver (Python)
  dest=192.168.70.140                                         GStreamer decode
                                                              cv2.imshow (X11)
                                                                 │
                                                                 ▼
                                                            host display
                                                            via /tmp/.X11-unix
```

The OAI core network is `oai-cn5g-public-net` (subnet `192.168.70.128/26`,
defined in [../OAI/oai-cn5g/docker-compose.yaml](../OAI/oai-cn5g/docker-compose.yaml)).
We attach as `external: true` so we never edit the team's OAI compose.

## Files

| file | purpose |
| --- | --- |
| [Dockerfile](Dockerfile) | CUDA runtime + Python3 + GStreamer (H.265) + OpenCV + PyTorch |
| [docker-compose.yaml](docker-compose.yaml) | base service definition, joins existing OAI bridge at .140 |
| [docker-compose.od-back.yaml](docker-compose.od-back.yaml) | Phase 2 overlay that requests GPU access and runs the OD back-half |
| [docker-compose.seg-back.yaml](docker-compose.seg-back.yaml) | overlay that requests GPU access and runs the LR-ASPP segmentation back-half |
| [docker-compose.fusion-back.yaml](docker-compose.fusion-back.yaml) | overlay that requests GPU access and runs RGB+radar fusion back-half worker(s) |
| [entrypoint.sh](entrypoint.sh) | adds the `10.0.0.0/16 via 192.168.70.134` return route (mirrors what oai-ext-dn does), then execs the receiver |

The receiver code lives in [../scan_receiver_v3_oai.py](../scan_receiver_v3_oai.py),
mounted read-only into the container at `/work/abiodun/`. Editing it on the
host and restarting the container picks up changes — no rebuild needed.

## First-time setup

```bash
# 1. OAI core must be up so the docker network exists.
../scripts/cn_start.sh

# 2. Allow container apps to draw on the host X display, build + start.
../scripts/receiver_container_up.sh

# 3. Watch the receiver come up.
sudo docker logs -f oai-perception-rx
# Expect:
#   [entrypoint] routes:
#     ... 10.0.0.0/16 via 192.168.70.134 ...
#   [receiver-oai] bind=0.0.0.0:65000
#   [receiver-oai] listening on 0.0.0.0:65000
```

The first build pulls Ubuntu 22.04 + apt packages — about 2 minutes. Later
runs reuse the image cache.

## Running the demo

CN + gNB + UE up (see [../scripts/README.md](../scripts/README.md)):

```bash
# Confirm UE is attached and the 5G data path is alive (iperf).
../scripts/ue_check.sh
../scripts/iperf3_uplink.sh client 1M 5    # quick health check

# Start the receiver container.
../scripts/receiver_container_up.sh

# Start CARLA, then in another terminal run the sender.
python3 ../scan_sender_v3_oai.py
```

A cv2 window titled `scan_receiver_v3_oai` opens on the host. Its HUD
shows live FPS and bind address. If FPS is close to the sender's framerate
(20), the 5G data path is sustaining the stream; if it's lower, look at
container logs and `tcpdump -i eth0` inside the container.

## Teardown

```bash
../scripts/receiver_container_down.sh
```

## OD back-half mode

The same container can run the split-inference object-detection back-half:

```bash
../scripts/receiver_container_od_back_up.sh
sudo docker logs -f oai-perception-rx
```

This uses `docker-compose.od-back.yaml`, mounts `../torch_cache` at
`/work/torch_cache`, and starts:

```bash
python3 -u /work/abiodun/carla_split_inference_udp_oai.py \
    --role back \
    --bind-host 0.0.0.0 \
    --remote-host 10.0.0.2 \
    --back-device cuda
```

Set `OD_BACK_EXTRA_ARGS="--disable-pretrained"` for a quick smoke test without
downloading COCO weights. The host front-half should bind to the UE tunnel:

```bash
python3 ../carla_split_inference_udp_oai.py \
    --role front \
    --bind-host 10.0.0.2 \
    --remote-host 192.168.70.140 \
    --headless
```

## Segmentation back-half mode

The same container can also run the LR-ASPP segmentation classifier head:

```bash
../scripts/receiver_container_seg_back_up.sh
sudo docker logs -f oai-perception-rx
```

This uses `docker-compose.seg-back.yaml`, mounts `../torch_cache` at
`/work/torch_cache`, and starts:

```bash
python3 -u /work/abiodun/carla_split_inference_udp_segmentation_oai.py \
    --role back \
    --bind-host 0.0.0.0 \
    --remote-host 10.0.0.2 \
    --back-device cuda
```

The host front-half should bind to the UE tunnel and send features to `.140`:

```bash
python3 ../carla_split_inference_udp_segmentation_oai.py \
    --role front \
    --bind-host 10.0.0.2 \
    --remote-host 192.168.70.140 \
    --front-device cuda \
    --camera-resolution 1080p
```

## RGB+radar fusion back-half mode

The same container can run the RGB+radar fusion split-inference back-half:

```bash
../scripts/receiver_container_fusion_back_up.sh
sudo docker logs -f oai-perception-rx
```

By default this starts two back-half workers in the same container, matching the
two pole-stream spatial-map baseline:

```text
stream 1: recv 0.0.0.0:51002, send -> 10.0.0.2:51004
stream 2: recv 0.0.0.0:51102, send -> 10.0.0.2:51104
```

For a single-worker smoke test:

```bash
FUSION_BACK_DUAL=0 ../scripts/receiver_container_fusion_back_up.sh
```

The host front-half commands should bind to the UE tunnel and send features to
`.140`:

```bash
python3 ../carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai.py \
    --role front \
    --bind-host 10.0.0.2 \
    --remote-host 192.168.70.140 \
    --sync-world \
    --traffic-light-id 14 \
    --camera-x 9 \
    --camera-y 2 \
    --camera-pitch -30 \
    --camera-yaw-offset 50 \
    --fusion-checkpoint checkpoints/fusion_object_best.pt \
    --front-device cuda \
    --headless
```

Use the same script for stream 2 with `--async-world`, the second camera pose,
and the `51101`-`51104` UDP port set.

## Troubleshooting

- **`cv2.imshow` fails / no window.** Confirm `xhost +local:` was run
  (the up script does this) and that `DISPLAY` is set on the host before
  running the up script.
- **Receiver shows 0 fps even though sender is running.** Check the route
  in the container: `sudo docker exec oai-perception-rx ip route`.
  You should see `10.0.0.0/16 via 192.168.70.134`. If not, restart the
  container.
- **Sender errors with "Cannot assign requested address" for `localaddr=10.0.0.2`.**
  The UE isn't attached or `oaitun_ue1` doesn't exist. Run `ue_check.sh`.
- **Stream arrives but looks corrupted / black.** Sender `pkt_size` mismatch.
  We set it to 1200 to match the GStreamer receiver MTU expectation; do
  not change without also adjusting the receiver caps.
