# OAI helper scripts

Quick-spin commands for the OAI 5G stack running in [/abiodun/OAI/](../OAI/).
All paths and IPs live in [config.env](config.env) — edit that one file if
anything moves.

## Layout

| script | what it does |
| --- | --- |
| [config.env](config.env) | shared paths + endpoints; every other script sources this |
| [cn_start.sh](cn_start.sh) | `docker compose up -d` for the 5G Core |
| [cn_stop.sh](cn_stop.sh) | `docker compose down` |
| [cn_status.sh](cn_status.sh) | container state + grep AMF/SMF logs for the UE IP |
| [gnb_start.sh](gnb_start.sh) | `nr-softmodem --rfsim` (blocks — its own terminal) |
| [gnb_start_ttracer.sh](gnb_start_ttracer.sh) | gNB launcher with OAI T-tracer enabled on `OAI_GNB_T_PORT` |
| [ue_start.sh](ue_start.sh) | `nr-uesoftmodem --rfsim` (blocks — its own terminal) |
| [ue_multi_start.sh](ue_multi_start.sh) | two UE tunnels in one `nr-uesoftmodem --num-ues 2` process |
| [ue_multi_start_ttracer.sh](ue_multi_start_ttracer.sh) | multi-UE launcher with OAI T-tracer enabled on `OAI_UE_T_PORT` |
| [ue_check.sh](ue_check.sh) | `ip addr` on `oaitun_ue1`, ping AMF/SMF/ext-DN through it |
| [ue_multi_check.sh](ue_multi_check.sh) | checks `oaitun_ue1` and `oaitun_ue2` for multi-client tests |
| [collect_oai_run_logs.sh](collect_oai_run_logs.sh) | captures OAI/network snapshots into a SceneSense run folder |
| [sample_oai_network_metrics.py](sample_oai_network_metrics.py) | samples UE tunnel bitrate/counters/ping during a run |
| [analyze_scenesense_app_metrics.py](analyze_scenesense_app_metrics.py) | summarizes and plots SceneSense application metrics, plus matching network metrics, by `run_group` |
| [parse_oai_gnb_mac_stats.py](parse_oai_gnb_mac_stats.py) | parses gNB MAC stdout summaries into BLER/HARQ/SNR/MAC-byte CSVs |
| [analyze_nrue_grant_metrics.py](analyze_nrue_grant_metrics.py) | summarizes `NRUE_MAC_DCI_GRANT.csv` into per-RNTI/window UE network-state features |
| [compare_nrue_gnb_grants.py](compare_nrue_gnb_grants.py) | validates UE decoded grants against gNB MAC/PHY T-tracer totals |
| [validate_nrue_grant_payload.py](validate_nrue_grant_payload.py) | validates UE decoded UL grant `tbs*8` against OAI's existing UE payload-bits trace |
| [run_logging_validation_analysis.sh](run_logging_validation_analysis.sh) | one-command post-processing for app, tunnel, UE grant, gNB, and validation outputs |
| [ttracer_build_tools.sh](ttracer_build_tools.sh) | builds the local OAI T-tracer tools used by smoke tests |
| [ttracer_record_smoke.sh](ttracer_record_smoke.sh) | records a short gNB or UE T-tracer raw file |
| [ttracer_extract_csv_smoke.sh](ttracer_extract_csv_smoke.sh) | replays a raw T-tracer file and extracts selected RAN CSVs |
| [iperf2_uplink.sh](iperf2_uplink.sh) | iperf2 UDP UE → ext-DN, `server` or `client` mode |
| [iperf3_uplink.sh](iperf3_uplink.sh) | same, iperf3 |
| [receiver_container_up.sh](receiver_container_up.sh) | build + start the `oai-perception-rx` container on the OAI network |
| [receiver_container_down.sh](receiver_container_down.sh) | stop it |

## Bring-up order (rfsim mode)

```bash
# terminal 1
./cn_start.sh
./cn_status.sh         # wait until oai-amf / oai-smf are Up + healthy

# terminal 2
./gnb_start.sh

# terminal 3
./ue_start.sh          # wait for "5GMM-REGISTERED" / oaitun_ue1 to appear

# terminal 4 — sanity check
./ue_check.sh
```

## First video demo over 5G

The video receiver runs in its own container ([../receiver_container/](../receiver_container/))
at `192.168.70.140` on the same docker network as `oai-ext-dn`. UE-bound RTP
traffic from the host traverses gNB → UPF → bridge → container, exactly like
iperf did.

```bash
# CN, gNB, UE already up; UE attached (see ue_check.sh).
./receiver_container_up.sh           # builds the image first time (~2 min)
sudo docker logs -f oai-perception-rx  # watch receiver come up

# in another terminal — start CARLA and the sender
python3 ../scan_sender_v3_oai.py
```

A cv2 window labeled `scan_receiver_v3_oai` should open on the host display,
showing the CARLA camera feed with a `bind=0.0.0.0:65000` HUD. FPS in the
HUD = end-to-end sustained rate over the 5G data plane.

Teardown: `./receiver_container_down.sh`.

## Split-inference OD over 5G

Phase 1 host-only checks should pass before using this path:

```bash
# terminal A — back-half on localhost
cd ../abiodun
python3 -u carla_split_inference_udp_oai.py \
    --role back \
    --bind-host 127.0.0.1 \
    --remote-host 127.0.0.1 \
    --disable-pretrained \
    --back-device cpu

# terminal B — front-half on localhost, CARLA already running
python3 carla_split_inference_udp_oai.py \
    --role front \
    --bind-host 127.0.0.1 \
    --remote-host 127.0.0.1 \
    --headless \
    --disable-pretrained \
    --front-device cpu \
    --metrics-warmup-frames 0 \
    --disable-live-plot
```

Phase 2 runs the back-half in the perception container with GPU access. Docker
must report an NVIDIA runtime first (`sudo docker info | grep -i runtime`).

```bash
# CN, gNB, UE already up; UE attached (see ue_check.sh).
./receiver_container_od_back_up.sh
sudo docker logs -f oai-perception-rx
# Expect:
#   [back] device=cuda recv 0.0.0.0:36001, send -> 10.0.0.2:36003

# in another terminal — front half on the host, bound to the UE tunnel
cd ..
python3 carla_split_inference_udp_oai.py \
    --role front \
    --bind-host 10.0.0.2 \
    --remote-host 192.168.70.140 \
    --headless \
    --front-device cuda \
    --metrics-warmup-frames 0 \
    --disable-live-plot
```

Use `--front-device cuda` for the Phase 2 comparison against the original
single-process demo, where both detector halves run on GPU when CUDA is
available. Switch it to `cpu` only when intentionally isolating front-half CPU
cost.

For a no-download smoke test of the container path, set:

```bash
OD_BACK_EXTRA_ARGS="--disable-pretrained" ./receiver_container_od_back_up.sh
```

For the real comparison run, omit `OD_BACK_EXTRA_ARGS` so torchvision loads the
COCO Faster R-CNN weights into the shared `abiodun/torch_cache/` volume. Compare
the resulting front-side metrics CSV against a `--role loopback` baseline.

## Split-inference segmentation over 5G

The LR-ASPP segmentation path mirrors the OD deployment, but uses separate UDP
ports (`36100`-`36103`) so it does not collide with an OD run.

```bash
# CN, gNB, UE already up; UE attached (see ue_check.sh).
./receiver_container_seg_back_up.sh
sudo docker logs -f oai-perception-rx
# Expect:
#   [seg-back] device=cuda recv 0.0.0.0:36101, send -> 10.0.0.2:36103

# in another terminal — front half on the host, bound to the UE tunnel
cd ..
python3 carla_split_inference_udp_segmentation_oai.py \
    --role front \
    --bind-host 10.0.0.2 \
    --remote-host 192.168.70.140 \
    --camera-resolution 1080p \
    --front-device cuda \
    --metrics-warmup-frames 0
```

The segmentation back-half sends the predicted mask at the LR-ASPP model input
size by default (`--mask-output-size model`) and the front resizes it for the
GUI. That keeps the return packet much smaller than sending a full 1080p mask.
Use `SEG_BACK_EXTRA_ARGS="--seg-disable-pretrained"` only for a quick transport
smoke test.

## Split-inference RGB+radar fusion over 5G

The fusion path mirrors the local two-pole baseline but moves the model back
half into the `.140` perception container. The container starts two back-half
workers by default so both pole streams can run at the same time.

```bash
# CN, gNB, UE already up; UE attached (see ue_check.sh).
./receiver_container_fusion_back_up.sh
sudo docker logs -f oai-perception-rx
# Expect:
#   [fusion-back] device=cuda recv 0.0.0.0:51002, send -> 10.0.0.2:51004
#   [fusion-back] device=cuda recv 0.0.0.0:51102, send -> 10.0.0.2:51104

# terminal B — spatial-map server on the host
cd ..
python3 real_time_spatial_map_server_fusion_object_v2.py \
    --object-yaw-map-offset-deg 10.0 \
    --focus-traffic-light-ids 14 \
    --focus-radius-m 20

# terminal C — stream 1 front half on the host, bound to the UE tunnel
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
    --result-timeout 1.5 \
    --headless

# terminal D — stream 2 front half on the host, bound to the UE tunnel
python3 carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai.py \
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
    --result-timeout 1.5 \
    --headless
```

For a one-stream smoke test of the container path:

```bash
FUSION_BACK_DUAL=0 ./receiver_container_fusion_back_up.sh
```

For the two-UE version, use [../FUSION_OAI_MULTI_UE_RUNBOOK.md](../FUSION_OAI_MULTI_UE_RUNBOOK.md).
The key change is that stream 1 binds to `10.0.0.2`, stream 2 binds to
`10.0.0.3`, and the fusion back-half container is started with:

```bash
FUSION_BACK_REMOTE_HOST_1=10.0.0.2 \
FUSION_BACK_REMOTE_HOST_2=10.0.0.3 \
./receiver_container_fusion_back_up.sh
```

## OAI T-tracer smoke test

For radio-side metrics, use the dedicated runbook:
[`../TTRACER_SMOKE_RUNBOOK.md`](../TTRACER_SMOKE_RUNBOOK.md).

Short version:

```bash
./ttracer_build_tools.sh
./gnb_start_ttracer.sh
./ue_multi_start_ttracer.sh

./ttracer_record_smoke.sh --run-group exp02_ttracer_smoke --source gnb --duration-s 60
./ttracer_record_smoke.sh --run-group exp02_ttracer_smoke --source ue --duration-s 60

./ttracer_extract_csv_smoke.sh --run-group exp02_ttracer_smoke --source gnb
./ttracer_extract_csv_smoke.sh --run-group exp02_ttracer_smoke --source ue --clean-output

python3 analyze_nrue_grant_metrics.py --run-group exp02_ttracer_smoke
```

For the full logging-validation pass after an official run, use:

```bash
./run_logging_validation_analysis.sh --run-group exp02_ttracer_smoke
```

This writes raw traces and extracted CSVs under
`../metrics_logs/scenesense_ttracer/<run_group>/`.

The default UE profile is intentionally clean: it records/extracts only the
local SceneSense event `NRUE_MAC_DCI_GRANT`. Use `--profile payload` if you
also want `UE_PHY_UL_PAYLOAD_TX_BITS` for validation, or `--profile legacy` if
you intentionally want the older UE PHY measurement/DCI CSVs. Rebuild the UE
softmodem after changing `common/utils/T/T_messages.txt`; the CSV will stay
empty or the event will be unknown until the generated T headers are refreshed
in the OAI build.

If you tee the gNB terminal output, parse the MAC summary blocks with:

```bash
python3 parse_oai_gnb_mac_stats.py \
  --input <run_dir>/oai_logs/gnb_stdout.log \
  --output-dir <run_dir>/oai_logs/gnb_mac_parsed
```

## iperf sanity tests

iperf2:
```bash
# terminal A
./iperf2_uplink.sh server
# terminal B
./iperf2_uplink.sh client 10M
```

iperf3:
```bash
./iperf3_uplink.sh server
./iperf3_uplink.sh client 1M 10
```

## 5QI experiments — where this is heading

OAI configures per-PDU-session QoS in the SMF/PCF YAML. Reference tutorial:
<https://openairinterface-docs-5b3d70.eurecom.io/projects/cn5g/Tutorials/QoS/>.

3GPP 5QI table (TS 23.501 §5.7.4): a handful of candidates worth running for
feature-transmission experiments —

| 5QI | type | PDB | PER | example use | why it matters for us |
| --- | --- | --- | --- | --- | --- |
| 9 | non-GBR | 300 ms | 1e-6 | default TCP | baseline, what you get today |
| 7 | non-GBR | 100 ms | 1e-3 | voice/video | stricter delay, looser loss |
| 2 | GBR | 150 ms | 1e-3 | conv. video | guaranteed bitrate path |
| 82 | delay-critical GBR | 10 ms | 1e-4 | discrete automation | the closest analogue to "safety-critical AV features" |
| 83 | delay-critical GBR | 10 ms | 1e-4 | discrete automation (small payload) | same delay, different payload class |
| 79 | non-GBR | 50 ms | 1e-2 | V2X messages | the V2X-tagged QCI for vehicle traffic |

What I'd actually measure once the iperf path is verified:

1. Fix scene + payload (one of: scan_sender_v1 detection features, segmentation features, multi-sensor).
2. Sweep 5QI ∈ {9, 7, 82, 79} on the PDU session carrying our UDP stream.
3. Per run, log: end-to-end frame latency, jitter, packet loss, **downstream
   mAP / mIoU** on the server side.
4. Plot accuracy-vs-5QI under fixed background load (use iperf to add cross-traffic).

That last point is the actual research signal — does the 3GPP-defined
guarantee actually translate to better task accuracy under contention? If yes,
"select 5QI per feature class" becomes a concrete contribution. If no, that's
also a finding (and a known criticism of static QoS classes).

This feeds straight into [`../research_direction.md`](../research_direction.md)
§4 (network axis) and §8 (RL agent inputs).
