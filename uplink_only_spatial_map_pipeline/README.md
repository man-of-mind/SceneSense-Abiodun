# Uplink-only spatial-map split-inference pipeline

Created: 2026-07-28

Purpose: prototype the realistic SceneSense path without modifying the existing
closed-loop CARLA split-inference code.

## Why this folder exists

The existing live deployment pipeline sends split features from the CARLA/front
side to the edge/back side, then sends detection results back to the car. The
front waits for those results before logging/publishing many downstream outputs.
That made sense for visualization/debugging, but it is not the final
spatial-map architecture.

The intended project architecture is:

```text
car sensors
  -> split feature uplink
  -> edge tail model
  -> detections
  -> spatial-map application
  -> small asynchronous warning/control message to relevant car(s)
```

The car does not need full detection masks/boxes returned every frame just to
draw overlays. The spatial map needs the detections to reason about visibility,
occlusion, association, and warnings.

## Baseline files copied here

- `carla_fusion_staleness_scenario_uplink_only.py`
  - copied from `../staleness/carla_fusion_staleness_scenario.py`
  - **current reportable Track-1 base** because it matches the corrected
    drivable deployment path used by `../downlink_latency_fps/run_common.sh`.
  - includes the corrected route/NPC/radar flags such as
    `--radar-temporal-window-frames` and `--npc-speed-difference-pct`.
- `carla_split_inference_udp_fusion_object_uplink_only_spatial.py`
  - copied from
    `../carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai.py`
  - prototype only; useful for debugging, but do **not** use for reportable
    Track-1 experiments because it is not the current corrected deployment path.
- `spatial_map_server_moving_ego_uplink_only_baseline.py`
  - copied from `../spatial_map_coop/spatial_map_server_moving_ego.py`
  - local baseline reference only; do not diverge unless the uplink-only payload
    requires server-side changes.

## Current closed-loop behavior to remove/decouple

In the original client, the front loop:

1. captures RGB/radar;
2. runs the split front/backbone;
3. sends feature tensors uplink;
4. waits for the back-half result using `result_store.wait_for(...)`;
5. only publishes to the spatial map inside `if result is not None`.

That means spatial-map updates are tied to result-return latency/timeouts. Over
OAI this reduces the effective send rate and creates the sparse burst/wait
traffic pattern observed in the diagnostics.

## Intended uplink-only behavior

For the new variant, the edge/back half should publish detections to the
spatial-map server immediately after tail inference:

```text
front/UE:
  capture -> front model -> send features
  no per-frame wait for result display

back/edge:
  receive features -> tail inference -> decode objects
  publish objects + camera pose/provenance to spatial map
  optional: send tiny ACK/metrics only, not full mask/objects
```

This lets us test the realistic spatial-map workload:

- uplink feature transport;
- edge inference;
- map ingestion/update latency;
- later, small asynchronous downlink warning messages.

## Key experiment question

Does vanilla OAI still assign low MCS and build RLC backlog when CARLA split
features are sent as an uplink-only stream, without the artificial car-side
result wait?

Possible outcomes:

| Outcome | Interpretation |
|---|---|
| MCS improves and queue shrinks | closed-loop result wait was a major cause of the sparse problematic pattern |
| MCS remains low and queue persists | split-feature uplink itself exposes OAI link-adaptation weakness |
| queue grows worse at true 10 FPS | no-AE payload is too large unless MCS is high or compression/ROI is used |
| reduced payload works | supports the RL/compression policy direction for map freshness |

## Metrics to keep

For comparability with the OAI diagnostics:

- front/back compute latency;
- feature payload bytes/chunks;
- edge tail latency;
- spatial-map publish latency;
- map-ingest timestamp;
- UE RLC LCID4 occupancy;
- UE BSR LCG1 backlog;
- UL MCS/PRB/TBS;
- retransmission/BLER branch metrics;
- effective frame send rate;
- spatial-map update rate and staleness.

## Separate ongoing track: closed-loop scheduler/MCS diagnostics

The existing closed-loop CARLA pipeline remains useful even if it is not the
final SceneSense architecture. It is a good stress workload for edge-AI uplink
bursts and scheduler/link-adaptation studies.

Do not collapse these two tracks:

1. **Uplink-only spatial-map pipeline**
   - realistic project architecture;
   - answer: what does SceneSense actually need?
2. **Closed-loop CARLA diagnostics**
   - controlled edge-AI burst workload;
   - answer: how do OAI scheduler/MCS policies behave under split-feature
     traffic?

## Scheduler/MCS policy ideas to test on the closed-loop diagnostics track

The existing hold-MCS patch should be treated as a diagnostic, not a final
solution. It proved that low MCS drives the RLC bottleneck, but under bad
channel/AWGN it can hold MCS too high and create retransmissions.

Policies worth comparing:

- vanilla OAI BLER/OLLA behavior;
- hold-MCS/few-samples guard;
- decrement only on real high BLER/HARQ evidence;
- AIMD/TCP-Reno-like MCS adaptation:
  - additive or slow increase after clean transmissions;
  - multiplicative/conservative decrease after real BLER/retransmission;
  - no decrease solely due to sparse traffic/inactivity;
- SNR/CQI-bounded MCS selection;
- known scheduler/link-adaptation algorithms from literature/OAI variants,
  if implementable in the current OAI branch.

The senior-supervisor question to answer:

> Is split-inference traffic genuinely different, or is the observed behavior
> mainly a limitation of vanilla OAI's simple MCS/link-adaptation logic?

## Near-term implementation steps

Status: **corrected-source Track-1 implementation is in this folder.**

Implemented in
`carla_fusion_staleness_scenario_uplink_only.py`:

- `--uplink-only-spatial-map` decouples the CARLA/front loop from the result
  return path.
- `--edge-result-mode auto|full|ack|none` keeps old closed-loop behavior by
  default, but uses no result downlink in uplink-only mode.
- the front feature payload now carries `stream_id`, `carla_timestamp`, camera
  pose/matrix, and a per-frame timing block.
- in loopback/back mode, the edge worker can publish detections directly to the
  spatial-map UDP server immediately after tail inference.
- `--edge-receive-queue-size` decouples UDP receive from tail inference; auto
  mode uses a 32-frame bounded queue in uplink-only mode and drops oldest frames
  if needed to preserve freshness.
- `--uplink-drain-grace-s` can wait briefly after a short front run so already
  received frames can flush to the map before shutdown.
- `--edge-metrics-csv` logs receive, tail, map-publish, and capture-to-edge
  timing from the edge side; a companion `.summary.json` records final edge
  counts such as processed frames and incomplete UDP feature messages dropped.
- the corrected staleness/deployment flags remain available, including
  `--radar-temporal-window-frames`, `--npc-speed-difference-pct`,
  queue-probe logging, tracked-target controls, and Experiment-3 controls.
- `--radar-rasterizer legacy|fast` selects the radar tensor rasterizer. The
  default remains `legacy`; `fast` is the validated vectorized path for
  throughput experiments.

Implemented in
`spatial_map_server_moving_ego_uplink_only_baseline.py`:

- `--ingest-metrics-csv` logs map ingest/update timing per received spatial
  packet.
- `--map-update-delay-ms` can emulate future spatial-map compute cost such as
  association, occlusion reasoning, or warning policy logic.

Completed:

1. Identify the minimum payload metadata the back half needs to publish directly
   to the spatial map:
   - stream id;
   - frame id/timestamps;
   - camera transform/matrix/intrinsics;
   - camera size/FoV;
   - decoded objects;
   - latency/provenance.
2. Add an edge-side spatial-map publisher to the copied uplink-only client.
3. Add a front option to avoid waiting for full result payloads.
4. Optionally return a tiny ACK/metrics payload to the car, or no result at all.
5. Run corrected-source ideal loopback no-return tests.
6. Add and run exact-FPS model-boundary/map offered-load replay:
   `replay_spatial_map_offered_load.py` and
   `run_track1_map_offered_load_replay.sh`.
7. Add fine-grained live frontend preparation timers to locate why the live
   CARLA run does not actually offer true 10/20 model-ready FPS.
8. Profile and optimize radar tensor construction in the live frontend.
   Findings are documented in `RADAR_RASTERIZER_OPTIMIZATION.md`.

Latest completed baseline:

9. Ran uplink-only Track-1 over default OAI 106PRB with UE/gNB t-tracer
   enabled. Results are in `TRACK1_OAI_DEFAULT106_RESULTS.md`; clean plots are
   under `plots/track1_oai_default106/`.

Next:

10. Repeat Track-1 over OAI with reduced feature payload knobs to test whether
    the RLC/BSR backlog and incomplete multi-chunk UDP reassembly drops shrink.
11. Add a real map-worker processing stage when we are ready to replace the
    current explicit `+30 ms` assumed map-compute budget.

## First loopback run recipe

Use the corrected drivable-scene spirit from `../downlink_latency_fps/run_common.sh`:
moving ego, fixed route, 28 vehicles, 35 pedestrians, seed 31, 200k radar,
no-AE checkpoint, per-channel uint8, zstd level 3.

Use `carla_fusion_staleness_scenario_uplink_only.py`, not the older prototype
copy, for any reportable loopback/OAI Track-1 run.

Terminal 1 — spatial-map server:

```bash
mkdir -p abiodun/uplink_only_spatial_map_pipeline/runs/loopback_10fps_no_delay
MPLCONFIGDIR=/tmp/matplotlib \
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python \
  abiodun/uplink_only_spatial_map_pipeline/spatial_map_server_moving_ego_uplink_only_baseline.py \
  --udp-host 127.0.0.1 \
  --udp-port 39201 \
  --api-host 127.0.0.1 \
  --api-port 5088 \
  --render-hz 4 \
  --focus-follow-stream-id uplink_loopback_10fps \
  --focus-radius-m 80 \
  --focus-follow-forward-bias 0.35 \
  --ingest-metrics-csv abiodun/uplink_only_spatial_map_pipeline/runs/loopback_10fps_no_delay/map_ingest_metrics.csv \
  --map-update-delay-ms 0
```

Terminal 2 — CARLA split-inference uplink-only loopback:

```bash
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python \
  abiodun/uplink_only_spatial_map_pipeline/carla_fusion_staleness_scenario_uplink_only.py \
  --role loopback \
  --uplink-only-spatial-map \
  --edge-result-mode auto \
  --sync-world \
  --fps 10 \
  --seed 31 \
  --sensor-platform ego_vehicle \
  --no-ego-freeze \
  --ego-ignore-lights-pct 50 \
  --ego-disable-lane-change \
  --ego-fixed-path-spawn-indices 80,85,91,94,99,80 \
  --ego-fixed-path-loop \
  --ego-spawn-index 80 \
  --ego-spawn-z-offset-m 0.15 \
  --camera-resolution custom \
  --camera-width 1280 \
  --camera-height 720 \
  --camera-fov 120 \
  --model-input-width 768 \
  --model-input-height 432 \
  --ego-camera-x 1.8 \
  --ego-camera-y 0.0 \
  --ego-camera-z 1.55 \
  --ego-camera-pitch -4.0 \
  --ego-camera-yaw 0.0 \
  --ego-radar-yaw 0.0 \
  --radar-hfov 120 \
  --radar-vfov 30 \
  --radar-range 120 \
  --radar-points-per-second 200000 \
  --radar-raster-radius-px 4 \
  --radar-rasterizer fast \
  --radar-temporal-window-frames 2 \
  --npc-vehicles 28 \
  --npc-pedestrians 35 \
  --spawn-radius 80 \
  --npc-speed-difference-pct 10 \
  --fusion-checkpoint abiodun/experiments/ae_integrated_20260710/noae_baseline/checkpoints/mprime_joint_noae/best.pt \
  --quantization-mode per_channel_uint8 \
  --entropy-coder zstd \
  --zstd-level 3 \
  --roi-threshold 0.0 \
  --spatial-map-stream \
  --spatial-map-host 127.0.0.1 \
  --spatial-map-port 39201 \
  --spatial-map-stream-id uplink_loopback_10fps \
  --edge-metrics-csv abiodun/uplink_only_spatial_map_pipeline/runs/loopback_10fps_no_delay/edge_uplink_metrics.csv \
  --metrics-run-dir abiodun/uplink_only_spatial_map_pipeline/runs/loopback_10fps_no_delay/front_metrics \
  --transport-label uplink_only_loopback_zstd_noae \
  --run-group uplink_only_loopback_10fps_no_delay \
  --run-id uplink_only_loopback_10fps_no_delay \
  --headless \
  --max-frames 600 \
  --uplink-drain-grace-s 5 \
  --front-device cuda \
  --back-device cuda \
  --back-log-every 100
```

Repeat with `--fps 20`, and then repeat both runs with
`--map-update-delay-ms 40` on the spatial-map server.
