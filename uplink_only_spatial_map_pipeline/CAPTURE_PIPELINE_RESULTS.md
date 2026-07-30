# Track-1 capture-pipeline diagnostic

Date: 2026-07-29

Purpose: test whether the uplink-only Track-1 live frontend is being limited by
the downstream encode/send/tail/map consumer path, or by CARLA sensor production
and front-side preparation.

## Code change

Added an opt-in bounded producer/consumer mode to
`carla_fusion_staleness_scenario_uplink_only.py`:

- `--capture-pipeline`
- `--capture-pipeline-queue-size`
- `--capture-pipeline-drop-oldest` / `--no-capture-pipeline-drop-oldest`

The default path is unchanged. The tested mode used queue size 2 and
`--no-capture-pipeline-drop-oldest`, so it preserves frames and exposes
backpressure instead of silently dropping prepared frames.

New timing fields now propagate from front to edge to map:

- `capture_pipeline_queue_wait_ms`
- `capture_pipeline_queue_depth`

## Runs

Baseline sequential fast-rasterizer run:

`runs/track1_ideal_loopback_matrix_20260729_fast_throughput`

Capture-pipeline runs:

- `runs/track1_ideal_loopback_matrix_20260729_fast_pipeline_10fps`
- `runs/track1_ideal_loopback_matrix_20260729_fast_pipeline_fps_sweep`

Setup for all rows:

- ideal loopback buffers active;
- no-AE baseline, ROI 0, `per_channel_uint8`;
- zstd feature transport;
- 200k radar PPS;
- fast radar rasterizer;
- drivable route `80,85,91,94,99,80`;
- 28 requested vehicles, 35 pedestrians, seed 31;
- artificial map delay 0 ms.

## Result table

| Target FPS | Mode | Sent to edge/map | Delivery | Actual send FPS | Send period p50 / p95 | Radar pts/frame p50 | Uplink payload p50 | Capture-pipeline queue wait p50 | Queue depth p95 | Capture→backbone p50 | Backbone→tail p50 | Backbone→map-update p50 | Capture→map-update p50 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | sequential | 180/180 | 100.0% | 6.37 | 154.8 / 201.5 ms | 19,896 | 1041.6 KB | - | - | 50.4 ms | 21.8 ms | 31.4 ms | 83.0 ms |
| 10 | capture pipeline q2 | 179/179 | 100.0% | 7.22 | 139.4 / 180.2 ms | 19,896 | 1048.0 KB | 0.1 ms | 0 | 45.1 ms | 24.0 ms | 33.7 ms | 81.4 ms |
| 20 | sequential | 180/180 | 100.0% | 8.14 | 122.2 / 158.1 ms | 9,950 | 1039.5 KB | - | - | 33.4 ms | 22.9 ms | 33.4 ms | 69.3 ms |
| 20 | capture pipeline q2 | 179/179 | 100.0% | 9.32 | 108.3 / 141.4 ms | 9,950 | 1042.8 KB | 0.1 ms | 0 | 33.2 ms | 25.6 ms | 35.0 ms | 70.9 ms |
| 30 | sequential | 180/180 | 100.0% | 9.36 | 104.3 / 131.3 ms | 6,619 | 1038.0 KB | - | - | 30.0 ms | 22.2 ms | 33.9 ms | 65.3 ms |
| 30 | capture pipeline q2 | 179/179 | 100.0% | 10.89 | 91.5 / 120.8 ms | 6,618 | 1039.0 KB | 0.1 ms | 0 | 27.9 ms | 27.9 ms | 38.3 ms | 67.5 ms |

Each capture-pipeline run attempted 180 scheduled frames but sent 179 because
the first post-warmup camera wait timed out once. After that startup miss,
sent-frame delivery to the edge and map was 100%.

## Interpretation

The capture pipeline improves the live frontend cadence, but it does not make
the 10 FPS / full-density 200k-PPS run reach true 10 FPS.

The key clue is that `capture_pipeline_queue_wait_ms` is about 0.1 ms p50 and
`capture_pipeline_queue_depth` has p95 0. The prepared-frame queue is almost
always empty. Therefore the consumer side is not waiting under a growing
backlog; the producer side is still pacing the run.

For the 10 FPS full-density row, the main p50 producer-side costs are:

- CARLA synchronous tick: about 71.6 ms in the pipeline run;
- camera frame wait: about 33.1 ms;
- radar tensor build: about 26.4 ms;
- capture-to-backbone input: about 45.1 ms, excluding the CARLA tick and camera
  wait as currently timestamped.

So the immediate conclusion is:

> Decoupling capture/prep from encode/send/tail/map helps modestly, but the
> remaining Track-1 live-loop ceiling is still dominated by CARLA tick/camera
> production plus front-side sensor preparation, not by map ingest or edge tail.

The model/uplink/tail/map path remains stable on ideal loopback. At the model
boundary, `backbone_input_to_map_update_done_ms` stays around 34-38 ms p50
across the capture-pipeline sweep.

## Next implication

For reporting Track-1 freshness, keep both latency definitions visible:

- `backbone_input_to_map_update_done_ms`: model-boundary to spatial-map update;
- full capture freshness: CARLA sensor capture/tick/camera wait + preparation +
  model/uplink/tail/map.

For optimization, the next useful target is not the edge/map consumer queue; it
is producer-side scheduling and sensor acquisition:

- reduce/parallelize CARLA tick + camera wait overhead if possible;
- keep fast radar rasterizer;
- consider a stricter asynchronous architecture only if we can preserve
  frame-to-radar alignment and avoid uncontrolled frame drops.

## CARLA-side timing diagnostics

Follow-up question: can the remaining CARLA-side overhead be reduced without
changing the experiment meaning?

Two low-risk diagnostics were run on the same 10 FPS / fast-rasterizer /
capture-pipeline setup.

### 1. Emit sensors every synchronous world tick

Run root:

`runs/track1_ideal_loopback_matrix_20260729_fast_pipeline_sensor_every_tick_10fps`

Change: set RGB/radar `sensor_tick=0.0`, while keeping the world fixed at
10 FPS and keeping the normal 28-vehicle / 35-pedestrian scene.

| Mode | Sent | Delivery | Actual send FPS | CARLA tick p50 | Camera wait p50 | Radar build p50 | Capture→backbone p50 | Capture→map p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| explicit `sensor_tick=1/fps` | 179 | 100.0% | 7.22 | 71.6 ms | 33.1 ms | 26.4 ms | 45.1 ms | 81.4 ms |
| `sensor_tick=0.0` every tick | 180 | 100.0% | 7.12 | 69.6 ms | 32.6 ms | 27.5 ms | 47.3 ms | 81.9 ms |

Conclusion: `sensor_tick=0.0` does not materially improve the normal-scene
runtime. The remaining overhead is not caused by the explicit sensor tick
setting.

### 2. Remove background actors as a diagnostic-only bound

Run root:

`runs/track1_ideal_loopback_matrix_20260729_fast_pipeline_no_background_10fps`

Change: same pipeline, but `NPC_VEHICLES=0` and `NPC_PEDESTRIANS=0`.

| Scene | Sent | Delivery | Actual send FPS | CARLA tick p50 | Camera wait p50 | Radar build p50 | Capture→backbone p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| normal 28 vehicles / 35 pedestrians | 179 | 100.0% | 7.22 | 71.6 ms | 33.1 ms | 26.4 ms | 45.1 ms |
| no background actors | 119 | 100.0% | 11.84 | 27.0 ms | 19.7 ms | 32.4 ms | 48.5 ms |

The empty-background run is not a reportable project condition because it
removes the traffic/clutter distribution we need. But it is useful diagnostic
evidence: CARLA can exceed 10 FPS when the scene is light, so the normal-scene
ceiling is dominated by simulation/render/sensor production cost from the
realistic traffic setup.

Practical conclusion: there is no low-risk CARLA-side switch that recovers true
10 FPS for the full-density normal scene. Reducing actors, using asynchronous
or stale-frame processing, or changing sensor synchronization could improve
wall-clock FPS, but those changes either alter the scene distribution or add
extra staleness/alignment risk. For Track-1 reporting, keep the normal-scene
setup and report the observed producer-side limitation explicitly.
