# Radar rasterizer optimization and Track-1 throughput

Date: 2026-07-29

## Summary

The live Track-1 frontend bottleneck was the radar tensor construction path, not
the ideal-loopback uplink or the edge/map path. The legacy Python point-paint
rasterizer cost about `139 ms` p50 in the 50-frame live profile. The validated
fast rasterizer reduces that to about `33 ms` p50 in the same live recipe, and
to `27 ms` p50 in the longer 10 FPS throughput sweep.

The fast rasterizer is opt-in:

```bash
--radar-rasterizer fast
```

The default remains `legacy` so earlier commands remain reproducible.

## Implementation

Shared implementation:

- `../pole_lraspp_multimodal_fusion/pole_lraspp_multimodal_fusion/radar_fusion.py`
  - added `rasterize_radar_channels_fast(...)`;
  - added `build_radar_sample(..., rasterizer="legacy"|"fast")`.

The fast path replaces the per-radar-point Python patch loop with vectorized
scatter plus max-filter dilation. It preserves the same square support and
border-padding behavior as the historical rasterizer.

Opt-in flag added to:

- `carla_fusion_staleness_scenario_uplink_only.py`
- `../staleness/carla_fusion_staleness_scenario.py`
- `../carla_split_inference_udp_fusion_object_pole_client_spatial_stream.py`
- `../carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai.py`
- `carla_split_inference_udp_fusion_object_uplink_only_spatial.py`

The wrapper `run_track1_ideal_loopback_matrix.sh` now accepts:

```bash
RADAR_RASTERIZER=fast
FPS_LIST="10 20 30"
MAP_DELAYS_MS="0"
MAX_FRAMES=180
```

## Validation

Dedicated same-frame validation:

- `RADAR_RASTERIZER_SHADOW_VALIDATION.md`
- run root: `runs/radar_rasterizer_shadow_20260729_30f`

Key result:

| Metric | Result |
|---|---:|
| Frames compared | 30 |
| Tensor max abs diff | `5.96e-08` |
| Tensor entries differing > `1e-6` | 0 |
| Occupancy changed pixels | 0 |
| Object-count delta | 0 on all frames |
| Unmatched decoded objects | 0 on all frames |
| Matched center-pixel max distance | 0 px |
| Matched world-XY max distance | 0.0102 m |

Conclusion: the model receives effectively the same radar tensor and produces
the same decoded object decisions on the same live frames. Use the offline
knob-matrix result as the model-accuracy anchor; use the same-frame shadow run
as the rasterizer-equivalence evidence.

## Speedup evidence

Synthetic 20k-point, 768x432, radius-4 benchmark:

| Test | Legacy | Fast | Difference |
|---|---:|---:|---:|
| Raster-only median | 102.6 ms | 7.6 ms | about 13.5x faster |
| Full `build_radar_sample` median | 161.8 ms | 34.4 ms | about 4.7x faster |
| Tensor max abs diff | - | - | `5.96e-08` |

Live 50-frame ideal-loopback profile, excluding first 10 frames:

| Metric | Legacy | Fast | Change |
|---|---:|---:|---:|
| Actual model-send rate | 3.98 FPS | 7.03 FPS | 1.77x |
| `radar_tensor_build_ms` p50 | 139.3 ms | 32.6 ms | -106.6 ms |
| `capture_to_backbone_input_ms` p50 | 152.9 ms | 53.6 ms | -99.3 ms |
| `capture_to_map_update_done_ms` p50 | 180.7 ms | 93.3 ms | -87.4 ms |

## Fresh Track-1 throughput sweep

Run root:

`runs/track1_ideal_loopback_matrix_20260729_fast_throughput`

Summary CSV:

`runs/track1_ideal_loopback_matrix_20260729_fast_throughput/track1_fast_throughput_summary.csv`

Command:

```bash
env RADAR_RASTERIZER=fast \
  FPS_LIST="10 20 30" \
  MAP_DELAYS_MS="0" \
  MAX_FRAMES=180 \
  STAMP=20260729_fast_throughput \
  bash abiodun/uplink_only_spatial_map_pipeline/run_track1_ideal_loopback_matrix.sh
```

All rows used:

- ideal loopback UDP buffers;
- no-AE baseline;
- zstd feature transport;
- `per_channel_uint8`, ROI 0;
- 200k radar PPS;
- radius 4;
- temporal radar window 2;
- moving ego on route `80,85,91,94,99,80`;
- 28 requested vehicles and 35 pedestrians;
- no artificial map delay.

First 10 frames are excluded from latency/rate statistics.

| Target FPS | Delivered | Actual model-send FPS | Map-update FPS | Radar points/frame p50 | Feature payload p50 | Radar build p50 / p95 | Core model→map p50 / p95 | Capture→map p50 / p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 180/180 | 6.38 | 6.40 | 19,896.5 | 1041.7 KB / 18 chunks | 27.1 / 56.0 ms | 31.2 / 83.3 ms | 82.3 / 152.5 ms |
| 20 | 180/180 | 8.19 | 8.19 | 9,951.0 | 1039.3 KB / 18 chunks | 14.9 / 31.1 ms | 32.9 / 78.4 ms | 68.4 / 115.1 ms |
| 30 | 180/180 | 9.40 | 9.38 | 6,624.0 | 1037.9 KB / 18 chunks | 12.5 / 24.5 ms | 33.8 / 74.0 ms | 65.3 / 105.7 ms |

Plots:

- `plots/track1_fast_rasterizer_actual_fps.pdf`
- `plots/track1_fast_rasterizer_latency_breakdown_p50.pdf`

## Closed-loop-style model-path latency breakdown

For comparison with the earlier closed-loop latency tables, use
`t_backbone_input_perf` as the start of the model path. This excludes optional
CARLA sensor/camera/radar preparation and measures:

```text
backbone input
  -> front/backbone compute
  -> feature serialization/compression
  -> UDP send/uplink/reassembly at edge
  -> edge queue
  -> edge tail inference
```

The closest uplink-only equivalent to closed-loop `capture -> result` without
downlink is:

```text
backbone_input_to_tail_done_ms
```

Fresh fast-rasterizer ideal-loopback results, excluding first 10 frames:

| Target FPS | Front backbone p50 / p95 | Feature serialize p50 / p95 | Send call p50 / p95 | Front→edge p50 / p95 | Edge queue p50 / p95 | Edge tail p50 / p95 | Backbone→edge recv p50 / p95 | Backbone→tail done p50 / p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 2.1 / 9.9 ms | 0.9 / 4.6 ms | 4.8 / 10.3 ms | 6.6 / 12.1 ms | 0.04 / 0.10 ms | 7.7 / 15.0 ms | 11.9 / 22.5 ms | 21.7 / 32.2 ms |
| 20 | 3.1 / 8.0 ms | 0.8 / 2.7 ms | 5.0 / 9.0 ms | 7.1 / 11.4 ms | 0.06 / 0.25 ms | 8.4 / 26.5 ms | 11.7 / 20.5 ms | 22.4 / 41.9 ms |
| 30 | 4.3 / 8.5 ms | 0.8 / 2.7 ms | 5.5 / 11.2 ms | 7.2 / 13.5 ms | 0.05 / 0.23 ms | 8.4 / 24.0 ms | 12.7 / 23.2 ms | 22.1 / 41.2 ms |

Do not add `send_call_ms` and `front_to_edge_ms` when computing a cumulative
total. `front_to_edge_ms` starts immediately before the UDP send call, so it
already includes the send/chunking call plus local loopback delivery and
edge-side reassembly. `send_call_ms` is useful as a diagnostic subcomponent.

If the spatial-map application is included, the next cumulative milestone is
`backbone_input_to_map_update_done_ms`: about `31-34 ms` p50 in this no-delay
map run.

## Interpretation

The uplink-only ideal-loopback path is stable after the optimization:

- all three runs delivered `180/180` frames to the map;
- edge receive queue drops: `0`;
- UDP partial-message drops: `0`;
- the core model-to-map path remains around `31-34 ms` p50.

The live frontend still does not reach the requested target rate in wall-clock
time. With this current synchronous CARLA setup, the measured throughput rises
from `6.38 FPS` at target 10 to `9.40 FPS` at target 30.

Important nuance: CARLA radar uses fixed `points_per_second`, so increasing
requested FPS reduces radar points per frame:

```text
radar points/frame ~= radar_points_per_second / fps
```

With `200k` radar PPS:

- 10 FPS gives about `20k` radar points/frame;
- 20 FPS gives about `10k` radar points/frame;
- 30 FPS gives about `6.7k` radar points/frame.

So the most apples-to-apples row for the earlier 10 FPS / 200k-PPS recipe is
the 10 FPS row: it now reaches about `6.4 FPS` with full ~20k radar points per
frame. The higher requested-FPS rows show the practical wall-clock ceiling when
the per-frame radar point count is lower.

## Current takeaway

The radar rasterizer optimization is real and validated. It moves the Track-1
live frontend from roughly `4 FPS` to `6-9 FPS`, depending on requested sensor
FPS and resulting radar points per frame. The remaining gap to a strict 10 FPS
full-density live run is no longer dominated by the old rasterizer loop; it is
now shared across CARLA synchronous tick/camera wait, model preprocessing,
front compute/send scheduling, and normal host scheduling overhead.

## Camera-resolution diagnostic

Question: since the model input is `768x432`, can we reduce CARLA camera render
from `1280x720` to direct `768x432` and recover the missing FPS?

Result: not in this run. Direct `768x432` capture did not reduce camera wait or
improve throughput.

No-return Track-1 ideal-loopback A/B, target 10 FPS, fast rasterizer:

| Camera capture | Model input | Delivered | Actual send FPS | Send period p50 / p95 | CARLA tick p50 | Camera wait p50 | Capture→backbone p50 | Capture→map p50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1280x720 | 768x432 | 180/180 | 6.38 | 154.5 / 203.3 ms | 55.8 ms | 31.6 ms | 49.4 ms | 82.3 ms |
| 768x432 | 768x432 | 180/180 | 6.25 | 157.0 / 213.0 ms | 59.9 ms | 31.2 ms | 47.9 ms | 78.7 ms |

Summary CSV:

`runs/track1_camera_resolution_ab_20260729_summary.csv`

Closed-loop/full-result sanity check for direct `768x432` capture:

`runs/camera_resolution_closedloop_768_20260729/camera_resolution_closedloop_ab_summary.json`

Compared with the earlier fast `1280x720 -> 768x432` closed-loop run:

| Camera capture | Eval frames | Vehicle preds <=40m | Matches | Precision | Recall | Loc mean / p50 / p95 | Object count p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1280x720 | 70 | 175 | 56 | 0.320 | 0.246 | 3.03 / 3.10 / 4.40 m | 4 |
| 768x432 | 70 | 163 | 52 | 0.319 | 0.228 | 2.48 / 1.73 / 4.84 m | 4 |

This is only a live-run sanity check using the same loose matcher as
`FAST_RASTERIZER_ACCURACY_AB.md`; it is not an offline accuracy benchmark. The
direct 768 run does not show an obvious output collapse, but it also does not
provide a throughput win. Keep `1280x720 -> 768x432` as the reportable default
unless a later same-frame/two-camera validation shows direct `768x432` is both
accuracy-safe and useful.
