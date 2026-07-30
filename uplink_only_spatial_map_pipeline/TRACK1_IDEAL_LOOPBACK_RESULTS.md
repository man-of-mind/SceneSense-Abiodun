# Track 1 ideal-loopback no-return results

Date: 2026-07-29

Run root:
`runs/track1_ideal_loopback_matrix_20260729_155647`

Summary CSV:
`runs/track1_ideal_loopback_matrix_20260729_155647/track1_ideal_loopback_summary.csv`

## Setup

- Track-1 no-return architecture:
  `CARLA/front -> split features -> edge tail -> spatial-map update`
- No full detections are returned to the car.
- Front feature transport uses zstd entropy coding.
- Edge-to-map spatial packets are compact zlib-compressed JSON detections/metadata.
- Ideal loopback UDP buffer was active:
  `net.core.rmem_max/wmem_max=8388608`, granted `SO_RCVBUF=16777216`.
- no-AE baseline, `per_channel_uint8`, ROI 0, 200k radar PPS.
- Corrected drivable route: `80,85,91,94,99,80`.
- 28 requested vehicles, 35 pedestrians, seed 31.
- 120 front frames per condition.
- First 10 delivered frames excluded from latency statistics below.

## Result table

| Condition | Target FPS | Artificial map delay | Delivery | Actual model-send FPS | Map-update FPS | Payload p50 | Core model→map p50 / p95 | Map service p50 | Map UDP queue/ingest p50 | Front→edge p50 | Tail p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ideal_none_fps10_map0` | 10 | 0 ms | 120/120 (100.0%) | 3.5 | 3.5 | 1042.5 KB / 18 chunks | 33.0 / 72.7 ms | 0.0 ms | 8.1 ms | 7.4 ms | 8.3 ms |
| `ideal_none_fps20_map0` | 20 | 0 ms | 120/120 (100.0%) | 5.6 | 5.6 | 1038.6 KB / 18 chunks | 37.6 / 72.4 ms | 0.0 ms | 9.4 ms | 7.7 ms | 10.5 ms |
| `ideal_none_fps10_map40` | 10 | 40 ms | 120/120 (100.0%) | 3.6 | 3.6 | 1045.7 KB / 18 chunks | 74.9 / 136.9 ms | 40.2 ms | 8.9 ms | 7.3 ms | 9.5 ms |
| `ideal_none_fps20_map40` | 20 | 40 ms | 120/120 (100.0%) | 5.7 | 5.7 | 1039.0 KB / 18 chunks | 74.5 / 156.1 ms | 40.3 ms | 9.0 ms | 7.4 ms | 8.2 ms |

## Interpretation

The no-return Track-1 path is clean on ideal loopback:

- no UDP multipart loss;
- no edge receive queue drops;
- edge tail remains small;
- spatial-map baseline service is near zero without artificial map delay;
- adding a 40 ms artificial map delay increases core model-to-map latency by
  about 40 ms, as expected.

Use `backbone_input_to_map_update_done_ms` as the core Track-1 model-path
latency. It starts when the fused RGB+radar model input enters the front
backbone and ends when the spatial-map application finishes the current map
update.

The `map UDP queue/ingest` term is not the final cooperative map reasoning
cost. It currently covers edge-published spatial packet delivery to the local
map server, socket scheduling, zlib decompression, JSON parsing/normalization,
and admission into the map update handler. Future association, occlusion
reasoning, cooperative fusion, and advisory generation should appear in
`map_service_ms`.

## Important caveat from the live CARLA matrix

The live CARLA front did **not** actually offer 10/20 model-input frames per
second in this run. Post-warm-up model-send rate was about:

- target 10 FPS -> actual model-send/map-update rate about 3.5-3.6 FPS;
- target 20 FPS -> actual model-send/map-update rate about 5.6-5.7 FPS.

So this matrix validates the latency of the model/uplink/tail/map path under
the live frontend, but it does **not** yet prove the map server can sustain a
true 10/20 FPS model-output arrival rate. That requires a separate replay or
synthetic model-boundary offered-load test using recorded feature/map packets
at fixed rates.

## Model-boundary offered-load replay

Completed after the live matrix above.

Run root:
`runs/track1_map_offered_load_replay_20260729_161223`

Summary CSV:
`runs/track1_map_offered_load_replay_20260729_161223/track1_map_offered_load_summary.csv`

This replay injects spatial-map packets at exact fixed rates, independent of
CARLA sensor/radar preparation. It uses a synthetic model/uplink/tail timing
base of about 22 ms before the map server, then sweeps artificial map service
delay.

| Condition | Target FPS | Artificial map delay | Delivery | Send rate | Map-update rate | Core model→map p50 / p95 | Map queue p50 / p95 | Map service p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `replay_fps10_map0` | 10 | 0 ms | 180/180 (100.0%) | 10.0 | 10.0 | 22.7 / 73.0 ms | 0.5 / 50.2 ms | 0.0 ms |
| `replay_fps20_map0` | 20 | 0 ms | 180/180 (100.0%) | 20.0 | 20.0 | 22.6 / 29.3 ms | 0.4 / 5.6 ms | 0.0 ms |
| `replay_fps30_map0` | 30 | 0 ms | 180/180 (100.0%) | 30.0 | 30.0 | 22.6 / 23.1 ms | 0.4 / 0.8 ms | 0.0 ms |
| `replay_fps10_map40` | 10 | 40 ms | 180/180 (100.0%) | 10.0 | 10.0 | 63.1 / 126.4 ms | 0.6 / 39.4 ms | 40.2 ms |
| `replay_fps20_map40` | 20 | 40 ms | 180/180 (100.0%) | 20.0 | 20.0 | 62.9 / 135.2 ms | 0.5 / 63.5 ms | 40.1 ms |
| `replay_fps30_map40` | 30 | 40 ms | 180/180 (100.0%) | 30.0 | 23.9 | 740.3 / 1515.4 ms | 678.0 / 1452.9 ms | 40.2 ms |
| `replay_fps10_map60` | 10 | 60 ms | 180/180 (100.0%) | 10.0 | 10.0 | 83.5 / 208.5 ms | 0.7 / 97.4 ms | 60.3 ms |
| `replay_fps20_map60` | 20 | 60 ms | 180/180 (100.0%) | 20.0 | 15.9 | 1123.6 / 2251.8 ms | 1013.7 / 2169.6 ms | 60.1 ms |
| `replay_fps30_map60` | 30 | 60 ms | 170/180 (94.4%) | 30.0 | 16.2 | 2539.4 / 4659.5 ms | 2441.3 / 4577.2 ms | 60.1 ms |

### Offered-load interpretation

- With no map compute, the map ingest/update path sustains 30 FPS cleanly.
- With 40 ms map compute, 10 and 20 FPS are still stable, but 30 FPS overloads
  the single-threaded map service path. The queue delay, not UDP transport, is
  the dominant source of staleness.
- With 60 ms map compute, 10 FPS is stable, while 20 and 30 FPS overload the
  map service path. This is exactly the service-rate limit:
  60 ms per frame is only about 16.7 frames/s capacity.
- Therefore, for a future single-threaded spatial-map stage, 40 ms/frame is
  acceptable at 10/20 FPS but not at 30 FPS. A 60 ms/frame map update needs
  either lower FPS, parallelism, frame dropping, or a lighter map update.

## Fine-grained live frontend profile

Run root:
`runs/live_front_prep_profile_50f`

Summary:
`runs/live_front_prep_profile_50f/live_front_prep_profile_summary.txt`

This run used the corrected live CARLA no-return path, ideal loopback buffers,
no-AE baseline, zstd feature transport, 200k radar PPS, 28 vehicles, 35
pedestrians, and the corrected drivable route. It processed 50/50 frames with
zero UDP partial drops. The first 10 frames are excluded from the statistics
below.

| Stage | p50 | p95 | Meaning |
|---|---:|---:|---|
| Actual model-send rate | 4.0 FPS | - | Live frontend cadence after warm-up |
| `sync_world_tick_ms` | 32.5 ms | 95.5 ms | CARLA synchronous tick cost |
| `camera_frame_wait_ms` | 33.6 ms | 47.6 ms | Wait for the current camera frame |
| `radar_wait_ms` | 0.0 ms | 0.0 ms | Radar packet wait; not the bottleneck |
| `radar_tensor_build_ms` | 139.3 ms | 180.3 ms | Build/project/rasterize radar tensor |
| `model_preprocess_ms` | 11.3 ms | 18.3 ms | Convert prepared image/radar tensors into model inputs |
| `capture_to_backbone_input_ms` | 152.9 ms | 198.4 ms | Optional full sensor/prep time before model begins |
| `front_backbone_ms` | 3.2 ms | 6.1 ms | Front/backbone split encoder |
| `feature_serialize_ms` | 0.7 ms | 1.5 ms | Feature serialization/compression |
| `send_call_ms` | 4.8 ms | 7.0 ms | Sender UDP chunking/send call |
| `front_to_edge_ms` | 6.7 ms | 9.3 ms | Ideal-loopback feature arrival at edge |
| `tail_ms` | 7.8 ms | 11.3 ms | Edge tail inference |
| `map_queue_ms` | 8.2 ms | 18.4 ms | Map UDP ingest/queue before update |
| `backbone_input_to_map_update_done_ms` | 28.5 ms | 44.6 ms | Core split model→map latency |
| `capture_to_map_update_done_ms` | 180.7 ms | 247.5 ms | Full sensor capture→map update age |

### Live profile interpretation

The split-inference/map path is not the live-loop bottleneck on ideal loopback:
from model input to completed map update is about 29 ms p50 / 45 ms p95.

The large optional `capture_to_map_update_done_ms` is dominated by frontend
sensor/radar preparation, especially `radar_tensor_build_ms` at about
139 ms p50 / 180 ms p95. The radar wait itself is essentially zero, so this is
compute/conversion work, not a missing-radar synchronization wait.

For the current Track-1 architecture, keep reporting both:

- `backbone_input_to_map_update_done_ms` for the split model/uplink/tail/map
  path; and
- `capture_to_map_update_done_ms` when discussing total information freshness
  from CARLA sensor capture to map update.

## Next actions

1. Use `--radar-rasterizer fast` for the next Track-1 loopback/OAI profiling
   runs; same-frame shadow validation now supports that choice.
2. Keep the replay/offered-load test as the clean way to budget future spatial
   map compute at exact 10/20/30 FPS.
3. Repeat Track-1 uplink-only over
   OAI with T-tracer enabled to see whether the radio-side MCS/RLC behavior
   changes when the car no longer waits for full detection results.

## Fast-rasterizer throughput rerun

Dedicated note:
`RADAR_RASTERIZER_OPTIMIZATION.md`

Run root:
`runs/track1_ideal_loopback_matrix_20260729_fast_throughput`

Summary CSV:
`runs/track1_ideal_loopback_matrix_20260729_fast_throughput/track1_fast_throughput_summary.csv`

Fresh ideal-loopback run after the same-frame validation, using
`RADAR_RASTERIZER=fast`, no artificial map delay, 180 frames per target FPS, and
the same no-AE/zstd/200k-PPS drivable-route recipe.

| Target FPS | Delivered | Actual model-send FPS | Map-update FPS | Radar points/frame p50 | Radar build p50 / p95 | Core model→map p50 / p95 | Capture→map p50 / p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 180/180 | 6.38 | 6.40 | 19,896.5 | 27.1 / 56.0 ms | 31.2 / 83.3 ms | 82.3 / 152.5 ms |
| 20 | 180/180 | 8.19 | 8.19 | 9,951.0 | 14.9 / 31.1 ms | 32.9 / 78.4 ms | 68.4 / 115.1 ms |
| 30 | 180/180 | 9.40 | 9.38 | 6,624.0 | 12.5 / 24.5 ms | 33.8 / 74.0 ms | 65.3 / 105.7 ms |

Interpretation: the optimized ideal-loopback uplink-only path is stable
(`180/180`, zero UDP partial drops, zero edge queue drops). The current live
frontend ceiling is roughly `9-10 FPS` in this setup, but the 20/30 FPS rows use
fewer radar points per frame because CARLA fixes radar density in
points/second. The most comparable row to the original 10 FPS / 200k-PPS
setting is therefore the 10 FPS row: about `6.4 FPS` with ~20k radar
points/frame.

## Radar tensor optimization note

An opt-in vectorized radar rasterizer was added after the live profile:
`--radar-rasterizer fast`.

The default remains `legacy`, so existing runs are unchanged unless the flag is
explicitly set. The fast path preserves the same projected square-patch tensor
recipe by replacing the Python per-point patch loop with vectorized scatter plus
max-filter dilation.

Synthetic 20k-point, 768x432, radius-4 benchmark:

| Test | Legacy | Fast | Difference |
|---|---:|---:|---:|
| Raster-only median | 102.6 ms | 7.6 ms | about 13.5x faster |
| Full `build_radar_sample` median | 161.8 ms | 34.4 ms | about 4.7x faster |
| Tensor max abs diff | - | - | `5.96e-08` |

Live 50-frame ideal-loopback validation run:
`runs/live_front_prep_profile_fast_50f`

Comparison against the earlier legacy profile, excluding the first 10 frames:

| Metric | Legacy p50 / p95 | Fast p50 / p95 | Change |
|---|---:|---:|---:|
| Actual model-send rate | 3.98 FPS | 7.03 FPS | 1.77x |
| `radar_tensor_build_ms` | 139.3 / 180.3 ms | 32.6 / 52.8 ms | -106.6 ms |
| `capture_to_backbone_input_ms` | 152.9 / 198.4 ms | 53.6 / 82.1 ms | -99.3 ms |
| `backbone_input_to_map_update_done_ms` | 28.5 / 44.6 ms | 37.7 / 80.2 ms | +9.2 ms |
| `capture_to_map_update_done_ms` | 180.7 / 247.5 ms | 93.3 / 136.1 ms | -87.4 ms |
| `front_to_edge_ms` | 6.7 / 9.3 ms | 7.8 / 13.4 ms | +1.1 ms |
| `tail_ms` | 7.8 / 11.3 ms | 10.2 / 21.4 ms | +2.5 ms |

Both runs processed 50/50 frames with zero UDP partial drops and zero edge
receive queue drops.

As a light sanity check, edge-side object counts did not collapse:
legacy post-warm-up median was 4 objects/frame and fast post-warm-up median was
also 4 objects/frame, with the same 2-7 object/frame range. This is not a
localization-accuracy proof, but it rules out an obvious empty-output failure.

The live result confirms the radar rasterizer was the dominant frontend
bottleneck. The fast path roughly halves full capture-to-map age and raises the
actual model-ready send rate from about 4 FPS to about 7 FPS. It does not yet
reach the requested 10 FPS; the remaining ceiling is now a mix of CARLA
synchronous tick/camera wait, model preprocessing, front compute/serialization,
and normal end-to-end scheduling overhead.

Important accuracy caveat: the no-return profile does not log object prediction
and ground-truth rows, so this run validates latency/reliability, not final
localization accuracy. The same-frame shadow validation below is the correct
evidence for fast-vs-legacy radar rasterizer equivalence.

## Fast rasterizer accuracy A/B

Dedicated note:
`FAST_RASTERIZER_ACCURACY_AB.md`

Run root:
`runs/accuracy_ab_fast_vs_legacy_20260729`

This follow-up used closed-loop/full-result loopback so object predictions and
vehicle ground truth were logged on the front side. Both conditions used the
same route, seed, model, 200k radar PPS, radius 4, and temporal window 2. The
first 10 of 80 frames were excluded.

Evaluation matched predicted vehicles to visible GT vehicles within 40 m using
nearest-neighbor XY distance with a 5 m maximum match radius.

| Metric | Legacy | Fast | Delta |
|---|---:|---:|---:|
| Eval frames | 70 | 70 | +0 |
| Visible GT vehicles <=40 m | 228 | 228 | +0 |
| Predicted vehicles <=40 m | 174 | 175 | +1 |
| Matched vehicles | 60 | 56 | -4 |
| Precision @5 m | 34.5% | 32.0% | -2.5 pp |
| Recall @5 m | 26.3% | 24.6% | -1.8 pp |
| Loc error mean | 3.016 m | 3.035 m | +0.018 m |
| Loc error p50 | 3.113 m | 3.095 m | -0.018 m |
| Loc error p90 | 4.126 m | 3.999 m | -0.127 m |
| Loc error p95 | 4.456 m | 4.405 m | -0.051 m |

Interpretation: this independent-run A/B did not show a fast-rasterizer-specific
localization shift. The matched localization-error distribution is essentially
unchanged. However, the ~3 m loose-matcher values in this table should not be
reported as the model localization floor or compared directly with the offline
knob-matrix metric. The small match-count difference should be treated
cautiously because these are independent live CARLA runs rather than a
same-frame shadow comparison.

## Same-frame radar rasterizer shadow validation

Dedicated note:
`RADAR_RASTERIZER_SHADOW_VALIDATION.md`

Run root:
`runs/radar_rasterizer_shadow_20260729_30f`

This is the stronger validation for the fast rasterizer. It builds both legacy
and fast radar tensors from the exact same live CARLA radar measurement in the
same frame, then runs both tensors through the model and compares the decoded
objects.

| Metric | Result |
|---|---:|
| Frames compared | 30 |
| Tensor max abs diff | `5.96e-08` |
| Tensor entries differing > `1e-6` | 0 |
| Occupancy changed pixels | 0 |
| Object-count delta | 0 on all frames |
| Legacy unmatched objects | 0 on all frames |
| Fast unmatched objects | 0 on all frames |
| Matched center-pixel max distance | 0 px |
| Matched world-XY max distance | 0.0102 m |
| Matched score max abs diff | 0.00269 |

Conclusion: the fast rasterizer is safe to use for Track-1 profiling. It gives
the model effectively the same radar tensor and produces the same decoded object
decisions on the same live frames. Keep the offline knob-matrix result as the
model-accuracy anchor for no-AE u8, about 0.95 m localization error; keep this
shadow result as the rasterizer-equivalence evidence.

## Capture-pipeline diagnostic

Dedicated note:
`CAPTURE_PIPELINE_RESULTS.md`

The uplink-only client now has an opt-in bounded producer/consumer mode:
`--capture-pipeline`. The default sequential path is unchanged.

Fresh ideal-loopback fast-rasterizer A/B:

| Target FPS | Mode | Sent to map | Actual send FPS | Queue wait p50 | Queue depth p95 | Capture→backbone p50 | Backbone→map p50 | Capture→map p50 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 10 | sequential | 180/180 | 6.37 | - | - | 50.4 ms | 31.4 ms | 83.0 ms |
| 10 | capture pipeline q2 | 179/179 | 7.22 | 0.1 ms | 0 | 45.1 ms | 33.7 ms | 81.4 ms |
| 20 | sequential | 180/180 | 8.14 | - | - | 33.4 ms | 33.4 ms | 69.3 ms |
| 20 | capture pipeline q2 | 179/179 | 9.32 | 0.1 ms | 0 | 33.2 ms | 35.0 ms | 70.9 ms |
| 30 | sequential | 180/180 | 9.36 | - | - | 30.0 ms | 33.9 ms | 65.3 ms |
| 30 | capture pipeline q2 | 179/179 | 10.89 | 0.1 ms | 0 | 27.9 ms | 38.3 ms | 67.5 ms |

Interpretation: the capture pipeline improves live cadence, but the prepared
frame queue is almost always empty. This means the downstream encode/send/tail
map path is not the limiter on ideal loopback. The remaining live frontend
ceiling is still dominated by producer-side CARLA tick/camera acquisition plus
front-side sensor preparation.

CARLA-side follow-up: setting RGB/radar `sensor_tick=0.0` every synchronous
tick did not improve the normal-scene 10 FPS run (`7.12 FPS` vs `7.22 FPS`).
A diagnostic no-background run reached `11.84 FPS`, with CARLA tick p50 falling
from `71.6 ms` to `27.0 ms` and camera wait p50 from `33.1 ms` to `19.7 ms`.
This confirms the remaining full-scene ceiling is mainly CARLA
simulation/render/sensor-production cost under realistic traffic, not a simple
sensor tick configuration issue.
