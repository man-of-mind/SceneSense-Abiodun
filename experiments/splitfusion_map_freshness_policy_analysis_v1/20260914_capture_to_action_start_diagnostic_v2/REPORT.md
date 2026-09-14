# Capture-to-action-start diagnostic

`capture_wall_s` is not a timestamp for a fully prepared seven-channel
tensor. It is host wall time when the raw RGB callback reaches the UE
process. `capture_started_ns` is recorded later, at entry to RGB/radar
tensor assembly immediately before the chosen action's UE dispatch.

Between those boundaries, the worker may wait for the matching radar
record, extract the rolling radar window, transform/rasterize radar,
convert RGB, freeze the evaluation snapshot, and wait for worker access.

## Distribution

Across 896,856 sent frames: median 36.1 ms,
p95 94.6 ms, p99 143.2 ms and maximum
487.4 ms. 75.0%
of frames start action processing within 50 ms of the RGB callback.

## Why the p99 reaches about 143 ms

| Component | All-frame median | All-frame p95 | Slowest-1% median | Slowest-1% p95 |
|---|---:|---:|---:|---:|
| sensor_wait_ms | 9.1 ms | 26.8 ms | 0.0 ms | 29.8 ms |
| radar_window_ms | 5.5 ms | 14.1 ms | 14.0 ms | 73.4 ms |
| radar_prepare_ms | 23.2 ms | 46.8 ms | 93.7 ms | 201.0 ms |
| rgb_convert_ms | 2.9 ms | 7.6 ms | 5.4 ms | 24.5 ms |
| scene_snapshot_ms | 0.1 ms | 0.3 ms | 0.1 ms | 1.4 ms |
| pre_front_compute_ms | 33.3 ms | 64.8 ms | 134.5 ms | 244.8 ms |

The tail is primarily preparation work, especially radar rasterization,
not a claim that CARLA needs 143 ms to generate every frame. In the
slowest 1%, radar preparation rises from a 23.2 ms overall median to
about 93.7 ms, and total pre-front compute rises from 33.3 to 134.5 ms.

The 100 ms line is the nominal source interval, not a per-frame action
deadline. A p99 above it means a small fraction of preparation operations
overlap or miss the next source opportunity; it does not change the
physical capture timestamp used for map AoI.
