# Live sensor-preparation path audit

This audit binds the additive P1--Pn profiler to the qualified direct edge-to-map
cell path. It does not alter the completed 288-cell campaign or any scientific
input.

## Thread and operation ownership

- CARLA RGB callback: `PassiveSplitCollector._on_rgb` timestamps and stores the
  image record, then notifies the sensor condition.
- CARLA semantic callback: `_on_semantic` stores the evaluation label image and
  notifies. Semantic scoring remains on the separate evaluation worker.
- CARLA radar callback: `_on_radar` calls `RadarSweepAggregator.ingest`, then
  stores the measurement and notifies. `ingest` parses the raw float32 record,
  reads the callback-time sensor transform, converts spherical returns to world
  coordinates and retains velocity/provenance. The callback is bounded to this
  necessary copy/transform/store work; it performs no rasterization, model work,
  scoring, or CARLA actor enumeration.
- Callback-to-worker handoff: `on_world_tick` offers every second registered
  tick to `LatestFramePendingSlot`, whose capacity remains exactly one. The
  `route-b-split-front` worker takes the latest token and runs `_process_token`.
- Sensor waiting: `_records_for` waits for the matching RGB/radar/semantic
  records. It is timed independently and is never classified as computation.
- Radar window extraction: the front worker calls
  `RadarSweepAggregator.window_detections` for the unchanged two-sweep/four-
  callback window. It motion-compensates world points into the current radar
  frame and reconstructs the spherical array and raw provenance.
- Radar preparation: the same front worker performs spherical-to-world,
  stationary-track update, world-to-camera conversion, camera projection and
  bounds calculation, the existing fast rasterizer, and evidence packaging in
  that order.
- Camera preparation: the front worker extracts BGR from CARLA BGRA, converts
  BGR to RGB, resizes to 768x448 with linear interpolation, packs the host
  tensor, copies to the selected CUDA device, and applies the frozen ImageNet
  normalization.
- Radar tensor preparation: the front worker resizes channel zero with nearest
  interpolation and the remaining channels with linear interpolation, stacks
  and packs them, copies them to CUDA, then concatenates RGB and radar into the
  seven-channel input.
- Immutable evaluation snapshot: the front worker freezes cached actor state at
  the synchronized CARLA snapshot. Actor discovery refresh and all object/
  segmentation scoring remain on evaluation threads; the profiler adds no
  ground-truth query.

The diagnostic uses `perf_counter_ns` for CPU intervals. CUDA event pairs are
recorded around the four GPU operations and read after one final event
synchronization per prepared frame. That barrier makes the detailed cell
diagnostic-only. A separate production-enqueue wall interval is retained and
does not synchronize after individual CUDA substages.

The live CSV adds existing radar-return count, ego speed, route tick and frame
identity plus safely available cached visible-actor count, acceleration and yaw
rate. It preserves exact NumPy radar tensor/evidence equality and exact PyTorch
model-input equality on the preregistered first eight transmitted frames.
