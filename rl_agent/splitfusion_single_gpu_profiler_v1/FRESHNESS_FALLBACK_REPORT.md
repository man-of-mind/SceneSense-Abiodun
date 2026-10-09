# Scene/network state freshness: fallback accounting and the smallest evidenced fix

Scope: retained evidence only (the frozen 1,000-frame capture `20261001T231343Z`,
then thread-cap arms A `20261002T024039Z` and B `20261002T024744Z`). Nothing was
re-run, no limit was changed, no observation was re-stamped, no frame's
descriptors were substituted. Script: `freshness_fallback_analysis_v1.py`;
outputs: `rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261002T025556Z_freshness_fallback_analysis/`.
Ages are on `CLOCK_MONOTONIC_RAW`; the camera-SI age is `state_commit − rgb_receipt`
against the unchanged 100-ms rule.

## 1. Every fallback, by reason

| Capture | Fallbacks | Scene-stale (camera_si) | MCS-stale | Startup (clock bridge warming) |
|---|---|---|---|---|
| frozen | 122 | 107 | 13 | 2 |
| A (baseline) | 105 | 91 | 12 | 2 |
| B (thread cap) | 16 | 11 | 3 | 2 |

No radar-P40 stale, backlog-stale or steady-state "missing" fallbacks occurred.

## 2. Scene-stale decomposition (RGB callback → state commit)

Descriptors belong to the exact inputs encoded: for every planned frame the
`rgb_receipt_raw_ns` the planner used equals the collector's own RGB-callback
trace entry for that CARLA frame (985/985, 986/986, 994/994) and lies within 2 ms
of the profiler's independent RGB-callback mark (984, 986, 994); the radar-window
digest bound into the plan is unique per sent frame in all three captures; the
planner refuses a plan unless the rasterized window digest equals the P40 window
digest, and materialization refuses any frame/capture/CARLA-timestamp/window drift.

Age at each milestone, P50/P95 ms (frozen capture; A is within a few ms of it):

| Milestone | Stale frames (n=107) | Admitted decisions (n=420) |
|---|---|---|
| token scheduled at world tick | −12.0 / 10.2 | −12.8 / 1.8 |
| **worker start** | **56.9 / 88.0** | −2.5 / 48.6 |
| sensor ready | 57.2 / 93.1 | 1.6 / 49.2 |
| radar window end | 71.9 / 119.5 | 11.1 / 56.6 |
| **rasterization end** | **116.4 / 211.8** | 40.2 / 84.1 |
| RGB convert end | 120.2 / 215.5 | 44.4 / 88.5 |
| SI/P40 start | 120.4 / 215.8 | 44.8 / 88.7 |
| SI/P40 end | 123.7 / 223.2 | 48.7 / 92.5 |
| state commit | 124.4 / 224.0 | 49.0 / 93.2 |

Segments, P50/P95 ms, stale vs admitted (frozen):

| Segment | Kind | Stale | Admitted |
|---|---|---|---|
| callback → worker start | **waiting** (worker busy with the previous frame) | 56.9 / 88.0 | −2.5 / 48.6 |
| worker start → sensors ready | waiting | 0.0 / 4.8 | 4.0 / 24.9 |
| radar window (under the sensor lock) | computation | 11.5 / 51.7 | 6.2 / 19.1 |
| radar rasterization (`build_radar_sample`, CPU) | computation | **44.0 / 118.7** | 24.6 / 50.2 |
| RGB convert | computation | 2.8 / 8.3 | 2.8 / 6.6 |
| scene snapshot (CARLA RPC) | synchronization | 0.1 / 0.3 | 0.1 / 0.3 |
| SI/P40 | computation | 2.2 / 26.6 | 2.2 / 9.1 |
| engine-lock wait | synchronization | 0.0 / 0.0 | 0.0 / 0.0 |
| telemetry snapshot + state build | computation | 0.3 / 0.7 | 0.3 / 0.6 |

What this says:

- The token is scheduled at the world tick ~12 ms **before** the image arrives,
  identically for stale and admitted frames. Stale frames differ at the next
  step: the single preparation worker picked them up ~57 ms (P50) after the
  callback because it was still finishing the previous frame; 62/107 waited
  ≥ 50 ms. Sensor waiting is not the cause (≈0 ms for stale frames).
- On those late frames the radar window and rasterization also ran slower
  (11.5 and 44 ms P50 vs 6.2 and 24.6 ms), and the age crossed 100 ms at
  rasterization end for 66/107 (A: 60/91), at radar-window end for 14, at RGB
  convert for 11, during SI/P40 for 11. 96/107 (A: 80/91) were over age before
  SI/P40 began; only 2/107 (A: 1/91) were already over age at worker start.
- Synchronization is negligible in the frozen capture and in A (lock wait 0,
  snapshot 0.1 ms). In B the 11 remaining stale frames are sporadic stalls:
  rasterization P95 236 ms, scene-snapshot RPC P95 89 ms, SI/P40 P95 44 ms; the
  worker-start wait is still present (46.7 ms P50) but admitted frames in B wait
  −10 ms P50 / 5 ms P95, i.e. the worker keeps up once the thread cap is on.
- Excess over the rule: P50 24 ms, P95 124 ms (frozen); 18.5 / 61.7 ms (A);
  17.2 / 258 ms (B).

## 3. MCS-stale decomposition (information available before the decision only)

The provider's reported age was re-derived from the retained T-tracer DCI
stream (event time of day → host clock; rows ordered by reader receipt). It
agrees with the provider within 5 ms in all 28 cases (13/13, 12/12, 3/3): no
selection or timestamp error, and no reader-to-publication delay (publication is
synchronous on accept). Causes:

| Cause | frozen | A | B |
|---|---|---|---|
| no DCI row at all with an event inside the 100-ms window | 6 | 7 | 1 |
| only ineligible rows in the window (retransmission rounds / other direction) | 6 | 4 | 2 |
| eligible grant inside the window, but it reached the reader after the boundary | 1 (7 late events) | 1 (16 late events) | 0 |

Source-to-reader delay is normally P50 1.0 ms, P95 6–8 ms, with bursts up to
57–90 ms (the two "arrived late" cases coincide with ~40 ms receipt gaps).
MCS staleness is therefore mostly the radio not issuing a new-data uplink
grant within the 100 ms preceding the decision — the UE's uplink is bursty
(one tensor per ~100-ms period) — not a pipeline defect. It cannot be reduced
without changing the rule or the eligibility, which this analysis does not do.

## 4. Startup readiness (separate from steady state)

Two frames per capture fall back with `CLOCK_BRIDGE_INVALID_OR_WARMING` plus
missing MCS/backlog. The causal clock bridge needs PDCP anchors, which exist
only once uplink traffic flows: first PDCP row 267–307 ms after the first route
tick; first admitted decision 441–577 ms after it. Structural, not a failure.

## 5. Smallest remaining evidenced fix (to be measured separately)

**Wire the already-validated, bit-exact radar-chain optimization into this
Phase-6 path.** The CUDA `scatter_reduce(amax)` + `max_pool2d` rasterizer,
the single-sort stationary-track accumulator and the identity-resize short
circuit (`rl_agent/splitfusion_sensor_optimization_v1/optimized_stages.py`,
`SENSOR_PREPARATION_OPTIMIZATION_VALIDATED`, 2026-09-14) are installed only in
the Phase-15 sensor-preparation cells (`splitfusion_supervisor_analysis_v1/
profiled_direct_adapter_v1.py:327-389`, via `rasterizer_override`). The Phase-6
live path still calls `build_radar_sample(..., rasterizer="fast")` on the CPU
(`ue_route_b_split_cell_adapter_v1.py:1569`), and that stage is the largest
computation in stale frames (44 ms P50 / 119 ms P95 frozen; 40 / 98 in A;
25 / 236 in B) and the first milestone over the rule in 66/107 (A 60/91).

- Mechanism: the same process-local seam pattern used for the profiler (an
  instance-level `build_radar_sample` override with the CUDA rasterizer and
  single-sort tracker), with the validated bit-exactness tests run first.
- Validated gains to expect: rasterization 7.03 → 1.02 ms; radar chain P50
  31.7 → 21.5 ms, P95 63.6 → 45.6 ms; tensors bit-exact.
- Expected benefit, bounded from the retained stale rows: had the pre-commit
  path been 10 ms faster, 22/107 (frozen), 30/91 (A) and 4/11 (B) scene-stale
  frames would have met the rule; 14 ms faster: 30 / 38 / 4; 20 ms faster:
  37 / 46 / 6. Stale frames' rasterization is ~2× the admitted median, so tail
  gains help more than the median suggests; a shorter worker cycle also shrinks
  the dominant waiting term (the next frame's queue wait), which is not
  quantified here.
- Caveats: it adds GPU work inside the UE process on the shared GPU (the
  validated report attributed ~10 ms of the chain to contention and saw P99
  regress in a pre-registered window); after the thread cap only 11 stale
  frames remain in B, so the measurable headroom is small; it must get its own
  capture with the same gates.

Not recommended as a "fix": computing SI/P40 and committing the state before
rasterization would have rescued 86/107 (A 78/91) frames, but it moves
rasterization inside the 170-ms action-open clock and changes the registered
addendum-6 boundary (the planner requires rasterization complete before the
plan). That is a design trade-off for Abiodun, reported, not taken.
