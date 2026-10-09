# Radar-preparation bundle on the Phase-6 live path: control vs treatment

Registration `experiment_radar_bundle_v1.json` (sealed before any arm); code
74d1e95; both arms retain the UE-only `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.
Result record with all evidence hashes: `RADAR_BUNDLE_CT_RESULT.json`.

## Exact intervention (one bundle, reported as one)

Reused unchanged from the validated optimization (`splitfusion_sensor_optimization_v1/optimized_stages.py`,
`splitfusion_supervisor_analysis_v1/profiled_sensor_stages.py`):

1. `CudaRadarRasterizer` as `rasterizer_override` of the validated
   `build_radar_sample_profiled` (the production builder has no override; its
   `profile_*` summary keys are stripped so the returned summary keeps exactly
   the production keys);
2. `SingleSortStationaryTrackAccumulator` with the production parameters,
   swapped before the tracker holds any state;
3. the identity radar-resize short circuit in an otherwise production-identical
   packing function (`prepare_live_input_profiled` is not used live: its CUDA
   events are a synchronization barrier its own docstring confines to
   diagnostics).

Attached in the UE child only, around the profiler's seams, so the Phase-6
`radar_tensor_start/end` stamps and the `ue.radar_rasterize` / `ue.input_7ch`
spans still bracket it; one recorded synthetic warm call at install (6.9 ms
live). Nothing else changed.

## Offline parity and cost (`…/20261002T032618Z_radar_bundle_offline_parity/`)

Actual Phase-6 proxy composition (`_PreparationTimer` + profiler span proxy +
bundle), synthetic 28-step episode at 0 / 12 / 19,404 / 40,000 returns:
tensor, evidence points, summary and tracker state exact 28/28; packing
`torch.equal` 28/28; SI/P40 and window digests invariant; real preloaded UE
front/ranker/AE/codec give identical inner payload bytes on three profiles;
existing bit-exact and codec-identity suites 32/32. Idle-host cost, warmed
equally, paths alternated, including transfers and the device sync:
build+packing 8.26 → 4.78 ms P50 at 19,404 returns, 12.6 → 6.9 ms at 40,000 —
a 3–6 ms compute gain; it was not treated as a prediction of the live result.

## Captures

| Arm | Run | Gates | G1 | E1–E4 | Export `MANIFEST.json` |
|---|---|---|---|---|---|
| C control (cap, production radar path) | `20261002T032923Z` | 9/9 `CAPTURE_VALID` | 999, decision-cycle boundary | pass | `1e1fe4b8…` |
| T treatment (cap, bundle) | `20261002T033601Z` | 9/9 `CAPTURE_VALID` | 1,000 at budget | pass | `6c76a3e7…` |

No protocol deviation: both
arms launched from a cold host after the same settle procedure, same warm-up,
route, seed, trace and safety bounds; no retry, no setup correction.

## Before/after (C → T; received-feedback percentiles; profiler cost included)

| Metric | C | T | T − C |
|---|---|---|---|
| Policy decisions / holds / fallbacks | 475 / 503 / 21 | 463 / 518 / 19 | −12 / +15 / −2 |
| **Timely feedback ≤ 170 ms / all decisions** | 339/475 (**71.4%**) | 323/463 (**69.8%**) | −1.6 pt |
| **Action-open → feedback P50 / P95 / P99 / max** | 155.9 / 220.9 / 290.3 / 345.8 | 156.1 / 215.7 / 297.8 / 675.6 | +0.1 / −5.2 / +7.5 / +330 |
| Missing replies (decisions without feedback) | 7 | 20 | +13 |
| SUCCESS / excluded (no eligible GT) / TIMEOUT / service failure | 296 / 43 / 130 / 6 | 297 / 26 / 126 / 14 | +1 / −17 / −4 / +8 |
| Capture → action-open P50 / P95 / P99 | 24.8 / 68.9 / 135.3 | 18.0 / 60.8 / 115.6 | −6.8 / −8.1 / −19.7 |
| Capture → feedback P50 / P95 / P99 | 181.7 / 264.2 / 346.1 | 176.4 / 239.8 / 319.9 | −5.3 / −24.4 / −26.2 |
| **Radar rasterize (whole builder) P50 / P95 / P99** | 16.0 / 31.3 / 63.7 | **8.5 / 23.2 / 48.5** | −7.6 / −8.2 / −15.2 |
| Worker delay (callback → worker start, admitted) P50 / P95 | −10.6 / 11.0 | −11.2 / 12.0 | ≈0 |
| Combined SI/P40 P50 / P95 | 1.97 / 6.52 | 2.11 / 7.80 | +0.1 / +1.3 |
| UE codec total P50 / P95 / P99 (wall) | 15.0 / 37.2 / 48.7 | 14.8 / 40.5 / 63.1 | −0.2 / +3.4 / +14.4 |
| Front backbone CUDA P50 / P95 | 1.14 / 3.58 | 1.19 / 5.54 | +0.05 / +1.96 |
| Action-open → last datagram P50 / P95 / P99 | 26.7 / 49.1 / 74.0 | 25.0 / 54.4 / 94.0 | −1.7 / +5.3 / +20 |
| Edge queue P95 / edge compute P50 / P95 / P99 (wall) | 29.3 / 53.7 / 82.5 / 142.4 | 38.9 / 54.6 / 81.0 / 226.8 | +9.6 / +0.9 / −1.6 / +84 |
| Edge frozen tail CUDA P50 / P95 | 38.0 / 50.6 | 38.5 / 52.3 | +0.5 / +1.7 |
| Evaluator P50 / P95 / P99 | 7.8 / 13.6 / 37.3 | 7.8 / 12.1 / 22.1 | 0 / −1.5 / −15 |
| CARLA world tick P50 / P95 / P99 | 24.7 / 62.5 / 122.3 | 27.7 / 88.9 / 133.9 | +3.0 / +26.4 / +11.6 |
| Ticks > 60 ms / > 100 ms (of ~2,045) | 106 / 41 | 181 / 80 | +75 / +39 |
| Edge supersessions (reward frames replaced / all) | 7 / 18 | 20 / 35 | +13 / +17 |
| Edge inter-arrival P5 (ms) / gaps < 50 ms | 32.6 / 84 | 24.1 / 105 | −8.5 / +21 |
| Scene-stale / MCS-stale / startup fallbacks | 12 / 7 / 2 | 8 / 9 / 2 | −4 / +2 / 0 |
| Dropped preparation opportunities | 20 | 24 | +4 |
| UE CPU mean (% one core) / busy threads mean | 105.5 / 5.6 | 102.3 / 5.5 | −3.2 / ≈0 |
| Host CPU busy mean / run queue P50 | 31.2% / 11 | 30.7% / 10 | ≈0 |
| GPU util mean / P95; memory P50; power mean | 55.3% / 71%; 8.28 GiB; 196 W | 57.3% / 76%; 8.41 GiB; 188 W | +2 / +5 pt; +0.13; −7.6 |
| Capture → map install P50 / P95 / P99 | 172.6 / 256.4 / 336.5 | 168.4 / 252.3 / 362.0 | −4.2 / −4.1 / +25.5 |
| Map outcomes installed / superseded / stale | 980 / 18 / 1 | 963 / 35 / 2 | −17 / +17 / +1 |
| Profiler hot path ms/frame (UE) | 1.02 | 1.22 | +0.2 |

Workload: payload mean 238 KB (C) vs 239 KB (T), 19.6 vs 19.6 datagrams, 2.06
vs 2.12 tensors per ticket — comparable. Identity reconciliation: zero
conflicts in both arms (G3); actor replay identical (G8).

## Reading

- The bundle did what it was meant to do on the UE CPU side: whole-builder
  rasterization halved at the median and the UE committed earlier (capture →
  action-open −7 ms P50, −20 ms P99), so capture-anchored latencies improved
  (capture → feedback −5 / −24 / −26 ms).
- The deadline that matters is action-open anchored, and it did not improve:
  P50 flat, P95 −5 ms, P99 +7.5 ms, timely fraction −1.6 pt. Two same-
  configuration captures (thread-cap B 75.2%, control C 71.4%) show the
  run-to-run spread is about 4 pt, so the primary difference is inside noise.
- The added per-frame GPU work (two kernels, H2D/D2H and a device sync inside
  the UE worker) coincides with a consistent tail degradation of every other
  GPU user: CARLA ticks > 100 ms doubled (41 → 80), UE codec P99 +14 ms with
  the front-backbone CUDA P95 +2 ms, edge compute P99 +84 ms, GPU utilization
  P95 71 → 76%. Earlier commits plus occasional edge stalls bunched arrivals at
  the edge (inter-arrival P5 32.6 → 24.1 ms), and the one-slot latest-frame
  edge admission superseded twice as many reward frames (7 → 20), which is
  exactly the rise in registered service failures (6 → 14) and missing replies
  (7 → 20). The 676 ms maximum is one such edge stall.
- Faster rasterization alone was not enough: the whole-pipeline feedback got
  no better and its failure modes shifted toward the edge.

## Recommendation: **reject for retention** (do not keep the bundle on this
shared-GPU configuration). The primary metric is inconclusive within one pair;
the secondary evidence is consistently adverse in the GPU-sharing tails. The
offline bit-exactness and the UE-side CPU gain are real and remain available if
the GPU contention is ever addressed structurally; that would be a different
experiment. Codex review follows; no further captures were run.
