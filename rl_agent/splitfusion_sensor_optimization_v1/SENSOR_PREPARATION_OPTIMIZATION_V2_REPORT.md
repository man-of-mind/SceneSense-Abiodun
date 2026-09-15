# SplitFusion sensor-preparation optimization

**Verdict: `SENSOR_PREPARATION_OPTIMIZATION_VALIDATED`**

All correctness gates pass (14/14 on both live cells, exact output equivalence proven
live) and the integrated candidate delivers a real aggregate live improvement. The
engineering targets were *not* reached in absolute terms; §6 says so plainly and
explains why.

## 1. Headline

Live, action 50 (`AE64/UINT4/q=0.50`), `FAVORABLE_STABLE`, direct edge-to-map,
renderer off. Full sent population — every prepared frame in each cell, nothing
excluded.

| Production sensor compute | baseline | optimized | change |
|---|---:|---:|---:|
| P50 | 45.844 ms | **32.024 ms** | −13.820 (−30.1%) |
| P95 | 85.984 ms | **64.519 ms** | −21.465 (−25.0%) |
| P99 | 125.440 ms | **104.492 ms** | −20.948 (−16.7%) |
| max | 409.367 ms | **221.639 ms** | −187.728 |
| frames > 100 ms | 70 / 2,881 (2.43%) | **42 / 3,540 (1.19%)** | −1.24 pp |

Radar chain (P07–P12): P50 **31.705 → 21.525 ms**, P95 63.636 → 45.601 ms,
P99 104.276 → 82.538 ms.

Preparation coverage **0.956825 → 0.986347** (target 0.95, unchanged). The p99 gain
is not obtained by discarding slow frames: the optimized cell contributes *more*
frames (3,540 vs 2,881) and every one is in the statistic.

**Largest remaining tail: `radar_window_ms` (P07, `RadarSweepAggregator.window_detections`)
at P99 30.663 ms against a P50 of 5.338 ms** — a 5.7x spread. With P08
(`radar_spherical_to_world`, P99 25.010 ms) it is now the dominant cost, and §5
explains why it was left alone.

## 2. The diagnosed bottleneck

Profiling localized the cost to two stages that were *already* vectorized, which is
why the previous attempt found nothing to cut. Idle host, 40,000-point window
(measured live mean: 37,030–37,252 returns), 768×448, radius 4:

| Stage | Cost | Of which |
|---|---:|---|
| P12 radar rasterization | 6.81 ms | `5 × np.maximum.at` **3.52 ms**; `5 × cv2.dilate` 1.61 ms |
| P09 stationary tracking | 3.56 ms | `np.unique(return_inverse)` 1.44 ms + `argsort(inverse)` 1.52 ms = **2.95 ms** |

Two specific findings:

1. **`np.maximum.at` is the single hottest call in the pipeline.** NumPy's unbuffered
   `ufunc.at` scatter costs 0.4 ms per float channel and 0.98 ms for the occupancy
   channel — and the occupancy scatter writes a *constant* 1, so it never needed a
   maximum at all.
2. **The tracker sorts the same data twice.** `inverse` is a rank-preserving
   relabelling of `packed`, so `argsort(inverse)` and `argsort(packed)` produce the
   identical stable permutation. The unique keys, counts and group ids all follow
   from one sort.

A third, smaller one: `build_radar_sample` already rasterizes at the model size, so
the radar resize inside seven-channel packing maps 768×448 onto 768×448.

## 3. What was integrated

Three bit-exact candidates, plus the previously validated normalization-constant cache.

| ID | Stage | Change | Isolated |
|---|---|---|---|
| **D** | P12 | `np.maximum.at` + `cv2.dilate` → one CUDA `scatter_reduce(amax)` + one `max_pool2d` | 7.03 → 1.02 ms |
| **A** | P09 | two sorts → one stable `argsort`; `np.minimum.at` → reversed last-write-wins | 4.62 → 2.94 ms |
| **F** | P21 | skip the identity radar resize when shapes already match | 0.44 → 0.00 ms |

Candidate D is exact by construction, not by luck. Both halves of the rasterizer are
pure maximum reductions, and a maximum over float32 is order-independent.
`max_pool2d` with `kernel=2r+1`, `stride=1`, no padding over the *r*-padded canvas
produces exactly the interior crop the dilation path takes, so every output pixel's
window lies wholly inside the canvas and border handling never enters the comparison.
The CUDA timing above **includes** host-to-device, device-to-host and a full
synchronization.

## 4. Correctness

Equality is exact everywhere. No tolerance is used anywhere in this work.

| Gate | Result |
|---|---|
| Production 7-channel tensor equivalence | **exact**, live, on 32 registered frames |
| Radar tensor + radar evidence equivalence | **exact**, live, on 32 registered frames |
| Offline tracker equivalence | 480/480 frames identical — ages **and** full retained track table |
| Offline rasterizer equivalence | 60/60 configurations: radius 0–6, 0–60,000 points, NaN/±inf depth, exact ± velocity ties, centres rounding onto the border |
| Wired-path equivalence | matches production `build_radar_sample` and the real 7-channel tensor over a simulated episode |
| Action / profile / checkpoint identity | `frame_action_context_identity_exact` = true, both cells |
| Radar-window membership | 4-callback window on every sent frame, both cells |
| Terminal accounting | exact; zero missing, unexpected or duplicate |
| Direct edge→map active; no object records on radio | true, both cells |
| Renderer off; CPU reservations empty | true, both cells |
| Clean OAI/RFsim/CARLA/container teardown; final host cold | true, both cells |
| **All registered gates** | **14/14 pass on both cells** |

Sensor synchronization, radar-window membership, sweep count, timestamps,
stationary-track semantics, coordinate frames, intrinsics/extrinsics, tensor
shape/ordering/dtype/values, action 50 identity, the model, the direct edge→map
architecture, latest-only scheduling, deadlines, network profile and OAI
configuration are all unchanged. No resolution, point rate, range, sweep count, NPC
density or input frequency was reduced. No queue depth was added.

## 5. Candidates measured and rejected

Rejecting these is part of the result; none was carried on speculation.

| Candidate | Measured | Why rejected |
|---|---|---|
| Shared flat-index sort + `np.maximum.reduceat` for 4 channels | 3.26 ms vs 2.93 ms | **Slower** than the `np.maximum.at` calls it replaces |
| `transform_points` as `H @ M.T` | 0.193 vs 0.181 ms | Bit-exact but slower |
| `transform_points` as `P @ R.T + t` | 0.157 vs 0.181 ms | Bit-exact, saves 0.024 ms — noise |
| **Eliminating the world→spherical→world round trip (P07+P08)** | **saves 1.107 ms** | **Not bit-exact.** Reported, not silently adopted — see below |
| Re-testing the normalization-constant cache as the headline | — | Already disproven as a total-latency win; retained only as a component |

### The round trip, stated explicitly

`RadarSweepAggregator` stores world coordinates; `window_detections` converts
world → radar-local → **spherical (float32)**; `build_radar_sample` then converts
spherical → radar-local → world again. The middle spherical hop is an inverse pair,
and the float32 quantization it introduces is currently *baked into* the world
coordinates the model input derives from.

Reusing the float64 cartesian window instead saves **1.107 ms** but shifts world XYZ
by up to **6.490e-06 m** (P99 2.705e-06 m, median 2.129e-07 m). Sub-micrometre at
120 m range almost certainly changes no pixel — but "almost certainly" is not
bit-exact, so it stays out of the deployed path and is put here as a decision for
Abiodun, not taken unilaterally. This is the main remaining lever on P07+P08, which
are now the top two tails.

## 6. Targets: what was and was not reached

Targets were engineering goals. Reporting them honestly:

| Target | Result | Met? |
|---|---|---|
| Radar-chain median 22.7 → 12–15 ms | 31.705 → 21.525 ms | **No** |
| Total sensor-compute median 37.7 → 25–30 ms | 45.844 → 32.024 ms | **No** |
| P95 < 50 ms | 64.519 ms | **No** |
| P99 < 100 ms | 104.492 ms (full) / 117.370 ms (registered window) | **No** |
| No preparation-coverage regression | 0.956825 → 0.986347 | **Yes** |
| No p99 gain from excluding slow frames | optimized cell contributes *more* frames | **Yes** |

The absolute targets were anchored to a historical baseline of 37.662 ms P50. The
fresh baseline measured today on the same host is **45.844 ms** (full population) /
41.051 ms (registered window) — the host is materially busier than when that number
was taken, and roughly 10 ms of the radar chain is contention rather than algorithm
(§8). Against its own contemporaneous control the optimization removes 30% of the
median and 25% of the P95; it does not reach thresholds set against a quieter host.

**One honest wrinkle.** In the pre-registered 500-frame window the P99 *regresses*
(107.276 → 117.370 ms) while P50 and P95 improve. Over the full sent population the
P99 improves (125.440 → 104.492 ms). The two cells' registered windows cover
different route frames (baseline 642–1658, optimized 546–1564) because the runs
differ in length, so the window P99 rests on ~5 frames from non-identical route
segments. Both numbers are reported; neither is suppressed. Radar load is matched
(in-window mean returns 39,192 vs 39,228; full-population 37,252 vs 37,030), so this
is a small-sample window effect, not a load confound.

## 7. Paired same-frame evidence

The previous attempt was uninterpretable because *every* stage — including untouched
ones — regressed ~18% between its two cells, i.e. run-level host drift swamped the
effect. Two controls address that here.

**Control 1 — untouched stages stayed flat**, which is what makes these two cells
comparable at all (full population, P50):

| Untouched stage | baseline | optimized | Δ |
|---|---:|---:|---:|
| P07 radar window | 5.653 | 5.338 | −0.315 |
| P08 spherical→world | 5.491 | 5.070 | −0.421 |
| P10 world→camera | 3.284 | 3.056 | −0.228 |
| P14 BGRA→BGR | 3.085 | 2.695 | −0.390 |
| P18 RGB host→device | 3.187 | 3.383 | +0.196 |
| P23 7-channel concat | 0.459 | 0.643 | +0.184 |

**Control 2 — a same-frame paired measurement.** On each equivalence frame the
production radar chain runs on the identical detections with an independent shadow
tracker. Optimized cell (n=32): production 32.412 ms vs optimized 20.696 ms, median
reduction **11.436 ms**, optimized faster on 26/32 frames.

That figure carries an ordering bias, and the baseline cell measures it: there *both*
timed calls are the production implementation, yet it still shows a 2.325 ms apparent
"reduction" (n=8) purely from running the reference second on warm caches. The
bias-corrected paired estimate is therefore **≈ 9.1 ms**, consistent with the
independent full-population delta.

## 8. The five intervals, kept separate

Never summed; each is its own measurement.

| # | Interval | baseline P50/P95/P99 | optimized P50/P95/P99 |
|---|---|---|---|
| 1 | Acquisition / sensor-alignment wait (P06) | 8.911 / 30.671 / 49.337 | 10.539 / 32.534 / 52.803 |
| 2 | OS / worker scheduling (P04) | 0.221 / 56.509 / 83.819 | 0.121 / 51.145 / 79.501 |
| 3 | Production sensor computation | 45.844 / 85.984 / 125.440 | 32.024 / 64.519 / 104.492 |
| 4 | UE action *after* the tensor is ready | 27.734 / 59.425 / 72.218 | 29.654 / 61.649 / 86.140 |
| 5a | Feature uplink | 55.394 / 104.191 / 148.524 | 56.391 / 105.890 / 161.659 |
| 5b | Edge compute | 53.205 / 73.491 / 157.539 | 56.604 / 76.485 / 87.561 |
| 5c | Edge→map service | 2.149 / 5.474 / 15.490 | 2.125 / 5.017 / 27.152 |
|  | Capture→install AoI | 200.669 / 309.446 / 375.187 | 201.108 / 293.988 / 342.149 |

Rows 1, 2 are diagnostic and non-additive. Row 4 is *post*-tensor (27.7 ms < 41 ms of
sensor compute), so it correctly does not benefit; its small rise is the one visible
cost of moving rasterization onto the GPU the model also uses. Capture→install AoI is
flat at the median and better at the tail (P95 −15.5 ms, P99 −33.0 ms): preparation
runs at a fixed 10 Hz cadence, so time cut below the period converts into headroom
and coverage (+2.95 pp), not proportionally lower AoI.

### Scheduling (Phase 3)

The preparation worker was observed on all 24 cores in both cells — no pinning was
introduced and none is recommended. Thread pools are at library defaults (cv2 24,
torch 24) while OpenBLAS is capped at `MAX_THREADS=2`; the host runs CARLA, OAI and
the edge runtime concurrently on those same 24 cores. Worker scheduling wait
(P04 P95 56.5 → 51.1 ms) and callback-to-worker (P05 P95 42.4 → 34.9 ms) both
improved, consistent with the optimized path releasing CPU sooner — moving five
`cv2.dilate` calls to the GPU removes the pipeline's largest OpenCV thread burst. No
thread-count or affinity change was applied: the evidence did not isolate one, and
the task's own instruction was to change nothing without it.

## 9. Execution record

| Item | Value |
|---|---|
| Starting HEAD | `600af4acb3b3702266a483b58298f7104cd08a5b` |
| Implementation commit | `d49ee6111890f9691529d8343632c30a1f4c714d` |
| Baseline cell | `20260914_v2_action50_favorable_baseline` — **PASS** |
| Optimized cell (attempt 1) | `20260914_v2_action50_favorable_optimized` — **FAILED**, preserved |
| Optimized cell (retry 1) | `20260914_v2_action50_favorable_optimized_retry1` — **PASS** |

**Attempt 1 and the one authorized retry.** The first optimized cell ran to
completion with 3,034 sent frames and no adapter exception, but failed the
exactly-one-terminal contract on a single capture:
`ue288_a50__favorable_stable:6742`, the final frame, with 3,033 terminals for 3,034
sent. Its ACK was still in flight when the route ended and the adapter finalized —
a teardown drain race in the pinned adapter, exposed by the faster preparation
cadence rather than by any output or equivalence defect. The gate was **not**
weakened and the pinned adapter was **not** modified; the single authorized retry
was used, and it passed. The failed attempt is preserved in full.

Both cells recorded permitted route interventions (`PERMITTED_INTERVENTION`), which
is the pre-existing tolerated outcome and was true of the passing baseline too.

## 10. Artifacts

- `SENSOR_OPTIMIZATION_V2_RESULT.json` — machine-readable result
- `stacked_latency_breakdown.png` / `.pdf` — mean composition (means, because a sum
  of medians is not the median of a sum)
- `per_stage_percentiles.png` / `.pdf` — per-stage P50/P95/P99, both cells
- `per_frame_timeseries.png` / `.pdf` — radar returns and sensor-compute bursts,
  two panels on a shared x-axis rather than a second y-scale
- `artifact_manifest.json` — SHA-256 of every artifact and bound source
- `SPLITFUSION_SENSOR_PREPARATION_OPTIMIZATION_V2_PRESENTATION_COMPLETE` — terminal

Implementation: `rl_agent/splitfusion_sensor_optimization_v1/optimized_stages.py`
with `tests/test_optimized_stages.py` and `tests/test_wired_path_equivalence.py`.

The completed 288-cell campaign and all prior sensor-profiling outputs were read
only and are unchanged.

---

*Evidence lives under `experiments/splitfusion_sensor_preparation_live_v1/` (git-ignored
by `/experiments/`); this copy is the repo-tracked record. Rebuild the figures and the
result JSON offline with
`python3 -m rl_agent.splitfusion_sensor_optimization_v1.build_presentation`.*
