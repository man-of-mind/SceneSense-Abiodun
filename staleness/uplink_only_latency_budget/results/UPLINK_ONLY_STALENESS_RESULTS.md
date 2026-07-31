# Uplink-only latency-budget staleness analysis — results (2026-07-30)

Redo of the staleness / latency-budget → RL-agent-constraint analysis for the **uplink-only Track-1
architecture** (`car → split features → edge tail → edge publishes to spatial map`). Supersedes, for the
uplink-only pipeline, the lag definition used in `../../STALENESS_RESULTS.md`.

**Everything here is ideal loopback, uplink-only.** No OAI transport is mixed in (OAI is a separate radio
study — `../../../uplink_only_spatial_map_pipeline/TRACK1_OAI_DEFAULT106_RESULTS.md`). No downlink /
result-return term exists in this architecture, so **`Y_down` is gone from the lag** — the edge publishes
detections straight to the map and the car receives only a tiny async warning.

Plan followed: `../PLAN.md`. Every guardrail in its 🚦 section was checked; see §6.

---

## 0. Headline

| | |
|---|---|
| **Staleness lag** | `L = Y_sensorprep + Y_front + Y_uplink + Y_tail + Y_mapinsert` = full **capture → map-update-done** age. **No `Y_down`.** |
| **Measured L (ideal loopback, optimized pipeline)** | **67.5 ms p50 / 101.8 ms p95** (fresh, 570 frames) — **93.3 / 136.1 ms** (Track-1 50-frame profile, kept as the conservative design anchor) |
| **Sensor prep share of L** | **57–65 %** — the single largest term. Not the network, not the model. |
| **Model floor** | **≈ 1.1 m**, latency- and FPS-independent. `ε < ~1.1 m` is infeasible by any latency/FPS action. |
| **Master constraint** | **`v · (L + s/FPS) ≤ √(ε² − 1.1²)`**, `s` = 0.5 (average query) or 1 (worst case). No `Y_down`. |

The single most consequential correction: using the **core split→map latency (38 ms)** instead of the full
capture→map age would understate a 32 mph car's error by **0.50 m** (1.44 m vs 1.94 m at L=93 ms). The object
keeps moving during sensor preparation, so sensor prep is part of the staleness lag.

---

## 1. Where the freshness age goes (Step A)

Current reporting uses the **optimized Track-1 pipeline** as the default. The older legacy-rasterizer
comparison is retired from the report/plots because that computation path was buggy and should not be used as
a presentation claim. Raw legacy artifacts remain in the folder only as provenance.

Re-derived from the **per-frame** `map_ingest_metrics.csv` of the optimized profile run
(`../../../uplink_only_spatial_map_pipeline/runs/live_front_prep_profile_fast_50f/`), first 10 frames excluded
as warm-up. The per-frame additivity residual `(prep + core − L)` is **0.000 ms** on every frame, so the
decomposition is exact, not approximate.

Presentation plot: `plots/presentation/presentation_staleness_budget_breakdown.pdf` · summary CSV:
`plots/presentation/presentation_latency_breakdown_summary.csv`

Presentation-ready versions of the main story plots are in `plots/presentation/`. These use the
"staleness budget" wording, fold radar tensor build into the total sensor-prep block, and use additive
stage definitions so the map term is `edge tail done → map update done` rather than a double-counted
edge-receive-to-map-publish interval.

| Component | bucket | optimized p50 / p95 |
|---|---|--:|
| CARLA sync tick | sensor prep | 33.4 / 86.5 |
| camera frame wait | sensor prep | 34.7 / 46.9 |
| radar packet wait | sensor prep | 0.0 / 0.0 |
| RGB convert | sensor prep | 3.9 / 7.6 |
| **radar tensor build** | sensor prep | **32.6 / 52.8** |
| model preprocess | sensor prep | 13.1 / 23.0 |
| **`Y_sensorprep`** (capture→backbone input) | rollup | **53.6 / 82.1** |
| front backbone (split encoder) | `Y_front` | 5.1 / 10.4 |
| feature serialize (zstd) | `Y_front` | 1.1 / 3.7 |
| `Y_uplink` (front→edge, loopback) | `Y_uplink` | 7.8 / 13.4 |
| `Y_tail` (edge tail inference) | `Y_tail` | 10.2 / 21.4 |
| map UDP ingest/queue | `Y_mapinsert` | 8.3 / 60.0 |
| map service (update apply) | `Y_mapinsert` | 0.0 / 0.0 |
| core split→map (**NOT the staleness lag**) | rollup | 37.7 / 80.2 |
| **`L` = capture → map update done** | **rollup** | **93.3 / 136.1** |

**Sensor prep is 57 % of conservative-anchor L.** The uplink itself is ~8 ms and the edge tail ~10 ms: on
ideal loopback the split/network path is a minor term. The important current takeaway is that frontend sensor
preparation is part of map staleness and must be included in the agent budget.

### 1a. Fresh independent measurement of L

A fresh uplink-only loopback run was made for this analysis (2026-07-30, reusing the running CARLA server):
3 traffic regimes × 200 frames, true uplink-only (`--edge-result-mode none`, edge publishes to the map),
optimized deployed recipe. **570 post-warm-up frames** vs 40 in the Track-1 profile.

CSV: `fresh_L_by_condition.csv` · plot: `plots/fresh_run_L_and_error.pdf` (left panel)

| condition | n | L p50 | L p95 | `Y_sensorprep` p50 | radar build p50 | core split→map p50 |
|---|--:|--:|--:|--:|--:|--:|
| `L_normal` (NPC speed 0 %) | 190 | 67.8 | 106.1 | 44.7 | 24.0 | 22.4 |
| `L_fast` (−45 %) | 190 | 67.9 | 100.4 | 43.6 | 23.6 | 23.2 |
| `L_veryfast` (−88 %) | 190 | 66.5 | 94.7 | 42.5 | 23.5 | 23.0 |
| **pooled** | **570** | **67.5** | **101.8** | **43.8** | **23.7** | **22.8** |

Two things worth stating plainly:

1. **L is insensitive to traffic speed regime** — 66.5–67.9 ms p50 across walk→32 mph traffic, a spread of
   1.4 ms. The lag is a pipeline property, not a scene property. Good news for the agent: it does not need
   to model L as a function of NPC speed.
2. **The fresh p50 is 25.8 ms *lower* than the Track-1 anchor (67.5 vs 93.3 ms, −27.6 %).** This is a real
   difference, not noise, and it is explained rather than averaged away:

| stage p50 | Track-1 fast_50f | fresh (570 f) | Δ |
|---|--:|--:|--:|
| radar tensor build | 32.6 | 23.7 | −8.9 |
| camera frame wait | 34.7 | 28.1 | −6.6 |
| map UDP queue | 8.3 | 3.0 | −5.3 |
| edge tail | 10.2 | 6.2 | −4.0 |
| front backbone | 5.1 | 2.0 | −3.1 |
| model preprocess | 13.1 | 18.7 | +5.6 |
| effective capture cadence | 7.10 FPS | 8.72 FPS | +1.62 |

The mechanism is a **self-reinforcing cadence/radar-density loop**: CARLA fixes radar density in
points *per second*, so a faster loop collects fewer points per frame, which makes the raster cheaper, which
speeds the loop further. At 8.72 FPS the frame carries ~23 k radar points vs ~28 k at 7.10 FPS (ratio 0.81);
radar build falls in the same proportion (23.7/32.6 = 0.73). The remaining deltas (map queue, tail, backbone)
are host-contention terms.

**Decision:** report `L` as a **measured range of 67–93 ms p50** for the ideal-loopback uplink-only path, and
compute the headline budgets at **both** anchors. The 93 ms figure is retained as the *conservative design
anchor* (it is what the PLAN specifies and it errs toward safety); 67.5 ms is the current *best estimate*
(5× the sample, matched to the traffic used for the accuracy dataset). All budget formulas below are closed
forms in `L`, so any operating value can be substituted.

---

## 2. Error vs the uplink-only lag (Step B)

Post-hoc on the **existing** speed-sweep opportunity-window captures (`../../metrics_logs/scenesense_runs/`,
6 `speedsweep_*` runs, **829 matched observations**, walk→32 mph). Staleness is object kinematics, so the
method is the original one re-parameterized with the new `L`: `error(v) = ‖pred(t) − GT_origin(t + L)‖`,
moving car-height ego, ≤25 m, 2 m match gate, score ≥0.2. No new captures were needed for this table.

Presentation plot: `plots/presentation/presentation_error_vs_speed_by_staleness.pdf` · CSV:
`error_vs_L_by_speed.csv`

Localization error (m) vs uplink-only `L`:

| speed band | n | L=0 (floor) | 38 ms *(core only — understates)* | **67 ms** (fresh p50) | **93 ms** (design anchor) | 136 ms (p95) |
|---|--:|--:|--:|--:|--:|--:|
| ~walk/slow | 148 | 1.16 | 1.17 | 1.17 | 1.17 | 1.17 |
| ~6 mph | 325 | 1.11 | 1.10 | 1.11 | 1.12 | 1.14 |
| ~10 mph | 98 | 1.11 | 1.15 | 1.19 | 1.24 | 1.35 |
| ~14 mph | 57 | 1.05 | 1.14 | 1.25 | 1.35 | 1.54 |
| ~18 mph | 94 | 1.13 | 1.17 | 1.25 | 1.35 | 1.55 |
| ~23 mph | 17 | 1.21 | 1.41 | 1.60 | 1.78 | 2.15 |
| ~28 mph | 39 | 1.19 | 1.12 | 1.20 | 1.35 | 1.69 |
| **~32 mph** | 51 | 1.29 | 1.44 | **1.67** | **1.94** | 2.49 |

- **Model floor ≈ 1.1 m at every speed** (1.16 m at v<1 mph, n=139; 1.13 m pooled). Anchored to the offline
  knob-matrix no-AE u8 result of **0.95 m** (`../../../rl_agent/PERMODEL_KNOB_MATRIX_ZSTD.md`); fresh-drive
  scenes run ~0.2 m above the offline held-out estimate, consistent with the earlier validation. The ~3 m
  loose-matcher numbers in `FAST_RASTERIZER_ACCURACY_AB.md` are **not** the floor and are not used here.
- **Pedestrians and slow traffic are latency-immune.** ~walk/slow stays near 1.16–1.17 m across the current
  optimized-anchor range. Fast cars are not: at the design anchor a 32 mph car is 1.94 m off vs 1.17 m for a
  pedestrian.
- **Using the core 38 ms instead of the full age understates fast-car error by 0.50 m** (1.44 vs 1.94 m at
  32 mph). This is exactly the trap guardrail 1 warns about.

---

## 3. FPS as map-hold staleness, on top of L

**Corrected framing (guardrail 5):** the spatial map holds the last detection between updates, so at update
rate FPS the held position is up to `1/FPS` stale → error ≈ `v·(1/FPS)` even at `L=0`. This is *not*
single-frame-vs-accumulation; per-frame accuracy is FPS-independent because the model is single-frame.
`s=0.5` = average query timing, `s=1` = worst case. Full tables in `error_vs_fps.csv`; presentation plot
`plots/presentation/presentation_fps_requirement_by_speed.pdf`.

Worst case (`s=1`) at the design anchor `L=93 ms`:

| band | 1 FPS | 5 FPS | 10 FPS | 15 FPS | 20 FPS | 25 FPS | 30 FPS |
|---|--:|--:|--:|--:|--:|--:|--:|
| ~walk/slow | 1.32 | 1.18 | 1.17 | 1.17 | 1.17 | 1.17 | 1.17 |
| ~6 mph | 3.63 | 1.34 | 1.19 | 1.16 | 1.15 | 1.14 | 1.14 |
| ~10 mph | 5.93 | 1.89 | 1.52 | 1.42 | 1.37 | 1.34 | 1.32 |
| ~18 mph | 8.72 | 2.58 | 1.89 | 1.68 | 1.59 | 1.54 | 1.51 |
| ~28 mph | 13.29 | 3.46 | 2.29 | 1.93 | 1.76 | 1.67 | 1.61 |
| **~32 mph** | 16.38 | 4.71 | 3.28 | 2.81 | 2.59 | 2.45 | 2.36 |

At 1 FPS a 32 mph car moves ~14.3 m between updates and the map is unusable regardless of network quality.
Gains saturate by ~20 FPS for everything up to ~18 mph; 32 mph is still improving at 30 FPS.

**Achievable FPS is CARLA/testbed-bound, not split-inference-bound (guardrail 7).** The live frontend tops out
at ~7–10 FPS after the optimization (8.72 FPS measured in the fresh run), limited by CARLA
simulation/render/sensor production plus sensor prep — a no-background diagnostic reached 11.84 FPS with CARLA
tick p50 falling 71.6 → 27.0 ms. The **map path itself sustains a true 30 FPS** when fed by the
model-boundary offered-load replay with no map compute. So FPS operating points above ~10 are *analytically*
valid here but not yet demonstrated end-to-end on this testbed; do not quote ~7–10 FPS as a limitation of
split inference.

---

## 4. Recomputed uplink-only budgets (Step C)

`B(ε) = √(ε² − 1.1²)` → ε=1.5 → 1.02 m, ε=2.0 → 1.67 m, ε=2.5 → 2.24 m, ε=3.0 → 2.79 m.

### 4a. Latency UPPER bound — max uplink-only `L` (ms, capture→map) to hold error ≤ ε

`measured` = interpolated from the direct `GT(t+L)` curve; `closed` = `B(ε)/v`.
`—` = model floor already exceeds ε (model problem, not a latency problem). CSV: `budget_latency_upper.csv`.

| band | v (m/s) | ε≤1.5 m | ε≤2.0 m | ε≤2.5 m | ε≤3.0 m |
|---|--:|--:|--:|--:|--:|
| ~walk/slow | 0.08 | >300 / — | >300 / — | >300 / — | >300 / — |
| ~6 mph | 3.23 | >300 / 316 | >300 / 518 | >300 / 696 | >300 / 865 |
| ~10 mph | 4.60 | 187 / 222 | >300 / 363 | >300 / 488 | >300 / 607 |
| ~14 mph | 6.31 | 127 / 162 | 224 / 265 | >300 / 356 | >300 / 442 |
| ~18 mph | 8.11 | 125 / 126 | 211 / 206 | 282 / 277 | >300 / 344 |
| ~23 mph | 10.52 | 52 / 97 | 119 / 159 | 171 / 213 | 220 / 265 |
| ~28 mph | 12.33 | 114 / 83 | 167 / 135 | 212 / 182 | 255 / 226 |
| **~32 mph** | 14.76 | **47 / 69** | **98 / 113** | 137 / 152 | 173 / 189 |

(cells are `measured / closed-form`; walk/slow closed-form values are >13 s, i.e. no practical limit.)

**Read against the measured operating range (67–93 ms), everything up to ~28 mph clears ε=2 m, and a 32 mph
car clears ε=2 m at 67 ms but is marginal at 93 ms (bound 98–113 ms).** At ε=1.5 m the 32 mph bound is
47–69 ms, i.e. tighter than the deployed lag — lane-adjacent accuracy on fast cars needs a shorter capture→map
age than the current frontend delivers.

### 4b. FPS LOWER bound at each measured anchor

`FPS_min(v,ε) = v / (B(ε) − v·L)` (worst case `s=1`). CSV: `budget_fps_lower.csv`.

| band | v (m/s) | ε≤1.5 m @68 / @93 ms | ε≤2.0 m @68 / @93 ms | ε≤2.5 m @68 / @93 ms | ε≤3.0 m @68 / @93 ms |
|---|--:|--:|--:|--:|--:|
| ~walk/slow | 0.08 | 0.1 / 0.1 | ~0 / ~0 | ~0 / ~0 | ~0 / ~0 |
| ~6 mph | 3.23 | 4.0 / 4.5 | 2.2 / 2.4 | 1.6 / 1.7 | 1.3 / 1.3 |
| ~10 mph | 4.60 | 6.5 / 7.8 | 3.4 / 3.7 | 2.4 / 2.5 | 1.9 / 1.9 |
| ~14 mph | 6.31 | 10.6 / 14.7 | 5.1 / 5.8 | 3.5 / 3.8 | 2.7 / 2.9 |
| ~18 mph | 8.11 | 17.2 / 30.9 | 7.2 / 8.9 | 4.8 / 5.5 | 3.6 / 4.0 |
| ~23 mph | 10.52 | 34.0 / 273.8 | 11.0 / 15.3 | 6.9 / 8.3 | 5.1 / 5.8 |
| ~28 mph | 12.33 | 65.8 / **INFEAS** | 14.7 / 23.7 | 8.7 / 11.3 | 6.3 / 7.5 |
| **~32 mph** | 14.76 | 621 / **INFEAS** | 21.9 / 50.3 | 11.8 / 17.0 | 8.2 / 10.4 |

`INFEAS` = `v·L` alone consumes the entire budget `B(ε)`; **no FPS fixes it — `L` must come down.** This is
the sharpest practical consequence of counting sensor prep: at L=93 ms, ε=1.5 m is unreachable at any update
rate for ≥28 mph, whereas at L=68 ms it is reachable (66 FPS at 28 mph — analytically, well above the current
testbed ceiling). The sensitivity of these cells to a 25 ms change in `L` is why `L` must be measured, not assumed.

### 4c. Headroom at the operating points

Remaining staleness budget `B(ε) − v·L` after the lag is paid. CSV: `budget_headroom.csv`.

| band | v (m/s) | `v·L` @68 ms | `v·L` @93 ms | B(1.5)−v·L @68 / @93 | B(2.0)−v·L @68 / @93 |
|---|--:|--:|--:|--:|--:|
| ~walk/slow | 0.08 | 0.01 | 0.01 | +1.01 / +1.01 | +1.67 / +1.66 |
| ~6 mph | 3.23 | 0.22 | 0.30 | +0.80 / +0.72 | +1.45 / +1.37 |
| ~10 mph | 4.60 | 0.31 | 0.43 | +0.71 / +0.59 | +1.36 / +1.24 |
| ~14 mph | 6.31 | 0.43 | 0.59 | +0.59 / +0.43 | +1.24 / +1.08 |
| ~18 mph | 8.11 | 0.55 | 0.76 | +0.47 / +0.26 | +1.12 / +0.91 |
| ~23 mph | 10.52 | 0.71 | 0.98 | +0.31 / +0.04 | +0.96 / +0.69 |
| ~28 mph | 12.33 | 0.83 | 1.15 | +0.19 / **−0.13** | +0.84 / +0.52 |
| **~32 mph** | 14.76 | 1.00 | 1.38 | +0.02 / **−0.36** | +0.67 / +0.29 |

### 4d. Master constraint (uplink-only)

> **`v · (L + s/FPS) ≤ √(ε² − 1.1²)`**
> with `L = Y_sensorprep + Y_front + Y_uplink + Y_tail + Y_mapinsert` and **no `Y_down`**.

Feasibility map: `plots/feasibility_L_fps.pdf` (ε=2 m boundary in `(L, FPS)` for pedestrian / 18 mph / 32 mph).

---

## 4e. Error split by road state (straight / curve / intersection) — added 2026-07-30

Reproduces the original Result 1a for the **uplink-only lag**, on the **same 829 observations** (origin GT,
202800/202800 origin rows), road state from the Town10 map at each target's GT position (`is_junction`; curve =
yaw-change >4°/5 m; else straight), evaluated at L ∈ {0, 67, 93, 136} ms.
Script: `../make_roadstate_speed_uplink_only.py` · plots: `plots/uplink_roadstate_{straight,curve,intersection}_speed.pdf`
· CSV: `roadstate_error_by_speed.csv`.

**Confound check (why this must be read per-speed).** Road state correlates with speed — the count matrix:

| speed band | straight | curve | junction |
|---|--:|--:|--:|
| walk/slow | 58 | 13 | 77 |
| ~6 mph | 147 | **142** | 36 |
| ~10 mph | 76 | **1** | 21 |
| ~14 mph | 28 | **2** | 27 |
| ~18 mph | 29 | 24 | 41 |
| ~23 mph | 5 | 0 | 12 |
| ~28–32 mph | 38 | 21 | 31 |

The **curve bin is 70 % ~6 mph** (142/203) with ~10/14 mph at n=1/2 — cars slow on curves. So the *pooled* curve
aggregate is speed-confounded; only straight-vs-intersection are comparable across speed (same limitation the
original flagged).

**At a fixed speed and the operating lag, road state has little independent effect.** Loc error (m) at L=93 ms:

| speed | straight | curve | intersection |
|---|--:|--:|--:|
| ~6 mph | 1.11 | 1.15 | 1.02 |
| ~18 mph | 1.30 | 1.35 | 1.38 |
| ~28–32 mph | 1.79 | 1.46 *(n=21, thin+confounded)* | 1.71 |

The spread across road states at a given speed is ≲0.1–0.3 m — within sampling/thin-bin noise. The dominant drivers
remain **object speed and L**, not road geometry. Curve's lower ~28–32 mph value rests on a thin, confounded bin and
must **not** be read as "curves are easier."

**Conclusion:** the uplink-only run confirms the original — road state is **not an independent staleness driver**; it
matters only through the speed distribution it correlates with. **No change to the agent constraints: condition on
speed and L, not road type.**

## 5. What was reused vs re-measured

| | |
|---|---|
| **Reused as-is** | The 6 `speedsweep_*` opportunity-window captures (829 observations) for all error(v), FPS and budget tables. The `RADAR_RASTERIZER_SHADOW_VALIDATION.md` equivalence result. The offline knob-matrix no-AE u8 floor anchor (0.95 m). The offered-load replay evidence that the map path sustains 30 FPS. |
| **Re-derived from raw data** | The current `L` decomposition — computed from the per-frame `map_ingest_metrics.csv` of the optimized profile run rather than quoted from the summary table (per-frame additivity residual 0.000 ms). |
| **Newly measured (fresh run, 2026-07-30)** | `fresh_run_20260730_000257/` — 6 conditions on the already-running CARLA server: 3 × 200-frame true-uplink-only conditions for `L` (570 post-warm-up frames) and 3 × 400-frame accuracy-instrumented conditions (613 matched observations). |
| **Newly computed** | Uplink-only error(v) at the current reporting anchors L∈{0, 38, 67, 93, 136} ms, both FPS tables (`s`=0.5 and 1) at the 67/93 ms L anchors, all three budget tables, presentation plots, and the distribution-averaged staleness `E_L[err]`. |

### 5a. The fresh run — what it is and is not used for

The uplink-only client cannot log both `L` and object motion in one condition: in true uplink-only mode
(`--edge-result-mode none`) the no-wait front loop skips the prediction/GT logging block entirely (verified:
header-only CSVs in the existing Track-1 profile runs). The fresh run therefore uses **two condition
families** so each number comes from a configuration that can measure it honestly:

- **`L_*` (3 × 200 frames)** — true uplink-only, edge publishes to the map. This is the *only* configuration in
  which `capture_to_map_update_done_ms` contains no downlink. Used for §1a. **Passes.**
- **`ACC_*` (3 × 400 frames)** — identical sensor/model/codec/rasterizer recipe with the result-return enabled
  purely so predictions and actor-origin GT are logged front-side, and the spatial-map stream off. Its
  downlink is **never** added to `L`.

**The `ACC_*` accuracy dataset was gated and partially rejected.** Its validation flags (full detail in
`run_log_fresh_run.txt`):

| check | result |
|---|---|
| F1 per-observation direct-vs-closed-form | mean −0.064 m, median −0.009 m — **pass** (method is sound) |
| F2 sample at v<1 mph | n=17 — **too thin to pin a floor** |
| F3 floor speed-ordering | walk/slow 1.47 m vs ~18 mph 1.02 m, inverted by +0.45 m — **fail** |
| F4 monotonicity of error(L) | non-monotonic in ~10 mph (n=69) and ~32 mph (n=15) — **fail** |

Cause of F3, established rather than assumed: the fresh `~walk/slow` bin averages **1.01 m/s** while the
baseline's averages **0.08 m/s**. With `--npc-ignore-lights-pct 100` and only 400 frames per regime the fresh
sweep produced almost no *stopped* vehicles, so the same nominal bin contains creeping vehicles rather than
parked ones — harder to localize. The bins hold different content, so the absolute floors are not comparable.

Consequently **no headline number is taken from the fresh accuracy dataset.** The floor, error(v) curves and
all budgets stay on the 829-observation baseline pool, which passes the full gate. The fresh accuracy data is
used only as a floor-insensitive consistency check on staleness *growth*, via the implied displacement
`√(err(L)² − err(0)²)` which should equal `v·L` regardless of each dataset's floor. The table below is a
validation stress check only; it is not part of the active reporting anchors:

Each dataset is compared against **its own** mean band speed, since the same nominal bin holds different
content in the two runs:

| band | fresh v | fresh implied | fresh exp `v·L` | baseline v | baseline implied | baseline exp `v·L` |
|---|--:|--:|--:|--:|--:|--:|
| ~walk/slow | 1.01 | 0.25 | 0.18 | 0.08 | 0.12 | 0.01 |
| ~6 mph | 2.22 | 0.00 | 0.40 | 3.23 | 0.41 | 0.58 |
| ~10 mph | 4.58 | 0.00 | 0.83 | 4.60 | 0.98 | 0.83 |
| ~14 mph | 6.30 | 0.67 | 1.14 | 6.31 | 1.42 | 1.14 |
| ~18 mph | 8.12 | **1.45** | 1.47 | 8.11 | 1.41 | 1.47 |
| ~23 mph | 10.70 | 1.68 | 1.93 | 10.52 | 2.30 | 1.90 |
| ~28 mph | 11.85 | **2.02** | 2.14 | 12.33 | 1.79 | 2.23 |
| ~32 mph | 14.51 | 2.06 | 2.62 | 14.76 | 2.82 | 2.67 |

The fresh dataset tracks its own `v·L` in the well-sampled fast bands (~18 mph n=181: 1.45 vs 1.47 expected;
~28 mph n=39: 2.02 vs 2.14) and collapses to 0.00 in the slow bands where its error(L) was non-monotonic. The
baseline tracks `v·L` across the board within ±0.3 m except ~6 mph. **Conclusion: the staleness physics
reproduces independently at speed on a fresh run; the fresh run's slow-speed floor does not, and is not used.**

---

## 6. Guardrail compliance (PLAN §🚦)

| # | Guardrail | Status |
|---|---|---|
| 1 | `L` = full capture→map age (optimized pipeline), not core split→map | **Held.** L=67/93 ms used throughout; the 38 ms core column is shown *only* labelled "understates", and the 0.50 m error it would hide at 32 mph is quantified. |
| 2 | No downlink term | **Held.** No `Y_down` anywhere. The one condition family with a result-return exists solely for GT/prediction logging and its downlink is excluded from `L`; the `L` numbers come only from true uplink-only conditions where the *edge* publishes. |
| 3 | GT = actor origin, not bbox centre | **Held.** `USING_ORIGIN=True` on both datasets, 0 rows missing `origin_x/y`; the loader *hard-fails* instead of falling back to `world_x/world_y`. Conventions verified to genuinely differ (‖origin−centre‖ p50 0.041 m, max 0.841 m). |
| 4 | Loopback only; label everything | **Held.** No OAI numbers used; every table and plot says "ideal loopback, uplink-only". |
| 5 | Corrected FPS framing | **Held.** Map-hold `s/FPS`, `s`∈{0.5,1}; explicitly noted per-frame accuracy is FPS-independent. |
| 6 | Floor ≈1.1 m is model-limited; anchor to offline 0.95 m | **Held.** Recovered 1.16 m at v<1 mph; anchored to offline no-AE u8 0.95 m; loose-matcher ~3 m figures explicitly excluded. ε<1.1 m flagged as a model problem. |
| 7 | Live FPS is CARLA/testbed-bound | **Held.** §3 states the ~7–10 FPS ceiling is CARLA sim/render + sensor prep and cites the offered-load replay showing the map path sustains 30 FPS. |
| 8 | Validate before findings; don't rescue broken data | **Held.** Both datasets passed through an explicit gate. The baseline passed (829 obs, floor 1.16 m, per-observation direct-vs-closed-form mean −0.022 m / median 0.000 m / sd 0.341 m). The fresh accuracy dataset **failed 3 of 4 checks and was demoted**, not rescued. |
| — | Scene density out of scope | **Held.** No density analysis here; it belongs to `../../../rl_agent/DENSITY_ADAPTIVE_KNOB_PLAN.md`. |

---

## 7. Caveats

- **Ideal loopback only.** `L` = 67–93 ms is the loopback age. Over OAI the uplink term alone is far larger
  and the ranking of terms changes; that is a separate study.
- **Latency is swept analytically on real captures.** Object motion and GT are real; `L` is applied as a
  time offset rather than by injecting transport delay. This isolates staleness cleanly but assumes the
  detection itself is unchanged by the lag — true for this pipeline, where `L` does not alter model input.
- **Thin bands.** ~23 mph (n=17) in the baseline and ~32 mph (n=15) in the fresh set are under-sampled;
  Town10 has a genuine occupancy valley at 20–26 mph. The ~28 mph baseline cell (1.35 m at 93 ms, below its
  own closed form of 1.65 m) reflects instantaneous-speed binning of targets that then decelerate — the
  per-band residual scatters both ways (mean +0.035 m) and is unbiased overall.
- **`L` is not a constant.** The 25.8 ms gap between the two measurements is driven by capture cadence and
  radar points/frame. The agent should treat `L` as an observed, sensor-prep-dominated quantity rather than a
  fixed constant — see the constraints doc.
- **Single-frame model, idealized association.** No tracker/temporal fusion in these numbers; a
  constant-velocity filter that predicts forward by `L` would recover part of the staleness term (~0.3–0.9 m
  on fast objects in the earlier study) at the cost of needing real data association.

## 8. Reproduce

```bash
AB=/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun
cd $AB
export MPLCONFIGDIR=/tmp/matplotlib-cache QT_QPA_PLATFORM=offscreen
PY=/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python

# Step A - L decomposition from the per-frame profile CSVs
$PY staleness/uplink_only_latency_budget/analyze_L_decomposition.py

# Steps B + C - validation gate, error(v), FPS tables, budgets, plots
$PY staleness/uplink_only_latency_budget/analyze_uplink_only_staleness.py

# Fresh run (needs a running CARLA on rpc-port 2000; reuses it, never starts/kills one).
# NOTE: do NOT export PYTHONPATH for the client - see the comment at the top of the script.
staleness/uplink_only_latency_budget/run_fresh_uplink_only_speedsweep.sh
$PY staleness/uplink_only_latency_budget/analyze_fresh_run.py
```

Logs: `run_log_staleness.txt`, `run_log_fresh_run.txt`, `../fresh_run.log`.
Agent-facing constraints: `UPLINK_ONLY_AGENT_CONSTRAINTS.md`.
