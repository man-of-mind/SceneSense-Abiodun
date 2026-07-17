# Latency & FPS requirement for localization — results (2026-07-16)

Supervisor's ask: quantify how localization error depends on **end-to-end latency** and **camera FPS**, per
object speed (pedestrian vs 20/30 mph car), by comparing the inferred position to the true position *at a
timestamp*. Error should shrink as latency drops and FPS rises.

> Supersedes the earlier pole-based numbers (the pole is out-of-domain; those were scrapped). All results here
> are on the **validated in-domain setup**: moving **car-height ego** (z=1.55 m, pitch −4°, FOV 120° — matches
> training), RGB+radar fusion, **loopback no-AE** (clean, 100% delivery), **origin GT convention** (see below).

## Method
- **Opportunity windows:** the ego drives among traffic; any vehicle that enters good range (in camera frustum,
  ≤25 m) and matches a prediction (tight 2 m gate) is an observation. Each is binned by its **measured
  instantaneous world speed**. NPC speed regime is swept per run (ignore-lights) to populate walk → ~32 mph.
- **Latency Y = capture → inference** (front + uplink + back). *Not* the downlink return (map is built at edge).
  Y is swept synthetically on one clean recording (same model, same detections — only Y varies → fair isolation).
- **GT convention fix (critical):** training regresses `actor.get_location()` = actor **origin**; the live GT
  logger had been recording the bbox **center**. That mismatch inflated live error ~1 m. Fixed — GT now logs
  `origin_x/y` and the analysis compares against it. (This also resolved the earlier model-validation scare.)
- **Single-frame vs temporal fusion:** the model is *single-frame* — each frame decoded independently, no
  accumulation, so per-detection accuracy is FPS-independent. To turn "more frames" into "more accuracy" you
  must **fuse** — a recursive constant-velocity Kalman filter accumulates past frames and **predicts forward by
  Y** to cancel staleness. Higher FPS → fresher, more frequent updates → better velocity → better prediction.

## Result 1 — localization error vs latency, per target speed
Plot: `plots/speed_error_requirement.pdf` (single-frame; 829 observations, walk→32 mph). Measured (uses real GT
positions at t+Y), and it tracks √(floor² + (v·Y)²) — e.g. 32 mph @269 ms predicted 3.59 m vs measured 4.36 m.

| target speed | Y=0 (floor) | 105 ms (AE-128) | 267 ms (no-AE) |
|---|--:|--:|--:|
| pedestrian / walk | 1.15 | 1.16 | 1.18 |
| ~6 mph | 1.10 | 1.12 | 1.30 |
| ~10 mph | 1.07 | 1.21 | 1.75 |
| ~14 mph | 1.09 | 1.43 | 2.25 |
| ~18 mph | 1.11 | 1.39 | 2.41 |
| ~28 mph | 1.19 | 1.61 | 3.17 |
| ~32 mph | 1.29 | ~2.0 | **4.36** |

- **Model floor ~1.1 m at every speed** (the model's own error, latency-independent) — matches offline (veh 0.88 m).
- **Pedestrians are latency-immune; fast cars are not.** At the no-AE operating latency (267 ms) a 32 mph car
  is **4.4 m** off vs ~1.2 m for a pedestrian. Compression (AE-128, 105 ms) roughly halves the fast-car error.

### Requirement table — max latency Y (ms) to keep error ≤ ε (— = model floor already exceeds ε)
| speed | ε≤1.5 m | ε≤2.0 m | ε≤2.5 m | ε≤3.0 m |
|---|--:|--:|--:|--:|
| walk / ~6 mph | >269 | >269 | >269 | >269 |
| ~10 mph | 187 | >269 | >269 | >269 |
| ~14–18 mph | ~125 | ~215 | >269 | >269 |
| ~28 mph | 112 | 166 | 212 | 255 |
| ~32 mph | 45 | 98 | 137 | 173 |

Lane-level (~0.5 m) is **model-limited** (floor ~1.1 m) — unreachable by any latency/FPS; needs a better model.

## Result 1a — per-speed error(Y) split by road state (straight / curve / intersection)
Plots: `plots/roadstate_straight_speed.pdf`, `plots/roadstate_curve_speed.pdf`, `plots/roadstate_intersection_speed.pdf`.
Post-hoc on the same speed-sweep observations (829 matched, `make_roadstate_speed_plots_with_curves.py`; road state
from the CARLA Town10 map at each target's position). Road-state threshold: curve = yaw-change >4°/5m (lowered from
strict 8°/5m to capture Town10's gentle curves). No re-capture.

**Descriptive values (not a causal curve effect):** the pooled curve rows grow more slowly with Y:
| road state | ~6 mph | ~18 mph | ~28-32 mph |
|---|--:|--:|--:|
| straight   | 1.07→1.36 m | 1.08→2.38 m | 1.24→4.00 m |
| **curve**  | **1.16→1.27 m** | **1.22→2.31 m** | **1.05→3.64 m** |
| intersection | 1.07→1.12 m | 1.10→2.48 m | 1.39→3.79 m |

**Interpretation limit:** the curve bin is **70% ~6 mph** (n=142/203), while ~10 mph has n=1 and ~14 mph n=2.
Cars slow down on curves, so the pooled curve aggregate is speed-confounded. The faster matched-speed cells are
also thin (n=21–24) and their lower Y=0 floor is consistent with sampling noise. Therefore this analysis supports
the straight-vs-intersection comparison only; it does **not** establish that curvature improves prediction. A curve
effect requires targeted curved-road captures at controlled speeds.

---

## Result 1b — detection distance & overall road-state aggregates
Post-hoc on the same speed-sweep observations (829 matched, `make_roadstate_breakdown.py`; road state from the
CARLA Town10 map at each target's position). No re-capture.

**Detection distance (ego camera → tracked car):** close-range by design.
| min | p25 | median | p75 | p90 | max |
|--:|--:|--:|--:|--:|--:|
| 3.3 | 8.0 | **13.1** | 17.7 | 21.4 | 24.9 m |
The analysis gates to ≤25 m (clean localization floor); the model *sees* cars out to ~60 m (all-in-view median
34 m) but we deliberately localize the close ones. So: **cars are tracked within 25 m, typically ~13 m away.**

**Road-state mix:** 46% straight, 24% curve, 30% intersection (when threshold is 4°/5m). Town10 curves are gentle
and sometimes embedded in junctions; lowering the threshold from 8° to 4° exposes them. Headline Result-1 (main
per-speed plot) uses a pooled dataset; the curve subset is retained only as an exploratory, speed-confounded split.

**Per-speed error(Y), split by road state** — shown in the three per-speed plots (Result 1a, above).
Aggregated latency-vs-error across all speeds (`plots/roadstate_error_latency.pdf`):
| road state | n | Y=0 | 105 ms | 267 ms |
|---|--:|--:|--:|--:|
| straight | 381 | 1.10 | 1.16 | 1.85 |
| curve | 203 | 1.15 | 1.16 | 1.66 |
| intersection | 245 | 1.17 | 1.21 | 2.00 |

**Supported interpretation:** straight and intersection subsets are well sampled; intersections have the larger
aggregate error at 267 ms (2.00 vs 1.85 m), plausibly because turning violates constant-velocity motion and views
are more oblique. The lower pooled curve value cannot be ranked against them because its speed mix is different.

## Result 2 — FPS as spatial-map staleness (the corrected framing)
**Context:** the spatial map queries the latest detection; between frames it holds a **stale position**. At update
rate FPS, the worst-case staleness is 1/FPS (detection queried just before next frame boundary). Static car → fine;
fast car → error ≈ v × (1/FPS) **even at ideal network conditions (Y=0)**. This is pure update-rate staleness.

**Plots:**
- `plots/fps_mapStaleness_worstcase.pdf`: FPS staleness at **ideal conditions (Y=0)** — shows best-case error per FPS
- `plots/fps_latency_matrix_*.pdf`: **Grouped bar charts** (10, 18, 32 mph) showing FPS × Latency interaction; 4 bars per FPS (Y=0, 50ms, 105ms, 267ms)
- `plots/fps_latency_*.pdf`: Line plots combining FPS + network latency

Computed post-hoc on existing speed-sweep captures: error = ||pred(t) − GT(t + latency + 1/FPS)||, no new runs needed.
Shows the classic **v × (1/FPS) relationship**:
| speed | 1 FPS | 5 FPS | 10 FPS | 20 FPS | 30 FPS |
|-------|-------|-------|--------|--------|--------|
| walk/slow | 1.29 | 1.17 | 1.17 | 1.17 | 1.17 |
| ~6 mph | 3.32 | 1.20 | 1.12 | 1.10 | 1.10 |
| ~10 mph | 5.41 | 1.54 | 1.25 | 1.16 | 1.14 |
| ~18 mph | 8.01 | 1.93 | 1.38 | 1.20 | 1.16 |
| ~28 mph | 12.14 | 2.36 | 1.40 | 1.14 | 1.12 |
| ~32 mph | 15.02 | 3.37 | 2.02 | 1.52 | 1.41 |

**Interpretation:** FPS is a **latency analog**. At 1 FPS a 32 mph car moves ~14.3 m between frames → 15 m error (even with zero network latency); at 30 FPS it moves ~0.48 m per frame → 1.4 m error (near the model floor). Very low FPS is unusable for fast cars. Adding network latency **on top** of map staleness makes it worse (see grouped bar charts).

**Key insight from matrix:** the best lever to reduce error for fast cars is **FPS** (huge drops 1→5→10 FPS); **latency** then becomes the secondary lever (bigger impact at low FPS, smaller at high FPS). For example, a 32 mph car at ideal conditions (Y=0): 1 FPS is 15 m, 5 FPS is 3.4 m, 10 FPS is 2.0 m. Adding 267ms latency alone doesn't cause that – it's the map staleness rate that dominates.

**FPS plateau (guardrail for RL agent):** improvement gain by speed band (per 10 FPS jump):
| speed | 5→10 FPS | 10→20 FPS | 20→30 FPS | **Saturates?** |
|-------|----------|----------|----------|----------|
| walk/slow | 0.00 m | 0.00 m | 0.00 m | **yes @ 10 FPS** |
| ~6 mph | 0.08 m | 0.02 m | 0.00 m | **yes @ 10 FPS** |
| ~10 mph | 0.29 m | 0.09 m | 0.02 m | **yes @ 20 FPS** |
| ~18 mph | 0.55 m | 0.18 m | 0.04 m | **yes @ 20 FPS** |
| ~28 mph | 0.97 m | 0.26 m | 0.02 m | **yes @ 25 FPS** |
| **~32 mph** | **1.35 m** | **0.50 m** | **0.11 m** | **30 FPS still improving** |

**Recommendation:** 20 FPS is the practical minimum (all sub-critical speeds saturate); for occasional fast cars
(28–32 mph), 25–30 FPS would be needed. Beyond 30 FPS the gain is <0.1 m for all speeds.

**FPS + network latency coupling:** combined plots (`fps_latency_*.pdf`) show total error at lag Y + 1/FPS. At operating
latencies (loopback 50ms, AE-128 105ms, no-AE 267ms), higher FPS recovers staleness margin. For example, at AE-128
(105ms) with a 32 mph car: 10 FPS → 3.0m, 20 FPS → 1.7m, 30 FPS → 1.5m. This coupling emphasizes the importance
of FPS when network latency is high.

---

## Result 2a — FPS & temporal accumulation (legacy — single-frame vs Kalman)
Plot: `plots/fps_fusion_22mph.pdf` (fast bin; also `fps_fusion_pedestrian.pdf` as the flat control). Real CARLA
captures at 5/10/20/30 FPS. "Temporal accumulation" = a recursive constant-velocity Kalman that fuses **all past
frames** of an object's track (recent-weighted) and predicts forward by Y to cancel staleness — vs "single-frame"
(memoryless). Accumulation DEPTH = track length: at 30 FPS ~9–64 frames fused per track (speed-dependent), at
10 FPS only ~3–5 (fast objects leave the near-zone fast, so they fuse fewer frames).

- **Single-frame accuracy is FPS-independent** (each frame is its own snapshot): floor flat across 10/20/30 FPS
  → the **model is FPS-robust** (running off its 10-FPS training does not degrade it).
- **Temporal accumulation + higher FPS lowers error for fast objects, and the gain grows with FPS** (~22 mph car,
  @269 ms): single-frame flat ~2.8–3.0 m; accumulated **2.59 → 2.33 → 1.99 m at 10 → 20 → 30 FPS** (gain −0.40 →
  −0.91 m). Higher FPS fuses more frames over a *fresher* window → better velocity → better forward prediction.
- **Takeaway:** raw per-frame perception does **not** get more accurate with FPS — you realize the "more FPS =
  less error" expectation only with **temporal accumulation**. → motivates (a) a tracker in the pipeline, (b) camera ≥ ~20 FPS.
- **30 mph can't be swept across FPS** (only 30 FPS has enough sustained tracks) — itself a finding: fast objects
  need high FPS just to *stay tracked*. ~22 mph is the honest fast bin.

## Ties to OAI + the headline
Same model, different transport Y and different FPS (per-frame accuracy is FPS-independent and channel-invariant):
- **no-AE (267 ms):** at 10 FPS, 32 mph car reaches 4.9 m; at 20 FPS drops to 2.5 m.
- **AE-128 (105 ms):** at 10 FPS 32 mph → 3.0 m; at 20 FPS → 1.7 m; approaches floor.
- **FPS ≥ 20 alone (zero latency):** 32 mph car holds ~1.4 m (near floor).
- **Fusion at ≥20 FPS with low latency:** recovers another ~0.3 m on fast objects.

**The three complementary levers:**
1. **Compression cuts transport latency** (no-AE 267 → AE-128 105 ms)
2. **FPS cuts map-staleness** (1/FPS latency analog; ≥20 FPS saturates)
3. **Temporal fusion (Kalman) + high FPS** recovers residual staleness under latency
4. **Model accuracy sets the floor** (~1.1 m, model-limited, not latency-limited)

## Honesty / caveats
- **Idealized data association:** the tracker uses ground-truth to assign detections to tracks (clean
  measurement); a deployed tracker must solve association itself, so real gains would be somewhat lower.
- **~22 mph is a Town10 occupancy valley** (cars cruise ~18 or jump to 26+); captured as a wider ~23 mph band.
- **Pinned fast-speed × per-FPS matrix not feasible** with natural traffic: a 30 mph car crosses the near zone
  in ~1–2 s, so low FPS can't sustain a track on it (**itself a finding**: fast objects need high FPS just to
  stay tracked). A clean pinned-speed accumulation sweep would need a sustained controlled target (deferred;
  convoy and controlled-target both drift out of view — a target-generation issue, not a model issue).
- **Offline vs fresh-scene:** floor here (~1.1 m) is ~0.2 m above the offline held-out estimate (0.88 m) —
  fresh-drive scenes are slightly harder; the ranking/knob-effects hold.

## Result 3 — FOV-position diagnostic: CORRECTED CENTERED GATE FAILED; NO LATERAL RUN

The old Result-3 dataset remains invalid: black/mis-decoded camera input and broken target/coordinate handling
produced kilometre-scale errors. No absolute or relative pattern from that deleted run is usable.

The hardened `carla_fov_diagnostic_exact.py` fixed several real camera, synchronization, transform, and spawn bugs,
but the later baseline-parity audit found that its outputs were collected with direct in-process inference, 5,000
radar points/s, and a permissive 5 m association gate. They therefore do not reproduce the intended no-AE deployed
recipe (loopback + uint8 + zlib + ROI 0 at 200,000 points/s). The centered error was also about 2 m; at a 2 m gate,
only 10/30 static-center and 40/60 moving-center samples matched. The claimed FOV curve and moving/static diagnosis
are withdrawn, and the generated FOV folders have been removed.

The later moving-ego/lead-NPC run was also the wrong protocol: it changed the route/background while attempting to
measure the centered condition. Its folder and dedicated analyzer were removed, its measurements are withdrawn,
and it must not be cited as Result 3.

The corrected canonical run is `experiment3_vehicle_lateral/centered_200k_15m_v1/`. It parks the ego at spawn 80,
places one tagged Lincoln exactly 15 m ahead at zero lateral offset, and uses the current no-AE checkpoint through
actual UDP loopback with the full 200k training radar recipe. A pre-run audit caught and removed another harness
error: immediately freezing the ego left it at CARLA's elevated spawn transform (camera world `z≈2.30 m`). The
retained run gives both Lincolns the training collector's 30 physics-settling ticks before freezing, producing ego
origin `z=0.020 m`, target origin `z=-0.003 m`, and camera world `z=1.565 m`, matching the original training view.

All non-accuracy checks passed: 60/60 loopback results, 60/60 visible target opportunities, exact 15.000/0.000 m
forward/lateral placement, zero-pixel mean center offset, and raw radar support in every frame (mean 1,860 points;
learned support score ~1.0). At score ≥0.20 with NMS radius 2/top-k 120, 59/60 opportunities matched within 2 m,
but conditional mean/median/p90 error was **1.474/1.483/1.569 m**, above the frozen ≤1.30/≤1.20 m mean/median
bounds. All 60 matched within 5 m at **1.506/1.483/1.577 m**. Saved overlays show ambiguous duplicate depth peaks
around the centered car rather than a transport, visibility, or radar-support failure. The lateral crossing was not
run. Full diagnosis: `experiment3_vehicle_lateral/README.md`.

A 10 m centered follow-up (`experiment3_vehicle_lateral/centered_200k_10m_v1/`) used the same corrected harness and
again passed all protocol checks: 60/60 loopback results, 60/60 visible target opportunities, exact 10.000/0.000 m
placement, camera world `z=1.565 m`, and raw radar support in every frame (mean 3,354 points). At the frozen
score ≥0.20 analysis threshold it still failed: only 52/60 frames had a score-qualified vehicle prediction and
20/60 matched within 2 m. The ≤2 m conditional mean/median/p90 was **1.239/1.251/1.336 m**, but availability and
median failed; within 5 m, 52/60 matched at **2.215/2.657/3.016 m**. A read-only threshold diagnostic showed the
correct peak is often present below the frozen threshold: score ≥0.10 gives 60/60 within 2 m with
**1.106/1.113/1.284 m** mean/median/p90. This points to score calibration / duplicate peak selection, not missing
radar or bad placement. No lateral sweep is registered until the threshold and target-selection rule are deliberately
re-frozen.

That parked-ego explanation was later superseded by a cleaner offline control: splitting the model's own 200k test
data by ego speed showed stopped-ego vehicle localization at **0.78 m**, better than moving-ego frames at
**0.96 m**, and about a third of the training set is stopped ego. Therefore, stopped/parked ego is not the root
cause. The controlled Experiment-3 failure is best treated as an artificial-scene artifact: one isolated tagged
vehicle in a deterministic dead-behind, sparse scene does not match the natural multi-object Town10 training
distribution.

The useful FOV answer now comes from a post-hoc natural-scene split in `staleness/fov_posthoc/`. On the offline
200k eval (`score ≥0.20`, match ≤2 m, vehicles ≤40 m), FOV edge position mainly hurts match availability and
medium/far localization, not near-field localization. Near vehicles (0-15 m) remain easy even near the edge
(edge median **0.208 m**, availability **0.845**). For 15-25 m, edge availability drops to **0.786** and median
error rises to **1.251 m**. For 25-40 m, edge availability is **0.792** and median error **1.174 m**. So the
actionable result is range-aware edge risk, not a blanket "center is always best" rule.

## Repro
- Speed sweep (Result 1): `run_speed_sweep.sh` + `run_speed_targeted.sh` → `make_speed_error_report.py` → split by road state in `make_roadstate_speed_plots_with_curves.py` and `make_roadstate_breakdown.py`.
- FPS as staleness (Result 2): post-hoc on same speed-sweep captures → `make_fps_speed_report.py` (no new runs; shows error at t+1/FPS, one line per FPS value) + combined with network latency.
- FPS + temporal accumulation (Result 2a, legacy): `run_fps_captures_ego.sh` (+ `run_fps_fast.sh`) → `make_fps_fusion_report.py` (per-speed, temporal-accumulation vs single-frame, accumulation-depth report).
- FOV-position diagnostic (Result 3): analyze the retained 15 m centered run with
  `python3 staleness/analyze_experiment3_vehicle_lateral.py experiment3_vehicle_lateral/centered_200k_15m_v1`.
  Analyze the 10 m follow-up with
  `python3 staleness/analyze_experiment3_vehicle_lateral.py experiment3_vehicle_lateral/centered_200k_10m_v1 --expected-forward-m 10.0`.
  Both intentionally exit with status 2 while the frozen center gate fails. No lateral command is registered.
- Natural-scene FOV split (Result 3 replacement): `python3 staleness/posthoc_fov_position_analysis.py` →
  `staleness/fov_posthoc/FOV_POSTHOC_RESULTS.md` and plots/CSVs in the same folder.
- Requirement/latency curve: `make_staleness_report.py <run> 2.0`.
- Scenario knobs: `--npc-speed-difference-pct`, `--npc-ignore-lights-pct`; GT logs `origin_*` (not bbox-center).
