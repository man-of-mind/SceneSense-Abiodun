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

## Result 1b — detection distance & road-state breakdown
Post-hoc on the same speed-sweep observations (829 matched, `make_roadstate_breakdown.py`; road state from the
CARLA Town10 map at each target's position). No re-capture.

**Detection distance (ego camera → tracked car):** close-range by design.
| min | p25 | median | p75 | p90 | max |
|--:|--:|--:|--:|--:|--:|
| 3.3 | 8.0 | **13.1** | 17.7 | 21.4 | 24.9 m |
The analysis gates to ≤25 m (clean localization floor); the model *sees* cars out to ~60 m (all-in-view median
34 m) but we deliberately localize the close ones. So: **cars are tracked within 25 m, typically ~13 m away.**

**Road-state mix:** 70% straight road, 30% intersection (no distinct standalone-curve component — Town10 curves
are gentle / inside junctions). So the headline result is a **mixture, straight-dominated**.

**Per-speed error(Y), split by road state** — the two plots `plots/roadstate_straight_speed.pdf` and
`plots/roadstate_intersection_speed.pdf` (each = the Result-1 per-speed family, but filtered to that road state).
**The speed/latency story is identical in both**: walk ≈ flat, faster = steeper, ~28–32 mph reaches ~3.8 m at
267 ms on *both* straight roads and at intersections. Aggregated (`plots/roadstate_error_latency.pdf`):
| road state | n | Y=0 | 105 ms | 267 ms |
|---|--:|--:|--:|--:|
| straight | 584 | 1.12 | 1.25 | 1.78 |
| intersection | 245 | 1.17 | 1.33 | **2.00** |
So road state does **not** change the trend — it mainly nudges the *floor* (intersections ~0.1–0.2 m worse:
cars turn → breaks the constant-velocity assumption, and views are more oblique). Next: *target* specific road
states (dwell the ego at an intersection vs a straight) for even cleaner per-state curves.

## Result 2 — FPS & temporal accumulation
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
Same model, different transport Y (per-frame accuracy is channel-invariant — proven in the OAI A/B):
- **no-AE (267 ms):** fast objects badly hurt (32 mph → 4.4 m single-frame).
- **AE-128 (105 ms):** ~halves fast-object error; meets ε≈2 m for most speeds.
- **+ fusion at ≥20 FPS:** recovers another ~0.3–0.6 m on fast objects.
So the levers are complementary: **compression cuts latency**, **fusion + FPS cancels residual staleness**,
**model accuracy sets the ~1.1 m floor**.

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

## Repro
- Speed sweep: `run_speed_sweep.sh` + `run_speed_targeted.sh` → `make_speed_error_report.py`.
- FPS captures: `run_fps_captures_ego.sh` (+ `run_fps_fast.sh`) → `make_fps_fusion_report.py` (per-speed,
  temporal-accumulation vs single-frame, accumulation-depth report). Legacy: `make_tracked_report.py`.
- Requirement/latency curve: `make_staleness_report.py <run> 2.0`.
- Scenario knobs added: `--npc-speed-difference-pct`, `--npc-ignore-lights-pct`; GT now logs `origin_*`.
