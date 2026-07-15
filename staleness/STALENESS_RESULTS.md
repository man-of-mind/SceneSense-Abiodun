# Staleness / latency-FPS requirement — first results (2026-07-14)

Supervisor's takeaway experiment: localization error vs latency (Y) and FPS, per object speed, to define
quantified latency+FPS thresholds. Method: **natural traffic** (any car that drives through the sensor view) —
static pole sensor + background NPCs, loopback (clean model output, full delivery). Per-object analysis
(`analyze_staleness.py`): track each GT actor, match to predictions (gate 4 m), compute
`error(Y)=||inferred(t) − GT_actor(t+Y)||` (supervisor's method) and `pure staleness=||GT(t+Y)−GT(t)||=v*Y`.
Run: `20260714_221836_front_fusion_tl_14` (30 vehicles, 19 peds, 200 frames; 52 matched detections).

## Total localization error vs latency (supervisor's method)
| speed | n | Y=0 | 50 | 100 | 150 | 200 | 269 ms |
|---|--:|--:|--:|--:|--:|--:|--:|
| ~5–13 mph | 14 | 2.69 | 2.73 | 2.79 | 2.88 | 2.98 | 3.17 m |
| ~13–22 mph | 37 | 2.47 | 2.53 | 2.63 | 2.77 | 2.96 | 3.27 m |
(Y=0 = model-error floor; here ~2.5 m — high due to far detections + loose 4 m match gate. Total error =
model floor ⊕ staleness combined vectorially, so growth looks sub-linear.)

## Pure staleness = target displacement v*Y (the clean, model-independent budget driver)
| speed | Y=50 | 100 | 150 | 200 | 269 ms |
|---|--:|--:|--:|--:|--:|
| ped ~1.5 m/s | 0.09 | 0.18 | 0.29 | 0.40 | 0.58 m |
| ~5–13 mph (~4 m/s) | 0.24 | 0.48 | 0.74 | 1.01 | 1.40 m |
| ~13–22 mph (~8 m/s) | 0.40 | 0.80 | 1.20 | 1.61 | 2.17 m |
Matches v*Y exactly (8 m/s × 0.269 s = 2.15 ≈ 2.17) → measurement validated.

## Quantified thresholds (Y ≤ ε/v) — the benchmarking basis
| object | ε=0.5 m | ε=1.0 m |
|---|--:|--:|
| pedestrian (1.5 m/s) | 333 ms | 667 ms |
| city car (8 m/s) | 63 ms | 125 ms |
| suburban (14 m/s) | 36 ms | 71 ms |
| highway (30 m/s) | 17 ms | 33 ms |
FPS adds an inter-frame term: total lag = Y + 1/FPS, so `v*(Y + 1/FPS) ≤ ε` (10 FPS adds up to v*0.1 m).

## Ties to OAI + standards
- no-AE u8 over OAI (Y≈267 ms) → city staleness **2.1 m** → fails 0.5 m badly.
- AE-128 u4 over OAI (Y≈105 ms) → city staleness **0.84 m** → meets 1.0 m, misses 0.5 m → motivates OAI config tuning.
- Standards anchor (`../rl_agent/STANDARDS_ANCHORS.md`): ~100 ms / ~0.5 m lane-level — our derived city budget (63 ms @ 0.5 m) is *tighter* than the 100 ms anchor, i.e. car speeds are demanding.

## FINAL — purely MEASURED trend (no formula), per supervisor + Abiodun's steer
Latency Y = **capture → inference (front + uplink + back)**; downlink return to the car is NOT counted
(the map is built at the edge). `error(lag) = ‖inferred(t) − true(t+lag)‖`, measured directly, per speed bin.
FPS is the *same* curve: it adds inter-frame lag, so effective lag = Y + 1/(2·FPS) (definitional, not a fit).
Run `20260714_223011` (38 veh + 24 ped, 400 fr), gate 4 m, 70 matches. Plot: `plots/staleness_requirement.pdf`.

| speed bin | 0 | 50 | 100 | 150 | 200 | 269 ms |
|---|--:|--:|--:|--:|--:|--:|
| pedestrian ~1.5 m/s (n=6) | 2.96 | 2.97 | 2.99 | 3.01 | 3.03 | 3.06 |
| ~5–13 mph (n=17) | 2.49 | 2.53 | 2.58 | 2.65 | 2.74 | 2.90 |
| ~13–24 mph (n=47) | 2.51 | 2.47 | 2.47 | 2.51 | 2.59 | 2.79 |

**Measured trend (the deliverable):** pedestrian FLAT (+0.1 m over 269 ms), cars RISE with latency (faster = steeper)
→ latency matters for fast objects, not slow ones. Error vs FPS: drops then plateaus ~10–15 FPS.
**Y=0 = the model's own error** (no staleness) — pedestrians ~3.0 m (harder to localize), cars ~2.5 m. This is
the accuracy ceiling the network cannot beat (a tighter target needs a better model, not just lower latency).
Caveat: absolute error is high (~2.5–3 m) because these are pole detections at range + a loose 4 m match gate;
more/closer samples would lower it, but the *trend* (the requirement signal) is what matters and is clear.

## (superseded) earlier model-formula version — kept for reference
Denser run (`20260714_223011`, 38 veh + 24 ped, 400 frames), tight 2 m gate, near-range floor:
- **Model-error FLOOR = 1.37 m** (near <20 m, n=19). Validated: measured 1.42 m @ Y=0 ≈ model 1.37 m.
- **Total-error model (validated): `error = √(floor² + (v·(Y + 1/(2·FPS)))²)`** — floor measured, staleness law
  (v·Y) measured, sqrt-combination checked on the ~8 m/s bin (model slightly conservative vs measured).
- **Plot:** `staleness/plots/staleness_requirement.pdf` — error-vs-Y and error-vs-FPS, one line per speed
  (pedestrian / 20 / 30 / 40 mph), with the floor, ε bands, and our OAI Y points (AE-128 u4 105 ms, no-AE 267 ms).

**Max latency Y (ms) to keep error ≤ ε, at 10 FPS** (— = model floor already exceeds ε):
| speed | ε=1.0 | ε=1.5 | ε=2.0 | ε=3.0 m |
|---|--:|--:|--:|--:|
| pedestrian 1.5 m/s | — | 357 | 921 | 1729 |
| car 20 mph | — | 19 | 114 | 250 |
| car 30 mph | — | <0 | 59 | 149 |
| car 40 mph | — | <0 | 31 | 99 |

**Two-bottleneck lesson (the headline):**
1. **Model accuracy sets the achievable floor (~1.4 m).** ε below ~1.4 m is **model-limited, not latency-limited** —
   no amount of latency/FPS reduction reaches it (so the lane-level ~0.5 m standard needs a *better model*, not just faster network).
2. **Above the floor, latency+FPS add staleness = v·(Y+1/(2·FPS)), tight for fast objects.** e.g. at ε=2 m:
   pedestrian tolerates ~920 ms, a 30 mph car only ~59 ms → our AE-128 u4 (105 ms) is already over for 30 mph
   → **motivates OAI config tuning** (and/or higher FPS). no-AE (267 ms) fails everything but slow pedestrians.

## Earlier caveats (now largely addressed) / remaining
- Model floor high (~2.5 m): tighten match gate to ~2 m + focus near-range → cleaner floor.
- Sparse (52 matches; pedestrian n=1): longer run / denser traffic for more samples, esp. pedestrians.
- Speeds capped ~10 m/s (natural Town10): 50/80 mph need a controlled target (harness has it; ego-mode
  world-projection bug to fix first — pred coords use pole reference, not the ego pose).
- FPS sweep (subsample) to be added alongside the Y sweep.

## TRACKER (temporal fusion) + real multi-FPS captures (2026-07-14)
Built a per-object constant-velocity Kalman tracker (staleness/make_tracked_report.py) and ran REAL CARLA
captures at 5/10/20/30 FPS (staleness/run_fps_captures.sh, equal ~25s sim). Two results:
- **Model is FPS-robust (verified, not assumed):** per-frame single-frame floor ~flat across real FPS
  (10→2.56m, 20→2.74m, 30→2.68m). Running off the 10-FPS training does not degrade per-frame accuracy
  (time-based radar stationary-age confirmed).
- **Tracking + higher FPS reduces error (confirms the "more frames → better" intuition):** fast-car bin,
  single-frame/tracked: 10FPS 2.56/2.60 (no gain), 20FPS 2.74/2.41, 30FPS 2.68/2.37 @ Y=0; and at Y=269ms
  20FPS 3.31/3.00, 30FPS 3.37/3.02 — tracking lowers the floor (noise averaging) AND cancels latency
  (velocity forward-prediction). Needs ≥~15-20 FPS to pay off (at 10 FPS too few updates).
- **Single-frame FPS plateaus; temporal fusion is what makes FPS worthwhile** -> "why we need a tracker"
  (single-frame vs tracked = a clean paper comparison). Plot: staleness/plots/tracked_vs_singleframe.pdf.
- Caveat: pedestrian tracked bin is noisy/anomalous (near-stationary + CV-KF over-fit + sparse samples);
  irrelevant since pedestrians are staleness-tolerant. Car bins are the meaningful result.
