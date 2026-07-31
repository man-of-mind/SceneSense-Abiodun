# RL agent — UPLINK-ONLY staleness-budget constraint spec (2026-07-30)

Citable constraint spec for the network-aware split-inference controller, **for the uplink-only Track-1
architecture** (`car → split features → edge tail → edge publishes to spatial map`; the car receives only a
tiny async warning, never full detections).

> **This does NOT overwrite `../../../rl_agent/AGENT_CONSTRAINTS.md`.** That document remains the record of
> the Stage-1 round-trip framing. **For any uplink-only work, this document supersedes it** on three points:
> the definition of the staleness budget, the removal of `Y_down`, and the inclusion of sensor prep. Where the two
> disagree about uplink-only numbers, this one is current.
>
> Evidence: `UPLINK_ONLY_STALENESS_RESULTS.md` (same folder). All numbers are **ideal loopback, uplink-only**.
> Not valid over OAI.

---

## 0. What changed from `AGENT_CONSTRAINTS.md`

| | old (Stage-1 round-trip framing) | **new (uplink-only)** |
|---|---|---|
| budget variable | old name: uplink/round-trip latency | **new name: staleness budget / freshness age** |
| lag | `L = Y_up + 1/FPS`, extended to `L_total = Y_up + 1/FPS + Y_down + Y_map_share` | **`L = Y_sensorprep + Y_front + Y_uplink + Y_tail + Y_mapinsert` (+ `s/FPS`)** |
| `Y_down` | present, to be measured before training | **deleted — the architecture has no result return** |
| sensor prep | not a term (lag was essentially transport) | **first-class, and the LARGEST term (57–65 % of L)** |
| operating anchor | 105 ms (AE-128 OAI) / 267 ms (no-AE OAI) | **67–93 ms p50 measured, ideal loopback** |
| floor | ≈1.1 m | ≈1.1 m (**unchanged**) |

**Delete `Y_down` if you copy the old formula.** It is not small-but-unmeasured here; it does not exist.

---

## 1. The one-line staleness model

> **error(v) ≈ √( floor² + (v · (L + s/FPS))² )**,  floor ≈ **1.1 m**
>
> `L = Y_sensorprep + Y_front + Y_uplink + Y_tail + Y_mapinsert`  — **frame capture → map-update-done**. **No `Y_down`.**
> `s` = 0.5 (average map-query timing) or 1 (worst case). `v` = tracked object world speed (m/s).

The agent should treat this as a **staleness budget**, not merely an uplink latency budget. The object keeps
moving from the moment the camera/radar frame is captured, so sensor preparation, front compute, uplink,
edge tail inference, and map insertion all age the detection. The controllable uplink may be only one term
inside the budget.

Validated on 829 real opportunity-window observations: per-observation direct-`GT(t+L)` vs this closed form
agrees to mean **−0.022 m**, median **0.000 m**, sd 0.341 m.

**Feasible motion budget:** `B(ε) = √(ε² − 1.1²)` → ε=1.5 → **1.02 m**, ε=2.0 → **1.67 m**,
ε=2.5 → **2.24 m**, ε=3.0 → **2.79 m**.

## 2. Hard floor — feasibility LOWER bound

- **ε_floor ≈ 1.1 m.** No latency or FPS action beats it. Model-limited (offline no-AE u8 **0.95 m**;
  live in-domain 1.16 m at v≈0).
- **Agent consequence:** an accuracy target **ε < ~1.1 m is infeasible** — do not reward chasing it. Flag
  lane-level (~0.5 m) as needing a **better model**, not a network action.
- Do **not** anchor the floor to loose-matcher live numbers (~3 m at a 5 m gate). Those are not the floor.

## 3. The staleness age `L` — what the agent must know

**`L` starts at frame capture and is dominated by sensor preparation, not by the network.** At the measured operating point (p50, ideal
loopback, optimized pipeline):

| term | fresh p50 | share |
|---|--:|--:|
| `Y_sensorprep` (CARLA tick + camera wait + radar raster + preprocess) | 43.8 ms | **65 %** |
| `Y_front` (split encoder + zstd serialize) | ~2.7 ms | 4 % |
| `Y_uplink` (front→edge, loopback) | 6.3 ms | 9 % |
| `Y_tail` (edge tail inference) | 6.2 ms | 9 % |
| `Y_mapinsert` (UDP ingest/queue + update apply) | ~3.0 ms | 4 % |
| **`L` total** | **67.5 ms** (p95 101.8) | |

(Shares are each stage's p50 over L's p50, so they sum to ~91 % rather than 100 % — medians of a sum are not
the sum of medians. Per-*frame* the decomposition is exact: residual 0.000 ms on every frame.)

**Map-insert definition:** for the additive staleness budget, `Y_mapinsert` is measured from
`edge tail done → map update done`. In the fresh run this is about **3 ms p50**. Do not add
`edge_to_map_publish_ms` to this term: that column starts at edge receive and therefore already includes
edge-tail work. The map packet is small, but the measured term still includes Python UDP receive scheduling,
zlib/JSON parsing/normalization, queue admission, and the trivial current update apply. It does **not** yet
include future Hungarian/JPDA association, occlusion reasoning, cooperative fusion, or warning selection.

Consequences for the controller:

1. **The compression knobs the agent controls (AE / quant / ROI) act on a small slice of `L` on loopback.**
   `Y_front + Y_uplink` is ~9 ms of 67 ms. Compression is still the right lever *over a real radio* (where
   the uplink term dominates and delivery matters), but on an ideal link the agent cannot buy much freshness
   by compressing harder. Do not let a loopback-trained policy conclude that payload reduction is a large
   latency lever — it is a *reliability and radio-occupancy* lever.
2. **`L` is measured, not constant.** Two same-recipe measurements gave 67.5 ms (570 frames) and 93.3 ms
   (40 frames) p50. The difference is capture cadence × radar points-per-frame, not the split path. **Put
   observed `L` (or its components) in the agent STATE**; do not hard-code a constant.
3. **`L` is insensitive to traffic speed regime** — 66.5–67.9 ms p50 across walk→32 mph traffic (spread
   1.4 ms). The agent does not need to model `L` as a function of scene speed.
4. **A frontend compute optimization is an accuracy action.** Reducing sensor-prep cost lowers `L`, which
   directly lowers motion-induced localization error for fast objects at identical model weights. If the
   action space is ever extended beyond compression, frontend compute belongs in it.

## 4. Staleness UPPER bound per speed — max capture→map `L` (ms) to hold error ≤ ε

Measured (interpolated from the direct `GT(t+L)` curve). `—` = floor already exceeds ε.

| object speed | ε ≤ 1.5 m | ε ≤ 2.0 m | ε ≤ 2.5 m | ε ≤ 3.0 m |
|---|--:|--:|--:|--:|
| pedestrian / ≤6 mph | no limit | no limit | no limit | no limit |
| ~10 mph | 187 | >300 | >300 | >300 |
| ~14 mph | 127 | 224 | >300 | >300 |
| ~18 mph | 125 | 211 | 282 | >300 |
| ~23 mph | 52 | 119 | 171 | 220 |
| ~28 mph | 114 | 167 | 212 | 255 |
| **~32 mph** | **47** | **98** | 137 | 173 |

Closed form (FPS high): **`L_max(v,ε) = B(ε)/v`**.

**Against the deployed 67–93 ms:** everything up to ~28 mph clears ε=2 m. A 32 mph car clears ε=2 m at 67 ms
and is **marginal at 93 ms** (bound 98 ms). ε=1.5 m on fast cars requires a *shorter* `L` than the current
frontend delivers.

## 5. FPS LOWER bound per speed — min map update rate

`FPS_min(v,ε) = v / (B(ε) − v·L)` (worst case `s=1`; use `s=0.5` for average query timing).
`INFEAS` = `v·L` alone consumes the whole budget → **no FPS fixes it; `L` must come down.**

| object speed | ε≤1.5 m @68 / @93 ms | ε≤2.0 m @68 / @93 ms | ε≤2.5 m @68 / @93 ms |
|---|--:|--:|--:|
| ≤6 mph | 4.0 / 4.5 | 2.2 / 2.4 | 1.6 / 1.7 |
| ~10 mph | 6.5 / 7.8 | 3.4 / 3.7 | 2.4 / 2.5 |
| ~14 mph | 10.6 / 14.7 | 5.1 / 5.8 | 3.5 / 3.8 |
| ~18 mph | 17.2 / 30.9 | 7.2 / 8.9 | 4.8 / 5.5 |
| ~23 mph | 34.0 / 273.8 | 11.0 / 15.3 | 6.9 / 8.3 |
| ~28 mph | 65.8 / **INFEAS** | 14.7 / 23.7 | 8.7 / 11.3 |
| **~32 mph** | 621 / **INFEAS** | **21.9 / 50.3** | 11.8 / 17.0 |

Note how violently these cells move for a 25 ms change in `L` — another reason `L` belongs in the state.

**Per-frame accuracy is FPS-independent** (the model is single-frame and FPS-robust). FPS buys only
*freshness*: the map holds the last detection between updates, so staleness ≈ `v·(s/FPS)`. Do **not** reward
FPS as if it improved per-frame perception.

## 6. MASTER staleness constraint (the agent must satisfy this)

> **`v · (L + s/FPS) ≤ √(ε² − 1.1²)`**
>
> `L = Y_sensorprep + Y_front + Y_uplink + Y_tail + Y_mapinsert`, from **frame capture** to **map update done**.
> **NO `Y_down`. NO `Y_map_share`** until a
> cooperative map-fusion stage actually exists and is instrumented (`map_service_ms` is currently 0.0 ms —
> the map does ingest and apply, not association/occlusion/fusion yet).

- **Hard constraint:** reject/penalize any action whose predicted `v·(L + s/FPS) > B(ε)` for the current
  tracked speed.
- **Cost side:** minimize bandwidth (payload × FPS) subject to the constraint → compress hard and drop FPS
  for slow objects, spend for fast ones.
- **Do not** target ε < 1.1 m (infeasible). **Do not** treat FPS as improving per-frame accuracy.

## 7. Policy insight — still speed-gated, but now prep-gated too

- **Slow / pedestrian (≤~10 mph):** latency- and FPS-immune across the current optimized reporting range
  (roughly 0→136 ms). Compress hard,
  drop FPS, save bandwidth.
- **Fast (≥~28 mph):** needs both low `L` and high update rate, and still cannot beat ~1.1 m. At `L`=93 ms,
  ε=1.5 m is infeasible at *any* FPS.
- **New:** the binding term for fast objects on loopback is **sensor prep**, which the agent does not
  currently control. Either extend the action space to frontend compute/sensor-prep settings, or treat `L`
  as an exogenous observed variable and let the agent manage only `FPS` and payload against it.
- **State should include:** tracked object speed (and its uncertainty), observed `L` or its sensor-prep
  component, and current map update rate. Range/FOV-position adds a secondary risk term (range-aware edge
  risk, not a blanket centre-FOV prior).

## 8. Operating anchors (keep transport labels explicit)

| condition | `L` p50 / p95 | meets ε=2 m up to | notes |
|---|--:|--:|---|
| **ideal loopback, uplink-only, optimized pipeline (fresh, 570 f)** | **67.5 / 101.8 ms** | ~32 mph | current best estimate; deployed recipe (no-AE u8, zstd, 200k PPS) |
| ideal loopback, uplink-only, optimized pipeline (Track-1, 40 f) | 93.3 / 136.1 ms | ~28 mph (32 marginal) | **conservative design anchor** |
| core split→map only (**not a staleness lag**) | 37.7 / 80.2 ms | — | omits sensor prep; understates 32 mph error by 0.50 m. **Never use as `L`.** |

Achievable live update rate is **~7–10 FPS, CARLA/testbed-bound** (sim/render + sensor prep), not a split-
inference limit — the map path itself sustains a true 30 FPS under model-boundary offered-load replay. FPS
operating points above ~10 in §5 are analytically valid but not yet demonstrated end-to-end here.

## 9. Provenance & caveats

- Evidence: `UPLINK_ONLY_STALENESS_RESULTS.md`; raw CSVs and plots in the same folder; logs
  `run_log_staleness.txt`, `run_log_fresh_run.txt`.
- Error(v)/floor/budgets come from the **829-observation** baseline speed-sweep pool (passes the full
  validation gate). The fresh run's accuracy conditions **failed 3 of 4 gate checks** (thin v≈0 sample,
  speed-inverted floor, non-monotonic bands) and were used only as a floor-insensitive check on staleness
  growth — no headline number derives from them. The fresh run's `L` conditions passed and are used for `L`.
- In-domain, ideal loopback, no channel impairment, single UE, Town10, walk→32 mph. Latency is swept
  analytically on real captures (real object motion and GT; `L` applied as a time offset).
- Single-frame model, idealized data association, no temporal fusion. A constant-velocity filter predicting
  forward by `L` would recover part of the staleness term (~0.3–0.9 m on fast objects) but must solve
  association itself.
- **Re-validate under a real channel / multi-UE (OAI + Sionna).** Over OAI the uplink term dominates and the
  term ranking in §3 inverts — the compression levers matter much more there.
- Scene **density** is deliberately out of scope here; it belongs to
  `../../../rl_agent/DENSITY_ADAPTIVE_KNOB_PLAN.md` (density affects payload via ROI/entropy coding, so its
  home is knob selection, not staleness).
