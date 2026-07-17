# RL agent — benchmark thresholds & guardrails (from Experiments 1 & 2)

Citable constraint spec for the network-aware split-inference controller. Derived from the two completed
staleness experiments (`staleness/STALENESS_RESULTS.md`): in-domain moving car-height ego, RGB+radar fusion,
no-AE u8, 200k radar recipe, GT = actor origin. Localization error = ‖predicted world xy − true world xy‖.
Numbers are the operating guardrails; treat as design bounds, not physical constants.

## 0. The one-line model
Total localization error combines the model's own floor with the object's displacement during the effective lag:

> **error(v) ≈ √( floor² + (v · L)² )**,  where **L = Y + 1/FPS** (transport latency + map-staleness),
> **floor ≈ 1.1 m** (the model's own error, latency/FPS-independent).

Everything below is this inequality solved for each knob. `v` = tracked object world speed (m/s).

## 1. Hard floor — feasibility LOWER BOUND (Exp 1)
- **ε_floor ≈ 1.1 m.** No latency or FPS choice localizes better than this; it's model-limited (offline veh-loc
  0.88 m; live in-domain ~1.0 m).
- **Consequence for the agent:** an accuracy target **ε < ~1.1 m is infeasible** — do not reward chasing it, and
  flag "lane-level (~0.5 m)" as needing a *better model*, not a network action.
- Feasible staleness budget for a target ε: **B(ε) = √(ε² − 1.1²)** → ε=1.5→1.02 m, ε=2.0→1.67 m, ε=2.5→2.24 m, ε=3.0→2.79 m.

## 2. Latency UPPER BOUND per speed (Exp 1) — max end-to-end Y (ms) to hold error ≤ ε
| object speed | ε ≤ 1.5 m | ε ≤ 2.0 m | ε ≤ 2.5 m | ε ≤ 3.0 m |
|---|--:|--:|--:|--:|
| pedestrian / ≤6 mph | no limit | no limit | no limit | no limit |
| ~10 mph | 187 | >269 | >269 | >269 |
| ~14–18 mph | ~125 | ~215 | >269 | >269 |
| ~28 mph | 112 | 166 | 212 | 255 |
| ~32 mph | **45** | **98** | 137 | 173 |
Closed form (FPS high): **Y_max(v,ε) = B(ε)/v**. Latency Y = capture→inference (front+uplink+back), NOT downlink.

## 3. FPS LOWER BOUND per speed (Exp 2) — min capture/update rate at ~0 network latency
Map holds the last detection between frames → worst-case staleness 1/FPS → error ≈ v·(1/FPS).
| object speed | ε ≤ 1.5 m | ε ≤ 2.0 m |
|---|--:|--:|
| ≤10 mph | ~5 | ~3 |
| ~18 mph | ~10 | ~5 |
| ~28 mph | ~15 | ~10 |
| ~32 mph | **~20** | **~10** |
Closed form: **FPS_min(v,ε) = v / (B(ε) − v·Y)** (worst-case; use B/√ with 1/(2·FPS) for average query timing).
Per-frame accuracy is FPS-independent (model is FPS-robust) — FPS only buys *freshness*.

## 4. MASTER constraint (the agent must satisfy this)
> **v · (Y + 1/FPS) ≤ √(ε² − 1.1²)**
The agent controls **Y** (via compression: AE bottleneck / quant / ROI → payload → latency) and can request **FPS**.
Both must be tight for fast objects; for slow objects both can be relaxed to save bandwidth.

**Discrete latency actions (operating-point anchors, from the OAI A/B):**
| action | Y (ms) | meets ε=2 m up to | notes |
|---|--:|--:|---|
| loopback / ideal | ~50 | ~32 mph | best case |
| AE-128 compression | ~105 | ~28 mph | ~8× payload cut; meets ε=2 m for most speeds |
| no-AE (baseline) | ~267 | ~10 mph | fails anything >~18 mph at ε=2 m |

## 5. Policy insight — everything is SPEED-GATED
- **Slow / pedestrian (≤~10 mph):** latency- AND FPS-immune → agent can compress hard and drop FPS to save bandwidth.
- **Fast (≥~28 mph):** must spend bandwidth on low latency (≤~100 ms) AND high FPS (≥~15–20) — and still can't beat ~1.1 m.
- → **object speed (and its uncertainty) belongs in the agent STATE**, as the variable that sets the whole budget.
  Distance/FOV-position may add a secondary edge-risk term (Exp 3, still open — see NEXT_EXPERIMENTS.md).

## 6. Suggested reward / constraint shaping
- **Hard constraint:** reject/penalize any action whose predicted `v·(Y+1/FPS) > B(ε)` for the current tracked speed.
- **Cost side:** minimize bandwidth (payload × FPS) subject to the constraint → the agent naturally compresses
  more and lowers FPS for slow objects, and spends for fast ones.
- **Do not** target ε < 1.1 m (infeasible), and **do not** treat FPS as improving per-frame accuracy (it doesn't).

## Provenance & caveats
- Exp 1 (`speed_error_requirement.pdf`) and Exp 2 (`fps_mapStaleness_worstcase.pdf`) — see `staleness/STALENESS_RESULTS.md`.
- In-domain, no channel impairment (rfsim), single UE, Town10; speeds walk→32 mph (28–32 via NPC over-speed).
- Latency is swept analytically on live captures (staleness measured from real GT motion; no real transport delay injected).
- Floor ~1.1 m is a same-distribution estimate; fresh-scene ~+0.2 m. Config levers (TDD/5QI) don't move transport
  in single-UE clear-channel (`oai_config_sweep/OAI_CONFIG_FINDINGS.md`); compression is the effective latency lever.
- These are BOUNDS for the controller, to be re-validated under a realistic channel / multi-UE (SIONA-RT phase).
