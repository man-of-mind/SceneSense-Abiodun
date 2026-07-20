# RL agent — benchmark thresholds & guardrails

Citable constraint spec for the network-aware split-inference controller. Derived from the two completed
staleness experiments (`staleness/STALENESS_RESULTS.md`): in-domain moving car-height ego, RGB+radar fusion,
no-AE u8, 200k radar recipe, GT = actor origin. Localization error = ‖predicted world xy − true world xy‖.
Numbers are the operating guardrails; treat as design bounds, not physical constants.

Updated 2026-07-17 with the meeting decision to extend the current uplink/capture-to-map model into a full
round-trip freshness budget. The measured constraints below are still Stage 1 evidence; downlink/result-return and
spatial-map sharing latency must now be logged before training an online policy.

## 0. The one-line model
Total localization error combines the model's own floor with the object's displacement during the effective lag.
For the current measured Stage-1 pipeline:

> **error(v) ≈ √( floor² + (v · L)² )**,  where **L = Y_up + 1/FPS** (Stage-1 transport + map-staleness),
> **floor ≈ 1.1 m** (the model's own error, latency/FPS-independent).

For the full cooperative-perception use case, extend the lag to:

> **L_total = Y_up + 1/FPS + Y_down + Y_map_share**

where `Y_up` is capture/front/uplink/edge-tail/map-insert age, `Y_down` is result/map return to the recipient
vehicle, and `Y_map_share` is any spatial-map selection/fusion/sharing operation. `Y_down` and `Y_map_share` are not
measured yet; use fixed placeholders only for planning until the next logging pass records them.

Everything below is this inequality solved for each knob. `v` = tracked object world speed (m/s).

## 1. Hard floor — feasibility LOWER BOUND (Exp 1)
- **ε_floor ≈ 1.1 m.** No latency or FPS choice localizes better than this; it's model-limited (offline veh-loc
  0.88 m; live in-domain ~1.0 m).
- **Consequence for the agent:** an accuracy target **ε < ~1.1 m is infeasible** — do not reward chasing it, and
  flag "lane-level (~0.5 m)" as needing a *better model*, not a network action.
- Feasible staleness budget for a target ε: **B(ε) = √(ε² − 1.1²)** → ε=1.5→1.02 m, ε=2.0→1.67 m, ε=2.5→2.24 m, ε=3.0→2.79 m.

## 2. Latency UPPER BOUND per speed (Exp 1) — max Stage-1 Y_up (ms) to hold error ≤ ε
| object speed | ε ≤ 1.5 m | ε ≤ 2.0 m | ε ≤ 2.5 m | ε ≤ 3.0 m |
|---|--:|--:|--:|--:|
| pedestrian / ≤6 mph | no limit | no limit | no limit | no limit |
| ~10 mph | 187 | >269 | >269 | >269 |
| ~14–18 mph | ~125 | ~215 | >269 | >269 |
| ~28 mph | 112 | 166 | 212 | 255 |
| ~32 mph | **45** | **98** | 137 | 173 |
Closed form (FPS high): **Y_max(v,ε) = B(ε)/v**. In the completed experiments, latency `Y_up` = capture→edge
inference/map update (front + uplink + back), NOT downlink/result return.

## 3. FPS LOWER BOUND per speed (Exp 2) — min capture/update rate at ~0 network latency
Map holds the last detection between frames → worst-case staleness 1/FPS → error ≈ v·(1/FPS).
| object speed | ε ≤ 1.5 m | ε ≤ 2.0 m |
|---|--:|--:|
| ≤10 mph | ~5 | ~3 |
| ~18 mph | ~10 | ~5 |
| ~28 mph | ~15 | ~10 |
| ~32 mph | **~20** | **~10** |
Closed form: **FPS_min(v,ε) = v / (B(ε) − v·Y_up)** (worst-case; use B/√ with 1/(2·FPS) for average query timing).
Per-frame accuracy is FPS-independent (model is FPS-robust) — FPS only buys *freshness*.

## 4. MASTER constraint (the agent must satisfy this)
> **v · (Y_up + 1/FPS) ≤ √(ε² − 1.1²)**
The agent controls **Y_up** (via compression: AE bottleneck / quant / ROI → payload → latency) and can request **FPS**.
Both must be tight for fast objects; for slow objects both can be relaxed to save bandwidth.

For the next round-trip version, replace `Y` with the measured uplink/downlink split:

> **v · (Y_up + 1/FPS + Y_down + Y_map_share) ≤ √(ε² − 1.1²)**

Use this as the design guardrail for cooperative spatial-map sharing. Until `Y_down` and `Y_map_share` are measured,
do not claim that a profile satisfies the final end-to-end budget; claim only that it satisfies the Stage-1
capture-to-map budget.

**Discrete latency anchors (keep transport labels explicit):**
| condition / action | latency anchor | meets ε=2 m up to | notes |
|---|---:|--:|---|
| ideal loopback / 8 MB buffers (**zlib**) | no-AE u8/zlib capture→result ~88–90 ms; post-send RTT ~43 ms; result downlink ~5 ms | ~32 mph | clean local floor for the **deployed zlib** codec |
| ideal loopback / 8 MB buffers (**zstd**) | same no-AE u8 payload ~45 ms capture→result (transport ~7 ms, front ~30 ms) | higher | **codec is a latency lever**: zstd ~4× faster (de)compress, same accuracy, payload ~±5%; live A/B confirmed (`CODEC_LATENCY_AB.md`) |
| bounded-buffer loopback | not a clean latency anchor; no-AE 200k calibration delivered 1/100 frames | n/a | use only as buffer/reliability artifact |
| AE-128 compression over OAI | ~105 ms | ~28 mph | ~8× payload cut; meets ε=2 m for most speeds |
| no-AE baseline over default OAI | ~267 ms | ~10 mph | fails anything >~18 mph at ε=2 m |

## 5. Policy insight — everything is SPEED-GATED
- **Slow / pedestrian (≤~10 mph):** latency- AND FPS-immune → agent can compress hard and drop FPS to save bandwidth.
- **Fast (≥~28 mph):** must spend bandwidth on low latency (≤~100 ms) AND high FPS (≥~15–20) — and still can't beat ~1.1 m.
- → **object speed (and its uncertainty) belongs in the agent STATE**, as the variable that sets the whole budget.
  Distance/FOV-position adds a secondary risk term: the natural-scene post-hoc FOV split shows edge position mainly
  lowers match availability and medium/far localization, so use range-aware edge risk rather than a blanket
  center-FOV prior.

**Reliability is part of freshness.** High requested FPS is not useful if the UE/front buffer queues stale frames or
the edge/downlink drops results. The next reliability experiment should measure generated FPS, delivered FPS, fresh
delivered FPS, queue wait, drops, and result age as a function of FPS × buffer size × payload profile.

## 6. Suggested reward / constraint shaping
- **Hard constraint:** reject/penalize any action whose predicted `v·(Y_up+1/FPS) > B(ε)` for the current tracked speed;
  after downlink instrumentation, use `v·(Y_up+1/FPS+Y_down+Y_map_share) > B(ε)`.
- **Cost side:** minimize bandwidth (payload × FPS) subject to the constraint → the agent naturally compresses
  more and lowers FPS for slow objects, and spends for fast ones.
- **Do not** target ε < 1.1 m (infeasible), and **do not** treat FPS as improving per-frame accuracy (it doesn't).

## 7. Immediate experiment order before agent training

1. **Downlink latency/payload logging:** repeat a small Experiment-1/2-style run and log edge result-ready →
   ego/recipient receive/display timestamps plus exact result payload bytes.
2. **FPS/buffer/reliability sweep:** measure delivered and fresh-delivered FPS under FPS × buffer size × payload
   profile, including queue wait, drops, timeout/no-result rate, and stale-result age.
3. **Experiment-3/FOV continuation:** keep it separate from the network guardrail work; use the natural-scene
   post-hoc FOV split as valid evidence unless a controlled training-like scene passes centered-baseline parity.
4. **Sionna/channel realism:** only after the logging schema is stable, feed ray-traced channel state into the same
   per-frame traces so latency, retransmission, delivery, and buffer behavior can be attributed correctly.

## Provenance & caveats
- Exp 1 (`speed_error_requirement.pdf`) and Exp 2 (`fps_mapStaleness_worstcase.pdf`) — see `staleness/STALENESS_RESULTS.md`.
- In-domain, no channel impairment (rfsim), single UE, Town10; speeds walk→32 mph (28–32 via NPC over-speed).
- Latency is swept analytically on live captures (staleness measured from real GT motion; no real transport delay injected).
- Floor ~1.1 m is a same-distribution estimate; fresh-scene ~+0.2 m. Config levers (TDD/5QI) don't move transport
  in single-UE clear-channel (`oai_config_sweep/OAI_CONFIG_FINDINGS.md`); compression is the effective latency lever.
- These are BOUNDS for the controller, to be re-validated under a realistic channel / multi-UE (Sionna-RT phase).
