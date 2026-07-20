# SceneSense UE Agent — Requirements & RL Design (living doc)

Status: LIVING DESIGN, reconciled 2026-07-17. Precursor: SCAN-AI (MobiCom 2026,
`abiodun/SCAN_AI_03_13_26_2.pdf`) — we follow its analysis→requirement→RL arc, one abstraction layer up.
The 2026-07-14 supervisor request is now folded in and its first two staleness analyses are complete.

## Current evidence boundary

- **Latency/speed requirement measured:** model floor is ~1.1 m; at 267 ms a ~32 mph target reaches
  4.36 m error, while AE-128's 105 ms operating point is about 2.0 m. See
  `../staleness/STALENESS_RESULTS.md` Result 1.
- **FPS/map freshness measured:** worst-case held-map age is `1/FPS`; at ~32 mph, zero-network-latency
  error falls from 15.02 m at 1 FPS to 1.52 m at 20 FPS. Stage-1 total lag is `Y_up + 1/FPS`. See Result 2.
- **Road-state context measured:** the speed/latency trend persists on straight roads and at intersections;
  intersections are worse in the aggregate at 267 ms. The curve subset is speed-confounded and does not
  support a curvature-effect claim.
- **Network knobs partially measured:** in single-UE RFsim, TDD 7:2→4:5 and 5QI 9→1 barely changed
  transport; compression remains the effective lever. Multi-UE/impaired-channel tests remain open.
- **Round-trip/downlink not yet measured:** current constraints describe capture→edge-map freshness. The next
  instrumentation pass must log result-return latency/payload and then extend the controller budget to include
  downlink and spatial-map sharing/operation latency.
- **Reliability under high FPS is not yet measured:** requested FPS must be checked against buffer queueing,
  drops, stale-result age, delivered FPS, and fresh-delivered FPS before training a learned policy.
- **Controller not implemented:** this document specifies the MDP; no trace-driven scorer, guardrail
  runtime, LinUCB/DQN policy, or online action loop exists yet.

## 0. Objective (why the UE agent exists)
The UE (vehicle) runs split-inference perception and streams **features** to the edge, which builds a shared
**spatial map**. The agent's job: deliver features **fast enough and accurate enough to keep the map fresh**
under the current scene dynamics and network state, by adapting compression (AE / quant / ROI) — and,
optionally, requesting network-side config (TDD / PRB / 5QI). Goal = a fresh, accurate map, not raw per-frame
accuracy. This differs from SCAN-AI (video → a human teleoperator); here the consumer is a *perception/map*.

## 1. Pipeline & the staleness timeline
Frame captured at time **X**; usable in the edge map at **X + Y_up**:
```
Y_up = front_ms (UE: backbone + AE-encode + quant + entropy + fragment)
     + uplink_transport (5G UL, the bottleneck)
     + back_ms (edge: entropy-decode + AE-decode + heads)
     + map_insert/render (fuse into spatial map)
```
Measured today (single-UE OAI, no impairment): no-AE u8 Y≈267 ms, AE-128 u8 ≈152 ms, AE-128 u4 ≈105 ms
(capture→map estimates; corresponding measured stream RTTs are 209/118/77 ms in `OAI_AB_RESULTS.md`;
front/transport/back breakdown is in `oai_latency_breakdown.pdf`; AE compute itself ≈0.27 ms, negligible).

For the full cooperative use case, a recipient vehicle acts after the map/result is returned:
```
L_total = Y_up + 1/FPS + Y_down + Y_map_share
```
where `Y_down` is edge result/map return to the ego/recipient vehicle and `Y_map_share` is the spatial-map
selection/fusion/sharing operation. These terms are the next logging target; until then the measured requirement is
only Stage 1 (`Y_up + 1/FPS`).

## 2. Requirements — the two coupled constraints
**Unified staleness model.** A tracked object at speed v: the map's newest info is between `Y_up` and
`Y_up + 1/FPS` old, so worst-case positional error
> **error ≈ v · (Y_up + 1/FPS) ≤ ε**    (Stage 1, capture→edge map)

For cooperative return/use:

> **error ≈ v · (Y_up + 1/FPS + Y_down + Y_map_share) ≤ ε**

This couples the two requirements the team asked about:
- **Latency Y_up** — capture→edge-map freshness (compression + config reduce it).
- **Downlink/map-sharing latency** — result/map return and downstream spatial-map operation; not measured yet.
- **Capture rate FPS** — how often we sample; capped by the transport bottleneck (payload × FPS ≤ uplink capacity).
- Trade-off: for given v, ε → need `Y_up + 1/FPS ≤ ε/v` for Stage 1, and the full `L_total ≤ ε/v` for cooperative
  return/use. Lower `Y_up` buys FPS/downlink headroom and vice-versa; all terms must be small for fast objects.

**Speed / environment dependence.** ε/v shrinks with speed, so the requirement is environment-specific
(examples at ε=0.5 m — illustrative, to be replaced by the derived value):
| environment | ~speed | budget `Y_up+1/FPS` (≤ε/v) | note |
|---|--:|--:|---|
| dense intersection | ~8 m/s | ~62 ms | slow but complex + cross-traffic (high FPS for coverage) |
| urban / curve | ~14 m/s | ~36 ms | |
| expressway | ~30 m/s | ~17 ms | fast but visually simple (cheap payload) |
*(FPS target separately: FPS ≥ v / map_resolution — e.g. 14 m/s, 1 m res → 14 FPS; 0.5 m → 28 FPS.)*

**Defining ε (NOT assumed — derived + anchored).**
1. **Standards anchors (CONFIRMED — see `STANDARDS_ANCHORS.md`, verified primary sources 2026-07-14):**
   - Latency: **~100 ms E2E** for cooperative perception / sensor sharing at low/mid automation (3GPP TS 22.186
     Advanced-Driving R.5.3-003 + NOTE 1, Extended-Sensors R.5.4-001; 5GAA ToD 100 ms uplink); **3–10 ms** at
     high automation / imminent collision (different regime — direct sidelink, not edge-relay).
   - Reliability: **~99%**. Update rate: **~10 Hz** (de-facto AD perception; ETSI CPM check ≤10 Hz).
   - Positional tolerance: **lane-level ~0.5 m** de-facto (low-confidence as a hard standard → we derive ours).
   - **Our configs vs anchor:** no-AE (267 ms/75%) FAILS; AE-128 u4 (105 ms/99%) MEETS the ~100 ms/99% budget.
     → compression moves us from noncompliant to standards-compliant for cooperative perception / ToD.
2. **Empirical derivation (completed 2026-07-16 — Analysis #1 below):** measured localization
   error vs latency Y and FPS per object type (car 30/20 mph, pedestrian), find the knee where staleness error
   exceeds tolerance. The model floor is ~1.1 m, so a 0.5 m envelope is not achievable by transport tuning;
   speed-conditioned latency ceilings for ε ∈ {1.5, 2.0, 2.5, 3.0 m} are reported in
   `../staleness/STALENESS_RESULTS.md` and cross-checked against the ~100 ms / ~10 Hz anchors.

**Per-config verdict (the deliverable):** each (OAI config × compression config) yields a measured
(sustainable FPS, `Y_up`, delivery). We then state, per config, whether it **meets** {FPS≥F, `Y_up≤Y_budget`} for
each environment tier, and later whether it meets the full `L_total` budget after downlink/map-sharing logging.
E.g. default OAI + no-AE fails (`Y_up=267 ms`, ~1–3 FPS sustainable, 75% delivery); AE-128 u4
(`Y_up=105 ms`, 99% delivery) meets the broad cooperative-perception latency/reliability anchor but does not meet
every fast-target positional envelope once downlink headroom is included. TDD/5QI tuning did not materially improve
the single-UE baseline.

## 3. RL design (MDP), mirroring SCAN-AI
**State** s = three blocks (SCAN-AI structure + our extensions):
- **Scene complexity** — ITU-T P.910 Spatial Info (Sobel edge density) + Temporal Info (frame diff), plus our
  objectness-derived features (object count, foreground fraction). Drives payload AND task difficulty.
- **Vehicle dynamics** — speed, longitudinal accel aₓ, lateral accel a_y (turning), road-type/speed-limit.
  Used TWICE: (a) leading indicator of payload surges (SCAN-AI), (b) sets the freshness requirement (our extension).
- **Network state** — allocated bandwidth B, SNR, delivery/loss, MAC buffer occupancy, current TDD/PRB/5QI capability,
  downlink result-return age, and fresh-delivered FPS.

**Action** a (discrete — resident models, per-frame routing):
- UE compression: AE ∈ {none, 32, 64, 128} × quant ∈ {u8, u6, u4} × ROI ∈ {0, 0.3, 0.5} (entropy fixed = zlib).
- (Optional, slower) network-side request: TDD pattern, PRB/BW, 5QI. → two-sided / two-timescale agent
  (slow: model switch + network config; fast: quant/ROI; per-frame safety guardrail).

**Reward** r = task-utility(map freshness × accuracy) − λ·delivery-loss − μ·switching-cost, **subject to** the
staleness constraint `v·(Y_up+1/FPS) ≤ ε` for Stage 1 and
`v·(Y_up+1/FPS+Y_down+Y_map_share) ≤ ε` for cooperative return/use (hard penalty on violation). Freshness ties
reward to Y+FPS, not just per-frame accuracy — the whole point.

**Algorithm.** Discrete action set → contextual bandit (LinUCB) or DQN. (SCAN-AI used SAC for *continuous*
bitrate; ours is discrete by design.) Optionally borrow SCAN-AI's **FiLM** idea: network state *gates* which
compression actions are feasible, rather than being a peer feature.

## 4. Analysis roadmap (SCAN-AI §3.3 analogue — do BEFORE finalizing RL)
1. **Staleness → localization error — COMPLETE (supervisor-specified 2026-07-14).**
   Per scenario (single **car @ 30 mph**, **car @ 20 mph**, **pedestrian @ walking ~1.4 m/s**), run the
   perception pipeline and, for each detection, record: `t_capture` (CARLA frame timestamp of the frame
   containing the object), `inferred_loc`, `t_inference` (when the location output is generated). Separately
   log the object's full CARLA **ground-truth trajectory** (loc per timestamp). Then compute the localization
   error as the supervisor defined it:
   - **staleness-inclusive:** `||inferred_loc − GT(t_inference)||` (what matters operationally)
   - **model-only floor:** `||inferred_loc − GT(t_capture)||`
   - **staleness component:** `≈ ||GT(t_capture) − GT(t_inference)|| ≈ v·Y_up`
   Latency `Y_up` = `t_inference − t_capture` (front + uplink + back). Sweep `Y_up` **post-hoc** (evaluate against
   `GT(t_capture+Y_up)` over a range of `Y_up`) for a smooth error-vs-latency curve; overlay real config points
   (105/152/267 ms). Sweep FPS by subsampling frames. Output per object type: error-vs-latency and
   error-vs-FPS curves → **quantified (ε, Y-budget, FPS) thresholds** = our benchmarking basis. Thresholds
   are object-type/speed-dependent (fast car tight, pedestrian loose) → feeds back into RL state.
   Results: `../staleness/STALENESS_RESULTS.md` Results 1, 1a–1b, and 2.
2. **OAI config sweep — PARTIAL, CONCLUSION ESTABLISHED FOR SINGLE UE.** TDD 7:2 vs 4:5 and 5QI 9 vs 1
   were measured on the fixed no-AE baseline. Both had little effect; extreme UL TDD is blocked by OAI
   K2/DCI constraints and wider PRBs require frequency/CORESET re-derivation. See
   `../oai_config_sweep/OAI_CONFIG_FINDINGS.md`. Revisit network actions under multi-UE contention or impairment.
3. **Scene → payload/accuracy/requirement** — SI/TI + dynamics across scene types (intersection/curve/expressway).
4. **Traffic characterization** — from `network_timeseries.csv`: UL-heavy (~200:1), bursty-periodic (per-frame
   burst of ~20 frags no-AE / ~3 AE, idle to next frame). Motivates the TDD/5QI choice.
5. **RL** — implement state/action/reward above; eval like SCAN-AI (generalization to unseen scenes/channels,
   zero/low loss vs static baselines, interpretability of decisions).

## 4a. Immediate action order before policy training

1. **Downlink latency/payload logging.** Repeat a small Experiment-1/2-style run and log edge tail done → result
   send → ego/recipient receive → display ready, with exact downlink payload bytes by result type.
2. **FPS × buffer reliability.** Sweep FPS and buffer policy/size to measure generated, queued, sent, edge-received,
   tail-completed, downlink-received, displayed, stale, and dropped frames. Report delivered FPS and
   fresh-delivered FPS separately.
3. **Experiment-3/FOV continuation.** Keep this independent of network guardrails. Use the natural-scene post-hoc
   FOV split as current evidence; only run a controlled lateral sweep if the centered training-like scene restores
   expected ~0.9–1.2 m localization under a frozen target-selection rule.
4. **Sionna/ray-traced channel realism.** After the logging schema is stable, add position-dependent channel traces
   and replay the same controller inputs under realistic latency/retransmission/delivery variation.

## 5. Open decisions / to-fold-in

- **Choose the deployment ε tier.** The experiment now provides speed-conditioned ceilings for 1.5/2.0/2.5/3.0 m;
  a 0.5 m target is below the current model floor and requires a better localizer, not lower latency.
- Discrete vs any-continuous knob; scope of the network-side agent (do we actually let the UE request TDD/5QI,
  or is that a separate network agent? — Subhramoy's end-to-end EV-agent × network-agent workflow).
- **FOV-position diagnostic:** natural-scene post-hoc split now supports a range-aware edge-risk term. A controlled
  lateral sweep is optional and should only continue after centered-baseline parity in a training-like scene.
- **Controller implementation:** build trace schema/scorer, hard task guardrails, static/replay baselines,
  then LinUCB before considering DQN.
