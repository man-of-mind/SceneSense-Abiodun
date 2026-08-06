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
| ideal loopback / 8 MB buffers (**zstd, DEPLOYED**) | no-AE u8 capture→result ~45 ms (transport ~7 ms, front ~30 ms); downlink ~5 ms | higher | **zstd is the deployed codec (2026-07-22).** Train the agent on `PERMODEL_KNOB_MATRIX_ZSTD.md`. |
| ideal loopback / 8 MB buffers (zlib, legacy) | same no-AE u8 payload ~88–90 ms (transport ~31 ms, front ~46 ms) | ~32 mph | pre-2026-07-22 codec; ~4× slower (de)compress at large payloads, **same accuracy**, payload ~±5% (`CODEC_LATENCY_AB.md`) |

> **Deployed codec = zstd (2026-07-22).** Entropy coding is lossless, so accuracy is codec-invariant (offline exact
> profile identical; live drivable A/B confirms). zstd cuts front+transport ~2–4× vs zlib and improves OAI delivery
> (72→84%) at no accuracy cost. Use **`PERMODEL_KNOB_MATRIX_ZSTD.md`** as the action-cost model. zstd is a free
> baseline win, NOT a fix for the OAI uplink bottleneck (delivery still ~84%, uplink handling ~151 ms for the ~1 MB
> no-AE burst) — the AE/quant/ROI payload knobs are still what the agent needs for reliable low-latency delivery.
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

---

## 8. Scene density + the segmentation constraint (2026-07-31, seg-inclusive re-run)

**One-line policy note (SUPERSEDES the earlier detection-only note):** *If the shared map carries
segmentation (drivable surface / lane / dense semantics), scene density is NOT a useful knob-selection
state variable and the ROI-drop knob must not be used for compression. The seg-safe operating point is
**ROI 0 + AE bottleneck + u4**, ≈90 KB, and it is **density-invariant** — the same knob is optimal from 1 to
5+ objects in view. Compression comes from the AE bottleneck and quant bits, not ROI and not density.*

**Why the flip.** The first density run scored **detection only** and found a density-adaptive ROI policy
(q 0.98→0.7, 6.8→43.7 KB, ~60% saving). Re-scoring with **segmentation** (mIoU + per-class IoU, gate G7
reproduces the matrix exactly) shows that policy was destroying seg: the ROI gate keeps only high-objectness
cells, so raising q is nearly free for object recall but **collapses the dense seg between objects** —
vehicle IoU **0.92 → 0.11** as q 0 → 0.98, at every density (`density_knob/DENSITY_KNOB_RESULTS.md` §2/§3).

**Seg-aware knob (the deliverable), joint detection+seg accept:**

| in-view objects (≤40 m) | knob | payload | uplink ms | in-view recall | loc MAE | mIoU | veh IoU |
|---|---|--:|--:|--:|--:|--:|--:|
| 1–2 | `ae32 / u4 / ROI 0` | 90.0 KB | ~39 (measured) | 0.933 | 0.75 m | 0.812 | 0.918 |
| 3–4 | `ae32 / u4 / ROI 0` | 89.5 KB | ~39 (measured) | 0.889 | 0.92 m | 0.822 | 0.932 |
| 5+  | `ae32 / u4 / ROI 0` | 89.3 KB | ~39 (measured) | 0.856 | 1.08 m | 0.848 | 0.896 |
| 0 (empty) | ROI drop tolerable* | — | — | n/a | n/a | * degenerate | * |

`*` empty-bin seg is degenerate (no in-view objects ⇒ noisy veh/person IoU; drivable-surface IoU ~0.99 at
all q). Safe default: treat empty as sparse (ROI 0) unless the map does not consume seg. Under the stricter
global ped-recall gate the matrix uses, the seg-aware pick tightens to `ae128/u4/ROI0` (129 KB) — still
ROI 0, still density-invariant.

- **Object-only exception:** if a deployment consumes ONLY object detections (no seg layer), the earlier
  detection-only density-adaptive policy applies (ROI q 0.98/0.9/0.9/0.7, AE 32/32/64/64, u4; 6.8→43.7 KB,
  ~60% saving) — but label it "object map only; segmentation not preserved" and carry the empty-bin FP
  caveat.
- **u4 at every density; no-AE Pareto-dominated everywhere** (0 of 72 no-AE profiles accepted). The AE
  *improves* accuracy while shrinking payload — it is not a compression concession.
- **State-variable consequence:** object **speed** stays in the agent state (latency/FPS budget); **density
  can be dropped** from the knob-selection state under the seg-aware policy (the knob is flat). The
  observability caveat (agent sees only a lagged density proxy) is therefore moot for knob selection when
  seg is on the map — a further reason to prefer the flat policy.
- Caveats: ideal loopback / uplink-only, in-domain Town10; bin 5+ n=135 (±2.5 pts recall); density
  correlates with proximity (20.0 m sparse → 12.2 m dense), not a pure density effect. Latency is
  **measured across the whole ROI range** (`loopback_latency_zstd.json`, 48 profiles incl. high-ROI
  sweep 2026-07-31): front ~25 ms flat, transport 1.3–4.1 ms, delivery 1.00; no policy pick depended on it.
  Full analysis: `density_knob/DENSITY_KNOB_RESULTS.md` (9/9 gates pass, incl. G7 seg).

---

## 9. LOCKED RL design synthesis (2026-07-31)

One sentence: **object speed sets the freshness budget; the channel sets the affordable payload; the knobs
spend payload to buy accuracy within that budget; scene-emptiness is a send-gate, not a knob.** This is the
distillation of every measured result above; treat it as the design spec for the SAC/PPO controller.

### 9.1 STATE
| variable | why | source / status |
|---|---|---|
| **object speed** (+ uncertainty) | dominant; sets the whole latency/FPS budget via the master inequality | §1–§5, MEASURED |
| **channel state** — CQI/SNR→achievable rate, UE buffer occupancy | sets affordable payload + delivery reliability; the binding constraint over OAI | **MEASURED (2026-08-04, clean 12-cell grid on fresh CARLA)** (`channel_condition_sweep/CHANNEL_SWEEP_RESULTS.md` + plots). Uplink-only, SINR, retx=0 everywhere; sharp payload-ordered knee (offered ~6 fps): **1 MB** survives only clear (97.5%), collapses at ≤19.5 dB (22%→4.6%, 6–15 s); **400 KB** holds to 15.6 dB (100%, ≤251 ms), collapses at 8.2 dB (31.5%); **90 KB (ae32/u4/ROI0 seg-safe floor) = 100% at EVERY rung, ≤175 ms.** Collapse = congestion (BSR pins at the ~48 MiB ceiling), not radio errors. Rule: `payload_budget(SNR)=capacity(SNR)/target_fps × margin`; budget @10 fps ≈ {clear 448, mild 339, mid 241, **strong 127**} KB → **the 90 KB floor fits everywhere; at ~8 dB even 400 KB doesn't fit (ROI-escalation region)**. CAVEAT: offered fps was ~6 (live-front, CARLA-render limited); capacity estimated from delivered ceilings (±~30%); a shaped-burst @10 fps re-run (Mode A, no CARLA) will pin the absolute knee — not blocking. Fast objects (32 mph) still need FPS ≥15 for the 2.0 m staleness budget. |
| **scene-empty gate** — max/count objectness on the CURRENT frame (pre-transmit) | decides send / skip; computed by the front backbone before compression, so NOT lagged | §8 (density-seg), available on the UE |
| **previous action + outcome** — last payload/FPS, last latency/delivery | channel telemetry is lagged, so the agent needs its last decision + result to act sensibly (POMDP) | added 2026-08-04 (POLICY_KICKOFF + state diagram) |
| **age-of-information (AoI)** — time since the last SUCCESSFUL (delivered) map update | true map staleness after skips/drops ≠ L+1/FPS; drives the composed loc-error and makes send/skip sequential | added 2026-08-05 (`collab/REVIEW_NOTES.md` item 17) |
| **estimated achievable UL capacity (+ confidence)** — from MCS × PRB/TBS × grant rate (load-independent), NOT raw scheduled throughput | the C1 mask + policy budget key off this lagged/noisy estimate, never the sim's true SNR; raw throughput under light load ≠ capacity (censored-observation trap) | added 2026-08-05 (`REVIEW_NOTES.md` items 15/A, 21) |
| ~~scene density (graded)~~ | **dropped** — the seg-aware knob is density-invariant | §8 |

> **Authoritative current spec:** `rl_agent/POLICY_KICKOFF.md` + the MDP state diagram supersede this table
> where they differ — they add **previous-action+outcome** (above) and make **send/skip** an explicit action.
> Constraints C1–C4 and reward cautions: see `collab/REVIEW_NOTES.md` (2026-08-05).

### 9.2 ACTION — payload levers, in cost order (cheapest first)
1. **Quant bits u8→u4** — nearly free (seg-lossless at ROI 0), ~2.0–2.4× payload cut. Use first.
2. **AE bottleneck** (none→128→64→32) — *improves* accuracy while compressing; no-AE is Pareto-dominated.
   The seg-safe payload floor with current models is **`ae32/u4/ROI0` ≈ 90 KB**.
3. **FPS** — buys freshness only (per-frame accuracy is FPS-independent); relax for slow objects.
4. **ROI drop q** — the big payload lever BUT it **destroys segmentation** (veh IoU 0.92→0.11). Two legitimate
   uses only: (a) **send-gate** — q→1 / skip when the dynamic scene is empty (the static seg layer is
   already mapped, so no new information is lost); (b) **channel-pressure escalation of LAST resort** — when
   a bad channel cannot deliver the ~90 KB seg-safe floor in the freshness budget AND the object is too fast
   to drop FPS, trade seg quality for delivery (q 0.3/0.5), paying the measured mIoU penalty explicitly. It
   is NOT a routine or density-indexed knob.

### 9.3 REWARD
- **Hard constraint:** `v·(L+1/FPS) ≤ √(ε²−1.1²)` with **L measured over OAI** (not loopback). Penalize
  violation; do not reward chasing ε < ~1.1 m (model floor, §1).
- **Accuracy term:** joint — detection recall + loc AND seg mIoU held (the §8 joint-accept). ROI's mIoU cost
  enters here, which is what makes lever 4 self-limiting.
- **Cost term:** minimize **airtime / PRB occupancy** (payload × FPS × retransmissions), NOT raw bytes —
  and this is where compression/gating earns its keep, most of all under a **bad channel** and under
  **multi-UE contention** (the scarce shared resource). On a single clean channel with the budget already
  met, there is little reason to compress; the value shows up exactly in the conditions the channel sweep
  (and eventually multi-UE) will create.
- **Reliability:** reward **fresh-delivered** frames, not sent frames.

### 9.4 What is transport-invariant vs must be re-measured over OAI
- **Invariant (reuse the loopback/offline knob matrix as-is):** accuracy-vs-knob (recall/loc/mIoU) and
  payload-vs-knob. Lossless codec ⇒ the entire seg/density accuracy story holds byte-for-byte over OAI.
- **Transport-dependent (the channel sweep must supply):** payload→latency→**delivery**. Loopback is linear
  and ~free; OAI is ~14× steeper, nonlinear near capacity, with a delivery cliff (75→99%). This is the
  reward's constraint + cost terms. → `channel_condition_sweep/CHANNEL_SWEEP_PLAN.md`.
