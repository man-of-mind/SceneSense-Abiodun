# UE-Side RL Agent — Plan of Attack (Month 2 catch-up → Month 3)
*Original plan written 2026-07-08; current status overlaid 2026-07-16.*

> Sections below preserve the workstream rationale and design decisions from
> the original plan. This status overlay is authoritative when an older status
> statement conflicts with later evidence.

## Current status overlay (2026-07-16)

- **H — sweep orchestration: complete.** Static and per-model jobs are
  reproducible and aggregated.
- **A/D — action/model characterization: complete for the priority fusion
  route.** M-prime plus integrated AE-128/64/32/no-AE models were evaluated
  across quantization 8/6/4 and ROI 0/0.3/0.5. The authoritative 42-profile
  table is `rl_agent/PERMODEL_KNOB_MATRIX.md`.
- **AE resolution:** the early standalone AE collapsed object localization;
  integrated joint training recovered it. Integrated AE models preserve
  useful segmentation, recall, and localization and are valid resident actions.
- **F — network baseline: partially complete.** The OAI compression A/B and a
  limited TDD/5QI configuration study are complete. Controlled impairment,
  background load, and multi-UE contention are not.
- **Requirements work: complete for the current staleness question.** Live
  results now quantify object-speed/latency and held-map `1/FPS` error. Object
  dynamics and `Y + 1/FPS` belong in state/reward/guardrails.
- **B — offline controller harness: not implemented.** There is no executable
  trace join, action catalog, reward scorer, baseline replay, or LinUCB result.
- **C — controller guardrails: not implemented.** `gate_a_check.py` is a model
  acceptance gate; it is not the proposed runtime accept/clamp/reject layer.
- **E/G — per-tensor packet priority and Sionna coupling: open.** They must not
  displace controller closure.

Current critical path: **B → C → simple baselines → LinUCB → controlled network
stress**. The spatial-map work is no longer paused globally, but its next
research step is real-data freshness/occlusion/warning evaluation rather than
additional visualization.

---

## 0. North Star (from the proposal / monthly checklist)
> **Learn a network-aware split-inference control policy that reduces payload/latency while preserving task
> utility.** The UE-side agent picks an operating point per frame/window: **AE bottleneck · quantization ·
> ROI threshold · which tensors to prioritize/drop · frame send/skip · redundancy** — subject to
> **guardrails** (pedestrian/cyclist/small-object recall, mIoU, foreground IoU must not silently collapse).

**Month-2 exit criterion (proposal):** *"policy can train or evaluate against static policies using the same
logged metrics."* → i.e. **offline controller harness + static sweeps**, not online RL. Everything below
serves that.

---

## 1. How the meeting notes map to the work
Your scattered notes are all real and all fit one loop. Mapping them:

| Meeting note | Workstream | Status of infra |
|---|---|---|
| Train models in parallel, many experiments | **H** parallel sweep runner | training entrypoints exist; **no** multi-config runner |
| Try different **autoencoders** (have a list) | **D** real AE bottlenecks | only a *trivial 1×1-conv* AE stub, off by default, **no trainer** |
| **Quantization levels, ROI thresholds** → payload/latency/accuracy | **A** static sweeps | knobs **exist** (`per_tensor/per_channel_uint8/uint4`, `zlib/zstd/none`, objectness ROI) |
| Know **important parts of tensors**; prioritize or send nothing; **tag UDP header** → informed vs uncontrolled vs no loss | **E** tensor-importance + packet tagging | **not built** — features are one pickled+compressed blob; header has no tensor identity |
| Various **network/channel conditions** | **F** channel sweeps | OAI **rfsimulator** channel models exist (C); usable now |
| Moving into **OAI** (prelim done) | reuse | OAI transport + rich radio/tunnel/T-tracer metrics **exist** |
| **Sionna ray-tracing + OSM** channels | **G** Sionna integration | **nothing exists** — larger, Month-3 build |
| All → **RL state/action/reward** | **B** offline controller harness | schema drafted (`SCENESENSE_RL_SCHEMA.md`); **no** harness/reward/baselines yet |

---

## 2. The system we're building (UE-side control loop)
```
  per frame/window:
    STATE  = scene (density, foreground, vulnerable-object flag, confidence)
           + payload/latency history + NETWORK (UE bitrate, RTT, gNB MCS/RB/BLER/HARQ, grants)
        │
        ▼
    POLICY (agent)  ── guardrails clamp/reject unsafe actions ──►
        │
        ▼
    ACTION = { AE profile, quant level, ROI threshold, per-tensor priority/drop,
               send/skip, redundancy }
        │
        ▼
    TRANSPORT over channel (OAI 5G / rfsim channel model / later Sionna)
        │
        ▼
    MEASURE  payload bytes · front/back/RTT latency · loss/timeout · task utility (recall/IoU/loc err)
        │
        ▼
    REWARD = task_utility − payload − latency − loss/timeout − guardrail_penalty
```
Month 2 runs this **offline on logged traces** (replay). Month 3+ closes the loop online.

---

## 3. Workstreams (each: what to build · reuse · priority)

### A. Static UE-side sweeps over existing knobs  ·  **MUST (this week)**  ·  checklist §3
Map the action space we already have, on the trained fusion model, on the canonical scenes.
- **Build:** `scripts/run_static_sweep.sh` + `scripts/analyze_static_sweep.py` — loop over
  `quantization ∈ {per_tensor_uint8, per_channel_uint8, per_channel_uint4}` × `entropy ∈ {zlib, zstd, none}`
  × `roi_objectness_threshold ∈ {off, low, med, high}`, on the loopback split path, logging the existing
  per-frame CSV (payload, front/back/RTT, task metrics via `evaluate_fusion`/`analyze_*`).
- **Reuse:** knobs already in `carla_split_inference_udp_data_collect.py` (codecs :2261–2510, ROI :2753) +
  the fusion OAI client + `analyze_scenesense_app_metrics.py`.
- **Output:** payload–latency–utility **Pareto** per route; the "best fixed static policy" + a
  "lowest-byte unsafe policy" (motivates guardrails).

### B. Offline controller harness  ·  **MUST (this week)**  ·  checklist §5 — the core Month-2 deliverable
- **Build (`abiodun/controller/`):**
  1. `trace_join.py` — join per-frame app CSV + network sampler + gNB/T-tracer + scenario meta + task
     summaries by `run_group`/frame-window (extends `analyze_scenesense_app_metrics.py`).
  2. `action_catalog.py` — discrete action profiles (safe / balanced / low-byte / hazard-guarded), with
     **route masking** for unsupported knobs.
  3. `reward.py` — the schema reward (task_utility − payload − latency − timeout/loss − guardrail_penalty).
  4. `baselines.py` — send-everything, always-low-byte, best-fixed, network-only rule, task-only rule,
     one scene+network heuristic.
  5. `bandit.py` — contextual bandit (LinUCB) **or** DQN over the discrete profiles (per schema: no SAC yet).
- **Reuse:** `SCENESENSE_RL_SCHEMA.md` (state/action/reward), `scenesense_tx_gate.py` (already reads a
  controller-written JSON — we become the writer).
- **Output:** replay CSV per policy + a "task-utility vs bytes vs latency/timeout" comparison → **the best
  heuristic the learned policy must beat.** *(This hits the Month-2 exit criterion.)*

### C. Guardrail thresholds in config  ·  **MUST (this week)**  ·  checklist §6
- **Build:** `controller/guardrails.yaml` (route task floors, vulnerable-object rules, network-fallback
  rules) + guardrail pass in the harness that reports **accepted / clamped / rejected** per action.

### D. Real autoencoder bottlenecks  ·  **SHOULD (start this week → early M3)**
**LOCKED list: bottleneck channels {128, 64, 32}, trained TASK-AWARE** (minimize downstream task loss, not
just reconstruction).
**Head-start found:** `abiodun/checkpoints/rd_ae_b128.pt` (supervisor's) = a working 128 AE with three parts:
`encoder` 1×1 conv 256→128, `decoder` 1×1 conv 128→256, and an **`importance_head` (128→64→128)** — i.e. the
AE and the tensor-importance mechanism (workstream E) in ONE module. **Caveats:** (a) the matching class is
NOT in the repo — current `FeatureAutoencoder`/`PerLevelFeatureAutoencoder` have encoder+decoder only, no
importance head — so we reconstruct the module from the checkpoint shapes; (b) **no trainer in-repo** → we
build it; (c) it's **256-channel (camera-OD/FPN path)**, so we need a **fusion-path variant** (compresses the
~960-ch `high` level; optionally the 40-ch `low`).
- **Build:** `feature_ae/` — reconstruct the AE-with-importance module (matches rd_ae_b128), a **task-aware
  trainer** (frozen perception model → encoder → [quant/drop] → decoder → task loss + recon loss +
  importance supervision), train {128 (fine-tune from rd_ae_b128), 64, 32} for the fusion path.
- **Reuse:** `--ae-mode/--ae-checkpoint` hook (`...data_collect.py:2599`) is the runtime drop-in point.
- **Output:** AE profiles → action-space entries; feed the A-sweeps.

### E. Tensor-importance + packet tagging (informed loss)  ·  **SHOULD (start this week → early M3)**
This is the meeting's key novelty: know which tensors matter, tag packets, drop the unimportant first.
**Head-start:** the `importance_head` in `rd_ae_b128` already scores per-(bottleneck-channel) importance — the
mechanism for "which tensors matter" — so E rides on the same module as D.
- **Build:**
  1. **Per-tensor framing:** restructure the UDP sender so each feature level (fusion: `low`~40ch,
     `high`~960ch; OD: FPN levels) is serialized/chunked **separately**, and add a **tensor tag** to the
     packet header (`HEADER_STRUCT` today = `message_id, chunk_index, total_chunks` → add
     `tensor_id, importance_class`). Receiver reconstructs with missing tensors zero-filled (graceful).
  2. **Importance ranking:** ablation study — zero each level / ROI region, measure task-utility drop →
     an importance weight per tensor (expect `high` semantic level ≫ `low`; ROI-inside ≫ outside).
  3. **Three loss modes:** *informed* (drop lowest-importance tensors/chunks under a byte budget),
     *uncontrolled* (random chunk loss), *none* — compare task-utility vs bytes.
- **Output:** proof that informed prioritization beats uncontrolled loss → the RL action
  "prioritize important tensors / send nothing this period," and a new action dimension.

### F. Channel/network condition sweeps  ·  **LATER (Month 3), quick subset now**
- **Now (cheap):** OAI **rfsimulator** channel models — sweep SNR / channel profile via
  `channelmod_rfsimu.conf`; also `tc netem`/iperf background load for delay/loss/bandwidth. Log the same
  metrics. Gives controlled network states for the RL without Sionna.
- **Output:** model/transport behavior under a small grid of channel conditions → RL network-state coverage.

### G. Sionna ray-tracing + OSM channels  ·  **LATER (Month 3)** — scope + groundwork only
Realistic site-specific channels. Big integration; do **groundwork** now, not the full pipeline.
- **Groundwork:** install Sionna; export a matching **OSM/CARLA-town scene** to a Sionna RT scene; place
  TX(gNB)/RX(ego) at CARLA coordinates; produce a **channel impulse response / path-loss trace** per ego
  position; define the **coupling interface** to OAI rfsim (or a standalone channel replay). Defer full
  coupling to Month 3.

### H. Parallel sweep/training orchestrator  ·  **ENABLER (early this week)**
The launcher runs one config; sweeps need fan-out.
- **Build:** `scripts/sweep_runner.py` — a small job runner (config list → N concurrent processes, GPU-aware,
  logs to per-config run folders). Powers A, D, E, F without touching the single-config trainer.

---

## 4. RL formulation (extends `SCENESENSE_RL_SCHEMA.md`)
- **State:** scene (density, foreground fraction, vulnerable-object flag, mean confidence) + payload/latency
  history + **network** (UE tx/rx Mbps, RTT, gNB MCS/RB/TBS/BLER/HARQ, grant rate) + **channel** (from F/G).
- **Action (discrete profiles):** AE profile · quant level · ROI threshold · **tensor-priority/drop set (from E)**
  · send/skip · redundancy. Route-masked.
- **Reward:** task_utility − payload − latency − timeout/loss − guardrail_penalty (weights fixed for now).
- **Guardrails (C):** task floors, vulnerable-object protection, confidence/network fallbacks.
- **Learning:** offline replay first; contextual bandit / DQN over profiles; **no SAC** until continuous knobs
  + simple baselines beaten.

---

## 5. The realistic plan for the LAST WEEK of Month 2
We are behind and it's one week, so hit the **Month-2 Definition of Done** (the proposal's actual exit
criterion) and *start* the high-value novelty. Order:

1. **H** sweep runner (½ day) → unblocks everything.
2. **A** static sweeps over quant×entropy×ROI → Pareto + best/worst fixed policy (1 day).
3. **B** offline controller harness: trace-join → action catalog → reward → baselines → 1 bandit/DQN (2 days).
4. **C** guardrails in config + accepted/clamped/rejected reporting (½ day).
5. Write the **Month-2 slide/report** (½ day): sweeps, best-fixed vs heuristic vs bandit, guardrail behavior.
6. **Start D + E** (real AE + tensor-tagging) as the bridge into Month 3 — likely finish early M3.

**Month-2 DoD hit by 1–5:** static payload/latency/task profiles collected; offline replay scores static +
heuristic + one learned policy on logged metrics; guardrails concrete. **Defer to Month 3:** full AE zoo,
full tensor-tagging loss study, channel-condition grid (F), Sionna (G).

---

## 5b. Decisions locked (2026-07)
- **Scope this week:** hit Month-2 DoD (H→A→B→C→report) **and** start novelty (D, E). ✅
- **Learner:** contextual bandit (**LinUCB**) over discrete profiles; DQN only if time. ✅
- **Sionna:** **Month 2** (parallel track G, starts now) — *overrides the M3 recommendation.*
  ⚠️ **Risk note:** Sionna is a from-scratch integration (install → OSM/CARLA scene export → CARLA-coord
  TX/RX → CIR → OAI coupling). Run it as a **separate parallel track so it does not displace H/A/B/C**
  (the DoD core). Realistic Month-2 target for G = *groundwork* (install + toy OSM scene + one CIR trace +
  the coupling interface spec); full OAI coupling likely still completes early Month 3. Treat a working
  channel grid via **OAI rfsim (F)** as the Month-2 fallback if Sionna slips.

## 5c. Locked knob config (the sweep + action space)
| Knob | Values | Notes |
|---|---|---|
| **AE bottleneck** | 128 / 64 / 32 ch, task-aware | 128 = build on `rd_ae_b128.pt`; 64/32 trained by us; fusion-path (960-ch) variant needed |
| **Quantization** | 8 / 6 / 4 bit | 8 & 4 exist; **add a 6-bit (`per_channel_uint6`) codec** |
| **ROI (objectness) threshold** | 0.1 / 0.3 / 0.5 | higher = more drop; watch pedestrian recall |
| **Frame send/skip** | send / skip-1 / skip-2 | *temporal* (whole frames) |
| **Tensor priority/drop** | via `importance_head` + per-tensor packet tag | *intra-frame*; informed vs uncontrolled vs none |
Baseline = uncompressed (default training) = accuracy ceiling; every knob measured as degradation from it.

## 6. Still-open inputs (need from you/supervisor)
1. ✅ **AE list — RESOLVED:** {128, 64, 32}, task-aware; head-start = `rd_ae_b128.pt` (has an importance_head).
2. ✅ **Sionna — Month 2** (parallel track; risk-flagged).  ✅ **Learner — LinUCB bandit.**
3. **Online vs offline this month** — schema + proposal say **offline replay only** in Month 2; online RL on
   live CARLA/OAI is Month 3. Confirm.
4. **Which trained model** is the controller's base — recommend the **moving-ego RGB+radar fusion model**
   (`mIoU 0.825`, realistic UE deployment). Confirm or point to the checkpoint.
5. **`rd_ae_b128` provenance** — confirm with supervisor whether a matching AE class/trainer exists in his
   own workspace (not this repo) before we rebuild the module + trainer from the checkpoint shapes.

## 6b. Training jobs (the "train models in parallel" track) + findings so far
Post-hoc knobs (quant) need no training. Two model-training jobs feed the action space — and they must be
**ORDERED**, because ROI + AE **compose** at inference (`ROI-drop → AE-encode → quant → AE-decode → heads`):
1. **Drop-aware ROI model M′ FIRST** — fine-tune from the 200k model with **objectness-guided
   feature-dropout, `q ~ Uniform(0, 0.8)`** per batch (importance = objectness/GT-object cells; q=fraction).
   Range includes q=0 → M′ handles BOTH full and dropped features → generalizes across all ROI thresholds
   (not per-threshold). *Motivation (validated):* base model tolerates only mild informed drop (30% free,
   50% ~1–2 pt); aggressive ROI is OOD → needs this.
2. **AE codecs {128,64,32} SECOND, trained on M′ with ROI-drop in the loop** — task-aware (output-distill
   M′), so the AE compresses the *dropped-feature* distribution it will actually face, against a teacher
   (M′) that behaves well on drops. **Training the AE on the plain 200k model first is premature** — it
   would compress full features it won't see and distill a broken teacher on dropped features (2026-07
   design catch). *(AE module `feature_ae/ae_model.py` is built + reused as-is; only teacher + in-loop drop change.)*
Deployed thing the agent controls = **one drop-aware M′ + AE decoders**; `{ROI, AE, quant}` compose on M′.
*(Advanced alt: a single JOINT fine-tune of M+AE with ROI/AE/quant all randomized in the loop.)*

**M′ = FULL-model fine-tune (backbone+heads)** — decided 2026-07 for long-term robustness ceiling + one
strong base for all RL analysis (over the faster head-only). **Reuse the exact 200k recipe** (borrow the
techniques so accuracy holds): Stage-1 seg AdamW lr1.5e-4 wd1e-4, strong+geometric aug, freeze_bn, bs24,
50ep, Lovász0.5, class-weights[0.5,1,4], person_miou selection, cosine+warmup, `object_total=0`; Stage-2
detection frozen-backbone, object-loss `{center:4,location:1.5,dim:0.6,yaw:0.3,parked:0.2,radar:0.1,bbox2d:1}`,
gated≤40m, NMS-6. **Add:** an opt-in objectness-guided feature-dropout hook in the editable `train_fusion.py`
(drop lowest-objectness `q~U(0,0.8)` per batch; default recipe unchanged when off). Heatmap-collapse
safeguards: warm head init + verify peaks. **ACCEPTANCE: M′ at q=0 ≥ 200k targets (mIoU 0.837 / veh 0.934 /
recall 0.775 / ped-loc 1.38m) + live peaks**, plus graceful drop-robustness.
**Then RE-RUN all sweeps (quant {8,6,4}×{none,zlib,zstd}, ROI, accuracy) on M′** — current results are the
plain-200k pre-robustness baseline; M′'s become the RL agent's action-cost model.

**Sweep findings locked (2026-07):** quantization {8,6,4} near-lossless (per-channel; per_tensor dominated;
uint4≈lossless, 7.4–7.9× w/ zlib/zstd, best delivery); ROI importance-drop free to ~30%, mild at 50%;
entropy coder is a free payload win (lossless, zero accuracy effect). ROI = quantile importance-drop (not
absolute threshold — the fusion objectness scale differs from the OD path); add an **objectness floor** as
the guardrail so aggressive fractions never drop true-object cells.

## 7. Runtime / deployment clarifications (from 2026-07 supervisor discussion)
- **ROI threshold = same model, post-hoc.** No per-threshold models. Optional later: ONE drop-aware model
  trained with region-dropout to degrade gracefully — still not one-per-threshold.
- **AE switching = one perception model + all AE decoders resident on the back half; select by tag.** Front
  uses AE-encoder(size); back half holds off/128/64/32 decoders in memory and **picks the matching decoder
  per frame from a profile tag** (`ae_id`, quant, roi). The perception net never switches/reloads. The tag
  rides in the payload today (read after reassembly) → we **promote it to the UDP header** (workstream E) so
  the agent and the network/QoS layer can act on it.
- **Data:** AEs train on features from the **frozen 200k model over EXISTING datasets** — no new perception
  collection to start. RL "data" = the **traces** the sweeps generate. New collection only if a *new
  scenario distribution* is wanted (confirm which "data collection" the supervisor meant).

## 8. Engineering vs Research split + monthly tracking (process)
**Every task is tagged [ENG] (plumbing) or [RES] (a scientific question/contribution).** Workstream tags:
| WS | Eng part | Research part |
|---|---|---|
| H sweep runner | **[ENG]** job fan-out | — |
| A static sweeps | [ENG] run/collect | **[RES]** which quant/entropy/ROI trades bytes for accuracy; the Pareto |
| B controller harness | [ENG] trace-join/catalog/replay | **[RES]** reward design; does a learned policy beat the best heuristic |
| C guardrails | [ENG] config+reporting | **[RES]** where the task/vulnerable-object floors sit |
| D autoencoders | [ENG] module+trainer plumbing | **[RES]** which bottleneck size / task-aware training retains accuracy |
| E tensor-importance | [ENG] per-tensor framing+header tag | **[RES]** importance ranking; informed-vs-uncontrolled-vs-no loss |
| F/G channel+Sionna | [ENG] rfsim/Sionna integration | **[RES]** model/policy behavior vs channel conditions |

**Monthly practice (every presentation):** keep `SCENESENSE_MONTHLY_CHECKLIST.md` ticked as items complete;
maintain a living **"Month N — done / remaining"** summary (split by [ENG]/[RES]) that we lift straight into
the month's slides. Update it at each work session, not just at month end.

---
*Anchors: `SCENESENSE_MONTHLY_CHECKLIST.md` (Month 2 §3/§5/§6), `SCENESENSE_RL_SCHEMA.md`,
`SCENESENSE_MONTH2_COMMANDS.md`. Infra: codecs/AE/ROI in `carla_split_inference_udp_data_collect.py`;
OAI metrics in `scripts/{sample_oai_network_metrics,parse_oai_gnb_mac_stats,analyze_nrue_grant_metrics}.py`.*
