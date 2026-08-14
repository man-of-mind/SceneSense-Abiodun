# The RL journey: environment, reward, baselines, results, and what comes next

**Audience:** advisor + team. **Status:** draft for a presentation in ~3 days; becomes slides once Tasks A/B/C
land. **Supersedes** `PRESENTATION_STORY.md` (written before the results). Every number here is traceable to an
artifact path; scope caveats are stated inline rather than in a footnote, because two of them changed our claims.

---

## 1. The question we set out to answer

A car doing cooperative perception must decide, every frame: **do I compute locally, or compress features and ship
them over 5G to an edge that fuses them into a shared map — and if so, at what quality and frame rate?**

The choice trades three things against each other: **perception quality** (how good is the map), **freshness**
(how stale is what the map believes), and **network cost/feasibility** (can the uplink actually carry it).
The proposal's hypothesis was that this is hard enough to need a **learned controller (RL)**.

**Our job was to test that hypothesis honestly before spending months training one.** This report is that test.

---

## 2. Why we built a surrogate environment first

Training RL directly against CARLA+OAI would be prohibitively slow (a single OAI run is 40–80 min) and
irreproducible. So we did what is standard for this class of problem: **measure the world once, then replay it.**

The environment is **table-driven**: every action's consequence comes from a measured lookup, not a guess.
Three measured tables plus a replayed CARLA corpus:

| Table | What it gives | Artifact |
|---|---|---|
| **Knob matrix** | accuracy ↔ payload ↔ compute latency for **36 compression profiles** (49.4 KB → 2,835 KB), transport-invariant | `PERMODEL_KNOB_MATRIX_ZSTD.md` |
| **Channel sweep** | delivery %, latency, scheduled UL throughput per (payload × channel rung) | `channel_condition_sweep/combined_surface.csv` |
| **Staleness / FPS** | how localization error grows with latency and object speed | `staleness/uplink_only_latency_budget/` |
| **CARLA corpus** | replayed per-frame scene state (objects, speeds, ranges, map AoI) | `policy_corpus_advisor_rich_v5` |

**This is itself a contribution:** the action→outcome map is *measured*, so the environment is honest about what
each choice actually costs.

---

## 3. The environment in numbers

**Scenario corpus (CARLA 0.10, Town10HD_Opt, native 10 Hz):**
- **24/24 runs collected, 8,480/8,480 frames** — clean completion, no partial runs
- **23 trajectories retained** (one impact run, `pcarv5_mixed_va01`, excluded by the QC gate)
- Split: **12 grouped training trajectories / 6 held-out test trajectories**; **2,638 frames evaluated per
  controller**. Trajectory-grouped splits — no frame from a training trajectory leaks into test.
- On-contract radar density **19,404.5 returns/frame** (the 10 Hz sensor contract; a 20 Hz world had silently
  halved this earlier — see §9)
- Scene regimes deliberately spanned: slow pedestrians and sustained ≥10 m/s vehicles; **54.43%** of frames carry
  GT-seeded map-freshness pressure; **19 runs** have slow-regime support

**Network-varying scenarios:**
- **4 measured channel rungs**, from the real OAI stack: **clear 50.3 dB / MCS 28**, **mild 19.5 / MCS 24**,
  **mid 15.6**, **strong 8.2 / MCS 9**
- Measured scheduled-UL capacity: **≈36.7 / 27.8 / ~20 / ~9–10 Mbps** respectively
- Sweep design: **3 payload classes × 4 rungs = 12 cells**, 120 s each (~1,200 frames/cell), every cell passing a
  health gate
- The **delivery knee is sharp and payload-ordered**: 1 MB gets 97.5% delivery on clear but **4.6% on strong**
  (15.5 s p50!); 400 KB holds to 15.6 dB; the **90 KB seg-safe payload delivers 100% at every rung**
- In the surrogate, the channel is a **4-rung sticky Markov process** with **±30% capacity uncertainty, a 2-step
  observation lag, and 5% noise** — the controller never sees true capacity or the future.

---

## 4. What the controller chooses (action space)

`SKIP` (send nothing this frame) **or** `SPLIT(compression profile, target FPS)` — 35 SPLIT actions + SKIP as
evaluated, drawn from the 36 measured profiles. Profiles vary autoencoder width, quantization, and ROI drop.
`LOCAL` (full on-car inference) is **not yet calibrated** and is excluded — a stated gap.

---

## 5. Reward and constraints — the two-layer design

We deliberately separated **what must never be violated** from **what we optimize**.

### 5a. Hard constraints (the shield) — safety first
```
C1 admission:   offered_mbps(a)  <=  pessimism * estimated_capacity     (pessimism = 0.70)
Safe set:       e_bound(a, s)    <=  eps
```
- **Localization error model:** `e_j = sqrt( base_loc(a)^2 + (v_j * AoI_map_j)^2 )` for each object *j*; the frame's
  governing error is `G = max_j e_j`. Intuition: *a fast object whose map entry is stale is where the map lies
  most.*
- Capacity is estimated **pessimistically** from an EWMA (α = 0.20) over a 1 s window with a **1-tick observation
  lag**, and unsent frames are replaced by newer ones (`newest_replaces_unsent`).
- **Honest scope:** the shield is **sound at 25 m but unsound at 40 m**, and it only protects **detected** objects
  — a detector miss is invisible to it. So we say the system *enforces a conservative action contract and
  quantifies violations/abstention*, **not** that it "guarantees" safety.

### 5b. Soft objective (the reward) — optimize inside the safe set
```
R(a,s) = w_task * U_task(a,s)  -  C_PRB(a,s)  -  lambda_switch * 1[mode changed]  -  w_E * ( G / eps )

U_task = 0.35 * segmentation  +  0.40 * pedestrian_recall  +  0.25 * vehicle_recall
```
Pedestrians weigh **more than** vehicles by design (advisor input: vulnerable road users dominate). An explicit ROI
cost was removed — ROI damage already shows up inside `U_task`, so charging it twice was double-counting.

---

## 6. The baselines — what each is, and why it earns a place

We built a **ladder** of increasingly capable controllers. The logic: *RL is only justified if everything simpler
leaves a real gap.*

| Controller | What it does | Why we included it |
|---|---|---|
| **Fixed** | Always the same action | Floor. Shows whether adaptation matters at all. |
| **Rule** | Hand-written thresholds on capacity/staleness | What an engineer would ship without ML. If this wins, nothing else is needed. |
| **Greedy** | Each frame, pick the **highest-reward action allowed by the shield** — no lookahead | The natural "act on what you see now" policy. This is our main comparison point. |
| **LinUCB** | **Contextual bandit** — learns a linear reward model per action from experience, with an optimism bonus for under-tried actions | Tests whether *learning from data* helps, without needing sequential credit assignment. Trained on the 12 training trajectories, **frozen** before test. |
| **MPC** | **Model-predictive control** — each tick, simulates several steps ahead using the measured channel/latency/kinematics models, picks the best first action, then replans | Tests whether **planning ahead** helps. Crucially, MPC is *the* test for sequential structure: if looking into the future doesn't help, there is little for RL to learn. It gets no future frames and no true capacity. |
| **Clairvoyant oracle** | Knows the realized outcome, picks the best action per state | A **reference ceiling**: how much is left on the table? |

**In plain terms for the talk:** greedy = "do the best thing right now." MPC = "think a few steps ahead." Oracle =
"cheat and see the answer." If greedy ≈ MPC ≈ oracle, the problem is a **lookup**, not a learning problem.

---

## 7. Result 1 — the single-UE ladder

Held-out test: 6 trajectories, **2,638 frames per controller**, paired channel/latency randomness.
(`rl_agent/policy/experiments/controller_ladder/20260813_063514`)

| controller | mean scored reward | matched-safe % |
|---|---|---|
| rule | 0.1918 | 91.13 |
| **greedy** | **0.1965** | 91.13 |
| linucb | 0.1906 | 91.13 |
| **mpc** | **0.1983** | 91.13 |

**What it tells us:**
- **MPC beats greedy by only 0.91%**, with a bootstrap interval covering zero, and they choose differently on just
  **2.54%** of frames. **Planning ahead buys almost nothing.**
- **LinUCB (the learner) was *worse* than greedy.** Learning a reward model added variance, not skill — because the
  reward model is already *measured*.
- All controllers hit an identical **91.13% matched-safe rate**: the shield, not the optimizer, determines safety.

**Read:** the sequential structure RL would exploit is largely absent. First **NO-GO**.

### 7a. "Is 0.19 a good reward?" — how to read the scale (expect this question)

Config: `w_task = 1.0`, `w_error = 0.05`, `lambda_switch = 0.10`, `eps = 2.0 m`; task weights
**0.35 / 0.40 / 0.25** summing to 1.0; task references **miou 0.840 / ped 0.887 / veh 0.927**
(`policy/configs/track_a_pilot.yaml`).

Because each metric is divided by its reference and the weights sum to one, **a delivered top-quality frame scores
`U_task` ≈ 1.0.** So the per-frame task ceiling is ~1.0 — but **0.19 is NOT "19% of achievable"**, for a reason
that is itself a finding:

**The controllers SKIP almost every frame:** `skip_pct` = **96.0% (greedy) / 98.6% (MPC) / 98.5% (rule) /
94.1% (LinUCB)**, with `over_budget_pct` **27.5%** and shield conditional false-rejects **20.5%**. So the mean
reward is dominated by **the retained map's value when you do not send**, not by delivery quality. Sending is
usually **not admissible** — precisely what the feasibility frontier predicts (only **7/200** cells feasible at
400 KiB).

**Penalty magnitudes, for the same question about the switch cost:**
- `lambda_switch = 0.10` — a 10% tax on *changing mode*; it creates hysteresis against flip-flopping. Real, not
  prohibitive.
- `w_error * (G/eps) = 0.05` when the error sits exactly at the safety bound, growing as staleness grows.
- Therefore **the reward is task-dominated; freshness acts as a mild tiebreaker.** *Open question for the advisor:*
  is that the intended balance, given freshness-awareness was the motivation? (The sensitivity sweep only spanned
  `w_error` ∈ [0.025, 0.10], so this was never tested at a stronger setting.)

**Why the absolute number is the wrong yardstick.** What matters is reward **relative to the achievable range at
this operating point** — SKIP-floor to oracle-ceiling. That is exactly what the **+1.383% oracle gap** (§8)
measures, which is why the oracle comparison, not the absolute 0.19, carries the conclusion.

**⚠️ The caveat this exposes — flag it on the slide, do not let a reviewer find it.** If ~96% of decisions are
effectively *forced* (only SKIP admissible), then there is little decision latitude for **any** controller,
including RL. The oracle changed **1.56%** of actions and SPLIT was chosen on ~4% of frames — the same order of
magnitude. So the near-tie may partly reflect **how little choice exists at this operating point**, rather than how
learnable the problem is.

**Required diagnostic (cheap, requested of codex):** conditional on frames where **≥2 actions are admissible**,
report (i) how often the oracle disagrees with greedy, and (ii) the reward gap. If the conditional gap is still
small, the NO-GO strengthens considerably. If it is large, the honest conclusion changes to *"there is little
headroom because there is little choice at 400 KiB"* — an operating-point statement, not a learnability one — and
the gate should be re-run at a payload where sending is routinely feasible (e.g. the 90 KB seg-safe point).

---

## 8. Result 2 — expanding the action space, then measuring the ceiling

**A fair objection to Result 1:** maybe greedy ≈ MPC only because the action space was too small — the controller
couldn't reduce payload/FPS to make deadlines feasible. So we **expanded the action space** (joint
UE × mode × profile/payload × FPS, expired-work dropping admissible) and measured **greedy against a clairvoyant
oracle** in that larger space. (`expanded_action_gate/20260813_233947_pdt`)

| | value |
|---|---|
| greedy | 0.192625 |
| oracle | 0.195290 |
| **lift** | **+1.383%** (95% CI [+0.19%, +0.35%]) |
| pre-registered bar | **+5% relative / +0.01 absolute** |
| states evaluated | 12,955 |
| **actions the oracle changed** | **1.56%** |
| verdict | `EXPANDED_SURROGATE_NO_GO_STOP` |

**What it tells us:** with *perfect knowledge of the outcome*, you would change roughly **1 decision in 64**, for
a **1.4%** gain — far below the bar we registered in advance. Greedy isn't just close in score; it makes nearly the
same *choices*.

**Honest scope — this matters and we corrected it:** the oracle is **one-step**, evaluated on the states **greedy
visits**, with matched support (`policy/expanded_gate.py:581`). It therefore does **not** upper-bound a sequential
policy that would reach different states. The defensible claim is *"no useful one-step headroom within the
static-quality, queue-free, matched-support contract"* — **not** "all RL is bounded by 1.4%." We initially
overstated this and fixed it.

---

## 9. Result 3 — the multi-UE detour: does contention create the missing headroom?

**Why we went there.** Single-UE greedy being near-optimal is unsurprising in hindsight: one car, one uplink, no
one to coordinate with. But **many cars sharing one cell** is a genuine coordination problem — congestion collapse,
tragedy-of-the-commons. If RL has a home, that's where.

**What we built.** A bounded, pre-registered measurement on the real OAI stack (`rl_agent/multiue_oai/`):
**N=2 UEs, strong AWGN, 400 KiB production-shaped messages, 9 trials, 2 independent restart/calibration blocks**,
comparing **decentralized hard-C1 admission** vs **centralized observable admission** at matched asymmetric load.

**The infrastructure was harder than the science** (worth one slide, it's a good lesson):
1. A missing per-UE RFsim channel object (`rfsimu_channel_ue1`) → silent fallback; fixed, plus a preflight guard.
2. Two UEs could not both attach under the harsh channel → attach on a **clean** channel, then switch to strong
   **at runtime**. Diagnosis came from one **4½-minute** test, not another 80-minute run.
3. Interface↔IP binding is **race-dependent** (`oaitun_ueN` is stable, the IP is not) → identity must key off the
   interface name, discovered per run. This was a real correctness bug: traffic could traverse the wrong radio UE.
4. The registered 250/500 ms deadline turned out **physically infeasible**: 400 KiB at the measured 6.077 Mbps
   ceiling is **~540 ms of serialization alone**. Measured **0/490 arrivals** inside 500 ms, fastest **515 ms**.

**Results.**
- **No coordination gap at N=2.** Centralized admission did not clear the deadline or latency thresholds.
- **Mechanism:** the **5G MAC scheduler is already the coordinator** — aggregate **6.0898 Mbps against a 6.0774
  Mbps calibrated ceiling** (residual ratio 1.0026). It is already at capacity and already reallocating to the
  heavy UE. An application-layer coordinator has nothing left to win *on capacity sharing*.
- **A first "GO" signal was an artifact.** The N=50 screen initially reported **+47.73 pp** worst-delivery lift —
  traced to a model bug (`allocate_equal_ratio` scaled all UEs uniformly instead of max-min work-conserving).
  Corrected: **216/216 cells valid, 0 survive** → `STOP_CHEAP_NO`. Static max-min lifts worst-case **allocation
  15.91% → 54.55%** yet gives **0 pp deadline lift** at both 0.25 s and 0.50 s.
- **The centralized rule was actively worse on timeliness:** 54-cell check gave **3,187** within-500 ms arrivals
  decentralized vs **127** centrally — oldest-pending spends capacity on already-expired updates.

**What it tells us:** coordination *works at what it does* (allocation) and still buys nothing on the metric the
application cares about — but note the comparator was **fairness-oriented, not deadline-aware**, and the deadline
metric was **saturated**. So the honest claim is narrow: *this central admission rule does not improve worst-UE
deadline delivery.* A deadline-aware coordinator that drops expired work was **never tested**.

---

## 10. So what did the whole journey tell us?

**Two independent mechanisms, pointing the same way — and we keep them separate on purpose** (fusing them into one
grand narrative would not survive review):

1. **Single-UE:** the action→accuracy map is *measured and monotone*, so choosing a compression knob is a
   **constrained lookup**, not a learning problem.
2. **Multi-UE:** the MAC scheduler already operates at the capacity ceiling, so application-layer capacity
   coordination has no room at N=2.

**Bottom line: do not train RL now.** Not because RL is bad, but because we **measured** that the simplest policy is
within ~1.4% of a one-step ceiling, and no cheaper rung on the ladder was left behind.

**And what is genuinely NOT falsified** — we are careful here:
- **Scene-conditioned quality.** `profile_quality()` takes **no scene argument**, so "different scenes want
  different profiles" was **structurally unrepresentable**, not rejected. (Task A tests it.)
- Queue-coupled dynamics; calibrated `LOCAL` actions; deadline-aware load shaping that drops expired work;
  Phase-2 object-selective map sharing.

---

## 11. The positive result — load shaping beats coordination

The most actionable finding falls straight out of the feasibility arithmetic
(`serialization = payload·8 / per-UE share ≤ deadline`):

| payload | feasible cells (of 200) |
|---|---|
| 400 KiB | **7** |
| 90 KB (seg-safe) | **58** |
| 49.4 KB | **89** |

**A ~12.7× larger feasible operating region from payload choice alone.** The deadline is met or missed by **what
you send**, not by how cleverly you schedule it. That validates the whole compression/knob line of work — and note
the honest boundary: even at 49.4 KB, **~55% of cells remain infeasible**. We report the envelope, not a cure.

**Related supporting result — bottleneck migration:** sensor preparation dominates the *optimized, uncongested*
path; **OAI queueing dominates under large payloads or poor channels**; task-aware compression moves the system
back into a compute/sensing-limited regime (RTT 209 → 77 ms, delivery 75% → 99%). Plus a stack-level fix: the UE
uplink latency root cause was **RLC queue-wait from the gNB UL scheduler's MCS cap**; a SINR-driven policy cut RTT
**186 → 48 ms**.

---

## 12. Next steps

**In flight now (desk-only, no CARLA/OAI):**
- **Task A — does context change the best profile?** Argmax-stability / rank-reversal screen over **all 36
  profiles** and 1,683 common samples, bucketed by **class mix, confidence, range, occlusion**. *Pre-registered
  asymmetry:* a detection-only null is **INCONCLUSIVE, not a closure**, because per-frame segmentation metrics
  don't exist yet and segmentation is the most profile-sensitive term.
- **Task B — vulnerable-object guardrails.** Low-confidence clamp + pedestrian/cyclist no-skip as **hard shield
  rules** (protects *observed* objects only). An unmet proposal commitment; needs no RL.
- **Task C — principled baselines.** Exact budgeted enumerator + **λ-RDO supported-hull lookup**, reporting action
  agreement, reward gap, and duality gap.

**The critical path to the Month-6 system:** **Phase-2 recipient-specific map sharing, integrated end-to-end**
(same path as multi-vehicle integration if scoped to one helper + one recipient), then run it over the existing
two-UE OAI system, then navigation warning/override. **~7–10 weeks, 9–12 with contingency.**

**The binding gap:** the **transport-conditioned cooperation gain** does not exist yet. The 1.40 m two-view
triangulation result is groundwork — **static egos, oracle association, no OAI transport.**

**Open decision for the advisor:** target venue. MobiSys/MobiCom likely need an **over-the-air** leg; MSWiM/SECON/
WoWMoM/VNC (or TMC/IoT-J) fit a real-stack-plus-simulation study. Non-blocking — stabilize Phase 2 over RFsim
either way.

---

## 13. Algorithms that actually suit this problem (and the comparison set)

**A useful framing:** our problem is a **rate–distortion optimization** where the distortion axis is *task utility*
(mIoU / recall / localization) rather than MSE, coupled to an **Age-of-Information** freshness penalty. That places
it precisely between two mature literatures — and tells us which baselines are the *principled* ones.

**For the current (whole-frame) contract:**
- **λ-RDO / Lagrangian hull lookup** — the canonical coding-theory method. *Caveat worth stating:*
  budget-constrained enumeration can pick **non-supported** Pareto points that **no λ can reach**, so
  "greedy == λ-RDO" must be measured, not assumed.
- **Deadline-aware rules:** least-slack / EDF-style, and **max-weight** scheduling — the natural fix for the
  oldest-pending weakness we found.
- **AoI-index-inspired heuristics** — freshness-first ordering. *Not* to be called Whittle without establishing
  indexability.
- Already have: fixed, rule, greedy, LinUCB, MPC, clairvoyant oracle.

**For Phase-2 object-selective map sharing (where the interesting theory lives):**
- **Restless bandits / Whittle index** — here it is *legitimate*, because **individual objects become natural
  scheduling arms**. This is the right home for a genuine Whittle baseline.
- **Submodular maximization** — "which objects to share under a byte budget to maximize map value" is a classic
  coverage problem. *If* the map-value function is submodular (plausible for coverage/occlusion — **must be
  verified**), then **greedy carries a (1 − 1/e) approximation guarantee**. That would give us a *theoretical*
  reason for greedy's near-optimality, which is stronger than our current empirical argument.
- **Assignment/matching** for recipient selection (who needs this object most).

**If Task A finds contextual rank reversals:** contextual bandits (LinUCB — have it), decision trees / rule lists,
or simply a **contextual lookup table keyed on class mix**. Most likely a lookup still suffices.

**Only if a residual sequential gap survives all of the above:** **masked categorical PPO** is the natural first
POMDP baseline; discrete SAC or DQN if the action space factorizes. **Continuous SAC is inappropriate** (our actions
are discrete) — worth noting since SCAN-AI used SAC for a continuous formulation.

**The gate for reopening RL:** a *new* pre-registered gap on an expanded contract, measured against these
baselines — not a retune of the gates we already passed.

---

## 14. Limitations to state on the slide, not hide

- **RFsim PHY, not over-the-air.** Real OAI protocol stack, scheduler, and measured channel models — but not a real
  radio.
- CARLA simulation, not physical vehicles/sensors.
- Measured contention at **N=2** only; N=50/100 is modeled (and v1 of that model had a bug we caught).
- The oracle gate is **queue-free, one-step, matched-support**.
- Evaluation is segmentation + pedestrian/vehicle recall & localization — **not true OD AP**; cyclists and small
  objects uncovered.
- Shield soundness is **regional** (sound @25 m, unsound @40 m) and covers **detected** objects only.
- Scene-conditioned perception utility **untested**.

---

## 15. The one-slide summary

> We built a measured, replayable environment for the split-inference decision (24 runs / 8,480 frames / 4 measured
> channel rungs / 36 compression profiles), with a two-layer controller: a hard safety shield plus a
> freshness-and-quality reward. We then climbed a ladder — rule, greedy, contextual bandit, MPC, clairvoyant oracle
> — and pre-registered what would count as a win for learning.
>
> **Learning did not earn its place.** Greedy is within **1.4%** of a one-step oracle and makes the same choice
> **98.4%** of the time; a contextual bandit did worse; MPC gained **0.9%** with an interval covering zero. At N=2
> the 5G scheduler is *already* the coordinator (6.090 vs 6.077 Mbps ceiling), so coordination added nothing.
>
> **What does work is load shaping:** payload choice expands the feasible operating region **~12.7×** (7 → 89 of
> 200 cells). The lever is *what you send*, not *how cleverly you schedule it*.
>
> So we are not training RL. We are building the end-to-end cooperative-perception system, and the remaining open
> question is a real one we can test cheaply: **does scene content change which compression profile is best?**

---

## 16. Slide-production requirements (Abiodun, 2026-08-14) — apply when building the deck

**Design principle: the deck must be self-contained.** Assume it gets forwarded to someone who was not in the room
and who asks no questions. Every symbol defined on or adjacent to the slide that uses it; no ambiguity; no
"as discussed."

1. **Block diagram of the decision loop** — the state → shield → action → outcome → reward → next-state cycle, so a
   reader sees the architecture before any equation. Source material exists in `REWARD_LOOP_DIAGRAM.md`; render it
   as a proper figure (not ASCII). A second, simpler diagram should show the **physical pipeline**: car sensors →
   front backbone → compress → 5G uplink (OAI) → edge fusion → shared map → back to the car's planner.
2. **Real CARLA frames wherever they ground a claim** — use captured frames, not clip art. Specifically:
   - a scene showing the **ego + helper vehicle + pedestrians** (motivates cooperation),
   - an **occlusion** case (motivates Phase 2),
   - side-by-side **ROI-drop damage to segmentation** vs intact (this is the density/seg finding and it is visually
     obvious),
   - a **radar point-cloud overlay** at the on-contract 19,404 returns/frame (shows the multi-modal input).
3. **Equations rendered properly** — LaTeX/MathType, not plain text or screenshots of code. The set that must
   appear: the reward `R(a,s)`, `U_task`, the per-object error `e_j` and `G = max_j e_j`, the C1 admission
   inequality, and the feasibility condition `payload*8 / per-UE-share <= deadline`.
4. **A notation table as its own slide (and repeated in the appendix)** — every symbol: `a`, `s`, `U_task`, `G`,
   `e_j`, `v_j`, `AoI`, `eps`, `kappa`/pessimism, `mu_hat`, `b(s)`, `w_task`, `w_error`, `lambda_switch`,
   `C_PRB`, plus units. No symbol used before it is defined.
5. **Clean, consistent plots** — use the validated palette already used for `plots/knob_accuracy_frontier.png`
   (blue `#2a78d6` = Pareto/primary, orange `#eb6834` = operating points, grey `#898781` = dominated/secondary;
   light surface `#fcfcfb`). Direct-label series rather than relying on legends alone; never a dual y-axis.
   Figures to produce/reuse:
   - knob accuracy-vs-payload frontier (**exists**),
   - channel sweep delivery/latency knee (**exists** in `channel_condition_sweep/plots/`),
   - the **controller ladder bar chart** with error bars (rule/greedy/LinUCB/MPC) — must show the overlap,
   - **greedy vs oracle** with the +1.383% gap and the pre-registered 5% bar drawn as a threshold line (the visual
     makes the NO-GO instantly legible),
   - the **deadline-feasibility frontier** (7 / 58 / 89 of 200 cells) — this is the positive result, give it a full
     slide,
   - the **skip-rate / admissibility** chart supporting §7a (why 0.19 is the number it is).
6. **State every scope caveat on the slide that makes the claim**, not only in the limitations slide — especially
   "one-step, matched-support oracle", "N=2 measured / N=50 modeled", "RFsim, not over-the-air".

**Build the deck only once Tasks A/B/C land**, so no number changes after the figures are made.
