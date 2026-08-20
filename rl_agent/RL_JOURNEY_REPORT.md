# The RL journey: environment, reward, baselines, results, and what comes next

> **Current next-step supersession (2026-08-19):** retain this report's evidence
> and causal caveats, but do not follow its helper-recipient Phase-2 next steps.
> The sole current execution path is `UE_AGENT_EXECUTION_CHECKLIST.md`.

**Audience:** advisor + team. **Status:** audited working record; Tasks A/B/C have landed, but the causal audit
re-scoped the controller results before slide preparation. **Supersedes** `PRESENTATION_STORY.md` (written before the results). Every number here is traceable to an
artifact path; scope caveats are stated inline rather than in a footnote, because two of them changed our claims.

> # ⛔ READ FIRST — §§7-8 ARE RE-SCOPED (causal audit, 2026-08-14)
>
> A causality audit **confirmed same-frame oracle observation leakage** in the policy surrogate. The replay exposes
> the object set, class, confidence, world position, speed and persistent track identity **before** action
> selection — but those predictions are produced by the edge back-half only *after* features are encoded,
> transmitted and decoded (`uplink_only_spatial_map_pipeline/...:2063` -> `policy/env.py:101`). Track keys are also
> GT-assisted, with unmatched detections removed (`policy/replay.py:211`). It is **not** action-branch leakage
> (SKIP and SPLIT see the same observation), but every action sees information that in the real split pipeline
> would only exist after inference and communication.
>
> **Consequences — apply before presenting:**
> - **STILL VALID:** the measured surfaces (knob matrix, channel sweep, staleness), Task A's contextual screen,
>   Task C's **static** 36-profile lambda-RDO result, the deadline-feasibility frontier, and the **multi-UE DG-A**
>   OAI measurements (real runs, not replay).
> - **RE-SCOPED:** the controller ladder (§7), the expanded-action gate (§8), Task B's replay numbers, and Task C's
>   **runtime** half. These may be cited **only** as a *noncausal, matched-support upper-bound study* — **never** as
>   a deployable observation-based controller evaluation. **The full dynamic NO-GO is not citable as a system
>   conclusion.**
> - **Move §§7-8 out of the main advisor narrative into backup**, with the caveat stated on the slide.
> - **Phase-1 infeasibility/abstention numbers must NOT be carried into Phase 2:** Phase 2 already performs
>   constant-velocity extrapolation (`phase2_map_sharing/engine.py:35`), so the old shield's frozen-object
>   `speed x age` model does not represent the intended Phase-2 map.
> - This is **not a replay bug fixable in software**: causal pre-action observations were never recorded, which is
>   why a new paired corpus is required.

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

**In plain terms for backup material:** greedy = "do the best thing right now." MPC = "think a few steps ahead."
Oracle = "cheat and see the answer." Their comparison probes headroom **inside this replay contract**; because the
state is noncausal, it cannot establish whether the deployable problem is a lookup or a learning problem.

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

**What it tells us inside the noncausal matched-support replay:**
- **MPC beats greedy by only 0.91%**, with a bootstrap interval covering zero, and they choose differently on just
  **2.54%** of frames. **Planning ahead buys almost nothing.**
- **LinUCB (the learner) was *worse* than greedy.** Learning a reward model added variance, not skill — because the
  reward model is already *measured*.
- All controllers hit an identical **91.13% matched-safe rate**: the shield, not the optimizer, determines safety.

**Read:** little sequential headroom appears under the legacy replay's oracle observation. This is useful diagnostic
evidence, but **not a deployable dynamic-controller NO-GO**. The static single-UE profile-selection NO-GO remains
separately supported by the measured-table analysis.

### 7a. "Is 0.19 a good reward?" — how to read the scale (expect this question)

Config: `w_task = 1.0`, `w_error = 0.05`, `lambda_switch = 0.10`, `eps = 2.0 m`; task weights
**0.35 / 0.40 / 0.25** summing to 1.0; task references **miou 0.840 / ped 0.887 / veh 0.927**
(`policy/configs/track_a_pilot.yaml`).

Because each metric is divided by its reference and the weights sum to one, **a delivered top-quality frame scores
`U_task` ≈ 1.0.** The per-frame task ceiling is therefore ~1.0, but **0.19 is not "19% of achievable"**: it is an
average over a sparse, scheduler-gated replay in which the controller rarely attempts an update. More importantly,
the same-frame object evidence used to score and choose actions is not a causal pre-action signal. The absolute
reward is consequently useful only for paired comparisons within this legacy surrogate.

**Penalty magnitudes, for the same question about the switch cost:**
- `lambda_switch = 0.10` — a 10% tax on *changing mode*; it creates hysteresis against flip-flopping. Real, not
  prohibitive.
- `w_error * (G/eps) = 0.05` when the error sits exactly at the safety bound, growing as staleness grows.
- Therefore **the reward is task-dominated; freshness acts as a mild tiebreaker.** *Open question for the advisor:*
  is that the intended balance, given freshness-awareness was the motivation? (The sensitivity sweep only spanned
  `w_error` ∈ [0.025, 0.10], so this was never tested at a stronger setting.)

**Why the absolute number is the wrong yardstick.** Paired reward gaps are more interpretable than the absolute
scale, but even the §8 oracle gap is bounded to the same noncausal, one-step, matched-support contract. It does not
close the causal dynamic-control question. A further diagnostic on this corpus cannot repair the missing pre-action
signals, so no additional replay sweep is required before Phase 2.

---

### 7b. ⚠️ RETRACTED AND REPLACED — the "97% abstention" diagnosis was wrong (codex audit, 2026-08-14)

**An earlier version of this section claimed the controller "abstains ~97% of the time" because of reward
mis-calibration at a "400 KiB operating point." Both claims are false. Retained here, corrected, because the
corrected version is the more interesting finding.**

**What was wrong:**
- The ladder ran at a **20 Hz policy clock over 49-129 KiB profiles (90 KiB preferred core)** — *not* 400 KiB.
  400 KiB belongs to the separate contention/frontier study.
- `split_pct` is **schedule-active time, not packets sent**. The scheduler accrues credit per target FPS and
  attempts capture only when credit reaches 1 (`policy/env.py:232`).
- Realized rates: 2,638 ticks at 20 Hz = **131.9 s**; greedy made **41 attempts / 40 deliveries** =
  **0.303 deliveries/s**, about **one delivered update every 3.3 s** (not the 0.16/s previously stated).
- `over_budget_pct` is **misnamed**: it flags *no localization-safe action exists*, **not** C1 capacity rejection
  (`policy/shield.py:334`).
- **Abstention is NOT trivially safe.** For an observed object SKIP retains the map state, AoI keeps growing, and
  **SKIP must itself stay below the localization bound.**

**The actual SKIP decomposition (greedy, 2,638 frames):**

| SKIP cause | frames |
|---|---:|
| No observed object — SKIP legitimately cheaper | 1,828 |
| Observed objects, but **no localization-safe action exists** | **693** |
| Observed objects, SKIP the only safe candidate | 11 |
| Observed objects, a safe SPLIT existed but reward chose SKIP | **1** |

**One frame.** Reward mis-calibration is not the story, so a `w_error` sweep would have been wasted effort.

**A structural hypothesis — STALE-LOCKOUT (not established).** When the map is already stale and nothing satisfies eps,
the degraded shield minimizes the *immediate peak* bound. SKIP is evaluated over a one-tick horizon while sending
carries capture wait + network latency — so **SKIP looks less bad right now even though sending is the only path to
recovery**, and the map gets staler, repeating. The 693-frame count is consistent with this mechanism but does not
identify it causally. Phase 2 must not inherit this frozen-object shield model; its map already extrapolates tracks.

**Also note:** Task B's vulnerable guardrail already raised SPLIT from **3.98% -> 29.38%**, so the old number
describes the *unguarded baseline*, not the current design.

**And a genuine environment/perception issue:** only **743** frames contain observed objects (1,895 do not), while
a further **434** frames contain GT objects but no matched observation — i.e. ~37% of GT-object frames are lost before
the controller ever sees them. That is a perception-coverage gap the controller cannot solve, and it means
whole-corpus percentages dilute the informative subset.

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

**What it tells us inside this replay:** with perfect one-step outcome knowledge, the oracle changes roughly
**1 decision in 64** for a **1.4%** gain, below the registered bar. This characterizes the legacy surrogate; it is
not a bound on a controller operating from causal pre-action signals.

**Honest scope — this matters and we corrected it:** the oracle is **one-step**, evaluated on the states **greedy
visits**, with matched support (`policy/expanded_gate.py:581`) and same-frame post-tail/GT-assisted observations.
It therefore does **not** upper-bound a causal sequential policy that would reach different states. The defensible
claim is *"no useful one-step headroom within the noncausal, static-quality, queue-free, matched-support contract"*
— **not** "all RL is bounded by 1.4%." We initially overstated this and fixed it.

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

**Bottom line: do not train RL now.** Static profile choice has not earned learning, the N=2 OAI admission study
found no coordination gap, and the legacy dynamic ladder cannot settle the causal question. Reopen learning only
if the paired causal Phase-2 ladder leaves a pre-registered residual sequential gap.

**And what is genuinely NOT falsified** — we are careful here:
- **Scene-conditioned quality.** `profile_quality()` takes **no scene argument**, so "different scenes want
  different profiles" was **structurally unrepresentable** in the controller. Task A later found no practical
  reversal on the available class/range contexts, but true occlusion and cyclists remain untested.
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

**Completed desk work:** Task A found no practical reversal on the available class/range contexts; Task B
implemented observed-vulnerable guardrail logic; Task C showed that lambda-supported-hull lookup is not equivalent
to exact enumeration over all 36 profiles. Task B's empirical replay deltas and Task C's runtime half inherit the
noncausal caveat.

**In flight now (design only, no CARLA/OAI):** freeze `phase2_paired_causal_v1`: causal pre-action provenance,
separate inference-placement/publication actions, schema-v2 uncertainty, pre-registered designed and naturalistic
suites, and a two-trajectory pilot gate. The pilot itself remains held for review.

**The critical path to the Month-6 system:** review the Phase-2 causal schema/corpus spec → pass the positive +
benign pilot → collect paired designed and naturalistic suites → establish C2 locally → repeat identical messages
over the existing two-UE OAI system → calibrate minimal LOCAL and run simple causal baselines → navigation
warning/optional override. **~7–10 weeks, 9–12 with contingency.**

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
- The legacy ladder/oracle is **noncausal same-frame post-tail, GT-assisted, queue-free, one-step,
  matched-support**.
- Evaluation is segmentation + pedestrian/vehicle recall & localization — **not true OD AP**; cyclists and small
  objects uncovered.
- Shield soundness is **regional** (sound @25 m, unsound @40 m) and covers **detected** objects only.
- Scene-conditioned perception utility has a scoped null over available class/range contexts; true occlusion and
  cyclists remain untested.

---

## 15. The one-slide summary

> We built a measured, replayable environment for the split-inference decision (24 runs / 8,480 frames / 4 measured
> channel rungs / 36 compression profiles), with a two-layer controller: a hard safety shield plus a
> freshness-and-quality reward. We then climbed a ladder — rule, greedy, contextual bandit, MPC, clairvoyant oracle
> — and pre-registered what would count as a win for learning.
>
> **Learning has not earned its place yet.** Static profile choice is a measured-table problem and, independently,
> the N=2 OAI admission study found the 5G scheduler already at the capacity ceiling. The old dynamic replay also
> showed little headroom, but a causal audit found post-tail/GT-assisted state, so that result is backup-only and
> does not close causal control.
>
> **What does work is load shaping:** payload choice expands the feasible operating region **~12.7×** (7 → 89 of
> 200 cells). The lever is *what you send*, not *how cleverly you schedule it*.
>
> So we are not training RL. We are first building a paired causal helper-recipient path and asking the citable
> question: **does cooperation advance the recipient's warning through the OAI stack, and do simple causal rules
> leave any sequential headroom?**

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
   - the **legacy controller ladder bar chart** with error bars (rule/greedy/LinUCB/MPC) — appendix only, with the
     noncausal same-frame/GT-assisted caveat on the figure,
   - **greedy vs oracle** with the +1.383% gap and the pre-registered 5% bar drawn as a threshold line (the visual
     illustrates the legacy replay result; do not label it a deployable NO-GO),
   - the **deadline-feasibility frontier** (7 / 58 / 89 of 200 cells) — this is the positive result, give it a full
     slide,
   - the **skip-rate / admissibility** chart supporting §7a (why 0.19 is the number it is).
6. **State every scope caveat on the slide that makes the claim**, not only in the limitations slide — especially
   "one-step, matched-support oracle", "N=2 measured / N=50 modeled", "RFsim, not over-the-air".

**Tasks A/B/C have landed.** Build the main narrative around the system, measured surfaces, and Phase-2 causal
plan; keep the legacy ladder/oracle figures in backup until the paired pilot and C2 evidence exist.
