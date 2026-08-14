# Formulation, structural result, baselines, related work, and paper framing

**Purpose.** Capture the mathematics of the controller we actually run, the structural result that explains the
RL NO-GO, the baseline suite a reviewer will ask for, and an honest novelty assessment. Written 2026-08-14 after
`EXPANDED_SURROGATE_NO_GO_STOP`; §8 revised the same day after Abiodun correctly rejected a
findings-as-contributions framing. Notation is plain-text on purpose (no LaTeX macros).

> **Stance:** we do NOT claim the controller as a novel algorithm. We state that we apply a constrained
> threshold/lookup policy, show it is near-optimal against a measured ceiling, and explain why via known theory.
> Forcing algorithmic novelty here would be dishonest and reviewers would see through it.

---

## 1. The controller we actually run

Per decision epoch, state `s` = (capacity estimate mu_hat and its sigma, per-object speeds, map AoI, object
count, unmapped fraction, previous action, in-flight/queue state). Action set `A` = {SKIP} + {SPLIT(profile p,
target fps f)} over the retained catalog (35 SPLIT + SKIP as evaluated; 36 measured profiles before pruning).

Implemented selection (`policy/shield.py:279`, `policy/controllers.py:193`):

```
a*(s) = argmax   R(a, s)
         a in A
  subject to     e_bound(a, s) <= eps          # safety shield (hard)
                 offered(a)    <= kappa*mu_hat # C1 admission (hard)

R(a, s) = w_task * U_hat_task(a)
        - C_PRB(a, s)
        - lambda_switch * 1[mode(a) != mode(a_prev)]
        - w_E * ( E_hat_G(a, s) / eps )
```

Ties break deterministically on `(reward, -bound_m, action_id)`. `C_PRB = offered(a)/capacity`.
`E_hat_G` is the expected max-over-objects localization error, with
`e_j = sqrt( base_loc(a)^2 + (v_j * AoI_map_j)^2 )`.

This is **not** greedy in the knapsack/incremental sense. It is a **myopic constrained argmax over a finite
measured catalog**, one action per epoch.

Key measured property: `U_hat_task(a)` is a **constant per action** — `policy/shield.py:49
profile_quality(action, reward_config)` takes **no observation argument**. Perception utility therefore cannot
depend on scene content in the current implementation. This is the scoping limit of the NO-GO, not a finding about
the world.

---

## 2. Structural result — hull + staircase (derived from measurement, not assumed)

Given (i) `U_task` context-free, (ii) the measured accuracy-vs-payload frontier monotone, and (iii) the
constraint collapsing to a single scalar per-frame budget

```
b(s) = kappa * mu_hat(s) / fps
```

the optimal per-epoch choice reduces to

```
a*(s) = argmax { U(a) : a in hull(catalog), payload(a) <= b(s) }
```

Two statable consequences:

- **R1 — only hull points are ever selected.** Dominated profiles are never optimal for any budget. This is why
  36 measured profiles pruned to 7 with no loss of achievable reward.
- **R2 — the optimal policy is a staircase (piecewise-constant) function of one scalar budget**, with breakpoints
  at hull payloads. The deployable artifact is a **breakpoint table**: `budget -> profile`.

**Lagrangian / dual view.** For each lambda >= 0,

```
a_lambda = argmax_a [ U(a) - lambda * payload(a) ]
```

Sweeping lambda >= 0 traces exactly `hull(catalog)`; the budget constraint selects `lambda*(b)`. Non-hull points
are optimal for no lambda.

**This is rate-distortion optimization (RDO)** with the distortion axis replaced by *task utility* (mIoU,
pedestrian/vehicle recall, localization error) instead of MSE, coupled to an AoI/freshness penalty
`w_E * E_G/eps`. We should present it as such — it is what the math already is.

---

## 3. Why the RL NO-GO is theory-predicted (the strongest framing we have)

Measured: `greedy 0.192625` vs `clairvoyant oracle 0.195290` = **+1.383%** (95% CI [+0.001929, +0.003452]),
below the pre-registered +5% / +0.01 bar; oracle changed only **1.56% of actions** over 12,955 states.

Two known results predict exactly this:

1. **RDO:** budget-constrained selection over a convex hull is a **lambda-threshold rule** — i.e. a lookup. There
   is no sequential structure in the rate/quality decision for a learner to discover.
2. **AoI scheduling:** our freshness term *is* Age of Information. For AoI problems, **index / threshold policies
   (Whittle-type) are provably optimal or near-optimal** across many formulations.

So the result is not an anticlimax. Frame it as: *the index/threshold structure predicted by RDO + AoI theory is
confirmed in a measured 5G cooperative-perception system, and we quantify the achievability gap (1.38%).* The
clairvoyant oracle is an **upper bound on every policy including RL**, so the ceiling argument is airtight for the
evaluated contract.

**Scope honestly (all three are real limits):** static per-profile perception utility (no scene conditioning),
queue-free surrogate for the gate, aggregate-pruned catalog.

---

## 4. Baseline suite — what we have, what is missing

Implemented (`policy/controllers.py`): `fixed`, `rule`, `greedy`, `linucb`, `mpc`, plus the clairvoyant oracle
(expanded-action gate). Missing and worth adding — these are the *principled* baselines from the two literatures
our formulation sits in, and a reviewer will ask:

| Baseline | Origin | Why it matters |
|---|---|---|
| **lambda-RDO / Lagrangian hull lookup** | coding theory | Canonical rate-quality baseline. If our greedy is *equivalent* to it, that is a **result** (R1/R2), not a weakness. |
| **AoI / Whittle-index policy** | freshness scheduling | Canonical freshness baseline; freshness-first rather than utility-first ordering. |
| **max-rate-that-fits** (utility-blind) | strawman | Isolates how much the utility *shape* buys over "send as much as fits". |
| **contextual lookup** (class-mix keyed) | ours, pending | Only if the argmax-stability screen (Task A) finds rank reversals. |

Adding lambda-RDO + Whittle-index lets us claim the lookup matches or beats the principled baselines from **both**
relevant literatures — a much stronger position than greedy-vs-MPC alone.

---

## 5. Design levers (how to influence the lookup's behaviour)

`lambda_switch` (hysteresis / mode-switch cost), `kappa` (C1 pessimism), `eps` (safety bound), `w_task` vs `w_E`
balance, and the task-metric weights (currently 0.35 seg / 0.40 ped / 0.25 veh). Characterising the **breakpoint
table as a function of these** is a legitimate sensitivity contribution and shows the policy is tunable rather
than hand-fitted. Note the pre-registration discipline: freeze before evaluating.

---

## 6. Related work map (VERIFY exact venues/years before citing — recalled from memory)

- **RDO / Lagrangian + convex hull:** Sullivan & Wiegand, *Rate-distortion optimization for video compression*,
  IEEE Signal Processing Magazine ~1998. The hull/lambda-sweep argument behind R1/R2.
- **Age of Information:** Kaul, Yates, Gruteser, *Real-time status: How often should one update?*, INFOCOM 2012;
  Yates et al., *Age of Information: An Introduction and Survey*, JSAC ~2021.
- **AoI scheduling / index policies:** Kadota, Sinha, Modiano — AoI scheduling in wireless networks, Whittle-index
  results. This is the theoretical backing for §3.
- **Split computing / edge inference:** Matsubara, Levorato, **Restuccia**, *Split Computing and Early Exiting for
  Deep Learning Applications: Survey and Research Challenges*, ACM Computing Surveys 2022. (Restuccia is at
  Northeastern — directly relevant to the IDCC x NEU collaboration.)
- **Task-oriented / machine-centric compression:** Choi & Bajic, deep feature compression for collaborative
  intelligence; the Video Coding for Machines (VCM) line; semantic/task-oriented communication surveys
  (e.g. Gunduz et al.).
- **Cooperative perception (the line we are differentiating from):** V2VNet, OPV2V, V2X-ViT, DiscoNet and
  successors — these assume idealized/abstract channels. Plus the V2X CoDriving paper in this repo
  (`V2X_for_AD.pdf`) and the SCAN-AI single-UE foundation (`SCAN_AI_03_13_26_2.pdf`).

---

## 7. Honest novelty assessment (controller only)

**Not novel:** the constrained argmax itself; RDO; AoI; split computing. Do not claim these.

**Novel enough to state:** the *coupling* of a measured task-RDO surface with AoI/freshness under a *measured* 5G
uplink — RDO work ignores freshness; AoI work uses abstract packet models rather than measured task utility. But
this is a supporting result, not the paper's contribution. See §8.

---

## 8. Paper framing — the Month-6 target

**Thesis-level gap.** Cooperative-perception research (V2VNet, OPV2V, V2X-ViT, DiscoNet, ...) almost universally
assumes an idealized or abstract channel: feature sharing "just works." Our claim:

> **Cooperative perception's design assumptions do not survive contact with a real 5G uplink.** We build the
> end-to-end multi-modal system over a real 5G stack, show what is actually achievable, and derive the design
> rules and safety guarantees that follow.

Everything in §8.3 below is **evidence for that thesis, not the contribution itself.** An earlier draft of this
section listed those measurements as the top-line contributions; that was wrong — they are process findings from
building the system, and a paper spined on them reads as "here is what we measured while debugging."

### 8.1 Target contributions (Month 6)

- **C1 — The system.** An end-to-end, safety-shielded, network-aware cooperative perception pipeline: multi-modal
  (RGB + radar) feature sharing over a real 5G stack, producing a shared spatial map, at real-time rates. The novel
  combination is split inference + multi-modal fusion + real 5G transport + an explicit safety constraint + a
  shared map. No prior system puts all five together.
- **C2 — The cooperation gain, quantified under real network conditions.** What cooperation buys that a single
  vehicle cannot obtain: occlusion recovery, extended effective range, and localization improvement (two-view
  triangulation at 1.40 m, beating radar) — measured under *real transport*, not an ideal channel. This is the
  "why cooperate at all" evidence, which the literature reports only in simulation.
- **C3 — A safety-and-network-aware guarantee.** The system guarantees localization error <= eps, or degrades
  gracefully / abstains, and we characterise exactly when that is achievable (the feasibility envelope).
  Cooperative-perception papers report mAP; almost none answer *"can I guarantee my error is under eps in time to
  act on it?"* That framing is ours.
- **C4 — Design rules that make it deployable.** The `budget -> profile` breakpoint table, payload/FPS/knob-vs-
  channel rules, and the demonstration that a **measured lookup suffices** — with the oracle ceiling showing we
  are not leaving performance on the table.

### 8.2 Banked vs pending (as of 2026-08-14)

| Component | Status |
|---|---|
| C1 system spine (split inference, 5G transport, shield, map) | **Largely banked** |
| C2 cooperation gain — two-view triangulation (1.40 m) | **Banked** |
| C2 occlusion recovery | **Phase 2 — not built** |
| C3 guarantee + feasibility envelope | **Banked** (shield + frontier) |
| C4 design rules + lookup sufficiency + oracle ceiling | **Banked** |
| Phase-2 map sharing (recipient-specific, warning timeliness) | **Not built** |
| Navigation override (Month 6) | **Not built** |
| Multi-vehicle end-to-end integration | **Not built** |
| Scene-conditioned knob selection (Phase-1 hypothesis) | **Untested** — Task A pending |
| Vulnerable-object guardrails | **Partially missing** — Task B |

**The gap to a strong paper is Phase 2 + end-to-end integration** — which is what months 4-6 are for. The plan is
sound; we are not off-track.

### 8.3 Supporting findings (evidence and design rationale, NOT headline contributions)

- **The radio is not the bottleneck.** Capture-to-map decomposition: sensor preparation is 57-65% of the budget,
  uplink only ~9% (`staleness/uplink_only_latency_budget/`, L = 67-93 ms p50). Motivates C4 and reframes where
  optimisation effort belongs.
- **Stack-level uplink root cause + fix.** UE RLC queue-wait driven by the QPSK/MCS cap in the gNB UL scheduler
  (`gNB_scheduler_ulsch.c`); SINR-driven UL MCS policy cuts RTT 186 -> 48 ms. Evidence that the "real stack"
  claim in the thesis is load-bearing.
- **Task-aware compression erases the transport penalty.** RTT 209 -> 77 ms, delivery 75% -> 99%.
- **Measured design surfaces.** Knob matrix (accuracy <-> payload <-> latency, transport-invariant), channel sweep,
  staleness/FPS localization requirement, and the segmentation finding (ROI drop destroys segmentation, so the
  seg-aware knob is density-invariant at ae32/u4/ROI0 ~90 KB).
- **Deadline-feasibility frontier.** 7/200 feasible cells at 400 KiB -> 58/200 at 90 KB -> 89/200 at 49.4 KB; even
  the smallest payload leaves ~55% infeasible. Directly supports C3/C4.
- **Falsification with a measured ceiling.** Lookup within 1.38% of a clairvoyant oracle; no N=2 coordination gap
  (MAC scheduler already at the capacity ceiling, 6.090 vs 6.077 Mbps).

### 8.4 Why the RL NO-GO helps a systems paper

For a systems venue, "a measured lookup suffices" is **stronger** than "we trained an RL agent": the controller is
simple, explainable, deterministic, and deployable, with a proof that it is within 1.38% of optimal. The
falsification becomes a design justification rather than a disappointment. Do not bury it, and do not apologise
for it.

### 8.5 Limitations to disclose up front (reviewers will find them anyway)

- **RFsim PHY, not over-the-air.** Real OAI protocol stack, real scheduler, measured channel models — but not a
  real radio. **This is the single biggest acceptance risk at MobiSys/MobiCom**, which strongly prefer real hardware.
- CARLA simulation rather than physical vehicles/sensors.
- Measured contention only at **N=2**; N=50/100 is modelled (and the model's first version had a max-min contract
  bug, since corrected).
- The oracle gate ran on a **queue-free** surrogate.
- Evaluation is segmentation + pedestrian/vehicle recall & localization, **not true OD AP**; cyclists and small
  objects are not covered.
- **Scene-conditioned perception utility is untested** (`profile_quality` has no scene argument).

### 8.6 Venue judgement (honest) — decide with the advisor BEFORE writing

The contributions are MobiSys/MobiCom-shaped, but the absence of an over-the-air component is a real acceptance
risk there. Two paths:
- **(a)** Add an OTA leg (USRP / real gNB) so C1/C2 are hardware-backed. New experimental work, but it is exactly
  what makes C1/C2 top-tier.
- **(b)** Target a venue matched to a simulation-plus-real-stack measurement study (MSWiM, SECON, WoWMoM, VNC; or
  TMC / IoT-J for the journal route) and reserve MobiSys/MobiCom for the OTA version.

This decides whether months 4-6 must include new hardware experiments, so it is the first thing to settle.
