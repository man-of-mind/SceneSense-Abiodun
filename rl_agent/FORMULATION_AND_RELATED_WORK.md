# Formulation, structural result, baselines, related work, and paper framing

**Purpose.** Capture the mathematics of the controller we actually run, the structural result that explains the
RL NO-GO, the baseline suite a reviewer will ask for, and an honest novelty assessment. Written 2026-08-14 after
`EXPANDED_SURROGATE_NO_GO_STOP`. Notation is plain-text on purpose (no LaTeX macros).

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

Key measured property (see §5 of this doc and `PERMODEL_KNOB_MATRIX_ZSTD.md`): `U_hat_task(a)` is a **constant
per action** — `policy/shield.py:49 profile_quality(action, reward_config)` takes **no observation argument**.
Perception utility therefore cannot depend on scene content in the current implementation. This is the scoping
limit of the NO-GO, not a finding about the world.

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
- **Cooperative perception:** the V2X CoDriving paper in this repo (`V2X_for_AD.pdf`) plus the SCAN-AI
  single-UE foundation (`SCAN_AI_03_13_26_2.pdf`).

---

## 7. Honest novelty assessment

**Not novel:** the constrained argmax itself; RDO; AoI; split computing. Do not claim these.

**Plausibly novel and sufficient for a systems/measurement paper:**
1. The **coupling** of a *measured* task-RDO surface with AoI/freshness under a *measured* 5G uplink — RDO work
   ignores freshness; AoI work uses abstract packet models rather than measured task utility.
2. The **measured design surfaces** (knob x channel rung x payload x deadline x N) for RGB+radar fusion
   cooperative perception over a real OAI 5G stack.
3. The **deadline-feasibility frontier** as a design rule (7/200 feasible cells at 400 KiB -> 58/200 at 90 KB ->
   89/200 at 49.4 KB; note even the smallest payload leaves ~55% infeasible).
4. A **falsification with a measured achievability ceiling** — rare in a field dominated by "we applied RL and it
   improved things".

---

## 8. Paper framing for MobiSys / MobiCom

**Ordering matters: lead with the measurement findings, not the falsification.**

- **C1 — The radio is not the bottleneck.** Capture-to-map latency decomposition: sensor preparation is 57-65% of
  the budget while the uplink is only ~9% (`staleness/uplink_only_latency_budget/`, L = 67-93 ms p50). Plus the
  deadline-feasibility frontier: feasibility is governed by **payload choice**, not by coordination or scheduling
  cleverness. This overturns the common framing that cooperative perception is radio-limited.
- **C2 — Stack-level uplink findings with fixes.** (a) Root cause of UE uplink latency: RLC queue-wait driven by
  the QPSK/MCS cap in the gNB UL scheduler (`gNB_scheduler_ulsch.c`); a SINR-driven UL MCS policy cuts RTT
  186 -> 48 ms. (b) Task-aware feature compression erases the ~5x transport penalty (RTT 209 -> 77 ms, delivery
  75% -> 99%). Concrete, actionable, reproducible.
- **C3 — Measured task-utility/rate/freshness surfaces + a design rule.** The knob matrix (accuracy <-> payload <->
  latency, transport-invariant), the channel sweep, the staleness/FPS localization requirement, and the
  segmentation finding (ROI drop destroys segmentation, so the seg-aware knob is density-invariant at
  ae32/u4/ROI0 ~90 KB). Deliverable: the `budget -> profile` breakpoint table.
- **C4 — Adaptive/learned control is unnecessary here, and we prove the ceiling.** A threshold/hull lookup is
  within **1.38%** of a clairvoyant oracle (below a pre-registered 5% bar); at N=2 there is no coordination gap
  because the MAC scheduler already operates at the capacity ceiling (6.090 vs 6.077 Mbps). Explained by RDO +
  AoI index-policy theory. Pre-registered gates and an oracle upper bound make this a *result*, not an absence.

**Limitations to disclose up front (reviewers will find them anyway):**
- **RFsim PHY, not over-the-air.** Real OAI protocol stack, real scheduler, measured channel models — but not a
  real radio. This is the single biggest risk at MobiSys/MobiCom, which strongly prefer real hardware.
- CARLA simulation rather than physical vehicles/sensors.
- Measured contention only at **N=2**; N=50/100 is modelled (and the model's first version had a
  max-min contract bug, corrected).
- The oracle gate ran on a **queue-free** surrogate.
- Evaluation is segmentation + pedestrian/vehicle recall & localization, **not true OD AP**; cyclists and small
  objects are not covered.
- **Scene-conditioned perception utility is untested** (`profile_quality` has no scene argument) — Task A pending.

**Venue judgement (honest).** The measurement contributions are MobiSys/MobiCom-shaped, but the absence of an
over-the-air component is a real acceptance risk at those venues. Two paths: (a) add an OTA leg (USRP/real gNB) to
make C1/C2 hardware-backed, or (b) target a venue better matched to a simulation-plus-real-stack measurement study
(e.g. MSWiM, SECON, WoWMoM, VNC, or TMC/IoT-J for the journal route) and keep MobiSys/MobiCom for the version with
OTA. Decide this with the advisor before writing, because it changes how much of C1/C2 needs new experiments.
