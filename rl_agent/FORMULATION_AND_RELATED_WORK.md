# Formulation, structural hypotheses, baselines, related work, and paper framing

**Purpose.** Capture the mathematics of the controller we actually run, what the NO-GO does and does not establish,
the baseline suite a reviewer will ask for, and an honest contribution/status assessment.

**Revision history.** v1 (2026-08-14, local Claude) framed measurement findings as the paper's contributions —
rejected by Abiodun as process findings. v2 reframed around the Month-6 system. **v3 (2026-08-14) integrates
codex's code audit, which corrected several overstatements in v1/v2 §§1-3 and §8.** Corrections are marked
`[v3 correction]` so the earlier error is visible rather than silently rewritten. Notation is plain-text (no LaTeX).

> **Stance:** we do NOT claim the controller as a novel algorithm. We apply a constrained threshold/lookup policy,
> report what it achieves against a measured reference, and relate it to known theory *without* claiming the theory
> proves our case. Forcing algorithmic novelty here would be dishonest and reviewers would see through it.

---

## 1. The controller we actually run

Per decision epoch, state `s` = (capacity estimate mu_hat and its sigma, per-object speeds, map AoI, object count,
unmapped fraction, previous action, in-flight/scheduler-credit summaries). Action set `A` = {SKIP} +
{SPLIT(profile p, target fps f)} (35 SPLIT + SKIP as evaluated; 36 measured profiles before pruning).

Implemented selection (`policy/shield.py:127` C1 admission, `:266` safe-set + graceful degradation, `:279` argmax;
`policy/controllers.py:193`):

```
a*(s) = argmax   R(a, s)
         a in A
  subject to     offered_mbps(a) <= pessimism * estimated_capacity   # C1 admission (hard)
                 e_bound(a, s)   <= eps                             # localization-safe set (hard)

R(a, s) = w_task * U_hat_task(a, s)
        - C_PRB(a, s)
        - lambda_switch * 1[mode(a) != mode(a_prev)]
        - w_E * ( E_hat_G(a, s) / eps )
```

Ties break deterministically on `(reward, -bound_m, action_id)`. `E_hat_G` is the expected max-over-objects
localization error, `e_j = sqrt( base_loc(a)^2 + (v_j * AoI_map_j)^2 )`.

This is a **finite, myopic constrained argmax over a measured catalog**, one action per epoch — not greedy in the
knapsack/incremental sense.

**`[v3 correction]` Three fixes to v1/v2:**
1. `C_PRB` is **averaged across capacity samples**, not simply `offered/current_capacity`.
2. **Expected task utility is NOT globally constant per action.** The *selected profile's quality scores* are
   static (`policy/shield.py:49 profile_quality(action, reward_config)` takes no observation), but **realized
   expected utility is state-dependent** through delivery probability and retained map quality. The correct narrow
   claim is: *perception quality is not conditioned on **scene content** (no class mix, confidence, range, or
   occlusion in `controllers.py:206 FEATURE_NAMES`)* — not that utility is context-free.
3. The observation carries scheduler credit and in-flight summaries but **no modeled shared network queue**
   (`policy/types.py:97`).

---

## 2. Structural HYPOTHESES (not results) — hull and staircase

**`[v3 correction]` v1/v2 stated these as derived results. They are not established. They are hypotheses for
Task C to test.**

The C1 payload-feasibility test can be written, **after FPS is fixed**, as a scalar budget:

```
b(s) = pessimism * mu_hat(s) / fps
```

**H1 (staircase).** *If* utility were context-free and the choice depended only on `b`, the optimal policy would be
a piecewise-constant `budget -> profile` staircase with breakpoints at frontier payloads.

**Why H1 does NOT currently hold.** `b` describes only payload feasibility once FPS is fixed. The full action
choice jointly depends on profile **and** FPS, plus speed, AoI, base localization error, latency/delivery, prior
map state, pending frames, safety feasibility, and switching cost. **The controller does not collapse to a
one-scalar policy.**

**H2 (hull sufficiency).** *If* the problem were scalar-utility budget-constrained selection, only upper-convex-hull
points would ever be selected.

**Why H2 is not established (and the precise reason matters):**
- The seven retained profiles are **hard-coded** (`policy/catalog.py:14`) and were produced by **tolerance-aware,
  five-objective epsilon-dominance plus intentionally retained ROI-escalation profiles** — *not* a scalar-utility
  convex-hull derivation (see `collab/REVIEW_NOTES.md:624` for the original pruning contract).
- Therefore **"36 -> 7 without loss of achievable reward" is UNPROVEN.**
- **Budget-constrained enumeration can select non-supported Pareto points; a lambda sweep returns only *supported*
  upper-convex-hull vertices. These are not generally the same set.** (Non-supported Pareto points sit in the
  duality gap and are unreachable by any lambda.) So "greedy == lambda-RDO" cannot be assumed — it must be measured.

**Lagrangian view (still worth stating, as the baseline definition):** for lambda >= 0,
`a_lambda = argmax_a [ U(a) - lambda * payload(a) ]`; sweeping lambda traces the supported hull.

The connection to **rate-distortion optimization** — task utility (mIoU, recall, localization) on the distortion
axis instead of MSE, coupled to an AoI/freshness penalty — is a legitimate *framing*, and lambda-RDO is the right
baseline. It is not a proof about our controller.

---

## 3. What the NO-GO does and does not establish

Measured: `greedy 0.192625` vs expanded-space oracle `0.195290` = **+1.383%** (95% CI [+0.001929, +0.003452]),
below the pre-registered +5% / +0.01 bar; oracle changed only **1.56% of actions** over 12,955 states.

**`[v3 correction]` v1/v2 claimed the oracle is "an upper bound on every policy including RL." That is WRONG.**
The expanded oracle is exact only for **one-step reward on the states greedy actually visits** — the
implementation advances greedy and scores the oracle counterfactually on that trajectory
(`policy/expanded_gate.py:581`). It therefore does **not** upper-bound a sequential policy that reaches different
future states.

**Defensible claim:**
> No useful one-step coordination headroom was found within the static-quality, queue-free, matched-support
> contract.

**Not claimable:** that all RL policies are bounded within 1.38%.

**`[v3 correction]` AoI theory does not predict our result either.** Kadota et al. (ToN 2018) prove guarantees for a
particular unreliable-broadcast scheduling model with **separable clients**; our action sends **one frame that can
refresh multiple objects** and couples profile, FPS, delivery, quality, and safety. The literature is *relevant
context*, not a theorem for this controller. Cite it as related work, not as an explanation of the measurement.

**Scope limits to state every time the NO-GO is mentioned:** static per-profile quality scores (no scene
conditioning), queue-free surrogate, matched-support/one-step evaluation, epsilon-dominance-pruned catalog.

---

## 4. Baseline suite

Implemented (`policy/controllers.py`): `fixed`, `rule`, `greedy`, `linucb`, `mpc`, plus the expanded-gate oracle.

**`[v3 correction]` Task C redesigned per codex — measure, do not assume:**

| Baseline | Purpose |
|---|---|
| **Exact measured-table budgeted enumerator** | Ground truth for "what does full enumeration pick?" — the reference H2 must be tested against. |
| **lambda-RDO supported-hull lookup** | The canonical coding-theory baseline. Compare against the enumerator and report **action agreement, reward gap, and any Lagrangian duality gap.** |
| **max-rate-that-fits** (utility-blind) | Isolates what the utility *shape* buys over "send as much as fits". |
| **AoI-index-inspired heuristic** | **Do NOT call this "Whittle-index"** unless indexability and per-object arm decomposition are established. A genuine Whittle baseline belongs in **Phase-2 object-selective map sharing**, where individual objects are natural scheduling arms. |
| **contextual lookup** (class-mix keyed) | Only if the argmax-stability screen (Task A) finds rank reversals. |

---

## 5. Design levers

`lambda_switch` (hysteresis), `pessimism`/kappa (C1 conservatism), `eps` (safety bound), `w_task` vs `w_E`, and the
task-metric weights (0.35 seg / 0.40 ped / 0.25 veh). Characterising the selected-action structure as a function of
these is a legitimate sensitivity contribution. Pre-registration discipline: freeze before evaluating.

---

## 6. Related work (verified by codex 2026-08-14)

- **RDO:** Sullivan & Wiegand, *Rate-Distortion Optimization for Video Compression*, IEEE Signal Processing
  Magazine 15(6), 74-90, 1998.
- **AoI:** Kaul, Yates, Gruteser, *Real-time Status: How Often Should One Update?*, IEEE INFOCOM 2012, 2731-2735.
  Survey: Yates et al., *Age of Information: An Introduction and Survey*, IEEE JSAC 39, 1183-1210, 2021.
- **AoI scheduling / index policies:** Kadota, Sinha, **Uysal-Biyikoglu, Singh**, Modiano, IEEE/ACM Transactions on
  Networking 26(6), 2637-2650, 2018 (arXiv:1801.01803). *(v1 mis-cited the author list.)*
- **Split computing:** Matsubara, Levorato, **Restuccia** (Northeastern — relevant to the IDCC x NEU collab), ACM
  Computing Surveys 55(5), Article 90, 2022 (arXiv:2103.04505).
- **Task-oriented compression:** Choi & Bajic — collaborative object detection, ICIP 2018 (arXiv:1802.03931);
  near-lossless collaborative intelligence, MMSP 2018 (arXiv:1804.09963).
- **Semantic/task-oriented communication:** Gunduz et al., *Beyond Transmitting Bits*, IEEE JSAC 41(1), 5-41, 2023.
- **Cooperative perception (the line we differentiate from):** V2VNet (ECCV 2020), OPV2V (ICRA 2022), V2X-ViT
  (ECCV 2022), DiscoNet (NeurIPS 2021). Plus `V2X_for_AD.pdf` and the SCAN-AI foundation
  (`SCAN_AI_03_13_26_2.pdf`) in this repo.

---

## 7. Honest novelty assessment (controller only)

**Not novel:** constrained argmax, RDO, AoI, split computing. Do not claim these.

**Statable:** the *coupling* of a measured task-utility/rate surface with AoI/freshness under a measured OAI 5G
protocol stack — RDO work ignores freshness; AoI work uses abstract packet models rather than measured task
utility. Supporting result, not the paper's contribution. See §8.

---

## 8. Paper framing — the Month-6 target

### 8.0 Thesis `[v3 correction — codex wording adopted]`

v1/v2 said cooperative-perception work assumes "feature sharing just works." **That is inaccurate:** V2VNet models
~25 Mbps and derives transmission delay from message size; V2X-ViT uses ~27 Mbps plus synthetic 0-200 ms
asynchrony. They *do* model communication. The accurate gap is that they do not model a **live protocol stack,
scheduler, queues, attach/routing failures, and application-to-map timing together.** Adopted thesis:

> Algorithmic cooperative-perception benchmarks commonly abstract communication as fixed-rate links, synthetic
> delay/noise, or message-size proxies. We build an instrumented, multi-modal cooperative-perception pipeline over
> the **OAI 5G protocol stack (RFsim)**, quantify its end-to-end safety/freshness envelope, and derive deployable
> coordination rules.

**Say "OAI 5G protocol stack over RFsim," never unqualified "real 5G uplink," until OTA exists.**

### 8.1 Target contributions (revised per codex)

- **C1 — The system.** An instrumented, safety-shielded, network-aware multi-modal (RGB + radar) cooperative
  perception pipeline over the OAI 5G stack, producing a shared spatial map. **Not "banked" until Phase-2
  information reaches the recipient/map end-to-end.** **Avoid "first" / "no prior system"** without a systematic
  review — v2 made that claim and it is unsupported.
- **C2 — Transport-conditioned cooperation gain.** *The binding contribution.* What cooperation buys that a single
  vehicle cannot get, measured **through the stack**. **`[v3 correction]` The 1.40 m two-view triangulation result
  is NOT yet C2 evidence:** it was two **static** CARLA egos with **oracle association** and **no OAI transport**
  (`cooperative_fusion/RESULTS_phase2_two_view.md:5`). It is promising groundwork; the transport-conditioned
  version must still be produced.
- **C3 — Safety contract and feasibility envelope.** `[v3 correction]` **Do not say "guarantee."** Say: *the system
  enforces a conservative, model-based action contract and quantifies violations, abstention, and graceful
  degradation.* C1-estimate misses, `ucb_k=0`, detector misses, and limited object scope preclude an unconditional
  localization guarantee (cf. shield sound @25 m / unsound @40 m).
- **C4 — Deployable design rules.** A **measured policy table / feasibility envelope**. Call it a compact
  `budget -> profile` breakpoint lookup only once Tasks A/C establish when that is equivalent to the full
  enumerator (see H1/H2, §2).

### 8.2 Banked vs pending (2026-08-14, corrected)

| Component | Status |
|---|---|
| Instrumented pipeline spine (split inference, OAI transport, shield, map) | **Largely built**, not end-to-end complete |
| Two-view triangulation 1.40 m | **Groundwork only** — static egos, oracle association, no transport |
| C3 action contract + feasibility envelope | **Built**, with quantified unsoundness regions |
| C4 measured policy table + one-step reference gap | **Built** (scope-limited per §3) |
| Phase-2 recipient-specific map sharing, end-to-end | **Not built — BINDING CONSTRAINT** |
| Multi-vehicle end-to-end integration | **Not built** (same path as Phase 2 if scoped to one helper + one recipient) |
| Occlusion recovery | **Not built** |
| Navigation warning/override | **Not built** |
| Scene-conditioned knob selection | **Untested** — Task A |
| Vulnerable-object guardrails | **Partially missing** — Task B (protects only *observed* pedestrians; detector misses stay outside shield knowledge) |

### 8.3 Supporting results (evidence for C1-C4, not contributions)

- **`[v3 correction]` Bottleneck migration — replaces the invalid standalone "radio is not the bottleneck" claim.**
  The 57-65% sensor-preparation figure is from **ideal loopback with no OAI**, and its own caveat says the ranking
  changes over OAI (`staleness/uplink_only_latency_budget/results/UPLINK_ONLY_STALENESS_RESULTS.md:7,:378`).
  Correct framing: *sensor preparation dominates the optimized uncongested path; OAI queueing dominates under large
  payloads or poor channels; task-aware compression moves the system back into a compute/sensing-limited regime.*
  This is both defensible and more interesting than the original claim.
- **Stack-level uplink root cause + fix.** UE RLC queue-wait driven by the QPSK/MCS cap in the gNB UL scheduler
  (`gNB_scheduler_ulsch.c`); SINR-driven UL MCS policy cuts RTT 186 -> 48 ms.
- **Task-aware compression erases the transport penalty.** RTT 209 -> 77 ms; delivery 75% -> 99%.
- **Measured design surfaces.** Knob matrix (transport-invariant), channel sweep, staleness/FPS localization
  requirement, and the segmentation finding (ROI drop destroys segmentation; seg-aware knob density-invariant at
  ae32/u4/ROI0 ~90 KB).
- **Deadline-feasibility frontier.** 7/200 feasible cells at 400 KiB -> 58/200 at 90 KB -> 89/200 at 49.4 KB; even
  the smallest payload leaves ~55% infeasible.
- **Scope-limited falsification.** No useful one-step headroom vs the matched-support oracle (+1.38%); no N=2
  coordination gap (MAC scheduler at the capacity ceiling, 6.090 vs 6.077 Mbps).

### 8.4 Why the scoped NO-GO helps a systems paper

For a systems venue, "a measured policy table suffices" beats "we trained an RL agent": simple, explainable,
deterministic, deployable, with a quantified gap to a one-step reference. Do not bury it and do not apologise for
it — but state its scope every time (§3).

### 8.5 Limitations to disclose up front

- **RFsim PHY, not over-the-air.** Real OAI protocol stack, scheduler, and measured channel models — but not a real
  radio. Biggest acceptance risk at MobiSys/MobiCom.
- CARLA simulation, not physical vehicles/sensors.
- Measured contention at **N=2** only; N=50/100 modelled (first model version had a max-min contract bug, corrected).
- Oracle gate: **queue-free, one-step, matched-support**.
- Segmentation + pedestrian/vehicle recall & localization; **not true OD AP**; cyclists and small objects uncovered.
- Scene-conditioned perception utility untested.
- Shield soundness is **regional** (sound @25 m, unsound @40 m), not unconditional.

### 8.6 Critical path and schedule (codex estimate, 2026-08-14)

**Binding constraint: Phase-2 recipient-specific map sharing, integrated end-to-end.** Phase 2 and multi-vehicle
integration are effectively the same path if scoped to one helper/source + one recipient. **OTA is a venue decision
and a parallel risk, not the first dependency** — stabilise Phase 2 over RFsim before moving to hardware.

1. Paper/venue contract + corrected formulation — **2-3 days**
2. Tasks A, B, corrected C — **1-2 weeks**
3. Canonical local Phase-2 path (association/tracking or explicitly scoped target IDs, shared-map publication,
   recipient selection, warning timing) — **2-3 weeks**
4. Same path over the existing two-UE OAI RFsim system — **1-2 weeks**
5. Navigation warning/override evaluation, no RL — **1-2 weeks**
6. Replicates, ablations, figures, packaging — **2-3 weeks**

**Realistic RFsim-backed completion: 7-10 weeks; 9-12 weeks with integration contingency.** OTA adds ~2-4 weeks
*only* if a known-good USRP/gNB/UE setup and experienced support already exist; otherwise 6-10+ weeks and high
schedule risk.

> **OPEN QUESTION FOR ABIODUN — schedule feasibility.** The repo/context records the IDCC x NEU collaboration as
> ending **2026-08-29**, which is ~2 weeks from this revision, against a 7-12 week critical path. Either the
> collaboration window has been extended, or the 6-month proposal covers a *forthcoming* period, or the scope must
> be cut hard to fit. **This must be resolved before committing to the plan above** — it changes what is
> achievable more than any technical decision in this document.
