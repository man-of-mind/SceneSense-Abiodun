# Reward formulation — network-aware split/local-inference controller (v4)

**Status:** consensus draft **v4** (local Claude + codex, 2026-08-06). v4 makes the live shield
observation-only and uncertainty-aware, makes the small expected-error margin mandatory, fixes the
multi-object/tail operation order, requires one shared shield across deployable baselines, and adds dual
oracles + MPC to test whether RL is needed at all. Builds on `AGENT_CONSTRAINTS.md §9`, `POLICY_KICKOFF.md`,
`state_diagram.md`, `collab/REVIEW_NOTES.md`. Advisor-pending: ε (default 2.0 m), ped-recall hard-floor?, 25
vs 40 m. **Not RL-ready until §7's LOCAL 4th table is measured.**

## 1. Decision each control step
`mode ∈ {SPLIT, LOCAL, SKIP}`
- **SPLIT** — front backbone → send features → edge intermediate-fusion. Sub-choice = ε-dominance-pruned knob
  (2 ROI0 seg-safe {ae32/u4 90 KB, ae128/u4 129 KB} + 5 sub-90 KB ROI-escalation) × FPS.
- **LOCAL** — full model on the vehicle → upload result → shared map. Sub-choice = FPS (compute-bound).
- **SKIP** — send nothing (only network-free action).
- **FPS set (config):** `{2, 5, 10, 15, 20}`. Track A has **35 SPLIT actions + one whole-frame SKIP = 36
  actions**. All weights/margins/denominators are declared in config.

## 2. State (adds to §9.1)
Channel-budget estimate (+confidence), object speed (+σ), front-side urgency, **per-object shared-map AoI**,
previous action+outcome, **scheduler phase + observable in-flight summary**, **local-compute headroom**
(available CPU/GPU / load, or ≥ measured max sustainable
full-local FPS for the device). Preserve repeatable per-object contribution provenance as specified in
`PHASE2_FORWARD_COMPAT.md`; a phase-1 controller may consume fixed-size summaries without discarding the raw
records. Denote the lagged/noisy observable state `s_obs`; the live shield and every deployable controller
receive the same `s_obs`. Latent simulator truth is retained only for outcome generation and the
separately-labelled clairvoyant upper bound (§8).

## 3. Feasibility masks (HARD, before the policy)
- **C1 channel mask — EVERY sending action (SPLIT *and* LOCAL):** `payload(mode,profile) × fps ≤
  pessimistic(channel-budget estimate from s_obs)`. LOCAL's result is small but **not network-free**. Neither
  the mask nor the live shield may read the simulator's true current channel/capacity.
- **Local-compute mask — LOCAL only:** admit only if the device sustains full-local at the required FPS
  (E1/E2/E6: full-local ≈ 5.5 FPS @1 core / 10 @2 vs split front 15.7 / 25.9 — LOCAL is not auto-feasible).
- **SKIP** always admissible. `A_m(s_obs)` = admitted set.

## 4. Localization error — post-action AoI, multi-object, ORDERED statistics
Each action is evaluated at its *resulting per-object shared-map AoI*. In phase 1, outcome `o` is the UE's
delivery/drop result; the same publication normally updates every object included in that frame. The schema
already permits phase-2 peer contributions to update objects independently:
```
AoI_{map,j,t+1}(a,o) = { capture→map latency of newest valid contribution for j, if one is published
                       { AoI_{map,j,t} + Δt(a), otherwise
e_j(a,s,o)     = sqrt( base_loc(a)² + (v_j · AoI_{map,j,t+1}(a,o))² )   for object j
G(a,s,o)       = aggregate over currently-present dynamic objects: max_j e_j (default; object-tail aggregate
                 is a configured robustness alternative); empty scene ⇒ G=0
E_expected(a,s)= E_o[G(a,s,o)] (use a labelled p50 proxy only when an outcome mean cannot be reconstructed)
E_risk(a,s)    = p95_o[G(a,s,o)] or CVaR_{α,o}[G(a,s,o)]
```
`base_loc(a)` from the knob matrix via the **monotone (affine) calibration** onto the ~1.1 m live floor
(level shifted, rankings preserved). Each object uses its own speed and the age of its newest valid map
contribution; a fresh update for object A never resets object B.

**The operation order is normative:** aggregate objects for each outcome first (`G`), then take expectation,
p95, or outcome-CVaR. In general `p95_o[max_j e_j] != max_j p95_o[e_j]`; implementations must not swap the
operations.

**Two statistics — do NOT conflate:**
- **`E_expected`** = expected `G` over delivery outcomes (p50 only as a labelled proxy) → a small mandatory
  within-band reward margin (§5).
- **`E_risk`** = reconstructed full-pipeline **p95** (sensor+front+network+edge+map, not `front_to_edge_p95`
  alone) or outcome-CVaR → the target tail statistic. Its conservative observable-state bound `B` forms the
  live **safety band `A_safe`** (§5); true `E_risk` supports the operating-envelope report. **Safety is a
  TAIL property, not a median.**

The sampled RL reward uses realized `G(a,s,o)`; oracle/bandit/MPC expected scoring uses `E_expected`. True
`E_risk` and `E_risk* = min_{a∈A_m} E_risk` are evaluation quantities; the deployable shield uses the
conservative observable-state estimates below.

The same outcome discipline applies to task quality. Define `U_task_post(a,s,o)` from the resulting **map**:
a delivered result installs the selected profile's quality for captured objects; a drop or SKIP retains each
object's prior valid contribution quality; a new object without a contribution is unobserved; empty scene is
zero. A dropped send must never earn the selected profile's accuracy. Contributions therefore retain a
`profile_id` and task-quality snapshot in addition to their provenance fields.

## 5. Safety as a LIVE shield (structural), not a big weight
A large weight can't guarantee safety dominance (two actions' safety gap can be tiny → a cost term overturns
it). So safety is structural — a **live model-based safety shield** after the hard masks: the onboard
surrogate enumerates the small catalog using only `s_obs`. For each action, form a calibrated conservative
bound (UCB shown; conformal/quantile bounds are acceptable):
```
B(a,s_obs) = E_hat_risk(a,s_obs) + k·sigma_hat(a,s_obs)       # E_risk^UCB
B*(s_obs)  = min_{a∈A_m(s_obs)} B(a,s_obs)
F_hat(s_obs)=1 iff some a∈A_m(s_obs) has B(a,s_obs) ≤ ε

A_safe(s_obs) = { a ∈ A_m : B(a,s_obs) ≤ ε }                 if F_hat=1
              = { a ∈ A_m : B(a,s_obs) ≤ B*(s_obs)+δ_loc }   if F_hat=0 (flag over-budget)
```
Calibrate `k`, `sigma_hat`, and `δ_loc` on held-out traces; `δ_loc` covers the jointly measured
localization + surrogate/tail-model uncertainty without double-counting the UCB margin. Report shield
**false-admission** and **false-rejection** rates. If an action/state is outside the surrogate's calibrated
support, reject actions without a valid conservative bound. If none can be bounded, enter a flagged
`shield_ood` degraded mode and use a fixed worst-case-risk fallback over `A_m` (do not assume SKIP or LOCAL is
always safest).

**The shield runs LIVE** (36-action Track A catalog → still cheap onboard). The policy optimizes only within
`A_safe`.
`U_task` and physical costs are the main drivers; `w_E>0` is a declared, mandatory small within-band margin
bias:
```
R_inner_sample(a,s,o) = w_task·U_task_post(a,s,o) − C_UE(a) − C_PRB(a) − 0.5·C_ROI(a) − 0.1·C_switch(a)
                        − w_E·G(a,s,o)/ε                       # sampled RL transition
R_inner_expected(a,s) = w_task·E_o[U_task_post(a,s,o)] − C_UE(a) − C_PRB(a) − 0.5·C_ROI(a) − 0.1·C_switch(a)
                        − w_E·E_expected(a,s)/ε                # oracle/bandit/MPC scoring
```
This makes safety **lexicographically dominant** and graceful degradation structural (`F_hat=0` → optimize
inside the near-best conservative band + flag). Every deployable controller in §8 must call the **same mask
and shield implementation** with the same catalog, `s_obs`, surrogate, calibration, and `δ_loc`.
*Scalar-weight ablation to compare:* `R = 10·r_safety + 2·U_task − …` (v1) — only a soft preference, so
validate its safety/resource Pareto.

### 5a. `U_task`
For each installed contribution, its quality snapshot is
`U_profile = w_mIoU·(mIoU/mIoU_ref) + w_ped·(ped_recall/ped_ref) + w_obj·(obj_recall/obj_ref)`, refs from
uncompressed/best-achievable. **Localization is NOT here — it's the safety term/band.** **Map coverage /
cooperative fusion — DEFERRED to phase 2** (map-side edge intelligence): phase-1 `U_task` = this car's own
post-action map perception quality averaged over currently present objects; modular hook for phase 2. Caveat:
with the cooperative payoff deferred, SPLIT's phase-1
reason to exist is **compute-offload** (LOCAL runs the full model → higher `C_UE`) + the payload/freshness
trade — see §8a for why "SPLIT-first" is a *hypothesis*, not a given.

### 5b. Cost normalization (denominators matter as much as weights)
`C_PRB = PRB-seconds(a) / PRB-second-budget` (measured PRB-time preferred; Track A uses the dimensionless
realized airtime fraction `offered_rate/true_capacity`; later use `payload×fps×(1+retx)/SE(MCS)` when needed);
`C_UE = compute-or-energy(a) / device-budget`; `C_ROI` = ROI-escalation last-resort cost; `C_switch` =
mode-switch hysteresis. Normalize all to comparable ranges before weighting; grid-search safety-band δ/ε and
UE-compute-vs-PRB.

### 5c. Phase-1 in-surrogate shield basis — UCB is INERT here (2026-08-10, Step-A safety-calibration result)
The `B = E_hat_risk + k·sigma_hat` UCB above is the **design** and stays in code, but in the phase-1 **surrogate**
it carries no leverage and must not be presented as a calibrated live margin:
- The Step-A grid (`policy/SAFETY_CALIBRATION_RESULTS.md`, 25 cells) showed numerical-zero `sigma_hat` for every
  C1-admitted/raw-safe/selected action at the adopted 0.70 floor (`max_selected_risk_sigma_m` was numerical zero)
  → varying `ucb_k ∈ [0,2]` changed **no** safe set or action on any of 1,699 frames. Some C1-rejected
  candidates do have nonzero ensemble spread, so `sigma_hat` is not globally zero. It is inert after C1 because
  the floor equals the minimum capacity multiplier: every admitted send delivers in all modeled outcomes, and
  a zero-spread fallback (SKIP / certain-delivery SPLIT) always exists.
- **Phase-1 operating values: `ucb_k = 0`, `c1_pessimism_factor = 0.70`.** The honest phase-1 shield basis is the
  **hard C1 mask + the deterministic p95 localization tail** (speed via 1.645σ, latency-p95) — NOT a tunable UCB.
  The 0.70 C1 value is a conservative engineering convention matching the modeled -30% capacity floor, not a
  statistically calibrated optimum; its observed 1/80 miss count has a wide descriptive Wilson interval.
  `ucb_k = 1.0` would pretend an inert margin; use `0`.
- The UCB/`sigma_hat` machinery activates only once a **validated residual/conformal** uncertainty model exists
  (a real prediction residual — i.e., **live validation**, not the deterministic surrogate composition). Defer
  `k`/`sigma_hat` calibration to that phase; the §5 interface is unchanged.
- The completed 4×3 estimator-quality sensitivity **falsified estimator lag/noise as the driver** of the ~42%
  full-GT false-reject rate at this fixed point: `(lag=0, noise=0)` recovered 0.00 percentage points versus the
  baseline, and the entire grid spanned only 0.10 points. Raw-safe sets changed on as many as 332/1,699 frames,
  but selections changed on at most 13. The remaining gap mixes speed uncertainty, observation mismatch,
  worst-object aggregation, and map-state trajectory; it is not yet attributable or irreducible.
- **Evidence caveat:** the vehicle-only, ~94%-SKIP replay yields a thin SPLIT denominator (Wilson upper ~20%),
  so no "calibrated-zero false-admit" is claimable in-surrogate regardless of knobs.

## 6. No arbitrary LOCAL penalty — physical opportunity cost only
LOCAL's disincentive = `C_UE` + compute mask + `C_switch` + its worse `base_loc`/lost cooperative-seg quality.
LOCAL is a reward-**emergent** last resort, **not a hard rule** — if a capable vehicle legitimately prefers it,
that's a **result**, not a bug (consistent with the split-inference study's conclusion).

## 7. LOCAL is a HYPOTHESIS until a measured 4th table
The 2.27 KB / ~42 ms is **detections-only**; OAI delivery **unmeasured**; feature-vs-detection accuracy/coverage
**not** compared. Compute/energy/FPS is **already measured — reuse E1/E2/E6**; the NEW delta to measure:
**(a) LOCAL's real result payload incl. seg/map, (b) its OAI delivery, (c) LOCAL-vs-SPLIT accuracy.** Mark
LOCAL **provisional** until then. LOCAL expands the feasible set but does **not** remove graceful degradation.

## 8. Controller ladder + architecture — simplest that works
1. **Clairvoyant true-state oracle (non-deployable):** true latent state/outcome model; upper bound only.
2. **Shielded observation-based one-step oracle (deployable benchmark):** enumerate the shared
   `A_m→A_safe` using `s_obs`, then maximize `R_inner_expected`. The gap to the clairvoyant oracle is the
   price of observability, telemetry lag, and conservative uncertainty.
3. **Hand-written rule/greedy baseline:** explicit capacity/AoI thresholds; no fitted policy and no full
   reward enumeration. This is distinct from the learned contextual bandit.
4. **Contextual bandit:** learned one-step/reactive policy; no transition lookahead.
5. **Shielded MPC / receding-horizon controller:** short-horizon lookahead over the surrogate's channel, AoI,
   switch-cost, and compute-headroom dynamics; replan from `s_obs` each step and pass every proposed action
   through the shared live shield. This is the interpretable anticipatory baseline.
6. **Masked Double/Dueling DQN.**
7. **Masked discrete/categorical SAC** over the flattened, mask-filtered catalog — **not**
   continuous-SAC+round.
8. **Maskable PPO** only if both value-based methods are unstable.

Except for the explicitly clairvoyant upper bound, **all controllers share exactly the same action catalog,
observable inputs, C1/local-compute masks, risk surrogate, uncertainty calibration, `δ_loc`, and `A_safe`
implementation.** Controller comparisons must isolate action selection, not safety handling.

- **Adoption rule:** use the simplest controller that captures the value. Prefer shield + rule/MPC if it
  matches RL, because it is easier to inspect and certify. Adopt RL as the proposed controller only if it
  beats both the contextual bandit **and MPC** on held-out §9 anticipatory metrics with comparable safety.
  Either outcome is publishable: “RL adds sequential value” or “a model-based controller suffices.”
- **Architecture:** baseline = concat MLP; **FiLM (network γ,β-gates app features) as an ABLATION** (proven for
  video bitrate, not this controller — keep separable so a policy failure isn't blamed on architecture).
- **SAC context:** SCAN-AI used continuous SAC because its action was a continuous bitrate and *discrete
  levels* caused H.265 I-frame instability — codec-specific, doesn't transfer to our discrete/mixed catalog.
  (Paper does not explicitly state DQN was tried.)

## 8a. Oracle acceptance tests — HYPOTHESES vs the MEASURED oracle (not pass conditions)
Whether the intended precedence holds **depends on the measured LOCAL `C_UE`/accuracy/payload (§7)**. On a
*capable* vehicle LOCAL may legitimately beat SPLIT even in a good channel (tiny payload + affordable compute)
— that's a **result, not misuse**. Define **"LOCAL misuse" relative to the measured oracle's optimum, NOT a
preset SPLIT preference.** With that caveat, the *hypothesized* pattern to check:

| regime | hypothesized mode | contingent on |
|---|---|---|
| good channel, objects present | SPLIT | measured `C_UE`/accuracy/payload making SPLIT reward-optimal |
| bad channel, AoI still fresh (`E_risk ≤ ε` w/o update) | SKIP (≻ LOCAL) | SKIP is free → wins the band unless staleness breaches ε |
| bad channel, AoI nearing budget, fast object, LOCAL feasible | LOCAL | LOCAL's measured fresh loc beats stale SPLIT |
| bad channel, fast object, LOCAL infeasible (compute) | least-bad admitted + flag | graceful degradation |

**RL vs bandit vs MPC:** a contextual bandit that observes AoI can already implement the *reactive* threshold
(skip while fresh, update near ε). MPC can also capture the principal **anticipatory/sequential** effects:
sending before a predicted fade, planning around mode-switch cost and changing compute headroom, and avoiding
future **AoI dead-ends** (states from which no admitted action can recover ε). RL is justified only by a
measured advantage beyond both reactive bandit and short-horizon MPC, not by hold-then-act alone.

## 9. Eval metrics (converged ≠ just reward)
C2 success rate where ε is truly feasible · regret vs true `E_risk*` where infeasible · shield false-admit /
false-reject rates on held-out traces · `shield_ood` rate · clairvoyant-vs-shielded-oracle gap · LOCAL-fallback
precision/recall vs the measured shielded oracle · LOCAL-misuse rate (vs oracle, not a preset preference) ·
mode-switch frequency · PRB-time + UE-compute cost · seg/recall retention · generalization across unseen
speed × channel × compute traces · controller regret · anticipatory-trace performance (pre-fade sends,
AoI-dead-end avoidance). Report bandit-vs-MPC-vs-RL with the identical shield; RL adoption requires a
statistically supported gain over both bandit and MPC at comparable safety.

## 10. Live conservative `B*` vs evaluation-only true `E_risk*`
(i) The **live safety shield** (§5) enumerates the ≈10-action catalog on `s_obs` and computes the conservative
`B* = min_a B(a,s_obs)` at inference. It never receives true current capacity or true counterfactual risks.
(ii) True `E_risk* = min_a E_risk(a,s)` is available in the surrogate/evaluation harness for feasibility,
regret, false-admit/false-reject scoring, and the clairvoyant oracle; it is **not a deployment input**.
(iii) The sampled reward needs realized `G`, not either minimum. If live online policy updates are later
required, counterfactual regret remains unobservable online, but the observation-based shield still works.

## 11. Multi-object / AoI precision
The canonical state stores `AoI_map,j` per tracked object, derived from that object's repeatable contribution
records (`PHASE2_FORWARD_COMPAT.md`). In phase 1, a derived scalar summary may be convenient because one
frame normally refreshes all included objects, but it must not replace the per-object records or become the
environment API. Aggregate over *currently-present* dynamic objects only. The required order is per-outcome
object aggregation `G` first, then expectation/p95/outcome-CVaR (§4). `SKIP` remains frame-level: a fresh peer
contribution can stop object *j* from binding, but whole-frame `SKIP` must still pass the aggregate shield.

## 12. Changelog & consensus
- **v2 (round 1, all codex corrections):** post-action AoI [bug]; C1 mask covers LOCAL [bug]; LOCAL provisional;
  safety band replaces "×10 weight"; multi-object aggregation; explicit `U_task`/normalization/calibration/
  p50-p95/pruned-catalog; FiLM→ablation; online-learning caveat.
- **v3 (round 2, all codex corrections):**
  1. **Safety band is a LIVE model-based shield** (onboard surrogate computes `E_risk*`); resolved the
     live-vs-training-only contradiction with §10. [fix]
  2. **`E_risk` (tail p95/CVaR) forms `A_safe`; `E_expected` (p50) drives the reward** — safety on the tail,
     not the median. [fix]
  3. **§8a demoted to hypotheses vs the measured oracle** — "good channel ⇒ SPLIT" is NOT a pass condition;
     LOCAL may legitimately win on a capable vehicle; "LOCAL misuse" defined vs the measured oracle. [local
     Claude over-claimed "correct & emergent" — conceded]
  4. **RL-vs-bandit justification = anticipatory/sequential effects**, since a bandit CAN do reactive
     hold-then-act given observable AoI. [local Claude over-claimed — conceded]
- **v4 (final non-blocking guardrails + controller-family framing):**
  1. Mandatory small normalized `w_E>0` margin: realized `G` for sampled RL, `E_expected` for expected
     controller scoring; task utility + physical costs remain the main inner-reward drivers.
  2. Live shield is observation-only and uncertainty-aware (`B=E_hat_risk+k·sigma_hat` or calibrated
     equivalent), with OOD degraded mode and held-out false-admit/false-reject evaluation.
  3. Multi-object aggregation order fixed: build per-outcome `G` first, then p95/CVaR across outcomes.
  4. Every deployable baseline uses the identical masks + shield; clairvoyant and shielded oracles are
     separately labelled.
  5. Added hand-written rule, contextual bandit, and shielded MPC before RL; adopt the simplest controller
     that works, and require RL to beat both bandit and MPC on anticipatory metrics.
  6. Forward-compatible schema clarification: canonical AoI is per-object shared-map age with repeatable
     contribution provenance; scalar AoI is only a phase-1 derived summary, and `SKIP` remains frame-level.
- Local Claude and codex concur; no open disagreement. **Next: measure the LOCAL 4th table → build both
  oracles + rule/bandit → MPC → DQN/discrete-SAC only if the simpler controllers leave sequential value.**
