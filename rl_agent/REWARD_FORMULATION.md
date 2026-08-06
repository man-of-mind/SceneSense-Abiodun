# Reward formulation — network-aware split/local-inference controller (v3)

**Status:** consensus draft **v3** (local Claude + codex, 2026-08-06). v2 fixed the first correction round;
v3 applies round 2 (live safety *shield* vs training-only; tail-risk `E_risk` forms the band vs `E_expected`
for the reward; §8a demoted to hypotheses-vs-measured-oracle; RL-vs-bandit = *anticipatory* effects). Builds on
`AGENT_CONSTRAINTS.md §9`, `POLICY_KICKOFF.md`, `state_diagram.md`, `collab/REVIEW_NOTES.md`. Advisor-pending:
ε (default 2.0 m), ped-recall hard-floor?, 25 vs 40 m. **Not RL-ready until §7's LOCAL 4th table is measured.**

## 1. Decision each control step
`mode ∈ {SPLIT, LOCAL, SKIP}`
- **SPLIT** — front backbone → send features → edge intermediate-fusion. Sub-choice = ε-dominance-pruned knob
  (2 ROI0 seg-safe {ae32/u4 90 KB, ae128/u4 129 KB} + 5 sub-90 KB ROI-escalation) × FPS.
- **LOCAL** — full model on the vehicle → upload result → shared map. Sub-choice = FPS (compute-bound).
- **SKIP** — send nothing (only network-free action).
- **FPS set (config):** `{2, 5, 10, 15, 20}`. All weights/margins/denominators in config.

## 2. State (adds to §9.1)
Channel-budget estimate (+confidence), object speed (+σ), front-side urgency, **AoI**, previous
action+outcome, **local-compute headroom** (available CPU/GPU / load, or ≥ measured max sustainable full-local
FPS for the device).

## 3. Feasibility masks (HARD, before the policy)
- **C1 channel mask — EVERY sending action (SPLIT *and* LOCAL):** `payload(mode,profile) × fps ≤
  pessimistic(channel-budget estimate)`. LOCAL's result is small but **not network-free**.
- **Local-compute mask — LOCAL only:** admit only if the device sustains full-local at the required FPS
  (E1/E2/E6: full-local ≈ 5.5 FPS @1 core / 10 @2 vs split front 15.7 / 25.9 — LOCAL is not auto-feasible).
- **SKIP** always admissible. `A_m(s)` = admitted set.

## 4. Localization error — post-action AoI, multi-object, TWO statistics
Each action is evaluated at its *resulting* AoI (outcome `o` = delivered / dropped):
```
AoI_{t+1}(a,o) = { capture→map latency(a)   if delivered ;  AoI_t + Δt(a)  if skip / drop }
e_j(a,s,o)     = sqrt( base_loc(a)² + (v_j · AoI_{t+1}(a,o))² )         for object j
E(a,s,o)       = aggregate over currently-present dynamic objects: worst-case max_j e_j (default) or CVaR_α
                 (empty scene ⇒ 0 penalty)
```
`base_loc(a)` from the knob matrix via the **monotone (affine) calibration** onto the ~1.1 m live floor
(level shifted, rankings preserved). Each object uses its own speed and the AoI of its most-recent update.

**Two statistics — do NOT conflate (codex round 2):**
- **`E_expected`** = p50 / expected over delivery outcomes → the **reward** (`R_inner`).
- **`E_risk`** = reconstructed full-pipeline **p95** (sensor+front+network+edge+map, not `front_to_edge_p95`
  alone) or outcome-CVaR → forms the **safety band `A_safe`** (§5) and the operating-envelope report.
  **Safety is a TAIL property, not a median.**

The sampled RL reward uses the *realized* outcome; the oracle/bandit use *expected* over the delivery
distribution. Let `E_risk*(s) = min_{a∈A_m(s)} E_risk(a,s)`; `F(s)=1` iff some admitted action meets
`E_risk ≤ ε`.

## 5. Safety as a LIVE shield (structural), not a big weight
A large weight can't guarantee safety dominance (two actions' safety gap can be tiny → a cost term overturns
it). So safety is structural — a **live model-based safety shield** after the hard masks: the **onboard
surrogate enumerates the small catalog, predicts `E_risk` per action, and admits**
```
A_safe(s) = { a ∈ A_m(s) : E_risk(a,s) ≤ ε }                 if F(s)=1   (meet ε on the TAIL)
          = { a ∈ A_m(s) : E_risk(a,s) ≤ E_risk*(s) + δ_loc } if F(s)=0   (near-best band; flag over-budget)
```
`δ_loc` ≈ localization measurement noise. **The shield runs LIVE** (≈10-action catalog → cheap onboard);
`E_risk*` is needed by the shield *at inference* — distinct from the reward signal (see §10). **The policy/RL
optimizes the inner objective (using `E_expected`) only within `A_safe`:**
```
R_inner(a,s) = w_task·U_task(a) − C_UE(a) − C_PRB(a) − 0.5·C_ROI(a) − 0.1·C_switch(a)
             (optionally − small·E_expected(a,s)/ε to prefer margin inside the band)
```
This makes safety **lexicographically dominant** and graceful degradation structural (F=0 → optimize inside
the near-best-tail band + flag). *Scalar-weight ablation to compare:* `R = 10·r_safety + 2·U_task − …` (v1) —
only a soft preference, so validate its safety/resource Pareto.

### 5a. `U_task`
`U_task = w_mIoU·(mIoU/mIoU_ref) + w_ped·(ped_recall/ped_ref) + w_obj·(obj_recall/obj_ref)`, refs from
uncompressed/best-achievable. **Localization is NOT here — it's the safety term/band.** **Map coverage /
cooperative fusion — DEFERRED to phase 2** (map-side edge intelligence): phase-1 `U_task` = this car's own
perception quality; modular hook for phase 2. Caveat: with the cooperative payoff deferred, SPLIT's phase-1
reason to exist is **compute-offload** (LOCAL runs the full model → higher `C_UE`) + the payload/freshness
trade — see §8a for why "SPLIT-first" is a *hypothesis*, not a given.

### 5b. Cost normalization (denominators matter as much as weights)
`C_PRB = PRB-seconds(a) / PRB-second-budget` (measured PRB-time preferred; else `payload×fps×(1+retx)/SE(MCS)`);
`C_UE = compute-or-energy(a) / device-budget`; `C_ROI` = ROI-escalation last-resort cost; `C_switch` =
mode-switch hysteresis. Normalize all to comparable ranges before weighting; grid-search safety-band δ/ε and
UE-compute-vs-PRB.

## 6. No arbitrary LOCAL penalty — physical opportunity cost only
LOCAL's disincentive = `C_UE` + compute mask + `C_switch` + its worse `base_loc`/lost cooperative-seg quality.
LOCAL is a reward-**emergent** last resort, **not a hard rule** — if a capable vehicle legitimately prefers it,
that's a **result**, not a bug (consistent with the split-inference study's conclusion).

## 7. LOCAL is a HYPOTHESIS until a measured 4th table
The 2.27 KB / ~42 ms is **detections-only**; OAI delivery **unmeasured**; feature-vs-detection accuracy/coverage
**not** compared. Compute/energy/FPS is **already measured — reuse E1/E2/E6**; the NEW delta to measure:
**(a) LOCAL's real result payload incl. seg/map, (b) its OAI delivery, (c) LOCAL-vs-SPLIT accuracy.** Mark
LOCAL **provisional** until then. LOCAL expands the feasible set but does **not** remove graceful degradation.

## 8. Algorithm ladder + architecture
1. **ORACLE** (enumerate `A_m`→`A_safe`, pick `R_inner`-max) — upper bound + reward sanity.
2. **Myopic bandit** — reactive baseline.
3. **Masked Double/Dueling DQN.**
4. **Masked discrete/categorical SAC** over the flattened, mask-filtered catalog — **not** continuous-SAC+round.
5. **Maskable PPO** only if both value-based are unstable.
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

**RL vs bandit (corrected):** a contextual bandit that observes AoI can ALREADY do the *reactive* threshold
(skip while fresh, update near ε) — that alone does **not** justify RL. RL earns its keep on genuinely
**anticipatory/sequential** effects: sending *before* a predicted fade; planning around mode-switch cost and
changing compute headroom; and avoiding future **AoI dead-ends** (states from which no admitted action can
recover ε). Those are the bandit-beating behaviors to demonstrate.

## 9. Eval metrics (converged ≠ just reward)
C2 success rate where ε feasible · regret vs `E_risk*` where infeasible · LOCAL-fallback precision/recall vs
the measured oracle · LOCAL-misuse rate (vs oracle, not a preset preference) · mode-switch frequency ·
PRB-time + UE-compute cost · seg/recall retention · generalization across unseen speed × channel × compute
traces · (RL-specific) advantage on anticipatory traces (pre-fade sends, AoI-dead-end avoidance).

## 10. Two distinct uses of `E_risk*` (resolves the §5-vs-§10 apparent contradiction)
(i) The **live safety shield** (§5) needs `E_risk*` *at inference* — the onboard surrogate computes it by
enumerating the ≈10-action catalog (cheap). (ii) The **reward/training signal** does not need it live. So the
deployed policy DOES run the shield (with `E_risk*`) but does not need `E_risk*` for reward updates. Caveat:
if **live online policy updates** are later required, the counterfactual regret in the scalar-ablation
`r_safety` isn't directly observable online — but the shield (action admission) still works.

## 11. Multi-object / AoI precision
AoI is per successfully-published update; with per-object update times, use per-object AoI in `e_j`, else the
map-level AoI is the tractable approximation. Aggregate over *currently-present* dynamic objects only.

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
- Local Claude concurs with all four; no open disagreement. **Next: measure the LOCAL 4th table → build the
  oracle (settle live-shield semantics + tail statistic first) → bandit → DQN/discrete-SAC.**
