# Reward formulation — network-aware split/local-inference controller (v2)

**Status:** consensus draft v2 (local Claude + codex, 2026-08-06). v2 applies codex's corrections — see the
changelog (§12). Builds on `AGENT_CONSTRAINTS.md §9`, `POLICY_KICKOFF.md`, `state_diagram.md`,
`collab/REVIEW_NOTES.md`. **Not yet "converged"** — ready for the 4th-table / oracle phase once §7's LOCAL
profile is measured. Advisor-pending: ε (default 2.0 m), ped-recall hard-floor?, 25 vs 40 m.

## 1. Decision each control step
`mode ∈ {SPLIT, LOCAL, SKIP}`
- **SPLIT** — front backbone → send features → edge intermediate-fusion. Sub-choice = one of the **ε-dominance
  pruned** knobs (2 ROI0 seg-safe {ae32/u4 90 KB, ae128/u4 129 KB} + 5 sub-90 KB ROI-escalation) × FPS.
- **LOCAL** — full model on the vehicle → upload result → shared map. Sub-choice = FPS (compute-bound).
- **SKIP** — send nothing (the only network-free action).
- **FPS set (config):** `{2, 5, 10, 15, 20}`. All weights/margins/denominators live in config.

## 2. State (adds to §9.1)
Existing: channel-budget estimate (+confidence), object speed (+σ), front-side urgency, **AoI**, previous
action+outcome. **NEW — local-compute headroom:** available CPU/GPU / current load, or ≥ the measured max
sustainable full-local FPS for the device.

## 3. Feasibility masks (HARD, before the policy) — **[corrected: C1 covers LOCAL too]**
- **C1 channel mask — applies to EVERY sending action (SPLIT *and* LOCAL):**
  `payload(mode,profile) × fps ≤ pessimistic(channel-budget estimate)`. LOCAL's result is small but **not
  network-free** — it still uploads over the same channel.
- **Local-compute mask — LOCAL only:** admit LOCAL only if the device sustains full-local at the required FPS
  (§2 headroom). Motivation: full-local ≈ 5.5 FPS @1 core / 10 @2 vs split front 15.7 / 25.9 — LOCAL is **not
  automatically feasible**.
- **SKIP** always admissible.
- `A_m(s)` = admitted set (passes the masks that apply to it).

## 4. Localization error — **[corrected: POST-action AoI + multi-object aggregate]**
The action changes AoI, so evaluate each action at its *resulting* AoI (outcome `o` = delivered / dropped):
```
AoI_{t+1}(a,o) = { capture→map latency(a)      if delivered
                 { AoI_t + Δt(a)               if skip / drop
e_j(a,s,o) = sqrt( base_loc(a)² + (v_j · AoI_{t+1}(a,o))² )        for object j
```
- `base_loc(a)` from the knob matrix, via the **monotone (affine) calibration** onto the ~1.1 m live floor
  (level shifted, knob rankings preserved). Use **p50** here for expected reward; **reconstructed
  full-pipeline p95** (sensor+front+network+edge+map, not `front_to_edge_p95` alone) for the envelope report.
- **Multi-object aggregate** over currently-present dynamic objects (a frame has several): default
  **worst-case `E(a,s,o) = max_j e_j`** (safety-conservative); if worst-case is too brittle to a single
  far/mis-tracked object, use **CVaR_α**. **Empty scene ⇒ no objects ⇒ zero localization penalty** (so
  correct SKIP is free). Each object uses its own speed and the AoI of its most-recent update.
- **Outcome handling:** the sampled RL reward uses the *realized* outcome; the oracle/bandit use the
  **expected** value over the delivery distribution (delivery prob from the transport table). Without this,
  SPLIT/LOCAL/SKIP would be scored at the same pre-action staleness — the bug v1 had.

Let `E*(s) = min_{a∈A_m(s)} E(a,s)` (expected), and `F(s)=1` iff some admitted action meets `E ≤ ε`.

## 5. Reward structure — **[corrected: safety BAND, not a "big weight"]**
A large weight does not guarantee safety dominance (two actions' safety difference can be arbitrarily small,
so a resource term can overturn it). So make safety **structural** — a second admission stage after the masks:
```
A_safe(s) = { a ∈ A_m(s) : E(a,s) ≤ ε }              if F(s)=1   (feasible → require meeting ε)
          = { a ∈ A_m(s) : E(a,s) ≤ E*(s) + δ_loc }  if F(s)=0   (infeasible → near-best band; flag over-budget)
```
`δ_loc` ≈ localization measurement noise. **The policy/RL optimizes the inner objective only within
`A_safe`:**
```
R_inner(a,s) =  w_task·U_task(a)  −  C_UE(a)  −  C_PRB(a)  −  0.5·C_ROI(a)  −  0.1·C_switch(a)
             (optionally − small·E(a,s)/ε to prefer margin inside the band)
```
This makes **safety lexicographically dominant** (guaranteed, not hoped), and graceful degradation is now
structural (F=0 → optimize perception/resources among the near-best-loc actions + flag). *Simpler alternative
to compare:* the scalar form `R = 10·r_safety + 2·U_task − C_UE − C_PRB − 0.5·C_ROI − 0.1·C_switch` (v1) —
keep it as an ablation, but validate its safety/resource Pareto since it's only a soft preference.

### 5a. `U_task` (define explicitly)
`U_task = w_mIoU·(mIoU/mIoU_ref) + w_ped·(ped_recall/ped_ref) + w_obj·(obj_recall/obj_ref)`, refs from
uncompressed/best-achievable measurements. **Localization is NOT here — it's the safety term/band** (no
double-count). **Map coverage / cooperative fusion — DEFERRED to phase 2** (Abiodun, 2026-08-06): multi-car
occlusion reasoning + map completeness will live as *map-side intelligence at the edge* in the project's
second half. So **phase-1 `U_task` = this car's own perception quality** — no multi-car coverage term now; it
slots in modularly for phase 2 (closes the earlier open fusion-side question). Caveat: with the cooperative
payoff deferred, phase-1 may favor LOCAL/SKIP slightly more than the full system will — SPLIT's *phase-1*
reason to exist is **compute-offload** (car runs only the backbone; LOCAL runs the full model → higher
`C_UE`) plus the payload/freshness trade, so the LOCAL-vs-SPLIT choice stays well-posed.

### 5b. Cost normalization (denominators matter as much as weights)
- `C_PRB = PRB-seconds(a) / PRB-second-budget` — measured PRB-time preferred; else `payload×fps×(1+retx)/SE(MCS)`.
- `C_UE = compute-or-energy(a) / device-budget` (LOCAL's full-model cost from §7's table).
- `C_ROI` = ROI-escalation last-resort cost (on top of its measured seg loss in `U_task`).
- `C_switch` = mode-switch hysteresis. **Normalize all terms to comparable ranges before weighting**; grid-search
  the weights, especially **safety-band δ / ε** and **UE-compute-vs-PRB**.

## 6. No arbitrary LOCAL penalty — physical opportunity cost only
LOCAL's disincentive = `C_UE` (full-model compute/energy) + the compute mask (missed deadlines) + `C_switch`
+ its worse `base_loc`/lost cooperative-seg quality. Emergent (to be *tested*, not asserted): good channel/slow
→ SPLIT; deep fade + fast → LOCAL; compute-constrained → LOCAL masked → least-bad admitted + flag. If a capable
vehicle legitimately prefers LOCAL often, that is a **result**, consistent with the split-inference study's
conclusion that split is not globally superior.

## 7. LOCAL is a HYPOTHESIS until a measured 4th table exists — **[corrected: don't overstate]**
The 2.27 KB / ~42 ms figure is **detections-only**; LOCAL delivery over OAI was **not** measured; and
feature-sharing vs detection-sharing accuracy/coverage were **not** directly compared. So do **not** assert
"LOCAL always delivers," "SPLIT has better `base_loc`," or "the oracle *must* pick LOCAL in the corner" — those
are **intended oracle-behavior tests under provisional assumptions**, not findings. Measure a `local_mode`
table first. **Scope (2026-08-06): the compute/energy/FPS tradeoff is ALREADY measured — reuse E1/E2/E6.**
The genuinely NEW measurements are: **(a) LOCAL's real result payload incl. segmentation/map content** (E-study
had detections-only, 2.27 KB), **(b) its delivery over OAI**, and **(c) a direct LOCAL-vs-SPLIT accuracy
comparison** (feature-fusion vs detection-sharing). So it's a small delta experiment, not a redo. Mark LOCAL
**provisional** until it exists. **LOCAL is a reward-EMERGENT last resort, not a hard rule:** its higher `C_UE`
(from E1/E2/E6) + its accuracy make the reward disfavor it unless SPLIT is too stale AND SKIP is unacceptable
(fast objects present) — do NOT hard-code "LOCAL only if X" (brittle; and legitimate frequent-LOCAL on a
capable car is a result, not a bug).
**LOCAL expands the feasible set but does not remove graceful degradation** — with poor enough compute + channel,
all modes can still miss ε.

## 8. Algorithm ladder + architecture
1. **ORACLE** (enumerate `A_m`→`A_safe`, pick `R_inner`-max) — upper bound + reward sanity. **Hypothesis
   tests** (not pass/fail facts): does it pick LOCAL in deep-fade/fast and reject it when SPLIT is safely
   cheaper, under the provisional LOCAL table?
2. **Myopic bandit** — non-sequential baseline (can't plan AoI/mode).
3. **Masked Double/Dueling DQN.**
4. **Masked discrete/categorical SAC** over the flattened, mask-filtered catalog — **not** continuous-SAC + rounding.
5. **Maskable PPO** only if both value-based methods are unstable.
- **Architecture — [corrected: FiLM is an ablation, not a given]:** baseline = plain concat MLP; **FiLM
  (network γ,β-gates the app features) as an ablation.** SCAN-AI proved FiLM for *video bitrate*, not this
  controller — keep them separable so a policy failure isn't blamed on architecture.
- **SAC context:** SCAN-AI used continuous SAC because its action was a continuous bitrate and *discrete
  levels* caused H.265 I-frame instability — codec-specific, doesn't transfer to our discrete/mixed catalog.
  (The paper does not explicitly say DQN was tried.)

## 8a. Oracle acceptance tests — expected mode per regime (operationalizes Abiodun's precedence, 2026-08-06)
The reward must produce this precedence **emergently** (no hard rules); the oracle is where we verify it:

| regime | expected mode | why |
|---|---|---|
| good channel, objects present | **SPLIT** | delivers fresh features at acceptable cost (+ phase-2 cooperative gain) |
| bad channel, AoI still fresh (loc ≤ ε w/o update) | **SKIP** (≻ LOCAL) | map still accurate; SKIP is free — don't spend compute/airtime; LOCAL's freshness buys nothing here |
| bad channel, AoI nearing budget, fast object, LOCAL feasible | **LOCAL** | fresh single-view beats stale SPLIT; SKIP would breach ε |
| bad channel, fast object, LOCAL infeasible (compute) | least-bad admitted + flag | graceful degradation |

**Key sequential behavior (why RL > bandit):** *skip while the map is fresh, then update — SPLIT if the
channel allows, else LOCAL — as AoI nears the ε budget.* A myopic bandit can't plan this hold-then-act.
**Object speed, not ego speed:** the corner is driven by the *tracked object's* world speed, so fast NPCs
(~28 mph) with a normal-speed ego realize it — no fast ego needed. The surrogate sweeps object speed as a
state parameter, so the oracle/envelope can probe arbitrary speeds even if live CARLA caps the ego.

## 9. Eval metrics (converged ≠ just reward)
C2 success rate where ε feasible · regret vs `E*` where infeasible · LOCAL-fallback precision/recall ·
LOCAL-misuse rate in good/slow states · mode-switch frequency · PRB-time + UE-compute cost · seg/recall
retention · generalization across unseen speed × channel × compute traces.

## 10. Online-learning caveat
`E*` (and the band) are fine for **surrogate training + a frozen live policy**. If **live online updates** are
later required, counterfactual `E*` is not directly observable — a different, observable online reward would be
needed then.

## 11. Multi-object / AoI precision note
AoI is tracked per successfully-published update; when the map holds per-object update times, use per-object
AoI in `e_j`; else the map-level AoI is the tractable approximation. Aggregate (worst-case / CVaR) over
*currently-present* dynamic objects only.

## 12. Changelog & consensus (v1 → v2, all codex 2026-08-06 corrections ACCEPTED)
1. **POST-action AoI** in the loc term (v1 used pre-action AoI → SPLIT/LOCAL/SKIP looked identical). [bug fix]
2. **C1 channel mask now covers LOCAL** uploads, not just SPLIT. [bug fix]
3. **LOCAL claims marked provisional** — hypotheses under the (unmeasured) 4th table, not findings.
4. **Safety BAND** (structural lexicographic safety) replaces the "×10 weight" as primary; scalar form kept as
   an ablation. [weight ≠ guarantee]
5. **Multi-object aggregation** (worst-case default / CVaR alt; empty ⇒ 0) added — v1 assumed a single object.
6. `U_task`, cost **normalization denominators**, `base_loc` calibration, p50/p95 split, pruned catalog + FPS
   set — all made explicit.
7. **FiLM demoted to an ablation** vs a concat-MLP baseline (not asserted for this controller).
8. Online-learning `E*`-observability caveat added.
Local-Claude concurs with every point; no open disagreement. **Next: measure the LOCAL 4th table, then build
the oracle and validate the hypotheses in §8.1 before any RL.**
