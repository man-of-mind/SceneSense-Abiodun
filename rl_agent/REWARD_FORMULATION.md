# Reward formulation — network-aware split/local-inference controller

**Status:** consensus draft (local Claude + codex, 2026-08-06). Builds on `AGENT_CONSTRAINTS.md §9`,
`POLICY_KICKOFF.md`, `state_diagram.md`, `collab/REVIEW_NOTES.md`. Advisor-pending inputs (do not block
drafting): ε value (default 2.0 m), pedestrian-recall hard-floor?, 25 m vs 40 m eval regime.
**This supersedes the single-blob perception utility in §9.3** (localization now lives in a feasibility-aware
safety term; segmentation/recall stay in the task utility — see §5).

## 1. Decision each control step
`mode ∈ {SPLIT, LOCAL, SKIP}`
- **SPLIT** — run the front backbone, send features → edge intermediate-fusion. Sub-choices: payload knob
  (quant u8→u4 · AE bottleneck none/128/64/32 · ROI) + FPS.
- **LOCAL** — run the FULL model on the vehicle, upload only the result → shared map. Sub-choice: FPS
  (compute-bound). Corner-case fallback for deep fade + fast object.
- **SKIP** — send nothing this step (correct when the dynamic scene is empty).

## 2. State (adds to §9.1)
Existing: channel-budget estimate (+confidence), object speed (+σ), front-side urgency, **AoI**, previous
action+outcome. **NEW — local-compute headroom:** available CPU/GPU budget / current load, or at minimum the
measured **max sustainable full-local FPS** for the configured device. (Local is only feasible with compute
headroom — see §3.)

## 3. Feasibility masks (HARD, applied before the policy)
- **C1 channel mask:** admit a SPLIT action only if `payload × fps ≤ pessimistic(channel-budget estimate)`
  (observation-based; estimate-misses under lag are logged diagnostics, not oracle-prevented).
- **Local-compute mask:** admit LOCAL only if the device can sustain full-local at the required FPS (from the
  §2 headroom). Motivation data: full-local ≈ 5.49 FPS @1 core / 9.96 @2 cores vs split front 15.68 / 25.91 —
  so **LOCAL is not automatically feasible** on a constrained vehicle.
- **SKIP** is always admissible.
- Admitted set `A_m(s)` = actions passing both masks.

## 4. Feasibility-aware SAFETY term (localization) — codex's formulation, adopted
Let `e(a,s) = sqrt( base_loc(mode,knob)² + (speed · AoI)² )` (AoI-composed loc error);
`e*(s) = min_{a∈A_m(s)} e(a,s)`; `F(s)=1` iff some admitted action meets `e ≤ ε`.
```
                ┌  1[e(a,s) ≤ ε]  − clip((e(a,s) − e*(s))/ε, 0, 1)      if F(s)=1
r_safety(a,s) = │
                └                  − clip((e(a,s) − e*(s))/ε, 0, 1)      if F(s)=0
```
- ε achievable → meeting it earns the bonus; worse-than-best still penalized (relative regret).
- ε physically impossible → **only relative-to-best regret; zero penalty for the physics.** This is what
  stops the agent being wrongly punished in the deep-fade/fast-object corner.
- Absolute over-budget error is still **logged** as an operating-envelope result.
- **Note (local Claude):** `e*(s)` is a *training/surrogate* quantity (the surrogate enumerates `A_m`); the
  deployed policy never needs it. And if the binary `1[e≤ε]` step hinders value learning, swap it for a smooth
  saturating bonus (e.g. `σ((ε−e)/ε_scale)`) — keep the regret term as is.

## 5. Ranked reward (normalize every term to a comparable range FIRST)
```
R = 10·r_safety  +  2·U_task  −  C_UE  −  C_PRB  −  0.5·C_ROI  −  0.1·C_switch
```
| # | term | what | why it ranks here |
|---|---|---|---|
| 1 | **r_safety** (×10) | AoI-composed localization feasibility (§4) | the shared map's *job* — a fresh, accurate localization is the safety deliverable. Weight exceeds the sum of the lower terms. |
| 2 | **U_task** (×2) | segmentation mIoU + pedestrian/object recall (**loc NOT here — it's in r_safety; no double-count**) | perception quality ranks above resource optimization |
| 3 | **C_UE**, **C_PRB** (×1) | on-car compute/energy (incl. LOCAL's full-model cost) ; MCS-scaled airtime `payload×fps×(1+retx)/SE(MCS)` (prefer measured PRB-seconds) | physical scarce resources; bite under bad channel + multi-UE contention |
| 4 | **C_ROI** (×0.5) | ROI-escalation last-resort cost, on top of its measured seg loss | keeps ROI a genuine last resort |
| 5 | **C_switch** (×0.1) | mode-switch hysteresis | prevents oscillation without blocking necessary reactions |

Ordering **safety ≫ perception ≫ resource ≫ ROI ≫ switch** mirrors SCAN-AI's own reward
(paper §4.2.3: `λ₂ ≫ λ₁ ≫ λ₃`, reliability > fidelity > smoothness). Weights are **initial priors** —
grid-search + ablate, especially **safety-vs-resource** and **UE-compute-vs-PRB**.

## 6. No arbitrary "local penalty" — physical opportunity cost only
LOCAL's disincentive must come from measured physics, not a hand-tuned constant: `C_UE` (full-model
compute/energy), the compute mask (missed deadlines), `C_switch`, and its worse `base_loc` / lost
cooperative-seg quality reflected in `U_task`. Emergent behavior:
- good channel / slow object → SPLIT wins (lower `base_loc`, offloads compute);
- deep fade + fast object → LOCAL wins (fresh result beats stale cooperative);
- compute-constrained vehicle → LOCAL masked out → least-bad admitted action + flag.
The motivation study concludes split is **not** globally superior — so if a capable vehicle legitimately
prefers LOCAL often, that is a **result to report, not something the reward should hide.**

## 7. LOCAL mode needs a MEASURED profile (a 4th surrogate table) — provisional until then
The 2.27 KB / ~42 ms figure is **detections only**; the shared map likely needs segmentation/map content that
LOCAL must also upload. Before LOCAL is trusted in training, measure a `local_mode` table:
full-local latency + sustainable FPS vs compute budget; UE energy/compute; **actual result payload incl.
seg/map**; delivery latency for that payload; detection/localization/segmentation/cooperative quality. Until
measured, mark LOCAL **provisional** (seed from the split-inference study) and don't let a trained policy rely
on unmeasured LOCAL numbers.

## 8. Algorithm ladder + architecture (validation-first — codex's, adopted)
1. **ORACLE** table lookup — enumerate `A_m(s)`, pick the reward-max action. Upper bound + **reward sanity
   gate: the oracle must choose LOCAL in the deep-fade/high-speed region and reject it when SPLIT is safely
   cheaper — validate this BEFORE any RL.**
2. **Myopic bandit** — non-sequential baseline (can't plan AoI/mode; the number RL must beat).
3. **Masked Double/Dueling DQN.**
4. **Masked discrete/categorical SAC** over the flattened, mask-filtered catalog — **NOT** continuous SAC with
   output rounding (the mode switch is categorical; rounding a continuous action is a known pathology).
5. **Maskable PPO** only if both value-based methods are unstable.
- **Architecture (all algos): adopt SCAN-AI's FiLM** — network/budget state γ,β-*gates* the perception/speed
  features instead of being concatenated as a peer. Matches "channel gates the payload decision"; proven in
  the sibling paper.
- **SAC context:** SCAN-AI used SAC because its action was a *continuous bitrate* and *discrete levels caused
  H.265 I-frame instability* — codec-specific, does not transfer to our discrete/mixed catalog. (The paper
  does **not** explicitly state DQN was tried; it says "discrete bitrate levels." Defer to the advisor if he
  has additional context.) Hence discrete-SAC, not continuous+round.

## 9. Eval metrics ("converged" ≠ just cumulative reward — codex's battery, adopted)
C2 success rate where ε is feasible · regret vs `e*` where ε is infeasible · LOCAL-fallback precision/recall ·
LOCAL-misuse rate in good/slow states · mode-switch frequency · PRB-time + UE-compute cost · seg/recall
retention · generalization across unseen speed × channel × compute traces.

## 10. Consensus record (local Claude ⇄ codex)
**Agreed** — codex's 2026-08-06 proposal is adopted in full; it improved the earlier draft in four places:
(i) LOCAL-compute feasibility mask + compute-headroom state; (ii) split localization into the feasibility-aware
`r_safety` (was bundled in the perception utility); (iii) discrete/categorical SAC rather than continuous-SAC
+ rounding; (iv) oracle-first validation before RL.
**Local-Claude additions:** (a) `e*` is training/surrogate-only, not needed live; (b) offer a smooth `r_safety`
bonus if the binary step hurts value learning; (c) keep FiLM regardless of algo; (d) normalize all terms
before the weights mean anything; (e) LOCAL stays provisional until its measured 4th table exists.
**Open / advisor:** ε=2.0, ped-recall hard floor, 25 vs 40 m, and — for the fusion side — whether LOCAL
(late-fusion detection sharing) retains multi-ego coverage or only loses the triangulation loc refinement
(decides whether a coverage term is needed in `U_task`).
