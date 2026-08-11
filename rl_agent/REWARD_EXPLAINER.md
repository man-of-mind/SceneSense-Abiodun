# Reward formulation — plain-language explainer (for the advisor discussion)

Companion to the formal, authoritative `REWARD_FORMULATION.md` (v4). This is the intuition-first walkthrough:
what each symbol means, why each term exists, the pilot values, and the open questions to brainstorm.
Math is in plain monospace (this repo/terminal does not render LaTeX).

## The one-sentence idea
Each frame, among the actions that are **safe enough** (a hard filter), pick the one with the highest
**benefit minus cost** — benefit = how much it improves the shared map; cost = airtime + compute + risky
shortcuts + thrashing.

## The key design decision: two layers
- **Layer 1 — the shield (hard safety):** discard any action that would let localization error exceed the
  budget `epsilon`. Safety is a yes/no filter.
- **Layer 2 — the reward (soft optimization):** among the survivors, maximize benefit minus cost.

**Why split it:** if safety were just one term in a weighted sum, a big enough benefit could outvote it and
"buy" an unsafe action. A hard filter means safety cannot be traded away. (The textbook alternative is a soft
"Lagrangian" penalty — be ready to say why we chose hard masking.)

## The safety side: localization error
```
e_j = sqrt( base_loc(a)^2  +  ( v_j * AoI_map_j )^2 )
```
| Symbol | Meaning | Intuition |
|---|---|---|
| `e_j` | localization error for object j (meters) | how far off the map's position for that object is |
| `base_loc(a)` | model's at-rest error for the compression level in action a | even a fresh detection isn't exact; heavier compression -> bigger base error (~1.1 m floor) |
| `v_j` | object j's speed (m/s) | fast objects drift faster |
| `AoI_map_j` | Age-of-Information: time since the map was last updated for object j | a stale entry drifts — the object moved, the map didn't |

**Why sqrt-of-squares:** two independent error sources — the detector's inherent error and drift from
staleness (speed x age) — combine like the legs of a right triangle. Sending resets `AoI_map_j` to ~0, which
kills the drift term. That is *why* sending helps.

Aggregate over all in-view objects, then take the tail over uncertain outcomes:
```
G          = max over objects j of  e_j        (empty scene => G = 0)

E_expected = mean over outcomes of  G          (the typical error)
E_risk     = p95 (or CVaR) over outcomes of G  (the bad-case error)   <- safety uses THIS
B          = E_risk + k * sigma_hat            (conservative bound; k = uncertainty margin)

safe set   = { action a : B <= epsilon }       (else: flagged "least-bad" degradation band)
```
- **Why `max` over objects:** the scene is not safe if even one object is badly localized. (Conservative —
  drives ~40% of frames infeasible even with perfect info. Softer alternative: quantile / CVaR over objects.)
- **Why the tail (p95), not the mean:** we care about bad cases, not the average.

## The reward (optimized only within the safe set)
```
R =    w_task       * U_task            (benefit: map quality)
     -               C_PRB              (airtime / network cost)
     -               C_UE               (on-car compute / energy)
     - lambda_ROI  * C_ROI              (penalty for risky ROI cropping)
     - lambda_switch * C_switch         (anti-thrashing / mode-switch)
     - w_E         * (G / epsilon)      (small nudge toward lower error)

U_task = 0.50*(mIoU/mIoU_ref) + 0.25*(ped_recall/ped_ref) + 0.25*(obj_recall/obj_ref)
```
| Symbol | Meaning | Intuition |
|---|---|---|
| `U_task` | map perception quality (seg + pedestrian + object), each normalized so best ~= 1.0 | how *useful* the map is. **Post-action**: a dropped send earns nothing |
| `C_PRB` | radio airtime ~ payload x fps / link efficiency (Track A: offered_rate / true_capacity) | don't congest the shared 5G link |
| `C_UE` | on-car compute / energy | running the model + encoding isn't free (makes LOCAL costly vs SPLIT) |
| `C_ROI` | penalty for cropping the region-of-interest to save bytes | an accuracy-risky shortcut -> discourage unless necessary |
| `C_switch` | penalty for changing mode (send/skip/local) | stability; avoid flip-flopping every frame |
| `w_E*(G/epsilon)` | tiny bias toward lower error among *already-safe* actions | a tiebreaker, NOT the safety mechanism (the shield is) |

## Values (pilot defaults — all "up for discussion")
| Parameter | Symbol | Value | Note |
|---|---|---|---|
| Localization budget | `epsilon` | 2.0 m | the headline safety target |
| Task weights | — | 0.50 / 0.25 / 0.25 | seg / pedestrian / object |
| References | — | 0.840 / 0.887 / 0.910 | best-achievable mIoU / ped / obj |
| Benefit weight | `w_task` | 1.0 | benefit is the main driver |
| Airtime weight | `lambda_PRB` | 1.0 | full-strength cost |
| ROI weight | `lambda_ROI` | 0.50 | half |
| Switch weight | `lambda_switch` | 0.10 | small |
| Margin weight | `w_E` | 0.05 | tiny nudge |
| Uncertainty margin | `k` | 1.0 (-> 0 in surrogate) | inert in-surrogate; lives at live-validation |
| Channel pessimism | — | 0.70 | admit only if payload x fps <= 0.7 x estimated capacity |
| Base loc floor | — | ~1.11 m | model's at-rest localization |

## Advisor brainstorm agenda (the genuinely unsettled things)
1. **`epsilon` value + context-dependence?** 2.0 m is a guess; even perfect info hits it only ~54% of the time.
   Maybe 2.5 m, or scale it with speed/scenario (tighter near pedestrians).
2. **Task-weight split (0.50/0.25/0.25).** Is segmentation really 2x a pedestrian? Pedestrians are
   safety-critical — more weight, or a hard floor (a constraint, not a reward term)? **See the "Proposed
   revision" section below** for a concrete pedestrian-weighting + hard-protection proposal.
3. **Hard shield vs soft (Lagrangian) safety.** We chose hard masking — right publishable argument, or add the
   soft-constraint ablation?
4. **`max` over objects.** Conservative (drives the ~40% infeasibility). Softer: quantile / CVaR over objects —
   which matches "the scene is safe"?
5. **Credit this car's contribution, or the map's overall quality?** The latter is the phase-2 multi-car
   framing — confirm the phase-1 choice.

## Proposed revision UNDER DISCUSSION (2026-08-11) — pedestrian weighting + hard protection
Not locked; a proposal to take to the advisor. Motivation (Abiodun): pedestrians are safety-critical
("we do not want to cause accidents"), so they should weigh at least as much as vehicles, and the object term
should be split into explicit classes.

**Proposed U_task (benefit side):**
```
U_task = 0.35 * segmentation_quality
       + 0.40 * pedestrian_recall
       + 0.25 * vehicle_recall
```
- Split the old lumped `obj_recall` into explicit `pedestrian_recall` + `vehicle_recall`, pedestrians highest.
- **Localization deliberately NOT added to U_task.** It is already handled on the safety side
  (`e_j = sqrt(base_loc^2 + (v_j*AoI)^2)`, the shield, and the small `w_E*(G/epsilon)` nudge). Putting it in
  U_task too would DOUBLE-COUNT the same physical quantity — the exact thing v4 removed. Loc stays on the
  safety side (where it is arguably the most-protected quantity), not in the benefit.

**The stronger safety lever (the real point):** a bigger *soft weight* is only a preference — a large enough
network-cost saving can still outvote it. For an accident-avoidance guarantee, protect pedestrians on the
*hard-constraint* side, not (only) the benefit side:
- **Tighter budget for pedestrians:** e.g. `epsilon_pedestrian = 1.0-1.5 m` vs `epsilon_general = 2.0 m`. The
  shield then structurally refuses to let a pedestrian go stale — it cannot be traded for airtime.
- **And/or a hard pedestrian-recall floor:** a constraint ("ped recall >= X"), not a reward term.
- Difference: soft weight = "agent *prefers* to protect pedestrians"; hard constraint = "agent *cannot* neglect
  pedestrians." For "don't cause accidents," you want the second.

**Practical caveat:** heavy `pedestrian_recall` weighting only bites once pedestrian detection is solid (the
current 200k-pps detector-config fix). Land that first, or the term is near-zero and gives little signal.

> Formal spec + changelog: `REWARD_FORMULATION.md` (v4, authoritative). MDP diagram: `state_diagram.md`.
