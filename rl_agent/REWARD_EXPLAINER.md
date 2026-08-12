# Reward formulation — plain-language explainer (for the advisor discussion)

Companion to the formal, authoritative `REWARD_FORMULATION.md` (v5). This is the intuition-first walkthrough:
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
j_G        = argmax object j                  (the freshness-driving object)

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
     - lambda_switch * C_switch         (anti-thrashing / mode-switch)
     - w_E         * (G / epsilon)      (small nudge toward lower error)

U_task = 0.35*(mIoU/mIoU_ref) + 0.40*(ped_recall/ped_ref) + 0.25*(vehicle_recall/vehicle_ref)
```
| Symbol | Meaning | Intuition |
|---|---|---|
| `U_task` | map perception quality (seg + pedestrian + vehicle), each normalized so best ~= 1.0 | how *useful* the map is. **Post-action**: a drop installs no new profile and retains prior valid quality |
| `C_PRB` | radio airtime ~ payload x fps / link efficiency (Track A: offered_rate / true_capacity) | don't congest the shared 5G link |
| `C_UE` | on-car compute / energy | running the model + encoding isn't free (makes LOCAL costly vs SPLIT) |
| `C_switch` | penalty for changing mode (send/skip/local) | stability; avoid flip-flopping every frame |
| `w_E*(G/epsilon)` | tiny bias toward lower error among *already-safe* actions | a tiebreaker, NOT the safety mechanism (the shield is) |

ROI cropping remains in the action catalog, but v5 gives it no separate cost. Its measured segmentation and
recall damage already lowers `U_task`, so another penalty would count the same downside twice.

## Values (pilot defaults — all "up for discussion")
| Parameter | Symbol | Value | Note |
|---|---|---|---|
| Localization budget | `epsilon` | 2.0 m | the headline safety target |
| Task weights | — | 0.35 / 0.40 / 0.25 | seg / pedestrian / vehicle |
| References | — | measured best-achievable per metric | mIoU / pedestrian recall / vehicle recall |
| Benefit weight | `w_task` | 1.0 | benefit is the main driver |
| Airtime weight | `lambda_PRB` | 1.0 | full-strength cost |
| Switch weight | `lambda_switch` | 0.10 | small |
| Margin weight | `w_E` | 0.05 | tiny nudge |
| Uncertainty margin | `k` | 1.0 (-> 0 in surrogate) | inert in-surrogate; lives at live-validation |
| Channel pessimism | — | 0.70 | admit only if payload x fps <= 0.7 x estimated capacity |
| Base loc floor | — | ~1.11 m | model's at-rest localization |

## Advisor brainstorm agenda (the genuinely unsettled things)
1. **`epsilon` value + context-dependence?** 2.0 m is a guess; even perfect info hits it only ~54% of the time.
   Maybe 2.5 m, or scale it with speed/scenario (tighter near pedestrians).
2. **Pedestrian hard protection.** v5 now gives pedestrian recall the largest soft weight, but should a tighter
   class-specific localization budget or hard recall floor make that protection non-tradeable?
3. **Hard shield vs soft (Lagrangian) safety.** We chose hard masking — right publishable argument, or add the
   soft-constraint ablation?
4. **`max` over objects.** Conservative (drives the ~40% infeasibility). Softer: quantile / CVaR over objects —
   which matches "the scene is safe"?
5. **Credit this car's contribution, or the map's overall quality?** The latter is the phase-2 multi-car
   framing — confirm the phase-1 choice.

## Post-meeting consensus (2026-08-11, advisor-endorsed) — formalized in v5
These changes are now part of the authoritative v5 equations and the locked §9 policy specification.

**(a) U_task split into explicit classes, pedestrians >= vehicles:**
```
U_task = 0.35 * segmentation_quality
       + 0.40 * pedestrian_recall
       + 0.25 * vehicle_recall
```
Split the old lumped `obj_recall` into explicit `pedestrian_recall` + `vehicle_recall`, pedestrians highest
(safety-critical). Advisor listed exactly these three map-quality components.

**(b) Drop the explicit ROI-cost term; let the agent learn it implicitly (advisor).** Remove
`- lambda_ROI * C_ROI` from the reward. Reason (same logic as localization): ROI-cropping's downside already
shows up as lower `U_task` (worse segmentation/recall — our density study proved ROI-drop destroys segmentation),
so a separate C_ROI penalty double-counts it. The agent learns to avoid ROI because it tanks U_task.

**(c) Localization stays on the safety side (confirmed).** NOT in U_task — the shield
(`e_j = sqrt(base_loc^2 + (v_j*AoI)^2)`, plus the small `w_E*(G/epsilon)` nudge) already carries it. Adding it to
U_task would double-count.

**Reward v5:**
```
R =    w_task       * U_task            (map quality: 0.35 seg / 0.40 ped / 0.25 vehicle)
     -               C_PRB              (airtime / network)
     -               C_UE               (on-car compute / energy; penalizes not using SPLIT)
     - lambda_switch * C_switch         (anti-thrashing)
     - w_E         * (G / epsilon)      (small nudge toward lower error; j_G = freshness-driving object)
```
(ROI cost removed vs v4.)

**Naming: `j_G` is the "freshness-driving object" (not "worst object"); `G` is its error.** It is the object
whose position goes stale fastest, so it sets when the map must be refreshed — see `REWARD_LOOP_DIAGRAM.md` for
the 2-3 object worked frame.

**The stronger safety lever (open, for a future meeting):** a bigger *soft weight* is only a preference — a
large enough cost saving can outvote it. For an accident-avoidance *guarantee*, protect pedestrians on the
*hard-constraint* side: a tighter `epsilon_pedestrian` (e.g. 1.0-1.5 m vs 2.0 m general) and/or a hard
pedestrian-recall floor. Soft weight = "prefers to protect"; hard constraint = "cannot neglect."

**Practical caveat:** heavy `pedestrian_recall` weighting only bites once pedestrian detection is solid — and the
2026-08-11 gate showed pedestrian detection is still ~17% even at 200k pps, so this is perception-limited for now.

> Formal spec + changelog: `REWARD_FORMULATION.md` (v5, authoritative). MDP diagram: `state_diagram.md`.
