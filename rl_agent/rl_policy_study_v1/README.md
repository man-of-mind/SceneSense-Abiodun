# SplitFusion RL policy study v1

Status: **MEASURED STEPS 5–6 DESIGN — NOT TRAINING OR DEPLOYMENT AUTHORITY**

This directory develops the first deployable-policy formulation for the
SplitFusion UE controller.  It is deliberately written as a study: each design
choice is introduced from first principles, connected to the measurements that
motivate it, and separated from choices that still need evidence.

The selected starting point is:

> **PPO with recurrent state, a hierarchical action policy, invalid-action
> masks, and Lagrangian cost critics.**

That sentence describes a composition of established techniques.  It is not
the name of a single published algorithm and it is not presented as a new RL
algorithm.  `Recurrent Hierarchical Masked PPO` may be used as convenient
project shorthand, but papers and presentations should spell out the
components.

## Why this study is separate

This study began on branch `rl-agent-study-v1` in a separate linked worktree so
it could not alter the active 288-cell campaign. The completed campaign and
Steps 1–4 consolidation were merged into the study branch after the campaign
terminated. The study remains isolated until its evidence and formulation are
reviewed; it does not modify the completed campaign.

The older files in `rl_agent/policy/` remain useful historical evidence, but
their dynamic-controller evaluation used noncausal post-tail information and
GT-assisted identities.  This study does not silently reuse that observation
contract.

## Study sequence

1. [RL from first principles](RL_FROM_FIRST_PRINCIPLES.md)
   - MDP and POMDP definitions
   - state, observation, action, policy, reward, value and return
   - why recurrence, hierarchy and masking are appropriate here
   - PPO and constrained-learning equations
   - discrete versus continuous ROI/drop control
   - causal inputs and forbidden information
2. [Measured surface and policy formulation](MEASURED_SURFACE_AND_POLICY_FORMULATION.md)
   - descriptive dominance and conditional outcome labels
   - split-only causal observation and action contract
   - reward/cost formulation
   - training-data and stochastic-emulator requirements
3. Initial policy-network implementation and hand-checkable synthetic cases.
4. Causal per-frame dataset materialization, reward normalization and
   sensitivity study.
5. Surrogate training followed by a separately authorized live evaluation.

## Reading the equations

The study uses the Markdown math syntax supported by VS Code and GitHub:
`$...$` for inline equations and `$$...$$` for display equations.  In VS Code,
use the built-in **Markdown: Open Preview to the Side** command (`Ctrl+K V` on
Linux/Windows) and ensure `markdown.math.enabled` has not been set to `false`.
No extension is required.

## Current decisions

| Question | Provisional decision | Why |
|---|---|---|
| Primary learning algorithm | PPO | Stable policy-gradient starting point; supports a conditional mixed policy without flattening future continuous controls |
| Memory | GRU | Network evolution, queues and map freshness are history-dependent and only partially observed |
| Current action | One of 72 measured `SPLIT` profiles | The 288 campaign directly measures these actions |
| Future top-level action | `SPLIT`, `LOCAL`, `SKIP` | `LOCAL` and `SKIP` remain unavailable until their transitions are measured/frozen |
| First ROI/drop control | Six discrete q anchors | Avoid learning through unvalidated interpolation |
| Continuous-q extension | Explicitly designed, deferred | Requires a dense-q smoothness/interpolation study before promotion |
| Invalid-action mask | Hard executability only | Poor but executable profiles remain available so the policy learns their consequences |
| Safety/freshness | Separate cost critics plus runtime integrity rules | Avoid hiding safety requirements inside one arbitrary scalar reward |
| Segmentation | Secondary utility, installation controlled by profile metadata | Object coverage, localization and freshness are primary |
| Algorithmic novelty claim | None | The contribution is the measured, integrated system and policy evidence |

## Evidence status and remaining gates

- **Complete:** 288-cell measurements, verified consolidation and descriptive
  Step-5 surface analysis.
- Materialize the seven-feature causal per-frame transition table. The
  aggregate 288-cell table is not directly a PPO replay buffer.
- Freeze the freshness, radar-risk and compute-slack reducers and prove the
  UE-visible availability time of the capacity lower bound.
- Separate causal measurements for `LOCAL` on CPU/GPU and the compact result
  upload.  Until then, `LOCAL` can exist in the architecture but cannot be
  trained from invented outcomes.
- A precise `SKIP` transition: tracks age, no new object is discovered, and no
  transmission/compute cost is charged.
- Independent trace seeds or live runs for final evaluation; training and
  reporting on the same four frozen traces would overstate generalization.
- Dense-q evidence before replacing the 72-profile categorical choice with a
  continuous q distribution.

## Primary references

- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), 2017.
- Huang and Ontañón, [A Closer Look at Invalid Action Masking in Policy Gradient Algorithms](https://arxiv.org/abs/2006.14171), 2020/2022.
- Achiam et al., [Constrained Policy Optimization](https://arxiv.org/abs/1705.10528), ICML 2017.

These references motivate components of the design.  In particular, using a
Lagrangian cost with PPO does **not** inherit CPO's theoretical guarantee.
