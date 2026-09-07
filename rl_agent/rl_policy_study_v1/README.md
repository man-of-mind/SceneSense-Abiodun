# SplitFusion RL policy study v1

Status: **EDUCATIONAL DESIGN — NOT TRAINING OR DEPLOYMENT AUTHORITY**

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

The active 288-cell campaign was launched from the main checkout.  This study
lives on branch `rl-agent-study-v1` in a separate linked worktree based at
commit `1c9e2ba`.  Nothing here should be copied into or merged with the active
checkout until that campaign terminates.

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
2. Empirical environment contract — after the 288-cell campaign completes.
3. Initial policy-network implementation and hand-checkable synthetic cases.
4. Reward normalization and sensitivity study.
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
| Top-level action | `SPLIT`, `LOCAL`, `SKIP` | These have different physical execution paths and costs |
| First SPLIT sub-action | One of 72 measured profiles | Every selected action has measured payload and perception evidence |
| First ROI/drop control | Six discrete q anchors | Avoid learning through unvalidated interpolation |
| Continuous-q extension | Explicitly designed, deferred | Requires a dense-q smoothness/interpolation study before promotion |
| Invalid-action mask | Hard executability only | Poor but executable profiles remain available so the policy learns their consequences |
| Safety/freshness | Separate cost critics plus runtime integrity rules | Avoid hiding safety requirements inside one arbitrary scalar reward |
| Segmentation | Secondary utility, installation controlled by profile metadata | Object coverage, localization and freshness are primary |
| Algorithmic novelty claim | None | The contribution is the measured, integrated system and policy evidence |

## Evidence still required

- Complete 288-cell measurements and a frozen parser/schema.
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
