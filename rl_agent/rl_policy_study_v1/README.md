# SplitFusion RL policy study v1

Status: **EDUCATIONAL DESIGN — NOT TRAINING OR DEPLOYMENT AUTHORITY**

This directory develops the first deployable-policy formulation for the
SplitFusion UE controller.  It is deliberately written as a study: each design
choice is introduced from first principles, connected to the measurements that
motivate it, and separated from choices that still need evidence.

The selected starting point is:

> **Split-only PPO with LSTM state, hard executability masks, an auxiliary
> next-channel forecast head, and separately reported cost critics.**

That sentence describes a composition of established techniques.  It is not
the name of a single published algorithm and it is not presented as a new RL
algorithm.  `Recurrent Hierarchical Masked PPO` may be used as convenient
project shorthand, but papers and presentations should spell out the
components.

## Repository status

The study began in a separate worktree while the 288-cell campaign was active.
After the campaign completed, its two documentation commits were integrated
into `master`. The campaign evidence remains immutable; this directory is the
new implementation boundary for the controller study.

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
2. [Empirical environment contract](EMPIRICAL_ENVIRONMENT_CONTRACT.md)
3. Initial policy-network implementation and hand-checkable synthetic cases.
4. Reward normalization and sensitivity study.
5. Surrogate training followed by a separately authorized live evaluation.

The compact [preliminary architecture](AGENT_ARCHITECTURE_V1.md) connects the
implemented network to the completed 288-cell evidence, latest-only edge
scheduling, explicit supersession feedback, and the final edge-optimization
measurements.

`model.py` now implements step 3's architecture boundary. It is deliberately
trainer-free: its CPU tests cover the 72-way output, fail-closed masks, LSTM
state carry, value/cost head shapes, and forecast-loss gradient flow. This does
not advance step 4 or authorize PPO training.

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
| Memory | LSTM + auxiliary one-step channel forecast | Network evolution, queues and map freshness are partially observed; the forecast is learned from history, never supplied from the future |
| Initial action space | 72 `SPLIT` profiles only | Every action has measured payload and perception evidence; `LOCAL`/`SKIP` await their own transition measurements |
| First ROI/drop control | Six discrete q anchors | Avoid learning through unvalidated interpolation |
| Continuous-q extension | Explicitly designed, deferred | Requires a dense-q smoothness/interpolation study before promotion |
| Invalid-action mask | Hard executability only | Poor but executable profiles remain available so the policy learns their consequences |
| Safety/freshness | Separate cost critics plus runtime integrity rules | Avoid hiding safety requirements inside one arbitrary scalar reward |
| Segmentation | Secondary utility, installation controlled by profile metadata | Object coverage, localization and freshness are primary |
| Algorithmic novelty claim | None | The contribution is the measured, integrated system and policy evidence |

## Evidence still required

- A causal stateful switching environment that reproduces the fixed-action live
  surface before PPO training.
- Separate causal measurements for `LOCAL` on CPU/GPU and the compact result
  upload.  Until then, `LOCAL` can exist in the architecture but cannot be
  trained from invented outcomes.
- A precise `SKIP` transition before it is added in a later action-space
  extension: tracks age, no new object is discovered, and no transmission or
  compute cost is charged.
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
