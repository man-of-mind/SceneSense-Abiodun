# Expanded SPLIT+SKIP action gate v3 — common-state upper-bound correction

> **Historical Phase-1 spec after the 2026-08-14 causal audit.** V3 repaired its registered comparator, but the
> shared replay state itself is noncausal and GT-assisted. Preserve this spec for reproducibility; do not use its
> outcome as deployable controller evidence.

Status: **pre-registered before the v3 outcome**. V2 completed technically but its scientific verdict is
invalid. V3 inherits every source, frontier cell, held-out group, seed, action, reward coefficient, threshold,
and authorization boundary through SHA-pinned v2 and v1 configs.

## Why the completed v2 verdict is invalid

V2 fixed the object-support mismatch, but two remaining structural defects mean its separately rolled-out
one-step controller was not an upper bound:

1. In an infeasible per-UE state, the local `best_bound + delta` degradation filter ran **before** joint rate
   allocation. For simultaneous fast objects it removed 10 FPS from both candidate sets. The exact allocator
   was then forced to choose one 20 FPS send plus one `SKIP`, even though two 10 FPS sends fit aggregate C1.
   `SKIP` on a newly observed object invoked the intentional 1,000,000 m sentinel and a −25,000 reward. This is
   the measured cause of v2's −9.99 N=2 lift; it is a comparator defect, not a negative result.
2. Greedy and oracle advanced separate map states. A one-step optimizer can change its future state and become
   worse over a rollout; it cannot bound a different controller's sequential return. Calling that comparison
   an achievability ceiling was logically wrong.

These defects were diagnosed from action traces and state semantics, independent of any desired RL outcome.
The immutable v2 directory remains intact with its technical `COMPLETED` sentinel, but
`EXPANDED_SURROGATE_NO_GO_STOP` is explicitly not accepted scientifically.

## Frozen v3 comparison

V3 advances only the decentralized greedy trajectory. Before each greedy step, both choices are evaluated on
that exact common map/scheduler/channel state:

- **expanded decentralized greedy:** observable objects and estimated equal cell share, unchanged;
- **counterfactual oracle:** matched deployable tracker keys with true kinematics and true cell capacity.

The oracle solves the exact multiple-choice aggregate-rate problem. If a joint combination exists in which
every UE action satisfies the 2 m localization bound, it ranks only those joint-safe combinations. Otherwise,
graceful degradation occurs at the **system level**: every supported hard-C1 action, including lower FPS and
`SKIP`, remains available and reward v5 ranks the least-bad joint combination. No local pre-pruning may remove a
combination before aggregate feasibility is known.

The registered primary comparison excludes frames on which decentralized greedy misses the contemporaneous
true aggregate C1 budget. When a joint-safe combination exists, it also excludes a greedy combination that is
not matched-truth safe; when no joint-safe combination exists, both arms remain in the graceful-degradation
comparison. The C1 miss fraction remains reported and must stay below the inherited 1% validity ceiling. These
filters prevent a hard-constraint oracle from being called worse merely because greedy violated a constraint.

The result is an exact **one-step counterfactual upper bound along greedy-visited states**. It is not a
future-perfect or sequential oracle, does not bound policies that deliberately visit different states, and
cannot prove project-wide RL impossibility. A positive gap authorizes only a queue-aware non-learning
max-weight baseline. A tie supports stopping this current surrogate direction but must be stated with the
common-state boundary.

No shared queue, LOCAL, OAI, CARLA, max-weight, MPC, or RL is introduced in v3.
