# Task C — measured-table enumerator, lambda-RDO, and freshness heuristic

This separates two questions that must not be conflated: the static 36-profile scalar rate-distortion problem, and the stateful retained-catalog SPLIT+SKIP surrogate.

## Static 36-profile H2 test

The lambda sweep supports 4/36 profiles: `ae32__uint4__roi0.5, ae64__uint4__roi0.5, ae32__uint4__roi0.0, ae128__uint4__roi0.0`.
Across all 36 measured payload breakpoints, supported-hull lookup agrees with exact budgeted enumeration on 80.56% and loses utility at 7 breakpoints.
Mean/max exact-minus-lambda utility gap: 0.000751 / 0.011686.
Mean/max Lagrangian duality gap: 0.001712 / 0.017359.
The exact static winner belongs to the prior retained-seven catalog at 88.89% of breakpoints.

## Held-out retained-catalog ladder

Lambda-RDO own-state agreement with full enumeration is 100.00%; its mean own-state predicted reward gap is 0.000000.
Its independent rollout matched-reward delta is +0.000000, trajectory-cluster CI [+0.000000, +0.000000].
The AoI-index-inspired heuristic (not Whittle) has 86.47% own-state agreement and matched-reward delta -0.006981, CI [-0.014058, +0.003618].

## Verdict boundary

H1/H2 are tested rather than assumed. Static hull agreement speaks only to the scalar profile problem after FPS/budget are fixed. Runtime agreement cannot prove the full controller collapses to one scalar because AoI, speed, FPS, latency, prior map state, pending frames, safety, and switching remain active.

Linked ladder artifact: `rl_agent/policy/experiments/controller_ladder/20260814_220006`.
Genuine Whittle-index evaluation remains deferred to Phase-2 object-selective sharing, where per-object arms and indexability can be defined.
