# Expanded SPLIT+SKIP action gate v1 — frozen desk-only contract

Status: **pre-registered before opening the outcome**. This artifact answers whether the accepted
reward-v5 replay contains useful one-step coordination headroom after restoring measured payload/profile and
FPS choices. It does not run OAI or CARLA and it does not authorize controller construction or RL.

## Scientific question and boundaries

The earlier 400 KiB multi-UE deadline experiment saturated its 250/500 ms metric: even whole-cell
serialization was too slow. That is a contract defect, not evidence that coordination is useless. The first
output here is therefore a deadline-feasibility frontier using production on-wire bytes, measured rung
ceilings, exact UE count, the 0.70 C1 margin, and accepted reward-v5 non-network latency. Queue-free latency is
only a **necessary lower bound**; a feasible cell is not a queue-delay guarantee.

The second output compares two ends of one deliberately small action space: all 35 measured
`SPLIT(profile,FPS)` actions plus `SKIP`. `LOCAL` is excluded because no calibrated latency/accuracy/cost table
exists. The decentralized arm sees only its own accepted replay observation and an equal observed cell share.
The oracle sees true objects and true cell capacity, then solves the per-frame multiple-choice aggregate-rate
problem exactly. It is a **joint true-state one-step upper bound**, not a future-perfect sequential oracle.

This v1 replay comparison has no shared queue. Hard C1 load shaping keeps aggregate offered load below the
chosen capacity envelope, and each UE retains the accepted event-driven latency/map surrogate. Consequently,
the comparison is an optimistic, queue-free headroom screen. It may justify building a queue-aware max-weight
baseline, but it cannot itself establish real multi-UE deadline performance. Existing unsent work is replaced
by the latest schedule; transmitted FIFO work and deadline-aware queue dropping are out of scope.

## Frozen sources and action semantics

The config pins SHA-256 values for the accepted reward-v5 resolved config, replay registry, seven-profile
catalog, combined channel surface, and DG-A measurement artifacts. Any mismatch fails closed. Only held-out
test trajectories are used. Each paired controller run receives the same scenario group, Markov-channel seed,
capacity realization, and per-UE latency random stream.

The frontier reports every combination of:

- canonical payloads plus the 400 KiB stress payload;
- N in {1, 2, 4, 50, 100}, all four measured MCS rungs, 250/500 ms, and 2/5/10/15/20 FPS;
- whole-cell optimistic, equal raw share, and equal C1 share envelopes.

For a profile, `nonnetwork_p95 = 122.71 ms + (front+back-36.8 ms)`. Serialization uses the exact production
UDP chunk/header accounting and the selected share. Rate feasibility separately requires the action's on-wire
offered rate to remain inside 0.70 of its raw share. The 400 KiB stress row uses the reference 36.8 ms compute.

For replay, both arms use reward v5 unchanged: 0.35 segmentation, 0.40 pedestrian recall, 0.25 vehicle recall,
plus localization error, PRB, and switching terms. All supported safe SPLIT actions remain candidates; the
old preferred-core pruning is intentionally not used because it would silently remove the smaller payloads
being tested. When no action satisfies the localization bound, the existing least-bound degradation rule and
`SKIP` remain available. The oracle is also subject to the same per-action support and localization shield and
to joint `sum(offered_mbps) <= 0.70 * true_cell_capacity`.

## Pre-registered outcome rule

The primary unit is each scenario group's UE-mean matched-truth reward-v5, first averaged over its three paired
channel seeds and then weighted equally across groups. A deterministic 10,000-replicate cluster bootstrap over
scenario groups gives the descriptive paired 95% interval. Episode reuse is disclosed, so this is not claimed
as an independent-trajectory population interval.

Oracle headroom is a candidate only when **all** conditions hold:

1. group-equal absolute reward lift is at least 0.01;
2. lift is at least 5% of `max(abs(greedy reward), 0.1)`;
3. paired bootstrap 95% lower bound is above zero;
4. mean lift has the same positive sign at N=2 and N=4; and
5. no group's worst-UE mean reward regresses by more than 0.005.

If all pass, stop with `CANDIDATE_HEADROOM_BUILD_MAX_WEIGHT_NEXT`; the next rung is a deadline-aware,
queue-aware non-learning max-weight controller, not RL. Otherwise stop with
`EXPANDED_SURROGATE_NO_GO_STOP`. Binary 250/500 ms capture-to-map fractions, action mix, aggregate C1 misses,
and the feasibility frontier are diagnostics and cannot override the registered primary rule.

Because the independent replay environments do not emulate a shared queue, the screen fails closed as
`HOLD_INVALID_QUEUE_FREE_C1_ENVELOPE` if decentralized observed-share actions exceed the contemporaneous true
aggregate capacity on more than 1% of frames. Below that pre-registered ceiling the miss fraction is reported
as a bounded approximation limitation; above it no greedy-oracle scientific verdict is issued.

The earlier accepted single-UE and DG-A results remain immutable. No post-outcome threshold, source, group,
or action-space change is permitted in this version.
