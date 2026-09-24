# Run-4 prospective 500-update smoke preregistration

Status: **frozen prospectively; no smoke training or diagnostic-panel reading
was performed by this work.**

Canonical preregistration SHA-256:
`d9bd577b4f80d0dbc0bc1633c40a18a13a644aa04d1624e428cd36dbdc2c5d87`

The executable record is in `smoke_preregistration.py`. This document explains
what that record permits and, more importantly, what it does not permit.

## Purpose

Run 4 first performs one bounded 500-update diagnostic with seed 17. Its job is
to answer whether the actor and critics are moving in the intended direction,
whether all parts of the hybrid action space are exercised, and whether the
new causal state—including real MCS/backlog and the exact previous
action/outcome—changes decisions in controlled comparisons.

Passing the smoke authorizes only the preregistered continuation. It is not a
claim of convergence, test performance, live readiness, or deployment safety.

## Frozen training mechanics

| Item | Frozen value |
|---|---:|
| Per-tensor gamma | 0.99 |
| Discrete entropy coefficient | 0.05 |
| Continuous entropy coefficient | 0.02 |
| Actor learning rate | 0.0003 |
| Critic learning rate | 0.0003 |
| Polyak tau | 0.005 |
| Batch size | 256 |
| Replay capacity | 65,536 |
| Environment collection | 4 transitions per gradient update |
| PyTorch intra-op threads | 4 |
| Initial seed | 17 |
| Later seed order | 29, then 43, only after continuation approval |
| Checkpoints | 0, 100, 250, 500, 1,500, 10,000 |
| Mandatory first stop | update 500 |

Gamma is a replay/transition binding. Replay stores the contract-derived
semi-Markov discount and the trainer consumes that stored value verbatim. The
trainer must not reconstruct a discount from duration.

## Warm-up before the first gradient

The warm-up contains exactly 288 causal decisions:

\[
12\ \text{modes}
\times 6\ \text{continuous-}q\text{ bins}
\times 4\ \text{samples per bin}
=288.
\]

The coverage check uses exact `(mode, q-bin)` identities, so the highest-q bin
for every mode cannot disappear behind an aggregate count. Warm-up must also
contain both successful and failed outcomes. Existing Run-4 exploration gates
still require naturally evolving SI, P40, prior UE UL MCS, pre-enqueue backlog,
previous action, previous quality and previous latency; zero-filled or constant
stand-ins do not satisfy those gates.

## Diagnostic panel: real held-out identities only

The panel is deterministic, but this preregistration does not invent its rows.
A future composite verifier must supply and hash-bind:

- validation-partition calibration cell identities;
- fit-validation scene identities, disjoint from fit/training scenes;
- the exact ordered context-row identities and digest;
- the accepted sequential radio/queue-kernel binding;
- the mode or modes having the widest registered continuous-q support; and
- a wall-clock budget registered before launch.

The verifier must prove that the calibration validation cells are disjoint from
calibration fit cells, the fit-validation scenes are disjoint from fit scenes,
the row order is deterministic, every row is observed rather than fabricated,
and the identities/digests and kernel binding reconcile.

`REGISTERED_COMPOSITE_VERIFIER_MANIFEST_SHA256` is deliberately `None` today.
Therefore, production panel binding and production continuation authorization
are structurally unavailable. Test-only synthetic fixtures use a different
evidence class; even an all-pass test fixture returns
`continuation_authorized = false`.

## Update-500 continuation gates

All gates must pass on the fixed panel.

### Integrity/acceptance gates

1. **Full mode/q coverage:** all 12 x 6 strata were observed, including every
   highest-q stratum.
2. **Outcome coverage:** at least one registered success and one registered
   failure occurred.
3. **Finite numerics:** parameters, optimizer state, losses, targets and
   diagnostics remain finite.
4. **Checkpoint/resume identity:** an update-250 checkpoint resumed to update
   500 is bit-identical to the uninterrupted path.
5. **Bounded runtime:** recorded wall time does not exceed the verifier-bound
   pre-launch budget. No runtime limit is chosen after seeing the run.

### Hypothesis/diagnostic gates

1. Critic rank correlation at update 500 is strictly higher than at update 0.
2. Action regret at update 500 is strictly lower than at update 0.
3. A widest-support mode is not the deterministic choice in every panel
   context.
4. Continuous-q regret at update 500 is strictly lower than at update 0.
5. Controlled paired comparisons report a nonzero response, with a confidence
   interval excluding zero, for each of:
   - prior UE UL MCS;
   - pre-enqueue RLC backlog;
   - camera SI;
   - radar P40;
   - previous action; and
   - previous outcome.

These are direction/nonzero checks because no independent Run-4 baseline yet
justifies a numeric learning-improvement threshold. They must be reported with
their raw estimates, confidence intervals and artifact hashes. They are
diagnostic hypotheses, not statistical proof of generalization.

## Continuation rule

Seed 17 stops at update 500. Updates 1,500/10,000 and seeds 29/43 remain locked
until:

1. the registered composite verifier issues the exact private panel binding;
2. the update-500 diagnostics bind that panel and this preregistration digest;
3. all ten gates pass; and
4. the resulting production assessment explicitly authorizes continuation.

Any changed hyperparameter, panel identity, source partition, runtime budget,
gate, seed order or checkpoint schedule requires a new preregistration digest.

