# Run-4 replay boundary

`replay.py` is a CPU-only, in-memory boundary between the attested Run-4
semi-Markov transition contract and a future trainer. It is not an environment,
calibration procedure, evidence loader, or empirical result.

## Accepted record

The only accepted input is an object whose exact type is
`SemiMarkovTransitionV2` and whose private contract attestation still matches
its current canonical fields. Subclasses, adapters, aggregate measurements,
and synthetic fixture/report labels are rejected. Replay never calls a
fixture's conversion method and never derives a transition from a 288-cell
aggregate.
Accepted transition objects are not exposed again through a public replay
accessor; resident audit exposes canonical digests and non-learning metadata.

Insertion independently checks that:

- current and successor features remain bound to their guarded states;
- `CONTINUES` carries the real next decision;
- the successor's `previous` outcome is exactly the current reward resolution;
- `TERMINATED` and `TRUNCATED` carry no fabricated successor;
- the executed `(mode_id, q_e4)`, reward, duration and externally derived
  discount are valid after float32 conversion; and
- the transition matches the buffer's frozen evidence binding.

## Required evidence binding

A production buffer cannot be created from bare SHA-256 strings. It requires
an opaque attestation issued by the accepted calibration/queue verifier, which
in turn binds explicit lowercase SHA-256 values for all of the following:

1. freshness policy;
2. empirical feature scaling;
3. accepted network calibration evidence; and
4. accepted queue-kernel evidence.

The binding also pins the Run-4 overall/feature/reward/transition schema
hashes, the action catalog, feature order/count, gamma, and the registered
semi-empirical evidence class. The buffer and every batch carry one exactly
equal binding.

This pre-calibration module intentionally contains no production attestation
issuer, so `ReplayBufferV1` currently fails closed. The accepted 12-cell
verifier must provide that integration after its verdict.

Unit tests use a private `_for_test_only` binding and
`_TestOnlyReplayBufferV1`, both permanently labelled
`TEST_ONLY_MECHANICS`. The production buffer rejects that binding, and the
label is retained in every test batch. These tests make no empirical claim and
emit no evidence. This module has no production defaults and discovers no
files.

## Stored tensors

Each sampled row contains:

- 21-D current state and, when present, the real 21-D successor;
- executed mode and `q_e4` as `int64`;
- reward and the caller-derived discount as CPU `float32`;
- exact hold duration as `int64`; and
- next-state, bootstrap, terminal and truncation masks as `bool`.

The discount is converted once from the attested transition and never
recomputed from float32 gamma. Boundary rows use a zero successor tensor only
as a rectangular-batch sentinel, with both `has_next_state` and `bootstrap`
false. Every public tensor accessor returns a clone.

Sampling is uniform without replacement and requires an explicit, non-default
CPU `torch.Generator`; global and unrelated RNG streams are not advanced.
FIFO eviction never erases the lifetime digest or logical decision indexes, so
duplicates and contradictory outcomes remain rejected after eviction.
