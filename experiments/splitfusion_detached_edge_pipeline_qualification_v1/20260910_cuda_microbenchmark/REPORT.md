# SplitFusion detached edge pipeline CUDA qualification

Status: `PARITY_ACCOUNTING_AND_OVERLAP_QUALIFIED`

## Result

The optimized tail's singleton serialization handoff was replaced in a new,
unpromoted candidate by an exclusive per-frame work product. One owner executes
the frozen CUDA tail and transfers the retained tensors to CPU. A distinct
owner constructs and publishes the compact service records using CPU tensors
only. The hash-pinned deployed edge was not changed.

Eight seeded synthetic C2 activations passed exact comparison against the
established optimized adapter:

- perception tensors and ordering: bit-identical;
- p025 retained indices: bit-identical;
- segmentation labels: bit-identical;
- serialized service records: byte-identical; and
- frozen FCOS state before/after: identical at
  `bdb3a245f26fb17a9b0185c6c140ebd3774aa5f3f6b4d984b7e1d9665c1f3a53`.

Median synthetic service timing, excluding the first frame, was 46.68 ms for
the established optimized adapter. The detached candidate divided this into
43.89 ms on the compute owner and 2.72 ms on the CPU publication owner. This is
a handoff measurement, not a live Route-B latency claim.

## Scheduling controls

At an artificial 5 ms arrival interval, both candidates maintained exactly one
compute owner, one publication owner, depth-one pending slots, complete terminal
accounting, and observed two-stage overlap. Latest-only/no-expiry installed 4
of 24 frames and explicitly superseded 20 pending frames. Latest-only/25-ms
installed 3, superseded 20, and explicitly expired one queued frame.

At the 100 ms control cadence, both policies installed all 12 frames with zero
supersession and zero expiry. No overlap occurred because the isolated
synthetic tail and publication work completed before the next arrival. Thus the
25 ms value is confirmed as an expiry ceiling, not an intentional hold.

## Interpretation

The concurrency design is valid, but CPU record construction is only about
2.5--2.7 ms in this microbenchmark. Separating it cannot by itself recover the
roughly 50 ms sought from the live service path. The next optimization target
is therefore the camera-aware post-processing inside the compute stage,
followed by the preregistered live action-50/action-71 policy comparison.

No scheduling policy is selected from this microbenchmark. Selection still
requires live map-AoI, useful-installation, supersession, bytes, compute, and
failure measurements.

## Scope and diagnostic history

No dataset frame, CARLA process, OAI/RFsim process, container, or network was
used. Two preliminary invocations correctly exposed an over-strict overlap
assertion, followed by one local reporting-variable placement error. A later
invocation exposed registry-read time inside the synthetic arrival generator;
that harness overhead was removed before the two results above were recorded.
None of these preliminary invocations changed model state or produced evidence.
