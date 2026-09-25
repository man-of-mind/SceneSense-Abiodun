# Run-4 modeled-composite training evidence

`MODELED_COMPOSITE_TRAINING` is an offline training evidence class. It is not
measured runtime evidence, not `CALIBRATED_EMPIRICAL`, not production
authorization, and not deployment-readiness evidence.

The composite may combine fit-derived scene quality/payload, radio queue,
MCS-transition, actor-inference/action-materialization and feedback-latency
components. Every component is pinned to source evidence and fit support.
Every generated cycle additionally states whether its target profile and mode
are directly represented. The two gaps are
kept separate:

- `PROFILE_TRANSFER_UNVALIDATED`
- `MODE_TRANSFER_UNVALIDATED`

Either gap requires widened uncertainty. A row outside payload, backlog, MCS,
quality or latency-residual fit support is refused rather than extrapolated.
Validation rows cannot be consumed by this fit-only generator.

## Latency and timeout rule

Latency follows the sequential-kernel v3 boundary. The authoritative value is
an integer-nanosecond total derived from ordered action-open and UE feedback
receipt endpoints on one clock. The retained direct action-50 source used a
fixed action and therefore omitted actor inference plus quantization/dispatch.
Every modeled endpoint record must add a positive, evidence-bound actor delay
to that fixed-action source total. Treating actor cost as zero is refused. The
optional six-stage diagnostic is absent; it is never zero-filled or
manufactured by adding percentiles.


If the endpoint-derived total is at most 170 ms, the projection may create an
on-time feedback result. If it is greater than 170 ms, the learning event is a
timeout closed at `170 ms + 1 ns`. The actual later arrival is retained only as
`late_orphan_action_open_to_feedback_ns`; it carries no quality or successful
latency and cannot retroactively change the reward.

## Structural boundary

An issued envelope exposes `export_for_offline_training()` only as an explicit
offline seam. It returns an attested `ModeledCompositeOfflineTransitionV1`,
not a bare `SemiMarkovTransitionV2`. Both the envelope and typed export refuse
`export_for_replay()`. Existing environment and production-runner code accepts
its own exact calibrated types, so neither modeled record can be relabelled
into them. Existing replay rejects both modeled types at its exact-type gate,
including the explicit export-then-insert path.

The transition schema has no per-row environment-evidence field. Therefore a
future modeled-only buffer must admit the typed export and preserve its
evidence, support and latency-projection digests; it must not weaken the
production replay gate or expose a bare transition. Production authorization
continues to require the separately attested calibrated environment and
verifier path.
