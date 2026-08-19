# Recipient fusion v3: deterministic equal-time fusion

Status: **offline implementation and unit tests only**. This engine is not
wired into CARLA, OAI, the completed v2 replay, or a paper result.

## Why v3 exists

`RecipientMapEngineV2` installs contributions sequentially. When recipient and
helper observations have the same measurement timestamp and associate to one
canonical track, v2 overwrites state, covariance, confidence, and motion noise
with the last installed observation. The completed calibration replay used a
fixed recipient-then-helper order, so its result is reproducible but its fused
state is not invariant to an arbitrary network arrival order.

V2 is frozen and remains the provenance implementation for that replay. V3 is
a create-only alternative; it does not reinterpret or overwrite v2 artifacts.

## V3 contract

V3 continues to accept `MapContributionV2` and emit `WarningEventV2`. It changes
only recipient-side map estimation after source observations associate to one
canonical track:

1. Retain the latest accepted estimate separately for each source UE.
2. Find the newest measurement timestamp represented on the track.
3. Fuse every source estimate at that timestamp in sorted source-ID order.
   Older source estimates remain historical provenance but do not alter the
   current fused state or count as current fusion evidence.
4. Weight each state by source confidence divided by position-covariance trace.
5. Form the fused covariance from the weighted Gaussian-mixture moment:

   \[
   \bar{x}=\sum_i w_i x_i,\qquad
   P=\sum_i w_i\left(P_i+(x_i-\bar{x})(x_i-\bar{x})^T\right).
   \]

   The disagreement term prevents two conflicting sources from creating a
   falsely precise estimate. It is deliberately conservative: the v2 wire
   contract has no cross-source covariance with which to justify independent
   information-filter fusion.
6. Fuse confidence and process noise with the same deterministic weights. Use
   the common minimum validity horizon and maximum capture/publication times.

The engine labels its tracks with
`quality_weighted_moment_equal_time_v1`, uses `map_track_v3_*` IDs, and exposes
the active fusion sources in snapshots and warning provenance.

## Warning confirmation and hysteresis

V3 does **not** silently add temporal warning hysteresis. The current wire
record lacks source-tracker hit streak, confirmed/tentative state, track age,
and miss streak. Reconstructing those fields inside the map would create a
second undocumented tracker and would not reproduce live causality.

An optional `warning_confirmation_policy` hook receives a frozen
`WarningConfirmationContextV3` containing only recipient-available fields:
canonical track/class, warning and fused-measurement times, fused confidence,
active source IDs, source-track IDs, source measurement times, and per-source
confidence. The hook must return `bool`; exceptions or non-boolean responses
fail closed. It is disabled by default, so v3 does not claim a persistence
benefit before the missing confirmation metadata is added to a future schema
and measured.

## Verified properties

Focused tests establish that:

- helper-first and recipient-first equal-time installs produce identical v3
  snapshots and warnings;
- a precise/high-confidence estimate receives greater weight while source
  disagreement remains in covariance;
- a strictly newer measurement supersedes an older source estimate in either
  cross-source arrival order;
- the confirmation hook sees only causal fusion metadata and filters warnings;
  and
- a non-boolean confirmation result fails closed.

## Limits and adoption gate

- V3 fixes latest-writer dependence **after association**. V2's streaming
  greedy association is retained. Ambiguous, closely spaced multi-object scenes
  can still require timestamp batching and deterministic global assignment.
- Confidence is not yet probability-calibrated, and cross-source error
  correlation remains unknown. The moment rule is a safe deterministic
  baseline, not a statistically optimal fusion claim.
- Only estimates in the newest equal-time cohort influence the state. A future
  asynchronous fusion design must propagate and correlate older estimates
  explicitly rather than mixing them ad hoc.
- The confirmation hook is an interface, not evidence that persistence or
  hysteresis improves the warning surface.
- No warning-threshold, false-warning, lead, payload, or safety claim transfers
  from v2 to v3 without a new versioned offline calibration/validation pass.
- Actual OAI enqueue, on-wire, reassembly, and install timing remains a separate
  blocking measurement.

Before adoption, run a bounded retained-artifact permutation audit, verify that
all equal-time source-order permutations agree, inspect ambiguous associations,
and then evaluate v3 on a pre-registered calibration split. Do not rerun or
rewrite the completed v2 provenance analysis.
