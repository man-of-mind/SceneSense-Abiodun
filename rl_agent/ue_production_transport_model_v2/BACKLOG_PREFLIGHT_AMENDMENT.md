# Prospective amendment: backlog preflight for the bounded exploratory smoke

Status: **frozen before the smoke runs**. Scope: the backlog state-variation
preflight only. Every other preflight gate — camera SI, radar P40, UL MCS,
action coverage, outcome coverage, exact previous-feedback propagation — is
unchanged.

## Why

The 12-cell capture measured three byte roles, and the registered warm-up
schedule keeps all 288 actions inside that support (6,621–427,849 wire bytes).
Fitted per-cycle service capacity is 700,350–1,339,307 bytes, which exceeds
the two-frame ingress of most in-support actions. The UE queue therefore
drains to zero in the large majority of warm-up cycles. That is a **property
of this operating envelope**, not a defect in the collector or a placeholder:
at these payloads the 273-PRB uplink genuinely does not queue.

Imposing a maximum-zero-fraction threshold here would be a post-hoc gate
invented to match an expectation, so none is imposed.

## Amended backlog criteria

All 288 warm-up backlog measurements must be:

1. **causal** — taken strictly before the frame-open/action-release instant;
2. **finite** — exact non-negative integers, no NaN or None;
3. **non-imputed** — produced by the fitted causal queue-transition head from
   the preceding resolved outcome, never zero-filled or defaulted.

And the population must show:

4. at least one exact-zero backlog;
5. at least one strictly positive backlog;
6. at least two distinct values;
7. a strictly positive span.

The smallest physically meaningful positive span is one byte,
`log1p(1)/log1p(50,000,000) = 0.039100` in the registered log-scaled feature,
so that is the bound used.

## Reported, not gated

Zero fraction, positive count, distinct count, span, modes covered and MCS
values covered are all reported. **No maximum-zero threshold applies.**

## The hard continuation gate is unchanged

The update-500 controlled backlog-response test remains the gate for
continuing to the three-seed campaign, and its confidence interval must
exclude zero. A weak backlog signal in the warm-up does not relax that; if the
actor's response to backlog is indistinguishable from zero at update 500, the
campaign does not start.
