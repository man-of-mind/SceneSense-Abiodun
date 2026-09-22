# Hybrid-SAC Run 2 emitted-float32 preregistration v2

This artifact supersedes Run 2 preregistration v1 at commit
`a35ddb12d0fdfd5a3f7ad273c8021a08ed160394`. The v1 coefficient receives a
retrospective **NO-GO**: although its binding inequality was strict in
binary64, both values became `-0.13886235654354095` (bits `0xbe0e31ef`) in
the float32 learning target.

## Frozen intervention

V2 pins the runtime operation order rather than treating a mathematical
rearrangement as equivalent:

```text
base64 = p * (Q - 0.25 * (L95 / 200.0)) + (1.0 - p) * (-1.0)
if L95 > 200.0:
    shaped64 = base64 - p * 0.5742957622788527
else:
    shaped64 = base64
emitted_target = torch.tensor(shaped64, dtype=torch.float32).item()
```

The coefficient is the smallest non-negative finite binary64 value that
makes every positive-admission infeasible action emit a target strictly below
the context's emitted-float32 constrained-winner target on registered training
support. It is `0x1.260a181a70290p-1` (bits `0x3fe260a181a70290`). Its
immediate predecessor, `0x1.260a181a7028fp-1`, fails one binding comparison.
An action with `p=0` continues to emit exactly `-1` and is excluded from the
conditional-survivor competition.

## Train-only evidence frozen before validation

The two exhaustive passes each evaluated all `52,240` executable actions in
all `1,564` registered training contexts (`81,703,360` action-context
evaluations per pass). The independent second pass checked `45,442,313`
positive-admission infeasible comparisons and found zero strict-ordering
violations. The shaped winner equals the emitted-float32 constrained winner in
all `1,564` contexts, with zero modeled P95 misses. Mean quality retention is
`96.493%`; mean admission changes by `+0.000979`.

No v2 validation outcome was queried and no policy was trained before this
artifact was frozen. Validation, if subsequently authorized, must use this
coefficient and exact emitted-target calculation without retuning. A failure
must be reported as penalty generalization failure and moved to a separately
preregistered design.

This remains modeled conditional-retained-survivor evidence, not a live
200-ms service-level guarantee. The complete frozen contract and later-run
criteria are in `preregistration_v2.json`.
