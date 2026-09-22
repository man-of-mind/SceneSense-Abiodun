# Train-only emitted-float32 exact P95 deadline penalty v2

V1 is superseded: its binary64 strict inequality collapsed to an
equal float32 learning target. V2 derives the coefficient against
the exact CPU float32 scalar emitted after the pinned binary64 formula.
Only the 391 registered training scene IDs were queried.

- Decision: **GO**
- Smallest binary64 lambda: `0.5742957622788527`
- Lambda hex/bits: `0x1.260a181a70290p-1` / `0x3fe260a181a70290`
- Immediate predecessor insufficient: `True`
- Strict infeasible ordering violations: `0`
- Shaped/constrained identity matches: `1564/1564`
- Train P95 misses: `0/1564`
- Mean quality: `0.704704` -> `0.679990` (96.493%)
- Mean admission: `0.998636` -> `0.999615` (+0.000979)
- Old-lambda emitted collisions: `1` (binding float32 bits `0xbe0e31ef`)
- Context-worst ULP margins p0/p50/p95/p100: `1/2085485523/2101074342/2109971129`

This is a modeled conditional-retained-survivor train-support proof,
not a live 200-ms SLA. No validation outcome was queried and no policy
training was run by this derivation.
