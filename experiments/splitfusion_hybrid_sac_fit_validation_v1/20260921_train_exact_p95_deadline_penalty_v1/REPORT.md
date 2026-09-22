# Train-only exact P95 deadline-penalty preflight

The coefficient is derived analytically from registered training
support; it was not selected from a post-hoc grid. The penalty is
inside the admitted branch because P95 is conditional on retained
survivors. Registered bundles are integrity-audited when loaded, but
selection queries only the 391 registered training scene IDs.

- Decision: **GO**
- Derived Delta: `0.574295760417`
- Derived lambda_D: `0.574295760417`
- Float64 search evaluations: `1`
- Unconstrained P95 misses: `415/1564`
- Constrained/shaped P95 misses: `0/1564`
- Mean quality: `0.704704` -> `0.679990` (96.493%)
- Mean admission: `0.998636` -> `0.999615` (+0.000979)
- Shaped/constrained identity matches: `1564/1564`
- Binding worst case: context `938`, mode `2`, q_e4 `9000`, ratio `0.574295760417`

This is modeled conditional-survivor compliance, not a live 200-ms SLA.
No policy training or validation selection was performed.
