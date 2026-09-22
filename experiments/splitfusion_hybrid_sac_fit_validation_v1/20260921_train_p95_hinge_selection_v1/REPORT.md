# Train-only P95 deadline-hinge selection

This CPU-only screen uses only the registered training partition. The
frozen fit-validation panel, completed D1/P50 runs and checkpoints were
not read for selection or modified.

Disclosure: an uncommitted exploratory probe of this same lambda grid
preceded the stricter zero-miss/quality/admission acceptance rule. The
screen is therefore transparent train-only design evidence, not a claim
that the final acceptance rule was registered before all inspection.

| lambda | P95 misses | miss rate | mean quality | mean P95 | mean utility |
|---:|---:|---:|---:|---:|---:|
| 0 | 415/1564 | 26.535% | 0.7047 | 191.4 ms | 0.463545 |
| 0.25 | 301/1564 | 19.246% | 0.6996 | 188.5 ms | 0.459229 |
| 0.5 | 236/1564 | 15.090% | 0.6954 | 187.1 ms | 0.456607 |
| 1 | 153/1564 | 9.783% | 0.6908 | 185.9 ms | 0.453896 |
| 2 | 98/1564 | 6.266% | 0.6868 | 185.1 ms | 0.451493 |
| 4 | 43/1564 | 2.749% | 0.6830 | 184.6 ms | 0.449839 |
| 8 | 20/1564 | 1.279% | 0.6808 | 184.4 ms | 0.449038 |
| 16 | 7/1564 | 0.448% | 0.6801 | 184.4 ms | 0.448909 |

Decision: **NO-GO**; no preregistered coefficient met every criterion.

Selection rule: choose the smallest preregistered coefficient with
zero modeled P95 deadline misses across all registered training
scene/profile contexts, at least 95% of lambda-zero mean perception
quality, and no more than 0.001 absolute loss in mean admission.
A finite hinge is not claimed to provide a live hard guarantee.
