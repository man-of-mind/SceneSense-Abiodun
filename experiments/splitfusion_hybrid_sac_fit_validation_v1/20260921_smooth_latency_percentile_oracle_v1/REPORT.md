# Smooth latency-percentile oracle screen

This is a CPU-only, exact-support, pre-training counterfactual on the frozen
fit-validation panel. It is not live evidence, a timeout-probability model,
or a new D1 result.

| utility | budget misses | mean quality | mean admission | mean latency | constrained reward cost |
|---|---:|---:|---:|---:|---:|
| D1_SMOOTH_P50_CONTROL | 2/340 (0.6%) | 0.6835 | 0.9986 | 166.9 ms | 0.000386 |
| COUNTERFACTUAL_SMOOTH_P95 | 81/340 (23.8%) | 0.6769 | 0.9989 | 189.6 ms | 0.010848 |
| COUNTERFACTUAL_SMOOTH_P99 | 222/340 (65.3%) | 0.6728 | 0.9990 | 212.8 ms | 0.129676 |

The smooth objective has no discontinuity at 200 ms. A miss means only
that the exact reward maximizer preferred a quality/admission trade-off
above the modeled conditional-quantile budget. The constrained comparator
shows what the best same-reward action would be if that modeled budget were
enforced. No failure or timeout probability is inferred.
