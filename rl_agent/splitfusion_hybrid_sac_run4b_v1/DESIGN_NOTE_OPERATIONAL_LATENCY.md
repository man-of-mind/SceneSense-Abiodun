# Superseding design note: Run-4B operational latency

`LATENCY_BOUNDARY_AUDIT.md` and its `STOP` verdict remain the historical
record. The existing Run-4 latency composition was not reused.

## Authorized decisions

1. Do not reuse the GT-contaminated residual.
2. Use the direct split-host operational boundary, `L_op = A + S + T + E + D`.
3. `GO_OPTION_1_POOLED_A_E_EXPLORATORY`: pool A and E across families.

## What Run-4B uses

The shared provider `rl_agent/splitfusion_operational_latency_v1/provider.py`
(see its `PROVIDER.md`). It replaces the old composition:

```text
retained_residual + send_span + transport + actor_reserve
```

The GT-free-residual and final-v3 family edge-pool interim design was
superseded by the split-host amendment before any code used it.

## Training semantics

- A timely success (`L_op <= 170 ms`) has reward
  `r = Q_perc - 0.25 * L_op / 170`.
- Transport failure or `L_op > 170 ms` gives `r = -1`.
- Q_perc enters only the offline training reward. It never enters the
  20-feature state, the operational prior or any live ACK.
