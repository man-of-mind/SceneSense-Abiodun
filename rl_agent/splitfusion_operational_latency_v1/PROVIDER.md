# Shared operational-latency provider v1

Label: `EXPLORATORY_POOLED_FAMILY_TRANSFER_ASSUMPTION`. Shared by Run-4B and
any later Run-5B, which must import `provider.py` unchanged and bind the same
`binding_sha256`. Binding evidence: `PROVIDER_BINDING.json`
(regenerate with `python -m rl_agent.splitfusion_operational_latency_v1.provider`).

```text
L_op = A + S + T + E + D        timely iff transport success and L_op <= 170 ms
```

| Term | Boundary | Source | n | p50 / p95 / p99 ms |
|---|---|---|---|---|
| A | action open -> first socket send | split-host POLICY_DECISION rows | 138 | 27.62 / 55.81 / 62.76 |
| S | first -> last socket send | 0.513047 ns/byte | - | deterministic |
| T | last send -> complete reassembly | transport v2b | - | modeled |
| E | complete reassembly -> publish start | split-host reward-requested POLICY_DECISION rows | 127 | 45.16 / 51.74 / 64.54 |
| D | evaluation completed -> UE receipt | retained action-50 probe rows | 555 | 5.46 / 9.71 / 16.74 |

- **A** already includes actor inference: live action open is stamped inside
  `build_state`, then `actor.act` runs. There is no actor reserve.
- **E exclusions:** 148 POLICY_HOLD rows (kept only as the
  `E_HOLD_SENSITIVITY` pool, never composed), 8 fallback ingest rows with no
  frame record, and 11 reward-requested decisions without an ingest row.
- **D exclusions:** 45 rows with a missing or non-finite value.
- **Excluded from L_op:**
  - GT wait;
  - GT scoring;
  - Q_perc computation;
  - map installation;
  - prediction-ready -> evaluation-enqueue;
  - the old actor reserve.
- **Pooling:** A and E are pooled across families. The split-host actor chose
  noAE in 1 of 138 decisions and AE128 in 7, so no family-specific
  distribution is built. On the earlier edge host, noAE edge service was
  roughly 2x AE32, so pooled E probably understates noAE latency.
- **Source status:** the split-host run ended `FAILED` on an
  evaluator/ground-truth fault. It is characterization, not qualification;
  the fault is outside the A/E boundaries.
- **RNG:** A, E and D use three independent `random.Random` streams, seeded
  from `(master seed, label)`. Each draws one index per decision before the
  action is known, and stream state is exported for exact resume.
