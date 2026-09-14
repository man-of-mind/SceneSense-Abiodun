# SplitFusion map-freshness policy synthesis

## Scientific decision

Strict latest-only is the edge scheduling policy. It never interrupts active CUDA work; when the worker becomes free it processes only the newest pending frame and records older pending frames as `SUPERSEDED_PENDING`. There is no 25-ms expiry and no FIFO fallback.

The current SPLIT surface does not reliably satisfy 150, 200 or 250 ms. The measured LOCAL proxy changes that boundary, but it does not remove the need for an explicit vehicle-compute cost or hardware-availability state.

## LOCAL result

The current FCOS full-local path takes 26.55 ms at p50 and 29.06 ms at p95 on the RTX 5090 proxy. Its compact p025 object result is about 10 KiB median. All 1,200 one-shot messages installed across the four OAI profiles; there were no duplicates, reassembly expirations or rejected identities.

The direct LOCAL clock starts when the normalized seven-channel tensor is ready. It is not physical sensor-capture AoI. To expose that boundary honestly, the table below adds constant offsets from 600 retained live preparation rows: 48.11 ms p50 and 99.08 ms p95. These are sensitivity cases, not matched LOCAL remeasurements.

| Profile | Budget | Best SPLIT fresh map | LOCAL + prep p50 | LOCAL + prep p95 stress |
|---|---:|---:|---:|---:|
| Favorable Stable | 150 ms | 8.9% (a71) | 36.1% | 0.0% |
| Favorable Stable | 200 ms | 37.4% (a71) | 84.8% | 35.2% |
| Favorable Stable | 250 ms | 68.9% (a71) | 98.5% | 83.9% |
| Mid Variable | 150 ms | 6.9% (a65) | 35.7% | 0.0% |
| Mid Variable | 200 ms | 33.3% (a65) | 84.1% | 34.8% |
| Mid Variable | 250 ms | 66.2% (a65) | 98.2% | 83.2% |
| Adverse Stable | 150 ms | 1.4% (a65) | 32.6% | 0.1% |
| Adverse Stable | 200 ms | 10.1% (a65) | 79.9% | 31.7% |
| Adverse Stable | 250 ms | 30.5% (a65) | 96.6% | 79.0% |
| Fade Recovery | 150 ms | 6.8% (a65) | 33.0% | 0.0% |
| Fade Recovery | 200 ms | 33.9% (a65) | 81.8% | 32.1% |
| Fade Recovery | 250 ms | 66.4% (a65) | 98.5% | 80.8% |

The p50-preparation LOCAL sensitivity exceeds the best raw SPLIT result in every profile and budget. At the p95 preparation stress, LOCAL is effectively unable to meet 150 ms, is similar to the best SPLIT region around 200 ms, and remains clearly stronger at 250 ms. The preparation path is therefore part of the policy boundary, not a harmless constant.

## What this means for PPO

If the reward contains only quality and freshness, `LOCAL` is nearly dominant: it has full-FP32 quality, a small compact uplink and much fresher map state. A useful controller must also charge measured vehicle compute/energy and expose local accelerator availability, occupancy, temperature or battery state. Otherwise PPO will simply learn `always LOCAL`.

For SPLIT, the network profile label is never a policy input. It is an analysis stratum. The recurrent policy observes causal SNR/MCS/capacity trends, current map AoI, time since useful install, edge busy/pending state, previous action and terminal outcome. The LSTM can infer the latent profile and its direction without seeing future trace values.

Use separate person and vehicle utilities. For budget `B`, a suitable starting reward is:

```math
r_t = (w_p Q_p(a_t)+w_v Q_v(a_t))F_B(\mathrm{AoI}_{t+1})
      -\lambda_b b(a_t)/B_{\max}
      -\lambda_v C_{\mathrm{vehicle}}(a_t)
      -\lambda_e C_{\mathrm{edge}}(a_t)
      -\lambda_s \mathbf{1}[a_t\ne a_{t-1}]
```

where `F_B` is evaluated on the map state, not merely on an ACK. A superseded SPLIT frame earns no new-map utility, while already spent radio/compute cost remains charged; it receives no extra arbitrary discard penalty.

## Training-data consequence

The simulator should contain two empirical branches:

1. 72 SPLIT actions × four measured network strata under strict latest-only scheduling.
2. One `LOCAL_PROXY` branch with measured local compute and four compact-result transport distributions, plus a sampled shared sensor-preparation term.

The budgets 150/200/250 ms may be separate experiments or a context variable in the state. Start with separate fixed-budget experiments for interpretability. Do not train from profile names, average network profiles together, or subtract a single latency constant from the 288 cells.

## Limits

- LOCAL compute was measured on a desktop RTX 5090, not production vehicle hardware.
- LOCAL transport has one 300-sample run per profile; it establishes feasibility, not run-to-run variance.
- The LOCAL sink is the versioned compact object-map boundary, not the future multi-UE fusion server.
- Dense segmentation is not transported and earns no map utility.
- Person localization aging remains unavailable in the 288-cell retained data; only vehicle localization has aligned secondary evidence.
