# Scientific findings for policy design

## Decisive queue result

With final edge service, FIFO has a cell-median p95 queue wait of 256.3 ms, a worst wait of 10.29 s and a maximum compute backlog of 103 frames. Its median-of-cell queue medians is 0 ms because many heavy-payload cells admit frames
too sparsely to queue; that zero must not be presented as absence of
FIFO congestion.

Strict latest-only caps pending compute at one frame and lowers the median cell map AoI from 371.6 to 308.0 ms. It processes fewer eventual updates, but improves timely useful-update yield and fresh-map time at
all three budgets. This is exactly the intended trade: discard obsolete
pending work, not the latest available pending frame.

## Freshness budgets

| Budget | Best favorable fresh-map time | Best adverse fresh-map time | Interpretation |
|---:|---:|---:|---|
| 150 ms | 8.9% | 1.4% | No split action is a credible hard-budget fallback |
| 200 ms | 37.4% | 10.1% | No split action is a credible hard-budget fallback |
| 250 ms | 68.9% | 30.5% | Useful split region appears, but still not continuously reliable |

The 150 ms budget is an aggressive diagnostic. The best split action
keeps the map inside it for less than one tenth of favorable route
time and about one percent of adverse route time. At 200 ms the best
actions remain below 40% favorable and near 10% adverse. At 250 ms
the best favorable action reaches roughly two thirds, but the best
adverse action remains near one third. A local-inference action is
therefore empirically motivated if any of these is a hard safety
budget; split inference remains useful for softer cooperative-map
objectives.

## Action-conditioned network behavior

Intrinsic validation quality is held fixed for an action. The network
changes whether and when that quality becomes available. Hence the
policy model must use P(outcome | action, network state), not a network
average over all actions. The heatmaps and fixed-action figure retain
all 72 action × four profile cells.

The quality-weighted scores are decision surrogates:

```math
M^{(c)}_{a,n}(B)=Q^{(c)}_a\,C_{a,n}(B)
```

They are not claims of newly measured network-specific model accuracy.
Person and vehicle scores remain separate; no unsupported weighting
between them is introduced.

## Aligned localization limitation

The retained live rows provide 252,327 vehicle rows with
both source-time and aligned-retrieval localization error, but
0 corresponding person rows. Person localization
versus staleness is therefore unsupported by this campaign and is not
silently inferred. Vehicle aligned error generally increases in the
older AoI bands, but the result is secondary because source/aligned
matching support can differ and aligned truth is sampled during exact
record retrieval after ACK rather than at an exact install interrupt.

## Consequences for the controller

The causal state should include lagged network state, current map AoI,
last useful installation time, edge busy/pending state, recent service
EWMA, previous action and terminal outcome. `SUPERSEDED_PENDING` is not
radio loss: it earns no installation utility, while bytes and compute
already spent remain charged. This gives PPO the correct delayed credit
signal without inventing a discard penalty.

Before policy training, a LOCAL action needs a matched measurement of
full local compute, quality, compact object-result bytes and map-install
latency. Local perception and local-to-cooperative-map publication must
remain distinct boundaries.
