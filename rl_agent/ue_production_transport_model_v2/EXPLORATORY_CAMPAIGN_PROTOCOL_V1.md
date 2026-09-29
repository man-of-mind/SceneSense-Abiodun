# Run-4 exploratory three-seed campaign protocol v1

This protocol authorizes only the bounded offline exploratory campaign needed
to train and inspect the agreed Hybrid-SAC agent. It does not claim
confirmatory validation, convergence, production readiness, or deployment
authorization.

## Scientific components held fixed

The campaign must not change:

- the 21-feature causal state contract (camera SI, radar P40, prior UE UL MCS,
  pre-action RLC backlog, and previous action/outcome features);
- the hybrid action `(mode, q)` and its registered per-mode continuous-q
  support;
- the agreed perception-quality/170 ms feedback reward;
- the v2 production transport artifact;
- SAC architecture, entropy coefficients, learning rates, batch size,
  transitions per update, replay capacity, or Polyak coefficient; or
- registered seeds `17, 29, 43` and checkpoint updates
  `0, 100, 250, 500, 1500, 10000`.

## Mandatory pre-continuation gate

Seed 17 must start from genesis and stop at update 500 before any longer run.
Continuation is allowed only when all of the following hold:

1. the 288-decision causal preflight passes;
2. every update metric is finite;
3. the update-500 canonical checkpoint exactly reproduces the retained
   `18e678ddf4e762a0cfcf244babea3e84cd91059fa026d9a74330ffc5260c20b2`;
4. a controlled backlog-only sweep, expressed in executed q and the combined
   reward-requested-plus-held-frame ingress bytes, has a 95% interval strictly in the physically expected
   direction: higher backlog increases drop fraction and decreases bytes; and
5. a fresh Python process restores update 250 and produces a byte-identical
   update-500 checkpoint.

This replaces the earlier unsupported statement that every original
fixed-panel continuation gate had been evaluated. The original preregistration
remains preserved. This bounded campaign is explicitly post-hoc exploratory
because the v2 transport model was developed from inspected evidence.

## Retention

Each seed retains every update metric, per-decision diagnostics, and complete
canonical event-sourced checkpoints at all registered boundaries. Any gate
failure stops the campaign; thresholds are not retuned after observation.
