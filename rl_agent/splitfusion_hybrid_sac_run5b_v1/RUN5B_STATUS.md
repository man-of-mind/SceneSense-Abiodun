# Run-5B status (2026-09-30): scaffolding complete; launch held

**Status: `RUN5B_SCAFFOLD_COMPLETE__LAUNCH_HELD_PENDING_RUN4B_OPERATIONAL_LATENCY_PROVIDER`.**

No smoke, deep training or evaluation has run. The training preregistration is **not sealed**.

## Built and tested offline (CPU only)

- **State: 21 features.**
  - Positions 0–19 are the Run-4B order, i.e. the Run-4 order with
    `prev_quality_qperc` removed.
  - Position 20 is `effective_external_ul_snr_proxy_scaled`, using the frozen Run-5 v2
    lease provider and the `(snr_db − 5.5)/19` scaling.
- **Transport-only prior.** The previous outcome is a `TransportPriorOutcomeV1`.
  - It holds the action, the terminal and the operational latency.
  - It has no quality field, and a successful prior needs no Q_perc.
  - The modeled projection never reads Q_perc: two resolutions that differ only in
    Q_perc give byte-identical priors.
- **Native builder.** The vector is built by `guard_run5b_state` plus
  `build_run5b_policy_features`. No Run-4 or Run-5 builder is called and nothing is
  sliced (a test patches both to fail).
  - An audit compares the result bit for bit with the environment's own measured
    values.
- **Models.** The actor takes 21 inputs and the critics 34. The hyper-parameters,
  cadence and seeds equal Run 4's and Run 5's.
- **Live actor.** Seed 43 at update 10,000 is pre-registered.
- **Identity checks.** These cover the schema id, feature order and its hash, the feature
  schema hash, the model binding, the preregistration and the tensor tree.
  - Refused: the old Run-4 21-D actor (its binding, schema and order, plus the frozen
    seed-43 tensors) and the old Run-5 22-D actor (its bindings, schemas, order,
    preregistration and width, sliced or padded).
  - Width alone is never treated as identity. Untrained Run-4 and Run-5B actors with the
    same init seed are bit-identical, and a test demonstrates this.
- **Tests: 52 pass, 7 skipped.** The skipped tests are the separate-process resume and
  smoke tests, which need the sealed preregistration.
  - With the same seed and actions, the outcomes are identical to the frozen Run-5
    collector, and the Run-5B state equals the Run-5 state minus feature 17.
  - Perturbing Q_perc changes the rewards but never the actor state.

## Why launch is held

The inherited Run-4 v2 kernel composes latency from a retained residual that includes
ground-truth evaluation and scoring time.

`run5b_campaign` therefore refuses smoke, deep and rehearsal runs with
`RUN5B_LAUNCH_HELD_PENDING_RUN4B_OPERATIONAL_LATENCY_PROVIDER` while
`RUN4B_LATENCY_PROVIDER is None`.

## To resume

1. Import Run 4B's hash-bound operational-latency provider and its binding unchanged. Do
   not reimplement it.
2. Swap it into the collector kernel.
3. Add a test that Run 4B and Run 5B produce identical latency draws for identical seed,
   state and action.
4. Seal `RUN5B_TRAINING_PREREGISTRATION.json`, then run the smoke and recovery gates.
