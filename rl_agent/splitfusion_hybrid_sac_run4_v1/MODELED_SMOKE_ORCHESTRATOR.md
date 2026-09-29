# Run-4 modeled smoke orchestrator

Status: implementation complete and tested with a pure typed fixture. This is
offline modeled training mechanics, not production or deployment evidence.

## Frozen execution

The orchestrator accepts only the registered Run-4 seeds 17, 29 and 43:

- 288 warm-up decisions per seed: 12 modes × 6 continuous-q strata × 4 samples;
- four new modeled transitions before each SAC gradient;
- batch size 256; and
- checkpoint boundaries 0, 100, 250, 500, 1,500 and 10,000.

`run_to_hard_stop()` preserves the mandatory seed-17 update-500 smoke stop.
`run_to_registered_update()` is the bounded continuation seam and refuses any
seed or target outside the registered sets.

Before update 1, the no-gradient preflight requires exact action identity,
complete 12×6×4 coverage, both successful and failed/timeout outcomes,
finite non-degenerate state and reward populations, duration `d=2`, the
registered MCS-acceptance binding, the exact modeled-composite binding, and no
validation evidence. It also proves that model and optimizer fingerprints did
not change while the 288 rows were collected.

## Collector boundary

The collector must implement `ModeledTransitionCollectorV1`. Each call returns
an exact attested `ModeledCompositeOfflineTransitionV1` plus causal
diagnostics. The orchestrator verifies the wrapper's already-existing sealed
modeled-replay hand-off, the requested/executed `(mode_id, q_e4)`, state
features, `d=2`, MCS acceptance, modeled-composite provenance and fit-only
partition before replay mutation.

The real composite collector lives in
`ue_production_transport_model_v2/collector_v1.py`. A caller cannot substitute
a mapping, bare transition, calibrated empirical row, held validation row, or
synthetic production fallback.

## Durable restore without private replay access

`modeled_offline_runtime.py` deliberately exposes no replay-deque or private
RNG snapshot. The orchestrator does not reach into those fields. Its durable
checkpoint is an event-sourced canonical JSON record containing:

- the collector's canonical checkpoint and exact transition history digests;
- the complete action/terminal/reward ledger;
- factory seed plan and modeled/replay binding documents;
- schedule position and the immutable warm-up report; and
- complete actor, twin-online/target-critic, optimizer, replay and
  decision-RNG boundary fingerprints.

Restore creates a fresh exact `ModeledCompositeOfflineRunnerV1` through
`ModeledCompositeOfflineFactoryV1`, restores the collector, regenerates every
attested wrapper and deterministically replays the collection/update schedule
from genesis. This reconstructs the replay FIFO and lifetime indexes, both
optimizers, all model weights, update count, and the replay/actor/target and
decision RNG stream positions. Continuation is refused unless the reconstructed
boundary is bit-identical to the stored checkpoint.

This costs replay time during recovery, but it preserves the modeled replay's
encapsulation and removes a second mutable snapshot format. Complete
checkpoints are retained only at the six registered boundaries.

## Verification

The focused tests cover:

- exact balanced schedule construction;
- the 288-transition no-gradient preflight;
- MCS-binding and validation-evidence refusal before replay mutation;
- create-only checkpoint publication, canonical read-back and tamper refusal;
- global PyTorch RNG neutrality; and
- uninterrupted update 500 versus restore-from-update-250 continuation with
  identical final checkpoint and every boundary fingerprint.

The tests are CPU-only. They do not launch CARLA, OAI, RFsim, Docker, CUDA or
network services and use only an explicitly labelled fake typed collector.
