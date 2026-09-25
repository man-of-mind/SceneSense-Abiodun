# Run-4 modeled-composite offline runtime

Status: implemented as a separate offline-only training boundary.

This integration exists because the final Run-4 training generator combines
measured sources, fit-derived dynamics, and deterministic contract transforms.
Those records must not be relabelled as calibrated empirical evidence. The
production empirical replay and production trainer remain unchanged.

## Evidence path

The only accepted input is an exact, attested
ModeledCompositeOfflineTransitionV1 produced by the modeled-composite
contract. The wrapper remains intact at the admission boundary.

The handoff to tensorization is package-private:

1. Re-attest the outer modeled wrapper.
2. Require the exact modeled-composite binding SHA-256 selected by the buffer.
3. Re-attest the enclosed SemiMarkovTransitionV2.
4. Require the enclosed transition digest to equal the public wrapper digest.
5. Tensorize only after all checks finish.

There is no public method that returns a bare transition for replay. The
production ReplayBufferV1 still accepts only the exact bare empirical
SemiMarkovTransitionV2 type and therefore rejects the modeled wrapper.

## MCS dynamics prerequisite

Construction of ModeledReplayBindingV1 calls
mcs_transition_acceptance.load_registered_acceptance(). This hash-verifies the
sealed report and its source files, recomputes the held-out result, and requires
the registered result to be accepted.

The modeled replay binding then seals all of the following:

- registered MCS acceptance-result SHA-256;
- accepted MCS model-binding SHA-256;
- accepted MCS source-evidence SHA-256;
- the non-deployment acceptance evidence class;
- the full modeled-composite binding and provider/verifier digests.

The UL_MCS_TRANSITION disclosure inside the modeled-composite binding must name
that exact accepted source as source_evidence_sha256 and that exact accepted
model as fit_support_sha256. Reconstructing a provider without the registered
acceptance is insufficient.

This held-out internal qualification is not a deployment or unseen-channel
generalization claim.

## Dedicated replay boundary

ModeledCompositeReplayBufferV1 is independent of ReplayBufferV1. It accepts
only exact modeled wrappers and retains:

- transition, envelope, support-use, and latency-projection digests;
- modeled-composite binding digest;
- logical session, UE, and decision identity;
- exact executed mode and q_e4;
- exact contract reward, duration, and already-derived discount;
- current and real successor feature vectors;
- explicit episode-boundary masks.

The buffer does not store the raw transition after tensorization. Duplicate
transition digests and conflicting logical identities remain lifetime indexes,
including after FIFO eviction. Sampling requires an explicit private CPU
torch.Generator and never uses the global generator.

The resulting ModeledReplayTensorBatchV1 carries the modeled evidence class
and complete provenance into every audit row. Tensor and audit accessors return
copies.

## Dedicated trainer boundary

ModeledCompositeHybridSacTrainerV1 accepts only an exact
ModeledReplayTensorBatchV1 with an exact ModeledReplayBindingV1. It refuses
production empirical batches, and the empirical trainer refuses modeled
batches.

The SAC numerical update is not forked: update_once is inherited unchanged
from the tested _Run4TrainerCore. Only construction and preflight are
modeled-specific. Preflight rechecks:

- model architecture, dtype, device, and optimizer wiring;
- exact modeled evidence binding;
- tensor shapes, dtypes, finiteness, and action ranges;
- successor, boundary, bootstrap, and zero-sentinel consistency;
- minimum action-hold duration and stored-discount range;
- modeled evidence class and non-production audit flags.

The trainer consumes the sealed per-row discount verbatim. It does not
recompute gamma raised to duration.

## Offline runner and factory

ModeledCompositeOfflineFactoryV1 constructs CPU-only models, replay, trainer,
and three distinct private RNG streams. ModeledCompositeOfflineRunnerV1
provides only:

- ingest or ingest_many for already-attested modeled wrappers;
- train_once for a sampled offline minibatch;
- a non-production audit summary.

It has no CARLA, OAI, Docker, socket, map, CUDA, or live-deployment hook.

## Claims deliberately not made

This module does not claim:

- calibrated empirical replay;
- production or deployment authorization;
- measured runtime output;
- unseen-channel generalization;
- live temporal validation;
- that a modeled training result is a deployment result.

Those remain separate validation stages.

## Focused acceptance tests

The focused tests cover:

- registered MCS acceptance and source/model cross-binding;
- refusal when only an unaccepted/reconstructed MCS provider is presented;
- wrapper, inner-transition, and binding tamper detection;
- rejection of bare transitions by modeled replay;
- rejection of modeled wrappers by empirical replay;
- bidirectional trainer evidence-class separation;
- exact tensorization and complete audit provenance;
- no raw-transition storage or laundering export;
- lifetime duplicate rejection;
- clone isolation and global-RNG neutrality;
- CPU-only end-to-end modeled ingestion and one inherited SAC update.
