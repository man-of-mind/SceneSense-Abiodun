# Claude prompt — SplitFusion Hybrid-SAC Phase 4b.2

Implement the smallest scientifically valid replay-buffer and one-update
Hybrid-SAC trainer smoke test. This phase proves numerical and contract
mechanics only. It is **not** deployable policy training, a convergence result,
or evidence that Hybrid SAC improves SplitFusion.

## Preconditions and safety

1. Read `CLAUDE.md` and these files completely before acting:
   - `experiments/splitfusion_supervisor_analysis_v1/20260917_hybrid_sac_architecture_v1/DESIGN.md`
   - `rl_agent/splitfusion_hybrid_sac_v1/action_contract.py`
   - `transaction_identity.py`
   - `reward_ticket_controller.py`
   - `state_reward_transition_contract.py`
   - `synthetic_contract_environment.py`
   - `anchor_store.py`
   - `hybrid_sac_models.py`
   - their tests.
2. The audited model-boundary commit `1e15899` must be an ancestor of the
   current HEAD. Verify that ancestry and report the actual full HEAD plus
   `git status` before editing. The later prompt-only handoff commit is
   expected; stop if any other unreviewed source change follows `1e15899`.
3. Preserve the two user-owned dirty paths byte-for-byte and never stage them:
   `OAI/openairinterface5g` and
   `pole_lraspp_multimodal_fusion/object_head_pilot_v1/lraspp_to_splitfusion_fcos_report_v1/FULL_TECHNICAL_REPORT_AVO_V2.md`.
4. No pull, merge, rebase, reset, push, network, CARLA, OAI, Docker, CUDA,
   evidence mutation, or global environment changes. CPU-only deterministic
   work. Imports must not read evidence, seed a global RNG, construct models,
   or launch anything.
5. Do not edit a frozen contract to make training code convenient. Stop and
   report a genuine contract inconsistency.

## Non-negotiable scientific boundary

- The production replay store accepts only an exact `ReplayTransitionV1` that
  revalidates, is learning-eligible, and carries a finite scalar reward.
- Never adapt the 288 aggregate anchors, `ActionQualityAnchor`,
  `NetworkProfileOutcome`, `SyntheticFixtureValueRecord`,
  `SyntheticRunReport`, or `SyntheticPolicyObservation` into production
  replay. They are not causal transitions.
- Exact-positive reward transitions remain fail-closed until authenticated
  per-frame quality/protocol-v2 exists. Do not weaken that gate.
- Production replay integration tests may construct legitimate
  `ACTION_PATH_FAILURE` transitions through the public contracts.
- If positive synthetic tensors are needed to test optimizer direction, keep
  them test-only, label them `SYNTHETIC_HYBRID_SAC_SMOKE_TEST_ONLY`, and prove
  the production replay store rejects them.
- Never interpolate unsupported continuous-q quality or describe a synthetic
  value as measured.

## Phase A — read-only plan; stop for Codex review

Before editing, report:

1. the exact `ReplayTransitionV1` fields to tensorize;
2. the terminated/truncated/next-state bootstrap truth table;
3. homogeneous bindings enforced per buffer;
4. duplicate and eviction policy;
5. trainer update order and RNG streams;
6. proposed files and tests.

Stop after this report. Do not begin Phase B until explicitly approved.

## Phase B — production in-memory replay boundary

Add only:

- `rl_agent/splitfusion_hybrid_sac_v1/replay_buffer.py`
- `rl_agent/splitfusion_hybrid_sac_v1/test_replay_buffer.py`

Do not edit `__init__.py` or an existing contract.

Implement bounded deterministic `ReplayBufferV1` and immutable
`ReplayTensorBatchV1`.

### Insertion

1. Require exact `ReplayTransitionV1` type.
2. Call its attestation/revalidation path on every insertion.
3. Require learning eligibility and finite `scalar_reward`.
4. Re-prove that stored policy features bind their state and optional next
   state.
5. Freeze the first record's reward-spec, normalization, freshness, schema,
   policy-feature-order/count and `gamma_per_tensor` bindings; reject mixtures.
   Different sessions and actor versions remain allowed because SAC is
   off-policy.
6. Reject every duplicate explicitly. Retain a lifetime in-memory seen-digest
   set even after capacity eviction. Document that it is not durable dedup.
7. Positive integer capacity; deterministic oldest-first eviction.
8. Sample uniformly without replacement using an explicit local
   `torch.Generator`; never touch global RNG.

### Tensor batch

Retain:

- state `[B,31]` float tensor in frozen feature order;
- executed `mode_id` int64 `[B]`;
- executed critic action `q_e4/9800` `[B]`—never sampled q;
- scalar reward, integer hold duration `d >= 2`;
- next-state tensor plus a separate `has_next_state`/bootstrap mask;
- terminated and truncated masks separately;
- transition hashes and transaction identities as audit metadata only;
- uniform gamma and contract hashes as batch metadata.

Bootstrap iff an exact next state exists and the transition is not a true
terminal. A truncation may bootstrap only with an exact next state. The trainer
must never evaluate a zero-filled rectangular sentinel as if it were a real
next state.

Tests must cover valid public-contract-built failure transitions; rejection of
all aggregate/synthetic/unattested/censored inputs; mutation detection; mixed
binding rejection; cross-session/off-policy acceptance; duplicate rejection
after eviction; deterministic sampling/eviction; exact state/mode/q/reward/d
tensorization; the full bootstrap truth table; no IDs in policy features; and
an import side-effect audit.

Run focused tests, the full Hybrid-SAC suite, `py_compile`, and
`git diff --check`. Commit Phase B separately, report hashes/counts, and stop.

## Phase C — exactly one trainer update

Only after review of Phase B, add:

- `rl_agent/splitfusion_hybrid_sac_v1/hybrid_sac_trainer.py`
- `rl_agent/splitfusion_hybrid_sac_v1/test_hybrid_sac_trainer.py`

Implement immutable `TrainerConfigV1`, `UpdateMetricsV1`, and
`HybridSacTrainerV1.update_once`. Do not add a long-running training loop.

Use explicit fixed positive `alpha_d` and `alpha_c`; no automatic entropy
tuning. Gamma must match the replay binding. Minimal smoke hypotheses may use
Adam actor/critic learning rates `3e-4`, batch 256 and Polyak `tau=0.005`, but
label these provisional rather than scientifically frozen.

Exact update order:

1. Validate all batch/config shapes, dtypes, devices, bindings and finiteness
   before any mutation.
2. Under `no_grad`, evaluate `soft_state_value` only for bootstrap-eligible
   rows, scatter into a zero vector, and compute
   `y = r + bootstrap * gamma**d * V(s')`.
3. Evaluate online Q1/Q2 at the executed mode and `q_e4/9800`; minimize the sum
   of the two MSE losses; validate gradients; step the critics.
4. Clear critic gradients. Compute the existing exact-enumeration
   `actor_objective`; validate gradients; step only the actor. Assert critic
   gradients remain absent while `dQ/dq` still reaches the actor.
5. Polyak-update both targets exactly once after successful online updates.
   No target actor and no target parameters in an optimizer.

Use separate explicit RNG streams for replay sampling, target-q sampling,
actor-q sampling and evaluation. Do not globally reseed per update.

Return finite diagnostics including both critic losses, actor loss, reward and
target ranges, Q means/twin gap, mean duration, bootstrap fraction, discrete
entropy, conditional log-probability/entropy estimate, requested/executed q
range and saturation, gradient norms, and actor/online/target parameter-delta
norms.

Tests must prove one deterministic finite update; correct `gamma**d`; online
actor/critics change; target change is exactly the Polyak equation; no critic
gradient contamination during the actor step; terminal/no-next rows never
reach target evaluation; exact resume/determinism for identical seeds;
different seeds diverge; all 12 actor branches receive gradients; executed q
is used; malformed/NaN input causes zero parameter and optimizer-state
mutation; targets are absent from optimizers; and replay-to-update integration
uses only genuine eligible contract records.

Run focused and full tests, `py_compile`, and `git diff --check`. Commit Phase C
separately and stop.

## Explicitly out of scope

No long training run, convergence claim/plot, checkpoint system, prioritized
replay, automatic entropy tuning, GPU benchmark, protocol-v2, continuous-q
quality interpolation, 288-anchor replay, live integration, LOCAL/SKIP,
GRU/LSTM, or scientific results. The final report must say “mechanics exercised
successfully,” never “agent trained.” Nothing is pushed.
