# Run-4 live-qualification package v2 — Phase 1: restore and freeze the actor

**Verdict:** `RESTORED_BIT_IDENTICAL_AND_EXPORTED`
**Date:** 2026-09-29. **Starting HEAD:** `5b5435501af3b2c78f90d3ba8fd180eeb459fef5`.
CPU only. No CARLA, OAI, RFsim, Docker, softmodem, map server or network service ran.
`torch.cuda.is_initialized()` was `False` at export and in every test.

## What was done

1. `export_frozen_actor_v2.py` re-hashed the five pinned sources and read the checkpoint with
   `read_checkpoint`. It checked the canonical digest `aba44d2d…6672`, update 10,000, and the
   boundary actor digest. It also checked that `SEED_COMPLETE.json` binds the same seed,
   update, checkpoint and factory.
2. `smoke_runner.build_harness(transport_model_v2.json, 43)` then
   `ModeledSmokeOrchestratorV1.restore(...)` replayed the full ledger from genesis: 40,288
   decisions and 10,000 updates, in 2,429.7 s with 4 torch threads. `restore` refuses any
   boundary that is not bit-identical; the export additionally re-compared the full
   `BoundaryFingerprintV1`. **No training step was taken after restoration.**
3. The restored actor's digest, computed with the exact `_boundary()` definition, is
   `b61f27a9bcd3512ecf52bc35854f6a723d550092db51cf3055347297039cebd3`, which equals the
   expected boundary digest.
4. A cloned CPU `state_dict` was written with `torch.save`, along with a sealed manifest. The
   manifest records every source hash, the feature/action/model bindings, seed/update, the
   actor digest, a per-tensor inventory, and 20 deterministic batch-1 fixture outputs from the
   restored in-memory actor.
5. The deployment loader reloaded the file with `torch.load(..., weights_only=True)`. Tensors
   matched exactly and fixture outputs (mode, `q_e4`, head-tensor digests) matched exactly.
   The loader then froze the actor: `eval()`, `requires_grad_(False)`, inference mode.

## Artifacts

Untracked evidence (the `.pt` file is gitignored):
`rl_agent/experiments/splitfusion_hybrid_sac_live_route_b_v2/20260929_seed43_update10000_actor_export/`

| File | SHA-256 |
|---|---|
| `actor_state_dict.pt` (100,348 B) | `d064013d011b67dcd2c7c23acc3c396afe6750be0d43ef0204f2fbecbb9b8e29` |
| `ACTOR_EXPORT_MANIFEST.json` | `3a59d1caaa0f7cd26fe22d7ebeab4466c99ae80b1664d14d398cdcb91ffab5f6` (self-digest `7f17a025…5140`) |
| `RESTORE_RESULT.json` | `4b1f393c50a48dbc8427c9bc6f8ca60831f998789ff4aacd026f53af0e806415` |

The tracked `ACTOR_BINDING_V2.json` is a byte copy of the export manifest. It pins the weights
file by SHA-256, and `load_registered_actor()` refuses any difference.

## Deployment rule (`FrozenRun4ActorV2`)

CPU float32, batch 1, `eval()`, `torch.inference_mode()`. The rule is
`ConditionalHybridActor.deterministic_execution`, unchanged: argmax categorical mode, then the
selected mode's conditional mean mapped through its registered `MODELED_SMOKE_SUPPORT` interval,
then `action_contract.round_half_up_q_e4`. `act()` accepts only an attested
`PolicyFeatureVectorV2`. `act_on_vector()` accepts only an exact finite 21-tuple, refuses
train mode, and asserts that `q_e4` lies inside the selected mode's support. Global Python,
NumPy and Torch RNG state is unchanged, because construction uses the RNG-restoring
`build_actor` and inference samples nothing.

## Tests

`python -m unittest rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_phase1_frozen_actor_v2`
→ **21/21 OK**. Coverage:

- source re-verification and boundary equality;
- the restore-result record, and the tracked binding as a byte copy of the export;
- exact `weights_only` reload, fixture equality and repeatability;
- independent recomputation of the deployment rule;
- frozen/eval/CPU/float32 checks and train-mode refusal;
- global-RNG neutrality and malformed-vector rejection;
- attested-feature requirement;
- the tamper matrix: unsealed edit, resealed fixture edit, seed 17/29, update 1,500, swapped
  feature order, wrong actor digest, and an altered tensor (one ULP) with either the old
  digest or a re-pinned digest and inventory.

Regression, all unchanged and OK:

| Suite | Tests |
|---|---|
| `splitfusion_hybrid_sac_run4_v1.test_run4_contract` | 23 |
| `test_models` | 6 |
| `test_state_adapter` | 25 |
| `test_production_state_provider` | 17 |
| `splitfusion_live_dispatch_v1.test_dynamic_execution_contract` | 20 |

## Files

Created: `__init__.py`, `frozen_actor_v2.py`, `export_frozen_actor_v2.py`,
`test_phase1_frozen_actor_v2.py`, `ACTOR_BINDING_V2.json`, `PHASE1_REPORT.md`.
Modified outside this package: none.

## Scope

This is a post-hoc development choice for a bounded qualification. The equality proven here is
reconstruction equality. It is not evidence of convergence, optimality or live behaviour.
