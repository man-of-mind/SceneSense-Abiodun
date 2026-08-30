# SplitFusion recovery implementation review packet

## 1. Commit boundary and changed files

Implementation start: `8caef38f6312add8a9a986a00217c7f98b275bd2` on local `master`. The implementation commit is the local `master` commit containing this packet; resolve it with `git rev-parse HEAD` after checkout. Git commit hashes cannot self-identify inside their own committed bytes, so the machine-verifiable binding used by qualification is the resolved commit plus `canonical_hash(package_hashes())`.

Independent-review remediation is based only on `0c7cab240d7149e4d313c101ac13d52b5b0f8c18`: aggregate rather than per-update reachability, streaming breaker reductions, a shared disposable update-447 boundary for candidates, and verified epoch-boundary scientific resume. No scientific equation or selection contract changed.

All changes are confined to this new package. Changed files are: `__init__.py`, `README.md`, `NUMERICAL_OPERATION_AUDIT.md`, `IMPLEMENTATION_REVIEW_PACKET.md`, `IMPLEMENTATION_REVIEW_PACKET.json`, `recovery_config.json`, `contracts.py`, `base_runtime.py`, `safe_math.py`, `audit.py`, `state_guard.py`, `guards.py`, `envelope.py`, `recovery_model.py`, `recovery_losses.py`, `replay.py`, `runner.py`, `precision_compare.py`, `qualify_recovery.py`, `continue_scientific.py`, `infer_recovered.py`, `evaluate_recovered.py`, `operation_inventory.py`, `tests/__init__.py`, and `tests/test_synthetic.py`.

## 2. Original immutability

The original package and experiment were read only. Original config SHA-256 remains `91889e4af2a5088853d192c4c16c39249c58e44b41b566e0e6d5586e5f717631`; epoch-9 SHA-256 remains `9aa3c1c1ad87889c730ff2ac0c936ed5b64ea23fd90c14ba3dbd16743046b2d4`. `git diff` reports no change under either original path. Pre-existing `OAI/openairinterface5g` dirt is excluded and untouched.

## 3. Equation and complete numerical-operation mapping

`NUMERICAL_OPERATION_AUDIT.md` maps D/G/S/A equations, normalizations, every scientific division, log/log1p, exp/expm1, sqrt, softmax, FCOS box decode, camera/world projection, score equation, and frozen evaluator division. `operation_inventory.py` provides repeatable AST enumeration. Only yaw normalization and invalid-range dimension-exp behavior differ; targets, losses, weights, architecture, and valid-value dimension equation do not.

## 4. Train/inference symmetry

Training `recovery_losses.geometry_losses` and inference `recovery_model._decode_geometry` both call `safe_math.normalize_yaw_fp32`. Tau is validated against the immutable candidate set. No independent inference epsilon, fallback, or magnitude penalty exists.

| Property | Training | Inference |
|---|---|---|
| Shared function | `normalize_yaw_fp32` | `normalize_yaw_fp32` |
| Raw dtype entering function | cast to FP32 | cast to FP32 |
| Norm/divisor | scaled FP32 L2 / `clamp_min(tau)` | identical |
| Tau source in scientific use | hash-bound qualified config via model construction | same qualified config and checkpoint binding |
| Raw-norm/affected diagnostics | carrier and tensor audit | decoded output audit |
| Fallback or target change | none | none |

## 5. Precision table

`precision_compare.py` compares BF16-tail and FP32 train-only execution on exactly the original eight registered calibration batches, including group/component losses, C2 gradients, yaw distributions, required parameter gradients, ratios, ordering divergence, output finiteness, and exact diagnostic state restoration. It cannot step or recalibrate D/G/S/A.

| Region | FP32 comparison | Registered BF16 path | Recovery rule |
|---|---|---|---|
| Seven-channel front/C2 | FP32 | FP32 | unchanged |
| Tail convolutions | autocast disabled | BF16 CUDA autocast | compare only; never recalibrate |
| Detection/geometry/semantic/auxiliary losses | FP32 | FP32 autocast-disabled loss block | unchanged |
| Yaw normalization | FP32 | FP32 | shared recovery function |
| World/dimensions | FP64 | FP64 | representability audit, no clamp |
| Transport | FP32 identity | FP32 identity | unchanged noAE |

## 6. Optimizer groups and proposed update

The breaker requires exact, nonoverlapping coverage by the following groups. It computes FP64 scalar L2 reductions for gradient/momentum/proposed-delta values from the registered SGD settings, including weight decay, dampening, nesterov/maximize semantics, and existing momentum buffers, without mutation. Each parameter is processed and released before the next; no FP64 parameter/gradient/momentum copies or proposed-update lists are retained. It also records global gradient norm and worst named parameter-relative delta.

| Optimizer group | Locked base LR | Exact membership rule |
|---|---:|---|
| `pretrained_backbone` | 0.001 | `front.W_rgb`, `front.bn1`, `front.layer1`, `tail.layer2`–`tail.layer4` |
| `pretrained_fpn_heads` | 0.0025 | official P3–P7 FPN, classification tower, FCOS regression head |
| `new` | 0.01 | radar stem, P2 path, copied two-class classifier, geometry, semantic and dense heads |

All use SGD momentum `0.9`, weight decay `0.0001`, and the unchanged absolute epoch schedule. The copied classifier deliberately remains in `new`.

## 7. Breaker placement

`runner.run_guarded_epoch` preserves loss finiteness, per-update required-gradient finiteness, post-step model finiteness, and post-step optimizer finiteness checks. An isolated exact-zero required gradient is written in each update record and never aborts. Qualification aggregates those records and requires every required trainable group to be nonzero at least once across the full disposable epoch10 plus epoch11-prefix range. `PreStepBreaker.check` remains after all four physical microbatches and before `optimizer.step`; violations preserve sample IDs, write a structured record, and raise with no clip, skip, step, or loss-magnitude criterion. Synthetic hash tests and audited update-447 boundary hashes prove non-mutation without hashing full state on each scientific update.

## 8. Candidate selection and healthy envelope

Candidates remain `[1e-5,1e-4,1e-3,1e-2]`; selected tau and all ceilings are null. Later qualification reproduces the original update 447 twice, captures one exact pre-update-447 model/optimizer/RNG state and four microbatches, evaluates every tau once at that common state with zero candidate steps, and repeats only the selected candidate boundary once. Selection remains fewest affected carriers then smallest tau. The only disposable optimizer steps are `2×446 + 1052 + 32 = 1976`. The full repaired epoch10 plus first 32 epoch11 updates must pass every finite/guard check and aggregate reachability, all audits, and the 12 GiB cap. No passing candidate creates `CONTRACT_INVALID_NO_CANDIDATE` and cannot qualify.

## 9. State-restoration proof design

`DiagnosticStateGuard` snapshots parameters, buffers, optimizer state, Python/NumPy/Torch/CUDA RNG, every module training mode, every parameter `requires_grad`, and existing gradients. Restoration happens in `finally` and exact pre/post hashes are recorded. FrozenBatchNorm and GroupNorm are accounted through state, modes, parameters, and buffers.

## 10. Deterministic replay design

The replay names epoch-9 explicitly, verifies its path/hash/schema/epoch/global/config and required model/optimizer/scheduler/RNG/sampler keys, checks finite model/buffer/optimizer state, restores optimizer and RNG, recreates epoch10 sampler/augmentation, steps only disposable updates1–446, verifies 16 preregistered IDs, and runs update447 forward/backward with no step. It isolates four microbatches and records all requested identities, losses, gradients, extrema, boundary state hashes, and proposed updates. Full model/optimizer hashes are computed only at the audited boundary. Two original reproductions use numerical tolerance, not byte-identical CUDA.

Scientific interruption recovery is explicitly requested with `--resume-existing-epoch-boundary`. It rejects missing, partial, noncontiguous, symlinked, breaker-failed, terminal, or hash-drifted state; verifies every completed checkpoint and sidecar against source, qualification, provenance, schedule, sampler, and exact global counter; restores the latest verified checkpoint model/optimizer/RNG; and begins only at the next epoch. New output remains create-only from the verified original epoch 9, and checkpoint writes refuse overwrite.

## 11. Tests run

- `python3 -m py_compile .../*.py` — pass.
- `python3 -m unittest ...tests.test_synthetic` — 19 CPU-only tests pass.
- Import smoke test, static forbidden-execution audit, original hash/provenance check, `git diff --check`, and final worktree audit are required before commit and recorded in the JSON packet.

Focused remediation coverage includes isolated-zero aggregate reachability, never-nonzero rejection, streaming SGD metric equivalence and pre/post non-mutation hashes, zero candidate optimizer steps and exact `1976` disposable-step accounting, verified checkpoint selection, partial-epoch rejection, checkpoint-binding rejection, and exact experiment-provenance binding. No test calls `optimizer.step`.

No test loaded a real sample, ran a real model forward/backward, invoked CUDA training, or called `optimizer.step`.

## 12. Commands explicitly not run

Not run: replay, failure reproduction, qualification, tau selection, envelope construction from real telemetry, real dataset access, real-model forward/backward, optimizer step, CUDA training, scientific continuation, experiment creation, validation prediction access, inference, evaluator, CARLA, OAI, locked test, q/quantization, AE/hybrid-q, deployment, or 288 campaign.

## 13. Assumptions and unresolved review inputs

The exact update-447 sample IDs were not derived during this implementation because doing so would process the real epoch10 ordering boundary. A reviewer must preregister the 16 IDs in `/tmp/splitfusion_recovery_review/EXPECTED_UPDATE_447_SAMPLE_IDS.json` before any replay. Independent review and later user authorization must be written outside this committed package and bind the resolved commit/package hash. No scientific conclusion about the suspected failure or any tau is asserted by this implementation.

## 14. Commands reserved for later (do not run during implementation)

After independent review prepares the required external JSON files, the exact qualification command is:

```bash
python3 -m pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_v1_numerical_recovery_v1.qualify_recovery --output /tmp/splitfusion_recovery_qualification_v1 --expected-update447 /tmp/splitfusion_recovery_review/EXPECTED_UPDATE_447_SAMPLE_IDS.json --independent-review /tmp/splitfusion_recovery_review/INDEPENDENT_SOURCE_REVIEW.json --execute-qualification DISPOSABLE_NO_SCIENTIFIC_STATE
```

Only after qualification, a second independent review writes `/tmp/splitfusion_recovery_qualification_v1/INDEPENDENT_QUALIFICATION_REVIEW.json` binding `RECOVERY_QUALIFICATION.json` and `QUALIFIED_RECOVERY_CONFIG.json`, and explicit user authorization binds those same hashes:

```bash
python3 -m pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_v1_numerical_recovery_v1.continue_scientific --qualification-dir /tmp/splitfusion_recovery_qualification_v1 --authorization /tmp/splitfusion_recovery_review/USER_AUTHORIZATION.json --output experiments/route_b_v3_1_splitfusion_fcos_r50_fpn_p2_p7_v1_recovery_v1/CREATE_ONLY_AFTER_AUTHORIZATION --execute-scientific-continuation AUTHORIZED_EPOCH9_TO_EPOCH26_RECOVERY
```

If that authorized process is interrupted after a complete recovered epoch checkpoint, the exact same command adds `--resume-existing-epoch-boundary`. It refuses an absent output, a run with no complete recovered checkpoint, any partial/noncontiguous next-epoch artifact, or any source/qualification/provenance binding drift.

Recovered inference is one command per epoch 16, 22, and 26, changing only `--epoch`:

```bash
python3 -m pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_v1_numerical_recovery_v1.infer_recovered --experiment experiments/route_b_v3_1_splitfusion_fcos_r50_fpn_p2_p7_v1_recovery_v1/CREATE_ONLY_AFTER_AUTHORIZATION --qualification-dir /tmp/splitfusion_recovery_qualification_v1 --authorization /tmp/splitfusion_recovery_review/USER_AUTHORIZATION.json --epoch 16 --execute-validation-inference TRAINING_COMPLETE_AND_AUTHORIZED
```

Final frozen five-checkpoint evaluation:

```bash
python3 -m pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_v1_numerical_recovery_v1.evaluate_recovered --recovered-experiment experiments/route_b_v3_1_splitfusion_fcos_r50_fpn_p2_p7_v1_recovery_v1/CREATE_ONLY_AFTER_AUTHORIZATION --qualification-dir /tmp/splitfusion_recovery_qualification_v1 --authorization /tmp/splitfusion_recovery_review/USER_AUTHORIZATION.json --execute-recovered-evaluation ALL_FIVE_INFERENCE_PASSES_COMPLETE
```

## 15. Implementation execution counters

All are zero: qualification runs, real replays, real model forwards, real model backwards, optimizer steps, scientific experiments created, validation predictions opened, inference runs, and evaluation runs. State remains `UNQUALIFIED_IMPLEMENTATION_ONLY`; no `QUALIFIED_TO_TRAIN` exists.
