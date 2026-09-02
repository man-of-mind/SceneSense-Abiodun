# Hybrid-q implementation review packet — Phase 1 findings, Phase 2 corrections

Phase 1 commit: `1b3bec4`. Phase 2 status: **source review complete, corrections applied,
training not executed.** No real data, checkpoint load, CUDA, cache build, training,
inference, validation/test evaluation or CARLA in either phase.

Bound contract: `../splitfusion_fcos_r50_fpn_p2_p7_person_p025_calibration_v1/PERCEPTION_FORWARD_LOCK_P025_V1.json`
(SHA-256 `86d6f13ae9168b33b697df5b785c5f7c320afc52cfdcded5b632d94a6d943fe1`).
Frozen checkpoint SHA-256: `da14d21edbd374c1c3abce02ca4674b9f4097becfba9759aba945cea160a297f` (not loaded).
Machine-readable contract: [locked_config.json](locked_config.json).

## 1. Changed files

Phase 2 modified every Phase-1 module and added `locked_config.json`. No file outside this
package was touched; the dirty `OAI/openairinterface5g` submodule was left unchanged.

| file | lines | sha256 (first 16) | phase-2 change |
| --- | --- | --- | --- |
| `__init__.py` | 6 | 9319a2b3ce9d492e | unchanged |
| `contract.py` | 268 | 85a648416f4cf2ec | locked constants, payload accounting, `load_locked_config` |
| `guards.py` | 336 | 77d8d8bfd168e74a | frozen-boundary, selection-integrity, module-state and optimizer-state guards |
| `ranker.py` | 112 | 15aeb4b8bd6d5e47 | frozen-shape forward, deterministic seeded construction |
| `selection.py` | 118 | ccc2b12919b078ea | public frozen API split from private generic path |
| `codec.py` | 263 | 7b3833398a84fea3 | frozen header enforcement, selection cross-check, framed ratio |
| `training.py` | 492 | a10775a3a9f3e051 | locked teacher/distillation/q-aware semantics, per-tensor qualification |
| `locked_config.json` | 184 | 24e56c04643e9e4c | **new** |
| `tests/__init__.py` | 1 | 1c1c24c4edd4acf7 | unchanged |
| `tests/test_synthetic.py` | 1152 | c6c032e67020f437 | 37 -> 75 tests |
| `README.md` | 273 | f93a3a9c657741d0 | rewritten for the locked contract |

## 2. Exact contracts

**Frozen surface untouched.** No code imports, wraps or mutates the seven-channel input,
perception weights, FCOS scoring, geometry head, segmentation head, evaluator, the p025
output threshold or the AVO definition. The frozen stack must be non-trainable and in
evaluation mode before an optimizer is built.

**Frozen C2 boundary.** Every production encode, decode, selection, masking and ranker entry
point requires `[256,112,192]` (or `[B,256,112,192]` for batched ranker input), FP32 and
finite values; scores must be `[112,192]`. `decode` fails closed unless the header specifies
exactly 256/112/192/FP32. The q=0 bypass validates before returning. Private generic helpers
(`_select_cells`, `_apply_selection`, `_encode`, `_decode`, `_score_any`) carry the
small-tensor layout tests and are not production API.

**Selection integrity.** For q>0, `encode` verifies q agreement, `cells == 21,504`, keep and
drop counts against the registered formula, `[112,192]` boolean mask, mask popcount, strict
ascending uniqueness of indices, and that mask and index set describe the same cells.

**q semantics.** `drop_count = floor(q*N + 0.5)`, `keep_count = N - drop_count`; keep counts
21,504 / 15,053 / 10,752 / 6,451 / 2,150 / 430. Ties resolved to the lower row-major index.

**Wire format.** Unchanged 44-byte versioned header, uncompressed fixed-order MSB-first
bitmask, retained cells ascending row-major with 256 contiguous FP32 channels, dropped cells
decoding to exact zeros.

**Payload accounting.** Tensor identity at q=0 is exact equality of decoded values. Framed
q=0 payload = 22,020,140 bytes; raw FP32 reference = 22,020,096 bytes, reported separately;
the framed payload is **not** wire-byte-identical to the unframed raw representation.
Primary ratio = framed q payload / framed q=0 payload:

| q | keep | actual bytes | framed ratio |
| --- | --- | --- | --- |
| 0.00 | 21,504 | 22,020,140 | 1.000000 |
| 0.30 | 15,053 | 15,417,004 | 0.700132 |
| 0.50 | 10,752 | 11,012,780 | 0.500123 |
| 0.70 | 6,451 | 6,608,556 | 0.300114 |
| 0.90 | 2,150 | 2,204,332 | 0.100105 |
| 0.98 | 430 | 443,052 | 0.020120 |

**Locked training semantics.** Teacher groups exactly `D`/`G`/`S`/`A`; per-group L1
normalization; equal-weight combination; `gradient_mass` diagnostic; `task_losses` kept
separately for train-only reference-scale construction. Distillation: listwise soft
cross-entropy, temperature 1.0, 4 epochs. q-aware: 8 epochs, one q per update, deterministic
0.30/0.50/0.70 cycle, q=0 parity-only with no ranker gradient, 0.90/0.98 evaluation stress
only, objective `mean(masked task loss / fit-train reference median) + 0.1 * distillation`.
AdamW, lr 1e-3, wd 1e-4, constant LR, clip 5.0, checkpoints at epochs 4/8/12, no
augmentation. Straight-through: exact hard forward, midpoint boundary, temperature 1.0,
explicit q=0 identity bypass.

**Gradient qualification.** Per-named-tensor tracking across the complete window: finite
loss every update, every observed gradient finite, every tensor present and nonzero at least
once, isolated zero-gradient batches logged, missing/disconnected tensors fail at window
end. Post-step: finite ranker parameters, finite tensor-valued optimizer state, and every
frozen parameter **and buffer** exactly unchanged.

**Deterministic initialization.** Seed 20260829 inside `torch.random.fork_rng`; the caller's
global RNG state is restored and not advanced.

## 3. Tests

`python3 -m unittest ...hybrid_q_v1.tests.test_synthetic` — **75 tests, all pass, CPU-only
synthetic tensors** (was 37 in Phase 1). Also run: `py_compile` on every module and
`git diff --check`.

| correction | tests |
| --- | --- |
| 1 frozen C2 boundary | `FrozenBoundaryCheck` (6): encode/selection/ranker/straight-through rejection of wrong channels, height, width, rank, dtype; non-finite rejection at every entry point; q=0 bypass validation; frozen-header decode rejection incl. dtype and unregistered q |
| 2 selection integrity | `SelectionIntegrityCheck` (9): q mismatch, cell count, keep and drop count, mask shape, mask dtype, popcount, unordered and duplicate indices, mask/index-set disagreement at equal cardinality, and the same cross-check on `apply_selection` |
| 3 payload accounting | `PayloadAccountingCheck` (3): q=0 tensor identity and 22,020,140-byte framing, framed-vs-raw distinction, framed-denominator ratios for all six q |
| 4 locked configuration | `LockedConfigCheck` (3): config/constant agreement, on-disk perception-lock SHA-256, locked training semantics |
| 5 runtime qualification | `GradientQualificationCheck` (8) and `FrozenStateCheck` (2): per-tensor tracking, isolated zero batch logged, never-nonzero fails, disconnected fails, incomplete window fails, non-finite loss/gradient, tensor-set drift, post-step parameter and optimizer-state finiteness, clip norm, frozen parameter+buffer immutability |
| 6 deterministic init | `DeterministicInitCheck` (3): seed reproducibility, caller RNG untouched, 2,144 parameters |
| retained Phase-1 coverage | `QSemanticsCheck`, `TieBreakCheck`, `MaskingCheck`, `CodecRoundTripCheck`, `MalformedPayloadCheck`, `RankerShapeCheck`, `OptimizerOwnershipCheck`, `TeacherMapCheck`, `QAwareContractCheck`, `StraightThroughCheck`, `ContractBindingCheck` |

Two test expectations were corrected during this phase, both because the tightened
implementation is now stricter than the Phase-1 test assumed: a header-corruption case on a
small generic payload now trips the frozen-dimension guard first (moved to the frozen path),
and the q=0 surrogate now produces a tensor with no autograd graph at all, so the test
asserts the absence of a gradient path instead of calling `backward()`.

## 4. Disposition of the eleven Phase-1 unresolved items

| # | Phase-1 item | disposition |
| --- | --- | --- |
| 1 | straight-through threshold unspecified | **Resolved (locked).** Midpoint between lowest retained and highest dropped score, temperature 1.0, no override exposed on the public entry point. |
| 2 | teacher combination weighting unregistered | **Resolved (locked).** Groups exactly D/G/S/A, per-group L1 normalization, equal-weight combination; unregistered group keys rejected. |
| 3 | distillation loss form a choice | **Resolved (locked).** Listwise soft cross-entropy against the L1-normalized teacher distribution, temperature 1.0, 4 epochs. |
| 4 | `loss_scales` recorded but unused | **Resolved (renamed).** Now `gradient_mass`, diagnostic only; dense `task_losses` are kept separately for future train-only reference-scale construction. |
| 5 | header accounting at q=0 ambiguous | **Resolved (locked).** Primary ratio uses the framed q=0 payload (22,020,140 B); the raw FP32 size (22,020,096 B) is reported separately and the two are never described as byte-identical. |
| 6 | bitmask uncompressed | **Deferred.** Fixed-order uncompressed mask is locked here; any bitmask-compression diagnostic belongs with the dense-zero zstd diagnostic in the quantization/AE phase. |
| 7 | FP32 only; composition with quantization/AE undecided | **Deferred.** Quantization, zstd and AE128/64/32 are separate permitted next changes; composition order is a quantization/AE-phase decision. |
| 8 | masking by zeroing, not a sparse forward | **Resolved (locked).** Dense zero-scattered edge input is retained because it preserves the frozen forward exactly; README and this packet state that hybrid-q reduces transport, not edge compute. |
| 9 | per-frame selection | **Resolved (locked for now).** Selection stays per-frame and unvectorized; its latency will be measured in Phase 3 before any optimization decision. |
| 10 | ranker init default, no seed policy | **Resolved.** Seed 20260829 via `torch.random.fork_rng`; reproducible and RNG-isolated. |
| 11 | q=0 returns the input object | **Resolved (locked).** Documented aliasing contract; the tensor is now validated before return and callers must not mutate it in place. |

## 5. Open items for later phases

These are consequences of the locked decisions, not unresolved design questions:

1. **Fit-train reference medians do not exist yet.** `ReferenceMedians` enforces
   `source == "fit_train"` and rejects validation- or test-derived scale, but the values must
   still be produced in a later phase before the q-aware objective can run.
2. **One q per optimizer update means a batch shares one q.** Selection is per-frame, so a
   batched q-aware update applies the same q to every frame in the batch. This is consistent
   with the locked schedule; it is recorded so it is not mistaken for per-frame q sampling.
3. **Selection latency is unmeasured.** Phase 3 measures it before any vectorization.
4. **Quantization / zstd / AE composition** remains out of scope, as above.

## 6. Explicitly not done

No GPU qualification, teacher-cache generation, training, inference or evaluation was
started. The test set remains untouched and reserved for independent publication
confirmation.
