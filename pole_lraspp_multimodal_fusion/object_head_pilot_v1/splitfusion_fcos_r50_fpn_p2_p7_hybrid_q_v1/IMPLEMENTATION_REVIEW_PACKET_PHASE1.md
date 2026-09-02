# Hybrid-q Phase 1 — implementation review packet

Status: **implementation complete, unreviewed.** Phase 1 was implementation-only. No real
data, no checkpoint load, no CUDA, no cache build, no training, no inference, no
validation/test evaluation, no CARLA. Independent review has **not** started.

Bound contract: `../splitfusion_fcos_r50_fpn_p2_p7_person_p025_calibration_v1/PERCEPTION_FORWARD_LOCK_P025_V1.json`
Frozen checkpoint SHA-256: `da14d21edbd374c1c3abce02ca4674b9f4097becfba9759aba945cea160a297f` (not loaded).

## 1. Changed files

All additions; no existing file was modified. The dirty `OAI/openairinterface5g` submodule
was left untouched.

| file | lines | sha256 (first 16) |
| --- | --- | --- |
| `__init__.py` | 6 | 9319a2b3ce9d492e |
| `contract.py` | 108 | 7cbefbffdffc87d8 |
| `guards.py` | 141 | b57790f48d5245b1 |
| `ranker.py` | 95 | c0524f5b8c903d0e |
| `selection.py` | 93 | 6e7c32d766e1687b |
| `codec.py` | 217 | 5aed2fa3cf21795f |
| `training.py` | 298 | ef26996246f8c2e2 |
| `tests/__init__.py` | 1 | 1c1c24c4edd4acf7 |
| `tests/test_synthetic.py` | 577 | 103e4cc376585fd9 |
| `README.md` | 189 | ea3c3503df7cf91e |

(This packet is added alongside them.)

## 2. Exact contracts

**Frozen surface untouched.** No code here imports, wraps or mutates the seven-channel
input, perception weights, FCOS scoring, geometry head, segmentation head, evaluator, the
p025 output threshold or the AVO definition. `contract.load_perception_lock()` reads the
lock JSON only, and raises if split shape, FP32 payload size or checkpoint hash disagree.

**Split tensor.** `[256, 112, 192]`, 21,504 spatial cells, row-major spatial indexing,
FP32 raw 22,020,096 bytes. All 256 channels of a cell are retained or dropped together.

**Ranker.** 1x1 Conv 256->8, ReLU, depthwise 3x3 Conv 8->8 pad 1, ReLU, 1x1 Conv 8->1.
Verified 2,145 parameters and 45,760,512 MACs (45.76 M) at 112x192. No BatchNorm, no
attention, no second backbone, no ROI model. Input is detached inside `forward`; the ranker
has no runtime access to RGB, radar, GT, detections, segmentation or geometry.

**q semantics.** `drop_count = floor(q*N + 0.5)`, `keep_count = N - drop_count`. Registered
keep counts at N=21,504 verified exactly: 21,504 / 15,053 / 10,752 / 6,451 / 2,150 / 430 for
q = 0.00 / 0.30 / 0.50 / 0.70 / 0.90 / 0.98. Highest scores kept; ties resolved to the lower
row-major index via a stable descending sort. q=0 bypasses the ranker and the mask entirely
and returns the input tensor object.

**Wire format.** `header(44) || bitmask || values`. Header `<4sHHHHIIIIIIQ` = magic `HQ1\0`,
version 1, dtype code, flags, reserved, C, H, W, q in ten-thousandths, keep count, mask
bytes, value bytes. Bitmask present only for q>0, `ceil(N/8)` bytes (2,688 at N=21,504),
bit 1 = retained, cell `i` at byte `i//8` bit `7-(i%8)` (MSB-first within a byte), padding
bits zero. Values: retained cells in ascending row-major order, all 256 FP32 channels
contiguous per cell, little-endian. Decode restores dense shape with exact zeros in dropped
cells.

**Measured serialized lengths** (actual `len(payload.data)`, dense FP32 raw = 22,020,096):

| q | keep | bytes | vs raw |
| --- | --- | --- | --- |
| 0.00 | 21,504 | 22,020,140 | 100.0002% |
| 0.30 | 15,053 | 15,417,004 | 70.0133% |
| 0.50 | 10,752 | 11,012,780 | 50.0124% |
| 0.70 | 6,451 | 6,608,556 | 30.0115% |
| 0.90 | 2,150 | 2,204,332 | 10.0105% |
| 0.98 | 430 | 443,052 | 2.0120% |

**Teacher maps.** `I_t(h,w) = sum_c |C2 * grad_t|`; every valid task normalized
independently before combination; absent / zero-gradient / non-finite tasks recorded in
`excluded_tasks` and excluded; a frame with zero valid tasks is not supervisable.
`TeacherCacheRecord` holds teacher maps, validity flags, identifiers and loss-scale
metadata only — **no C2 tensors**.

**Straight-through.** Forward equals the hard exact-cardinality mask bit-exactly; sigmoid
surrogate supplies the gradient; temperature is required from configuration with no default.

**Gradient policy.** Finiteness required on every update. Nonzero-gradient evidence required
over a qualification window. Isolated zero-gradient batches are logged, not failures. There
is no per-batch exact-zero-gradient abort anywhere in the package.

## 3. Tests

`python3 -m unittest ...hybrid_q_v1.tests.test_synthetic` — **37 tests, all pass, CPU-only
synthetic tensors.** Coverage against the 12 required items:

| # | requirement | test |
| --- | --- | --- |
| 1 | exact keep counts, all six q | `QSemanticsCheck.test_registered_keep_counts_are_exact` |
| 2 | deterministic tie-breaking | `TieBreakCheck` (3 tests) |
| 3 | q=0 exact identity and ranker bypass | `DenseIdentityCheck` (2 tests, ranker stub raises if called) |
| 4 | all-256-channel retention, exact zeros | `MaskingCheck.test_retained_cells_keep_all_channels_and_dropped_cells_are_zero` |
| 5 | sparse pack/unpack round trip | `CodecRoundTripCheck.test_sparse_round_trip_matches_masked_tensor` + byte/bit-order and value-order tests |
| 6 | malformed payloads fail closed | `MalformedPayloadCheck` (7 tests: truncation, magic, version, dtype, flags, reserved, dims, q, keep count, mask length, value length, popcount, padding bits, duplicate/unordered indices, dtype/NaN on encode) |
| 7 | actual serialized byte count reported | `CodecRoundTripCheck.test_serialized_byte_counts_are_measured_and_reported` (asserts and prints the table above) |
| 8 | ranker parameter and MAC counts | `RankerShapeCheck` (recomputed independently from realized weight shapes) |
| 9 | optimizer owns only ranker params | `OptimizerOwnershipCheck` (3 tests) |
| 10 | teacher normalization and absent tasks | `TeacherMapCheck` (5 tests) |
| 11 | straight-through forward == hard mask | `StraightThroughCheck` (3 tests) |
| 12 | frozen params unchanged after a step | `FrozenWeightCheck.test_frozen_parameters_are_unchanged_after_an_optimizer_step` |

Also run: `python3 -m py_compile` on every module, and `git diff --check`.

One real defect was found and fixed by test 11 during implementation: the surrogate
originally computed `hard + soft - soft.detach()`, which in float32 does not return exactly
`hard` (e.g. `1 + 0.7 - 0.7 != 1.0`). It is now grouped as `hard + (soft - soft.detach())`.

## 4. Unresolved issues requiring Phase-2 review

1. **Straight-through threshold is an implementation choice.** The surrogate centres the
   sigmoid at the midpoint between the lowest kept and highest dropped score. The contract
   did not specify a threshold. It interacts directly with the (deliberately unset)
   temperature and should be reviewed before any training run.
2. **Teacher combination weighting is unregistered.** Valid task maps are currently combined
   as a uniform mean of independently normalized maps. `task_weights` is exposed but unused,
   and the task set (detection / segmentation / geometry) is not registered. The default
   normalization is `l1`; `max` is available. Needs a registered decision.
3. **Distillation loss form is a choice, not a contract.** Implemented as listwise soft
   cross-entropy against the normalized teacher distribution. The brief asked only for an
   interface. If a different objective (e.g. ranking or MSE on normalized maps) is intended,
   this is the point to change it.
4. **`loss_scales` are recorded but unused.** Raw per-task gradient mass is captured for
   later loss-scale metadata; no policy consumes it yet.
5. **Header accounting at q=0.** The q=0 dense payload is 22,020,140 bytes — 44 bytes above
   raw. Whether reported compression ratios should be taken against raw FP32 or against the
   q=0 payload is a reporting decision that must be fixed before any payload claim.
6. **The bitmask is fixed-order and uncompressed.** 2,688 bytes is 0.61% of the q=0.98
   payload. The contract requires a fixed-order bitmask, so no RLE/entropy coding was added;
   if a bitmask diagnostic is wanted it belongs alongside the dense-zero zstd diagnostic.
7. **FP32 only.** Quantization, zstd and the AE families are separate permitted next changes
   and are deliberately not composed with hybrid-q here. Composition order (mask then
   quantize, or quantize then mask) is undecided.
8. **Masking is applied by zeroing, not by a sparse forward.** The frozen tail still receives
   a dense `[256,112,192]` tensor, which is what preserves the frozen forward exactly. This
   means hybrid-q reduces transport, not edge compute — worth stating explicitly in any
   latency claim.
9. **Selection is per-frame.** `select_cells` takes `[H,W]`; batched training must iterate
   frames. Fine for correctness, possibly a throughput item later.
10. **Ranker initialization is PyTorch default.** No registered init scheme or seed policy
    yet; training reproducibility will need one.
11. **`select_and_apply` returns the input object itself at q=0** (identity by construction).
    Callers must not mutate the returned tensor in place expecting a copy.

## 5. Explicitly not done

No independent review, GPU qualification, teacher-cache generation, training, inference or
evaluation was started. The test set remains untouched and reserved for independent
publication confirmation.
