# Hybrid-q v1 — spatial transport sparsification at the frozen C2 split

Phase 2 status: **source locked, training not executed.** Nothing in this package loads the
frozen checkpoint, reads real data, uses CUDA, builds a cache, trains, runs inference,
evaluates validation/test data, or launches CARLA.

The machine-readable contract is [locked_config.json](locked_config.json); `contract.load_locked_config()`
reads it and fails closed on any drift from the module constants.

## Architecture placement

Hybrid-q is **transport-only**. It sits on the wire between the UE-side split point and
the edge-side perception tail:

```
RGB(3) + radar raster(4)  ->  [FROZEN trunk]  ->  fused C2 Z [256,112,192]
                                                      |
                                          hybrid-q  ( ranker -> q -> mask -> codec )
                                                      |
                                        wire payload  ->  decode to dense [256,112,192]
                                                      |
                                       [FROZEN FCOS + geometry + segmentation heads]
                                                      |
                                        frozen p025 person threshold, frozen evaluator
```

Bound to `../splitfusion_fcos_r50_fpn_p2_p7_person_p025_calibration_v1/PERCEPTION_FORWARD_LOCK_P025_V1.json`
(SHA-256 `86d6f13a…3fe1`) and checkpoint SHA-256 `da14d21e…0a297f`. It does **not** modify the
seven-channel input, frozen perception weights, FCOS scoring, the geometry head, the
segmentation head, the evaluator, the p025 output threshold or the AVO definition. The
frozen model runs in evaluation mode.

## Frozen C2 boundary

Every production entry point — `codec.encode`, `codec.decode`, `select_cells`,
`apply_selection`, `select_and_apply`, `SpatialRanker.forward`, `SpatialRanker.score_cells`
and `training.straight_through_mask` — enforces:

- one frame `[256, 112, 192]`, or `[B, 256, 112, 192]` for batched ranker input;
- FP32;
- finite values;
- scores `[112, 192]`.

`decode` additionally fails closed unless the header specifies exactly 256 channels,
height 112, width 192 and FP32. The q=0 bypass validates the tensor **before** returning it.

Private generic helpers (`_select_cells`, `_apply_selection`, `_encode`, `_decode`,
`SpatialRanker._score_any`) exist only so small-tensor wire-layout tests stay readable.
They are not part of the production API.

## The ranker

`ranker.SpatialRanker` is exactly:

| layer | shape |
| --- | --- |
| 1x1 Conv | 256 -> 8 |
| ReLU | |
| depthwise 3x3 Conv (groups=8, padding=1) | 8 -> 8 |
| ReLU | |
| 1x1 Conv | 8 -> 1 |

- **2,145** trainable parameters, **45,760,512 MACs** (~45.76 M) at 112x192.
- No BatchNorm, no attention, no second backbone, no object-level ROI model.
- Input detached inside `forward`: gradients reach ranker parameters only.
- Runtime input is fused C2 and nothing else — no RGB, radar, GT, detections,
  segmentation, geometry or target-region side channel.

**Deterministic construction.** `build_ranker()` initializes at the registered seed
**20260829** inside `torch.random.fork_rng`, so two builds are identical and construction
does not advance the caller's global RNG sequence.

## Exact q semantics

q is the **spatial drop fraction**:

```python
N = H * W
drop_count = floor(q * N + 0.5)
keep_count = N - drop_count
```

Registered values at `N = 112 * 192 = 21,504`:

| q | keep | drop | role |
| --- | --- | --- | --- |
| 0.00 | 21,504 | 0 | parity monitoring |
| 0.30 | 15,053 | 6,451 | q-aware training |
| 0.50 | 10,752 | 10,752 | q-aware training |
| 0.70 | 6,451 | 15,053 | q-aware training |
| 0.90 | 2,150 | 19,354 | evaluation stress |
| 0.98 | 430 | 21,074 | evaluation stress |

The highest-scoring cells are kept. **Ties prefer the lower row-major spatial index**, via a
stable descending sort of the row-major flattened scores. All 256 channels of a spatial
cell are retained or removed together.

### q = 0 identity

`select_and_apply(c2, ranker, 0.00)` validates the tensor, then returns the input object
itself with `None` for the selection — the ranker is never invoked and no mask is built, so
dense identity is exact by construction. **Callers must not mutate the returned tensor in
place.** On the wire, q=0 emits the dense form (no bitmask) and decodes bit-exactly.

### Selection integrity

For q>0, encoding cross-checks the supplied selection before framing: `selection.q` equals
the requested q; `cells == 21,504`; keep and drop counts match the registered formula; mask
shape is `[112,192]` and boolean; mask popcount equals the keep count; indices are strictly
ascending and unique; and the mask and the index set describe **exactly the same cells**.

## Wire layout

A payload is `header || bitmask || values`.

**Header** (44 bytes, little-endian, `struct` format `<4sHHHHIIIIIIQ`): magic `HQ1\0`,
format version, dtype code, flags, reserved, channels, height, width, q in
ten-thousandths, keep count, mask byte count, value byte count.

**Bitmask** (q>0 only): one fixed-order bit per spatial cell, `ceil(N/8)` = 2,688 bytes.
Bit value **1 means retained**. Cell `i` lives in byte `i // 8` at bit `7 - (i % 8)` —
bytes ascend with cell index and the **most significant bit within a byte is the lowest
cell index**. Padding bits past cell N-1 must be zero. The mask is **uncompressed**; that is
locked for this phase.

**Values**: retained cells in **ascending row-major cell order**, each holding all 256 FP32
channels contiguously (cell-major, channel-contiguous — not `[C,H,W]` channel-major),
little-endian.

Decoding rebuilds the dense shape with **exact zeros** in dropped cells. The edge receives a
dense zero-scattered tensor, which is what preserves the frozen forward exactly.

## Payload accounting

Two distinct quantities, never conflated:

- **framed q=0 payload = 22,020,140 bytes** — the actual serialized wire payload at q=0;
- **raw FP32 reference = 22,020,096 bytes** — the unframed tensor size.

The framed q=0 payload is **not** wire-byte-identical to the former unframed raw
representation; framing adds the 44-byte versioned header. Tensor identity at q=0 is about
the decoded values, which are exactly equal to the input.

**Primary hybrid-q compression ratio = actual serialized q payload bytes / actual serialized
q=0 payload bytes.** Measured (`SparsePayload.framed_ratio`):

| q | keep | actual bytes | framed ratio |
| --- | --- | --- | --- |
| 0.00 | 21,504 | 22,020,140 | 1.000000 |
| 0.30 | 15,053 | 15,417,004 | 0.700132 |
| 0.50 | 10,752 | 11,012,780 | 0.500123 |
| 0.70 | 6,451 | 6,608,556 | 0.300114 |
| 0.90 | 2,150 | 2,204,332 | 0.100105 |
| 0.98 | 430 | 443,052 | 0.020120 |

The raw FP32 size is reported separately via `codec.raw_fp32_reference_bytes`. No bitmask
compression and no zstd implementation belongs in this phase.

## Teacher-map contract

Supervision uses exactly the registered frozen-model loss groups:

| group | definition |
| --- | --- |
| `D` | FCOS classification, box regression and centerness |
| `G` | registered geometry loss |
| `S` | registered semantic loss |
| `A` | registered dense-depth and radar-consistency loss |

```
I_t(h, w) = sum_c | C2(c, h, w) * grad_t(c, h, w) |
```

Each valid group is **L1-normalized independently**, then valid groups are combined with
**equal weight**. Unregistered group keys are rejected. Absent, zero-gradient and non-finite
groups are recorded in `excluded_groups` with a reason and excluded — never silently treated
as valid supervision. A frame with no valid group returns `importance=None`.

No class-specific rewritten losses, target-region side channels, RGB, radar, predictions or
GT enter the ranker runtime.

`gradient_mass` is the raw pre-normalization L1 mass per group and is **diagnostic only**.
`task_losses` holds dense task-loss values **separately**, for future train-only
reference-scale construction. `TeacherCacheRecord` carries teacher maps, validity flags,
identifiers, `gradient_mass`, `task_losses` and the checkpoint hash — **full C2 tensors are
deliberately not cached**.

## Locked training semantics (not executed)

**Distillation** — listwise soft cross-entropy against the L1-normalized teacher
distribution, temperature **1.0**, **4 epochs**.

**q-aware phase** — **8 epochs**, exactly **one q per optimizer update**, deterministic
repeated cycle **0.30, 0.50, 0.70** (`training.q_for_update` / `q_aware_schedule`). q=0 is
parity monitoring only and produces **no** ranker-training gradient; q=0.90 and q=0.98 are
later evaluation stress points only.

Objective:

```
mean(valid masked task loss / frozen train-reference median for that task) + 0.1 * distillation loss
```

`training.ReferenceMedians` requires `source == "fit_train"`; validation- or test-derived
scale is rejected. Those medians do not exist yet and must be produced from fit-training
data in a later phase.

**Optimization** — AdamW, lr `1e-3`, weight decay `1e-4`, constant LR, global ranker
gradient-norm clip `5.0`, checkpoints after epochs **4, 8, 12**, **no augmentation** so
cached teacher maps stay aligned with canonical frames. The optimizer owns ranker
parameters only, and the frozen stack must be non-trainable and in eval mode.

**Straight-through mask** — forward is the exact hard mask; the boundary is the midpoint
between the lowest retained and highest dropped score; temperature is fixed at **1.0**. The
public entry point takes `(scores, q)` only — no production threshold or temperature
override is exposed. q=0 is an explicit identity bypass with no surrogate gradient.

## Runtime guards

All fail-closed, in `guards.py`: non-finite ranker scores, invalid or unregistered q,
incorrect keep cardinality, malformed bitmask or payload, duplicate or unordered retained
indices, incompatible shape/dtype/version, non-finite decoded values, selection/mask
disagreement, frozen state drift, and accidental optimizer ownership of frozen parameters.

**Gradient qualification** (`training.GradientQualification`) tracks **every named trainable
ranker tensor** across the complete window and requires:

- a finite loss on every update;
- every observed gradient finite;
- every ranker tensor to receive a gradient and be nonzero at least once in the window;
- isolated zero-gradient batches are **logged**, not failures;
- parameters still missing or disconnected at window end **fail**.

After a step, `training.require_post_step_health` requires finite ranker parameters and
finite tensor-valued optimizer state, and `guards.require_module_state_unchanged` verifies
every frozen perception **parameter and buffer** is exactly unchanged.

## Locked implementation decisions

Deliberately unchanged and not up for re-litigation in this phase: midpoint straight-through
boundary; listwise distillation; uncompressed fixed-order MSB-first mask; dense
zero-scattered edge input; per-frame selection (**not** vectorized — its latency will be
measured in Phase 3 before deciding whether optimization is necessary); q=0 returning the
validated input tensor object; FP32 transport only.

Because masking is applied by zeroing, hybrid-q reduces **transport**, not edge compute.
Any latency claim must say so explicitly.

## Phase boundaries

Not done and not authorized by this commit: GPU qualification, teacher-cache generation,
training, inference, validation or test evaluation, OAI/CARLA transport measurement,
quantization, zstd and the AE families. The test set remains untouched and reserved for
independent publication confirmation.

## Commands intended for later qualification (not executed here)

Only the first is in scope for Phase 2.

```bash
# Phase 2 (run): CPU synthetic tests
python3 -m unittest pole_lraspp_multimodal_fusion.object_head_pilot_v1.\
splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.tests.test_synthetic -v

# Phase 3+ (NOT run here):
#   GPU forward-parity qualification at q=0 against the frozen p025 checkpoint
#   selection latency measurement, before any vectorization decision
#   teacher-cache generation (maps, validity, gradient_mass, task_losses; no C2)
#   fit-train reference-median construction
#   4 distillation epochs, then 8 q-aware epochs over the 0.30/0.50/0.70 cycle
#   frozen-validation sweep over q in {0.00, 0.30, 0.50, 0.70, 0.90, 0.98}
#   measured payload / UE / network / edge / end-to-end latency over OAI
```
