# Hybrid-q v1 — spatial transport sparsification at the frozen C2 split

Phase 1: implementation, contract and CPU-only synthetic tests. Nothing in this package
loads the frozen checkpoint, reads real data, uses CUDA, builds a cache, trains, runs
inference, evaluates, or launches CARLA.

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

It is bound to `../splitfusion_fcos_r50_fpn_p2_p7_person_p025_calibration_v1/PERCEPTION_FORWARD_LOCK_P025_V1.json`
and checkpoint SHA-256 `da14d21e…0a297f`. It does **not** modify the seven-channel input,
frozen perception weights, FCOS scoring, the geometry head, the segmentation head, the
evaluator, the p025 output threshold or the AVO definition. `contract.load_perception_lock()`
reads that JSON and fails closed if the split shape, payload size or checkpoint hash drift.

## The ranker

`ranker.SpatialRanker` is exactly:

| layer | shape |
| --- | --- |
| 1x1 Conv | 256 -> 8 |
| ReLU | |
| depthwise 3x3 Conv (groups=8, padding=1) | 8 -> 8 |
| ReLU | |
| 1x1 Conv | 8 -> 1 |

- **2,145** trainable parameters.
- **45,760,512 MACs** (~45.76 M) for a 112x192 C2 tensor.
- No BatchNorm, no attention, no second backbone, no object-level ROI model.
- The input is detached inside `forward`, so gradients reach ranker parameters only.
  The ranker sees fused C2 and nothing else — no RGB, radar, GT, detections,
  segmentation or geometry at runtime.

## Exact q semantics

q is the **spatial drop fraction**:

```python
N = H * W
drop_count = floor(q * N + 0.5)
keep_count = N - drop_count
```

Registered values and their keep counts at `N = 112 * 192 = 21,504`:

| q | keep | drop |
| --- | --- | --- |
| 0.00 | 21,504 | 0 |
| 0.30 | 15,053 | 6,451 |
| 0.50 | 10,752 | 10,752 |
| 0.70 | 6,451 | 15,053 |
| 0.90 | 2,150 | 19,354 |
| 0.98 | 430 | 21,074 |

The highest-scoring cells are kept. **Ties prefer the lower row-major spatial index**,
implemented as a stable descending sort of the row-major flattened scores, so selection is
deterministic and repeatable for any score tensor.

All 256 channels of a spatial cell are retained or removed together; there is no
per-channel decision anywhere in the package.

### q = 0 identity

`selection.select_and_apply(c2, ranker, 0.00)` returns the input tensor object itself and
`None` for the selection. The ranker is never invoked and no mask is built, so dense
identity is exact by construction rather than by numerical luck. On the wire, q=0 emits the
dense form (no bitmask, every cell present) and decodes bit-exactly to the original tensor.

## Wire layout

A payload is `header || bitmask || values`.

**Header** (44 bytes, little-endian, `struct` format `<4sHHHHIIIIIIQ`): magic `HQ1\0`,
format version, dtype code, flags, reserved, channels, height, width, q in
ten-thousandths, keep count, mask byte count, value byte count. Minimal, but sufficient to
fail closed on incompatible shape, dtype, q, keep cardinality or format version.

**Bitmask** (q>0 only): one fixed-order bit per spatial cell, `ceil(N/8)` bytes — 2,688
bytes at N=21,504. Bit value **1 means retained**. Cell `i` lives in byte `i // 8` at bit
`7 - (i % 8)`, i.e. bytes ascend with cell index and the **most significant bit within a
byte is the lowest cell index** (`np.packbits(..., bitorder="big")`). Padding bits past
cell N-1 must be zero.

**Values**: retained cells in **ascending row-major cell order**, each retained cell
holding all 256 FP32 channels contiguously (cell-major, channel-contiguous — not the
`[C,H,W]` channel-major layout). Little-endian `<f4`.

Decoding rebuilds the original dense shape and fills dropped cells with **exact zeros**.

### Measured serialized payload length

These are `len(payload.data)` from the synthetic round trip, not a theoretical tensor
estimate. Dense FP32 raw reference: 22,020,096 bytes.

| q | keep | actual bytes | vs raw |
| --- | --- | --- | --- |
| 0.00 | 21,504 | 22,020,140 | 100.0002% |
| 0.30 | 15,053 | 15,417,004 | 70.0133% |
| 0.50 | 10,752 | 11,012,780 | 50.0124% |
| 0.70 | 6,451 | 6,608,556 | 30.0115% |
| 0.90 | 2,150 | 2,204,332 | 10.0105% |
| 0.98 | 430 | 443,052 | 2.0120% |

The overhead above the ideal ratio is the 44-byte header plus the 2,688-byte bitmask.

Dense-zero zstd may later be measured as a **diagnostic**. It is not the sparse wire
representation and no zstd number should be reported as the hybrid-q payload.

## Teacher-map contract

```
I_t(h, w) = sum_c | C2(c, h, w) * grad_t(c, h, w) |
```

`training.build_teacher_maps` normalizes **each valid task map independently** (`l1` or
`max`) before combining, so a task with a 1000x larger gradient scale cannot dominate.
Tasks that are absent, zero-gradient or non-finite are recorded in `excluded_tasks` with a
reason and excluded — never silently treated as valid supervision. A frame with no valid
task returns `importance=None` and `is_supervisable == False`. Raw per-task masses are
retained separately in `loss_scales`.

`training.TeacherCacheRecord` is the contract for a future cache artifact: teacher map,
validity flags, identifiers, loss-scale metadata and the frozen checkpoint hash.
**Full C2 tensors are deliberately not cached.**

## Straight-through mask

`training.straight_through_mask(scores, q, temperature=...)` returns the hard
exact-cardinality 0/1 mask in the forward pass (bit-exact — the surrogate difference is
grouped so it contributes exactly zero) with a sigmoid surrogate gradient. **The
temperature is required from configuration**; Phase 1 does not choose or tune a production
value.

## Runtime guards

All in `guards.py`, all fail-closed: non-finite ranker scores, invalid or unregistered q,
incorrect keep cardinality, malformed bitmask or payload, duplicate or unordered retained
indices, incompatible shape/dtype/version, non-finite decoded values, and accidental
optimizer ownership of frozen parameters.

**Gradient policy.** There is deliberately *no* rule that aborts because a parameter group
had an exact-zero gradient on one batch. `training.GradientQualification` requires
finiteness on **every** update, requires nonzero-gradient evidence **over a qualification
window**, and merely logs isolated zero-gradient batches in `zero_gradient_batches`.

## Phase boundaries

Phase 1 (this commit) ends here. Not yet done, and not authorized by this commit:

- independent review of this implementation;
- GPU qualification against the frozen checkpoint;
- teacher-cache generation over real frames;
- ranker distillation training;
- inference, validation or test-set evaluation;
- OAI / CARLA transport measurement.

The test set remains untouched and reserved for independent publication confirmation.

## Commands intended for later qualification (not executed here)

Only the first is in scope for Phase 1; the rest are recorded for the later phases.

```bash
# Phase 1 (run): CPU synthetic tests
python3 -m unittest pole_lraspp_multimodal_fusion.object_head_pilot_v1.\
splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.tests.test_synthetic -v

# Phase 2+ (NOT run in Phase 1):
#   GPU forward-parity qualification at q=0 against the frozen p025 checkpoint
#   teacher-cache generation (teacher maps + validity + loss scales only, no C2)
#   ranker distillation training with an explicitly configured temperature
#   frozen-validation sweep over q in {0.00, 0.30, 0.50, 0.70, 0.90, 0.98}
#   measured payload / UE / network / edge / end-to-end latency over OAI
```
