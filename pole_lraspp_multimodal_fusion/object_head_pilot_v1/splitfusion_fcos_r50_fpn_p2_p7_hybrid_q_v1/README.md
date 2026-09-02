# Hybrid-q v1 — spatial transport sparsification at the frozen C2 split

Status: **Phase 6 is complete** — the validation accuracy-payload curve is measured at the six
registered q values on the stable epoch-4 ranker (section below), and the transport path now also
exposes a **continuous-q execution interface** (`continuous_q.py`) that can serve any q in
[0, 0.98] at the existing 1e-4 wire resolution. The continuous interface is execution only: it
adds no measurement, and accuracy away from the six measured anchors remains unvalidated.

Earlier phase results, unchanged:
`experiments/splitfusion_fcos_hybrid_q_v1/20260902_004213_phase3_gpu_qualification/`, terminal
`HYBRID_Q_PHASE3_QUALIFIED`; and
`experiments/splitfusion_fcos_hybrid_q_v1/20260901_180439_phase4_teacher_cache/`
(`teacher_cache_manifest.json`, `fit_reference_medians.json`,
`PHASE4_TEACHER_CACHE_SUMMARY.md`), terminal `HYBRID_Q_PHASE4_TEACHER_CACHE_COMPLETE`. The cache
shards themselves are large and deliberately untracked. Phase 5 is
`.../20260901_185725_phase5_ranker_training` (terminal `ROI_DROP_NOT_SAFE_ON_TRAIN_HOLDOUT`) and
Phase 6 is `.../20260902_182401_phase6_validation_curve` (terminal
`HYBRID_Q_PHASE6_VALIDATION_CURVE_COMPLETE`).

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
| 1x1 Conv, **no bias** | 8 -> 1 |

- **2,144** trainable parameters in **five** named tensors (`reduce.weight`, `reduce.bias`,
  `depthwise.weight`, `depthwise.bias`, `score.weight`), **45,760,512 MACs** (~45.76 M) at
  112x192. The MAC count is bias-free and therefore unchanged.
- **The final 1x1 conv has no bias.** A single global scalar added to every cell score is
  unidentifiable: it cannot change the cell ranking, the exact-cardinality selection or the
  hard mask, listwise softmax distillation is invariant to it, and a straight-through
  gradient on it would not correspond to any change in the transported cell set.
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
`task_losses` holds dense task-loss values **separately**, and Phase 4 used them to build the
frozen fit-train reference medians. `TeacherCacheRecord` declares the per-record content:
teacher maps, validity flags, identifiers, `gradient_mass`, `task_losses` and the checkpoint
hash — **full C2 tensors are deliberately not cached**. The Phase-4 shard is the columnar
realization of that record plus the `fit`/`holdout` split label; checkpoint, locked-config and
package-source hashes are held once per shard and so bind every record in it.

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
scale is rejected. Those medians now exist and are frozen — see the Phase-4 section below.

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

## Phase 3 qualification (executed)

`python3 -m ...splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.gpu_qualification --execute
HYBRID_Q_PHASE3_TRAIN_ONLY_GPU_QUALIFICATION --output <dir>` runs one bounded train-only
qualification against the frozen epoch-26 checkpoint behind the p025 forward lock, with every
perception parameter in eval mode and `requires_grad=False` and gradients admitted only through
C2 into the ranker mask. It qualified at **physical batch 16** (first attempt, no CUDA OOM,
peak allocated **23,828.7 MiB**) on four deterministic seed-20260829 fit-training batches, each
frame carrying both vehicle and person GT:

- **q=0 parity exact**: the ranker is never invoked, framed encode/decode is bit-identical, all
  37 compared FCOS/semantic/dense-depth/geometry tensors and the FCOS anchors match the existing
  noAE path bit-for-bit, the final p025 service outputs match, and no frozen parameter or buffer
  changed.
- **Teacher maps**: D, G, S and A all valid on every batch, every map finite, non-negative and
  positive-mass, each L1-normalized independently, combined map finite and positive. No cache written.
- **Four disposable updates** (one listwise distillation, then q-aware 0.30 / 0.50 / 0.70): all
  2,144 trainable parameters belong to the ranker, all five named tensors received finite and
  at-least-once-nonzero gradients, loss / clipped norm / parameters / optimizer state stayed
  finite, the frozen stack stayed exactly unchanged, and q=0 was never a training update. The
  disposable ranker state was discarded.
- **Mask and transport**: masks nested over 0.30/0.50/0.70/0.90/0.98 from one common ranking,
  keep counts exactly the registered table, retained C2 values bit-identical and dropped cells
  exact zero at every q.
- **Latency** is a single-frame diagnostic only (ranker ~0.12 ms median); it must not drive any
  tuning or selection rewrite. zstd was not run.

The provisional reference scales used inside that window are **disposable** and must not become
the Phase-4 frozen medians. They did not: Phase 4 recomputed all four medians from 847 fit-only
batches, and every value differs (e.g. G 0.12767770 disposable vs 0.10442041 frozen).

## Locked train fit/holdout split

Bound in `locked_config.json` (`train_split`) and enforced by `contract.py`, which fails closed
on episode-name, count and frame-identity drift. The partition is by episode over the registered
`split == "train"` manifest rows in manifest order.

| partition | episodes | frames | ordered-sample-id sha256 |
| --- | --- | --- | --- |
| fit | 8 | 13,543 | `3e20ccee…fa252e` |
| holdout | `canonical_v3_03_train_30_30_s503_tm1503`, `canonical_v3_04_train_50_50_s504_tm1504` | 3,284 | `8c7c4cc6…754f0a` |

The holdout partition is for **checkpoint selection only**: it must never contribute to the frozen
reference medians or to any optimizer step. Phase 3's seeded qualification batches included some
holdout frames; that was disposable qualification and is not carried into any Phase-4 number.

## Phase 4 teacher cache and reference medians (executed)

One create-only generation, `teacher_cache.py`, artifact
`experiments/splitfusion_fcos_hybrid_q_v1/20260901_180439_phase4_teacher_cache`. Terminal
`HYBRID_Q_PHASE4_TEACHER_CACHE_COMPLETE`. No ranker was constructed, no optimizer existed, no
optimizer step was taken, and no validation or test frame was read.

All **16,827** training frames were cached — 13,543 fit and 3,284 holdout — in **66** split-pure,
uncompressed 256-frame shards totalling 1.35 GiB, each written via temporary file then atomic
rename with a recorded sha256. Physical batch 16, augmentation off, seed 20260829, bf16 tail with
the fp32 C2 boundary. 755 s wall clock; peak allocated 23,476.3 MiB. Maps are FP32 and are not
converted to FP16. A bounded smoke (one fit batch, one holdout batch) exercised the real
write/read path first and its temporary output was then removed.

Teacher validity: D, S and A valid on all 16,827 frames; **G valid on 16,098**, with 729
`zero_gradient` exclusions. Those are legitimate absent supervision — 721 frames carry zero
POSITIVE GT objects and 8 carry a single GT object that matched no FCOS location — so they are
reported, not failed. Every frame retains at least three valid groups. Minimum map value
`6.82e-07`, minimum map mass `0.99999982`.

Frozen fit-reference medians, over 847 fit batches in registered order (846 x 16 + 1 x 7), holdout
batched separately and contributing nothing:

| group | frozen median | contributing fit batches |
| --- | --- | --- |
| D | 0.7242346405982971 | 847 |
| G | 0.1044204086065292 | 834 |
| S | 0.0963049530982971 | 847 |
| A | 0.0136528844013810 | 847 |

G's 13 rejected batches are those whose geometry loss was exactly zero, so its median is taken
over an even count and exercises the two-central-value rule.

This artifact is teacher supervision and loss scale only. It is **not** evidence about accuracy,
payload, latency or transport at any q.

## Phase 5 ranker training and train-holdout selection (executed)

`experiments/splitfusion_fcos_hybrid_q_v1/20260901_185725_phase5_ranker_training`.
**Terminal `ROI_DROP_NOT_SAFE_ON_TRAIN_HOLDOUT`** — no checkpoint/q pair passed every
registered preservation gate, so no checkpoint was selected and validation was not opened.

One 12-epoch scientific run: 10,164 updates (847 x 12), 6,776 q-aware, 81.2 min. Every fit
frame exactly once per epoch under a `20260829 + epoch` shuffle, batch 16 with the final
partial batch retained, **0** holdout frames in any optimizer batch, gradient qualification
passed every epoch, and all 371 frozen perception parameters and buffers bit-identical at
epochs 4, 8 and 12. The q cycle ran 0.30/0.50/0.70 continuously across epoch boundaries
(282/283 per q per epoch).

**The q-aware stage diverged.** Total loss rose monotonically 31.1 -> 245.3 across epochs
5-12, essentially every update was clipped at norm 5.0, and the ranker weight norm grew
5.87 -> 31.43 -> 40.56 at the three candidates. The hard mask is scale-invariant, so nothing
bounds score magnitude except the `0.1 x` distillation term, which loses; the normalized
geometry contribution went 10.33 -> 47.79, ending worse than a randomly initialized ranker.
This is a property of the locked hyper-parameters, not a numerical fault: stage A is
well-behaved throughout and every runtime and ownership gate held on all 10,164 updates. No
retune was attempted.

The q=0 holdout baseline **exactly reproduces** the published frozen p025 train-holdout
result — 2249/253/307 TP/FP/FN over 2,556 AVO-observable actor-frames, with precision, recall
and F1 bit-identical and XY MAE differing by 6.1e-06 m — confirming the evaluation is the
frozen p025 pipeline rather than a re-implementation.

The closest pair is epoch 4 (distillation-only) at q=0.30, worst absolute degradation 0.037936
at 0.700132 of the framed q=0 payload. **Every object-level detection and localization gate
passes there; all four failures are segmentation or segmentation-coupled** (vehicle IoU
+0.0379, foreground mIoU +0.0269, person box-mask IoU +0.0159, and person precision +0.0186
downstream of the p025 `semantic_support_threshold` 0.10 gate). Vehicle precision, person
recall and person 20-40 m recall all improve slightly. This corroborates
`rl_agent/density_knob/DENSITY_KNOB_RESULTS.md`: the dense per-pixel semantic head, not the
sparse object heads, is the binding constraint on spatial cell drop.

Scoring reuses the frozen v3.1 `audit_v1.score_arm` and `score_contract_v1.score_segmentation`
and the frozen p025 AVO view unmodified. Those scorers hardcode `contracts/<contract>/val/`,
so the train contract directory is exposed at that path by a read-only symlink alias: the
scoring code executed is byte-identical and only the split it reads changes. The validation
contract directory is never linked or read.

This does **not** show ROI drop is unachievable. The eight q-aware epochs are not a fair test
of their own hypothesis because they diverged, and the only well-behaved candidate never saw
the mask in the loop. A bounded, pre-registered change to the q-aware stage is untested and
therefore not falsified; it is not authorized by this commit.

## Phase 6 validation accuracy-payload curve (executed)

`experiments/splitfusion_fcos_hybrid_q_v1/20260902_182401_phase6_validation_curve`
(`validation_curve.csv`, `validation_curve.json`, `PHASE6_VALIDATION_CURVE_REPORT.md`), terminal
`HYBRID_Q_PHASE6_VALIDATION_CURVE_COMPLETE`. Measurement only, over the registered 3,345-frame
validation split at the one stable Phase-5 checkpoint `ranker_epoch_04.pt`: it trains nothing,
selects nothing, moves no threshold and leaves the locked test split untouched. The q=0 row is
the frozen p025 validation result re-scored by the identical Phase-6 scoring functions, which
reproduces all fifteen published values exactly.

| q | cells | framed bytes | ratio | veh F1 | person AVO F1 | fg mIoU | service |
|---|---|---|---|---|---|---|---|
| 0.00 | 21,504 | 22,020,140 | 1.0000 | 0.8989 | 0.7087 | 0.7135 | 7/9 |
| 0.30 | 15,053 | 15,417,004 | 0.7001 | 0.8991 | 0.6823 | 0.6881 | 7/9 |
| 0.50 | 10,752 | 11,012,780 | 0.5001 | 0.8973 | 0.6772 | 0.6559 | 4/9 |
| 0.70 | 6,451 | 6,608,556 | 0.3001 | 0.8854 | 0.6729 | 0.5816 | 4/9 |
| 0.90 | 2,150 | 2,204,332 | 0.1001 | 0.8195 | 0.5930 | 0.3850 | 3/9 |
| 0.98 | 430 | 443,052 | 0.0201 | 0.5391 | 0.4457 | 0.1288 | 3/9 |

Segmentation, not detection, is the binding constraint: vehicle F1 stays within 0.002 of
baseline at half the payload, while foreground mIoU falls monotonically. Under the cascade
registered before the measurement, q=0.30/0.50/0.70 are
localization-preserving/segmentation-reduced and q=0.90/0.98 are unusable.

## Low-q validation sweep (executed)

`experiments/splitfusion_fcos_hybrid_q_v1/20260902_195312_low_q_validation_curve`, terminal
`HYBRID_Q_LOW_Q_VALIDATION_CURVE_COMPLETE`. The denser ladder below q=0.30 that Phase 6 left
untested, measured on the same fixed validation split, through `continuous_q.transport`, at the
same stable epoch-4 ranker. It reuses the q=0 and q=0.30 rows verbatim rather than re-scoring them.

| q | cells | framed bytes | ratio | veh F1 | person AVO F1 | fg mIoU | preservation gates |
|---|---|---|---|---|---|---|---|
| 0.00 | 21,504 | 22,020,140 | 1.0000 | 0.8989 | 0.7087 | 0.7135 | 12/12 (dense identity) |
| 0.05 | 20,429 | 20,922,028 | 0.9501 | 0.8987 | 0.7015 | 0.7114 | 11/12 |
| 0.10 | 19,354 | 19,821,228 | 0.9001 | 0.8984 | 0.6968 | 0.7065 | 11/12 |
| 0.15 | 18,278 | 18,719,404 | 0.8501 | 0.8989 | 0.6952 | 0.7050 | 11/12 |
| 0.20 | 17,203 | 17,618,604 | 0.8001 | 0.8996 | 0.6928 | 0.6999 | 7/12 |
| 0.25 | 16,128 | 16,517,804 | 0.7501 | 0.8993 | 0.6884 | 0.6953 | 7/12 |
| 0.30 | 15,053 | 15,417,004 | 0.7001 | 0.8991 | 0.6823 | 0.6881 | 7/12 |

**Result: no q below 0.30 passes all twelve preservation gates** (`largest_q_passing_all_gates`
is null; q=0 is excluded from the verdict because it is the dense identity the gates are measured
against). q=0.05/0.10/0.15 miss on `person_avo_precision` alone; from q=0.20 the segmentation and
IoU gates go with it. The gate set is `contract.HOLDOUT_PRESERVATION_GATES`, registered before the
sweep and neither invented nor retuned for it, and there is no independent test-set confirmation.
So a cheap near-lossless rung below q=0.30 was looked for and **not** found — it is now measured
absent, not merely untested.

## Continuous-q execution interface

`continuous_q.py` lets the stable ranker and the existing v1 codec execute **any** finite q in
[0, 0.98], quantized only to the 1e-4 resolution the 44-byte header already carries
(`contract._q_to_e4`). It is additive and execution-only: `ranker.py`, `selection.py`,
`codec.py`, `guards.py` and `locked_config.json` are unmodified, and it reuses their private
generic helpers rather than reimplementing the sort, the mask or the wire format.

- `quantize_q(q) -> ContinuousQ` resolves the request onto the wire grid and reports `wire_q`,
  `q_e4`, the exact keep/drop counts, whether the value happens to be registered, and whether it
  is the dense bypass. A request is **never** snapped to an anchor.
- `select_cells`, `select_and_apply`, `encode`, `decode` and the end-to-end `transport` mirror
  the registered production call order, preserve the exact q=0 bypass, emit the same 44-byte
  header and sparse layout, and decode to the frozen `[256,112,192]` FP32 boundary with retained
  values bit-exact and dropped cells exactly zero.
- Fails closed on q outside [0, 0.98], non-finite q, malformed payloads and inconsistent
  q/keep-count/mask metadata, via the existing guards.

`tests/test_continuous_q.py` proves byte-for-byte parity with the discrete path at all six
registered q (same keep count, indices, mask, masked tensor, serialized payload and decoded
tensor), and executes q=0.2345 end to end: it stays 0.2345, keeps exactly 16,461 cells, is
neither q=0 nor q=0.30, and its retained cells are a strict superset of the q=0.30 mask.

**Arbitrary-q wire execution is mechanically qualified to the 1e-4 header resolution.** Every q
the interface accepts serialises, round-trips and decodes exactly, with the requested q equal to
the wire q, the exact keep cardinality, retained values bit-exact and dropped cells exactly zero;
Phase 7 re-confirmed this on 1,408 live payloads at eleven q values.

**Mechanical qualification is not measured accuracy.** Perception accuracy is validated only at
the measured q anchors — the six Phase-6 anchors plus the five low-q sweep points. Accuracy at any
other q is unmeasured and is neither interpolated nor validated here, and **no claim is made that
arbitrary continuous q is universally near-lossless**; the low-q sweep found no all-gate-passing
rung below q=0.30 at all. `contract.snap_continuous_q` remains available as a caller-side
conservative policy for accuracy-critical serving.

## Phase 7 lossless zstd transport measurement (executed)

`experiments/splitfusion_fcos_hybrid_q_v1/20260902_212403_phase7_zstd_measurement`
(`phase7_zstd_measurement.csv`, `.json`, `PHASE7_ZSTD_MEASUREMENT_REPORT.md`), terminal
`HYBRID_Q_PHASE7_ZSTD_MEASUREMENT_COMPLETE`. Transport size and **host** encode/decode cost only:
it trains nothing, evaluates no perception metric, touches no validation or test frame and changes
no model output. `zstd_transport.py` wraps `SparsePayload.data`; the zero-scattered dense tensor is
never compressed. `ranker.py`, `selection.py`, `codec.py`, `guards.py` and `locked_config.json` are
unmodified, which the run re-verifies by hash through `source_delta`.

Frozen compressor: python-zstandard 0.25.0 over zstd 1.5.7 (`cext`), level 1, `threads=0`, no
dictionary, `write_checksum=True`, `write_content_size=True`, `write_dict_id=False`, one
independent frame per camera frame, one reused compressor/decompressor context, no level search.
Data: a deterministic balanced 128-frame sample, 16 evenly spaced frames from each of the eight
registered fit episodes, no augmentation, zero holdout/validation/test frames.

| q | framed sparse B | zstd B median | zstd/sparse | vs framed q=0 | reduction | vs zstd q=0 |
|---|---|---|---|---|---|---|
| 0.00 | 22,020,140 | 15,993,204 | 0.7261 | 0.7261 | 27.39% | 1.0000 |
| 0.05 | 20,922,028 | 15,164,810 | 0.7245 | 0.6884 | 31.16% | 0.9480 |
| 0.10 | 19,821,228 | 14,328,548 | 0.7224 | 0.6503 | 34.97% | 0.8955 |
| 0.15 | 18,719,404 | 13,495,203 | 0.7205 | 0.6125 | 38.75% | 0.8435 |
| 0.20 | 17,618,604 | 12,663,262 | 0.7185 | 0.5749 | 42.51% | 0.7917 |
| 0.25 | 16,517,804 | 11,838,578 | 0.7165 | 0.5375 | 46.25% | 0.7402 |
| 0.30 | 15,417,004 | 11,021,751 | 0.7145 | 0.5003 | 49.97% | 0.6889 |
| 0.50 | 11,012,780 | 7,763,193 | 0.7059 | 0.3530 | 64.70% | 0.4862 |
| 0.70 | 6,608,556 | 4,592,850 | 0.6953 | 0.2087 | 79.13% | 0.2874 |
| 0.90 | 2,204,332 | 1,493,420 | 0.6771 | 0.0678 | 93.22% | 0.0933 |
| 0.98 | 443,052 | 290,842 | 0.6565 | 0.0132 | 98.68% | 0.0182 |

zstd buys a further **27-34% off the sparse payload at every q, losslessly**, so it composes with
sparsification instead of competing with it: q=0 alone drops 22.02 MB to 15.99 MB with no accuracy
cost whatsoever, and q=0.30 reaches an even 50.0% total reduction against the framed dense payload.
The ratio also improves monotonically as q rises, 0.726 down to 0.657, so the retained high-ranked
cells compress slightly *better* than the full tensor. This phase records that trend but does not
establish its mechanism, and nothing here depends on the explanation. All latency and throughput
columns are in the CSV.

**Round-trip integrity: 1,408/1,408** payloads (128 frames x 11 q) decompressed byte-for-byte
identical to `SparsePayload.data`, with requested q equal to wire q at 1e-4, header `q_e4` equal to
round(q x 10,000), exact keep count and framed length, retained values bit-identical, dropped
locations exactly zero, masks nested on all 128 frames, and the q=0 ranker bypass taken every time.
Timing and verification run as separate passes, so verification allocations never enter a
measurement. Frozen perception and ranker state are unchanged at the end.

Host cost on the measurement workstation (Intel Ultra 9 285K, RTX 5090, CPU threads pinned to 1):
zstd compression runs 534-612 MB/s and decompression 1,154-1,252 MB/s over the payload, so at q=0
compression costs 36.0 ms and decompression 17.6 ms, falling to 0.83 ms and 0.38 ms at q=0.98. These
are **current-host numbers, not Raspberry Pi UE or OAI transport latencies** — byte sizes are
host-independent, latencies are not. Sparse encode/decode are the existing codec's cost, reported
only for proportion. Threads are pinned because at torch's 24-thread default the same decode call
varied 5-68 ms while its components summed to 3-7 ms, which measured thread-pool scheduling on a
hybrid P/E-core CPU rather than codec cost.

## Phase boundaries

Not done and not authorized: test-set evaluation, OAI/CARLA transport measurement, quantization,
the AE families, accuracy measurement at any unregistered q, and any retune of the locked training
configuration. The test set remains untouched and reserved for independent publication
confirmation. zstd is now measured for **size and host cost only** (Phase 7); it has no measured
network or edge-deployment latency, and lossless compression by construction leaves the accuracy
curve untouched.

## Commands

Only the two CPU test commands are in scope for this commit; the phase runners below are recorded as executed history.

```bash
# Run: CPU synthetic tests
python3 -m unittest pole_lraspp_multimodal_fusion.object_head_pilot_v1.\
splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.tests.test_synthetic -v

# Run: CPU continuous-q interface tests
python3 -m unittest pole_lraspp_multimodal_fusion.object_head_pilot_v1.\
splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.tests.test_continuous_q -v

# Phase 4 (executed once; create-only, and the shards are not tracked):
python3 -m pole_lraspp_multimodal_fusion.object_head_pilot_v1.\
splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.teacher_cache \
  --execute HYBRID_Q_PHASE4_TEACHER_CACHE \
  --output experiments/splitfusion_fcos_hybrid_q_v1/<timestamp>_phase4_teacher_cache

# Phase 5 (executed once each; the run is create-only and resumable by epoch):
python3 -m pole_lraspp_multimodal_fusion.object_head_pilot_v1.\
splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.phase5_training \
  --execute HYBRID_Q_PHASE5_RANKER_TRAINING \
  --output experiments/splitfusion_fcos_hybrid_q_v1/<timestamp>_phase5_ranker_training

python3 -m pole_lraspp_multimodal_fusion.object_head_pilot_v1.\
splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.phase5_holdout \
  --execute HYBRID_Q_PHASE5_HOLDOUT_EVALUATION \
  --training experiments/splitfusion_fcos_hybrid_q_v1/<timestamp>_phase5_ranker_training

# Phase 7 (executed once; create-only, aggregates only and no payload blobs):
python3 -m pole_lraspp_multimodal_fusion.object_head_pilot_v1.\
splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.phase7_zstd_measurement \
  --execute HYBRID_Q_PHASE7_ZSTD_MEASUREMENT \
  --output experiments/splitfusion_fcos_hybrid_q_v1/<timestamp>_phase7_zstd_measurement

# Still NOT run here:
#   measured network / edge / end-to-end latency over OAI
#   any Raspberry Pi UE timing
```

The GPU phase runners need a torch build with kernels for this host's GPU (the Phase 4-7 runs used
torch 2.10.0.dev+cu128 on an RTX 5090, sm_120). A cu121 torch imports fine but dies in the first
frozen-backbone conv with `no kernel image is available for execution on the device`.
