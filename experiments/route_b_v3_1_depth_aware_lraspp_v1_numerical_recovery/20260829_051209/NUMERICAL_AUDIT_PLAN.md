# Depth-aware LR-ASPP deterministic numerical audit plan

Registered before either clean replay and before any repair.

## Immutable inputs and reconstruction

The failed experiment remains read-only. Both reproductions will be separate processes built from commit `049f7029d9156871e02b2aed34da3cdbcbd842ef`, the exact official MobileNetV3-Large weight, scientific seed `20260829`, epoch-1 sampler seed `20260830`, physical batch 16, accumulation 1, full FP32 AdamW, and the registered LR schedule. The existing hash-validated train cache will be memory-mapped read-only.

Each process will reseed Python, NumPy, Torch CPU and CUDA before dataset/model construction. It will reconstruct the epoch-1 `RandomSampler` permutation, load ordered batches 1–15 with the registered deterministic per-sample augmentation, and hash each batch's input, segmentation, heatmap, dense targets/masks, radar points, owner targets, sample IDs, and collision records. It will build a pristine official-initialized model and empty AdamW state, then execute updates 1–13 exactly as `train.py` does. Batch 14 will initially stop after forward and loss construction.

The lost failed-process state is not claimed byte-identical. Deterministic reproduction is established only if the two new pristine processes agree mutually and both recover the recorded batch-14 identities and failure.

## Instrumentation

Immediately summarized tensors will include model inputs; every leaf backbone, fusion, neck, and head module output; all raw heads; gathered positive-cell predictions; logits and softmax; raw residuals; `anchor + delta*residual`; weighted depth contributions; summed log depth; `expm1` input/output; final decoded depth; physical-ray offsets and ray components; derived local XYZ; raw log dimensions; dimension `exp`, clamp and log; dense-depth and radar samples; every unweighted/weighted loss; total loss; gradients before/after clipping; parameters; persistent buffers; and AdamW step/`exp_avg`/`exp_avg_sq` state.

Every summary records operation/module name, shape, dtype, finite and non-finite counts, finite minimum/maximum, absolute maximum, and the producing batch/sample/object/head when applicable. Functional loss operations receive explicit probes because module hooks cannot observe them. Summaries are reduced immediately; full activation histories are not retained.

## Equality and hashes

- Tensor bytes are hashed with name, dtype, shape, and contiguous CPU bytes using SHA-256.
- Nested model, optimizer, RNG, batch, target, sampler, and sample-ID states use canonical key ordering and length-delimited components.
- Initial model/buffer, initial optimizer, Python/NumPy/Torch RNG, each ordered batch, and post-update-13 model/optimizer/RNG hashes must be identical across reproductions.
- CUDA deterministic equality requires exact hashes for persistent state and batch tensors. Operation summaries must be exactly equal for finite counts and extrema unless a documented CUDA reduction permits only bit-level nondeterminism; any such mismatch fails the deterministic-reproduction gate rather than being excused.
- Both runs must identify the same first non-finite operation, head, positive object/cell, and causal input.

## Allowed single-repair classes

- **Class A:** one algebraically equivalent evaluation that preserves architecture, targets, weights, loss meaning, capacity, initialization, optimizer, and schedule. Safe-domain forward equivalence and finite backward behavior are mandatory.
- **Class B:** zero-initialize only the final depth-residual convolution, and only if all five protocol predicates prove unbounded depth `expm1` is first non-finite and no Class-A FP32 equivalent exists. This is an explicit initialization correction, not an algebraic equivalence.

Exactly one repair may be registered before scientific source changes. Two independent repaired replays, batches 14–15, existing contract tests, memory, and an entirely finite disposable epoch 1 must pass. Qualification state is discarded.

## Prohibited repair classes

Residual sigmoid/tanh/clipping, depth/log-depth maximum clamps, overflow bins/classes, anchor/range changes, loss/endpoint removal or reweighting, optimizer/clip/LR/schedule changes, unrelated initialization changes, batch reduction, sample/object exclusion, architecture variants, retries with alternate fixes, and validation-informed changes are forbidden. If the necessary repair falls outside the registered Class A or narrow Class B, the terminal is `DEPTH_AWARE_CONTRACT_INVALID`.
