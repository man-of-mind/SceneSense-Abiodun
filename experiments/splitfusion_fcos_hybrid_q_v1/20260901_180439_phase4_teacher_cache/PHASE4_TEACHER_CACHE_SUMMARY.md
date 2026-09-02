# Hybrid-q Phase 4 — frozen train-only teacher cache and fit-reference medians

**Terminal: `HYBRID_Q_PHASE4_TEACHER_CACHE_COMPLETE`.** One create-only generation, executed
once. No ranker was constructed, no optimizer existed, no optimizer step was taken, and no
validation or test frame was read.

| | |
| --- | --- |
| artifact | `experiments/splitfusion_fcos_hybrid_q_v1/20260901_180439_phase4_teacher_cache` |
| checkpoint | frozen epoch-26, `da14d21e…0a297f`, numerical-recovery runtime, eval, `requires_grad=False` |
| locked config | `b2b0d842…fcde6` |
| device | RTX 5090, 31.35 GiB total (gate: >= 30 GiB) |
| batch | physical 16, the Phase-3-qualified size |
| precision | bf16 autocast tail, fp32 C2 boundary and losses (registered) |
| seed / augmentation | 20260829 / disabled |
| wall clock | 755.0 s (12.6 min) |
| peak VRAM | allocated 23,476.3 MiB, reserved 24,284.0 MiB |

## 1. Locked train fit/holdout split

Partition is by episode over the registered `split == "train"` manifest rows in manifest order,
and is now bound in `locked_config.json` (`train_split`) and enforced by `contract.py`.

| partition | episodes | frames | ordered-sample-id sha256 |
| --- | --- | --- | --- |
| fit | 8 | **13,543** | `3e20ccee…fa252e` |
| holdout | `canonical_v3_03_train_30_30_s503_tm1503`, `canonical_v3_04_train_50_50_s504_tm1504` | **3,284** | `8c7c4cc6…754f0a` |
| total | 10 | **16,827** | |

All three counts match the registered expectation exactly. `load_locked_config` fails closed on
episode-name, count and frame-identity drift; this was confirmed by mutation (fit count, holdout
episode list, fit identity digest and cache batch size each raise).

The holdout partition is reserved for checkpoint selection. It contributed **nothing** to the
reference medians below. Phase 3's seeded qualification batches happened to include some holdout
frames; that was disposable qualification and is not carried into any Phase-4 number.

## 2. Cache contents and layout

Per frame: sample ID, episode ID, split label, the combined FP32 `[112,192]` teacher map, D/G/S/A
validity flags, per-group exclusion reasons, per-group gradient masses, and the batch's dense q=0
task losses. Checkpoint, locked-config and package-source hashes are stored once per shard and so
bind every record in it.

Deliberately absent: C2 tensors, RGB, radar, GT, model outputs, and anything validation- or
test-derived. Enforced two ways — the only tensor in a deserialized shard must be `importance`
with shape `[n,112,192]`, and shard bytes must stay inside the teacher-map budget (a leaked C2
tensor would add ~22 MB per frame).

| | |
| --- | --- |
| shards | **66**, 256 frames each, split-pure (a shard never mixes fit and holdout) |
| total bytes | 1,452,208,370 (1.35 GiB); tensor payload 1,447,391,232 |
| compression | none |
| write | temporary file then atomic rename; sha256 recorded per shard |
| free disk before run | 86.9 GiB against a 2.35 GiB requirement |

Maps are stored FP32 and are **not** converted to FP16.

## 3. Teacher maps

`I_t(h,w) = sum_c |C2(c,h,w) * grad_t(c,h,w)|`, each valid group L1-normalized independently,
valid groups combined with equal weight, combined map L1-normalized.

| group | valid frames | excluded | reason |
| --- | --- | --- | --- |
| D | 16,827 / 16,827 | 0 | — |
| G | **16,098** / 16,827 | **729** | `zero_gradient` |
| S | 16,827 / 16,827 | 0 | — |
| A | 16,827 / 16,827 | 0 | — |

The 729 G exclusions are legitimate absent supervision, not a defect: **721** of them are frames
carrying zero POSITIVE GT objects, and the other **8** carry exactly one GT object that produced
no matched FCOS location, so the geometry loss has an empty matched set. Per the Phase-4 contract
these are reported rather than failed, and every one of the 16,827 frames still has at least one
valid group — in fact at least three. No frame was dropped.

Over the whole cache the minimum map value is `6.82e-07` (non-negative) and the minimum map mass
is `0.99999982`, i.e. every combined map is finite, non-negative and positive-mass, sitting at the
expected L1 norm of 1 up to FP32 summation error.

## 4. Frozen fit-reference medians

Fit partition only, batch 16, deterministic registered fit-frame order, augmentation off, every
fit frame exactly once: **847 fit batches** (846 x 16 + 1 x 7). Holdout is batched separately into
206 batches so no holdout frame can enter a fit batch. Conventional median over finite positive
q=0 task losses, averaging the two central values for an even count.

| group | frozen median | contributing fit batches | rejected | min | max |
| --- | --- | --- | --- | --- | --- |
| D | **0.7242346405982971** | 847 / 847 | 0 | 0.103286 | 1.283679 |
| G | **0.1044204086065292** | **834** / 847 | 13 | 0.001096 | 23.274580 |
| S | **0.0963049530982971** | 847 / 847 | 0 | 0.000039 | 0.343960 |
| A | **0.0136528844013810** | 847 / 847 | 0 | 0.006062 | 0.066143 |

All four are finite and positive. G's 13 rejected batches are the batches whose geometry loss was
exactly zero — the same empty-matched-set condition as above — so G's median is taken over an even
count and does exercise the two-central-value rule. The 847 raw per-batch task-loss rows are
retained in `fit_reference_medians.json` so each median is independently recomputable.

## 5. Bounded smoke

One fit batch and one holdout batch were written to a temporary smoke cache first (4.4 s, 2
shards) and verified: shard write and read-back succeeded; sample IDs and split labels reconciled
against the locked registered order; maps were FP32 `[112,192]`, finite, non-negative and
positive-mass; the only serialized tensor was the teacher map, with no C2 tensor present; and the
frozen model state was unchanged. Only the temporary smoke output was removed before the full run.

## 6. Completion gates

| gate | result |
| --- | --- |
| exactly 16,827 unique training frames | 16,827 cached, 16,827 unique |
| exactly 13,543 fit and 3,284 holdout | 13,543 / 3,284 |
| no validation or test frames | cached set equals the registered train set exactly; val intersection 0 |
| no missing, duplicated or unexpected IDs | set and order both match the locked partition |
| all maps finite, non-negative, positive-mass | min value 6.82e-07, min mass 0.99999982 |
| all shard hashes verified | 66 / 66, re-verified independently in a fresh process |
| D/G/S/A validity and exclusion counts reported | see section 3 |
| four finite positive fit-reference medians | see section 4 |
| no frozen parameter or buffer changed | `require_module_state_unchanged` passes after the smoke and at end of run |
| no optimizer step occurred | no optimizer and no ranker were ever constructed |

## 7. Scope

This artifact is teacher supervision and loss scale only. It is not evidence about accuracy,
payload, latency or transport at any q. The ranker is untrained; distillation and the q-aware
stage have not been run. Nothing here touches quantization, zstd, the AE families, OAI or CARLA,
and the locked test split remains unopened.
