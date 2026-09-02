# Hybrid-q Phase 3 — bounded train-only GPU qualification

**Terminal: `HYBRID_Q_PHASE3_QUALIFIED`**  ·  generated 2026-09-02T00:42:31.870699+00:00

Ranker correction applied, frozen epoch-26 model integrated, one bounded train-only GPU
qualification executed. No teacher cache, no epoch, no validation/test access, no evaluation,
no CARLA, no zstd.

## Binding and hashes

| item | value |
| --- | --- |
| perception forward lock | `86d6f13ae9168b33b697df5b785c5f7c320afc52cfdcded5b632d94a6d943fe1` |
| frozen checkpoint (epoch 26) | `da14d21edbd374c1c3abce02ca4674b9f4097becfba9759aba945cea160a297f` |
| checkpoint training-source commit | `0d5697b46e901d4494572b2e9a9863326947d24c` |
| recovery runtime | `splitfusion_fcos_r50_fpn_p2_p7_v1_numerical_recovery_v1.build_recovery_model`, yaw tau 0.01 |
| service policy | `splitfusion_fcos_r50_fpn_p2_p7_person_p025_calibration_v1.apply_p025_service_policy`, person threshold 0.25 |
| seed | 20260829 |
| precision | bf16 autocast tail, fp32 C2 boundary and losses (registered) |
| GPU / torch | NVIDIA GeForce RTX 5090 / 2.10.0.dev20251114+cu128 |

Hybrid-q package source SHA-256:

| file | sha256 |
| --- | --- |
| `__init__.py` | `9319a2b3ce9d492e1577908148ef31187ed648ef81944365dd907dee717b206c` |
| `codec.py` | `7b3833398a84fea31f65b86ec294c6675727390035b5761d372fd5a3cbba7b79` |
| `contract.py` | `aa73918d4cb6b2f656172a561ad7f63e8cf2d6f1b80aae5891c87a21b75704de` |
| `gpu_qualification.py` | `0032b9ac7edb89ec1b01616ea563bec73d04c20b164d6bbf8297f36d30eea7e6` |
| `guards.py` | `77d8d8bfd168e74a7f0b6a7e3c8e7abc4c3549a86cda392feb394d7580a33031` |
| `locked_config.json` | `d0aa723de315fe6009ab02ead7231ff4042fa874cc629c334e9ff9a82dae82f5` |
| `ranker.py` | `462536991f195651a1ee641f8e83444882ec370a8dffab72f13f0d770422b353` |
| `selection.py` | `ccc2b12919b078eac7af6131418989567d618d42f7b908b2db74df42e0342a71` |
| `tests/__init__.py` | `1c1c24c4edd4acf7bd8af46b63f47089fa1d86aa6481b9e43923be8810cf1692` |
| `tests/test_synthetic.py` | `de2388e391a069a38a712b20bdc1b19215c3d3c5cf51e45a2062460f97fd346b` |
| `training.py` | `a10775a3a9f3e051e3456c6c85cdc76fff47f417e9bd24ddecb6633dcda1161a` |

> hybrid_q_source_sha256 describes the committed source. Three files were edited after the run and before the commit, none of which affects a measured number: gpu_qualification.py (grad-norm result keys renamed to global_grad_norm_pre_clip / per_tensor_grad_norm_post_clip plus one comment), locked_config.json (status string and a phase3_qualification metadata block; every field load_locked_config validates is unchanged) and tests/test_synthetic.py (assertions updated to the Phase-3 facts). The executed q semantics, ranker, codec, guards, training primitives and perception binding are byte-identical to the committed source.

## 1. Ranker correction

- final Conv2d(8,1,kernel_size=1) now has bias=False
- parameter count **2144** in **5** named tensors: `reduce.weight`, `reduce.bias`, `depthwise.weight`, `depthwise.bias`, `score.weight`
- MAC count at 112x192 unchanged at **45,760,512**
- reason: a global scalar score bias cannot alter cell ranking; listwise softmax distillation is invariant to it and a straight-through gradient on it would not correspond to a change in the hard mask

## 2-3. Batch-size qualification and sample IDs

Ladder [16, 8, 4]; attempts: batch 16 -> ok.
**Selected physical batch 16** on the first attempt (no CUDA OOM, no fallback).
Peak allocated VRAM **23828.7 MiB** (reserved 24690.0 MiB).
runtime sizing only; the effective scientific batch is unchanged.

Four deterministic fit-training batches (train split, augmentation disabled), each frame carrying both vehicle and person GT:

| batch | vehicle GT | person GT | sample IDs |
| --- | --- | --- | --- |
| 1 | 42 | 28 | `extra_v3_13_train_30_30_s805_tm1805_001230_frame5147`<br>`extra_v3_12_train_50_50_s804_tm1804_001109_frame4665`<br>`extra_v3_12_train_50_50_s804_tm1804_000929_frame3945`<br>`canonical_v3_04_train_50_50_s504_tm1504_000980_frame4146`<br>`extra_v3_13_train_30_30_s805_tm1805_000305_frame1447`<br>`extra_v3_10_train_50_50_s802_tm1802_001431_frame5952`<br>`extra_v3_14_train_50_50_s806_tm1806_000767_frame3290`<br>`canonical_v3_02_train_50_50_s502_tm1502_000801_frame3422`<br>`extra_v3_09_train_30_30_s801_tm1801_001168_frame4901`<br>`extra_v3_14_train_50_50_s806_tm1806_001085_frame4562`<br>`extra_v3_10_train_50_50_s802_tm1802_000077_frame536`<br>`extra_v3_12_train_50_50_s804_tm1804_001040_frame4389`<br>`canonical_v3_04_train_50_50_s504_tm1504_001031_frame4350`<br>`canonical_v3_02_train_50_50_s502_tm1502_000063_frame470`<br>`extra_v3_12_train_50_50_s804_tm1804_001783_frame7361`<br>`extra_v3_10_train_50_50_s802_tm1802_000982_frame4156` |
| 2 | 43 | 32 | `extra_v3_14_train_50_50_s806_tm1806_001191_frame4986`<br>`canonical_v3_02_train_50_50_s502_tm1502_001448_frame6010`<br>`extra_v3_11_train_30_30_s803_tm1803_001300_frame5425`<br>`extra_v3_13_train_30_30_s805_tm1805_000778_frame3339`<br>`extra_v3_10_train_50_50_s802_tm1802_000834_frame3564`<br>`extra_v3_14_train_50_50_s806_tm1806_000158_frame854`<br>`extra_v3_12_train_50_50_s804_tm1804_000098_frame621`<br>`extra_v3_10_train_50_50_s802_tm1802_001725_frame7128`<br>`extra_v3_13_train_30_30_s805_tm1805_001024_frame4323`<br>`extra_v3_09_train_30_30_s801_tm1801_000074_frame525`<br>`extra_v3_09_train_30_30_s801_tm1801_000119_frame705`<br>`canonical_v3_02_train_50_50_s502_tm1502_000765_frame3278`<br>`extra_v3_12_train_50_50_s804_tm1804_000092_frame597`<br>`canonical_v3_01_train_30_30_s501_tm1501_001176_frame4918`<br>`canonical_v3_04_train_50_50_s504_tm1504_001158_frame4858`<br>`extra_v3_10_train_50_50_s802_tm1802_000849_frame3624` |
| 3 | 40 | 27 | `canonical_v3_04_train_50_50_s504_tm1504_001394_frame5802`<br>`canonical_v3_01_train_30_30_s501_tm1501_000155_frame834`<br>`canonical_v3_02_train_50_50_s502_tm1502_001391_frame5782`<br>`extra_v3_11_train_30_30_s803_tm1803_000140_frame785`<br>`extra_v3_12_train_50_50_s804_tm1804_001047_frame4417`<br>`extra_v3_12_train_50_50_s804_tm1804_001666_frame6893`<br>`canonical_v3_04_train_50_50_s504_tm1504_001565_frame6486`<br>`canonical_v3_04_train_50_50_s504_tm1504_001591_frame6590`<br>`extra_v3_12_train_50_50_s804_tm1804_001243_frame5201`<br>`extra_v3_12_train_50_50_s804_tm1804_000914_frame3885`<br>`canonical_v3_04_train_50_50_s504_tm1504_000013_frame278`<br>`extra_v3_14_train_50_50_s806_tm1806_000074_frame518`<br>`extra_v3_09_train_30_30_s801_tm1801_001434_frame5965`<br>`canonical_v3_04_train_50_50_s504_tm1504_001127_frame4734`<br>`canonical_v3_02_train_50_50_s502_tm1502_000685_frame2958`<br>`extra_v3_10_train_50_50_s802_tm1802_000961_frame4072` |
| 4 | 44 | 34 | `extra_v3_14_train_50_50_s806_tm1806_000906_frame3846`<br>`extra_v3_11_train_30_30_s803_tm1803_000848_frame3617`<br>`extra_v3_11_train_30_30_s803_tm1803_001084_frame4561`<br>`extra_v3_14_train_50_50_s806_tm1806_000832_frame3550`<br>`extra_v3_12_train_50_50_s804_tm1804_000381_frame1753`<br>`canonical_v3_04_train_50_50_s504_tm1504_000902_frame3834`<br>`extra_v3_10_train_50_50_s802_tm1802_000775_frame3328`<br>`extra_v3_09_train_30_30_s801_tm1801_000536_frame2373`<br>`canonical_v3_04_train_50_50_s504_tm1504_000913_frame3878`<br>`canonical_v3_04_train_50_50_s504_tm1504_000451_frame2030`<br>`extra_v3_12_train_50_50_s804_tm1804_001502_frame6237`<br>`extra_v3_14_train_50_50_s806_tm1806_000843_frame3594`<br>`canonical_v3_03_train_30_30_s503_tm1503_001427_frame5922`<br>`extra_v3_14_train_50_50_s806_tm1806_001451_frame6026`<br>`canonical_v3_03_train_30_30_s503_tm1503_000186_frame958`<br>`extra_v3_09_train_30_30_s801_tm1801_001176_frame4933` |

## 4. q=0 parity

Fixed training batch 1 (16 frames), fp32_no_autocast_inference_mode. **All exact.**

| check | result |
| --- | --- |
| ranker invocations at q=0 | 0 (bypassed every frame: True) |
| C2 before vs after framed encode/decode | bit-identical: True (framed payload 22,020,140 B) |
| raw FCOS / semantic / dense-depth / geometry tensors vs noAE path | 37 tensors bit-identical: True |
| FCOS anchors | bit-identical: True |
| final p025 service outputs | bit-identical: True, mismatched fields [] |
| frozen parameters and buffers | unchanged: True |

## 5. Teacher-map qualification

Detached FP32 C2 leaf, `I_t(h,w) = sum_c |C2 * grad_t|`, l1 normalization per group, equal_weight combination. No teacher cache written (False).

Valid groups on every batch: **D, G, S, A** (all four). No absent, zero-mass, non-finite or disconnected group on any batch.

| batch | q=0 task loss D / G / S / A | raw gradient mass D / G / S / A | combined map min | combined map L1 |
| --- | --- | --- | --- | --- |
| 1 | 0.74046 / 0.12203 / 0.08349 / 0.01657 | 0.16280 / 0.07725 / 0.03140 / 0.01882 | 1.086e-06 | 1.000000 |
| 2 | 0.73401 / 0.11735 / 0.08018 / 0.01411 | 0.34923 / 0.20157 / 0.21849 / 0.01751 | 9.072e-07 | 1.000000 |
| 3 | 0.73116 / 0.20574 / 0.12758 / 0.01376 | 0.08504 / 0.16294 / 0.03064 / 0.02739 | 1.155e-06 | 1.000000 |
| 4 | 0.77818 / 0.13333 / 0.09002 / 0.01328 | 0.43994 / 0.74795 / 0.03477 / 0.01316 | 7.654e-07 | 1.000000 |

All group losses finite; all importance maps finite, non-negative and positive-mass; combined map finite and positive.

**Disposable** provisional reference scales (median q=0 task loss over the same train-only batches) — A=0.013938, D=0.737231, G=0.127678, S=0.086757. These must **not** become the Phase-4 frozen medians.

## 6. Disposable optimizer qualification

Fresh seeded ranker + ranker-only AdamW (AdamW lr=1e-3 wd=1e-4 constant, clip 5.0, ranker parameters only). Window of 4 updates: one listwise distillation, then q-aware at 0.30, 0.50, 0.70 in exact order. q=0 used as a training update: False.

| # | kind | q | loss | pre-clip grad norm | clip applied | all 5 tensors nonzero |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | listwise_distillation | — | 9.979189 | 0.083317 | False | True |
| 2 | q_aware | 0.3 | 5.577605 | 3.784474 | False | True |
| 3 | q_aware | 0.5 | 16.734179 | 8.746886 | True | True |
| 4 | q_aware | 0.7 | 29.952875 | 9.942804 | True | True |

Per-tensor gradient norms (post-clip):

| update | `reduce.weight` | `reduce.bias` | `depthwise.weight` | `depthwise.bias` | `score.weight` |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.066669 | 0.013560 | 0.039933 | 0.020496 | 0.017270 |
| 2 | 1.917267 | 0.451144 | 0.421284 | 1.964589 | 2.530951 |
| 3 | 1.687447 | 0.386162 | 0.454776 | 3.047475 | 3.536873 |
| 4 | 1.317600 | 0.331478 | 0.552797 | 3.198934 | 3.551800 |

- trainable parameters: **2144**, all belonging to the ranker: True
- all 5 named ranker tensors received finite gradients on every update; disconnected [], never-nonzero [], zero-gradient batches []
- loss, gradient norm, parameters and optimizer state finite after every step
- frozen perception parameters and buffers exactly unchanged after every step; no frozen parameter received a gradient
- **disposable ranker state discarded and not retained: True**; full training must restart from seed 20260829

## 7. Mask and transport qualification

From one common ranking on a single frame: masks nested over increasing q: **True**; keep counts exact: **True**; codec round trips exact: **True**.

| q | keep | registered keep | framed bytes | framed ratio | retained bit-identical | dropped exact zero |
| --- | --- | --- | --- | --- | --- | --- |
| 0.30 | 15,053 | 15,053 | 15,417,004 | 0.700132 | True | True |
| 0.50 | 10,752 | 10,752 | 11,012,780 | 0.500123 | True | True |
| 0.70 | 6,451 | 6,451 | 6,608,556 | 0.300114 | True | True |
| 0.90 | 2,150 | 2,150 | 2,204,332 | 0.100105 | True | True |
| 0.98 | 430 | 430 | 443,052 | 0.020120 | True | True |

## 8. Latency and payload (diagnostic)

One frame, 5 warm-up + 20 measured repetitions. single-frame diagnostic; no tuning or selection rewrite is authorized on it. zstd run: False.

Ranker GPU time: median **0.118 ms**, p95 **0.127 ms** (q-independent).

| q | selection ms (med / p95) | GPU->CPU + pack ms | unpack + CPU->GPU ms | total prep ms | framed payload B | ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 0.00 | 0.000 / 0.000 | 9.561 / 14.787 | 12.842 / 65.365 | 27.956 / 79.642 | 22,020,140 | 1.000000 |
| 0.30 | 0.404 / 0.419 | 2.725 / 3.180 | 8.747 / 52.178 | 11.977 / 55.421 | 15,417,004 | 0.700132 |
| 0.50 | 0.394 / 0.541 | 2.099 / 2.956 | 29.627 / 74.510 | 32.645 / 77.087 | 11,012,780 | 0.500123 |
| 0.70 | 0.375 / 0.395 | 1.564 / 1.828 | 5.314 / 7.554 | 7.324 / 9.451 | 6,608,556 | 0.300114 |
| 0.90 | 0.303 / 0.343 | 0.740 / 0.766 | 12.580 / 73.948 | 13.818 / 75.093 | 2,204,332 | 0.100105 |
| 0.98 | 0.276 / 0.345 | 0.395 / 0.403 | 6.034 / 39.907 | 6.814 / 40.707 | 443,052 | 0.020120 |

Reading the table honestly: the two host-side stages are noisy at this sample size — the p95 column is
dominated by allocator and page-fault variance, and the q=0.50 median total exceeds the q=0.30 median
total, which is host noise rather than a real effect. Only the monotone stages (packing time and payload
bytes falling with q) and the sub-0.13 ms ranker cost are worth carrying forward. Total prep is the sum of
independently timed stages, so its p95 is the p95 of the summed vector, not a true end-to-end p95. No
tuning or selection rewrite is authorized on this measurement.

## Scope closed

train-only: True · augmentation: False · validation/test accessed: False · teacher cache written: False · epochs trained: 0 · evaluation run: False · CARLA launched: False · zstd run: False

Stopped before teacher-cache generation and training. Machine-readable result: [`qualification_result.json`](qualification_result.json).
