# SplitFusion AE128 — Phase 9B bounded GPU qualification

Qualification only. **No scientific AE training was started, no validation or
test split was read, no AE64/AE32 was constructed or trained, and no checkpoint
was written.** Five disposable updates were taken on a freshly constructed
AE128 purely to show that the Phase-9A implementation executes correctly on the
accelerator; the AE and its optimizer were discarded. Every loss number below
is the untrained implementation's behaviour under a diagnostic optimizer and is
**not** an accuracy, payload or convergence result.

Runner: `ae_gpu_qualification.py`.
Artifact: `experiments/splitfusion_fcos_ae_v1/20260903_010229_phase9b_ae128_gpu_qualification/`
(`ae128_gpu_qualification.json`, sha256
`45c3af302f7e29b6c56d37c96b5392fe194a623dd581d1f913c227847aba5447`).
Host: RTX 5090 (31.35 GiB), `/usr/bin/python3` 3.10.12, torch
2.10.0.dev20251114+cu128 / CUDA 12.8. Wall clock 8.4 s.

## Binding, verified before any GPU work

Reused verbatim from `phase5_common.bind_inputs()` plus the stable-ranker hash,
and each of the four authorized digests is additionally restated in the runner
so a contract edit cannot move them silently.

| Input | sha256 | verified |
| --- | --- | --- |
| Frozen perception checkpoint (epoch 26) | `da14d21e…0a297f` | yes |
| Stable epoch-4 ranker | `07781c56…9b79cb5` | yes |
| p025 forward lock | `86d6f13a…6d943fe1` | yes |
| Hybrid-q locked configuration | `b2b0d842…50bcdefe`* | yes |
| Phase-4 teacher-cache manifest | `e1ef600e…7fc273` | yes |
| Phase-4 teacher-cache shards | all **66/66** hashes | yes |

\* full value `b2b0d8427bd867f46058ebba49ac6a183eb89413b4d69326fef93b150ebfcde6`.

`phase5_common.source_delta` confirms every frozen-semantics hybrid-q module is
bit-identical to what Phase 4 recorded. Fit frames only: 13,543 fit frames,
digest `3e20ccee…fa252e`; **0 holdout, validation or test frames were read**.

## Frozen state

Perception (371 parameter+buffer tensors) and the stable ranker (5 tensors) were
put in eval mode with `requires_grad=False` and every gradient cleared. A
per-tensor sha256 plus an aggregate digest was recorded **before** qualification:

- perception aggregate `bdb3a245f26fb17a9b0185c6c140ebd3774aa5f3f6b4d984b7e1d9665c1f3a53`
- ranker aggregate `604fb51428e30801699fcc5af30b7a6a2d190113ce2a4761e752f7252ad6a621`

Both aggregates and all 376 per-tensor hashes are **identical after
qualification**, and `guards.require_module_state_unchanged` was additionally
re-checked after every single update.

## Teacher-cache join

80 fit frames were picked by a seeded permutation
(`torch.Generator().manual_seed(20260829)`) over the fit partition. The fit
position selects which shard to open, but the record is then looked up in that
shard's own `sample_ids` **by identity**; a frame absent from the shard the
position pointed at fails closed. 43 shards were opened, 80/80 frames joined.
Every joined record supplied one FP32 `[112,192]` combined importance map plus
`valid_groups`/`excluded_groups`, and the observed minimum over the selection is
**3 valid groups** (the frames that lose one lose `G` with the recorded
`zero_gradient` reason), meeting the registered `>= 3` requirement.

## Batch sizing and VRAM

Batch **16** was tried once and succeeded; the batch-8 fallback was never
needed. Peak allocated VRAM **4.056 GiB** (reserved 4.486 GiB) against the
30 GiB budget.

## The five disposable updates

One Stage-A update at q=0, then exactly one Stage-B cycle — one q shared per
batch, independent per-frame top-K masks. Per-frame keep counts were exact and
identical within every batch. `total = plain + combined_importance`; per-frame
errors are normalized within each frame.

| # | stage | q | keep/frame | total | plain | comb.-imp. | pf-plain med / p95 / max | pf-imp med / p95 / max | ‖g‖ | enc / dec | s |
| --: | --- | --: | --: | --: | --: | --: | --- | --- | --: | --- | --: |
| 1 | A | 0.00 | 21,504 | 0.997764 | 0.498535 | 0.499229 | 0.4975 / 0.5041 / 0.5081 | 0.4987 / 0.5080 / 0.5113 | 0.9052 | 2.46e-4 / 0.9052 | 0.32 |
| 2 | B | 0.00 | 21,504 | 0.920130 | 0.457369 | 0.462762 | 0.4570 / 0.4633 / 0.4649 | 0.4615 / 0.4680 / 0.4727 | 0.8812 | 0.2677 / 0.8395 | 0.05 |
| 3 | B | 0.30 | 15,053 | 0.959531 | 0.524638 | 0.434892 | 0.5236 / 0.5450 / 0.5522 | 0.4338 / 0.4473 / 0.4494 | 0.6524 | 0.1586 / 0.6328 | 0.07 |
| 4 | B | 0.50 | 10,752 | 1.032408 | 0.602642 | 0.429765 | 0.6042 / 0.6148 / 0.6172 | 0.4310 / 0.4397 / 0.4429 | 0.5290 | 0.1154 / 0.5163 | 0.06 |
| 5 | B | 0.70 | 6,451 | 1.191662 | 0.719694 | 0.471968 | 0.7212 / 0.7335 / 0.7379 | 0.4687 / 0.5135 / 0.5765 | 0.4509 | 0.1019 / 0.4393 | 0.06 |

Optimizer: disposable AdamW, lr 1e-3, weight decay 1e-4, owning AE parameters
only (`ae_loss.require_ae_only_optimizer`). **These settings are diagnostic and
are not the final scientific training configuration.** AE parameters and all
AdamW state stayed finite on every update.

### Gradient reachability

All 8 named AE tensors received a finite, nonzero gradient — in fact on *every*
update, so there were no zero-gradient or missing-gradient batches at all, and
`GradientQualification.require_qualified()` passed over the complete 5-update
window. No gradient reached the ranker or any frozen perception parameter
(`.grad is None` everywhere, re-checked after each step).

**Observation for the training review, not a defect.** At the committed
orthogonal initialization the encoder is nearly gradient-starved: on update 1
the encoder norm is 2.46e-4 against a decoder norm of 0.905. That is the
expected consequence of the init — the residual is exactly the component of C2
orthogonal to the encoder's row space, so projecting it back through `W` almost
cancels. It recovers immediately (0.268 by update 2, and `project` reaches
0.0986 by update 5), but the first updates move the decoder almost alone and a
warm-up or an encoder-side learning-rate choice is worth considering when the
real training protocol is designed.

## Frozen tail on reconstructed C2

The unchanged frozen tail was run once on AE-reconstructed FP32 C2 at q=0 and
q=0.70. Both produced **78 output tensors, all finite**, with a shape signature
identical to the signature the same tail produces from the original C2 (that
one reference forward establishes the expected shapes and is the only extra
call). **No accuracy was scored.**

## Raw-byte UINT8 + mandatory-zstd round trip

Routing tag `3660115144` (nonzero qualification tag derived from a sha256 of a
qualification label; it is a routing discriminator, not a checkpoint identity).

| q | keep | pre-zstd B | analytical B | agrees | zstd B | ratio | decompressions in `receive` | out C2 |
| --: | --: | --: | --: | --- | --: | --: | --: | --- |
| 0.00 | 21,504 | 2,753,586 | 2,753,586 | yes | 2,452,473 | 0.8906 | 1 | `[256,112,192]`, finite, `cuda:0` |
| 0.30 | 15,053 | 1,930,546 | 1,930,546 | yes | 1,734,524 | 0.8985 | 1 | `[256,112,192]`, finite, `cuda:0` |

Header family id (1 = AE128), transported latent width (128) and routing tag all
agree with the encoding AE; the decoder was selected from the **inner header of
the received bytes**, and a counting zstd codec confirms exactly one
decompression per receive. The two compressed sizes are single-frame diagnostics
on an untrained AE and are **not** a payload measurement.

## Batch-1 latency (CUDA events, 20 warm-ups, 100 measured)

| stage | median | p95 |
| --- | --: | --: |
| AE128 encoder (UE side) | **0.0731 ms** | **0.0749 ms** |
| AE128 decoder (edge side) | **0.1239 ms** | **0.1275 ms** |

Dense all-one keep mask; the decoder's input latent is dense at every q, so its
cost does not vary with q. GPU compute only — no host transfer, quantization or
zstd is included.

## Source defect found and corrected

One real defect, exposed by exactly this qualification:

**Both deployable receive paths could not use a device-resident decoder.**
`ae_uint8_transport.decode_sparse` rebuilds the latent from host bytes and
therefore always returns it on CPU. `PreloadedAeDecoders.receive` and
`ae_uint8_transport.reconstruct_c2` then handed that CPU latent straight to the
selected decoder, so a CUDA-resident AE — i.e. the only way the edge would
actually run — failed with
`Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should
be the same`. Phase 9A never saw this because both its tests are CPU-only.

Fix: a new `ae_uint8_transport.decoder_device()` reads the device off the
selected decoder's own parameters, and both receive paths move the decoded
latent and mask onto it. No wire format, header, quantization, q semantics or
frozen module changed. Covered by one new case in the **existing**
`tests/test_ae_transport.py` family (accelerator-gated), which asserts the
received C2 is byte-identical to decoding the same latent by hand on the device
and that `reconstruct_c2` agrees.

All 82 tests pass: the frozen hybrid-q suites (`test_synthetic`,
`test_continuous_q`, `test_uint8_transport`) and the AE suites
(`test_ae_model`, `test_ae_transport`).

## Terminal

```
SPLITFUSION_AE128_GPU_QUALIFIED_AWAITING_TRAINING_REVIEW
```

Every gate in the runner is fail-closed: a violated gate raises and no terminal
file is written, so the terminal marker exists only for a completely clean run.
