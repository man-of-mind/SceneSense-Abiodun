# SplitFusion-FCOS AE128/AE64/AE32 — Phase 9A implementation review

Implementation and interfaces only. Nothing here loaded a checkpoint, ran CUDA,
read a dataset or cache, trained, inferred, validated, evaluated or launched
CARLA. Every number below is analytical or a parameter count; **payload,
accuracy and latency remain unmeasured.**

## Scope and frozen surface

One shared, channel-parameterized `SplitFeatureAE` serves all three families;
only the latent channel count B in {128, 64, 32} changes. Spatial resolution is
exactly 112x192 on both sides of the bottleneck — the AE compresses channels and
never resamples.

```
C2 [256,112,192]
    |- frozen ranker scores the original FP32 C2
    '- AE encoder -> latent [B,112,192]
                  -> ranker spatial keep mask applied to the latent
                  -> per-channel UINT8
                  -> mandatory zstd (existing level-1 codec, unchanged)
====================== network ======================
zstd decode -> UINT8 dequantize -> latent zero-scatter
            -> AE decoder (latent + reconstructed mask)
            -> reconstructed FP32 C2 -> unchanged frozen perception tail
```

Nothing frozen was modified. `git diff` against `69bad51` shows no tracked file
changed except the pre-existing dirty `OAI/openairinterface5g` submodule, which
was left exactly as found. The frozen SplitFusion source/checkpoint, the stable
epoch-4 ranker, `ranker.py`, `selection.py`, `codec.py`, `guards.py`,
`continuous_q.py`, the validated noAE UINT8 codec and its Phase-8B results,
service thresholds/scoring/AVO and the zstd level-1 implementation are imported
and reused, never edited. Zstd level 1 versus level 3 is deliberately not
touched here; it is a separate later measurement.

## Package contents

`pole_lraspp_multimodal_fusion/object_head_pilot_v1/splitfusion_fcos_r50_fpn_p2_p7_ae_v1/`

| File | Role |
| --- | --- |
| `ae_contract.py` | AE constants, family ids, and fail-closed validators (reuses the frozen contract and guard error classes) |
| `ae_model.py` | `SplitFeatureAE` and `build_split_feature_ae` |
| `ae_composition.py` | latent mask / composition helper (ranker + encoder + drop ordering) |
| `ae_loss.py` | task-aware reconstruction loss and the trainer-facing masking/schedule/ownership interfaces |
| `ae_uint8_transport.py` | AE-latent UINT8 wire plus the mandatory zstd wrapper |
| `ae_family_dispatch.py` | minimal per-packet decoder-selection adapter (added per the runtime-switching clarification) |
| `tests/test_ae_model.py`, `tests/test_ae_transport.py` | the two CPU synthetic checks |

## Architecture

Lightweight, asymmetric, no BatchNorm anywhere.

```
encoder:  p = Conv2d(256 -> B, 1x1)(C2)
          z = p + depthwise_3x3(GELU(p))

decoder:  x = Conv2d(B+1 -> 256, 1x1)(concat(z, mask))
          C2_hat = x + depthwise_3x3(GELU(x))
```

The reconstructed binary keep mask is concatenated as **one additional decoder
input channel**, always the last one (index B), so the first B channels stay in
latent order. This is not a new side channel: the sparse transport header
already carries that bitmask, so the decoder is only told what it can read off
the wire. At q=0 the mask is all ones (`keep_mask=None` is accepted and means
exactly that).

`SplitFeatureAE.decode(latent, keep_mask)` takes no C2 argument, so **no skip
connection containing original C2 can cross the transport boundary** — it is
unrepresentable, not merely discouraged.

### Initialization

Construction is deterministic and RNG-neutral: layer construction *and*
initialization run inside `torch.random.fork_rng`, seeded by
`AE_INIT_BASE_SEED (20260829) + B`, so families are deterministically separated
and the caller's global RNG stream is left exactly where it was.

- encoder projection: orthonormal rows (`W W^T = I_B`);
- decoder latent weights: exactly `W^T`;
- decoder mask weights: zero;
- both residual depthwise kernels: zero;
- all biases: zero.

At initialization the AE is therefore the rank-B orthogonal channel projection
`W^T W` applied per cell — a stable starting point — while both spatial-context
branches contribute exactly nothing until trained. Note this is *not* an
identity, which is the point: channel compression is lossy by construction and
**no q=0 identity is claimed anywhere.**

## Static accounting

Parameters (biases included) and MACs at 112x192 = 21,504 cells (biases excluded
from MACs; the mask channel is counted in the decoder's 1x1).

| B | encoder params | decoder params | total params | encoder MACs | decoder MACs | total MACs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 34,176 | 35,840 | 70,016 | 729,415,680 | 759,693,312 | 1,489,108,992 |
| 64 | 17,088 | 19,456 | 36,544 | 364,707,840 | 407,371,776 | 772,079,616 |
| 32 | 8,544 | 11,264 | 19,808 | 182,353,920 | 231,211,008 | 413,564,928 |

Closed forms: `encoder_params = 267B`, `decoder_params = 256(B+1) + 2816`,
`encoder_MACs = cells * B * 265`, `decoder_MACs = cells * 256 * (B+10)`. These
are arithmetic, not timings: **no latency was measured.**

## Composition rules (enforced in `ae_composition.compose`)

1. The ranker always scores the **original FP32 C2** object; it never sees the
   latent, at any q.
2. The AE encoder runs on the complete frame, **before** spatial dropping.
3. Because the ranker is AE-independent, one q-independent cell ordering induces
   the keep set for every family: AE128, AE64 and AE32 transport exactly the
   same cells at the same q (checked in the transport test).
4. All B latent channels of a retained cell stay together (cell-major wire).
5. q=0 bypasses the ranker but **not** the AE.
6. Arbitrary q in [0, 0.98] at 1e-4 resolution is mechanically supported by
   reusing the frozen `continuous_q` grid unchanged. Executability is not
   measured accuracy; q=0.90 and q=0.98 remain evaluation/emergency values.

## AE UINT8 wire

A **separate** codec from the validated noAE `uint8_codec` (magic `HQ8\0`,
codec id 1, 256 channels), which is unchanged. The AE wire is magic `AE8\0`,
codec id 2, and accepts only B in {128, 64, 32}.

Fixed 50-byte little-endian header, `"<4sHHHIIIIIIIIQ"`:

| Offset | Bytes | Field | Meaning |
| ---: | ---: | --- | --- |
| 0 | 4 | magic | `AE8\0` |
| 4 | 2 | version | 1 |
| 6 | 2 | codec id | 2 (`ae_latent_uint8`) |
| 8 | 2 | family id | 1=AE128, 2=AE64, 3=AE32 (0=noAE never valid here) |
| 10 | 4 | checkpoint binding | 32-bit provenance word; 0 = unbound |
| 14 | 4 | B | latent channels, must match the family id |
| 18 | 4 | H | 112 |
| 22 | 4 | W | 192 |
| 26 | 4 | q_e4 | q in ten-thousandths |
| 30 | 4 | keep count | must equal the frozen keep formula at that q |
| 34 | 4 | mask bytes | 0 at q=0, else 2,688 |
| 38 | 4 | range bytes | 8B |
| 42 | 8 | value bytes | keep * B |

Then: the existing MSB-first spatial bitmask (q>0 only, unchanged
`_pack_bitmask`/`_unpack_bitmask`), B little-endian FP32 min/max pairs in
ascending channel order, and the UINT8 value block.

- Ranges are computed **once per frame from the complete latent, before q**, and
  reused for every q of that frame, so a retained cell quantizes to the same
  code at every q and a q sweep is a pure subset relation on the value block.
- Only retained latent cells are quantized; dropped cells are never encoded.
- Deterministic value order, cell-major: retained cells in ascending row-major
  cell index, and within one cell the B latent channels in ascending channel
  index.
- Decoding reconstructs a dense FP32 latent with **exact zeros** at dropped
  cells and returns the reconstructed keep mask for the decoder.
- Constant channels (span <= 1e-12) encode as code 0 and decode to the channel
  minimum; the epsilon is imported from the noAE codec so the two cannot drift.
- Rejected fail-closed: bad magic/version/codec identity, unregistered or
  disagreeing family id, unregistered B, wrong H/W, off-grid or out-of-range q,
  wrong keep count, wrong mask/range/value block lengths, bitmask popcount
  disagreement, set padding bits, non-finite or reversed ranges, truncation and
  trailing bytes.
- The complete payload always passes through the existing mandatory zstd
  wrapper: `encode`/`encode_frame` return an `AeZstdPacket` and the decoder
  accepts only that type, so the wrapper cannot be skipped. Level 1 is
  unchanged and untuned.

### Analytical pre-zstd payload sizes

Exact byte counts (header + mask + ranges + values); the noAE UINT8 column is
the existing validated wire at the same q, for orientation only.

| q | keep cells | AE128 | AE64 | AE32 | noAE UINT8 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 21,504 | 2,753,586 | 1,376,818 | 688,434 | 5,507,116 |
| 0.30 | 15,053 | 1,930,546 | 966,642 | 484,690 | 3,858,348 |
| 0.50 | 10,752 | 1,380,018 | 691,378 | 347,058 | 2,757,292 |
| 0.70 | 6,451 | 829,490 | 416,114 | 209,426 | 1,656,236 |
| 0.90 | 2,150 | 278,962 | 140,850 | 71,794 | 555,180 |
| 0.98 | 430 | 58,802 | 30,770 | 16,754 | 114,860 |

These are **pre-zstd analytical sizes**. The compressed size on the wire is not
predicted by them and has not been measured.

## Per-frame model-family provenance

Added per the runtime-switching clarification, as an interface/provenance
safeguard only. Phase 9A implements **no** model loading, no live switching and
no new registry, and does not duplicate the existing profile/model framework or
copy the frozen tail.

The existing noAE envelope does not carry an explicit profile/family id — it is
identified only implicitly by magic `HQ8\0` + codec id 1 + 256 channels — so the
family id is carried explicitly in the new AE header instead of being inferred
from mutable global state. `ae_family_dispatch.identify_sparse_frame` maps both
wires onto one `WireFamily` (noAE = family 0) without re-framing the frozen one.

`PreloadedAeDecoders` is a read-only view over AE pairs the existing preloaded
mechanism already constructed. `select_for_packet` picks the decoder that the
*individual packet* declares, and `require_family_agreement` fails closed unless
three facts agree: family id, transported latent channel count, and the
registered checkpoint binding. A packet delayed or reordered across a profile
switch is therefore refused rather than reconstructed by the wrong AE. Phase 9A
loads no checkpoint, so `checkpoint_binding` defaults to 0 (unbound);
`bind_checkpoint` and `checkpoint_binding_from_sha256` are pure interface hooks
that touch no file.

Training does not use any of this: a training run selects one bottleneck family
explicitly and keeps it.

## Training interfaces (implemented, not executed)

`ae_loss.training_protocol()` returns the locked intended later protocol as
data; it is also stated in the module docstring:

- only AE parameters trainable; frozen perception and the stable ranker stay in
  eval mode with `requires_grad=False`, enforced by
  `require_ae_only_optimizer` / `require_frozen_companions`;
- fit episodes for optimization, reserved train-holdout for selection;
- Stage A: dense q=0 reconstruction;
- Stage B: balanced batch-level round robin over q = {0, .30, .50, .70}, one q
  per batch (`stage_b_q_for_update`);
- q=.90/.98 excluded from optimization (`require_optimization_q` refuses them)
  and retained for later evaluation;
- no fake quantization and no zstd inside training — the loss refuses non-FP32
  inputs — so one AE checkpoint can later serve UINT8/UINT6/UINT4;
- ranker masks are hard and detached (`ae_composition.detached_hard_mask`); no
  gradient can enter the ranker.

### Task-aware reconstruction loss

Two unit-weighted, scale-free components, each reported separately, over the
existing Phase-4 D/G/S/A importance-map interface (`TeacherMapResult` or a plain
group mapping both accepted):

```
e(h,w)     = sum_c (C2_hat - C2)^2          g(h,w) = sum_c C2^2
plain      = sum e / sum g
group_t    = sum_hw I_t * e / sum_hw I_t * g
importance = mean over valid t of group_t          (equal weight)
total      = plain + importance
```

An unavailable, non-finite, negative, zero-mass or zero-reference-energy map is
ignored with a recorded reason; **at least three valid groups are required** or
the call fails closed. No raw multitask detection/segmentation loss weight is
introduced: the task maps enter only as spatial weightings of the same
reconstruction error.

## What was run

Exactly two CPU synthetic tests, plus `py_compile` on every module and a `git
diff` check against `69bad51`.

1. `tests/test_ae_model.py` — for B in {128, 64, 32}: RNG-neutral and repeatable
   construction, per-family seed separation, orthonormal encoder rows,
   transposed decoder latent weights, zero mask/depthwise/bias initialization,
   exact unbatched and batched tensor shapes, the rank-B projection identity at
   init (and explicitly *not* an identity map), keep-mask input behaviour
   including the `None`/all-one equivalence and malformed-mask rejection, closed-
   form parameter and MAC counts, finite forward/backward with no gradient
   reaching the ranker, AE-only optimizer ownership (and rejection of an
   intruding parameter), unregistered-family rejection, the <3-valid-group
   failure, the non-FP32 loss-input refusal, and the Stage-B q cycle.
2. `tests/test_ae_transport.py` — end to end at q=0, q=0.50 (registered) and
   q=0.2345 (arbitrary), for all three families: the ranker receives the
   original FP32 C2 object, q=0 bypasses it while the AE still runs, exact keep
   count and mask, family-invariant keep indices, ranges taken from the complete
   pre-q latent, byte-exact zstd round trip against independently rebuilt sparse
   bytes, analytical size agreement, bounded UINT8 retained error
   (<= span/510 + FP32 slack), exact zero-scatter, correct [256,112,192] decoder
   output, non-identity at q=0, per-packet decoder selection, wrong-family and
   wrong-binding refusal, and 17 malformed-payload rejections.

Both pass, as do the frozen package's existing `test_uint8_transport` and
`test_continuous_q` (4 tests) — the frozen wire is unaffected.

```
Ran 2 tests in 1.649s   OK      # AE package
Ran 4 tests in 1.237s   OK      # frozen hybrid-q regression
```

## Explicitly unmeasured

**Payload, accuracy and latency remain unmeasured.** No AE was trained, no
checkpoint exists, no zstd ratio was measured on AE bytes, no detection,
segmentation, geometry or dense-depth metric was computed at any q, and no
encoder/decoder timing was taken on any device. The zstd level-1 versus level-3
comparison is a separate later measurement over the trained families. The
analytical tables above bound the wire format only, and the initialization is a
starting point, not a result.
