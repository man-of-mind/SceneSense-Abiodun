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
| `ae_composition.py` | latent mask / composition helper, single frame (`compose`) and batched training (`compose_batch`) |
| `ae_loss.py` | task-aware reconstruction loss over the Phase-4 teacher cache, plus the trainer-facing masking/schedule/ownership interfaces |
| `ae_uint8_transport.py` | AE-latent UINT8 wire plus the mandatory zstd wrapper |
| `ae_family_dispatch.py` | minimal per-frame decoder-selection adapter over the received wire bytes |
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
   In the batched training form, `compose_batch` encodes the whole
   `[N,256,112,192]` batch, runs the ranker once on the batch under `no_grad`,
   and then takes a **stable top-K independently per frame**, stacking the
   per-frame masks into `[N,112,192]`. Candidates are never flattened across the
   batch and no single global top-K set is ever formed, so every frame keeps
   exactly `keep_count` cells and no frame can spend another frame's budget. One
   q applies to the whole batch, taken from the locked Stage-B cycle.
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
| 10 | 4 | routing tag | 32-bit routing discriminator; 0 (unbound) is refused |
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
  disagreeing family id, an unbound routing tag, unregistered B, wrong H/W,
  off-grid or out-of-range q,
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

An interface and provenance safeguard only. Phase 9A implements **no** model
loading, no live switching and no new registry, and does not duplicate the
existing profile/model framework or copy the frozen tail.

The existing noAE envelope does not carry an explicit profile/family id — it is
identified only implicitly by magic `HQ8\0` + codec id 1 + 256 channels — so the
family id is carried explicitly in the new AE header instead of being inferred
from mutable global state. `ae_family_dispatch.identify_sparse_frame` maps both
wires onto one `WireFamily` (noAE = family 0) without re-framing the frozen one.

**Dispatch runs on the received bytes.** Only the compressed zstd byte string is
guaranteed to cross the network, so Python dataclass fields are not treated as
wire provenance. `PreloadedAeDecoders.receive(frame_bytes)` is the deployable
edge entry point: it takes the raw compressed bytes, decompresses them **exactly
once**, inspects the authoritative inner AE header, selects the already-preloaded
decoder from the header's family id / latent channel count / routing tag,
dequantizes and zero-scatters from the bytes it already has (no second zstd
pass), and reconstructs C2 with exactly that decoder. An `AeZstdPacket` may be
supplied as an optional local cross-check via `expected_packet`, but it is never
required to discover the decoder. Any disagreement fails closed, so a frame
delayed or reordered across a profile switch is refused rather than
reconstructed by the wrong AE.

The 32-bit field is a **routing tag**, not a cryptographic checkpoint identity —
32 truncated bits cannot authenticate a checkpoint, and the authoritative full
SHA-256 remains bound by the profile registry. Its only job is to route a frame
back to the AE that produced it. `routing_tag_from_sha256` derives one from a
digest the caller already holds (no file access). An unbound tag (0) is refused
everywhere it matters: `encode_frame`/`encode` will not emit one, `inspect`
rejects one on the wire, `require_family_agreement` rejects an unbound selected
decoder, and `PreloadedAeDecoders` refuses to register an unbound pair at all.
A freshly constructed AE is unbound until `bind_routing_tag` is called.

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
- ranker masks are hard and detached (`ae_composition.detached_hard_mask`,
  `ae_composition.compose_batch`); no gradient can enter the ranker.

### Task-aware reconstruction loss

Supervision is consumed **exactly as the Phase-4 teacher cache stores it**. A
shard holds, per frame, one combined FP32 `importance` map plus `valid_groups`
and `excluded_groups`; it does **not** hold four separate D/G/S/A maps.
`CachedTeacherBatch.from_shard(payload, offsets)` slices a loaded shard into a
training batch reading only those three fields, and `from_teacher_maps` builds
the same object from in-memory `TeacherMapResult`s, which is the representation
the cache writer itself consumes. No cache is built or rebuilt.

Two unit-weighted, scale-free components, each reported separately:

```
e(h,w)     = sum_c (C2_hat - C2)^2          g(h,w) = sum_c C2^2
plain      = sum e / sum g
combined   = sum_hw I * e / sum_hw I * g
total      = plain + combined
```

`I` is the cached combined map (already L1-normalized per frame by the Phase-4
producer; re-normalized here defensively, a no-op on a well-formed entry). Both
ratios are taken over the whole batch so the two components are normalized
consistently.

**Every** frame in the batch must carry at least three valid D/G/S/A groups
according to the cached `valid_groups`, or the call fails closed naming the thin
batch positions. Group availability and exclusion reasons are reported as
metadata only — there is deliberately **no per-group reconstruction term**,
because there is no per-group map to compute one from, and fabricating one would
invent supervision the cache does not contain. No raw multitask
detection/segmentation loss weight is introduced: the cached map enters only as a
spatial weighting of the same reconstruction error.

## What was run

Exactly two CPU synthetic tests, plus `py_compile` on every module and a `git
diff` check against `69bad51`.

1. `tests/test_ae_model.py` — for B in {128, 64, 32}: RNG-neutral and repeatable
   construction, per-family seed separation, orthonormal encoder rows,
   transposed decoder latent weights, zero mask/depthwise/bias initialization,
   exact unbatched and batched tensor shapes, the rank-B projection identity at
   init (and explicitly *not* an identity map), keep-mask input behaviour
   including the `None`/all-one equivalence and malformed-mask rejection, closed-
   form parameter and MAC counts, batched composition at q=0 and q=0.50 with
   equal per-frame keep counts under a ranker whose frames occupy disjoint score
   ranges (a global top-K would have starved frame 0), exact per-frame
   zero-scatter, Stage-B q enforcement, finite forward/backward with no gradient
   reaching the ranker, AE-only optimizer ownership (and rejection of an
   intruding parameter), unregistered-family rejection, shard-style
   `CachedTeacherBatch` construction and slicing, rejection of a shard missing a
   stored field or naming an unregistered group, the <3-valid-group per-frame
   failure, refusal of raw per-group maps, the non-FP32 loss-input refusal, and
   the Stage-B q cycle.
2. `tests/test_ae_transport.py` — end to end at q=0, q=0.50 (registered) and
   q=0.2345 (arbitrary), for all three families: the ranker receives the
   original FP32 C2 object, q=0 bypasses it while the AE still runs, exact keep
   count and mask, family-invariant keep indices, ranges taken from the complete
   pre-q latent, byte-exact zstd round trip against independently rebuilt sparse
   bytes, analytical size agreement, bounded UINT8 retained error
   (<= span/510 + FP32 slack), exact zero-scatter, correct [256,112,192] decoder
   output, non-identity at q=0, dispatch from the raw compressed bytes alone
   with exactly one decompression and a result identical to the packet path,
   refusal of an unbound routing tag on encode and on registration, wrong-family
   and wrong-tag refusal, disagreeing optional cross-check refusal, and 18
   malformed-payload rejections routed through the byte path.

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
