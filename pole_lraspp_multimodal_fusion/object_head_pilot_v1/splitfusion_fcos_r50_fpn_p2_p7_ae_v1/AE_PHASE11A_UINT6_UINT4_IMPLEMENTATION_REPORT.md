# Phase 11A — shared UINT6/UINT4 transport implementation

## Outcome

The two missing quantizers for the 72-profile design are implemented as one
shared transport for `noAE`, `AE128`, `AE64`, and `AE32`. This phase performs no
model inference or scientific validation. The existing FP32 and UINT8 wires are
unchanged.

The completed Phase-10B AE64/AE32 UINT8 evidence was recorded first in commit
`c0d49ae`. Its original verdicts remain unchanged: neither family passed the
strict acceptance contract; every profile remains technically executable and
available to the later action catalog with its measured quality tier.

## Transport path

```text
original FP32 C2
  ├─ noAE: use C2 directly
  └─ AE: full-frame encoder -> B-channel latent
             │
             ├─ calculate per-channel min/max on the complete feature
             ├─ rank original FP32 C2 (q>0 only)
             ├─ retain the exact stable top-K cells
             ├─ affine quantize every retained channel to UINT6 or UINT4
             ├─ MSB-first bit-pack retained values
             ├─ frame identity + q + mask + FP32 ranges + values
             └─ mandatory existing zstd level 1

edge: one zstd decompression -> inspect header -> unpack/dequantize/scatter
  ├─ noAE: reconstructed C2 goes directly to the frozen tail
  └─ AE: header selects the already-loaded decoder -> reconstructed C2
```

The AE family is selected from the received inner header, not from current
controller state. A family/latent-width/routing-tag disagreement fails closed.
The noAE family carries family id 0, 256 channels, and routing tag 0; each AE
family requires its non-zero checkpoint-derived routing tag.

## Quantization and bit order

For channel `c`, with full-feature minimum `m_c` and maximum `M_c`:

```text
L = 2^bits - 1
code = round(clamp((x - m_c)/(M_c - m_c), 0, 1) * L)
reconstruction = m_c + code/L * (M_c - m_c)
```

Constant-span channels encode code zero and reconstruct their minimum. UINT4
places the first code in the high nibble. UINT6 places four consecutive codes
in three bytes, from the most significant bit downward. Retained cells remain
ascending row-major, with all transported channels of one cell adjacent. Any
unused low bits in the last byte must be zero.

## Wire schema

One 52-byte little-endian header (`<4sHHHHIIIIIIIIQ`) records:

```text
magic, version, codec id, bit width, family id, routing tag,
channels, height, width, q_e4, keep count,
mask bytes, range bytes, packed-value bytes
```

The body is `mask | FP32 channel ranges | packed values`. At q=0 the ranker is
bypassed and the mask block is absent, but quantization or AE reconstruction is
not bypassed. Every packet is one independent zstd frame.

## Exact analytical pre-zstd bytes

Columns are q = `0.00 / 0.30 / 0.50 / 0.70 / 0.90 / 0.98`.

| Family | UINT6 bytes | UINT4 bytes |
| --- | --- | --- |
| noAE | 4,130,868 / 2,894,964 / 2,069,172 / 1,243,380 / 417,588 / 87,348 | 2,754,612 / 1,931,572 / 1,381,044 / 830,516 / 279,988 / 59,828 |
| AE128 | 2,065,460 / 1,448,852 / 1,035,956 / 623,060 / 210,164 / 45,044 | 1,377,332 / 967,156 / 691,892 / 416,628 / 141,364 / 31,284 |
| AE64 | 1,032,756 / 725,796 / 519,348 / 312,900 / 106,452 / 23,892 | 688,692 / 484,948 / 347,316 / 209,684 / 72,052 / 17,012 |
| AE32 | 516,404 / 364,268 / 261,044 / 157,820 / 54,596 / 13,316 | 344,372 / 243,844 / 175,028 / 106,212 / 37,396 / 9,876 |

These are structural sizes before zstd, not measured network payloads. No
accuracy or latency conclusion follows from them.

## Files and verification scope

- `lowbit_transport.py`: shared quantization, packing, framing, zstd transmit,
  parsing, dequantization, and zero scatter.
- `lowbit_dispatch.py`: one-decompression family-aware receive path.
- `tests/test_lowbit_transport.py`: two focused CPU tests.

The checks cover known byte-exact UINT4/UINT6 packing, padding rejection,
analytical accounting for all four families, q=0 ranker bypass, stable q>0
selection, mandatory-zstd raw-byte dispatch, exact zero scatter, bounded affine
quantization error, family routing, and malformed bit-width rejection.

No checkpoint, dataset, cache, CUDA context, training, inference, validation,
test split, CARLA, threshold, scorer, model parameter, or existing codec was
accessed or changed.

## Next gate

Phase 11B should be a small GPU qualification using disposable frames and the
selected frozen checkpoints. Only after that passes should one shared Phase-11C
runner measure the 48 UINT6/UINT4 family/q profiles. That run must keep the
locked Phase-10B verdicts and thresholds unchanged and classify rather than
discard degraded profiles.
