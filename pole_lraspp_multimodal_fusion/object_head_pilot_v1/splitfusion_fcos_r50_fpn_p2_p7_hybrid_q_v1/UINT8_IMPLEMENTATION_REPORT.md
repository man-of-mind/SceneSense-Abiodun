# Hybrid-q per-channel UINT8 implementation record

This is an implementation-only CPU qualification of a separate noAE UINT8
wire. It does not make an accuracy, zstd-ratio, or deployment-latency claim.
The frozen perception and original FP32 sparse codec remain unchanged.

## Runtime pipeline and API

The deployable sequence is fixed:

```
original FP32 C2 [256,112,192]
  -> full-C2 channel ranges computed once
  -> stable ranker on that original FP32 C2 (q>0 only)
  -> existing continuous-q selection
  -> quantize retained values only
  -> UINT8 sparse frame
  -> mandatory Phase-7 zstd-1 frame
  -> zstd decode -> UINT8 dequantize -> spatial scatter -> dense FP32 C2
```

`uint8_codec.py` exposes:

- `prepare(c2) -> PreparedUint8Frame`: compute the 256 full-frame FP32
  min/max pairs once and retain the original FP32 C2.
- `encode(prepared, q, selection=None) -> Uint8SparsePayload`: accept continuous
  q at 1e-4 wire resolution and create diagnostic pre-zstd bytes. q=0 has no
  selection or mask but still quantizes all values.
- `inspect(payload) -> InspectedUint8Payload`: fail-closed parsing that exposes
  validated header, indices, ranges, and cell-major UINT8 values.
- `decode(payload) -> (dense_fp32_c2, q)`: affine dequantization followed by
  zero-filled spatial scatter.
- `analytical_size(q) -> AnalyticalPayloadSize`: exact pre-zstd accounting.

`uint8_zstd_transport.py` exposes:

- `prepare_frame(c2) -> PreparedUint8Frame`.
- `encode(prepared, ranker, q, wire_codec=None) -> Uint8ZstdTransport`: rank and
  select before quantization, then always call the unchanged Phase-7
  `ZstdWireCodec.compress` implementation.
- `decode(packet, wire_codec=None) -> (dense_fp32_c2, q)`: accept only a
  `Uint8ZstdPacket`, decompress first, then invoke the UINT8 decoder.
- `decompress_payload(packet, wire_codec=None) -> bytes`: diagnostic byte-exact
  zstd round-trip inspection; it is not an alternative deployable encoder.

There is no raw/deactivated-zstd mode in the transport wrapper. It reuses the
Phase-7 compressor implementation and its exact fixed configuration: level 1,
threads 0, no dictionary, frame checksum and content size enabled, dictionary
ID disabled.

## UINT8 wire schema

The fixed 44-byte little-endian header is `"<4sHHIIIIIIIQ"`:

| Offset | Bytes | Field | Required value/meaning |
| ---: | ---: | --- | --- |
| 0 | 4 | magic | `HQ8\0` |
| 4 | 2 | version | 1 |
| 6 | 2 | codec ID | 1 (`per_channel_uint8`) |
| 8 | 4 | C | 256 |
| 12 | 4 | H | 112 |
| 16 | 4 | W | 192 |
| 20 | 4 | q_e4 | exact continuous-q wire integer, 0 through 9800 |
| 24 | 4 | keep count | exact count implied by q_e4 |
| 28 | 4 | mask bytes | 0 at q=0; otherwise 2,688 |
| 32 | 4 | range bytes | exactly 2,048 |
| 36 | 8 | value bytes | keep count multiplied by 256 |

The body is, in order: the existing MSB-first mask for q>0; 256 interleaved
little-endian FP32 `[min, max]` pairs; then retained UINT8 values in ascending
cell order with all 256 channels contiguous per cell. The decoder rejects wrong
identity/version/dimensions/q/cardinality/block lengths, invalid or nonfinite
ranges, min greater than max, malformed masks, truncation, and trailing bytes.

For nonconstant channel `c`, encoding and decoding are:

```
u8 = round(clamp((x - min[c]) / (max[c] - min[c]), 0, 1) * 255)
x_hat = (u8 / 255) * (max[c] - min[c]) + min[c]
```

For span at most `1e-12`, every retained code is zero and decode returns the
stored channel minimum (equal to the maximum for an exactly constant channel).

## Exact analytical payload sizes before zstd

The ratio denominator is the existing framed FP32 q=0 payload, 22,020,140
bytes. The FP32-value column is analytical only; every UINT8 value block is
exactly one quarter of that corresponding FP32 value block (4.000x smaller).

| q | Keep | Header | Mask | Ranges | UINT8 values | Total | Total / framed FP32 q=0 | Corresponding FP32 values | Value-block reduction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 21,504 | 44 | 0 | 2,048 | 5,505,024 | 5,507,116 | 0.250094504 | 22,020,096 | 4.000x |
| 0.30 | 15,053 | 44 | 2,688 | 2,048 | 3,853,568 | 3,858,348 | 0.175219049 | 15,414,272 | 4.000x |
| 0.50 | 10,752 | 44 | 2,688 | 2,048 | 2,752,512 | 2,757,292 | 0.125216824 | 11,010,048 | 4.000x |
| 0.70 | 6,451 | 44 | 2,688 | 2,048 | 1,651,456 | 1,656,236 | 0.075214599 | 6,605,824 | 4.000x |
| 0.90 | 2,150 | 44 | 2,688 | 2,048 | 550,400 | 555,180 | 0.025212374 | 2,201,600 | 4.000x |
| 0.98 | 430 | 44 | 2,688 | 2,048 | 110,080 | 114,860 | 0.005216134 | 440,320 | 4.000x |

These are uncompressed framing facts only. No compressed-size or latency
measurement was performed.

## CPU checks and frozen bindings

The two focused synthetic methods cover quantization error and constant-channel
behavior; exact zero scatter; q=0 ranker bypass; registered and arbitrary q;
nested masks; shared-cell code identity; exact header q/counts; byte-exact zstd
round trip; and malformed metadata rejection.

Hash-only verification (no checkpoint deserialization) matched the Phase-7
record:

| Frozen binding | SHA-256 |
| --- | --- |
| `ranker.py` | `462536991f195651a1ee641f8e83444882ec370a8dffab72f13f0d770422b353` |
| `selection.py` | `ccc2b12919b078eac7af6131418989567d618d42f7b908b2db74df42e0342a71` |
| `codec.py` | `7b3833398a84fea31f65b86ec294c6675727390035b5761d372fd5a3cbba7b79` |
| `guards.py` | `77d8d8bfd168e74a7f0b6a7e3c8e7abc4c3549a86cda392feb394d7580a33031` |
| `continuous_q.py` | `8ea72faed324c29b7106bd5f6277699bd7e7ba16a66073905f9ff28c69bab23c` |
| `contract.py` | `748eb39c0913c9d4c449784d23b7ec752307ed47eafb1b0ef0ae25df4b4f3d4c` |
| `zstd_transport.py` | `57d1846b3fdc4084266e5a8adcc7abf99556ed2b187befec729003fcdb77edec` |
| `training.py` | `a10775a3a9f3e051e3456c6c85cdc76fff47f417e9bd24ddecb6633dcda1161a` |
| `teacher_cache.py` | `14f866d483c245f14078c6234a357427bccc4bfa1dd533b138677f1e17f579be` |
| `gpu_qualification.py` | `0032b9ac7edb89ec1b01616ea563bec73d04c20b164d6bbf8297f36d30eea7e6` |
| `__init__.py` | `9319a2b3ce9d492e1577908148ef31187ed648ef81944365dd907dee717b206c` |
| `locked_config.json` | `b2b0d8427bd867f46058ebba49ac6a183eb89413b4d69326fef93b150ebfcde6` |
| perception forward lock | `86d6f13ae9168b33b697df5b785c5f7c320afc52cfdcded5b632d94a6d943fe1` |
| stable ranker checkpoint | `07781c56a4c0f306f16d332f64627ce6b9458e154f40ab9fef89f89909b79cb5` |
| frozen perception checkpoint | `da14d21edbd374c1c3abce02ca4674b9f4097becfba9759aba945cea160a297f` |

The checks use the Phase-7 `/usr/bin/python3` runtime with CUDA hidden. They do
not load a model/checkpoint, execute a model forward, or access a data cache,
training, inference, validation, test, or CARLA path.
