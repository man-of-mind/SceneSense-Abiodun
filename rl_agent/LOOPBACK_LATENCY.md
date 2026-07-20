# Loopback latency sweep (M') — IDEAL transport

> ⚠️ **Codec matters (2026-07-20).** The `transport ms` column bundles receive-side reassembly **+ decompression**,
> and the entropy codec dominates it at large payloads. Compare the last two rows: `q_u8_zstd` (982 KB → **7.4 ms**)
> vs `q_u8_zlib` (1016 KB → **31.0 ms**) — same payload, **zlib ~4× slower than zstd** both directions (front too:
> 29.8 vs 47.5 ms). All other rows here are **zstd**. The **deployed pipeline defaults to zlib**, so the knob-matrix
> latency built from the mostly-zstd `loopback_latency_zstd.json` under-predicts live no-AE latency by ~4×. The
> zlib-measured counterpart is `loopback_latency_zlib.json` / `PERMODEL_KNOB_MATRIX_ZLIB.md`. Accuracy + payload are
> codec-invariant (lossless).

Split-inference latency per profile under an **ideal local transport** (8 MB socket buffers via net.core.rmem_max/wmem_max; NO bandwidth cap / no Linux tc shaping). Columns: front (UE compute), back (edge compute), transport (localhost round-trip), RTT (total). delivery ~1.0 by design here — REAL reliability under bandwidth/RF-loss is the OAI + Sionna phase (Month 3).

| profile | quant | entropy | payload KB | front ms | back ms | transport ms | RTT ms | delivery | frames |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|
| ae_b64_u8_zstd | per_channel_uint8 | zstd | 269.2 | 24.9 | 11.0 | 2.0 | 13.0 | 1.0 | 300 |
| ae_b128_u8_zstd | per_channel_uint8 | zstd | 342.6 | 25.5 | 9.3 | 2.2 | 11.5 | 1.0 | 300 |
| q_u4_zstd | per_channel_uint4 | zstd | 367.2 | 27.6 | 9.2 | 4.6 | 13.9 | 1.0 | 300 |
| roi0.5_u8_zstd | per_channel_uint8 | zstd | 555.3 | 30.5 | 7.6 | 6.2 | 13.8 | 1.0 | 300 |
| roi0.3_u8_zstd | per_channel_uint8 | zstd | 717.5 | 31.0 | 7.8 | 6.8 | 14.6 | 1.0 | 300 |
| q_u6_zstd | per_channel_uint6 | zstd | 730.3 | 31.2 | 11.0 | 6.3 | 17.3 | 1.0 | 300 |
| roi0.1_u8_zstd | per_channel_uint8 | zstd | 893.9 | 31.4 | 7.7 | 7.4 | 15.0 | 1.0 | 300 |
| q_u8_zstd | per_channel_uint8 | zstd | 982.1 | 29.8 | 7.6 | 7.4 | 15.0 | 1.0 | 300 |
| q_u8_zlib | per_channel_uint8 | zlib | 1016.5 | 47.5 | 9.9 | 31.0 | 40.9 | 1.0 | 300 |
