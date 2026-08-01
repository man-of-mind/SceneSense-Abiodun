# Loopback latency sweep (M') — IDEAL transport

Split-inference latency per profile under an **ideal local transport** (8 MB socket buffers via net.core.rmem_max/wmem_max; NO bandwidth cap / no Linux tc shaping). Columns: front (UE compute), back (edge compute), transport (localhost round-trip), RTT (total). delivery ~1.0 by design here — REAL reliability under bandwidth/RF-loss is the OAI + Sionna phase (Month 3).

| profile | quant | entropy | payload KB | front ms | back ms | transport ms | RTT ms | delivery | frames |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|
| ae_b32_u4_roi0.98_zstd | per_channel_uint4 | zstd | 8.2 | 24.5 | 10.3 | 1.3 | 11.6 | 1.0 | 300 |
| ae_b64_u4_roi0.98_zstd | per_channel_uint4 | zstd | 11.1 | 25.3 | 13.3 | 1.5 | 14.7 | 1.0 | 300 |
| ae_b128_u4_roi0.98_zstd | per_channel_uint4 | zstd | 16.6 | 24.9 | 11.2 | 1.5 | 12.7 | 1.0 | 300 |
| ae_b32_u4_roi0.9_zstd | per_channel_uint4 | zstd | 18.6 | 25.1 | 11.7 | 1.5 | 13.2 | 1.0 | 300 |
| ae_b64_u4_roi0.9_zstd | per_channel_uint4 | zstd | 23.3 | 25.7 | 13.4 | 1.6 | 15.0 | 1.0 | 300 |
| ae_b128_u4_roi0.9_zstd | per_channel_uint4 | zstd | 32.2 | 25.6 | 12.3 | 1.6 | 13.9 | 1.0 | 300 |
| ae_b32_u4_roi0.7_zstd | per_channel_uint4 | zstd | 38.1 | 24.6 | 12.4 | 1.6 | 14.0 | 1.0 | 300 |
| noae_u4_roi0.98_zstd | per_channel_uint4 | zstd | 38.2 | 25.8 | 15.9 | 2.3 | 18.3 | 1.0 | 300 |
| ae_b64_u4_roi0.7_zstd | per_channel_uint4 | zstd | 46.9 | 26.1 | 11.8 | 1.8 | 13.6 | 1.0 | 300 |
| ae_b128_u4_roi0.7_zstd | per_channel_uint4 | zstd | 61.9 | 24.5 | 12.4 | 2.0 | 14.4 | 1.0 | 300 |
| noae_u4_roi0.9_zstd | per_channel_uint4 | zstd | 84.4 | 30.7 | 13.3 | 2.9 | 16.2 | 1.0 | 300 |
| noae_u4_roi0.7_zstd | per_channel_uint4 | zstd | 171.8 | 33.9 | 12.3 | 4.1 | 16.4 | 1.0 | 300 |
