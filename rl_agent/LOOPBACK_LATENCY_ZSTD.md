# Loopback latency sweep (M') — IDEAL transport

Split-inference latency per profile under an **ideal local transport** (8 MB socket buffers via net.core.rmem_max/wmem_max; NO bandwidth cap / no Linux tc shaping). Columns: front (UE compute), back (edge compute), transport (localhost round-trip), RTT (total). delivery ~1.0 by design here — REAL reliability under bandwidth/RF-loss is the OAI + Sionna phase (Month 3).

| profile | quant | entropy | payload KB | front ms | back ms | transport ms | RTT ms | delivery | frames |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|
| ae_b32_u4_roi0.5_zstd | per_channel_uint4 | zstd | 52.4 | 27.0 | 12.7 | 1.8 | 14.5 | 1.0 | 300 |
| ae_b64_u4_roi0.5_zstd | per_channel_uint4 | zstd | 61.0 | 24.6 | 11.6 | 1.9 | 13.5 | 1.0 | 300 |
| ae_b32_u4_roi0.3_zstd | per_channel_uint4 | zstd | 65.6 | 26.5 | 11.9 | 1.8 | 13.7 | 1.0 | 300 |
| ae_b64_u4_roi0.3_zstd | per_channel_uint4 | zstd | 73.7 | 25.8 | 12.5 | 2.1 | 14.6 | 1.0 | 300 |
| ae_b128_u4_roi0.5_zstd | per_channel_uint4 | zstd | 77.8 | 28.0 | 12.9 | 2.3 | 15.2 | 1.0 | 300 |
| ae_b32_u4_roi0.0_zstd | per_channel_uint4 | zstd | 89.2 | 24.7 | 12.1 | 2.0 | 14.2 | 1.0 | 300 |
| ae_b128_u4_roi0.3_zstd | per_channel_uint4 | zstd | 91.6 | 25.3 | 11.5 | 2.2 | 13.7 | 1.0 | 300 |
| ae_b64_u4_roi0.0_zstd | per_channel_uint4 | zstd | 98.9 | 22.5 | 15.5 | 2.3 | 17.8 | 1.0 | 300 |
| ae_b32_u6_roi0.5_zstd | per_channel_uint6 | zstd | 100.8 | 29.1 | 13.0 | 2.3 | 15.2 | 1.0 | 300 |
| ae_b64_u6_roi0.5_zstd | per_channel_uint6 | zstd | 119.6 | 26.2 | 13.3 | 2.4 | 15.7 | 1.0 | 300 |
| ae_b128_u4_roi0.0_zstd | per_channel_uint4 | zstd | 121.6 | 26.8 | 16.0 | 2.6 | 18.6 | 1.0 | 300 |
| ae_b32_u6_roi0.3_zstd | per_channel_uint6 | zstd | 125.8 | 26.7 | 13.2 | 2.0 | 15.2 | 1.0 | 300 |
| ae_b32_u8_roi0.5_zstd | per_channel_uint8 | zstd | 130.9 | 27.4 | 11.9 | 2.3 | 14.2 | 1.0 | 300 |
| ae_b64_u6_roi0.3_zstd | per_channel_uint6 | zstd | 145.3 | 28.6 | 12.7 | 2.2 | 15.0 | 1.0 | 300 |
| ae_b128_u6_roi0.5_zstd | per_channel_uint6 | zstd | 154.2 | 28.0 | 13.4 | 2.5 | 15.9 | 1.0 | 300 |
| ae_b64_u8_roi0.5_zstd | per_channel_uint8 | zstd | 155.8 | 25.8 | 12.4 | 2.2 | 14.6 | 1.0 | 300 |
| ae_b32_u8_roi0.3_zstd | per_channel_uint8 | zstd | 169.1 | 28.9 | 13.0 | 2.1 | 15.0 | 1.0 | 300 |
| ae_b32_u6_roi0.0_zstd | per_channel_uint6 | zstd | 174.3 | 23.9 | 12.4 | 2.0 | 14.4 | 1.0 | 300 |
| ae_b128_u6_roi0.3_zstd | per_channel_uint6 | zstd | 186.2 | 25.6 | 13.5 | 2.2 | 15.7 | 1.0 | 300 |
| ae_b64_u8_roi0.3_zstd | per_channel_uint8 | zstd | 196.9 | 25.6 | 12.7 | 2.4 | 15.1 | 1.0 | 300 |
| ae_b64_u6_roi0.0_zstd | per_channel_uint6 | zstd | 199.8 | 23.3 | 16.0 | 2.1 | 18.1 | 1.0 | 300 |
| ae_b128_u8_roi0.5_zstd | per_channel_uint8 | zstd | 207.1 | 24.6 | 12.5 | 2.8 | 15.4 | 1.0 | 300 |
| noae_u4_roi0.5_zstd | per_channel_uint4 | zstd | 221.1 | 28.9 | 11.8 | 4.6 | 16.4 | 1.0 | 300 |
| ae_b32_u8_roi0.0_zstd | per_channel_uint8 | zstd | 231.9 | 24.8 | 14.2 | 2.0 | 16.1 | 1.0 | 300 |
| ae_b128_u8_roi0.3_zstd | per_channel_uint8 | zstd | 252.4 | 25.6 | 12.2 | 2.6 | 14.8 | 1.0 | 300 |
| ae_b128_u6_roi0.0_zstd | per_channel_uint6 | zstd | 253.8 | 23.2 | 13.8 | 2.3 | 16.1 | 1.0 | 300 |
| ae_b64_u8_roi0.0_zstd | per_channel_uint8 | zstd | 269.2 | 22.9 | 13.2 | 2.1 | 15.3 | 1.0 | 300 |
| noae_u4_roi0.3_zstd | per_channel_uint4 | zstd | 271.6 | 28.5 | 15.8 | 4.8 | 20.6 | 1.0 | 300 |
| ae_b128_u8_roi0.0_zstd | per_channel_uint8 | zstd | 342.5 | 24.5 | 14.0 | 2.5 | 16.6 | 1.0 | 300 |
| noae_u4_roi0.0_zstd | per_channel_uint4 | zstd | 366.8 | 28.0 | 13.4 | 5.7 | 19.1 | 1.0 | 300 |
| noae_u6_roi0.5_zstd | per_channel_uint6 | zstd | 422.5 | 34.0 | 13.2 | 5.9 | 19.0 | 1.0 | 300 |
| noae_u6_roi0.3_zstd | per_channel_uint6 | zstd | 526.6 | 32.2 | 12.3 | 6.2 | 18.5 | 1.0 | 300 |
| noae_u8_roi0.5_zstd | per_channel_uint8 | zstd | 565.9 | 29.0 | 10.6 | 7.5 | 18.1 | 1.0 | 300 |
| noae_u8_roi0.3_zstd | per_channel_uint8 | zstd | 717.7 | 32.2 | 12.1 | 9.4 | 21.5 | 1.0 | 300 |
| noae_u6_roi0.0_zstd | per_channel_uint6 | zstd | 730.4 | 34.6 | 13.9 | 7.3 | 21.2 | 1.0 | 300 |
| noae_u8_roi0.0_zstd | per_channel_uint8 | zstd | 984.2 | 27.8 | 10.6 | 8.7 | 19.3 | 1.0 | 300 |
