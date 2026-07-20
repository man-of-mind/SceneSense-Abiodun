# Loopback latency sweep (M') — IDEAL transport

Split-inference latency per profile under an **ideal local transport** (8 MB socket buffers via net.core.rmem_max/wmem_max; NO bandwidth cap / no Linux tc shaping). Columns: front (UE compute), back (edge compute), transport (localhost round-trip), RTT (total). delivery ~1.0 by design here — REAL reliability under bandwidth/RF-loss is the OAI + Sionna phase (Month 3).

| profile | quant | entropy | payload KB | front ms | back ms | transport ms | RTT ms | delivery | frames |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|
| ae_b32_u4_roi0.5_zlib | per_channel_uint4 | zlib | 56.4 | 27.5 | 8.8 | 3.6 | 12.4 | 1.0 | 300 |
| ae_b64_u4_roi0.5_zlib | per_channel_uint4 | zlib | 65.3 | 28.0 | 11.7 | 4.0 | 15.7 | 1.0 | 300 |
| ae_b32_u4_roi0.3_zlib | per_channel_uint4 | zlib | 67.0 | 27.4 | 8.6 | 3.9 | 12.5 | 1.0 | 300 |
| ae_b64_u4_roi0.3_zlib | per_channel_uint4 | zlib | 75.2 | 27.6 | 9.6 | 4.2 | 13.7 | 1.0 | 300 |
| ae_b128_u4_roi0.5_zlib | per_channel_uint4 | zlib | 80.5 | 27.8 | 8.4 | 4.5 | 12.9 | 1.0 | 300 |
| ae_b32_u4_roi0.0_zlib | per_channel_uint4 | zlib | 88.3 | 26.6 | 10.5 | 4.6 | 15.1 | 1.0 | 300 |
| ae_b128_u4_roi0.3_zlib | per_channel_uint4 | zlib | 93.7 | 28.5 | 9.4 | 4.7 | 14.1 | 1.0 | 300 |
| ae_b64_u4_roi0.0_zlib | per_channel_uint4 | zlib | 99.3 | 26.8 | 10.4 | 4.9 | 15.3 | 1.0 | 300 |
| ae_b32_u6_roi0.5_zlib | per_channel_uint6 | zlib | 101.8 | 28.5 | 8.7 | 4.6 | 13.2 | 1.0 | 300 |
| ae_b64_u6_roi0.5_zlib | per_channel_uint6 | zlib | 117.9 | 29.0 | 9.3 | 5.0 | 14.4 | 1.0 | 300 |
| ae_b128_u4_roi0.0_zlib | per_channel_uint4 | zlib | 123.0 | 26.7 | 10.9 | 5.6 | 16.5 | 1.0 | 300 |
| ae_b32_u6_roi0.3_zlib | per_channel_uint6 | zlib | 127.3 | 29.6 | 8.4 | 5.2 | 13.6 | 1.0 | 300 |
| ae_b32_u8_roi0.5_zlib | per_channel_uint8 | zlib | 127.9 | 29.2 | 7.9 | 5.3 | 13.2 | 1.0 | 300 |
| ae_b64_u6_roi0.3_zlib | per_channel_uint6 | zlib | 145.1 | 29.9 | 8.1 | 5.6 | 13.7 | 1.0 | 300 |
| ae_b64_u8_roi0.5_zlib | per_channel_uint8 | zlib | 154.5 | 29.1 | 8.6 | 6.1 | 14.7 | 1.0 | 300 |
| ae_b128_u6_roi0.5_zlib | per_channel_uint6 | zlib | 156.1 | 30.3 | 8.5 | 6.3 | 14.8 | 1.0 | 300 |
| ae_b32_u8_roi0.3_zlib | per_channel_uint8 | zlib | 164.8 | 29.6 | 8.2 | 6.0 | 14.3 | 1.0 | 300 |
| ae_b32_u6_roi0.0_zlib | per_channel_uint6 | zlib | 171.0 | 27.7 | 9.8 | 6.3 | 16.1 | 1.0 | 300 |
| ae_b128_u6_roi0.3_zlib | per_channel_uint6 | zlib | 186.3 | 30.9 | 8.8 | 6.9 | 15.7 | 1.0 | 300 |
| ae_b64_u8_roi0.3_zlib | per_channel_uint8 | zlib | 190.6 | 30.7 | 9.7 | 7.0 | 16.7 | 1.0 | 300 |
| ae_b64_u6_roi0.0_zlib | per_channel_uint6 | zlib | 195.4 | 28.6 | 8.2 | 7.0 | 15.1 | 1.0 | 300 |
| ae_b128_u8_roi0.5_zlib | per_channel_uint8 | zlib | 203.0 | 30.5 | 10.7 | 7.7 | 18.4 | 1.0 | 300 |
| ae_b32_u8_roi0.0_zlib | per_channel_uint8 | zlib | 224.7 | 28.7 | 7.9 | 7.4 | 15.3 | 1.0 | 300 |
| ae_b128_u6_roi0.0_zlib | per_channel_uint6 | zlib | 246.8 | 30.0 | 8.1 | 8.5 | 16.6 | 1.0 | 300 |
| ae_b128_u8_roi0.3_zlib | per_channel_uint8 | zlib | 247.7 | 30.9 | 11.7 | 8.6 | 20.4 | 1.0 | 300 |
| noae_u4_roi0.5_zlib | per_channel_uint4 | zlib | 250.4 | 33.7 | 7.4 | 10.6 | 18.0 | 1.0 | 300 |
| ae_b64_u8_roi0.0_zlib | per_channel_uint8 | zlib | 259.4 | 29.3 | 10.4 | 8.5 | 18.9 | 1.0 | 300 |
| noae_u4_roi0.3_zlib | per_channel_uint4 | zlib | 291.6 | 33.7 | 11.9 | 11.8 | 23.7 | 1.0 | 300 |
| ae_b128_u8_roi0.0_zlib | per_channel_uint8 | zlib | 331.5 | 30.8 | 10.1 | 10.4 | 20.5 | 1.0 | 300 |
| noae_u4_roi0.0_zlib | per_channel_uint4 | zlib | 385.9 | 35.2 | 9.6 | 14.6 | 24.2 | 1.0 | 300 |
| noae_u6_roi0.5_zlib | per_channel_uint6 | zlib | 454.0 | 39.6 | 10.1 | 16.8 | 26.9 | 1.0 | 300 |
| noae_u6_roi0.3_zlib | per_channel_uint6 | zlib | 564.7 | 42.5 | 9.2 | 19.5 | 28.8 | 1.0 | 300 |
| noae_u8_roi0.5_zlib | per_channel_uint8 | zlib | 579.2 | 39.4 | 8.3 | 20.0 | 28.3 | 1.0 | 300 |
| noae_u8_roi0.3_zlib | per_channel_uint8 | zlib | 745.3 | 43.6 | 8.1 | 23.8 | 31.9 | 1.0 | 300 |
| noae_u6_roi0.0_zlib | per_channel_uint6 | zlib | 760.3 | 44.8 | 10.9 | 24.8 | 35.7 | 1.0 | 300 |
| noae_u8_roi0.0_zlib | per_channel_uint8 | zlib | 1018.5 | 46.0 | 9.1 | 30.7 | 39.8 | 1.0 | 300 |
