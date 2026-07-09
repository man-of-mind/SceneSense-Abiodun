# Loopback latency / reliability sweep (M', CARLA transport)

Real split-inference transport metrics per quant x entropy profile. `delivery_rate` = fraction of frames whose result returned within the timeout (loopback reliability = payload/fragmentation-driven; true channel loss arrives with OAI). This establishes the **payload -> {latency, reliability}** curve; ROI/AE configs move along it by their (offline-measured) payload.

| profile | quant | entropy | payload KB | front ms | RTT ms | delivery | frames |
|---|---|---|--:|--:|--:|--:|--:|
| q_pchan_u4_zstd | per_channel_uint4 | zstd | 366.1 | 28.0 | 12.5 | 1.0 | 300 |
| q_pchan_u4_zlib | per_channel_uint4 | zlib | 385.3 | 35.6 | 21.7 | 1.0 | 300 |
| q_pchan_u4_none | per_channel_uint4 | none | 717.2 | 27.7 | 15.2 | 0.24 | 300 |
| q_pchan_u6_zstd | per_channel_uint6 | zstd | 730.9 | 32.0 | 18.2 | 0.317 | 300 |
| q_pchan_u6_zlib | per_channel_uint6 | zlib | 761.0 | 47.7 | 39.9 | 0.117 | 300 |
| q_pchan_u8_zstd | per_channel_uint8 | zstd | 982.9 | 31.6 | 21.5 | 0.107 | 300 |
| q_pchan_u8_zlib | per_channel_uint8 | zlib | 1016.4 | 50.5 | 46.6 | 0.11 | 300 |
| q_pchan_u8_none | per_channel_uint8 | none | 1426.0 | 28.2 | 16.6 | 0.127 | 300 |
