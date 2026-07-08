# Static sweep — payload + front-latency by compression profile

Deterministic aggregation of the quant×entropy sweep (loopback). Accuracy-vs-compression is a separate offline eval (deterministic, human-validated). `frames_with_result` shows how often the loopback result returned (low = the ~1MB payload fragments over UDP; why live accuracy is unreliable).

| variant | quant | entropy | payload KB (comp) | KB (uncomp) | compression× | front_ms | frames | w/result |
|---|---|---|---|---|---|---|---|---|
| q_pchan_u4_none | per_channel_uint4 | none | 717.2 | 2835.0 | 3.95 | 26.7 | 300 | 91 |
| q_pchan_u4_zlib | per_channel_uint4 | zlib | 381.4 | 2835.0 | 7.43 | 34.9 | 300 | 300 |
| q_pchan_u4_zstd | per_channel_uint4 | zstd | 359.4 | 2835.0 | 7.89 | 27.8 | 300 | 300 |
| q_pchan_u6_none | per_channel_uint6 | none | 1071.6 | 2835.0 | 2.65 | 29.6 | 300 | 47 |
| q_pchan_u6_zlib | per_channel_uint6 | zlib | 758.9 | 2835.0 | 3.74 | 48.1 | 300 | 36 |
| q_pchan_u6_zstd | per_channel_uint6 | zstd | 727.0 | 2835.0 | 3.9 | 33.4 | 300 | 86 |
| q_pchan_u8_none | per_channel_uint8 | none | 1426.0 | 2835.0 | 1.99 | 27.6 | 300 | 32 |
| q_pchan_u8_zlib | per_channel_uint8 | zlib | 1025.3 | 2835.0 | 2.77 | 50.5 | 300 | 27 |
| q_pchan_u8_zstd | per_channel_uint8 | zstd | 985.9 | 2835.0 | 2.88 | 31.8 | 300 | 45 |
| q_ptensor_u8_none | per_tensor_uint8 | none | 1418.2 | 2835.0 | 2.0 | 28.7 | 300 | 24 |
| q_ptensor_u8_zlib | per_tensor_uint8 | zlib | 805.5 | 2835.0 | 3.52 | 46.2 | 300 | 38 |
