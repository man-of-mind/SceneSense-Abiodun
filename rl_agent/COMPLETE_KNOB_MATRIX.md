# COMPLETE KNOB MATRIX (M', offline: accuracy + payload)

Action profiles vs task accuracy and on-wire payload (entropy-coded bytes). Latency / delivery-rate / reliability under channel = OAI/network phase (not here).

Clean baseline: **clean_noquant** payload=1425.3KB mIoU=0.841 ped-recall=0.787  (accept tol = 2%)

| profile | quant | entropy | ROI q | AE | payload KB | payload % | mIoU | veh IoU | ped recall | obj recall | loc m | ped-loc m | front ms | RTT ms | delivery | accept |
|---|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|
| ae_b32_roi0.3 | per_channel_uint8 | zlib | 0.30 | 32 | 161.4 | 11% | 0.797 | 0.828 | 0.052 | 0.023 | 3.00 | 3.09 | ~28.0 | - | ~1.000 | - |
| ae_b64_roi0.3 | per_channel_uint8 | zlib | 0.30 | 64 | 189.0 | 13% | 0.802 | 0.835 | 0.282 | 0.133 | 2.93 | 3.04 | ~28.0 | - | ~1.000 | - |
| ae_b32 | per_channel_uint8 | zlib | 0.00 | 32 | 225.8 | 16% | 0.831 | 0.927 | 0.052 | 0.023 | 3.00 | 3.09 | ~28.0 | - | ~1.000 | - |
| ae_b64 | per_channel_uint8 | zlib | 0.00 | 64 | 264.0 | 19% | 0.834 | 0.928 | 0.278 | 0.132 | 2.92 | 3.02 | ~28.0 | - | ~1.000 | - |
| ae_b128 | per_channel_uint8 | zlib | 0.00 | 128 | 337.7 | 24% | 0.834 | 0.928 | 0.222 | 0.226 | 2.50 | 2.77 | ~28.0 | - | ~1.000 | - |
| quant_uint4_zlib | per_channel_uint4 | zlib | 0.00 | - | 358.6 | 25% | 0.840 | 0.932 | 0.755 | 0.817 | 1.32 | 1.56 | 35.6 | 21.7 | 1.000 | - |
| quant_uint4_zstd | per_channel_uint4 | zstd | 0.00 | - | 364.7 | 26% | 0.840 | 0.932 | 0.755 | 0.817 | 1.32 | 1.56 | 28.0 | 12.5 | 1.000 | - |
| roi_0.7 | per_channel_uint8 | zlib | 0.70 | - | 380.6 | 27% | 0.779 | 0.779 | 0.778 | 0.831 | 1.30 | 1.49 | ~33.8 | - | ~1.000 | - |
| roi_0.5 | per_channel_uint8 | zlib | 0.50 | - | 556.6 | 39% | 0.797 | 0.822 | 0.778 | 0.832 | 1.30 | 1.49 | ~31.5 | - | ~0.608 | - |
| roi_0.3 | per_channel_uint8 | zlib | 0.30 | - | 714.4 | 50% | 0.811 | 0.845 | 0.790 | 0.837 | 1.21 | 1.39 | ~27.8 | - | ~0.246 | - |
| quant_uint4_none | per_channel_uint4 | none | 0.00 | - | 716.6 | 50% | 0.840 | 0.932 | 0.755 | 0.817 | 1.32 | 1.56 | 27.7 | 15.2 | 0.240 | - |
| quant_uint6_zlib | per_channel_uint6 | zlib | 0.00 | - | 730.0 | 51% | 0.841 | 0.933 | 0.790 | 0.837 | 1.22 | 1.41 | 47.7 | 39.9 | 0.117 | Y |
| quant_uint6_zstd | per_channel_uint6 | zstd | 0.00 | - | 733.1 | 51% | 0.841 | 0.933 | 0.790 | 0.837 | 1.22 | 1.41 | 32.0 | 18.2 | 0.317 | Y |
| roi_0.1 | per_channel_uint8 | zlib | 0.10 | - | 898.5 | 63% | 0.833 | 0.909 | 0.789 | 0.836 | 1.21 | 1.38 | ~37.7 | - | ~0.111 | Y |
| quant_uint8_zlib | per_channel_uint8 | zlib | 0.00 | - | 992.2 | 70% | 0.841 | 0.933 | 0.790 | 0.837 | 1.21 | 1.39 | 50.5 | 46.6 | 0.110 | Y |
| quant_uint8_zstd | per_channel_uint8 | zstd | 0.00 | - | 992.8 | 70% | 0.841 | 0.933 | 0.790 | 0.837 | 1.21 | 1.39 | 31.6 | 21.5 | 0.107 | Y |
| quant_uint6_none | per_channel_uint6 | none | 0.00 | - | 1071.0 | 75% | 0.841 | 0.933 | 0.790 | 0.837 | 1.22 | 1.41 | ~47.5 | - | ~0.112 | Y |
| quant_uint8_none | per_channel_uint8 | none | 0.00 | - | 1425.3 | 100% | 0.841 | 0.933 | 0.790 | 0.837 | 1.21 | 1.39 | 28.2 | 16.6 | 0.127 | Y |
| clean_noquant | none | zlib | 0.00 | - | nan | nan% | 0.841 | 0.933 | 0.787 | 0.835 | 1.21 | 1.39 | - | - | - | Y |

## Pareto pick (min payload within accuracy tolerance)
**quant_uint6_zlib** — payload 730.0KB (51% of clean), mIoU 0.841, ped-recall 0.790, loc 1.22m.

## For the RL controller
- This table is the offline action-cost model: each row is a discrete action, columns are the reward terms (task utility) and the payload/latency/reliability cost.
- **front ms / RTT ms / delivery** are measured on the loopback (CARLA transport) for the pure quant x entropy profiles; `~` marks ROI/AE profiles whose latency/reliability follow the same **payload -> {latency, reliability}** curve (see LOOPBACK_LATENCY.md) via their payload column.
- Loopback delivery reflects payload/fragmentation; TRUE channel loss + variable latency arrive with the OAI/Sionna network phase, which replaces the loopback transport column.
