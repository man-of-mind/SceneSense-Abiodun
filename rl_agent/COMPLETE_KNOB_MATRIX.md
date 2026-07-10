# COMPLETE KNOB MATRIX (M', Month-2 static knobs)

Action profiles vs **accuracy**, **payload** (entropy-coded bytes), and **latency** (front=UE compute, back=edge compute, transport=localhost round-trip). Transport is an **IDEAL local link** (8 MB socket buffers, NO bandwidth cap / no Linux tc shaping), so delivery is ~100% and not a differentiator here. **Reliability + latency under a real channel (bandwidth, RF loss) = OAI + Sionna, Month 3.** `~` latency = interpolated from the measured payload->latency curve (loopback client runs quant x entropy natively; ROI/AE latency inferred by payload).

Clean baseline: **clean_noquant** payload=2835.0KB mIoU=0.841 ped-recall=0.787  (accept tol = 2%)

| profile | quant | entropy | ROI q | AE | payload KB | payload % | mIoU | veh IoU | ped recall | obj recall | loc m | ped-loc m | front ms | back ms | transport ms | accept |
|---|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|
| comb_ae32_u4 | per_channel_uint4 | zstd | 0.00 | 32 | 87.7 | 3% | 0.836 | 0.931 | 0.488 | 0.422 | 2.12 | 2.30 | ~24.9 | ~11.0 | ~2.0 | - |
| comb_ae64_u4 | per_channel_uint4 | zstd | 0.00 | 64 | 97.0 | 3% | 0.835 | 0.931 | 0.535 | 0.460 | 2.16 | 2.30 | ~24.9 | ~11.0 | ~2.0 | - |
| comb_ae64_roi0.5 | per_channel_uint8 | zstd | 0.50 | 64 | 153.2 | 5% | 0.785 | 0.838 | 0.231 | 0.192 | 2.23 | 2.67 | ~24.9 | ~11.0 | ~2.0 | - |
| comb_ae32_roi0.3 | per_channel_uint8 | zstd | 0.30 | 32 | 165.2 | 6% | 0.809 | 0.851 | 0.496 | 0.430 | 2.11 | 2.27 | ~24.9 | ~11.0 | ~2.0 | - |
| comb_ae64_roi0.3 | per_channel_uint8 | zstd | 0.30 | 64 | 194.8 | 7% | 0.808 | 0.850 | 0.551 | 0.469 | 2.15 | 2.28 | ~24.9 | ~11.0 | ~2.0 | - |
| comb_u4_roi0.5 | per_channel_uint4 | zstd | 0.50 | - | 226.6 | 8% | 0.756 | 0.712 | 0.732 | 0.802 | 1.40 | 1.68 | ~24.9 | ~11.0 | ~2.0 | - |
| ae_v2clean_b32 | per_channel_uint8 | zstd | 0.00 | 32 | 231.2 | 8% | 0.837 | 0.932 | 0.496 | 0.429 | 2.11 | 2.27 | ~24.9 | ~11.0 | ~2.0 | - |
| comb_ae128_roi0.3 | per_channel_uint8 | zstd | 0.30 | 128 | 252.0 | 9% | 0.807 | 0.848 | 0.540 | 0.461 | 2.13 | 2.38 | ~24.9 | ~11.0 | ~2.0 | - |
| ae_v2clean_b64 | per_channel_uint8 | zlib | 0.00 | 64 | 259.8 | 9% | 0.836 | 0.932 | 0.552 | 0.471 | 2.15 | 2.29 | 24.9 | 11.0 | 2.0 | - |
| comb_u4_roi0.3 | per_channel_uint4 | zstd | 0.30 | - | 271.8 | 10% | 0.801 | 0.820 | 0.760 | 0.820 | 1.33 | 1.57 | ~24.9 | ~10.9 | ~2.0 | - |
| ae_v2clean_b128 | per_channel_uint8 | zlib | 0.00 | 128 | 333.2 | 12% | 0.836 | 0.932 | 0.539 | 0.468 | 2.15 | 2.39 | 25.5 | 9.3 | 2.2 | - |
| quant_uint4_zlib | per_channel_uint4 | zlib | 0.00 | - | 358.6 | 13% | 0.840 | 0.932 | 0.755 | 0.817 | 1.32 | 1.56 | 27.6 | 9.2 | 4.6 | - |
| quant_uint4_zstd | per_channel_uint4 | zstd | 0.00 | - | 364.7 | 13% | 0.840 | 0.932 | 0.755 | 0.817 | 1.32 | 1.56 | 27.6 | 9.2 | 4.6 | - |
| roi_0.7 | per_channel_uint8 | zlib | 0.70 | - | 380.6 | 13% | 0.779 | 0.779 | 0.778 | 0.831 | 1.30 | 1.49 | ~27.8 | ~9.1 | ~4.7 | - |
| roi_0.5 | per_channel_uint8 | zlib | 0.50 | - | 556.6 | 20% | 0.797 | 0.822 | 0.778 | 0.832 | 1.30 | 1.49 | 30.5 | 7.6 | 6.2 | - |
| roi_0.3 | per_channel_uint8 | zlib | 0.30 | - | 714.4 | 25% | 0.811 | 0.845 | 0.790 | 0.837 | 1.21 | 1.39 | 31.0 | 7.8 | 6.8 | - |
| quant_uint4_none | per_channel_uint4 | none | 0.00 | - | 716.6 | 25% | 0.840 | 0.932 | 0.755 | 0.817 | 1.32 | 1.56 | 27.6 | 9.2 | 4.6 | - |
| quant_uint6_zlib | per_channel_uint6 | zlib | 0.00 | - | 730.0 | 26% | 0.841 | 0.933 | 0.790 | 0.837 | 1.22 | 1.41 | 31.2 | 11.0 | 6.3 | Y |
| quant_uint6_zstd | per_channel_uint6 | zstd | 0.00 | - | 733.1 | 26% | 0.841 | 0.933 | 0.790 | 0.837 | 1.22 | 1.41 | 31.2 | 11.0 | 6.3 | Y |
| roi_0.1 | per_channel_uint8 | zlib | 0.10 | - | 898.5 | 32% | 0.833 | 0.909 | 0.789 | 0.836 | 1.21 | 1.38 | 31.4 | 7.7 | 7.4 | Y |
| quant_uint8_zlib | per_channel_uint8 | zlib | 0.00 | - | 992.2 | 35% | 0.841 | 0.933 | 0.790 | 0.837 | 1.21 | 1.39 | 29.8 | 7.6 | 7.4 | Y |
| quant_uint8_zstd | per_channel_uint8 | zstd | 0.00 | - | 992.8 | 35% | 0.841 | 0.933 | 0.790 | 0.837 | 1.21 | 1.39 | 29.8 | 7.6 | 7.4 | Y |
| quant_uint6_none | per_channel_uint6 | none | 0.00 | - | 1071.0 | 38% | 0.841 | 0.933 | 0.790 | 0.837 | 1.22 | 1.41 | 31.2 | 11.0 | 6.3 | Y |
| quant_uint8_none | per_channel_uint8 | none | 0.00 | - | 1425.3 | 50% | 0.841 | 0.933 | 0.790 | 0.837 | 1.21 | 1.39 | 29.8 | 7.6 | 7.4 | Y |
| fp16_zstd_lossless | none(fp16) | zstd | 0.00 | - | 2216.0 | 78% | 0.841 | 0.933 | 0.787 | 0.835 | 1.21 | 1.39 | ~29.8 | ~7.6 | ~7.4 | Y |
| uncompressed_fp16 | none(fp16) | - | 0.00 | - | 2835.0 | 100% | 0.841 | 0.933 | 0.787 | 0.835 | 1.21 | 1.39 | ~29.8 | ~7.6 | ~7.4 | Y |
| clean_noquant | none | zlib | 0.00 | - | nan | nan% | 0.841 | 0.933 | 0.787 | 0.835 | 1.21 | 1.39 | - | - | - | Y |

## Pareto pick (min payload within accuracy tolerance)
**quant_uint6_zlib** — payload 730.0KB (26% of clean), mIoU 0.841, ped-recall 0.790, loc 1.22m.

## For the RL controller
- This table is the offline action-cost model: each row is a discrete action, columns are the reward terms (task utility) and the payload/latency/reliability cost.
- **front ms / RTT ms / delivery** are measured on the loopback (CARLA transport) for the pure quant x entropy profiles; `~` marks ROI/AE profiles whose latency/reliability follow the same **payload -> {latency, reliability}** curve (see LOOPBACK_LATENCY.md) via their payload column.
- Loopback delivery reflects payload/fragmentation; TRUE channel loss + variable latency arrive with the OAI/Sionna network phase, which replaces the loopback transport column.
