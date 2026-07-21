# PER-MODEL KNOB MATRIX — **latency = zstd (reference codec (deployed = zlib)), ALL 36 profiles MEASURED** (M', Month-2)

> ✅ **Codec + provenance (2026-07-20).** Latency (front/back/transport) measured live with `--entropy-coder zstd`
> under ideal 8 MB-buffer loopback, **100% delivery on all 36 AE×quant×ROI profiles**
> (`loopback_latency_zstd.json`, batch `sweeps_loopback_ideal_zstd_full`). **No interpolation** except the
> synthetic `uncompressed_fp16` anchor. Accuracy + payload from the per-model offline eval
> (`sweeps_permodel_zstd`, also zstd) — genuinely zstd-measured, nothing copied or flagged. Accuracy is
> codec-invariant (lossless); payload is zstd's own (compression ratios differ ~±5% between codecs). Counterpart:
> `PERMODEL_KNOB_MATRIX_ZLIB.md`; A/B: `CODEC_LATENCY_AB.md`; grouped: `PERMODEL_KNOB_MATRIX_ZSTD_BYMODEL.md`.

Action profiles vs **accuracy**, **payload** (entropy-coded bytes), and **latency** (front=UE compute, back=edge compute, transport=localhost round-trip; **zstd, measured**). Transport is an **IDEAL local link** (8 MB socket buffers, NO bandwidth cap / no Linux tc shaping). **Reliability + latency under a real channel = OAI + Sionna, Month 3.**

Clean baseline: **ae128__clean** payload=2835.0KB mIoU=0.819 ped-recall=0.883  (accept tol = 2%)

| profile | quant | entropy | ROI q | AE | payload KB | payload % | mIoU | veh IoU | ped recall | obj recall | loc m | ped-loc m | front ms | back ms | transport ms | accept |
|---|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|
| ae32__uint4__roi0.5 | per_channel_uint4 | zstd | 0.50 | 32 | 49.4 | 2% | 0.656 | 0.459 | 0.852 | 0.895 | 0.93 | 1.13 | 27.0 | 12.7 | 1.8 | - |
| ae64__uint4__roi0.5 | per_channel_uint4 | zstd | 0.50 | 64 | 61.3 | 2% | 0.702 | 0.574 | 0.871 | 0.901 | 0.93 | 1.14 | 24.6 | 11.6 | 1.9 | - |
| ae32__uint4__roi0.3 | per_channel_uint4 | zstd | 0.30 | 32 | 64.1 | 2% | 0.715 | 0.633 | 0.860 | 0.899 | 0.88 | 1.06 | 26.5 | 11.9 | 1.8 | - |
| ae64__uint4__roi0.3 | per_channel_uint4 | zstd | 0.30 | 64 | 75.2 | 3% | 0.739 | 0.672 | 0.862 | 0.898 | 0.87 | 1.06 | 25.8 | 12.5 | 2.1 | - |
| ae128__uint4__roi0.5 | per_channel_uint4 | zstd | 0.50 | 128 | 79.2 | 3% | 0.621 | 0.382 | 0.878 | 0.908 | 0.94 | 1.13 | 28.0 | 12.9 | 2.3 | - |
| ae32__uint4__roi0.0 | per_channel_uint4 | zstd | 0.00 | 32 | 90.0 | 3% | 0.822 | 0.915 | 0.860 | 0.900 | 0.88 | 1.06 | 24.7 | 12.1 | 2.0 | - |
| ae32__uint6__roi0.5 | per_channel_uint6 | zstd | 0.50 | 32 | 94.2 | 3% | 0.759 | 0.740 | 0.850 | 0.895 | 0.92 | 1.10 | 29.1 | 13.0 | 2.3 | - |
| ae128__uint4__roi0.3 | per_channel_uint4 | zstd | 0.30 | 128 | 96.4 | 3% | 0.684 | 0.536 | 0.886 | 0.910 | 0.88 | 1.08 | 25.3 | 11.5 | 2.2 | - |
| ae64__uint4__roi0.0 | per_channel_uint4 | zstd | 0.00 | 64 | 101.1 | 4% | 0.824 | 0.915 | 0.862 | 0.898 | 0.88 | 1.07 | 22.5 | 15.5 | 2.3 | - |
| ae64__uint6__roi0.5 | per_channel_uint6 | zstd | 0.50 | 64 | 117.9 | 4% | 0.781 | 0.792 | 0.867 | 0.897 | 0.91 | 1.11 | 26.2 | 13.3 | 2.4 | - |
| ae32__uint8__roi0.5 | per_channel_uint8 | zstd | 0.50 | 32 | 122.7 | 4% | 0.773 | 0.781 | 0.848 | 0.895 | 0.92 | 1.10 | 27.4 | 11.9 | 2.3 | - |
| ae32__uint6__roi0.3 | per_channel_uint6 | zstd | 0.30 | 32 | 124.4 | 4% | 0.786 | 0.808 | 0.863 | 0.902 | 0.88 | 1.06 | 26.7 | 13.2 | 2.0 | - |
| ae128__uint4__roi0.0 | per_channel_uint4 | zstd | 0.00 | 128 | 129.2 | 5% | 0.819 | 0.913 | 0.887 | 0.910 | 0.88 | 1.07 | 26.8 | 16.0 | 2.6 | Y |
| ae64__uint6__roi0.3 | per_channel_uint6 | zstd | 0.30 | 64 | 148.6 | 5% | 0.795 | 0.828 | 0.864 | 0.901 | 0.87 | 1.04 | 28.6 | 12.7 | 2.2 | - |
| ae64__uint8__roi0.5 | per_channel_uint8 | zstd | 0.50 | 64 | 153.9 | 5% | 0.793 | 0.825 | 0.869 | 0.897 | 0.91 | 1.11 | 25.8 | 12.4 | 2.2 | - |
| ae128__uint6__roi0.5 | per_channel_uint6 | zstd | 0.50 | 128 | 154.6 | 5% | 0.714 | 0.627 | 0.881 | 0.910 | 0.94 | 1.15 | 28.0 | 13.4 | 2.5 | - |
| ae32__uint8__roi0.3 | per_channel_uint8 | zstd | 0.30 | 32 | 163.4 | 6% | 0.799 | 0.846 | 0.864 | 0.902 | 0.88 | 1.06 | 28.9 | 13.0 | 2.1 | - |
| ae32__uint6__roi0.0 | per_channel_uint6 | zstd | 0.00 | 32 | 174.7 | 6% | 0.822 | 0.916 | 0.865 | 0.902 | 0.88 | 1.06 | 23.9 | 12.4 | 2.0 | Y |
| ae128__uint6__roi0.3 | per_channel_uint6 | zstd | 0.30 | 128 | 192.8 | 7% | 0.756 | 0.734 | 0.883 | 0.909 | 0.87 | 1.05 | 25.6 | 13.5 | 2.2 | - |
| ae64__uint8__roi0.3 | per_channel_uint8 | zstd | 0.30 | 64 | 195.7 | 7% | 0.805 | 0.856 | 0.864 | 0.899 | 0.86 | 1.03 | 25.6 | 12.7 | 2.4 | Y |
| ae128__uint8__roi0.5 | per_channel_uint8 | zstd | 0.50 | 128 | 200.6 | 7% | 0.730 | 0.674 | 0.881 | 0.909 | 0.94 | 1.14 | 24.6 | 12.5 | 2.8 | - |
| ae64__uint6__roi0.0 | per_channel_uint6 | zstd | 0.00 | 64 | 203.1 | 7% | 0.825 | 0.916 | 0.866 | 0.901 | 0.87 | 1.04 | 23.3 | 16.0 | 2.1 | Y |
| ae32__uint8__roi0.0 | per_channel_uint8 | zstd | 0.00 | 32 | 228.6 | 8% | 0.822 | 0.916 | 0.863 | 0.902 | 0.88 | 1.06 | 24.8 | 14.2 | 2.0 | - |
| noae__uint4__roi0.5 | per_channel_uint4 | zstd | 0.50 | - | 239.4 | 8% | 0.613 | 0.363 | 0.817 | 0.860 | 1.11 | 1.29 | 28.9 | 11.8 | 4.6 | - |
| ae128__uint8__roi0.3 | per_channel_uint8 | zstd | 0.30 | 128 | 251.8 | 9% | 0.759 | 0.744 | 0.883 | 0.908 | 0.87 | 1.05 | 25.6 | 12.2 | 2.6 | - |
| ae128__uint6__roi0.0 | per_channel_uint6 | zstd | 0.00 | 128 | 260.9 | 9% | 0.819 | 0.914 | 0.884 | 0.909 | 0.87 | 1.05 | 23.2 | 13.8 | 2.3 | Y |
| ae64__uint8__roi0.0 | per_channel_uint8 | zstd | 0.00 | 64 | 267.7 | 9% | 0.825 | 0.916 | 0.864 | 0.900 | 0.87 | 1.04 | 22.9 | 13.2 | 2.1 | Y |
| noae__uint4__roi0.3 | per_channel_uint4 | zstd | 0.30 | - | 289.0 | 10% | 0.719 | 0.593 | 0.851 | 0.879 | 1.04 | 1.22 | 28.5 | 15.8 | 4.8 | - |
| ae128__uint8__roi0.0 | per_channel_uint8 | zstd | 0.00 | 128 | 341.0 | 12% | 0.819 | 0.914 | 0.883 | 0.908 | 0.87 | 1.05 | 24.5 | 14.0 | 2.5 | Y |
| noae__uint4__roi0.0 | per_channel_uint4 | zstd | 0.00 | - | 392.0 | 14% | 0.838 | 0.931 | 0.843 | 0.875 | 1.05 | 1.22 | 28.0 | 13.4 | 5.7 | - |
| noae__uint6__roi0.5 | per_channel_uint6 | zstd | 0.50 | - | 452.2 | 16% | 0.774 | 0.766 | 0.832 | 0.867 | 1.04 | 1.19 | 34.0 | 13.2 | 5.9 | - |
| noae__uint6__roi0.3 | per_channel_uint6 | zstd | 0.30 | - | 565.6 | 20% | 0.797 | 0.808 | 0.853 | 0.877 | 0.96 | 1.10 | 32.2 | 12.3 | 6.2 | - |
| noae__uint8__roi0.5 | per_channel_uint8 | zstd | 0.50 | - | 598.1 | 21% | 0.782 | 0.789 | 0.838 | 0.868 | 1.04 | 1.19 | 29.0 | 10.6 | 7.5 | - |
| noae__uint8__roi0.3 | per_channel_uint8 | zstd | 0.30 | - | 759.7 | 27% | 0.803 | 0.825 | 0.853 | 0.878 | 0.95 | 1.08 | 32.2 | 12.1 | 9.4 | - |
| noae__uint6__roi0.0 | per_channel_uint6 | zstd | 0.00 | - | 784.8 | 28% | 0.840 | 0.931 | 0.852 | 0.878 | 0.96 | 1.09 | 34.6 | 13.9 | 7.3 | - |
| noae__uint8__roi0.0 | per_channel_uint8 | zstd | 0.00 | - | 1050.3 | 37% | 0.840 | 0.931 | 0.855 | 0.879 | 0.95 | 1.08 | 27.8 | 10.6 | 8.7 | - |
| fp16_zstd_lossless | none(fp16) | zstd | 0.00 | - | 2216.0 | 78% | 0.819 | 0.914 | 0.883 | 0.908 | 0.87 | 1.06 | ~27.8 | ~10.6 | ~8.7 | Y |
| uncompressed_fp16 | none(fp16) | - | 0.00 | - | 2835.0 | 100% | 0.819 | 0.914 | 0.883 | 0.908 | 0.87 | 1.06 | ~27.8 | ~10.6 | ~8.7 | Y |
| ae128__clean | none | zstd | 0.00 | - | nan | nan% | 0.819 | 0.914 | 0.883 | 0.908 | 0.87 | 1.06 | - | - | - | Y |
| ae32__clean | none | zstd | 0.00 | - | nan | nan% | 0.822 | 0.916 | 0.863 | 0.902 | 0.88 | 1.06 | - | - | - | - |
| ae64__clean | none | zstd | 0.00 | - | nan | nan% | 0.825 | 0.916 | 0.864 | 0.900 | 0.87 | 1.04 | - | - | - | Y |
| noae__clean | none | zstd | 0.00 | - | nan | nan% | 0.839 | 0.931 | 0.853 | 0.878 | 0.95 | 1.08 | - | - | - | - |

## Pareto pick (min payload within accuracy tolerance)
**ae128__uint4__roi0.0** — payload 129.2KB (5% of clean), mIoU 0.819, ped-recall 0.887, loc 0.88m.

## For the RL controller
- This table is the offline action-cost model: each row is a discrete action, columns are the reward terms (task utility) and the payload/latency/reliability cost.
- **front ms / RTT ms / delivery** are measured on the loopback (CARLA transport) for the pure quant x entropy profiles; `~` marks ROI/AE profiles whose latency/reliability follow the same **payload -> {latency, reliability}** curve (see LOOPBACK_LATENCY.md / loopback_latency_zstd.json) via their payload column.
- Loopback delivery reflects payload/fragmentation; TRUE channel loss + variable latency arrive with the OAI/Sionna network phase, which replaces the loopback transport column.
