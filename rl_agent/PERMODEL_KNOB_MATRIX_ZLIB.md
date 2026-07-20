# PER-MODEL KNOB MATRIX — **latency = zlib (DEPLOYED codec), ALL 36 profiles MEASURED** (M', Month-2)

> ✅ **Codec + provenance (2026-07-20).** Latency (front/back/transport) is measured live with `--entropy-coder zlib`
> — the deployed codec — under ideal 8 MB-buffer loopback, **100% delivery on all 36 AE×quant×ROI profiles**
> (`loopback_latency_zlib.json`, batch `sweeps_loopback_ideal_zlib_full`). **No interpolation** except the synthetic
> `uncompressed_fp16` 100%-payload anchor. **Use THIS matrix for the agent cost model.** Accuracy + payload come from
> the per-model offline eval (`sweeps_permodel`, also zlib) — genuinely zlib-measured, not copied. Accuracy is
> codec-invariant (lossless); payload is ~±5% codec-dependent. zstd counterpart: `PERMODEL_KNOB_MATRIX_ZSTD.md`
> (~4× lower transport at large payloads); A/B: `CODEC_LATENCY_AB.md`.

Action profiles vs **accuracy**, **payload** (entropy-coded bytes), and **latency** (front=UE compute, back=edge compute, transport=localhost round-trip; **zlib, deployed, measured**). Transport is an **IDEAL local link** (8 MB socket buffers, NO bandwidth cap / no Linux tc shaping), so delivery is ~100% and not a differentiator here. **Reliability + latency under a real channel (bandwidth, RF loss) = OAI + Sionna, Month 3.**

Clean baseline: **ae128__clean** payload=2835.0KB mIoU=0.819 ped-recall=0.883  (accept tol = 2%)

| profile | quant | entropy | ROI q | AE | payload KB | payload % | mIoU | veh IoU | ped recall | obj recall | loc m | ped-loc m | front ms | back ms | transport ms | accept |
|---|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|
| ae32__uint4__roi0.5 | per_channel_uint4 | zlib | 0.50 | 32 | 48.7 | 2% | 0.656 | 0.459 | 0.852 | 0.895 | 0.93 | 1.13 | 27.5 | 8.8 | 3.6 | - |
| ae64__uint4__roi0.5 | per_channel_uint4 | zlib | 0.50 | 64 | 60.0 | 2% | 0.702 | 0.574 | 0.871 | 0.901 | 0.93 | 1.14 | 28.0 | 11.7 | 4.0 | - |
| ae32__uint4__roi0.3 | per_channel_uint4 | zlib | 0.30 | 32 | 62.9 | 2% | 0.715 | 0.633 | 0.860 | 0.899 | 0.88 | 1.06 | 27.4 | 8.6 | 3.9 | - |
| ae64__uint4__roi0.3 | per_channel_uint4 | zlib | 0.30 | 64 | 73.9 | 3% | 0.739 | 0.672 | 0.862 | 0.898 | 0.87 | 1.06 | 27.6 | 9.6 | 4.2 | - |
| ae128__uint4__roi0.5 | per_channel_uint4 | zlib | 0.50 | 128 | 77.7 | 3% | 0.621 | 0.382 | 0.878 | 0.908 | 0.94 | 1.13 | 27.8 | 8.4 | 4.5 | - |
| ae32__uint4__roi0.0 | per_channel_uint4 | zlib | 0.00 | 32 | 87.5 | 3% | 0.822 | 0.915 | 0.860 | 0.900 | 0.88 | 1.06 | 26.6 | 10.5 | 4.6 | - |
| ae32__uint6__roi0.5 | per_channel_uint6 | zlib | 0.50 | 32 | 94.0 | 3% | 0.759 | 0.740 | 0.850 | 0.895 | 0.92 | 1.10 | 28.5 | 8.7 | 4.6 | - |
| ae128__uint4__roi0.3 | per_channel_uint4 | zlib | 0.30 | 128 | 95.0 | 3% | 0.684 | 0.536 | 0.886 | 0.910 | 0.88 | 1.08 | 28.5 | 9.4 | 4.7 | - |
| ae64__uint4__roi0.0 | per_channel_uint4 | zlib | 0.00 | 64 | 99.3 | 4% | 0.824 | 0.915 | 0.862 | 0.898 | 0.88 | 1.07 | 26.8 | 10.4 | 4.9 | - |
| ae64__uint6__roi0.5 | per_channel_uint6 | zlib | 0.50 | 64 | 116.9 | 4% | 0.781 | 0.792 | 0.867 | 0.897 | 0.91 | 1.11 | 29.0 | 9.3 | 5.0 | - |
| ae32__uint8__roi0.5 | per_channel_uint8 | zlib | 0.50 | 32 | 121.8 | 4% | 0.773 | 0.781 | 0.848 | 0.895 | 0.92 | 1.10 | 29.2 | 7.9 | 5.3 | - |
| ae32__uint6__roi0.3 | per_channel_uint6 | zlib | 0.30 | 32 | 123.6 | 4% | 0.786 | 0.808 | 0.863 | 0.902 | 0.88 | 1.06 | 29.6 | 8.4 | 5.2 | - |
| ae128__uint4__roi0.0 | per_channel_uint4 | zlib | 0.00 | 128 | 127.4 | 4% | 0.819 | 0.913 | 0.887 | 0.910 | 0.88 | 1.07 | 26.7 | 10.9 | 5.6 | Y |
| ae64__uint6__roi0.3 | per_channel_uint6 | zlib | 0.30 | 64 | 147.3 | 5% | 0.795 | 0.828 | 0.864 | 0.901 | 0.87 | 1.04 | 29.9 | 8.1 | 5.6 | - |
| ae64__uint8__roi0.5 | per_channel_uint8 | zlib | 0.50 | 64 | 152.1 | 5% | 0.793 | 0.825 | 0.869 | 0.897 | 0.91 | 1.11 | 29.1 | 8.6 | 6.1 | - |
| ae128__uint6__roi0.5 | per_channel_uint6 | zlib | 0.50 | 128 | 153.2 | 5% | 0.714 | 0.627 | 0.881 | 0.910 | 0.94 | 1.15 | 30.3 | 8.5 | 6.3 | - |
| ae32__uint8__roi0.3 | per_channel_uint8 | zlib | 0.30 | 32 | 161.5 | 6% | 0.799 | 0.846 | 0.864 | 0.902 | 0.88 | 1.06 | 29.6 | 8.2 | 6.0 | - |
| ae32__uint6__roi0.0 | per_channel_uint6 | zlib | 0.00 | 32 | 172.7 | 6% | 0.822 | 0.916 | 0.865 | 0.902 | 0.88 | 1.06 | 27.7 | 9.8 | 6.3 | Y |
| ae128__uint6__roi0.3 | per_channel_uint6 | zlib | 0.30 | 128 | 190.9 | 7% | 0.756 | 0.734 | 0.883 | 0.909 | 0.87 | 1.05 | 30.9 | 8.8 | 6.9 | - |
| ae64__uint8__roi0.3 | per_channel_uint8 | zlib | 0.30 | 64 | 193.2 | 7% | 0.805 | 0.856 | 0.864 | 0.899 | 0.86 | 1.03 | 30.7 | 9.7 | 7.0 | Y |
| ae128__uint8__roi0.5 | per_channel_uint8 | zlib | 0.50 | 128 | 199.5 | 7% | 0.730 | 0.674 | 0.881 | 0.909 | 0.94 | 1.14 | 30.5 | 10.7 | 7.7 | - |
| ae64__uint6__roi0.0 | per_channel_uint6 | zlib | 0.00 | 64 | 200.2 | 7% | 0.825 | 0.916 | 0.866 | 0.901 | 0.87 | 1.04 | 28.6 | 8.2 | 7.0 | Y |
| ae32__uint8__roi0.0 | per_channel_uint8 | zlib | 0.00 | 32 | 225.0 | 8% | 0.822 | 0.916 | 0.863 | 0.902 | 0.88 | 1.06 | 28.7 | 7.9 | 7.4 | - |
| noae__uint4__roi0.5 | per_channel_uint4 | zlib | 0.50 | - | 234.8 | 8% | 0.613 | 0.363 | 0.817 | 0.860 | 1.11 | 1.29 | 33.7 | 7.4 | 10.6 | - |
| ae128__uint8__roi0.3 | per_channel_uint8 | zlib | 0.30 | 128 | 250.3 | 9% | 0.759 | 0.744 | 0.883 | 0.908 | 0.87 | 1.05 | 30.9 | 11.7 | 8.6 | - |
| ae128__uint6__roi0.0 | per_channel_uint6 | zlib | 0.00 | 128 | 257.8 | 9% | 0.819 | 0.914 | 0.884 | 0.909 | 0.87 | 1.05 | 30.0 | 8.1 | 8.5 | Y |
| ae64__uint8__roi0.0 | per_channel_uint8 | zlib | 0.00 | 64 | 262.7 | 9% | 0.825 | 0.916 | 0.864 | 0.900 | 0.87 | 1.04 | 29.3 | 10.4 | 8.5 | Y |
| noae__uint4__roi0.3 | per_channel_uint4 | zlib | 0.30 | - | 285.0 | 10% | 0.719 | 0.593 | 0.851 | 0.879 | 1.04 | 1.22 | 33.7 | 11.9 | 11.8 | - |
| ae128__uint8__roi0.0 | per_channel_uint8 | zlib | 0.00 | 128 | 337.7 | 12% | 0.819 | 0.914 | 0.883 | 0.908 | 0.87 | 1.05 | 30.8 | 10.1 | 10.4 | Y |
| noae__uint4__roi0.0 | per_channel_uint4 | zlib | 0.00 | - | 387.6 | 14% | 0.838 | 0.931 | 0.843 | 0.875 | 1.05 | 1.22 | 35.2 | 9.6 | 14.6 | - |
| noae__uint6__roi0.5 | per_channel_uint6 | zlib | 0.50 | - | 449.0 | 16% | 0.774 | 0.766 | 0.832 | 0.867 | 1.04 | 1.19 | 39.6 | 10.1 | 16.8 | - |
| noae__uint6__roi0.3 | per_channel_uint6 | zlib | 0.30 | - | 564.9 | 20% | 0.797 | 0.808 | 0.853 | 0.877 | 0.96 | 1.10 | 42.5 | 9.2 | 19.5 | - |
| noae__uint8__roi0.5 | per_channel_uint8 | zlib | 0.50 | - | 586.6 | 21% | 0.782 | 0.789 | 0.838 | 0.868 | 1.04 | 1.19 | 39.4 | 8.3 | 20.0 | - |
| noae__uint8__roi0.3 | per_channel_uint8 | zlib | 0.30 | - | 753.6 | 27% | 0.803 | 0.825 | 0.853 | 0.878 | 0.95 | 1.08 | 43.6 | 8.1 | 23.8 | - |
| noae__uint6__roi0.0 | per_channel_uint6 | zlib | 0.00 | - | 783.3 | 28% | 0.840 | 0.931 | 0.852 | 0.878 | 0.96 | 1.09 | 44.8 | 10.9 | 24.8 | - |
| noae__uint8__roi0.0 | per_channel_uint8 | zlib | 0.00 | - | 1052.9 | 37% | 0.840 | 0.931 | 0.855 | 0.879 | 0.95 | 1.08 | 46.0 | 9.1 | 30.7 | - |
| uncompressed_fp16 | none(fp16) | - | 0.00 | - | 2835.0 | 100% | 0.819 | 0.914 | 0.883 | 0.908 | 0.87 | 1.06 | ~46.0 | ~9.1 | ~30.7 | Y |
| ae128__clean | none | zlib | 0.00 | - | nan | nan% | 0.819 | 0.914 | 0.883 | 0.908 | 0.87 | 1.06 | - | - | - | Y |
| ae32__clean | none | zlib | 0.00 | - | nan | nan% | 0.822 | 0.916 | 0.863 | 0.902 | 0.88 | 1.06 | - | - | - | - |
| ae64__clean | none | zlib | 0.00 | - | nan | nan% | 0.825 | 0.916 | 0.864 | 0.900 | 0.87 | 1.04 | - | - | - | Y |
| noae__clean | none | zlib | 0.00 | - | nan | nan% | 0.839 | 0.931 | 0.853 | 0.878 | 0.95 | 1.08 | - | - | - | - |

## Pareto pick (min payload within accuracy tolerance)
**ae128__uint4__roi0.0** — payload 127.4KB (4% of clean), mIoU 0.819, ped-recall 0.887, loc 0.88m.

## For the RL controller
- This table is the offline action-cost model: each row is a discrete action, columns are the reward terms (task utility) and the payload/latency/reliability cost.
- **front ms / RTT ms / delivery** are measured on the loopback (CARLA transport) for the pure quant x entropy profiles; `~` marks ROI/AE profiles whose latency/reliability follow the same **payload -> {latency, reliability}** curve (see LOOPBACK_LATENCY_ZLIB.md / loopback_latency_zlib.json) via their payload column.
- Loopback delivery reflects payload/fragmentation; TRUE channel loss + variable latency arrive with the OAI/Sionna network phase, which replaces the loopback transport column.
