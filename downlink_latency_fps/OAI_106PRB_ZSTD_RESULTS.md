# OAI 106PRB zstd CARLA Result

Date: 2026-07-21

Purpose: test whether the lossless `zstd` entropy coder improves the live
CARLA Step-1 deployment over the same default OAI 106PRB path previously run
with `zlib`.

> **2026-07-22 correction.** The earlier live OAI zlib/zstd comparison used an
> obsolete frontend scene (`60` vehicles, `20` pedestrians, obey-all-lights) and
> often produced congested/occluded bus-tunnel behavior. Those raw runs,
> summaries, and plots were removed on 2026-07-22 to avoid accidental reporting.
> Use only the corrected drivable deployment rerun below:
> `28` vehicles, `35` pedestrians, seed `31`, ego ignores lights `50%`, fixed
> waypoint loop `80,85,91,94,99,80`.

## Corrected 106PRB OAI drivable-scene rerun

Batch: `drivable_ab_20260721_233016`

Runs:

- zlib: `downlink_oai_default_drivable_zlib_fps10_drivable_ab_20260721_233016`
- zstd: `downlink_oai_default_drivable_zstd_fps10_drivable_ab_20260721_233016`

Both runs used:

- default OAI 106PRB, numerology 1, default TDD;
- live CARLA frontend, 10FPS target, 1300 frames;
- no-AE checkpoint, per-channel-u8, ROI 0, 200k radar PPS;
- corrected drivable deployment scene: 28 vehicles, 35 pedestrians, seed 31,
  `--ego-ignore-lights-pct 50`, fixed waypoint loop `80,85,91,94,99,80`.

| Metric | zlib corrected scene | zstd corrected scene | Change |
|---|---:|---:|---:|
| Frames | 1300 | 1300 | same |
| Returned frames | 937 | 1087 | +150 |
| Delivery | 72.1% | 83.6% | +11.5 percentage points |
| Feature payload p50 | 1084.8 KB | 1055.2 KB | -2.7% |
| Feature chunks p50 | 19 | 19 | same |
| Capture→result p50 | 251.4 ms | 188.0 ms | -63.4 ms / -25.2% |
| Front p50 | 48.0 ms | 25.2 ms | -22.8 ms / -47.5% |
| RTT p50 | 202.5 ms | 162.2 ms | -40.3 ms / -19.9% |
| RTT p95 | 237.7 ms | 175.3 ms | -62.4 ms / -26.3% |
| Feature/uplink handling p50 | 183.8 ms | 151.1 ms | -32.7 ms / -17.8% |
| Edge tail p50 | 7.1 ms | 6.9 ms | roughly same |
| Downlink p50 | 9.0 ms | 3.0 ms | lower; result payload also smaller |

Interpretation: after fixing the frontend scene, the zstd conclusion still
holds. zstd improves both delivery and latency on the same default 106PRB OAI
path, while the model/back-half compute stays unchanged. It is still not a full
fix: delivery is `83.6%`, not near-100%, and feature/uplink handling is still
about `151 ms`.

### Corrected-scene live accuracy sanity

The corrected scene also removes the old congestion/bus occlusion artifact from
the live localization sanity check. Using score `>=0.20`, origin GT, class-aware
greedy matching:

| View, returned frames only | zlib | zstd |
|---|---:|---:|
| 5 m gate, GT ≤40 m: loc mean / median / p90 | 1.82 / 1.39 / 3.96 m | 1.86 / 1.28 / 4.02 m |
| 2 m gate, GT ≤20 m: loc mean / median / p90 | 0.90 / 0.81 / 1.50 m | 0.80 / 0.74 / 1.31 m |
| Matched score median, 5 m gate | 0.518 | 0.548 |

This is the expected reading: zstd does not degrade task accuracy; if anything,
the extra delivered frames give slightly more matched opportunities. The exact
codec-invariance proof remains the offline matrix, where the same
`noae__uint8__roi0.0` profile has identical task metrics under zlib and zstd.

## Artifacts

- Corrected zlib run:
  `runs/oai_default_drivable_zlib/fps_10_drivable_ab_20260721_233016/`
- Corrected zstd run:
  `runs/oai_default_drivable_zstd/fps_10_drivable_ab_20260721_233016/`
- Corrected comparison summary CSV:
  `runs/downlink_fps_summary_drivable_ab_20260721_233016.csv`
- Corrected latency/delivery plot:
  `plots/oai_bottleneck/oai_106prb_drivable_zlib_vs_zstd.pdf`
- Corrected live accuracy sanity plot:
  `plots/oai_bottleneck/oai_106prb_drivable_zlib_vs_zstd_accuracy.pdf`
- Generic repeatable runner for either codec:
  `run_oai_default_codec_10fps.sh`
