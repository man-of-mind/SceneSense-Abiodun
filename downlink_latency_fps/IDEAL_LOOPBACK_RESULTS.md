# Ideal Loopback Results

Date: 2026-07-22

Status: **corrected drivable-scene rerun complete**.

Batch: `drivable_rerun_20260722_loopback`

This rerun re-establishes the local software/loopback floor using the corrected
`run_common.sh` deployment scene:

- 28 vehicles;
- 35 pedestrians;
- seed 31;
- ego ignore-lights 50%;
- fixed waypoint loop `80,85,91,94,99,80`;
- no-AE checkpoint, per-channel-u8, ROI 0, 200k radar PPS;
- lossless zstd entropy coding;
- live CARLA frontend, one-loop equivalent per FPS point.

## Corrected ideal-loopback FPS sweep

| FPS target | Frames | Returned | Delivery | Feature payload p50 | Result payload p50 | Front p50 | Back p50 | Uplink handling p50 | Downlink p50 | RTT p50 / p95 | Capture→result p50 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 650 | 650 | 100.0% | 1047.3 KB, 18 chunks | 3.3 KB, 1 chunk | 27.0 ms | 8.0 ms | 7.6 ms | 1.7 ms | 17.6 / 39.3 ms | 45.4 ms |
| 10 | 1300 | 1300 | 100.0% | 1053.9 KB, 18 chunks | 2.4 KB, 1 chunk | 26.7 ms | 8.3 ms | 7.6 ms | 1.6 ms | 18.3 / 39.2 ms | 46.1 ms |
| 20 | 2600 | 2600 | 100.0% | 1053.2 KB, 18 chunks | 2.8 KB, 1 chunk | 26.7 ms | 8.8 ms | 7.5 ms | 1.6 ms | 18.7 / 41.2 ms | 46.4 ms |
| 30 | 3900 | 3900 | 100.0% | 1052.0 KB, 18 chunks | 2.8 KB, 1 chunk | 27.2 ms | 7.8 ms | 7.7 ms | 1.7 ms | 17.4 / 37.4 ms | 45.3 ms |

## Interpretation

The corrected loopback sweep is stable across 5–30 FPS:

- delivery remains `100%`;
- front-side feature preparation/compression is about `27 ms`;
- edge tail inference is about `8 ms`;
- downlink/result return is only about `1.6–1.7 ms`;
- capture→result is about `45–46 ms`.

This is the clean local floor for the Step-1 latency study. The OAI runs should
be interpreted as extra transport/tunnel/RAN behavior on top of this floor, not
as model-compute overhead.

## Artifacts

- Summary CSV:
  `runs/downlink_fps_summary_drivable_rerun_20260722_loopback.csv`
- Run folders:
  `runs/ideal_loopback/fps_{5,10,20,30}_drivable_rerun_20260722_loopback/`
- Presentation plots:
  `plots/ideal_loopback_latency_breakdown.pdf`
  `plots/ideal_loopback_payloads.pdf`
  `plots/oai_bottleneck/corrected_ideal_loopback_fps_sweep.pdf`
