# OAI Config CARLA Frontend Results

Date: 2026-07-22

Purpose: track live CARLA frontend Step-1 OAI measurements using the corrected
deployment scene.

## 2026-07-22 cleanup note

The earlier default/UL-heavy/273PRB live rows used the wrong frontend command:
`60` vehicles, `20` pedestrians, and obey-all-lights behavior. Those raw run
folders, summaries, and plots were deleted on 2026-07-22. They should not be
reported or used for decisions.

Going forward, rerun each OAI condition with the corrected `run_common.sh`
deployment scene:

- 28 vehicles;
- 35 pedestrians;
- seed 31;
- ego ignore-lights 50%;
- fixed waypoint loop `80,85,91,94,99,80`;
- no-AE checkpoint, per-channel-u8, ROI 0, 200k radar PPS;
- live CARLA frontend, 10 FPS target, 1300 frames.

## Corrected default 106PRB codec A/B

Batch `drivable_ab_20260721_233016` reran the default 106PRB OAI path with the
corrected drivable scene.

Important: this is the normal closed-loop frontend mode. It waits for the result
or timeout before advancing to the next frame. So it measures deployed
application behavior, not an open-loop offered-load stress test.

| Condition | RAN config | Frames | Returned | Delivery | Ego speed mean | Moving frac | RTT p50 | RTT p95 | Front p50 | Back p50 | Downlink p50 | Feature/uplink handling p50 | Capture→result p50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Corrected default OAI, zlib | 106 PRB, mu=1, default TDD | 1300 | 937 | 72.1% | 2.1 m/s | 66.4% | 202.5 ms | 237.7 ms | 48.0 ms | 7.1 ms | 9.0 ms | 183.8 ms | 251.4 ms |
| Corrected default OAI, zstd | 106 PRB, mu=1, default TDD | 1300 | 1087 | 83.6% | 2.1 m/s | 66.4% | 162.2 ms | 175.3 ms | 25.2 ms | 6.9 ms | 3.0 ms | 151.1 ms | 188.0 ms |

## Interpretation

The corrected drivable-scene zstd rerun is currently the reportable OAI 106PRB
application-side comparison:

- delivery improves from `72.1%` to `83.6%`;
- RTT p50 improves from `202.5 ms` to `162.2 ms`;
- feature/uplink handling p50 improves from `183.8 ms` to `151.1 ms`;
- capture-to-result p50 improves from `251.4 ms` to `188.0 ms`;
- back-half compute stays effectively unchanged.

This does **not** remove the OAI bottleneck. Even with zstd, delivery is still
below the target for a reliable live deployment, and the feature/uplink handling
p50 remains about `151 ms` for a ~`1.06 MB` no-AE feature burst. The useful
lesson is narrower but strong: lossless codec choice is a valuable knob, and
payload reduction beyond entropy coding is still likely needed.

## Conditions to rerun with corrected command

The following previously measured conditions were intentionally removed and
should be rerun before being reported:

- ideal loopback FPS sweep;
- default OAI FPS sweep;
- OAI queue/backlog probes;
- UL-heavy 106PRB;
- 273PRB wider-bandwidth live CARLA run;
- 273PRB T-tracer live CARLA run.

## Artifacts

- Corrected OAI zlib closed-loop run:
  `runs/oai_default_drivable_zlib/fps_10_drivable_ab_20260721_233016/`
- Corrected OAI zstd closed-loop run:
  `runs/oai_default_drivable_zstd/fps_10_drivable_ab_20260721_233016/`
- Corrected OAI zlib-vs-zstd summary CSV:
  `runs/downlink_fps_summary_drivable_ab_20260721_233016.csv`
- Corrected OAI zlib-vs-zstd plot:
  `plots/oai_bottleneck/oai_106prb_drivable_zlib_vs_zstd.pdf`
- Corrected OAI zlib-vs-zstd accuracy plot:
  `plots/oai_bottleneck/oai_106prb_drivable_zlib_vs_zstd_accuracy.pdf`
- Repeatable codec runner:
  `run_oai_default_codec_10fps.sh`
