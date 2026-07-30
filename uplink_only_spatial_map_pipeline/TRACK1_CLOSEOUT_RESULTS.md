# Track 1 closeout: uplink-only spatial-map pipeline

Date: 2026-07-30

Track 1 tests the project-relevant path: CARLA/front split features go uplink to the edge tail and are published toward the spatial-map side. The car does **not** wait for returned detections.

## Reportable runs

| Run | Target FPS | Actual FPS | Payload p50 KiB | Chunks p50 | Delivery | UDP partial drops | Edge queue drops | Uplink p50 ms | Capture→tail p50 ms | Backbone→tail p50 ms | MCS p50 | BSR p95 KiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline u8 roi0 1MB 10fps | 10 | 7.08 | 1048.9 | 18 | 95.4% | 59 | 0 | 65.6 | 154.8 | 93.9 | 16 | 1034.5 |
| noae u4 roi0 394KB 10fps | 10 | 7.37 | 394.0 | 7 | 100.0% | 0 | 0 | 58.9 | 146.4 | 87.1 | 7 | 392.5 |
| ae128 u6 roi05 157KB 10fps | 10 | 7.27 | 157.0 | 3 | 99.8% | 0 | 3 | 51.7 | 135.2 | 73.2 | 3 | 156.5 |
| ae128 u6 roi05 157KB 20fps probe | 20 | 9.12 | 156.9 | 3 | 99.0% | 0 | 13 | 46.0 | 113.0 | 74.4 | 3 | 155.2 |

## Main conclusions

1. **Payload reduction fixes the UDP/reassembly problem.** The 1 MB baseline had 59 UDP partial-message drops and 95.4% delivery. The 394 KiB no-AE/uint4 point reached 100.0% delivery with zero UDP partial drops. The 157 KiB AE-128 point also had zero UDP partial drops.
2. **Latency improves, but not proportionally to payload.** Capture→tail p50 moved from 154.8 ms to 146.4 ms to 135.2 ms. The OAI uplink term improves, but sensor/front production still dominates the map staleness budget.
3. **Actual FPS is still front/CARLA limited.** At 10 FPS target, actual send rate stayed around 7.1--7.4 FPS. The AE-128 20 FPS probe reached 9.1 FPS, not 20 FPS, while keeping UDP partial drops at zero. This matches the ideal-loopback capacity observation that increasing target FPS raises actual FPS sublinearly.
4. **Residual AE delivery misses are edge-queue, not UDP/OAI partial reassembly.** AE-128 had 3 edge queue drops at 10 FPS and 13 at the 20 FPS probe. These are startup/drain-side application queue drops; they are separate from the old 1 MB UDP partial-message loss.
5. **Accuracy should be reported from the per-model knob matrix, not inferred from OAI transport.** Network transport changes frame availability/staleness; it does not change the decoded-frame model accuracy. The matching offline matrix entries are in `rl_agent/PERMODEL_KNOB_MATRIX_GROUPED.md` / `PERMODEL_KNOB_MATRIX_ZSTD.md`.

## Implementation note

During the AE run, the first attempt failed because the copied Track-1 live script still treated AE as an external standalone split-AE checkpoint. The current reportable AE run uses the **integrated AE checkpoint as the main fusion checkpoint**, matching the per-model matrix. The Track-1 runner now allows checkpoint override, and the copied live script attaches the integrated `feature_ae` before loading checkpoint weights.

## What this closes

Track 1 can be wrapped with this framing:

- The project-relevant uplink-only pipeline removes the closed-loop result-wait idle pattern.
- With 1 MB no-AE features, OAI default 106PRB still shows partial-message loss and ~155 ms capture→tail p50.
- Reducing feature payload to ~394 KiB or ~157 KiB removes UDP partial loss and lowers p50 staleness.
- The remaining ceiling is no longer mainly radio reassembly; it is a combination of CARLA/sensor/front production cadence plus small edge queue behavior.

## Plots

- `plots/track1_oai_reduced_payload/track1_oai_payload_latency_reliability.pdf`
- `plots/track1_oai_reduced_payload/track1_oai_radio_backlog_reduced_payload.pdf`
- `plots/track1_oai_reduced_payload/track1_oai_payload_comparison_summary.csv`

## Recommended next work after Track 1

1. Add real spatial-map worker timing instead of the current assumed +30 ms map compute.
2. Increase/warm the edge receive queue before starting timed captures if we want delivery accounting to exclude startup drops.
3. Carry the reduced-payload findings into the RL/action policy: payload reduction is a reliability and staleness lever, but FPS target alone cannot overcome the CARLA/front production ceiling.
