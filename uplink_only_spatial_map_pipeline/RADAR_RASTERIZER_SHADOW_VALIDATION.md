# Same-frame radar rasterizer shadow validation

Date: 2026-07-29

Run root:
`runs/radar_rasterizer_shadow_20260729_30f`

Shadow CSV:
`runs/radar_rasterizer_shadow_20260729_30f/radar_rasterizer_shadow.csv`

Summary JSON:
`runs/radar_rasterizer_shadow_20260729_30f/radar_rasterizer_shadow_summary.json`

## Purpose

Validate that `--radar-rasterizer fast` does not change the radar tensor or
decoded model outputs in a way that would affect localization/model accuracy.

This validation avoids the earlier weak spot where legacy and fast were compared
using separate live CARLA runs and a loose GT matcher. Here, both rasterizers
are evaluated on the exact same radar measurement in the exact same frame.

## Method

For each live frame:

1. Convert the CARLA radar measurement once.
2. Project radar points once.
3. Update the stationary tracker once.
4. Rasterize the same projected points through both:
   - legacy Python point-paint loop;
   - fast vectorized scatter + max-filter dilation.
5. Apply the same temporal max-pool history to both tensors.
6. Compare the final 4-channel radar tensors.
7. Run both same-frame tensors through the model locally and compare decoded
   objects.

Command features:

- 30 live CARLA frames.
- moving ego, route `80,85,91,94,99,80`
- no-AE baseline checkpoint
- 200k radar PPS
- radius 4
- temporal window 2
- active runtime rasterizer: `fast`
- shadow decode enabled with `--radar-rasterizer-shadow-decode`

## Tensor result

| Metric | Result |
|---|---:|
| Frames compared | 30 |
| Radar points/frame, p50 | 19,728.5 |
| Tensor max abs diff | `5.96e-08` |
| Tensor mean abs diff, mean | `3.35e-10` |
| Tensor entries differing > `1e-6` | 0 |
| Tensor entries differing > `1e-4` | 0 |
| Occupancy changed pixels | 0 |
| Velocity max abs diff | 0 |
| Stationary-age max abs diff | 0 |
| Range max abs diff | `5.96e-08` |

Interpretation: the final model-input radar tensor is equivalent to numerical
precision. The only nonzero tensor difference is a tiny float-rounding-level
range-channel difference.

## Decoded-object result

| Metric | Result |
|---|---:|
| Legacy object count p50 | 3/frame |
| Fast object count p50 | 3/frame |
| Object-count delta | 0 on all frames |
| Legacy unmatched objects | 0 on all frames |
| Fast unmatched objects | 0 on all frames |
| Matched center-pixel max distance | 0 px |
| Matched world-XY max distance | 0.0102 m |
| Matched world-XY mean distance | 0.0025 m |
| Matched score max abs diff | 0.00269 |

Interpretation: decoded object decisions are stable. Every legacy decoded object
matched a fast decoded object in every frame; centers were identical in pixel
space, and world-coordinate differences were millimeter-to-centimeter scale.

## Dense-logit note

The dense segmentation/object maps are not bit-identical across the two local
forward passes:

| Dense output | Max abs diff |
|---|---:|
| Segmentation logits | 0.069 |
| Object map | 3.439 |

This should not be interpreted as a task-level localization change. The decoded
object outputs are the relevant downstream signal, and those were stable. The
dense object-map maximum can occur at non-selected/background cells and is more
sensitive than the final NMS/decoded-object output.

## Conclusion

The same-frame validation supports using `--radar-rasterizer fast` for the next
Track-1 loopback/OAI profiling runs. It avoids the earlier live-run matching
artifact and gives direct evidence that the model receives effectively the same
radar tensor and produces the same decoded object decisions.

Do not report the earlier ~3 m live A/B matcher value as the model localization
floor. The matrix/offline baseline remains the correct model-accuracy anchor
for no-AE u8: about `0.95 m` localization error.

