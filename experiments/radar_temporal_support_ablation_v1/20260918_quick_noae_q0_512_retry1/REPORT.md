# Quick radar temporal-support sensitivity check

## Decision

Retain the existing rolling 200 ms (two-sweep) radar contract. On a paired,
deterministic 512-frame validation subset, replacing it with only the current
100 ms sweep produced mixed changes rather than a consistent quality gain.
This check did not change production settings, train a model, or launch CARLA,
OAI, or a live network workload.

## Paired design

- Frozen noAE FCOS checkpoint, UINT8, `q=0`
- Identical 512 RGB frames, frame identities, calibration, model weights,
  transport codec, post-processing, p025 policy, and frozen scorers
- 256 route-spanning frames from each of the two registered validation episodes
- Only radar support changes: current + previous sweep (200 ms) versus current
  sweep alone (100 ms)
- Thirty-two persisted 200 ms radar tensors were reconstructed exactly from
  retained point records (`maximum_absolute_error = 0`)

## Result

| Metric | Rolling 200 ms | Current 100 ms | 100 ms minus 200 ms |
|---|---:|---:|---:|
| Vehicle IoU | 0.902400 | 0.900215 | -0.002186 |
| Person box-mask IoU | 0.530367 | 0.525319 | -0.005047 |
| Foreground mIoU | 0.716383 | 0.712767 | -0.003616 |
| Vehicle F1 | 0.896795 | 0.899718 | +0.002923 |
| Vehicle XY MAE (m; lower is better) | 0.475247 | 0.487259 | +0.012012 |
| Person AVO F1 | 0.713287 | 0.710434 | -0.002853 |
| Person AVO recall | 0.716628 | 0.709602 | -0.007026 |
| Person AVO recall, 20--40 m | 0.581749 | 0.574144 | -0.007605 |
| Person AVO XY MAE (m; lower is better) | 0.819382 | 0.811936 | -0.007446 |

The 100 ms condition slightly improves vehicle F1 and person XY error, but it
slightly worsens both segmentation measures, person recall (including 20--40
m), and vehicle localization. Its median compressed feature payload changes by
only -2,847 bytes (-0.08%), because the transmitted representation has the same
shape in both conditions.

## Interpretation boundary

This is a frozen-model sensitivity check, not proof that 200 ms is universally
optimal. The model was trained with 200 ms support, and the current-only input
retains tracker ages learned from the historical stream. A production switch
to 100 ms would therefore require native 100 ms tracker replay, model
requalification or fine-tuning, and live validation. The present mixed result
does not justify that detour.

The completed 288-cell campaign remains valid for its registered 200 ms sensor
contract. It must not be relabelled as a 100 ms campaign.

## Evidence

- `quick_radar_support_ablation.json` SHA-256:
  `ea7ac63a687de0902ad39990857dbb005bf229564d030809e90a5b4067d3b53a`
- Frozen checkpoint SHA-256:
  `da14d21edbd374c1c3abce02ca4674b9f4097becfba9759aba945cea160a297f`
- Dataset manifest SHA-256:
  `5d65e6eb14aadea11ca6bab6e82f0c94c31a50746611d167d282d8988a4504c2`
