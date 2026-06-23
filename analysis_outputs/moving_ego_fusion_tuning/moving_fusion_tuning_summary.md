# Moving-Ego Fusion Tuning Summary

## Current Baseline

- Overall 8-loop moving model: mIoU `0.825`, vehicle IoU `0.874`, person IoU `0.630`.
- Low density is the main vehicle-IoU bottleneck: vehicle IoU `0.773`, gap to 0.90 target `0.127`.
- Medium density is also below target: vehicle IoU `0.828`, gap `0.072`.
- Crowded density already reaches the vehicle target: vehicle IoU `0.903`.

## Interpretation

- More crowded-only data is not the cleanest next move because crowded traffic already performs best. It may improve the easiest bucket while leaving low/medium weak.
- The useful test is whether tuning the objective raises low/medium vehicle IoU without sacrificing crowded performance.
- With person IoU near `0.63`, mIoU `0.85` is mathematically hard unless vehicle IoU gets very high or person IoU also improves. For now, vehicle IoU > `0.90` is the practical near-term gate.

## Tuning Results

- Best `overall` vehicle IoU: `0.874` from `baseline_8loop` (delta vs 8-loop `0.000`).
- Best `low` vehicle IoU: `0.773` from `baseline_8loop` (delta vs 8-loop `0.000`).
- Best `medium` vehicle IoU: `0.828` from `baseline_8loop` (delta vs 8-loop `0.000`).
- Best `crowded` vehicle IoU: `0.903` from `baseline_8loop` (delta vs 8-loop `0.000`).

## Files

- `moving_fusion_tuning_summary.csv`
- `moving_fusion_tuning_vehicle_iou_by_density.png`
- `moving_fusion_tuning_miou_by_density.png`
- `moving_fusion_tuning_delta_vs_baseline.png`, created once tuning rows exist
