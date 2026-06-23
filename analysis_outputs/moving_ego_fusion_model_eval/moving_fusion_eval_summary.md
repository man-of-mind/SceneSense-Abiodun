# Moving-Ego Fusion Evaluation Summary

## Main Readout

- The 8-loop moving model is currently the stronger moving-domain checkpoint: mIoU=0.825, vehicle IoU=0.874, person IoU=0.630.
- The 12-loop/more-data run did not improve segmentation on its own: mIoU=0.813, vehicle IoU=0.846, person IoU=0.624.
- Localization improved slightly with the 12-loop run, but remains weak enough that it should be treated as an engineering target, not a solved metric: F1 0.287 -> 0.307, XY error 1.430 m -> 1.373 m.
- Parked A+B model performance on moving data remains a negative control, not the main path forward: mIoU=0.262, vehicle IoU=0.054.

## Next Model Direction

- Keep the moving model as the main candidate for moving-domain work.
- Do not assume more repeated loops are enough; the next improvement should add route/viewpoint diversity, sensor-processing improvements, or training loss/threshold tuning.
- Evaluate the moving model on parked View A/B only as a domain-gap diagnostic, not as the success criterion.

## Generated Artifacts

- `moving_fusion_segmentation_8_vs_12loops.png`
- `moving_fusion_localization_8_vs_12loops.png`
- `moving_fusion_domain_gap_segmentation.png`
- `moving_fusion_eval_summary.csv`
