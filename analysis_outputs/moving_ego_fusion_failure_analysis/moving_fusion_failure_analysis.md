# Moving-Ego Fusion Failure and Data-Quality Analysis

## What We Can Inspect Locally

- Evaluation metrics JSON, learned-object TP/FP/FN CSVs, training curves, and the remote GPU per-density segmentation metrics are available locally.

## Key Findings

- The 8-loop model remains better for segmentation: mIoU `0.825`, vehicle IoU `0.874`, person IoU `0.630`.
- The 12-loop model is worse on segmentation despite more samples: mIoU `0.813`, vehicle IoU `0.846`, person IoU `0.624`.
- The 12-loop model improves localization slightly: object F1 `0.287` -> `0.307`, XY MAE `1.430m` -> `1.373m`.
- The improvement is class-skewed: vehicle localization improves, while person localization slightly degrades.
- The 12k-radar/class-aware pilot did not beat the 8-loop model yet: mIoU `0.805`, vehicle IoU `0.826`, person IoU `0.620`, object F1 `0.172`.

## Per-Density Segmentation
- `low`: mIoU `0.793`, vehicle IoU `0.773`, person IoU `0.634`, fusion gain over RGB baseline `+0.337`.
- `medium`: mIoU `0.812`, vehicle IoU `0.828`, person IoU `0.638`, fusion gain over RGB baseline `+0.294`.
- `crowded`: mIoU `0.829`, vehicle IoU `0.903`, person IoU `0.616`, fusion gain over RGB baseline `+0.279`.
- Segmentation does not simply collapse in crowded traffic. Vehicle IoU improves with density because there are more visible vehicle pixels, while person IoU is slightly best in medium density and lowest in crowded scenes.

## Moving model, 8 loops
- `vehicle` localization: precision `0.348`, recall `0.290`, F1 `0.317`, XY MAE `1.370m`.
- `person` localization: precision `0.268`, recall `0.221`, F1 `0.242`, XY MAE `1.546m`.
- `all` localization: precision `0.316`, recall `0.262`, F1 `0.287`, XY MAE `1.430m`.
- Density-level localization pressure:
  - `low`: TP `688`, FP `2323`, FN `1215`, F1 `0.280`.
  - `medium`: TP `1371`, FP `2778`, FN `4138`, F1 `0.284`.
  - `crowded`: TP `1577`, FP `2771`, FN `4872`, F1 `0.292`.
- Training peaked at validation mIoU `0.822` on epoch `69` and person IoU `0.621` on epoch `69`.

## Moving model, 12 loops
- `vehicle` localization: precision `0.379`, recall `0.348`, F1 `0.363`, XY MAE `1.311m`.
- `person` localization: precision `0.235`, recall `0.204`, F1 `0.218`, XY MAE `1.533m`.
- `all` localization: precision `0.324`, recall `0.291`, F1 `0.307`, XY MAE `1.373m`.
- Density-level localization pressure:
  - `low`: TP `816`, FP `3158`, FN `1297`, F1 `0.268`.
  - `medium`: TP `2046`, FP `3773`, FN `5360`, F1 `0.309`.
  - `crowded`: TP `2255`, FP `3737`, FN `5798`, F1 `0.321`.
- Training peaked at validation mIoU `0.811` on epoch `60` and person IoU `0.624` on epoch `56`.

## Moving radar-12k pilot, 2 loops
- `vehicle` localization: precision `0.282`, recall `0.143`, F1 `0.190`, XY MAE `1.369m`.
- `person` localization: precision `0.200`, recall `0.129`, F1 `0.157`, XY MAE `1.621m`.
- `all` localization: precision `0.234`, recall `0.136`, F1 `0.172`, XY MAE `1.495m`.
- Density-level localization pressure:
  - `low`: TP `117`, FP `787`, FN `550`, F1 `0.149`.
  - `medium`: TP `261`, FP `604`, FN `1487`, F1 `0.200`.
  - `crowded`: TP `281`, FP `762`, FN `2149`, F1 `0.162`.
- Training peaked at validation mIoU `0.804` on epoch `24` and person IoU `0.622` on epoch `24`.

## Diagnosis

- Repeating the same route more times mostly adds near-neighbor views. It increases object-head training examples, but does not add enough new visual geometry to improve segmentation.
- Pixel segmentation is reasonably stable across density; the bigger weakness is object/localization reliability, especially false negatives in medium/crowded scenes.
- The moving model's segmentation ceiling is now likely caused by route/view diversity, class imbalance, and/or label difficulty rather than insufficient epochs alone.
- The object head is still the weakest piece. False positives are high in every density bucket, and false negatives dominate medium/crowded scenes.
- Person localization is not fixed by more route loops; this supports the LiDAR/radar-processing investigation for sparse pedestrian returns.

## Recommended Next Experiment

1. Keep the 8-loop moving checkpoint as the current best segmentation checkpoint.
2. Stop adding repeated loops on the same route as the first fix; per-density segmentation shows the model already handles the three density levels fairly consistently.
3. Try a targeted training recipe instead of another repeated-route data run: stronger class weighting/person sampling, lower object score threshold sweep, and radar-processing changes for pedestrian support.
4. Add route diversity only if it changes viewpoint geometry, not just loop count.
