# Moving-Ego Radar Model Ablation

| Radar pps | Person support | Status | mIoU | Vehicle IoU | Person IoU | Loc F1 | XY MAE (m) |
|---:|---|---|---:|---:|---:|---:|---:|
| 5000 | bbox | ok | 0.797 | 0.833 | 0.592 | 0.164 | 1.528 |
| 5000 | radius | ok | 0.789 | 0.784 | 0.616 | 0.188 | 1.551 |
| 12000 | bbox | ok | 0.812 | 0.841 | 0.626 | 0.143 | 1.642 |
| 12000 | radius | ok | 0.805 | 0.826 | 0.620 | 0.172 | 1.495 |

## Interpretation Guide

- `5k:bbox -> 5k:radius` isolates the geometry/association change.
- `5k:radius -> 12k:radius` isolates radar point-density under the same person geometry.
- `12k:bbox -> 12k:radius` checks whether geometry still matters when radar is denser.
- Compare against the support-level factorial table before deciding whether the model learned to exploit the extra radar evidence.
