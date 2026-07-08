# CEP (Circular Error Probability) — localization accuracy vs distance

CEP50 = radius (m) within which 50% of matched detections land (median radial error); CEP95 = 95th percentile. Reused existing per-detection eval (global_xy_error_m) + gt_distance_m.


## Person — CEP50 (m) by distance & pps
| pps | 0-5 m | 5-10 m | 10-15 m | 15-20 m | 20-25 m | 25-30 m | 30-35 m | 35-40 m |
|---|---|---|---|---|---|---|---|---|
| 150k | 1.29 | 0.75 | 1.36 | 1.40 | 1.64 | 1.46 | 1.66 | 1.24 |
| 200k | 1.26 | 0.86 | 1.25 | 1.68 | 2.30 | 1.51 | 1.41 | 1.27 |
| 250k | 0.90 | 1.00 | 1.25 | 1.85 | 1.65 | 1.48 | 1.67 | 1.52 |
| 300k | 0.81 | 1.09 | 1.13 | 1.61 | 1.68 | 1.35 | 1.72 | 1.33 |

## Vehicle — CEP50 (m) by distance & pps
| pps | 0-5 m | 5-10 m | 10-15 m | 15-20 m | 20-25 m | 25-30 m | 30-35 m | 35-40 m |
|---|---|---|---|---|---|---|---|---|
| 150k | 0.46 | 0.73 | 0.89 | 1.07 | 1.40 | 1.78 | 1.53 | 1.31 |
| 200k | 0.42 | 0.60 | 0.86 | 1.14 | 1.76 | 1.74 | 1.39 | 1.36 |
| 250k | 0.36 | 0.78 | 0.84 | 1.20 | 1.66 | 1.51 | 1.22 | 1.05 |
| 300k | 0.48 | 0.60 | 0.91 | 1.20 | 1.60 | 1.74 | 1.17 | 1.36 |
