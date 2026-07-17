# Experiment 3 — 10 m centered score-threshold diagnostic

Official centered gate remains score >= 0.20 unless deliberately re-frozen.
At that threshold, the 10 m centered run fails and no lateral sweep should be
interpreted.

The important diagnostic finding is that the correct target peak often exists at
a lower score. Lowering only the analysis threshold makes the 10 m target look
accurate, which points to score calibration / duplicate-peak ranking rather than
visibility, placement, radar support, or loopback delivery.

| analysis score threshold | prediction frames | <=2 m matches | mean error | median error | p90 error | mean forward error | mean lateral error |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 60/60 | 60/60 | 0.984 m | 1.026 m | 1.079 m | 0.082 m | -0.936 m |
| 0.10 | 60/60 | 60/60 | 1.106 m | 1.113 m | 1.284 m | 0.813 m | -0.717 m |
| 0.15 | 59/60 | 57/60 | 1.216 m | 1.194 m | 1.590 m | 0.865 m | -0.812 m |
| 0.20 | 52/60 | 20/60 | 2.215 m | 2.657 m | 2.992 m | 1.697 m | -1.412 m |
| 0.25 | 38/60 | 0/60 | 2.981 m | 2.999 m | 3.139 m | 2.206 m | -2.003 m |
| 0.30 | 30/60 | 0/60 | 3.030 m | 3.028 m | 3.154 m | 2.251 m | -2.026 m |

Example frame 8880:

- score 0.229 candidate: 3.158 m from target origin
- score 0.176 candidate: 1.242 m from target origin
- score 0.120 candidate: 1.958 m from target origin
- score 0.057 candidate: 1.208 m from target origin

So the 10 m centered scene is not a missing-detection case. It is a target-peak
selection/calibration case. A 10 m lateral sweep is only meaningful after the
analysis threshold and target-selection rule are explicitly frozen.
