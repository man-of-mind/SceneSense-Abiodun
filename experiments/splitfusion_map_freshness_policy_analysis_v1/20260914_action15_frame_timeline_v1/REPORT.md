# Action 15 frame-level preparation diagnostic

Each figure shows every sent frame across the retained Action-15 measurement
interval. The interval is approximately five minutes per network profile.
Red markers identify the exact frames in that cell's slowest one percent.

## Profile summary

| Network profile | Frames | Duration | Delay median | Delay P95 | Delay P99 | Maximum | r(delay, radar preparation) | r(delay, acceleration) | r(delay, radar returns) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Favorable | 2,907 | 5.02 min | 40.2 ms | 103.5 ms | 164.3 ms | 286.9 ms | 0.779 | -0.018 | -0.000 |
| Mid-variable | 2,961 | 5.05 min | 36.4 ms | 89.6 ms | 138.4 ms | 263.6 ms | 0.734 | -0.012 | 0.086 |
| Fade/recovery | 2,865 | 5.02 min | 39.7 ms | 113.3 ms | 164.3 ms | 386.1 ms | 0.752 | 0.010 | 0.061 |
| Adverse | 2,973 | 5.01 min | 31.0 ms | 72.0 ms | 114.4 ms | 200.6 ms | 0.759 | 0.002 | -0.034 |

## What the synchronized traces show

Across the six profile pairs, the one-second ego-speed traces correlate
strongly (0.917 to 0.964), confirming that the four runs follow closely
aligned motion patterns. In contrast, the one-second preparation-delay
traces correlate only -0.123 to 0.093: the delay bursts do not recur at
the same elapsed route times.

Within each profile, delay correlates strongly with radar preparation
(0.734 to 0.779) and total pre-front compute (0.788 to 0.830), but not
with acceleration (-0.018 to 0.010) or radar-return count (-0.034 to
0.086). This evidence supports preparation/runtime variability rather
than scene density or vehicle motion as the primary Action-15 cause.

## Interpretation boundary

Action ID is not route position. The four cells were separate live runs,
so visually similar percentile bands do not establish frame-aligned scene
bursts. The synchronized timelines make the within-cell relationships
inspectable. Correlation describes association, not causation.

The callback-to-action interval ends before final seven-channel tensor
assembly, but already contains radar-window extraction, radar rasterisation,
CARLA-image conversion and the evaluation-only scene snapshot.
