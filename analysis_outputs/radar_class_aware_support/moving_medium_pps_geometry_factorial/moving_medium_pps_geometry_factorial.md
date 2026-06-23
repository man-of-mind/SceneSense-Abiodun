# Moving Medium 2-Loop Radar Support: Point Density vs Geometry

This is a controlled same-route comparison using the medium-density 2-loop moving-ego datasets. The 5k and 12k runs have the same route distance and loop count; the bbox/radius variants are computed offline from the same saved radar points and actor labels.

| Configuration | Class | Support rate | Supported / total | Mean support points |
|---|---|---:|---:|---:|
| 5k radar, bbox person gate | person | 8.4% | 759/9066 | 0.14 |
| 5k radar, bbox person gate | vehicle | 49.1% | 2859/5824 | 7.01 |
| 5k radar, radius person gate | person | 15.9% | 1442/9066 | 0.33 |
| 5k radar, radius person gate | vehicle | 49.1% | 2859/5824 | 7.00 |
| 12k radar, bbox person gate | person | 15.4% | 1392/9066 | 0.34 |
| 12k radar, bbox person gate | vehicle | 54.9% | 3210/5852 | 16.85 |
| 12k radar, radius person gate | person | 26.3% | 2388/9066 | 0.81 |
| 12k radar, radius person gate | vehicle | 54.9% | 3210/5852 | 16.84 |

## Readout

- Geometry helps pedestrians: at 5k, radius gating improves person support from 8.4% to 15.9%; at 12k, it improves person support from 15.4% to 26.3%.
- Radar point density also helps: bbox-only 5k to 12k improves person support from 8.4% to 15.4%; radius-gated 5k to 12k improves from 15.9% to 26.3%.
- The two changes are complementary for support labels, but support-level gains did not automatically become model-level gains in the small 2-loop pilot. The model still needs enough data/epochs and possibly loss tuning to learn from the better radar evidence.
- Vehicles are much less sensitive to the person geometry rule, as expected; the main vehicle gain comes from denser radar points.
