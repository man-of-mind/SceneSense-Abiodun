# SplitFusion latency, quality, and sensor analysis

This report uses the corrected direct edge-to-map counterfactual. It is
an offline causal replay, not a live remeasurement of the 288 cells.

## Main interpretation

- Validation quality is action-dependent and therefore repeats across network profiles; latency and survival move with the network.
- A point is absent from an edge/total plot when that cell produced no useful map installation. Absence is not encoded as zero latency.
- The combined score is a provisional presentation coordinate, not the PPO reward.
- Physical map installation ends at the edge-host map. Compact controller feedback to the UE is a later observation and is excluded from map AoI.

## Action-balanced stage percentiles

| Profile | Stage | P50 (ms) | P95 (ms) | P99 (ms) | Actions represented |
|---|---|---:|---:|---:|---:|
| Favorable Stable | Model front backbone | 1.3 | 4.4 | 11.1 | 72/72 |
| Favorable Stable | UE action path | 25.0 | 53.4 | 75.7 | 72/72 |
| Favorable Stable | Feature uplink | 55.9 | 93.7 | 137.4 | 64/72 |
| Favorable Stable | Edge-to-map service | 74.1 | 171.8 | 230.0 | 71/72 |
| Favorable Stable | Action-to-map total | 167.8 | 265.6 | 323.6 | 71/72 |
| Mid Variable | Model front backbone | 1.3 | 4.3 | 10.1 | 72/72 |
| Mid Variable | UE action path | 24.9 | 53.3 | 76.8 | 72/72 |
| Mid Variable | Feature uplink | 66.0 | 108.9 | 154.2 | 56/72 |
| Mid Variable | Edge-to-map service | 74.7 | 166.7 | 227.1 | 62/72 |
| Mid Variable | Action-to-map total | 179.5 | 285.0 | 334.5 | 62/72 |
| Adverse Stable | Model front backbone | 1.3 | 4.0 | 8.9 | 72/72 |
| Adverse Stable | UE action path | 23.8 | 50.4 | 71.7 | 72/72 |
| Adverse Stable | Feature uplink | 64.7 | 95.7 | 128.9 | 43/72 |
| Adverse Stable | Edge-to-map service | 71.7 | 164.9 | 223.7 | 46/72 |
| Adverse Stable | Action-to-map total | 171.2 | 268.5 | 322.0 | 46/72 |
| Fade Recovery | Model front backbone | 1.4 | 4.5 | 10.9 | 72/72 |
| Fade Recovery | UE action path | 24.7 | 53.4 | 76.6 | 72/72 |
| Fade Recovery | Feature uplink | 61.0 | 105.2 | 140.4 | 59/72 |
| Fade Recovery | Edge-to-map service | 75.2 | 164.9 | 228.5 | 63/72 |
| Fade Recovery | Action-to-map total | 179.8 | 273.9 | 326.4 | 63/72 |

Percentile aggregation is action-balanced: each displayed value is the median of the corresponding per-action cell percentile. It is not a pooled-frame percentile dominated by high-throughput actions.

## Sensor preparation boundary

The existing 288 evidence supports a trustworthy breakdown only to the retained function boundaries: callback-to-worker scheduling, radar-window extraction, aggregate radar preparation, RGB conversion, and evaluation-snapshot capture. The inner radar projection/tracking/rasterization functions were not individually timed, so this report does not invent them.

Camera and radar callbacks are distinct CARLA callback threads, but the numerical RGB conversion and radar-window/raster preparation used by one frame run sequentially inside the single `route-b-split-front` worker. Evaluation is already dispatched separately after its immutable scene snapshot is captured.

`sensor_wait_ms` is shown only as a non-additive operational diagnostic because that wait can begin before the RGB capture event used as the action-age boundary.

## Clock and denominator integrity

The same-host clock bridge used 344,177 anchors; its absolute error P99 was 0.001099 ms.
Every stage carries its own count. Pure model-front and complete UE action timing use all sent frames; feature-uplink timing uses only retained observed complete edge receipts (never imputed arrivals); edge and total timing use useful direct-map installations.

## Quality definition

$$
Q_{\mathrm{loc}}=\sqrt{\mathrm{IoU}_{\mathrm{vehicle}}\,\mathrm{IoU}_{\mathrm{person}}},
\qquad
Q_{\mathrm{joint}}=\sqrt{mIoU_{\mathrm{seg}}\,Q_{\mathrm{loc}}}.
$$

All terms are frozen validation metrics in $[0,1]$. The geometric mean is conservative: an action cannot appear strong merely because one quality dimension hides a weak one.

## Figure guide

Figures 01–04 show all available action/profile points and label Pareto-frontier action IDs. Figure 05 gives the requested P50/P95/P99 bars for model front, feature uplink and edge-to-map service. Figure 06 exposes the retained sensor-preparation boundaries.
