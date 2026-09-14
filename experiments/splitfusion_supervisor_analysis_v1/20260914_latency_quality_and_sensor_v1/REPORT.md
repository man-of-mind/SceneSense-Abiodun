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
| Favorable Stable | Latest-only edge wait | 0.0 | 47.8 | 82.9 | 71/72 |
| Favorable Stable | FCOS model tail | 21.2 | 24.8 | 29.5 | 72/72 |
| Favorable Stable | Other tail processing | 30.8 | 59.2 | 105.9 | 72/72 |
| Favorable Stable | Map service | 4.8 | 42.6 | 113.1 | 72/72 |
| Favorable Stable | Complete edge-to-map service | 74.1 | 171.8 | 230.0 | 71/72 |
| Favorable Stable | Action-to-map total | 167.8 | 265.6 | 323.6 | 71/72 |
| Mid Variable | Model front backbone | 1.3 | 4.3 | 10.1 | 72/72 |
| Mid Variable | UE action path | 24.9 | 53.3 | 76.8 | 72/72 |
| Mid Variable | Feature uplink | 66.0 | 108.9 | 154.2 | 56/72 |
| Mid Variable | Latest-only edge wait | 0.0 | 48.4 | 82.4 | 62/72 |
| Mid Variable | FCOS model tail | 21.2 | 24.8 | 29.5 | 72/72 |
| Mid Variable | Other tail processing | 30.8 | 59.2 | 105.9 | 72/72 |
| Mid Variable | Map service | 4.8 | 42.6 | 113.1 | 72/72 |
| Mid Variable | Complete edge-to-map service | 74.7 | 166.7 | 227.1 | 62/72 |
| Mid Variable | Action-to-map total | 179.5 | 285.0 | 334.5 | 62/72 |
| Adverse Stable | Model front backbone | 1.3 | 4.0 | 8.9 | 72/72 |
| Adverse Stable | UE action path | 23.8 | 50.4 | 71.7 | 72/72 |
| Adverse Stable | Feature uplink | 64.7 | 95.7 | 128.9 | 43/72 |
| Adverse Stable | Latest-only edge wait | 0.0 | 48.0 | 81.1 | 46/72 |
| Adverse Stable | FCOS model tail | 21.2 | 24.8 | 29.5 | 72/72 |
| Adverse Stable | Other tail processing | 30.8 | 59.2 | 105.9 | 72/72 |
| Adverse Stable | Map service | 4.8 | 42.6 | 113.1 | 72/72 |
| Adverse Stable | Complete edge-to-map service | 71.7 | 164.9 | 223.7 | 46/72 |
| Adverse Stable | Action-to-map total | 171.2 | 268.5 | 322.0 | 46/72 |
| Fade Recovery | Model front backbone | 1.4 | 4.5 | 10.9 | 72/72 |
| Fade Recovery | UE action path | 24.7 | 53.4 | 76.6 | 72/72 |
| Fade Recovery | Feature uplink | 61.0 | 105.2 | 140.4 | 59/72 |
| Fade Recovery | Latest-only edge wait | 0.0 | 44.2 | 80.9 | 63/72 |
| Fade Recovery | FCOS model tail | 21.2 | 24.8 | 29.5 | 72/72 |
| Fade Recovery | Other tail processing | 30.8 | 59.2 | 105.9 | 72/72 |
| Fade Recovery | Map service | 4.8 | 42.6 | 113.1 | 72/72 |
| Fade Recovery | Complete edge-to-map service | 75.2 | 164.9 | 228.5 | 63/72 |
| Fade Recovery | Action-to-map total | 179.8 | 273.9 | 326.4 | 63/72 |

Percentile aggregation is action-balanced: each displayed value is the median of the corresponding per-action cell percentile. It is not a pooled-frame percentile dominated by high-throughput actions.
The stage percentiles are marginals with different conditional denominators and must not be added to reconstruct an end-to-end percentile. Pure tail, tail support and direct map installation use the relevant live family-anchor distributions; edge queue and complete edge-to-map service come from the 288-cell causal replay.

## Interpreting the long tails

- Latest-only scheduling removes FIFO backlog, not non-preemptive waiting. While one frame is executing, a single newest pending frame may still wait until that execution finishes; older pending frames are replaced. This produces a zero action-balanced P50 but a 44–48 ms P95.
- The 105.9 ms action-balanced P99 for other tail processing is measured in the final optimized live anchors. Frame-level inspection localizes its family-dependent spikes mainly to unpack/dequantization, camera-aware post-processing, p025 filtering, and compact serialization. The pure FCOS CUDA tail remains approximately 21.2/24.8/29.5 ms at P50/P95/P99.
- Map service is direct publication-to-install on the edge-host path and does not traverse the radio. Its approximately 4.8 ms P50 but 42.6/113.1 ms P95/P99 comes from rare publisher scheduling and map-ingest/install stalls; it is not 5G downlink latency.

## Sensor preparation boundary

The existing 288 evidence supports a trustworthy breakdown only to the retained function boundaries: callback-to-worker scheduling, radar-window extraction, aggregate radar preparation, RGB conversion, and evaluation-snapshot capture. The inner radar projection/tracking/rasterization functions were not individually timed, so this report does not invent them.

Camera and radar callbacks are distinct CARLA callback threads, but the numerical RGB conversion and radar-window/raster preparation used by one frame run sequentially inside the single `route-b-split-front` worker. Evaluation is already dispatched separately after its immutable scene snapshot is captured.

`sensor_wait_ms` is shown only as a non-additive operational diagnostic because that wait can begin before the RGB capture event used as the action-age boundary.

## Clock and denominator integrity

The same-host clock bridge used 344,177 anchors; its absolute error P99 was 0.001099 ms.
Every stage carries its own count. Pure model-front and complete UE action timing use all sent frames; feature-uplink timing uses only retained observed complete edge receipts (never imputed arrivals); edge and total timing use useful direct-map installations.

## Quality definition

$$
Q_{\mathrm{overlap}}=\sqrt{\mathrm{IoU}_{\mathrm{vehicle}}\,\mathrm{IoU}_{\mathrm{person}}},
\qquad
e_{xy}=\sqrt{\frac{e_{\mathrm{vehicle}}^2+e_{\mathrm{person}}^2}{2}},
\qquad Q_{xy}=\exp(-e_{xy}/1\,\mathrm{m}),
$$

$$
Q_{\mathrm{loc}}=\sqrt{Q_{\mathrm{overlap}}Q_{xy}},
\qquad
Q_{\mathrm{joint}}=\sqrt{mIoU_{\mathrm{seg}}Q_{\mathrm{loc}}}.
$$

The overlap terms measure spatial box/footprint agreement; they are not the semantic-segmentation mIoU. Centroid XY MAE is explicit through a smooth one-metre reference scale. The one-metre value normalizes the presentation coordinate and is not a correctness gate. The geometric means are conservative: one strong dimension cannot hide a weak one.

## Figure guide

Figures 01–04 show all available action/profile points and label Pareto-frontier action IDs. Figure 05 gives the requested P50/P95/P99 decomposition into the complete UE action path, observed feature uplink, latest-only edge waiting, pure FCOS tail, other tail processing, and map service. Figure 06 exposes the retained sensor-preparation boundaries.
