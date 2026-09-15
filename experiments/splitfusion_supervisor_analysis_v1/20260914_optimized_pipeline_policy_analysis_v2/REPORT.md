# SplitFusion optimized-pipeline latency and quality analysis

This is an offline causal replay of the immutable 288-cell sent-frame
population. It applies the subsequently live-validated sensor preparation,
repaired-v3 edge service, renderer-off direct edge-to-map service, and
latest-only scheduling. It is not a second 288-cell live campaign.

## What changed and what did not

- The measured radio reassembly/admission outcomes and per-frame transport delays are held fixed. No missing uplink frame is fabricated.
- Sensor timing is changed by equal-percentile mapping from the contemporaneous live baseline distribution to the live optimized distribution; this is not a constant subtraction.
- The measured family service shapes are retained and rescaled to the newest repaired-v3 live edge-compute medians.
- Direct map service is sampled from the renderer-off live action-50 run. It is independent of split action and does not traverse the radio.
- Validation quality is action-dependent and unchanged by runtime optimization. Network profile changes delivery, latency, and freshness—not the offline quality anchor.
- A point is absent when its conditional stage has no trustworthy sample or no useful map installation; absence is never encoded as zero latency.
- Physical map age ends at authoritative map installation. The later compact UE feedback is excluded.

## Action-balanced stage percentiles

| Profile | Stage | P50 (ms) | P95 (ms) | P99 (ms) | Actions represented |
|---|---|---:|---:|---:|---:|
| Favorable Stable | Optimized sensor computation | 30.4 | 58.2 | 100.4 | 72/72 |
| Favorable Stable | Model front backbone | 1.3 | 4.4 | 11.1 | 72/72 |
| Favorable Stable | UE action after tensor ready | 25.3 | 54.0 | 76.1 | 72/72 |
| Favorable Stable | Optimized UE capture-to-send path | 55.5 | 120.6 | 175.0 | 72/72 |
| Favorable Stable | Feature uplink | 55.6 | 93.3 | 137.2 | 64/72 |
| Favorable Stable | Tail-busy wait | 0.0 | 43.8 | 75.7 | 71/72 |
| Favorable Stable | FCOS model tail | 21.2 | 24.8 | 29.5 | 72/72 |
| Favorable Stable | Other tail processing | 32.0 | 61.5 | 111.5 | 72/72 |
| Favorable Stable | Map service | 3.9 | 9.6 | 31.5 | 72/72 |
| Favorable Stable | Complete edge-to-map service | 60.3 | 143.5 | 198.8 | 71/72 |
| Favorable Stable | RGB-capture-to-map total | 182.4 | 280.1 | 330.9 | 71/72 |
| Mid Variable | Optimized sensor computation | 30.1 | 57.7 | 99.3 | 72/72 |
| Mid Variable | Model front backbone | 1.3 | 4.3 | 10.1 | 72/72 |
| Mid Variable | UE action after tensor ready | 25.4 | 53.8 | 77.4 | 72/72 |
| Mid Variable | Optimized UE capture-to-send path | 54.4 | 118.8 | 172.7 | 72/72 |
| Mid Variable | Feature uplink | 65.6 | 108.4 | 154.0 | 56/72 |
| Mid Variable | Tail-busy wait | 0.0 | 39.1 | 74.4 | 62/72 |
| Mid Variable | FCOS model tail | 21.2 | 24.8 | 29.5 | 72/72 |
| Mid Variable | Other tail processing | 32.0 | 61.5 | 111.5 | 72/72 |
| Mid Variable | Map service | 3.9 | 9.6 | 31.5 | 72/72 |
| Mid Variable | Complete edge-to-map service | 59.4 | 136.1 | 193.8 | 62/72 |
| Mid Variable | RGB-capture-to-map total | 192.5 | 290.4 | 336.5 | 62/72 |
| Adverse Stable | Optimized sensor computation | 29.6 | 53.7 | 98.4 | 72/72 |
| Adverse Stable | Model front backbone | 1.3 | 4.0 | 8.9 | 72/72 |
| Adverse Stable | UE action after tensor ready | 24.0 | 50.6 | 72.2 | 72/72 |
| Adverse Stable | Optimized UE capture-to-send path | 53.6 | 115.3 | 162.5 | 72/72 |
| Adverse Stable | Feature uplink | 64.5 | 95.5 | 128.5 | 43/72 |
| Adverse Stable | Tail-busy wait | 0.0 | 40.4 | 75.7 | 46/72 |
| Adverse Stable | FCOS model tail | 21.2 | 24.8 | 29.5 | 72/72 |
| Adverse Stable | Other tail processing | 32.0 | 61.5 | 111.5 | 72/72 |
| Adverse Stable | Map service | 3.9 | 9.6 | 31.5 | 72/72 |
| Adverse Stable | Complete edge-to-map service | 58.7 | 137.4 | 195.8 | 46/72 |
| Adverse Stable | RGB-capture-to-map total | 181.3 | 276.1 | 331.2 | 46/72 |
| Fade Recovery | Optimized sensor computation | 30.9 | 59.6 | 100.4 | 72/72 |
| Fade Recovery | Model front backbone | 1.4 | 4.5 | 10.9 | 72/72 |
| Fade Recovery | UE action after tensor ready | 25.4 | 53.9 | 77.5 | 72/72 |
| Fade Recovery | Optimized UE capture-to-send path | 56.1 | 123.2 | 175.2 | 72/72 |
| Fade Recovery | Feature uplink | 60.6 | 104.9 | 140.1 | 59/72 |
| Fade Recovery | Tail-busy wait | 0.0 | 38.8 | 73.5 | 63/72 |
| Fade Recovery | FCOS model tail | 21.2 | 24.8 | 29.5 | 72/72 |
| Fade Recovery | Other tail processing | 32.0 | 61.5 | 111.5 | 72/72 |
| Fade Recovery | Map service | 3.9 | 9.6 | 31.5 | 72/72 |
| Fade Recovery | Complete edge-to-map service | 59.3 | 136.8 | 194.1 | 63/72 |
| Fade Recovery | RGB-capture-to-map total | 194.7 | 285.8 | 333.0 | 63/72 |

Percentile aggregation is action-balanced: each displayed value is the median of the corresponding per-action cell percentile. It is not a pooled-frame percentile dominated by high-throughput actions.
The stage percentiles are marginals with different conditional denominators and must not be added to reconstruct an end-to-end percentile. Figure 04 and the `total_*` columns provide the causally simulated RGB-capture-to-map distribution.

## Scheduling and causal boundaries

- Optimized sensor computation excludes waiting for CARLA to produce a synchronized sample. It covers the measured production preparation work before the action starts.
- UE action-after-tensor-ready runs from `capture_started_ns` through `send_finished_ns`: front inference, ranker/selection, compression/packing, serialization, and the UDP send loop.
- Optimized UE capture-to-send runs from the RGB callback timestamp through the shifted send completion, so it includes the pre-action preparation delay as well as the UE action path.
- Feature uplink runs from UE send completion to complete edge reassembly. Only retained same-clock observed receipts are plotted; imputed arrivals used by the scheduler are excluded from this metric.
- Latest-only scheduling removes a multi-frame FIFO but cannot preempt a running CUDA/tail call. At most one newest frame waits; older pending frames receive explicit `SUPERSEDED_PENDING` outcomes.
- Edge-to-map includes tail-busy waiting, edge processing/publication, and renderer-off map service. RGB-capture-to-map includes the entire causal path used for physical freshness.

## Sensor optimization

Figure 06 uses the full sent-frame populations from the live action-50 FAVORABLE_STABLE baseline and optimized cells. It reports P07–P25 at their measured function boundaries. Camera and radar callbacks are distinct, but the numerical preparation for one selected frame remains sequential in the front worker. Component percentiles are marginal and cannot be summed.

## Clock and denominator integrity

The same-host clock bridge used 344,177 anchors; its absolute error P99 was 0.001099 ms.
Every stage carries its own count. Sensor, pure-front, UE-action, and UE-pipeline timing use all sent frames. Feature-uplink timing uses only retained observed complete edge receipts. Edge and total timing use useful direct-map installations.

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

The overlap terms measure spatial box/footprint agreement; they are not class-specific semantic-segmentation IoUs. Centroid XY MAE is explicit through a smooth one-metre reference scale. The one-metre value normalizes the presentation coordinate and is not a correctness gate. Geometric means are conservative: one strong dimension cannot hide a weak one. `Q_joint` is a presentation coordinate, not a calibrated probability and not yet the PPO reward.

## Figure guide

Figures 01a–01f show quality against optimized RGB-capture-to-UE-send latency. Figures 02a–02f use observed feature-uplink latency. Figures 03a–03f use optimized edge-reassembly-to-map latency. Figures 04a–04f use optimized RGB-capture-to-map latency. Views a–e keep semantic mIoU, vehicle overlap, person box-mask overlap, vehicle centroid error, and person centroid error separate; view f restores joint model quality. Figure 05 gives action-balanced P50/P95/P99 causal-stage marginals. Figure 06 shows the measured live sensor function breakdown before and after optimization.

## Limitations

- Sensor optimization is anchored by one live action/profile because the sensor path is action-independent; run-to-run host variation remains possible.
- Newest edge medians are live for one action per family. The older hash-verified family distributions provide the residual shape because the newest publication ledger did not survive that validation run.
- Renderer-off map service is a single action-50 live pool and is intentionally treated as action-independent.
- The replay can estimate changed installation and freshness behavior under these measured transformations, but it is not a substitute for a new 288-cell live campaign.
