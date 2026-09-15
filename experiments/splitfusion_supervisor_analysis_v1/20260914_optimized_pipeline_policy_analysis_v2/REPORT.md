# SplitFusion optimized-pipeline latency and quality analysis

This is an offline causal replay of the immutable 288-cell sent-frame
population. It applies the subsequently live-validated sensor preparation,
repaired-v3 edge service, renderer-off direct edge-to-map service, and
latest-only scheduling. It is not a second 288-cell live campaign.

## What changed and what did not

- The measured radio reassembly/admission outcomes and observed per-frame transport delays are held fixed. No missing uplink frame is fabricated.
- Scheduler-only imputed capture-to-arrival samples are causally floored at shifted UE send completion; observed uplink timing is never imputed into the network plots.
- Sensor timing is changed by equal-percentile mapping from the contemporaneous live baseline distribution to the live optimized distribution; this is not a constant subtraction.
- The measured family service shapes are retained and rescaled to the newest repaired-v3 live edge-compute medians.
- Direct map service is sampled from the renderer-off live action-50 run. It is independent of split action and does not traverse the radio.
- Validation quality is action-dependent and unchanged by runtime optimization. Network profile changes delivery, latency, and freshness—not the offline quality anchor.
- A point is absent when its conditional stage has no trustworthy sample or no useful map installation; absence is never encoded as zero latency.
- Physical map age ends at authoritative map installation. The later compact UE feedback is excluded.

## Action-balanced stage percentiles

| Profile | Stage | P50 (ms) | P95 (ms) | P99 (ms) | Actions represented |
|---|---|---:|---:|---:|---:|
| Favorable Stable | Optimized sensor compute before concatenation | 30.1 | 55.1 | 96.8 | 72/72 |
| Favorable Stable | Model front backbone | 1.3 | 4.4 | 11.1 | 72/72 |
| Favorable Stable | 7-channel-concat-start-to-UDP-send path | 26.1 | 55.3 | 78.8 | 72/72 |
| Favorable Stable | Feature uplink | 47.1 | 68.1 | 92.4 | 43/72 |
| Favorable Stable | Tail-busy wait | 0.0 | 42.0 | 74.1 | 71/72 |
| Favorable Stable | FCOS model tail | 21.2 | 24.8 | 29.5 | 72/72 |
| Favorable Stable | Other tail processing | 32.0 | 61.5 | 111.5 | 72/72 |
| Favorable Stable | Map service | 3.9 | 9.6 | 31.5 | 72/72 |
| Favorable Stable | Complete edge-to-map service | 60.2 | 143.3 | 199.1 | 71/72 |
| Favorable Stable | Action-start-to-map total | 151.2 | 249.0 | 302.7 | 71/72 |
| Mid Variable | Optimized sensor compute before concatenation | 29.9 | 54.6 | 95.7 | 72/72 |
| Mid Variable | Model front backbone | 1.3 | 4.3 | 10.1 | 72/72 |
| Mid Variable | 7-channel-concat-start-to-UDP-send path | 26.1 | 55.2 | 79.4 | 72/72 |
| Mid Variable | Feature uplink | 53.3 | 82.3 | 111.0 | 43/72 |
| Mid Variable | Tail-busy wait | 0.0 | 39.3 | 74.1 | 62/72 |
| Mid Variable | FCOS model tail | 21.2 | 24.8 | 29.5 | 72/72 |
| Mid Variable | Other tail processing | 32.0 | 61.5 | 111.5 | 72/72 |
| Mid Variable | Map service | 3.9 | 9.6 | 31.5 | 72/72 |
| Mid Variable | Complete edge-to-map service | 59.2 | 135.8 | 192.2 | 62/72 |
| Mid Variable | Action-start-to-map total | 157.7 | 253.2 | 302.6 | 62/72 |
| Adverse Stable | Optimized sensor compute before concatenation | 29.4 | 50.6 | 94.7 | 72/72 |
| Adverse Stable | Model front backbone | 1.3 | 4.0 | 8.9 | 72/72 |
| Adverse Stable | 7-channel-concat-start-to-UDP-send path | 24.7 | 51.9 | 73.9 | 72/72 |
| Adverse Stable | Feature uplink | 64.5 | 95.5 | 128.5 | 43/72 |
| Adverse Stable | Tail-busy wait | 0.0 | 40.4 | 75.2 | 46/72 |
| Adverse Stable | FCOS model tail | 21.2 | 24.8 | 29.5 | 72/72 |
| Adverse Stable | Other tail processing | 32.0 | 61.5 | 111.5 | 72/72 |
| Adverse Stable | Map service | 3.9 | 9.6 | 31.5 | 72/72 |
| Adverse Stable | Complete edge-to-map service | 58.6 | 136.0 | 193.9 | 46/72 |
| Adverse Stable | Action-start-to-map total | 151.6 | 249.2 | 302.0 | 46/72 |
| Fade Recovery | Optimized sensor compute before concatenation | 30.6 | 56.5 | 96.8 | 72/72 |
| Fade Recovery | Model front backbone | 1.4 | 4.5 | 10.9 | 72/72 |
| Fade Recovery | 7-channel-concat-start-to-UDP-send path | 26.1 | 55.2 | 79.6 | 72/72 |
| Fade Recovery | Feature uplink | 52.2 | 75.9 | 96.1 | 43/72 |
| Fade Recovery | Tail-busy wait | 0.0 | 39.0 | 73.0 | 63/72 |
| Fade Recovery | FCOS model tail | 21.2 | 24.8 | 29.5 | 72/72 |
| Fade Recovery | Other tail processing | 32.0 | 61.5 | 111.5 | 72/72 |
| Fade Recovery | Map service | 3.9 | 9.6 | 31.5 | 72/72 |
| Fade Recovery | Complete edge-to-map service | 59.5 | 137.3 | 193.1 | 63/72 |
| Fade Recovery | Action-start-to-map total | 160.7 | 245.0 | 301.1 | 63/72 |

## Complete feature delivery

| Profile | Complete reassemblies | Frames sent | Delivery |
|---|---:|---:|---:|
| Favorable Stable | 198,092 | 223,662 | 88.6% |
| Mid Variable | 181,825 | 226,716 | 80.2% |
| Adverse Stable | 147,449 | 223,753 | 65.9% |
| Fade Recovery | 185,366 | 222,725 | 83.2% |

Percentile aggregation is action-balanced: each displayed value is the median of the corresponding per-action cell percentile. It is not a pooled-frame percentile dominated by high-throughput actions.
The stage percentiles are marginals with different conditional denominators and must not be added to reconstruct an end-to-end percentile. Figure 04 and the `total_*` columns provide the causally simulated seven-channel-concatenation-start-to-map distribution.

## Scheduling and causal boundaries

- Optimized sensor computation excludes waiting for CARLA to produce a synchronized sample and ends when seven-channel concatenation starts.
- The UE action path begins at seven-channel concatenation and continues through front inference, ranker/selection, compression/packing, serialization, and the UDP send loop. The 288 cells timestamp the boundary immediately after concatenation; the live optimized P23 distribution is therefore added explicitly.
- Feature uplink runs from UE send completion to complete edge reassembly. Only retained same-clock observed receipts are plotted; imputed arrivals used by the scheduler are excluded from this metric.
- Latest-only scheduling removes a multi-frame FIFO but cannot preempt a running CUDA/tail call. At most one newest frame waits; older pending frames receive explicit `SUPERSEDED_PENDING` outcomes.
- Edge-to-map includes tail-busy waiting, edge processing/publication, and renderer-off map service. Action-start-to-map excludes sensor preparation, as requested. Capture-to-map remains in `capture_total_*` for physical freshness accounting but is not used in Figures 01 or 04.

## Sensor optimization

Figure 06 uses the full sent-frame populations from the live action-50 FAVORABLE_STABLE baseline and optimized cells. Its presentation-local labels run sequentially from P01 to P18. The CSV retains each original profiler identifier in `source_stage`; those source identifiers are P07–P23 plus P25 because callback/wait, evaluation-only, and diagnostic synchronization stages are outside this production-compute breakdown. Camera and radar callbacks are distinct, but the numerical preparation for one selected frame remains sequential in the front worker. Component percentiles are marginal and cannot be summed.

## Clock and denominator integrity

The same-host clock bridge used 344,177 anchors; its absolute error P99 was 0.001099 ms.
Every stage carries its own count. Sensor, pure-front, and UE-action timing use all sent frames. Feature-uplink timing uses the same 43-action observed support in every profile. Edge and total timing use useful direct-map installations.
The scheduler causality floor affected 22,666 imputed arrivals. It prevents a replay-only feature arrival from preceding that frame's shifted send completion; it does not change measured reassembly/admission counts or enter the observed-uplink plots.

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

Figures 01a–01f show quality against the cross-profile-pooled seven-channel-concatenation-start-to-UDP-send latency. Figures 02a–02f use observed feature-uplink latency. Figures 03a–03f use optimized edge-reassembly-to-map latency. Figures 04a–04f use optimized seven-channel-concatenation-start-to-map latency. Views a–e keep semantic mIoU, vehicle overlap, person box-mask overlap, vehicle centroid error, and person centroid error separate; view f restores joint model quality. Figure 05 gives action-balanced P50/P95/P99 causal-stage marginals using common action support for uplink. Figure 06 shows the measured live sensor function breakdown before and after optimization. Figure 07 reports weighted complete feature delivery for each network profile.

## Limitations

- Sensor optimization is anchored by one live action/profile because the sensor path is action-independent; run-to-run host variation remains possible.
- Newest edge medians are live for one action per family. The older hash-verified family distributions provide the residual shape because the newest publication ledger did not survive that validation run.
- Renderer-off map service is a single action-50 live pool and is intentionally treated as action-independent.
- The replay can estimate changed installation and freshness behavior under these measured transformations, but it is not a substitute for a new 288-cell live campaign.
