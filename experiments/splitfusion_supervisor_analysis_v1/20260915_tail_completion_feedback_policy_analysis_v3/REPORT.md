# SplitFusion model-tail-completion latency and quality analysis

This is an offline causal replay of the immutable 288-cell sent-frame
population. It applies the subsequently live-validated sensor preparation,
repaired-v3 edge service, renderer-off direct edge-to-map service, and
latest-only scheduling. It is not a second 288-cell live campaign.

## What changed and what did not

- The measured radio reassembly/admission outcomes and observed per-frame transport delays are held fixed. No missing uplink frame is fabricated.
- Scheduler-only imputed capture-to-arrival samples are causally floored at shifted UE send completion; observed uplink timing is never imputed into the network plots.
- Sensor timing is changed by equal-percentile mapping from the contemporaneous live baseline distribution to the live optimized distribution; this is not a constant subtraction.
- Edge compute is composed from action/profile-specific pre-frozen work plus paired live family tail-to-ready and post-ready stages; only post-ready work is scaled to the newest repaired-v3 compute-only family medians.
- Direct map service is sampled from the renderer-off live action-50 run. It is independent of split action and does not traverse the radio.
- Validation quality is action-dependent and unchanged by runtime optimization. Network profile changes delivery, latency, and freshness—not the offline quality anchor.
- A point is absent when its conditional stage has no trustworthy sample; absence is never encoded as zero latency.
- Presentation P50 scatter points require at least 10 frame samples. Cross-profile P50/P95/P99 bars require one common action support with at least 100 samples per stage/profile. Lower-support values remain in the CSV.
- Physical map age ends at authoritative map installation. The later compact UE feedback is excluded.
- The proposed early endpoint ends at model-tail completion on the edge. It is a progress/feedback-ready boundary, not evidence that post-processing or map installation succeeded, and its return trip to the UE has not yet been measured.

## Action-balanced stage percentiles

| Profile | Stage | P50 (ms) | P95 (ms) | P99 (ms) | Actions represented |
|---|---|---:|---:|---:|---:|
| Favorable Stable | Optimized sensor compute before concatenation | 30.4 | 55.4 | 97.6 | 40/72 |
| Favorable Stable | Model front backbone | 1.3 | 4.5 | 11.5 | 40/72 |
| Favorable Stable | 7-channel-concat-start-to-UDP-send path | 24.2 | 52.6 | 76.7 | 40/72 |
| Favorable Stable | Feature uplink | 46.3 | 66.9 | 86.0 | 40/72 |
| Favorable Stable | Tail-busy wait | 0.0 | 25.9 | 53.3 | 40/72 |
| Favorable Stable | FCOS model tail | 21.2 | 24.8 | 36.6 | 40/72 |
| Favorable Stable | Other tail processing | 32.0 | 61.5 | 113.2 | 40/72 |
| Favorable Stable | Map service | 3.9 | 9.6 | 31.5 | 40/72 |
| Favorable Stable | Edge reassembly to model-tail completion | 30.4 | 66.4 | 95.0 | 40/72 |
| Favorable Stable | Sensor-compute start to model-tail completion (ACK return excluded) | 141.7 | 206.9 | 273.3 | 40/72 |
| Favorable Stable | Complete edge-to-map service | 60.4 | 107.6 | 161.9 | 40/72 |
| Favorable Stable | Action-start-to-map total | 138.2 | 201.2 | 255.2 | 40/72 |
| Mid Variable | Optimized sensor compute before concatenation | 30.2 | 56.5 | 97.6 | 40/72 |
| Mid Variable | Model front backbone | 1.3 | 4.3 | 11.1 | 40/72 |
| Mid Variable | 7-channel-concat-start-to-UDP-send path | 24.2 | 53.1 | 76.8 | 40/72 |
| Mid Variable | Feature uplink | 51.2 | 78.0 | 104.6 | 40/72 |
| Mid Variable | Tail-busy wait | 0.0 | 31.2 | 56.0 | 40/72 |
| Mid Variable | FCOS model tail | 21.2 | 24.8 | 36.6 | 40/72 |
| Mid Variable | Other tail processing | 32.0 | 61.5 | 113.2 | 40/72 |
| Mid Variable | Map service | 3.9 | 9.6 | 31.5 | 40/72 |
| Mid Variable | Edge reassembly to model-tail completion | 30.9 | 70.1 | 95.4 | 40/72 |
| Mid Variable | Sensor-compute start to model-tail completion (ACK return excluded) | 148.9 | 220.1 | 290.9 | 40/72 |
| Mid Variable | Complete edge-to-map service | 60.7 | 109.5 | 165.8 | 40/72 |
| Mid Variable | Action-start-to-map total | 146.2 | 207.6 | 268.6 | 40/72 |
| Adverse Stable | Optimized sensor compute before concatenation | 30.2 | 55.1 | 96.8 | 40/72 |
| Adverse Stable | Model front backbone | 1.4 | 4.3 | 10.3 | 40/72 |
| Adverse Stable | 7-channel-concat-start-to-UDP-send path | 24.7 | 51.9 | 74.1 | 40/72 |
| Adverse Stable | Feature uplink | 61.6 | 90.0 | 120.1 | 40/72 |
| Adverse Stable | Tail-busy wait | 0.0 | 29.7 | 58.5 | 40/72 |
| Adverse Stable | FCOS model tail | 21.2 | 24.8 | 36.6 | 40/72 |
| Adverse Stable | Other tail processing | 32.0 | 61.5 | 113.2 | 40/72 |
| Adverse Stable | Map service | 3.9 | 9.6 | 31.5 | 40/72 |
| Adverse Stable | Edge reassembly to model-tail completion | 30.1 | 68.7 | 98.8 | 40/72 |
| Adverse Stable | Sensor-compute start to model-tail completion (ACK return excluded) | 159.4 | 231.8 | 302.9 | 40/72 |
| Adverse Stable | Complete edge-to-map service | 60.4 | 108.4 | 165.8 | 40/72 |
| Adverse Stable | Action-start-to-map total | 154.9 | 220.6 | 282.1 | 40/72 |
| Fade Recovery | Optimized sensor compute before concatenation | 30.8 | 58.4 | 104.0 | 40/72 |
| Fade Recovery | Model front backbone | 1.4 | 4.6 | 11.4 | 40/72 |
| Fade Recovery | 7-channel-concat-start-to-UDP-send path | 24.5 | 53.4 | 77.4 | 40/72 |
| Fade Recovery | Feature uplink | 48.6 | 70.3 | 93.2 | 40/72 |
| Fade Recovery | Tail-busy wait | 0.0 | 29.6 | 57.2 | 40/72 |
| Fade Recovery | FCOS model tail | 21.2 | 24.8 | 36.6 | 40/72 |
| Fade Recovery | Other tail processing | 32.0 | 61.5 | 113.2 | 40/72 |
| Fade Recovery | Map service | 3.9 | 9.6 | 31.5 | 40/72 |
| Fade Recovery | Edge reassembly to model-tail completion | 30.5 | 68.0 | 98.3 | 40/72 |
| Fade Recovery | Sensor-compute start to model-tail completion (ACK return excluded) | 146.9 | 213.9 | 281.5 | 40/72 |
| Fade Recovery | Complete edge-to-map service | 60.3 | 108.0 | 165.1 | 40/72 |
| Fade Recovery | Action-start-to-map total | 141.8 | 204.5 | 262.9 | 40/72 |

## Delivery and completion outcomes

| Profile | Frames sent | Feature reassembled | Tail completed | Map installed |
|---|---:|---:|---:|---:|
| Favorable Stable | 223,662 | 88.6% | 77.8% | 77.8% |
| Mid Variable | 226,716 | 80.2% | 67.0% | 67.0% |
| Adverse Stable | 223,753 | 65.9% | 52.7% | 52.6% |
| Fade Recovery | 222,725 | 83.2% | 68.6% | 68.6% |

Percentile aggregation is action-balanced: each displayed value is the median of the corresponding per-action cell percentile. It is not a pooled-frame percentile dominated by high-throughput actions.
The stage percentiles are marginals with different conditional denominators and must not be added to reconstruct an end-to-end percentile. Figure 04 and the `sensor_model_ready_*` columns provide the causally simulated production-sensor-compute-start-to-model-tail-completion distribution.

## Scheduling and causal boundaries

- Optimized sensor computation excludes waiting for CARLA to produce a synchronized sample and ends when seven-channel concatenation starts.
- The UE action path begins at seven-channel concatenation and continues through front inference, ranker/selection, compression/packing, serialization, and the UDP send loop. The 288 cells timestamp the boundary immediately after concatenation; the live optimized P23 distribution is therefore added explicitly.
- Feature uplink runs from UE send completion to complete edge reassembly. Only retained same-clock observed receipts are plotted; imputed arrivals used by the scheduler are excluded from this metric.
- Latest-only scheduling removes a multi-frame FIFO but cannot preempt a running CUDA/tail call. At most one newest frame waits; older pending frames receive explicit `SUPERSEDED_PENDING` outcomes.
- Model-tail completion includes latest-only tail-busy waiting plus the feature reconstruction required before the model can run: zstd decompression, unpack/dequantization, optional AE decoding, camera-pose reconstruction, and synchronized `decode_tail`. It excludes camera-aware post-processing, p025 filtering, serialization, publication and map service.
- Figure 04 begins at production sensor-compute start and ends at model-tail completion on the edge. It therefore answers the proposed early-control question, but does not include the still-unimplemented compact progress-ACK return trip to the UE.
- The older map-install endpoints remain in the CSV as `edge_map_*`, `total_*`, and `capture_total_*`; they are not substituted with an early success claim.

## Sensor optimization

Figure 06 uses the full sent-frame populations from the live action-50 FAVORABLE_STABLE baseline and optimized cells. Its presentation-local labels run sequentially from P01 to P18. The CSV retains each original profiler identifier in `source_stage`; those source identifiers are P07–P23 plus P25 because callback/wait, evaluation-only, and diagnostic synchronization stages are outside this production-compute breakdown. Camera and radar callbacks are distinct, but the numerical preparation for one selected frame remains sequential in the front worker. Component percentiles are marginal and cannot be summed.

## Clock and denominator integrity

The same-host clock bridge used 344,177 anchors; its absolute error P99 was 0.001099 ms.
Every action-level cross-profile percentile in this report and Figure 05 uses the same 40-action support across all stages, with at least 100 frame samples per stage/profile. The per-cell CSV still retains every available sample and its actual denominator; Figures 02--04 require 10 samples for a displayed P50 and label low-support and absent outcomes rather than silently treating them as zero.
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

Figures 01a–01f show quality against the cross-profile-pooled seven-channel-concatenation-start-to-UDP-send latency. Figures 02a–02f use observed feature-uplink latency. Figures 03a–03f use edge-reassembly-to-model-tail-completion latency, including latest-only tail-busy waiting and required reconstruction. Figures 04a–04f use production-sensor-compute-start-to-model-tail-completion latency. Views a–e keep semantic mIoU, vehicle overlap, person box-mask overlap, vehicle centroid error, and person centroid error separate; view f restores joint model quality. Figure 05 gives action-balanced P50/P95/P99 causal-stage marginals on one common action support. Figure 06 shows the measured live sensor function breakdown before and after optimization. Figure 07 reports measured feature delivery plus counterfactual tail-completion and map-install rates for each network profile.

## Limitations

- Sensor optimization is anchored by one live action/profile because the sensor path is action-independent; run-to-run host variation remains possible.
- Newest edge medians are live for one action per family. The older hash-verified family distributions provide the residual shape because the newest publication ledger did not survive that validation run.
- Renderer-off map service is a single action-50 live pool and is intentionally treated as action-independent.
- Model-tail completion is reconstructed from each original action/profile's pre-frozen timing plus paired live family tail-to-ready and calibrated post-ready work because the retained 288 cells have no direct timestamp at that boundary. It must be confirmed by a short live early-ACK experiment before deployment claims.
- The NoAE per-frame internal-stage shape comes from the v2 diagnostic and is total-calibrated to the repaired-v3 live family median. A repaired-v3 NoAE per-frame decomposition remains a live-validation item.
- Of 602,315 scheduler arrivals, 258,138 use within-cell/action/profile imputation for replay ordering. They are excluded from observed-uplink plots, but Figure 04 remains a counterfactual rather than a directly observed end-to-end distribution.
- The pure FCOS `decode_tail` duration alone is not the edge contribution: the compressed feature must be reconstructed before the tail can execute.
- The replay can estimate changed installation and freshness behavior under these measured transformations, but it is not a substitute for a new 288-cell live campaign.
