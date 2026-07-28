# RFsim AWGN hold-MCS diagnostic — vanilla OAI vs hold-few-samples logic

Date: 2026-07-27

Purpose: check whether the `SCENESENSE_HOLD_MCS_FEW_SAMPLES=1` logic still behaves safely when RFsim AWGN channel modeling creates a genuinely imperfect channel, instead of the near-zero-BLER ideal RFsim channel.

## Setup

- RAN: 273 PRB RFsim, AWGN channelmod enabled.
- gNB config wrapper: `OAI/openairinterface5g/targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.273PRB.scenesense_rfsim.awgn.conf`
- UE config: `OAI/openairinterface5g/targets/PROJECTS/GENERIC-NR-5GC/CONF/ue.awgn.conf`
- Channel model list: `channelmod_rfsimu.conf`
- App: CARLA no-AE baseline, zstd, ROI 0.0, per-channel uint8, ~1 MB feature payload.
- Requested load: short diagnostic, 300 requested frames at nominal 10 FPS.
  Because the CARLA frontend is still closed-loop/wait-for-result, the actual
  wall-clock trace is much longer than 30 s when OAI RTT/timeouts are large
  (`~250–260 s` in the application metric timeline).
- Runner: `oai_layer_latency/run_awgn_vanilla_vs_hold_273prb.sh`

## Initial short run

| Logic | Run group | HOLD flag |
|---|---|---:|
| Vanilla OAI | `downlink_oai_bw273_awgn_vanilla_fps10_awgn273_short_20260727_194554_vanilla` | 0 |
| Hold few samples | `downlink_oai_bw273_awgn_hold_fps10_awgn273_short_20260727_194554_hold` | 1 |

## Main result

| Metric | Vanilla OAI | Hold few samples |
|---|---:|---:|
| Requested frames | 300 | 300 |
| Returned results | 252 | 256 |
| Delivery | 84.0% | 85.3% |
| Median front | 24.2 ms | 24.2 ms |
| Median uplink handling | 260.0 ms | 219.3 ms |
| Median edge back-half | 7.6 ms | 7.2 ms |
| Median downlink | 7.7 ms | 7.2 ms |
| Median post-send RTT | 281.5 ms | 235.3 ms |
| Estimated capture→result | 305.7 ms | 259.5 ms |
| Median UL scheduled rate | 9.1 Mbps | 12.6 Mbps |
| Median active-window MCS | 18.2 | 25.3 |
| Mean UL retx rate | 1.15% | 6.43% |
| gNB PUSCH SNR p50 | 20.0 dB | 20.0 dB |

## Why MCS ~25 still gives >200 ms latency here

This AWGN run is **not** directly comparable to the earlier clean-RFsim fixed
MCS28 diagnostic that reached about `74 ms` capture→result. The earlier fixed
MCS28 result was under the near-ideal RFsim channel, with near-zero
retransmission pressure and MCS pinned at 28. In this AWGN run, the median gNB
PUSCH SNR is around `20 dB`, MCS is lower (`~25` for hold-few-samples), and the
link now has real retransmission pressure.

The delivered-frame latency is dominated by waiting for the uplink feature burst
to finish and for the result to return:

| Metric | Vanilla OAI | Hold few samples |
|---|---:|---:|
| Delivered post-send RTT p50 / p95 | 281.5 / 599.0 ms | 235.3 / 325.0 ms |
| Result-wait p50 | 269.5 ms | 228.8 ms |
| Downlink p50 | 7.7 ms | 7.2 ms |
| Timeouts | 48 / 300 | 44 / 300 |
| Scheduled TBS drain p50 / p95 | 11.6 / 21.6 Mbps | 18.3 / 23.3 Mbps |
| Decoded LCID4 drain p50 / p95 | 10.3 / 17.7 Mbps | 16.6 / 20.5 Mbps |

So the right explanation is: **high median MCS helped, but AWGN retransmissions
and unstable effective drain still kept the 1 MB CARLA feature burst in the
hundreds-of-ms regime.** Downlink remains cheap; the extra time is still mostly
the uplink/result-wait path.

## RLC/drain evidence

The new drain plot compares:

- scheduled uplink TBS at the UE grant level, and
- decoded LCID4 uplink data at the gNB MAC/RLC boundary.

This is useful as a **drain proxy**: it shows how much useful uplink data is
being scheduled and decoded over time. In this AWGN run, the hold-few-samples
logic increases median effective drain, but retransmission spikes make the drain
less stable than the clean-channel fixed-MCS case.

Important caveat on this initial short run: it did **not** record the UE
queue/BSR profile (`NRUE_MAC_RLC_BUFFER_STATUS` / BSR events were not captured).
The all-metrics rerun below supersedes this limitation and provides the direct
UE RLC/BSR evidence.

## Interpretation

- AWGN channelmod worked: unlike the previous ideal RFsim traces, retransmission pressure and nonzero BLER appeared.
- The hold-few-samples logic kept MCS much higher and reduced median latency by about 46 ms capture→result.
- However, the higher MCS came with meaningfully higher retransmission pressure: ~6.4% mean UL retx vs ~1.1% for vanilla.
- This means the first hold patch is not yet a safe final policy for degraded channels. It helps the bursty-good-channel case, but under AWGN it can hold MCS high even when the channel is showing real retransmission/BLER evidence.

## Policy implication

The safer next logic should not simply hold whenever `num_sched <= 3`.

Candidate rule:

1. If filtered BLER is above the upper threshold, decrement MCS even if the update window has few samples.
2. Else if the window has too few samples and no retransmission evidence, hold the previous MCS.
3. Else if BLER is below the lower threshold, increase MCS.
4. Else hold within the target window.

In short: hold only for sparse, clean evidence — not for sparse bad-channel evidence.

## All-metrics rerun: why high MCS still did not reach clean-channel latency

After the first AWGN plots were inconclusive, we reran the same pair with:

- UE T-tracer profile `all`: PHY/MCS + BSR/RLC queue + PDCP/RLC timestamp events.
- gNB T-tracer profile `latency`: full gNB panel + BLER/OLLA MCS decision events.
- Same 273PRB AWGN setup, no-AE zstd ROI0 uint8 payload, 300 requested frames.

| Logic | Run group |
|---|---|
| Vanilla OAI | `downlink_oai_bw273_awgn_vanilla_fps10_awgn273_allmetrics_20260727_202430_vanilla` |
| Hold few samples | `downlink_oai_bw273_awgn_hold_fps10_awgn273_allmetrics_20260727_202430_hold` |

| Metric | Vanilla OAI | Hold few samples |
|---|---:|---:|
| Returned / requested frames | 224 / 300 | 218 / 300 |
| Median capture→result | 318.9 ms | 269.7 ms |
| Median post-send RTT | 294.6 ms | 243.5 ms |
| Median feature-upload/result-wait path | 280.2 ms | 225.2 ms |
| Median downlink | 6.1 ms | 7.8 ms |
| Median active-window MCS | 15.7 | 25.1 |
| Mean UL retx rate | 1.17% | 7.24% |
| BLER updates with retransmission evidence | 2.8% | 18.3% |
| High-BLER decrease branch share | 1.2% | 9.2% |
| UE RLC queue wait, mean | 191 ms | 130 ms |
| UE PDCP-ingress→gNB PDCP-deliver mean | 193 ms | 134 ms |

Interpretation:

- Hold-few-samples **does help**: it raises MCS and reduces mean UE RLC queue
  wait by about `61 ms`.
- It does **not** recover the clean-RFsim fixed-MCS28 latency because AWGN makes
  the higher MCS expensive: retransmission evidence jumps from `2.8%` to
  `18.3%` of BLER update windows, and high-BLER decrease decisions become much
  more common.
- The bottleneck is still mostly UE RLC waiting/drain. The all-metrics analyzer
  attributes `~97%` of the hold-MCS RAN uplink transit to RLC queue wait.
- The frame-level application latency is higher than the median per-SDU RAN
  transit because the CARLA frame is split into many chunks; the edge cannot run
  until the tail chunks arrive. Retransmissions and bursty drain therefore show
  up strongly in `feature_upload_payload_handling_ms`.

So the corrected answer is: **high MCS alone is not sufficient. We need high MCS
only when the channel can support it, plus stable drain of the whole feature
burst.** The current hold-few-samples patch is good for the clean sparse-burst
case, but under AWGN it can be too aggressive.

## Artifacts

- Summary CSV: `oai_layer_latency/plots/awgn_vanilla_vs_hold_summary.csv`
- Summary plot: `oai_layer_latency/plots/awgn_vanilla_vs_hold_summary.png`
- Summary plot PDF: `oai_layer_latency/plots/awgn_vanilla_vs_hold_summary.pdf`
- Timeseries plot: `oai_layer_latency/plots/awgn_vanilla_vs_hold_timeseries.png`
- Timeseries plot PDF: `oai_layer_latency/plots/awgn_vanilla_vs_hold_timeseries.pdf`
- SNR/MCS/retransmission plot: `oai_layer_latency/plots/awgn_snr_mcs_retx_timeseries.png`
- SNR/MCS/retransmission plot PDF: `oai_layer_latency/plots/awgn_snr_mcs_retx_timeseries.pdf`
- Drain/latency/retransmission plot: `oai_layer_latency/plots/awgn_drain_latency_retx_timeseries.png`
- Drain/latency/retransmission plot PDF: `oai_layer_latency/plots/awgn_drain_latency_retx_timeseries.pdf`
- Drain/latency/retransmission summary CSV: `oai_layer_latency/plots/awgn_drain_latency_retx_summary.csv`
- All-metrics summary CSV: `oai_layer_latency/plots/awgn_allmetrics_summary.csv`
- All-metrics summary bars: `oai_layer_latency/plots/awgn_allmetrics_summary_bars.png`
- All-metrics summary bars PDF: `oai_layer_latency/plots/awgn_allmetrics_summary_bars.pdf`
- All-metrics scheduler plot: `oai_layer_latency/plots/awgn_allmetrics_scheduler_timeseries.png`
- All-metrics scheduler plot PDF: `oai_layer_latency/plots/awgn_allmetrics_scheduler_timeseries.pdf`
- All-metrics queue/grant plot: `oai_layer_latency/plots/awgn_allmetrics_queue_grant_timeseries.png`
- All-metrics queue/grant plot PDF: `oai_layer_latency/plots/awgn_allmetrics_queue_grant_timeseries.pdf`
