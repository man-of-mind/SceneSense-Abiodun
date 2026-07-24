# OAI latency investigation deck talk track

Deck: `oai_layer_latency/slides/oai_latency_investigation_20260723.pptx`

## One-minute presentation version

We reran the OAI latency study with the corrected CARLA drivable route and focused on the live split-fusion path. The result is asymmetric: the downlink is cheap because it only returns compact detections — boxes, centroids, scores — but the uplink sends a dense split-feature tensor of roughly 1 MB/frame in the no-AE profile. Layer timestamps show the long delay is not edge inference or PDCP handoff; it is UE RLC queue-wait while the burst drains over the OAI uplink. The advisor check is now clean: iperf and CARLA are both observed on LCID 4 / LCG 1 with high RFsim gNB PUSCH SNR around 50.5 dB, so we do not see a special traffic-class explanation. The pre/post scheduler trace also ruled out the PHR helper: selected, pre-PHR, post-PHR, and final MCS were already low. The final direct BLER/OLLA trace shows the mechanism: at observed closed-loop pace, the selector spends about 79% of update decisions in the few-scheduled-samples branch and stays at MCS 4/8; at open-loop 10 FPS, enough scheduled windows arrive to keep ratcheting MCS to 23/25. Fixed MCS28 collapses RTT from about 186 ms to 47 ms, proving spectral efficiency is the lever, but it is only a diagnostic control. The deployable mitigation is payload reduction: the AE-128/u6/ROI0.5 profile cuts the feature burst to about 153 KB, gives 99.8% overall delivery, and after startup it delivered every frame.

## Main caveats

- Do not present fixed MCS28 as the deployment fix; it can fail under realistic fading.
- Do not overstate CQI/RSRP from the current extraction. The live CQI/RSRP fields looked unreliable; the deck uses gNB PUSCH SNR plus OAI's SINR→MCS scheduler table.
- Fixed MCS28 is diagnostic, not the deployment policy. The scheduler-side path to investigate is BLER/OLLA behavior in `get_mcs_from_bler()` for sparse low-latency bursts.

## Selected MCS table

| MCS | Modulation | Qm | R x1024 | Efficiency bits/RE |
|---:|---|---:|---:|---:|
| 0 | QPSK | 2 | 120 | 0.2344 |
| 2 | QPSK | 2 | 193 | 0.3770 |
| 4 | QPSK | 2 | 308 | 0.6016 |
| 5 | QPSK | 2 | 379 | 0.7402 |
| 8 | QPSK | 2 | 602 | 1.1758 |
| 10 | 16QAM | 4 | 340 | 1.3281 |
| 16 | 16QAM | 4 | 658 | 2.5703 |
| 20 | 64QAM | 6 | 567 | 3.3223 |
| 24 | 64QAM | 6 | 772 | 4.5234 |
| 28 | 64QAM | 6 | 948 | 5.5547 |

## OAI SINR→MCS scheduler thresholds

Source: `gNB_scheduler_primitives.c` `SINRx10_MCS_mapping`; OAI comment says the table targets BLER around `10^-3`.

| MCS | SINR threshold | Margin at measured 50.5 dB | Efficiency bits/RE |
|---:|---:|---:|---:|
| 0 | -1.0 dB | +51.5 dB | 0.2344 |
| 4 | 2.4 dB | +48.1 dB | 0.6016 |
| 8 | 5.6 dB | +44.9 dB | 1.1758 |
| 16 | 12.4 dB | +38.1 dB | 2.5703 |
| 24 | 19.4 dB | +31.1 dB | 4.5234 |
| 28 | 24.5 dB | +26.0 dB | 5.5547 |

## Source artifacts used

- `downlink_latency_fps/plots/oai_bottleneck/corrected_transport_latency_breakdown.png`
- `downlink_latency_fps/plots/oai_bottleneck/corrected_transport_reliability_rtt.png`
- `downlink_latency_fps/plots/oai_bottleneck/oai_106prb_drivable_zlib_vs_zstd.png`
- `oai_layer_latency/plots/complementary_latency_summary.png`
- `oai_layer_latency/plots/complementary_mcs_prb_summary.png`
- `oai_layer_latency/plots/complementary_rlc_buffer_timeseries.png`
- `oai_layer_latency/plots/complementary_gnb_snr_timeseries.png`
- `oai_layer_latency/plots/uplink_mcs_bottleneck_summary.png`
- `oai_layer_latency/plots/advisor_iperf_vs_carla_bsr_mcs.png`
- `oai_layer_latency/plots/uplink_drain_rate.png`
- `oai_layer_latency/plots/bler_olla_branch_comparison.png`
- `oai_layer_latency/plots/bler_olla_mcs_timeseries.png`
- `oai_layer_latency/plots/bler_olla_num_sched_timeseries.png`
- `metrics_logs/scenesense_ttracer/carla_shape_udp_bw273_mcsdecision_20260723_171153/gnb/analysis/mcs_decision_timeseries.png`
- `metrics_logs/scenesense_ttracer/carla_shape_udp_bw273_mcsdecision_20260723_171153/gnb/analysis/mcs_decision_phr_drop_hist.png`
- `oai_layer_latency/plots/complementary_experiment_summary.csv`
- `OAI/openairinterface5g/openair2/LAYER2/NR_MAC_gNB/gNB_scheduler_ulsch.c`
- `OAI/openairinterface5g/openair2/LAYER2/NR_MAC_gNB/gNB_scheduler_primitives.c`

## Numeric rows loaded into the deck

| Condition | Payload KB | Delivery | RTT p50 | RAN UL p50 | RLC queue | MCS p50/p95 | PRB p50 | SNR p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 273 adaptive uint8 | 1054.1 | 76.5% | 186.0 ms | 112.2 ms | 103.3 ms | 4/8 | 273 | 50.5 dB |
| 273 adaptive uint4 | 394.6 | 99.8% | 112.7 ms | 61.7 ms | 54.4 ms | 2/5 | 273 | 50.5 dB |
| 273 fixed MCS28 | 1055.6 | 73.8% | 47.0 ms | 16.8 ms | 12.7 ms | 28/28 | 273 | 50.5 dB |
| 106 fixed MCS28 | 1051.6 | 72.5% | 47.2 ms | 14.7 ms | 5.9 ms | 28/28 | 106 | 50.5 dB |
| 106 default AE128 u6 r0.5 | 152.7 | 99.8% | 64.2 ms | 33.6 ms | 29.3 ms | 2/5 | 106 | 50.5 dB |
