# Complementary OAI layer-latency experiments

Date: 2026-07-23

Purpose: complement the earlier OAI layer-localization result by testing:

1. whether a smaller validated no-AE payload reduces UE RLC queueing under normal
   OAI adaptive/link-adaptation behavior; and
2. whether fixed MCS28 produces the same low-latency behavior on both 273PRB and
   106PRB 4DL/5UL; and
3. whether a much smaller validated AE-128 payload on the default 106PRB path
   removes the remaining delivery/queueing issue without changing the OAI config.

All runs use the corrected drivable CARLA route, the same fusion checkpoint,
200k radar PPS, zstd entropy coding, and the latency T-tracer profile. Most
rows are no-AE/ROI `0.0`; the final row explicitly uses AE-128, uint6, ROI
`0.5`.

## Matrix profile confirmation

The per-model knob matrix validates the smaller payload profile:

| Profile | Offline payload | mIoU | veh IoU | ped recall | obj recall | loc MAE | ped-loc MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| no-AE, uint4, ROI 0.0 | 387.6 KB | 0.838 | 0.931 | 0.843 | 0.875 | 1.05 m | 1.22 m |
| AE-128, uint6, ROI 0.5 | 154.6 KB | 0.714 | 0.627 | 0.881 | 0.910 | 0.94 m | 1.15 m |

The no-AE uint4 live OAI run measured the same operating point at `394.6 KB`
p50, `7` UDP chunks. The AE-128/uint6/ROI0.5 run measured `152.7 KB` p50, `3`
UDP chunks, matching the matrix scale. Small differences are expected because
the live runs use live CARLA payloads and the current zstd path.

## Result summary

| Condition | Payload p50 | Delivery | App RTT p50 / p95 | RAN UL p50 / p95 | RLC mean queue | RLC p95 occupancy | MCS p50 / p95 | PRB p50 | TBS p50 | gNB PUSCH SNR p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 273PRB adaptive, uint8 | 1054.1 KB | 76.5% | 186.0 / 203.4 ms | 112.2 / 162.7 ms | 103.3 ms | 942.9 KB | 4 / 8 | 273 | 1697 B | 50.5 dB |
| 273PRB adaptive, uint4 | 394.6 KB | 99.8% | 112.7 / 127.0 ms | 61.7 / 94.9 ms | 54.4 ms | 353.0 KB | 2 / 5 | 273 | 1441 B | 50.5 dB |
| 273PRB fixed MCS28, uint8 | 1055.6 KB | 73.8% | 47.0 / 57.9 ms | 16.8 / 26.1 ms | 12.7 ms | 0.0 KB | 28 / 28 | 273 | 5637 B | 50.5 dB |
| 106PRB 4DL/5UL fixed MCS28, uint8 | 1051.6 KB | 72.5% | 47.2 / 62.7 ms | 14.7 / 25.4 ms | 5.9 ms | 0.0 KB | 28 / 28 | 106 | 8961 B | 50.5 dB |
| 106PRB default adaptive, AE-128 uint6 ROI0.5 | 152.7 KB | 99.8% | 64.2 / 76.8 ms | 33.6 / 50.6 ms | 29.3 ms | 118.6 KB | 2 / 5 | 106 | 561 B | 50.5 dB |

## Interpretation

1. **Payload reduction helps exactly where expected.**
   - Reducing no-AE payload from ~`1.05 MB` to ~`395 KB` reduces app RTT p50
     from `186 ms` to `113 ms`.
   - True RAN uplink p50 drops from `112 ms` to `62 ms`.
   - RLC mean queueing drops from `103 ms` to `54 ms`.
   - Delivery improves sharply from `76.5%` to `99.8%`.

2. **But adaptive OAI still selects very low MCS under CARLA bursts.**
   - The uint4 run does **not** make MCS recover; MCS is still QPSK-region and
     even lower than the uint8 adaptive run (`2/5` p50/p95 versus `4/8`).
   - This means int4 is a strong payload/queue relief knob, not a fix for the
     OAI link-adaptation behavior.

3. **AE-128 + uint6 + ROI0.5 largely removes the steady-state delivery issue on default 106PRB.**
   - Payload drops to `152.7 KB` p50 / `3` UDP chunks.
   - App RTT p50 drops to `64.2 ms`; measured RAN UL p50 drops to `33.6 ms`.
   - RLC p95 occupancy falls to `118.6 KB`, roughly one compressed feature
     frame rather than a multi-frame backlog.
   - The two missed frames were the first two startup frames; after warmup,
     frames `3..1300` were delivered (`1298/1298`), so the remaining loss is
     not a steady-state OAI transport pattern in this run.

4. **Fixed MCS28 collapses RLC queueing on both bandwidths.**
   - 273PRB fixed MCS28: RAN UL p50 `16.8 ms`, RLC queue `12.7 ms`.
   - 106PRB 4DL/5UL fixed MCS28: RAN UL p50 `14.7 ms`, RLC queue `5.9 ms`.
   - The 106PRB fixed-MCS run is not worse than 273PRB in application RTT for
     this closed-loop workload.

5. **SNR is not the cause in RFsim.**
   - gNB PUSCH SNR is flat at about `50.5 dB` in all runs.
   - The adaptive low-MCS behavior is therefore not explained by channel quality
     in this RFsim setup.

6. **Fixed MCS28 remains a diagnostic override, not the deployment fix.**
   - It proves the controlling lever is UL MCS/link adaptation.
   - In real fading/Sionna channels, a blanket fixed MCS28 could raise BLER.
   - The advisor follow-up trace later showed the PHR helper did not reduce MCS
     in the observed-cadence run; the real follow-up should target OAI's
     BLER/OLLA MCS-selection behavior under sparse bursty uplink traffic.

## Presentation plots

- `plots/complementary_latency_summary.pdf`
  - Main story: adaptive uint4 reduces queue/latency; fixed MCS28 collapses it.
- `plots/complementary_mcs_prb_summary.pdf`
  - Shows adaptive CARLA remains low-MCS while fixed controls pin MCS28.
- `plots/complementary_rlc_buffer_timeseries.pdf`
  - Shows whole-frame RLC buildup for adaptive uint8, smaller buildup for uint4,
    near-zero p95 occupancy for fixed-MCS runs, and much smaller AE-128
    occupancy on default 106PRB.
- `plots/complementary_gnb_snr_timeseries.pdf`
  - Shows flat RFsim gNB PUSCH SNR, so channel quality is not the explanation.

## Artifacts

- Summary CSV:
  `plots/complementary_experiment_summary.csv`
- 273PRB adaptive uint4 run:
  `../metrics_logs/scenesense_ttracer/downlink_oai_bw273_mu1_ttracer_int4_adaptive_fps10_int4_adaptive_20260723/`
- 273PRB fixed MCS28 run:
  `../metrics_logs/scenesense_ttracer/downlink_oai_bw273_mu1_ttracer_forcemcs28_fps10_forcemcs28_bw273_20260723/`
- 106PRB 4DL/5UL fixed MCS28 run:
  `../metrics_logs/scenesense_ttracer/downlink_oai_ulheavy_106_ttracer_forcemcs28_fps10_forcemcs28_ulheavy106_20260723/`
- 106PRB default adaptive AE-128/uint6/ROI0.5 run:
  `../metrics_logs/scenesense_ttracer/downlink_oai_default106_ttracer_ae128_u6_roi05_fps10_ae128_u6_roi05_default106_20260723/`
