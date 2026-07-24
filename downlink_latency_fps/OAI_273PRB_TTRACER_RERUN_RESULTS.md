# OAI 273PRB CARLA + T-tracer Rerun

Date: 2026-07-22

Status: **corrected drivable-scene rerun complete**.

Batch: `drivable_rerun_20260722_bw273`

This rerun used the corrected `run_common.sh` deployment scene:

- 28 vehicles;
- 35 pedestrians;
- seed 31;
- ego ignore-lights 50%;
- fixed waypoint loop `80,85,91,94,99,80`;
- no-AE checkpoint, per-channel-u8, ROI 0, 200k radar PPS;
- live CARLA frontend, 10 FPS target, 1300 frames;
- working 273PRB OAI recipe: matching gNB 273PRB config and UE launch using
  the validated center-frequency/SSB settings.

## Latency and reliability

| Condition | Frames | Returned | Delivery | Feature payload p50 | Result payload p50 | Front p50 | Back p50 | Uplink handling p50 | Downlink p50 | RTT p50 / p95 | Capture→result p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 273PRB, mu=1, zstd, T-tracer | 1300 | 1110 | 85.4% | 1054.6 KB, 19 chunks | 2.4 KB, 1 chunk | 26.0 ms | 7.4 ms | 174.2 ms | 3.0 ms | 186.1 / 202.6 ms | 212.7 ms |

## T-tracer summary

| Metric | 273PRB value |
|---|---:|
| App-offered traffic, 1s window p50 / p95 | 17.20 / 26.09 Mbps |
| UE tunnel TX, 1s window p50 / p95 | 17.17 / 26.36 Mbps |
| Scheduled uplink, 1s window p50 / p95 | 17.58 / 25.18 Mbps |
| Average PRB allocation, 1s window p50 / p95 | 260.0 / 265.0 PRB |
| Average MCS, 1s window p50 | 3.90 |
| p95 MCS inside 1s windows, median over windows | 8.0 |
| Retransmission proxy from extracted grant windows | 0.0 |

The 273PRB run clearly allocates close to the wider-bandwidth PRB ceiling, but
the scheduled uplink rate is not meaningfully higher than the corrected 106PRB
UL-heavy run because the MCS is lower. This is why wider bandwidth alone did
not reduce application latency in this setup.

## RF-quality caveat

The extracted UE PHY SNR/CQI/RSRP fields are not currently reportable. They are
sentinel/placeholder-like in this RFsim extraction (`RSRP=-2147483648`, CQI/SNR
around `-90`) and should not be used as channel-quality evidence. The valid
t-tracer evidence from this run is PRB allocation, MCS, scheduled TBS/rate, and
tunnel/app traffic rate.

## Artifacts

- Summary CSV:
  `runs/downlink_fps_summary_drivable_rerun_20260722_bw273.csv`
- T-tracer artifact folder:
  `../metrics_logs/carla_oai_ttracer/downlink_oai_bw273_mu1_ttracer_fps10_drivable_rerun_20260722_bw273/`
- Presentation plots:
  `plots/oai_ttracer/ttracer_ul_mcs_prb_timeseries_bw273.pdf`
  `plots/oai_ttracer/ttracer_tunnel_tx_rx_timeseries_bw273.pdf`
