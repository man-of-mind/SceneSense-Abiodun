# Scheduler panel plot sources

These are the exact files used for the 4-row slide plots. MCS/PRB are from UE-visible uplink DCI grant windows; throughput is UE tunnel TX; SNR is gNB PUSCH SNR where captured. The corrected drivable UL-heavy 106PRB and drivable 273PRB runs did not capture gNB PUSCH SNR, so their SNR rows use same-configuration RFsim controls only to show the channel remained high-SNR. MCS, PRB, and UE tunnel throughput for those rows still come from the corrected drivable CARLA runs.

| Run | MCS p50 | PRB p50 | Scheduled Mbps p50 | UE tunnel TX Mbps p50 | SNR p50 | SNR source note |
|---|---:|---:|---:|---:|---:|---|
| iperf3 UDP uplink 9 Mbps 106PRB, 7DL/2UL | 28.0 | 51.3 | 21.3 | 9.2 | 50.5 | exact |
| CARLA default OAI 106PRB, 7DL/2UL | 2.2 | 94.9 | 4.0 | 3.8 | 50.5 | exact; reduced-payload trace |
| CARLA UL-heavy OAI 106PRB, 4DL/5UL | 4.5 | 101.0 | 17.8 | 17.2 | 50.5 | same-config RFsim control |
| CARLA wider BW OAI 273PRB, 7DL/2UL | 3.9 | 260.0 | 17.6 | 17.2 | 50.5 | 273PRB RFsim control |

## Full source paths

### iperf3 UDP uplink 9 Mbps 106PRB, 7DL/2UL

- Grant source: `metrics_logs/scenesense_ttracer/oai_iperf_default_udp9m_20260720/ue/analysis/nrue_grant_windows.csv`
- Network source: `metrics_logs/scenesense_network/oai_iperf_default_udp9m_20260720/network_timeseries.csv`
- SNR source: `metrics_logs/scenesense_ttracer/oai_iperf_default_udp9m_20260720/gnb/csv/GNB_MAC_PUSCH_POWER_CONTROL.csv`

### CARLA default OAI 106PRB, 7DL/2UL

- Grant source: `metrics_logs/carla_oai_ttracer/downlink_oai_default106_ttracer_ae128_u6_roi05_fps10_ae128_u6_roi05_default106_20260723/nrue_ul_grant_windows_compact.csv`
- Network source: `metrics_logs/carla_oai_ttracer/downlink_oai_default106_ttracer_ae128_u6_roi05_fps10_ae128_u6_roi05_default106_20260723/network_timeseries.csv`
- SNR source: `metrics_logs/carla_oai_ttracer/downlink_oai_default106_ttracer_ae128_u6_roi05_fps10_ae128_u6_roi05_default106_20260723/gnb_pusch_power_compact.csv`

### CARLA UL-heavy OAI 106PRB, 4DL/5UL

- Grant source: `metrics_logs/carla_oai_ttracer/downlink_oai_ulheavy_106_ttracer_fps10_drivable_rerun_20260722_ulheavy106/nrue_ul_grant_windows_compact.csv`
- Network source: `metrics_logs/carla_oai_ttracer/downlink_oai_ulheavy_106_ttracer_fps10_drivable_rerun_20260722_ulheavy106/network_timeseries.csv`
- SNR source: `metrics_logs/carla_oai_ttracer/downlink_oai_ulheavy_106_ttracer_forcemcs28_fps10_forcemcs28_ulheavy106_20260723/gnb_pusch_power_compact.csv`

### CARLA wider BW OAI 273PRB, 7DL/2UL

- Grant source: `metrics_logs/carla_oai_ttracer/downlink_oai_bw273_mu1_ttracer_fps10_drivable_rerun_20260722_bw273/nrue_ul_grant_windows_compact.csv`
- Network source: `metrics_logs/carla_oai_ttracer/downlink_oai_bw273_mu1_ttracer_fps10_drivable_rerun_20260722_bw273/network_timeseries.csv`
- SNR source: `metrics_logs/carla_oai_ttracer/downlink_oai_bw273_mu1_ttracer_fps10_layerinstr_20260723_145921/gnb_pusch_power_compact.csv`

