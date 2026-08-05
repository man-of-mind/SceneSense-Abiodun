# 106PRB AWGN ladder compact summary

Base batch: `track2_sinr_awgn_ladder_20260803`

| profile | policy | noise_power_dB | snr_p50_db | mcs_p50 | mcs_p95 | ul_sched_mbps | retx_rate_pct | delivery_pct | uplink_p50_ms | uplink_p95_ms | capture_result_p50_ms | capture_result_p95_ms | olla_bler_status | hypothesis_read | run_group |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mild | vanilla | -10.000 | 19.500 | 25.000 | 27.000 | 15.948 | 3.461 | 97.333 | 249.315 | 342.038 | 318.595 | 473.315 | available | BLER not persistent; channel may be too mild for decisive bad-channel test | downlink_oai_default106_awgn_mild_track2_vanilla_fps10_track2_sinr_awgn_ladder_20260803_mild_vanilla |
| mild | sinr | -10.000 | 19.500 | 24.000 | 24.000 | 16.712 | 0.000 | 99.667 | 247.561 | 275.479 | 318.114 | 373.184 | not_applicable_sinr_policy | SINR-driven: MCS follows avg_snr; OLLA BLER columns are N/A by design | downlink_oai_default106_awgn_mild_track2_sinr_fps10_track2_sinr_awgn_ladder_20260803_mild_sinr |
| medium | vanilla | -5.000 | 9.800 | 13.000 | 15.000 | 10.269 | 4.971 | 96.333 | 597.059 | 788.938 | 676.009 | 875.740 | available | MCS lowers under sustained BLER; compare latency/retransmission tradeoff | downlink_oai_default106_awgn_medium_track2_vanilla_fps10_track2_sinr_awgn_ladder_20260803_medium_vanilla |
| medium | sinr | -5.000 | 9.800 | 12.000 | 12.000 | 9.769 | 0.001 | 97.000 | 604.191 | 635.081 | 681.965 | 727.147 | not_applicable_sinr_policy | SINR-driven: MCS follows avg_snr; OLLA BLER columns are N/A by design | downlink_oai_default106_awgn_medium_track2_sinr_fps10_track2_sinr_awgn_ladder_20260803_medium_sinr |
| strong | vanilla | -4.000 | 8.200 | 10.000 | 13.000 | 8.559 | 4.617 | 95.333 | 777.609 | 962.890 | 853.399 | 1048.284 | available | MCS lowers under sustained BLER; compare latency/retransmission tradeoff | downlink_oai_default106_awgn_strong_track2_vanilla_fps10_track2_sinr_awgn_ladder_20260803_strong_vanilla |
| strong | sinr | -4.000 | 8.200 | 9.000 | 9.000 | 8.453 | 0.000 | 97.000 | 751.956 | 787.313 | 832.406 | 880.456 | not_applicable_sinr_policy | SINR-driven: MCS follows avg_snr; OLLA BLER columns are N/A by design | downlink_oai_default106_awgn_strong_track2_sinr_fps10_track2_sinr_awgn_ladder_20260803_strong_sinr |
