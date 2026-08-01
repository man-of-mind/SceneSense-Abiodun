# AWGN bad-channel four-way MCS-policy gate check

Purpose: compare default OAI adaptive behavior, hold-few-samples, uncapped AIMD, and capped AIMD under the same RFsim AWGN bad-channel setup. This checks whether the proposed split-inference-aware policy lowers BLER/retransmission pressure when the channel is actually bad, while avoiding unnecessary MCS reduction for sparse clean CARLA bursts.

Status: diagnostic-only. Do not use this as the official Track 2 bad-channel comparison against the 106PRB clear-channel baseline, because this run family uses 273PRB. The official policy gate must use one fixed OAI configuration, preferably the existing default 106PRB setup, and vary only the channel/policy.

## Runs

Common setup:

- CARLA closed-loop split-inference frontend, 10 FPS target, 300 requested frames.
- 273PRB RFsim AWGN OAI setup: `gnb.sa.band78.fr1.273PRB.scenesense_rfsim.awgn.conf` + `ue.awgn.conf`.
- Model/payload: no-AE, ROI 0.0, per-channel uint8, zstd, fast radar rasterizer, 200k radar points/s.
- Scene: 28 vehicles + 35 pedestrians.
- T-tracer: UE `all`, gNB `latency`.

Valid run groups:

- P0 vanilla: `downlink_oai_bw273_awgn_track2_vanilla_fps10_track2_awgn273_vanilla_cap3_20260801_vanilla`
- P2 hold-few: `downlink_oai_bw273_awgn_track2_hold_fps10_track2_awgn273_p2p3_20260801_hold`
- P3 AIMD uncapped: `downlink_oai_bw273_awgn_track2_aimd_fps10_track2_awgn273_p2p3_20260801_aimd`
- P4 AIMD cap=3: `downlink_oai_bw273_awgn_track2_aimd_cap_fps10_track2_awgn273_vanilla_cap3_20260801_aimd_cap`

## Summary

Important caveat: this is an internal AWGN 273PRB policy comparison. The absolute MCS values here should not be compared directly with the official good-channel 106PRB vanilla baseline. The 106PRB baseline uses many small low-MCS grants, while this 273PRB AWGN gate uses fewer larger grants and a different BLER-update sample pattern. See `AWGN_VANILLA_MCS_VERIFICATION.md`. A fair bad-channel gate should rerun vanilla / hold-few / uncapped AIMD / capped AIMD on 106PRB AWGN.

| Policy | Median uplink | p95 uplink | Median capture→result | UL MCS avg / p50 / p95 | UL scheduled rate | Retx rate | Bad retx windows | RLC mean queue wait |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 vanilla | 218.3 ms | 311.4 ms | 293.3 ms | 23.16 / 25 / 27 | 17.73 Mbps | 5.14% | 124 / 1131 | 132.3 ms |
| P2 hold-few | 216.9 ms | 303.3 ms | 289.1 ms | 24.53 / 25 / 27 | 18.46 Mbps | 6.62% | 147 / 1083 | 128.7 ms |
| P3 AIMD uncapped | 283.0 ms | 452.6 ms | 365.0 ms | 18.20 / 18 / 26 | 15.15 Mbps | 1.77% | 62 / 1264 | 167.1 ms |
| P4 AIMD cap=3 | 219.7 ms | 262.1 ms | 290.3 ms | 24.13 / 24 / 27 | 18.25 Mbps | 3.63% | 90 / 1047 | 126.0 ms |

## BLER evidence

BLER is from `GNB_MAC_BLER_MCS_DECISION`.

- Per-update BLER window = `num_retx / num_sched`.
- Filtered BLER = OAI scheduler state after the 0.9 IIR filter.
- Thresholds in the current OAI config: lower 5%, upper 15%.

| Policy | Filtered BLER mean | Filtered BLER p95 | Time above 15% upper threshold | Per-window BLER p95 | Branch-2 MCS decreases |
|---|---:|---:|---:|---:|---:|
| P0 vanilla | 5.16% | 15.81% | 6.54% | 52.09% | 63 |
| P2 hold-few | 6.75% | 17.32% | 10.43% | 56.00% | 91 |
| P3 AIMD uncapped | 2.05% | 5.92% | 0.08% | 0.00% | 47 |
| P4 AIMD cap=3 | 3.82% | 8.83% | 0.00% | 48.00% | 72 |

Interpretation:

- Vanilla and hold-few preserve high MCS and good latency, but BLER/retransmission pressure remains high.
- Uncapped AIMD strongly suppresses BLER/retransmissions, but its large one-step MCS drops slow the uplink drain and increase queueing latency.
- Capped AIMD is the best balance so far: it lowers the filtered BLER state below the upper threshold, cuts retransmission pressure versus vanilla/hold-few, and keeps latency close to the high-MCS policies.

## Presentation plots

- `plots/awgn_policy_fourway_scheduler_summary_timeseries.pdf`
- `plots/awgn_policy_fourway_bler_window_smallmultiples.pdf`
- `plots/awgn_policy_fourway_ran_timeseries.pdf`

## Recommendation

Use P4 capped AIMD as the next candidate to gate on both good-channel and bad-channel runs. It is not final yet, but it is better behaved than both extremes:

- less fragile than hold-few / vanilla under AWGN bad channel;
- less latency-punishing than uncapped AIMD.

Next useful check: repeat the good-channel 106PRB closed-loop run with `SCENESENSE_MCS_POLICY=aimd` and `SCENESENSE_AIMD_MAX_DROP=3` to confirm capped AIMD still behaves like hold-few when BLER is zero.
