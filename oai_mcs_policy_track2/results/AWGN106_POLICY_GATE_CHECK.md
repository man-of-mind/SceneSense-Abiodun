# Official 106PRB AWGN policy gate check

Purpose: inspect the official bad-channel Track 2 policy gate after correcting the earlier 273PRB/106PRB mismatch. These runs keep the default 106PRB OAI configuration fixed and vary only the channel condition and MCS policy.

## Run validity

All four policy runs completed cleanly with `rc=0`.

Common setup:

- gNB config: `gnb.sa.band78.fr1.106PRB.scenesense_rfsim.awgn.conf`
- UE config: `ue.awgn.conf`
- UE launch: `-r 106 -C 3619200000`
- RFsim channel model: enabled, `rfsim_chanmod=1`
- TDD/PRB family: default 106PRB path, same family as the clear-channel baseline
- Model/payload: no-AE, ROI 0.0, per-channel uint8, zstd
- Radar: fast rasterizer, 200k points/s
- Scene: 28 vehicles + 35 pedestrians
- Requested frames: 300 at 10 FPS

Valid run groups:

- P0 vanilla: `downlink_oai_default106_awgn_track2_vanilla_fps10_track2_awgn106_20260801_vanilla`
- P2 hold-few: `downlink_oai_default106_awgn_track2_hold_fps10_track2_awgn106_20260801_hold`
- P3 AIMD uncapped: `downlink_oai_default106_awgn_track2_aimd_fps10_track2_awgn106_20260801_aimd`
- P4 AIMD cap=3: `downlink_oai_default106_awgn_track2_aimd_cap_fps10_track2_awgn106_20260801_aimd_cap`

No run-log crashes, attach failures, back-half failures, Python tracebacks, or fatal errors were found in the artifact logs.

## Summary table

Clear-channel vanilla is included only as context; the four AWGN rows are the fair bad-channel comparison.

| Policy | Frames / returned | Delivery | SNR p50 | MCS avg / p50 / p95 | Retx rate | Filtered BLER p95 | Median uplink | p95 uplink | RLC queue wait | Capture→result p50 / p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clear P0 vanilla context | 1300 / 1207 | 92.85% | 50.5 dB | 7.37 / 7 / 13 | 0.00% | 0.00% | 142.5 ms | 157.2 ms | 88.3 ms | 209.5 / 252.3 ms |
| P0 vanilla AWGN | 300 / 291 | 97.00% | 19.5 dB | 22.22 / 24 / 27 | 3.26% | 15.99% | 255.7 ms | 380.8 ms | 148.7 ms | 333.2 / 489.6 ms |
| P2 hold-few AWGN | 300 / 289 | 96.33% | 19.5 dB | 24.84 / 26 / 27 | 6.39% | 19.72% | 246.0 ms | 310.4 ms | 131.7 ms | 321.7 / 399.8 ms |
| P3 AIMD uncapped AWGN | 300 / 290 | 96.67% | 19.5 dB | 19.04 / 19 / 26 | 1.14% | 4.71% | 311.7 ms | 465.8 ms | 174.7 ms | 383.3 / 579.2 ms |
| P4 AIMD cap=3 AWGN | 300 / 292 | 97.33% | 19.5 dB | 24.45 / 24 / 27 | 2.93% | 8.92% | 246.8 ms | 283.7 ms | 130.6 ms | 322.7 / 378.9 ms |

## Suspiciousness check

No evidence of a run mix-up:

- all four AWGN policies used 106PRB, not 273PRB;
- all four used AWGN channelmod;
- policy envs match labels:
  - vanilla: `mcs_policy=vanilla`
  - hold: `hold_mcs_few_samples=1`, `mcs_policy=legacy`
  - AIMD uncapped: `mcs_policy=aimd`, `aimd_max_drop=uncapped`
  - AIMD cap=3: `mcs_policy=aimd`, `aimd_max_drop=3`
- all four used the same CARLA/model/payload setup.

One counterintuitive but explainable behavior:

- AWGN vanilla has much higher MCS than clear-channel vanilla, even though SNR is lower.
- This is not a PRB mismatch anymore; both are 106PRB.
- The likely cause is OAI's BLER/few-sample interaction: clear-channel split-inference traffic repeatedly enters the sparse-sample branch and decays to low MCS, while the AWGN run creates enough BLER/retransmission feedback to keep vanilla operating at high MCS with slow one-step backoff.
- Therefore, do not interpret MCS alone as channel quality. Use BLER, retransmission rate, RLC queue wait, and uplink latency together.

## Interpretation

- AWGN did bite: SNR fell from ~50.5 dB to ~19.5 dB and retransmissions/BLER became nonzero.
- Hold-few keeps high MCS and good latency, but it has the worst retransmission/BLER pressure.
- Uncapped AIMD strongly suppresses BLER/retransmissions, but it overreacts and increases RLC queueing/uplink latency.
- Capped AIMD is the best balance in this 106PRB AWGN gate:
  - lower retx/BLER than vanilla and hold-few;
  - latency close to hold-few;
  - much lower tail latency than uncapped AIMD.

## Plots

- `plots/awgn106_policy_scheduler_summary_timeseries.pdf`
- `plots/awgn106_policy_bler_window_smallmultiples.pdf`

## Current conclusion

The official 106PRB AWGN gate is usable. Nothing in the run artifacts suggests a command/config mistake. The result supports using capped AIMD (`AIMD_MAX_DROP=3`) as the next policy candidate, pending the matching clear-channel 106PRB cap=3 run.

