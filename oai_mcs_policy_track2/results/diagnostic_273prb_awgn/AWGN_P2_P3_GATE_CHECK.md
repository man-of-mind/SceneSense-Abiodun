# AWGN bad-channel gate check: P2 hold-few vs P3 AIMD

Purpose: before freezing the Track 2 MCS policy, check whether the new AIMD logic still backs off under degraded channel conditions instead of simply preserving high MCS for CARLA bursty traffic.

## Run validity

- Date/run window: 2026-07-31 evening.
- Channel setup: RFsim AWGN channel model enabled with the existing 273PRB AWGN config.
- Common CARLA/model setup: closed-loop CARLA split-inference, 10 FPS target, 300 requested frames, no-AE back-half, ROI 0.0, per-channel uint8, zstd, fast radar rasterizer, 200k radar points/s, 28 vehicles + 35 pedestrians.
- Valid P2 run group: `downlink_oai_bw273_awgn_track2_hold_fps10_track2_awgn273_p2p3_20260801_hold`
- Valid P3 run group: `downlink_oai_bw273_awgn_track2_aimd_fps10_track2_awgn273_p2p3_20260801_aimd`
- Note: the P2 run log contains an earlier failed sandbox start at 21:42:04. The valid completed P2 run starts at 21:42:54 and finishes with `rc=0`.

## Main comparison

| Metric | P2 hold-few | P3 AIMD | Interpretation |
|---|---:|---:|---|
| Returned frames | 288 / 300 | 293 / 300 | Both runs completed with high frame return. |
| Delivery | 96.0% | 97.7% | Similar, slightly higher for AIMD. |
| Median feature payload | 1045.8 KiB | 1043.1 KiB | Same workload. |
| UL scheduled rate | 18.46 Mbps | 15.15 Mbps | AIMD intentionally drains slower because it selects lower MCS. |
| UL MCS avg / p50 / p95 | 24.53 / 25 / 27 | 18.20 / 18 / 26 | AIMD backs off under bad-channel evidence. |
| UL retransmission grants | 1622 / 24493 | 614 / 34601 | AIMD strongly reduces retransmission pressure. |
| UL retransmission rate | 6.62% | 1.77% | Main safety win for AIMD. |
| BLER-update windows with retx | 147 / 1083 | 62 / 1264 | AIMD sees fewer bad windows after backing off. |
| Median feature uplink latency | 216.9 ms | 283.0 ms | Reliability win costs latency in AWGN. |
| p95 feature uplink latency | 303.3 ms | 452.6 ms | Tail latency is worse for AIMD under this channel. |
| Median capture→result | 289.1 ms | 365.0 ms | End-to-end follows the uplink queueing change. |
| p95 capture→result | 385.8 ms | 546.5 ms | Tail also follows lower drain rate. |
| RLC mean queue wait | 128.7 ms | 167.1 ms | Queueing remains the dominant latency source. |
| UE PDCP→gNB PDCP mean transit | 131.3 ms | 167.5 ms | Confirms RLC queue wait explains nearly all RAN uplink delay. |
| gNB PUSCH SNR p50 | 20.0 dB | 20.0 dB | Same AWGN condition. |

## Scheduler branch behavior

Branch labels:

- Branch 1: increase MCS on clean/low-BLER evidence.
- Branch 2: decrease MCS on high-BLER/retransmission evidence.
- Branch 3: sparse sample window.
- Branch 4: hold inside target BLER window.

| Policy | Branch 1 increase | Branch 2 decrease | Branch 3 sparse | Branch 4 hold | What changed |
|---|---:|---:|---:|---:|---|
| P2 hold-few | 117 | 91 | 448 | 427 | Sparse windows are held, but high-BLER decrease is still only `-1` MCS. |
| P3 AIMD | 667 | 47 | 464 | 86 | Sparse clean windows are held; real bad windows trigger larger backoff. |

Extra branch detail:

- P2 branch-2 median step: old MCS 26 → new MCS 25, so the bad-channel response is gentle but leaves many retransmissions.
- P3 branch-2 median step: old MCS 27 → new MCS 13, so the bad-channel response is much stronger.
- P3 branch-3 sparse windows had 0% nonzero-retransmission windows, which is the desired behavior: sparse clean CARLA bursts are not punished.

## Takeaway

P3 AIMD passes the safety gate: under AWGN it does lower MCS and reduces retransmission/BLER pressure, while under the earlier good-channel run it behaved like hold-few and stayed near MCS 28.

However, P3 is likely too conservative for this AWGN case. The large multiplicative backoff protects reliability, but it reduces the RLC drain rate and increases uplink latency. So P3 is a good proof that the logic is no longer blindly holding high MCS, but it should not be frozen as the final tuned policy yet.

Recommended next policy candidate: keep the AIMD structure, but make the decrease less aggressive, for example cap a single bad-window drop to a small number of MCS steps or use a gentler multiplicative factor. Then re-run the same good-channel and AWGN gates.

