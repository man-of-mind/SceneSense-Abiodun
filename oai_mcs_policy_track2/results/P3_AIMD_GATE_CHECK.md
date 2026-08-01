# Track 2 P3 AIMD gate check

Run inspected:

- `downlink_oai_default106_ttracer_fps10_track2_aimd_current_20260801_default106_noae`

Reference runs:

- P0 current vanilla adaptive: `downlink_oai_default106_ttracer_fps10_track2_vanilla_current_20260801_default106_noae`
- P2 hold-few-samples: `downlink_oai_default106_ttracer_fps10_track2_holdfew_20260801_default106_noae`

## Verdict

P3 behaves as intended under the current good RFsim channel.

The gNB run log confirms:

- `mcs_policy=aimd`
- `force_ul_mcs=adaptive`
- `hold_mcs_few_samples=0`

Interpretation:

- This was not vanilla, not fixed-MCS, and not the P2 boolean hold-few path.
- The AIMD policy climbed to MCS 28 and held it because the BLER/retransmission evidence was zero.
- Under this good-channel condition, P3 is intentionally close to P2.
- Compared with current vanilla, P3 removes the low-MCS/RLC queue bottleneck.

## Main comparison

| Metric | P0 current vanilla | P2 hold few samples | P3 AIMD |
|---|---:|---:|---:|
| Frames / returned | 1300 / 1207 | 1300 / 1196 | 1300 / 1200 |
| Delivery | 92.85% | 92.00% | 92.31% |
| Uplink payload p50 | 1046.6 KiB | 1049.0 KiB | 1048.1 KiB |
| Front feature build p50 | 55.7 ms | 56.4 ms | 54.8 ms |
| Feature uplink p50 | 142.5 ms | 37.5 ms | 37.4 ms |
| Feature uplink p95 | 157.2 ms | 44.0 ms | 42.9 ms |
| Capture→result p50 | 209.5 ms | 105.2 ms | 103.5 ms |
| Capture→result p95 | 252.3 ms | 148.7 ms | 139.1 ms |
| Edge tail p50 | 7.0 ms | 7.5 ms | 6.9 ms |
| Downlink p50 | 3.1 ms | 2.3 ms | 2.2 ms |
| UL scheduled Mbps | 21.5 Mbps | 27.6 Mbps | 28.3 Mbps |
| UL MCS avg / p50 / p95 | 7.37 / 7 / 13 | 27.73 / 28 / 28 | 27.72 / 28 / 28 |
| UL PRB p50 / p95 | 106 / 106 | 106 / 106 | 106 / 106 |
| UL grant TBS p50 / p95 | 1089 / 3521 B | 8961 / 10247 B | 8961 / 10247 B |
| UL retx rate | 0.0 | 0.0 | 0.0 |
| RLC LCID4 p95 occupancy | 1010 KB | 398 KB | 416 KB |
| RLC queue wait estimate | ~88 ms | ~14 ms | ~14 ms |
| UE PDCP→gNB PDCP p50 | 87.5 ms | 16.3 ms | 16.2 ms |
| gNB PUSCH SNR p50 | 50.5 dB | 50.5 dB | 50.5 dB |
| UE tunnel avg TX | 21.3 Mbps | 26.5 Mbps | 27.2 Mbps |
| UE tunnel p95 TX | 35.3 Mbps | 52.5 Mbps | 52.5 Mbps |
| UE tunnel TX drops | 1334 | 1953 | 1847 |

## BLER/MCS branch evidence

For the uplink BLER/MCS decision trace:

| Trace item | P0 current vanilla | P2 hold few samples | P3 AIMD |
|---|---:|---:|---:|
| Updated UL decisions | 62,840 | 49,370 | 48,223 |
| Branch 1: increase | 18,012 | 28 | 27 |
| Branch 2: high BLER decrease | 0 | 0 | 0 |
| Branch 3: too few samples | 44,828 | 44,816 | 43,666 |
| Branch 4: hold within target | 0 | 4,526 | 4,530 |
| Rows with retransmission evidence | 0 | 0 | 0 |
| BLER window > 0 rows | 0 | 0 | 0 |
| Max BLER window | 0 ppm | 0 ppm | 0 ppm |

Interpretation:

- P0 vanilla sees many sparse update windows and stays at low MCS.
- P2 and P3 still see sparse update windows, but do not penalize clean sparse windows.
- P3 did not need to exercise its multiplicative backoff path because no BLER/retransmission occurred.

## Run-quality notes

- P3 has the same route/workload settings as P0/P2: no-AE, ROI 0, zstd, 200k radar PPS, fast rasterizer, 10 FPS target, 1300 frames.
- P3 has the same startup camera timeout warning pattern seen in P0/P2.
- One P3 successful frame had a startup-adjacent uplink outlier of ~1387 ms. Only one successful P3 frame exceeded 75 ms uplink, and p95/p99 remain clean. Treat this as a startup artifact, not steady-state behavior.
- Delivery remains ~92%, so P3 does not solve the missing returned-frame issue.

## Accuracy sanity

Using `abiodun/staleness/validate_accuracy.py`, score >= 0.2, 5 m gate:

| Scope | P0 current vanilla | P2 hold few samples | P3 AIMD |
|---|---:|---:|---:|
| All frames loc MAE | 1.698 m | 1.811 m | 1.768 m |
| Common returned frames loc MAE | 1.680 m | 1.798 m | 1.760 m |
| Common returned frames after startup loc MAE | 1.647 m | 1.771 m | 1.734 m |

Interpretation:

- Do not claim model-accuracy improvement from P3.
- The accuracy sanity is in the same rough live-deployment range and does not invalidate the radio/latency conclusion.
- GT rows, in-frustum GT count, ego speed, and radar projected points are effectively identical across P0/P2/P3.

## Recommendation

P3 is a better policy candidate than P2 conceptually because it preserves the good-channel latency gain while adding a backoff path for real BLER/retransmission.

However, this good-channel run only proves the no-regression/good-channel side. It does not yet prove bad-channel safety because branch 2 was never exercised.

Next clean step:

- either run a controlled bad-channel sanity test to trigger retransmissions and verify AIMD backs off,
- or, if we want to avoid the earlier AWGN ambiguity for now, freeze P3 as the good-channel policy candidate and move to plotting/reporting P0/P2/P3.
