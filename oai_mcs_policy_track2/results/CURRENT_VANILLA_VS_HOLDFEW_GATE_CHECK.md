# Track 2 gate check: current vanilla adaptive vs hold-few-samples

Runs compared:

- Current vanilla adaptive: `downlink_oai_default106_ttracer_fps10_track2_vanilla_current_20260801_default106_noae`
- Hold-few-samples policy: `downlink_oai_default106_ttracer_fps10_track2_holdfew_20260801_default106_noae`

## Verdict

This is now a clean apples-to-apples radio/latency comparison.

The current vanilla rerun reproduces the original vanilla behavior closely, so the earlier baseline was not a fluke:

- MCS remains low: p50 `7`, p95 `13`.
- PRBs are already saturated: p50/p95 `106/106`.
- BLER/retransmission is still effectively zero.
- The MCS reduction is still driven by the "too few scheduled samples" branch, not by actual observed retransmission.
- Feature uplink latency remains high: p50 `142.5 ms`.

The hold-few-samples policy changes only the MCS adaptation behavior and gives the expected improvement:

- MCS rises to p50/p95 `28/28`.
- Feature uplink latency drops to p50 `37.5 ms`.
- Capture→result p50 drops from `209.5 ms` to `105.2 ms`.
- RLC queueing estimate drops from `~88 ms` to `~14 ms`.
- UE PDCP→gNB PDCP p50 drops from `87.5 ms` to `16.3 ms`.

Important caveat: delivery rate remains around `92%` in both runs, so this policy fixes the uplink latency/MCS bottleneck but does not explain or solve missing returned frames.

## Main metrics

| Metric | Current vanilla adaptive | Hold few samples |
|---|---:|---:|
| Frames / returned | 1300 / 1207 | 1300 / 1196 |
| Delivery | 92.85% | 92.00% |
| Uplink payload p50 | 1046.6 KiB | 1049.0 KiB |
| Front feature build p50 | 55.7 ms | 56.4 ms |
| Feature uplink p50 | 142.5 ms | 37.5 ms |
| Feature uplink p95 | 157.2 ms | 44.0 ms |
| Edge tail p50 | 7.0 ms | 7.5 ms |
| Downlink p50 | 3.1 ms | 2.3 ms |
| Capture→result p50 | 209.5 ms | 105.2 ms |
| Capture→result p95 | 252.3 ms | 148.7 ms |
| UE tunnel avg TX | 21.3 Mbps | 26.5 Mbps |
| UE tunnel p95 TX | 35.3 Mbps | 52.5 Mbps |
| UL scheduled Mbps | 21.5 Mbps | 27.6 Mbps |
| UL MCS avg / p50 / p95 | 7.37 / 7 / 13 | 27.73 / 28 / 28 |
| UL PRB p50 / p95 | 106 / 106 | 106 / 106 |
| UL grant TBS p50 / p95 | 1089 / 3521 B | 8961 / 10247 B |
| UL retx rate | 0.0 | 0.0 |
| RLC LCID4 p95 occupancy | 1010 KB | 398 KB |
| RLC queue wait estimate | ~88 ms | ~14 ms |
| UE PDCP→gNB PDCP p50 | 87.5 ms | 16.3 ms |
| gNB PUSCH SNR p50 | 50.5 dB | 50.5 dB |
| UE tunnel TX drops | 1334 | 1953 |

## BLER/MCS branch evidence

For the uplink BLER/MCS decision trace:

| Trace item | Current vanilla adaptive | Hold few samples |
|---|---:|---:|
| Updated UL decisions | 62,840 | 49,370 |
| Branch 1: increase | 18,012 | 28 |
| Branch 2: high BLER decrease | 0 | 0 |
| Branch 3: too few samples | 44,828 | 44,816 |
| BLER window > 0 rows | 0 | 0 |
| Max BLER window | 0 ppm | 0 ppm |

Interpretation:

- In both runs, there is no observed BLER/retransmission evidence forcing MCS down.
- Vanilla adaptive repeatedly hits branch 3 and decrements/holds the MCS low because the traffic is sparse/bursty.
- Hold-few-samples still records branch 3 events, but does not penalize MCS for those sparse windows. Since BLER remains zero, the policy allows MCS to stay high.

## Workload equivalence checks

| Check | Result |
|---|---|
| CARLA frames | both 1300 |
| Simulated route duration | both 129.9 s |
| GT rows | both 35,100 |
| In-frustum GT under 40 m | both 2,409 |
| Radar projected points | identical frame-by-frame |
| Ego speed | identical frame-by-frame |
| Uplink feature chunks | 18 p50 in both |
| Backend worker | same container image, same no-AE checkpoint, single worker |
| Policy flags | current vanilla `hold_mcs_few_samples=0`; P2 `hold_mcs_few_samples=1`; both `force_ul_mcs=-1` |

Feature compressed bytes are not bit-identical frame-by-frame, but they are very close and strongly correlated. This is acceptable for a live CARLA/GPU comparison and does not affect the radio conclusion.

## Accuracy sanity

Using `abiodun/staleness/validate_accuracy.py`, score >= 0.2, 5 m gate:

| Scope | Current vanilla loc MAE | Hold-few loc MAE |
|---|---:|---:|
| All frames | 1.698 m | 1.811 m |
| Common returned frames | 1.695 m | 1.805 m |
| Common returned frames, after startup | 1.662 m | 1.779 m |

Interpretation:

- Do not claim an accuracy improvement from this policy run.
- The policy conclusion should be limited to latency/MCS/RLC behavior.
- The accuracy delta is small enough that it does not invalidate the radio result, but large enough that it should be reported only as a sanity check.

## Recommendation

Proceed to the next policy experiment only after keeping this pair as the clean Track 2 P0/P2 baseline:

- P0: current vanilla adaptive OAI
- P2: hold-few-samples

Next policy to test should be a safer BLER-aware/AIMD variant that preserves the P2 benefit under good channel but can back off when actual BLER/retransmission appears.
