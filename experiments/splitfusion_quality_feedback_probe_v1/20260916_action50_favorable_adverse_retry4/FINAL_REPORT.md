# Exact Quality Feedback Timing — Live Action-50 Probe

Status: `SPLITFUSION_QUALITY_FEEDBACK_LIVE_PROBE_V1_COMPLETE`

This bounded live diagnostic measured exact CARLA segmentation/localization
quality feedback for action 50 under `FAVORABLE_STABLE` and `ADVERSE_STABLE`.
Each cell transmitted exactly 300 frames. It did not complete Route B and makes
no full-route claim.

Exact per-frame quality is privileged CARLA ground truth for simulator training
and evaluation. It is not a deployable real-world feedback signal.

## Population

| Profile | Sent / reassembled | Final-prediction eligible | Exact quality ACK received | Evaluation failures |
|---|---:|---:|---:|---:|
| FAVORABLE_STABLE | 300 / 300 | 276 | 276 | 0 |
| ADVERSE_STABLE | 300 / 300 | 279 | 279 | 0 |

The 24 and 21 non-eligible frames had no final production prediction. They were
not evaluator failures.

## Policy-boundary latency (ms)

| Profile | Action start -> quality ACK P50 / P95 / P99 | First feature send -> quality ACK P50 / P95 / P99 | Final prediction -> quality ACK P50 / P95 / P99 |
|---|---:|---:|---:|
| FAVORABLE_STABLE | 187.878 / 653.604 / 793.266 | 153.164 / 610.575 / 759.509 | 9.491 / 494.575 / 629.531 |
| ADVERSE_STABLE | 207.696 / 345.571 / 586.905 | 173.819 / 302.085 / 546.854 | 9.469 / 81.272 / 356.636 |

`Action start` is immediately before constructing the seven-channel input.
The complete interval includes UE preparation/front/compression, feature
uplink, edge processing, exact scoring, compact ACK transmission, and ACK
receipt at the UE.

## Measured component latency (ms)

| Component | FAVORABLE P50 / P95 / P99 | ADVERSE P50 / P95 / P99 |
|---|---:|---:|
| Action start -> first feature send | 29.758 / 62.421 / 115.813 | 28.569 / 66.716 / 95.657 |
| First feature send -> model ready | 98.071 / 190.680 / 314.726 | 132.331 / 210.752 / 306.914 |
| Model ready -> final p025 prediction | 23.226 / 33.883 / 123.157 | 21.815 / 32.614 / 181.065 |
| Final prediction -> exact quality complete | 3.477 / 492.138 / 621.729 | 3.021 / 65.925 / 350.616 |
| Quality complete -> socket send | 0.245 / 0.593 / 1.111 | 0.227 / 0.595 / 1.227 |
| Compact ACK downlink | 4.833 / 9.457 / 14.782 | 5.494 / 8.976 / 18.241 |

Component percentiles must not be added to reconstruct an end-to-end
percentile; the end-to-end distribution is computed frame by frame.

## Concurrency result

The early semantic branch was exact-equal to the production mask in all bounded
parity samples. Semantic scoring completed before final prediction for 269/276
(97.5%) favorable and 276/279 (98.9%) adverse eligible frames. Its median
arithmetic cost was 3.405 and 3.497 ms, respectively, so that work was hidden
behind object post-processing. Final localization scoring was only 0.265 and
0.222 ms median.

The remaining long tail is not FCOS or quality-score arithmetic. Exact ground
truth read/wait P95 was 129.6 ms favorable and 59.1 ms adverse. With one final
evaluator, those stalls caused head-of-line queue tails. The next safe
optimization is to create/prefetch object ground truth from the already frozen
CARLA tick alongside semantic scoring, reuse the early semantic result instead
of rereading it, and leave only localization, merge/encode, and ACK transmission
after final p025 records are ready.

## Decision-window consequence

From action start, exact feedback arrived before the next decision for only:

| Profile | 8 FPS (125 ms) | 9 FPS (111.1 ms) | 10 FPS (100 ms) |
|---|---:|---:|---:|
| FAVORABLE_STABLE | 13/300 (4.33%) | 4/300 (1.33%) | 2/300 (0.67%) |
| ADVERSE_STABLE | 0/300 | 0/300 | 0/300 |

Therefore this measurement does not support a synchronous exact-reward design
at 8–10 FPS. The compact ACK itself is fast; the outcome is already too late
before that ACK begins. A frame-ID-correlated delayed training reward, or a
deployable quality proxy for immediate control, remains necessary unless the
whole upstream path is materially shortened.

## Integrity

- Both cells passed and all registered files re-hashed without mismatch.
- Terminal accounting: 300/300 dispositions in each cell.
- Packet proof: edge = pcap = UE (276 favorable, 279 adverse), zero capture drops.
- Maximum quality ACK payload: 779 B favorable, 773 B adverse.
- Eight serial/parity validations per cell were exact.
- All lifecycle, RFsim restoration, CARLA shutdown, and cold-host gates passed.
