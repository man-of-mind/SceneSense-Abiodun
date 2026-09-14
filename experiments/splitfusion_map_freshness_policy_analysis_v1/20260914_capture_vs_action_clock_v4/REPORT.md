# SplitFusion 288-cell dual-clock freshness analysis

The completed 288-cell measurements remain immutable. This offline
analysis reports two different, complementary clocks:

- **Physical capture clock:** camera capture to map installation. This is
  the actual age of information and remains the safety/freshness metric.
- **Action-start clock:** entry to seven-channel tensor assembly to map
  installation. This removes work completed before the selected action
  can influence the system and is the controller-attributable metric.

The action-start clock does not replace physical AoI. In particular, a
map may satisfy an action-service budget while still be too old physically.

## Clock bridge

The bridge uses 344,177 same-event
UE receive timestamps. Its absolute offset deviation is
0.001099 ms at p99;
robust campaign drift is 0.000003 ms.

## Strict latest-only aggregate

| Quantity | Prior capture-clock replay | Corrected capture clock | Corrected action-start clock |
|---|---:|---:|---:|
| Median cell install latency | 233.2 ms | 235.9 ms | 190.0 ms |
| Median cell time-weighted map age | 308.0 ms | 309.9 ms | 264.2 ms |
| Route time with map age ≤150 ms | 1.10% | 1.07% | 5.75% |
| Route time with map age ≤200 ms | 7.70% | 7.54% | 19.13% |
| Route time with map age ≤250 ms | 21.54% | 21.21% | 35.96% |

Preparation before the action boundary is measured, not guessed:
median 36.2 ms and
cell-median p95 94.3 ms.
The causal correction delayed 22,453
imputed arrivals that the prior replay had placed before their own
measured UE transmission-start boundary (encoding complete, before
the first datagram). Observed arrivals were not changed. The later
send-loop completion is not a causal lower bound because edge receipt
can overlap the host's multi-datagram send loop.

## FIFO versus strict latest-only after correction

| Queue policy | Physical map AoI | Action-clock map age | Physical fresh ≤200 | Action-clock fresh ≤200 |
|---|---:|---:|---:|---:|
| FIFO_NO_DISCARD | 372.8 ms | 326.7 ms | 6.98% | 17.43% |
| LATEST_ONLY_NO_EXPIRY | 309.9 ms | 264.2 ms | 7.54% | 19.13% |

## Strict latest-only by network profile

| Profile | Capture→action start | Physical map AoI | Action-clock map age | Physical fresh ≤200 | Action-clock fresh ≤200 |
|---|---:|---:|---:|---:|---:|
| FAVORABLE_STABLE | 37.0 ms | 294.7 ms | 247.4 ms | 10.91% | 26.52% |
| MID_VARIABLE | 36.0 ms | 305.7 ms | 260.2 ms | 8.47% | 21.13% |
| FADE_RECOVERY | 37.2 ms | 305.6 ms | 257.4 ms | 8.76% | 22.25% |
| ADVERSE_STABLE | 35.1 ms | 350.5 ms | 304.9 ms | 2.01% | 6.53% |

## Raw-freshness action winners under strict latest-only

| Profile | Budget | Capture-clock winner | Capture fresh | Action-clock winner | Action-clock fresh |
|---|---:|---:|---:|---:|---:|
| FAVORABLE_STABLE | 150 ms | 71 | 8.66% | 71 | 28.16% |
| FAVORABLE_STABLE | 200 ms | 71 | 36.74% | 71 | 63.54% |
| FAVORABLE_STABLE | 250 ms | 71 | 68.23% | 71 | 85.04% |
| MID_VARIABLE | 150 ms | 65 | 6.76% | 65 | 25.97% |
| MID_VARIABLE | 200 ms | 65 | 32.62% | 65 | 60.91% |
| MID_VARIABLE | 250 ms | 65 | 65.31% | 65 | 84.25% |
| FADE_RECOVERY | 150 ms | 65 | 6.66% | 65 | 26.49% |
| FADE_RECOVERY | 200 ms | 65 | 33.37% | 65 | 61.61% |
| FADE_RECOVERY | 250 ms | 65 | 65.75% | 65 | 84.06% |
| ADVERSE_STABLE | 150 ms | 65 | 1.35% | 65 | 7.10% |
| ADVERSE_STABLE | 200 ms | 65 | 9.87% | 65 | 25.17% |
| ADVERSE_STABLE | 250 ms | 65 | 29.95% | 65 | 52.17% |

## Policy interpretation

- Reward physical map freshness/utility; do not reward the action clock
  as though it were the age of the sensed world.
- Give the policy the input age at decision time. That state tells it
  how much of the freshness budget preparation has already consumed.
- Use action-service latency for action attribution and diagnosis. It
  prevents fixed CARLA/sensor work from being blamed on compression.
- The 100 ms source cadence is a sampling interval, not a latency term.
- Both FIFO and strict latest-only reuse the same measured transport
  surface and final edge calibration; this remains a counterfactual
  queue replay, not a new live campaign.
