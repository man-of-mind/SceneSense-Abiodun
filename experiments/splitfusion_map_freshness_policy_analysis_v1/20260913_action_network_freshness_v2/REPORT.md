# SplitFusion action–network–freshness policy analysis

This is an offline counterfactual analysis of the immutable 288-cell
campaign. Measured captures, payloads, complete reassemblies, edge
admissions and the final qualified edge-service calibration are held
fixed. Queue discipline is the only experimental factor.

The budgets are 150, 200 and 250 ms. A 100 ms budget is not used as
the primary analysis because the source itself produces frames every
100 ms; the historical 100 ms figure remains provenance only.

## Queue-policy comparison

| policy | queue median | queue p95 | max queue | useful install/sent | map AoI |
|---|---:|---:|---:|---:|---:|
| FIFO_NO_DISCARD | 0.0 ms | 256.3 ms | 10290.1 ms | 0.6373 | 371.6 ms |
| LATEST_ONLY_NO_EXPIRY | 0.0 ms | 50.6 ms | 234.6 ms | 0.6195 | 308.0 ms |

## Strict latest-only by network profile

| profile | useful install/sent | map AoI | map fresh ≤150 ms | map fresh ≤200 ms | map fresh ≤250 ms |
|---|---:|---:|---:|---:|---:|
| FAVORABLE_STABLE | 0.7360 | 293.1 ms | 1.65% | 11.12% | 29.68% |
| MID_VARIABLE | 0.6261 | 304.2 ms | 1.23% | 8.65% | 23.85% |
| FADE_RECOVERY | 0.6456 | 304.3 ms | 1.29% | 8.94% | 24.57% |
| ADVERSE_STABLE | 0.4702 | 349.6 ms | 0.22% | 2.06% | 7.95% |

## Scientific interpretation

- `FIFO_NO_DISCARD` drains every admitted frame, including work that
  completes after the route observation window. It is a backlog
  baseline, not the recommended map scheduler.
- `LATEST_ONLY_NO_EXPIRY` never interrupts active CUDA work. While it
  runs, each newer arrival replaces the single pending frame. The
  newest pending frame is always selected next, regardless of how
  long it waited; there is no 25 ms expiry.
- Pre-edge transport failures and measured admission rejections are
  identical under both policies.
- Fresh-map fractions count the initial no-map interval as not fresh.
  Time-weighted map AoI is conditional on a map being available and is
  therefore reported separately from map availability.
- Quality-weighted freshness is a decision surrogate, not a claim that
  network conditions change the intrinsic validation accuracy.
- The aligned localization check is secondary: aligned truth is sampled
  during exact-record retrieval after ACK, and its matching support can
  differ from the source-time matches.
