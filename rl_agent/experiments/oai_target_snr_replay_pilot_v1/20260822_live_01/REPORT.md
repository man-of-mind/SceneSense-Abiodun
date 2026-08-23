# OAI target-SNR replay integration pilot

**Verdict:** `REPLAY_INTEGRATION_PASS`

This was one short integration pilot using saved target-SNR design traces. The target values are not measured achieved-OAI traces and were converted through the measured command mapping; RFsim did not receive target PUSCH SNR directly.

## Upper-anchor attempts

| Attempt | RFsim noise command (dB) | Achieved median PUSCH SNR (dB) | PUSCH samples | Status |
|---:|---:|---:|---:|---|
| 1 | -12.50 | 25.00 | 235 | TAIL_ACCEPTED |

## Piecewise target-to-RFsim mapping

| Achieved-SNR anchor (dB) | RFsim noise command (dB) | Source |
|---:|---:|---|
| 5.50 | -2.25 | EXISTING_MEASURED_ANCHOR |
| 6.00 | -2.50 | EXISTING_MEASURED_ANCHOR |
| 6.50 | -3.00 | EXISTING_MEASURED_ANCHOR |
| 7.50 | -3.50 | EXISTING_MEASURED_ANCHOR |
| 8.50 | -4.00 | EXISTING_MEASURED_ANCHOR |
| 10.00 | -5.00 | EXISTING_MEASURED_ANCHOR |
| 16.00 | -8.00 | EXISTING_MEASURED_ANCHOR |
| 19.50 | -10.00 | EXISTING_MEASURED_ANCHOR |
| 25.00 | -12.50 | PILOT_MEASURED_UPPER_ATTEMPT_1 |

## Replay result by profile

| Profile | Applied/expected | Late | Skipped | Valid SNR intervals | Correlation | MAE (dB) | Delivery | RTT p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `FAVORABLE_STABLE` | 100/100 | 0 | 0 | 100 | 0.974 | 0.420 | 100.00% | 31.034 |
| `MID_VARIABLE` | 100/100 | 0 | 0 | 100 | 0.985 | 0.383 | 100.00% | 59.509 |
| `ADVERSE_STABLE` | 100/100 | 0 | 0 | 99 | 0.990 | 0.294 | 100.00% | 55.636 |
| `FADE_RECOVERY` | 100/100 | 0 | 0 | 100 | 0.982 | 0.268 | 100.00% | 39.053 |

## Overall

- Commands applied: 400 / 400 (100.00%)
- Commands late/skipped: 0 / 0
- Command ACK p50/p95/max: 0.079 / 0.305 / 1.286 ms
- Valid achieved-SNR intervals: 399 / 400
- Tracking MAE / p95 absolute error: 0.341 / 1.001 dB
- Application delivery: 400 / 400 (100.00%)
- Application RTT p50/p95/max: 22.621 / 47.543 / 109.675 ms
- UE/PDU disconnection: False
- Clean `noise_power_dB=-50` restore: True
- Cleanup clean: True
- Command-to-first-observed-effect latency: unavailable with the stock tracer; no causal value is claimed.
