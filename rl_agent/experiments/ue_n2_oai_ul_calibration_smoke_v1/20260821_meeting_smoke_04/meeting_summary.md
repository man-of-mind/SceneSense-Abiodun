# UE-N2 bounded OAI calibration smoke

**Status:** `UE_N2_SMOKE_CAPTURED_PARTIAL_EVIDENCE`

This is partial evidence: command timing and descriptive per-plateau radio response are usable, but the stock tracer cannot prove a causal first-effect timestamp.

- Commands sent: 120 / 120
- Handler bracket p50/p95/max: 0.123 / 0.409 / 1.048 ms
- All responses before the next 100-ms boundary: True
- Descriptive monotone command-to-SNR response: True
- Restored to clean -50: True

| RFsim noise command (dB) | median PUSCH SNR (dB) | median scheduler EMA SNR (dB) | median final MCS | EMA status |
|---:|---:|---:|---:|---|
| -10 | 19.50 | 19.50 | 24.00 | SETTLED_COUNT_GATE |
| -8 | 16.00 | 15.60 | 19.00 | SETTLED_COUNT_GATE |
| -5 | 10.00 | 9.80 | 12.00 | SETTLED_COUNT_GATE |
| -4 | 8.50 | 8.30 | 9.00 | SETTLED_COUNT_GATE |

Direct UL BLER remains unavailable; no value was zero-filled.
