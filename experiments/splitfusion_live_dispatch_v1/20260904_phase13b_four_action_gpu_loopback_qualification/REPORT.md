# Phase 13B four-action GPU/localhost qualification

Status: `SPLITFUSION_LIVE_DISPATCH_FOUR_ACTION_GPU_LOOPBACK_QUALIFIED`.

This is a bounded functional qualification, not a latency or 36-profile campaign.

| action | profile | family | quantizer | q_e4 | keep | SFD1 bytes | UDP datagrams | detections | parity |
|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 0 | split_noae_uint8_q0000 | noAE | UINT8 | 0 | 21504 | 3583884 | 287 | 39 | exact |
| 20 | split_ae128_uint8_q5000 | AE128 | UINT8 | 5000 | 10752 | 1213684 | 98 | 41 | exact |
| 46 | split_ae64_uint6_q9000 | AE64 | UINT6 | 9000 | 2150 | 99670 | 8 | 39 | exact |
| 71 | split_ae32_uint4_q9800 | AE32 | UINT4 | 9800 | 430 | 6086 | 1 | 49 | exact |

All four live messages used the committed SFD1 and inner codec bytes, exactly one live zstd decompression and one live tail call. Direct diagnostic calls were counted separately.

The same Phase-11B-bound fit frame and one resident camera calibration were used throughout. No holdout, validation, or test frame was opened. No tensor, SFD1 blob, datagram, prediction payload, or serialized service payload is retained.

Monotonic stage boundaries were checked only for presence, order, and non-negative duration. No latency, FPS, throughput, Pi/OAI, or deployment-performance result is claimed.

The 288-cell campaign remains blocked pending the later localhost measurement, RFsim calibration, and 16-cell OAI pilot.
