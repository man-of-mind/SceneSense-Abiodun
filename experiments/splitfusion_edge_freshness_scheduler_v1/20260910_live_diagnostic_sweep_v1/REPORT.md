# SplitFusion edge freshness scheduling sweep

This is a counterfactual discrete-event simulation over the four-action
live timing diagnostic. It does not modify or replace the measured
288-cell evidence.

## Evidence and reproduction

Missing timestamps belong primarily to frames replaced by the live
depth-one pending slot. Their arrivals and service times are explicitly
imputed by the frozen within-action interpolation rule. Therefore the
sweep is suitable for policy screening, not a new measurement claim.

| action | measured completions | reproduced | measured replacements | reproduced | pass |
|---:|---:|---:|---:|---:|:---:|
| 15 | 230 | 235 | 70 | 65 | yes |
| 30 | 226 | 230 | 72 | 68 | yes |
| 50 | 274 | 278 | 26 | 22 | yes |
| 71 | 284 | 288 | 16 | 12 | yes |

## Policy sweep

### Action 15 — `split_noae_uint4_q7000`

| policy | installed | superseded | wait-expired | median AoI (ms) | map AoI (ms) | updates/s |
|---|---:|---:|---:|---:|---:|---:|
| fifo | 18 | 0 | 0 | 474.2 | 2903.8 | 0.58 |
| latest_only | 211 | 60 | 0 | 314.8 | 491.3 | 6.82 |
| latest_only_wait_0ms | 147 | 12 | 117 | 271.0 | 488.1 | 4.75 |
| latest_only_wait_5ms | 149 | 13 | 114 | 271.9 | 487.1 | 4.81 |
| latest_only_wait_10ms | 154 | 15 | 107 | 270.2 | 483.5 | 4.98 |
| latest_only_wait_20ms | 161 | 16 | 98 | 273.4 | 481.2 | 5.20 |
| latest_only_wait_25ms | 164 | 16 | 95 | 274.9 | 480.1 | 5.30 |

### Action 30 — `split_ae128_uint4_q0000`

| policy | installed | superseded | wait-expired | median AoI (ms) | map AoI (ms) | updates/s |
|---|---:|---:|---:|---:|---:|---:|
| fifo | 8 | 0 | 0 | 433.3 | 7783.9 | 0.26 |
| latest_only | 99 | 33 | 0 | 340.1 | 4733.1 | 3.24 |
| latest_only_wait_0ms | 73 | 11 | 50 | 312.7 | 4737.4 | 2.39 |
| latest_only_wait_5ms | 73 | 12 | 48 | 312.7 | 4737.3 | 2.39 |
| latest_only_wait_10ms | 75 | 11 | 47 | 312.7 | 4736.4 | 2.45 |
| latest_only_wait_20ms | 79 | 11 | 43 | 318.8 | 4735.2 | 2.58 |
| latest_only_wait_25ms | 81 | 11 | 41 | 318.8 | 4734.8 | 2.65 |

### Action 50 — `split_ae64_uint4_q5000`

| policy | installed | superseded | wait-expired | median AoI (ms) | map AoI (ms) | updates/s |
|---|---:|---:|---:|---:|---:|---:|
| fifo | 224 | 0 | 0 | 316.4 | 775.8 | 7.39 |
| latest_only | 277 | 22 | 0 | 261.3 | 317.1 | 9.14 |
| latest_only_wait_0ms | 193 | 6 | 100 | 228.4 | 318.5 | 6.37 |
| latest_only_wait_5ms | 198 | 6 | 95 | 227.5 | 316.6 | 6.54 |
| latest_only_wait_10ms | 201 | 6 | 92 | 229.2 | 315.9 | 6.63 |
| latest_only_wait_20ms | 213 | 6 | 80 | 230.8 | 311.6 | 7.03 |
| latest_only_wait_25ms | 220 | 7 | 72 | 231.5 | 310.3 | 7.26 |

### Action 71 — `split_ae32_uint4_q9800`

| policy | installed | superseded | wait-expired | median AoI (ms) | map AoI (ms) | updates/s |
|---|---:|---:|---:|---:|---:|---:|
| fifo | 283 | 0 | 0 | 282.1 | 381.3 | 9.36 |
| latest_only | 286 | 12 | 0 | 208.7 | 270.3 | 9.46 |
| latest_only_wait_0ms | 196 | 9 | 93 | 186.5 | 278.2 | 6.48 |
| latest_only_wait_5ms | 212 | 9 | 77 | 185.5 | 272.1 | 7.01 |
| latest_only_wait_10ms | 221 | 9 | 68 | 184.9 | 268.7 | 7.31 |
| latest_only_wait_20ms | 246 | 9 | 43 | 187.4 | 262.4 | 8.14 |
| latest_only_wait_25ms | 252 | 10 | 36 | 187.8 | 261.5 | 8.34 |

## Interpretation boundary

A lower queue budget may reduce update count while improving the
freshness of accepted work. No policy is selected from installed
count alone. Promotion requires joint consideration of time-weighted
AoI, useful update rate, supersession/waste, and later GPU/live parity.
