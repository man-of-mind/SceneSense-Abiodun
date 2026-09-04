# Phase 11C — zstd level 1/3/5 sweep

Terminal: `HYBRID_Q_PHASE11C_ZSTD_LEVEL_SWEEP_COMPLETE`

This is a lossless host codec comparison over real 72-profile inner payloads. It performs no perception scoring or accuracy measurement; byte-exact recovery means zstd level cannot change perception accuracy. Timings are not Raspberry Pi or OAI latency claims.

## Aggregate comparison

| level | round trips | compressed bytes | compression ms | decompression ms | codec ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 9,216 | 5,747,358,582 | 8111.239 | 4614.391 | 12725.630 |
| 3 | 9,216 | 5,820,164,186 | 18385.398 | 5305.381 | 23690.779 |
| 5 | 9,216 | 5,835,367,252 | 34775.714 | 5306.207 | 40081.921 |

| comparison | size saving | compression Δ ms | decompression Δ ms | codec Δ ms |
| --- | ---: | ---: | ---: | ---: |
| L3 vs L1 | -72,805,604 (-1.267%) | 10274.160 | 690.990 | 10965.150 |
| L5 vs L1 | -88,008,670 (-1.531%) | 26664.475 | 691.816 | 27356.291 |
| L5 vs L3 | -15,203,066 (-0.261%) | 16390.316 | 0.826 | 16391.142 |

## Integrity and scope

- exact byte-equal round trips: 27,648/27,648
- headers checked: 27,648
- frozen state unchanged: True
- payload blobs retained: 0
- train-fit sample: 128 frames, 16 deterministic endpoint-inclusive frames from each of eight fit episodes; holdout/validation/test frames read: 0
- registered network bandwidth projections: none; no exact rates exist in the repository
- conclusion: workload/network dependent — no level is both no larger and no slower in every profile; select no production level here and seek Raspberry Pi/OAI confirmation

The JSON carries all paired per-frame break-even summaries. The CSV carries every family × quantizer × q × level row.
