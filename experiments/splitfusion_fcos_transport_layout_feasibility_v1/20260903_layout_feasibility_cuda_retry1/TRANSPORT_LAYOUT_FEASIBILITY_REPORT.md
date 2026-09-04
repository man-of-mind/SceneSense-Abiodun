# Transport layout feasibility

Terminal: `TRANSPORT_LAYOUT_FEASIBILITY_COMPLETE`

| layout | profiles | size change vs current | codec latency Δ ms |
| --- | ---: | ---: | ---: |
| CHANNEL_MAJOR | 72 | -3.940% | 46636.905 |
| CHANNEL_MAJOR_MODULAR_DELTA | 72 | -3.538% | 235844.079 |
| CURRENT_CELL_MAJOR | 72 | 0.000% | 0.000 |

## Per quantizer

| quantizer | layout | size change vs current | codec latency Δ ms |
| --- | --- | ---: | ---: |
| UINT4 | CHANNEL_MAJOR | -8.376% | 13213.883 |
| UINT4 | CHANNEL_MAJOR_MODULAR_DELTA | -8.050% | 77394.518 |
| UINT4 | CURRENT_CELL_MAJOR | 0.000% | 0.000 |
| UINT6 | CHANNEL_MAJOR | -3.564% | 23267.480 |
| UINT6 | CHANNEL_MAJOR_MODULAR_DELTA | -2.365% | 86700.858 |
| UINT6 | CURRENT_CELL_MAJOR | 0.000% | 0.000 |
| UINT8 | CHANNEL_MAJOR | -2.401% | 10155.542 |
| UINT8 | CHANNEL_MAJOR_MODULAR_DELTA | -2.591% | 71748.703 |
| UINT8 | CURRENT_CELL_MAJOR | 0.000% | 0.000 |

## Result

- classification: **NOT_USEFUL**
- production layout change recommended: False
- exact layout/zstd round trips: 27,648/27,648
- frozen state unchanged: True
- network profiles with explicit usable bandwidth: none found; paired break-even bandwidths are reported in JSON only.
- no tail, heads, p025, evaluator, validation/test frame or payload blob was retained.
