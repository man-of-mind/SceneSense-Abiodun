# UE-vs-gNB Grant Validation

| RNTI | Dir | UE grants | gNB MAC grants | UE TBS bytes | gNB MAC TBS bytes | UE/gNB MAC ratio | gNB PHY bytes | UE/gNB PHY ratio | UE MCS | gNB MAC MCS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0x03c0 | dl | 2246 | 588 | 53086 | 13969 | 3.800272 | 0.000000 |  | 0.000000 | 0.000000 |
| 0x03c0 | ul | 10040 | 10041 | 15677088 | 15677180 | 0.999994 | 15675002.000000 | 1.000133 | 8.000000 | 8.000000 |
| 0x0efc | dl | 2225 | 588 | 57491 | 15038 | 3.823048 | 0.000000 |  | 0.000000 | 0.000000 |
| 0x0efc | ul | 10119 | 10119 | 15806473 | 15806473 | 1.000000 | 15804115.000000 | 1.000149 | 8.000000 | 8.000000 |

Ratios close to 1.0 are expected when UE and gNB traces cover the same time interval. Small differences can come from starting the two recorders at slightly different times.
