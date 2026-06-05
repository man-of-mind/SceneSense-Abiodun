# Camera-Only OD vs SEG Latency Comparison

Generated: 2026-06-04T19:55:58

## Headline

- OD median RTT: loopback 8.2 ms, OAI 74.9 ms.
- SEG median RTT: loopback 13.4 ms, OAI 107.9 ms.
- OD payload median: 86.6 KiB; SEG payload median: 397.9 KiB.
- SEG/OD payload ratio: 4.59x on loopback.

## OAI Config

- RAN mode: NR SA over OAI RFsim, single UE for OD/SEG latency runs
- Carrier: band n78, DL 3.6192 GHz, 106 PRB at 30 kHz SCS
- Bandwidth: 106 PRB = about 40 MHz channel class (38.16 MHz RB span)
- TDD pattern: 5 ms: 7 DL slots + mixed slot (6 DL sym, 4 UL sym) + 2 UL slots
- Core QoS: DNN oai, S-NSSAI SST=1, 5QI=9, AMBR UL/DL 10Gbps/10Gbps
- UE path: IMSI 001010000000001, oaitun_ue1 10.0.0.2 -> perception RX 192.168.70.140

## Per-Run Summary

| task | transport | frames | receive | median RTT ms | p95 RTT ms | median payload KiB |
|---|---:|---:|---:|---:|---:|---:|
| OD | Loopback | 1911 | 100.0% | 8.2 | 19.1 | 86.6 |
| OD | OAI | 1011 | 100.0% | 74.9 | 142.6 | 86.3 |
| SEG | Loopback | 1770 | 100.0% | 13.4 | 17.1 | 397.9 |
| SEG | OAI | 775 | 98.8% | 107.9 | 201.1 | 396.3 |
