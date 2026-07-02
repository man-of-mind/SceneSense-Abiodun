# PPS split-inference deployment (loopback, 2 crowded loops, seed 31)

back_ms / RTT / transport_est are over the frames whose result returned (results_n); front_ms & payload are per-frame.

| pps | frames | results_n | front_ms | back_ms | RTT_ms(loopback) | transport_est_ms | payload_KB(comp) | payload_KB(uncomp) | RTT_p95_ms |
|---|---|---|---|---|---|---|---|---|---|
| 100000 | 400 | 56 | 49.7 | 8.9 | 42.0 | 33.0 | 1073.6 | 2835.0 | 50.8 |
| 150000 | 400 | 61 | 49.3 | 7.2 | 39.4 | 32.2 | 1041.4 | 2835.0 | 43.7 |
| 200000 | 400 | 75 | 49.7 | 8.5 | 41.1 | 32.6 | 1048.3 | 2835.0 | 51.9 |
| 250000 | 400 | 68 | 48.9 | 7.2 | 39.4 | 32.2 | 1032.4 | 2835.0 | 43.6 |
| 300000 | 400 | 78 | 49.8 | 7.3 | 40.3 | 33.0 | 1059.8 | 2835.0 | 45.4 |
