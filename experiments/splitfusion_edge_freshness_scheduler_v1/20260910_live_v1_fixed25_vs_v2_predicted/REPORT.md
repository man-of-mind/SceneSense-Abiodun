# SplitFusion live v1/fixed-25 versus v2/predicted payoff

Each cell used 300 transmitted frames from live CARLA Route B and a fresh `FAVORABLE_STABLE` OAI/RFsim lifecycle. Map utility is credited only by the authoritative `ACK_INSTALLED` path.

The end-to-end decomposition uses same-host wall-clock boundaries and reconciles per installed frame: capture→first send + feature uplink + edge queue + edge service/result send + result→map install = install AoI.

| action | variant | sent | installed | useful | ≤100 ms | ≤500 ms | queue med | edge service med | install AoI med | time-weighted AoI |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | `V1_FIXED_25_MS` | 300 | 86 | 86 | 0 | 85 | 0.28 | 121.61 | 320.08 | 459.90 |
| 30 | `V2_PREDICTED_INSTALL_HORIZON` | 300 | 138 | 138 | 0 | 138 | 0.28 | 77.28 | 282.57 | 370.91 |
| 15 | `V1_FIXED_25_MS` | 300 | 170 | 170 | 0 | 170 | 0.19 | 121.06 | 284.20 | 442.84 |
| 15 | `V2_PREDICTED_INSTALL_HORIZON` | 300 | 253 | 253 | 0 | 253 | 0.38 | 77.48 | 239.43 | 472.63 |
| 50 | `V1_FIXED_25_MS` | 300 | 212 | 212 | 0 | 209 | 0.17 | 101.85 | 243.12 | 366.43 |
| 50 | `V2_PREDICTED_INSTALL_HORIZON` | 300 | 298 | 298 | 0 | 298 | 0.17 | 65.19 | 210.02 | 268.67 |
| 71 | `V1_FIXED_25_MS` | 300 | 245 | 245 | 0 | 245 | 0.17 | 92.25 | 198.00 | 294.43 |
| 71 | `V2_PREDICTED_INSTALL_HORIZON` | 300 | 287 | 287 | 0 | 287 | 0.15 | 54.31 | 152.25 | 229.30 |

Total wall time: 20.0 minutes.

The predicted policy uses only registered initial estimates and causal EWMA updates from previously completed frames. It never interrupts a running CUDA kernel and never reads a current frame's future realized service time.
