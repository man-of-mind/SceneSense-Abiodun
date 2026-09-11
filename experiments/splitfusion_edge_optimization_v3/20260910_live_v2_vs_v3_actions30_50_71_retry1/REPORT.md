# Final SplitFusion edge optimization: live v2 versus v3

Each cell used 300 transmitted frames from live CARLA Route B and a fresh `FAVORABLE_STABLE` OAI/RFsim lifecycle. Both variants used the same causal predicted-install scheduler; only the edge implementation changed.

| action | variant | installed | edge service med (ms) | install AoI med (ms) | queue med (ms) |
|---:|---|---:|---:|---:|---:|
| 30 | `V2_PREDICTED_INSTALL_HORIZON` | 124 | 79.11 | 288.63 | 0.40 |
| 30 | `V3_OVERLAPPED_FINAL` | 135 | 73.03 | 286.07 | 0.29 |
| 50 | `V2_PREDICTED_INSTALL_HORIZON` | 293 | 64.63 | 211.93 | 0.18 |
| 50 | `V3_OVERLAPPED_FINAL` | 283 | 58.32 | 202.05 | 0.17 |
| 71 | `V2_PREDICTED_INSTALL_HORIZON` | 299 | 57.79 | 170.29 | 0.15 |
| 71 | `V3_OVERLAPPED_FINAL` | 297 | 49.23 | 163.16 | 0.14 |

Total wall time: 15.3 minutes.

The comparison does not claim 100 ms service readiness or run-to-run variance characterization.
