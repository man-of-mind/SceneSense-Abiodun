# Direct BLER/OLLA MCS trace results

This is the final scheduler-side check requested after the advisor discussion. It instruments `get_mcs_from_bler()` directly and compares sparse CARLA-like bursts against a dense 10 FPS open-loop control on the same 273 PRB RFsim path. Summaries below use the active traffic window only, not the tracer idle tail.

## Main takeaway

- The low-MCS behavior is visible inside the BLER/OLLA MCS selector itself, before the later PHR/RB helper.
- The decisive difference is scheduling cadence/sample availability: sparse closed-loop-style bursts repeatedly hit the `num_sched <= 3` branch, while dense open-loop traffic gives the selector enough high-sample windows to keep ratcheting MCS upward.
- This explains why iperf/open-loop UDP can ramp to high MCS while the CARLA closed-loop app remains stuck near QPSK despite high RFsim PUSCH SNR.

## Summary table

| Run | MCS p50 | MCS p95 | Last nonzero MCS | num_sched p50 | num_sched p95 | Increase % | Few-samples % | High-BLER dec % | Hold % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| observed pace 1.2FPS | 4.0 | 8.0 | 1.0 | 1.0 | 60.0 | 21.4% | 78.6% | 0.0% | 0.0% |
| open loop 10FPS | 23.0 | 25.0 | 1.0 | 3.5 | 60.0 | 50.0% | 50.0% | 0.0% | 0.0% |

## Branch-code reference

| Branch | Meaning |
|---:|---|
| 0 | no update / (diff < 10) |
| 1 | increase / (low BLER) |
| 2 | decrease / (high BLER) |
| 3 | decrease/hold-low / (few samples) |
| 4 | hold / (in target) |

## Artifacts

- Summary CSV: `/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/oai_layer_latency/plots/bler_olla_summary.csv`
- Windowed CSV: `/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/oai_layer_latency/plots/bler_olla_windows.csv`
- Plot: `/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/oai_layer_latency/plots/bler_olla_branch_comparison.png`
- Plot: `/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/oai_layer_latency/plots/bler_olla_branch_comparison.pdf`
- Plot: `/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/oai_layer_latency/plots/bler_olla_mcs_timeseries.png`
- Plot: `/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/oai_layer_latency/plots/bler_olla_mcs_timeseries.pdf`
- Plot: `/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/oai_layer_latency/plots/bler_olla_num_sched_timeseries.png`
- Plot: `/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/oai_layer_latency/plots/bler_olla_num_sched_timeseries.pdf`
