# SplitFusion sensor-preparation optimization result

Status: **PASS (structural qualification); targeted allocation removed, no total-latency improvement claim**

## Bound executions

- Analyzer repair: `d8447f3b4226122377e85f608dfa32276f4b33cc`
- Optimization implementation: `8ea61a2ef2b63eb0fd4d318ce4d7dd4f7693b10d`
- Preserved live baseline: `experiments/splitfusion_sensor_preparation_live_v1/20260914_action50_favorable_baseline_retry1`
- Create-only retrospective baseline: `experiments/splitfusion_sensor_preparation_live_v1/20260914_action50_favorable_baseline_retry1_retrospective_analysis`
- Create-only optimized cell: `experiments/splitfusion_sensor_preparation_live_v1/20260914_action50_favorable_optimized_v1`
- Both cells used action 50 (`AE64/UINT4/q=0.50`), `FAVORABLE_STABLE`, direct edge-to-map, renderer off and empty CPU reservations.

The retrospective baseline binds `PASSED.json`, `per_frame_metrics.csv`, `direct_edge_publication.csv`, `direct_map_ingest.csv`, the campaign ledger and the attempt manifest. The source was reopened read-only and its hashes were rechecked after analysis.

## Optimization decision and result

Radar rasterization was the largest compute stage, but it was already the exact vectorized production implementation. Window construction, coordinate transforms and stationary tracking are scientifically required. The largest clearly avoidable repeated allocation was therefore P19: constructing the two immutable ImageNet normalization tensors on CUDA for every frame.

The optimized collector constructs the same float32 tensors once on its selected device and reuses them for that collector lifetime. It changes no sensor value, radar window, interpolation, model input, action, codec, queue, deadline or radio setting. Exact radar tensor, radar evidence and seven-channel PyTorch input equality passed on all eight registered equivalence frames.

P19 improved from 1.344/5.776/7.030 ms to 0.001/0.002/0.002 ms at P50/P95/P99. Total sensor compute did not improve in the independent live run: 37.662/62.141/127.464 ms became 44.452/90.064/129.109 ms. The optimization therefore qualifies as output-equivalent and removes its target allocation, but this pair does not support a claim of end-to-end speedup.

## Sensor stages

Each row is the registered 500-frame analysis window except the asynchronous semantic callback, which has its observed count. `%` is the stage sum divided by total sensor-compute sum. Callback, wait, evaluation-only and CUDA-event rows are non-additive diagnostics and their percentages must not be summed.

| Stage | Category | Baseline N | Baseline P50 | P95 | P99 | Baseline % | Optimized N | Optimized P50 | P95 | P99 | Optimized % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P01 RGB callback | callback | 500 | 0.032 | 0.099 | 2.202 | 0.367 | 500 | 0.033 | 0.774 | 9.135 | 0.745 |
| P02 radar callback | callback | 500 | 0.752 | 4.178 | 11.379 | 3.499 | 500 | 0.734 | 8.650 | 24.201 | 4.288 |
| P03 semantic callback | callback | 498 | 0.029 | 6.870 | 19.303 | 3.059 | 499 | 0.029 | 8.902 | 23.944 | 3.604 |
| P04 worker scheduling wait | wait | 500 | 0.126 | 33.533 | 49.286 | 12.001 | 500 | 2.533 | 61.140 | 87.315 | 31.987 |
| P05 RGB callback-to-worker | wait | 500 | 0.063 | 17.091 | 54.023 | 7.087 | 500 | 0.613 | 49.956 | 75.360 | 19.883 |
| P06 sensor wait | wait | 500 | 11.574 | 29.092 | 44.343 | 30.806 | 500 | 4.708 | 29.997 | 48.549 | 16.834 |
| P07 radar-window extraction | compute | 500 | 4.506 | 14.288 | 29.212 | 15.249 | 500 | 6.089 | 15.397 | 34.540 | 14.932 |
| P08 radar spherical-to-world | compute | 500 | 4.265 | 11.320 | 19.017 | 12.524 | 500 | 5.555 | 16.998 | 29.523 | 14.006 |
| P09 stationary-track update | compute | 500 | 4.382 | 9.878 | 13.840 | 12.894 | 500 | 4.664 | 13.755 | 27.562 | 13.004 |
| P10 world-to-camera | compute | 500 | 1.895 | 8.604 | 14.455 | 6.911 | 500 | 3.557 | 9.589 | 23.889 | 8.876 |
| P11 projection and bounds | compute | 500 | 0.231 | 1.361 | 5.835 | 1.427 | 500 | 0.304 | 2.086 | 8.807 | 1.395 |
| P12 radar rasterization | compute | 500 | 7.096 | 15.872 | 28.895 | 21.846 | 500 | 8.427 | 21.992 | 33.506 | 21.181 |
| P13 radar evidence packaging | compute | 500 | 0.277 | 0.887 | 3.561 | 1.069 | 500 | 0.313 | 1.385 | 4.070 | 1.074 |
| P14 CARLA BGRA-to-BGR | compute | 500 | 2.439 | 5.710 | 9.033 | 7.604 | 500 | 2.891 | 8.937 | 18.071 | 8.711 |
| P15 BGR-to-RGB | compute | 500 | 0.187 | 1.039 | 3.803 | 0.958 | 500 | 0.262 | 1.797 | 4.883 | 1.099 |
| P16 RGB resize | compute | 500 | 0.162 | 1.402 | 3.974 | 0.998 | 500 | 0.260 | 2.447 | 4.285 | 1.168 |
| P17 RGB tensor pack | compute | 500 | 0.080 | 0.234 | 1.331 | 0.330 | 500 | 0.084 | 0.275 | 0.849 | 0.246 |
| P18 RGB host-to-device | CUDA event | 500 | 2.112 | 6.809 | 9.956 | 6.968 | 500 | 3.219 | 10.786 | 22.280 | 8.416 |
| P19 normalization constants | compute | 500 | 1.344 | 5.776 | 7.030 | 4.670 | 500 | 0.001 | 0.002 | 0.002 | 0.002 |
| P20 RGB normalize | CUDA event | 500 | 0.014 | 0.157 | 1.755 | 0.209 | 500 | 0.012 | 0.426 | 3.088 | 0.281 |
| P21 radar resize/pack | compute | 500 | 0.913 | 2.485 | 10.437 | 3.384 | 500 | 1.184 | 3.420 | 6.744 | 3.017 |
| P22 radar host-to-device | CUDA event | 500 | 0.256 | 0.933 | 5.246 | 1.225 | 500 | 0.329 | 1.257 | 3.560 | 1.135 |
| P23 seven-channel concatenate | CUDA event | 500 | 0.665 | 3.179 | 3.733 | 2.478 | 500 | 0.533 | 3.198 | 3.573 | 2.112 |
| P24 immutable evaluation snapshot | evaluation only | 500 | 0.110 | 0.210 | 0.402 | 0.317 | 500 | 0.124 | 0.316 | 0.604 | 0.560 |
| P25 unattributed pre-front | compute | 500 | 0.089 | 0.149 | 0.326 | 0.244 | 500 | 0.089 | 0.161 | 0.376 | 0.214 |
| P26 diagnostic final-sync wait | diagnostic | 500 | 0.605 | 3.268 | 4.123 | 2.476 | 500 | 0.376 | 3.177 | 3.885 | 1.906 |
| **Total sensor compute** | compute total | **500** | **37.662** | **62.141** | **127.464** | **100.000** | **500** | **44.452** | **90.064** | **129.109** | **100.000** |

## Complete-path timing

Only same-domain or explicitly bridged intervals are reported. Unavailable values are not inferred, imputed or set to zero.

| Metric | Baseline N | Baseline P50/P95/P99 ms | Optimized N | Optimized P50/P95/P99 ms |
|---|---:|---:|---:|---:|
| Complete UE action (`send_finished_ns - capture_started_ns`) | 453 | 26.556 / 53.279 / 66.677 | 455 | 32.119 / 61.034 / 79.900 |
| Pure feature uplink | 0 | **UNAVAILABLE—no baseline UE clock anchors** | 455 | 54.734 / 95.869 / 144.497 |
| Edge compute | 453 | 53.464 / 67.544 / 102.831 | 455 | 51.947 / 76.301 / 119.052 |
| Direct publication-to-install map service | 453 | 2.070 / 4.693 / 13.095 | 455 | 2.215 / 7.736 / 28.846 |
| Capture-to-install AoI | 453 | 190.967 / 261.766 / 327.730 | 455 | 210.378 / 305.306 / 383.157 |
| Action-start-to-install | 0 | **UNAVAILABLE—no baseline UE clock anchors** | 455 | 158.994 / 241.231 / 311.028 |

The optimized cell recorded 2,986 complete adjacent UE `time.time_ns()` / `time.perf_counter_ns()` anchors. Absolute offset-deviation P50/P95/P99 was 0.000099/0.000272/0.000494 ms (maximum 0.005386 ms), passing the prospectively fixed P99 <= 1 ms gate. Legacy edge-to-UE result fields were not used.

## Structural and operational result

- Baseline retrospective: 2,794/2,794 installed updates joined to complete durable publication records.
- Optimized: 2,854/2,854 installed updates joined to complete durable publication records; 2,858 publication rows survived teardown.
- Optimized install-before-ACK: 2,854/2,854; ACK-before-install: 0.
- Optimized terminal accounting: 2,986 captures, zero missing, unexpected or duplicate terminals.
- Object records on radio: false; dense label map on radio: false; UE record-bearing messages: 0.
- Renderer: off. CPU reservations: empty and serialized as complete `--option=` tokens.
- Baseline preparation coverage: 2,935/2,991 = 0.981277; 56 dropped.
- Optimized preparation coverage: 2,986/3,074 = 0.971373; 88 dropped. Both exceed the unchanged 0.95 gate.
- Final host-cold gate: PASS for both cells.

Total sensor-compute correlation with recorded scene covariates was weak in both windows. Baseline Pearson `r` was 0.021 for radar-point count, 0.063 for ego speed, 0.044 for acceleration and 0.038 for yaw rate; optimized values were -0.001, -0.036, 0.040 and 0.015. The larger optimized worker-scheduling and callback-to-worker tails, rather than scene-complexity correlation, are consistent with live host-scheduling variation. This is diagnostic evidence, not a causal attribution.

## Artifact hashes

Baseline retrospective:

- `SENSOR_PROFILE_RESULT.json`: `99283431b35e8c978e75d061867b9d93d3b0d81f9649fab3c868e888eeebb83a`
- `source_binding.json`: `a6c5f9cdf994d0aa18326cb500586982067885db10a427c571673833baa51f2a`
- `artifact_manifest.json`: `9d7f0199e553c51ebafd96949cbf70f032ec25a045816809cc41ae198239c47a`
- terminal: `4f94c4b3610968a3b69ae6144dd97d44825cbd7fcce4e268b507f0212477d1a4`

Optimized cell:

- `SENSOR_PROFILE_RESULT.json`: `4f00d9ca94f6c29d441994927203b92bff7f45e3951bde28d467446a8e41015b`
- `run_manifest.json`: `e70390a8cc66ac4e33d99d10f70e06ec2cce7dcfc21add0a3fd8cfcbd5b833e8`
- `campaign_ledger.json`: `7a606520f234ab3736793dabf627b6fe8f0c63ce19edfb714eca70bb8433b713`
- `PASSED.json`: `5b05556e3dee48a9f9dfd90248178b794bfe0f226db797f89990374a46707ea3`
- `per_frame_metrics.csv`: `31b5dce767bb3a868f58fbe3e4a73da6b45a9f8c371dafd701be04d4fa18fa71`
- `direct_edge_publication.csv`: `bbd411aa61cb44634ff437fc6367935ea1ac1b5ac7f58e47c23945c58a715bcf`
- `direct_map_ingest.csv`: `54843a8be4ca6574374ef7d19ee1deeeaf14755d442250211362e60fb65dcffb`
- `artifact_manifest.json`: `0cc44ad097b823f3b690e71a377a22ec4b03dd0d3d4568943d4fe077723c4477`
- optimized success terminal: `5f26c611255ed1bde5e67fd72874993e3ce781481c6d66081f6ac1c885759331`

No completed 288-cell evidence was read for mutation or changed by this work.
