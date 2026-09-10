# SplitFusion live CARLA/OAI timing diagnostic — measured decomposition

Run `20260909_live_carla_actions30_15_50_71_retry3` · implementation commit `ce4a2e90f4a7a6b604b8e88c6657b9d7b5ff0f68` · 10.2 min wall · NVIDIA GeForce RTX 5090.

Four actions, four fresh CARLA lifecycles and four fresh CN5G/gNB/UE/edge lifecycles. Each cell drives the qualified live Route-B configuration — Town10HD_Opt `data_collection/routes/town10hd_opt_route_b_full_map_loop_v1.json`, 50 vehicles / 50 pedestrians, scenario seed 31, traffic-manager seed 31, Epic quality — with one action fixed throughout, FAVORABLE_STABLE restarted from its identical first sample, a 300-transmitted-frame budget and a 90 s safety timeout. The full Route-B loop is deliberately not completed.

## What each number is, and is not

- **No stored dataset frame is replayed.** Every measured payload is built from live CARLA RGB + radar. The only synthetic input is the edge's CUDA warm-up (a seeded random 7-channel tensor on its own stream), which the edge marks WARMUP and which is excluded from every statistic.
- **`application_feature_uplink_ms` = edge complete reassembly − UE first datagram send.** Application-level feature-uplink handling through OAI, **not** PHY/RLC latency: it includes the host UDP send path, the UE tunnel, the core and RAN transport, edge socket receipt, fragmentation and message reassembly. Sensor preparation and CARLA waiting are structurally excluded — the clock starts at the wall time taken immediately before the first datagram reaches the socket.
- **Two tail measurements, reported separately.** `decode_tail_cuda_ms` is a dedicated CUDA event pair with nothing but `model.decode_tail(batch, dense=False)` between the two records. `deployed_tail_service_ms` is the wall span from edge worker start to compact-result publication, so it carries live contention, post-processing, p025 filtering, segmentation construction, serialization and publication.
- The earlier round-trip residual is **not** reused or relabelled. Every uplink quantity is a one-way difference between two same-host `time.time_ns()` boundaries.

Clock domain: UE host and edge container verified to share one wall-clock domain from paired `time.time_ns()` / `time.monotonic_ns()` anchors at both ends of both processes (worst wall−monotonic offset skew 241 ns). No derived interval in any artifact is negative.

Collector configuration: the qualified collector runs with its map install and install-feedback deployment functions **enabled** and its segmentation-quality, object-ground-truth and exact-installed-record **evaluation scaffolding disabled**, so the deployed service span carries deployment contention without offline scoring work.

## Commissioned comparisons

**1 & 2 — pure `decode_tail` CUDA time versus the two published spans.** The ~73.4 ms Phase-13C `tail_gpu_ms` and the ~111.6 ms Phase-15 live `frozen_tail` are both spans over the *whole* tail adapter call, not over `decode_tail`.

| action | profile | decode_tail CUDA (ms) | deployed service (ms) | live frozen_tail (ms) | Δ CUDA vs 73.4 | Δ CUDA vs 111.6 | deployed / CUDA |
|---|---|---:|---:|---:|---:|---:|---:|
| 30 | `split_ae128_uint4_q0000` | 21.26 | 160.38 | 115.73 | -52.14 | -90.34 | 7.5x |
| 15 | `split_noae_uint4_q7000` | 21.24 | 161.12 | 115.56 | -52.16 | -90.36 | 7.6x |
| 50 | `split_ae64_uint4_q5000` | 20.39 | 144.50 | 109.33 | -53.01 | -91.21 | 7.1x |
| 71 | `split_ae32_uint4_q9800` | 20.79 | 136.34 | 103.44 | -52.61 | -90.81 | 6.6x |

**3 — where the difference goes.** Wall-clock stage groups partition the live tail service span (medians, ms), with detection count beside post-processing so the two are never conflated:

| action | camera_pose_reconstruct | tail_inference_block | camera_aware_postprocess | p025_service_filter | segmentation_upsample_argmax | compact_result_serialization | span | non-inference share | detections |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 0.22 | 28.44 | 71.48 | 14.72 | 0.07 | 17.64 | 132.57 | 78.6% | 35 |
| 15 | 0.20 | 28.83 | 71.91 | 13.18 | 0.07 | 20.19 | 134.38 | 78.5% | 42 |
| 50 | 0.20 | 27.35 | 66.53 | 13.79 | 0.07 | 18.54 | 126.48 | 78.4% | 38 |
| 71 | 0.13 | 28.17 | 61.51 | 13.32 | 0.06 | 23.93 | 127.12 | 77.8% | 51 |

**4 — application uplink latency versus payload** (descending payload):

| action | payload (B) | datagrams/msg | transmitted | complete reassemblies | complete fraction | uplink median (ms) | uplink p95 (ms) | UE send loop median (ms) | detections |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 916,272 | 74 | 300 | 299 | 99.7% | 140.3 | 2,705.9 | 0.49 | 35 |
| 15 | 488,455 | 40 | 300 | 300 | 100.0% | 76.1 | 550.8 | 0.25 | 42 |
| 50 | 265,045 | 22 | 300 | 300 | 100.0% | 57.9 | 106.0 | 0.16 | 38 |
| 71 | 6,464 | 1 | 300 | 300 | 100.0% | 35.6 | 55.1 | 0.03 | 51 |

**5 — does edge queue wait track the deployed tail service time?** Paired actions: 4. Queue-wait medians [44.8, 55.29, 52.12, 39.73] ms against deployed-service medians [160.38, 161.12, 144.5, 136.34] ms. Tracks service time: False.

> Detection counts differ per live scene, and post-processing and p025 filtering scale with detection count, so differences between actions must not be read as payload effects alone. Detection counts are reported beside post-processing latency for every action.

## Action 30 — `split_ae128_uint4_q0000`

Transmitted 300/300 frames (reached budget: True, stop reason `TRANSMITTED_BUDGET_REACHED`) over 30.9 s of route across 620 ticks. Preparation opportunities dropped: 8 ({'WARMUP_NO_COMPLETE_RADAR_WINDOW': 1, 'DROPPED_INCOMPLETE_RADAR_WINDOW': 1, 'DROPPED_REPLACED_BY_NEWER_FRAME': 7, 'SENT': 300}). Edge: 21,293 datagrams, 299 complete reassemblies, 1 incomplete expiries, 299 admissions (136 replacements), 163 tail completions, 163 compact results. On-time against the 100 ms service target: 0/163; within the 500 ms horizon: 86/163.

GPU: utilization median 54.0% (p95 72.0%), memory 8,578 MiB, concurrent compute processes median 6.0, 79 samples, throttle reasons observed: ['0x0000000000000000']. Radio telemetry `COLLECTED`: PUSCH SNR median 20.50 dB (19100 samples), final UL MCS median 25.0. Clean −50 dB restore verified: True.

### Live path through OAI and the deployed edge

| stage | n | median | p90 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| UE send loop (first->final datagram) | 300 | 0.49 | 1.24 | 1.58 | 0.32 | 16.27 |
| application feature uplink (first send->edge reassembled) | 163 | 140.27 | 2,298.04 | 2,705.89 | 62.22 | 2,820.63 |
| post-send to reassembly (final send->edge reassembled) | 163 | 139.53 | 2,297.67 | 2,705.41 | 53.27 | 2,820.13 |
| edge first datagram->complete reassembly | 163 | 69.48 | 155.08 | 249.94 | 46.98 | 280.48 |
| edge queue wait (reassembled->worker start) | 163 | 44.80 | 96.38 | 109.42 | 0.14 | 212.57 |
| zstd decompression | 163 | 0.87 | 1.18 | 1.36 | 0.76 | 8.26 |
| unpack / dequantization | 163 | 12.69 | 34.63 | 39.83 | 6.00 | 159.23 |
| AE decode | 163 | 1.66 | 2.70 | 3.60 | 1.00 | 34.14 |
| camera-pose / calibration reconstruction | 163 | 0.22 | 0.98 | 1.15 | 0.12 | 1.74 |
| decode_tail CUDA (device, pure) | 163 | 21.26 | 23.78 | 24.81 | 14.36 | 101.78 |
| decode_tail inference block (launch + completion) | 163 | 28.44 | 36.82 | 42.16 | 16.50 | 137.47 |
| camera-aware post-processing | 163 | 69.66 | 93.45 | 108.72 | 25.70 | 636.45 |
| p025 service filtering | 163 | 13.58 | 20.62 | 23.50 | 5.78 | 97.51 |
| post-processing + p025 combined | 163 | 83.05 | 112.16 | 121.70 | 34.52 | 704.25 |
| 720x1280 segmentation interpolate + argmax | 163 | 0.07 | 0.11 | 0.17 | 0.05 | 1.25 |
| compact object-result serialization | 163 | 17.64 | 28.25 | 33.61 | 6.23 | 137.52 |
| frozen_tail stage (deployed definition) | 163 | 115.73 | 143.59 | 166.89 | 52.30 | 851.57 |
| total edge processing | 163 | 158.78 | 199.91 | 229.63 | 80.55 | 1,031.85 |
| deployed tail service (worker start->result published) | 163 | 160.38 | 204.08 | 230.92 | 81.60 | 1,047.27 |
| detections per frame | 163 | 35.00 | 41.00 | 43.00 | 15.00 | 56.00 |

### Sensor preparation (reported; structurally outside every uplink interval)

| stage | n | median | p90 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| preparation queue wait | 300 | 45.87 | 77.03 | 92.52 | 21.98 | 289.21 |
| CARLA sensor wait | 300 | 11.38 | 19.64 | 24.08 | 0.00 | 41.76 |
| radar logical-sweep window | 300 | 4.67 | 8.18 | 10.28 | 1.45 | 44.21 |
| radar rasterisation | 300 | 20.91 | 33.46 | 46.88 | 11.27 | 202.08 |
| RGB conversion | 300 | 3.18 | 5.69 | 6.83 | 2.28 | 45.76 |
| scene snapshot | 300 | 0.10 | 0.14 | 0.23 | 0.08 | 1.72 |
| total pre-front preparation | 300 | 29.35 | 47.48 | 62.15 | 17.09 | 278.84 |
| UE front/ranker/AE/quantize/zstd | 300 | 32.86 | 51.07 | 56.50 | 14.58 | 246.14 |

## Action 15 — `split_noae_uint4_q7000`

Transmitted 300/300 frames (reached budget: True, stop reason `TRANSMITTED_BUDGET_REACHED`) over 30.6 s of route across 612 ticks. Preparation opportunities dropped: 5 ({'WARMUP_NO_COMPLETE_RADAR_WINDOW': 1, 'DROPPED_INCOMPLETE_RADAR_WINDOW': 1, 'DROPPED_REPLACED_BY_NEWER_FRAME': 4, 'SENT': 300}). Edge: 11,904 datagrams, 300 complete reassemblies, 0 incomplete expiries, 300 admissions (132 replacements), 168 tail completions, 168 compact results. On-time against the 100 ms service target: 0/168; within the 500 ms horizon: 131/168.

GPU: utilization median 52.0% (p95 75.0%), memory 8,699 MiB, concurrent compute processes median 6.0, 81 samples, throttle reasons observed: ['0x0000000000000000']. Radio telemetry `COLLECTED`: PUSCH SNR median 16.50 dB (13333 samples), final UL MCS median 20.0. Clean −50 dB restore verified: True.

### Live path through OAI and the deployed edge

| stage | n | median | p90 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| UE send loop (first->final datagram) | 300 | 0.25 | 0.51 | 0.73 | 0.20 | 8.07 |
| application feature uplink (first send->edge reassembled) | 168 | 76.12 | 416.17 | 550.85 | 41.74 | 823.01 |
| post-send to reassembly (final send->edge reassembled) | 168 | 75.66 | 415.92 | 550.37 | 41.33 | 822.81 |
| edge first datagram->complete reassembly | 168 | 38.93 | 130.00 | 134.96 | 26.17 | 157.67 |
| edge queue wait (reassembled->worker start) | 168 | 55.29 | 114.73 | 129.16 | 0.11 | 178.45 |
| zstd decompression | 168 | 0.45 | 0.69 | 0.78 | 0.43 | 2.77 |
| unpack / dequantization | 168 | 15.38 | 30.79 | 44.15 | 7.73 | 141.47 |
| AE decode | 168 | 2.00 | 2.94 | 3.28 | 1.46 | 11.21 |
| camera-pose / calibration reconstruction | 168 | 0.20 | 1.28 | 1.63 | 0.14 | 5.04 |
| decode_tail CUDA (device, pure) | 168 | 21.24 | 23.57 | 24.28 | 14.67 | 119.24 |
| decode_tail inference block (launch + completion) | 168 | 28.83 | 39.56 | 43.13 | 17.03 | 173.50 |
| camera-aware post-processing | 168 | 68.99 | 99.14 | 137.60 | 26.15 | 655.66 |
| p025 service filtering | 168 | 12.18 | 20.37 | 22.10 | 5.06 | 90.42 |
| post-processing + p025 combined | 168 | 82.10 | 113.67 | 161.58 | 32.96 | 746.08 |
| 720x1280 segmentation interpolate + argmax | 168 | 0.07 | 0.10 | 0.14 | 0.05 | 1.97 |
| compact object-result serialization | 168 | 20.19 | 32.83 | 38.72 | 9.28 | 164.26 |
| frozen_tail stage (deployed definition) | 168 | 115.56 | 158.04 | 201.59 | 51.97 | 854.45 |
| total edge processing | 168 | 158.27 | 207.66 | 247.98 | 87.16 | 1,037.49 |
| deployed tail service (worker start->result published) | 168 | 161.12 | 209.46 | 249.86 | 89.06 | 1,051.27 |
| detections per frame | 168 | 42.00 | 47.00 | 49.00 | 21.00 | 54.00 |

### Sensor preparation (reported; structurally outside every uplink interval)

| stage | n | median | p90 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| preparation queue wait | 300 | 42.94 | 65.89 | 80.24 | 23.47 | 143.92 |
| CARLA sensor wait | 300 | 11.63 | 18.99 | 21.87 | 0.00 | 43.94 |
| radar logical-sweep window | 300 | 4.33 | 7.98 | 9.95 | 1.48 | 38.79 |
| radar rasterisation | 300 | 20.87 | 32.27 | 37.63 | 11.59 | 117.16 |
| RGB conversion | 300 | 2.69 | 5.27 | 5.76 | 2.28 | 10.72 |
| scene snapshot | 300 | 0.10 | 0.15 | 0.21 | 0.08 | 3.34 |
| total pre-front preparation | 300 | 29.31 | 43.54 | 50.17 | 17.56 | 132.62 |
| UE front/ranker/AE/quantize/zstd | 300 | 27.11 | 45.99 | 57.23 | 10.52 | 288.23 |

## Action 50 — `split_ae64_uint4_q5000`

Transmitted 300/300 frames (reached budget: True, stop reason `TRANSMITTED_BUDGET_REACHED`) over 30.6 s of route across 611 ticks. Preparation opportunities dropped: 4 ({'WARMUP_NO_COMPLETE_RADAR_WINDOW': 1, 'DROPPED_INCOMPLETE_RADAR_WINDOW': 1, 'DROPPED_REPLACED_BY_NEWER_FRAME': 3, 'SENT': 300}). Edge: 6,567 datagrams, 300 complete reassemblies, 0 incomplete expiries, 300 admissions (100 replacements), 200 tail completions, 200 compact results. On-time against the 100 ms service target: 0/200; within the 500 ms horizon: 197/200.

GPU: utilization median 59.0% (p95 74.0%), memory 8,549 MiB, concurrent compute processes median 6.0, 78 samples, throttle reasons observed: ['0x0000000000000000']. Radio telemetry `COLLECTED`: PUSCH SNR median 16.50 dB (9303 samples), final UL MCS median 20.0. Clean −50 dB restore verified: True.

### Live path through OAI and the deployed edge

| stage | n | median | p90 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| UE send loop (first->final datagram) | 300 | 0.16 | 0.38 | 0.63 | 0.13 | 3.84 |
| application feature uplink (first send->edge reassembled) | 200 | 57.94 | 86.10 | 105.96 | 24.52 | 168.07 |
| post-send to reassembly (final send->edge reassembled) | 200 | 57.79 | 85.94 | 105.67 | 24.25 | 167.77 |
| edge first datagram->complete reassembly | 200 | 20.86 | 59.88 | 72.15 | 13.25 | 112.16 |
| edge queue wait (reassembled->worker start) | 200 | 52.12 | 101.17 | 115.05 | 0.08 | 148.72 |
| zstd decompression | 200 | 0.22 | 0.30 | 0.44 | 0.22 | 0.81 |
| unpack / dequantization | 200 | 6.39 | 17.18 | 23.45 | 2.65 | 35.23 |
| AE decode | 200 | 1.03 | 1.77 | 2.22 | 0.60 | 18.21 |
| camera-pose / calibration reconstruction | 200 | 0.20 | 0.94 | 1.17 | 0.12 | 2.61 |
| decode_tail CUDA (device, pure) | 200 | 20.39 | 23.16 | 24.31 | 14.33 | 205.30 |
| decode_tail inference block (launch + completion) | 200 | 27.35 | 34.85 | 39.68 | 16.48 | 250.81 |
| camera-aware post-processing | 200 | 64.84 | 81.47 | 84.95 | 26.15 | 238.74 |
| p025 service filtering | 200 | 12.26 | 20.81 | 24.97 | 6.42 | 48.45 |
| post-processing + p025 combined | 200 | 77.39 | 95.19 | 103.34 | 33.69 | 287.19 |
| 720x1280 segmentation interpolate + argmax | 200 | 0.07 | 0.08 | 0.11 | 0.05 | 0.65 |
| compact object-result serialization | 200 | 18.54 | 27.50 | 30.13 | 6.02 | 62.66 |
| frozen_tail stage (deployed definition) | 200 | 109.33 | 132.06 | 137.98 | 53.47 | 394.78 |
| total edge processing | 200 | 142.81 | 166.61 | 179.55 | 74.69 | 427.43 |
| deployed tail service (worker start->result published) | 200 | 144.50 | 169.65 | 182.11 | 75.86 | 428.71 |
| detections per frame | 200 | 38.00 | 43.00 | 46.00 | 21.00 | 52.00 |

### Sensor preparation (reported; structurally outside every uplink interval)

| stage | n | median | p90 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| preparation queue wait | 300 | 40.15 | 61.13 | 71.28 | 20.71 | 172.39 |
| CARLA sensor wait | 300 | 11.31 | 20.15 | 23.00 | 0.00 | 62.98 |
| radar logical-sweep window | 300 | 4.28 | 8.03 | 9.57 | 1.21 | 28.54 |
| radar rasterisation | 300 | 19.04 | 28.55 | 34.12 | 10.63 | 144.60 |
| RGB conversion | 300 | 2.54 | 5.44 | 6.09 | 2.29 | 9.34 |
| scene snapshot | 300 | 0.09 | 0.13 | 0.22 | 0.08 | 1.13 |
| total pre-front preparation | 300 | 27.12 | 40.23 | 47.08 | 14.65 | 158.61 |
| UE front/ranker/AE/quantize/zstd | 300 | 26.81 | 45.93 | 51.85 | 7.44 | 295.25 |

## Action 71 — `split_ae32_uint4_q9800`

Transmitted 300/300 frames (reached budget: True, stop reason `TRANSMITTED_BUDGET_REACHED`) over 30.4 s of route across 609 ticks. Preparation opportunities dropped: 3 ({'WARMUP_NO_COMPLETE_RADAR_WINDOW': 1, 'DROPPED_INCOMPLETE_RADAR_WINDOW': 1, 'DROPPED_REPLACED_BY_NEWER_FRAME': 2, 'SENT': 300}). Edge: 300 datagrams, 300 complete reassemblies, 0 incomplete expiries, 300 admissions (117 replacements), 183 tail completions, 183 compact results. On-time against the 100 ms service target: 0/183; within the 500 ms horizon: 174/183.

GPU: utilization median 52.0% (p95 80.0%), memory 8,539 MiB, concurrent compute processes median 6.0, 83 samples, throttle reasons observed: ['0x0000000000000000']. Radio telemetry `COLLECTED`: PUSCH SNR median 21.00 dB (2994 samples), final UL MCS median 25.0. Clean −50 dB restore verified: True.

### Live path through OAI and the deployed edge

| stage | n | median | p90 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| UE send loop (first->final datagram) | 300 | 0.03 | 0.06 | 0.06 | 0.02 | 0.62 |
| application feature uplink (first send->edge reassembled) | 183 | 35.59 | 45.28 | 55.08 | 5.79 | 75.72 |
| post-send to reassembly (final send->edge reassembled) | 183 | 35.57 | 45.25 | 55.06 | 5.74 | 75.69 |
| edge first datagram->complete reassembly | 183 | 0.05 | 0.07 | 0.07 | 0.01 | 0.42 |
| edge queue wait (reassembled->worker start) | 183 | 39.73 | 101.40 | 113.92 | 0.08 | 141.28 |
| zstd decompression | 183 | 0.02 | 0.02 | 0.02 | 0.02 | 0.63 |
| unpack / dequantization | 183 | 1.70 | 6.78 | 8.91 | 0.71 | 20.92 |
| AE decode | 183 | 0.62 | 1.48 | 1.77 | 0.39 | 15.72 |
| camera-pose / calibration reconstruction | 183 | 0.13 | 1.16 | 1.35 | 0.09 | 2.62 |
| decode_tail CUDA (device, pure) | 183 | 20.79 | 23.25 | 24.74 | 14.12 | 166.51 |
| decode_tail inference block (launch + completion) | 183 | 28.17 | 37.73 | 46.63 | 16.39 | 217.65 |
| camera-aware post-processing | 183 | 60.11 | 89.70 | 153.63 | 26.71 | 392.67 |
| p025 service filtering | 183 | 12.50 | 24.88 | 37.61 | 6.04 | 154.58 |
| post-processing + p025 combined | 183 | 72.44 | 111.42 | 159.68 | 35.11 | 488.56 |
| 720x1280 segmentation interpolate + argmax | 183 | 0.06 | 0.08 | 0.12 | 0.05 | 1.86 |
| compact object-result serialization | 183 | 23.93 | 40.07 | 55.23 | 10.60 | 326.98 |
| frozen_tail stage (deployed definition) | 183 | 103.44 | 151.86 | 226.48 | 53.18 | 696.34 |
| total edge processing | 183 | 133.94 | 196.73 | 311.79 | 68.71 | 1,063.84 |
| deployed tail service (worker start->result published) | 183 | 136.34 | 200.61 | 319.25 | 69.91 | 1,085.37 |
| detections per frame | 183 | 51.00 | 58.00 | 60.00 | 35.00 | 64.00 |

### Sensor preparation (reported; structurally outside every uplink interval)

| stage | n | median | p90 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| preparation queue wait | 300 | 41.26 | 57.84 | 66.14 | 24.45 | 127.98 |
| CARLA sensor wait | 300 | 11.82 | 21.61 | 23.36 | 0.00 | 86.33 |
| radar logical-sweep window | 300 | 3.79 | 6.23 | 7.19 | 1.29 | 21.22 |
| radar rasterisation | 300 | 20.70 | 28.79 | 38.98 | 12.60 | 105.18 |
| RGB conversion | 300 | 2.50 | 5.42 | 6.32 | 2.29 | 9.83 |
| scene snapshot | 300 | 0.09 | 0.14 | 0.16 | 0.08 | 1.47 |
| total pre-front preparation | 300 | 27.68 | 39.94 | 46.90 | 17.63 | 118.49 |
| UE front/ranker/AE/quantize/zstd | 300 | 17.65 | 31.40 | 37.62 | 6.37 | 304.81 |

## Limitations

- **Live scenes are not matched across actions.** Each cell drives the same route from the same start with the same seeds, but a bounded 300-frame window covers a different stretch of road depending on how fast frames were transmitted, and detection counts differ. Post-processing and p025 filtering scale with detection count, so between-action differences are **not** attributable to payload alone. Detection counts are reported beside post-processing latency.
- **The Route-B loop is not completed**, so no route-completion, preparation-coverage or age-of-information claim can be read off this run, and the campaign's own route/coverage acceptance gate is inapplicable and unused.
- **The diagnostic edge classifies deadlines instead of dropping.** Each frame carries `service_target_met` / `processing_horizon_met` and is still processed, because dropping late frames would delete the decomposition being measured. Model, codec, thresholds, action definitions and wire bytes are unchanged, and the instrumented tail was proved bit-identical to the production tail before measurement.
- **Per-stage CUDA and wall times overlap by construction.** `decode_tail_cuda_ms` is device time for the tail kernels; most of that same time appears as wall time in the immediately following finite check, where the deployed path first synchronizes. The additive wall partition is the stage-group table; the CUDA columns are the device-side attribution of the same work.
- Warm-up used a seeded synthetic tensor rather than a live or stored frame, so its detection count and therefore its post-processing cost differ from the live scenes; warm-up frames are excluded from every statistic.
- One bounded execution per action; medians are within-run, so no run-to-run variance is characterized. **No optimization was attempted.**

## Artifacts

`LIVE_RUN_MANIFEST.json` (immutable pre-run binding), `per_frame/action_*.csv` (one per action), `action_summary.csv`, `LIVE_DIAGNOSTIC_RESULTS.json`, `REPORT.md`, `live_timing_breakdown.pdf`, `live_timing_breakdown.png`, `ARTIFACT_MANIFEST.json`, terminal file. No RGB or radar frame, C2 tensor, compressed payload blob, prediction, segmentation label map or raw OAI tracer log is retained.

