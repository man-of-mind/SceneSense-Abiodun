# SplitFusion edge optimization — four-action live before/after

Run `20260909_live_actions30_15_50_71` · implementation commit `ce6af96f258af663fd26b23423158a2ed0f87f41` · 9.9 min wall · NVIDIA GeForce RTX 5090.

Four actions, four fresh CARLA lifecycles and four fresh CN5G/gNB/UE/edge lifecycles. Each cell drives the qualified live Route-B configuration — Town10HD_Opt `data_collection/routes/town10hd_opt_route_b_full_map_loop_v1.json`, 50 vehicles / 50 pedestrians, scenario seed 31, traffic-manager seed 31, Epic quality — with one action fixed throughout, FAVORABLE_STABLE restarted from its identical first sample, a 300-transmitted-frame budget and a 90 s safety timeout. The full Route-B loop is deliberately not completed.

## What each number is, and is not

- **No stored dataset frame is replayed.** Every measured payload is built from live CARLA RGB + radar. The only synthetic input is the edge's CUDA warm-up (a seeded random 7-channel tensor on its own stream), which the edge marks WARMUP and which is excluded from every statistic.
- **`application_feature_uplink_ms` = edge complete reassembly − UE first datagram send.** Application-level feature-uplink handling through OAI, **not** PHY/RLC latency: it includes the host UDP send path, the UE tunnel, the core and RAN transport, edge socket receipt, fragmentation and message reassembly. Sensor preparation and CARLA waiting are structurally excluded — the clock starts at the wall time taken immediately before the first datagram reaches the socket.
- **Two tail measurements, reported separately.** `decode_tail_cuda_ms` is a dedicated CUDA event pair with nothing but `model.decode_tail(batch, dense=False)` between the two records. `deployed_tail_service_ms` is the wall span from edge worker start to compact-result publication, so it carries live contention, post-processing, p025 filtering, segmentation construction, serialization and publication.
- The earlier round-trip residual is **not** reused or relabelled. Every uplink quantity is a one-way difference between two same-host `time.time_ns()` boundaries.

Clock domain: UE host and edge container verified to share one wall-clock domain from paired `time.time_ns()` / `time.monotonic_ns()` anchors at both ends of both processes (worst wall−monotonic offset skew 98 ns). No derived interval in any artifact is negative.

Collector configuration: the qualified collector runs with its map install and install-feedback deployment functions **enabled** and its segmentation-quality, object-ground-truth and exact-installed-record **evaluation scaffolding disabled**, so the deployed service span carries deployment contention without offline scoring work.

## Commissioned comparisons

**1 & 2 — pure `decode_tail` CUDA time versus the two published spans.** The ~73.4 ms Phase-13C `tail_gpu_ms` and the ~111.6 ms Phase-15 live `frozen_tail` are both spans over the *whole* tail adapter call, not over `decode_tail`.

| action | profile | decode_tail CUDA (ms) | deployed service (ms) | live frozen_tail (ms) | Δ CUDA vs 73.4 | Δ CUDA vs 111.6 | deployed / CUDA |
|---|---|---:|---:|---:|---:|---:|---:|
| 30 | `split_ae128_uint4_q0000` | 20.81 | 116.12 | 88.67 | -52.59 | -90.79 | 5.6x |
| 15 | `split_noae_uint4_q7000` | 21.17 | 122.71 | 92.10 | -52.23 | -90.43 | 5.8x |
| 50 | `split_ae64_uint4_q5000` | 20.51 | 99.48 | 81.20 | -52.89 | -91.09 | 4.9x |
| 71 | `split_ae32_uint4_q9800` | 20.23 | 95.27 | 82.81 | -53.17 | -91.37 | 4.7x |

**3 — where the difference goes.** Wall-clock stage groups partition the live tail service span (medians, ms), with detection count beside post-processing so the two are never conflated:

| action | camera_pose_reconstruct | tail_inference_block | camera_aware_postprocess | p025_service_filter | segmentation_upsample_argmax | compact_result_serialization | span | non-inference share | detections |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 0.23 | 29.06 | 43.77 | 11.93 | 0.07 | 2.66 | 87.71 | 66.9% | 36 |
| 15 | 0.19 | 29.40 | 47.89 | 11.89 | 0.07 | 3.08 | 92.51 | 68.2% | 42 |
| 50 | 0.21 | 28.38 | 37.67 | 12.42 | 0.07 | 2.54 | 81.30 | 65.1% | 36 |
| 71 | 0.14 | 26.60 | 45.07 | 9.77 | 0.07 | 3.28 | 84.91 | 68.7% | 51 |

**4 — application uplink latency versus payload** (descending payload):

| action | payload (B) | datagrams/msg | transmitted | complete reassemblies | complete fraction | uplink median (ms) | uplink p95 (ms) | UE send loop median (ms) | detections |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 924,112 | 74 | 300 | 298 | 99.3% | 670.1 | 3,347.3 | 0.44 | 36 |
| 15 | 488,786 | 40 | 300 | 300 | 100.0% | 74.9 | 390.0 | 0.28 | 42 |
| 50 | 264,332 | 22 | 300 | 300 | 100.0% | 57.2 | 117.8 | 0.17 | 36 |
| 71 | 6,423 | 1 | 300 | 300 | 100.0% | 36.3 | 52.6 | 0.03 | 51 |

**5 — does edge queue wait track the deployed tail service time?** Paired actions: 4. Queue-wait medians [30.0, 47.03, 30.7, 15.49] ms against deployed-service medians [116.12, 122.71, 99.48, 95.27] ms. Tracks service time: False.

> Detection counts differ per live scene, and post-processing and p025 filtering scale with detection count, so differences between actions must not be read as payload effects alone. Detection counts are reported beside post-processing latency for every action.

## Action 30 — `split_ae128_uint4_q0000`

Transmitted 300/300 frames (reached budget: True, stop reason `TRANSMITTED_BUDGET_REACHED`) over 30.9 s of route across 619 ticks. Preparation opportunities dropped: 8 ({'WARMUP_NO_COMPLETE_RADAR_WINDOW': 1, 'DROPPED_INCOMPLETE_RADAR_WINDOW': 1, 'DROPPED_REPLACED_BY_NEWER_FRAME': 6, 'STALE_BEFORE_SEND': 1, 'SENT': 300}). Edge: 21,457 datagrams, 298 complete reassemblies, 2 incomplete expiries, 298 admissions (72 replacements), 226 tail completions, 226 compact results. On-time against the 100 ms service target: 0/226; within the 500 ms horizon: 96/226.

GPU: utilization median 49.0% (p95 70.0%), memory 8,602 MiB, concurrent compute processes median 6.0, 86 samples, throttle reasons observed: ['0x0000000000000000']. Radio telemetry `COLLECTED`: PUSCH SNR median 20.50 dB (18993 samples), final UL MCS median 25.0. Clean −50 dB restore verified: True.

### Live path through OAI and the deployed edge

| stage | n | median | p90 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| UE send loop (first->final datagram) | 300 | 0.44 | 1.04 | 1.36 | 0.31 | 14.20 |
| application feature uplink (first send->edge reassembled) | 226 | 670.15 | 3,220.23 | 3,347.30 | 63.67 | 3,644.25 |
| post-send to reassembly (final send->edge reassembled) | 226 | 669.74 | 3,219.55 | 3,346.93 | 62.04 | 3,643.72 |
| edge first datagram->complete reassembly | 226 | 71.84 | 139.98 | 226.87 | 45.38 | 310.60 |
| edge queue wait (reassembled->worker start) | 226 | 30.00 | 85.33 | 96.82 | 0.11 | 129.54 |
| zstd decompression | 226 | 0.86 | 1.10 | 1.23 | 0.77 | 7.22 |
| unpack / dequantization | 226 | 14.13 | 33.48 | 44.23 | 6.97 | 89.59 |
| AE decode | 226 | 1.74 | 2.54 | 2.97 | 1.06 | 17.89 |
| camera-pose / calibration reconstruction | 226 | 0.23 | 1.20 | 1.39 | 0.13 | 4.38 |
| decode_tail CUDA (device, pure) | 226 | 20.81 | 23.78 | 24.80 | 14.39 | 131.53 |
| decode_tail inference block (launch + completion) | 226 | 29.06 | 37.94 | 40.84 | 16.51 | 191.01 |
| camera-aware post-processing | 226 | 42.67 | 65.80 | 70.21 | 13.93 | 477.94 |
| p025 service filtering | 226 | 11.27 | 18.57 | 22.08 | 6.06 | 27.42 |
| post-processing + p025 combined | 226 | 54.71 | 77.33 | 84.49 | 20.30 | 484.19 |
| 720x1280 segmentation interpolate + argmax | 226 | 0.07 | 0.10 | 0.16 | 0.05 | 0.74 |
| compact object-result serialization | 226 | 2.66 | 3.81 | 4.15 | 1.42 | 10.93 |
| frozen_tail stage (deployed definition) | 226 | 88.67 | 112.86 | 121.23 | 38.19 | 681.42 |
| total edge processing | 226 | 113.87 | 150.83 | 161.57 | 56.98 | 770.06 |
| deployed tail service (worker start->result published) | 226 | 116.12 | 153.83 | 166.36 | 58.20 | 771.49 |
| detections per frame | 226 | 36.00 | 42.00 | 45.00 | 16.00 | 54.00 |

### Sensor preparation (reported; structurally outside every uplink interval)

| stage | n | median | p90 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| preparation queue wait | 300 | 46.85 | 81.36 | 90.83 | 21.26 | 140.84 |
| CARLA sensor wait | 300 | 10.35 | 21.06 | 23.91 | 0.00 | 72.49 |
| radar logical-sweep window | 300 | 4.51 | 8.24 | 9.39 | 1.26 | 26.18 |
| radar rasterisation | 300 | 21.19 | 33.86 | 43.85 | 11.60 | 108.99 |
| RGB conversion | 300 | 2.96 | 5.85 | 7.39 | 2.20 | 17.77 |
| scene snapshot | 300 | 0.10 | 0.15 | 0.27 | 0.08 | 0.66 |
| total pre-front preparation | 300 | 29.67 | 45.93 | 55.16 | 15.67 | 119.73 |
| UE front/ranker/AE/quantize/zstd | 300 | 35.36 | 55.40 | 62.10 | 14.91 | 72.20 |

## Action 15 — `split_noae_uint4_q7000`

Transmitted 300/300 frames (reached budget: True, stop reason `TRANSMITTED_BUDGET_REACHED`) over 31.1 s of route across 623 ticks. Preparation opportunities dropped: 10 ({'WARMUP_NO_COMPLETE_RADAR_WINDOW': 1, 'DROPPED_INCOMPLETE_RADAR_WINDOW': 1, 'DROPPED_REPLACED_BY_NEWER_FRAME': 9, 'SENT': 300}). Edge: 11,907 datagrams, 300 complete reassemblies, 0 incomplete expiries, 300 admissions (70 replacements), 230 tail completions, 230 compact results. On-time against the 100 ms service target: 0/230; within the 500 ms horizon: 206/230.

GPU: utilization median 56.0% (p95 74.0%), memory 8,611 MiB, concurrent compute processes median 6.0, 77 samples, throttle reasons observed: ['0x0000000000000000']. Radio telemetry `COLLECTED`: PUSCH SNR median 17.00 dB (13280 samples), final UL MCS median 20.0. Clean −50 dB restore verified: True.

### Live path through OAI and the deployed edge

| stage | n | median | p90 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| UE send loop (first->final datagram) | 300 | 0.28 | 0.71 | 1.03 | 0.20 | 7.50 |
| application feature uplink (first send->edge reassembled) | 230 | 74.91 | 242.60 | 389.97 | 39.24 | 577.91 |
| post-send to reassembly (final send->edge reassembled) | 230 | 74.49 | 242.38 | 389.69 | 39.02 | 576.80 |
| edge first datagram->complete reassembly | 230 | 37.72 | 116.13 | 133.92 | 26.09 | 155.40 |
| edge queue wait (reassembled->worker start) | 230 | 47.03 | 91.68 | 101.72 | 0.11 | 150.46 |
| zstd decompression | 230 | 0.47 | 0.59 | 0.67 | 0.42 | 4.91 |
| unpack / dequantization | 230 | 17.85 | 40.94 | 52.16 | 10.02 | 75.14 |
| AE decode | 230 | 2.02 | 2.78 | 3.08 | 1.46 | 8.71 |
| camera-pose / calibration reconstruction | 230 | 0.19 | 1.15 | 1.36 | 0.13 | 2.35 |
| decode_tail CUDA (device, pure) | 230 | 21.17 | 23.60 | 24.36 | 14.42 | 48.13 |
| decode_tail inference block (launch + completion) | 230 | 29.40 | 38.53 | 40.38 | 16.55 | 54.72 |
| camera-aware post-processing | 230 | 46.33 | 63.21 | 70.69 | 14.65 | 176.08 |
| p025 service filtering | 230 | 10.27 | 17.97 | 21.50 | 5.68 | 97.39 |
| post-processing + p025 combined | 230 | 58.10 | 75.36 | 84.51 | 20.79 | 234.21 |
| 720x1280 segmentation interpolate + argmax | 230 | 0.07 | 0.09 | 0.12 | 0.06 | 1.21 |
| compact object-result serialization | 230 | 3.08 | 4.10 | 5.16 | 1.73 | 100.98 |
| frozen_tail stage (deployed definition) | 230 | 92.10 | 113.60 | 122.98 | 42.65 | 289.85 |
| total edge processing | 230 | 119.81 | 154.72 | 169.53 | 62.55 | 353.76 |
| deployed tail service (worker start->result published) | 230 | 122.71 | 159.29 | 174.64 | 67.63 | 366.59 |
| detections per frame | 230 | 42.00 | 49.00 | 50.00 | 23.00 | 53.00 |

### Sensor preparation (reported; structurally outside every uplink interval)

| stage | n | median | p90 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| preparation queue wait | 300 | 44.00 | 81.20 | 102.64 | 23.14 | 249.00 |
| CARLA sensor wait | 300 | 11.79 | 21.06 | 23.48 | 0.00 | 49.84 |
| radar logical-sweep window | 300 | 4.33 | 7.97 | 9.36 | 1.28 | 75.17 |
| radar rasterisation | 300 | 19.69 | 34.79 | 56.84 | 11.23 | 195.72 |
| RGB conversion | 300 | 2.58 | 6.19 | 7.55 | 2.27 | 14.81 |
| scene snapshot | 300 | 0.09 | 0.15 | 0.28 | 0.08 | 1.03 |
| total pre-front preparation | 300 | 27.81 | 46.15 | 68.20 | 15.34 | 225.24 |
| UE front/ranker/AE/quantize/zstd | 300 | 31.26 | 55.67 | 64.56 | 11.88 | 322.86 |

## Action 50 — `split_ae64_uint4_q5000`

Transmitted 300/300 frames (reached budget: True, stop reason `TRANSMITTED_BUDGET_REACHED`) over 30.4 s of route across 609 ticks. Preparation opportunities dropped: 3 ({'WARMUP_NO_COMPLETE_RADAR_WINDOW': 1, 'DROPPED_INCOMPLETE_RADAR_WINDOW': 1, 'DROPPED_REPLACED_BY_NEWER_FRAME': 2, 'SENT': 300}). Edge: 6,571 datagrams, 300 complete reassemblies, 0 incomplete expiries, 300 admissions (26 replacements), 274 tail completions, 274 compact results. On-time against the 100 ms service target: 0/274; within the 500 ms horizon: 273/274.

GPU: utilization median 59.0% (p95 72.0%), memory 8,574 MiB, concurrent compute processes median 6.0, 76 samples, throttle reasons observed: ['0x0000000000000000']. Radio telemetry `COLLECTED`: PUSCH SNR median 17.00 dB (9181 samples), final UL MCS median 21.0. Clean −50 dB restore verified: True.

### Live path through OAI and the deployed edge

| stage | n | median | p90 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| UE send loop (first->final datagram) | 300 | 0.17 | 0.43 | 0.85 | 0.12 | 11.25 |
| application feature uplink (first send->edge reassembled) | 274 | 57.21 | 98.80 | 117.79 | 20.88 | 194.31 |
| post-send to reassembly (final send->edge reassembled) | 274 | 56.92 | 98.62 | 117.62 | 20.66 | 194.14 |
| edge first datagram->complete reassembly | 274 | 21.65 | 60.20 | 72.26 | 3.01 | 96.36 |
| edge queue wait (reassembled->worker start) | 274 | 30.70 | 79.68 | 90.83 | 0.09 | 134.59 |
| zstd decompression | 274 | 0.23 | 0.32 | 0.42 | 0.21 | 2.51 |
| unpack / dequantization | 274 | 7.44 | 20.98 | 24.56 | 2.89 | 44.54 |
| AE decode | 274 | 1.05 | 1.99 | 2.34 | 0.56 | 26.84 |
| camera-pose / calibration reconstruction | 274 | 0.21 | 1.02 | 1.14 | 0.11 | 3.67 |
| decode_tail CUDA (device, pure) | 274 | 20.51 | 23.16 | 24.12 | 14.14 | 96.15 |
| decode_tail inference block (launch + completion) | 274 | 28.38 | 36.75 | 39.30 | 16.42 | 101.21 |
| camera-aware post-processing | 274 | 36.00 | 54.07 | 58.50 | 13.46 | 70.51 |
| p025 service filtering | 274 | 11.17 | 20.39 | 22.80 | 5.87 | 43.05 |
| post-processing + p025 combined | 274 | 48.92 | 66.11 | 72.83 | 20.81 | 93.31 |
| 720x1280 segmentation interpolate + argmax | 274 | 0.07 | 0.09 | 0.12 | 0.05 | 0.57 |
| compact object-result serialization | 274 | 2.54 | 3.22 | 3.68 | 1.45 | 5.01 |
| frozen_tail stage (deployed definition) | 274 | 81.20 | 99.43 | 106.43 | 40.16 | 177.96 |
| total edge processing | 274 | 97.05 | 118.79 | 133.78 | 48.58 | 217.17 |
| deployed tail service (worker start->result published) | 274 | 99.48 | 123.03 | 136.68 | 50.65 | 220.64 |
| detections per frame | 274 | 36.00 | 44.00 | 47.00 | 19.00 | 53.00 |

### Sensor preparation (reported; structurally outside every uplink interval)

| stage | n | median | p90 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| preparation queue wait | 300 | 45.12 | 65.55 | 79.81 | 23.32 | 180.65 |
| CARLA sensor wait | 300 | 10.87 | 20.32 | 23.14 | 0.00 | 31.72 |
| radar logical-sweep window | 300 | 4.40 | 7.98 | 10.21 | 1.53 | 91.16 |
| radar rasterisation | 300 | 20.69 | 31.58 | 41.38 | 12.51 | 153.01 |
| RGB conversion | 300 | 4.02 | 6.62 | 7.65 | 2.29 | 24.92 |
| scene snapshot | 300 | 0.10 | 0.14 | 0.21 | 0.08 | 0.80 |
| total pre-front preparation | 300 | 30.37 | 45.56 | 53.98 | 16.91 | 161.75 |
| UE front/ranker/AE/quantize/zstd | 300 | 26.56 | 46.81 | 51.49 | 8.38 | 298.49 |

## Action 71 — `split_ae32_uint4_q9800`

Transmitted 300/300 frames (reached budget: True, stop reason `TRANSMITTED_BUDGET_REACHED`) over 30.4 s of route across 609 ticks. Preparation opportunities dropped: 3 ({'WARMUP_NO_COMPLETE_RADAR_WINDOW': 1, 'DROPPED_INCOMPLETE_RADAR_WINDOW': 1, 'DROPPED_REPLACED_BY_NEWER_FRAME': 2, 'SENT': 300}). Edge: 300 datagrams, 300 complete reassemblies, 0 incomplete expiries, 300 admissions (16 replacements), 284 tail completions, 284 compact results. On-time against the 100 ms service target: 0/284; within the 500 ms horizon: 282/284.

GPU: utilization median 65.0% (p95 76.0%), memory 8,564 MiB, concurrent compute processes median 6.0, 77 samples, throttle reasons observed: ['0x0000000000000000']. Radio telemetry `COLLECTED`: PUSCH SNR median 21.00 dB (3047 samples), final UL MCS median 25.0. Clean −50 dB restore verified: True.

### Live path through OAI and the deployed edge

| stage | n | median | p90 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| UE send loop (first->final datagram) | 300 | 0.03 | 0.06 | 0.07 | 0.01 | 0.86 |
| application feature uplink (first send->edge reassembled) | 284 | 36.29 | 42.35 | 52.55 | 8.26 | 70.70 |
| post-send to reassembly (final send->edge reassembled) | 284 | 36.24 | 42.32 | 52.53 | 8.03 | 70.64 |
| edge first datagram->complete reassembly | 284 | 0.05 | 0.08 | 0.09 | 0.03 | 1.21 |
| edge queue wait (reassembled->worker start) | 284 | 15.49 | 65.72 | 77.80 | 0.07 | 111.10 |
| zstd decompression | 284 | 0.02 | 0.03 | 0.03 | 0.02 | 1.69 |
| unpack / dequantization | 284 | 2.57 | 8.34 | 10.61 | 0.76 | 33.57 |
| AE decode | 284 | 0.73 | 1.48 | 1.61 | 0.39 | 12.00 |
| camera-pose / calibration reconstruction | 284 | 0.14 | 1.10 | 1.22 | 0.08 | 2.55 |
| decode_tail CUDA (device, pure) | 284 | 20.23 | 22.74 | 23.28 | 14.14 | 45.49 |
| decode_tail inference block (launch + completion) | 284 | 26.60 | 34.38 | 38.88 | 16.49 | 73.34 |
| camera-aware post-processing | 284 | 43.86 | 60.13 | 63.75 | 11.26 | 465.23 |
| p025 service filtering | 284 | 8.83 | 16.98 | 19.17 | 5.47 | 75.00 |
| post-processing + p025 combined | 284 | 53.46 | 71.09 | 77.80 | 20.60 | 478.73 |
| 720x1280 segmentation interpolate + argmax | 284 | 0.07 | 0.07 | 0.08 | 0.05 | 0.18 |
| compact object-result serialization | 284 | 3.28 | 3.97 | 4.31 | 2.12 | 6.91 |
| frozen_tail stage (deployed definition) | 284 | 82.81 | 107.25 | 115.19 | 38.64 | 553.70 |
| total edge processing | 284 | 92.34 | 117.34 | 125.99 | 45.93 | 564.89 |
| deployed tail service (worker start->result published) | 284 | 95.27 | 119.78 | 128.46 | 47.30 | 566.30 |
| detections per frame | 284 | 51.00 | 61.00 | 63.00 | 31.00 | 69.00 |

### Sensor preparation (reported; structurally outside every uplink interval)

| stage | n | median | p90 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| preparation queue wait | 300 | 45.44 | 65.45 | 73.54 | 23.24 | 124.91 |
| CARLA sensor wait | 300 | 11.23 | 22.97 | 25.53 | 0.00 | 46.91 |
| radar logical-sweep window | 300 | 3.95 | 6.88 | 8.37 | 1.45 | 35.43 |
| radar rasterisation | 300 | 23.16 | 35.06 | 41.21 | 14.94 | 95.66 |
| RGB conversion | 300 | 2.58 | 5.74 | 6.45 | 2.31 | 16.71 |
| scene snapshot | 300 | 0.09 | 0.13 | 0.17 | 0.08 | 2.67 |
| total pre-front preparation | 300 | 31.97 | 46.31 | 56.10 | 19.03 | 113.75 |
| UE front/ranker/AE/quantize/zstd | 300 | 20.21 | 36.56 | 40.60 | 5.84 | 294.38 |

## Limitations

- **Live scenes are not matched across actions.** Each cell drives the same route from the same start with the same seeds, but a bounded 300-frame window covers a different stretch of road depending on how fast frames were transmitted, and detection counts differ. Post-processing and p025 filtering scale with detection count, so between-action differences are **not** attributable to payload alone. Detection counts are reported beside post-processing latency.
- **The Route-B loop is not completed**, so no route-completion, preparation-coverage or age-of-information claim can be read off this run, and the campaign's own route/coverage acceptance gate is inapplicable and unused.
- **The diagnostic edge classifies deadlines instead of dropping.** Each frame carries `service_target_met` / `processing_horizon_met` and is still processed, because dropping late frames would delete the decomposition being measured. Model, codec, thresholds, action definitions and wire bytes are unchanged, and the instrumented tail was proved bit-identical to the production tail before measurement.
- **Per-stage CUDA and wall times overlap by construction.** `decode_tail_cuda_ms` is device time for the tail kernels; most of that same time appears as wall time in the immediately following finite check, where the deployed path first synchronizes. The additive wall partition is the stage-group table; the CUDA columns are the device-side attribution of the same work.
- Warm-up used a seeded synthetic tensor rather than a live or stored frame, so its detection count and therefore its post-processing cost differ from the live scenes; warm-up frames are excluded from every statistic.
- One bounded execution per action; medians are within-run, so no run-to-run variance is characterized. The output-preserving edge optimization candidate was exercised.

## Artifacts

`LIVE_RUN_MANIFEST.json` (immutable pre-run binding), `per_frame/action_*.csv` (one per action), `action_summary.csv`, `LIVE_DIAGNOSTIC_RESULTS.json`, `REPORT.md`, `live_timing_breakdown.pdf`, `live_timing_breakdown.png`, `ARTIFACT_MANIFEST.json`, terminal file. No RGB or radar frame, C2 tensor, compressed payload blob, prediction, segmentation label map or raw OAI tracer log is retained.

## Before/after edge optimization

The baseline and candidate use the same four actions, live Route-B contract and FAVORABLE_STABLE process. Scenes are separate live realizations, so timing distributions—not individual frames—are compared. Every cell separately proves exact output parity during warm-up before accepting measured traffic.

| action | deployed service before | after | saving | postprocess before | after | p025 before | after | serialization before | after | uplink before | after |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 160.38 ms | 116.12 ms | 44.26 ms | 69.66 ms | 42.67 ms | 13.58 ms | 11.27 ms | 17.64 ms | 2.66 ms | 140.27 ms | 670.15 ms |
| 15 | 161.12 ms | 122.71 ms | 38.41 ms | 68.99 ms | 46.33 ms | 12.18 ms | 10.27 ms | 20.19 ms | 3.08 ms | 76.12 ms | 74.91 ms |
| 50 | 144.50 ms | 99.48 ms | 45.03 ms | 64.84 ms | 36.00 ms | 12.26 ms | 11.17 ms | 18.54 ms | 2.54 ms | 57.94 ms | 57.21 ms |
| 71 | 136.34 ms | 95.27 ms | 41.06 ms | 60.11 ms | 43.86 ms | 12.50 ms | 8.83 ms | 23.93 ms | 3.28 ms | 35.59 ms | 36.29 ms |

The deployed edge-service saving is consistent: 38.41–45.03 ms across all
four actions, with a mean of 42.19 ms. The faster worker also completed more
frames and replaced fewer pending frames:

| action | tail completions before → after | queue replacements before → after | within 500 ms before → after |
|---:|---:|---:|---:|
| 30 | 163 → 226 | 136 → 72 | 86 → 96 |
| 15 | 168 → 230 | 132 → 70 | 131 → 206 |
| 50 | 200 → 274 | 100 → 26 | 197 → 273 |
| 71 | 183 → 284 | 117 → 16 | 174 → 282 |

The feature-uplink implementation, feature bytes and OAI path are unchanged.
However, the latency rows contain only frames which survived the edge's
latest-frame queue and reached the tail. Optimization changes that selected
subset. This explains why action 30's reported uplink median changes sharply
and means the paired uplink medians are not a transport-equivalence test.
Actions 15, 50 and 71 happen to agree closely, but that is supporting context,
not the binding proof. End-to-end AoI and queue effects are nonlinear and are
not inferred by subtracting the service saving from the 288-cell measurements.
