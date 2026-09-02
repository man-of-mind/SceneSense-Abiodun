# Hybrid-q Phase 7 — lossless zstd over the sparse transport wire

Generated 2026-09-02T21:28:08.305106+00:00 · terminal `HYBRID_Q_PHASE7_ZSTD_MEASUREMENT_COMPLETE`

## Scope

Transport size and **host** encode/decode cost only. No training, no perception evaluation, no validation or test data, no change to any model output. zstd is lossless, so retained values survive bit-exactly and the Phase-6 accuracy curve is unchanged by construction.

**All latencies below are current-host numbers** (Intel(R) Core(TM) Ultra 9 285K, NVIDIA GeForce RTX 5090), with C2 resident on the GPU. They are **not** Raspberry Pi UE latencies and **not** OAI transport latencies. Byte sizes are host-independent; latencies are not.

## Compressor

- implementation: `python-zstandard` 0.25.0
- zstd library: 1.5.7 (backend `cext`)
- level 1, threads 0 (inline, no worker pool), dictionary none, `write_checksum=True`, `write_content_size=True`, `write_dict_id=False`
- one independent zstd frame per camera frame; no concatenation, no batch API; one reused compressor and one reused decompressor context; no level search

## Data

- 128 training-**fit** frames: 16 evenly spaced frames from each of the 8 registered fit episodes, no augmentation
- validation/test/holdout frames used: 0
- source manifest sha256: `5d65e6eb14aadea11ca6bab6e82f0c94c31a50746611d167d282d8988a4504c2`
- ordered selected-ID digest: `cffe5d873298c8b785c72f06fdf31aef75f07dfdfca6f248c04cca81adaf45ae`
- ordered (episode, frame, ID) digest: `2931ae6b820d1ac72a63d8f84ed19e894b9c425b6dfec33ac7f853cd53fe32f4`

## Round-trip integrity

- **1408/1408** payloads decompressed byte-for-byte identical to `SparsePayload.data`
- requested q == wire q at 1e-4 on all 1408 payloads; header `q_e4` == round(q x 10000)
- exact keep count and framed length on every payload
- retained values bit-identical; dropped locations exactly zero
- masks nested on all 128 frames
- q=0 took its exact ranker-bypass path on every frame

## Size

| q | cells kept | framed sparse B | zstd B mean | median | p5 | p95 | zstd/sparse | vs framed q=0 | reduction | vs zstd q=0 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.00** | 21504 | 22,020,140 | 15,989,551 | 15,993,204 | 15,821,680 | 16,160,783 | 0.7261 | 0.7261 | 27.39% | 1.0000 |
| 0.05 | 20429 | 20,922,028 | 15,158,171 | 15,164,810 | 14,994,786 | 15,324,014 | 0.7245 | 0.6884 | 31.16% | 0.9480 |
| 0.10 | 19354 | 19,821,228 | 14,318,962 | 14,328,548 | 14,168,670 | 14,468,124 | 0.7224 | 0.6503 | 34.97% | 0.8955 |
| 0.15 | 18278 | 18,719,404 | 13,486,556 | 13,495,203 | 13,350,608 | 13,626,817 | 0.7205 | 0.6125 | 38.75% | 0.8435 |
| 0.20 | 17203 | 17,618,604 | 12,658,404 | 12,663,262 | 12,535,100 | 12,793,609 | 0.7185 | 0.5749 | 42.51% | 0.7917 |
| 0.25 | 16128 | 16,517,804 | 11,835,152 | 11,838,578 | 11,719,424 | 11,967,430 | 0.7165 | 0.5375 | 46.25% | 0.7402 |
| **0.30** | 15053 | 15,417,004 | 11,015,628 | 11,021,751 | 10,905,156 | 11,142,733 | 0.7145 | 0.5003 | 49.97% | 0.6889 |
| **0.50** | 10752 | 11,012,780 | 7,773,802 | 7,763,193 | 7,676,589 | 7,879,066 | 0.7059 | 0.3530 | 64.70% | 0.4862 |
| **0.70** | 6451 | 6,608,556 | 4,594,683 | 4,592,850 | 4,544,573 | 4,645,071 | 0.6953 | 0.2087 | 79.13% | 0.2874 |
| **0.90** | 2150 | 2,204,332 | 1,492,588 | 1,493,420 | 1,476,727 | 1,508,849 | 0.6771 | 0.0678 | 93.22% | 0.0933 |
| **0.98** | 430 | 443,052 | 290,880 | 290,842 | 287,375 | 294,393 | 0.6565 | 0.0132 | 98.68% | 0.0182 |

`zstd_ratio` = compressed / sparse framed. `vs framed q=0` = compressed / 22,020,140 (the framed dense payload). `vs zstd q=0` = total compressed at q / total compressed at q=0 over the same 128 frames. The unframed raw FP32 reference is 22,020,096 bytes, 44 bytes below the framed q=0 payload; it is a reference, not a wire format.

## Latency (current host, milliseconds)

| q | zstd comp med | p95 | zstd decomp med | p95 | sparse enc med | sparse dec med | rank+order med | UE total med | p95 | edge total med | p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 35.998 | 37.251 | 17.583 | 21.715 | 3.358 | 36.418 | 0.000 | 39.404 | 41.025 | 53.857 | 69.535 |
| 0.05 | 34.291 | 35.192 | 16.730 | 17.282 | 3.582 | 38.903 | 0.273 | 39.069 | 40.575 | 55.529 | 68.986 |
| 0.10 | 32.498 | 33.653 | 15.888 | 16.289 | 3.455 | 38.256 | 0.273 | 37.176 | 38.809 | 54.120 | 56.473 |
| 0.15 | 30.711 | 31.856 | 15.068 | 15.430 | 3.274 | 37.769 | 0.273 | 35.163 | 36.967 | 52.782 | 66.297 |
| 0.20 | 29.005 | 29.894 | 14.200 | 14.517 | 3.121 | 37.101 | 0.273 | 33.271 | 34.859 | 51.297 | 63.410 |
| 0.25 | 27.188 | 28.266 | 13.394 | 13.793 | 2.954 | 36.543 | 0.273 | 31.284 | 33.070 | 49.929 | 58.966 |
| 0.30 | 25.367 | 26.370 | 12.501 | 12.772 | 2.797 | 35.818 | 0.273 | 29.293 | 30.855 | 48.310 | 55.911 |
| 0.50 | 18.295 | 18.895 | 9.008 | 9.222 | 2.135 | 33.014 | 0.273 | 21.548 | 22.792 | 42.055 | 45.075 |
| 0.70 | 11.155 | 11.522 | 5.484 | 5.617 | 1.460 | 30.337 | 0.273 | 13.671 | 14.571 | 35.833 | 37.702 |
| 0.90 | 3.736 | 3.848 | 1.849 | 1.920 | 0.905 | 27.527 | 0.273 | 5.645 | 6.224 | 29.390 | 31.686 |
| 0.98 | 0.829 | 0.874 | 0.384 | 0.401 | 0.519 | 26.122 | 0.273 | 2.279 | 2.922 | 26.509 | 27.958 |

UE total = ranker + full ordering + selection + masking + sparse encode + zstd compress. The complete ranker and ordering cost is charged in full to every q>0 row and is never divided among q values. q=0 excludes ranker, sorting and selection because it keeps its exact bypass path. Sparse-encode latency includes the GPU-to-host transfer. Edge total = zstd decompress + sparse decode.

### Fixed-q corroboration

The primary table walks all eleven q values inside each frame, so the host allocator never holds a warm free list for the size it needs next. A deployed UE instead serves one q at a time and reuses same-sized buffers frame after frame. Repeating a single q on one frame (9 repeats, first discarded) isolates that difference:

| q | zstd comp med | zstd decomp med | sparse decode med | primary sparse decode med |
|---:|---:|---:|---:|---:|
| 0.00 | 36.232 | 17.265 | 36.094 | 36.418 |
| 0.05 | 34.405 | 16.523 | 39.297 | 38.903 |
| 0.10 | 32.567 | 15.747 | 37.164 | 38.256 |
| 0.15 | 30.751 | 14.892 | 36.317 | 37.769 |
| 0.20 | 29.059 | 14.149 | 36.847 | 37.101 |
| 0.25 | 27.480 | 13.417 | 36.270 | 36.543 |
| 0.30 | 25.526 | 12.614 | 35.837 | 35.818 |
| 0.50 | 18.297 | 8.998 | 32.859 | 33.014 |
| 0.70 | 11.072 | 5.427 | 30.007 | 30.337 |
| 0.90 | 3.855 | 1.848 | 27.199 | 27.527 |
| 0.98 | 0.839 | 0.382 | 26.121 | 26.122 |

Every column agrees with the primary table: sparse decode within 1.45 ms and zstd compression within 0.29 ms at every q. With host threads pinned there is **no** cold/warm allocator gap to correct for — the primary latencies track payload size rather than the order q is walked in. This is one frame repeated, so it corroborates the primary numbers rather than replacing them.

## Throughput (current host)

| q | compression MB/s median | decompression MB/s median |
|---:|---:|---:|
| 0.00 | 611.7 | 1,252.3 |
| 0.05 | 610.1 | 1,250.6 |
| 0.10 | 609.9 | 1,247.6 |
| 0.15 | 609.5 | 1,242.3 |
| 0.20 | 607.4 | 1,240.8 |
| 0.25 | 607.5 | 1,233.2 |
| 0.30 | 607.8 | 1,233.3 |
| 0.50 | 602.0 | 1,222.5 |
| 0.70 | 592.4 | 1,205.1 |
| 0.90 | 590.1 | 1,192.1 |
| 0.98 | 534.3 | 1,154.1 |

Throughput is measured over `sparse_payload_bytes` (the compressor's input), so it is comparable across q.

## Interpretation limits

- Sizes and latencies here are transport facts. They carry no accuracy claim: perception accuracy is validated only at the measured q anchors, and executability at an unmeasured q is not a measured accuracy result.
- The 128 frames are a deterministic balanced **training-fit** sample chosen for workload characterization. They are not a held-out estimate of anything.
- Level 1 was frozen, not selected. No level search was run, so nothing here says level 1 is optimal.
- **Host CPU threads are pinned to 1** (torch's default here is 24). At the default, the same sparse-decode call was observed anywhere from 5 ms to 68 ms while its component operations still summed to 3-7 ms — one `torch.isfinite` over the 22 MB dense tensor took 28 ms in one sample and 1 ms in the next. Those multi-thread medians measured thread-pool scheduling on a hybrid P/E-core CPU, not codec cost, and are not reported. The zstd stage is unaffected either way: it is a single-threaded C call and its timings agree across every configuration tried.
- Sparse encode and decode are the **existing** codec's cost, unchanged by this phase. They are reported so the zstd stage can be seen in proportion, not as a zstd result.
