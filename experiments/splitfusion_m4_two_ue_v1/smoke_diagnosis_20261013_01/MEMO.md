# M4 two UE smoke diagnosis

Two-UE integration works; timeliness bottleneck remains under diagnosis.

The first passing smoke is a **functional integration success, not performance qualification**. The dominant observed delay lies between the UE's last packet-send call and the edge's verified-frame observation. Multi-megabyte RLC backlog makes uplink queueing the leading explanation. The evidence does not separate its contribution from host buffering, receiver scheduling and validation. No runtime change is justified by this diagnosis alone.

## Reconciled outcomes

Capture: `smoke_live_20261013_07`, 13 October 2026. Each source's first-to-last capture spans about 42.6 seconds of wall time; the smoke budget is simulated time. Source/frame/session identities agree across controller, transport, edge and map evidence.

| Measurement | UE0 | UE1 |
|---|---:|---:|
| Assigned and sent frames | 208 | 208 |
| Policy decisions / holds / fallbacks | 103 / 103 / 2 | 100 / 100 / 8 |
| Fully reassembled, including expired frames | 207 | 208 |
| Expired at EDGE_AFTER_REASSEMBLY | 172 | 179 |
| Admitted reassembly events | 35 | 29 |
| Completed predictions, all frame kinds | 33 | 28 |
| Map observations / accepted installations | 33 / 29 | 28 / 26 |
| Local terminal FEEDBACK_TIMEOUT | 178 | 180 |
| Other terminal | 1 superseded | 2 stale at map |
| Completed policy predictions / scored replies | 14 / 14 | 11 / 11 |
| Timely policy predictions / feedback / scored feedback | 0 / 0 / 0 | 0 / 0 / 0 |

The disjoint terminal totals are **29 + 178 + 1 = 208** and **26 + 180 + 2 = 208**. UE0's timeouts comprise 172 later edge-expiry observations, four later map rejections, one before-tail expiry and frame 52 with no edge event. UE1's timeouts comprise 179 later edge-expiry observations and one before-tail expiry. Late diagnostic rows are not additional terminals. The two before-tail aborts are not completed predictions. All 203 policy reward resolutions are TIMEOUT.

The 351 after-reassembly expiries are **not packet loss**. UE0 frame 52 remains an unexplained missing edge observation; its packet-level fate cannot be established. The historical analyzer's sent/offered zero came from absent `per_frame_metrics.csv`. There are 208 recorded assignments and transmissions per source; the broader collector-opportunity count remains unknown, not zero. The historical full-route STOP and smoke PASS are unchanged.

## Where the delay accumulates

Milliseconds below are independently measured interval medians, **not additive stage medians**. UE0 / UE1 sample sizes follow each row.

| Interval and clock | UE0 P50 | UE1 P50 | n |
|---|---:|---:|---:|
| RGB receipt → last send, RAW | 99.6 | 110.4 | 208 / 208 |
| Front + ranker + AE + codec, RAW; included above | 13.7 | 13.8 | 208 / 208 |
| Last send → verified frame, RAW | 2442.0 | 4697.9 | 207 / 208 |
| Edge queue, WALL; completed predictions only | 0.48 | 0.75 | 33 / 28 |
| Edge compute, WALL; completed predictions only | 60.4 | 59.5 | 33 / 28 |
| Action-open → completed policy prediction, RAW | 305.4 | 316.1 | 14 / 11 |
| Action-open → scored feedback receipt, RAW | 365.9 | 359.7 | 14 / 11 |
| Capture → actual map installation, WALL | 361.3 | 387.8 | 29 / 26 |

On the expired-after-reassembly subset alone, post-send medians are 2619 / 4808 ms, while RGB-to-send medians are 95.7 / 107.9 ms. Post-send delay exceeds pre-send delay on 406 of 415 observed chains. Thus the multi-second gap is not directly accounted for by frontend work, the measured survivor edge queue/compute, or downstream GT scoring. Separate encoding and shared-GPU wait measurements are unavailable; indirect host/GPU contention is not excluded. Edge queue P95 is 30.1 / 71.8 ms despite its small median.

GT was scorer-ready before the prediction on all 25 scored decisions. Its additional prediction-to-reply interval was 55.0 / 46.9 ms median, after predictions were already late. GT capacity refusals nevertheless occurred on 36 / 37 decisions and must not be hidden: only 67 / 63 had a ready target, of which 67 / 55 were ready within 170 ms of action-open. A capacity refusal is missing GT, not a scored zero-quality or zero-eligible frame. The existing exact actor-replay and 25 exact rescoring results are retained, not rerun.

## Channel and payload evidence

`ATTACHED.json` binds UE0 to RNTI 12632 and UE1 to RNTI 15481. Filtering reconstructed radio **event** timestamps to each source's first-to-last capture gives 2209 / 2215 PUSCH samples: **20.5 dB median**, 20–21 dB range, MCS 25 median. Including the observed edge-drain window leaves the median unchanged. Therefore the 3 dB command/measurement difference is not explained by startup samples or mixing the two RNTIs. It is not proof of a calibration defect. All recorded commands target 23.5 dB with RFsim noise −12 dB, and all 103 / 100 recorded actor inputs agree with a prior valid 23.5 dB acknowledgement. Command and measurement are different quantities.

RLC occupancy, summed over distinct LCIDs per source/tick, has median **1.98 / 4.11 MB** and maximum **3.82 / 6.71 MB**, with nonzero queues on about 92% of sampled ticks. Earlier capture-time backlog and subsequent send-to-verification delay correlate (descriptive Pearson r = 0.90 / 0.86; temporally dependent samples, not a causal test). PUSCH grants have median 273 PRBs and scheduled-TB totals of 7.66 / 7.56 Mbit/s over the capture window. These are allocated transport-block bytes, **not application goodput or an available-bandwidth measurement**. Observed UL DCI retransmission rounds are 14 / 12 versus 2197 / 2207 round-zero grants. Radio backlog is better supported than a GT explanation, but its root cause and receiver contribution remain unresolved.

Sender counters show 37.88 / 40.52 MB in 3143 / 3357 UDP application datagrams. Subtracting their retained eight-byte chunk headers gives mean SFD4 payloads of **182.0 / 194.7 kB per sent frame**. All-sent mode counts are mode 11: 174 / 160, mode 8: 32 / 48, mode 10: 2 / 0; policy q_e4 medians are 3384 / 3072.5. Exact payload lengths survive for only 41 / 35 frames: admitted-subset P50/P99 = 151.7/319.4 and 125.5/328.4 kB (n = 35/29); seven planned retained samples per source have P50 = 179.3/189.1 kB. These subsets are not all-sent payload distributions. Full-population payload quantiles are unknown; compressed sizes cannot be recovered from actions alone.

## One next intervention

Add **source/frame/chunk-bound receiver boundary timing** in a separately authorized instrumentation change: retain kernel receive timestamps for first/last chunks, corresponding receive-return timestamps and a pre-admission reassembly-complete timestamp, including rejected frames. Pair WALL and MONOTONIC_RAW explicitly. Join these to the existing sender marks and source-bound RLC trace. This is the smallest measurement that separates pre-receiver transport delay from socket backlog/receiver work; it avoids choosing a bandwidth, GT or GPU fix prematurely. No instrumentation or new experiment was implemented here.

## Presentation and reproduction

Exactly two figures are supplied, each as a 3840 × 2160 PNG and matching PDF/SVG:

- `01_frame_outcomes`: disjoint sent-frame terminals, with policy-feedback timeliness separately annotated.
- `02_delay_localization`: stale-inclusive edge-observation ages, all-decision completion fractions, and separate conditional intervals. The 500-ms processing horizon starts at capture WALL; the original 170-ms policy-feedback deadline starts at action-open RAW.

Three supervisor talking points:

1. Both sources work end to end and shut down cleanly; only 55 of 416 sent frames were installed, with no timely policy feedback.
2. Most failures are fully reassembled but stale frames, not demonstrated packet loss; survivor compute medians hide the multi-second pre-processing delay.
3. Large source-specific RLC queues make uplink queueing the leading explanation. Measure the receiver boundary before changing production behavior.

From the working repository, in a host-visible shell:

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B experiments/splitfusion_m4_two_ue_v1/smoke_diagnosis_20261013_01/analyze.py
```

The worker is CPU-only, one core, nice 19, with single-thread numerical libraries. The supervisor pauses it while the live-host lock or CARLA/OAI process is present. The single-UE demo's saved report records completed cleanup; the final host-visible check also found the lock free and no CARLA/OAI processes. The final analysis and rendering took about two seconds; coordination events are retained. No live run, production edit, cleanup action or historical-evidence modification was performed.

`FRAME_TIMING.csv` contains all 416 source/frame joins, `CORRECTED_ANALYSIS.json` the separate corrected assessment, and `TIMING_SCHEMA.md` the clock/availability rules. `INPUT_SHA256.json`, `SOURCE_BINDING.json` and `REPRODUCIBILITY.json` bind the evidence and interpretation sources. Run `python3 -B experiments/splitfusion_m4_two_ue_v1/smoke_diagnosis_20261013_01/verify.py` for the bounded artifact checks and input-hash verification.
