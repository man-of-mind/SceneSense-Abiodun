# M4 two UE uplink service diagnosis

The offered traffic exceeds delivered uplink service, and nearly all the accumulating byte deficit is visible in the UE RLC queues. Service is limited by a very slow radio clock, approximately equal sharing of uplink opportunities, and lower spectral efficiency than the single UE calibration predicts. The precise thread or wait responsible for the clock slowdown remains unresolved. **Offline evidence supports a controlled admission-rate intervention; no additional smoke was run.** Functional integration success and the historical timeliness verdicts remain unchanged.

## Demonstrated load and queue imbalance

The common observation window is **2026-10-14 01:15:31.915659–01:16:13.608225 UTC**, or October 13 at 18:15 local time: 41.692566 wall seconds, first userspace send through last userspace send across both UEs. Decimal MB and Mbit/s are used. This differs slightly from the earlier capture-timestamp window.

| Observation | UE0 | UE1 |
|---|---:|---:|
| Sent frames | 208 | 208 |
| Policy decisions / holds / fallbacks | 102 / 101 / 5 | 102 / 102 / 4 |
| UDP bytes offered / observed received, MB | 39.097 / 31.698 | 43.508 / 31.521 |
| Offered feature payload, Mbit/s | 7.497 | 8.343 |
| Scheduled DCI TB volume, Mbit/s | 6.385 | 6.332 |
| New-transmission TB volume excluding retransmission grants, Mbit/s | 6.343 | 6.293 |
| Received feature chunks, Mbit/s | 6.078 | 6.044 |
| Completed-feature goodput, Mbit/s | 5.979 | 5.996 |
| Sent minus received UDP, MB | 7.399 | 11.986 |
| RLC queue start → end, MB | 0 → 7.425 | 0 → 11.963 |
| RLC median, MB | 4.309 | 6.379 |
| Completing kernel arrival minus last send, median / P99, seconds | 5.361 / 8.142 | 7.850 / 12.437 |
| Frames with that interval observed, including stale frames | 177 / 208 | 156 / 208 |

The endpoint byte deficits agree with RLC growth within **25.8/23.8 kB**. These are different layers with headers, fragmentation, in-flight data and asynchronous sampling, not an exact packet-level conservation equation. RLC endpoint observations precede the final send boundary by 30.9/2.0 ms. PDCP volume is retained separately in [SUMMARY.json](SUMMARY.json); it is not added to UDP or TB volume. Scheduled TB includes padding and headers and is not application goodput.

Holds contribute **49.8%/50.0% of feature bytes**; fallbacks contribute only 0.086%/0.061%. The growth is not a fallback storm. The queues drain after sending stops but still contain **5.09/9.84 MB** at receiver close, 2.712 s later. The 31/52 frames without completed reassembly are censored, not proven packet loss. The timing distributions include 171/150 after-reassembly expiries, not just the 5/6 prediction survivors. Previously validated kernel-to-application medians are only 0.137/0.134 ms. Core residence is not independently timestamped, but a multi-megabyte hidden core queue is unnecessary to explain the observed deficit.

## Why wall clock service is low

**Measured:** only **3.56 NR seconds advance in 41.7 wall seconds**, approximately 0.0853 NR-s/wall-s. This is the radio frame/slot clock, not CARLA simulation time. Each UE gets approximately **42 grants per wall second**, with **50.2%/49.8%** of allocated PRB-symbols. Median grant width is 273 PRBs, dominant MCS is 25/table 0, and the retained MAC snapshot reports one UL layer. A full-width 13-symbol grant carries 22,547 TB bytes. Retransmission grants are just 0.79%/0.68%; no forced MCS or power-headroom MCS reduction appears. Neither starvation nor a simple per-UE bitrate cap explains these traces.

**Configuration and source supported:** the 5-ms 4D5U period contains five full UL slots plus a mixed slot. The deployed log lists full-slot TDAs with k2=6 and a three-symbol mixed TDA with k2=10. Full UL slots **5 and 15 never receive a grant in 354 fully covered NR frames**: the required k2=6 scheduling origin would itself be an UL slot. Eight other full UL slots plus two mixed slots are reachable. Their saturated MCS-25 arithmetic is 190,110 TB bytes per 10-ms NR frame; multiplied by the observed radio clock rate, this is approximately **13.0 Mbit/s aggregate**, close to the observed 12.72. Thus configured 100 MHz is not continuous wall-clock capacity. Filling the two missing opportunities would be only a nominal 23.7% increase at an unchanged radio clock, not a demonstrated wall-clock improvement.

The compatible saved V11 GOOD_ALL reference uses the same 100-MHz/273-PRB/4D5U profile. Over 444.82 s and 3,588 sent frames, it offers 7.37 Mbit/s, has median payload 111.8 kB, median RLC queue zero, PUSCH 23.5 dB/MCS 27, and a radio clock rate of **0.663 NR-s/wall-s**. Its first 41.69 s similarly give 0.659, with 347 frames and 8.33 Mbit/s offered. M4 has fewer frames per second but larger mean payloads, 187.8/209.0 kB, and two senders. V11 scheduled TB volume is 16.21 Mbit/s, not measured application goodput or a matched capacity test. Different sessions, scenes, load and composition prevent causal attribution of the roughly eightfold clock-rate difference.

RFsim source waits for samples from every connected peer and processes each modeled peer in the receive path. That supports CPU work or peer synchronization as explanations for the slowdown. **Unresolved:** no retained per-thread CPU/off-CPU or peer-wait measurements distinguish them. The shared CPU mask alone does not establish contention; exact sender/radio/core packet residence also remains unpartitioned.

## Channel target and measured SNR

Both source-bound channels were acknowledged at noise −12, target 23.5 dB, before sending. All 826 commands per UE retain that same pair; repeated ACKs include 3/18 late acknowledgements but no change in target. Capture-only, RNTI-bound PUSCH samples have median **20.5 dB** for both UEs (n=1,770/1,757). This is not startup contamination or swapped source identity.

**Strong source-supported mechanism:** the gNB sums both modeled UE inputs into one receive buffer. Each `rxAddInput` call adds independent Gaussian noise, including zero-signal samples. With two equal noise terms, variance doubles: **23.5 − 10 log10(2) = 20.49 dB**, matching the measurement. The single-UE calibration maps −12 to 23.5 dB; duplicating its per-link noise does not preserve that received SNR. No global RFsim noise override is configured. This is quantitatively consistent, not a controlled noise intervention. Current source/library copies match across worktrees; executables match the sealed preparation, but the historical RFsim shared-library hash was not retained. Exact identities are in [SOURCE_BINDING.json](SOURCE_BINDING.json).

Keep four statistics separate: commanded target **23.5 dB**; quantized per-transmission PUSCH median **20.5 dB**; scheduler EMA median **20.6 dB**, initially higher and converging; and the final MAC power-control statistic **16.8/16.3 dB**, which adds in-flight TPC to the average and is not capture-aligned. The MAC control target is 15 dB, not the commanded channel target. The SINR policy selects MCS 25 above 20.2 dB and MCS 27 above 22.0 dB, explaining the observed spectral-efficiency difference without an unintended MCS cap.

Source anchors, relative to `OAI/openairinterface5g`: `radio/rfsimulator/simulator.cpp:1278,1420`; `radio/rfsimulator/apply_channelmod.c:216`; `openair2/LAYER2/NR_MAC_gNB/nr_radio_config.c:1087`; `gNB_scheduler_ulsch.c:961,2244,2584`; `gNB_scheduler_primitives.c:230,4263`. The latter two are in the same gNB directory. Input hashes include deployed configuration, commands, logs, calibration and all joined traces.

## One next action

**A separately controlled per-UE wall-clock byte-admission cap, initially 5 Mbit/s of serialized feature payload, with only the latest unsent frame retained and no extra FIFO.** This is approximately 17% below the measured 6.04–6.08 Mbit/s received-chunk service. Its mechanism is to stop injecting bytes faster than they drain, rather than buffer stale work. It is a bounded overload mitigation study, not a fix or diagnosis of the radio clock slowdown.

Risks are reduced temporal coverage, altered feedback/state histories and fresh-frame drops during bursts; whole-frame admission must preserve lifecycle accounting and report every unadmitted frame. The current trace cannot predict future actor outputs or promise the 170-ms deadline. Validate with unchanged timing anchors and explicit offered/admitted/dropped denominators: RLC backlog should cease secular growth, last-send-to-kernel delay should stop rising, and prediction/feedback timeliness and map age must be reported independently. Do not combine this intervention with noise, TDA or affinity changes. **Nothing was implemented or launched here.**

## Artifacts and reproduction

- [Load backlog and delay PNG](01_LOAD_BACKLOG_DELAY.png): aligned rates and byte deficits, with stale arrivals and censored-frame rug.
- [Radio resources PNG](02_RADIO_RESOURCES.png): actual slot allocations, clock dilation, scheduling share and the source-supported noise calculation.
- [Per-UE table](PER_UE.csv), [one-second joined data](ALIGNED_1S.csv), [frame timing](FRAME_DELAY.csv), [source-bound grants](UL_GRANTS.csv), [slot allocations](ALLOCATION_BY_SLOT.csv), [summary](SUMMARY.json), [input hashes](INPUT_SHA256.json).

Exactly two 3840×2160 PNGs. RAW send clocks are mapped to WALL using the retained paired-clock interpolation; its empirical uncertainty envelope is 9.63 µs. Radio event times use the existing local-time reconstructor, not ingest timestamps. In one-second plots, each frame's bytes are assigned to its last-send mark because individual chunk-send times are unavailable. Clock rates use unwrapped frame/slot differences; RLC sums distinct logical channels per source/tick. No timeout timestamp is treated as prediction latency.

From the working repository, reproduce into a new output directory:

~~~bash
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B \
  experiments/splitfusion_m4_two_ue_v1/uplink_service_audit_20261013_01/analyze.py \
  /tmp/m4_uplink_service_reproduction_01
~~~

Final computation/rendering took 6.42 s on CPU 23, nice 19, single-threaded numeric libraries. Only this new diagnostic directory was written; historical evidence and production code were preserved. No live resources were started or stopped; Firefox, MPS and UI services were untouched. Observed remaining disk: **26.08 GB / 24.29 GiB**. No deletion, tuning, registration, commit or push.
