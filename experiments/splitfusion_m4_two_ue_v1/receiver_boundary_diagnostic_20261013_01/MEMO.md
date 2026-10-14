# M4 receiver boundary diagnosis

**Verdict A: the multi-second delay largely precedes kernel arrival.** The single authorized smoke passed both functional gates; it does not qualify two-UE timeliness. For fully reconstructed frames, including expired frames, median last-send-to-completing-kernel-arrival was **5.36 s / 7.85 s** for UE0/UE1. The median per-frame share of send-to-full-verification time occurring before that arrival was **99.970% / 99.978%**. This identifies the boundary, not radio scheduling alone.

## Counts and timing

| Observation | UE0 | UE1 |
|---|---:|---:|
| Sent frames | 208 | 208 |
| Policy decisions / holds / fallbacks | 102 / 101 / 5 | 102 / 102 / 4 |
| Fully reassembled / expired after reassembly | 177 / 171 | 156 / 150 |
| Completed predictions / accepted installations | 5 / 5 | 6 / 5 |
| Terminal timeouts | 203 | 203 |
| Scored feedback / timely policy feedback | 1 / 0 of 102 | 2 / 0 of 102 |
| Incomplete with observed chunks / no observed chunk | 3 / 28 | 1 / 51 |
| Last send to completing kernel RX median / P99 | 5361 / 8142 ms | 7850 / 12437 ms |
| Matched kernel RX to app receive median / P99 | 0.137 / 1.605 ms | 0.134 / 1.492 ms |
| App receive to reassembly completion median | 0.064 ms | 0.067 ms |
| Reassembly to full verification observation median | 1.337 ms | 1.398 ms |

The 406 timeouts reconcile to 321 after-reassembly expiries, one before-tail rejection, one completed prediction rejected by the map, and 83 frames without completed reassembly. Unobserved arrivals are censored at receiver close, **not proven packet loss**. Received datagrams were 2803/3232 and 2781/3587 sent; no observed duplicates or reordering. Exact payload/chunk ledgers reconcile to both sender byte totals.

Across **all 5,584 received datagrams**, kernel-to-app P99 was 2.21/2.31 ms; maxima were 82/80 ms. Receiver servicing has occasional outliers but does not account for the multi-second delay. Frontend/ranker/AE/codec medians were 11.0/13.7 ms; RGB-receipt-to-last-send medians were 90/105 ms. GPU-contention time is not separately measured.

Edge compute medians were 56.7/62.9 ms and legacy receive-entry-to-compute queue medians 0.41/0.51 ms, **conditional on only 5/6 predictions**. Capture-to-prediction medians were 375/405 ms; accepted-map age medians were 382/411 ms. No policy prediction or scored feedback met the 170-ms action-open deadline. The three scored replies arrived at 273 ms and 340/508 ms. GT target admission—not scoring completion—had medians 121/124 ms over 61/62 decisions. GT work runs in parallel; these durations must not be added.

## Backlog and channel evidence

Median source-bound RLC backlog was **4.29/6.37 MB** and grew with pre-kernel delay. Offered serialized feature load was 7.50/8.35 Mbit/s; capture-window scheduled PUSCH TB volume was 6.36/6.32 Mbit/s, while completed-feature goodput was 5.98/5.95 Mbit/s. TB volume includes transport overhead/retransmissions and is not application goodput. This supports an upstream offered-load/service imbalance; it does not isolate sender-stack, radio, core, or virtual-network residence.

Commanded SNR remained 23.5 dB with RFsim −12. Capture-only measured PUSCH median was **20.5 dB**, median MCS 25, over 1765/1754 transmissions. Attribution uses the attached RNTIs 8094/60647; RLC additionally requires UE ID. Radio event clocks use the retained time-of-day reconstructor. No SNR recalibration was performed.

**One next action:** in a separately authorized measurement task, add one source/frame/chunk-bound software timestamp at UPF N6 egress and join it to these edge kernel timestamps. That partitions the still-unresolved pre-arrival path without changing offered load, radio scheduling, or inference.

## Measurement validity

SO_TIMESTAMPNS_NEW supplies a CLOCK_REALTIME software receive stamp delivered with each edge UDP datagram. On this Docker bridge/veth/UPF path it is not an over-the-air timestamp; the control message does not identify which virtual-device handoff originally stamped the packet. Source identity comes from the registered receiving endpoint, then the reconstructed envelope—not the NAT peer address. See the [Linux timestamping documentation](https://docs.kernel.org/networking/timestamping.html).

There were 5,586 paired WALL/RAW samples and 416 independent sender pairs. Piecewise-linear conversion had 4.16 µs maximum held-out residual, 0.824 µs maximum sender cross-check residual, and a **9.63 µs empirical uncertainty envelope** including worst pairing bracket. Offset drift was 1.370 ms across the recording, approximately 30.28 ppm; no extrapolation or clipping was used. This is an empirical check, not a formal bound on unobserved clock steps.

Frame intervals use the exact datagram that completed reassembly. Per-chunk sender-to-kernel intervals are bounded by the recorded first/last userspace send marks; individual chunk-send times remain unknown. Empty CSV cells mean unknown/inapplicable. RAW, WALL and ordinary MONOTONIC fields remain distinct. Timeout receipts are never prediction latency. “Full verification” is the existing edge_wire observation after context/camera verification; processor-only verification is also retained separately.

All 12,186 audit rows were written; timestamp metadata was present for every received datagram. No logging drops, ancillary truncations, writer errors, or payload truncations occurred. Queue peak was 13 of 32,768. Receiver instrumentation accumulated 259 ms wall time, about 46.4 µs/datagram; producer logging accumulated 117 ms with 1.885 ms maximum call. These counters overlap and include descheduling; they must not be summed. Writer CPU and causal observer effects are unmeasured.

## Qualification and cleanup

Only edge_entry.py, datapath.py and frontend.py changed; receiver_timing.py was added. The original transport, GT/model stack, deadlines, scheduling and radio settings remain hash-identical. Focused real-socket checks passed on host and deployed slim image, covering source collisions, duplicates/reordering, incomplete/expired frames, missing/truncated ancillary data, and byte/routing parity. Production binding/startup checks passed. The 720-source deployment preparation is **07ec210311434678c3172af18a0f2ca7966673ca0a9fb83b71e850ff46e7a04b**; exact changes/hashes are in [DEPLOY_PLAN.json](DEPLOY_PLAN.json).

Existing assessment ran once: actor replay **204/204 exact**, rescoring **3/3 exact**. Only 1/2 frames were scored per UE, with zero undefined overall qualities; missing GT is separate from scored zero-eligible classes. Vehicle localisation within 1 m was 4/5 and 6/11 eligible objects; vehicle segmentation IoU 0.736/0.770. These tiny survivor populations do not establish quality performance. All three scored frames had zero eligible pedestrians.

The unchanged full-route analyzer's STOP remains in [existing_assessment_01](existing_assessment_01/ASSESSMENT.json). Its missing per_frame_metrics input yields offered/sent=0; strict None-versus-False and nonempty zero-fault-dictionary checks also fail. Corrected smoke accounting is separate; no historical verdict or production checker was edited.

Exactly one attempt: deployment smoke_live_20261013_08, preparation_smoke_20261013_11. Configured budget was 20 simulated seconds; existing bounded-loop termination recorded **21.0 simulated seconds / 42.35 wall seconds**, and parent launch-through-cleanup took **104.45 seconds**. Registered preparation through launch took 320 seconds; existing assessment 2.61 seconds and final diagnostic computation/rendering about 1.8 seconds.

Ownership cleanup passed: no CARLA/OAI/edge processes or UE tunnels remained. Ten task-created core containers were stopped and retained. Approved idle MPS/remote desktop, Firefox PID 929095, and both UI servers were preserved. Firefox remained by explicit user instruction; its timing effect is unmeasured. Capture occupies 564.5 MB; free-space change across the launch was 575.0 MB. Final observed free disk is **26.09 GB / 24.3 GiB**. No deletion, retry, optimization, commit or push.

## Artifacts and reproduction

- [Frame timing CSV](analysis_01/FRAME_TIMING.csv), [chunk timing CSV](analysis_01/CHUNK_TIMING.csv), [compact results](analysis_01/RESULTS.csv), [summary](analysis_01/SUMMARY.json).
- [Delay breakdown PNG](analysis_01/01_DELAY_BREAKDOWN.png): independently measured median/P99 intervals and conditional sample sizes; quantiles are not additive.
- [Backlog and delay PNG](analysis_01/02_BACKLOG_AND_DELAY.png): aligned evidence, censored-frame rug, and scheduled TB versus feature goodput. Exactly two 3840×2160 PNGs; no PDF/SVG.
- [Actual launch command and result](smoke_launch.json), [cleanup](POSTFLIGHT.json), [preserved sessions](FINAL_CLEANUP_DETAIL.json), [clock validation](analysis_01/CLOCK_VALIDATION.json), [final hashes](FINAL_SHA256.json).

From the working repository, reproduce saved-evidence analysis into a **new** directory:

~~~bash
env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python3 -B experiments/splitfusion_m4_two_ue_v1/receiver_boundary_diagnostic_20261013_01/analyze.py \
  /tmp/m4_receiver_diagnosis_reproduction_01
~~~

