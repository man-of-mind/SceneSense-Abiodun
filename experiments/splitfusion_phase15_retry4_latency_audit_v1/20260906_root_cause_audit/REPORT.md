# Phase-15 retry4 root-cause latency audit

- Terminal classification: `ROOT_CAUSE_LOCALIZED`
- Audited evidence: `experiments/splitfusion_16_cell_live_carla_oai_pilot_v1/20260905_live_carla_oai_pilot_retry4`
- Evidence integrity: 133 hashes checked, 0 mismatches
- Audit scope: offline, read-only. No experiment, container, model or threshold was touched.

## 1. Action identity (verified before analysis)

| Action | Profile | Family | Quantizer | q_e4 | Catalog agrees |
|---:|---|---|---|---:|---|
| 0 | `split_noae_uint8_q0000` | noAE | UINT8 | 0 | True |
| 20 | `split_ae128_uint8_q5000` | AE128 | UINT8 | 5000 | True |
| 46 | `split_ae64_uint6_q9000` | AE64 | UINT6 | 9000 | True |
| 71 | `split_ae32_uint4_q9800` | AE32 | UINT4 | 9800 | True |

Bound from the 72-action catalog and cross-checked against every per-cell `resolved_config.yaml` record, not against the ordering of `campaign.actions.profile_ids`. Disagreements: none.

## 2. Directly measured facts

| Action | feat dg/frame | app bytes/frame | result dg/msg | edge completions (lower bound) | results at UE | installs | median AoI ms (median of cell medians) | timely feedback |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 288 | 3593210 | n/a | 0 | 0 | 0 | n/a | 0 |
| 20 | 97 | 1207653 | 102 | 3142 | 187 | 187 | 3889 | 0 |
| 46 | 9 | 100190 | 102 | 3693 | 277 | 277 | 33905 | 0 |
| 71 | 1 | 6420 | 103 | 4348 | 283 | 283 | 110237 | 0 |

- **Zero** frames in **any** of the 16 cells met the 500 ms feedback deadline (0 of 35870 sent).
- **Zero** installs met the 100 ms `service_deadline_ms` (0 of 747 installs).
- Feature reassembly is strictly all-or-nothing: across all decoded frames, `feature_received_datagrams` equals `datagrams` sent and duplicates are zero.
- Installation is lossless downstream of the UE result loop: results ingested at the UE equals maps installed equals ACK rows, in all 16 cells.
- `installation -> feedback emission` median is ~0.01 ms and `emission -> receipt` ~0.1 ms. Feedback delivery contributes nothing to the deadline miss.

## 3. Where the frames go (funnel)

Per-cell detail is in `funnel_by_cell.csv`. Aggregated by action:

| Action | sent | reached+processed by edge | results back at UE | installed | edge completion rate | result survival rate |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8741 | 0 | 0 | 0 | 0.000 | n/a |
| 20 | 8645 | 3142 | 187 | 187 | 0.363 | 0.060 |
| 46 | 9117 | 3693 | 277 | 277 | 0.405 | 0.075 |
| 71 | 9367 | 4348 | 283 | 283 | 0.464 | 0.065 |

Reading direction matters here:

- Edge-side counts are **lower bounds**. `edge_counters` rides inside each returned result, so any edge work performed after the last surviving result is invisible from the UE. `edge completion rate` is therefore a lower bound and `result survival rate` is an **upper** bound.
- Action 0's `0` edge completions is **absence of evidence, not measured zero**. Every edge-side counter travels inside a returned result, and action 0 returned none, so action 0 has no edge-side evidence at all. See the unresolved list.

## 4. Source-code facts

- `ue_route_b_split_cell_adapter_v1.py:784` — `prepared_queue` is a FIFO `queue.Queue(maxsize=4)`. `on_world_tick` uses `put_nowait` and classifies overflow as `DROPPED_QUEUE_FULL`. There is no latest-frame-first policy: the oldest queued frame is always served first.
- `ue_route_b_split_cell_adapter_v1.py:1006-1026` — exactly **one** worker thread drains that queue, and `_process_token` runs radar-window assembly, radar rasterisation, front inference, encode, send **and** the evaluation-only `_ground_truth` call serially on it.
- `ue_route_b_split_cell_adapter_v1.py:1084-1085` — `service_deadline_at` and `ack_timeout_at` are computed here and then only ever written into rows or compared to label a row `late` (`ue_map_install_feedback_v1.py:148` and `:198`). Neither identifier occurs anywhere in `live_pilot_runtime.py` or the map server, so no edge admission, decode, tail, publication or install step is gated on the deadline. A 500 ms timeout cancels and obsoletes nothing.
- `splitfusion_live_dispatch_v1/live_pilot_runtime.py:440-475` — the edge is a single blocking `recvfrom` loop; reassembly, decode, frozen tail, label copy, base64, JSON and 100+ `sendto` calls all run inline. While one frame is processed the socket is not drained, so the only buffer is the kernel SO_RCVBUF.
- `splitfusion_live_dispatch_v1/live_pilot_runtime.py:473` — every result carries `semantic_labels_b64`, a base64 720x1280 uint8 mask, so the result message is ~1.23 MB for **every** action, independent of the feature payload it answers.
- `phase2_map_sharing/transport.py:39-101` — `ChunkReassembler` is all-or-nothing with a 2 s timeout and no retransmission (`runtime.retransmission = false`). One lost datagram destroys the whole message and the loss is never counted at either endpoint.
- `splitfusion_live_dispatch_v1/live_pilot_runtime.py:331-357` — the UE result loop is also a single thread, and it performs a 1.23 MB base64 decode, a 921,600-byte `np.save` to disk and a zlib compression per result before returning to `recvfrom`. Result delivery is serialised and blocking.
- `uplink_only_spatial_map_pipeline/spatial_map_server_moving_ego_uplink_only_baseline.py:1713` — install and `_emit_install_feedback` are unconditional; the map server never inspects a deadline.
- `rl_agent/ue_map_install_feedback_v1.py:192` — `record_expired` does not remove the capture from `pending`, so a capture receives a `TIMEOUT_NO_ACK` terminal row at capture+500 ms and a later ACK is appended as a second, non-terminal, `late=True` diagnostic row. The 100% `TIMEOUT_NO_ACK` terminal distribution alongside non-zero `ack_installed_frames` is therefore correct accounting, not a defect.
- Edge `ready.json` declares `tail_device: cuda:0` and every decoded frame reports `reconstructed_device = cuda:0`. The tail ran on the intended device.

## 5. Clock discipline

- **ue_process_perf_counter_ns** — capture_started_ns, ue_prepare_finished_ns, send_finished_ns, edge_result_received_ns, queue_wait_ms — one process, differences valid
- **edge_container_perf_counter_ns** — edge_timing_ns stage boundaries, edge_received_ns, tail_finished_ns — separate container process; only intra-domain DURATIONS are used
- **host_wall_clock_time_time_s** — capture_at, service_deadline_at, install_timestamp, feedback_emit_at, feedback_received_at, ack_timeout_at — all time.time() on the one physical host (the map server is a local subprocess bound to 127.0.0.1), so differences are valid

- Cross-domain test: Wall-clock capture->install AoI minus UE-perf send->result round trip has per-cell medians spanning 287-861 ms across 12 cells. That residual is exactly the unmeasured capture->send head. It is computed on the installed subset only (24-109 frames per cell), so it is noisier than the full-population head, but it sits in the same 0.3-0.9 s band as the independently measured prepared-queue wait (per-cell medians 513-684 ms) plus front time (16-39 ms), and it is two to three orders of magnitude smaller than the 3 s-110 s AoI spread that the two clocks independently agree on. A constant offset, drift, unit mismatch or timestamp reuse large enough to manufacture that spread would necessarily show up in this residual, and does not. No timestamp was corrected post hoc.
- Intervals declared unavailable rather than reconstructed:
  - `complete_edge_receive_to_edge_processing_start`: live_pilot_runtime.run_edge_service calls edge.process() on the same statement that stamps edge_received_ns; the two instants are not separately recorded.
  - `edge_queue_wait`: No application-level edge queue exists in source. Feature datagrams wait only in the kernel SO_RCVBUF, which is not instrumented and emits no drop counter.
  - `feature_datagrams_observed_at_receiver`: feature_received_datagrams is carried inside the edge result, so it exists only for messages that both completed at the edge and returned intact to the UE.
  - `incomplete_or_expired_feature_reassemblies`: ChunkReassembler.expired_messages is never exported by either endpoint, and a feature message that never completes produces no record anywhere.
  - `result_publication_to_ue_ingestion`: Publication is edge_container_perf_counter_ns, UE ingestion is ue_process_perf_counter_ns; offset unproven.
  - `tail_completion_to_result_publication`: tail_finished_ns is edge_container_perf_counter_ns and is not retained in per_frame_metrics.csv; the publication instant is not stamped at all.
  - `ue_ingestion_to_installation`: UE ingestion is ue_process_perf_counter_ns, installation is host_wall_clock_time_time_s; offset unproven. The wall-clock capture->installation AoI is reported instead.
  - `ue_send_to_complete_edge_receive`: Same unproven ue_process_perf_counter_ns -> edge_container_perf_counter_ns offset; only the UE-observed round trip is computable.
  - `ue_send_to_first_edge_receive`: UE send uses ue_process_perf_counter_ns; the only edge receive timestamp (edge_received_ns) uses edge_container_perf_counter_ns. No pairing event is recorded in both domains, so the offset is unproven.

## 6. Hypotheses

### H1_large_feature_messages_fail_multi_datagram — `CONFIRMED_FOR_ACTION_0_CONTRADICTED_AS_GENERAL_CAUSE`

Action 0 sends 288.0 datagrams/frame and produced 0 installs in 4/4 cells, while action 71 sends 1.0 datagram/frame yet still reached the edge on only 0.468 of sent frames. Fragment count therefore explains action 0 but cannot explain the 0.36-0.46 edge completion rate shared by actions 20, 46 and 71 across a 190x payload range.

### H2_small_messages_overload_downstream_fifo — `CONFIRMED`

The edge is a single blocking recvfrom loop with no application queue, so the kernel SO_RCVBUF (16,777,216 reported bytes) is the only buffer and it is bounded in BYTES, not frames. Median capture->install AoI orders strictly inversely with payload: {0: None, 20: 3889, 46: 33905, 71: 110237} ms for actions 0/20/46/71 at 3.59 MB / 1.21 MB / 100 kB / 6.4 kB per frame. Per-action median AoI monotone fraction {0: None, 20: 0.59, 46: 0.77, 71: 0.97}: action 71 AoI rises almost strictly monotonically through its route (never saturating inside ~430 s), while actions 20 and 46 plateau at a fixed backlog depth. That is the signature of a FIFO whose depth in frames, not in bytes, sets the delay.

### H3_expired_frames_keep_consuming_work — `CONFIRMED`

`service_deadline_at`/`ack_timeout_at` occur in the runtime only where a CSV row is labelled (`ue_map_install_feedback_v1.py:148` and `:198`) and where the capture deadline is first computed (`ue_route_b_split_cell_adapter_v1.py:1084`). They appear nowhere in `live_pilot_runtime.py` or the map server, so no edge admission, tail, publication or install step is ever gated on them. Installed frames carry median AoI up to 110237 ms against a 500 ms ack timeout, so post-timeout work is measured, not merely possible.

### H4_result_downlink_is_the_bottleneck — `CONFIRMED`

The median result message is 102-103 datagrams in every cell (per-message range 101-105), regardless of action, because the payload is a base64 720x1280 uint8 label map. Across all 16 cells the edge is directly observed to have tail-completed at least 11,183 frames while only 747 results returned intact to the UE. Because the edge counters are LOWER bounds, the per-action survival rates {0: None, 20: 0.0595, 46: 0.0694, 71: 0.0607} are UPPER bounds: the true result-path survival is at most ~7% and may be lower. The uplink cannot explain this, because the same bound holds for action 71, whose feature is a single datagram.

### H5_tail_on_unintended_device_or_high_latency — `CONTRADICTED_ON_DEVICE_CONFIRMED_ON_LATENCY`

reconstructed_device is ['cuda:0'] on every decoded frame and edge ready.json declares tail_device cuda:0, so the device is the intended one. Median frozen_tail is {0: None, 20: 177.2, 46: 169.7, 71: 182.5} ms by action, which alone caps the edge below 6 Hz against a 10 Hz offer.

### H6_timestamp_domain_mismatch_creates_artificial_latency — `CONTRADICTED`

Wall-clock capture->install AoI and UE-perf send->result round trip are measured on two independent clocks. Their per-frame residual is the unmeasured capture->send head, and its per-cell medians span 287-861 ms across 12 cells (computed on the installed subset only, so it is noisier than the full-population head). That is the same order as the independently measured prepared-queue wait plus front time, and two to three orders of magnitude below the 3 s-110 s AoI spread the two clocks agree on. A domain offset, drift, unit error or timestamp reuse large enough to manufacture that spread would have to appear in this residual and does not. No timestamp was corrected post hoc.

### H7_evaluation_work_reduces_preparation_coverage — `STRONGLY_SUPPORTED_MAGNITUDE_UNRESOLVED`

ue_route_b_split_cell_adapter_v1._process_token calls _ground_truth on the single preparation worker thread after send, and _feedback_worker/_segmentation_worker run further GT and mask work in the same interpreter. Measured preparation_start->encoding_complete is only 16-39 ms while the achieved worker period implied by coverage is ~135-150 ms, so ~110-120 ms per frame is spent outside split inference. No per-stage timer separates GT from radar/image assembly.

### H8_radar_sync_or_scheduling_causes_most_prepared_drops — `CONTRADICTED`

Of 14697 classified preparation losses, 14539 (98.93%) are DROPPED_QUEUE_FULL, 142 (0.97%) are DROPPED_SENSOR_LATE_OR_MISSING and 16 (0.11%) are DROPPED_INCOMPLETE_RADAR_WINDOW. Radar synchronisation accounts for roughly one percent of preparation loss.

## 7. Preparation-loss attribution

| Cause | Frames | Share |
|---|---:|---:|
| bounded_preparation_queue_overflow | 14539 | 0.9892 |
| sensor_synchronisation_late_or_missing | 142 | 0.0097 |
| missing_or_incomplete_radar_window | 16 | 0.0011 |
| split_processing_failure | 0 | 0.0000 |

Total classified preparation losses: **14697** (plus 153 warmup frames before the first complete radar window, which are not losses). Preparation is **stationary**: no cell shows a monotone quartile trend in coverage or in median prepared-queue wait, the full quartile spread is at most 0.204, and coverage peaks in route quartile 2 in 15 of 16 cells. A pattern that reproduces at the same route position across 16 independent cells and all four network profiles is a route-geometry effect, not drift and not a growing backlog. Preparation loss is therefore a steady-state throughput deficit, and it is independent of the network and of the transport-side AoI growth.

## 8. Evidence-supported inference

- The edge socket receive buffer is bounded in bytes (16,777,216 reported), not in frames, so its depth in FRAMES is inversely proportional to payload size: ~4.7 frames at action 0's 3.59 MB, ~14 at action 20's 1.21 MB, ~167 at action 46's 100 kB and ~2,620 at action 71's 6.4 kB. With a measured edge service rate of only 2.2-3.1 Hz against a ~6-7 Hz offer, backlog accumulates until the buffer saturates, and the saturated backlog delay is (buffer frames / service rate). That predicts a few seconds for action 20, tens of seconds for action 46, and a backlog that cannot saturate inside a ~430 s route for action 71 — which is exactly the measured ordering and exactly the measured monotone AoI growth for action 71.
- The inverse latency ordering is therefore REAL and mechanistic, not a clock artifact and not a property of the radio: a smaller payload buys more admitted frames, and every admitted frame is served FIFO from an ever-older backlog.
- Action 0's zero installs are consistent with uplink fragment loss: 288 datagrams/frame at ~6 frames/s is ~196 Mbps offered on the uplink, far above the registered 100 MHz 4D5U profile, and one lost fragment of 288 destroys the message. A weaker but independent bound: if action 0's edge had completed as many frames as action 20's (419-1036), then at action 20's measured ~6% result survival rate the probability of observing zero returned results across four cells is negligible. Action 0 almost certainly failed on the uplink, not on the downlink — but see the unresolved list, because no edge-side evidence survives for it.
- The 100 ms service deadline is unreachable before the network is even reached. The prepared-queue wait alone has per-cell medians of 513-684 ms, and the front adds 16-39 ms, so a frame is already 0.5-0.7 s old at the instant the UE finishes sending it. Even a zero-latency transport and a zero-latency edge could not have produced a single on-deadline install in this pilot. This is measured on one clock inside one process and does not depend on any cross-domain assumption.
- Preparation coverage of 0.67-0.76 follows arithmetically from a single-threaded worker serving ~7.4 frames/s against a 10 Hz opportunity stream through a depth-4 FIFO. Split inference accounts for only 16-39 ms of that ~135-150 ms worker period.

## 9. Unresolved questions (evidence not retained)

- How many feature datagrams actually arrived at the edge. Neither endpoint exports `ChunkReassembler.expired_messages`, and a message that never completes leaves no record, so uplink radio loss cannot be separated from kernel-socket-buffer overflow at the edge.
- How many result datagrams the edge actually sent and how many arrived. The result path has no sender-side counter and no per-datagram receipt record, so the measured 3-6% result survival cannot be decomposed into downlink radio loss versus UE receive-buffer overflow versus UE result-loop blocking.
- Whether action 0 ever completed a single message at the edge. No result returned, and every edge-side counter travels only inside a returned result, so action 0 has no edge-side evidence whatsoever.
- The split between evaluation-only ground-truth work and radar/image assembly inside the ~110-120 ms of non-split worker time. `_process_token` has no per-stage timer around `_ground_truth`.
- Whether the ~180 ms median frozen-tail latency is inherent to the tail or reflects GPU contention with the co-resident UE front on the same cuda:0 device. Nothing records GPU occupancy or per-process utilisation.

## 10. Minimal remediation specification (NOT implemented)

- Take the segmentation mask off the deployment downlink WITHOUT losing it. `semantic_labels_b64` makes every result ~1.23 MB in ~102 datagrams for every action, and that fixed cost is what destroys installed-frame delivery. The mask is evaluation-only evidence, and the edge already has a writable state mount (`/work/torch_cache`), so it can be persisted edge-side and correlated offline by frame_id while only the object records travel the downlink. This preserves segmentation evidence coverage and the measurement contract; it must not be implemented as simply deleting the mask.
- Make the edge non-blocking: drain `recvfrom` on a dedicated thread into an explicit bounded, latest-frame-first queue with a classified drop counter, so the kernel byte-buffer stops acting as a hidden unbounded FIFO and the payload-inverse AoI ordering disappears.
- Enforce the deadline. Carry `capture_timestamp_ns` (already in the SFD1 v2 envelope) into an explicit obsolescence test at edge admission, before the tail, and at map install, and count each discard. Today nothing reads the deadline, so 100% of the edge's work after the first few seconds is spent on frames that can never be timely.
- Move evaluation-only ground truth off the preparation worker onto its own bounded queue, and add a per-stage timer so preparation coverage loss becomes attributable rather than inferred.
- Instrument the two blind stages: export `ChunkReassembler.expired_messages` and per-message expected/received datagram counts at both endpoints, and add a result-path sender counter. Without these, uplink loss and buffer overflow remain permanently inseparable.
- Re-derive the 100 ms `service_deadline_ms` only after the 0.5-0.7 s capture->send head is fixed; the threshold itself is not the defect and must not be relaxed to manufacture a pass.

### Shortest follow-up measurement

Two cells only — actions **20** (`split_ae128_uint8_q5000`) and **71** (`split_ae32_uint4_q9800`) under **FAVORABLE_STABLE**, one route each — with the mask removed from the result path, an explicit bounded latest-frame-first edge queue, deadline-based discard, and the two new loss counters. Those two actions bracket the payload range by ~190x and are the two extremes of the observed inverse AoI ordering, so they are jointly sufficient to falsify the fix. The predicted direction and magnitude, to be pre-registered by Abiodun before the run rather than set by this audit, is: installed-frame AoI for the two actions converges to within roughly one edge service period instead of differing by ~30x; result survival rises by more than an order of magnitude from the measured <=8% upper bound; and AoI stops growing monotonically through the route for action 71. A second 16-cell pilot is NOT requested and would add no discriminating power until this two-cell probe settles the mechanism.

## 11. Scope of the conclusion

`ROOT_CAUSE_LOCALIZED` is claimed at **stage** granularity and no finer. Specifically:

- **Localized by direct measurement**: the deadline miss (over-determined by the UE head alone); the preparation deficit and its 98.93% attribution to bounded-queue overflow; the inverse AoI ordering and its monotone growth; the collapse between edge tail completion and result arrival at the UE.
- **Localized by evidence-supported inference, not direct measurement**: the byte-bounded-FIFO depth arithmetic that predicts the per-action AoI plateaus; action 0's failure being on the uplink.
- **NOT localized, and no amount of re-analysis of this evidence will localize it**: uplink radio loss versus edge socket-buffer overflow; downlink radio loss versus UE result-loop blocking; the split between evaluation GT and sensor assembly inside the UE worker period. These need the counters listed in the remediation specification.

This audit did not run any experiment, did not modify retry4, and did not move any gate, timeout, queue size, payload, threshold or trace. The 500 ms ack timeout and the 100 ms service deadline are reported against as-registered and are not the defect.

Conclusion: `ROOT_CAUSE_LOCALIZED`
