# M4 timing table definitions

`FRAME_TIMING.csv` has one row per `(stream_id, frame_id)` and retains source session, UE ID, RNTI, tensor sequence, action kind, mode and q. Both sources reuse the same 208 CARLA frame IDs, so frame ID alone is never a join key. All 416 controller assignments match transmitted identities and exactly one source-bound terminal. Every edge wire's action identity and capture timestamp match its originating controller record.

Empty cells mean **unknown or inapplicable**, never zero. Terminal receipt timestamps are evidence of a terminal only; neither timeout nor STALE timestamps become prediction latency. Quantiles use NumPy's linear percentile definition; interpolated P95/P99 can differ from historical nearest-order-statistic summaries.

## Recorded boundaries

| Columns or stage | Evidence and interpretation |
|---|---|
| `capture_wall_ns` | Controller `capture.ns`; CLOCK_REALTIME at the RGB callback in the retained collector, not an exposure-time measurement or CARLA simulation time. |
| `rgb_receipt_raw_ns` | Controller stages; CLOCK_MONOTONIC_RAW RGB callback receipt. This is the anchor for pre-send RAW intervals, not a conversion of capture WALL. |
| Radar window, tensor, sensor preparation, SI/P40, input 7ch, front start/end | Controller `stages`, all RAW, present on every sent frame. Front span includes front network, ranker, AE and codec; separate encoding-only and GPU-contention time are unavailable. |
| `action_open_raw_ns` | Controller `action_open.ns`, RAW. Present on policy decisions and fallback openings, absent on holds. Policy latency summaries exclude fallback/holds. |
| Ledger staged, commit registered, first/last packet send | Controller `stages`, RAW. First/last marks bracket the userspace send loop. They are not physical radio transmission timestamps. |
| `edge_verified_raw_ns`, `edge_verified_wall_ns` | First source/frame `edge_wire` event, 207/208 rows per UE. Verification follows full reassembly; it is not a kernel arrival timestamp. Successful frames may have queued before verification. Expired-after-reassembly frames are verified on the terminal path without entering the compute queue. Duplicate verification of the two before-tail failures is not counted twice. |
| `admission_log_raw_ns` | `reassembled` event, 35/29 frames. Emitted only after the 500-ms deadline admission check; not all completed reassemblies. |
| `reassembly_last_recv_wall_ns` | Map wire `edge_timing.reassembly_complete_wall_s`, available for 33/28 completed predictions. Despite the legacy name, the writer samples WALL on entry to `DataPath.receive`, after `recvfrom`, before ingest/unpack/deadline checks on the completing datagram. Actual post-assembly timestamp is not separately retained. |
| `compute_start_wall_ns` | Map wire timing, before processor verification/decode/tail. Edge queue is that time minus the preceding receive-entry WALL mark. |
| `prediction_ready_wall_ns`, `prediction_ready_raw_ns` | Explicit `prediction_ready` event after compute returns CPU-published predictions. Present only for 33/28 completed predictions. Never inferred from terminal messages. |
| `gt_published_raw_ns`, `gt_ready_score_admitted_raw_ns` | GT service events `published` and `target_admitted`, available for 67/63 policy decisions. The latter means target admission at the scorer, not scoring completion. |
| GT prediction received/result sent; feedback emitted | Corresponding service events, RAW. GT readiness runs concurrently with the prediction path; its duration is not added to prediction latency. The emitted notification is recorded after the sending operation. |
| `policy_feedback_raw_ns` | Controller R4FB receipt, RAW, matched to the same source/frame/decision. 14/11 delivered scored replies; all late. |
| `map_arrival_wall_ns`, `map_install_wall_ns` | Map observations, WALL. Install timestamp exists only when the map accepted the result. |
| `ledger_receipt_wall_s` | Source map-feedback ledger terminal observation, WALL seconds. `capture_to_map_ack_ms` is computed only for RESULT_INSTALLED, not local watchdogs or rejections. |
| `kernel_receive_wall_ns`, `exact_reassembly_raw_ns`, `encoding_only_ms`, `gpu_wait_ms` | Explicitly unknown for all rows. |

## Clocks and populations

WALL and MONOTONIC_RAW are never directly subtracted. The separately logged radio-command `command_ack_monotonic_ns` is ordinary MONOTONIC and is not used as RAW. Acknowledged actor features use the recorded RAW acknowledgement observation instead. Paired edge WALL-minus-RAW offsets span only 0.340 ms across the evidence, a sanity check rather than a clock conversion. Timing values from floating-point WALL seconds carry their original precision.

Figure 2's left panel divides observed verified frames by **all 208 sent frames per source**; the missing UE0 frame remains in the denominator. This is an observation-age curve, not a prediction or packet-arrival CDF. Its 500-ms line references the actual processing horizon, while the event itself is a later observation. Explicit terminal stage, not the curve alone, establishes 172/179 after-reassembly expiries. The right panel divides completed policy predictions or scored replies by **all 103/100 policy decisions** and uses action-open RAW; incomplete outcomes never become artificial late samples. No stage quantiles are summed.

The radio reader reuses the retained time-of-day reconstruction functions without importing live/model code. It reconstructs event WALL against receipt WALL with UTC−7 for this capture, preserving the original two-second reconstruction guard. Main PUSCH statistics use each source's first-to-last capture WALL window; whole-log and capture-through-last-edge-event windows are separately reported. RNTI attribution is taken from `ATTACHED.json`, not inferred from traffic order or NAT addresses. No packet/frame-to-PUSCH mapping is claimed. RLC samples require both RNTI and UE ID and sum distinct LCIDs within a scheduler tick; the incomplete final tick is excluded.

Known SFD4 byte lengths come from admission events or actual retained wire-file lengths, with overlaps checked. Sender aggregate bytes include eight-byte chunk headers; mean SFD4 length subtracts exactly those headers. Payload quantiles over missing frames are not reconstructed from mode/q because zstd output depends on data.

## Interpretation source binding

Eleven inspected M4, edge, wire and telemetry implementation files match their exact registered source hashes. The inherited base collector is not separately listed in the outer M4 inventory; its current main/deployment copies agree byte for byte. This narrower scope is explicit in `SOURCE_BINDING.json`. Existing replay/rescoring results are cited from the immutable historical assessment, not freshly asserted from model execution.
