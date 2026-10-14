# M4 source isolation implementation and offline qualification

**The GT ownership and Run4 adapter blockers are resolved in the tested components. Full M4 remains incomplete, not ready for live preparation.** The remaining implementation is the two-ego world/collector bootstrap and production OAI, state-builder and endpoint wiring. No deployment, live registration, CARLA/OAI launch or GPU inference occurred. The original REPORT.md and all failed proofs remain unchanged.

## Ownership and implemented files

All changes are confined to `rl_agent/splitfusion_m4_two_ue_v1/` and this evidence directory:

- `identity.py`: explicit ego/UE/IMSI/tunnel/IP/RNTI/run/stream/attach-session bindings; rejects ambiguous rosters. Fixture network identities are synthetic, not observed leases.
- `gt.py`, `service_entry.py`: one CPU service, producer history, accounting session, prep/score pair, capability/socket, cancellation and retirement lifecycle per source. Shared-edge dispatch selects the bound evaluator without `_instance`, installation scopes or global source redirection. Original frame and ticket numbers remain intact. Replies carry source and exact action identity internally; the original R4FB wire is unchanged and destination checked before delivery.
- `transport.py`: channel-bound reassembly, followed by the unchanged latest-per-stream pending slot and deadline check. NAT-observed addresses are not identities; supersession stays within one source.
- `map_adapter.py`: authoritative Run4 validation first, then an explicit cooperative identity lookup and new observation envelope. Continuous q, mode, optional anchor, records and timestamps survive unchanged. Pose provenance and map-feedback identity must match. Headless JSON only; no fusion. A 250-ms installed result is accepted independently of the 170-ms feedback boundary.
- `composition.py`: independent frozen-actor controllers/telemetry, shared read-only CPU actor weights, and both-source completion/drain bookkeeping. This is not yet the live world/collector constructor.
- `offline_checks.py`, `qualification.py`: focused regressions, real local service controls, exact rescoring and the bounded capacity comparison.

The retained containment worker composition still installs accounting **V3.v2**, capability checks, tilt geometry, FREEZE lifetime repair, GC, capacity and cleanup inside each isolated child. Its numerical code was not edited. One shared inference callback serves both sources; no GPU/model stack was duplicated. An explicit source-local socket shutdown fixes the retained blocked-receiver join during one-source teardown. Early GT refusals are retained until the corresponding prediction ticket exists, preventing a lost fault response.

## Correctness evidence

[offline_checks_04](offline_checks_04/RESULT.json): **9/9 passed**. Includes frame 4172 and equal sequences, interleaved chunks, reversed replies, duplicate/unknown/wrong-source replies, wrong destinations, session rollover, source-specific supersession/terminals, frozen actor context independence, off-anchor Run4 maps and both-source drain/error handling. Negative controls exercise the actual boundary methods; these are not live radio tests.

[correctness_04](correctness_04/SUMMARY.json): **both sources exactly match their isolated real-service controls**. The single retained corpus is explicitly reused as two synthetic sources, with different cancellations/refusals/interleavings—not represented as two independently captured egos.

| Concurrent source | Frames | Scored and exactly rescored | Other outcomes |
|---|---:|---:|---|
| UE 0 | 11 | 7 | 1 cancellation, 1 geometry refusal, 1 horizon timeout, 1 edge terminal |
| UE 1 | 14 | 12 | 1 cancellation, 1 geometry refusal |

Parity includes every numerical measurement, Q_perc, feedback kind/reason, commit and before/after history hashes, and retirement flags. Only scheduling/receipt timestamps are outside that semantic comparison. Cancellation and contained refusal leave history consistent with each isolated control. UE 1's bound, uncommitted frame **1495** survived UE 0's complete teardown, then completed normally. Both accounting sessions closed; retained jobs/slots, payload/reservation bytes and waiting predictions reached zero. All successful services/workers exited normally.

Earlier evidence is retained: smoke_01 encountered sandbox socket restrictions; smoke_02 passed using authorized local IPC. correctness_01 exposed test-order-dependent outcomes, corrected with observed service-state barriers. correctness_02 exposed the blocked receiver join; correctness_03 exposed its expected closed-socket wakeup being treated as a fault. correctness_04 passes the corrected code. These were implementation checks, not live attempts or performance repetitions.

## One bounded capacity comparison

[capacity_01](capacity_01/CAPACITY.json): one source versus two concurrent sources, **49 offers per source at 200-ms cadence**, without waiting for each score before the next offer. Every arm scored and exactly rescored 49/49; both concurrent sources match the single-source semantic output. No refusals or errors. Queue maxima were one prep request, one score request, one retained prep job and one scorer slot per source; freeze backlog remained zero and all queues drained. Maximum release lag was 1.21 ms.

Pairs below are P50/P99 in milliseconds; each worker-operation row has n=49 per source. Accounting has n=196 calls per source.

| Measurement | One source | Two sources UE 0 | Two sources UE 1 |
|---|---:|---:|---:|
| FREEZE thread CPU | 64.59 / 81.00 | 68.88 / 87.93 | 69.16 / 86.28 |
| FREEZE wall time | 64.60 / 82.80 | 69.71 / 94.18 | 69.58 / 87.42 |
| BIND thread CPU | 3.57 / 3.75 | 3.64 / 3.86 | 3.66 / 3.94 |
| Score TARGET thread CPU | 9.94 / 12.28 | 10.22 / 12.95 | 10.25 / 13.51 |
| Score PREDICTION thread CPU | 0.61 / 1.07 | 0.62 / 1.02 | 0.62 / 1.03 |
| FREEZE enqueue-to-dispatch wait | 0.103 / 0.126 | 0.099 / 0.128 | 0.099 / 0.119 |
| Accounting thread CPU per call | 1.26 / 3.95 | 1.42 / 4.07 | 1.48 / 4.07 |

TARGET/PREDICTION are separate scorer operations: scoring occurs when the second side of the join arrives, so neither row alone is an end-to-end scoring latency. All operation wall/CPU/queue distributions are in CAPACITY.json. Prep operation CPU occupancy was 36.5% for one source, 39.0%/39.1% for two; score occupancy was 7.0%, 7.2%/7.3%, relative to the offer window. Coordinator/driver CPU is not included in those figures. Sampled RSS maxima per source were approximately 718 MiB prep, 652 MiB score, 647 MiB coordinator and 16 MiB archive; sums of maxima are not simultaneous private-memory peaks.

Placement was explicitly **prep/score 4/5 and 6/7; coordinators 20/21**, within the retained edge partition. No radio CPUs were reassigned. There was no competing GPU inference. This is an offline placement, not a qualified live repartition. Early/late slices are the first/last ten states of the retained 49-frame history: FREEZE CPU medians 71.73→55.86 ms alone and 76.67→61.14 / 76.53→61.08 ms concurrently. They do not establish late full-Town10-history capacity. No growing backlog or capacity bottleneck was observed in this bounded workload; no live feedback percentage is inferred. The performance comparison ran once (36.08 s); final correctness took 27.16 s and boundary tests 1.49 s.

## Remaining work and next command

The production entry point still must construct two source-bound collectors/sensor sets and native 22-feature/ACKed-SNR builders under **one** full-Town10 world/traffic owner, validate the trailing spawn geometry, connect actual two-UE attach/lease/RNTI discovery and tracer fan-out, and wire the shared edge/map journal with frame-bound pose publication and both-stream teardown. Existing components are available for that integration; the current package does not perform it. Full per-UE/combined live analysis and mandatory startup binding are also not wired. Preserve 100 MHz/273 PRB/4D5U and commanded 23.5 dB; actual two-UE attachment and achieved PUSCH remain future live checks.

**No executable M4 live-preparation, smoke or capture CLI is claimed.** The exact offline commands used, from the working repository, are:

```bash
env -u PYTHONPATH CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /usr/bin/python3 -B -m rl_agent.splitfusion_m4_two_ue_v1.offline_checks --output experiments/splitfusion_m4_two_ue_v1/offline_checks_04
env -u PYTHONPATH CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /usr/bin/python3 -B -m rl_agent.splitfusion_m4_two_ue_v1.qualification --phase correctness --output experiments/splitfusion_m4_two_ue_v1/correctness_04
env -u PYTHONPATH CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /usr/bin/python3 -B -m rl_agent.splitfusion_m4_two_ue_v1.qualification --phase capacity --correctness experiments/splitfusion_m4_two_ue_v1/correctness_04/SUMMARY.json --output experiments/splitfusion_m4_two_ue_v1/capacity_01
```

Outputs are create-only: reproduction needs unused output names. Real service tests require local UNIX IPC permissions. Do not rerun capacity as an automatic optimization loop. The three final source inventories are identical and were rechecked against final files. Their artifact SHA256 values are respectively `358eb75ccee53ec27eca86e18b0c6c5e5b596b102a321eccc55a00b04c9a73fe`, `c1796eaf5e0a9bd99f1b67525456c45e05daabd5a604845520dc5501f279dadd`, and `4ab60715b80f5de6c22999db05411e6786f3c80045fd7c3edf1db4f64734b9e2`.

## Storage and cleanup

Disk free was **12.7486 GB before**, **12.0789 GB after** (11.25 GiB); still below the unchanged 16-GiB live gate. Evidence occupies 651.7 MB including retained failed checks; code occupies 111 kB. No captures were copied wholesale, deleted or archived.

The new GT-only run used 88,592,384 allocated bytes per 49-frame source (1.81 MB/decision). Scaling that component to 1,950 decisions per source gives about **7.05 GB** across two sources; this is part of, not additional to, the previous **18.36-GB whole-pipeline estimate**. Retain the **22.95-GB planning allowance** with 25% margin and provisional **32-GiB free-space budget**, pending complete retention/capacity integration. Neither is a proved peak bound or a lowered gate.

Final host process checks found no M4 service, GT worker or archive child remaining. Successful runs removed their own IPC endpoints. One stale socket inode remains inside the preserved correctness_02 failure directory; it has no live owner and is not reused. No radio/host settings were changed, and no cleanup of user data was performed. Historical verdicts, viewer/Claude files and shared handoffs were untouched. No commit or push.
