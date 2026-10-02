# Decision memo: one remaining actionable bottleneck in the cap-only pipeline

Scope: 30-minute review of retained evidence only (no new captures). Configuration
under review: UE `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, production CPU radar
preparation, frozen Run-4 actor/state/reward, Q_perc-inclusive 170-ms deadline,
k_min = 2. Primary evidence (exports re-verified against their manifests):
C `20261002T032922Z_radar_bundle_arm_C_control_1000` and B
`20261002T024743Z_threadcap_arm_B_treatment_1000`. The rejected radar treatment T
is used only as contrast. HEAD 98813d5.

## 1. What accounts for late and missing feedback

Per reward ticket, by exact identity, on one raw clock (C; B agrees within 1-3 ms):

| Segment, P50 ms | timely (339) | received late (129) | missing (7) |
|---|---|---|---|
| action-open -> last datagram (UE) | 25.6 | 35.2 | 20.7 |
| last datagram -> edge reassembly (uplink) | 52.2 | 62.3 | 47.5 |
| edge queue (reassembly -> take) | 0.5 | 0.5 (P95 39.9) | - |
| edge process (verify+compute+map build) | 51.9 | 59.4 (P95 132.7) | - |
| map publish, then evaluator submit | 1.7 + 1.7 | 1.8 + 1.8 (P95 9.7 + 10.8) | - |
| submit -> Q_perc start / Q_perc / emit -> UE receipt | 2.2 / 3.3 / 5.1 | 2.9 / 3.7 / 5.6 | - |
| **action-open -> UE receipt** | **149.2** (P95 166.8) | **186.1** (P95 271.9) | - |

- **Received-late (129 in C, 116 in B):** the timely median already sits at 149 ms
  of 170. Late tickets are ~10 ms slower in each of the three big stages (UE
  send, uplink, edge) rather than broken in one. The excess over 170 ms is small
  for most: P25/P50/P75 = 5.4/16.1/36.4 ms (C), 5.1/13.0/28.6 (B); 30/129 and
  29/116 are within 5 ms, 53 and 48 within 10 ms. Only 1 late ticket per arm
  coincides with an edge stall (process > 70 ms overlapping a > 60 ms CARLA
  tick). GT readiness is not a cause (GT ready at action-open + 45 ms, far
  before the tail finishes).
- **Missing (7 in C, 2 in B):** 6 + 2 are `REGISTERED_SERVICE_FAILURE` =
  `EDGE_PENDING_REPLACED`: the reward frame was still in the edge's one-slot
  pending queue when the next held frame arrived ~100 ms later, i.e. the edge
  was occupied >= ~100 ms by a stalled predecessor (C frames 2018/2138/2520/
  3030/3034/3040; B 3033/3277). 1 in C (frame 1608) never reached the edge in
  time (uplink 112.7 ms) and timed out.

## 2. Edge stalls versus ordinary frames

- Ordinary edge frame (C, P50): verify 0.6, zstd 0.2, dequantize 4.8, AE 1.1,
  **frozen tail 38.0**, publish/serialize 2.2, map build 1.8; host-side gap
  inside `edge.process` only 0.16 ms (P95 0.47); tail wall equals its CUDA
  interval (-0.03 ms). Output serialization is CPU record building on
  already-CPU tensors, not a D2H copy.
- Stalls (edge.process > 120 ms: 23/981 in C, 5/993 in B): in every case the
  **frozen-tail CUDA interval balloons** (83-347 ms) while decode stages stay
  normal, and the stall overlaps a long CARLA tick (90-393 ms; the tick itself
  stalls) and UE GPU work, at 57-63% GPU utilization (not saturated). Examples:
  C frame 3024 (process 424 ms, tail 347, tick overlap 393), 2022 (261/188/131),
  3036 (226/189/192); B 3275 (265/148/132), 1501 (207/192/100). Stalls do not
  cluster on the edge's 1-s report period (phase uniform) and do not grow over
  the run (process P50 by fifth 53.6-54.1; `json.dumps` of the full report is
  <= 6 ms) - the periodic report writer is **not** a cause.
- Held-frame arrival bursts: 10 Hz arrivals with inter-arrival P5 32.6 ms (C);
  supersessions (18 in C, 35 in T) rise when arrivals bunch behind a stall.
  Bursts are a consequence of stalls plus earlier UE commits, not an
  independent cause.
- **What the retained data cannot distinguish:** inside a long CUDA-event
  interval, kernel slow-down under time-slicing versus context-switch /
  submission waiting. Missing measurement: one short per-kernel GPU timeline
  (Nsight Systems or CUPTI, ~20 s, edge + UE + CARLA) - not another campaign.

## 3. Source review: one concrete avoidable serialization on the 170-ms path

`rl_agent/splitfusion_hybrid_sac_live_route_b_v2/phase6_edge_runtime_v2.py`,
`process_loop` and `Run4EdgeProcessorV2.process`: for a reward frame the compute
thread (1) runs the tail, (2) **builds the map update** (`MP.build_run4_map_update`,
1.75 ms P50), returns, (3) **publishes the map** (`Run4MapPublisherV2.publish`:
zlib + chunk + UDP, 1.65 ms P50, P95 9.7 ms on late frames), and only then (4)
`evaluator.submit(processed.evaluation)` hands the ticket to the evaluator
thread. Steps 2-3 are required work for the map, but their position ahead of the
submit is pure serialization: the ticket's inputs (`records`, `labels`,
`gt_identity`) exist at the end of step 1, `build_run4_map_update` copies the
records (`[dict(item) for item in records]`, run4_map_protocol_v2.py:149) so
nothing it does can alter the evaluator's input, and the evaluator runs on its
own thread. Measured cost on the reward path: compute end -> evaluator submit
1.7 ms P50 / 2.7 P95 (timely), 1.8 / 10.8 (late), plus the map build 1.75 ms
inside `process`. Everything else on the edge host side is required work or
negligible; the UE's 7-channel upload (float conversion before the H2D copy,
~1-2 ms) is a smaller second candidate and is not proposed now.

## 4. The ONE proposed intervention

**Submit the evaluation ticket immediately after the tail compute, before the
map update is built and published.** Same thread, same objects, same ticket
contents; the map update and publication follow in the same order as today,
delayed only by a queue put. Expected effect from the retained tickets: the
reward path shortens by ~3.4 ms P50 (up to ~20 ms at P95 on late frames); had
it been in place, 23/129 late tickets in C and 23/116 in B would have met the
deadline (timely 71.4 -> 76.2% and 75.2 -> 80.0%), with no change to quality,
identities, held-frame processing, latest-frame semantics, or map delivery
content. It targets the 170-ms path directly because the saved time sits
between tail completion and feedback emission on every reward ticket.

Parity checks required before live: (a) evaluator input identity - for
replayed frames the `records`/`labels`/`gt_identity` the ticket carries are
byte-identical under both orders (records are copied by the map builder;
assert no aliasing); (b) map update bytes identical (`validate_run4_map_update`
and the published payload digest unchanged); (c) Q_perc/reward arithmetic
unchanged (existing phase-6 tests); (d) terminal/supersession accounting
unchanged (`send_terminal` paths untouched); (e) no new synchronization on the
compute thread. Risks: GIL interleaving - the evaluator's GT read and 3.3-ms
numpy evaluation now overlap the map build/publish on the compute thread,
which may delay the map publish by a few ms (map AoI is not on the 170-ms
path; measure it); the evaluator may now observe GT slightly earlier (GT is
ready ~90 ms before the tail finishes, so no change expected).

## 5. Minimal test plan (qualified profiler, cap-only control configuration)

Control: current order (fresh 1,000-frame capture). Treatment: submit-before-
publish, implemented as a process-local seam in the edge entry point around the
unchanged `phase6_edge_runtime_v2` (no pinned file edited), registered as a
sealed addendum with the arm, changed files, tests and fresh output roots.
Primary: timely feedback / all policy decisions; action-open -> feedback
P50/P95/P99 with missing replies separately. Secondary: compute end -> evaluator
submit, map build/publish latency, capture -> map install (map AoI), edge
queue, supersessions, Q_perc values on replayed identities. Gates G1-G9 and
E-checks as before; the 999-frame boundary exception and safety bounds retained;
no retry once frames are sent. One pair; a ~5-pt change is within the observed
4-pt run-to-run spread, so a null result would be inconclusive, not negative.

Not proposed: moving action-open, relaxing 170 ms or freshness, dropping held
frames, changing latest-frame admission, removing Q_perc, precision, retraining,
MPS or GPU partitioning. The GPU time-slicing stalls remain the dominant cause
of missing replies and the extreme tail; they need the per-kernel timeline
above before any intervention can be justified.
