# Single-UE policy-feedback latency spikes: offline diagnosis

13 October 2026. Offline and CPU-only. Read-only on two retained captures. No runtime edits, live run, training,
cleanup or commit. M4 (two-UE) is out of scope and is not touched.

## Answer

- **Supported cause.** The downlink is not the cause. The late decisions have two sources:
  1. Most late decisions are **marginal misses** caused by small upstream increases on top of a fixed post-prediction chain.
  2. The extreme tail is mostly **UE-process stalls**. A smaller part comes from GT-preparation and service stalls.
- **Why "99 % of predictions are on time" still gives fewer timely feedbacks.** After the edge completes a prediction, a
  further **24.4 ms (median)** passes before the UE receives the feedback. This was measured paired, per decision, in
  both captures. So a prediction finishing after about **146 ms** usually misses the 170-ms deadline. 68 % (V11) and
  66 % (demo_03) of late decisions miss by 10 ms or less.
- **Contributing stages in ordinary late decisions** (170–200 ms, compared with on-time decisions):
  - UE encode/send: +4.0 / +4.6 ms
  - uplink: +3.9 / +4.9 ms
  - edge compute: +6.5 / +6.2 ms
  - These rise with payload. Spearman correlation of receipt with payload is 0.37 / 0.35.
- **Contributing stages in extreme decisions (>200 ms):**
  - The UE stage expands by about 80 ms on average.
  - In V11, the extra wait for GT expands by 16 ms (prep FREEZE CPU about 170 ms against 74 ms normally).
  - One score-worker gen2 GC of 75 ms (demo_03).
  - Four stalls of 60–160 ms in the UE result thread.
- **Downlink measurement.** The edge-send → UE-tunnel interval does **not** expand in late decisions:
  - P50 4.3 ms late against 3.7 ms on time (V11); 4.4 against 3.9 ms (demo_03).
  - P99 is 8.3–8.9 ms in every group.
  - The DL DCI that carried each feedback was HARQ round 0 for 3,272 out of 3,272 decisions. No DL retransmission occurred in any feedback window.
- **Unresolved intervals:**
  - The mechanism of the UE-process stalls.
  - Inside edge-send → DL DCI: core/UPF vs gNB queue vs TDD slot wait. This is a single interval with no gNB-DL or UPF timestamps. It is material in only 3 demo_03 decisions (26–36 ms).
  - One 45-ms result-return gap (V11, decision 1680).
- **Live diagnostic needed for the downlink?** No. The recorded timestamps already place the spikes elsewhere, so no
  radio follow-up instrumentation is specified (task step 4 stops here).

## Captures and labels

| Label | Root (deployment worktree `…/splitfusion_single_gpu_profiler_v1/`) | Report / verdict kept |
|---|---|---|
| V11_GOOD_ALL | `20261009_m3_route_b_v11/good_all_prospective_01` | PASS_M3_FULL_ROUTE_INTEGRATION 24/24 (accepted reference). The historical V3-pinned tool's STOP on 3 SCHEDULE checks is unchanged. |
| V20_DEMO_03 | `20261013_m3_route_b_v20/good_all_live_view_03` | PASS_M3_FULL_ROUTE_INTEGRATION, CAPTURE_VALID, P0 0.955. It is a **display demonstration, not a performance measurement**. |

**Viewing in demo_03.** The run was watched off-host through an SSH tunnel. The viewer **server** ran on this host
(CPUs 22–23, nice 19, CPU P50 1 %). Those CPUs are inside the edge container's partition (4–7, 20–23,
`PARTITION_APPLY_RESULT.json`). This is a possible contention source and was not measured.

**demo_02 is not used.** It was the latest viewing run, but it is contaminated: it used on-host viewing and recording,
the disk filled, and the host exit status was 1. Its capture root has been deleted.

**Static-modes substitution was not needed.** demo_03's timing records were complete.

Both captures run GOOD_ALL at 23.5 dB (RFsim −12), traffic 50/50, with the same route. The sessions and scenes differ,
so V11 against demo_03 is descriptive only.

## Reproduction before attribution

| | decisions | feedback ≤170 ms | prediction ≤170 ms | missing receipt | service outcome |
|---|---:|---:|---:|---:|---|
| V11 recorded = reproduced | 1,771 | 1,546 | 1,752 | 0 | 1,771 SCORED |
| demo_03 recorded = reproduced | 1,501 | 1,104 | 1,476 | 0 | 1,501 SCORED |

- "Recorded" means the analyzer's `METRICS.json`. `diagnose.py` refuses to proceed unless the counts match.
- Controller resolutions:
  - V11: SUCCESS 1,459, TIMEOUT 225, EVALUATOR_FAULT 87.
  - demo_03: SUCCESS 1,068, TIMEOUT 397, EVALUATOR_FAULT 36.
  - EVALUATOR_FAULT means the decision was scored, with quality undefined because there was no eligible GT. Its receipt is still timed.
- No decision was refused or timed out without a receipt. Both captures contain zero missing outcomes, and the code would report them as group MISSING.

## Definitions

- **Time origin.** The UE `action_open` (`CLOCK_MONOTONIC_RAW`) of each decision. This is the origin of the 170-ms
  inclusive deadline. CARLA capture happens about 68 ms earlier and is outside the deadline.
- **Feedback receipt.** The UE result thread reads `CLOCK_MONOTONIC_RAW` immediately after `recvfrom()` returns
  (`phase6_ue_runtime_v2._result_loop`). A single host was used. The edge container shares the host `boot_id`, so the
  edge's raw clock is the host clock.
- **Edge "emitted" (`feedback_emitted` in the existing analyzer).** This is a **post-send** stamp, taken after
  `_emit()` → `sendto()` returns. The `edge.feedback_emit` profiler span starts **before** encode+`sendto`. This memo
  uses that span start as the pre-send boundary. The span lasts 0.10 ms at P50.
- **UE tunnel arrival.** The kernel rx stamp of the exact feedback datagram on `oaitun_ue1`
  (`quality_ack_oaitun_ue1.pcap`). It is joined by SHA-256 of the UDP payload to the edge `feedback_sha256`:
  1,771/1,771 and 1,501/1,501 matched, with zero duplicates.
  - **Wall→raw bridge.** Uses the per-second upper envelope of the radio-telemetry drain's back-to-back `(wall, raw)`
    reads (108k / 92k pairs).
  - **Clock drift.** Wall time drifts against raw by **−3.2 ms (V11) and −13.0 ms (demo_03)** over the run, so a
    constant offset would be wrong by more than the downlink interval itself.
  - **Validation.** The mapped edge `emit_wall_ns` minus the raw `emitted` stamp is ≤ 0 for every decision, at
    −0.017 / −0.019 ms P50. The two stamps are taken consecutively. No decision has tunnel arrival after UE receipt
    or before the pre-send stamp.
- **DL DCI.** The last UE-side `NRUE_MAC_DCI_GRANT` with direction DL inside [pre-send, tunnel arrival]. It comes from
  `ttracer/ue/ue.raw`, decoded read-only; the file parses to its exact length, and the stamp is `CLOCK_REALTIME` at
  the UE MAC. Exactly one DCI was found in 1,755/1,771 and 1,482/1,501 windows, otherwise two. The gNB T-tracer
  holds UL streams only.
- **Stage partition.** Fourteen consecutive intervals per decision that sum **exactly** to the receipt. The maximum
  residual is 6e-14 ms, there are no negative stages, and the partition is complete for 3,272/3,272 decisions. The
  intervals are:
  - UE actor → wait before 7-ch input → input build → encode+send
  - uplink (last datagram sent → edge reassembly)
  - edge queue+compute
  - prediction edge → GT service
  - **extra wait for GT** = max(0, GT published − prediction received at service)
  - score-channel wait after both inputs → scoring op
  - result service → edge
  - edge feedback build (pre-send)
  - **edge send → UE tunnel** (send syscall, docker bridge, UPF/GTP-U, gNB PDCP/RLC/MAC queue, DL scheduling and TDD
    slot wait, RFsim, UE PHY/MAC/RLC/PDCP, TUN write — **not pure over-the-air latency**)
  - **UE tunnel → app receive** (kernel socket + Python thread wake)

  Concurrent work, such as GT preparation while the prediction is in flight, appears only through its join (the extra
  wait), so nothing is double-counted.
- **Groups.** On time ≤170 ms; late 170–200 ms; extreme >200 ms. The extreme threshold is a fixed 200 ms, not a
  per-capture percentile.
- **Spike class.** Each late decision is compared with the on-time median of each stage. If one stage exceeds that
  median by ≥20 ms, the decision is SINGLE_STAGE for that stage; otherwise it is DIFFUSE.
- **UE GT-freeze stall.** UE `freeze_start → freeze_frozen` ≥ 60 ms. The on-time P50 is 11.8 ms.

## Results

### 1. The post-prediction chain makes a 99 % prediction rate compatible with a 74–87 % feedback rate

The chain from prediction ready to UE receipt is measured paired over all decisions:

| | P50 | P95 | P99 |
|---|---:|---:|---:|
| V11 | 24.4 ms | 37.2 ms | 59.4 ms |
| demo_03 | 24.3 ms | 34.4 ms | 45.2 ms |

On-time medians of the parts of that chain (V11 / demo_03):

| Part | V11 | demo_03 |
|---|---:|---:|
| Prediction edge → service | 6.1 ms | 6.2 ms |
| Score wait | 1.9 ms | 1.8 ms |
| Scoring op | 9.3 ms | 8.9 ms |
| Result → edge | 0.6 ms | 0.6 ms |
| Feedback build | 0.13 ms | 0.14 ms |
| **Edge send → UE tunnel** | **3.7 ms** | **3.9 ms** |
| Tunnel → app | 0.04 ms | 0.05 ms |

- The downlink-side interval is about 15 % of the chain.
- Late decisions whose prediction completed after 170 − 24.4 ≈ 145.6 ms: 145/225 in V11 and 305/397 in demo_03. Only
  34 and 66 such decisions were still on time.
- Group means of receipt:
  - V11: 154 ms on time, 177 ms late (n = 199), 268 ms extreme (n = 26).
  - demo_03: 157 ms on time, 178 ms late (n = 371), 259 ms extreme (n = 26).

### 2. Which interval expands: ordinary late decisions are diffuse, extremes are single-stage

Spike classes of late decisions:

| | DIFFUSE | GT-wait | UE stages* | edge | score | result return | tunnel→app | send→tunnel | uplink |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V11 (225) | 165 | 19 | 27 | 9 | 1 | 2 | 2 | 0 | 0 |
| demo_03 (397) | 329 | 6 | 47 | 9 | 1 | 0 | 2 | 2 | 1 |

\* UE stages = actor + wait before input + input build + encode/send.

**Join-aware arithmetic attribution.** This is not causal. It counts late decisions that would be ≤170 ms if one stage
were at its on-time median. The counts overlap, so they are not additive.

| Stage at its on-time median | V11 (of 225) | demo_03 (of 397) |
|---|---:|---:|
| Edge compute | 97 | 139 |
| UE encode/send | 75 | 135 |
| Uplink | 70 | 114 |
| Edge send → tunnel | 20 | 36 |
| Downlink set to **zero** | 85 | 142 |

Even an instant downlink would rescue only about one third of late decisions. Many misses are only a few ms past the
deadline, so removing about 4 ms anywhere has a similar effect.

### 3. The extreme tail (26 + 26 decisions >200 ms)

- **UE-process stalls of 130–250 ms.** 11 decisions in V11 and 8 in demo_03 had UE GT-freeze stalls, all of them late.
  - In each case, the UE freeze thread (`freeze_start → frozen`) and the prediction path stretch by the same amount.
    The prediction path is stretched either by waiting before `ue.input_7ch` or by the input span itself (for example
    input 203 ms against freeze 209 ms). These are two different UE threads.
  - Because of this, the GT branch is late too. These stalls delay **both** inputs, not one stage.
  - The CARLA world tick during these stalls is normal (P50 26–28 ms).
  - Inter-arrival of the stalls: 43 s (V11) and 34 s (demo_03) median, irregular (P95 70–85 s).
  - The **mechanism is unresolved.** There is no UE-process GC trace and no per-thread CPU or scheduling record. The
    shape (several threads pausing together) suggests a process-wide pause, such as GIL hold, GC or deschedule. This
    is an unverified hypothesis.
- **Moderate UE input/encode stalls of 60–95 ms**, without a freeze stall. These occur mainly in demo_03.
- **GT publication late** (V11: 6 extreme, 19 late in total).
  - For 5 of the 6 extreme GT-late decisions, prep FREEZE CPU is 140–198 ms, against 74 ms on time. The sixth (d1684,
    78.7 ms CPU) received its UE freeze late (45 ms). Examples: d0 is the first,
    cold decision; d1055 and d1060 occur at about 261 s.
  - There was no in-request prep gen2 GC (0 in both captures). The cause is scene- and history-dependent GT
    preparation compute.
- **Score-worker gen2 GC.** demo_03 d1395 had a 75.4-ms gen2 GC inside a PREDICTION op. This was the only score-worker
  gen2 in that run; V11 had 0.
- **UE result-thread receive stalls.** The interval from tunnel to app was 158 and 163 ms (V11 d1213 and d1615), and
  152 and 60 ms (demo_03 d1028 and d81).
  - Three of these overlap a 130–227-ms `carla.world_tick` on the UE MainThread. A concurrently stretched
    `ue.scene_snapshot` was also seen.
  - So the feedback had **already arrived in the UE kernel** and waited for the UE application.
- **Downlink-path outliers** (demo_03 only): d1049 at 35.6 ms, d338 at 27.8 ms and d556 at 26.5 ms.
  - The time is entirely **before** the DL DCI: edge→tunnel minus DCI→tunnel = 35.3 / 27.5 / 26.2 ms. DCI → tunnel
    was 0.3 ms.
  - Round 0, with one DCI in the window.
  - Two of these changed the outcome: d1049 at 212.6 ms and d556 at 176.9 ms. In V11 the maximum edge-send → tunnel
    was 10.3 ms.

### 4. Periodicity and associations

These are associations, not proof.

- **Periodicity.** The late-indicator autocorrelation is small and positive at almost every lag from 1 to 30
  (0.03–0.10). There is no isolated peak, which suggests slow non-stationarity rather than a fixed period. Late
  fraction by decision index mod 12 shows no consistent pattern. The earlier "every 12th decision" GT-GC stall is
  absent: there were 0 in-request prep gen2 collections in both captures.
- **Route segment.** Minute 4 is the worst full minute in both captures: on time 0.80 (V11) and 0.585 (demo_03). V11's
  partial final minute (n = 96) was 0.76. In that
  minute GT-published P50 rises to about 135 ms, close to prediction.
- **Rank correlation with receipt:**
  - payload 0.37 / 0.35
  - prep FREEZE CPU 0.27 / 0.19
  - UE freeze duration 0.28 / 0.16
  - edge send → tunnel 0.04 / 0.04
- **Channel.** The channel was constant at GOOD_ALL. The prior UL MCS state was 27–28.
- **No separate DL load signal.** The carrier DL DCIs were mostly MCS 25–28 on 5 PRB. All were round 0, with no
  retransmissions anywhere in the run.

### 5. demo_03 against V11

This comparison is descriptive and not attributed. demo_03's on-time edge compute median is 52.2 ms against 48.3 ms,
and UE encode/send is 13.8 against 12.0 ms. That moves more decisions past the ~146-ms cutoff, which is why the run has
371 marginal-late decisions against 199. The off-host viewing with an on-host viewer server is a possible contributor
but was not measured.

## Scope notes

- **Map ACK vs policy feedback.** These are different messages and were not merged here. Map installation does not
  wait for policy feedback: per the reports, 397 (demo_03) and 225 (V11) late-feedback frames were still installed.
- **Not covered:** the mechanism of the UE stalls and the internals of the core/gNB-DL interval. A future
  investigation of the UE stalls would need UE-process `gc.callbacks` and per-thread CPU/run-queue sampling. That is
  outside this question; it is not proposed or implemented here.

## Files (this directory)

- `DECISIONS_JOINED.csv` — one row per decision, 3,272 rows. Contains:
  - all instants (ms after action-open)
  - the 14 partition stages and sub-intervals (send→DCI, DCI→tunnel, post-send→tunnel)
  - DCI fields
  - GC, queue and overlap associations
  - group, spike class and outcome fields
- `SUMMARY.json` — counts, clock-bridge validation, per-group P50/P95/P99 for each stage (P99 only for n ≥ 100),
  attribution counts, periodicity, by-minute table, input SHA-256 values and the representatives list.
- `01_stage_contributions.png` — mean stage contributions per group. Means are additive, so the bars sum to mean
  receipt. Also shows the excess of each stage over on time.
- `02_representative_timeline.png` — 15 decisions on one clock, with the concurrent GT branch and UE freeze shown.
- Code: `rl_agent/splitfusion_single_ue_feedback_latency_v1/` (`sources.py`, `diagnose.py`, `plots.py`). It reuses
  `splitfusion_run5_full_live_analysis_v1.report.rows`. Rerun with:

  `env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 python3 -B -m rl_agent.splitfusion_single_ue_feedback_latency_v1.diagnose --capture V11_GOOD_ALL=<root>=experiments/splitfusion_m3_route_b_v11/analysis_01/METRICS.json --capture V20_DEMO_03=<root>=experiments/splitfusion_m3_route_b_v20_demo_03/metrics_01/METRICS.json --out <new dir>`
