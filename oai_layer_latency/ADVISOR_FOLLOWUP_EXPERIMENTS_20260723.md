# Advisor follow-up experiments: OAI UL MCS / QoS / burst-shape investigation

Date: 2026-07-23

Purpose: answer three follow-up questions raised after the first OAI latency
localization:

1. Can a CARLA-shaped non-CARLA UDP stream reproduce the low-MCS/RLC-queue
   behavior?
2. Is the low MCS caused by `nr_ue_max_mcs_min_rb()` reducing a high selected
   MCS after SNR/BLER selection?
3. Is the split-feature traffic being mapped to a special QoS/5QI/DRB class that
   would justify a lower PER target and therefore lower MCS?

## Result in one line

The bottleneck is reproduced by **large split-feature-sized bursts at the
observed closed-loop cadence**, but the direct scheduler trace shows the low MCS
is already selected upstream of the PHR helper. Runtime bearer evidence shows
the traffic uses the default data bearer (`QFI 1`, `5QI 9`, `DRB 1`, `LCID 4`,
`LCG 1`), not a special split-inference traffic class.

## Experiment 1: CARLA-shaped UDP burst, with and without CARLA-observed pacing

Traffic generator: `oai_layer_latency/carla_shaped_udp_burst_sender.py`.

Runner: `oai_layer_latency/run_carla_shaped_udp_burst_273prb.sh`.

Both use 273PRB RFsim, adaptive MCS, zstd-like full no-AE frame size
(`1,079,400 B`, 18 UDP chunks at 60 KB/chunk).

| Run | Offered pattern | Offered app rate | MCS p50 / p95 | RAN UL p50 / p95 | RLC mean queue | LCID4 p95 occupancy | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| `carla_shape_udp_bw273_20260723_165614` | open-loop 10 FPS | 83.2 Mbps | 23 / 25 | 20.7 / 34.1 ms | 17.4 ms | 987.3 KB | OAI ramps MCS high when backlog is continuously fed. |
| `carla_shape_udp_bw273_observedpace_20260723_170132` | observed closed-loop pace (~1.2 FPS) | 10.0 Mbps | 4 / 8 | 118.4 / 169.1 ms | 104.4 ms | 921.9 KB | Reproduces CARLA low-MCS/RLC-wait behavior without CARLA/model compute. |
| `carla_shape_udp_bw273_mcsdecision_20260723_171153` | observed closed-loop pace + MCS-decision trace | 10.0 Mbps | 4 / 8 | 118.9 / 169.9 ms | 104.4 ms | 922.9 KB | Same behavior; includes pre/post scheduler trace. |

Takeaway: “large BSR” alone is not enough. The same 1 MB burst size under
continuous open-loop pressure ramps to high MCS. The low-MCS behavior appears
when the traffic has the sparse/closed-loop cadence that the CARLA deployment
actually creates.

## Experiment 2: direct pre/post MCS decision trace

Instrumentation added:

- `GNB_MAC_UL_MCS_DECISION` in `T_messages.txt`.
- Trace site in `gNB_scheduler_ulsch.c`, bracketing:
  selected MCS → `nr_ue_max_mcs_min_rb()` → optional fixed-MCS override →
  final RB/TBS allocation.
- T-tracer `latency` gNB profile now extracts this event.

Run group:
`carla_shape_udp_bw273_mcsdecision_20260723_171153`.

Key result from
`../metrics_logs/scenesense_ttracer/carla_shape_udp_bw273_mcsdecision_20260723_171153/gnb/analysis/mcs_decision_summary.md`:

| Metric | Value |
|---|---:|
| scheduler rows with data queued | 79,806 |
| gNB SNR p50 | 50.3 dB |
| selected MCS p50 / p95 | 4 / 8 |
| pre-PHR MCS p50 / p95 | 4 / 8 |
| post-PHR MCS p50 / p95 | 4 / 8 |
| final MCS p50 / p95 | 4 / 8 |
| rows where `nr_ue_max_mcs_min_rb()` reduced MCS | 0 |
| rows where final MCS differed from selected MCS | 0 |

Interpretation: the prior PHR-helper explanation is not supported by this
direct trace. In this run, MCS is already low at the initial MCS-selection
stage; the PHR-normalized helper does not reduce it.

Likely next source-code lead: with HARQ enabled (`ul_harq_round_max` default 4),
`pf_ul()` uses `get_mcs_from_bler()`, not the direct `get_mcs_from_SINRx10()`
path. The BLER/OLLA helper updates every 10 radio frames and decreases MCS when
the scheduled count in the window is too small. That is consistent with the
observed cadence sensitivity: open-loop 10 FPS keeps the scheduler active and
ramps MCS upward, while sparse closed-loop bursts keep it in a low-MCS state.

Presentation-safe plots:

- `../metrics_logs/scenesense_ttracer/carla_shape_udp_bw273_mcsdecision_20260723_171153/gnb/analysis/mcs_decision_timeseries.pdf`
- `../metrics_logs/scenesense_ttracer/carla_shape_udp_bw273_mcsdecision_20260723_171153/gnb/analysis/mcs_decision_phr_drop_hist.pdf`

## Experiment 3: QFI/5QI/DRB/LCID mapping sanity check

Static config:

- UE requests DNN `oai` and NSSAI SST `1`:
  `openairinterface5g/targets/PROJECTS/GENERIC-NR-5GC/CONF/ue.conf`.
- Current SMF YAML config maps DNN `oai` to `5qi: 9`, `session_ambr_ul:
  10Gbps`, `session_ambr_dl: 10Gbps`.
- SMF startup logs also report `5qi: 9`, AMBR 10Gbps, and `enable_qers: No`.
- The SQL seed still contains `5qi: 6` for DNN `oai`, but the runtime RAN log
  used `5QI 9`; for this run, the runtime evidence is the source of truth.

Runtime RAN evidence from the mcsdecision run:

- UE log: requested PDU session ID 1, DNN `oai`, NSSAI `1.ffffff`.
- gNB log: received PDU Session Resource Setup Request.
- gNB log: added QoS flow `QFI=1`.
- gNB log: created/assigned `DRB 1` for `QFI 1 (5QI 9)`.
- UE/gNB logs: SDAP default DRB is `DRB 1`; PDCP/RLC added `DRB 1`.

Runtime trace evidence:

| Trace | Finding |
|---|---|
| `GNB_MAC_LCID_UL` | all received UL data on `LCID 4` (`1.410 Gb` logged, i.e. `176.3 MB`; this OAI trace field is bits because the source logs `mac_len * 8`) |
| `NRUE_MAC_RLC_BUFFER_STATUS` | only `LCID 4 / LCG 1` has nonzero data-buffer occupancy; max `1,099,470 B` |
| `NRUE_MAC_BSR_STATUS` | only `LCG 1` reports nonzero backlog; max `1,099,470 B` |

Interpretation: current evidence does **not** support a special traffic-class
mapping for split tensors. iperf/CARLA/synthetic payloads are carried on the
normal data bearer path, and this OAI setup does not appear to activate per-flow
QER treatment for these packets.

## Updated mechanism hypothesis

What is now well supported:

- The deployed uplink latency is dominated by UE RLC queue-wait.
- Queue-wait is driven by low adaptive UL MCS under closed-loop burst cadence.
- Payload reduction helps because it reduces how much data sits behind those
  low-MCS grants.
- Fixed MCS28 collapses the queue, proving MCS is the key lever.

What is **not** supported after the direct trace:

- The claim that `nr_ue_max_mcs_min_rb()` is reducing a high selected MCS to QPSK
  in this workload.
- The claim that split-feature packets are mapped to a special QoS/5QI class.

## Experiment 4: direct BLER/OLLA selector trace

Instrumentation added:

- `GNB_MAC_BLER_MCS_DECISION` in `T_messages.txt`.
- Trace site inside `get_mcs_from_bler()`.
- Branch codes:
  - `0`: no update yet (`diff < BLER_UPDATE_FRAME`);
  - `1`: increase MCS because BLER is below the lower threshold and enough
    samples were scheduled;
  - `2`: decrease MCS because BLER is above the upper threshold;
  - `3`: decrease/hold-low because too few samples were scheduled
    (`num_sched <= 3`);
  - `4`: hold MCS inside the BLER target window.

Runs:

| Run | Traffic pattern | MCS p50 / p95 | `num_sched` p50 | Increase branch | Few-samples branch |
|---|---|---:|---:|---:|---:|
| `carla_shape_udp_bw273_blertrace_observed_20260723_173640` | observed closed-loop pace (~1.2 FPS) | 4 / 8 | 1.0 | 21.4% | 78.6% |
| `carla_shape_udp_bw273_blertrace_openloop_20260723_174042` | open-loop 10 FPS | 23 / 25 | 3.5 | 50.0% | 50.0% |

Interpretation:

- The low-MCS state is produced inside the BLER/OLLA selector itself, upstream
  of `nr_ue_max_mcs_min_rb()`.
- The high-BLER decrease branch was `0%` in both runs, so this is not a real
  RF channel/PER failure in the RFsim setup.
- The sparse observed cadence spends most update decisions in the
  `num_sched <= 3` branch, which walks MCS back down between bursts.
- The open-loop 10 FPS stream gives enough dense scheduling windows for the
  low-BLER increase branch to keep MCS high.

This is the cleanest current explanation for the advisor puzzle: iperf/open-loop
traffic can reach high MCS on the same RFsim path, while CARLA-like closed-loop
split tensors remain near QPSK because the traffic cadence interacts badly with
OAI's BLER/OLLA update logic.

Presentation-safe plots:

- `../oai_layer_latency/plots/bler_olla_branch_comparison.pdf`
- `../oai_layer_latency/plots/bler_olla_mcs_timeseries.pdf`
- `../oai_layer_latency/plots/bler_olla_num_sched_timeseries.pdf`
