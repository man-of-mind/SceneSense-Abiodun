# M4 radio clock bottleneck diagnosis

The limiting stage is **RFsim channel processing in both UE radio threads**, not a peer that is idle or mainly waiting to be scheduled. Each UE consumes approximately one CPU while processing available downlink samples. The gNB then waits for those UEs to supply uplink samples. This identifies the execution bottleneck; it does not yet isolate the instruction-level reason channel processing is so expensive.

Exactly one unchanged 20-s simulation-budget smoke ran: deployment capture `smoke_live_20261013_09`. Both functional gates passed, with 208 frames sent per UE. No performance qualification, tuning or repair is claimed.

## Measured bottleneck and peer dependency

The common first-to-last-send window is 41.733 wall seconds. Each principal thread has 84 snapshots spanning approximately 41.500 seconds inside it. CPU seconds come from `/proc` and native thread CPU clocks; waits are instrumented wall durations, not inferred from sleeping states.

| Thread | PID / TID | Total CPU s | Channel CPU s | epoll wait s | Runnable delay s | Radio s per wall s |
|---|---|---:|---:|---:|---:|---:|
| UE0 `UEthread_0` | 2391900 / 2392034 | 39.61 | 36.965 | 0.267 | 1.467 | 0.10789 |
| UE1 `UEthread_1` | 2391900 / 2392036 | 39.37 | 36.945 | 1.432 | 0.570 | 0.10783 |
| gNB `ru_thread` | 2391483 / 2391692 | 10.60 | 6.830 across two peers | 30.416 | 0.009 | 0.10788 |

UE channel processing accounts for 89.1%/89.0% of each thread's observation window and 93.3%/93.8% of its total CPU. Runnable delay is only 3.5%/1.4%. The gNB spends 73.3% of its window in epoll and 23.2% processing available samples. Thus its waits are downstream of busy UE peers, not evidence that the gNB scheduler or socket receive path is the primary bottleneck. Both UEs occupy one process; helper/tracer forks are not additional UEs.

All three threads retain CPU mask `0-3,8-19`, SCHED_RR priority 97. They migrate within that mask; the mask alone does not establish contention. No effective CPU quota or throttling was observed. Per-thread scheduler-delay counters advance despite the saved global `sched_schedstats=0`; the table reports those actual deltas. This is saturation of serial UE work, not exhaustion of all 16 allowed radio CPUs. Major-fault deltas are zero for these threads.

gNB port 4043 receives from UE0 port 36238, RU0, RNTI 22080, and UE1 port 36246, RU1, RNTI 1647. Its first-insufficient-peer loop attributes 49,265 callbacks / 24.717 s to UE0 and 15,483 / 6.438 s to UE1. Both callback P99s fall in the 2–5-ms bin; since-start maxima are 6.483/3.178 ms. These are individual polling callbacks, not uninterrupted stall episodes. The scan selects the first insufficient peer, so the larger UE0 total does **not** prove UE0 uniquely causes the stall. Both peers' sample clocks advance together while both compute.

`READ` contains processing and waits; channel work is inside processing; epoll is inside the wait callbacks. These durations and concurrent UE/gNB times must not be added into a sequential total. Counts include calls completed between snapshots; in-flight boundary calls are identified in the data. [Thread](analysis_01/THREADS.csv), [process](analysis_01/PROCESSES.csv), [peer](analysis_01/PEERS.csv) and [stage](analysis_01/NATIVE_STAGES.csv) tables retain all measured roles, tails and counters.

## Clock and backlog cross check

The retained clock calculation reproduces 0.08538/0.08527 radio-s/wall-s, with no backwards frame-slot or wall-clock steps. Its explicit SFN-wrap fixture passes. In this capture, independent RLC frame/slot clocks give 0.10769/0.10766, agreeing with native sample-counter progress at 122.88 Msamples/s to within the sampling-window differences. This is approximately 9.3 wall seconds per radio second.

| Send-window quantity | UE0 | UE1 |
|---|---:|---:|
| Sent frames | 208 | 208 |
| Offered / received feature chunks, Mb/s | 7.631 / 7.368 | 8.328 / 7.935 |
| RLC queue increase, MB | 1.186 | 1.751 |
| Payload P50 / P99, kB | 171.4 / 318.1 | 183.0 / 324.5 |
| Last send to completing kernel datagram P50 / P99, ms | 1514.8 / 4844.3 | 2645.8 / 5144.5 |
| Frames with that kernel endpoint | 206 | 205 |
| Measured PUSCH median, dB | 20.5 | 21.0 |
| Measured PUSCH samples | 2230 | 2277 |

Kernel-endpoint measurements include stale frames; missing endpoints stay unknown. Grants remain approximately balanced: 2231/2277, with only 8/9 retransmission grants. PUSCH samples are capture-window and RNTI bound. Commanded SNR remains 23.5 dB with noise −12. The existing noise-summation explanation was not isolated here.

The new clock is faster than the previous smoke's 0.0853, but configuration equality does not make two sessions a controlled before/after comparison. Different payloads, backlog trajectories, host execution and instrumentation preclude an improvement claim. The same applies to the retained unmatched single-UE 0.663 reference.

## One proposed controlled repair

**Make Gaussian RNG state private to each RFsim UE or RU**, preserving the generator algorithm and configured noise distribution. Do not implement it as part of this task.

The measured `rxAddInput` stage calls `gaussZiggurat` twice per complex output sample. Its qualified source uses shared writable `jsr`, `hz` and `iz`, with additional static state in `nfix`. Disassembly of the actual UE executable confirms non-TLS global writes. Both UE threads execute this path concurrently in one address space, exposing an unsynchronised shared-state/data-race hazard and a plausible cache-coherence cost. [Source excerpts, hashes and peer reconciliation](SOURCE_EVIDENCE.json) and [binary disassembly](UE_gaussZiggurat_disassembly.txt) bind this inference to the deployed code.

The fraction of UE channel CPU caused by RNG versus convolution, rounding or memory work is **not measured**. Supplemental perf attached early to the gNB and its tracer fork, missing the UE process; its [940 capture-window samples](PERF_GNB_ONLY.txt) must not be presented as UE attribution. No retry followed. The proposed private-state change is therefore a narrowly justified controlled repair candidate, not a proven speedup. Its risks are altered noise sequences, seed ownership and cross-UE correlation; a later authorized check must preserve noise power/distribution and use stable per-RU seed ownership while comparing channel CPU and radio progress. No admission cap is proposed.

## Instrumentation limits and preservation

Only a separately loaded RFsim library was built. Qualified executables, the original shared library, channel-processing object and production sources remain hash-identical. Mandatory source/slim-startup/host checks passed; all 720 preparation sources and the diagnostic binding still validate. Instrumentation uses bounded 4736-byte single-writer records, with ten records observed, no unstable snapshots and no synchronous per-sample logging.

The external 500-ms sampler ran on CPU22 at nice19, outside protected radio/inference CPUs but sharing front/map capacity. It used 4.335 CPU s over 110.002 s overall. Capture-window sample bodies used 1.725 CPU s, or 4.13% of one CPU; that subtotal excludes cgroup/flush bookkeeping. A 305-ns empty-scope microcheck projects 0.162 s over 531,153 scopes, **not** a bound on total native instrumentation overhead. Peer bookkeeping, cache perturbation and perf overhead were not isolated. [Overhead details](OVERHEAD_AND_SCHEDULING.json).

The native IPv4-only address decoder cannot decode the gNB's IPv4-mapped IPv6 sockets. Those address strings are ignored; unique reversed TCP ports, PID/RU ownership, source activation and the original connection log reconcile both peers. No source attribution depends on the invalid strings.

A create-only prelaunch bookkeeping refusal occurred before any live resource or smoke invocation; its original record is preserved. There was one actual live attempt and no replacement. Owned teardown and postflight passed; the ten pre-existing core containers are stopped again. Firefox, both UI servers, MPS and remote desktop retain their original process identities. No exclusive GPU ownership is claimed. Final host check found no live radio/CARLA process or running container and a free experiment lock. No evidence was deleted or archived; no commit/push occurred. Capture allocation is 601.0 MB and diagnosis approximately 37.1 MB before final metadata. Remaining disk is **25.28 GB / 23.54 GiB**.

## Artifacts and reproduction

The two 3840×2160 PNGs are [radio progress and backlog](analysis_01/01_RADIO_PROGRESS_BACKLOG.png) and [CPU and peer waits](analysis_01/02_CPU_PEER_WAIT.png). Joined frame, thread, native-stage and RLC data, identities and clock uncertainty are in [analysis_01](analysis_01). Build commands are in [BUILD.json](BUILD.json); the already-executed live command is in [smoke_launch.json](smoke_launch.json), not a request to run again. Registration, diagnostic source and qualified binary hashes are in [FINAL_CHECK.json](FINAL_CHECK.json).

Reproduce saved-data analysis only, from the working repository, with a fresh output path:

```bash
env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 -B experiments/splitfusion_m4_two_ue_v1/radio_clock_diagnostic_20261013_01/analyze.py \
  /tmp/m4_radio_clock_reproduction_01
```

This uses CPU22/nice19 and refuses a detected live capture. It does not start a service, compile code or run an experiment.
