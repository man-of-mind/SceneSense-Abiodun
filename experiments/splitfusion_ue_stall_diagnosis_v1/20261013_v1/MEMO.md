# UE processing/receive stalls: what the existing evidence can and cannot narrow

13 Oct 2026. Offline, read-only, CPU only. This continues `../../splitfusion_single_ue_feedback_latency_v1/20261013_v1/MEMO.md`.
It uses the same two captures: V11_GOOD_ALL (accepted reference) and V20_DEMO_03 (a demonstration, not a controlled comparison).
The cases are the ones already identified there; no new search was made.
Evidence: `CASES.csv` / `CASES.json` here (51 cases), code `rl_agent/splitfusion_ue_stall_diagnosis_v1/cases.py`.
No plots, no runtime changes.

## Supported findings

1. **Large UE stalls before send (17 of 38 category-A cases; 100–250 ms) are process-wide CPU-bound GIL monopolisation.**
   - In each of these cases the UE `gt-freeze` thread's own CPU time is **0.83–0.94 of its wall time**. For example:

     | Decision | Freeze wall | Freeze thread CPU |
     |---|---:|---:|
     | V11 d1748 | 263.5 ms | 239.0 ms |
     | demo_03 d1310 | 242.2 ms | 214.4 ms |

     A normal freeze uses about 12 ms of CPU. The CPU figure is `time.thread_time_ns()` from `freeze_start` to seal, recorded in each FREEZE message (`FRAMES.jsonl` → `freeze_stats`).
   - Over the same window, **no other UE thread produces any record**:
     - no `profiler-flush` gauge (it normally samples every 100 ms; gaps here are 110–302 ms)
     - no CARLA sensor callback
     - no MainThread tick start (normally every ~125 ms)
     - no GT-build span
   - The front thread is stopped at a point where only cheap Python remains. The OPEN/PLAN messages had already reached the service (e.g. 7.9 ms), and `_run4_on_planned` only opens a gate.
   - **Excluded:**
     - Freeze-thread blocking or descheduling, because it was on CPU.
     - Process-wide descheduling, because one thread ran.
     - A shared lock: the freeze thread takes no lock that the front or sampler threads need. `_send_lock` is used only after seal; the gauges are lock-free reads.
   - **What remains:** one thread doing ~120–230 ms of extra CPU work while holding the GIL inside `freeze_codec.freeze`.
   - Inside `freeze`, SHA-256 of large buffers releases the GIL; the per-actor Python/CARLA-binding loop and object allocation do not.
2. **`world.tick()` releases the GIL.** During the 13,322 ordinary ticks (<100 ms), other UE threads log at their normal rate (about 380 records/s). Over all ticks, 68,915 records fall inside ticks against an expected 66,996 at the 0.41 duty cycle (V11); 60,918 against 57,181 (demo_03).
   - So the long `carla.world_tick` spans (218–227 ms) seen with receive stalls are **not a CARLA call holding the GIL**.
   - Inside ticks of 100 ms or more, the record rate collapses from about 380/s to 40–126/s. The MainThread could only return from the tick once it got the GIL back.
3. **Receive stalls (category C, 4 cases) are the same process-wide silence, but the busy thread is not identified.**
   - The datagram reached `oaitun_ue1` and then 60–163 ms passed with no UE record at all. The gauge gaps were 162–304 ms.
   - The receipt stamp follows within 2–7 ms of the end of the silence. Receipt is stamped immediately after `recvfrom`, before any lock, so `_engine_lock` is not involved.
   - No GT freeze overlaps any C silence, so the CPU-burning freeze thread is not the cause there.
4. **Category B (GT/service, 9 cases) splits three ways:**
   - **GT prep was CPU-bound.** In 5 V11 cases, prep FREEZE CPU/wall was 139–198 / 143–200 ms, against about 74 ms normally. There was no in-request gen2 GC.
   - **A garbage-collection pause is directly demonstrated in one case.** demo_03 d1395 had a score-worker gen2 GC of 75.4 ms wall / 66.6 ms CPU inside its PREDICTION op (op 86.3 ms wall).
   - **2 cases have off-CPU time** (prep wall 120–128 ms against CPU 79–80 ms). The 41–48 ms gap is time off CPU, and its cause is not identified.
   - The remaining case, the 45-ms result return in V11 d1680, has no records inside it.
5. **The moderate UE stalls (21 cases, 35–95 ms in actor, input or encode) are thread-local, not process-wide.**
   - Other threads keep logging, and gauge gaps stay at about 100–118 ms. This includes two cases where the freeze was long but CPU-light: d787 at 0.19 and d1231 at 0.13 CPU/wall.
   - CUDA-event spans are about equal to wall time (e.g. 68.0 / 68.1 ms). Events enqueued by a slow CPU would show the same pattern, so this does **not** separate GPU waiting from CPU slowness.

## Unresolved mechanisms

| Category | What is known | What is not known |
|---|---|---|
| A large (17) | The freeze thread holds the GIL while on CPU for 120–230 ms more than normal | Which kind of CPU work it is (see below) |
| C (4) | A process-wide GIL silence; not the freeze thread; not `world.tick` | Which thread held the GIL, and what it was doing |
| A moderate (21) and B off-CPU (2) | The delay is local to one thread or worker | GPU queueing, CPU preemption (the UE child is unpinned), or lock/IO wait |
| B CPU-bound (5) | GT prep compute rose to 140–200 ms | Which prep step grew (not profiled per step) |

The kind of CPU work in category A could be one of three things:

- **CPython cyclic GC (gen2).** It runs in the allocating thread and counts as that thread's CPU. It is consistent with the irregular 34–43-s spacing and with GC pauses already seen in the GT workers.
- **Kernel time in page faults or direct reclaim** during allocation. `thread_time` includes system time.
- **A long C-level call that does not release the GIL.**

The records cannot distinguish these: there is no UE GC trace, no user/system split, no fault counts and no per-thread CPU for any other UE thread. A Python-level sampler inside the UE cannot fill this gap, because it is itself silenced while the GIL is held.

## Case table (representative; all 51 rows in `CASES.csv`)

All times are ms after UE action-open, with `CLOCK_MONOTONIC_RAW`. Sources:

- `FRAMES.jsonl` (freeze_stats, GT requests)
- `profiler/ue.profile.jsonl` (spans, marks, gauges, threads)
- `gt_gc_trace/*_gc_trace.json`

| Category | Capture / decision | Expanded interval | Window | Freeze wall / CPU | Other UE threads inside window | Reading |
|---|---|---|---:|---:|---|---|
| A large | V11 d1748 | wait before input | 7.8–257.5 | 263.5 / 239.0 | none; gauge gap 258 | freeze thread holds GIL, CPU-bound |
| A large | V11 d684 | wait before input | 3.7–166.6 | 182.3 / 161.2 | none; gauge gap 259 | same |
| A large | demo_03 d1310 | input build | 7.7–248.2 | 242.2 / 214.4 | none until 233; gauge gap 302 | same |
| A large | demo_03 d1168 | wait before input | 5.7–213.1 | 217.5 / 199.6 | none; gauge gap 263 | same |
| A moderate | demo_03 d885 | input build | 13.8–107.7 | 28.8 / 11.7 | MainThread ×3, callback, GT build | thread-local; GPU vs CPU not separable |
| A moderate | V11 d787 | input build | 13.7–72.0 | 67.6 / 13.0 | callback, MainThread | freeze off CPU too; cause not identified |
| B | V11 d1060 | extra wait for GT | – | prep FREEZE 200.2 / 198.2 | – | GT prep CPU-bound |
| B | demo_03 d1395 | scoring op | – | score op 86.3 / 74.8 | – | **score gen2 GC 75.4 ms (demonstrated)** |
| B | V11 d1684 | extra wait for GT | – | prep FREEZE 120.6 / 78.7 | – | prep off CPU ~42 ms; not identified |
| C | V11 d1213 | tunnel → app | 162.6–321.0 | 18.8 / 11.6 | none until 314 (tick end); gauge gap 218 | process-wide silence; holder unknown |
| C | V11 d1615 | tunnel → app | 161.6–324.3 | 25.1 / 12.3 | none; gauge gap 304 | same, with no long tick recorded |
| C | demo_03 d1028 | tunnel → app | 165.0–317.4 | 17.1 / 11.4 | none until 315 (tick end); gauge gap 274 | same |

## Actionable defect?

The only directly demonstrated mechanism with a narrow repair is the **score-worker in-request gen2 GC**:

- One event of 75 ms in 3,272 decisions; it changed one decision's outcome.
- The adopted GC policy is installed only in the prep worker; the score worker runs stock GC (`gc_policy: control` in `score_gc_trace.json`).
- The smallest repair would be to apply the same already-adopted policy to the score worker under a new registration.

It is not recommended now because the impact is minimal. The large UE stalls have **no** demonstrated mechanism, so no UE repair is proposed.

## ONE recommended next action

In the next *separately approved* single-UE capture (no extra run just for this), add **record-only** UE telemetry for exactly the missing boundaries. GC, affinity, priority and deadlines stay unchanged.

1. **UE-child `gc.callbacks`.** Record start/stop raw time, generation, objects collected, and the triggering thread name. Use the existing GT-worker `worker_trace` gen2 format and the existing `ue_substitution` seam, with policy `control` (record only).
2. **An out-of-process per-thread sampler of the UE child.** Read `/proc/<ue_pid>/task/*/stat` and `schedstat` every 20 ms: utime, stime, minflt, majflt, state and run-queue delay per tid. Run it in the runner parent or a tiny process on the radio CPU set. It must be outside the UE process, because a Python sampler inside it is silenced exactly when the GIL is held.

How to read the result for each stall window:

| Observation | Conclusion |
|---|---|
| A gen2 callback spans the silence in the CPU-burning thread | **GC** |
| No GC, but stime and faults surge | **Kernel memory management** |
| No GC, utime dominates in one thread | **Non-GIL-releasing C call**; py-spy on the next run would then be justified |
| All threads off CPU with growing run-queue delay | **Descheduling / contention** |

The same sampler names the GIL holder for the category-C silences and separates CPU from wait for the moderate stalls.
