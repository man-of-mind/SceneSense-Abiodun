# Run-4 1,000-frame profiling capture: preparation and analysis handoff

**Status: prepared, not executed.**

The capture runs the same single-host Phase-6 live path that qualified in
attempt 3, with the same instrumentation and the same frozen Run-4 actor. It is
an observation run. It changes no policy, optimizes nothing, and makes no
diagnosis.

## How the capture is bound

- **Registration:** `capture_run4_1000_v1.json` (`PROFILING_CAPTURE`).
- **Pins:** it pins three things by SHA-256 and ancestry:
  - the qualifying result `QUALIFICATION_ATTEMPT_3_RESULT.json`;
  - addendum 2;
  - qualified commit `59f40d1`.
- **Instrumentation unchanged:** the runner refuses to launch if any
  instrumentation file has changed since `59f40d1`. The files are recorder,
  sampler, NVML, identity, stages, seams, edge, map, child and analyzer.
- **Unchanged from the qualification:**
  - actor `d064013d…` and the 21-D state;
  - reward formula and the inclusive 170-ms clock, with Q_perc evaluation and
    UE receipt inside it;
  - k_min = 2;
  - FAVORABLE_STABLE, Route B, Epic rendering;
  - codec, edge, map and OAI configuration;
  - sampler interval of 100 ms.
- **The only lifecycle difference:** the route safety bound is 600 s and the
  child bound is 1,800 s, instead of 120 s and 1,200 s.
  - The collector's safety timeout is a **wall-clock** bound on the route
    loop.
  - At 120 s, a 1,000-frame route (about 220 s) would stop near frame 540.
  - These are upper bounds only. No reward, deadline, GT or edge timeout
    changes.
- **Gates:** G1–G9 as qualified. G1 accepts exactly one of two outcomes:
  - 1,000 frames at `TRANSMITTED_BUDGET_REACHED`;
  - exactly 999 frames at a verified `DECISION_CYCLE_BOUNDARY` (budget 1,000,
    no decision cap).

  A gate failure makes the capture `CAPTURE_INVALID`. There is no retry.

## Launch (needs explicit authorization)

```bash
cd neu_collab/abiodun_run4_single_gpu_profiler_v1
env -u PYTHONPATH python3 -m rl_agent.splitfusion_single_gpu_profiler_v1.runner_v1 \
  --registration rl_agent/splitfusion_single_gpu_profiler_v1/capture_run4_1000_v1.json \
  --execute SINGLE_GPU_PROFILER_V1_RUN4_PROFILING_CAPTURE_1000_EXECUTE \
  --output-root rl_agent/experiments/splitfusion_single_gpu_profiler_v1/<UTC>_run4_profiling_capture_1000
```

Before launch, check that the host is cold:

- no CARLA, OAI or edge container is running;
- `/proc/loadavg` is quiet;
- there are no foreign I/O jobs (the preflight records these);
- the OAI tree is bind-mounted read-only into the worktree.

The previous attempt directory was about 318 MB for 300 frames, so expect about
1 GB for 1,000.

## What the analyses receive

After the gates pass, the runner writes `<output_root>/export_v1/`. It is
create-only and frozen, and it comes with a `MANIFEST.json` that holds the
source and export SHA-256 values.

| File | One row per |
|---|---|
| `frames.csv` | transmitted frame: identity, kind, payload, map/ticket outcome, and per registered stage the start, wall duration and CUDA duration |
| `spans.csv` | span, from all processes |
| `marks.csv` | mark: receipts, reassembly, queue, terminals |
| `gauges.csv` | queue-depth sample |
| `gpu.csv` | sampler tick: GPU and host |
| `procs.csv` | role process per tick |
| `threads.csv` | per-thread CPU delta per tick |
| `gpu_procs.csv` | per-process GPU memory or utilization sample |
| `tickets.csv` | reward ticket |

The full schema is in the `export_v1.py` docstring and in `PROFILER_V1_DESIGN.md`.

## Rules for the independent Claude and Codex analyses

1. **Same inputs.** Each analysis starts with
   `python3 -m rl_agent.splitfusion_single_gpu_profiler_v1.export_v1 <output_root> --verify-only`
   and records the manifest's `export_sha256`. If the hashes differ, the two
   analyses are not comparable.
2. **Independence.** Neither analysis reads the other's notes or conclusions
   before writing its own. Each writes down its questions and methods before
   computing results.
3. **Clocks.**
   - Subtract only `*_raw_ns` from `*_raw_ns`; that is the host-wide raw clock,
     verified per attempt.
   - `capture_wall_ns` is realtime and needs the per-process offsets in the
     manifest.
4. **What the measurements mean.**
   - `cuda_ms` is current-stream execution between two events; side-stream
     work appears only as wall time.
   - NVML utilization is a driver-sampled window, not an instantaneous value.
   - Profiler cost sits inside the spans it brackets. Attempt 3 measured about
     0.84 ms per frame on the UE frame worker; it is not subtracted.
5. **Not evidence for this capture.**
   - The 300-frame qualification (attempt 3) qualifies the instrumentation
     only.
   - Attempts 1 and 2 are preserved failures and must not be pooled.
6. **Host conditions to report.** The pre-existing MPS daemon
   (`nvidia-cuda-mps-control -d`, active since 2026-07-25) and any foreign
   processes the preflight records.
7. **Scope.** Bottleneck findings are inputs to a later, separately authorized
   decision. They change no policy and no pipeline here.
