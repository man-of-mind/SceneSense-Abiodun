# Single-GPU profiler v1: design and evidence schema

Instrumentation only. It is not an optimization and not a policy change. It
records CPU and GPU contention on the single-host Phase-6 live path at base
commit `6199d49`: CARLA, the UE, the actor, the edge, the evaluator, the map
and OAI, all running on the local RTX 5090.

## Scope and invariants

- **Additive.** Everything is in `rl_agent/splitfusion_single_gpu_profiler_v1/`.
  Every tracked file outside the package is byte-identical to `6199d49`.
  - The runner preflight checks this with `git diff --ignore-submodules=all`.
  - A test checks it too.
  - The OAI submodule is a read-only bind mount of the main checkout's built
    tree, the same as earlier worktrees.
- **Policy-neutral.** The profiler core does not name a policy:
  `recorder_v1`, `sampler_v1`, `nvml_v1`, `identity_v1`, `stages_v1`,
  `analyze_v1` and `child_v1` (enforced by a test).
  - The actor, the state schema and the checkpoint are read from the
    binding manifest named in the run configuration, and from the live
    actor object.
  - The only file that names Run-4 is `qualification_run4_300_v1.json`.
- **Pass-through seams.**
  - Every wrapper calls the original with the same arguments and returns its
    result unchanged.
  - Every exception from instrumented code is re-raised unchanged.
  - A failure inside the profiler (write error, full buffer, broken gauge,
    CUDA query error) is counted and swallowed. It never reaches the live
    path.
- **No synchronization.**
  - GPU time comes from a pair of `torch.cuda.Event(enable_timing=True)`
    recorded on the current stream.
  - A background thread resolves the pair with the non-blocking `query()`.
  - `synchronize` is never called, and the profiler never initializes CUDA.
- **Unchanged:**
  - actor, 21-D state, reward and the inclusive 170-ms deadline, k_min = 2;
  - FAVORABLE_STABLE, Route B, seeds, sensors, Epic rendering;
  - codec, edge model, map path, OAI;
  - precision, MPS, CUDA scheduling and timeouts.

## Where the seams attach

All seams are process-local and installed after the unchanged Phase-6 seams.

| Process | Entry point | What is wrapped |
|---|---|---|
| UE child | `child_v1` → `phase6_live_child_nobuild_v2.main` | See the UE rows of the stage table. Wrapped through module attributes the Phase-6 code resolves at call time, through instance attributes on the constructed runtime/engine/pipeline, and through the collector and `SamplingWorld` subclasses. |
| Edge (container) | `edge_profiled_v1` → `phase6_edge_runtime_v2.main` | `ChunkReassembler`, `LatestFramePendingSlot`, `Run4EdgeProcessorV2`, `StageRecorder` (Phase-6 edge and detached runtime), `publish_cpu`, tail `snapshot`, `build_run4_map_update`, `Run4MapPublisherV2`, `Run4EvaluatorV2` |
| Map (host) | `map_profiled_v1.py` → `phase6_map_server_v2.install` then the same `main` | ingest handler, locked install, install-feedback response |
| Sampler | `sampler_v1` (separate process, started by the runner before the cell and stopped after teardown) | read-only `/proc` and NVML (via `ctypes`; nothing installed) |

### Two worktree and startup details

- **Namespace repair.** `run_route_b` repairs the `pole_lraspp_multimodal_fusion`
  namespace path before it imports the route module. The UE seam imports that
  module earlier, at seam-install time, so it applies the identical,
  idempotent repair first.
- **Edge import order.** The edge seam imports `phase6_edge_runtime_v2` first,
  so the module's top-level imports run in their original order.

## Stage vocabulary (`stages_v1.STAGES`)

The registry records three things for each stage:

- its exact allowed parent on the same thread;
- its coverage denominator (`scope`);
- whether it carries CUDA-event time.

The CUDA column means the GPU execution time of the current stream between the
two events. It does not include the tail's side stream; that work's
synchronization shows up as wall time.

| Stage | Process | Parent | CUDA | Source |
|---|---|---|---|---|
| `carla.world_tick` | UE | root | | `world.tick()` call: server step and render |
| `carla.rgb_receipt` (mark) | UE | root | | RGB callback |
| `ue.worker_frame` | UE | root | | collector `_process_token` |
| `ue.sensor_wait`, `ue.radar_window`, `ue.radar_rasterize`, `ue.rgb_convert`, `ue.scene_snapshot`, `ue.submit` | UE | `ue.worker_frame` | | wrapped |
| `ue.si_p40`, `ue.udp_send` | UE | `ue.submit` | | Phase-6 stage stamps, re-emitted as derived spans (not re-measured) |
| `ue.plan` | UE | `ue.submit` | | guard, state, actor, hold, fallback |
| `agent.telemetry_snapshot`, `agent.state_build`, `agent.actor` | UE | `ue.plan` | | `agent.actor` carries the decision and the 21 features |
| `ue.input_7ch` | UE | `ue.submit` | yes | seven-channel preparation (H2D) |
| `ue.codec.{total_ue_preparation, front_backbone, ranker_selection, ae_encode, quantize_pack, zstd_compression}` | UE | registered `StageRecorder` nesting | yes | same names as the base `StageRecorder` |
| `ue.sfd4_pack` | UE | `ue.submit` | | |
| `ue.gt_build` | UE | root (GT worker) | | evaluation scaffolding (CARLA GT) |
| `ue.feedback_receipt`, `ue.terminal_receipt` (marks) | UE | root | | stamped with the authoritative receipt stamp |
| `edge.datagram_first`, `edge.reassembly_complete`, `edge.queue_offer`, `edge.queue_take` (marks) | edge | root | | |
| `edge.process` → `edge.verify`, `edge.compute`, `edge.map_update_build` | edge | | `compute` yes | identity bound after verify |
| `edge.codec.{total_edge_processing, zstd_decompression, unpack_dequantize, ae_decode, frozen_tail}` | edge | `edge.compute` | yes | |
| `edge.publish_cpu` → `edge.codec.output_serialization`, `edge.output_snapshot` | edge | `edge.compute` | yes | output construction |
| `edge.map_publish` | edge | root | | map publication (direct edge-to-map) |
| `edge.evaluator_submit` (mark), `edge.qperc_evaluate` → `edge.feedback_emit` | edge | evaluator thread | | Q_perc evaluator and R4FB emission |
| `edge.terminal_built` (mark) | edge | root | | stale and superseded terminals |
| `map.ingest_handle` → `map.install`, `map.respond` | map | | | map install, kept separate from publication |

## Evidence schema

All files are JSONL, written create-only (`open(..., "x")`) in batches by one
flush thread every 0.2 s. Every record carries a sequence number (`seq`). The
recorder files carry:

- the PID (in the header);
- the role;
- the native thread ID (`tid`);
- the phase (`ph`: `setup`, `prewarm` or `live`);
- whatever frame identity is bound: `frame_id`, `tensor_seq`, `session_uuid`,
  `decision_seq`, `ticket_seq`, `reward_requested`, `frame_kind`, `mode_id`
  and `q_e4`.

| File | Writer | Record kinds |
|---|---|---|
| `cells/<cell>/profiler/ue.profile.jsonl` | UE child | `header`, `thread`, `span`, `mark`, `cuda`, `gauge`, `clock`, `note`, `footer` |
| `cells/<cell>/profiler/edge/edge.profile.jsonl` | edge container | as above. Copied create-only out of the edge scratch before the scratch is deleted. |
| `cells/<cell>/profiler/map.profile.jsonl` | map | as above |
| `profiler/sampler.jsonl` | sampler | `header`, `sample`, `footer` |

### Recorder records

- **`span`:** `start_raw_ns`, `end_raw_ns`, `parent` (the `seq` of the
  enclosing span on the same thread), `depth`, `ok` (false if the
  instrumented code raised), `derived` (true for re-emitted stamps), and
  `cuda: "PENDING"` when a CUDA pair was recorded.
- **`cuda`:** `span_seq`, `gpu_ms`, and a `status`:
  - `RESOLVED`;
  - `UNRESOLVED_TIMEOUT` (more than 10 s);
  - `UNRESOLVED_AT_CLOSE`;
  - `QUERY_ERROR` or `ELAPSED_ERROR`.
- **`gauge` (every 100 ms):** queue depths:
  - UE: preparation slot, GT queue, open ticket, sensor stages, sent;
  - edge: pending slot, partial reassemblies, evaluator queue and
    outstanding tickets;
  - map: ingest queue.
- **`clock` (every 1 s):** a bracketed `raw`/`monotonic`/`realtime` triple.
- **`note`:** the policy runtime identity, the first state vector's
  identity, the policy identity at close, and the per-frame Phase-6 stage
  stamps (`frame`).
- **`footer`:**
  - counters: records, dropped, batches, write failure, internal errors,
    CUDA resolved/unresolved;
  - gauge errors;
  - `hot_path_ns_total` and `hot_path_calls`: profiler time spent on the
    instrumented threads;
  - writer CPU, bytes written, CUDA state.

### Sampler `sample` record

- `t_raw_ns`, `t_wall_ns`, `late_ns`, `cost_ns`, `cpu_ns`.
- `host`: load averages, `procs_running`/`procs_blocked`, CPU busy %.
- `gpu`:
  - utilization (GPU and memory controller);
  - used and total memory;
  - SM and memory clocks;
  - power, temperature, P-state.
- `gpu_proc_mem`: per-process compute and graphics memory.
- `gpu_proc_util`: per-process SM and memory utilization samples newer than
  the previous call.
- `procs[]`: for each role process, PID, role, CPU %, RSS, thread count,
  voluntary and involuntary context switches. For the UE, edge and map it
  also carries per-thread CPU deltas, with the container-namespace TID for
  the edge.

The sampler runs on an absolute `CLOCK_MONOTONIC_RAW` schedule.

- A late tick is never replayed; every scheduled instant it missed counts
  as a dropped sample.
- The footer reports the interval P50/P95/P99/max, dropped samples,
  per-sample cost, the sampler's own CPU, and any NVML queries the driver
  did not support.

## Clock domains

- **Span and mark boundaries:** `CLOCK_MONOTONIC_RAW` in every process.
  - The container shares the host kernel, and Docker does not namespace
    monotonic clocks.
  - The analyzer verifies a single raw domain:
    - the per-process `realtime − raw` medians agree within 2 ms;
    - all processes report the same `boot_id`.
- **Phase-6's own stamps** are unchanged:
  - raw for state commit, action open, receipt and the deadline;
  - realtime for capture and the edge/map wall stamps;
  - `time.monotonic_ns` in the base `StageRecorder`.
- **NVML utilization** is the driver's sampled window, not an instantaneous
  value.

## Offline analysis (`analyze_v1`) and qualification gates

- **Validity:** for each stream:
  - exactly one header and one footer;
  - unique sequence numbers;
  - no negative interval;
  - every span inside its parent, with the registered parent stage;
  - no crossed siblings;
  - monotonic per-thread starts and marks;
  - every CUDA span resolved, or explicitly unresolved.
- **Identity:** every live record that names a frame equals that frame's
  transmitted identity. Edge and map records may not name an untransmitted
  frame.
- **Coverage:** frames covered per registered denominator, with the missing
  count and examples.
- **Semantics:**
  - every live actor call is replayed offline through the configured frozen
    actor (`act_on_vector`), and mode, q_e4, state and head digests must be
    identical;
  - every resolution obeys the configured deadline and reward formula
    exactly;
  - every feedback-receipt mark equals the authoritative receipt stamp.
- **Gates:** G1–G9 in `qualification_run4_300_v1.json`, registered before any
  profiled evidence existed. The 300-frame result qualifies the
  instrumentation only. It is not the profiling baseline.

## Pre-existing host condition (recorded, not changed)

`nvidia-cuda-mps-control -d` has run under this account since 2026-07-25, and
`nvidia-cuda-mps-server` since 2026-08-25. Both were therefore present during
the `6199d49` reference run. This task neither enabled nor altered MPS. The
runner records the MPS processes and pipe directory, and any foreign I/O
processes, before and after the run.
