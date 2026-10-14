# M4 two UE preparation blocker

**BLOCKED — not READY_FOR_TWO_UE_LIVE_PREFLIGHT.** The qualified single-UE GT path cannot safely accept two streams in one service. The failure is reproduced in its actual sender and receiver. Disk also fails the unchanged launch minimum. No M4 registration, deployment, smoke test or capture was performed.

## Concrete integration boundary

The CPU fixture submits two valid, different stream/session identities with the same CARLA frame 4172. Both predictions serialize successfully, but the retained evaluator keeps only the second pending ticket. When the fixture returns the first request's frame-only `UNSCORED` response, actual feedback construction uses the second session. Two terminal notifications likewise contain no source identity. Nested evaluator installation is explicitly refused.

This is an unsupported two-source reuse, **not a claim that previous single-UE results were corrupted**. The relevant boundaries are:

- `splitfusion_exact_gt_live_v2/edge.py`: process-global evaluator; `_materialized`, `_pending` and `_frames` keyed only by frame; frame-only terminal/emission messages.
- `splitfusion_exact_gt_live_v2/service.py`: `frame()` and service lifecycle keyed only by frame; one UE connection and one edge connection; frame-only result responses. Changing only the edge pending key is insufficient.
- `phase6_edge_runtime_v2.py` and `splitfusion_direct_edge_map_v1/map_ingest.py`: each sends feedback to one configured destination. M4 needs verified source routing, not the NAT-observed sender address.
- `splitfusion_run5_full_live_v1/runtime_policy.py`: explicitly permits one physical UE/runtime per policy scope. Two unmodified child launchers would also duplicate route/world ownership rather than provide the required single ticker.

A separate incompatibility is reproduced: the real Run4 validator accepts an off-anchor mode-11/q_e4=4321 (q=0.4321) map document, but `coop_input_v1.adapter_direct_v1` refuses its schema. That adapter requires the older catalog `action_id/profile_id` representation. Neither retagging the message nor fabricating an anchor is acceptable. The explicit `CoopIdentityConfig` lookup remains reusable.

Resolving this requires a source-isolated GT/service composition or end-to-end source-aware internal IPC, connected to the two-ego lifecycle and feedback router. The Run4 wire can remain unchanged. Work stopped at this substantial integration boundary under the task's blocker provision; the production integration remains incomplete.

## Implemented and checked

Added only `rl_agent/splitfusion_m4_two_ue_v1/{__init__,check_boundaries}.py` and this experiment directory. The executable probe reuses the actual prediction serializer, native-size synthetic label array, context validator, memfd sealing, GT send/receive loops, feedback encoder, map validator and existing telemetry fixtures. Only socket I/O and thread startup are replaced. It performs no inference, socket connection or service launch; CUDA remained uninitialized.

[BOUNDARY_PROBE_01.json](BOUNDARY_PROBE_01.json) records the counterexamples, exact source hashes, registration hash and three passing existing negative-control tests: foreign-RNTI DCI, foreign-UE RLC, and cross-session/cross-UE decision admission. Selected runtime sources match both checkouts and the actual V20 registration. This is **not** complete prospective source/startup verification or two-UE qualification. V20's original STOP remains unchanged.

The actual V20 registration file SHA256 is `d28f6c5e269fceb8f5554da0caad1bc4e2c3838160cdaa8c8d19267af365c76d`. No new qualifications are claimed for actor-history independence, source-specific supersession/terminal accounting, both-route completion/drain, radio attach, per-UE SNR, or combined analysis.

## Required ownership after the blocker

The integration must retain one CARLA/background-traffic owner, two sensor/pose/actor-history contexts, and explicit `ego ↔ OAI UE ↔ IMSI/tunnel ↔ dynamic lease/RNTI ↔ stream/session` bindings. Both egos must complete and drain. Front contexts are separate; GPU, radio cell and host CPU resources are shared. GT logical ownership must be separate; whether services are process-isolated or source-aware is not yet implemented. A shared tail requires source-bound dispatch; simply launching two complete edge/model stacks is not a validated substitute.

Reuse the historical two-logical-UE attach and tracer relay, but not fixed leases, majority-guessed RNTI ownership, old radio settings or the localhost action-50 demo. Preserve 100 MHz/273 PRB/4D5U and commanded 23.5 dB. Verify each UE's achieved PUSCH separately before measurement; the single-UE noise mapping is not a two-UE calibration proof. Headless map output must preserve raw per-source observations and pose provenance without association or unique-object claims.

## Commands and stopping point

From the working repository, reproduce the focused offline check:

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /usr/bin/python3 -B -m rl_agent.splitfusion_m4_two_ue_v1.check_boundaries
df -B1 .
```

The probe prints evidence and exits **2 for BLOCKED**. The recorded invocation additionally used `--output experiments/splitfusion_m4_two_ue_v1/BOUNDARY_PROBE_01.json`; output is create-only, so do not reuse that filename.

**There is no safe M4 smoke or full-route command yet.** Do not run historical launch commands as substitutes. Mandatory production binding/startup, exclusive-host checks, actual two-UE attach/data-path smoke and per-UE channel verification remain prerequisites for a future launcher. Cleanup command: none required or executed; no owned live processes, containers, radio modifications or CARLA actors were created.

## Disk headroom

The final read-only snapshot was **12,750,327,808 bytes free (12.75 GB, 11.87 GiB)**, about **4.43 GB short** of the unchanged 16-GiB gate. Free space changed during preparation; recheck before any future launch.

Measured allocated attempt sizes are 7,262,527,488 bytes / 3,085 frames for V19 and 8,286,314,496 bytes / 3,576 frames for V20. Scaling the larger bytes/frame rate to two 3,900-frame ceilings gives **18.36 GB**; a 25% planning margin gives **22.95 GB**. This is a conservative planning estimate from retained attempts, not a proved peak bound: fixed retention, transient GT storage and shared-resource slowdown need aggregate capacity validation. Provisionally budget **32 GiB free** for two streams; this does not replace or lower existing checks and is not a sealed M4 capacity requirement.

No evidence was deleted or archived. No viewer/Claude files, shared handoffs, qualified runtime sources or historical verdicts were edited. No commit or push was made.
