# M4 executable composition and preflight handoff

The two-source production entry point is implemented and its focused CPU checks pass. **Live-host preflight is blocked by external prerequisites, not storage:** `oai-cn5g-public-net` and `oai-amf`, `oai-smf`, `oai-upf`, `mysql` are absent. A separate host GPU query also reports an existing MPS server (PID 794777) and remote-desktop CUDA client (PID 3031); exclusive ownership has not passed. None was stopped or changed.

Status: `OFFLINE_COMPOSITION_PASS__LIVE_HOST_PREREQUISITES_BLOCKED`. Do not interpret this as `READY_FOR_TWO_UE_LIVE_PREFLIGHT_PENDING_STORAGE`: the storage gate now passes, but the named host prerequisites do not. No CARLA, OAI, GPU inference, live attempt, deployment, training, evidence deletion, commit or push occurred.

## Executable composition

`rl_agent.splitfusion_m4_two_ue_v1.cli` implements `prepare`, `preflight`, `check`, `smoke`, `capture`, `analyze` and ownership-checked `cleanup`. Outputs are create-only; there is no capture retry.

One retained density runner owns world load, traffic and ticking. Two retained route drivers rendezvous at that single tick owner, with distinct sensors, collectors, native 22-feature builders and controllers. The following spawn uses the actual lane graph and must satisfy the approximately 15-m gap/heading/bounding-box checks. Each ego must complete its own route and drain; only then is that ego removed so it cannot obstruct the other at the common finish point. No teleport or collision bypass is used. The 3,900-frame bound is per source, never route completion.

The established `--num-ues 2` softmodem and tracer relay resolve registered IMSIs/tunnels to actual leases and unambiguous RNTIs. Per-datagram `IP_PKTINFO` binds each front sender to its own tunnel and source lease; feedback uses that source's destination. The two ACKed 100-ms GOOD_ALL publications feed separate native SNR providers. Commanded 23.5 dB / noise −12 remains distinct from measured per-transmission PUSCH SNR/MCS. Two-UE calibration is **unverified**.

Two independent front offer paths share immutable frontend weights; one detached edge inference stack serves source-local reassembly/pending slots. The qualified isolated GT services retain containment, accounting V3.v2, FREEZE/GC/P3 and capacity checks. Actual headless Run4 ingest installs into source-separated histories and journals original records, continuous actions, identity and owned pose. The compact exact-record HTTP endpoint serves the retained collectors only; there is no UI or fusion. Map eligibility retains its freshness/order rules independently of 170-ms feedback.

Startup orders attach, telemetry/ACKed SNR, shared frontend prewarm, world/catalogues, both collectors/GT READY, then shared-edge prewarm/READY and route ticks. Both final-frame drains precede edge/GT shutdown. Normal teardown records every owned resource, restores both RFsim channels to −50, restores the host affinity, and preserves the original failure plus cleanup errors. Recovery refuses active parents or unverified/reused PIDs. A CARLA crash before its RPC ownership certificate requires ownership inspection; recovery never guesses a process group.

## Changed files and CPU placement

New composition files are confined to `rl_agent/splitfusion_m4_two_ue_v1/`:

| Purpose | Files |
| --- | --- |
| Entry and binding | `cli.py`, `configuration.py`, `preflight.py`, `binding.py`, `host.py` |
| World and frontend | `bootstrap.py`, `world.py`, `frontend.py`, `policy.py` |
| Radio and lifecycle | `radio.py`, `radio_runtime.py`, `telemetry.py`, `sender.py`, `resources.py` |
| Shared edge and map | `edge_components.py`, `edge_entry.py`, `datapath.py`, `map_service.py` |
| Evidence | `analysis.py`, `composed_checks.py` |

The previously qualified `identity.py`, `gt.py`, `service_entry.py`, `transport.py`, `map_adapter.py`, `composition.py` and old qualification tests remain byte-identical. The slim-safe edge dependency split is required to have AST-identical evaluator/router/processor class bodies; it avoids importing host SciPy/YAML dependencies merely to verify a registration.

Prospective placement preserves radio CPUs **0–3, 8–19** and the non-radio pool **4–7, 20–23**. GT prep/score/coordinator are **4/5/20** and **6/7/21**; shared inference **20–21**; frontend workers **22** and **23**; map **22–23**. Host/world/CARLA inherit the non-radio pool. Coordinator/inference, front/map and world contention are explicit and **not live-qualified**; the GT-only capacity result does not prove shared-GPU or full-route capacity.

## Verification and analysis

[composed_checks_13/RESULT.json](composed_checks_13/RESULT.json) passes **10/10**, 2.584 s, CUDA uninitialized. It invokes the real runtime constructor with labelled external GPU/socket doubles and exercises two native controller contexts, equal frame/sequence numbers, interleaved ingress, source-bound GT refusal/feedback, wrong-source rejection, supersession/stale terminals, actual map normalization/journaling of vehicles and pedestrians, separate completion/drain, original-error preservation, CLI failure paths and empty-source assessment. Rehashed subscriber/endpoint/GT-path swaps, configuration drift, inventory omission, isolated source mismatch and wrong live tokens are rejected.

The real-service `correctness_04`, 9-test `offline_checks_04`, and `capacity_01` proofs were hash-verified and reused only for unchanged components. **No capacity benchmark was repeated.** Failed development checks and superseded preparations remain intact; only `preparation_06` binds final sources. The test fixtures do not represent two captured egos.

[VERIFICATION_03.json](VERIFICATION_03.json) records exact final commands/results. Fresh-interpreter host offline preflight passes; pinned, read-only, network-disabled slim-image edge CLI preflight passes, both without CUDA initialization. `preflight --host` exits **2**, correctly refusing the missing network. Actual attach, spawn/lane geometry, achieved channel, shared-GPU load, live failure cleanup and full-route/density/collision acceptance remain future checks.

The assessment is bound to this preparation, reports per-UE results before combined observations, and fails closed on incomplete sources. It separates capture-to-CPU-prediction-ready latency from action-open-to-R4FB feedback latency, including timely scored feedback; STALE/timeout receipts are never prediction latency. It reports source radio/queue/compute statistics, payload/actions, original map freshness, missing GT versus scored undefined/zero-eligible quality, all-decision actor replay and exact scored-frame rescoring under the retained P3 rule. Ordered route coverage keeps the existing 0.98 / 8-m rule. Combined detections are observations, not unique objects or cooperative gains. HARQ rounds are not inferred from the retained PUSCH event.

## Commands and deployment boundary

From the working repository, the executed preparation command was:

```bash
env -u PYTHONPATH CUDA_VISIBLE_DEVICES= python3 -B -m rl_agent.splitfusion_m4_two_ue_v1.cli prepare --output experiments/splitfusion_m4_two_ue_v1/preparation_06
```

No files were deployed to `../abiodun_run4_single_gpu_profiler_v1`. Before an approved deployment launch, install the new package and the exact inventory dependencies/proof records in the intended worktree, prepare a **new worktree-bound** preparation there, and repeat its offline and slim-container preflights. Copying a main-worktree preparation is not sufficient: the host checker rejects a different repository binding. The commands below name the current main-worktree preparation; use the newly verified deployment preparation if deploying.

```bash
m4_prep=experiments/splitfusion_m4_two_ue_v1/preparation_06/PREPARATION.json
env -u PYTHONPATH CUDA_VISIBLE_DEVICES= python3 -B -m rl_agent.splitfusion_m4_two_ue_v1.cli preflight --preparation "$m4_prep"
env -u PYTHONPATH CUDA_VISIBLE_DEVICES= python3 -B -m rl_agent.splitfusion_m4_two_ue_v1.cli preflight --preparation "$m4_prep" --host
```

Only after external CN5G restoration, exclusive-host coordination, passing checks and **separate live authorization**, choose the approved command; these are not an automatic sequence:

```bash
env -u PYTHONPATH -u CUDA_VISIBLE_DEVICES OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -B -m rl_agent.splitfusion_m4_two_ue_v1.cli smoke --preparation "$m4_prep" --output experiments/splitfusion_m4_two_ue_v1/smoke_live_01 --execute-token M4_TWO_UE_SMOKE_SEPARATELY_AUTHORIZED
env -u PYTHONPATH -u CUDA_VISIBLE_DEVICES OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -B -m rl_agent.splitfusion_m4_two_ue_v1.cli capture --preparation "$m4_prep" --output experiments/splitfusion_m4_two_ue_v1/full_live_01 --execute-token M4_TWO_UE_CAPTURE_SEPARATELY_AUTHORIZED
env -u PYTHONPATH CUDA_VISIBLE_DEVICES= python3 -B -m rl_agent.splitfusion_m4_two_ue_v1.cli analyze --preparation "$m4_prep" --capture experiments/splitfusion_m4_two_ue_v1/full_live_01 --output experiments/splitfusion_m4_two_ue_v1/full_analysis_01
```

Smoke is a bounded 20-s data-path check, explicitly not full-route completion. Cleanup normally runs automatically on success or failure. For an interrupted approved attempt, the first command below only inspects; the second requests ownership-checked recovery without deleting evidence:

```bash
python3 -B -m rl_agent.splitfusion_m4_two_ue_v1.cli cleanup --capture experiments/splitfusion_m4_two_ue_v1/full_live_01
python3 -B -m rl_agent.splitfusion_m4_two_ue_v1.cli cleanup --capture experiments/splitfusion_m4_two_ue_v1/full_live_01 --execute-owned
```

## Binding and storage

Final preparation digest: `813c34e9448ae8db8599b5280dbc8202fc818e0c04c491dfd0a27eae6118ee48`; source-inventory digest: `1c6c22ccf501b8c94cd5d320647e9b91abe1bd62e21b14d76078431fb10c1526`. The [preparation](preparation_06/PREPARATION.json) lists **707 exact source/model/config paths**; file SHA256 and proof hashes are in VERIFICATION_03.json. The deployed actor remains `bb9b0d256de79b4498cd622e1daa59edcf43ff3072dc9f675b265aac579ed3ce`.

[PROTECTED_INPUTS_03.json](PROTECTED_INPUTS_03.json) was written at the start for cleanup coordination; [PROTECTED_INPUTS_04.json](PROTECTED_INPUTS_04.json) supplements it with the final exact inventory, including calibration, model checkpoints/selections, priors and constructor cache. Preserve the named 49-frame corpus, V20 registration, three M4 proof directories and numeric-producer/scorer prerequisites. No unrelated captures are reserved.

Last measured free space: **35,791,691,776 bytes** (35.79 GB / 33.33 GiB), above the unchanged **16-GiB** live gate and the provisional 32-GiB planning allowance. New check/preparation directories total **15.41 MB**, within the 100-MB offline budget. The existing prospective capture allowance remains **22.95 GB**, not a proved peak. No data cleanup or archive operation was performed. Temporary CPU-verification containers exited; existing GPU clients and historical evidence were left untouched.
