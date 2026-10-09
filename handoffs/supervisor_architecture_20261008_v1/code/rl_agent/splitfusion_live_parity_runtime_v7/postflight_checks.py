"""Additive GT optimization scope proof; retained v2 correctness gates remain."""
import hashlib
import json
from pathlib import Path
from rl_agent.splitfusion_live_parity_runtime_v4.contract import OPTIMIZATION_SOURCES as RETAINED_V1_SOURCES


SCHEMA = "splitfusion.parity_runtime.optimized_gt_scope.v7"
SETTING_FIELDS = frozenset(("environment", "calling_thread_affinity", "torch_threads",
                            "torch_interop_threads", "opencv_threads"))
ENV_FIELDS = frozenset(("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                       "CUDA_VISIBLE_DEVICES", "CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"))


def verify_input_scope(cell, *, expected_prospective_sha256):
    from . import contract as C
    from rl_agent.splitfusion_paired_quality_replay_v1.capture import verify_archive
    cell = Path(cell)
    scope, scope_sha = _read(cell/"PAIRED_INPUT_SCOPE.json")
    require(scope.get("schema") == "splitfusion.paired_input.scope.v7"
        and scope.get("complete") is True and scope.get("returncode") == 0
        and scope.get("error") is None and scope.get("observer_errors") == []
        and scope.get("observer_error_count") == 0
        and scope.get("max_frames") == C.INPUT_MAX_FRAMES and scope.get("max_bytes") == C.INPUT_MAX_BYTES
        and scope.get("prospective_record_sha256") == expected_prospective_sha256,
        "input capture scope incomplete or foreign")
    record = scope["archive"]
    manifest = verify_archive(cell/"paired_input_archive", expected_manifest_sha256=record["manifest_sha256"])
    require(manifest.get("max_frames") == C.INPUT_MAX_FRAMES and manifest.get("max_bytes") == C.INPUT_MAX_BYTES,
        "input archive differs from registered bounds")
    require(record.get("complete") is True and record["frames"] == len(manifest["frames"])
        and scope["policy_opportunities"] == len(manifest["frames"])
        and record["drain"] == manifest["drain"] and record["retained_bytes"] == manifest["retained_bytes"],
        "input archive population/drain differs")
    return {"passed": True, "scope_sha256": scope_sha, "manifest_sha256": record["manifest_sha256"],
        "prepared_policy_inputs": record["frames"], "transmission_and_prediction_join_required": True,
        "runtime_fix_claim": False}


def require(value, message):
    if not value:
        raise ValueError(message)


def _read(path):
    require(path.is_file() and not path.is_symlink(), "scope evidence missing or symbolic")
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _settings(value):
    require(isinstance(value, dict) and set(value) == SETTING_FIELDS, "stable configuration fields incomplete")
    env = value["environment"]
    require(isinstance(env, dict) and set(env) == ENV_FIELDS, "environment invariant fields incomplete")
    require(all(x is None or isinstance(x, str) for x in env.values()), "invalid retained environment value")
    require(env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == "1", "retained UE numerical thread cap missing")
    cpus = value["calling_thread_affinity"]
    require(isinstance(cpus, list) and bool(cpus) and all(type(c) is int and c >= 0 for c in cpus)
            and cpus == sorted(set(cpus)), "invalid calling-thread affinity")
    for key in ("torch_threads", "torch_interop_threads", "opencv_threads"):
        require(type(value[key]) is int and value[key] > 0, "invalid numerical-library thread setting")


def verify_optimization_scope(cell, *, expected_sources, expected_worker_sources, expected_prospective_sha256):
    """Read only. Timeliness is descriptive, never an all-on-time requirement."""
    cell = Path(cell)
    require(cell.is_dir() and not cell.is_symlink(), "unsafe qualification cell")
    require(isinstance(expected_sources, dict) and bool(expected_sources)
            and all(isinstance(name, str) and Path(name).name == name and name.endswith(".py")
                    and _digest(value) for name, value in expected_sources.items()), "invalid prospective optimization inventory")
    require(_digest(expected_prospective_sha256), "prospective record hash required")
    scope, scope_sha = _read(cell/"GT_OPTIMIZATION_SCOPE.json")
    require(scope.get("schema") == SCHEMA and scope.get("status") == "COMPLETE"
            and type(scope.get("returncode")) is int and scope["returncode"] == 0
            and "error" in scope and scope["error"] is None, "optimization scope did not finish successfully")
    require(scope.get("optimization_sources") == expected_sources
            and scope.get("worker_sources") == expected_worker_sources
            and scope.get("prospective_record_sha256") == expected_prospective_sha256,
            "optimization scope differs from prospective binding")
    require(scope.get("worker_drained") is True and scope.get("ownership_manifest_complete") is True,
            "optimization scope exited before verified GT drain")
    state = scope.get("optimization", {})
    require(state.get("schema") == "splitfusion.cpu_gt_optimization.scope.v2"
            and state.get("sources") == expected_sources, "inner optimization source binding differs")
    require(state.get("retained_v1_sources") == RETAINED_V1_SOURCES,
            "retained v1 optimization source binding differs")
    candidate = state.get("candidate", {})
    for key in ("fast_labels", "fast_metadata", "all_pixel_validation", "fsync_unchanged"):
        require(candidate.get(key) is True, "selected exact v2 candidate scope missing")
    for key in ("frame_value_cache", "deferred_evidence", "new_gpu_work", "live_qualified"):
        require(candidate.get(key) is False, "v2 candidate changed its admitted scope")
    require(type(candidate.get("new_threads")) is int and candidate["new_threads"] == 0,
            "v2 candidate introduced threads")
    require(state.get("retained_settings_unchanged") is True
            and "source_integrity_error" in state and state["source_integrity_error"] is None,
            "retained settings or optimization sources changed")
    for key in ("new_python_threads", "new_queues"):
        require(type(state.get(key)) is int and state[key] == 0, "optimization introduced concurrency")
    for key in ("new_gpu_work", "forensic_writes_deferred", "live_qualified"):
        require(state.get(key) is False, "optimization changed its admitted scope")
    _settings(state.get("settings_before")); _settings(state.get("settings_after"))
    require(state["settings_before"] == state["settings_after"], "configuration differs despite unchanged flag")
    for key in ("cuda_lifecycle_before", "cuda_lifecycle_after"):
        require(type(state.get(key)) is bool, "CUDA lifecycle telemetry missing")
    require(not state["cuda_lifecycle_before"] or state["cuda_lifecycle_after"], "CUDA initialization moved backwards")
    require(state.get("cuda_initialized_by_retained_path") is
            (not state["cuda_lifecycle_before"] and state["cuda_lifecycle_after"]),
            "CUDA lifecycle attribution differs")
    require(type(state.get("prewarm_wall_ns")) is int and state["prewarm_wall_ns"] >= 0,
            "prewarm measurement missing")
    prewarm = state.get("prewarm", {})
    source, target = prewarm.get("source", {}), prewarm.get("target", {})
    require(source.get("fixed_grid_proved") is True and source.get("state_or_history_created") is False
            and isinstance(source.get("source_pins"), dict) and bool(source["source_pins"]),
            "exact source prewarm proof missing")
    require(target.get("frame_history_advanced") is False and type(target.get("frame_inputs_consumed")) is int
            and target["frame_inputs_consumed"] == 0 and isinstance(target.get("sources"), dict)
            and bool(target["sources"]), "target prewarm advanced state or lacked source proof")
    ownership_dir = cell/"source_ownership"
    require(ownership_dir.is_dir() and not ownership_dir.is_symlink(), "ownership archive missing/unsafe")
    ownership, ownership_sha = _read(ownership_dir/"MANIFEST.json")
    require(ownership.get("schema") == "splitfusion.parity_repair.exact_source_ownership.v2"
            and ownership.get("complete") is True and ownership.get("errors") == []
            and ownership.get("worker_drained") is True and ownership.get("worker_alive") is False
            and ownership.get("pending_at_close") == ownership.get("owned_bytes_at_close") == 0,
            "actual ownership archive contradicts completed scope")
    # Full native ownership events/population and all five timed stages are
    # independently verified by repair_v2.install_postflight_reader, not here.
    worker = verify_worker_scope(cell, expected_sources=expected_worker_sources,
        outer_scope=scope)
    return {"passed": True, "schema": SCHEMA, "scope_sha256": scope_sha,
            "ownership_manifest_sha256": ownership_sha, "optimization_sources": dict(expected_sources),
            "cpu_worker": worker,
            "retained_configuration_unchanged": True, "worker_drained": True,
            "cuda_lifecycle_before": state["cuda_lifecycle_before"],
            "cuda_lifecycle_after": state["cuda_lifecycle_after"],
            "latency_qualification_claim": False}


def _child_context(context, pid):
    require(isinstance(context, dict) and context.get("pid") == pid
        and context.get("cuda_initialized") is False, "CPU child context identity/CUDA differs")
    schedule = context.get("scheduling", {})
    affinity = schedule.get("affinity")
    require(isinstance(affinity, list) and bool(affinity) and affinity == sorted(set(affinity))
        and all(type(v) is int and v >= 0 for v in affinity), "CPU child affinity missing")
    require(set(schedule) == {"affinity", "scheduler", "nice"}
        and type(schedule["scheduler"]) is int and type(schedule["nice"]) is int,
        "CPU child scheduling context missing")
    threads = context.get("threads")
    require(isinstance(threads, dict) and bool(threads)
        and str(pid) in threads
        and all(str(tid).isdigit() and value == schedule for tid, value in threads.items()),
        "CPU child thread scheduling differs")
    require(context.get("environment") == dict(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1") and type(context.get("torch_threads")) is int
        and context["torch_threads"] == 1,
        "CPU child environment/cap differs")
    require(all(type(context.get(key)) is int and context[key] > 0
        for key in ("torch_interop_threads", "opencv_threads")), "CPU library context missing")
    return schedule


def _verify_worker_observer(cell, *, proof, sources, pid, ready, schedule, ipc, stage_rows,
                            timing_manifest, target_identities):
    from . import contract as C
    timeline, digest = _read(Path(cell)/"CPU_GT_PROCESS_TIMELINE.json")
    require(proof.get("timeline_sha256") == digest
        and timeline.get("schema") == "splitfusion.cpu_gt_process_timeline.v7"
        and timeline.get("factory") == "rl_agent.splitfusion_live_parity_runtime_v7.worker_observer.ObservedClient"
        and timeline.get("sources") == sources and timeline.get("max_frames") == C.BUDGET,
        "CPU process observer source/factory binding differs")
    require(timeline.get("clock") == "CLOCK_MONOTONIC_RAW" and timeline.get("cpu_clock") == "thread_time_ns"
        and timeline.get("errors") == [] and timeline.get("dropped") == 0
        and timeline.get("new_threads") == timeline.get("new_queues") == 0,
        "CPU process observation failed or changed concurrency")
    for key in ("complete", "worker_drained", "child_stopped", "observation_cost_included"):
        require(timeline.get(key) is True, "CPU process observation incomplete: " + key)
    clients = timeline.get("clients")
    require(isinstance(clients, list) and len(clients) == 1, "CPU process lifecycle count differs")
    client = clients[0]
    require(client.get("pid") == pid and client.get("ready") == ready
        and client.get("failed") is False and client.get("exitcode") == 0,
        "CPU process READY/exit evidence differs")
    closed = client.get("close", {})
    require(isinstance(closed, dict) and closed.get("status") == "OK" and closed.get("processed") == len(ipc),
        "CPU process CLOSE count differs")
    require(_child_context(closed.get("process_context"), pid) == schedule, "CPU process CLOSE scheduling changed")
    require(closed.get("timing") == timing_manifest, "CPU process CLOSE timing manifest differs")
    requests = timeline.get("requests")
    require(isinstance(requests, list) and len(requests) == len(ipc), "CPU process request coverage differs")
    for index, (record, request) in enumerate(zip(requests, ipc)):
        identity = record.get("identity", {})
        require(record.get("status") == "OK" and record.get("error") is None
            and identity == target_identities[index]
            and identity.get("frame_id") == request["frame_id"]
            and record.get("ipc") == {key: value for key, value in request.items() if key != "stage"},
            "CPU process request/parent IPC differs")
        require(_child_context(record.get("process_context"), pid) == schedule, "CPU process request scheduling changed")
        worker = record.get("worker", {})
        begin, end = worker.get("start_raw_ns"), worker.get("end_raw_ns")
        require(worker.get("pid") == pid and type(begin) is int and type(end) is int
            and request["sealed_raw_ns"] <= begin <= end <= request["completed_raw_ns"]
            and type(worker.get("thread_cpu_ns")) is int and worker["thread_cpu_ns"] >= 0,
            "CPU process execution clocks/PID missing")
        rows = record.get("rows")
        require(isinstance(rows, list) and len(rows) == 5
            and all(row.get("process_pid") == pid and row.get("process_role") == "dedicated_cpu_gt" for row in rows),
            "CPU process stage role/PID missing")
        stripped = [{key: value for key, value in row.items() if key not in ("process_pid", "process_role")} for row in rows]
        require(stripped == stage_rows[index*5:(index+1)*5]
            and all(begin <= row["start_raw_ns"] <= row["end_raw_ns"] <= end for row in rows),
            "CPU process request stages differ from actual flushed child timing")
    return dict(passed=True, timeline_sha256=digest, requests=len(requests), genuine_child_pid=pid,
        per_request_context_verified=True, close_exitcode=0)


def verify_worker_scope(cell, *, expected_sources, outer_scope):
    """No timing threshold: require exact ownership, source and child evidence."""
    from . import contract as C
    cell = Path(cell)
    state, digest = _read(cell/"GT_WORKER_SCOPE.json")
    require(isinstance(expected_sources, dict) and bool(expected_sources)
        and all(isinstance(name, str) and not Path(name).is_absolute() and ".." not in Path(name).parts
                and _digest(sha) for name, sha in expected_sources.items()), "invalid worker source binding")
    proof = outer_scope.get("cpu_worker", {})
    require(proof.get("complete") is True and proof.get("scope_sha256") == digest
        and proof.get("state") == state, "outer CPU worker scope is absent or differs")
    require(state.get("schema") == "splitfusion.cpu_gt_worker.runtime_scope.v4"
        and state.get("sources") == expected_sources and state.get("max_frames") == C.BUDGET
        and state.get("errors") == [] and state.get("live_qualified") is False,
        "CPU worker state or source binding differs")
    for key in ("complete", "joint_drained", "preserved_before_teardown", "preservation_started",
                "worker_close_succeeded", "no_cuda_work"):
        require(state.get(key) is True, "CPU worker completion proof missing: " + key)
    require(all(type(state.get(key)) is int for key in ("new_cpu_processes", "new_queues", "new_python_threads"))
        and state["new_cpu_processes"] == 1 and state["new_queues"] == state["new_python_threads"] == 0,
        "CPU worker concurrency differs")
    pid = state.get("child_pid")
    require(type(pid) is int and pid > 0, "CPU worker PID missing")
    ready = state.get("child_ready", {})
    require(ready.get("status") == "OK" and ready.get("pid") == pid
        and ready.get("cuda_initialized") is False, "CPU worker readiness missing")
    schedule = _child_context(ready.get("process_context"), pid)
    begin, ready_ns = state.get("started_raw_ns"), state.get("ready_raw_ns")
    require(type(begin) is int and type(ready_ns) is int and 0 < begin <= ready_ns,
        "CPU worker startup clocks invalid")
    ipc = state.get("parent_ipc_records")
    require(isinstance(ipc, list) and 0 < len(ipc) <= C.BUDGET, "CPU worker IPC evidence missing/unbounded")
    frames, previous_end = [], ready_ns
    for row in ipc:
        require(row.get("stage") == "parent_snapshot_and_ipc" and row.get("child_pid") == pid,
            "CPU worker IPC role/PID differs")
        frame = row.get("frame_id")
        require(type(frame) is int and frame >= 0, "CPU worker IPC frame invalid")
        frames.append(frame)
        times = [row.get(key) for key in ("start_raw_ns", "packed_raw_ns", "sealed_raw_ns", "completed_raw_ns")]
        require(all(type(value) is int for value in times) and previous_end <= times[0]
            and times == sorted(times), "CPU worker IPC clocks crossed")
        previous_end = times[-1]
        require(type(row.get("parent_thread_cpu_ns")) is int and row["parent_thread_cpu_ns"] >= 0
            and type(row.get("request_bytes")) is int and 0 < row["request_bytes"] <= 32*1024*1024,
            "CPU worker IPC cost or size invalid")
    require(frames == sorted(set(frames)), "CPU worker frames duplicate/out of order")
    targets, target_manifest_sha = _read(cell/"parity_complete_targets/MANIFEST.json")
    target_rows = targets.get("frames")
    require(targets.get("schema") == "splitfusion.complete_target_preservation.v1"
        and isinstance(target_rows, list) and len(target_rows) == len(frames)
        and all(isinstance(row, dict) and isinstance(row.get("identity"), dict) for row in target_rows),
        "CPU worker target manifest missing/foreign")
    target_identities = [row["identity"] for row in target_rows]
    require([identity.get("frame_id") for identity in target_identities] == frames,
        "CPU worker IPC population differs from preserved complete targets")
    timing = state.get("child_timing", {})
    source = cell/"cpu_gt_worker/engine/complete_gt_stage_timing"
    destination = cell/"complete_gt_stage_timing"
    require(timing.get("source") == str(source.resolve()) and timing.get("destination") == str(destination.resolve())
        and timing.get("process_pid") == pid and timing.get("process_role") == "dedicated_cpu_gt"
        and timing.get("copied_only_after_joint_drain") is True, "CPU worker timing provenance differs")
    require(source.is_dir() and not source.is_symlink() and destination.is_dir() and not destination.is_symlink(),
        "CPU worker original/copied timing directories missing")
    for folder in (source, destination):
        require({path.name for path in folder.iterdir()} == {"MANIFEST.json", "stages.jsonl"}
            and not any(path.is_symlink() for path in folder.iterdir()), "CPU timing inventory differs")
    original, original_sha = _read(source/"MANIFEST.json")
    copied, copied_sha = _read(destination/"MANIFEST.json")
    require(original_sha == copied_sha == timing.get("original_manifest_sha256")
        and original == copied, "CPU timing manifest copy differs")
    raw = (source/"stages.jsonl").read_bytes()
    require(raw == (destination/"stages.jsonl").read_bytes()
        and original.get("sha256") == {"stages.jsonl": hashlib.sha256(raw).hexdigest()}
        and original.get("schema") == "splitfusion.live_parity_repair.gt_stage_timing.v2"
        and original.get("flush_after_worker_drain") is True and original.get("dropped") == 0
        and original.get("bound") == 8*C.BUDGET
        and original.get("clock") == "CLOCK_MONOTONIC_RAW" and original.get("cpu_clock") == "thread_time_ns"
        and original.get("work_and_readiness_unchanged") is True, "CPU timing evidence invalid")
    rows = [json.loads(line) for line in raw.splitlines()]
    require(len(rows) == original.get("records") == timing.get("records") == 5*len(frames),
        "CPU stage count differs from IPC population")
    stages = ["avo_scoring_and_sidecar", "complete_source_build", "prepare_complete_target",
              "publish_complete_target", "archive_auxiliary_sources"]
    for index, request in enumerate(ipc):
        selected = rows[5*index:5*(index+1)]
        require([row.get("stage") for row in selected] == stages, "CPU stage ordering differs")
        previous = request["sealed_raw_ns"]
        for row in selected:
            start, end = row.get("start_raw_ns"), row.get("end_raw_ns")
            require(row.get("frame_id") == request["frame_id"] and row.get("completed") is True
                and type(start) is int and type(end) is int and previous <= start <= end <= request["completed_raw_ns"]
                and row.get("wall_ns") == end-start and type(row.get("thread_cpu_ns")) is int
                and row["thread_cpu_ns"] >= 0, "CPU stage identity/interval incomplete")
            previous = end
    observer = _verify_worker_observer(cell, proof=proof, sources=expected_sources, pid=pid,
        ready=ready, schedule=schedule, ipc=ipc, stage_rows=rows,
        timing_manifest=original, target_identities=target_identities)
    return dict(passed=True, scope_sha256=digest, child_pid=pid, requests=len(ipc), genuine_child_stages=len(rows),
        complete_target_manifest_sha256=target_manifest_sha, target_population_and_full_identities_equal=True,
        original_timing_manifest_sha256=original_sha, copied_byte_identical=True,
        child_cpu_only=True, joint_drained=True, observer=observer, latency_qualification_claim=False)
