"""Read-only qualification of scoring admission and subsequent durable evidence.

Packet readiness is never represented as disk READY. Retained numerical,
source, actor, prediction and pose checks are composed with explicit joins.
"""
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
from types import FunctionType

from rl_agent.splitfusion_live_parity_runtime_v7 import postflight_checks as V7

SCHEMA = "splitfusion.parity_runtime.optimized_gt_scope.v8"
REFERENCE_KIND = "SCORING_PACKET_SHA256_NOT_DURABLE_READY_SHA256"
TRANSPORT = "IMMUTABLE_SCORING_PACKET__DURABILITY_DEFERRED__EXACT_NUMERICS_V1"
STAGES = ("avo_scoring_and_sidecar", "complete_source_build", "prepare_complete_target",
          "publish_scoring_packet", "serialize_archive", "enqueue_archive")
require = V7.require


def _read(path):
    from rl_agent.splitfusion_live_parity_repair_v2.readiness import _read as raw_read, _json
    path = Path(path)
    require(not any(p.is_symlink() for p in (path, *path.parents)), "symbolic evidence path")
    raw = raw_read(path)
    return _json(raw), hashlib.sha256(raw).hexdigest()


def _index(rows, key="identity"):
    result = {}
    require(isinstance(rows, list), "bounded evidence list required")
    for row in rows:
        value = row[key]
        frame = value["frame_id"] if isinstance(value, dict) else value
        require(type(frame) is int and frame >= 0 and frame not in result, "duplicate/invalid evidence frame")
        result[frame] = row
    return result


def _interval(start, end, label):
    require(type(start) is int and type(end) is int and 0 < start <= end, label + " clocks crossed/missing")
    return (end-start)/1e6


def _packet_reference(record):
    receipt = record.get("scoring_handoff", {})
    require(V7._digest(receipt.get("packet_sha256")) and receipt.get("archive_durable") is False,
            "missing explicit scoring packet reference")
    return receipt["packet_sha256"]


def verify_gt_timing(directory, *, expected_frames):
    directory = Path(directory)
    require(directory.is_dir() and not directory.is_symlink()
        and {p.name for p in directory.iterdir()} == {"MANIFEST.json", "stages.jsonl"},
        "GT timing inventory differs")
    manifest, digest = _read(directory / "MANIFEST.json")
    require(not (directory / "stages.jsonl").is_symlink(), "symbolic GT timing")
    raw = (directory / "stages.jsonl").read_bytes()
    require(manifest.get("schema") == "splitfusion.gt_scoring_handoff.stage_timing.v1"
        and manifest.get("clock") == "CLOCK_MONOTONIC_RAW" and manifest.get("cpu_clock") == "thread_time_ns"
        and manifest.get("flush_after_worker_drain") is True and manifest.get("dropped") == 0
        and manifest.get("scoring_ready_is_not_archive_durable") is True
        and manifest.get("work_and_readiness_unchanged") is False
        and manifest.get("sha256") == {"stages.jsonl": hashlib.sha256(raw).hexdigest()},
        "GT scoring timing contract differs")
    rows = [json.loads(line) for line in raw.splitlines()]
    expected = {(frame, stage) for frame in expected_frames for stage in STAGES}
    seen = set()
    for row in rows:
        key = row.get("frame_id"), row.get("stage")
        require(key in expected and key not in seen and row.get("completed") is True,
            "GT timing missing, foreign, duplicate or failed stage")
        seen.add(key)
        _interval(row.get("start_raw_ns"), row.get("end_raw_ns"), "GT stage")
        require(row.get("wall_ns") == row["end_raw_ns"]-row["start_raw_ns"]
            and type(row.get("thread_cpu_ns")) is int and row["thread_cpu_ns"] >= 0,
            "GT stage wall/CPU duration differs")
    require(seen == expected and manifest.get("records") == len(rows)
        and len(rows) <= manifest.get("bound", -1), "GT stage population differs")
    for frame in expected_frames:
        selected = [r for r in rows if r["frame_id"] == frame]
        require([r["stage"] for r in selected] == list(STAGES)
            and all(a["end_raw_ns"] <= b["start_raw_ns"] for a,b in zip(selected, selected[1:])),
            "GT stage sequence crossed")
    return dict(passed=True, manifest_sha256=digest, frames=len(expected_frames), stages=len(rows),
        latency_qualification_claim=False, rows=rows)


def _verify_broker(edge, *, binding, uid, gid, max_frames):
    require(edge.get("schema") == "splitfusion.gt_packet_edge_scope.v1"
        and edge.get("binding") == binding and edge.get("max_frames") == max_frames
        and edge.get("complete") is True and edge.get("errors") == []
        and edge.get("transport_semantics") == TRANSPORT and edge.get("target_reference_kind") == REFERENCE_KIND
        and edge.get("new_broker_threads") == 1 and edge.get("durability_claim") is False
        and edge.get("reward_deadline_changed") is False and edge.get("quality_formula_changed") is False,
        "edge handoff scope incomplete or foreign")
    broker = edge.get("broker", {})
    require(broker.get("binding") == binding and broker.get("started") is True
        and broker.get("closed") is True and broker.get("worker_alive") is False
        and broker.get("failure") is None and broker.get("rejected_credentials") is None
        and broker.get("all_consumed") is True
        and broker.get("pending_jobs") == broker.get("pending_bytes") == broker.get("payloads_retained_for_unconsumed") == 0,
        "broker failed, still alive or has leftover scoring packets")
    policy = broker.get("credential_policy", {})
    require(policy == edge.get("publisher_credential_policy")
        and policy.get("publisher_uid") == uid and policy.get("publisher_gid") == gid
        and policy.get("explicitly_supplied") is True and policy.get("mode") == "0600"
        and policy.get("authentication") == "SO_PEERCRED_BEFORE_APPLICATION_READ",
        "broker publisher authentication differs from registration")
    socket = broker.get("socket_proof", {})
    require(socket.get("uid") == uid and socket.get("gid") == gid and socket.get("mode") == "0600"
        and all(type(socket.get(k)) is int and socket[k] > 0 for k in ("device", "inode")),
        "broker socket ownership/mode proof differs")
    ledger = _index(broker.get("ledger"))
    require(0 < len(ledger) <= max_frames and broker.get("admitted") == broker.get("consumed") == len(ledger),
        "broker admission/consumption count differs")
    require(broker.get("limits") == dict(max_jobs=min(2,max_frames), max_bytes=32*1024*1024, max_frames=max_frames)
        and type(broker.get("high_water_jobs")) is int and 0 < broker["high_water_jobs"] <= min(2,max_frames)
        and type(broker.get("high_water_bytes")) is int and 0 < broker["high_water_bytes"] <= 32*1024*1024,
        "broker pending capacity differs from registered bound")
    for row in ledger.values():
        credentials = row.get("publisher_credentials", {})
        require(row.get("binding") == binding and row.get("consumed") is True
            and credentials.get("uid") == uid and credentials.get("gid") == gid
            and type(credentials.get("pid")) is int and credentials["pid"] >= 0,
            "scoring publisher credentials or source binding differ")
    for field in ("prediction_state", "pose_state"):
        state = edge.get(field, {})
        require(state.get("flushed") is True and not state.get("errors") and not state.get("archive_errors"),
            "edge prediction/pose archive did not flush")
    close = edge.get("evaluator_close", {})
    require(close.get("worker_alive") is False and close.get("parity_archive_complete") is True
        and close.get("parity_archive_errors") == [], "actual evaluator close missing or failed")
    return ledger


def _verify_archive(archive, *, binding, roots, max_frames, preserved_cell=None):
    from rl_agent.splitfusion_live_gt_handoff_v1 import contract as H
    require(archive.get("schema") == "splitfusion.gt_archive_handoff.v1"
        and archive.get("binding") == {"schema": H.SCHEMA, "source_binding": binding}
        and archive.get("failed") is False and archive.get("fault") is None
        and archive.get("outstanding_jobs") == archive.get("outstanding_bytes") == 0
        and archive.get("child_returncode") == 0 and archive.get("child_stderr") == ""
        and type(archive.get("child_pid")) is int and archive["child_pid"] > 0,
        "archive did not drain and exit successfully")
    submissions = _index(archive.get("submissions"))
    completions = _index(archive.get("completions"))
    require(set(submissions) == set(completions) and 0 < len(submissions) <= max_frames
        and archive.get("submitted") == archive.get("completed") == len(submissions),
        "archive submission/completion population differs")
    require(len({r["job_id"] for r in submissions.values()}) == len(submissions), "duplicate archive job")
    for frame, submitted in submissions.items():
        completed = completions[frame]
        require(completed.get("status") == "COMPLETE" and completed.get("durable") is True
            and completed.get("error") is None and completed.get("identity") == submitted["identity"]
            and completed.get("job_id") == submitted["job_id"]
            and completed.get("pid") == archive.get("child_pid")
            and submitted.get("ownership") == "SEALED_KERNEL_TRANSFER_NOT_DURABLE",
            "archive completion is unsuccessful or belongs to another request")
        _interval(completed.get("start_ns"), completed.get("end_ns"), "archive CLOCK_MONOTONIC service")
        admitted = {(r["root"], r["relative"]): r for r in submitted["files"]}
        written = {(r["root"], r["relative"]): r for r in completed["written"]}
        require(len(admitted) == len(submitted["files"]) and len(written) == len(completed["written"])
            and set(admitted) == set(written), "archive omitted or duplicated files")
        require(sum(row["bytes"] for row in admitted.values()) == submitted["bytes"] == completed["bytes"],
            "archive admitted byte count differs")
        finalized = {r["path"]:r for r in completed["finalized_ready"]}
        require(len(finalized) == len(completed["finalized_ready"])
            and set(finalized) == set(submitted["stamp_ready_paths"]), "finalized READY population differs")
        for key, row in written.items():
            index, relative = key
            require(type(index) is int and 0 <= index < len(roots) and isinstance(relative, str)
                and not Path(relative).is_absolute() and all(p not in ("", ".", "..") for p in relative.split("/")),
                "archive path escapes registered roots")
            original_path = roots[index]/relative
            path = original_path
            if preserved_cell is not None and index == 0:
                # The original shared GT directory is a retained TemporaryDirectory.
                # Its explicit existing preservation maps top-level AVO files and
                # complete target/source directories, without changing any bytes.
                base = "run5_live_avo_inputs" if len(Path(relative).parts) == 1 else "parity_complete_targets"
                path = Path(preserved_cell)/base/relative
            require(path.is_file() and not any(p.is_symlink() for p in (path,*path.parents)),
                "archive file missing or symbolic")
            raw = path.read_bytes()
            require(hashlib.sha256(raw).hexdigest() == row["sha256"] and len(raw) == row["bytes"]
                and row["ready"] == admitted[key]["ready"], "actual archived bytes differ from writer receipt")
            if str(original_path) in finalized:
                ready = json.loads(raw)
                receipt = finalized[str(original_path)]
                require(receipt["sha256"] == row["sha256"] and ready["identity"] == submitted["identity"]
                    and all(type(ready.get(k)) is int and ready[k] > 0 and ready[k] == receipt[k]
                            for k in ("ready_raw_ns", "ready_wall_ns")), "actual finalized READY differs")
                unstamped = dict(ready, ready_raw_ns=0, ready_wall_ns=0)
                encoded = json.dumps(unstamped, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
                require(hashlib.sha256(encoded).hexdigest() == admitted[key]["sha256"]
                    and len(encoded) == admitted[key]["bytes"], "READY changed content beyond archival timestamps")
            else:
                require(row["sha256"] == admitted[key]["sha256"] and row["bytes"] == admitted[key]["bytes"],
                    "non-stamped archive bytes changed after admission")
    return submissions, completions


def verify_input_scope(cell, *, expected_prospective_sha256):
    from . import contract as C
    from rl_agent.splitfusion_paired_quality_replay_v1.capture import verify_archive
    scope, digest = _read(Path(cell)/"PAIRED_INPUT_SCOPE.json")
    require(scope.get("schema") == "splitfusion.paired_input.scope.v8"
        and scope.get("complete") is True and scope.get("returncode") == 0 and scope.get("error") is None
        and scope.get("observer_errors") == [] and scope.get("observer_error_count") == 0
        and scope.get("max_frames") == C.INPUT_MAX_FRAMES and scope.get("max_bytes") == C.INPUT_MAX_BYTES
        and scope.get("prospective_record_sha256") == expected_prospective_sha256, "input capture incomplete/foreign")
    record = scope["archive"]
    manifest = verify_archive(Path(cell)/"paired_input_archive", expected_manifest_sha256=record["manifest_sha256"])
    require(manifest.get("max_frames") == C.INPUT_MAX_FRAMES and manifest.get("max_bytes") == C.INPUT_MAX_BYTES
        and record.get("complete") is True and record["frames"] == len(manifest["frames"])
        and scope["policy_opportunities"] == len(manifest["frames"])
        and record["drain"] == manifest["drain"] and record["retained_bytes"] == manifest["retained_bytes"],
        "input archive population/bounds/drain differ")
    return dict(passed=True, scope_sha256=digest, manifest_sha256=record["manifest_sha256"],
        prepared_policy_inputs=record["frames"], runtime_fix_claim=False)


def verify_worker_scope(cell, *, expected_sources, outer_scope):
    from . import contract as C
    from rl_agent.splitfusion_live_gt_handoff_v1 import contract as H
    cell = Path(cell)
    state, digest = _read(cell/"GT_WORKER_SCOPE.json")
    proof = outer_scope.get("cpu_worker", {})
    require(proof.get("complete") is True and proof.get("scope_sha256") == digest
        and proof.get("state") == state, "actual worker scope differs from outer proof")
    require(state.get("schema") == "splitfusion.gt_scoring_handoff.runtime_scope.v1"
        and state.get("sources") == expected_sources and state.get("bootstrap_sources") == H.sources()
        and state.get("max_frames") == C.BUDGET and state.get("errors") == []
        and state.get("live_qualified") is False, "worker source/state binding differs")
    for key in ("complete", "joint_drained", "preserved_before_teardown", "preservation_started",
                "worker_close_succeeded", "no_cuda_work", "deferred_evidence",
                "scoring_ready_is_not_archive_durable", "pending_receipts_resolved_before_preservation"):
        require(state.get(key) is True, "worker drain/preservation proof missing: "+key)
    require(state.get("new_cpu_processes") == 2 and state.get("new_queues") == 2
        and state.get("new_python_threads") == 0 and not state.get("failed_partial_preservation"),
        "handoff worker concurrency or preservation differs")
    pid = state.get("child_pid")
    require(type(pid) is int and pid > 0, "GT child PID missing")
    ready = state.get("child_ready", {})
    require(ready.get("status") == "OK" and ready.get("pid") == pid
        and ready.get("cuda_initialized") is False, "GT child READY missing")
    schedule = V7._child_context(ready.get("process_context"), pid)
    H.validate_context(ready["process_context"], H.sources())
    _interval(state.get("started_raw_ns"), state.get("ready_raw_ns"), "GT startup")
    requests = state.get("parent_ipc_records")
    require(isinstance(requests, list) and 0 < len(requests) <= C.BUDGET, "GT IPC population missing/unbounded")
    ipc = _index(requests, "frame_id")
    timing = verify_gt_timing(cell/"complete_gt_stage_timing", expected_frames=ipc)
    source = cell/"cpu_gt_worker/engine/complete_gt_stage_timing"
    copied = cell/"complete_gt_stage_timing"
    provenance = state.get("child_timing", {})
    require(provenance.get("source") == str(source.resolve())
        and provenance.get("destination") == str(copied.resolve())
        and provenance.get("process_pid") == pid and provenance.get("process_role") == "dedicated_cpu_gt"
        and provenance.get("copied_only_after_joint_drain") is True
        and provenance.get("original_manifest_sha256") == timing["manifest_sha256"]
        and provenance.get("records") == timing["stages"], "GT timing provenance differs")
    for name in ("MANIFEST.json", "stages.jsonl"):
        require((source/name).is_file() and not (source/name).is_symlink()
            and (source/name).read_bytes() == (copied/name).read_bytes(), "GT timing copy differs")
    drain, drain_sha = _read(cell/"cpu_gt_worker/engine/HANDOFF_DRAIN.json")
    ledger, ledger_sha = _read(cell/"cpu_gt_worker/engine/HANDOFF_ARCHIVE_LEDGER.json")
    require(state.get("archival_drain_sha256") == proof.get("drain_sha256") == drain_sha
        and state.get("archival_ledger_sha256") == proof.get("ledger_sha256") == ledger_sha
        and state.get("handoff_ledger") == drain.get("ledger")
        and drain.get("ledger", {}).get("sha256") == ledger_sha
        and drain.get("binding") == ledger.get("binding") == H.binding()
        and drain.get("archive_drained") is True
        and drain.get("failed_batches") == ledger.get("failed_batches") == [],
        "durable drain/ledger differs or contains failed work")
    require(drain.get("ledger", {}).get("path") == str(cell/"cpu_gt_worker/engine/HANDOFF_ARCHIVE_LEDGER.json"),
        "archive ledger path differs")
    archive = ledger.get("archive", {})
    compact = {k:v for k,v in archive.items() if k not in ("submissions", "completions")}
    compact["completions"] = [{k:v for k,v in row.items() if k != "written"} for row in archive.get("completions", [])]
    require(state.get("archive") == drain.get("archive") == compact, "compact archive receipts differ")
    timeline, timeline_sha = _read(cell/"CPU_GT_PROCESS_TIMELINE.json")
    require(proof.get("timeline_sha256") == timeline_sha
        and timeline.get("schema") == "splitfusion.gt_handoff_process_timeline.v8"
        and timeline.get("factory") == "rl_agent.splitfusion_live_parity_runtime_v8.worker_observer.ObservedClient"
        and timeline.get("sources") == expected_sources and timeline.get("bootstrap_sources") == H.sources()
        and timeline.get("handoff_binding") == H.binding() and timeline.get("max_frames") == C.BUDGET
        and timeline.get("clock") == "CLOCK_MONOTONIC_RAW" and timeline.get("cpu_clock") == "thread_time_ns"
        and timeline.get("errors") == [] and timeline.get("dropped") == 0
        and timeline.get("new_threads") == timeline.get("new_queues") == 0,
        "GT observer source/clock/error contract differs")
    for key in ("complete", "worker_drained", "child_stopped", "observation_cost_included"):
        require(timeline.get(key) is True, "GT observer close incomplete: "+key)
    clients = timeline.get("clients", [])
    require(len(clients) == 1, "GT observer child population differs")
    client = clients[0]
    require(client.get("pid") == pid and client.get("ready") == ready and client.get("failed") is False
        and client.get("exitcode") == 0, "genuine GT READY/exit differs")
    closed = client.get("close", {})
    require(closed.get("status") == "OK" and closed.get("processed") == len(ipc)
        and closed.get("timing") == drain, "genuine GT CLOSE differs from durable drain")
    require(V7._child_context(closed.get("process_context"), pid) == schedule, "GT CLOSE scheduling changed")
    H.validate_context(closed["process_context"], H.sources())
    startup = client.get("archive_startup", {})
    require(startup.get("pid") == archive.get("child_pid") and startup.get("pid") != pid
        and startup.get("parent_pid") == pid and startup.get("environment") == H.ENVIRONMENT
        and startup.get("inspection") == "PROCFS_BEFORE_SENSOR_STARTUP"
        and startup.get("scheduling", {}).get("cpus") == schedule["affinity"]
        and startup.get("scheduling", {}).get("nice") == schedule["nice"]
        and archive.get("child_returncode") == 0,
        "second genuine CPU archive process startup/exit evidence differs")
    observed = _index(timeline.get("requests"))
    require(set(observed) == set(ipc), "GT observer IPC population differs")
    previous = state["ready_raw_ns"]
    for frame, request in ipc.items():
        times = [request.get(k) for k in ("start_raw_ns", "packed_raw_ns", "sealed_raw_ns", "completed_raw_ns")]
        require(all(type(t) is int for t in times) and previous <= times[0] and times == sorted(times)
            and request.get("child_pid") == pid and request.get("stage") == "parent_snapshot_and_ipc",
            "GT parent IPC order/PID differs")
        previous = times[-1]
        record = observed[frame]
        require(record.get("status") == "OK" and record.get("error") is None
            and record.get("ipc") == {k:v for k,v in request.items() if k != "stage"}, "GT observed IPC differs")
        require(V7._child_context(record.get("process_context"), pid) == schedule, "GT request scheduling changed")
        H.validate_context(record["process_context"], H.sources())
        worker = record.get("worker", {})
        require(worker.get("pid") == pid and times[2] <= worker.get("start_raw_ns", -1)
            <= worker.get("end_raw_ns", -1) <= times[3], "GT genuine child interval differs")
        rows = record.get("rows", [])
        require(len(rows) == len(STAGES)
            and all(r.get("process_pid") == pid and r.get("process_role") == "dedicated_cpu_gt" for r in rows),
            "GT stages lack genuine child role/PID")
        stripped = [{k:v for k,v in row.items() if k not in ("process_pid", "process_role")} for row in rows]
        require(stripped == [r for r in timing["rows"] if r["frame_id"] == frame]
            and all(worker["start_raw_ns"] <= r["start_raw_ns"] <= r["end_raw_ns"] <= worker["end_raw_ns"] for r in rows),
            "GT observation differs from original flushed stage rows")
    return dict(passed=True, scope_sha256=digest, timeline_sha256=timeline_sha,
        child_pid=pid, archive_child_pid=startup["pid"], new_cpu_processes=2, archive_drained=True,
        requests=len(ipc), original_timing_manifest_sha256=timing["manifest_sha256"],
        latency_qualification_claim=False)


def verify_optimization_scope(cell, *, expected_sources, expected_worker_sources, expected_prospective_sha256):
    original = V7.verify_optimization_scope
    verifier = FunctionType(original.__code__, dict(original.__globals__, SCHEMA=SCHEMA,
        verify_worker_scope=verify_worker_scope), original.__name__, original.__defaults__, original.__closure__)
    return verifier(cell, expected_sources=expected_sources, expected_worker_sources=expected_worker_sources,
        expected_prospective_sha256=expected_prospective_sha256)


def verify_handoff_scope(cell, *, expected_handoff_binding, expected_bootstrap_sources,
                         expected_publisher_uid, expected_publisher_gid):
    from . import contract as C
    from rl_agent.splitfusion_live_gt_handoff_v1 import scoring_packet as P, contract as H
    from rl_agent.splitfusion_live_parity_runtime_v1 import postflight as retained, pose
    from rl_agent.splitfusion_live_parity_repair_v1 import target_transport as T
    cell = Path(cell)
    require(expected_handoff_binding == H.binding() and expected_bootstrap_sources == H.sources(),
        "handoff sources differ from prospective registration")
    data = retained._population(cell)
    edge_dir = cell/"profiler/edge"
    edge, edge_sha = _read(edge_dir/"EDGE_HANDOFF_SCOPE.json")
    timeline, _ = _read(cell/"CPU_GT_PROCESS_TIMELINE.json")
    startup = timeline["clients"][0]["archive_startup"]
    require(startup.get("uid") == [expected_publisher_uid]*4
        and startup.get("gid") == [expected_publisher_gid]*4, "archive process owner differs from registration")
    broker = _verify_broker(edge, binding=expected_handoff_binding,
        uid=expected_publisher_uid, gid=expected_publisher_gid, max_frames=C.BUDGET)
    reads = _index(edge.get("reads"))
    ledger, ledger_sha = _read(cell/"cpu_gt_worker/engine/HANDOFF_ARCHIVE_LEDGER.json")
    require(ledger.get("binding") == expected_handoff_binding and ledger.get("failed_batches") == [],
        "handoff ledger binding differs or has failed batches")
    packets = _index(ledger.get("scoring_packets"))
    worker_requests = _index(timeline.get("requests"))
    targets, targets_sha = _read(cell/"parity_complete_targets/MANIFEST.json")
    target_rows = _index(targets.get("frames"))
    require(set(data["sent"]) == set(packets) == set(broker) == set(reads) == set(target_rows) == set(worker_requests),
        "producer, packet, broker, read, target and sent populations differ")
    records = data["records"]
    roots = {str(Path(r["path"]).parent) for r in records.values()}
    require(len(roots) == 1, "archive shared evidence roots differ")
    original_edge = Path(next(iter(roots)))
    require(original_edge.is_absolute(), "original shared evidence root must be absolute")
    archive_roots = tuple(dict.fromkeys((original_edge, cell.resolve(), (cell/"cpu_gt_worker/engine").resolve())))
    submissions, completions = _verify_archive(ledger["archive"], binding=expected_handoff_binding,
        roots=archive_roots, max_frames=C.BUDGET, preserved_cell=cell)
    require(set(submissions) == set(packets), "admitted scoring/archival work is unaccounted")
    closed, _ = _read(edge_dir/"evaluator_close_result.json")
    require(edge["evaluator_close"] == closed, "edge scope differs from genuine evaluator close")
    pose_rows = _index([r for r in pose.read_archive(edge_dir/"pose_archive")["events"]
        if r.get("kind") == "EDGE_TARGET_RECONCILED"], "gt_identity")
    require(set(pose_rows) == set(packets), "scoring pose population differs")
    joined = []
    for frame, packet in packets.items():
        identity = data["gt"][frame]
        record, observed, read, target = records[frame], broker[frame], reads[frame], target_rows[frame]
        require(record.get("scoring_handoff") == packet and packet.get("identity") == identity
            and worker_requests[frame].get("identity") == identity
            and packet.get("schema") == H.SCHEMA and packet.get("archive_durable") is False
            and all(r.get("identity") == identity for r in (observed, read, target, completions[frame])),
            "producer/broker/archive full scoring identity differs")
        require(packet.get("broker") == dict(status="ADMITTED", binding=expected_handoff_binding,
            identity=identity, packet_sha256=packet["packet_sha256"]), "producer admission ACK differs")
        bundle = T.verify_full_inputs(cell/"parity_complete_targets", expected_identity=identity,
            expected_manifest_sha256=target["manifest_sha256"])
        metadata = T._input_metadata(bundle.inputs)
        reproduced = P.encode_packet(identity=identity, prepared=bundle.prepared,
            avo_document=data["sidecars"][frame], input_metadata=metadata,
            ready_raw_ns=observed["ready_raw_ns"], ready_wall_ns=observed["ready_wall_ns"],
            binding=expected_handoff_binding)
        digest = hashlib.sha256(reproduced).hexdigest()
        require(digest == packet.get("packet_sha256") == observed.get("packet_sha256") == read.get("packet_sha256")
            and len(reproduced) == packet.get("packet_bytes") == observed.get("bytes"),
            "scoring packet differs from exact preserved native target/metadata/AVO bytes")
        require(packet.get("packet_construct_start_raw_ns") == observed["ready_raw_ns"],
            "producer packet construction clock differs")
        _interval(observed["ready_raw_ns"], observed["admitted_raw_ns"], "packet admission")
        _interval(observed["admitted_raw_ns"], packet.get("accepted_response_raw_ns"), "packet admission ACK")
        for name in ("prepared_content_sha256", "input_content_sha256"):
            require(packet.get(name) == read.get(name) == bundle.manifest[name]
                == record.get("complete_target", {}).get(name), "packet and durable input/prepared content differ")
        require(read.get("avo_sha256") == record["sha256"]
            and read.get("input_metadata_sha256") == P.digest(T._canonical(metadata))
            and read.get("admitted_raw_ns") == observed["admitted_raw_ns"]
            and read.get("admitted_wall_ns") == observed["admitted_wall_ns"], "edge actual scoring read differs")
        _interval(read["admitted_raw_ns"], read.get("decoded_raw_ns"), "packet decode")
        receipt = record.get("archive_receipt", {})
        require(receipt.get("durable") is True and receipt.get("completion") == completions[frame]
            and receipt.get("job_id") == submissions[frame]["job_id"], "producer durable receipt differs")
        ready_path = str(Path(record["complete_target"]["directory"])/T.READY_NAME)
        finalized = completions[frame]["finalized_ready"]
        require(len(finalized) == 1 and finalized[0]["path"] == ready_path
            and finalized[0]["sha256"] == record["complete_target"]["manifest_sha256"]
            == target["manifest_sha256"] == bundle.manifest_sha256
            and all(finalized[0][k] == bundle.manifest[k] for k in ("ready_raw_ns", "ready_wall_ns")),
            "finalized archive READY differs from preserved target")
        timing = data["evaluations"][frame].get("timing", {})
        require(timing.get("target_reference_kind") == REFERENCE_KIND
            and timing.get("gt_transport_semantics") == TRANSPORT
            and timing.get("scoring_packet_sha256") == timing.get("target_bundle_sha256") == digest
            and timing.get("prepared_content_sha256") == packet["prepared_content_sha256"]
            and timing.get("input_content_sha256") == packet["input_content_sha256"]
            and timing.get("producer_gt_ready_raw_ns") == observed["ready_raw_ns"]
            and timing.get("packet_admitted_raw_ns") == observed["admitted_raw_ns"]
            and timing.get("packet_admitted_wall_ns") == observed["admitted_wall_ns"]
            and timing.get("packet_decode_completed_raw_ns") == read["decoded_raw_ns"],
            "actual evaluator packet read/timing differs")
        event = pose_rows[frame]
        require(event.get("gt_identity") == identity and event.get("target_reference_kind") == REFERENCE_KIND
            and event.get("gt_transport_semantics") == TRANSPORT
            and event.get("target_manifest_sha256") == event.get("scoring_packet_sha256") == digest,
            "pose references a different packet or wrong reference kind")
        joined.append(dict(identity=identity, scoring_packet_sha256=digest,
            archived_ready_sha256=bundle.manifest_sha256, source_manifest_sha256=target["source_manifest_sha256"],
            avo_sha256=record["sha256"], prepared_content_sha256=packet["prepared_content_sha256"],
            input_content_sha256=packet["input_content_sha256"], archive_job_id=receipt["job_id"]))
    return dict(passed=True, edge_scope_sha256=edge_sha, ledger_sha256=ledger_sha,
        target_manifest_sha256=targets_sha, accepted=len(joined), rejected=0, failed=0, unresolved=0,
        broker_thread_stopped=True, exact_packet_archive_join=joined, latency_qualification_claim=False)


@contextmanager
def install_postflight_reader():
    """Retain all checks, changing two explicitly labeled packet references."""
    from rl_agent.splitfusion_live_parity_runtime_v1 import postflight as retained
    from rl_agent.splitfusion_live_parity_repair_v2 import readiness
    from rl_agent.splitfusion_live_parity_repair_v2.source_ownership import verify_ownership_archive
    original = retained.verify_cell
    source = inspect.getsource(original)
    before = 'target_frames[frame]["manifest_sha256"]'
    require(source.count(before) == 2, "retained target reference verifier seam drift")
    source = source.replace(before, '_packet_reference(data["records"][frame])')
    namespace = dict(original.__globals__, R=readiness, _packet_reference=_packet_reference)
    exec(compile(source, "<v8-retained-parity-with-explicit-packet-reference>", "exec"), namespace)
    replacement = namespace["verify_cell"]
    def verified(cell):
        result = replacement(cell)
        data = retained._population(cell)
        ownership = verify_ownership_archive(Path(cell)/"source_ownership", expected_identities=data["gt"],
            expected_action_identities={frame: row["run4_identity"] for frame,row in data["sent"].items()})
        timing = verify_gt_timing(Path(cell)/"complete_gt_stage_timing", expected_frames=data["gt"])
        return dict(result, exact_source_ownership=ownership,
            complete_gt_stage_timing={k:v for k,v in timing.items() if k != "rows"})
    retained.verify_cell = verified
    try:
        yield
    finally:
        retained.verify_cell = original


def latency_report(cell):
    """Subtract only within the same recorded clock; archive MONOTONIC stays separate."""
    cell = Path(cell)
    from rl_agent.splitfusion_live_parity_runtime_v1 import postflight as retained
    data = retained._population(cell)
    timeline, _ = _read(cell/"CPU_GT_PROCESS_TIMELINE.json")
    ledger, _ = _read(cell/"cpu_gt_worker/engine/HANDOFF_ARCHIVE_LEDGER.json")
    edge, _ = _read(cell/"profiler/edge/EDGE_HANDOFF_SCOPE.json")
    broker, requests = _index(edge["broker"]["ledger"]), _index(timeline["requests"])
    archive = _index(ledger["archive"]["completions"])
    feedback = _index(data["ue"].get("feedback_rows", []), "frame_id")
    events = [json.loads(line) for line in (cell/"source_ownership/events.jsonl").read_bytes().splitlines()]
    admitted = _index([r for r in events if r["kind"] == "SOURCE_ADMITTED"], "frame_id")
    consumed = _index([r for r in events if r["kind"] == "SOURCE_CONSUMED"], "frame_id")
    rows, late, missing = [], [], []
    for frame, decision in data["decisions"].items():
        action = decision["action_open"]
        require(action.get("domain") == "CLOCK_MONOTONIC_RAW", "action-open clock differs")
        opening = action["ns"]
        evaluate = data["evaluations"][frame]
        timing, worker = evaluate["timing"], requests[frame]["worker"]
        item = dict(frame_id=frame,
            gt_owned_queue_ms=_interval(admitted[frame]["observed_raw_ns"], consumed[frame]["observed_raw_ns"], "GT owned queue"),
            gt_worker_service_ms=_interval(worker["start_raw_ns"], worker["end_raw_ns"], "GT worker service"),
            scoring_packet_available_ms=_interval(opening, broker[frame]["admitted_raw_ns"], "scoring availability"),
            evaluation_ms=_interval(timing["evaluator_start_wall_ns"], timing["evaluator_end_wall_ns"], "evaluation wall"),
            packet_to_feedback_emit_ms=_interval(broker[frame]["admitted_wall_ns"], evaluate["emit_wall_ns"], "packet/feedback wall"),
            archive_service_monotonic_ms=_interval(archive[frame]["start_ns"], archive[frame]["end_ns"], "archive service"))
        received = feedback.get(frame)
        if received is None:
            missing.append(frame)
            item.update(feedback_received=False, inclusive_170ms_passed=False)
        else:
            receipt = received.get("receipt", {})
            require(receipt.get("domain") == "CLOCK_MONOTONIC_RAW"
                and received.get("sha256") == evaluate.get("feedback_sha256"), "feedback clock/content differs")
            duration = _interval(opening, receipt.get("ns"), "inclusive action/feedback")
            timely = receipt["ns"] - opening <= 170_000_000
            item.update(feedback_received=True, inclusive_action_feedback_ms=duration, inclusive_170ms_passed=timely)
            if not timely:
                late.append(frame)
        rows.append(item)
    return dict(deadline_ms=170, deadline_inclusive=True, timing_passed=not late and not missing,
        late_frames=late, missing_feedback_frames=missing, rows=rows,
        clocks=dict(gt_queue_service_packet_feedback="CLOCK_MONOTONIC_RAW", evaluation="wall_ns",
            archive_service="CLOCK_MONOTONIC"), cross_clock_subtraction=False,
        profiler_or_gt_cost_subtracted=False)


def population_accounting(cell):
    """Keep failed/rejected/unresolved records visible even when clean gates fail."""
    cell = Path(cell)
    report = dict(accepted=[], rejected=[], failed=[], unresolved=[], errors=[])
    for path, fields in ((cell/"run5_live_avo_gt_summary.json", (("records", "accepted"), ("rejected_records", "rejected"))),
            (cell/"cpu_gt_worker/engine/HANDOFF_ARCHIVE_LEDGER.json", (("failed_batches", "failed"),))):
        try:
            value, _ = _read(path)
            for name, group in fields:
                report[group] = value[name]
        except Exception as exc:
            report["errors"].append(f"{path.name}: {type(exc).__name__}: {exc}")
    partial = cell/"FAILED_ARCHIVE_PARTIAL_PRESERVATION.json"
    if partial.exists():
        value, _ = _read(partial)
        report["unresolved"] = value.get("unresolved", [])
        report["failed_partial_preservation"] = value
    report["counts"] = {name:len(report[name]) for name in ("accepted", "rejected", "failed", "unresolved")}
    report["clean"] = not any(report[name] for name in ("rejected", "failed", "unresolved", "errors"))
    return report


def postflight(output):
    from . import contract as C
    from .host_capture import verify_prospective
    from rl_agent.splitfusion_run5_live_avo_v1 import host_capture as H, verify_live_gt_capture as V
    from rl_agent.splitfusion_live_parity_runtime_v1 import postflight as retained
    output = Path(output).resolve()
    record = verify_prospective(output)
    original = H.postflight
    retained_host = FunctionType(original.__code__, dict(original.__globals__, _verify_prospective=verify_prospective),
        original.__name__, original.__defaults__, original.__closure__)
    old = V.verify_capture
    V.verify_capture = retained.verify_avo_capture
    try:
        retained_host(output)
    finally:
        V.verify_capture = old
    with install_postflight_reader():
        result = retained.postflight(output)
    cells = sorted((output/"capture/cells").iterdir())
    checks = {}
    require(len(cells) == 1, "one registered qualification cell required")
    cell = cells[0]
    expected_sha = C.sha(output/"PROSPECTIVE.json")
    operations = dict(
        optimization=lambda:verify_optimization_scope(cell, expected_sources=record["optimization_sources"],
            expected_worker_sources=record["worker_sources"], expected_prospective_sha256=expected_sha),
        paired_input=lambda:verify_input_scope(cell, expected_prospective_sha256=expected_sha),
        handoff=lambda:verify_handoff_scope(cell, expected_handoff_binding=record["handoff_binding"],
            expected_bootstrap_sources=record["bootstrap_sources"],
            expected_publisher_uid=record["publisher"]["publisher_uid"],
            expected_publisher_gid=record["publisher"]["publisher_gid"]))
    for name, operation in operations.items():
        try:
            checks[name] = operation()
        except Exception as exc:
            checks[name] = dict(passed=False, error=f"{type(exc).__name__}: {exc}")
    accounting = population_accounting(cell)
    try:
        timing = latency_report(cell)
    except Exception as exc:
        timing = dict(timing_passed=False, evidence_valid=False, error=f"{type(exc).__name__}: {exc}")
    integrity = result.get("passed") is True and accounting["clean"] and all(v.get("passed") is True for v in checks.values())
    if timing.get("evidence_valid") is False:
        integrity = False
    summary = dict(schema="splitfusion.gt_handoff_live_qualification.v8", passed=integrity,
        integrity_passed=integrity, timing_passed=timing.get("timing_passed") is True,
        status=("SCORING_HANDOFF_ENGINEERING_PASS__TIMING_REPORTED_SEPARATELY" if integrity
            else "STOP_SCORING_HANDOFF_ENGINEERING_FAILED"), retained_parity=result,
        **checks, accounting=accounting, latency=timing, training_authorized=False,
        deadline_success_improvement_established=False)
    retained.write(output/"OPTIMIZED_POSTFLIGHT.json", summary)
    return summary
