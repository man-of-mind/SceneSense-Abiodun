"""Pinned-image real-entry packet/evaluator proof with one host publisher.

Only EP.main is intercepted. The new stdlib startup entry, registered edge,
broker, real asynchronous evaluator, original pose/scorers and close execute.
The host publishes a synthetic fixture through the existing evidence mount;
this proof starts no route, radio, model inference, CUDA or live READY.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import importlib
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
import threading
import time
import traceback
from unittest import mock

from rl_agent.splitfusion_live_parity_runtime_v7.container_check import (
    IMAGE_ID, REGISTRATION, READY, FORBIDDEN, canonical, sha, write_x, require,
    forbidden, deny_imports, verify_source_manifest, _ticket_and_target,
)

SCHEMA = "splitfusion.parity_runtime.edge_startup_scope.v8"
SOCKET = Path("/work/torch_cache/segmentation_evidence/gt_packet.sock")
CHECKS = ("entry_imported", "registration_checked", "edge_pose_scope",
    "close_evidence_scope", "repaired_evaluation_scope", "retained_dispatch_reached",
    "forbidden_ue_dependencies_absent", "no_cuda_initialized", "no_live_services_started",
    "target_roundtrip_scored", "actual_close_archives_flushed", "actual_startup_entry_exited",
    "root_container_unprivileged_publisher", "credential_checked_shared_inode",
    "scoring_packet_distinct_from_durable_ready", "no_disk_gt_read_during_scoring",
    "actual_evaluator_thread_drained", "actual_broker_drained_and_removed")


@contextmanager
def deny_live_services():
    """Allow only the local handoff socket and its two named actual threads."""
    state = dict(forbidden=[], allowed_threads=[], unix_sockets=0)
    old_socket, old_popen, old_start = socket.socket, subprocess.Popen, threading.Thread.start

    def fail(name):
        state["forbidden"].append(name)
        raise RuntimeError("diagnostic forbids live service/process start: " + name)

    class LocalSocket(old_socket):
        def __init__(self, family=socket.AF_INET, type=socket.SOCK_STREAM, proto=0, fileno=None):
            if family != socket.AF_UNIX or type != socket.SOCK_STREAM or proto != 0:
                fail("non-handoff-socket")
            state["unix_sockets"] += 1
            super().__init__(family, type, proto, fileno)

    def start(thread, *args, **kwargs):
        target = getattr(thread, "_target", None)
        allowed = {
            "exact-gt-scoring-handoff": ("rl_agent.splitfusion_live_gt_handoff_v1.broker", "Server._serve"),
            "run4-evaluator": ("rl_agent.splitfusion_hybrid_sac_live_route_b_v2.phase6_edge_runtime_v2",
                               "Run4EvaluatorV2._run"),
        }
        actual = (getattr(target, "__module__", None), getattr(target, "__qualname__", None))
        if allowed.get(thread.name) != actual or thread.name in state["allowed_threads"]:
            fail("thread:" + thread.name)
        state["allowed_threads"].append(thread.name)
        return old_start(thread, *args, **kwargs)

    socket.socket = LocalSocket
    subprocess.Popen = lambda *args, **kwargs: fail("subprocess")
    threading.Thread.start = start
    try:
        yield state
    finally:
        socket.socket, subprocess.Popen, threading.Thread.start = old_socket, old_popen, old_start


def synthetic_registration():
    from . import contract as C
    root = Path(C.__file__).resolve().parents[2]
    uid, gid = os.getuid(), os.getgid()
    require(uid != 0, "diagnostic registration must be authored by unprivileged host")
    result = dict(id=C.ID, edge_module=C.EDGE, transmitted_budget=C.BUDGET,
        live_metric_contract=C.metrics(), diagnostic_only=True, live_authorization=False,
        claim_scope="CPU_CONTAINER_ENTRY_PACKET_SCORING_DIAGNOSTIC_ONLY",
        handoff_binding=C.handoff_binding(root), bootstrap_sources=C.bootstrap_inventory(root),
        worker_sources=C.worker_inventory(root), publisher_uid=uid, publisher_gid=gid,
        socket_mode="0600", publisher_authentication="SO_PEERCRED_BEFORE_APPLICATION_READ")
    result["registration_sha256"] = C.canonical(result)
    return result


def docker_command(*, repo_root, binding_directory, partition_code, scratch_directory):
    """Reviewable argv only. The caller starts the separate host publisher."""
    paths = [Path(p).resolve(strict=True) for p in
             (repo_root, binding_directory, partition_code, scratch_directory)]
    require(all(p.is_dir() and "," not in str(p) for p in paths), "explicit diagnostic directories required")
    repo, binding, code, scratch = paths
    require(scratch != repo and not scratch.is_relative_to(repo), "diagnostic scratch must be isolated from repository")
    evidence = scratch / "segmentation_evidence"
    require(evidence.is_dir() and not evidence.is_symlink(), "create isolated shared evidence directory first")
    command = ["docker", "run", "--rm", "--pull", "never", "--runtime", "runc",
        "--user", "0:0", "--network", "none", "--read-only", "--cap-drop", "ALL",
        "--cap-add", "CHOWN", "--cap-add", "FOWNER", "--cap-add", "DAC_OVERRIDE",
        "--security-opt", "no-new-privileges", "--cpus", "2", "--pids-limit", "128",
        "--tmpfs", "/tmp:rw,nosuid,size=256m", "--tmpfs", "/work/torch_cache:rw,nosuid,size=64m"]
    for source, destination, readonly in ((repo, "/work/abiodun", True),
            (binding, "/partition_output", True), (code, "/cpu_partition_code", True),
            (scratch, "/diagnostic", False),
            (evidence, "/work/torch_cache/segmentation_evidence", False)):
        command += ["--mount", f"type=bind,src={source},dst={destination}" + (",readonly" if readonly else "")]
    for value in ("CUDA_VISIBLE_DEVICES=", "NVIDIA_VISIBLE_DEVICES=void", "PYTHONDONTWRITEBYTECODE=1",
                  "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "PYTHONPATH=/work/abiodun"):
        command += ["--env", value]
    return command + ["--workdir", "/work/abiodun", "--entrypoint", "python3", IMAGE_ID, "-m",
        "rl_agent.splitfusion_live_parity_runtime_v8.container_check", "run",
        "--source-manifest", "/partition_output/SOURCE_MANIFEST.json", "--report", "/diagnostic/REPORT.json",
        "--repo-root", "/work/abiodun", "--image-id", IMAGE_ID]


def _wait_document(path, *, timeout_s=30.):
    require(type(timeout_s) in (int, float) and 0 < timeout_s <= 60, "bounded diagnostic wait required")
    deadline = time.monotonic() + timeout_s
    path = Path(path)
    while True:
        require(time.monotonic() < deadline, "diagnostic peer did not become ready")
        if path.exists():
            require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 65536,
                    "safe bounded peer document required")
            try:
                return json.loads(path.read_bytes())
            except json.JSONDecodeError:
                # The create-only writer may be between open and its fsync.
                # Incomplete data never becomes authority and stays bounded.
                pass
        time.sleep(.01)


def wait_and_publish(scratch_directory, *, timeout_s=30.):
    """Run outside Docker as the exact registered unprivileged publisher."""
    from rl_agent.splitfusion_live_gt_handoff_v1 import broker as B, scoring_packet as P
    root = Path(scratch_directory).resolve(strict=True)
    request = _wait_document(root / "PUBLISH_REQUEST.json", timeout_s=timeout_s)
    require(os.getuid() != 0 and request.get("publisher_uid") == os.getuid()
            and request.get("publisher_gid") == os.getgid(), "publisher must match the real unprivileged host")
    path = root / "scoring_packet.bin"
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= P.MAX_PACKET_BYTES,
            "safe bounded synthetic packet required")
    payload = path.read_bytes()
    require(P.digest(payload) == request["packet_sha256"], "diagnostic fixture packet drift")
    endpoint = root / "segmentation_evidence/gt_packet.sock"
    info = endpoint.lstat()
    require(stat.S_ISSOCK(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o600
            and (info.st_uid, info.st_gid) == (os.getuid(), os.getgid()), "registered socket owner/mode differs")
    answer = B.publish(endpoint, payload, binding=request["binding"], timeout_s=3.)
    result = dict(passed=True, acknowledgment=answer, publisher_uid=os.getuid(), publisher_gid=os.getgid(),
        socket_proof=dict(device=info.st_dev, inode=info.st_ino, uid=info.st_uid, gid=info.st_gid, mode="0600"))
    write_x(root / "HOST_PUBLISHER.json", result)
    return result


def exercise_scoring(profile, repo_root, checks, registration):
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as E
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
    from rl_agent.splitfusion_quality_feedback_probe_v1 import gt_evidence as G
    from rl_agent.splitfusion_live_parity_runtime_v1 import pose
    from rl_agent.splitfusion_live_parity_repair_v1 import target_transport as T
    from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG
    from rl_agent.splitfusion_live_gt_handoff_v1 import scoring_packet as P, edge as H
    diagnostic = profile.parent
    directory = profile / "diagnostic_targets"
    ticket, expected = _ticket_and_target(directory)
    bundle = T.read_bundle(directory, expected_identity=ticket.gt_identity, timeout_s=0.)
    metadata = json.loads((bundle.directory / "inputs.json").read_bytes())
    avo = json.loads(LG.sidecar_path(directory, ticket.gt_identity).read_bytes())
    avo.update(record_kind="AVO_SCORED_FRAME", native_shape=[720, 1280])
    for row in avo["avo"]["qualified"]:
        row["qualified"] = True
    packet = P.encode_packet(identity=ticket.gt_identity, prepared=bundle.prepared,
        input_metadata=metadata, avo_document=avo, ready_raw_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
        ready_wall_ns=time.time_ns(), binding=registration["handoff_binding"])
    write_x(diagnostic / "scoring_packet.bin", packet)
    write_x(diagnostic / "PUBLISH_REQUEST.json", dict(binding=registration["handoff_binding"],
        packet_sha256=P.digest(packet), publisher_uid=registration["publisher_uid"],
        publisher_gid=registration["publisher_gid"]))
    publisher = _wait_document(diagnostic / "HOST_PUBLISHER.json")
    require(publisher.get("passed") is True and publisher["acknowledgment"]["packet_sha256"] == P.digest(packet),
            "actual host publication missing")
    sent, received = [], threading.Event()
    def send(value):
        sent.append(value)
        received.set()
    worker = E.Run4EvaluatorV2(spec=W.load_run4_quality_spec(repo_root), send=send,
        read_ground_truth=lambda **kw: G.read_ground_truth(directory, **kw), match_distance_m=3.,
        gt_timeout_s=.3, probe_timeout_s=.002, poll_interval_s=.001, evidence_dir=directory)
    with ExitStack() as stack:
        for owner, name in ((T, "read_bundle"), (T, "_read"), (LG, "read_avo_bundle")):
            stack.enter_context(mock.patch.object(owner, name,
                side_effect=AssertionError("durable GT read during packet scoring")))
        worker.start()
        try:
            worker.submit(ticket)
            require(received.wait(3.), "actual evaluator did not return fixture feedback")
        finally:
            result = worker.close(timeout_s=3.)
    checks["no_disk_gt_read_during_scoring"] = True
    checks["actual_evaluator_thread_drained"] = not worker._thread.is_alive() and result["worker_alive"] is False
    feedback, reason = W.quality_feedback(worker._spec, expected.measurement, ticket.envelope)
    require(len(sent) == 1 and sent[0] == W.encode_feedback(feedback, reason)
            and worker.records[0]["measurement"] == expected.measurement, "actual evaluator/feedback numerics differ")
    checks["target_roundtrip_scored"] = True
    timing = worker.records[0]["timing"]
    durable_digest = sha(bundle.directory / "READY.json")
    require(timing["scoring_packet_sha256"] == P.digest(packet)
            and timing["target_bundle_sha256"] == P.digest(packet)
            and timing["target_reference_kind"] == H.REFERENCE_KIND
            and timing["gt_transport_semantics"] == H.TRANSPORT
            and P.digest(packet) != durable_digest, "packet/durable reference semantics conflated")
    checks["scoring_packet_distinct_from_durable_ready"] = True
    require(result["parity_archive_complete"] is True and result["pose_archive_complete"] is True,
            "real prediction and pose archives did not flush")
    prediction = json.loads((profile / "prediction_archive/MANIFEST.json").read_bytes())
    poses = pose.read_archive(profile / "pose_archive")
    require(prediction["frames"] == prediction["evaluated_frames"] == 1
            and poses["manifest"]["counts"].get("EDGE_TARGET_RECONCILED") == 1,
            "actual archives lost diagnostic ticket")
    checks["actual_close_archives_flushed"] = True
    return dict(measurement=expected.measurement, feedback_reason=worker.records[0]["reason"],
        native_prediction_hw=list(ticket.predicted_mask.shape), timing=timing, worker_thread_started=True,
        real_ready_published=False, close_result=result, publisher=publisher,
        packet_sha256=P.digest(packet), durable_fixture_ready_sha256=durable_digest,
        archive_manifests=dict(prediction=sha(profile / "prediction_archive/MANIFEST.json"),
                              pose=sha(profile / "pose_archive/MANIFEST.json")))


def run_check(*, source_manifest, report, repo_root, image_id):
    report, repo_root = Path(report), Path(repo_root)
    require(image_id == IMAGE_ID and os.geteuid() == 0, "exact image and genuine root-container entry required")
    require(not report.exists(), "diagnostic report is create-only")
    checks = dict.fromkeys(CHECKS, False)
    output = dict(schema=SCHEMA, passed=False, image_id=image_id, sources={}, checks=checks,
        cuda_initialized=False, live_started=False, diagnostic_only=True, errors=[], logs={})
    log = report.with_name("container_check.log")
    require(not log.exists() and not READY.exists(), "diagnostic log/READY already exists")
    imports, services = [], dict(forbidden=[], allowed_threads=[])
    old_thread_run, old_thread_start, old_path = threading.Thread.run, threading.Thread.start, list(sys.path)
    try:
        output["sources"] = verify_source_manifest(source_manifest, repo_root)
        output["source_manifest_sha256"] = sha(source_manifest)
        with deny_imports() as imports, deny_live_services() as services:
            from . import contract as C, entries, startup
            from rl_agent.splitfusion_single_gpu_profiler_v1 import edge_profiled_v1 as EP
            from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_prewarm_v2 as PW
            from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as E
            import torch
            shim = importlib.import_module(C.EDGE)
            require(shim.edge_entry is startup.edge_entry, "registered shim does not dispatch the actual startup guard")
            checks["entry_imported"] = True
            output["edge_module"] = C.EDGE
            registration = json.loads(REGISTRATION.read_bytes())
            require(registration.get("diagnostic_only") is True and registration.get("live_authorization") is False
                    and type(registration.get("publisher_uid")) is int and registration["publisher_uid"] > 0,
                    "non-live registration with unprivileged host publisher required")
            output["publisher"] = {key: registration[key] for key in
                ("publisher_uid", "publisher_gid", "socket_mode", "publisher_authentication")}
            output["registration_sha256"] = sha(REGISTRATION)
            profile = report.parent / "profiler"
            profile.mkdir(exist_ok=False)
            old_eval, old_close, old_verify = (E.Run4EvaluatorV2._evaluate_with,
                E.Run4EvaluatorV2.close, E.Run4EdgeProcessorV2.verify)
            old_main, old_installer, old_publish = EP.main, EP.install_edge_seams, PW.publish_ready_after_warmup
            old_init, old_lazy = torch.cuda.init, torch.cuda._lazy_init
            def forbid_cuda(*args, **kwargs):
                services["forbidden"].append("cuda")
                raise RuntimeError("diagnostic forbids CUDA initialization")
            torch.cuda.init = torch.cuda._lazy_init = forbid_cuda
            called = []
            def intercepted(argv=None):
                called.append(list(argv or ()))
                require(len(called) == 1, "retained dispatch repeated")
                checks["registration_checked"] = True
                checks["edge_pose_scope"] = E.Run4EdgeProcessorV2.verify is not old_verify
                checks["close_evidence_scope"] = E.Run4EvaluatorV2.close is not old_close
                checks["repaired_evaluation_scope"] = E.Run4EvaluatorV2._evaluate_with is not old_eval
                checks["retained_dispatch_reached"] = True
                require(all(checks[k] for k in ("edge_pose_scope", "close_evidence_scope", "repaired_evaluation_scope")),
                        "actual combined edge scopes missing")
                output["scoring"] = exercise_scoring(profile, repo_root, checks, registration)
                return 0
            EP.main = intercepted
            try:
                require(shim.edge_entry(["--run4-config", str(report.parent / "edge_config.json"),
                    "--ready-file", str(READY)]) == 0, "actual startup entry returned failure")
                require(len(called) == 1, "retained EP.main was not reached")
            finally:
                EP.main, EP.install_edge_seams = old_main, old_installer
                PW.publish_ready_after_warmup = old_publish
                torch.cuda.init, torch.cuda._lazy_init = old_init, old_lazy
            startup_root = READY.parent / "profiler/startup_v2"
            require((startup_root / "STARTED.json").is_file() and not (startup_root / "FAILED.json").exists()
                    and json.loads((startup_root / "EXITED.json").read_bytes())["returncode"] == 0,
                    "actual stdlib startup lifecycle evidence missing")
            output["startup"] = {}
            for name in ("STARTED.json", "EXITED.json"):
                value = json.loads((startup_root / name).read_bytes())
                write_x(report.parent / ("startup_" + name), value)
                output["startup"][name] = value
            checks["actual_startup_entry_exited"] = True
            state = json.loads((profile / "EDGE_HANDOFF_SCOPE.json").read_bytes())
            closed = json.loads((profile / "evaluator_close_result.json").read_bytes())
            require(state.get("complete") is True and state.get("errors") == []
                    and state.get("durability_claim") is False and closed == output["scoring"]["close_result"],
                    "actual edge handoff did not preserve its genuine close")
            broker = state["broker"]
            checks["actual_broker_drained_and_removed"] = (broker["all_consumed"] is True
                and broker["worker_alive"] is False and broker["failure"] is None
                and broker["admitted"] == broker["consumed"] == 1 and not SOCKET.exists())
            publisher = output["scoring"]["publisher"]
            credentials = broker["ledger"][0]["publisher_credentials"]
            checks["root_container_unprivileged_publisher"] = (
                state["publisher_credential_policy"]["creator_uid"] == 0
                and credentials["uid"] == registration["publisher_uid"] > 0
                and credentials["gid"] == registration["publisher_gid"] and credentials["pid"] == 0)
            checks["credential_checked_shared_inode"] = (broker["socket_proof"] == publisher["socket_proof"]
                and broker["rejected_credentials"] is None
                and broker["credential_policy"]["explicitly_supplied"] is True)
            output["handoff_scope"] = state
            output["cuda_initialized"] = torch.cuda.is_initialized()
            checks["no_cuda_initialized"] = not output["cuda_initialized"]
            checks["no_live_services_started"] = (not services["forbidden"] and not READY.exists()
                and services["allowed_threads"] == ["exact-gt-scoring-handoff", "run4-evaluator"])
            checks["forbidden_ue_dependencies_absent"] = not imports and not any(forbidden(n) for n in sys.modules)
        require(verify_source_manifest(source_manifest, repo_root) == output["sources"], "diagnostic source set changed")
        output["sources_unchanged"] = True
        output["diagnostic_code_sha256"] = sha(__file__)
        output["passed"] = all(checks.values())
        require(output["passed"], "one or more startup/scoring checks failed")
    except BaseException as exc:
        output["errors"].append(dict(type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc()))
    finally:
        threading.Thread.run, threading.Thread.start = old_thread_run, old_thread_start
        sys.path[:] = old_path
        output["forbidden_import_attempts"], output["service_diagnostics"] = imports, services
        output["ready_exists"] = READY.exists()
        write_x(log, output)
        output["logs"] = {log.name: sha(log)}
        write_x(report, output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare-registration", "run", "publish"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scratch-directory", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--image-id")
    args = parser.parse_args(argv)
    if args.operation == "prepare-registration":
        if args.output is None:
            parser.error("--output required")
        write_x(args.output, synthetic_registration())
        return 0
    if args.operation == "publish":
        if args.scratch_directory is None:
            parser.error("--scratch-directory required")
        result = wait_and_publish(args.scratch_directory)
    else:
        if args.source_manifest is None or args.report is None or args.image_id is None:
            parser.error("run requires --source-manifest, --report and --image-id")
        result = run_check(source_manifest=args.source_manifest, report=args.report,
                           repo_root=args.repo_root, image_id=args.image_id)
    print(json.dumps({"passed": result["passed"], "errors": result.get("errors", [])}, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
