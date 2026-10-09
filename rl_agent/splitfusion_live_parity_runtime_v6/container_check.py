"""Fresh-process edge startup/scoring proof; never starts a live service.

Run in the registered image with a read-only staged repository, read-only
diagnostic registration and partition code, --network none, no GPU and an
isolated writable diagnostic directory. The retained EP.main is the sole
entry interception: registration, pose, evaluator and ordering scopes execute.
Synthetic targets test dependencies and invariants, not live perception quality.
"""
from __future__ import annotations

import argparse
import builtins
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import traceback

SCHEMA = "splitfusion.parity_runtime.edge_startup_scope.v6"
IMAGE_ID = "sha256:2be62d533b8077ceecab5455d5377f2f952b6a50d43ff8c04dc89ce18027d6ba"
REGISTRATION = Path("/partition_output/registration.json")
READY = Path("/work/torch_cache/ready.json")
FORBIDDEN = ("scipy", "carla", "rl_agent.splitfusion_live_parity_repair_v1.composition",
             "rl_agent.splitfusion_live_parity_repair_v1.gt_bridge",
             "rl_agent.splitfusion_live_parity_repair_v1.source_geometry")
CHECKS = ("entry_imported", "registration_checked", "edge_pose_scope", "close_evidence_scope",
          "repaired_evaluation_scope", "retained_dispatch_reached", "forbidden_ue_dependencies_absent",
          "no_cuda_initialized", "no_services_started", "target_roundtrip_scored",
          "actual_close_archives_flushed")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_x(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(value if isinstance(value, bytes) else canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def forbidden(name):
    return any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN)


@contextmanager
def deny_imports():
    """Catch eager, lazy, relative and already-cached forbidden imports."""
    require(not any(forbidden(name) for name in sys.modules), "diagnostic requires a fresh Python process")
    attempted = []
    original = builtins.__import__
    def check(name):
        if forbidden(name):
            attempted.append(name)
            raise ModuleNotFoundError(f"diagnostic forbids dependency: {name}")
    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        absolute = name
        if level:
            absolute = importlib.util.resolve_name("." * level + name, (globals or {}).get("__package__", ""))
        check(absolute)
        for part in fromlist or ():
            if part != "*":
                check(absolute + "." + part)
        return original(name, globals, locals, fromlist, level)
    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            check(fullname)
            return None
    finder = Finder()
    builtins.__import__ = guarded
    sys.meta_path.insert(0, finder)
    try:
        yield attempted
    finally:
        builtins.__import__ = original
        sys.meta_path.remove(finder)


@contextmanager
def deny_services():
    """No sockets, external commands or started Python threads in this proof."""
    attempted = []
    old_socket, old_popen, old_start = socket.socket, subprocess.Popen, threading.Thread.start
    def fail(name):
        attempted.append(name)
        raise RuntimeError(f"diagnostic forbids service/process start: {name}")
    class NoSocket(old_socket):
        def __new__(cls, *args, **kwargs):
            fail("socket")
    def no_start(thread, *args, **kwargs):
        fail("thread:" + thread.name)
    socket.socket = NoSocket
    subprocess.Popen = lambda *args, **kwargs: fail("subprocess")
    threading.Thread.start = no_start
    try:
        yield attempted
    finally:
        socket.socket, subprocess.Popen, threading.Thread.start = old_socket, old_popen, old_start


def synthetic_registration():
    """Host precursor only: exact edge validator inputs, zero live authority."""
    from . import contract as C
    result = {"id": C.ID, "edge_module": C.EDGE, "transmitted_budget": C.BUDGET, "live_metric_contract": C.metrics(),
              "diagnostic_only": True, "live_authorization": False,
              "claim_scope": "CPU_CONTAINER_IMPORT_SCORING_DIAGNOSTIC_ONLY"}
    result["registration_sha256"] = C.canonical(result)
    return result


def verify_source_manifest(path, repo_root):
    document = json.loads(Path(path).read_bytes())
    sources = document["sources"]
    require(isinstance(sources, dict) and bool(sources), "nonempty host-produced source inventory required")
    root = Path(repo_root).resolve(strict=True)
    for name, digest in sources.items():
        item = root / name
        require(not Path(name).is_absolute() and ".." not in Path(name).parts
                and item.is_file() and not item.is_symlink() and item.resolve().is_relative_to(root),
                "unsafe source inventory path")
        require(isinstance(digest, str) and len(digest) == 64 and sha(item) == digest,
                f"staged diagnostic source changed: {name}")
    return sources


def docker_command(*, repo_root, binding_directory, partition_code, scratch_directory):
    """Return reviewable argv only. This module never executes Docker."""
    paths = [Path(p).resolve(strict=True) for p in (repo_root, binding_directory, partition_code, scratch_directory)]
    require(all(p.is_dir() and "," not in str(p) for p in paths), "explicit diagnostic directories required")
    repo, binding, code, scratch = paths
    require(scratch != repo and not scratch.is_relative_to(repo), "diagnostic scratch must be isolated from repository")
    command = ["docker", "run", "--rm", "--pull", "never", "--runtime", "runc",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--network", "none", "--read-only", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", "--cpus", "2", "--pids-limit", "128",
        "--tmpfs", "/tmp:rw,nosuid,size=256m", "--tmpfs", "/work/torch_cache:rw,nosuid,size=64m"]
    for source, destination, readonly in ((repo, "/work/abiodun", True), (binding, "/partition_output", True),
                                         (code, "/cpu_partition_code", True), (scratch, "/diagnostic", False)):
        command += ["--mount", f"type=bind,src={source},dst={destination}" + (",readonly" if readonly else "")]
    for value in ("CUDA_VISIBLE_DEVICES=", "NVIDIA_VISIBLE_DEVICES=void", "PYTHONDONTWRITEBYTECODE=1",
                  "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "PYTHONPATH=/work/abiodun"):
        command += ["--env", value]
    return command + ["--workdir", "/work/abiodun", "--entrypoint", "python3", IMAGE_ID, "-m",
        "rl_agent.splitfusion_live_parity_runtime_v6.container_check", "run",
        "--source-manifest", "/partition_output/SOURCE_MANIFEST.json",
        "--report", "/diagnostic/REPORT.json", "--repo-root", "/work/abiodun", "--image-id", IMAGE_ID]


def _ticket_and_target(directory):
    """Prepared-target fixture entering exactly where the real edge receives GT."""
    import cv2
    import numpy as np
    from rl_agent.splitfusion_live_dispatch_v1.frame_context import build_frame_context_v1
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import continuous_execution_v2 as X
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as E
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as MP, run4_live_wire_v2 as W
    from rl_agent.splitfusion_live_parity_runtime_v1 import pose
    from rl_agent.splitfusion_live_parity_repair_v1 import target_transport as T, evaluation as V
    from rl_agent.splitfusion_live_target_parity_v1 import adapter as A
    from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG
    context = build_frame_context_v1(stream_id="container_cpu_diagnostic", frame_id=17,
        sequence_id=1, capture_timestamp_ns=1790000000000000000, ego_world_x=1., ego_world_y=2.,
        ego_world_z=.1, ego_world_pitch=1., ego_world_yaw=23., ego_world_roll=2.)
    inner = b"CPU_DIAGNOSTIC_NOT_MODEL_FEATURES"
    envelope = X.ExecutionEnvelopeV3(mode_id=11, q_e4=9800, keep_count=21126, anchor_action_id=71,
        reward_requested=True, session_uuid="00000000-0000-4000-8000-00000000c0dd",
        controller_lineage_sha256="a"*64, decision_seq=0, ticket_seq=0, frame_id=17,
        tensor_seq=1, capture_timestamp_ns=context.capture_timestamp_ns,
        execution_bundle_sha256="b"*64, inner_payload_sha256=hashlib.sha256(inner).hexdigest(), inner_payload=inner)
    envelope, context = W.unpack_sfd4(W.pack_sfd4(envelope, context))
    identity = MP.gt_identity(run_id="cpu_diagnostic", cell_id="cell", stream_id=context.stream_id,
        frame_id=17, anchor_action_id=71, anchor_profile_id="diagnostic_profile",
        capture_timestamp_ns=context.capture_timestamp_ns)
    sample = "cpu_diagnostic_frame17"
    frame = dict(sample_id=sample, experiment_id="cpu_diagnostic", frame_id=17,
        camera_width=1280, camera_height=720, mask_path="mask.png", instance_raw_path="raw.png",
        camera_inverse_matrix_json=json.dumps(np.linalg.inv(pose.context_camera(context)).tolist()))
    provenance = dict(schema=A.SCHEMA, scope=T.LIVE_SCOPE, sample_id=sample,
        complete=dict.fromkeys(A.COMPLETENESS_KEYS, True), capture_identity=identity,
        evidence_sha256={"synthetic_diagnostic_not_live_sources": "c"*64})
    inputs = A.CompleteTargetInputs(frame, "val", np.zeros((720, 1280, 4), np.uint8),
        np.zeros((720, 1280), np.uint8), np.zeros((720, 1280), np.uint8), [], [], [],
        {key: [] for key in A.QUARANTINE_KEYS}, provenance)
    inputs = replace(inputs, provenance={**provenance, "input_content_sha256": A.input_content_sha256(inputs)})
    labels = np.zeros((432, 768), np.uint8)
    labels[30:70, 40:90] = 1
    labels[90:120, 130:150] = 2
    ignore = np.zeros_like(labels)
    seg = labels.copy(); seg[0:3] = 255
    boxes = [dict(sample_id=sample, frame_id=17, label="vehicle", contract_state="positive",
        object_world_x=10., object_world_y=2., gt_bbox_x=66., gt_bbox_y=50., gt_bbox_w=80.,
        gt_bbox_h=60., gt_bbox_area_px=4800., source_identity="diagnostic_vehicle")]
    encoded_seg, encoded_ignore = (cv2.imencode(".png", a)[1].tobytes() for a in (seg, ignore))
    target = dict(contract=A.CONTRACT, provenance=inputs.provenance, live_qualified=False,
        segmentation=seg, object_ignore=ignore, segmentation_png=encoded_seg, object_ignore_png=encoded_ignore,
        tables={"object_boxes.csv": boxes, "object_ignore_regions.csv": [],
                "target_manifest.csv": [{"sample_id": sample}]})
    prepared = V.restore_prepared_target(frame_id=17, sample_id=sample, target=target,
        content_sha256=V._target_digest(17, sample, target))
    T.write_bundle(directory, identity=identity, inputs=inputs, prepared=prepared)
    restored = T.verify_full_inputs(directory, expected_identity=identity)
    require(restored.native_inputs_verified and restored.prepared.content_sha256 == prepared.content_sha256,
            "synthetic target roundtrip differs")
    predictions = []
    for cls, x, y in (("vehicle", 10., 2.), ("person", 15., 4.)):
        predictions.append(dict(class_name=cls, frame_id=17, score=.8, world_x=x, world_y=y, world_z=1.,
            size_x=2., size_y=1., size_z=1.5, yaw_sin=0., yaw_cos=1., center_x_px=65., center_y_px=50.,
            bbox_x0=40., bbox_y0=30., bbox_x1=90., bbox_y1=70.))
    avo = {"frame_id": 17, "threshold": .65, "qualified": [{"frame_id": 17, "gt_actor_id": "person1",
        "world_x": 15., "world_y": 4., "avo": .8}], "structural_ignored": []}
    write_x(LG.sidecar_path(directory, identity), {"schema": LG.SCHEMA, "mask_contract": LG.MASK_CONTRACT,
        "identity": identity, "source_frame_id": 17, "avo": avo, "ready_wall_ns": 1790000000000000001})
    native_labels = cv2.resize(labels, (1280, 720), interpolation=cv2.INTER_NEAREST)
    ticket = E.EvaluationTicketV2(envelope=envelope, context=context, records=tuple(predictions),
        predicted_mask=native_labels, gt_identity=identity, enqueued_wall_ns=1790000000000000002)
    expected = V.evaluate_prepared(identity=MP.run4_identity(envelope), prepared=prepared,
        predicted_labels=native_labels, predictions=predictions, avo=avo)
    require(expected.measurement["loc_vehicle_tp"] == expected.measurement["loc_person_tp"] == 1,
            "real lazy localization scorers did not match synthetic positives")
    require(expected.measurement["seg_vehicle_intersection_pixels"] > 0
            and expected.measurement["seg_person_intersection_pixels"] > 0,
            "real original segmentation scorer differed")
    return ticket, expected


def exercise_scoring(profile, repo_root, checks):
    """Drive the real evaluator synchronously; never start its worker thread."""
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as E
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
    from rl_agent.splitfusion_quality_feedback_probe_v1 import gt_evidence as G
    from rl_agent.splitfusion_live_parity_runtime_v1 import pose
    ticket, expected = _ticket_and_target(profile / "diagnostic_targets")
    sent = []
    worker = E.Run4EvaluatorV2(spec=W.load_run4_quality_spec(repo_root), send=sent.append,
        read_ground_truth=lambda **kw: G.read_ground_truth(profile / "diagnostic_targets", **kw),
        match_distance_m=3., gt_timeout_s=.02)
    class UnstartedThread:
        # Only lifecycle I/O is substituted. The actual close wrappers and all
        # target/scorer/pose/prediction methods run unchanged.
        def join(self, timeout=None):
            pass
        def is_alive(self):
            return False
    worker._thread = UnstartedThread()
    worker.submit(ticket)
    owned = worker._queue.get_nowait()
    worker.evaluate(owned)
    require(len(sent) == 1 and worker.records[0]["measurement"] == expected.measurement,
            "real composed evaluator did not score the transported target")
    checks["target_roundtrip_scored"] = True
    result = worker.close()
    require(result["worker_alive"] is False and result["parity_archive_complete"] is True
            and result["pose_archive_complete"] is True, "actual archive close wrappers failed")
    require(json.loads((profile/"evaluator_close_result.json").read_bytes()) == result,
            "close evidence scope did not retain actual result")
    predictions = json.loads((profile/"prediction_archive"/"MANIFEST.json").read_bytes())
    poses = pose.read_archive(profile/"pose_archive")
    require(predictions["frames"] == predictions["evaluated_frames"] == 1
            and poses["manifest"]["counts"].get("EDGE_TARGET_RECONCILED") == 1,
            "actual scorer archives do not contain the diagnostic ticket")
    # Cover the formerly missing native writer -> evaluation -> close -> reader
    # boundary, not just independent small-mask and native-scoring unit tests.
    from rl_agent.splitfusion_live_parity_repair_v2 import readiness
    from rl_agent.splitfusion_live_parity_repair_v2.prediction_archive import PredictionArchive
    from rl_agent.splitfusion_live_parity_repair_v1 import target_transport as T
    target_path = T.bundle_path(profile / "diagnostic_targets", ticket.gt_identity)
    verified = readiness.verify_offline_evidence(
        target_directory=profile / "diagnostic_targets",
        target_descriptors=[{"identity": dict(ticket.gt_identity), "manifest_sha256": sha(target_path / "READY.json")}],
        archive_directory=profile / "prediction_archive",
        expected_archive_manifest_sha256=sha(profile / "prediction_archive/MANIFEST.json"),
        evaluator_result=result, expected_ticket_identities=[PredictionArchive.identity(ticket)])
    require(verified["archived_predictions"] == verified["scored_predictions"] == 1,
            "native archive did not pass the actual offline verifier")
    checks["actual_close_archives_flushed"] = True
    return {"measurement": expected.measurement, "feedback_reason": worker.records[0]["reason"],
            "native_prediction_hw": list(ticket.predicted_mask.shape), "reader_verification": verified,
            "worker_thread_started": False, "real_ready_published": False,
            "close_result": result,
            "archive_manifests": {"prediction": sha(profile/"prediction_archive"/"MANIFEST.json"),
                                  "pose": sha(profile/"pose_archive"/"MANIFEST.json")}}


def run_check(*, source_manifest, report, repo_root, image_id):
    """Container execution; a host launcher independently binds image identity."""
    report, repo_root = Path(report), Path(repo_root)
    require(image_id == IMAGE_ID, "diagnostic must use the exact registered image")
    require(not report.exists(), "diagnostic report is create-only")
    checks = dict.fromkeys(CHECKS, False)
    output = {"schema": SCHEMA, "passed": False, "image_id": image_id, "sources": {},
              "checks": checks, "cuda_initialized": False, "live_started": False,
              "diagnostic_only": True, "errors": [], "logs": {}}
    log = report.with_name("container_check.log")
    require(not log.exists() and not READY.exists(), "diagnostic log/READY already exists")
    imports, services = [], []
    old_thread_run, old_thread_start, old_path = threading.Thread.run, threading.Thread.start, list(sys.path)
    ep = pw = None
    try:
        output["sources"] = verify_source_manifest(source_manifest, repo_root)
        output["source_manifest_sha256"] = sha(source_manifest)
        with deny_imports() as imports, deny_services() as services:
            from . import contract as C, entries
            from rl_agent.splitfusion_single_gpu_profiler_v1 import edge_profiled_v1 as EP
            from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_prewarm_v2 as PW
            from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as E
            from rl_agent.splitfusion_live_parity_repair_v1 import runtime
            import torch
            checks["entry_imported"] = True
            output["edge_module"] = C.EDGE
            registration = json.loads(REGISTRATION.read_bytes())
            require(registration.get("diagnostic_only") is True and registration.get("live_authorization") is False,
                    "container diagnostic requires its non-live synthetic registration")
            output["registration_sha256"] = sha(REGISTRATION)
            profile = report.parent / "profiler"
            profile.mkdir(exist_ok=False)
            old_eval, old_close, old_scope = E.Run4EvaluatorV2._evaluate_with, E.Run4EvaluatorV2.close, runtime.install_edge_evaluation
            ep, old_main, old_installer = EP, EP.main, EP.install_edge_seams
            pw, old_publish = PW, PW.publish_ready_after_warmup
            old_init, old_lazy = torch.cuda.init, torch.cuda._lazy_init
            def forbid_cuda(*args, **kwargs):
                services.append("cuda")
                raise RuntimeError("diagnostic forbids CUDA initialization")
            torch.cuda.init = torch.cuda._lazy_init = forbid_cuda
            called = []
            def intercepted(argv=None):
                called.append(list(argv or ()))
                require(len(called) == 1, "retained dispatch repeated")
                checks["registration_checked"] = True  # actual entries.edge reached its delegate
                checks["edge_pose_scope"] = runtime.install_edge_evaluation is not old_scope
                checks["close_evidence_scope"] = E.Run4EvaluatorV2.close is not old_close
                checks["repaired_evaluation_scope"] = E.Run4EvaluatorV2._evaluate_with is not old_eval
                checks["retained_dispatch_reached"] = True
                require(all(checks[k] for k in ("edge_pose_scope", "close_evidence_scope", "repaired_evaluation_scope")),
                        "actual nested edge scopes missing")
                output["scoring"] = exercise_scoring(profile, repo_root, checks)
                return 0
            EP.main = intercepted
            try:
                require(entries.edge(["--run4-config", str(report.parent/"edge_config.json"),
                                      "--ready-file", str(READY)]) == 0, "actual edge entry returned failure")
                require(len(called) == 1, "retained EP.main was not reached")
            finally:
                EP.main, EP.install_edge_seams = old_main, old_installer
                PW.publish_ready_after_warmup = old_publish
                torch.cuda.init, torch.cuda._lazy_init = old_init, old_lazy
            output["cuda_initialized"] = torch.cuda.is_initialized()
            checks["no_cuda_initialized"] = not output["cuda_initialized"]
            checks["no_services_started"] = not services and not READY.exists()
            checks["forbidden_ue_dependencies_absent"] = not imports and not any(forbidden(n) for n in sys.modules)
        require(verify_source_manifest(source_manifest, repo_root) == output["sources"], "diagnostic source set changed")
        output["sources_unchanged"] = True
        output["diagnostic_code_sha256"] = sha(__file__)
        output["passed"] = all(checks.values())
        require(output["passed"], "one or more startup/scoring checks failed")
    except BaseException as exc:
        output["errors"].append({"type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
    finally:
        # The real partition installer wraps these globals; no service starts,
        # but the diagnostic must restore its process before writing proof.
        threading.Thread.run, threading.Thread.start = old_thread_run, old_thread_start
        sys.path[:] = old_path
        output["forbidden_import_attempts"], output["service_start_attempts"] = imports, services
        output["ready_exists"] = READY.exists()
        write_x(log, output)
        output["logs"] = {log.name: sha(log)}
        write_x(report, output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare-registration", "run"))
    parser.add_argument("--output", type=Path)
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
    if args.source_manifest is None or args.report is None or args.image_id is None:
        parser.error("run requires --source-manifest, --report and --image-id")
    result = run_check(source_manifest=args.source_manifest, report=args.report,
                       repo_root=args.repo_root, image_id=args.image_id)
    print(json.dumps({"passed": result["passed"], "report": str(args.report), "errors": result["errors"]}, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
