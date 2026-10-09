"""One registered CPU-worker check around the unchanged Run-5 live lifecycle."""
import json
from pathlib import Path
import sys

from . import contract as C
from rl_agent.splitfusion_live_parity_runtime_v1.entries import (
    write_x, evidence_scope, edge_close_evidence,
)


def _worker_proof(attempt, state, sources):
    path = Path(attempt) / "GT_WORKER_SCOPE.json"
    if not path.is_file() or path.is_symlink():
        return {"complete": False, "error": "CPU-worker scope evidence missing"}
    saved = json.loads(path.read_bytes())
    timeline_path = Path(attempt) / "CPU_GT_PROCESS_TIMELINE.json"
    if not timeline_path.is_file() or timeline_path.is_symlink():
        return {"complete": False, "error": "CPU-worker process timeline missing"}
    timeline = json.loads(timeline_path.read_bytes())
    checks = (saved == state, saved.get("schema") == "splitfusion.cpu_gt_worker.runtime_scope.v4",
        saved.get("sources") == sources, saved.get("complete") is True,
        saved.get("errors") == [], saved.get("joint_drained") is True,
        saved.get("worker_close_succeeded") is True, saved.get("preserved_before_teardown") is True,
        saved.get("new_cpu_processes") == 1, saved.get("new_queues") == 0,
        saved.get("new_python_threads") == 0, saved.get("no_cuda_work") is True,
        timeline.get("sources") == sources, timeline.get("complete") is True,
        timeline.get("dropped") == 0, timeline.get("errors") == [])
    return {"complete": all(checks), "scope_sha256": C.sha(path), "state": saved,
        "timeline_sha256": C.sha(timeline_path)}


def _scope_report(record, prospective_sha256, optimization, ownership, cpu_worker, result, error):
    drain = ownership["inspect_drain"]() if ownership is not None else {}
    manifest = ownership.get("manifest") if ownership is not None else None
    sources = record["optimization_sources"]
    valid = (optimization is not None and optimization.get("sources") == sources
        and optimization.get("retained_settings_unchanged") is True
        and optimization.get("settings_before") == optimization.get("settings_after")
        and optimization.get("source_integrity_error") is None)
    drained = drain.get("worker_drained") is True
    complete = isinstance(manifest, dict) and manifest.get("complete") is True
    worker_complete = isinstance(cpu_worker, dict) and cpu_worker.get("complete") is True
    return {"schema": "splitfusion.parity_runtime.optimized_gt_scope.v7",
        "status": "COMPLETE" if result == 0 and error is None and valid and drained and complete and worker_complete else "FAILED",
        "optimization_sources": dict(sources), "worker_sources": dict(record["worker_sources"]),
        "prospective_record_sha256": prospective_sha256,
        "optimization": optimization, "worker_drained": drained, "cpu_worker": cpu_worker,
        "ownership_manifest_complete": complete, "returncode": result, "error": error}


def child(argv=None):
    from rl_agent.splitfusion_live_parity_repair_v1 import composition
    from rl_agent.splitfusion_single_gpu_profiler_v1 import child_v1 as BASE
    from rl_agent.splitfusion_live_parity_runtime_v1 import pose
    from rl_agent.splitfusion_live_parity_repair_v2.source_ownership import install_source_ownership
    from rl_agent.splitfusion_live_gt_optimization_v4.runtime import install_cpu_gt_worker
    from .worker_observer import observe_cpu_worker, FACTORY
    from .optimization_scope import install_cpu_gt_optimizations
    from .startup import preserve_startup
    from .host_capture import verify_prospective
    from rl_agent.splitfusion_paired_quality_replay_v1.capture import InputArchive
    from rl_agent.splitfusion_paired_quality_replay_v1.scope import install_capture, PINS
    args = list(sys.argv[1:] if argv is None else argv)
    attempt = BASE.attempt_dir_from_argv(args)
    config = BASE.load_config(attempt)
    if config is None or config.get("qualification_id") != C.ID:
        raise ValueError("enabled, sealed parity profiler configuration required")
    if C.canonical({k: v for k, v in config.items() if k != "sha256"}) != config.get("sha256"):
        raise ValueError("profiler configuration digest mismatch")
    capture = Path(attempt).resolve().parent.parent
    record = verify_prospective(capture.parent)
    prospective_sha256 = C.sha(capture.parent / "PROSPECTIVE.json")
    if record["budget"] != C.BUDGET:
        raise ValueError("child archive bound differs from registration")
    sources, worker_sources = record.get("optimization_sources"), record.get("worker_sources")
    timeout = record.get("worker_timeout_s")
    if (not isinstance(sources, dict) or not sources or not isinstance(worker_sources, dict)
            or not worker_sources or type(timeout) not in (int, float) or not 0 < timeout <= 30.
            or record.get("worker_client_factory") != FACTORY):
        raise ValueError("prospectively bound optimization/worker sources and bounded timeout required")
    archive = InputArchive(max_frames=C.INPUT_MAX_FRAMES, max_bytes=C.INPUT_MAX_BYTES)
    archive_record = None
    optimization = ownership = worker_state = worker_proof = result = error = None
    try:
        with install_capture(archive), install_cpu_gt_optimizations(expected_sources=dict(sources)) as optimization:
            with install_source_ownership(attempt_dir=attempt, max_frames=C.BUDGET) as ownership:
                # Inner worker scope aborts/joins BEFORE ownership snapshots its
                # drain on failure. No parent complete-GT timing scope exists.
                with observe_cpu_worker(attempt_dir=attempt, max_frames=C.BUDGET, expected_sources=worker_sources,
                        drain_check=lambda: ownership["inspect_drain"]()["worker_drained"]) as observation, \
                        install_cpu_gt_worker(attempt_dir=attempt, max_frames=C.BUDGET,
                        expected_sources=dict(worker_sources), timeout_s=float(timeout),
                        _client_factory=observation["factory"]) as worker_state, \
                        pose.install_ue_pose(attempt_dir=attempt, max_frames=C.BUDGET), \
                        evidence_scope(attempt), preserve_startup(attempt):
                    result = composition.run_ue_child(args, max_frames=C.BUDGET, edge_module=C.EDGE)
        worker_proof = _worker_proof(attempt, worker_state, worker_sources)
        if result == 0 and _scope_report(record, prospective_sha256, optimization, ownership, worker_proof, result, None)["status"] != "COMPLETE":
            raise RuntimeError("CPU GT scope did not close with verified settings, joint drain and preserved evidence")
        if result == 0:
            manifest = archive.flush(Path(attempt)/"paired_input_archive",
                collectors=ownership["collectors"], source_binding=PINS)
            archive_record = {"manifest_sha256": C.sha(Path(attempt)/"paired_input_archive/MANIFEST.json"),
                "frames": len(manifest["frames"]), "retained_bytes": manifest["retained_bytes"],
                "drain": manifest["drain"], "complete": True}
    except BaseException as exc:
        error = {"type": type(exc).__name__, "error": str(exc)}
        raise
    finally:
        try:
            if worker_proof is None:
                worker_proof = _worker_proof(attempt, worker_state, worker_sources)
            write_x(Path(attempt)/"PAIRED_INPUT_SCOPE.json", {
                "schema": "splitfusion.paired_input.scope.v7", "returncode": result,
                "error": error, "complete": result == 0 and error is None and archive_record is not None,
                "max_frames": C.INPUT_MAX_FRAMES, "max_bytes": C.INPUT_MAX_BYTES,
                "observer_errors": archive.errors, "observer_error_count": archive.error_count,
                "policy_opportunities": archive.policy_opportunities, "archive": archive_record,
                "prospective_record_sha256": prospective_sha256,
                "quality_diagnostic_only": True, "runtime_fix_claim": False})
            write_x(Path(attempt) / "GT_OPTIMIZATION_SCOPE.json",
                _scope_report(record, prospective_sha256, optimization, ownership, worker_proof, result, error))
        except Exception as evidence_error:
            if error is None:
                raise
            print(f"CPU GT scope evidence failure: {evidence_error!r}", file=sys.stderr, flush=True)
    return result


def run_edge_only(argv, *, max_frames):
    sys.path.insert(0, "/cpu_partition_code")
    import edge_gate
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_prewarm_v2 as PW
    from rl_agent.splitfusion_single_gpu_profiler_v1 import edge_profiled_v1 as EP
    from rl_agent.splitfusion_single_gpu_profiler_v1 import edge_profiled_v1_ordering as ORDERED
    from rl_agent.splitfusion_live_parity_repair_v1 import runtime
    edge_gate.install("/partition_output", PW)
    with runtime.install_edge_evaluation(
            archive_directory=EP.profile_path(argv).parent / "prediction_archive", max_frames=max_frames):
        return ORDERED.main(argv)


def edge(argv=None):
    from rl_agent.splitfusion_single_gpu_profiler_v1 import edge_profiled_v1 as EP
    from rl_agent.splitfusion_live_parity_runtime_v1 import pose
    from rl_agent.splitfusion_live_parity_repair_v2.runtime import install_prediction_contract
    args = list(sys.argv[1:] if argv is None else argv)
    profile = EP.profile_path(args).parent
    registration = json.loads(Path("/partition_output/registration.json").read_text())
    if (registration.get("id") != C.ID or registration.get("transmitted_budget") != C.BUDGET
            or registration.get("edge_module") != C.EDGE
            or registration.get("live_metric_contract") != C.metrics()
            or registration.get("registration_sha256") != C.canonical(
                {k: v for k, v in registration.items() if k != "registration_sha256"})):
        raise ValueError("edge registration differs from CPU GT runtime")
    with install_prediction_contract(), \
            pose.install_edge_pose(archive_directory=profile / "pose_archive", max_frames=C.BUDGET), \
            edge_close_evidence(profile):
        return run_edge_only(args, max_frames=C.BUDGET)
