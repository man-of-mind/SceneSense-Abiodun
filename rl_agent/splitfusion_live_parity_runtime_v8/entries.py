"""One registered Run-5 qualification with scoring before archival durability."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys

from . import contract as C
from rl_agent.splitfusion_live_gt_handoff_v1 import contract as H
from rl_agent.splitfusion_live_parity_runtime_v1.entries import write_x, evidence_scope

REGISTRATION = Path("/partition_output/registration.json")
SOCKET = Path("/work/torch_cache/segmentation_evidence") / H.SOCKET_NAME


def _worker_proof(attempt, state, sources, bootstrap_sources, binding):
    """Bind genuine two-process drain evidence; never fabricate v7 timelines."""
    attempt = Path(attempt)
    paths = {"scope": attempt / "GT_WORKER_SCOPE.json",
        "drain": attempt / "cpu_gt_worker/engine/HANDOFF_DRAIN.json",
        "ledger": attempt / "cpu_gt_worker/engine/HANDOFF_ARCHIVE_LEDGER.json",
        "timeline": attempt / "CPU_GT_PROCESS_TIMELINE.json"}
    if any(not path.is_file() or path.is_symlink() for path in paths.values()):
        return {"complete": False, "error": "handoff worker/drain/ledger evidence missing"}
    saved, drain, ledger = (json.loads(paths[name].read_bytes()) for name in ("scope", "drain", "ledger"))
    timeline = json.loads(paths["timeline"].read_bytes())
    archive = saved.get("archive") or {}
    context = (saved.get("child_ready") or {}).get("process_context", {})
    try:
        H.validate_context(context, bootstrap_sources)
        context_valid = True
    except ValueError:
        context_valid = False
    checks = (saved == state,
        saved.get("schema") == "splitfusion.gt_scoring_handoff.runtime_scope.v1",
        saved.get("sources") == sources, saved.get("bootstrap_sources") == bootstrap_sources,
        saved.get("complete") is True, saved.get("errors") == [],
        saved.get("joint_drained") is True, saved.get("worker_close_succeeded") is True,
        saved.get("preserved_before_teardown") is True,
        saved.get("pending_receipts_resolved_before_preservation") is True,
        saved.get("new_cpu_processes") == 2, saved.get("new_queues") == 2,
        saved.get("new_python_threads") == 0, saved.get("no_cuda_work") is True,
        saved.get("deferred_evidence") is True, saved.get("scoring_ready_is_not_archive_durable") is True,
        context_valid, drain.get("binding") == binding, ledger.get("binding") == binding,
        drain.get("archive_drained") is True, archive == drain.get("archive"),
        archive.get("failed") is False, archive.get("outstanding_jobs") == 0,
        archive.get("outstanding_bytes") == 0, archive.get("child_returncode") == 0,
        archive.get("completed") == archive.get("submitted"),
        type(archive.get("child_pid")) is int, archive.get("child_pid") != saved.get("child_pid"),
        saved.get("archival_drain_sha256") == C.sha(paths["drain"]),
        saved.get("archival_ledger_sha256") == C.sha(paths["ledger"]),
        saved.get("handoff_ledger") == drain.get("ledger"),
        drain.get("ledger", {}).get("sha256") == C.sha(paths["ledger"]),
        timeline.get("schema") == "splitfusion.gt_handoff_process_timeline.v8",
        timeline.get("sources") == sources, timeline.get("bootstrap_sources") == bootstrap_sources,
        timeline.get("handoff_binding") == binding, timeline.get("complete") is True,
        timeline.get("dropped") == 0, timeline.get("errors") == [])
    return {"complete": all(checks), "scope_sha256": C.sha(paths["scope"]), "state": saved,
        "drain_sha256": C.sha(paths["drain"]), "ledger_sha256": C.sha(paths["ledger"]),
        "timeline_sha256": C.sha(paths["timeline"])}


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
    return {"schema": "splitfusion.parity_runtime.optimized_gt_scope.v8",
        "status": "COMPLETE" if result == 0 and error is None and valid and drained and complete and worker_complete else "FAILED",
        "optimization_sources": dict(sources), "worker_sources": dict(record["worker_sources"]),
        "bootstrap_sources": dict(record["bootstrap_sources"]), "handoff_binding": record["handoff_binding"],
        "prospective_record_sha256": prospective_sha256,
        "optimization": optimization, "worker_drained": drained, "cpu_worker": cpu_worker,
        "ownership_manifest_complete": complete, "returncode": result, "error": error}


def _credentials(record):
    values = tuple(record.get(key) for key in ("publisher_uid", "publisher_gid"))
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("explicit registered publisher UID/GID required")
    return values


def child(argv=None):
    from rl_agent.splitfusion_live_parity_repair_v1 import composition
    from rl_agent.splitfusion_single_gpu_profiler_v1 import child_v1 as BASE
    from rl_agent.splitfusion_live_parity_runtime_v1 import pose
    from rl_agent.splitfusion_live_parity_repair_v2.source_ownership import install_source_ownership
    from rl_agent.splitfusion_live_gt_handoff_v1.runtime import install_cpu_gt_worker
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
    bootstrap_sources, binding = record.get("bootstrap_sources"), record.get("handoff_binding")
    timeout = record.get("worker_timeout_s")
    if (not isinstance(sources, dict) or not sources or not isinstance(worker_sources, dict)
            or not worker_sources or type(timeout) not in (int, float) or not 0 < timeout <= 30.
            or bootstrap_sources != H.sources() or binding != H.binding()
            or record.get("worker_client_factory") != FACTORY):
        raise ValueError("prospectively bound optimization/handoff sources and bounded timeout required")
    _credentials(record)
    archive = InputArchive(max_frames=C.INPUT_MAX_FRAMES, max_bytes=C.INPUT_MAX_BYTES)
    archive_record = None
    optimization = ownership = worker_state = worker_proof = result = error = None
    try:
        with install_capture(archive), install_cpu_gt_optimizations(expected_sources=dict(sources)) as optimization:
            with install_source_ownership(attempt_dir=attempt, max_frames=C.BUDGET) as ownership:
                # Worker shutdown and receipt resolution precede the retained
                # ownership snapshot and original target/source preservation.
                with observe_cpu_worker(attempt_dir=attempt, max_frames=C.BUDGET,
                        expected_sources=worker_sources, expected_bootstrap_sources=bootstrap_sources,
                        handoff_binding=binding,
                        drain_check=lambda: ownership["inspect_drain"]()["worker_drained"]) as observation, \
                        install_cpu_gt_worker(attempt_dir=attempt, max_frames=C.BUDGET,
                        expected_sources=dict(worker_sources), expected_bootstrap_sources=dict(bootstrap_sources),
                        timeout_s=float(timeout), _client_factory=observation["factory"]) as worker_state, \
                        pose.install_ue_pose(attempt_dir=attempt, max_frames=C.BUDGET), \
                        evidence_scope(attempt), preserve_startup(attempt):
                    result = composition.run_ue_child(args, max_frames=C.BUDGET, edge_module=C.EDGE)
        worker_proof = _worker_proof(attempt, worker_state, worker_sources, bootstrap_sources, binding)
        if result == 0 and _scope_report(record, prospective_sha256, optimization, ownership, worker_proof, result, None)["status"] != "COMPLETE":
            raise RuntimeError("GT handoff scope did not close with verified settings, joint drain and preserved evidence")
        if result == 0:
            manifest = archive.flush(Path(attempt) / "paired_input_archive",
                collectors=ownership["collectors"], source_binding=PINS)
            archive_record = {"manifest_sha256": C.sha(Path(attempt) / "paired_input_archive/MANIFEST.json"),
                "frames": len(manifest["frames"]), "retained_bytes": manifest["retained_bytes"],
                "drain": manifest["drain"], "complete": True}
    except BaseException as exc:
        error = {"type": type(exc).__name__, "error": str(exc)}
        raise
    finally:
        try:
            if worker_proof is None:
                worker_proof = _worker_proof(attempt, worker_state, worker_sources, bootstrap_sources, binding)
            write_x(Path(attempt) / "PAIRED_INPUT_SCOPE.json", {
                "schema": "splitfusion.paired_input.scope.v8", "returncode": result,
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
            print(f"GT handoff scope evidence failure: {evidence_error!r}", file=sys.stderr, flush=True)
    return result


def run_edge_only(argv, *, max_frames):
    sys.path.insert(0, "/cpu_partition_code")
    import edge_gate
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_prewarm_v2 as PW
    from rl_agent.splitfusion_single_gpu_profiler_v1 import edge_profiled_v1_ordering as ORDERED
    if max_frames != C.BUDGET:
        raise ValueError("registered edge frame bound required")
    edge_gate.install("/partition_output", PW)
    return ORDERED.main(argv)


@contextmanager
def _full_edge_close():
    """Observe outside the privately composed pose close without replacing it."""
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as E
    klass, original = E.Run4EvaluatorV2, E.Run4EvaluatorV2.close
    def install(state):
        composed = klass.close
        def close(evaluator, timeout_s=None):
            result = composed(evaluator, timeout_s)
            state["evaluator_close"] = dict(result)
            return result
        klass.close = close
    try:
        yield install
    finally:
        klass.close = original


def edge(argv=None):
    from rl_agent.splitfusion_single_gpu_profiler_v1 import edge_profiled_v1 as EP
    from rl_agent.splitfusion_live_parity_repair_v2.runtime import install_prediction_contract
    from rl_agent.splitfusion_live_gt_handoff_v1.edge import install_edge_handoff
    args = list(sys.argv[1:] if argv is None else argv)
    profile = EP.profile_path(args).parent
    registration = json.loads(REGISTRATION.read_bytes())
    root = Path(__file__).resolve().parents[2]
    if (registration.get("id") != C.ID or registration.get("transmitted_budget") != C.BUDGET
            or registration.get("edge_module") != C.EDGE
            or registration.get("live_metric_contract") != C.metrics()
            or registration.get("handoff_binding") != C.handoff_binding(root)
            or registration.get("bootstrap_sources") != C.bootstrap_inventory(root)
            or registration.get("worker_sources") != C.worker_inventory(root)
            or registration.get("registration_sha256") != C.canonical(
                {k: v for k, v in registration.items() if k != "registration_sha256"})):
        raise ValueError("edge registration differs from GT handoff runtime")
    publisher_uid, publisher_gid = _credentials(registration)
    state = error = None
    try:
        with install_prediction_contract(), _full_edge_close() as observe_close:
            with install_edge_handoff(socket_path=SOCKET,
                    binding=registration["handoff_binding"], max_frames=C.BUDGET,
                    archive_directory=profile / "prediction_archive", pose_archive_directory=profile / "pose_archive",
                    publisher_uid=publisher_uid, publisher_gid=publisher_gid) as state:
                observe_close(state)
                return run_edge_only(args, max_frames=C.BUDGET)
    except BaseException as exc:
        error = exc
        raise
    finally:
        if state is not None:
            try:
                # These are the full genuine close results, including retained
                # prediction/pose close wrappers, after broker shutdown.
                write_x(profile / "EDGE_HANDOFF_SCOPE.json", state)
                if isinstance(state.get("evaluator_close"), dict):
                    write_x(profile / "evaluator_close_result.json", state["evaluator_close"])
            except Exception as evidence_error:
                if error is None:
                    raise
                print(f"edge handoff evidence failure: {evidence_error!r}", file=sys.stderr, flush=True)
