"""Additive GT optimization scope proof; retained v2 correctness gates remain."""
import hashlib
import json
from pathlib import Path


SCHEMA = "splitfusion.parity_runtime.optimized_gt_scope.v4"
SETTING_FIELDS = frozenset(("environment", "calling_thread_affinity", "torch_threads",
                            "torch_interop_threads", "opencv_threads"))
ENV_FIELDS = frozenset(("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                       "CUDA_VISIBLE_DEVICES", "CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"))


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


def verify_optimization_scope(cell, *, expected_sources, expected_prospective_sha256):
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
            and scope.get("prospective_record_sha256") == expected_prospective_sha256,
            "optimization scope differs from prospective binding")
    require(scope.get("worker_drained") is True and scope.get("ownership_manifest_complete") is True,
            "optimization scope exited before verified GT drain")
    state = scope.get("optimization", {})
    require(state.get("schema") == "splitfusion.cpu_gt_optimization.scope.v1"
            and state.get("sources") == expected_sources, "inner optimization source binding differs")
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
    return {"passed": True, "schema": SCHEMA, "scope_sha256": scope_sha,
            "ownership_manifest_sha256": ownership_sha, "optimization_sources": dict(expected_sources),
            "retained_configuration_unchanged": True, "worker_drained": True,
            "cuda_lifecycle_before": state["cuda_lifecycle_before"],
            "cuda_lifecycle_after": state["cuda_lifecycle_after"],
            "latency_qualification_claim": False}
