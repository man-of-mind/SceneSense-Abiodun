"""Parameterized fixed child/edge entries over the retained lifecycle."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys

from .source_verify import verify_launcher_sources, verify_registered_sources


def registration(path, root, C):
    value = json.loads(Path(path).read_bytes())
    if (value.get("id") != C.ID or value.get("transmitted_budget") != 1000
            or value.get("g1_boundary_budget") != 1000
            or value.get("exact_gt_limits") != C.limits(1000)
            or value.get("child_module") != C.CHILD or value.get("edge_module") != C.EDGE
            or value.get("live_metric_contract") != C.metrics()
            or value.get("registration_sha256") != C.canonical(
                {key: item for key, item in value.items() if key != "registration_sha256"})):
        raise ValueError("M1 fixed entry requires its exact sealed registration")
    exact_check = verify_registered_sources(value.get("exact_gt_sources"), root)
    verifier = value.get("import_safe_source_verification") or {}
    launcher_check = verify_launcher_sources(verifier.get("launcher_sources"), root)
    if (verifier.get("schema") != "splitfusion.m1_fixed_live_v2.import_safe_verification.v1"
            or verifier.get("exact_sources") != exact_check
            or verifier.get("launcher_check") != launcher_check
            or verifier.get("dependency_policy") != "STDLIB_ONLY_NO_REPLAY_OR_GEOMETRY_IMPORT"):
        raise ValueError("M1 fixed import-safe verification identity drift")
    gc_reg = value.get("gt_gc") or {}
    if gc_reg != C.gc_registration(gc_reg.get("arm"), 1000, gc_reg.get("producer")):
        raise ValueError("M1 fixed GC/producer registration drift")
    retention = value.get("paired_retention") or {}
    if retention != C.retention_registration(retention.get("plan_sha256")):
        raise ValueError("M1 fixed retention registration drift")
    if (value.get("external_fixed_action") != C.FIXED_ACTION
            or value.get("zero_completion_semantics") != C.ZERO_COMPLETION
            or value.get("retention_resource_plan") != C.RETENTION_RESOURCE_PLAN
            or "forced_action" in value):
        raise ValueError("M1 fixed identity/resource/zero-completion registration drift")
    return value


@contextmanager
def _retained_contract(C):
    from rl_agent.splitfusion_run5_full_live_retention_v1 import entries, host, retention, gc_seams
    modules = (entries, host, retention, gc_seams)
    saved = [module.C for module in modules]
    saved_registration = entries.registration
    for module in modules:
        module.C = C
    entries.registration = lambda path, root: registration(path, root, C)
    try:
        yield entries
    finally:
        entries.registration = saved_registration
        for module, value in zip(modules, saved):
            module.C = value


def child(C, argv=None):
    from rl_agent.splitfusion_single_gpu_profiler_v1 import child_v1 as BASE
    from .fixed import install_fixed_baseline
    args = list(sys.argv[1:] if argv is None else argv)
    attempt = Path(BASE.attempt_dir_from_argv(args)).resolve(strict=True)
    with install_fixed_baseline(fixed_action=C.FIXED_ACTION,
            log_path=attempt / "m1_fixed_action" / "FIXED_ACTIONS.jsonl"), \
            _retained_contract(C) as retained:
        return retained.child(args)


def edge(C, argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    with _retained_contract(C) as retained:
        return retained.edge(args)

