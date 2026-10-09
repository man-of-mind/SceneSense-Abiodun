"""M3.1 v7 prospective identity: the v6 cross-process binding plus the GT prep FREEZE exception-lifetime repair.

V6 (``good_all_prospective_02``) started correctly and stopped at frame 3557: the GT prep worker replied
``BufferError: cannot close exported pointers exist`` from FREEZE.  The retained ``PrepCore.handle`` keeps the decoded
source (borrowed mapped image views) alive in its executing frame when preparation raises, so the mapping release can
replace the preparation error with BufferError.  The initiating error of frame 3557 was not recorded; it is unresolved.

v7 changes one runtime behaviour, in the GT prep worker process only:

* ``WORKER_ENTRY`` is the native v7 worker trampoline.  For ``--role prep`` it installs
  ``splitfusion_gt_freeze_lifetime_v1.repair`` (the offline-qualified PrepCore subclass: release the borrowed source in
  ``finally``, print the chained exception, keep one bounded fatal-error snapshot in the trace directory) and then runs
  the unchanged v4 worker trampoline.  ``--role score`` runs the unchanged v4 worker trampoline, nothing installed.
* ``SERVICE_ENTRY`` is the native v7 service trampoline, a versioned copy of v3 ``service_entry`` reading THIS
  contract: the retained worker substitution selects the worker entry from its contract, and reusing the v3 module
  would need its global ``C`` rebound (the pattern v6 removed).

Everything else is v6: per-role launcher binding, child/edge production scopes, retention plan, limits, schedule,
actor, pose/map repairs, capacity substitutions and the abort exporter.  The scientific runtime basis stays v4.
"""
from pathlib import Path
import sys

from rl_agent.splitfusion_m3_route_b_v3 import contract as V3
from rl_agent.splitfusion_m3_route_b_v4 import contract as V4
from rl_agent.splitfusion_m3_route_b_v4.contract import *  # noqa: F401,F403

PACKAGE = "rl_agent/splitfusion_m3_route_b_v7"
MODULE = "rl_agent.splitfusion_m3_route_b_v7"
LAUNCHER_REVISION = "M3_1_ROUTE_B_STARTUP_BINDING_V7"
ID = "M3_1_ROUTE_B_GOOD_ALL_V7"
RUNTIME_REVISION = ID
SCIENTIFIC_RUNTIME_BASIS = V4.ID

CHILD = "rl_agent.splitfusion_single_gpu_profiler_v1.child_v1_m3_route_b_v7"
EDGE = "rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1_m3_route_b_v7"
RUNNER = MODULE + ".runner"
PARTITION = MODULE + ".partition"
HOST = MODULE + ".host"
ENTRIES = MODULE + ".entries"
GC_SEAM = MODULE + ".gc_seams"
RETENTION = MODULE + ".retention"
SERVICE_ENTRY = MODULE + ".service_entry"           # native copy of v3 service_entry reading this contract
WORKER_ENTRY = MODULE + ".worker_entry"             # prep: lifetime repair; both roles -> RETAINED_WORKER_ENTRY
RETAINED_WORKER_ENTRY = V4.WORKER_ENTRY             # unchanged v4 trampoline -> unchanged v3 worker implementation
RETENTION_PLAN_ID = V4.RETENTION_PLAN_ID
EDGE_REGISTRATION_PATH = "/partition_output/registration.json"
VERIFY_FLAG = "--m3-verify-startup"
SHIMS = {f"{PACKAGE}/runtime_shims/child_v1_m3_route_b_v7.py":
         "rl_agent/splitfusion_single_gpu_profiler_v1/child_v1_m3_route_b_v7.py",
         f"{PACKAGE}/runtime_shims/edge_profiled_v1_m3_route_b_v7.py":
         "rl_agent/splitfusion_single_gpu_profiler_v1/edge_profiled_v1_m3_route_b_v7.py"}

# The GT prep FREEZE lifetime repair, exactly the files qualified offline in
# experiments/splitfusion_gt_freeze_lifetime_v1/qualification_02 (RESULT.json "sources").
PREP_LIFETIME_MODULE = "rl_agent.splitfusion_gt_freeze_lifetime_v1.repair"
PREP_LIFETIME_PACKAGE = "rl_agent/splitfusion_gt_freeze_lifetime_v1"
PREP_LIFETIME_QUALIFIED = {
    f"{PREP_LIFETIME_PACKAGE}/__init__.py": "e7b78cd9f694dd1f913cab9308bf0290fe863e7142990dbd7a12b58ab3ac58e4",
    f"{PREP_LIFETIME_PACKAGE}/repair.py": "8f415c8b93087455d1acdd13f6b97f25bb831ae031fc2c8da172877318240db3"}
FAILURE_SNAPSHOT_DIR = "freeze_failure_v1"          # created by the repair inside TRACE_DIR, fatal FREEZE only
PREP_BINDING_RECORD = "prep_lifetime_binding.json"  # written by the v7 worker trampoline, prep role only

# Contract constants read by retained modules that keep their own contract binding inside a v7 run (see v6).
# v3 ``service_entry`` is no longer executed (v7 runs its own copy); v3 ``worker_entry`` still runs under the
# unchanged v4 trampoline, which binds it to the v4 contract.
ROLE_CONSTANTS = (
    ("rl_agent.splitfusion_m3_route_b_v3.entries", "v3",
     ("CONTROLLER_MODULE", "RETAINED_CONTROLLER_MODULE", "MAP_SERVER_TRAMPOLINE", "RETAINED_MAP_SERVER")),
    ("rl_agent.splitfusion_m3_route_b_v3.capacity", "v3", ("CAPACITY_SUBSTITUTIONS", "POSE_SUBSTITUTIONS")),
    ("rl_agent.splitfusion_m3_route_b_v3.pose_tolerance", "v3", ("POSE",)),
    ("rl_agent.splitfusion_m3_route_b_v3.route_completion", "v3", ("MAP_JOURNAL_DIR",)),
    ("rl_agent.splitfusion_m3_route_b_v3.good_all_controller", "v3",
     ("ORIGINAL_PROFILE", "SCHEDULE", "channel_registration", "schedule_prefix")),
    ("rl_agent.splitfusion_m3_route_b_v3.map_class", "v3", ("MAP_CLASS",)),
    ("rl_agent.splitfusion_m3_route_b_v3.retention", "v3", ("RETENTION_DIR", "RETENTION_RESULT_FILE")),
    ("rl_agent.splitfusion_m3_route_b_v3.worker_entry", "v4",
     ("ARMS", "BUDGETS", "GC_POLICY", "POSE", "PRODUCERS", "PRODUCER_CLASS", "SAFETY", "trace_limits")),
)
V3_ENTRY_HELPERS = ("install_ue_capacity", "install_map_trampoline", "install_good_all_controller", "_write")


def module_identity(function):
    """Import name of the module that defines ``function`` (a ``-m`` entry point executes as ``__main__``)."""
    spec = getattr(sys.modules.get(function.__module__), "__spec__", None)
    return spec.name if spec is not None else function.__module__


def role_constant_audit():
    """Per (module, name): the value the retained module reads equals the v7 value (functions: same object)."""
    contracts = dict(v3=V3, v4=V4)
    out = {}
    for module, bound, names in ROLE_CONSTANTS:
        for name in names:
            seen, mine = getattr(contracts[bound], name), globals()[name]
            out[f"{module}[C={bound}].{name}"] = (seen is mine) if callable(mine) else (seen == mine)
    return out


def token(arm, budget=FRAME_CEILING, producer="v2"):
    if arm not in ARMS or producer not in PRODUCERS:
        raise ValueError("unknown arm/producer")
    return f"M3_1_ROUTE_B_GOOD_ALL_V7_{arm.upper()}_{producer.upper()}_C{int(budget):04d}_ONCE_EXECUTE_V1"


def prep_lifetime_registration():
    return dict(module=PREP_LIFETIME_MODULE, core="PrepCoreLifetime", roles=["prep"],
                sources=dict(PREP_LIFETIME_QUALIFIED), failure_snapshot=f"{TRACE_DIR}/{FAILURE_SNAPSHOT_DIR}",
                binding_record=f"{TRACE_DIR}/{PREP_BINDING_RECORD}", max_source_bytes=32 * 1024 * 1024,
                fatal_freeze_only=True, gt_semantics_changed=False)


def gc_registration(arm, budget, producer="v2"):
    value = V3.gc_registration(arm, budget, producer)
    value["trampolines"] = dict(service=SERVICE_ENTRY, worker=WORKER_ENTRY)
    value["implementations"] = dict(service=SERVICE_ENTRY, worker="rl_agent.splitfusion_m3_route_b_v3.worker_entry",
                                    retained_worker_trampoline=RETAINED_WORKER_ENTRY)
    value["prep_freeze_lifetime"] = prep_lifetime_registration()
    value["seam"] = GC_SEAM
    return value


def retention_registration(plan_sha256):
    value = V3.retention_registration(plan_sha256)
    value["plan_id"] = RETENTION_PLAN_ID
    value["loader"] = RETENTION + ".load_observer"
    return value


def retention_plan(budget):
    return V4.retention_plan(budget)


def own_sources(root):
    root = Path(root).resolve(strict=True)
    result = dict(V4.own_sources(root))
    result.update({str(p.relative_to(root)): sha(p) for p in sorted((root / PACKAGE).glob("*.py"))})
    result.update({name: sha(root / name) for name in PREP_LIFETIME_QUALIFIED})
    return result


def exact_sources(root):
    result = dict(V3.E.exact_sources(root))
    result.update(own_sources(root))
    return dict(sorted(result.items()))


def source_inventory(root):
    root = Path(root).resolve(strict=True)
    result = dict(V3.E.source_inventory(root))
    result.update(own_sources(root))
    for path in [root / (module.replace(".", "/") + ".py") for module in (CHILD, EDGE)] + [root / MAP_SERVER_TRAMPOLINE]:
        if not path.is_file() or path.is_symlink():
            raise ValueError("missing real M3 v7 trampoline: " + str(path))
        result[str(path.relative_to(root))] = sha(path)
    for relative, digest in result.items():
        path = root / relative
        if not path.is_file() or path.is_symlink() or sha(path) != digest:
            raise ValueError("M3 v7 source drift: " + relative)
    for relative, digest in PREP_LIFETIME_QUALIFIED.items():
        if result[relative] != digest:
            raise ValueError("prep lifetime repair differs from the qualified source: " + relative)
    return dict(sorted(result.items()))


def launcher_binding():
    """The exact module identity of every validating function, per process role (registered and re-checked)."""
    return dict(
        launcher_revision=LAUNCHER_REVISION, runtime_revision=RUNTIME_REVISION,
        scientific_runtime_basis=SCIENTIFIC_RUNTIME_BASIS, contract_module=MODULE + ".contract",
        host=dict(verify_prospective=HOST, verify_prerequisite=HOST, check_prerequisites=HOST,
                  verify_startup_proof=HOST),
        runner=dict(module=RUNNER, validate_parent=RUNNER, verify_prospective=HOST, verify_prerequisite=HOST),
        child=dict(module=CHILD, entries=ENTRIES, validate_child_startup=ENTRIES, verify_prospective=HOST,
                   registration=ENTRIES, load_observer=RETENTION, seam_parameters=GC_SEAM),
        edge=dict(module=EDGE, entries=ENTRIES, validate_edge_startup=ENTRIES, registration=ENTRIES,
                  registration_path=EDGE_REGISTRATION_PATH),
        gt_service=dict(seam=GC_SEAM, trampoline=SERVICE_ENTRY, implementation=SERVICE_ENTRY,
                        retained_service=SERVICE_MODULE,
                        registration_check="rl_agent.splitfusion_exact_gt_live_v2.service.verify_sources"),
        gt_workers=dict(trampoline=WORKER_ENTRY, retained_trampoline=RETAINED_WORKER_ENTRY,
                        implementation="rl_agent.splitfusion_m3_route_b_v3.worker_entry",
                        prep_core=PREP_LIFETIME_MODULE + ".PrepCoreLifetime",
                        score_core="rl_agent.splitfusion_exact_gt_service_proto_v2.worker.ScoreCore",
                        registration_check=None),
        partition=PARTITION)


def lineage(root, actor_boundary, support_sha256, budget, arm, producer="v2", retention_plan_sha256=None):
    if arm not in ARMS or producer not in PRODUCERS:
        raise ValueError("unknown arm/producer")
    if not isinstance(retention_plan_sha256, str) or len(retention_plan_sha256) != 64:
        raise ValueError("retention launcher lineage requires the pre-launch retention plan digest")
    return canonical(dict(
        launcher_revision=LAUNCHER_REVISION, runtime_revision=RUNTIME_REVISION,
        scientific_runtime_basis=SCIENTIFIC_RUNTIME_BASIS, launcher_binding=launcher_binding(),
        retention_plan_sha256=retention_plan_sha256,
        retention_registration=retention_registration(retention_plan_sha256),
        channel=channel_registration(root), route=route_registration(root), capacity=capacity_registration(),
        evidence_additions=evidence_registration(), pose_repair=pose_registration(), map_class=map_class_registration(),
        actor_boundary_sha256=actor_boundary, support_sha256=support_sha256,
        sources=source_inventory(root), metrics=metrics(), state="ORIGINAL_RUN5_22D_QUALITY_AWARE",
        controller="RewardHoldControllerV2", deadline_ns=170_000_000, deadline_inclusive=True,
        fixed_fallback="mode11_q9800_a71", no_retraining=True, forced_action=None, budget=int(budget),
        gt_gc_arm=arm, gc_policy=GC_POLICY, gc_safety=SAFETY, numeric_producer=producer,
        producer_substitutions=list(SUBSTITUTIONS) if producer == "v2" else []))
