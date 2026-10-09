"""M3.1 v6 prospective identity: one coherent registration across every process role.

The v5 attempt failed before any frame.  The v5 parent verified a v5 prospective, but the UE child it launched
(v4 trampoline -> v4 entries -> v3 ``entries.child``) resolved ``from .host import verify_prospective`` to the v3
host bound to the v3 contract, and rejected the v5 schema ("foreign GC-qualification registration").  The same
child would next have failed in v3 ``retention.load_observer`` (v3 plan id vs the registered v4 plan), and the
GT-service seam only worked because v4 rebound ``v3.gc_seams.C`` in place.

v6 removes every implicit cross-version resolution on the registration/verifier path:

* parent (host/runner), UE child, edge, GT-service seam and retention loader are native v6 functions that read
  ONE contract (this module) and are recorded in the registration as ``launcher_binding``;
* the GT service and both GT workers are the unchanged v4 trampolines over the unchanged v3 implementations (the
  exact path of the sealed v4 real-service high-coordinate proof); they perform no registration validation and
  read only contract constants that are identical in v3, v4 and v6 (asserted by ``role_constant_audit``);
* the registration documents what actually runs: ``gt_gc.trampolines`` names the v4 service/worker entries and
  ``paired_retention.plan_id`` names the plan id actually written into the unchanged plan file (both were stale
  v3 values in the v4/v5 registrations).

Scientific runtime unchanged from v4 (``SCIENTIFIC_RUNTIME_BASIS``): frozen actor, route, GOOD_ALL schedule,
inclusive 170-ms deadline, input processing, GT/GC policy, producer v2, five pose repairs, map class/publication,
retention plan, capacity limits and the abort-evidence exporter.
"""
from pathlib import Path
import sys

from rl_agent.splitfusion_m3_route_b_v3 import contract as V3
from rl_agent.splitfusion_m3_route_b_v4 import contract as V4
from rl_agent.splitfusion_m3_route_b_v4.contract import *  # noqa: F401,F403

PACKAGE = "rl_agent/splitfusion_m3_route_b_v6"
MODULE = "rl_agent.splitfusion_m3_route_b_v6"
LAUNCHER_REVISION = "M3_1_ROUTE_B_STARTUP_BINDING_V6"
ID = "M3_1_ROUTE_B_GOOD_ALL_V6"
RUNTIME_REVISION = ID
SCIENTIFIC_RUNTIME_BASIS = V4.ID

CHILD = "rl_agent.splitfusion_single_gpu_profiler_v1.child_v1_m3_route_b_v6"
EDGE = "rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1_m3_route_b_v6"
RUNNER = MODULE + ".runner"
PARTITION = MODULE + ".partition"
HOST = MODULE + ".host"
ENTRIES = MODULE + ".entries"
GC_SEAM = MODULE + ".gc_seams"
RETENTION = MODULE + ".retention"
SERVICE_ENTRY = V4.SERVICE_ENTRY          # unchanged v4 trampoline -> unchanged v3 service implementation
WORKER_ENTRY = V4.WORKER_ENTRY            # unchanged v4 trampoline -> unchanged v3 worker implementation
RETENTION_PLAN_ID = V4.RETENTION_PLAN_ID  # the unchanged v4 plan (same file digest as the sealed v4/v5 roots)
EDGE_REGISTRATION_PATH = "/partition_output/registration.json"
VERIFY_FLAG = "--m3-verify-startup"
SHIMS = {f"{PACKAGE}/runtime_shims/child_v1_m3_route_b_v6.py":
         "rl_agent/splitfusion_single_gpu_profiler_v1/child_v1_m3_route_b_v6.py",
         f"{PACKAGE}/runtime_shims/edge_profiled_v1_m3_route_b_v6.py":
         "rl_agent/splitfusion_single_gpu_profiler_v1/edge_profiled_v1_m3_route_b_v6.py"}

# Contract constants read by retained modules that keep their own contract binding inside a v6 run:
# (module, the contract that module actually reads in its process, names).  Each value must be identical to the
# v6 value; everything else such a module would read (ID, entries, plan, sources) is supplied explicitly by v6.
# The list is checked against the modules' source by test_m3_v6 (every ``C.<name>`` they reference is covered).
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
    ("rl_agent.splitfusion_m3_route_b_v3.service_entry", "v4", ("ARMS", "PRODUCERS", "WORKER_MODULE", "WORKER_ENTRY")),
    ("rl_agent.splitfusion_m3_route_b_v3.worker_entry", "v4",
     ("ARMS", "BUDGETS", "GC_POLICY", "POSE", "PRODUCERS", "PRODUCER_CLASS", "SAFETY", "trace_limits")),
)
# Helpers of v3 entries that v6 calls; every other v3 entries function is replaced by a native v6 function.
V3_ENTRY_HELPERS = ("install_ue_capacity", "install_map_trampoline", "install_good_all_controller", "_write")


def module_identity(function):
    """Import name of the module that defines ``function`` (a ``-m`` entry point executes as ``__main__``)."""
    spec = getattr(sys.modules.get(function.__module__), "__spec__", None)
    return spec.name if spec is not None else function.__module__


def role_constant_audit():
    """Per (module, name): the value the retained module reads equals the v6 value (functions: same object)."""
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
    return f"M3_1_ROUTE_B_GOOD_ALL_V6_{arm.upper()}_{producer.upper()}_C{int(budget):04d}_ONCE_EXECUTE_V1"


def gc_registration(arm, budget, producer="v2"):
    value = V3.gc_registration(arm, budget, producer)
    value["trampolines"] = dict(service=SERVICE_ENTRY, worker=WORKER_ENTRY)
    value["implementations"] = dict(service="rl_agent.splitfusion_m3_route_b_v3.service_entry",
                                    worker="rl_agent.splitfusion_m3_route_b_v3.worker_entry")
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
            raise ValueError("missing real M3 v6 trampoline: " + str(path))
        result[str(path.relative_to(root))] = sha(path)
    for relative, digest in result.items():
        path = root / relative
        if not path.is_file() or path.is_symlink() or sha(path) != digest:
            raise ValueError("M3 v6 source drift: " + relative)
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
        gt_service=dict(seam=GC_SEAM, trampoline=SERVICE_ENTRY,
                        implementation="rl_agent.splitfusion_m3_route_b_v3.service_entry",
                        registration_check="rl_agent.splitfusion_exact_gt_live_v2.service.verify_sources"),
        gt_workers=dict(trampoline=WORKER_ENTRY, implementation="rl_agent.splitfusion_m3_route_b_v3.worker_entry",
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
