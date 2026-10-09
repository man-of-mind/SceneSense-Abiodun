"""M3.1 v8 prospective identity: the v7 binding (cross-process roles + GT prep FREEZE lifetime repair) plus the
offline-qualified tilt-aware pedestrian AVO geometry in the GT prep worker.

V7 (``good_all_prospective_01``) stopped at frame 3502: the GT prep worker refused a scoreable pedestrian whose frozen
pitch/roll exceeded 1e-6 deg (``yaw-only AVO does not support tilted actor 117: TILTED_SCOREABLE_TARGET``).  The
candidate ``rl_agent/splitfusion_avo_tilt_v1`` (report ``experiments/splitfusion_avo_tilt_v1/REPORT.md``) uses the
frozen CARLA actor rotation for tilted pedestrians and the frozen yaw rotation (bit-identical) for upright ones.

v8 changes one runtime behaviour, in the GT prep worker process only:

* ``WORKER_ENTRY`` is the native v8 worker trampoline.  For ``--role prep`` it installs the qualified switch
  ``splitfusion_avo_tilt_v1.switch`` (``splitfusion_numeric_gt_producer_v1.avo.numeric_avo`` ->
  ``splitfusion_avo_tilt_v1.quality.numeric_avo``) and then runs the unchanged v7 worker trampoline (FREEZE lifetime
  repair, then the v4 trampoline -> v3 worker).  ``--role score`` runs the v7 trampoline unchanged.
* ``SERVICE_ENTRY`` is the native v8 service trampoline (versioned copy of v7's, reading THIS contract), so the
  retained worker spawn selects the v8 worker entry without rebinding an older module global.

The candidate sources are bound at the digests qualified by ``experiments/splitfusion_avo_tilt_v1/service_proof_02``.
Everything else is v7: per-role launcher binding, child/edge production scopes, retention plan, limits, schedule,
actor, reward, deadline, pose/map repairs, capacity substitutions, GC policy and the abort exporter.  The scientific
runtime basis stays v4.
"""
from pathlib import Path
import sys

from rl_agent.splitfusion_m3_route_b_v3 import contract as V3
from rl_agent.splitfusion_m3_route_b_v4 import contract as V4
from rl_agent.splitfusion_m3_route_b_v7 import contract as V7
from rl_agent.splitfusion_m3_route_b_v4.contract import *  # noqa: F401,F403

PACKAGE = "rl_agent/splitfusion_m3_route_b_v8"
MODULE = "rl_agent.splitfusion_m3_route_b_v8"
LAUNCHER_REVISION = "M3_1_ROUTE_B_STARTUP_BINDING_V8"
ID = "M3_1_ROUTE_B_GOOD_ALL_V8"
RUNTIME_REVISION = ID
SCIENTIFIC_RUNTIME_BASIS = V4.ID

CHILD = "rl_agent.splitfusion_single_gpu_profiler_v1.child_v1_m3_route_b_v8"
EDGE = "rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1_m3_route_b_v8"
RUNNER = MODULE + ".runner"
PARTITION = MODULE + ".partition"
HOST = MODULE + ".host"
ENTRIES = MODULE + ".entries"
GC_SEAM = MODULE + ".gc_seams"
RETENTION = MODULE + ".retention"
SERVICE_ENTRY = MODULE + ".service_entry"           # native copy of the v7 service_entry reading this contract
WORKER_ENTRY = MODULE + ".worker_entry"             # prep: tilt-aware AVO switch; both roles -> V7_WORKER_ENTRY
V7_WORKER_ENTRY = V7.WORKER_ENTRY                   # unchanged v7 trampoline: prep FREEZE lifetime repair
RETAINED_WORKER_ENTRY = V4.WORKER_ENTRY             # unchanged v4 trampoline -> unchanged v3 worker implementation
RETENTION_PLAN_ID = V4.RETENTION_PLAN_ID
EDGE_REGISTRATION_PATH = "/partition_output/registration.json"
VERIFY_FLAG = "--m3-verify-startup"
SHIMS = {f"{PACKAGE}/runtime_shims/child_v1_m3_route_b_v8.py":
         "rl_agent/splitfusion_single_gpu_profiler_v1/child_v1_m3_route_b_v8.py",
         f"{PACKAGE}/runtime_shims/edge_profiled_v1_m3_route_b_v8.py":
         "rl_agent/splitfusion_single_gpu_profiler_v1/edge_profiled_v1_m3_route_b_v8.py"}

# The v7 GT prep FREEZE lifetime repair, unchanged (qualification_02 digests).
PREP_LIFETIME_MODULE = V7.PREP_LIFETIME_MODULE
PREP_LIFETIME_PACKAGE = V7.PREP_LIFETIME_PACKAGE
PREP_LIFETIME_QUALIFIED = dict(V7.PREP_LIFETIME_QUALIFIED)
FAILURE_SNAPSHOT_DIR = V7.FAILURE_SNAPSHOT_DIR
PREP_BINDING_RECORD = V7.PREP_BINDING_RECORD        # still written by the v7 worker trampoline (prep role)

# The tilt-aware pedestrian AVO: exactly the runtime files qualified in
# experiments/splitfusion_avo_tilt_v1/service_proof_02 (RESULT.json "sources"; also replay_3502_02, upright_parity_02).
AVO_TILT_PACKAGE = "rl_agent/splitfusion_avo_tilt_v1"
AVO_TILT_MODULE = "rl_agent.splitfusion_avo_tilt_v1"
AVO_TILT_QUALIFIED = {
    f"{AVO_TILT_PACKAGE}/__init__.py": "ef79a34679f3901240e17068bbcd3f1ed0a601a1a54516039a1acd3c77a0add9",
    f"{AVO_TILT_PACKAGE}/geometry.py": "d0f34d88be786a76df56f6224b7eb17c7ad7311729654e7a0b23c193ad6cd26a",
    f"{AVO_TILT_PACKAGE}/quality.py": "d536bc4fe24d12ac0f7cd2f115141d72bb6d49e62a0589b814ef634d0b736404",
    f"{AVO_TILT_PACKAGE}/switch.py": "81f20880d7c09391c40516b4dfabaf0da77fbd4500746a4774fd55125156bff6"}
AVO_GEOMETRY_VERSION = "route_b_actor_volume_visibility_v1__tilt_aware_candidate_v1"   # = geometry.GEOMETRY_VERSION
AVO_SWITCH = AVO_TILT_MODULE + ".switch"
AVO_SUBSTITUTION = dict(target="rl_agent.splitfusion_numeric_gt_producer_v1.avo:numeric_avo",
                        replacement=AVO_TILT_MODULE + ".quality:numeric_avo", process="GT prep worker only")
AVO_UPRIGHT_BOUNDARY_DEG = 1e-6                     # unchanged pre-existing boundary; selects the representation only
AVO_GROUND_RULE_ENVELOPE_M = 0.03                   # = core.GROUND_REJECT_MARGIN_M (frozen); beyond it: fatal refusal
AVO_BINDING_RECORD = "avo_tilt_binding.json"        # written by the v8 worker trampoline, prep role only
# Candidate evidence reused (main checkout, each sealed by its OUTPUT_SHA256.json): RESULT.json digests.
AVO_TILT_EVIDENCE_DIR = "experiments/splitfusion_avo_tilt_v1"
AVO_TILT_EVIDENCE = {
    "replay_3502_02": "6379728d1e8c08f2f62908be33178f06bfda869d61b82fc12eabf802afdd43c9",
    "upright_parity_02": "dbf4fe439e611c8aff9e502ea26b1694795ef03ec277f56ecc65def90990175a",
    "service_proof_02": "e18d90180ef5b7b53c3adb26675b3ed444d750c999a1057dfe3986ed5dcfdd35"}
AVO_TILT_EVIDENCE_SCHEMAS = {
    "replay_3502_02": "splitfusion.avo_tilt_v1.replay_3502.v1",
    "upright_parity_02": "splitfusion.avo_tilt_v1.upright_parity.v1",
    "service_proof_02": "splitfusion.avo_tilt_v1.service_proof.v1"}

# Contract constants read by retained modules that keep their own contract binding inside a v8 run (as in v7).
ROLE_CONSTANTS = V7.ROLE_CONSTANTS
V3_ENTRY_HELPERS = V7.V3_ENTRY_HELPERS


def module_identity(function):
    """Import name of the module that defines ``function`` (a ``-m`` entry point executes as ``__main__``)."""
    spec = getattr(sys.modules.get(function.__module__), "__spec__", None)
    return spec.name if spec is not None else function.__module__


def role_constant_audit():
    """Per (module, name): the value the retained module reads equals the v8 value (functions: same object)."""
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
    return f"M3_1_ROUTE_B_GOOD_ALL_V8_{arm.upper()}_{producer.upper()}_C{int(budget):04d}_ONCE_EXECUTE_V1"


def avo_registration():
    return dict(geometry_version=AVO_GEOMETRY_VERSION, switch=AVO_SWITCH, substitution=dict(AVO_SUBSTITUTION),
                roles=["prep"], sources=dict(AVO_TILT_QUALIFIED), upright_boundary_deg=AVO_UPRIGHT_BOUNDARY_DEG,
                upright_representation="frozen core.yaw_rotation (bit-identical to v7)",
                tilted_representation="frozen CARLA actor matrix 3x3 (validated vs CARLA Euler, proper rotation)",
                ground_rule_envelope_m=AVO_GROUND_RULE_ENVELOPE_M,
                refused_fatal=["TILT_OUTSIDE_GROUND_RULE_ENVELOPE_TARGET", "nonzero bbox.rotation",
                               "invalid/inconsistent rotation evidence", "foreign geometry version"],
                silently_skipped=False, thresholds_changed=False, binding_record=f"{TRACE_DIR}/{AVO_BINDING_RECORD}")


def gc_registration(arm, budget, producer="v2"):
    value = V7.gc_registration(arm, budget, producer)
    value["trampolines"] = dict(service=SERVICE_ENTRY, worker=WORKER_ENTRY)
    value["implementations"] = dict(value["implementations"], service=SERVICE_ENTRY, v7_worker=V7_WORKER_ENTRY)
    value["avo_geometry"] = avo_registration()
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
    """v4 sources, every file of this package and of the sealed v7 package (its worker trampoline and contract run in
    the prep/score workers), the FREEZE lifetime repair and the tilt-aware AVO runtime files."""
    root = Path(root).resolve(strict=True)
    result = dict(V4.own_sources(root))
    for package in (V7.PACKAGE, PACKAGE):
        result.update({str(p.relative_to(root)): sha(p) for p in sorted((root / package).glob("*.py"))})
    result.update({name: sha(root / name) for name in PREP_LIFETIME_QUALIFIED})
    result.update({name: sha(root / name) for name in AVO_TILT_QUALIFIED})
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
            raise ValueError("missing real M3 v8 trampoline: " + str(path))
        result[str(path.relative_to(root))] = sha(path)
    for relative, digest in result.items():
        path = root / relative
        if not path.is_file() or path.is_symlink() or sha(path) != digest:
            raise ValueError("M3 v8 source drift: " + relative)
    for relative, digest in PREP_LIFETIME_QUALIFIED.items():
        if result[relative] != digest:
            raise ValueError("prep lifetime repair differs from the qualified source: " + relative)
    for relative, digest in AVO_TILT_QUALIFIED.items():
        if result[relative] != digest:
            raise ValueError("tilt-aware AVO differs from the qualified source: " + relative)
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
        gt_workers=dict(trampoline=WORKER_ENTRY, v7_trampoline=V7_WORKER_ENTRY,
                        retained_trampoline=RETAINED_WORKER_ENTRY,
                        implementation="rl_agent.splitfusion_m3_route_b_v3.worker_entry",
                        prep_core=PREP_LIFETIME_MODULE + ".PrepCoreLifetime",
                        prep_avo=AVO_SUBSTITUTION["replacement"], prep_avo_switch=AVO_SWITCH,
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
        avo_geometry=avo_registration(),
        actor_boundary_sha256=actor_boundary, support_sha256=support_sha256,
        sources=source_inventory(root), metrics=metrics(), state="ORIGINAL_RUN5_22D_QUALITY_AWARE",
        controller="RewardHoldControllerV2", deadline_ns=170_000_000, deadline_inclusive=True,
        fixed_fallback="mode11_q9800_a71", no_retraining=True, forced_action=None, budget=int(budget),
        gt_gc_arm=arm, gc_policy=GC_POLICY, gc_safety=SAFETY, numeric_producer=producer,
        producer_substitutions=list(SUBSTITUTIONS) if producer == "v2" else []))
