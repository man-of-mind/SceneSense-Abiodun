"""Prospective contract for the M1 1,000-frame static action-0 live baseline.

This is the retained Run-5 retention launcher with one deliberate selection
difference: every new decision opens action 0 directly.  Actor proposal and
actor-only state admission are not prerequisites.  State/telemetry assembly is
still attempted for observation, but typed admission failures are recorded and
cannot substitute action 71.  Holds, transport, GT, radio, edge, cycle closure,
and outcome-independent paired retention remain the retained implementation.
"""
from pathlib import Path

from rl_agent.splitfusion_run5_full_live_retention_v1 import contract as B

PACKAGE = "rl_agent/splitfusion_m1_static_live_v1"
CHILD = "rl_agent.splitfusion_single_gpu_profiler_v1.child_v1_m1_static_live_v1"
EDGE = "rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1_m1_static_live_v1"
RUNNER = "rl_agent.splitfusion_m1_static_live_v1.runner"
PARTITION = "rl_agent.splitfusion_m1_static_live_v1.partition"
ID = "M1_STATIC_NOAE_UINT8_NODROP_B1000_V1"
ARMS = ("candidate",)
BUDGETS = (1000,)

# The adopted, already-qualified service/worker implementation is unchanged.
SERVICE_MODULE = B.SERVICE_MODULE
SERVICE_ENTRY = B.SERVICE_ENTRY
WORKER_MODULE = B.WORKER_MODULE
WORKER_ENTRY = B.WORKER_ENTRY
PRODUCERS = B.PRODUCERS
PRODUCER_CLASS = B.PRODUCER_CLASS
PRODUCER_SOURCES = B.PRODUCER_SOURCES
SUBSTITUTIONS = B.SUBSTITUTIONS
TRACE_DIR = B.TRACE_DIR
GC_SOURCES = B.GC_SOURCES
INPUT_SOURCES = B.INPUT_SOURCES
EXPECTED_CHANGED_FROM_V1 = B.EXPECTED_CHANGED_FROM_V1
SOCKET_NAME, CAPABILITY_NAME = B.SOCKET_NAME, B.CAPABILITY_NAME
GC_POLICY, SAFETY = B.GC_POLICY, B.SAFETY
RETENTION_SOURCES = B.RETENTION_SOURCES
RETENTION_PLAN_FILE = B.RETENTION_PLAN_FILE
RETENTION_DIR = B.RETENTION_DIR
RETENTION_RESULT_FILE = B.RETENTION_RESULT_FILE
V1 = B.V1
sha, canonical = B.sha, B.canonical
limits, pipeline_delta = B.limits, B.pipeline_delta
trace_limits = B.trace_limits
retention_plan = B.retention_plan
retention_registration = B.retention_registration

FIXED_ACTION = dict(
    action_id=0, profile_id="split_noae_uint8_q0000", family="noAE",
    quantizer="UINT8", mode_id=0, q_e4=0, keep_count=21504, drop_count=0,
    measurement_status="MEASURED_ANCHOR",
    execution_bundle_sha256="583cbab6b68449495e779259507fdf7dcdfba56b5d31894ab9f8955da4deb10a",
    support="EXTERNAL_FIXED_BASELINE_OUTSIDE_ACTOR_SUPPORT_NOT_CLAMPED")

ZERO_COMPLETION = dict(
    evidence_collection="evaluated independently from prediction delivery",
    delivery_checks="NOT_EXERCISED when completed predictions == 0; never reported as pass",
    protocol_faults="FAIL even when completed predictions == 0",
    live_quality="UNAVAILABLE when completed predictions == 0; no offline substitution",
    latency="completed predictions only; unavailable when count is zero")

VALIDATION = dict(
    selection="all decisions and holds must execute action 0; actor calls and fallbacks must both be zero",
    infrastructure="input, transport, resource, protocol and cleanup checks remain fail-closed",
    completion="zero completion is an observed outcome under ZERO_COMPLETION, not a failed delivery check",
    retention="outcome-independent registered decision ordinals; missing/not-reached reported",
    frame_budget="retained whole-cycle rule; 1,000 frame budget is not 1,000 decisions")


def metrics():
    return dict(B.metrics(),
        action_selection="EXTERNAL_FIXED_A0_DIRECT_NO_ACTOR_NO_ADMISSION_GATE_V1",
        actor_proposal_required=False, actor_state_admission_required=False,
        observational_telemetry=True, typed_admission_reasons_report_only=True,
        fallback_substitution=False, fixed_action=dict(FIXED_ACTION),
        zero_completion_semantics=dict(ZERO_COMPLETION))


def gc_registration(arm, budget, producer="v2"):
    value = B.gc_registration(arm, budget, producer)
    value["validation"] = VALIDATION
    return value


def token(arm, budget=1000, producer="v2"):
    if (arm, int(budget), producer) != ("candidate", 1000, "v2"):
        raise ValueError("M1 static baseline permits candidate/v2/B1000 exactly once")
    return "M1_STATIC_NOAE_UINT8_NODROP_CANDIDATE_V2_B1000_ONCE_EXECUTE_V1"


def own_sources(root):
    root = Path(root).resolve(strict=True)
    out = dict(B.own_sources(root))
    out.update({str(p.relative_to(root)): sha(p)
                for p in sorted((root / PACKAGE).glob("*.py"))})
    return out


def exact_sources(root):
    # Host-side registration construction may use the full replay-derived
    # retained inventory.  Edge startup never calls this function: it uses the
    # import-safe path manifest and byte checker in source_verify instead.  Keep
    # this set byte-for-byte compatible with the prerequisite-qualified base;
    # M1 launcher sources are separately sealed in registration.
    result = dict(B.exact_sources(root))
    from .source_manifest import EXPECTED_PATHS_SHA256, EXPECTED_SOURCE_COUNT
    from .source_verify import paths_sha256
    if (len(result) != EXPECTED_SOURCE_COUNT
            or paths_sha256(result) != EXPECTED_PATHS_SHA256):
        raise ValueError("host-generated M1 exact-source inventory differs from import-safe manifest")
    return dict(sorted(result.items()))


def source_inventory(root):
    root = Path(root).resolve(strict=True)
    result = dict(B.source_inventory(root))
    result.update(own_sources(root))
    for module in (CHILD, EDGE):
        path = root / (module.replace(".", "/") + ".py")
        if not path.is_file() or path.is_symlink():
            raise ValueError("missing real M1 static trampoline: " + str(path))
        result[str(path.relative_to(root))] = sha(path)
    for relative, digest in result.items():
        path = root / relative
        if not path.is_file() or path.is_symlink() or sha(path) != digest:
            raise ValueError("M1 static source drift: " + relative)
    return dict(sorted(result.items()))


def lineage(root, actor_boundary, support_sha256, budget, arm, producer="v2",
            retention_plan_sha256=None):
    if (arm, int(budget), producer) != ("candidate", 1000, "v2"):
        raise ValueError("foreign M1 static configuration")
    if not isinstance(retention_plan_sha256, str) or len(retention_plan_sha256) != 64:
        raise ValueError("retention plan digest required")
    return canonical(dict(
        actor_boundary_sha256=actor_boundary,
        actor_role="frozen deployment identity audited and loaded by retained startup; never called for selection",
        support_sha256=support_sha256, sources=source_inventory(root), metrics=metrics(),
        state="OBSERVATIONAL_ONLY_NOT_AN_ADMISSION_GATE", controller="RewardHoldControllerV2",
        deadline_ns=170_000_000, deadline_inclusive=True,
        fixed_action=dict(FIXED_ACTION), fallback_substitution=None, no_retraining=True,
        budget=int(budget), gt_gc_arm=arm, gc_policy=GC_POLICY, gc_safety=SAFETY,
        numeric_producer=producer, producer_substitutions=list(SUBSTITUTIONS),
        retention_plan_sha256=retention_plan_sha256,
        retention_registration=retention_registration(retention_plan_sha256),
        zero_completion=dict(ZERO_COMPLETION)))
