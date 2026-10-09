"""Concrete contracts for the two authorized M1 fixed baselines.

The instances are module-like objects because the mature retained launcher
expects a contract namespace.  Unknown attributes delegate to the retained
Run-5 retention contract; only fixed selection identity and launcher paths
differ.
"""
from pathlib import Path

from rl_agent.splitfusion_run5_full_live_retention_v1 import contract as B


PACKAGE = "rl_agent/splitfusion_m1_fixed_live_v2"
EXPECTED_SOURCE_COUNT = 461
EXPECTED_PATHS_SHA256 = "901b3e352f542c39c54ec0a27c5862279bad93c0131dc44818ea047ada3a7276"
INPUT_BYTES = 9_633_792
SFD4_FRAMING_BYTES = 455
SAMPLE_COUNT = 100
MAX_WIRE_BYTES = 2 * 1024 * 1024
MAX_HOST_BYTES = 1_342_177_280
MAX_STORAGE_BYTES = 1_342_177_280


ZERO_COMPLETION = dict(
    evidence_collection="evaluated independently from prediction delivery",
    delivery_checks="NOT_EXERCISED when successful predictions == 0; never reported as pass",
    protocol_faults="FAIL even when successful predictions == 0",
    live_quality="UNAVAILABLE when successful predictions == 0; no offline substitution",
    latency="successful edge.compute completions only; unavailable when count is zero")


class FixedContract:
    def __init__(self, *, key, identifier, action, child, edge, runner, partition,
                 token, pre_zstd_bytes):
        self.KEY = key
        self.ID = identifier
        self.FIXED_ACTION = dict(action)
        self.CHILD, self.EDGE = child, edge
        self.RUNNER, self.PARTITION = runner, partition
        self.EXECUTE_TOKEN = token
        self.PACKAGE = PACKAGE
        self.__file__ = __file__
        self.ARMS = ("candidate",)
        self.BUDGETS = (1000,)
        self.PRODUCERS = B.PRODUCERS
        for name in (
                "SERVICE_MODULE", "SERVICE_ENTRY", "WORKER_MODULE", "WORKER_ENTRY",
                "PRODUCER_CLASS", "PRODUCER_SOURCES", "SUBSTITUTIONS", "TRACE_DIR",
                "GC_SOURCES", "INPUT_SOURCES", "EXPECTED_CHANGED_FROM_V1",
                "SOCKET_NAME", "CAPABILITY_NAME", "GC_POLICY", "SAFETY",
                "RETENTION_SOURCES", "RETENTION_PLAN_FILE", "RETENTION_DIR",
                "RETENTION_RESULT_FILE", "V1"):
            setattr(self, name, getattr(B, name))
        self.sha, self.canonical = B.sha, B.canonical
        self.limits, self.pipeline_delta = B.limits, B.pipeline_delta
        self.trace_limits = B.trace_limits
        self.ZERO_COMPLETION = dict(ZERO_COMPLETION)
        self.VALIDATION = dict(
            selection=("all decisions and holds must execute the registered fixed identity; "
                       "actor calls and fallbacks must both be zero"),
            infrastructure="input, transport, resource, protocol and cleanup checks remain fail-closed",
            completion=("zero successful completion is an observed outcome; delivery-dependent checks "
                        "are NOT_EXERCISED"),
            retention=("outcome-independent every-fifth decision ordinal across controller-session "
                       "resets; missing/not-reached reported"),
            frame_budget="retained whole-cycle rule; 1,000 frame budget is not 1,000 decisions")
        wire_bound = int(pre_zstd_bytes) + SFD4_FRAMING_BYTES
        per_sample = INPUT_BYTES + wire_bound
        self.RETENTION_RESOURCE_PLAN = dict(
            sampled_decisions=SAMPLE_COUNT, input_bytes_per_sample=INPUT_BYTES,
            action_pre_zstd_bytes_upper_bound=int(pre_zstd_bytes),
            sfd4_framing_bytes=SFD4_FRAMING_BYTES,
            wire_bytes_per_sample_upper_bound=wire_bound,
            registered_max_wire_bytes=MAX_WIRE_BYTES,
            retained_bytes_upper_bound=SAMPLE_COUNT * per_sample,
            registered_max_host_bytes=MAX_HOST_BYTES,
            registered_max_storage_bytes=MAX_STORAGE_BYTES,
            checks=dict(wire=wire_bound <= MAX_WIRE_BYTES,
                        host=SAMPLE_COUNT * per_sample <= MAX_HOST_BYTES,
                        storage=SAMPLE_COUNT * per_sample <= MAX_STORAGE_BYTES))
        if not all(self.RETENTION_RESOURCE_PLAN["checks"].values()):
            raise ValueError("fixed action exceeds registered retention resources")

    def __getattr__(self, name):
        return getattr(B, name)

    def metrics(self):
        return dict(B.metrics(),
            action_selection=f"EXTERNAL_FIXED_{self.KEY}_DIRECT_NO_ACTOR_NO_ADMISSION_GATE_V2",
            actor_proposal_required=False, actor_state_admission_required=False,
            observational_telemetry=True, typed_admission_reasons_report_only=True,
            fallback_substitution=False, fixed_action=dict(self.FIXED_ACTION),
            zero_completion_semantics=dict(self.ZERO_COMPLETION),
            successful_prediction=("edge.compute ended successfully with CPU prediction materialized; "
                                   "map publication is separately counted"))

    def gc_registration(self, arm, budget, producer="v2"):
        value = B.gc_registration(arm, budget, producer)
        value["validation"] = self.VALIDATION
        return value

    def token(self, arm, budget=1000, producer="v2"):
        if (arm, int(budget), producer) != ("candidate", 1000, "v2"):
            raise ValueError("M1 fixed baseline permits candidate/v2/B1000 exactly once")
        return self.EXECUTE_TOKEN

    def retention_plan(self, budget):
        return B.retention_plan(budget)

    def retention_registration(self, plan_sha256):
        return B.retention_registration(plan_sha256)

    def own_sources(self, root):
        root = Path(root).resolve(strict=True)
        result = dict(B.own_sources(root))
        result.update({str(path.relative_to(root)): self.sha(path)
                       for path in sorted((root / PACKAGE).glob("*.py"))})
        return dict(sorted(result.items()))

    def exact_sources(self, root):
        result = dict(B.exact_sources(root))
        from .source_verify import paths_sha256
        if (len(result) != EXPECTED_SOURCE_COUNT
                or paths_sha256(result) != EXPECTED_PATHS_SHA256):
            raise ValueError("host exact-source inventory differs from import-safe manifest")
        return dict(sorted(result.items()))

    def source_inventory(self, root):
        root = Path(root).resolve(strict=True)
        result = dict(B.source_inventory(root))
        result.update(self.own_sources(root))
        for module in (self.CHILD, self.EDGE, self.RUNNER, self.PARTITION):
            path = root / (module.replace(".", "/") + ".py")
            if not path.is_file() or path.is_symlink():
                raise ValueError("missing real M1 fixed launcher source: " + str(path))
            result[str(path.relative_to(root))] = self.sha(path)
        return dict(sorted(result.items()))

    def lineage(self, root, actor_boundary, support_sha256, budget, arm,
                producer="v2", retention_plan_sha256=None):
        if (arm, int(budget), producer) != ("candidate", 1000, "v2"):
            raise ValueError("foreign M1 fixed configuration")
        if not isinstance(retention_plan_sha256, str) or len(retention_plan_sha256) != 64:
            raise ValueError("retention plan digest required")
        return self.canonical(dict(
            actor_boundary_sha256=actor_boundary,
            actor_role="frozen deployment identity audited at startup; never called for selection",
            support_sha256=support_sha256, sources=self.source_inventory(root), metrics=self.metrics(),
            state="OBSERVATIONAL_ONLY_NOT_AN_ADMISSION_GATE", controller="RewardHoldControllerV2",
            deadline_ns=170_000_000, deadline_inclusive=True,
            fixed_action=dict(self.FIXED_ACTION), fallback_substitution=None, no_retraining=True,
            budget=int(budget), gt_gc_arm=arm, gc_policy=self.GC_POLICY, gc_safety=self.SAFETY,
            numeric_producer=producer, producer_substitutions=list(self.SUBSTITUTIONS),
            retention_plan_sha256=retention_plan_sha256,
            retention_registration=self.retention_registration(retention_plan_sha256),
            retention_resource_plan=self.RETENTION_RESOURCE_PLAN,
            zero_completion=dict(self.ZERO_COMPLETION)))


AE32_UINT8_A54 = FixedContract(
    key="AE32_UINT8_A54_Q0_NODROP",
    identifier="M1_STATIC_AE32_UINT8_A54_Q0_NODROP_B1000_V2",
    action=dict(action_id=54, profile_id="split_ae32_uint8_q0000", family="AE32",
        quantizer="UINT8", mode_id=9, q_e4=0, keep_count=21504, drop_count=0,
        measurement_status="MEASURED_ANCHOR",
        execution_bundle_sha256="dbe6a1e5ee1b0c0e2ab8617a35ad6388f2325aff3b1761c1cf958a8a1ee976cc",
        support="CATALOGUE_ACTION_54__EXTERNAL_FIXED_NOT_ACTOR_SELECTED"),
    child="rl_agent.splitfusion_m1_fixed_live_v2.child_ae32",
    edge="rl_agent.splitfusion_m1_fixed_live_v2.edge_ae32",
    runner="rl_agent.splitfusion_m1_fixed_live_v2.runner_ae32",
    partition="rl_agent.splitfusion_m1_fixed_live_v2.partition_ae32",
    token="M1_STATIC_AE32_UINT8_A54_Q0_NODROP_CANDIDATE_V2_B1000_ONCE_EXECUTE",
    pre_zstd_bytes=688_434)

MODE11_UINT4_Q6000 = FixedContract(
    key="MODE11_AE32_UINT4_Q6000_OFF_ANCHOR",
    identifier="M1_STATIC_MODE11_AE32_UINT4_Q6000_OFFANCHOR_B1000_V2",
    action=dict(action_id=None, profile_id=None, family="AE32", quantizer="UINT4",
        mode_id=11, q_e4=6000, keep_count=8602, drop_count=12902,
        measurement_status="UNMEASURED_OFF_ANCHOR",
        execution_bundle_sha256="7e45fe306975b7658e92a652f3535de850083ab24e1e8e49f47ca29e7200f488",
        support="OFF_ANCHOR_EXECUTION_CONTRACT_IDENTITY__NO_CATALOGUE_ID_INVENTED"),
    child="rl_agent.splitfusion_m1_fixed_live_v2.child_mode11",
    edge="rl_agent.splitfusion_m1_fixed_live_v2.edge_mode11",
    runner="rl_agent.splitfusion_m1_fixed_live_v2.runner_mode11",
    partition="rl_agent.splitfusion_m1_fixed_live_v2.partition_mode11",
    token="M1_STATIC_MODE11_AE32_UINT4_Q6000_OFFANCHOR_CANDIDATE_V2_B1000_ONCE_EXECUTE",
    pre_zstd_bytes=140_628)

CONTRACTS = {"ae32_uint8_a54": AE32_UINT8_A54,
             "mode11_uint4_q6000": MODE11_UINT4_Q6000}

