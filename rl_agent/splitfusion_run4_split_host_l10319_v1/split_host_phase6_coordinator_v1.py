"""Additive coordinator seam for the W10275/L10319 Phase-6 split host.

This module is deliberately import-pure and has no CLI.  It composes, rather
than reimplements, the qualified Phase-6 child, the local-RAN plan, the
prestarted remote edge lifecycle, and the existing HIGH-worker GT sender.

The coordinator owns no remote process.  In particular, the proxy installed
in place of the frozen child's local-edge hooks only opens/closes the local GT
sender and creates attempt-local staging.  It cannot start or stop the remote
CN or edge.  Policy deadlines remain on W10275's CLOCK_MONOTONIC_RAW; remote
monotonic timestamps are never subtracted from local timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Optional, Sequence

from . import contract as C
from . import gt_sender_integration as GSI
from . import local_ran_lifecycle_v1 as LR
from . import remote_edge_lifecycle_v1 as RE


SCHEMA = "scenesense.run4.split_host_phase6_coordinator.v1"
REMOTE_READY_SCHEMA = "scenesense.run4.split_host_remote_ready.v1"
FEEDBACK_ROUTE_SCHEMA = "scenesense.run4.remote_feedback_route.v1"
RADIO_PROOF_SCHEMA = "scenesense.run4.radio_tensor_path_proof.v1"
POLICY_OWNERSHIP_SCHEMA = "scenesense.run4.ue_policy_ownership.v1"
REMOTE_RETRIEVAL_SCHEMA = "scenesense.run4.remote_evidence_retrieval.v1"
REMOTE_TEARDOWN_RELEASE_SCHEMA = (
    "scenesense.run4.split_host_remote_teardown_release.v1"
)
REMOTE_ABORT_RELEASE_SCHEMA = (
    "scenesense.run4.split_host_remote_abort_release.v1"
)
EDGE_RECEIVE_PORT = 51002
UE_CONTROL_HOST = "10.0.0.2"
UE_CONTROL_PORT = 51014
POLICY_TABLE = 9999
POLICY_INTERFACE = "oaitun_ue1"
GT_WORKER_NAME = "route-b-object-gt-evaluation"
LOCAL_GT_FINAL = Path("run4_phase6") / "split_host_gt_sender_final.json"
LOCAL_PROXY_ROOT = Path("run4_phase6") / "split_host_remote_edge_proxy"
EDGE_EVIDENCE_LEAF = "segmentation_evidence"


class SplitHostCoordinatorError(RuntimeError):
    """A split-host release fact or process-local seam failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SplitHostCoordinatorError(message)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"),
                   allow_nan=False).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def validate_campaign_binding(campaign: Mapping[str, Any]) -> Mapping[str, Any]:
    """Require the frozen route to use the registered split-host endpoints."""
    _require(type(campaign) is dict, "campaign must be an exact dictionary")
    runtime = campaign.get("runtime")
    _require(type(runtime) is dict, "campaign runtime is absent")
    topology = C.default_topology()
    expected = {
        "architecture": "DIRECT_EDGE_TO_MAP_V1",
        "edge_remote_host": topology.edge_ip,
        "edge_receive_port": EDGE_RECEIVE_PORT,
        "direct_map_ingest_port": topology.map_port,
        "ue_bind_host": UE_CONTROL_HOST,
        "ue_control_port": UE_CONTROL_PORT,
        "object_records_on_radio": False,
    }
    drift = {name: (runtime.get(name), value) for name, value in expected.items()
             if runtime.get(name) != value}
    _require(not drift, f"split-host campaign endpoint drift: {drift}")
    return expected


def source_route_probe_argv() -> tuple[str, ...]:
    """The post-attach lookup for a locally generated source-bound socket."""
    return (
        "ip", "-j", "route", "get", C.default_topology().edge_ip,
        "from", UE_CONTROL_HOST,
    )


def validate_ue_source_route(route_json: str) -> Mapping[str, Any]:
    """Prove UE tensor traffic uses table 9999 and the OAI tunnel.

    A normal host lookup without ``from 10.0.0.2`` is not equivalent
    and is deliberately inadmissible.  The LAN route through L10319 is valid
    only for the GT sideband, never for UE payload traffic.
    """
    try:
        rows = json.loads(route_json)
    except json.JSONDecodeError as exc:
        raise SplitHostCoordinatorError(f"UE source route is not JSON: {exc}") from exc
    _require(type(rows) is list and len(rows) == 1 and type(rows[0]) is dict,
             "UE source route must contain exactly one route")
    row = rows[0]
    topology = C.default_topology()
    source = str(row.get("prefsrc") or row.get("src") or row.get("from") or "")
    table = str(row.get("table") or "")
    gateway = str(row.get("gateway") or "")
    checks = {
        "destination": str(row.get("dst")) == topology.edge_ip,
        "source": source == UE_CONTROL_HOST,
        "device": str(row.get("dev")) == POLICY_INTERFACE,
        "table": table == str(POLICY_TABLE),
        "not_lan_gateway": gateway != topology.remote_lan_ip,
        "not_lan_device": str(row.get("dev")) != "wlp130s0f0",
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    _require(not failed, f"UE source route bypasses the radio path: {failed}")
    return {
        "probe_argv": list(source_route_probe_argv()),
        "destination": topology.edge_ip,
        "source": UE_CONTROL_HOST,
        "device": POLICY_INTERFACE,
        "table": POLICY_TABLE,
        "gateway": gateway or None,
        "radio_path": "PASS",
    }


@dataclass(frozen=True)
class OwnedPolicyRuleV1:
    """One rule proven absent before and added by this attempt."""

    priority: int
    before_absent: bool

    def validate(self) -> "OwnedPolicyRuleV1":
        _require(type(self.priority) is int and 1 <= self.priority <= 32765,
                 "attempt-owned policy-rule priority is invalid")
        _require(self.before_absent is True,
                 "a pre-existing policy rule cannot become attempt-owned")
        return self

    @property
    def cleanup_argv(self) -> tuple[str, ...]:
        self.validate()
        return (
            "sudo", "ip", "rule", "del", "priority", str(self.priority),
            "from", UE_CONTROL_HOST, "lookup", str(POLICY_TABLE),
        )


@dataclass(frozen=True)
class OwnedPolicyRouteV1:
    """One exact /32 route proven absent before and added by this attempt."""

    destination: str
    before_absent: bool

    def validate(self) -> "OwnedPolicyRouteV1":
        _require(self.destination == f"{C.default_topology().edge_ip}/32",
                 "attempt-owned route must be the exact remote edge /32")
        _require(self.before_absent is True,
                 "a pre-existing policy route cannot become attempt-owned")
        return self

    @property
    def cleanup_argv(self) -> tuple[str, ...]:
        self.validate()
        return (
            "sudo", "ip", "route", "del", self.destination,
            "dev", POLICY_INTERFACE, "src", UE_CONTROL_HOST,
            "table", str(POLICY_TABLE),
        )


@dataclass(frozen=True)
class PolicyRoutingOwnershipV1:
    """Diff-based ownership; cleanup never flushes a table or foreign rules."""

    schema: str
    before_rules_sha256: str
    before_table_sha256: str
    rules: tuple[OwnedPolicyRuleV1, ...]
    routes: tuple[OwnedPolicyRouteV1, ...]

    def validate(self) -> "PolicyRoutingOwnershipV1":
        _require(self.schema == POLICY_OWNERSHIP_SCHEMA,
                 "policy-ownership schema drift")
        for value in (self.before_rules_sha256, self.before_table_sha256):
            _require(len(value) == 64 and all(c in "0123456789abcdef" for c in value),
                     "policy snapshot digest is invalid")
        priorities = [rule.validate().priority for rule in self.rules]
        _require(len(set(priorities)) == len(priorities),
                 "attempt-owned policy priorities are duplicated")
        destinations = [route.validate().destination for route in self.routes]
        _require(len(set(destinations)) == len(destinations),
                 "attempt-owned policy routes are duplicated")
        _require(bool(self.rules) or bool(self.routes),
                 "no attempt-owned policy mutations were recorded")
        for command in self.cleanup_commands:
            rendered = " ".join(command)
            _require("flush" not in rendered and "wlp130s0f0" not in rendered,
                     "policy cleanup is broad or targets the LAN")
        return self

    @property
    def cleanup_commands(self) -> tuple[tuple[str, ...], ...]:
        return tuple(rule.cleanup_argv for rule in reversed(self.rules)) + tuple(
            route.cleanup_argv for route in reversed(self.routes)
        )


def validate_remote_feedback_route(document: Mapping[str, Any]) -> Mapping[str, Any]:
    """Consume an L10319 docker-exec route observation, never run local Docker."""
    topology = C.default_topology()
    required = {
        "schema", "observed_host", "container_ip", "destination", "via",
        "device", "returncode", "route_text",
    }
    _require(type(document) is dict and set(document) == required,
             "remote feedback-route fields drift")
    _require(document["schema"] == FEEDBACK_ROUTE_SCHEMA,
             "remote feedback-route schema drift")
    checks = {
        "host": document["observed_host"] == topology.remote_name,
        "container": document["container_ip"] == topology.edge_ip,
        "destination": document["destination"] == UE_CONTROL_HOST,
        "via": document["via"] == topology.upf_ip,
        "device": document["device"] == "eth0",
        "returncode": type(document["returncode"]) is int
                      and document["returncode"] == 0,
        "route": f"{UE_CONTROL_HOST} via {topology.upf_ip}"
                 in str(document["route_text"]),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    _require(not failed, f"remote edge feedback route drift: {failed}")
    return dict(document)


@dataclass(frozen=True)
class RadioTensorObservationV1:
    """Identity/digest evidence from one side of the UE-radio tensor probe."""

    observer: str
    session_uuid: str
    frame_id: int
    tensor_seq: int
    payload_sha256: str
    datagram_count: int
    payload_bytes: int
    destination: str
    destination_port: int
    interface: str

    def validate(self) -> "RadioTensorObservationV1":
        _require(self.observer in {"W10275_OAITUN_CAPTURE", "L10319_EDGE_RECEIPT"},
                 "radio proof observer drift")
        _require(bool(self.session_uuid), "radio proof session is empty")
        _require(type(self.frame_id) is int and self.frame_id >= 0,
                 "radio proof frame is invalid")
        _require(type(self.tensor_seq) is int and self.tensor_seq >= 0,
                 "radio proof tensor sequence is invalid")
        _require(len(self.payload_sha256) == 64
                 and all(c in "0123456789abcdef" for c in self.payload_sha256),
                 "radio proof payload digest is invalid")
        _require(type(self.datagram_count) is int and self.datagram_count > 0,
                 "radio proof datagram count is invalid")
        _require(type(self.payload_bytes) is int and self.payload_bytes > 0,
                 "radio proof payload length is invalid")
        _require((self.destination, self.destination_port)
                 == (C.default_topology().edge_ip, EDGE_RECEIVE_PORT),
                 "radio proof destination drift")
        expected_interface = (POLICY_INTERFACE if self.observer == "W10275_OAITUN_CAPTURE"
                              else "REMOTE_EDGE_RECEIVER")
        _require(self.interface == expected_interface,
                 "radio proof observation interface drift")
        return self

    def shared_identity(self) -> tuple[Any, ...]:
        self.validate()
        return (
            self.session_uuid, self.frame_id, self.tensor_seq,
            self.payload_sha256, self.datagram_count, self.payload_bytes,
            self.destination, self.destination_port,
        )


def validate_radio_tensor_path(capture: RadioTensorObservationV1,
                               receipt: RadioTensorObservationV1) -> Mapping[str, Any]:
    _require(capture.observer == "W10275_OAITUN_CAPTURE",
             "radio proof capture observer drift")
    _require(receipt.observer == "L10319_EDGE_RECEIPT",
             "radio proof receipt observer drift")
    _require(capture.shared_identity() == receipt.shared_identity(),
             "oaitun capture and remote edge receipt do not reconcile")
    return {
        "schema": RADIO_PROOF_SCHEMA,
        "identity_sha256": _canonical_sha256(capture.shared_identity()),
        "source_interface": POLICY_INTERFACE,
        "destination": f"{C.default_topology().edge_ip}:{EDGE_RECEIVE_PORT}",
        "remote_receipt": True,
        "cross_host_latency_computed": False,
    }


@dataclass(frozen=True)
class PrestartedRemoteEdgeV1:
    plan_sha256: str
    attempt_id: str
    project_name: str
    container_id: str
    ready_sha256: str
    gt_ready_sha256: str
    feedback_route_sha256: str

    def validate(self) -> "PrestartedRemoteEdgeV1":
        for value in (self.plan_sha256, self.ready_sha256,
                      self.gt_ready_sha256, self.feedback_route_sha256):
            _require(len(value) == 64 and all(c in "0123456789abcdef" for c in value),
                     "prestarted remote-edge digest is invalid")
        _require(bool(self.attempt_id) and bool(self.container_id),
                 "prestarted remote-edge identity is empty")
        _require(self.project_name == f"run4-edge-l10319-{self.attempt_id}",
                 "prestarted remote-edge project identity drift")
        return self


def validate_prestarted_remote_edge(
        *, plan: RE.RemoteEdgeLifecyclePlan,
        container_observation: Mapping[str, Any],
        ready_record: Mapping[str, Any],
        gt_ready_record: Mapping[str, Any],
        feedback_route: Mapping[str, Any],
) -> PrestartedRemoteEdgeV1:
    """Bind all remote readiness facts without granting local ownership."""
    _require(plan.live_run_authorized is False,
             "remote startup plan unexpectedly authorizes a live run")
    RE.validate_container_observation(container_observation, plan=plan)
    RE.validate_ready_record(ready_record, plan=plan)
    RE.validate_gt_ready_record(gt_ready_record, plan=plan)
    validated_route = validate_remote_feedback_route(feedback_route)
    evidence = plan.as_evidence()
    return PrestartedRemoteEdgeV1(
        plan_sha256=_canonical_sha256(evidence),
        attempt_id=plan.invocation.attempt_id,
        project_name=plan.invocation.project_name,
        container_id=str(container_observation["container_id"]),
        ready_sha256=_canonical_sha256(dict(ready_record)),
        gt_ready_sha256=_canonical_sha256(dict(gt_ready_record)),
        feedback_route_sha256=_canonical_sha256(validated_route),
    ).validate()


@dataclass(frozen=True)
class RemoteEvidenceRetrievalPlanV1:
    """Ordering owned by the remote lifecycle, never by the local child."""

    schema: str
    remote_attempt_root: str
    local_destination: str
    required_relative_paths: tuple[str, ...]
    transfer_owner: str = "REMOTE_LIFECYCLE_OWNER_AFTER_LOCAL_SENDER_CLOSE"
    local_may_stop_remote_cn: bool = False
    plan_only: bool = True
    operationally_consumed_by_local_coordinator: bool = False

    def validate(self) -> "RemoteEvidenceRetrievalPlanV1":
        _require(self.schema == REMOTE_RETRIEVAL_SCHEMA,
                 "remote evidence retrieval schema drift")
        for value, label in ((self.remote_attempt_root, "remote attempt root"),
                             (self.local_destination, "local destination")):
            path = Path(value)
            _require(path.is_absolute() and str(path) not in {"/", "/tmp", "/home"},
                     f"{label} is unsafe")
        required = {
            "state/ready.json", "state/remote_gt_listener_ready.json",
            "state/remote_gt_listener_final.json",
            "evidence/run4_phase6_edge_report.json", "OUTPUT_MANIFEST.json",
        }
        _require(set(self.required_relative_paths) == required,
                 "remote evidence retrieval set drift")
        _require(self.transfer_owner
                 == "REMOTE_LIFECYCLE_OWNER_AFTER_LOCAL_SENDER_CLOSE",
                 "remote evidence transfer ownership drift")
        _require(self.local_may_stop_remote_cn is False,
                 "local coordinator cannot stop the remote CN")
        _require(self.plan_only is True
                 and self.operationally_consumed_by_local_coordinator is False,
                 "remote evidence retrieval is a plan-only external obligation")
        return self


@dataclass(frozen=True)
class SplitHostCoordinatorPrerequisitesV1:
    remote: PrestartedRemoteEdgeV1
    source_route: Mapping[str, Any]
    radio_tensor_path: Mapping[str, Any]
    policy_ownership: PolicyRoutingOwnershipV1

    def validate(self) -> "SplitHostCoordinatorPrerequisitesV1":
        topology = C.default_topology()
        _require(self.source_route.get("radio_path") == "PASS"
                 and self.source_route.get("device") == POLICY_INTERFACE
                 and self.source_route.get("table") == POLICY_TABLE,
                 "UE source-route release gate is absent")
        _require(self.radio_tensor_path.get("schema") == RADIO_PROOF_SCHEMA
                 and self.radio_tensor_path.get("remote_receipt") is True
                 and self.radio_tensor_path.get("cross_host_latency_computed") is False,
                 "radio tensor-path release gate is absent")
        _require(self.radio_tensor_path.get("destination")
                 == f"{topology.edge_ip}:{EDGE_RECEIVE_PORT}",
                 "radio tensor-path destination drift")
        self.policy_ownership.validate()
        self.remote.validate()
        return self


@dataclass(frozen=True)
class SplitHostCoordinatorPreRunV1:
    """Facts sufficient to arm the first real split-host decision.

    The tensor-path proof cannot exist until that decision is transmitted.
    Keeping this type distinct from ``SplitHostCoordinatorPrerequisitesV1``
    prevents an observer from fabricating a pre-run PASS merely to satisfy a
    circular prerequisite.  ``finalize`` is the only transition to the full
    release contract.
    """

    remote: PrestartedRemoteEdgeV1
    source_route: Mapping[str, Any]
    policy_ownership: PolicyRoutingOwnershipV1

    def validate(self) -> "SplitHostCoordinatorPreRunV1":
        _require(self.source_route.get("radio_path") == "PASS"
                 and self.source_route.get("device") == POLICY_INTERFACE
                 and self.source_route.get("table") == POLICY_TABLE,
                 "UE source-route release gate is absent")
        self.policy_ownership.validate()
        self.remote.validate()
        return self

    def finalize(
        self, capture: "RadioTensorObservationV1",
        receipt: "RadioTensorObservationV1",
    ) -> SplitHostCoordinatorPrerequisitesV1:
        self.validate()
        return SplitHostCoordinatorPrerequisitesV1(
            remote=self.remote,
            source_route=self.source_route,
            radio_tensor_path=validate_radio_tensor_path(capture, receipt),
            policy_ownership=self.policy_ownership,
        ).validate()


def build_remote_abort_release(
    *, remote: PrestartedRemoteEdgeV1,
    sender_ever_connected: bool,
    local_gt_sender_final_sha256: Optional[str],
) -> Optional[Mapping[str, Any]]:
    """Build the exact-attempt abort release, or no release if never connected.

    This function performs no cleanup and cannot authorize a scientific PASS.
    A never-connected local sender has nothing to release; supplying a digest
    in that case is refused instead of creating misleading closure evidence.
    """
    remote.validate()
    _require(type(sender_ever_connected) is bool,
             "abort sender-connected fact must be exact bool")
    if not sender_ever_connected:
        _require(local_gt_sender_final_sha256 is None,
                 "never-connected sender cannot have an abort-release digest")
        return None
    _require(_is_sha256(local_gt_sender_final_sha256),
             "connected sender lacks a valid final-evidence digest")
    return {
        "schema": REMOTE_ABORT_RELEASE_SCHEMA,
        "local_gt_sender_was_connected": True,
        "local_gt_sender_closed": True,
        "local_gt_sender_final_sha256": local_gt_sender_final_sha256,
        "remote_attempt_id": remote.attempt_id,
        "remote_project_name": remote.project_name,
        "remote_plan_sha256": remote.plan_sha256,
        "scientific_pass": False,
        "remote_cn_teardown_permitted_to_local_coordinator": False,
    }


@dataclass(frozen=True)
class SplitHostCoordinatorPlanV1:
    schema: str
    local_ran: LR.LocalRanPlan
    prerequisites: SplitHostCoordinatorPrerequisitesV1
    map_endpoint: str
    tensor_endpoint: str
    feedback_endpoint: str
    gt_sideband_endpoint: str
    gt_sideband_source: str
    local_cn_operations: bool
    local_edge_operations: bool
    remote_cn_teardown_permitted: bool
    policy_deadline_clock: str
    cross_host_monotonic_comparison_permitted: bool
    live_run_authorized: bool

    def validate(self) -> "SplitHostCoordinatorPlanV1":
        topology = C.default_topology()
        _require(self.schema == SCHEMA, "coordinator schema drift")
        self.local_ran.validate()
        self.prerequisites.validate()
        _require(self.map_endpoint == f"{topology.map_ip}:{topology.map_port}",
                 "local map endpoint drift")
        _require(self.tensor_endpoint == f"{topology.edge_ip}:{EDGE_RECEIVE_PORT}",
                 "remote tensor endpoint drift")
        _require(self.feedback_endpoint == f"{UE_CONTROL_HOST}:{UE_CONTROL_PORT}",
                 "radio feedback endpoint drift")
        _require(self.gt_sideband_endpoint
                 == f"{topology.edge_ip}:{GSI.REGISTERED_PORT}",
                 "GT sideband endpoint drift")
        _require(self.gt_sideband_source == topology.local_lan_ip,
                 "GT sideband source must be W10275 LAN")
        _require(not self.local_cn_operations and not self.local_edge_operations,
                 "coordinator gained local CN/edge ownership")
        _require(not self.remote_cn_teardown_permitted,
                 "coordinator gained remote CN teardown authority")
        _require(self.policy_deadline_clock == "CLOCK_MONOTONIC_RAW_ON_W10275_ONLY"
                 and not self.cross_host_monotonic_comparison_permitted,
                 "policy deadline left the local monotonic clock")
        _require(self.live_run_authorized is False,
                 "offline coordinator cannot authorize the live run")
        return self


def build_coordinator_plan(
        *, state_dir: Path,
        prerequisites: SplitHostCoordinatorPrerequisitesV1,
        root: Path = LR.ROOT,
) -> SplitHostCoordinatorPlanV1:
    topology = C.default_topology()
    return SplitHostCoordinatorPlanV1(
        schema=SCHEMA,
        local_ran=LR.build_local_ran_plan(state_dir, root=root),
        prerequisites=prerequisites,
        map_endpoint=f"{topology.map_ip}:{topology.map_port}",
        tensor_endpoint=f"{topology.edge_ip}:{EDGE_RECEIVE_PORT}",
        feedback_endpoint=f"{UE_CONTROL_HOST}:{UE_CONTROL_PORT}",
        gt_sideband_endpoint=f"{topology.edge_ip}:{GSI.REGISTERED_PORT}",
        gt_sideband_source=topology.local_lan_ip,
        local_cn_operations=False,
        local_edge_operations=False,
        remote_cn_teardown_permitted=False,
        policy_deadline_clock="CLOCK_MONOTONIC_RAW_ON_W10275_ONLY",
        cross_host_monotonic_comparison_permitted=False,
        live_run_authorized=False,
    ).validate()


def split_host_map_endpoint(*, port: int, **_ignored: Any) -> Any:
    """Static registered LAN endpoint; never inspects a local Docker bridge."""
    topology = C.default_topology()
    _require(type(port) is int and port == topology.map_port,
             "split-host map port drift")
    from rl_agent.splitfusion_direct_edge_map_v1.endpoint import DirectMapEndpoint
    return DirectMapEndpoint(
        host=topology.map_ip, port=topology.map_port,
        network_name="SPLIT_HOST_LAN", subnet="10.21.16.0/24",
        bridge_interface="", resolution="split_host_contract_static_lan_endpoint",
    )


def _default_modules() -> Any:
    from rl_agent import ue_map_install_feedback_v1 as feedback
    from rl_agent import ue_route_b_split_cell_adapter_v1 as pinned
    from rl_agent.splitfusion_direct_edge_map_v1 import adapter_direct_v1 as direct
    from rl_agent.splitfusion_quality_feedback_probe_v1 import adapter_quality_v1 as quality
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_live_child_v2 as child
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_live_child_nobuild_v2 as nobuild
    return SimpleNamespace(child=child, nobuild=nobuild, pinned=pinned,
                           direct=direct, quality=quality, feedback=feedback)


class SplitHostPhase6ChildContextV1:
    """Process-local adapter around the unchanged Phase-6 child.

    ``install`` first invokes the frozen no-build/recorder stack with only the
    endpoint resolver temporarily replaced.  The GT sender is then wrapped
    around the already-recorded writers.  The local-edge hooks are replaced by
    an attempt-local proxy; no Docker operation is reachable through them.
    ``close`` restores every process-global touched by this adapter.
    """

    def __init__(
        self, *, prerequisites: (
            SplitHostCoordinatorPrerequisitesV1 | SplitHostCoordinatorPreRunV1
        ),
        retrieval: RemoteEvidenceRetrievalPlanV1,
        modules: Any = None,
        sender_factory: Callable[..., Any] = GSI.HighWorkerGtSenderV1,
        decision_cap: Optional[int] = None,
    ) -> None:
        self.prerequisites = prerequisites.validate()
        self._final_prerequisites = (
            self.prerequisites
            if type(self.prerequisites) is SplitHostCoordinatorPrerequisitesV1
            else None
        )
        self.retrieval = retrieval.validate()
        self.modules = modules if modules is not None else _default_modules()
        self.sender_factory = sender_factory
        _require(decision_cap is None or type(decision_cap) is int and decision_cap == 1,
                 "bounded split-host decision cap must be exactly one")
        self.decision_cap = decision_cap
        self._saved: list[tuple[Any, str, Any]] = []
        self._saved_endpoint: Optional[dict[str, Any]] = None
        self._decision_cap_active = False
        self._saved_decision_cap: Any = None
        self._installed = False
        self._closed = False
        self._sender: Any = None
        self._sender_connected = False
        self._sender_ever_connected = False
        self._sender_snapshot_written = False
        self._runtime: Any = None
        self._attempt_dir: Optional[Path] = None

    def _remember(self, owner: Any, name: str) -> None:
        self._saved.append((owner, name, getattr(owner, name)))

    def __enter__(self) -> "SplitHostPhase6ChildContextV1":
        _require(not self._installed and not self._closed,
                 "split-host context cannot be re-entered")
        M = self.modules
        for owner, names in (
            (M.child, ("install_run4_seams", "verify_feedback_path")),
            (M.pinned, ("start_map_process", "start_live_edge", "stop_live_edge",
                        "stop_tail", "LivePilotCellRuntime", "SceneSnapshotSource",
                        "PassiveSplitCollector", "seed_cell_edge_state")),
            (M.direct, ("resolve_direct_map_endpoint", "DIRECT_EDGE_MODULE",
                        "DIRECT_MAP_SERVER", "subprocess")),
            (M.quality, ("write_object_ground_truth", "write_semantic_ground_truth")),
            (M.feedback, ("InstallFeedbackLedger", "FIELDS")),
        ):
            for name in names:
                self._remember(owner, name)
        self._saved_endpoint = dict(M.direct._ENDPOINT)
        if self.decision_cap is not None:
            cap = getattr(M.nobuild, "_DECISION_CAP", None)
            _require(type(cap) is dict and set(cap) == {"value"},
                     "frozen child decision-cap seam is absent or drifted")
            self._saved_decision_cap = cap["value"]
            cap["value"] = self.decision_cap
            self._decision_cap_active = True
        M.child.install_run4_seams = self.install
        M.child.verify_feedback_path = self.verify_feedback_path
        self._installed = True
        return self

    def _resolve_identity(self, gt_identity: Mapping[str, Any]) -> Any:
        _require(self._runtime is not None, "GT identity arrived before runtime creation")
        identities = getattr(self._runtime, "_run4_identity", None)
        _require(type(identities) is dict, "Run-4 runtime identity map is absent")
        frame = int(gt_identity["frame_id"])
        _require(frame in identities, "Run-4 identity is absent for GT frame")
        return GSI.identity_from_ue_maps(
            gt_identity=dict(gt_identity), run4_identity=identities[frame]
        )

    @staticmethod
    def _is_high_worker() -> bool:
        return threading.current_thread().name == GT_WORKER_NAME

    def install(self, campaign: Mapping[str, Any], *, attempt_dir: Path,
                **kwargs: Any) -> Any:
        _require(self._installed and not self._closed,
                 "split-host context is not active")
        validate_campaign_binding(campaign)
        self._attempt_dir = Path(attempt_dir).resolve()
        M = self.modules

        # Bypass only the local-Docker endpoint resolver while the unchanged
        # installer binds all other scientific/runtime seams.
        previous_resolver = M.direct.resolve_direct_map_endpoint
        M.direct.resolve_direct_map_endpoint = split_host_map_endpoint
        try:
            seams = M.nobuild.install_run4_seams_nobuild(
                campaign, attempt_dir=self._attempt_dir, **kwargs
            )
        finally:
            M.direct.resolve_direct_map_endpoint = previous_resolver

        endpoint = M.direct._ENDPOINT.get("endpoint")
        _require(endpoint is not None
                 and (str(endpoint.host), int(endpoint.port))
                 == (C.default_topology().map_ip, C.default_topology().map_port),
                 "base installer did not retain the split-host map endpoint")

        # Capture the real runtime produced by the existing factory.  The
        # sender resolver reads only its already-existing identity map.
        base_factory = M.pinned.LivePilotCellRuntime

        def runtime_factory(**factory_kwargs: Any) -> Any:
            runtime = base_factory(**factory_kwargs)
            self._runtime = runtime
            return runtime

        M.pinned.LivePilotCellRuntime = runtime_factory
        self._sender = self.sender_factory(
            resolve_high_identity=self._resolve_identity,
            is_high_worker=self._is_high_worker,
            host=C.default_topology().edge_ip,
            port=GSI.REGISTERED_PORT,
        )
        M.quality.write_object_ground_truth, M.quality.write_semantic_ground_truth = (
            self._sender.wrap_after_recorder(
                M.quality.write_object_ground_truth,
                M.quality.write_semantic_ground_truth,
            )
        )

        def start_remote_proxy(_campaign: Mapping[str, Any], _cell: Mapping[str, Any],
                               _temporary_dir: Path) -> Path:
            del _campaign, _cell, _temporary_dir
            _require(not self._sender_connected,
                     "split-host remote-edge proxy started twice")
            root = self._attempt_dir / LOCAL_PROXY_ROOT
            evidence = root / EDGE_EVIDENCE_LEAF
            evidence.mkdir(parents=True, exist_ok=False)
            self._sender.connect()
            self._sender_connected = True
            self._sender_ever_connected = True
            return root

        def stop_remote_proxy(_scratch: Optional[Path]) -> bool:
            del _scratch
            self._close_and_snapshot_sender()
            # This reports release of the local proxy only.  The remote owner
            # must stop the remote listener after this method has returned.
            return True

        def no_local_tail_stop() -> bool:
            return True

        M.pinned.start_live_edge = start_remote_proxy
        M.pinned.stop_live_edge = stop_remote_proxy
        M.pinned.stop_tail = no_local_tail_stop
        return seams

    def verify_feedback_path(self, *, map_host: str, map_port: int,
                             ue_host: str, ue_port: int, **_ignored: Any) -> Mapping[str, Any]:
        topology = C.default_topology()
        _require((str(map_host), int(map_port)) == (topology.map_ip, topology.map_port),
                 "unchanged child supplied a foreign map endpoint")
        _require((str(ue_host), int(ue_port)) == (UE_CONTROL_HOST, UE_CONTROL_PORT),
                 "unchanged child supplied a foreign feedback endpoint")
        return {
            "map_endpoint": f"{map_host}:{map_port}",
            "ue_endpoint": f"{ue_host}:{ue_port}",
            "edge_route_to_ue": "PREVALIDATED_REMOTE_DOCKER_EXEC_ROUTE",
            "remote_feedback_route_sha256":
                self.prerequisites.remote.feedback_route_sha256,
            "via_upf": True,
            "local_docker_exec_used": False,
        }

    def _close_and_snapshot_sender(self) -> None:
        if self._sender is None:
            return
        if self._sender_snapshot_written:
            return
        snapshot = dict(self._sender.close())
        self._sender_connected = False
        _require(self._attempt_dir is not None, "sender has no attempt directory")
        target = self._attempt_dir / LOCAL_GT_FINAL
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(snapshot, sort_keys=True, indent=2) + "\n")
        self._sender_snapshot_written = True

    def finalize_radio_tensor_path(
        self, capture: RadioTensorObservationV1,
        receipt: RadioTensorObservationV1,
    ) -> SplitHostCoordinatorPrerequisitesV1:
        """Bind the first real decision proof after both captures close."""
        _require(self._installed and not self._closed,
                 "radio tensor-path proof finalized outside active context")
        _require(self._final_prerequisites is None,
                 "radio tensor-path proof was already finalized")
        _require(type(self.prerequisites) is SplitHostCoordinatorPreRunV1,
                 "full prerequisites cannot be finalized a second time")
        self._final_prerequisites = self.prerequisites.finalize(capture, receipt)
        return self._final_prerequisites

    @property
    def sender_ever_connected(self) -> bool:
        """Whether this attempt ever completed the local GT sender connect."""
        return self._sender_ever_connected

    def remote_abort_release(self) -> Optional[Mapping[str, Any]]:
        """Return exact-attempt abort evidence; never a scientific release."""
        digest: Optional[str] = None
        if self._sender_ever_connected:
            _require(self._sender_snapshot_written and self._attempt_dir is not None,
                     "connected sender was not closed before abort release")
            path = self._attempt_dir / LOCAL_GT_FINAL
            _require(path.is_file(), "local GT sender final evidence is absent")
            digest = _sha256_file(path)
        return build_remote_abort_release(
            remote=self.prerequisites.remote,
            sender_ever_connected=self._sender_ever_connected,
            local_gt_sender_final_sha256=digest,
        )

    def remote_teardown_release(self) -> Mapping[str, Any]:
        """Release only after local GT closure and first-real tensor proof."""
        _require(self._sender_snapshot_written and self._attempt_dir is not None,
                 "remote teardown requested before local GT sender snapshot")
        _require(self._final_prerequisites is not None,
                 "remote teardown requested before radio tensor-path proof")
        path = self._attempt_dir / LOCAL_GT_FINAL
        _require(path.is_file(), "local GT sender final evidence is absent")
        return {
            "schema": REMOTE_TEARDOWN_RELEASE_SCHEMA,
            "local_gt_sender_closed": True,
            "local_gt_sender_final_sha256": _sha256_file(path),
            "radio_tensor_path_sha256": _canonical_sha256(
                self._final_prerequisites.radio_tensor_path),
            "remote_attempt_id": self._final_prerequisites.remote.attempt_id,
            "remote_project_name": self._final_prerequisites.remote.project_name,
            "remote_plan_sha256": self._final_prerequisites.remote.plan_sha256,
            "remote_teardown_owner": "REMOTE_LIFECYCLE_OWNER_ONLY",
            "remote_cn_teardown_permitted_to_local_coordinator": False,
        }

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._close_and_snapshot_sender()
        finally:
            M = self.modules
            if self._decision_cap_active:
                M.nobuild._DECISION_CAP["value"] = self._saved_decision_cap
                self._decision_cap_active = False
            for owner, name, value in reversed(self._saved):
                setattr(owner, name, value)
            M.direct._ENDPOINT.clear()
            M.direct._ENDPOINT.update(self._saved_endpoint or {})
            self._closed = True

    def __exit__(self, _kind: Any, _value: Any, _traceback: Any) -> None:
        self.close()


def run_unchanged_child(args: Any, *,
                        prerequisites: SplitHostCoordinatorPrerequisitesV1,
                        retrieval: RemoteEvidenceRetrievalPlanV1,
                        modules: Any = None,
                        sender_factory: Callable[..., Any] = GSI.HighWorkerGtSenderV1) -> int:
    """Invoke the existing child under process-local seams.

    This function does not start the local RAN or the remote CN/edge.  Those
    are independently owned prerequisites.  It also cannot stop them.
    """
    _require(type(prerequisites) is SplitHostCoordinatorPrerequisitesV1,
             "generic child helper requires full radio-proof prerequisites")
    context = SplitHostPhase6ChildContextV1(
        prerequisites=prerequisites, retrieval=retrieval,
        modules=modules, sender_factory=sender_factory,
    )
    with context:
        return int(context.modules.child.run(args))


__all__ = [
    "SCHEMA", "FEEDBACK_ROUTE_SCHEMA", "RADIO_PROOF_SCHEMA",
    "POLICY_OWNERSHIP_SCHEMA", "REMOTE_RETRIEVAL_SCHEMA",
    "REMOTE_TEARDOWN_RELEASE_SCHEMA", "REMOTE_ABORT_RELEASE_SCHEMA",
    "EDGE_RECEIVE_PORT", "UE_CONTROL_HOST", "UE_CONTROL_PORT",
    "POLICY_TABLE", "POLICY_INTERFACE", "SplitHostCoordinatorError",
    "validate_campaign_binding", "source_route_probe_argv",
    "validate_ue_source_route", "OwnedPolicyRuleV1", "OwnedPolicyRouteV1",
    "PolicyRoutingOwnershipV1", "validate_remote_feedback_route",
    "RadioTensorObservationV1", "validate_radio_tensor_path",
    "PrestartedRemoteEdgeV1", "validate_prestarted_remote_edge",
    "RemoteEvidenceRetrievalPlanV1", "SplitHostCoordinatorPrerequisitesV1",
    "SplitHostCoordinatorPreRunV1", "build_remote_abort_release",
    "SplitHostCoordinatorPlanV1", "build_coordinator_plan",
    "split_host_map_endpoint", "SplitHostPhase6ChildContextV1",
    "run_unchanged_child",
]
