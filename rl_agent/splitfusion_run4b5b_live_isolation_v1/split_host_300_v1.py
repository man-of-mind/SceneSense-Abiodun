"""Split-host 300-transmitted-frame B characterization (W10275 -> L10319).

This is the production 300-frame entry point over the already-working
one-frame factory and lifecycle.  It changes only what a 300-frame run needs:

* an exact 300-frame config schema, UE request and edge request (the
  one-frame schemas are unchanged and cannot be confused with these);
* the registered Run-4 action hold: a new policy decision opens only when the
  previous ticket is resolved *and* ``K_MIN`` tensors of its action were
  transmitted; held tensors reuse the active action exactly with
  ``reward_requested = False``; the registered external fallback (mode 11,
  q_e4 9800) is used, without calling the actor or opening a ticket, only
  when the causal radio state refuses a decision;
* the uplink keepalive stays active for the whole run;
* the existing ``b_ue_process_v1.execute_300`` loop and the GT-free
  tail-output ACK; ground truth is only spooled raw during the live phase;
* after the live phase and service teardown (before the final CARLA stop):
  GT materialization, prediction download, decision-only Q_perc scoring.
  A post-run failure is recorded as quality analysis incomplete and never
  rewrites the operational outcome.

Importing this module performs no I/O and launches nothing.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import json
import math
import os
import queue
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import (
    phase6_decision_engine_v2 as P6E,
    reward_hold_controller_v2 as RHC,
)
from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as R4C
from rl_agent.splitfusion_hybrid_sac_run4b_v1 import contract as R4B

from . import b_edge_engineering_request_v2 as ER
from . import b_one_frame_production_factory_v1 as FACTORY
from . import b_one_frame_processor_v2 as P2
from . import b_opportunity_processor_v1 as BASE
from . import b_route_bridge_v4 as V4
from . import b_ue_process_v1 as UE
from . import b_validation_runner_v1 as VALID
from . import final_actor_gate_v2 as F
from . import one_frame_config_builder_v1 as BUILDER
from . import one_frame_engineering_v1 as O
from . import operational_ack_v1 as ACK
from . import production_lifecycle_adapter_v1 as LIFE


CONFIG_SCHEMA = "scenesense.splitfusion.run4b5b.split_host_300.v1"
SEAL_SCHEMA = "scenesense.splitfusion.run4b5b.split_host_300_seal.v1"
RESULT_SCHEMA = "scenesense.splitfusion.run4b5b.split_host_300_result.v1"
TIMING_SCHEMA = "scenesense.splitfusion.run4b5b.split_host_300_frame.v1"
POSTRUN_SCHEMA = "scenesense.splitfusion.run4b5b.split_host_300_postrun.v1"
PURPOSE = "SPLIT_HOST_300_FRAME_CHARACTERIZATION__NOT_POLICY_QUALIFICATION"
EXECUTE_TOKEN = "SPLITFUSION_RUN4B5B_SPLIT_HOST_300_V1_EXECUTE"
FACTORY_MODULE = (
    "rl_agent.splitfusion_run4b5b_live_isolation_v1.split_host_300_v1")
TRANSMITTED_BUDGET = UE.TRANSMITTED_BUDGET
MAXIMUM_LOOP_SIM_S = 600.0
K_MIN = RHC.K_MIN
FALLBACK = dict(P6E.FALLBACK)
FALLBACK_SEQ = P6E.FALLBACK_SEQ
FRAME_KINDS = ("POLICY_DECISION", "POLICY_HOLD", "FALLBACK")

if TRANSMITTED_BUDGET != 300 or ER.TRANSMITTED_BUDGET_300 != 300 or K_MIN != 2:
    raise RuntimeError("registered 300-frame/hold constants drifted")


class SplitHost300Error(O.OneFrameEngineeringError):
    """The split-host 300-frame contract was violated."""


def _require(value: bool, message: str) -> None:
    if not value:
        raise SplitHost300Error(message)


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise SplitHost300Error("value is not canonical JSON") from exc


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True,
                          allow_nan=False, default=str) + "\n").encode("ascii")
    with Path(path).open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Split300ConfigV1:
    run_id: str
    cell_id: str
    variant: str
    actor_manifest_path: Path
    actor_manifest_sha256: str
    actor_weights_path: Path
    actor_weights_sha256: str
    actor_evidence_root: Path
    local_repository: Path
    remote_repository: Path
    local_attempt_root: Path
    remote_attempt_root: Path
    edge_campaign_config: Path
    route_config: Path
    network: O.NetworkBindingV1
    transmitted_budget: int
    deadline_ns: int
    clock_domain: str
    ack_semantics: str
    postrun_semantics: str
    purpose: str
    policy_performance_claim: bool
    maximum_loop_sim_s: float
    factory_module: str

    def __post_init__(self) -> None:
        _require(type(self.run_id) is str
                 and bool(O._RUN.fullmatch(self.run_id)), "run_id is unsafe")
        _require(type(self.cell_id) is str
                 and bool(O._CELL.fullmatch(self.cell_id)), "cell_id is unsafe")
        _require(self.variant in {F.RUN4B_VARIANT, F.RUN5B_VARIANT},
                 "variant is not an exact final B actor")
        for name in (
            "actor_manifest_path", "actor_weights_path", "actor_evidence_root",
            "local_repository", "remote_repository", "local_attempt_root",
            "remote_attempt_root", "edge_campaign_config", "route_config",
        ):
            object.__setattr__(self, name,
                               O._absolute(str(getattr(self, name)), name))
        O._sha(self.actor_manifest_sha256, "actor_manifest_sha256")
        O._sha(self.actor_weights_sha256, "actor_weights_sha256")
        _require(self.actor_manifest_path.name.endswith("MANIFEST_V2.json"),
                 "actor manifest filename is not the final V2 authority")
        _require(self.actor_weights_path.name == "actor_state_dict.pt",
                 "actor weights filename drift")
        _require(self.local_attempt_root != self.remote_attempt_root,
                 "local and remote attempt roots overlap")
        _require(self.local_repository != self.local_attempt_root
                 and self.remote_repository != self.remote_attempt_root,
                 "attempt root equals a repository")
        _require(type(self.network) is O.NetworkBindingV1,
                 "network binding has a foreign type")
        _require(type(self.transmitted_budget) is int
                 and self.transmitted_budget == TRANSMITTED_BUDGET,
                 "split-host budget must be exactly 300 transmitted frames")
        _require(self.deadline_ns == O.DEADLINE_NS,
                 "operational ACK deadline drift")
        _require(self.clock_domain == O.CLOCK_DOMAIN, "clock domain drift")
        _require(self.ack_semantics == O.ACK_SEMANTICS,
                 "operational ACK semantics drift")
        _require(self.postrun_semantics == O.POSTRUN_SEMANTICS,
                 "post-run semantics drift")
        _require(self.purpose == PURPOSE, "split-host purpose drift")
        _require(self.policy_performance_claim is False,
                 "characterization may not claim policy performance")
        _require(type(self.maximum_loop_sim_s) is float
                 and self.maximum_loop_sim_s == MAXIMUM_LOOP_SIM_S,
                 "route safety bound drift")
        _require(self.factory_module == FACTORY_MODULE,
                 "split-host factory module drift")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "Split300ConfigV1":
        fields = {"schema", *cls.__dataclass_fields__}
        _require(type(raw) is dict and set(raw) == fields,
                 "split-host config fields are incomplete or foreign")
        _require(raw["schema"] == CONFIG_SCHEMA, "split-host schema drift")
        lowered = _canonical(raw).decode("ascii").lower()
        for forbidden in ("quality_ack", "live_qperc", "gt_feedback",
                          "reward_ticket", "map_install_ack"):
            _require(forbidden not in lowered,
                     f"retired live-quality path present: {forbidden}")
        values = dict(raw)
        values.pop("schema")
        values["network"] = O.NetworkBindingV1.from_mapping(values["network"])
        return cls(**values)

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"schema": CONFIG_SCHEMA}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, Path):
                value = str(value)
            elif isinstance(value, O.NetworkBindingV1):
                value = value.as_dict()
            result[name] = value
        return result

    def binding_sha256(self) -> str:
        return hashlib.sha256(_canonical(self.as_dict())).hexdigest()

    def lifecycle_settings(self) -> LIFE.ProductionLifecycleSettingsV1:
        return LIFE.ProductionLifecycleSettingsV1(
            local_repository=self.local_repository,
            remote_repository=self.remote_repository,
            remote_attempt_base=self.remote_attempt_root.parent)


def build_config(**kwargs: Any) -> Split300ConfigV1:
    """Reuse the one-frame builder's path/network checks, then bind 300."""
    base = BUILDER.build(**kwargs)
    values = {name: getattr(base, name)
              for name in Split300ConfigV1.__dataclass_fields__
              if name in base.__dataclass_fields__}
    values.update(transmitted_budget=TRANSMITTED_BUDGET, purpose=PURPOSE,
                  policy_performance_claim=False,
                  maximum_loop_sim_s=MAXIMUM_LOOP_SIM_S,
                  factory_module=FACTORY_MODULE)
    return Split300ConfigV1(**values)


def seal(config: Split300ConfigV1) -> dict[str, Any]:
    _require(type(config) is Split300ConfigV1, "config type is foreign")
    return {"schema": SEAL_SCHEMA, "binding_sha256": config.binding_sha256(),
            "config": config.as_dict()}


def load_config(path: Path) -> Split300ConfigV1:
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(),
             "sealed split-host config is absent or a symlink")
    raw = json.loads(path.read_text(encoding="utf-8"))
    _require(type(raw) is dict
             and set(raw) == {"schema", "binding_sha256", "config"},
             "sealed config envelope is incomplete or foreign")
    _require(raw["schema"] == SEAL_SCHEMA, "sealed config schema drift")
    config = Split300ConfigV1.from_mapping(raw["config"])
    _require(raw["binding_sha256"] == config.binding_sha256(),
             "sealed config binding differs")
    return config


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------
def build_ue_request_300(config: Split300ConfigV1,
                         actor: F.LoadedFinalActorV2) -> UE.BUEProcessRequestV1:
    """The frozen 300-frame UE request, validated by its own contract."""
    request = dataclasses.replace(
        FACTORY.build_ue_request(config, actor),
        transmitted_budget=TRANSMITTED_BUDGET,
        ack_semantics=VALID.ACK_SEMANTICS,
        postrun_semantics=VALID.POSTRUN_SEMANTICS)
    request.validate()
    return request


def build_edge_request_300(config: Split300ConfigV1,
                           actor: F.LoadedFinalActorV2
                           ) -> tuple[str, dict[str, Any]]:
    """The one-frame edge request fields under the separate 300 contract."""
    _, base = FACTORY.build_edge_request(config, actor)
    raw = {**base, "schema": ER.SCHEMA_300, "purpose": ER.PURPOSE_300,
           "claim_scope": ER.CLAIM_SCOPE_300,
           "transmitted_budget": ER.TRANSMITTED_BUDGET_300}
    encoded = base64.urlsafe_b64encode(_canonical(raw)).decode(
        "ascii").rstrip("=")
    decoded = ER.decode_and_validate_300(encoded)
    _require(decoded == raw, "300-frame edge request round-trip differs")
    return encoded, decoded


# ---------------------------------------------------------------------------
# Registered hold gate and hold-aware processor
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ActiveDecisionV1:
    identity: ACK.FrameActionIdentityV1
    profile: Any
    decision_seq: int


class OperationalHoldGateV1:
    """``RewardHoldControllerV2.can_decide`` resolved by the operational ACK.

    A new decision is permitted only when no ticket exists, or the active
    ticket is resolved and at least ``K_MIN`` tensors of its action were
    transmitted.  The sequential UE loop resolves every ticket (ACK or
    170-ms timeout) before it asks for the next frame, so the resolution is
    the ``previous`` operational outcome; any other value fails closed.
    """

    def __init__(self) -> None:
        self.active: Optional[ActiveDecisionV1] = None
        self.tensors = 0
        self.next_decision_seq = 0

    def must_hold(self, previous: Optional[ACK.OperationalOutcomeV1]) -> bool:
        if self.active is None:
            _require(previous is None,
                     "an operational outcome exists before any decision")
            return False
        _require(type(previous) is ACK.OperationalOutcomeV1
                 and previous.identity == self.active.identity,
                 "the active decision is unresolved at the next frame")
        return self.tensors < K_MIN

    def opened(self, decision: ActiveDecisionV1) -> None:
        _require(decision.decision_seq == self.next_decision_seq,
                 "decision sequence is not contiguous")
        self.active, self.tensors = decision, 1
        self.next_decision_seq += 1

    def held(self) -> None:
        _require(self.active is not None, "a hold has no active action")
        self.tensors += 1


def observation_from_features(features: Sequence[float],
                              scaling: R4B.ScalingV1) -> dict[str, Any]:
    """Evidence only: invert the first four registered state features."""
    return {
        "camera_si": features[0] * scaling.camera_si_scale
        + scaling.camera_si_center,
        "radar_p40": features[1],
        "prior_ul_mcs": int(round(features[2] * (R4B.UL_MCS_MAX
                                                 - R4B.UL_MCS_MIN)
                                  + R4B.UL_MCS_MIN)),
        "pre_action_rlc_backlog_bytes": int(round(math.expm1(
            features[3] * scaling.backlog_log1p_scale))),
    }


class HoldGatedOpportunityProcessorV1(P2.OneFrameOpportunityProcessorV2):
    """The one-frame decision mechanics plus the registered hold/fallback."""

    def __init__(self, *, timing_path: Path, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.gate = OperationalHoldGateV1()
        self.timing_path = Path(timing_path)
        self.counters = {kind: 0 for kind in FRAME_KINDS}
        self.fallback_profile = self.contract.resolve_q_e4(
            FALLBACK["mode_id"], FALLBACK["q_e4"])
        profile = self.fallback_profile
        _require((profile.action_id, profile.profile_id, profile.family,
                  profile.quantizer)
                 == (FALLBACK["anchor_action_id"], FALLBACK["profile_id"],
                     FALLBACK["family"], FALLBACK["quantizer"]),
                 "registered fallback identity does not reconcile")
        self._rows = 0

    def __call__(self, opportunity: V4.RouteOpportunityV4,
                 previous: Optional[ACK.OperationalOutcomeV1]
                 ) -> UE.BTransmissionV1:
        stamps: dict[str, int] = {}
        opened = int(opportunity.action_open_monotonic_raw_ns)

        def mark(name: str) -> None:
            stamps[name] = (time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                            - opened)

        self.stage_timing = {"action_open_monotonic_raw_ns": opened,
                             "stages_ns_after_action_open": stamps}
        mark("processor_entry")
        extra: dict[str, Any] = {}
        if self.gate.must_hold(previous):
            active = self.gate.active
            assert active is not None
            result = self._transmit(opportunity, active.profile, mark,
                                    decision_seq=active.decision_seq,
                                    reward_requested=False)
            self.gate.held()
            kind = "POLICY_HOLD"
        else:
            try:
                features = self._features(opportunity, previous)
            except R4C.ExternalFallbackRequired as exc:
                mark("state_features")
                result = self._transmit(opportunity, self.fallback_profile,
                                        mark, decision_seq=FALLBACK_SEQ,
                                        reward_requested=False)
                kind = "FALLBACK"
                extra["fallback_reasons"] = str(exc).split("; ")
            else:
                mark("state_features")
                mode_id, q_e4 = BASE._actor_action(
                    self.actor.loaded.module, features)
                mark("actor")
                profile = self.contract.resolve_q_e4(mode_id, q_e4)
                decision_seq = self.gate.next_decision_seq
                result = self._transmit(opportunity, profile, mark,
                                        decision_seq=decision_seq,
                                        reward_requested=True)
                self.gate.opened(ActiveDecisionV1(
                    result.identity, profile, decision_seq))
                kind = "POLICY_DECISION"
                extra["features"] = list(features)
                extra["observation"] = observation_from_features(
                    features, self.scaling)
        _require(result.decision_frame is (kind == "POLICY_DECISION"),
                 "frame kind and decision flag disagree")
        self.counters[kind] += 1
        self._record(kind, opportunity, previous, result, extra)
        return result

    def _record(self, kind: str, opportunity: Any,
                previous: Optional[ACK.OperationalOutcomeV1],
                result: UE.BTransmissionV1, extra: Mapping[str, Any]) -> None:
        identity = result.identity
        row = {
            "schema": TIMING_SCHEMA, "frame_index": opportunity.sequence,
            "kind": kind, "frame_id": identity.frame_id,
            "tensor_seq": identity.tensor_seq,
            "decision_seq": identity.decision_seq,
            "mode_id": identity.mode_id, "q_e4": identity.q_e4,
            "keep_count": identity.keep_count,
            "profile_id": identity.profile_id,
            "payload_bytes": result.payload_bytes,
            "datagrams": self.stage_timing.get("datagrams"),
            "capture_timestamp_ns": identity.capture_timestamp_ns,
            "action_open_monotonic_raw_ns":
                result.action_open_monotonic_raw_ns,
            "stages_ns_after_action_open":
                dict(self.stage_timing["stages_ns_after_action_open"]),
            "identity_sha256": identity.exact_sha256(),
            "previous_identity_sha256": (None if previous is None else
                                         previous.identity.exact_sha256()),
            "previous_terminal": (None if previous is None else
                                  previous.terminal.value),
            **extra,
        }
        line = (json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                ).encode("ascii")
        with self.timing_path.open("xb" if self._rows == 0 else "ab") as handle:
            handle.write(line)
            handle.flush()
        self._rows += 1


def build_split_host_300_pipeline(
        *, request: UE.BUEProcessRequestV1, evidence_root: Path,
        controller_lineage_sha256: str, dependencies: Any, cell_id: str,
        route_kwargs: Mapping[str, Any], raw_spool_root: Path,
        timing_path: Path, postrun_materializer: Any,
        route_driver: Any = None) -> V4.BRouteBridgeV4:
    _require(dependencies.variant is request.variant,
             "dependency/request variant mismatch")
    actor = BASE.load_final_actor_v1(variant=request.variant,
                                     evidence_root=Path(evidence_root))
    processor = HoldGatedOpportunityProcessorV1(
        timing_path=timing_path, request=request, actor=actor,
        controller_lineage_sha256=controller_lineage_sha256,
        telemetry=dependencies.telemetry,
        dynamic_contract=dependencies.dynamic_contract,
        continuous_ue=dependencies.continuous_ue,
        sender=dependencies.sender, remote=dependencies.remote,
        input_builder=dependencies.input_builder, cell_id=cell_id,
        chunk_bytes=dependencies.chunk_bytes,
        snr_provider=dependencies.snr_reader)
    return V4.BRouteBridgeV4(
        variant=request.variant,
        feature_schema_sha256=request.feature_schema_sha256,
        actor_boundary_sha256=request.actor_boundary_sha256,
        processor=processor,
        route_driver=route_driver or V4.pinned_route_driver(route_kwargs),
        raw_spool_root=Path(raw_spool_root),
        postrun_materializer=postrun_materializer,
        transmitted_budget=TRANSMITTED_BUDGET)


# ---------------------------------------------------------------------------
# ACK receiver: every datagram is stamped at arrival, not when consumed
# ---------------------------------------------------------------------------
class ThreadedOperationalAckReceiverV1:
    """Drain the UE ACK socket continuously on CLOCK_MONOTONIC_RAW.

    The sequential loop reads ACKs only while a ticket is open.  A late ACK
    that arrives while a held frame is being prepared must keep its true
    arrival stamp; this receiver stamps in a dedicated reader thread and the
    loop consumes ``(packet, receipt)`` pairs in arrival order.
    """

    def __init__(self, host: str, port: int) -> None:
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind((host, port))
        self._socket.settimeout(0.05)
        self._queue: "queue.Queue[tuple[bytes, int]]" = queue.Queue()
        self._stop = threading.Event()
        self.received = 0
        self.error: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="b300-ack-receiver")
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    packet, _peer = self._socket.recvfrom(1 << 20)
                except socket.timeout:
                    continue
                receipt = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                self.received += 1
                self._queue.put((packet, receipt))
        except BaseException as exc:  # surfaced on the next receive
            if not self._stop.is_set():
                self.error = exc

    def receive_until(self, identity: ACK.FrameActionIdentityV1,
                      deadline_monotonic_raw_ns: int
                      ) -> Optional[tuple[bytes, int]]:
        if self.error is not None:
            raise SplitHost300Error(f"ACK receiver failed: {self.error}")
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            pass
        remaining = (deadline_monotonic_raw_ns
                     - time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW))
        if remaining < 0:
            return None
        try:
            return self._queue.get(timeout=remaining / 1_000_000_000)
        except queue.Empty:
            return None

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._socket.close()
        _require(not self._thread.is_alive(), "ACK receiver did not stop")


# ---------------------------------------------------------------------------
# Live result
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Split300LiveResultV1:
    run_id: str
    variant: str
    transmitted_frames: int
    policy_decisions: int
    held_frames: int
    fallback_frames: int
    operational_successes: int
    operational_timeouts: int
    late_orphan_acks: int
    unknown_orphan_acks: int
    live_qperc_computed: bool
    gt_used_for_ack_or_state: bool
    ue_result_sha256: str

    def validate(self, config: Split300ConfigV1) -> None:
        _require(self.run_id == config.run_id
                 and self.variant == config.variant, "run/variant drift")
        _require(self.transmitted_frames == TRANSMITTED_BUDGET,
                 "transmitted frames differ from the exact budget")
        _require(self.policy_decisions + self.held_frames
                 + self.fallback_frames == self.transmitted_frames,
                 "frame kinds do not reconcile with transmitted frames")
        _require(self.operational_successes + self.operational_timeouts
                 == self.policy_decisions,
                 "operational terminals do not reconcile with decisions")
        _require(self.live_qperc_computed is False
                 and self.gt_used_for_ack_or_state is False,
                 "live GT/Q_perc entered the operational path")
        O._sha(self.ue_result_sha256, "ue_result_sha256")


# ---------------------------------------------------------------------------
# Production ops: the one-frame factory with the 300-frame hooks
# ---------------------------------------------------------------------------
class RealSplitHost300OpsV1(FACTORY.RealProductionOpsV1):
    def __init__(self, settings: LIFE.ProductionLifecycleSettingsV1) -> None:
        super().__init__(settings)
        self.live_result: Optional[Split300LiveResultV1] = None
        self.postrun: Optional[dict[str, Any]] = None
        self.result_document: Optional[dict[str, Any]] = None

    def _ue_request(self, config: Any, actor: F.LoadedFinalActorV2
                    ) -> UE.BUEProcessRequestV1:
        return build_ue_request_300(config, actor)

    def _edge_plan(self, config: Any, actor: F.LoadedFinalActorV2
                   ) -> FACTORY.RemoteEdgePlanV1:
        return FACTORY.build_remote_edge_plan(
            config, actor, request_builder=build_edge_request_300)

    def _edge_ready_scope(self) -> tuple[str, str, int]:
        return ER.PURPOSE_300, ER.CLAIM_SCOPE_300, ER.TRANSMITTED_BUDGET_300

    def _maximum_loop_sim_s(self, config: Any) -> float:
        return float(config.maximum_loop_sim_s)

    def _build_pipeline(self, state: FACTORY.StartedOneFrameV1, *,
                        route_kwargs: Mapping[str, Any],
                        paths: FACTORY.AttemptPathsV1) -> Any:
        from . import postrun_gt_materializer_v1 as GTM
        return build_split_host_300_pipeline(
            request=state.ue_request,
            evidence_root=state.config.actor_evidence_root,
            controller_lineage_sha256=state.controller_lineage_sha256,
            dependencies=state.dependencies, cell_id=state.config.cell_id,
            route_kwargs=route_kwargs,
            raw_spool_root=paths.get("raw_gt_spool"),
            timing_path=paths.get("frame_stage_timing"),
            postrun_materializer=GTM.PostRouteGroundTruthMaterializerV1())

    def _install_keepalive_policy(self, state: FACTORY.StartedOneFrameV1
                                  ) -> None:
        # The keepalive is never stopped by a decision; it runs for the whole
        # run and is stopped by dependencies.close() at teardown.
        _require(getattr(state.dependencies, "uplink_keepalive", None)
                 is not None, "uplink keepalive is absent")

    def start(self, config: Any, actor: F.LoadedFinalActorV2
              ) -> FACTORY.StartedOneFrameV1:
        state = super().start(config, actor)
        for owned in FACTORY.attempt_paths(config).phase("postrun"):
            if owned.exists():
                self.stop(state)
                raise SplitHost300Error(
                    f"post-run path exists after startup: {owned}")
        return state

    def execute(self, state: FACTORY.StartedOneFrameV1
                ) -> Split300LiveResultV1:
        _require(state.pipeline is not None and state.dependencies is not None,
                 "split-host lifecycle was not completely started")
        receiver = ThreadedOperationalAckReceiverV1(O.UE_TUNNEL_IP, O.ACK_PORT)
        # The live loop closes the receiver and the bridge (route join and
        # raw-spool seal) before it returns.  It never touches GT or Q_perc.
        result = UE.execute_300(state.ue_request, state.pipeline, receiver)
        report = json.loads((state.ue_request.output_root
                             / UE.REPORT_NAME).read_text(encoding="ascii"))
        snapshot = ACK.OperationalEvidenceStoreV1.open_existing(
            state.ue_request.evidence_root / "operational_evidence"
        ).verify_all(require_all_resolved=True)
        counters = state.pipeline.processor.counters
        live = Split300LiveResultV1(
            run_id=state.config.run_id, variant=state.config.variant,
            transmitted_frames=int(result["transmitted_frames"]),
            policy_decisions=int(report["policy_decisions"]),
            held_frames=int(counters["POLICY_HOLD"]),
            fallback_frames=int(counters["FALLBACK"]),
            operational_successes=int(report["operational_successes"]),
            operational_timeouts=int(report["operational_timeouts"]),
            late_orphan_acks=len(snapshot.late_orphans),
            unknown_orphan_acks=len(snapshot.unknown_orphans),
            live_qperc_computed=bool(report["live_qperc_computed"]),
            gt_used_for_ack_or_state=bool(report["gt_used_for_ack_or_state"]),
            ue_result_sha256=str(result["result_sha256"]))
        _require(live.policy_decisions == counters["POLICY_DECISION"]
                 == len(snapshot.outcomes),
                 "processor/UE decision counts differ")
        live.validate(state.config)
        self.live_result = live
        return live

    # -- post-run: after RAN/map/edge teardown, before the final CARLA stop --
    def _after_services_stopped(self, state: FACTORY.StartedOneFrameV1,
                                errors: list[str]) -> None:
        if self.live_result is None:
            return
        paths = FACTORY.attempt_paths(state.config)
        try:
            self.postrun = run_postrun_quality(
                state, paths, download=self._download_predictions)
        except BaseException as exc:  # recorded, never rewrites the live run
            self.postrun = {"status": "QUALITY_ANALYSIS_INCOMPLETE",
                            "error": f"{type(exc).__name__}: {exc}"}
        document = {
            "schema": RESULT_SCHEMA, "purpose": PURPOSE,
            "policy_performance_claim": False,
            "config_binding_sha256": state.config.binding_sha256(),
            "operational_status": "LIVE_300_COMPLETE",
            "live": dataclasses.asdict(self.live_result),
            "postrun": self.postrun,
            "teardown_errors_before_carla_stop": list(errors),
        }
        try:
            _write_create_only(paths.get("split_host_300_result"), document)
        except BaseException as exc:
            errors.append(f"result write: {type(exc).__name__}: {exc}")
        self.result_document = document

    def _download_predictions(self, state: FACTORY.StartedOneFrameV1,
                              local: Path) -> Path:
        assert state.edge_plan is not None
        runtime = state.edge_plan.runtime_root
        local.mkdir(exist_ok=False)
        command = self._remote_shell(
            ("tar", "-C", str(runtime), "-cf", "-", "prediction"))
        ssh = subprocess.Popen(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
             O.REMOTE_SSH, command], stdout=subprocess.PIPE)
        try:
            extract = subprocess.run(
                ["tar", "-C", str(local), "-xf", "-", "--no-same-owner"],
                stdin=ssh.stdout, timeout=600, check=False)
        finally:
            if ssh.stdout is not None:
                ssh.stdout.close()
            ssh_code = ssh.wait(timeout=600)
        _require(ssh_code == 0 and extract.returncode == 0,
                 "remote prediction download failed")
        return local / "prediction"


def _copy_decision_ground_truth(source: Path, target: Path,
                                decision_ids: set[str]) -> dict[str, Any]:
    """Byte-exact decision-only view of the post-route GT store."""
    from . import postrun_artifact_v1 as ART
    records = ART.GroundTruthEvidenceStoreV1.open_existing(source).verify_all()
    store = ART.GroundTruthEvidenceStoreV1.create(target)
    present = set()
    for record in records:
        if record.identity_sha256 not in decision_ids:
            continue
        shutil.copyfile(source / record.artifact_relative_path,
                        store.root / record.artifact_relative_path)
        shutil.copyfile(source / "records" / f"{record.identity_sha256}.json",
                        store.records / f"{record.identity_sha256}.json")
        present.add(record.identity_sha256)
    store.verify_all()
    return {"all_frame_gt_records": len(records),
            "decision_gt_records": len(present),
            "decisions_missing_gt": sorted(decision_ids - present)}


def run_postrun_quality(state: FACTORY.StartedOneFrameV1,
                        paths: FACTORY.AttemptPathsV1, *,
                        download: Any) -> dict[str, Any]:
    """Post-run evidence; each step's failure is recorded, never raised."""
    from . import operational_trace_v1 as TRACE

    out = paths.get("postrun_quality")
    out.mkdir(exist_ok=False)
    status: dict[str, Any] = {"schema": POSTRUN_SCHEMA,
                              "phase": "AFTER_LIVE_AND_SERVICE_TEARDOWN"}
    evidence_root = state.ue_request.evidence_root
    gt_root = evidence_root / "carla_gt_postrun"
    try:
        records = state.pipeline.materialize_postroute(gt_root)
        status["ground_truth"] = {"status": "MATERIALIZED",
                                  "records": int(records)}
    except BaseException as exc:
        status["ground_truth"] = {"status": "FAILED",
                                  "error": f"{type(exc).__name__}: {exc}"}
    prediction_root: Optional[Path] = None
    try:
        prediction_root = download(state, paths.get("remote_prediction"))
        from . import branch_evidence_v1 as BE
        predictions = BE.PredictionEvidenceStoreV1.open_existing(
            prediction_root).verify_all()
        status["predictions"] = {"status": "DOWNLOADED_VERIFIED",
                                 "records": len(predictions)}
    except BaseException as exc:
        prediction_root = None
        status["predictions"] = {"status": "FAILED",
                                 "error": f"{type(exc).__name__}: {exc}"}
    trace_root = evidence_root / "operational_trace"
    decision_ids = {record.identity_sha256 for record in
                    TRACE.OperationalTraceStoreV1.open_existing(
                        trace_root).verify_all()}
    status["decisions"] = len(decision_ids)
    if (status["ground_truth"]["status"] != "MATERIALIZED"
            or prediction_root is None):
        status["status"] = "QUALITY_ANALYSIS_INCOMPLETE"
    else:
        try:
            status["ground_truth_decisions"] = _copy_decision_ground_truth(
                gt_root, out / "ground_truth_decisions", decision_ids)
            from .postrun_evaluator_v1 import PostRunEvaluatorV1
            from .postrun_operational_population_v1 import (
                evaluate_from_operational_trace,
            )
            from .registered_quality_v1 import load_registered_quality_spec
            evaluator = PostRunEvaluatorV1(
                reward_spec=load_registered_quality_spec(
                    state.config.local_repository))
            result = evaluate_from_operational_trace(
                evaluator, operational_trace_root=trace_root,
                prediction_root=prediction_root,
                ground_truth_root=out / "ground_truth_decisions",
                output_root=out / "evaluation")
            status["evaluation"] = {
                "status": "COMPLETE", "frame_rows": result.frame_count,
                "quality_defined": result.quality_defined_count,
                "summary": str(result.summary_path)}
            status["status"] = "QUALITY_ANALYSIS_COMPLETE"
        except BaseException as exc:
            status["evaluation"] = {"status": "FAILED",
                                    "error": f"{type(exc).__name__}: {exc}"}
            status["status"] = "QUALITY_ANALYSIS_INCOMPLETE"
    _write_create_only(out / "POSTRUN_STATUS.json", status)
    return status


# ---------------------------------------------------------------------------
# Lifecycle and gate
# ---------------------------------------------------------------------------
class ProductionSplitHost300LifecycleV1:
    def __init__(self, settings: LIFE.ProductionLifecycleSettingsV1, *,
                 ops: Optional[RealSplitHost300OpsV1] = None) -> None:
        self.settings = settings
        self.ops = ops or RealSplitHost300OpsV1(settings)
        self.state: Optional[FACTORY.StartedOneFrameV1] = None

    def preflight(self, config: Split300ConfigV1,
                  actor: F.LoadedFinalActorV2) -> None:
        self.ops.preflight(config, actor)

    def start(self, config: Split300ConfigV1,
              actor: F.LoadedFinalActorV2) -> None:
        _require(self.state is None, "split-host lifecycle already started")
        self.state = self.ops.start(config, actor)

    def execute(self, config: Split300ConfigV1,
                actor: F.LoadedFinalActorV2) -> Split300LiveResultV1:
        _require(self.state is not None, "split-host lifecycle is not started")
        return self.ops.execute(self.state)

    def stop(self, config: Split300ConfigV1) -> None:
        state, self.state = self.state, None
        if state is not None:
            self.ops.stop(state)


def preflight(config: Split300ConfigV1,
              lifecycle: Any) -> F.LoadedFinalActorV2:
    O.require_create_only_targets(config)
    actor = O._load_selected_actor(config)
    lifecycle.preflight(config, actor)
    return actor


def run(config: Split300ConfigV1, lifecycle: Any) -> dict[str, Any]:
    actor = preflight(config, lifecycle)
    started = False
    primary: Optional[BaseException] = None
    try:
        lifecycle.start(config, actor)
        started = True
        live = lifecycle.execute(config, actor)
        _require(type(live) is Split300LiveResultV1,
                 "lifecycle returned a foreign live result")
        live.validate(config)
    except BaseException as exc:
        primary = exc
        raise
    finally:
        if started:
            try:
                lifecycle.stop(config)
            except BaseException:
                if primary is None:
                    raise
    document = getattr(lifecycle.ops, "result_document", None)
    _require(type(document) is dict, "durable split-host result is absent")
    return document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--build-config", type=Path, metavar="OUTPUT")
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--execute")
    parser.add_argument("--run-id")
    parser.add_argument("--cell-id")
    parser.add_argument("--variant",
                        choices=(F.RUN4B_VARIANT, F.RUN5B_VARIANT))
    for option in (
        "actor-manifest-path", "actor-weights-path", "actor-evidence-root",
        "local-repository", "remote-repository", "local-attempt-root",
        "remote-attempt-root", "edge-campaign-config", "route-config",
    ):
        parser.add_argument("--" + option, type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.build_config is not None:
        config = build_config(
            run_id=args.run_id, cell_id=args.cell_id, variant=args.variant,
            actor_manifest_path=args.actor_manifest_path,
            actor_weights_path=args.actor_weights_path,
            actor_evidence_root=args.actor_evidence_root,
            local_repository=args.local_repository,
            remote_repository=args.remote_repository,
            local_attempt_root=args.local_attempt_root,
            remote_attempt_root=args.remote_attempt_root,
            edge_campaign_config=args.edge_campaign_config,
            route_config=args.route_config)
        O.require_create_only_targets(config)
        BUILDER._write_create_only(args.build_config, seal(config))
        print(json.dumps({"status": "SPLIT_HOST_300_CONFIG_CREATED",
                          "path": str(args.build_config),
                          "binding_sha256": config.binding_sha256(),
                          "services_launched": False}, sort_keys=True))
        return 0
    _require(args.config is not None, "--config is required")
    config = load_config(args.config)
    lifecycle = ProductionSplitHost300LifecycleV1(config.lifecycle_settings())
    if args.preflight:
        actor = preflight(config, lifecycle)
        print(json.dumps({
            "status": "PREFLIGHT_PASS", "schema": RESULT_SCHEMA,
            "config_binding_sha256": config.binding_sha256(),
            "variant": config.variant,
            "actor_state_dict_sha256": actor.identity.actor_state_dict_sha256,
            "purpose": PURPOSE, "services_launched": False}, sort_keys=True))
        return 0
    _require(args.execute == EXECUTE_TOKEN, "production execution token differs")
    document = run(config, lifecycle)
    print(json.dumps({"status": "SPLIT_HOST_300_LIVE_COMPLETE",
                      "live": document["live"],
                      "postrun_status": (document["postrun"] or {}).get("status")},
                     sort_keys=True))
    return 0


__all__ = [
    "CONFIG_SCHEMA", "PURPOSE", "EXECUTE_TOKEN", "K_MIN", "FALLBACK",
    "Split300ConfigV1", "build_config", "seal", "load_config",
    "build_ue_request_300", "build_edge_request_300", "ActiveDecisionV1",
    "OperationalHoldGateV1", "HoldGatedOpportunityProcessorV1",
    "build_split_host_300_pipeline", "ThreadedOperationalAckReceiverV1",
    "Split300LiveResultV1", "RealSplitHost300OpsV1", "run_postrun_quality",
    "ProductionSplitHost300LifecycleV1", "preflight", "run", "main",
]
