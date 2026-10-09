"""Run-4B UE seams over the unchanged single-host Phase-6 harness.

Only four things differ from Phase 6, each installed as a module seam inside
the dedicated child process (the Phase-6 package files are not edited):

* the frozen actor is the exact Run-4B-Joint seed-43/update-10,000 export;
* the live state is the exact Run-4B 20-feature vector (current UE MCS and
  RLC backlog, previous action, previous operational success/latency; no
  SNR, no previous Q_perc or reward).  The Run-4 freshness/causality guard
  and radio-support refusal are applied unchanged to the four observations;
* a ticket resolves on the GT-free operational ACK (SUCCESS iff the ACK is
  received at most 170 ms after action open) or by the unchanged timeout;
  live Q_perc feedback is refused;
* the UE control loop decodes that ACK.

The hold controller (k_min = 2), fallback, decision engine, frame accounting,
cycle budget, edge identity checks and terminals are the Phase-6 code paths.
Importing this module performs no I/O and does not import torch.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import (
    live_state_v2 as LS,
    phase6_decision_engine_v2 as E,
    reward_hold_controller_v2 as R,
    ue_telemetry_provider_v2 as T,
)
from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as contract
from rl_agent.splitfusion_hybrid_sac_run4b_v1 import contract as C4B

from . import ack_wire_v1 as AW

ACTOR_EXPORT_DIR = Path(
    "/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/"
    "PythonAPI/neu_collab/abiodun/rl_agent/experiments/"
    "splitfusion_hybrid_sac_run4b_v1/20261001_7598e4d_joint_channel_comparator_v1/"
    "campaign/seed_43/final_actor")
ACTOR_STATE_DICT_SHA256 = (
    "643cb7adee66bc7e82665bf5cdbbcead39d6690c7af4fe988bd48be730753c3c")
ACTOR_TREE_SHA256 = (
    "998a519f2f2039b1779c381ce247fbaeeb42f0edd138def75efd66d1932c6fc9")
ACTOR_EXPORT_SCHEMA = "splitfusion.run4b_joint_channel.actor_export.v1"
ACTOR_SEED, ACTOR_UPDATE = 43, 10000
# Run-4B training scaling (environment.SharedSourcesV1.scaling(), FIT scene
# catalog); numerically identical to the Phase-6 live TRAINING_SCALING.
SCALING = C4B.ScalingV1(camera_si_center=117.08553307797729,
                        camera_si_scale=13.336440703846854,
                        backlog_log1p_scale=math.log1p(50_000_000))
SCALING_SHA256 = (
    "b8a673047420408bbf6abf6d62226a336b8a966ef129a38bcd8699fd77ec04d7")
if SCALING.sha256 != SCALING_SHA256:
    raise RuntimeError("Run-4B training scaling reconstruction drifted")
if (SCALING.camera_si_center, SCALING.camera_si_scale,
        SCALING.backlog_log1p_scale) != (
        LS.TRAINING_SCALING.camera_si_center, LS.TRAINING_SCALING.camera_si_scale,
        LS.TRAINING_SCALING.backlog_log1p_scale):
    raise RuntimeError("Run-4B and Phase-6 live scaling differ")
CONTROLLER = "OperationalHoldControllerV1"
ENGINE = "Run4BDecisionEngineV1"
SOURCE = "splitfusion_run4b_single_host_v1:operational_hold_controller"


class Run4BUeError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Run4BUeError(message)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Frozen Run-4B actor
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Run4BActorDecisionV1:
    mode_id: int
    q_e4: int
    q_support_lower_e4: int
    q_support_upper_e4: int
    state_sha256: str
    actor_tree_sha256: str


class Run4BFeatureVectorV1:
    """The exact 20-value Run-4B state (FEATURE_ORDER), immutable."""

    def __init__(self, values: Sequence[float]) -> None:
        values = tuple(float(v) for v in values)
        _require(len(values) == C4B.FEATURE_COUNT, "feature width is not 20")
        _require(all(math.isfinite(v) for v in values), "non-finite feature")
        self._values = values

    def as_tuple(self) -> tuple[float, ...]:
        return self._values

    def as_dict(self) -> dict[str, float]:
        return dict(zip(C4B.FEATURE_ORDER, self._values))


class Run4BFrozenActorV1:
    """Read-only CPU actor; batch one; the registered deterministic rule."""

    def __init__(self, actor: Any, manifest: Mapping[str, Any]) -> None:
        import torch

        actor.eval()
        actor.requires_grad_(False)
        for name, value in list(actor.named_parameters()) + list(actor.named_buffers()):
            _require(value.device.type == "cpu", f"{name} is not on CPU")
            if value.is_floating_point():
                _require(value.dtype is torch.float32, f"{name} is not float32")
        self._actor = actor
        self.manifest = dict(manifest)
        lower, upper = actor.active_q_e4_bounds()
        self._lower = tuple(int(item) for item in lower)
        self._upper = tuple(int(item) for item in upper)
        self.boundary_sha256 = str(manifest["actor_tree_sha256"])

    @property
    def module(self) -> Any:
        return self._actor

    def act(self, features: Run4BFeatureVectorV1) -> Run4BActorDecisionV1:
        import torch

        _require(type(features) is Run4BFeatureVectorV1,
                 "features must be an exact Run4BFeatureVectorV1")
        _require(not self._actor.training, "actor left eval mode")
        values = features.as_tuple()
        state = torch.tensor((values,), dtype=torch.float32)
        rng = torch.random.get_rng_state().clone()
        cuda_before = torch.cuda.is_initialized()
        with torch.inference_mode():
            execution = self._actor.deterministic_execution(state)
        _require(torch.equal(rng, torch.random.get_rng_state()),
                 "deterministic actor changed the global RNG")
        _require(torch.cuda.is_initialized() == cuda_before,
                 "CPU actor initialized CUDA")
        mode_id = int(execution.mode_index[0])
        q_e4 = int(execution.q_e4[0])
        _require(0 <= mode_id < C4B.MODE_COUNT, "mode out of range")
        _require(self._lower[mode_id] <= q_e4 <= self._upper[mode_id],
                 "q_e4 escaped the selected mode's registered support")
        return Run4BActorDecisionV1(
            mode_id=mode_id, q_e4=q_e4,
            q_support_lower_e4=self._lower[mode_id],
            q_support_upper_e4=self._upper[mode_id],
            state_sha256=hashlib.sha256(json.dumps(values).encode()).hexdigest(),
            actor_tree_sha256=self.boundary_sha256)


def load_run4b_actor(export_dir: Path = ACTOR_EXPORT_DIR) -> Run4BFrozenActorV1:
    """Verify the exact export (file, tree, schema, seed/update) and load it."""
    import torch

    from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as R4O
    from rl_agent.splitfusion_hybrid_sac_run4b_v1 import models as M4B
    from rl_agent.splitfusion_hybrid_sac_v1.hybrid_sac_models import build_actor

    export_dir = Path(export_dir)
    weights = export_dir / "actor_state_dict.pt"
    manifest_path = export_dir / "ACTOR_EXPORT.json"
    _require(weights.is_file() and not weights.is_symlink()
             and manifest_path.is_file(), "Run-4B actor export is absent")
    _require(_sha_file(weights) == ACTOR_STATE_DICT_SHA256,
             "Run-4B actor_state_dict digest differs from the pinned actor")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema": ACTOR_EXPORT_SCHEMA, "seed": ACTOR_SEED, "update": ACTOR_UPDATE,
        "actor_state_dict_sha256": ACTOR_STATE_DICT_SHA256,
        "actor_tree_sha256": ACTOR_TREE_SHA256,
        "feature_order": list(C4B.FEATURE_ORDER),
        "feature_schema_sha256": C4B.FEATURE_SCHEMA_SHA256,
        "model_binding_sha256": M4B.MODEL_BINDING_SHA256,
        "preregistered_live_actor": True,
    }
    drift = {key: (manifest.get(key), value) for key, value in expected.items()
             if manifest.get(key) != value}
    _require(not drift, f"Run-4B actor export drift: {sorted(drift)}")
    actor = build_actor(M4B.model_config(), seed=0)
    actor.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"),
                          strict=True)
    _require(R4O._tree_sha256(actor.state_dict()) == ACTOR_TREE_SHA256,
             "Run-4B actor tensor tree differs")
    return Run4BFrozenActorV1(actor, manifest)


# ---------------------------------------------------------------------------
# Operational prior and the 20-feature live state
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LivePriorV1:
    """The previous decision's GT-free operational outcome, causally stamped."""

    identity: contract.DecisionIdentityV1
    prior: C4B.OperationalPriorV1
    available_timestamp_ns: int
    clock_domain: str


@dataclass(frozen=True)
class GuardedRun4BStateV1:
    boundary: contract.DecisionBoundaryV1
    observation_guard: contract.GuardedPolicyStateV2
    observation: C4B.ObservationV1
    prior: C4B.OperationalPriorV1


def build_run4b_live_state(
    *, identity: contract.DecisionIdentityV1, boundary: contract.DecisionBoundaryV1,
    camera_si: contract.ScalarObservationV1, radar_p40: contract.ScalarObservationV1,
    prior_ul_mcs: contract.PriorUlGrantObservationV1,
    pre_action_rlc_backlog: contract.ScalarObservationV1,
    previous: Optional[LivePriorV1],
) -> tuple[GuardedRun4BStateV1, Run4BFeatureVectorV1]:
    """Drop-in for ``live_state_v2.build_live_state`` with the Run-4B state.

    The unchanged Run-4 guard is applied to the four observations; it is run
    on the same session/UE at sequence 0 because its ``PolicyStateV2`` can
    only carry a Q_perc-bearing previous outcome.  The operational prior is
    checked here with the same two causal rules the guard applies to a
    previous outcome (same clock domain; available by state commit) plus the
    exact immediately-preceding identity rule.
    """
    _require(identity == boundary.identity, "state/boundary identity differ")
    shim = contract.DecisionIdentityV1(identity.session_uuid, identity.ue_id, 0)
    shim_boundary = contract.DecisionBoundaryV1(
        identity=shim, state_commit_timestamp_ns=boundary.state_commit_timestamp_ns,
        action_open_timestamp_ns=boundary.action_open_timestamp_ns,
        clock_domain=boundary.clock_domain)
    guarded = contract.guard_state_for_action(contract.PolicyStateV2(
        identity=shim, camera_si=camera_si, radar_p40=radar_p40,
        prior_ul_mcs=prior_ul_mcs, pre_action_rlc_backlog=pre_action_rlc_backlog,
        previous=None), shim_boundary, LS.TRAINING_FRESHNESS)
    if identity.decision_seq == 0:
        _require(previous is None, "decision 0 cannot carry a previous outcome")
        prior = C4B.OperationalPriorV1.genesis()
    else:
        _require(type(previous) is LivePriorV1,
                 "every later decision must carry the preceding outcome")
        _require(previous.identity.session_uuid == identity.session_uuid
                 and previous.identity.ue_id == identity.ue_id
                 and previous.identity.decision_seq + 1 == identity.decision_seq,
                 "previous outcome is not the immediately preceding decision")
        if previous.clock_domain != boundary.clock_domain:
            raise contract.ExternalFallbackRequired(
                "previous outcome and state boundary use different clocks")
        if previous.available_timestamp_ns > boundary.state_commit_timestamp_ns:
            raise contract.ExternalFallbackRequired(
                "previous outcome was unavailable when state was committed")
        prior = previous.prior
    mcs = prior_ul_mcs.observation.value
    backlog = pre_action_rlc_backlog.value
    low, high = LS.RADIO_SUPPORT["prior_ul_mcs"]
    if not low <= mcs <= high:
        raise contract.ExternalFallbackRequired(
            f"prior UL MCS {mcs} outside registered training support [{low}, {high}]")
    low, high = LS.RADIO_SUPPORT["backlog_bytes"]
    if not low <= backlog <= high:
        raise contract.ExternalFallbackRequired(
            f"raw backlog {backlog} B outside registered training support "
            f"[{low}, {high}]; clipping must not hide it")
    observation = C4B.ObservationV1(
        camera_si=float(camera_si.value), radar_p40=float(radar_p40.value),
        prior_ul_mcs=int(mcs), pre_action_rlc_backlog_bytes=int(backlog))
    features = Run4BFeatureVectorV1(C4B.build_features(observation, prior, SCALING))
    return (GuardedRun4BStateV1(boundary=boundary, observation_guard=guarded,
                                observation=observation, prior=prior), features)


# ---------------------------------------------------------------------------
# GT-free resolution and the hold controller
# ---------------------------------------------------------------------------
TERMINALS = ("SUCCESS", "TIMEOUT", "SERVICE_FAILURE")


@dataclass(frozen=True)
class OperationalResolutionV1:
    identity: contract.DecisionIdentityV1
    action: Any
    terminal: str
    action_open_timestamp_ns: int
    resolution_timestamp_ns: int
    operational_latency_ns: Optional[int]
    clock_domain: str
    source: str = SOURCE
    learning_included: bool = True

    def __post_init__(self) -> None:
        _require(self.terminal in TERMINALS, "unknown operational terminal")
        if self.terminal == "SUCCESS":
            _require(type(self.operational_latency_ns) is int
                     and 0 < self.operational_latency_ns <= C4B.DEADLINE_NS,
                     "a success needs a latency in (0, 170 ms]")
        else:
            _require(self.operational_latency_ns is None,
                     "a non-success carries no successful latency")

    def to_canonical_dict(self) -> dict[str, Any]:
        return {"schema": "scenesense.run4b_single_host.operational_resolution.v1",
                "identity": self.identity.to_canonical_dict(),
                "action": self.action.to_canonical_dict(),
                "terminal": self.terminal, "timely": self.terminal == "SUCCESS",
                "action_open_timestamp_ns": self.action_open_timestamp_ns,
                "resolution_timestamp_ns": self.resolution_timestamp_ns,
                "operational_latency_ns": self.operational_latency_ns,
                "clock_domain": self.clock_domain, "source": self.source,
                "learning_included": self.learning_included,
                "live_qperc_computed": False}


class OperationalHoldControllerV1(R.RewardHoldControllerV2):
    """``RewardHoldControllerV2`` with GT-free operational resolution.

    Hold/k_min gating, identity matching, duplicate/late/orphan handling,
    registered edge terminals and the timeout boundary are inherited; only
    what a resolution *is* changes.  No Q_perc or reward exists live.
    """

    def _resolve(self, ticket: Any, *, kind: Any, resolution_ns: int,
                 q_perc: Optional[float], by: str) -> None:
        _require(q_perc is None, "a Q_perc value reached the live controller")
        mapping = {contract.RewardEventKind.TIMEOUT: "TIMEOUT",
                   contract.RewardEventKind.REGISTERED_SERVICE_FAILURE:
                       "SERVICE_FAILURE"}
        _require(kind in mapping, f"live resolution kind {kind} is not GT-free")
        self._resolve_operational(ticket, terminal=mapping[kind],
                                  resolution_ns=resolution_ns, by=by)

    def _resolve_operational(self, ticket: Any, *, terminal: str,
                             resolution_ns: int, by: str) -> None:
        latency = (int(resolution_ns - ticket.action_open_ns)
                   if terminal == "SUCCESS" else None)
        ticket.resolution = OperationalResolutionV1(
            identity=ticket.identity, action=ticket.action, terminal=terminal,
            action_open_timestamp_ns=int(ticket.action_open_ns),
            resolution_timestamp_ns=int(resolution_ns),
            operational_latency_ns=latency, clock_domain=self.clock_domain)
        ticket.resolved_by = by

    def previous_for_next_decision(self) -> Optional[LivePriorV1]:
        self._require_healthy()
        ticket = self._ticket
        if ticket is None:
            return None
        resolution = ticket.resolution
        _require(resolution is not None, "previous ticket is unresolved")
        return LivePriorV1(
            identity=ticket.identity,
            prior=C4B.OperationalPriorV1.from_outcome(
                mode_id=int(ticket.action.mode_id), q_e4=int(ticket.action.q_e4),
                timely=resolution.terminal == "SUCCESS",
                operational_latency_ns=resolution.operational_latency_ns),
            available_timestamp_ns=int(resolution.resolution_timestamp_ns),
            clock_domain=self.clock_domain)

    def on_feedback(self, fb: Any, *, receipt_ns: int) -> R.FeedbackClass:
        self._faulted = "live quality feedback is forbidden in Run-4B"
        raise R.ConflictingFeedbackError(self._faulted)

    def _ack_matches(self, ticket: Any, ack: AW.OperationalAckV1) -> bool:
        frame = ticket.reward_frame
        return (frame is not None and ack.reward_requested is True
                and (ack.frame_id, ack.tensor_seq, ack.capture_timestamp_ns)
                == (frame.frame_id, frame.tensor_seq, frame.capture_timestamp_ns)
                and (ack.mode_id, ack.q_e4, ack.anchor_action_id)
                == (ticket.action.mode_id, ticket.action.q_e4, ticket.action.action_id)
                and ack.execution_bundle_sha256 == ticket.bundle_sha256)

    def on_operational_ack(self, ack: AW.OperationalAckV1, *, payload: bytes,
                           receipt_ns: int) -> R.FeedbackClass:
        """Same identity discipline as ``on_feedback``; SUCCESS iff <= 170 ms."""
        self._require_healthy()
        self.poll(receipt_ns)
        record = {"receipt_ns": receipt_ns, "request_key": list(ack.request_key),
                  "sha256": hashlib.sha256(payload).hexdigest(),
                  "source": "OPERATIONAL_ACK"}
        key = ack.request_key
        ticket = self._ticket
        live_key = None if ticket is None else (
            self.session_uuid, self.lineage, ticket.identity.decision_seq,
            ticket.ticket_seq)
        target = ticket if key == live_key else self._closed.get(key)
        if target is None:
            record["class"] = R.FeedbackClass.UNKNOWN_ORPHAN.value
            self.ledger.feedback.append(record)
            return R.FeedbackClass.UNKNOWN_ORPHAN
        if target.accepted_bytes is not None:
            if payload == target.accepted_bytes:
                record["class"] = R.FeedbackClass.DUPLICATE_IGNORED.value
                self.ledger.feedback.append(record)
                return R.FeedbackClass.DUPLICATE_IGNORED
            self._faulted = f"conflicting operational ACK for {key}"
            raise R.ConflictingFeedbackError(self._faulted)
        if not self._ack_matches(target, ack):
            self._faulted = f"operational ACK identity disagrees with {key}"
            raise R.ConflictingFeedbackError(self._faulted)
        record["observed_latency_ns"] = int(receipt_ns - target.action_open_ns)
        if target.resolution is not None:   # timed out before this arrived
            record["class"] = R.FeedbackClass.LATE_ORPHAN.value
            self.ledger.feedback.append(record)
            return R.FeedbackClass.LATE_ORPHAN
        self._resolve_operational(target, terminal="SUCCESS",
                                  resolution_ns=receipt_ns, by="OPERATIONAL_ACK")
        target.accepted_bytes = payload
        record["class"] = R.FeedbackClass.ACCEPTED.value
        self.ledger.feedback.append(record)
        return R.FeedbackClass.ACCEPTED


_BASE_ENGINE = E.Run4DecisionEngineV2


class Run4BDecisionEngineV1(_BASE_ENGINE):
    """The Phase-6 engine whose sessions use the operational controller."""

    def _new_session(self) -> None:
        self.controllers.append(OperationalHoldControllerV1(
            session_uuid=self._session_factory(), ue_id=self.ue_id,
            controller_lineage_sha256=self.lineage, clock_domain=self.clock_domain))

    def on_feedback(self, fb: Any, *, receipt_raw_ns: int) -> R.FeedbackClass:
        raise R.ConflictingFeedbackError("live quality feedback is forbidden")

    def on_operational_ack(self, ack: AW.OperationalAckV1, *, payload: bytes,
                           receipt_raw_ns: int) -> R.FeedbackClass:
        controller = self.controller_for(ack.session_uuid)
        target = controller if controller is not None else self.controller
        return target.on_operational_ack(ack, payload=payload,
                                         receipt_ns=receipt_raw_ns)


def controller_lineage_sha256() -> str:
    """Execution/controller lineage; distinct from the actor boundary."""
    return hashlib.sha256(json.dumps({
        "actor_state_dict_sha256": ACTOR_STATE_DICT_SHA256,
        "actor_tree_sha256": ACTOR_TREE_SHA256,
        "controller": CONTROLLER, "engine": ENGINE,
        "fallback": "mode11_q9800_a71",
        "ack": AW.SCHEMA}, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------
# UE runtime: the Phase-6 runtime with the operational-ACK control branch
# ---------------------------------------------------------------------------
def build_run4b_runtime_class(base_builder: Any, bindings: Any):  # pragma: no cover - live
    from rl_agent.splitfusion_direct_edge_map_v1 import live_pilot_runtime_direct_v1 as DR
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import (
        phase6_ue_runtime_v2 as U,
        run4_live_wire_v2 as W,
        run4_map_protocol_v2 as MP,
    )

    Base = base_builder(bindings)

    class Run4BLiveRuntimeV1(Base):
        def _result_loop(self) -> None:
            import socket as _socket

            self._run4_ready.wait()
            while not self.stop_event.is_set():
                try:
                    datagram, address = self.receiver.recvfrom(65535)
                except _socket.timeout:
                    continue
                except OSError:
                    return
                receipt_raw = T.raw_now_ns()
                received_at = time.time()
                try:
                    if AW.is_ack(datagram):
                        ack = AW.decode(datagram)
                        with self._engine_lock:
                            outcome = self.engine.on_operational_ack(
                                ack, payload=datagram, receipt_raw_ns=receipt_raw)
                        self._feedback_rows.append({
                            "frame_id": ack.frame_id, "class": outcome.value,
                            "kind": "OPERATIONAL_ACK", "tensor_seq": ack.tensor_seq,
                            "decision_seq": ack.decision_seq,
                            "session_uuid": ack.session_uuid,
                            "receipt": W.raw(receipt_raw).to_dict(),
                            "receipt_raw_ns": receipt_raw,
                            "receipt_wall_s": received_at,
                            "tail_output_sha256": ack.tail_output_sha256,
                            "edge_tail_ready_wall_ns": ack.tail_ready_wall_ns,
                            "edge_ack_send_wall_ns": ack.ack_send_wall_ns,
                            "edge_reassembly_complete_wall_s":
                                ack.reassembly_complete_wall_s,
                            "edge_compute_start_wall_s": ack.compute_start_wall_s,
                            "source_address": f"{address[0]}:{address[1]}",
                            "bytes": len(datagram),
                            "sha256": hashlib.sha256(datagram).hexdigest()})
                        continue
                    _require(datagram[:4] != W.MAGIC_FB,
                             "live quality feedback (R4FB) reached the Run-4B UE")
                    message = DR.protocol.decode(datagram)
                    DR.protocol.assert_no_object_records(message)
                    schema = str(message.get("schema") or "")
                    _require(schema in (MP.RUN4_FEEDBACK_SCHEMA, MP.RUN4_TERMINAL_SCHEMA),
                             f"unknown control schema {schema!r}")
                    frame_id = int(message["frame_id"])
                    _require(self._run4_identity.get(frame_id)
                             == dict(message.get("run4_identity") or {}),
                             "control message Run-4 identity drift")
                    message["_feedback_bytes"] = len(datagram)
                    self.ledger.enqueue(message, received_at)
                    terminal = U.registered_terminal_from_message(message)
                    if terminal is not None:
                        with self._engine_lock:
                            klass = self.engine.on_registered_terminal(
                                terminal, receipt_raw_ns=receipt_raw)
                        self._terminal_rows.append({
                            "frame_id": terminal.frame_id, "class": klass.value,
                            "outcome": terminal.outcome,
                            "agent_credit": terminal.agent_credit,
                            "stage": terminal.stage,
                            "request_key": list(terminal.request_key),
                            "receipt": W.raw(receipt_raw).to_dict()})
                    with self.lock:
                        if (bool(message.get("terminal"))
                                and frame_id not in self._terminal_frames):
                            self._terminal_frames[frame_id] = str(
                                message.get("outcome") or "")
                            self._published_frames.add(frame_id)
                            self.completed += 1
                except R.ConflictingFeedbackError as exc:
                    with self.lock:
                        self.errors.append(f"conflicting operational ACK: {exc}")
                except Exception as exc:  # noqa: BLE001
                    with self.lock:
                        self.errors.append(f"{type(exc).__name__}: {exc}")
                    return

    Run4BLiveRuntimeV1.__name__ = "Run4BLiveRuntimeV1"
    return Run4BLiveRuntimeV1


_INSTALLED: dict[str, Any] = {}


def install_ue_seams() -> dict[str, Any]:  # child process only
    """Install the Run-4B actor/state/engine/runtime seams (idempotent)."""
    if _INSTALLED:
        return dict(_INSTALLED)
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import (
        frozen_actor_v2 as FA,
        phase6_ue_runtime_v2 as U,
    )

    base_builder = U.build_run4_runtime_class
    FA.load_registered_actor = lambda *args, **kwargs: load_run4b_actor()
    LS.build_live_state = build_run4b_live_state
    E.Run4DecisionEngineV2 = Run4BDecisionEngineV1
    U.build_run4_runtime_class = (
        lambda bindings: build_run4b_runtime_class(base_builder, bindings))
    _INSTALLED.update({"actor": "load_run4b_actor",
                       "state": "build_run4b_live_state",
                       "engine": ENGINE, "controller": CONTROLLER,
                       "runtime": "Run4BLiveRuntimeV1"})
    return dict(_INSTALLED)


__all__ = [
    "ACTOR_EXPORT_DIR", "ACTOR_STATE_DICT_SHA256", "ACTOR_TREE_SHA256",
    "SCALING", "Run4BFeatureVectorV1", "Run4BFrozenActorV1", "load_run4b_actor",
    "LivePriorV1", "build_run4b_live_state", "OperationalResolutionV1",
    "OperationalHoldControllerV1", "Run4BDecisionEngineV1",
    "controller_lineage_sha256", "build_run4b_runtime_class", "install_ue_seams",
]
