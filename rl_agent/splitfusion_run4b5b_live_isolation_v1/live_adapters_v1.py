"""Pure adapters for the isolated Run-4B/Run-5B live path.

The module deliberately owns no socket, CUDA object, model, or service.  It
binds a frozen actor to one of the two exact B-variant feature schemas and
provides the small edge-side ordering seam needed by a future process runner:

``usable tail output -> operational ACK -> independent map/evidence queues``.

The ACK callback is invoked synchronously before either branch can observe the
output.  Slow or failed branch callbacks cannot change the ACK.  Non-decision
hold/fallback frames may use :meth:`TailOutputDispatchV1.dispatch_map_only`;
they update the spatial map but cannot create a second ACK for an existing
policy ticket.  Prediction evidence is retained only for the decision frame
whose operational outcome is being measured.

Importing this module starts no thread and performs no I/O.
"""

from __future__ import annotations

import enum
import hashlib
import json
import math
import queue
import re
import threading
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

from . import branch_evidence_v1 as B
from . import feature_schema_authority_v1 as AUTH
from . import operational_ack_v1 as A


FEATURE_SCHEMA = "scenesense.splitfusion.run4b5b.policy_features.v1"
ACTOR_MANIFEST_SCHEMA = "scenesense.splitfusion.run4b5b.actor_manifest.v1"
DECISION_RULE = "BATCH1_CPU_DETERMINISTIC_ARGMAX__DECIMAL_HALF_UP_Q_E4"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_DETAIL_CODE_RE = re.compile(r"[A-Z0-9][A-Z0-9_.:-]{0,127}")


class LiveAdapterError(RuntimeError):
    """A B-path actor or edge adapter contract was violated."""


class ActorBindingError(LiveAdapterError):
    """A frozen actor or its exact feature schema does not match."""


class EdgeDispatchError(LiveAdapterError):
    """The immediate-ACK/fan-out ordering contract was violated."""


def _require(condition: bool, message: str,
             error: type[LiveAdapterError] = LiveAdapterError) -> None:
    if not condition:
        raise error(message)


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise LiveAdapterError("value is not canonically serializable") from exc


def _digest(value: Any, field: str,
            error: type[LiveAdapterError] = LiveAdapterError) -> str:
    _require(type(value) is str and bool(_SHA256_RE.fullmatch(value)),
             f"{field} is not a lowercase SHA-256", error)
    return value


class ActorVariant(str, enum.Enum):
    RUN4B = "RUN4B_MCS_BACKLOG_NO_QPERC"
    RUN5B = "RUN5B_MCS_BACKLOG_SNR_NO_QPERC"


RUN4B_FEATURE_ORDER: tuple[str, ...] = (
    "camera_si_scaled",
    "radar_p40",
    "prior_ul_mcs_normalized",
    "pre_action_rlc_backlog_log1p_scaled",
    *(f"prev_joint_mode_{index}_one_hot" for index in range(12)),
    "prev_q_normalized",
    "prev_operational_latency_normalized",
    "prev_present",
    "prev_operational_success",
)
RUN5B_FEATURE_ORDER: tuple[str, ...] = (
    *RUN4B_FEATURE_ORDER,
    "effective_external_ul_snr_proxy_scaled",
)


def expected_feature_order(variant: ActorVariant) -> tuple[str, ...]:
    _require(type(variant) is ActorVariant, "variant must be ActorVariant",
             ActorBindingError)
    return (RUN4B_FEATURE_ORDER if variant is ActorVariant.RUN4B
            else RUN5B_FEATURE_ORDER)


def feature_schema_sha256(variant: ActorVariant,
                          order: Sequence[str]) -> str:
    _require(type(variant) is ActorVariant, "variant must be ActorVariant",
             ActorBindingError)
    exact = tuple(order)
    _require(exact == expected_feature_order(variant),
             "feature order differs from the selected B variant",
             ActorBindingError)
    try:
        return AUTH.feature_schema_sha256(variant.value, exact)
    except AUTH.FeatureSchemaAuthorityError as exc:
        raise ActorBindingError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class BActorManifestV1:
    """Minimal live binding copied from a selected Run-4B/Run-5B actor."""

    variant: ActorVariant
    feature_order: tuple[str, ...]
    feature_count: int
    feature_schema_sha256: str
    actor_boundary_sha256: str
    weights_file_sha256: str
    selected_seed: int
    selected_update: int
    decision_rule: str = DECISION_RULE

    def __post_init__(self) -> None:
        _require(type(self.variant) is ActorVariant,
                 "manifest variant must be ActorVariant", ActorBindingError)
        _require(type(self.feature_order) is tuple
                 and all(type(item) is str and item for item in self.feature_order),
                 "manifest feature order must be an exact non-empty tuple",
                 ActorBindingError)
        expected = expected_feature_order(self.variant)
        _require(self.feature_order == expected,
                 "manifest feature order differs from the selected B variant",
                 ActorBindingError)
        _require(type(self.feature_count) is int
                 and self.feature_count == len(expected),
                 "manifest feature count differs from its exact order",
                 ActorBindingError)
        _digest(self.feature_schema_sha256, "feature_schema_sha256",
                ActorBindingError)
        _require(self.feature_schema_sha256
                 == feature_schema_sha256(self.variant, expected),
                 "manifest feature schema digest differs", ActorBindingError)
        _digest(self.actor_boundary_sha256, "actor_boundary_sha256",
                ActorBindingError)
        _digest(self.weights_file_sha256, "weights_file_sha256",
                ActorBindingError)
        _require(type(self.selected_seed) is int and self.selected_seed >= 0,
                 "selected_seed must be nonnegative", ActorBindingError)
        _require(type(self.selected_update) is int and self.selected_update > 0,
                 "selected_update must be positive", ActorBindingError)
        _require(self.decision_rule == DECISION_RULE,
                 "deployment decision rule drifted", ActorBindingError)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "BActorManifestV1":
        fields = {
            "schema", "variant", "feature_order", "feature_count",
            "feature_schema_sha256", "actor_boundary_sha256",
            "weights_file_sha256", "selected_seed", "selected_update",
            "decision_rule",
        }
        _require(isinstance(raw, Mapping) and set(raw) == fields,
                 "actor manifest fields are incomplete or foreign",
                 ActorBindingError)
        _require(raw["schema"] == ACTOR_MANIFEST_SCHEMA,
                 "actor manifest schema drift", ActorBindingError)
        try:
            variant = ActorVariant(raw["variant"])
        except (ValueError, TypeError) as exc:
            raise ActorBindingError("unknown actor variant") from exc
        return cls(
            variant=variant,
            feature_order=tuple(raw["feature_order"]),
            feature_count=raw["feature_count"],
            feature_schema_sha256=raw["feature_schema_sha256"],
            actor_boundary_sha256=raw["actor_boundary_sha256"],
            weights_file_sha256=raw["weights_file_sha256"],
            selected_seed=raw["selected_seed"],
            selected_update=raw["selected_update"],
            decision_rule=raw["decision_rule"],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": ACTOR_MANIFEST_SCHEMA,
            "variant": self.variant.value,
            "feature_order": list(self.feature_order),
            "feature_count": self.feature_count,
            "feature_schema_sha256": self.feature_schema_sha256,
            "actor_boundary_sha256": self.actor_boundary_sha256,
            "weights_file_sha256": self.weights_file_sha256,
            "selected_seed": self.selected_seed,
            "selected_update": self.selected_update,
            "decision_rule": self.decision_rule,
        }


@dataclass(frozen=True, slots=True)
class BoundActorDecisionV1:
    variant: ActorVariant
    mode_id: int
    q_e4: int
    state_sha256: str
    feature_schema_sha256: str
    actor_boundary_sha256: str


class BoundFrozenActorV1:
    """Schema guard around either selected B actor's batch-1 CPU call."""

    def __init__(self, actor: Any, manifest: BActorManifestV1, *,
                 observed_actor_boundary_sha256: str) -> None:
        _require(type(manifest) is BActorManifestV1,
                 "manifest must be exactly BActorManifestV1", ActorBindingError)
        _digest(observed_actor_boundary_sha256,
                "observed_actor_boundary_sha256", ActorBindingError)
        _require(observed_actor_boundary_sha256
                 == manifest.actor_boundary_sha256,
                 "loaded actor differs from the selected actor boundary",
                 ActorBindingError)
        _require(callable(getattr(actor, "act_on_vector", None)),
                 "actor lacks act_on_vector", ActorBindingError)
        self._actor = actor
        self.manifest = manifest

    def act(self, values: tuple[float, ...]) -> BoundActorDecisionV1:
        _require(type(values) is tuple
                 and len(values) == self.manifest.feature_count,
                 "actor input is not the exact registered feature tuple",
                 ActorBindingError)
        normalized: list[float] = []
        for index, value in enumerate(values):
            _require(not isinstance(value, bool)
                     and isinstance(value, (int, float))
                     and math.isfinite(float(value)),
                     f"feature {index} is not a finite real", ActorBindingError)
            normalized.append(float(value))
        exact = tuple(normalized)
        raw = self._actor.act_on_vector(exact)
        mode_id = (raw.get("mode_id") if isinstance(raw, Mapping)
                   else getattr(raw, "mode_id", None))
        q_e4 = (raw.get("q_e4") if isinstance(raw, Mapping)
                else getattr(raw, "q_e4", None))
        _require(type(mode_id) is int and 0 <= mode_id < 12,
                 "actor returned an invalid mode_id", ActorBindingError)
        _require(type(q_e4) is int and 0 <= q_e4 <= 9800,
                 "actor returned an invalid q_e4", ActorBindingError)
        reported_boundary = (raw.get("actor_boundary_sha256")
                             if isinstance(raw, Mapping)
                             else getattr(raw, "actor_boundary_sha256", None))
        if reported_boundary is not None:
            _require(reported_boundary == self.manifest.actor_boundary_sha256,
                     "actor decision reports a foreign boundary",
                     ActorBindingError)
        return BoundActorDecisionV1(
            variant=self.manifest.variant,
            mode_id=mode_id,
            q_e4=q_e4,
            state_sha256=hashlib.sha256(
                _canonical(list(exact))).hexdigest(),
            feature_schema_sha256=self.manifest.feature_schema_sha256,
            actor_boundary_sha256=self.manifest.actor_boundary_sha256,
        )


def identity_from_phase6(*, run_id: str, cell_id: str, envelope: Any,
                         context: Any, profile: Any,
                         require_operational_ack: bool = True
                         ) -> A.FrameActionIdentityV1:
    """Translate the verified SFD4/profile objects without re-resolving q."""
    _require(type(require_operational_ack) is bool,
             "require_operational_ack must be bool", EdgeDispatchError)
    reward_requested = getattr(envelope, "reward_requested", None)
    _require(type(reward_requested) is bool,
             "envelope lacks a boolean reward_requested", EdgeDispatchError)
    if require_operational_ack:
        _require(reward_requested,
                 "only a decision frame can request an operational ACK",
                 EdgeDispatchError)
    for left, right, label in (
        (getattr(context, "frame_id", None), getattr(envelope, "frame_id", None),
         "frame_id"),
        (getattr(context, "sequence_id", None),
         getattr(envelope, "tensor_seq", None), "tensor_seq"),
        (getattr(context, "capture_timestamp_ns", None),
         getattr(envelope, "capture_timestamp_ns", None),
         "capture_timestamp_ns"),
        (getattr(profile, "mode_id", None), getattr(envelope, "mode_id", None),
         "mode_id"),
        (getattr(profile, "q_e4", None), getattr(envelope, "q_e4", None),
         "q_e4"),
        (getattr(profile, "keep_count", None),
         getattr(envelope, "keep_count", None), "keep_count"),
        (getattr(profile, "execution_bundle_sha256", None),
         getattr(envelope, "execution_bundle_sha256", None),
         "execution_bundle_sha256"),
    ):
        _require(left == right, f"{label} differs across verified inputs",
                 EdgeDispatchError)
    anchor = getattr(envelope, "anchor_action_id", None)
    profile_action = getattr(profile, "action_id", None)
    _require(anchor == profile_action,
             "anchor action differs across envelope/profile", EdgeDispatchError)
    profile_id = getattr(profile, "profile_id", None) if anchor is not None else None
    return A.FrameActionIdentityV1(
        run_id=run_id,
        cell_id=cell_id,
        stream_id=getattr(context, "stream_id", None),
        session_uuid=getattr(envelope, "session_uuid", None),
        controller_lineage_sha256=getattr(
            envelope, "controller_lineage_sha256", None),
        decision_seq=getattr(envelope, "decision_seq", None),
        ticket_seq=getattr(envelope, "ticket_seq", None),
        frame_id=getattr(envelope, "frame_id", None),
        tensor_seq=getattr(envelope, "tensor_seq", None),
        capture_timestamp_ns=getattr(envelope, "capture_timestamp_ns", None),
        mode_id=getattr(envelope, "mode_id", None),
        q_e4=getattr(envelope, "q_e4", None),
        keep_count=getattr(envelope, "keep_count", None),
        anchor_action_id=anchor,
        profile_id=profile_id,
        execution_bundle_sha256=getattr(
            envelope, "execution_bundle_sha256", None),
    )


def encode_prediction_tail_output(
        *, identity: A.FrameActionIdentityV1,
        objects: Sequence[Mapping[str, Any]], semantic_mask: Any) -> bytes:
    """Use the shared post-run bundle schema; never duplicate its codec.

    The import is intentionally lazy.  Merely importing the live actor/ACK
    adapters therefore does not import NumPy or any evaluation code.  The
    edge calls this only after the qualified tail has produced its final
    object records and class-label mask.
    """
    from . import postrun_artifact_v1 as artifact

    payload = artifact.encode_bundle(
        kind=artifact.PREDICTION_KIND,
        identity=identity,
        objects=objects,
        semantic_mask=semantic_mask,
    )
    # Decode once at the producer boundary so a foreign kind or identity can
    # never be acknowledged and persisted as if it were this frame.
    artifact.decode_bundle(
        payload,
        expected_kind=artifact.PREDICTION_KIND,
        expected_identity=identity,
    )
    return payload


@dataclass(frozen=True, slots=True)
class BranchCallbackResultV1:
    status: B.BranchStatus
    detail_code: str
    evidence_sha256: Optional[str]

    def __post_init__(self) -> None:
        _require(type(self.status) is B.BranchStatus,
                 "callback status must be BranchStatus", EdgeDispatchError)
        _require(type(self.detail_code) is str
                 and bool(_DETAIL_CODE_RE.fullmatch(self.detail_code)),
                 "callback detail code is empty or unsafe", EdgeDispatchError)
        if self.evidence_sha256 is not None:
            _digest(self.evidence_sha256, "callback evidence_sha256",
                    EdgeDispatchError)
        if self.status is B.BranchStatus.SUCCEEDED:
            _require(self.evidence_sha256 is not None,
                     "successful callback requires an evidence digest",
                     EdgeDispatchError)


@dataclass(frozen=True, slots=True)
class ImmutableTailWorkV1:
    identity: A.FrameActionIdentityV1
    branch: B.Branch
    tail_output: bytes
    tail_output_sha256: str
    enqueued_monotonic_raw_ns: int
    publication: Optional[B.TailOutputPublicationV1]

    def __post_init__(self) -> None:
        _require(type(self.identity) is A.FrameActionIdentityV1,
                 "tail work has a foreign identity", EdgeDispatchError)
        _require(type(self.branch) is B.Branch,
                 "tail work has an invalid branch", EdgeDispatchError)
        _require(type(self.tail_output) is bytes and bool(self.tail_output),
                 "tail work output must be non-empty bytes", EdgeDispatchError)
        _digest(self.tail_output_sha256, "tail_output_sha256",
                EdgeDispatchError)
        _require(hashlib.sha256(self.tail_output).hexdigest()
                 == self.tail_output_sha256,
                 "tail work bytes differ from their digest", EdgeDispatchError)
        _require(type(self.enqueued_monotonic_raw_ns) is int
                 and self.enqueued_monotonic_raw_ns >= 0,
                 "tail work enqueue time must be nonnegative",
                 EdgeDispatchError)
        if self.publication is not None:
            _require(type(self.publication) is B.TailOutputPublicationV1,
                     "tail work publication has the wrong type",
                     EdgeDispatchError)
            expected = (self.publication.map_work
                        if self.branch is B.Branch.MAP
                        else self.publication.evaluation_work)
            _require(expected.identity == self.identity
                     and expected.tail_output_sha256 == self.tail_output_sha256,
                     "tail work differs from its publication",
                     EdgeDispatchError)
        else:
            _require(self.branch is B.Branch.MAP,
                     "only auxiliary map work may omit a publication",
                     EdgeDispatchError)


class _BoundedBranchWorkerV1:
    def __init__(self, *, name: str, branch: B.Branch, depth: int,
                 callback: Callable[[ImmutableTailWorkV1],
                                    BranchCallbackResultV1],
                 coordinator: B.TailOutputBranchCoordinatorV1,
                 coordinator_lock: threading.Lock) -> None:
        _require(type(depth) is int and depth > 0,
                 "branch queue depth must be positive", EdgeDispatchError)
        _require(callable(callback), "branch callback is not callable",
                 EdgeDispatchError)
        self.name = name
        self.branch = branch
        self._callback = callback
        self._coordinator = coordinator
        self._coordinator_lock = coordinator_lock
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=depth)
        self._sentinel = object()
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._stopped = False
        self.errors: list[str] = []
        self.auxiliary_results: list[tuple[str, BranchCallbackResultV1]] = []

    def start(self) -> None:
        _require(not self._started and not self._stopped,
                 f"{self.name} worker already started/stopped",
                 EdgeDispatchError)
        self._thread = threading.Thread(target=self._run, name=self.name,
                                        daemon=True)
        self._started = True
        self._thread.start()

    def submit(self, work: ImmutableTailWorkV1) -> bool:
        _require(self._started and not self._stopped,
                 f"{self.name} worker is not accepting work",
                 EdgeDispatchError)
        _require(type(work) is ImmutableTailWorkV1
                 and work.branch is self.branch,
                 f"{self.name} received foreign branch work",
                 EdgeDispatchError)
        try:
            self._queue.put_nowait(work)
        except queue.Full:
            return False
        return True

    def _complete(self, work: ImmutableTailWorkV1,
                  result: BranchCallbackResultV1) -> None:
        _require(type(result) is BranchCallbackResultV1,
                 "branch callback returned a foreign result",
                 EdgeDispatchError)
        if work.publication is None:
            self.auxiliary_results.append(
                (work.identity.exact_sha256(), result))
            return
        branch_work = (work.publication.map_work
                       if work.branch is B.Branch.MAP
                       else work.publication.evaluation_work)
        with self._coordinator_lock:
            self._coordinator.complete(
                branch_work,
                status=result.status,
                detail_code=result.detail_code,
                evidence_sha256=result.evidence_sha256,
            )

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._sentinel:
                    return
                try:
                    result = self._callback(item)
                    self._complete(item, result)
                except Exception as exc:  # branch failure cannot retract ACK
                    self.errors.append(f"{type(exc).__name__}: {exc}")
                    failed = BranchCallbackResultV1(
                        status=B.BranchStatus.FAILED,
                        detail_code="CALLBACK_FAILED",
                        evidence_sha256=None,
                    )
                    try:
                        self._complete(item, failed)
                    except Exception as completion_exc:
                        # A conflicting/foreign completion is audit-visible but
                        # must not kill this independent worker or strand later
                        # queue items behind Queue.join().
                        self.errors.append(
                            f"completion {type(completion_exc).__name__}: "
                            f"{completion_exc}")
            finally:
                self._queue.task_done()

    def drain(self) -> None:
        _require(self._started, f"{self.name} worker was not started",
                 EdgeDispatchError)
        self._queue.join()

    def stop(self) -> None:
        if self._stopped:
            return
        _require(self._started, f"{self.name} worker was not started",
                 EdgeDispatchError)
        self.drain()
        self._queue.put(self._sentinel)
        self._queue.join()
        assert self._thread is not None
        self._thread.join(timeout=5.0)
        _require(not self._thread.is_alive(),
                 f"{self.name} worker did not stop", EdgeDispatchError)
        self._stopped = True


@dataclass(frozen=True, slots=True)
class DispatchReceiptV1:
    identity_sha256: str
    ack_packet_sha256: str
    map_enqueued: bool
    prediction_enqueued: bool


class TailOutputDispatchV1:
    """Immediate operational ACK plus two explicit, bounded side branches."""

    def __init__(self, *, ack_sender: Callable[[bytes], Any], queue_depth: int,
                 map_callback: Callable[[ImmutableTailWorkV1],
                                        BranchCallbackResultV1],
                 prediction_callback: Callable[[ImmutableTailWorkV1],
                                               BranchCallbackResultV1]) -> None:
        _require(callable(ack_sender), "ACK sender is not callable",
                 EdgeDispatchError)
        self._ack_sender = ack_sender
        self.coordinator = B.TailOutputBranchCoordinatorV1()
        self._coordinator_lock = threading.Lock()
        self._map = _BoundedBranchWorkerV1(
            name="run4b5b-map-branch", branch=B.Branch.MAP,
            depth=queue_depth, callback=map_callback,
            coordinator=self.coordinator,
            coordinator_lock=self._coordinator_lock,
        )
        self._prediction = _BoundedBranchWorkerV1(
            name="run4b5b-prediction-branch", branch=B.Branch.EVALUATION,
            depth=queue_depth, callback=prediction_callback,
            coordinator=self.coordinator,
            coordinator_lock=self._coordinator_lock,
        )
        self._started = False
        self._stopped = False

    @property
    def worker_errors(self) -> tuple[str, ...]:
        return tuple(self._map.errors + self._prediction.errors)

    @property
    def auxiliary_map_results(self) -> tuple[tuple[str, BranchCallbackResultV1], ...]:
        return tuple(self._map.auxiliary_results)

    def start(self) -> None:
        _require(not self._started and not self._stopped,
                 "tail dispatcher already started/stopped", EdgeDispatchError)
        self._map.start()
        self._prediction.start()
        self._started = True

    @staticmethod
    def _work(publication: B.TailOutputPublicationV1, branch: B.Branch,
              tail_output: bytes, enqueued_ns: int) -> ImmutableTailWorkV1:
        return ImmutableTailWorkV1(
            identity=publication.ack.identity,
            branch=branch,
            tail_output=bytes(tail_output),
            tail_output_sha256=publication.ack.tail_output_sha256,
            enqueued_monotonic_raw_ns=enqueued_ns,
            publication=publication,
        )

    def dispatch_decision(self, identity: A.FrameActionIdentityV1,
                          tail_output: bytes, *,
                          tail_ready_monotonic_raw_ns: int) -> DispatchReceiptV1:
        _require(self._started and not self._stopped,
                 "tail dispatcher is not active", EdgeDispatchError)
        _require(type(tail_ready_monotonic_raw_ns) is int
                 and tail_ready_monotonic_raw_ns >= 0,
                 "tail-ready time must be nonnegative", EdgeDispatchError)
        publication = self.coordinator.publish(identity, bytes(tail_output))
        ack_packet = A.encode_ack(publication.ack)

        # This synchronous call is deliberately before either queue offer.
        # Therefore neither branch can delay the operational outcome.
        self._ack_sender(ack_packet)

        map_work = self._work(publication, B.Branch.MAP, tail_output,
                              tail_ready_monotonic_raw_ns)
        prediction_work = self._work(publication, B.Branch.EVALUATION,
                                     tail_output,
                                     tail_ready_monotonic_raw_ns)
        map_enqueued = self._map.submit(map_work)
        prediction_enqueued = self._prediction.submit(prediction_work)
        if not map_enqueued:
            with self._coordinator_lock:
                self.coordinator.complete(
                    publication.map_work, status=B.BranchStatus.SKIPPED,
                    detail_code="QUEUE_FULL", evidence_sha256=None)
        if not prediction_enqueued:
            with self._coordinator_lock:
                self.coordinator.complete(
                    publication.evaluation_work,
                    status=B.BranchStatus.SKIPPED,
                    detail_code="QUEUE_FULL", evidence_sha256=None)
        return DispatchReceiptV1(
            identity_sha256=identity.exact_sha256(),
            ack_packet_sha256=hashlib.sha256(ack_packet).hexdigest(),
            map_enqueued=map_enqueued,
            prediction_enqueued=prediction_enqueued,
        )

    def dispatch_map_only(self, identity: A.FrameActionIdentityV1,
                          tail_output: bytes, *,
                          tail_ready_monotonic_raw_ns: int) -> bool:
        """Queue a hold/fallback output without creating or sending an ACK."""
        _require(self._started and not self._stopped,
                 "tail dispatcher is not active", EdgeDispatchError)
        work = ImmutableTailWorkV1(
            identity=identity,
            branch=B.Branch.MAP,
            tail_output=bytes(tail_output),
            tail_output_sha256=hashlib.sha256(tail_output).hexdigest(),
            enqueued_monotonic_raw_ns=tail_ready_monotonic_raw_ns,
            publication=None,
        )
        return self._map.submit(work)

    def drain(self) -> None:
        _require(self._started, "tail dispatcher was not started",
                 EdgeDispatchError)
        self._map.drain()
        self._prediction.drain()

    def stop(self) -> None:
        if self._stopped:
            return
        _require(self._started, "tail dispatcher was not started",
                 EdgeDispatchError)
        self._map.stop()
        self._prediction.stop()
        self._stopped = True


def prediction_store_callback(
        store: B.PredictionEvidenceStoreV1,
        *, clock: Callable[[], int],
        ) -> Callable[[ImmutableTailWorkV1], BranchCallbackResultV1]:
    """Adapt the create-only prediction store to the bounded worker."""
    _require(type(store) is B.PredictionEvidenceStoreV1,
             "store must be exactly PredictionEvidenceStoreV1",
             EdgeDispatchError)
    _require(callable(clock), "prediction clock is not callable",
             EdgeDispatchError)

    def retain(work: ImmutableTailWorkV1) -> BranchCallbackResultV1:
        _require(work.publication is not None
                 and work.branch is B.Branch.EVALUATION,
                 "prediction store received non-decision work",
                 EdgeDispatchError)
        record = store.write(work.publication, work.tail_output, int(clock()))
        return BranchCallbackResultV1(
            status=B.BranchStatus.SUCCEEDED,
            detail_code="PREDICTION_RETAINED",
            evidence_sha256=hashlib.sha256(record.canonical_bytes()).hexdigest(),
        )

    return retain
