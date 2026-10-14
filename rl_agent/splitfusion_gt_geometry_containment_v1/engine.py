"""``ContainedEngine``: the unchanged ``DeferredProducerEngine`` plus per-frame containment of the recognised refusal.

Unchanged path: every method delegates to the inherited implementation.  Only when the inherited
``prepare_unbound`` raises an error that ``classify.recognise`` accepts does anything differ:

1. The inherited ``prepare_unbound`` has ALREADY done its normal failure cleanup: the owner transaction was cancelled
   through the pinned producer (exact history rollback, verified by the producer itself) and the engine recorded the
   error in its sticky ``_failure``.  We then verify, against a snapshot taken immediately before the call, that the
   rollback is complete: producer history digest, owner retained frames, reservation, pending transaction, payload
   bytes and owner failure are all back to their pre-frame values, no engine frame was issued, nothing is unbound,
   and the engine failure text is exactly the one written for this error.  Any mismatch -> the error is re-raised
   and the engine stays failed (fatal, exactly as before).
2. Only then are the error's finished frames cleared (they would otherwise keep the borrowed decoded source views
   exported past the FREEZE mapping release), the engine failure for THIS error cleared, and a refused unbound slot
   created with a normal
   ``UnboundHandle``.  The slot holds no owner token, payload, products or source bytes.  The one-unbound-at-a-time
   rule, the service's ``prep_open`` ordering and the freeze gate sequence are therefore unchanged.
3. ``bind`` on that handle first applies the SAME plan validation as a normal frame's bind, in the same order and
   with the same failure semantics (owner ``bind_plan`` checks -> fatal; identity/anchor -> the existing identity
   ``DeferralRefused``; payload-independent admission context checks incl. the live-context/source-camera pose
   agreement on the retained source camera -> fatal).  Only a valid plan then gets the existing ``DeferralRefused``
   with reason ``GT_GEOMETRY_ENVELOPE_*``.  The
   unchanged PrepCore maps it to ``Refused('DEFERRAL_REFUSED_' + reason)``; the unchanged service records exactly one
   ``NO_GT_DEFERRAL_REFUSED_<reason>`` outcome, releases a waiting prediction and sends the edge an explicit UNSCORED
   (-> retained GROUND_TRUTH_UNAVAILABLE feedback).  ``cancel_unbound`` (hold/fallback/send failure/abandon) and
   ``close`` release the slot.  No score, quality, target, ACK or prediction is ever produced for the frame.
"""
from __future__ import annotations

import copy
import traceback

from rl_agent.splitfusion_exact_gt_service_proto_v2 import deferred as D
import numpy as np

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_numeric_gt_integration_v2 import ready as R
from rl_agent.splitfusion_numeric_gt_runtime_v1 import producer as PR
from rl_agent.splitfusion_run5_live_avo_v1.quality_adapter import LiveGtContractError

from . import classify as CL

MAX_MEMORY_RECORDS = 64        # bounded in-process refusal records (counters are unbounded integers)


class ContainmentError(PR.ProducerError):
    pass


def _owner_view(engine):
    owner = engine._owner
    producer = owner._producer
    return dict(history_sha256=producer.history_sha256(), producer_staged_none=getattr(producer, "staged", None) is None,
                owner_frames=tuple(sorted(owner._frames)), owner_reserved=owner._reserved,
                owner_pending_none=owner._pending is None, owner_failure=owner._failure,
                owner_payload_bytes=owner.snapshot()["payload_bytes"], engine_issued=engine._issued,
                engine_frames=tuple(sorted(engine._frames)), engine_unbound_none=engine._unbound is None)


class ContainedEngine(D.DeferredProducerEngine):
    """Never constructed directly: ``adopt`` converts the exact constructed (possibly capacity-substituted) engine."""

    def __init__(self, *args, **kwargs):
        raise ContainmentError("ContainedEngine is adopted from a constructed DeferredProducerEngine, never built")

    # ------------------------------------------------------------------ containment state
    def _refusal(self, handle):
        state = self._unbound
        if (type(handle) is D.UnboundHandle and state is not None and state.handle is handle
                and handle.sequence in self._contained_refusals):
            return self._contained_refusals[handle.sequence]
        return None

    def contained_refusal(self, handle):
        """The pending refusal record for this handle (read-only copy), or None."""
        record = self._refusal(handle)
        return None if record is None else copy.deepcopy(record)

    def _decline(self, why):
        declined = self._containment["declined"]
        if why in declined or len(declined) < 32:
            declined[why] = declined.get(why, 0) + 1

    # ------------------------------------------------------------------ inherited operations
    def prepare_unbound(self, sourcekwargs, *, population, **kwargs):
        if self._failure is not None or self._unbound is not None or self._closed:
            return super().prepare_unbound(sourcekwargs, population=population, **kwargs)   # raises as before
        before = _owner_view(self)
        try:
            return super().prepare_unbound(sourcekwargs, population=population, **kwargs)
        except LiveGtContractError as error:
            record, why = CL.recognise(error)
            if record is None:
                self._decline(why)
                raise
            after = _owner_view(self)
            expected_failure = f"unbound prepare failed: {type(error).__name__}: {error}"[:1024]
            problems = [k for k in before if k not in ("engine_unbound_none",) and before[k] != after[k]]
            if not after["engine_unbound_none"]:
                problems.append("engine_unbound_not_released")
            if self._failure != expected_failure:
                problems.append("engine_failure_text")
            if problems:
                self._decline("ROLLBACK_INVARIANT:" + ",".join(problems))
                raise
            # The refused error and its finished frames form a cycle (``_envelope`` holds ``error`` as a local) that
            # would keep the borrowed decoded views exported past the caller's mmap release (the V6 BufferError
            # class).  Clear those frames now; only the plain-data record is kept.
            traceback.clear_frames(error.__traceback__)
            error.__traceback__ = None
            self._failure = None
            template = C._json_owned(sourcekwargs["identity"])
            self._unbound_sequence += 1
            handle = D.UnboundHandle(self._unbound_sequence, template["frame_id"], C.sha256(C.canonical(template)),
                                     self._unbound_issuer)
            now = PR._raw()
            self._unbound = D._Unbound(handle, None, None, template, None, None, None, None, None,
                                       before["owner_payload_bytes"], now, now)
            record = dict(record, frame_id=template["frame_id"], unbound_sequence=handle.sequence,
                          history_sha256_before=before["history_sha256"],
                          history_sha256_after=after["history_sha256"], resolution=None)
            self._contained_refusals[handle.sequence] = record
            self._refused_geometry[handle.sequence] = dict(camera_matrix=np.array(
                sourcekwargs["camera_matrix"], dtype=np.float64, copy=True).tolist())
            c = self._containment
            c["contained"] += 1
            c["by_reason"][record["reason"]] = c["by_reason"].get(record["reason"], 0) + 1
            if len(c["records"]) < MAX_MEMORY_RECORDS:
                c["records"].append(record)
            else:
                c["records_dropped"] += 1
            return handle

    def bind(self, handle, plan_identity_bytes, *, frame_context):
        record = self._refusal(handle)
        if record is None:
            return super().bind(handle, plan_identity_bytes, frame_context=frame_context)
        self._check()
        state = self._take_unbound(handle)
        geometry = self._refused_geometry[handle.sequence]
        # Same three validation layers, order and failure semantics as the inherited bind of a normal frame; only
        # after all of them pass is the geometry refusal returned (review_v1 blocker 2).
        plan = self._validate_plan(plan_identity_bytes)                       # 1. owner bind_plan checks: fatal
        capture = plan["capture"]
        provenance = None
        try:
            if C.canonical(capture) != C.canonical(state.template):           # 2. identity: existing refusal
                provenance = D.anchor_provenance(state.template, plan_identity_bytes)
        except D.DeferralRefused as refused:
            self._release_refused(handle, record, "IDENTITY_REFUSED:" + refused.reason)
            self._deferral["refused"] += 1
            self._deferral.setdefault("refused_reasons", {})
            self._deferral["refused_reasons"][refused.reason] = self._deferral["refused_reasons"].get(refused.reason, 0) + 1
            raise
        except BaseException as error:
            self._failure = self._failure or f"plan binding failed: {error}"[:1024]
            raise
        try:                                                                   # 3. admission context checks: fatal
            self._validate_admission(state.template, plan, provenance, frame_context, plan_identity_bytes, geometry)
        except BaseException as error:
            self._failure = self._failure or f"bound admission failed: {type(error).__name__}: {error}"[:1024]
            raise
        self._release_refused(handle, record, "BIND_REFUSED")
        reason = record["reason"]
        self._deferral["refused"] += 1
        self._deferral.setdefault("refused_reasons", {})
        self._deferral["refused_reasons"][reason] = self._deferral["refused_reasons"].get(reason, 0) + 1
        self._containment["bind_refused"] += 1
        raise D.DeferralRefused(f"ground truth unavailable for this frame: {record['status']} "
                                f"(actor {record['offender']['gt_actor_id']})", reason)

    def _release_refused(self, handle, record, resolution):
        self._unbound = None
        del self._contained_refusals[handle.sequence]
        self._refused_geometry.pop(handle.sequence, None)
        record["resolution"] = resolution

    def _validate_plan(self, plan_identity_bytes):
        """The checks of ``DeferredProducerOwner.bind_plan`` (deferred.py, its ``try`` block), verbatim in order.

        As there, a failure records the owner failure ``invalid producer plan: ...`` and the engine failure
        ``plan binding failed: ...`` (``DeferredProducerEngine.bind``) and re-raises the original exception."""
        owner = self._owner
        try:
            plan = C._parse(plan_identity_bytes)
            if C.canonical(plan) != plan_identity_bytes or not isinstance(plan, dict):
                raise D.OWN.OwnershipError("canonical full plan identity bytes required")
            capture, action = plan["capture"], plan["action"]
            C._capture(capture)
            if type(plan.get("frame_context")) is not dict:
                raise D.OWN.OwnershipError("full immutable plan must include its frame context")
            from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as MP
            MP.validate_run4_identity(action)
            if action["reward_requested"] is not True or action["frame_id"] != capture["frame_id"]:
                raise D.OWN.OwnershipError("only accepted reward-requested frame plans are admitted")
            if action.get("frame_kind") != "POLICY_DECISION":
                raise D.OWN.OwnershipError("only policy decisions are admitted; holds/fallbacks never bind")
            if len(plan_identity_bytes) >= self._reservation:
                raise D.OWN.OwnershipError("explicit frame payload reservation required")
        except BaseException as error:
            owner._failure = owner._failure or f"invalid producer plan: {error}"[:1024]
            self._failure = self._failure or f"plan binding failed: {error}"[:1024]
            raise
        return plan

    def _validate_admission(self, template, plan, provenance, frame_context, plan_identity_bytes, geometry):
        """Every check of ``ready.admit_payload`` -> ``ready._metadata`` that does not read the (never produced)
        payload, with the same expected values the inherited ``bind`` passes; plus the live-context/source-camera
        pose agreement of ``ready._frame_geometry`` on the retained source camera, using the comparison that
        ``_frame_geometry`` uses in this process (the registered P3 rule when installed, else ``PO._agree``)."""
        action = C._json_owned(plan["action"] if provenance is None else provenance["neutral_action_identity"])
        capture = C._json_owned(template)
        C._capture(capture)
        R._action(action, capture)
        C.require(all(C.is_digest(x) for x in (C.sha256(plan_identity_bytes), self._run_binding,
                                                 self._producer_binding)), "expected binding invalid")
        R._context_document(frame_context)
        R._prediction_contract(C._json_owned(self._prediction_contract), frame_context)
        C.require((frame_context.stream_id, frame_context.frame_id, frame_context.capture_timestamp_ns)
                  == (capture["stream_id"], capture["frame_id"], capture["capture_timestamp_ns"])
                  and frame_context.sequence_id == action["tensor_seq"], "capture/action/live context disagreement")
        camera = R.PO._matrix(R._numeric_matrix(geometry["camera_matrix"], 4, "camera"), "producer camera")
        substituted = getattr(R._frame_geometry, "__m3_capacity_substitution__", None) == "P3_GT_LIVE_CONTEXT_SOURCE_CAMERA"
        agree = R._M3_AGREE_RECONSTRUCTED if substituted else R.PO._agree
        agree(camera, R.PO.context_camera(frame_context, R._CAMERA_REGISTRY), "live context/source camera")

    def cancel_unbound(self, handle, reason):
        record = self._refusal(handle)
        if record is None:
            return super().cancel_unbound(handle, reason)
        self._take_unbound(handle)
        PR._require(type(reason) is str and bool(reason), "explicit cancellation reason required")
        self._release_refused(handle, record, "CANCELLED_UNBOUND:" + reason[:64])
        self._deferral["cancelled_unbound"] += 1
        self._containment["cancelled_unbound"] += 1
        producer = self._owner._producer
        return {"frame_id": handle.frame_id, "history_sha256": producer.history_sha256(),
                "processed_frames": producer.processed_frames}

    def close(self, *, abandon=False):
        if self._unbound is not None and self._unbound.handle.sequence in self._contained_refusals:
            state, self._unbound = self._unbound, None
            self._contained_refusals.pop(state.handle.sequence)["resolution"] = "CLOSED_WHILE_UNBOUND"
            self._refused_geometry.pop(state.handle.sequence, None)
            if not abandon:                  # same semantics as the inherited close for an unbound transaction
                self._failure = self._failure or "closed with an unbound transaction"
        return super().close(abandon=abandon)

    def snapshot(self):
        value = super().snapshot()
        c = self._containment
        value["deferral"]["geometry_containment"] = dict(
            version=CL.VERSION, contained=c["contained"], bind_refused=c["bind_refused"],
            cancelled_unbound=c["cancelled_unbound"], by_reason=dict(c["by_reason"]), declined=dict(c["declined"]),
            pending=len(self._contained_refusals), records_kept=len(c["records"]), records_dropped=c["records_dropped"])
        return value


def adopt(engine):
    """Convert one exactly-typed, freshly constructed ``DeferredProducerEngine`` in place (no state is copied)."""
    if type(engine) is not D.DeferredProducerEngine:
        raise ContainmentError(f"containment adopts only DeferredProducerEngine, not {type(engine).__name__}")
    if engine._unbound is not None or engine._issued or engine._failure is not None or engine._closed:
        raise ContainmentError("containment must be adopted before the engine is used")
    engine.__class__ = ContainedEngine
    engine._contained_refusals = {}
    engine._refused_geometry = {}
    engine._containment = dict(contained=0, bind_refused=0, cancelled_unbound=0, by_reason={}, declined={},
                               records=[], records_dropped=0)
    return engine
