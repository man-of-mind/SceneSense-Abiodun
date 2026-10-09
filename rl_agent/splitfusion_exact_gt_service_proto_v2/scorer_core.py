"""Scorer-process core: prepared-target cache, exact join, exactly-once scoring.

A published ReadyTarget packet is admitted ONCE (unchanged ``ready.decode_target``)
when it arrives, normally before the prediction. A prediction is joined by the
exact (capture identity, action identity, frame context) triple that the edge
derives from its own ticket, admitted with the unchanged
``scoring.admit_prediction``, joined and scored by the unchanged numerical
scorer and quality formula. Evidence formatting (the unchanged live score
document builder) and durable archival happen only after the result was handed
out, and retire the slot only against an exact durable receipt.

Capacity is bounded by slots, bytes and waiting predictions; nothing is evicted.
Over-capacity, duplicate, late-after-terminal and non-decision inputs are explicit
``Refused`` outcomes that leave state unchanged. Non-decision (hold/fallback)
predictions are never scored, so no feedback can be manufactured for them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import time

import numpy as np

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_numeric_gt_integration_v2 import ready as R, scoring as Q
from rl_agent.splitfusion_numeric_gt_runtime_v1 import live_archive as LA, scorer as SCORER

LABEL_BYTES = 720 * 1280
UNSCORED_SCHEMA = "splitfusion.exact_gt_service_proto.scorer_unscored_terminal.v1"
PROOF_SCHEMA = "splitfusion.exact_gt_service_proto.scorer_durable_proof.v1"
IDENTITY_FIELDS = {"capture_identity", "action_identity", "plan_sha256", "run_binding", "producer_binding",
                   "frame_context", "prediction_contract"}
PREDICTION_FIELDS = {"gt_identity", "action_identity", "capture_timestamp_ns", "prediction_digest"}


class Refused(Exception):
    """Protocol refusal: the request is rejected and no state changed."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


class ScorerFault(RuntimeError):
    pass


def _raw():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


def triple_key(capture, action, context_doc):
    """Join key on the action-neutral source identity (v2).

    For template captures this is the identical v1 key. For a catalogue-anchor capture the key is its
    neutral source identity; exact agreement of the received identity with the target's explicit anchor
    provenance mapping is then REQUIRED at scoring (fail closed), so neutralisation never joins
    inputs that the mapping does not bind.
    """
    from rl_agent.splitfusion_numeric_gt_runtime_v1 import staging as ST
    from .deferred import neutral_action
    return C.canonical([ST.neutral_identity(C._json_owned(capture)), neutral_action(action), C._json_owned(context_doc)])


def join_identity(received, provenance, target_capture):
    """Derive the identity handed to the unchanged pinned admission/join.

    Without provenance the received gt_identity must already equal the target capture exactly.
    With provenance the received gt_identity must equal the mapping's selected capture exactly and the
    action identity must equal the mapping's; the join identity then carries the mapping's neutral
    capture (the identity the target's hashed products were computed on). Anything else fails closed.
    """
    received = C._json_owned(received)
    if provenance is None:
        C.require(C.canonical(received["gt_identity"]) == C.canonical(target_capture),
                  "prediction capture identity differs from the target and no anchor provenance binds it")
        return received, "IDENTITY"
    C.require(C.canonical(received["gt_identity"]) == C.canonical(provenance["selected_capture"])
              and C.canonical(received["action_identity"]) == C.canonical(provenance["action_identity"])
              and C.canonical(provenance["neutral_capture"]) == C.canonical(target_capture),
              "prediction identity differs from the target's anchor provenance mapping")
    return dict(received, gt_identity=C._json_owned(provenance["neutral_capture"]),
                action_identity=C._json_owned(provenance["neutral_action_identity"])), "ANCHOR_PROVENANCE_V1"


def expected_from_identity(identity_bytes):
    value = C._parse(identity_bytes)
    C.require(type(value) is dict and C.canonical(value) == identity_bytes and set(value) == IDENTITY_FIELDS,
              "exact canonical full target identity required")
    expected = SCORER._expected(dict(expected_capture_identity=value["capture_identity"],
        expected_action_identity=value["action_identity"], frame_context=value["frame_context"],
        plan_sha256=value["plan_sha256"], run_binding=value["run_binding"],
        producer_binding=value["producer_binding"], prediction_contract=value["prediction_contract"]))
    C.require(SCORER.registration_key(expected) == identity_bytes, "registration key drift")
    return value, expected


@dataclass
class Slot:
    key: bytes
    triple: bytes
    target: object
    packet_bytes: int
    state: str = "TARGET_READY"     # -> SCORED -> DELIVERED -> ARCHIVE_SUBMITTED ; or UNSCORED -> ARCHIVE_SUBMITTED
    prediction: object = None
    result: object = None
    answer: dict | None = None
    unscored_reason: str | None = None
    provenance: dict | None = None
    join: dict | None = None
    files: dict | None = None
    events: list = field(default_factory=list)


class ProtoScorer:
    def __init__(self, *, run_binding, producer_binding, max_jobs, max_bytes, reservation_bytes,
                 max_waiting, max_frames=1000, spec=None, quality=None):
        C.require(C.is_digest(run_binding) and C.is_digest(producer_binding), "scorer bindings required")
        C.require(all(type(v) is int and v > 0 for v in (max_jobs, max_bytes, reservation_bytes, max_waiting, max_frames))
                  and reservation_bytes <= max_bytes, "explicit positive scorer bounds required")
        if spec is None:
            from pathlib import Path
            from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
            from rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid import quality
            spec = W.load_run4_quality_spec(Path(__file__).resolve().parents[2])
        self.spec, self.quality = spec, quality
        self.scorer = Q.NumericalScorer()
        self.run_binding, self.producer_binding = run_binding, producer_binding
        self.max_jobs, self.max_bytes, self.reservation = max_jobs, max_bytes, reservation_bytes
        self.max_waiting, self.max_frames = max_waiting, max_frames
        self.slots, self.index, self.waiting, self.terminal = {}, {}, {}, {}
        self.counters = dict(targets=0, predictions=0, scored=0, delivered=0, unscored=0, retired=0,
                             waiting_joined=0, refused={})
        self.failure = None
        self.closed = False

    # ------------------------------------------------------------------ helpers
    def _check(self):
        if self.closed:
            raise ScorerFault("scorer core is closed")
        if self.failure is not None:
            raise ScorerFault(self.failure)

    def _refuse(self, reason):
        self.counters["refused"][reason] = self.counters["refused"].get(reason, 0) + 1
        raise Refused(reason)

    def _slot(self, key):
        slot = self.slots.get(key)
        if slot is None:
            self._refuse("UNKNOWN_OR_RETIRED_IDENTITY")
        return slot

    def reserved(self):
        return len(self.slots) * self.reservation

    def _logical_bytes(self, slot):
        amount = slot.packet_bytes
        if slot.prediction is not None:
            amount += len(slot.prediction.labels) + len(slot.prediction.raw_records)
        if slot.files is not None:
            amount += sum(map(len, slot.files.values()))
        C.require(amount <= self.reservation, "scorer slot exceeds its reservation")
        return amount

    # ------------------------------------------------------------------ inputs
    def admit_target(self, identity_bytes, semantic_sha256, packet, provenance=None):
        self._check()
        value, expected = expected_from_identity(identity_bytes)
        if provenance is not None:
            from .deferred import validate_anchor_provenance
            try:
                provenance = validate_anchor_provenance(C._parse(provenance.encode() if type(provenance) is str
                                                                 else provenance), target_identity=value)
            except Exception as error:
                self.failure = f"anchor provenance integrity failure: {error}"[:1024]
                raise
        if value["run_binding"] != self.run_binding or value["producer_binding"] != self.producer_binding:
            self._refuse("FOREIGN_BINDING")
        triple = triple_key(value["capture_identity"], value["action_identity"], value["frame_context"])
        if identity_bytes in self.slots or triple in self.index:
            self._refuse("DUPLICATE_TARGET")
        if triple in self.terminal:
            self._refuse("LATE_AFTER_TERMINAL")
        if len(self.slots) >= self.max_jobs or self.reserved() + self.reservation > self.max_bytes:
            self._refuse("SCORER_CAPACITY")
        try:
            target = R.decode_target(packet, **expected)
            C.require(target.identity_bytes == identity_bytes and target.semantic_sha256 == semantic_sha256,
                      "published target identity/semantic differs")
        except Exception as error:
            self.failure = f"target admission failed: {error}"[:1024]
            raise
        slot = Slot(identity_bytes, triple, target, len(packet))
        slot.provenance = provenance
        slot.events.append(("target_admitted", _raw()))
        self._logical_bytes(slot)
        self.slots[identity_bytes], self.index[triple] = slot, identity_bytes
        self.counters["targets"] += 1
        waiting = self.waiting.pop(triple, None)
        if waiting is not None:
            self.counters["waiting_joined"] += 1
            return dict(status="ADMITTED", joined_result=self._score(slot, waiting))
        return dict(status="ADMITTED", joined_result=None)

    def admit_prediction(self, identity, frame_context_doc, records, labels):
        self._check()
        C.require(type(identity) is dict and set(identity) == PREDICTION_FIELDS and type(labels) is bytes
                  and len(labels) == LABEL_BYTES and type(records) is list, "exact prediction inputs required")
        action = identity["action_identity"]
        if action.get("reward_requested") is not True or action.get("frame_kind") != "POLICY_DECISION":
            self._refuse("NO_FEEDBACK_FOR_NON_DECISION")
        triple = triple_key(identity["gt_identity"], action, frame_context_doc)
        if triple in self.terminal:
            self._refuse("LATE_AFTER_TERMINAL")
        prediction = dict(identity=C._json_owned(identity), records=C._json_owned(records), labels=labels)
        key = self.index.get(triple)
        if key is None:
            if triple in self.waiting:
                self._refuse("DUPLICATE_PREDICTION")
            if len(self.waiting) >= self.max_waiting:
                self._refuse("WAITING_CAPACITY")
            self.waiting[triple] = prediction
            self.counters["predictions"] += 1
            return dict(status="WAITING_FOR_TARGET", result=None)
        slot = self.slots[key]
        if slot.prediction is not None or slot.state != "TARGET_READY":
            self._refuse("DUPLICATE_PREDICTION" if slot.prediction is not None else "SLOT_ALREADY_TERMINAL")
        self.counters["predictions"] += 1
        return dict(status="SCORED", result=self._score(slot, prediction))

    def _score(self, slot, prediction):
        try:
            identity, mode = join_identity(prediction["identity"], slot.provenance,
                                           C.thaw(slot.target.metadata["capture_identity"]))
            slot.join = dict(mode=mode, received_identity=prediction["identity"], join_identity=identity)
            view = np.frombuffer(prediction["labels"], np.uint8).reshape(720, 1280)
            admitted = Q.admit_prediction(self.scorer, identity=identity, labels=view,
                records=prediction["records"], prediction_contract=C.thaw(slot.target.metadata["prediction_contract"]))
            slot.events.append(("prediction_admitted", _raw()))
            frame = Q.join_admitted_inputs(self.scorer, target=slot.target, prediction=admitted)
            slot.events.append(("score_start", _raw()))
            result = self.scorer.score(frame)
            value = self.quality.evaluate_exact_quality(self.spec, result.measurement)
            slot.events.append(("score_end", _raw()))
        except Exception as error:
            self.failure = f"numerical scoring failed: {type(error).__name__}: {error}"[:1024]
            raise
        q_perc = None if value is None else float(value.q_perc)
        slot.prediction, slot.result, slot.state = admitted, result, "SCORED"
        slot.answer = dict(frame_id=frame.frame_id, measurement=C._json_owned(result.measurement), q_perc=q_perc,
                           quality_defined=value is not None, ready_semantic_sha256=slot.target.semantic_sha256,
                           identity_sha256=hashlib.sha256(slot.key).hexdigest())
        self._logical_bytes(slot)
        self.counters["scored"] += 1
        return dict(slot.answer)

    # ------------------------------------------------------------------ outcomes
    def get_result(self, identity_bytes):
        self._check()
        slot = self._slot(identity_bytes)
        return None if slot.answer is None else dict(slot.answer)

    def result_delivered(self, identity_bytes):
        self._check()
        slot = self._slot(identity_bytes)
        if slot.state != "SCORED":
            self._refuse("DELIVERY_REQUIRES_UNDELIVERED_SCORE")
        slot.state = "DELIVERED"
        slot.events.append(("result_delivered", _raw()))
        self.counters["delivered"] += 1
        return True

    def abandon(self, triple, reason):
        """Explicit no-score terminal (missing prediction, refused target, edge terminal)."""
        self._check()
        C.require(type(reason) is str and reason.isupper() and 3 <= len(reason) <= 64, "bounded reason required")
        if triple in self.waiting:
            self.waiting.pop(triple)
            self._terminal(triple, "WAITING_ABANDONED_" + reason)
            return dict(status="WAITING_DROPPED", identity_sha256=None)
        key = self.index.get(triple)
        if key is None:
            if triple in self.terminal:
                self._refuse("ALREADY_TERMINAL")
            self._terminal(triple, "NO_INPUT_" + reason)
            return dict(status="NOTHING_HELD", identity_sha256=None)
        slot = self.slots[key]
        if slot.state != "TARGET_READY":
            self._refuse("ALREADY_SCORED_OR_TERMINAL")
        slot.state, slot.unscored_reason = "UNSCORED", reason
        self.counters["unscored"] += 1
        return dict(status="UNSCORED", identity_sha256=hashlib.sha256(key).hexdigest())

    def _terminal(self, triple, outcome):
        C.require(len(self.terminal) < self.max_frames, "bounded terminal ledger exceeded")
        self.terminal[triple] = outcome

    # ------------------------------------------------------------------ evidence (off the feedback path)
    def archive_build(self, identity_bytes):
        self._check()
        slot = self._slot(identity_bytes)
        if slot.files is not None:
            return dict(slot.files)
        target = slot.target
        if slot.state == "DELIVERED":
            _, score_bytes = LA._scored_values(target, slot.result, slot.answer["q_perc"], self.spec)
            files = {"numerical_score.json": score_bytes, "target_metadata.json": target.metadata_bytes,
                     "segmentation.u8": target.segmentation_bytes, "object_ignore.u8": target.object_ignore_bytes,
                     "predicted_labels.u8": slot.prediction.labels}
            if slot.provenance is not None:
                files["anchor_provenance.json"] = C.canonical(dict(
                    schema="splitfusion.exact_gt_service.scorer_anchor_join.v1", provenance=slot.provenance,
                    join_mode=slot.join["mode"], prediction_identity_as_received=slot.join["received_identity"],
                    join_identity=slot.join["join_identity"]))
        elif slot.state == "UNSCORED":
            terminal = dict(schema=UNSCORED_SCHEMA, ready_identity=C._parse(slot.key),
                ready_identity_sha256=hashlib.sha256(slot.key).hexdigest(),
                ready_semantic_sha256=target.semantic_sha256, reason=slot.unscored_reason,
                score_exists=False, quality_feedback_emitted=False)
            files = {"unscored_terminal.json": C.canonical(terminal), "target_metadata.json": target.metadata_bytes}
            if slot.provenance is not None:
                files["anchor_provenance.json"] = C.canonical(dict(
                    schema="splitfusion.exact_gt_service.scorer_anchor_join.v1", provenance=slot.provenance,
                    join_mode=None, prediction_identity_as_received=None, join_identity=None))
        else:
            self._refuse("EVIDENCE_REQUIRES_DELIVERED_OR_UNSCORED")
        slot.files = files
        self._logical_bytes(slot)
        slot.state = "ARCHIVE_SUBMITTED_" + ("SCORED" if "numerical_score.json" in files else "UNSCORED")
        return dict(files)

    def archive_durable(self, identity_bytes, receipt, relative_prefix):
        self._check()
        slot = self._slot(identity_bytes)
        if slot.files is None:
            self._refuse("DURABILITY_PRECEDES_EVIDENCE")
        capture = C.thaw(slot.target.metadata["capture_identity"])
        C.require(type(receipt) is dict and receipt.get("status") == "COMPLETE" and receipt.get("durable") is True
                  and receipt.get("error") is None
                  and C.canonical(receipt.get("identity")) == C.canonical(dict(capture, evidence_role="scorer")),
                  "actual complete durable scorer receipt required")
        rows = {row["relative"]: row for row in receipt.get("written", [])}
        C.require(set(rows) == {f"{relative_prefix}/{name}" for name in slot.files}, "scorer receipt inventory differs")
        for name, raw in slot.files.items():
            row = rows[f"{relative_prefix}/{name}"]
            C.require(row.get("sha256") == hashlib.sha256(raw).hexdigest() and row.get("bytes") == len(raw),
                      "durable receipt does not bind exact scorer file: " + name)
        manifest = {name: hashlib.sha256(raw).hexdigest() for name, raw in slot.files.items()}
        scored = slot.state == "ARCHIVE_SUBMITTED_SCORED"
        proof = dict(schema=PROOF_SCHEMA, status="DURABLE", scored=scored,
                     ready_identity_sha256=hashlib.sha256(slot.key).hexdigest(),
                     ready_semantic_sha256=slot.target.semantic_sha256,
                     archive_manifest_sha256=hashlib.sha256(C.canonical(manifest)).hexdigest(),
                     receipt_sha256=hashlib.sha256(C.canonical(receipt)).hexdigest())
        self._terminal(slot.triple, "RETIRED_" + ("SCORED" if scored else "UNSCORED"))
        del self.slots[identity_bytes], self.index[slot.triple]
        slot.target = slot.prediction = slot.result = slot.files = slot.provenance = slot.join = None
        self.counters["retired"] += 1
        return proof

    # ------------------------------------------------------------------ lifecycle
    def snapshot(self):
        states = {}
        for slot in self.slots.values():
            states[slot.state] = states.get(slot.state, 0) + 1
        return dict(valid=self.failure is None, failure=self.failure, closed=self.closed,
                    retained_slots=len(self.slots), reserved_bytes=self.reserved(),
                    logical_bytes=sum(self._logical_bytes(s) for s in self.slots.values()),
                    waiting_predictions=len(self.waiting), terminal_ledger=len(self.terminal),
                    max_jobs=self.max_jobs, max_bytes=self.max_bytes, max_waiting=self.max_waiting,
                    states=states, counters=dict(self.counters, refused=dict(self.counters["refused"])))

    def close(self, *, abandon=False):
        if self.closed:
            return self.snapshot()
        if (self.slots or self.waiting) and not abandon:
            self._refuse("CLOSE_WITH_RETAINED_WORK")
        self.slots.clear(); self.index.clear(); self.waiting.clear()
        self.closed = True
        return self.snapshot()
