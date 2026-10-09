"""Producer-side numerical engine; transport, launch and fsync are external.

The engine owns actual integration-v2 transactions on one process/thread. Source
kwargs are local decoded snapshot values, never pickled fixture objects. Prepare
finishes every use of source-image/actor views before returning. Source numerics
and validated readiness precede send-confirmed publication. Legacy artifact
construction follows actual SCORE/feedback notifications and may contend with
the next GT computation on this worker; it is measured, not made free.

Legacy AVO artifact clocks are currently supported only as explicitly recorded
offline replay inputs. No live prescore-ready clocks are manufactured. The
separate runtime event clocks always describe observations in this invocation.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
import io
import json
import os
from pathlib import PurePosixPath
import re
import threading
import time
from types import MappingProxyType

import numpy as np

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_numeric_gt_integration_v2 import ready as R, composition as COM
from rl_agent.splitfusion_numeric_gt_integration_v2.ownership import ProducerOwner
from rl_agent.splitfusion_numeric_gt_producer_v1 import artifacts as AR, payload as P, source as NS
from rl_agent.splitfusion_live_parity_repair_v1 import target_transport as T
from rl_agent.splitfusion_run5_live_avo_v1 import gt_source_sync as SY


SCHEMA = "splitfusion.numeric_gt.producer_runtime.v1"
ARCHIVE_SCHEMA = "splitfusion.numeric_gt.producer_archive.v1"
LIVE_SOURCE_EXPORT_SCHEMA = "splitfusion.numeric_gt.live_source_export.v1"
LIVE_UNSCORED_TERMINAL_SCHEMA = "splitfusion.numeric_gt.live_unscored_terminal.v1"
MAX_LIVE_SOURCE_EXPORT_BYTES = C.MAX_METADATA_BYTES
RESERVATION = 64 * 1024 * 1024


class ProducerError(RuntimeError):
    pass


def _raw():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


def _require(value, message):
    if not value:
        raise ProducerError(message)


def _canonical(value):
    return P.A.canonical(P.json_value(value))


def _basename(value):
    _require(type(value) is str and bool(value) and value not in (".", "..")
        and PurePosixPath(value).name == value and "/" not in value and "\\" not in value
        and "\x00" not in value and len(value) <= 255, "safe artifact basename required")
    return value


def _npy(value):
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    return stream.getvalue()


@dataclass(frozen=True, slots=True)
class ProducerToken:
    sequence: int
    _issuer: object


@dataclass(frozen=True, slots=True)
class Prepared:
    token: ProducerToken
    identity_bytes: bytes
    semantic_sha256: str
    numerical_ready_raw_ns: int


@dataclass(frozen=True, slots=True)
class DeferredArchive:
    descriptor_bytes: bytes
    files: object
    identity_bytes: bytes
    semantic_sha256: str
    archive_sha256: str
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class LiveSourceExport:
    """Immutable finite-JSON evidence borrowed from one published owned frame.

    These are canonical JSON bytes, not historical AVO documents or a claim
    about scoring, feedback, or archival. The owner retains the numerical frame.
    """
    schema: str
    identity_bytes: bytes
    semantic_sha256: str
    input_metadata_bytes: bytes
    source_synchronization_bytes: bytes
    numerical_avo_bytes: bytes
    producer_events_bytes: bytes
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class LiveUnscoredTerminal:
    """Two immutable files awaiting an actual durable ArchiveClient receipt."""
    files: object
    identity_bytes: bytes
    semantic_sha256: str
    clock_descriptor_sha256: str
    terminal_sha256: str
    payload_bytes: int


@dataclass
class _Frame:
    token: ProducerToken
    owner_token: object
    plan: bytes
    prepared: Prepared
    clocks: dict
    filenames: dict
    prescore_filename: str
    rgb_bytes: bytes | None
    rgb_shape: tuple | None
    owner_payload_bytes: int
    metadata_bytes: int
    events: dict = field(default_factory=dict)
    committed: bool = False
    packet: bytes | None = None
    scored: bool = False
    feedback: bool = False
    archive: DeferredArchive | None = None
    live_unscored_terminal: LiveUnscoredTerminal | None = None
    scorer_archived: bool = False
    durable: bool = False
    consumer_failed: bool = False


class ProducerEngine:
    def __init__(self, *, producer_args, run_binding, producer_binding, prediction_contract,
                 max_frames, max_retained_jobs=2, max_retained_bytes=2 * RESERVATION,
                 reserved_bytes=RESERVATION, offline_replay=False):
        _require(type(offline_replay) is bool, "explicit offline replay boolean required")
        _require(C.is_digest(run_binding) and C.is_digest(producer_binding), "real run/producer bindings required")
        _require(type(max_frames) is int and 1 <= max_frames <= 1000, "bounded frame count required")
        _require(type(reserved_bytes) is int and 0 < reserved_bytes <= max_retained_bytes,
                 "bounded per-frame storage reservation required")
        contract = C._parse(C.canonical(prediction_contract))
        args = dict(producer_args)
        source_max_frames = args.get("max_frames")
        _require(type(source_max_frames) is int and 1 <= source_max_frames <= 1000,
                 "explicit registered producer max_frames in [1,1000] required")
        _require(max_frames <= source_max_frames, "runtime frame budget exceeds registered producer capacity")
        # This registered source limit is part of conflict evidence and therefore
        # of original prepared/provenance digests. A shorter runtime replay must
        # not rewrite the scientific history contract to its request count.
        producer = NS.NumericSourceProducer(**args)
        _require(producer.bound.binding_sha256 == producer_binding, "actual producer source binding differs")
        self._owner = ProducerOwner(producer, max_retained_jobs=max_retained_jobs,
                                    max_retained_bytes=max_retained_bytes)
        self._thread = threading.current_thread()
        self._issuer = object()
        self._run_binding, self._producer_binding = run_binding, producer_binding
        self._prediction_contract = contract
        self._reservation, self._max_frames = reserved_bytes, max_frames
        self._source_max_frames = source_max_frames
        self._offline = offline_replay
        self._frames = {}
        self._completed_events = {}
        self._issued = self._retired = 0
        self._failure = None
        self._closed = False

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, kind, error, traceback):
        if error is not None:
            self._failure = self._failure or f"engine scope failed: {error}"[:1024]
        self.close(abandon=error is not None)

    def _check(self, *, healthy=True):
        if threading.current_thread() is not self._thread:
            self._failure = self._failure or "producer engine used from another thread"
            raise ProducerError(self._failure)
        _require(not self._closed, "producer engine is closed")
        if healthy:
            _require(self._failure is None, self._failure or "producer engine failed")

    def _fail(self, message):
        self._failure = self._failure or str(message)[:1024]
        raise ProducerError(str(message))

    def _row(self, token, *, healthy=True):
        self._check(healthy=healthy)
        if (type(token) is not ProducerToken or token._issuer is not self._issuer
                or self._frames.get(token.sequence) is None
                or self._frames[token.sequence].token is not token):
            self._fail("foreign or retired producer-engine token")
        return self._frames[token.sequence]

    @contextmanager
    def _observed_clock(self, value):
        if value is None:
            yield
            return
        _require(self._offline and type(value) is int and value > 0,
                 "borrowed source clock requires explicit offline replay")
        # The process-wide ProducerOwner lease serializes this temporary seam.
        original = SY.source_alignment_record
        def alignment(*args, **kwargs):
            record = original(*args, **kwargs)
            record["observed_raw_ns"] = value
            return record
        SY.source_alignment_record = alignment
        try:
            yield
        finally:
            SY.source_alignment_record = original

    def prepare(self, plan_identity_bytes, sourcekwargs, *, frame_context, population,
                archive_clocks=None, payload_filenames=None, prescore_filename=None,
                replay_observed_raw_ns=None):
        self._check()
        owner_token = None
        before_bytes = self._owner.snapshot()["payload_bytes"]
        begin = _raw()
        try:
            _require(self._issued < self._max_frames, "runtime frame budget exhausted")
            if not self._offline:
                _require(archive_clocks is None and replay_observed_raw_ns is None,
                         "recorded legacy clocks require explicit offline replay")
            if archive_clocks is not None:
                _require(type(archive_clocks) is dict and set(archive_clocks) == set(AR.AVO_CLOCKS)
                    and all(type(x) is int and x > 0 for x in archive_clocks.values()),
                    "seven actual recorded AVO clocks required")
            clocks = None if archive_clocks is None else dict(archive_clocks)
            names = None if payload_filenames is None else dict(payload_filenames)
            if names is not None:
                _require(set(names) == {"depth_bgra", "rawsemantic"}, "exact AVO native filenames required")
                for name in names.values():
                    _basename(name)
                _basename(prescore_filename)
                _require(len(set((*names.values(), prescore_filename))) == 3, "duplicate AVO artifact filenames")
            else:
                _require(prescore_filename is None, "AVO filename declarations must be complete")
            plan = C._parse(plan_identity_bytes)
            _require(C.canonical(population) == C.canonical(sourcekwargs["dynamic_snapshot_evidence"]),
                     "population differs from owned source kwargs")
            ids, digest = population["registered_population_ids"], population["population_sha256"]
            _require(type(ids) in (list, tuple) and all(type(i) is int for i in ids)
                and len(set(ids)) == len(ids) and C.is_digest(digest), "complete registered population required")
            if self._owner.population != frozenset(ids):
                self._owner.register_population(ids, digest, before_frame_id=sourcekwargs["identity"]["frame_id"])
            rgb = sourcekwargs["rgb"].get("_rgb_image")
            rgb_bytes = rgb_shape = None
            if rgb is not None:
                rgb_shape = (int(rgb.height), int(rgb.width), 4)
                _require(rgb_shape == (720, 1280, 4) and type(rgb.frame) is int
                    and rgb.frame == sourcekwargs["identity"]["frame_id"]
                    and float(rgb.timestamp) == float(sourcekwargs["rgb"]["timestamp"]),
                    "native RGB source identity/geometry differs")
                _require(isinstance(rgb.raw_data, (bytes, bytearray, memoryview))
                    and len(rgb.raw_data) == int(np.prod(rgb_shape)), "complete native RGB byte buffer required")
                _require(np.array_equal(np.asarray(rgb.transform.get_matrix()),
                    np.asarray(sourcekwargs["camera_matrix"])), "native RGB camera pose differs")
                rgb_bytes = bytes(rgb.raw_data)
            with self._observed_clock(replay_observed_raw_ns):
                owner_token = self._owner.compute(plan_identity_bytes, sourcekwargs, reserved_bytes=self._reservation)
            staged = self._owner._staged_for_assembly(owner_token)
            numeric, avo, payload = COM.assemble(staged, sourcekwargs)
            self._owner.stage_products(owner_token, target=numeric, avo=avo, payload=payload)
            target = R.admit_payload(payload, expected_capture_identity=sourcekwargs["identity"],
                expected_action_identity=plan["action"], frame_context=frame_context,
                plan_sha256=C.sha256(plan_identity_bytes), run_binding=self._run_binding,
                producer_binding=self._producer_binding, prediction_contract=self._prediction_contract)
            _require(self._owner.set_ready(owner_token, target) is None, "prepare cannot commit without send confirmation")
            self._issued += 1
            token = ProducerToken(self._issued, self._issuer)
            observed = _raw()
            prepared = Prepared(token, target.identity_bytes, target.semantic_sha256, observed)
            metadata_size = len(_canonical(dict(clocks=clocks, filenames=names, prescore_filename=prescore_filename)))
            row = _Frame(token, owner_token, plan_identity_bytes, prepared, clocks, names,
                prescore_filename, rgb_bytes, rgb_shape,
                self._owner.snapshot()["payload_bytes"] - before_bytes, metadata_size,
                events=dict(prepare_start_raw_ns=begin, numerical_ready_raw_ns=observed))
            self._budget(row)
            self._frames[token.sequence] = row
            return prepared
        except BaseException as error:
            self._failure = self._failure or f"prepare failed: {type(error).__name__}: {error}"[:1024]
            if owner_token is not None:
                try:
                    self._owner.cancel(owner_token, "runtime preparation failed before send")
                except BaseException as cleanup_error:
                    raise error from cleanup_error
            raise

    def _payload_bytes(self, row):
        actual = row.owner_payload_bytes + row.metadata_bytes + len(row.rgb_bytes or b"") + len(row.packet or b"")
        if row.archive is not None:
            actual += row.archive.payload_bytes
        if row.live_unscored_terminal is not None:
            actual += row.live_unscored_terminal.payload_bytes
        return actual

    def _budget(self, row):
        actual = self._payload_bytes(row)
        _require(actual <= self._reservation, "runtime owned payload exceeds frame reservation")
        return actual

    def confirm_send(self, token, plan_identity_bytes):
        row = self._row(token)
        if type(plan_identity_bytes) is not bytes or plan_identity_bytes != row.plan:
            self._fail("send confirmation differs from exact full plan")
        try:
            result = self._owner.confirm_send(row.owner_token, plan_identity_bytes)
            _require(result is not None, "real producer commit required")
            row.committed = True
            row.events.setdefault("send_confirmed_raw_ns", _raw())
            return result
        except BaseException as error:
            self._failure = self._failure or f"send commit failed: {error}"[:1024]
            raise

    def cancel(self, token, plan_identity_bytes, reason):
        row = self._row(token, healthy=False)
        if type(plan_identity_bytes) is not bytes or plan_identity_bytes != row.plan:
            self._fail("cancellation differs from exact full plan")
        try:
            result = self._owner.cancel(row.owner_token, reason)
            self._retire(row, "CANCELLED")
            return result
        except BaseException as error:
            self._failure = self._failure or f"producer cancellation failed: {error}"[:1024]
            raise

    def publish_ready(self, token):
        row = self._row(token)
        if not row.committed:
            self._fail("ready publication precedes confirmed-send history commit")
        if row.packet is None:
            row.packet = R.encode_target(self._owner.ready_target(row.owner_token))
            try:
                self._budget(row)
            except BaseException as error:
                self._failure = self._failure or f"ready packet retention failed: {error}"[:1024]
                raise
            row.events["ready_published_raw_ns"] = _raw()
        return row.packet

    def export_live_source(self, token, identity_bytes, semantic_sha256):
        """Copy only bounded numerical source primitives after committed publication.

        This is a read-only, idempotent same-thread borrow. It deliberately does
        not build PNGs, historical AVO documents, or a terminal outcome.
        """
        self._check()
        _require(not self._offline, "live source export requires observed live source clocks")
        row = self._notification(token, identity_bytes, semantic_sha256)
        _require(row.committed and row.packet is not None,
                 "live source export precedes committed target publication")
        staged, _, avo, _ = self._owner._archive_for_owner(row.owner_token)
        target = self._owner.ready_target(row.owner_token)
        _require(target.identity_bytes == row.prepared.identity_bytes
                 and target.semantic_sha256 == row.prepared.semantic_sha256,
                 "owned ready target differs from published frame")
        metadata = C.thaw(target.metadata)
        capture = C._parse(identity_bytes)["capture_identity"]
        input_bytes = _canonical(T._input_metadata(staged.inputs))
        sync_bytes = _canonical(staged.sync)
        avo_bytes = _canonical(avo.avo)
        _require(all(len(raw) <= C.MAX_METADATA_BYTES for raw in
                     (input_bytes, sync_bytes, avo_bytes))
                 and len(identity_bytes) + len(input_bytes) + len(sync_bytes) + len(avo_bytes)
                     <= MAX_LIVE_SOURCE_EXPORT_BYTES,
                 "bounded live source export size exceeded")
        inputs, sync, numerical_avo = map(C._parse, (input_bytes, sync_bytes, avo_bytes))
        T._validate_metadata(inputs, capture)
        _require(C.canonical(inputs["frame"]) == C.canonical(metadata["frame"])
                 and C.canonical(inputs["provenance"]) == C.canonical(metadata["provenance"])
                 and inputs["provenance"]["input_content_sha256"] == metadata["input_content_sha256"],
                 "complete source metadata differs from published target")
        C._avo(numerical_avo, capture["frame_id"])
        _require(C.canonical(numerical_avo) == C.canonical(metadata["avo"]),
                 "numerical AVO differs from published target")
        _require(sync.get("schema") == SY.SYNC_SCHEMA and sync.get("identity") == capture,
                 "source synchronization differs from published capture")
        SY.validate_source_alignment(sync)
        _require(type(sync.get("observed_raw_ns")) is int
                 and 0 < sync["observed_raw_ns"] < 2**63,
                 "actual source alignment RAW observation required")
        rgb = sync["sources"]["rgb"]
        _require(rgb["timestamp"] == metadata["frame"]["timestamp"]
                 and (rgb["height"], rgb["width"]) == (720, 1280),
                 "synchronized RGB timestamp/geometry differs from target")
        camera = np.asarray(rgb["camera_matrix"], dtype=np.float64)
        target_camera = np.asarray(metadata["camera_matrix"], dtype=np.float64)
        _require(camera.shape == target_camera.shape == (4, 4)
                 and np.isfinite(camera).all() and np.isfinite(target_camera).all()
                 and np.allclose(camera, target_camera, rtol=0, atol=SY.POSE_ATOL),
                 "synchronized RGB pose differs from published target")
        from .live_clock import CLOCK_DOMAIN, read_boot_id
        stamp_names = (("prepare_start", row.events["prepare_start_raw_ns"]),
                       ("source_alignment_observed", sync["observed_raw_ns"]),
                       ("numeric_ready", row.events["numerical_ready_raw_ns"]),
                       ("ready_published", row.events["ready_published_raw_ns"]))
        _require(all(type(stamp) is int and 0 < stamp < 2**63 for _, stamp in stamp_names)
                 and all(a[1] <= b[1] for a, b in zip(stamp_names, stamp_names[1:])),
                 "producer-owned RAW event order differs")
        boot_id, pid = read_boot_id(), os.getpid()
        event_bytes = _canonical(dict(schema="splitfusion.numeric_gt.producer_live_events.v1",
            clock_domain=CLOCK_DOMAIN, ready_identity=C._parse(identity_bytes),
            ready_identity_sha256=C.sha256(identity_bytes), semantic_sha256=semantic_sha256,
            role="producer", pid=pid, boot_id=boot_id,
            events=[dict(name=name, raw_ns=stamp) for name, stamp in stamp_names]))
        _require(len(identity_bytes) + len(input_bytes) + len(sync_bytes) + len(avo_bytes)
                 + len(event_bytes) <= MAX_LIVE_SOURCE_EXPORT_BYTES,
                 "bounded live source export size exceeded")
        return LiveSourceExport(LIVE_SOURCE_EXPORT_SCHEMA, bytes(identity_bytes), semantic_sha256,
            input_bytes, sync_bytes, avo_bytes, event_bytes,
            len(identity_bytes) + len(input_bytes) + len(sync_bytes) + len(avo_bytes)
                + len(event_bytes))

    def _notification(self, token, identity_bytes, semantic_sha256):
        row = self._row(token, healthy=False)
        if (type(identity_bytes) is not bytes or identity_bytes != row.prepared.identity_bytes
                or semantic_sha256 != row.prepared.semantic_sha256 or row.packet is None
                or row.consumer_failed):
            self._fail("foreign notification or notification before committed publication")
        return row

    def score_completed(self, token, identity_bytes, semantic_sha256):
        row = self._notification(token, identity_bytes, semantic_sha256)
        if not row.scored:
            self._owner.complete_scoring(row.owner_token)
            row.scored = True
            row.events["score_completed_raw_ns"] = _raw()

    def feedback_confirmed(self, token, identity_bytes, semantic_sha256):
        row = self._notification(token, identity_bytes, semantic_sha256)
        if not row.scored:
            self._fail("feedback completion precedes actual score completion")
        if not row.feedback:
            self._owner.confirm_feedback(row.owner_token)
            row.feedback = True
            row.events["feedback_confirmed_raw_ns"] = _raw()

    def complete_live_scored(self, token, identity_bytes, semantic_sha256, scorer_proof):
        """Retire an actually scored frame only after scorer's durable live proof.

        The coordinator must deliver this proof from the authenticated scorer
        completion reply. It is not a call to the historical feedback-confirmed
        path and never asserts that an on-time radio reply happened by itself.
        """
        row = self._notification(token, identity_bytes, semantic_sha256)
        _require(not self._offline and row.committed and row.scored and not row.feedback
                 and row.archive is None and row.live_unscored_terminal is None,
                 "live scored retirement requires scored published frame, no historical terminal")
        proof = C._json_owned(scorer_proof)
        _require(type(proof) is dict and set(proof) == {
            "status", "terminal", "archive_manifest_sha256", "sidecar_sha256",
            "ready_identity_sha256", "ready_semantic_sha256", "live_qualified"}
            and proof["status"] == "DURABLE"
            and proof["terminal"] in ("FEEDBACK_ON_TIME", "FEEDBACK_TIMEOUT")
            and proof["ready_identity_sha256"] == C.sha256(identity_bytes)
            and proof["ready_semantic_sha256"] == semantic_sha256
            and C.is_digest(proof["archive_manifest_sha256"])
            and C.is_digest(proof["sidecar_sha256"])
            and proof["live_qualified"] is False,
            "scorer durable proof differs from this scored frame")
        # The real scorer has already verified archive bytes, receipt and
        # completion-sidecar readback before producing this authenticated reply.
        self._owner.resolve_live_scored_terminal(row.owner_token, terminal=proof["terminal"])
        row.events["live_terminal"] = proof["terminal"]
        row.events["live_archive_manifest_sha256"] = proof["archive_manifest_sha256"]
        row.events["live_durable_sidecar_sha256"] = proof["sidecar_sha256"]
        row.events["live_proof_observed_raw_ns"] = _raw()
        self._retire(row, "LIVE_SCORED_DURABLE")
        return dict(status="RETIRED", terminal=proof["terminal"],
                    ready_identity_sha256=C.sha256(identity_bytes),
                    semantic_sha256=semantic_sha256,
                    archive_manifest_sha256=proof["archive_manifest_sha256"],
                    durable_sidecar_sha256=proof["sidecar_sha256"],
                    live_qualified=False)

    def prepare_live_unscored_terminal(self, token, identity_bytes, semantic_sha256,
                                       clock_descriptor_bytes, reason,
                                       fault_feedback_packet_bytes=None,
                                       ue_received_packet_sha256=None):
        """Create compact no-score evidence without pretending a score exists.

        The returned immutable files still require a real durable receipt. A
        missing score is an ordinary terminal outcome, not producer corruption.
        """
        row = self._notification(token, identity_bytes, semantic_sha256)
        _require(not self._offline and row.committed and not row.scored and not row.feedback
                 and row.archive is None, "unscored live terminal requires published no-score frame")
        _require(type(reason) is str and re.fullmatch(r"[A-Z][A-Z0-9_]{2,63}", reason) is not None,
                 "bounded unscored terminal reason required")
        _require(type(clock_descriptor_bytes) is bytes
                 and 0 < len(clock_descriptor_bytes) <= C.MAX_METADATA_BYTES,
                 "bounded observed live clock descriptor required")
        if row.live_unscored_terminal is not None:
            prior = row.live_unscored_terminal
            _require(prior.files["live_clock.json"] == clock_descriptor_bytes
                     and C._parse(prior.files["unscored_terminal.json"])["reason"] == reason
                     and C._parse(prior.files["unscored_terminal.json"]).get("fault_feedback_packet_sha256")
                         == (None if fault_feedback_packet_bytes is None else C.sha256(fault_feedback_packet_bytes))
                     and C._parse(prior.files["unscored_terminal.json"]).get("ue_received_packet_sha256")
                         == ue_received_packet_sha256,
                     "unscored terminal is create-only for its exact evidence")
            return prior
        from .live_clock import (ASYNC_HANDOFF_ID, ASYNC_SCHEMA,
                                 LiveClockLedger, SCHEMA as CLOCK_SCHEMA)
        clock = C._parse(clock_descriptor_bytes)
        handoff_id = clock.get("handoff_id")
        _require((clock.get("schema"), handoff_id) in (
                     (CLOCK_SCHEMA, None), (ASYNC_SCHEMA, ASYNC_HANDOFF_ID)),
                 "unknown or crossed live clock handoff version")
        _require(C.canonical(clock) == clock_descriptor_bytes
                 and clock.get("ready_identity") == C._parse(identity_bytes)
                 and clock.get("ready_identity_sha256") == C.sha256(identity_bytes)
                 and clock.get("semantic_sha256") == semantic_sha256
                 and clock.get("scoring_state") in
                     ("NOT_SCORED_AT_TIMEOUT", "UNSCORED_FAULT_FEEDBACK_EMITTED"),
                 "unscored clock descriptor differs from published frame or invents scoring")
        fault = clock.get("unscored_fault")
        if clock["scoring_state"] == "UNSCORED_FAULT_FEEDBACK_EMITTED":
            from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
            _require(type(fault) is dict and type(fault_feedback_packet_bytes) is bytes
                     and 0 < len(fault_feedback_packet_bytes) <= 1024
                     and fault.get("feedback_packet_sha256") == C.sha256(fault_feedback_packet_bytes)
                     and fault.get("reason") is not None
                     and fault.get("score_exists") is False
                     and fault.get("quality_value") is None,
                     "unscored fault needs exact bounded emitted packet and reason")
            try:
                feedback, decoded_reason = W.decode_feedback(fault_feedback_packet_bytes)
            except (W.WireError, ValueError, TypeError) as exc:
                raise ProducerError("unscored fault packet is not a valid Run-4 feedback") from exc
            action = clock["ready_identity"]["action_identity"]
            capture = clock["ready_identity"]["capture_identity"]
            for name in ("session_uuid", "controller_lineage_sha256", "decision_seq",
                         "ticket_seq", "frame_id", "tensor_seq", "mode_id", "q_e4",
                         "execution_bundle_sha256", "anchor_action_id", "reward_requested"):
                _require(getattr(feedback, name) == action[name],
                         "unscored fault packet belongs to another action: " + name)
            _require(feedback.capture_timestamp_ns == capture["capture_timestamp_ns"]
                     and feedback.frame_id == capture["frame_id"]
                     and feedback.kind in ("EVALUATOR_FAULT", "INFRASTRUCTURE_FAULT")
                     and feedback.q_perc is None and decoded_reason.name == fault["reason"],
                     "unscored fault packet lacks matching frame, kind, reason or absent quality")
            receipt_seen = (clock["deadline"]["feedback_received_raw_ns"] is not None
                            or clock["deadline"]["late_feedback_observed_raw_ns"] is not None)
            _require((ue_received_packet_sha256 == C.sha256(fault_feedback_packet_bytes))
                     if receipt_seen else ue_received_packet_sha256 is None,
                     "UE-observed fault packet digest differs from emitted packet")
        else:
            _require(fault is None and fault_feedback_packet_bytes is None
                     and ue_received_packet_sha256 is None,
                     "ordinary no-score timeout cannot claim emitted fault feedback")
        _require(type(clock.get("events")) is list and 1 <= len(clock["events"]) <= 32,
                 "bounded observed unscored event ledger required")
        replay = LiveClockLedger(identity_bytes, semantic_sha256,
            handoff_id=handoff_id)
        for event in clock["events"]:
            _require(type(event) is dict, "complete observed event required")
            observed = replay.record(role=event.get("role"), name=event.get("name"),
                raw_ns=event.get("raw_ns"), pid=event.get("pid"), boot_id=event.get("boot_id"),
                ready_identity_bytes=identity_bytes, semantic_sha256=semantic_sha256)
            _require(C.canonical(observed) == C.canonical(event),
                     "unscored event differs from exact owner-stamped identity")
        _require(replay.archive_descriptor(avo_sha256=clock.get("avo_sha256"),
                 source_synchronization_sha256=clock.get("source_synchronization_sha256"),
                 fault_feedback_sha256=None if fault is None else fault["feedback_packet_sha256"],
                 fault_reason=None if fault is None else fault["reason"])
                 == clock_descriptor_bytes,
                 "unscored descriptor cannot be reproduced from actual events")
        _require(fault is not None or replay.deadline()["terminal"] == "FEEDBACK_TIMEOUT",
                 "ordinary unscored retirement needs observed UE timeout")
        staged, _, avo, _ = self._owner._archive_for_owner(row.owner_token)
        _require(clock["source_synchronization_sha256"] == C.sha256(_canonical(staged.sync))
                 and clock["avo_sha256"] == C.sha256(_canonical(avo.avo)),
                 "unscored clock differs from exact numerical source/AVO")
        stamps = {event["name"]: event["raw_ns"] for event in clock["events"]
                  if event["role"] == "producer"}
        _require(stamps["prepare_start"] == row.events["prepare_start_raw_ns"]
                 and stamps["source_alignment_observed"] == staged.sync["observed_raw_ns"]
                 and stamps["numeric_ready"] == row.events["numerical_ready_raw_ns"]
                 and stamps["ready_published"] == row.events["ready_published_raw_ns"],
                 "unscored clock producer stamps differ from this source frame")
        target = self._owner.ready_target(row.owner_token)
        provenance_sha = C.sha256(C.canonical(target.metadata["provenance"]))
        terminal = dict(schema=LIVE_UNSCORED_TERMINAL_SCHEMA,
            ready_identity=C._parse(identity_bytes), ready_identity_sha256=C.sha256(identity_bytes),
            ready_semantic_sha256=semantic_sha256,
            source_provenance_sha256=provenance_sha,
            input_content_sha256=target.metadata["input_content_sha256"],
            numerical_avo_sha256=clock["avo_sha256"],
            source_synchronization_sha256=clock["source_synchronization_sha256"],
            clock_descriptor_sha256=C.sha256(clock_descriptor_bytes),
            deadline=clock["deadline"], terminal=clock["deadline"]["terminal"], reason=reason,
            score_exists=False,
            feedback_receipt_claim=clock["deadline"]["feedback_received_raw_ns"] is not None,
            quality_feedback_claim=False,
            fault_feedback_packet_sha256=None if fault is None else C.sha256(fault_feedback_packet_bytes),
            fault_feedback_kind=None if fault is None else feedback.kind,
            fault_feedback_reason=None if fault is None else fault["reason"],
            ue_received_packet_sha256=ue_received_packet_sha256,
            historical_avo_document=False, old_prepared_digest_claim=False)
        terminal_bytes = C.canonical(terminal)
        _require(len(terminal_bytes) + len(clock_descriptor_bytes) <= C.MAX_METADATA_BYTES,
                 "bounded unscored terminal files required")
        files = MappingProxyType({"live_clock.json": bytes(clock_descriptor_bytes),
                                  "unscored_terminal.json": terminal_bytes})
        result = LiveUnscoredTerminal(files, bytes(identity_bytes), semantic_sha256,
            C.sha256(clock_descriptor_bytes), C.sha256(terminal_bytes),
            len(clock_descriptor_bytes) + len(terminal_bytes))
        _require(self._payload_bytes(row) + result.payload_bytes <= self._reservation,
                 "runtime owned payload exceeds frame reservation")
        row.live_unscored_terminal = result
        row.events["live_unscored_terminal_prepared_raw_ns"] = _raw()
        return result

    def complete_live_unscored(self, token, identity_bytes, semantic_sha256,
                               terminal_sha256, receipt, relative_prefix):
        """Retire ordinary no-score timeout after exact durable terminal files."""
        row = self._notification(token, identity_bytes, semantic_sha256)
        _require(not self._offline and row.committed and not row.scored and not row.feedback
                 and row.archive is None and type(row.live_unscored_terminal) is LiveUnscoredTerminal
                 and terminal_sha256 == row.live_unscored_terminal.terminal_sha256,
                 "unscored completion requires retained terminal evidence")
        terminal = row.live_unscored_terminal
        prefix = PurePosixPath(relative_prefix) if type(relative_prefix) is str else None
        _require(prefix is not None and relative_prefix and not prefix.is_absolute()
                 and str(prefix) == relative_prefix
                 and all(part not in (".", "..") for part in prefix.parts),
                 "safe unscored terminal archive prefix required")
        value = C._json_owned(receipt)
        capture = C._parse(identity_bytes)["capture_identity"]
        _require(type(value) is dict and value.get("status") == "COMPLETE"
                 and value.get("durable") is True and value.get("error") is None
                 and type(value.get("job_id")) is int and value["job_id"] > 0
                 and value.get("identity") == capture,
                 "actual durable unscored terminal receipt required")
        written = value.get("written")
        _require(type(written) is list and 1 <= len(written) <= 256
                 and len(C.canonical(value)) <= C.MAX_METADATA_BYTES,
                 "bounded unscored terminal inventory required")
        names = {str(prefix / name) for name in terminal.files}
        selected = {}
        for item in written:
            _require(type(item) is dict and type(item.get("relative")) is str,
                     "malformed durable terminal inventory row")
            name = item["relative"]
            if name.startswith(relative_prefix + "/"):
                _require(set(item) == {"root", "relative", "sha256", "bytes", "ready"}
                         and name in names and name not in selected and type(item.get("root")) is int
                         and item["root"] >= 0 and item.get("ready") is False,
                         "foreign or duplicate terminal file in receipt")
                selected[name] = item
        _require(set(selected) == names and len({item["root"] for item in selected.values()}) == 1,
                 "durable receipt misses exact terminal files or spans roots")
        for name, raw in terminal.files.items():
            item = selected[str(prefix / name)]
            _require(item.get("sha256") == C.sha256(raw)
                     and type(item.get("bytes")) is int and item["bytes"] == len(raw),
                     "durable unscored terminal bytes/hash differ: " + name)
        terminal_kind = C._parse(terminal.files["unscored_terminal.json"])["terminal"]
        self._owner.resolve_live_unscored_terminal(row.owner_token, terminal=terminal_kind)
        row.events["live_terminal"] = terminal_kind
        row.events["live_unscored_terminal_sha256"] = terminal.terminal_sha256
        row.events["live_receipt_sha256"] = C.sha256(C.canonical(value))
        row.events["live_proof_observed_raw_ns"] = _raw()
        self._retire(row, "LIVE_UNSCORED_TIMEOUT_DURABLE" if terminal_kind == "FEEDBACK_TIMEOUT"
                     else "LIVE_UNSCORED_FAULT_DURABLE")
        return dict(status="RETIRED", terminal=terminal_kind,
                    ready_identity_sha256=C.sha256(identity_bytes),
                    semantic_sha256=semantic_sha256,
                    unscored_terminal_sha256=terminal.terminal_sha256,
                    archive_receipt_sha256=C.sha256(C.canonical(value)),
                    score_exists=False,
                    feedback_receipt_claim=terminal_kind == "FEEDBACK_ON_TIME",
                    live_qualified=False)

    def build_archive(self, token):
        row = self._row(token)
        if not (row.scored and row.feedback):
            self._fail("deferred artifacts require score and feedback completion")
        if row.archive is not None:
            return row.archive
        try:
            _require(self._offline and row.clocks is not None and row.filenames is not None,
                "live legacy archive-clock schema is not implemented; explicit recorded offline clocks required")
            row.events["archive_build_start_raw_ns"] = _raw()
            staged, numeric, avo, payload = self._owner._archive_for_owner(row.owner_token)
            legacy = AR.legacy_target(staged, numeric)
            prepared = AR.prepared_target(staged, legacy)
            documents = AR.avo_documents(staged, avo, identity=staged.identity, clocks=row.clocks,
                payload_filenames=row.filenames, prescore_filename=row.prescore_filename)
            metadata = json.loads(_canonical(T._input_metadata(staged.inputs)))
            legacy_metadata = {k: v for k, v in legacy.items() if k not in (
                "segmentation", "object_ignore", "segmentation_png", "object_ignore_png", "_csv_bytes")}
            files = {
                "inputs.json": _canonical(metadata), "target.json": _canonical(legacy_metadata),
                "segmentation.png": legacy["segmentation_png"], "object_ignore.png": legacy["object_ignore_png"],
                "depth_bgra.npy": _npy(staged.depth_raw), "raw_semantic.npy": _npy(staged.semantic),
                "original_v3_mask.npy": _npy(staged.mask), "radar_world_xyz.npy": _npy(staged.radar),
                "SOURCE_EVIDENCE.json": _canonical(staged.evidence),
                "avo.json": documents["document_bytes"],
                "producer_binding.json": _canonical(AR.binding(payload, legacy, prepared,
                    avo_document_sha256=documents["document_sha256"])),
                **legacy["_csv_bytes"],
            }
            # Original AVO document filenames are preserved, with exact original
            # np.save bytes. Collision may only reuse identical existing bytes.
            for name, raw in ((row.filenames["depth_bgra"], files["depth_bgra.npy"]),
                    (row.filenames["rawsemantic"], files["raw_semantic.npy"]),
                    (row.prescore_filename, documents["prescore_bytes"])):
                _require(name not in files or files[name] == raw, "conflicting artifact filename")
                files[name] = raw
            if row.rgb_bytes is not None:
                files["rgb_bgra.npy"] = _npy(np.frombuffer(row.rgb_bytes, np.uint8).reshape(row.rgb_shape))
            inventory = {name: dict(bytes=len(raw), sha256=C.sha256(raw)) for name, raw in files.items()}
            descriptor = dict(schema=ARCHIVE_SCHEMA, ready_identity=C._parse(row.prepared.identity_bytes),
                ready_semantic_sha256=row.prepared.semantic_sha256, input_metadata=metadata,
                avo_document=documents["document"], legacy_target_meta=legacy_metadata,
                old_prepared_sha256=prepared.content_sha256, prepared_frame_id=prepared.frame_id,
                prepared_sample_id=prepared.sample_id, files=inventory,
                clock_semantics="OFFLINE_RECORDED_CLOCKS_NO_LIVE_CLAIM", offline_replay=True,
                rgb_source_present=row.rgb_bytes is not None)
            encoded = _canonical(descriptor)
            _require(len(encoded) <= C.MAX_METADATA_BYTES, "archive descriptor bound exceeded")
            _require("producer_descriptor.json" not in files, "reserved archive descriptor filename")
            files["producer_descriptor.json"] = encoded
            archive = DeferredArchive(encoded, MappingProxyType(files), row.prepared.identity_bytes,
                row.prepared.semantic_sha256, C.sha256(encoded), sum(map(len, files.values())))
            row.archive = archive
            self._budget(row)
            row.events["archive_build_end_raw_ns"] = _raw()
            return archive
        except BaseException as error:
            self._failure = self._failure or f"deferred artifact construction failed: {error}"[:1024]
            raise

    def scorer_archive_completed(self, token, identity_bytes, semantic_sha256):
        row = self._notification(token, identity_bytes, semantic_sha256)
        if row.archive is None or not row.feedback:
            self._fail("scorer archival completion precedes producer artifact construction")
        row.scorer_archived = True
        row.events.setdefault("scorer_archive_completed_raw_ns", _raw())
        self._release_if_done(row)

    def archive_durable(self, token, receipt, *, relative_prefix):
        row = self._row(token, healthy=False)
        try:
            _require(row.archive is not None, "durability precedes producer artifacts")
            _require(not row.consumer_failed, "durability cannot revive a failed consumer")
            prefix = PurePosixPath(relative_prefix)
            _require(type(relative_prefix) is str and relative_prefix and not prefix.is_absolute()
                and str(prefix) == relative_prefix and all(x not in (".", "..") for x in prefix.parts),
                "exact safe relative archive prefix required")
            capture = C._parse(row.prepared.identity_bytes)["capture_identity"]
            _require(type(receipt) is dict and receipt.get("status") == "COMPLETE"
                and receipt.get("durable") is True and receipt.get("error") is None
                and C.canonical(receipt.get("identity")) == C.canonical(capture),
                "actual complete durable archive receipt for this capture required")
            written = receipt.get("written")
            _require(type(written) is list and len(written) <= 256, "bounded archive file completion inventory required")
            rows = {}
            for item in written:
                _require(type(item) is dict and type(item.get("relative")) is str
                    and type(item.get("root")) is int and item["root"] >= 0
                    and item["relative"] not in rows, "duplicate/malformed completed artifact path")
                rows[item["relative"]] = item
            _require({name for name in rows if name.startswith(relative_prefix + "/")} ==
                {str(prefix / name) for name in row.archive.files}, "producer receipt inventory differs")
            roots = set()
            for name, raw in row.archive.files.items():
                item = rows.get(str(prefix / name))
                _require(item is not None and item.get("sha256") == C.sha256(raw)
                    and type(item.get("bytes")) is int and item["bytes"] == len(raw),
                    "durable receipt does not bind exact producer file: " + name)
                roots.add(item["root"])
            _require(len(roots) == 1, "producer archive files span differing roots")
            row.durable = True
            row.events.setdefault("archive_durable_observed_raw_ns", _raw())
            self._release_if_done(row)
        except BaseException as error:
            self._failure = self._failure or f"archive receipt rejected: {error}"[:1024]
            raise

    def _release_if_done(self, row):
        if row.scorer_archived and row.durable:
            self._owner.complete_archive(row.owner_token)
            self._retire(row, "DURABLE")

    def _retire(self, row, outcome):
        row.events["retired_raw_ns"] = _raw()
        events = dict(row.events, outcome=outcome,
            frame_id=C._parse(row.prepared.identity_bytes)["capture_identity"]["frame_id"])
        _require(len(C.canonical(events)) <= 4096 and len(self._completed_events) < self._max_frames,
                 "bounded completed runtime event ledger exceeded")
        self._completed_events[str(row.token.sequence)] = events
        self._frames.pop(row.token.sequence)
        self._retired += 1
        row.packet = row.rgb_bytes = row.archive = row.live_unscored_terminal = None

    def fail(self, token, *, consumer, reason):
        row = self._row(token, healthy=False)
        _require(not row.consumer_failed, "consumer failure already recorded")
        self._failure = self._failure or f"{consumer} failed: {reason}"[:1024]
        if row.committed:
            self._owner.consumer_failed(row.owner_token, consumer=consumer, reason=reason)
            row.consumer_failed = True
            row.events["consumer_failed_raw_ns"] = _raw()
        else:
            self._owner.cancel(row.owner_token, reason)
            self._retire(row, "CANCELLED_AFTER_FAILURE")

    def snapshot(self):
        _require(threading.current_thread() is self._thread, "snapshot belongs to producer thread")
        return dict(schema=SCHEMA, valid=self._failure is None and self._owner.snapshot()["valid"],
            failure=self._failure, closed=self._closed, issued=self._issued, retired=self._retired,
            runtime_max_frames=self._max_frames, producer_max_frames=self._source_max_frames,
            offline_replay=self._offline, retained_jobs=len(self._frames),
            retained_payload_bytes=sum(self._payload_bytes(r) for r in self._frames.values()),
            completed_event_bytes=len(C.canonical(self._completed_events)),
            max_completed_event_bytes=4096 * self._max_frames + 16384,
            owner=self._owner.snapshot(), events={**{k: dict(v) for k, v in self._completed_events.items()},
                **{str(r.token.sequence): dict(r.events) for r in self._frames.values()}},
            transport_owned_here=False, fsync_owned_here=False, live_archive_clock_schema_implemented=False)

    def close(self, *, abandon=False):
        _require(threading.current_thread() is self._thread, "close belongs to producer thread")
        if self._closed:
            return self.snapshot()
        if abandon and self._frames:
            self._failure = self._failure or "explicit failed-runtime cleanup"
        try:
            self._owner.close(abandon=abandon)
        except BaseException as error:
            self._failure = self._failure or f"producer close failed: {error}"[:1024]
            if self._owner.snapshot()["closed"]:
                for row in tuple(self._frames.values()):
                    self._retire(row, "FAILED_CLOSE")
                self._closed = True
            raise
        for row in tuple(self._frames.values()):
            self._retire(row, "ABANDONED" if row.committed else "CANCELLED_ON_CLOSE")
        self._closed = True
        return self.snapshot()
