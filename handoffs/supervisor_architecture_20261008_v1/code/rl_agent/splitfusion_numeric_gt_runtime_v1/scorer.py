"""Bounded single-thread scorer child engine; no OS transport or live launch.

Only primitive expected metadata and immutable buffers cross this API. All
ReadyTarget, prediction and ScoreResult capabilities are admitted/created in
this process and remain here until scoring, parent-observed feedback and actual
durable archive receipts are complete. Parent feedback acknowledgements are
harness observations, NOT measurements of the radio or its 170-ms boundary.

Retention bounds cover serialized logical payloads, not Python object overhead,
peak decoder scratch or RSS. The parent owns authenticated IPC, archive roots,
request timeouts and explicit failure teardown. This engine never evicts work.
"""
from dataclasses import dataclass, field, fields
import hashlib
import os
from pathlib import Path, PurePosixPath
import struct
import threading
import time
from types import MappingProxyType

import cv2
import numpy as np

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C, evidence as V
from rl_agent.splitfusion_numeric_gt_integration_v2 import ready as R, scoring as Q, join as J
from rl_agent.splitfusion_minimal_quality_scorer_v1 import scorer as S
from rl_agent.splitfusion_live_dispatch_v1 import frame_context as FC
from rl_agent.splitfusion_live_parity_runtime_v1 import pose as PO
from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E


class ScorerError(RuntimeError):
    pass


class ScorerCapacityError(ScorerError):
    pass


def _expected(value):
    """Own primitive registration values and reconstruct a real FrameContext."""
    keys = {"expected_capture_identity", "expected_action_identity", "frame_context", "plan_sha256",
            "run_binding", "producer_binding", "prediction_contract"}
    C.require(type(value) is dict and set(value) == keys, "complete scorer registration expected values required")
    context = value["frame_context"]
    if type(context) is not FC.FrameContextV1:
        document = C._json_owned(context)
        C.require(type(document) is dict and set(document) == {"stream_id", "frame_id", "sequence_id",
            "capture_timestamp_ns", "ego_world", "camera_model_sha256", "camera_mount_sha256"},
            "complete primitive FrameContext required")
        C.require(type(document["ego_world"]) is list and len(document["ego_world"]) == 6,
                  "complete ego pose required")
        context = FC.FrameContextV1(**{k: v for k, v in document.items() if k != "ego_world"},
                                   ego_world=FC.Pose6D(*document["ego_world"]))
    # Reconstruct even caller-provided context to avoid retaining caller aliases.
    doc = R._context_document(context)
    context = FC.FrameContextV1(**{k: v for k, v in doc.items() if k != "ego_world"},
                               ego_world=FC.Pose6D(*doc["ego_world"]))
    result = {k: C._json_owned(v) for k, v in value.items() if k != "frame_context"}
    result["frame_context"] = context
    capture, action = result["expected_capture_identity"], result["expected_action_identity"]
    C._capture(capture); R._action(action, capture)
    C.require(all(C.is_digest(result[k]) for k in ("plan_sha256", "run_binding", "producer_binding")),
              "scorer registration digests missing")
    C.require((context.stream_id, context.frame_id, context.capture_timestamp_ns, context.sequence_id)
              == (capture["stream_id"], capture["frame_id"], capture["capture_timestamp_ns"], action["tensor_seq"]),
              "scorer registration context identity differs")
    R._prediction_contract(result["prediction_contract"], context)
    return result


def registration_key(expected):
    """Exact v2 ReadyTarget identity, computable before either input arrives."""
    expected = _expected(expected)
    return C.canonical(dict(capture_identity=expected["expected_capture_identity"],
        action_identity=expected["expected_action_identity"], plan_sha256=expected["plan_sha256"],
        run_binding=expected["run_binding"], producer_binding=expected["producer_binding"],
        frame_context=PO.context_document(expected["frame_context"]), prediction_contract=expected["prediction_contract"]))


@dataclass
class _Slot:
    key: bytes
    expected: dict
    reservation: int
    target: object = None
    prediction: object = None
    result: object = None
    scored: object = None
    feedback: bool = False
    files: object = None
    hashes: object = None
    file_bytes: int = 0
    result_bytes: int = 0
    archived: bool = False
    live_archive: object = None
    live_clock: object = None
    live_terminal: str | None = None
    live_completion_bytes: bytes | None = None
    live_receipt_sha256: str | None = None
    live_receipt_job_id: int | None = None
    live_relative_prefix: str | None = None
    live_selected_files: object = None
    # These instants are taken by the scorer itself, never reconstructed from
    # coordinator-supplied clock JSON.  They are retained with this slot.
    scorer_events: list = field(default_factory=list)


def _safe_relative(value, *, prefix=False):
    if type(value) is not str or not value or "\\" in value or "\x00" in value:
        return False
    path = PurePosixPath(value)
    return bool(path.parts) and not path.is_absolute() and ".." not in path.parts and str(path) == value.rstrip("/") \
        and (not prefix or value.endswith("/"))


def _decode_legacy_png(raw):
    # Bound dimensions before invoking a native decoder.
    C.require(type(raw) is bytes and 33 <= len(raw) <= C.MAX_METADATA_BYTES
        and raw[:8] == b"\x89PNG\r\n\x1a\n" and raw[12:16] == b"IHDR"
        and struct.unpack("!II", raw[16:24]) == (768, 432), "bounded original GT PNG required")
    value = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    C.require(type(value) is np.ndarray and value.dtype == np.uint8 and value.shape == (432, 768),
              "original GT PNG decode failed")
    return value


class ScorerEngine:
    def __init__(self, *, run_binding, producer_binding, max_jobs, max_bytes, spec=None, quality=None):
        C.require(C.is_digest(run_binding) and C.is_digest(producer_binding), "explicit scorer run/producer binding required")
        C.require(type(max_jobs) is int and 1 <= max_jobs <= 1000 and type(max_bytes) is int and max_bytes > 0,
                  "explicit bounded scorer capacity required")
        if (spec is None) != (quality is None):
            raise ScorerError("quality spec and module must be supplied together")
        if spec is None:
            from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as quality_wire
            from rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid import quality
            spec = quality_wire.load_run4_quality_spec(Path(__file__).resolve().parents[2])
        self.scorer = S.NumericalScorer()
        self._spec, self._quality = spec, quality
        self._join = J.BoundedTargetPredictionJoin(self.scorer, max_jobs=max_jobs, max_bytes=max_bytes)
        self._run, self._producer = run_binding, producer_binding
        self._max_jobs, self._max_bytes = max_jobs, max_bytes
        self._rows = {}
        self._reserved = self._released = self._scored = self._attached = 0
        self._thread = threading.current_thread()
        self._failure = None
        self._closed = False

    def _check(self, *, healthy=True):
        if threading.current_thread() is not self._thread:
            self._fail("scorer engine accessed from a foreign thread")
        if self._closed:
            raise ScorerError("scorer engine is closed")
        if healthy and self._failure:
            raise ScorerError(self._failure)

    def _fail(self, message, error=ScorerError):
        self._failure = self._failure or str(message)[:1024]
        raise error(str(message))

    def _slot(self, key, *, healthy=True):
        self._check(healthy=healthy)
        if type(key) is not bytes or key not in self._rows:
            self._fail("unregistered or retired scorer identity")
        return self._rows[key]

    def register(self, key, *, expected, reservation_bytes):
        self._check()
        try:
            owned = _expected(expected)
            C.require(type(key) is bytes and registration_key(owned) == key, "registration full identity differs")
            C.require(owned["run_binding"] == self._run and owned["producer_binding"] == self._producer,
                      "foreign scorer registration binding")
            if key in self._rows:
                C.require(self._rows[key].reservation == reservation_bytes, "conflicting reservation")
                return False
            if (type(reservation_bytes) is not int or reservation_bytes <= 0
                    or len(self._rows) >= self._max_jobs or self._reserved + reservation_bytes > self._max_bytes):
                self._fail("scorer retention capacity exceeded; no eviction", ScorerCapacityError)
            self._join.register(key, reserved_bytes=reservation_bytes)
            self._rows[key] = _Slot(key, owned, reservation_bytes)
            self._reserved += reservation_bytes
            return True
        except Exception as error:
            self._failure = self._failure or f"scorer registration failed: {error}"[:1024]
            raise

    def _account(self, row):
        # The join owns target/prediction capabilities also shared by ScoreResult.
        target = row.target
        pred = row.prediction
        amount = len(row.key)
        if target is not None:
            amount += len(target.metadata_bytes) + len(target.segmentation_bytes) + len(target.object_ignore_bytes)
        if pred is not None:
            amount += len(pred.labels) + len(pred.raw_records) + len(C.canonical(pred.identity))
        amount += row.result_bytes + row.file_bytes
        if row.live_completion_bytes is not None:
            amount += len(row.live_completion_bytes)
        if amount > row.reservation:
            self._fail("scorer frame exceeds reserved logical storage", ScorerCapacityError)
        return amount

    @staticmethod
    def _stamp_scorer(row, name):
        row.scorer_events.append((name, time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW), os.getpid()))

    def put_target(self, key, packet):
        row = self._slot(key)
        try:
            target = R.decode_target(packet, **row.expected)
            C.require(target.identity_bytes == key, "target belongs to another full identity")
            changed = self._join.put_target(target)
            if changed:
                row.target = target
            self._account(row)
            if changed:
                self._stamp_scorer(row, "target_admitted")
            return changed
        except Exception as error:
            self._failure = self._failure or f"target arrival failed: {error}"[:1024]
            raise

    def put_prediction(self, key, *, identity, labels, shape, records, prediction_contract):
        row = self._slot(key)
        try:
            C.require(type(labels) is bytes and type(shape) in (tuple, list)
                and len(shape) == 2 and all(type(x) is int for x in shape)
                and tuple(shape) == (720, 1280) and len(labels) == 720 * 1280,
                "exact primitive native prediction bytes/shape required")
            view = np.frombuffer(labels, np.uint8).reshape(tuple(shape))
            admitted = Q.admit_prediction(self.scorer, identity=identity, labels=view, records=records,
                                          prediction_contract=prediction_contract)
            changed = self._join.put_prediction(key, admitted)
            if changed:
                row.prediction = admitted
            self._account(row)
            if changed:
                self._stamp_scorer(row, "prediction_admitted")
            return changed
        except Exception as error:
            self._failure = self._failure or f"prediction arrival failed: {error}"[:1024]
            raise

    def score_ready(self, key):
        """Return None until both inputs exist; a repeated call never rematches."""
        row = self._slot(key)
        if row.scored is not None:
            return row.scored
        try:
            pair = self._join.wait_take(key, timeout=0)
            if pair is None:
                return None
            frame = Q.join_admitted_inputs(self.scorer, target=pair.target, prediction=pair.prediction)
            self._stamp_scorer(row, "score_start")
            result = self.scorer.score(frame)
            value = self._quality.evaluate_exact_quality(self._spec, result.measurement)
            qperc = None if value is None else float(value.q_perc)
            answer = C._freeze(dict(frame_id=frame.frame_id, measurement=result.measurement, q_perc=qperc,
                quality_defined=value is not None, ready_semantic_sha256=pair.target.semantic_sha256,
                identity_sha256=hashlib.sha256(key).hexdigest(), live_qualified=False))
            row.result, row.scored = result, answer
            row.result_bytes = len(C.canonical(dict(answer=answer, associations={f.name: getattr(result.associations, f.name)
                for f in fields(result.associations)}, confusion=result.confusion, vehicle_detection=result.vehicle_detection,
                ground_truth=frame.ground_truth, numeric_predictions=frame.predictions)))
            self._account(row)
            self._scored += 1
            self._stamp_scorer(row, "score_end")
            return answer
        except Exception as error:
            self._failure = self._failure or f"numerical scoring failed: {error}"[:1024]
            raise

    def confirm_feedback(self, key):
        """Parent observed callback completion; not a radio ACK or reward timer."""
        row = self._slot(key)
        if row.live_archive is not None:
            self._fail("harness feedback acknowledgement cannot replace observed live UE terminal")
        if row.scored is None:
            self._fail("feedback completion precedes numerical scoring")
        changed = not row.feedback
        row.feedback = True
        self._release_if_complete(row)
        return changed

    def attach_archive(self, key, *, ready_semantic_sha256, input_metadata, avo_document,
                       legacy_target_meta, segmentation_png, object_ignore_png, old_prepared_sha256):
        """Validate primitive deferred bytes, then reuse retained associations."""
        row = self._slot(key)
        if row.scored is None or not row.feedback:
            self._fail("deferred attachment precedes score/parent feedback acknowledgement")
        if row.files is not None:
            self._fail("archive attachment is create-only for this accepted frame")
        try:
            C.require(ready_semantic_sha256 == row.target.semantic_sha256, "foreign deferred ready target")
            metadata = C._json_owned(legacy_target_meta)
            C.require(type(metadata) is dict and not ({"segmentation", "object_ignore", "segmentation_png",
                "object_ignore_png", "_csv_bytes"} & set(metadata)), "primitive legacy target metadata required")
            C.require(len(C.canonical(metadata)) <= C.MAX_METADATA_BYTES, "legacy metadata bound exceeded")
            source = V.admit_source_evidence(input_metadata=input_metadata, avo_document=avo_document,
                expected_capture_identity=row.expected["expected_capture_identity"])
            legacy = dict(metadata, segmentation_png=segmentation_png, object_ignore_png=object_ignore_png,
                segmentation=_decode_legacy_png(segmentation_png), object_ignore=_decode_legacy_png(object_ignore_png))
            prepared = E.restore_prepared_target(frame_id=row.target.metadata["capture_identity"]["frame_id"],
                sample_id=row.target.metadata["sample_id"], target=legacy, content_sha256=old_prepared_sha256)
            reference = Q.attach_archive(row.target, source_evidence=source, target_tables=legacy["tables"],
                old_prepared_sha256=old_prepared_sha256, prepared_target=prepared)
            files = Q.archive_payloads(row.result, archive_reference=reference)
            C.require(set(files) == {"predicted_labels.png", "evaluation.json"}
                and all(type(data) is bytes and len(data) <= C.MAX_METADATA_BYTES for data in files.values()),
                "unexpected scorer archive output/bounds")
            row.files = MappingProxyType(dict(files))
            row.hashes = MappingProxyType({name: hashlib.sha256(data).hexdigest() for name, data in files.items()})
            row.file_bytes = sum(len(data) for data in files.values())
            self._account(row)
            self._attached += 1
            return dict(files=row.files, sha256=row.hashes, ready_semantic_sha256=row.target.semantic_sha256,
                        identity_sha256=hashlib.sha256(key).hexdigest())
        except Exception as error:
            self._failure = self._failure or f"deferred archive attachment failed: {error}"[:1024]
            raise

    def attach_live_archive(self, key, *, input_metadata, source_synchronization,
                            numerical_avo, clock_events, legacy_artifacts=None,
                            handoff_id=None):
        """Attach a scored ticket to observed, versioned numerical evidence.

        The caller delivers non-scorer owner-stamped events over the
        authenticated coordinator channel. Scorer admissions and score bounds
        come only from this slot's actual RAW stamps; caller-supplied scorer
        events are refused. This method does not install a live route or
        declare the 170-ms controller verdict.
        """
        from . import live_archive as L, live_clock as CLOCK
        row = self._slot(key)
        if row.scored is None or row.result is None or row.target is None or row.prediction is None:
            self._fail("live archive requires a completed retained numerical score")
        if row.files is not None or row.live_archive is not None or row.feedback:
            self._fail("live and historical archive ownership may not be mixed or repeated")
        try:
            C.require(type(clock_events) in (list, tuple) and 1 <= len(clock_events) <= 32,
                      "bounded complete observed clock events required")
            ledger = CLOCK.LiveClockLedger(key, row.target.semantic_sha256,
                handoff_id=handoff_id)
            scorer_pid, scorer_boot = os.getpid(), CLOCK.read_boot_id()
            C.require(tuple(name for name, _, _ in row.scorer_events).count("target_admitted") == 1
                      and tuple(name for name, _, _ in row.scorer_events).count("prediction_admitted") == 1
                      and tuple(name for name, _, _ in row.scorer_events).count("score_start") == 1
                      and tuple(name for name, _, _ in row.scorer_events).count("score_end") == 1,
                      "complete scorer-owned admissions and score bounds required")
            for supplied in clock_events:
                event = C._json_owned(supplied)
                C.require(type(event) is dict and set(event) == {
                    "role", "name", "raw_ns", "pid", "boot_id", "clock_domain",
                    "ready_identity", "ready_identity_sha256", "semantic_sha256"},
                    "complete owner-stamped clock event required")
                C.require(event["role"] not in ("archive_client", "scorer")
                    and event["name"] not in ("archive_build_start", "archive_build_end",
                                                "durable_completion_observed", "late_feedback_observed"),
                    "scorer/archive/late events must be recorded by their actual owner")
                actual = ledger.record(role=event["role"], name=event["name"], raw_ns=event["raw_ns"],
                    pid=event["pid"], boot_id=event["boot_id"],
                    ready_identity_bytes=key, semantic_sha256=row.target.semantic_sha256)
                C.require(C.canonical(actual) == C.canonical(event),
                          "clock event contents differ from admitted full frame identity")
            for name, raw_ns, pid in row.scorer_events:
                C.require(pid == scorer_pid, "scorer-owned score stamp belongs to another process")
                ledger.record(role="scorer", name=name, raw_ns=raw_ns, pid=pid, boot_id=scorer_boot,
                    ready_identity_bytes=key, semantic_sha256=row.target.semantic_sha256)
            terminal = ledger.deadline()
            C.require(terminal["terminal"] in ("FEEDBACK_ON_TIME", "FEEDBACK_TIMEOUT"),
                      "observed UE terminal required for scored live archive")
            ledger.record(role="scorer", name="archive_build_start",
                raw_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW), pid=scorer_pid, boot_id=scorer_boot,
                ready_identity_bytes=key, semantic_sha256=row.target.semantic_sha256)
            archive = L.build_live_archive(ready_target=row.target, score_result=row.result,
                q_perc=row.scored["q_perc"], quality_spec=self._spec,
                input_metadata=input_metadata, source_synchronization=source_synchronization,
                numerical_avo=numerical_avo, clock_ledger=ledger,
                legacy_artifacts=legacy_artifacts)
            ledger.record(role="scorer", name="archive_build_end",
                raw_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW), pid=scorer_pid, boot_id=scorer_boot,
                ready_identity_bytes=key, semantic_sha256=row.target.semantic_sha256)
            C.require(archive.ready_identity_bytes == key
                      and archive.semantic_sha256 == row.target.semantic_sha256,
                      "live archive returned foreign identity or semantic result")
            L.verify_live_archive(archive)
            row.live_archive = archive
            row.live_clock = ledger
            row.live_terminal = terminal["terminal"]
            row.files = archive.files
            row.hashes = MappingProxyType({name: hashlib.sha256(data).hexdigest()
                                           for name, data in row.files.items()})
            row.file_bytes = sum(map(len, row.files.values()))
            self._account(row)
            self._attached += 1
            return dict(files=row.files, sha256=row.hashes,
                        archive_manifest_sha256=archive.manifest_sha256,
                        ready_semantic_sha256=row.target.semantic_sha256,
                        identity_sha256=hashlib.sha256(key).hexdigest(),
                        terminal=row.live_terminal,
                        clock_descriptor_bytes=archive.clock_descriptor_bytes,
                        live_qualified=False)
        except Exception as error:
            self._failure = self._failure or f"live archive attachment failed: {error}"[:1024]
            raise

    def confirm_live_archive(self, key, receipt, relative_prefix, *,
                             archive_submitted_event, durable_completion_event):
        """Release a scored live ticket only after exact durable file receipt.

        The completion clock record is returned separately because its digest
        cannot be included in the archive it certifies.
        """
        row = self._slot(key)
        if (row.live_archive is None or row.live_clock is None or row.live_terminal is None
                or row.archived or row.live_completion_bytes is not None):
            self._fail("live durable receipt precedes a scored terminal/archive or repeats completion")
        try:
            C.require(_safe_relative(relative_prefix, prefix=True), "safe live scorer archive prefix required")
            value = C._json_owned(receipt)
            C.require(type(value) is dict and value.get("status") == "COMPLETE"
                      and value.get("durable") is True and value.get("error") is None
                      and type(value.get("job_id")) is int and value["job_id"] > 0
                      and C.canonical(value.get("identity")) ==
                          C.canonical(row.expected["expected_capture_identity"]),
                      "actual durable archive receipt for this capture required")
            written = value.get("written")
            C.require(type(written) is list and len(written) <= C.MAX_ROWS,
                      "bounded durable archive inventory required")
            selected = {}
            expected = {relative_prefix + name for name in row.files}
            for item in written:
                C.require(type(item) is dict and type(item.get("relative")) is str,
                          "malformed durable archive inventory row")
                name = item["relative"]
                if not name.startswith(relative_prefix):
                    continue
                C.require(set(item) == {"root", "relative", "sha256", "bytes", "ready"}
                          and name in expected and name not in selected
                          and type(item.get("root")) is int and item["root"] >= 0
                          and type(item.get("bytes")) is int and item["bytes"] >= 0
                          and item.get("ready") is False,
                          "foreign, duplicate or malformed live archive file receipt")
                selected[name] = item
            C.require(set(selected) == expected and len({item["root"] for item in selected.values()}) == 1,
                      "live archive receipt misses a file or spans roots")
            for name, raw in row.files.items():
                item = selected[relative_prefix + name]
                C.require(item.get("sha256") == row.hashes[name] and item["bytes"] == len(raw),
                          "live archive durable bytes/hash differ: " + name)
            submit = C._json_owned(archive_submitted_event)
            completed = C._json_owned(durable_completion_event)
            C.require(type(submit) is dict and type(completed) is dict,
                      "actual archive submit/completion events required")
            for event, expected_name in ((submit, "archive_submitted"),
                                         (completed, "durable_completion_observed")):
                C.require(set(event) == {"role", "name", "raw_ns", "pid", "boot_id", "clock_domain",
                                          "ready_identity", "ready_identity_sha256", "semantic_sha256"}
                          and event["role"] == "archive_client" and event["name"] == expected_name,
                          "complete archive client owner event required")
            actual_submit = row.live_clock.record(role="archive_client", name="archive_submitted",
                raw_ns=submit["raw_ns"], pid=submit["pid"], boot_id=submit["boot_id"],
                ready_identity_bytes=key, semantic_sha256=row.target.semantic_sha256)
            C.require(C.canonical(actual_submit) == C.canonical(submit),
                      "archive submit event full identity differs")
            completion = row.live_clock.durability_record(descriptor_bytes=row.live_archive.clock_descriptor_bytes,
                receipt=value, raw_ns=completed["raw_ns"], pid=completed["pid"], boot_id=completed["boot_id"],
                ready_identity_bytes=key, semantic_sha256=row.target.semantic_sha256)
            C.require(C.canonical(row.live_clock._events[("archive_client", "durable_completion_observed")])
                      == C.canonical(completed), "archive completion event full identity differs")
            manifest_sha256 = row.live_archive.manifest_sha256
            row.live_completion_bytes = completion
            row.live_receipt_sha256 = C.sha256(C.canonical(value))
            row.live_receipt_job_id = value["job_id"]
            row.live_relative_prefix = relative_prefix
            row.live_selected_files = C._json_owned(selected)
            self._account(row)
            return dict(status="COMPLETION_SIDECAR_PENDING_DURABILITY", terminal=row.live_terminal,
                archive_manifest_sha256=manifest_sha256,
                clock_completion_bytes=completion, clock_completion_sha256=hashlib.sha256(completion).hexdigest(),
                live_qualified=False)
        except Exception as error:
            self._failure = self._failure or f"live archive receipt failed: {error}"[:1024]
            raise

    def confirm_live_completion_durable(self, key, proof):
        """Release only after the separate completion sidecar is durably read back.

        The coordinator authenticates the proof-producing archive process. This
        child verifies its exact readback bytes against the completion record it
        retained, the previous ArchiveClient receipt, and its live archive.
        """
        row = self._slot(key)
        if (row.live_archive is None or row.live_completion_bytes is None
                or row.live_receipt_sha256 is None or row.live_receipt_job_id is None
                or row.live_selected_files is None or row.archived):
            self._fail("live completion proof precedes a retained receipt or repeats release")
        try:
            C.require(type(proof) is dict and set(proof) == {
                "schema", "path", "bytes", "sha256", "job_id", "receipt_sha256",
                "archive_manifest_sha256", "clock_completion_sha256", "clock_completion_bytes",
                "created_exclusive", "file_fsynced", "directory_fsynced", "readback_verified",
                "readback_bytes"}, "exact postreceipt sidecar durability proof required")
            C.require(proof["schema"] == "splitfusion.numeric_gt.live_completion_durability_proof.v1"
                      and type(proof["path"]) is str and Path(proof["path"]).is_absolute()
                      and ".." not in Path(proof["path"]).parts
                      and Path(proof["path"]).name == "live_archive_completion.json"
                      and all(proof[name] is True for name in
                              ("created_exclusive", "file_fsynced", "directory_fsynced", "readback_verified")),
                      "sidecar must have a create-only fsynced absolute path and verified readback")
            raw = proof["readback_bytes"]
            C.require(type(raw) is bytes and 0 < len(raw) <= C.MAX_METADATA_BYTES
                      and type(proof["bytes"]) is int and proof["bytes"] == len(raw)
                      and C.is_digest(proof["sha256"]) and C.sha256(raw) == proof["sha256"],
                      "sidecar readback differs from durable proof bytes/hash")
            C.require(proof["job_id"] == row.live_receipt_job_id
                      and proof["receipt_sha256"] == row.live_receipt_sha256
                      and proof["archive_manifest_sha256"] == row.live_archive.manifest_sha256
                      and proof["clock_completion_sha256"] == C.sha256(row.live_completion_bytes)
                      and proof["clock_completion_bytes"] == len(row.live_completion_bytes),
                      "sidecar proof is not bound to this receipt, manifest and completion record")
            document = C._parse(raw)
            C.require(C.canonical(document) == raw
                      and document.get("schema") == "splitfusion.numeric_gt.live_receipt.v1"
                      and document.get("job_id") == row.live_receipt_job_id
                      and document.get("root_index") ==
                          next(iter({item["root"] for item in row.live_selected_files.values()}))
                      and document.get("relative_prefix") == row.live_relative_prefix
                      and document.get("archive_manifest_sha256") == row.live_archive.manifest_sha256
                      and document.get("archive_receipt_sha256") == row.live_receipt_sha256
                      and document.get("files") == row.live_selected_files
                      and C.canonical(document.get("clock_completion")) == row.live_completion_bytes
                      and C.sha256(C.canonical(document.get("archive_receipt"))) == row.live_receipt_sha256,
                      "sidecar readback does not contain this exact archive receipt and completion")
            row.archived = True
            terminal, manifest_sha = row.live_terminal, row.live_archive.manifest_sha256
            identity_sha, semantic_sha = C.sha256(key), row.target.semantic_sha256
            self._release_if_complete(row)
            return dict(status="DURABLE", terminal=terminal,
                        archive_manifest_sha256=manifest_sha,
                        sidecar_sha256=proof["sha256"],
                        ready_identity_sha256=identity_sha,
                        ready_semantic_sha256=semantic_sha,
                        live_qualified=False)
        except Exception as error:
            self._failure = self._failure or f"live sidecar durability failed: {error}"[:1024]
            raise

    def confirm_archive(self, key, receipt, relative_prefix):
        """Verify the real ArchiveClient COMPLETE receipt for both scorer files.

        Parent authenticates the receipt and binds its archive root. This engine
        checks exact capture, durable status, safe destination prefix and actual
        per-file byte counts/hashes. Other producer files may share the receipt.
        """
        row = self._slot(key)
        if row.files is None or not row.feedback or row.scored is None or row.live_archive is not None:
            self._fail("archive receipt precedes scoring/feedback/deferred evidence")
        try:
            value = C._json_owned(receipt)
            C.require(type(value) is dict and value.get("status") == "COMPLETE" and value.get("durable") is True,
                      "actual durable COMPLETE archive receipt required")
            C.require(C.canonical(value.get("identity")) == C.canonical(row.expected["expected_capture_identity"]),
                      "foreign archive receipt capture identity")
            C.require(_safe_relative(relative_prefix, prefix=True), "safe scorer archive prefix required")
            written = value.get("written")
            C.require(type(written) is list and len(written) <= C.MAX_ROWS, "bounded archive written inventory required")
            selected = {}
            expected_names = {relative_prefix + name for name in row.files}
            for item in written:
                C.require(type(item) is dict and type(item.get("relative")) is str, "invalid archive written item")
                relative = item["relative"]
                if relative not in expected_names:
                    continue
                C.require(relative not in selected and type(item.get("root")) is int and item["root"] >= 0,
                          "duplicate/invalid scorer file receipt")
                selected[relative] = item
            C.require(set(selected) == expected_names, "archive receipt missing scorer files")
            C.require(len({item["root"] for item in selected.values()}) == 1, "scorer archive roots disagree")
            for name, data in row.files.items():
                item = selected[relative_prefix + name]
                C.require(item.get("sha256") == row.hashes[name] and type(item.get("bytes")) is int
                    and item["bytes"] == len(data) and item.get("ready") is False,
                    "archive receipt scorer bytes/hash/READY designation differ")
            row.archived = True
            self._release_if_complete(row)
            return True
        except Exception as error:
            self._failure = self._failure or f"archive receipt failed: {error}"[:1024]
            raise

    def _release_if_complete(self, row):
        if row.scored is not None and row.archived and (
                (row.live_archive is None and row.feedback)
                or (row.live_archive is not None and row.live_terminal is not None)):
            self._join.release(row.key)
            del self._rows[row.key]
            self._reserved -= row.reservation
            self._released += 1
            row.target = row.prediction = row.result = row.scored = row.files = row.hashes = None
            row.live_archive = row.live_clock = None

    def snapshot(self):
        if threading.current_thread() is not self._thread:
            self._fail("foreign thread snapshot")
        join = self._join.snapshot()
        return dict(valid=self._failure is None and join["valid"], failure=self._failure or join["failure"],
            closed=self._closed, retained_jobs=len(self._rows), reserved_bytes=self._reserved,
            scored=self._scored, attached=self._attached, released=self._released, join=join,
            feedback_scope="PARENT_HARNESS_CALLBACK_NOT_RADIO", live_qualified=False)

    def close(self, *, abandon=False):
        self._check(healthy=False)
        if self._rows and not abandon:
            self._fail("scorer close before feedback/archive drain; ownership retained")
        if self._rows:
            self._failure = self._failure or "explicit failure abandonment of scorer frames"
            for key in tuple(self._rows):
                self._join.abandon(key, self._failure)
                self._reserved -= self._rows[key].reservation
                del self._rows[key]
        self._join.close()
        self._closed = True
        return self.snapshot()
