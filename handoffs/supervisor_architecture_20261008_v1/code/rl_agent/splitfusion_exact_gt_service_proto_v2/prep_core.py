"""Preparation-process core: freeze admission, deferred GT preparation, binding, commit, publication.

Single thread, one request at a time. Each job is keyed by its freeze descriptor
sequence. The decoded source is borrowed only inside the sealed mapping for
``prepare_unbound``; the mapping must release cleanly before the reply. The ready
packet is published once, as a new sealed memfd, for the scorer process.
"""
from __future__ import annotations

import secrets
import time

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_numeric_gt_runtime_v1 import process as RP

from . import channel as CH, freeze_codec as FZ
from .deferred import DeferredProducerEngine, DeferralRefused
from .scorer_core import Refused


class PrepCore:
    def __init__(self, config):
        engine = dict(config["engine"])
        self.offline = engine["offline_replay"]
        self.engine = DeferredProducerEngine(**engine)
        self.session_id = secrets.token_hex(32)
        self.gate = FZ.FreezeGate(self.session_id)
        self.jobs = {}

    def ready(self):
        return dict(session_id=self.session_id, offline_replay=self.offline)

    def _job(self, args, *phases):
        job = self.jobs.get(args.get("job"))
        if job is None or (phases and job["phase"] not in phases):
            raise Refused("UNKNOWN_JOB_OR_WRONG_PHASE")
        return job

    def handle(self, op, args, fd):
        e = self.engine
        if op == "FREEZE":
            C.require(fd is not None, "sealed source fd required")
            offline = args.get("offline")
            C.require((offline is not None) == self.offline, "offline replay inputs only in offline mode")
            with self.gate.admit(args["descriptor"].encode(), fd) as admitted:
                decoded = dict(decoded_raw_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
                               decoded_thread_cpu_ns=time.thread_time_ns())
                kwargs = admitted.kwargs
                handle = e.prepare_unbound(kwargs, population=kwargs["dynamic_snapshot_evidence"], **(offline or {}))
                del kwargs
                sequence, frame = admitted.descriptor["sequence"], admitted.descriptor["frame_id"]
            self.jobs[sequence] = dict(phase="UNBOUND", frame_id=frame, handle=handle)
            return dict(job=sequence, frame_id=frame, **decoded), None
        if op == "VOID":
            return dict(voided=self.gate.void(args["descriptor"].encode())), None
        if op == "BIND":
            job = self._job(args, "UNBOUND")
            plan = args["plan"].encode()
            try:
                prepared = e.bind(job["handle"], plan, frame_context=RP.context(args["frame_context"]))
            except DeferralRefused as refused:
                job.update(phase="REFUSED_CANCELLED", handle=None)
                raise Refused("DEFERRAL_REFUSED_" + refused.reason)
            job.update(phase="BOUND", prepared=prepared, plan=plan, handle=None)
            provenance = e.provenance(prepared.token)
            return dict(identity=prepared.identity_bytes.decode(), semantic_sha256=prepared.semantic_sha256,
                        provenance=None if provenance is None else C.canonical(provenance).decode()), None
        if op == "CANCEL_UNBOUND":
            job = self._job(args, "UNBOUND")
            e.cancel_unbound(job["handle"], args["reason"])
            job.update(phase="CANCELLED", handle=None)
            return dict(history_sha256=e._owner._producer.history_sha256()), None
        if op == "CONFIRM_SEND":
            job = self._job(args, "BOUND")
            C.require(args["plan"].encode() == job["plan"], "send confirmation differs from bound plan")
            committed = e.confirm_send(job["prepared"].token, job["plan"])
            job["phase"] = "COMMITTED"
            return dict(commit=committed), None
        if op == "CANCEL_BOUND":
            job = self._job(args, "BOUND")
            e.cancel(job["prepared"].token, job["plan"], args["reason"])
            job["phase"] = "CANCELLED"
            return dict(history_sha256=e._owner._producer.history_sha256()), None
        if op == "PUBLISH":
            job = self._job(args, "COMMITTED")
            packet = e.publish_ready(job["prepared"].token)
            out, size = CH.seal_bytes([packet])
            job["phase"] = "PUBLISHED"
            p = job["prepared"]
            return dict(identity=p.identity_bytes.decode(), semantic_sha256=p.semantic_sha256, size=size), out
        p = None
        if op in ("RESULT_DELIVERED", "EVIDENCE", "EVIDENCE_DURABLE", "UNSCORED_TERMINAL", "UNSCORED_DURABLE"):
            job = self._job(args, "PUBLISHED", "DELIVERED", "EVIDENCE", "UNSCORED")
            p = job["prepared"]
            note = (p.token, p.identity_bytes, p.semantic_sha256)
        if op == "RESULT_DELIVERED":
            C.require(job["phase"] == "PUBLISHED", "result delivery once, after publication")
            e.score_completed(*note)
            if self.offline:
                e.feedback_confirmed(*note)   # offline archive path requires this flag
            job["phase"] = "DELIVERED"
            return dict(ok=True), None
        if op == "EVIDENCE":
            C.require(job["phase"] == "DELIVERED", "evidence after delivered result")
            if self.offline:
                archive = e.build_archive(p.token)
                files = dict(archive.files)
            else:
                export = e.export_live_source(*note)
                files = {"input_metadata.json": export.input_metadata_bytes,
                         "source_synchronization.json": export.source_synchronization_bytes,
                         "numerical_avo.json": export.numerical_avo_bytes,
                         "producer_events.json": export.producer_events_bytes,
                         **e.provenance_files(p.token)}
            job.update(phase="EVIDENCE", files=files)
            index, chunks = CH.pack_files(files)
            out, _ = CH.seal_bytes(chunks)
            return dict(index=index), out
        if op == "EVIDENCE_DURABLE":
            C.require(job["phase"] == "EVIDENCE", "durability after evidence")
            if self.offline:
                e.scorer_archive_completed(*note)
                e.archive_durable(p.token, args["receipt"], relative_prefix=args["relative_prefix"])
                outcome = "DURABLE"
            else:
                outcome = e.complete_scored_durable(*note, files=job["files"], receipt=args["receipt"],
                    relative_prefix=args["relative_prefix"], scorer_proof=args["scorer_proof"])["outcome"]
            job.update(phase="RETIRED", files=None, prepared=None)
            return dict(outcome=outcome), None
        if op == "UNSCORED_TERMINAL":
            C.require(job["phase"] == "PUBLISHED", "unscored terminal only for an undelivered published frame")
            files = e.prepare_proto_unscored(*note, args["reason"])
            job.update(phase="UNSCORED", files=files)
            index, chunks = CH.pack_files(files)
            out, _ = CH.seal_bytes(chunks)
            return dict(index=index), out
        if op == "UNSCORED_DURABLE":
            C.require(job["phase"] == "UNSCORED", "unscored durability after its terminal files")
            outcome = e.complete_proto_unscored(*note, receipt=args["receipt"],
                                                relative_prefix=args["relative_prefix"])["outcome"]
            job.update(phase="RETIRED", files=None, prepared=None)
            return dict(outcome=outcome), None
        if op == "SNAPSHOT":
            snap = e.snapshot()
            return dict(engine=dict(valid=snap["valid"], failure=snap["failure"], issued=snap["issued"],
                retired=snap["retired"], retained_jobs=snap["retained_jobs"],
                retained_payload_bytes=snap["retained_payload_bytes"], owner=snap["owner"], deferral=snap["deferral"]),
                gate=dict(admitted=self.gate.admitted, failure=self.gate.failure,
                          last_sequence=self.gate.last_sequence, last_frame=self.gate.last_frame),
                history_sha256=e._owner._producer.history_sha256(),
                jobs={str(k): v["phase"] for k, v in self.jobs.items()}), None
        if op == "CLOSE":
            snap = e.close(abandon=bool(args.get("abandon")))
            return dict(closed=True, retired=snap["retired"], owner=snap["owner"]), None
        raise C.ContractError("unknown prep operation " + str(op))
