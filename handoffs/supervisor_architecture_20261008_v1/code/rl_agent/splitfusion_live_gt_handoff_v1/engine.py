"""Exact GT computation with scoring admission before asynchronous persistence.

The same accepted post-send frame advances the same producer history exactly
once. Packet admission is not archive completion. All queued evidence must
drain before collector preservation; a late archive fault invalidates the run,
never retroactively changes a reward already emitted by the evaluator.
"""
import hashlib
import json
import os
from pathlib import Path
import time
from types import MethodType

import numpy as np

from rl_agent.splitfusion_live_gt_optimization_v4 import worker as W
from rl_agent.splitfusion_live_gt_optimization_v6.engine import Engine as V6Engine
from rl_agent.splitfusion_live_parity_repair_v1 import gt_bridge as G, target_transport as T
from rl_agent.splitfusion_quality_feedback_probe_v1 import gt_evidence as GE
from . import contract as C
from .archive import ArchiveClient
from .broker import publish
from .scoring_packet import encode_packet
from .deferred_files import FileBatch, target_files, resolve_records


def timing_flush(log, directory, *, worker_drained):
    """Version changed readiness instead of claiming original disk semantics."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    with log.lock:
        if log.closed:
            raise ValueError("handoff timing already flushed")
        log.closed = True
        rows, dropped = list(log.rows), log.dropped
    encoded = b"".join(W.canonical(r)+b"\n" for r in rows)
    result = dict(schema="splitfusion.gt_scoring_handoff.stage_timing.v1", records=len(rows),
        dropped=dropped, bound=log.limit, clock="CLOCK_MONOTONIC_RAW", cpu_clock="thread_time_ns",
        flush_after_worker_drain=worker_drained, work_and_readiness_unchanged=False,
        scoring_ready_is_not_archive_durable=True,
        sha256={"stages.jsonl": hashlib.sha256(encoded).hexdigest()})
    T._write_x(directory/"stages.jsonl", encoded)
    T._write_x(directory/"MANIFEST.json", W.canonical(result))
    return result


class Engine(V6Engine):
    def __init__(self, *, handoff_config=None, **kwargs):
        self.archiver = None
        self.archive_roots = None
        self.handoffs = []
        self.pending_records = []
        self.archive_result = None
        self.failed_batches = []
        self.handoff_binding = C.binding()
        super().__init__(**kwargs)
        try:
            # Reuse the V6 privately specialized producer, depth decoder and
            # exact AVO/preparation functions. No reimplementation of geometry.
            scoped = dict(zip(self.write.__code__.co_freevars, self.write.__closure__))
            original = scoped["private_write"].cell_contents
            closed = {n: c.cell_contents for n, c in zip(original.__code__.co_freevars, original.__closure__)}
            self.producer_type = original.__globals__["S"].LiveCompleteTargetProducer
            self.prepare = original.__globals__["evaluation"].prepare_target
            self.avo_write = closed["original"][2]
            self.semantic_attempt = closed["attempt_dir"]
            self.validate_inputs = T.write_bundle.__wrapped__.__globals__["_validate_inputs"]
            self.log.flush = MethodType(timing_flush, self.log)
            self.write = self._write
            self.publish_timed = self.log.wrap("publish_scoring_packet", self._publish)
            self.serialize_timed = self.log.wrap("serialize_archive", self._serialize)
            self.enqueue_timed = self.log.wrap("enqueue_archive", self._enqueue)
            if handoff_config is not None:
                W.require(set(handoff_config) == {"directory", "evidence_root"}, "invalid handoff startup roots")
                self._start_archive(tuple(dict.fromkeys((
                    Path(handoff_config["directory"]).resolve(strict=True),
                    Path(handoff_config["evidence_root"]).resolve(strict=True),
                    self.attempt.resolve(strict=True)))))
        except BaseException:
            self.close()
            raise

    def _start_archive(self, roots):
        W.require(self.archiver is None, "archive process already started")
        self.archive_roots = roots
        self.archiver = ArchiveClient(roots=roots, max_jobs=C.MAX_JOBS,
            max_bytes=C.MAX_BYTES, max_frames=self.max_frames,
            binding={"schema": C.SCHEMA, "source_binding": self.handoff_binding}, timeout_s=30.)

    def _ensure_archiver(self, directory, sync_evidence):
        # Explicit roots, not arbitrary filesystem destinations. The caller's
        # attempt and edge mount already exist before any scoring request.
        root = Path(directory).resolve(strict=True)
        sync_root = Path(sync_evidence.directory).resolve().parent if sync_evidence is not None else self.attempt
        roots = tuple(dict.fromkeys((root, sync_root.resolve(strict=True), self.attempt.resolve(strict=True))))
        if self.archiver is None:
            W.require(self.allow_replay_clock, "live archiver must start before sensor startup")
            self._start_archive(roots)
        W.require(roots == self.archive_roots, "archive roots changed mid-session")
        self.archiver.poll()  # a prior disk failure stops new scoring admission
        return root

    def _publish(self, directory, *, identity, inputs, prepared, avo_document):
        # Original complete native-input validator precedes packet admission;
        # all source geometry/history and AVO validators already ran unchanged.
        self.validate_inputs(inputs, identity)
        packet_start = W.raw()
        packet = encode_packet(identity=identity, prepared=prepared,
            input_metadata=T._input_metadata(inputs), avo_document=avo_document,
            ready_raw_ns=packet_start, ready_wall_ns=time.time_ns(), binding=self.handoff_binding)
        accepted = publish(Path(directory)/C.SOCKET_NAME, packet, binding=self.handoff_binding, timeout_s=3.)
        receipt = dict(schema=C.SCHEMA, identity=dict(identity), packet_sha256=hashlib.sha256(packet).hexdigest(),
            packet_bytes=len(packet), prepared_content_sha256=prepared.content_sha256,
            input_content_sha256=inputs.provenance["input_content_sha256"],
            packet_construct_start_raw_ns=packet_start, accepted_response_raw_ns=W.raw(),
            broker=accepted, archive_durable=False)
        self.handoffs.append(receipt)
        return receipt

    def _serialize(self, batch, directory, *, identity, inputs, prepared, evidence, rgb, radar):
        desc, ready = target_files(batch, directory, identity=identity, inputs=inputs, prepared=prepared)
        source = Path(desc["directory"]).with_suffix(".sources")
        if G.sha(evidence) != inputs.provenance["evidence_sha256"]["owned_source"]:
            raise ValueError("producer source evidence digest differs")
        image = rgb.get("_rgb_image")
        if image is None or int(image.frame) != int(inputs.frame["frame_id"]):
            raise ValueError("owned native RGB needed for reproducible perception")
        pixels = np.frombuffer(image.raw_data, np.uint8).reshape(int(image.height), int(image.width), 4)
        payloads = {"SOURCE_EVIDENCE.json": G.canonical(evidence),
            "rgb_bgra.npy": T._array_bytes(pixels), "radar_world_xyz.npy": T._array_bytes(radar)}
        for name, value in payloads.items():
            batch.add(source/name, value)
        document = dict(schema="splitfusion.live_parity_source_archive.v1",
            identity=inputs.provenance["capture_identity"],
            sha256={name: hashlib.sha256(p).hexdigest() for name, p in payloads.items()})
        encoded = G.canonical(document)
        batch.add(source/"MANIFEST.json", encoded)
        return desc, dict(directory=str(source), manifest_sha256=hashlib.sha256(encoded).hexdigest()), ready

    def _enqueue(self, batch, *, identity, ready_paths=()):
        job = self.archiver.submit(dict(identity), batch.files, ready_paths=ready_paths,
                                   stamp_ready_paths=ready_paths)
        return dict(job_id=job, files=len(batch.files), bytes=batch.bytes, durable=False,
                    submitted_raw_ns=W.raw())

    def _rejected_sources(self, batch, kwargs, radar):
        """Failure-only raw observations, never an accepted source/target.

        Keep encoded image bytes even when their dimensions/timestamps are the
        reason validation failed. Per-source errors and budget elision remain
        explicit; diagnostic serialization cannot replace the primary error.
        """
        maximum, reserve = 16*1024*1024, 32*1024
        document = dict(schema="splitfusion.rejected_raw_sources.v1", accepted=False,
            identity=dict(kwargs["identity"]), max_payload_bytes=maximum,
            captured_payload_bytes=0, sources={})
        destination = self.attempt/"rejected_sources"/f"frame_{kwargs['identity']['frame_id']:010d}"
        sources = (("rgb", kwargs.get("rgb", {}).get("_rgb_image")),
                   ("depth", kwargs.get("depth_image")),
                   ("semantic", kwargs.get("semantic_image")), ("radar", radar))
        for role, value in sources:
            row = document["sources"][role] = dict(status="NOT_CAPTURED")
            try:
                if value is None:
                    row["reason"] = "SOURCE_ABSENT"
                    continue
                budget = min(maximum-document["captured_payload_bytes"],
                             batch.max_bytes-batch.bytes-reserve)
                if role == "radar":
                    if not isinstance(value, np.ndarray) or value.dtype.hasobject:
                        raise ValueError("non-object native radar ndarray required")
                    row.update(shape=list(value.shape), dtype=str(value.dtype))
                    if value.nbytes + 1024 > budget:
                        row["reason"] = "REJECTED_SOURCE_BYTE_LIMIT"
                        continue
                    payload = T._array_bytes(value)
                    path = destination/"radar.npy"
                    row["source_sha256"] = hashlib.sha256(value.tobytes(order="C")).hexdigest()
                    row["encoding"] = "NUMPY_ALLOW_PICKLE_FALSE"
                else:
                    row["header"] = {name+"_repr": repr(getattr(value, name, None))
                        for name in ("frame", "timestamp", "width", "height")}
                    raw = value.raw_data
                    if not isinstance(raw, (bytes, bytearray, memoryview)):
                        raise ValueError("encoded sensor bytes required")
                    size = raw.nbytes if isinstance(raw, memoryview) else len(raw)
                    if size > budget:
                        row["reason"] = "REJECTED_SOURCE_BYTE_LIMIT"
                        continue
                    payload = bytes(raw)
                    path = destination/(role+".bgra.bin")
                    row["source_sha256"] = hashlib.sha256(payload).hexdigest()
                    row["encoding"] = "EXACT_ENCODED_BGRA_BYTES_UNVALIDATED"
                if len(payload) > budget:
                    row["reason"] = "REJECTED_SOURCE_BYTE_LIMIT"
                    continue
                batch.add(path, payload)
                row.update(status="CAPTURED_NOT_ACCEPTED", path=str(path), bytes=len(payload),
                           sha256=hashlib.sha256(payload).hexdigest())
                document["captured_payload_bytes"] += len(payload)
            except Exception as diagnostic_error:
                row.update(reason="DIAGNOSTIC_SERIALIZATION_FAILED",
                           error=f"{type(diagnostic_error).__name__}: {diagnostic_error}")
        return document

    def _write(self, directory, **kwargs):
        frame = kwargs["identity"]["frame_id"]
        W.require(self.owner.live.last_datagram.get(frame) is not None, "GT precedes last send")
        frozen = kwargs["frozen_world"]
        population = frozen.population_evidence
        W.require(frame == population["frame_id"], "population belongs to another frame")
        radar = self.local.kwargs.get("radar_points", {}).get("world_xyz")
        W.require(radar is not None, "complete GT needs owned radar")
        root = self._ensure_archiver(directory, kwargs.get("sync_evidence"))
        batch = FileBatch(self.archive_roots)
        old_writer = GE._create_identical_or_fail
        record = None
        try:
            # This process owns one synchronous GT request. No numerical or
            # policy thread runs here. All source validation remains original;
            # only its evidence sink becomes immutable queued bytes.
            GE._create_identical_or_fail = batch.add
            with self.depth.request_scope(identity=kwargs["identity"], depth_image=kwargs["depth_image"]):
                if self.owner._complete_producer is None:
                    self.owner._complete_producer = self.producer_type(static_catalogue_dir=self.catalogue,
                        experiment_id="live_parity_"+Path(self.semantic_attempt).name,
                        ego_id=int(self.owner.ego.id), dynamic_population_ids=population["registered_population_ids"],
                        population_evidence_sha256=population["population_sha256"], max_frames=self.max_frames)
                producer = self.owner._complete_producer
                if producer.population != frozenset(population["registered_population_ids"]):
                    producer.register_population(population["registered_population_ids"],
                        population["population_sha256"], before_frame_id=frame)
                record = self.avo_write(root, **kwargs)
                avo_document = json.loads(batch.files[record["path"]])
                inputs = producer.build(**{k: kwargs[k] for k in ("identity", "rgb", "semantic_image", "depth_image",
                    "frozen_world", "camera_matrix", "camera_inverse")}, radar_world_xyz=radar,
                    dynamic_snapshot_evidence=population)
                prepared = self.prepare(inputs)
                handoff = self.publish_timed(root, identity=kwargs["identity"], inputs=inputs,
                                            prepared=prepared, avo_document=avo_document)
                desc, sources, ready = self.serialize_timed(batch, root, identity=kwargs["identity"],
                    inputs=inputs, prepared=prepared, evidence=producer.last_evidence,
                    rgb=kwargs["rgb"], radar=radar)
                receipt = self.enqueue_timed(batch, identity=kwargs["identity"], ready_paths=(ready,))
                result = {**record, "complete_identity": dict(kwargs["identity"]),
                    "complete_target": desc, "complete_sources": sources,
                    "scoring_handoff": handoff, "archive_receipt": receipt}
                self.pending_records.append(result)
                return result
        except BaseException as exc:
            # Preserve observed/partial AVO inputs even on a failed validator.
            # This is failed infrastructure evidence, never a scoring default.
            rejection = dict(schema=C.SCHEMA, identity=kwargs["identity"],
                status="FAILED", error=f"{type(exc).__name__}: {exc}", observed_raw_ns=W.raw(),
                scoring_packet_admitted=any(x["identity"] == kwargs["identity"] for x in self.handoffs))
            try:
                rejection["rejected_sources"] = self._rejected_sources(batch, kwargs, radar)
                batch.add(self.attempt/f"rejected_{frame}.json", W.canonical(rejection))
            except Exception as diagnostic_error:
                rejection["diagnostic_preservation_error"] = f"{type(diagnostic_error).__name__}: {diagnostic_error}"
            try:
                receipt = self.archiver.submit(dict(kwargs["identity"]), batch.files)
                self.failed_batches.append(dict(identity=dict(kwargs["identity"]), job_id=receipt,
                    original_error=rejection["error"], rejection=rejection))
            except Exception as archival_error:
                # Immutable source/sync diagnostics remain available to this
                # process; never report preservation when admission failed.
                self.failed_batches.append(dict(identity=dict(kwargs["identity"]),
                    archive_error=f"{type(archival_error).__name__}: {archival_error}",
                    original_error=rejection["error"], unpreserved_bytes=batch.bytes, rejection=rejection))
            if record is not None:
                exc.avo_evidence_record = {**record, "error": rejection["error"]}
            raise
        finally:
            GE._create_identical_or_fail = old_writer

    def close(self):
        if self.closed:
            return self.close_result
        archive_error = None
        if self.archiver is not None:
            try:
                self.archive_result = self.archiver.close()
                resolve_records(self.pending_records, self.archive_result["completions"])
            except Exception as exc:
                archive_error = exc
                self.archive_result = self.archiver.close_result
        result = super().close()
        full = dict(binding=self.handoff_binding, scoring_packets=list(self.handoffs), archive=self.archive_result,
                    failed_batches=list(self.failed_batches))
        encoded = W.canonical(full)
        T._write_x(self.attempt/"HANDOFF_ARCHIVE_LEDGER.json", encoded)
        archive_compact = None if self.archive_result is None else {
            k: v for k, v in self.archive_result.items() if k not in ("submissions", "completions")}
        if archive_compact is not None:
            archive_compact["completions"] = [{k: v for k, v in row.items() if k != "written"}
                                              for row in self.archive_result["completions"]]
        result.update(handoff_schema=C.SCHEMA, binding=self.handoff_binding,
            ledger=dict(path=str(self.attempt/"HANDOFF_ARCHIVE_LEDGER.json"),
                        sha256=hashlib.sha256(encoded).hexdigest(), packets=len(self.handoffs)),
            archive=archive_compact, failed_batches=list(self.failed_batches),
            archive_drained=archive_error is None and self.archiver is not None)
        # Small summary after drain is not on any action deadline.
        T._write_x(self.attempt/"HANDOFF_DRAIN.json", W.canonical(result))
        if archive_error is not None:
            raise archive_error
        return result
