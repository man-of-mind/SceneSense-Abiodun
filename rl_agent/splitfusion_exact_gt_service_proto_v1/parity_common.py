"""Shared offline drivers for exact parity on the sealed 49 v11 frames.

Reference = the deployed runtime ``process.Engine`` producer role, driven
in-process through STAGE_SOURCE / PREPARE_STAGED / CONFIRM_SEND / PUBLISH_READY
and its offline archive lifecycle. Candidate = ``DeferredProducerEngine``.
Archive durability receipts here are structural harness receipts (no fsync);
they exercise retirement, not durability. No timing is recorded or claimed.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_numeric_gt_integration_v2 import replay as H, fixtures as F, ready as R, scoring as Q
from rl_agent.splitfusion_numeric_gt_producer_v1 import cases as K, artifacts as AR
from rl_agent.splitfusion_numeric_gt_producer_v1.replay_v11 import KEYS, quality_bridge
from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E
from rl_agent.splitfusion_live_parity_runtime_v1 import pose as PO
from rl_agent.splitfusion_numeric_gt_runtime_v1 import process as RP, staging as ST
from rl_agent.splitfusion_numeric_gt_runtime_v1.scorer import registration_key

from .deferred import DeferredProducerEngine

ROOT = K.ROOT
PACKAGE = Path(__file__).resolve().parent
EVIDENCE_ROOT = ROOT / "experiments/splitfusion_exact_gt_service_proto_v1"
MAX_FRAMES = 100


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def source_inventory():
    paths = sorted(PACKAGE.glob("*.py"))
    for package in ("splitfusion_numeric_gt_runtime_v1", "splitfusion_numeric_gt_integration_v2",
                    "splitfusion_numeric_gt_producer_v1", "splitfusion_live_gt_optimization_v4"):
        paths += sorted((ROOT / "rl_agent" / package).glob("*.py"))
    return {str(p.relative_to(ROOT)): H.sha(p) for p in paths}


def create_output(name):
    output = (EVIDENCE_ROOT / name).absolute()
    C.require(not output.exists() and output.resolve().is_relative_to(EVIDENCE_ROOT)
              and not any(p.is_symlink() for p in output.parents), "create-only evidence directory required")
    output.mkdir(parents=True, exist_ok=False)
    return output


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=1, allow_nan=False)
        stream.write("\n")


def seal_output(output):
    inventory = {str(p.relative_to(output)): H.sha(p) for p in sorted(output.rglob("*"))
                 if p.is_file() and p.name != "OUTPUT_SHA256.json"}
    write_json(output / "OUTPUT_SHA256.json", inventory)


class Corpus:
    """Loaded once; every driver reads the same immutable fixture values."""

    def __init__(self, frames=49):
        self.data = H.load_inputs(frames)
        self.cases = self.data["cases"]
        self.predictions = self.data["predictions"]
        self.contract = self.data["prediction_contract"]
        self.startup = self.data["startup"]
        self.sources = source_inventory()
        self.binding = C.sha256(C.canonical(dict(scope="EXACT_GT_SERVICE_PROTO_V1_OFFLINE",
            sources=self.sources, input_seal=K.INPUT_SEAL_SHA)))
        self.spec, self.quality = quality_bridge()

    def producer_args(self):
        return {k: str(v) if isinstance(v, Path) else v
                for k, v in K.producer_arguments(self.cases[0]).items()}

    def engine_config(self, *, offline=True):
        config = dict(run_binding=self.binding, producer_binding=self.startup["binding_sha256"],
                      prediction_contract=self.contract, max_frames=MAX_FRAMES,
                      producer_args=self.producer_args(), offline_replay=offline)
        if not offline:
            config["live_process_contract"] = RP.LIVE_PROCESS_CONTRACT
        return config

    def expected(self, case):
        return H.expectation(self.data, case, self.binding)

    def plan(self, case):
        return F.plan_for(case, self.predictions[case.frame_id], self.data["contexts"][case.frame_id]["context"])

    def context(self, case):
        return self.data["contexts"][case.frame_id]["context"]

    def sourcekwargs(self, case):
        return {k: case.kwargs[k] for k in KEYS}

    def staged_packet(self, case):
        chunks, extras = ST.encode_stage(self.sourcekwargs(case), case.kwargs["policy_timestamp_s"])
        return b"".join(chunks), extras

    @staticmethod
    def offline_inputs(case):
        clocks = {k: case.avo_document[k] for k in AR.AVO_CLOCKS if k != "sync_observed_raw_ns"}
        clocks["sync_observed_raw_ns"] = case.avo_prescore["source_synchronization"]["observed_raw_ns"]
        return dict(archive_clocks=clocks,
                    payload_filenames={k: v["filename"] for k, v in case.avo_document["payloads"].items()},
                    prescore_filename=case.avo_document["prescore_input"]["filename"],
                    replay_observed_raw_ns=case.evidence["source_synchronization"]["observed_raw_ns"])


def harness_receipt(capture, prefix, files):
    """Structural receipt only; parity drivers never claim durability."""
    return dict(status="COMPLETE", durable=True, error=None, identity=capture, job_id=1,
                written=[dict(root=0, relative=f"{prefix}/{name}", sha256=sha(raw), bytes=len(raw), ready=False)
                         for name, raw in files.items()])


def score_exact(corpus, case, packet):
    """Pinned admission + scorer on the published packet, against retained truth."""
    expected = corpus.expected(case)
    target = R.decode_target(packet, **expected)
    pred = corpus.predictions[case.frame_id]
    scorer = Q.NumericalScorer()
    prediction = Q.admit_prediction(scorer, identity=pred["identity"], labels=pred["labels"],
                                    records=pred["records"], prediction_contract=corpus.contract)
    result = scorer.score(Q.join_admitted_inputs(scorer, target=target, prediction=prediction))
    value = corpus.quality.evaluate_exact_quality(corpus.spec, result.measurement)
    q_perc = None if value is None else float(value.q_perc)
    original = E.evaluate_prepared(identity=pred["identity"]["action_identity"], prepared=case.bundle.prepared,
        predicted_labels=pred["labels"], predictions=pred["records"], avo=case.avo_document["avo"])
    return dict(q_perc=q_perc, q_perc_exact=q_perc == pred["q_perc"],
                measurement_exact=C.canonical(result.measurement) == C.canonical(original.measurement),
                measurement_sha256=sha(C.canonical(result.measurement)),
                identity_key_exact=target.identity_bytes == registration_key(expected))


def archive_checks(case, files, descriptor):
    return dict(segmentation_png=files["segmentation.png"] == case.segmentation_png,
                ignore_png=files["object_ignore.png"] == case.object_ignore_png,
                prepared_digest=descriptor["old_prepared_sha256"] == case.ready["prepared_content_sha256"],
                avo_document=sha(files["avo.json"]) == case.avo_sha256,
                avo_prescore=sha(files[case.avo_document["prescore_input"]["filename"]]) == case.prescore_sha256)


def history(engine_like):
    producer = engine_like._owner._producer
    return producer.history_sha256()


class ReferenceDriver:
    """Deployed runtime producer role, in-process, offline replay mode."""

    def __init__(self, corpus):
        self.corpus = corpus
        self.engine = RP.Engine("producer", corpus.engine_config(offline=True))
        self.job = 0

    @property
    def inner(self):
        return self.engine.engine

    def run(self, case):
        c = self.corpus
        self.job += 1
        job = self.job
        packet, extras = c.staged_packet(case)
        before = history(self.inner)
        self.engine.call("STAGE_SOURCE", dict(job=job, source_packet=packet, stage_extras=extras))
        request = dict(job=job, plan_bytes=c.plan(case), capture_identity=case.kwargs["identity"],
                       frame_context=PO.context_document(c.context(case)),
                       population=case.evidence["dynamic_snapshot_evidence"], **c.offline_inputs(case))
        prepared = self.engine.call("PREPARE_STAGED", request)
        committed = self.engine.call("CONFIRM_SEND", dict(job=job, plan_bytes=c.plan(case)))
        ready = self.engine.call("PUBLISH_READY", dict(job=job))
        row = self.retire(case, job, prepared)
        return dict(row, history_before=before, identity_bytes=prepared["identity_bytes"],
                    semantic_sha256=prepared["semantic_sha256"], commit=committed, packet=ready,
                    history_after=history(self.inner))

    def retire(self, case, job, prepared):
        note = dict(job=job, identity_bytes=prepared["identity_bytes"], semantic_sha256=prepared["semantic_sha256"])
        self.engine.call("SCORE_COMPLETED", note)
        self.engine.call("FEEDBACK_CONFIRMED", note)
        archive = self.engine.call("BUILD_ARCHIVE", dict(job=job))
        prefix = f"frame_{case.frame_id}/producer"
        self.engine.call("SCORER_ARCHIVE_COMPLETED", note)
        self.engine.call("ARCHIVE_DURABLE", dict(job=job, relative_prefix=prefix,
            receipt=harness_receipt(case.kwargs["identity"], prefix, archive["files"])))
        return dict(archive_files={k: sha(v) for k, v in archive["files"].items()},
                    archive_descriptor=json.loads(archive["descriptor_bytes"]), archive_raw=archive["files"])

    def close(self):
        return self.engine.call("CLOSE", {})


class DeferredDriver:
    """Candidate: prepare_unbound on the decoded staged template, bind later."""

    def __init__(self, corpus, *, source="staged"):
        self.corpus = corpus
        config = corpus.engine_config(offline=True)
        self.engine = DeferredProducerEngine(**config)
        self.source = source

    def decoded(self, case):
        if self.source == "staged":
            packet, extras = self.corpus.staged_packet(case)
            return ST.decode_stage(packet, extras)
        raise ValueError("unknown source mode")

    def unbound(self, case, kwargs=None):
        kwargs = self.decoded(case) if kwargs is None else kwargs
        C.require(kwargs["identity"] == ST.neutral_identity(case.kwargs["identity"]), "decoded template differs")
        return self.engine.prepare_unbound(kwargs, population=case.evidence["dynamic_snapshot_evidence"],
                                           **self.corpus.offline_inputs(case))

    def run(self, case, *, action="accept", kwargs=None, refuse_plan=None):
        c = self.corpus
        before = history(self.engine)
        handle = self.unbound(case, kwargs)
        mid = history(self.engine)
        if action in ("hold", "fallback"):
            self.engine.cancel_unbound(handle, action.upper() + "_NEVER_BOUND")
            return dict(action=action, history_before=before, history_staged=mid, history_after=history(self.engine))
        if action == "refuse":
            try:
                self.engine.bind(handle, refuse_plan, frame_context=c.context(case))
            except Exception as error:   # DeferralRefused expected
                return dict(action=action, refused=type(error).__name__, history_before=before,
                            history_after=history(self.engine))
            raise AssertionError("non-template plan was not refused")
        prepared = self.engine.bind(handle, c.plan(case), frame_context=c.context(case))
        if action == "unsent":
            self.engine.cancel(prepared.token, c.plan(case), "SEND_FAILED_AFTER_BIND")
            return dict(action=action, history_before=before, history_after=history(self.engine))
        committed = self.engine.confirm_send(prepared.token, c.plan(case))
        packet = self.engine.publish_ready(prepared.token)
        row = self.retire(case, prepared)
        return dict(row, action="accept", history_before=before, identity_bytes=prepared.identity_bytes,
                    semantic_sha256=prepared.semantic_sha256, commit=committed, packet=packet,
                    history_after=history(self.engine))

    def retire(self, case, prepared):
        e, t = self.engine, prepared.token
        e.score_completed(t, prepared.identity_bytes, prepared.semantic_sha256)
        e.feedback_confirmed(t, prepared.identity_bytes, prepared.semantic_sha256)
        archive = e.build_archive(t)
        prefix = f"frame_{case.frame_id}/producer"
        e.scorer_archive_completed(t, prepared.identity_bytes, prepared.semantic_sha256)
        e.archive_durable(t, harness_receipt(case.kwargs["identity"], prefix, archive.files), relative_prefix=prefix)
        return dict(archive_files={k: sha(v) for k, v in archive.files.items()},
                    archive_descriptor=json.loads(archive.descriptor_bytes), archive_raw=dict(archive.files))

    def close(self):
        return self.engine.close()


def environment_ok():
    C.require(sys.executable == "/usr/bin/python3" and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
              "CPU-only registered interpreter required")
    C.require(np.__version__ == "1.26.4", "NumPy 1.26.4 required for bit-exact replay")
