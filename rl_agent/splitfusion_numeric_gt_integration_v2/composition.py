"""Actual component composition for offline parity and explicitly scoped timing.

Numerical readiness/scoring precedes all full AVO/PNG artifact construction.
The original actor/reward/deadline and live pipeline are not installed here.
Send and feedback acknowledgements below are deliberate harness events, not
network measurements. Serialized wire encoding/admission is real; OS IPC,
process scheduling, archive durability and the live early-plan hook are not.
"""
from contextlib import ExitStack, contextmanager
import json
import time
from unittest.mock import patch

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C, evidence as V
from rl_agent.splitfusion_numeric_gt_producer_v1 import artifacts as AR, avo as NA, payload as P, target as NT
from rl_agent.splitfusion_numeric_gt_producer_v1.replay_v11 import KEYS
from rl_agent.splitfusion_minimal_quality_scorer_v1 import scorer as S
from rl_agent.splitfusion_minimal_quality_scorer_v1.validate import forbidden, guarded_score
from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E, target_transport as T
from rl_agent.splitfusion_run5_live_avo_v1 import gt_source_sync as SY
from . import ready as R, scoring as Q, fixtures as F

RESERVATION = 64 * 1024 * 1024


class Meter:
    def __init__(self):
        self.spans = {}

    @contextmanager
    def span(self, name):
        C.require(name not in self.spans, "duplicate timing boundary")
        raw, cpu = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW), time.thread_time_ns()
        try:
            yield
        finally:
            self.spans[name] = dict(wall_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW) - raw,
                                   thread_cpu_ns=time.thread_time_ns() - cpu)

    def call(self, name, function):
        with self.span(name):
            return function()


@contextmanager
def fixture_clock(case):
    original = SY.source_alignment_record
    def alignment(*args, **kwargs):
        record = original(*args, **kwargs)
        record["observed_raw_ns"] = case.evidence["source_synchronization"]["observed_raw_ns"]
        return record
    with patch.object(SY, "source_alignment_record", alignment):
        yield


def register_population(owner, case):
    population = case.evidence["dynamic_snapshot_evidence"]
    if owner.population != frozenset(population["registered_population_ids"]):
        owner.register_population(population["registered_population_ids"], population["population_sha256"],
                                  before_frame_id=case.frame_id)


def assemble(staged, kwargs):
    numeric = NT.numeric_target(frame=staged.frame, split="val", depth=staged.depth, semantic=staged.semantic,
        original_mask=staged.mask, boxes=staged.boxes, visibility=staged.visibility, static_rows=staged.static,
        quarantine=staged.conflict["quarantine"], vehicle_mask=staged.vehicle_mask)
    avo = NA.numeric_avo(identity=kwargs["identity"], rgb=kwargs["rgb"], semantic_image=kwargs["semantic_image"],
        depth_image=kwargs["depth_image"], frozen_world=kwargs["frozen_world"], camera_matrix=staged.camera_matrix,
        camera_inverse=staged.camera_inverse, depth=staged.depth, sync_validated=True)
    return numeric, avo, P.build_payload(staged=staged, target=numeric, avo=avo)


def candidate(case, prediction, *, owner, scorer, join, expected, spec, quality, guarded=False, index=0):
    """One actual numerical transaction, then deferred exact evidence.

    Alternate successful send-confirmation order and input-admission order.
    Those are offline events only; no live network/timer policy is substituted.
    """
    from .ownership import ProducerOwner
    C.require(type(owner) is ProducerOwner, "actual bounded producer owner required")
    meter = Meter()
    kwargs = {key: case.kwargs[key] for key in KEYS}
    plan = F.plan_for(case, prediction, expected["frame_context"])
    with meter.span("candidate_pre_archive_total"), fixture_clock(case), ExitStack() as guards:
        if guarded:
            for name in ("legacy_target", "prepared_target", "avo_documents"):
                guards.enter_context(patch.object(AR, name, forbidden))
            for name in ("imencode", "imdecode", "imread"):
                guards.enter_context(patch("cv2." + name, forbidden))
        register_population(owner, case)
        token = meter.call("source_compute_and_ownership", lambda: owner.compute(plan, kwargs, reserved_bytes=RESERVATION))
        staged = owner._staged_for_assembly(token)
        numeric, avo, payload = meter.call("numeric_target_avo_payload", lambda: assemble(staged, kwargs))
        meter.call("products_ownership", lambda: owner.stage_products(token, target=numeric, avo=avo, payload=payload))
        if index % 2:
            C.require(owner.confirm_send(token, plan) is None, "send alone must not commit history")
        admitted = meter.call("target_admission_with_pose", lambda: R.admit_payload(payload, **expected))
        committed = meter.call("ready_history_admission", lambda: owner.set_ready(token, admitted))
        if not index % 2:
            C.require(committed is None, "numerical admission alone must not commit history")
            committed = meter.call("send_confirmation_commit", lambda: owner.confirm_send(token, plan))
        C.require(committed["history_after_sha256"] == payload.history_after_sha256, "real history commit differs")
        admitted = owner.ready_target(token)
        meter.call("join_registration", lambda: join.register(admitted.identity_bytes, reserved_bytes=RESERVATION))

        def prediction_admission():
            return Q.admit_prediction(scorer, identity=prediction["identity"], labels=prediction["labels"],
                records=prediction["records"], prediction_contract=expected["prediction_contract"])
        if index % 2:
            pred = meter.call("prediction_admission", prediction_admission)
            meter.call("prediction_arrival", lambda: join.put_prediction(admitted.identity_bytes, pred))
        packet = meter.call("wire_encode", lambda: R.encode_target(admitted))
        received = meter.call("wire_decode_with_pose", lambda: R.decode_target(packet, **expected))
        C.require(received.semantic_sha256 == admitted.semantic_sha256, "wire target differs")
        meter.call("target_arrival", lambda: join.put_target(received))
        if not index % 2:
            pred = meter.call("prediction_admission", prediction_admission)
            meter.call("prediction_arrival", lambda: join.put_prediction(admitted.identity_bytes, pred))
        pair = meter.call("join_take", lambda: join.wait_take(admitted.identity_bytes, timeout=0))
        C.require(pair is not None, "both admitted inputs must wake join")
        frame = meter.call("admitted_join", lambda: Q.join_admitted_inputs(scorer, target=pair.target, prediction=pair.prediction))
        if guarded:
            result, matching = meter.call("numerical_score", lambda: guarded_score(scorer, frame))
        else:
            result = meter.call("numerical_score", lambda: scorer.score(frame))
            matching = None
        value = meter.call("quality_formula", lambda: quality.evaluate_exact_quality(spec, result.measurement))
        qperc = None if value is None else float(value.q_perc)
        owner.complete_scoring(token)
        owner.confirm_feedback(token)  # Harness acknowledgement; not a UE receipt.
    # Nothing below belongs to numerical readiness or the feedback callback.
    C.require(owner.snapshot()["retained_jobs"] >= 1, "archive ownership released before completion")
    with meter.span("deferred_archive_total"):
        staged, numeric, avo, payload = owner._archive_for_owner(token)
        legacy = meter.call("archive_legacy_target", lambda: AR.legacy_target(staged, numeric))
        prepared = meter.call("archive_prepared_validator", lambda: AR.prepared_target(staged, legacy))
        clocks = {k: case.avo_document[k] for k in AR.AVO_CLOCKS if k != "sync_observed_raw_ns"}
        clocks["sync_observed_raw_ns"] = case.avo_prescore["source_synchronization"]["observed_raw_ns"]
        documents = meter.call("archive_avo_documents", lambda: AR.avo_documents(staged, avo,
            identity=kwargs["identity"], clocks=clocks,
            payload_filenames={k: v["filename"] for k, v in case.avo_document["payloads"].items()},
            prescore_filename=case.avo_document["prescore_input"]["filename"]))
        source = meter.call("archive_source_admission", lambda: V.admit_source_evidence(
            input_metadata=json.loads(P.A.canonical(T._input_metadata(staged.inputs))),
            avo_document=json.loads(documents["document_bytes"]), expected_capture_identity=kwargs["identity"]))
        reference = meter.call("archive_binding", lambda: Q.attach_archive(received, source_evidence=source,
            target_tables=legacy["tables"], old_prepared_sha256=prepared.content_sha256, prepared_target=prepared))
        with ExitStack() as guards:
            if guarded:
                guards.enter_context(patch.dict(scorer._detection, match_frame=forbidden, load_gt=forbidden,
                                                load_predictions=forbidden))
                guards.enter_context(patch.dict(scorer._person, greedy_match=forbidden))
            evidence = meter.call("archive_evidence_from_associations", lambda: Q.reconstruct_evidence(result,
                archive_reference=reference))
            archived = meter.call("archive_payloads_from_associations", lambda: Q.archive_payloads(result,
                archive_reference=reference))
        owner.complete_archive(token)
        join.release(admitted.identity_bytes)
    return dict(frame_id=case.frame_id, q_perc=qperc, measurement=result.measurement, evidence=evidence,
        archived=archived, prepared=prepared, legacy=legacy, documents=documents, packet_bytes=len(packet),
        producer_semantic_sha256=payload.semantic_sha256, ready_semantic_sha256=received.semantic_sha256,
        history_before_sha256=payload.history_before_sha256, history_after_sha256=payload.history_after_sha256,
        timings=meter.spans, matching_calls=matching, owner_snapshot=owner.snapshot(), join_snapshot=join.snapshot(),
        archive_absent_before_scoring=guarded, acknowledgement_scope="OFFLINE_HARNESS_NOT_NETWORK_RECEIPT")


def check_exact(value, case, prediction):
    oracle = E.evaluate_prepared(identity=prediction["identity"]["action_identity"], prepared=case.bundle.prepared,
        predicted_labels=prediction["labels"], predictions=prediction["records"], avo=case.avo_document["avo"])
    checks = dict(measurement=value["measurement"] == oracle.measurement,
        received_qperc=value["q_perc"] == prediction["q_perc"],
        evidence=value["evidence"] == prediction["evidence"],
        evidence_sha256=C.sha256(S.canonical(value["evidence"])) == prediction["evidence_sha256"],
        archive_payloads=value["archived"] == E.archive_payloads(oracle),
        prepared_digest=value["prepared"].content_sha256 == case.ready["prepared_content_sha256"],
        segmentation_png=value["legacy"]["segmentation_png"] == case.segmentation_png,
        object_ignore_png=value["legacy"]["object_ignore_png"] == case.object_ignore_png,
        avo_document=value["documents"]["document_sha256"] == case.avo_sha256,
        avo_prescore=value["documents"]["prescore_sha256"] == case.prescore_sha256,
        owner_valid=value["owner_snapshot"]["valid"],
        owner_drained=value["owner_snapshot"]["retained_jobs"] == value["owner_snapshot"]["reserved_bytes"] == 0,
        join_valid=value["join_snapshot"]["valid"],
        join_drained=value["join_snapshot"]["pending_jobs"] == value["join_snapshot"]["reserved_bytes"] == 0)
    C.require(all(checks.values()), "v2 component parity failed: " + str(case.frame_id) + " " + str(checks))
    return checks
