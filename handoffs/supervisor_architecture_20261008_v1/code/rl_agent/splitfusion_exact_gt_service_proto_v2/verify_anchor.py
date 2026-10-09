"""Anchor-binding parity proof on all 49 sealed v11 frames (CPU only, offline, no timing).

For every frame one of the seven screen anchors is selected (round robin). Reference = the retained
runtime producer computing directly on the anchor-labelled selected capture (STAGE_SOURCE ->
PREPARE_STAGED -> CONFIRM_SEND -> PUBLISH_READY -> offline archive). Candidate = the v2 deferred
engine: prepare_unbound on the action-neutral template -> bind (anchor provenance) -> confirm_send ->
publish_ready -> retirement, then the v2 ProtoScorer: admit_target(+provenance) / admit_prediction
(anchor identity), alternating arrival order, exact scoring, result delivery, scorer archive and
durable retirement.

Proves for identical inputs: segmentation/IGNORE masks, AVO/eligibility/object tables, history
before/after, measurement and Q_perc identical; enumerates every metadata/digest difference and
requires each to be an identity-bearing field. Also checks that neutralising the selected capture
reproduces the source identity and that the mapping binds the admitted target.

    cd abiodun && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        /usr/bin/python3 -m rl_agent.splitfusion_exact_gt_service_proto_v2.verify_anchor --name anchor_parity_01
"""
from __future__ import annotations

import argparse
import collections
import json

import numpy as np

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_numeric_gt_integration_v2 import ready as R, scoring as Q
from rl_agent.splitfusion_live_parity_runtime_v1 import pose as PO
from rl_agent.splitfusion_numeric_gt_runtime_v1 import staging as ST

from . import parity_common as P
from .deferred import validate_anchor_provenance
from .scorer_core import ProtoScorer, expected_from_identity

ALLOWED_SUFFIXES = ("/action_id", "/profile_id", "/anchor_action_id")


def as_bytes(value):
    return value if isinstance(value, bytes) else str(value).encode()


def allowed(path):
    """Identity fields, digests of identity-bearing content, or byte sizes of identity-bearing files."""
    parts = [p.split("[")[0] for p in path.split("/")]
    if path.endswith(ALLOWED_SUFFIXES):
        return True
    if any("sha256" in part for part in parts):           # digest values (incl. evidence_sha256/<name>)
        return True
    return parts[-1] == "bytes" and "files" in parts      # serialized length of an identity-bearing file


def pinned_score(corpus, target, identity, case):
    pred = corpus.predictions[case.frame_id]
    scorer = Q.NumericalScorer()
    prediction = Q.admit_prediction(scorer, identity=identity, labels=pred["labels"], records=pred["records"],
                                    prediction_contract=corpus.contract)
    result = scorer.score(Q.join_admitted_inputs(scorer, target=target, prediction=prediction))
    value = corpus.quality.evaluate_exact_quality(corpus.spec, result.measurement)
    return result.measurement, None if value is None else float(value.q_perc)


def run(name):
    P.environment_ok()
    corpus = P.Corpus(49)
    output = P.create_output(name)
    P.write_json(output / "PLAN.json", dict(schema="splitfusion.exact_gt_service_proto_v2.anchor_parity_plan.v1",
        frames=len(corpus.cases), anchors=list(P.SCREEN_ANCHORS), assignment="round robin by frame index",
        reference="retained runtime producer on the anchor-labelled selected capture",
        candidate="v2 deferred engine + anchor provenance + v2 ProtoScorer, alternating arrival order",
        allowed_difference_rule="leaf ends in /action_id, /profile_id or /anchor_action_id; or any path component is a digest (contains sha256); or a files/<name>/bytes size of an identity-bearing file",
        source_sha256=corpus.sources, timing_recorded=False, live=False, gpu=False))
    rows, failures, differences = [], [], collections.Counter()
    # The pinned producer owner is process-wide: reference first (all frames), then candidate.
    ref = P.AnchorReferenceDriver(corpus)
    reference = {}
    try:
        for index, case in enumerate(corpus.cases):
            anchor = P.SCREEN_ANCHORS[index % len(P.SCREEN_ANCHORS)]
            reference[case.frame_id] = ref.run_anchor(case, P.anchor_variant(corpus, case, anchor))
    finally:
        ref.close()
    cand = P.DeferredDriver(corpus)
    scorer = ProtoScorer(run_binding=corpus.binding, producer_binding=corpus.startup["binding_sha256"], max_jobs=4,
                         max_bytes=32 * 1024 * 1024, reservation_bytes=8 * 1024 * 1024, max_waiting=2, max_frames=400,
                         spec=corpus.spec, quality=corpus.quality)
    try:
        for index, case in enumerate(corpus.cases):
            anchor = P.SCREEN_ANCHORS[index % len(P.SCREEN_ANCHORS)]
            variant = P.anchor_variant(corpus, case, anchor)
            r = reference[case.frame_id]
            # ---- candidate producer
            before = P.history(cand.engine)
            handle = cand.unbound(case)
            prepared = cand.engine.bind(handle, variant["plan_bytes"], frame_context=corpus.context(case))
            provenance = cand.engine.provenance(prepared.token)
            committed = cand.engine.confirm_send(prepared.token, variant["plan_bytes"])
            packet = cand.engine.publish_ready(prepared.token)
            cand_identity = C._parse(prepared.identity_bytes)
            mapping_ok = (provenance is not None and validate_anchor_provenance(provenance, target_identity=cand_identity) is not None
                          and C.canonical(ST.neutral_identity(variant["capture"])) == C.canonical(cand_identity["capture_identity"])
                          and provenance["selected_capture"] == variant["capture"]
                          and provenance["catalogue"]["action_id"] == anchor)
            # ---- both targets through the unchanged admission
            ref_value, ref_expected = expected_from_identity(as_bytes(r["identity_bytes"]))
            cand_value, cand_expected = expected_from_identity(prepared.identity_bytes)
            t_ref = R.decode_target(r["packet"], **ref_expected)
            t_cand = R.decode_target(packet, **cand_expected)
            meta_ref, meta_cand = C.thaw(t_ref.metadata), C.thaw(t_cand.metadata)
            meta_diff = P.json_diff(meta_ref, meta_cand, "target_metadata")
            identity_diff = P.json_diff(ref_value, cand_value, "target_identity")
            commit_diff = P.json_diff(r["commit"], committed, "commit")
            # ---- candidate retirement (offline archive) and archive comparison
            retired = cand.retire(case, prepared)
            archive_diff = []
            for fname in sorted(set(r["archive_files"]) | set(retired["archive_raw"])):
                a, b = r["archive_files"].get(fname), retired["archive_raw"].get(fname)
                if a is None or b is None:
                    archive_diff.append(f"archive/{fname}:missing")
                elif a != b:
                    try:
                        archive_diff += P.json_diff(json.loads(a), json.loads(b), f"archive/{fname}")
                    except (ValueError, UnicodeDecodeError):
                        archive_diff.append(f"archive/{fname}:binary")
            # ---- scoring: reference pinned join vs v2 scorer path (alternating order) + retirement
            m_ref, q_ref = pinned_score(corpus, t_ref, variant["prediction_identity"], case)
            pred = corpus.predictions[case.frame_id]
            labels = pred["labels"].tobytes(order="C")
            context_doc = PO.context_document(corpus.context(case))
            if index % 2 == 0:
                admitted = scorer.admit_target(prepared.identity_bytes, prepared.semantic_sha256, packet,
                                               provenance=C.canonical(provenance).decode())
                assert admitted["joined_result"] is None
                answer = scorer.admit_prediction(variant["edge_identity"], context_doc, pred["records"], labels)["result"]
                order = "target_first"
            else:
                waiting = scorer.admit_prediction(variant["edge_identity"], context_doc, pred["records"], labels)
                assert waiting["status"] == "WAITING_FOR_TARGET"
                answer = scorer.admit_target(prepared.identity_bytes, prepared.semantic_sha256, packet,
                                             provenance=C.canonical(provenance).decode())["joined_result"]
                order = "prediction_first"
            scorer.result_delivered(prepared.identity_bytes)
            files = scorer.archive_build(prepared.identity_bytes)
            proof = scorer.archive_durable(prepared.identity_bytes, P.harness_receipt(
                dict(cand_identity["capture_identity"], evidence_role="scorer"), f"frame_{case.frame_id}/scorer", files),
                f"frame_{case.frame_id}/scorer")
            join_doc = json.loads(files["anchor_provenance.json"])
            checks = dict(
                mapping=mapping_ok,
                segmentation_identical=t_ref.segmentation_bytes == t_cand.segmentation_bytes,
                ignore_identical=t_ref.object_ignore_bytes == t_cand.object_ignore_bytes,
                avo_and_eligibility_identical=C.canonical(meta_ref["avo"]) == C.canonical(meta_cand["avo"]),
                object_boxes_identical=C.canonical(meta_ref["object_boxes"]) == C.canonical(meta_cand["object_boxes"]),
                history_identical=(r["history_before"], r["history_after"]) == (before, retired.get("history_after", P.history(cand.engine))),
                measurement_identical=C.canonical(m_ref) == C.canonical(answer["measurement"]),
                q_perc_identical=q_ref == answer["q_perc"],
                q_perc_equals_sealed_v11=answer["q_perc"] == pred["q_perc"],
                scorer_join_archived=join_doc["join_mode"] == "ANCHOR_PROVENANCE_V1"
                    and join_doc["prediction_identity_as_received"] == variant["edge_identity"]
                    and join_doc["join_identity"]["gt_identity"] == cand_identity["capture_identity"],
                scorer_retired=proof["status"] == "DURABLE" and proof["scored"] is True,
                differences_only_identity_fields=all(allowed(p) for p in meta_diff + identity_diff + commit_diff + archive_diff))
            for p in meta_diff + identity_diff + commit_diff + archive_diff:
                differences[p] += 1
            row = dict(frame_id=case.frame_id, anchor=anchor, order=order, checks=checks, ok=all(checks.values()),
                       semantic_digest_differs=t_ref.semantic_sha256 != t_cand.semantic_sha256,
                       q_perc=answer["q_perc"], differences=sorted(meta_diff + identity_diff + commit_diff + archive_diff))
            rows.append(row)
            if not row["ok"]:
                failures.append(row)
                break
        snapshot = scorer.snapshot()
        result = dict(schema="splitfusion.exact_gt_service_proto_v2.anchor_parity_result.v1",
            status="ANCHOR_BINDING_EXACT" if not failures and len(rows) == len(corpus.cases)
                   and snapshot["retained_slots"] == 0 and snapshot["waiting_predictions"] == 0 else "FAIL",
            frames=len(rows), anchors_exercised=sorted({r["anchor"] for r in rows}),
            per_anchor=dict(collections.Counter(r["anchor"] for r in rows)),
            orders=dict(collections.Counter(r["order"] for r in rows)),
            undefined_quality=sum(r["q_perc"] is None for r in rows),
            legitimate_differences=dict(sorted(differences.items())),
            numerical_fields_compared=["segmentation_u8", "object_ignore_u8", "avo (qualified/ignored/structural, eligibility)",
                                       "object_boxes", "history before/after", "measurement", "q_perc", "q_perc vs sealed v11"],
            semantic_digest_differs_frames=sum(r["semantic_digest_differs"] for r in rows),
            semantic_digest_note="target semantic digest covers identity metadata, so it differs by design; masks/AVO/boxes compared directly",
            scorer_snapshot=snapshot, deferral=cand.engine.snapshot()["deferral"],
            timing_recorded=False, performance_claim=False, live_qualified=False)
        P.write_json(output / "RESULT.json", result)
        print(json.dumps({k: result[k] for k in ("status", "frames", "per_anchor", "orders", "legitimate_differences")},
                         indent=1)[:4000])
        if result["status"] != "ANCHOR_BINDING_EXACT":
            raise AssertionError("anchor parity failure: " + json.dumps(failures[:1], default=str)[:2000])
    finally:
        try:
            cand.close()
        finally:
            with (output / "FRAMES.jsonl").open("x") as stream:
                for row in rows:
                    stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            P.seal_output(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    run(parser.parse_args().name)
