"""Step 3: prepared-target cache and scorer composition in separate processes.

Part A (live100): the 48 archived live100 ready targets are rebuilt from their
sealed archive bytes, admitted once into the scorer process cache and joined
with the archived native predictions. Every quality outcome, measurement and
the full numerical_score.json document must equal the archive byte-for-byte,
including undefined-quality frames. Alternate frames send the prediction before
the target (waiting path). Evidence is then made durable with the unchanged
ArchiveClient (real fsync) and every slot retires.

Part B (v11): all 49 frames end-to-end through BOTH processes: UE-role binary
freeze -> prep process (deferred binding, commit, publication) -> sealed packet
-> scorer process cache -> prediction -> score; then off-path evidence and real
durable retirement on both sides. Compared with the in-parent runtime reference
and the sealed v11 truth.

No duration is recorded. Correctness only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_numeric_gt_integration_v2 import ready as R
from rl_agent.splitfusion_live_parity_runtime_v1 import pose as PO
from rl_agent.splitfusion_live_gt_handoff_v1.archive import ArchiveClient
from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E

from . import parity_common as P, channel as CH, service as SV
from .scorer_core import expected_from_identity

NEU = P.ROOT.parent
LIVE100 = NEU / ("abiodun_run4_single_gpu_profiler_v1/rl_agent/experiments/splitfusion_single_gpu_profiler_v1/"
                 "20261005_numeric_gt_completion_live100_v1")
LIVE100_ARCHIVE = LIVE100 / "capture/cells/a71__favorable_stable/profiler/edge/numeric_runtime/archive/numeric_gt"
LIVE100_SEAL = P.ROOT / "experiments/splitfusion_numeric_gt_runtime_v1/completion_live100_result_01/SOURCE_EVIDENCE_SHA256.json"
PREDICTION_FIELDS = ("gt_identity", "action_identity", "capture_timestamp_ns", "prediction_digest")
FILES = ("target_metadata.json", "segmentation.u8", "object_ignore.u8", "predicted_labels.u8", "numerical_score.json")


def sealed_bytes(seal, path):
    raw = path.read_bytes()
    entry = seal["source_files"][str(path.relative_to(LIVE100))]
    C.require(entry["sha256"] == hashlib.sha256(raw).hexdigest() and entry["bytes"] == len(raw),
              "live100 input differs from its seal: " + str(path))
    return raw


def live100_frames():
    seal = json.loads(LIVE100_SEAL.read_bytes())
    C.require(seal["source_root"] == str(LIVE100), "live100 seal root differs")
    frames = []
    for directory in sorted(LIVE100_ARCHIVE.glob("frame_*")):
        raw = {name: sealed_bytes(seal, directory / name) for name in FILES}
        metadata = C._parse(raw["target_metadata.json"])
        score = C._parse(raw["numerical_score.json"])
        identity = C.canonical({k: metadata[k] for k in ("capture_identity", "action_identity", "plan_sha256",
            "run_binding", "producer_binding", "frame_context", "prediction_contract")})
        _, expected = expected_from_identity(identity)
        target = R._admit(metadata, raw["segmentation.u8"], raw["object_ignore.u8"], **expected)
        C.require(target.metadata_bytes == raw["target_metadata.json"]
                  and target.semantic_sha256 == score["ready_semantic_sha256"]
                  and C._parse(target.identity_bytes) == score["ready_identity"], "rebuilt live100 target differs")
        frames.append(dict(directory=directory.name, raw=raw, metadata=metadata, score=score,
                           identity=target.identity_bytes, semantic=target.semantic_sha256,
                           packet=R.encode_target(target)))
    C.require(len(frames) == 48, "48 archived live100 decisions required")
    return frames, seal


def part_a(output):
    frames, seal = live100_frames()
    run = {f["metadata"]["run_binding"] for f in frames}
    producer = {f["metadata"]["producer_binding"] for f in frames}
    C.require(len(run) == len(producer) == 1, "one live100 run/producer binding required")
    config = dict(run_binding=run.pop(), producer_binding=producer.pop(), max_jobs=4, max_bytes=4 * 8 * 1024 * 1024,
                  reservation_bytes=8 * 1024 * 1024, max_waiting=2, max_frames=100)
    directory = output / "part_a_runtime"
    directory.mkdir()
    (directory / "durable").mkdir()
    score = SV.WorkerClient("score", config, log_path=directory / "score.log")
    archive = ArchiveClient(roots=[str((directory / "durable").resolve())], max_jobs=4, max_bytes=256 * 1024 * 1024,
                            max_frames=100, binding=dict(scope="exact_gt_service_proto_v2_part_a"))
    harness = SV.Harness.__new__(SV.Harness)
    harness.archive, harness.archive_root, harness.score = archive, directory / "durable", score
    rows = []
    try:
        for index, f in enumerate(frames):
            s = f["score"]
            identity = {k: s["prediction_identity"][k] for k in PREDICTION_FIELDS}
            context = f["metadata"]["frame_context"]
            labels = f["raw"]["predicted_labels.u8"]
            prediction_first = bool(index % 2)
            if prediction_first:
                waiting = SV.ok(harness.predict(identity, context, s["prediction_records"], labels), "PREDICTION")
                C.require(waiting["status"] == "WAITING_FOR_TARGET", "prediction-first must wait")
            fd, size = CH.seal_bytes([f["packet"]])
            try:
                admitted = SV.ok(score.call("TARGET", dict(identity=f["identity"].decode(), semantic_sha256=f["semantic"],
                                                           size=size), fd=fd), "TARGET")
            finally:
                os.close(fd)
            if prediction_first:
                result = admitted["joined_result"]
            else:
                C.require(admitted["joined_result"] is None, "target alone must not score")
                result = SV.ok(harness.predict(identity, context, s["prediction_records"], labels), "PREDICTION")["result"]
            repeat = harness.predict(identity, context, s["prediction_records"], labels)
            C.require(repeat.status == "REFUSED" and repeat.reason == "DUPLICATE_PREDICTION", "duplicate must be refused")
            SV.ok(score.call("RESULT_DELIVERED", dict(identity=f["identity"].decode())), "RESULT_DELIVERED")
            sfiles = harness.files_of(score.call("ARCHIVE_BUILD", dict(identity=f["identity"].decode())))
            prefix = f"{f['directory']}/scorer"
            capture = f["metadata"]["capture_identity"]
            receipt = harness.durable(dict(capture, evidence_role="scorer"), prefix, sfiles)
            SV.ok(score.call("ARCHIVE_DURABLE", dict(identity=f["identity"].decode(), receipt=receipt,
                                                     relative_prefix=prefix)), "ARCHIVE_DURABLE")
            checks = dict(q_perc=result["q_perc"] == s["q_perc"], quality_defined=result["quality_defined"] == s["quality_defined"],
                          measurement=C.canonical(result["measurement"]) == C.canonical(s["measurement"]),
                          numerical_score_json_bytes=sfiles["numerical_score.json"] == f["raw"]["numerical_score.json"],
                          target_metadata_bytes=sfiles["target_metadata.json"] == f["raw"]["target_metadata.json"],
                          predicted_labels_bytes=sfiles["predicted_labels.u8"] == labels,
                          masks=sfiles["segmentation.u8"] == f["raw"]["segmentation.u8"]
                                and sfiles["object_ignore.u8"] == f["raw"]["object_ignore.u8"],
                          ready_semantic=result["ready_semantic_sha256"] == s["ready_semantic_sha256"])
            rows.append(dict(part="A_live100", frame=f["directory"], order="prediction_first" if prediction_first
                             else "target_first", quality_defined=s["quality_defined"], checks=checks, ok=all(checks.values())))
            if not rows[-1]["ok"]:
                raise AssertionError("Part A mismatch " + f["directory"] + " " + str(checks))
        snapshot = SV.ok(score.call("SNAPSHOT"), "SNAPSHOT")
        C.require(snapshot["retained_slots"] == 0 and snapshot["waiting_predictions"] == 0, "slots must all retire")
        closed = SV.ok(score.close(), "CLOSE")
        archive_summary = archive.close()
    except BaseException:
        score.abort(); archive.abort()
        raise
    return rows, dict(frames=len(rows), all_exact=all(r["ok"] for r in rows),
        undefined_quality=sum(not r["quality_defined"] for r in rows), scorer_snapshot=snapshot, scorer_closed=closed,
        durable_completions=archive_summary["completed"], live100_seal=str(LIVE100_SEAL))


def part_b(output):
    corpus = P.Corpus(49)
    ref = P.ReferenceDriver(corpus)
    reference = {c.frame_id: ref.run(c) for c in corpus.cases}
    ref.close()
    prep_config = dict(engine=corpus.engine_config(offline=True))
    score_config = dict(run_binding=corpus.binding, producer_binding=corpus.startup["binding_sha256"], max_jobs=4,
                        max_bytes=4 * 8 * 1024 * 1024, reservation_bytes=8 * 1024 * 1024, max_waiting=2, max_frames=100)
    h = SV.Harness(output / "part_b_runtime", prep_config=prep_config, score_config=score_config, archive_frames=200)
    rows = []
    try:
        for index, case in enumerate(corpus.cases):
            r = reference[case.frame_id]
            pred = corpus.predictions[case.frame_id]
            identity = {k: pred["identity"][k] for k in PREDICTION_FIELDS}
            context = PO.context_document(corpus.context(case))
            h.freeze(corpus.sourcekwargs(case), case.kwargs["policy_timestamp_s"], offline=corpus.offline_inputs(case))
            job = h.finish_freeze()["job"]
            plan = corpus.plan(case)
            bound = SV.ok(h.prep.call("BIND", dict(job=job, plan=plan.decode(), frame_context=context)), "BIND")
            committed = SV.ok(h.prep.call("CONFIRM_SEND", dict(job=job, plan=plan.decode())), "CONFIRM_SEND")["commit"]
            prediction_first = bool(index % 2)
            labels = pred["labels"].tobytes(order="C")
            if prediction_first:
                SV.ok(h.predict(identity, context, pred["records"], labels), "PREDICTION")
            reply = h.prep.call("PUBLISH", dict(job=job))
            published = SV.ok(reply, "PUBLISH")
            try:
                packet = CH.read_sealed(reply.fd, published["size"])
                target = SV.ok(h.score.call("TARGET", dict(identity=published["identity"],
                    semantic_sha256=published["semantic_sha256"], size=published["size"]), fd=reply.fd), "TARGET")
            finally:
                os.close(reply.fd)
            result = target["joined_result"] if prediction_first else \
                SV.ok(h.predict(identity, context, pred["records"], labels), "PREDICTION")["result"]
            SV.ok(h.score.call("RESULT_DELIVERED", dict(identity=published["identity"])), "score RESULT_DELIVERED")
            SV.ok(h.prep.call("RESULT_DELIVERED", dict(job=job)), "prep RESULT_DELIVERED")
            sfiles, pfiles, proof, outcome = h.retire_scored(job, case.kwargs["identity"], published["identity"],
                                                               case.frame_id)
            original = E.evaluate_prepared(identity=pred["identity"]["action_identity"], prepared=case.bundle.prepared,
                predicted_labels=pred["labels"], predictions=pred["records"], avo=case.avo_document["avo"])
            descriptor = json.loads(pfiles["producer_descriptor.json"])
            checks = dict(identity=published["identity"].encode() == r["identity_bytes"],
                          bound_identity=bound["identity"] == published["identity"],
                          semantic=published["semantic_sha256"] == r["semantic_sha256"],
                          commit=committed == r["commit"], packet=packet == r["packet"],
                          producer_archive_files={k: P.sha(v) for k, v in pfiles.items()} == r["archive_files"],
                          q_perc_retained=result["q_perc"] == pred["q_perc"],
                          measurement_original=C.canonical(result["measurement"]) == C.canonical(original.measurement),
                          scorer_score_document_measurement=C.canonical(C._parse(sfiles["numerical_score.json"])["measurement"])
                              == C.canonical(original.measurement),
                          **P.archive_checks(case, pfiles, descriptor))
            rows.append(dict(part="B_v11_two_process", frame_id=case.frame_id, order="prediction_first" if prediction_first
                             else "target_first", q_perc=result["q_perc"], checks=checks, ok=all(checks.values()),
                             producer_outcome=outcome["outcome"]))
            if not rows[-1]["ok"]:
                raise AssertionError(f"Part B mismatch {case.frame_id}: {checks}")
        prep_snap, score_snap = h.snapshots()
        C.require(prep_snap["engine"]["retained_jobs"] == 0 and score_snap["retained_slots"] == 0, "all work retires")
        closed = h.close()
    except BaseException:
        h.abort()
        raise
    return rows, dict(frames=len(rows), all_exact=all(r["ok"] for r in rows),
        undefined_quality=sum(r["q_perc"] is None for r in rows), prep_snapshot=prep_snap, scorer_snapshot=score_snap,
        durable_completions=closed["archive"]["completed"], prep_close=closed["prep"], score_close=closed["score"])


def run(name):
    P.environment_ok()
    output = P.create_output(name)
    P.write_json(output / "PLAN.json", dict(schema="splitfusion.exact_gt_service_proto.step3_plan.v1",
        processes=["prep worker", "score worker", "unchanged ArchiveClient (fsync)"],
        affinity_or_priority_changed=False, timing_recorded=False, live=False, gpu=False,
        prediction_identity_projection=list(PREDICTION_FIELDS)))
    rows, summary = [], {}
    try:
        a_rows, summary["part_a_live100"] = part_a(output)
        rows += a_rows
        b_rows, summary["part_b_v11"] = part_b(output)
        rows += b_rows
        result = dict(schema="splitfusion.exact_gt_service_proto.step3_result.v1", status="STEP3_COMPOSITION_EXACT",
                      **summary, timing_recorded=False, performance_claim=False, live_qualified=False)
        P.write_json(output / "RESULT.json", result)
        print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items()
              if kk in ("frames", "all_exact", "undefined_quality", "durable_completions")}) for k, v in result.items()}, indent=1))
    except BaseException as error:
        P.write_json(output / "FAILURE.json", dict(error=f"{type(error).__name__}: {error}", completed=len(rows)))
        raise
    finally:
        with (output / "FRAMES.jsonl").open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        for path in output.rglob("durable"):
            if path.is_dir():
                for item in sorted(path.rglob("*"), reverse=True):
                    item.unlink() if item.is_file() else item.rmdir()
        P.seal_output(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    run(parser.parse_args().name)
