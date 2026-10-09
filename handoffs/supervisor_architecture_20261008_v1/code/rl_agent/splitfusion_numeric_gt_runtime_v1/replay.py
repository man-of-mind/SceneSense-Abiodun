"""Offline real-process composition: source worker, scorer worker, durable archive.

No CARLA server, models, OAI, radio or Docker is launched. Replays retained
source/prediction bytes through actual sealed-descriptor IPC and real fsync.
All send/feedback notifications are explicit harness events, never radio ACKs.
No attempt is made to infer a 170-ms live success rate from this run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import tempfile
import time

import numpy as np

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_numeric_gt_integration_v2 import replay as H, fixtures as F, ready as R
from rl_agent.splitfusion_numeric_gt_producer_v1 import cases as K, artifacts as AR
from rl_agent.splitfusion_numeric_gt_producer_v1.replay_v11 import KEYS
from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E
from rl_agent.splitfusion_live_parity_runtime_v1 import pose as PO
from rl_agent.splitfusion_live_gt_handoff_v1.archive import ArchiveClient
from .source_codec import encode_source
from .staging import encode_stage, neutral_identity
from .scorer import registration_key
from .process import Client
from . import ipc as I

ROOT = H.ROOT
OUTPUT_ROOT = ROOT / "experiments/splitfusion_numeric_gt_runtime_v1"
PACKAGE = Path(__file__).resolve().parent
RESERVATION = 64 * 1024 * 1024


def inventory(data):
    result = H.inventory(data["startup"])
    for package in (PACKAGE, ROOT / "rl_agent/splitfusion_live_gt_optimization_v4",
                    ROOT / "rl_agent/splitfusion_live_gt_handoff_v1"):
        for path in package.glob("*.py"):
            result[str(path.relative_to(ROOT))] = H.sha(path)
    return dict(sorted(result.items()))


def expected_wire(expected):
    return dict(expected, frame_context=PO.context_document(expected["frame_context"]))


def wait_archive(archive, job, timeout=30):
    deadline = time.monotonic() + timeout
    while True:
        archive.poll()
        matches = [r for r in archive.completions if r["job_id"] == job]
        if matches:
            C.require(len(matches) == 1 and matches[0]["durable"], "exactly one durable completion required")
            return matches[0]
        remaining = deadline - time.monotonic()
        C.require(remaining > 0, "bounded archive completion timeout")
        select.select([archive.socket], [], [], min(remaining, 1))


def quantile(values):
    return dict(n=len(values), **{name: float(np.quantile(values, p)) for name, p in
        (("p50", .5), ("p95", .95), ("p99", .99), ("max", 1))})


def run(output, frames, cpu, *, staged_source=False):
    C.require(frames in (2, 49), "smoke2 or entire chronological49 only")
    data = H.load_inputs(frames)
    sources = inventory(data)
    output = Path(output).absolute()
    C.require(not output.exists() and output.resolve().is_relative_to(OUTPUT_ROOT)
        and not any(p.is_symlink() for p in output.parents), "create-only runtime evidence directory")
    output.mkdir(parents=True, exist_ok=False)
    binding = C.sha256(C.canonical(dict(scope="OFFLINE_REAL_PROCESS_NUMERIC_GT", sources=sources,
                                      input_seal=K.INPUT_SEAL_SHA)))
    before_affinity = sorted(os.sched_getaffinity(0))
    before_fds = len(os.listdir("/proc/self/fd"))
    H.write(output / "PLAN.json", dict(schema="splitfusion.numeric_gt.runtime.offline_plan.v1",
        source_sha256=sources, input_sha256=data["inputs"], component_sha256=data["seals"], run_binding=binding,
        frames=frames, cpu=cpu, environment=data["environment"], affinity_before=before_affinity,
        actual_processes=["producer", "scorer", "stdlib_durable_archive"],
        source_handoff="unchanged V4 snapshot inside sealed memfd primitive envelope",
        input_orders="alternating target-first and prediction-first; latter admits prediction while producer computes",
        authentication="private inherited socket capability + creator UID/PID, launched child PID and nonce; no named listener",
        clocks="actual RAW process observations; legacy AVO clocks explicitly recorded offline inputs",
        feedback="harness callback receipt, NOT radio ACK", artifact_policy="temporary durable readback, hashes retained; no duplicate raw corpus kept",
        archive_execution="artifact reconstruction after score in producer; actual fsync in separate stdlib process",
        queue_scope="bounded max2 retained jobs, one outstanding RPC per role; this harness drains each frame before next",
        performance_scope="CPU-only real IPC service observations; not a loaded live-route benchmark",
        staged_source_before_plan=staged_source,
        staged_source_scope="offline order rehearsal; no live action-open timer or sensor callback" if staged_source else None,
        live_launch=False, deadline_claim=False))
    producer = scorer = archive = None
    rows, durable_inventory = [], []
    try:
        config = dict(run_binding=binding, producer_binding=data["startup"]["binding_sha256"],
                      prediction_contract=data["prediction_contract"], max_frames=frames,
                      producer_args={k: str(v) if isinstance(v, Path) else v
                                     for k, v in K.producer_arguments(data["cases"][0]).items()},
                      offline_replay=True)
        producer = Client("producer", config, directory=output / "producer_process", source_sha256=sources, cpu=cpu)
        scorer = Client("scorer", dict(run_binding=binding, producer_binding=data["startup"]["binding_sha256"],
            max_jobs=2, max_bytes=2 * RESERVATION), directory=output / "scorer_process", source_sha256=sources, cpu=cpu)
        with tempfile.TemporaryDirectory(prefix="durable_readback_", dir=output) as archive_root:
            archive_root = Path(archive_root)
            archive = ArchiveClient(roots=[archive_root], max_jobs=2, max_bytes=128 * 1024 * 1024,
                                    max_frames=frames, binding=dict(run_binding=binding))
            H.write(output / "PROCESS_READY.json", dict(producer=producer.ready, scorer=scorer.ready, archive=archive.ready))
            for index, case in enumerate(data["cases"]):
                pred = data["predictions"][case.frame_id]
                expected = H.expectation(data, case, binding)
                key = registration_key(expected)
                plan = F.plan_for(case, pred, expected["frame_context"])
                H.write(output / f"frame_{case.frame_id}_PLAN.json", dict(expected=expected_wire(expected),
                    job=index, plan_sha256=C.sha256(plan), identity_sha256=C.sha256(key)))
                scorer.call("REGISTER", dict(key=key, expected=expected_wire(expected), reservation_bytes=RESERVATION))
                begin = I.raw()
                sourcekwargs = {k: case.kwargs[k] for k in KEYS}
                if staged_source:
                    chunks, extras = encode_stage(sourcekwargs, case.kwargs["policy_timestamp_s"])
                else:
                    chunks, extras = encode_source(sourcekwargs, case.kwargs["policy_timestamp_s"])
                packed = I.raw()
                stage_end = None
                if staged_source:
                    staged = producer.call("STAGE_SOURCE", dict(job=index,
                        source_packet=b"".join(chunks), stage_extras=extras))
                    C.require(staged["capture_template"] == neutral_identity(case.kwargs["identity"]),
                              "source stage selected an action or a foreign frame")
                    stage_end = I.raw()
                clocks = {k: case.avo_document[k] for k in AR.AVO_CLOCKS if k != "sync_observed_raw_ns"}
                clocks["sync_observed_raw_ns"] = case.avo_prescore["source_synchronization"]["observed_raw_ns"]
                request = dict(job=index, plan_bytes=plan,
                    frame_context=PO.context_document(expected["frame_context"]),
                    population=case.evidence["dynamic_snapshot_evidence"], archive_clocks=clocks,
                    payload_filenames={k: v["filename"] for k, v in case.avo_document["payloads"].items()},
                    prescore_filename=case.avo_document["prescore_input"]["filename"],
                    replay_observed_raw_ns=case.evidence["source_synchronization"]["observed_raw_ns"])
                if staged_source:
                    request["capture_identity"] = case.kwargs["identity"]
                else:
                    request.update(source_packet=b"".join(chunks), source_extras=extras)
                producer.begin("PREPARE_STAGED" if staged_source else "PREPARE", request)
                prediction_value = dict(key=key, identity=pred["identity"], labels=pred["labels"].tobytes(order="C"),
                    shape=list(pred["labels"].shape), records=pred["records"], prediction_contract=data["prediction_contract"])
                if index % 2:
                    scorer.call("PREDICTION", prediction_value)
                    C.require(scorer.call("SCORE", dict(key=key)) is None, "prediction alone must not score")
                prepared = producer.finish()
                C.require(prepared["identity_bytes"] == key, "real-process producer/scorer registered identity differs")
                committed = producer.call("CONFIRM_SEND", dict(job=index, plan_bytes=plan))
                packet = producer.call("PUBLISH_READY", dict(job=index))
                scorer.call("TARGET", dict(key=key, packet=packet))
                target_admitted = I.raw()
                if not index % 2:
                    C.require(scorer.call("SCORE", dict(key=key)) is None, "target alone must not score")
                    scorer.call("PREDICTION", prediction_value)
                score_begin = I.raw()
                score = scorer.call("SCORE", dict(key=key))
                callback = I.raw()
                # This is the harness's actual callback receipt; not radio or a
                # success/timeout substitution in the policy controller.
                scorer.call("FEEDBACK_CONFIRMED", dict(key=key))
                note = dict(job=index, identity_bytes=key, semantic_sha256=prepared["semantic_sha256"])
                producer.call("SCORE_COMPLETED", note)
                producer.call("FEEDBACK_CONFIRMED", note)
                archival = producer.call("BUILD_ARCHIVE", dict(job=index))
                descriptor = json.loads(archival["descriptor_bytes"])
                pfiles = archival["files"]
                C.require(C.sha256(archival["descriptor_bytes"]) == archival["archive_sha256"], "archive descriptor mismatch")
                arguments = {name: descriptor[name] for name in ("ready_semantic_sha256", "input_metadata", "avo_document",
                                                                 "legacy_target_meta", "old_prepared_sha256")}
                arguments.update(segmentation_png=pfiles["segmentation.png"], object_ignore_png=pfiles["object_ignore.png"])
                sarchive = scorer.call("ATTACH_ARCHIVE", dict(key=key, arguments=arguments))
                sfiles = sarchive["files"]
                # Independent original evaluator enters only AFTER measured
                # production/scoring to check retained exactness.
                original = E.evaluate_prepared(identity=pred["identity"]["action_identity"], prepared=case.bundle.prepared,
                    predicted_labels=pred["labels"], predictions=pred["records"], avo=case.avo_document["avo"])
                checks = dict(measurement=C.canonical(score["measurement"]) == C.canonical(original.measurement),
                    qperc=score["q_perc"] == pred["q_perc"],
                    scoring_evidence=sfiles == E.archive_payloads(original),
                    segmentation_png=pfiles["segmentation.png"] == case.segmentation_png,
                    ignore_png=pfiles["object_ignore.png"] == case.object_ignore_png,
                    prepared_digest=descriptor["old_prepared_sha256"] == case.ready["prepared_content_sha256"],
                    avo_document=C.sha256(pfiles["avo.json"]) == case.avo_sha256,
                    avo_prescore=C.sha256(pfiles[case.avo_document["prescore_input"]["filename"]]) == case.prescore_sha256,
                    full_identity=score["identity_sha256"] == C.sha256(key),
                    ready_semantic=score["ready_semantic_sha256"] == prepared["semantic_sha256"])
                # Independently admit the received target and its source pose.
                target = R.decode_target(packet, **expected)
                checks["committed_history"] = target.metadata["history_after_sha256"] == committed["history_after_sha256"]
                if not all(checks.values()):
                    H.write(output / f"frame_{case.frame_id}_PARITY_FAILURE.json", dict(checks=checks,
                        candidate_descriptor=descriptor, original_target=case.target_json,
                        original_prepared_sha256=case.ready["prepared_content_sha256"],
                        candidate_scoring_evidence=json.loads(sfiles["evaluation.json"]),
                        original_scoring_evidence=pred["evidence"]))
                C.require(all(checks.values()), "real-process composition parity: " + str(checks))
                pfx, sfx = f"frame_{case.frame_id}/producer", f"frame_{case.frame_id}/scorer/"
                files = {str(archive_root / pfx / name): raw for name, raw in pfiles.items()}
                files.update({str(archive_root / sfx / name): raw for name, raw in sfiles.items()})
                job = archive.submit(case.kwargs["identity"], files)
                receipt = wait_archive(archive, job)
                for name, raw in files.items():
                    path = Path(name)
                    C.require(H.sha(path) == C.sha256(raw) and path.stat().st_size == len(raw), "durable readback mismatch")
                scorer.call("ARCHIVE_DURABLE", dict(key=key, receipt=receipt, relative_prefix=sfx))
                producer.call("SCORER_ARCHIVE_COMPLETED", note)
                producer.call("ARCHIVE_DURABLE", dict(job=index, receipt=receipt, relative_prefix=pfx))
                ps, ss = producer.call("SNAPSHOT"), scorer.call("SNAPSHOT")
                C.require(ps["valid"] and ss["valid"] and ps["retained_jobs"] == ss["retained_jobs"] == 0,
                          "post-durable ownership not fully released")
                durable_inventory.append(dict(frame_id=case.frame_id, receipt=receipt))
                timing = dict(source_pack_ms=(packed - begin) / 1e6,
                    source_start_to_edge_target_admission_ms=(target_admitted - begin) / 1e6,
                    joined_score_rpc_to_callback_ms=(callback - score_begin) / 1e6,
                    source_start_to_callback_ms=(callback - begin) / 1e6)
                if stage_end is not None:
                    timing.update(source_stage_rpc_ms=(stage_end - packed) / 1e6,
                                  plan_to_target_admission_ms=(target_admitted - stage_end) / 1e6,
                                  plan_to_callback_ms=(callback - stage_end) / 1e6)
                rows.append(dict(frame_id=case.frame_id, order="prediction_first" if index % 2 else "target_first",
                    checks=checks, q_perc=score["q_perc"], history_after_sha256=committed["history_after_sha256"],
                    timing=timing,
                    ready_packet_bytes=len(packet), durable_files=len(files), durable_bytes=sum(map(len, files.values())),
                    producer_snapshot=ps, scorer_snapshot=ss))
                H.write(output / f"frame_{case.frame_id}.json", rows[-1])
                if (index + 1) % 10 == 0:
                    print(json.dumps(dict(composed_process_frames=index + 1, total=frames)), flush=True)
            archive_summary = archive.close()
            H.write(output / "DURABLE_ARCHIVE_RECEIPTS.json", dict(archive=archive_summary, frames=durable_inventory,
                readback_verified=True, scratch_payloads_removed_on_scope_exit=True))
        producer_end = producer.call("SNAPSHOT")
        scorer_end = scorer.call("SNAPSHOT")
        producer.close(); scorer.close()
        H.write(output / "PROCESS_TIMELINE.json", dict(producer=producer.events, scorer=scorer.events))
        C.require(inventory(data) == sources, "runtime source binding changed")
        H.check_unchanged(data, H.inventory(data["startup"]))
        C.require(sum(r["q_perc"] is None for r in rows) == (8 if frames == 49 else 0), "undefined population differs")
        result = dict(status=("ALL49_STAGED_PROCESS_COMPOSITION_EXACT_DURABLE" if staged_source else
                              "ALL49_PROCESS_COMPOSITION_EXACT_DURABLE") if frames == 49 else
                             ("STAGED_SMOKE2_PROCESS_COMPOSITION_EXACT" if staged_source else
                              "SMOKE2_PROCESS_COMPOSITION_EXACT"),
            frames=frames, undefined_quality=sum(r["q_perc"] is None for r in rows), all_exact=True,
            actual_producer_pid=producer.process.pid, actual_scorer_pid=scorer.process.pid,
            actual_archive_pid=archive_summary["child_pid"],
            unique_child_pids=len({producer.process.pid, scorer.process.pid, archive_summary["child_pid"]}),
            timings_ms={name: quantile([r["timing"][name] for r in rows]) for name in rows[0]["timing"]},
            producer=producer_end, scorer=scorer_end, archive_completed=archive_summary["completed"],
            durable_readback_files=sum(r["durable_files"] for r in rows),
            durable_readback_bytes=sum(r["durable_bytes"] for r in rows),
            current_affinity=sorted(os.sched_getaffinity(0)), parent_affinity_unchanged=sorted(os.sched_getaffinity(0)) == before_affinity,
            source_files=len(sources), input_files=len(data["inputs"]), component_hashes_unchanged=True,
            active_route_modified=False, live_deadline_claim=False, staged_source=staged_source,
            limitations=["Offline retained source and prediction fixtures, not live sensors/inference.",
                "Harness send/callback notifications are not radio sends/feedback.",
                "Early live plan hook and new live archive-clock schema are not installed.",
                "Each frame drains before next; sustained concurrent archival/backpressure is not measured.",
                "Artifact reconstruction still occupies producer after scoring; only durable writes are in archive child.",
                "Stage observations include real local IPC but not loaded CARLA/OAI/GPU scheduling or downlink."])
        H.write(output / "RESULT.json", result)
        print(json.dumps({k: result[k] for k in ("status", "frames", "undefined_quality", "unique_child_pids", "timings_ms")}))
    except BaseException as error:
        H.write(output / "FAILURE.json", dict(error=f"{type(error).__name__}: {error}", completed=len(rows), live_run=False))
        raise
    finally:
        for client in (producer, scorer, archive):
            if client is not None and not client.closed:
                client.abort()
        if not (output / "PROCESS_TIMELINE.json").exists():
            H.write(output / "PROCESS_TIMELINE.json", dict(producer=[] if producer is None else producer.events,
                scorer=[] if scorer is None else scorer.events, incomplete=True,
                pending_producer=None if producer is None else producer.pending,
                pending_scorer=None if scorer is None else scorer.pending))
        if archive is not None and archive.failed:
            H.write(output / "ARCHIVE_FAILURE_STATUS.json", archive.close_result)
        H.write(output / "TEARDOWN.json", dict(child_returncodes={name: None if c is None else c.process.poll()
                for name, c in (("producer", producer), ("scorer", scorer), ("archive", archive))},
            parent_affinity_unchanged=sorted(os.sched_getaffinity(0)) == before_affinity,
            before_fds=before_fds, after_fds=len(os.listdir("/proc/self/fd")), live_services_started=False))
        H.write(output / "OUTPUT_SHA256.json", {str(p.relative_to(output)): H.sha(p)
            for p in sorted(output.rglob("*")) if p.is_file()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, choices=(2, 49), default=49)
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument("--staged-source", action="store_true",
                        help="stage the immutable source before sending the small action plan")
    args = parser.parse_args()
    run(args.output, args.frames, args.cpu, staged_source=args.staged_source)
