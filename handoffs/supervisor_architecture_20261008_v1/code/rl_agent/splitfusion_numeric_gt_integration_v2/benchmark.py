"""Serial, paired CPU benchmark of composed numerical readiness and scoring.

Not a route, deadline, IPC or durable-archive benchmark. Baseline is the active
v11 V1/V2/V6 source composition, not the older unoptimized reference. Both arms
consume the same retained images, original inputs and actual UE/edge contexts.
All deferred candidate work is timed separately, never silently discarded.
"""
import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from types import FunctionType, SimpleNamespace

import cv2
import numpy as np

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_numeric_gt_producer_v1 import benchmark as PB, cases as K, source as NS
from rl_agent.splitfusion_numeric_gt_producer_v1.replay_v11 import KEYS, quality_bridge
from rl_agent.splitfusion_live_gt_optimization_v2 import candidate as V2C
from rl_agent.splitfusion_live_gt_handoff_v1 import scoring_packet as P
from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E, target_transport as T
from rl_agent.splitfusion_live_parity_repair_v2 import prediction_archive as PA
from rl_agent.splitfusion_live_parity_runtime_v1 import pose
from . import composition as X, replay as H, scoring as Q
from .ownership import ProducerOwner
from .join import BoundedTargetPredictionJoin


@dataclass(frozen=True)
class RetainedPredictionTicket:
    """Archive API view of retained identity/context; not a new live envelope."""
    envelope: object
    context: object
    gt_identity: dict
    enqueued_wall_ns: int
    predicted_mask: object
    records: tuple


def ticket_for(prediction, context):
    identity = prediction["identity"]
    envelope = SimpleNamespace(**identity["action_identity"], capture_timestamp_ns=context.capture_timestamp_ns)
    return RetainedPredictionTicket(envelope, context, identity["gt_identity"], identity["enqueued_wall_ns"],
                                    prediction["labels"], tuple(prediction["records"]))


def baseline(case, prediction, *, base, archive, context, validator, sender_encode, binding, spec, quality, sink):
    meter, source_stages = X.Meter(), {}
    kwargs = {key: case.kwargs[key] for key in KEYS}
    with meter.span("baseline_pre_archive_total"), X.fixture_clock(case):
        X.register_population(base["producer"], case)
        inputs, prepared, avo = meter.call("source_avo_prepared", lambda: PB.run_baseline(
            base, case, kwargs, sink, source_stages))
        meter.call("publish_input_validation", lambda: validator(inputs, kwargs["identity"]))
        # Retained ready clocks are packet-format fields, not measured live clocks.
        raw = case.avo_document["end_raw_ns"]
        wall = case.avo_document["ready_wall_ns"]
        wire = meter.call("packet_encode_with_metadata", lambda: sender_encode(identity=kwargs["identity"],
            prepared=prepared, avo_document=avo, input_metadata=T._input_metadata(inputs),
            ready_raw_ns=raw, ready_wall_ns=wall, binding=binding))
        meter.call("packet_sender_inspect", lambda: P.inspect_packet(wire, binding=binding,
                                                                      expected_identity=kwargs["identity"]))
        meter.call("packet_broker_inspect", lambda: P.inspect_packet(wire, binding=binding,
                                                                      expected_identity=kwargs["identity"]))
        ticket = ticket_for(prediction, context)
        meter.call("prediction_archive_add", lambda: archive.add(ticket))
        owned = meter.call("prediction_archive_owned_ticket", lambda: archive.owned_ticket(ticket))
        decoded = meter.call("packet_decode_and_restore", lambda: P.decode_packet(wire, binding=binding,
                                                                              expected_identity=kwargs["identity"]))
        def pose_check():
            inverse = np.asarray(json.loads(decoded["input_metadata"]["frame"]["camera_inverse_matrix_json"]), np.float64)
            pose._agree(pose.context_camera(context), np.linalg.inv(inverse), "baseline retained context/GT")
        meter.call("pose_reconciliation", pose_check)
        result = meter.call("evaluate_prepared", lambda: E.evaluate_prepared(
            identity=prediction["identity"]["action_identity"], prepared=decoded["prepared"],
            predicted_labels=owned.predicted_mask, predictions=owned.records, avo=decoded["avo"]))
        value = meter.call("quality_formula", lambda: quality.evaluate_exact_quality(spec, result.measurement))
    with meter.span("baseline_post_score_evidence_registration"):
        archive.evaluated(ticket, result.evidence)
    return dict(timings=meter.spans, source_substages=source_stages, measurement=result.measurement,
        q_perc=None if value is None else float(value.q_perc), evidence=result.evidence,
        prepared_sha256=prepared.content_sha256, packet_bytes=len(wire))


def quantiles(values):
    return dict(n=len(values), **{k: float(np.quantile(values, p)) for k, p in
        (("min", 0), ("p50", .5), ("p95", .95), ("p99", .99), ("max", 1))})


def summary(rows):
    stages = {}
    for arm in ("baseline", "candidate"):
        names = sorted(set.intersection(*(set(r[arm]["timings"]) for r in rows)))
        stages[arm] = {name: {metric: quantiles([r[arm]["timings"][name][metric] / 1e6 for r in rows])
            for metric in ("wall_ns", "thread_cpu_ns")} for name in names}
    paired = {metric: quantiles([(r["baseline"]["timings"]["baseline_pre_archive_total"][metric]
                - r["candidate"]["timings"]["candidate_pre_archive_total"][metric]) / 1e6 for r in rows])
              for metric in ("wall_ns", "thread_cpu_ns")}
    return dict(stages_ms=stages, paired_pre_archive_saving_ms=paired,
        aggregation="Quantiles of measured per-frame intervals/differences; no sum of stage quantiles.")


def run(output, *, frames, passes, cpu, correctness):
    C.require(frames in (2, 49) and passes in (1, 2), "bounded smoke2 or full49 serial passes required")
    data = H.load_inputs(frames)
    sources = H.inventory(data["startup"])
    proof = json.loads((correctness / "RESULT.json").read_bytes())
    plan0 = json.loads((correctness / "PLAN.json").read_bytes())
    manifest0 = json.loads((correctness / "OUTPUT_SHA256.json").read_bytes())
    C.require(all(H.sha(correctness / k) == v for k, v in manifest0.items()), "correctness output seal differs")
    C.require(proof["status"] == "ALL49_ARCHIVE_FREE_NUMERIC_COMPOSITION_EXACT"
        and plan0["source_sha256"] == sources, "full parity on identical sources required before timing")
    allowed = os.sched_getaffinity(0)
    C.require(cpu in allowed, "benchmark CPU outside current allowed set")
    output = H.create_output(output)
    binding = C.sha256(C.canonical(dict(scope="OFFLINE_COMPOSED_NUMERICAL_TIMING", sources=sources)))
    load_before = Path("/proc/loadavg").read_text().split()[:3]
    old_cv = cv2.getNumThreads()
    H.write(output / "PLAN.json", dict(schema="splitfusion.numeric_gt.v2.serial_timing_plan", sources=sources,
        input_sha256=data["inputs"], component_evidence_sha256=data["seals"], binding=binding,
        correctness_manifest=H.sha(correctness / "OUTPUT_SHA256.json"), frames=frames, passes=passes,
        order="alternate baseline/candidate each chronological frame, reverse order on second pass",
        baseline="ACTIVE_V11_V1_V2_V6_SOURCE_PLUS_PACKET_AND_ORIGINAL_SCORER",
        candidate="ARCHIVE_FREE_V2_NUMERICAL_READY_WIRE_OWNERSHIP_JOIN_SCORE",
        cpu=cpu, affinity_before=sorted(allowed), cv2_threads_before=old_cv, threads_during=1,
        load_average_before=load_before, environment=data["environment"],
        exclusions=["live dispatch queue", "OS IPC", "broker process scheduling", "archive process concurrency",
            "disk durability", "CARLA/OAI/GPU", "actor/codec/uplink", "feedback socket send/downlink"],
        clock_scope="RAW wall intervals and caller thread CPU; packet fields use retained fixture clocks",
        prediction_scope="native retained outputs, full original archive add/owned_ticket vs actual new admission",
        deferred_work="candidate archive fully executed serially after score before next frame; timed separately",
        independent_samples=frames, repeated_measurements=frames * passes,
        live_authorization=False, live_deadline_claim=False))
    rows = []
    try:
        os.sched_setaffinity(0, {cpu})
        cv2.setNumThreads(1)
        spec, quality = quality_bridge()
        private = V2C.private_functions(fast_labels=True, fast_metadata=True)
        validator = private["write"].__globals__["_validate_inputs"]
        # UE source process has the retained optimized restore; the separate
        # edge evaluator does not. Clone only the sender call, never install it.
        sender_encode = FunctionType(P.encode_packet.__code__, dict(P.encode_packet.__globals__,
            E=SimpleNamespace(restore_prepared_target=private["restore"])),
            P.encode_packet.__name__, P.encode_packet.__defaults__, P.encode_packet.__closure__)
        sender_encode.__kwdefaults__ = P.encode_packet.__kwdefaults__
        scorer = Q.NumericalScorer()
        with tempfile.TemporaryDirectory(prefix="numeric-v2-serial-") as sink:
            for pass_index in range(passes):
                base = PB.live_composition(K.producer_arguments(data["cases"][0]))
                archive = PA.PredictionArchive(max_frames=frames)
                producer = NS.NumericSourceProducer(**K.producer_arguments(data["cases"][0]))
                join = BoundedTargetPredictionJoin(scorer, max_jobs=2, max_bytes=2 * X.RESERVATION)
                with ProducerOwner(producer, max_retained_jobs=2, max_retained_bytes=2 * X.RESERVATION) as owner:
                    for index, case in enumerate(data["cases"]):
                        prediction = data["predictions"][case.frame_id]
                        order = ("baseline", "candidate") if (index + pass_index) % 2 == 0 else ("candidate", "baseline")
                        values = {}
                        for arm in order:
                            if arm == "baseline":
                                values[arm] = baseline(case, prediction, base=base, archive=archive,
                                    context=data["contexts"][case.frame_id]["context"], validator=validator, sender_encode=sender_encode,
                                    binding=binding, spec=spec, quality=quality, sink=sink)
                            else:
                                values[arm] = X.candidate(case, prediction, owner=owner, scorer=scorer, join=join,
                                    expected=H.expectation(data, case, binding), spec=spec, quality=quality, index=index)
                        # Numerical/evidence assertions are outside the timing boundary.
                        exact = X.check_exact(values["candidate"], case, prediction)
                        b, n = values["baseline"], values["candidate"]
                        C.require(b["measurement"] == n["measurement"] and b["q_perc"] == n["q_perc"]
                            and b["evidence"] == n["evidence"] and b["prepared_sha256"] == n["prepared"].content_sha256,
                            "paired benchmark numerical/evidence mismatch")
                        row = dict(pass_index=pass_index, frame_id=case.frame_id, order=order, all_exact=all(exact.values()),
                            q_perc=n["q_perc"], **{arm: {k: values[arm][k] for k in ("timings", "packet_bytes")}
                                for arm in ("baseline", "candidate")}, baseline_source_substages=b["source_substages"])
                        rows.append(row)
                        H.write(output / f"frame_{pass_index}_{case.frame_id}.json", row)
                        if (index + 1) % 10 == 0:
                            print(json.dumps(dict(pass_index=pass_index, completed=index + 1, total=frames)), flush=True)
                join.close()
        H.check_unchanged(data, sources)
        result = dict(status="SERIAL_COMPOSED_CPU_BENCHMARK_EXACT_NOT_LIVE" if frames == 49 else "SMOKE2_NOT_PERFORMANCE",
            unique_frames=frames, passes=passes, measurements=len(rows), all_exact=True, **summary(rows),
            by_pass={str(p): summary([r for r in rows if r["pass_index"] == p]) for p in range(passes)},
            after_first_frame=summary([r for r in rows if r["frame_id"] != data["cases"][0].frame_id]),
            load_average_before=load_before, load_average_after=Path("/proc/loadavg").read_text().split()[:3],
            serialized_not_parallel_pipeline=True, archive_overlap_validated=False, OS_IPC_measured=False,
            packet_size_bytes={arm: quantiles([r[arm]["packet_bytes"] for r in rows]) for arm in ("baseline", "candidate")},
            live_deadline_claim=False)
        H.write(output / "RESULT.json", result)
        print(json.dumps(dict(status=result["status"], frames=frames,
            baseline_p50_ms=result["stages_ms"]["baseline"]["baseline_pre_archive_total"]["wall_ns"]["p50"],
            candidate_p50_ms=result["stages_ms"]["candidate"]["candidate_pre_archive_total"]["wall_ns"]["p50"],
            paired_saving_ms=result["paired_pre_archive_saving_ms"]["wall_ns"]["p50"])), flush=True)
    except BaseException as error:
        H.write(output / "FAILURE.json", dict(error=f"{type(error).__name__}: {error}", completed=len(rows), live_run=False))
        raise
    finally:
        cv2.setNumThreads(old_cv)
        os.sched_setaffinity(0, allowed)
        H.write(output / "RESTORATION.json", dict(affinity_restored=os.sched_getaffinity(0) == allowed,
            cv2_threads_restored=cv2.getNumThreads() == old_cv, no_services_started=True))
        H.write(output / "OUTPUT_SHA256.json", {p.name: H.sha(p) for p in sorted(output.iterdir()) if p.is_file()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--correctness", type=Path, required=True)
    parser.add_argument("--frames", type=int, choices=(2, 49), default=49)
    parser.add_argument("--passes", type=int, choices=(1, 2), default=2)
    parser.add_argument("--cpu", type=int, default=4)
    args = parser.parse_args()
    run(args.output, frames=args.frames, passes=args.passes, cpu=args.cpu, correctness=args.correctness)
