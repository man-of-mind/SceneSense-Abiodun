"""Real multi-process integration rehearsal (CPU only, v11 fixtures, no CARLA/radio/GPU).

Actual code under test: ``EarlyGtUeHandoff`` (stage_before_action, on_plan, after_submit,
terminals, gt-freeze thread, close), the host ``service`` process with its real prep/score
workers and the unchanged ArchiveClient (fsync), and ``EarlyGtEvaluatorV1`` (bounded submit,
sender/receiver threads, retained ``_emit``/``quality_feedback``). Only the CARLA collector,
the Run-4 runtime and the radio are fixture stand-ins (the retained fixture builders of the
numeric-GT rehearsal). Registered scenarios:

  v2 additions: seven_anchor_lifecycle (both arrival orders), off_anchor, bind_refused_after_prediction,
  bind_refused_before_prediction, waiting_capacity, anchor_send_failure, subsequent_valid_frames (leak check);
  retained: horizon_timeout_then_late_prediction, edge_terminal, refused_freeze_void, scorer_capacity,
  producer_capacity, shutdown_drain.
Durations are incidental and not performance evidence.
"""
import argparse
import collections
import json
import os
from pathlib import Path
import queue
import time
from types import SimpleNamespace as NS

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as CC
from rl_agent.splitfusion_numeric_gt_runtime_v1.async_live_rehearsal import _owned_collector, _planned, _pose
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as E
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
from rl_agent.splitfusion_numeric_gt_producer_v1.replay_v11 import quality_bridge
from rl_agent.splitfusion_exact_gt_service_proto_v2 import parity_common as P

from . import contract as C
from .ue import EarlyGtUeHandoff, raw
from .edge import install_early_gt_edge, EarlyGtEvaluatorV1

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "experiments/splitfusion_exact_gt_live_v2"


class Rig:
    def __init__(self, out, corpus, limits, evidence=None):
        self.c = corpus
        self.cases = corpus.cases
        self.ue_dir = out / "ue_attempt"
        self.ue_dir.mkdir()
        self.evidence = out / "edge_evidence" if evidence is None else Path(evidence)
        self.evidence.mkdir(parents=True)
        registration = dict(exact_gt_limits=limits, exact_gt_sources=C.exact_sources(ROOT),
            numeric_run_binding=corpus.binding, numeric_producer_binding=corpus.startup["binding_sha256"],
            numeric_prediction_contract=corpus.contract, exact_gt_socket_name=C.SOCKET_NAME,
            exact_gt_capability_name=C.CAPABILITY_NAME)
        self.h = EarlyGtUeHandoff(attempt_dir=self.ue_dir, registration=registration, service_config={})
        self.h.collector = _owned_collector(self.cases)
        self.h.evidence_dir = self.evidence
        engine = dict(producer_args=corpus.producer_args(), run_binding=corpus.binding,
            producer_binding=corpus.startup["binding_sha256"], prediction_contract=corpus.contract,
            max_frames=100, max_retained_jobs=limits["prep_jobs"],
            max_retained_bytes=limits["prep_jobs"] * 64 * 1024 * 1024, offline_replay=False)
        self.h.start_service(engine)
        self.payloads = queue.Queue()
        self.feedback = {}
        self.rows = {}

    def admit(self, index, *, send=True, anchor=None, tamper=False):
        case = self.cases[index]
        frame_id = case.frame_id
        expected = self.c.expected(case)
        action, capture, context = expected["expected_action_identity"], case.kwargs["identity"], expected["frame_context"]
        if anchor is not None:
            variant = P.anchor_variant(self.c, case, anchor)
            action, capture = variant["action"], variant["capture"]
            if tamper:     # catalogue-inconsistent: capture/anchor id of `anchor`, q of another anchor
                action = dict(action, q_e4=9800 if action["q_e4"] != 9800 else 7000)
        runtime = NS(pipeline=NS(run_id=capture["run_id"], cell_id=capture["cell_id"], stream_id=capture["stream_id"],
                                 decisions=[]), engine=NS(lineage=action["controller_lineage_sha256"]),
                     _numeric_current_ego_pose=_pose(context), last_datagram={})
        self.h.stage_before_action(runtime, frame_id=frame_id, carla_timestamp=case.kwargs["policy_timestamp_s"],
                                   capture_timestamp_ns=capture["capture_timestamp_ns"], ego_pose=_pose(context))
        opened = raw()
        runtime.pipeline.decisions.append(dict(frame_id=frame_id, action_open=dict(domain="CLOCK_MONOTONIC_RAW", ns=opened)))
        t0 = time.monotonic()
        self.h.on_plan(runtime, _planned(action, capture, context))
        on_plan_s = time.monotonic() - t0
        row = dict(index=index, frame_id=frame_id, action=action, capture=capture, context=context, opened=opened,
                   runtime=runtime, on_plan_returned_quickly=on_plan_s < 0.05, anchor=anchor, tamper=tamper)
        self.rows[frame_id] = row
        if send is not None:
            self.send(frame_id, send)
        return row

    def send(self, frame_id, ok=True):
        row = self.rows[frame_id]
        if ok:
            row["runtime"].last_datagram[frame_id] = raw()
        self.h.after_submit(row["runtime"], frame_id, {"sent": ok})

    def predict(self, frame_id):
        row = self.rows[frame_id]
        pred = self.c.predictions[frame_id]
        envelope = NS(**row["action"], capture_timestamp_ns=row["capture"]["capture_timestamp_ns"])
        ticket = E.EvaluationTicketV2(envelope=envelope, context=row["context"], records=tuple(pred["records"]),
            predicted_mask=pred["labels"], gt_identity=row["capture"], enqueued_wall_ns=time.time_ns())
        self.ev._materialized[frame_id] = raw()
        self.ev.submit(ticket)

    def await_feedback(self, frame_id, timeout=30):
        deadline = time.monotonic() + timeout
        while frame_id not in self.feedback:
            CC.require(time.monotonic() < deadline, f"feedback for {frame_id} missing")
            try:
                payload = self.payloads.get(timeout=0.05)
            except queue.Empty:
                continue
            feedback, reason = W.decode_feedback(payload)
            self.feedback[int(feedback.frame_id)] = dict(kind=feedback.kind, q_perc=feedback.q_perc, reason=reason.name,
                                                         received=raw())
        return self.feedback[frame_id]

    def terminal(self, frame_id):
        row = self.rows[frame_id]
        now = raw()
        fb = self.feedback.get(frame_id)
        if fb is not None and fb["received"] - row["opened"] <= 170_000_000:
            self.h.on_feedback(frame_id, fb["received"])
        else:
            self.h.on_controller_poll(NS(resolution=None), None, now,
                NS(resolution=object(), resolved_by="TIMEOUT", reward_frame=NS(frame_id=frame_id)))


def run(name):
    P.environment_ok()
    out = EVIDENCE / name
    out.mkdir(parents=True, exist_ok=False)
    corpus = P.Corpus(49)
    limits = dict(C.LIMITS, prep_jobs=4, scorer_jobs=2, max_waiting=1, horizon_ms=1500, drain_s=60, max_frames=49,
                  prep_cpu=None, score_cpu=None, service_cpu=None)
    sources = C.exact_sources(ROOT)
    plan = dict(schema="splitfusion.exact_gt_live_v2.rehearsal_plan.v1", sources=sources, limits=limits,
                scenarios="see module docstring", anchors=list(P.SCREEN_ANCHORS), live_services_started=False,
                deadline_claim=False)
    (out / "PLAN.json").write_text(json.dumps(plan, indent=1, sort_keys=True))
    spec, _ = quality_bridge()
    rig, result, scenarios = None, None, {}
    A = list(P.SCREEN_ANCHORS)
    try:
        rig = Rig(out, corpus, limits)
        with install_early_gt_edge(socket_path=rig.evidence / C.SOCKET_NAME,
                                   capability_path=rig.evidence / C.CAPABILITY_NAME, connect_timeout_s=30):
            rig.ev = EarlyGtEvaluatorV1(spec=spec, read_ground_truth=None, send=rig.payloads.put, match_distance_m=2.,
                                        gt_timeout_s=2.)
            rig.ev.start()
            f = [c.frame_id for c in rig.cases]

            def scored(i, anchor=None, prediction_first=False):
                rig.admit(i, anchor=anchor)
                if not prediction_first:
                    time.sleep(0.8)
                rig.predict(f[i]); rig.await_feedback(f[i]); rig.terminal(f[i])

            # seven anchors through the full lifecycle, alternating arrival order
            for k, anchor in enumerate(A):
                scored(k, anchor, prediction_first=bool(k % 2))
            scenarios["seven_anchor_lifecycle"] = {str(f[k]): A[k] for k in range(7)}
            # off-anchor (template) target-first and prediction-first
            scored(7); scored(8, prediction_first=True)
            scenarios["off_anchor"] = [f[7], f[8]]
            # BIND refused AFTER the prediction reached the scorer (it waits, then is released)
            rig.admit(9, anchor=33, tamper=True); rig.predict(f[9]); rig.await_feedback(f[9]); rig.terminal(f[9])
            scenarios["bind_refused_after_prediction"] = f[9]
            # BIND refused BEFORE the prediction arrives
            rig.admit(10, anchor=50, tamper=True); time.sleep(0.8); rig.predict(f[10]); rig.await_feedback(f[10])
            rig.terminal(f[10])
            scenarios["bind_refused_before_prediction"] = f[10]
            # waiting capacity (max_waiting=1): two predictions before their targets
            rig.admit(11, anchor=35); rig.predict(f[11]); time.sleep(0.02)
            rig.admit(12, anchor=53); rig.predict(f[12])
            for i in (11, 12):
                rig.await_feedback(f[i]); rig.terminal(f[i])
            scenarios["waiting_capacity"] = dict(waiting=f[11], refused=f[12])
            time.sleep(1.0)
            # anchor send failure after binding: no commit, history unchanged
            rig.admit(13, anchor=65, send=False); time.sleep(0.4); rig.terminal(f[13])
            scenarios["anchor_send_failure"] = f[13]
            # horizon timeout, then a late anchor prediction that must not be scored
            rig.admit(14, anchor=67); time.sleep(limits["horizon_ms"] / 1e3 + 0.6); rig.predict(f[14])
            rig.await_feedback(f[14]); rig.terminal(f[14])
            scenarios["horizon_timeout_then_late_prediction"] = f[14]
            # edge terminal before any prediction
            rig.admit(15, anchor=17); time.sleep(0.2); rig.ev.notify_terminal(f[15], "STALE_BEFORE_MAP", "DECODE")
            time.sleep(0.5); rig.terminal(f[15])
            scenarios["edge_terminal"] = f[15]
            # refused freeze -> VOID
            rig.admit(16, send=None, anchor=33); time.sleep(0.05); rig.admit(17, send=None); time.sleep(0.05)
            rig.admit(18, send=None, anchor=50); time.sleep(0.3)
            for i in (16, 17, 18):
                rig.send(f[i])
            for i in (16, 17, 18):          # one waiting prediction at a time (rehearsal max_waiting=1)
                rig.predict(f[i]); rig.await_feedback(f[i]); rig.terminal(f[i])
            scenarios["refused_freeze_void"] = [f[16], f[17], f[18]]
            time.sleep(1.0)
            # scorer capacity: 19, 20 hold slots (terminals withheld), 21's target refused
            for i, anchor in ((19, 35), (20, None)):
                rig.admit(i, anchor=anchor); time.sleep(0.6); rig.predict(f[i]); rig.await_feedback(f[i])
            rig.admit(21, anchor=53); time.sleep(0.8); rig.predict(f[21]); rig.await_feedback(f[21])
            scenarios["scorer_capacity"] = f[21]
            # producer capacity: 22 refused (VOID)
            rig.admit(22, anchor=65); time.sleep(0.3); rig.predict(f[22]); rig.await_feedback(f[22])
            scenarios["producer_capacity"] = f[22]
            for i in (19, 20, 21, 22):
                rig.terminal(f[i])
            time.sleep(1.5)
            # subsequent valid frames: capacity must not have leaked
            subsequent = []
            for n, i in enumerate(range(23, 41)):
                anchor = A[n % 7] if n % 3 != 2 else None
                scored(i, anchor, prediction_first=bool(n % 2))
                subsequent.append(f[i])
            scenarios["subsequent_valid_frames"] = subsequent
            # shutdown drain: scored frame whose UE terminal never arrives before close
            rig.admit(41, anchor=67); time.sleep(0.6); rig.predict(f[41]); rig.await_feedback(f[41])
            scenarios["shutdown_drain"] = f[41]
            edge_report = rig.ev.close()
        ue_result = rig.h.close()
        service_dir = rig.ue_dir / "numeric_gt/service"
        service = [json.loads(l) for l in open(service_dir / "FRAMES.jsonl")]
        service_result = json.loads((service_dir / "SERVICE_RESULT.json").read_text())
        outcomes = {row["frame_id"]: row.get("outcome") for row in service}
        refused_reason = "NO_GT_DEFERRAL_REFUSED_CATALOGUE_ANCHOR_MISMATCH"
        expected = {**{f[k]: "SCORED" for k in range(9)}, f[9]: refused_reason, f[10]: refused_reason,
                    f[11]: "SCORED", f[12]: "UNSCORED_SCORER_WAITING_CAPACITY",
                    f[13]: ("CANCELLED_SEND_FAILED", "NO_GT_SEND_FAILED_BEFORE_ADMISSION"),
                    f[14]: "UNSCORED_NO_RESULT_BY_HORIZON", f[15]: "UNSCORED_EDGE_TERMINAL_STALE_BEFORE_MAP",
                    f[16]: "SCORED", f[17]: "SCORED", f[18]: "NO_GT_FREEZE_QUEUE_FULL", f[19]: "SCORED", f[20]: "SCORED",
                    f[21]: "UNSCORED_SCORER_SCORER_CAPACITY", f[22]: "NO_GT_PRODUCER_CAPACITY",
                    **{fid: "SCORED" for fid in subsequent}, f[41]: "SCORED"}
        by_frame = {row["frame_id"]: row for row in service}
        committed = [row["frame_id"] for row in service if row.get("committed")]
        commits = [row["commit"] for row in service if row.get("committed")]
        chain = all(a["history_after_sha256"] == b["history_before_sha256"] for a, b in zip(commits, commits[1:]))
        index = {c.frame_id: i for i, c in enumerate(rig.cases)}
        ref = P.AnchorReferenceDriver(corpus)
        reference = {}
        for fid in committed:
            row = rig.rows[fid]
            case = rig.cases[index[fid]]
            if row["anchor"] is None:
                reference[fid] = dict(ref.run(case), variant=None)
            else:
                variant = P.anchor_variant(corpus, case, row["anchor"])
                reference[fid] = dict(ref.run_anchor(case, variant), variant=variant)
        ref.close()
        from rl_agent.splitfusion_exact_gt_service_proto_v2.verify_anchor import pinned_score, as_bytes
        from rl_agent.splitfusion_exact_gt_service_proto_v2.scorer_core import expected_from_identity
        from rl_agent.splitfusion_numeric_gt_integration_v2 import ready as RR
        quality = {}
        for row in service:
            if row.get("outcome") != "SCORED":
                continue
            fid = row["frame_id"]
            case = rig.cases[index[fid]]
            ref_row = reference[fid]
            if ref_row["variant"] is None:
                expected_q = P.score_exact(corpus, case, ref_row["packet"])["q_perc"]
            else:
                _, exp = expected_from_identity(as_bytes(ref_row["identity_bytes"]))
                _, expected_q = pinned_score(corpus, RR.decode_target(ref_row["packet"], **exp),
                                             ref_row["variant"]["prediction_identity"], case)
            fb = rig.feedback[fid]
            archive = service_dir / "archive" / f"frame_{fid}"
            anchored = rig.rows[fid]["anchor"] is not None
            provenance_archived = ((archive / "scorer/anchor_provenance.json").exists()
                                   and (archive / "producer/anchor_provenance.json").exists()) if anchored else \
                not (archive / "scorer/anchor_provenance.json").exists()
            quality[fid] = dict(service=row["result"]["q_perc"], reference=expected_q, emitted=fb["q_perc"],
                                anchor=rig.rows[fid]["anchor"], provenance_archived=provenance_archived,
                                exact=row["result"]["q_perc"] == expected_q == fb["q_perc"] and provenance_archived)
        history_exact = commits[-1]["history_after_sha256"] == reference[committed[-1]]["history_after"]
        fault_frames = (f[9], f[10], f[12], f[14], f[18], f[21], f[22])
        faults = {fid: rig.feedback[fid] for fid in fault_frames if fid in rig.feedback}
        score_snapshot = (service_result.get("closed") or {}).get("score") or {}
        prep_snapshot = ((service_result.get("closed") or {}).get("prep") or {}).get("owner") or {}
        emitted = collections.Counter(int(r["frame_id"]) for r in rig.ev.records)
        checks = dict(outcomes=all(outcomes.get(k) in (v if isinstance(v, tuple) else (v,)) for k, v in expected.items()),
            quality_exact=all(q["exact"] for q in quality.values()) and len(quality) == sum(v == "SCORED" for v in expected.values()),
            all_seven_anchors_scored=sorted({q["anchor"] for q in quality.values() if q["anchor"] is not None}) == sorted(A),
            history_chain=chain, history_exact_vs_reference=history_exact,
            bind_refused_after_prediction_released=by_frame[f[9]].get("score_abandon_status") == "WAITING_DROPPED",
            bind_refused_before_prediction_no_scorer_entry=by_frame[f[10]].get("score_abandon_status") is None
                and not by_frame[f[10]].get("prediction_queued_to_scorer"),
            refused_frames_not_committed=not any(by_frame[x].get("committed") for x in (f[9], f[10])),
            unscored_emit_fault_not_score=all(v["q_perc"] is None and v["reason"] == "GROUND_TRUTH_UNAVAILABLE"
                                              for v in faults.values()) and len(faults) == len(fault_frames),
            scorer_released_everything=score_snapshot.get("waiting_predictions") == 0 and score_snapshot.get("retained_slots") == 0,
            producer_released_everything=prep_snapshot.get("retained_jobs") == 0,
            edge_emission_exactly_once=all(v == 1 for v in emitted.values()) and set(emitted) == set(rig.feedback),
            on_plan_nonblocking=all(r["on_plan_returned_quickly"] for r in rig.rows.values()),
            ue_errors=not ue_result["errors"], service_errors=not ue_result["service"]["errors"],
            edge_failure=edge_report["failure"] is None, service_exit_zero=ue_result["service_returncode"] == 0,
            all_retired=ue_result["service"]["outcomes"].get("None", 0) == 0)
        status = "PASS" if all(checks.values()) else "FAIL"
        result = dict(schema="splitfusion.exact_gt_live_v2.rehearsal.v1", status=status, checks=checks,
            outcomes={str(k): v for k, v in outcomes.items()}, expected={str(k): v for k, v in expected.items()},
            quality={str(k): v for k, v in quality.items()}, faults={str(k): v for k, v in faults.items()},
            scenarios=scenarios, refusals=service_result.get("refusals"), scorer_snapshot=score_snapshot,
            edge_report={k: v for k, v in edge_report.items() if k not in ("frames", "records")},
            edge_emissions_per_frame=dict(collections.Counter(emitted.values())),
            ue_result=ue_result, sources=sources, all_children_exit_zero=checks["service_exit_zero"],
            live_services_started=False, deadline_claim=False, performance_claim=False,
            history_committed=committed)
        print(json.dumps(dict(status=status, checks=checks), indent=1))
    except BaseException as error:
        result = dict(status="FAIL", error=f"{type(error).__name__}: {error}", scenarios=scenarios,
                      live_services_started=False, deadline_claim=False)
        if rig is not None and rig.h.process is not None and rig.h.process.poll() is None:
            rig.h.abort(error)
        raise
    finally:
        (out / "RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
        manifest = {str(p.relative_to(out)): CC.sha256(p.read_bytes()) for p in sorted(out.rglob("*"))
                    if p.is_file() and p.name != "OUTPUT_SHA256.json" and "archive" not in p.parts}
        (out / "OUTPUT_SHA256.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    run(parser.parse_args().name)
