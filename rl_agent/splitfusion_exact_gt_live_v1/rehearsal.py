"""Real multi-process integration rehearsal (CPU only, v11 fixtures, no CARLA/radio/GPU).

Actual code under test: ``EarlyGtUeHandoff`` (stage_before_action, on_plan, after_submit,
terminals, gt-freeze thread, close), the host ``service`` process with its real prep/score
workers and the unchanged ArchiveClient (fsync), and ``EarlyGtEvaluatorV1`` (bounded submit,
sender/receiver threads, retained ``_emit``/``quality_feedback``). Only the CARLA collector,
the Run-4 runtime and the radio are fixture stand-ins (the retained fixture builders of the
numeric-GT rehearsal). Registered scenarios:

  target_first, prediction_first, send_failure, horizon_timeout_then_late_prediction,
  edge_terminal, refused_freeze_void, scorer_capacity, producer_capacity, shutdown_drain.
Durations are incidental and not performance evidence.
"""
import argparse
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
from rl_agent.splitfusion_exact_gt_service_proto_v1 import parity_common as P

from . import contract as C
from .ue import EarlyGtUeHandoff, raw
from .edge import install_early_gt_edge, EarlyGtEvaluatorV1

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "experiments/splitfusion_exact_gt_live_v1"


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

    def admit(self, index, *, send=True):
        case = self.cases[index]
        frame_id = case.frame_id
        expected = self.c.expected(case)
        action, capture, context = expected["expected_action_identity"], case.kwargs["identity"], expected["frame_context"]
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
                   runtime=runtime, on_plan_returned_quickly=on_plan_s < 0.05)
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
    limits = dict(C.LIMITS, prep_jobs=4, scorer_jobs=2, horizon_ms=1500, drain_s=60, max_frames=49,
                  prep_cpu=None, score_cpu=None, service_cpu=None)
    sources = C.exact_sources(ROOT)
    plan = dict(schema="splitfusion.exact_gt_live.rehearsal_plan.v1", sources=sources, limits=limits,
                scenarios="see module docstring", live_services_started=False, deadline_claim=False)
    (out / "PLAN.json").write_text(json.dumps(plan, indent=1, sort_keys=True))
    spec, _ = quality_bridge()
    rig, result, scenarios = None, None, {}
    try:
        rig = Rig(out, corpus, limits)
        with install_early_gt_edge(socket_path=rig.evidence / C.SOCKET_NAME,
                                   capability_path=rig.evidence / C.CAPABILITY_NAME, connect_timeout_s=30):
            rig.ev = EarlyGtEvaluatorV1(spec=spec, read_ground_truth=None, send=rig.payloads.put, match_distance_m=2.,
                                        gt_timeout_s=2.)
            rig.ev.start()
            f = [c.frame_id for c in rig.cases]
            # target_first
            rig.admit(0); time.sleep(0.8); rig.predict(f[0]); rig.await_feedback(f[0]); rig.terminal(f[0])
            scenarios["target_first"] = f[0]
            # prediction_first (prediction before the freeze has been computed)
            rig.admit(1); rig.predict(f[1]); rig.await_feedback(f[1]); rig.terminal(f[1])
            scenarios["prediction_first"] = f[1]
            # send_failure (no prediction can exist); UE times out
            rig.admit(2, send=False); time.sleep(0.4); rig.terminal(f[2])
            scenarios["send_failure"] = f[2]
            # horizon timeout, then a late prediction that must not be scored
            rig.admit(3); time.sleep(limits["horizon_ms"] / 1e3 + 0.6); rig.predict(f[3]); rig.await_feedback(f[3])
            rig.terminal(f[3])
            scenarios["horizon_timeout_then_late_prediction"] = f[3]
            # edge terminal before any prediction
            rig.admit(4); time.sleep(0.2); rig.ev.notify_terminal(f[4], "STALE_BEFORE_MAP", "DECODE")
            time.sleep(0.5); rig.terminal(f[4])
            scenarios["edge_terminal"] = f[4]
            # refused freeze -> VOID (5 open without send, 6 queued, 7 refused)
            rig.admit(5, send=None); time.sleep(0.05); rig.admit(6, send=None); time.sleep(0.05)
            rig.admit(7, send=None); time.sleep(0.3)
            for i in (5, 6, 7):
                rig.send(f[i])
            for i in (5, 6, 7):
                rig.predict(f[i])
            for i in (5, 6, 7):
                rig.await_feedback(f[i]); rig.terminal(f[i])
            scenarios["refused_freeze_void"] = [f[5], f[6], f[7]]
            time.sleep(1.0)              # let 0..7 retire so capacity starts empty
            # scorer capacity: 8 and 9 hold slots (terminals withheld), 10's target is refused
            for i in (8, 9):
                rig.admit(i); time.sleep(0.6); rig.predict(f[i]); rig.await_feedback(f[i])
            rig.admit(10); time.sleep(0.8); rig.predict(f[10]); rig.await_feedback(f[10])
            scenarios["scorer_capacity"] = f[10]
            # producer capacity: 8, 9, 10 committed and unretired -> 11 refused (VOID)
            rig.admit(11); time.sleep(0.3); rig.predict(f[11]); rig.await_feedback(f[11])
            scenarios["producer_capacity"] = f[11]
            for i in (8, 9, 10, 11):
                rig.terminal(f[i])
            # shutdown drain: scored frame whose UE terminal never arrives before close
            time.sleep(1.5)
            rig.admit(12); time.sleep(0.6); rig.predict(f[12]); rig.await_feedback(f[12])
            scenarios["shutdown_drain"] = f[12]
            edge_report = rig.ev.close()
        ue_result = rig.h.close()
        service = [json.loads(l) for l in open(rig.ue_dir / "numeric_gt/service/FRAMES.jsonl")]
        outcomes = {row["frame_id"]: row.get("outcome") for row in service}
        expected = {f[0]: "SCORED", f[1]: "SCORED", f[2]: ("CANCELLED_SEND_FAILED", "NO_GT_SEND_FAILED_BEFORE_ADMISSION"),
                    f[3]: "UNSCORED_NO_RESULT_BY_HORIZON", f[4]: "UNSCORED_EDGE_TERMINAL_STALE_BEFORE_MAP",
                    f[5]: "SCORED", f[6]: "SCORED", f[7]: "NO_GT_FREEZE_QUEUE_FULL", f[8]: "SCORED", f[9]: "SCORED",
                    f[10]: "UNSCORED_SCORER_SCORER_CAPACITY", f[11]: "NO_GT_PRODUCER_CAPACITY", f[12]: "SCORED"}
        committed = [row["frame_id"] for row in service if row.get("committed")]
        commits = [row["commit"] for row in service if row.get("committed")]
        chain = all(a["history_after_sha256"] == b["history_before_sha256"] for a, b in zip(commits, commits[1:]))
        index = {c.frame_id: i for i, c in enumerate(rig.cases)}
        ref = P.ReferenceDriver(corpus)
        reference = {fid: ref.run(rig.cases[index[fid]]) for fid in committed}
        ref.close()
        quality = {}
        for row in service:
            if row.get("outcome") == "SCORED":
                case = rig.cases[index[row["frame_id"]]]
                expected_q = P.score_exact(corpus, case, reference[row["frame_id"]]["packet"])["q_perc"]
                fb = rig.feedback[row["frame_id"]]
                quality[row["frame_id"]] = dict(service=row["result"]["q_perc"], reference=expected_q,
                    emitted=fb["q_perc"], exact=row["result"]["q_perc"] == expected_q == fb["q_perc"])
        history_exact = commits[-1]["history_after_sha256"] == reference[committed[-1]]["history_after"]
        faults = {fid: rig.feedback[fid] for fid in (f[3], f[7], f[10], f[11]) if fid in rig.feedback}
        checks = dict(outcomes=all(outcomes.get(k) in (v if isinstance(v, tuple) else (v,)) for k, v in expected.items()),
            quality_exact=all(q["exact"] for q in quality.values()) and len(quality) == 7,
            history_chain=chain, history_exact_vs_reference=history_exact,
            unscored_emit_fault_not_score=all(v["q_perc"] is None and v["reason"] == "GROUND_TRUTH_UNAVAILABLE"
                                              for v in faults.values()) and len(faults) == 4,
            on_plan_nonblocking=all(r["on_plan_returned_quickly"] for r in rig.rows.values()),
            ue_errors=not ue_result["errors"], service_errors=not ue_result["service"]["errors"],
            edge_failure=edge_report["failure"] is None, service_exit_zero=ue_result["service_returncode"] == 0,
            all_retired=ue_result["service"]["outcomes"].get("None", 0) == 0)
        status = "PASS" if all(checks.values()) else "FAIL"
        result = dict(schema="splitfusion.exact_gt_live.rehearsal.v1", status=status, checks=checks,
            outcomes={str(k): v for k, v in outcomes.items()}, expected={str(k): v for k, v in expected.items()},
            quality={str(k): v for k, v in quality.items()}, faults={str(k): v for k, v in faults.items()},
            scenarios=scenarios, edge_report={k: v for k, v in edge_report.items() if k != "frames"},
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
