"""Offline start, use and clean stop of the ACTUAL host early-GT service through the GC-qualification trampolines,
with the exact registered 1,000-frame limits, for BOTH arms (control, then candidate).

The retained v2 rehearsal rig (``EarlyGtUeHandoff`` -> host service with real prep/score workers and archive;
``EarlyGtEvaluatorV1``; v11 fixtures stand in for CARLA/radio/runtime). The UE-side seam is the same
``install_ue_service_seam`` the live child installs. Twelve frames per arm exercise the candidate's warm-up freeze
(after 2 decisions) and several idle collections. Checks: substitution shapes and roles, policy installed only in the
candidate prep worker, GC state restored at worker exit, traces written and bounded, exact scoring and archive
rescore, service/workers exit 0. Quality values are mechanics checks; timings are not performance evidence.

Run from the MAIN checkout (v11 fixtures live there):
  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
    /usr/bin/python3 -m rl_agent.splitfusion_run5_full_live_gc_v1.gc_service_rehearsal --out <dir>
"""
import argparse
import json
from pathlib import Path
import time

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as CC
from rl_agent.splitfusion_exact_gt_service_proto_v2 import parity_common as P
from rl_agent.splitfusion_exact_gt_live_v2 import contract as EC
from rl_agent.splitfusion_exact_gt_live_v2 import rehearsal as R
from rl_agent.splitfusion_exact_gt_live_v2.edge import install_early_gt_edge, EarlyGtEvaluatorV1
from rl_agent.splitfusion_exact_gt_live_v2.ue import EarlyGtUeHandoff
from rl_agent.splitfusion_numeric_gt_producer_v1.replay_v11 import quality_bridge
from rl_agent.splitfusion_numeric_gt_runtime_v1.async_live_rehearsal import _owned_collector

from . import contract as C
from . import gc_trace
from .gc_seams import install_ue_service_seam

ROOT = Path(__file__).resolve().parents[2]
BUDGET = 1000
FRAMES = 12


class GcRig(R.Rig):
    def __init__(self, out, corpus, limits):
        self.c, self.cases = corpus, corpus.cases
        self.ue_dir = out / "ue_attempt"
        self.ue_dir.mkdir()
        self.evidence = out / "edge_evidence"
        self.evidence.mkdir(parents=True)
        registration = dict(exact_gt_limits=limits, exact_gt_sources=C.exact_sources(ROOT),
            numeric_run_binding=corpus.binding, numeric_producer_binding=corpus.startup["binding_sha256"],
            numeric_prediction_contract=corpus.contract, exact_gt_socket_name=EC.SOCKET_NAME,
            exact_gt_capability_name=EC.CAPABILITY_NAME)
        self.h = EarlyGtUeHandoff(attempt_dir=self.ue_dir, registration=registration, service_config={})
        self.h.collector = _owned_collector(self.cases)
        self.h.evidence_dir = self.evidence
        self.engine = dict(producer_args=dict(corpus.producer_args(), max_frames=int(limits["max_frames"])),
            run_binding=corpus.binding, producer_binding=corpus.startup["binding_sha256"],
            prediction_contract=corpus.contract, max_frames=int(limits["max_frames"]),
            max_retained_jobs=int(limits["prep_jobs"]), max_retained_bytes=int(limits["prep_jobs"]) * 64 * 1024 * 1024,
            offline_replay=False)
        self.h.start_service(self.engine)
        import queue
        self.payloads, self.feedback, self.rows = queue.Queue(), {}, {}


def arm_run(out, arm, corpus, limits, spec):
    out.mkdir()
    seam_reg = dict(id=C.ID, transmitted_budget=BUDGET, gt_gc=C.gc_registration(arm, BUDGET))
    rig = None
    with install_ue_service_seam(registration=seam_reg, output=out) as trace_dir:
        try:
            started = time.monotonic()
            rig = GcRig(out, corpus, limits)
            ready_s = time.monotonic() - started
            with install_early_gt_edge(socket_path=rig.evidence / EC.SOCKET_NAME,
                                       capability_path=rig.evidence / EC.CAPABILITY_NAME, connect_timeout_s=30):
                rig.ev = EarlyGtEvaluatorV1(spec=spec, read_ground_truth=None, send=rig.payloads.put,
                                            match_distance_m=2., gt_timeout_s=2.)
                rig.ev.start()
                f = [c.frame_id for c in rig.cases]
                for i in range(FRAMES):
                    rig.admit(i)
                    if i % 2 == 0:
                        time.sleep(0.25)
                    rig.predict(f[i]); rig.await_feedback(f[i]); rig.terminal(f[i])
                    time.sleep(0.12)                     # an inter-decision gap, as live
                edge_report = rig.ev.close()
            ue_result = rig.h.close()
        except BaseException as error:
            if rig is not None and rig.h.process is not None and rig.h.process.poll() is None:
                rig.h.abort(error)
            raise
    service_dir = rig.ue_dir / "numeric_gt/service"
    rows = [json.loads(line) for line in open(service_dir / "FRAMES.jsonl")]
    service_result = json.loads((service_dir / "SERVICE_RESULT.json").read_text())
    from rl_agent.splitfusion_fixed_action_screen_analysis_v2.analyze import rescore
    ref = P.AnchorReferenceDriver(corpus)
    index = {c.frame_id: i for i, c in enumerate(rig.cases)}
    exact = []
    for row in rows:
        case = rig.cases[index[row["frame_id"]]]
        expected = P.score_exact(corpus, case, ref.run(case)["packet"])["q_perc"]
        exact.append(row.get("outcome") == "SCORED" and (row.get("result") or {}).get("q_perc") == expected
                     == (rig.feedback.get(row["frame_id"]) or {}).get("q_perc")
                     and rescore(service_dir / "archive" / f"frame_{row['frame_id']}" / "scorer"))
    ref.close()
    traces = {role: json.loads((trace_dir / f"{role}_gc_trace.json").read_text()) for role in ("prep", "score")}
    service_sub = json.loads((trace_dir / "service_substitution.json").read_text())
    ue_sub = json.loads((trace_dir / "ue_substitution.json").read_text())
    prep, score = traces["prep"], traces["score"]
    events = prep["policy_events"]
    idle = [e for e in events if e.get("kind") == "idle_full_collect"]
    freeze = [e for e in events if e.get("kind") == "warmup_collect_freeze"]
    summary = gc_trace.summarize(prep, C.GC_POLICY["warmup_decisions"])
    checks = dict(
        ue_substitution_exactly_one_and_restored=len(ue_sub["substitutions"]) == 1 and ue_sub["restored"] is True
            and C.SERVICE_ENTRY in ue_sub["substitutions"][0]["argv"],
        service_substitution_roles=sorted((s["role"], s["gc_policy"]) for s in service_sub["substitutions"])
            == sorted([("prep", arm), ("score", "control")]) and service_sub["returncode"] == 0,
        roles_seen=prep["role_seen"] == "prep" and score["role_seen"] == "score" and not prep["refusal"] and not score["refusal"],
        policy_only_in_candidate_prep=prep["policy_installed"] is (arm == "candidate") and score["policy_installed"] is False,
        gc_restored_both=prep["restored_equals_initial"] is True and score["restored_equals_initial"] is True,
        candidate_freeze_and_idle=(arm == "control" and not events) or (arm == "candidate" and len(freeze) == 1
            and freeze[0]["decision"] == C.GC_POLICY["warmup_decisions"] and len(idle) >= FRAMES - 4
            and prep["safety_stop"] is None),
        traces_bounded=prep["dropped"] == {"ops": 0, "gen2": 0} and len(prep["ops"]) <= prep["limits"]["max_ops"],
        frames_scored_exact=len(rows) == FRAMES and all(exact),
        service_ready_with_exact_1000_limits=service_result.get("limits") == limits and limits["max_frames"] == 1000,
        service_exit_zero=ue_result["service_returncode"] == 0 and not ue_result["errors"]
            and not ue_result["service"]["errors"] and edge_report["failure"] is None)
    return dict(arm=arm, status="PASS" if all(checks.values()) else "FAIL", checks=checks, ready_s=round(ready_s, 3),
                prep_ops=len(prep["ops"]), prep_gen2=len(prep["gen2"]), prep_gen2_feedback_after_warmup=summary["gen2_feedback_after_warmup"],
                prep_gen2_feedback_warmup=summary["gen2_feedback_warmup"], prep_decisions=summary["decisions"],
                idle_collections=len(idle), idle_wall_ms_max=max((e["wall_ms"] for e in idle), default=None),
                idle_request_arrivals=sum(bool(e.get("request_arrived_during")) for e in idle),
                skipped_pending=sum(e.get("kind") == "skipped_pending" for e in events),
                freeze_count=freeze[0]["freeze_count"] if freeze else None,
                gc_state_at_close=prep["gc_state_at_close"], gc_state_after_restore=prep["gc_state_after_restore"])


def run(out):
    P.environment_ok()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    corpus = P.Corpus(49)
    limits = C.limits(BUDGET)
    sources = C.exact_sources(ROOT)
    (out / "PLAN.json").write_text(json.dumps(dict(schema="splitfusion.run5_full_live_gc.service_rehearsal_plan.v1",
        sources=sources, limits=limits, frames_per_arm=FRAMES, arms=list(C.ARMS), live_services_started=False,
        performance_claim=False), indent=1, sort_keys=True))
    spec, _ = quality_bridge()
    result = None
    try:
        arms = {arm: arm_run(out / arm, arm, corpus, limits, spec) for arm in C.ARMS}
        status = "PASS" if all(a["status"] == "PASS" for a in arms.values()) else "FAIL"
        result = dict(schema="splitfusion.run5_full_live_gc.service_rehearsal.v1", status=status, arms=arms,
                      limits=limits, sources=sources, all_children_exit_zero=all(a["checks"]["service_exit_zero"]
                      for a in arms.values()), live_services_started=False, performance_claim=False)
        print(json.dumps(dict(status=status, arms={k: v["checks"] for k, v in arms.items()}), indent=1))
    except BaseException as error:
        result = dict(status="FAIL", error=f"{type(error).__name__}: {error}", live_services_started=False)
        raise
    finally:
        (out / "RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
        manifest = {str(p.relative_to(out)): CC.sha256(p.read_bytes()) for p in sorted(out.rglob("*"))
                    if p.is_file() and p.name != "OUTPUT_SHA256.json" and "archive" not in p.parts}
        (out / "OUTPUT_SHA256.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    run(parser.parse_args().out)
