"""Offline rehearsal THROUGH THE INTENDED LAUNCHER: actual host early-GT service, started by the gc_v2 UE seam
(``install_ue_service_seam``) -> ``service_entry`` -> guarded ``worker_entry`` (role checks, adopted GC policy,
safety stops, tracing, producer switch), with the exact registered 1,000-frame limits.

Two arms, identical scenario sequence:
  candidate_v2           adopted GC policy + numeric producer v2 (the registered validation configuration)
  candidate_v1_rollback  adopted GC policy + producer v1 (the registered rollback; no substitution)
Scenarios: 8 scored frames (target-first / prediction-first), a send failure (cancel), an edge terminal before any
prediction, a horizon timeout followed by a late prediction (must not be scored), then 3 more scored frames.
Checks per arm: selected producer actually constructed, substitutions restored, roles, GC policy only in prep, GC
state restored, exact scoring against the reference scorer and archive rescore, commit history chain, every job
retired, service/workers exit 0. Across arms: every frame's outcome, Q_perc and commit digest identical (rollback).

  cd <abiodun>; env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
    /usr/bin/python3 -m rl_agent.splitfusion_run5_full_live_retention_v1.gc_service_rehearsal --out <dir>
"""
import argparse
import hashlib
import json
from pathlib import Path
import queue
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
ARMS = {"candidate_v2": ("candidate", "v2"), "candidate_v1_rollback": ("candidate", "v1")}


class Rig(R.Rig):
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
        self.payloads, self.feedback, self.rows = queue.Queue(), {}, {}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def arm_run(out, name, corpus, limits, spec):
    arm, producer = ARMS[name]
    out.mkdir()
    seam_reg = dict(id=C.ID, transmitted_budget=BUDGET, gt_gc=C.gc_registration(arm, BUDGET, producer))
    rig, scenario = None, {}
    with install_ue_service_seam(registration=seam_reg, output=out) as trace_dir:
        try:
            rig = Rig(out, corpus, limits)
            with install_early_gt_edge(socket_path=rig.evidence / EC.SOCKET_NAME,
                                       capability_path=rig.evidence / EC.CAPABILITY_NAME, connect_timeout_s=30):
                rig.ev = EarlyGtEvaluatorV1(spec=spec, read_ground_truth=None, send=rig.payloads.put,
                                            match_distance_m=2., gt_timeout_s=2.)
                rig.ev.start()
                f = [c.frame_id for c in rig.cases]

                def scored(i, prediction_first):
                    rig.admit(i)
                    if not prediction_first:
                        time.sleep(0.3)
                    rig.predict(f[i]); rig.await_feedback(f[i]); rig.terminal(f[i])
                    time.sleep(0.12)
                for i in range(8):
                    scored(i, prediction_first=bool(i % 2))
                scenario["scored_first"] = f[:8]
                rig.admit(8, send=False); time.sleep(0.4); rig.terminal(f[8])
                scenario["send_failure_cancel"] = f[8]
                rig.admit(9); time.sleep(0.2); rig.ev.notify_terminal(f[9], "STALE_BEFORE_MAP", "DECODE")
                time.sleep(0.5); rig.terminal(f[9])
                scenario["edge_terminal_before_prediction"] = f[9]
                rig.admit(10); time.sleep(limits["horizon_ms"] / 1e3 + 0.6); rig.predict(f[10])
                rig.await_feedback(f[10]); rig.terminal(f[10])
                scenario["horizon_timeout_then_late_prediction"] = f[10]
                for i in (11, 12, 13):
                    scored(i, prediction_first=bool(i % 2))
                scenario["scored_after"] = [f[11], f[12], f[13]]
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
    frames = {}
    for row in rows:
        fid = row["frame_id"]
        case = rig.cases[index[fid]]
        q_service = (row.get("result") or {}).get("q_perc")
        fb = rig.feedback.get(fid) or {}
        entry = dict(outcome=row.get("outcome"), q_service=q_service, q_emitted=fb.get("q_perc"), feedback=fb.get("kind"),
                     committed=bool(row.get("committed")), commit=digest(row.get("commit")))
        if row.get("outcome") == "SCORED":
            expected = P.score_exact(corpus, case, ref.run(case)["packet"])["q_perc"]
            entry.update(q_reference=expected, exact=q_service == expected == fb.get("q_perc"),
                         archive_rescore_exact=rescore(service_dir / "archive" / f"frame_{fid}" / "scorer"))
        frames[str(fid)] = entry
    ref.close()
    commits = [row["commit"] for row in rows if row.get("committed")]
    traces = {role: json.loads((trace_dir / f"{role}_gc_trace.json").read_text()) for role in ("prep", "score")}
    service_sub = json.loads((trace_dir / "service_substitution.json").read_text())
    ue_sub = json.loads((trace_dir / "ue_substitution.json").read_text())
    prep, score = traces["prep"], traces["score"]
    summary = gc_trace.summarize(prep, C.GC_POLICY["warmup_decisions"])
    closed = service_result.get("closed") or {}
    scored_frames = [v for v in frames.values() if v["outcome"] == "SCORED"]
    checks = dict(
        producer_constructed=prep["producer_constructed"] == C.PRODUCER_CLASS[producer],
        producer_switch_restored=(prep["producer_switch_restored"] is True) if producer == "v2" else prep["producer_switch_restored"] is None,
        substitution_args=sorted((s["role"], s["gc_policy"], s["producer"]) for s in service_sub["substitutions"])
            == sorted([("prep", arm, producer), ("score", "control", "v1")]) and service_sub["returncode"] == 0,
        ue_seam_once_restored=len(ue_sub["substitutions"]) == 1 and ue_sub["restored"] is True,
        roles=prep["role_seen"] == "prep" and score["role_seen"] == "score" and not prep["refusal"] and not score["refusal"],
        gc_policy_only_prep=prep["policy_installed"] is True and score["policy_installed"] is False and prep["safety_stop"] is None,
        gc_state_restored=prep["restored_equals_initial"] is True and score["restored_equals_initial"] is True,
        scored_exact=len(scored_frames) == 11 and all(v["exact"] and v["archive_rescore_exact"] for v in scored_frames),
        cancel_and_terminals_unscored=all(frames[str(x)]["outcome"] != "SCORED" for x in (
            scenario["send_failure_cancel"], scenario["edge_terminal_before_prediction"],
            scenario["horizon_timeout_then_late_prediction"])),
        history_chain=all(a["history_after_sha256"] == b["history_before_sha256"] for a, b in zip(commits, commits[1:])),
        all_retired=ue_result["service"]["outcomes"].get("None", 0) == 0
            and (closed.get("score") or {}).get("retained_slots") == 0
            and ((closed.get("prep") or {}).get("owner") or {}).get("retained_jobs") == 0,
        service_ready_with_exact_1000_limits=service_result.get("limits") == limits and limits["max_frames"] == 1000,
        service_exit_zero=ue_result["service_returncode"] == 0 and not ue_result["errors"]
            and not ue_result["service"]["errors"] and edge_report["failure"] is None)
    return dict(arm=name, gc=arm, producer=producer, status="PASS" if all(checks.values()) else "FAIL", checks=checks,
                scenario=scenario, frames=frames, outcomes=ue_result["service"]["outcomes"],
                gc_summary={k: v for k, v in summary.items() if k not in ("freeze_ms", "gen2_all_ms")},
                producer_v2_sources=prep.get("producer_v2_sources"))


def run(out):
    P.environment_ok()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    corpus = P.Corpus(49)
    limits = C.limits(BUDGET)
    sources = C.exact_sources(ROOT)
    (out / "PLAN.json").write_text(json.dumps(dict(schema="splitfusion.run5_full_live_retention_v1.launcher_rehearsal_plan.v1",
        sources=sources, limits=limits, arms=ARMS, live_services_started=False, performance_claim=False), indent=1,
        sort_keys=True))
    spec, _ = quality_bridge()
    result = None
    try:
        arms = {name: arm_run(out / name, name, corpus, limits, spec) for name in ARMS}
        a, b = arms["candidate_v2"]["frames"], arms["candidate_v1_rollback"]["frames"]
        keys = ("outcome", "q_service", "q_emitted", "feedback", "committed", "commit")
        identical = a.keys() == b.keys() and all({k: a[f][k] for k in keys} == {k: b[f][k] for k in keys} for f in a)
        status = "PASS" if identical and all(x["status"] == "PASS" for x in arms.values()) else "FAIL"
        result = dict(schema="splitfusion.run5_full_live_retention_v1.launcher_rehearsal.v1", status=status, arms=arms,
                      rollback_outputs_identical=identical, limits=limits, sources=sources,
                      all_children_exit_zero=all(x["checks"]["service_exit_zero"] for x in arms.values()),
                      live_services_started=False, performance_claim=False)
        print(json.dumps(dict(status=status, rollback_outputs_identical=identical,
                              arms={k: v["checks"] for k, v in arms.items()}), indent=1))
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
