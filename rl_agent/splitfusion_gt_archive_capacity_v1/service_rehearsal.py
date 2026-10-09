"""Offline start, use and clean stop of the ACTUAL exact-GT v2 service composition at the exact 1,000-frame limits.

The retained v2 rehearsal rig (``EarlyGtUeHandoff`` -> host service process with its real prep/score workers and the
repaired ArchiveClient; ``EarlyGtEvaluatorV1``; v11 fixtures stand in for CARLA/radio/runtime), with the pilot
registration's exact ``exact_gt_limits`` for budget 1000 (max_frames 1000 -> archive and scorer sized 4,000, CPUs
4/5/6) and the producer registered for 1,000 frames. This is the start-up that failed live with
``ArchiveError: archive frame bound differs``. Four frames (two target-first, two prediction-first) go through the
whole lifecycle to show archive durability and exact scoring at this configuration; then the service is closed
and every child must exit 0. Quality values are mechanics checks, not performance evidence.

Run from the MAIN checkout (v11 fixtures live there):
  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
    /usr/bin/python3 -m rl_agent.splitfusion_gt_archive_capacity_v1.service_rehearsal --out <dir>
"""
import argparse
import collections
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
from rl_agent.splitfusion_run5_pilot_live_v1 import contract as PL

ROOT = Path(__file__).resolve().parents[2]
BUDGET = 1000


class CapacityRig(R.Rig):
    """``R.Rig`` with the registered 1,000-frame engine bound instead of its fixed 100."""
    def __init__(self, out, corpus, limits):
        self.c = corpus
        self.cases = corpus.cases
        self.ue_dir = out / "ue_attempt"
        self.ue_dir.mkdir()
        self.evidence = out / "edge_evidence"
        self.evidence.mkdir(parents=True)
        registration = dict(exact_gt_limits=limits, exact_gt_sources=EC.exact_sources(ROOT),
            numeric_run_binding=corpus.binding, numeric_producer_binding=corpus.startup["binding_sha256"],
            numeric_prediction_contract=corpus.contract, exact_gt_socket_name=EC.SOCKET_NAME,
            exact_gt_capability_name=EC.CAPABILITY_NAME)
        self.h = EarlyGtUeHandoff(attempt_dir=self.ue_dir, registration=registration, service_config={})
        self.h.collector = _owned_collector(self.cases)
        self.h.evidence_dir = self.evidence
        engine = dict(producer_args=dict(corpus.producer_args(), max_frames=int(limits["max_frames"])),
            run_binding=corpus.binding, producer_binding=corpus.startup["binding_sha256"],
            prediction_contract=corpus.contract, max_frames=int(limits["max_frames"]),
            max_retained_jobs=int(limits["prep_jobs"]), max_retained_bytes=int(limits["prep_jobs"]) * 64 * 1024 * 1024,
            offline_replay=False)
        self.engine = engine
        self.h.start_service(engine)          # the exact ue.py start-up path (score/archive sized 4 x max_frames)
        self.payloads = queue.Queue()
        self.feedback = {}
        self.rows = {}


def run(out):
    P.environment_ok()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    corpus = P.Corpus(49)
    limits = PL.limits(BUDGET)                # EXACT registered limits (no rehearsal overrides)
    sources = EC.exact_sources(ROOT)
    (out / "PLAN.json").write_text(json.dumps(dict(schema="splitfusion.gt_archive_capacity_v1.service_rehearsal_plan.v1",
        sources=sources, limits=limits, archive_and_scorer_max_frames=4 * limits["max_frames"], frames=4,
        live_services_started=False, performance_claim=False), indent=1, sort_keys=True))
    spec, _ = quality_bridge()
    rig, result = None, None
    try:
        started = time.monotonic()
        rig = CapacityRig(out, corpus, limits)
        ready_s = time.monotonic() - started
        with install_early_gt_edge(socket_path=rig.evidence / EC.SOCKET_NAME,
                                   capability_path=rig.evidence / EC.CAPABILITY_NAME, connect_timeout_s=30):
            rig.ev = EarlyGtEvaluatorV1(spec=spec, read_ground_truth=None, send=rig.payloads.put, match_distance_m=2.,
                                        gt_timeout_s=2.)
            rig.ev.start()
            f = [c.frame_id for c in rig.cases]
            order = {}
            for i in range(4):
                prediction_first = bool(i % 2)
                rig.admit(i)
                if not prediction_first:
                    time.sleep(0.8)
                rig.predict(f[i]); rig.await_feedback(f[i]); rig.terminal(f[i])
                order[f[i]] = "PREDICTION_FIRST" if prediction_first else "TARGET_FIRST"
            edge_report = rig.ev.close()
        ue_result = rig.h.close()
        service_dir = rig.ue_dir / "numeric_gt/service"
        service = [json.loads(line) for line in open(service_dir / "FRAMES.jsonl")]
        service_result = json.loads((service_dir / "SERVICE_RESULT.json").read_text())
        from rl_agent.splitfusion_fixed_action_screen_analysis_v2.analyze import rescore
        ref = P.AnchorReferenceDriver(corpus)
        index = {c.frame_id: i for i, c in enumerate(rig.cases)}
        frames = {}
        for row in service:
            fid = row["frame_id"]
            case = rig.cases[index[fid]]
            expected_q = P.score_exact(corpus, case, ref.run(case)["packet"])["q_perc"]
            archive = service_dir / "archive" / f"frame_{fid}"
            fb = rig.feedback.get(fid) or {}
            q_service = (row.get("result") or {}).get("q_perc")
            frames[str(fid)] = dict(order=order.get(fid), outcome=row.get("outcome"), q_service=q_service,
                q_emitted=fb.get("q_perc"), q_reference=expected_q,
                producer_archived=sorted(p.name for p in (archive / "producer").glob("*")),
                scorer_archived=sorted(p.name for p in (archive / "scorer").glob("*")),
                archive_rescore_exact=rescore(archive / "scorer") if row.get("outcome") == "SCORED" else False,
                exact=row.get("outcome") == "SCORED" and q_service == expected_q == fb.get("q_perc"))
        ref.close()
        commits = [row["commit"] for row in service if row.get("committed")]
        closed = service_result.get("closed") or {}
        emitted = collections.Counter(int(r["frame_id"]) for r in rig.ev.records)
        checks = dict(
            service_ready_with_exact_1000_limits=service_result.get("limits") == limits and limits["max_frames"] == 1000,
            four_frames_scored=len(frames) == 4 and all(v["outcome"] == "SCORED" for v in frames.values()),
            quality_exact=all(v["exact"] for v in frames.values()),
            archive_durable_producer_and_scorer=all(v["producer_archived"] and v["scorer_archived"] for v in frames.values()),
            archive_rescore_exact=all(v["archive_rescore_exact"] for v in frames.values()),
            history_chain=all(a["history_after_sha256"] == b["history_before_sha256"] for a, b in zip(commits, commits[1:])),
            edge_emission_exactly_once=all(v == 1 for v in emitted.values()) and set(emitted) == set(rig.feedback),
            scorer_released_everything=(closed.get("score") or {}).get("waiting_predictions") == 0
                and (closed.get("score") or {}).get("retained_slots") == 0,
            producer_released_everything=((closed.get("prep") or {}).get("owner") or {}).get("retained_jobs") == 0,
            ue_errors=not ue_result["errors"], service_errors=not ue_result["service"]["errors"],
            edge_failure=edge_report["failure"] is None, service_exit_zero=ue_result["service_returncode"] == 0,
            all_retired=ue_result["service"]["outcomes"].get("None", 0) == 0)
        status = "PASS" if all(checks.values()) else "FAIL"
        result = dict(schema="splitfusion.gt_archive_capacity_v1.service_rehearsal.v1", status=status, checks=checks,
            frames=frames, limits=limits, engine={k: v for k, v in rig.engine.items() if k != "prediction_contract"},
            archive_and_scorer_max_frames=4 * limits["max_frames"], service_ready_s=round(ready_s, 3),
            refusals=service_result.get("refusals"), ue_result=ue_result, sources=sources,
            all_children_exit_zero=checks["service_exit_zero"], live_services_started=False, performance_claim=False,
            quality_values_are_mechanics_only=True)
        print(json.dumps(dict(status=status, checks=checks), indent=1))
    except BaseException as error:
        result = dict(status="FAIL", error=f"{type(error).__name__}: {error}", live_services_started=False)
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
    parser.add_argument("--out", required=True)
    run(parser.parse_args().out)
