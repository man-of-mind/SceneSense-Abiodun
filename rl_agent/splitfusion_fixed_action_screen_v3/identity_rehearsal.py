"""Offline lifecycle of the three registered identities through the real early-GT v2 service (CPU only).

Uses the retained v2 rehearsal rig unchanged (``EarlyGtUeHandoff`` with the host service process and its real
prep/score workers and ArchiveClient, ``EarlyGtEvaluatorV1``; v11 fixtures stand in for CARLA/radio/runtime).
Each arm's exact dynamic-contract identity is placed on sealed v11 sources -- anchors 67/69 as catalogue-anchor
captures (anchor provenance path), the off-anchor m11 q8000 as the action-neutral template capture with
``anchor_action_id=None`` -- and run target-first and prediction-first. Checks per frame: SCORED; service ==
emitted == independent reference Q_perc; anchor provenance archived for anchors and ABSENT for the off-anchor;
the archived scorer directory re-scores exactly with the live integration check's ``rescore``; exactly one edge
emission; unbroken history chain; service/UE/edge exit clean. Only identity fields change; prediction labels and
every source byte are the sealed v11 ones, so the quality values are mechanics checks, not performance evidence.

Run from the MAIN checkout (v11 fixtures live there):
  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
    /usr/bin/python3 -m rl_agent.splitfusion_fixed_action_screen_v3.identity_rehearsal --name identity_rehearsal_01
"""
import argparse
import collections
import json
from pathlib import Path
import time

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as CC
from rl_agent.splitfusion_exact_gt_service_proto_v2 import parity_common as P
from rl_agent.splitfusion_exact_gt_live_v2 import contract as EC
from rl_agent.splitfusion_exact_gt_live_v2 import rehearsal as R
from rl_agent.splitfusion_exact_gt_live_v2.edge import install_early_gt_edge, EarlyGtEvaluatorV1
from rl_agent.splitfusion_numeric_gt_producer_v1.replay_v11 import quality_bridge
from rl_agent.splitfusion_numeric_gt_runtime_v1.async_live_rehearsal import _planned, _pose
from types import SimpleNamespace as NS

from . import contract as C

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "experiments/splitfusion_fixed_action_screen_v3"


def identity_variant(corpus, case, forced):
    """Sealed source with the arm's exact identity as the selected action (mirrors ``P.anchor_variant``)."""
    plan = CC._parse(corpus.plan(case))
    CC.require(plan["capture"]["action_id"] is None and plan["action"]["anchor_action_id"] is None,
               "fixture source must be an action-neutral template")
    capture = dict(plan["capture"]) if forced["action_id"] is None else \
        dict(plan["capture"], action_id=forced["action_id"], profile_id=forced["profile_id"])
    action = dict(plan["action"], mode_id=forced["mode_id"], q_e4=forced["q_e4"], anchor_action_id=forced["action_id"],
                  execution_bundle_sha256=forced["execution_bundle_sha256"])
    plan_bytes = CC.canonical(dict(plan, capture=capture, action=action))
    pred = corpus.predictions[case.frame_id]
    identity = dict(CC._json_owned(pred["identity"]), gt_identity=capture, action_identity=action)
    edge_identity = {k: identity[k] for k in ("gt_identity", "action_identity", "capture_timestamp_ns", "prediction_digest")}
    return dict(action_id=forced["action_id"], plan_bytes=plan_bytes, capture=capture, action=action,
                prediction_identity=identity, edge_identity=edge_identity, profile_id=forced["profile_id"])


class IdentityRig(R.Rig):
    def admit_variant(self, index, variant):
        case = self.cases[index]
        frame_id = case.frame_id
        context = self.c.expected(case)["frame_context"]
        action, capture = variant["action"], variant["capture"]
        runtime = NS(pipeline=NS(run_id=capture["run_id"], cell_id=capture["cell_id"], stream_id=capture["stream_id"],
                                 decisions=[]), engine=NS(lineage=action["controller_lineage_sha256"]),
                     _numeric_current_ego_pose=_pose(context), last_datagram={})
        self.h.stage_before_action(runtime, frame_id=frame_id, carla_timestamp=case.kwargs["policy_timestamp_s"],
                                   capture_timestamp_ns=capture["capture_timestamp_ns"], ego_pose=_pose(context))
        opened = R.raw()
        runtime.pipeline.decisions.append(dict(frame_id=frame_id, action_open=dict(domain="CLOCK_MONOTONIC_RAW", ns=opened)))
        t0 = time.monotonic()
        self.h.on_plan(runtime, _planned(action, capture, context))
        row = dict(index=index, frame_id=frame_id, action=action, capture=capture, context=context, opened=opened,
                   runtime=runtime, on_plan_returned_quickly=time.monotonic() - t0 < 0.05, anchor=variant["action_id"],
                   tamper=False, variant=variant)
        self.rows[frame_id] = row
        self.send(frame_id, True)
        return row


def run(name):
    from rl_agent.splitfusion_fixed_action_screen_analysis_v2.analyze import rescore
    P.environment_ok()
    out = EVIDENCE / name
    out.mkdir(parents=True, exist_ok=False)
    corpus = P.Corpus(49)
    limits = dict(EC.LIMITS, prep_jobs=4, scorer_jobs=2, max_waiting=1, horizon_ms=1500, drain_s=60, max_frames=49,
                  prep_cpu=None, score_cpu=None, service_cpu=None)
    resolved = {key: C.resolve(mode, q) for key, mode, q in C.ARMS}
    sources = C.exact_sources(ROOT)
    (out / "PLAN.json").write_text(json.dumps(dict(schema="splitfusion.fixed_action_screen_v3.identity_rehearsal_plan.v1",
        sources=sources, limits=limits, arms=resolved, frames_per_arm=2, orders=["TARGET_FIRST", "PREDICTION_FIRST"],
        live_services_started=False, performance_claim=False), indent=1, sort_keys=True))
    spec, _ = quality_bridge()
    rig, result, plan = None, None, {}
    try:
        rig = IdentityRig(out, corpus, limits)
        with install_early_gt_edge(socket_path=rig.evidence / EC.SOCKET_NAME,
                                   capability_path=rig.evidence / EC.CAPABILITY_NAME, connect_timeout_s=30):
            rig.ev = EarlyGtEvaluatorV1(spec=spec, read_ground_truth=None, send=rig.payloads.put, match_distance_m=2.,
                                        gt_timeout_s=2.)
            rig.ev.start()
            index = 0
            for key, _, _ in C.ARMS:
                for prediction_first in (False, True):
                    case = rig.cases[index]
                    variant = identity_variant(corpus, case, resolved[key])
                    rig.admit_variant(index, variant)
                    if not prediction_first:
                        time.sleep(0.8)
                    rig.predict(case.frame_id)
                    rig.await_feedback(case.frame_id)
                    rig.terminal(case.frame_id)
                    plan[case.frame_id] = dict(arm=key, prediction_first=prediction_first)
                    index += 1
            edge_report = rig.ev.close()
        ue_result = rig.h.close()
        service_dir = rig.ue_dir / "numeric_gt/service"
        service = [json.loads(l) for l in open(service_dir / "FRAMES.jsonl")]
        by_frame = {row["frame_id"]: row for row in service}
        ref = P.AnchorReferenceDriver(corpus)
        positions = {c.frame_id: i for i, c in enumerate(rig.cases)}
        from rl_agent.splitfusion_exact_gt_service_proto_v2.verify_anchor import pinned_score, as_bytes
        from rl_agent.splitfusion_exact_gt_service_proto_v2.scorer_core import expected_from_identity
        from rl_agent.splitfusion_numeric_gt_integration_v2 import ready as RR
        frames = {}
        for fid, meta in plan.items():
            row, case = rig.rows[fid], rig.cases[positions[fid]]
            reference = ref.run_anchor(case, row["variant"])
            _, exp = expected_from_identity(as_bytes(reference["identity_bytes"]))
            _, expected_q = pinned_score(corpus, RR.decode_target(reference["packet"], **exp),
                                         row["variant"]["prediction_identity"], case)
            service_row = by_frame.get(fid) or {}
            archive = service_dir / "archive" / f"frame_{fid}"
            anchored = row["anchor"] is not None
            provenance = (archive / "scorer/anchor_provenance.json").exists() and (archive / "producer/anchor_provenance.json").exists()
            absent = not (archive / "scorer/anchor_provenance.json").exists() and not (archive / "producer/anchor_provenance.json").exists()
            fb = rig.feedback.get(fid) or {}
            q_service = (service_row.get("result") or {}).get("q_perc")
            frames[str(fid)] = dict(meta, outcome=service_row.get("outcome"), q_service=q_service, q_emitted=fb.get("q_perc"),
                q_reference=expected_q, action=dict(mode_id=row["action"]["mode_id"], q_e4=row["action"]["q_e4"],
                anchor_action_id=row["action"]["anchor_action_id"]), capture_action_id=row["capture"]["action_id"],
                provenance_archived=provenance, provenance_absent=absent,
                provenance_correct=provenance if anchored else absent,
                archive_rescore_exact=rescore(archive / "scorer") if service_row.get("outcome") == "SCORED" else False,
                exact=service_row.get("outcome") == "SCORED" and q_service == expected_q == fb.get("q_perc"))
        ref.close()
        commits = [row["commit"] for row in service if row.get("committed")]
        emitted = collections.Counter(int(r["frame_id"]) for r in rig.ev.records)
        score_snapshot = ((json.loads((service_dir / "SERVICE_RESULT.json").read_text()).get("closed") or {}).get("score") or {})
        checks = dict(
            all_scored=all(f["outcome"] == "SCORED" for f in frames.values()) and len(frames) == 2 * len(C.ARMS),
            quality_exact=all(f["exact"] for f in frames.values()),
            provenance_correct=all(f["provenance_correct"] for f in frames.values()),
            offanchor_is_template=all(f["capture_action_id"] is None and f["action"]["anchor_action_id"] is None
                                      for f in frames.values() if f["arm"] == "arm2_m11_q8000_offanchor"),
            archive_rescore_exact=all(f["archive_rescore_exact"] for f in frames.values()),
            history_chain=all(a["history_after_sha256"] == b["history_before_sha256"] for a, b in zip(commits, commits[1:])),
            edge_emission_exactly_once=all(v == 1 for v in emitted.values()) and set(emitted) == set(rig.feedback),
            scorer_released_everything=score_snapshot.get("waiting_predictions") == 0 and score_snapshot.get("retained_slots") == 0,
            ue_errors=not ue_result["errors"], service_errors=not ue_result["service"]["errors"],
            edge_failure=edge_report["failure"] is None, service_exit_zero=ue_result["service_returncode"] == 0)
        status = "PASS" if all(checks.values()) else "FAIL"
        result = dict(schema="splitfusion.fixed_action_screen_v3.identity_rehearsal.v1", status=status, checks=checks,
                      frames=frames, arms_exercised=sorted({f["arm"] for f in frames.values()}), arms=resolved,
                      sources=sources, all_children_exit_zero=checks["service_exit_zero"], live_services_started=False,
                      performance_claim=False, quality_values_are_mechanics_only=True)
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
    parser.add_argument("--name", required=True)
    run(parser.parse_args().name)
