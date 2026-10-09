"""Short real-service proof of the FINAL v11 deployment path (offline, CPU only; no CARLA server, OAI, GPU or radio).

v11 changes the GT prep worker only: the CORRECTED exact storage accounting V2 (one bounded AccountingSession) with a
bounded per-call timer, installed by the v11 worker trampoline before the unchanged v8 trampoline.  This proof is the v8 ``tilt_service_proof`` run through
the v11 chain, with the v8 checks kept and the accounting/timer checks added:

One arm, the production chain registered for v11: v11 UE seam -> v11 GT service trampoline -> real ``exact_gt_live_v2``
service -> v11 worker trampoline (prep: storage-accounting switch + timer) -> v8 worker trampoline (prep: tilt-aware
AVO switch) -> v7 worker trampoline (prep: FREEZE lifetime repair) -> v4 trampoline -> real prep and score workers;
registered M3 limits, GC arm ``candidate``, producer v2.

  frame 1527 (corpus index 19)  upright (retained frame)
  frame 1531 (corpus index 20)  SYNTHETIC live V7 tilt (pitch -0.0315 deg, roll -0.0106 deg) on scoreable actor 101,
                                the supported tilted case (``splitfusion_avo_tilt_v1.fixtures``; depth not re-rendered)
  frame 1535 (corpus index 21)  upright follow-on frame

Registered expectations (v11 additions): the accounting switch and the timer are installed in the prep worker only
(same pid as the AVO switch and FREEZE repair) and restored; the accounting sources are the qualified digests; the
timer recorded every call (no drop, no error, no foreign caller), each call nested inside a prep FREEZE/BIND trace op.
v8 expectations, unchanged: the service and worker trampolines are the ones started; the switch is installed in the
prep worker only and restored; the v7 repair is still installed and restored; every frame is scored, exact against the
in-process reference producer with the same geometry and sequence, and its archive re-scores exactly; every archive
carries the tilt-aware geometry version and actor 101 is handled ``TILT_AWARE_GEOMETRY``; the upright frames keep the
score and measurement of the unchanged yaw-only (v7) reference producer run on the untilted corpus over the same
frame sequence, and the recorded score; the service closes with rc 0, no errors, no failure snapshot and no surviving
worker process.  Reported, not gated: the measurement against the historical v11 ``evaluate_prepared`` document,
which differs on frames 1527/1531 under BOTH geometries (pre-existing; superseded check of tilt_service_proof_01).  The broader proofs are reused, not repeated: upright parity (49 frames), the saved
frame-3502 failure replay, old-geometry refusal and the out-of-envelope fail-closed arm
(``experiments/splitfusion_avo_tilt_v1``), and the FREEZE lifetime proof (``experiments/splitfusion_m3_route_b_v7``).

  cd <checkout>; env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
    PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B -m rl_agent.splitfusion_m3_route_b_v11.service_proof --name <dir>
"""
import argparse
import json
import os
from pathlib import Path

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as CC
from rl_agent.splitfusion_exact_gt_service_proto_v2 import parity_common as P
from rl_agent.splitfusion_exact_gt_live_v2 import contract as EC
from rl_agent.splitfusion_exact_gt_live_v2.edge import install_early_gt_edge, EarlyGtEvaluatorV1
from rl_agent.splitfusion_numeric_gt_producer_v1.replay_v11 import quality_bridge
from rl_agent.splitfusion_m3_route_b_v7.freeze_service_proof import Rig, digest, _json
from rl_agent.splitfusion_avo_tilt_v1 import geometry as G, switch
from rl_agent.splitfusion_avo_tilt_v1.fixtures import LIVE_V7_TILT, TiltCorpus
from rl_agent.splitfusion_avo_tilt_v1.service_proof import _scored_sequence

from . import contract as C
from . import gc_seams as SEAM
from .host import SERVICE_PROOF_SCHEMA

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "experiments/splitfusion_m3_route_b_v11"
BUDGET = C.FRAME_CEILING
ARM, PRODUCER = "candidate", "v2"                  # the registered GC arm / numeric producer
TILTS = {20: ("101", *LIVE_V7_TILT, "LIVE_V7_FRAME_3502_TILT_ON_ACTOR_101")}
UPRIGHT, TILTED, FOLLOW_ON = 19, 20, 21
SCENARIO = (UPRIGHT, TILTED, FOLLOW_ON)
WORKER_NEEDLES = (C.WORKER_ENTRY, C.V8_WORKER_ENTRY, C.V7_WORKER_ENTRY, C.SERVICE_ENTRY, "splitfusion_exact_gt_live_v2.service")


def surviving_processes():
    found = []
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if str(os.getpid()) != path.parent.name and any(n in command for n in WORKER_NEEDLES):
            found.append(dict(pid=int(path.parent.name), command=command[:300]))
    return found


def arm_run(out, corpus, limits, spec):
    out.mkdir()
    registration = dict(id=C.ID, transmitted_budget=BUDGET, gt_gc=C.gc_registration(ARM, BUDGET, PRODUCER))
    state, rig, edge_error = {}, None, None
    with SEAM.install_ue_service_seam(registration=registration, output=out) as trace_dir:
        try:
            rig = Rig(out, corpus, limits, C.exact_sources(ROOT), BUDGET)
            with install_early_gt_edge(socket_path=rig.evidence / EC.SOCKET_NAME,
                                       capability_path=rig.evidence / EC.CAPABILITY_NAME, connect_timeout_s=30):
                rig.ev = EarlyGtEvaluatorV1(spec=spec, read_ground_truth=None, send=rig.payloads.put,
                                            match_distance_m=2., gt_timeout_s=2.)
                rig.ev.start()
                run = _scored_sequence(rig, SCENARIO, state)
                try:
                    edge_report = rig.ev.close()
                except BaseException as error:
                    edge_error, edge_report = f"{type(error).__name__}: {error}"[:600], None
            ue_result = rig.h.close()
        except BaseException as error:
            if rig is not None and rig.h.process is not None and rig.h.process.poll() is None:
                rig.h.abort(error)
            raise
    # ---- everything below runs after service teardown (service and both workers have exited)
    after = surviving_processes()
    service_dir = rig.ue_dir / "numeric_gt/service"
    rows = [json.loads(line) for line in open(service_dir / "FRAMES.jsonl")] if (service_dir / "FRAMES.jsonl").exists() else []
    service = _json(service_dir / "SERVICE_RESULT.json") or {}
    from rl_agent.splitfusion_fixed_action_screen_analysis_v2.analyze import rescore
    index = {c.frame_id: i for i, c in enumerate(rig.cases)}
    frames = {}
    switch.install()                              # the reference uses the deployed geometry, same frame sequence
    try:
        ref = P.AnchorReferenceDriver(corpus)
        try:
            for row in rows:
                fid = row["frame_id"]
                fb = rig.feedback.get(fid) or {}
                entry = dict(outcome=row.get("outcome"), q_service=(row.get("result") or {}).get("q_perc"),
                             q_emitted=fb.get("q_perc"), feedback=fb.get("kind"), committed=bool(row.get("committed")),
                             commit=digest(row.get("commit")))
                scorer_dir = service_dir / "archive" / f"frame_{fid}" / "scorer"
                if row.get("outcome") == "SCORED":
                    case = rig.cases[index[fid]]
                    reference = P.score_exact(corpus, case, ref.run(case)["packet"])
                    metadata = json.loads((scorer_dir / "target_metadata.json").read_bytes())
                    score = json.loads((scorer_dir / "numerical_score.json").read_bytes())
                    entry.update(q_reference=reference["q_perc"],
                                 exact=entry["q_service"] == reference["q_perc"] == fb.get("q_perc"),
                                 archive_rescore_exact=rescore(scorer_dir),
                                 measurement_service_sha256=CC.sha256(CC.canonical(score["measurement"])),
                                 measurement_reference_sha256=reference["measurement_sha256"],
                                 recorded_q_exact=reference["q_perc_exact"],
                                 historical_evaluator_measurement_exact=reference["measurement_exact"],
                                 avo=metadata["avo"])
                frames[str(fid)] = entry
        finally:
            ref.close()
    finally:
        switch.uninstall()
    docs = {n: _json(trace_dir / n) for n in ("prep_gc_trace.json", "score_gc_trace.json", "service_substitution.json",
                                              "ue_substitution.json", C.PREP_BINDING_RECORD, C.AVO_BINDING_RECORD,
                                              C.ACCOUNTING_BINDING_RECORD, C.ACCOUNTING_TIMING_RECORD)}
    sub, ue_sub = docs["service_substitution.json"] or {}, docs["ue_substitution.json"] or {}
    prep_log = service_dir / "workers/prep.log"
    return dict(frames=frames, scenario=run, state=state, errors=list(service.get("errors") or []),
                outcomes=service.get("outcomes"), service_returncode=rig.h.process.returncode,
                ue_result=ue_result, edge_close_error=edge_error,
                edge_failure=None if edge_report is None else edge_report.get("failure"),
                service_substitutions=sub.get("substitutions"), service_entry_module=sub.get("entry_module"),
                service_contract_module=sub.get("contract_module"), service_record_returncode=sub.get("returncode"),
                ue_substitutions=ue_sub.get("substitutions"), ue_seam_module=ue_sub.get("seam_module"),
                ue_restored=ue_sub.get("restored"), avo_binding=docs[C.AVO_BINDING_RECORD],
                prep_lifetime=docs[C.PREP_BINDING_RECORD], prep_gc_trace_exit=(docs["prep_gc_trace.json"] or {}).get("exit"),
                accounting_binding=docs[C.ACCOUNTING_BINDING_RECORD], accounting_timing=docs[C.ACCOUNTING_TIMING_RECORD],
                prep_ops=[dict(op=o.get("op"), begin_raw_ns=o.get("begin_raw_ns"), end_raw_ns=o.get("end_raw_ns"))
                          for o in (docs["prep_gc_trace.json"] or {}).get("ops", [])],
                prep_gc_trace_written=docs["prep_gc_trace.json"] is not None,
                score_gc_trace_written=docs["score_gc_trace.json"] is not None,
                failure_snapshot_exists=(trace_dir / C.FAILURE_SNAPSHOT_DIR).exists(), surviving_processes=after,
                prep_log_tail=prep_log.read_text(errors="replace")[-3000:] if prep_log.exists() else "")


def timing_nesting(timing, ops):
    """Each recorded accounting call lies inside one prep FREEZE or BIND op (CLOCK_MONOTONIC_RAW, same process)."""
    spans = sorted((o["begin_raw_ns"], o["end_raw_ns"], o["op"]) for o in ops
                   if o.get("op") in ("FREEZE", "BIND") and o.get("begin_raw_ns") and o.get("end_raw_ns"))
    names = list(timing.get("caller_names", []))
    per_op, outside = {}, 0
    for begin, end, caller in zip(timing.get("begin_raw_ns", []), timing.get("end_raw_ns", []), timing.get("caller", [])):
        owner = [op for b, e, op in spans if b <= begin and end <= e]
        if len(owner) != 1:
            outside += 1
            continue
        key = f"{owner[0]}:{names[caller]}"
        per_op[key] = per_op.get(key, 0) + 1
    recorded = len(timing.get("cpu_ns", []))
    return dict(recorded=recorded, outside=outside, by_op_and_caller=per_op, all_nested=recorded > 0 and outside == 0)


def yaw_only_reference(corpus, indices):
    """The unchanged yaw-only (v7) reference producer on the UNTILTED corpus, same frame sequence, nothing installed."""
    if switch.installed():
        raise RuntimeError("yaw-only reference requires the original producer AVO step")
    ref = P.AnchorReferenceDriver(corpus)
    try:
        out = {}
        for i in indices:
            case = corpus.cases[i]
            value = P.score_exact(corpus, case, ref.run(case)["packet"])
            out[i] = dict(frame_id=case.frame_id, q_perc=value["q_perc"], measurement_sha256=value["measurement_sha256"],
                          recorded_q_exact=value["q_perc_exact"],
                          historical_evaluator_measurement_exact=value["measurement_exact"])
        return out
    finally:
        ref.close()


def run(name):
    P.environment_ok()
    out = EVIDENCE / name
    out.mkdir(parents=True, exist_ok=False)
    limits = C.limits(BUDGET)
    spec, _ = quality_bridge()
    corpus = TiltCorpus(TILTS)
    f = {i: corpus.cases[i].frame_id for i in SCENARIO}
    before = surviving_processes()
    result = None
    try:
        arm = arm_run(out / "v10_upright_tilt_follow_on", corpus, limits, spec)
        yaw = yaw_only_reference(P.Corpus(49), SCENARIO)
        frames = arm["frames"]
        scored = {k: v for k, v in frames.items() if v["outcome"] == "SCORED"}
        avo, life = arm["avo_binding"] or {}, arm["prep_lifetime"] or {}
        worker_argv = [s.get("argv", []) for s in arm["service_substitutions"] or []]
        tilted = frames.get(str(f[TILTED]), {}).get("avo", {}).get("avo_geometry", {})
        acc, timing = arm["accounting_binding"] or {}, arm["accounting_timing"] or {}
        session = acc.get("accounting_session") or {}
        nesting = timing_nesting(timing, arm["prep_ops"])
        checks = dict(
            no_gt_process_before=before == [],
            ue_seam_started_v10_service_trampoline=len(arm["ue_substitutions"] or []) == 1
                and C.SERVICE_ENTRY in arm["ue_substitutions"][0]["argv"] and arm["ue_restored"] is True
                and arm["ue_seam_module"] == C.GC_SEAM,
            v10_service_trampoline_ran=arm["service_entry_module"] == C.SERVICE_ENTRY
                and arm["service_contract_module"] == C.MODULE + ".contract",
            v10_worker_trampoline_both_roles=len(worker_argv) == 2 and all(C.WORKER_ENTRY in a for a in worker_argv)
                and sorted((s["role"], s["gc_policy"], s["producer"]) for s in arm["service_substitutions"])
                == sorted([("prep", ARM, PRODUCER), ("score", "control", "v1")]),
            candidate_installed_in_prep_and_restored=avo.get("installed") is True and avo.get("restored") is True
                and avo.get("exit") is None and avo.get("entry_module") == C.V8_WORKER_ENTRY
                and avo.get("v7_worker_entry") == C.V7_WORKER_ENTRY,
            accounting_installed_in_prep_and_restored=acc.get("installed") is True and acc.get("timer_installed") is True
                and acc.get("timer_removed") is True and acc.get("restored") is True and acc.get("exit") is None
                and acc.get("entry_module") == C.WORKER_ENTRY and acc.get("v8_worker_entry") == C.V8_WORKER_ENTRY
                and acc.get("pid") == avo.get("pid") == life.get("pid"),
            accounting_is_qualified_source=acc.get("version") == C.ACCOUNTING_VERSION
                and acc.get("substitution") == C.ACCOUNTING_SUBSTITUTION
                and acc.get("sources") == {n: C.ACCOUNTING_QUALIFIED[f"{C.ACCOUNTING_PACKAGE}/{n}.py"]
                                           for n in ("accounting", "switch")},
            accounting_fast_path_active_and_bounded=acc.get("fast_path_available") is True
                and all(reason is None for reason in (acc.get("capabilities") or {"missing": "capability"}).values())
                and session.get("version") == C.ACCOUNTING_VERSION
                and session.get("max_objects") == C.ACCOUNTING_MAX_OBJECTS
                and session.get("calls", 0) >= timing.get("calls", 0) > 0
                and session.get("snapshot_fast", 0) > 0 and session.get("owner_shortcut_hits", 0) > 0
                and session.get("memo_objects", 1 << 30) <= C.ACCOUNTING_MAX_OBJECTS
                and session.get("snapshot_fallbacks") == 0
                and session.get("owner_shortcut_rejections") == 0
                and session.get("witness_ineligible") == 0
                and acc.get("accounting_session_closed") is True,
            accounting_timing_complete=timing.get("calls", 0) >= 2 * len(SCENARIO)
                and timing.get("recorded") == timing.get("calls") and timing.get("dropped") == 0
                and timing.get("errors") == 0 and timing.get("foreign_frames") == 0
                and timing.get("capacity") == C.ACCOUNTING_TIMING_CAPACITY
                and all(b > 0 for b in timing.get("result_bytes", [0])),
            accounting_calls_nested_in_prep_freeze_or_bind=nesting["all_nested"],
            candidate_is_qualified_source=avo.get("geometry_version") == C.AVO_GEOMETRY_VERSION == G.GEOMETRY_VERSION
                and avo.get("sources") == {n: C.AVO_TILT_QUALIFIED[f"{C.AVO_TILT_PACKAGE}/{n}.py"]
                                           for n in ("geometry", "quality", "switch")},
            v7_freeze_repair_installed_and_restored=life.get("installed") is True and life.get("restored") is True
                and life.get("exit") is None and life.get("entry_module") == C.V7_WORKER_ENTRY
                and life.get("pid") == avo.get("pid"),
            all_three_frames_scored=sorted(scored) == sorted(str(f[i]) for i in SCENARIO),
            scores_exact_vs_same_geometry_reference=bool(scored) and all(v["exact"] for v in scored.values())
                and all(v["measurement_service_sha256"] == v["measurement_reference_sha256"] for v in scored.values()),
            archive_rescore_exact=bool(scored) and all(v["archive_rescore_exact"] for v in scored.values()),
            archives_carry_evaluator_version=bool(scored) and all(
                v["avo"]["avo_geometry"]["version"] == C.AVO_GEOMETRY_VERSION
                and all(r["algorithm_version"] == C.AVO_GEOMETRY_VERSION for r in v["avo"]["qualified"])
                for v in scored.values()),
            tilted_actor_handled_tilt_aware=any(t.get("gt_actor_id") == "101" and t.get("handling") == "TILT_AWARE_GEOMETRY"
                                                for t in tilted.get("tilted_people", [])),
            upright_frames_have_no_tilted_people=all(
                not frames[str(f[i])]["avo"]["avo_geometry"].get("tilted_people") for i in (UPRIGHT, FOLLOW_ON)),
            upright_frames_identical_to_yaw_only_reference=all(
                frames[str(f[i])]["q_service"] == yaw[i]["q_perc"]
                and frames[str(f[i])]["measurement_service_sha256"] == yaw[i]["measurement_sha256"]
                for i in (UPRIGHT, FOLLOW_ON)),
            upright_frames_keep_recorded_score=all(frames[str(f[i])]["recorded_q_exact"] for i in (UPRIGHT, FOLLOW_ON)),
            clean_shutdown=arm["service_returncode"] == 0 and not arm["errors"] and arm["ue_result"] is not None
                and arm["ue_result"]["service_returncode"] == 0 and not arm["ue_result"]["errors"]
                and arm["edge_close_error"] is None and arm["edge_failure"] is None
                and arm["service_record_returncode"] == 0 and arm["state"].get("stopped_at") is None,
            worker_exit_records_written=arm["prep_gc_trace_written"] and arm["score_gc_trace_written"],
            no_failure_snapshot=arm["failure_snapshot_exists"] is False,
            no_surviving_gt_process=arm["surviving_processes"] == [])
        status = "PASS" if all(checks.values()) else "FAIL"
        historical = {str(f[i]): dict(v11=frames[str(f[i])]["historical_evaluator_measurement_exact"],
                                      yaw_only=yaw[i]["historical_evaluator_measurement_exact"]) for i in SCENARIO}
        result = dict(schema=SERVICE_PROOF_SCHEMA, status=status, checks=checks, arm=arm, timing_nesting=nesting, fixtures=corpus.fixture_evidence,
                      yaw_only_reference=yaw, historical_evaluator_measurement_exact_reported_not_gated=historical,
                      frames=dict(upright=f[UPRIGHT], tilted=f[TILTED], follow_on=f[FOLLOW_ON]),
                      geometry_version=C.AVO_GEOMETRY_VERSION, limits=limits, sources=C.exact_sources(ROOT),
                      launcher_binding=C.launcher_binding(), live_services_started=False, carla_server_started=False,
                      oai_started=False, performance_claim=False,
                      accounting_session=session,
                      scope="final v11 GT service/worker path (v8 tilt scenario); accounting exactness proofs reused "
                            "from experiments/splitfusion_gt_storage_accounting_v3 (process_qualification_V10_03, "
                            "replay_qualification_03, output_parity_03, service_proof_04, "
                            "regression_cache_correctness_01), geometry proofs from "
                            "experiments/splitfusion_avo_tilt_v1")
        print(json.dumps(dict(status=status, failed=[k for k, v in checks.items() if not v],
                              frames=[(k, v["outcome"], v.get("q_service"), v.get("q_reference")) for k, v in frames.items()]),
                         indent=1, default=str))
    except BaseException as error:
        result = dict(schema=SERVICE_PROOF_SCHEMA, status="FAIL", error=f"{type(error).__name__}: {error}",
                      live_services_started=False)
        raise
    finally:
        (out / "RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str) + "\n")
        manifest = {str(p.relative_to(out)): CC.sha256(p.read_bytes()) for p in sorted(out.rglob("*"))
                    if p.is_file() and p.name != "OUTPUT_SHA256.json" and "archive" not in p.parts}
        (out / "OUTPUT_SHA256.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    if result["status"] != "PASS":
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    run(parser.parse_args().name)
