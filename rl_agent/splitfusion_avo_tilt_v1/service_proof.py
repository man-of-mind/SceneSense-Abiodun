"""Real-service proof of the tilt-aware AVO candidate (offline, CPU only; no CARLA server, OAI, GPU or radio).

Every arm runs UE seam -> GT service trampoline -> real ``exact_gt_live_v2`` service -> GT worker trampolines ->
real prep and score workers (prep FREEZE -> BIND -> CONFIRM_SEND -> PUBLISH; score TARGET/PREDICTION join), with
the registered M3 limits and the V7 FREEZE-lifetime repair.  Expected scores come from the in-process runtime
producer (``AnchorReferenceDriver``) with the SAME geometry as the arm; archived scorer directories are re-scored.

  upright_v7          V7 path, retained corpus (all upright), v3 rehearsal scenario (11 scored frames, cancel, edge
  upright_candidate   candidate path, same            terminal, horizon timeout) -> scores/measurements identical
  tilt_v7             V7 path, synthetic tilts on frames 1531/1535/1539 -> the old rejection, reported verbatim
  tilt_candidate      candidate path, same fixtures   -> every frame scored, exact, archived with the version
  envelope_candidate  candidate path, 20-deg target   -> fails closed with the original error + FREEZE snapshot

  cd <checkout>; env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
    PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B -m rl_agent.splitfusion_avo_tilt_v1.service_proof --name <dir>
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as CC
from rl_agent.splitfusion_exact_gt_service_proto_v2 import parity_common as P
from rl_agent.splitfusion_exact_gt_live_v2 import contract as EC
from rl_agent.splitfusion_exact_gt_live_v2.edge import install_early_gt_edge, EarlyGtEvaluatorV1
from rl_agent.splitfusion_numeric_gt_producer_v1.replay_v11 import quality_bridge
from rl_agent.splitfusion_m3_route_b_v7 import contract as V7C, gc_seams as V7_SEAM
from rl_agent.splitfusion_m3_route_b_v7.freeze_service_proof import Rig, _normal, digest, _json

from . import contract as C, gc_seams as CAND_SEAM, geometry as G, switch
from .fixtures import LIVE_V7_TILT, TiltCorpus

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "experiments/splitfusion_avo_tilt_v1"
BUDGET = C.FRAME_CEILING
ARM, PRODUCER = "candidate", "v2"
PATHS = dict(v7=(V7C, V7_SEAM), candidate=(C, CAND_SEAM))
TILTS = {20: ("101", *LIVE_V7_TILT, "LIVE_V7_FRAME_3502_TILT_ON_ACTOR_101"),
         21: ("101", 6.0, 2.0, "IN_ENVELOPE_6_DEG_PITCH_2_DEG_ROLL"),
         22: ("101", -0.03154, 0.01056, "SIGN_FLIPPED_LIVE_PITCH"),
         23: ("101", 20.0, 0.0, "OUT_OF_ENVELOPE_20_DEG_TARGET")}
TILT_SCENARIO = (19, 20, 21, 22)          # 1527 upright, 1531 live tilt, 1535 6 deg, 1539 sign-flipped
ENVELOPE_SCENARIO = (19, 23)              # 1527 upright, 1543 out of envelope
KEYS = ("outcome", "q_service", "q_emitted", "feedback", "committed")


def _scored_sequence(rig, indices, state):
    f = [c.frame_id for c in rig.cases]
    for i in indices:
        try:
            rig.admit(i, send=True); rig.predict(f[i]); rig.await_feedback(f[i], timeout=10); rig.terminal(f[i])
        except BaseException as error:           # a fail-closed service: the frame that killed it is recorded
            state["stopped_at"] = f[i]
            state["ue_error"] = f"{type(error).__name__}: {error}"[:600]
            break
        time.sleep(0.12)
    deadline = time.monotonic() + (30 if state.get("stopped_at") else 0)
    while state.get("stopped_at") and rig.h.process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)
    return dict(indices=list(indices), frames=[f[i] for i in indices])


def arm_run(out, name, corpus, limits, spec, *, path, scenario, expect_failure):
    contract, seam = PATHS[path]
    out.mkdir()
    registration = dict(id=contract.ID, transmitted_budget=BUDGET, gt_gc=contract.gc_registration(ARM, BUDGET, PRODUCER))
    state, rig = {}, None
    with seam.install_ue_service_seam(registration=registration, output=out) as trace_dir:
        try:
            rig = Rig(out, corpus, limits, contract.exact_sources(ROOT), BUDGET)
            with install_early_gt_edge(socket_path=rig.evidence / EC.SOCKET_NAME,
                                       capability_path=rig.evidence / EC.CAPABILITY_NAME, connect_timeout_s=30):
                rig.ev = EarlyGtEvaluatorV1(spec=spec, read_ground_truth=None, send=rig.payloads.put,
                                            match_distance_m=2., gt_timeout_s=2.)
                rig.ev.start()
                run = _normal(rig, limits) if scenario == "normal" else _scored_sequence(rig, scenario, state)
                try:
                    rig.ev.close()
                except BaseException as error:
                    if not expect_failure:
                        raise
                    state["edge_close_error"] = f"{type(error).__name__}: {error}"[:300]
            try:
                ue_result = rig.h.close()
            except BaseException as error:
                if not expect_failure:
                    raise
                state["close_error"], ue_result = f"{type(error).__name__}: {error}"[:600], None
        except BaseException as error:
            if rig is not None and rig.h.process is not None and rig.h.process.poll() is None:
                rig.h.abort(error)
            raise
    # ---- after teardown
    service_dir = rig.ue_dir / "numeric_gt/service"
    rows = [json.loads(line) for line in open(service_dir / "FRAMES.jsonl")] if (service_dir / "FRAMES.jsonl").exists() else []
    service = _json(service_dir / "SERVICE_RESULT.json") or {}
    from rl_agent.splitfusion_fixed_action_screen_analysis_v2.analyze import rescore
    index = {c.frame_id: i for i, c in enumerate(rig.cases)}
    frames = {}
    if path == "candidate":
        switch.install()                          # the reference must use the arm's geometry
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
                    expected = P.score_exact(corpus, case, ref.run(case)["packet"])["q_perc"]
                    metadata = json.loads((scorer_dir / "target_metadata.json").read_bytes())
                    score = json.loads((scorer_dir / "numerical_score.json").read_bytes())
                    entry.update(q_reference=expected, exact=entry["q_service"] == expected == fb.get("q_perc"),
                                 archive_rescore_exact=rescore(scorer_dir), measurement=digest(score["measurement"]),
                                 avo=metadata["avo"])
                frames[str(fid)] = entry
        finally:
            ref.close()
    finally:
        switch.uninstall()
    docs = {n: _json(trace_dir / n) for n in ("prep_gc_trace.json", "score_gc_trace.json", "service_substitution.json",
                                              C.PREP_BINDING_RECORD, C.AVO_BINDING_RECORD)}
    prep_log = (service_dir / "workers/prep.log")
    prep_text = prep_log.read_text(errors="replace") if prep_log.exists() else ""
    snapshot_dir = trace_dir / C.FAILURE_SNAPSHOT_DIR
    snapshot = None
    if (snapshot_dir / "READY.json").exists():
        ready = json.loads((snapshot_dir / "READY.json").read_bytes())
        error = json.loads((snapshot_dir / "ERROR.json").read_bytes())
        snapshot = dict(hashes_verify=all(hashlib.sha256((snapshot_dir / n).read_bytes()).hexdigest() == d
                                          for n, d in ready["sha256"].items()),
                        exception_type=error["exception_type"], exception=error["exception"],
                        frame_id=json.loads(json.loads((snapshot_dir / "REQUEST.json").read_bytes())["descriptor"])["frame_id"])
    sub = docs["service_substitution.json"] or {}
    return dict(arm=name, path=path, frames=frames, scenario=run, state=state, errors=list(service.get("errors") or []),
                service_returncode=rig.h.process.returncode, ue_result_ok=ue_result is not None,
                worker_entries=sorted({a[a.index("-m") + 1] for a in (s.get("argv", []) for s in sub.get("substitutions", []))
                                       if "-m" in a}),
                prep_lifetime=docs[C.PREP_BINDING_RECORD], avo_binding=docs[C.AVO_BINDING_RECORD],
                prep_log_tail=prep_text[-3000:], snapshot=snapshot, outcomes=service.get("outcomes"))


def _strip(avo):
    value = json.loads(json.dumps(avo))
    value.pop("avo_geometry", None)
    for row in value.get("qualified", []):
        row.pop("algorithm_version", None)
    return value


def scored(arm):
    return {k: v for k, v in arm["frames"].items() if v["outcome"] == "SCORED"}


def exact(arm):
    return bool(scored(arm)) and all(v["exact"] and v["archive_rescore_exact"] for v in scored(arm).values())


def run(name):
    P.environment_ok()
    out = EVIDENCE / name
    out.mkdir(parents=True, exist_ok=False)
    limits = C.limits(BUDGET)
    spec, _ = quality_bridge()
    base = P.Corpus(49)
    tilted = TiltCorpus(TILTS)
    f = {i: tilted.cases[i].frame_id for i in (19, 20, 21, 22, 23)}
    arms, result = {}, None
    try:
        arms["upright_v7"] = arm_run(out / "upright_v7", "upright_v7", base, limits, spec, path="v7", scenario="normal",
                                     expect_failure=False)
        arms["upright_candidate"] = arm_run(out / "upright_candidate", "upright_candidate", base, limits, spec,
                                            path="candidate", scenario="normal", expect_failure=False)
        arms["tilt_v7"] = arm_run(out / "tilt_v7", "tilt_v7", tilted, limits, spec, path="v7", scenario=TILT_SCENARIO,
                                  expect_failure=True)
        arms["tilt_candidate"] = arm_run(out / "tilt_candidate", "tilt_candidate", tilted, limits, spec, path="candidate",
                                         scenario=TILT_SCENARIO, expect_failure=False)
        arms["envelope_candidate"] = arm_run(out / "envelope_candidate", "envelope_candidate", tilted, limits, spec,
                                             path="candidate", scenario=ENVELOPE_SCENARIO, expect_failure=True)
        uv, uc, tv, tc, ec = (arms[k] for k in ("upright_v7", "upright_candidate", "tilt_v7", "tilt_candidate",
                                                "envelope_candidate"))
        live_tilted = tc["frames"].get(str(f[20]), {}).get("avo", {})
        checks = dict(
            upright_v7_exact=exact(uv) and len(scored(uv)) == 11,
            upright_candidate_exact=exact(uc) and len(scored(uc)) == 11,
            upright_outcomes_and_q_identical=uv["frames"].keys() == uc["frames"].keys() and all(
                {k: uv["frames"][x][k] for k in KEYS} == {k: uc["frames"][x][k] for k in KEYS} for x in uv["frames"]),
            upright_measurements_identical=all(uv["frames"][x]["measurement"] == uc["frames"][x]["measurement"]
                                               for x in scored(uv)),
            upright_avo_identical_except_version_labels=all(_strip(uv["frames"][x]["avo"]) == _strip(uc["frames"][x]["avo"])
                                                            for x in scored(uv)),
            candidate_archives_carry_geometry_version=all(
                v["avo"]["avo_geometry"]["version"] == G.GEOMETRY_VERSION
                and all(r["algorithm_version"] == G.GEOMETRY_VERSION for r in v["avo"]["qualified"])
                for arm in (uc, tc) for v in scored(arm).values()),
            candidate_paths_started_candidate_worker=uc["worker_entries"] == tc["worker_entries"] == [C.WORKER_ENTRY],
            candidate_prep_switch_installed_and_restored=all(
                (a["avo_binding"] or {}).get("installed") is True and (a["avo_binding"] or {}).get("restored") is True
                and (a["prep_lifetime"] or {}).get("installed") is True for a in (uc, tc)),
            # A fatally failed prep worker is SIGTERMed by the service during its finally, so its exit records can be
            # lost (known V7 limitation); the envelope arm is proven by the candidate-only error text and its snapshot.
            envelope_arm_ran_candidate_geometry=bool(ec["errors"]) and "tilt-aware AVO v1" in ec["errors"][0]
                and ec["worker_entries"] == [C.WORKER_ENTRY],
            v7_paths_unmodified=uv["avo_binding"] is None and tv["avo_binding"] is None
                and uv["worker_entries"] == [V7C.WORKER_ENTRY],
            tilt_v7_old_rejection_reported_verbatim=bool(tv["errors"]) and tv["errors"][0].startswith(
                "WorkerFailed: prep FREEZE: LiveGtContractError: yaw-only AVO does not support tilted actor 101: "
                "TILTED_SCOREABLE_TARGET") and "BufferError" not in " ".join(tv["errors"]),
            tilt_v7_stopped_at_first_tilted_frame=tv["state"].get("stopped_at") == f[20]
                and sorted(scored(tv)) == [str(f[19])] and exact(tv),
            tilt_v7_snapshot_preserved=bool(tv["snapshot"]) and tv["snapshot"]["hashes_verify"]
                and tv["snapshot"]["frame_id"] == f[20],
            tilt_candidate_all_scored_exact=sorted(scored(tc)) == sorted(str(f[i]) for i in TILT_SCENARIO) and exact(tc)
                and tc["service_returncode"] == 0 and not tc["errors"] and tc["ue_result_ok"],
            tilt_candidate_live_tilt_handled=any(t["gt_actor_id"] == "101" and t["handling"] == "TILT_AWARE_GEOMETRY"
                                                 for t in live_tilted.get("avo_geometry", {}).get("tilted_people", [])),
            untilted_frame_identical_across_paths=(
                tc["frames"][str(f[19])]["q_service"] == tv["frames"][str(f[19])]["q_service"]),
            envelope_fails_closed_with_original_error=bool(ec["errors"]) and ec["errors"][0].startswith(
                "WorkerFailed: prep FREEZE: LiveGtContractError: tilt-aware AVO v1 does not support actor 101: "
                "TILT_OUTSIDE_GROUND_RULE_ENVELOPE_TARGET") and "BufferError" not in " ".join(ec["errors"])
                and ec["service_returncode"] not in (0, None),
            envelope_snapshot_preserved=bool(ec["snapshot"]) and ec["snapshot"]["hashes_verify"]
                and ec["snapshot"]["frame_id"] == f[23] and ec["snapshot"]["exception_type"] == "LiveGtContractError",
            envelope_failed_frame_not_scored=str(f[23]) not in scored(ec) and sorted(scored(ec)) == [str(f[19])])
        status = "PASS" if all(checks.values()) else "FAIL"
        result = dict(schema="splitfusion.avo_tilt_v1.service_proof.v1", status=status, checks=checks, arms=arms,
                      fixtures=tilted.fixture_evidence, geometry_version=G.GEOMETRY_VERSION, limits=limits,
                      sources=C.exact_sources(ROOT), live_services_started=False, carla_server_started=False,
                      scope="service path of the candidate; synthetic tilts on retained frames (depth not re-rendered)")
        print(json.dumps(dict(status=status, failed=[k for k, v in checks.items() if not v],
                              tilt_v7_errors=tv["errors"][:1], envelope_errors=ec["errors"][:1],
                              tilt_candidate=[(k, v["outcome"], v.get("q_service")) for k, v in tc["frames"].items()]),
                         indent=1, default=str))
    except BaseException as error:
        result = dict(status="FAIL", error=f"{type(error).__name__}: {error}", live_services_started=False,
                      arms_completed=sorted(arms))
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
