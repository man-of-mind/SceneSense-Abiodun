"""In-process engine qualification of the containment (offline replay, recorded clocks, CPU only).

Same design and comparison fields as ``splitfusion_exact_gt_service_proto_v2.verify_deferred`` Run B: a candidate
engine sequence with non-accepted frames versus the retained runtime reference that never sees those frames.
The tilt-aware AVO switch AND (correction 01) the V16 storage-accounting V3.v2 session are installed for candidate
AND reference, as in the M3 prep worker; the session's snapshot is recorded.

  U  upright parity: adopted ContainedEngine, base corpus, all 49 accepted   == reference, all fields exact
  C  containment: fixture corpus, refused TARGET / POSSIBLE_COMPETITOR frames (resolved by BIND refusal or by
     hold), interleaved with accepted frames                                   == reference without those frames
  S  stress: every one of the 49 frames refused (TARGET tilt on its own target); history never advances; owner
     capacity/reservation/pending return to zero after each; bounded memory/record growth
  F  fail closed: unrelated corruption (non-zero bbox rotation) and an inside-envelope tilt are NOT contained
  K  close with a refused frame pending: abandon=True releases; abandon=False fails exactly like an unbound close

    cd <checkout>; env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
      PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B -m rl_agent.splitfusion_gt_geometry_containment_v1.verify_engine \\
      --name <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tracemalloc

from rl_agent.splitfusion_avo_tilt_v1 import switch
from rl_agent.splitfusion_exact_gt_service_proto_v2 import deferred as D, parity_common as P
from rl_agent.splitfusion_exact_gt_service_proto_v2.verify_deferred import comparable
from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_numeric_gt_runtime_v1 import producer as PR
from rl_agent.splitfusion_run5_live_avo_v1.quality_adapter import LiveGtContractError

from . import classify as CL, engine as E
from .fixtures import FixtureCorpus, S1_TILT

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "experiments/splitfusion_gt_geometry_containment_v1"

# Pedestrian targets per frame of the retained corpus (fixtures.targets): none in cases 0-5 (146 is beyond 40 m),
# 146 in 6-19, 101 in 19-31, 122 in 31-44; the stress run reads them per frame.


PLAN_C = {
    7: ("TILT", "146", *S1_TILT, "S1_FRAME_6146_ACTOR_106_TILT_ON_TARGET"),
    10: ("TILT", "146", 20.0, 0.0, "OUT_OF_ENVELOPE_20_DEG_TARGET"),
    11: ("TILT", "146", 0.0, 12.0, "OUT_OF_ENVELOPE_12_DEG_ROLL_TARGET_HELD"),
    19: ("RELOCATE_TILT", "101", 20.0, 0.0, "POSSIBLE_COMPETITOR_RELOCATED_BEHIND_TARGET_101"),
    24: ("TILT", "101", -15.0, 5.0, "OUT_OF_ENVELOPE_NEG_PITCH_TARGET"),
    25: ("TILT", "101", *S1_TILT, "S1_TILT_ON_TARGET_SECOND_TIME"),
    40: ("TILT", "122", 0.0, -30.0, "OUT_OF_ENVELOPE_30_DEG_ROLL_TARGET"),
}
ACTION_C = {11: "hold"}                     # refused frames not listed are resolved by BIND refusal
EXPECTED_REASON = {i: (CL.REASONS[CL.STATUS_COMPETITOR] if spec[0] == "RELOCATE_TILT" else CL.REASONS[CL.STATUS_TARGET])
                   for i, spec in PLAN_C.items()}


def owner_state(engine):
    o = engine._owner
    return dict(history=o._producer.history_sha256(), owner_frames=len(o._frames), reserved=o._reserved,
                pending=o._pending is not None, owner_failure=o._failure, engine_failure=engine._failure,
                unbound=engine._unbound is not None, issued=engine._issued)


def contained_driver(corpus):
    driver = P.DeferredDriver(corpus)
    E.adopt(driver.engine)
    return driver


def run_refused(driver, case, action):
    """prepare_unbound (contained) then BIND refusal or hold; returns evidence."""
    engine = driver.engine
    before = owner_state(engine)
    handle = driver.unbound(case)
    record = engine.contained_refusal(handle)
    mid = owner_state(engine)
    out = dict(contained=record is not None, record=record, state_before=before, state_after_freeze=mid)
    if record is None:                 # not a recognised refusal: release it like a hold, report, never bind it
        engine.cancel_unbound(handle, "HOLD_NEVER_BOUND")
        out["resolution"] = "NOT_CONTAINED_PREPARED_THEN_HELD"
    elif action == "hold":
        engine.cancel_unbound(handle, "HOLD_NEVER_BOUND")
        out["resolution"] = "hold"
    else:
        try:
            engine.bind(handle, driver.corpus.plan(case), frame_context=driver.corpus.context(case))
            out["resolution"] = "BOUND_UNEXPECTEDLY"
        except D.DeferralRefused as refused:
            out.update(resolution="bind_refused", reason=refused.reason, message=str(refused))
    out["state_after"] = owner_state(engine)
    out["history_restored"] = out["state_after"]["history"] == before["history"]
    out["capacity_restored"] = all(out["state_after"][k] == v for k, v in
                                   dict(owner_frames=0, reserved=0, pending=False, owner_failure=None,
                                        engine_failure=None, unbound=False).items())
    return out


def run_u(base):
    ref = P.ReferenceDriver(base)
    ref_rows = {c.frame_id: comparable(ref.run(c)) for c in base.cases}
    ref.close()
    cand = contained_driver(base)
    rows = []
    for case in base.cases:
        got = comparable(cand.run(case))
        rows.append(dict(frame_id=case.frame_id, exact={k: ref_rows[case.frame_id][k] == got[k] for k in got}))
    snap = cand.engine.snapshot()["deferral"]
    cand.close()
    return dict(frames=len(rows), all_exact=all(all(r["exact"].values()) for r in rows), containment=snap,
                failures=[r for r in rows if not all(r["exact"].values())][:3])


def run_c(fixture):
    accepted = [c for i, c in enumerate(fixture.cases) if i not in PLAN_C]
    ref = P.ReferenceDriver(fixture)
    ref_rows = {c.frame_id: comparable(ref.run(c)) for c in accepted}
    ref.close()
    cand = contained_driver(fixture)
    rows = []
    for index, case in enumerate(fixture.cases):
        if index in PLAN_C:
            r = run_refused(cand, case, ACTION_C.get(index, "bind"))
            ok = (r["contained"] and r["history_restored"] and r["capacity_restored"]
                  and r["record"]["reason"] == EXPECTED_REASON[index]
                  and (r["resolution"] == "hold" or (r["resolution"] == "bind_refused"
                                                     and r["reason"] == EXPECTED_REASON[index])))
            rows.append(dict(index=index, frame_id=case.frame_id, label=PLAN_C[index][-1], ok=ok, **r))
        else:
            got = comparable(cand.run(case))
            exact = {k: ref_rows[case.frame_id][k] == got[k] for k in got}
            rows.append(dict(index=index, frame_id=case.frame_id, ok=all(exact.values()), exact=exact))
    snap = cand.engine.snapshot()
    cand.close()
    return dict(rows=rows, accepted=len(accepted), refused=len(PLAN_C),
                accepted_exact=all(r["ok"] for r in rows if r["index"] not in PLAN_C),
                refused_ok=all(r["ok"] for r in rows if r["index"] in PLAN_C),
                engine_valid=snap["valid"], containment=snap["deferral"]["geometry_containment"],
                deferral={k: v for k, v in snap["deferral"].items() if k != "geometry_containment"})


def run_s():
    from .fixtures import targets
    targets_by_case = targets(P.Corpus(49))
    plan = {i: ("TILT", t[0], 20.0, 0.0, "STRESS_20_DEG_PITCH_ON_TARGET") for i, t in enumerate(targets_by_case) if t}
    corpus = FixtureCorpus(plan)
    cand = contained_driver(corpus)
    initial = owner_state(cand.engine)
    tracemalloc.start()
    marks, rows = {}, []
    for index, case in enumerate(corpus.cases):
        if index not in plan:                       # no pedestrian target in this frame: accepted normally
            cand.run(case)
            initial = owner_state(cand.engine)
            continue
        try:
            r = run_refused(cand, case, "hold" if index % 5 == 0 else "bind")
            rows.append(dict(index=index, frame_id=case.frame_id, contained=r["contained"],
                             reason=r["record"]["reason"] if r["record"] else None,
                             history_equals_initial=r["state_after"]["history"] == initial["history"],
                             capacity_restored=r["capacity_restored"]))
        except LiveGtContractError as error:       # not a recognised refusal for this frame: record and stop
            rows.append(dict(index=index, frame_id=case.frame_id, contained=False, error=str(error)[:300]))
            break
        if index in (15, 48):
            marks[index] = tracemalloc.take_snapshot()
    growth = None
    if 15 in marks and 48 in marks:
        stats = marks[48].compare_to(marks[15], "filename")
        growth = sum(s.size_diff for s in stats)
    tracemalloc.stop()
    snap = cand.engine.snapshot()
    cand.close()
    return dict(frames=len(rows), target_frames=len(plan), accepted_without_target=49 - len(plan),
                contained=sum(r["contained"] for r in rows), rows=rows,
                history_never_advanced=all(r.get("history_equals_initial") for r in rows),
                capacity_restored_every_frame=all(r.get("capacity_restored") for r in rows),
                python_heap_growth_bytes_after_refusal_10_to_43=growth, engine_valid=snap["valid"],
                containment=snap["deferral"]["geometry_containment"],
                owner=dict(retained_jobs=snap["owner"]["retained_jobs"], reserved_bytes=snap["owner"]["reserved_bytes"]))


def run_f():
    out = {}
    for label, plan in (("bbox_rotation", {20: ("BBOX_ROTATION", "101", 5.0, "NONZERO_BBOX_ROTATION")}),
                        ("inside_envelope_tilt", {20: ("TILT", "101", 6.0, 2.0, "IN_ENVELOPE_6_2_DEG")})):
        corpus = FixtureCorpus(plan)
        cand = contained_driver(corpus)
        outcome = dict(fixture=corpus.fixture_evidence)
        try:
            for index in (19, 20, 21):
                handle = cand.unbound(corpus.cases[index])
                outcome[str(index)] = "contained" if cand.engine.contained_refusal(handle) else "prepared"
                cand.engine.cancel_unbound(handle, "HOLD_NEVER_BOUND")
        except BaseException as error:
            outcome.update(error_type=type(error).__name__, error=str(error)[:400],
                           engine_failure=cand.engine._failure, engine_valid=cand.engine.snapshot()["valid"],
                           declined=cand.engine.snapshot()["deferral"]["geometry_containment"]["declined"])
        try:
            cand.engine.close(abandon=True)
        except BaseException:
            pass
        out[label] = outcome
    return out


def run_k(fixture):
    out = {}
    for abandon in (True, False):
        cand = contained_driver(fixture)
        for case in fixture.cases[:7]:
            cand.run(case)
        handle = cand.unbound(fixture.cases[7])
        pending = cand.engine.contained_refusal(handle) is not None
        try:
            cand.engine.close(abandon=abandon)
            out[str(abandon)] = dict(pending=pending, closed=True, failure=cand.engine._failure)
        except BaseException as error:
            out[str(abandon)] = dict(pending=pending, closed=cand.engine._closed, failure=cand.engine._failure,
                                     error=f"{type(error).__name__}: {error}"[:300])
    return out


def run(name):
    P.environment_ok()
    out = EVIDENCE / name
    out.mkdir(parents=True, exist_ok=False)
    from rl_agent.splitfusion_gt_storage_accounting_v3 import switch as ACCT
    from rl_agent.splitfusion_m3_route_b_v16 import contract as V16C
    switch.install()
    session = ACCT.install(max_objects=V16C.ACCOUNTING_MAX_OBJECTS)
    try:
        base = P.Corpus(49)
        fixture = FixtureCorpus(PLAN_C)
        result = dict(schema=CL.VERSION + ".verify_engine", mode="offline replay, recorded clocks, in-process engines",
                      fixtures=fixture.fixture_evidence, plan={str(k): list(v) for k, v in PLAN_C.items()},
                      U=run_u(base), C=run_c(fixture), S=run_s(), F=run_f(), K=run_k(fixture))
        result["accounting_session"] = session.snapshot()
    finally:
        ACCT.uninstall()
        switch.uninstall()
    u, c, s, f, k = (result[x] for x in "UCSFK")
    checks = dict(
        U_upright_all_exact=u["all_exact"] and u["frames"] == 49 and u["containment"]["geometry_containment"]["contained"] == 0,
        C_accepted_exact_vs_reference_without_refused=c["accepted_exact"] and c["accepted"] == 42,
        C_refused_contained_restored_and_reasoned=c["refused_ok"] and c["containment"]["contained"] == 7
            and c["containment"]["bind_refused"] == 6 and c["containment"]["cancelled_unbound"] == 1
            and c["containment"]["pending"] == 0 and c["engine_valid"],
        C_target_and_competitor_both_exercised=c["containment"]["by_reason"] == {
            CL.REASONS[CL.STATUS_TARGET]: 6, CL.REASONS[CL.STATUS_COMPETITOR]: 1},
        S_every_target_frame_contained=s["frames"] == s["target_frames"] == s["contained"] >= 35
            and s["engine_valid"],
        S_history_never_advanced_and_capacity_restored=s["history_never_advanced"] and s["capacity_restored_every_frame"]
            and s["owner"] == dict(retained_jobs=0, reserved_bytes=0),
        S_memory_records_bounded=s["containment"]["records_kept"] <= E.MAX_MEMORY_RECORDS
            and s["containment"]["records_kept"] + s["containment"]["records_dropped"] == s["contained"],
        F_bbox_rotation_not_contained=f["bbox_rotation"].get("error_type") is not None
            and f["bbox_rotation"].get("engine_valid") is False and "contained" not in f["bbox_rotation"].values(),
        F_inside_envelope_scored_not_refused=f["inside_envelope_tilt"].get("20") == "prepared"
            and "error_type" not in f["inside_envelope_tilt"],
        K_close_abandon_releases=k["True"].get("closed") is True and k["True"]["pending"],
        K_close_without_abandon_fails_like_unbound=k["False"]["pending"]
            and k["False"]["failure"] == "closed with an unbound transaction",
        accounting_v3_session_active=result["accounting_session"]["closed"] is False
            and result["accounting_session"]["fast_path"] is True and result["accounting_session"]["calls"] > 0
            and result["accounting_session"]["failed_calls"] == 0)
    result.update(checks=checks, status="PASS" if all(checks.values()) else "FAIL", live_services_started=False)
    (out / "RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str) + "\n")
    (out / "OUTPUT_SHA256.json").write_text(json.dumps({"RESULT.json": P.sha((out / "RESULT.json").read_bytes())},
                                                       indent=1) + "\n")
    print(json.dumps(dict(status=result["status"], failed=[k for k, v in checks.items() if not v],
                          C=dict(containment=c["containment"], deferral=c["deferral"]),
                          S={k: v for k, v in s.items() if k != "rows"}, F=f, K=k), indent=1, default=str))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    run(parser.parse_args().name)
