"""Real-service proof of the geometry-refusal containment (offline, CPU only; no CARLA server, OAI, GPU or radio).

Every arm: UE seam -> GT service trampoline -> real ``exact_gt_live_v2`` service -> GT worker trampolines -> real prep
and score workers (FREEZE through the sealed memfd/FreezeGate path) -> real early-GT edge evaluator -> R4FB feedback,
with the registered M3 limits, the V7 FREEZE-lifetime repair and the tilt-aware AVO.  Reuses the avo_tilt
``service_proof.arm_run`` machinery (V7 ``Rig``); expected scores come from the in-process runtime producer with
the same geometry, run over the SCORED cases only (so exactness also proves refused frames left no history).

Correction 01: the unchanged control is the CURRENT V16 GT stack (accounting V3.v2 + timer, tilt AVO, FREEZE
lifetime repair), and every arm proves the V3.v2 accounting session was ACTIVE (binding record + per-call timing:
exact per-caller call counts, no errors/drops/foreign frames, identical accounting results V16 vs candidate).

  upright_v16        V16 path, retained corpus, V3 normal scenario (11 scored, send-failure cancel, edge
  upright_contained  containment path, same                terminal, horizon timeout) -> identical outcomes/scores
  contained          containment path, labelled fixtures: 14 refused frames (12 TARGET incl. the S1 pitch/roll, 1
                     POSSIBLE_COMPETITOR; GT-first, prediction-first, send failure after admission, prediction after
                     the horizon,
                     back-to-back) interleaved with 20 valid frames -> refused frames get exactly one explicit
                     NO_GT_DEFERRAL_REFUSED_GT_GEOMETRY_ENVELOPE_* outcome and GROUND_TRUTH_UNAVAILABLE feedback;
                     valid frames scored exactly; worker alive; bounded diagnostics; prep fd count flat
  control_cancel     UNCHANGED V16 path, retained corpus, same sequence with each refused frame replaced by a
                     send-failure cancellation -> valid-frame q/measurement/AVO identical to ``contained``
  fail_closed_bbox   containment path: valid, contained refusal, valid, then a non-zero bbox rotation -> fatal with
                     the original error, V7 FREEZE snapshot of that frame, refusal diagnostics retained

  cd <checkout>; env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
    PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B -m rl_agent.splitfusion_gt_geometry_containment_v1.service_proof \\
    --name <dir>
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from rl_agent.splitfusion_avo_tilt_v1 import service_proof as AP, switch
from rl_agent.splitfusion_exact_gt_live_v2 import contract as EC
from rl_agent.splitfusion_exact_gt_live_v2.edge import install_early_gt_edge, EarlyGtEvaluatorV1
from rl_agent.splitfusion_exact_gt_service_proto_v2 import parity_common as P
from rl_agent.splitfusion_m3_route_b_v7.freeze_service_proof import Rig, _json, _normal, digest
from rl_agent.splitfusion_m3_route_b_v16 import contract as V16C, gc_seams as V16_SEAM
from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as CC
from rl_agent.splitfusion_numeric_gt_producer_v1.replay_v11 import quality_bridge

from . import classify as CL, contract as C, core as K, gc_seams as SEAM
from .fixtures import FixtureCorpus, S1_TILT, targets

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "experiments/splitfusion_gt_geometry_containment_v1"
BUDGET = C.FRAME_CEILING
ARM, PRODUCER = "candidate", "v2"
PATHS = dict(v16=(V16C, V16_SEAM), contained=(C, SEAM))
ACCOUNTING_CALLERS = ("compute_unbound", "stage_products", "bind_plan", "set_ready")
OUTCOME = {CL.STATUS_TARGET: "NO_GT_DEFERRAL_REFUSED_" + CL.REASONS[CL.STATUS_TARGET],
           CL.STATUS_COMPETITOR: "NO_GT_DEFERRAL_REFUSED_" + CL.REASONS[CL.STATUS_COMPETITOR]}
VALID = (6, 8, 9, 12, 13, 15, 18, 20, 21, 26, 30, 31, 33, 35, 37, 39, 41, 43, 45, 46)
REFUSED = {7: "gt_first", 10: "prediction_first", 11: "send_failure", 14: "prediction_after_horizon",
           19: "gt_first", 24: "gt_first", 25: "prediction_first", 32: "gt_first", 34: "prediction_first",
           36: "gt_first", 38: "prediction_first", 40: "gt_first", 42: "prediction_first", 44: "gt_first"}
TILTS = {7: S1_TILT, 25: S1_TILT, 10: (20.0, 0.0), 14: (-15.0, 5.0), 24: (0.0, -30.0)}


def fixture_plan(base_targets):
    plan = {}
    for index in REFUSED:
        if index == 19:
            plan[index] = ("RELOCATE_TILT", "101", 20.0, 0.0, "POSSIBLE_COMPETITOR_RELOCATED_BEHIND_TARGET_101")
            continue
        pitch, roll = TILTS.get(index, (12.0, 4.0))
        label = "S1_FRAME_6146_TILT_ON_TARGET" if (pitch, roll) == S1_TILT else f"OUT_OF_ENVELOPE_{pitch}_{roll}_TARGET"
        plan[index] = ("TILT", base_targets[index][0], pitch, roll, label)
    return plan


def prep_fds(service_pid):
    """Open descriptors of the live prep worker (child of the service whose argv has ``--role prep``)."""
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as stream:
                ppid = int(stream.read().rsplit(")", 1)[1].split()[1])
            if ppid != service_pid:
                continue
            argv = open(f"/proc/{entry}/cmdline", "rb").read().split(b"\0")
            if b"--role" in argv and argv[argv.index(b"--role") + 1] == b"prep":
                return len(os.listdir(f"/proc/{entry}/fd"))
        except (OSError, ValueError, IndexError):
            continue
    return None


def _sequence(rig, limits, state, *, replace_refused_with_cancel):
    f = [c.frame_id for c in rig.cases]
    fds = state.setdefault("prep_fds", [])
    for i in sorted(VALID + tuple(REFUSED)):
        mode = REFUSED.get(i)
        try:
            if mode is None:
                rig.admit(i)
                rig.predict(f[i]); rig.await_feedback(f[i]); rig.terminal(f[i])
            elif replace_refused_with_cancel:
                rig.admit(i, send=False); time.sleep(0.4); rig.terminal(f[i])      # unsent: no prediction (V3 pattern)
            elif mode == "send_failure":                   # send failure reported AFTER the FREEZE was admitted
                rig.admit(i, send=None); time.sleep(0.4); rig.send(f[i], ok=False); time.sleep(0.3); rig.terminal(f[i])
            elif mode == "gt_first":
                rig.admit(i); time.sleep(0.3); rig.predict(f[i]); rig.await_feedback(f[i]); rig.terminal(f[i])
            elif mode == "prediction_first":
                rig.admit(i); rig.predict(f[i]); rig.await_feedback(f[i]); rig.terminal(f[i])
            elif mode == "prediction_after_horizon":
                rig.admit(i); time.sleep(limits["horizon_ms"] / 1e3 + 0.6); rig.predict(f[i])
                rig.await_feedback(f[i]); rig.terminal(f[i])
        except BaseException as error:
            state.update(stopped_at=f[i], ue_error=f"{type(error).__name__}: {error}"[:600])
            break
        if mode is not None and not replace_refused_with_cancel:
            fds.append(dict(after_frame=f[i], fds=prep_fds(rig.h.process.pid)))
        time.sleep(0.12)
    return dict(valid=[f[i] for i in VALID], refused={str(f[i]): m for i, m in REFUSED.items()})


def _fail_closed_sequence(rig, state):
    f = [c.frame_id for c in rig.cases]
    for i in (18, 19, 20, 21):
        try:
            rig.admit(i); rig.predict(f[i]); rig.await_feedback(f[i], timeout=10); rig.terminal(f[i])
        except BaseException as error:
            state.update(stopped_at=f[i], ue_error=f"{type(error).__name__}: {error}"[:600])
            break
        time.sleep(0.12)
    deadline = time.monotonic() + 30
    while state.get("stopped_at") and rig.h.process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)
    return dict(frames=[f[i] for i in (18, 19, 20, 21)])


def arm_run(out, name, corpus, limits, spec, *, path, scenario, expect_failure):
    """``splitfusion_avo_tilt_v1.service_proof.arm_run`` with this package's path and scenarios added."""
    contract, seam = PATHS[path]
    out.mkdir()
    registration = dict(id=contract.ID, transmitted_budget=BUDGET, gt_gc=contract.gc_registration(ARM, BUDGET, PRODUCER))
    state, rig, run = {}, None, None
    with seam.install_ue_service_seam(registration=registration, output=out) as trace_dir:
        try:
            rig = Rig(out, corpus, limits, contract.exact_sources(ROOT), BUDGET)
            with install_early_gt_edge(socket_path=rig.evidence / EC.SOCKET_NAME,
                                       capability_path=rig.evidence / EC.CAPABILITY_NAME, connect_timeout_s=30):
                rig.ev = EarlyGtEvaluatorV1(spec=spec, read_ground_truth=None, send=rig.payloads.put,
                                            match_distance_m=2., gt_timeout_s=2.)
                rig.ev.start()
                if scenario == "normal":
                    run = _normal(rig, limits)
                elif scenario == "fail_closed":
                    run = _fail_closed_sequence(rig, state)
                else:
                    run = _sequence(rig, limits, state, replace_refused_with_cancel=scenario == "cancel")
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
    service_dir = rig.ue_dir / "numeric_gt/service"
    rows = [json.loads(line) for line in open(service_dir / "FRAMES.jsonl")] if (service_dir / "FRAMES.jsonl").exists() else []
    service = _json(service_dir / "SERVICE_RESULT.json") or {}
    from rl_agent.splitfusion_fixed_action_screen_analysis_v2.analyze import rescore
    index = {c.frame_id: i for i, c in enumerate(rig.cases)}
    frames = {}
    switch.install()                                    # every path here uses the tilt-aware AVO
    try:
        ref = P.AnchorReferenceDriver(corpus)
        try:
            for row in rows:
                fid = row["frame_id"]
                fb = rig.feedback.get(fid) or {}
                entry = dict(outcome=row.get("outcome"), q_service=(row.get("result") or {}).get("q_perc"),
                             q_emitted=fb.get("q_perc"), feedback=fb.get("kind"), feedback_reason=fb.get("reason"),
                             feedback_ms=None if not fb else (fb["received"] - rig.rows[fid]["opened"]) / 1e6,
                             committed=bool(row.get("committed")), commit=digest(row.get("commit")),
                             bind_refused=row.get("bind_refused"), unscored_sent=bool(row.get("edge_unscored_sent")),
                             prediction_received=bool(row.get("prediction_received_raw_ns")),
                             target_admitted=bool(row.get("target_admitted")), identity=row.get("identity") is not None,
                             prep_requests=[(r["op"], r["status"], r.get("reason")) for r in row.get("requests", [])
                                            if r["worker"] == "prep"])
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
                                              C.PREP_BINDING_RECORD, C.AVO_BINDING_RECORD, C.CONTAINMENT_BINDING_RECORD,
                                              C.ACCOUNTING_BINDING_RECORD, C.ACCOUNTING_TIMING_RECORD)}
    accounting = None
    binding, timing = docs[C.ACCOUNTING_BINDING_RECORD], docs[C.ACCOUNTING_TIMING_RECORD]
    if binding is not None and timing is not None:
        names = timing["caller_names"]
        accounting = dict(
            entry_module=binding.get("entry_module"), version=binding.get("version"),
            installed=binding.get("installed"), timer_installed=binding.get("timer_installed"),
            timer_removed=binding.get("timer_removed"), restored=binding.get("restored"),
            fast_path_available=binding.get("fast_path_available"), session_closed=binding.get("accounting_session_closed"),
            session=binding.get("accounting_session"), wrapped=timing["wrapped"], calls=timing["calls"],
            recorded=timing["recorded"], dropped=timing["dropped"], errors=timing["errors"],
            foreign_frames=timing["foreign_frames"],
            by_caller={name: sum(1 for c in timing["caller"] if names[c] == name) for name in names},
            result_bytes_sha256=hashlib.sha256(json.dumps(timing["result_bytes"]).encode()).hexdigest(),
            result_bytes_by_caller={name: [b for c, b in zip(timing["caller"], timing["result_bytes"]) if names[c] == name]
                                    for name in names})
    diag_root = trace_dir / C.DIAGNOSTICS_DIR
    diagnostics = None
    if diag_root.is_dir():
        sources = sorted(p.name for p in diag_root.iterdir() if p.is_dir())
        diagnostics = dict(counters=_json(diag_root / "COUNTERS.json"),
                           records=[json.loads(x) for x in (diag_root / "REFUSALS.jsonl").read_text().splitlines()]
                           if (diag_root / "REFUSALS.jsonl").exists() else [],
                           sources=sources, sources_verify=all(
                               all(hashlib.sha256((diag_root / s / n).read_bytes()).hexdigest() == d
                                   for n, d in json.loads((diag_root / s / "READY.json").read_bytes())["sha256"].items())
                               for s in sources))
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
    prep_snapshot = ((service.get("snapshot") or {}).get("prep") or {})
    prep_log = service_dir / "workers/prep.log"
    return dict(arm=name, path=path, frames=frames, scenario=run, state=state, errors=list(service.get("errors") or []),
                refusals=service.get("refusals"), outcomes=service.get("outcomes"),
                service_returncode=rig.h.process.returncode, ue_result_ok=ue_result is not None,
                worker_entries=sorted({a[a.index("-m") + 1] for a in (s.get("argv", []) for s in sub.get("substitutions", []))
                                       if "-m" in a}),
                prep_engine=dict(valid=(prep_snapshot.get("engine") or {}).get("valid"),
                                 failure=(prep_snapshot.get("engine") or {}).get("failure"),
                                 retained_jobs=(prep_snapshot.get("engine") or {}).get("retained_jobs"),
                                 deferral=(prep_snapshot.get("engine") or {}).get("deferral"),
                                 owner=(prep_snapshot.get("engine") or {}).get("owner"),
                                 gate=prep_snapshot.get("gate"), jobs=prep_snapshot.get("jobs")),
                prep_lifetime=docs[C.PREP_BINDING_RECORD], avo_binding=docs[C.AVO_BINDING_RECORD],
                containment_binding=docs[C.CONTAINMENT_BINDING_RECORD], accounting=accounting,
                prep_gc=None if docs["prep_gc_trace.json"] is None else dict(
                    exit=docs["prep_gc_trace.json"].get("exit"), safety_stop=docs["prep_gc_trace.json"].get("safety_stop"),
                    rss_final=docs["prep_gc_trace.json"].get("rss_final")),
                diagnostics=diagnostics, snapshot=snapshot,
                prep_log_tail=(prep_log.read_text(errors="replace") if prep_log.exists() else "")[-3000:])


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
    base_targets = targets(base)
    plan = fixture_plan(base_targets)
    fixtures = FixtureCorpus(plan)
    f = [c.frame_id for c in base.cases]
    arms, result = {}, None
    try:
        arms["upright_v16"] = arm_run(out / "upright_v16", "upright_v16", base, limits, spec, path="v16",
                                      scenario="normal", expect_failure=False)
        arms["upright_contained"] = arm_run(out / "upright_contained", "upright_contained", base, limits, spec,
                                            path="contained", scenario="normal", expect_failure=False)
        arms["contained"] = arm_run(out / "contained", "contained", fixtures, limits, spec, path="contained",
                                    scenario="sequence", expect_failure=False)
        arms["control_cancel"] = arm_run(out / "control_cancel", "control_cancel", base, limits, spec, path="v16",
                                         scenario="cancel", expect_failure=False)
        bbox = FixtureCorpus({19: plan[19], 21: ("BBOX_ROTATION", base_targets[21][0], 5.0, "NONZERO_BBOX_ROTATION")})
        arms["fail_closed_bbox"] = arm_run(out / "fail_closed_bbox", "fail_closed_bbox", bbox, limits, spec,
                                           path="contained", scenario="fail_closed", expect_failure=True)
        ua, uc, ct, cc, fc = (arms[k] for k in ("upright_v16", "upright_contained", "contained", "control_cancel",
                                                "fail_closed_bbox"))

        def accounting_active(arm):
            a = arm["accounting"] or {}
            return (a.get("installed") is True and a.get("timer_installed") is True and a.get("timer_removed") is True
                    and a.get("restored") is True and a.get("fast_path_available") is True and a.get("session_closed")
                    and a.get("version") == V16C.ACCOUNTING_VERSION and a.get("wrapped", "").endswith("AccountingSession")
                    and a.get("calls", 0) > 0 and a.get("calls") == a.get("recorded")
                    and (a.get("dropped"), a.get("errors"), a.get("foreign_frames")) == (0, 0, 0))
        refused_ids = {str(f[i]): i for i in REFUSED}
        status_of = {str(f[i]): (CL.STATUS_COMPETITOR if plan[i][0] == "RELOCATE_TILT" else CL.STATUS_TARGET)
                     for i in REFUSED}
        refused_rows = {k: ct["frames"].get(k, {}) for k in refused_ids}
        send_failure = str(f[11])
        bind_refused_rows = {k: v for k, v in refused_rows.items() if k != send_failure}
        valid_ids = [str(f[i]) for i in VALID]
        deferral = (ct["prep_engine"]["deferral"] or {})
        containment = deferral.get("geometry_containment") or {}
        fds = [x["fds"] for x in ct["state"].get("prep_fds", []) if x["fds"] is not None]
        records = (ct["diagnostics"] or {}).get("records", [])
        checks = dict(
            upright_both_exact=exact(ua) and exact(uc) and len(scored(ua)) == len(scored(uc)) == 11,
            upright_outcomes_q_measurement_avo_identical=ua["frames"].keys() == uc["frames"].keys() and all(
                {k: ua["frames"][x][k] for k in AP.KEYS} == {k: uc["frames"][x][k] for k in AP.KEYS}
                and ua["frames"][x].get("measurement") == uc["frames"][x].get("measurement")
                and ua["frames"][x].get("avo") == uc["frames"][x].get("avo") for x in ua["frames"]),
            upright_contained_installed_never_triggered=(uc["containment_binding"] or {}).get("installed") is True
                and (uc["containment_binding"] or {}).get("restored") is True
                and ((uc["prep_engine"]["deferral"] or {}).get("geometry_containment") or {}).get("contained") == 0,
            accounting_v3_active_in_every_completed_arm=all(accounting_active(a) for a in (ua, uc, ct, cc)),
            accounting_entry_modules=(ua["accounting"] or {}).get("entry_module") == V16C.WORKER_ENTRY
                and (uc["accounting"] or {}).get("entry_module") == (ct["accounting"] or {}).get("entry_module") == C.WORKER_ENTRY,
            upright_accounting_identical_v16_vs_contained=(ua["accounting"] or {}).get("by_caller")
                == (uc["accounting"] or {}).get("by_caller")
                and (ua["accounting"] or {}).get("result_bytes_sha256") == (uc["accounting"] or {}).get("result_bytes_sha256"),
            contained_accounting_calls_exact=(ct["accounting"] or {}).get("by_caller") == {
                "compute_unbound": len(VALID) + len(REFUSED), "stage_products": len(VALID), "bind_plan": len(VALID),
                "set_ready": len(VALID)},
            contained_valid_frame_accounting_identical_to_v16_control=all(
                (ct["accounting"] or {}).get("result_bytes_by_caller", {}).get(name)
                == (cc["accounting"] or {}).get("result_bytes_by_caller", {}).get(name)
                for name in ("stage_products", "bind_plan", "set_ready")),
            contained_paths_started_containment_worker=uc["worker_entries"] == ct["worker_entries"] == [C.WORKER_ENTRY],
            contained_layers_installed_and_restored=all(
                (a[k] or {}).get("installed") is True and (a[k] or {}).get("restored") is True
                for a in (uc, ct) for k in ("avo_binding", "prep_lifetime", "containment_binding"))
                and all((a["avo_binding"] or {}).get("installed") is True and (a["prep_lifetime"] or {}).get("installed")
                        is True and a["containment_binding"] is None for a in (ua, cc)),
            contained_service_clean=ct["service_returncode"] == 0 and not ct["errors"] and ct["ue_result_ok"]
                and not ct["state"].get("ue_error") and ct["prep_engine"]["valid"] is True,
            contained_all_frames_have_exactly_one_outcome=all(ct["frames"].get(k, {}).get("outcome")
                                                              for k in list(refused_ids) + valid_ids),
            contained_valid_frames_scored_exact=all(ct["frames"][k]["outcome"] == "SCORED" for k in valid_ids)
                and exact(ct) and len(scored(ct)) == len(VALID),
            contained_refusals_explicit_no_gt=all(v["outcome"] == OUTCOME[status_of[k]]
                and v["bind_refused"] == "DEFERRAL_REFUSED_" + CL.REASONS[status_of[k]]
                and not v["committed"] and not v["target_admitted"] and not v["identity"]
                for k, v in bind_refused_rows.items()),
            contained_refusals_get_gt_unavailable_feedback=all(v["feedback"] == "EVALUATOR_FAULT"
                and v["feedback_reason"] == "GROUND_TRUTH_UNAVAILABLE" and v["q_emitted"] is None
                and v["q_service"] is None and v["unscored_sent"] for v in bind_refused_rows.values()),
            contained_send_failure_resolved_without_score=refused_rows[send_failure].get("outcome") in (
                OUTCOME[CL.STATUS_TARGET], "CANCELLED_SEND_FAILED") and not refused_rows[send_failure].get("committed")
                and refused_rows[send_failure].get("q_service") is None
                and ("FREEZE", "OK", None) in refused_rows[send_failure].get("prep_requests", []),
            contained_competitor_exercised=refused_rows[str(f[19])].get("outcome") == OUTCOME[CL.STATUS_COMPETITOR],
            contained_engine_counters_consistent=containment.get("contained") == len(REFUSED)
                and containment.get("bind_refused", 0) + containment.get("cancelled_unbound", 0) == len(REFUSED)
                and containment.get("bind_refused") == sum(v.get("outcome", "").startswith("NO_GT_DEFERRAL_REFUSED_GT_GEOMETRY")
                                                           for v in refused_rows.values())
                and containment.get("pending") == 0 and containment.get("declined") == {},
            contained_capacity_released=(ct["prep_engine"]["owner"] or {}).get("retained_jobs") == 0
                and (ct["prep_engine"]["owner"] or {}).get("reserved_bytes") == 0,
            contained_diagnostics_bounded_and_verified=bool(ct["diagnostics"]) and len(records) == len(REFUSED)
                and ct["diagnostics"]["sources"] == [f"source_{i:03d}" for i in range(1, K.MAX_SOURCES + 1)]
                and ct["diagnostics"]["sources_verify"] and ct["diagnostics"]["counters"]["write_errors"] == 0,
            contained_no_fatal_snapshot=ct["snapshot"] is None and "BufferError" not in ct["prep_log_tail"],
            contained_prep_fds_flat=len(fds) == len(REFUSED) and max(fds) - min(fds) <= 2,
            control_cancel_clean_and_exact=cc["service_returncode"] == 0 and not cc["errors"] and exact(cc)
                and len(scored(cc)) == len(VALID),
            valid_frames_identical_to_unchanged_path_with_cancel=all(
                ct["frames"][k]["q_service"] == cc["frames"][k]["q_service"]
                and ct["frames"][k]["measurement"] == cc["frames"][k]["measurement"]
                and ct["frames"][k]["avo"] == cc["frames"][k]["avo"] for k in valid_ids),
            fail_closed_original_error=bool(fc["errors"]) and fc["errors"][0].startswith(
                "WorkerFailed: prep FREEZE: LiveGtContractError: non-zero pedestrian bounding-box rotation")
                and fc["service_returncode"] not in (0, None) and "BufferError" not in " ".join(fc["errors"]),
            fail_closed_snapshot_of_the_corrupt_frame=bool(fc["snapshot"]) and fc["snapshot"]["hashes_verify"]
                and fc["snapshot"]["frame_id"] == f[21],
            fail_closed_prior_refusal_contained_and_recorded=fc["frames"].get(str(f[19]), {}).get("outcome")
                == OUTCOME[CL.STATUS_COMPETITOR] and bool(fc["diagnostics"]) and len(fc["diagnostics"]["records"]) == 1
                and sorted(scored(fc)) == sorted([str(f[18]), str(f[20])]))
        status = "PASS" if all(checks.values()) else "FAIL"
        result = dict(schema=CL.VERSION + ".service_proof", status=status, checks=checks, arms=arms,
                      fixtures=fixtures.fixture_evidence, bbox_fixture=bbox.fixture_evidence,
                      plan={str(k): list(v) for k, v in plan.items()}, refused_modes=REFUSED, valid=VALID,
                      limits=limits, sources=C.exact_sources(ROOT), live_services_started=False,
                      carla_server_started=False,
                      scope="service path; synthetic tilts/relocation on retained frames (depth not re-rendered)")
        print(json.dumps(dict(status=status, failed=[k for k, v in checks.items() if not v],
                              contained_outcomes=ct["outcomes"], containment=containment, fds=fds,
                              fail_closed_errors=fc["errors"][:1]), indent=1, default=str))
    except BaseException as error:
        result = dict(status="FAIL", error=f"{type(error).__name__}: {error}", live_services_started=False,
                      arms_completed=sorted(arms))
        raise
    finally:
        (out / "RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str) + "\n")
        manifest = {str(p.relative_to(out)): CC.sha256(p.read_bytes()) for p in sorted(out.rglob("*"))
                    if p.is_file() and p.name != "OUTPUT_SHA256.json" and "archive" not in p.parts
                    and not p.name.endswith(".bin")}
        (out / "OUTPUT_SHA256.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    if result["status"] != "PASS":
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    run(parser.parse_args().name)
