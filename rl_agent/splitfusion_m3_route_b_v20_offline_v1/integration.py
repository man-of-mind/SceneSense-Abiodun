"""Version-bound integration verdict for the V20 live-view GOOD_ALL capture (copy of the V17/V11 evaluator; identical
checks; the schedule expectations come from the VERIFIED V20 registration, i.e. V11's GOOD_ALL 23.5 dB / RFsim -12.0).

As for V11, the host's numeric exit status IS captured at launch, so the retained check set runs unmodified and must
itself return PASS_ACTOR_DRIVEN_INTEGRATION.  Channel performance (timeliness, measured SNR, MCS) is NOT an
integration criterion; the registered channel-fidelity checks are reported separately by the analysis.  The live-view
tap status is reported in ``info.live_view`` and is never an integration criterion.

This is the retained V3 check set with its contract reads made explicit. The
caller supplies a binding only after the capture prospective, prerequisite,
registration, source inventory and declared V20 revision have been verified.
No historical module global is rebound.

PASS_M3_FULL_ROUTE_INTEGRATION requires:
  RETAINED  ``splitfusion_run5_full_live_analysis_v1.integration.evaluate`` == PASS_ACTOR_DRIVEN_INTEGRATION (R/X/A/E/S/Q
            unchanged; G1 passes through the registered ROUTE_COMPLETED branch) and the retained GC-trace check passes;
  ROUTE     child terminal ROUTE_COMPLETED; run_route_b accepted; durable route summary persisted with matching identity,
            loops_completed 1, no watchdog; ordered progress coverage >= 0.98 within 8 m on the measured ego poses;
  SCHEDULE  every radio-trace row matches the VERIFIED REGISTRATION's schedule (v20: GOOD_ALL /
            CONST_GOOD_ALL_23P5DB_V1 / 23.5 dB / noise_power_dB -12.0), timing status in
            {ACK_ON_TIME, ACK_LATE, SKIP_OBSOLETE_NEVER_BURST} (counted), clean -50 dB restore verified; controller argv names
            the registered (sealed V11 GOOD_ALL) module; every admitted causal-SNR decision value is the registered
            target (23.5 dB) and its scaled feature (23.5-5.5)/19;
  CAPACITY  C1..C5 recorded installed in their processes and restored;
  REPAIR    FINISH_RESULT inner type bool, cleanup_ok true;
  MAP       map ingest CSV present; every ACK_INSTALLED frame has an INSTALLED journal record; journal write errors 0;
  EDGE      guarded evaluator installed; guarded errors counted (reported, not a failure by themselves).
  GT (v20)  the reviewed GT containment was installed in the prep worker and restored; the prep engine's containment
            counters are consistent with the service outcomes (bind refusals == NO_GT_DEFERRAL_REFUSED_GT_GEOMETRY_*
            frames; contained == bind_refused + cancelled_unbound; nothing pending; NOTHING DECLINED); every geometry
            refused frame is uncommitted, has no admitted target and no score; refusal diagnostics bounded with no
            write error; storage accounting V3.v2 session active (installed, timer installed/removed, restored,
            fast path, closed), timing complete (0 dropped/errors/foreign) and its per-caller counts reconcile
            with the prep FREEZE/BIND replies (compute_unbound = FREEZE OK; stage_products = FREEZE OK - contained;
            bind_plan = set_ready = BIND OK).  Geometry refusals are REGISTERED GT-unavailable outcomes, reported in
            ``info.gt_containment`` (never an integration failure by themselves; the retained Q_bounded refusal
            bound applies unchanged).  Actor replay covers ALL policy decisions; exact rescoring SCORED only.
Anything else is STOP with named reasons; a ceiling/timeout/watchdog end is INCOMPLETE_ROUTE.
"""
import csv
import json
from pathlib import Path

from rl_agent.splitfusion_m3_route_b_v3_analysis import route_geometry as RG


def _j(path):
    try:
        return json.loads(Path(path).read_bytes())
    except (OSError, ValueError):
        return None


def _rows(path):
    try:
        with open(path, newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def _class_join(cell, journal, binding):
    """Installed (journal) object type vs the exact scorer prediction class of the same decision frame (no GT)."""
    mapping = binding.MAP_CLASS["mapping"]
    checked = mismatched = unjoined = frames = 0
    for row in journal:
        if row["outcome"] != "OK" or not row.get("map_document"):
            continue
        path = cell / f"numeric_gt/service/archive/frame_{row['frame_id']}/scorer/numerical_score.json"
        if not path.exists():
            continue                                    # hold/fallback frames: no scorer record (type checked by counts)
        frames += 1
        preds = json.loads(path.read_bytes()).get("prediction_records") or []
        for o in row["map_document"]["record"].get("objects", []):
            hit = [p for p in preds if abs(p["score"] - o["score"]) < 1e-6 and abs(p["world_x"] - o["location"]["x"]) < 1e-3
                   and abs(p["world_y"] - o["location"]["y"]) < 1e-3]
            if len(hit) != 1:
                unjoined += 1
                continue
            checked += 1
            mismatched += int(o.get("type") != mapping.get(hit[0]["class_name"]))
    return dict(frames=frames, checked_objects=checked, mismatched=mismatched, unjoined=unjoined)


def evaluate(arm_dir, *, launch_log, launch_terminal, launch_returncode, root, binding, do_rescore=True):
    from rl_agent.splitfusion_run5_full_live_analysis_v1.integration import evaluate as retained
    from rl_agent.splitfusion_run5_full_live_retention_v1_analysis.checks import check as gc_check
    arm = Path(arm_dir)
    checks, reasons, info = {}, [], {}

    def need(name, ok):
        checks[name] = bool(ok)
        if not ok:
            reasons.append(name)
    # V20 (as V11/V17) captures the host's real numeric exit status at launch, so the retained check set runs complete: no
    # predicate is replaced and nothing is inferred.  The independently parsed success terminal is kept as a
    # second, separate requirement.
    if not isinstance(launch_returncode, int):
        raise ValueError("V20 requires the captured numeric host exit status")
    base = retained(arm, launch_returncode=launch_returncode, launch_log=launch_log, do_rescore=do_rescore)
    gc = gc_check(arm)
    terminal_ok = launch_terminal.get("accepted") is True
    need("RETAINED_actor_driven_integration", base.get("verdict") == "PASS_ACTOR_DRIVEN_INTEGRATION")
    need("RETAINED_recorded_success_terminal_no_exception", terminal_ok)
    need("RETAINED_gc_trace", gc.get("passed") is True)
    info["retained"] = dict(verdict=base["verdict"], reasons=base.get("reasons"), info=base.get("info"),
                            numeric_host_exit_status=launch_returncode, recorded_terminal=launch_terminal,
                            correction=None)
    cells = sorted((arm / "capture/cells").glob("*")) if (arm / "capture/cells").is_dir() else []
    if len(cells) != 1:
        return dict(verdict="STOP", reasons=["MISSING_CELL"], checks=checks, info=info, retained_checks=base.get("checks"))
    cell = cells[0]
    result = _j(cell / "CELL_RESULT.json") or {}
    child = result.get("child") or {}
    detail = child.get("route_detail") or {}
    completion = child.get("m3_route_completion") or {}
    summary = _j(cell / "route_metrics_summary.json") or {}
    need("ROUTE_child_terminal", completion.get("terminal") == "ROUTE_COMPLETED" and child.get("stop_reason") == "ROUTE_COMPLETED")
    need("ROUTE_run_route_b_accepted", completion.get("run_route_b_accepted") is True and detail.get("route_completed") is True)
    need("ROUTE_summary", detail.get("route_summary_persisted") is True and detail.get("route_summary_identity_ok") is True
         and summary.get("loops_completed") == 1 and summary.get("watchdog_aborted_any") is False)
    route = RG.load_route(root)
    poses = RG.poses_from_archive(cell / "pose_archive/pose_evidence.jsonl") if (cell / "pose_archive").is_dir() else []
    coverage = RG.ordered_coverage([(x, y) for _, x, y, _ in poses], route)
    need("ROUTE_ordered_progress_coverage", coverage["coverage"] >= binding.PROGRESS_COVERAGE_MIN)
    info["route"] = dict(completion=completion, outcome=(detail.get("route_outcome") or {}).get("classification"),
                         summary={k: summary.get(k) for k in ("status", "loops_completed", "driven_distance_m_median",
                                                              "simulation_duration_s_median", "wall_clock_duration_s_total",
                                                              "intervention_count_total", "collision_incident_count_total",
                                                              "return_position_error_m_max", "return_heading_error_deg_max",
                                                              "planned_route_length_m")},
                         coverage={k: coverage[k] for k in ("points", "matched", "coverage", "first_unmatched", "radius_m")},
                         pose_samples=len(poses))
    radio = _rows(cell / "radio_trace.csv")
    schedule = binding.SCHEDULE
    statuses = {}
    for row in radio:
        statuses[row["command_timing_status"]] = statuses.get(row["command_timing_status"], 0) + 1
    need("SCHEDULE_rows", bool(radio) and all(
        r["profile_id"] == schedule["schedule_id"] and r["trace_id"] == schedule["trace_id"]
        and float(r["target_snr_db"]) == schedule["target_snr_db"]
        and float(r["mapped_rfsim_command_db"]) == schedule["expected_rfsim_command_db"]
        and r["command_timing_status"] in ("ACK_ON_TIME", "ACK_LATE", "SKIP_OBSOLETE_NEVER_BURST") for r in radio))
    need("SCHEDULE_restore", (child.get("cleanup") or {}).get("target_snr_restored") is True)
    launch = _j(cell / "run5_policy/CONTROLLER_LAUNCH.json") or {}
    need("SCHEDULE_controller_module", binding.CONTROLLER_MODULE in (launch.get("argv") or []))
    snr = _j(cell / "run5_policy/SNR_EVIDENCE.json") or {}
    admitted = [r for r in snr.get("decision_snr") or [] if r.get("admitted")]
    values = sorted({(r.get("snr") or {}).get("value_db") for r in admitted} - {None})
    kinds = sorted({(r.get("snr") or {}).get("kind") for r in admitted} - {None})
    need("SCHEDULE_actor_observation_constant", bool(admitted) and values == [schedule["target_snr_db"]] and all(
        abs(float(r["snr_scaled"]) - (schedule["target_snr_db"] - 5.5) / 19.0) < 1e-12 for r in admitted))
    info["schedule"] = dict(rows=len(radio), timing_status=statuses, admitted_decisions=len(admitted),
                            not_admitted=sum(1 for r in snr.get("decision_snr") or [] if not r.get("admitted")),
                            admitted_proxy_values_db=values, admitted_proxy_kind=kinds,
                            proxy_semantics="commanded target ACKed by RFsim, published by the controller; NOT a measured "
                                            "achieved SNR")
    ue_cap = _j(cell / "m3_route/UE_CAPACITY_SUBSTITUTIONS.json") or {}
    service_sub = _j(arm / binding.TRACE_DIR / "service_substitution.json") or {}
    prep = _j(arm / binding.TRACE_DIR / "prep_gc_trace.json") or {}
    ids = lambda doc: [r["id"] for r in (doc or {}).get("installed", [])]
    need("CAPACITY_ue", ids(ue_cap) == ["C4_UE_TICKETS", "C5_UE_POSE_ARCHIVE", "P1_OWNED_POSE_RECONSTRUCTED",
                                        "P2_WIRE_SOURCE_CAMERA"] and ue_cap.get("restored") is True)
    pose_doc = _j(cell / "m3_route/POSE_TOLERANCE.json") or {}
    pose_checks = pose_doc.get("checks") or {}
    need("POSE_rule_registered", pose_doc.get("rule") == binding.POSE)
    need("POSE_reconstructed_checks_exercised_no_rejection",
         {"owned ego", "ego/mount RGB camera", "ego/mount radar", "wire/source camera"} <= set(pose_checks)
         and all(v["calls"] > 0 and v["rejected"] == 0 for v in pose_checks.values()))
    info["pose"] = pose_checks
    class_doc = _j(cell / "direct_edge_map/m3_map_class_conversion.json") or {}
    counts = class_doc.get("counts") or {}
    need("MAP_CLASS_conversion_complete", counts.get("objects", -1) == counts.get("typed") and counts.get("unknown_class") == 0
         and counts.get("updates") is not None and class_doc.get("rule") == binding.MAP_CLASS)
    info["map_class"] = counts
    need("CAPACITY_service", ids(service_sub.get("m3_capacity")) == ["C3_SERVICE_ARCHIVE_SIZING"]
         and (service_sub.get("m3_capacity") or {}).get("restored") is True)
    need("CAPACITY_prep", ids(prep.get("m3_capacity")) == ["C1_PREP_DEFERRED_ENGINE", "C2_PREP_PRODUCER_SOURCE",
                                                            "P3_GT_LIVE_CONTEXT_SOURCE_CAMERA"]
         and (prep.get("m3_capacity") or {}).get("restored") is True)
    score = _j(arm / binding.TRACE_DIR / "score_gc_trace.json") or {}
    need("POSE_gt_workers", ids(score.get("m3_capacity")) == ["P3_GT_LIVE_CONTEXT_SOURCE_CAMERA"]
         and (score.get("m3_capacity") or {}).get("restored") is True
         and all((((w.get("m3_pose") or {}).get("checks") or {}).get("live context/source camera") or {}).get("calls", 0) > 0
                 and (((w.get("m3_pose") or {}).get("checks") or {}).get("live context/source camera") or {}).get("rejected") == 0
                 for w in (prep, score)))
    info["pose_gt_workers"] = {role: (w.get("m3_pose") or {}).get("checks") for role, w in (("prep", prep), ("score", score))}
    finish = _j(cell / "m3_route/FINISH_RESULT.json") or {}
    need("REPAIR_finish_result", finish.get("inner_type") == "bool" and finish.get("cleanup_ok") is True)
    ingest = _rows(cell / "direct_edge_map/direct_map_ingest.csv")
    feedback = _rows(cell / "map_feedback.csv")
    acked = {int(r["frame_id"]) for r in feedback if r.get("status") == "ACK_INSTALLED"}
    journal = []
    path = cell / binding.MAP_JOURNAL_DIR / "installed_map_records.jsonl"
    if path.exists():
        with path.open() as handle:
            journal = [json.loads(line) for line in handle]
    journaled = {r["frame_id"] for r in journal if r["outcome"] == "OK" and r.get("map_document")}
    js = _j(cell / binding.MAP_JOURNAL_DIR / "JOURNAL_SUMMARY.json") or {}
    need("MAP_ingest_present", bool(ingest))
    need("MAP_journal_covers_acked_installs", bool(acked) and acked <= journaled)
    join = _class_join(cell, journal, binding)
    need("MAP_CLASS_installed_type_equals_prediction_class", join["checked_objects"] > 0 and join["mismatched"] == 0
         and join["unjoined"] == 0)
    info["map_class_join"] = join
    need("MAP_journal_clean", (js.get("counts") or {}).get("write_errors") == 0 and (js.get("counts") or {}).get("parse_errors") == 0)
    info["map"] = dict(ingest_rows=len(ingest), outcomes={o: sum(1 for r in ingest if r["outcome"] == o)
                                                           for o in sorted({r["outcome"] for r in ingest})},
                       ack_installed=len(acked), journal_installed=len(journaled), journal_counts=js.get("counts"))
    edge = _j(cell / "profiler/edge/numeric_runtime/EDGE_EARLY_GT.json") or {}
    guard = ((edge.get("state") or {}).get("m3_edge_guard")) or {}
    counters = (edge.get("state") or {}).get("evaluator_counters") or {}
    need("EDGE_guard_installed", guard.get("installed") is True and guard.get("instance_class") is not None)
    info["edge_guard"] = dict(instance_class=guard.get("instance_class"), guarded_errors=len(guard.get("guarded_errors") or []),
                              queue_overflow=counters.get("queue_overflow"), submit_max_ns=counters.get("m3_submit_max_ns"),
                              overflow_branch_max_ns=counters.get("m3_overflow_branch_max_ns"))
    gt_containment(arm, cell, binding, need, info)
    # v20: the display-only live-view taps are REPORTED, never an integration criterion (they are off the decision path).
    live = cell / "live_view"
    info["live_view"] = dict(map_tap=_j(live / "map_tap_status.json"), ue_pose_tap=_j(live / "ue_pose_tap_status.json"),
                             files=sorted(p.name for p in live.iterdir()) if live.is_dir() else None,
                             integration_criterion=False)
    incomplete = not checks.get("ROUTE_child_terminal", False)
    verdict = ("PASS_M3_FULL_ROUTE_INTEGRATION" if not reasons
               else "INCOMPLETE_ROUTE" if incomplete else "STOP")
    return dict(schema="splitfusion.m3_route_b_v20_offline_v1.integration_verdict.v1", verdict=verdict, reasons=reasons, checks=checks,
                info=info, retained_checks=base.get("checks"), gc_trace=gc,
                performance_not_an_integration_criterion=True)


def gt_containment(arm, cell, binding, need, info):
    """v20 registered GT containment + accounting checks (see module docstring)."""
    from rl_agent.splitfusion_m3_route_b_v20 import contract as V20
    trace = arm / binding.TRACE_DIR
    service = _j(cell / "numeric_gt/service/SERVICE_RESULT.json") or {}
    path = cell / "numeric_gt/service/FRAMES.jsonl"
    frames = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    frames = [f for f in frames if f.get("decision")]
    engine = ((service.get("snapshot") or {}).get("prep") or {}).get("engine") or {}
    counters = (engine.get("deferral") or {}).get("geometry_containment") or {}
    record = _j(trace / V20.CONTAINMENT_BINDING_RECORD) or {}
    accounting = _j(trace / V20.ACCOUNTING_BINDING_RECORD) or {}
    timing = _j(trace / V20.ACCOUNTING_TIMING_RECORD) or {}
    diagnostics = _j(trace / V20.CONTAINMENT_DIAGNOSTICS_DIR / "COUNTERS.json")
    geometry = [f for f in frames if str(f.get("outcome") or "").startswith(V20.CONTAINMENT_OUTCOME_PREFIX)]

    def replies(op):
        return sum(1 for f in frames for r in f.get("requests") or []
                   if r.get("worker") == "prep" and r.get("op") == op and r.get("status") == "OK")
    names = timing.get("caller_names") or []
    by_caller = {n: sum(1 for c in timing.get("caller") or [] if names[c] == n) for n in names}
    freeze_ok, bind_ok, contained = replies("FREEZE"), replies("BIND"), counters.get("contained")
    need("GT_containment_installed_and_restored", record.get("installed") is True and record.get("restored") is True
         and record.get("exit") is None and record.get("entry_module") == V20.WORKER_ENTRY
         and record.get("base_stack") == V20.V16_WORKER_ENTRY
         and record.get("sources") == {n: V20.CONTAINMENT_QUALIFIED[f"{V20.CONTAINMENT_PACKAGE}/{n}.py"]
                                       for n in ("classify", "core", "engine")})
    need("GT_containment_counters_consistent", counters.get("version") == V20.CONTAINMENT_VERSION
         and counters.get("declined") == {} and counters.get("pending") == 0
         and contained == counters.get("bind_refused", -1) + counters.get("cancelled_unbound", -1)
         and counters.get("bind_refused") == len(geometry))
    need("GT_containment_refused_frames_never_scored", all(
        not f.get("committed") and not f.get("target_admitted") and f.get("result") is None
        and str(f.get("bind_refused") or "").startswith("DEFERRAL_REFUSED_GT_GEOMETRY_ENVELOPE_") for f in geometry))
    need("GT_containment_diagnostics_bounded", (contained == 0 and diagnostics is None) or (
        diagnostics is not None and diagnostics.get("refusals") == contained and diagnostics.get("write_errors") == 0
        and diagnostics.get("records_written", 0) + diagnostics.get("records_dropped", 0) == contained))
    session = accounting.get("accounting_session") or {}
    need("GT_accounting_v3_active_and_restored", accounting.get("installed") is True
         and accounting.get("timer_installed") is True and accounting.get("timer_removed") is True
         and accounting.get("restored") is True and accounting.get("exit") is None
         and accounting.get("fast_path_available") is True and accounting.get("accounting_session_closed") is True
         and accounting.get("version") == V20.ACCOUNTING_VERSION and session.get("version") == V20.ACCOUNTING_VERSION
         and session.get("calls", 0) > 0 and session.get("failed_calls") == 0)
    need("GT_accounting_timing_complete", timing.get("schema") == V20.ACCOUNTING_TIMING_SCHEMA
         and timing.get("calls", 0) > 0 and timing.get("recorded") == timing.get("calls")
         and (timing.get("dropped"), timing.get("errors"), timing.get("foreign_frames")) == (0, 0, 0))
    need("GT_accounting_per_caller_reconciles", isinstance(contained, int)
         and by_caller.get("compute_unbound") == freeze_ok and by_caller.get("stage_products") == freeze_ok - contained
         and by_caller.get("bind_plan") == bind_ok and by_caller.get("set_ready") == bind_ok)
    outcomes = {}
    for f in frames:
        outcomes[f.get("outcome")] = outcomes.get(f.get("outcome"), 0) + 1
    info["gt_containment"] = dict(
        engine_counters=counters, diagnostics=diagnostics, binding_record={k: record.get(k) for k in
                                                                            ("installed", "restored", "pid", "entry_module")},
        geometry_refused_frames=[dict(frame_id=f["frame_id"], outcome=f.get("outcome"),
                                      prediction_received=bool(f.get("prediction_received_raw_ns")),
                                      unscored_sent=bool(f.get("edge_unscored_sent"))) for f in geometry],
        decision_outcomes=outcomes, freeze_ok=freeze_ok, bind_ok=bind_ok, accounting_by_caller=by_caller,
        accounting_session={k: session.get(k) for k in ("calls", "full_calls", "failed_calls", "fast_path",
                                                         "peak_memo_objects", "max_objects", "closed")},
        semantics="geometry refusals are registered GT-unavailable outcomes; actor replay covers all policy "
                  "decisions; exact rescoring covers SCORED decisions only")
