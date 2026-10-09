"""M3.1 integration verdict (registered before launch). Integration is reported separately from performance.

PASS_M3_FULL_ROUTE_INTEGRATION requires:
  RETAINED  ``splitfusion_run5_full_live_analysis_v1.integration.evaluate`` == PASS_ACTOR_DRIVEN_INTEGRATION (R/X/A/E/S/Q
            unchanged; G1 passes through the registered ROUTE_COMPLETED branch) and the retained GC-trace check passes;
  ROUTE     child terminal ROUTE_COMPLETED; run_route_b accepted; durable route summary persisted with matching identity,
            loops_completed 1, no watchdog; ordered progress coverage >= 0.98 within 8 m on the measured ego poses;
  SCHEDULE  every radio-trace row is GOOD_ALL / CONST_GOOD_ALL_21P5DB_V1 / 21.5 dB / noise_power_dB -11.0, timing status in
            {ACK_ON_TIME, ACK_LATE, SKIP_OBSOLETE_NEVER_BURST} (counted), clean -50 dB restore verified; controller argv names
            the GOOD_ALL module; every admitted causal-SNR decision value is 21.5 dB;
  CAPACITY  C1..C5 recorded installed in their processes and restored;
  REPAIR    FINISH_RESULT inner type bool, cleanup_ok true;
  MAP       map ingest CSV present; every ACK_INSTALLED frame has an INSTALLED journal record; journal write errors 0;
  EDGE      guarded evaluator installed; guarded errors counted (reported, not a failure by themselves).
Anything else is STOP with named reasons; a ceiling/timeout/watchdog end is INCOMPLETE_ROUTE.
"""
import csv
import json
from pathlib import Path

from rl_agent.splitfusion_m3_route_b_v3 import contract as C
from . import route_geometry as RG


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


def _class_join(cell, journal):
    """Installed (journal) object type vs the exact scorer prediction class of the same decision frame (no GT)."""
    mapping = C.MAP_CLASS["mapping"]
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


def evaluate(arm_dir, *, launch_returncode, launch_log, root, do_rescore=True):
    from rl_agent.splitfusion_run5_full_live_analysis_v1.integration import evaluate as retained
    from rl_agent.splitfusion_run5_full_live_retention_v1_analysis.checks import check as gc_check
    arm = Path(arm_dir)
    checks, reasons, info = {}, [], {}

    def need(name, ok):
        checks[name] = bool(ok)
        if not ok:
            reasons.append(name)
    base = retained(arm, launch_returncode=launch_returncode, launch_log=launch_log, do_rescore=do_rescore)
    gc = gc_check(arm)
    need("RETAINED_actor_driven_integration", base["verdict"] == "PASS_ACTOR_DRIVEN_INTEGRATION")
    need("RETAINED_gc_trace", gc.get("passed") is True)
    info["retained"] = dict(verdict=base["verdict"], reasons=base.get("reasons"), info=base.get("info"))
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
    need("ROUTE_ordered_progress_coverage", coverage["coverage"] >= C.PROGRESS_COVERAGE_MIN)
    info["route"] = dict(completion=completion, outcome=(detail.get("route_outcome") or {}).get("classification"),
                         summary={k: summary.get(k) for k in ("status", "loops_completed", "driven_distance_m_median",
                                                              "simulation_duration_s_median", "wall_clock_duration_s_total",
                                                              "intervention_count_total", "collision_incident_count_total",
                                                              "return_position_error_m_max", "return_heading_error_deg_max",
                                                              "planned_route_length_m")},
                         coverage={k: coverage[k] for k in ("points", "matched", "coverage", "first_unmatched", "radius_m")},
                         pose_samples=len(poses))
    radio = _rows(cell / "radio_trace.csv")
    schedule = C.SCHEDULE
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
    need("SCHEDULE_controller_module", C.CONTROLLER_MODULE in (launch.get("argv") or []))
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
    service_sub = _j(arm / C.TRACE_DIR / "service_substitution.json") or {}
    prep = _j(arm / C.TRACE_DIR / "prep_gc_trace.json") or {}
    ids = lambda doc: [r["id"] for r in (doc or {}).get("installed", [])]
    need("CAPACITY_ue", ids(ue_cap) == ["C4_UE_TICKETS", "C5_UE_POSE_ARCHIVE", "P1_OWNED_POSE_RECONSTRUCTED",
                                        "P2_WIRE_SOURCE_CAMERA"] and ue_cap.get("restored") is True)
    pose_doc = _j(cell / "m3_route/POSE_TOLERANCE.json") or {}
    pose_checks = pose_doc.get("checks") or {}
    need("POSE_rule_registered", pose_doc.get("rule") == C.POSE)
    need("POSE_reconstructed_checks_exercised_no_rejection",
         {"owned ego", "ego/mount RGB camera", "ego/mount radar", "wire/source camera"} <= set(pose_checks)
         and all(v["calls"] > 0 and v["rejected"] == 0 for v in pose_checks.values()))
    info["pose"] = pose_checks
    class_doc = _j(cell / "direct_edge_map/m3_map_class_conversion.json") or {}
    counts = class_doc.get("counts") or {}
    need("MAP_CLASS_conversion_complete", counts.get("objects", -1) == counts.get("typed") and counts.get("unknown_class") == 0
         and counts.get("updates") is not None and class_doc.get("rule") == C.MAP_CLASS)
    info["map_class"] = counts
    need("CAPACITY_service", ids(service_sub.get("m3_capacity")) == ["C3_SERVICE_ARCHIVE_SIZING"]
         and (service_sub.get("m3_capacity") or {}).get("restored") is True)
    need("CAPACITY_prep", ids(prep.get("m3_capacity")) == ["C1_PREP_DEFERRED_ENGINE", "C2_PREP_PRODUCER_SOURCE",
                                                            "P3_GT_LIVE_CONTEXT_SOURCE_CAMERA"]
         and (prep.get("m3_capacity") or {}).get("restored") is True)
    score = _j(arm / C.TRACE_DIR / "score_gc_trace.json") or {}
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
    path = cell / C.MAP_JOURNAL_DIR / "installed_map_records.jsonl"
    if path.exists():
        with path.open() as handle:
            journal = [json.loads(line) for line in handle]
    journaled = {r["frame_id"] for r in journal if r["outcome"] == "OK" and r.get("map_document")}
    js = _j(cell / C.MAP_JOURNAL_DIR / "JOURNAL_SUMMARY.json") or {}
    need("MAP_ingest_present", bool(ingest))
    need("MAP_journal_covers_acked_installs", bool(acked) and acked <= journaled)
    join = _class_join(cell, journal)
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
    incomplete = not checks.get("ROUTE_child_terminal", False)
    verdict = ("PASS_M3_FULL_ROUTE_INTEGRATION" if not reasons
               else "INCOMPLETE_ROUTE" if incomplete else "STOP")
    return dict(schema="splitfusion.m3_route_b_v3.integration_verdict.v1", verdict=verdict, reasons=reasons, checks=checks,
                info=info, retained_checks=base.get("checks"), gc_trace=gc,
                performance_not_an_integration_criterion=True)
