"""Actor-driven integration verdict for one pilot live run (no forced-action exemptions).

PASS requires every condition below; anything else (including missing evidence) is STOP with named reasons.
  R  retained runner verdict CAPTURE_VALID and passed; G1-G9 all true, exactly that set (G8 = reward/timeout
     semantics incl. the retained actor replay with the PILOT loader -- no exemption); Phase-6 gates P0,P1,P3-P8
     all true; cell PASSED; child rc 0; collector stop == passing G1 branch; no collector failures; cleanup.
     Registered exception: G1 may pass on TRANSMITTED_BUDGET_REACHED or the retained verified whole-cycle boundary.
  X  launch exit 0 with no exception in the launch log.
  A  actor: retained actor_replay passed; independent replay of the frozen pilot actor on every recorded agent.actor
     feature vector equals the recorded decision (mode, q, state digest, heads digest) and the executed
     POLICY_DECISION; actor audits before/after PASS with the pilot hash; runtime bound to the pilot actor.
  E  executed actions: every POLICY_DECISION/HOLD inside the pilot support (modes 5/8/11, brackets), no masked
     mode; every FALLBACK executed the retained a71 (mode 11, q 9800).
  S  state: 22 features; every POLICY_DECISION has an admitted causal-SNR row whose scaled value equals feature 21;
     previous-outcome features (17 prev Q_perc, 20 prev success) equal the previous resolution of the same session.
  Q  exact re-score of every SCORED decision; terminal accounting; history chain; bounded queues; no pre-action GT
     work; service/UE/edge clean exit and restoration; retained thread caps/affinity; anchor provenance present
     exactly for catalogue-anchor executed actions.
"""
import collections
import json
from pathlib import Path

from rl_agent.splitfusion_fixed_action_screen_analysis_v2.analyze import load_jsonl, profile, rescore
from rl_agent.splitfusion_run5_restricted_pilot_v1 import support as SUP

G_ALL = ("G1_FRAME_ACCOUNTING", "G2_ACTOR_HASH_UNCHANGED", "G3_ZERO_IDENTITY_CONFLICTS", "G4_COVERAGE_REPORTED",
         "G5_SAMPLER_REPORTED", "G6_NO_PROFILER_INDUCED_FAULT", "G7_INSTRUMENTATION_EVIDENCE_VALID",
         "G8_REWARD_TIMEOUT_SEMANTICS", "G9_HOST_RESTORED_CLEAN")
P_ALL = ("P0_POLICY_COVERAGE", "P1_IDENTITY", "P3_HOLD_TICKET", "P4_ACCOUNTING", "P5_NO_ACTOR_AFTER_REFUSAL",
         "P6_RESTORE_COLD", "P7_FEEDBACK_OVER_DOWNLINK", "P8_NO_INFRASTRUCTURE_FAULT")
PILOT_TREE = "bf8b0e96c7023dfca0b3e5323b2829b952afe0d7c3a385b727c57268e5323a21"


def _j(path):
    try:
        return json.loads(Path(path).read_bytes())
    except (OSError, ValueError):
        return None


def evaluate(arm_dir, *, launch_returncode, launch_log, do_rescore=True):
    from rl_agent.splitfusion_run5_pilot_live_v1.actor import load_registered_actor
    arm = Path(arm_dir)
    reasons, checks, info = [], {}, {}

    def need(name, ok):
        checks[name] = bool(ok)
        if not ok:
            reasons.append(name)
        return ok
    reg = _j(arm / "registration.json") or {}
    qual = _j(arm / "capture/QUALIFICATION_RESULT.json")
    cells = sorted((arm / "capture/cells").glob("*")) if (arm / "capture/cells").is_dir() else []
    if qual is None or len(cells) != 1:
        return dict(verdict="STOP", reasons=["MISSING_RESULTS"], checks={})
    cell = cells[0]
    cell_result = _j(cell / "CELL_RESULT.json") or {}
    child = cell_result.get("child") or {}
    collector = child.get("collector") or {}
    gates = qual.get("gates") or {}
    phase6 = ((qual.get("phase6_report_only") or {}).get("gates")) or {}
    g1 = qual.get("g1_detail") or {}
    analysis = _j(arm / "capture/PROFILER_ANALYSIS.json") or {}
    sem = analysis.get("semantics") or {}
    # R
    need("R_verdict_capture_valid", qual.get("verdict") == "CAPTURE_VALID" and qual.get("passed") is True)
    need("R_g_gates_all_true", set(gates) == set(G_ALL) and all(gates[k] is True for k in G_ALL))
    need("R_phase6_gates_all_true", set(phase6) == set(P_ALL) and all(phase6[k] is True for k in P_ALL))
    need("R_cell_passed", cell_result.get("status") == "PASSED" and not cell_result.get("error") and not child.get("error"))
    need("R_child_rc0", child.get("return_code") == 0)
    need("R_collector_stop_matches_g1", g1.get("passed") is True and collector.get("stop_reason") == g1.get("stop_reason"))
    need("R_no_collector_failures", collector.get("collector_failures") == [])
    need("R_child_cleanup", all((child.get("cleanup") or {}).get(k) is True for k in ("edge_stopped", "map_stopped", "target_snr_restored")))
    need("R_cell_cleanup", (cell_result.get("cleanup") or {}).get("all_gates_passed") is True)
    info["g1_stop_reason"] = g1.get("stop_reason")
    # X
    tail = Path(launch_log).read_text()[-6000:] if Path(launch_log).exists() else ""
    exceptions = [l for l in tail.splitlines() if l and not l.startswith((" ", "\t", "Traceback"))
                  and ("Error" in l.split(":")[0] or "Exception" in l.split(":")[0])]
    need("X_launch_rc0_no_exception", launch_returncode == 0 and not exceptions)
    # A
    need("A_retained_actor_replay", (sem.get("actor_replay") or {}).get("passed") is True)
    need("A_reward_semantics_and_receipt_marks", (sem.get("reward") or {}).get("passed") is True
         and (sem.get("receipt_marks") or {}).get("passed") is True)
    audits = qual.get("audits") or {}
    need("A_audits_pass_with_pilot_hash", all(((audits.get(k) or {}).get("verdict") == "PASS"
                                               and (audits.get(k) or {}).get("actor_boundary_sha256", PILOT_TREE) == PILOT_TREE)
                                              for k in ("before", "after")))
    snr = _j(cell / "run5_policy/SNR_EVIDENCE.json") or {}
    need("A_runtime_bound_to_pilot", snr.get("actor_tree_sha256") == PILOT_TREE and snr.get("runtime_bound") is True
         and snr.get("controller_started") is True)
    evidence = _j(cell / "run4_phase6/PHASE6_UE_EVIDENCE.json") or {}
    decisions = evidence.get("decisions") or []
    policy = [d for d in decisions if d["kind"] == "POLICY_DECISION"]
    by_frame = {int(d["frame_id"]): d for d in policy}
    _, _, spans = profile(cell / "profiler/ue.profile.jsonl")
    actor = load_registered_actor()
    mismatches = []
    for span in spans:
        fid = int(span["frame_id"])
        d = actor.act_on_vector(tuple(float(v) for v in span["features"]))
        got = (d.mode_id, d.q_e4, d.state_sha256, d.heads_sha256)
        rec = (span.get("decided_mode_id"), span.get("decided_q_e4"), span.get("state_sha256"), span.get("heads_sha256"))
        live = by_frame.get(fid) or {}
        if got != rec or (live.get("mode_id"), live.get("q_e4")) != got[:2]:
            mismatches.append(dict(frame_id=fid, replay=got, recorded=rec, executed=(live.get("mode_id"), live.get("q_e4"))))
    need("A_independent_pilot_replay", len(spans) == len(policy) > 0 and not mismatches)
    info["replay"] = dict(spans=len(spans), policy_decisions=len(policy), mismatches=mismatches[:5])
    # E
    need("E_policy_actions_in_support", all(SUP.in_support(int(d["mode_id"]), int(d["q_e4"]))
                                            for d in decisions if d["kind"] in ("POLICY_DECISION", "POLICY_HOLD")))
    need("E_fallbacks_a71", all((int(d["mode_id"]), int(d["q_e4"]), d.get("anchor_action_id")) == (11, 9800, 71)
                                for d in decisions if d["kind"] == "FALLBACK"))
    info["fallback_reasons"] = dict(collections.Counter(r.split(":")[0] for d in decisions if d["kind"] == "FALLBACK"
                                                        for r in d.get("fallback_reasons", [])))
    # S
    rows = {int(r["frame_id"]): r for r in snr.get("decision_snr") or [] if r.get("admitted")}
    span_by_frame = {int(s["frame_id"]): s for s in spans}
    snr_ok = all(fid in rows and abs(rows[fid]["snr_scaled"] - span_by_frame[fid]["features"][21]) < 1e-12
                 and len(span_by_frame[fid]["features"]) == 22 for fid in by_frame if fid in span_by_frame)
    need("S_causal_snr_feature", snr_ok and set(by_frame) <= set(span_by_frame))
    resolutions = {int(r["action_open_timestamp_ns"]): r for r in evidence.get("resolutions") or []}
    prev_ok, prev_checked = True, 0
    ordered = sorted(policy, key=lambda d: d["action_open"]["ns"])
    for a, b in zip(ordered, ordered[1:]):
        if a["decision_session_uuid"] != b["decision_session_uuid"] or int(b["frame_id"]) not in span_by_frame:
            continue
        res = resolutions.get(int(a["action_open"]["ns"]))
        if res is None:
            continue
        f = span_by_frame[int(b["frame_id"])]["features"]
        success = res.get("terminal") == "SUCCESS"
        expect_q = float(res["q_perc"]) if success else 0.0
        prev_checked += 1
        prev_ok &= (f[20] == (1.0 if success else 0.0)) and abs(f[17] - expect_q) < 1e-9
    need("S_previous_qperc_feature", prev_ok and prev_checked > 0)
    info["previous_outcome_pairs_checked"] = prev_checked
    # Q
    frames = [f for f in load_jsonl(cell / "numeric_gt/service/FRAMES.jsonl") if f.get("decision")]
    scored = [f for f in frames if f.get("outcome") == "SCORED"]
    rescored = {f["frame_id"]: rescore(cell / f"numeric_gt/service/archive/frame_{f['frame_id']}/scorer")
                for f in scored} if do_rescore else {}
    need("Q_exact_rescore", (not do_rescore) or (len(rescored) == len(scored) > 0 and all(rescored.values())))
    edge_doc = _j(cell / "profiler/edge/numeric_runtime/EDGE_EARLY_GT.json") or {}
    state = edge_doc.get("state") or {}
    records = state.get("records") or []
    emissions = collections.Counter(int(r["frame_id"]) for r in records)
    reached = {f["frame_id"] for f in frames if f.get("prediction_received_raw_ns")}
    need("Q_terminal_accounting", all(f.get("outcome") for f in frames) and set(emissions) == reached <= {f["frame_id"] for f in frames}
         and all(v == 1 for v in emissions.values()) and all(r["q_perc"] is None for r in records if r["kind"] != "DELIVERED_SUCCESS")
         and len(evidence.get("feedback_rows") or []) == len(records))
    commits = [f["commit"] for f in frames if f.get("committed")]
    need("Q_history_chain", all(a["history_after_sha256"] == b["history_before_sha256"] for a, b in zip(commits, commits[1:])))
    service = _j(cell / "numeric_gt/service/SERVICE_RESULT.json") or {}
    limits = service.get("limits") or {}
    samples = _j(cell / "numeric_gt/service/QUEUE_SAMPLES.json") or []
    need("Q_bounded", len(service.get("refusals") or []) <= max(5, len(frames) // 20) and all(
        s["prep_retained"] <= limits.get("prep_jobs", 0) and s["scorer_slots"] <= limits.get("scorer_jobs", 0)
        and s["freeze_queue"] <= limits.get("freeze_queue", 0) for s in samples)
        and (state.get("evaluator_counters") or {}).get("queue_overflow", 1) == 0 and limits.get("max_frames") == reg.get("transmitted_budget"))
    need("Q_no_pre_action_gt_work", all(next((e["raw_ns"] for e in f["events"] if e["name"] == "freeze_enqueued_on_ue_thread"), -1)
                                        >= f["action_open_raw_ns"] for f in frames if f.get("outcome") != "NO_GT_UE_FREEZE_QUEUE_FULL"))
    ue_result = _j(cell / "numeric_gt/EXACT_GT_UE_RESULT.json") or {}
    restoration = _j(arm / "RESTORATION_LATEST.json") or {}
    need("Q_cleanup", not service.get("errors") and not ue_result.get("errors") and ue_result.get("service_returncode") == 0
         and edge_doc.get("error") is None and restoration.get("passed") is True and not qual.get("leftover_processes"))
    threadcap = (qual.get("threadcap") or {}).get("checks") or {}
    affinity = _j(arm / "AFFINITY_APPLIED.json") or {}
    need("Q_retained_settings", threadcap.get("E1_UE_SETTINGS_RECORDED") is True and threadcap.get("E2_UE_CAP_STATE_AS_REGISTERED") is True
         and affinity.get("passed") is True)
    executed = {int(d["frame_id"]): d for d in decisions}

    def provenance(fid):
        return [(cell / f"numeric_gt/service/archive/frame_{fid}/{side}/anchor_provenance.json").exists()
                for side in ("scorer", "producer")]
    need("Q_anchor_provenance_by_executed_identity", all(
        provenance(f["frame_id"]) == ([True, True] if (executed.get(f["frame_id"]) or {}).get("anchor_action_id") is not None
                                      else [False, False]) for f in scored))
    info.update(decisions=len(frames), scored=len(scored), rescored_exact=sum(rescored.values()) if rescored else None,
                outcomes=service.get("outcomes"), refusals=len(service.get("refusals") or []),
                policy_decisions=len(policy), holds=sum(d["kind"] == "POLICY_HOLD" for d in decisions),
                fallbacks=sum(d["kind"] == "FALLBACK" for d in decisions), retained_verdict=qual.get("verdict"),
                gates=gates, phase6=phase6, coverage=(evidence.get("coverage") or {}))
    return dict(verdict="PASS_ACTOR_DRIVEN_INTEGRATION" if not reasons else "STOP", reasons=reasons, checks=checks, info=info)
