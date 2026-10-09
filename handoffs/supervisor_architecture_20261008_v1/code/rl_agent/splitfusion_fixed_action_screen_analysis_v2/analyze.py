"""Per-arm integration checks (registered I1-I10), per-decision rows and per-arm metrics.

Read-only on attempts. Run from the main checkout; the scorer modules used for exact offline
re-scoring are verified byte-identical to the live worktree copies before use.
"""
import collections
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

MAIN = Path(__file__).resolve().parents[2]
WT = MAIN.parent / "abiodun_run4_single_gpu_profiler_v1"
DEADLINE = 170.0
COLD = 2


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def q(values):
    v = sorted(float(x) for x in values if x is not None)
    if not v:
        return dict(n=0)
    n = len(v)
    med = v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2
    return dict(n=n, min=round(v[0], 3), p50=round(med, 3), p90=round(v[max(0, math.ceil(.90 * n) - 1)], 3),
                p95=round(v[max(0, math.ceil(.95 * n) - 1)], 3), max=round(v[-1], 3), mean=round(sum(v) / n, 4))


def wilson(k, n, z=1.959964):
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 4), round(centre + half, 4)]


def load_jsonl(path):
    return [json.loads(line) for line in open(path)] if Path(path).exists() else []


def profile(path):
    spans, marks, all_spans = collections.defaultdict(dict), collections.defaultdict(dict), []
    for line in open(path):
        d = json.loads(line)
        fid = d.get("frame_id", d.get("carla_frame"))
        if d.get("r") == "span" and d.get("stage") == "agent.actor" and d.get("ph") == "live":
            all_spans.append(d)
        if fid is None or d.get("ph") != "live":
            continue
        if d["r"] == "span":
            spans[int(fid)].setdefault(d["stage"], (d["start_raw_ns"], d["end_raw_ns"]))
        elif d["r"] == "mark":
            marks[int(fid)].setdefault(d["stage"], d)
    return spans, marks, all_spans


class Arm:
    def __init__(self, directory):
        self.dir = Path(directory).resolve(strict=True)
        self.reg = json.loads((self.dir / "registration.json").read_bytes())
        self.forced = self.reg["forced_action"]
        cells = list((self.dir / "capture/cells").iterdir()) if (self.dir / "capture/cells").is_dir() else []
        self.cell = cells[0] if len(cells) == 1 else None

    def j(self, relative, base=None):
        path = (base or self.cell) / relative
        return json.loads(path.read_bytes()) if path.exists() else None


# --------------------------------------------------------------------------- exact re-score
SCORER_MODULES = ("rl_agent/splitfusion_numeric_gt_integration_v1/contract.py", "rl_agent/splitfusion_numeric_gt_integration_v2/ready.py",
                  "rl_agent/splitfusion_numeric_gt_integration_v2/scoring.py", "rl_agent/splitfusion_exact_gt_service_proto_v2/scorer_core.py",
                  "rl_agent/splitfusion_exact_gt_service_proto_v2/deferred.py",
                  "rl_agent/splitfusion_hybrid_sac_live_route_b_v2/run4_live_wire_v2.py",
                  "rl_agent/splitfusion_minimal_quality_scorer_v1/scorer.py",
                  "rl_agent/splitfusion_minimal_quality_scorer_v1/shared_admission.py")


def scorer_sources_identical():
    return {m: sha(MAIN / m) == sha(WT / m) for m in SCORER_MODULES}


def rescore(directory):
    from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
    from rl_agent.splitfusion_numeric_gt_integration_v2 import ready as R, scoring as Q
    from rl_agent.splitfusion_exact_gt_service_proto_v2.scorer_core import expected_from_identity, join_identity
    from rl_agent.splitfusion_exact_gt_service_proto_v2.deferred import validate_anchor_provenance
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
    from rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid import quality
    spec = W.load_run4_quality_spec(MAIN)
    metadata = C._parse((directory / "target_metadata.json").read_bytes())
    score = json.loads((directory / "numerical_score.json").read_bytes())
    identity = C.canonical({k: metadata[k] for k in ("capture_identity", "action_identity", "plan_sha256", "run_binding",
                                                    "producer_binding", "frame_context", "prediction_contract")})
    target_doc, expected = expected_from_identity(identity)
    target = R._admit(metadata, (directory / "segmentation.u8").read_bytes(), (directory / "object_ignore.u8").read_bytes(),
                      **expected)
    anchor_path = directory / "anchor_provenance.json"
    if anchor_path.exists():              # anchor: re-derive the join identity from the archived mapping
        join = json.loads(anchor_path.read_bytes())
        mapping = validate_anchor_provenance(join["provenance"], target_identity=target_doc)
        derived, mode = join_identity(join["prediction_identity_as_received"], mapping, target_doc["capture_identity"])
        if not (mode == join["join_mode"] == "ANCHOR_PROVENANCE_V1" and C.canonical(derived) == C.canonical(join["join_identity"])
                and C.canonical(derived) == C.canonical(score["prediction_identity"])):
            return False
    elif C.canonical(score["prediction_identity"]["gt_identity"]) != C.canonical(target_doc["capture_identity"]):
        return False
    labels = np.frombuffer((directory / "predicted_labels.u8").read_bytes(), np.uint8).reshape(720, 1280)
    scorer = Q.NumericalScorer()
    prediction = Q.admit_prediction(scorer, identity=score["prediction_identity"], labels=labels,
                                    records=score["prediction_records"], prediction_contract=metadata["prediction_contract"])
    result = scorer.score(Q.join_admitted_inputs(scorer, target=target, prediction=prediction))
    value = quality.evaluate_exact_quality(spec, result.measurement)
    value = None if value is None else float(value.q_perc)
    return value == score["q_perc"] and C.canonical(result.measurement) == C.canonical(score["measurement"]) \
        and target.semantic_sha256 == score["ready_semantic_sha256"]


# --------------------------------------------------------------------------- forced replay
def forced_replay(arm, decisions_by_frame):
    from rl_agent.splitfusion_run5_live_avo_v1.audit_actor import load_registered_actor
    actor = load_registered_actor()
    _, _, spans = profile(arm.cell / "profiler/ue.profile.jsonl")
    logged = {r["frame_id"]: r for r in load_jsonl(arm.cell / "forced_action/FORCED_ACTIONS.jsonl")}
    mismatches, replayed = [], 0
    for span in spans:
        fid = int(span["frame_id"])
        decision = actor.act_on_vector(tuple(float(v) for v in span["features"]))
        replayed += 1
        got = (decision.mode_id, decision.q_e4, decision.state_sha256, decision.heads_sha256)
        recorded = (span.get("decided_mode_id"), span.get("decided_q_e4"), span.get("state_sha256"), span.get("heads_sha256"))
        p = (logged.get(fid) or {}).get("proposed") or {}
        proposal = (p.get("proposed_mode_id"), p.get("proposed_q_e4"), p.get("state_sha256"), p.get("heads_sha256"))
        executed = (decisions_by_frame.get(fid) or {})
        if got != recorded or got != proposal or (executed.get("mode_id"), executed.get("q_e4")) != (arm.forced["mode_id"], arm.forced["q_e4"]):
            mismatches.append(dict(frame_id=fid, replay=got, span=recorded, logged=proposal,
                                   executed=(executed.get("mode_id"), executed.get("q_e4"))))
    return dict(replayed=replayed, mismatches=len(mismatches), examples=mismatches[:5],
                proposal_distribution=dict(collections.Counter(f"{r['proposed']['proposed_mode_id']}/{r['proposed']['proposed_q_e4']}"
                                                               for r in logged.values() if r.get("proposed"))))


# --------------------------------------------------------------------------- integration checks
def integration(arm, launch_returncode=None, do_rescore=True):
    d = dict(arm=arm.dir.name, forced_action=arm.forced, launch_returncode=launch_returncode)
    if arm.cell is None:
        d.update(checks=dict(cell_present=False), passed=False)
        return d
    qual = arm.j("capture/QUALIFICATION_RESULT.json", arm.dir) or {}
    gates = qual.get("gates") or {}
    analysis = arm.j("capture/PROFILER_ANALYSIS.json", arm.dir) or {}
    semantics = analysis.get("semantics") or {}
    phase6 = ((qual.get("phase6_report_only") or {}).get("gates")) or {}
    threadcap = (qual.get("threadcap") or {}).get("checks") or {}
    service = arm.j("numeric_gt/service/SERVICE_RESULT.json") or {}
    ue_result = arm.j("numeric_gt/EXACT_GT_UE_RESULT.json") or {}
    edge_doc = arm.j("profiler/edge/numeric_runtime/EDGE_EARLY_GT.json") or {}
    state = edge_doc.get("state") or {}
    frames = [f for f in load_jsonl(arm.cell / "numeric_gt/service/FRAMES.jsonl") if f.get("decision")]
    samples = arm.j("numeric_gt/service/QUEUE_SAMPLES.json") or []
    evidence = arm.j("run4_phase6/PHASE6_UE_EVIDENCE.json") or {}
    forced_summary = arm.j("forced_action/FORCED_ACTION_SUMMARY.json") or {}
    restoration = arm.j("RESTORATION_LATEST.json", arm.dir) or {}
    affinity = arm.j("AFFINITY_APPLIED.json", arm.dir) or {}
    audits = qual.get("audits") or {}
    plans = load_jsonl(arm.cell / "forced_action/FORCED_ACTIONS.jsonl")
    decisions_by_frame = {r["frame_id"]: r["executed"] for r in plans if r["kind"] == "POLICY_DECISION"}
    replay = forced_replay(arm, decisions_by_frame)
    records = state.get("records") or []
    emissions = collections.Counter(int(r["frame_id"]) for r in records)
    decision_ids = {f["frame_id"] for f in frames}
    reached = {f["frame_id"] for f in frames if f.get("prediction_received_raw_ns")}
    commits = [f["commit"] for f in frames if f.get("committed")]
    limits = service.get("limits") or {}
    scored = [f for f in frames if f.get("outcome") == "SCORED"]
    rescored = {}
    if do_rescore:
        for f in scored:
            rescored[f["frame_id"]] = rescore(arm.cell / f"numeric_gt/service/archive/frame_{f['frame_id']}/scorer")
    executed_ok = all((r["executed"]["mode_id"], r["executed"]["q_e4"]) == (arm.forced["mode_id"], arm.forced["q_e4"])
                      for r in plans if r["kind"] in ("POLICY_DECISION", "POLICY_HOLD"))
    fallback_ok = all((r["executed"]["mode_id"], r["executed"]["q_e4"]) == (11, 9800) for r in plans if r["kind"] == "FALLBACK")
    once = all(r["actor_calls_this_frame"] == (1 if r["kind"] == "POLICY_DECISION" else 0) for r in plans)
    checks = dict(
        I1_retained_gates=all(gates.get(k) is True for k in ("G1_FRAME_ACCOUNTING", "G2_ACTOR_HASH_UNCHANGED",
            "G3_ZERO_IDENTITY_CONFLICTS", "G4_COVERAGE_REPORTED", "G5_SAMPLER_REPORTED", "G6_NO_PROFILER_INDUCED_FAULT",
            "G7_INSTRUMENTATION_EVIDENCE_VALID", "G9_HOST_RESTORED_CLEAN")),
        I2_g8_components_except_actor_replay=((semantics.get("reward") or {}).get("passed") is True
            and (semantics.get("receipt_marks") or {}).get("passed") is True
            and all(phase6.get(k) is True for k in ("P3_HOLD_TICKET", "P4_ACCOUNTING", "P5_NO_ACTOR_AFTER_REFUSAL"))),
        I3_forced_action=(bool(plans) and executed_ok and fallback_ok and once and replay["mismatches"] == 0
            and replay["replayed"] == len(decisions_by_frame) == len(frames)
            and (forced_summary.get("counters") or {}).get("executed_mismatch") == 0),
        I4_terminal_accounting=(all(f.get("outcome") for f in frames) and set(emissions) == reached <= decision_ids
            and all(v == 1 for v in emissions.values())
            and all(r["q_perc"] is None for r in records if r["kind"] != "DELIVERED_SUCCESS")
            and len(evidence.get("feedback_rows") or []) == len(records)),
        I5_exact_quality_offline=(not do_rescore) or (len(rescored) == len(scored) and all(rescored.values())),
        I6_history=all(a["history_after_sha256"] == b["history_before_sha256"] for a, b in zip(commits, commits[1:])),
        I7_no_pre_action_gt_work=all(
            next((e["raw_ns"] for e in f["events"] if e["name"] == "freeze_enqueued_on_ue_thread"), -1) >= f["action_open_raw_ns"]
            for f in frames if f.get("outcome") != "NO_GT_UE_FREEZE_QUEUE_FULL"),
        I8_bounded=(len(service.get("refusals") or []) <= 5 and all(
            s["prep_retained"] <= limits.get("prep_jobs", 0) and s["scorer_slots"] <= limits.get("scorer_jobs", 0)
            and s["freeze_queue"] <= limits.get("freeze_queue", 0) for s in samples)
            and (state.get("evaluator_counters") or {}).get("queue_overflow", 1) == 0),
        I9_cleanup=(not service.get("errors") and not ue_result.get("errors") and ue_result.get("service_returncode") == 0
            and edge_doc.get("error") is None and restoration.get("passed") is True
            and launch_returncode in (None, 0) and not qual.get("leftover_processes")),
        I11_anchor_provenance=(arm.forced.get("action_id") is None) or all(
            (arm.cell / f"numeric_gt/service/archive/frame_{f['frame_id']}/scorer/anchor_provenance.json").exists()
            and (arm.cell / f"numeric_gt/service/archive/frame_{f['frame_id']}/producer/anchor_provenance.json").exists()
            for f in scored),
        I10_retained_settings=(threadcap.get("E1_UE_SETTINGS_RECORDED") is True and threadcap.get("E2_UE_CAP_STATE_AS_REGISTERED") is True
            and affinity.get("passed") is True and (audits.get("before") or {}).get("verdict") == "PASS"
            and (audits.get("after") or {}).get("verdict") == "PASS"))
    d.update(checks=checks, passed=all(checks.values()), retained_verdict=qual.get("verdict"), retained_gates=gates,
             g1_detail=qual.get("g1_detail"), g8_retained=gates.get("G8_REWARD_TIMEOUT_SEMANTICS"),
             actor_replay_retained=(semantics.get("actor_replay") or {}).get("passed"), forced_replay=replay,
             nongated_threadcap=dict(E3=threadcap.get("E3_SCOPE_OTHER_PROCESSES_BASELINE"),
                                     E5=threadcap.get("E5_EDGE_ORDERING_AS_REGISTERED")),
             decisions=len(frames), scored=len(scored), rescored_exact=sum(rescored.values()) if rescored else None,
             outcomes=service.get("outcomes"), refusals=service.get("refusals"),
             forced_counters=forced_summary.get("counters"))
    return d


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--launch-returncode", type=int, default=None)
    args = parser.parse_args()
    result = integration(Arm(args.arm), args.launch_returncode)
    print(json.dumps(result, indent=1, default=str))
    sys.exit(0 if result["passed"] else 3)
