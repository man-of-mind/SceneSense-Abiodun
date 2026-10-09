"""Analysis of the ONE action-0 noAE baseline capture (read-only on the attempt; create-only outputs).

Per-decision rows come from the retained full Run-5 row builder (``splitfusion_run5_full_live_analysis_v1.report.rows``,
itself ``fixed_action_screen_analysis_v2.report.rows_for``): every boundary is ms after the decision's action open.
Registered decision classes (ACCEPTANCE.json, before any frame):
  TIMELY                  result feedback (scored: DELIVERED_SUCCESS or quality-undefined) received at the UE <= 170 ms incl.
  RECEIVED_LATE           result feedback received > 170 ms
  TERMINAL_WITHOUT_RESULT a receipt exists but the edge record is a terminal without a result (e.g. STALE)
  MISSING                 nothing received within the retained bounded observation/drain window
Latency rule: a timeout timestamp is never a feedback latency. Received-only quantiles are CONDITIONAL ON RECEIPT whenever
any decision lacks a result; an unconditional P_k is reported only when identifiable (rank ceil(k n / 100) <= results
received), else UNIDENTIFIABLE. P99 from < 100 results is labelled SPARSE.
Comparator: the deployed agent's FAVORABLE_STABLE capture (retention_v1 validation_01, 494 decisions) - DIFFERENT SESSION,
different n, descriptive only. Full-local is not part of the split-payload comparison.

  python3 -m rl_agent.splitfusion_noae_baseline_live_v1_analysis.analyze --campaign experiments/splitfusion_noae_baseline_live_v1/capture_01 \
      --out experiments/splitfusion_noae_baseline_live_v1/analysis_01
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from rl_agent.splitfusion_fixed_action_screen_analysis_v2.analyze import wilson
from rl_agent.splitfusion_run5_full_live_analysis_v1 import report as RP

MAIN = Path(__file__).resolve().parents[2]
WT = MAIN.parent / "abiodun_run4_single_gpu_profiler_v1"
ROOT = WT / "rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261007_noae_baseline_live_v1/baseline"
AGENT = MAIN / "experiments/splitfusion_run5_full_live_retention_v1/validation_01_analysis"
AGENT_REL = "validation/DECISIONS_validation_b1000.jsonl"
DEADLINE = 170.0
FALLBACK = (11, 9800, 71)
BOUNDARIES = (  # (label, start key, end key); every value ms after action open; differences only when both present
    ("action open -> encode end", None, "ue_encode_end"),
    ("encode (UE codec total)", "ue_encode_start", "ue_encode_end"),
    ("first -> last datagram sent", "ue_first_send", "ue_last_send"),
    ("uplink: last send -> edge reassembly", "ue_last_send", "edge_reassembly_complete"),
    ("edge: reassembly -> prediction", "edge_reassembly_complete", "edge_compute_end"),
    ("GT published (after action open)", None, "gt_published"),
    ("return: scoring end -> UE receipt", "scoring_worker_end", "ue_receipt"),
    ("action open -> UE receipt", None, "ue_receipt"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantiles(values):
    v = np.sort(np.asarray([x for x in values if x is not None], dtype=float))
    if len(v) == 0:
        return dict(n=0)
    return dict(n=int(len(v)), p50=float(np.percentile(v, 50)), p90=float(np.percentile(v, 90)),
                p99=float(np.percentile(v, 99)), min=float(v[0]), max=float(v[-1]), method="numpy linear",
                p99_label="SPARSE (n<100)" if len(v) < 100 else "")


def unconditional(received, n):
    """Order-statistic quantile over ALL n decisions with non-results at +inf (beyond the observation window)."""
    v = sorted(received)
    out = {}
    for k in (50, 90, 99):
        rank = math.ceil(k * n / 100)
        out[f"p{k}"] = v[rank - 1] if rank <= len(v) else "UNIDENTIFIABLE (beyond the observation window)"
        if k == 99 and rank <= len(v) and n < 100:
            out["p99_label"] = "SPARSE (n<100)"
    out.update(n_decisions=n, results_received=len(v), method="order statistic rank ceil(k n/100); non-results = +inf")
    return out


def is_result(r):
    return r.get("feedback_kind") == "DELIVERED_SUCCESS" or (
        r.get("feedback_kind") == "EVALUATOR_FAULT" and r.get("feedback_reason") == "QUALITY_UNDEFINED_NO_ELIGIBLE_GT")


def classify(r):
    """Result receipt first (a scored result: DELIVERED_SUCCESS, or quality-undefined EVALUATOR_FAULT /
    QUALITY_UNDEFINED_NO_ELIGIBLE_GT); else any received terminal; else missing."""
    if r["ue_receipt"] is not None and is_result(r):
        return "TIMELY" if r["ue_receipt"] <= DEADLINE else "RECEIVED_LATE"
    if r["ue_receipt"] is not None or r.get("terminal_receipt") is not None:
        return "TERMINAL_WITHOUT_RESULT"
    return "MISSING"


def stage_table(rows):
    table = {}
    for label, start, end in BOUNDARIES:
        vals = [r[end] - (0.0 if start is None else r[start]) for r in rows
                if r.get(end) is not None and (start is None or r.get(start) is not None)]
        table[label] = dict(quantiles(vals), available=len(vals), of=len(rows))
    return table


def baseline_rows():
    arm, rows, evidence = RP.rows(ROOT)
    forced = {r["frame_id"]: r for r in map(json.loads, open(arm.cell / "forced_action/FORCED_ACTIONS.jsonl"))}
    opens = {f["frame_id"]: f["action_open_raw_ns"] for f in map(json.loads, open(arm.cell / "numeric_gt/service/FRAMES.jsonl"))
             if f.get("decision")}
    terminals = collections.defaultdict(list)
    for t in evidence["terminal_rows"]:
        terminals[int(t["frame_id"])].append(t)
    for r in rows:
        ts = [t for t in terminals.get(r["frame_id"], []) if (t.get("receipt") or {}).get("ns") is not None]
        first = min(ts, key=lambda t: t["receipt"]["ns"]) if ts else None
        r["terminal_receipt"] = None if first is None else (first["receipt"]["ns"] - opens[r["frame_id"]]) / 1e6
        r["terminal_outcome"] = None if first is None else first.get("outcome")
        r["terminal_stage"] = None if first is None else first.get("stage")
        r["terminal_class"] = None if first is None else first.get("class")
        fr = forced.get(r["frame_id"]) or {}
        ex = fr.get("executed") or {}
        r["plan_kind"] = fr.get("kind")
        r["executed_identity"] = (ex.get("mode_id"), ex.get("q_e4"), ex.get("action_id"))
        r["is_baseline"] = fr.get("kind") == "POLICY_DECISION" and r["executed_identity"] == (0, 0, 0)
        r["class"] = classify(r)
    return arm, rows, evidence, list(forced.values())


def agent_rows():
    path = AGENT / AGENT_REL
    assert sha(path) == json.loads((AGENT / "OUTPUT_SHA256.json").read_bytes())[AGENT_REL]
    rows = [json.loads(line) for line in open(path)]
    assert len(rows) == 494 and all(r["network_profile_id"] == "FAVORABLE_STABLE" for r in rows)
    for r in rows:
        r["class"] = classify(r)
    return rows, sha(path)


def summarize(rows, plans=None):
    n = len(rows)
    classes = collections.Counter(r["class"] for r in rows)
    results = [r["ue_receipt"] for r in rows if r["class"] in ("TIMELY", "RECEIVED_LATE")]
    any_receipt = [r["ue_receipt"] for r in rows if r["ue_receipt"] is not None]
    complete = all(r["class"] in ("TIMELY", "RECEIVED_LATE") for r in rows)
    payload = [r["payload_bytes"] for r in rows if r["payload_bytes"]]
    out = dict(decisions=n, classes={k: classes.get(k, 0) for k in ("TIMELY", "RECEIVED_LATE", "TERMINAL_WITHOUT_RESULT", "MISSING")},
               fractions={k: classes.get(k, 0) / n for k in ("TIMELY", "RECEIVED_LATE", "TERMINAL_WITHOUT_RESULT", "MISSING")},
               timely_wilson=wilson(classes.get("TIMELY", 0), n),
               result_delivery_fraction=len(results) / n, result_delivery_wilson=wilson(len(results), n),
               feedback_latency_ms=dict(
                   label="UNCONDITIONAL (every decision received a result)" if complete else "CONDITIONAL ON RESULT RECEIPT",
                   results=quantiles(results), unconditional=unconditional(results, n),
                   any_receipt_including_terminals=quantiles(any_receipt)),
               terminal_without_result=dict(
                   receipt_ms_after_action_open=quantiles([r.get("terminal_receipt") for r in rows
                                                           if r["class"] == "TERMINAL_WITHOUT_RESULT"]),
                   outcomes=dict(collections.Counter(str(r.get("terminal_outcome")) for r in rows
                                                     if r["class"] == "TERMINAL_WITHOUT_RESULT")),
                   note="terminal receipt times are NOT feedback latencies (no result was delivered)"),
               payload_application_frame_bytes=quantiles(payload),
               datagrams_per_frame=quantiles([r["payload_chunks"] for r in rows]),
               datagrams_at_edge=quantiles([r["datagrams_at_edge"] for r in rows]),
               reassembled_at_edge=sum(r["edge_reassembly_complete"] is not None for r in rows),
               prediction_at_edge=sum(r["edge_compute_end"] is not None for r in rows),
               scored=sum(r["service_outcome"] == "SCORED" for r in rows),
               service_outcomes=dict(collections.Counter(r["service_outcome"] for r in rows)),
               feedback_kinds=dict(collections.Counter(str(r.get("feedback_kind")) for r in rows)),
               controller_resolutions=dict(collections.Counter(str(r.get("controller_resolution")) for r in rows)),
               boundaries_ms=stage_table(rows),
               quality_scored=dict(vehicle=RP.pooled([r for r in rows if r["service_outcome"] == "SCORED"], "vehicle"),
                                   person=RP.pooled([r for r in rows if r["service_outcome"] == "SCORED"], "person")))
    return out


def run(campaign, out):
    campaign, out = Path(campaign), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    verdict = json.loads((campaign / "VERDICT_baseline.json").read_bytes())
    arm, rows, evidence, plans = baseline_rows()
    kinds = collections.Counter(p["kind"] for p in plans)
    fallback_plans = [p for p in plans if p["kind"] == "FALLBACK"]
    baseline = [r for r in rows if r["is_baseline"]]
    not_baseline = [r for r in rows if not r["is_baseline"]]
    opportunities = evidence["opportunities"]
    agent, agent_sha = agent_rows()
    integ = verdict.get("integration") or {}
    summary = dict(
        schema="splitfusion.noae_baseline_live_v1.analysis.v1", verdict=verdict.get("verdict"), reasons=verdict.get("reasons"),
        conditions=verdict.get("conditions"), reported_outcomes=verdict.get("reported_outcomes"),
        plans=dict(kinds), baseline_decisions=len(baseline), holds=kinds.get("POLICY_HOLD", 0),
        fallbacks=len(fallback_plans),
        fallback_identities=sorted({(p["executed"]["mode_id"], p["executed"]["q_e4"], p["executed"]["action_id"]) for p in fallback_plans}),
        fallback_reasons=dict(collections.Counter(x.split(":")[0] for d in evidence["decisions"] if d["kind"] == "FALLBACK"
                                                  for x in d.get("fallback_reasons", []))),
        decision_rows_not_baseline=len(not_baseline),
        admission=dict(opportunities=len(opportunities), admitted=sum(bool(o["admitted"]) for o in opportunities)),
        forced_counters=(integ.get("forced_counters")), forced_replay=integ.get("forced_replay"),
        rescore=dict(scored=integ.get("scored"), rescored_exact=integ.get("rescored_exact")),
        baseline=summarize(baseline),
        agent_comparator=dict(source=str((AGENT / AGENT_REL).relative_to(MAIN)), sha256=agent_sha,
                              label="DEPLOYED AGENT, DIFFERENT SESSION (2026-10-06), descriptive only", **summarize(agent)))
    with open(out / "DECISIONS_baseline_b0100.jsonl", "x") as stream:
        for r in rows:
            stream.write(json.dumps(r, default=str, sort_keys=True) + "\n")
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    (out / "OUTPUT_SHA256.json").write_text(json.dumps({p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}, indent=1))
    print(json.dumps(dict(verdict=summary["verdict"], baseline_decisions=len(baseline), plans=dict(kinds),
                          classes=summary["baseline"]["classes"]), default=str))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.campaign, args.out)
