"""Pure parsing of one worker GC trace (shared by the offline rehearsal and the live analysis).

Decision index of an op = number of FREEZE requests seen up to and including it (FREEZE opens a decision in the prep
worker). Gen2 collections are classified by enclosing op: FREEZE/BIND (feedback path), other request ops, or between
requests (op None).
"""
import numpy as np

FEEDBACK_OPS = ("FREEZE", "BIND")


def decision_index(ops):
    out, k = {}, 0
    for rec in ops:
        if rec["op"] == "FREEZE":
            k += 1
        out[rec["seq"]] = k
    return out


def summarize(trace, warmup):
    ops = trace["ops"]
    dec = decision_index(ops)
    by_seq = {r["seq"]: r for r in ops}
    gen2 = trace["gen2"]

    def ms(ns):
        return ns / 1e6
    feedback = [g for g in gen2 if g["op"] in FEEDBACK_OPS]
    in_fb_warm = [g for g in feedback if dec.get(g["op_seq"], 0) <= warmup]
    in_fb_late = [g for g in feedback if dec.get(g["op_seq"], 0) > warmup]
    other_req = [g for g in gen2 if g["op"] is not None and g["op"] not in FEEDBACK_OPS]
    between = [g for g in gen2 if g["op"] is None]
    freeze_ms = np.array([ms(r["end_raw_ns"] - r["begin_raw_ns"]) for r in ops if r["op"] == "FREEZE"])
    rss = np.array([r["rss1"] for r in ops], dtype=float)
    events = trace.get("policy_events") or []
    idle = [e for e in events if e.get("kind") == "idle_full_collect"]
    return dict(
        decisions=max(dec.values(), default=0), ops=len(ops), dropped=trace.get("dropped"),
        gen2_total=len(gen2), gen2_feedback_warmup=len(in_fb_warm), gen2_feedback_after_warmup=len(in_fb_late),
        gen2_other_requests=len(other_req), gen2_between_requests=len(between),
        gen2_feedback_after_warmup_ms=[round(ms(g["wall_ns"]), 3) for g in in_fb_late],
        gen2_all_ms=[round(ms(g["wall_ns"]), 3) for g in gen2],
        freeze_ms=freeze_ms.tolist(),
        rss_first=float(rss[0]) if len(rss) else None, rss_last=float(rss[-1]) if len(rss) else None,
        rss_max=float(rss.max()) if len(rss) else None,
        idle_collections=len(idle), idle_wall_ms=[e["wall_ms"] for e in idle],
        idle_request_arrivals=sum(bool(e.get("request_arrived_during")) for e in idle),
        skipped_pending=sum(e.get("kind") == "skipped_pending" for e in events),
        safety_stop=trace.get("safety_stop"), policy_installed=trace.get("policy_installed"),
        restored_equals_initial=trace.get("restored_equals_initial"), op_by_seq=None if not by_seq else len(by_seq))
