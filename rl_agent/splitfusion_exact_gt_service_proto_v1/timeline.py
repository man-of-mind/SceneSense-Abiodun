"""Recorded v11 arrival spacing for the 49 frames that also have sealed sources and predictions.

All offsets are from that frame's recorded UE action-open (CLOCK_MONOTONIC_RAW). The v11 capture
already used the corrected perception-input function (recorded ue.input_7ch ~15 ms).
Holds/fallbacks are kept as UE-load frames only; they never carry GT or feedback.
"""
import collections
import hashlib
import json

from . import parity_common as P

CELL = P.K.CELL
EVIDENCE = CELL / "run4_phase6/PHASE6_UE_EVIDENCE.json"
UE_PROFILE = CELL / "profiler/ue.profile.jsonl"
EDGE_PROFILE = CELL / "profiler/edge/edge.profile.jsonl"


def _profile(path):
    spans, marks = collections.defaultdict(dict), collections.defaultdict(dict)
    with open(path) as stream:
        for line in stream:
            d = json.loads(line)
            f = d.get("frame_id")
            if f is None:
                continue
            if d.get("r") == "span":
                spans[int(f)].setdefault(d["stage"], (d["start_raw_ns"], d["end_raw_ns"]))
            elif d.get("r") == "mark":
                marks[int(f)].setdefault(d["stage"], d["t_raw_ns"])
    return spans, marks


def recorded(case_frame_ids):
    decisions = [d for d in json.loads(EVIDENCE.read_bytes())["decisions"] if d.get("action_open")]   # unpaced records skipped
    us, _ = _profile(UE_PROFILE)
    es, em = _profile(EDGE_PROFILE)
    by_frame = {d["frame_id"]: d for d in decisions}
    start = min(d["action_open"]["ns"] for d in decisions if d["frame_id"] in set(case_frame_ids))
    events = []
    for d in sorted(decisions, key=lambda x: x["action_open"]["ns"]):
        ao = d["action_open"]["ns"]
        if ao < start:
            continue
        stages = d["stages"]
        row = dict(frame_id=d["frame_id"], kind=d["kind"], open_ms=(ao - start) / 1e6,
                   input_start_ms=(stages["input_7ch_start_raw_ns"] - ao) / 1e6,
                   send_ms=(stages["last_packet_send_raw_ns"] - ao) / 1e6)
        f = d["frame_id"]
        if d["kind"] == "POLICY_DECISION":
            C = P.C
            C.require(f in case_frame_ids, "recorded decision without sealed source")
            row.update(plan_end_ms=(us[f]["ue.plan"][1] - ao) / 1e6,
                       recorded_input_ms=(us[f]["ue.input_7ch"][1] - us[f]["ue.input_7ch"][0]) / 1e6,
                       compute_end_after_send_ms=(es[f]["edge.compute"][1] - stages["last_packet_send_raw_ns"]) / 1e6,
                       evaluator_submit_after_send_ms=(em[f]["edge.evaluator_submit"] - stages["last_packet_send_raw_ns"]) / 1e6)
        events.append(row)
    decided = [e["frame_id"] for e in events if e["kind"] == "POLICY_DECISION"]
    P.C.require(decided == list(case_frame_ids), "recorded decision order differs from sealed cases")
    inputs = {str(p.relative_to(P.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (EVIDENCE, UE_PROFILE, EDGE_PROFILE)}
    return events, inputs


def compressed(events, spacing_ms):
    """Same per-frame offsets, decisions every ``spacing_ms`` with no holds (stress variant)."""
    out = []
    for k, e in enumerate([e for e in events if e["kind"] == "POLICY_DECISION"]):
        out.append(dict(e, open_ms=k * spacing_ms))
    return out
