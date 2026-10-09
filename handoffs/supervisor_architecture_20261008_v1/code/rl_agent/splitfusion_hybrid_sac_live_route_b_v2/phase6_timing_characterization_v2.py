#!/usr/bin/env python3
"""Addendum 11: offline timing characterization of one 300-frame Phase-6 cell.

Reads only the preserved evidence of one cell attempt. It does not rescore,
re-reward or re-time anything. Two independent measurements are reported:

* **Frozen policy deadline**: UE feedback receipt - action-open <= 170 ms
  (``CLOCK_MONOTONIC_RAW``). This is the unchanged Run-4 contract.
* **System KPI**: UE feedback receipt - sensor capture <= 200 ms. This is a
  report-only KPI that never alters any reward or timeout.

It also reports whether each reward arrived before the **actual next eligible
decision**. That is the plan instant (``si_p40_end_raw_ns``) of the first frame
planned after the ticket transmitted ``k_min = 2`` tensors, as recorded. No
sensor period is assumed.

Clocks: UE raw and UE wall are bridged per frame by the capture pair
(``capture`` wall ns stamped in the collector's RGB callback microseconds
after ``rgb_receipt_raw_ns``). Edge, map and evaluator stamps are host wall
clock (same host). They are converted with the median per-frame bridge, and
the bridge spread is reported.

Every sent frame is included, uncensored. A missing stage stays ``None`` and is
counted, never dropped. Importing this module performs no I/O.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

SCHEMA = "scenesense.run4_live_v2.phase6_timing_characterization.v1"
POLICY_DEADLINE_MS = 170.0
SYSTEM_KPI_MS = 200.0
K_MIN = 2
PERCENTILES = (50, 90, 95, 99)


def nearest_rank(values: Sequence[float], p: float) -> Optional[float]:
    vals = sorted(values)
    if not vals:
        return None
    rank = max(1, int(-(-p * len(vals) // 100)))       # ceil(p/100 * n)
    return vals[min(rank, len(vals)) - 1]


def summary(values: Iterable[Optional[float]]) -> dict[str, Any]:
    values = list(values)
    present = [float(v) for v in values if v is not None]
    out = {"n": len(present), "missing": len(values) - len(present)}
    for p in PERCENTILES:
        out[f"p{p}"] = nearest_rank(present, p)
    out["max"] = max(present) if present else None
    out["method"] = "nearest-rank"
    return out


def fraction(flags: Sequence[Optional[bool]]) -> dict[str, Any]:
    n = len(flags)
    yes = sum(1 for f in flags if f is True)
    return {"count": yes, "n": n, "fraction": (yes / n) if n else None,
            "undetermined": sum(1 for f in flags if f is None)}


def _ms(a: Optional[int], b: Optional[int]) -> Optional[float]:
    return None if a is None or b is None else (int(b) - int(a)) / 1e6


def _wall_s_to_ns(value: Any) -> Optional[int]:
    return None if value in (None, "") else int(round(float(value) * 1e9))


def overlap_ms(a: tuple, intervals: Sequence[tuple]) -> float:
    s, e = a
    total = 0
    for s2, e2 in intervals:
        total += max(0, min(e, e2) - max(s, s2))
    return total / 1e6


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def analyze(cell: Path) -> dict[str, Any]:
    cell = Path(cell)
    ue = json.loads((cell / "run4_phase6/PHASE6_UE_EVIDENCE.json").read_text(encoding="utf-8"))
    edge_path = cell / "run4_phase6/edge_report.json"
    edge = json.loads(edge_path.read_text(encoding="utf-8")) if edge_path.is_file() else {}
    terminals = _read_csv(cell / "map_feedback.csv")
    ingest = {int(r["frame_id"]): r for r in _read_csv(cell / "direct_edge_map/direct_map_ingest.csv")}
    per_frame = {int(r["frame_id"]): r for r in _read_csv(cell / "per_frame_metrics.csv")}
    decisions = sorted(ue["decisions"], key=lambda r: r["frame_id"])
    tickets = {int(t["frame_id"]): t for t in ue["gt_objects"]["tickets"]}
    transmitted = {int(x["frame_id"]): x["run4_identity"] for x in ue["transmitted_identities"]}
    evaluations = defaultdict(list)
    for e in edge.get("evaluations", []):
        evaluations[int(e["frame_id"])].append(e)
    feedback = defaultdict(list)
    for f in ue["feedback_rows"]:
        feedback[int(f["frame_id"])].append(f)

    # per-frame raw<->wall bridge from the capture pair
    offsets = [int(r["capture"]["ns"]) - int(r["stages"]["rgb_receipt_raw_ns"])
               for r in decisions if r.get("stages", {}).get("rgb_receipt_raw_ns")
               and r.get("capture", {}).get("ns")]
    bridge = int(statistics.median(offsets)) if offsets else None

    def wall_to_raw(ns: Optional[int]) -> Optional[int]:
        return None if ns is None or bridge is None else int(ns) - bridge

    frames, decision_frames = [], {}
    reuse = Counter((transmitted[f].get("session_uuid"), transmitted[f].get("decision_seq"))
                    for f in transmitted
                    if transmitted[f].get("frame_kind") in ("POLICY_DECISION", "POLICY_HOLD"))
    for rec in decisions:
        f = int(rec["frame_id"])
        st = rec.get("stages", {})
        capture_raw = st.get("rgb_receipt_raw_ns")
        ao = (rec.get("action_open") or {}).get("ns")
        plan = st.get("si_p40_end_raw_ns")
        anchor = ao if ao is not None else plan
        ing = ingest.get(f, {})
        t = tickets.get(f, {})
        ident = transmitted.get(f, {})
        pf = per_frame.get(f, {})
        row = {
            "frame_id": f, "kind": rec.get("kind"), "mode_id": rec.get("mode_id"),
            "q_e4": rec.get("q_e4"), "tensor_seq": rec.get("tensor_seq"),
            "decision_seq": ident.get("decision_seq"), "ticket_seq": ident.get("ticket_seq"),
            "reward_requested": ident.get("reward_requested"),
            "action_reuse_count": (reuse.get((ident.get("session_uuid"), ident.get("decision_seq")))
                                   if rec.get("kind") in ("POLICY_DECISION", "POLICY_HOLD") else None),
            "payload_bytes": int(pf["payload_bytes"]) if pf.get("payload_bytes") else None,
            "datagrams": int(pf["payload_chunks"]) if pf.get("payload_chunks") else None,
            "sent": f in transmitted,
            "anchor": "action_open" if ao is not None else "plan_instant",
            "capture_to_anchor_ms": _ms(capture_raw, anchor),
            "input_7ch_ms": _ms(st.get("input_7ch_start_raw_ns"), st.get("front_start_raw_ns")),
            "front_ms": _ms(st.get("front_start_raw_ns"), st.get("front_end_raw_ns")),
            "anchor_to_first_datagram_ms": _ms(anchor, st.get("first_packet_send_raw_ns")),
            "anchor_to_last_datagram_ms": _ms(anchor, st.get("last_packet_send_raw_ns")),
        }
        reasm = wall_to_raw(_wall_s_to_ns(ing.get("edge_reassembly_complete_wall_s")))
        cstart = wall_to_raw(_wall_s_to_ns(ing.get("edge_compute_start_wall_s")))
        pstart = wall_to_raw(_wall_s_to_ns(ing.get("edge_publish_start_wall_s")))
        install = wall_to_raw(_wall_s_to_ns(ing.get("map_install_at")))
        row.update({
            "last_datagram_to_uplink_complete_ms": _ms(st.get("last_packet_send_raw_ns"), reasm),
            "edge_queue_ms": _ms(reasm, cstart), "edge_compute_ms": _ms(cstart, pstart),
            "anchor_to_map_install_ms": _ms(anchor, install),
            "capture_to_map_install_ms": _ms(capture_raw, install),
            "map_outcome": ing.get("outcome"),
            "gt_ready_after_anchor_ms": _ms(anchor, wall_to_raw(t.get("objects_write_end_wall_ns"))),
            "gt_queue_class": t.get("queue_class"), "gt_low_skipped": t.get("low_skipped"),
            "_intervals": {"front": (st.get("front_start_raw_ns"), st.get("front_end_raw_ns")),
                           "edge": (cstart, pstart)},
            "_raw": {"capture": capture_raw, "anchor": anchor, "plan": plan, "ao": ao},
        })
        ev = evaluations.get(f, [])
        if ev:
            e = ev[0]
            row.update(evaluator_enqueue_after_anchor_ms=_ms(anchor, wall_to_raw(e.get("enqueued_wall_ns"))),
                       evaluator_ms=_ms(e.get("enqueued_wall_ns"), e.get("emit_wall_ns")),
                       feedback_emit_after_anchor_ms=_ms(anchor, wall_to_raw(e.get("emit_wall_ns"))),
                       evaluator_kind=e.get("kind"))
        fbs = feedback.get(f, [])
        if fbs:
            receipt = int(fbs[0]["receipt"]["ns"])
            row.update(feedback_class=fbs[0]["class"], feedback_kind=fbs[0]["kind"],
                       q_perc=fbs[0]["q_perc"],
                       action_open_to_feedback_ms=_ms(ao, receipt),
                       capture_to_feedback_ms=_ms(capture_raw, receipt))
            row["_raw"]["receipt"] = receipt
        frames.append(row)
        if rec.get("kind") == "POLICY_DECISION":
            decision_frames[f] = row

    # next eligible decision: first frame planned after k_min tensors of the ticket
    by_frame = {r["frame_id"]: r for r in frames}
    order = [r["frame_id"] for r in frames]
    resolutions = [r for r in ue["resolutions"]]
    res_by_seq = {(r["identity"]["session_uuid"], r["identity"]["decision_seq"]): r
                  for r in resolutions if r.get("identity")}
    tickets_out = []
    for f, row in sorted(decision_frames.items()):
        seq = row["decision_seq"]
        session = transmitted.get(f, {}).get("session_uuid")
        same = [g for g in order if by_frame[g]["decision_seq"] == seq
                and by_frame[g]["kind"] in ("POLICY_DECISION", "POLICY_HOLD")
                and transmitted.get(g, {}).get("session_uuid") == session]
        after_kmin = same[K_MIN - 1] if len(same) >= K_MIN else None
        eligible = next((g for g in order if after_kmin is not None and g > after_kmin), None)
        eligible_plan = by_frame[eligible]["_raw"]["plan"] if eligible is not None else None
        receipt = row["_raw"].get("receipt")
        res = res_by_seq.get((session, seq))
        tickets_out.append({
            "frame_id": f, "decision_seq": seq, "mode_id": row["mode_id"], "q_e4": row["q_e4"],
            "payload_bytes": row["payload_bytes"], "tensors_carried": len(same),
            "next_eligible_frame": eligible,
            "next_eligible_kind": by_frame[eligible]["kind"] if eligible is not None else None,
            "action_open_to_next_eligible_ms": _ms(row["_raw"]["ao"], eligible_plan),
            "action_open_to_feedback_ms": row.get("action_open_to_feedback_ms"),
            "capture_to_feedback_ms": row.get("capture_to_feedback_ms"),
            "feedback_before_next_eligible": (None if eligible_plan is None else
                                              (receipt is not None and receipt < eligible_plan)),
            "feedback_class": row.get("feedback_class"), "q_perc": row.get("q_perc"),
            "resolution_terminal": None if res is None else res.get("terminal"),
            "learning_included": None if res is None else res.get("learning_included"),
            "reward": None if res is None else res.get("reward"),
        })

    # GPU overlap: UE front vs edge tail vs CARLA render (tick -> RGB receipt)
    trace = ue.get("carla_trace") or {}
    rgb = {}
    for fr, raw in trace.get("rgb", []):
        rgb.setdefault(int(fr), int(raw))
    carla = sorted((int(raw), rgb[int(fr)]) for fr, _t, raw in trace.get("ticks", [])
                   if int(fr) in rgb and rgb[int(fr)] > int(raw))
    fronts = [r["_intervals"]["front"] for r in frames if None not in r["_intervals"]["front"]]
    edges = [r["_intervals"]["edge"] for r in frames if None not in r["_intervals"]["edge"]]
    gpu = []
    for r in frames:
        fs, fe = r["_intervals"]["front"]
        es, ee = r["_intervals"]["edge"]
        entry = {"frame_id": r["frame_id"], "kind": r["kind"]}
        if fs is not None and fe is not None:
            entry["front_ms"] = (fe - fs) / 1e6
            entry["front_overlap_edge_ms"] = overlap_ms((fs, fe), edges)
            entry["front_overlap_carla_ms"] = overlap_ms((fs, fe), carla)
        if es is not None and ee is not None:
            entry["edge_ms"] = (ee - es) / 1e6
            entry["edge_overlap_front_ms"] = overlap_ms((es, ee), fronts)
            entry["edge_overlap_carla_ms"] = overlap_ms((es, ee), carla)
        gpu.append(entry)

    reward_rows = tickets_out
    sent = [r for r in frames if r["sent"]]
    installs = [r["capture_to_map_install_ms"] for r in sent]
    out = {
        "schema": SCHEMA, "cell": str(cell), "claim": "TIMING_CHARACTERIZATION_NOT_QUALIFICATION",
        "clock_bridge": {"method": "median(capture wall - rgb_receipt raw)", "n": len(offsets),
                         "spread_us": ((max(offsets) - min(offsets)) / 1e3) if offsets else None},
        "counts": {"planned_frames": len(frames), "sent_frames": len(sent),
                   "by_kind": dict(Counter(r["kind"] for r in frames)),
                   "reward_tickets": len(reward_rows)},
        # a ticket without any feedback is a measured miss, never undetermined
        "policy_deadline_action_open_to_feedback_le_170": fraction(
            [t["action_open_to_feedback_ms"] is not None
             and t["action_open_to_feedback_ms"] <= POLICY_DEADLINE_MS for t in reward_rows]),
        "system_kpi_capture_to_feedback_le_200": fraction(
            [t["capture_to_feedback_ms"] is not None
             and t["capture_to_feedback_ms"] <= SYSTEM_KPI_MS for t in reward_rows]),
        "feedback_before_next_eligible_decision": fraction(
            [t["feedback_before_next_eligible"] for t in reward_rows]),
        "distributions_ms": {
            "action_open_to_feedback": summary([t["action_open_to_feedback_ms"] for t in reward_rows]),
            "capture_to_feedback": summary([t["capture_to_feedback_ms"] for t in reward_rows]),
            "action_open_to_next_eligible": summary([t["action_open_to_next_eligible_ms"] for t in reward_rows]),
            "capture_to_anchor": summary([r["capture_to_anchor_ms"] for r in sent]),
            "anchor_to_first_datagram": summary([r["anchor_to_first_datagram_ms"] for r in sent]),
            "anchor_to_last_datagram": summary([r["anchor_to_last_datagram_ms"] for r in sent]),
            "input_7ch": summary([r["input_7ch_ms"] for r in sent]),
            "front": summary([r["front_ms"] for r in sent]),
            "last_datagram_to_uplink_complete": summary([r["last_datagram_to_uplink_complete_ms"] for r in sent]),
            "edge_queue": summary([r["edge_queue_ms"] for r in sent]),
            "edge_compute": summary([r["edge_compute_ms"] for r in sent]),
            "capture_to_map_install": summary(installs),
            "anchor_to_map_install": summary([r["anchor_to_map_install_ms"] for r in sent]),
            "reward_gt_ready_after_action_open": summary(
                [r["gt_ready_after_anchor_ms"] for r in sent if r["kind"] == "POLICY_DECISION"]),
            "evaluator": summary([r.get("evaluator_ms") for r in sent if r["kind"] == "POLICY_DECISION"]),
            "feedback_emit_after_action_open": summary(
                [r.get("feedback_emit_after_anchor_ms") for r in sent if r["kind"] == "POLICY_DECISION"]),
            "front_overlap_edge": summary([g.get("front_overlap_edge_ms") for g in gpu if "front_ms" in g]),
            "front_overlap_carla": summary([g.get("front_overlap_carla_ms") for g in gpu if "front_ms" in g]),
            "edge_overlap_front": summary([g.get("edge_overlap_front_ms") for g in gpu if "edge_ms" in g]),
            "edge_overlap_carla": summary([g.get("edge_overlap_carla_ms") for g in gpu if "edge_ms" in g]),
        },
        "by_kind_ms": {
            kind: {k: summary([r.get(k + "_ms") for r in sent if r["kind"] == kind])
                   for k in ("capture_to_anchor", "anchor_to_last_datagram", "input_7ch", "front",
                             "edge_compute", "capture_to_map_install", "action_open_to_feedback")}
            for kind in sorted({r["kind"] for r in sent})},
        "outcomes": {
            "resolution_terminals": dict(Counter(t["resolution_terminal"] for t in reward_rows)),
            "timeouts": sum(1 for t in reward_rows if t["resolution_terminal"] == "TIMEOUT"),
            "late_feedback": sum(1 for f in ue["feedback_rows"] if f["class"] == "LATE_ORPHAN"),
            "exclusions": sum(1 for t in reward_rows if t["learning_included"] is False),
            "feedback_classes": dict(Counter(f["class"] for f in ue["feedback_rows"])),
            "evaluator_counters": (edge.get("evaluator") or {}),
            "map_outcomes": dict(Counter(r.get("outcome") for r in terminals)),
            "gt_low_skipped": sum(1 for t in tickets.values() if t.get("low_skipped")),
        },
        "integration": integration_checks(ue, edge, terminals, ingest, transmitted, feedback,
                                          reward_rows, sent),
        "tickets": reward_rows,
        "frames": [{k: v for k, v in r.items() if not k.startswith("_")} for r in frames],
        "gpu_overlap": gpu,
    }
    return out


def integration_checks(ue, edge, terminals, ingest, transmitted, feedback, reward_rows,
                       sent) -> dict[str, Any]:
    reward_frames = {f for f, i in transmitted.items() if i.get("reward_requested") is True}
    checks: dict[str, Any] = {}
    # identity: edge evaluations and map identities equal the transmitted identity
    ev_mismatch = [int(e["frame_id"]) for e in edge.get("evaluations", [])
                   if e.get("run4_identity") != transmitted.get(int(e["frame_id"]))]
    fb_foreign = [f for f in feedback if f not in reward_frames]
    checks["evaluation_identity_mismatches"] = ev_mismatch
    checks["feedback_for_non_reward_frames"] = fb_foreign
    checks["feedback_unknown_orphans"] = [f["frame_id"] for f in ue["feedback_rows"]
                                          if f["class"] == "UNKNOWN_ORPHAN"]
    checks["duplicate_feedback_frames"] = [f for f, rows in feedback.items() if len(rows) > 1]
    checks["duplicate_ignored"] = [f["frame_id"] for f in ue["feedback_rows"]
                                   if f["class"] == "DUPLICATE_IGNORED"]
    # terminal accounting: exactly one terminal per transmitted frame
    per_frame = Counter(int(r["frame_id"]) for r in terminals
                        if str(r.get("terminal", "")).lower() in ("true", "1"))
    checks["frames_without_terminal"] = sorted(set(transmitted) - set(per_frame))
    checks["frames_with_multiple_terminals"] = sorted(f for f, n in per_frame.items() if n > 1)
    checks["terminals_for_untransmitted_frames"] = sorted(set(per_frame) - set(transmitted))
    # supersession: a superseded frame names a newer transmitted frame
    bad_super = []
    for r in terminals:
        by = r.get("superseded_by_frame_id")
        if by not in (None, ""):
            if int(by) <= int(r["frame_id"]) or int(by) not in transmitted:
                bad_super.append(int(r["frame_id"]))
    checks["invalid_supersession"] = bad_super
    installed = [int(r["frame_id"]) for r in terminals if r.get("outcome") == "RESULT_INSTALLED"]
    checks["installed_without_install_stamp"] = [
        f for f in installed if not (ingest.get(f, {}).get("map_install_at"))]
    checks["unresolved_tickets_at_close"] = ue.get("unresolved_tickets_at_close")
    checks["tickets_without_resolution"] = [t["frame_id"] for t in reward_rows
                                            if t["resolution_terminal"] is None]
    checks["faulted"] = ue.get("faulted")
    checks["passed"] = (not ev_mismatch and not fb_foreign
                        and not checks["feedback_unknown_orphans"]
                        and not checks["duplicate_feedback_frames"]
                        and not checks["duplicate_ignored"]
                        and not checks["frames_without_terminal"]
                        and not checks["frames_with_multiple_terminals"]
                        and not checks["terminals_for_untransmitted_frames"]
                        and not bad_super and not checks["installed_without_install_stamp"]
                        and not checks["unresolved_tickets_at_close"]
                        and not checks["tickets_without_resolution"]
                        and not checks["faulted"])
    return checks


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cell", type=Path)
    parser.add_argument("--write", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = analyze(args.cell)
    with args.write.open("x", encoding="utf-8") as handle:     # create-only
        json.dump(result, handle, indent=1, sort_keys=True, default=str)
        handle.write("\n")
    print(json.dumps({k: result[k] for k in (
        "counts", "policy_deadline_action_open_to_feedback_le_170",
        "system_kpi_capture_to_feedback_le_200", "feedback_before_next_eligible_decision",
        "outcomes")}, indent=1, default=str))
    print("integration_passed", result["integration"]["passed"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
