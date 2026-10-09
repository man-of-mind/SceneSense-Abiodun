"""Per-decision transport rows from the existing boundary tables (no new measurement).

T = edge_reassembly_complete - ue_last_send (ms): the T_v2b boundary and the environment's stage E. Both stamps
are on the one host monotonic timeline the boundary tables already use. A missing stamp or a non-positive interval
would be counted and excluded (registered policy of the transport contract); in these sessions there are none.
"""
from __future__ import annotations

import json
from pathlib import Path

from rl_agent.ue_production_queue_capture_v1 import contract as V1

from . import plan as P

ON_TIME = ("TIMELY_DEFINED", "TIMELY_UNDEFINED")


class DataError(RuntimeError):
    pass


def _require(ok: bool, msg: str) -> None:
    if not ok:
        raise DataError(msg)


def load_rows(session: str) -> tuple[list[dict], dict]:
    rows_rel, _, _, partition = P.SESSIONS[session]
    raw = [json.loads(l) for l in (P.ROOT / rows_rel).read_text().splitlines() if l.strip()]
    out, audit = [], dict(rows=len(raw), cold=0, not_policy=0, missing_stamp=0, inversion=0, missing_mcs=0,
                          receipt_rule_mismatch=0)
    for r in raw:
        if r.get("cold"):
            audit["cold"] += 1
            continue
        if (r.get("executed_action") or {}).get("kind") != "POLICY_DECISION":
            audit["not_policy"] += 1
            continue
        if r.get("ue_last_send") is None or r.get("edge_reassembly_complete") is None:
            audit["missing_stamp"] += 1
            continue
        if r.get("prior_ul_mcs") is None:
            audit["missing_mcs"] += 1
            continue
        t = float(r["edge_reassembly_complete"]) - float(r["ue_last_send"])
        if t <= 0:
            audit["inversion"] += 1
            continue
        on_time = r["outcome_class"] in ON_TIME
        audit["receipt_rule_mismatch"] += int(on_time != (float(r["ue_receipt"]) <= P.DEADLINE_MS))
        out.append(dict(session=session, partition=partition, decision_index=int(r["decision_index"]),
                        frame_id=int(r["frame_id"]), T=t, payload=int(r["payload_bytes"]),
                        wire=V1.udp_application_bytes(int(r["payload_bytes"])), mcs=int(r["prior_ul_mcs"]),
                        backlog=float(r["pre_action_rlc_backlog_bytes"]), on_time=on_time,
                        outcome=r["outcome_class"], receipt=float(r["ue_receipt"]), mode=int(r["executed_action"]["mode_id"]),
                        q=int(r["executed_action"]["q_e4"]), camera_si=float(r["camera_si"]),
                        radar_p40=float(r["radar_p40"]), access=float(r["edge_first_datagram"]) - float(r["ue_last_send"])))
    _require(audit["receipt_rule_mismatch"] == 0, f"{session}: outcome class disagrees with receipt <= 170 ms")
    return out, audit


def load_all(transport) -> tuple[list[dict], dict]:
    rows, audits = [], {}
    for s in P.SESSIONS:
        r, a = load_rows(s)
        for x in r:
            pr = transport.predict(pre_enqueue_backlog_bytes=x["backlog"], wire_bytes=x["wire"], prior_ul_mcs=x["mcs"])
            x["mu_v2b"], x["p_v2b"] = pr.conditional_latency_ms, pr.on_time_probability
        rows += r
        audits[s] = a
    return rows, audits


def low_region(x: dict) -> bool:
    return x["mcs"] <= P.LOW_MCS_MAX and x["backlog"] == 0


def high_region(x: dict) -> bool:
    return x["mcs"] >= P.MCS_REF and x["backlog"] == 0
