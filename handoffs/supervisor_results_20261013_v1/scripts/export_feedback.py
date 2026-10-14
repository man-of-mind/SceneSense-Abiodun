"""Export packet traffic and recorded UE reward resolutions. No live imports."""
import csv
import hashlib
import json
import socket
import struct
import uuid
from collections import Counter
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
ARMS = ("agent", "mode03", "mode09")
FB = struct.Struct("<4sHBBBBHIQQQQQ16s32s32sd")
KINDS = {1: "DELIVERED_SUCCESS", 2: "REGISTERED_DELIVERY_FAILURE", 3: "REGISTERED_SERVICE_FAILURE",
         4: "INFRASTRUCTURE_FAULT", 5: "EVALUATOR_FAULT"}


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def quantiles(values):
    values = [v for v in values if v is not None]
    return {"n": len(values), **{f"p{q}": float(np.percentile(values, q)) if values else None for q in (50, 95, 99)}}


def parse_pcap(path):
    b = path.read_bytes()
    assert b[:4] == b"\xd4\xc3\xb2\xa1" and struct.unpack_from("<I", b, 20)[0] == 101
    rows, offset = [], 24
    while offset < len(b):
        assert offset + 16 <= len(b)
        sec, us, incl, original = struct.unpack_from("<IIII", b, offset)
        ip = b[offset + 16:offset + 16 + incl]
        offset += 16 + incl
        assert len(ip) == incl == original and ip[0] >> 4 == 4 and ip[9] == 17
        ihl = (ip[0] & 15) * 4
        assert struct.unpack_from("!H", ip, 6)[0] & 0x3FFF == 0, "Fragmented packet"
        ip_len = struct.unpack_from("!H", ip, 2)[0]
        src, dst, udp_len, _ = struct.unpack_from("!HHHH", ip, ihl)
        assert ip_len == len(ip) == ihl + udp_len
        payload = ip[ihl + 8:ihl + udp_len]
        r = {"packet_wall_ns": sec * 10**9 + us * 1000,
             "source_ip": socket.inet_ntoa(ip[12:16]), "destination_ip": socket.inet_ntoa(ip[16:20]),
             "source_port": src, "destination_port": dst, "ip_bytes": ip_len,
             "udp_payload_bytes": len(payload), "payload_sha256": hashlib.sha256(payload).hexdigest(),
             "message_type": "OTHER_CONTROL", "frame_id": None, "wire_q_perc": None,
             "wire_feedback_kind": None, "other_control_json": None}
        if payload.startswith(b"R4FB"):
            assert len(payload) == FB.size + 32 and hashlib.sha256(payload[:-32]).digest() == payload[-32:]
            (magic, version, kind, flags, reason, mode, q, anchor, decision, ticket,
             frame, tensor, capture, session, lineage, bundle, quality) = FB.unpack(payload[:-32])
            assert version == 1 and kind in KINDS and flags & 2 and not flags & ~7
            assert (anchor != 0xFFFFFFFF) == bool(flags & 1)
            r.update(message_type="R4FB", frame_id=frame, wire_feedback_kind=KINDS[kind],
                     wire_q_perc=quality if flags & 4 else None, wire_reason_code=reason,
                     mode=mode, q=q / 10000, session_uuid=str(uuid.UUID(bytes=session)),
                     decision_seq=decision, ticket_seq=ticket, tensor_seq=tensor,
                     capture_timestamp_ns=capture)
        else:
            r["other_control_json"] = json.dumps(json.loads(payload.decode()), sort_keys=True)
        rows.append(r)
    assert offset == len(b)
    return rows


def export(output):
    output.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((PACKAGE / "data/METRICS.json").read_text())
    all_packets, all_decisions, all_rates, summaries = [], [], [], {}
    for arm in ARMS:
        raw = PACKAGE / "raw" / arm
        phase = json.loads((raw / "PHASE6_UE_EVIDENCE.json").read_text())
        origin, duration = metrics[arm]["capture_origin_wall_s"], metrics[arm]["duration_s"]
        packets = parse_pcap(raw / "quality_ack_oaitun_ue1.pcap")
        received = {r["sha256"]: r for r in phase["feedback_rows"]}
        assert len(received) == len(phase["feedback_rows"])
        packet_by_hash = {r["payload_sha256"]: r for r in packets if r["message_type"] == "R4FB"}
        assert len(packet_by_hash) == sum(r["message_type"] == "R4FB" for r in packets)
        assert set(packet_by_hash) == set(received), "PCAP/UE quality-feedback mismatch"
        for r in packets:
            r.update(arm=arm, packet_elapsed_s=r["packet_wall_ns"] / 1e9 - origin)
            fb = received.get(r["payload_sha256"])
            r["matched_ue_feedback"] = fb is not None
            if fb:
                assert r["frame_id"] == fb["frame_id"] and r["udp_payload_bytes"] == fb["bytes"]
                assert r["wire_feedback_kind"] == fb["kind"] and r["wire_q_perc"] == fb["q_perc"]
        frame_by_identity = {(r["session_uuid"], r["decision_seq"]): r
                             for r in phase["frames"] if r["reward_requested"]}
        decisions = {r["frame_id"]: r for r in phase["decisions"] if r["kind"] == "POLICY_DECISION"}
        fb_by_frame = {r["frame_id"]: r for r in phase["feedback_rows"]}
        assert len(frame_by_identity) == len(decisions) == len(phase["resolutions"]) == metrics[arm]["decisions"]
        rows = []
        for res in phase["resolutions"]:
            ident = res["identity"]
            fr = frame_by_identity[(ident["session_uuid"], ident["decision_seq"])]
            fid = fr["frame_id"]
            dec, fb = decisions[fid], fb_by_frame.get(fid)
            assert dec["decision_session_uuid"] == ident["session_uuid"]
            assert dec["action_open"]["ns"] == res["action_open_timestamp_ns"]
            packet = packet_by_hash[fb["sha256"]] if fb else None
            receipt = fb["receipt"]["ns"] if fb else None
            latency = (receipt - dec["action_open"]["ns"]) / 1e6 if receipt is not None else None
            if res["terminal"] == "SUCCESS":
                assert fb and latency <= 170 and res["latency_ms"] == latency
                assert np.isclose(res["reward"], res["q_perc"] - .25 * latency / 170, atol=1e-14, rtol=0)
            elif res["terminal"] == "TIMEOUT":
                assert res["reward"] == -1 and res["latency_ms"] is None
            else:
                assert res["reward"] is None and not res["learning_included"]
            rows.append({"arm": arm, "frame_id": fid,
                         "capture_elapsed_s": dec["capture"]["ns"] / 1e9 - origin,
                         "capture_wall_ns": dec["capture"]["ns"], "session_uuid": ident["session_uuid"],
                         "decision_seq": ident["decision_seq"], "mode": dec["mode_id"], "q": dec["q_e4"] / 10000,
                         "action_open_raw_ns": dec["action_open"]["ns"], "feedback_received": fb is not None,
                         "feedback_receipt_raw_ns": receipt, "action_to_feedback_ms": latency,
                         "feedback_timely": latency <= 170 if latency is not None else False,
                         "feedback_kind": fb["kind"] if fb else None, "feedback_class": fb["class"] if fb else None,
                         "feedback_reason": fb["reason"] if fb else None,
                         "wire_q_perc": fb["q_perc"] if fb else None,
                         "udp_payload_bytes": fb["bytes"] if fb else None,
                         "packet_wall_ns": packet["packet_wall_ns"] if packet else None,
                         "packet_elapsed_s": packet["packet_elapsed_s"] if packet else None,
                         "feedback_payload_sha256": fb["sha256"] if fb else None,
                         "controller_terminal": res["terminal"], "logged_reward": res["reward"],
                         "reward_schema_id": res["reward_schema_id"], "reward_q_perc": res["q_perc"],
                         "reward_latency_ms": res["latency_ms"], "learning_included_flag": res["learning_included"],
                         "resolution_raw_ns": res["resolution_timestamp_ns"],
                         "action_to_resolution_ms": (res["resolution_timestamp_ns"] - dec["action_open"]["ns"]) / 1e6})
        rows.sort(key=lambda r: r["capture_elapsed_s"])
        assert len({r["frame_id"] for r in rows}) == len(rows)
        assert sum(r["feedback_timely"] for r in rows) == metrics[arm]["feedback_timely"]
        rates = []
        for second in range(int(duration)):
            window = [r for r in packets if second <= r["packet_elapsed_s"] < second + 1]
            quality = [r for r in window if r["message_type"] == "R4FB"]
            rates.append({"arm": arm, "bin_start_s": second, "bin_end_s": second + 1,
                          "feedback_port_packet_count": len(window), "r4fb_packet_count": len(quality),
                          "other_control_packet_count": len(window) - len(quality),
                          "feedback_port_udp_payload_Mbps": sum(r["udp_payload_bytes"] for r in window) * 8 / 1e6,
                          "r4fb_udp_payload_Mbps": sum(r["udp_payload_bytes"] for r in quality) * 8 / 1e6,
                          "feedback_port_ip_Mbps": sum(r["ip_bytes"] for r in window) * 8 / 1e6})
        summaries[arm] = {"policy_decisions": len(rows), "recorded_feedback": len(received),
                          "missing_feedback": len(rows) - len(received), "exact_pcap_hash_matches": len(packet_by_hash),
                          "packet_types": dict(Counter(r["message_type"] for r in packets)),
                          "outside_complete_rate_bins": sum(not 0 <= r["packet_elapsed_s"] < int(duration) for r in packets),
                          "controller_terminals": dict(Counter(r["controller_terminal"] for r in rows)),
                          "observed_action_to_feedback_ms": quantiles(r["action_to_feedback_ms"] for r in rows),
                          "logged_reward_defined_only": quantiles(r["logged_reward"] for r in rows),
                          "quality_feedback_udp_payload_bytes": quantiles(r["udp_payload_bytes"] for r in packets if r["message_type"] == "R4FB"),
                          "rates_1s_Mbps": {key: quantiles(r[key] for r in rates) for key in
                                             ("feedback_port_udp_payload_Mbps", "r4fb_udp_payload_Mbps", "feedback_port_ip_Mbps")}}
        all_packets.extend(packets); all_decisions.extend(rows); all_rates.extend(rates)
    write_csv(output / "DOWNLINK_PACKETS.csv", all_packets)
    write_csv(output / "FEEDBACK_REWARD_DECISIONS.csv", all_decisions)
    write_csv(output / "DOWNLINK_RATES_1S.csv", all_rates)
    (output / "DOWNLINK_SUMMARY.json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PACKAGE / "feedback_regenerated")
    export(parser.parse_args().output.resolve())
