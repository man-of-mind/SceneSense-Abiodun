"""Step 1: observation contract and mechanism, from existing evidence only (no fit).

A. Reproduce each recorded pre-action backlog from the raw RLC series with the live selector rule
   (latest complete tick with source < cutoff and available < cutoff; cutoff = state commit) and confirm the
   recorded backlog_age_ns (= action_open - sample source).
B. Byte definition: RLC-entering bytes per frame (PDCP SDU sum between the frame's first send and the next frame's
   first send) vs the frame's UDP application bytes.
C. Mechanism for every positive recorded backlog: which transmitted frame(s) the bytes at the sample instant belong
   to, from the RLC trajectory (buffer just before each frame's first send, and at the sample).
"""
from __future__ import annotations

import argparse
import bisect
import json
import statistics
from pathlib import Path

from rl_agent.ue_production_queue_capture_v1 import contract as V1

from . import sessions as S


def run(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    report = dict(schema="splitfusion.run5_backlog_transition.contract_check.v1", sessions={})
    for name, (rel, cell_rel, part) in S.SESSIONS.items():
        cell = S.PROFILER / cell_rel
        ticks = S.load_rlc_ticks(cell)
        srcs = [t.source_raw_ns for t in ticks]
        frames = S.load_frames(cell)
        dec = S.load_decisions(rel)
        pdcp = S.load_pdcp(cell)
        pt = [p[0] for p in pdcp]
        cum = [0]
        for _, b in pdcp:
            cum.append(cum[-1] + b)
        # ---- A: selector reproduction
        a = dict(n=0, bytes_match=0, age_match_1ms=0, age_ms=[], not_found=0, ingest_ms=[], tick_spacing_ms=None, mism=[])
        for f in frames:
            if f["kind"] != "POLICY_DECISION" or f["age"] is None or f["frame_id"] not in dec:
                continue
            rec = dec[f["frame_id"]].get("pre_action_rlc_backlog_bytes")
            if rec is None:
                continue
            cutoff = f["commit"]
            i = bisect.bisect_left(srcs, cutoff) - 1
            while i >= 0 and ticks[i].available_raw_ns >= cutoff:
                i -= 1
            a["n"] += 1
            if i < 0:
                a["not_found"] += 1
                continue
            t = ticks[i]
            a["bytes_match"] += int(t.bytes == int(rec))
            age = f["open"] - t.source_raw_ns
            a["age_match_1ms"] += int(abs(age - f["age"]) <= 1_000_000)
            a["age_ms"].append(f["age"] / 1e6)
            a["ingest_ms"].append((t.available_raw_ns - t.source_raw_ns) / 1e6)
            if t.bytes != int(rec) and len(a["mism"]) < 5:
                a["mism"].append(dict(frame=f["frame_id"], rec=rec, rebuilt=t.bytes, age_rec_ms=f["age"] / 1e6, age_rebuilt_ms=age / 1e6))
        gaps = [(b - a_) / 1e6 for a_, b in zip(srcs[:-1], srcs[1:]) if 0 < b - a_ < 5e8]
        a["tick_spacing_ms"] = dict(p50=statistics.median(gaps), p99=sorted(gaps)[int(.99 * len(gaps))])
        a["age_ms"] = dict(p50=statistics.median(a["age_ms"]), p99=sorted(a["age_ms"])[int(.99 * len(a["age_ms"]))], max=max(a["age_ms"]))
        a["ingest_ms"] = dict(p50=statistics.median(a["ingest_ms"]), p99=sorted(a["ingest_ms"])[int(.99 * len(a["ingest_ms"]))])
        # ---- B: bytes entering RLC per frame
        ratios = []
        for f0, f1 in zip(frames, frames[1:]):
            r0 = dec.get(f0["frame_id"])
            pb = None
            if r0 is not None and f0["kind"] == "POLICY_DECISION":
                pb = int(r0["payload_bytes"])
            if pb is None:
                continue
            lo, hi = bisect.bisect_left(pt, f0["first_send"] - 1_000_000), bisect.bisect_left(pt, f1["first_send"] - 1_000_000)
            ratios.append((cum[hi] - cum[lo]) / V1.udp_application_bytes(pb))
        b = dict(n=len(ratios), pdcp_over_udp_app_p50=statistics.median(ratios), p05=sorted(ratios)[int(.05 * len(ratios))], p95=sorted(ratios)[int(.95 * len(ratios))])
        # ---- C: mechanism for positive recorded backlogs
        fidx = {f["frame_id"]: k for k, f in enumerate(frames)}
        cases = []
        for f in frames:
            if f["kind"] != "POLICY_DECISION" or f["age"] is None or f["frame_id"] not in dec:
                continue
            rec = dec[f["frame_id"]].get("pre_action_rlc_backlog_bytes")
            if not rec or int(rec) <= 0:
                continue
            k = fidx[f["frame_id"]]
            sample_t = f["open"] - f["age"]
            prev = [frames[j] for j in range(max(0, k - 4), k)]
            sent = [p for p in prev if p["first_send"] < sample_t]
            if not sent:
                continue
            last = sent[-1]
            before_last = S.tick_at(ticks, srcs, last["first_send"] - 500_000)
            lo = bisect.bisect_left(pt, last["first_send"] - 1_000_000)
            hi = bisect.bisect_left(pt, sample_t)
            entered = cum[hi] - cum[lo]
            cases.append(dict(frame=f["frame_id"], rec=int(rec), mcs=dec[f["frame_id"]].get("prior_ul_mcs"),
                              last_kind=last["kind"], last_frame=last["frame_id"],
                              ms_last_send_to_sample=(sample_t - last["last_send"]) / 1e6,
                              ms_prev_open_to_sample=None,
                              buffer_before_last_send=before_last.bytes if before_last else None,
                              rlc_bytes_entered_since_last_first_send=entered,
                              residual_le_last_frame_rlc_bytes=int(rec) <= entered,
                              last_alone=(before_last is not None and before_last.bytes == 0 and int(rec) <= entered)))
        c = dict(n=len(cases), n_gt1kb=sum(1 for x in cases if x["rec"] > 1000),
                 last_kind_counts={k: sum(1 for x in cases if x["last_kind"] == k) for k in ("POLICY_DECISION", "POLICY_HOLD", "FALLBACK")},
                 residual_within_last_frame=sum(x["residual_le_last_frame_rlc_bytes"] for x in cases),
                 last_frame_alone=sum(x["last_alone"] for x in cases),
                 gt1kb_last_frame_alone=sum(x["last_alone"] for x in cases if x["rec"] > 1000),
                 ms_last_send_to_sample_gt1kb=sorted(round(x["ms_last_send_to_sample"], 1) for x in cases if x["rec"] > 1000),
                 cases=cases)
        # decision cadence (live wall time) and held-frame offset relative to the decision's action open
        opens = [f["open"] for f in frames if f["kind"] == "POLICY_DECISION"]
        cad = [(b_ - a_) / 1e6 for a_, b_ in zip(opens, opens[1:]) if (b_ - a_) < 1e9]
        held_off, dec_send = [], []
        for k, f in enumerate(frames):
            if f["kind"] == "POLICY_DECISION":
                dec_send.append((f["last_send"] - f["open"]) / 1e6)
                if k + 1 < len(frames) and frames[k + 1]["kind"] == "POLICY_HOLD":
                    held_off.append((frames[k + 1]["last_send"] - f["open"]) / 1e6)
        commit_to_open = [(f["open"] - f["commit"]) / 1e6 for f in frames if f["kind"] == "POLICY_DECISION" and f["commit"]]
        timeline = dict(cadence_ms_p50=statistics.median(cad), cadence_p10=sorted(cad)[len(cad) // 10], cadence_p90=sorted(cad)[9 * len(cad) // 10],
                        decision_last_send_after_open_ms_p50=statistics.median(dec_send),
                        held_last_send_after_open_ms_p50=statistics.median(held_off), held_p10=sorted(held_off)[len(held_off) // 10], held_p90=sorted(held_off)[9 * len(held_off) // 10],
                        commit_to_open_ms_p50=statistics.median(commit_to_open))
        report["sessions"][name] = dict(partition=part, ticks=len(ticks), A_selector=a, B_bytes=b, C_mechanism=c, timeline=timeline)
        print(f"{name:14s} [{part[:4]}] A: reproduced {a['bytes_match']}/{a['n']} bytes, age within 1 ms {a['age_match_1ms']}, not found {a['not_found']}; "
              f"sample age P50 {a['age_ms']['p50']:.2f} P99 {a['age_ms']['p99']:.2f} max {a['age_ms']['max']:.1f} ms; tick spacing P50 {a['tick_spacing_ms']['p50']:.2f} ms")
        print(f"{'':14s} B: PDCP bytes / UDP application bytes P50 {b['pdcp_over_udp_app_p50']:.4f} [P05 {b['p05']:.4f}, P95 {b['p95']:.4f}] (n {b['n']})")
        print(f"{'':14s} C: positive {c['n']} (>1 KB {c['n_gt1kb']}); last frame sent before sample: {c['last_kind_counts']}; residual within last frame's RLC bytes {c['residual_within_last_frame']}; "
              f"last frame alone (buffer 0 before its send) {c['last_frame_alone']} (>1 KB: {c['gt1kb_last_frame_alone']})")
        print(f"{'':14s}    ms from last frame's last send to sample (>1 KB): {c['ms_last_send_to_sample_gt1kb']}")
        print(f"{'':14s} T: cadence P50 {timeline['cadence_ms_p50']:.0f} [{timeline['cadence_p10']:.0f}, {timeline['cadence_p90']:.0f}] ms; decision last send +{timeline['decision_last_send_after_open_ms_p50']:.0f} ms; "
              f"held last send +{timeline['held_last_send_after_open_ms_p50']:.0f} [{timeline['held_p10']:.0f}, {timeline['held_p90']:.0f}] ms; commit->open {timeline['commit_to_open_ms_p50']:.3f} ms")
    (out / "CONTRACT_CHECK.json").write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    run(Path(p.parse_args().out))
