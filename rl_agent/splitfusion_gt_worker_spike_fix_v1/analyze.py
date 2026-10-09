"""Summaries of replay traces: per-decision prep timing, GC overlap, spikes, RSS, idle maintenance."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def pct(v):
    v = np.asarray(v, float)
    return dict(n=int(v.size), p50=round(float(np.percentile(v, 50)), 3), p95=round(float(np.percentile(v, 95)), 3),
                p99=round(float(np.percentile(v, 99)), 3), max=round(float(v.max()), 3)) if v.size else dict(n=0)


def decisions(trace):
    """One row per decision: FREEZE (G6 decode/validate/rehash, G7 compute) + BIND; GC inside each."""
    ops = trace["ops"]
    gcs = trace["gc_events"]
    by_seq = {}
    for g in gcs:
        by_seq.setdefault(g["op_seq"], []).append(g)
    rows, k = [], -1
    for i, op in enumerate(ops):
        if op["op"] != "FREEZE":
            continue
        k += 1
        bind = next((o for o in ops[i + 1:] if o["op"] == "BIND"), None)
        g6 = (op["decoded_raw_ns"] - op["begin_raw_ns"]) / 1e6
        g7 = (op["end_raw_ns"] - op["decoded_raw_ns"]) / 1e6
        mine = by_seq.get(op["seq"], [])
        def gc_in(lo, hi):
            return [g for g in mine if lo <= g["start_raw_ns"] < hi]
        g6gc, g7gc = gc_in(op["begin_raw_ns"], op["decoded_raw_ns"]), gc_in(op["decoded_raw_ns"], op["end_raw_ns"] + 1)
        bgc = by_seq.get(bind["seq"], []) if bind else []
        rows.append(dict(k=k, frame_id=op.get("frame_id"), freeze_ms=(op["end_raw_ns"] - op["begin_raw_ns"]) / 1e6,
            freeze_cpu_ms=op["cpu_ns"] / 1e6, g6_ms=g6, g7_ms=g7,
            bind_ms=None if bind is None else (bind["end_raw_ns"] - bind["begin_raw_ns"]) / 1e6,
            prep_ms=(op["end_raw_ns"] - op["begin_raw_ns"]) / 1e6 + (0 if bind is None else (bind["end_raw_ns"] - bind["begin_raw_ns"]) / 1e6),
            gc_ms_g6=sum(g["wall_ns"] for g in g6gc) / 1e6, gc_ms_g7=sum(g["wall_ns"] for g in g7gc) / 1e6,
            gc_ms_bind=sum(g["wall_ns"] for g in bgc) / 1e6,
            gen2_in_freeze=sum(1 for g in mine if g["gen"] == 2), gen2_ms_freeze=sum(g["wall_ns"] for g in mine if g["gen"] == 2) / 1e6,
            gen2_in_bind=sum(1 for g in bgc if g["gen"] == 2),
            rss_mb=op["rss1"] / 2**20))
    return rows


def all_request_gc(trace):
    out = {}
    for g in trace["gc_events"]:
        key = (g["gen"], g["op_seq"] is not None)
        d = out.setdefault(f"gen{g['gen']}_{'in_request' if g['op_seq'] is not None else 'outside_request'}", [])
        d.append(g["wall_ns"] / 1e6)
    return {k: dict(count=len(v), total_ms=round(sum(v), 3), **pct(v)) for k, v in out.items()}


def spikes(values, margin=30.0):
    v = np.asarray(values, float)
    med = float(np.median(v))
    return [i for i, x in enumerate(v) if x > med + margin], med


if __name__ == "__main__":
    out = Path(sys.argv[1])
    for t in sorted(out.glob("trace_*.json")):
        tr = json.loads(t.read_text())
        rows = decisions(tr)
        fz = [r["freeze_ms"] for r in rows]
        idx, med = spikes(fz)
        print(t.name, tr["mode"], "FREEZE", pct(fz), "spikes>med+30:", idx)
        for i in idx:
            r = rows[i]
            print(f"   k={i} freeze={r['freeze_ms']:.1f} g6={r['g6_ms']:.1f} g7={r['g7_ms']:.1f} gc_g6={r['gc_ms_g6']:.1f} gc_g7={r['gc_ms_g7']:.1f} gen2={r['gen2_in_freeze']}")
        print("   GC:", all_request_gc(tr))
        g2 = [(g["op_seq"], round(g["wall_ns"] / 1e6, 1), g["collected"]) for g in tr["gc_events"] if g["gen"] == 2]
        print("   gen2 events (op_seq, ms, collected):", g2)
        print("   policy:", [(e["kind"], e.get("decision"), round(e.get("wall_ms", 0), 1)) for e in tr["policy_events"]][:60])
        print("   rss MB first/last/max:", round(rows[0]["rss_mb"], 1), round(rows[-1]["rss_mb"], 1), round(max(r["rss_mb"] for r in rows), 1))
