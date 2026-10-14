"""Per-case evidence for the stall decisions identified in splitfusion_single_ue_feedback_latency_v1 (read-only, CPU).

For each case window (the expanded interval, ms after UE action-open, CLOCK_MONOTONIC_RAW) this joins only records
that already exist: UE freeze-thread wall vs its own thread CPU (FREEZE message stats), GT worker request wall vs
thread CPU, GT worker gen2 events, and every UE profiler record by thread inside the window (gauges from the
profiler-flush thread; spans/marks from MainThread, CARLA sensor callback threads, GT build threads, the receive
thread). It classifies nothing beyond what those records show.

  env -u PYTHONPATH OMP_NUM_THREADS=1 python3 -B -m rl_agent.splitfusion_ue_stall_diagnosis_v1.cases \
    --joined <DECISIONS_JOINED.csv> --capture V11_GOOD_ALL=<root> --capture V20_DEMO_03=<root> --out <new dir>
"""
import argparse
import bisect
import collections
import csv
import json
from pathlib import Path

UE_STAGES = ("st_ue_actor", "st_ue_pre_input", "st_ue_input", "st_ue_encode_send")
GT_SERVICE_STAGES = ("st_gt_extra_wait", "st_score_wait", "st_score_exec", "st_result_return", "st_pred_to_service",
                     "st_feedback_prep")
WINDOW = {"st_ue_actor": (None, "actor_end"), "st_ue_pre_input": ("actor_end", "ue_input_start"),
          "st_ue_input": ("ue_input_start", "ue_input_end"), "st_ue_encode_send": ("ue_input_end", "ue_last_send")}


def f(x):
    return None if x in ("", None) else float(x)


def select(rows):
    out = []
    for r in rows:
        cls = r["spike_class"].replace("SINGLE_STAGE:", "")
        if r["group"] == "EXTREME_GT200" and cls in UE_STAGES or r["ue_freeze_stall"] == "True":
            out.append(("A_UE_BEFORE_SEND", cls if cls in UE_STAGES else "st_ue_pre_input", r))
        elif r["group"] == "EXTREME_GT200" and cls in GT_SERVICE_STAGES:
            out.append(("B_GT_OR_SERVICE", cls, r))
        elif f(r["st_tun_to_app"]) is not None and f(r["st_tun_to_app"]) >= 20:
            out.append(("C_AT_UE_NOT_RECEIVED", "st_tun_to_app", r))
    return out


def load_capture(root):
    root = Path(root)
    cell = next((root / "capture/cells").iterdir())
    frames = {}
    for line in open(cell / "numeric_gt/service/FRAMES.jsonl"):
        d = json.loads(line)
        if d.get("decision"):
            frames[d["frame_id"]] = d
    names, recs = {}, []
    for line in open(cell / "profiler/ue.profile.jsonl"):
        d = json.loads(line)
        r = d.get("r")
        if r == "thread":
            names[d["tid"]] = d["name"]
        elif r == "gauge":
            recs.append((d["t_raw_ns"], "profiler-flush(gauge)"))
        elif r == "span" and d.get("ph") == "live":
            recs.append((d["start_raw_ns"], ("tid", d["tid"], d["stage"] + ":start")))
            recs.append((d["end_raw_ns"], ("tid", d["tid"], d["stage"] + ":end")))
        elif r == "mark" and d.get("ph") == "live":
            recs.append((d["t_raw_ns"], ("tid", d["tid"], d["stage"])))
    recs.sort(key=lambda x: x[0])
    gc = {}
    for role in ("prep", "score"):
        p = root / f"gt_gc_trace/{role}_gc_trace.json"
        gc[role] = [(g["start_raw_ns"], g["wall_ns"], g["cpu_ns"], g.get("op")) for g in json.loads(p.read_bytes())["gen2"]]
    return dict(frames=frames, names=names, recs=recs, times=[x[0] for x in recs], gc=gc)


def label(rec, names):
    if isinstance(rec, str):
        return rec
    name = names.get(rec[1], f"tid{rec[1]}")
    if name.startswith("Dummy-"):
        name = "carla-callback"
    return name


def evidence(cap, kind, stage, r):
    fr = cap["frames"][int(r["frame_id"])]
    ao = fr["action_open_raw_ns"]
    ms = lambda ns: (ns - ao) / 1e6
    if kind == "A_UE_BEFORE_SEND":
        a, b = WINDOW[stage]
        lo, hi = (0.0 if a is None else f(r[a])), f(r[b])
    elif kind == "C_AT_UE_NOT_RECEIVED":
        lo, hi = f(r["ue_tun_arrival"]), f(r["ue_receipt"])
    else:
        lo, hi = None, None
    out = dict(kind=kind, capture=r["capture"], decision_index=int(r["decision_index"]), frame_id=int(r["frame_id"]),
               stage=stage, receipt_ms=round(f(r["ue_receipt"]), 1), excess_ms=round(f(r["dominant_excess_ms"]), 1))
    ev = {}
    for e in fr["events"]:
        ev.setdefault(e["role"] + ":" + e["name"], e["raw_ns"])
    fs = fr.get("freeze_stats") or {}
    if "ue_freeze_thread:freeze_start" in ev and "ue_freeze_thread:freeze_sealed" in ev:
        wall = ms(ev["ue_freeze_thread:freeze_sealed"]) - ms(ev["ue_freeze_thread:freeze_start"])
        out.update(ue_freeze_wall_ms=round(wall, 1), ue_freeze_thread_cpu_ms=fs.get("thread_cpu_ms") and round(fs["thread_cpu_ms"], 1))
    req = {(q["worker"], q["op"]): q for q in fr.get("requests", [])}
    for key, col in ((("prep", "FREEZE"), "prep_freeze"), (("score", r["scoring_op"]), "score_op")):
        q = req.get(key)
        if q:
            out[f"{col}_wall_ms"] = round((q["timing"]["end_raw_ns"] - q["timing"]["begin_raw_ns"]) / 1e6, 1)
            out[f"{col}_cpu_ms"] = round(q["timing"]["thread_cpu_ns"] / 1e6, 1)
    end = ao + f(r["ue_receipt"]) * 1e6
    for role in ("prep", "score"):
        hits = [g for g in cap["gc"][role] if ao <= g[0] <= end]
        out[f"{role}_gen2_in_decision"] = [dict(at_ms=round(ms(g[0]), 1), wall_ms=round(g[1] / 1e6, 1),
                                                cpu_ms=round(g[2] / 1e6, 1), op=g[3]) for g in hits]
    if lo is not None and hi is not None and hi > lo:
        out.update(window_ms=[round(lo, 1), round(hi, 1)], window_len_ms=round(hi - lo, 1))
        a_ns, b_ns = ao + lo * 1e6, ao + hi * 1e6
        i0, i1 = bisect.bisect_left(cap["times"], a_ns), bisect.bisect_right(cap["times"], b_ns)
        inside = collections.Counter(label(x[1], cap["names"]) for x in cap["recs"][i0:i1])
        out["records_inside_window_by_thread"] = dict(inside)
        # longest silence of the gauge sampler overlapping the window
        g = [t for t, x in cap["recs"] if x == "profiler-flush(gauge)"] if "gauge_times" not in cap else cap["gauge_times"]
        cap["gauge_times"] = g
        k = bisect.bisect_left(g, a_ns)
        prev, nxt = g[max(0, k - 1)], g[min(len(g) - 1, bisect.bisect_right(g, b_ns))]
        gaps = [(y - x) / 1e6 for x, y in zip(g[max(0, k - 1):], g[max(0, k - 1) + 1:]) if x <= b_ns]
        out["gauge_longest_gap_overlapping_ms"] = round(max(gaps), 1) if gaps else None
        out["gauge_gap_bounds_ms"] = [round(ms(prev), 1), round(ms(nxt), 1)]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--joined", type=Path, required=True)
    ap.add_argument("--capture", action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=False)
    rows = list(csv.DictReader(open(a.joined)))
    caps = {}
    for spec in a.capture:
        lab, root = spec.split("=", 1)
        caps[lab] = load_capture(root)
    cases = [evidence(caps[r["capture"]], k, s, r) for k, s, r in select(rows)]
    for c in cases:
        c.pop("gauge_times", None)
    (a.out / "CASES.json").write_text(json.dumps(cases, indent=1))
    cols = ["kind", "capture", "decision_index", "frame_id", "stage", "receipt_ms", "excess_ms", "window_ms", "window_len_ms",
            "ue_freeze_wall_ms", "ue_freeze_thread_cpu_ms", "prep_freeze_wall_ms", "prep_freeze_cpu_ms", "score_op_wall_ms",
            "score_op_cpu_ms", "gauge_longest_gap_overlapping_ms", "records_inside_window_by_thread",
            "prep_gen2_in_decision", "score_gen2_in_decision"]
    with open(a.out / "CASES.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for c in cases:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in c.items()})
    print(collections.Counter(c["kind"] for c in cases))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
