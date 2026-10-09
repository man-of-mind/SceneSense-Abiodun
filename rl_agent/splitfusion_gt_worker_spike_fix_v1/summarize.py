"""Control-vs-candidate summary of a replay directory -> SUMMARY.json + DECISIONS.csv (+ printed table)."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from .analyze import decisions, pct

FIELDS = ("frame_id", "identity", "semantic", "commit", "q_perc", "measurement", "producer_files")
COLD = 2          # first decisions of each fresh worker, reported separately (bench.py convention)


def main(out):
    out = Path(out)
    sessions = json.loads((out / "SESSIONS.json").read_text())
    ref = [[r[f] for f in FIELDS] for r in sessions[0]["rows"]]
    parity = dict(all_sessions_identical_outputs=all([[r[f] for f in FIELDS] for r in s["rows"]] == ref for s in sessions),
                  history_sha256=sorted({s["history_sha256"] for s in sessions}),
                  q_perc_matches_recorded=all(r["q_perc_matches_recorded"] for s in sessions for r in s["rows"]),
                  per_frame_output_digest=hashlib.sha256(json.dumps(ref, sort_keys=True).encode()).hexdigest())
    arms, csv_rows, per_session = {}, [], []
    for s in sessions:
        tr = json.loads((out / s["trace"]).read_text())
        rows = decisions(tr)
        fz = np.array([r["freeze_ms"] for r in rows])
        med = float(np.median(fz))
        spikes = [r["k"] for r in rows if r["freeze_ms"] > med + 30 and r["k"] >= COLD]
        ops = tr["ops"]
        gaps = [(b["begin_raw_ns"] - a["end_raw_ns"]) / 1e6 for a, b in zip(ops, ops[1:]) if a["op"] == "EVIDENCE_DURABLE"]
        idle = [e for e in tr["policy_events"] if e["kind"] != "skipped_pending"]
        info = dict(index=s["index"], mode=s["mode"], status=s["status"], decisions=len(s["rows"]),
                    prep_returncode=s["prep_returncode"], engine_retired=s["engine"]["retired"],
                    engine_retained=s["engine"]["retained_jobs"], jobs_phase=s["jobs_phase"],
                    ops=len(ops), op_counts={o: sum(1 for x in ops if x["op"] == o) for o in sorted({x["op"] for x in ops})},
                    freeze_median_ms=med, spikes_gt_median_plus_30=spikes,
                    in_request_gen2=[(g["op_seq"], round(g["wall_ns"] / 1e6, 2)) for g in tr["gc_events"]
                                     if g["gen"] == 2 and g["op_seq"] is not None],
                    in_request_gc_ms_total=sum(g["wall_ns"] for g in tr["gc_events"] if g["op_seq"] is not None) / 1e6,
                    prep_cpu_ms_total=sum(o["cpu_ns"] for o in ops) / 1e6,
                    idle_maintenance=dict(count=len(idle), skipped_pending=sum(1 for e in tr["policy_events"]
                                          if e["kind"] == "skipped_pending"),
                                          request_arrived_during=sum(1 for e in idle if e["request_arrived_during"]),
                                          cpu_ms_total=sum(e["cpu_ms"] for e in idle),
                                          wall=pct([e["wall_ms"] for e in idle]),
                                          wall_excl_warmup=pct([e["wall_ms"] for e in idle if e["kind"] == "idle_full_collect"])),
                    replay_gap_after_evidence_durable_ms=pct(gaps),
                    rss_mb=dict(first=rows[0]["rss_mb"], last=rows[-1]["rss_mb"], peak=max(r["rss_mb"] for r in rows),
                                final_process=tr["rss_final"] / 2**20,
                                slope_mb_per_decision_after_cold=float(np.polyfit([r["k"] for r in rows[COLD:]],
                                                                                  [r["rss_mb"] for r in rows[COLD:]], 1)[0])),
                    gc_state_at_close=tr["gc_state_at_close"], gc_state_after_uninstall=tr["gc_state_after_uninstall"])
        per_session.append(info)
        a = arms.setdefault(s["mode"], dict(rows=[], spikes=0, cpu=0.0, idle_cpu=0.0, gen2=0, decisions=0))
        a["rows"] += [r for r in rows if r["k"] >= COLD]
        a["spikes"] += len(spikes); a["cpu"] += info["prep_cpu_ms_total"]; a["idle_cpu"] += info["idle_maintenance"]["cpu_ms_total"]
        a["gen2"] += len(info["in_request_gen2"]); a["decisions"] += len(rows)
        for r in rows:
            csv_rows.append(dict(session=s["index"], mode=s["mode"], **{k: (round(v, 4) if isinstance(v, float) else v)
                                                                         for k, v in r.items()}))
    # Parent-observed GT path incl. any queueing behind the worker: FREEZE sent -> BIND reply end.
    for s in sessions:
        a = arms[s["mode"]]
        a.setdefault("parent", []).extend(r["metrics"]["DELIVERY_ms"] + r["metrics"]["PREPARE_ms"]
                                          for r in s["rows"] if r["k"] >= COLD)
    table = {}
    for mode, a in arms.items():
        rs = a["rows"]
        table[mode] = dict(warm_decisions=len(rs), freeze_ms=pct([r["freeze_ms"] for r in rs]),
                           prep_freeze_plus_bind_ms=pct([r["prep_ms"] for r in rs]), g6_ms=pct([r["g6_ms"] for r in rs]),
                           g7_ms=pct([r["g7_ms"] for r in rs]), bind_ms=pct([r["bind_ms"] for r in rs]),
                           freeze_cpu_ms=pct([r["freeze_cpu_ms"] for r in rs]), spikes=a["spikes"],
                           parent_observed_send_to_bind_end_ms=pct(a["parent"]),
                           parent_observed_spikes=int(sum(1 for x in a["parent"] if x > np.median(a["parent"]) + 30)),
                           in_request_gen2=a["gen2"], prep_request_cpu_ms_per_decision=a["cpu"] / a["decisions"],
                           idle_maintenance_cpu_ms_per_decision=a["idle_cpu"] / a["decisions"],
                           total_cpu_ms_per_decision=(a["cpu"] + a["idle_cpu"]) / a["decisions"])
    summary = dict(parity=parity, arms=table, sessions=per_session,
                   claims=dict(offline_replay_only=True, live_timeliness_established=False))
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=1, default=str))
    with open(out / "DECISIONS.csv", "w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        w.writeheader(); w.writerows(csv_rows)
    print(json.dumps(dict(parity=parity, arms=table), indent=1))
    for i in per_session:
        print(i["index"], i["mode"], "spikes", i["spikes_gt_median_plus_30"], "gen2", i["in_request_gen2"],
              "idle", i["idle_maintenance"]["count"], i["idle_maintenance"]["wall_excl_warmup"].get("max"),
              "rss", round(i["rss_mb"]["first"], 1), round(i["rss_mb"]["peak"], 1), round(i["rss_mb"]["slope_mb_per_decision_after_cold"], 3))


if __name__ == "__main__":
    main(sys.argv[1])
