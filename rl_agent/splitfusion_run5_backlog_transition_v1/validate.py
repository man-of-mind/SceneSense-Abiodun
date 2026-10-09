"""LOSO validation of the transition against observed next-decision backlog (plan acceptance A1-A4)."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from rl_agent.ue_production_queue_capture_v1 import contract as V1
from rl_agent.ue_production_transport_model_v2 import artifact_v2 as A2

from . import sessions as S
from .transition import RHO, RateTableV1, next_backlog

THR = 1000


def units(name: str) -> list[dict]:
    rel, cell_rel, part = S.SESSIONS[name]
    cell = S.PROFILER / cell_rel
    frames = S.load_frames(cell)
    dec = S.load_decisions(rel)
    with open(cell / "per_frame_metrics.csv") as f:
        payload = {int(r["frame_id"]): int(float(r["payload_bytes"])) for r in csv.DictReader(f)
                   if r.get("frame_id") and r.get("payload_bytes") and r.get("prepare_status") == "SENT"}
    pol = [f for f in frames if f["kind"] == "POLICY_DECISION" and f["frame_id"] in dec and f["age"] is not None]
    out = []
    for a, b in zip(pol, pol[1:]):
        ra, rb = dec[a["frame_id"]], dec[b["frame_id"]]
        if int(rb["decision_index"]) != int(ra["decision_index"]) + 1:
            continue
        if None in (ra.get("pre_action_rlc_backlog_bytes"), rb.get("pre_action_rlc_backlog_bytes"),
                    ra.get("prior_ul_mcs"), rb.get("prior_ul_mcs")):
            continue
        t_sample = (b["open"] - b["age"] - a["open"]) / 1e6
        between = [f for f in frames if a["open"] <= f["first_send"] < b["open"] - b["age"] and f["frame_id"] >= a["frame_id"]]
        if any(f["frame_id"] not in payload for f in between):
            continue
        events = [((f["first_send"] - a["open"]) / 1e6, RHO * V1.udp_application_bytes(payload[f["frame_id"]])) for f in between]
        t_switch = events[1][0] if len(events) > 1 else math.inf
        out.append(dict(session=name, partition=part, k=int(ra["decision_index"]), B_k=float(ra["pre_action_rlc_backlog_bytes"]),
                        mcs_k=int(ra["prior_ul_mcs"]), mcs_k1=int(rb["prior_ul_mcs"]), events=events, t_switch=t_switch,
                        t_sample=t_sample, obs=float(rb["pre_action_rlc_backlog_bytes"]), n_frames=len(between),
                        payload_k=int(ra["payload_bytes"]), outcome_k1=rb.get("outcome_class"),
                        held_bytes=events[1][1] if len(events) > 1 else None))
    return out


def predict(u: dict, table: RateTableV1) -> float:
    return next_backlog(carry_bytes=u["B_k"], events=u["events"], t_sample=u["t_sample"], mcs_k=u["mcs_k"],
                        mcs_k1=u["mcs_k1"], t_switch=u["t_switch"], table=table)


def metrics(rows: list[dict]) -> dict:
    obs_pos = [r for r in rows if r["obs"] > THR]
    pred_pos = [r for r in rows if r["pred"] > THR]
    tp = [r for r in obs_pos if r["pred"] > THR]
    drained = [r for r in rows if r["obs"] == 0]
    fp = [r for r in drained if r["pred"] > THR]
    logs = [abs(math.log(r["pred"] / r["obs"])) for r in tp]
    return dict(n=len(rows), observed_gt1kb=len(obs_pos), predicted_gt1kb=len(pred_pos), true_pos=len(tp),
                recall=len(tp) / len(obs_pos) if obs_pos else None, precision=len(tp) / len(pred_pos) if pred_pos else None,
                drained=len(drained), false_pos=len(fp), fp_share=len(fp) / len(drained) if drained else None,
                median_abs_log_ratio=statistics.median(logs) if logs else None,
                obs_bytes_p50=statistics.median([r["obs"] for r in obs_pos]) if obs_pos else None,
                pred_bytes_p50_on_tp=statistics.median([r["pred"] for r in tp]) if tp else None)


def group(r):
    m = r["mcs_k1"]
    return "MCS<=10" if m <= 10 else ("MCS 11-15" if m <= 15 else "MCS>=16")


def run(fit_dir: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    fitdoc = json.loads((fit_dir / "FIT.json").read_text())
    all_table = RateTableV1.from_dict(fitdoc["all_fit"])
    rows = []
    for name, v in S.SESSIONS.items():
        table = RateTableV1.from_dict(fitdoc["loso"][name]) if v[2] == "FIT" else all_table
        for u in units(name):
            u["pred"] = predict(u, table)
            u["old_head"] = 0.0
            rows.append(u)
    fit_rows = [r for r in rows if r["partition"] == "FIT"]
    res = dict(schema="splitfusion.run5_backlog_transition.validation.v1", threshold_bytes=THR,
               pooled_fit_loso=metrics(fit_rows), by_session={}, by_mcs_fit={}, fade_diagnostic=metrics([r for r in rows if r["session"] == "fade"]),
               fade_by_mcs={})
    for name in S.SESSIONS:
        res["by_session"][name] = metrics([r for r in rows if r["session"] == name])
    for g in ("MCS<=10", "MCS 11-15", "MCS>=16"):
        res["by_mcs_fit"][g] = metrics([r for r in fit_rows if group(r) == g])
        res["fade_by_mcs"][g] = metrics([r for r in rows if r["session"] == "fade" and group(r) == g])
    # acceptance
    p = res["pooled_fit_loso"]
    a1 = 0.5 * p["observed_gt1kb"] <= p["predicted_gt1kb"] <= 2.0 * p["observed_gt1kb"] and (p["recall"] or 0) >= 0.5
    a2 = p["fp_share"] <= 0.02 and all(res["by_session"][n]["fp_share"] <= 0.04 for n, v in S.SESSIONS.items() if v[2] == "FIT")
    a3 = p["median_abs_log_ratio"] is not None and p["median_abs_log_ratio"] <= math.log(2)
    base = fitdoc["all_fit"]
    a4 = all(all(abs(r / r0 - 1) <= 0.30 for (lo, hi), r, r0 in zip(t["bins"], t["rate_bytes_per_ms"], base["rate_bytes_per_ms"]) if lo >= 16)
             for t in fitdoc["loso"].values())
    old = dict(predicted_gt1kb=0, recall=0.0)
    res["acceptance"] = dict(A1=a1, A2=a2, A3=a3, A4=a4, old_head_A1=False, old_head=old,
                             decision="READY_FOR_SEPARATELY_APPROVED_TRAINING_COMPARISON" if all((a1, a2, a3, a4)) else "NOT_SUPPORTED")
    # T_v2b immediate interpretation check at observed positive backlog (existing data, all sessions)
    (out / "VALIDATION.json").write_text(json.dumps(res, indent=1))
    with open(out / "PER_TRANSITION.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "events"} | dict(n_events=len(r["events"]))) + "\n")
    def show(label, m):
        print(f"{label:22s} n {m['n']:4d} obs>1KB {m['observed_gt1kb']:3d} pred>1KB {m['predicted_gt1kb']:3d} TP {m['true_pos']:3d} "
              f"recall {m['recall'] if m['recall'] is None else round(m['recall'],2)} precision {m['precision'] if m['precision'] is None else round(m['precision'],2)} "
              f"| drained {m['drained']:4d} FP {m['false_pos']:3d} ({(m['fp_share'] or 0):.3f}) | |log ratio| P50 {m['median_abs_log_ratio'] if m['median_abs_log_ratio'] is None else round(m['median_abs_log_ratio'],2)} "
              f"obs P50 {m['obs_bytes_p50']} pred P50 {m['pred_bytes_p50_on_tp']}")
    show("POOLED FIT (LOSO)", p)
    for n in S.SESSIONS:
        show(n, res["by_session"][n])
    for g in ("MCS<=10", "MCS 11-15", "MCS>=16"):
        show("FIT " + g, res["by_mcs_fit"][g])
    for g in ("MCS<=10", "MCS 11-15", "MCS>=16"):
        show("FADE(diag) " + g, res["fade_by_mcs"][g])
    print("acceptance:", res["acceptance"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fit", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    run(Path(a.fit), Path(a.out))
