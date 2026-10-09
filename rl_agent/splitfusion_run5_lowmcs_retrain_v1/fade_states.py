"""Same-state action query on the FADE_RECOVERY capture's recorded 22-D states (PREVIOUSLY EXAMINED evidence).

Each actor's deterministic action (the evaluation rule) on every recorded warm decision state. The deployed actor must
reproduce its recorded (mode, q) exactly, which validates the state recovery. This is an action comparison on fixed
states: no payload, latency or quality is implied (those need re-encoding and a measurement), and it is not fresh
confirmation of anything.

  python3 -m rl_agent.splitfusion_run5_lowmcs_retrain_v1.fade_states --runs <ABS runs_01> --out <ABS dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from rl_agent.splitfusion_run5_channel_response_v1_analysis import actor_sensitivity as AS
from rl_agent.splitfusion_run5_channel_response_v1_analysis.channel_analysis import ROOT as CAPTURE
from rl_agent.splitfusion_run5_reward_trials_v1 import evaluate as RE

from . import evaluate as EV
from . import registration as REG

JOINED = REG.ROOT / "experiments/splitfusion_run5_channel_response_v1/analysis_01/DECISIONS_JOINED.jsonl"


def run(runs: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    cell = next((CAPTURE / "capture/cells").iterdir())
    spans = AS.load_spans(cell)
    rows = [json.loads(l) for l in JOINED.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if not r["cold"] and r["frame_id"] in spans]
    pol = {k: v for k, v in EV.policies(runs).items() if v["kind"] == "actor"}
    actions = {}
    for label, spec in pol.items():
        actor = RE.load_actor(REG.ROOT / spec["path"])
        actions[label] = [RE.act(actor, spans[r["frame_id"]]["features"]) for r in rows]
    repro = sum((m, q) == (r["mode"], r["q_e4"]) for (m, q), r in zip(actions["HIST_seed43"], rows))
    strata = {"mcs_9_10|B0": lambda r: r["prior_ul_mcs"] <= 10 and r["pre_action_rlc_backlog_bytes"] == 0,
              "mcs_11_15|B0": lambda r: 11 <= r["prior_ul_mcs"] <= 15 and r["pre_action_rlc_backlog_bytes"] == 0,
              "mcs_le15|Bpos": lambda r: r["prior_ul_mcs"] <= 15 and r["pre_action_rlc_backlog_bytes"] > 0,
              "mcs_ge16": lambda r: r["prior_ul_mcs"] >= 16}
    summary = {}
    for label, acts in actions.items():
        e = {}
        for s, f in strata.items():
            idx = [i for i, r in enumerate(rows) if f(r)]
            q = [acts[i][1] for i in idx]
            e[s] = dict(n=len(idx), mode11_share=float(np.mean([acts[i][0] == 11 for i in idx])),
                        q_p10_p50_p90=[float(np.percentile(q, p)) for p in (10, 50, 90)])
        q_all = [a[1] for a in acts]
        e["spearman_q_snr_descriptive"] = float(stats.spearmanr(q_all, [r["snr_db"] for r in rows]).statistic)
        summary[label] = e
    doc = dict(schema="splitfusion.run5_lowmcs_retrain_v1.fade_states.v1", status="PREVIOUSLY_EXAMINED_EVIDENCE",
               decisions=len(rows), deployed_reproduces_recorded_actions=f"{repro}/{len(rows)}", summary=summary,
               claim="same-state action comparison only; no payload/latency/quality inference; not confirmation")
    (out / "FADE_STATES.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
    return doc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    d = run(a.runs, a.out)
    print(d["deployed_reproduces_recorded_actions"])
    for k, v in d["summary"].items():
        print(k, {s: (x["n"], x["q_p10_p50_p90"][1], round(x["mode11_share"], 2)) for s, x in v.items() if isinstance(x, dict)},
              round(v["spearman_q_snr_descriptive"], 2))
