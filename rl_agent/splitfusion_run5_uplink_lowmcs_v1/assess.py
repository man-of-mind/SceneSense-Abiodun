"""A6 (training-state support) and the registered verdict, applied mechanically to eval_01.

    python -m rl_agent.splitfusion_run5_uplink_lowmcs_v1.assess --run <dir>       # writes <dir>/ASSESSMENT.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from . import model as M
from . import plan as P

RUNS = P.ROOT / "experiments/splitfusion_run5_reward_trials_v1/runs_01"


def training_support(c: M.LowMcsCorrectionV1) -> dict:
    n = low = 0
    flags = {M.UNRESOLVED_BACKLOG: 0, M.EXTRAPOLATED_PAYLOAD: 0, M.EXTRAPOLATED_MCS: 0}
    for f in sorted(glob.glob(str(RUNS / "*/decisions.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            n += 1
            low += r["prior_ul_mcs"] <= P.LOW_MCS_MAX
            for flag in c.flags(backlog_bytes=r["backlog_bytes"], wire_bytes=r["wire_bytes"], prior_ul_mcs=r["prior_ul_mcs"]):
                flags[flag] += 1
    flagged = sum(flags.values())
    return dict(source="reward-trial training decisions (9 runs, binding_02 environment, old-model policies)",
                decisions=n, low_mcs=low, flags=flags, flagged_fraction=flagged / n,
                flagged_fraction_of_low_mcs=flagged / low, passed=flagged / n <= 0.05)


def run(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    ev = json.loads((run_dir / "eval_01/EVALUATION.json").read_text())
    c = M.LowMcsCorrectionV1.load(run_dir / "fit_01/CANDIDATE.json")
    acc = ev["acceptance"]
    a6 = training_support(c)
    cand, old = acc["candidate"], acc["old"]
    checks = dict(A1=cand["A1"], A2=cand["A2"], A3=cand["A3"], A4=acc["A4"]["passed"], A5=acc["A5"]["passed"], A6=a6["passed"])
    need = (not old["A1"]) or (not old["A2"])
    if not checks["A5"] or not (checks["A1"] and checks["A2"]) and (old["A1"] and old["A2"]):
        verdict = "REJECTED"
    elif all(checks.values()) and need:
        verdict = "READY_FOR_ONE_TARGETED_RETRAINING_CANDIDATE"
    elif checks["A3"] and checks["A5"]:
        verdict = "REQUIRES_ONE_MISSING_MEASUREMENT"
    else:
        verdict = "REJECTED"
    low = ev["summary"]["mcs_le15_B0"]
    margins = {t: dict(measured=low[t]["measured_on_time"], wilson95=low[t]["wilson95"],
                       old=low[t]["old"]["own_chain_p_timely"], candidate=low[t]["candidate"]["own_chain_p_timely"],
                       candidate_env=low[t]["candidate"]["env_idle_p_timely"]) for t in ("T1", "T2", "T3")}
    doc = dict(schema="splitfusion.run5_uplink_lowmcs_v1.assessment.v1", plan_sha256=ev["plan_sha256"],
               checks=checks, old_fails_A1_or_A2=need, A6=a6, verdict=verdict, rule=P.VERDICT_RULE,
               tercile_margins=margins, stop=P.STOP)
    (run_dir / "ASSESSMENT.json").write_text(json.dumps(doc, indent=1, sort_keys=True, default=float))
    return doc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    d = run(p.parse_args().run)
    print(json.dumps({k: d[k] for k in ("checks", "old_fails_A1_or_A2", "verdict", "A6")}, indent=1))
