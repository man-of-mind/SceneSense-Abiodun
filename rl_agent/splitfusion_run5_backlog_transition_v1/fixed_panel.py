"""Small fixed-action panel (diagnostic, no policy): where does the corrected transition produce backlog, by profile?
m11 q3000 / q5000 / q7000, 1,200 decisions each, old vs corrected transition, identical seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as orch
from rl_agent.splitfusion_run5_full_v1 import campaign as FCMP
from rl_agent.splitfusion_run5_full_v1 import collector as FC

from .collector import BacklogTransitionV1, FullRun5BacklogCollectorV1

PROFILES = ("FAVORABLE_STABLE", "MID_VARIABLE", "ADVERSE_STABLE", "FADE_RECOVERY")


def run(fit: Path, timing: Path, out: Path, n: int, seed: int) -> None:
    out.mkdir(parents=True, exist_ok=False)
    shared, latency = FCMP.components()
    res = {}
    for q in (3000, 5000, 7000):
        for label in ("old", "corrected"):
            bt = BacklogTransitionV1(fit, timing) if label == "corrected" else None
            c = (FullRun5BacklogCollectorV1(latency=latency, seed=seed, shared_sources=shared, backlog_transition=bt))
            for i in range(n):
                c.collect_transition(orch.ModeledActionRequestV1(decision_ordinal=i, mode_id=11, q_e4=q,
                                                                 source="STOCHASTIC_ACTOR", warmup_q_bin_index=None))
            for prof in PROFILES + ("ALL",):
                rs = [r for r in c.diagnostics if prof == "ALL" or r["profile"] == prof]
                inc = [r for r in rs if not r["excluded"]]
                res[f"q{q}|{label}|{prof}"] = dict(
                    n=len(rs), wire_p50_kb=sorted(r["wire_bytes"] for r in rs)[len(rs) // 2] / 1e3,
                    backlog_gt1kb=sum(r["backlog_bytes"] > 1000 for r in rs) / len(rs),
                    backlog_max_kb=max(r["backlog_bytes"] for r in rs) / 1e3,
                    timely=sum(r["terminal"] == "SUCCESS" for r in inc) / len(inc),
                    reward=sum(r["reward"] for r in inc) / len(inc))
    for q in (3000, 5000, 7000):
        for prof in PROFILES + ("ALL",):
            o, c = res[f"q{q}|old|{prof}"], res[f"q{q}|corrected|{prof}"]
            print(f"m11 q{q} ({c['wire_p50_kb']:.0f} KB) {prof:17s} n {c['n']:4d}: B>1KB {o['backlog_gt1kb']:.3f}->{c['backlog_gt1kb']:.3f} "
                  f"(max {c['backlog_max_kb']:.0f} KB) timely {o['timely']:.3f}->{c['timely']:.3f} reward {o['reward']:.3f}->{c['reward']:.3f}")
    (out / "FIXED_PANEL.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fit", required=True)
    p.add_argument("--timing", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=1200)
    p.add_argument("--seed", type=int, default=9017)
    a = p.parse_args()
    run(Path(a.fit), Path(a.timing), Path(a.out), a.n, a.seed)
