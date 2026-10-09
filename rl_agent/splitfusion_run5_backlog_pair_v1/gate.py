"""Pre-training sensitivity gate (GATE_RULE.json, frozen before this ran). Fixed m11 actions, training environment."""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np

from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as orch

from . import env as E
from . import rules as RU

ACTIONS = {"q3000": 3000, "q5000": 5000, "q7000": 7000}
VARIANTS = ("OLD", "NEW_A", "NEW_B")
PROFILES = ("FAVORABLE_STABLE", "MID_VARIABLE", "ADVERSE_STABLE", "FADE_RECOVERY")


def _job(args):
    action, variant, seed, n = args
    shared, latency = E.components(neutralized=variant == "NEW_B")
    bt = None if variant == "OLD" else E.transition()
    col = E.TrialBacklogCollectorV1(latency=latency, seed=seed, shared_sources=shared, backlog_transition=bt)
    for i in range(n):
        col.collect_transition(orch.ModeledActionRequestV1(decision_ordinal=i, mode_id=11, q_e4=ACTIONS[action],
                                                           source="STOCHASTIC_ACTOR", warmup_q_bin_index=None))
    rows = [dict(i=d["ordinal"], profile=d["profile"], backlog=d["backlog_bytes"], reward=d["reward"],
                 excluded=d["excluded"], receipt=d["receipt_ms"], carry=d["carry_in_ms"], late=d["uplink_late"],
                 mcs=d["prior_ul_mcs"], scene=d["reward_scene_key"]) for d in col.diagnostics]
    return action, variant, seed, rows, (bt.flags if bt else None)


def S_of(rows):
    inc = [r["reward"] for r in rows if not r["excluded"]]
    return sum(inc) / len(inc) if inc else float("nan")


def delta(runs, X, large, small, seeds, select, rng, B=1000, block=50):
    def value(idx_by_seed):
        def S(v, a):
            rs = [runs[(a, v, s)][i] for s in seeds for i in idx_by_seed[s] if select(runs[(a, v, s)][i])]
            return S_of(rs)
        return (S(X, large) - S("OLD", large)) - (S(X, small) - S("OLD", small))
    full = {s: list(range(len(runs[(large, X, s)]))) for s in seeds}
    point = value(full)
    boots = []
    for _ in range(B):
        idx = {}
        for s in seeds:
            n = len(full[s])
            starts = rng.integers(0, n - block + 1, size=math.ceil(n / block))
            idx[s] = [k for st in starts for k in range(st, st + block)]
        boots.append(value(idx))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return dict(delta=point, ci95=[float(lo), float(hi)])


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [c - h, c + h]


def run(out: Path, workers: int) -> None:
    out.mkdir(parents=True, exist_ok=False)
    gate = json.loads((out.parent / "GATE_RULE.json").read_text())
    seeds, n = gate["seeds"], gate["decisions_per_seed"]
    jobs = [(a, v, s, n) for a in ACTIONS for v in VARIANTS for s in seeds]
    runs, flags = {}, {}
    with mp.get_context("spawn").Pool(workers) as pool:
        for a, v, s, rows, fl in pool.imap_unordered(_job, jobs):
            runs[(a, v, s)] = rows
            flags[f"{a}|{v}|{s}"] = fl
    # CRN check: identical scene and profile streams across every run of a seed
    for s in seeds:
        streams = {tuple((r["scene"], r["profile"]) for r in runs[(a, v, s)]) for a in ACTIONS for v in VARIANTS}
        assert len(streams) == 1, f"seed {s}: exogenous streams differ"
    rng = np.random.default_rng(20261007)
    res = dict(schema="splitfusion.run5_backlog_pair.gate.v1", gate_rule_sha256=None, crn="scene and profile streams identical", exposure={},
               S={}, ranking={}, deltas={}, bpos={}, zero_backlog_identity={}, timing_flags=flags)
    profsel = {p: (lambda r, p=p: r["profile"] == p) for p in PROFILES}
    profsel["ALL"] = lambda r: True
    for a in ACTIONS:
        for v in VARIANTS:
            rows = [r for s in seeds for r in runs[(a, v, s)]]
            res["exposure"][f"{a}|{v}"] = {p: dict(n=sum(f(r) for r in rows), gt1kb=sum(f(r) and r["backlog"] > 1000 for r in rows),
                                                   pos=sum(f(r) and r["backlog"] > 0 for r in rows)) for p, f in profsel.items()}
            res["S"][f"{a}|{v}"] = {p: S_of([r for r in rows if f(r)]) for p, f in profsel.items()}
            bp = [r for r in rows if r["backlog"] > 1000]
            k = sum(r["receipt"] is not None and r["receipt"] <= 170.0 for r in bp)
            res["bpos"][f"{a}|{v}"] = dict(n=len(bp), timely=k / len(bp) if bp else None, wilson=wilson(k, len(bp)),
                                          S=S_of(bp) if bp else None,
                                          by_profile={p: sum(r["profile"] == p for r in bp) for p in PROFILES})
    for v in VARIANTS:
        res["ranking"][v] = {p: sorted(ACTIONS, key=lambda a: -res["S"][f"{a}|{v}"][p]) for p in profsel}
    for X in ("NEW_A", "NEW_B"):
        for large, small in gate["quantities"]["pairs"]:
            for p, f in profsel.items():
                res["deltas"][f"{X}|{large}-{small}|{p}"] = delta(runs, X, large, small, seeds, f, rng)
    # zero-backlog identity between NEW_A and NEW_B (same transition): outcomes may differ only through carry-in
    for a in ACTIONS:
        mism = same_ctx = 0
        for s in seeds:
            for x, y in zip(runs[(a, "NEW_A", s)], runs[(a, "NEW_B", s)]):
                if x["backlog"] == 0 and y["backlog"] == 0 and x["carry"] == y["carry"]:
                    same_ctx += 1
                    mism += (x["receipt"], x["reward"], x["late"]) != (y["receipt"], y["reward"], y["late"])
        res["zero_backlog_identity"][a] = dict(zero_backlog_same_carry=same_ctx, outcome_mismatches=mism)
    # verdict
    g1 = sum(res["exposure"][f"{a}|NEW_A"]["ALL"]["gt1kb"] for a in ACTIONS)
    survive = [k for k, d in res["deltas"].items() if k.startswith("NEW_B|") and d["delta"] <= -0.005 and d["ci95"][1] < 0]
    dA = res["deltas"]["NEW_A|q3000-q5000|ALL"]["delta"]
    dB = res["deltas"]["NEW_B|q3000-q5000|ALL"]["delta"]
    G = dict(G1=dict(gt1kb_pooled=g1, passed=g1 >= 100), G2=dict(surviving=survive, passed=bool(survive)),
             G3=dict(delta_A=dA, delta_B=dB, ratio=dB / dA if dA else None, passed=dA < 0 and dB / dA >= 0.25))
    verdict = ("PROCEED" if all(x["passed"] for x in G.values()) else "STOP_SPARSE" if not G["G1"]["passed"]
               else "STOP_PENALTY_DEPENDENT")
    res["gate"] = dict(checks=G, verdict=verdict)
    import hashlib
    res["gate_rule_sha256"] = hashlib.sha256((out.parent / "GATE_RULE.json").read_bytes()).hexdigest()
    (out / "GATE.json").write_text(json.dumps(res, indent=1, default=str))
    # compact print
    for a in ACTIONS:
        print(a, "exposure B>1KB NEW_A:", {p: res["exposure"][f"{a}|NEW_A"][p]["gt1kb"] for p in profsel})
        print("   S  OLD/NEW_A/NEW_B:", {p: [round(res["S"][f"{a}|{v}"][p], 4) for v in VARIANTS] for p in profsel})
        print("   at B>1KB: NEW_A", {k: res["bpos"][f"{a}|NEW_A"][k] for k in ("n", "timely", "S")}, "NEW_B", {k: res["bpos"][f"{a}|NEW_B"][k] for k in ("n", "timely", "S")})
    print("ranking:", json.dumps(res["ranking"]))
    for k, d in res["deltas"].items():
        print(f"  {k:40s} {d['delta']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]")
    print("zero-backlog identity:", res["zero_backlog_identity"])
    print("GATE:", json.dumps(res["gate"], default=str))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--workers", type=int, default=9)
    a = p.parse_args()
    run(a.out, a.workers)
