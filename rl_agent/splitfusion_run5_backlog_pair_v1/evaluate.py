"""Matched evaluation of CONTROL_s17, CANDIDATE_s17 and the deployed HIST_seed43 on identical held-out schedules.
Reuses the low-MCS retrain evaluation helpers (quality, retention, paired block CI, read-only queue-order
instrumentation). Variants: E1/NEW_A (primary), E1/NEW_B (diagnostic penalty), E1/OLD_A, E2/NEW_A."""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as orch
from rl_agent.splitfusion_run5_full_v1 import collector as FC
from rl_agent.splitfusion_run5_lowmcs_retrain_v1 import evaluate as LE
from rl_agent.splitfusion_run5_lowmcs_retrain_v1 import registration as LREG
from rl_agent.splitfusion_run5_reward_trials_v1 import evaluate as RE
from rl_agent.splitfusion_run5_reward_trials_v1 import rewards as R

from . import env as E

ROOT = LREG.ROOT
PROFILES = ("FAVORABLE_STABLE", "MID_VARIABLE", "ADVERSE_STABLE", "FADE_RECOVERY")
SETTINGS = {"E1|NEW_A": ("HELDOUT_GCV2", True, False), "E1|NEW_B": ("HELDOUT_GCV2", True, True),
            "E1|OLD_A": ("HELDOUT_GCV2", False, False), "E2|NEW_A": ("RETENTION_V1", True, False)}


def policies(runs: Path) -> dict:
    out = {}
    for arm in ("CONTROL", "CANDIDATE"):
        r = json.loads((runs / f"{arm}_seed17/RUN_RESULT.json").read_text())
        assert LREG.sha(ROOT / r["actor"]["path"]) == r["actor"]["sha256"]
        out[f"{arm}_s17"] = dict(path=r["actor"]["path"], sha256=r["actor"]["sha256"])
    assert LREG.sha(ROOT / LREG.DEPLOYED["path"]) == LREG.DEPLOYED["sha256"]
    out["HIST_seed43"] = dict(path=LREG.DEPLOYED["path"], sha256=LREG.DEPLOYED["sha256"])
    return out


def run_policy(label, spec, setting, seed, n):
    torch.set_num_threads(1)
    FC.compose = LE.instrumented_compose
    pool, new_transition, neutral = SETTINGS[setting]
    shared, latency = E.components(split="held_scene", pool=pool, neutralized=neutral)
    correction = LE.CAMP.load_correction()
    cat = shared["catalog"]
    bt = E.transition() if new_transition else None
    col = E.TrialBacklogCollectorV1(latency=latency, seed=seed, shared_sources=shared, backlog_transition=bt)
    actor = RE.load_actor(ROOT / spec["path"])
    rows = []
    for i in range(n):
        mode, q = RE.act(actor, col.current_state_features())
        col.collect_transition(orch.ModeledActionRequestV1(decision_ordinal=i, mode_id=mode, q_e4=q,
                                                           source="STOCHASTIC_ACTOR", warmup_q_bin_index=None))
        d = col.diagnostics[-1]
        key = d["reward_scene_key"]
        flip = dict(LE.LAST)
        row = dict(i=i, mode=mode, q=q, terminal=d["terminal"], excluded=d["excluded"], receipt=d["receipt_ms"],
                   timely=d["receipt_ms"] is not None and d["receipt_ms"] <= 170.0, profile=d["profile"],
                   snr=d["snr_db"], mcs=d["prior_ul_mcs"], backlog=d["backlog_bytes"], wire=d["wire_bytes"], scene=key,
                   q_perc=d["q_perc"], late_uplink=d["uplink_late"],
                   flags=list(correction.flags(backlog_bytes=d["backlog_bytes"], wire_bytes=d["wire_bytes"],
                                               prior_ul_mcs=d["prior_ul_mcs"])),
                   flip_gain=flip["flip_gain"], envelope=flip["envelope"], timely_env=flip["envelope"] <= 170.0,
                   value={a: cat.value_for(a, key, mode_id=mode, q_e4=q) for a in R.ARMS})
        row.update(RE.expected(cat.components(key, mode_id=mode, q_e4=q)))
        row.update({f"ref_{k}": v for k, v in RE.expected([(1.0, cat.reference_components(key))]).items()})
        rows.append(row)
    return rows, (dict(bt.flags) if bt else None)


def _job(args):
    label, spec, setting, seed, n = args
    rows, flags = run_policy(label, spec, setting, seed, n)
    return label, setting, seed, rows, flags


def summary(rows) -> dict:
    out = dict(all=LE.stratum(rows))
    for p in PROFILES:
        out[p] = LE.stratum([r for r in rows if r["profile"] == p])
    pos = [r for r in rows if r["backlog"] > 1000]
    out["backlog_gt1kb"] = dict(n=len(pos), timely=float(np.mean([r["timely"] for r in pos])) if pos else None,
                                by_profile={p: sum(r["profile"] == p for r in pos) for p in PROFILES},
                                any_positive=sum(r["backlog"] > 0 for r in rows))
    return out


def evaluate(runs: Path, out: Path, n: int, workers: int) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    reg = json.loads((runs.parent / "REGISTRATION.json").read_text())
    pol = policies(runs)
    seeds = reg["evaluation"]["eval_seeds"]
    jobs = [(l, s_, st, sd, n) for l, s_ in pol.items() for st in SETTINGS for sd in seeds]
    raw, flags = collections.defaultdict(dict), {}
    with mp.get_context("spawn").Pool(workers) as pool:
        for label, st, sd, rows, fl in pool.imap_unordered(_job, jobs):
            raw[(label, st)][sd] = rows
            flags[f"{label}|{st}|{sd}"] = fl
    with gzip.open(out / "EVAL_DECISIONS.jsonl.gz", "wt") as f:
        for (label, st), per in sorted(raw.items()):
            for sd, rows in sorted(per.items()):
                for r in rows:
                    f.write(json.dumps(dict(r, policy=label, setting=st, eval_seed=sd), separators=(",", ":")) + "\n")
    for st in SETTINGS:
        for sd in seeds:
            assert len({tuple(r["scene"] for r in raw[(l, st)][sd]) for l in pol}) == 1, "scene streams differ"
            assert len({tuple(r["profile"] for r in raw[(l, st)][sd]) for l in pol}) == 1, "profile streams differ"
    cat = lambda l, st: [r for sd in seeds for r in raw[(l, st)][sd]]
    results = {l: {st: summary(cat(l, st)) for st in SETTINGS} for l in pol}
    pairs = {}
    for st in SETTINGS:
        for a, b in (("CANDIDATE_s17", "CONTROL_s17"), ("CANDIDATE_s17", "HIST_seed43"), ("CONTROL_s17", "HIST_seed43")):
            A, B = cat(a, st), cat(b, st)
            entry = dict(all=LE.paired(A, B))
            for p in PROFILES:
                entry[p] = LE.paired(A, B, lambda x, y, p=p: x["profile"] == p)
            n_pos = sum(1 for x, y in zip(A, B) if x["backlog"] > 0 or y["backlog"] > 0)
            entry["either_backlog_pos"] = (LE.paired(A, B, lambda x, y: x["backlog"] > 0 or y["backlog"] > 0)
                                           if n_pos else dict(n=0, note="no positive-backlog decision in either policy"))
            pairs[f"{st}|{a}-{b}"] = entry
    doc = dict(schema="splitfusion.run5_backlog_pair.evaluation.v1", policies=pol, settings=SETTINGS, eval_seeds=seeds,
               decisions_per_seed=n, results=results, paired=pairs, timing_flags=flags,
               decision=decide(results, pairs, reg), live_claim=False,
               registration_sha256=hashlib.sha256((runs.parent / "REGISTRATION.json").read_bytes()).hexdigest())
    (out / "EVALUATION.json").write_text(json.dumps(doc, indent=1, sort_keys=True, default=str))
    return doc


def summarise_saved(runs: Path, out: Path) -> dict:
    """Rebuild every summary from EVAL_DECISIONS.jsonl.gz (written by evaluate before summarising)."""
    reg = json.loads((runs.parent / "REGISTRATION.json").read_text())
    pol = policies(runs)
    raw = collections.defaultdict(lambda: collections.defaultdict(list))
    with gzip.open(out / "EVAL_DECISIONS.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            raw[(r.pop("policy"), r.pop("setting"))][r.pop("eval_seed")].append(r)
    seeds = reg["evaluation"]["eval_seeds"]
    for key, per in raw.items():
        for sd in seeds:
            per[sd].sort(key=lambda r: r["i"])
            assert len(per[sd]) == 6000, (key, sd, len(per[sd]))
    for st in SETTINGS:
        for sd in seeds:
            assert len({tuple(r["scene"] for r in raw[(l, st)][sd]) for l in pol}) == 1, "scene streams differ"
            assert len({tuple(r["profile"] for r in raw[(l, st)][sd]) for l in pol}) == 1, "profile streams differ"
    cat = lambda l, st: [r for sd in seeds for r in raw[(l, st)][sd]]
    results = {l: {st: summary(cat(l, st)) for st in SETTINGS} for l in pol}
    pairs = {}
    for st in SETTINGS:
        for a, b in (("CANDIDATE_s17", "CONTROL_s17"), ("CANDIDATE_s17", "HIST_seed43"), ("CONTROL_s17", "HIST_seed43")):
            A, B = cat(a, st), cat(b, st)
            entry = dict(all=LE.paired(A, B))
            for p in PROFILES:
                entry[p] = LE.paired(A, B, lambda x, y, p=p: x["profile"] == p)
            n_pos = sum(1 for x, y in zip(A, B) if x["backlog"] > 0 or y["backlog"] > 0)
            entry["either_backlog_pos"] = (LE.paired(A, B, lambda x, y: x["backlog"] > 0 or y["backlog"] > 0)
                                           if n_pos else dict(n=0, note="no positive-backlog decision in either policy"))
            pairs[f"{st}|{a}-{b}"] = entry
    doc = dict(schema="splitfusion.run5_backlog_pair.evaluation.v1", policies=pol, settings=SETTINGS, eval_seeds=seeds,
               decisions_per_seed=6000, results=results, paired=pairs, decision=decide(results, pairs, reg),
               live_claim=False, rebuilt_from="EVAL_DECISIONS.jsonl.gz (simulation complete; first summary pass "
                                              "failed on an empty positive-backlog subset)",
               registration_sha256=hashlib.sha256((runs.parent / "REGISTRATION.json").read_bytes()).hexdigest())
    (out / "EVALUATION.json").write_text(json.dumps(doc, indent=1, sort_keys=True, default=str))
    return doc


def decide(results, pairs, reg) -> dict:
    g = lambda l, st="E1|NEW_A", k="all": results[l][st][k]
    p1 = pairs["E1|NEW_A|CANDIDATE_s17-CONTROL_s17"]
    c = {}
    ci = p1["all"]["ci95"]
    c["S1"] = dict(dP=p1["all"]["dP"], ci95=ci, passed=p1["all"]["dP"] >= 0.010 and ci is not None and ci[0] > 0)
    fav = p1["FAVORABLE_STABLE"]
    cf, kf = g("CANDIDATE_s17", k="FAVORABLE_STABLE"), g("CONTROL_s17", k="FAVORABLE_STABLE")
    ret_ok = {f"{m}_{cls}": cf["retention"][cls][m] >= kf["retention"][cls][m] - 0.020
              for cls in ("vehicle", "person") for m in ("loc", "seg", "precision")}
    c["S2"] = dict(dP=fav["dP"], ci95=fav["ci95"], timely=[cf["timely"], kf["timely"]], retention_ok=ret_ok,
                   passed=fav["dP"] >= -0.005 and fav["ci95"] is not None and fav["ci95"][0] >= -0.015
                   and cf["timely"] >= kf["timely"] - 0.010 and all(ret_ok.values()))
    guard = {}
    for k, tiers in LREG.GUARDRAILS.items():
        for cls, tier in tiers.items():
            x, y = g("CANDIDATE_s17")["retention"][cls][k], g("CONTROL_s17")["retention"][cls][k]
            guard[f"{k}_{cls}"] = dict(candidate=x, control=y, tier=tier, passed=x >= tier or (y < tier and x >= y - 0.01))
    c["S3"] = dict(detail=guard, passed=all(v["passed"] for v in guard.values()))
    c["S4"] = dict(candidate=g("CANDIDATE_s17")["timely"], control=g("CONTROL_s17")["timely"],
                   passed=g("CANDIDATE_s17")["timely"] >= g("CONTROL_s17")["timely"] - 0.02)
    sens = {st: pairs[f"{st}|CANDIDATE_s17-CONTROL_s17"]["all"]["dP"] for st in ("E1|NEW_B", "E1|OLD_A", "E2|NEW_A")}
    c["S5"] = dict(dP=sens, passed=all(v > 0 for v in sens.values()))
    c["S6"] = dict(flagged=g("CANDIDATE_s17")["flagged"], passed=g("CANDIDATE_s17")["flagged"] <= 0.05)
    verdict = "REPLICATION_JUSTIFIED" if all(v["passed"] for v in c.values()) else "NO_EVIDENCE_FOR_REPLICATION"
    return dict(checks=c, verdict=verdict, note=reg["success"]["note"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--decisions", type=int, default=6000)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--summarise-saved", action="store_true")
    a = p.parse_args()
    d = summarise_saved(a.runs, a.out) if a.summarise_saved else evaluate(a.runs, a.out, a.decisions, a.workers)
    print(json.dumps(d["decision"], indent=1, default=str))
