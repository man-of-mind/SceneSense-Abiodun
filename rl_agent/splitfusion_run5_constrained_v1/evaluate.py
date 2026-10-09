"""Offline comparison: each constrained seed vs the deployed actor on identical scene/channel streams (closed loop).

  python3 -m rl_agent.splitfusion_run5_constrained_v1.evaluate --root <ABS experiments/splitfusion_run5_constrained_v1> --workers 20

Environments E1/E2 (held_scene + HELDOUT_GCV2 / RETENTION_V1) x CORRECTED/OLD x eval seeds 9017/9029/9043 x 6,000
decisions. Quality uses each evaluation set's own reference totals; P from aggregate counts. Acceptance = REGISTRATION.json.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from rl_agent.splitfusion_corrected_latency_env_v1 import collector as CRC
from rl_agent.splitfusion_run5_constrained_headroom_v1 import collect as HK
from rl_agent.splitfusion_run5_constrained_headroom_v1 import metrics as HM

from . import registration as REG

EVAL_SEEDS, N, BLOCK = (9017, 9029, 9043), 6000, 50
PROFILES = ("FAVORABLE_STABLE", "MID_VARIABLE", "ADVERSE_STABLE", "FADE_RECOVERY")
COLS = HK.COLS + ["profile"]


def run(spec, env, variant, seed, n):
    import torch
    from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as orch
    from rl_agent.splitfusion_run5_full_v1 import collector as FC
    from rl_agent.splitfusion_run5_lowmcs_retrain_v1 import campaign as LCAMP
    from rl_agent.splitfusion_run5_lowmcs_retrain_v1 import evaluate as LEV
    from rl_agent.splitfusion_run5_reward_trials_v1 import collector as C
    from rl_agent.splitfusion_run5_reward_trials_v1 import evaluate as RE
    torch.set_num_threads(1)
    FC.compose = LEV._REAL_COMPOSE
    split, pool = HK.ENVS[env]
    shared, latency = LCAMP.components(split=split, pool=pool, corrected=variant == "CORRECTED")
    cat = shared["catalog"]
    col = C.TrialRun5CollectorV1(latency=latency, seed=seed, shared_sources=shared, arm="C0")
    actor = RE.load_actor(REG.ROOT / spec["path"])
    out = np.zeros((n, len(COLS)))
    for i in range(n):
        mode, q = RE.act(actor, col.current_state_features())
        col.collect_transition(orch.ModeledActionRequestV1(decision_ordinal=i, mode_id=mode, q_e4=q,
                                                           source="STOCHASTIC_ACTOR", warmup_q_bin_index=None))
        d = col.diagnostics[-1]
        key = d["reward_scene_key"]
        e = RE.expected(cat.components(key, mode_id=mode, q_e4=q))
        r = RE.expected([(1.0, cat.reference_components(key))])
        v = cat.value_for("C0", key, mode_id=mode, q_e4=q)
        row = [d["receipt_ms"] <= 170.0, d["excluded"], d["prior_ul_mcs"], d["snr_db"], d["backlog_bytes"], d["wire_bytes"],
               q, mode, d["receipt_ms"], np.nan if v is None else v]
        row += [e[f"{c}_{f}"] for c in "vp" for f in HK.FIELDS] + [r[f"{c}_{f}"] for c in "vp" for f in HK.FIELDS]
        row.append(PROFILES.index(d["profile"]))
        out[i] = row
    return out


def job(args):
    key, spec, env, variant, seed, n, out_dir = args
    path = Path(out_dir) / f"{key}__{env}__{variant}__{seed}.npz"
    if not path.exists():
        np.savez_compressed(path, data=run(spec, env, variant, seed, n).astype(np.float32), cols=np.array(COLS))
    return str(path)


def load(d, key, env, var):
    parts = [np.load(Path(d) / f"{key}__{env}__{var}__{s}.npz") for s in EVAL_SEEDS]
    return HM.as_dict(np.concatenate([p["data"] for p in parts]), parts[0]["cols"])


def contributions(a, ev, ep):
    t = a["timely"] > 0.5
    return t * (0.4 * a["v_hits"] / ev + 0.6 * a["p_hits"] / ep)


def paired_dP(a, b, mask=None):
    m = np.ones(len(a["mcs"]), bool) if mask is None else mask
    A, B = HM.subset(a, m), HM.subset(b, m)
    ev, ep = A["v_eligible"].sum(), A["p_eligible"].sum()
    assert np.isclose(ev, B["v_eligible"].sum()) and np.isclose(ep, B["p_eligible"].sum()), "denominators differ"
    d = contributions(A, ev, ep) - contributions(B, ev, ep)
    nb = len(d) // BLOCK
    blocks = d[: nb * BLOCK].reshape(nb, BLOCK).sum(1)
    rng = np.random.default_rng(0)
    sums = blocks[rng.integers(0, nb, (4000, nb))].sum(1) * (len(d) / (nb * BLOCK))
    return dict(dP=float(d.sum()), ci95=[float(np.percentile(sums, 2.5)), float(np.percentile(sums, 97.5))], n=int(len(d)))


def describe(a) -> dict:
    m = HM.metrics(a)
    t = a["timely"] > 0.5
    m.update(q_p10_p50_p90=[float(np.percentile(a["q"], p)) for p in (10, 50, 90)],
             mode11_share=float((a["mode"] == 11).mean()), snr_p50=float(np.median(a["snr"])),
             mcs_p50=float(np.median(a["mcs"])), positive_backlog=int((a["backlog"] > 0).sum()),
             on_time_only={c: HM.metrics(HM.subset(a, t))[c]["retention"] for c in ("vehicle", "person")} if t.any() else None)
    return m


def evaluate(root: Path, workers: int) -> dict:
    reg = REG.verify(root / "registration_01/REGISTRATION.json", seed=17)
    runs = root / "runs_01"
    pol = {}
    for s in REG.SEEDS:
        r = json.loads((runs / f"CON_seed{s}/RUN_RESULT.json").read_text())
        assert r["updates"] == REG.UPDATES and REG.sha(REG.ROOT / r["actor"]["path"]) == r["actor"]["sha256"]
        pol[f"CON_seed{s}"] = dict(path=r["actor"]["path"], sha256=r["actor"]["sha256"])
    assert REG.sha(REG.ROOT / reg["deployed"]["path"]) == reg["deployed"]["sha256"]
    pol["HIST_seed43"] = dict(path=reg["deployed"]["path"], sha256=reg["deployed"]["sha256"])
    out = root / "eval_01"
    out.mkdir(exist_ok=True)
    jobs = [(k, spec, env, var, s, N, str(out)) for k, spec in pol.items() for env in ("E1", "E2")
            for var in ("CORRECTED", "OLD") for s in EVAL_SEEDS]
    with mp.get_context("spawn").Pool(workers) as p:
        list(p.imap_unordered(job, jobs))
    res = {}
    for env in ("E1", "E2"):
        for var in ("CORRECTED", "OLD"):
            arr = {k: load(out, k, env, var) for k in pol}
            dep = arr["HIST_seed43"]
            for k, a in arr.items():
                assert np.array_equal(a["mcs"], dep["mcs"]) and np.array_equal(a["profile"], dep["profile"]), "streams differ"
            block = {}
            for k, a in arr.items():
                e = dict(overall=describe(a))
                for i, prof in enumerate(PROFILES):
                    e[prof] = describe(HM.subset(a, a["profile"] == i))
                for st, f in HM.STRATA.items():
                    if st != "all":
                        e[st] = describe(HM.subset(a, f(a)))
                if k != "HIST_seed43":
                    e["paired_vs_deployed"] = dict(overall=paired_dP(a, dep), **{
                        prof: paired_dP(a, dep, a["profile"] == i) for i, prof in enumerate(PROFILES)}, **{
                        st: paired_dP(a, dep, f(a)) for st, f in HM.STRATA.items() if st != "all"})
                block[k] = e
            res[f"{env}|{var}"] = block
    decision = decide(res)
    doc = dict(schema="splitfusion.run5_constrained_v1.evaluation.v1", registration_sha256=REG.sha(root / "registration_01/REGISTRATION.json"),
               policies=pol, eval_seeds=EVAL_SEEDS, decisions_per_seed=N, results=res, decision=decision,
               evidence_label="RUN5_CONSTRAINED_MODELED", live_claim=False)
    (out / "EVALUATION.json").write_text(json.dumps(doc, indent=1, sort_keys=True, default=float))
    return doc


FLOORS = {("loc", "vehicle"): 0.95, ("loc", "person"): 0.90, ("seg", "vehicle"): 0.85, ("seg", "person"): 0.80,
          ("precision", "vehicle"): 0.95}


def decide(res) -> dict:
    E1 = res["E1|CORRECTED"]
    dep = E1["HIST_seed43"]
    per_seed = {}
    for s in REG.SEEDS:
        k = f"CON_seed{s}"
        c = E1[k]
        pv = c["paired_vs_deployed"]
        q1 = pv["overall"]["dP"] >= 0.010 and pv["overall"]["ci95"][0] > 0
        floors = {f"{m}_{cls}": c["overall"][cls]["retention"][m] for (m, cls) in FLOORS}
        q2 = all(c["overall"][cls]["retention"][m] >= t for (m, cls), t in FLOORS.items())
        pp_c, pp_d = c["overall"]["person"]["retention"]["precision"], dep["overall"]["person"]["retention"]["precision"]
        q3 = pp_c >= pp_d - 0.005
        fav, fdep, fpv = c["FAVORABLE_STABLE"], dep["FAVORABLE_STABLE"], pv["FAVORABLE_STABLE"]
        ret_ok = all(fav[cls]["retention"][m] >= fdep[cls]["retention"][m] - 0.020
                     for m in ("loc", "seg", "precision") for cls in ("vehicle", "person"))
        floors_fav = all(fav[cls]["retention"][m] >= t for (m, cls), t in FLOORS.items())
        q4 = (fpv["dP"] >= -0.005 and fpv["ci95"][0] >= -0.015 and fav["timely"] >= fdep["timely"] - 0.010
              and ret_ok and floors_fav)
        e2 = res["E2|CORRECTED"][k]["paired_vs_deployed"]["overall"]["dP"]
        old = res["E1|OLD"][k]["paired_vs_deployed"]["overall"]["dP"]
        q5 = e2 > 0 and old >= -0.010
        per_seed[k] = dict(Q1=dict(passed=bool(q1), dP=pv["overall"]["dP"], ci95=pv["overall"]["ci95"]),
                           Q2=dict(passed=bool(q2), retention=floors),
                           Q3=dict(passed=bool(q3), candidate=pp_c, deployed=pp_d),
                           Q4=dict(passed=bool(q4), dP=fpv["dP"], ci95=fpv["ci95"], timely=fav["timely"],
                                   deployed_timely=fdep["timely"], retention_within_0_02=bool(ret_ok),
                                   floors_absolute_in_favorable=bool(floors_fav)),
                           Q5=dict(passed=bool(q5), E2_dP=e2, E1_OLD_dP=old),
                           qualifies=bool(q1 and q2 and q3 and q4 and q5))
    qualified = all(v["qualifies"] for v in per_seed.values())
    ps = sorted((E1[f"CON_seed{s}"]["overall"]["P"], s) for s in REG.SEEDS)
    return dict(per_seed=per_seed, configuration_qualifies=qualified,
                recommended_checkpoint=(f"CON_seed{ps[1][1]} u10000" if qualified else None),
                outcome=("RECOMMEND_PREDECLARED_CHECKPOINT_FOR_SEPARATELY_AUTHORISED_LIVE_CHECK" if qualified
                         else "CLOSE_BRANCH_RETAIN_DEPLOYED_ACTOR"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--workers", type=int, default=20)
    a = p.parse_args()
    d = evaluate(a.root, a.workers)
    print(json.dumps(d["decision"], indent=1, default=float))
