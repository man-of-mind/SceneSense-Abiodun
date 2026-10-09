"""Driver. Steps (create-only outputs in <root>):

  select    TRAINING evidence only -> RULE.json (frozen before any held-out evaluation of the rule)
  evaluate  the frozen rule and the best feasible fixed action, closed loop, E1/E2 x CORRECTED/OLD x 3 eval seeds
  analyze   identical-requirement tables, privileged diagnostics (E1-CORRECTED seed 9017 only), SNR x MCS table, figure

    python3 -m rl_agent.splitfusion_run5_constrained_headroom_v1.headroom --root <ABS dir> --step select|evaluate|analyze
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from rl_agent.splitfusion_run5_lowmcs_retrain_v1 import registration as REG

from . import batch as B
from . import collect as K
from . import metrics as MT
from . import select as SEL
from . import snr_check as SNR

EVAL_SEEDS = (9017, 9029, 9043)
RETRAIN_EVAL = REG.ROOT / "experiments/splitfusion_run5_lowmcs_retrain_v1/eval_01/EVAL_DECISIONS.jsonl.gz"
POLICIES = ["HIST_seed43", "C0_seed17", "C0_seed29", "C0_seed43", "NEW_seed17", "NEW_seed29", "NEW_seed43",
            "FIXED_m11_q7000", "FIXED_m11_q8000"]


def load(root, key, env, var, seeds):
    parts = [np.load(root / "runs_01" / f"{key}__{env}__{var}__{s}.npz") for s in seeds]
    return MT.as_dict(np.concatenate([p["data"] for p in parts]), parts[0]["cols"])


def step_select(root):
    acts = {a: load(root, f"A_m{a[0]}_q{a[1]}", "TRAIN", "CORRECTED", B.TRAIN_SEEDS) for a in K.ACTIONS}
    c0 = [MT.metrics(load(root, f"C0_seed{s}", "TRAIN", "CORRECTED", B.TRAIN_SEEDS)) for s in REG.SEEDS]
    c0_ret = {f"{k}_{c}": float(np.mean([m[c]["retention"][k] for m in c0])) for k, c in MT.TIERS}
    choice = SEL.choose(acts, c0_ret)
    choice["per_stratum_feasibility_train"] = SEL.per_stratum_feasibility(acts, c0_ret)
    choice.update(evidence="TRAINING ONLY: FIT scenes + TRAIN_GCV2, CORRECTED, eval seeds 8017/8029, 71 grid actions",
                  c0_train_retention=c0_ret, declared_before_heldout=True)
    out = root / "RULE.json"
    with open(out, "x") as f:
        json.dump(choice, f, indent=1, sort_keys=True)
    return choice


def step_select_s2(root):
    """DECISION SCENARIO S2 (training evidence only; declared before held-out evaluation): the five registered tiers the
    deployed actor attains stay absolute; person precision is held at NON-REGRESSION vs the deployed actor's own
    training-evidence retention instead of the registered 0.90 (which S1 / RULE.json enforces). Not a lowered threshold:
    it quantifies the cost of the 0.90 person-precision requirement for a decision by Abiodun."""
    acts = {a: load(root, f"A_m{a[0]}_q{a[1]}", "TRAIN", "CORRECTED", B.TRAIN_SEEDS) for a in K.ACTIONS}
    c0 = [MT.metrics(load(root, f"C0_seed{s}", "TRAIN", "CORRECTED", B.TRAIN_SEEDS)) for s in REG.SEEDS]
    c0_ret = {f"{k}_{c}": float(np.mean([m[c]["retention"][k] for m in c0])) for k, c in MT.TIERS}
    hist = MT.metrics(load(root, "HIST_seed43", "TRAIN", "CORRECTED", B.TRAIN_SEEDS))
    override = {"precision_person": hist["person"]["retention"]["precision"]}
    choice = SEL.choose(acts, c0_ret, override=override)
    choice.update(scenario="S2_DEPLOYED_PARITY_PERSON_PRECISION", override=override,
                  deployed_train_retention={f"{k}_{c}": hist[c]["retention"][k] for k, c in MT.TIERS},
                  evidence="TRAINING ONLY", declared_before_heldout=True)
    with open(root / "RULE_S2.json", "x") as f:
        json.dump(choice, f, indent=1, sort_keys=True)
    return choice


def step_evaluate(root, workers):
    rule = json.loads((root / "RULE.json").read_text())
    r2 = json.loads((root / "RULE_S2.json").read_text())
    specs = {"RULE_MCS3": dict(kind="rule", rule={k: tuple(v) for k, v in rule["rule"].items()}),
             "BEST_FIXED_FEASIBLE": dict(kind="fixed", mode=rule["best_fixed_feasible"][0], q=rule["best_fixed_feasible"][1]),
             "RULE_S2": dict(kind="rule", rule={k: tuple(v) for k, v in r2["rule"].items()}),
             "BEST_FIXED_S2": dict(kind="fixed", mode=r2["best_fixed_feasible"][0], q=r2["best_fixed_feasible"][1])}
    js = [(k, spec, env, var, s, B.N, str(root / "runs_01")) for k, spec in specs.items()
          for env in ("E1", "E2") for var in ("CORRECTED", "OLD") for s in EVAL_SEEDS]
    with mp.get_context("spawn").Pool(workers) as pool:
        list(pool.imap_unordered(K.job, js))


def oracle(acts, ref_e, targets, iters=3000):
    """PRIVILEGED per-decision hindsight choice (knows each decision's realised latency and scene quality under every
    action) maximising P subject to the same aggregate constraints, by Lagrangian dual subgradient. Diagnostic bound only."""
    keys = sorted(acts)
    t = np.stack([acts[a]["timely"] > 0.5 for a in keys], 1)
    val = np.stack([0.4 * acts[a]["v_hits"] / ref_e["v"] + 0.6 * acts[a]["p_hits"] / ref_e["p"] for a in keys], 1) * t
    g = []
    for name, (tier_target, num, den, ref) in targets.items():
        G = np.stack([num(acts[a]) - tier_target * ref * den(acts[a]) for a in keys], 1)
        g.append(G / max(1e-9, np.abs(G).sum(1).mean()))
    g = np.stack(g, 2)                                          # (decision, action, constraint)
    lam = np.zeros(g.shape[2])
    best = None
    for it in range(iters):
        score = val + (g * lam).sum(2)
        pick = score.argmax(1)
        gs = g[np.arange(len(pick)), pick].sum(0)
        if (gs >= 0).all():
            pv = val[np.arange(len(pick)), pick].sum()
            if best is None or pv > best[0]:
                best = (pv, pick.copy())
        lam = np.maximum(0.0, lam - (0.5 / np.sqrt(it + 1)) * gs / len(pick))
    return keys, best


def step_analyze(root):
    from rl_agent.splitfusion_run5_constrained_headroom_v1 import figure as FG
    rule = json.loads((root / "RULE.json").read_text())
    ev = MT.load_retrain_eval(RETRAIN_EVAL)
    out = {}
    tables = {}
    for env in ("E1", "E2"):
        for var in ("CORRECTED", "OLD"):
            arrays = {p: ev[(p, env, var)] for p in POLICIES}
            for k in ("RULE_MCS3", "BEST_FIXED_FEASIBLE", "RULE_S2", "BEST_FIXED_S2"):
                arrays[k] = load(root, k, env, var, EVAL_SEEDS)
            for p in POLICIES:
                assert np.array_equal(arrays[p]["mcs"], arrays["RULE_MCS3"]["mcs"]), "MCS streams differ"
            m = {p: {st: MT.metrics(MT.subset(a, f(a))) for st, f in MT.STRATA.items()} for p, a in arrays.items()}
            c0 = {(k, c): float(np.mean([m[f"C0_seed{s}"]["all"][c]["retention"][k] for s in REG.SEEDS])) for k, c in MT.TIERS}
            att = {p: MT.attainment(m[p]["all"], c0) for p in arrays}
            tables[f"{env}|{var}"] = dict(metrics=m, attainment=att)
    out["tables"] = tables
    # privileged diagnostics on E1-CORRECTED seed 9017 only (all 71 fixed actions on identical streams)
    acts = {a: load(root, f"A_m{a[0]}_q{a[1]}", "E1", "CORRECTED", (B.ORACLE_SEED,)) for a in K.ACTIONS}
    t9017 = {p: MT.metrics(MT.subset(ev[(p, "E1", "CORRECTED")], ev[(p, "E1", "CORRECTED")]["eval_seed"] == B.ORACLE_SEED))
             for p in POLICIES}
    t9017["RULE_MCS3"] = MT.metrics(load(root, "RULE_MCS3", "E1", "CORRECTED", (B.ORACLE_SEED,)))
    c0_h = {f"{k}_{c}": float(np.mean([t9017[f"C0_seed{s}"][c]["retention"][k] for s in REG.SEEDS])) for k, c in MT.TIERS}
    bound = SEL.choose(acts, c0_h, margin=0.0)                   # rule class, selected IN-SAMPLE on held-out
    any_a = acts[K.ACTIONS[0]]
    total = SEL.sums(any_a)
    targets = {}
    for name, tier, num, den, ref in SEL.constraint_rows(total):
        att = bound["constraints"][name]["attainable_on_training"]
        targets[name] = (tier if att else c0_h[name] - 0.01, num, den, ref)
    keys, best = oracle(acts, {c: total[f"ref_{c}_eligible"] for c in "vp"}, targets)
    if best is not None:
        pick = best[1]
        sel = {c: np.array([acts[keys[j]][c][i] for i, j in enumerate(pick)]) for c in any_a}
        orc = MT.metrics(sel)
        orc["action_share_top"] = sorted(((int((pick == j).sum()), list(keys[j])) for j in set(pick.tolist())), reverse=True)[:8]
    else:
        orc = None
    fixed = {f"m{a[0]}_q{a[1]}": MT.metrics(acts[a]) for a in K.ACTIONS}
    t9017["RULE_S2"] = MT.metrics(load(root, "RULE_S2", "E1", "CORRECTED", (B.ORACLE_SEED,)))
    dep = {f"{k}_{c}": t9017["HIST_seed43"][c]["retention"][k] for k, c in MT.TIERS}
    parity = {n: v for n, v in dep.items() if v < MT.TIERS[tuple(n.split("_"))]}
    parity_bound = SEL.choose(acts, c0_h, margin=0.0, override=parity)
    out["privileged_E1_CORRECTED_seed9017"] = dict(
        status="DIAGNOSTIC BOUNDS ONLY (privileged: selection on the evaluation data itself / hindsight per decision)",
        policies=t9017, rule_class_bound=bound, hindsight_oracle=orc,
        deployed_parity_overrides=parity, rule_class_bound_deployed_parity=parity_bound, fixed_actions=fixed, c0_heldout_retention=c0_h)
    out["snr_mcs_joint_E1_CORRECTED"] = SNR.joint_table(ev, ["HIST_seed43", "C0_seed17", "NEW_seed17"]
                                                      ) | SNR.joint_table({}, [("RULE_MCS3", load(root, "RULE_MCS3", "E1", "CORRECTED", EVAL_SEEDS))])
    out["feature_maps"] = SNR.feature_maps()
    out["rule"] = rule
    (root / "HEADROOM.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    FG.draw(out, root)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--step", required=True, choices=("select", "select_s2", "evaluate", "analyze"))
    p.add_argument("--workers", type=int, default=20)
    a = p.parse_args()
    if a.step == "select":
        print(json.dumps({k: v for k, v in step_select(a.root).items() if k != "constraints"}, indent=1, default=str))
        print(json.dumps(json.loads((a.root / "RULE.json").read_text())["constraints"], indent=1))
    elif a.step == "select_s2":
        print(json.dumps({k: v for k, v in step_select_s2(a.root).items() if k not in ("constraints", "per_stratum_feasibility_train")}, indent=1, default=str))
    elif a.step == "evaluate":
        step_evaluate(a.root, a.workers)
    else:
        step_analyze(a.root)
