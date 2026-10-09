"""Part 3: four controllers on identical held-out scene/channel streams, same environment version and accounting.

  A  deployed actor unchanged (saved retrain evaluation HIST_seed43; this audit's instrumented re-run must match it exactly)
  B  deployed actor restricted to mode 11 at its OWN mode-11 conditional-mean q (this audit, evalb.py)
  C  fixed m11 q6000: the best fixed action under the deployed-parity tiers chosen on TRAINING evidence only (RULE_S2.json
     best_fixed_feasible; saved headroom evaluation BEST_FIXED_S2)
  D  S2 MCS->q rule (RULE_S2.json, training-chosen; saved headroom evaluation RULE_S2)
  ref fixed m11 q7000 (saved retrain evaluation; chosen before evaluation from live D10 evidence, NOT FIT-selected: reference only)

Primary E1-CORRECTED; E2 / OLD sensitivity. Paired block bootstrap over 300-decision blocks within each eval seed.

    env -u PYTHONPATH CUDA_VISIBLE_DEVICES= python3 -m rl_agent.splitfusion_run5_mode_value_audit_v1.compare --out <ABS dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rl_agent.splitfusion_run5_constrained_headroom_v1 import headroom as H
from rl_agent.splitfusion_run5_constrained_headroom_v1 import metrics as MT

from . import evalb as EB

PROFILES = EB.PROFILES
HROOT = H.REG.ROOT / "experiments/splitfusion_run5_constrained_headroom_v1"
BLOCK = 300
NBOOT = 2000
FAV_MARGIN = dict(dP_point=-0.005, dP_ci_low=-0.015, timely=-0.010, retention=-0.020)


def load_mine(root, kind, env, var):
    parts = [np.load(root / "runs_01" / f"{kind}__{env}__{var}__{s}.npz") for s in EB.EVAL_SEEDS]
    a = MT.as_dict(np.concatenate([p["data"] for p in parts]), parts[0]["cols"])
    a["eval_seed"] = np.repeat(EB.EVAL_SEEDS, [len(p["data"]) for p in parts]).astype(float)
    return a


def p_components(a):
    t = a["timely"] > 0.5
    return np.stack([t * a["v_hits"], a["v_eligible"], t * a["p_hits"], a["p_eligible"], t.astype(float)], 1)


def block_sums(a, mask):
    comp = p_components(a) * mask[:, None]
    blocks = []
    for s in EB.EVAL_SEEDS:
        idx = np.where(a["eval_seed"] == s)[0]
        for b in range(0, len(idx), BLOCK):
            blocks.append(comp[idx[b:b + BLOCK]].sum(0))
    return np.array(blocks), np.array([mask[np.where(a["eval_seed"] == s)[0]][b:b + BLOCK].sum()
                                       for s in EB.EVAL_SEEDS for b in range(0, (a["eval_seed"] == s).sum(), BLOCK)])


def P_of(S):
    return 0.4 * S[..., 0] / S[..., 1] + 0.6 * S[..., 2] / S[..., 3]


def paired_ci(a, b, mask, rng):
    Sa, n = block_sums(a, mask)
    Sb, _ = block_sums(b, mask)
    keep = n > 0
    Sa, Sb, n = Sa[keep], Sb[keep], n[keep]
    idx = rng.integers(0, len(Sa), size=(NBOOT, len(Sa)))
    A, B, N = Sa[idx].sum(1), Sb[idx].sum(1), n[idx].sum(1)
    dP = P_of(B) - P_of(A)
    dT = (B[:, 4] - A[:, 4]) / N
    return dict(dP=float(P_of(Sb.sum(0)) - P_of(Sa.sum(0))), dP_ci=[float(np.quantile(dP, .025)), float(np.quantile(dP, .975))],
                dtimely=float((Sb[:, 4].sum() - Sa[:, 4].sum()) / n.sum()),
                dtimely_ci=[float(np.quantile(dT, .025)), float(np.quantile(dT, .975))])


def summary(a):
    m = MT.metrics(a)
    q = a["q"]
    modes, cnt = np.unique(a["mode"], return_counts=True)
    return dict(n=m["n"], P=m["P"], timely=m["timely"], wire_p50_kb=m["wire_p50_kb"], wire_p99_kb=m["wire_p99_kb"],
                R_v=m["vehicle"]["recall_within_tol"], R_p=m["person"]["recall_within_tol"],
                seg_v=m["vehicle"]["seg_iou"], seg_p=m["person"]["seg_iou"], prec_v=m["vehicle"]["precision"],
                prec_p=m["person"]["precision"], retention={f"{k}_{c}": m[c]["retention"][k] for k, c in MT.TIERS},
                tiers_absolute={f"{k}_{c}": bool(m[c]["retention"][k] >= t) for (k, c), t in MT.TIERS.items()},
                mode_share={str(int(k)): float(v / len(q)) for k, v in zip(modes, cnt)},
                q_p5_p50_p95=[float(np.quantile(q, x)) for x in (.05, .5, .95)], q_std=float(q.std()),
                n_mcs_le_10=int((a["mcs"] <= 10).sum()), n_backlog_gt_1kb=int((a["backlog"] > 1000).sum()),
                n_excluded=int((a["excluded"] > 0.5).sum()), realized_C0_reward=m["realized_C0_reward"])


def main(root: Path):
    rng = np.random.default_rng(20261007)
    saved = MT.load_retrain_eval(H.RETRAIN_EVAL)
    out = dict(controllers={"A": "deployed actor bb9b0d25 unchanged (argmax mode, conditional-mean q)",
                            "B": "deployed actor, mode forced to 11, q = its own mode-11 conditional mean on [0, 9791]",
                            "C": "fixed m11 q6000 (RULE_S2.json best_fixed_feasible; TRAINING evidence only)",
                            "D": "S2 rule m11 q7000/q7000/q6000 for prior MCS 9-10/11-15/>=16 (TRAINING evidence only)",
                            "REF_q7000": "fixed m11 q7000 (pre-chosen from live D10 evidence; reference, not FIT-selected)"},
               margins_favorable_vs_A=FAV_MARGIN, settings={})
    for env in ("E1", "E2"):
        for var in ("CORRECTED", "OLD"):
            A_s = saved[("HIST_seed43", env, var)]
            A = load_mine(root, "A", env, var)
            for c in ("mode", "q", "receipt", "timely", "v_hits", "p_hits", "mcs", "wire"):
                assert np.allclose(A[c], A_s[c], equal_nan=True), f"A re-run differs from saved evaluation on {c} ({env},{var})"
            pol = dict(A=A, B=load_mine(root, "B", env, var),
                       C=H.load(HROOT, "BEST_FIXED_S2", env, var, H.EVAL_SEEDS),
                       D=H.load(HROOT, "RULE_S2", env, var, H.EVAL_SEEDS),
                       REF_q7000=saved[("FIXED_m11_q7000", env, var)])
            for k, a in pol.items():
                assert np.array_equal(a["mcs"], A["mcs"]), f"{k}: MCS stream differs"
                a["eval_seed"] = A["eval_seed"]
                a["profile"] = A["profile"]
            res = {}
            for pname in PROFILES + ("ALL",):
                mask = np.ones(len(A["mcs"]), bool) if pname == "ALL" else A["profile"] == PROFILES.index(pname)
                row = {k: summary(MT.subset(a, mask)) for k, a in pol.items()}
                for k in ("B", "C", "D", "REF_q7000"):
                    row[k]["paired_vs_A"] = paired_ci(pol["A"], pol[k], mask.astype(float), rng)
                if pname == "FAVORABLE_STABLE":
                    for k in ("B", "C", "D", "REF_q7000"):
                        pv = row[k]["paired_vs_A"]
                        ni = dict(dP=pv["dP"] >= FAV_MARGIN["dP_point"] and pv["dP_ci"][0] >= FAV_MARGIN["dP_ci_low"],
                                  timely=row[k]["timely"] >= row["A"]["timely"] + FAV_MARGIN["timely"],
                                  **{f"ret_{r}": row[k]["retention"][r] >= row["A"]["retention"][r] + FAV_MARGIN["retention"]
                                     for r in row["A"]["retention"]})
                        row[k]["favorable_noninferior_vs_A"] = dict(checks=ni, all=all(ni.values()))
                res[pname] = row
            out["settings"][f"{env}_{var}"] = res
            print(env, var, "parity A==saved OK")
    with open(root / "PART3_CONTROLLERS.json", "x") as f:
        json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    main(p.parse_args().out)
