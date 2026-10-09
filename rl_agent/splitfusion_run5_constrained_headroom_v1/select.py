"""Choose ONE MCS-stratum rule from TRAINING evidence only (FIT scenes + TRAIN_GCV2, corrected), all 71 grid actions in
support (12 modes), exact linear constraints over all 71^3 stratum combinations. Declared before held-out evaluation:

  maximise  P_train(rule)
  s.t. for every registered requirement (k, class) with tier t:
         if some single combination attains t on training evidence:   retention_k >= t + MARGIN
         else (registered shared-miss rule, comparator = C0 seed mean on the same training evidence):
                                                                       retention_k >= C0_k - 0.01 + MARGIN
  MARGIN = 0.005 (declared selection margin for training-to-held-out noise; thresholds themselves unchanged).

Constraints are exact: loc  sum(hits) - t*ref_recall*sum(eligible) >= 0;  seg  sum(inter) - t*ref_iou*sum(union) >= 0;
precision  sum(tp) - t*ref_prec*sum(tp+fp) >= 0, with the reference quantities action-independent (per-scene cell).
Also reported: the best single fixed action under the same constraints (non-adaptive comparator).
"""
from __future__ import annotations

import itertools

import numpy as np

from . import collect as K
from . import metrics as MT

MARGIN = 0.005
STRATA = ("mcs_9_10", "mcs_11_15", "mcs_ge16")


def sums(a):
    t = a["timely"] > 0.5
    s = {}
    for c in "vp":
        s[f"th_{c}"] = (t * a[f"{c}_hits"]).sum()
        for f in K.FIELDS:
            s[f"{c}_{f}"] = a[f"{c}_{f}"].sum()
            s[f"ref_{c}_{f}"] = a[f"ref_{c}_{f}"].sum()
    s["n"] = len(t)
    s["timely"] = t.sum()
    return s


def stratum_mask(a, st):
    return MT.STRATA[st](a)


def constraint_rows(total_ref):
    """(name, numerator field(s), coefficient on denominator field(s)) for the linear form."""
    out = []
    for (k, cls), tier in MT.TIERS.items():
        c = MT.CLS[cls]
        if k == "loc":
            ref = total_ref[f"ref_{c}_hits"] / total_ref[f"ref_{c}_eligible"]
            out.append((f"{k}_{cls}", tier, lambda s, c=c: s[f"{c}_hits"], lambda s, c=c: s[f"{c}_eligible"], ref))
        elif k == "seg":
            ref = total_ref[f"ref_{c}_inter"] / total_ref[f"ref_{c}_union"]
            out.append((f"{k}_{cls}", tier, lambda s, c=c: s[f"{c}_inter"], lambda s, c=c: s[f"{c}_union"], ref))
        else:
            ref = total_ref[f"ref_{c}_tp"] / (total_ref[f"ref_{c}_tp"] + total_ref[f"ref_{c}_fp"])
            out.append((f"{k}_{cls}", tier, lambda s, c=c: s[f"{c}_tp"], lambda s, c=c: s[f"{c}_tp"] + s[f"{c}_fp"], ref))
    return out


def choose(action_arrays: dict, c0_retention: dict, margin: float = MARGIN, override: dict | None = None) -> dict:
    """action_arrays[(m, q)] = training arrays (both seeds concatenated, identical scene/MCS streams)."""
    acts = sorted(action_arrays)
    per = {st: [sums(MT.subset(action_arrays[a], stratum_mask(action_arrays[a], st))) for a in acts] for st in STRATA}
    any_a = action_arrays[acts[0]]
    total = sums(any_a)
    E = {c: total[f"ref_{c}_eligible"] for c in "vp"}
    cons = constraint_rows(total)
    A = len(acts)
    grid = lambda st, fn: np.array([fn(s) for s in per[st]])
    P = (0.4 * (grid(STRATA[0], lambda s: s["th_v"])[:, None, None] + grid(STRATA[1], lambda s: s["th_v"])[None, :, None]
                + grid(STRATA[2], lambda s: s["th_v"])[None, None, :]) / E["v"]
         + 0.6 * (grid(STRATA[0], lambda s: s["th_p"])[:, None, None] + grid(STRATA[1], lambda s: s["th_p"])[None, :, None]
                  + grid(STRATA[2], lambda s: s["th_p"])[None, None, :]) / E["p"])
    feasible = np.ones((A, A, A), bool)
    detail = {}
    for name, tier, num, den, ref in cons:
        n = [grid(st, num) for st in STRATA]
        d = [grid(st, den) for st in STRATA]
        N = n[0][:, None, None] + n[1][None, :, None] + n[2][None, None, :]
        D = d[0][:, None, None] + d[1][None, :, None] + d[2][None, None, :]
        ret = N / D / ref
        attainable = bool((ret >= tier).any())
        target = tier + margin if attainable else c0_retention[name] - 0.01 + margin
        if override and name in override:                    # labelled decision scenario only (see headroom.py)
            target = override[name] + margin
        feasible &= ret >= target
        detail[name] = dict(tier=tier, attainable_on_training=attainable, target_used=target,
                            max_attainable=float(ret.max()), c0_train=c0_retention[name])
        detail[name]["_ret"] = ret
    if not feasible.any():
        return dict(feasible=False, constraints={k: {x: y for x, y in v.items() if x != "_ret"} for k, v in detail.items()})
    idx = np.unravel_index(np.where(feasible, P, -np.inf).argmax(), P.shape)
    rule = {st: acts[i] for st, i in zip(STRATA, idx)}
    diag = np.array([feasible[i, i, i] for i in range(A)])
    best_fixed = acts[int(np.where(diag, np.array([P[i, i, i] for i in range(A)]), -np.inf).argmax())] if diag.any() else None
    return dict(feasible=True, rule={k: list(v) for k, v in rule.items()}, P_train=float(P[idx]),
                feasible_combinations=int(feasible.sum()), best_fixed_feasible=list(best_fixed) if best_fixed else None,
                best_fixed_P_train=float(P[(acts.index(best_fixed),) * 3]) if best_fixed else None,
                retention_train={k: float(v["_ret"][idx]) for k, v in detail.items()},
                constraints={k: {x: y for x, y in v.items() if x != "_ret"} for k, v in detail.items()},
                margin=margin)


def per_stratum_feasibility(action_arrays: dict, c0_retention: dict) -> dict:
    """TRAINING evidence: could each MCS stratum meet the requirements ON ITS OWN (stratum-level retention vs the
    stratum's own reference)? Reports, per stratum, the best single action meeting every requirement that is attainable
    in that stratum, which requirements are unattainable there, and the P cost relative to the unconstrained best."""
    acts = sorted(action_arrays)
    out = {}
    for st in STRATA:
        rows = []
        for a in acts:
            s = MT.subset(action_arrays[a], stratum_mask(action_arrays[a], st))
            rows.append((a, MT.metrics(s)))
        unattainable = [f"{k}_{c}" for (k, c), t in MT.TIERS.items() if max(m[c]["retention"][k] for _, m in rows) < t]
        ok = [(a, m) for a, m in rows if all(m[c]["retention"][k] >= t for (k, c), t in MT.TIERS.items()
                                             if f"{k}_{c}" not in unattainable)]
        best_any = max(rows, key=lambda r: r[1]["P"])
        best_ok = max(ok, key=lambda r: r[1]["P"]) if ok else None
        out[st] = dict(n=rows[0][1]["n"], unattainable_requirements=unattainable,
                       best_unconstrained=dict(action=list(best_any[0]), P=best_any[1]["P"],
                                               seg_v=best_any[1]["vehicle"]["retention"]["seg"]),
                       best_meeting_attainable=None if best_ok is None else dict(
                           action=list(best_ok[0]), P=best_ok[1]["P"], timely=best_ok[1]["timely"],
                           wire_p50_kb=best_ok[1]["wire_p50_kb"], seg_v=best_ok[1]["vehicle"]["retention"]["seg"]))
    return out
