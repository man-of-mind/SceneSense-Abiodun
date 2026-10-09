"""Registration (create-only, before training) for the ONE constrained candidate.

    python3 -m rl_agent.splitfusion_run5_constrained_v1.registration --out <ABS dir>

All training numbers come from FIT evidence only:
  * reference constants rho/pi/kappa: FIT catalogue reference cells;
  * targets: the five registered floors + TRAIN_MARGIN; person precision = the DEPLOYED actor's FIT-evidence retention
    (corrected training environment, FIT scenes, TRAIN_GCV2, eval seeds 8017/8029; existing headroom runs) + TRAIN_MARGIN;
  * lambda0: exact Lagrangian dual of max mean(D*Y) s.t. retention_k >= target_k over the 71^3 MCS-stratum rule class on
    the same FIT runs (projected subgradient, fixed 3000 iterations).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_models as RM
from rl_agent.splitfusion_run5_constrained_headroom_v1 import collect as HK
from rl_agent.splitfusion_run5_constrained_headroom_v1 import metrics as HM
from rl_agent.splitfusion_run5_full_v1 import support as S
from rl_agent.splitfusion_run5_lowmcs_retrain_v1 import registration as LREG

from . import objective as O

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
SCHEMA = "splitfusion.run5_constrained_v1.registration.v1"
SEEDS, UPDATES, SMOKE = (17, 29, 43), 10_000, (17, 250)
SOURCES = ("__init__.py", "objective.py", "trainer.py", "registration.py", "campaign.py")
HEADROOM_RUNS = ROOT / "experiments/splitfusion_run5_constrained_headroom_v1/runs_01"
FIT_SEEDS = (8017, 8029)
STRATA = ("mcs_9_10", "mcs_11_15", "mcs_ge16")

ACCEPTANCE = {
    "population": "E1-CORRECTED (held-out scenes + HELDOUT_GCV2, frozen candidate latency), 3 eval seeds x 6,000 decisions, "
                  "identical scene/channel streams for candidate and deployed (asserted); reference-relative quality from "
                  "THIS evaluation set's actual reference totals; P from aggregate counts",
    "Q1_overall_gain": "P(candidate) - P(deployed) >= +0.010 and its paired block-bootstrap 95% CI lower bound > 0",
    "Q2_five_floors": "absolute: loc v >= 0.95, loc p >= 0.90, seg v >= 0.85, seg p >= 0.80, prec v >= 0.95 (no shared-miss exception)",
    "Q3_person_precision": "person precision retention(candidate) >= person precision retention(deployed) - 0.005 on the same streams",
    "Q4_favorable_preservation": "FAVORABLE_STABLE profile only: dP >= -0.005 and paired CI lower bound >= -0.015; "
                                 "timeliness >= deployed - 0.010; each of the six retentions >= deployed's - 0.020 "
                                 "(and the five floors still absolute within FAVORABLE_STABLE)",
    "Q5_sensitivity": "E2-CORRECTED dP > 0 and E1-OLD dP >= -0.010",
    "qualification": "the CONFIGURATION qualifies only if all three seeds pass Q1-Q5",
    "checkpoint_if_qualified": "u10000 actor of the seed with the median E1-CORRECTED P; recommended only for a separately "
                               "authorised live check",
    "if_not_qualified": "close this training branch, retain the deployed actor, no further tuning round",
    "reported_not_gated": "per-profile timeliness/P/recall/retention/payload P50-P99/mode-q with SNR and MCS; where the "
                          "poor-channel gains occur; positive-backlog strata (state 'untested' if absent)",
}
ACCOUNTING = {
    "replay": "non-excluded transitions only (unchanged); minibatch reward = r^P + lambda . g/kappa at sampling time; "
              "r^P = D*Y on SUCCESS, 0 on TIMEOUT (late undefined is TIMEOUT); timely undefined is excluded + session reset",
    "dual": "stochastic decisions only (warm-up excluded); window = last 2,000 stochastic decisions INCLUDING excluded ones "
            "(registered retention is all-scored); update every 400 stochastic decisions (= every 100 updates)",
    "evaluation": "all decisions in latency/timeliness; quality counts over all decisions (all-scored) and separately "
                  "on-time-only; P = (0.4 sum(timely*hits_v)/sum(eligible_v) + 0.6 sum(timely*hits_p)/sum(eligible_p))",
}


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_digest() -> str:
    return hashlib.sha256(json.dumps({n: sha(PACKAGE / n) for n in SOURCES}, sort_keys=True).encode()).hexdigest()


def _load(key):
    parts = [np.load(HEADROOM_RUNS / f"{key}__TRAIN__CORRECTED__{s}.npz") for s in FIT_SEEDS]
    return HM.as_dict(np.concatenate([p["data"] for p in parts]), parts[0]["cols"])


def deployed_fit_person_precision() -> float:
    return float(HM.metrics(_load("HIST_seed43"))["person"]["retention"]["precision"])


def lambda0(targets: dict, iters: int = 3000) -> tuple[list, dict]:
    acts = sorted(HK.ACTIONS)
    arrays = {a: _load(f"A_m{a[0]}_q{a[1]}") for a in acts}
    first = arrays[acts[0]]
    masks = {st: HM.STRATA[st](first) for st in STRATA}
    n_incl = float((first["excluded"] < 0.5).sum())
    J = []
    for st in STRATA:
        vals = []
        for a in acts:
            x = HM.subset(arrays[a], masks[st])
            ev, ep = x["v_eligible"], x["p_eligible"]
            den = 0.4 * (ev > 0) + 0.6 * (ep > 0)
            y = np.where(den > 0, (0.4 * np.where(ev > 0, x["v_hits"] / np.maximum(ev, 1e-12), 0)
                                   + 0.6 * np.where(ep > 0, x["p_hits"] / np.maximum(ep, 1e-12), 0)) / np.maximum(den, 1e-12), 0)
            vals.append(float(((x["timely"] > 0.5) * (x["excluded"] < 0.5) * y).sum()) / n_incl)
        J.append(np.array(vals))
    Jc = J[0][:, None, None] + J[1][None, :, None] + J[2][None, None, :]
    from rl_agent.splitfusion_run5_constrained_headroom_v1 import select as HS
    total = HS.sums(first)
    rows = {name: (num, den, ref) for name, _t, num, den, ref in HS.constraint_rows(total)}
    per = {st: [HS.sums(HM.subset(arrays[a], masks[st])) for a in acts] for st in STRATA}
    R = []
    for name in O.CONSTRAINTS:
        num, den, ref = rows[name]
        n = [np.array([num(s) for s in per[st]]) for st in STRATA]
        d = [np.array([den(s) for s in per[st]]) for st in STRATA]
        N = n[0][:, None, None] + n[1][None, :, None] + n[2][None, None, :]
        D = d[0][:, None, None] + d[1][None, :, None] + d[2][None, None, :]
        R.append((N / D / ref - targets[name]).ravel())
    R = np.stack(R, 1)
    Jf = Jc.ravel()
    lam, best = np.zeros(len(O.CONSTRAINTS)), (np.inf, None)
    for it in range(iters):
        score = Jf + R @ lam
        i = int(score.argmax())
        if score[i] < best[0]:
            best = (float(score[i]), lam.copy())
        lam = np.clip(lam - (0.5 / np.sqrt(it + 1)) * R[i], 0.0, O.LAMBDA_MAX)
    lam_star = best[1]
    i = int((Jf + R @ lam_star).argmax())
    feas = (R >= 0).all(1)
    j = int(np.where(feas, Jf, -np.inf).argmax())
    unr = np.unravel_index(j, Jc.shape)
    return lam_star.tolist(), dict(dual_value=best[0], primal_best_feasible_J=float(Jf[j]),
                                   primal_best_feasible_rule={st: list(acts[k]) for st, k in zip(STRATA, unr)},
                                   lagrangian_argmax_margins=R[i].tolist())


def document() -> dict:
    from rl_agent.splitfusion_run5_reward_trials_v1 import catalog as K
    probe = O.LocCatalogV1("fit", "0" * 64)
    constants = O.fit_constants(probe)
    targets = {k: O.REGISTERED_TIERS[k] + O.TRAIN_MARGIN for k in O.CONSTRAINTS}
    dep_pp = deployed_fit_person_precision()
    targets["precision_person"] = dep_pp + O.TRAIN_MARGIN
    obj = O.definition(dict(fit_reference=constants, targets=targets))
    obj_sha = O.sha(obj)
    lam0, lam_diag = lambda0(targets)
    lreg = LREG.document()
    return dict(
        schema=SCHEMA, objective=obj, objective_sha256=obj_sha, targets=targets,
        target_derivation=dict(five_floors=f"registered tier + {O.TRAIN_MARGIN} (FIT-only training margin)",
                               person_precision=f"deployed FIT-evidence retention {dep_pp:.4f} + {O.TRAIN_MARGIN}",
                               evaluation_is_separate="held-out attainment and deployed comparison decide; FIT targets prove nothing held-out"),
        lambda0=lam0, lambda0_diagnostics=lam_diag, seeds=list(SEEDS), updates=UPDATES, smoke=list(SMOKE),
        source_digest=source_digest(), evaluation_source_sha256=sha(PACKAGE / "evaluate.py"),
        model_binding_sha256=RM.RUN5_TRAINING_MODEL_BINDING_SHA256,
        support_sha256=S.SUPPORT_SHA256, latency=dict(candidate=lreg["candidate"], binding_02=lreg["binding_02"],
                                                      t_v2b_sha256=lreg["t_v2b_sha256"], lowmcs_sources=lreg["lowmcs_sources"]),
        reward_trials_source_digest=lreg["reward_trials_source_digest"], deployed=LREG.DEPLOYED,
        comparators_context=LREG.COMPARATORS, acceptance=ACCEPTANCE, accounting=ACCOUNTING,
        fit_evidence=str(HEADROOM_RUNS.relative_to(ROOT)), checkpoints=[0, 250, 2500, 5000, 7500, 10000],
        scope="one configuration; no sweep, transport refit, live capture or deployment")


class RegistrationError(RuntimeError):
    pass


def verify(path: Path, *, seed: int) -> dict:
    doc = json.loads(Path(path).read_text())
    if doc.get("schema") != SCHEMA:
        raise RegistrationError("foreign registration")
    if doc["source_digest"] != source_digest():
        raise RegistrationError("training source changed after registration")
    lreg = LREG.document()
    if doc["latency"]["candidate"] != lreg["candidate"] or doc["latency"]["binding_02"] != lreg["binding_02"]:
        raise RegistrationError("latency evidence changed")
    if doc["model_binding_sha256"] != RM.RUN5_TRAINING_MODEL_BINDING_SHA256 or doc["support_sha256"] != S.SUPPORT_SHA256:
        raise RegistrationError("model or support changed")
    if O.sha(doc["objective"]) != doc["objective_sha256"]:
        raise RegistrationError("objective document altered")
    if seed not in doc["seeds"]:
        raise RegistrationError("seed outside scope")
    return doc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    out = p.parse_args().out
    if not out.is_absolute():
        raise SystemExit("absolute --out required")
    out.mkdir(parents=True, exist_ok=True)
    doc = document()
    with open(out / "REGISTRATION.json", "x") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    print(out / "REGISTRATION.json", sha(out / "REGISTRATION.json"))
    print(json.dumps(dict(targets=doc["targets"], lambda0=doc["lambda0"], diag=doc["lambda0_diagnostics"]), indent=1))
