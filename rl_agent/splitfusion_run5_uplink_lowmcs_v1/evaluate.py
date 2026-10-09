"""Old (binding_02 TRAIN_GCV2 + T_v2b) versus candidate, on the held-out FADE_RECOVERY session and regression pools.

    python -m rl_agent.splitfusion_run5_uplink_lowmcs_v1.evaluate --run <dir>      # reads <dir>/fit_01, writes <dir>/eval_01

Own-chain composition (primary): each validation decision's own decomposed chain keeps every measured stage, GT path,
score costs and measured carry-in; only stage E is replaced by the model's moved E for every base chain j of the
pool (the exact environment E move, u = 1 so lateness is taken as its probability, not a draw). Hence
P(timely_i) = mean_j (1 - p_late_ij) * 1[compose(v_i with E_ij).receipt <= 170].
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from dataclasses import replace

import numpy as np
from scipy import stats

from rl_agent.splitfusion_corrected_latency_env_v1.chains import RoleCostsV1, compose
from rl_agent.splitfusion_run5_full_v1 import evidence as EVF
from rl_agent.splitfusion_run5_reward_trials_v1 import latency as LR
from rl_agent.splitfusion_run5_reward_trials_v1 import validate as RV

from . import data as DT
from . import fit as FT
from . import latency as LC
from . import model as M
from . import plan as P

VALIDATION = "fade"


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [c - h, c + h]


def newcombe(k1, n1, k2, n2):
    """95% hybrid-score interval for p1 - p2 (Newcombe 1998, method 10)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return [d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)]


def models(correction):
    transport = FT.load_transport()
    old = LR.PoolLatencyModelV1(P.BINDING_02, transport, pool=P.BASE_POOL)
    cand = LC.CandidatePoolLatencyModelV1(P.BINDING_02, transport, pool=P.BASE_POOL, correction=correction)
    mean_only = LC.CandidatePoolLatencyModelV1(
        P.BINDING_02, transport, pool=P.BASE_POOL,
        correction=M.LowMcsCorrectionV1(correction.kappa_ms_per_mb_mcs, 1.0, correction.fit_wire_max, correction.fit_mcs_min))
    return transport, {"old": old, "candidate": cand, "mean_only": mean_only}


def validation_chains():
    rows_rel, run, _, _ = P.SESSIONS[VALIDATION]
    frames = P.WT / f"{run}/capture/cells/{P.CELL[VALIDATION]}/numeric_gt/service/FRAMES.jsonl"
    chains, refused = EVF.load_chains(dict(source_id=VALIDATION, row_filter=None, rows=str(P.ROOT / rows_rel),
                                           frames=str(frames)))
    vectors, rejected = EVF.decompose_all(chains)
    return {c.frame_id: (c, vectors.get(c.chain_id)) for c in chains}, refused, rejected


def with_own_session_donor(vector, chain, pool):
    """Fill the worker-order roles this chain did not measure from the nearest opposite-order chain of the SAME
    session (camera SI / radar P40, unscaled ranks as in the environment's donor rule)."""
    ctx = chain.context
    cands = sorted((math.hypot(c.context.camera_si - ctx.camera_si, (c.context.radar_p40 - ctx.radar_p40) * 100.0), c.chain_id, v)
                   for c, v in pool if v is not None and v.realized_ordering is not vector.realized_ordering)
    donor, own = cands[0][2].costs, vector.costs
    return vector.with_costs(RoleCostsV1(
        target_admit=own.target_admit or donor.target_admit, prediction_score=own.prediction_score or donor.prediction_score,
        prediction_store=own.prediction_store or donor.prediction_store, target_score=own.target_score or donor.target_score,
        donor_chain_id=cands[0][1]))


def own_chain(model, row, chain, vector):
    ranked = model.ranked(row["camera_si"], row["radar_p40"])
    E, pl, timely = [], [], []
    for j in model._ids:
        d = model.move(j, ranked=ranked, mode_id=row["mode"], q_e4=row["q"], prior_ul_mcs=row["mcs"],
                       backlog_bytes=row["backlog"], wire_bytes=row["wire"], u=1.0)
        e = d.vector.E
        receipt = compose(vector.with_uplink(e)).receipt
        E.append(e); pl.append(d.uplink_late_probability); timely.append(receipt <= P.DEADLINE_MS)
    E, pl, timely = np.array(E), np.array(pl), np.array(timely)
    w = 1.0 - pl
    pit = float((w * (E <= row["T"])).mean())
    return dict(p_timely=float((w * timely).mean()), e_mean=float(E.mean()), e_p50=float(np.median(E)),
                p_late_mean=float(pl.mean()), pit=pit)


def env_level(model, row, carry_in):
    """Environment draw on the k = 5 nearest base chains, expectation over the lateness uniform."""
    ranked = model.ranked(row["camera_si"], row["radar_p40"])
    acc = 0.0
    for dist, j in ranked[:5]:
        d = model.move(j, ranked=ranked, mode_id=row["mode"], q_e4=row["q"], prior_ul_mcs=row["mcs"],
                       backlog_bytes=row["backlog"], wire_bytes=row["wire"], u=1.0, distance=dist)
        acc += (1 - d.uplink_late_probability) * (compose(d.vector, carry_in=carry_in).receipt <= P.DEADLINE_MS)
    return acc / 5


def groups(rows, cuts):
    out = {}
    for label, sel in (("mcs_le15_B0", lambda r: r["mcs"] <= 15 and r["backlog"] == 0),
                       ("mcs_le10_B0", lambda r: r["mcs"] <= 10 and r["backlog"] == 0),
                       ("mcs_le15_Bpos", lambda r: r["mcs"] <= 15 and r["backlog"] > 0),
                       ("mcs_ge16_B0", lambda r: r["mcs"] >= 16 and r["backlog"] == 0)):
        pop = [r for r in rows if sel(r)]
        bins = {"T1": [r for r in pop if r["payload"] <= cuts[0]], "T2": [r for r in pop if cuts[0] < r["payload"] <= cuts[1]],
                "T3": [r for r in pop if r["payload"] > cuts[1]], "all": pop}
        out[label] = bins
    return out


def summarize(bins, names):
    res = {}
    for b, rs in bins.items():
        n, k = len(rs), sum(r["on_time"] for r in rs)
        e = dict(n=n, measured_on_time=k / n if n else None, wilson95=wilson(k, n),
                 payload_kb=[min(r["payload"] for r in rs) / 1e3, float(np.median([r["payload"] for r in rs])) / 1e3,
                             max(r["payload"] for r in rs) / 1e3] if rs else None,
                 T_measured_p50=float(np.median([r["T"] for r in rs])) if rs else None)
        for m in names:
            if not rs or m not in rs[0].get("own", {}):
                continue
            o = [r["own"][m] for r in rs]
            e[m] = dict(own_chain_p_timely=float(np.mean([x["p_timely"] for x in o])),
                        transport_mean_error_ms=float(np.mean([x["e_mean"] - r["T"] for x, r in zip(o, rs)])),
                        transport_median_abs_error_ms=float(np.median([abs(x["e_p50"] - r["T"]) for x, r in zip(o, rs)])),
                        coverage={q: float(np.mean([x["pit"] <= q for x in o])) for q in (0.5, 0.8, 0.9)},
                        p_late_mean=float(np.mean([x["p_late_mean"] for x in o])))
            if "env" in rs[0] and m in rs[0]["env"]:
                e[m]["env_idle_p_timely"] = float(np.mean([r["env"][m] for r in rs]))
        res[b] = e
    return res


def acceptance(summary, fit_doc, regression):
    low = summary["mcs_le15_B0"]
    out = {}
    for m in ("old", "candidate", "mean_only"):
        a1 = all(low[t]["wilson95"][0] <= low[t][m]["own_chain_p_timely"] <= low[t]["wilson95"][1] for t in ("T1", "T2", "T3"))
        k1, n1 = round(low["T1"]["measured_on_time"] * low["T1"]["n"]), low["T1"]["n"]
        k3, n3 = round(low["T3"]["measured_on_time"] * low["T3"]["n"]), low["T3"]["n"]
        ci = newcombe(k1, n1, k3, n3)
        pred = low["T1"][m]["own_chain_p_timely"] - low["T3"][m]["own_chain_p_timely"]
        a2 = ci[0] <= pred <= ci[1]
        n = low["all"]["n"]
        region = [stats.binom.ppf(0.025, n, 0.8) / n, stats.binom.ppf(0.975, n, 0.8) / n]
        a3 = (all(abs(low[t][m]["transport_mean_error_ms"]) <= 10 for t in ("T1", "T2", "T3"))
              and region[0] <= low["all"][m]["coverage"][0.8] <= region[1])
        out[m] = dict(A1=bool(a1), A2=bool(a2), contrast_predicted=pred, contrast_measured=(k1 / n1 - k3 / n3), contrast_ci95=ci,
                      A3=bool(a3), p80_region=region)
    ks = [v["kappa"] for v in fit_doc["loso"].values()]
    k0 = fit_doc["estimate"]["kappa"]
    out["A4"] = dict(passed=all(0.5 * k0 <= k <= 1.5 * k0 for k in ks), loso_range=[min(ks), max(ks)], kappa=k0)
    out["A5"] = regression
    return out


def regression_pools(models_):
    """Existing out-of-sample procedure (validate.predict, TRAIN_GCV2 base, measured carry-in, 20 draws, same seeds)."""
    res = {}
    for i, pool in enumerate(("HELDOUT_GCV2", "RETENTION_V1")):
        target = LR.PoolLatencyModelV1(P.BINDING_02, models_["old"].transport, pool=pool)
        chains = sorted(target.chains.values(), key=lambda c: c.chain_id)
        measured = RV._summary([c.instants["ue_receipt"] for c in chains])
        res[pool] = dict(measured=measured)
        for m in ("old", "candidate"):
            pred, _ = RV.predict(models_[m], chains, seed=2000 + i)
            res[pool][m] = RV._summary(pred)
        res[pool]["timely_change"] = res[pool]["candidate"]["timely"] - res[pool]["old"]["timely"]
    return res


def run(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    out = run_dir / "eval_01"
    out.mkdir(parents=False, exist_ok=False)
    fit_doc = json.loads((run_dir / "fit_01/FIT.json").read_text())
    correction = M.LowMcsCorrectionV1.load(run_dir / "fit_01/CANDIDATE.json")
    transport, models_ = models(correction)
    rows, _ = DT.load_all(transport)
    val = [r for r in rows if r["session"] == VALIDATION]
    chains, refused, rejected = validation_chains()
    reproduced, missing, gated = 0, [], []
    for r in val:
        c, v = chains.get(r["frame_id"], (None, None))
        if v is None:
            missing.append(r["frame_id"])
            continue
        if abs(v.E - r["T"]) > 1e-6:
            raise RuntimeError(f"stage E != reassembly - last send for frame {r['frame_id']}")
        if v.A + v.B + v.C + v.D > v.bind_end:          # measured send-gated CONFIRM_SEND (binding_02 gate rule)
            v = replace(v, bind_end=v.A + v.B + v.C + v.D)
            gated.append(r["frame_id"])
        reproduced += abs(compose(v).receipt - c.instants["ue_receipt"]) <= 1e-6
        v = with_own_session_donor(v, c, list(chains.values()))
        r["own"] = {m: own_chain(models_[m], r, c, v) for m in (("old", "candidate", "mean_only")
                                                               if r["mcs"] <= 15 else ("old", "candidate"))}
        r["env"] = {m: env_level(models_[m], r, None) for m in ("old", "candidate")}
        r["env_measured_carry"] = {m: env_level(models_[m], r, c.service.carry_in) for m in ("old", "candidate")}
    val = [r for r in val if "own" in r]
    low15 = sorted(r["payload"] for r in val if r["mcs"] <= 15 and r["backlog"] == 0)
    cuts = [float(np.percentile(low15, 100 / 3)), float(np.percentile(low15, 200 / 3))]
    summary = {g: summarize(b, ("old", "candidate", "mean_only")) for g, b in groups(val, cuts).items()}
    # comparability view only (the decision memo's split); not a law
    memo = {}
    for label, sel in (("le63kb", lambda r: r["payload"] <= 63_000), ("gt63kb", lambda r: r["payload"] > 63_000)):
        memo[label] = summarize({"x": [r for r in val if r["mcs"] <= 15 and r["backlog"] == 0 and sel(r)]},
                                ("old", "candidate"))["x"]
    regression = regression_pools(models_)
    regression_ok = dict(passed=all(abs(v["timely_change"]) <= 0.01 for v in regression.values()), pools=regression)
    acc = acceptance(summary, fit_doc, regression_ok)
    per_decision = [dict(decision_index=r["decision_index"], frame_id=r["frame_id"], mcs=r["mcs"], backlog=r["backlog"],
                         payload=r["payload"], T=r["T"], on_time=r["on_time"], receipt=r["receipt"],
                         own={m: dict(p_timely=x["p_timely"], e_mean=x["e_mean"], pit=x["pit"]) for m, x in r["own"].items()},
                         env_idle=r["env"], env_measured_carry=r["env_measured_carry"]) for r in val]
    doc = dict(schema="splitfusion.run5_uplink_lowmcs_v1.evaluation.v1", plan_sha256=fit_doc["plan_sha256"],
               correction=correction.to_dict(), base_pool=P.BASE_POOL, binding_02_sha256=P.BINDING_02_SHA256,
               old_binding_sha256=models_["old"].binding_sha256, candidate_binding_sha256=models_["candidate"].binding_sha256,
               beta_pool_ms=models_["candidate"].beta_pool,
               validation=dict(session=VALIDATION, decisions=len(val), chains_refused=len(refused), rejected=rejected,
                               no_vector=missing, send_gated_measured_chains=gated, own_chain_exact_reproduction=f"{reproduced}/{len(val)}",
                               tercile_cuts_bytes=cuts, status="held out of the fit; previously inspected (not untouched)"),
               summary=summary, memo_split_comparability_only=memo, acceptance=acc)
    (out / "EVALUATION.json").write_text(json.dumps(doc, indent=1, sort_keys=True, default=float))
    with open(out / "PER_DECISION.jsonl", "w") as f:
        for x in per_decision:
            f.write(json.dumps(x, default=float) + "\n")
    return doc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    d = run(p.parse_args().run)
    print(json.dumps(dict(validation=d["validation"], acceptance=d["acceptance"]), indent=1, default=float))
    for g in ("mcs_le15_B0", "mcs_le10_B0", "mcs_le15_Bpos", "mcs_ge16_B0"):
        for t, e in d["summary"][g].items():
            print(g, t, e["n"], "meas %.3f" % e["measured_on_time"] if e["n"] else "-", e["wilson95"],
                  {m: (round(e[m]["own_chain_p_timely"], 3), round(e[m]["transport_mean_error_ms"], 1),
                       round(e[m].get("env_idle_p_timely", float("nan")), 3)) for m in ("old", "candidate", "mean_only") if m in e})
