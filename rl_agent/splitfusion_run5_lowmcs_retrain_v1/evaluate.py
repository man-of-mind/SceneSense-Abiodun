"""Fair offline comparison: every policy in the SAME environment (common scene/channel streams and draws).

Policies: NEW_seed17/29/43 (this run), C0_seed17/29/43 (reward trials, binding_02 without the correction), the deployed
HIST_seed43, and FIXED_m11_q7000 / q8000. Environments E1 (held_scene + HELDOUT_GCV2, primary) and E2 (held_scene +
RETENTION_V1), each with the CORRECTED latency model (primary) and the OLD one (sensitivity). Per-decision rows reuse the
reward-trial evaluation fields; added: candidate flags (computed from the state for both variants), the read-only
queue-order instrumentation and the monotone-envelope receipt.

  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python3 -m rl_agent.splitfusion_run5_lowmcs_retrain_v1.evaluate \
      --runs <ABS runs_01> --registration <ABS REGISTRATION.json> --out <ABS eval_01> --workers 12
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
import multiprocessing as mp
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from rl_agent.splitfusion_corrected_latency_env_v1 import chains as CH
from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as orch
from rl_agent.splitfusion_run5_full_v1 import collector as FC
from rl_agent.splitfusion_run5_reward_trials_v1 import collector as C
from rl_agent.splitfusion_run5_reward_trials_v1 import evaluate as RE
from rl_agent.splitfusion_run5_reward_trials_v1 import rewards as R

from . import campaign as CAMP
from . import registration as REG

ROOT = REG.ROOT
POOLS = {"E1": "HELDOUT_GCV2", "E2": "RETENTION_V1"}
VARIANTS = ("CORRECTED", "OLD")
STRATA = {"mcs_9_10": lambda r: r["mcs"] <= 10, "mcs_11_15": lambda r: 11 <= r["mcs"] <= 15,
          "mcs_ge16": lambda r: r["mcs"] >= 16}
BACKLOG = {"B0": lambda r: r["backlog"] == 0, "Bpos": lambda r: r["backlog"] > 0}
MIN_STRATUM = 100

# ------------------------------------------------------------------ read-only queue-order instrumentation
_REAL_COMPOSE = CH.compose
LAST: dict = {}


def order_flip(vector, carry_in, comp) -> dict:
    """For a GT_FIRST composition: move the prediction earlier to the order boundary (E only); if the result is
    PREDICTION_FIRST with a LATER receipt, this decision gained from the later prediction (envelope = that receipt)."""
    if comp.ordering is not CH.Ordering.GT_FIRST:
        return dict(flip_gain=False, envelope=comp.receipt)
    r_p = sum(getattr(vector, s) for s in CH.PREDICTION_PATH)
    free = -math.inf if carry_in is None else carry_in
    shift = r_p - max(vector.G, free) + 1e-6
    if shift <= 0 or shift > vector.E:
        return dict(flip_gain=False, envelope=comp.receipt, boundary_unreachable=shift > vector.E)
    alt = _REAL_COMPOSE(replace(vector, E=vector.E - shift), carry_in=carry_in)
    gain = alt.ordering is CH.Ordering.PREDICTION_FIRST and alt.receipt > comp.receipt + 1e-9
    return dict(flip_gain=bool(gain), envelope=max(comp.receipt, alt.receipt) if gain else comp.receipt,
                gain_ms=alt.receipt - comp.receipt if gain else 0.0)


def instrumented_compose(vector, *, carry_in=CH._MEASURED):
    comp = _REAL_COMPOSE(vector, carry_in=carry_in)          # the returned composition is the real one, unchanged
    LAST.clear()
    LAST.update(order_flip(vector, vector.carry_in if carry_in is CH._MEASURED else carry_in, comp))
    return comp


# ------------------------------------------------------------------ policies
def policies(runs_dir: Path) -> dict:
    out = {}
    for s in REG.SEEDS:
        r = json.loads((runs_dir / f"NEW_seed{s}/RUN_RESULT.json").read_text())
        assert r["updates"] == REG.UPDATES and REG.sha(ROOT / r["actor"]["path"]) == r["actor"]["sha256"]
        out[f"NEW_seed{s}"] = dict(kind="actor", group="NEW", seed=s, path=r["actor"]["path"], sha256=r["actor"]["sha256"])
    for label, spec in REG.COMPARATORS.items():
        assert REG.sha(ROOT / spec["path"]) == spec["sha256"]
        out[label] = dict(kind="actor", group="C0", seed=int(label[-2:]), **spec)
    assert REG.sha(ROOT / REG.DEPLOYED["path"]) == REG.DEPLOYED["sha256"]
    out[REG.DEPLOYED["label"]] = dict(kind="actor", group="HIST", seed=43, path=REG.DEPLOYED["path"],
                                      sha256=REG.DEPLOYED["sha256"])
    for label, (m, q) in REG.FIXED.items():
        out[label] = dict(kind="fixed", group="FIXED", mode=m, q=q)
    return out


def run_policy(label, spec, env, variant, seed, n) -> list[dict]:
    torch.set_num_threads(1)
    FC.compose = instrumented_compose                          # worker process only (spawn); results unchanged
    shared, latency = CAMP.components(split="held_scene", pool=POOLS[env], corrected=variant == "CORRECTED")
    correction = CAMP.load_correction()
    cat = shared["catalog"]
    col = C.TrialRun5CollectorV1(latency=latency, seed=seed, shared_sources=shared, arm=REG.ARM)
    actor = RE.load_actor(ROOT / spec["path"]) if spec["kind"] == "actor" else None
    rows = []
    for i in range(n):
        mode, q = RE.act(actor, col.current_state_features()) if actor else (spec["mode"], spec["q"])
        col.collect_transition(orch.ModeledActionRequestV1(decision_ordinal=i, mode_id=mode, q_e4=q,
                                                           source="STOCHASTIC_ACTOR", warmup_q_bin_index=None))
        d = col.diagnostics[-1]
        key = d["reward_scene_key"]
        flip = dict(LAST)
        row = dict(i=i, mode=mode, q=q, terminal=d["terminal"], excluded=d["excluded"], receipt=d["receipt_ms"],
                   timely=d["receipt_ms"] is not None and d["receipt_ms"] <= 170.0, profile=d["profile"],
                   snr=d["snr_db"], mcs=d["prior_ul_mcs"], backlog=d["backlog_bytes"], wire=d["wire_bytes"],
                   scene=key, q_perc=d["q_perc"], late_uplink=d["uplink_late"], gated="CONFIRM_SEND_GATED" in d["floors"],
                   flags=list(correction.flags(backlog_bytes=d["backlog_bytes"], wire_bytes=d["wire_bytes"],
                                               prior_ul_mcs=d["prior_ul_mcs"])),
                   flip_gain=flip["flip_gain"], envelope=flip["envelope"], gain_ms=flip.get("gain_ms", 0.0),
                   timely_env=flip["envelope"] <= 170.0,
                   value={a: cat.value_for(a, key, mode_id=mode, q_e4=q) for a in R.ARMS})
        assert abs(flip["envelope"] - d["receipt_ms"]) < 1e-9 or flip["flip_gain"]
        row.update(RE.expected(cat.components(key, mode_id=mode, q_e4=q)))
        row.update({f"ref_{k}": v for k, v in RE.expected([(1.0, cat.reference_components(key))]).items()})
        rows.append(row)
    return rows


def _job(args):
    label, spec, env, variant, seed, n = args
    return label, env, variant, seed, run_policy(label, spec, env, variant, seed, n)


# ------------------------------------------------------------------ summaries
def pct(v, p):
    return float(np.percentile(v, p)) if len(v) else None


def stratum(rows) -> dict:
    """Compact per-stratum summary (all-scored and on-time-only kept separate)."""
    n = len(rows)
    if n == 0:
        return dict(n=0)
    qa, qt, ref = RE.quality(rows), RE.quality(rows, timely_only=True), RE.quality(rows, prefix="ref_")
    timely_rows = [r for r in rows if r["timely"]]
    qto = RE.quality(timely_rows) if timely_rows else None
    rv, rp = qt["vehicle"]["recall_within_tolerance"], qt["person"]["recall_within_tolerance"]
    P = None if rv is None or rp is None else 0.4 * rv + 0.6 * rp
    env_rows = [dict(r, timely=r["timely_env"]) for r in rows]
    qe = RE.quality(env_rows, timely_only=True)
    Pe = (None if qe["vehicle"]["recall_within_tolerance"] is None or qe["person"]["recall_within_tolerance"] is None
          else 0.4 * qe["vehicle"]["recall_within_tolerance"] + 0.6 * qe["person"]["recall_within_tolerance"])
    receipts = [r["receipt"] for r in rows]
    wires = [r["wire"] for r in rows]
    ret = {cls: {k: RE._ratio(qa[cls][m], ref[cls][m]) for k, m in (("loc", "recall_within_tolerance"),
                                                                       ("seg", "seg_iou"), ("precision", "precision"))}
           for cls in RE.CLASSES}
    return dict(
        n=n, P=P, P_envelope=Pe, timely=float(np.mean([r["timely"] for r in rows])),
        receipt_p50=pct(receipts, 50), receipt_p95=pct(receipts, 95), receipt_p99=pct(receipts, 99),
        recall_timely={c: qt[c]["recall_within_tolerance"] for c in RE.CLASSES},
        recall_all_scored={c: qa[c]["recall_within_tolerance"] for c in RE.CLASSES},
        recall_on_time_only={c: (qto[c]["recall_within_tolerance"] if qto else None) for c in RE.CLASSES},
        seg_iou_all={c: qa[c]["seg_iou"] for c in RE.CLASSES}, precision_all={c: qa[c]["precision"] for c in RE.CLASSES},
        retention=ret, excluded_timely_undefined=sum(r["excluded"] for r in rows),
        wire_p50=pct(wires, 50), wire_p95=pct(wires, 95), wire_p99=pct(wires, 99),
        q_p10_p50_p90=[pct([r["q"] for r in rows], p) for p in (10, 50, 90)],
        mode_share={m: c / n for m, c in sorted(collections.Counter(r["mode"] for r in rows).items())},
        flagged=sum(bool(r["flags"]) for r in rows) / n,
        flags={f: sum(f in r["flags"] for r in rows) for f in CAMP.FLAGS},
        flip_gain=sum(r["flip_gain"] for r in rows), flip_gain_rate=sum(r["flip_gain"] for r in rows) / n)


def summarize(rows) -> dict:
    out = dict(all=stratum(rows))
    for s, f in STRATA.items():
        sub = [r for r in rows if f(r)]
        out[s] = stratum(sub)
        for b, g in BACKLOG.items():
            subb = [r for r in sub if g(r)]
            out[f"{s}|{b}"] = stratum(subb) if len(subb) >= MIN_STRATUM else dict(n=len(subb), note="n < 100: not reported")
    return out


def contrib(rows, ev, ep, key="timely"):
    return np.asarray([r[key] * (0.4 * r["v_hits"] / ev + 0.6 * r["p_hits"] / ep) for r in rows])


def paired(a, b, select=None) -> dict:
    idx = [k for k in range(len(a)) if select is None or select(a[k], b[k])]
    A, B = [a[k] for k in idx], [b[k] for k in idx]
    ev, ep = sum(r["v_eligible"] for r in A), sum(r["p_eligible"] for r in A)
    assert (ev, ep) == (sum(r["v_eligible"] for r in B), sum(r["p_eligible"] for r in B)), "eligibility differs"
    d = contrib(A, ev, ep) - contrib(B, ev, ep)
    d_env = contrib(A, ev, ep, "timely_env") - contrib(B, ev, ep, "timely_env")
    unflag = np.asarray([not (x["flags"] or y["flags"]) for x, y in zip(A, B)])
    flip = np.asarray([x["flip_gain"] or y["flip_gain"] for x, y in zip(A, B)])
    return dict(n=len(idx), dP=float(d.sum()), ci95=RE.block_ci(d) if len(d) >= 2 * RE.BLOCK else None,
                dP_envelope=float(d_env.sum()), dP_unflagged_contribution=float(d[unflag].sum()),
                dP_flip_gain_contribution=float(d[flip].sum()), decisions_with_flip_gain=int(flip.sum()))


def evaluate(runs_dir: Path, registration: Path, out: Path, n: int, workers: int) -> dict:
    for p in (runs_dir, registration, out):
        if not Path(p).is_absolute():
            raise ValueError("absolute paths only")
    REG.verify(registration, seed=17)
    out.mkdir(parents=True, exist_ok=False)
    pol = policies(runs_dir)
    seeds = REG.EVALUATION["eval_seeds"]
    jobs = [(label, spec, env, var, s, n) for label, spec in pol.items() for env in POOLS for var in VARIANTS for s in seeds]
    raw = collections.defaultdict(dict)
    with mp.get_context("spawn").Pool(workers) as pool:
        for label, env, var, s, rows in pool.imap_unordered(_job, jobs):
            raw[(label, env, var)][s] = rows
    with gzip.open(out / "EVAL_DECISIONS.jsonl.gz", "wt") as stream:
        for (label, env, var), per in sorted(raw.items()):
            for s, rows in sorted(per.items()):
                for r in rows:
                    stream.write(json.dumps(dict(r, policy=label, env=env, variant=var, eval_seed=s),
                                            separators=(",", ":")) + "\n")
    crn = {}
    for env in POOLS:
        for var in VARIANTS:
            for s in seeds:
                scenes = {tuple(r["scene"] for r in raw[(l, env, var)][s]) for l in pol}
                mcs = {tuple(r["mcs"] for r in raw[(l, env, var)][s]) for l in pol}
                assert len(scenes) == 1, f"{env}/{var}/{s}: scene streams differ"
                crn[f"{env}|{var}|{s}"] = dict(scene_streams_identical=True, mcs_streams_identical=len(mcs) == 1)
    results = {}
    for (label, env, var), per in sorted(raw.items()):
        pooled = [r for s in seeds for r in per[s]]
        results.setdefault(label, {}).setdefault(env, {})[var] = dict(
            pooled=summarize(pooled), per_eval_seed={str(s): stratum(per[s])["P"] for s in seeds})
    pairs = {}
    for env in POOLS:
        for var in VARIANTS:
            cat = lambda l: [r for s in seeds for r in raw[(l, env, var)][s]]
            for s in REG.SEEDS:
                a = cat(f"NEW_seed{s}")
                for ref in (f"C0_seed{s}", REG.DEPLOYED["label"], *REG.FIXED):
                    b = cat(ref)
                    entry = dict(all=paired(a, b))
                    if all(v["mcs_streams_identical"] for k, v in crn.items() if k.startswith(f"{env}|{var}|")):
                        for st, f in STRATA.items():
                            entry[st] = paired(a, b, lambda x, y, f=f: f(x))
                    pairs[f"{env}|{var}|NEW_seed{s}-{ref}"] = entry
            for s in REG.SEEDS:                                     # C0 vs deployed, for context
                pairs[f"{env}|{var}|C0_seed{s}-HIST_seed43"] = dict(all=paired(cat(f"C0_seed{s}"), cat("HIST_seed43")))
    doc = dict(schema="splitfusion.run5_lowmcs_retrain_v1.evaluation.v1", registration_sha256=REG.sha(registration),
               policies=pol, pools=POOLS, variants=VARIANTS, eval_seeds=seeds, decisions_per_seed=n, crn=crn,
               results=results, paired=pairs, decision=decide(results, pairs),
               evidence_label=REG.EVIDENCE_LABEL, live_claim=False)
    (out / "EVALUATION.json").write_text(json.dumps(doc, indent=1, sort_keys=True, default=str))
    return doc


# ------------------------------------------------------------------ registered rule (mechanical)
def decide(results, pairs) -> dict:
    S = REG.SEEDS
    get = lambda label, env="E1", var="CORRECTED": results[label][env][var]["pooled"]
    dP = lambda env, var, ref, st="all": [pairs[f"{env}|{var}|NEW_seed{s}-{ref(s)}"][st]["dP"] for s in S]
    c0 = lambda s: f"C0_seed{s}"
    mean = lambda v: float(np.mean(v))
    checks = {}
    d1 = dP("E1", "CORRECTED", c0)
    checks["R1_benefit"] = dict(per_seed=d1, mean=mean(d1), passed=mean(d1) >= 0.010 and all(x > 0 for x in d1))
    low = pairs[f"E1|CORRECTED|NEW_seed17-C0_seed17"].get("mcs_9_10")
    d2 = dP("E1", "CORRECTED", c0, "mcs_9_10") if low else None
    checks["R2_low_mcs_target"] = dict(per_seed=d2, passed=bool(d2) and all(x > 0 for x in d2))
    tn = mean([get(f"NEW_seed{s}")["all"]["timely"] for s in S])
    tc = mean([get(f"C0_seed{s}")["all"]["timely"] for s in S])
    checks["R3_timeliness"] = dict(new=tn, c0=tc, passed=tn >= tc - 0.02)
    guard = {}
    for k, tiers in REG.GUARDRAILS.items():
        for cls, tier in tiers.items():
            xv = mean([get(f"NEW_seed{s}")["all"]["retention"][cls][k] for s in S])
            cv = mean([get(f"C0_seed{s}")["all"]["retention"][cls][k] for s in S])
            guard[f"{k}_{cls}"] = dict(new=xv, c0=cv, tier=tier, passed=xv >= tier or (cv < tier and xv >= cv - 0.01))
    checks["R4_guardrails"] = dict(detail=guard, passed=all(g["passed"] for g in guard.values()))
    pn = mean([get(f"NEW_seed{s}")["all"]["P"] for s in S])
    ph = get("HIST_seed43")["all"]["P"]
    checks["R5_vs_deployed"] = dict(new_mean=pn, deployed=ph, passed=pn > ph)
    e2 = dP("E2", "CORRECTED", c0)
    e1o = dP("E1", "OLD", c0)
    checks["R6_model_uncertainty"] = dict(E2_corrected=e2, E1_old=e1o,
                                          passed=mean(e2) > 0 and mean(e1o) >= -0.01)
    flagged = mean([get(f"NEW_seed{s}")["all"]["flagged"] for s in S])
    unfl = sum(pairs[f"E1|CORRECTED|NEW_seed{s}-C0_seed{s}"]["all"]["dP_unflagged_contribution"] for s in S)
    tot = sum(d1)
    checks["R7_support"] = dict(new_flagged_rate=flagged, unflagged_share=unfl / tot if tot else None,
                                passed=flagged <= 0.05 and tot > 0 and unfl / tot >= 0.5)
    env_d = [pairs[f"E1|CORRECTED|NEW_seed{s}-C0_seed{s}"]["all"]["dP_envelope"] for s in S]
    checks["R8_queue_order"] = dict(envelope_per_seed=env_d, envelope_mean=mean(env_d),
                                    passed=tot > 0 and mean(env_d) > 0 and mean(env_d) >= 0.5 * mean(d1))
    core = all(checks[k]["passed"] for k in ("R1_benefit", "R2_low_mcs_target", "R3_timeliness", "R4_guardrails",
                                             "R5_vs_deployed"))
    robust = all(checks[k]["passed"] for k in ("R6_model_uncertainty", "R7_support", "R8_queue_order"))
    verdict = ("MERITS_BOUNDED_LIVE_VALIDATION" if core and robust else
               "MODEL_UNCERTAINTY_PREVENTS_DECISION" if core else "NO_QUALIFYING_IMPROVEMENT")
    ps = sorted((get(f"NEW_seed{s}")["all"]["P"], s) for s in S)
    return dict(rule="REGISTRATION.json rules R1-R8", checks=checks, verdict=verdict,
                future_checkpoint_rule=REG.FUTURE_CHECKPOINT_RULE,
                median_seed_if_ever_authorized=ps[1][1], promoted=None)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", required=True, type=Path)
    p.add_argument("--registration", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--decisions", type=int, default=REG.EVALUATION["decisions_per_seed"])
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args()
    d = evaluate(a.runs, a.registration, a.out, a.decisions, a.workers)
    print(json.dumps(d["decision"], indent=1, default=str))
