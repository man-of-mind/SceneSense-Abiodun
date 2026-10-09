"""Fit the candidate on the FIT sessions only (frozen plan); write CANDIDATE.json + FIT.json (create-only).

    python -m rl_agent.splitfusion_run5_uplink_lowmcs_v1.fit --out <dir>/fit_01
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from rl_agent.ue_production_transport_model_v2 import artifact_v2 as A2

from . import data as DT
from . import model as M
from . import plan as P

BOOT = 2000
BLOCK = 25


def load_transport() -> A2.ProductionTransportModelV2:
    path = P.ROOT / P.T_V2B_RELPATH
    if hashlib.sha256(path.read_bytes()).hexdigest() != "9919e5285d454ec742d877ca33af0df30277df82fe3cf665288c1102b6be286c":
        raise RuntimeError("T_v2b artifact drifted")
    return A2.ProductionTransportModelV2.load(path)


def x_of(r) -> float:
    return max(0, r["wire"] - P.W0_BYTES) / 1e6 * (P.MCS_REF - r["mcs"])


def spread(v) -> float:
    return float(np.percentile(v, 90) - np.percentile(v, 10))


def estimate(rows, sessions, intercept=False) -> dict:
    """kappa (and lambda_L) from the given sessions' rows; bH is each session's own high-MCS mean residual."""
    out_low, out_high = [], []
    for s in sessions:
        hi = [r for r in rows if r["session"] == s and DT.high_region(r)]
        lo = [r for r in rows if r["session"] == s and DT.low_region(r)]
        if not hi:
            continue
        b_h = float(np.mean([r["T"] - r["mu_v2b"] for r in hi]))
        out_high += [r["T"] - r["mu_v2b"] - b_h for r in hi]
        out_low += [(x_of(r), r["T"] - r["mu_v2b"] - b_h) for r in lo]
    x = np.array([a for a, _ in out_low]); d = np.array([b for _, b in out_low])
    if intercept:
        A = np.vstack([np.ones_like(x), x]).T
        (a, k), *_ = np.linalg.lstsq(A, d, rcond=None)
        return dict(intercept_ms=float(a), kappa=float(k), n_low=int(x.size))
    k = float((x * d).sum() / (x * x).sum())
    lam = spread(d - k * x) / spread(np.array(out_high))
    return dict(kappa=k, lambda_low=float(lam), n_low=int(x.size), n_high=len(out_high),
                spread_low_ms=spread(d - k * x), spread_high_ms=spread(np.array(out_high)),
                mean_low_contrast_ms=float(d.mean()))


def bootstrap(rows, sessions, seed=0) -> dict:
    blocks = sorted({(r["session"], r["decision_index"] // BLOCK) for r in rows if r["session"] in sessions})
    by = {}
    for r in rows:
        by.setdefault((r["session"], r["decision_index"] // BLOCK), []).append(r)
    rng = np.random.default_rng(seed)
    ks, ls = [], []
    for _ in range(BOOT):
        pick = rng.integers(0, len(blocks), len(blocks))
        sample = []
        for i in pick:
            for r in by[blocks[i]]:
                sample.append(r)
        e = estimate(sample, sessions)
        ks.append(e["kappa"]); ls.append(e["lambda_low"])
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return dict(kappa_ci95=q(ks), lambda_low_ci95=q(ls), resamples=BOOT, block=f"(session, decision_index // {BLOCK})")


def run(out: Path) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    plan_doc = json.loads((out.parent / "FITTING_PLAN.json").read_text())
    if plan_doc["plan_sha256"] != P.canonical(P.document()):
        raise RuntimeError("plan module differs from the frozen FITTING_PLAN.json")
    transport = load_transport()
    rows, audits = DT.load_all(transport)
    fit_sessions = [s for s, v in P.SESSIONS.items() if v[3] == "FIT"]
    main = estimate(rows, fit_sessions)
    fit_low = [r for r in rows if r["session"] in fit_sessions and DT.low_region(r)]
    boot = bootstrap(rows, fit_sessions)
    loso = {s: estimate(rows, [t for t in fit_sessions if t != s]) for s in fit_sessions}
    sens = dict(with_intercept=estimate(rows, fit_sessions, intercept=True),
                validation_session_alone_INTERNAL=estimate(rows, ["fade"]),
                all_eight_sessions=estimate(rows, list(P.SESSIONS)))
    correction = M.LowMcsCorrectionV1(
        kappa_ms_per_mb_mcs=main["kappa"], lambda_low=main["lambda_low"],
        fit_wire_max=int(max(r["wire"] for r in fit_low)), fit_mcs_min=int(min(r["mcs"] for r in fit_low)),
        provenance=dict(plan_sha256=plan_doc["plan_sha256"], fit_sessions=fit_sessions, t_v2b_sha256=transport.document_sha256,
                        n_low=main["n_low"], n_high=main["n_high"]))
    per_session = {}
    for s in P.SESSIONS:
        lo = [r for r in rows if r["session"] == s and DT.low_region(r)]
        per_session[s] = dict(partition=P.SESSIONS[s][3], warm=sum(r["session"] == s for r in rows), low_region=len(lo),
                              low_positive_backlog=sum(r["session"] == s and r["mcs"] <= P.LOW_MCS_MAX and r["backlog"] > 0 for r in rows),
                              high_region=sum(r["session"] == s and DT.high_region(r) for r in rows),
                              low_payload_kb=[float(np.percentile([r["payload"] for r in lo], p)) / 1e3 for p in (0, 50, 100)] if lo else None,
                              low_mcs=sorted({r["mcs"] for r in lo}))
    doc = dict(schema="splitfusion.run5_uplink_lowmcs_v1.fit.v1", plan_sha256=plan_doc["plan_sha256"], audits=audits,
               per_session=per_session, estimate=main, bootstrap=boot, loso=loso, sensitivities=sens,
               candidate=correction.to_dict(),
               interpretation=dict(
                   kappa=f"{main['kappa']:.1f} ms per MB per MCS index below 16 at zero backlog",
                   example_ms_per_100kb={m: main["kappa"] * 0.1 * (16 - m) for m in (9, 10, 12, 15)}))
    (out / "CANDIDATE.json").write_text(json.dumps(correction.to_dict(), indent=1, sort_keys=True))
    (out / "FIT.json").write_text(json.dumps(doc, indent=1, sort_keys=True, default=float))
    return doc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    d = run(p.parse_args().out)
    print(json.dumps({k: d[k] for k in ("estimate", "bootstrap", "sensitivities", "audits")}, indent=1, default=float))
    print(json.dumps(d["loso"], indent=0, default=float)); print(json.dumps(d["per_session"], indent=0))
