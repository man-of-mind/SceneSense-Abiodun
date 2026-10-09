"""Identical requirements and denominators for every policy (registered tiers; all-scored retention vs the per-scene
grid reference cell mode 0 / q 0; misses count; undefined quality stays in latency accounting)."""
from __future__ import annotations

import gzip
import json

import numpy as np

from rl_agent.splitfusion_run5_lowmcs_retrain_v1 import registration as REG

from . import collect as K

TIERS = {(k, c): t for k, d in REG.GUARDRAILS.items() for c, t in d.items()}      # (loc|seg|precision, vehicle|person)
CLS = {"vehicle": "v", "person": "p"}
STRATA = {"all": lambda a: np.ones(len(a["mcs"]), bool), "mcs_9_10": lambda a: a["mcs"] <= 10,
          "mcs_11_15": lambda a: (a["mcs"] >= 11) & (a["mcs"] <= 15), "mcs_ge16": lambda a: a["mcs"] >= 16}


def as_dict(data, cols):
    return {str(c): data[:, i].astype(np.float64) for i, c in enumerate(cols)}


def subset(a, mask):
    return {k: v[mask] for k, v in a.items()}


def metrics(a) -> dict:
    t = a["timely"] > 0.5
    out = dict(n=int(len(t)), timely=float(t.mean()), wire_p50_kb=float(np.percentile(a["wire"], 50)) / 1e3,
               wire_p99_kb=float(np.percentile(a["wire"], 99)) / 1e3)
    rt = {}
    for cls, c in CLS.items():
        el, ref_el = a[f"{c}_eligible"].sum(), a[f"ref_{c}_eligible"].sum()
        assert abs(el - ref_el) < 1e-6 * max(1.0, el), "eligible GT differs from the reference denominator"
        hits, th = a[f"{c}_hits"].sum(), (t * a[f"{c}_hits"]).sum()
        tp, fp, i, u = (a[f"{c}_{f}"].sum() for f in ("tp", "fp", "inter", "union"))
        rtp, rfp, ri, ru, rh = (a[f"ref_{c}_{f}"].sum() for f in ("tp", "fp", "inter", "union", "hits"))
        out[cls] = dict(eligible=float(el), recall_within_tol=hits / el, timely_recall_within_tol=th / el,
                        precision=tp / (tp + fp), seg_iou=i / u,
                        retention=dict(loc=(hits / el) / (rh / el), seg=(i / u) / (ri / ru),
                                       precision=(tp / (tp + fp)) / (rtp / (rtp + rfp))))
        rt[cls] = th / el
    out["P"] = 0.4 * rt["vehicle"] + 0.6 * rt["person"]
    inc = a["excluded"] < 0.5
    out["realized_C0_reward"] = float(np.nanmean(np.where(t[inc], a["value_C0"][inc], 0.0)))
    return out


def attainment(m, c0_mean: dict) -> dict:
    """Absolute tier attainment, kept separate from passes obtained only through the registered shared-miss exception
    (C0 also below the tier and value >= C0 - 0.01)."""
    out = {}
    for (k, cls), tier in TIERS.items():
        v, c0 = m[cls]["retention"][k], c0_mean[(k, cls)]
        absolute = v >= tier
        shared = (not absolute) and c0 < tier and v >= c0 - 0.01
        out[f"{k}_{cls}"] = dict(value=v, tier=tier, absolute=bool(absolute), shared_miss_only=bool(shared),
                                 registered_pass=bool(absolute or shared))
    return out


def load_retrain_eval(path) -> dict:
    """Existing retrain evaluation (PREVIOUSLY EXAMINED): per (policy, env, variant) arrays in collect.COLS layout."""
    rows = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            key = (r["policy"], r["env"], r["variant"])
            rows.setdefault(key, []).append(r)
    out = {}
    for key, rs in rows.items():
        rs.sort(key=lambda r: (r["eval_seed"], r["i"]))
        a = {c: np.array([float(r[c]) if r[c] is not None else np.nan for r in rs]) for c in
             ("timely", "excluded", "mcs", "snr", "backlog", "wire", "q", "mode", "receipt")}
        a["value_C0"] = np.array([np.nan if r["value"]["C0"] is None else r["value"]["C0"] for r in rs])
        for c in "vp":
            for f in K.FIELDS:
                a[f"{c}_{f}"] = np.array([r[f"{c}_{f}"] for r in rs])
                a[f"ref_{c}_{f}"] = np.array([r[f"ref_{c}_{f}"] for r in rs])
        a["eval_seed"] = np.array([r["eval_seed"] for r in rs])
        out[key] = a
    return out
