"""Held-out evaluation of the trained pilot actors and the supported fixed settings (modeled, offline).

Evaluation environment: EVAL chains, EVAL channel trace, held-out scenes supported by the EVAL strata, causal
fallback frames, unchanged outcome semantics. Common random numbers: every policy is run with the same evaluation
seeds (9017, 9029, 9043), so the scene streams, chain indices, fallback uniforms and trace start are identical
draws; trajectories still differ where actions change timing, fallbacks or session breaks.
Policies: each trained actor at its final checkpoint, deterministic (argmax mode, conditional mean q = the frozen
actor's registered deployment rule); and the seven measured fixed settings (m5 7000/9800, m8 5000/9800,
m11 3000/7000/8000). Sensitivity: every composed receipt shifted by -5 / 0 / +5 ms.
Per policy both reward definitions are reported, plus timeliness, per-class within-tolerance recall, segmentation
IoU, precision and false positives, localisation tails, action distribution, fallback frames and undefined counts.
Quality metrics come from the exact grid rows of the executed (scene, mode, q); off-grid q uses the registered
linear weights on the bracketing rows (expected counts). Offline modeled results; they do not establish live
performance.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from rl_agent.splitfusion_corrected_latency_env_v1 import collector as CR
from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as orch
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_models as RM

from . import authorization as A, collector as C, support as S
from .campaign import BINDING, components

EVAL_SEEDS = (9017, 9029, 9043)
SHIFTS = (-5.0, 0.0, 5.0)
FIXED = tuple((m, q) for m in S.ALLOWED_MODES for q in S.MEASURED_POINTS[m])
TOL = {"vehicle": 1.0, "person": 1.2}
BLOCK = 50


def _actor(path: str) -> S.PilotRestrictedActor:
    actor = S.PilotRestrictedActor(RM.run5_model_config())
    actor.load_state_dict(torch.load(path, weights_only=True))
    actor.eval()
    return actor


def run_policy(spec: dict, seed: int, shift: float, n: int, authorization: str) -> list[dict]:
    torch.set_num_threads(1)
    shared = CR.load_shared_sources()
    lat, cat, trace, fb = components("EVAL")
    col = C.PilotRun5CollectorV1(latency=lat, catalog=cat, trace=trace, fallback=fb, arm="CONTROL", seed=seed,
                                 authorization=Path(authorization), binding_dir=BINDING, shared_sources=shared,
                                 receipt_shift_ms=shift)
    actor = _actor(spec["path"]) if spec["kind"] == "actor" else None
    rows = []
    for i in range(n):
        if actor is None:
            mode, q = spec["mode_id"], spec["q_e4"]
        else:
            with torch.no_grad():
                d = actor.deterministic_execution(torch.tensor((col.current_state_features(),), dtype=torch.float32))
            mode, q = int(d.mode_index[0]), int(d.q_e4[0])
        col.collect_transition(orch.ModeledActionRequestV1(decision_ordinal=i, mode_id=mode, q_e4=q,
                                                           source="STOCHASTIC_ACTOR", warmup_q_bin_index=None))
        diag = col.diagnostics[-1]
        comps = cat.components(diag["reward_scene_key"], mode_id=mode, q_e4=q)
        rows.append(dict({k: diag[k] for k in ("ordinal", "mode_id", "q_e4", "terminal", "excluded", "control_reward",
                                               "candidate_reward", "q_perc", "receipt_ms", "fallback_frames_after",
                                               "mcs_gap", "interp_weight")},
                         components=[(w, c) for w, c in comps]))
    return rows


def _wq(values, weights, p):
    if not values:
        return None
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    c = np.cumsum(w) / w.sum()
    return float(v[min(len(v) - 1, int(np.searchsorted(c, p / 100.0)))])


def quality_block(rows: list[dict]) -> dict:
    out = {}
    for cls in TOL:
        el = tp = fp = hits = inter = union = 0.0
        errs, ws = [], []
        for r in rows:
            for w, c in r["components"]:
                k = c[cls]
                el += w * k["eligible"]; tp += w * k["tp"]; fp += w * k["fp"]; hits += w * k["hits"]
                inter += w * k["inter"]; union += w * k["union"]
                errs += k["errors"]; ws += [w] * len(k["errors"])
        beyond = sum(w for e, w in zip(errs, ws) if e > TOL[cls])
        out[cls] = dict(eligible=round(el, 3), recall_within_tolerance=None if el == 0 else hits / el,
                        recall_gate=None if el == 0 else tp / el, precision=None if tp + fp == 0 else tp / (tp + fp),
                        fp_per_decision=fp / max(1, len(rows)), seg_iou=None if union == 0 else inter / union,
                        error_p50=_wq(errs, ws, 50), error_p90=_wq(errs, ws, 90), error_p95=_wq(errs, ws, 95),
                        error_max=max(errs) if errs else None,
                        share_beyond_tolerance=None if not ws else beyond / sum(ws))
    return out


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    included = [r for r in rows if not r["excluded"]]
    timely = [r for r in rows if r["receipt_ms"] is not None and r["receipt_ms"] <= 170.0]
    fallback_frames = sum(r["fallback_frames_after"] or 0 for r in rows) + C.STARTUP_FALLBACKS
    modes = collections.Counter(r["mode_id"] for r in rows)
    q_by_mode = {m: [r["q_e4"] for r in rows if r["mode_id"] == m] for m in S.ALLOWED_MODES}
    return dict(
        decisions=n, opportunities=n + fallback_frames, fallback_frames=fallback_frames,
        excluded_timely_undefined=sum(r["excluded"] for r in rows),
        late_undefined=sum(1 for r in rows if r["terminal"] == "TIMEOUT" and r["q_perc"] is None),
        timely=len(timely), timely_rate=len(timely) / n,
        timely_rate_per_opportunity=len(timely) / (n + fallback_frames),
        success=sum(r["terminal"] == "SUCCESS" for r in rows), timeouts=sum(r["terminal"] == "TIMEOUT" for r in rows),
        control_reward_mean=float(np.mean([r["control_reward"] for r in included])),
        candidate_reward_mean=float(np.mean([r["candidate_reward"] for r in included])),
        q_perc_mean_timely_defined=float(np.mean([r["q_perc"] for r in timely if r["q_perc"] is not None] or [np.nan])),
        receipt_p50=float(np.median([r["receipt_ms"] for r in rows if r["receipt_ms"] is not None])),
        receipt_p95=float(np.percentile([r["receipt_ms"] for r in rows if r["receipt_ms"] is not None], 95)),
        mode_share={m: modes.get(m, 0) / n for m in S.ALLOWED_MODES},
        q_quantiles={m: (None if not v else [int(np.percentile(v, p)) for p in (5, 50, 95)]) for m, v in q_by_mode.items()},
        quality_all=quality_block(rows), quality_timely=quality_block(timely), mcs_gap_mean=float(np.mean([r["mcs_gap"] for r in rows])),
        interpolated_share=sum(r["interp_weight"] is not None for r in rows) / n)


def block_ci(diffs: list[float], seed: int = 0) -> list[float]:
    d = np.asarray(diffs, float)
    nb = len(d) // BLOCK
    blocks = d[: nb * BLOCK].reshape(nb, BLOCK).mean(1)
    rng = np.random.default_rng(seed)
    means = blocks[rng.integers(0, nb, (4000, nb))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _job(args):
    label, spec, seed, shift, n, auth = args
    return label, seed, shift, run_policy(spec, seed, shift, n, auth)


def evaluate(runs_dir: Path, out: Path, authorization: Path, n: int, workers: int) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    policies = {}
    for result in sorted(Path(runs_dir).glob("*/RUN_RESULT.json")):
        r = json.loads(result.read_text())
        policies[f"{r['arm']}_seed{r['seed']}"] = dict(kind="actor", path=r["actor"]["path"], sha256=r["actor"]["sha256"])
    for m, q in FIXED:
        policies[f"FIXED_m{m}_q{q}"] = dict(kind="fixed", mode_id=m, q_e4=q)
    A.verify(authorization, arm="CONTROL", seed=EVAL_SEEDS[0], binding_dir=BINDING)
    jobs = [(label, spec, seed, shift, n, str(authorization)) for label, spec in policies.items()
            for seed in EVAL_SEEDS for shift in SHIFTS]
    raw = collections.defaultdict(dict)
    with mp.get_context("spawn").Pool(workers) as pool:
        for label, seed, shift, rows in pool.imap_unordered(_job, jobs):
            raw[(label, shift)][seed] = rows
    results = {}
    with open(out / "EVAL_DECISIONS.jsonl", "x") as stream:
        for (label, shift), per_seed in sorted(raw.items()):
            for seed, rows in sorted(per_seed.items()):
                for r in rows:
                    stream.write(json.dumps(dict({k: v for k, v in r.items() if k != "components"}, policy=label,
                                                 seed=seed, shift=shift)) + "\n")
            pooled = [r for s in EVAL_SEEDS for r in per_seed[s]]
            results.setdefault(label, {})[str(shift)] = dict(pooled=summarize(pooled),
                                                             per_seed={str(s): {k: v for k, v in summarize(per_seed[s]).items()
                                                                                if k in ("control_reward_mean", "candidate_reward_mean",
                                                                                         "timely_rate")} for s in EVAL_SEEDS})
    comparisons = {}
    for ref in ("FIXED_m11_q3000", "FIXED_m11_q7000"):
        for label in policies:
            if label == ref:
                continue
            for shift in SHIFTS:
                entry = {}
                for metric in ("control", "candidate", "timely"):
                    diffs = []
                    for s in EVAL_SEEDS:
                        a, b = raw[(label, shift)][s], raw[(ref, shift)][s]
                        for x, y in zip(a, b):           # CRN pairing by decision ordinal (same scene draw)
                            if metric == "timely":
                                diffs.append(float(x["receipt_ms"] is not None and x["receipt_ms"] <= 170)
                                             - float(y["receipt_ms"] is not None and y["receipt_ms"] <= 170))
                            elif not x["excluded"] and not y["excluded"]:
                                diffs.append(x[f"{metric}_reward"] - y[f"{metric}_reward"])
                    entry[metric] = dict(mean=float(np.mean(diffs)), block_ci=block_ci(diffs), n=len(diffs))
                comparisons[f"{label}_minus_{ref}|shift{shift}"] = entry
    document = dict(schema="splitfusion.run5_restricted_pilot.evaluation.v1", policies=policies, eval_seeds=EVAL_SEEDS,
                    shifts=SHIFTS, decisions_per_seed=n, results=results, comparisons=comparisons,
                    evidence_label=A.EVIDENCE_LABEL, live_claim=False)
    (out / "EVALUATION.json").write_text(json.dumps(document, indent=1, sort_keys=True, default=str))
    return document


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--decisions", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    evaluate(args.runs, args.out, args.authorization, args.decisions, args.workers)
