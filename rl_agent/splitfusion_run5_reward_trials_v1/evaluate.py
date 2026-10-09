"""Held-out evaluation of the reward-trial actors (offline, modelled; protocol fixed in PREREGISTRATION.md).

Policies: every RUN_RESULT.json actor in ``--runs`` (deterministic deployment rule: argmax mode at the conditional-
mean q) plus the deployed historical full Run-5 seed-43 actor (reference only). Environments: E1 = held_scene
catalogue + HELDOUT_GCV2 pool (primary); E2 = held_scene + RETENTION_V1 (secondary). Original Run-5 channel (four
profiles), evaluation seeds 9017/9029/9043, common random numbers. Per decision: action, payload, profile, outcome,
receipt, all three arms' quality values, and per-class expected counts for the policy's (scene, mode, q) and for
the per-scene reference cell (mode 0, q 0 = noAE/UINT8, no drop).

  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= python3 -m rl_agent.splitfusion_run5_reward_trials_v1.evaluate \
      --runs experiments/splitfusion_run5_reward_trials_v1/runs_01 --out experiments/splitfusion_run5_reward_trials_v1/eval_01
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as orch
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_models as RM
from rl_agent.splitfusion_hybrid_sac_v1.hybrid_sac_models import ConditionalHybridActor, _local_torch_seed
from rl_agent.splitfusion_run5_full_v1 import support as S

from . import campaign as CAMP, collector as C, rewards as R

ROOT = CAMP.ROOT
EVAL_SEEDS = (9017, 9029, 9043)
DECISIONS = 6000
ENVIRONMENTS = {"E1": ("held_scene", "HELDOUT_GCV2"), "E2": ("held_scene", "RETENTION_V1")}
HISTORICAL = dict(label="HIST_seed43", path="experiments/splitfusion_run5_full_v1/runs_01/seed43/actor_u10000.pt",
                  sha256="bb9b0d256de79b4498cd622e1daa59edcf43ff3072dc9f675b265aac579ed3ce")
CLASSES = ("vehicle", "person")
FIELDS = ("eligible", "hits", "tp", "fp", "inter", "union")
BLOCK = 50
GUARDRAILS = {"loc": {"vehicle": 0.95, "person": 0.90}, "seg": {"vehicle": 0.85, "person": 0.80},
              "precision": {"vehicle": 0.95, "person": 0.90}}


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_actor(path: Path) -> ConditionalHybridActor:
    with _local_torch_seed(0):                                    # RNG-neutral construction (as the live loader)
        actor = ConditionalHybridActor(RM.run5_model_config())
    actor.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    actor.eval().requires_grad_(False)
    low, high = actor.active_q_e4_bounds()
    assert tuple(zip(map(int, low), map(int, high))) == S.BOUNDS, "actor support differs"
    return actor


def act(actor, features) -> tuple[int, int]:
    state = torch.tensor((tuple(float(v) for v in features),), dtype=torch.float32)
    with torch.inference_mode():
        execution = actor.deterministic_execution(state)
    return int(execution.mode_index[0]), int(execution.q_e4[0])


def expected(components) -> dict:
    out = {}
    for cls in CLASSES:
        for f in FIELDS:
            out[f"{cls[0]}_{f}"] = sum(w * c[cls][f] for w, c in components)
    return out


def run_policy(label: str, path: str, env: str, seed: int, n: int) -> list[dict]:
    torch.set_num_threads(1)
    split, pool = ENVIRONMENTS[env]
    shared, latency = CAMP.components("C0", split=split, pool=pool)
    cat = shared["catalog"]
    col = C.TrialRun5CollectorV1(latency=latency, seed=seed, shared_sources=shared, arm="C0")
    actor = load_actor(ROOT / path)
    rows = []
    for i in range(n):
        mode, q = act(actor, col.current_state_features())
        col.collect_transition(orch.ModeledActionRequestV1(decision_ordinal=i, mode_id=mode, q_e4=q,
                                                           source="STOCHASTIC_ACTOR", warmup_q_bin_index=None))
        d = col.diagnostics[-1]
        key = d["reward_scene_key"]
        row = dict(i=i, mode=mode, q=q, terminal=d["terminal"], excluded=d["excluded"], receipt=d["receipt_ms"],
                   timely=d["receipt_ms"] is not None and d["receipt_ms"] <= 170.0, profile=d["profile"],
                   snr=d["snr_db"], mcs=d["prior_ul_mcs"], backlog=d["backlog_bytes"], wire=d["wire_bytes"],
                   scene=key, q_perc=d["q_perc"], late_uplink=d["uplink_late"],
                   gated="CONFIRM_SEND_GATED" in d["floors"],
                   value={a: cat.value_for(a, key, mode_id=mode, q_e4=q) for a in R.ARMS})
        row.update(expected(cat.components(key, mode_id=mode, q_e4=q)))
        row.update({f"ref_{k}": v for k, v in expected([(1.0, cat.reference_components(key))]).items()})
        rows.append(row)
    return rows


# ------------------------------------------------------------------------------------------------ summaries
def _ratio(a, b):
    return None if b == 0 else a / b


def quality(rows: list[dict], prefix: str = "", timely_only: bool = False) -> dict:
    out = {}
    for cls in CLASSES:
        c = cls[0]
        tot = {f: sum(r[f"{prefix}{c}_{f}"] * (r["timely"] if timely_only and f == "hits" else 1) for r in rows)
               for f in FIELDS}
        out[cls] = dict(eligible=tot["eligible"], recall_within_tolerance=_ratio(tot["hits"], tot["eligible"]),
                        recall_gate=_ratio(tot["tp"], tot["eligible"]), precision=_ratio(tot["tp"], tot["tp"] + tot["fp"]),
                        fp_per_decision=tot["fp"] / max(1, len(rows)), seg_iou=_ratio(tot["inter"], tot["union"]))
    return out


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    q_all = quality(rows)
    q_timely = quality(rows, timely_only=True)                   # hits only when timely (misses/late fail)
    ref = quality(rows, prefix="ref_")
    primary = 0.4 * q_timely["vehicle"]["recall_within_tolerance"] + 0.6 * q_timely["person"]["recall_within_tolerance"]
    retention = {}
    for cls in CLASSES:
        retention[cls] = dict(loc=_ratio(q_all[cls]["recall_within_tolerance"], ref[cls]["recall_within_tolerance"]),
                              seg=_ratio(q_all[cls]["seg_iou"], ref[cls]["seg_iou"]),
                              precision=_ratio(q_all[cls]["precision"], ref[cls]["precision"]))
    timely_rows = [r for r in rows if r["timely"]]
    retention_timely = {cls: dict(loc=_ratio(quality(timely_rows)[cls]["recall_within_tolerance"],
                                             quality(timely_rows, prefix="ref_")[cls]["recall_within_tolerance"]),
                                  seg=_ratio(quality(timely_rows)[cls]["seg_iou"], quality(timely_rows, prefix="ref_")[cls]["seg_iou"]))
                        for cls in CLASSES}
    included = [r for r in rows if not r["excluded"]]
    realized = {a: float(np.mean([(r["value"][a] if r["timely"] else 0.0) for r in included])) for a in R.ARMS}
    modes = collections.Counter(r["mode"] for r in rows)
    receipts = [r["receipt"] for r in rows if r["receipt"] is not None]
    by_profile = {}
    for p in sorted({r["profile"] for r in rows}):
        pr = [r for r in rows if r["profile"] == p]
        by_profile[p] = dict(decisions=len(pr), timely=float(np.mean([r["timely"] for r in pr])),
                             wire_p50=float(np.median([r["wire"] for r in pr])),
                             q_p50=float(np.median([r["q"] for r in pr])),
                             mode11_share=sum(r["mode"] == 11 for r in pr) / len(pr),
                             mode_share={m: sum(r["mode"] == m for r in pr) / len(pr) for m in range(12) if any(r["mode"] == m for r in pr)},
                             snr_p50=float(np.median([r["snr"] for r in pr])),
                             timely_loc=0.4 * quality(pr, timely_only=True)["vehicle"]["recall_within_tolerance"]
                             + 0.6 * quality(pr, timely_only=True)["person"]["recall_within_tolerance"])
    return dict(decisions=n, primary_timely_loc=primary, timely=float(np.mean([r["timely"] for r in rows])),
                excluded_timely_undefined=sum(r["excluded"] for r in rows),
                late_undefined=sum(1 for r in rows if not r["timely"] and r["q_perc"] is None),
                otr_mean=float(np.mean([r["timely"] * _y(r) for r in included])),
                realized_reward_by_formula=realized, quality_all=q_all, quality_timely_hits=q_timely,
                reference=ref, retention=retention, retention_timely_subset=retention_timely,
                receipt_p50=float(np.median(receipts)), receipt_p95=float(np.percentile(receipts, 95)),
                wire_p5_p50_p95=[float(np.percentile([r["wire"] for r in rows], p)) for p in (5, 50, 95)],
                q_p5_p50_p95=[float(np.percentile([r["q"] for r in rows], p)) for p in (5, 50, 95)],
                mode_share={m: modes.get(m, 0) / n for m in range(12)}, by_profile=by_profile,
                uplink_late=sum(r["late_uplink"] for r in rows), confirm_gated=sum(r["gated"] for r in rows))


def _y(r) -> float:
    """Class-weighted within-tolerance recall of the decision (eligible-class renormalised), for OTR."""
    num = den = 0.0
    for cls, w in (("v", 0.4), ("p", 0.6)):
        if r[f"{cls}_eligible"] > 0:
            num += w * r[f"{cls}_hits"] / r[f"{cls}_eligible"]
            den += w
    return num / den if den else 0.0


def contributions(rows: list[dict], ev: float, ep: float) -> np.ndarray:
    return np.asarray([r["timely"] * (0.4 * r["v_hits"] / ev + 0.6 * r["p_hits"] / ep) for r in rows])


def block_ci(diffs: np.ndarray, seed: int = 0) -> list[float]:
    nb = len(diffs) // BLOCK
    blocks = diffs[: nb * BLOCK].reshape(nb, BLOCK).sum(1)
    rng = np.random.default_rng(seed)
    sums = blocks[rng.integers(0, nb, (4000, nb))].sum(1) * (len(diffs) / (nb * BLOCK))
    return [float(np.percentile(sums, 2.5)), float(np.percentile(sums, 97.5))]


def _job(args):
    label, path, env, seed, n = args
    return label, env, seed, run_policy(label, path, env, seed, n)


def evaluate(runs_dir: Path, out: Path, n: int, workers: int) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    policies = {}
    for result in sorted(Path(runs_dir).glob("*/RUN_RESULT.json")):
        r = json.loads(result.read_text())
        assert r["updates"] == 10_000 and sha(ROOT / r["actor"]["path"]) == r["actor"]["sha256"]
        policies[f"{r['arm']}_seed{r['seed']}"] = dict(arm=r["arm"], seed=r["seed"], path=r["actor"]["path"],
                                                       sha256=r["actor"]["sha256"])
    assert sha(ROOT / HISTORICAL["path"]) == HISTORICAL["sha256"], "historical actor drifted"
    policies[HISTORICAL["label"]] = dict(arm="HIST", seed=43, path=HISTORICAL["path"], sha256=HISTORICAL["sha256"])
    jobs = [(label, p["path"], env, seed, n) for label, p in policies.items() for env in ENVIRONMENTS for seed in EVAL_SEEDS]
    raw = collections.defaultdict(dict)
    with mp.get_context("spawn").Pool(workers) as pool:
        for label, env, seed, rows in pool.imap_unordered(_job, jobs):
            raw[(label, env)][seed] = rows
    with gzip.open(out / "EVAL_DECISIONS.jsonl.gz", "wt") as stream:
        for (label, env), per_seed in sorted(raw.items()):
            for seed, rows in sorted(per_seed.items()):
                for r in rows:
                    stream.write(json.dumps(dict(r, policy=label, env=env, eval_seed=seed), separators=(",", ":")) + "\n")
    results = {}
    for (label, env), per_seed in sorted(raw.items()):
        pooled = [r for s in EVAL_SEEDS for r in per_seed[s]]
        results.setdefault(label, {})[env] = dict(
            pooled=summarize(pooled),
            per_eval_seed={str(s): {k: v for k, v in summarize(per_seed[s]).items()
                                    if k in ("primary_timely_loc", "timely")} for s in EVAL_SEEDS})
    # Scene streams must be identical across policies (common random numbers).
    for env in ENVIRONMENTS:
        for seed in EVAL_SEEDS:
            streams = {tuple(r["scene"] for r in raw[(label, env)][seed]) for label in policies}
            assert len(streams) == 1, f"{env}/{seed}: scene streams differ across policies"
    paired = {}
    for env in ENVIRONMENTS:
        for arm in ("C1", "C2", "HIST"):
            for seed in (17, 29, 43):
                x = f"{arm}_seed{seed}" if arm != "HIST" else HISTORICAL["label"]
                if x not in policies:
                    continue
                y = f"C0_seed{seed}"
                a = [r for s in EVAL_SEEDS for r in raw[(x, env)][s]]
                b = [r for s in EVAL_SEEDS for r in raw[(y, env)][s]]
                ev, ep = sum(r["v_eligible"] for r in a), sum(r["p_eligible"] for r in a)
                assert (ev, ep) == (sum(r["v_eligible"] for r in b), sum(r["p_eligible"] for r in b))
                d = contributions(a, ev, ep) - contributions(b, ev, ep)        # sums to the pooled primary difference
                paired[f"{env}|{x}-{y}"] = dict(primary_diff=float(d.sum()), block_ci=block_ci(d))
                if arm == "HIST":
                    break
    document = dict(schema="splitfusion.run5_reward_trials.evaluation.v1", policies=policies, environments=ENVIRONMENTS,
                    eval_seeds=EVAL_SEEDS, decisions_per_seed=n, results=results, paired_crn=paired,
                    decision=decide(results, policies), evidence_label="RUN5_REWARD_TRIAL_MODELED", live_claim=False)
    (out / "EVALUATION.json").write_text(json.dumps(document, indent=1, sort_keys=True, default=str))
    return document


# ------------------------------------------------------------------------------------------------ decision rule
def arm_values(results, policies, env, getter):
    out = collections.defaultdict(list)
    for label, p in policies.items():
        if p["arm"] in R.ARMS:
            out[p["arm"]].append((p["seed"], getter(results[label][env]["pooled"])))
    return {a: sorted(v) for a, v in out.items()}


def decide(results, policies) -> dict:
    """Pre-registered rule (PREREGISTRATION.md); applied mechanically."""
    prim = lambda e: arm_values(results, policies, e, lambda s: s["primary_timely_loc"])
    mean = lambda vals: float(np.mean([v for _, v in vals]))
    p1, p2 = prim("E1"), prim("E2")
    timely = arm_values(results, policies, "E1", lambda s: s["timely"])
    ret = {(k, cls): arm_values(results, policies, "E1", lambda s, k=k, cls=cls: s["retention"][cls][k])
           for k in GUARDRAILS for cls in ("vehicle", "person")}
    person_t = arm_values(results, policies, "E1", lambda s: s["quality_timely_hits"]["person"]["recall_within_tolerance"])
    c0 = mean(p1["C0"])
    verdicts = {}
    for x in ("C1", "C2"):
        checks = {}
        delta = mean(p1[x]) - c0
        checks["1_delta_primary_E1_ge_0.02"] = dict(value=delta, passed=delta >= 0.02)
        checks["2_every_seed_above_C0_mean"] = dict(value=[v for _, v in p1[x]], c0_mean=c0,
                                                     passed=all(v > c0 for _, v in p1[x]))
        dt = mean(timely[x]) - mean(timely["C0"])
        checks["3_timeliness_not_below_C0_minus_0.02"] = dict(value=dt, passed=dt >= -0.02)
        guard = {}
        for (k, cls), vals in ret.items():
            xv, cv, tier = mean(vals[x]), mean(vals["C0"]), GUARDRAILS[k][cls]
            shared_miss = cv < tier
            guard[f"{k}_{cls}"] = dict(value=xv, c0=cv, tier=tier, shared_miss=shared_miss,
                                       passed=(xv >= tier) or (shared_miss and xv >= cv - 0.01))
        checks["4_guardrails"] = dict(detail=guard, passed=all(g["passed"] for g in guard.values()))
        d2 = mean(p2[x]) - mean(p2["C0"])
        checks["5_direction_E2"] = dict(value=d2, passed=d2 > 0)
        verdicts[x] = dict(checks=checks, supported=all(c["passed"] for c in checks.values()), delta_E1=delta,
                           person_timely_recall=mean(person_t[x]))
    supported = [x for x in ("C1", "C2") if verdicts[x]["supported"]]
    if not supported:
        selected, outcome = None, "NO_CLEAR_WINNER_RETAIN_C0_OR_CURRENT_DEPLOYMENT"
    elif len(supported) == 1:
        selected, outcome = supported[0], "SUPPORTED"
    else:
        a, b = supported
        if abs(verdicts[a]["delta_E1"] - verdicts[b]["delta_E1"]) >= 0.005:
            selected = max(supported, key=lambda x: verdicts[x]["delta_E1"])
        else:
            selected = max(supported, key=lambda x: verdicts[x]["person_timely_recall"])
        outcome = "SUPPORTED"
    candidate_seed = None
    if selected:
        ordered = sorted(p1[selected], key=lambda t: t[1])
        candidate_seed = ordered[1][0]
    return dict(rule="PREREGISTRATION.md", verdicts=verdicts, outcome=outcome, selected_arm=selected,
                candidate_seed_median_primary=candidate_seed, c0_primary_mean_E1=c0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--decisions", type=int, default=DECISIONS)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    doc = evaluate(args.runs, args.out, args.decisions, args.workers)
    print(json.dumps(doc["decision"], indent=1, default=str))
