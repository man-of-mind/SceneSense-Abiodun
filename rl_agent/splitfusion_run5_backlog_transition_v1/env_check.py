"""Environment check (no training): (1) disabled correction reproduces the old collector exactly; (2) enabled, the
deployed actor (unchanged, deterministic act_on_vector) in the binding_01 environment, old vs corrected transition on
identical seeds, reported per hidden profile. Diagnostic only; not a policy evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch

from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as orch
from rl_agent.splitfusion_run5_full_live_v1.actor import load_registered_actor
from rl_agent.splitfusion_run5_full_v1 import campaign as FCMP
from rl_agent.splitfusion_run5_full_v1 import collector as FC

from .collector import BacklogTransitionV1, FullRun5BacklogCollectorV1


def drive(collector, actor, n: int) -> list[dict]:
    rows = []
    for i in range(n):
        d = actor.act_on_vector(tuple(float(v) for v in collector.current_state_features()))
        req = orch.ModeledActionRequestV1(decision_ordinal=i, mode_id=int(d.mode_id), q_e4=int(d.q_e4),
                                          source="STOCHASTIC_ACTOR", warmup_q_bin_index=None)
        t = collector.collect_transition(req)
        diag = dict(collector.diagnostics[-1])
        diag["transition_digest"] = None if t is None else t.digest
        rows.append(diag)
    return rows


def summarise(rows: list[dict]) -> dict:
    out = {}
    for prof in ("FAVORABLE_STABLE", "MID_VARIABLE", "ADVERSE_STABLE", "FADE_RECOVERY", "ALL"):
        rs = [r for r in rows if prof == "ALL" or r["profile"] == prof]
        inc = [r for r in rs if not r["excluded"]]
        B = [r["backlog_bytes"] for r in rs]
        out[prof] = dict(n=len(rs), backlog_pos=sum(b > 0 for b in B) / len(B), backlog_gt1kb=sum(b > 1000 for b in B) / len(B),
                         backlog_p99=sorted(B)[int(.99 * len(B))], backlog_max=max(B),
                         timely=sum(r["terminal"] == "SUCCESS" for r in inc) / len(inc),
                         reward_mean=statistics.fmean(r["reward"] for r in inc),
                         wire_p50_kb=statistics.median(r["wire_bytes"] for r in rs) / 1e3,
                         q_p50=statistics.median(r["q_e4"] for r in rs),
                         mode11=sum(r["mode_id"] == 11 for r in rs) / len(rs),
                         timely_after_gt1kb=(lambda s: sum(r["terminal"] == "SUCCESS" for r in s) / len(s) if s else None)(
                             [r for r in inc if r["backlog_bytes"] > 1000]),
                         n_gt1kb=sum(1 for r in inc if r["backlog_bytes"] > 1000),
                         q_p50_at_gt1kb=(statistics.median(r["q_e4"] for r in rs if r["backlog_bytes"] > 1000)
                                         if any(r["backlog_bytes"] > 1000 for r in rs) else None))
    return out


def run(fit: Path, timing: Path, out: Path, n_identity: int, n_compare: int, seed: int) -> None:
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    shared, latency = FCMP.components()
    actor = load_registered_actor()
    # (1) disabled reproduces the old collector exactly
    old = drive(FC.FullRun5CollectorV1(latency=latency, seed=seed, shared_sources=shared), actor, n_identity)
    off_c = FullRun5BacklogCollectorV1(latency=latency, seed=seed, shared_sources=shared, backlog_transition=None)
    off = drive(off_c, actor, n_identity)
    same = json.dumps(old, sort_keys=True, default=str) == json.dumps(off, sort_keys=True, default=str)
    print(f"(1) disabled == old over {n_identity} decisions: {same}; positive backlog in old: {sum(r['backlog_bytes'] > 0 for r in old)}")
    # (2) corrected vs old, same seed, deterministic deployed actor
    bt = BacklogTransitionV1(fit, timing)
    old2 = old if n_compare == n_identity else drive(FC.FullRun5CollectorV1(latency=latency, seed=seed, shared_sources=shared), actor, n_compare)
    new = drive(FullRun5BacklogCollectorV1(latency=latency, seed=seed, shared_sources=shared, backlog_transition=bt), actor, n_compare)
    res = dict(schema="splitfusion.run5_backlog_transition.env_check.v1", seed=seed, identity=dict(n=n_identity, identical=same),
               old=summarise(old2), corrected=summarise(new), flags=bt.flags, actor="deployed seed-43 bb9b0d25 (deterministic)",
               note="diagnostic: the deployed actor was trained without backlog states; not a policy evaluation")
    for prof in ("FAVORABLE_STABLE", "MID_VARIABLE", "ADVERSE_STABLE", "FADE_RECOVERY", "ALL"):
        o, c = res["old"][prof], res["corrected"][prof]
        print(f"{prof:17s} n {c['n']:4d} | B>0 {o['backlog_pos']:.3f}->{c['backlog_pos']:.3f} B>1KB {o['backlog_gt1kb']:.3f}->{c['backlog_gt1kb']:.3f} "
              f"(max {c['backlog_max']/1e3:.0f} KB) | timely {o['timely']:.3f}->{c['timely']:.3f} reward {o['reward_mean']:.3f}->{c['reward_mean']:.3f} "
              f"| wire P50 {o['wire_p50_kb']:.0f}->{c['wire_p50_kb']:.0f} KB q P50 {o['q_p50']:.0f}->{c['q_p50']:.0f} | after B>1KB: n {c['n_gt1kb']} timely {c['timely_after_gt1kb']} q P50 {c['q_p50_at_gt1kb']}")
    print("flags:", bt.flags)
    (out / "ENV_CHECK.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fit", required=True)
    p.add_argument("--timing", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-identity", type=int, default=600)
    p.add_argument("--n-compare", type=int, default=2400)
    p.add_argument("--seed", type=int, default=9017)
    a = p.parse_args()
    run(Path(a.fit), Path(a.timing), Path(a.out), a.n_identity, a.n_compare, a.seed)
