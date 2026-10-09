"""Train ONE arm (CONTROL or CANDIDATE), seed 17, 10,000 updates, in the low-MCS retrain configuration.

Identical to ``splitfusion_run5_lowmcs_retrain_v1.campaign.train`` (same orchestrator, collector, warm-up, SAC settings,
checkpoints, authorization label = the low-MCS retrain registration, so CONTROL must reproduce its NEW_seed17 digest);
the only arm difference is the backlog transition (CONTROL: disabled; CANDIDATE: frozen corrected transition).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch

from rl_agent.splitfusion_run5_full_v1 import campaign as FCAMP
from rl_agent.splitfusion_run5_lowmcs_retrain_v1 import campaign as LCAMP
from rl_agent.splitfusion_run5_lowmcs_retrain_v1 import registration as LREG
from rl_agent.splitfusion_run5_reward_trials_v1 import trainer as TR

from . import env as E
from . import rules as RU

SEED = 17
LREG_FILE = LREG.ROOT / "experiments/splitfusion_run5_lowmcs_retrain_v1/registration_01/REGISTRATION.json"
DIAG = tuple(LCAMP.DIAG_KEYS) + ("backlog_bytes",)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def train(arm: str, out: Path, registration: Path) -> dict:
    assert arm in ("CONTROL", "CANDIDATE") and out.is_absolute()
    torch.set_num_threads(TR.CONFIG.torch_intraop_threads)
    run_dir = out / f"{arm}_seed{SEED}"
    run_dir.mkdir(parents=True, exist_ok=False)
    LREG.verify(LREG_FILE, seed=SEED)                       # the inherited configuration is unchanged
    auth = sha(LREG_FILE)
    shared, latency = E.components()
    bt = E.transition() if arm == "CANDIDATE" else None
    factory = lambda: E.TrialBacklogCollectorV1(latency=latency, seed=SEED, shared_sources=shared, backlog_transition=bt)
    start = time.time()
    o = TR.TrialOrchestratorV1(collector_factory=factory, seed=SEED, authorization_sha256=auth)
    src = {n: sha(Path(__file__).parent / n) for n in ("env.py", "campaign.py", "rules.py")}
    src.update({f"transition/{n}": sha(RU.PACKAGE.parent / "splitfusion_run5_backlog_transition_v1" / n)
                for n in ("transition.py", "collector.py")})
    (run_dir / "RUN_BINDING.json").write_text(json.dumps(dict(
        o.binding_document, arm=arm, binding_sha256=o.binding_sha256, latency_binding_sha256=latency.binding_sha256,
        collector_binding_sha256=o.collector.collector_binding_sha256,
        backlog_transition=None if bt is None else dict(binding_sha256=bt.binding_sha256, fit=str(RU.FIT_JSON),
                                                        fit_sha256=sha(RU.FIT_JSON), timing_sha256=sha(RU.TIMING_JSON)),
        pair_registration=str(registration), pair_registration_sha256=sha(registration), authorization_label=str(LREG_FILE),
        source_sha256=src, lowmcs_source_digest=LREG.source_digest(), lowmcs_correction=latency.correction.to_dict(),
        host=dict(pid=os.getpid(), threads=torch.get_num_threads())), indent=1, sort_keys=True, default=str))
    metrics = open(run_dir / "metrics.jsonl", "x", buffering=1)
    o.warmup_phase()
    checkpoints = {0: o.save(run_dir / "checkpoints/u00000.pt")}
    while o.update_count < RU.REGISTRATION["updates"]:
        m = o.step_update()
        metrics.write(json.dumps(FCAMP._metrics(o.update_count, m, time.time() - start)) + "\n")
        if o.update_count in (2500, 5000, 7500, 10000):
            checkpoints[o.update_count] = o.save(run_dir / f"checkpoints/u{o.update_count:05d}.pt")
    metrics.close()
    diag = o.collector.diagnostics
    with open(run_dir / "decisions.jsonl", "x") as f:
        for d in diag:
            f.write(json.dumps({k: d.get(k) for k in DIAG}) + "\n")
    actor_path = run_dir / "actor_u10000.pt"
    torch.save(o.actor.state_dict(), actor_path)
    def bexp(rows):
        n = len(rows)
        return dict(decisions=n, pos=sum(d["backlog_bytes"] > 0 for d in rows), gt1kb=sum(d["backlog_bytes"] > 1000 for d in rows),
                    by_profile={p: sum(1 for d in rows if d["profile"] == p and d["backlog_bytes"] > 1000)
                                for p in ("FAVORABLE_STABLE", "MID_VARIABLE", "ADVERSE_STABLE", "FADE_RECOVERY")},
                    timely_at_gt1kb=(lambda s: sum(d["terminal"] != "TIMEOUT" for d in s) / len(s) if s else None)(
                        [d for d in rows if d["backlog_bytes"] > 1000]))
    tail = diag[-4000:]
    result = dict(arm=arm, seed=SEED, updates=o.update_count, decisions=o.decision_count, seconds=round(time.time() - start, 1),
                  final_digest=o.state_digest(), checkpoint_sha256={str(k): v["sha256"] for k, v in checkpoints.items()},
                  actor=dict(path=str(actor_path.relative_to(LREG.ROOT)), sha256=sha(actor_path)),
                  backlog_exposure=dict(all=bexp(diag), warmup=bexp(diag[:288]), stochastic=bexp(diag[288:]), last_4000=bexp(tail)),
                  timing_flags=None if bt is None else dict(bt.flags), exposure=LCAMP.exposure(diag),
                  last_4000_timely=sum(d["terminal"] != "TIMEOUT" for d in tail) / len(tail),
                  last_4000_mode_share={m: sum(d["mode_id"] == m for d in tail) / len(tail) for m in range(12)},
                  last_4000_q_p50=sorted(d["q_e4"] for d in tail)[len(tail) // 2],
                  last_4000_wire_p50=sorted(d["wire_bytes"] for d in tail)[len(tail) // 2])
    (run_dir / "RUN_RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--registration", required=True, type=Path)
    a = p.parse_args()
    r = train(a.arm, a.out, a.registration)
    print(json.dumps({k: r[k] for k in ("arm", "seconds", "backlog_exposure", "last_4000_timely", "last_4000_q_p50", "timing_flags")}, default=str))
