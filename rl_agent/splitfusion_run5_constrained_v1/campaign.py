"""Train one seed of the constrained candidate to 10,000 updates (create-only; absolute paths).

  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 python3 -m rl_agent.splitfusion_run5_constrained_v1.campaign \
      --seed 17 --out <ABS runs dir> --registration <ABS REGISTRATION.json> [--smoke-at 250]

Procedure = the reward-trial / low-MCS campaign (original warm-up, 4 transitions per update, finite-loss check every update,
smoke/resume proof by deterministic re-execution incl. the dual state), with the constrained orchestrator. Latency: the
frozen low-MCS candidate on binding_02 TRAIN_GCV2. Logged per update: losses, entropy, grad norms, minibatch reward,
lambda; per dual update: window means of g/kappa (DUAL.jsonl).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from rl_agent.splitfusion_run5_full_v1 import campaign as FCAMP
from rl_agent.splitfusion_run5_lowmcs_retrain_v1 import campaign as LCAMP
from rl_agent.splitfusion_run5_reward_trials_v1 import collector as TC

from . import objective as O
from . import registration as REG
from . import trainer as TR

ROOT = REG.ROOT
DIAG_KEYS = LCAMP.DIAG_KEYS + ("g",)


def components(doc: dict, *, split="fit", pool="TRAIN_GCV2"):
    shared = TC.CR.load_shared_sources()
    shared["catalog"] = O.LocCatalogV1(split, doc["objective_sha256"])
    from rl_agent.splitfusion_run5_uplink_lowmcs_v1 import latency as LC
    latency = LC.CandidatePoolLatencyModelV1(LCAMP.REG.BINDING_02, shared["model"], pool=pool,
                                             correction=LCAMP.load_correction())
    return shared, latency


def factories(doc, seed):
    shared, latency = components(doc)
    consts = doc["objective"]["constants"]["fit_reference"]
    collector = lambda: TR.ConstrainedCollectorV1(latency=latency, seed=seed, shared_sources=shared,
                                                  objective_sha256=doc["objective_sha256"], targets=doc["targets"],
                                                  constants=consts)
    return collector, latency


def train(seed: int, out: Path, registration: Path, smoke_at, target: int = REG.UPDATES) -> dict:
    out, registration = Path(out), Path(registration)
    if not (out.is_absolute() and registration.is_absolute()):
        raise ValueError("absolute paths only")
    torch.set_num_threads(TR.CONFIG.torch_intraop_threads)
    run_dir = out / f"CON_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    doc = REG.verify(registration, seed=seed)
    reg_sha = REG.sha(registration)
    collector, latency = factories(doc, seed)
    kw = dict(lambda0=np.asarray(doc["lambda0"]), objective_sha256=doc["objective_sha256"])
    start = time.time()
    o = TR.ConstrainedOrchestratorV1(collector_factory=collector, seed=seed, authorization_sha256=reg_sha, **kw)
    (run_dir / "RUN_BINDING.json").write_text(json.dumps(dict(
        o.binding_document, binding_sha256=o.binding_sha256, latency_binding_sha256=latency.binding_sha256,
        registration=str(registration), registration_sha256=reg_sha, source_digest=REG.source_digest(),
        checkpoints=list(TR.CHECKPOINTS), host=dict(pid=os.getpid(), threads=torch.get_num_threads())), indent=1, sort_keys=True))
    metrics = open(run_dir / "metrics.jsonl", "x", buffering=1)
    decisions = open(run_dir / "decisions.jsonl", "x")
    written, dual_written = 0, 0
    dual = open(run_dir / "DUAL.jsonl", "x", buffering=1)

    def flush():
        nonlocal written, dual_written
        for d in o.collector.diagnostics[written:]:
            decisions.write(json.dumps({k: d[k] for k in DIAG_KEYS}) + "\n")
        written = len(o.collector.diagnostics)
        for x in o.dual_log[dual_written:]:
            dual.write(json.dumps(x) + "\n")
        dual_written = len(o.dual_log)
        decisions.flush()

    def log(m):
        metrics.write(json.dumps(dict(FCAMP._metrics(o.update_count, m, time.time() - start), lam=o.lam.tolist())) + "\n")

    o.warmup_phase()
    flush()
    checkpoints = {0: o.save(run_dir / "checkpoints/u00000.pt")}
    smoke = None
    while o.update_count < target:
        m = o.step_update()
        log(m)
        if o.update_count in TR.CHECKPOINTS:
            checkpoints[o.update_count] = o.save(run_dir / f"checkpoints/u{o.update_count:05d}.pt")
            flush()
        if smoke_at is not None and o.update_count == smoke_at:
            restored = TR.ConstrainedOrchestratorV1.restore(run_dir / f"checkpoints/u{smoke_at:05d}.pt",
                                                            collector_factory=collector, seed=seed,
                                                            authorization_sha256=reg_sha, **kw)
            eq0 = restored.state_digest() == o.state_digest()
            a, b = [], []
            for _ in range(5):
                lm = o.step_update()
                log(lm)
                a.append((lm.critic_loss, lm.actor_loss, lm.reward_mean))
                rm = restored.step_update()
                b.append((rm.critic_loss, rm.actor_loss, rm.reward_mean))
            smoke = dict(update=smoke_at, dual_updates_before_smoke=len(restored.dual_log), restore_digest_equal=eq0,
                         lambda_equal=bool(np.array_equal(restored.lam, o.lam)), next_5_losses_equal=a == b,
                         digest_equal_after_5=restored.state_digest() == o.state_digest())
            smoke["passed"] = all(smoke[k] for k in ("restore_digest_equal", "lambda_equal", "next_5_losses_equal",
                                                     "digest_equal_after_5")) and smoke["dual_updates_before_smoke"] > 0
            (run_dir / "SMOKE.json").write_text(json.dumps(smoke, indent=1, sort_keys=True))
            del restored
            if not smoke["passed"]:
                raise RuntimeError("smoke resume proof failed")
    flush()
    for f in (metrics, decisions, dual):
        f.close()
    actor_path = run_dir / f"actor_u{o.update_count:05d}.pt"
    torch.save(o.actor.state_dict(), actor_path)
    diag = o.collector.diagnostics
    tail = diag[-4000:]
    g_tail = np.array([d["g"] for d in diag[-O.WINDOW:]]).mean(0)
    result = dict(seed=seed, objective_sha256=doc["objective_sha256"], updates=o.update_count, decisions=o.decision_count,
                  seconds=round(time.time() - start, 1), replay=len(o.replay), final_digest=o.state_digest(),
                  checkpoint_sha256={str(k): v["sha256"] for k, v in checkpoints.items()},
                  actor={"path": str(actor_path.relative_to(ROOT)), "sha256": REG.sha(actor_path)}, smoke=smoke,
                  registration_sha256=reg_sha, lambda_final=o.lam.tolist(), dual_updates=len(o.dual_log),
                  final_window_g_over_kappa=dict(zip(O.CONSTRAINTS, g_tail.tolist())),
                  terminals={k: sum(1 for d in diag if d["terminal"] == k) for k in ("SUCCESS", "TIMEOUT", "EVALUATOR_FAULT")},
                  last_4000_mode_share={m: sum(1 for d in tail if d["mode_id"] == m) / len(tail) for m in range(12)},
                  last_4000_timely=sum(1 for d in tail if d["terminal"] != "TIMEOUT") / len(tail),
                  exposure=LCAMP.exposure(diag), coverage=LCAMP.coverage(diag))
    (run_dir / "RUN_RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", required=True, type=int, choices=REG.SEEDS)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--registration", required=True, type=Path)
    p.add_argument("--smoke-at", type=int, default=None)
    a = p.parse_args()
    r = train(a.seed, a.out, a.registration, a.smoke_at)
    print(json.dumps({k: r[k] for k in ("seed", "updates", "seconds", "last_4000_timely", "lambda_final")}))
