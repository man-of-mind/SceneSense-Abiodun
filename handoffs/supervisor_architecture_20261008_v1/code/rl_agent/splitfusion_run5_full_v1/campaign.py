"""Run one full Run-5 training run (seed) to 10,000 updates; create-only outputs.

  python -m rl_agent.splitfusion_run5_full_v1.campaign --seed 17 \
      --out experiments/splitfusion_run5_full_v1/runs_01 --authorization <AUTH.json> [--smoke-at 250]

Smoke (seed 17): every update must be finite (``require_finite``) and every action inside the original support
(enforced by the orchestrator and collector); at ``--smoke-at`` the checkpoint bundle is restored into a fresh
orchestrator by deterministic re-execution, its full state digest must equal the live one, and both must produce
identical losses and digests for the next 5 updates. The smoke run then continues as the registered run.
Any exception stops the run (no retry).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch

from . import authorization as A, collector as C, latency as L, trainer as TR

ROOT = Path(__file__).resolve().parents[2]
BINDING = ROOT / "experiments/splitfusion_run5_full_v1/binding_01"
DIAG_KEYS = ("ordinal", "source", "mode_id", "q_e4", "terminal", "excluded", "reward", "q_perc", "receipt_ms", "profile",
             "prior_ul_mcs", "snr_db", "backlog_bytes", "wire_bytes", "uplink_late", "floors", "base_chain_id")


def components(binding_dir: Path = BINDING):
    shared = C.load_shared_sources()
    latency = L.FullSupportLatencyModelV1(binding_dir, shared["model"])
    return shared, latency


def _metrics(update, m, elapsed):
    return dict(update=update, t=round(elapsed, 2), critic_loss=m.critic_loss, actor_loss=m.actor_loss,
                discrete_entropy=m.discrete_entropy, reward_mean=m.reward_mean, target_mean=m.target_mean,
                q_executed_mean=m.q_executed_mean, critic_grad_norm=m.critic_grad_norm, actor_grad_norm=m.actor_grad_norm)


def _decision(d):
    return {k: d[k] for k in DIAG_KEYS}


def train(seed: int, out: Path, authorization: Path, smoke_at: int | None, target: int = A.UPDATES) -> dict:
    torch.set_num_threads(TR.CONFIG.torch_intraop_threads)
    run_dir = Path(out) / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    A.verify(authorization, seed=seed, binding_dir=BINDING)
    auth_sha = hashlib.sha256(Path(authorization).read_bytes()).hexdigest()
    shared, latency = components()
    factory = lambda: C.FullRun5CollectorV1(latency=latency, seed=seed, shared_sources=shared)
    start = time.time()
    o = TR.FullOrchestratorV1(collector_factory=factory, seed=seed, authorization_sha256=auth_sha)
    (run_dir / "RUN_BINDING.json").write_text(json.dumps(dict(
        o.binding_document, binding_sha256=o.binding_sha256, latency_binding_sha256=latency.binding_sha256,
        fit_scenes=o.collector.catalog.scene_count, authorization=str(authorization), authorization_sha256=auth_sha,
        evidence_label=A.EVIDENCE_LABEL, host=dict(pid=os.getpid(), threads=torch.get_num_threads(),
                                                   torch=torch.__version__)), indent=1, sort_keys=True))
    metrics = open(run_dir / "metrics.jsonl", "x", buffering=1)
    decisions = open(run_dir / "decisions.jsonl", "x")
    written = 0

    def flush():
        nonlocal written
        for d in o.collector.diagnostics[written:]:
            decisions.write(json.dumps(_decision(d)) + "\n")
        written = len(o.collector.diagnostics)
        decisions.flush()

    o.warmup_phase()
    flush()
    checkpoints = {0: o.save(run_dir / "checkpoints/u00000.pt")}
    smoke = None
    while o.update_count < target:
        m = o.step_update()
        metrics.write(json.dumps(_metrics(o.update_count, m, time.time() - start)) + "\n")
        if o.update_count in TR.CHECKPOINTS:
            checkpoints[o.update_count] = o.save(run_dir / f"checkpoints/u{o.update_count:05d}.pt")
            flush()
        if smoke_at is not None and o.update_count == smoke_at:
            restored = TR.FullOrchestratorV1.restore(run_dir / f"checkpoints/u{smoke_at:05d}.pt",
                                                     collector_factory=factory, seed=seed, authorization_sha256=auth_sha)
            equal_at_restore = restored.state_digest() == o.state_digest()
            live_next, restored_next = [], []
            for _ in range(5):
                lm = o.step_update()
                metrics.write(json.dumps(_metrics(o.update_count, lm, time.time() - start)) + "\n")
                live_next.append(lm.critic_loss)
                restored_next.append(restored.step_update().critic_loss)
            smoke = dict(update=smoke_at, finite_losses_every_update=True, all_actions_in_support=True,
                         bundle=checkpoints[smoke_at], restore_digest_equal=equal_at_restore,
                         next_5_critic_losses_equal=live_next == restored_next,
                         digest_equal_after_5=restored.state_digest() == o.state_digest())
            smoke["passed"] = smoke["restore_digest_equal"] and smoke["next_5_critic_losses_equal"] and smoke["digest_equal_after_5"]
            (run_dir / "SMOKE.json").write_text(json.dumps(smoke, indent=1, sort_keys=True))
            del restored
            if not smoke["passed"]:
                raise TR.FullTrainingError("smoke resume proof failed")
    flush()
    metrics.close()
    decisions.close()
    actor_path = run_dir / f"actor_u{o.update_count:05d}.pt"
    torch.save(o.actor.state_dict(), actor_path)
    diag = o.collector.diagnostics
    tail = diag[-4000:]
    result = dict(seed=seed, updates=o.update_count, decisions=o.decision_count, seconds=round(time.time() - start, 1),
                  replay=len(o.replay), final_digest=o.state_digest(), checkpoints=sorted(checkpoints),
                  actor={"path": str(actor_path), "sha256": hashlib.sha256(actor_path.read_bytes()).hexdigest()},
                  smoke=smoke, sessions=o.collector._session_index + 1,
                  terminals={k: sum(1 for d in diag if d["terminal"] == k) for k in ("SUCCESS", "TIMEOUT", "EVALUATOR_FAULT")},
                  last_4000_mode_share={m: sum(1 for d in tail if d["mode_id"] == m) / len(tail) for m in range(12)},
                  last_4000_timely=sum(1 for d in tail if d["terminal"] == "SUCCESS") / len(tail),
                  floors={f: sum(1 for d in diag if f in d["floors"]) for f in ("D_ZERO", "E_ACCESS")})
    (run_dir / "RUN_RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int, choices=A.SEEDS)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--smoke-at", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(train(args.seed, args.out, args.authorization, args.smoke_at)))
