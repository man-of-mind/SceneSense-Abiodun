"""Run one reward-trial training run (arm, seed) to 10,000 updates; create-only outputs.

  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 python3 -m rl_agent.splitfusion_run5_reward_trials_v1.campaign \
      --arm C1 --seed 17 --out experiments/splitfusion_run5_reward_trials_v1/runs_01 \
      --authorization experiments/splitfusion_run5_reward_trials_v1/authorization_01/AUTHORIZATION.json [--smoke-at 250]

Same procedure as ``splitfusion_run5_full_v1.campaign`` (fresh state, original warm-up, 4 transitions per update,
finite-loss check every update, smoke/resume proof at ``--smoke-at``: restore by deterministic re-execution, equal
state digest, identical next 5 updates). Differences: the arm's reward, the TRAIN_GCV2 pool of binding_02, and
bounded checkpoint retention (updates 0, 250, 2500, 5000, 7500, 10000). Any exception stops the run (no retry).
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

from . import authorization as A, collector as C, latency as L, rewards as R, trainer as TR

ROOT = Path(__file__).resolve().parents[2]
BINDING = ROOT / "experiments/splitfusion_run5_reward_trials_v1/binding_02"
DIAG_KEYS = FCAMP.DIAG_KEYS + ("reward_scene_key",)


def components(arm: str, *, split: str = "fit", pool: str = TR.TRAIN_POOL, binding_dir: Path = BINDING):
    shared = C.load_shared_sources(arm, split)
    latency = L.PoolLatencyModelV1(binding_dir, shared["model"], pool=pool)
    return shared, latency


def train(arm: str, seed: int, out: Path, authorization: Path, smoke_at: int | None, target: int = A.UPDATES) -> dict:
    torch.set_num_threads(TR.CONFIG.torch_intraop_threads)
    run_dir = Path(out) / f"{arm}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    A.verify(authorization, arm=arm, seed=seed, binding_dir=BINDING)
    auth_sha = hashlib.sha256(Path(authorization).read_bytes()).hexdigest()
    shared, latency = components(arm)
    factory = lambda: C.TrialRun5CollectorV1(latency=latency, seed=seed, shared_sources=shared, arm=arm)
    start = time.time()
    o = TR.TrialOrchestratorV1(collector_factory=factory, seed=seed, authorization_sha256=auth_sha)
    (run_dir / "RUN_BINDING.json").write_text(json.dumps(dict(
        o.binding_document, binding_sha256=o.binding_sha256, latency_binding_sha256=latency.binding_sha256,
        latency_pool_chains=len(latency.chains), fit_scenes=o.collector.catalog.scene_count,
        authorization=str(authorization), authorization_sha256=auth_sha, source_digest=A.source_digest(),
        evidence_label=A.EVIDENCE_LABEL, checkpoints=list(TR.CHECKPOINTS),
        host=dict(pid=os.getpid(), threads=torch.get_num_threads(), torch=torch.__version__)), indent=1, sort_keys=True))
    metrics = open(run_dir / "metrics.jsonl", "x", buffering=1)
    decisions = open(run_dir / "decisions.jsonl", "x")
    written = 0

    def flush():
        nonlocal written
        for d in o.collector.diagnostics[written:]:
            decisions.write(json.dumps({k: d[k] for k in DIAG_KEYS}) + "\n")
        written = len(o.collector.diagnostics)
        decisions.flush()

    o.warmup_phase()
    flush()
    checkpoints = {0: o.save(run_dir / "checkpoints/u00000.pt")}
    smoke = None
    while o.update_count < target:
        m = o.step_update()
        metrics.write(json.dumps(FCAMP._metrics(o.update_count, m, time.time() - start)) + "\n")
        if o.update_count in TR.CHECKPOINTS:
            checkpoints[o.update_count] = o.save(run_dir / f"checkpoints/u{o.update_count:05d}.pt")
            flush()
        if smoke_at is not None and o.update_count == smoke_at:
            restored = TR.TrialOrchestratorV1.restore(run_dir / f"checkpoints/u{smoke_at:05d}.pt",
                                                      collector_factory=factory, seed=seed, authorization_sha256=auth_sha)
            equal_at_restore = restored.state_digest() == o.state_digest()
            live_next, restored_next = [], []
            for _ in range(5):
                lm = o.step_update()
                metrics.write(json.dumps(FCAMP._metrics(o.update_count, lm, time.time() - start)) + "\n")
                live_next.append((lm.critic_loss, lm.actor_loss, lm.reward_mean))
                rm = restored.step_update()
                restored_next.append((rm.critic_loss, rm.actor_loss, rm.reward_mean))
            smoke = dict(update=smoke_at, finite_losses_every_update=True, all_actions_in_support=True,
                         bundle=checkpoints[smoke_at], restore_digest_equal=equal_at_restore,
                         next_5_losses_equal=live_next == restored_next,
                         digest_equal_after_5=restored.state_digest() == o.state_digest())
            smoke["passed"] = smoke["restore_digest_equal"] and smoke["next_5_losses_equal"] and smoke["digest_equal_after_5"]
            (run_dir / "SMOKE.json").write_text(json.dumps(smoke, indent=1, sort_keys=True))
            del restored
            if not smoke["passed"]:
                raise TR.TrialTrainingError("smoke resume proof failed")
    flush()
    metrics.close()
    decisions.close()
    actor_path = run_dir / f"actor_u{o.update_count:05d}.pt"
    torch.save(o.actor.state_dict(), actor_path)
    diag = o.collector.diagnostics
    tail = diag[-4000:]
    result = dict(arm=arm, seed=seed, reward_sha256=R.reward_sha256(arm), updates=o.update_count,
                  decisions=o.decision_count, seconds=round(time.time() - start, 1), replay=len(o.replay),
                  final_digest=o.state_digest(), checkpoints=sorted(checkpoints),
                  checkpoint_sha256={str(k): v["sha256"] for k, v in checkpoints.items()},
                  actor={"path": str(actor_path.relative_to(ROOT)), "sha256": hashlib.sha256(actor_path.read_bytes()).hexdigest()},
                  smoke=smoke, sessions=o.collector._session_index + 1,
                  terminals={k: sum(1 for d in diag if d["terminal"] == k) for k in ("SUCCESS", "TIMEOUT", "EVALUATOR_FAULT")},
                  last_4000_mode_share={m: sum(1 for d in tail if d["mode_id"] == m) / len(tail) for m in range(12)},
                  last_4000_success=sum(1 for d in tail if d["terminal"] == "SUCCESS") / len(tail),
                  last_4000_timely=sum(1 for d in tail if d["terminal"] != "TIMEOUT") / len(tail),
                  floors={f: sum(1 for d in diag if f in d["floors"]) for f in ("D_ZERO", "E_ACCESS")})
    (run_dir / "RUN_RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=A.ARMS)
    parser.add_argument("--seed", required=True, type=int, choices=A.SEEDS)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--smoke-at", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(train(args.arm, args.seed, args.out, args.authorization, args.smoke_at)))
