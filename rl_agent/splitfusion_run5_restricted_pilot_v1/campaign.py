"""Run one registered pilot training run (arm x seed) to 10,000 updates, create-only outputs.

  python -m rl_agent.splitfusion_run5_restricted_pilot_v1.campaign train --arm CONTROL --seed 17 \
      --out experiments/splitfusion_run5_restricted_pilot_v1/runs_01 --authorization <AUTH.json> [--smoke-at 250]

Smoke (seed 17, both arms): every update must be finite (``require_finite``) and every action inside the support
(enforced in the orchestrator/collector); at ``--smoke-at`` the registered checkpoint bundle is restored into a fresh
orchestrator by deterministic re-execution, its full state digest must equal the live one, and both must produce
identical metrics and digests for the next 5 updates. The smoke run then continues as the registered run.
Any exception stops the run (no retry, no restart).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

from rl_agent.splitfusion_corrected_latency_env_v1 import collector as CR

from . import authorization as A, collector as C, latency as L, quality as Q, trainer as TR

ROOT = Path(__file__).resolve().parents[2]
BINDING = ROOT / "experiments/splitfusion_run5_restricted_pilot_v1/binding_01"


def components(split: str, binding_dir: Path = BINDING):
    lat = L.PilotLatencyModelV1(binding_dir, split)
    cat = Q.PilotSceneCatalogV1("fit" if split == "TRAIN" else "held_scene", lat.scene_supported)
    trace = [t for t in json.loads((binding_dir / "SPLIT.json").read_text())["trace"] if t["split"] == split]
    fb = C.FallbackModelV1(json.loads((binding_dir / "FALLBACK.json").read_text()))
    return lat, cat, trace, fb


def _metrics(update, m, elapsed):
    return dict(update=update, t=round(elapsed, 2), critic_loss=m.critic_loss, actor_loss=m.actor_loss,
                discrete_entropy=m.discrete_entropy, reward_mean=m.reward_mean, target_mean=m.target_mean,
                q_executed_mean=m.q_executed_mean, critic_grad_norm=m.critic_grad_norm, actor_grad_norm=m.actor_grad_norm)


def _decision(d):
    return {k: d[k] for k in ("ordinal", "source", "mode_id", "q_e4", "terminal", "excluded", "control_reward",
                              "candidate_reward", "q_perc", "receipt_ms", "mcs_gap", "interp_weight",
                              "fallback_frames_after")}


def train(arm: str, seed: int, out: Path, authorization: Path, smoke_at: int | None, target: int = A.UPDATES) -> dict:
    torch.set_num_threads(TR.CONFIG.torch_intraop_threads)
    run_dir = Path(out) / f"{arm}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    auth_doc = A.verify(authorization, arm=arm, seed=seed, binding_dir=BINDING)
    auth_sha = hashlib.sha256(Path(authorization).read_bytes()).hexdigest()
    shared = CR.load_shared_sources()
    lat, cat, trace, fb = components("TRAIN")
    factory = lambda: C.PilotRun5CollectorV1(latency=lat, catalog=cat, trace=trace, fallback=fb, arm=arm, seed=seed,
                                             authorization=authorization, binding_dir=BINDING, shared_sources=shared)
    start = time.time()
    o = TR.PilotOrchestratorV1(collector_factory=factory, seed=seed, arm=arm, authorization_sha256=auth_sha)
    (run_dir / "RUN_BINDING.json").write_text(json.dumps(dict(o.binding_document, binding_sha256=o.binding_sha256,
        latency_binding_sha256=lat.binding_sha256, split_sha256=lat.split_sha256, supported_fit_scenes=cat.scene_count,
        authorization=str(authorization), authorization_sha256=auth_sha, evidence_label=A.EVIDENCE_LABEL,
        host=dict(pid=os.getpid(), threads=torch.get_num_threads(), torch=torch.__version__)), indent=1, sort_keys=True))
    metrics = open(run_dir / "metrics.jsonl", "x", buffering=1)
    decisions = open(run_dir / "decisions.jsonl", "x")
    written = 0

    def flush_decisions():
        nonlocal written
        for d in o.collector.diagnostics[written:]:
            decisions.write(json.dumps(_decision(d)) + "\n")
        written = len(o.collector.diagnostics)
        decisions.flush()

    o.warmup_phase()
    flush_decisions()
    checkpoints = {0: o.save(run_dir / "checkpoints/u00000.pt")}
    smoke = None
    while o.update_count < target:
        m = o.step_update()
        metrics.write(json.dumps(_metrics(o.update_count, m, time.time() - start)) + "\n")
        if o.update_count in TR.CHECKPOINTS:
            checkpoints[o.update_count] = o.save(run_dir / f"checkpoints/u{o.update_count:05d}.pt")
            flush_decisions()
        if smoke_at is not None and o.update_count == smoke_at:
            bundle = run_dir / f"checkpoints/u{smoke_at:05d}.pt"
            restored = TR.PilotOrchestratorV1.restore(bundle, collector_factory=factory, seed=seed, arm=arm,
                                                      authorization_sha256=auth_sha)
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
                         digest_equal_after_5=restored.state_digest() == o.state_digest(),
                         passed=equal_at_restore and live_next == restored_next and restored.state_digest() == o.state_digest())
            (run_dir / "SMOKE.json").write_text(json.dumps(smoke, indent=1, sort_keys=True))
            del restored
            if not smoke["passed"]:
                raise TR.PilotTrainingError("smoke resume proof failed")
    flush_decisions()
    metrics.close()
    decisions.close()
    actor_path = run_dir / f"actor_u{o.update_count:05d}.pt"
    torch.save(o.actor.state_dict(), actor_path)
    diag = o.collector.diagnostics
    tail = diag[-4000:]
    result = dict(arm=arm, seed=seed, updates=o.update_count, decisions=o.decision_count, seconds=round(time.time() - start, 1),
                  replay=len(o.replay), final_digest=o.state_digest(), checkpoints=sorted(checkpoints),
                  actor={"path": str(actor_path), "sha256": hashlib.sha256(actor_path.read_bytes()).hexdigest()},
                  smoke=smoke, fallback_frames_total=o.collector.fallback_frames_total,
                  terminals={k: sum(1 for d in diag if d["terminal"] == k) for k in ("SUCCESS", "TIMEOUT", "EVALUATOR_FAULT")},
                  last_4000_mode_share={m: sum(1 for d in tail if d["mode_id"] == m) / len(tail) for m in (5, 8, 11)},
                  mcs_gap_mean=sum(d["mcs_gap"] for d in diag) / len(diag))
    (run_dir / "RUN_RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    return result


def resume(arm: str, seed: int, out: Path, authorization: Path, target: int = A.UPDATES) -> dict:
    """Continue a stopped run from its latest registered checkpoint (deterministic restore, as proven by the smoke).

    The run keeps its registered identity (binding incl. the original authorization digest); collectors verify the
    superseding authorization. Before continuing, the restored collector's regenerated decisions must equal the
    decisions logged by the stopped run, line for line. New outputs are create-only, suffixed ``_resume_01``.
    """
    torch.set_num_threads(TR.CONFIG.torch_intraop_threads)
    run_dir = Path(out) / f"{arm}_seed{seed}"
    _ = A.verify(authorization, arm=arm, seed=seed, binding_dir=BINDING)
    registered = json.loads((run_dir / "RUN_BINDING.json").read_text())
    original_auth_sha = registered["authorization_sha256"]
    shared = CR.load_shared_sources()
    lat, cat, trace, fb = components("TRAIN")
    factory = lambda: C.PilotRun5CollectorV1(latency=lat, catalog=cat, trace=trace, fallback=fb, arm=arm, seed=seed,
                                             authorization=authorization, binding_dir=BINDING, shared_sources=shared)
    bundles = sorted((run_dir / "checkpoints").glob("u*.pt"))
    bundle = bundles[-1]
    start = time.time()
    o = TR.PilotOrchestratorV1.restore(bundle, collector_factory=factory, seed=seed, arm=arm,
                                       authorization_sha256=original_auth_sha)
    _require = TR._require
    _require(o.binding_sha256 == registered["binding_sha256"], "resumed binding differs from the registered run")
    logged = (run_dir / "decisions.jsonl").read_text().splitlines()
    regenerated = [json.dumps(_decision(d)) for d in o.collector.diagnostics[:len(logged)]]
    _require(len(logged) <= o.decision_count and regenerated == logged, "regenerated decisions differ from the logged run")
    metrics = open(run_dir / "metrics_resume_01.jsonl", "x", buffering=1)
    checkpoints = {}
    while o.update_count < target:
        m = o.step_update()
        metrics.write(json.dumps(_metrics(o.update_count, m, time.time() - start)) + "\n")
        if o.update_count in TR.CHECKPOINTS:
            checkpoints[o.update_count] = o.save(run_dir / f"checkpoints/u{o.update_count:05d}.pt")
    metrics.close()
    with open(run_dir / "decisions_full.jsonl", "x") as stream:
        for d in o.collector.diagnostics:
            stream.write(json.dumps(_decision(d)) + "\n")
    actor_path = run_dir / f"actor_u{o.update_count:05d}.pt"
    torch.save(o.actor.state_dict(), actor_path)
    diag = o.collector.diagnostics
    tail = diag[-4000:]
    result = dict(arm=arm, seed=seed, updates=o.update_count, decisions=o.decision_count, resumed_from=str(bundle),
                  resume_seconds=round(time.time() - start, 1), logged_prefix_verified=len(logged),
                  superseding_authorization=str(authorization),
                  superseding_authorization_sha256=hashlib.sha256(Path(authorization).read_bytes()).hexdigest(),
                  stop_cause="inherited session-UUID formatter invalid at session index 1000 (identity error); fixed in "
                             "the pilot collector; indices < 1000 unchanged",
                  replay=len(o.replay), final_digest=o.state_digest(), checkpoints_after_resume=sorted(checkpoints),
                  actor={"path": str(actor_path), "sha256": hashlib.sha256(actor_path.read_bytes()).hexdigest()},
                  smoke=json.loads((run_dir / "SMOKE.json").read_text()) if (run_dir / "SMOKE.json").exists() else None,
                  fallback_frames_total=o.collector.fallback_frames_total,
                  terminals={k: sum(1 for d in diag if d["terminal"] == k) for k in ("SUCCESS", "TIMEOUT", "EVALUATOR_FAULT")},
                  sessions=o.collector._session_index + 1,
                  last_4000_mode_share={m: sum(1 for d in tail if d["mode_id"] == m) / len(tail) for m in (5, 8, 11)},
                  mcs_gap_mean=sum(d["mcs_gap"] for d in diag) / len(diag))
    (run_dir / "RUN_RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "resume"))
    parser.add_argument("--arm", required=True, choices=A.ARMS)
    parser.add_argument("--seed", required=True, type=int, choices=A.SEEDS)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--smoke-at", type=int, default=None)
    args = parser.parse_args()
    if args.command == "train":
        print(json.dumps(train(args.arm, args.seed, args.out, args.authorization, args.smoke_at)))
    else:
        print(json.dumps(resume(args.arm, args.seed, args.out, args.authorization)))
