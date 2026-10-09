"""Train one seed (C0 reward, binding_02 TRAIN_GCV2 + frozen low-MCS uplink correction) to 10,000 updates.

  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 python3 -m rl_agent.splitfusion_run5_lowmcs_retrain_v1.campaign \
      --seed 17 --out <ABSOLUTE runs dir> --registration <ABSOLUTE REGISTRATION.json> [--smoke-at 250]

The procedure is ``splitfusion_run5_reward_trials_v1.campaign.train`` for arm C0 (same orchestrator, collector, warm-up,
SAC settings, checkpoints, finite-loss and smoke/resume proof); the only difference is the latency model:
``CandidatePoolLatencyModelV1`` with the frozen CANDIDATE.json. Unsupported inputs follow the candidate's declared
behaviour; their flags are carried in each decision's ``floors`` and counted in RUN_RESULT.json. Create-only outputs.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import time
from pathlib import Path

import torch

from rl_agent.splitfusion_run5_full_v1 import campaign as FCAMP
from rl_agent.splitfusion_run5_reward_trials_v1 import campaign as RCAMP
from rl_agent.splitfusion_run5_reward_trials_v1 import collector as C
from rl_agent.splitfusion_run5_reward_trials_v1 import rewards as R
from rl_agent.splitfusion_run5_reward_trials_v1 import trainer as TR
from rl_agent.splitfusion_run5_uplink_lowmcs_v1 import latency as LC
from rl_agent.splitfusion_run5_uplink_lowmcs_v1 import model as M

from . import registration as REG

ROOT = REG.ROOT
DIAG_KEYS = RCAMP.DIAG_KEYS
FLAGS = (M.UNRESOLVED_BACKLOG, M.EXTRAPOLATED_PAYLOAD, M.EXTRAPOLATED_MCS)
WARMUP_DECISIONS = 288


def load_correction() -> M.LowMcsCorrectionV1:
    expected = json.loads((REG.LOWMCS_RUN / "OUTPUT_SHA256.json").read_text())["fit_01/CANDIDATE.json"]
    if REG.sha(REG.CANDIDATE) != expected:
        raise RuntimeError("frozen candidate differs from its sealed manifest")
    return M.LowMcsCorrectionV1.load(REG.CANDIDATE)


def components(*, split: str = "fit", pool: str = TR.TRAIN_POOL, corrected: bool = True):
    shared = C.load_shared_sources(REG.ARM, split)
    if corrected:
        latency = LC.CandidatePoolLatencyModelV1(REG.BINDING_02, shared["model"], pool=pool, correction=load_correction())
    else:
        latency = RCAMP.L.PoolLatencyModelV1(REG.BINDING_02, shared["model"], pool=pool)
    return shared, latency


def exposure(diag) -> dict:
    def block(rows):
        n = len(rows)
        low = [d for d in rows if d["prior_ul_mcs"] <= 15]
        out = dict(decisions=n, low_mcs=len(low))
        for f in FLAGS:
            k = sum(f in d["floors"] for d in rows)
            out[f] = dict(count=k, rate=k / n if n else None, rate_given_low_mcs=k / len(low) if low else None)
        any_flag = sum(any(f in d["floors"] for f in FLAGS) for d in rows)
        out["any"] = dict(count=any_flag, rate=any_flag / n if n else None,
                          rate_given_low_mcs=any_flag / len(low) if low else None)
        return out
    warm = [d for d in diag if d["ordinal"] < WARMUP_DECISIONS]
    stoch = [d for d in diag if d["ordinal"] >= WARMUP_DECISIONS]
    return dict(all=block(diag), warmup=block(warm), stochastic=block(stoch), last_4000=block(diag[-4000:]))


def coverage(diag) -> dict:
    def block(rows):
        n = len(rows)
        modes = collections.Counter(d["mode_id"] for d in rows)
        qs = collections.Counter((d["mode_id"], d["q_e4"] // 1000 * 1000) for d in rows)
        return dict(decisions=n, sources=dict(collections.Counter(d["source"] for d in rows)),
                    mode_share={m: modes.get(m, 0) / n for m in range(12)},
                    q_bins={f"{m}|{b}": c for (m, b), c in sorted(qs.items())})
    return dict(warmup=block([d for d in diag if d["ordinal"] < WARMUP_DECISIONS]),
                stochastic=block([d for d in diag if d["ordinal"] >= WARMUP_DECISIONS]))


def train(seed: int, out: Path, registration: Path, smoke_at: int | None, target: int = REG.UPDATES) -> dict:
    out, registration = Path(out), Path(registration)
    if not (out.is_absolute() and registration.is_absolute()):
        raise ValueError("use absolute --out and --registration paths")
    torch.set_num_threads(TR.CONFIG.torch_intraop_threads)
    run_dir = out / f"NEW_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    REG.verify(registration, seed=seed)
    reg_sha = REG.sha(registration)
    shared, latency = components()
    factory = lambda: C.TrialRun5CollectorV1(latency=latency, seed=seed, shared_sources=shared, arm=REG.ARM)
    start = time.time()
    o = TR.TrialOrchestratorV1(collector_factory=factory, seed=seed, authorization_sha256=reg_sha)
    (run_dir / "RUN_BINDING.json").write_text(json.dumps(dict(
        o.binding_document, binding_sha256=o.binding_sha256, latency_binding_sha256=latency.binding_sha256,
        latency_pool_chains=len(latency.chains), correction=latency.correction.to_dict(), beta_pool_ms=latency.beta_pool,
        fit_scenes=o.collector.catalog.scene_count, registration=str(registration), registration_sha256=reg_sha,
        source_digest=REG.source_digest(), evidence_label=REG.EVIDENCE_LABEL, checkpoints=list(TR.CHECKPOINTS),
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
                                                      collector_factory=factory, seed=seed, authorization_sha256=reg_sha)
            equal_at_restore = restored.state_digest() == o.state_digest()
            live_next, restored_next = [], []
            for _ in range(5):
                lm = o.step_update()
                metrics.write(json.dumps(FCAMP._metrics(o.update_count, lm, time.time() - start)) + "\n")
                live_next.append((lm.critic_loss, lm.actor_loss, lm.reward_mean))
                rm = restored.step_update()
                restored_next.append((rm.critic_loss, rm.actor_loss, rm.reward_mean))
            smoke = dict(update=smoke_at, bundle=checkpoints[smoke_at], restore_digest_equal=equal_at_restore,
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
    result = dict(arm=REG.ARM, seed=seed, reward_sha256=R.reward_sha256(REG.ARM), updates=o.update_count,
                  decisions=o.decision_count, seconds=round(time.time() - start, 1), replay=len(o.replay),
                  final_digest=o.state_digest(), checkpoints=sorted(checkpoints),
                  checkpoint_sha256={str(k): v["sha256"] for k, v in checkpoints.items()},
                  actor={"path": str(actor_path.relative_to(ROOT)), "sha256": REG.sha(actor_path)},
                  smoke=smoke, sessions=o.collector._session_index + 1, registration_sha256=reg_sha,
                  terminals={k: sum(1 for d in diag if d["terminal"] == k) for k in ("SUCCESS", "TIMEOUT", "EVALUATOR_FAULT")},
                  last_4000_mode_share={m: sum(1 for d in tail if d["mode_id"] == m) / len(tail) for m in range(12)},
                  last_4000_success=sum(1 for d in tail if d["terminal"] == "SUCCESS") / len(tail),
                  last_4000_timely=sum(1 for d in tail if d["terminal"] != "TIMEOUT") / len(tail),
                  floors={f: sum(1 for d in diag if f in d["floors"]) for f in
                          ("D_ZERO", "E_ACCESS", "CONFIRM_SEND_AFTER_BIND", "CONFIRM_SEND_GATED") + FLAGS},
                  exposure=exposure(diag), coverage=coverage(diag))
    (run_dir / "RUN_RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int, choices=REG.SEEDS)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--smoke-at", type=int, default=None)
    args = parser.parse_args()
    r = train(args.seed, args.out, args.registration, args.smoke_at)
    print(json.dumps({k: r[k] for k in ("seed", "updates", "seconds", "last_4000_timely", "exposure")}, default=str))
