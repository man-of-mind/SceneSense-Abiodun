"""Controller B (deployed actor restricted to mode 11, using the actor's OWN mode-11 conditional mean q on the mode-11
support) plus an instrumented re-run of controller A (deployed, unchanged) in the identical headroom collector.

Adds per-decision: channel profile, the actor's 12 mode probabilities and all 12 mode-conditional deterministic q_e4.
Everything else (environment, streams, accounting, columns) is collect.run of splitfusion_run5_constrained_headroom_v1.
Nothing is trained; CPU only.

    env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python3 -m rl_agent.splitfusion_run5_mode_value_audit_v1.evalb --out <ABS dir>
"""
from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as orch
from rl_agent.splitfusion_hybrid_sac_v1.hybrid_sac_models import quantize_q_e4
from rl_agent.splitfusion_run5_constrained_headroom_v1 import collect as K
from rl_agent.splitfusion_run5_full_v1 import collector as FC
from rl_agent.splitfusion_run5_full_v1 import support as S
from rl_agent.splitfusion_run5_lowmcs_retrain_v1 import campaign as CAMP
from rl_agent.splitfusion_run5_lowmcs_retrain_v1 import evaluate as EV
from rl_agent.splitfusion_run5_lowmcs_retrain_v1 import registration as REG
from rl_agent.splitfusion_run5_reward_trials_v1 import collector as C
from rl_agent.splitfusion_run5_reward_trials_v1 import evaluate as RE

PROFILES = ("FAVORABLE_STABLE", "MID_VARIABLE", "ADVERSE_STABLE", "FADE_RECOVERY")
EXTRA = ["profile"] + [f"p{m}" for m in range(12)] + [f"qm{m}" for m in range(12)]
COLS = list(K.COLS) + EXTRA
EVAL_SEEDS = (9017, 9029, 9043)
TRAIN_SEEDS = (8017, 8029)
N = 6000


def heads(actor, features):
    """(argmax mode, its q_e4, 12 probs, 12 mode-conditional deterministic q_e4) — the same mapping as
    ConditionalHybridActor.deterministic_execution, evaluated for every mode."""
    state = torch.tensor((tuple(float(v) for v in features),), dtype=torch.float32)
    with torch.inference_mode():
        h = actor(state)
        probs = torch.softmax(h.logits, -1)[0]
        idx = torch.arange(12)
        q_all = quantize_q_e4(actor._map_pre_squash_to_q(h.mean[0], mode_index=idx))
        ex = actor.deterministic_execution(state)
    m = int(ex.mode_index[0])
    assert int(q_all[m]) == int(ex.q_e4[0]), "per-mode q mapping disagrees with deterministic_execution"
    return m, int(ex.q_e4[0]), probs.numpy().astype(np.float64), q_all.numpy().astype(np.float64)


def run(kind: str, env: str, variant: str, seed: int, n: int) -> np.ndarray:
    torch.set_num_threads(1)
    FC.compose = EV._REAL_COMPOSE
    split, pool = K.ENVS[env]
    shared, latency = CAMP.components(split=split, pool=pool, corrected=variant == "CORRECTED")
    cat = shared["catalog"]
    col = C.TrialRun5CollectorV1(latency=latency, seed=seed, shared_sources=shared, arm="C0")
    path = REG.ROOT / REG.DEPLOYED["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == REG.DEPLOYED["sha256"], "deployed weights hash mismatch"
    actor = RE.load_actor(path)
    out = np.zeros((n, len(COLS)), dtype=np.float64)
    for i in range(n):
        m_arg, q_arg, probs, q_all = heads(actor, col.current_state_features())
        if kind == "A":
            mode, q = m_arg, q_arg
        else:                                          # B: mode 11 at the actor's own mode-11 conditional mean
            mode, q = 11, int(q_all[11])
        assert S.in_support(mode, q), (mode, q)
        col.collect_transition(orch.ModeledActionRequestV1(decision_ordinal=i, mode_id=mode, q_e4=q,
                                                           source="STOCHASTIC_ACTOR", warmup_q_bin_index=None))
        d = col.diagnostics[-1]
        key = d["reward_scene_key"]
        e = RE.expected(cat.components(key, mode_id=mode, q_e4=q))
        r = RE.expected([(1.0, cat.reference_components(key))])
        v = cat.value_for("C0", key, mode_id=mode, q_e4=q)
        row = [d["receipt_ms"] <= 170.0, d["excluded"], d["prior_ul_mcs"], d["snr_db"], d["backlog_bytes"], d["wire_bytes"],
               q, mode, d["receipt_ms"], np.nan if v is None else v]
        row += [e[f"{c}_{f}"] for c in "vp" for f in K.FIELDS] + [r[f"{c}_{f}"] for c in "vp" for f in K.FIELDS]
        row += [PROFILES.index(d["profile"])] + list(probs) + list(q_all)
        out[i] = row
    return out


def job(args):
    kind, env, variant, seed, n, out_dir = args
    path = Path(out_dir) / f"{kind}__{env}__{variant}__{seed}.npz"
    if not path.exists():
        np.savez_compressed(path, data=run(kind, env, variant, seed, n), cols=np.array(COLS))
    return str(path)


def jobs(out):
    js = [(k, env, var, s, N, out) for k in ("A", "B") for env in ("E1", "E2") for var in ("CORRECTED", "OLD")
          for s in EVAL_SEEDS]
    js += [(k, "TRAIN", "CORRECTED", s, N, out) for k in ("A", "B") for s in TRAIN_SEEDS]
    return js


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.smoke:
        print(run("A", "E1", "CORRECTED", 9017, 20)[:, [6, 7, 8]])
        print(run("B", "E1", "CORRECTED", 9017, 20)[:, [6, 7, 8]])
        raise SystemExit
    js = jobs(str(a.out))
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for i, path in enumerate(pool.imap_unordered(job, js)):
            print(i, len(js), path, flush=True)
    print("DONE", len(js))
