"""Batch 1: fixed actions + comparators on TRAIN (choice evidence) and fixed actions on one held-out seed (oracle only).

    python3 -m rl_agent.splitfusion_run5_constrained_headroom_v1.batch --out <ABS dir> --workers 20
"""
import argparse
import multiprocessing as mp
from pathlib import Path

from rl_agent.splitfusion_run5_lowmcs_retrain_v1 import registration as REG

from . import collect as K

TRAIN_SEEDS = (8017, 8029)          # distinct from training seeds (17/29/43) and evaluation seeds (9017/9029/9043)
ORACLE_SEED = 9017
N = 6000


def comparators():
    out = {f"C0_seed{s}": dict(kind="actor", path=v["path"]) for s, v in ((int(k[-2:]), v) for k, v in REG.COMPARATORS.items())}
    out["HIST_seed43"] = dict(kind="actor", path=REG.DEPLOYED["path"])
    for s in REG.SEEDS:
        out[f"NEW_seed{s}"] = dict(kind="actor", path=f"experiments/splitfusion_run5_lowmcs_retrain_v1/runs_01/NEW_seed{s}/actor_u10000.pt")
    return out


def jobs(out):
    js = []
    for m, q in K.ACTIONS:
        spec = dict(kind="fixed", mode=m, q=q)
        js += [(f"A_m{m}_q{q}", spec, "TRAIN", "CORRECTED", s, N, out) for s in TRAIN_SEEDS]
        js.append((f"A_m{m}_q{q}", spec, "E1", "CORRECTED", ORACLE_SEED, N, out))
    for key, spec in comparators().items():
        js += [(key, spec, "TRAIN", "CORRECTED", s, N, out) for s in TRAIN_SEEDS]
    return js


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--workers", type=int, default=20)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    js = jobs(str(a.out))
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for i, path in enumerate(pool.imap_unordered(K.job, js)):
            if i % 25 == 0:
                print(i, len(js), path, flush=True)
    print("DONE", len(js))
