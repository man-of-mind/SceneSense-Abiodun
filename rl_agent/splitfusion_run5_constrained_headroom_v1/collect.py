"""Run fixed grid actions, comparator actors and one MCS-stratum rule in the existing corrected latency environment and
store compact per-decision arrays (float32 npz). Nothing is measured; the grid and latency binding are reused as is.

Environments: TRAIN = FIT scenes + TRAIN_GCV2 (training evidence, used to choose the baseline); E1/E2 = held-out scenes
+ HELDOUT_GCV2 / RETENTION_V1 (evaluation). Variant CORRECTED (frozen low-MCS candidate) or OLD.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as orch
from rl_agent.splitfusion_run5_full_v1 import collector as FC
from rl_agent.splitfusion_run5_full_v1 import support as S
from rl_agent.splitfusion_run5_lowmcs_retrain_v1 import campaign as CAMP
from rl_agent.splitfusion_run5_lowmcs_retrain_v1 import evaluate as EV
from rl_agent.splitfusion_run5_reward_trials_v1 import collector as C
from rl_agent.splitfusion_run5_reward_trials_v1 import evaluate as RE
from rl_agent.ue_production_transport_model_v2 import scene_source as SS

ROOT = CAMP.ROOT
ENVS = {"TRAIN": ("fit", "TRAIN_GCV2"), "E1": ("held_scene", "HELDOUT_GCV2"), "E2": ("held_scene", "RETENTION_V1")}
ACTIONS = tuple((m, q) for m in range(12) for q in SS.Q_E4_GRID if S.in_support(m, q))      # 71, all 12 modes
FIELDS = ("eligible", "hits", "tp", "fp", "inter", "union")
COLS = (["timely", "excluded", "mcs", "snr", "backlog", "wire", "q", "mode", "receipt", "value_C0"]
        + [f"{c}_{f}" for c in "vp" for f in FIELDS] + [f"ref_{c}_{f}" for c in "vp" for f in FIELDS])


def stratum(mcs: int) -> str:
    return "mcs_9_10" if mcs <= 10 else "mcs_11_15" if mcs <= 15 else "mcs_ge16"


def run(spec: dict, env: str, variant: str, seed: int, n: int) -> np.ndarray:
    torch.set_num_threads(1)
    FC.compose = EV._REAL_COMPOSE
    split, pool = ENVS[env]
    shared, latency = CAMP.components(split=split, pool=pool, corrected=variant == "CORRECTED")
    cat = shared["catalog"]
    col = C.TrialRun5CollectorV1(latency=latency, seed=seed, shared_sources=shared, arm="C0")
    actor = RE.load_actor(ROOT / spec["path"]) if spec["kind"] == "actor" else None
    out = np.zeros((n, len(COLS)), dtype=np.float64)
    for i in range(n):
        if actor is not None:
            mode, q = RE.act(actor, col.current_state_features())
        elif spec["kind"] == "rule":                       # causal input only: the decision's prior UL MCS
            mode, q = spec["rule"][stratum(int(col.context.prior_ul_mcs))]
        else:
            mode, q = spec["mode"], spec["q"]
        col.collect_transition(orch.ModeledActionRequestV1(decision_ordinal=i, mode_id=mode, q_e4=q,
                                                           source="STOCHASTIC_ACTOR", warmup_q_bin_index=None))
        d = col.diagnostics[-1]
        key = d["reward_scene_key"]
        e = RE.expected(cat.components(key, mode_id=mode, q_e4=q))
        r = RE.expected([(1.0, cat.reference_components(key))])
        v = cat.value_for("C0", key, mode_id=mode, q_e4=q)
        row = [d["receipt_ms"] <= 170.0, d["excluded"], d["prior_ul_mcs"], d["snr_db"], d["backlog_bytes"], d["wire_bytes"],
               q, mode, d["receipt_ms"], np.nan if v is None else v]
        row += [e[f"{c}_{f}"] for c in "vp" for f in FIELDS] + [r[f"{c}_{f}"] for c in "vp" for f in FIELDS]
        out[i] = row
    return out


def job(args):
    key, spec, env, variant, seed, n, out_dir = args
    path = Path(out_dir) / f"{key}__{env}__{variant}__{seed}.npz"
    if not path.exists():
        np.savez_compressed(path, data=run(spec, env, variant, seed, n).astype(np.float32), cols=np.array(COLS))
    return str(path)
