#!/usr/bin/env python3
"""Run-5B-only gates and diagnostics used by the derived runner.

* preflight/smoke SNR gates: feature 21 is the decision's causal sample,
  no generated SNR tick precedes or equals the observed tick, no current SNR
  equals a later generated sample, profiles balanced, reward formula exact,
  and no interior Q_perc value of decision k appears in state k+1;
* descriptive controlled/shuffled SNR diagnostics (no gate);
* disk preflight;
* fresh-process verification of every saved actor (``verify-actors`` CLI).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from rl_agent.splitfusion_hybrid_sac_run4b_v1 import contract as C4

from . import run5b_state_contract as C

ROOT = Path(__file__).resolve().parents[2]
BUNDLE_BYTES_ESTIMATE = 16 * 2**20      # full 65,536-row 21-D replay + models + Adam
LOG_BYTES_PER_DECISION = 1024
MIN_RESERVE_BYTES = 2 * 2**30


def reward_matches(diag: Mapping[str, Any]) -> bool:
    if diag["terminal"] == C4.RewardKind.TIMELY_SUCCESS.value:
        latency_ns = diag["operational_latency_ns"]
        expected = diag["q_perc_training_only"] - C4.LATENCY_WEIGHT * (
            (latency_ns / 1_000_000.0) / C4.DEADLINE_MS)
        return (diag["reward"] == expected and 0 < latency_ns <= C4.DEADLINE_NS
                and latency_ns == diag["composed_total_ns"]
                and latency_ns == sum(diag["components_ns"].values()))
    return (diag["reward"] == C4.FAILURE_REWARD and diag["operational_latency_ns"] is None
            and (not diag["transport_success"] or diag["composed_total_ns"] > C4.DEADLINE_NS))


def qperc_leaks(q_percs: Sequence[float], states: Sequence[Sequence[float]]) -> int:
    """Interior Q_perc of decision k found in state k+1 (0/1 are structural)."""
    return sum(int(0.0 < q < 1.0 and q in states[k + 1])
               for k, q in enumerate(q_percs[:-1]))


def _snr_checks(snr_db, features, successors, generated_ok, q_percs, states, diags):
    later_equal = sum(int(snr_db[i] in set(successors[i:])) for i in range(len(snr_db)))
    return {
        "snr_feature_is_causal_sample": all(
            f == (s - 5.5) / 19.0 for f, s in zip(features, snr_db)),
        "no_future_snr_tick": all(generated_ok),
        "current_snr_never_a_later_generated_sample": later_equal == 0,
        "snr_varies": len(set(features)) >= 2,
        "reward_formula_exact": all(reward_matches(d) for d in diags),
        "no_qperc_in_successor_state": qperc_leaks(q_percs, states) == 0,
    }


def preflight_snr_checks(runner, transitions) -> dict[str, bool]:
    diags = [t.diagnostics for t in transitions]
    checks = _snr_checks([d["snr_db"] for d in diags],
                         [t.state[C.SNR_FEATURE_INDEX] for t in transitions],
                         [d["successor_snr_db"] for d in diags],
                         [d["generated_ticks_after_observed"] for d in diags],
                         [d["q_perc_training_only"] for d in diags],
                         [t.state for t in transitions], diags)
    checks["channel_future_sample_violations_zero"] = runner.env.future_sample_violations == 0
    return checks


def _f32(value: float) -> float:
    return float(torch.tensor(value, dtype=torch.float32))


def smoke_snr_gates(seed_dir: Path, runner) -> dict[str, bool]:
    """Ledger rows are float64; the replay stores float32 states (amendment A1).

    Comparisons against replay states are made at float32; the float64
    feature/SNR identity is checked on the ledger itself.
    """
    rows = [json.loads(line) for line in (Path(seed_dir) / "DECISIONS.jsonl").read_text()
            .splitlines()]
    replay = runner.replay.state_dict()
    states = [tuple(float(v) for v in row) for row in replay["state"]]
    if len(states) != len(rows):
        raise RuntimeError("replay does not hold every smoke decision")
    checks = _snr_checks([r["snr_db"] for r in rows], [r["snr_feature"] for r in rows],
                         [r["successor_snr_db"] for r in rows],
                         [r["generated_ticks_after_observed"] for r in rows],
                         [_f32(r["q_perc_training_only"]) for r in rows], states, rows)
    balance = runner.env.profile_balance()
    checks["four_profiles_balanced"] = (len(balance) == 4 and min(balance.values()) >= 1
                                        and max(balance.values()) - min(balance.values()) <= 1)
    checks["snr_states_match_ledger"] = all(
        s[C.SNR_FEATURE_INDEX] == _f32(r["snr_feature"]) for s, r in zip(states, rows))
    return checks


def snr_diagnostics(runner, seed: int) -> dict[str, Any]:
    """Descriptive only: controlled SNR sweep and shuffled-SNR control."""
    states = runner.replay.state_dict()["state"].clone()
    low, high = states.clone(), states.clone()
    low[:, C.SNR_FEATURE_INDEX], high[:, C.SNR_FEATURE_INDEX] = 0.0, 1.0
    order = list(range(states.shape[0]))
    random.Random(seed).shuffle(order)
    shuffled = states.clone()
    shuffled[:, C.SNR_FEATURE_INDEX] = states[order, C.SNR_FEATURE_INDEX]
    with torch.no_grad():
        def heads(x):
            out = runner.actor(x)
            return torch.softmax(out.logits, -1), runner.actor.deterministic_execution(x)
        p_low, e_low = heads(low)
        p_high, e_high = heads(high)
        p_true, e_true = heads(states)
        p_shuf, e_shuf = heads(shuffled)
    return {
        "n_states": int(states.shape[0]),
        "controlled_sweep_0_to_1_mode_tv_mean": float(0.5 * (p_high - p_low).abs().sum(-1).mean()),
        "controlled_sweep_argmax_mode_change_rate": float(
            (e_high.mode_index != e_low.mode_index).float().mean()),
        "controlled_sweep_q_e4_delta_mean": float(
            (e_high.q_e4 - e_low.q_e4).float().mean()),
        "shuffled_mode_tv_mean": float(0.5 * (p_true - p_shuf).abs().sum(-1).mean()),
        "shuffled_argmax_mode_change_rate": float(
            (e_true.mode_index != e_shuf.mode_index).float().mean()),
        "shuffled_abs_q_e4_delta_mean": float((e_true.q_e4 - e_shuf.q_e4).abs().float().mean()),
        "claim": "descriptive only; no direction or magnitude gate",
    }


def require_disk(out: Path, *, seeds: Sequence[int], target: int) -> dict[str, Any]:
    from .run5b_runner import CHECKPOINT_UPDATES, TRANSITIONS_PER_UPDATE, WARMUP
    bundles = sum(1 for u in CHECKPOINT_UPDATES if u <= target)
    decisions = WARMUP + TRANSITIONS_PER_UPDATE * target
    estimate = len(seeds) * (bundles * BUNDLE_BYTES_ESTIMATE
                             + decisions * LOG_BYTES_PER_DECISION + BUNDLE_BYTES_ESTIMATE)
    probe = Path(out).resolve()
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    required = estimate + max(estimate, MIN_RESERVE_BYTES)
    report = {"estimate_bytes": estimate, "required_bytes": required, "free_bytes": free,
              "passed": free >= required}
    if not report["passed"]:
        raise RuntimeError(f"INSUFFICIENT_DISK {report}")
    return report


# ---------------------------------------------------------------------------
# Fresh-process actor verification
# ---------------------------------------------------------------------------


def verify_actors_here(out: Path) -> list[dict[str, Any]]:
    from rl_agent.splitfusion_hybrid_sac_run4_v1.modeled_smoke_orchestrator import _tree_sha256

    from . import run5b_models as M
    from . import run5b_runner as R

    torch.set_num_threads(R.THREADS)
    sources = R.E.load_sources(R.EVIDENCE_ROOT)
    results = []
    for seed_dir in sorted(Path(out).glob("seed_*")):
        seed = int(seed_dir.name.split("_")[1])
        runner = R.Run5BRunnerV1(sources, seed)
        for bundle in sorted((seed_dir / "checkpoints").glob("update_*")):
            payload = R.read_bundle(bundle, runner)
            actor = M.load_run5b_actor(payload["manifest"], payload["training"]["actor"],
                                       registration_sha256=runner.registration_sha256)
            results.append({"seed": seed, "bundle": bundle.name,
                            "update": payload["manifest"]["update"],
                            "actor_tree_sha256": _tree_sha256(actor.state_dict()),
                            "committed_sha256": (bundle / "COMMITTED").read_text().strip(),
                            "pid": os.getpid(), "loader": "torch.load(weights_only=True)"})
        export = seed_dir / "final_actor"
        if export.is_dir():
            actor = R.load_actor_export(export)
            manifest = json.loads((export / "ACTOR_EXPORT.json").read_text())
            results.append({"seed": seed, "bundle": "final_actor",
                            "update": manifest["update"],
                            "actor_tree_sha256": _tree_sha256(actor.state_dict()),
                            "actor_state_dict_sha256": manifest["actor_state_dict_sha256"],
                            "preregistered_live_actor": manifest["preregistered_live_actor"],
                            "pid": os.getpid(), "loader": "torch.load(weights_only=True)"})
    return results


def fresh_process_verify(out: Path) -> dict[str, Any]:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["CUDA_VISIBLE_DEVICES"] = ""
    done = subprocess.run([sys.executable, "-m", "rl_agent.splitfusion_hybrid_sac_run5b_v1"
                           ".run5b_checks", "verify-actors", "--out", str(out)],
                          cwd=ROOT, env=env, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"fresh-process actor verification failed: {done.stderr[-2000:]}")
    results = json.loads(done.stdout.strip().splitlines()[-1])
    if not results or any(r["pid"] == os.getpid() for r in results):
        raise RuntimeError("fresh-process verification incomplete")
    return {"verified": len(results), "pid": results[0]["pid"], "actors": results}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify-actors")
    verify.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(verify_actors_here(Path(args.out))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
