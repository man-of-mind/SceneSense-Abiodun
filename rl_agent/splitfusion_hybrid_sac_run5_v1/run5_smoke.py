#!/usr/bin/env python3
"""The single bounded Run-5 smoke: seed 17, 500 SAC updates, CPU only.

Refuses to start unless the host is confirmed cold (no CARLA, OAI/RFsim,
Run-4 Phase-6 live runner, OAI containers or GPU compute processes).  Writes
event checkpoints and materialized sidecars at updates 0, 100, 250 and 500,
then verifies cold-load and both resume paths, and reports the controlled-SNR
and temporally-shuffled-SNR probes.  It makes no convergence or
policy-performance claim and never starts the 3 x 10,000 campaign.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from rl_agent.splitfusion_hybrid_sac_v1.action_contract import EXPECTED_MODE_COUNT
from rl_agent.ue_production_transport_model_v2 import smoke_runner as R4S

from . import run5_collector as RC
from . import run5_training as RT

SCHEMA = "splitfusion.run5.smoke_500.v1"
SEED = 17
TARGET = 500
ARTIFACT = Path("rl_agent/experiments/ue_production_queue_capture_v1/20260929_model_v2b/"
                "transport_model_v2.json")
HOT_PROCESS_PATTERN = (r"CarlaUE|CarlaUnreal|nr-softmodem|nr-uesoftmodem|lte-softmodem|"
                       r"phase6_live|live_route_b|rfsim")


def host_state() -> dict[str, Any]:
    own = {str(os.getpid()), str(os.getppid())}
    found = subprocess.run(["pgrep", "-af", HOT_PROCESS_PATTERN], capture_output=True,
                           text=True).stdout.splitlines()
    processes = [line for line in found if line.split(" ", 1)[0] not in own
                 and "pgrep" not in line and "run5_smoke" not in line]
    docker = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True,
                            text=True)
    containers = [n for n in docker.stdout.split() if "oai" in n.lower() or "carla" in n.lower()]
    gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name",
                          "--format=csv,noheader"], capture_output=True, text=True)
    gpu_apps = [line for line in gpu.stdout.splitlines() if line.strip()]
    load = [float(v) for v in Path("/proc/loadavg").read_text().split()[:3]]
    cold = (not processes and not containers and not gpu_apps and docker.returncode == 0
            and gpu.returncode == 0 and load[0] < 0.5 * (os.cpu_count() or 1))
    return {"cold": cold, "hot_processes": processes, "hot_containers": containers,
            "gpu_compute_apps": gpu_apps, "loadavg": load, "cpu_count": os.cpu_count(),
            "docker_ok": docker.returncode == 0, "nvidia_smi_ok": gpu.returncode == 0}


def _shim(actor):
    return type("Runner", (), {"model_bundle": type("Bundle", (), {"actor": actor})()})()


def _summaries(orchestrator, states, keys):
    shim = _shim(orchestrator.actor)
    catalog = orchestrator.collector.catalog
    return [R4S._actor_physical_summary(shim, catalog, key, tuple(state))
            for state, key in zip(states, keys)]


def _mode_probs(actor, states) -> torch.Tensor:
    with torch.no_grad():
        return torch.softmax(actor(torch.tensor(states, dtype=torch.float32)).logits, dim=-1)


def controlled_snr_probe(orchestrator) -> dict[str, Any]:
    history = orchestrator.history
    diagnostics = orchestrator.collector.diagnostics()
    keys = [(d["reward_scene_key"], d["held_scene_key"]) for d in diagnostics]
    low = [tuple(r.state[:21]) + (0.0,) for r in history]
    high = [tuple(r.state[:21]) + (1.0,) for r in history]
    s_low, s_high = _summaries(orchestrator, low, keys), _summaries(orchestrator, high, keys)
    dq = [h["expected_q_exec"] - l["expected_q_exec"] for l, h in zip(s_low, s_high)]
    dw = [h["expected_wire_bytes"] - l["expected_wire_bytes"] for l, h in zip(s_low, s_high)]
    tv = 0.5 * (_mode_probs(orchestrator.actor, high)
                - _mode_probs(orchestrator.actor, low)).abs().sum(-1)
    q_mean, q_ci = R4S._bootstrap_mean_ci(dq, draws=2000, seed=SEED)
    w_mean, w_ci = R4S._bootstrap_mean_ci(dw, draws=2000, seed=SEED + 1)
    return {"probe": "CONTROLLED_SNR_ONLY_SWEEP_5.5_TO_24.5_DB_ON_OBSERVED_STATES",
            "n_states": len(history),
            "expected_q_exec_delta_mean": q_mean, "expected_q_exec_delta_ci95": q_ci,
            "expected_two_tensor_ingress_bytes_delta_mean": w_mean,
            "expected_two_tensor_ingress_bytes_delta_ci95": w_ci,
            "mode_distribution_tv_mean": float(tv.mean()),
            "claim": "descriptive only at 500 updates; no direction is required"}


def shuffled_snr_control(orchestrator) -> dict[str, Any]:
    history = orchestrator.history
    diagnostics = orchestrator.collector.diagnostics()
    keys = [(d["reward_scene_key"], d["held_scene_key"]) for d in diagnostics]
    rng = random.Random(SEED)
    snr = [r.state[21] for r in history]
    rng.shuffle(snr)
    real = [tuple(r.state) for r in history]
    shuffled = [tuple(r.state[:21]) + (s,) for r, s in zip(history, snr)]
    s_real, s_shuf = _summaries(orchestrator, real, keys), _summaries(orchestrator, shuffled, keys)
    dq = [abs(a["expected_q_exec"] - b["expected_q_exec"]) for a, b in zip(s_real, s_shuf)]
    tv = 0.5 * (_mode_probs(orchestrator.actor, real)
                - _mode_probs(orchestrator.actor, shuffled)).abs().sum(-1)
    one_hot = torch.nn.functional.one_hot(torch.tensor([r.mode_id for r in history]),
                                          EXPECTED_MODE_COUNT).to(torch.float32)
    q_norm = torch.tensor([r.q_e4 / 9800.0 for r in history], dtype=torch.float32)
    with torch.no_grad():
        q_real = torch.min(*orchestrator.critics.q_values(
            torch.tensor(real, dtype=torch.float32), one_hot, q_norm))
        q_shuf = torch.min(*orchestrator.critics.q_values(
            torch.tensor(shuffled, dtype=torch.float32), one_hot, q_norm))
    rewards = torch.tensor([r.reward for r in history], dtype=torch.float32)
    return {"control": "TEMPORALLY_SHUFFLED_SNR_FEATURE_ON_OBSERVED_STATES", "seed": SEED,
            "n_states": len(history),
            "actor_expected_q_exec_abs_delta_mean": statistics.fmean(dq),
            "actor_mode_distribution_tv_mean": float(tv.mean()),
            "critic_executed_action_abs_q_delta_mean": float((q_real - q_shuf).abs().mean()),
            "critic_q_minus_immediate_reward_mse_real": float(((q_real - rewards) ** 2).mean()),
            "critic_q_minus_immediate_reward_mse_shuffled": float(((q_shuf - rewards) ** 2).mean()),
            "claim": "descriptive only; no gate on direction or magnitude"}


def trajectory_gates(orchestrator) -> dict[str, Any]:
    history = orchestrator.history
    diagnostics = orchestrator.collector.diagnostics()
    warm = len(orchestrator.schedule)
    post = history[warm:]
    modes_all = sorted({r.mode_id for r in history})
    post_modes = {str(m): sum(1 for r in post if r.mode_id == m) for m in range(12)}
    post_q = [r.q_e4 for r in post]
    per_mode_q = {m: len({r.q_e4 for r in post if r.mode_id == m}) for m in range(12)}
    columns = {"snr": 21, "mcs": 2, "backlog": 3, "camera_si": 0, "radar_p40": 1,
               "prev_quality": 17, "prev_latency": 18, "prev_success": 20, "prev_q": 16}
    variation = {name: {"distinct": len({r.state[i] for r in history}),
                        "span": max(r.state[i] for r in history) - min(r.state[i] for r in history)}
                 for name, i in columns.items()}
    prev_mode_distinct = len({max(range(12), key=lambda k: r.state[4 + k])
                              for r in history[1:]})
    params_finite = all(bool(torch.isfinite(p).all()) for p in
                        (*orchestrator.actor.parameters(), *orchestrator.critics.parameters()))
    metric_finite = all(all(math.isfinite(float(getattr(m, f))) for f in
                            ("critic_loss", "actor_loss", "critic_grad_norm", "actor_grad_norm",
                             "target_mean", "discrete_entropy")) for m in orchestrator.metrics)
    # Current SNR vs every sample generated after it (by construction disjoint ticks).
    later_equal = 0
    successors = [d["successor_snr_db"] for d in diagnostics]
    for index, d in enumerate(diagnostics):
        # successors[index:] are samples generated after decision index's action.
        later_equal += int(d["snr_db"] in set(successors[index:]))
    gates = {
        "losses_gradients_parameters_finite": params_finite and metric_finite,
        "all_12_modes_explored": modes_all == list(range(12)),
        "continuous_q_non_degenerate": (len(set(post_q)) > 12
                                        and statistics.pstdev(post_q) > 0
                                        and sum(1 for v in per_mode_q.values() if v > 1) >= 2),
        "state_features_vary": all(v["distinct"] >= 2 and v["span"] > 0
                                   for v in variation.values()) and prev_mode_distinct >= 2,
        "no_future_snr_tick_in_current_state": (
            orchestrator.collector._channel.future_sample_violations == 0
            and all(d["generated_ticks_after_observed"] for d in diagnostics)
            and later_equal == 0),
    }
    return {"gates": gates, "variation": variation, "post_warmup_mode_counts": post_modes,
            "post_warmup_distinct_q": len(set(post_q)), "per_mode_distinct_q": per_mode_q,
            "previous_mode_distinct": prev_mode_distinct,
            "current_snr_equal_to_a_later_generated_sample": later_equal,
            "post_warmup_success_rate": sum(r.terminal == "SUCCESS" for r in post) / len(post),
            "post_warmup_mean_reward": statistics.fmean(r.reward for r in post),
            "decisions": len(history)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    host = host_state()
    print(json.dumps({"host": host}, indent=1))
    if not host["cold"]:
        print("HOST_NOT_COLD: smoke refused", file=sys.stderr)
        return 3
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    started = time.time()
    shared = RC.build_shared_sources(ARTIFACT, args.evidence_root)

    def factory():
        return RC.Run5ModeledCollectorV1(artifact_path=ARTIFACT, seed=SEED,
                                         evidence_root=args.evidence_root, shared_sources=shared)

    orchestrator = RT.Run5OrchestratorV1(collector_factory=factory, seed=SEED)
    checkpoints: dict[int, RT.Run5CheckpointV1] = {}
    milestones = []

    def on_checkpoint(checkpoint):
        update = checkpoint.update_count
        event_path = output / f"update_{update:06d}.checkpoint.json"
        file_sha = RT.write_event_checkpoint(event_path, checkpoint)
        require_equal(RT.read_event_checkpoint(event_path).canonical_sha256,
                      checkpoint.canonical_sha256, "event checkpoint round-trip")
        manifest_sha = RT.write_sidecar(output / f"update_{update:06d}.sidecar",
                                        orchestrator, checkpoint)
        checkpoints[update] = checkpoint
        metrics = orchestrator.metrics[-1] if orchestrator.metrics else None
        milestones.append({"update": update, "decision_count": checkpoint.decision_count,
                           "checkpoint_sha256": checkpoint.canonical_sha256,
                           "checkpoint_file_sha256": file_sha,
                           "sidecar_manifest_sha256": manifest_sha,
                           "metrics": None if metrics is None else {
                               f: getattr(metrics, f) for f in metrics.__dataclass_fields__}})

    orchestrator.run_to(TARGET, checkpoint_callback=on_checkpoint, emit_current=True)
    train_seconds = time.time() - started
    final_boundary = dict(checkpoints[TARGET].boundary)

    resume = {}
    for update in RT.SMOKE_CHECKPOINTS:
        actor = RT.cold_load_actor(output / f"update_{update:06d}.sidecar", checkpoints[update])
        resume[f"cold_load_actor_{update}"] = True
    for path_name, restorer in (
            ("event_replay", lambda cp: RT.Run5OrchestratorV1.restore_by_replay(
                cp, collector_factory=factory)),
            ("sidecar", lambda cp: RT.Run5OrchestratorV1.restore_from_sidecar(
                cp, output / f"update_{cp.update_count:06d}.sidecar",
                collector_factory=factory))):
        resumed = restorer(RT.read_event_checkpoint(output / "update_000250.checkpoint.json"))
        seen = []
        resumed.run_to(TARGET, checkpoint_callback=lambda cp: seen.append(cp))
        resume[f"{path_name}_250_to_500_bit_identical"] = (
            dict(seen[-1].boundary) == final_boundary
            and seen[-1].canonical_sha256 == checkpoints[TARGET].canonical_sha256)
    run4_refused = False
    try:
        RT.read_event_checkpoint(next(Path(args.evidence_root).glob(
            "rl_agent/experiments/splitfusion_hybrid_sac_run4_v2_smoke/*/checkpoints/*.json")))
    except RT.Run5TrainingError:
        run4_refused = True
    resume["run4_checkpoint_refused"] = run4_refused

    trajectory = trajectory_gates(orchestrator)
    gates = {**trajectory["gates"],
             "exact_cold_load_and_resume": all(resume.values()),
             "preflight_passed": bool(orchestrator.preflight["passed"])}
    document = {
        "schema": SCHEMA, "seed": SEED, "target_update": TARGET, "host": host,
        "evidence_root": str(Path(args.evidence_root).resolve()),
        "binding": orchestrator.binding_document, "binding_sha256": orchestrator.binding_sha256,
        "preflight": orchestrator.preflight, "milestones": milestones,
        "trajectory": trajectory, "resume": resume,
        "controlled_snr_response_at_500": controlled_snr_probe(orchestrator),
        "shuffled_snr_control_at_500": shuffled_snr_control(orchestrator),
        "gates": gates, "all_gates_passed": all(gates.values()),
        "train_seconds": train_seconds, "total_seconds": time.time() - started,
        "claims_not_made": ["convergence", "policy performance", "deployment readiness"],
        "campaign_started": False,
    }
    with (output / "RUN5_SMOKE_500.json").open("x") as handle:
        json.dump(document, handle, indent=1, sort_keys=True, default=str)
        handle.write("\n")
    print(json.dumps({"gates": gates, "resume": resume,
                      "controlled": document["controlled_snr_response_at_500"],
                      "shuffled": document["shuffled_snr_control_at_500"]}, indent=1))
    return 0 if document["all_gates_passed"] else 2


def require_equal(a, b, what):
    if a != b:
        raise RuntimeError(f"{what} differs")


if __name__ == "__main__":
    sys.exit(main())
