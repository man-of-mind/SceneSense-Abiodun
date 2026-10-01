#!/usr/bin/env python3
"""Run-5B campaign runner: bounded smoke, then deep training only when authorized.

    python -m rl_agent.splitfusion_hybrid_sac_run5b_v1.run5b_campaign \
        --mode smoke --campaign-dir DIR --seed 17 --evidence-root ../abiodun [--resume]

* ``--mode deep`` refuses unless ``DEEP_TRAINING_AUTHORIZATION.json`` exists
  and names the sealed preregistration.  No such file is created here.
* Every registered boundary is one atomic bundle (``run5b_bundle``).
* ``--resume`` selects the latest verified bundle, restores it without any
  gradient replay, continues in the same directory and verifies the
  append-only metric/decision ledgers against the bundle.
* SIGINT/SIGTERM set a stop flag; the current update finishes, an emergency
  bundle is committed and the process exits with status 75.
* Completion verifies every boundary, exports the chosen actor atomically,
  re-loads it in a fresh process with ``weights_only=True`` and only then writes
  ``SEED_COMPLETE.json`` (deep) / ``SMOKE_SEED_COMPLETE.json`` (smoke).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from rl_agent.splitfusion_hybrid_sac_v1.action_contract import EXPECTED_MODE_COUNT, Q_E4_MAX
from rl_agent.splitfusion_hybrid_sac_v1.hybrid_sac_models import build_actor
from rl_agent.ue_production_transport_model_v2 import smoke_runner as R4S

from . import run5b_bundle as B
from . import run5b_collector as RC
from . import run5b_models as RM
from . import run5b_preregistration as PR
from . import run5b_state_contract as C
from . import run5b_training as RT

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
ARTIFACT = ROOT / ("rl_agent/experiments/ue_production_queue_capture_v1/20260929_model_v2b/"
                   "transport_model_v2.json")
AUTHORIZATION = PACKAGE / "DEEP_TRAINING_AUTHORIZATION.json"
EXIT_STOPPED = 75
EXIT_REFUSED = 4
HOT_PROCESS_PATTERN = (r"CarlaUE|CarlaUnreal|nr-softmodem|nr-uesoftmodem|lte-softmodem|"
                       r"phase6_live|live_route_b|rfsim")
# Persistent host services that hold a GPU context without running CUDA work.
# Allowed only by exact process name and only while GPU utilization is <= 5 %.
GPU_SYSTEM_DAEMONS = frozenset({"/usr/libexec/gnome-remote-desktop-daemon",
                                "nvidia-cuda-mps-server"})
GPU_IDLE_UTILIZATION_PERCENT = 5

# Conservative disk model (bytes).  Calibrated against the smoke and reported.
FIXED_BUNDLE_BYTES = 8 * 2**20
PER_DECISION_EVENT_BYTES = 2048
PER_DECISION_LOG_BYTES = 2048
PER_UPDATE_LOG_BYTES = 1024
MIN_RESERVE_BYTES = 2 * 2**30


class CampaignRefused(RuntimeError):
    pass


# LAUNCH HOLD (user instruction 2026-09-30).  The inherited Run-4 v2 kernel
# composes latency from a retained residual that includes ground-truth
# evaluation/scoring time.  Run 5B must not smoke- or deep-train on it.  Launch
# stays refused until Run 4B exposes its hash-bound operational-latency
# provider; that exact provider and binding are then imported here (never
# reimplemented), so Run 4B and Run 5B share identical latency draws for
# identical seed/state/action inputs.
RUN4B_LATENCY_PROVIDER = None
LAUNCH_HOLD_TOKEN = "RUN5B_LAUNCH_HELD_PENDING_RUN4B_OPERATIONAL_LATENCY_PROVIDER"


def require_run4b_latency_provider() -> None:
    require(RUN4B_LATENCY_PROVIDER is not None, LAUNCH_HOLD_TOKEN)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CampaignRefused(message)


# ---------------------------------------------------------------------------
# Host and disk preflight
# ---------------------------------------------------------------------------


def host_state() -> dict[str, Any]:
    own = {str(os.getpid()), str(os.getppid())}
    found = subprocess.run(["pgrep", "-af", HOT_PROCESS_PATTERN], capture_output=True,
                           text=True).stdout.splitlines()
    processes = [line for line in found if line.split(" ", 1)[0] not in own
                 and "pgrep" not in line and "run5b_campaign" not in line]
    docker = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True,
                            text=True)
    containers = [n for n in docker.stdout.split() if "oai" in n.lower() or "carla" in n.lower()]
    gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name",
                          "--format=csv,noheader"], capture_output=True, text=True)
    apps = [line.strip() for line in gpu.stdout.splitlines() if line.strip()]
    allowed = [a for a in apps if a.split(",", 1)[-1].strip() in GPU_SYSTEM_DAEMONS]
    gpu_apps = [a for a in apps if a not in allowed]
    utilization = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True)
    util = [int(v) for v in utilization.stdout.split()] if utilization.returncode == 0 else []
    load = [float(v) for v in Path("/proc/loadavg").read_text().split()[:3]]
    cold = (not processes and not containers and not gpu_apps and docker.returncode == 0
            and gpu.returncode == 0 and bool(util)
            and max(util) <= GPU_IDLE_UTILIZATION_PERCENT
            and load[0] < 0.5 * (os.cpu_count() or 1))
    return {"cold": cold, "hot_processes": processes, "hot_containers": containers,
            "gpu_compute_apps": gpu_apps, "allowlisted_gpu_system_daemons": allowed,
            "gpu_utilization_percent": util, "loadavg": load, "cpu_count": os.cpu_count(),
            "docker_ok": docker.returncode == 0, "nvidia_smi_ok": gpu.returncode == 0}


TEMPORARY_ROOTS = ("/tmp", "/var/tmp", "/dev/shm", "/run")
VOLATILE_FILESYSTEMS = frozenset({"tmpfs", "ramfs", "overlay", "squashfs"})


def require_durable_directory(path: Path) -> dict[str, Any]:
    """Deep checkpoints must live on a persistent filesystem, never a temp tree."""
    resolved = Path(path).resolve()
    text = str(resolved)
    require(not any(text == root or text.startswith(root + "/") for root in TEMPORARY_ROOTS),
            f"campaign directory {resolved} is under a temporary root")
    require(not any(part.startswith(".") for part in resolved.parts[1:]),
            f"campaign directory {resolved} has a hidden/staging-like component")
    best, fstype = "", None
    for line in Path("/proc/mounts").read_text().splitlines():
        fields = line.split()
        mount = fields[1]
        if (text == mount or text.startswith(mount.rstrip("/") + "/")) and len(mount) > len(best):
            best, fstype = mount, fields[2]
    require(fstype is not None and fstype not in VOLATILE_FILESYSTEMS,
            f"campaign directory {resolved} is on a volatile filesystem ({fstype})")
    return {"path": text, "mount": best, "fstype": fstype}


def bundle_bytes(update: int) -> int:
    decisions = PR.CONFIG.warmup_decision_count + PR.CONFIG.environment_transitions_per_update * update
    return FIXED_BUNDLE_BYTES + PER_DECISION_EVENT_BYTES * decisions


def seed_bytes(target: int, checkpoints: Sequence[int]) -> int:
    decisions = PR.CONFIG.warmup_decision_count + PR.CONFIG.environment_transitions_per_update * target
    boundaries = sum(bundle_bytes(u) for u in checkpoints if u <= target)
    emergency = bundle_bytes(target)          # at most one emergency bundle per seed
    staging = bundle_bytes(target)            # one in-flight staging copy
    final_actor = FIXED_BUNDLE_BYTES
    logs = PER_DECISION_LOG_BYTES * decisions + PER_UPDATE_LOG_BYTES * target
    return boundaries + emergency + staging + final_actor + logs


def disk_preflight(directory: Path, *, seeds: Sequence[int], target: int,
                   checkpoints: Sequence[int]) -> dict[str, Any]:
    estimate = sum(seed_bytes(target, checkpoints) for _ in seeds)
    reserve = max(estimate, MIN_RESERVE_BYTES)
    probe = Path(directory).resolve()
    while not probe.exists():            # a nested, not-yet-created campaign path
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    deep = 3 * seed_bytes(PR.CONFIG.deep_target_update, PR.CONFIG.deep_checkpoints)
    return {"seeds": list(seeds), "target_update": target, "estimate_bytes": estimate,
            "reserve_bytes": reserve, "required_bytes": estimate + reserve,
            "free_bytes": free, "passed": free >= estimate + reserve,
            "deep_three_seed_estimate_bytes": deep,
            "deep_three_seed_required_bytes": deep + max(deep, MIN_RESERVE_BYTES),
            "deep_three_seed_passed_now": free >= deep + max(deep, MIN_RESERVE_BYTES),
            "model": {"fixed_bundle_bytes": FIXED_BUNDLE_BYTES,
                      "per_decision_event_bytes": PER_DECISION_EVENT_BYTES,
                      "per_decision_log_bytes": PER_DECISION_LOG_BYTES,
                      "per_update_log_bytes": PER_UPDATE_LOG_BYTES,
                      "reserve": "max(estimate, 2 GiB)"}}


# ---------------------------------------------------------------------------
# Fresh-process actor verification
# ---------------------------------------------------------------------------


def verify_actor_bundle(path: Path, *, seed: int, update: int) -> dict[str, Any]:
    bundle = B.verify_bundle(path)
    manifest = bundle.manifest
    require(bundle.kind in ("final_actor", "checkpoint") and bundle.update == update,
            "actor bundle identity")
    require(manifest["seed"] == seed and manifest["update_count"] == update, "seed/update")
    RM.require_run5b_identity(manifest, preregistration_sha256=manifest["preregistration_sha256"])
    require(manifest["preregistration_sha256"] == PR.load_sealed()["sha256"],
            "actor bundle belongs to another preregistration")
    data = bundle.payload("actor_state_dict.pt")
    state = B.torch_from_bytes(data)
    require(tuple(state["encoder.0.weight"].shape) == (128, 21), "actor is not 21-input")
    require(bool(state["encoder.0.weight"][:, RT.SNR_INDEX].abs().sum() > 0),
            "actor SNR column is identically zero")
    actor, _ = RM.require_run5b_checkpoint(
        manifest=manifest, actor_state=state, critic_state=None,
        preregistration_sha256=manifest["preregistration_sha256"],
        expected_tree_sha256=manifest["actor_tree_sha256"])
    actor.eval()
    tree = RT._tree_sha256(actor.state_dict())
    require(tree == manifest["actor_tree_sha256"], "actor tensor-tree hash differs")
    require(RT.actor_fixture_outputs(actor) == manifest["actor_fixtures"],
            "deterministic probe outputs differ")
    return {"verified": True, "pid": os.getpid(), "file_sha256": B.sha256_bytes(data),
            "tree_sha256": tree, "seed": seed, "update": update,
            "model_binding_sha256": manifest["model_binding_sha256"],
            "manifest_sha256": bundle.manifest_sha256, "bundle": bundle.name,
            "loader": "torch.load(weights_only=True)"}


# ---------------------------------------------------------------------------
# Smoke diagnostics
# ---------------------------------------------------------------------------


def _shim(actor):
    return type("Runner", (), {"model_bundle": type("Bundle", (), {"actor": actor})()})()


def _summaries(orchestrator, states, keys):
    shim = _shim(orchestrator.actor)
    return [R4S._actor_physical_summary(shim, orchestrator.collector.catalog, key, tuple(state))
            for state, key in zip(states, keys)]


def _mode_probs(actor, states) -> torch.Tensor:
    with torch.no_grad():
        return torch.softmax(actor(torch.tensor(states, dtype=torch.float32)).logits, dim=-1)


def controlled_snr_probe(orchestrator, seed: int) -> dict[str, Any]:
    history = orchestrator.history
    keys = [(d["reward_scene_key"], d["held_scene_key"])
            for d in orchestrator.collector.diagnostics()]
    low = [tuple(r.state[:RT.SNR_INDEX]) + (0.0,) for r in history]
    high = [tuple(r.state[:RT.SNR_INDEX]) + (1.0,) for r in history]
    s_low, s_high = _summaries(orchestrator, low, keys), _summaries(orchestrator, high, keys)
    dq = [h["expected_q_exec"] - l["expected_q_exec"] for l, h in zip(s_low, s_high)]
    dw = [h["expected_wire_bytes"] - l["expected_wire_bytes"] for l, h in zip(s_low, s_high)]
    tv = 0.5 * (_mode_probs(orchestrator.actor, high)
                - _mode_probs(orchestrator.actor, low)).abs().sum(-1)
    q_mean, q_ci = R4S._bootstrap_mean_ci(dq, draws=2000, seed=seed)
    w_mean, w_ci = R4S._bootstrap_mean_ci(dw, draws=2000, seed=seed + 1)
    return {"probe": "CONTROLLED_SNR_ONLY_SWEEP_5.5_TO_24.5_DB_ON_OBSERVED_STATES",
            "n_states": len(history), "expected_q_exec_delta_mean": q_mean,
            "expected_q_exec_delta_ci95": q_ci,
            "expected_two_tensor_ingress_bytes_delta_mean": w_mean,
            "expected_two_tensor_ingress_bytes_delta_ci95": w_ci,
            "mode_distribution_tv_mean": float(tv.mean()),
            "claim": "descriptive only at 500 updates; no direction is required"}


def shuffled_snr_control(orchestrator, seed: int) -> dict[str, Any]:
    history = orchestrator.history
    keys = [(d["reward_scene_key"], d["held_scene_key"])
            for d in orchestrator.collector.diagnostics()]
    snr = [r.state[RT.SNR_INDEX] for r in history]
    random.Random(seed).shuffle(snr)
    real = [tuple(r.state) for r in history]
    shuffled = [tuple(r.state[:RT.SNR_INDEX]) + (s,) for r, s in zip(history, snr)]
    s_real, s_shuf = _summaries(orchestrator, real, keys), _summaries(orchestrator, shuffled, keys)
    dq = [abs(a["expected_q_exec"] - b["expected_q_exec"]) for a, b in zip(s_real, s_shuf)]
    tv = 0.5 * (_mode_probs(orchestrator.actor, real)
                - _mode_probs(orchestrator.actor, shuffled)).abs().sum(-1)
    one_hot = torch.nn.functional.one_hot(torch.tensor([r.mode_id for r in history]),
                                          EXPECTED_MODE_COUNT).to(torch.float32)
    q_norm = torch.tensor([r.q_e4 / float(Q_E4_MAX) for r in history], dtype=torch.float32)
    with torch.no_grad():
        q_real = torch.min(*orchestrator.critics.q_values(
            torch.tensor(real, dtype=torch.float32), one_hot, q_norm))
        q_shuf = torch.min(*orchestrator.critics.q_values(
            torch.tensor(shuffled, dtype=torch.float32), one_hot, q_norm))
    return {"control": "AUDIT_ONLY_IN_SUPPORT_TEMPORALLY_SHUFFLED_SNR_ACTOR_CRITIC_INPUT",
            "seed": seed, "n_states": len(history),
            "actor_expected_q_exec_abs_delta_mean": statistics.fmean(dq),
            "actor_mode_distribution_tv_mean": float(tv.mean()),
            "critic_executed_action_abs_q_delta_mean": float((q_real - q_shuf).abs().mean()),
            "claim": "descriptive only; no gate on direction or magnitude"}


def exploration_report(orchestrator, metrics_lines: Sequence[dict]) -> dict[str, Any]:
    warm = len(orchestrator.schedule)
    warmup = orchestrator.history[:warm]
    actor = [r for r in orchestrator.history[warm:] if r.request.source == "STOCHASTIC_ACTOR"]
    counts = [sum(1 for r in actor if r.mode_id == m) for m in range(EXPECTED_MODE_COUNT)]
    total = sum(counts)
    entropy = -sum((c / total) * math.log(c / total) for c in counts if c)
    per_mode = {str(m): {"count": counts[m],
                         "distinct_q": len({r.q_e4 for r in actor if r.mode_id == m}),
                         "q_min": min((r.q_e4 for r in actor if r.mode_id == m), default=None),
                         "q_max": max((r.q_e4 for r in actor if r.mode_id == m), default=None)}
                for m in range(EXPECTED_MODE_COUNT)}
    q = [r.q_e4 for r in actor]
    return {
        "deterministic_warmup": {"decisions": len(warmup),
                                 "modes": sorted({r.mode_id for r in warmup}),
                                 "source": sorted({r.request.source for r in warmup})},
        "post_warmup_actor": {
            "decisions": total, "mode_counts": counts,
            "modes_selected": sum(1 for c in counts if c),
            "empirical_mode_entropy_nats": entropy,
            "max_entropy_nats": math.log(EXPECTED_MODE_COUNT),
            "mean_policy_discrete_entropy": statistics.fmean(
                m["discrete_entropy"] for m in metrics_lines),
            "q_e4_min": min(q), "q_e4_max": max(q), "distinct_q_e4": len(set(q)),
            "q_e4_pstdev": statistics.pstdev(q), "per_mode": per_mode}}


def smoke_gates(orchestrator, metrics_lines, host, resume_proof) -> dict[str, Any]:
    history = orchestrator.history
    diagnostics = orchestrator.collector.diagnostics()
    exploration = exploration_report(orchestrator, metrics_lines)
    variation = RT.state_variation(history)
    successors = [d["successor_snr_db"] for d in diagnostics]
    later_equal = sum(int(d["snr_db"] in set(successors[i:])) for i, d in enumerate(diagnostics))
    previous_mismatches = sum(
        int(tuple(r.state[RT.PREVIOUS_SLICE]) != RT.expected_previous_features(p))
        for p, r in zip(history, history[1:]))
    params_finite = all(bool(torch.isfinite(p).all()) for p in
                        (*orchestrator.actor.parameters(), *orchestrator.critics.parameters()))
    metrics_finite = all(math.isfinite(float(line[f])) for line in metrics_lines
                         for f in ("critic_loss", "actor_loss", "critic_grad_norm",
                                   "actor_grad_norm", "target_mean", "discrete_entropy"))
    post = exploration["post_warmup_actor"]
    spread_modes = sum(1 for v in post["per_mode"].values() if v["distinct_q"] > 1)
    gates = {
        "host_confirmed_cold": bool(host.get("cold")),
        "losses_gradients_parameters_finite": params_finite and metrics_finite
                                               and len(metrics_lines) == orchestrator.update_count,
        "all_12_modes_selected_by_post_warmup_actor": post["modes_selected"] == 12,
        "continuous_q_non_degenerate": (post["distinct_q_e4"] > 12 and post["q_e4_pstdev"] > 0
                                        and spread_modes >= 6),
        "state_features_vary": all(v["distinct"] >= 2 and v["span"] > 0
                                   for v in variation.values()),
        "reward_matches_registered_formula": all(RT.reward_matches(r) for r in history),
        "previous_outcome_encoding_equals_transport_prior": previous_mismatches == 0,
        "no_qperc_in_actor_state": (RT.qperc_leaks(history) == 0
                                    and not any("qperc" in n or "quality" in n
                                                for n in C.RUN5B_POLICY_FEATURE_ORDER)),
        "current_snr_never_a_future_sample": (
            orchestrator.collector._channel.future_sample_violations == 0
            and all(d["generated_ticks_after_observed"] for d in diagnostics)
            and later_equal == 0),
        "four_profiles_balanced": _balanced(orchestrator.collector._channel.profile_balance()),
        **resume_proof,
    }
    return {"gates": gates, "exploration": exploration, "state_variation": variation,
            "previous_outcome_mismatches": previous_mismatches,
            "previous_qperc_leaks": RT.qperc_leaks(history),
            "current_snr_equal_to_a_later_generated_sample": later_equal,
            "profile_segments_audit_only": orchestrator.collector._channel.profile_balance()}


def _balanced(counts: dict[str, int]) -> bool:
    values = list(counts.values())
    return len(values) == 4 and max(values) - min(values) <= 1 and min(values) >= 1


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _metrics_dict(metrics) -> dict[str, Any]:
    return {f: getattr(metrics, f) for f in metrics.__dataclass_fields__}


def _decision_dict(orchestrator, ordinal: int, record) -> dict[str, Any]:
    diag = orchestrator.collector.diagnostics()[ordinal]
    return {"source": record.request.source, "mode_id": record.mode_id, "q_e4": record.q_e4,
            "reward": record.reward, "terminal": record.terminal, "q_perc": record.q_perc,
            "latency_ms": record.latency_ms, "snr_db": diag["snr_db"],
            "prior_ul_mcs": diag["prior_ul_mcs"], "successor_mcs": diag["successor_mcs"],
            "backlog_bytes": diag["pre_enqueue_backlog_bytes"], "digest": record.digest}


def run(args) -> int:
    require_run4b_latency_provider()
    prereg = PR.load_sealed()
    smoke = args.mode == "smoke"
    target = PR.CONFIG.smoke_stop_update if smoke else PR.CONFIG.deep_target_update
    checkpoints = PR.CONFIG.smoke_checkpoints if smoke else PR.CONFIG.deep_checkpoints
    if not smoke:
        document = json.loads(AUTHORIZATION.read_text()) if AUTHORIZATION.is_file() else None
        if not document or document.get("preregistration_sha256") != prereg["sha256"]:
            print("RUN5B_DEEP_TRAINING_NOT_AUTHORIZED", file=sys.stderr)
            return EXIT_REFUSED
    require(args.seed in PR.CONFIG.seed_order, "seed is not registered")
    host = {"cold": None, "skipped_for_tests": True} if args.skip_host_check_for_tests \
        else host_state()
    if not args.skip_host_check_for_tests and not host["cold"]:
        print(json.dumps({"HOST_NOT_COLD": host}), file=sys.stderr)
        return EXIT_REFUSED
    campaign = Path(args.campaign_dir)
    seed_dir = campaign / f"seed_{args.seed}"
    checkpoint_dir = seed_dir / "checkpoints"
    if smoke:
        reserve_seeds = [args.seed]
    else:
        durable = require_durable_directory(campaign)
        reserve_seeds = [seed for seed in PR.CONFIG.seed_order
                         if not (campaign / f"seed_{seed}" / "SEED_COMPLETE.json").exists()]
    disk = disk_preflight(campaign, seeds=reserve_seeds or [args.seed], target=target,
                          checkpoints=checkpoints)
    if not disk["passed"]:
        print(json.dumps({"DISK_PREFLIGHT_FAILED": disk}), file=sys.stderr)
        return EXIT_REFUSED
    if args.resume:
        require(checkpoint_dir.is_dir(), "--resume needs an existing seed directory")
    else:
        require(not seed_dir.exists(), "seed directory exists; use --resume")
        checkpoint_dir.mkdir(parents=True)
    torch.set_num_threads(PR.CONFIG.torch_intraop_threads)
    shared = RC.build_shared_sources(ARTIFACT, Path(args.evidence_root))

    def factory():
        return RC.Run5BModeledCollectorV1(artifact_path=ARTIFACT, seed=args.seed,
                                         evidence_root=Path(args.evidence_root),
                                         shared_sources=shared)

    runs = B.AppendOnlyJsonl(seed_dir / "runs.jsonl")
    metrics_log = B.AppendOnlyJsonl(seed_dir / "metrics.jsonl")
    decisions_log = B.AppendOnlyJsonl(seed_dir / "decisions.jsonl")
    selection = B.select_resume(checkpoint_dir)
    started = time.time()
    if selection.bundle is None:
        require(not args.resume or len(metrics_log) == 0, "ledgers exist without a bundle")
        orchestrator = RT.Run5BOrchestratorV1(collector_factory=factory, seed=args.seed,
                                             checkpoint_updates=checkpoints,
                                             preregistration_sha256=prereg["sha256"])
        resumed_from = None
    else:
        require(args.resume, "bundles exist; use --resume")
        manifest = selection.bundle.manifest
        metrics_log.require_prefix(manifest["metrics_prefix"])
        decisions_log.require_prefix(manifest["decisions_prefix"])
        orchestrator = RT.Run5BOrchestratorV1.restore_from_bundle(
            selection.bundle, collector_factory=factory, preregistration_sha256=prereg["sha256"])
        require(orchestrator.checkpoint_updates == tuple(sorted(checkpoints)),
                "resumed bundle belongs to another mode")
        resumed_from = selection.bundle.name
    runs.record(len(runs), {"event": "RESUME" if resumed_from else "START", "mode": args.mode,
                            "resumed_from": resumed_from, "latest_status": selection.latest_status,
                            "ignored_staging": list(selection.ignored_staging),
                            "update_count": orchestrator.update_count, "host": host,
                            "disk_preflight": disk, "pid": os.getpid(), "utc_s": time.time()})

    def on_signal(signum, _frame):
        orchestrator.stop_requested = True
        orchestrator.stop_signal = signal.Signals(signum).name

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    def on_decision(ordinal, record):
        decisions_log.record(ordinal, _decision_dict(orchestrator, ordinal, record))

    def on_update(metrics):
        metrics_log.record(metrics.update_index - 1, _metrics_dict(metrics))
        if args.stop_after_update is not None and metrics.update_index == args.stop_after_update:
            orchestrator.stop_requested = True
            orchestrator.stop_signal = "STOP_AFTER_UPDATE_TEST_HOOK"

    def on_boundary(kind):
        payloads, identity = orchestrator.bundle_payloads()
        identity.update({
            "kind": kind.upper(), "selection_candidate": kind == "checkpoint",
            "metrics_prefix": metrics_log.prefix(orchestrator.update_count),
            "decisions_prefix": decisions_log.prefix(orchestrator.decision_count),
            "actor_tree_sha256": RT._tree_sha256(orchestrator.actor.state_dict()),
            "mode": args.mode})
        B.publish_bundle(checkpoint_dir, B.bundle_name(kind, orchestrator.update_count),
                         payloads, identity)

    status = orchestrator.run_to(target, on_decision=on_decision, on_update=on_update,
                                 on_boundary=on_boundary)
    if status == "STOPPED":
        runs.record(len(runs), {"event": "STOPPED", "update_count": orchestrator.update_count,
                                "reason": getattr(orchestrator, "stop_signal", None),
                                "utc_s": time.time()})
        print(json.dumps({"status": "STOPPED", "update": orchestrator.update_count}))
        return EXIT_STOPPED
    return complete(args, orchestrator, seed_dir, checkpoint_dir, target, checkpoints, prereg,
                    metrics_log, decisions_log, runs, host, disk, started, smoke)


def complete(args, orchestrator, seed_dir, checkpoint_dir, target, checkpoints, prereg,
             metrics_log, decisions_log, runs, host, disk, started, smoke) -> int:
    verified = []
    for update in checkpoints:
        bundle = B.verify_bundle(checkpoint_dir / B.bundle_name("checkpoint", update))
        require(bundle.manifest["update_count"] == update and bundle.manifest["seed"] == args.seed,
                "registered boundary identity differs")
        verified.append({"name": bundle.name, "manifest_sha256": bundle.manifest_sha256,
                         "event_sha256": bundle.manifest["event_sha256"]})
    chosen = B.verify_bundle(checkpoint_dir / B.bundle_name("checkpoint", target))
    require(chosen.manifest["selection_candidate"] is True, "chosen actor is not a candidate")
    actor_name = B.bundle_name("final_actor", target)
    if not (seed_dir / actor_name).exists():
        B.publish_bundle(seed_dir, actor_name,
                         {"actor_state_dict.pt": chosen.payload("actor_state_dict.pt")},
                         {"seed": args.seed, "update_count": target,
                          "source_bundle": chosen.name,
                          "source_manifest_sha256": chosen.manifest_sha256,
                          "model_binding": chosen.manifest["model_binding"],
                          "model_binding_sha256": chosen.manifest["model_binding_sha256"],
                          "feature_schema_id": chosen.manifest["feature_schema_id"],
                          "feature_schema_sha256": chosen.manifest["feature_schema_sha256"],
                          "feature_order": chosen.manifest["feature_order"],
                          "feature_order_sha256": chosen.manifest["feature_order_sha256"],
                          "binding_sha256": chosen.manifest["binding_sha256"],
                          "preregistration_sha256": prereg["sha256"],
                          "actor_tree_sha256": chosen.manifest["actor_tree_sha256"],
                          "actor_fixtures": chosen.manifest["actor_fixtures"],
                          "registered_live_actor": (not smoke
                                                    and args.seed == PR.CONFIG.live_actor_seed
                                                    and target == PR.CONFIG.live_actor_update),
                          "selection_rule": PR.DESIGN["live_actor"]["rule"]},
                         update_latest=False)
    evaluation = [(checkpoint_dir / B.bundle_name("checkpoint", u), u)
                  for u in PR.CONFIG.evaluation_checkpoints if u <= target]
    fresh_results = fresh_verify([(seed_dir / actor_name, target), *evaluation], args.seed)
    fresh_result = fresh_results[0]
    evaluation_actors = fresh_results[1:]
    require(fresh_result["verified"] and fresh_result["pid"] != os.getpid()
            and fresh_result["tree_sha256"] == chosen.manifest["actor_tree_sha256"],
            "fresh-process verification does not match the chosen actor")
    metrics_lines = [json.loads(line) for line in metrics_log.lines]
    report = None
    if smoke:
        resume_proof = {"committed_bundles_verified": len(verified) == len(checkpoints),
                        "fresh_process_weights_only_actor_load": True,
                        "evaluation_checkpoint_actors_cold_loadable":
                            all(r["verified"] for r in evaluation_actors)}
        report = {
            "schema": "splitfusion.run5b.smoke_report.v1", "seed": args.seed, "target": target,
            "host": host, "disk_preflight": disk, "preflight": orchestrator.preflight,
            **smoke_gates(orchestrator, metrics_lines, host, resume_proof),
            "controlled_snr_response_at_500": controlled_snr_probe(orchestrator, args.seed),
            "shuffled_snr_control_at_500": shuffled_snr_control(orchestrator, args.seed),
            "metrics_at_boundaries": {str(u): metrics_lines[u - 1] for u in checkpoints if u},
            "eligible_gt": PR.DESIGN["scenes"]["eligible_gt"],
            "session_rollovers": 0,
            "claims_not_made": ["convergence", "policy performance", "deployment readiness"],
            "campaign_started": False}
        report["all_gates_passed"] = all(report["gates"].values())
        B.atomic_write_file(seed_dir / "RUN5B_SMOKE_REPORT.json",
                            json.dumps(report, indent=1, sort_keys=True, default=str).encode())
    completion = {
        "schema": "splitfusion.run5b.seed_complete.v1", "mode": args.mode, "seed": args.seed,
        "target_update": target, "preregistration_sha256": prereg["sha256"],
        "binding_sha256": orchestrator.binding_sha256,
        "model_binding_sha256": RM.RUN5B_TRAINING_MODEL_BINDING_SHA256,
        "feature_schema_sha256": C.FEATURE_SCHEMA_SHA256,
        "boundaries": verified, "final_actor": fresh_result,
        "evaluation_checkpoint_actors": evaluation_actors,
        "metrics_prefix": metrics_log.prefix(target),
        "decisions_prefix": decisions_log.prefix(orchestrator.decision_count),
        "smoke_all_gates_passed": None if report is None else report["all_gates_passed"],
        "seconds_this_invocation": time.time() - started}
    name = "SMOKE_SEED_COMPLETE.json" if smoke else "SEED_COMPLETE.json"
    B.atomic_write_file(seed_dir / name, B.canonical_bytes(completion))
    runs.record(len(runs), {"event": "COMPLETE", "file": name, "utc_s": time.time()})
    print(json.dumps({"status": "COMPLETE", "file": name,
                      "smoke_all_gates_passed": completion["smoke_all_gates_passed"]}))
    return 0 if report is None or report["all_gates_passed"] else 2


def fresh_verify(items, seed: int) -> list[dict[str, Any]]:
    """Cold-load every (bundle, update) in ONE new interpreter with weights_only."""
    command = [sys.executable, "-m", "rl_agent.splitfusion_hybrid_sac_run5b_v1.run5b_campaign",
               "--seed", str(seed)]
    for path, update in items:
        command += ["--verify-actor", str(path), "--expect-update", str(update)]
    fresh = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    require(fresh.returncode == 0, f"fresh-process actor verification failed: {fresh.stderr}")
    results = json.loads(fresh.stdout.strip().splitlines()[-1])
    require(len(results) == len(items) and all(r["verified"] for r in results)
            and all(r["pid"] != os.getpid() for r in results),
            "fresh-process verification incomplete")
    return results


def campaign_complete(campaign_dir: Path) -> dict[str, Any]:
    """Bind all three deep SEED_COMPLETE manifests after re-verifying every actor."""
    prereg = PR.load_sealed()
    seeds = {}
    for seed in PR.CONFIG.seed_order:
        seed_dir = Path(campaign_dir) / f"seed_{seed}"
        path = seed_dir / "SEED_COMPLETE.json"
        require(path.is_file(), f"seed {seed} is not complete")
        completion = json.loads(path.read_text())
        require(completion["mode"] == "deep"
                and completion["target_update"] == PR.CONFIG.deep_target_update
                and completion["preregistration_sha256"] == prereg["sha256"],
                f"seed {seed} completion is not a deep run under this preregistration")
        for update in PR.CONFIG.deep_checkpoints:
            B.verify_bundle(seed_dir / "checkpoints" / B.bundle_name("checkpoint", update))
        items = [(seed_dir / B.bundle_name("final_actor", PR.CONFIG.deep_target_update),
                  PR.CONFIG.deep_target_update)] + [
            (seed_dir / "checkpoints" / B.bundle_name("checkpoint", u), u)
            for u in PR.CONFIG.evaluation_checkpoints]
        seeds[str(seed)] = {"seed_complete_sha256": B.sha256_bytes(path.read_bytes()),
                            "actors": fresh_verify(items, seed)}
    document = {"schema": "splitfusion.run5b.campaign_complete.v1",
                "preregistration_sha256": prereg["sha256"], "seeds": seeds,
                "registered_live_actor": {"seed": PR.CONFIG.live_actor_seed,
                                          "update": PR.CONFIG.live_actor_update},
                "registered_actor_rule": PR.DESIGN["live_actor"]["rule"]}
    B.atomic_write_file(Path(campaign_dir) / "CAMPAIGN_COMPLETE.json", B.canonical_bytes(document))
    return document


def rehearsal(args) -> int:
    """Clean-process, zero-update launch rehearsal for one deep seed.

    Performs the full deep launch preflight (seal, host, durable path, disk,
    evidence, models, collector) plus the 288-decision no-gradient warm-up
    preflight, then stops.  It creates no campaign or checkpoint directory,
    takes no optimizer step, and writes only ``--rehearsal-report``.
    """
    require_run4b_latency_provider()
    prereg = PR.load_sealed()
    campaign = Path(args.campaign_dir)
    require(not campaign.exists(), "rehearsal must not touch an existing campaign directory")
    durable = require_durable_directory(campaign)
    host = host_state()
    disk = disk_preflight(campaign, seeds=PR.CONFIG.seed_order,
                          target=PR.CONFIG.deep_target_update,
                          checkpoints=PR.CONFIG.deep_checkpoints)
    torch.set_num_threads(PR.CONFIG.torch_intraop_threads)
    shared = RC.build_shared_sources(ARTIFACT, Path(args.evidence_root))
    factory = lambda: RC.Run5BModeledCollectorV1(
        artifact_path=ARTIFACT, seed=args.seed, evidence_root=Path(args.evidence_root),
        shared_sources=shared)
    started = time.time()
    steps = {"adam": 0}
    original_step = torch.optim.Adam.step

    def counted(self, *a, **k):
        steps["adam"] += 1
        return original_step(self, *a, **k)

    torch.optim.Adam.step = counted
    try:
        orchestrator = RT.Run5BOrchestratorV1(
            collector_factory=factory, seed=args.seed, checkpoint_updates=PR.CONFIG.deep_checkpoints,
            preregistration_sha256=prereg["sha256"])
        preflight = orchestrator.run_preflight()
    finally:
        torch.optim.Adam.step = original_step
    document = {
        "schema": "splitfusion.run5b.launch_rehearsal.v1", "seed": args.seed, "pid": os.getpid(),
        "preregistration_sha256": prereg["sha256"], "binding_sha256": orchestrator.binding_sha256,
        "host": host, "durable_path": durable, "disk_preflight": disk,
        "update_count": orchestrator.update_count, "optimizer_steps": steps["adam"],
        "optimizer_states_empty": (not orchestrator.trainer.actor_optimizer.state
                                   and not orchestrator.trainer.critic_optimizer.state),
        "campaign_directory_created": campaign.exists(),
        "preflight_passed": preflight["passed"], "preflight": preflight,
        "warmup_seconds": time.time() - started,
        "checkpoint_updates": list(PR.CONFIG.deep_checkpoints),
        "evaluation_checkpoints": list(PR.CONFIG.evaluation_checkpoints),
        "target_update": PR.CONFIG.deep_target_update}
    document["passed"] = (host["cold"] and disk["passed"] and preflight["passed"]
                          and document["optimizer_steps"] == 0
                          and document["optimizer_states_empty"]
                          and not document["campaign_directory_created"]
                          and orchestrator.update_count == 0)
    B.atomic_write_file(Path(args.rehearsal_report),
                        json.dumps(document, indent=1, sort_keys=True, default=str).encode())
    print(json.dumps({"passed": document["passed"], "seed": args.seed,
                      "optimizer_steps": steps["adam"]}))
    return 0 if document["passed"] else 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "deep"))
    parser.add_argument("--campaign-dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--evidence-root")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-update", type=int, default=None)
    parser.add_argument("--skip-host-check-for-tests", action="store_true")
    parser.add_argument("--verify-actor", action="append")
    parser.add_argument("--expect-update", type=int, action="append")
    parser.add_argument("--rehearsal", action="store_true")
    parser.add_argument("--rehearsal-report")
    parser.add_argument("--finalize-campaign", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_actor:
        require(len(args.verify_actor) == len(args.expect_update or []),
                "each --verify-actor needs one --expect-update")
        print(json.dumps([verify_actor_bundle(Path(path), seed=args.seed, update=update)
                          for path, update in zip(args.verify_actor, args.expect_update)]))
        return 0
    try:
        if args.rehearsal:
            require(args.mode == "deep" and args.rehearsal_report, "rehearsal is deep-mode only")
            return rehearsal(args)
        if args.finalize_campaign:
            print(json.dumps(campaign_complete(Path(args.campaign_dir))))
            return 0
        return run(args)
    except (CampaignRefused, B.BundleError, PR.PreregistrationError,
            RM.Run5BModelError) as exc:
        print(f"RUN5B_REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
