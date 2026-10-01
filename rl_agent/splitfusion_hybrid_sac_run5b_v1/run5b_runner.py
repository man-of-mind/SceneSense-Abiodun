#!/usr/bin/env python3
"""Run-5B offline Hybrid-SAC runner, durable checkpoints and campaign CLI.

DERIVED MECHANICALLY from ``splitfusion_hybrid_sac_run4b_v1/runner.py`` by
``derive_from_run4b.py``; every difference is one listed substitution.
Run 5B is Run 4B on the shared joint SNR/MCS channel
(``splitfusion_joint_channel_v1``) with the causal UL-SNR proxy as feature 21.
The campaign records, but does not enforce, the Run-4B family-dominance window.

CPU-only and offline: no CARLA, OAI, Docker, CUDA or network activity.

Frozen settings (unchanged from Run-4): seeds 17/29/43, gamma 0.99,
alpha_d 0.05, alpha_c 0.02, actor/critic LR 3e-4, tau 0.005, batch 256,
replay 65,536, 288 stratified warm-up decisions, 4 new transitions per update,
4 intra-op threads.  Checkpoints: 0, 100, 250, 500, then every 500 to 10,000.

A checkpoint bundle is a create-only directory published by atomic rename:

    training_state.pt       actor, twin critics + targets, both optimizers,
                            replay contents, all torch generator states
    environment_state.json  environment, MCS and A/E/D latency RNG states,
                            prior, context, counters and ledger chains
    manifest.json           schema, full binding, file hashes, fingerprint
    COMMITTED               SHA-256 of manifest.json, written last

``LATEST`` in the checkpoint directory is replaced atomically afterwards.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import torch

from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as R4O
from rl_agent.splitfusion_hybrid_sac_v1 import action_contract
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import (
    ExecutedActionIdentity,
)
from rl_agent.splitfusion_operational_latency_v1 import provider as OPL

from rl_agent.splitfusion_joint_channel_v1 import joint_channel as E

from . import run5b_checks as K
from . import run5b_learner as L
from . import run5b_models as M
from . import run5b_registration as REG
from . import run5b_state_contract as C

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = Path(os.environ.get("RUN5B_EVIDENCE_ROOT",
                                    ROOT.parent / "abiodun")).resolve()
RUNNER_SCHEMA = "splitfusion.run5b.runner.v1"
BUNDLE_SCHEMA = "splitfusion.run5b.checkpoint_bundle.v1"
EXPORT_SCHEMA = "splitfusion.run5b.actor_export.v1"
SEEDS = (17, 29, 43)
WARMUP = 288
TRANSITIONS_PER_UPDATE = 4
BATCH = 256
THREADS = 4
SMOKE_UPDATE = 500
FINAL_UPDATE = 10_000
RESUME_STOP_UPDATE = 250
CHECKPOINT_UPDATES = (0, 100, 250, 500) + tuple(range(1000, FINAL_UPDATE + 1, 500))
LIVE_ACTOR = {"seed": 43, "update": 10_000}
# Pre-registered campaign stop: the policy is "dominated by the
# under-observed families" if noAE+AE128 exceed this share of the
# stochastic-actor decisions since the previous registered checkpoint, at
# any checkpoint >= update 500.
UNDER_OBSERVED_FAMILIES = ("noAE", "AE128")
DOMINANCE_MAX_SHARE = 0.50
REGISTERED_PROVIDER_BINDING_SHA256 = (
    "3cc1e6e36ae6e4f43dc8110077c62efbf5c719aaa179935b0f6de4c5bb1f6c29")
TRAINER_CONFIG = L.TrainerConfigV1(alpha_d=0.05, alpha_c=0.02, tau=0.005,
                                   actor_lr=3e-4, critic_lr=3e-4,
                                   nominal_batch_size=BATCH)
BUNDLE_FILES = ("training_state.pt", "environment_state.json", "manifest.json")
GENERATOR_NAMES = ("decision_q", "decision_mode", "replay", "trainer_target",
                   "trainer_actor")


class RunnerError(RuntimeError):
    """A runner, schedule or checkpoint invariant failed (fail closed)."""


class CheckpointRefused(RunnerError):
    """A bundle is foreign, tampered or bound to a different contract."""


class DominanceStop(RunnerError):
    """The pre-registered under-observed-family dominance stop fired."""


def _require(condition: bool, message: str, error=RunnerError) -> None:
    if not condition:
        raise error(message)


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _chain(previous: str, row: Mapping[str, Any]) -> str:
    return _sha_bytes(previous.encode("ascii") + _canonical(row))


GENESIS_CHAIN = "0" * 64


def fixture_states(scaling: C.ScalingV1) -> list[tuple[float, ...]]:
    """20 deterministic, run-independent actor fixtures."""
    states = []
    for index in range(20):
        if index % 3 == 0:
            prior = C.OperationalPriorV1.genesis()
        elif index % 3 == 1:
            prior = C.OperationalPriorV1.from_outcome(
                mode_id=index % 12, q_e4=(index * 487) % 9801, timely=True,
                operational_latency_ns=60_000_000 + index * 5_000_000)
        else:
            prior = C.OperationalPriorV1.from_outcome(
                mode_id=(index * 5) % 12, q_e4=(index * 911) % 9801,
                timely=False, operational_latency_ns=None)
        observation = C.ObservationV1(
            camera_si=scaling.camera_si_center
            + (index - 10) * 0.2 * scaling.camera_si_scale,
            radar_p40=(index % 11) / 10.0, prior_ul_mcs=(index * 3) % 29,
            pre_action_rlc_backlog_bytes=(index % 4) * 250_000)
        states.append(C.build_features(observation, prior, scaling,
                                       (index % 20) / 19.0))
    return states


class Run5BRunnerV1:
    """One registered seed: models, replay, trainer, environment, schedule."""

    def __init__(self, sources: E.JointSourcesV1, seed: int) -> None:
        _require(seed in SEEDS, f"seed {seed} is not registered")
        _require(torch.get_num_threads() == THREADS,
                 "torch intra-op threads must be exactly 4")
        _require(sources.run4b.provider.binding_sha256
                 == REGISTERED_PROVIDER_BINDING_SHA256
                 or not sources.run4b.provider.pinned,
                 "operational-latency provider drift")
        self.sources = sources
        self.seed = seed
        self.plan = R4O.RunnerSeedPlanV1.for_registered_seed(seed)
        plan = self.plan
        self.actor, self.critics = M.build_models(
            actor_seed=plan.actor_seed, critic_seed=plan.critic_seed)
        self.generators = {name: torch.Generator(device="cpu")
                           for name in GENERATOR_NAMES}
        for name, value in (("decision_q", plan.decision_q_seed),
                            ("decision_mode", plan.decision_mode_seed),
                            ("replay", plan.replay_seed),
                            ("trainer_target", plan.target_seed),
                            ("trainer_actor", plan.trainer_actor_seed)):
            self.generators[name].manual_seed(value)
        self.trainer = L.Run4BTrainerV1(
            actor=self.actor, critics=self.critics, config=TRAINER_CONFIG,
            target_generator=self.generators["trainer_target"],
            actor_generator=self.generators["trainer_actor"])
        self.replay = L.ReplayBufferV1(L.REPLAY_CAPACITY)
        self.env = E.JointChannelEnvironmentV1(sources, seed=seed)
        self.snr = C.ModeledSnrObserverV1(seed)
        self.registration_sha256 = REG.sealed_sha256()
        _require(sources.binding_sha256 == REG.sealed_joint_channel_binding_sha256(),
                 "joint-channel binding differs from the Run-5B registration")
        self.schedule = R4O.build_frozen_warmup_schedule(seed)
        _require(len(self.schedule) == WARMUP, "warm-up is not 288 decisions")
        self.decision_count = 0
        self.decision_chain = GENESIS_CHAIN
        self.metrics_chain = GENESIS_CHAIN
        self.preflight: Optional[dict[str, Any]] = None
        self.binding = self._binding_document()
        self.binding_sha256 = C.canonical_sha256(self.binding)

    # -- binding ---------------------------------------------------------
    def _binding_document(self) -> dict[str, Any]:
        cfg = TRAINER_CONFIG
        return {
            "schema": RUNNER_SCHEMA,
            "seed": self.seed,
            "seed_plan": self.plan.to_dict(),
            "feature_schema_id": C.FEATURE_SCHEMA["schema_id"],
            "feature_schema_sha256": C.FEATURE_SCHEMA_SHA256,
            "feature_order": list(C.FEATURE_ORDER),
            "feature_order_sha256": C.FEATURE_ORDER_SHA256,
            "joint_channel_binding_sha256": self.sources.binding_sha256,
            "registration_sha256": self.registration_sha256,
            "reward_schema_sha256": C.REWARD_SCHEMA_SHA256,
            "model_binding_sha256": M.MODEL_BINDING_SHA256,
            "environment_binding": self.sources.binding_document(),
            "trainer": {"alpha_d": cfg.alpha_d, "alpha_c": cfg.alpha_c,
                        "tau": cfg.tau, "actor_lr": cfg.actor_lr,
                        "critic_lr": cfg.critic_lr, "batch": BATCH,
                        "gamma": C.GAMMA, "replay_capacity": L.REPLAY_CAPACITY,
                        "warmup": WARMUP,
                        "transitions_per_update": TRANSITIONS_PER_UPDATE,
                        "threads": THREADS},
            "checkpoint_updates": list(CHECKPOINT_UPDATES),
        }

    @property
    def update_count(self) -> int:
        return self.trainer.update_count

    # -- acting ----------------------------------------------------------
    def _resolve(self, mode_id: int, q_e4: int) -> None:
        catalog = self.sources.run4b.action_catalog
        action = ExecutedActionIdentity.from_executable_action(
            catalog.resolve(mode_id, q_e4 / float(action_contract.Q_E4_SCALE)),
            catalog)
        _require((action.mode_id, action.q_e4) == (mode_id, q_e4),
                 "catalog changed the selected action")

    def _next_action(self, features: tuple[float, ...]) -> dict[str, Any]:
        ordinal = self.decision_count
        if ordinal < WARMUP:
            selected = self.schedule.action_at(ordinal)
            return {"source": "STRATIFIED_WARMUP", "mode_id": selected.mode_id,
                    "q_e4": selected.q_e4,
                    "warmup_q_bin": selected.q_bin_index}
        state = torch.tensor((features,), dtype=torch.float32)
        with torch.no_grad():
            sample = self.actor.sample_all_modes(
                state, generator=self.generators["decision_q"])
            mode_id = int(torch.multinomial(
                sample.probs[0], 1,
                generator=self.generators["decision_mode"])[0])
            q_e4 = int(sample.q_e4[0, mode_id])
        return {"source": "STOCHASTIC_ACTOR", "mode_id": mode_id,
                "q_e4": q_e4, "warmup_q_bin": None}

    def collect_one(self) -> tuple[dict[str, Any], E.E4.TransitionV1]:
        features = C.features_for(self.env, self.snr)
        choice = self._next_action(features)
        self._resolve(choice["mode_id"], choice["q_e4"])
        transition = self.env.step(choice["mode_id"], choice["q_e4"])
        _require(transition.state == features[:C.SNR_FEATURE_INDEX],
                 "state changed during step")
        next_features = C.features_for(self.env, self.snr)
        _require(transition.next_state == next_features[:C.SNR_FEATURE_INDEX],
                 "successor Run-4B prefix differs")
        transition = dataclasses.replace(transition, state=features,
                                         next_state=next_features)
        self.replay.add(state=transition.state,
                        next_state=transition.next_state,
                        mode_id=transition.mode_id, q_e4=transition.q_e4,
                        reward=transition.reward, discount=transition.discount)
        diag = transition.diagnostics
        row = {"ordinal": self.decision_count, "source": choice["source"],
               "warmup_q_bin": choice["warmup_q_bin"],
               "transition_sha256": transition.digest(),
               **{k: diag[k] for k in (
                   "mode_id", "q_e4", "family", "terminal", "reward",
                   "q_perc_training_only", "operational_latency_ns",
                   "composed_total_ns", "components_ns", "transport_success",
                   "prior_ul_mcs", "pre_action_backlog_bytes",
                   "reward_wire_bytes", "snr_db", "successor_mcs",
                   "successor_snr_db", "generated_ticks_after_observed")},
               "snr_feature": transition.state[C.SNR_FEATURE_INDEX],
               "prev_present": transition.state[18],
               "prev_success": transition.state[19]}
        self.decision_count += 1
        self.decision_chain = _chain(self.decision_chain, row)
        return row, transition

    # -- fingerprints ----------------------------------------------------
    def _model_sha256(self) -> str:
        return R4O._tree_sha256({"actor": self.actor.state_dict(),
                                 "critics": self.critics.state_dict()})

    def _optimizer_sha256(self) -> str:
        return R4O._tree_sha256({
            "actor": self.trainer.actor_optimizer.state_dict(),
            "critic": self.trainer.critic_optimizer.state_dict()})

    def training_state(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critics": self.critics.state_dict(),
            "actor_optimizer": self.trainer.actor_optimizer.state_dict(),
            "critic_optimizer": self.trainer.critic_optimizer.state_dict(),
            "replay": self.replay.state_dict(),
            "generators": {name: self.generators[name].get_state()
                           for name in GENERATOR_NAMES},
        }

    def counters(self) -> dict[str, Any]:
        return {"update_count": self.update_count,
                "decision_count": self.decision_count,
                "decision_chain": self.decision_chain,
                "metrics_chain": self.metrics_chain}

    def fingerprint(self) -> str:
        return C.canonical_sha256({
            "training": R4O._tree_document(self.training_state()),
            "environment": self.env.state_dict(),
            "counters": self.counters()})

    # -- preflight -------------------------------------------------------
    def run_preflight(self, on_decision: Callable) -> dict[str, Any]:
        _require(self.decision_count == 0 and self.update_count == 0,
                 "preflight must start at genesis")
        model_before = self._model_sha256()
        optimizer_before = self._optimizer_sha256()
        rows, transitions = [], []
        for _ in range(WARMUP):
            row, transition = self.collect_one()
            on_decision(row)
            rows.append(row)
            transitions.append(transition)
        checks: dict[str, bool] = {}
        checks["exact_schedule_identity"] = all(
            (r["mode_id"], r["q_e4"]) == (self.schedule.action_at(i).mode_id,
                                          self.schedule.action_at(i).q_e4)
            for i, r in enumerate(rows))
        cells: dict[tuple[int, int], int] = {}
        for r in rows:
            key = (r["mode_id"], r["warmup_q_bin"])
            cells[key] = cells.get(key, 0) + 1
        checks["coverage_12x6x4"] = (len(cells) == 72 and set(cells.values())
                                     == {4})
        terminals = {r["terminal"] for r in rows}
        checks["success_and_failure_present"] = terminals == {
            "TIMELY_SUCCESS", "TIMEOUT_OR_FAILURE"}
        checks["finite_states_rewards"] = all(
            math.isfinite(v) for t in transitions
            for v in (*t.state, *t.next_state, t.reward))
        distinct = {name: len({t.state[i] for t in transitions})
                    for i, name in enumerate(C.FEATURE_ORDER[:4])}
        checks["si_p40_mcs_vary"] = all(distinct[n] >= 2 for n in (
            "camera_si_scaled", "radar_p40", "prior_ul_mcs_normalized"))
        backlog = [r["pre_action_backlog_bytes"] for r in rows]
        checks["backlog_zero_and_positive"] = (min(backlog) == 0
                                               and max(backlog) > 0)
        prior_ok = transitions[0].state[C.PRIOR_SLICE] == (0.0,) * 16
        for previous, current in zip(transitions, transitions[1:]):
            expected = C.C4.build_features(
                C.ObservationV1(1.0, 0.0, 0, 0),
                C.OperationalPriorV1.from_outcome(
                    mode_id=previous.mode_id, q_e4=previous.q_e4,
                    timely=previous.diagnostics["terminal"] == "TIMELY_SUCCESS",
                    operational_latency_ns=previous.diagnostics[
                        "operational_latency_ns"]),
                self.env.scaling)[4:]
            prior_ok = prior_ok and current.state[C.PRIOR_SLICE] == expected
            prior_ok = prior_ok and previous.next_state == current.state
        checks["exact_operational_prior_propagation"] = prior_ok
        checks["duration_2_discount"] = all(
            t.discount == C.DISCOUNT for t in transitions)
        checks["no_gradient_during_collection"] = (
            self._model_sha256() == model_before
            and self._optimizer_sha256() == optimizer_before
            and self.update_count == 0)
        checks["provider_binding_registered"] = (
            self.sources.run4b.provider.binding_sha256
            == REGISTERED_PROVIDER_BINDING_SHA256)
        checks.update(K.preflight_snr_checks(self, transitions))
        rewards = [r["reward"] for r in rows]
        report = {
            "schema": "splitfusion.run5b.preflight_288.v1",
            "seed": self.seed, "decisions": len(rows), "checks": checks,
            "passed": all(checks.values()),
            "terminal_counts": {k: sum(r["terminal"] == k for r in rows)
                                for k in sorted(terminals)},
            "reward_min": min(rewards), "reward_max": max(rewards),
            "reward_mean": sum(rewards) / len(rewards),
            "distinct_values_first_four_features": distinct,
            "backlog_zero_fraction": sum(b == 0 for b in backlog) / len(backlog),
            "decision_chain": self.decision_chain,
        }
        _require(report["passed"], f"preflight failed: {checks}")
        self.preflight = report
        self.preflight_states = [t.state for t in transitions]
        return report

    # -- training --------------------------------------------------------
    def train_once(self) -> L.UpdateMetricsV1:
        batch = self.replay.sample(BATCH, self.generators["replay"])
        metrics = self.trainer.update_once(batch)
        metrics.require_finite()
        return metrics

    def policy_panel(self) -> dict[str, Any]:
        """Mean mode/family probabilities on the 20 fixtures (no RNG use)."""
        states = torch.tensor(fixture_states(self.env.scaling),
                              dtype=torch.float32)
        with torch.no_grad():
            probs = torch.softmax(self.actor(states).logits, dim=-1).mean(0)
        modes = [float(v) for v in probs]
        families = {f: sum(modes[3 * i:3 * i + 3])
                    for i, f in enumerate(("noAE", "AE128", "AE64", "AE32"))}
        return {"mode_probability_mean": modes, "family_probability_mean":
                families}

    def run_to(self, target: int, *, on_decision: Callable,
               on_update: Callable, on_checkpoint: Callable) -> None:
        _require(target in CHECKPOINT_UPDATES, "target is not registered")
        _require(target >= self.update_count, "target precedes current state")
        if self.decision_count == 0:
            self.preflight_report = self.run_preflight(on_decision)
            on_checkpoint(self)
        while self.update_count < target:
            for _ in range(TRANSITIONS_PER_UPDATE):
                row, _transition = self.collect_one()
                on_decision(row)
            metrics = self.train_once()
            row = {k: getattr(metrics, k) for k in (
                "update_index", "batch_size", "reward_mean", "target_mean",
                "discount_min", "discount_max", "critic_loss", "actor_loss",
                "critic_grad_norm", "actor_grad_norm", "actor_delta_norm",
                "critic_delta_norm", "target_delta_norm", "discrete_entropy",
                "q_executed_mean")}
            self.metrics_chain = _chain(self.metrics_chain, row)
            on_update(row)
            if self.update_count in CHECKPOINT_UPDATES:
                on_checkpoint(self)
        _require(self.decision_count == WARMUP
                 + TRANSITIONS_PER_UPDATE * self.update_count,
                 "decision/update accounting differs")


# ---------------------------------------------------------------------------
# Durable bundles
# ---------------------------------------------------------------------------
def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def bundle_name(update: int) -> str:
    return f"update_{update:06d}"


def write_bundle(runner: Run5BRunnerV1, checkpoint_dir: Path) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final = checkpoint_dir / bundle_name(runner.update_count)
    _require(not final.exists(), f"create-only bundle exists: {final}")
    staging = checkpoint_dir / f".staging_{final.name}_{os.getpid()}"
    _require(not staging.exists(), "stale staging directory")
    staging.mkdir()
    try:
        import io
        buffer = io.BytesIO()
        torch.save(runner.training_state(), buffer)
        training = buffer.getvalue()
        environment = _canonical({"environment": runner.env.state_dict(),
                                  "counters": runner.counters(),
                                  "preflight": runner.preflight})
        manifest = {
            "schema": BUNDLE_SCHEMA,
            "seed": runner.seed,
            "update": runner.update_count,
            "decision_count": runner.decision_count,
            "feature_order": list(C.FEATURE_ORDER),
            "feature_schema_sha256": C.FEATURE_SCHEMA_SHA256,
            "model_binding_sha256": M.MODEL_BINDING_SHA256,
            "operational_latency_provider_sha256":
                runner.sources.run4b.provider.binding_sha256,
            "feature_schema_id": C.FEATURE_SCHEMA["schema_id"],
            "feature_order_sha256": C.FEATURE_ORDER_SHA256,
            "registration_sha256": runner.registration_sha256,
            "joint_channel_binding_sha256": runner.sources.binding_sha256,
            "actor_tree_sha256": R4O._tree_sha256(runner.actor.state_dict()),
            "runner_binding": runner.binding,
            "runner_binding_sha256": runner.binding_sha256,
            "counters": runner.counters(),
            "state_fingerprint": runner.fingerprint(),
            "files": {"training_state.pt": {"sha256": _sha_bytes(training),
                                            "bytes": len(training)},
                      "environment_state.json": {
                          "sha256": _sha_bytes(environment),
                          "bytes": len(environment)}},
            "policy_panel": runner.policy_panel(),
        }
        manifest_bytes = _canonical(manifest)
        _write_new(staging / "training_state.pt", training)
        _write_new(staging / "environment_state.json", environment)
        _write_new(staging / "manifest.json", manifest_bytes)
        _write_new(staging / "COMMITTED",
                   (_sha_bytes(manifest_bytes) + "\n").encode("ascii"))
        _fsync_dir(staging)
        os.rename(staging, final)
        _fsync_dir(checkpoint_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    latest_tmp = checkpoint_dir / f".LATEST.{os.getpid()}.tmp"
    latest_tmp.write_text(json.dumps({
        "bundle": final.name, "update": runner.update_count,
        "manifest_sha256": _sha_bytes(manifest_bytes)}) + "\n")
    with latest_tmp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(latest_tmp, checkpoint_dir / "LATEST")
    _fsync_dir(checkpoint_dir)
    return final


def read_bundle(path: Path, runner: Run5BRunnerV1) -> dict[str, Any]:
    """Verify a bundle against the runner's exact binding; return payloads.

    Refusal order: committed marker, manifest schema, feature order, feature
    schema hash, model binding hash, provider binding, runner binding, file
    hashes; only then are tensors loaded (weights_only).
    """
    refuse = CheckpointRefused
    path = Path(path)
    _require(path.is_dir(), f"not a Run-5B bundle directory: {path}", refuse)
    _require((path / "COMMITTED").is_file(), "bundle is not COMMITTED", refuse)
    _require((path / "manifest.json").is_file(), "bundle lacks manifest.json",
             refuse)
    manifest_bytes = (path / "manifest.json").read_bytes()
    _require((path / "COMMITTED").read_text().strip()
             == _sha_bytes(manifest_bytes), "COMMITTED/manifest mismatch",
             refuse)
    manifest = json.loads(manifest_bytes)
    M.require_run5b_identity(manifest, registration_sha256=runner.registration_sha256)
    _require(manifest.get("schema") == BUNDLE_SCHEMA,
             f"foreign checkpoint schema {manifest.get('schema')!r}", refuse)
    _require(manifest.get("feature_order") == list(C.FEATURE_ORDER),
             "feature order differs from Run-5B", refuse)
    _require(manifest.get("feature_schema_sha256") == C.FEATURE_SCHEMA_SHA256,
             "feature schema hash differs", refuse)
    _require(manifest.get("model_binding_sha256") == M.MODEL_BINDING_SHA256,
             "model binding hash differs", refuse)
    _require(manifest.get("operational_latency_provider_sha256")
             == runner.sources.run4b.provider.binding_sha256,
             "operational-latency provider binding differs", refuse)
    _require(manifest.get("runner_binding_sha256") == runner.binding_sha256
             and C.canonical_sha256(manifest.get("runner_binding"))
             == runner.binding_sha256, "runner binding differs", refuse)
    _require(manifest.get("seed") == runner.seed, "seed differs", refuse)
    actual = sorted(p.name for p in path.iterdir())
    _require(actual == sorted((*BUNDLE_FILES, "COMMITTED")),
             f"bundle file set differs: {actual}", refuse)
    for name, meta in manifest["files"].items():
        data = (path / name).read_bytes()
        _require(_sha_bytes(data) == meta["sha256"]
                 and len(data) == meta["bytes"], f"{name} hash differs", refuse)
    training = torch.load(path / "training_state.pt", weights_only=True,
                          map_location="cpu")
    environment = json.loads((path / "environment_state.json").read_bytes())
    return {"manifest": manifest, "training": training,
            "environment": environment}


def restore_runner(path: Path, sources: E.JointSourcesV1,
                   seed: int) -> Run5BRunnerV1:
    runner = Run5BRunnerV1(sources, seed)
    payload = read_bundle(path, runner)
    training = payload["training"]
    runner.actor.load_state_dict(training["actor"], strict=True)
    runner.critics.load_state_dict(training["critics"], strict=True)
    runner.trainer.actor_optimizer.load_state_dict(training["actor_optimizer"])
    runner.trainer.critic_optimizer.load_state_dict(training["critic_optimizer"])
    runner.replay.load_state_dict(training["replay"])
    for name in GENERATOR_NAMES:
        runner.generators[name].set_state(training["generators"][name])
    environment = payload["environment"]
    runner.env.load_state_dict(environment["environment"])
    counters = environment["counters"]
    runner.trainer.update_count = int(counters["update_count"])
    runner.decision_count = int(counters["decision_count"])
    runner.decision_chain = counters["decision_chain"]
    runner.metrics_chain = counters["metrics_chain"]
    runner.preflight = environment["preflight"]
    M.validate_models(runner.actor, runner.critics)
    runner.trainer._assert_optimizer_wiring()
    _require(runner.fingerprint() == payload["manifest"]["state_fingerprint"],
             "restored state fingerprint differs", CheckpointRefused)
    return runner


def export_actor(runner: Run5BRunnerV1, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=False)
    import io
    buffer = io.BytesIO()
    torch.save(runner.actor.state_dict(), buffer)
    data = buffer.getvalue()
    _write_new(out_dir / "actor_state_dict.pt", data)
    states = torch.tensor(fixture_states(runner.env.scaling), dtype=torch.float32)
    with torch.no_grad():
        heads = runner.actor(states)
    manifest = {
        "schema": EXPORT_SCHEMA, "seed": runner.seed,
        "update": runner.update_count,
        "actor_state_dict_sha256": _sha_bytes(data),
        "actor_tree_sha256": R4O._tree_sha256(runner.actor.state_dict()),
        "feature_order": list(C.FEATURE_ORDER),
        "feature_schema_sha256": C.FEATURE_SCHEMA_SHA256,
        "model_binding_sha256": M.MODEL_BINDING_SHA256,
        "runner_binding_sha256": runner.binding_sha256,
        "feature_schema_id": C.FEATURE_SCHEMA["schema_id"],
        "feature_order_sha256": C.FEATURE_ORDER_SHA256,
        "registration_sha256": runner.registration_sha256,
        "joint_channel_binding_sha256": runner.sources.binding_sha256,
        "operational_latency_provider_sha256":
            runner.sources.run4b.provider.binding_sha256,
        "fixture_outputs_sha256": R4O._tree_sha256(
            {"logits": heads.logits, "mean": heads.mean,
             "log_std": heads.log_std}),
        "preregistered_live_actor": (runner.seed == LIVE_ACTOR["seed"] and
                                     runner.update_count == LIVE_ACTOR["update"]),
    }
    _write_new(out_dir / "ACTOR_EXPORT.json",
               json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n")
    return manifest


def load_actor_export(out_dir: Path) -> Any:
    refuse = CheckpointRefused
    manifest_path = Path(out_dir) / "ACTOR_EXPORT.json"
    _require(manifest_path.is_file(), "no Run-5B ACTOR_EXPORT.json", refuse)
    manifest = json.loads(manifest_path.read_text())
    M.require_run5b_identity(manifest, registration_sha256=REG.sealed_sha256())
    _require(manifest.get("schema") == EXPORT_SCHEMA, "foreign export schema",
             refuse)
    _require(manifest.get("feature_order") == list(C.FEATURE_ORDER),
             "export feature order differs", refuse)
    _require(manifest.get("feature_schema_sha256") == C.FEATURE_SCHEMA_SHA256
             and manifest.get("model_binding_sha256") == M.MODEL_BINDING_SHA256,
             "export schema/model hash differs", refuse)
    data = (Path(out_dir) / "actor_state_dict.pt").read_bytes()
    _require(_sha_bytes(data) == manifest["actor_state_dict_sha256"],
             "actor file hash differs", refuse)
    from rl_agent.splitfusion_hybrid_sac_v1.hybrid_sac_models import build_actor
    actor = build_actor(M.model_config(), seed=0)
    actor.load_state_dict(torch.load(Path(out_dir) / "actor_state_dict.pt",
                                     weights_only=True, map_location="cpu"),
                          strict=True)
    _require(R4O._tree_sha256(actor.state_dict())
             == manifest["actor_tree_sha256"], "actor tree differs", refuse)
    return actor


# ---------------------------------------------------------------------------
# Logs, dominance and CLI
# ---------------------------------------------------------------------------
class SeedLogs:
    """Append-only JSONL logs reconciled against the ledger chains."""

    def __init__(self, seed_dir: Path) -> None:
        self.decisions = seed_dir / "DECISIONS.jsonl"
        self.metrics = seed_dir / "UPDATE_METRICS.jsonl"
        self.checkpoints = seed_dir / "CHECKPOINT_EVENTS.jsonl"

    def truncate_to(self, runner: Run5BRunnerV1) -> None:
        for path, count, chain in (
                (self.decisions, runner.decision_count, runner.decision_chain),
                (self.metrics, runner.update_count, runner.metrics_chain)):
            lines = path.read_text().splitlines()[:count] if path.exists() else []
            _require(len(lines) == count, f"{path.name} shorter than checkpoint")
            digest = GENESIS_CHAIN
            for line in lines:
                digest = _chain(digest, json.loads(line))
            _require(digest == chain, f"{path.name} chain differs from bundle")
            path.write_text("".join(line + "\n" for line in lines))

    @staticmethod
    def append(path: Path, row: Mapping[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True,
                                    separators=(",", ":")) + "\n")


def family_window(seed_dir: Path, start: int, stop: int) -> dict[str, Any]:
    """Executed actor family shares for decisions [start, stop)."""
    counts = {f: 0 for f in ("noAE", "AE128", "AE64", "AE32")}
    actor_count = 0
    with (seed_dir / "DECISIONS.jsonl").open() as handle:
        for index, line in enumerate(handle):
            if index < start:
                continue
            if index >= stop:
                break
            row = json.loads(line)
            if row["source"] == "STOCHASTIC_ACTOR":
                counts[row["family"]] += 1
                actor_count += 1
    shares = {f: (c / actor_count if actor_count else 0.0)
              for f, c in counts.items()}
    under = sum(shares[f] for f in UNDER_OBSERVED_FAMILIES)
    return {"actor_decisions": actor_count, "family_counts": counts,
            "family_shares": shares, "under_observed_share": under}


def run_seed(out: Path, seed: int, target: int, *, resume: bool,
             enforce_dominance: bool) -> dict[str, Any]:
    torch.set_num_threads(THREADS)
    sources = E.load_sources(EVIDENCE_ROOT)
    seed_dir = out / f"seed_{seed}"
    checkpoint_dir = seed_dir / "checkpoints"
    logs = SeedLogs(seed_dir)
    if resume:
        latest = json.loads((checkpoint_dir / "LATEST").read_text())
        runner = restore_runner(checkpoint_dir / latest["bundle"], sources, seed)
        logs.truncate_to(runner)
    else:
        _require(not seed_dir.exists(), f"seed directory exists: {seed_dir}")
        seed_dir.mkdir(parents=True)
        runner = Run5BRunnerV1(sources, seed)
    previous = {"update": runner.update_count,
                "decisions": runner.decision_count}

    def on_checkpoint(r: Run5BRunnerV1) -> None:
        if r.update_count == 0 and r.preflight is not None:
            (seed_dir / "PREFLIGHT_288.json").write_text(
                json.dumps(r.preflight, indent=2, sort_keys=True) + "\n")
        bundle = write_bundle(r, checkpoint_dir)
        window = family_window(seed_dir, previous["decisions"], r.decision_count)
        event = {"update": r.update_count, "decision_count": r.decision_count,
                 "bundle": bundle.name,
                 "manifest_sha256": (bundle / "COMMITTED").read_text().strip(),
                 "state_fingerprint": json.loads(
                     (bundle / "manifest.json").read_text())["state_fingerprint"],
                 "window_since_update": previous["update"],
                 "window_family": window, "policy_panel": r.policy_panel()}
        SeedLogs.append(logs.checkpoints, event)
        previous.update(update=r.update_count, decisions=r.decision_count)
        if (enforce_dominance and r.update_count >= SMOKE_UPDATE
                and window["under_observed_share"] > DOMINANCE_MAX_SHARE):
            (seed_dir / "HALTED_DOMINANCE.json").write_text(
                json.dumps(event, indent=2, sort_keys=True) + "\n")
            raise DominanceStop(
                f"seed {r.seed} update {r.update_count}: noAE+AE128 share "
                f"{window['under_observed_share']:.3f} > {DOMINANCE_MAX_SHARE}")

    runner.run_to(target,
                  on_decision=lambda row: SeedLogs.append(logs.decisions, row),
                  on_update=lambda row: SeedLogs.append(logs.metrics, row),
                  on_checkpoint=on_checkpoint)
    return {"seed": seed, "update": runner.update_count,
            "decisions": runner.decision_count,
            "fingerprint": runner.fingerprint()}


def _file_sha(path: Path) -> str:
    return _sha_bytes(Path(path).read_bytes())


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def _spawn(args: list[str], log: Path) -> subprocess.Popen:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["CUDA_VISIBLE_DEVICES"] = ""
    handle = log.open("ab")
    return subprocess.Popen(
        [sys.executable, "-m", "rl_agent.splitfusion_hybrid_sac_run5b_v1.run5b_runner",
         *args], cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)


def compare_seed_dirs(reference: Path, candidate: Path, update: int
                      ) -> dict[str, Any]:
    name = bundle_name(update)
    checks = {}
    for item in ("training_state.pt", "environment_state.json", "manifest.json",
                 "COMMITTED"):
        checks[f"bundle_{item}"] = (
            _file_sha(reference / "checkpoints" / name / item)
            == _file_sha(candidate / "checkpoints" / name / item))
    for item in ("DECISIONS.jsonl", "UPDATE_METRICS.jsonl"):
        checks[f"log_{item}"] = (_file_sha(reference / item)
                                 == _file_sha(candidate / item))
    ref = json.loads((reference / "checkpoints" / name / "manifest.json"
                      ).read_text())
    cand = json.loads((candidate / "checkpoints" / name / "manifest.json"
                       ).read_text())
    checks["state_fingerprint"] = (ref["state_fingerprint"]
                                   == cand["state_fingerprint"])
    return {"update": update, "checks": checks, "passed": all(checks.values()),
            "reference_fingerprint": ref["state_fingerprint"],
            "candidate_fingerprint": cand["state_fingerprint"]}


def cmd_smoke(args) -> int:
    out = Path(args.out)
    K.require_disk(out, seeds=(17,), target=SMOKE_UPDATE)
    result = run_seed(out, 17, SMOKE_UPDATE, resume=False,
                      enforce_dominance=False)
    sources = E.load_sources(EVIDENCE_ROOT)
    seed_dir = out / "seed_17"
    reload_checks = {}
    for update in (0, 100, 250, 500):
        try:
            restored = restore_runner(seed_dir / "checkpoints" /
                                      bundle_name(update), sources, 17)
            reload_checks[str(update)] = restored.update_count == update
        except Exception as exc:  # recorded and failed below
            reload_checks[str(update)] = f"{type(exc).__name__}: {exc}"
    metrics = [json.loads(line) for line in
               (seed_dir / "UPDATE_METRICS.jsonl").read_text().splitlines()]
    finite = all(math.isfinite(float(v)) for row in metrics
                 for v in row.values())
    window = family_window(seed_dir, WARMUP, result["decisions"])
    preflight = json.loads((seed_dir / "PREFLIGHT_288.json").read_text())
    gates = {
        "preflight_288": preflight["passed"],
        "finite_update_metrics": finite and len(metrics) == SMOKE_UPDATE,
        "decision_accounting": result["decisions"]
        == WARMUP + TRANSITIONS_PER_UPDATE * SMOKE_UPDATE,
        "bundles_0_100_250_500_reload_bit_exact": all(
            v is True for v in reload_checks.values()),
        "provider_binding_registered": sources.run4b.provider.binding_sha256
        == REGISTERED_PROVIDER_BINDING_SHA256,
        "not_dominated_by_under_observed_families":
            window["under_observed_share"] <= DOMINANCE_MAX_SHARE,
    }
    final_runner = restore_runner(seed_dir / "checkpoints" / bundle_name(SMOKE_UPDATE),
                                  sources, 17)
    gates.update(K.smoke_snr_gates(seed_dir, final_runner))
    report = {"schema": "splitfusion.run5b.smoke_500_gate.v1",
              "code_commit": _git_head(), "result": result, "gates": gates,
              "passed": all(gates.values()), "reload_checks": reload_checks,
              "actor_family_occupancy_updates_1_500": window,
              "snr_diagnostics_at_500": K.snr_diagnostics(final_runner, 17)}
    (out / "SMOKE_500_GATE.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": report["passed"], "gates": gates}))
    return 0 if report["passed"] else 3


def cmd_run(args) -> int:
    try:
        result = run_seed(Path(args.out), args.seed, args.target,
                          resume=args.resume,
                          enforce_dominance=args.enforce_dominance)
    except DominanceStop as exc:
        print(f"DOMINANCE_STOP {exc}")
        return 4
    print(json.dumps(result))
    return 0


def cmd_resume_test(args) -> int:
    smoke, work = Path(args.smoke), Path(args.work)
    _require(not work.exists(), f"work directory exists: {work}")
    work.mkdir(parents=True)
    log = work / "resume_test.log"
    first = _spawn(["run", "--out", str(work), "--seed", "17", "--target",
                    str(RESUME_STOP_UPDATE)], log)
    code_first = first.wait()
    second = _spawn(["run", "--out", str(work), "--seed", "17", "--target",
                     str(SMOKE_UPDATE), "--resume"], log)
    code_second = second.wait()
    comparison = (compare_seed_dirs(smoke / "seed_17", work / "seed_17",
                                    SMOKE_UPDATE)
                  if code_first == 0 and code_second == 0 else
                  {"passed": False, "checks": {}})
    report = {"schema": "splitfusion.run5b.resume_equivalence.v1",
              "code_commit": _git_head(),
              "procedure": ("process 1: genesis -> update 250 and exit; "
                            "process 2: restore update-250 bundle -> 500; "
                            "compare with the uninterrupted smoke process"),
              "process_exit_codes": [code_first, code_second],
              "first_pid": first.pid, "second_pid": second.pid,
              **comparison}
    (work / "RESUME_EQUIVALENCE.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": report["passed"],
                      "checks": report.get("checks")}))
    return 0 if report["passed"] else 3


def cmd_campaign(args) -> int:
    torch.set_num_threads(THREADS)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    K.require_disk(out, seeds=SEEDS, target=FINAL_UPDATE)
    start = {"schema": "splitfusion.run5b.campaign.v1", "status": "RUNNING",
             "code_commit": _git_head(), "seeds": list(SEEDS),
             "target_update": FINAL_UPDATE,
             "checkpoint_updates": list(CHECKPOINT_UPDATES),
             "preregistered_live_actor": LIVE_ACTOR,
             "dominance_rule": {"families": list(UNDER_OBSERVED_FAMILIES),
                                "max_share": DOMINANCE_MAX_SHARE},
             "smoke_gate_sha256": _file_sha(Path(args.smoke_gate)),
             "resume_equivalence_sha256": _file_sha(Path(args.resume_report)),
             "started_unix": time.time()}
    for path in (args.smoke_gate, args.resume_report):
        _require(json.loads(Path(path).read_text())["passed"] is True,
                 f"gate did not pass: {path}")
    (out / "CAMPAIGN_START.json").write_text(
        json.dumps(start, indent=2, sort_keys=True) + "\n")
    processes = {seed: _spawn(["run", "--out", str(out), "--seed", str(seed),
                               "--target", str(FINAL_UPDATE)],
                              out / f"seed_{seed}.log") for seed in SEEDS}
    codes: dict[int, Optional[int]] = {seed: None for seed in SEEDS}
    halted = False
    while any(code is None for code in codes.values()):
        time.sleep(5)
        for seed, process in processes.items():
            if codes[seed] is None:
                codes[seed] = process.poll()
                if codes[seed] not in (None, 0):
                    halted = True
        if halted:
            for seed, process in processes.items():
                if codes[seed] is None:
                    process.send_signal(signal.SIGTERM)
                    codes[seed] = process.wait()
    if halted:
        final = {**start, "status": "HALTED", "exit_codes": codes,
                 "finished_unix": time.time()}
        (out / "CAMPAIGN_HALTED.json").write_text(
            json.dumps(final, indent=2, sort_keys=True) + "\n")
        print("HALTED")
        return 4
    return finalize_campaign(out, exit_codes=codes)


def finalize_campaign(out: Path, *, exit_codes: Optional[Mapping] = None,
                      note: Optional[str] = None) -> int:
    """Independently verify every seed at update 10,000, then export actors."""
    torch.set_num_threads(THREADS)
    start = json.loads((out / "CAMPAIGN_START.json").read_text())
    sources = E.load_sources(EVIDENCE_ROOT)
    verification, exports = {}, {}
    for seed in SEEDS:
        seed_dir = out / f"seed_{seed}"
        checkpoints = sorted(p.name for p in (seed_dir / "checkpoints").iterdir()
                             if p.name.startswith("update_"))
        final_line = json.loads(
            (out / f"seed_{seed}.log").read_text().splitlines()[-1])
        runner = restore_runner(seed_dir / "checkpoints" /
                                bundle_name(FINAL_UPDATE), sources, seed)
        logs = SeedLogs(seed_dir)
        logs.truncate_to(runner)  # verifies both ledger chains; no-op length
        checks = {
            "all_registered_bundles": checkpoints == [
                bundle_name(u) for u in CHECKPOINT_UPDATES],
            "no_dominance_halt": not (seed_dir / "HALTED_DOMINANCE.json").exists(),
            "process_final_line_update": final_line.get("update") == FINAL_UPDATE,
            "restored_fingerprint_equals_process": runner.fingerprint()
            == final_line.get("fingerprint"),
            "decision_accounting": runner.decision_count
            == WARMUP + TRANSITIONS_PER_UPDATE * FINAL_UPDATE,
        }
        _require(all(checks.values()), f"seed {seed} verification: {checks}")
        verification[str(seed)] = checks
        exports[str(seed)] = export_actor(runner, seed_dir / "final_actor")
    fresh = K.fresh_process_verify(out)
    final = {**start, "status": "COMPLETE", "exit_codes": exit_codes,
             "fresh_process_actor_verification": fresh,
             "finished_unix": time.time(), "seed_verification": verification,
             "actor_exports": exports, "finalization_note": note}
    _write_new(out / "CAMPAIGN_COMPLETE.json",
               json.dumps(final, indent=2, sort_keys=True).encode() + b"\n")
    print("COMPLETE")
    return 0


def cmd_finalize(args) -> int:
    return finalize_campaign(Path(args.out), note=args.note)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--out", required=True)
    run = sub.add_parser("run")
    run.add_argument("--out", required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--target", type=int, required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--enforce-dominance", action="store_true")
    resume = sub.add_parser("resume-test")
    resume.add_argument("--smoke", required=True)
    resume.add_argument("--work", required=True)
    campaign = sub.add_parser("campaign")
    campaign.add_argument("--out", required=True)
    campaign.add_argument("--smoke-gate", required=True)
    campaign.add_argument("--resume-report", required=True)
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--out", required=True)
    finalize.add_argument("--note", default=None)
    args = parser.parse_args(argv)
    return {"smoke": cmd_smoke, "run": cmd_run, "resume-test": cmd_resume_test,
            "campaign": cmd_campaign,
            "finalize": cmd_finalize}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
