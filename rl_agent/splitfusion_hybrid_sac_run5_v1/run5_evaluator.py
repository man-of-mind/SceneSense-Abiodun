#!/usr/bin/env python3
"""Registered Run-5 post-training evaluator (diagnostic only; never trains).

Contexts
--------
One trajectory per (network profile, validation seed): 4 profiles x seeds
9017/9029/9043 = 12 contexts.  A trajectory visits the 255 sealed eligible
``held_scene`` scenes once each, in route order, one policy decision per scene
occurrence.  Previous action/outcome, backlog and channel evolve across the
whole trajectory; the environment is reset only at the start of a context.

Actor-independent tapes
-----------------------
Every policy in a context receives the identical scene order, held-tensor
scene, retained-residual draw, transport success draw and joint SNR/MCS
channel (fixed profile; ``derive(validation_seed, 'validation-channel')`` as
preregistered).  None of those streams is consumed differently by different
actions.  Backlog and previous action/outcome are per-policy consequences.
The tape digest of each trajectory is recorded and must be identical across
every policy of the context.

Policies (deterministic evaluation rules)
-----------------------------------------
* Run 5, seeds 17/29/43 x checkpoints 500/1500/2500/5000/7500/10000, true
  causal SNR - argmax mode at its conditional mean (``deterministic_execution``);
  update 10,000 is the only registered final actor, other checkpoints are
  learning-progress diagnostics;
* the same 18 actors with an audit-only, prospectively fixed, in-support
  shuffled SNR input (a fixed permutation of the context's own exogenous SNR
  tape); the environment still uses the true channel;
* the frozen native 21-D Run-4 seed-43/update-10,000 actor through its own
  registered ``act_on_vector`` rule on features 0-20;
* one fixed catalogue action selected from FIT evidence only
  (``FIXED_ACTION_SELECTION.json``).

Per decision the evaluator also scores every registered catalogue anchor with
the exact Run-4 kernel composition (expected and realized immediate reward on
the same tape) to form the catalogue-oracle regret.  Anchors outside the
fitted transport support are refused and counted, never extrapolated.

Timeouts and registered failures stay in every unconditional metric.  Any
evaluator/infrastructure exception is recorded as a fault row and reported
separately; nothing is silently dropped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import torch

from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import frozen_actor_v2 as FA
from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as orch
from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as R4
from rl_agent.splitfusion_hybrid_sac_v1 import action_contract as AC
from rl_agent.splitfusion_hybrid_sac_v1.hybrid_sac_models import build_actor
from rl_agent.ue_production_transport_model_v2 import artifact_v2 as A2
from rl_agent.ue_production_transport_model_v2 import collector_v1 as CV

from . import run5_bundle as B
from . import run5_channel as J
from . import run5_collector as RC
from . import run5_held_scene_partition as H
from . import run5_models as RM
from . import run5_preregistration as PR
from . import run5_snr_v2 as SNR
from . import run5_training as RT

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
CAMPAIGN = PACKAGE / "campaign_runs" / "run5_three_seed_10000_v1"
ARTIFACT = ROOT / ("rl_agent/experiments/ue_production_queue_capture_v1/20260929_model_v2b/"
                   "transport_model_v2.json")
MANIFEST_PATH = PACKAGE / "RUN5_EVALUATION_MANIFEST.json"
FIXED_ACTION_PATH = PACKAGE / "FIXED_ACTION_SELECTION.json"
SCHEMA = "splitfusion.run5.heldscene_evaluation.v1"
PREREGISTRATION_SHA256 = "2270baa0cf025b5e64a85644a28b8fd98e1f9004b11dcd6cb39fb5f61c5d18cf"
RUN4_ACTOR_WEIGHTS_SHA256 = "d064013d011b67dcd2c7c23acc3c396afe6750be0d43ef0204f2fbecbb9b8e29"
PROFILES = J.TRAINING_PROFILES
VALIDATION_SEEDS = PR.CONFIG.validation_seeds
TRAINING_SEEDS = PR.CONFIG.seed_order
EVAL_CHECKPOINTS = PR.CONFIG.evaluation_checkpoints
FINAL_UPDATE = PR.CONFIG.deep_target_update
FIXED_SELECTION_SEED = 7017
FIXED_SELECTION_DECISIONS = 1000
SWEEP_WORKERS = 12
SWEEP_OUTPUT_RELPATH = "rl_agent/splitfusion_hybrid_sac_run5_v1/evaluation_runs/heldscene_eval_v1"
SWEEP_COMMAND = ("CUDA_VISIBLE_DEVICES='' python3 -m rl_agent.splitfusion_hybrid_sac_run5_v1."
                 "run5_evaluator --evidence-root <main checkout> --run --workers 12 "
                 f"--output-dir {SWEEP_OUTPUT_RELPATH}")
DEADLINE_NS = R4.REWARD_DEADLINE_NS


class EvaluatorError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluatorError(message)


def _sha(value: Any) -> str:
    return hashlib.sha256(B.canonical_bytes(value)).hexdigest()


def contexts() -> list[tuple[str, int]]:
    return [(profile, seed) for profile in PROFILES for seed in VALIDATION_SEEDS]


# ---------------------------------------------------------------------------
# Exogenous channel and evaluation collector
# ---------------------------------------------------------------------------


class FixedProfileChannelV1(J.JointSnrMcsChannelV1):
    """The registered joint channel held on one profile for a whole trajectory."""

    def __init__(self, *, kernel, design, seed: int, profile: str) -> None:
        require(profile in PROFILES, f"unknown profile {profile}")
        self._fixed = PROFILES.index(profile)
        super().__init__(kernel=kernel, design=design, seed=seed)
        self.binding_sha256 = _sha({"base": self.binding_sha256, "fixed_profile": profile})

    def _next_block_profile(self) -> int:
        self._segments_started += 1
        self._profile_segments[self._fixed] += 1
        return self._fixed


def validation_channel(shared, profile: str, validation_seed: int) -> FixedProfileChannelV1:
    return FixedProfileChannelV1(kernel=shared["snr_kernel"], design=shared["design"],
                                 seed=J.derive_seed(validation_seed, "validation-channel"),
                                 profile=profile)


def snr_tape(shared, profile: str, validation_seed: int, decisions: int) -> list[float]:
    """The exogenous SNR sequence the collector will observe (channel-only replay)."""
    channel = validation_channel(shared, profile, validation_seed)
    values = []
    for _ in range(decisions):
        values.append(channel.observe().snr_db)
        channel.advance(J.SUPPORTED_DURATION_TENSORS)
    return values


def shuffle_permutation(profile: str, validation_seed: int, decisions: int) -> list[int]:
    order = list(range(decisions))
    random.Random(J.derive_seed(validation_seed, f"snr-shuffle:{profile}")).shuffle(order)
    return order


class Run5EvaluationCollectorV1(RC.Run5ModeledCollectorV1):
    """Training collector machinery on sealed held scenes and exogenous tapes.

    Feature scaling (camera centre/scale, backlog) stays the training scaling:
    the parent computes it from the FIT catalogue before the held catalogue is
    swapped in, so no validation statistic enters the state.
    """

    def __init__(self, *, shared_sources, held_catalog: H.HeldSceneCatalogV1,
                 profile: str, validation_seed: int, evidence_root: Path) -> None:
        require(validation_seed in VALIDATION_SEEDS, "not a registered validation seed")
        self._held_catalog = held_catalog
        self._profile = profile
        self._validation_seed = validation_seed
        super().__init__(artifact_path=ARTIFACT, seed=validation_seed,
                         evidence_root=evidence_root, shared_sources=shared_sources)

    def _start_session(self) -> None:
        self.catalog = self._held_catalog
        label = f"{self._profile}:{self._validation_seed}"
        self._residual_rng = random.Random(J.derive_seed(self._validation_seed,
                                                         f"eval-residual:{label}"))
        self._transport_rng = random.Random(J.derive_seed(self._validation_seed,
                                                          f"eval-transport:{label}"))
        self._scene_index = 0
        self._tape: list[dict[str, Any]] = []
        self._channel = validation_channel(self._shared, self._profile, self._validation_seed)
        self._snr_adapter = SNR.ModeledLeaseSnrAdapterV1(
            provider_id=RC.SCHEMA, session_uuid=self._session_uuid(), ue_id=CV.UE_ID)
        self._take_channel_observation()
        self._backlog_bytes = 0
        self._history = []
        self._actions = []
        self._diagnostics = []
        self._feature_cache = {}
        self.previous_context = None
        self.context = self._new_context()
        self._env = self._build_env()
        self._env.reset(session_uuid=self._session_uuid(), ue_id=CV.UE_ID)

    def _new_context(self):
        keys = self._held_catalog.keys
        last = len(keys) - 1
        index = self._scene_index
        self._scene_index += 1
        reward_key, held_key = keys[min(index, last)], keys[min(index + 1, last)]
        camera_si, radar_p40 = self._held_catalog.scene_descriptors(reward_key)
        context = CV._DecisionContext(
            reward_scene_key=reward_key, held_scene_key=held_key, camera_si=camera_si,
            radar_p40=radar_p40, prior_ul_mcs=int(self._mcs_current),
            backlog_bytes=int(self._backlog_bytes),
            retained_residual_ns=self._residual_rng.choice(self.retained_residuals),
            success_draw=self._transport_rng.random())
        context.snr_db = float(self._snr_current)
        self._tape.append({"scene": reward_key, "held": held_key,
                           "residual_ns": context.retained_residual_ns,
                           "success_draw": context.success_draw.hex(),
                           "snr_db": context.snr_db.hex(), "mcs": context.prior_ul_mcs})
        return context

    def exogenous_tape(self, decisions: int) -> list[dict[str, Any]]:
        """Actor-independent fields only (MCS is channel-driven, not action-driven)."""
        return self._tape[:decisions]


# ---------------------------------------------------------------------------
# Kernel-exact immediate reward for any action on the current context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionScore:
    in_support: bool
    expected: Optional[float]
    realized: Optional[float]
    q_perc: Optional[float]


def score_action(collector: Run5EvaluationCollectorV1, mode_id: int, q_e4: int) -> ActionScore:
    """Mirror of ``collector_v1._RealKernel.execute_cycle`` without stepping."""
    context = collector.context
    draw = collector.catalog.draw(context.reward_scene_key, mode_id=mode_id, q_e4=q_e4)
    try:
        prediction = collector.model.predict(
            pre_enqueue_backlog_bytes=float(context.backlog_bytes),
            wire_bytes=int(draw.wire_bytes), prior_ul_mcs=int(context.prior_ul_mcs))
    except A2.OutOfSupport:
        return ActionScore(False, None, None, float(draw.q_perc))
    send_span_ns = int(round(CV.SEND_SPAN_NS_PER_BYTE * draw.wire_bytes))
    transport_ns = int(round(prediction.conditional_latency_ms * 1e6))
    composed_ns = (context.retained_residual_ns + send_span_ns + transport_ns
                   + collector.actor_reserve_ns)
    if composed_ns <= DEADLINE_NS:
        latency_ms = composed_ns / 1_000_000.0
        success_reward = float(draw.q_perc) - R4.REWARD_LATENCY_WEIGHT * (
            latency_ms / R4.REWARD_DEADLINE_MS)
    else:
        success_reward = R4.REGISTERED_FAILURE_REWARD
    p = prediction.on_time_probability
    expected = p * success_reward + (1.0 - p) * R4.REGISTERED_FAILURE_REWARD
    realized = success_reward if context.success_draw < p else R4.REGISTERED_FAILURE_REWARD
    return ActionScore(True, expected, realized, float(draw.q_perc))


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


@dataclass
class PolicySpec:
    policy_id: str
    family: str                  # RUN5 | RUN5_SHUFFLED_SNR | RUN4 | FIXED_ACTION
    training_seed: Optional[int] = None
    update: Optional[int] = None
    registered_final: bool = False
    actor_tree_sha256: Optional[str] = None
    actor_file_sha256: Optional[str] = None
    mode_id: Optional[int] = None
    q_e4: Optional[int] = None


def policy_specs(evidence: dict, fixed: dict) -> list[PolicySpec]:
    specs = []
    for seed in TRAINING_SEEDS:
        for update in EVAL_CHECKPOINTS:
            manifest = evidence["seeds"][str(seed)]["checkpoints"][str(update)]["manifest"]
            for family in ("RUN5", "RUN5_SHUFFLED_SNR"):
                specs.append(PolicySpec(
                    policy_id=f"{family.lower()}_seed{seed}_u{update:05d}", family=family,
                    training_seed=seed, update=update, registered_final=update == FINAL_UPDATE,
                    actor_tree_sha256=manifest["actor_tree_sha256"],
                    actor_file_sha256=manifest["files"]["actor_state_dict.pt"]["sha256"]))
    specs.append(PolicySpec("run4_seed43_u10000", "RUN4", 43, 10_000, True,
                            actor_file_sha256=RUN4_ACTOR_WEIGHTS_SHA256))
    specs.append(PolicySpec(f"fixed_mode{fixed['mode_id']}_q{fixed['q_e4']}", "FIXED_ACTION",
                            mode_id=fixed["mode_id"], q_e4=fixed["q_e4"]))
    return specs


def load_run5_actor(spec: PolicySpec):
    path = (CAMPAIGN / f"seed_{spec.training_seed}" / "checkpoints"
            / B.bundle_name("checkpoint", spec.update))
    bundle = B.verify_bundle(path)
    require(bundle.manifest["actor_tree_sha256"] == spec.actor_tree_sha256, "actor tree differs")
    data = bundle.payload("actor_state_dict.pt")
    require(B.sha256_bytes(data) == spec.actor_file_sha256, "actor file differs")
    actor = build_actor(RM.run5_model_config(), seed=0)
    actor.load_state_dict(B.torch_from_bytes(data), strict=True)
    actor.eval()
    actor.requires_grad_(False)
    require(RT._tree_sha256(actor.state_dict()) == spec.actor_tree_sha256, "loaded tree differs")
    require(RT.actor_fixture_outputs(actor) == bundle.manifest["actor_fixtures"],
            "actor probe outputs differ")
    return actor


def make_policy(spec: PolicySpec, evidence_root: Path, shuffled_values: Optional[list[float]]):
    if spec.family in ("RUN5", "RUN5_SHUFFLED_SNR"):
        actor = load_run5_actor(spec)
        lower, upper = actor.active_q_e4_bounds()

        def act(k: int, state: tuple[float, ...]) -> tuple[int, int]:
            values = list(state)
            if spec.family == "RUN5_SHUFFLED_SNR":
                values[RT.SNR_INDEX] = SNR.scale_snr_db(shuffled_values[k])
            with torch.inference_mode():
                execution = actor.deterministic_execution(
                    torch.tensor((tuple(values),), dtype=torch.float32))
            mode, q = int(execution.mode_index[0]), int(execution.q_e4[0])
            require(int(lower[mode]) <= q <= int(upper[mode]), "q escaped the mode support")
            return mode, q
        return act, actor
    if spec.family == "RUN4":
        frozen = FA.load_registered_actor(Path(evidence_root))
        require(FA.sha256_file(Path(evidence_root) / FA.ACTOR_EXPORT_RELPATH
                               / "actor_state_dict.pt") == RUN4_ACTOR_WEIGHTS_SHA256,
                "Run-4 comparator weights differ")

        def act(k: int, state: tuple[float, ...]) -> tuple[int, int]:
            decision = frozen.act_on_vector(tuple(float(v) for v in state[:21]))
            return decision.mode_id, decision.q_e4
        return act, None
    mode, q = spec.mode_id, spec.q_e4
    return (lambda k, state: (mode, q)), None


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------


def anchors() -> list[tuple[int, int]]:
    return [(a.mode.mode_id, a.q_e4) for a in AC.load_contract().anchors]


def run_trajectory(*, spec: PolicySpec, act: Callable, collector: Run5EvaluationCollectorV1,
                   decisions: int, catalogue: Sequence[tuple[int, int]],
                   reference_actor=None, shuffled_values=None) -> tuple[list[dict], list[dict]]:
    rows = []
    for k in range(decisions):
        state = collector.current_state_features()
        try:
            mode_id, q_e4 = act(k, state)
            own = score_action(collector, mode_id, q_e4)
            scores = [score_action(collector, m, q) for m, q in catalogue]
            supported = [s for s in scores if s.in_support]
            oracle_expected = max(s.expected for s in supported)
            oracle_realized = max(s.realized for s in supported)
            disagreement = None
            if reference_actor is not None:        # one-step shuffled-SNR probe
                shuffled = list(state)
                shuffled[RT.SNR_INDEX] = SNR.scale_snr_db(shuffled_values[k])
                with torch.inference_mode():
                    execution = reference_actor.deterministic_execution(
                        torch.tensor((tuple(shuffled),), dtype=torch.float32))
                disagreement = {"mode": int(execution.mode_index[0]),
                                "q_e4": int(execution.q_e4[0])}
            record = collector.collect(orch.ModeledActionRequestV1(
                decision_ordinal=k, mode_id=mode_id, q_e4=q_e4,
                source="STOCHASTIC_ACTOR",          # collector label for any non-warm-up
                warmup_q_bin_index=None))           # action; policy identity is below
        except Exception as exc:  # noqa: BLE001 - recorded, reported, never dropped
            rows.append({"k": k, "policy_id": spec.policy_id, "fault": True,
                         "fault_kind": "EVALUATOR_FAULT", "fault_reason": repr(exc)[:300]})
            break
        diag = collector.diagnostics()[-1]
        require(own.in_support and own.realized == record.reward,
                "kernel mirror disagrees with the executed reward")
        rows.append({
            "k": k, "policy_id": spec.policy_id, "fault": False,
            "mode_id": mode_id, "q_e4": q_e4, "reward": record.reward,
            "terminal": record.terminal, "latency_ms": record.latency_ms,
            "delivered_q_perc": record.q_perc, "executed_q_perc": own.q_perc,
            "expected_reward": own.expected, "oracle_expected_reward": oracle_expected,
            "oracle_realized_reward": oracle_realized,
            "oracle_refused_anchors": len(scores) - len(supported),
            "backlog_bytes": diag["pre_enqueue_backlog_bytes"], "mcs": diag["prior_ul_mcs"],
            "snr_db": diag["snr_db"], "scene": diag["reward_scene_key"],
            "shuffled_probe": disagreement})
    return rows, collector.exogenous_tape(len([r for r in rows if not r["fault"]]))


def evaluate_context(args: tuple) -> dict[str, Any]:
    profile, validation_seed, evidence_root, output_dir, specs_json, partition_keys = args
    torch.set_num_threads(1)
    evidence_root = Path(evidence_root)
    shared = RC.build_shared_sources(ARTIFACT, evidence_root)
    held = H.HeldSceneCatalogV1(evidence_root, partition_keys)
    decisions = held.scene_count
    catalogue = anchors()
    tape = snr_tape(shared, profile, validation_seed, decisions)
    permutation = shuffle_permutation(profile, validation_seed, decisions)
    shuffled_values = [tape[i] for i in permutation]
    specs = [PolicySpec(**s) for s in specs_json]
    true_actors = {}
    out, tapes = [], {}
    started = time.time()
    for spec in specs:
        act, actor = make_policy(spec, evidence_root, shuffled_values)
        if spec.family == "RUN5":
            true_actors[spec.policy_id] = actor
        collector = Run5EvaluationCollectorV1(
            shared_sources=shared, held_catalog=held, profile=profile,
            validation_seed=validation_seed, evidence_root=evidence_root)
        rows, exogenous = run_trajectory(
            spec=spec, act=act, collector=collector, decisions=decisions, catalogue=catalogue,
            reference_actor=actor if spec.family == "RUN5" else None,
            shuffled_values=shuffled_values)
        for row in rows:
            row.update(profile=profile, validation_seed=validation_seed)
        require([float.fromhex(t["snr_db"]) for t in exogenous] == tape[:len(exogenous)],
                "collector SNR differs from the exogenous tape")
        tapes[spec.policy_id] = _sha(exogenous)
        out.extend(rows)
    require(len(set(tapes.values())) == 1, "policies in one context saw different tapes")
    path = Path(output_dir) / "rows" / f"{profile}__{validation_seed}.jsonl"
    B.atomic_write_file(path, b"".join(B.canonical_bytes(r) + b"\n" for r in out))
    return {"profile": profile, "validation_seed": validation_seed, "rows": len(out),
            "faults": sum(r["fault"] for r in out), "tape_sha256": next(iter(tapes.values())),
            "snr_tape_sha256": _sha([v.hex() for v in tape]),
            "shuffle_permutation_sha256": _sha(permutation),
            "rows_file_sha256": B.sha256_bytes(path.read_bytes()),
            "seconds": time.time() - started}


# ---------------------------------------------------------------------------
# Metrics (every seed separately; mean/range across seeds; no selection)
# ---------------------------------------------------------------------------


def _pct(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))]


def trajectory_metrics(rows: list[dict]) -> dict[str, Any]:
    ok = [r for r in rows if not r["fault"]]
    n = len(ok)
    success = [r for r in ok if r["terminal"] == "SUCCESS"]
    latency = [r["latency_ms"] for r in success]
    q = [r["q_e4"] for r in ok]
    return {
        "decisions": n, "faults": len(rows) - n,
        "unconditional_reward_mean": statistics.fmean(r["reward"] for r in ok) if n else None,
        "success_rate": len(success) / n if n else None,
        "timeout_rate": sum(r["terminal"] == "TIMEOUT" for r in ok) / n if n else None,
        "registered_delivery_failure_rate":
            sum(r["terminal"] == "REGISTERED_DELIVERY_FAILURE" for r in ok) / n if n else None,
        "latency_ms_p50": _pct(latency, 0.5), "latency_ms_p95": _pct(latency, 0.95),
        "latency_ms_p99": _pct(latency, 0.99), "latency_censored_count": n - len(success),
        "eligible_gt_executed_q_perc_mean": statistics.fmean(r["executed_q_perc"] for r in ok)
        if n else None,
        "eligible_gt_delivered_q_perc_mean": statistics.fmean(r["delivered_q_perc"]
                                                              for r in success) if success else None,
        "mode_counts": [sum(r["mode_id"] == m for r in ok) for m in range(12)],
        "q_e4_mean": statistics.fmean(q) if q else None,
        "q_e4_pstdev": statistics.pstdev(q) if len(q) > 1 else None,
        "q_e4_distinct": len(set(q)),
        "q_e4_histogram_1000": [sum(1000 * b <= v < 1000 * (b + 1) or (b == 9 and v >= 9000)
                                    for v in q) for b in range(10)],
        "oracle_expected_regret_mean": statistics.fmean(
            r["oracle_expected_reward"] - r["expected_reward"] for r in ok) if n else None,
        "oracle_realized_regret_mean": statistics.fmean(
            r["oracle_realized_reward"] - r["reward"] for r in ok) if n else None,
        "oracle_refused_anchor_mean": statistics.fmean(r["oracle_refused_anchors"] for r in ok)
        if n else None,
    }


def summarize(rows: list[dict], specs: list[PolicySpec]) -> dict[str, Any]:
    by = {}
    for row in rows:
        by.setdefault((row["policy_id"], row["profile"], row["validation_seed"]), []).append(row)
    per_policy_context = {f"{p}|{prof}|{v}": trajectory_metrics(r)
                          for (p, prof, v), r in sorted(by.items())}
    per_policy = {s.policy_id: trajectory_metrics([r for r in rows if r["policy_id"] == s.policy_id])
                  for s in specs}
    keys = ("unconditional_reward_mean", "success_rate", "latency_ms_p95",
            "eligible_gt_executed_q_perc_mean", "oracle_expected_regret_mean")
    fixed = next(s.policy_id for s in specs if s.family == "FIXED_ACTION")
    paired = {}
    for spec in specs:
        if spec.family != "RUN5":
            continue
        entry = {}
        for comparator in ("run4_seed43_u10000", fixed):
            diffs = {k: [] for k in keys}
            for profile, seed in contexts():
                a = per_policy_context.get(f"{spec.policy_id}|{profile}|{seed}")
                b = per_policy_context.get(f"{comparator}|{profile}|{seed}")
                for k in keys:
                    if a and b and a[k] is not None and b[k] is not None:
                        diffs[k].append(a[k] - b[k])
            entry[comparator] = {k: {"mean": statistics.fmean(v) if v else None,
                                     "min": min(v) if v else None, "max": max(v) if v else None,
                                     "contexts": len(v)} for k, v in diffs.items()}
        shuffled_id = spec.policy_id.replace("run5_", "run5_shuffled_snr_")
        own = [r for r in rows if r["policy_id"] == spec.policy_id and not r["fault"]]
        probes = [r for r in own if r["shuffled_probe"] is not None]
        entry["shuffled_snr"] = {
            "one_step_mode_disagreement_rate": statistics.fmean(
                r["shuffled_probe"]["mode"] != r["mode_id"] for r in probes) if probes else None,
            "one_step_abs_q_e4_change_mean": statistics.fmean(
                abs(r["shuffled_probe"]["q_e4"] - r["q_e4"]) for r in probes) if probes else None,
            "closed_loop_metric_change": {k: (None if per_policy[shuffled_id][k] is None
                                              or per_policy[spec.policy_id][k] is None else
                                              per_policy[shuffled_id][k] - per_policy[spec.policy_id][k])
                                          for k in keys}}
        paired[spec.policy_id] = entry
    across_seeds = {}
    for update in EVAL_CHECKPOINTS:
        ids = [f"run5_seed{s}_u{update:05d}" for s in TRAINING_SEEDS]
        entry = {}
        for k in keys:
            values = [per_policy[i][k] for i in ids if per_policy[i][k] is not None]
            entry[k] = {"per_seed": {i: per_policy[i][k] for i in ids},
                        "mean": statistics.fmean(values) if values else None,
                        "min": min(values) if values else None,
                        "max": max(values) if values else None}
        across_seeds[str(update)] = entry
    return {"per_policy": per_policy, "per_policy_context": per_policy_context,
            "paired": paired, "run5_across_seeds": across_seeds,
            "registered_final_actor_update": FINAL_UPDATE,
            "selection": "none: every seed and checkpoint reported; update 10,000 is the only "
                         "registered final actor",
            "faults_total": sum(r["fault"] for r in rows)}


# ---------------------------------------------------------------------------
# FIT-only fixed-action selection
# ---------------------------------------------------------------------------


def select_fixed_action(evidence_root: Path) -> dict[str, Any]:
    """Highest mean realized reward on a FIT-scene training-style rollout.

    Candidates: catalogue anchors whose payload is inside the fitted transport
    support on every FIT scene.  Each runs FIXED_SELECTION_DECISIONS decisions
    in the Run-5 training collector (FIT catalogue, four balanced profiles) with
    the dedicated seed 7017.  No held scene is read.  A candidate whose rollout
    drives backlog or payload outside the fitted transport support cannot be
    scored by the model and is recorded ineligible (never extrapolated).
    Ties -> lower action_id.
    """
    torch.set_num_threads(1)
    shared = RC.build_shared_sources(ARTIFACT, Path(evidence_root))
    catalog, model = shared["catalog"], shared["model"]
    candidates = []
    for anchor in AC.load_contract().anchors:
        wires = [catalog.draw(k, mode_id=anchor.mode.mode_id, q_e4=anchor.q_e4).wire_bytes
                 for k in catalog.keys]
        if min(wires) >= model._min_wire and max(wires) <= model._max_wire:
            candidates.append(anchor)
    results = []
    for anchor in candidates:
        collector = RC.Run5ModeledCollectorV1(artifact_path=ARTIFACT, seed=FIXED_SELECTION_SEED,
                                              evidence_root=Path(evidence_root),
                                              shared_sources=shared)
        rewards, left_support = [], None
        for k in range(FIXED_SELECTION_DECISIONS):
            try:
                rewards.append(collector.collect(orch.ModeledActionRequestV1(
                    decision_ordinal=k, mode_id=anchor.mode.mode_id, q_e4=anchor.q_e4,
                    source="STOCHASTIC_ACTOR", warmup_q_bin_index=None)).reward)
            except A2.OutOfSupport as exc:     # queue saturated past the fitted support
                left_support = {"decision": k, "reason": str(exc)[:200]}
                break
        results.append({"action_id": anchor.action_id, "mode_id": anchor.mode.mode_id,
                        "q_e4": anchor.q_e4, "profile_id": anchor.profile_id,
                        "eligible": left_support is None,
                        "left_transport_support": left_support,
                        "decisions": len(rewards),
                        "mean_reward": statistics.fmean(rewards) if rewards else None})
    eligible = [r for r in results if r["eligible"]]
    require(bool(eligible), "no fixed-action candidate stays inside the transport support")
    best = max(eligible, key=lambda r: (r["mean_reward"], -r["action_id"]))
    return {"schema": "splitfusion.run5.fixed_action_selection.v1",
            "evidence": "FIT scenes only; seed 7017; 1000 decisions per candidate",
            "candidates": len(results), "eligible_candidates": len(eligible),
            "results": results, "selected": best,
            "mode_id": best["mode_id"], "q_e4": best["q_e4"]}


# ---------------------------------------------------------------------------
# Manifest and CLI
# ---------------------------------------------------------------------------

EVALUATOR_SOURCES = ("run5_evaluator.py", "run5_held_scene_partition.py", "run5_collector.py",
                     "run5_channel.py", "run5_snr_v2.py", "run5_models.py", "run5_training.py",
                     "run5_bundle.py", "successor_mcs_snr_audit.py", "run5_state_contract.py",
                     "run5_preregistration.py")
REUSED_SOURCES = ("rl_agent/ue_production_transport_model_v2/collector_v1.py",
                  "rl_agent/ue_production_transport_model_v2/scene_source.py",
                  "rl_agent/ue_production_transport_model_v2/artifact_v2.py",
                  "rl_agent/splitfusion_hybrid_sac_live_route_b_v2/frozen_actor_v2.py",
                  "rl_agent/splitfusion_hybrid_sac_live_route_b_v2/ACTOR_BINDING_V2.json",
                  "rl_agent/splitfusion_hybrid_sac_run4_v1/environment.py",
                  "rl_agent/splitfusion_hybrid_sac_run4_v1/run4_contract.py",
                  "rl_agent/splitfusion_hybrid_sac_v1/hybrid_sac_models.py",
                  "rl_agent/splitfusion_hybrid_sac_v1/action_contract.py")


def manifest_document(evidence_root: Path) -> dict[str, Any]:
    prereg = PR.load_sealed()
    require(prereg["sha256"] == PREREGISTRATION_SHA256, "preregistration differs")
    partition = H.load_sealed_partition(Path(evidence_root))
    evidence = json.loads((PACKAGE / "RUN5_DEEP_CAMPAIGN_EVIDENCE.json").read_text())
    fixed = json.loads(FIXED_ACTION_PATH.read_text())
    specs = policy_specs(evidence, fixed)
    decisions = partition["eligible_count"]
    per_context = len(specs) * decisions
    return {
        "schema": SCHEMA, "prospective": True, "results_inspected_before_seal": False,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "campaign_evidence_sha256": B.sha256_bytes(
            (PACKAGE / "RUN5_DEEP_CAMPAIGN_EVIDENCE.json").read_bytes()),
        "campaign_complete_sha256": evidence["campaign_complete_sha256"],
        "held_scene_partition": {
            "file_sha256": B.sha256_bytes(H.SEALED_PARTITION.read_bytes()),
            "partition_sha256": H.partition_sha256(partition),
            "eligible_keys_sha256": partition["eligible_keys_sha256"],
            "candidates": partition["candidate_count"], "eligible": decisions,
            "excluded": partition["excluded"]},
        "replaces": ("the 85-scene fit_validation partition 44dad342... is inside the training "
                     "catalogue and is not used for the registered result"),
        "contexts": [{"profile": p, "validation_seed": s} for p, s in contexts()],
        "policies": [s.__dict__ for s in specs],
        "fixed_action_selection_sha256": B.sha256_bytes(FIXED_ACTION_PATH.read_bytes()),
        "run4_actor_weights_sha256": RUN4_ACTOR_WEIGHTS_SHA256,
        "catalogue_sha256": AC.load_contract().catalog_sha256,
        "catalogue_anchor_count": len(anchors()),
        "semantics": {
            "trajectory": ("one per (profile, validation seed); 255 held scenes in route order, "
                           "one decision per scene; reset only at trajectory start"),
            "tapes": ("identical scene order, held-tensor scene, residual draw, success draw and "
                      "fixed-profile joint channel for every policy; backlog and previous "
                      "action/outcome per policy"),
            "channel_seed": "derive(validation_seed, 'validation-channel'), profile held fixed",
            "action_rule": "deterministic evaluation (argmax mode, conditional mean q)",
            "run4_input": "features 0-20 through frozen_actor_v2.act_on_vector",
            "shuffled_snr": ("audit-only actor input: the context's own exogenous SNR tape "
                             "permuted by Random(derive(validation_seed, 'snr-shuffle:<profile>')); "
                             "in support by construction; environment uses the true channel"),
            "oracle": ("per decision, max over in-support catalogue anchors of the kernel-exact "
                       "expected (primary) and realized (secondary) immediate reward on the same "
                       "tape; refused anchors counted"),
            "unconditional": "timeouts and registered failures included in every rate and mean",
            "faults": "evaluator/infrastructure exceptions recorded as fault rows, reported apart",
            "latency": "p50/p95/p99 over delivered decisions, with censored count",
            "q_perc": "eligible-GT Q_perc of the executed action (all) and of deliveries",
            "reporting": "every seed separately plus mean/min/max; no seed/checkpoint selection",
            "diagnostic_only": "cannot change training or hyper-parameters"},
        "row_counts": {"contexts": len(contexts()), "policies_per_context": len(specs),
                       "decisions_per_trajectory": decisions,
                       "decision_rows_per_context": per_context,
                       "decision_rows_total": per_context * len(contexts()),
                       "anchor_scores_total": per_context * len(contexts()) * len(anchors())},
        "proposed_run": {"command": SWEEP_COMMAND, "workers": SWEEP_WORKERS,
                         "output_dir": SWEEP_OUTPUT_RELPATH,
                         "measured_ms_per_decision_fit_fixture": 14.5,
                         "estimated_cpu_seconds": round(0.0145 * per_context * len(contexts())),
                         "estimated_wall_minutes": "3-5 with 12 workers (one per context)",
                         "estimated_disk_bytes": 650 * per_context * len(contexts()) + 5_000_000},
        "source_sha256": {**{f"rl_agent/splitfusion_hybrid_sac_run5_v1/{n}":
                             B.sha256_bytes((PACKAGE / n).read_bytes()) for n in EVALUATOR_SOURCES},
                          **{p: B.sha256_bytes((ROOT / p).read_bytes()) for p in REUSED_SOURCES}},
    }


def run_sweep(evidence_root: Path, output_dir: Path, workers: int) -> int:
    sealed = json.loads(MANIFEST_PATH.read_text())
    require(output_dir == (ROOT / sealed["proposed_run"]["output_dir"]).resolve(),
            "output directory differs from the sealed manifest")
    current = manifest_document(evidence_root)
    require(sealed == current, "evaluation manifest drifted from the sealed version")
    require(not output_dir.exists(), "evaluation output directory is create-only")
    (output_dir / "rows").mkdir(parents=True)
    partition = H.load_sealed_partition(evidence_root)
    specs_json = sealed["policies"]
    jobs = [(p, s, str(evidence_root), str(output_dir), specs_json, partition["eligible_keys"])
            for p, s in contexts()]
    started = time.time()
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        results = pool.map(evaluate_context, jobs)
    rows = []
    for result in results:
        path = output_dir / "rows" / f"{result['profile']}__{result['validation_seed']}.jsonl"
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    counts = sealed["row_counts"]
    complete = len(rows) == counts["decision_rows_total"] and not any(r["fault"] for r in rows)
    specs = [PolicySpec(**s) for s in specs_json]
    summary = summarize(rows, specs)
    B.atomic_write_file(output_dir / "SUMMARY.json",
                        json.dumps(summary, indent=1, sort_keys=True).encode())
    B.atomic_write_file(output_dir / "EVALUATION_COMPLETE.json", B.canonical_bytes({
        "schema": SCHEMA, "manifest_sha256": B.sha256_bytes(MANIFEST_PATH.read_bytes()),
        "contexts": results, "rows": len(rows), "expected_rows": counts["decision_rows_total"],
        "complete": complete, "faults": summary["faults_total"],
        "summary_sha256": B.sha256_bytes((output_dir / "SUMMARY.json").read_bytes()),
        "seconds": time.time() - started}))
    print(json.dumps({"rows": len(rows), "expected": counts["decision_rows_total"],
                      "complete": complete, "faults": summary["faults_total"]}))
    return 0 if complete else 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--select-fixed-action", action="store_true")
    parser.add_argument("--seal-manifest", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=SWEEP_WORKERS)
    args = parser.parse_args(argv)
    root = args.evidence_root.resolve()
    if args.select_fixed_action:
        require(not FIXED_ACTION_PATH.exists(), "fixed-action selection is create-only")
        document = select_fixed_action(root)
        with FIXED_ACTION_PATH.open("x") as handle:
            json.dump(document, handle, indent=1, sort_keys=True)
            handle.write("\n")
        print(json.dumps(document["selected"]))
        return 0
    if args.seal_manifest:
        require(not MANIFEST_PATH.exists(), "evaluation manifest is create-only")
        document = manifest_document(root)
        with MANIFEST_PATH.open("x") as handle:
            json.dump(document, handle, indent=1, sort_keys=True)
            handle.write("\n")
        print(json.dumps(document["row_counts"]))
        return 0
    if args.run:
        require(args.output_dir is not None, "--run needs --output-dir")
        return run_sweep(root, args.output_dir.resolve(), args.workers)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
