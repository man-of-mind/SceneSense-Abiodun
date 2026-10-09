#!/usr/bin/env python3
"""Pre-deep-training readiness audit.  Zero updates, zero optimizer steps.

Reads the committed seed-17 smoke bundles, restores them without gradient
replay (``update_once`` and ``Adam.step`` are patched to fail), and checks:

1. contract/dataflow - fresh native 22-D models, Run-4 meanings of 0-20,
   previous action/outcome propagation, feature 21 = causal ACKed proxy,
   SNR outside the reward, registered reward/profiles/seeds/hyper-parameters;
2. checkpoint/recovery - every bundle's contents and the committed
   250 -> 500 resume-equivalence evidence;
3. launch rehearsal inputs - the complete hidden profile schedule for all
   training and validation seeds (audit-only, hashed), and disk/inode needs
   extrapolated from the actual smoke bundle sizes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from unittest import mock

import torch

from rl_agent.splitfusion_hybrid_sac_run4_v1 import models as R4M
from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as R4
from rl_agent.splitfusion_hybrid_sac_run4_v1 import smoke_preregistration as R4PREREG
from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as R4ORCH

from . import run5_bundle as B
from . import run5_channel as J
from . import run5_collector as RC
from . import run5_models as RM
from . import run5_preregistration as PR
from . import run5_snr_v2 as SNR
from . import run5_training as RT

PACKAGE = Path(__file__).resolve().parent
SMOKE = PACKAGE / "smoke_runs"
RUNS = {"uninterrupted": SMOKE / "20260929_seed17_uninterrupted" / "seed_17",
        "interrupted_resumed": SMOKE / "20260929_seed17_interrupted_resumed" / "seed_17"}
ARTIFACT = PACKAGE.parents[1] / ("rl_agent/experiments/ue_production_queue_capture_v1/"
                                 "20260929_model_v2b/transport_model_v2.json")
EXPECTED_BUNDLE_MEMBERS = {"event.json", "training_state.pt", "actor_state_dict.pt",
                           "channel_state.json", B.MANIFEST, B.COMMITTED}


def _sha(value) -> str:
    return hashlib.sha256(B.canonical_bytes(value)).hexdigest()


def check(results: dict, name: str, condition: bool, detail=None) -> None:
    results[name] = {"passed": bool(condition), **({"detail": detail} if detail is not None else {})}


def contract_pass(evidence_root: Path) -> tuple[dict, dict]:
    results: dict = {}
    bundles = {u: B.verify_bundle(RUNS["uninterrupted"] / "checkpoints" / f"checkpoint_{u:06d}")
               for u in PR.CONFIG.smoke_checkpoints}
    prereg_sha = bundles[500].manifest["preregistration_sha256"]

    # -- fresh native 22-D models ---------------------------------------
    plan = RT.seed_plan(17)
    fresh_actor, fresh_critics = RM.build_run5_models(actor_seed=plan["actor_seed"],
                                                     critic_seed=plan["critics_seed"])
    actor0 = B.torch_from_bytes(bundles[0].payload("actor_state_dict.pt"))
    state0 = B.torch_from_bytes(bundles[0].payload("training_state.pt"))
    critics0 = {**state0["online_critics"], **state0["target_critics"]}
    check(results, "update0_actor_is_fresh_run5_init",
          all(torch.equal(v, actor0[k]) for k, v in fresh_actor.state_dict().items()))
    check(results, "update0_critics_and_targets_are_fresh_run5_init",
          all(torch.equal(v, critics0[k]) for k, v in fresh_critics.state_dict().items()))
    check(results, "update0_optimizers_have_no_step",
          len(state0["actor_optimizer"]["state"]) == 0
          and len(state0["critic_optimizer"]["state"]) == 0)
    run4_plan = R4ORCH.RunnerSeedPlanV1.for_registered_seed(17)
    run4 = R4M.build_run4_models(actor_seed=run4_plan.actor_seed,
                                 critic_seed=run4_plan.critic_seed)
    w5, w4 = actor0["encoder.0.weight"], run4.actor.state_dict()["encoder.0.weight"]
    check(results, "not_transferred_or_padded_from_run4",
          tuple(w5.shape) == (128, 22) and tuple(w4.shape) == (128, 21)
          and not torch.equal(w5[:, :21], w4) and bool(w5[:, 21].abs().sum() > 0)
          and all(bool(B.torch_from_bytes(b.payload("actor_state_dict.pt"))["encoder.0.weight"]
                       [:, 21].abs().sum() > 0) for b in bundles.values()))
    check(results, "critic_input_width_35",
          all(tuple(critics0[f"{n}.trunk.0.weight"].shape) == (128, 35)
              for n in ("critic_1", "critic_2", "target_1", "target_2")))

    # -- restore 500 without any optimizer step -------------------------
    torch.set_num_threads(PR.CONFIG.torch_intraop_threads)
    shared = RC.build_shared_sources(ARTIFACT, evidence_root)
    factory = lambda: RC.Run5ModeledCollectorV1(artifact_path=ARTIFACT, seed=17,
                                                evidence_root=evidence_root,
                                                shared_sources=shared)
    with mock.patch.object(RT.Run5HybridSacTrainerV1, "update_once",
                           side_effect=AssertionError("gradient step")), \
            mock.patch.object(torch.optim.Adam, "step", side_effect=AssertionError("optimizer")):
        orchestrator = RT.Run5OrchestratorV1.restore_from_bundle(
            bundles[500], collector_factory=factory, preregistration_sha256=prereg_sha)
    history = orchestrator.history
    diagnostics = orchestrator.collector.diagnostics()
    check(results, "restored_500_without_gradient_or_optimizer_step",
          orchestrator.update_count == 500 and len(history) == 2288)

    # -- features 0-20: Run-4 meanings and connected ---------------------
    check(results, "feature_order_is_run4_prefix_plus_snr",
          RM.RUN5_TRAINING_MODEL_BINDING["policy_feature_order"][:21]
          == tuple(R4.POLICY_FEATURE_ORDER)
          and RM.RUN5_TRAINING_MODEL_BINDING["policy_feature_order"][21]
          == "effective_external_ul_snr_proxy_scaled")
    scaling = orchestrator.collector._provider.scaling
    meaning = 0
    for record, d in zip(history, diagnostics):
        meaning += int(record.state[0] != (d["camera_si"] - scaling.camera_si_center)
                       / scaling.camera_si_scale
                       or record.state[1] != d["radar_p40"]
                       or record.state[2] != d["prior_ul_mcs"] / 28.0
                       or record.state[3] != __import__("math").log1p(
                           d["pre_enqueue_backlog_bytes"]) / scaling.backlog_log1p_scale)
    check(results, "features_0_3_equal_run4_definitions_of_the_measured_values", meaning == 0,
          {"mismatches": meaning})
    variation = RT.state_variation(history)
    check(results, "features_connected_and_varying",
          all(v["distinct"] >= 2 and v["span"] > 0 for v in variation.values()), variation)

    # -- previous action/outcome --------------------------------------------
    mismatches = sum(int(tuple(r.state[4:21]) != RT.expected_previous_features(p))
                     for p, r in zip(history, history[1:]))
    outcomes = {p.terminal for p in history[:-1]}
    check(results, "previous_action_and_resolved_outcome_populate_next_state",
          mismatches == 0 and history[0].state[19] == 0.0 and outcomes >= {"SUCCESS", "TIMEOUT"},
          {"mismatches": mismatches, "outcomes_seen": sorted(outcomes)})

    # -- feature 21 ------------------------------------------------------------
    acks = list(orchestrator.collector._snr_adapter._acks)
    beats = list(orchestrator.collector._snr_adapter._heartbeats)
    bad = 0
    for index, (record, d) in enumerate(zip(history, diagnostics)):
        ack = next(a for a in acks if a.command_id.endswith(f"-{record.decision_seq}"))
        bad += int(not ack.usable or ack.target_snr_db != d["snr_db"]
                   or record.state[21] != (d["snr_db"] - 5.5) / 19.0
                   or not 5.5 <= d["snr_db"] <= 24.5
                   or not d["generated_ticks_after_observed"])
    successors = [d["successor_snr_db"] for d in diagnostics]
    future_hits = sum(int(d["snr_db"] in set(successors[i:])) for i, d in enumerate(diagnostics))
    check(results, "feature21_is_the_acked_active_leased_proxy_only",
          bad == 0 and future_hits == 0 and len(acks) == len(beats) >= len(history)
          and all(a.usable for a in acks),
          {"violations": bad, "equal_to_a_later_generated_sample": future_hits})
    check(results, "feature21_has_no_profile_or_hidden_input",
          set(J.ChannelObservationV1.__dataclass_fields__) == {"snr_db", "mcs", "tick"}
          and not any(t in n for n in RC.Run5CollectedTransitionV1.__dataclass_fields__
                      for t in ("profile", "hidden", "trace", "markov")))
    check(results, "no_zero_filled_snr_path",
          all(r.state[21] > 0.0 for r in history) and SNR.scale_snr_db(5.5) == 0.0,
          "modeled SNR guard failures raise; no fallback value is ever written")

    # -- reward ------------------------------------------------------------------
    check(results, "reward_equals_registered_formula_and_excludes_snr",
          all(RT.reward_matches(r) for r in history)
          and "SNR" in PR.DESIGN["reward"]["absent_terms"]
          and PR.DESIGN["reward"]["schema_sha256"] == R4.REWARD_SCHEMA_SHA256
          and R4.REWARD_DEADLINE_MS == 170.0 and R4.REWARD_LATENCY_WEIGHT == 0.25
          and R4.REGISTERED_FAILURE_REWARD == -1.0)

    # -- registered design unchanged ------------------------------------------
    sealed = json.loads((PACKAGE / PR.SEALED_FILENAME).read_text())
    run4 = R4PREREG.FROZEN_CONFIG
    same = all(getattr(PR.CONFIG, n) == getattr(run4, n) for n in (
        "gamma_per_tensor", "alpha_d", "alpha_c", "actor_learning_rate", "critic_learning_rate",
        "polyak_tau", "batch_size", "replay_capacity", "warmup_decision_count",
        "environment_transitions_per_update", "seed_order"))
    check(results, "registered_reward_profiles_seeds_hyperparameters_unchanged",
          same and sealed["config"] == json.loads(json.dumps(
              {k: list(v) if isinstance(v, tuple) else v
               for k, v in PR.asdict(PR.CONFIG).items()}))
          and sealed["design"] == PR.DESIGN
          and list(J.TRAINING_PROFILES) == PR.DESIGN["channel"]["profiles"])
    return results, {"orchestrator": orchestrator, "bundles": bundles}


def checkpoint_pass() -> dict:
    results: dict = {}
    for run, root in RUNS.items():
        for u in PR.CONFIG.smoke_checkpoints:
            bundle = B.verify_bundle(root / "checkpoints" / f"checkpoint_{u:06d}")
            members = {p.name for p in bundle.path.iterdir()}
            state = B.torch_from_bytes(bundle.payload("training_state.pt"))
            event = json.loads(bundle.payload("event.json"))
            channel = json.loads(bundle.payload("channel_state.json"))
            complete = (members == EXPECTED_BUNDLE_MEMBERS
                        and set(state) >= {"online_critics", "target_critics",
                                           "actor_optimizer", "critic_optimizer", "generators"}
                        and set(state["generators"]) == set(RT.GENERATOR_NAMES)
                        and {"ledger", "collector_checkpoint", "boundary", "preflight",
                             "checkpoint_updates"} <= set(event)
                        and {"tick", "rng_states", "profile_block_remaining", "segment_left",
                             "hidden_state"} <= set(channel)
                        and event["decision_count"] == 288 + 4 * u
                        and bundle.manifest["selection_candidate"] is True)
            check(results, f"{run}/checkpoint_{u:06d}_complete", complete)
    equivalence = json.loads((SMOKE / "RESUME_EQUIVALENCE_250_TO_500.json").read_text())
    live = {u: all(B.verify_bundle(RUNS[r] / "checkpoints" / f"checkpoint_{u:06d}").manifest_sha256
                   for r in RUNS) and
            B.verify_bundle(RUNS["uninterrupted"] / "checkpoints" / f"checkpoint_{u:06d}")
            .manifest["files"] == B.verify_bundle(RUNS["interrupted_resumed"] / "checkpoints"
                                                  / f"checkpoint_{u:06d}").manifest["files"]
            for u in PR.CONFIG.smoke_checkpoints}
    check(results, "update500_stop_resume_equivalence_still_valid",
          equivalence["all_equal"] and all(live.values()), live)
    for run, root in RUNS.items():
        final = B.verify_bundle(root / "final_actor_000500")
        check(results, f"{run}/final_actor_manifested",
              final.manifest["actor_tree_sha256"]
              == B.verify_bundle(root / "checkpoints/checkpoint_000500").manifest["actor_tree_sha256"])
    return results


def profile_schedule(evidence_root: Path) -> dict:
    kernel, design = J.load_accepted_kernel(evidence_root), J.load_design()
    decisions = PR.CONFIG.warmup_decision_count + (
        PR.CONFIG.environment_transitions_per_update * PR.CONFIG.deep_target_update)
    out = {}
    for label, seeds, namespace in (("training", PR.CONFIG.seed_order, "train-channel"),
                                    ("validation", PR.CONFIG.validation_seeds,
                                     "validation-channel")):
        for seed in seeds:
            channel = J.JointSnrMcsChannelV1(kernel=kernel, design=design,
                                             seed=J.derive_seed(seed, namespace))
            segments = [channel._profile]
            started = channel._segments_started
            for _ in range(decisions):
                channel.observe()
                channel.advance(2)
                if channel._segments_started != started:
                    segments.append(channel._profile)
                    started = channel._segments_started
            names = [J.TRAINING_PROFILES[i] for i in segments]
            blocks = [names[i:i + 4] for i in range(0, len(names) - len(names) % 4, 4)]
            out[f"{label}_seed_{seed}"] = {
                "decisions": decisions, "ticks_after_burn_in": channel._tick,
                "segments": len(names), "balance": channel.profile_balance(),
                "every_complete_block_has_each_profile_once":
                    all(sorted(b) == sorted(J.TRAINING_PROFILES) for b in blocks),
                "segment_sequence_sha256": hashlib.sha256(
                    json.dumps(names).encode()).hexdigest(),
                "first_8_segments": names[:8]}
    streams = [v["segment_sequence_sha256"] for v in out.values()]
    return {"audit_only": "hidden profile schedule; never an actor input",
            "action_independent": ("profile blocks are drawn from a dedicated RNG stream and "
                                   "segments advance exactly 2 ticks per decision, so the "
                                   "schedule does not depend on the policy"),
            "all_streams_distinct": len(set(streams)) == len(streams), "schedules": out,
            "schedule_set_sha256": _sha(out)}


def disk_projection(campaign_dir: Path) -> dict:
    sizes = []
    for u in PR.CONFIG.smoke_checkpoints:
        path = RUNS["uninterrupted"] / "checkpoints" / f"checkpoint_{u:06d}"
        sizes.append((288 + 4 * u, sum(p.stat().st_size for p in path.iterdir())))
    (d0, s0), (d1, s1) = sizes[0], sizes[-1]
    per_decision = (s1 - s0) / (d1 - d0)
    fixed = s0 - per_decision * d0
    root = RUNS["uninterrupted"]
    log_per_decision = (root / "decisions.jsonl").stat().st_size / 2288
    log_per_update = (root / "metrics.jsonl").stat().st_size / 500
    final_actor = sum(p.stat().st_size for p in (root / "final_actor_000500").iterdir())
    target = PR.CONFIG.deep_target_update
    decisions = 288 + 4 * target
    per_seed = (sum(fixed + per_decision * (288 + 4 * u) for u in PR.CONFIG.deep_checkpoints)
                + 2 * (fixed + per_decision * decisions)            # emergency + staging
                + final_actor + log_per_decision * decisions + log_per_update * target)
    three = 3 * per_seed
    inodes_per_seed = len(PR.CONFIG.deep_checkpoints) * 7 + 2 * 7 + 4 + 8
    probe = campaign_dir.resolve()
    while not probe.exists():
        probe = probe.parent
    usage, vfs = shutil.disk_usage(probe), os.statvfs(probe)
    return {"measured_from": "four seed-17 smoke bundles + ledgers",
            "bundle_fixed_bytes": fixed, "bundle_bytes_per_decision": per_decision,
            "ledger_bytes_per_decision": log_per_decision,
            "metrics_bytes_per_update": log_per_update,
            "calibrated_three_seed_bytes": three,
            "calibrated_three_seed_with_2x_margin_bytes": 2 * three,
            "inodes_three_seed": 3 * inodes_per_seed,
            "free_bytes": usage.free, "free_inodes": vfs.f_favail,
            "calibrated_passes": usage.free > 2 * three and vfs.f_favail > 30 * inodes_per_seed}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    contract, _ = contract_pass(args.evidence_root)
    checkpoints = checkpoint_pass()
    schedule = profile_schedule(args.evidence_root)
    disk = disk_projection(args.campaign_dir)
    document = {"schema": "splitfusion.run5.deep_readiness_audit.v1", "pid": os.getpid(),
                "optimizer_steps": 0, "training_checkpoints_created": 0,
                "contract_dataflow": contract, "checkpoint_recovery": checkpoints,
                "profile_schedule": schedule, "disk_projection": disk}
    document["passed"] = (all(v["passed"] for v in contract.values())
                          and all(v["passed"] for v in checkpoints.values())
                          and schedule["all_streams_distinct"]
                          and all(s["every_complete_block_has_each_profile_once"]
                                  for s in schedule["schedules"].values())
                          and disk["calibrated_passes"])
    B.atomic_write_file(args.output, json.dumps(document, indent=1, sort_keys=True,
                                                default=str).encode())
    failed = [k for section in (contract, checkpoints) for k, v in section.items()
              if not v["passed"]]
    print(json.dumps({"passed": document["passed"], "failed": failed,
                      "schedule_set_sha256": schedule["schedule_set_sha256"],
                      "disk_three_seed_calibrated_gb": disk["calibrated_three_seed_bytes"] / 1e9},
                     indent=1))
    return 0 if document["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
