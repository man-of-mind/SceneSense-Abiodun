#!/usr/bin/env python3
"""Phase F: registered 288 no-gradient preflight, then the seed-17 smoke.

Uses only ``build_frozen_warmup_schedule()``; no ad-hoc action grid exists
here.  The preflight lower bounds are bound to registered evidence (the FIT
scene catalogue and the sealed MCS model support), and the backlog bound
follows ``BACKLOG_PREFLIGHT_AMENDMENT.md``.

Running this module executes CPU-only offline training.  It launches no
radio, no CARLA, no container and no network service.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as orch
from rl_agent.splitfusion_hybrid_sac_run4_v1 import smoke_preregistration
from rl_agent.splitfusion_hybrid_sac_run4_v1 import trainer
from rl_agent.splitfusion_hybrid_sac_v1 import hybrid_sac_models

from . import collector_v1 as CV
from . import contract_v2 as C2


PREFLIGHT_SCHEMA = "scenesense.run4_v2_smoke.v1"

# Evidence-bound lower bounds.  Scene bounds are 50% of the FIT catalogue's
# own span; the MCS bound is 50% of the sealed model's support; the backlog
# bound is the smallest physically meaningful positive span (one byte).
CAMERA_SI_MIN_DISTINCT, CAMERA_SI_MIN_SPAN = 100, 2.2263
RADAR_P40_MIN_DISTINCT, RADAR_P40_MIN_SPAN = 100, 0.2254
MCS_MIN_DISTINCT, MCS_MIN_SPAN = 5, 0.3214
BACKLOG_MIN_DISTINCT = 2
BACKLOG_MIN_SPAN = math.log1p(1) / math.log1p(
    C2.RLC_AM_TX_ADMISSION_CEILING_BYTES)


def build_variation_contract(
    evidence_sha256: str,
) -> orch.PreflightVariationContractV1:
    return orch.PreflightVariationContractV1(
        contract_id="run4-production-transport-v2-preflight",
        contract_version=1,
        evidence_sha256=evidence_sha256,
        feature_schema_sha256=orch.contract.FEATURE_SCHEMA_SHA256,
        source_partition=orch.FIT_PARTITION_LABEL,
        requirements=(
            orch.PreflightFeatureRequirementV1(
                "camera_si_scaled", CAMERA_SI_MIN_DISTINCT,
                CAMERA_SI_MIN_SPAN),
            orch.PreflightFeatureRequirementV1(
                "radar_p40", RADAR_P40_MIN_DISTINCT, RADAR_P40_MIN_SPAN),
            orch.PreflightFeatureRequirementV1(
                "prior_ul_mcs_normalized", MCS_MIN_DISTINCT, MCS_MIN_SPAN),
            # Amended: see BACKLOG_PREFLIGHT_AMENDMENT.md.  No maximum-zero
            # threshold is imposed.
            orch.PreflightFeatureRequirementV1(
                "pre_action_rlc_backlog_log1p_scaled",
                BACKLOG_MIN_DISTINCT, BACKLOG_MIN_SPAN),
        ))


def backlog_report(diagnostics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    values = [row["pre_enqueue_backlog_bytes"] for row in diagnostics]
    causal = all(isinstance(v, int) and v >= 0 for v in values)
    positive = [v for v in values if v > 0]
    zeros = [v for v in values if v == 0]
    return {
        "amendment": "BACKLOG_PREFLIGHT_AMENDMENT.md",
        "measurements": len(values),
        "all_causal_finite_non_imputed": causal,
        "zero_fraction": len(zeros) / len(values) if values else math.nan,
        "positive_count": len(positive),
        "distinct_values": len(set(values)),
        "span_bytes": max(values) - min(values) if values else 0,
        "min_bytes": min(values) if values else None,
        "max_bytes": max(values) if values else None,
        "modes_covered": sorted({row["mode_id"] for row in diagnostics}),
        "mcs_values_covered": sorted({row["prior_ul_mcs"]
                                      for row in diagnostics}),
        "has_zero": bool(zeros), "has_positive": bool(positive),
        "maximum_zero_threshold_imposed": False,
        "passed": (causal and bool(zeros) and bool(positive)
                   and len(set(values)) >= BACKLOG_MIN_DISTINCT
                   and (max(values) - min(values)) > 0),
    }


def trajectory_report(diagnostics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rewards = [row["reward"] for row in diagnostics]
    warmup = rewards[:288]
    post = rewards[288:]
    modes = collections_counter(row["mode_id"] for row in diagnostics[288:])
    return {
        "decisions": len(diagnostics),
        "warmup_mean_reward": statistics.fmean(warmup) if warmup else None,
        "post_warmup_mean_reward": statistics.fmean(post) if post else None,
        "post_warmup_success_rate": (
            sum(1 for row in diagnostics[288:]
                if row["terminal"] == "SUCCESS") / len(post)) if post else None,
        "post_warmup_mode_counts": modes,
        "post_warmup_distinct_q": len({row["q_e4"]
                                       for row in diagnostics[288:]}),
        "backlog_zero_fraction_overall": (
            sum(1 for row in diagnostics
                if row["pre_enqueue_backlog_bytes"] == 0) / len(diagnostics)),
        "backlog_distinct_overall": len({row["pre_enqueue_backlog_bytes"]
                                         for row in diagnostics}),
    }


def collections_counter(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[str(value)] = out.get(str(value), 0) + 1
    return out


def _actor_physical_summary(
    runner, catalog, scene_keys: Sequence[str], features: tuple[float, ...]
) -> dict[str, float | int]:
    """Map conditional actor means through registered per-mode q support."""
    tensor = torch.tensor([features], dtype=torch.float32)
    actor = runner.model_bundle.actor
    with torch.no_grad():
        heads = actor(tensor)
        probabilities = torch.softmax(heads.logits, dim=-1).squeeze(0)
        lower_e4, upper_e4 = actor.active_q_e4_bounds()
        z = 0.5 * (torch.tanh(heads.mean.squeeze(0)) + 1.0)
        q = (
            lower_e4.to(dtype=torch.float32)
            + (upper_e4 - lower_e4).to(dtype=torch.float32) * z
        ) / 10000.0
        q_e4 = hybrid_sac_models.quantize_q_e4(q)
    if not scene_keys:
        raise ValueError("at least one scene key is required")
    wires = torch.tensor(
        [
            sum(
                catalog.draw(
                    scene_key, mode_id=mode_id,
                    q_e4=int(q_e4[mode_id])
                ).wire_bytes
                for scene_key in scene_keys
            )
            for mode_id in range(12)
        ],
        dtype=torch.float64,
    )
    weights = probabilities.to(dtype=torch.float64)
    q_exec = q_e4.to(dtype=torch.float64) / 10000.0
    deterministic_mode = int(torch.argmax(probabilities).item())
    return {
        "expected_q_exec": float((weights * q_exec).sum().item()),
        "expected_wire_bytes": float((weights * wires).sum().item()),
        "deterministic_mode_id": deterministic_mode,
        "deterministic_q_e4": int(q_e4[deterministic_mode]),
        "deterministic_wire_bytes": int(wires[deterministic_mode].item()),
    }


def _bootstrap_mean_ci(
    values: Sequence[float], *, draws: int, seed: int
) -> tuple[float, list[float]]:
    import numpy as _np
    array = _np.asarray(values, dtype=_np.float64)
    if array.ndim != 1 or array.size < 1 or not bool(_np.isfinite(array).all()):
        raise ValueError("bootstrap values must be a non-empty finite vector")
    generator = _np.random.default_rng(seed)
    indices = generator.integers(0, array.size, size=(draws, array.size))
    means = _np.sort(array[indices].mean(axis=1))
    return (
        float(array.mean()),
        [
            float(means[int(0.025 * (draws - 1))]),
            float(means[int(0.975 * (draws - 1))]),
        ],
    )


def backlog_response_probe(
    orchestrator, *, draws: int = 2000
) -> dict[str, Any]:
    """Controlled physical q/payload response to backlog only.

    All other state features and each row's reward-scene identity remain fixed.
    Unlike the retired raw-tanh proxy, this maps every conditional head through
    that mode's registered q support, exact wire quantization and scene payload
    function before forming the categorical expectation.
    """
    collector = orchestrator.collector
    runner = orchestrator.runner
    diagnostics = collector.diagnostics()
    history = collector.history()
    if len(diagnostics) != len(history) or not diagnostics:
        raise ValueError("collector diagnostics/history are incomplete")
    backlogs = [row["pre_enqueue_backlog_bytes"] for row in diagnostics]
    low_raw, high_raw = min(backlogs), max(backlogs)
    index = orch.contract.POLICY_FEATURE_ORDER.index(
        "pre_action_rlc_backlog_log1p_scaled"
    )
    low = C2.backlog_scaled(float(low_raw))
    high = C2.backlog_scaled(float(high_raw))

    expected_q_effects: list[float] = []
    expected_wire_effects: list[float] = []
    deterministic_q_effects: list[float] = []
    deterministic_wire_effects: list[float] = []
    for item, row in zip(history, diagnostics):
        base = list(item.state_features)
        base[index] = low
        scene_keys = (row["reward_scene_key"], row["held_scene_key"])
        low_summary = _actor_physical_summary(
            runner, collector.catalog, scene_keys, tuple(base)
        )
        base[index] = high
        high_summary = _actor_physical_summary(
            runner, collector.catalog, scene_keys, tuple(base)
        )
        expected_q_effects.append(
            float(high_summary["expected_q_exec"])
            - float(low_summary["expected_q_exec"])
        )
        expected_wire_effects.append(
            float(high_summary["expected_wire_bytes"])
            - float(low_summary["expected_wire_bytes"])
        )
        deterministic_q_effects.append(
            (int(high_summary["deterministic_q_e4"])
             - int(low_summary["deterministic_q_e4"])) / 10000.0
        )
        deterministic_wire_effects.append(
            float(high_summary["deterministic_wire_bytes"])
            - float(low_summary["deterministic_wire_bytes"])
        )

    q_mean, q_ci = _bootstrap_mean_ci(
        expected_q_effects, draws=draws, seed=C2.MODEL_SEED
    )
    wire_mean, wire_ci = _bootstrap_mean_ci(
        expected_wire_effects, draws=draws, seed=C2.MODEL_SEED + 1
    )
    desired_direction = q_ci[0] > 0.0 and wire_ci[1] < 0.0
    return {
        "probe": "CONTROLLED_BACKLOG_ONLY_PHYSICAL_ACTION_SWEEP",
        "n_states": len(expected_q_effects),
        "backlog_low_bytes": low_raw,
        "backlog_high_bytes": high_raw,
        "backlog_low_scaled": low,
        "backlog_high_scaled": high,
        "expected_q_exec_delta_mean": q_mean,
        "expected_q_exec_delta_ci95": q_ci,
        "expected_two_tensor_ingress_bytes_delta_mean": wire_mean,
        "expected_two_tensor_ingress_bytes_delta_ci95": wire_ci,
        "deterministic_q_exec_delta_mean": statistics.fmean(
            deterministic_q_effects
        ),
        "deterministic_two_tensor_ingress_bytes_delta_mean": statistics.fmean(
            deterministic_wire_effects
        ),
        "desired_direction": (
            "higher backlog -> higher drop fraction and fewer two-tensor ingress bytes"
        ),
        "desired_direction_passed": desired_direction,
        "bootstrap_draws": draws,
    }


def build_harness(
    artifact_path: Path,
    seed: int = smoke_preregistration.FROZEN_CONFIG.initial_smoke_seed,
):
    torch.set_num_threads(
        smoke_preregistration.FROZEN_CONFIG.torch_intraop_threads)
    seed_plan = orch.RunnerSeedPlanV1.for_registered_seed(seed)
    probe = CV.RealModeledTransitionCollectorV1(
        artifact_path=artifact_path, seed=seed
    )
    shared = probe.shared_sources()
    binding = probe._modeled_binding()
    factory = orch.ModeledSmokeRunnerFactoryV1(
        modeled_binding=binding,
        gamma=smoke_preregistration.FROZEN_CONFIG.gamma_per_tensor,
        freshness_policy_sha256=probe._provider.freshness.canonical_sha256(),
        empirical_scaling_sha256=probe._provider.scaling.canonical_sha256(),
        trainer_config=trainer.TrainerConfigV1(
            alpha_d=smoke_preregistration.FROZEN_CONFIG.alpha_d,
            alpha_c=smoke_preregistration.FROZEN_CONFIG.alpha_c,
            tau=smoke_preregistration.FROZEN_CONFIG.polyak_tau,
            actor_lr=smoke_preregistration.FROZEN_CONFIG.actor_learning_rate,
            critic_lr=smoke_preregistration.FROZEN_CONFIG.critic_learning_rate,
            nominal_batch_size=smoke_preregistration.FROZEN_CONFIG.batch_size),
        seed_plan=seed_plan)

    def collector_factory():
        return CV.RealModeledTransitionCollectorV1(
            artifact_path=artifact_path, seed=seed, shared_sources=shared)

    contract = build_variation_contract(probe.binding_evidence_sha256)
    return factory, collector_factory, contract


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)

    output = Path(args.output_dir)
    if output.exists():
        raise SystemExit("smoke output directory is create-only")
    output.mkdir(parents=True, exist_ok=False)

    factory, collector_factory, contract = build_harness(Path(args.artifact))
    orchestrator = orch.ModeledSmokeOrchestratorV1(
        runner_factory=factory, collector_factory=collector_factory,
        preflight_variation_contract=contract)

    report = orchestrator.run_no_gradient_preflight()
    diagnostics = orchestrator.collector.diagnostics()
    backlog = backlog_report(diagnostics)
    document = {
        "schema": PREFLIGHT_SCHEMA,
        "contract_v2_sha256": C2.CONTRACT_V2_SHA256,
        "evidence_class": C2.EVIDENCE_CLASS,
        "decision_count": report.decision_count,
        "success_count": report.success_count,
        "failure_count": report.failure_count,
        "feature_diagnostics": [item.to_dict() if hasattr(item, "to_dict")
                                else item
                                for item in report.feature_diagnostics],
        "backlog_preflight": backlog,
        "schedule_id": orchestrator.schedule.config.schedule_id,
    }
    with (output / "PREFLIGHT_288.json").open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    print(json.dumps({
        "decisions": report.decision_count,
        "success": report.success_count, "failure": report.failure_count,
        "backlog_passed": backlog["passed"],
        "backlog_zero_fraction": backlog["zero_fraction"],
        "backlog_positive_count": backlog["positive_count"],
    }, indent=2))
    if not backlog["passed"]:
        return 1
    if args.preflight_only:
        return 0

    # Full canonical event-sourced checkpoints, not metadata-only sidecars.
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    milestones: list[dict[str, Any]] = []

    def on_checkpoint(event) -> None:
        metrics = event.latest_metrics
        checkpoint_path = (
            checkpoint_dir / f"update_{event.update:06d}.checkpoint.json"
        )
        file_sha256 = orch.write_checkpoint(
            checkpoint_path, event.checkpoint
        )
        roundtrip = orch.read_checkpoint(checkpoint_path)
        if roundtrip.canonical_sha256 != event.checkpoint.canonical_sha256:
            raise RuntimeError("durable checkpoint round-trip differs")
        payload = {
            "update": event.update,
            "checkpoint_sha256": event.checkpoint.canonical_sha256,
            "checkpoint_file_sha256": file_sha256,
            "checkpoint_relpath": str(checkpoint_path.relative_to(output)),
            "decision_count": event.checkpoint.decision_count,
            "metrics": (dataclasses.asdict(metrics)
                        if metrics is not None else None),
        }
        milestones.append(payload)
        metrics_path = (
            checkpoint_dir / f"update_{event.update:06d}.metrics.json"
        )
        with metrics_path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")

    summary = orchestrator.run_to_hard_stop(checkpoint_callback=on_checkpoint)
    diagnostics = orchestrator.collector.diagnostics()
    probe = backlog_response_probe(orchestrator)
    manifest_document = {
        "schema": "scenesense.run4_full_checkpoint_manifest.v1",
        "seed": factory.seed_plan.master_seed,
        "checkpoints": milestones,
    }
    manifest_bytes = json.dumps(
        manifest_document, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False
    ).encode("ascii")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    with (output / "CHECKPOINT_MANIFEST.json").open(
        "x", encoding="utf-8"
    ) as handle:
        json.dump(
            {**manifest_document, "manifest_sha256": manifest_sha256},
            handle, indent=2, sort_keys=True
        )
        handle.write("\n")
    document = {
        "schema": PREFLIGHT_SCHEMA,
        "contract_v2_sha256": C2.CONTRACT_V2_SHA256,
        "evidence_class": C2.EVIDENCE_CLASS,
        "summary": {
            "starting_update": summary.starting_update,
            "final_update": summary.final_update,
            "final_decision_count": summary.final_decision_count,
            "emitted_checkpoint_updates":
                list(summary.emitted_checkpoint_updates),
            "preflight_report_sha256": summary.preflight_report_sha256,
            "final_checkpoint_sha256": summary.final_checkpoint_sha256,
        },
        "milestones": milestones,
        "checkpoint_manifest_sha256": manifest_sha256,
        "trajectory": trajectory_report(diagnostics),
        "backlog_response_at_500": probe,
    }
    with (output / "SMOKE_500.json").open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    print(json.dumps({
        "final_update": summary.final_update,
        "final_decisions": summary.final_decision_count,
        "checkpoints": list(summary.emitted_checkpoint_updates),
        "backlog_response_desired_direction_passed":
            probe["desired_direction_passed"],
        "backlog_response_expected_q_delta":
            probe["expected_q_exec_delta_mean"],
        "backlog_response_expected_wire_bytes_delta":
            probe["expected_two_tensor_ingress_bytes_delta_mean"],
    }, indent=2))
    return 0 if probe["desired_direction_passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
