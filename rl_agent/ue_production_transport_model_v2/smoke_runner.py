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


def build_harness(artifact_path: Path):
    torch.set_num_threads(
        smoke_preregistration.FROZEN_CONFIG.torch_intraop_threads)
    probe = CV.RealModeledTransitionCollectorV1(artifact_path=artifact_path)
    shared = probe.shared_sources()
    binding = probe._modeled_binding()
    factory = orch.ModeledSmokeRunnerFactoryV1(
        modeled_binding=binding, gamma=0.99,
        freshness_policy_sha256=probe._provider.freshness.canonical_sha256(),
        empirical_scaling_sha256=probe._provider.scaling.canonical_sha256(),
        trainer_config=trainer.TrainerConfigV1(
            alpha_d=smoke_preregistration.FROZEN_CONFIG.alpha_d,
            alpha_c=smoke_preregistration.FROZEN_CONFIG.alpha_c,
            tau=smoke_preregistration.FROZEN_CONFIG.polyak_tau,
            actor_lr=smoke_preregistration.FROZEN_CONFIG.actor_learning_rate,
            critic_lr=smoke_preregistration.FROZEN_CONFIG.critic_learning_rate,
            nominal_batch_size=smoke_preregistration.FROZEN_CONFIG.batch_size),
        seed_plan=orch.RunnerSeedPlanV1.seed17())

    def collector_factory():
        return CV.RealModeledTransitionCollectorV1(
            artifact_path=artifact_path, shared_sources=shared)

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

    result = orchestrator.run_to_hard_stop()
    with (output / "SMOKE_500.json").open("x", encoding="utf-8") as handle:
        json.dump({"result": result, "schema": PREFLIGHT_SCHEMA},
                  handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    print("smoke complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
