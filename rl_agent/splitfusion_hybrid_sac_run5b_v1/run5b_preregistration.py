#!/usr/bin/env python3
"""Prospective Run-5B training preregistration (sealed before any Run-5B smoke).

Every training number equals the Run-4/Run-5 value: the SAC hyper-parameters,
warm-up, checkpoint cadence, seeds, smoke stop and the three-seed 10,000-update
campaign.  A test asserts that equality; neither the Run-4 nor the Run-5
preregistration identity enters the Run-5B runtime binding.

What is new is the state (21 features, no Q_perc) and the a-priori live-actor
registration: seed 43 at update 10,000.

``seal()`` writes ``RUN5B_TRAINING_PREREGISTRATION.json`` with the SHA-256 of
every Run-5B source, every reused Run-5/Run-4 source and every config or
fitted-model artifact read.  ``load_sealed()`` recomputes all of them and
refuses on any drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as R4

from . import run5b_state_contract as C

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
SEALED_FILENAME = "RUN5B_TRAINING_PREREGISTRATION.json"
SCHEMA = "splitfusion.run5b.training_preregistration.v1"
DEEP_TARGET_UPDATE = 10_000
RUN5_PREREGISTRATION_SHA256 = "2270baa0cf025b5e64a85644a28b8fd98e1f9004b11dcd6cb39fb5f61c5d18cf"


@dataclass(frozen=True)
class Run5BTrainingConfigV1:
    gamma_per_tensor: float = 0.99
    alpha_d: float = 0.05
    alpha_c: float = 0.02
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    polyak_tau: float = 0.005
    batch_size: int = 256
    replay_capacity: int = 65_536
    hidden_width: int = 128
    hidden_depth: int = 2
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    warmup_mode_count: int = 12
    warmup_q_bin_count: int = 6
    warmup_samples_per_mode_q_bin: int = 4
    warmup_decision_count: int = 288
    environment_transitions_per_update: int = 4
    duration_tensors: int = 2
    torch_intraop_threads: int = 4
    seed_order: tuple[int, ...] = (17, 29, 43)
    smoke_seed: int = 17
    smoke_stop_update: int = 500
    smoke_checkpoints: tuple[int, ...] = (0, 100, 250, 500)
    deep_target_update: int = DEEP_TARGET_UPDATE
    deep_checkpoints: tuple[int, ...] = (0, 100, 250) + tuple(range(500, DEEP_TARGET_UPDATE + 1, 500))
    evaluation_checkpoints: tuple[int, ...] = (500, 1500, 2500, 5000, 7500, 10_000)
    validation_seeds: tuple[int, ...] = (9017, 9029, 9043)
    live_actor_seed: int = 43
    live_actor_update: int = DEEP_TARGET_UPDATE


CONFIG = Run5BTrainingConfigV1()

DESIGN: dict[str, Any] = {
    "state": {
        "feature_count": C.RUN5B_POLICY_FEATURE_COUNT,
        "feature_order": list(C.RUN5B_POLICY_FEATURE_ORDER),
        "positions_0_19": ("Run-4B: the Run-4 order and numeric semantics with "
                           "prev_quality_qperc removed"),
        "position_20": ("effective_external_ul_snr_proxy_scaled = (snr_db - 5.5) / 19.0 under the "
                        "frozen Run-5 v2 lease provider; out of support -> external fallback"),
        "previous_outcome": ("TransportPriorOutcomeV1: action, terminal and operational "
                             "latency only; a successful prior needs no Q_perc"),
        "construction": "native Run-5B builder; no Run-4/Run-5 builder output is sliced",
        "never_in_state": ["Q_perc", "reward", "ground truth", "profile id",
                           "hidden Markov state", "trace index", "future SNR",
                           "next channel command", "RFsim noise command", "gNB PUSCH SNR"],
    },
    "reward": {
        "schema_sha256": R4.REWARD_SCHEMA_SHA256,
        "success": "Q_perc - 0.25 * (L_ms / 170.0) (latency <= 170 ms, inclusive)",
        "timeout_or_registered_failure": -1.0,
        "infrastructure_or_evaluator_fault": "excluded",
        "operational_latency": "action-open to feedback, one clock, the Run-4 reward clock",
        "q_perc_role": "training reward evidence only; never actor state or operational prior",
    },
    "model": {"actor_input_width": 21, "critic_input_width": 34,
              "networks": "Run-4/Run-5 ConditionalHybridActor and TwinHybridCritics unchanged"},
    "channel": {
        "profiles": ["FAVORABLE_STABLE", "MID_VARIABLE", "ADVERSE_STABLE", "FADE_RECOVERY"],
        "source": "frozen Run-5 JointSnrMcsChannelV1, balanced 4-segment blocks, tapes unchanged",
        "training_channel_seed": "derive(seed, 'train-channel') (Run-5 rule)",
        "validation_channel_seed": "derive(validation_seed, 'validation-channel') (Run-5 rule)",
        "snr_audit": "not repeated; the accepted Run-5 SNR-tilted kernel is reused",
    },
    "scenes": {
        "training": "Run-4 FIT scene catalogue; scenes with undefined Q_perc excluded",
        "eligible_gt": "all reported performance is conditional on eligible ground truth",
    },
    "checkpoint_identity": {
        "refused": ["old Run-4 21-D actor (binding, feature schema/order, frozen tensors)",
                    "old Run-5 22-D actor (binding, feature schema/order, preregistration, width)",
                    "any Run-5B bundle whose schema id, feature order/hash, model binding, "
                    "preregistration or tensor tree differs"],
        "width_alone": "insufficient; Run-4 and Run-5B are both 21-D",
    },
    "smoke_gates": [
        "all losses, gradients and parameters finite",
        "all 12 modes selected by the post-warm-up stochastic actor",
        "post-warm-up continuous q non-degenerate (distinct q, per-mode spread)",
        "camera SI, radar P40, MCS, backlog, SNR, prev mode, prev q, prev latency, "
        "prev presence and prev success all vary",
        "reward equals the registered formula on every decision",
        "previous-outcome encoding equals the transport prior on every transition",
        "no Q_perc-derived value in any actor state",
        "current SNR is never a sample generated after the action",
        "four profiles balanced",
        "every registered boundary bundle verifies; exact cold actor load",
        "separate-process 250->500 resume bit-identical to the uninterrupted run",
    ],
    "deep_validation": {
        "design": "the sealed Run-5 held-scene evaluation adapted to the 21-D Run-5B state",
        "contexts": "255 held scenes x 4 profiles x validation seeds 9017/9029/9043",
        "comparators": ["frozen Run-5 22-D actors (update 10,000)", "Run-4 seed-43 actor",
                        "FIT-only fixed action", "one-step catalogue oracle"],
        "metrics": ["reward", "Q_perc", "operational latency", "timeout rate",
                    "mode and q diversity", "oracle regret", "true vs shuffled SNR",
                    "learning progression over evaluation checkpoints"],
        "use": "diagnostics only; never alter training or select hyper-parameters",
    },
    "live_actor": {
        "seed": CONFIG.live_actor_seed, "update": CONFIG.live_actor_update,
        "rule": ("registered before training; no validation-based seed or checkpoint "
                 "selection; emergency bundles are never candidates"),
    },
    "campaign": {"seeds": [17, 29, 43], "target_update": DEEP_TARGET_UPDATE,
                 "recovery_checkpoint_max_gap_updates": 500,
                 "deep_training_authorization": (
                     "user instruction 2026-09-30: proceed directly to deep training if the "
                     "smoke and recovery gates pass; stop only for a genuine failed gate")},
}

RUN5B_SOURCES = (
    "__init__.py", "run5b_state_contract.py", "run5b_models.py", "run5b_collector.py",
    "run5b_training.py", "run5b_bundle.py", "run5b_campaign.py", "run5b_preregistration.py",
)
REUSED_SOURCES = (
    "rl_agent/splitfusion_hybrid_sac_run5_v1/run5_state_contract.py",
    "rl_agent/splitfusion_hybrid_sac_run5_v1/run5_snr_v2.py",
    "rl_agent/splitfusion_hybrid_sac_run5_v1/run5_channel.py",
    "rl_agent/splitfusion_hybrid_sac_run5_v1/run5_collector.py",
    "rl_agent/splitfusion_hybrid_sac_run5_v1/run5_models.py",
    "rl_agent/splitfusion_hybrid_sac_run5_v1/run5_preregistration.py",
    "rl_agent/splitfusion_hybrid_sac_run5_v1/successor_mcs_snr_audit.py",
    "rl_agent/splitfusion_hybrid_sac_run4_v1/run4_contract.py",
    "rl_agent/splitfusion_hybrid_sac_run4_v1/models.py",
    "rl_agent/splitfusion_hybrid_sac_run4_v1/trainer.py",
    "rl_agent/splitfusion_hybrid_sac_run4_v1/environment.py",
    "rl_agent/splitfusion_hybrid_sac_run4_v1/exploration.py",
    "rl_agent/splitfusion_hybrid_sac_run4_v1/sequential_kernel.py",
    "rl_agent/splitfusion_hybrid_sac_run4_v1/mcs_transition_provider.py",
    "rl_agent/splitfusion_hybrid_sac_run4_v1/dynamic_mcs_273prb_evidence.py",
    "rl_agent/splitfusion_hybrid_sac_run4_v1/modeled_smoke_orchestrator.py",
    "rl_agent/splitfusion_hybrid_sac_run4_v1/checkpoint_io.py",
    "rl_agent/splitfusion_hybrid_sac_v1/hybrid_sac_models.py",
    "rl_agent/splitfusion_hybrid_sac_v1/action_contract.py",
    "rl_agent/splitfusion_hybrid_sac_v1/transaction_identity.py",
    "rl_agent/splitfusion_hybrid_sac_v1/modeled_smoke_support.py",
    "rl_agent/ue_production_transport_model_v2/collector_v1.py",
    "rl_agent/ue_production_transport_model_v2/contract_v2.py",
    "rl_agent/ue_production_transport_model_v2/artifact_v2.py",
    "rl_agent/ue_production_transport_model_v2/scene_source.py",
    "rl_agent/ue_production_transport_model_v2/smoke_runner.py",
)
CONFIG_ARTIFACTS = (
    "rl_agent/configs/network_profile_design_v2.json",
    "rl_agent/experiments/ue_production_queue_capture_v1/20260929_model_v2b/transport_model_v2.json",
    "rl_agent/splitfusion_hybrid_sac_run5_v1/SUCCESSOR_MCS_SNR_AUDIT.json",
    "rl_agent/splitfusion_hybrid_sac_run5_v1/RETAINED_SNR_RESIDUAL_AUDIT.json",
    "rl_agent/splitfusion_hybrid_sac_run5_v1/RUN5_TRAINING_PREREGISTRATION.json",
    "rl_agent/splitfusion_hybrid_sac_live_route_b_v2/ACTOR_BINDING_V2.json",
    "rl_agent/splitfusion_action_catalog_v1/splitfusion_72_action_catalog.json",
)


class PreregistrationError(RuntimeError):
    pass


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(root: Path = ROOT) -> dict[str, str]:
    paths = [f"rl_agent/splitfusion_hybrid_sac_run5b_v1/{name}" for name in RUN5B_SOURCES]
    return {path: _sha_file(root / path) for path in (*paths, *REUSED_SOURCES, *CONFIG_ARTIFACTS)}


def document(root: Path = ROOT) -> dict[str, Any]:
    from . import run5b_models as M

    return {
        "schema": SCHEMA, "prospective": True,
        "written_before_any_run5b_smoke": True,
        "config": {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(CONFIG).items()},
        "design": DESIGN,
        "amendments": [],
        "feature_schema_id": C.FEATURE_SCHEMA_ID,
        "feature_schema_sha256": C.FEATURE_SCHEMA_SHA256,
        "feature_order_sha256": C.FEATURE_ORDER_SHA256,
        "feature_schema": json.loads(json.dumps(C.FEATURE_SCHEMA_DESCRIPTOR,
                                                default=lambda v: dict(v))),
        "model_binding_sha256": M.RUN5B_TRAINING_MODEL_BINDING_SHA256,
        "source_sha256": source_hashes(root),
    }


def seal(root: Path = ROOT) -> Path:
    path = PACKAGE / SEALED_FILENAME
    if path.exists():
        raise PreregistrationError("the preregistration is create-only")
    path.write_text(json.dumps(document(root), indent=1, sort_keys=True) + "\n")
    return path


def load_sealed(root: Path = ROOT) -> dict[str, Any]:
    """Recompute every bound identity; refuse on any drift."""
    path = PACKAGE / SEALED_FILENAME
    if not path.is_file():
        raise PreregistrationError("Run-5B training preregistration is not sealed")
    sealed = json.loads(path.read_text())
    current = json.loads(json.dumps(document(root)))
    for key in ("schema", "config", "design", "amendments", "feature_schema_id",
                "feature_schema_sha256", "feature_order_sha256", "feature_schema",
                "model_binding_sha256"):
        if sealed.get(key) != current[key]:
            raise PreregistrationError(f"preregistration {key} differs from the code")
    drifted = sorted(k for k, v in current["source_sha256"].items()
                     if sealed["source_sha256"].get(k) != v)
    if drifted or set(sealed["source_sha256"]) != set(current["source_sha256"]):
        raise PreregistrationError(f"sealed sources drifted: {drifted}")
    return {"document": sealed, "sha256": _sha_file(path)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args(argv)
    if args.seal:
        print(seal())
    print(json.dumps({"sha256": load_sealed()["sha256"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
