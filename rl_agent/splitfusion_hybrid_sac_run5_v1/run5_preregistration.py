#!/usr/bin/env python3
"""Prospective Run-5 training preregistration (written before any Run-5 smoke).

The frozen numbers below are Run-5's own registration.  Several equal Run 4's
values by design (the Hybrid-SAC hyper-parameters, reward and deadline must
not change); that equality is asserted by a test, never by importing the
Run-4 smoke preregistration into the runtime binding.

``seal()`` writes ``RUN5_TRAINING_PREREGISTRATION.json`` with the exact SHA-256
of every Run-5 source file, every reused Run-4 source file and every config /
fitted-model artifact the run reads.  ``load_sealed()`` recomputes all of them
and refuses on any drift, so the runtime binding names only this document.
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

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
SEALED_FILENAME = "RUN5_TRAINING_PREREGISTRATION.json"
SCHEMA = "splitfusion.run5.training_preregistration.v1"
DEEP_TARGET_UPDATE = 10_000


@dataclass(frozen=True)
class Run5TrainingConfigV1:
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


CONFIG = Run5TrainingConfigV1()

DESIGN: dict[str, Any] = {
    "state": {
        "feature_count": 22,
        "positions_0_20": "unchanged Run-4 definitions, order and numeric semantics",
        "position_21": "effective_external_ul_snr_proxy_scaled = (snr_db - 5.5) / 19.0",
        "snr_support_db": [5.5, 24.5],
        "snr_semantics": ("latest successful unclamped ACKed target active at state "
                          "commit (< action-open) with a live controller lease; "
                          "missing/invalid/foreign/out-of-support -> external fallback"),
        "never_in_state": ["profile id", "hidden Markov state", "trace index", "future SNR",
                           "next channel command", "RFsim noise command", "gNB PUSCH SNR"],
        "previous_action_outcome": "Run-4 prev mode one-hot, prev q, prev Q_perc, "
                                   "prev latency, prev present, prev success",
    },
    "reward": {
        "schema_sha256": R4.REWARD_SCHEMA_SHA256,
        "success": "Q_perc - 0.25 * latency_ms / 170 (eligible feedback, latency <= 170 ms)",
        "registered_delivery_failure_or_timeout": -1.0,
        "infrastructure_or_evaluator_fault": "excluded",
        "deadline_ms": R4.REWARD_DEADLINE_MS,
        "absent_terms": ["SNR", "p_admit", "payload", "switching", "aggression"],
    },
    "channel": {
        "profiles": ["FAVORABLE_STABLE", "MID_VARIABLE", "ADVERSE_STABLE", "FADE_RECOVERY"],
        "sampling": ("300-tick segments; every block of 4 consecutive segments holds each "
                     "profile exactly once in a seeded order; hidden state continuous"),
        "successor_mcs": "accepted SNR-tilted Run-4 kernel (SUCCESSOR_MCS_SNR_AUDIT.json)",
        "kernel_transfer": {"FAVORABLE_STABLE": "PROFILE_TRANSFER_UNVALIDATED",
                            "ADVERSE_STABLE": "PROFILE_TRANSFER_UNVALIDATED"},
        "training_channel_seed": "derive(seed, 'train-channel')",
        "validation_channel_seed": "derive(validation_seed, 'validation-channel')",
    },
    "scenes": {
        "training": "Run-4 FIT scene catalogue; scenes with undefined Q_perc excluded",
        "eligible_gt": ("all reported performance is conditional on eligible ground truth; "
                        "no empty-scene reward is defined"),
        "live_reporting": "Option C: live session rollovers are counted and reported",
    },
    "smoke_gates": [
        "all losses, gradients and parameters finite",
        "all 12 modes selected by the post-warm-up stochastic actor",
        "post-warm-up continuous q non-degenerate (distinct q, per-mode spread)",
        "camera SI, radar P40, MCS, backlog, SNR, prev mode, prev q, prev Q_perc, "
        "prev latency, prev presence and prev success all vary",
        "current SNR is never a sample generated after the action",
        "every registered boundary bundle verifies; exact cold actor load",
        "separate-process 250->500 resume bit-identical to uninterrupted run",
        "controlled-SNR and shuffled-SNR diagnostics reported (no direction gate)",
    ],
    "deep_validation": {
        "contexts": ("same independent contexts for Run-4 and Run-5: held-out scene "
                     "partition x 4 profiles x validation seeds 9017/9029/9043"),
        "metrics": ["reward", "success/deadline rate", "latency", "Q_perc",
                    "mode and q distribution", "oracle regret vs registered action catalogue",
                    "true causal SNR vs audit-only in-support shuffled-SNR actor input"],
        "coverage": "all three seeds x all registered evaluation checkpoints",
        "checkpoint_selection": ("the update-10,000 actor of each seed is the registered "
                                 "actor; no validation-based checkpoint selection; emergency "
                                 "checkpoints are never candidates"),
        "use": "diagnostics only; they never alter training or select hyper-parameters",
    },
    "campaign": {"seeds": [17, 29, 43], "target_update": DEEP_TARGET_UPDATE,
                 "recovery_checkpoint_max_gap_updates": 500,
                 "deep_training_authorized": False},
}

AMENDMENTS = (
    {
        "id": "A1_COLD_HOST_GPU_SYSTEM_DAEMON_ALLOWLIST",
        "made_before_any_smoke_data": True,
        "superseded_seal_sha256": "8bd54c22f1852428767fbf4f4433001341aea4dfd2ce547bc630e79bc4b7a32e",
        "reason": ("the first smoke launch refused at the cold-host gate because nvidia-smi "
                   "lists two persistent host services (gnome-remote-desktop-daemon, up 85 days; "
                   "nvidia-cuda-mps-server, up 35 days) as compute apps at 1 % GPU utilization; "
                   "no training step ran"),
        "change": ("exactly those two process names are allow-listed only while GPU "
                   "utilization <= 5 % and no other compute app exists; CARLA/OAI/RFsim/"
                   "Phase-6/container checks are unchanged"),
        "config_or_design_changed": False,
    },
)

RUN5_SOURCES = (
    "__init__.py", "run5_state_contract.py", "run5_snr_v2.py", "run5_models.py",
    "run5_channel.py", "run5_collector.py", "run5_training.py", "run5_bundle.py",
    "run5_campaign.py", "run5_preregistration.py", "successor_mcs_snr_audit.py",
)
REUSED_SOURCES = (
    "rl_agent/splitfusion_hybrid_sac_run4_v1/run4_contract.py",
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
    "rl_agent/splitfusion_action_catalog_v1/splitfusion_72_action_catalog.json",
)


class PreregistrationError(RuntimeError):
    pass


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def source_hashes(root: Path = ROOT) -> dict[str, str]:
    paths = [f"rl_agent/splitfusion_hybrid_sac_run5_v1/{name}" for name in RUN5_SOURCES]
    return {path: _sha_file(root / path) for path in (*paths, *REUSED_SOURCES, *CONFIG_ARTIFACTS)}


def document(root: Path = ROOT) -> dict[str, Any]:
    from . import run5_models as RM
    from . import run5_snr_v2 as SNR

    return {
        "schema": SCHEMA, "prospective": True,
        "written_before_any_run5_smoke": True,
        "config": {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(CONFIG).items()},
        "design": DESIGN,
        "amendments": list(AMENDMENTS),
        "feature_schema_sha256": SNR.FEATURE_SCHEMA_SHA256,
        "feature_schema": SNR.FEATURE_SCHEMA_DESCRIPTOR,
        "model_binding_sha256": RM.RUN5_TRAINING_MODEL_BINDING_SHA256,
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
        raise PreregistrationError("Run-5 training preregistration is not sealed")
    sealed = json.loads(path.read_text())
    current = document(root)
    for key in ("schema", "config", "design", "amendments", "feature_schema_sha256",
                "feature_schema",
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
