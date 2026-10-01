#!/usr/bin/env python3
"""Prospective Run-5B registration (sealed before any Run-5B preflight/smoke).

Binds the 21-D feature schema, the model binding, the shared operational-
latency provider digest, the shared joint SNR/MCS channel binding digest,
the Run-4B trainer/checkpoint/campaign settings (the derived runner's
constants) and the SHA-256 of every Run-5B, shared and pinned Run-4B source.
``load_sealed`` recomputes every cheap identity and refuses on drift; the
runner additionally requires the live joint-channel binding to equal the
sealed digest.

    python -m rl_agent.splitfusion_hybrid_sac_run5b_v1.run5b_registration --seal
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from rl_agent.splitfusion_hybrid_sac_run4b_v1 import contract as C4
from rl_agent.splitfusion_joint_channel_v1 import joint_channel as JC

from . import run5b_models as M
from . import run5b_state_contract as C

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
SEALED = PACKAGE / "RUN5B_REGISTRATION.json"
SCHEMA = "splitfusion.run5b.registration.v2"
RUN5B_SOURCES = ("__init__.py", "run5b_state_contract.py", "run5b_models.py",
                 "run5b_learner.py", "run5b_runner.py", "run5b_checks.py",
                 "run5b_registration.py", "derive_from_run4b.py")
PINNED_RUN4B_SOURCES = ("rl_agent/splitfusion_hybrid_sac_run4b_v1/learner.py",
                        "rl_agent/splitfusion_hybrid_sac_run4b_v1/runner.py",
                        "rl_agent/splitfusion_hybrid_sac_run4b_v1/models.py",
                        "rl_agent/splitfusion_hybrid_sac_run4b_v1/RUN4B_REGISTRATION.json",
                        "rl_agent/splitfusion_operational_latency_v1/PROVIDER_BINDING.json")


class RegistrationError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def static_document(root: Path = ROOT) -> dict[str, Any]:
    from . import run5b_runner as R

    cfg = R.TRAINER_CONFIG
    return {
        "schema": SCHEMA,
        "registered_before_any_run5b_training": True,
        "decision": JC.DECISION,
        "scope": ("OFFLINE_MODELED_EXPLORATORY_TRAINING; "
                  "EXPLORATORY_POOLED_FAMILY_TRANSFER_ASSUMPTION; NOT DEPLOYMENT AUTHORIZATION"),
        "feature_schema": C.FEATURE_SCHEMA,
        "feature_schema_sha256": C.FEATURE_SCHEMA_SHA256,
        "feature_order_sha256": C.FEATURE_ORDER_SHA256,
        "model_binding": M.MODEL_BINDING,
        "model_binding_sha256": M.MODEL_BINDING_SHA256,
        "operational_latency_provider_sha256": JC.REGISTERED_PROVIDER_BINDING_SHA256,
        "reward_schema_sha256": C4.REWARD_SCHEMA_SHA256,
        "observation_space_difference_vs_run4b_joint": (
            "Run-5B exposes the current causal SNR as feature 21; Run-4B-Joint withholds it. "
            "Channel, tapes, provider, reward, prior and SAC are identical."),
        "comparator": ("Run-4B-Joint (same splitfusion_joint_channel_v1 authority, 20-D). The "
                       "completed SNR-free Run-4B campaign is a pilot, not the paired ablation."),
        "trainer": {"alpha_d": cfg.alpha_d, "alpha_c": cfg.alpha_c, "tau": cfg.tau,
                    "actor_lr": cfg.actor_lr, "critic_lr": cfg.critic_lr, "batch": R.BATCH,
                    "gamma": C.GAMMA, "replay_capacity": R.L.REPLAY_CAPACITY,
                    "warmup": R.WARMUP, "transitions_per_update": R.TRANSITIONS_PER_UPDATE,
                    "threads": R.THREADS},
        "seeds": list(R.SEEDS),
        "checkpoint_updates": list(R.CHECKPOINT_UPDATES),
        "smoke": {"seed": 17, "stop_update": R.SMOKE_UPDATE,
                  "resume": "separate process genesis->250, separate process 250->500"},
        "preregistered_live_actor": dict(R.LIVE_ACTOR),
        "campaign_stop_rules": ["parity drift", "schema/provider mismatch",
                                "numerical failure", "checkpoint failure", "insufficient disk"],
        "family_dominance": ("Run-4B noAE+AE128 window recorded at every checkpoint and "
                             "reported; enforced in the smoke gate only, not a campaign stop"),
        "source_sha256": {**{f"rl_agent/splitfusion_hybrid_sac_run5b_v1/{n}":
                             _sha(PACKAGE / n) for n in RUN5B_SOURCES},
                          **{p: _sha(Path(root) / p) for p in PINNED_RUN4B_SOURCES},
                          **JC.shared_source_hashes(root)},
    }


def seal(evidence_root: Path) -> Path:
    if SEALED.exists():
        raise RegistrationError("the Run-5B registration is create-only")
    sources = JC.load_sources(evidence_root)
    document = {**static_document(),
                "joint_channel_binding": sources.binding_document(),
                "joint_channel_binding_sha256": sources.binding_sha256}
    with SEALED.open("x") as handle:
        handle.write(json.dumps(document, indent=1, sort_keys=True) + "\n")
    return SEALED


@functools.lru_cache(maxsize=1)
def load_sealed() -> dict[str, Any]:
    if not SEALED.is_file():
        raise RegistrationError("Run-5B registration is not sealed")
    sealed = json.loads(SEALED.read_text())
    current = json.loads(json.dumps(static_document()))
    for key, value in current.items():
        if key != "source_sha256" and sealed.get(key) != value:
            raise RegistrationError(f"registration {key} differs from the code")
    drift = sorted(k for k, v in current["source_sha256"].items()
                   if sealed["source_sha256"].get(k) != v)
    if drift or set(sealed["source_sha256"]) != set(current["source_sha256"]):
        raise RegistrationError(f"registered sources drifted: {drift}")
    return {"document": sealed, "sha256": _sha(SEALED)}


def sealed_sha256() -> str:
    return load_sealed()["sha256"]


def sealed_joint_channel_binding_sha256() -> str:
    return load_sealed()["document"]["joint_channel_binding_sha256"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", action="store_true")
    parser.add_argument("--evidence-root", type=Path, default=ROOT.parent / "abiodun")
    args = parser.parse_args(argv)
    if args.seal:
        print(seal(args.evidence_root.resolve()))
    document = load_sealed()
    print(json.dumps({"sha256": document["sha256"],
                      "joint_channel_binding_sha256":
                          document["document"]["joint_channel_binding_sha256"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
