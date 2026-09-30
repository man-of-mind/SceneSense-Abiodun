#!/usr/bin/env python3
"""Phase 5 (plan only): sealed live binding manifest and preflight verifier.

``build_manifest`` re-hashes every identity a separately authorized Phase-6
300-frame qualification must bind, and seals the result.  ``verify_manifest``
rebuilds it from the working tree and fails on any drift.  Nothing here
launches CARLA, OAI, CUDA, a model or a network service; the checkpoint
hashes are file reads only.

Usage::

    python -m rl_agent.splitfusion_hybrid_sac_live_route_b_v2.readiness_v2 build
    python -m rl_agent.splitfusion_hybrid_sac_live_route_b_v2.readiness_v2 verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
PACKAGE_RELPATH = "rl_agent/splitfusion_hybrid_sac_live_route_b_v2"
MANIFEST_PATH = PACKAGE / "LIVE_BINDING_MANIFEST_V2.json"
CONFIG_PATH = PACKAGE / "live_qualification_300_v2.json"
SCHEMA = "scenesense.run4_live_v2.live_binding_manifest.v1"
TELEMETRY_EVIDENCE_RELPATH = (
    "rl_agent/experiments/splitfusion_hybrid_sac_live_route_b_v2/"
    "20260929_phase2c_ue_telemetry_qualification")
TELEMETRY_TERMINAL_SHA256 = (
    "5d30ad0345290bbc14905daf5f9a31d8711437c917b6cb29b32ee330b1196df6")
TELEMETRY_RESULT_SHA256 = (
    "5305904219abf8a1ce84335f726cb13b527cfa7994353ef96f9c77ed50752fbd")
PACKAGE_SOURCES = (
    "frozen_actor_v2.py", "export_frozen_actor_v2.py", "ACTOR_BINDING_V2.json",
    "ue_telemetry_provider_v2.py", "phase2_telemetry_qualification.py",
    "continuous_execution_v2.py", "reward_hold_controller_v2.py", "live_state_v2.py",
    "live_qualification_300_v2.json",
    # Phase 6 runner and its versioned adapters.
    "phase6_decision_engine_v2.py", "run4_live_wire_v2.py", "run4_map_protocol_v2.py",
    "run4_ue_ledger_v2.py", "phase6_edge_runtime_v2.py", "phase6_map_server_v2.py",
    "phase6_ue_runtime_v2.py", "phase6_live_child_v2.py", "phase6_live_runner_v2.py",
    # Prospective addendum 2 (option c): reporting only.
    "phase6_result_reporting_v2.py", "phase6_prospective_addendum_2.json",
    # Setup-repair addendum 3: no-build, image-bound edge launch (mechanics only).
    "phase6_edge_launch_v2.py", "phase6_live_child_nobuild_v2.py",
    "phase6_edge_startup_qualification_v2.py", "phase6_setup_repair_addendum_3.json",
    # Setup-repair addendum 4: metadata-only ready-record contract repair.
    "phase6_edge_runtime_v2.py", "test_phase6_ready_contract_v2.py",
    "phase6_setup_repair_addendum_4.json",
    # Addendum 5: the three live-path repairs (reward/state/thresholds unchanged);
    # the repaired runtime/controller/engine files are already pinned above.
    "phase6_engineering_gates_v2.py", "test_phase6_live_path_repair_v2.py",
    "phase6_live_path_repair_addendum_5.json",
    # Addendum 6: boundary restoration, cycle-aware stop, GT handoff diagnosis.
    "phase6_gt_handoff_v2.py", "test_phase6_repair_diagnostic_v2.py",
    "phase6_repair_diagnostic_addendum_6.json",
    # Addendum 7: deterministic pre-warm and reward-priority object GT.
    "phase6_prewarm_v2.py", "phase6_gt_priority_v2.py",
    "test_phase6_prewarm_gt_priority_v2.py", "phase6_prewarm_gt_priority_addendum_7.json",
    "phase6_handshake_manifest_v7.json",
)


class ReadinessError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReadinessError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("ascii")).hexdigest()


def build_manifest(repo_root: Path = ROOT) -> dict[str, Any]:
    import torch

    from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as contract
    from rl_agent.splitfusion_hybrid_sac_v1 import action_contract as ac
    from rl_agent.splitfusion_hybrid_sac_v1 import transaction_identity as ti
    from rl_agent.splitfusion_hybrid_sac_v1.modeled_smoke_support import (
        MODELED_SMOKE_SUPPORT_SHA256,
    )
    from rl_agent.splitfusion_live_dispatch_v1 import dynamic_execution_contract as dec
    from rl_agent.ue_mcs_backlog_near_capacity_v1 import radio_binding as RB

    from . import frozen_actor_v2 as FA
    from . import live_state_v2 as LS
    from . import phase2_telemetry_qualification as Q
    from . import reward_hold_controller_v2 as R
    from . import ue_telemetry_provider_v2 as T

    # Actor: pinned sources, tracked binding == export manifest, weights hash.
    sources = FA.verify_pinned_sources(repo_root)
    actor_manifest = FA.load_json(FA.TRACKED_BINDING_PATH)
    FA.verify_manifest(actor_manifest)
    export_dir = repo_root / FA.ACTOR_EXPORT_RELPATH
    require((export_dir / "ACTOR_EXPORT_MANIFEST.json").read_bytes()
            == FA.TRACKED_BINDING_PATH.read_bytes(), "actor binding drift")
    weights = export_dir / actor_manifest["actor"]["weights_file_name"]
    require(sha256_file(weights) == actor_manifest["actor"]["weights_file_sha256"],
            "actor weights drift")

    # Telemetry qualification evidence.
    evidence = repo_root / TELEMETRY_EVIDENCE_RELPATH
    require(sha256_file(evidence / "TERMINAL.json") == TELEMETRY_TERMINAL_SHA256,
            "telemetry TERMINAL drift")
    require(sha256_file(evidence / "QUALIFICATION_RESULT.json")
            == TELEMETRY_RESULT_SHA256, "telemetry result drift")
    result = json.loads((evidence / "QUALIFICATION_RESULT.json").read_text())
    require(result["status"] == "PASSED" and result["gates_sha256"] == Q.gates_sha256(),
            "telemetry qualification is not a PASSED run of the registered gates")

    # Execution bundle and model checkpoints (hash-verified by the loader).
    dynamic = dec.load_dynamic_execution_contract()
    checkpoints = {}
    for name, value in json.loads(
            (repo_root / "rl_agent/splitfusion_live_dispatch_v1/runtime_binding.json")
            .read_text())["selected_checkpoints"].items():
        path = repo_root / value["path"]
        require(path.is_file(), f"checkpoint missing: {value['path']}")
        require(sha256_file(path) == value["sha256"], f"checkpoint drift: {name}")
        checkpoints[name] = dict(value)

    radio = RB.verify("before_preflight", repo_root)
    body = {
        "schema": SCHEMA,
        "scope": ("PLAN_ONLY__PHASE6_REQUIRES_SEPARATE_AUTHORIZATION__NO_LIVE_RUN_"
                  "PERFORMED_BY_THIS_MANIFEST"),
        "actor": {
            "seed": FA.SELECTED.seed, "update": FA.SELECTED.update,
            "boundary_sha256": actor_manifest["actor"]["boundary_sha256"],
            "weights_file_sha256": actor_manifest["actor"]["weights_file_sha256"],
            "export_manifest_sha256": actor_manifest["manifest_sha256"],
            "tracked_binding_file_sha256": sha256_file(FA.TRACKED_BINDING_PATH),
            "pinned_sources": sources,
            "deployment_rule": FA.DEPLOYMENT_RULE,
        },
        "contracts": {
            "run4_contract_schema_sha256": contract.SCHEMA_SHA256,
            "feature_schema_sha256": contract.FEATURE_SCHEMA_SHA256,
            "reward_schema_sha256": contract.REWARD_SCHEMA_SHA256,
            "transition_schema_sha256": contract.TRANSITION_SCHEMA_SHA256,
            "policy_feature_order": list(contract.POLICY_FEATURE_ORDER),
            "action_identity_schema_sha256": ti.ACTION_IDENTITY_SCHEMA_SHA256,
            "action_catalog_sha256": ac.CATALOG_SHA256,
            "q_support_sha256": MODELED_SMOKE_SUPPORT_SHA256,
            "training_freshness_sha256": T.TRAINING_FRESHNESS.canonical_sha256(),
            "training_scaling_sha256": LS.TRAINING_SCALING.canonical_sha256(),
            "radio_support": {k: list(v) for k, v in LS.RADIO_SUPPORT.items()},
            "reward_deadline_ns": R.REWARD_DEADLINE_NS,
            "timeout_resolution_elapsed_ns": R.TIMEOUT_RESOLUTION_ELAPSED_NS,
            "k_min": R.K_MIN,
        },
        "telemetry": {
            "provider_sha256": sha256_file(PACKAGE / "ue_telemetry_provider_v2.py"),
            "qualification_runner_sha256": sha256_file(
                PACKAGE / "phase2_telemetry_qualification.py"),
            "gates_sha256": Q.gates_sha256(),
            "evidence_relpath": TELEMETRY_EVIDENCE_RELPATH,
            "terminal_sha256": TELEMETRY_TERMINAL_SHA256,
            "result_sha256": TELEMETRY_RESULT_SHA256,
            "status": result["status"],
        },
        "execution": {
            "runtime_binding_sha256": dynamic.runtime_binding_sha256,
            "behavioral_source_binding_sha256": dynamic.behavioral_source_binding_sha256,
            "action_contract_source_sha256": dec.ACTION_CONTRACT_SOURCE_SHA256,
            "selected_checkpoints": checkpoints,
            "startup_artifact_count": len(dynamic.startup_artifacts),
        },
        "radio": {
            "radio_binding_problems": radio.get("problems", []),
            "radio_binding_files": {k: v.get("observed_sha256")
                                    for k, v in radio["files"].items()},
            "emitter_pins": dict(Q.EMITTER_PINS),
            "emitter_pin_ownership": Q.EMITTER_PIN_OWNERSHIP,
            "production_config_relpath": Q.PRODUCTION_CONFIG_RELPATH,
        },
        "package_sources": {name: sha256_file(PACKAGE / name)
                            for name in PACKAGE_SOURCES},
        "torch_cuda_initialized_while_building": torch.cuda.is_initialized(),
    }
    require(not body["torch_cuda_initialized_while_building"], "CUDA was initialized")
    return {**body, "manifest_sha256": canonical_sha256(body)}


def verify_manifest(repo_root: Path = ROOT, path: Path = MANIFEST_PATH) -> dict[str, Any]:
    recorded = json.loads(Path(path).read_text(encoding="utf-8"))
    body = {k: v for k, v in recorded.items() if k != "manifest_sha256"}
    require(canonical_sha256(body) == recorded["manifest_sha256"],
            "manifest self-digest differs (tampered)")
    rebuilt = build_manifest(repo_root)
    require(rebuilt == recorded, "live binding drifted from the sealed manifest")
    return {"verified": True, "manifest_sha256": recorded["manifest_sha256"]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"))
    args = parser.parse_args(argv)
    if args.command == "build":
        manifest = build_manifest()
        with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(manifest["manifest_sha256"])
        return 0
    print(json.dumps(verify_manifest(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
