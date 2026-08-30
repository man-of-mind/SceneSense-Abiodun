from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any, Mapping

import torch

from .base_runtime import load_base
from .continue_scientific import (_checkpoint, _recovery_binding, _require_runtime_source_binding,
                                  _select_verified_resume_checkpoint)
from .contracts import (atomic_json, atomic_text, canonical_hash, current_commit, load_json,
                        load_recovery_config, package_hashes, resolve_repo_path, sha256,
                        verify_original_provenance)
from .envelope import build_healthy_envelope
from .guards import PreStepBreaker
from .recovery_model import build_recovery_model
from .runner import run_guarded_epoch


PROTOCOL_PATH = Path(__file__).with_name("PROTOCOL_AMENDMENT_EPOCH10_GATE.json")
EXECUTION_TOKEN = "AUTHORIZED_RECOVERED_EPOCH10_GATE_ONLY"


def _load_protocol() -> Mapping[str, Any]:
    protocol = load_json(PROTOCOL_PATH)
    checks = (
        protocol.get("schema") == "splitfusion_fcos_prospective_epoch10_protocol_amendment_v1",
        protocol.get("state") == "PROSPECTIVE_EPOCH10_GATE_ONLY",
        protocol.get("prior_qualification", {}).get("qualification_pass_claimed") is False,
        protocol.get("prior_qualification", {}).get("replay_metric_agreement_gate")
            == "RETIRED_AS_INVALID_FOR_NONDETERMINISTIC_CUDA_TRAJECTORIES",
        float(protocol.get("yaw_repair", {}).get("tau", 0.0)) == 1e-2,
        protocol.get("breaker_envelope", {}).get("formula")
            == "10 * maximum value across both replay histories",
        protocol.get("scientific_gate", {}).get("only_epoch") == 10,
        protocol.get("scientific_gate", {}).get("epoch11_authorized") is False,
        protocol.get("scientific_gate", {}).get("validation_authorized") is False,
        protocol.get("scientific_gate", {}).get("write_training_complete") is False,
    )
    if not all(checks):
        raise RuntimeError("prospective epoch-10 protocol amendment drift")
    return protocol


def _verify_replay(path: Path, expected_sha256: str, protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    path = path.resolve(strict=True)
    if sha256(path) != expected_sha256:
        raise RuntimeError(f"replay hash drift: {path}")
    replay = load_json(path)
    healthy = replay.get("healthy_records", [])
    boundary = replay.get("boundary", {})
    update_numbers = [int(row.get("update_in_epoch", -1)) for row in healthy]
    carrier_rows = [carrier
                    for microbatch in boundary.get("physical_microbatches", [])
                    for carrier in microbatch.get("geometry", {}).get("carrier_identities", [])]
    tau = float(protocol["yaw_repair"]["tau"])
    affected = sum(float(row["raw_yaw_norm"]) < tau for row in carrier_rows)
    checks = (
        replay.get("schema") == "splitfusion_fcos_explicit_epoch10_replay_v1",
        replay.get("normalization") == "original",
        replay.get("tau") is None,
        int(replay.get("source_checkpoint_epoch", -1)) == 9,
        int(replay.get("source_global_update", -1)) == 9468,
        replay.get("validation_accessed") is False,
        replay.get("latest_checkpoint_discovery_used") is False,
        len(healthy) == int(protocol["replay_evidence"]["required_healthy_updates_per_replay"]),
        update_numbers == list(range(1, 447)),
        all(row.get("finite") is True and row.get("source") == "EPOCH10_EXPLICIT_REPLAY"
            and int(row.get("epoch", -1)) == 10 for row in healthy),
        int(boundary.get("epoch", -1)) == 10,
        int(boundary.get("update_in_epoch", -1)) == 447,
        int(boundary.get("global_update_if_stepped", -1)) == 9915,
        len(boundary.get("sample_ids", []))
            == int(protocol["replay_evidence"]["required_update447_sample_ids"]),
        boundary.get("optimizer_step_executed") is False,
        boundary.get("model_optimizer_unchanged_by_forward_backward") is True,
        len(carrier_rows) == int(protocol["replay_evidence"]["required_yaw_carriers_per_boundary"]),
        affected == 0,
    )
    if not all(checks):
        raise RuntimeError(f"replay contract verification failed: {path}")
    return replay


def verify_replay_union(replay_a_path: Path, replay_b_path: Path,
                        protocol: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any], dict[str, Any]]:
    expected = protocol["replay_evidence"]
    replay_a = _verify_replay(replay_a_path, str(expected["original_a_sha256"]), protocol)
    replay_b = _verify_replay(replay_b_path, str(expected["original_b_sha256"]), protocol)
    sample_ids_a = list(replay_a["boundary"]["sample_ids"])
    sample_ids_b = list(replay_b["boundary"]["sample_ids"])
    if sample_ids_a != sample_ids_b:
        raise RuntimeError("verified replay update-447 identities disagree")
    envelope = build_healthy_envelope([*replay_a["healthy_records"], *replay_b["healthy_records"]])
    envelope.update({
        "schema": "splitfusion_fcos_replay_union_healthy_numerical_envelope_v1",
        "protocol_amendment": "PROSPECTIVE_EPOCH10_GATE_ONLY",
        "replay_agreement_required": False,
        "replay_a_sha256": str(expected["original_a_sha256"]),
        "replay_b_sha256": str(expected["original_b_sha256"]),
        "union_observation_count": len(replay_a["healthy_records"]) + len(replay_b["healthy_records"]),
        "update447_sample_ids": sample_ids_a,
    })
    if envelope["union_observation_count"] != 892:
        raise RuntimeError("replay union must contain exactly 892 healthy observations")
    return replay_a, replay_b, envelope


def _verify_authorization(path: Path, protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    authorization = load_json(path.resolve(strict=True))
    source_sha256 = canonical_hash(package_hashes())
    expected = protocol["replay_evidence"]
    checks = (
        authorization.get("schema") == "splitfusion_fcos_epoch10_gate_authorization_v1",
        authorization.get("authorized") is True,
        authorization.get("scope") == "RECOVERED_EPOCH10_ONLY",
        authorization.get("source_commit") == current_commit(),
        authorization.get("source_files_sha256") == source_sha256,
        authorization.get("protocol_amendment_sha256") == sha256(PROTOCOL_PATH),
        authorization.get("replay_a_sha256") == expected["original_a_sha256"],
        authorization.get("replay_b_sha256") == expected["original_b_sha256"],
        float(authorization.get("yaw_tau", 0.0)) == 1e-2,
        authorization.get("validation_authorized") is False,
        authorization.get("epoch11_authorized") is False,
    )
    if not all(checks):
        raise RuntimeError("epoch-10 gate authorization does not bind the exact source and evidence")
    return authorization


def _prospective_config(protocol: Mapping[str, Any], envelope: Mapping[str, Any]) -> dict[str, Any]:
    original = load_recovery_config()["original"]
    return {
        "schema": "splitfusion_fcos_prospective_epoch10_gate_config_v1",
        "state": "PROSPECTIVE_EPOCH10_GATE_ONLY",
        "source_commit": current_commit(),
        "source_files_sha256": canonical_hash(package_hashes()),
        "selected_tau": 1e-2,
        "threshold_formula": "10 * maximum_healthy_value",
        "ceilings": envelope["ceilings"],
        "original_checkpoint_sha256": original["checkpoint_sha256"],
        "original_source_canonical_sha256": original["source_canonical_sha256"],
        "original_checkpoint_source_canonical_sha256": original["checkpoint_source_canonical_sha256"],
        "original_config_sha256": original["config_sha256"],
        "original_registration_sha256": original["registration_sha256"],
        "prior_qualification_passed": False,
        "protocol_amendment_sha256": sha256(PROTOCOL_PATH),
        "maximum_epoch_authorized": 10,
        "validation_authorized": False,
    }


def _evidence_hashes(replay_a: Path, replay_b: Path, authorization: Path) -> dict[str, str]:
    return {
        "PROTOCOL_AMENDMENT_EPOCH10_GATE.json": sha256(PROTOCOL_PATH),
        "REPLAY_A.json": sha256(replay_a),
        "REPLAY_B.json": sha256(replay_b),
        "EPOCH10_GATE_AUTHORIZATION.json": sha256(authorization),
    }


def _runtime_failure_record(output: Path, error: BaseException) -> dict[str, Any]:
    breaker_records = sorted((output / "numerical_failures").glob("BREAKER_*.json")) if output.is_dir() else []
    runtime_records = sorted((output / "numerical_failures").glob("RUNTIME_*.json")) if output.is_dir() else []
    detail_path = (breaker_records or runtime_records)[-1] if (breaker_records or runtime_records) else None
    detail = load_json(detail_path) if detail_path is not None else {}
    return {
        "schema": "splitfusion_fcos_recovered_epoch10_gate_failure_v1",
        "terminal": "RECOVERED_EPOCH10_GATE_FAILED",
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "detail_artifact": str(detail_path) if detail_path is not None else None,
        "sample_ids": list(detail.get("sample_ids", [])),
        "unsafe_step_prevented": detail.get("action") == "abort_before_optimizer_step",
        "validation_accessed": False,
        "epoch11_accessed": False,
    }


def _run(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    failure_artifact = args.failure_artifact.resolve()
    if output.exists():
        raise FileExistsError("recovered epoch-10 gate output is create-only")
    if failure_artifact.exists():
        raise FileExistsError("epoch-10 gate failure artifact is create-only")
    protocol = _load_protocol()
    replay_a_path = args.replay_a.resolve(strict=True)
    replay_b_path = args.replay_b.resolve(strict=True)
    authorization_path = args.authorization.resolve(strict=True)
    _authorization = _verify_authorization(authorization_path, protocol)
    replay_a, replay_b, envelope = verify_replay_union(replay_a_path, replay_b_path, protocol)
    provenance = verify_original_provenance(checkpoint_metadata=True)
    prospective = _prospective_config(protocol, envelope)
    evidence_hashes = _evidence_hashes(replay_a_path, replay_b_path, authorization_path)

    base = load_base()
    immutable = load_recovery_config()
    original_experiment = resolve_repo_path(immutable["original"]["experiment"])
    config = load_json(resolve_repo_path(immutable["original"]["config"]))
    runtime = load_json(original_experiment / "QUALIFIED_RUNTIME.json")
    calibration = load_json(original_experiment / "LOSS_CALIBRATION.json")
    priors = load_json(original_experiment / "TRAIN_ONLY_PRIORS.json")
    locked = immutable["locked_science"]
    if ((runtime["physical_batch"], runtime["gradient_accumulation"]) != (4, 4)
            or int(config["scientific_seed"]) != int(locked["scientific_seed"])):
        raise RuntimeError("locked epoch-10 batch/seed contract drift")
    if not torch.cuda.is_available():
        raise RuntimeError("recovered epoch-10 gate requires CUDA")
    device = torch.device("cuda:0")
    total_memory = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, 12288 * 2**20 / total_memory), device)
    model, build_report = build_recovery_model(priors, 1e-2, device)
    optimizer = base.train.build_optimizer(model, config)
    state = torch.load(Path(provenance["checkpoint"]), map_location="cpu", weights_only=False)
    if int(state.get("epoch", -1)) != 9 or int(state.get("global_optimizer_update", -1)) != 9468:
        raise RuntimeError("prospective gate accepts only the verified original epoch-9 checkpoint")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    base.common.restore_rng(state["rng"])
    if not base.train.all_model_finite(model) or not base.train.optimizer_finite(optimizer):
        raise FloatingPointError("epoch-9 recovery state is nonfinite")

    dataset_root = (base.common.ROOT / config["dataset_root"]).resolve(strict=True)
    rows = base.data.load_split_rows(dataset_root, "train")
    cache = base.data.DepthCache((base.common.ROOT / config["train_depth_cache"]).resolve(strict=True), rows)
    dataset = base.data.RouteBDataset(dataset_root, "train", int(config["scientific_seed"]), cache, augment=True)

    output.mkdir(parents=True, exist_ok=False)
    for name in ("checkpoints", "training_metrics", "gradient_telemetry", "numerical_failures"):
        (output / name).mkdir()
    atomic_json(output / "PROTOCOL_AMENDMENT.json", protocol)
    atomic_json(output / "UNION_BREAKER_ENVELOPE.json", envelope)
    atomic_json(output / "EPOCH10_GATE_CONFIG.json", prospective)
    atomic_json(output / "RECOVERY_PROVENANCE.json", {
        "schema": "splitfusion_fcos_prospective_epoch10_recovery_v1",
        "source_commit": current_commit(),
        "source_files": package_hashes(),
        "source_files_sha256": canonical_hash(package_hashes()),
        "original_epoch9": provenance,
        "original_epochs_10_26": "CORRUPTED_FINITE_GRADIENT_TRAJECTORY_DO_NOT_USE",
        "prior_qualification_passed": False,
        "protocol_amendment": protocol,
        "protocol_amendment_sha256": sha256(PROTOCOL_PATH),
        "replay_evidence": evidence_hashes,
        "prospective_config": prospective,
        "build_report": build_report,
        "start_epoch": 10,
        "maximum_epoch_authorized": 10,
        "validation_accessed": False,
    })
    atomic_json(output / "RECOVERED_EPOCH10_GATE_STARTED.json", {
        "schema": "splitfusion_fcos_recovered_epoch10_gate_started_v1",
        "source_commit": current_commit(),
        "source_checkpoint_epoch": 9,
        "source_global_optimizer_update": 9468,
        "fixed_tau": 1e-2,
        "union_observations": envelope["union_observation_count"],
        "validation_accessed": False,
        "epoch11_authorized": False,
    })
    atomic_json(output / "STATUS.json", {"state": "training", "epoch_complete": 9,
        "global_optimizer_update": 9468, "validation_accessed": False}, overwrite=True)

    _require_runtime_source_binding(prospective)
    breaker = PreStepBreaker(prospective["ceilings"], output / "numerical_failures")
    base.model.configure_trainability(model, 10)
    dataset.set_epoch(10)
    loader, sampler = base.train.dataloader(dataset, int(config["scientific_seed"]), 10, 4,
                                             int(config["training"]["workers"]))
    global_update, summary = run_guarded_epoch(base, model, optimizer, loader, config,
        calibration["multipliers"], 10, 9468, 4, breaker)
    reachability = summary["aggregate_required_gradient_reachability"]
    failures = list((output / "numerical_failures").glob("*.json"))
    gate_checks = (
        int(summary.get("epoch", -1)) == 10,
        int(summary.get("updates", -1)) == 1052,
        int(global_update) == 10520,
        summary.get("all_updates_finite") is True,
        summary.get("pre_step_breaker_checked_every_update") is True,
        reachability.get("all_required_trainable_groups_finite_every_update") is True,
        reachability.get("all_required_trainable_groups_observed_nonzero") is True,
        not failures,
        base.train.all_model_finite(model),
        base.train.optimizer_finite(optimizer),
        float(summary.get("peak_allocated_mib", float("inf"))) <= 12288.0,
    )
    if not all(gate_checks):
        raise RuntimeError("recovered epoch-10 scientific gate checks failed")

    atomic_json(output / "training_metrics/epoch_010.json", summary)
    atomic_json(output / "gradient_telemetry/epoch_010.json", {
        "schema": "splitfusion_fcos_epoch10_gate_aggregate_gradient_telemetry_v1",
        "epoch": 10,
        "aggregate_required_gradient_reachability": reachability,
        "source": "per-update required-gradient evidence from the scientific epoch",
        "additional_diagnostic_backward_passes": 0,
        "validation_accessed": False,
    })
    payload = _checkpoint(base, model, optimizer, epoch=10, global_update=global_update,
                          sampler=sampler, lrs=summary["last_lrs"], qualified=prospective,
                          qualification_hashes=evidence_hashes)
    checkpoint = output / "checkpoints/epoch_010.pt"
    sidecar = output / "checkpoints/epoch_010.json"
    if checkpoint.exists() or sidecar.exists():
        raise FileExistsError("refusing to overwrite recovered epoch-10 checkpoint")
    base.common.atomic_torch(checkpoint, payload)
    checkpoint_sha256 = sha256(checkpoint)
    atomic_json(sidecar, {"epoch": 10, "path": str(checkpoint), "sha256": checkpoint_sha256,
                          "global_optimizer_update": global_update})
    atomic_json(output / "STATUS.json", {"state": "training", "epoch_complete": 10,
        "global_optimizer_update": global_update, "validation_accessed": False}, overwrite=True)

    verified_epoch, verified_global, verified_state = _select_verified_resume_checkpoint(
        output, expected_recovery=_recovery_binding(prospective, evidence_hashes), config=config,
        train_frames=int(locked["train_frames"]), effective_batch=int(locked["effective_batch"]))
    if ((verified_epoch, verified_global) != (10, 10520)
            or int(verified_state.get("epoch", -1)) != 10):
        raise RuntimeError("saved epoch-10 state is not exactly resume-ready")
    resume_record = {
        "schema": "splitfusion_fcos_epoch10_exact_resume_readiness_v1",
        "verified_checkpoint": str(checkpoint),
        "verified_checkpoint_sha256": checkpoint_sha256,
        "completed_epoch": 10,
        "global_optimizer_update": 10520,
        "resume_epoch_when_later_authorized": 11,
        "next_global_optimizer_update_when_later_stepped": 10521,
        "model_optimizer_rng_schedule_counter_state_present": True,
        "source_and_protocol_binding_verified": True,
        "requires_new_user_instruction": True,
        "validation_accessed": False,
    }
    atomic_json(output / "EXACT_RESUME_READINESS.json", resume_record)
    completion = {
        "schema": "splitfusion_fcos_recovered_epoch10_gate_complete_v1",
        "terminal": "RECOVERED_EPOCH10_GATE_PASSED_AWAITING_REVIEW",
        "source_commit": current_commit(),
        "source_files_sha256": canonical_hash(package_hashes()),
        "replay_hashes_verified": {"original_a": sha256(replay_a_path), "original_b": sha256(replay_b_path)},
        "replay_metric_agreement_required": False,
        "fixed_tau": 1e-2,
        "affected_carriers": {"original_a": 0, "original_b": 0, "carriers_each": 831},
        "union_envelope_sha256": sha256(output / "UNION_BREAKER_ENVELOPE.json"),
        "union_envelope_ceilings": prospective["ceilings"],
        "optimizer_updates_completed": int(summary["updates"]),
        "global_optimizer_update": int(global_update),
        "all_updates_finite": True,
        "aggregate_required_gradient_reachability": reachability,
        "circuit_breaker_events": 0,
        "peak_allocated_mib": float(summary["peak_allocated_mib"]),
        "wall_seconds": float(summary["wall_seconds"]),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "exact_resume_readiness": resume_record,
        "validation_accessed": False,
        "epoch11_accessed": False,
        "training_complete_written": False,
    }
    atomic_json(output / "RECOVERED_EPOCH10_GATE_COMPLETE.json", completion)
    atomic_text(output / "RECOVERED_EPOCH10_GATE_COMPLETE", "RECOVERED_EPOCH10_GATE_COMPLETE\n")
    atomic_json(output / "STATUS.json", {"state": "awaiting_review", "epoch_complete": 10,
        "global_optimizer_update": global_update, "validation_accessed": False,
        "epoch11_accessed": False}, overwrite=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Prospective recovered scientific epoch-10 gate")
    parser.add_argument("--replay-a", required=True, type=Path)
    parser.add_argument("--replay-b", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--failure-artifact", required=True, type=Path)
    parser.add_argument("--execute-recovered-epoch10-gate", required=True, choices=(EXECUTION_TOKEN,))
    args = parser.parse_args()
    try:
        return _run(args)
    except BaseException as error:
        record = _runtime_failure_record(args.output.resolve(), error)
        if args.output.is_dir():
            output_failure = args.output.resolve() / "RECOVERED_EPOCH10_GATE_FAILED.json"
            if not output_failure.exists():
                atomic_json(output_failure, record)
            status_path = args.output.resolve() / "STATUS.json"
            if status_path.parent.is_dir():
                atomic_json(status_path, {"state": "failed", "terminal": "RECOVERED_EPOCH10_GATE_FAILED",
                    "error": str(error), "validation_accessed": False, "epoch11_accessed": False}, overwrite=True)
        if not args.failure_artifact.exists():
            atomic_json(args.failure_artifact.resolve(), record)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
